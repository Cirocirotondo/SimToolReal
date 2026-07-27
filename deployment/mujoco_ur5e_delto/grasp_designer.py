from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Lock
from typing import Literal, Optional

import numpy as np
import tyro
from scipy.spatial.transform import Rotation

# Python 3.8 compatibility for viser static-file serving.
if not hasattr(Path, "is_relative_to"):

    def _is_relative_to(self: Path, *other: Path) -> bool:
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    Path.is_relative_to = _is_relative_to  # type: ignore[attr-defined]

import viser

from deployment.mujoco_ur5e_delto.mujoco_sim import (
    Ur5eDeltoMujocoConfig,
    Ur5eDeltoMujocoSim,
)
from deployment.mujoco_ur5e_delto.policy_adapter import (
    ARM_DOF,
    DEFAULT_JOINT_POS,
    N_ACT,
    HandSide,
    joint_names_for_hand,
    validate_hand_side,
)


@dataclass
class Args:
    hand_side: Literal["right", "left"] = "right"
    """Delto hand side to load."""

    object_name: Literal["cube", "dumbbell_20x9x9cm"] = "cube"
    """Object geometry to edit."""

    object_size: str = "0.05,0.05,0.05"
    """Cube dimensions in meters; ignored for the fixed-size dumbbell."""

    port: int = 8081
    """Viser GUI port."""

    output_path: Path = Path("deployment/mujoco_ur5e_delto/grasp_candidate.json")
    """Directory and default filename for saved JSON candidates."""

    grasp_name: Optional[str] = None
    """Initial grasp name. Defaults to the output path stem."""

    load_path: Optional[Path] = None
    """Optional grasp JSON to load at startup and through the Load button."""

    enable_viewer: bool = True
    """Open the MuJoCo passive viewer."""

    table_center_z: float = -0.125
    """Table body center z. -0.125 gives a table top near z=0.025."""

    workspace_y: float = -0.6
    """World y offset of table/object area."""

    sim_dt: float = 1.0 / 600.0
    """MuJoCo simulation timestep."""

    control_hz: float = 60.0
    """Designer update frequency."""

    arm_kp: float = 10000.0
    """Arm PD proportional gain. The standard MuJoCo backend uses 300."""

    arm_kv: float = 170.0
    """Arm PD velocity damping. The standard MuJoCo backend uses 20."""


def _parse_vec3(text: str) -> np.ndarray:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected three comma-separated values, got {text!r}")
    return np.array(values, dtype=np.float32)


def _quat_wxyz_from_euler_deg(euler_deg: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].astype(np.float32)


def _euler_deg_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_euler(
        "xyz", degrees=True
    )


def _filename_from_grasp_name(grasp_name: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", grasp_name.strip()).strip("._")
    return filename or "grasp_candidate"


class GraspDesigner:
    def __init__(self, args: Args):
        self.args = args
        self.hand_side: HandSide = validate_hand_side(args.hand_side)
        self.object_name = args.object_name
        if self.object_name == "dumbbell_20x9x9cm":
            self.object_size = np.array([0.09, 0.20, 0.09], dtype=np.float32)
        else:
            self.object_size = _parse_vec3(args.object_size)
        object_scales = self.object_size / 0.04

        self.sim = Ur5eDeltoMujocoSim(
            Ur5eDeltoMujocoConfig(
                enable_viewer=args.enable_viewer,
                sim_dt=args.sim_dt,
                arm_kp=args.arm_kp,
                arm_kv=args.arm_kv,
                hand_side=self.hand_side,
                table_center_z=args.table_center_z,
                workspace_y=args.workspace_y,
                object_name=self.object_name,
                object_scales=object_scales.astype(np.float32),
                show_goal_marker=False,
                initial_joint_pos=DEFAULT_JOINT_POS.copy(),
            )
        )
        self.joint_names = joint_names_for_hand(self.hand_side)
        self.q_targets = self.sim.get_sim_state()["joint_positions"].astype(np.float32)
        self.object_pos = self.sim.get_sim_state()["object_pos"].astype(np.float32)
        self.object_euler_deg = _euler_deg_from_quat_wxyz(
            self.sim.get_sim_state()["object_quat_wxyz"]
        ).astype(np.float32)
        self.physics_running = False
        self.gravity_enabled = False
        self.grasp_name = args.grasp_name or args.output_path.stem
        self.running = True
        self._state_lock = Lock()
        self._commands = SimpleQueue()

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self.status = None
        self.joint_sliders = []
        self.object_pos_sliders = {}
        self.object_rot_sliders = {}
        self.physics_checkbox = None
        self.gravity_checkbox = None
        self.impulse_linear = None
        self.impulse_angular = None
        self.grasp_name_input = None
        self._build_gui()
        if args.load_path is not None and args.load_path.exists():
            self._load_grasp(args.load_path)
        self._apply_edit_state()

    def close(self) -> None:
        self.running = False
        self.sim.close()
        self.server.stop()

    def _build_gui(self) -> None:
        self.server.gui.add_markdown(
            "# MuJoCo Grasp Designer\n"
            "Use this panel to edit hand joints and object pose. Keep physics off "
            "while designing; turn it on to test gravity/contact stability."
        )

        with self.server.gui.add_folder("Mode"):
            self.physics_checkbox = self.server.gui.add_checkbox(
                "Physics running", initial_value=self.physics_running
            )
            self.gravity_checkbox = self.server.gui.add_checkbox(
                "Gravity enabled", initial_value=self.gravity_enabled
            )
            self.physics_checkbox.on_update(lambda _: self._update_mode())
            self.gravity_checkbox.on_update(lambda _: self._update_mode())

            reset_button = self.server.gui.add_button("Reset to Edit Pose")
            reset_button.on_click(lambda _: self._commands.put(("reset", None)))

            capture_button = self.server.gui.add_button("Capture Current Object Pose")
            capture_button.on_click(
                lambda _: self._commands.put(("capture_object_pose", None))
            )

        with self.server.gui.add_folder("Arm Joints"):
            for idx, name in enumerate(self.joint_names[:ARM_DOF]):
                slider = self.server.gui.add_slider(
                    name,
                    min=float(self.sim.lower_limits[idx]),
                    max=float(self.sim.upper_limits[idx]),
                    step=0.001,
                    initial_value=float(self.q_targets[idx]),
                )
                slider.on_update(lambda event, i=idx: self._set_joint(i, event.target.value))
                self.joint_sliders.append(slider)

        with self.server.gui.add_folder("Hand Joints"):
            for idx, name in enumerate(self.joint_names[ARM_DOF:], start=ARM_DOF):
                slider = self.server.gui.add_slider(
                    name,
                    min=float(self.sim.lower_limits[idx]),
                    max=float(self.sim.upper_limits[idx]),
                    step=0.001,
                    initial_value=float(self.q_targets[idx]),
                )
                slider.on_update(lambda event, i=idx: self._set_joint(i, event.target.value))
                self.joint_sliders.append(slider)

        with self.server.gui.add_folder("Object Pose"):
            for axis, idx in zip("xyz", range(3)):
                slider = self.server.gui.add_slider(
                    f"object {axis} (m)",
                    min=-1.0 if axis != "z" else 0.0,
                    max=1.0,
                    step=0.001,
                    initial_value=float(self.object_pos[idx]),
                )
                slider.on_update(
                    lambda event, i=idx: self._set_object_pos(i, event.target.value)
                )
                self.object_pos_sliders[axis] = slider

            for axis, idx in zip(("roll", "pitch", "yaw"), range(3)):
                slider = self.server.gui.add_slider(
                    f"object {axis} (deg)",
                    min=-180.0,
                    max=180.0,
                    step=1.0,
                    initial_value=float(self.object_euler_deg[idx]),
                )
                slider.on_update(
                    lambda event, i=idx: self._set_object_rot(i, event.target.value)
                )
                self.object_rot_sliders[axis] = slider

        with self.server.gui.add_folder("Disturbance"):
            self.impulse_linear = self.server.gui.add_vector3(
                "linear velocity kick (m/s)",
                initial_value=(0.0, 0.0, 0.0),
                step=0.01,
            )
            self.impulse_angular = self.server.gui.add_vector3(
                "angular velocity kick (rad/s)",
                initial_value=(0.0, 0.0, 0.0),
                step=0.1,
            )
            kick_button = self.server.gui.add_button("Apply Kick")
            kick_button.on_click(lambda _: self._queue_kick())

        with self.server.gui.add_folder("Save / Load"):
            self.grasp_name_input = self.server.gui.add_text(
                "Grasp name", initial_value=self.grasp_name
            )
            self.grasp_name_input.on_update(
                lambda event: self._set_grasp_name(event.target.value)
            )
            self.server.gui.add_markdown(
                f"Save directory: `{self.args.output_path.parent}`"
            )
            save_button = self.server.gui.add_button("Save Grasp JSON")
            save_button.on_click(lambda _: self._commands.put(("save", None)))
            load_button = self.server.gui.add_button("Load Grasp JSON")
            load_button.on_click(lambda _: self._commands.put(("load", None)))

        self.status = self.server.gui.add_markdown("Starting...")

    def _update_mode(self) -> None:
        with self._state_lock:
            self.physics_running = bool(self.physics_checkbox.value)
            self.gravity_enabled = bool(self.gravity_checkbox.value)

    def _set_joint(self, index: int, value: float) -> None:
        with self._state_lock:
            self.q_targets[index] = float(value)

    def _set_object_pos(self, index: int, value: float) -> None:
        with self._state_lock:
            self.object_pos[index] = float(value)

    def _set_object_rot(self, index: int, value: float) -> None:
        with self._state_lock:
            self.object_euler_deg[index] = float(value)

    def _set_grasp_name(self, value: str) -> None:
        with self._state_lock:
            self.grasp_name = str(value)

    def _snapshot_controls(self):
        with self._state_lock:
            return (
                self.q_targets.copy(),
                self.object_pos.copy(),
                self.object_euler_deg.copy(),
                self.physics_running,
                self.gravity_enabled,
            )

    def _grasp_name_and_output_path(self):
        with self._state_lock:
            grasp_name = self.grasp_name.strip() or "grasp_candidate"
        filename = _filename_from_grasp_name(grasp_name) + ".json"
        return grasp_name, self.args.output_path.parent / filename

    def _apply_object_pose(
        self, object_pos: np.ndarray, object_euler_deg: np.ndarray
    ) -> None:
        quat_wxyz = _quat_wxyz_from_euler_deg(object_euler_deg)
        self.sim.set_object_pose(object_pos.astype(np.float32), quat_wxyz)

    def _apply_edit_state(self) -> None:
        q_targets, object_pos, object_euler_deg, _, _ = self._snapshot_controls()
        self.sim.set_gravity_enabled(False)
        self.sim.set_robot_joint_pos_targets(q_targets)
        self.sim.set_robot_joint_positions(q_targets)
        self._apply_object_pose(object_pos, object_euler_deg)

    def _capture_current_object_pose(self) -> None:
        state = self.sim.get_sim_state()
        object_pos = state["object_pos"].astype(np.float32)
        object_euler_deg = _euler_deg_from_quat_wxyz(
            state["object_quat_wxyz"]
        ).astype(np.float32)
        with self._state_lock:
            self.object_pos = object_pos
            self.object_euler_deg = object_euler_deg
        for axis, idx in zip("xyz", range(3)):
            self.object_pos_sliders[axis].value = float(object_pos[idx])
        for axis, idx in zip(("roll", "pitch", "yaw"), range(3)):
            self.object_rot_sliders[axis].value = float(object_euler_deg[idx])

    def _queue_kick(self) -> None:
        linear = np.array(self.impulse_linear.value, dtype=np.float64)
        angular = np.array(self.impulse_angular.value, dtype=np.float64)
        self._commands.put(("kick", (linear, angular)))

    def _apply_kick(self, linear: np.ndarray, angular: np.ndarray) -> None:
        self.sim.add_object_velocity(linear, angular)

    def _process_commands(self) -> None:
        while True:
            try:
                command, payload = self._commands.get_nowait()
            except Empty:
                return

            if command == "reset":
                self._apply_edit_state()
            elif command == "capture_object_pose":
                self._capture_current_object_pose()
            elif command == "kick":
                self._apply_kick(*payload)
            elif command == "save":
                self._save_grasp()
            elif command == "load":
                self._load_grasp_from_default_path()
            else:
                raise RuntimeError(f"Unknown GUI command: {command}")

    def _save_grasp(self) -> None:
        state = self.sim.get_sim_state()
        q_targets, _, _, _, _ = self._snapshot_controls()
        q_actual = state["joint_positions"].astype(np.float32)
        grasp_name, output_path = self._grasp_name_and_output_path()
        payload = {
            "format_version": 2,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "grasp_name": grasp_name,
            "hand_side": self.hand_side,
            "object_name": self.object_name,
            "object_size_m": self.object_size.tolist(),
            "object_asset_path": (
                "assets/urdf/dumbbell_20x9x9cm/Dumbbell_20x9x9cm.stl"
                if self.object_name == "dumbbell_20x9x9cm"
                else None
            ),
            "joint_names": self.joint_names,
            "joint_pos": q_actual.astype(float).tolist(),
            "joint_targets": q_targets.astype(float).tolist(),
            "arm_dof_pos": q_actual[:ARM_DOF].astype(float).tolist(),
            "hand_dof_pos": q_actual[ARM_DOF:].astype(float).tolist(),
            "object_pose": {
                "pos": state["object_pos"].astype(float).tolist(),
                "quat_wxyz": state["object_quat_wxyz"].astype(float).tolist(),
                "euler_xyz_deg": _euler_deg_from_quat_wxyz(
                    state["object_quat_wxyz"]
                ).astype(float).tolist(),
            },
            "notes": "Saved from deployment/mujoco_ur5e_delto/grasp_designer.py",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._set_status(f"Saved `{output_path}`")

    def _load_grasp_from_default_path(self) -> None:
        _, named_output_path = self._grasp_name_and_output_path()
        path = self.args.load_path or named_output_path
        if not path.exists():
            self._set_status(f"Cannot load: `{path}` does not exist")
            return
        self._load_grasp(path)

    def _load_grasp(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        q = np.array(payload["joint_pos"], dtype=np.float32)
        if q.shape != (N_ACT,):
            raise ValueError(f"joint_pos in {path} has shape {q.shape}, expected {(N_ACT,)}")
        object_pose = payload["object_pose"]
        object_pos = np.array(object_pose["pos"], dtype=np.float32)
        object_euler_deg = np.array(
            object_pose.get(
                "euler_xyz_deg",
                _euler_deg_from_quat_wxyz(
                    np.array(object_pose["quat_wxyz"], dtype=np.float32)
                ),
            ),
            dtype=np.float32,
        )
        grasp_name = str(payload.get("grasp_name", path.stem))
        with self._state_lock:
            self.q_targets = q
            self.object_pos = object_pos
            self.object_euler_deg = object_euler_deg
            self.grasp_name = grasp_name
        for idx, slider in enumerate(self.joint_sliders):
            slider.value = float(q[idx])
        for axis, idx in zip("xyz", range(3)):
            self.object_pos_sliders[axis].value = float(object_pos[idx])
        for axis, idx in zip(("roll", "pitch", "yaw"), range(3)):
            self.object_rot_sliders[axis].value = float(object_euler_deg[idx])
        self.grasp_name_input.value = grasp_name
        self._apply_edit_state()
        if int(payload.get("format_version", 1)) < 2:
            self._set_status(
                f"Loaded legacy grasp `{path}`. Its joint_pos may contain PD targets "
                "instead of the gravity-deflected physical pose. Save it again to "
                "upgrade the format."
            )
        else:
            self._set_status(f"Loaded `{path}`")

    def _set_status(self, message: str) -> None:
        if self.status is not None:
            self.status.content = message

    def _update_status(self) -> None:
        state = self.sim.get_sim_state()
        _, _, _, physics_running, gravity_enabled = self._snapshot_controls()
        fingertip_dist = np.linalg.norm(
            state["fingertip_positions"] - state["object_pos"][None, :],
            axis=-1,
        )
        mode = "TEST" if physics_running else "EDIT"
        gravity = "on" if gravity_enabled and physics_running else "off"
        self._set_status(
            f"Mode: **{mode}** | gravity: **{gravity}** | "
            f"contacts: `{self.sim.data.ncon}` | "
            f"min fingertip-object distance: `{float(fingertip_dist.min()):.3f} m`"
        )

    def run(self) -> None:
        print(f"Open the control panel at http://localhost:{self.args.port}")
        control_dt = 1.0 / self.args.control_hz
        physics_steps = max(1, int(round(control_dt / self.args.sim_dt)))
        last_status = 0.0
        try:
            while self.running:
                start = time.time()
                if self.args.enable_viewer and not self.sim.viewer.is_running():
                    break
                self._process_commands()
                q_targets, _, _, physics_running, gravity_enabled = (
                    self._snapshot_controls()
                )
                self.sim.set_robot_joint_pos_targets(q_targets)
                if physics_running:
                    self.sim.set_gravity_enabled(gravity_enabled)
                    for _ in range(physics_steps):
                        self.sim.sim_step()
                    if self.args.enable_viewer:
                        self.sim.viewer.sync()
                else:
                    self._apply_edit_state()

                now = time.time()
                if now - last_status > 0.25:
                    self._update_status()
                    last_status = now
                sleep_dt = control_dt - (time.time() - start)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
        finally:
            self.close()


def main() -> None:
    args = tyro.cli(Args)
    designer = GraspDesigner(args)
    designer.run()


if __name__ == "__main__":
    main()
