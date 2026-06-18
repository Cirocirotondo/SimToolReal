from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from deployment.mujoco_ur5e_delto.policy_adapter import (
    DEFAULT_JOINT_POS,
    FINGERTIP_BODY_NAMES,
    FINGERTIP_LOCAL_OFFSETS,
    JOINT_NAMES,
    LOWER_LIMITS,
    N_ACT,
    UPPER_LIMITS,
)
from isaacgymenvs.utils.utils import get_repo_root_dir


@dataclass
class Ur5eDeltoMujocoConfig:
    enable_viewer: bool = True
    sim_dt: float = 1.0 / 600.0
    robot_urdf_path: Path = Path(
        "assets/urdf/ur5e_delto_description/ur5e_left_dg5f.urdf"
    )
    workspace_y: float = -0.6
    """Table/object y position in front of the robot base."""
    table_center_z: float = 0.38
    """Z position of the table body center."""
    table_object_z_offset: float = 0.25
    """Object center height above the table body center."""
    goal_object_z_offset: float = 0.35
    """Goal object center height above the table body center."""
    show_goal_marker: bool = True
    """Display the visual-only goal body and frame in the MuJoCo viewer."""
    show_object_frame: bool = True
    """Display local axes on the movable object in the MuJoCo viewer."""
    initial_joint_pos: np.ndarray = field(default_factory=lambda: DEFAULT_JOINT_POS.copy())
    object_name: str = "cube"
    object_scales: np.ndarray = field(
        default_factory=lambda: np.array([1.25, 1.25, 1.25], dtype=np.float32)
    )
    object_start_pos: Optional[np.ndarray] = None
    object_start_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    goal_object_start_pos: Optional[np.ndarray] = None
    goal_object_start_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )


class Ur5eDeltoMujocoSim:
    def __init__(self, config: Ur5eDeltoMujocoConfig):
        self.config = config
        self.repo_root = get_repo_root_dir()
        self.robot_joint_pos_targets = self.config.initial_joint_pos.copy()
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="simtoolreal_mujoco_")
        self._init_scene()
        self.set_robot_joint_positions(self.config.initial_joint_pos)
        self.set_robot_joint_pos_targets(self.config.initial_joint_pos)
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        if hasattr(self, "viewer"):
            self.viewer.close()
        self._tmp_dir.cleanup()

    def _make_mujoco_compatible_urdf(self) -> Path:
        urdf_path = self.config.robot_urdf_path
        if not urdf_path.is_absolute():
            urdf_path = self.repo_root / urdf_path
        text = urdf_path.read_text(encoding="utf-8")
        if "<mujoco>" not in text:
            text = text.replace(
                '<robot name="ur5e_left_dg5f">',
                '<robot name="ur5e_left_dg5f">\n'
                '  <mujoco><compiler strippath="false"/></mujoco>',
                1,
            )

        def replace_mesh(match: re.Match[str]) -> str:
            filename = match.group(1)
            if filename.startswith("urdf/"):
                mesh_path = self.repo_root / "assets" / filename
            else:
                mesh_path = urdf_path.parent / filename
            return f'filename="{mesh_path}"'

        text = re.sub(r'filename="([^"]+)"', replace_mesh, text)
        tmp_path = Path(self._tmp_dir.name) / "ur5e_left_dg5f_mujoco.urdf"
        tmp_path.write_text(text, encoding="utf-8")
        return tmp_path

    def _init_scene(self) -> None:
        spec = mujoco.MjSpec()
        spec.from_file(str(self._make_mujoco_compatible_urdf()))
        spec.discardvisual = False

        self._add_world(spec)
        self._add_position_actuators(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config.sim_dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.iterations = 20
        self.model.opt.ls_iterations = 50

        self._joint_qpos_adrs = np.array(
            [self.model.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=np.int32
        )
        self._joint_dof_adrs = np.array(
            [self.model.joint(name).dofadr[0] for name in JOINT_NAMES], dtype=np.int32
        )
        self._actuator_ids = np.array(
            [self.model.actuator(f"{name}_pos").id for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._validate()
        if self.config.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _add_world(self, spec: mujoco.MjSpec) -> None:
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = np.array([1.5, 1.5, 0.05])
        floor.rgba = np.array([0.2, 0.25, 0.28, 1.0])

        table = spec.worldbody.add_body()
        table.name = "table"
        table.pos = np.array([0.0, self.config.workspace_y, self.config.table_center_z])
        table_geom = table.add_geom()
        table_geom.name = "table_geom"
        table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = np.array([0.475 / 2.0, 0.4 / 2.0, 0.3 / 2.0])
        table_geom.rgba = np.array([0.9, 0.9, 0.9, 1.0])
        table_geom.friction = np.array([1.0, 0.005, 0.0001])

        self._add_object(
            spec=spec,
            name="object",
            pos=self._workspace_pos(self.object_start_pos),
            quat_wxyz=self.config.object_start_quat_wxyz,
            rgba=np.array([0.45, 0.45, 0.45, 1.0]),
            movable=True,
            disable_contact=False,
            add_frame=self.config.show_object_frame,
            frame_name="object",
        )
        goal_rgba = (
            np.array([0.1, 0.9, 0.2, 0.45])
            if self.config.show_goal_marker
            else np.array([0.1, 0.9, 0.2, 0.0])
        )
        self._add_object(
            spec=spec,
            name="goal_object",
            pos=self._workspace_pos(self.goal_object_start_pos),
            quat_wxyz=self.config.goal_object_start_quat_wxyz,
            rgba=goal_rgba,
            movable=False,
            disable_contact=True,
            add_frame=self.config.show_goal_marker,
            frame_name="goal",
        )

        light = spec.worldbody.add_light()
        light.name = "key_light"
        light.pos = np.array([0.0, -1.0, 1.5])
        light.dir = np.array([0.0, 0.5, -1.0])
        light.directional = True

    def _workspace_pos(self, pos: np.ndarray) -> np.ndarray:
        ret = pos.copy()
        ret[1] += self.config.workspace_y
        return ret

    @property
    def object_start_pos(self) -> np.ndarray:
        if self.config.object_start_pos is not None:
            return self.config.object_start_pos
        return np.array(
            [0.0, 0.0, self.config.table_center_z + self.config.table_object_z_offset],
            dtype=np.float32,
        )

    @property
    def goal_object_start_pos(self) -> np.ndarray:
        if self.config.goal_object_start_pos is not None:
            return self.config.goal_object_start_pos
        return np.array(
            [0.12, 0.0, self.config.table_center_z + self.config.goal_object_z_offset],
            dtype=np.float32,
        )

    def _add_object(
        self,
        *,
        spec: mujoco.MjSpec,
        name: str,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        rgba: np.ndarray,
        movable: bool,
        disable_contact: bool,
        add_frame: bool = False,
        frame_name: str = "frame",
    ) -> None:
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = pos
        body.quat = quat_wxyz
        if movable:
            joint = body.add_joint()
            joint.name = f"{name}_free_joint"
            joint.type = mujoco.mjtJoint.mjJNT_FREE

        if self.config.object_name == "hammer":
            geoms = self._add_hammer_geoms(body, name, rgba)
        else:
            geom = body.add_geom()
            geom.name = f"{name}_cube_geom"
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            size = 0.04 * self.config.object_scales / 2.0
            geom.size = size.astype(float)
            geom.density = 400.0
            geom.rgba = rgba
            geom.friction = np.array([1.0, 0.005, 0.0001])
            self._set_low_bounce_contact(geom)
            geoms = [geom]

        if disable_contact:
            for geom in geoms:
                geom.contype = 0
                geom.conaffinity = 0

        if add_frame:
            self._add_local_frame(body, frame_name)

    @staticmethod
    def _set_low_bounce_contact(geom) -> None:
        # MuJoCo has no direct "restitution" scalar here; contact bounce is
        # shaped through solref/solimp. Keep this only mildly more damped than
        # the default contact settings so the cube is less springy without
        # becoming unnaturally "dead".
        geom.solref = np.array([0.03, 1.2])
        geom.solimp = np.array([0.90, 0.95, 0.01, 0.5, 2.0])

    def _add_local_frame(self, body, frame_name: str) -> None:
        axis_length = 0.09
        thickness = 0.004
        axes = [
            (
                "x",
                np.array([axis_length / 2.0, 0.0, 0.0]),
                np.array([axis_length / 2.0, thickness, thickness]),
                np.array([1.0, 0.1, 0.1, 1.0]),
            ),
            (
                "y",
                np.array([0.0, axis_length / 2.0, 0.0]),
                np.array([thickness, axis_length / 2.0, thickness]),
                np.array([0.1, 0.8, 0.1, 1.0]),
            ),
            (
                "z",
                np.array([0.0, 0.0, axis_length / 2.0]),
                np.array([thickness, thickness, axis_length / 2.0]),
                np.array([0.1, 0.3, 1.0, 1.0]),
            ),
        ]
        for axis_name, pos, size, rgba in axes:
            geom = body.add_geom()
            geom.name = f"{frame_name}_{axis_name}_axis"
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.pos = pos
            geom.size = size
            geom.rgba = rgba
            geom.contype = 0
            geom.conaffinity = 0

    def _add_hammer_geoms(self, body, name: str, rgba: np.ndarray):
        handle = body.add_geom()
        handle.name = f"{name}_hammer_handle"
        handle.type = mujoco.mjtGeom.mjGEOM_BOX
        handle.size = np.array([0.141 / 2.0, 0.03025 / 2.0, 0.0271 / 2.0])
        handle.density = 400.0
        handle.rgba = rgba
        handle.friction = np.array([1.0, 0.005, 0.0001])
        self._set_low_bounce_contact(handle)

        head = body.add_geom()
        head.name = f"{name}_hammer_head"
        head.type = mujoco.mjtGeom.mjGEOM_BOX
        head.size = np.array([0.03025 / 2.0, 0.09 / 2.0, 0.045 / 2.0])
        head.pos = np.array([0.141 / 2.0, 0.0, 0.0])
        head.density = 400.0
        head.rgba = rgba
        head.friction = np.array([1.0, 0.005, 0.0001])
        self._set_low_bounce_contact(head)
        return [handle, head]

    def _add_position_actuators(self, spec: mujoco.MjSpec) -> None:
        for name in JOINT_NAMES:
            actuator = spec.add_actuator()
            actuator.name = f"{name}_pos"
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.target = name
            actuator.ctrllimited = True
            actuator.ctrlrange = np.array(
                [LOWER_LIMITS[JOINT_NAMES.index(name)], UPPER_LIMITS[JOINT_NAMES.index(name)]]
            )
            kp = 300.0 if JOINT_NAMES.index(name) < 6 else 5.0
            kv = 20.0 if JOINT_NAMES.index(name) < 6 else 0.25
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.gainprm[0] = kp
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kv

    def _validate(self) -> None:
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        missing_joints = [name for name in JOINT_NAMES if name not in joint_names]
        if missing_joints:
            raise RuntimeError(f"Missing MuJoCo joints: {missing_joints}")
        actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        missing_actuators = [
            f"{name}_pos" for name in JOINT_NAMES if f"{name}_pos" not in actuator_names
        ]
        if missing_actuators:
            raise RuntimeError(f"Missing MuJoCo actuators: {missing_actuators}")

    def set_robot_joint_positions(self, q: np.ndarray) -> None:
        if q.shape != (N_ACT,):
            raise ValueError(f"q.shape={q.shape}, expected {(N_ACT,)}")
        self.data.qpos[self._joint_qpos_adrs] = q
        self.data.qvel[self._joint_dof_adrs] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_robot_joint_pos_targets(self, q_targets: np.ndarray) -> None:
        if q_targets.shape != (N_ACT,):
            raise ValueError(f"q_targets.shape={q_targets.shape}, expected {(N_ACT,)}")
        self.robot_joint_pos_targets = q_targets.copy()

    def sim_step(self) -> None:
        self.data.ctrl[self._actuator_ids] = self.robot_joint_pos_targets
        mujoco.mj_step(self.model, self.data)

    def step_for(self, dt: float) -> None:
        steps = max(1, int(round(dt / self.config.sim_dt)))
        for _ in range(steps):
            self.sim_step()
            if self.config.enable_viewer:
                self.viewer.sync()

    def body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = self.model.body(body_name).id
        return self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy()

    def fingertip_positions(self) -> np.ndarray:
        positions = []
        for body_name, local_offset in zip(FINGERTIP_BODY_NAMES, FINGERTIP_LOCAL_OFFSETS):
            pos, quat_wxyz = self.body_pose(body_name)
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            positions.append(pos + Rotation.from_quat(quat_xyzw).apply(local_offset))
        return np.array(positions, dtype=np.float32)

    def get_sim_state(self) -> dict[str, np.ndarray]:
        object_pos, object_quat_wxyz = self.body_pose("object")
        goal_pos, goal_quat_wxyz = self.body_pose("goal_object")
        palm_pos, palm_quat_wxyz = self.body_pose("wrist_3_link")
        palm_center_pos = palm_pos + Rotation.from_quat(
            palm_quat_wxyz[[1, 2, 3, 0]]
        ).apply(np.array([0.0, 0.0, 0.16]))
        return {
            "joint_positions": self.data.qpos[self._joint_qpos_adrs].copy(),
            "joint_velocities": self.data.qvel[self._joint_dof_adrs].copy(),
            "palm_pos": palm_center_pos.astype(np.float32),
            "palm_quat_wxyz": palm_quat_wxyz.copy(),
            "fingertip_positions": self.fingertip_positions(),
            "object_pos": object_pos.copy(),
            "object_quat_wxyz": object_quat_wxyz.copy(),
            "goal_object_pos": goal_pos.copy(),
            "goal_object_quat_wxyz": goal_quat_wxyz.copy(),
        }

    def run_open_loop(self) -> None:
        try:
            while True:
                start = time.time()
                self.sim_step()
                if self.config.enable_viewer:
                    self.viewer.sync()
                sleep_dt = self.config.sim_dt - (time.time() - start)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
        finally:
            self.close()


def main() -> None:
    sim = Ur5eDeltoMujocoSim(Ur5eDeltoMujocoConfig(enable_viewer=True))
    sim.run_open_loop()


if __name__ == "__main__":
    main()
