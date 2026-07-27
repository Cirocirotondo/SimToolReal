from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from deployment.mujoco_ur5e_delto.policy_adapter import (
    DEFAULT_HAND_SIDE,
    DEFAULT_JOINT_POS,
    N_ACT,
    HandSide,
    fingertip_body_names_for_hand,
    fingertip_local_offsets_for_hand,
    joint_limits_for_hand,
    joint_names_for_hand,
    robot_urdf_path_for_hand,
    validate_hand_side,
)
from isaacgymenvs.utils.utils import get_repo_root_dir


@dataclass
class Ur5eDeltoMujocoConfig:
    enable_viewer: bool = True
    sim_dt: float = 1.0 / 600.0
    arm_kp: float = 300.0
    """Arm position-controller proportional gain."""
    arm_kv: float = 20.0
    """Arm position-controller velocity damping."""
    hand_kp: float = 5.0
    """Hand position-controller proportional gain."""
    hand_kv: float = 0.25
    """Hand position-controller velocity damping."""
    hand_side: HandSide = DEFAULT_HAND_SIDE
    robot_urdf_path: Optional[Path] = None
    workspace_y: float = -0.6
    """Table/object y position in front of the robot base."""
    table_center_z: float = 0.38
    """Z position of the table body center."""
    table_object_z_offset: float = 0.25
    """Object center height above the table body center."""
    goal_object_z_offset: float = 0.35
    """Goal object center height above the table body center."""
    table_size_x: float = 0.8
    """Full table size along the MuJoCo x axis."""
    table_size_y: float = 0.8
    """Full table size along the MuJoCo y axis."""
    table_size_z: float = 0.3
    """Full table size along the MuJoCo z axis."""
    show_goal_marker: bool = True
    """Display the visual-only goal body and frame in the MuJoCo viewer."""
    show_object_frame: bool = True
    """Display local axes on the movable object in the MuJoCo viewer."""
    initial_joint_pos: np.ndarray = field(default_factory=lambda: DEFAULT_JOINT_POS.copy())
    joint_limit_overrides: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    """Optional per-joint limits for specialized interactive tools."""
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

    def __post_init__(self) -> None:
        self.hand_side = validate_hand_side(self.hand_side)
        for name in ("arm_kp", "arm_kv", "hand_kp", "hand_kv"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.robot_urdf_path is None:
            self.robot_urdf_path = robot_urdf_path_for_hand(self.hand_side)


class Ur5eDeltoMujocoSim:
    def __init__(self, config: Ur5eDeltoMujocoConfig):
        self.config = config
        self.repo_root = get_repo_root_dir()
        self.joint_names = joint_names_for_hand(self.config.hand_side)
        self.fingertip_body_names = fingertip_body_names_for_hand(
            self.config.hand_side
        )
        self.fingertip_local_offsets = fingertip_local_offsets_for_hand(
            self.config.hand_side
        )
        lower_limits, upper_limits = joint_limits_for_hand(self.config.hand_side)
        self.lower_limits = lower_limits.copy()
        self.upper_limits = upper_limits.copy()
        for joint_name, limits in self.config.joint_limit_overrides.items():
            if joint_name not in self.joint_names:
                raise ValueError(f"Unknown joint limit override: {joint_name}")
            lower, upper = limits
            if lower >= upper:
                raise ValueError(
                    f"Invalid limits for {joint_name}: lower={lower}, upper={upper}"
                )
            index = self.joint_names.index(joint_name)
            self.lower_limits[index] = lower
            self.upper_limits[index] = upper
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
        assert urdf_path is not None
        if not urdf_path.is_absolute():
            urdf_path = self.repo_root / urdf_path
        text = urdf_path.read_text(encoding="utf-8")
        if "<mujoco>" not in text:
            text = re.sub(
                r'(<robot\s+name="[^"]+">)',
                r'\1\n  <mujoco><compiler strippath="false"/></mujoco>',
                text,
                count=1,
            )

        def replace_mesh(match: re.Match[str]) -> str:
            filename = match.group(1)
            if filename.startswith("urdf/"):
                mesh_path = self.repo_root / "assets" / filename
            else:
                mesh_path = urdf_path.parent / filename
            return f'filename="{mesh_path}"'

        text = re.sub(r'filename="([^"]+)"', replace_mesh, text)
        tmp_path = (
            Path(self._tmp_dir.name)
            / f"ur5e_{self.config.hand_side}_dg5f_mujoco.urdf"
        )
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
        for joint_name, (lower, upper) in self.config.joint_limit_overrides.items():
            joint_id = self.model.joint(joint_name).id
            self.model.jnt_range[joint_id] = np.array([lower, upper])
        self.model.opt.timestep = self.config.sim_dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.iterations = 20
        self.model.opt.ls_iterations = 50

        self._joint_qpos_adrs = np.array(
            [self.model.joint(name).qposadr[0] for name in self.joint_names],
            dtype=np.int32,
        )
        self._joint_dof_adrs = np.array(
            [self.model.joint(name).dofadr[0] for name in self.joint_names],
            dtype=np.int32,
        )
        self._actuator_ids = np.array(
            [self.model.actuator(f"{name}_pos").id for name in self.joint_names],
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
        table_geom.size = np.array(
            [
                self.config.table_size_x / 2.0,
                self.config.table_size_y / 2.0,
                self.config.table_size_z / 2.0,
            ]
        )
        table_geom.rgba = np.array([0.9, 0.9, 0.9, 1.0])
        table_geom.friction = np.array([1.0, 0.005, 0.0001])
        self._set_low_bounce_contact(table_geom)

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
        elif self.config.object_name == "dumbbell_20x9x9cm":
            geoms = self._add_dumbbell_20x9x9cm_geoms(body, name, rgba)
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
        # shaped through solref/solimp. Use a stiff, damped contact so the cube
        # does not visibly sink into the table while keeping bounce low.
        geom.solref = np.array([0.012, 1.4])
        geom.solimp = np.array([0.95, 0.99, 0.002, 0.5, 2.0])

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

    def _add_dumbbell_20x9x9cm_geoms(
        self, body, name: str, rgba: np.ndarray
    ):
        # The source STL is an exact union of these three boxes. Keeping them
        # separate avoids MuJoCo convexifying the concave dumbbell silhouette.
        parts = (
            ("handle", [0.03, 0.14, 0.03], [0.0, 0.0, 0.0]),
            ("head_negative_y", [0.09, 0.03, 0.09], [0.0, -0.085, 0.0]),
            ("head_positive_y", [0.09, 0.03, 0.09], [0.0, 0.085, 0.0]),
        )
        geoms = []
        for part_name, full_size, pos in parts:
            geom = body.add_geom()
            geom.name = f"{name}_dumbbell_{part_name}"
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = np.asarray(full_size, dtype=np.float64) / 2.0
            geom.pos = np.asarray(pos, dtype=np.float64)
            geom.density = 400.0
            geom.rgba = rgba
            geom.friction = np.array([1.0, 0.005, 0.0001])
            self._set_low_bounce_contact(geom)
            geoms.append(geom)

        # Massless, non-colliding plates disambiguate all six local directions.
        # MuJoCo box sizes are half-extents.
        face_alpha = float(rgba[3])
        face_specs = (
            (
                "px_negative_head",
                [0.0455, -0.085, 0.0],
                [0.0005, 0.015, 0.045],
                [1.0, 0.15, 0.15, face_alpha],
            ),
            (
                "px_positive_head",
                [0.0455, 0.085, 0.0],
                [0.0005, 0.015, 0.045],
                [1.0, 0.15, 0.15, face_alpha],
            ),
            (
                "nx_negative_head",
                [-0.0455, -0.085, 0.0],
                [0.0005, 0.015, 0.045],
                [1.0, 0.5, 0.0, face_alpha],
            ),
            (
                "nx_positive_head",
                [-0.0455, 0.085, 0.0],
                [0.0005, 0.015, 0.045],
                [1.0, 0.5, 0.0, face_alpha],
            ),
            (
                "py",
                [0.0, 0.1005, 0.0],
                [0.045, 0.0005, 0.045],
                [0.15, 0.85, 0.15, face_alpha],
            ),
            (
                "ny",
                [0.0, -0.1005, 0.0],
                [0.045, 0.0005, 0.045],
                [0.1, 0.55, 0.45, face_alpha],
            ),
            (
                "pz_negative_head",
                [0.0, -0.085, 0.0455],
                [0.045, 0.015, 0.0005],
                [0.2, 0.35, 1.0, face_alpha],
            ),
            (
                "pz_positive_head",
                [0.0, 0.085, 0.0455],
                [0.045, 0.015, 0.0005],
                [0.2, 0.35, 1.0, face_alpha],
            ),
            (
                "nz_negative_head",
                [0.0, -0.085, -0.0455],
                [0.045, 0.015, 0.0005],
                [1.0, 0.85, 0.1, face_alpha],
            ),
            (
                "nz_positive_head",
                [0.0, 0.085, -0.0455],
                [0.045, 0.015, 0.0005],
                [1.0, 0.85, 0.1, face_alpha],
            ),
        )
        for face_name, pos, size, face_rgba in face_specs:
            face = body.add_geom()
            face.name = f"{name}_dumbbell_face_{face_name}"
            face.type = mujoco.mjtGeom.mjGEOM_BOX
            face.pos = np.asarray(pos, dtype=np.float64)
            face.size = np.asarray(size, dtype=np.float64)
            face.rgba = np.asarray(face_rgba, dtype=np.float64)
            face.density = 0.0
            face.contype = 0
            face.conaffinity = 0
            geoms.append(face)
        return geoms

    def _add_position_actuators(self, spec: mujoco.MjSpec) -> None:
        for index, name in enumerate(self.joint_names):
            actuator = spec.add_actuator()
            actuator.name = f"{name}_pos"
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.target = name
            actuator.ctrllimited = True
            actuator.ctrlrange = np.array(
                [self.lower_limits[index], self.upper_limits[index]]
            )
            kp = self.config.arm_kp if index < 6 else self.config.hand_kp
            kv = self.config.arm_kv if index < 6 else self.config.hand_kv
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.gainprm[0] = kp
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kv

    def _validate(self) -> None:
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        missing_joints = [
            name for name in self.joint_names if name not in joint_names
        ]
        if missing_joints:
            raise RuntimeError(f"Missing MuJoCo joints: {missing_joints}")
        actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        missing_actuators = [
            f"{name}_pos"
            for name in self.joint_names
            if f"{name}_pos" not in actuator_names
        ]
        if missing_actuators:
            raise RuntimeError(f"Missing MuJoCo actuators: {missing_actuators}")

    def set_robot_joint_positions(self, q: np.ndarray) -> None:
        if q.shape != (N_ACT,):
            raise ValueError(f"q.shape={q.shape}, expected {(N_ACT,)}")
        self.data.qpos[self._joint_qpos_adrs] = q
        self.data.qvel[self._joint_dof_adrs] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.config.enable_viewer:
            self.viewer.sync()

    def set_robot_joint_pos_targets(self, q_targets: np.ndarray) -> None:
        if q_targets.shape != (N_ACT,):
            raise ValueError(f"q_targets.shape={q_targets.shape}, expected {(N_ACT,)}")
        self.robot_joint_pos_targets = q_targets.copy()

    def set_goal_object_pose(
        self, pos: np.ndarray, quat_wxyz: np.ndarray
    ) -> None:
        if pos.shape != (3,):
            raise ValueError(f"pos.shape={pos.shape}, expected {(3,)}")
        if quat_wxyz.shape != (4,):
            raise ValueError(
                f"quat_wxyz.shape={quat_wxyz.shape}, expected {(4,)}"
            )
        goal_body_id = self.model.body("goal_object").id
        self.model.body_pos[goal_body_id] = pos
        self.model.body_quat[goal_body_id] = quat_wxyz
        mujoco.mj_forward(self.model, self.data)
        if self.config.enable_viewer:
            self.viewer.sync()

    def set_object_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        if pos.shape != (3,):
            raise ValueError(f"pos.shape={pos.shape}, expected {(3,)}")
        if quat_wxyz.shape != (4,):
            raise ValueError(
                f"quat_wxyz.shape={quat_wxyz.shape}, expected {(4,)}"
            )
        joint = self.model.joint("object_free_joint")
        qposadr = joint.qposadr[0]
        dofadr = joint.dofadr[0]
        self.data.qpos[qposadr : qposadr + 3] = pos
        self.data.qpos[qposadr + 3 : qposadr + 7] = quat_wxyz
        self.data.qvel[dofadr : dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.config.enable_viewer:
            self.viewer.sync()

    def add_object_velocity(
        self,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        if linear_velocity.shape != (3,):
            raise ValueError(
                f"linear_velocity.shape={linear_velocity.shape}, expected {(3,)}"
            )
        if angular_velocity.shape != (3,):
            raise ValueError(
                f"angular_velocity.shape={angular_velocity.shape}, expected {(3,)}"
            )
        dofadr = self.model.joint("object_free_joint").dofadr[0]
        self.data.qvel[dofadr : dofadr + 3] += linear_velocity
        self.data.qvel[dofadr + 3 : dofadr + 6] += angular_velocity

    def set_gravity_enabled(self, enabled: bool) -> None:
        self.model.opt.gravity[:] = np.array(
            [0.0, 0.0, -9.81] if enabled else [0.0, 0.0, 0.0],
            dtype=np.float64,
        )

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
        for body_name, local_offset in zip(
            self.fingertip_body_names, self.fingertip_local_offsets
        ):
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
