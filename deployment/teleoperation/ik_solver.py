from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from transforms import make_transform


@dataclass(frozen=True)
class IkDiagnostics:
    position_error_m: float
    orientation_error_rad: float
    maximum_joint_velocity_rad_s: float


class Ur5DampedLeastSquaresIk:
    """Small dependency-free differential IK solver built on MuJoCo Jacobians."""

    JOINT_NAMES = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )

    def __init__(
        self,
        *,
        model_path: str | Path,
        end_effector_body: str,
        damping: float,
        position_gain: float,
        orientation_gain: float,
        maximum_joint_velocity_rad_s: float,
    ) -> None:
        if damping <= 0.0:
            raise ValueError("IK damping must be positive")
        if maximum_joint_velocity_rad_s <= 0.0:
            raise ValueError("maximum_joint_velocity_rad_s must be positive")
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path)))
        self.data = mujoco.MjData(self.model)
        self.body_id = self.model.body(end_effector_body).id
        self.damping = float(damping)
        self.position_gain = float(position_gain)
        self.orientation_gain = float(orientation_gain)
        self.maximum_joint_velocity_rad_s = float(
            maximum_joint_velocity_rad_s
        )

        self.joint_ids = np.array(
            [self.model.joint(name).id for name in self.JOINT_NAMES],
            dtype=np.int32,
        )
        self.qpos_addresses = np.array(
            [self.model.jnt_qposadr[index] for index in self.joint_ids],
            dtype=np.int32,
        )
        self.dof_addresses = np.array(
            [self.model.jnt_dofadr[index] for index in self.joint_ids],
            dtype=np.int32,
        )

    def forward(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (6,):
            raise ValueError(f"Expected six UR5 joints, got {q.shape}")
        self.data.qpos[self.qpos_addresses] = q
        mujoco.mj_forward(self.model, self.data)
        return make_transform(
            self.data.xmat[self.body_id].reshape(3, 3),
            self.data.xpos[self.body_id],
        )

    def _clamp_configuration(self, q: np.ndarray) -> np.ndarray:
        result = q.copy()
        for output_index, joint_id in enumerate(self.joint_ids):
            if self.model.jnt_limited[joint_id]:
                low, high = self.model.jnt_range[joint_id]
                result[output_index] = np.clip(
                    result[output_index], low, high
                )
        return result

    def step(
        self,
        q: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, IkDiagnostics]:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        current = self.forward(q)
        position_error = target[:3, 3] - current[:3, 3]
        orientation_error = Rotation.from_matrix(
            target[:3, :3] @ current[:3, :3].T
        ).as_rotvec()

        jac_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(
            self.model,
            self.data,
            jac_position,
            jac_rotation,
            self.body_id,
        )
        jacobian = np.vstack(
            (
                jac_position[:, self.dof_addresses],
                jac_rotation[:, self.dof_addresses],
            )
        )
        error = np.concatenate(
            (
                self.position_gain * position_error,
                self.orientation_gain * orientation_error,
            )
        )

        regularized = (
            jacobian.T @ jacobian
            + self.damping**2 * np.eye(len(self.dof_addresses))
        )
        joint_velocity = np.linalg.solve(
            regularized,
            jacobian.T @ error,
        )
        joint_velocity = np.clip(
            joint_velocity,
            -self.maximum_joint_velocity_rad_s,
            self.maximum_joint_velocity_rad_s,
        )
        next_q = self._clamp_configuration(
            np.asarray(q, dtype=np.float64) + dt * joint_velocity
        )
        return next_q, IkDiagnostics(
            position_error_m=float(np.linalg.norm(position_error)),
            orientation_error_rad=float(np.linalg.norm(orientation_error)),
            maximum_joint_velocity_rad_s=float(
                np.max(np.abs(joint_velocity))
            ),
        )
