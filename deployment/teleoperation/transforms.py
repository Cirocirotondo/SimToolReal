from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a nearly orthogonal matrix onto SO(3)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Rotation must have shape (3, 3), got {matrix.shape}")
    u, _, vt = np.linalg.svd(matrix)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = project_rotation(rotation)
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(
            f"Translation must have shape (3,), got {translation.shape}"
        )
    result[:3, 3] = translation
    return result


def pose_array_to_matrix(pose: Any) -> np.ndarray:
    """Convert [x, y, z, qw, qx, qy, qz] to a homogeneous transform."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"Expected a finite 7D pose, got shape {pose.shape}")
    qw, qx, qy, qz = pose[3:]
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return make_transform(rotation, pose[:3])


def pose_dict_to_matrix(pose: dict[str, Any]) -> np.ndarray:
    """Accept the dictionary format published by tag-pose-estimation."""
    if "matrix_4x4" in pose:
        matrix = np.asarray(pose["matrix_4x4"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("matrix_4x4 must have shape (4, 4)")
        return make_transform(matrix[:3, :3], matrix[:3, 3])

    position = np.asarray(pose.get("position"), dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("Pose dictionary does not contain a 3D position")

    if "rotation_matrix" in pose:
        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
    elif "quaternion_wxyz" in pose:
        qw, qx, qy, qz = np.asarray(
            pose["quaternion_wxyz"], dtype=np.float64
        )
        rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    elif "quaternion_xyzw" in pose:
        rotation = Rotation.from_quat(
            np.asarray(pose["quaternion_xyzw"], dtype=np.float64)
        ).as_matrix()
    else:
        raise ValueError("Pose dictionary does not contain an orientation")

    return make_transform(rotation, position)


def pose_to_matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, dict):
        return pose_dict_to_matrix(pose)
    return pose_array_to_matrix(pose)


def apply_local_origin_offset(
    pose: np.ndarray,
    offset_local_m: np.ndarray,
) -> np.ndarray:
    """Move a pose origin by a rigid offset expressed in its local frame."""
    pose = np.asarray(pose, dtype=np.float64)
    offset_local_m = np.asarray(offset_local_m, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("pose must have shape (4, 4)")
    if offset_local_m.shape != (3,):
        raise ValueError("offset_local_m must have shape (3,)")
    result = pose.copy()
    result[:3, 3] += pose[:3, :3] @ offset_local_m
    return result


class RelativeBoardMapper:
    """Map a captured MANUS board pose to the robot home end-effector pose."""

    def __init__(
        self,
        *,
        initial_world_board: np.ndarray,
        home_model_ee: np.ndarray,
        translation_axis_map: np.ndarray,
        orientation_axis_map: np.ndarray | None = None,
        orientation_mode: str = "mapped-local",
        position_scale: float = 1.0,
        track_orientation: bool = True,
    ) -> None:
        if position_scale <= 0.0:
            raise ValueError("position_scale must be positive")
        self.initial_world_board = np.asarray(
            initial_world_board, dtype=np.float64
        ).copy()
        self.home_model_ee = np.asarray(home_model_ee, dtype=np.float64).copy()
        self.translation_axis_map = np.asarray(
            translation_axis_map, dtype=np.float64
        )
        if self.translation_axis_map.shape != (3, 3):
            raise ValueError("translation_axis_map must have shape (3, 3)")
        if not np.allclose(
            self.translation_axis_map.T @ self.translation_axis_map,
            np.eye(3),
            atol=1e-6,
        ):
            raise ValueError("translation_axis_map must be orthonormal")
        if orientation_axis_map is None:
            orientation_axis_map = np.eye(3)
        self.orientation_axis_map = np.asarray(
            orientation_axis_map, dtype=np.float64
        )
        if self.orientation_axis_map.shape != (3, 3):
            raise ValueError("orientation_axis_map must have shape (3, 3)")
        if not np.allclose(
            self.orientation_axis_map.T @ self.orientation_axis_map,
            np.eye(3),
            atol=1e-6,
        ):
            raise ValueError("orientation_axis_map must be orthonormal")
        if orientation_mode not in ("mapped-local", "spatial-relative"):
            raise ValueError(
                "orientation_mode must be 'mapped-local' or "
                "'spatial-relative'"
            )
        self.orientation_mode = orientation_mode
        self.position_scale = float(position_scale)
        self.track_orientation = bool(track_orientation)

    def target(self, world_board: np.ndarray) -> np.ndarray:
        world_board = np.asarray(world_board, dtype=np.float64)
        delta_world = (
            world_board[:3, 3] - self.initial_world_board[:3, 3]
        )
        result = self.home_model_ee.copy()
        result[:3, 3] = (
            self.home_model_ee[:3, 3]
            + self.position_scale
            * (self.translation_axis_map @ delta_world)
        )
        if self.track_orientation:
            if self.orientation_mode == "spatial-relative":
                spatial_relative_rotation = (
                    world_board[:3, :3]
                    @ self.initial_world_board[:3, :3].T
                )
                result[:3, :3] = project_rotation(
                    spatial_relative_rotation
                    @ self.home_model_ee[:3, :3]
                )
            else:
                local_relative_rotation = (
                    self.initial_world_board[:3, :3].T
                    @ world_board[:3, :3]
                )
                relative_rotation_vector = Rotation.from_matrix(
                    local_relative_rotation
                ).as_rotvec()
                mapped_relative_rotation = Rotation.from_rotvec(
                    self.orientation_axis_map @ relative_rotation_vector
                ).as_matrix()
                result[:3, :3] = project_rotation(
                    self.home_model_ee[:3, :3] @ mapped_relative_rotation
                )
        return result


def limit_pose_step(
    previous: np.ndarray,
    requested: np.ndarray,
    *,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> np.ndarray:
    result = requested.copy()
    delta = requested[:3, 3] - previous[:3, 3]
    distance = float(np.linalg.norm(delta))
    if distance > max_translation_step_m:
        result[:3, 3] = (
            previous[:3, 3] + delta * max_translation_step_m / distance
        )

    relative = Rotation.from_matrix(
        requested[:3, :3] @ previous[:3, :3].T
    )
    rotvec = relative.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if angle > max_rotation_step_rad:
        limited_delta = Rotation.from_rotvec(
            rotvec * max_rotation_step_rad / angle
        )
        result[:3, :3] = limited_delta.as_matrix() @ previous[:3, :3]
    return result
