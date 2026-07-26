from __future__ import annotations

import json
from pathlib import Path
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


def load_transform(path: str | Path, *keys: str) -> np.ndarray:
    """Load a 4x4 transform from .npy or from a calibration JSON."""
    path = Path(path).expanduser().resolve()
    if path.suffix == ".npy":
        matrix = np.load(path)
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        value: Any = payload
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
                break
        if isinstance(value, dict) and "matrix_4x4" in value:
            value = value["matrix_4x4"]
        matrix = np.asarray(value, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported transform file: {path}")

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path} does not contain a finite 4x4 transform")
    return make_transform(matrix[:3, :3], matrix[:3, 3])


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("Cannot average an empty transform list")
    translations = np.array([item[:3, 3] for item in transforms])
    rotations = Rotation.from_matrix(
        np.array([item[:3, :3] for item in transforms])
    ).mean()
    return make_transform(rotations.as_matrix(), translations.mean(axis=0))


class RelativeWristMapper:
    """Map a captured MANUS wrist pose to the robot home end-effector pose."""

    def __init__(
        self,
        *,
        initial_world_wrist: np.ndarray,
        home_model_ee: np.ndarray,
        rotation_model_from_world: np.ndarray,
        position_scale: float = 1.0,
        track_orientation: bool = True,
    ) -> None:
        if position_scale <= 0.0:
            raise ValueError("position_scale must be positive")
        self.initial_world_wrist = np.asarray(
            initial_world_wrist, dtype=np.float64
        ).copy()
        self.home_model_ee = np.asarray(home_model_ee, dtype=np.float64).copy()
        self.rotation_model_from_world = project_rotation(
            rotation_model_from_world
        )
        self.position_scale = float(position_scale)
        self.track_orientation = bool(track_orientation)

    def target(self, world_wrist: np.ndarray) -> np.ndarray:
        world_wrist = np.asarray(world_wrist, dtype=np.float64)
        delta_world = (
            world_wrist[:3, 3] - self.initial_world_wrist[:3, 3]
        )
        result = self.home_model_ee.copy()
        result[:3, 3] = (
            self.home_model_ee[:3, 3]
            + self.position_scale
            * (self.rotation_model_from_world @ delta_world)
        )
        if self.track_orientation:
            local_relative_rotation = (
                self.initial_world_wrist[:3, :3].T
                @ world_wrist[:3, :3]
            )
            result[:3, :3] = project_rotation(
                self.home_model_ee[:3, :3] @ local_relative_rotation
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
