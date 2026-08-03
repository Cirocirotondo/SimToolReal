"""Batched reference-motion loading and interpolation for imitation learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEMONSTRATION_DIRECTORY = (
    REPO_ROOT / "deployment" / "teleoperation" / "demonstrations"
)


def sample_reference_phases(
    num_samples: int,
    max_phase: float,
    distribution: str,
    device,
) -> torch.Tensor:
    """Sample RSI phases from a uniform or phase-zero-weighted distribution."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if not 0.0 < max_phase <= 1.0:
        raise ValueError("max_phase must be in (0, 1]")

    distribution = str(distribution).lower()
    uniform = torch.rand(num_samples, device=device)
    if distribution == "uniform":
        normalized_phase = uniform
    elif distribution == "triangular":
        # Inverse CDF of Triangular(0, 1, mode=0):
        # F(x) = 1 - (1 - x)^2. Its density is largest at phase zero,
        # decreases linearly, and has mean 1/3.
        normalized_phase = 1.0 - torch.sqrt(1.0 - uniform)
    else:
        raise ValueError(
            "referenceInitDistribution must be 'uniform' or 'triangular', "
            f"got {distribution!r}"
        )
    return normalized_phase * max_phase


def resolve_demonstration(path_or_name: str) -> Path:
    supplied = Path(path_or_name).expanduser()
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.extend(
            [Path.cwd() / supplied, REPO_ROOT / supplied, DEFAULT_DEMONSTRATION_DIRECTORY / supplied]
        )
    for candidate in candidates:
        path = candidate.with_suffix(".npz") if candidate.suffix != ".npz" else candidate
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve demonstration {path_or_name!r}")


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        axis=-1,
    )


def _rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_xyz = quaternion[..., :3]
    uv = np.cross(q_xyz, vector)
    uuv = np.cross(q_xyz, uv)
    return vector + 2.0 * (quaternion[..., 3:4] * uv + uuv)


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("Demonstration contains an invalid zero quaternion")
    return quaternions / norms


def _smooth_signal(
    values: np.ndarray,
    timestamps: np.ndarray,
    window_s: float,
) -> np.ndarray:
    """Apply a centered triangular filter without shifting the trajectory."""
    if window_s <= 0.0 or len(values) < 3:
        return values.astype(np.float32, copy=False)
    median_dt = float(np.median(np.diff(timestamps)))
    window_samples = max(3, int(round(window_s / median_dt)))
    if window_samples % 2 == 0:
        window_samples += 1
    largest_odd_window = len(values) if len(values) % 2 else len(values) - 1
    window_samples = min(window_samples, largest_odd_window)
    if window_samples < 3:
        return values.astype(np.float32, copy=False)

    half = window_samples // 2
    ramp = np.arange(1, half + 2, dtype=np.float64)
    kernel = np.concatenate([ramp, ramp[-2::-1]])
    kernel /= kernel.sum()
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    smoothed = np.stack(
        [
            np.convolve(padded[:, column], kernel, mode="valid")
            for column in range(values.shape[1])
        ],
        axis=-1,
    )
    return smoothed.astype(np.float32)


def _filtered_derivative(
    values: np.ndarray,
    timestamps: np.ndarray,
    window_s: float,
) -> np.ndarray:
    derivative = np.gradient(values, timestamps, axis=0)
    return _smooth_signal(derivative, timestamps, window_s)


def _quaternion_angular_velocity_xyzw(
    quaternions: np.ndarray,
    timestamps: np.ndarray,
    window_s: float,
) -> np.ndarray:
    """Differentiate orientations into world-frame angular velocities."""
    continuous = _normalize_quaternions(quaternions).copy()
    for index in range(1, len(continuous)):
        if np.dot(continuous[index - 1], continuous[index]) < 0.0:
            continuous[index] *= -1.0

    angular_velocity = np.empty((len(continuous), 3), dtype=np.float64)
    for index in range(len(continuous)):
        lower = max(index - 1, 0)
        upper = min(index + 1, len(continuous) - 1)
        delta = _quat_multiply_xyzw(
            continuous[upper : upper + 1],
            np.concatenate(
                [
                    -continuous[lower : lower + 1, :3],
                    continuous[lower : lower + 1, 3:4],
                ],
                axis=-1,
            ),
        )[0]
        if delta[3] < 0.0:
            delta *= -1.0
        vector_norm = float(np.linalg.norm(delta[:3]))
        if vector_norm < 1e-8:
            rotation_vector = 2.0 * delta[:3]
        else:
            angle = 2.0 * np.arctan2(vector_norm, delta[3])
            rotation_vector = delta[:3] * (angle / vector_norm)
        angular_velocity[index] = rotation_vector / (
            timestamps[upper] - timestamps[lower]
        )
    return _smooth_signal(angular_velocity, timestamps, window_s)


@dataclass(frozen=True)
class ReferenceSample:
    arm_q: torch.Tensor
    arm_dq: torch.Tensor
    hand_q: torch.Tensor
    hand_dq: torch.Tensor
    palm_pos: torch.Tensor
    palm_quat_xyzw: torch.Tensor
    palm_lin_vel: torch.Tensor
    palm_ang_vel: torch.Tensor
    object_pos: Optional[torch.Tensor] = None
    object_quat_xyzw: Optional[torch.Tensor] = None
    object_lin_vel: Optional[torch.Tensor] = None
    object_ang_vel: Optional[torch.Tensor] = None


class DemonstrationReference:
    """One motion clip resident on the training device."""

    def __init__(
        self,
        path_or_name: str,
        *,
        device: str | torch.device,
        hand_source: str = "measured",
        world_yaw_offset_deg: float = 180.0,
        world_position_offset_m: tuple[float, float, float] = (0.0, 0.6, 0.0),
        ee_to_palm_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.16),
        ee_to_palm_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        velocity_filter_window_s: float = 0.0,
        load_object_pose: bool = False,
        object_position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        object_orientation_offset_xyzw: tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    ) -> None:
        self.velocity_filter_window_s = float(velocity_filter_window_s)
        if self.velocity_filter_window_s < 0.0:
            raise ValueError("velocity_filter_window_s must be non-negative")
        self.path = resolve_demonstration(path_or_name)
        with np.load(self.path, allow_pickle=False) as data:
            arm_q = np.asarray(data["arm_q"], dtype=np.float32)
            arm_dq = np.asarray(data["arm_dq"], dtype=np.float32)
            ee_pose = np.asarray(data["ee_pose_measured"], dtype=np.float32)
            hand_key = f"hand_q_{hand_source}"
            valid_key = f"{hand_key}_valid"
            hand_q = np.asarray(data[hand_key], dtype=np.float32)
            timestamps = np.asarray(data["timestamp"], dtype=np.float64)
            derivative_timestamps = np.asarray(
                data["monotonic_timestamp"]
                if "monotonic_timestamp" in data
                else timestamps,
                dtype=np.float64,
            )
            object_pose = None
            if load_object_pose:
                if "cube_pose" not in data:
                    raise ValueError(
                        f"{self.path.name} does not contain cube_pose required "
                        "for object tracking"
                    )
                object_pose = np.asarray(data["cube_pose"], dtype=np.float32)
                if "cube_pose_valid" in data:
                    object_pose_valid = np.asarray(
                        data["cube_pose_valid"], dtype=bool
                    )
                    if object_pose_valid.shape != (len(timestamps),):
                        raise ValueError(
                            "Invalid cube_pose_valid: expected "
                            f"{(len(timestamps),)}, got {object_pose_valid.shape}"
                        )
                    invalid = np.flatnonzero(~object_pose_valid)
                    if len(invalid):
                        raise ValueError(
                            f"{self.path.name} has {len(invalid)} invalid cube_pose "
                            "samples required for object tracking"
                        )
            if valid_key in data and not np.all(data[valid_key]):
                invalid = np.flatnonzero(~np.asarray(data[valid_key], dtype=bool))
                raise ValueError(
                    f"{self.path.name} has {len(invalid)} invalid {hand_source} hand samples"
                )

        count = len(timestamps)
        expected = {
            "arm_q": (count, 6),
            "arm_dq": (count, 6),
            "ee_pose_measured": (count, 7),
            hand_key: (count, 20),
        }
        for name, (array, shape) in {
            "arm_q": (arm_q, expected["arm_q"]),
            "arm_dq": (arm_dq, expected["arm_dq"]),
            "ee_pose_measured": (ee_pose, expected["ee_pose_measured"]),
            hand_key: (hand_q, expected[hand_key]),
        }.items():
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"Invalid {name}: expected finite {shape}, got {array.shape}")
        if object_pose is not None and (
            object_pose.shape != (count, 7)
            or not np.all(np.isfinite(object_pose))
        ):
            raise ValueError(
                f"Invalid cube_pose: expected finite {(count, 7)}, "
                f"got {object_pose.shape}"
            )
        if (
            count < 2
            or derivative_timestamps.shape != (count,)
            or not np.all(np.isfinite(derivative_timestamps))
            or not np.all(np.diff(derivative_timestamps) > 0)
        ):
            raise ValueError(
                "Demonstration derivative timestamps must be finite and "
                "strictly increasing"
            )
        if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
            raise ValueError("Demonstration timestamps must be finite and strictly increasing")

        relative_time = timestamps - timestamps[0]
        derivative_time = derivative_timestamps - derivative_timestamps[0]
        self.duration_s = float(relative_time[-1])
        if self.duration_s <= 0:
            raise ValueError("Demonstration duration must be positive")

        # Express the physical UR-base EE pose in the simulation world.
        yaw = np.deg2rad(world_yaw_offset_deg)
        world_quat = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        ee_quat = _normalize_quaternions(ee_pose[:, 3:7])
        sim_ee_quat = _quat_multiply_xyzw(
            np.broadcast_to(world_quat, ee_quat.shape), ee_quat
        )
        palm_local_quat = _normalize_quaternions(
            np.asarray(ee_to_palm_quat_xyzw, dtype=np.float32)[None]
        )[0]
        palm_quat = _normalize_quaternions(
            _quat_multiply_xyzw(
                sim_ee_quat, np.broadcast_to(palm_local_quat, sim_ee_quat.shape)
            )
        )
        c, s = np.cos(yaw), np.sin(yaw)
        world_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        sim_ee_pos = (
            ee_pose[:, :3] @ world_rotation.T
            + np.asarray(world_position_offset_m, dtype=np.float32)
        )
        palm_offset = np.asarray(ee_to_palm_offset_m, dtype=np.float32)
        palm_pos = sim_ee_pos + _rotate_xyzw(
            sim_ee_quat, np.broadcast_to(palm_offset, sim_ee_pos.shape)
        )

        object_pos = None
        object_quat = None
        object_lin_vel = None
        object_ang_vel = None
        if object_pose is not None:
            object_position_offset = np.asarray(
                object_position_offset_m, dtype=np.float32
            )
            if object_position_offset.shape != (3,):
                raise ValueError(
                    "object_position_offset_m must contain exactly 3 values"
                )
            object_quat_base = _normalize_quaternions(object_pose[:, 3:7])
            object_orientation_offset = np.asarray(
                object_orientation_offset_xyzw, dtype=np.float32
            )
            if object_orientation_offset.shape != (4,):
                raise ValueError(
                    "object_orientation_offset_xyzw must contain exactly 4 values"
                )
            object_orientation_offset = _normalize_quaternions(
                object_orientation_offset[None]
            )[0]
            object_quat_world = _quat_multiply_xyzw(
                np.broadcast_to(world_quat, object_quat_base.shape),
                object_quat_base,
            )
            object_quat = _normalize_quaternions(
                _quat_multiply_xyzw(
                    object_quat_world,
                    np.broadcast_to(
                        object_orientation_offset, object_quat_world.shape
                    ),
                )
            )
            object_pos = (
                object_pose[:, :3] @ world_rotation.T
                + np.asarray(world_position_offset_m, dtype=np.float32)
                + object_position_offset
            )
            object_lin_vel = _filtered_derivative(
                object_pos,
                derivative_time,
                self.velocity_filter_window_s,
            )
            object_ang_vel = _quaternion_angular_velocity_xyzw(
                object_quat,
                derivative_time,
                self.velocity_filter_window_s,
            )

        hand_dq = _filtered_derivative(
            hand_q,
            derivative_time,
            self.velocity_filter_window_s,
        )
        palm_lin_vel = _filtered_derivative(
            palm_pos,
            derivative_time,
            self.velocity_filter_window_s,
        )
        palm_ang_vel = _quaternion_angular_velocity_xyzw(
            palm_quat,
            derivative_time,
            self.velocity_filter_window_s,
        )
        self.time = torch.as_tensor(relative_time, dtype=torch.float32, device=device)
        self.arm_q = torch.as_tensor(arm_q, device=device)
        self.arm_dq = torch.as_tensor(arm_dq, device=device)
        self.hand_q = torch.as_tensor(hand_q, device=device)
        self.hand_dq = torch.as_tensor(hand_dq, device=device)
        self.palm_pos = torch.as_tensor(palm_pos, dtype=torch.float32, device=device)
        self.palm_quat = torch.as_tensor(palm_quat, dtype=torch.float32, device=device)
        self.palm_lin_vel = torch.as_tensor(
            palm_lin_vel, dtype=torch.float32, device=device
        )
        self.palm_ang_vel = torch.as_tensor(
            palm_ang_vel, dtype=torch.float32, device=device
        )
        self.object_pos = (
            torch.as_tensor(object_pos, dtype=torch.float32, device=device)
            if object_pos is not None
            else None
        )
        self.object_quat = (
            torch.as_tensor(object_quat, dtype=torch.float32, device=device)
            if object_quat is not None
            else None
        )
        self.object_lin_vel = (
            torch.as_tensor(object_lin_vel, dtype=torch.float32, device=device)
            if object_lin_vel is not None
            else None
        )
        self.object_ang_vel = (
            torch.as_tensor(object_ang_vel, dtype=torch.float32, device=device)
            if object_ang_vel is not None
            else None
        )

    def recompute_hand_velocity(self) -> None:
        """Refresh filtered hand velocities after joint-limit clamping."""
        hand_q = self.hand_q.detach().cpu().numpy()
        timestamps = self.time.detach().cpu().numpy().astype(np.float64)
        hand_dq = _filtered_derivative(
            hand_q,
            timestamps,
            self.velocity_filter_window_s,
        )
        self.hand_dq.copy_(
            torch.as_tensor(hand_dq, dtype=self.hand_dq.dtype, device=self.hand_dq.device)
        )

    @staticmethod
    def _slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = torch.abs(dot).clamp(max=1.0)
        linear = dot > 0.9995
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta).clamp_min(1e-8)
        result = (
            torch.sin((1.0 - alpha) * theta) / sin_theta * q0
            + torch.sin(alpha * theta) / sin_theta * q1
        )
        result = torch.where(linear, (1.0 - alpha) * q0 + alpha * q1, result)
        return torch.nn.functional.normalize(result, dim=-1)

    def sample(self, phase: torch.Tensor) -> ReferenceSample:
        phase = phase.clamp(0.0, 1.0)
        query_time = phase * self.duration_s
        upper = torch.searchsorted(self.time, query_time, right=True)
        upper = upper.clamp(1, len(self.time) - 1)
        lower = upper - 1
        alpha = (
            (query_time - self.time[lower])
            / (self.time[upper] - self.time[lower]).clamp_min(1e-8)
        ).unsqueeze(-1)

        def lerp(values: torch.Tensor) -> torch.Tensor:
            return torch.lerp(values[lower], values[upper], alpha)

        return ReferenceSample(
            arm_q=lerp(self.arm_q),
            arm_dq=lerp(self.arm_dq),
            hand_q=lerp(self.hand_q),
            hand_dq=lerp(self.hand_dq),
            palm_pos=lerp(self.palm_pos),
            palm_quat_xyzw=self._slerp(
                self.palm_quat[lower], self.palm_quat[upper], alpha
            ),
            palm_lin_vel=lerp(self.palm_lin_vel),
            palm_ang_vel=lerp(self.palm_ang_vel),
            object_pos=(
                lerp(self.object_pos) if self.object_pos is not None else None
            ),
            object_quat_xyzw=(
                self._slerp(
                    self.object_quat[lower], self.object_quat[upper], alpha
                )
                if self.object_quat is not None
                else None
            ),
            object_lin_vel=(
                lerp(self.object_lin_vel)
                if self.object_lin_vel is not None
                else None
            ),
            object_ang_vel=(
                lerp(self.object_ang_vel)
                if self.object_ang_vel is not None
                else None
            ),
        )
