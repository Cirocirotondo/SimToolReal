"""Batched reference-motion loading and interpolation for imitation learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEMONSTRATION_DIRECTORY = (
    REPO_ROOT / "deployment" / "teleoperation" / "demonstrations"
)


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


@dataclass(frozen=True)
class ReferenceSample:
    arm_q: torch.Tensor
    arm_dq: torch.Tensor
    hand_q: torch.Tensor
    hand_dq: torch.Tensor
    palm_pos: torch.Tensor
    palm_quat_xyzw: torch.Tensor


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
    ) -> None:
        self.path = resolve_demonstration(path_or_name)
        with np.load(self.path, allow_pickle=False) as data:
            arm_q = np.asarray(data["arm_q"], dtype=np.float32)
            arm_dq = np.asarray(data["arm_dq"], dtype=np.float32)
            ee_pose = np.asarray(data["ee_pose_measured"], dtype=np.float32)
            hand_key = f"hand_q_{hand_source}"
            valid_key = f"{hand_key}_valid"
            hand_q = np.asarray(data[hand_key], dtype=np.float32)
            timestamps = np.asarray(data["timestamp"], dtype=np.float64)
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
        if count < 2 or not np.all(np.diff(timestamps) > 0):
            raise ValueError("Demonstration timestamps must be finite and strictly increasing")

        relative_time = timestamps - timestamps[0]
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

        hand_dq = np.gradient(hand_q, relative_time, axis=0).astype(np.float32)
        self.time = torch.as_tensor(relative_time, dtype=torch.float32, device=device)
        self.arm_q = torch.as_tensor(arm_q, device=device)
        self.arm_dq = torch.as_tensor(arm_dq, device=device)
        self.hand_q = torch.as_tensor(hand_q, device=device)
        self.hand_dq = torch.as_tensor(hand_dq, device=device)
        self.palm_pos = torch.as_tensor(palm_pos, dtype=torch.float32, device=device)
        self.palm_quat = torch.as_tensor(palm_quat, dtype=torch.float32, device=device)

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
        )
