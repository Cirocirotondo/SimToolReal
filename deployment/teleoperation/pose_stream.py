from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation
import zmq

from transforms import make_transform, pose_to_matrix


@dataclass(frozen=True)
class PoseSample:
    transform: np.ndarray
    confidence: float
    received_at: float
    publisher_timestamp: Optional[float]


class BoardPoseStream:
    """Receive the newest board pose published by tag-pose-estimation."""

    def __init__(
        self,
        *,
        address: str,
        board_id: str,
        minimum_confidence: float,
    ) -> None:
        self.board_id = str(board_id)
        self.minimum_confidence = float(minimum_confidence)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(address)
        self.latest: Optional[PoseSample] = None

    def poll(self) -> Optional[PoseSample]:
        while True:
            try:
                message = self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return self.latest

            poses = message.get("poses")
            if not isinstance(poses, dict) or self.board_id not in poses:
                continue
            raw_pose: Any = poses[self.board_id]
            confidence = (
                float(raw_pose.get("confidence", 1.0))
                if isinstance(raw_pose, dict)
                else 1.0
            )
            if confidence < self.minimum_confidence:
                continue
            try:
                transform = pose_to_matrix(raw_pose)
            except (TypeError, ValueError):
                continue
            self.latest = PoseSample(
                transform=transform,
                confidence=confidence,
                received_at=time.monotonic(),
                publisher_timestamp=message.get("timestamp"),
            )

    def close(self) -> None:
        self.socket.close()
        self.context.term()


class WristPoseFilter:
    """Low-pass filtering plus rejection of implausible camera jumps."""

    def __init__(
        self,
        *,
        translation_alpha: float,
        rotation_alpha: float,
        max_translation_jump_m: float,
        max_rotation_jump_rad: float,
    ) -> None:
        for name, value in (
            ("translation_alpha", translation_alpha),
            ("rotation_alpha", rotation_alpha),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        self.translation_alpha = float(translation_alpha)
        self.rotation_alpha = float(rotation_alpha)
        self.max_translation_jump_m = float(max_translation_jump_m)
        self.max_rotation_jump_rad = float(max_rotation_jump_rad)
        self.filtered: Optional[np.ndarray] = None
        self.rejected_count = 0

    def update(self, value: np.ndarray) -> Optional[np.ndarray]:
        value = np.asarray(value, dtype=np.float64)
        if self.filtered is None:
            self.filtered = value.copy()
            return self.filtered.copy()

        translation_jump = float(
            np.linalg.norm(value[:3, 3] - self.filtered[:3, 3])
        )
        rotation_delta = Rotation.from_matrix(
            self.filtered[:3, :3].T @ value[:3, :3]
        )
        rotation_jump = float(rotation_delta.magnitude())
        if (
            translation_jump > self.max_translation_jump_m
            or rotation_jump > self.max_rotation_jump_rad
        ):
            self.rejected_count += 1
            return None

        translation = (
            (1.0 - self.translation_alpha) * self.filtered[:3, 3]
            + self.translation_alpha * value[:3, 3]
        )
        incremental_rotation = Rotation.from_rotvec(
            self.rotation_alpha * rotation_delta.as_rotvec()
        )
        rotation = self.filtered[:3, :3] @ incremental_rotation.as_matrix()
        self.filtered = make_transform(rotation, translation)
        return self.filtered.copy()
