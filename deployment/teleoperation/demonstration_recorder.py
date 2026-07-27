from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


HAND_DOF = 20


def matrix_to_pose_xyzw(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    pose = np.empty(7, dtype=np.float64)
    pose[:3] = transform[:3, 3]
    pose[3:] = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return pose


def robot_state_pose_xyzw(raw_pose) -> np.ndarray:
    """Normalize UR state poses to [x, y, z, qx, qy, qz, qw]."""
    raw_pose = np.asarray(raw_pose, dtype=np.float64)
    if raw_pose.shape == (7,):
        return raw_pose.copy()
    if raw_pose.shape == (6,):
        return np.concatenate(
            [raw_pose[:3], Rotation.from_rotvec(raw_pose[3:]).as_quat()]
        )
    return np.full(7, np.nan, dtype=np.float64)


class HandBridgeProcess:
    def __init__(
        self,
        *,
        script: Path,
        ros_setup: Path,
        workspace_setup: Path,
        host: str,
        port: int,
        state_topic: str,
        command_topic: str,
    ) -> None:
        command = (
            f"source {ros_setup} && "
            f"source {workspace_setup} && "
            f"exec python3 {script} "
            f"--host {host} --port {port} "
            f"--state-topic {state_topic} "
            f"--command-topic {command_topic}"
        )
        self.command = command
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            ["/bin/bash", "-lc", self.command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.0)


class HandStateStream:
    def __init__(self, host: str, port: int) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind((host, port))
        self.latest: Optional[dict] = None
        self.received_at: Optional[float] = None

    def poll(self) -> Optional[dict]:
        while True:
            try:
                payload, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                return self.latest
            try:
                message = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            self.latest = message
            self.received_at = time.monotonic()

    def close(self) -> None:
        self.socket.close()


class DemonstrationRecorder:
    """Collect fixed-shape control samples and save one compressed episode."""

    def __init__(
        self,
        *,
        output_directory: Path,
        metadata: dict,
    ) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.started_wall = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        episode_id = self.started_wall.strftime("%Y%m%d_%H%M%S_%f")
        self.stem = self.output_directory / f"demo_{episode_id}"
        self.metadata = metadata
        self.samples: dict[str, list] = {
            "timestamp": [],
            "monotonic_timestamp": [],
            "arm_q": [],
            "arm_dq": [],
            "ee_pose_measured": [],
            "ee_pose_commanded": [],
            "hand_q_measured": [],
            "hand_q_commanded": [],
            "hand_q_measured_valid": [],
            "hand_q_commanded_valid": [],
            "cube_pose": [],
            "cube_pose_valid": [],
            "cube_pose_confidence": [],
            "episode_done": [],
            "episode_success": [],
        }
        self.saved = False

    def append(
        self,
        *,
        arm_q: np.ndarray,
        arm_dq: np.ndarray,
        ee_pose_measured: np.ndarray,
        ee_pose_commanded: np.ndarray,
        hand_message: Optional[dict],
        hand_valid: bool,
        cube_pose: np.ndarray,
        cube_valid: bool,
        cube_confidence: float,
    ) -> None:
        hand_measured = np.full(HAND_DOF, np.nan, dtype=np.float64)
        hand_commanded = np.full(HAND_DOF, np.nan, dtype=np.float64)
        measured_valid = False
        commanded_valid = False
        if hand_message is not None and hand_valid:
            measured = np.asarray(
                hand_message.get("hand_q_measured", []),
                dtype=np.float64,
            )
            commanded = np.asarray(
                hand_message.get("hand_q_commanded", []),
                dtype=np.float64,
            )
            if measured.shape == (HAND_DOF,):
                hand_measured = measured
                measured_valid = bool(
                    hand_message.get("hand_q_measured_valid", False)
                )
            if commanded.shape == (HAND_DOF,):
                hand_commanded = commanded
                commanded_valid = bool(
                    hand_message.get("hand_q_commanded_valid", False)
                )

        values = {
            "timestamp": time.time(),
            "monotonic_timestamp": time.monotonic(),
            "arm_q": np.asarray(arm_q, dtype=np.float64),
            "arm_dq": np.asarray(arm_dq, dtype=np.float64),
            "ee_pose_measured": np.asarray(
                ee_pose_measured, dtype=np.float64
            ),
            "ee_pose_commanded": np.asarray(
                ee_pose_commanded, dtype=np.float64
            ),
            "hand_q_measured": hand_measured,
            "hand_q_commanded": hand_commanded,
            "hand_q_measured_valid": measured_valid,
            "hand_q_commanded_valid": commanded_valid,
            "cube_pose": np.asarray(cube_pose, dtype=np.float64),
            "cube_pose_valid": bool(cube_valid),
            "cube_pose_confidence": float(cube_confidence),
            "episode_done": False,
            "episode_success": False,
        }
        for key, value in values.items():
            self.samples[key].append(value)

    def finalize(
        self,
        *,
        success: bool,
        termination_reason: str,
    ) -> Optional[tuple[Path, Path]]:
        if self.saved:
            return None
        self.saved = True
        if not self.samples["timestamp"]:
            return None

        self.samples["episode_done"][-1] = True
        self.samples["episode_success"][-1] = bool(success)
        arrays = {
            key: np.asarray(values)
            for key, values in self.samples.items()
        }

        data_path = self.stem.with_suffix(".npz")
        temporary_data_path = data_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_data_path, **arrays)
        os.replace(temporary_data_path, data_path)

        ended_wall = datetime.now(timezone.utc)
        metadata = {
            **self.metadata,
            "episode_id": self.stem.name,
            "started_at_utc": self.started_wall.isoformat(),
            "ended_at_utc": ended_wall.isoformat(),
            "duration_s": time.monotonic() - self.started_monotonic,
            "sample_count": len(self.samples["timestamp"]),
            "episode_success": bool(success),
            "termination_reason": termination_reason,
            "data_file": data_path.name,
            "schema_version": 1,
            "units": {
                "joint_position": "rad",
                "joint_velocity": "rad/s",
                "position": "m",
                "quaternion": "xyzw",
                "timestamp": "s",
            },
            "coordinate_frames": {
                "ee_pose_measured": "ur5_base",
                "ee_pose_commanded": "ur5_base/model_base",
                "cube_pose": "ur5_base",
            },
        }
        metadata_path = self.stem.with_suffix(".json")
        temporary_metadata_path = metadata_path.with_suffix(".tmp")
        with temporary_metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_metadata_path, metadata_path)
        return data_path, metadata_path
