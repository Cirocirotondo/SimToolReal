#!/usr/bin/env python3
"""Replay a recorded teleoperation demonstration in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.mujoco_ur5e_delto.mujoco_sim import (  # noqa: E402
    Ur5eDeltoMujocoConfig,
    Ur5eDeltoMujocoSim,
)


DEFAULT_DEMO_DIRECTORY = HERE / "demonstrations"
# MuJoCo local XYZ dimensions. The 15 cm axis is horizontal in the tracked pose.
DEFAULT_OBJECT_SIZE_M = np.array([0.15, 0.05, 0.05])
DEFAULT_HAND_MOUNT_YAW_OFFSET_DEG = 60.0


def resolve_demo(argument: str) -> tuple[Path, Path | None]:
    """Resolve a demo name, stem, .npz path, or metadata .json path."""
    supplied = Path(argument).expanduser()
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.append(DEFAULT_DEMO_DIRECTORY / supplied)

    for candidate in candidates:
        suffix = candidate.suffix.lower()
        data_path = candidate.with_suffix(".npz") if suffix == ".json" else candidate
        if data_path.suffix.lower() != ".npz":
            data_path = data_path.with_suffix(".npz")
        if data_path.is_file():
            metadata_path = data_path.with_suffix(".json")
            return data_path.resolve(), (
                metadata_path.resolve() if metadata_path.is_file() else None
            )

    raise FileNotFoundError(
        f"Could not find demonstration {argument!r}. Pass a demo name or a "
        f".npz/.json path (default directory: {DEFAULT_DEMO_DIRECTORY})."
    )


def load_metadata(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def recording_scene_parameters(metadata: dict) -> tuple[np.ndarray, float]:
    """Read scene information, with defaults for schema-v1 recordings."""
    object_metadata = metadata.get("object", {})
    object_size = np.asarray(
        object_metadata.get(
            "size_m",
            metadata.get("object_size_m", DEFAULT_OBJECT_SIZE_M),
        ),
        dtype=float,
    )
    if object_size.shape != (3,) or not np.all(object_size > 0):
        raise ValueError(f"Invalid object size in metadata: {object_size}")
    hand_offset = float(
        metadata.get(
            "hand_mount_yaw_offset_deg",
            DEFAULT_HAND_MOUNT_YAW_OFFSET_DEG,
        )
    )
    return object_size, hand_offset


def validate_recording(arrays: np.lib.npyio.NpzFile) -> int:
    if "arm_q" not in arrays:
        raise ValueError("Recording has no 'arm_q' trajectory")
    arm_q = arrays["arm_q"]
    if arm_q.ndim != 2 or arm_q.shape[1] != 6 or len(arm_q) == 0:
        raise ValueError(f"Expected arm_q shape (N, 6), got {arm_q.shape}")
    if "timestamp" in arrays and arrays["timestamp"].shape != (len(arm_q),):
        raise ValueError("timestamp and arm_q have different sample counts")
    return len(arm_q)


def sample_times(arrays: np.lib.npyio.NpzFile, count: int, fallback_hz: float) -> np.ndarray:
    if "timestamp" in arrays:
        times = np.asarray(arrays["timestamp"], dtype=float)
        if np.all(np.isfinite(times)) and np.all(np.diff(times) >= 0):
            return times - times[0]
    return np.arange(count, dtype=float) / fallback_hz


def hand_trajectory(
    arrays: np.lib.npyio.NpzFile, count: int, source: str
) -> tuple[np.ndarray, np.ndarray]:
    key = f"hand_q_{source}"
    valid_key = f"{key}_valid"
    if key not in arrays:
        return np.zeros((count, 20)), np.zeros(count, dtype=bool)
    hand_q = np.asarray(arrays[key], dtype=float)
    if hand_q.shape != (count, 20):
        raise ValueError(f"Expected {key} shape ({count}, 20), got {hand_q.shape}")
    valid = (
        np.asarray(arrays[valid_key], dtype=bool)
        if valid_key in arrays
        else np.all(np.isfinite(hand_q), axis=1)
    )
    return hand_q, valid & np.all(np.isfinite(hand_q), axis=1)


def ur_pose_to_combined_model(pose_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert UR-base pose to the grasp-designer model's world frame."""
    position = pose_xyzw[:3] * np.array([-1.0, -1.0, 1.0])
    quat_ur_wxyz = pose_xyzw[[6, 3, 4, 5]]
    quat_model_wxyz = np.empty(4, dtype=float)
    # The combined URDF faces the opposite x/y direction: R_model = Rz(pi) R_ur.
    mujoco.mju_mulQuat(
        quat_model_wxyz,
        np.array([0.0, 0.0, 0.0, 1.0]),
        quat_ur_wxyz,
    )
    return position, quat_model_wxyz


def apply_hand_mount_yaw_offset(
    sim: Ur5eDeltoMujocoSim, offset_deg: float
) -> None:
    """Rotate only the DG5F mount relative to the UR wrist."""
    mount_id = sim.model.body("rl_dg_mount").id
    original = sim.model.body_quat[mount_id].copy()
    half_angle = math.radians(offset_deg) / 2.0
    yaw_quaternion = np.array(
        [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
    )
    mujoco.mju_mulQuat(sim.model.body_quat[mount_id], original, yaw_quaternion)
    mujoco.mj_forward(sim.model, sim.data)


def configure_visual_scene(sim: Ur5eDeltoMujocoSim) -> None:
    """Configure non-physical background geometry for trajectory replay."""
    table_geom = sim.model.geom("table_geom")
    table_geom.contype = 0
    table_geom.conaffinity = 0
    floor_geom = sim.model.geom("floor")
    floor_geom.contype = 0
    floor_geom.conaffinity = 0


def replay(
    data_path: Path,
    metadata: dict,
    *,
    speed: float,
    loop: bool,
    fallback_hz: float,
    hand_source: str,
) -> None:
    object_size_m, hand_mount_offset_deg = recording_scene_parameters(metadata)
    with np.load(data_path, allow_pickle=False) as arrays:
        count = validate_recording(arrays)
        arm_q = np.asarray(arrays["arm_q"], dtype=float)
        hand_q, hand_valid = hand_trajectory(arrays, count, hand_source)
        relative_times = sample_times(arrays, count, fallback_hz)
        cube_pose = arrays["cube_pose"] if "cube_pose" in arrays else None
        cube_valid = arrays["cube_pose_valid"] if "cube_pose_valid" in arrays else None

        print(
            f"Playing {data_path.name}: {count} samples, "
            f"{relative_times[-1]:.2f} s at {speed:g}x "
            f"(right DG5F: {hand_source})"
        )
        sim = Ur5eDeltoMujocoSim(
            Ur5eDeltoMujocoConfig(
                enable_viewer=False,
                hand_side="right",
                floor_z=-0.20,
                # The table is 0.30 m thick, so this puts its top at z=-0.03 m.
                table_center_z=-0.18,
                workspace_y=-0.6,
                show_goal_marker=False,
                show_object_frame=True,
                object_name="cuboid",
                object_scales=object_size_m / 0.04,
                initial_joint_pos=np.concatenate([arm_q[0], np.zeros(20)]),
            )
        )
        apply_hand_mount_yaw_offset(sim, hand_mount_offset_deg)
        configure_visual_scene(sim)
        robot_q = np.concatenate([arm_q[0], np.zeros(20)])
        try:
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                while viewer.is_running():
                    started = time.monotonic()
                    for index in range(count):
                        if not viewer.is_running():
                            return
                        deadline = started + relative_times[index] / speed
                        while viewer.is_running() and time.monotonic() < deadline:
                            time.sleep(min(0.002, deadline - time.monotonic()))

                        robot_q[:6] = arm_q[index]
                        if hand_valid[index]:
                            robot_q[6:] = hand_q[index]
                        sim.set_robot_joint_positions(robot_q)

                        valid_cube = (
                            cube_pose is not None
                            and (cube_valid is None or bool(cube_valid[index]))
                        )
                        if valid_cube:
                            pose = np.asarray(cube_pose[index], dtype=float)
                            if pose.shape == (7,) and np.all(np.isfinite(pose)):
                                position, quaternion = ur_pose_to_combined_model(pose)
                                sim.set_object_pose(position, quaternion)
                        viewer.sync()
                    if not loop:
                        while viewer.is_running():
                            viewer.sync()
                            time.sleep(0.02)
                        return
        finally:
            sim.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a saved teleoperation demonstration in MuJoCo."
    )
    parser.add_argument(
        "demo",
        help="demo name/stem, or path to its .npz or .json file",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="playback speed (default: 1)"
    )
    parser.add_argument("--loop", action="store_true", help="repeat until closed")
    parser.add_argument(
        "--fallback-hz",
        type=float,
        default=50.0,
        help="rate used if recording timestamps are invalid (default: 50)",
    )
    parser.add_argument(
        "--hand-source",
        choices=("measured", "commanded"),
        default="measured",
        help="hand trajectory to display (default: measured)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="validate and describe the demo without opening a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0 or args.fallback_hz <= 0:
        raise SystemExit("--speed and --fallback-hz must be greater than zero")

    data_path, metadata_path = resolve_demo(args.demo)
    metadata = load_metadata(metadata_path)
    object_size_m, hand_offset_deg = recording_scene_parameters(metadata)
    if args.info:
        with np.load(data_path, allow_pickle=False) as arrays:
            count = validate_recording(arrays)
            times = sample_times(arrays, count, args.fallback_hz)
            _, hand_valid = hand_trajectory(arrays, count, args.hand_source)
        print(f"Demonstration: {data_path}")
        print(f"Samples:       {count}")
        print(f"Duration:      {times[-1]:.3f} s")
        print("MuJoCo model:  UR5e + right Tesollo DG5F")
        print(
            f"Valid hand:    {np.count_nonzero(hand_valid)}/{count} "
            f"({args.hand_source})"
        )
        print(f"Object size:   {object_size_m.tolist()} m")
        print(f"Hand offset:   {hand_offset_deg:g} deg")
        return

    replay(
        data_path,
        metadata,
        speed=args.speed,
        loop=args.loop,
        fallback_hz=args.fallback_hz,
        hand_source=args.hand_source,
    )


if __name__ == "__main__":
    main()
