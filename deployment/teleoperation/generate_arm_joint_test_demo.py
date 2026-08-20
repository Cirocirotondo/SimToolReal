#!/usr/bin/env python3
"""Generate a safe 60 Hz demonstration that excites every UR5e joint.

Each arm joint moves independently through

    initial -> +amplitude -> initial -> -amplitude -> initial

using minimum-jerk transitions.  The hand remains in its neutral open pose so
that hand posture does not confound the arm-controller test.  The output
contains the same six fields as the processed demonstrations used by the
training and Isaac Gym viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_SOURCE = (
    HERE
    / "demonstrations_good"
    / "demo_20260727_152551_335339_60hz.npz"
)
DEFAULT_OUTPUT = (
    HERE
    / "demonstrations_good"
    / "demo_synthetic_arm_all_joints_60hz.npz"
)
ROBOT_URDF = (
    REPO_ROOT
    / "assets"
    / "urdf"
    / "ur5e_delto_description"
    / "ur5e_right_dg5f_mount_60deg.urdf"
)

TARGET_HZ = 60.0
ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# Conservative excursions around the initial pose, in ARM_JOINT_NAMES order.
ARM_AMPLITUDES_RAD = np.asarray(
    (0.30, 0.25, 0.30, 0.35, 0.40, 0.50), dtype=np.float64
)


def _load_arm_limits() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ET.parse(ROBOT_URDF).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    lower = []
    upper = []
    velocity = []
    for name in ARM_JOINT_NAMES:
        joint = joints.get(name)
        limit = None if joint is None else joint.find("limit")
        if limit is None:
            raise ValueError(f"Missing limits for {name!r} in {ROBOT_URDF}")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
        velocity.append(float(limit.attrib["velocity"]))
    return (
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(velocity, dtype=np.float64),
    )


def _append_hold(
    positions: List[np.ndarray],
    velocities: List[np.ndarray],
    pose: np.ndarray,
    interval_count: int,
) -> None:
    for _ in range(interval_count):
        positions.append(pose.copy())
        velocities.append(np.zeros_like(pose))


def _append_minimum_jerk_transition(
    positions: List[np.ndarray],
    velocities: List[np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    interval_count: int,
) -> None:
    duration_s = interval_count / TARGET_HZ
    delta = end - start
    for interval in range(1, interval_count + 1):
        phase = interval / interval_count
        blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        blend_rate = (
            30.0 * phase**2 - 60.0 * phase**3 + 30.0 * phase**4
        ) / duration_s
        positions.append(start + blend * delta)
        velocities.append(blend_rate * delta)


def generate_arm_trajectory(
    initial_arm_q: np.ndarray,
    *,
    hold_duration_s: float = 1.0,
    transition_duration_s: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return timestamps, positions and analytical velocities at exactly 60 Hz."""
    initial_arm_q = np.asarray(initial_arm_q, dtype=np.float64)
    if initial_arm_q.shape != (len(ARM_JOINT_NAMES),):
        raise ValueError(
            f"Expected initial arm pose shape ({len(ARM_JOINT_NAMES)},), "
            f"got {initial_arm_q.shape}"
        )

    hold_intervals = round(hold_duration_s * TARGET_HZ)
    transition_intervals = round(transition_duration_s * TARGET_HZ)
    if hold_intervals < 1 or transition_intervals < 2:
        raise ValueError("Hold and transition durations are too short")

    positions = [initial_arm_q.copy()]
    velocities = [np.zeros_like(initial_arm_q)]
    _append_hold(positions, velocities, initial_arm_q, hold_intervals)

    current = initial_arm_q.copy()
    for joint_index, amplitude in enumerate(ARM_AMPLITUDES_RAD):
        waypoints = []
        for offset in (amplitude, 0.0, -amplitude, 0.0):
            waypoint = initial_arm_q.copy()
            waypoint[joint_index] += offset
            waypoints.append(waypoint)
        for waypoint in waypoints:
            _append_minimum_jerk_transition(
                positions,
                velocities,
                current,
                waypoint,
                transition_intervals,
            )
            current = waypoint

    _append_hold(positions, velocities, initial_arm_q, hold_intervals)

    arm_q = np.asarray(positions, dtype=np.float64)
    arm_dq = np.asarray(velocities, dtype=np.float64)
    times_s = np.arange(len(arm_q), dtype=np.float64) / TARGET_HZ
    return times_s, arm_q, arm_dq


def generate_demo(source: Path, output: Path, *, overwrite: bool = False) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source demonstration not found: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --overwrite to replace it."
        )

    with np.load(source, allow_pickle=False) as arrays:
        required = {
            "timestamp",
            "monotonic_timestamp",
            "arm_q",
        }
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(f"{source.name} is missing fields: {missing}")
        initial_arm_q = np.asarray(arrays["arm_q"][0], dtype=np.float64)
        timestamp_start = float(arrays["timestamp"][0])
        monotonic_start = float(arrays["monotonic_timestamp"][0])

    times_s, arm_q, arm_dq = generate_arm_trajectory(initial_arm_q)
    hand_q = np.zeros((len(times_s), 20), dtype=np.float64)
    hand_dq = np.zeros_like(hand_q)

    lower, upper, velocity_limit = _load_arm_limits()
    if np.any(arm_q < lower) or np.any(arm_q > upper):
        raise ValueError("Generated arm positions exceed the URDF limits")
    if np.any(np.abs(arm_dq) > velocity_limit):
        raise ValueError("Generated arm velocities exceed the URDF limits")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        timestamp=timestamp_start + times_s,
        arm_q=arm_q,
        arm_dq=arm_dq,
        hand_q_measured=hand_q,
        hand_dq_measured=hand_dq,
        monotonic_timestamp=monotonic_start + times_s,
    )
    return output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output if it already exists.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output = generate_demo(args.source, args.output, overwrite=args.overwrite)
    with np.load(output, allow_pickle=False) as arrays:
        duration_s = float(
            arrays["monotonic_timestamp"][-1]
            - arrays["monotonic_timestamp"][0]
        )
        print(f"Saved {output}")
        print(f"Samples: {len(arrays['timestamp'])}")
        print(f"Duration: {duration_s:.3f} s")
        print(f"Frequency: {TARGET_HZ:.1f} Hz")


if __name__ == "__main__":
    main()
