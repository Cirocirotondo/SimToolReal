#!/usr/bin/env python3
"""Generate a 60 Hz diagnostic demonstration for the arm and enabled hand joints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

try:
    from .generate_arm_joint_test_demo import (
        ARM_JOINT_NAMES,
        DEFAULT_SOURCE,
        ROBOT_URDF,
        TARGET_HZ,
        _append_hold,
        _append_minimum_jerk_transition,
        _load_arm_limits,
        generate_arm_trajectory,
    )
except ImportError:
    from generate_arm_joint_test_demo import (
        ARM_JOINT_NAMES,
        DEFAULT_SOURCE,
        ROBOT_URDF,
        TARGET_HZ,
        _append_hold,
        _append_minimum_jerk_transition,
        _load_arm_limits,
        generate_arm_trajectory,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE
    / "demonstrations_good"
    / "demo_synthetic_arm_hand_all_joints_60hz.npz"
)
HAND_JOINT_NAMES = tuple(
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
DISABLED_HAND_JOINT_NAMES = {
    "rj_dg_2_1",
    "rj_dg_3_1",
    "rj_dg_4_1",
    "rj_dg_5_2",
}
HAND_AMPLITUDE_FRACTION = 0.4
HAND_MAX_AMPLITUDE_RAD = 0.4
HAND_MIN_DIRECTION_RANGE_RAD = 0.1
HAND_TRANSITION_DURATION_S = 0.5
FINAL_HOLD_DURATION_S = 1.0


def _load_hand_limits() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ET.parse(ROBOT_URDF).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    lower = []
    upper = []
    velocity = []
    for name in HAND_JOINT_NAMES:
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


def generate_hand_trajectory() -> Tuple[np.ndarray, np.ndarray]:
    """Excite each enabled hand joint independently from the neutral open pose."""
    lower, upper, _velocity = _load_hand_limits()
    transition_intervals = round(HAND_TRANSITION_DURATION_S * TARGET_HZ)
    if transition_intervals < 2:
        raise ValueError("Hand transition duration is too short")

    neutral = np.zeros(len(HAND_JOINT_NAMES), dtype=np.float64)
    positions = []  # type: List[np.ndarray]
    velocities = []  # type: List[np.ndarray]
    current = neutral.copy()

    for joint_index in range(len(HAND_JOINT_NAMES)):
        if HAND_JOINT_NAMES[joint_index] in DISABLED_HAND_JOINT_NAMES:
            continue
        offsets = []
        positive_range = upper[joint_index]
        negative_range = -lower[joint_index]
        if positive_range >= HAND_MIN_DIRECTION_RANGE_RAD:
            offsets.append(
                min(
                    HAND_MAX_AMPLITUDE_RAD,
                    HAND_AMPLITUDE_FRACTION * positive_range,
                )
            )
            offsets.append(0.0)
        if negative_range >= HAND_MIN_DIRECTION_RANGE_RAD:
            offsets.append(
                -min(
                    HAND_MAX_AMPLITUDE_RAD,
                    HAND_AMPLITUDE_FRACTION * negative_range,
                )
            )
            offsets.append(0.0)
        if not offsets:
            raise ValueError(
                f"No safe test motion available for {HAND_JOINT_NAMES[joint_index]}"
            )

        for offset in offsets:
            waypoint = neutral.copy()
            waypoint[joint_index] = offset
            _append_minimum_jerk_transition(
                positions,
                velocities,
                current,
                waypoint,
                transition_intervals,
            )
            current = waypoint

    _append_hold(
        positions,
        velocities,
        neutral,
        round(FINAL_HOLD_DURATION_S * TARGET_HZ),
    )
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(velocities, dtype=np.float64),
    )


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
        required = {"timestamp", "monotonic_timestamp", "arm_q"}
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(f"{source.name} is missing fields: {missing}")
        initial_arm_q = np.asarray(arrays["arm_q"][0], dtype=np.float64)
        timestamp_start = float(arrays["timestamp"][0])
        monotonic_start = float(arrays["monotonic_timestamp"][0])

    _arm_times, arm_q_test, arm_dq_test = generate_arm_trajectory(initial_arm_q)
    hand_q_test, hand_dq_test = generate_hand_trajectory()

    arm_hold = np.repeat(initial_arm_q[None, :], len(hand_q_test), axis=0)
    arm_q = np.concatenate((arm_q_test, arm_hold), axis=0)
    arm_dq = np.concatenate((arm_dq_test, np.zeros_like(arm_hold)), axis=0)
    hand_hold = np.zeros((len(arm_q_test), len(HAND_JOINT_NAMES)), dtype=np.float64)
    hand_q = np.concatenate((hand_hold, hand_q_test), axis=0)
    hand_dq = np.concatenate((np.zeros_like(hand_hold), hand_dq_test), axis=0)
    times_s = np.arange(len(arm_q), dtype=np.float64) / TARGET_HZ

    arm_lower, arm_upper, arm_velocity_limit = _load_arm_limits()
    hand_lower, hand_upper, hand_velocity_limit = _load_hand_limits()
    if np.any(arm_q < arm_lower) or np.any(arm_q > arm_upper):
        raise ValueError("Generated arm positions exceed the URDF limits")
    if np.any(hand_q < hand_lower) or np.any(hand_q > hand_upper):
        raise ValueError("Generated hand positions exceed the URDF limits")
    if np.any(np.abs(arm_dq) > arm_velocity_limit):
        raise ValueError("Generated arm velocities exceed the URDF limits")
    if np.any(np.abs(hand_dq) > hand_velocity_limit):
        raise ValueError("Generated hand velocities exceed the URDF limits")

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
