#!/usr/bin/env python3
"""Plot the joint trajectories of a recorded teleoperation demonstration."""

from pathlib import Path
import sys
from typing import Dict, Tuple
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DEMO_PATH = (
    HERE
    / "demonstrations_good"
    / "demo_20260727_152551_335339.npz"
)

ARM_JOINT_LABELS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]
FINGER_LABELS = ["thumb", "index", "middle", "ring", "pinky"]
ARM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TARGET_HZ = 60.0
GAUSSIAN_RESAMPLE_SIGMA_S = 0.025
HAND_NEGATIVE_CLAMP_THRESHOLD_RAD = -0.04  # 0.04 rad = 2.3 deg
Demo = Dict[str, np.ndarray]
ROBOT_URDF_PATH = (
    HERE.parents[1]
    / "assets"
    / "urdf"
    / "ur5e_delto_description"
    / "ur5e_right_dg5f.urdf"
)
ARM_URDF_JOINT_NAMES = [f"{label}_joint" for label in ARM_JOINT_LABELS]
HAND_URDF_JOINT_NAMES = [
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]


def plot_graphs(
    demo: Demo,
    times: np.ndarray,
    new_times: np.ndarray,
    arm_q_60hz: np.ndarray,
    arm_dq_60hz: np.ndarray,
    hand_q_60hz: np.ndarray,
    hand_dq_60hz: np.ndarray,
    path: Path,
) -> None:
    """Create all original and resampled plots, then show them together."""
    original_figure, (position_axes, velocity_axes, hand_axes) = plt.subplots(
        3, 1, figsize=(13.0, 9.0), sharex=True
    )

    for index, label in enumerate(ARM_JOINT_LABELS):
        position_axes.plot(
            times, demo["arm_q"][:, index], label=label, color=ARM_COLORS[index]
        )
        velocity_axes.plot(times, demo["arm_dq"][:, index], color=ARM_COLORS[index])
    position_axes.set_ylabel("arm_q [rad]")
    position_axes.legend(ncol=6, fontsize=8)
    velocity_axes.set_ylabel("arm_dq [rad/s]")

    # One color per finger; its four joints share it.
    for joint in range(20):
        hand_axes.plot(
            times,
            demo["hand_q_measured"][:, joint],
            color=ARM_COLORS[joint // 4],
            label=FINGER_LABELS[joint // 4] if joint % 4 == 0 else None,
        )
    hand_axes.set_ylabel("hand_q_measured [rad]")
    hand_axes.set_xlabel("time [s]")
    hand_axes.legend(ncol=5, fontsize=8)

    original_figure.suptitle(f"{path.stem} — original")
    original_figure.tight_layout()

    resampled_figure, (
        arm_60hz_axes,
        arm_velocity_60hz_axes,
        hand_60hz_axes,
        hand_velocity_60hz_axes,
    ) = plt.subplots(
        4, 1, figsize=(13.0, 10.0), sharex=True
    )

    for index, label in enumerate(ARM_JOINT_LABELS):
        arm_60hz_axes.plot(
            new_times,
            arm_q_60hz[:, index],
            label=label,
            color=ARM_COLORS[index],
        )
    arm_60hz_axes.set_ylabel("arm_q at 60 Hz [rad]")
    arm_60hz_axes.legend(ncol=6, fontsize=8)

    for index in range(len(ARM_JOINT_LABELS)):
        arm_velocity_60hz_axes.plot(
            new_times,
            arm_dq_60hz[:, index],
            color=ARM_COLORS[index],
        )
    arm_velocity_60hz_axes.set_ylabel("arm_dq at 60 Hz [rad/s]")

    for joint in range(20):
        hand_60hz_axes.plot(
            new_times,
            hand_q_60hz[:, joint],
            color=ARM_COLORS[joint // 4],
            label=FINGER_LABELS[joint // 4] if joint % 4 == 0 else None,
        )
    hand_60hz_axes.set_ylabel("hand_q_measured at 60 Hz [rad]")
    hand_60hz_axes.legend(ncol=5, fontsize=8)

    for joint in range(20):
        hand_velocity_60hz_axes.plot(
            new_times,
            hand_dq_60hz[:, joint],
            color=ARM_COLORS[joint // 4],
        )
    hand_velocity_60hz_axes.set_ylabel("hand_dq at 60 Hz [rad/s]")
    hand_velocity_60hz_axes.set_xlabel("time [s]")

    resampled_figure.suptitle(
        f"{path.stem} — interpolated at {TARGET_HZ:g} Hz and smoothed"
    )
    resampled_figure.tight_layout()

    plt.show()

def check_validities(demo: Demo) -> None:
    """Check that the recorded joint trajectories are valid."""
    if not np.all(np.isfinite(demo["arm_q"])):
        raise ValueError("Some arm_q values are not finite.")
    if not np.all(np.isfinite(demo["arm_dq"])):
        raise ValueError("Some arm_dq values are not finite.")
    if not np.all(demo["hand_q_measured_valid"]):
        raise ValueError("Some hand_q_measured values are invalid.")
    if not np.all(np.isfinite(demo["hand_q_measured"])):
        raise ValueError("Some hand_q_measured values are not finite.")

def repair_hand_trajectory(demo: Demo) -> None:
    """Drop the stale first frame the recorder captures before the hand reports.

    Sample 0 sits 8-14 deg away from sample 1 on several joints, against a 1.8 deg
    p99 elsewhere, and a centered filter would smear it over the episode start.
    """
    demo["hand_q_measured"][0] = demo["hand_q_measured"][1]


def check_boundaries(demo: Demo) -> None:
    """Raise if a recorded joint position or velocity exceeds its URDF limit. For the joint limits, an error of HAND_NEGATIVE_CLAMP_THRESHOLD_RAD (0.04 rad) is allowed for the hand joints, to account for the fact that the hand can be slightly below its zero position when fully open."""
    root = ET.parse(ROBOT_URDF_PATH).getroot()
    urdf_joints = {
        joint.get("name"): joint
        for joint in root.findall("joint")
        if joint.get("name") is not None
    }
    joint_names = ARM_URDF_JOINT_NAMES + HAND_URDF_JOINT_NAMES
    missing_joints = [name for name in joint_names if name not in urdf_joints]
    if missing_joints:
        raise ValueError(
            f"URDF {ROBOT_URDF_PATH} is missing joints: {missing_joints}"
        )

    position_limits = []
    velocity_limits = []
    for name in joint_names:
        limit = urdf_joints[name].find("limit")
        if limit is None:
            raise ValueError(f"URDF joint {name!r} has no <limit> element")
        try:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
            velocity = float(limit.attrib["velocity"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"URDF joint {name!r} has invalid limits") from error
        if lower > upper or velocity < 0.0:
            raise ValueError(f"URDF joint {name!r} has invalid limits")
        position_limits.append((lower, upper))
        velocity_limits.append(velocity)

    position_limits_array = np.asarray(position_limits, dtype=np.float64)
    lower_limits = position_limits_array[:, 0]
    upper_limits = position_limits_array[:, 1]
    velocity_limits_array = np.asarray(velocity_limits, dtype=np.float64)

    checks = [
        (
            "arm_q",
            np.asarray(demo["arm_q"], dtype=np.float64),
            ARM_URDF_JOINT_NAMES,
            lower_limits[: len(ARM_URDF_JOINT_NAMES)],
            upper_limits[: len(ARM_URDF_JOINT_NAMES)],
        ),
        (
            "hand_q_measured",
            np.asarray(demo["hand_q_measured"], dtype=np.float64),
            HAND_URDF_JOINT_NAMES,
            lower_limits[len(ARM_URDF_JOINT_NAMES) :],
            upper_limits[len(ARM_URDF_JOINT_NAMES) :],
        ),
        (
            "arm_dq",
            np.asarray(demo["arm_dq"], dtype=np.float64),
            ARM_URDF_JOINT_NAMES,
            -velocity_limits_array[: len(ARM_URDF_JOINT_NAMES)],
            velocity_limits_array[: len(ARM_URDF_JOINT_NAMES)],
        ),
    ]

    errors = []
    for field, values, names, lower, upper in checks:
        expected_shape = (len(demo["timestamp"]), len(names))
        if values.shape != expected_shape:
            errors.append(
                f"{field} has shape {values.shape}, expected {expected_shape}"
            )
            continue

        if field in {"hand_q_measured"}:
            zero_lower_limit = np.isclose(lower, 0.0)
            small_negative = (
                (values < 0.0)
                & (values > HAND_NEGATIVE_CLAMP_THRESHOLD_RAD)
                & zero_lower_limit
            )
            values[small_negative] = 0.0
            demo[field][...] = values

        violations = ~np.isfinite(values) | (values < lower) | (values > upper)
        if not np.any(violations):
            continue

        sample_index, joint_index = np.argwhere(violations)[0]
        value = values[sample_index, joint_index]
        errors.append(
            f"{field}: {np.count_nonzero(violations)} values outside the URDF "
            f"limits; first violation at sample {sample_index}, joint "
            f"{names[joint_index]!r}: {value:g} not in "
            f"[{lower[joint_index]:g}, {upper[joint_index]:g}]"
        )

    if errors:
        raise ValueError("Joint boundary check failed:\n- " + "\n- ".join(errors))


def gaussian_resample(
    values: np.ndarray, times: np.ndarray, new_times: np.ndarray, sigma_s: float
) -> np.ndarray:
    """Smooth and resample in one step, weighting the true recorded timestamps.

    Filtering on the original grid and interpolating afterwards would leave the
    linear-interpolation corners; interpolating first would make the filter
    remove artifacts it created itself. Evaluating the kernel directly at the
    new times is the limit of the first order and avoids both.
    """
    weights = np.exp(-0.5 * ((new_times[:, None] - times[None, :]) / sigma_s) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ values


def compute_joint_velocities(
    joint_positions: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Differentiate joint positions with respect to their timestamps."""
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    if joint_positions.ndim != 2:
        raise ValueError(
            "joint_positions must have shape (samples, joints), "
            f"got {joint_positions.shape}"
        )
    if times.shape != (len(joint_positions),):
        raise ValueError(
            f"times has shape {times.shape}, expected ({len(joint_positions)},)"
        )
    if len(times) < 3:
        raise ValueError("At least three samples are required to compute velocities")
    if not np.all(np.isfinite(joint_positions)) or not np.all(np.isfinite(times)):
        raise ValueError("Joint positions and timestamps must be finite")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("Timestamps must be strictly increasing")

    return np.gradient(joint_positions, times, axis=0, edge_order=2)


def save_processed_demo(
    demo: Demo,
    new_times: np.ndarray,
    arm_q_60hz: np.ndarray,
    arm_dq_60hz: np.ndarray,
    hand_q_60hz: np.ndarray,
    hand_dq_60hz: np.ndarray,
    path: Path,
) -> Path:
    """Save the processed 60 Hz joint trajectories without altering the source."""
    new_times = np.asarray(new_times, dtype=np.float64)
    sample_count = len(new_times)
    arrays = {
        "timestamp": float(demo["timestamp"][0]) + new_times,
        "arm_q": np.asarray(arm_q_60hz, dtype=np.float64),
        "arm_dq": np.asarray(arm_dq_60hz, dtype=np.float64),
        "hand_q_measured": np.asarray(hand_q_60hz, dtype=np.float64),
        "hand_dq_measured": np.asarray(hand_dq_60hz, dtype=np.float64),
    }
    if "monotonic_timestamp" in demo:
        arrays["monotonic_timestamp"] = (
            float(demo["monotonic_timestamp"][0]) + new_times
        )

    expected_shapes = {
        "timestamp": (sample_count,),
        "arm_q": (sample_count, 6),
        "arm_dq": (sample_count, 6),
        "hand_q_measured": (sample_count, 20),
        "hand_dq_measured": (sample_count, 20),
    }
    if "monotonic_timestamp" in arrays:
        expected_shapes["monotonic_timestamp"] = (sample_count,)

    for name, expected_shape in expected_shapes.items():
        values = arrays[name]
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {values.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")

    if sample_count < 2 or not np.all(np.diff(new_times) > 0.0):
        raise ValueError("new_times must contain at least two increasing timestamps")

    output_path = path.with_name(f"{path.stem}_processed_60hz.npz")
    np.savez_compressed(output_path, **arrays)
    print(f"Saved processed demonstration: {output_path}")
    return output_path


def from_50hz_to_60hz(
    demo: Demo, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate arm and hand joint positions at 60 Hz."""
    duration_s = float(times[-1])
    sample_count = int(np.floor(duration_s * TARGET_HZ)) + 1
    new_times = np.arange(sample_count, dtype=np.float64) / TARGET_HZ

    arm_q_60hz = gaussian_resample(
        demo["arm_q"],
        times,
        new_times,
        2 * GAUSSIAN_RESAMPLE_SIGMA_S,
    )
    hand_q_60hz = gaussian_resample(
        demo["hand_q_measured"],
        times,
        new_times,
        2 * GAUSSIAN_RESAMPLE_SIGMA_S,
    )
    return new_times, arm_q_60hz, hand_q_60hz

def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DEMO_PATH
    with np.load(path) as archive:
        demo = {key: archive[key] for key in archive.files}
    times = demo["timestamp"] - demo["timestamp"][0]

    check_validities(demo)
    repair_hand_trajectory(demo)
    check_boundaries(demo)

    new_times, arm_q_60hz, hand_q_60hz = from_50hz_to_60hz(demo, times)

    arm_dq_60hz = compute_joint_velocities(arm_q_60hz, new_times)
    hand_dq_60hz = compute_joint_velocities(hand_q_60hz, new_times)

    plot_graphs(
        demo,
        times,
        new_times,
        arm_q_60hz,
        arm_dq_60hz,
        hand_q_60hz,
        hand_dq_60hz,
        path,
    )

    save_processed_demo(demo, new_times, arm_q_60hz, arm_dq_60hz, hand_q_60hz, hand_dq_60hz, path)


if __name__ == "__main__":
    main()
