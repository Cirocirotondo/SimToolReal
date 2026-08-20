#!/usr/bin/env python3
"""Replay a recorded teleoperation demonstration in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def _configure_qt_environment() -> None:
    """Use the virtualenv's Qt plugins rather than incompatible ROS ones."""
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    kept_paths = [
        entry
        for entry in library_path.split(":")
        if "/opt/ros/" not in entry and "/gazebo" not in entry.lower()
    ]
    if kept_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(kept_paths)
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)

    os.environ.pop("QT_PLUGIN_PATH", None)
    pyqt_platform_plugins = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "PyQt5"
        / "Qt5"
        / "plugins"
        / "platforms"
    )
    if pyqt_platform_plugins.is_dir():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(pyqt_platform_plugins)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-simtoolreal")


_configure_qt_environment()

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
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
DEFAULT_DEMO_PATH = (
    HERE
    / "demonstrations_good"
    / "demo_20260727_152551_335339_60hz.npz"
)
# MuJoCo local XYZ dimensions. The 15 cm axis is horizontal in the tracked pose.
DEFAULT_OBJECT_SIZE_M = np.array([0.15, 0.05, 0.05])
DEFAULT_HAND_MOUNT_YAW_OFFSET_DEG = 60.0
ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HAND_JOINT_NAMES = tuple(
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES
# Body pairs whose collision meshes overlap by construction in the combined
# URDF, so their contacts are permanent and non-physical. MuJoCo's parent/child
# filter does not catch the first pair: base_link_inertia is welded to the world
# and a world parent is exempt from that filter, so the base mesh collides with
# the shoulder mesh across shoulder_pan_joint itself. The resulting constraint
# force cancels the position actuator almost exactly and freezes the joint.
NON_PHYSICAL_COLLISION_PAIRS = (
    ("base_link_inertia", "shoulder_link"),
    ("rl_dg_1_2", "rl_dg_palm"),
    ("rl_dg_4_2", "rl_dg_palm"),
)


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
    print(f"Using hand trajectory: {key} (valid: {valid_key})")
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
    object_geom = sim.model.geom("object_cube_geom")
    object_geom.contype = 0
    object_geom.conaffinity = 0
    object_geom.rgba = np.array([0.0, 0.0, 0.0, 0.0])


def disable_collision_pairs(
    sim: Ur5eDeltoMujocoSim,
    body_pairs: tuple[tuple[str, str], ...],
) -> None:
    """Stop the listed body pairs from colliding, leaving every other pair on.

    Two MuJoCo geoms are candidates for contact when contype and conaffinity
    overlap in either direction. Giving each listed body its own contype bit,
    then clearing that bit from its partners' conaffinity, suppresses exactly
    those pairs while keeping both bodies colliding with the rest of the scene.
    """
    model = sim.model
    all_bits = (1 << 31) - 1

    partners_by_body: dict[str, set[str]] = {}
    for body_a, body_b in body_pairs:
        partners_by_body.setdefault(body_a, set()).add(body_b)
        partners_by_body.setdefault(body_b, set()).add(body_a)

    body_bits = {
        name: 1 << (index + 1)
        for index, name in enumerate(sorted(partners_by_body))
    }
    for name, partners in partners_by_body.items():
        conaffinity = all_bits
        for partner in partners:
            conaffinity &= ~body_bits[partner]
        body_id = model.body(name).id
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                model.geom_contype[geom_id] = body_bits[name]
                model.geom_conaffinity[geom_id] = conaffinity


def plot_joint_tracking(
    times_s: np.ndarray,
    target_positions: np.ndarray,
    actual_positions: np.ndarray,
    *,
    title: str,
) -> None:
    """Show selectable commanded and measured MuJoCo joint trajectories."""
    expected_shape = (len(times_s), len(JOINT_NAMES))
    if target_positions.shape != expected_shape or actual_positions.shape != expected_shape:
        raise ValueError(
            f"Expected target and actual trajectories with shape {expected_shape}, got "
            f"{target_positions.shape} and {actual_positions.shape}"
        )

    figure, axes = plt.subplots(figsize=(13, 7))
    figure.subplots_adjust(left=0.10, right=0.71, bottom=0.11, top=0.91)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(JOINT_NAMES)))
    joint_lines: list[tuple[object, object]] = []

    for index, joint_name in enumerate(JOINT_NAMES):
        (target_line,) = axes.plot(
            times_s,
            target_positions[:, index],
            color=colors[index],
            linewidth=1.4,
            label=f"{joint_name} target",
        )
        (actual_line,) = axes.plot(
            times_s,
            actual_positions[:, index],
            color=colors[index],
            linewidth=1.2,
            linestyle="--",
            label=f"{joint_name} actual",
        )
        joint_lines.append((target_line, actual_line))

    axes.set_title(title)
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Joint position [rad]")
    axes.grid(True, alpha=0.35)

    checkbox_axes = figure.add_axes((0.74, 0.11, 0.24, 0.80))
    checkboxes = CheckButtons(checkbox_axes, JOINT_NAMES, [True] * len(JOINT_NAMES))
    checkbox_axes.set_title("Visible joints", fontsize="medium")

    def visible_lines() -> list[object]:
        return [
            line
            for target_line, actual_line in joint_lines
            if target_line.get_visible()
            for line in (target_line, actual_line)
        ]

    def refresh_legend() -> None:
        if axes.legend_ is not None:
            axes.legend_.remove()
        lines = visible_lines()
        if lines:
            axes.legend(handles=lines, loc="upper left", fontsize="x-small", ncol=2)

    def refresh_y_limits() -> None:
        lines = visible_lines()
        if not lines:
            return
        values = np.concatenate([line.get_ydata() for line in lines])
        lower = float(np.min(values))
        upper = float(np.max(values))
        padding = max(0.02, 0.05 * (upper - lower))
        axes.set_ylim(lower - padding, upper + padding)

    def toggle_joint(joint_name: str) -> None:
        index = JOINT_NAMES.index(joint_name)
        target_line, actual_line = joint_lines[index]
        visible = not target_line.get_visible()
        target_line.set_visible(visible)
        actual_line.set_visible(visible)
        refresh_legend()
        refresh_y_limits()
        figure.canvas.draw_idle()

    def set_all_visible(visible: bool) -> None:
        for index, (target_line, _actual_line) in enumerate(joint_lines):
            if target_line.get_visible() != visible:
                checkboxes.set_active(index)
        refresh_legend()
        refresh_y_limits()
        figure.canvas.draw_idle()

    checkboxes.on_clicked(toggle_joint)
    select_all_axes = figure.add_axes((0.74, 0.025, 0.11, 0.05))
    clear_all_axes = figure.add_axes((0.87, 0.025, 0.11, 0.05))
    # Buttons must be kept referenced: an unassigned Button() is garbage
    # collected immediately and its clicks then silently do nothing.
    select_all_button = Button(select_all_axes, "All")
    select_all_button.on_clicked(lambda _event: set_all_visible(True))
    clear_all_button = Button(clear_all_axes, "None")
    clear_all_button.on_clicked(lambda _event: set_all_visible(False))
    refresh_legend()
    refresh_y_limits()
    plt.show()


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
                show_object_frame=False,
                object_name="cuboid",
                object_scales=object_size_m / 0.04,
                initial_joint_pos=np.concatenate([arm_q[0], np.zeros(20)]),
            )
        )
        apply_hand_mount_yaw_offset(sim, hand_mount_offset_deg)
        configure_visual_scene(sim)
        # Match the Isaac Gym training/replay setup, which runs with gravity
        # disabled and relies purely on the PD position drive.
        sim.set_gravity_enabled(False)
        disable_collision_pairs(sim, NON_PHYSICAL_COLLISION_PAIRS)
        robot_q = np.concatenate([arm_q[0], np.zeros(20)])
        target_positions = np.empty((count, len(JOINT_NAMES)), dtype=float)
        actual_positions = np.empty((count, len(JOINT_NAMES)), dtype=float)
        recorded_count = 0
        try:
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                while viewer.is_running():
                    started = time.monotonic()
                    viewer_closed = False
                    for index in range(count):
                        if not viewer.is_running():
                            viewer_closed = True
                            break
                        deadline = started + relative_times[index] / speed
                        while viewer.is_running() and time.monotonic() < deadline:
                            time.sleep(min(0.002, deadline - time.monotonic()))
                        if not viewer.is_running():
                            viewer_closed = True
                            break

                        robot_q[:6] = arm_q[index]
                        if hand_valid[index]:
                            robot_q[6:] = hand_q[index]
                        target_positions[index] = robot_q
                        # Use MuJoCo's position actuators instead of writing qpos
                        # directly, so the logged state is the actual response.
                        sim.set_robot_joint_pos_targets(robot_q)
                        actual_positions[index] = sim.get_sim_state()[
                            "joint_positions"
                        ]
                        recorded_count = index + 1

                        valid_cube = (
                            cube_pose is not None
                            and (cube_valid is None or bool(cube_valid[index]))
                        )
                        if valid_cube:
                            pose = np.asarray(cube_pose[index], dtype=float)
                            if pose.shape == (7,) and np.all(np.isfinite(pose)):
                                position, quaternion = ur_pose_to_combined_model(pose)
                                sim.set_object_pose(position, quaternion)

                        if index + 1 < count:
                            sim.step_for(relative_times[index + 1] - relative_times[index])
                        viewer.sync()
                    if viewer_closed:
                        break
                    if not loop:
                        while viewer.is_running():
                            viewer.sync()
                            time.sleep(0.02)
        finally:
            sim.close()

        if recorded_count:
            plot_joint_tracking(
                relative_times[:recorded_count],
                target_positions[:recorded_count],
                actual_positions[:recorded_count],
                title=f"MuJoCo joint tracking — {data_path.name}",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a saved teleoperation demonstration in MuJoCo."
    )
    parser.add_argument(
        "demo",
        nargs="?",
        default=str(DEFAULT_DEMO_PATH),
        help=(
            "demo name/stem, or path to its .npz or .json file "
            f"(default: {DEFAULT_DEMO_PATH})"
        ),
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
