#!/usr/bin/env python3
"""Interactively plot the joint trajectories stored in a demonstration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib


def _configure_qt_environment() -> None:
    """Use the PyQt5 plugins from this venv, not the ROS/Gazebo ones."""
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

    # QT_PLUGIN_PATH can point to a system/ROS Qt installation, which is not
    # ABI-compatible with the PyQt5 binary installed in this virtualenv.
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

    # The default user configuration directory is not writable in some shared
    # environments.  This avoids a cache warning without touching user files.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-simtoolreal")


_configure_qt_environment()
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DEMO_PATH = (
    HERE
    / "demonstrations_good"
    / "demo_20260727_152551_335339_60hz.npz"
)

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


def load_joint_trajectory(
    path: Path, hand_source: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load timestamps and the 26 joint positions/velocities from a demo archive."""
    if not path.is_file():
        raise FileNotFoundError(f"Demo file does not exist: {path}")

    hand_pos_key = f"hand_q_{hand_source}"
    hand_vel_key = f"hand_dq_{hand_source}"
    with np.load(path, allow_pickle=False) as demo:
        required_keys = {"timestamp", "arm_q", "arm_dq", hand_pos_key}
        missing = required_keys.difference(demo.files)
        if missing:
            available = ", ".join(demo.files)
            raise KeyError(
                f"{path} is missing {sorted(missing)}. Available fields: {available}"
            )

        timestamps = np.asarray(demo["timestamp"], dtype=np.float64)
        arm_q = np.asarray(demo["arm_q"], dtype=np.float64)
        arm_dq = np.asarray(demo["arm_dq"], dtype=np.float64)
        hand_q = np.asarray(demo[hand_pos_key], dtype=np.float64)
        hand_dq = (
            np.asarray(demo[hand_vel_key], dtype=np.float64)
            if hand_vel_key in demo.files
            else np.gradient(hand_q, timestamps, axis=0)
        )

    if timestamps.ndim != 1:
        raise ValueError(f"timestamp must be one-dimensional, got {timestamps.shape}")
    if arm_q.shape != (len(timestamps), len(ARM_JOINT_NAMES)):
        raise ValueError(f"arm_q has unexpected shape {arm_q.shape}")
    if hand_q.shape != (len(timestamps), len(HAND_JOINT_NAMES)):
        raise ValueError(f"{hand_pos_key} has unexpected shape {hand_q.shape}")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(arm_q)) or not np.all(np.isfinite(hand_q)):
        raise ValueError("The selected trajectory contains non-finite values.")

    # Demos may use an absolute wall-clock timestamp. Plot elapsed time instead.
    elapsed_time_s = timestamps - timestamps[0]
    positions = np.concatenate((arm_q, hand_q), axis=1)
    velocities = np.concatenate((arm_dq, hand_dq), axis=1)
    return elapsed_time_s, positions, velocities


def create_interactive_plot(
    times_s: np.ndarray,
    joint_data: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    """Open an interactive plot with a checkbox for every robot joint."""
    figure, axes = plt.subplots(figsize=(13, 7))
    figure.subplots_adjust(left=0.10, right=0.71, bottom=0.11, top=0.91)

    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(JOINT_NAMES)))
    lines = []
    for index, joint_name in enumerate(JOINT_NAMES):
        (line,) = axes.plot(
            times_s,
            joint_data[:, index],
            label=joint_name,
            color=colors[index],
            linewidth=1.3,
        )
        lines.append(line)

    axes.set_title(title)
    axes.set_xlabel("Time [s]")
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.35)
    axes.legend(loc="upper left", fontsize="small", ncol=2)

    checkbox_axes = figure.add_axes((0.74, 0.11, 0.24, 0.80))
    checkboxes = CheckButtons(checkbox_axes, JOINT_NAMES, [True] * len(JOINT_NAMES))
    checkbox_axes.set_title("Visible joints", fontsize="medium")

    def refresh_legend() -> None:
        visible_lines = [line for line in lines if line.get_visible()]
        if axes.legend_ is not None:
            axes.legend_.remove()
        if visible_lines:
            axes.legend(handles=visible_lines, loc="upper left", fontsize="small", ncol=2)

    def refresh_y_limits() -> None:
        """Fit the vertical scale to the currently visible trajectories."""
        visible_lines = [line for line in lines if line.get_visible()]
        if not visible_lines:
            return

        values = np.concatenate([line.get_ydata() for line in visible_lines])
        lower = float(np.min(values))
        upper = float(np.max(values))
        span = upper - lower
        padding = max(0.02, 0.05 * span)
        axes.set_ylim(lower - padding, upper + padding)

    def toggle_joint(joint_name: str) -> None:
        index = JOINT_NAMES.index(joint_name)
        lines[index].set_visible(not lines[index].get_visible())
        refresh_legend()
        refresh_y_limits()
        figure.canvas.draw_idle()

    checkboxes.on_clicked(toggle_joint)

    select_all_axes = figure.add_axes((0.74, 0.025, 0.11, 0.05))
    clear_all_axes = figure.add_axes((0.87, 0.025, 0.11, 0.05))
    select_all_button = Button(select_all_axes, "All")
    clear_all_button = Button(clear_all_axes, "None")

    def set_all_visible(visible: bool) -> None:
        for index, line in enumerate(lines):
            if line.get_visible() != visible:
                checkboxes.set_active(index)
        refresh_legend()
        refresh_y_limits()
        figure.canvas.draw_idle()

    select_all_button.on_clicked(lambda _event: set_all_visible(True))
    clear_all_button.on_clicked(lambda _event: set_all_visible(False))
    refresh_y_limits()
    # Keep references alive for the lifetime of the window (matplotlib drops
    # widgets with no surviving reference, silently breaking their callbacks).
    figure._widget_refs = (checkboxes, select_all_button, clear_all_button)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a demo's arm and hand joint positions interactively."
    )
    parser.add_argument(
        "demo_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DEMO_PATH,
        help=f"Demo .npz file. Default: {DEFAULT_DEMO_PATH}",
    )
    parser.add_argument(
        "--hand-source",
        choices=("measured", "commanded"),
        default="measured",
        help="Hand trajectory field to plot (default: measured).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times_s, positions, velocities = load_joint_trajectory(args.demo_path, args.hand_source)
    create_interactive_plot(
        times_s,
        positions,
        title=f"Joint positions — {args.demo_path.name}",
        ylabel="Joint position [rad]",
    )
    create_interactive_plot(
        times_s,
        velocities,
        title=f"Joint velocities — {args.demo_path.name}",
        ylabel="Joint velocity [rad/s]",
    )
    plt.show()


if __name__ == "__main__":
    main()
