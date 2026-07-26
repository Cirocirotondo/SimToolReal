#!/usr/bin/env python3
"""Show the camera-observed MANUS frame without starting robot control."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_PYTHON = Path(
    "/home/duplo/git/robohand-robohand2/.venv/bin/python"
)
DEFAULT_PROJECT = Path(
    "/home/duplo/git/robohand/src/tag-pose-estimation"
)
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / (
    "manus_overhead_pose_estimation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only MANUS-board pose estimation and show its configured "
            "MAPPED frame in the OpenCV camera window."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Pose-estimation configuration file.",
    )
    parser.add_argument(
        "--pose-python",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Python interpreter containing RealSense and OpenCV dependencies.",
    )
    parser.add_argument(
        "--pose-project",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Root of the tag-pose-estimation project.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Do not resolve this symlink: executing the resolved /usr/bin/python
    # would bypass the virtual environment containing zmq and RealSense.
    python = args.pose_python.expanduser().absolute()
    project = args.pose_project.expanduser().resolve()
    config = args.config.expanduser().resolve()
    script = project / "scripts" / "run_pose_estimation.py"

    for path, description in (
        (python, "pose-estimation Python interpreter"),
        (script, "pose-estimation script"),
        (config, "pose-estimation configuration"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")

    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(project) + (
        os.pathsep + current_pythonpath if current_pythonpath else ""
    )

    print("Starting MANUS frame viewer.")
    print("The thin axes show the board frame.")
    print("The thick X', Y', Z' axes labeled MAPPED show the hand frame.")
    print("Press q in the OpenCV window or Ctrl+C in this terminal to stop.")
    os.chdir(project)
    os.execve(
        python,
        [
            str(python),
            str(script),
            "--config",
            str(config),
        ],
        environment,
    )


if __name__ == "__main__":
    main()
