#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Optional

import numpy as np

from ik_solver import Ur5DampedLeastSquaresIk
from pose_stream import BoardPoseStream, WristPoseFilter
from robot_io import CommandStreamer, RobotIo
from transforms import (
    RelativeWristMapper,
    average_transforms,
    limit_pose_step,
    load_transform,
)


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    for section in ("tracking", "robot", "mapping", "filter", "ik", "safety"):
        if section not in config:
            raise ValueError(f"Configuration is missing section {section!r}")
    return config


def wait_for_wrist_capture(
    stream: BoardPoseStream,
    pose_filter: WristPoseFilter,
    *,
    board_from_wrist: np.ndarray,
    sample_count: int,
    timeout_s: float,
    maximum_position_std_m: float,
) -> np.ndarray:
    print(
        f"Collecting {sample_count} fresh wrist samples. "
        "Hold the MANUS wrist still at the chosen initial position..."
    )
    deadline = time.monotonic() + timeout_s
    transforms: list[np.ndarray] = []
    last_received_at: Optional[float] = None
    while time.monotonic() < deadline and len(transforms) < sample_count:
        sample = stream.poll()
        if sample is None or sample.received_at == last_received_at:
            time.sleep(0.005)
            continue
        last_received_at = sample.received_at
        world_from_wrist = sample.transform @ board_from_wrist
        filtered = pose_filter.update(world_from_wrist)
        if filtered is not None:
            transforms.append(filtered)
        time.sleep(0.002)
    if len(transforms) < sample_count:
        raise TimeoutError(
            f"Only received {len(transforms)}/{sample_count} valid wrist poses"
        )

    positions = np.array([transform[:3, 3] for transform in transforms])
    maximum_std = float(np.max(np.std(positions, axis=0)))
    if maximum_std > maximum_position_std_m:
        raise RuntimeError(
            "Initial wrist was not stable: maximum coordinate standard "
            f"deviation was {maximum_std * 1000.0:.1f} mm, limit is "
            f"{maximum_position_std_m * 1000.0:.1f} mm"
        )
    print(
        f"Initial wrist capture accepted; maximum position std: "
        f"{maximum_std * 1000.0:.2f} mm"
    )
    return average_transforms(transforms)


def wait_for_robot_state(
    robot: RobotIo,
    timeout_s: float,
) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = robot.poll_state()
        if state is not None:
            return state
        time.sleep(0.01)
    raise TimeoutError("No UR5 state received from the low-level controller")


def check_workspace(
    target_model: np.ndarray,
    rotation_model_from_robot_base: np.ndarray,
    minimum_base: np.ndarray,
    maximum_base: np.ndarray,
) -> None:
    position_base = (
        rotation_model_from_robot_base.T @ target_model[:3, 3]
    )
    if np.any(position_base < minimum_base) or np.any(
        position_base > maximum_base
    ):
        raise RuntimeError(
            "Requested wrist target left the configured workspace: "
            f"base position={position_base.round(4).tolist()}, "
            f"minimum={minimum_base.tolist()}, maximum={maximum_base.tolist()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track the camera-observed MANUS wrist and control only the UR5e "
            "arm. Finger control remains in the official Tesollo ROS pipeline."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "arm_teleop.json",
    )
    parser.add_argument(
        "--send-to-robot",
        action="store_true",
        help="Enable physical UR5 commands; default is camera + IK dry-run.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive capture/physical-command confirmations.",
    )
    parser.add_argument("--max-runtime", type=float, default=0.0)
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Track translation while holding the initial robot orientation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    tracking = config["tracking"]
    robot_config = config["robot"]
    mapping = config["mapping"]
    filtering = config["filter"]
    ik_config = config["ik"]
    safety = config["safety"]

    model_path = resolve_path(config_path, robot_config["model_path"])
    low_level_config_path = resolve_path(
        config_path, robot_config["low_level_config_path"]
    )
    wrist_calibration_path = resolve_path(
        config_path, tracking["wrist_calibration_file"]
    )
    robot_base_calibration_path = resolve_path(
        config_path, mapping["world_from_robot_base_file"]
    )

    board_from_wrist = load_transform(
        wrist_calibration_path,
        "T_board_from_wrist",
        "T_board_wrist",
    )
    world_from_robot_base = load_transform(robot_base_calibration_path)
    rotation_robot_base_from_world = world_from_robot_base[:3, :3].T
    rotation_model_from_robot_base = np.asarray(
        mapping["rotation_model_from_robot_base"], dtype=np.float64
    )
    rotation_model_from_world = (
        rotation_model_from_robot_base @ rotation_robot_base_from_world
    )

    ik = Ur5DampedLeastSquaresIk(
        model_path=model_path,
        end_effector_body=robot_config["end_effector_body"],
        damping=ik_config["damping"],
        position_gain=ik_config["position_gain"],
        orientation_gain=ik_config["orientation_gain"],
        maximum_joint_velocity_rad_s=ik_config[
            "maximum_joint_velocity_rad_s"
        ],
    )
    home_q = np.asarray(robot_config["home_q_rad"], dtype=np.float64)
    if home_q.shape != (6,):
        raise ValueError("robot.home_q_rad must contain six values")

    pose_stream = BoardPoseStream(
        address=tracking["pose_address"],
        board_id=str(tracking["wrist_board_id"]),
        minimum_confidence=tracking["minimum_confidence"],
    )
    pose_filter = WristPoseFilter(
        translation_alpha=filtering["translation_alpha"],
        rotation_alpha=filtering["rotation_alpha"],
        max_translation_jump_m=filtering["maximum_input_jump_m"],
        max_rotation_jump_rad=np.deg2rad(
            filtering["maximum_input_rotation_jump_deg"]
        ),
    )

    robot: Optional[RobotIo] = None
    streamer: Optional[CommandStreamer] = None
    last_state: Optional[dict] = None
    q_command = home_q.copy()

    try:
        if args.send_to_robot:
            robot = RobotIo(low_level_config_path)
            last_state = wait_for_robot_state(
                robot, safety["robot_state_start_timeout_s"]
            )
            q_measured = np.asarray(last_state["Q"][:6], dtype=np.float64)
            home_error = np.max(np.abs(q_measured - home_q))
            if home_error > np.deg2rad(safety["home_tolerance_deg"]):
                raise RuntimeError(
                    "Robot is not sufficiently close to the configured home "
                    f"pose: maximum error={np.rad2deg(home_error):.2f} deg, "
                    f"limit={safety['home_tolerance_deg']:.2f} deg"
                )
            q_command = q_measured.copy()
            print(
                "UR5 state ready at home; measured q deg: "
                f"{np.rad2deg(q_measured).round(2).tolist()}"
            )
        else:
            print("DRY RUN: no command socket will be opened.")

        initial_world_wrist = wait_for_wrist_capture(
            pose_stream,
            pose_filter,
            board_from_wrist=board_from_wrist,
            sample_count=tracking["initial_sample_count"],
            timeout_s=tracking["initial_capture_timeout_s"],
            maximum_position_std_m=tracking[
                "maximum_initial_position_std_m"
            ],
        )
        home_model_ee = ik.forward(q_command)
        wrist_mapper = RelativeWristMapper(
            initial_world_wrist=initial_world_wrist,
            home_model_ee=home_model_ee,
            rotation_model_from_world=rotation_model_from_world,
            position_scale=mapping["position_scale"],
            track_orientation=(
                mapping["track_orientation"] and not args.position_only
            ),
        )
        target_model = home_model_ee.copy()

        print("Initial MANUS wrist position now corresponds to UR5 home.")
        print(
            "Home end-effector position in MuJoCo model: "
            f"{home_model_ee[:3, 3].round(4).tolist()}"
        )
        if not args.yes:
            answer = input(
                "Type START to begin tracking"
                + (" and then SEND to enable the real arm: "
                   if args.send_to_robot else ": ")
            ).strip()
            required = "START SEND" if args.send_to_robot else "START"
            if answer != required:
                print(f"Cancelled; expected exactly {required!r}.")
                return

        if args.send_to_robot:
            assert robot is not None
            streamer = CommandStreamer(
                command_socket=robot.command_socket,
                frequency_hz=robot_config["command_stream_hz"],
            )
            streamer.set_target(q_command)
            streamer.start()
            print("REAL UR5 COMMANDS ENABLED. Finger commands are untouched.")

        control_hz = float(robot_config["control_hz"])
        dt = 1.0 / control_hz
        start_time = time.monotonic()
        next_step = start_time
        last_log = 0.0
        minimum_workspace = np.asarray(
            safety["workspace_min_robot_base_m"], dtype=np.float64
        )
        maximum_workspace = np.asarray(
            safety["workspace_max_robot_base_m"], dtype=np.float64
        )

        while True:
            now = time.monotonic()
            sample = pose_stream.poll()
            if sample is None or (
                now - sample.received_at > tracking["maximum_pose_age_s"]
            ):
                raise RuntimeError("MANUS wrist pose became stale or missing")

            world_from_wrist = sample.transform @ board_from_wrist
            filtered_wrist = pose_filter.update(world_from_wrist)
            if filtered_wrist is not None:
                requested_target = wrist_mapper.target(filtered_wrist)
                target_model = limit_pose_step(
                    target_model,
                    requested_target,
                    max_translation_step_m=(
                        safety["maximum_target_speed_m_s"] * dt
                    ),
                    max_rotation_step_rad=np.deg2rad(
                        safety["maximum_target_rotation_speed_deg_s"]
                    )
                    * dt,
                )

            check_workspace(
                target_model,
                rotation_model_from_robot_base,
                minimum_workspace,
                maximum_workspace,
            )
            q_command, diagnostics = ik.step(q_command, target_model, dt)

            tracking_error = 0.0
            if robot is not None:
                state = robot.poll_state()
                if state is not None:
                    last_state = state
                if (
                    last_state is None
                    or now - last_state["_received_at"]
                    > safety["maximum_robot_state_age_s"]
                ):
                    raise RuntimeError("UR5 state became stale")
                measured_q = np.asarray(last_state["Q"][:6], dtype=np.float64)
                tracking_error = float(
                    np.max(np.abs(q_command - measured_q))
                )
                if tracking_error > np.deg2rad(
                    safety["maximum_joint_tracking_error_deg"]
                ):
                    raise RuntimeError(
                        "UR5 joint tracking error exceeded safety limit: "
                        f"{np.rad2deg(tracking_error):.2f} deg"
                    )
                assert streamer is not None
                streamer.set_target(q_command)

            if now - last_log >= 1.0 / config["logging"]["print_hz"]:
                print(
                    f"target_xyz_model={target_model[:3, 3].round(4).tolist()} "
                    f"| pos_err={diagnostics.position_error_m * 1000:.1f} mm "
                    f"| rot_err={np.rad2deg(diagnostics.orientation_error_rad):.1f} deg "
                    f"| q_deg={np.rad2deg(q_command).round(1).tolist()} "
                    f"| tracking={np.rad2deg(tracking_error):.1f} deg "
                    f"| rejected_poses={pose_filter.rejected_count}"
                )
                last_log = now

            if args.max_runtime > 0.0 and now - start_time >= args.max_runtime:
                print(f"Reached --max-runtime={args.max_runtime:g} s.")
                break
            next_step += dt
            time.sleep(max(0.0, next_step - time.monotonic()))
            if next_step < time.monotonic() - dt:
                next_step = time.monotonic()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if streamer is not None:
            if robot is not None:
                state = robot.poll_state()
                if state is not None:
                    q_hold = np.asarray(state["Q"][:6], dtype=np.float64)
                    streamer.set_target(q_hold)
                    time.sleep(safety["shutdown_hold_s"])
            streamer.stop()
        if robot is not None:
            robot.close()
        pose_stream.close()


if __name__ == "__main__":
    main()
