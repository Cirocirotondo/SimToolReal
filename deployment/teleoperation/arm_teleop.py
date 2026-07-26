#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Optional

import numpy as np

from ik_solver import Ur5DampedLeastSquaresIk
from pose_stream import BoardPoseStream, WristPoseFilter
from robot_io import CommandStreamer, RobotIo
from transforms import (
    RelativeBoardMapper,
    apply_local_origin_offset,
    limit_pose_step,
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


class PoseEstimatorProcess:
    def __init__(
        self,
        *,
        python: Path,
        project_root: Path,
        script: Path,
        config: Path,
    ) -> None:
        self.python = python
        self.project_root = project_root
        self.script = script
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.output_thread: Optional[threading.Thread] = None

    def _forward_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for raw_line in self.process.stdout:
            line = raw_line.rstrip()
            if (
                line.startswith("Running at ")
                or (
                    line.startswith("Published ")
                    and line.endswith(" board poses")
                )
            ):
                continue
            if line:
                print(f"[pose estimator] {line}", flush=True)

    def start(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(self.project_root) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        command = [
            str(self.python),
            str(self.script),
            "--config",
            str(self.config),
        ]
        print("Starting overhead MANUS pose estimator:")
        print("  " + " ".join(command))
        self.process = subprocess.Popen(
            command,
            cwd=self.project_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.output_thread = threading.Thread(
            target=self._forward_output,
            name="pose-estimator-output",
            daemon=True,
        )
        self.output_thread.start()

    def check_running(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(
                "The automatically started pose estimator exited with code "
                f"{self.process.returncode}"
            )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        if self.output_thread is not None:
            self.output_thread.join(timeout=1.0)


def countdown_board_capture(
    stream: BoardPoseStream,
    pose_filter: WristPoseFilter,
    *,
    origin_offset_board_m: np.ndarray,
    countdown_s: int,
    timeout_s: float,
    pose_estimator: Optional[PoseEstimatorProcess],
) -> np.ndarray:
    print("Waiting for the camera-observed MANUS board...")
    deadline = time.monotonic() + timeout_s
    last_received_at: Optional[float] = None
    latest_board: Optional[np.ndarray] = None
    while time.monotonic() < deadline:
        if pose_estimator is not None:
            pose_estimator.check_running()
        sample = stream.poll()
        if sample is not None and sample.received_at != last_received_at:
            last_received_at = sample.received_at
            tracked_pose = apply_local_origin_offset(
                sample.transform,
                origin_offset_board_m,
            )
            filtered = pose_filter.update(tracked_pose)
            if filtered is not None:
                latest_board = filtered
                break
        time.sleep(0.01)
    if latest_board is None:
        raise TimeoutError("No valid MANUS board pose was received")

    print("MANUS board detected. Hold it at the desired neutral position.")
    for remaining in range(countdown_s, 0, -1):
        print(f"Mapping MANUS position to UR5 home in {remaining}...")
        interval_end = time.monotonic() + 1.0
        while time.monotonic() < interval_end:
            if pose_estimator is not None:
                pose_estimator.check_running()
            sample = stream.poll()
            if sample is not None and sample.received_at != last_received_at:
                last_received_at = sample.received_at
                tracked_pose = apply_local_origin_offset(
                    sample.transform,
                    origin_offset_board_m,
                )
                filtered = pose_filter.update(tracked_pose)
                if filtered is not None:
                    latest_board = filtered
            time.sleep(0.005)

    sample = stream.poll()
    if (
        sample is None
        or time.monotonic() - sample.received_at > 0.25
        or latest_board is None
    ):
        raise RuntimeError("MANUS board pose was not fresh at countdown end")
    print("GO: current MANUS board pose is now the UR5 home pose.")
    return latest_board


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


def move_robot_to_home(
    robot: RobotIo,
    streamer: CommandStreamer,
    home_q: np.ndarray,
    *,
    timeout_s: float,
    tolerance_deg: float,
    settle_s: float,
    maximum_state_age_s: float,
) -> dict:
    print("Moving UR5 to the configured home joint position...")
    print(f"  target q deg: {np.rad2deg(home_q).round(2).tolist()}")
    streamer.set_target(home_q)

    deadline = time.monotonic() + timeout_s
    reached_since: Optional[float] = None
    last_log = 0.0
    latest_state: Optional[dict] = None
    latest_errors_deg = np.full(6, np.inf)

    while True:
        now = time.monotonic()
        state = robot.poll_state()
        if state is not None:
            latest_state = state
        if (
            latest_state is None
            or now - latest_state["_received_at"] > maximum_state_age_s
        ):
            raise RuntimeError("UR5 state became stale while moving home")

        measured_q = np.asarray(latest_state["Q"][:6], dtype=np.float64)
        latest_errors_deg = np.rad2deg(np.abs(home_q - measured_q))
        maximum_error_deg = float(np.max(latest_errors_deg))
        if maximum_error_deg <= tolerance_deg:
            if reached_since is None:
                reached_since = now
            elif now - reached_since >= settle_s:
                print(
                    "UR5 home reached; measured q deg: "
                    f"{np.rad2deg(measured_q).round(2).tolist()}"
                )
                return latest_state
        else:
            reached_since = None

        if now - last_log >= 0.5:
            worst_joint = int(np.argmax(latest_errors_deg)) + 1
            print(
                "UR5 homing: "
                f"max error={maximum_error_deg:.2f} deg "
                f"at joint {worst_joint}"
            )
            last_log = now

        if now >= deadline:
            raise TimeoutError(
                f"UR5 did not reach home within {timeout_s:g} s; "
                f"joint errors={latest_errors_deg.round(2).tolist()} deg"
            )
        time.sleep(0.01)


def check_workspace(
    target_model: np.ndarray,
    minimum_model: np.ndarray,
    maximum_model: np.ndarray,
) -> None:
    position_model = target_model[:3, 3]
    if np.any(position_model < minimum_model) or np.any(
        position_model > maximum_model
    ):
        raise RuntimeError(
            "Requested wrist target left the configured workspace: "
            f"model position={position_model.round(4).tolist()}, "
            f"minimum={minimum_model.tolist()}, maximum={maximum_model.tolist()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track the camera-observed MANUS board and control only the UR5e "
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
        "--no-start-pose-estimator",
        action="store_true",
        help="Use an already running pose estimator instead of starting it.",
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
    translation_axis_map = np.asarray(
        mapping["translation_axis_map"], dtype=np.float64
    )
    orientation_axis_map = np.asarray(
        mapping["orientation_axis_map"], dtype=np.float64
    )
    spatial_orientation_axis_map = np.asarray(
        mapping.get("spatial_orientation_axis_map", np.eye(3)),
        dtype=np.float64,
    )
    origin_offset_board_m = np.asarray(
        mapping["tracking_origin_offset_board_m"],
        dtype=np.float64,
    )
    if origin_offset_board_m.shape != (3,):
        raise ValueError(
            "mapping.tracking_origin_offset_board_m must contain 3 values"
        )

    pose_estimator: Optional[PoseEstimatorProcess] = None
    if (
        tracking["auto_start_pose_estimator"]
        and not args.no_start_pose_estimator
    ):
        pose_estimator_config = resolve_path(
            config_path, tracking["pose_estimator_config"]
        )
        pose_estimator = PoseEstimatorProcess(
            python=Path(tracking["pose_estimator_python"]).expanduser(),
            project_root=Path(
                tracking["pose_estimator_project_root"]
            ).expanduser(),
            script=Path(tracking["pose_estimator_script"]).expanduser(),
            config=pose_estimator_config,
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
    pose_stream: Optional[BoardPoseStream] = None
    last_state: Optional[dict] = None
    q_command = home_q.copy()

    try:
        if args.send_to_robot:
            robot = RobotIo(low_level_config_path)
            last_state = wait_for_robot_state(
                robot, safety["robot_state_start_timeout_s"]
            )
            q_measured = np.asarray(last_state["Q"][:6], dtype=np.float64)
            q_command = q_measured.copy()
            streamer = CommandStreamer(
                command_socket=robot.command_socket,
                frequency_hz=robot_config["command_stream_hz"],
            )
            streamer.set_target(q_command)
            streamer.start()
            print(
                "UR5 state ready; initial measured q deg: "
                f"{np.rad2deg(q_measured).round(2).tolist()}"
            )
            # Allow the low-level controller's SUB socket to connect to this
            # command PUB while the measured pose is held.
            time.sleep(safety["command_connection_delay_s"])
            last_state = move_robot_to_home(
                robot,
                streamer,
                home_q,
                timeout_s=safety["home_timeout_s"],
                tolerance_deg=safety["home_tolerance_deg"],
                settle_s=safety["home_settle_s"],
                maximum_state_age_s=safety["maximum_robot_state_age_s"],
            )
            q_command = home_q.copy()
        else:
            print("DRY RUN: no command socket will be opened.")

        if pose_estimator is not None:
            pose_estimator.start()
        pose_stream = BoardPoseStream(
            address=tracking["pose_address"],
            board_id=str(tracking["wrist_board_id"]),
            minimum_confidence=tracking["minimum_confidence"],
        )

        initial_world_board = countdown_board_capture(
            pose_stream,
            pose_filter,
            origin_offset_board_m=origin_offset_board_m,
            countdown_s=int(tracking["countdown_s"]),
            timeout_s=tracking["initial_pose_timeout_s"],
            pose_estimator=pose_estimator,
        )
        home_model_ee = ik.forward(home_q)
        board_mapper = RelativeBoardMapper(
            initial_world_board=initial_world_board,
            home_model_ee=home_model_ee,
            translation_axis_map=translation_axis_map,
            orientation_axis_map=orientation_axis_map,
            spatial_orientation_axis_map=spatial_orientation_axis_map,
            orientation_mode=mapping.get(
                "orientation_mapping_mode",
                "mapped-local",
            ),
            position_scale=mapping["position_scale"],
            track_orientation=(
                mapping["track_orientation"] and not args.position_only
            ),
        )
        target_model = home_model_ee.copy()

        print("Initial MANUS board position now corresponds to UR5 home.")
        print(
            "Home end-effector position in MuJoCo model: "
            f"{home_model_ee[:3, 3].round(4).tolist()}"
        )
        print(
            "Translation axes: MANUS forward = world -X -> model +X; "
            "MANUS left = world +Y -> model -Y."
        )
        orientation_mode = mapping.get(
            "orientation_mapping_mode",
            "mapped-local",
        )
        print(f"Orientation mapping mode: {orientation_mode}")
        if orientation_mode == "spatial-relative":
            print(
                "Using R_target = R_current * R_initial^-1 * R_EE_home; "
                "orientation_axis_map is not used."
            )
            print("Spatial rotation-vector axis map:")
            print(spatial_orientation_axis_map)
        else:
            print("Orientation axis map:")
            print(orientation_axis_map)
        print(
            "Tracked origin offset in board frame: "
            f"{origin_offset_board_m.round(6).tolist()} m."
        )

        if args.send_to_robot:
            assert robot is not None and streamer is not None
            streamer.set_target(q_command)
            print(
                "REAL UR5 TELEOPERATION ENABLED. "
                "Finger commands are untouched."
            )

        control_hz = float(robot_config["control_hz"])
        dt = 1.0 / control_hz
        start_time = time.monotonic()
        next_step = start_time
        last_log = 0.0
        minimum_workspace = np.asarray(
            safety["workspace_min_model_m"], dtype=np.float64
        )
        maximum_workspace = np.asarray(
            safety["workspace_max_model_m"], dtype=np.float64
        )

        while True:
            now = time.monotonic()
            sample = pose_stream.poll()
            if sample is None or (
                now - sample.received_at > tracking["maximum_pose_age_s"]
            ):
                raise RuntimeError("MANUS board pose became stale or missing")

            tracked_pose = apply_local_origin_offset(
                sample.transform,
                origin_offset_board_m,
            )
            filtered_board = pose_filter.update(tracked_pose)
            if filtered_board is not None:
                requested_target = board_mapper.target(filtered_board)
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
                    f"| ik_qdot={diagnostics.maximum_joint_velocity_rad_s:.2f} rad/s "
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
            streamer.stop()
        if robot is not None:
            print("Requesting explicit UR5 speedStop...")
            robot.request_stop()
            robot.close()
        if pose_stream is not None:
            pose_stream.close()
        if pose_estimator is not None:
            pose_estimator.stop()


if __name__ == "__main__":
    main()
