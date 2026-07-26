from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zmq


HERE = Path(__file__).parent
SIMTOOLREAL_ROOT = Path(
    os.environ.get("SIMTOOLREAL_ROOT", HERE.parent.parent)
).expanduser()
REPO_ROOT = SIMTOOLREAL_ROOT
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "train_dir/simtoolreal/2026-06-10/train_07_sim2real_resume_resume_2026-06-10_15-00-42"
    / "runs/00_train_07_sim2real_resume_resume_2026-06-10_15-00-42"
)
RUN_DIR = Path(os.environ.get("SIMTOOLREAL_RUN_DIR", DEFAULT_RUN_DIR)).expanduser()
CONFIG_PATH = RUN_DIR / "config.yaml"
NN_DIR = RUN_DIR / "nn"
LOW_LEVEL_CONFIG_PATH = HERE / "pc_ur_new.json"

CONTROL_HZ = 60.0
POLICY_ACTION_HZ = 60.0
POLICY_ACTION_DT = 1.0 / POLICY_ACTION_HZ
STATE_TIMEOUT_S = 2.0
PRINT_PERIOD_S = 1.0
MAX_ARM_TARGET_ERROR_DEG = 10.0
ARM_DOF = 6
N_ACT = 26
N_OBS = 131
DEFAULT_POSE_ESTIMATION_ADDRESS = "tcp://127.0.0.1:5557"
DEFAULT_POSE_BOARD_ID = "0"
DEFAULT_POSE_TIMEOUT_S = 1.5
DEFAULT_POSE_STARTUP_TIMEOUT_S = 5.0
DEFAULT_POSE_MIN_CONFIDENCE = 0.5
STOP_HOLD_DURATION_S = 0.5
STOP_HOLD_HZ = 100.0

# Policy-side cube scale used by the existing UR5e+Delto MuJoCo adapter.
CUBE_OBJECT_SCALES = np.array([1.25, 1.25, 1.25], dtype=np.float32)
OBJECT_BASE_SIZE_M = 0.04
FAKE_OBJECT_X_OFFSET_M = -0.30
FAKE_OBJECT_Y_OFFSET_M = 0.15
TABLE_SIZE_M = np.array([0.475, 0.4, 0.3], dtype=np.float32)
TABLE_SURFACE_Z_M = 0.0
GOAL_OBJECT_OFFSET_M = np.array([0.0, 0.0, 0.124], dtype=np.float32)
ROBOT_BASE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
PALM_LOCAL_OFFSET_M = np.array([0.0, 0.16, 0.0], dtype=np.float64)

# The Universal Robots controller `base` frame is rotated by 180 degrees around
# Z relative to the URDF/MuJoCo base-link convention used by this viewer.
UR_CONTROLLER_BASE_TO_MODEL_BASE_ROTATION = np.diag(
    [-1.0, -1.0, 1.0]
).astype(np.float64)


class SimpleRateLimiter:
    def __init__(self, frequency: float) -> None:
        self.period = 1.0 / frequency
        self.next_time = time.monotonic()

    def sleep(self) -> None:
        self.next_time += self.period
        sleep_dt = self.next_time - time.monotonic()
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            self.next_time = time.monotonic()


class CommandStreamer:
    def __init__(self, command_socket: zmq.Socket, frequency: float) -> None:
        if frequency <= 0.0:
            raise ValueError(f"Command stream frequency must be positive, got {frequency}")
        self.command_socket = command_socket
        self.period = 1.0 / frequency
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.target_q: Optional[list[float]] = None
        self.thread = threading.Thread(target=self._run, name="command-streamer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def set_target(self, target_q: np.ndarray) -> None:
        target = np.asarray(target_q, dtype=float)
        if target.shape[0] != ARM_DOF:
            raise ValueError(f"target_q has shape {target.shape}; expected {ARM_DOF}")
        with self.lock:
            self.target_q = target.tolist()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        next_send = time.monotonic()
        while not self.stop_event.is_set():
            with self.lock:
                target_q = None if self.target_q is None else list(self.target_q)
            if target_q is not None:
                self.command_socket.send_json({"target_q": target_q})

            next_send += self.period
            sleep_dt = next_send - time.monotonic()
            if sleep_dt > 0:
                self.stop_event.wait(sleep_dt)
            else:
                next_send = time.monotonic()


def best_checkpoint_path(nn_dir: Path) -> Path:
    best_path = RUN_DIR / "best" / "model.pth"
    if best_path.exists():
        return best_path
    if best_path.is_symlink():
        raise FileNotFoundError(
            f"Default checkpoint symlink is broken: {best_path} -> {best_path.readlink()}"
        )

    candidates = []
    pattern = re.compile(r"_rew_([-+]?\d+(?:\.\d+)?)\.pth$")
    for path in nn_dir.glob("*.pth"):
        match = pattern.search(path.name)
        if match is not None:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        fallback = nn_dir / f"{RUN_DIR.name}.pth"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            "No policy checkpoint found. Expected one of:\n"
            f"  - {best_path}\n"
            f"  - {fallback}\n"
            f"  - any '*_rew_*.pth' file in {nn_dir}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def make_zmq_sockets(config: dict) -> tuple[zmq.Context, zmq.Socket, zmq.Socket]:
    context = zmq.Context()

    command_socket = context.socket(zmq.PUB)
    command_socket.bind(f"tcp://*:{config['socket_port']}")

    state_socket = context.socket(zmq.SUB)
    state_socket.setsockopt(zmq.CONFLATE, 1)
    state_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    state_socket.connect(f"tcp://127.0.0.1:{config['publisher_port']}")

    return context, command_socket, state_socket


def receive_latest_robot_state(state_socket: zmq.Socket) -> Optional[dict]:
    state = None
    while True:
        try:
            state = state_socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return state


def send_zero_velocity_hold(
    command_socket: zmq.Socket,
    state_socket: zmq.Socket,
    fallback_state: Optional[dict],
    duration_s: float = STOP_HOLD_DURATION_S,
    frequency_hz: float = STOP_HOLD_HZ,
) -> bool:
    """Brake by repeatedly targeting the latest measured joint position.

    The low-level protocol accepts joint positions, not joint velocities. Its
    joint-mode velocity command is proportional to target_q - current_q, so
    continuously sending the latest measured Q requests approximately zero
    joint velocity.
    """
    latest_q = None
    fallback_q = (
        np.asarray(fallback_state.get("Q", []), dtype=np.float64)
        if isinstance(fallback_state, dict)
        else np.empty(0, dtype=np.float64)
    )
    if fallback_q.shape[0] >= ARM_DOF and np.all(
        np.isfinite(fallback_q[:ARM_DOF])
    ):
        latest_q = fallback_q[:ARM_DOF].copy()

    deadline = time.monotonic() + duration_s
    period = 1.0 / frequency_hz
    sent_count = 0
    while time.monotonic() < deadline:
        state = receive_latest_robot_state(state_socket)
        if state is not None:
            measured_q = np.asarray(state.get("Q", []), dtype=np.float64)
            if measured_q.shape[0] >= ARM_DOF and np.all(
                np.isfinite(measured_q[:ARM_DOF])
            ):
                latest_q = measured_q[:ARM_DOF].copy()

        if latest_q is not None:
            command_socket.send_json({"target_q": latest_q.tolist()})
            sent_count += 1

        time.sleep(period)

    if sent_count == 0:
        print(
            "WARNING: could not send the zero-velocity hold because no valid "
            "robot joint state was available."
        )
        return False

    print(
        "Zero-velocity hold sent using the latest measured joint position "
        f"({sent_count} commands over {duration_s:g} s)."
    )
    return True


def make_pose_estimation_socket(
    context: zmq.Context,
    address: str,
) -> zmq.Socket:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(address)
    return socket


def receive_latest_object_pose(
    pose_socket: zmq.Socket,
    board_id: str,
    minimum_confidence: float,
) -> Optional[dict]:
    latest_pose = None
    while True:
        try:
            message = pose_socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return latest_pose

        poses = message.get("poses")
        if not isinstance(poses, dict):
            continue

        pose = poses.get(str(board_id))
        if not isinstance(pose, dict):
            continue

        confidence = float(pose.get("confidence", 0.0))
        if confidence < minimum_confidence:
            continue

        position = np.asarray(pose.get("position"), dtype=np.float64)
        rotation = np.asarray(pose.get("rotation_matrix"), dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (3, 3):
            continue
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            continue

        # Project small numerical errors onto the closest proper rotation.
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt

        latest_pose = {
            "position": position,
            "rotation_matrix": rotation,
            "confidence": confidence,
            "publisher_timestamp": message.get("timestamp"),
        }


def robot_state_to_policy_q(
    state: dict,
    previous_q: Optional[np.ndarray],
    default_joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = default_joint_pos.copy()
    qd = np.zeros(N_ACT, dtype=np.float32)

    if previous_q is not None:
        q[:] = previous_q

    arm_q = np.asarray(state.get("Q", []), dtype=np.float32)
    arm_qd = np.asarray(state.get("Qd", []), dtype=np.float32)
    if arm_q.shape[0] < ARM_DOF:
        raise ValueError(f"Robot state Q has shape {arm_q.shape}; expected at least {ARM_DOF}")

    q[:ARM_DOF] = arm_q[:ARM_DOF]
    if arm_qd.shape[0] >= ARM_DOF:
        qd[:ARM_DOF] = arm_qd[:ARM_DOF]
    return q, qd


def format_deg(values: np.ndarray) -> list[float]:
    return np.rad2deg(values).round(3).tolist()


def print_model_pose_debug(label: str, model, data) -> None:
    print(f"[debug] {label}", flush=True)
    print(f"[debug]   data.qpos_deg: {format_deg(data.qpos[:ARM_DOF])}", flush=True)
    if model.nu >= ARM_DOF:
        print(f"[debug]   data.ctrl_deg: {format_deg(data.ctrl[:ARM_DOF])}", flush=True)
    if model.nkey > 0:
        print(
            f"[debug]   home.qpos_deg: {format_deg(model.key_qpos[0, :ARM_DOF])}",
            flush=True,
        )


def wait_for_debug_step() -> None:
    print("Press Space/Enter for next step, or q to stop: ", end="", flush=True)
    if sys.stdin.isatty():
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()
    else:
        key = input().strip().lower()

    if key.lower() == "q":
        raise KeyboardInterrupt


def wait_for_space(prompt: str) -> None:
    print(prompt, end="", flush=True)
    if sys.stdin.isatty():
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                key = sys.stdin.read(1)
                if key == " ":
                    print()
                    return
                if key.lower() == "q":
                    raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
        while True:
            key = input().strip().lower()
            if key in ("", "space"):
                return
            if key == "q":
                raise KeyboardInterrupt


def print_home_confirmation(state_q: np.ndarray, home_q: np.ndarray) -> None:
    print()
    print("state")
    print(f"  q_rad: {state_q.round(6).tolist()}")
    print(f"  q_deg: {format_deg(state_q)}")
    print("target home position")
    print(f"  q_rad: {home_q.round(6).tolist()}")
    print(f"  q_deg: {format_deg(home_q)}")


def set_viewer_box(
    viewer,
    geom_index: int,
    pos: np.ndarray,
    size: np.ndarray,
    rgba: np.ndarray,
    geom_type,
    rotation: Optional[np.ndarray] = None,
) -> None:
    mat = (
        np.eye(3, dtype=np.float64)
        if rotation is None
        else np.asarray(rotation, dtype=np.float64)
    ).reshape(-1)
    mujoco = sys.modules["mujoco"]
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_index],
        geom_type,
        size.astype(np.float64),
        pos.astype(np.float64),
        mat,
        rgba.astype(np.float32),
    )


def update_viewer_markers(
    viewer,
    table_pos: np.ndarray,
    object_pos: np.ndarray,
    object_rotation: np.ndarray,
    goal_object_pos: np.ndarray,
    goal_object_rotation: np.ndarray,
    target_palm_pos: Optional[np.ndarray] = None,
) -> None:
    mujoco = sys.modules["mujoco"]
    viewer.user_scn.ngeom = 4 if target_palm_pos is not None else 3
    set_viewer_box(
        viewer=viewer,
        geom_index=0,
        pos=table_pos,
        size=TABLE_SIZE_M / 2.0,
        rgba=np.array([0.85, 0.85, 0.85, 0.35], dtype=np.float32),
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
    )
    set_viewer_box(
        viewer=viewer,
        geom_index=1,
        pos=object_pos,
        size=OBJECT_BASE_SIZE_M * CUBE_OBJECT_SCALES / 2.0,
        rgba=np.array([0.45, 0.45, 0.45, 1.0], dtype=np.float32),
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        rotation=object_rotation,
    )
    set_viewer_box(
        viewer=viewer,
        geom_index=2,
        pos=goal_object_pos,
        size=OBJECT_BASE_SIZE_M * CUBE_OBJECT_SCALES / 2.0,
        rgba=np.array([0.1, 0.9, 0.2, 0.35], dtype=np.float32),
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        rotation=goal_object_rotation,
    )
    if target_palm_pos is not None:
        set_viewer_box(
            viewer=viewer,
            geom_index=3,
            pos=target_palm_pos,
            size=np.array([0.035, 0.035, 0.035], dtype=np.float32),
            rgba=np.array([1.0, 0.75, 0.05, 0.85], dtype=np.float32),
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the trained SimToolReal UR5 arm policy on the real low-level arm controller."
    )
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--low-level-config-path", type=Path, default=LOW_LEVEL_CONFIG_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--send-to-robot", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever.")
    parser.add_argument(
        "--control-hz",
        type=float,
        default=CONTROL_HZ,
        help=(
            "Frequency at which the policy target is updated. The action "
            "increment remains calibrated at 60 Hz, so lower values slow the "
            "motion instead of making each step larger."
        ),
    )
    parser.add_argument("--command-stream-hz", type=float, default=CONTROL_HZ)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--debug-step",
        action="store_true",
        help="Print every step and wait for Space/Enter before sending/continuing.",
    )
    parser.add_argument(
        "--ignore-robot-state",
        action="store_true",
        help=(
            "Run an ideal closed-loop MuJoCo arm simulation instead of using "
            "Q from the low-level robot-state publisher."
        ),
    )
    parser.add_argument(
        "--use-pose-estimation",
        "--use_pose_estimation",
        dest="use_pose_estimation",
        action="store_true",
        help=(
            "Use the cube pose published by tag-pose-estimation instead of "
            "the built-in static cube pose."
        ),
    )
    parser.add_argument(
        "--pose-estimation-address",
        default=DEFAULT_POSE_ESTIMATION_ADDRESS,
        help="ZMQ address of the tag pose-estimation publisher.",
    )
    parser.add_argument(
        "--pose-board-id",
        default=DEFAULT_POSE_BOARD_ID,
        help="Board ID in the pose-estimation 'poses' dictionary.",
    )
    parser.add_argument(
        "--pose-min-confidence",
        type=float,
        default=DEFAULT_POSE_MIN_CONFIDENCE,
        help="Minimum accepted pose-estimation confidence.",
    )
    parser.add_argument(
        "--pose-timeout",
        type=float,
        default=DEFAULT_POSE_TIMEOUT_S,
        help="Stop if no valid cube pose is received for this many seconds.",
    )
    parser.add_argument("--print-every", type=float, default=PRINT_PERIOD_S)
    args = parser.parse_args()

    if args.ignore_robot_state and args.send_to_robot:
        parser.error(
            "--ignore-robot-state is a simulation mode and cannot be combined "
            "with --send-to-robot"
        )
    if not 0.0 <= args.pose_min_confidence <= 1.0:
        parser.error("--pose-min-confidence must be between 0 and 1")
    if args.pose_timeout <= 0.0:
        parser.error("--pose-timeout must be positive")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch

    import mujoco
    import mujoco.viewer
    from scipy.spatial.transform import Rotation

    from deployment.mujoco_ur5e_delto.policy_adapter import (
        DEFAULT_JOINT_POS,
        FINGERTIP_LOCAL_OFFSETS,
        LOWER_LIMITS,
        UPPER_LIMITS,
        build_observation,
        compute_targets,
        create_rl_player,
        read_policy_cfg,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    checkpoint_path = args.checkpoint_path or best_checkpoint_path(NN_DIR)
    cfg = read_policy_cfg(args.config_path)
    env_cfg = cfg["task"]["env"]
    num_obs = int(env_cfg.get("numObservations", N_OBS) or N_OBS)
    num_act = int(env_cfg.get("numActions", N_ACT) or N_ACT)
    if (num_obs, num_act) != (N_OBS, N_ACT):
        raise RuntimeError(
            f"This controller expects a UR5e+Delto policy with {N_OBS}/{N_ACT} "
            f"obs/actions, got {num_obs}/{num_act}"
        )
    default_joint_pos = DEFAULT_JOINT_POS.copy()
    default_arm_dof_pos = np.asarray(
        env_cfg.get("defaultArmDofPos", default_joint_pos[:ARM_DOF]),
        dtype=np.float32,
    )
    if default_arm_dof_pos.shape[0] != ARM_DOF:
        raise ValueError(
            f"defaultArmDofPos has shape {default_arm_dof_pos.shape}; expected {ARM_DOF}"
        )
    default_joint_pos[:ARM_DOF] = default_arm_dof_pos

    with args.low_level_config_path.open(encoding="utf-8") as f:
        low_level_cfg = json.load(f)

    context, command_socket, state_socket = make_zmq_sockets(low_level_cfg)
    pose_socket = None
    if args.use_pose_estimation:
        pose_socket = make_pose_estimation_socket(
            context, args.pose_estimation_address
        )
    print(f"Policy config:     {args.config_path}")
    print(f"Policy checkpoint: {checkpoint_path}")
    print(f"Listening state:   tcp://127.0.0.1:{low_level_cfg['publisher_port']}")
    print(f"Publishing target: tcp://*:{low_level_cfg['socket_port']}")
    print("Robot state input: IGNORED." if args.ignore_robot_state else "Robot state input: ENABLED.")
    print("Robot publishing is ENABLED." if args.send_to_robot else "Dry run: not sending robot commands.")
    print(f"Command stream:    {args.command_stream_hz:g} Hz" if args.send_to_robot else "Command stream:    disabled")
    print(
        f"Policy target rate: {args.control_hz:g} Hz "
        f"(action increment calibrated at {POLICY_ACTION_HZ:g} Hz)"
    )
    if pose_socket is not None:
        print(
            f"Cube pose input:   {args.pose_estimation_address} "
            f"(board {args.pose_board_id}, confidence >= "
            f"{args.pose_min_confidence:g})"
        )
    else:
        print("Cube pose input:   built-in static pose")
    command_streamer = None
    if args.send_to_robot:
        command_streamer = CommandStreamer(command_socket, args.command_stream_hz)
        command_streamer.start()

    robot_brake_completed = False

    def brake_robot_before_shutdown(
        fallback_state: Optional[dict] = None,
    ) -> None:
        nonlocal robot_brake_completed
        if robot_brake_completed or command_streamer is None:
            return
        robot_brake_completed = True
        command_streamer.stop()
        send_zero_velocity_hold(
            command_socket,
            state_socket,
            fallback_state,
        )

    # This also covers Ctrl+C during the home-position prompts, before the main
    # policy loop's try/finally block has started.
    atexit.register(brake_robot_before_shutdown)

    robot_base_y = 0.6 #float(env_cfg.get("robotBaseY", 0.6))
    table_pose_dy = -0.6 # float(env_cfg.get("tablePoseDy", -0.6))
    table_y = robot_base_y + table_pose_dy
    # MuJoCo box positions refer to their center. Put the table center half its
    # height below zero so that its upper surface is exactly at world z = 0.
    table_z = TABLE_SURFACE_Z_M - float(TABLE_SIZE_M[2]) / 2.0

    model_path = HERE / "assets" / "universal_robots_ur5e" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    base_body_id = model.body("base").id
    model.body_pos[base_body_id] = np.array([0.0, robot_base_y, 0.0], dtype=np.float64)
    model.body_quat[base_body_id] = ROBOT_BASE_QUAT_WXYZ
    data = mujoco.MjData(model)
    target_data = mujoco.MjData(model)
    data.qpos[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    target_data.qpos[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    data.ctrl[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    target_data.ctrl[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, target_data)
    if args.debug_step:
        print(f"[debug] defaultArmDofPos_deg: {format_deg(default_joint_pos[:ARM_DOF])}", flush=True)
        print(
            f"[debug] robot base world pos: {model.body_pos[base_body_id].round(6).tolist()}",
            flush=True,
        )
        print(
            f"[debug] robot base world quat_wxyz: {model.body_quat[base_body_id].round(6).tolist()}",
            flush=True,
        )
        print_model_pose_debug("after initial qpos/ctrl setup", model, data)
    wrist_3_body_id = model.body("wrist_3_link").id
    table_pos = np.array([0.0, table_y, table_z], dtype=np.float32)

    model_base_quat_wxyz = model.body_quat[base_body_id].copy()
    model_base_rotation = Rotation.from_quat(
        model_base_quat_wxyz[[1, 2, 3, 0]]
    ).as_matrix()
    controller_base_to_model_world_rotation = (
        model_base_rotation @ UR_CONTROLLER_BASE_TO_MODEL_BASE_ROTATION
    )
    model_base_position = model.body_pos[base_body_id].copy()

    def pose_estimation_to_model_world(pose: dict):
        position = (
            model_base_position
            + controller_base_to_model_world_rotation @ pose["position"]
        )
        rotation = (
            controller_base_to_model_world_rotation
            @ pose["rotation_matrix"]
        )
        quat_xyzw = Rotation.from_matrix(rotation).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return (
            position.astype(np.float32),
            rotation.astype(np.float64),
            quat_wxyz.astype(np.float32),
        )

    object_pos = np.array(
        [
            FAKE_OBJECT_X_OFFSET_M,
            table_y + FAKE_OBJECT_Y_OFFSET_M,
            TABLE_SURFACE_Z_M
            + OBJECT_BASE_SIZE_M * float(CUBE_OBJECT_SCALES[2]) / 2.0,
        ],
        dtype=np.float32,
    )
    object_rotation = np.eye(3, dtype=np.float64)
    object_quat_wxyz = np.array(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    last_valid_pose_time = None
    latest_pose_confidence = None

    if pose_socket is not None:
        print(
            "Waiting for the first valid cube pose from "
            f"{args.pose_estimation_address}..."
        )
        deadline = time.monotonic() + DEFAULT_POSE_STARTUP_TIMEOUT_S
        initial_pose = None
        while time.monotonic() < deadline:
            initial_pose = receive_latest_object_pose(
                pose_socket,
                args.pose_board_id,
                args.pose_min_confidence,
            )
            if initial_pose is not None:
                break
            time.sleep(0.01)
        if initial_pose is None:
            raise TimeoutError(
                "No valid cube pose received from pose estimation within "
                f"{DEFAULT_POSE_STARTUP_TIMEOUT_S:g} seconds."
            )
        (
            object_pos,
            object_rotation,
            object_quat_wxyz,
        ) = pose_estimation_to_model_world(initial_pose)
        last_valid_pose_time = time.monotonic()
        latest_pose_confidence = initial_pose["confidence"]

    # The goal is anchored once from the initial cube pose; it must not follow
    # the live cube position, otherwise the policy target would move with it.
    goal_object_pos = object_pos + GOAL_OBJECT_OFFSET_M
    goal_object_rotation = object_rotation.copy()
    goal_object_quat_wxyz = object_quat_wxyz.copy()
    if args.debug_step:
        print(
            "[debug] table/object positions: "
            f"robotBaseY={robot_base_y}, tablePoseDy={table_pose_dy}, "
            f"table_center={table_pos.round(6).tolist()}, "
            f"object_pos={object_pos.round(6).tolist()}, "
            f"goal_object_pos={goal_object_pos.round(6).tolist()}",
            flush=True,
        )
    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        update_viewer_markers(
            viewer,
            table_pos,
            object_pos,
            object_rotation,
            goal_object_pos,
            goal_object_rotation,
        )
        viewer.sync()
        if args.debug_step:
            print_model_pose_debug("after viewer launch/sync", model, data)

    latest_state = {
        "Q": default_joint_pos[:ARM_DOF].tolist(),
        "Qd": [0.0] * ARM_DOF,
        "source": "config_default",
    }
    if not args.ignore_robot_state:
        latest_state = None
        deadline = time.monotonic() + STATE_TIMEOUT_S
        while time.monotonic() < deadline:
            latest_state = receive_latest_robot_state(state_socket)
            if latest_state is not None and "Q" in latest_state:
                break
            time.sleep(0.01)
        if latest_state is None or "Q" not in latest_state:
            raise TimeoutError("No robot state received. Is the low-level controller running?")
        if args.debug_step:
            print(f"[debug] initial robot state keys: {sorted(latest_state.keys())}", flush=True)
            print(
                f"[debug] initial robot Q_deg: {format_deg(np.asarray(latest_state['Q'])[:ARM_DOF])}",
                flush=True,
            )

    home_q = default_joint_pos[:ARM_DOF].copy()
    if args.ignore_robot_state:
        print(
            "Simulation mode: initializing the ideal simulated state at the "
            "policy home pose."
        )
    else:
        state_q = np.asarray(latest_state["Q"], dtype=np.float32)[:ARM_DOF]
        # Give the C++ subscriber a moment to connect before the first target.
        time.sleep(1.0)
        print_home_confirmation(state_q=state_q, home_q=home_q)
        wait_for_space("Press Space to send the home command, or q to stop: ")
        if command_streamer is not None:
            command_streamer.set_target(home_q)
            print("Home command streaming.")
        else:
            print("Dry run: home command not sent.")

    data.qpos[:ARM_DOF] = home_q
    data.qvel[:ARM_DOF] = 0.0
    data.ctrl[:ARM_DOF] = home_q
    mujoco.mj_forward(model, data)
    if viewer is not None:
        update_viewer_markers(
            viewer,
            table_pos,
            object_pos,
            object_rotation,
            goal_object_pos,
            goal_object_rotation,
        )
        viewer.sync()

    if args.ignore_robot_state:
        latest_state = {
            "Q": home_q.tolist(),
            "Qd": [0.0] * ARM_DOF,
            "source": "ideal_simulation",
        }
    else:
        wait_for_space(
            "Press Space after the robot is at home to start the policy, or q to stop: "
        )
        latest_home_state = receive_latest_robot_state(state_socket)
        if latest_home_state is not None and "Q" in latest_home_state:
            latest_state = latest_home_state
        else:
            latest_state = {
                "Q": home_q.tolist(),
                "Qd": [0.0] * ARM_DOF,
                "source": "home_target",
            }

    player = create_rl_player(
        simtoolreal_root=REPO_ROOT,
        config_path=args.config_path,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    data.qpos[:ARM_DOF] = home_q
    data.ctrl[:ARM_DOF] = home_q
    mujoco.mj_forward(model, data)
    if viewer is not None:
        update_viewer_markers(
            viewer,
            table_pos,
            object_pos,
            object_rotation,
            goal_object_pos,
            goal_object_rotation,
        )
        viewer.sync()
    if args.debug_step:
        print_model_pose_debug("after policy load and resync", model, data)

    obs_list = env_cfg["obsList"]
    if args.control_hz <= 0.0:
        raise ValueError(f"--control-hz must be positive, got {args.control_hz}")
    # Keep these two time scales separate. loop_dt describes elapsed wall time
    # between target updates; POLICY_ACTION_DT is the fixed integration step
    # used during training. At --control-hz 1, the target therefore advances
    # by one 60 Hz-sized increment per second instead of one huge 1 Hz step.
    loop_dt = 1.0 / args.control_hz
    dof_speed_scale = float(env_cfg["dofSpeedScale"])
    arm_moving_average = float(env_cfg["armMovingAverage"])
    hand_moving_average = float(env_cfg["handMovingAverage"])

    prev_q = None
    prev_targets = None
    rate = SimpleRateLimiter(frequency=args.control_hz)
    last_print = 0.0
    step = 0
    stopped_by_viewer = False

    try:
        while args.max_steps <= 0 or step < args.max_steps:
            if viewer is not None and not viewer.is_running():
                stopped_by_viewer = True
                break

            if pose_socket is not None:
                pose_update = receive_latest_object_pose(
                    pose_socket,
                    args.pose_board_id,
                    args.pose_min_confidence,
                )
                if pose_update is not None:
                    (
                        object_pos,
                        object_rotation,
                        object_quat_wxyz,
                    ) = pose_estimation_to_model_world(pose_update)
                    last_valid_pose_time = time.monotonic()
                    latest_pose_confidence = pose_update["confidence"]
                elif (
                    last_valid_pose_time is None
                    or time.monotonic() - last_valid_pose_time
                    > args.pose_timeout
                ):
                    raise TimeoutError(
                        "Cube pose estimation is stale: no valid update for "
                        f"more than {args.pose_timeout:g} seconds."
                    )

            if args.ignore_robot_state:
                q = (
                    prev_targets.copy()
                    if prev_targets is not None
                    else default_joint_pos.copy()
                )
                qd = (
                    np.zeros_like(q)
                    if prev_q is None
                    else (q - prev_q) / loop_dt
                )
                latest_state = {
                    "Q": q[:ARM_DOF].tolist(),
                    "Qd": qd[:ARM_DOF].tolist(),
                    "source": "ideal_simulation",
                }
            else:
                state = receive_latest_robot_state(state_socket)
                if state is not None:
                    latest_state = state

                q, qd = robot_state_to_policy_q(
                    latest_state,
                    previous_q=prev_q,
                    default_joint_pos=default_joint_pos,
                )
            prev_q = q.copy()

            data.qpos[:ARM_DOF] = q[:ARM_DOF]
            data.qvel[:ARM_DOF] = qd[:ARM_DOF]
            data.ctrl[:ARM_DOF] = q[:ARM_DOF]
            mujoco.mj_forward(model, data)
            if viewer is not None:
                update_viewer_markers(
                    viewer,
                    table_pos,
                    object_pos,
                    object_rotation,
                    goal_object_pos,
                    goal_object_rotation,
                )
                viewer.sync()

            palm_quat_wxyz = data.xquat[wrist_3_body_id].copy()
            palm_rot = Rotation.from_quat(palm_quat_wxyz[[1, 2, 3, 0]])
            palm_pos = data.xpos[wrist_3_body_id].copy() + palm_rot.apply(PALM_LOCAL_OFFSET_M)
            fingertip_positions = palm_pos[None, :] + palm_rot.apply(
                FINGERTIP_LOCAL_OFFSETS
            )

            sim_state = {
                "joint_positions": q,
                "joint_velocities": qd,
                "palm_pos": palm_pos.astype(np.float32),
                "palm_quat_wxyz": palm_quat_wxyz.astype(np.float32),
                "fingertip_positions": fingertip_positions.astype(np.float32),
                "object_pos": object_pos,
                "object_quat_wxyz": object_quat_wxyz,
                "goal_object_pos": goal_object_pos,
                "goal_object_quat_wxyz": goal_object_quat_wxyz,
            }

            obs = build_observation(
                sim_state=sim_state,
                object_scales=CUBE_OBJECT_SCALES,
                obs_list=obs_list,
                prev_targets=prev_targets,
            )
            obs_t = torch.from_numpy(obs).float().to(args.device)
            # print("OBSERVATIONS: ", obs_t)
            with torch.no_grad():
                action = player.get_normalized_action(obs_t, deterministic_actions=True)

            targets = compute_targets(
                actions=action.cpu().numpy()[0],
                q=q,
                prev_targets=prev_targets,
                control_dt=POLICY_ACTION_DT,
                dof_speed_scale=dof_speed_scale,
                arm_moving_average=arm_moving_average,
                hand_moving_average=hand_moving_average,
            )
            targets[:ARM_DOF] = np.clip(
                targets[:ARM_DOF], LOWER_LIMITS[:ARM_DOF], UPPER_LIMITS[:ARM_DOF]
            )

            arm_error_deg = np.rad2deg(np.abs(targets[:ARM_DOF] - q[:ARM_DOF]))
            if arm_error_deg.max() > MAX_ARM_TARGET_ERROR_DEG:
                raise RuntimeError(
                    "Refusing to send arm target too far from current state: "
                    f"{arm_error_deg.round(2).tolist()} deg"
                )

            action_arm = action.cpu().numpy()[0, :ARM_DOF]
            target_palm_pos = None
            if viewer is not None:
                target_data.qpos[:ARM_DOF] = targets[:ARM_DOF]
                target_data.qvel[:ARM_DOF] = 0.0
                target_data.ctrl[:ARM_DOF] = targets[:ARM_DOF]
                mujoco.mj_forward(model, target_data)
                target_palm_quat_wxyz = target_data.xquat[wrist_3_body_id].copy()
                target_palm_rot = Rotation.from_quat(
                    target_palm_quat_wxyz[[1, 2, 3, 0]]
                )
                target_palm_pos = target_data.xpos[
                    wrist_3_body_id
                ].copy() + target_palm_rot.apply(PALM_LOCAL_OFFSET_M)
                update_viewer_markers(
                    viewer,
                    table_pos,
                    object_pos,
                    object_rotation,
                    goal_object_pos,
                    goal_object_rotation,
                    target_palm_pos=target_palm_pos,
                )
                viewer.sync()

            if command_streamer is not None:
                command_streamer.set_target(targets[:ARM_DOF])

            if args.debug_step:
                print()
                print(f"step:             {step}")
                print(f"send_to_robot:    {args.send_to_robot}")
                print(f"q_rad:            {q[:ARM_DOF].round(6).tolist()}")
                print(f"q_deg:            {format_deg(q[:ARM_DOF])}")
                print(f"target_q_rad:     {targets[:ARM_DOF].round(6).tolist()}")
                print(f"target_q_deg:     {format_deg(targets[:ARM_DOF])}")
                print(f"target_delta_deg: {format_deg(targets[:ARM_DOF] - q[:ARM_DOF])}")
                print(f"qd_rad_s:         {qd[:ARM_DOF].round(6).tolist()}")
                print(f"action_arm:       {action_arm.round(6).tolist()}")
                print(f"object_pos:       {object_pos.round(6).tolist()}")
                if latest_pose_confidence is not None:
                    print(
                        f"pose_confidence:   {latest_pose_confidence:.3f}"
                    )
                print(f"state_source:     {latest_state.get('source', 'robot_state_publisher')}")
                
                if "timestamp_ms" in latest_state:
                    print(f"state_timestamp:  {latest_state['timestamp_ms']}")
                if target_palm_pos is not None:
                    print(f"target_palm_pos:  {target_palm_pos.round(6).tolist()}")
                wait_for_debug_step()

            prev_targets = targets
            now = time.monotonic()
            if not args.debug_step and now - last_print >= args.print_every:
                last_print = now
                pose_status = (
                    f" object_pos={object_pos.round(3).tolist()} "
                    f"pose_confidence={latest_pose_confidence:.3f}"
                    if latest_pose_confidence is not None
                    else ""
                )
                print(
                    f"step={step:06d} "
                    f"q_deg={np.rad2deg(q[:ARM_DOF]).round(2).tolist()} "
                    f"target_deg={np.rad2deg(targets[:ARM_DOF]).round(2).tolist()} "
                    f"action_arm={action_arm.round(3).tolist()}"
                    f"{pose_status}"
                )

            step += 1
            if not args.debug_step:
                rate.sleep()

        if args.max_steps > 0 and step >= args.max_steps and not stopped_by_viewer:
            if prev_targets is not None and command_streamer is not None:
                command_streamer.set_target(prev_targets[:ARM_DOF])
            print(
                f"Reached --max-steps={args.max_steps}. "
                "Holding the last target position; close the viewer or press Ctrl+C to stop."
            )
            hold_rate = SimpleRateLimiter(frequency=args.control_hz)
            while viewer is None or viewer.is_running():
                if pose_socket is not None:
                    pose_update = receive_latest_object_pose(
                        pose_socket,
                        args.pose_board_id,
                        args.pose_min_confidence,
                    )
                    if pose_update is not None:
                        (
                            object_pos,
                            object_rotation,
                            object_quat_wxyz,
                        ) = pose_estimation_to_model_world(pose_update)
                        last_valid_pose_time = time.monotonic()
                        latest_pose_confidence = pose_update["confidence"]
                    elif (
                        last_valid_pose_time is None
                        or time.monotonic() - last_valid_pose_time
                        > args.pose_timeout
                    ):
                        raise TimeoutError(
                            "Cube pose estimation is stale: no valid update "
                            f"for more than {args.pose_timeout:g} seconds."
                        )

                if args.ignore_robot_state:
                    q = (
                        prev_targets.copy()
                        if prev_targets is not None
                        else default_joint_pos.copy()
                    )
                    qd = np.zeros_like(q)
                    latest_state = {
                        "Q": q[:ARM_DOF].tolist(),
                        "Qd": qd[:ARM_DOF].tolist(),
                        "source": "ideal_simulation",
                    }
                else:
                    state = receive_latest_robot_state(state_socket)
                    if state is not None:
                        latest_state = state

                    q, qd = robot_state_to_policy_q(
                        latest_state,
                        previous_q=prev_q,
                        default_joint_pos=default_joint_pos,
                    )
                prev_q = q.copy()

                data.qpos[:ARM_DOF] = q[:ARM_DOF]
                data.qvel[:ARM_DOF] = qd[:ARM_DOF]
                data.ctrl[:ARM_DOF] = q[:ARM_DOF]
                mujoco.mj_forward(model, data)

                target_palm_pos = None
                if prev_targets is not None:
                    target_data.qpos[:ARM_DOF] = prev_targets[:ARM_DOF]
                    target_data.qvel[:ARM_DOF] = 0.0
                    target_data.ctrl[:ARM_DOF] = prev_targets[:ARM_DOF]
                    mujoco.mj_forward(model, target_data)
                    target_palm_quat_wxyz = target_data.xquat[wrist_3_body_id].copy()
                    target_palm_rot = Rotation.from_quat(
                        target_palm_quat_wxyz[[1, 2, 3, 0]]
                    )
                    target_palm_pos = target_data.xpos[
                        wrist_3_body_id
                    ].copy() + target_palm_rot.apply(PALM_LOCAL_OFFSET_M)

                if viewer is not None:
                    update_viewer_markers(
                        viewer,
                        table_pos,
                        object_pos,
                        object_rotation,
                        goal_object_pos,
                        goal_object_rotation,
                        target_palm_pos=target_palm_pos,
                    )
                    viewer.sync()

                hold_rate.sleep()
    except KeyboardInterrupt:
        print("\nCtrl+C received: braking the robot before shutdown.")
    finally:
        brake_robot_before_shutdown(latest_state)
        atexit.unregister(brake_robot_before_shutdown)
        if viewer is not None:
            viewer.close()
        if pose_socket is not None:
            pose_socket.close()
        command_socket.close()
        state_socket.close()
        context.term()
        print("Stopped.")


if __name__ == "__main__":
    main()
