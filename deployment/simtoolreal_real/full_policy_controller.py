from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
import zmq


HERE = Path(__file__).resolve().parent
SIMTOOLREAL_ROOT = Path(
    os.environ.get("SIMTOOLREAL_ROOT", HERE.parent.parent)
).expanduser()

ARM_DOF = 6
HAND_DOF = 20
POLICY_ACTION_HZ = 60.0
POLICY_ACTION_DT = 1.0 / POLICY_ACTION_HZ
DEFAULT_CONTROL_HZ = 60.0
DEFAULT_HAND_COMMAND_STREAM_HZ = 100.0
DEFAULT_ARM_COMMAND_STREAM_HZ = 60.0
DEFAULT_ARM_STATE_TIMEOUT_S = 0.5
DEFAULT_MAX_ARM_TARGET_ERROR_DEG = 10.0
DEFAULT_ARM_HOME_TIMEOUT_S = 30.0
DEFAULT_ARM_HOME_TOLERANCE_DEG = 2.0
DEFAULT_ARM_LOW_LEVEL_CONFIG_PATH = HERE / "pc_ur_new.json"
REAL_ARM_WRIST_3_OFFSET_RAD = np.deg2rad(-60.0)   # this is the angle at which the tesollo hand is mounted on the UR5 wrist_3 joint
DEFAULT_STATE_ADDRESS = "127.0.0.1"
DEFAULT_STATE_PORT = 5563
DEFAULT_COMMAND_ADDRESS = "127.0.0.1"
DEFAULT_COMMAND_PORT = 5562
DEFAULT_STATE_TIMEOUT_S = 0.5
DEFAULT_MAX_TARGET_STEP_RAD = 0.10
DEFAULT_MAX_TRACKING_ERROR_RAD = 0.35
TRACKING_WARNING_PERIOD_S = 1.0
DEFAULT_NEUTRAL_TIMEOUT_S = 15.0
DEFAULT_NEUTRAL_TOLERANCE_RAD = 0.05
DEFAULT_COLLISION_WARNING_PERIOD_S = 1.0
DEFAULT_COLLISION_DISTANCE_THRESHOLD_M = 0.0
DEFAULT_POSE_ADDRESS = "tcp://127.0.0.1:5557"
DEFAULT_POSE_TIMEOUT_S = 1.5
DEFAULT_POSE_CONFIDENCE = 0.5
STOP_HOLD_DURATION_S = 0.5
STOP_HOLD_HZ = 50.0

# Reflection across the hand's xz plane maps legacy left-hand policy/model
# joint coordinates to the geometrically equivalent physical right hand.
LEFT_TO_RIGHT_SIGN = np.array(
    [
        -1,
        -1,
        -1,
        -1,
        -1,
        +1,
        +1,
        +1,
        -1,
        +1,
        +1,
        +1,
        -1,
        +1,
        +1,
        +1,
        -1,
        -1,
        +1,
        +1,
    ],
    dtype=np.float32,
)
RIGHT_HAND_JOINT_NAMES = [
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]
ARM_BODY_NAMES = {
    "base",
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
}
HAND_BODY_PREFIX = {
    "right": "rl_dg_",
    "left": "ll_dg_",
}
HAND_ARM_EXCLUDED_BODIES = {
    "right": {"rl_dg_mount", "rl_dg_base"},
    "left": {"ll_dg_mount", "ll_dg_base"},
}
TABLE_BODY_NAME = "table"
ANSI_BOLD_BRIGHT_YELLOW = "\033[1;93m"
ANSI_BOLD_ORANGE = "\033[1;38;5;208m"
ANSI_BOLD_GREEN = "\033[1;92m"
ANSI_RESET = "\033[0m"
UR_CONTROLLER_BASE_TO_MUJOCO_BASE_ROTATION = np.diag(
    [-1.0, -1.0, 1.0]
).astype(np.float64)


class RateLimiter:
    def __init__(self, frequency_hz: float) -> None:
        self.period = 1.0 / frequency_hz
        self.next_time = time.monotonic()

    def sleep(self) -> None:
        self.next_time += self.period
        delay = self.next_time - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        else:
            self.next_time = time.monotonic()


def policy_arm_to_real(q_policy: np.ndarray) -> np.ndarray:
    """Map policy/MuJoCo arm joints to the mechanically offset real UR5."""
    q_real = np.asarray(q_policy, dtype=np.float32).copy()
    if q_real.shape != (ARM_DOF,):
        raise ValueError(
            f"Arm joint vector has shape {q_real.shape}; expected {(ARM_DOF,)}"
        )
    q_real[-1] += REAL_ARM_WRIST_3_OFFSET_RAD
    return q_real


def real_arm_to_policy(q_real: np.ndarray) -> np.ndarray:
    """Map measured real UR5 joints back to policy/MuJoCo coordinates."""
    q_policy = np.asarray(q_real, dtype=np.float32).copy()
    if q_policy.shape != (ARM_DOF,):
        raise ValueError(
            f"Arm joint vector has shape {q_policy.shape}; expected {(ARM_DOF,)}"
        )
    q_policy[-1] -= REAL_ARM_WRIST_3_OFFSET_RAD
    return q_policy


class HandBridgeClient:
    def __init__(
        self,
        *,
        state_address: str,
        state_port: int,
        command_address: str,
        command_port: int,
    ) -> None:
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_socket.setblocking(False)
        self.state_socket.bind((state_address, state_port))
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_endpoint = (command_address, command_port)
        self.command_sequence = 0

    def close(self) -> None:
        self.state_socket.close()
        self.command_socket.close()

    def receive_latest_state(self) -> Optional[dict]:
        latest = None
        while True:
            try:
                payload, _ = self.state_socket.recvfrom(65535)
            except BlockingIOError:
                return latest
            try:
                message = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if message.get("type") != "hand_state":
                continue
            q = np.asarray(message.get("positions"), dtype=np.float32)
            qd = np.asarray(message.get("velocities"), dtype=np.float32)
            if q.shape != (HAND_DOF,) or qd.shape != (HAND_DOF,):
                continue
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                continue
            latest = {"q": q, "qd": qd, "received_at": time.monotonic()}

    def send_target(self, q_right: np.ndarray) -> None:
        self.command_sequence += 1
        payload = {
            "type": "hand_target",
            "sequence": self.command_sequence,
            "positions": np.asarray(q_right, dtype=float).tolist(),
        }
        self.command_socket.sendto(
            json.dumps(payload, separators=(",", ":")).encode(),
            self.command_endpoint,
        )


class HandCommandStreamer:
    def __init__(
        self,
        bridge: HandBridgeClient,
        frequency_hz: float,
    ) -> None:
        self.bridge = bridge
        self.period = 1.0 / frequency_hz
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.target: Optional[np.ndarray] = None
        self.thread = threading.Thread(
            target=self._run,
            name="hand-command-streamer",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def set_target(self, target: np.ndarray) -> None:
        with self.lock:
            self.target = np.asarray(target, dtype=np.float32).copy()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        next_send = time.monotonic()
        while not self.stop_event.is_set():
            with self.lock:
                target = None if self.target is None else self.target.copy()
            if target is not None:
                self.bridge.send_target(target)
            next_send += self.period
            delay = next_send - time.monotonic()
            if delay > 0.0:
                self.stop_event.wait(delay)
            else:
                next_send = time.monotonic()


def policy_to_right_hand(
    q_policy: np.ndarray,
    hand_side: str,
) -> np.ndarray:
    q_policy = np.asarray(q_policy, dtype=np.float32)
    if hand_side == "right":
        return q_policy.copy()
    return LEFT_TO_RIGHT_SIGN * q_policy


def right_hand_to_policy(
    q_right: np.ndarray,
    hand_side: str,
) -> np.ndarray:
    q_right = np.asarray(q_right, dtype=np.float32)
    if hand_side == "right":
        return q_right.copy()
    return LEFT_TO_RIGHT_SIGN * q_right


def print_warning(message: str, *, category: str = "tracking") -> None:
    text = f"WARNING: {message}"
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        color = (
            ANSI_BOLD_ORANGE
            if category == "collision"
            else ANSI_BOLD_BRIGHT_YELLOW
        )
        text = f"{color}{text}{ANSI_RESET}"
    print(text, flush=True)


def monitored_collisions(
    *,
    mujoco_module,
    model,
    data,
    distance_threshold_m: float,
    hand_side: str,
) -> list[dict]:
    """Return deepest monitored hand-arm, hand-table, and arm-table contacts."""
    collisions_by_pair: dict[tuple[str, str, str], dict] = {}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.dist > distance_threshold_m:
            continue

        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        body1_id = int(model.geom_bodyid[geom1_id])
        body2_id = int(model.geom_bodyid[geom2_id])
        body1 = model.body(body1_id).name or f"body_{body1_id}"
        body2 = model.body(body2_id).name or f"body_{body2_id}"

        hand_body_prefix = HAND_BODY_PREFIX[hand_side]
        excluded_hand_bodies = HAND_ARM_EXCLUDED_BODIES[hand_side]
        body1_is_hand = body1.startswith(hand_body_prefix)
        body2_is_hand = body2.startswith(hand_body_prefix)
        category = None
        if (
            body1_is_hand
            and body1 not in excluded_hand_bodies
            and body2 in ARM_BODY_NAMES
        ) or (
            body2_is_hand
            and body2 not in excluded_hand_bodies
            and body1 in ARM_BODY_NAMES
        ):
            category = "hand-arm"
        elif (
            body1_is_hand and body2 == TABLE_BODY_NAME
        ) or (
            body2_is_hand and body1 == TABLE_BODY_NAME
        ):
            category = "hand-table"
        elif (
            body1 in ARM_BODY_NAMES and body2 == TABLE_BODY_NAME
        ) or (
            body2 in ARM_BODY_NAMES and body1 == TABLE_BODY_NAME
        ):
            category = "arm-table"
        if category is None:
            continue

        force_torque = np.zeros(6, dtype=np.float64)
        if int(contact.efc_address) >= 0:
            mujoco_module.mj_contactForce(
                model, data, contact_index, force_torque
            )
        normal_force = float(abs(force_torque[0]))
        geom1 = model.geom(geom1_id).name or f"geom_{geom1_id}"
        geom2 = model.geom(geom2_id).name or f"geom_{geom2_id}"
        pair_key = (category, *sorted((body1, body2)))
        collision = {
            "category": category,
            "body1": body1,
            "body2": body2,
            "geom1": geom1,
            "geom2": geom2,
            "distance_m": float(contact.dist),
            "normal_force_n": normal_force,
        }
        previous = collisions_by_pair.get(pair_key)
        if (
            previous is None
            or collision["distance_m"] < previous["distance_m"]
            or collision["normal_force_n"] > previous["normal_force_n"]
        ):
            collisions_by_pair[pair_key] = collision
    return list(collisions_by_pair.values())


def format_collisions(collisions: list[dict]) -> str:
    return "; ".join(
        (
            f"{collision['category']} "
            f"{collision['body1']} ↔ {collision['body2']} "
            f"(penetration={max(0.0, -collision['distance_m']) * 1000.0:.2f} mm, "
            f"normal_force={collision['normal_force_n']:.2f} N, "
            f"geoms={collision['geom1']} ↔ {collision['geom2']})"
        )
        for collision in collisions
    )


def unscale(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (2.0 * x - upper - lower) / (upper - lower)


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat)[[1, 2, 3, 0]]


def object_keypoints(
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    object_scales: np.ndarray,
    keypoint_scale: float,
    object_base_size: float,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    signs = np.array(
        [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]],
        dtype=np.float32,
    )
    offsets = (
        signs * object_base_size * keypoint_scale * object_scales / 2.0
    )
    return position[None, :] + Rotation.from_quat(
        quat_wxyz_to_xyzw(quat_wxyz)
    ).apply(offsets)


def keypoint_goal_distance(
    *,
    sim_state: dict[str, np.ndarray],
    object_scales: np.ndarray,
    keypoint_scale: float,
    object_base_size: float,
) -> float:
    object_kps = object_keypoints(
        sim_state["object_pos"].astype(np.float32),
        sim_state["object_quat_wxyz"].astype(np.float32),
        object_scales,
        keypoint_scale,
        object_base_size,
    )
    goal_kps = object_keypoints(
        sim_state["goal_object_pos"].astype(np.float32),
        sim_state["goal_object_quat_wxyz"].astype(np.float32),
        object_scales,
        keypoint_scale,
        object_base_size,
    )
    return float(np.linalg.norm(object_kps - goal_kps, axis=-1).max())


def configured_goal_success_tolerance(env_cfg: dict) -> float:
    tolerance = env_cfg.get("evalSuccessTolerance")
    if tolerance is None:
        tolerance = env_cfg.get("targetSuccessTolerance")
    if tolerance is None:
        tolerance = env_cfg.get("successTolerance", 0.075)
    return float(tolerance)


def target_volume_bounds(
    env_cfg: dict, workspace_y: float
) -> tuple[np.ndarray, np.ndarray]:
    configured_mins = env_cfg.get("targetVolumeMins")
    configured_maxs = env_cfg.get("targetVolumeMaxs")
    region_scale = float(env_cfg.get("targetVolumeRegionScale", 1.0))
    if configured_mins is not None and configured_maxs is not None:
        mins = np.asarray(configured_mins, dtype=np.float32)
        maxs = np.asarray(configured_maxs, dtype=np.float32)
        origin = (mins + maxs) / 2.0
        lower_extent = -(maxs - mins) / 2.0
        upper_extent = (maxs - mins) / 2.0
    else:
        origin = np.array([0.0, 0.05, 0.8], dtype=np.float32)
        lower_extent = np.array([-0.4, -0.05, -0.12], dtype=np.float32)
        upper_extent = np.array([0.4, 0.3, 0.25], dtype=np.float32)

    origin[1] += workspace_y
    return (
        origin + lower_extent * region_scale,
        origin + upper_extent * region_scale,
    )


def sample_delta_quat_wxyz(
    quat_wxyz: np.ndarray, delta_rotation_degrees: float
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    axis = np.random.uniform(0.0, 1.0, size=3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(
        -np.deg2rad(delta_rotation_degrees),
        np.deg2rad(delta_rotation_degrees),
    )
    current = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])
    delta = Rotation.from_rotvec(axis * angle)
    quat_xyzw = (current * delta).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].astype(np.float32)


def sample_next_goal(
    *,
    sim_state: dict[str, np.ndarray],
    env_cfg: dict,
    workspace_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    sampling_type = env_cfg.get("goalSamplingType", "delta")
    goal_pos = sim_state["goal_object_pos"].astype(np.float32).copy()
    goal_quat_wxyz = sim_state["goal_object_quat_wxyz"].astype(np.float32).copy()
    volume_min, volume_max = target_volume_bounds(env_cfg, workspace_y)
    delta_distance = float(env_cfg.get("deltaGoalDistance", 0.1))
    delta_rotation_degrees = float(env_cfg.get("deltaRotationDegrees", 90.0))

    if sampling_type == "delta":
        goal_pos += np.random.uniform(
            -delta_distance, delta_distance, size=3
        ).astype(np.float32)
        goal_quat_wxyz = sample_delta_quat_wxyz(
            goal_quat_wxyz, delta_rotation_degrees
        )
    elif sampling_type == "coin_flip":
        if np.random.uniform() < 0.5:
            goal_pos += np.random.uniform(
                -delta_distance, delta_distance, size=3
            ).astype(np.float32)
        else:
            goal_quat_wxyz = sample_delta_quat_wxyz(
                goal_quat_wxyz, delta_rotation_degrees
            )
    else:
        goal_pos = np.random.uniform(volume_min, volume_max).astype(np.float32)
        goal_quat_wxyz = Rotation.random().as_quat()[[3, 0, 1, 2]].astype(
            np.float32
        )

    goal_pos = np.clip(goal_pos, volume_min, volume_max)
    return goal_pos, goal_quat_wxyz


def print_goal_reached(
    *,
    goal_index: int,
    progress: int,
    distance_m: float,
    tolerance_m: float,
) -> None:
    print(
        f"{ANSI_BOLD_GREEN}GOAL REACHED #{goal_index} at policy step "
        f"{progress}: keypoint distance={distance_m:.4f} m < "
        f"tolerance={tolerance_m:.4f} m{ANSI_RESET}",
        flush=True,
    )


def phase_observation(progress: int, waypoint_steps: np.ndarray) -> np.ndarray:
    phase = 0
    if progress >= int(waypoint_steps[1]):
        phase = 1
    if progress >= int(waypoint_steps[2]):
        phase = 2
    if progress >= int(waypoint_steps[3]):
        phase = 3
    result = np.zeros(4, dtype=np.float32)
    result[phase] = 1.0
    return result


def scripted_arm_target(
    progress: int,
    waypoint_steps: np.ndarray,
    waypoint_poses: np.ndarray,
) -> np.ndarray:
    if progress <= waypoint_steps[0]:
        return waypoint_poses[0].copy()
    for index in range(len(waypoint_steps) - 1):
        step_a = waypoint_steps[index]
        step_b = waypoint_steps[index + 1]
        if progress <= step_b:
            alpha = np.clip((progress - step_a) / (step_b - step_a), 0.0, 1.0)
            return (
                waypoint_poses[index]
                + alpha * (waypoint_poses[index + 1] - waypoint_poses[index])
            ).astype(np.float32)
    return waypoint_poses[-1].copy()


def build_observation(
    *,
    sim_state: dict[str, np.ndarray],
    obs_list: list[str],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    prev_targets: np.ndarray,
    object_scales: np.ndarray,
    object_base_size: float,
    keypoint_scale: float,
    progress: int,
    waypoint_steps: np.ndarray,
) -> np.ndarray:
    q = sim_state["joint_positions"].astype(np.float32)
    qd = sim_state["joint_velocities"].astype(np.float32)
    palm_pos = sim_state["palm_pos"].astype(np.float32)
    palm_quat_xyzw = quat_wxyz_to_xyzw(
        sim_state["palm_quat_wxyz"]
    ).astype(np.float32)
    fingertip_rel_palm = (
        sim_state["fingertip_positions"].astype(np.float32) - palm_pos[None, :]
    )
    object_pos = sim_state["object_pos"].astype(np.float32)
    object_quat_wxyz = sim_state["object_quat_wxyz"].astype(np.float32)
    goal_pos = sim_state["goal_object_pos"].astype(np.float32)
    goal_quat_wxyz = sim_state["goal_object_quat_wxyz"].astype(np.float32)
    object_kps = object_keypoints(
        object_pos,
        object_quat_wxyz,
        object_scales,
        keypoint_scale,
        object_base_size,
    )
    goal_kps = object_keypoints(
        goal_pos,
        goal_quat_wxyz,
        object_scales,
        keypoint_scale,
        object_base_size,
    )
    obs_dict = {
        "joint_pos": unscale(q, lower_limits, upper_limits),
        "joint_vel": qd,
        "prev_action_targets": prev_targets.astype(np.float32),
        "palm_pos": palm_pos,
        "palm_rot": palm_quat_xyzw,
        "object_rot": quat_wxyz_to_xyzw(object_quat_wxyz).astype(np.float32),
        "fingertip_pos_rel_palm": fingertip_rel_palm.reshape(-1),
        "keypoints_rel_palm": (object_kps - palm_pos[None, :]).reshape(-1),
        "keypoints_rel_goal": (object_kps - goal_kps).reshape(-1),
        "object_scales": object_scales.astype(np.float32),
        "progress": np.array(
            [np.log(float(progress) / 10.0 + 1.0)], dtype=np.float32
        ),
        "phase": phase_observation(progress, waypoint_steps),
    }
    unsupported = [name for name in obs_list if name not in obs_dict]
    if unsupported:
        raise RuntimeError(
            "The hand-only real-world adapter does not support observations: "
            + ", ".join(unsupported)
        )
    return np.concatenate([obs_dict[name].reshape(-1) for name in obs_list]).astype(
        np.float32
    )


def compute_hand_targets(
    *,
    action: np.ndarray,
    previous_targets: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    hand_speed_scale: float,
    moving_average: float,
    action_scale: float,
    max_target_step_rad: float,
) -> np.ndarray:
    action = np.clip(action, -1.0, 1.0) * action_scale
    raw = previous_targets + hand_speed_scale * POLICY_ACTION_DT * action
    raw = np.clip(raw, lower_limits, upper_limits)
    target = moving_average * raw + (1.0 - moving_average) * previous_targets
    delta = np.clip(
        target - previous_targets,
        -max_target_step_rad,
        max_target_step_rad,
    )
    return np.clip(
        previous_targets + delta, lower_limits, upper_limits
    ).astype(np.float32)


def compute_arm_targets(
    *,
    action: np.ndarray,
    previous_targets: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    dof_speed_scale: float,
    moving_average: float,
    action_scale: float,
) -> np.ndarray:
    """Match the relative arm-target update used during policy training."""
    action = np.clip(action, -1.0, 1.0) * action_scale
    raw = (
        previous_targets
        + dof_speed_scale * POLICY_ACTION_DT * action
    )
    raw = np.clip(raw, lower_limits, upper_limits)
    target = (
        moving_average * raw
        + (1.0 - moving_average) * previous_targets
    )
    return np.clip(target, lower_limits, upper_limits).astype(np.float32)


def load_policy(
    *,
    config_path: Path,
    checkpoint_path: Path,
    num_observations: int,
    num_actions: int,
    device: str,
):
    import torch

    if str(SIMTOOLREAL_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMTOOLREAL_ROOT))
    rl_games_path = SIMTOOLREAL_ROOT / "rl_games"
    if str(rl_games_path) not in sys.path:
        sys.path.insert(0, str(rl_games_path))
    from deployment.rl_player import RlPlayer

    original_torch_load = torch.load
    if device == "cpu":
        def torch_load_on_cpu(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device("cpu"))
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_on_cpu
    try:
        return RlPlayer(
            num_observations=num_observations,
            num_actions=num_actions,
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            device=device,
        )
    finally:
        torch.load = original_torch_load


def wait_for_initial_state(
    bridge: HandBridgeClient, timeout_s: float
) -> dict[str, np.ndarray]:
    deadline = time.monotonic() + timeout_s
    latest = None
    while time.monotonic() < deadline:
        state = bridge.receive_latest_state()
        if state is not None:
            latest = state
            break
        time.sleep(0.01)
    if latest is None:
        raise TimeoutError(
            "No DG5F state received from the ROS bridge. Is "
            "dg5f_policy_ros_bridge.py running?"
        )
    return latest


def parse_vector(values: list[float], expected: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (expected,):
        raise ValueError(f"{name} must contain {expected} values")
    return result


def make_pose_socket(address: str) -> tuple[zmq.Context, zmq.Socket]:
    context = zmq.Context()
    pose_socket = context.socket(zmq.SUB)
    pose_socket.setsockopt(zmq.LINGER, 0)
    pose_socket.setsockopt(zmq.CONFLATE, 1)
    pose_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    pose_socket.connect(address)
    return context, pose_socket


def receive_latest_pose(
    pose_socket: zmq.Socket,
    board_id: str,
    minimum_confidence: float,
) -> Optional[dict]:
    latest = None
    while True:
        try:
            message = pose_socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return latest
        pose = message.get("poses", {}).get(str(board_id))
        if not isinstance(pose, dict):
            continue
        confidence = float(pose.get("confidence", 0.0))
        position = np.asarray(pose.get("position"), dtype=np.float64)
        rotation = np.asarray(pose.get("rotation_matrix"), dtype=np.float64)
        if confidence < minimum_confidence:
            continue
        if position.shape != (3,) or rotation.shape != (3, 3):
            continue
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            continue
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        latest = {
            "position": position,
            "rotation": rotation,
            "confidence": confidence,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one SimToolReal policy for the complete UR5e + DG5F system, "
            "with independently enabled real arm and hand outputs."
        )
    )
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--left-hand",
        action="store_true",
        help=(
            "Use a legacy left-hand policy and show the left hand in MuJoCo. "
            "The default is a right-hand policy/model."
        ),
    )
    parser.add_argument(
        "--send-to-arm",
        action="store_true",
        help="Send policy arm targets to the real UR5 low-level controller.",
    )
    parser.add_argument(
        "--use-real-arm-state",
        action="store_true",
        help="Use the real UR5 joint state without sending arm commands.",
    )
    parser.add_argument("--send-to-hand", action="store_true")
    parser.add_argument(
        "--use-real-hand-state",
        action="store_true",
        help="Mirror /joint_states from the ROS bridge without sending commands.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the real-hardware confirmation prompt.",
    )
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--debug-step", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument(
        "--hand-command-stream-hz",
        "--command-stream-hz",
        dest="hand_command_stream_hz",
        type=float,
        default=DEFAULT_HAND_COMMAND_STREAM_HZ,
    )
    parser.add_argument(
        "--arm-command-stream-hz",
        type=float,
        default=DEFAULT_ARM_COMMAND_STREAM_HZ,
    )
    parser.add_argument(
        "--low-level-config-path",
        type=Path,
        default=DEFAULT_ARM_LOW_LEVEL_CONFIG_PATH,
    )
    parser.add_argument(
        "--arm-state-timeout",
        type=float,
        default=DEFAULT_ARM_STATE_TIMEOUT_S,
    )
    parser.add_argument(
        "--max-arm-target-error-deg",
        type=float,
        default=DEFAULT_MAX_ARM_TARGET_ERROR_DEG,
    )
    parser.add_argument(
        "--arm-home-timeout",
        type=float,
        default=DEFAULT_ARM_HOME_TIMEOUT_S,
    )
    parser.add_argument(
        "--arm-home-tolerance-deg",
        type=float,
        default=DEFAULT_ARM_HOME_TOLERANCE_DEG,
    )
    parser.add_argument("--print-every", type=float, default=1.0)
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale the 20 hand actions before target integration.",
    )
    parser.add_argument(
        "--arm-action-scale",
        type=float,
        default=1.0,
        help="Scale the 6 arm actions before target integration.",
    )
    parser.add_argument(
        "--max-target-step-rad",
        type=float,
        default=DEFAULT_MAX_TARGET_STEP_RAD,
    )
    parser.add_argument(
        "--max-tracking-error-rad",
        type=float,
        default=DEFAULT_MAX_TRACKING_ERROR_RAD,
    )
    parser.add_argument(
        "--tracking-error-mode",
        choices=("stop", "warn"),
        default="stop",
        help=(
            "Action taken when hand tracking error exceeds its limit. "
            "'stop' raises RuntimeError; 'warn' logs and continues."
        ),
    )
    parser.add_argument(
        "--neutral-timeout",
        type=float,
        default=DEFAULT_NEUTRAL_TIMEOUT_S,
        help="Maximum time allowed to reach the neutral hand pose.",
    )
    parser.add_argument(
        "--neutral-tolerance-rad",
        type=float,
        default=DEFAULT_NEUTRAL_TOLERANCE_RAD,
        help="Maximum per-joint error before the neutral pose is accepted.",
    )
    parser.add_argument(
        "--collision-mode",
        choices=("off", "warn", "stop"),
        default="warn",
        help=(
            "Action taken for monitored MuJoCo hand-arm, hand-table, or "
            "arm-table collisions."
        ),
    )
    parser.add_argument(
        "--collision-distance-threshold-m",
        type=float,
        default=DEFAULT_COLLISION_DISTANCE_THRESHOLD_M,
        help=(
            "Report contacts whose signed geom distance is at or below this "
            "value. Zero reports touching/penetrating geoms."
        ),
    )
    parser.add_argument(
        "--collision-warning-period",
        type=float,
        default=DEFAULT_COLLISION_WARNING_PERIOD_S,
    )
    parser.add_argument("--state-address", default=DEFAULT_STATE_ADDRESS)
    parser.add_argument("--state-port", type=int, default=DEFAULT_STATE_PORT)
    parser.add_argument("--command-address", default=DEFAULT_COMMAND_ADDRESS)
    parser.add_argument("--command-port", type=int, default=DEFAULT_COMMAND_PORT)
    parser.add_argument(
        "--state-timeout", type=float, default=DEFAULT_STATE_TIMEOUT_S
    )
    parser.add_argument("--use-pose-estimation", action="store_true")
    parser.add_argument("--pose-address", default=DEFAULT_POSE_ADDRESS)
    parser.add_argument("--pose-board-id", default="0")
    parser.add_argument(
        "--pose-min-confidence",
        type=float,
        default=DEFAULT_POSE_CONFIDENCE,
    )
    parser.add_argument(
        "--pose-timeout", type=float, default=DEFAULT_POSE_TIMEOUT_S
    )
    parser.add_argument(
        "--goal-success-tolerance-m",
        "--goal-tolerance-m",
        dest="goal_success_tolerance_m",
        type=float,
        default=None,
        help=(
            "Goal-reached keypoint distance tolerance in meters. By default, "
            "read evalSuccessTolerance/targetSuccessTolerance/successTolerance "
            "from the training config."
        ),
    )
    parser.add_argument(
        "--goal-success-steps",
        type=int,
        default=None,
        help=(
            "Number of near-goal policy steps needed before sampling the next "
            "goal. By default, read successSteps from the training config."
        ),
    )
    parser.add_argument(
        "--goal-quat-wxyz",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        default=None,
        help="Override the goal quaternion from the training config.",
    )
    args = parser.parse_args()

    if args.send_to_arm:
        args.use_real_arm_state = True
    if args.send_to_hand:
        args.use_real_hand_state = True
    if args.control_hz <= 0.0:
        parser.error("--control-hz must be positive")
    if args.hand_command_stream_hz <= 0.0:
        parser.error("--hand-command-stream-hz must be positive")
    if args.arm_command_stream_hz <= 0.0:
        parser.error("--arm-command-stream-hz must be positive")
    if not 0.0 < args.action_scale <= 1.0:
        parser.error("--action-scale must be in (0, 1]")
    if not 0.0 < args.arm_action_scale <= 1.0:
        parser.error("--arm-action-scale must be in (0, 1]")
    if args.max_target_step_rad <= 0.0:
        parser.error("--max-target-step-rad must be positive")
    if args.max_tracking_error_rad <= 0.0:
        parser.error("--max-tracking-error-rad must be positive")
    if args.neutral_timeout <= 0.0:
        parser.error("--neutral-timeout must be positive")
    if args.neutral_tolerance_rad <= 0.0:
        parser.error("--neutral-tolerance-rad must be positive")
    if args.collision_warning_period <= 0.0:
        parser.error("--collision-warning-period must be positive")
    if args.state_timeout <= 0.0:
        parser.error("--state-timeout must be positive")
    if args.arm_state_timeout <= 0.0:
        parser.error("--arm-state-timeout must be positive")
    if args.max_arm_target_error_deg <= 0.0:
        parser.error("--max-arm-target-error-deg must be positive")
    if args.arm_home_timeout <= 0.0:
        parser.error("--arm-home-timeout must be positive")
    if args.arm_home_tolerance_deg <= 0.0:
        parser.error("--arm-home-tolerance-deg must be positive")
    if args.pose_timeout <= 0.0:
        parser.error("--pose-timeout must be positive")
    if args.goal_success_tolerance_m is not None and args.goal_success_tolerance_m <= 0.0:
        parser.error("--goal-success-tolerance-m must be positive")
    if args.goal_success_steps is not None and args.goal_success_steps <= 0:
        parser.error("--goal-success-steps must be positive")
    if not 0.0 <= args.pose_min_confidence <= 1.0:
        parser.error("--pose-min-confidence must be between 0 and 1")

    if str(SIMTOOLREAL_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMTOOLREAL_ROOT))

    import torch
    import mujoco
    from deployment.mujoco_ur5e_delto.mujoco_sim import (
        Ur5eDeltoMujocoConfig,
        Ur5eDeltoMujocoSim,
    )
    from deployment.mujoco_ur5e_delto.policy_adapter import (
        DEFAULT_JOINT_POS,
        joint_limits_for_hand,
    )
    from deployment.simtoolreal_real.ur5_policy_arm_controller import (
        CommandStreamer as ArmCommandStreamer,
        make_zmq_sockets as make_arm_zmq_sockets,
        receive_latest_robot_state,
        send_zero_velocity_hold,
    )

    # MuJoCo <=3.3 exposed this compiler option directly on MjSpec; 3.9
    # removed the Python attribute. The shared simulator still assigns it.
    # Add a no-op compatibility property here without changing sim2sim code.
    if not hasattr(mujoco.MjSpec, "discardvisual"):
        setattr(
            mujoco.MjSpec,
            "discardvisual",
            property(lambda _spec: False, lambda _spec, _value: None),
        )
    if not hasattr(mujoco.MjsLight, "directional"):
        setattr(
            mujoco.MjsLight,
            "directional",
            property(lambda _light: True, lambda _light, _value: None),
        )

    # MjSpec.from_file changed from mutating the instance to returning a new
    # instance. Keep this compatibility adapter private to the real-world
    # controller instead of altering deployment/mujoco_ur5e_delto.
    def init_scene_compatible(self) -> None:
        spec = mujoco.MjSpec.from_file(
            str(self._make_mujoco_compatible_urdf())
        )
        spec.compiler.discardvisual = False
        spec.compiler.fusestatic = False
        self._add_world(spec)
        self._add_position_actuators(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config.sim_dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.iterations = 20
        self.model.opt.ls_iterations = 50
        self._joint_qpos_adrs = np.array(
            [self.model.joint(name).qposadr[0] for name in self.joint_names],
            dtype=np.int32,
        )
        self._joint_dof_adrs = np.array(
            [self.model.joint(name).dofadr[0] for name in self.joint_names],
            dtype=np.int32,
        )
        self._actuator_ids = np.array(
            [
                self.model.actuator(f"{name}_pos").id
                for name in self.joint_names
            ],
            dtype=np.int32,
        )
        self._validate()
        if self.config.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    Ur5eDeltoMujocoSim._init_scene = init_scene_compatible

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")
        args.device = "cpu"

    with args.config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    env_cfg = cfg["task"]["env"]
    hand_side = "left" if args.left_hand else "right"
    configured_hand_side_raw = env_cfg.get("handSide")
    configured_hand_side = (
        None
        if configured_hand_side_raw is None
        else str(configured_hand_side_raw).lower()
    )
    if configured_hand_side not in {None, "right", "left"}:
        raise RuntimeError(
            f"Unsupported policy handSide={configured_hand_side_raw!r}."
        )
    if configured_hand_side is not None and configured_hand_side != hand_side:
        expected_flag = (
            "add --left-hand"
            if configured_hand_side == "left"
            else "remove --left-hand"
        )
        raise RuntimeError(
            f"Policy config was trained for handSide={configured_hand_side_raw!r}, "
            f"but the controller selected {hand_side!r}; {expected_flag}."
        )
    if not env_cfg.get("useRelativeHandControl", False):
        raise RuntimeError("This controller currently expects relative hand control.")
    if env_cfg.get("useRelativeControl", False):
        raise RuntimeError(
            "Policies whose hand action reference is measured Q are not yet supported."
        )

    policy_is_hand_only = bool(env_cfg.get("controlHandOnly", False))
    policy_num_actions = HAND_DOF if policy_is_hand_only else ARM_DOF + HAND_DOF
    obs_list = list(env_cfg["obsList"])
    episode_length = int(env_cfg.get("episodeLength", 600))
    neutral_hand_policy = DEFAULT_JOINT_POS[ARM_DOF:].copy()
    default_q = DEFAULT_JOINT_POS.copy()
    default_q[:ARM_DOF] = np.asarray(
        env_cfg["defaultArmDofPos"], dtype=np.float32
    )
    arm_home_q = default_q[:ARM_DOF].copy()
    arm_home_real_q = policy_arm_to_real(arm_home_q)

    if policy_is_hand_only:
        if not env_cfg.get("useScriptedArmTrajectory", False):
            raise RuntimeError(
                "A 20-action hand-only policy requires its scripted arm trajectory."
            )
        waypoints = env_cfg["scriptedArmTrajectory"]["waypoints"]
        if len(waypoints) < 4:
            raise RuntimeError("At least four scripted arm waypoints are required.")
        order = np.argsort([int(item["step"]) for item in waypoints])
        waypoint_steps = np.array(
            [int(waypoints[index]["step"]) for index in order],
            dtype=np.int32,
        )
        waypoint_poses = np.array(
            [waypoints[index]["q"] for index in order], dtype=np.float32
        )
    else:
        # Full policies do not use a scripted trajectory. These placeholder
        # values are needed only by the generic phase-observation helper; the
        # MuJoCo arm target is computed from actions[:6] below.
        waypoint_steps = np.array(
            [0, episode_length, episode_length + 1, episode_length + 2],
            dtype=np.int32,
        )
        waypoint_poses = np.repeat(
            default_q[:ARM_DOF][None, :], 4, axis=0
        )

    object_base_size = float(env_cfg.get("objectBaseSize", 0.04))
    keypoint_scale = float(env_cfg.get("keypointScale", 1.0))
    fixed_size = np.asarray(
        env_cfg.get("fixedSize", [0.05, 0.05, 0.05]), dtype=np.float32
    )
    object_scales = fixed_size / object_base_size
    goal_success_tolerance_m = (
        float(args.goal_success_tolerance_m)
        if args.goal_success_tolerance_m is not None
        else configured_goal_success_tolerance(env_cfg) * keypoint_scale
    )
    goal_success_steps = (
        int(args.goal_success_steps)
        if args.goal_success_steps is not None
        else int(env_cfg.get("successSteps", 1))
    )
    force_consecutive_goal_steps = bool(
        env_cfg.get("forceConsecutiveNearGoalSteps", False)
    )
    arm_dof_speed_scale = float(env_cfg["dofSpeedScale"])
    arm_moving_average = float(env_cfg["armMovingAverage"])
    hand_speed_scale = float(env_cfg["handDofSpeedScale"])
    hand_moving_average = float(env_cfg["handMovingAverage"])

    lower_limits, upper_limits = joint_limits_for_hand(hand_side)
    lower_limits = lower_limits.copy()
    upper_limits = upper_limits.copy()
    object_pose_cfg = np.asarray(
        env_cfg.get(
            "objectStartPose", [0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 1.0]
        ),
        dtype=np.float32,
    )
    raw_goal_pose = env_cfg.get("goalObjectPose")
    if isinstance(raw_goal_pose, (list, tuple)) and len(raw_goal_pose) == 7:
        goal_pose_cfg = np.asarray(raw_goal_pose, dtype=np.float32)
    else:
        # Full policies such as train_b63 sample their goal online and retain
        # an unresolved `${eval:"None"}` in the saved YAML. Use the same
        # fixed vertical visualization offset as the arm-only controller.
        goal_pose_cfg = object_pose_cfg.copy()
        goal_pose_cfg[2] += 0.124
    if object_pose_cfg.shape != (7,) or goal_pose_cfg.shape != (7,):
        raise ValueError("objectStartPose and goalObjectPose must have 7 values.")
    object_quat = object_pose_cfg[[6, 3, 4, 5]]
    goal_quat = (
        parse_vector(args.goal_quat_wxyz, 4, "--goal-quat-wxyz")
        if args.goal_quat_wxyz is not None
        else goal_pose_cfg[[6, 3, 4, 5]]
    )
    object_quat /= np.linalg.norm(object_quat)
    goal_quat /= np.linalg.norm(goal_quat)

    sim_config = Ur5eDeltoMujocoConfig(
        enable_viewer=not args.no_viewer,
        hand_side=hand_side,
        initial_joint_pos=default_q,
        object_scales=object_scales,
        object_start_pos=object_pose_cfg[:3],
        object_start_quat_wxyz=object_quat,
        goal_object_start_pos=goal_pose_cfg[:3],
        table_center_z=float(env_cfg.get("tableResetZ", -0.15)),
        table_object_z_offset=float(env_cfg.get("tableObjectZOffset", 0.25)),
        # The URDF base is at y=0. In Isaac Gym the robot base is shifted by
        # robotBaseY, while the table/object workspace is tablePoseDy away
        # from it; only that relative displacement belongs in this model.
        workspace_y=float(env_cfg.get("tablePoseDy", -0.6)),
        goal_object_start_quat_wxyz=goal_quat,
    )
    sim = Ur5eDeltoMujocoSim(sim_config)

    arm_context = None
    arm_command_socket = None
    arm_state_socket = None
    latest_arm_state = None
    latest_arm_state_time = None
    if args.use_real_arm_state:
        with args.low_level_config_path.open(encoding="utf-8") as stream:
            arm_low_level_cfg = json.load(stream)
        (
            arm_context,
            arm_command_socket,
            arm_state_socket,
        ) = make_arm_zmq_sockets(arm_low_level_cfg)
        print(
            "Waiting for UR5 state on "
            f"tcp://127.0.0.1:{arm_low_level_cfg['publisher_port']}..."
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            received_arm_state = receive_latest_robot_state(arm_state_socket)
            if received_arm_state is not None:
                arm_q = np.asarray(
                    received_arm_state.get("Q", []),
                    dtype=np.float32,
                )
                if arm_q.shape[0] >= ARM_DOF and np.all(
                    np.isfinite(arm_q[:ARM_DOF])
                ):
                    latest_arm_state = received_arm_state
                    latest_arm_state_time = time.monotonic()
                    break
            time.sleep(0.01)
        if latest_arm_state is None:
            raise TimeoutError(
                "No valid UR5 state received within 5 seconds. "
                "Is the low-level arm controller running?"
            )
        default_q[:ARM_DOF] = real_arm_to_policy(
            np.asarray(
                latest_arm_state["Q"][:ARM_DOF],
                dtype=np.float32,
            )
        )
        sim.set_robot_joint_positions(default_q)

    bridge = None
    latest_real_state = None
    if args.use_real_hand_state:
        bridge = HandBridgeClient(
            state_address=args.state_address,
            state_port=args.state_port,
            command_address=args.command_address,
            command_port=args.command_port,
        )
        print(
            f"Waiting for hand state on udp://{args.state_address}:{args.state_port}..."
        )
        latest_real_state = wait_for_initial_state(bridge, 5.0)
        default_q[ARM_DOF:] = right_hand_to_policy(
            latest_real_state["q"],
            hand_side,
        )
        sim.set_robot_joint_positions(default_q)

    pose_context = None
    pose_socket = None
    last_pose_time = None
    latest_pose_confidence = None
    object_joint = sim.model.joint("object_free_joint")
    object_qpos_address = int(object_joint.qposadr[0])
    object_dof_address = int(object_joint.dofadr[0])
    goal_body_id = sim.model.body("goal_object").id
    goal_position_delta = (
        goal_pose_cfg[:3] - object_pose_cfg[:3]
    ).astype(np.float64)

    def apply_estimated_object_pose(pose: dict, *, set_goal: bool) -> None:
        from scipy.spatial.transform import Rotation

        position = (
            UR_CONTROLLER_BASE_TO_MUJOCO_BASE_ROTATION @ pose["position"]
        )
        rotation = (
            UR_CONTROLLER_BASE_TO_MUJOCO_BASE_ROTATION @ pose["rotation"]
        )
        quat_xyzw = Rotation.from_matrix(rotation).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        sim.data.qpos[object_qpos_address : object_qpos_address + 3] = position
        sim.data.qpos[
            object_qpos_address + 3 : object_qpos_address + 7
        ] = quat_wxyz
        sim.data.qvel[object_dof_address : object_dof_address + 6] = 0.0
        if set_goal:
            sim.model.body_pos[goal_body_id] = position + goal_position_delta
        mujoco.mj_forward(sim.model, sim.data)

    if args.use_pose_estimation:
        pose_context, pose_socket = make_pose_socket(args.pose_address)
        print(f"Waiting for cube pose on {args.pose_address}...")
        deadline = time.monotonic() + 5.0
        initial_pose = None
        while time.monotonic() < deadline:
            initial_pose = receive_latest_pose(
                pose_socket,
                args.pose_board_id,
                args.pose_min_confidence,
            )
            if initial_pose is not None:
                break
            time.sleep(0.01)
        if initial_pose is None:
            raise TimeoutError("No valid cube pose received within 5 seconds.")
        apply_estimated_object_pose(initial_pose, set_goal=True)
        last_pose_time = time.monotonic()
        latest_pose_confidence = initial_pose["confidence"]

    initial_state = sim.get_sim_state()
    initial_state["joint_positions"] = default_q.copy()
    preview_obs = build_observation(
        sim_state=initial_state,
        obs_list=obs_list,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        prev_targets=default_q,
        object_scales=object_scales,
        object_base_size=object_base_size,
        keypoint_scale=keypoint_scale,
        progress=0,
        waypoint_steps=waypoint_steps,
    )
    player = load_policy(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        num_observations=preview_obs.size,
        num_actions=policy_num_actions,
        device=args.device,
    )

    commanded_components = []
    if args.send_to_arm:
        commanded_components.append("ARM")
    if args.send_to_hand:
        commanded_components.append("HAND")
    if commanded_components:
        mode = "REAL " + " + ".join(commanded_components) + " COMMANDING"
    elif args.use_real_arm_state or args.use_real_hand_state:
        mode = "REAL STATE / DRY RUN"
    else:
        mode = "SIMULATION"
    print(f"Mode:              {mode}")
    print(f"Policy config:     {args.config_path}")
    print(f"Policy checkpoint: {args.checkpoint_path}")
    print(
        "Policy type:       "
        + (
            "20-action hand-only"
            if policy_is_hand_only
            else "26-action full arm + hand policy"
        )
    )
    print(f"Observations:      {preview_obs.size}")
    print(
        f"Policy actions:    {policy_num_actions} "
        f"({HAND_DOF} hand actions used)"
    )
    print(f"Policy rate:       {args.control_hz:g} Hz")
    print(
        f"Arm command stream: {args.arm_command_stream_hz:g} Hz"
        if args.send_to_arm
        else "Arm command stream: disabled"
    )
    print(
        f"Hand command stream: {args.hand_command_stream_hz:g} Hz"
        if args.send_to_hand
        else "Hand command stream: disabled"
    )
    print(
        f"Action integration: {POLICY_ACTION_HZ:g} Hz training timestep "
        "(lower control-hz slows motion)"
    )
    print(f"Action scale:      {args.action_scale:g}")
    print(f"Arm action scale:  {args.arm_action_scale:g}")
    print(f"Target step limit: {args.max_target_step_rad:g} rad")
    print(
        f"Tracking limit:    {args.max_tracking_error_rad:g} rad "
        f"({args.tracking_error_mode})"
    )
    print(
        f"Collision monitor: {args.collision_mode} "
        f"(distance <= {args.collision_distance_threshold_m:g} m; "
        "hand-arm, hand-table, arm-table)"
    )
    if args.send_to_arm:
        print(
            "Arm startup pose:  policy home "
            f"(tolerance {args.arm_home_tolerance_deg:g} deg, "
            f"timeout {args.arm_home_timeout:g} s)"
        )
    if args.send_to_hand:
        print(
            "Startup pose:      neutral 0 rad "
            f"(tolerance {args.neutral_tolerance_rad:g} rad, "
            f"timeout {args.neutral_timeout:g} s)"
        )
    print(
        f"Cube pose:         {args.pose_address} / board {args.pose_board_id}"
        if pose_socket is not None
        else "Cube pose:         MuJoCo simulation"
    )
    print(
        "Goal success:     "
        f"tolerance={goal_success_tolerance_m:.4f} m, "
        f"steps={goal_success_steps}, "
        f"sampling={env_cfg.get('goalSamplingType', 'delta')}"
    )
    print(f"Policy/MuJoCo hand: {hand_side}")
    print(
        "Real wrist_3 offset: "
        f"{np.rad2deg(REAL_ARM_WRIST_3_OFFSET_RAD):+g} deg "
        "(policy/MuJoCo -> real UR5)"
    )
    if hand_side == "left":
        print(
            "Legacy left-policy -> physical right-hand signs: "
            + " ".join(f"{int(value):+d}" for value in LEFT_TO_RIGHT_SIGN)
        )
    else:
        print("Right-policy -> physical right hand: direct joint mapping")

    if (args.send_to_arm or args.send_to_hand) and not args.yes:
        enabled_outputs = " + ".join(commanded_components)
        answer = input(
            f"REAL {enabled_outputs} COMMANDS ARE ENABLED. "
            "Type SEND to start, anything else to abort: "
        )
        if answer.strip() != "SEND":
            print("Aborted; no command sent.")
            player.reset()
            sim.close()
            if bridge is not None:
                bridge.close()
            if pose_socket is not None:
                pose_socket.close()
            if pose_context is not None:
                pose_context.term()
            if arm_command_socket is not None:
                arm_command_socket.close()
            if arm_state_socket is not None:
                arm_state_socket.close()
            if arm_context is not None:
                arm_context.term()
            return

    hand_command_streamer = None
    if args.send_to_hand:
        hand_command_streamer = HandCommandStreamer(
            bridge, args.hand_command_stream_hz
        )
        hand_command_streamer.start()

    arm_command_streamer = None
    if args.send_to_arm:
        arm_command_streamer = ArmCommandStreamer(
            arm_command_socket,
            args.arm_command_stream_hz,
        )
        arm_command_streamer.start()

    previous_targets = default_q.copy()
    latest_real_state_time = (
        latest_real_state["received_at"] if latest_real_state is not None else None
    )
    loop_dt = 1.0 / args.control_hz
    rate = RateLimiter(args.control_hz)
    last_print = 0.0
    last_tracking_warning = 0.0
    last_collision_warning = 0.0
    collision_was_active = False
    progress = 0
    hand_command_was_sent = False
    arm_command_was_sent = False
    near_goal_steps = 0
    goals_reached = 0
    latest_goal_distance = None

    try:
        if args.send_to_arm:
            print("Moving the real UR5 to the policy home pose...")
            # Allow the low-level SUB socket to finish connecting to this PUB.
            time.sleep(0.5)
            arm_command_streamer.set_target(arm_home_real_q)
            arm_command_was_sent = True
            home_deadline = time.monotonic() + args.arm_home_timeout
            home_reached_since = None
            last_home_print = 0.0

            while True:
                now = time.monotonic()
                received_arm_state = receive_latest_robot_state(
                    arm_state_socket
                )
                if received_arm_state is not None:
                    latest_arm_state = received_arm_state
                    latest_arm_state_time = now
                if (
                    latest_arm_state is None
                    or latest_arm_state_time is None
                    or now - latest_arm_state_time > args.arm_state_timeout
                ):
                    raise TimeoutError(
                        "UR5 state became stale while moving to home."
                    )

                measured_arm_q_real = np.asarray(
                    latest_arm_state.get("Q", []),
                    dtype=np.float32,
                )
                if measured_arm_q_real.shape[0] < ARM_DOF or not np.all(
                    np.isfinite(measured_arm_q_real[:ARM_DOF])
                ):
                    raise RuntimeError("Received an invalid UR5 joint state.")
                measured_arm_q_real = measured_arm_q_real[:ARM_DOF]
                home_errors_deg = np.rad2deg(
                    np.abs(arm_home_real_q - measured_arm_q_real)
                )
                max_home_error_deg = float(np.max(home_errors_deg))
                if max_home_error_deg <= args.arm_home_tolerance_deg:
                    if home_reached_since is None:
                        home_reached_since = now
                    elif now - home_reached_since >= 0.2:
                        break
                else:
                    home_reached_since = None

                if now - last_home_print >= 0.5:
                    worst_index = int(np.argmax(home_errors_deg))
                    print(
                        "UR5 home startup: "
                        f"max error={max_home_error_deg:.2f} deg at "
                        f"joint {worst_index + 1}"
                    )
                    last_home_print = now

                if now >= home_deadline:
                    raise TimeoutError(
                        "The UR5 did not reach the policy home pose within "
                        f"{args.arm_home_timeout:g} s. Joint errors: "
                        f"{home_errors_deg.round(2).tolist()} deg"
                    )
                time.sleep(0.01)

            previous_targets[:ARM_DOF] = arm_home_q
            default_q[:ARM_DOF] = arm_home_q
            measured_arm_q = real_arm_to_policy(measured_arm_q_real)
            sim.data.qpos[
                sim._joint_qpos_adrs[:ARM_DOF]
            ] = measured_arm_q
            sim.data.qvel[sim._joint_dof_adrs[:ARM_DOF]] = 0.0
            mujoco.mj_forward(sim.model, sim.data)
            print("UR5 home pose reached.")

        if args.send_to_hand:
            neutral_right = policy_to_right_hand(
                neutral_hand_policy,
                hand_side,
            )
            print(
                "Moving the real hand to the neutral startup pose "
                "(all joints at 0 rad)..."
            )
            hand_command_streamer.set_target(neutral_right)
            hand_command_was_sent = True
            neutral_deadline = time.monotonic() + args.neutral_timeout
            neutral_reached_since = None
            last_neutral_print = 0.0

            while True:
                now = time.monotonic()
                received = bridge.receive_latest_state()
                if received is not None:
                    latest_real_state = received
                    latest_real_state_time = received["received_at"]
                if (
                    latest_real_state is None
                    or latest_real_state_time is None
                    or now - latest_real_state_time > args.state_timeout
                ):
                    raise TimeoutError(
                        "DG5F state became stale while moving to neutral."
                    )

                neutral_errors = np.abs(
                    neutral_right - latest_real_state["q"]
                )
                max_neutral_error = float(np.max(neutral_errors))
                if max_neutral_error <= args.neutral_tolerance_rad:
                    if neutral_reached_since is None:
                        neutral_reached_since = now
                    elif now - neutral_reached_since >= 0.2:
                        break
                else:
                    neutral_reached_since = None

                if now - last_neutral_print >= 0.5:
                    worst_index = int(np.argmax(neutral_errors))
                    print(
                        "Neutral startup: "
                        f"max error={max_neutral_error:.3f} rad at "
                        f"{RIGHT_HAND_JOINT_NAMES[worst_index]} "
                        f"(measured={latest_real_state['q'][worst_index]:.3f} rad)"
                    )
                    last_neutral_print = now

                if now >= neutral_deadline:
                    offending_indices = np.flatnonzero(
                        neutral_errors > args.neutral_tolerance_rad
                    )
                    offending_joints = "; ".join(
                        (
                            f"{RIGHT_HAND_JOINT_NAMES[index]}: "
                            f"error={neutral_errors[index]:.3f} rad, "
                            f"measured={latest_real_state['q'][index]:.3f} rad"
                        )
                        for index in offending_indices
                    )
                    raise TimeoutError(
                        "The hand did not reach the neutral startup pose "
                        f"within {args.neutral_timeout:g} s. "
                        f"Joints outside tolerance: {offending_joints}"
                    )
                time.sleep(0.01)

            measured_hand_policy = right_hand_to_policy(
                latest_real_state["q"],
                hand_side,
            )
            default_q[ARM_DOF:] = neutral_hand_policy
            previous_targets[ARM_DOF:] = neutral_hand_policy
            neutral_full_q = np.concatenate(
                [default_q[:ARM_DOF], measured_hand_policy]
            ).astype(np.float32)
            neutral_full_target = np.concatenate(
                [default_q[:ARM_DOF], neutral_hand_policy]
            ).astype(np.float32)
            sim.set_robot_joint_positions(neutral_full_q)
            sim.set_robot_joint_pos_targets(neutral_full_target)
            print(
                "Neutral startup pose reached. Starting policy inference."
            )

        while args.max_steps <= 0 or progress < args.max_steps:
            if progress >= episode_length:
                print(f"Episode complete at step {progress}.")
                break
            if not args.no_viewer and not sim.viewer.is_running():
                break

            arm_target = (
                scripted_arm_target(
                    progress, waypoint_steps, waypoint_poses
                )
                if policy_is_hand_only
                else previous_targets[:ARM_DOF].copy()
            )
            if pose_socket is not None:
                pose = receive_latest_pose(
                    pose_socket,
                    args.pose_board_id,
                    args.pose_min_confidence,
                )
                if pose is not None:
                    apply_estimated_object_pose(pose, set_goal=False)
                    last_pose_time = time.monotonic()
                    latest_pose_confidence = pose["confidence"]
                if (
                    last_pose_time is None
                    or time.monotonic() - last_pose_time > args.pose_timeout
                ):
                    raise TimeoutError(
                        "Cube pose estimation is stale; stopping policy commands."
                    )

            measured_arm_q = None
            measured_arm_q_real = None
            measured_arm_qd = None
            if arm_state_socket is not None:
                received_arm_state = receive_latest_robot_state(
                    arm_state_socket
                )
                if received_arm_state is not None:
                    latest_arm_state = received_arm_state
                    latest_arm_state_time = time.monotonic()
                if (
                    latest_arm_state is None
                    or latest_arm_state_time is None
                    or time.monotonic() - latest_arm_state_time
                    > args.arm_state_timeout
                ):
                    raise TimeoutError(
                        "UR5 state is stale; stopping all policy commands."
                    )
                measured_arm_q_real = np.asarray(
                    latest_arm_state.get("Q", []),
                    dtype=np.float32,
                )
                measured_arm_qd = np.asarray(
                    latest_arm_state.get("Qd", []),
                    dtype=np.float32,
                )
                if measured_arm_q_real.shape[0] < ARM_DOF or not np.all(
                    np.isfinite(measured_arm_q_real[:ARM_DOF])
                ):
                    raise RuntimeError("Received an invalid UR5 Q state.")
                measured_arm_q_real = measured_arm_q_real[:ARM_DOF]
                measured_arm_q = real_arm_to_policy(measured_arm_q_real)
                if measured_arm_qd.shape[0] < ARM_DOF or not np.all(
                    np.isfinite(measured_arm_qd[:ARM_DOF])
                ):
                    measured_arm_qd = np.zeros(ARM_DOF, dtype=np.float32)
                else:
                    measured_arm_qd = measured_arm_qd[:ARM_DOF]
                sim.data.qpos[
                    sim._joint_qpos_adrs[:ARM_DOF]
                ] = measured_arm_q
                sim.data.qvel[
                    sim._joint_dof_adrs[:ARM_DOF]
                ] = measured_arm_qd
                mujoco.mj_forward(sim.model, sim.data)

            measured_hand_qd_policy = None
            if bridge is not None:
                received = bridge.receive_latest_state()
                if received is not None:
                    latest_real_state = received
                    latest_real_state_time = received["received_at"]
                if (
                    latest_real_state is None
                    or latest_real_state_time is None
                    or time.monotonic() - latest_real_state_time > args.state_timeout
                ):
                    raise TimeoutError(
                        "DG5F state is stale; stopping policy commands."
                    )
                measured_hand_policy = right_hand_to_policy(
                    latest_real_state["q"],
                    hand_side,
                )
                measured_hand_qd_policy = right_hand_to_policy(
                    latest_real_state["qd"],
                    hand_side,
                )
                tracking_errors = np.abs(
                    previous_targets[ARM_DOF:] - measured_hand_policy
                )
                tracking_error = float(np.max(tracking_errors))
                if (
                    args.send_to_hand
                    and tracking_error > args.max_tracking_error_rad
                ):
                    target_right = policy_to_right_hand(
                        previous_targets[ARM_DOF:],
                        hand_side,
                    )
                    measured_right = latest_real_state["q"]
                    offending_indices = np.flatnonzero(
                        tracking_errors > args.max_tracking_error_rad
                    )
                    offending_joints = "; ".join(
                        (
                            f"{RIGHT_HAND_JOINT_NAMES[index]}: "
                            f"error={tracking_errors[index]:.3f} rad, "
                            f"target={target_right[index]:.3f} rad, "
                            f"measured={measured_right[index]:.3f} rad"
                        )
                        for index in offending_indices
                    )
                    message = (
                        "Hand target tracking error exceeded the safety "
                        f"limit: {tracking_error:.3f} > "
                        f"{args.max_tracking_error_rad:.3f} rad. "
                        f"Offending joints: {offending_joints}"
                    )
                    if args.tracking_error_mode == "stop":
                        raise RuntimeError(message)
                    warning_time = time.monotonic()
                    if (
                        warning_time - last_tracking_warning
                        >= TRACKING_WARNING_PERIOD_S
                    ):
                        print_warning(message)
                        last_tracking_warning = warning_time
                if policy_is_hand_only:
                    arm_state_for_sim = (
                        measured_arm_q
                        if measured_arm_q is not None
                        else arm_target
                    )
                    measured_full_q = np.concatenate(
                        [arm_state_for_sim, measured_hand_policy]
                    ).astype(np.float32)
                    sim.set_robot_joint_positions(measured_full_q)
                else:
                    # Preserve the simulated arm's q/qd while synchronizing
                    # only the hand DOFs from the physical right hand.
                    sim.data.qpos[
                        sim._joint_qpos_adrs[ARM_DOF:]
                    ] = measured_hand_policy
                    sim.data.qvel[
                        sim._joint_dof_adrs[ARM_DOF:]
                    ] = measured_hand_qd_policy
                    mujoco.mj_forward(sim.model, sim.data)

            sim_state = sim.get_sim_state()
            if measured_arm_qd is not None:
                sim_state["joint_velocities"][:ARM_DOF] = measured_arm_qd
            if measured_hand_qd_policy is not None:
                sim_state["joint_velocities"][
                    ARM_DOF:
                ] = measured_hand_qd_policy

            latest_goal_distance = keypoint_goal_distance(
                sim_state=sim_state,
                object_scales=object_scales,
                keypoint_scale=keypoint_scale,
                object_base_size=object_base_size,
            )
            near_goal = latest_goal_distance < goal_success_tolerance_m
            if force_consecutive_goal_steps:
                near_goal_steps = near_goal_steps + 1 if near_goal else 0
            elif near_goal:
                near_goal_steps += 1

            if near_goal_steps >= goal_success_steps:
                goals_reached += 1
                print_goal_reached(
                    goal_index=goals_reached,
                    progress=progress,
                    distance_m=latest_goal_distance,
                    tolerance_m=goal_success_tolerance_m,
                )
                next_goal_pos, next_goal_quat_wxyz = sample_next_goal(
                    sim_state=sim_state,
                    env_cfg=env_cfg,
                    workspace_y=sim.config.workspace_y,
                )
                sim.set_goal_object_pose(next_goal_pos, next_goal_quat_wxyz)
                near_goal_steps = 0
                print(
                    "Sampled next goal: "
                    f"position={np.round(next_goal_pos, 3).tolist()}, "
                    f"quaternion_wxyz={np.round(next_goal_quat_wxyz, 3).tolist()}",
                    flush=True,
                )
                sim_state = sim.get_sim_state()
                if measured_arm_qd is not None:
                    sim_state["joint_velocities"][:ARM_DOF] = measured_arm_qd
                if measured_hand_qd_policy is not None:
                    sim_state["joint_velocities"][
                        ARM_DOF:
                    ] = measured_hand_qd_policy

            obs = build_observation(
                sim_state=sim_state,
                obs_list=obs_list,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                prev_targets=previous_targets,
                object_scales=object_scales,
                object_base_size=object_base_size,
                keypoint_scale=keypoint_scale,
                progress=progress,
                waypoint_steps=waypoint_steps,
            )
            obs_tensor = torch.as_tensor(
                obs[None, :], dtype=torch.float32, device=args.device
            )
            with torch.inference_mode():
                policy_action = (
                    player.get_normalized_action(
                        obs_tensor, deterministic_actions=True
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
            if args.zero_action:
                policy_action.fill(0.0)

            if policy_is_hand_only:
                simulated_arm_action = None
                hand_action = policy_action
            else:
                simulated_arm_action = policy_action[:ARM_DOF]
                hand_action = policy_action[ARM_DOF:]
                arm_target = compute_arm_targets(
                    action=simulated_arm_action,
                    previous_targets=previous_targets[:ARM_DOF],
                    lower_limits=lower_limits[:ARM_DOF],
                    upper_limits=upper_limits[:ARM_DOF],
                    dof_speed_scale=arm_dof_speed_scale,
                    moving_average=arm_moving_average,
                    action_scale=args.arm_action_scale,
                )

            hand_target_policy = compute_hand_targets(
                action=hand_action,
                previous_targets=previous_targets[ARM_DOF:],
                lower_limits=lower_limits[ARM_DOF:],
                upper_limits=upper_limits[ARM_DOF:],
                hand_speed_scale=hand_speed_scale,
                moving_average=hand_moving_average,
                action_scale=args.action_scale,
                max_target_step_rad=args.max_target_step_rad,
            )
            full_target = np.concatenate(
                [arm_target, hand_target_policy]
            ).astype(np.float32)
            sim.set_robot_joint_pos_targets(full_target)

            if args.send_to_arm:
                arm_target_real = policy_arm_to_real(arm_target)
                arm_target_errors_deg = np.rad2deg(
                    np.abs(arm_target_real - measured_arm_q_real)
                )
                max_arm_target_error_deg = float(
                    np.max(arm_target_errors_deg)
                )
                if (
                    max_arm_target_error_deg
                    > args.max_arm_target_error_deg
                ):
                    raise RuntimeError(
                        "Arm target tracking error exceeded the safety limit: "
                        f"{max_arm_target_error_deg:.2f} > "
                        f"{args.max_arm_target_error_deg:.2f} deg. "
                        "Per-joint errors: "
                        f"{arm_target_errors_deg.round(2).tolist()} deg"
                    )
                arm_command_streamer.set_target(arm_target_real)
                arm_command_was_sent = True

            target_right = policy_to_right_hand(
                hand_target_policy,
                hand_side,
            )
            if args.send_to_hand:
                hand_command_streamer.set_target(target_right)
                hand_command_was_sent = True

            now = time.monotonic()
            if args.debug_step or now - last_print >= args.print_every:
                measured = sim_state["joint_positions"][ARM_DOF:]
                print(
                    f"step={progress:04d} phase={int(np.argmax(phase_observation(progress, waypoint_steps)))} "
                    f"|hand_action|max={np.max(np.abs(hand_action)):.3f} "
                    + (
                        f"sim_arm_action={np.round(simulated_arm_action, 3).tolist()} "
                        f"sim_arm_q={np.round(sim_state['joint_positions'][:ARM_DOF], 3).tolist()} "
                        f"sim_arm_target={np.round(arm_target, 3).tolist()} "
                        if simulated_arm_action is not None
                        else ""
                    )
                    + f"|target-q|max={np.max(np.abs(hand_target_policy - measured)):.3f} "
                    + (
                        f"confidence={latest_pose_confidence:.3f} "
                        if latest_pose_confidence is not None
                        else ""
                    )
                    + (
                        f"goal_dist={latest_goal_distance:.4f} "
                        if latest_goal_distance is not None
                        else ""
                    )
                    + f"right_target={np.round(target_right, 3).tolist()}"
                )
                last_print = now

            if args.debug_step:
                answer = input("Enter/Space: next step; q: stop > ")
                if answer.strip().lower() == "q":
                    break

            sim.step_for(POLICY_ACTION_DT)
            if not policy_is_hand_only and measured_arm_q is None:
                # Match ur5_policy_arm_controller.py's ideal simulation mode:
                # when no real arm state is selected, represent the arm at the
                # policy target rather than allowing simulator sag.
                simulated_arm_velocity = (
                    arm_target - previous_targets[:ARM_DOF]
                ) / loop_dt
                sim.data.qpos[
                    sim._joint_qpos_adrs[:ARM_DOF]
                ] = arm_target
                sim.data.qvel[
                    sim._joint_dof_adrs[:ARM_DOF]
                ] = simulated_arm_velocity
                mujoco.mj_forward(sim.model, sim.data)
                if not args.no_viewer:
                    sim.viewer.sync()

            if args.collision_mode != "off":
                collisions = monitored_collisions(
                    mujoco_module=mujoco,
                    model=sim.model,
                    data=sim.data,
                    distance_threshold_m=args.collision_distance_threshold_m,
                    hand_side=hand_side,
                )
                if collisions:
                    collision_message = (
                        "MuJoCo monitored collision: "
                        + format_collisions(collisions)
                    )
                    if args.collision_mode == "stop":
                        raise RuntimeError(collision_message)
                    collision_time = time.monotonic()
                    if (
                        not collision_was_active
                        or collision_time - last_collision_warning
                        >= args.collision_warning_period
                    ):
                        print_warning(
                            collision_message,
                            category="collision",
                        )
                        last_collision_warning = collision_time
                    collision_was_active = True
                else:
                    if collision_was_active:
                        print(
                            "MuJoCo monitored collision cleared.",
                            flush=True,
                        )
                    collision_was_active = False
            previous_targets = full_target
            progress += 1
            rate.sleep()
    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    finally:
        if hand_command_streamer is not None:
            hand_command_streamer.stop()
        if arm_command_streamer is not None:
            arm_command_streamer.stop()
        if (
            args.send_to_arm
            and arm_command_was_sent
            and arm_command_socket is not None
            and arm_state_socket is not None
        ):
            print("Requesting zero UR5 joint velocity before shutdown...")
            send_zero_velocity_hold(
                arm_command_socket,
                arm_state_socket,
                latest_arm_state,
            )
        if (
            bridge is not None
            and args.send_to_hand
            and hand_command_was_sent
        ):
            print("Holding the latest measured hand position before shutdown...")
            deadline = time.monotonic() + STOP_HOLD_DURATION_S
            while time.monotonic() < deadline:
                state = bridge.receive_latest_state()
                if state is not None:
                    latest_real_state = state
                if latest_real_state is not None:
                    bridge.send_target(latest_real_state["q"])
                time.sleep(1.0 / STOP_HOLD_HZ)
        player.reset()
        sim.close()
        if bridge is not None:
            bridge.close()
        if pose_socket is not None:
            pose_socket.close()
        if pose_context is not None:
            pose_context.term()
        if arm_command_socket is not None:
            arm_command_socket.close()
        if arm_state_socket is not None:
            arm_state_socket.close()
        if arm_context is not None:
            arm_context.term()


if __name__ == "__main__":
    main()
