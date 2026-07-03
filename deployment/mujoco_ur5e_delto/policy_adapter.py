from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation


N_OBS = 131
N_ACT = 26
ARM_DOF = 6
OBJECT_KEYPOINT_BASE_SIZE = 0.04
OBJECT_KEYPOINT_SCALE = 1.5
HandSide = Literal["right", "left"]
DEFAULT_HAND_SIDE: HandSide = "right"

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ARM_LOWER_LIMITS = np.array(
    [
        -2 * math.pi,
        -2 * math.pi,
        -math.pi,
        -2 * math.pi,
        -2 * math.pi,
        -2 * math.pi,
    ],
    dtype=np.float32,
)

ARM_UPPER_LIMITS = np.array(
    [
        2 * math.pi,
        2 * math.pi,
        math.pi,
        2 * math.pi,
        2 * math.pi,
        2 * math.pi,
    ],
    dtype=np.float32,
)


def validate_hand_side(hand_side: str) -> HandSide:
    normalized = hand_side.lower()
    if normalized not in {"right", "left"}:
        raise ValueError(
            f"hand_side must be 'right' or 'left', got {hand_side!r}"
        )
    return normalized


def robot_urdf_path_for_hand(hand_side: HandSide = DEFAULT_HAND_SIDE) -> Path:
    side = validate_hand_side(hand_side)
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "assets"
        / "urdf"
        / "ur5e_delto_description"
        / f"ur5e_{side}_dg5f.urdf"
    )


def joint_names_for_hand(hand_side: HandSide = DEFAULT_HAND_SIDE) -> list[str]:
    side = validate_hand_side(hand_side)
    joint_prefix = "rj" if side == "right" else "lj"
    hand_joint_names = [
        f"{joint_prefix}_dg_{finger}_{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    ]
    return ARM_JOINT_NAMES + hand_joint_names


def fingertip_body_names_for_hand(
    hand_side: HandSide = DEFAULT_HAND_SIDE,
) -> list[str]:
    side = validate_hand_side(hand_side)
    link_prefix = "rl" if side == "right" else "ll"
    return [f"{link_prefix}_dg_{finger}_4" for finger in range(1, 6)]


def fingertip_local_offsets_for_hand(
    hand_side: HandSide = DEFAULT_HAND_SIDE,
) -> np.ndarray:
    side = validate_hand_side(hand_side)
    thumb_y = 0.0363 if side == "right" else -0.0363
    return np.array(
        [
            [0.0, thumb_y, 0.0],
            [0.0, 0.0, 0.0255],
            [0.0, 0.0, 0.0255],
            [0.0, 0.0, 0.0255],
            [0.0, 0.0, 0.0363],
        ],
        dtype=np.float32,
    )


@lru_cache(maxsize=2)
def joint_limits_for_hand(
    hand_side: HandSide = DEFAULT_HAND_SIDE,
) -> tuple[np.ndarray, np.ndarray]:
    side = validate_hand_side(hand_side)
    root = ET.parse(robot_urdf_path_for_hand(side)).getroot()
    limits_by_joint = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and joint.get("type") != "fixed":
            limits_by_joint[joint.get("name")] = (
                float(limit.get("lower")),
                float(limit.get("upper")),
            )

    hand_names = joint_names_for_hand(side)[ARM_DOF:]
    hand_limits = [limits_by_joint[name] for name in hand_names]
    lower = np.concatenate(
        [ARM_LOWER_LIMITS, np.array([limit[0] for limit in hand_limits])]
    ).astype(np.float32)
    upper = np.concatenate(
        [ARM_UPPER_LIMITS, np.array([limit[1] for limit in hand_limits])]
    ).astype(np.float32)
    return lower, upper


JOINT_NAMES = joint_names_for_hand()
FINGERTIP_BODY_NAMES = fingertip_body_names_for_hand()
FINGERTIP_LOCAL_OFFSETS = fingertip_local_offsets_for_hand()
LOWER_LIMITS, UPPER_LIMITS = joint_limits_for_hand()

DEFAULT_JOINT_POS = np.array(
    [
        -1.5708,
        -1.0,
        2.0,
        -1.0,
        1.571,
        -1.571,
        *([0.0] * 20),
    ],
    dtype=np.float32,
)

OBJECT_KEYPOINT_SIGNS = np.array(
    [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=np.float32
)


def read_policy_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rl_player(simtoolreal_root: Path):
    import sys

    sys.path.insert(0, str(simtoolreal_root))
    sys.path.insert(0, str(simtoolreal_root / "rl_games"))
    from deployment.rl_player import RlPlayer

    return RlPlayer


def create_rl_player(
    *,
    simtoolreal_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: str,
):
    rl_player_cls = load_rl_player(simtoolreal_root)
    original_torch_load = torch.load
    if device == "cpu":

        def torch_load_on_cpu(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device("cpu"))
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_on_cpu
    try:
        return rl_player_cls(
            num_observations=N_OBS,
            num_actions=N_ACT,
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            device=device,
        )
    finally:
        torch.load = original_torch_load


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_wxyz[[1, 2, 3, 0]]


def unscale(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (2.0 * x - upper - lower) / (upper - lower)


def scale(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def object_keypoints(
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
    object_scales: np.ndarray,
    *,
    object_base_size: float = OBJECT_KEYPOINT_BASE_SIZE,
    keypoint_scale: float = OBJECT_KEYPOINT_SCALE,
) -> np.ndarray:
    rot = Rotation.from_quat(quat_xyzw)
    offsets = (
        OBJECT_KEYPOINT_SIGNS
        * object_base_size
        * keypoint_scale
        / 2.0
        * object_scales
    )
    return pos[None, :] + rot.apply(offsets)


def build_observation(
    *,
    sim_state: dict[str, np.ndarray],
    object_scales: np.ndarray,
    obs_list: list[str],
    prev_targets: Optional[np.ndarray],
    hand_side: HandSide = DEFAULT_HAND_SIDE,
) -> np.ndarray:
    lower_limits, upper_limits = joint_limits_for_hand(hand_side)
    q = sim_state["joint_positions"].astype(np.float32)
    qd = sim_state["joint_velocities"].astype(np.float32)
    palm_pos = sim_state["palm_pos"].astype(np.float32)
    palm_quat_xyzw = quat_wxyz_to_xyzw(sim_state["palm_quat_wxyz"]).astype(np.float32)
    fingertip_rel_palm = (
        sim_state["fingertip_positions"].astype(np.float32) - palm_pos[None, :]
    )

    object_pos = sim_state["object_pos"].astype(np.float32)
    object_quat_xyzw = quat_wxyz_to_xyzw(sim_state["object_quat_wxyz"]).astype(
        np.float32
    )
    goal_pos = sim_state["goal_object_pos"].astype(np.float32)
    goal_quat_xyzw = quat_wxyz_to_xyzw(sim_state["goal_object_quat_wxyz"]).astype(
        np.float32
    )

    object_kps = object_keypoints(object_pos, object_quat_xyzw, object_scales)
    goal_kps = object_keypoints(goal_pos, goal_quat_xyzw, object_scales)
    targets = prev_targets if prev_targets is not None else q

    obs_dict = {
        "joint_pos": unscale(q, lower_limits, upper_limits),
        "joint_vel": qd,
        "prev_action_targets": targets.astype(np.float32),
        "palm_pos": palm_pos,
        "palm_rot": palm_quat_xyzw,
        "object_rot": object_quat_xyzw,
        "fingertip_pos_rel_palm": fingertip_rel_palm.reshape(-1),
        "keypoints_rel_palm": (object_kps - palm_pos[None, :]).reshape(-1),
        "keypoints_rel_goal": (object_kps - goal_kps).reshape(-1),
        "object_scales": object_scales.astype(np.float32),
    }
    obs = np.concatenate([obs_dict[name].reshape(-1) for name in obs_list])
    if obs.shape != (N_OBS,):
        raise RuntimeError(f"Observation shape {obs.shape} does not match {(N_OBS,)}")
    return obs.astype(np.float32)[None, :]


def compute_targets(
    *,
    actions: np.ndarray,
    q: np.ndarray,
    prev_targets: Optional[np.ndarray],
    control_dt: float,
    dof_speed_scale: float,
    arm_moving_average: float,
    hand_moving_average: float,
    hand_side: HandSide = DEFAULT_HAND_SIDE,
) -> np.ndarray:
    lower_limits, upper_limits = joint_limits_for_hand(hand_side)
    if actions.shape != (N_ACT,):
        raise ValueError(f"actions.shape={actions.shape}, expected {(N_ACT,)}")
    prev = prev_targets if prev_targets is not None else q
    targets = prev.copy()
    targets[:ARM_DOF] = (
        prev[:ARM_DOF] + dof_speed_scale * control_dt * actions[:ARM_DOF]
    )
    targets[:ARM_DOF] = np.clip(
        targets[:ARM_DOF], lower_limits[:ARM_DOF], upper_limits[:ARM_DOF]
    )
    targets[:ARM_DOF] = (
        arm_moving_average * targets[:ARM_DOF]
        + (1.0 - arm_moving_average) * prev[:ARM_DOF]
    )
    targets[ARM_DOF:] = scale(
        actions[ARM_DOF:], lower_limits[ARM_DOF:], upper_limits[ARM_DOF:]
    )
    targets[ARM_DOF:] = (
        hand_moving_average * targets[ARM_DOF:]
        + (1.0 - hand_moving_average) * prev[ARM_DOF:]
    )
    return np.clip(targets, lower_limits, upper_limits).astype(np.float32)
