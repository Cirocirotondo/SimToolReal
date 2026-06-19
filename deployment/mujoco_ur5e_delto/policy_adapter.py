from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation


N_OBS = 131
N_ACT = 26
ARM_DOF = 6
OBJECT_KEYPOINT_BASE_SIZE = 0.04
OBJECT_KEYPOINT_SCALE = 1.5

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "lj_dg_1_1",
    "lj_dg_1_2",
    "lj_dg_1_3",
    "lj_dg_1_4",
    "lj_dg_2_1",
    "lj_dg_2_2",
    "lj_dg_2_3",
    "lj_dg_2_4",
    "lj_dg_3_1",
    "lj_dg_3_2",
    "lj_dg_3_3",
    "lj_dg_3_4",
    "lj_dg_4_1",
    "lj_dg_4_2",
    "lj_dg_4_3",
    "lj_dg_4_4",
    "lj_dg_5_1",
    "lj_dg_5_2",
    "lj_dg_5_3",
    "lj_dg_5_4",
]

FINGERTIP_BODY_NAMES = [
    "ll_dg_1_4",
    "ll_dg_2_4",
    "ll_dg_3_4",
    "ll_dg_4_4",
    "ll_dg_5_4",
]

FINGERTIP_LOCAL_OFFSETS = np.array(
    [
        [0.0, 0.0363, 0.0],
        [0.0, 0.0, 0.0255],
        [0.0, 0.0, 0.0255],
        [0.0, 0.0, 0.0255],
        [0.0, 0.0, 0.0255],
    ],
    dtype=np.float32,
)

LOWER_LIMITS = np.array(
    [
        -2 * math.pi,
        -2 * math.pi,
        -math.pi,
        -2 * math.pi,
        -2 * math.pi,
        -2 * math.pi,
        -0.8901179185171081,
        -0.0,
        -math.pi / 2,
        -math.pi / 2,
        -0.6108652381980153,
        -0.0,
        0.0,
        0.0,
        -0.6108652381980153,
        0.0,
        0.0,
        0.0,
        -0.4188790204786391,
        0.0,
        0.0,
        0.0,
        -1.0471975511965976,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

UPPER_LIMITS = np.array(
    [
        2 * math.pi,
        2 * math.pi,
        math.pi,
        2 * math.pi,
        2 * math.pi,
        2 * math.pi,
        0.3839724354387525,
        math.pi,
        0.0,
        0.0,
        0.4188790204786391,
        2.007128639793479,
        math.pi / 2,
        math.pi / 2,
        0.6108652381980153,
        1.9547687622336491,
        math.pi / 2,
        math.pi / 2,
        0.6108652381980153,
        1.9024088846738192,
        math.pi / 2,
        math.pi / 2,
        0.017453292519943295,
        0.4188790204786391,
        math.pi / 2,
        math.pi / 2,
    ],
    dtype=np.float32,
)

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
    pos: np.ndarray, quat_xyzw: np.ndarray, object_scales: np.ndarray
) -> np.ndarray:
    rot = Rotation.from_quat(quat_xyzw)
    offsets = (
        OBJECT_KEYPOINT_SIGNS
        * OBJECT_KEYPOINT_BASE_SIZE
        * OBJECT_KEYPOINT_SCALE
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
) -> np.ndarray:
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
        "joint_pos": unscale(q, LOWER_LIMITS, UPPER_LIMITS),
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
) -> np.ndarray:
    if actions.shape != (N_ACT,):
        raise ValueError(f"actions.shape={actions.shape}, expected {(N_ACT,)}")
    prev = prev_targets if prev_targets is not None else q
    targets = prev.copy()
    targets[:ARM_DOF] = (
        prev[:ARM_DOF] + dof_speed_scale * control_dt * actions[:ARM_DOF]
    )
    targets[:ARM_DOF] = np.clip(
        targets[:ARM_DOF], LOWER_LIMITS[:ARM_DOF], UPPER_LIMITS[:ARM_DOF]
    )
    targets[:ARM_DOF] = (
        arm_moving_average * targets[:ARM_DOF]
        + (1.0 - arm_moving_average) * prev[:ARM_DOF]
    )
    targets[ARM_DOF:] = scale(actions[ARM_DOF:], LOWER_LIMITS[ARM_DOF:], UPPER_LIMITS[ARM_DOF:])
    targets[ARM_DOF:] = (
        hand_moving_average * targets[ARM_DOF:]
        + (1.0 - hand_moving_average) * prev[ARM_DOF:]
    )
    return np.clip(targets, LOWER_LIMITS, UPPER_LIMITS).astype(np.float32)
