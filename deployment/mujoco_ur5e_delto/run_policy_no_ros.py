from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from deployment.mujoco_ur5e_delto.mujoco_sim import (
    Ur5eDeltoMujocoConfig,
    Ur5eDeltoMujocoSim,
)
from deployment.mujoco_ur5e_delto.policy_adapter import (
    DEFAULT_JOINT_POS,
    N_ACT,
    N_OBS,
    build_observation,
    compute_targets,
    create_rl_player,
    read_policy_cfg,
)
from isaacgymenvs.utils.utils import get_repo_root_dir


@dataclass
class Args:
    config_path: Path
    """Path to the SimToolReal policy config.yaml."""

    checkpoint_path: Path
    """Path to the SimToolReal policy checkpoint model.pth."""

    object_name: str = "cube"
    """Object primitive to spawn: cube or hammer."""

    scene_height: str = "default"
    """Scene height preset: default, train7, from-config, or high_table."""

    device: str = "cpu"
    """Torch device for policy inference: cpu or cuda."""

    enable_viewer: bool = True
    """Open the passive MuJoCo viewer."""

    max_steps: int = 0
    """Maximum policy steps to run. Use 0 to run forever."""

    policy_start_delay_sec: float = 0.0
    """Seconds to wait before starting policy inference."""

    wait_for_enter: bool = False
    """Wait for Enter in the terminal before starting policy inference."""


def object_scales_for(name: str) -> np.ndarray:
    if name == "hammer":
        return np.array([0.141, 0.03025, 0.0271], dtype=np.float32)
    if name == "cube":
        return np.array([1.25, 1.25, 1.25], dtype=np.float32)
    raise ValueError(f"Unsupported object_name={name!r}; expected 'cube' or 'hammer'")


def scene_config_for(scene_height: str, env_cfg: dict) -> dict:
    initial_joint_pos = DEFAULT_JOINT_POS.copy()
    if "defaultArmDofPos" in env_cfg:
        initial_joint_pos[:6] = np.array(env_cfg["defaultArmDofPos"], dtype=np.float32)

    if scene_height == "high_table":
        table_center_z = 0.38
        table_size_x = 0.475
        table_size_y = 0.4
    elif scene_height in {"default", "train7", "from-config"}:
        table_center_z = float(env_cfg.get("tableResetZ", -0.125))
        table_size_x = 0.8
        table_size_y = 0.8
    else:
        raise ValueError(
            f"Unsupported scene_height={scene_height!r}; "
            "expected 'default', 'train7', 'from-config', or 'high_table'"
        )

    return {
        "table_center_z": table_center_z,
        "table_object_z_offset": float(env_cfg.get("tableObjectZOffset", 0.25)),
        "table_size_x": table_size_x,
        "table_size_y": table_size_y,
        "initial_joint_pos": initial_joint_pos,
    }


def main() -> None:
    args = tyro.cli(Args)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    cfg = read_policy_cfg(args.config_path)
    num_obs = int(cfg["task"]["env"].get("numObservations", N_OBS) or N_OBS)
    num_act = int(cfg["task"]["env"].get("numActions", N_ACT) or N_ACT)
    if (num_obs, num_act) != (N_OBS, N_ACT):
        raise RuntimeError(
            f"This runner expects a UR5e+Delto policy with {N_OBS}/{N_ACT} "
            f"obs/actions, got {num_obs}/{num_act} from {args.config_path}"
        )

    env_cfg = cfg["task"]["env"]
    control_dt = 1.0 / 60.0
    object_scales = object_scales_for(args.object_name)
    scene_cfg = scene_config_for(args.scene_height, env_cfg)
    sim = Ur5eDeltoMujocoSim(
        Ur5eDeltoMujocoConfig(
            enable_viewer=args.enable_viewer,
            object_name=args.object_name,
            object_scales=object_scales,
            **scene_cfg,
        )
    )
    player = create_rl_player(
        simtoolreal_root=get_repo_root_dir(),
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )

    prev_targets = None
    obs_list = env_cfg["obsList"]
    hand_moving_average = float(env_cfg["handMovingAverage"])
    arm_moving_average = float(env_cfg["armMovingAverage"])
    dof_speed_scale = float(env_cfg["dofSpeedScale"])

    step = 0
    print(
        "Running MuJoCo UR5e+Delto policy: "
        f"object={args.object_name}, scene-height={args.scene_height}, "
        f"device={args.device}, obs/actions={N_OBS}/{N_ACT}"
    )
    if args.policy_start_delay_sec > 0:
        print(f"Waiting {args.policy_start_delay_sec:.1f}s before starting the policy")
        delay_start = time.time()
        while time.time() - delay_start < args.policy_start_delay_sec:
            sim.step_for(control_dt)
            time.sleep(control_dt)

    if args.wait_for_enter:
        input("Press Enter to start the policy...")

    try:
        while args.max_steps <= 0 or step < args.max_steps:
            start = time.time()
            sim_state = sim.get_sim_state()
            obs = build_observation(
                sim_state=sim_state,
                object_scales=object_scales,
                obs_list=obs_list,
                prev_targets=prev_targets,
            )
            obs_t = torch.from_numpy(obs).float().to(args.device)
            with torch.no_grad():
                action = player.get_normalized_action(obs_t, deterministic_actions=True)
            q = sim_state["joint_positions"]
            targets = compute_targets(
                actions=action.cpu().numpy()[0],
                q=q,
                prev_targets=prev_targets,
                control_dt=control_dt,
                dof_speed_scale=dof_speed_scale,
                arm_moving_average=arm_moving_average,
                hand_moving_average=hand_moving_average,
            )
            sim.set_robot_joint_pos_targets(targets)
            prev_targets = targets
            sim.step_for(control_dt)

            elapsed = time.time() - start
            sleep_dt = control_dt - elapsed
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                print(
                    f"Policy loop slower than real time: "
                    f"{1.0 / max(elapsed, 1e-6):.1f} Hz"
                )
            step += 1
    finally:
        sim.close()


if __name__ == "__main__":
    main()
