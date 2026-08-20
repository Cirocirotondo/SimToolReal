#!/usr/bin/env python3
"""Replay a 60 Hz joint demo through the training environment's env.step()."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Sequence


def _configure_isaac_gym_graphics_environment() -> None:
    """Prevent ROS/Gazebo graphics libraries from shadowing NVIDIA libraries."""
    if os.environ.get("ISAACGYM_PRESERVE_ROS_GRAPHICS_PATH") != "1":
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

    nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    if os.path.isfile(nvidia_icd):
        os.environ.setdefault("VK_ICD_FILENAMES", nvidia_icd)
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


_configure_isaac_gym_graphics_environment()

# Isaac Gym must be imported before PyTorch.
from isaacgym import gymapi, gymtorch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import torch  # noqa: E402

from deployment.isaac.isaac_env import create_env_from_cfg  # noqa: E402
from isaacgymenvs.tasks.simtoolreal.env_motion_imitation_llcfix import (  # noqa: E402
    SimToolRealMotionImitationLLCFix00,
)
from isaacgymenvs.utils.torch_jit_utils import scale, unscale  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
HYDRA_CONFIG_DIR = REPO_ROOT / "isaacgymenvs" / "cfg"
DEFAULT_DEMO_PATH = (
    HERE
    / "demonstrations_good"
    / "demo_synthetic_arm_hand_all_joints_60hz.npz"
)
TASK_NAME = "SimToolRealMotionImitationLLCFix00"
TRAIN_CONFIG_NAME = "SimToolRealMotionImitationLLCFix00PPO"
TARGET_HZ = 60.0


def _validate_demo(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Demonstration not found: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "timestamp",
            "monotonic_timestamp",
            "arm_q",
            "arm_dq",
            "hand_q_measured",
            "hand_dq_measured",
        }
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(f"{path.name} is missing fields: {missing}")
        sample_count = len(arrays["monotonic_timestamp"])
        expected_shapes = {
            "arm_q": (sample_count, 6),
            "arm_dq": (sample_count, 6),
            "hand_q_measured": (sample_count, 20),
            "hand_dq_measured": (sample_count, 20),
        }
        for name, expected_shape in expected_shapes.items():
            values = np.asarray(arrays[name])
            if values.shape != expected_shape:
                raise ValueError(
                    f"Expected {name} shape {expected_shape}, got {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")
        timestamps = np.asarray(
            arrays["monotonic_timestamp"], dtype=np.float64
        )

    if sample_count < 2 or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("The demonstration timestamps must be increasing")
    nominal_hz = 1.0 / float(np.median(np.diff(timestamps)))
    if not np.isclose(nominal_hz, TARGET_HZ, rtol=0.0, atol=1.0e-3):
        raise ValueError(
            f"Expected a {TARGET_HZ:g} Hz demonstration, got {nominal_hz:.6f} Hz"
        )
    return sample_count


def create_rl_environment(
    demo_path: Path,
    *,
    sample_count: int,
    device: str,
    headless: bool,
) -> SimToolRealMotionImitationLLCFix00:
    """Compose LLCFix00 and create the same task class used by training."""
    with initialize_config_dir(
        version_base="1.1",
        config_dir=str(HYDRA_CONFIG_DIR),
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task={TASK_NAME}",
                f"train={TRAIN_CONFIG_NAME}",
            ],
        )

    OmegaConf.set_struct(cfg, False)
    cfg.task.env.demonstration = str(demo_path)
    cfg.task.env.episodeLength = sample_count
    cfg.task.env.capture_video = False
    cfg.task.env.enableCameraSensors = False
    cfg.task.env.visualizeReferenceRobotInVideo = False
    cfg.task.env.referenceVisualizationActorEnabled = False
    cfg.task.env.useReferenceStateInitialization = False
    cfg.task.env.referenceStateInitProbability = 0.0
    cfg.task.env.imitationEarlyTermination = False
    cfg.task.env.randomize = False
    cfg.task.env.useActionDelay = False
    cfg.sim_device = device
    cfg.rl_device = device
    cfg.pipeline = "cpu" if device == "cpu" else "gpu"

    env = create_env_from_cfg(
        cfg=cfg,
        headless=headless,
        enable_viewer_sync_at_start=not headless,
        episode_length=sample_count,
    )
    if not isinstance(env, SimToolRealMotionImitationLLCFix00):
        raise TypeError(
            f"Expected {TASK_NAME}, got {type(env).__name__}"
        )
    if env.num_envs != 1:
        raise RuntimeError(f"Expected one environment, got {env.num_envs}")
    if env.reference.sample_count != sample_count:
        raise RuntimeError(
            f"Environment loaded {env.reference.sample_count} reference samples, "
            f"expected {sample_count}"
        )
    return env


def reset_to_first_reference(env: SimToolRealMotionImitationLLCFix00) -> None:
    """Place the controlled robot exactly at reference sample zero."""
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    env.reset_idx(env_ids, tensor_reset=True)
    env.set_reference_phase(env_ids, 0.0, flush=False)
    env.set_actor_root_state_tensor_indexed()
    env.set_dof_state_tensor_indexed()
    env.gym.set_dof_position_target_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.cur_targets),
    )

    # Flush indexed state writes and refresh the tensors without advancing the
    # task reference/progress counters.
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.populate_sim_buffers()
    env.current_reference = env.reference.sample(env.reference_index)
    env.populate_obs_and_states_buffers()
    env.clamp_obs()
    env.progress_buf.zero_()
    env.reset_buf.zero_()


def next_reference_action(
    env: SimToolRealMotionImitationLLCFix00,
    reference_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next demo angles and their normalized LLCFix00 network action."""
    indices = torch.full(
        (env.num_envs,),
        reference_index,
        dtype=torch.long,
        device=env.device,
    )
    reference = env.reference.sample(indices)
    joint_angles = torch.cat((reference.arm_q, reference.hand_q), dim=-1)
    actions = unscale(
        joint_angles,
        env.arm_hand_dof_lower_limits,
        env.arm_hand_dof_upper_limits,
    ).clamp(-1.0, 1.0)

    reconstructed_angles = scale(
        actions,
        env.arm_hand_dof_lower_limits,
        env.arm_hand_dof_upper_limits,
    )
    if not torch.allclose(
        reconstructed_angles,
        joint_angles,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            f"Reference sample {reference_index} cannot be represented by "
            "normalized actions within [-1, 1]"
        )
    return actions, joint_angles


def replay_with_rl_step(
    env: SimToolRealMotionImitationLLCFix00,
    *,
    print_every: int,
) -> None:
    """Feed reference sample i+1 to the real training-time env.step()."""
    if print_every < 0:
        raise ValueError("print_every must be non-negative")

    reset_to_first_reference(env)
    arm_peak_error = 0.0
    hand_peak_error = 0.0
    reward_sum = 0.0
    step_count = 0

    for reference_index in range(1, env.reference.sample_count):
        actions, target_angles = next_reference_action(env, reference_index)
        _obs, reward, done, extras = env.step(actions)
        del _obs, extras

        actual_angles = env.arm_hand_dof_pos[:, : env.num_hand_arm_dofs]
        errors = torch.abs(actual_angles - target_angles)
        arm_error = float(torch.amax(errors[:, : env.num_arm_dofs]).item())
        hand_error = float(torch.amax(errors[:, env.num_arm_dofs :]).item())
        arm_peak_error = max(arm_peak_error, arm_error)
        hand_peak_error = max(hand_peak_error, hand_error)
        reward_sum += float(reward[0].item())
        step_count += 1

        if print_every and (
            step_count == 1
            or step_count % print_every == 0
            or reference_index == env.reference.last_index
        ):
            print(
                f"step={step_count:04d}/{env.reference.last_index:04d} "
                f"reference_index={reference_index:04d} "
                f"reward={float(reward[0].item()):.6f} "
                f"arm_max_error={arm_error:.6f} rad "
                f"hand_max_error={hand_error:.6f} rad",
                flush=True,
            )

        expected_done = reference_index == env.reference.last_index
        actual_done = bool(done[0].item())
        if actual_done and not expected_done:
            raise RuntimeError(
                f"Environment terminated unexpectedly at reference sample "
                f"{reference_index}"
            )

    print(
        f"Completed {step_count} env.step(actions) calls; "
        f"mean_reward={reward_sum / step_count:.6f}, "
        f"arm_peak_error={arm_peak_error:.6f} rad, "
        f"hand_peak_error={hand_peak_error:.6f} rad",
        flush=True,
    )


def destroy_environment(env: SimToolRealMotionImitationLLCFix00) -> None:
    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
        env.viewer = None
    env.gym.destroy_sim(env.sim)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "demo_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DEMO_PATH,
        help=f"Processed 60 Hz demo. Default: {DEFAULT_DEMO_PATH}",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run the RL environment and PhysX on CPU.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the Isaac Gym viewer.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=60,
        help="Print tracking diagnostics every N env.step() calls; 0 disables.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    demo_path = args.demo_path.expanduser().resolve()
    sample_count = _validate_demo(demo_path)
    device = "cpu" if args.cpu else "cuda:0"
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu to run on CPU")

    env = create_rl_environment(
        demo_path,
        sample_count=sample_count,
        device=device,
        headless=args.headless,
    )
    try:
        print(
            f"Replaying {demo_path} through {type(env).__name__}.step(actions) "
            f"({sample_count} samples at {TARGET_HZ:.0f} Hz)",
            flush=True,
        )
        replay_with_rl_step(env, print_every=args.print_every)
    finally:
        destroy_environment(env)


if __name__ == "__main__":
    main()
