"""Headless, deterministic motion-imitation evaluation for training callbacks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from dextoolbench.interactive_eval_common import (
    checkpoint_payload,
    install_path_is_relative_to_backport,
)

install_path_is_relative_to_backport()


def _uses_sapg_exploration_observation(config_path: str) -> bool:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    expl_type = (
        config.get("train", {})
        .get("params", {})
        .get("config", {})
        .get("expl_type", "none")
    )
    return str(expl_type).startswith("mixed_expl")


def _checkpoint_env_state(checkpoint) -> Optional[Dict[str, Any]]:
    env_state = checkpoint_payload(checkpoint).get("env_state")
    return env_state if isinstance(env_state, dict) else None


def _scalar(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _component(extras: Dict[str, Any], name: str) -> float:
    value = extras.get("episode_cumulative", {}).get(name)
    if value is None:
        return 0.0
    return _scalar(value[0])


def _reset_at_phase_zero(env):
    import torch
    from isaacgym import gymtorch

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    env.reset_idx(env_ids, tensor_reset=True)
    # Isaac Gym tensor setters must be called at most once between simulation
    # steps. Queue the main and reference actors, then apply both DOF states in
    # one indexed call; otherwise the second setter (green robot) replaces the
    # pending phase-zero reset of the controlled robot.
    env.set_reference_phase(env_ids, 0.0, flush=False)
    env.set_actor_root_state_tensor_indexed()
    env._set_reference_visualization_robot(
        env.current_reference,
        flush=False,
    )
    env.set_dof_state_tensor_indexed()

    # Indexed DOF-state writes do not immediately update all rigid-body link
    # transforms used by observations and rendering. Advance PhysX once while
    # holding both robots at the phase-zero target, without advancing the task
    # phase or progress counters.
    env.gym.set_dof_position_target_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.cur_targets),
    )
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.populate_sim_buffers()
    env.current_reference = env.reference.sample(env.phase)
    env.populate_obs_and_states_buffers()
    env.clamp_obs()
    return env.obs_buf.to(env.rl_device)


def _capture_frame(env) -> np.ndarray:
    from isaacgym import gymapi

    # Match SimToolReal._capture_video: synchronize completed GPU physics and
    # update the graphics transforms before rendering camera sensors. Without
    # step_graphics(), each rigid body's visual remains at a stale transform
    # and an articulated robot appears as disconnected links.
    if env.device != "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(
        env.sim,
        env.envs[env.index_to_view],
        env.camera_handle,
        gymapi.IMAGE_COLOR,
    )
    if image.size == 0:
        raise RuntimeError("Isaac Gym returned an empty evaluation camera frame")
    return image.reshape(
        env.camera_properties.height,
        env.camera_properties.width,
        4,
    ).copy()


def _termination_reason(env, completed: bool, final_errors: Dict[str, float]) -> str:
    if completed:
        return "completed"
    reasons = []
    if final_errors["position_error_m"] > env.ee_position_termination_distance:
        reasons.append("position")
    if final_errors["rotation_error_rad"] > env.ee_rotation_termination_angle:
        reasons.append("orientation")
    if final_errors["hand_error_rad"] > env.hand_termination_error:
        reasons.append("hand")
    return "+".join(reasons) if reasons else "unknown"


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    *,
    visualize_reference: bool = True,
) -> Dict[str, Any]:
    # Isaac Gym must be imported before torch.
    from isaacgym import gymapi  # noqa: F401
    import imageio
    import torch

    from deployment.isaac.isaac_env import create_env
    from deployment.rl_player import RlPlayer
    from isaacgymenvs.tasks.simtoolreal.env_motion_imitation import (
        SimToolRealMotionImitation,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    env = create_env(
        config_path=config_path,
        headless=True,
        device=device,
        # Training saves a complete resolved Hydra config. Merging it with the
        # deployment wrapper's generic SimToolRealLSTM config changes robot
        # construction/reset state and produces detached-looking links.
        merge_with_default_config=False,
        overrides={
            "task.env.capture_video": True,
            "task.env.enableCameraSensors": True,
            "task.env.visualizeReferenceRobotInVideo": visualize_reference,
            "task.env.referenceVisualizationActorEnabled": visualize_reference,
            "task.env.useReferenceStateInitialization": False,
            "task.env.referenceStateInitProbability": 0.0,
            # A periodic evaluation video should cover the complete
            # demonstration even when the policy crosses a training-time
            # termination threshold. Threshold violations are reported
            # separately in the metrics below.
            "task.env.imitationEarlyTermination": False,
        },
    )
    if not isinstance(env, SimToolRealMotionImitation):
        raise TypeError(
            "Periodic evaluation requires SimToolRealMotionImitation, got "
            f"{type(env).__name__}"
        )

    # The callback captures every frame explicitly. Keep capture_video enabled
    # so the green reference actor remains active, but suppress the task's
    # frequency-based recorder in this subprocess.
    env.cfg["env"]["_suppressAutomaticVideoCapture"] = True

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    env_state = _checkpoint_env_state(checkpoint)
    if env_state is not None:
        env.set_env_state(env_state)

    policy = RlPlayer(
        int(env.num_obs),
        int(env.num_acts),
        config_path,
        checkpoint_path,
        device,
        env.num_envs,
        append_exploration_observation=(
            _uses_sapg_exploration_observation(config_path)
        ),
    )
    policy.reset()
    obs = _reset_at_phase_zero(env)
    if visualize_reference:
        env._update_reference_visualization_robot()

    initial_reference = env.reference.sample(env.phase)
    initial_position_error = torch.linalg.vector_norm(
        env.palm_center_pos - initial_reference.palm_pos, dim=-1
    )[0]
    initial_rotation_error = env._quaternion_angle(
        env._palm_rot, initial_reference.palm_quat_xyzw
    )[0]
    initial_hand_error = torch.linalg.vector_norm(
        env.arm_hand_dof_pos[:, env.num_arm_dofs :]
        - initial_reference.hand_q,
        dim=-1,
    )[0]
    initial_robot_q = env.arm_hand_dof_pos[0].detach().cpu().tolist()
    initial_reference_q = torch.cat(
        [initial_reference.arm_q[0], initial_reference.hand_q[0]], dim=-1
    ).detach().cpu().tolist()
    initial_max_joint_error = max(
        abs(actual - target)
        for actual, target in zip(initial_robot_q, initial_reference_q)
    )

    component_names = [
        "ee_position_reward",
        "ee_rotation_reward",
        "hand_pose_reward",
        "imitation_reward",
        "kuka_actions_penalty",
        "hand_actions_penalty",
        "arm_action_delta_penalty",
        "hand_action_delta_penalty",
        "arm_joint_velocity_penalty",
        "arm_joint_acceleration_penalty",
        "hand_joint_acceleration_penalty",
        "total_reward",
    ]
    if env.velocity_tracking_enabled:
        component_names.extend(
            [
                "pose_imitation_reward",
                "palm_linear_velocity_reward",
                "palm_angular_velocity_reward",
                "hand_velocity_reward",
                "velocity_imitation_reward",
            ]
        )
    component_sums = {name: 0.0 for name in component_names}
    errors: Dict[str, List[float]] = {
        "position_error_m": [],
        "rotation_error_rad": [],
        "hand_error_rad": [],
    }
    initial_velocity_errors: Dict[str, float] = {}
    if env.velocity_tracking_enabled:
        palm_linear_velocity, palm_angular_velocity = (
            env._palm_center_velocity()
        )
        initial_velocity_errors = {
            "initial_linear_velocity_error_mps": float(
                torch.linalg.vector_norm(
                    palm_linear_velocity - initial_reference.palm_lin_vel,
                    dim=-1,
                )[0].item()
            ),
            "initial_angular_velocity_error_radps": float(
                torch.linalg.vector_norm(
                    palm_angular_velocity - initial_reference.palm_ang_vel,
                    dim=-1,
                )[0].item()
            ),
            "initial_hand_velocity_error_radps": float(
                torch.linalg.vector_norm(
                    env.arm_hand_dof_vel[:, env.num_arm_dofs :]
                    - initial_reference.hand_dq,
                    dim=-1,
                )[0].item()
            ),
        }
        errors.update(
            {
                "linear_velocity_error_mps": [],
                "angular_velocity_error_radps": [],
                "hand_velocity_error_radps": [],
            }
        )
    frames = [_capture_frame(env)]
    max_abs_action = 0.0
    first_threshold_violation_step: Optional[int] = None
    first_threshold_violation_reason: Optional[str] = None
    max_steps = int(math.ceil(env.reference.duration_s / env.control_dt)) + 2
    completed = False

    for step in range(1, max_steps + 1):
        action = policy.get_normalized_action(obs, deterministic_actions=True)
        max_abs_action = max(max_abs_action, float(action.abs().max().item()))
        obs_dict, _, done, extras = env.step(action)
        obs = obs_dict["obs"]

        for name in component_names:
            component_sums[name] += _component(extras, name)
        errors["position_error_m"].append(
            _scalar(extras["imitation/position_error_m"])
        )
        errors["rotation_error_rad"].append(
            _scalar(extras["imitation/rotation_error_rad"])
        )
        errors["hand_error_rad"].append(
            _scalar(extras["imitation/hand_error_rad"])
        )
        if env.velocity_tracking_enabled:
            for name in (
                "linear_velocity_error_mps",
                "angular_velocity_error_radps",
                "hand_velocity_error_radps",
            ):
                errors[name].append(_scalar(extras[f"imitation/{name}"]))
        if first_threshold_violation_step is None:
            threshold_reasons = []
            if (
                errors["position_error_m"][-1]
                > env.ee_position_termination_distance
            ):
                threshold_reasons.append("position")
            if (
                errors["rotation_error_rad"][-1]
                > env.ee_rotation_termination_angle
            ):
                threshold_reasons.append("orientation")
            if errors["hand_error_rad"][-1] > env.hand_termination_error:
                threshold_reasons.append("hand")
            if threshold_reasons:
                first_threshold_violation_step = step
                first_threshold_violation_reason = "+".join(
                    threshold_reasons
                )
        frames.append(_capture_frame(env))

        if bool(done[0].item()):
            completed = bool(env.phase[0].item() >= 1.0)
            break
    else:
        step = max_steps

    video_path = output_path / "evaluation.mp4"
    fps = int(round(1.0 / env.control_dt))
    imageio.mimsave(
        video_path,
        frames,
        fps=fps,
    )

    final_errors = {
        name: values[-1] if values else 0.0 for name, values in errors.items()
    }
    metrics: Dict[str, Any] = {
        "episode_reward": component_sums["total_reward"],
        "episode_steps": step,
        "episode_duration_s": step * float(env.control_dt),
        "final_phase": float(env.phase[0].item()),
        "completed": int(completed),
        "early_terminated": int(not completed),
        "termination_reason": _termination_reason(
            env, completed, final_errors
        ),
        "visualized_reference": int(visualize_reference),
        "initial_position_error_m": float(initial_position_error.item()),
        "initial_rotation_error_rad": float(initial_rotation_error.item()),
        "initial_hand_error_rad": float(initial_hand_error.item()),
        "initial_max_joint_error_rad": initial_max_joint_error,
        "initial_robot_q": initial_robot_q,
        "initial_reference_q": initial_reference_q,
        "threshold_violated": int(first_threshold_violation_step is not None),
        "first_threshold_violation_step": first_threshold_violation_step,
        "first_threshold_violation_s": (
            first_threshold_violation_step * float(env.control_dt)
            if first_threshold_violation_step is not None
            else None
        ),
        "first_threshold_violation_reason": (
            first_threshold_violation_reason
        ),
        "max_abs_action": max_abs_action,
        "video_path": str(video_path),
    }
    metrics.update(initial_velocity_errors)
    for name, total in component_sums.items():
        metrics[f"{name}_sum"] = total
        metrics[f"{name}_mean"] = total / max(step, 1)
    for name, values in errors.items():
        metrics[f"{name}_mean"] = float(np.mean(values)) if values else 0.0
        metrics[f"{name}_max"] = float(np.max(values)) if values else 0.0
        metrics[f"{name}_final"] = values[-1] if values else 0.0

    metrics["arm_action_cost_sum"] = -component_sums["kuka_actions_penalty"]
    metrics["hand_action_cost_sum"] = -component_sums["hand_actions_penalty"]
    metrics["arm_delta_action_cost_sum"] = -component_sums[
        "arm_action_delta_penalty"
    ]
    metrics["hand_delta_action_cost_sum"] = -component_sums[
        "hand_action_delta_penalty"
    ]
    metrics["arm_joint_velocity_cost_sum"] = -component_sums[
        "arm_joint_velocity_penalty"
    ]
    metrics["arm_joint_acceleration_cost_sum"] = -component_sums[
        "arm_joint_acceleration_penalty"
    ]
    metrics["hand_joint_acceleration_cost_sum"] = -component_sums[
        "hand_joint_acceleration_penalty"
    ]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument(
        "--disable-reference-visualization",
        action="store_true",
        help="Evaluate without creating the auxiliary green reference actor.",
    )
    args = parser.parse_args()

    metrics = evaluate(
        args.config,
        args.checkpoint,
        args.output_dir,
        visualize_reference=not args.disable_reference_visualization,
    )
    result_path = Path(args.result_json).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Periodic evaluation results: {result_path}")


if __name__ == "__main__":
    main()
