"""Joint-space motion imitation after the DG5F low-level-controller fix."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from isaacgym import gymtorch
import torch
from torch import Tensor

from isaacgymenvs.tasks.simtoolreal.demonstration_reference import (
    JointDemonstrationReference60Hz,
    JointReferenceSample,
)
from isaacgymenvs.tasks.simtoolreal.env import (
    DEBUG_RSI,
    DEBUG_RSI_MAX_RESETS,
    DEBUG_RSI_STEPS_PER_RESET,
    SimToolReal,
)
from isaacgymenvs.utils.torch_jit_utils import scale, tensor_clamp, unscale


class SimToolRealMotionImitationLLCFix00(SimToolReal):
    """Discrete 60 Hz joint imitation with absolute joint-angle actions."""

    RIGHT_HAND_60_DEG_ASSET = (
        "urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
    )

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        env_cfg = cfg["env"]
        hand_mount_yaw_deg = float(env_cfg.get("handMountYawOffsetDeg", 60.0))
        if str(env_cfg.get("handSide", "right")).lower() != "right":
            raise ValueError("LLC-fix motion imitation supports the right DG5F only")
        if not math.isclose(hand_mount_yaw_deg, 60.0, abs_tol=1e-6):
            raise ValueError(
                "The LLC-fix robot asset requires a 60 degree DG5F mount, "
                f"got {hand_mount_yaw_deg:g}"
            )

        half_mount_yaw = math.radians(hand_mount_yaw_deg) / 2.0
        mount_quat_xyzw = [
            0.0,
            0.0,
            math.sin(half_mount_yaw),
            math.cos(half_mount_yaw),
        ]
        env_cfg["asset"]["robot"] = self.RIGHT_HAND_60_DEG_ASSET
        env_cfg["asset"]["deltoRobots"]["right"] = self.RIGHT_HAND_60_DEG_ASSET
        env_cfg["palmOrientationOffsetQuatXyzw"] = mount_quat_xyzw
        env_cfg["handMountYawOffsetDeg"] = hand_mount_yaw_deg

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        env_cfg = self.cfg["env"]
        if not math.isclose(self.dt, 1.0 / 60.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "LLC-fix imitation requires one 60 Hz control step per "
                f"reference sample, got dt={self.dt:.9g} s"
            )

        self.reference = JointDemonstrationReference60Hz(
            env_cfg["demonstration"], device=self.device
        )
        reference_q = torch.cat(
            [self.reference.arm_q, self.reference.hand_q], dim=-1
        )
        clamped_q = tensor_clamp(
            reference_q,
            self.arm_hand_dof_lower_limits,
            self.arm_hand_dof_upper_limits,
        )
        if torch.any(torch.abs(reference_q - clamped_q) > 1e-7):
            raise ValueError(
                "The processed 60 Hz demonstration contains joint positions "
                "outside the simulation limits"
            )
        if DEBUG_RSI:
            self._debug_rsi_log(f"demonstration      : {self.reference.path}")
            self._debug_rsi_log_reference_limits(reference_q, 0)

        self.reference_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Compatibility surface used by the shared imitation evaluators. The
        # task itself remains discrete: set_reference_phase converts a phase
        # to the nearest 60 Hz sample instead of interpolating.
        self.reference_phase_start = 0.0
        self.reference_phase_end = 1.0
        self.reference_phase_span = 1.0
        self.reference_init_max_phase = 1.0
        self.phase = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.current_reference: JointReferenceSample = self.reference.sample(
            self.reference_index
        )

        self.joint_position_reward_weight = float(
            env_cfg.get("jointPositionRewardWeight", 0.8)
        )
        self.joint_velocity_reward_weight = float(
            env_cfg.get("jointVelocityRewardWeight", 0.2)
        )
        if not math.isclose(
            self.joint_position_reward_weight
            + self.joint_velocity_reward_weight,
            1.0,
            abs_tol=1e-8,
        ):
            raise ValueError("Joint position and velocity reward weights must sum to 1")
        self.joint_position_reward_scale = float(
            env_cfg.get("jointPositionRewardScale", 10.0)
        )
        self.joint_velocity_reward_scale = float(
            env_cfg.get("jointVelocityRewardScale", 0.01)
        )
        if (
            self.joint_position_reward_scale <= 0.0
            or self.joint_velocity_reward_scale <= 0.0
        ):
            raise ValueError("Gaussian joint reward scales must be positive")

        self.early_termination = bool(
            env_cfg.get("imitationEarlyTermination", True)
        )
        self.arm_position_termination_threshold = float(
            env_cfg.get("armJointPositionTerminationThresholdRad", 0.35)
        )
        self.hand_position_termination_threshold = float(
            env_cfg.get("handJointPositionTerminationThresholdRad", 0.50)
        )
        self.termination_grace_steps = int(
            env_cfg.get("jointPositionTerminationGraceSteps", 5)
        )
        if (
            self.arm_position_termination_threshold <= 0.0
            or self.hand_position_termination_threshold <= 0.0
            or self.termination_grace_steps < 0
        ):
            raise ValueError("Invalid joint-position early-termination settings")

        for key in (
            "joint_position_reward",
            "joint_velocity_reward",
            "total_reward",
            "episode_steps",
        ):
            if key not in self.rewards_episode:
                self.rewards_episode[key] = torch.zeros(
                    self.num_envs, dtype=torch.float32, device=self.device
                )

        print(
            f"LLC-fix joint reference: {self.reference.path} "
            f"({self.reference.sample_count} samples, "
            f"{self.reference.duration_s:.3f} s at 60 Hz; discrete indexing)"
        )

        self.velocity_tracking_enabled = False
        self.object_tracking_enabled = False
        self.fingertip_tracking_enabled = False

    def _init_reference_state_initialization(self) -> None:
        """Use the trajectory itself as the only reset-state source."""
        self.use_reference_state_initialization = True
        self.reference_state_init_probability = 1.0
        self.reference_state_reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.reference_state_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.reference_state_reset_count = 0
        self.regular_state_reset_count = 0
        self.reference_state_names = []

    def _apply_reference_state_initialization(self, env_ids: Tensor) -> None:
        # The discrete trajectory is loaded after SimToolReal construction;
        # reset_idx applies it below once it exists.
        return

    def reset_idx(
        self,
        env_ids: Tensor,
        reset_buf_idxs=None,
        episode_reset=True,
        tensor_reset=True,
    ) -> None:
        super().reset_idx(
            env_ids,
            reset_buf_idxs=reset_buf_idxs,
            episode_reset=episode_reset,
            tensor_reset=tensor_reset,
        )
        if not hasattr(self, "reference") or len(env_ids) == 0 or not tensor_reset:
            return

        # Uniform RSI over all non-terminal discrete samples. The last sample
        # remains reachable after one or more environment steps but is not a
        # zero-length episode start.
        sampled_indices = torch.randint(
            low=0,
            high=self.reference.last_index,
            size=(len(env_ids),),
            device=self.device,
        )
        self._set_reference_indices(env_ids, sampled_indices, flush=False)
        self.reference_state_reset_mask[env_ids] = True
        self.reference_state_reset_count += len(env_ids)

        scalars = self.extras.setdefault("scalars", {})
        scalars["rsi/reference_reset_fraction"] = 1.0
        scalars["rsi/reference_reset_count_total"] = int(
            self.reference_state_reset_count
        )
        scalars["rsi/start_phase_mean"] = float(
            self.phase[env_ids].mean().item()
        )

    def _set_reference_indices(
        self,
        env_ids: Tensor,
        indices: Tensor,
        *,
        flush: bool,
    ) -> None:
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        reference = self.reference.sample(indices)
        robot_pos = torch.cat([reference.arm_q, reference.hand_q], dim=-1)
        robot_vel = torch.cat([reference.arm_dq, reference.hand_dq], dim=-1)

        self.reference_index[env_ids] = indices
        self.phase[env_ids] = self.reference.phase(indices)
        self.arm_hand_dof_pos[env_ids] = robot_pos
        self.arm_hand_dof_vel[env_ids] = robot_vel
        self.prev_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos
        self.cur_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos
        self.prev_penalized_actions[env_ids] = 0.0
        self.action_deltas[env_ids] = 0.0

        robot_indices = self.robot_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(robot_indices),
            len(env_ids),
        )
        self.deferred_set_dof_state_tensor_indexed([robot_indices])
        self.current_reference = self.reference.sample(self.reference_index)
        if (
            DEBUG_RSI
            and bool((env_ids == 0).any())
            and getattr(self, "_debug_rsi_resets", 0) < DEBUG_RSI_MAX_RESETS
        ):
            row = int((env_ids == 0).nonzero()[0].item())
            expected_q = robot_pos[row]
            error = (self.arm_hand_dof_pos[0] - expected_q).abs()
            self._debug_rsi_resets = getattr(self, "_debug_rsi_resets", 0) + 1
            self._debug_rsi_step = 0
            self._debug_rsi_log(
                f"[RSI #{self._debug_rsi_resets}] index={int(indices[row])}"
                f"  phase={float(self.phase[0]):.4f}"
                f"  tensor_arm_err={float(error[: self.num_arm_dofs].max()):.6f}"
                f"  tensor_hand_err={float(error[self.num_arm_dofs :].max()):.6f}"
            )
            self._debug_rsi_log(
                "    ref_arm = "
                f"{[round(v, 3) for v in expected_q[: self.num_arm_dofs].tolist()]}"
            )
            self._debug_rsi_log(
                "    dof_arm = "
                f"{[round(v, 3) for v in self.arm_hand_dof_pos[0, : self.num_arm_dofs].tolist()]}"
            )
        if flush:
            self.set_dof_state_tensor_indexed()
            self.populate_sim_buffers()
            self.populate_obs_and_states_buffers()
            self.clamp_obs()

    def set_reference_phase(
        self,
        env_ids: Tensor,
        phase,
        *,
        flush: bool = True,
    ) -> None:
        """Reset to the nearest discrete sample for evaluator compatibility."""
        if len(env_ids) == 0:
            return
        phase_tensor = torch.as_tensor(
            phase, dtype=torch.float32, device=self.device
        )
        if phase_tensor.ndim == 0:
            phase_tensor = phase_tensor.expand(len(env_ids))
        phase_tensor = phase_tensor.reshape(-1)
        if len(phase_tensor) != len(env_ids):
            raise ValueError(
                f"Expected {len(env_ids)} phase values, got {len(phase_tensor)}"
            )
        indices = torch.round(
            phase_tensor.clamp(0.0, 1.0) * self.reference.last_index
        ).to(dtype=torch.long)
        self._set_reference_indices(env_ids, indices, flush=flush)

    def _set_reference_visualization_robot(
        self,
        reference: JointReferenceSample,
        *,
        flush: bool = True,
    ) -> None:
        """Place the optional green evaluation actor at a joint reference."""
        if not self.VISUALIZE_REFERENCE_ROBOT:
            return
        env_id = int(self.index_to_view)
        reference_q = torch.cat(
            [reference.arm_q[env_id], reference.hand_q[env_id]], dim=-1
        )
        self.visualization_robot_arm_hand_dof_pos[env_id].copy_(reference_q)
        self.visualization_robot_arm_hand_dof_vel[env_id].zero_()
        self.cur_targets[env_id, self.num_hand_arm_dofs :].copy_(reference_q)
        actor_index = self.visualization_robot_indices[env_id : env_id + 1].to(
            torch.int32
        )
        self.deferred_set_dof_state_tensor_indexed([actor_index])
        if flush:
            self.set_dof_state_tensor_indexed()
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.cur_targets)
            )

    def _update_reference_visualization_robot(self) -> None:
        self._set_reference_visualization_robot(self.current_reference)

    def next_reference_action(self) -> Tuple[Tensor, Tensor]:
        """Return the normalized action for the next reference sample."""
        next_indices = (self.reference_index + 1).clamp(
            max=self.reference.last_index
        )
        reference = self.reference.sample(next_indices)
        target_angles = torch.cat(
            [reference.arm_q, reference.hand_q],
            dim=-1,
        )
        actions_ideal = unscale(
            target_angles,
            self.arm_hand_dof_lower_limits,
            self.arm_hand_dof_upper_limits,
        ).clamp(-1.0, 1.0)
        return actions_ideal, target_angles

    # def step(self, actions: Tensor):
    #     """DEBUG: reset first, then execute the next reference action."""
    #     del actions

    #     # VecTask normally applies pending resets from pre_physics_step(),
    #     # after the caller has already supplied an action. That ordering makes
    #     # the first ideal action after RSI belong to the previous episode.
    #     # Consume full-episode resets here so next_reference_action() observes
    #     # the newly sampled reference indices instead.
    #     reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    #     if len(reset_env_ids) > 0:
    #         self.reset_idx(reset_env_ids, None)

    #     actions, _target_angles = self.next_reference_action()
    #     return super().step(actions)

    def pre_physics_step(
        self, actions: Tensor, joint_pos_targets: Optional[Tensor] = None
    ) -> None:
        if joint_pos_targets is not None:
            raise ValueError("LLC-fix task computes absolute targets from actions")
        normalized_actions = actions[:, : self.num_hand_arm_dofs].clamp(-1.0, 1.0)
        absolute_targets = scale(
            normalized_actions,
            self.arm_hand_dof_lower_limits,
            self.arm_hand_dof_upper_limits,
        )
        if self.VISUALIZE_REFERENCE_ROBOT:
            next_indices = (self.reference_index + 1).clamp(
                max=self.reference.last_index
            )
            # Queue the visualization actor before the base method performs
            # its single combined DOF-state flush. Isaac Gym's tensor setters
            # must not be called more than once for the same simulation step.
            self._set_reference_visualization_robot(
                self.reference.sample(next_indices), flush=False
            )
        super().pre_physics_step(actions, joint_pos_targets=absolute_targets)
        if DEBUG_RSI and getattr(self, "_debug_rsi_step", -1) == 0:
            # This is after the deferred reset-state flush in the base method,
            # but before VecTask calls gym.simulate(). Refreshing here tells us
            # whether PhysX accepted the indexed reset independently from what
            # happens during the following dynamics step.
            self.gym.refresh_dof_state_tensor(self.sim)
            reference = self.reference.sample(self.reference_index)
            reference_q = torch.cat(
                [reference.arm_q[0], reference.hand_q[0]], dim=-1
            )
            error = (self.arm_hand_dof_pos[0] - reference_q).abs()
            self._debug_rsi_log(
                "  [pre-sim]"
                f" index={int(self.reference_index[0])}"
                f"  arm_err="
                f"{[round(v, 6) for v in error[: self.num_arm_dofs].tolist()]}"
                f"  hand_err_max={float(error[self.num_arm_dofs :].max()):.6f}"
                f"  arm_vel="
                f"{[round(v, 3) for v in self.arm_hand_dof_vel[0, : self.num_arm_dofs].tolist()]}"
            )

    def compute_imitation_reward(self) -> Tuple[Tensor, Tensor]:
        reference_q = torch.cat(
            [self.current_reference.arm_q, self.current_reference.hand_q], dim=-1
        )
        reference_dq = torch.cat(
            [self.current_reference.arm_dq, self.current_reference.hand_dq], dim=-1
        )
        position_delta = self.arm_hand_dof_pos - reference_q
        velocity_delta = self.arm_hand_dof_vel - reference_dq
        position_mse = torch.mean(position_delta.square(), dim=-1)
        velocity_mse = torch.mean(velocity_delta.square(), dim=-1)
        position_reward = torch.exp(
            -self.joint_position_reward_scale * position_mse
        )
        velocity_reward = torch.exp(
            -self.joint_velocity_reward_scale * velocity_mse
        )
        reward = (
            self.joint_position_reward_weight * position_reward
            + self.joint_velocity_reward_weight * velocity_reward
        )
        self.rew_buf[:] = reward

        arm_max_error = torch.amax(
            torch.abs(position_delta[:, : self.num_arm_dofs]), dim=-1
        )
        hand_max_error = torch.amax(
            torch.abs(position_delta[:, self.num_arm_dofs :]), dim=-1
        )
        grace_finished = self.progress_buf > self.termination_grace_steps
        arm_diverged = (
            arm_max_error > self.arm_position_termination_threshold
        ) & grace_finished
        hand_diverged = (
            hand_max_error > self.hand_position_termination_threshold
        ) & grace_finished
        diverged = torch.zeros_like(arm_diverged)
        if self.early_termination:
            diverged = arm_diverged | hand_diverged
        finished = self.reference_index >= self.reference.last_index
        self.reset_buf[:] = finished | diverged
        self.reset_goal_buf[:] = False

        components = {
            "joint_position_reward": (
                self.joint_position_reward_weight * position_reward
            ),
            "joint_velocity_reward": (
                self.joint_velocity_reward_weight * velocity_reward
            ),
            "total_reward": reward,
            "episode_steps": torch.ones_like(reward),
        }
        for name, value in components.items():
            self.rewards_episode[name] += value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = components
        self.extras["imitation/phase"] = self.phase.mean()
        self.extras["imitation/joint_position_mse"] = position_mse.mean()
        self.extras["imitation/joint_velocity_mse"] = velocity_mse.mean()
        self.extras["imitation/arm_max_error_rad"] = arm_max_error.mean()
        self.extras["imitation/hand_max_error_rad"] = hand_max_error.mean()
        self.extras["imitation/early_termination_fraction"] = (
            diverged.float().mean()
        )
        self.extras["imitation/completion_fraction"] = finished.float().mean()
        self.extras["imitation/termination_arm_fraction"] = (
            arm_diverged.float().mean()
        )
        self.extras["imitation/termination_hand_fraction"] = (
            hand_diverged.float().mean()
        )
        return reward, finished

    def populate_obs_and_states_buffers(self) -> None:
        obs = {
            "joint_pos": unscale(
                self.arm_hand_dof_pos,
                self.arm_hand_dof_lower_limits,
                self.arm_hand_dof_upper_limits,
            ),
            "joint_vel": self.arm_hand_dof_vel,
            "prev_action_targets": self.prev_targets[:, : self.num_hand_arm_dofs],
            "palm_pos": self.palm_center_pos,
            "palm_rot": self._palm_rot,
            "fingertip_pos_rel_palm": self.fingertip_pos_rel_palm.reshape(
                self.num_envs, -1
            ),
            "phase": self.phase.unsqueeze(-1),
        }
        self.states_buf = torch.cat(
            [obs[name].reshape(self.num_envs, -1) for name in self.state_list],
            dim=-1,
        )
        self.obs_buf = torch.cat(
            [obs[name].reshape(self.num_envs, -1) for name in self.obs_list],
            dim=-1,
        )

    def post_physics_step(self) -> None:
        self.frame_since_restart += 1
        self.progress_buf += 1
        self.randomize_buf += 1
        self.reference_index.add_(1).clamp_(max=self.reference.last_index)
        self.phase.copy_(self.reference.phase(self.reference_index))
        self.populate_sim_buffers()
        self.current_reference = self.reference.sample(self.reference_index)
        if (
            DEBUG_RSI
            and getattr(self, "_debug_rsi_step", DEBUG_RSI_STEPS_PER_RESET)
            < DEBUG_RSI_STEPS_PER_RESET
        ):
            # populate_sim_buffers() has refreshed the state from PhysX. This
            # measures the real post-step state, rather than only the tensors
            # assigned by _set_reference_indices() during the reset.
            reference_q = torch.cat(
                [
                    self.current_reference.arm_q[0],
                    self.current_reference.hand_q[0],
                ],
                dim=-1,
            )
            error = (self.arm_hand_dof_pos[0] - reference_q).abs()
            target_error = (
                self.prev_targets[0, : self.num_hand_arm_dofs] - reference_q
            ).abs()
            green = ""
            if self.VISUALIZE_REFERENCE_ROBOT:
                green_error = (
                    self.visualization_robot_arm_hand_dof_pos[0] - reference_q
                ).abs()
                green = (
                    "  green_arm_err="
                    f"{float(green_error[: self.num_arm_dofs].max()):.4f}"
                )
            self._debug_rsi_log(
                f"  [+{self._debug_rsi_step}]"
                f" index={int(self.reference_index[0])}"
                f" phase={float(self.phase[0]):.4f}"
                f"  arm_err="
                f"{[round(v, 3) for v in error[: self.num_arm_dofs].tolist()]}"
                f"  hand_err_max={float(error[self.num_arm_dofs :].max()):.3f}"
                f"  arm_tgt_err_max="
                f"{float(target_error[: self.num_arm_dofs].max()):.4f}"
                f"{green}"
            )
            if self._debug_rsi_step == 0:
                self._debug_rsi_log(
                    "    post_arm = "
                    f"{[round(v, 3) for v in self.arm_hand_dof_pos[0, : self.num_arm_dofs].tolist()]}"
                    "  post_arm_vel = "
                    f"{[round(v, 3) for v in self.arm_hand_dof_vel[0, : self.num_arm_dofs].tolist()]}"
                )
            self._debug_rsi_step += 1
        self.compute_imitation_reward()
        self.populate_obs_and_states_buffers()
        self.clamp_obs()
        self._capture_video_if_needed()
