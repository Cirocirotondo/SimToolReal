"""DeepMimic-style motion imitation for the UR5e + DG5F robot."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from isaacgym import gymtorch
import torch
from torch import Tensor

from isaacgymenvs.tasks.simtoolreal.demonstration_reference import (
    DemonstrationReference,
    ReferenceSample,
    sample_reference_phases,
)
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from isaacgymenvs.utils.torch_jit_utils import quat_rotate, tensor_clamp, unscale


class SimToolRealMotionImitation(SimToolReal):
    """Track a teleoperation clip without using object state."""

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
            raise ValueError("Motion imitation currently supports the right DG5F only")
        if not math.isclose(hand_mount_yaw_deg, 60.0, abs_tol=1e-6):
            raise ValueError(
                "The motion-imitation robot asset is calibrated for a 60 degree "
                f"DG5F mount, got {hand_mount_yaw_deg:g}"
            )

        half_mount_yaw = math.radians(hand_mount_yaw_deg) / 2.0
        mount_quat_xyzw = [
            0.0,
            0.0,
            math.sin(half_mount_yaw),
            math.cos(half_mount_yaw),
        ]
        # Migrate saved configs from before the physical hand-mount transform
        # was represented in the imitation task.
        env_cfg["asset"]["robot"] = self.RIGHT_HAND_60_DEG_ASSET
        env_cfg["asset"]["deltoRobots"]["right"] = self.RIGHT_HAND_60_DEG_ASSET
        env_cfg["palmOrientationOffsetQuatXyzw"] = mount_quat_xyzw
        legacy_reference_quat = env_cfg.get(
            "demonstrationEeToPalmQuatXyzw",
            [0.0, 0.0, 0.0, 1.0],
        )
        if "handMountYawOffsetDeg" not in env_cfg and all(
            math.isclose(float(value), expected, abs_tol=1e-7)
            for value, expected in zip(
                legacy_reference_quat,
                [0.0, 0.0, 0.0, 1.0],
            )
        ):
            env_cfg["demonstrationEeToPalmQuatXyzw"] = mount_quat_xyzw
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
        self.reference = DemonstrationReference(
            env_cfg["demonstration"],
            device=self.device,
            hand_source=env_cfg.get("demonstrationHandSource", "measured"),
            world_yaw_offset_deg=float(
                env_cfg.get("demonstrationWorldYawOffsetDeg", 180.0)
            ),
            world_position_offset_m=tuple(
                env_cfg.get("demonstrationWorldPositionOffset", [0.0, 0.6, 0.0])
            ),
            ee_to_palm_offset_m=tuple(
                env_cfg.get("demonstrationEeToPalmOffset", [0.0, 0.0, 0.16])
            ),
            ee_to_palm_quat_xyzw=tuple(
                env_cfg.get("demonstrationEeToPalmQuatXyzw", mount_quat_xyzw)
            ),
        )
        reference_q = torch.cat(
            [self.reference.arm_q, self.reference.hand_q], dim=-1
        )
        clamped_reference_q = tensor_clamp(
            reference_q,
            self.arm_hand_dof_lower_limits,
            self.arm_hand_dof_upper_limits,
        )
        clamped_count = int(
            torch.count_nonzero(
                torch.abs(clamped_reference_q - reference_q) > 1e-7
            ).item()
        )
        self.reference.arm_q.copy_(clamped_reference_q[:, : self.num_arm_dofs])
        self.reference.hand_q.copy_(
            clamped_reference_q[:, self.num_arm_dofs :]
        )
        if clamped_count:
            hand_velocity = torch.gradient(
                self.reference.hand_q,
                spacing=(self.reference.time,),
                dim=(0,),
            )[0]
            self.reference.hand_dq.copy_(hand_velocity)
            print(
                f"Clamped {clamped_count} demonstration joint samples "
                "to the simulation URDF limits"
            )
        self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.phase_delta = self.dt / self.reference.duration_s
        self.reference_init_max_phase = float(
            env_cfg.get("referenceInitMaxPhase", 1.0)
        )
        if not 0.0 < self.reference_init_max_phase <= 1.0:
            raise ValueError("referenceInitMaxPhase must be in (0, 1]")
        self.reference_init_distribution = str(
            env_cfg.get("referenceInitDistribution", "uniform")
        ).lower()
        if self.reference_init_distribution not in {
            "uniform",
            "triangular",
        }:
            raise ValueError(
                "referenceInitDistribution must be 'uniform' or "
                f"'triangular', got {self.reference_init_distribution!r}"
            )

        self.ee_position_reward_weight = float(env_cfg["eePositionRewardWeight"])
        self.ee_rotation_reward_weight = float(env_cfg["eeRotationRewardWeight"])
        self.hand_pose_reward_weight = float(env_cfg["handPoseRewardWeight"])
        self.ee_position_reward_scale = float(env_cfg["eePositionRewardScale"])
        self.ee_rotation_reward_scale = float(env_cfg["eeRotationRewardScale"])
        self.hand_pose_reward_scale = float(env_cfg["handPoseRewardScale"])
        weight_sum = (
            self.ee_position_reward_weight
            + self.ee_rotation_reward_weight
            + self.hand_pose_reward_weight
        )
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("Imitation reward weights must sum to 1")

        self.early_termination = bool(env_cfg.get("imitationEarlyTermination", True))
        self.ee_position_termination_distance = float(
            env_cfg["eePositionTerminationDistance"]
        )
        self.ee_rotation_termination_angle = float(
            env_cfg["eeRotationTerminationAngleRad"]
        )
        self.hand_termination_error = float(env_cfg["handTerminationError"])
        # The palm-center OSC needs the current wrist orientation on the very
        # first pre-physics step; the base OSC does not require this state.
        self.populate_sim_buffers()
        self.current_reference: ReferenceSample = self.reference.sample(self.phase)

        for key in (
            "ee_position_reward",
            "ee_rotation_reward",
            "hand_pose_reward",
            "imitation_reward",
            "episode_steps",
        ):
            if key not in self.rewards_episode:
                self.rewards_episode[key] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )
        print(
            f"Motion imitation reference: {self.reference.path} "
            f"({self.reference.duration_s:.3f} s, phase delta {self.phase_delta:.8f})"
        )

    def _init_reference_state_initialization(self) -> None:
        """Initialize trajectory RSI without loading the base grasp-state files."""
        env_cfg = self.cfg["env"]
        self.use_reference_state_initialization = bool(
            env_cfg.get("useReferenceStateInitialization", False)
        )
        self.reference_state_init_probability = float(
            env_cfg.get("referenceStateInitProbability", 0.0)
        )
        if not 0.0 <= self.reference_state_init_probability <= 1.0:
            raise ValueError("referenceStateInitProbability must be in [0, 1]")

        # Keep the bookkeeping fields expected by the base environment, while
        # deliberately avoiding its discrete grasp-state JSON loader.
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
        # The base reset calls this before the demonstration has necessarily
        # been loaded. Continuous trajectory RSI is applied below by reset_idx.
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

        use_random_phase = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        if self.use_reference_state_initialization:
            use_random_phase = (
                torch.rand(len(env_ids), device=self.device)
                < self.reference_state_init_probability
            )
        sampled_phase = sample_reference_phases(
            len(env_ids),
            self.reference_init_max_phase,
            self.reference_init_distribution,
            self.device,
        )
        self.phase[env_ids] = torch.where(
            use_random_phase, sampled_phase, torch.zeros_like(sampled_phase)
        )
        self.reference_state_reset_mask[env_ids] = use_random_phase
        self.reference_state_reset_count += int(use_random_phase.sum().item())
        self.regular_state_reset_count += int((~use_random_phase).sum().item())

        scalars = self.extras.setdefault("scalars", {})
        scalars["rsi/reference_reset_fraction"] = float(
            use_random_phase.float().mean().item()
        )
        scalars["rsi/reference_reset_count_total"] = int(
            self.reference_state_reset_count
        )
        scalars["rsi/start_phase_mean"] = float(
            self.phase[env_ids].mean().item()
        )
        scalars["rsi/random_start_phase_mean"] = (
            float(sampled_phase[use_random_phase].mean().item())
            if bool(use_random_phase.any().item())
            else 0.0
        )

        self.set_reference_phase(env_ids, self.phase[env_ids], flush=False)

    def set_reference_phase(
        self,
        env_ids: Tensor,
        phase,
        *,
        flush: bool = True,
    ) -> None:
        """Place selected environments exactly at a demonstration phase."""
        if len(env_ids) == 0:
            return
        phase_tensor = torch.as_tensor(
            phase, dtype=self.phase.dtype, device=self.device
        )
        if phase_tensor.ndim == 0:
            phase_tensor = phase_tensor.expand(len(env_ids))
        phase_tensor = phase_tensor.reshape(-1)
        if len(phase_tensor) != len(env_ids):
            raise ValueError(
                f"Expected {len(env_ids)} phase values, got {len(phase_tensor)}"
            )
        phase_tensor = phase_tensor.clamp(0.0, 1.0)
        self.phase[env_ids] = phase_tensor

        reference = self.reference.sample(phase_tensor)
        robot_pos = torch.cat([reference.arm_q, reference.hand_q], dim=-1)
        robot_vel = torch.cat([reference.arm_dq, reference.hand_dq], dim=-1)
        robot_pos = tensor_clamp(
            robot_pos,
            self.arm_hand_dof_lower_limits,
            self.arm_hand_dof_upper_limits,
        )
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
        self.current_reference = self.reference.sample(self.phase)
        if flush:
            self.set_dof_state_tensor_indexed()
            self.populate_sim_buffers()
            self.populate_obs_and_states_buffers()
            self.clamp_obs()

    @staticmethod
    def _skew(vector: Tensor) -> Tensor:
        zeros = torch.zeros_like(vector[:, 0])
        x, y, z = vector.unbind(-1)
        return torch.stack(
            [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
        ).reshape(-1, 3, 3)

    def _operational_space_arm_targets(self, arm_actions: Tensor) -> Tensor:
        if arm_actions.shape[-1] != 6:
            raise ValueError("Palm operational-space control requires 6 arm actions")
        desired_twist = torch.empty_like(arm_actions)
        desired_twist[:, :3] = (
            arm_actions[:, :3]
            * self.operational_space_arm_translation_speed_scale
            * self.dt
        )
        desired_twist[:, 3:] = (
            arm_actions[:, 3:]
            * self.operational_space_arm_rotation_speed_scale
            * self.dt
        )

        wrist_jacobian = self.robot_jacobian[
            :, self.arm_ee_jacobian_index, :, : self.num_arm_dofs
        ]
        local_offset = self.palm_center_offset
        world_offset = quat_rotate(self._palm_link_rot, local_offset)
        linear = wrist_jacobian[:, :3] - torch.bmm(
            self._skew(world_offset), wrist_jacobian[:, 3:]
        )
        palm_jacobian = torch.cat([linear, wrist_jacobian[:, 3:]], dim=1)
        transpose = palm_jacobian.transpose(1, 2)
        identity = torch.eye(6, device=self.device).unsqueeze(0)
        lhs = torch.bmm(palm_jacobian, transpose) + (
            self.operational_space_ik_damping**2
        ) * identity
        q_delta_unclipped = torch.bmm(
            transpose, torch.linalg.solve(lhs, desired_twist.unsqueeze(-1))
        ).squeeze(-1)
        q_delta = q_delta_unclipped.clamp(
            -self.operational_space_max_joint_delta,
            self.operational_space_max_joint_delta,
        )
        targets = tensor_clamp(
            self.prev_targets[:, : self.num_arm_dofs] + q_delta,
            self.arm_hand_dof_lower_limits[: self.num_arm_dofs],
            self.arm_hand_dof_upper_limits[: self.num_arm_dofs],
        )
        applied_delta = targets - self.prev_targets[:, : self.num_arm_dofs]
        achieved = torch.bmm(
            palm_jacobian, applied_delta.unsqueeze(-1)
        ).squeeze(-1)
        self.operational_space_requested_twist[:] = desired_twist
        self.operational_space_achieved_twist[:] = achieved
        self.operational_space_ik_residual_norm[:] = torch.linalg.vector_norm(
            desired_twist - achieved, dim=-1
        )
        self.operational_space_joint_delta_norm[:] = torch.linalg.vector_norm(
            applied_delta, dim=-1
        )
        self.operational_space_joint_delta_clipped[:] = torch.any(
            torch.abs(q_delta_unclipped)
            > self.operational_space_max_joint_delta,
            dim=-1,
        )
        return targets

    @staticmethod
    def _quaternion_angle(q: Tensor, target: Tensor) -> Tensor:
        dot = torch.abs(torch.sum(q * target, dim=-1)).clamp(max=1.0)
        return 2.0 * torch.acos(dot)

    def compute_imitation_reward(self) -> Tuple[Tensor, Tensor]:
        ref = self.current_reference
        position_error = torch.linalg.vector_norm(
            self.palm_center_pos - ref.palm_pos, dim=-1
        )
        rotation_error = self._quaternion_angle(self._palm_rot, ref.palm_quat_xyzw)
        hand_error = torch.linalg.vector_norm(
            self.arm_hand_dof_pos[:, self.num_arm_dofs :] - ref.hand_q, dim=-1
        )

        position_reward = torch.exp(
            -self.ee_position_reward_scale * position_error.square()
        )
        rotation_reward = torch.exp(
            -self.ee_rotation_reward_scale * rotation_error.square()
        )
        hand_reward = torch.exp(
            -self.hand_pose_reward_scale * hand_error.square()
        )
        imitation_reward = (
            self.ee_position_reward_weight * position_reward
            + self.ee_rotation_reward_weight * rotation_reward
            + self.hand_pose_reward_weight * hand_reward
        )
        arm_actions = self.actions[:, : self.num_arm_dofs]
        hand_actions = self.actions[:, self.num_arm_dofs : self.num_hand_arm_dofs]
        arm_delta = self.action_deltas[:, : self.num_arm_dofs]
        hand_delta = self.action_deltas[
            :, self.num_arm_dofs : self.num_hand_arm_dofs
        ]
        arm_action_penalty = (
            -self.kuka_actions_penalty_scale
            * torch.sum(arm_actions.square(), dim=-1)
        )
        hand_action_penalty = (
            -self.hand_actions_penalty_scale
            * torch.sum(hand_actions.square(), dim=-1)
        )
        arm_delta_penalty = (
            -self.arm_action_delta_penalty_scale
            * torch.sum(arm_delta.square(), dim=-1)
        )
        hand_delta_penalty = (
            -self.hand_action_delta_penalty_scale
            * torch.sum(hand_delta.square(), dim=-1)
        )
        reward = (
            imitation_reward
            + arm_action_penalty
            + hand_action_penalty
            + arm_delta_penalty
            + hand_delta_penalty
        )
        self.rew_buf[:] = reward

        finished = self.phase >= 1.0
        position_diverged = (
            position_error > self.ee_position_termination_distance
        )
        rotation_diverged = (
            rotation_error > self.ee_rotation_termination_angle
        )
        hand_diverged = hand_error > self.hand_termination_error
        diverged = torch.zeros_like(finished)
        if self.early_termination:
            diverged = position_diverged | rotation_diverged | hand_diverged
        self.reset_buf[:] = finished | diverged
        self.reset_goal_buf[:] = False

        components = {
            "ee_position_reward": self.ee_position_reward_weight * position_reward,
            "ee_rotation_reward": self.ee_rotation_reward_weight * rotation_reward,
            "hand_pose_reward": self.hand_pose_reward_weight * hand_reward,
            "imitation_reward": imitation_reward,
            "kuka_actions_penalty": arm_action_penalty,
            "hand_actions_penalty": hand_action_penalty,
            "arm_action_delta_penalty": arm_delta_penalty,
            "hand_action_delta_penalty": hand_delta_penalty,
            "total_reward": reward,
            "episode_steps": torch.ones_like(reward),
        }
        for name, value in components.items():
            self.rewards_episode[name] += value
        self.extras["rewards_episode"] = self.rewards_episode
        # RLGPUAlgoObserver treats these as current-step values. It logs their
        # means under reward_step/* and accumulates them until each episode ends.
        self.extras["episode_cumulative"] = components
        self.extras["imitation/position_error_m"] = position_error.mean()
        self.extras["imitation/rotation_error_rad"] = rotation_error.mean()
        self.extras["imitation/hand_error_rad"] = hand_error.mean()
        self.extras["imitation/phase"] = self.phase.mean()
        self.extras["imitation/early_termination_fraction"] = diverged.float().mean()
        self.extras["imitation/completion_fraction"] = finished.float().mean()
        self.extras["imitation/termination_position_fraction"] = (
            position_diverged.float().mean()
        )
        self.extras["imitation/termination_rotation_fraction"] = (
            rotation_diverged.float().mean()
        )
        self.extras["imitation/termination_hand_fraction"] = (
            hand_diverged.float().mean()
        )
        self.extras["control/arm_action_rms"] = torch.sqrt(
            torch.mean(arm_actions.square())
        )
        self.extras["control/hand_action_rms"] = torch.sqrt(
            torch.mean(hand_actions.square())
        )
        self.extras["control/arm_action_delta_rms"] = torch.sqrt(
            torch.mean(arm_delta.square())
        )
        self.extras["control/hand_action_delta_rms"] = torch.sqrt(
            torch.mean(hand_delta.square())
        )
        self.extras["control/osc_joint_delta_clipped_fraction"] = (
            self.operational_space_joint_delta_clipped.float().mean()
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

    def _set_reference_visualization_robot(
        self,
        reference: ReferenceSample,
        *,
        flush: bool = True,
    ) -> None:
        """Set the green robot before simulation updates its link transforms."""
        if not self.VISUALIZE_REFERENCE_ROBOT:
            return

        env_id = int(self.index_to_view)
        reference_q = torch.cat(
            [
                reference.arm_q[env_id],
                reference.hand_q[env_id],
            ],
            dim=-1,
        )
        self.visualization_robot_arm_hand_dof_pos[env_id].copy_(reference_q)
        self.visualization_robot_arm_hand_dof_vel[env_id].zero_()
        self.cur_targets[
            env_id, self.num_hand_arm_dofs :
        ].copy_(reference_q)

        actor_index = self.visualization_robot_indices[env_id : env_id + 1].to(
            torch.int32
        )
        self.deferred_set_dof_state_tensor_indexed([actor_index])
        if flush:
            self.set_dof_state_tensor_indexed()
            self.gym.set_dof_position_target_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.cur_targets),
            )

    def _update_reference_visualization_robot(self) -> None:
        """Compatibility helper for explicitly positioning the current reference."""
        self._set_reference_visualization_robot(self.current_reference)

    def pre_physics_step(
        self, actions, joint_pos_targets: Optional[Tensor] = None
    ) -> None:
        super().pre_physics_step(actions, joint_pos_targets=joint_pos_targets)
        if self.VISUALIZE_REFERENCE_ROBOT:
            # post_physics_step advances phase before rendering, so place the
            # green actor at that same upcoming reference before PhysX runs.
            next_phase = (self.phase + self.phase_delta).clamp(max=1.0)
            self._set_reference_visualization_robot(
                self.reference.sample(next_phase)
            )

    def post_physics_step(self) -> None:
        self.frame_since_restart += 1
        self.progress_buf += 1
        self.randomize_buf += 1
        self.phase.add_(self.phase_delta).clamp_(max=1.0)
        self.populate_sim_buffers()
        self.current_reference = self.reference.sample(self.phase)
        _, finished = self.compute_imitation_reward()
        self.populate_obs_and_states_buffers()
        self.clamp_obs()
        self._capture_video_if_needed()
