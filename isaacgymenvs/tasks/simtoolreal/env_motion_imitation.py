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
from isaacgymenvs.utils.torch_jit_utils import (
    quat_conjugate,
    quat_mul,
    quat_rotate,
    tensor_clamp,
    unscale,
)


class SimToolRealMotionImitation(SimToolReal):
    """Track a teleoperation clip, optionally including its physical object."""

    RIGHT_HAND_60_DEG_ASSET = (
        "urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
    )
    REGULARIZATION_SCALE_CONFIGS = (
        ("kuka_actions_penalty_scale", "KukaActionsPenaltyScale"),
        ("hand_actions_penalty_scale", "HandActionsPenaltyScale"),
        ("arm_action_delta_penalty_scale", "ArmActionDeltaPenaltyScale"),
        ("hand_action_delta_penalty_scale", "HandActionDeltaPenaltyScale"),
        ("arm_joint_velocity_penalty_scale", "ArmJointVelocityPenaltyScale"),
        (
            "arm_joint_acceleration_penalty_scale",
            "ArmJointAccelerationPenaltyScale",
        ),
        (
            "hand_joint_acceleration_penalty_scale",
            "HandJointAccelerationPenaltyScale",
        ),
    )
    GAUSSIAN_REGULARIZATION_CONFIGS = (
        ("kuka_actions", "kukaActions"),
        ("hand_actions", "handActions"),
        ("arm_action_delta", "armActionDelta"),
        ("hand_action_delta", "handActionDelta"),
        ("arm_joint_velocity", "armJointVelocity"),
        ("arm_joint_acceleration", "armJointAcceleration"),
        ("hand_joint_acceleration", "handJointAcceleration"),
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
        self.object_tracking_enabled = bool(
            env_cfg.get("objectTrackingEnabled", False)
        )
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
            velocity_filter_window_s=float(
                env_cfg.get("demonstrationVelocityFilterWindowS", 0.0)
            ),
            load_object_pose=self.object_tracking_enabled,
            object_position_offset_m=tuple(
                env_cfg.get("demonstrationObjectPositionOffset", [0.0, 0.0, 0.0])
            ),
            object_orientation_offset_xyzw=tuple(
                env_cfg.get(
                    "demonstrationObjectOrientationOffsetQuatXyzw",
                    [0.0, 0.0, 0.0, 1.0],
                )
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
            if self.reference.velocity_filter_window_s > 0.0:
                self.reference.recompute_hand_velocity()
            else:
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
        self.reference_init_anchor_probability = float(
            env_cfg.get("referenceInitAnchorProbability", 0.0)
        )
        self.reference_init_anchor_phase = float(
            env_cfg.get("referenceInitAnchorPhase", 0.0)
        )
        self.reference_init_anchor_jitter = float(
            env_cfg.get("referenceInitAnchorJitter", 0.0)
        )
        if not 0.0 <= self.reference_init_anchor_probability <= 1.0:
            raise ValueError("referenceInitAnchorProbability must be in [0, 1]")
        if self.reference_init_anchor_jitter < 0.0:
            raise ValueError("referenceInitAnchorJitter must be non-negative")
        if self.reference_init_anchor_probability > 0.0:
            if not self.use_reference_state_initialization:
                raise ValueError(
                    "referenceInitAnchorProbability requires "
                    "useReferenceStateInitialization"
                )
            if not 0.0 <= self.reference_init_anchor_phase <= 1.0:
                raise ValueError(
                    "referenceInitAnchorPhase must be within [0, 1]"
                )
        self.reference_anchor_reset_count = 0

        self.ee_position_reward_weight = float(env_cfg["eePositionRewardWeight"])
        self.ee_rotation_reward_weight = float(env_cfg["eeRotationRewardWeight"])
        self.hand_pose_reward_weight = float(env_cfg["handPoseRewardWeight"])
        self.ee_position_reward_scale = float(env_cfg["eePositionRewardScale"])
        self.ee_rotation_reward_scale = float(env_cfg["eeRotationRewardScale"])
        self.hand_pose_reward_scale = float(env_cfg["handPoseRewardScale"])
        self.robot_imitation_reward_weight = float(
            env_cfg.get("robotImitationRewardWeight", 1.0)
        )
        self.object_keypoint_reward_weight = float(
            env_cfg.get("objectKeypointRewardWeight", 0.0)
        )
        self.object_keypoint_reward_scale = float(
            env_cfg.get("objectKeypointRewardScale", 100.0)
        )
        for name, value in (
            ("robotImitationRewardWeight", self.robot_imitation_reward_weight),
            ("objectKeypointRewardWeight", self.object_keypoint_reward_weight),
            ("objectKeypointRewardScale", self.object_keypoint_reward_scale),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.object_tracking_enabled:
            if self.reference.object_pos is None:
                raise ValueError(
                    "objectTrackingEnabled requires object poses in the demonstration"
                )
            if self.object_keypoint_reward_weight <= 0.0:
                raise ValueError(
                    "objectTrackingEnabled requires objectKeypointRewardWeight > 0"
                )
            if self.num_keypoints != 4:
                raise ValueError(
                    "Object imitation expects the four corresponding keypoints "
                    f"used by object-to-goal, got {self.num_keypoints}"
                )
        weight_sum = (
            self.ee_position_reward_weight
            + self.ee_rotation_reward_weight
            + self.hand_pose_reward_weight
        )
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("Imitation reward weights must sum to 1")

        self.velocity_tracking_enabled = bool(
            env_cfg.get("velocityTrackingEnabled", False)
        )
        self.pose_imitation_reward_weight = 1.0
        self.velocity_imitation_reward_weight = 0.0
        if self.velocity_tracking_enabled:
            self.pose_imitation_reward_weight = float(
                env_cfg["poseImitationRewardWeight"]
            )
            self.velocity_imitation_reward_weight = float(
                env_cfg["velocityImitationRewardWeight"]
            )
            outer_weight_sum = (
                self.pose_imitation_reward_weight
                + self.velocity_imitation_reward_weight
            )
            if abs(outer_weight_sum - 1.0) > 1e-6:
                raise ValueError(
                    "Pose and velocity imitation reward weights must sum to 1"
                )

            self.palm_linear_velocity_reward_weight = float(
                env_cfg["palmLinearVelocityRewardWeight"]
            )
            self.palm_angular_velocity_reward_weight = float(
                env_cfg["palmAngularVelocityRewardWeight"]
            )
            self.hand_velocity_reward_weight = float(
                env_cfg["handVelocityRewardWeight"]
            )
            velocity_weight_sum = (
                self.palm_linear_velocity_reward_weight
                + self.palm_angular_velocity_reward_weight
                + self.hand_velocity_reward_weight
            )
            if abs(velocity_weight_sum - 1.0) > 1e-6:
                raise ValueError("Velocity reward weights must sum to 1")

            self.palm_linear_velocity_reward_scale = float(
                env_cfg["palmLinearVelocityRewardScale"]
            )
            self.palm_angular_velocity_reward_scale = float(
                env_cfg["palmAngularVelocityRewardScale"]
            )
            self.hand_velocity_reward_scale = float(
                env_cfg["handVelocityRewardScale"]
            )
            self.velocity_tracking_window_steps = int(
                env_cfg.get("velocityTrackingWindowSteps", 0)
            )
            self.velocity_reward_warmup_steps = int(
                env_cfg.get("velocityRewardWarmupSteps", 0)
            )
            if self.velocity_tracking_window_steps < 0:
                raise ValueError(
                    "velocityTrackingWindowSteps must be non-negative"
                )
            if self.velocity_reward_warmup_steps < 0:
                raise ValueError(
                    "velocityRewardWarmupSteps must be non-negative"
                )

        self.early_termination = bool(env_cfg.get("imitationEarlyTermination", True))
        self.ee_position_termination_distance = float(
            env_cfg["eePositionTerminationDistance"]
        )
        self.ee_rotation_termination_angle = float(
            env_cfg["eeRotationTerminationAngleRad"]
        )
        self.hand_termination_error = float(env_cfg["handTerminationError"])
        self.object_early_termination_enabled = bool(
            env_cfg.get("objectEarlyTerminationEnabled", False)
        )
        self.object_position_termination_distance = float(
            env_cfg.get("objectPositionTerminationDistance", 0.0)
        )
        self.object_termination_grace_steps = int(
            env_cfg.get("objectTerminationGraceSteps", 0)
        )
        if (
            self.object_early_termination_enabled
            and not self.object_tracking_enabled
        ):
            raise ValueError(
                "objectEarlyTerminationEnabled requires objectTrackingEnabled"
            )
        if (
            self.object_early_termination_enabled
            and self.object_position_termination_distance <= 0.0
        ):
            raise ValueError(
                "objectPositionTerminationDistance must be positive when object "
                "early termination is enabled"
            )
        if self.object_termination_grace_steps < 0:
            raise ValueError("objectTerminationGraceSteps must be non-negative")
        self.arm_joint_velocity_penalty_scale = float(
            env_cfg.get("armJointVelocityPenaltyScale", 0.0)
        )
        self.arm_joint_acceleration_penalty_scale = float(
            env_cfg.get("armJointAccelerationPenaltyScale", 0.0)
        )
        self.hand_joint_acceleration_penalty_scale = float(
            env_cfg.get("handJointAccelerationPenaltyScale", 0.0)
        )
        for name, scale in (
            ("armJointVelocityPenaltyScale", self.arm_joint_velocity_penalty_scale),
            (
                "armJointAccelerationPenaltyScale",
                self.arm_joint_acceleration_penalty_scale,
            ),
            (
                "handJointAccelerationPenaltyScale",
                self.hand_joint_acceleration_penalty_scale,
            ),
        ):
            if scale < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self._init_positive_gaussian_regularization(env_cfg)
        self._init_regularization_curriculum(env_cfg)
        # The palm-center OSC needs the current wrist orientation on the very
        # first pre-physics step; the base OSC does not require this state.
        self.populate_sim_buffers()
        self.previous_imitation_joint_vel = self.arm_hand_dof_vel.clone()
        self.current_reference: ReferenceSample = self.reference.sample(self.phase)
        if self.velocity_tracking_enabled:
            self.velocity_reward_age = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        if (
            self.velocity_tracking_enabled
            and self.velocity_tracking_window_steps > 0
        ):
            history_shape = (
                self.num_envs,
                self.velocity_tracking_window_steps,
            )
            self.velocity_history_palm_pos = torch.zeros(
                *history_shape, 3, dtype=torch.float, device=self.device
            )
            self.velocity_history_palm_quat = torch.zeros(
                *history_shape, 4, dtype=torch.float, device=self.device
            )
            self.velocity_history_hand_q = torch.zeros(
                *history_shape,
                self.num_hand_dofs,
                dtype=torch.float,
                device=self.device,
            )
            self.velocity_history_valid = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.velocity_history_age = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.velocity_history_cursor = 0

        for key in (
            "ee_position_reward",
            "ee_rotation_reward",
            "hand_pose_reward",
            "imitation_reward",
            "object_keypoint_reward",
            "arm_joint_velocity_penalty",
            "arm_joint_acceleration_penalty",
            "hand_joint_acceleration_penalty",
            "episode_steps",
        ):
            if key not in self.rewards_episode:
                self.rewards_episode[key] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )
        if self.positive_gaussian_regularization_enabled:
            for name, _ in self.GAUSSIAN_REGULARIZATION_CONFIGS:
                key = f"{name}_regularization_reward"
                if key not in self.rewards_episode:
                    self.rewards_episode[key] = torch.zeros(
                        self.num_envs, dtype=torch.float, device=self.device
                    )
        if self.velocity_tracking_enabled:
            for key in (
                "pose_imitation_reward",
                "palm_linear_velocity_reward",
                "palm_angular_velocity_reward",
                "hand_velocity_reward",
                "velocity_imitation_reward",
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
        use_anchor_phase = torch.zeros_like(use_random_phase)
        if self.reference_init_anchor_probability > 0.0:
            use_anchor_phase = use_random_phase & (
                torch.rand(len(env_ids), device=self.device)
                < self.reference_init_anchor_probability
            )
            anchor_phase = torch.full_like(
                sampled_phase, self.reference_init_anchor_phase
            )
            if self.reference_init_anchor_jitter > 0.0:
                anchor_phase += (
                    2.0 * torch.rand_like(anchor_phase) - 1.0
                ) * self.reference_init_anchor_jitter
            anchor_phase.clamp_(0.0, 1.0)
            sampled_phase = torch.where(
                use_anchor_phase, anchor_phase, sampled_phase
            )
        self.phase[env_ids] = torch.where(
            use_random_phase, sampled_phase, torch.zeros_like(sampled_phase)
        )
        self.reference_state_reset_mask[env_ids] = use_random_phase
        self.reference_state_reset_count += int(use_random_phase.sum().item())
        self.regular_state_reset_count += int((~use_random_phase).sum().item())
        self.reference_anchor_reset_count += int(use_anchor_phase.sum().item())

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
        scalars["rsi/anchor_reset_fraction"] = float(
            use_anchor_phase.float().mean().item()
        )
        scalars["rsi/anchor_reset_count_total"] = int(
            self.reference_anchor_reset_count
        )
        scalars["rsi/anchor_start_phase_mean"] = (
            float(sampled_phase[use_anchor_phase].mean().item())
            if bool(use_anchor_phase.any().item())
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
        if hasattr(self, "previous_imitation_joint_vel"):
            self.previous_imitation_joint_vel[env_ids] = robot_vel
        if hasattr(self, "velocity_history_valid"):
            self.velocity_history_valid[env_ids] = False
            self.velocity_history_age[env_ids] = 0
        if hasattr(self, "velocity_reward_age"):
            self.velocity_reward_age[env_ids] = 0

        if self.object_tracking_enabled:
            if any(
                value is None
                for value in (
                    reference.object_pos,
                    reference.object_quat_xyzw,
                    reference.object_lin_vel,
                    reference.object_ang_vel,
                )
            ):
                raise RuntimeError("Object reference sample is incomplete")
            object_indices = self.object_indices[env_ids]
            self.root_state_tensor[object_indices, 0:3] = reference.object_pos
            self.root_state_tensor[
                object_indices, 3:7
            ] = reference.object_quat_xyzw
            self.root_state_tensor[object_indices, 7:10] = reference.object_lin_vel
            self.root_state_tensor[object_indices, 10:13] = reference.object_ang_vel
            self.deferred_set_actor_root_state_tensor_indexed([object_indices])

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
            self.set_actor_root_state_tensor_indexed()
            self.populate_sim_buffers()
            self.populate_obs_and_states_buffers()
            self.clamp_obs()

    def _reference_object_keypoints(self, reference: ReferenceSample) -> Tensor:
        """Transform the four object-to-goal keypoints by a reference pose."""
        if reference.object_pos is None or reference.object_quat_xyzw is None:
            raise RuntimeError("Object tracking requires a complete object pose")
        offsets = self.object_keypoint_offsets_fixed_size
        num_keypoints = offsets.shape[1]
        reference_quat = reference.object_quat_xyzw.unsqueeze(1).expand(
            -1, num_keypoints, -1
        )
        rotated_offsets = quat_rotate(
            reference_quat.reshape(-1, 4), offsets.reshape(-1, 3)
        ).reshape(self.num_envs, num_keypoints, 3)
        return reference.object_pos.unsqueeze(1) + rotated_offsets

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

    def _palm_center_velocity(self) -> Tuple[Tensor, Tensor]:
        """Return palm-center linear and angular velocity in world coordinates."""
        link_linear_velocity = self._palm_state[:, 7:10]
        angular_velocity = self._palm_state[:, 10:13]
        world_offset = quat_rotate(
            self._palm_link_rot,
            self.palm_center_offset,
        )
        center_linear_velocity = link_linear_velocity + torch.cross(
            angular_velocity,
            world_offset,
            dim=-1,
        )
        return center_linear_velocity, angular_velocity

    @staticmethod
    def _quaternion_interval_angular_velocity(
        current: Tensor,
        previous: Tensor,
        interval_s: Tensor,
    ) -> Tensor:
        """Return shortest-path world-frame angular velocity over an interval."""
        delta = quat_mul(current, quat_conjugate(previous))
        delta = torch.where(delta[:, 3:4] < 0.0, -delta, delta)
        vector = delta[:, :3]
        vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(
            vector_norm,
            delta[:, 3:4].clamp(min=0.0),
        )
        rotation_vector = torch.where(
            vector_norm > 1e-8,
            vector * (angle / vector_norm.clamp(min=1e-8)),
            2.0 * vector,
        )
        return rotation_vector / interval_s.unsqueeze(-1)

    def _matched_window_velocities(
        self,
        ref: ReferenceSample,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Estimate simulated and reference velocities over the same interval."""
        cursor = self.velocity_history_cursor
        invalid = ~self.velocity_history_valid
        self.velocity_history_palm_pos[invalid] = self.palm_center_pos[
            invalid
        ].unsqueeze(1)
        self.velocity_history_palm_quat[invalid] = self._palm_rot[
            invalid
        ].unsqueeze(1)
        self.velocity_history_hand_q[invalid] = self.arm_hand_dof_pos[
            invalid, self.num_arm_dofs : self.num_hand_arm_dofs
        ].unsqueeze(1)
        self.velocity_history_valid[invalid] = True

        previous_palm_pos = self.velocity_history_palm_pos[:, cursor]
        previous_palm_quat = self.velocity_history_palm_quat[:, cursor]
        previous_hand_q = self.velocity_history_hand_q[:, cursor]
        interval_steps = self.velocity_history_age.clamp(
            max=self.velocity_tracking_window_steps
        )
        safe_interval_s = (
            interval_steps.clamp(min=1).to(dtype=torch.float) * self.dt
        )

        palm_linear_velocity = (
            self.palm_center_pos - previous_palm_pos
        ) / safe_interval_s.unsqueeze(-1)
        palm_angular_velocity = self._quaternion_interval_angular_velocity(
            self._palm_rot,
            previous_palm_quat,
            safe_interval_s,
        )
        hand_velocity = (
            self.arm_hand_dof_pos[
                :, self.num_arm_dofs : self.num_hand_arm_dofs
            ]
            - previous_hand_q
        ) / safe_interval_s.unsqueeze(-1)

        previous_phase = (
            self.phase
            - interval_steps.to(dtype=self.phase.dtype) * self.phase_delta
        ).clamp(min=0.0)
        previous_ref = self.reference.sample(previous_phase)
        reference_linear_velocity = (
            ref.palm_pos - previous_ref.palm_pos
        ) / safe_interval_s.unsqueeze(-1)
        reference_angular_velocity = (
            self._quaternion_interval_angular_velocity(
                ref.palm_quat_xyzw,
                previous_ref.palm_quat_xyzw,
                safe_interval_s,
            )
        )
        reference_hand_velocity = (
            ref.hand_q - previous_ref.hand_q
        ) / safe_interval_s.unsqueeze(-1)

        # The first post-reset sample has no elapsed interval on either side.
        no_history = interval_steps == 0
        for velocity in (
            palm_linear_velocity,
            palm_angular_velocity,
            hand_velocity,
            reference_linear_velocity,
            reference_angular_velocity,
            reference_hand_velocity,
        ):
            velocity[no_history] = 0.0

        self.velocity_history_palm_pos[:, cursor].copy_(self.palm_center_pos)
        self.velocity_history_palm_quat[:, cursor].copy_(self._palm_rot)
        self.velocity_history_hand_q[:, cursor].copy_(
            self.arm_hand_dof_pos[
                :, self.num_arm_dofs : self.num_hand_arm_dofs
            ]
        )
        self.velocity_history_age.add_(1).clamp_(
            max=self.velocity_tracking_window_steps
        )
        self.velocity_history_cursor = (
            cursor + 1
        ) % self.velocity_tracking_window_steps

        return (
            palm_linear_velocity,
            palm_angular_velocity,
            hand_velocity,
            reference_linear_velocity,
            reference_angular_velocity,
            reference_hand_velocity,
        )

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
        pose_imitation_reward = (
            self.ee_position_reward_weight * position_reward
            + self.ee_rotation_reward_weight * rotation_reward
            + self.hand_pose_reward_weight * hand_reward
        )
        imitation_reward = pose_imitation_reward

        velocity_components = {}
        velocity_errors = {}
        pose_component_scale = torch.ones_like(pose_imitation_reward)
        if self.velocity_tracking_enabled:
            if self.velocity_tracking_window_steps > 0:
                (
                    palm_linear_velocity,
                    palm_angular_velocity,
                    hand_velocity,
                    reference_linear_velocity,
                    reference_angular_velocity,
                    reference_hand_velocity,
                ) = self._matched_window_velocities(ref)
            else:
                palm_linear_velocity, palm_angular_velocity = (
                    self._palm_center_velocity()
                )
                hand_velocity = self.arm_hand_dof_vel[
                    :, self.num_arm_dofs :
                ]
                reference_linear_velocity = ref.palm_lin_vel
                reference_angular_velocity = ref.palm_ang_vel
                reference_hand_velocity = ref.hand_dq
            palm_linear_velocity_error = torch.linalg.vector_norm(
                palm_linear_velocity - reference_linear_velocity,
                dim=-1,
            )
            palm_angular_velocity_error = torch.linalg.vector_norm(
                palm_angular_velocity - reference_angular_velocity,
                dim=-1,
            )
            hand_velocity_error = torch.linalg.vector_norm(
                hand_velocity - reference_hand_velocity,
                dim=-1,
            )
            palm_linear_velocity_reward = torch.exp(
                -self.palm_linear_velocity_reward_scale
                * palm_linear_velocity_error.square()
            )
            palm_angular_velocity_reward = torch.exp(
                -self.palm_angular_velocity_reward_scale
                * palm_angular_velocity_error.square()
            )
            hand_velocity_reward = torch.exp(
                -self.hand_velocity_reward_scale
                * hand_velocity_error.square()
            )
            velocity_imitation_reward = (
                self.palm_linear_velocity_reward_weight
                * palm_linear_velocity_reward
                + self.palm_angular_velocity_reward_weight
                * palm_angular_velocity_reward
                + self.hand_velocity_reward_weight
                * hand_velocity_reward
            )
            if self.velocity_reward_warmup_steps > 0:
                velocity_warmup_factor = (
                    self.velocity_reward_age.to(dtype=torch.float)
                    / float(self.velocity_reward_warmup_steps)
                ).clamp(max=1.0)
            else:
                velocity_warmup_factor = torch.ones_like(
                    pose_imitation_reward
                )
            effective_velocity_weight = (
                self.velocity_imitation_reward_weight
                * velocity_warmup_factor
            )
            pose_component_scale = 1.0 - effective_velocity_weight
            imitation_reward = (
                pose_component_scale * pose_imitation_reward
                + effective_velocity_weight * velocity_imitation_reward
            )
            velocity_components = {
                "pose_imitation_reward": (
                    pose_component_scale * pose_imitation_reward
                ),
                "palm_linear_velocity_reward": (
                    effective_velocity_weight
                    * self.palm_linear_velocity_reward_weight
                    * palm_linear_velocity_reward
                ),
                "palm_angular_velocity_reward": (
                    effective_velocity_weight
                    * self.palm_angular_velocity_reward_weight
                    * palm_angular_velocity_reward
                ),
                "hand_velocity_reward": (
                    effective_velocity_weight
                    * self.hand_velocity_reward_weight
                    * hand_velocity_reward
                ),
                "velocity_imitation_reward": velocity_imitation_reward,
            }
            velocity_errors = {
                "linear_velocity_error_mps": palm_linear_velocity_error,
                "angular_velocity_error_radps": palm_angular_velocity_error,
                "hand_velocity_error_radps": hand_velocity_error,
            }
            self.velocity_reward_age.add_(1)
        imitation_reward = (
            self.robot_imitation_reward_weight * imitation_reward
        )
        velocity_components = {
            name: self.robot_imitation_reward_weight * value
            for name, value in velocity_components.items()
        }

        object_keypoint_distances = torch.zeros(
            self.num_envs, self.num_keypoints, device=self.device
        )
        object_keypoint_max_error = torch.zeros_like(imitation_reward)
        object_keypoint_mean_error = torch.zeros_like(imitation_reward)
        object_keypoint_reward = torch.zeros_like(imitation_reward)
        object_position_error = torch.zeros_like(imitation_reward)
        if self.object_tracking_enabled:
            reference_object_keypoints = self._reference_object_keypoints(ref)
            object_position_error = torch.linalg.vector_norm(
                self.object_pos - ref.object_pos, dim=-1
            )
            object_keypoint_delta = (
                self.obj_keypoint_pos_fixed_size - reference_object_keypoints
            )
            object_keypoint_distances = torch.linalg.vector_norm(
                object_keypoint_delta, dim=-1
            )
            object_keypoint_max_error = object_keypoint_distances.max(dim=-1).values
            object_keypoint_mean_error = object_keypoint_distances.mean(dim=-1)
            object_keypoint_reward = (
                self.object_keypoint_reward_weight
                * torch.exp(
                    -self.object_keypoint_reward_scale
                    * object_keypoint_max_error.square()
                )
            )
        arm_actions = self.actions[:, : self.num_arm_dofs]
        hand_actions = self.actions[:, self.num_arm_dofs : self.num_hand_arm_dofs]
        arm_delta = self.action_deltas[:, : self.num_arm_dofs]
        hand_delta = self.action_deltas[
            :, self.num_arm_dofs : self.num_hand_arm_dofs
        ]
        arm_action_regularization = self._regularization_contribution(
            "kuka_actions", arm_actions, self.kuka_actions_penalty_scale
        )
        hand_action_regularization = self._regularization_contribution(
            "hand_actions", hand_actions, self.hand_actions_penalty_scale
        )
        arm_delta_regularization = self._regularization_contribution(
            "arm_action_delta", arm_delta, self.arm_action_delta_penalty_scale
        )
        hand_delta_regularization = self._regularization_contribution(
            "hand_action_delta", hand_delta, self.hand_action_delta_penalty_scale
        )
        joint_vel = self.arm_hand_dof_vel[:, : self.num_hand_arm_dofs]
        joint_acceleration = (
            joint_vel - self.previous_imitation_joint_vel
        ) / self.control_dt
        arm_joint_velocity_regularization = self._regularization_contribution(
            "arm_joint_velocity",
            joint_vel[:, : self.num_arm_dofs],
            self.arm_joint_velocity_penalty_scale,
        )
        arm_joint_acceleration_regularization = (
            self._regularization_contribution(
                "arm_joint_acceleration",
                joint_acceleration[:, : self.num_arm_dofs],
                self.arm_joint_acceleration_penalty_scale,
            )
        )
        hand_joint_acceleration_regularization = (
            self._regularization_contribution(
                "hand_joint_acceleration",
                joint_acceleration[:, self.num_arm_dofs :],
                self.hand_joint_acceleration_penalty_scale,
            )
        )
        self.previous_imitation_joint_vel.copy_(joint_vel)
        reward = (
            imitation_reward
            + object_keypoint_reward
            + arm_action_regularization
            + hand_action_regularization
            + arm_delta_regularization
            + hand_delta_regularization
            + arm_joint_velocity_regularization
            + arm_joint_acceleration_regularization
            + hand_joint_acceleration_regularization
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
        object_position_diverged = torch.zeros_like(finished)
        if self.object_early_termination_enabled:
            object_grace_period_finished = (
                self.progress_buf > self.object_termination_grace_steps
            )
            object_position_diverged = (
                object_position_error > self.object_position_termination_distance
            ) & object_grace_period_finished
        diverged = torch.zeros_like(finished)
        if self.early_termination:
            diverged = (
                position_diverged
                | rotation_diverged
                | hand_diverged
                | object_position_diverged
            )
        self.reset_buf[:] = finished | diverged
        self.reset_goal_buf[:] = False

        components = {
            "ee_position_reward": (
                self.robot_imitation_reward_weight
                * pose_component_scale
                * self.ee_position_reward_weight
                * position_reward
            ),
            "ee_rotation_reward": (
                self.robot_imitation_reward_weight
                * pose_component_scale
                * self.ee_rotation_reward_weight
                * rotation_reward
            ),
            "hand_pose_reward": (
                self.robot_imitation_reward_weight
                * pose_component_scale
                * self.hand_pose_reward_weight
                * hand_reward
            ),
            "imitation_reward": imitation_reward,
            "object_keypoint_reward": object_keypoint_reward,
            "total_reward": reward,
            "episode_steps": torch.ones_like(reward),
        }
        regularization_values = {
            "kuka_actions": arm_action_regularization,
            "hand_actions": hand_action_regularization,
            "arm_action_delta": arm_delta_regularization,
            "hand_action_delta": hand_delta_regularization,
            "arm_joint_velocity": arm_joint_velocity_regularization,
            "arm_joint_acceleration": arm_joint_acceleration_regularization,
            "hand_joint_acceleration": hand_joint_acceleration_regularization,
        }
        if self.positive_gaussian_regularization_enabled:
            components.update(
                {
                    f"{name}_regularization_reward": value
                    for name, value in regularization_values.items()
                }
            )
        else:
            components.update(
                {
                    f"{name}_penalty": value
                    for name, value in regularization_values.items()
                }
            )
        components.update(velocity_components)
        for name, value in components.items():
            self.rewards_episode[name] += value
        self.extras["rewards_episode"] = self.rewards_episode
        # RLGPUAlgoObserver treats these as current-step values. It logs their
        # means under reward_step/* and accumulates them until each episode ends.
        self.extras["episode_cumulative"] = components
        self.extras["imitation/position_error_m"] = position_error.mean()
        self.extras["imitation/rotation_error_rad"] = rotation_error.mean()
        self.extras["imitation/hand_error_rad"] = hand_error.mean()
        if self.object_tracking_enabled:
            self.extras["object_tracking/keypoint_mean_error_m"] = (
                object_keypoint_mean_error.mean()
            )
            self.extras["object_tracking/keypoint_max_error_m"] = (
                object_keypoint_max_error.mean()
            )
            self.extras["object_tracking/keypoint_reward"] = (
                object_keypoint_reward.mean()
            )
            self.extras["object_tracking/position_error_m"] = (
                object_position_error.mean()
            )
        for name, value in velocity_errors.items():
            self.extras[f"imitation/{name}"] = value.mean()
        if self.velocity_tracking_enabled:
            self.extras["imitation/velocity_reward_warmup_factor"] = (
                velocity_warmup_factor.mean()
            )
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
        if self.object_tracking_enabled:
            self.extras["object_tracking/termination_position_fraction"] = (
                object_position_diverged.float().mean()
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
        self.extras["control/arm_joint_velocity_rms_radps"] = torch.sqrt(
            torch.mean(joint_vel[:, : self.num_arm_dofs].square())
        )
        self.extras["control/arm_joint_acceleration_rms_radps2"] = torch.sqrt(
            torch.mean(
                joint_acceleration[:, : self.num_arm_dofs].square()
            )
        )
        self.extras["control/hand_joint_acceleration_rms_radps2"] = torch.sqrt(
            torch.mean(joint_acceleration[:, self.num_arm_dofs :].square())
        )
        self._log_positive_gaussian_regularization()
        self._log_regularization_curriculum()
        self.extras["control/osc_joint_delta_clipped_fraction"] = (
            self.operational_space_joint_delta_clipped.float().mean()
        )
        return reward, finished

    def _init_positive_gaussian_regularization(self, env_cfg) -> None:
        self.positive_gaussian_regularization_enabled = bool(
            env_cfg.get("positiveGaussianRegularizationEnabled", False)
        )
        self.gaussian_regularization_scales = {}
        self.gaussian_regularization_sigmas = {}
        for name, config_prefix in self.GAUSSIAN_REGULARIZATION_CONFIGS:
            scale = float(
                env_cfg.get(f"{config_prefix}RegularizationScale", 0.0)
            )
            sigma = float(
                env_cfg.get(f"{config_prefix}RegularizationSigma", 1.0)
            )
            if scale < 0.0:
                raise ValueError(
                    f"{config_prefix}RegularizationScale must be non-negative"
                )
            if sigma <= 0.0:
                raise ValueError(
                    f"{config_prefix}RegularizationSigma must be positive"
                )
            self.gaussian_regularization_scales[name] = scale
            self.gaussian_regularization_sigmas[name] = sigma

    def _regularization_contribution(
        self,
        name: str,
        value: Tensor,
        legacy_penalty_scale: float,
    ) -> Tensor:
        squared_norm = torch.sum(value.square(), dim=-1)
        if not self.positive_gaussian_regularization_enabled:
            return -legacy_penalty_scale * squared_norm
        scale = self.gaussian_regularization_scales[name]
        sigma = self.gaussian_regularization_sigmas[name]
        return scale * torch.exp(-squared_norm / sigma)

    def _log_positive_gaussian_regularization(self) -> None:
        if not self.positive_gaussian_regularization_enabled:
            return
        for name, _ in self.GAUSSIAN_REGULARIZATION_CONFIGS:
            self.extras[f"gaussian_regularization/{name}_scale"] = (
                self.gaussian_regularization_scales[name]
            )
            self.extras[f"gaussian_regularization/{name}_sigma"] = (
                self.gaussian_regularization_sigmas[name]
            )

    def _init_regularization_curriculum(self, env_cfg) -> None:
        self.regularization_curriculum_enabled = bool(
            env_cfg.get("regularizationCurriculumEnabled", False)
        )
        self.regularization_curriculum_warmup_steps = int(
            env_cfg.get("regularizationCurriculumWarmupSteps", 0)
        )
        self.regularization_curriculum_ramp_steps = int(
            env_cfg.get("regularizationCurriculumRampSteps", 0)
        )
        self.regularization_curriculum_step = 0
        if self.regularization_curriculum_warmup_steps < 0:
            raise ValueError(
                "regularizationCurriculumWarmupSteps must be non-negative"
            )
        if (
            self.regularization_curriculum_enabled
            and self.regularization_curriculum_ramp_steps <= 0
        ):
            raise ValueError(
                "regularizationCurriculumRampSteps must be positive when the "
                "regularization curriculum is enabled"
            )

        self.regularization_curriculum_initial_scales = {}
        self.regularization_curriculum_target_scales = {}
        for attribute, config_suffix in self.REGULARIZATION_SCALE_CONFIGS:
            initial_scale = float(getattr(self, attribute))
            target_scale = float(
                env_cfg.get(
                    f"regularizationCurriculumTarget{config_suffix}",
                    initial_scale,
                )
            )
            if target_scale < 0.0:
                raise ValueError(
                    f"regularizationCurriculumTarget{config_suffix} must be "
                    "non-negative"
                )
            self.regularization_curriculum_initial_scales[attribute] = (
                initial_scale
            )
            self.regularization_curriculum_target_scales[attribute] = target_scale
        self._update_regularization_curriculum_scales()

    def _regularization_curriculum_alpha(self) -> float:
        if not self.regularization_curriculum_enabled:
            return 0.0
        ramp_progress = (
            self.regularization_curriculum_step
            - self.regularization_curriculum_warmup_steps
        ) / self.regularization_curriculum_ramp_steps
        ramp_progress = min(max(ramp_progress, 0.0), 1.0)
        # Smoothstep has zero slope at both ends, avoiding sudden changes in
        # the reward gradient when the ramp starts or reaches its target.
        return ramp_progress * ramp_progress * (3.0 - 2.0 * ramp_progress)

    def _update_regularization_curriculum_scales(self) -> None:
        alpha = self._regularization_curriculum_alpha()
        for attribute, _ in self.REGULARIZATION_SCALE_CONFIGS:
            initial_scale = self.regularization_curriculum_initial_scales[attribute]
            target_scale = self.regularization_curriculum_target_scales[attribute]
            setattr(
                self,
                attribute,
                initial_scale + alpha * (target_scale - initial_scale),
            )

    def _log_regularization_curriculum(self) -> None:
        if not self.regularization_curriculum_enabled:
            return
        self.extras["regularization_curriculum/alpha"] = (
            self._regularization_curriculum_alpha()
        )
        self.extras["regularization_curriculum/step"] = (
            self.regularization_curriculum_step
        )
        for attribute, _ in self.REGULARIZATION_SCALE_CONFIGS:
            metric_name = attribute[: -len("_penalty_scale")]
            self.extras[f"regularization_curriculum/{metric_name}_scale"] = getattr(
                self, attribute
            )

    def get_env_state(self):
        state = super().get_env_state()
        state["regularization_curriculum_step"] = (
            self.regularization_curriculum_step
        )
        state["reference_anchor_reset_count"] = self.reference_anchor_reset_count
        return state

    def set_env_state(self, env_state) -> None:
        super().set_env_state(env_state)
        self._update_regularization_curriculum_scales()

    def populate_obs_and_states_buffers(self) -> None:
        keypoints_rel_goal = self.keypoints_rel_goal_fixed_size
        if self.object_tracking_enabled:
            keypoints_rel_goal = (
                self.obj_keypoint_pos_fixed_size
                - self._reference_object_keypoints(self.current_reference)
            )
        obs = {
            "joint_pos": unscale(
                self.arm_hand_dof_pos,
                self.arm_hand_dof_lower_limits,
                self.arm_hand_dof_upper_limits,
            ),
            "joint_vel": self.arm_hand_dof_vel,
            "prev_action_targets": self.prev_targets[:, : self.num_hand_arm_dofs],
            "palm_pos": self.palm_center_pos,
            "reference_palm_pos": self.current_reference.palm_pos,
            "palm_rot": self._palm_rot,
            "reference_palm_rot": self.current_reference.palm_quat_xyzw,
            "reference_hand_q": unscale(
                self.current_reference.hand_q,
                self.arm_hand_dof_lower_limits[self.num_arm_dofs :],
                self.arm_hand_dof_upper_limits[self.num_arm_dofs :],
            ),
            "reference_palm_lin_vel": self.current_reference.palm_lin_vel,
            "reference_palm_ang_vel": self.current_reference.palm_ang_vel,
            "reference_hand_dq": self.current_reference.hand_dq,
            "object_rot": self.object_rot,
            "object_vel": self.object_state[:, 7:13],
            "keypoints_rel_palm": self.keypoints_rel_palm.reshape(
                self.num_envs, -1
            ),
            "keypoints_rel_goal": keypoints_rel_goal.reshape(
                self.num_envs, -1
            ),
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
        self._set_reference_visualization_object(self.current_reference)

    def _set_reference_visualization_object(
        self,
        reference: ReferenceSample,
        *,
        flush: bool = True,
    ) -> None:
        """Place the green goal object at the demonstration pose in eval only."""
        if not self.object_tracking_enabled or not self.VISUALIZE_REFERENCE_ROBOT:
            return
        if reference.object_pos is None or reference.object_quat_xyzw is None:
            raise RuntimeError("Object visualization requires a reference pose")

        env_id = int(self.index_to_view)
        goal_index = self.goal_object_indices[env_id : env_id + 1]
        self.goal_states[env_id, 0:3].copy_(reference.object_pos[env_id])
        self.goal_states[env_id, 3:7].copy_(reference.object_quat_xyzw[env_id])
        self.goal_states[env_id, 7:13].zero_()
        self.root_state_tensor[goal_index, :].copy_(self.goal_states[env_id])
        self.deferred_set_actor_root_state_tensor_indexed([goal_index])
        if flush:
            self.set_actor_root_state_tensor_indexed()

    def pre_physics_step(
        self, actions, joint_pos_targets: Optional[Tensor] = None
    ) -> None:
        super().pre_physics_step(actions, joint_pos_targets=joint_pos_targets)
        if self.VISUALIZE_REFERENCE_ROBOT:
            # post_physics_step advances phase before rendering, so place the
            # green actor at that same upcoming reference before PhysX runs.
            next_phase = (self.phase + self.phase_delta).clamp(max=1.0)
            next_reference = self.reference.sample(next_phase)
            self._set_reference_visualization_robot(next_reference)
            self._set_reference_visualization_object(next_reference)

    def post_physics_step(self) -> None:
        self.frame_since_restart += 1
        if self.regularization_curriculum_enabled:
            self.regularization_curriculum_step += 1
            self._update_regularization_curriculum_scales()
        self.progress_buf += 1
        self.randomize_buf += 1
        self.phase.add_(self.phase_delta).clamp_(max=1.0)
        self.populate_sim_buffers()
        self.current_reference = self.reference.sample(self.phase)
        _, finished = self.compute_imitation_reward()
        self.populate_obs_and_states_buffers()
        self.clamp_obs()
        self._capture_video_if_needed()
