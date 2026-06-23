import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from typing import Dict, List

from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from isaacgymenvs.utils.torch_jit_utils import scale, tensor_clamp


class SimToolRealHandOnly(SimToolReal):
    """SimToolReal variant whose policy controls only the hand DOFs.

    The base task still owns the full robot state and controller. This adapter
    exposes a smaller action space to RL and expands hand-only actions back to
    the full arm+hand action vector before calling the base implementation.
    """

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
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        assert not self.privileged_actions, (
            "SimToolRealHandOnly currently expects privilegedActions=False"
        )
        assert not self.cfg["env"].get("useActionDelay", False), (
            "SimToolRealHandOnly scripted targets currently expect useActionDelay=False"
        )
        self.full_num_actions = self.num_hand_arm_dofs
        self.num_actions = self.num_hand_dofs
        self.cfg["env"]["numActions"] = self.num_actions
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0,
            np.ones(self.num_actions) * 1.0,
        )
        self._init_scripted_arm_trajectory()
        self.reset_if_not_lifted_by_deadline = self.cfg["env"].get(
            "resetIfNotLiftedByDeadline", False
        )
        self.not_lifted_deadline_step = int(
            self.cfg["env"].get("notLiftedDeadlineStep", 420)
        )
        self.not_lifted_height_threshold = float(
            self.cfg["env"].get("notLiftedHeightThreshold", 0.05)
        )
        self.reset_if_fingertips_far_after_grasp = self.cfg["env"].get(
            "resetIfFingertipsFarAfterGrasp", False
        )
        self.fingertips_far_distance_threshold = float(
            self.cfg["env"].get("fingertipsFarDistanceThreshold", 0.12)
        )
        self.fingertips_far_patience_steps = int(
            self.cfg["env"].get("fingertipsFarPatienceSteps", 30)
        )
        self.fingertips_far_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _init_scripted_arm_trajectory(self) -> None:
        trajectory_cfg = self.cfg["env"].get("scriptedArmTrajectory", {})
        waypoints = trajectory_cfg.get("waypoints", [])
        assert len(waypoints) >= 2, "scriptedArmTrajectory requires at least 2 waypoints"

        steps = []
        poses = []
        for waypoint in waypoints:
            step = int(waypoint["step"])
            pose = waypoint["q"]
            assert len(pose) == self.num_arm_dofs, (
                "Each scripted arm waypoint must have one value per arm DOF"
            )
            steps.append(step)
            poses.append(pose)

        order = np.argsort(np.array(steps))
        self.scripted_arm_steps = torch.tensor(
            [steps[i] for i in order], dtype=torch.float32, device=self.device
        )
        self.scripted_arm_poses = torch.tensor(
            [poses[i] for i in order], dtype=torch.float32, device=self.device
        )
        self.phase_obs_size = int(self.cfg["env"].get("phaseObsSize", 4))
        assert self.phase_obs_size == 4, "SimToolRealHandOnly expects phaseObsSize=4"

    def compute_scripted_arm_targets(self) -> torch.Tensor:
        progress = self.progress_buf.to(torch.float32)
        targets = self.scripted_arm_poses[0].unsqueeze(0).repeat(self.num_envs, 1)

        for idx in range(len(self.scripted_arm_steps) - 1):
            step_a = self.scripted_arm_steps[idx]
            step_b = self.scripted_arm_steps[idx + 1]
            pose_a = self.scripted_arm_poses[idx]
            pose_b = self.scripted_arm_poses[idx + 1]
            alpha = torch.clamp((progress - step_a) / (step_b - step_a), 0.0, 1.0)
            segment_targets = pose_a + alpha.unsqueeze(-1) * (pose_b - pose_a)
            in_segment = (progress >= step_a) & (progress <= step_b)
            targets = torch.where(in_segment.unsqueeze(-1), segment_targets, targets)

        after_last = progress > self.scripted_arm_steps[-1]
        targets = torch.where(
            after_last.unsqueeze(-1),
            self.scripted_arm_poses[-1].unsqueeze(0),
            targets,
        )
        return tensor_clamp(
            targets,
            self.arm_hand_dof_lower_limits[: self.num_arm_dofs],
            self.arm_hand_dof_upper_limits[: self.num_arm_dofs],
        )

    def compute_phase_obs(self) -> torch.Tensor:
        """Return one-hot [approach, grasp_hold, lift, lifted_hold]."""
        progress = self.progress_buf
        grasp_start = int(self.scripted_arm_steps[1].item())
        lift_start = int(self.scripted_arm_steps[2].item())
        lifted_hold_start = int(self.scripted_arm_steps[3].item())

        phase_idx = torch.zeros_like(progress, dtype=torch.long)
        phase_idx = torch.where(progress >= grasp_start, 1, phase_idx)
        phase_idx = torch.where(progress >= lift_start, 2, phase_idx)
        phase_idx = torch.where(progress >= lifted_hold_start, 3, phase_idx)
        return F.one_hot(phase_idx, num_classes=self.phase_obs_size).to(torch.float32)

    def expand_hand_actions(self, hand_actions: torch.Tensor) -> torch.Tensor:
        assert hand_actions.shape == (self.num_envs, self.num_hand_dofs), (
            f"Expected hand actions with shape "
            f"({self.num_envs}, {self.num_hand_dofs}), got {hand_actions.shape}"
        )
        full_actions = torch.zeros(
            (self.num_envs, self.full_num_actions),
            dtype=hand_actions.dtype,
            device=hand_actions.device,
        )
        full_actions[:, self.num_arm_dofs : self.num_hand_arm_dofs] = hand_actions
        return full_actions

    def compute_hand_targets(self, hand_actions: torch.Tensor) -> torch.Tensor:
        hand_dof_slice = slice(self.num_arm_dofs, self.num_hand_arm_dofs)
        if self.use_relative_hand_control:
            if self.use_relative_control:
                hand_reference = self.arm_hand_dof_pos[:, hand_dof_slice]
            else:
                hand_reference = self.prev_targets[:, hand_dof_slice]
            hand_targets = (
                hand_reference
                + self.relative_hand_dof_speed_scale * self.dt * hand_actions
            )
            hand_targets = tensor_clamp(
                hand_targets,
                self.arm_hand_dof_lower_limits[hand_dof_slice],
                self.arm_hand_dof_upper_limits[hand_dof_slice],
            )
        else:
            hand_targets = scale(
                hand_actions,
                self.arm_hand_dof_lower_limits[hand_dof_slice],
                self.arm_hand_dof_upper_limits[hand_dof_slice],
            )

        hand_targets = (
            self.hand_moving_average * hand_targets
            + (1.0 - self.hand_moving_average) * self.prev_targets[:, hand_dof_slice]
        )
        return tensor_clamp(
            hand_targets,
            self.arm_hand_dof_lower_limits[hand_dof_slice],
            self.arm_hand_dof_upper_limits[hand_dof_slice],
        )

    def compute_joint_pos_targets(self, hand_actions: torch.Tensor) -> torch.Tensor:
        joint_pos_targets = self.prev_targets[:, : self.num_hand_arm_dofs].clone()
        joint_pos_targets[:, : self.num_arm_dofs] = self.compute_scripted_arm_targets()
        joint_pos_targets[:, self.num_arm_dofs : self.num_hand_arm_dofs] = (
            self.compute_hand_targets(hand_actions)
        )
        return joint_pos_targets

    def pre_physics_step(self, actions, joint_pos_targets=None):
        hand_actions = actions.to(self.device)
        full_actions = self.expand_hand_actions(hand_actions)
        if joint_pos_targets is None:
            joint_pos_targets = self.compute_joint_pos_targets(hand_actions)
        super().pre_physics_step(full_actions, joint_pos_targets=joint_pos_targets)

    def populate_obs_and_states_buffers(self) -> None:
        super().populate_obs_and_states_buffers()
        if "phase" not in self.obs_list and "phase" not in self.state_list:
            return

        phase_obs = self.compute_phase_obs()
        obs_dict = {"phase": phase_obs}
        if "phase" in self.state_list:
            self.states_buf = self._replace_buffer_from_obs_dict(
                self.states_buf, self.state_list, obs_dict
            )
        if "phase" in self.obs_list:
            self.obs_buf = self._replace_buffer_from_obs_dict(
                self.obs_buf, self.obs_list, obs_dict
            )
            self.obs_queue[:, 0] = self.obs_buf.clone()

    def _extra_reset_rules(self, resets: torch.Tensor) -> torch.Tensor:
        resets = super()._extra_reset_rules(resets)
        zeros = torch.zeros_like(self.reset_buf)

        if not self.reset_if_not_lifted_by_deadline:
            not_lifted_by_deadline = zeros
        else:
            not_lifted_by_deadline = (
                (self.progress_buf >= self.not_lifted_deadline_step)
                & (
                    self.object_pos[:, 2]
                    < self.object_init_state[:, 2] + self.not_lifted_height_threshold
                )
            )
            not_lifted_by_deadline = not_lifted_by_deadline.to(resets.dtype)

        if not self.reset_if_fingertips_far_after_grasp:
            fingertips_far_too_long = zeros
        else:
            grasp_started = self.progress_buf >= int(self.scripted_arm_steps[1].item())
            all_fingertips_far = (
                self.curr_fingertip_distances.min(dim=-1).values
                > self.fingertips_far_distance_threshold
            )
            fingertips_far = grasp_started & all_fingertips_far
            self.fingertips_far_counter = torch.where(
                fingertips_far,
                self.fingertips_far_counter + 1,
                torch.zeros_like(self.fingertips_far_counter),
            )
            fingertips_far_too_long = (
                self.fingertips_far_counter >= self.fingertips_far_patience_steps
            ).to(resets.dtype)

        self.extras["reset/not_lifted_by_deadline"] = not_lifted_by_deadline.float().mean()
        self.extras["reset/fingertips_far_after_grasp"] = (
            fingertips_far_too_long.float().mean()
        )
        return resets | not_lifted_by_deadline | fingertips_far_too_long

    def _replace_buffer_from_obs_dict(
        self, buffer: torch.Tensor, keys: List[str], obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        columns = []
        offset = 0
        for key in keys:
            size = self.obs_type_size_dict[key]
            if key in obs_dict:
                columns.append(obs_dict[key].reshape(self.num_envs, size))
            else:
                columns.append(buffer[:, offset : offset + size])
            offset += size
        return torch.cat(columns, dim=-1)
