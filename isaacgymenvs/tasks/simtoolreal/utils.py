# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def sample_reset_dof_position_delta(
    base_pos: Tensor,
    lower_limits: Tensor,
    upper_limits: Tensor,
    noise_coeff: Tensor,
    sampling: str,
    direction_constraints: Optional[Tensor] = None,
) -> Tensor:
    """Sample joint reset offsets while respecting limits and directions.

    Direction constraints use -1 for negative-only, 0 for symmetric, and +1
    for positive-only sampling. They broadcast over the leading dimensions.
    """
    distance_negative = base_pos - lower_limits
    distance_positive = upper_limits - base_pos

    if sampling == "gaussian_closest_limit":
        distance_to_closest_limit = torch.minimum(
            distance_positive, distance_negative
        )
        delta = (
            noise_coeff
            * distance_to_closest_limit
            * torch.randn_like(base_pos)
        )
    elif sampling != "split_gaussian":
        raise ValueError(
            "resetDofPosSampling must be 'gaussian_closest_limit' or "
            f"'split_gaussian', got {sampling!r}"
        )
    else:
        distance_negative = torch.clamp(distance_negative, min=0.0)
        distance_positive = torch.clamp(distance_positive, min=0.0)
        epsilon = torch.finfo(base_pos.dtype).eps
        can_move_negative = distance_negative > epsilon
        can_move_positive = distance_positive > epsilon

        choose_positive = torch.rand_like(base_pos) < 0.5
        choose_positive = torch.where(
            ~can_move_negative & can_move_positive,
            torch.ones_like(choose_positive),
            choose_positive,
        )
        choose_positive = torch.where(
            can_move_negative & ~can_move_positive,
            torch.zeros_like(choose_positive),
            choose_positive,
        )

        available_distance = torch.where(
            choose_positive, distance_positive, distance_negative
        )
        direction = torch.where(
            choose_positive,
            torch.ones_like(base_pos),
            -torch.ones_like(base_pos),
        )
        magnitude = torch.abs(torch.randn_like(base_pos))
        delta = noise_coeff * available_distance * magnitude * direction
        delta = torch.where(
            can_move_negative | can_move_positive,
            delta,
            torch.zeros_like(delta),
        )

    if direction_constraints is not None:
        if not torch.all(
            (direction_constraints == -1)
            | (direction_constraints == 0)
            | (direction_constraints == 1)
        ):
            raise ValueError("direction_constraints values must be -1, 0, or 1")
        delta = torch.where(
            direction_constraints > 0,
            torch.abs(delta),
            delta,
        )
        delta = torch.where(
            direction_constraints < 0,
            -torch.abs(delta),
            delta,
        )

    return delta


def disable_actor_self_collision_pair(
    gym,
    env,
    actor: int,
    body_name_a: str,
    body_name_b: str,
) -> int:
    """Disable collision only between two rigid bodies of one actor."""
    body_names = tuple(gym.get_actor_rigid_body_names(env, actor))
    missing = [
        name for name in (body_name_a, body_name_b) if name not in body_names
    ]
    if missing:
        raise ValueError(f"Actor is missing rigid bodies required for filtering: {missing}")

    shape_ranges = gym.get_actor_rigid_body_shape_indices(env, actor)
    shape_properties = gym.get_actor_rigid_shape_properties(env, actor)

    used_filter_bits = 0
    for properties in shape_properties:
        used_filter_bits |= int(properties.filter)
    filter_bit = 1
    while used_filter_bits & filter_bit:
        filter_bit <<= 1
    if filter_bit >= (1 << 31):
        raise RuntimeError("No unused rigid-shape collision-filter bit is available")

    for body_name in (body_name_a, body_name_b):
        shape_range = shape_ranges[body_names.index(body_name)]
        for shape_index in range(
            shape_range.start,
            shape_range.start + shape_range.count,
        ):
            shape_properties[shape_index].filter |= filter_bit

    gym.set_actor_rigid_shape_properties(env, actor, shape_properties)
    return filter_bit


DG5F_WRIST_COLLISION_BODY_NAMES = (
    "rl_dg_1_2",
    "rl_dg_4_2",
)


def disable_dg5f_wrist_self_collisions(
    gym,
    env,
    actor: int,
    wrist_body_name: str = "wrist_3_link",
) -> Tuple[int, ...]:
    """Disable known wrist/hand mesh intersections only.

    Each wrist/body pair receives a distinct filter bit. Reusing one bit for
    every hand body would also suppress collisions between the fingers.
    """
    return tuple(
        disable_actor_self_collision_pair(
            gym,
            env,
            actor,
            wrist_body_name,
            hand_body_name,
        )
        for hand_body_name in DG5F_WRIST_COLLISION_BODY_NAMES
    )


def populate_dof_properties(hand_arm_dof_props, arm_dofs: int, hand_dofs: int) -> None:
    assert len(hand_arm_dof_props["stiffness"]) == arm_dofs + hand_dofs

    import numpy as np

    if arm_dofs != 7:
        # Fallback for non-KUKA arms (e.g. UR5): keep asset-provided arm gains,
        # and only enforce the hand tuning profile.
        arm_stiffnesses = hand_arm_dof_props["stiffness"][0:arm_dofs].copy()
        arm_dampings = hand_arm_dof_props["damping"][0:arm_dofs].copy()
        arm_efforts = hand_arm_dof_props["effort"][0:arm_dofs].copy()
        hand_arm_dof_props["stiffness"][0:arm_dofs] = arm_stiffnesses
        hand_arm_dof_props["damping"][0:arm_dofs] = arm_dampings
        hand_arm_dof_props["effort"][0:arm_dofs] = arm_efforts
    else:
        kuka_efforts = [300, 300, 300, 300, 300, 300, 300]
        kuka_stiffnesses = [600, 600, 500, 400, 200, 200, 200]
        kuka_dampings = [
            27.027026473513512,
            27.027026473513512,
            24.672186769721083,
            22.067474708266914,
            9.752538131173853,
            9.147747263670984,
            9.147747263670984,
        ]
        kuka_gear_ratios = [160, 160, 160, 160, 100, 160, 160]
        kuka_rotor_inertias = [
            0.0001321,
            0.0001321,
            0.0001321,
            0.0001321,
            0.0001321,
            0.0000454,
            0.0000454,
        ]

        assert (
            len(kuka_stiffnesses)
            == len(kuka_dampings)
            == len(kuka_gear_ratios)
            == len(kuka_rotor_inertias)
            == arm_dofs
        ), (
            f"{len(kuka_stiffnesses)} != {len(kuka_dampings)} != {len(kuka_gear_ratios)} != {len(kuka_rotor_inertias)} != {arm_dofs}"
        )
        kuka_reflected_inertias = [
            n * n * J for n, J in zip(kuka_gear_ratios, kuka_rotor_inertias)
        ]
        computed_kuka_armatures = kuka_reflected_inertias
        kuka_armatures = [
            3.3817600000000003,
            3.3817600000000003,
            3.3817600000000003,
            3.3817600000000003,
            1.3210000000000002,
            1.16224,
            1.16224,
        ]
        assert np.allclose(computed_kuka_armatures, kuka_armatures), (
            f"computed_kuka_armatures: {computed_kuka_armatures}, kuka_armatures: {kuka_armatures}"
        )

        kuka_damping_ratio = 0.3
        computed_kuka_dampings = [
            2 * kuka_damping_ratio * np.sqrt(kuka_stiffnesses[i] * kuka_armatures[i])
            for i in range(arm_dofs)
        ]
        assert np.allclose(computed_kuka_dampings, kuka_dampings), (
            f"computed_kuka_dampings: {computed_kuka_dampings}, kuka_dampings: {kuka_dampings}"
        )

        hand_arm_dof_props["stiffness"][0:arm_dofs] = kuka_stiffnesses
        hand_arm_dof_props["damping"][0:arm_dofs] = kuka_dampings
        # Not setting armature matches real KUKA robot behavior
        # hand_arm_dof_props["armature"][0:arm_dofs] = kuka_armatures
        hand_arm_dof_props["effort"][0:arm_dofs] = kuka_efforts

    # Assumes Sharpa hand order
    # ['left_thumb_CMC_FE', 'left_thumb_CMC_AA', 'left_thumb_MCP_FE', 'left_thumb_MCP_AA', 'left_thumb_IP',
    #  'left_index_MCP_FE', 'left_index_MCP_AA', 'left_index_PIP', 'left_index_DIP',
    #  'left_middle_MCP_FE', 'left_middle_MCP_AA', 'left_middle_PIP', 'left_middle_DIP',
    #  'left_ring_MCP_FE', 'left_ring_MCP_AA', 'left_ring_PIP', 'left_ring_DIP',
    #  'left_pinky_CMC', 'left_pinky_MCP_FE', 'left_pinky_MCP_AA', 'left_pinky_PIP', 'left_pinky_DIP']
    hand_stiffnesses = [
        6.95,
        13.2,
        4.76,
        6.62,
        0.9,
        4.76,
        6.62,
        0.9,
        0.9,
        4.76,
        6.62,
        0.9,
        0.9,
        4.76,
        6.62,
        0.9,
        0.9,
        1.38,
        4.76,
        6.62,
        0.9,
        0.9,
    ]
    hand_dampings = [
        0.28676845,
        0.40845109,
        0.20394083,
        0.24044435,
        0.04190723,
        0.20859232,
        0.24595532,
        0.04243185,
        0.03504461,
        0.2085923,
        0.24595532,
        0.04243185,
        0.03504461,
        0.20859226,
        0.24595528,
        0.04243183,
        0.0350446,
        0.02782345,
        0.20859229,
        0.24595528,
        0.04243183,
        0.0350446,
    ]
    hand_armatures = [
        0.0032,
        0.0032,
        0.00265,
        0.00265,
        0.0006,
        0.00265,
        0.00265,
        0.0006,
        0.00042,
        0.00265,
        0.00265,
        0.0006,
        0.00042,
        0.00265,
        0.00265,
        0.0006,
        0.00042,
        0.00012,
        0.00265,
        0.00265,
        0.0006,
        0.00042,
    ]
    hand_frictions = [
        0.132,
        0.132,
        0.07456,
        0.07456,
        0.01276,
        0.07456,
        0.07456,
        0.01276,
        0.00378738,
        0.07456,
        0.07456,
        0.01276,
        0.00378738,
        0.07456,
        0.07456,
        0.01276,
        0.00378738,
        0.012,
        0.07456,
        0.07456,
        0.01276,
        0.00378738,
    ]
    if hand_dofs == len(hand_stiffnesses):
        hand_arm_dof_props["stiffness"][arm_dofs:] = hand_stiffnesses
        hand_arm_dof_props["damping"][arm_dofs:] = hand_dampings
        hand_arm_dof_props["armature"][arm_dofs:] = hand_armatures
        hand_arm_dof_props["friction"][arm_dofs:] = hand_frictions
    elif hand_dofs == 20:
        # Tesollo Delto DG5F (20 DoF). Without this branch hand_dofs never
        # matches the 22-DoF Sharpa table above, so these DOFs silently kept
        # Isaac Gym's asset defaults (stiffness ~= float32 max, damping = 0),
        # which is an effectively undamped, torque-saturated position drive.
        #
        # The real hand's ros2_control config (dg5f_right_controller.yaml)
        # gives p=1.5, i=0.0, d=0.0 uniformly for every joint, but that "p"
        # is not a Newton-metre/radian gain: system_interface.cpp feeds it
        # through a closed-source CurrentControl()/ConvertDuty() pipeline
        # (libdelto_gripper_helper.so) that outputs a raw PWM duty cycle, so
        # there is no publicly documented Nm/rad value for these joints.
        #
        # Estimated instead from URDF mass/inertia data (assets/urdf/
        # ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf), same
        # method as the KUKA arm gains above: pick a stiffness and damping
        # ratio, derive damping from each joint's own reflected inertia
        # (link + downstream chain inertia about that joint's axis, rigid
        # chain at the URDF zero pose) via critical-damping.
        #   - stiffness: uniform 42.97 Nm/rad for 19 of the 20 joints, sized
        #     so the joint's 7.5 Nm URDF effort limit saturates at a 10 deg
        #     position error. That 10 deg target is a judgment call, not a
        #     sourced number - tighten it if fingertips still sag too much
        #     under grasp contact, loosen it if the lightest joints (joint
        #     4, the smallest reflected inertia) start ringing again.
        #   - damping: critically damped (zeta=1) per joint, i.e.
        #     2*sqrt(stiffness * I_joint), using each joint's own reflected
        #     inertia so lighter and heavier joints are equally well damped
        #     despite sharing one stiffness value.
        #   - rj_dg_1_1 uses a slightly lower empirical damping (0.1). Its
        #     former large tracking error exposed non-physical collisions
        #     between the wrist and DG5F meshes. The two observed intersecting
        #     pairs are filtered when the actor is created in env.py;
        #     finger/finger collisions remain enabled.
        #   - rj_dg_1_2 retains the empirically tuned 400 Nm/rad stiffness.
        # Order matches HAND_JOINT_NAMES: rj_dg_{finger}_{joint} for
        # finger in 1..5, joint in 1..4.
        hand_arm_dof_props["stiffness"][arm_dofs:] = [
            42.9718, 400.0, 42.9718, 42.9718,
            42.9718, 42.9718, 42.9718, 42.9718,
            42.9718, 42.9718, 42.9718, 42.9718,
            42.9718, 42.9718, 42.9718, 42.9718,
            42.9718, 42.9718, 42.9718, 42.9718,
        ]
        hand_arm_dof_props["damping"][arm_dofs:] = [
            0.1, 0.9475, 0.3012, 0.1821,
            0.7523, 0.4126, 0.2856, 0.1365,
            0.7587, 0.4126, 0.2856, 0.1365,
            0.7274, 0.4126, 0.2856, 0.1365,
            0.2662, 0.4796, 0.3012, 0.1821,
        ]


def tolerance_curriculum(
    last_curriculum_update: int,
    frames_since_restart: int,
    curriculum_interval: int,
    prev_episode_successes: Tensor,
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    tolerance_curriculum_increment: float,
) -> Tuple[float, int]:
    """
    Returns: new tolerance, new last_curriculum_update
    """
    if frames_since_restart - last_curriculum_update < curriculum_interval:
        return success_tolerance, last_curriculum_update

    mean_successes_per_episode = prev_episode_successes.mean()
    if mean_successes_per_episode < 3.0:
        # this policy is not good enough with the previous tolerance value, keep training for now...
        return success_tolerance, last_curriculum_update

    # decrease the tolerance now
    success_tolerance *= tolerance_curriculum_increment
    success_tolerance = min(success_tolerance, initial_tolerance)
    success_tolerance = max(success_tolerance, target_tolerance)

    print(
        f"Prev episode successes: {mean_successes_per_episode}, success tolerance: {success_tolerance}"
    )

    last_curriculum_update = frames_since_restart
    return success_tolerance, last_curriculum_update


def interp_0_1(x_curr: float, x_initial: float, x_target: float) -> float:
    """
    Outputs 1 when x_curr == x_target (curriculum completed)
    Outputs 0 when x_curr == x_initial (just started training)
    Interpolates value in between.
    """
    span = x_initial - x_target
    return (x_initial - x_curr) / span


def tolerance_successes_objective(
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    successes: Tensor,
) -> Tensor:
    """
    Objective for the PBT. This basically prioritizes tolerance over everything else when we
    execute the curriculum, after that it's just #successes.
    """
    # this grows from 0 to 1 as we reach the target tolerance
    if initial_tolerance > target_tolerance:
        # makeshift unit tests:
        eps = 1e-5
        assert (
            abs(interp_0_1(initial_tolerance, initial_tolerance, target_tolerance))
            < eps
        )
        assert (
            abs(interp_0_1(target_tolerance, initial_tolerance, target_tolerance) - 1.0)
            < eps
        )
        mid_tolerance = (initial_tolerance + target_tolerance) / 2
        assert (
            abs(interp_0_1(mid_tolerance, initial_tolerance, target_tolerance) - 0.5)
            < eps
        )

        tolerance_objective = interp_0_1(
            success_tolerance, initial_tolerance, target_tolerance
        )
    else:
        tolerance_objective = 1.0

    if success_tolerance > target_tolerance:
        # add succeses with a small coefficient to differentiate between policies at the beginning of training
        # increment in tolerance improvement should always give higher value than higher successes with the
        # previous tolerance, that's why this coefficient is very small
        true_objective = (successes * 0.01) + tolerance_objective
    else:
        # basically just the successes + tolerance objective so that true_objective never decreases when we cross
        # the threshold
        true_objective = successes + tolerance_objective

    return true_objective
