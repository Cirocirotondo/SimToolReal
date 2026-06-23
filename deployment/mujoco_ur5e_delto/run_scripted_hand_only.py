from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import yaml

from deployment.mujoco_ur5e_delto.mujoco_sim import (
    Ur5eDeltoMujocoConfig,
    Ur5eDeltoMujocoSim,
)
from deployment.mujoco_ur5e_delto.policy_adapter import (
    ARM_DOF,
    DEFAULT_JOINT_POS,
    LOWER_LIMITS,
    N_ACT,
    UPPER_LIMITS,
)


@dataclass
class Args:
    config_path: Path = Path("isaacgymenvs/cfg/task/SimToolRealTrainC1.yaml")
    """Path to the C1 task YAML with scriptedArmTrajectory."""

    object_name: str = "cube"
    """Object primitive to spawn: cube or hammer."""

    enable_viewer: bool = True
    """Open the passive MuJoCo viewer."""

    max_steps: int = 0
    """Maximum control steps to run. Use 0 to run forever."""

    control_hz: float = 60.0
    """Scripted controller frequency."""

    realtime: bool = True
    """Sleep to keep the loop close to real time."""

    wait_for_enter: bool = True
    """Show the initial scene and wait for Enter before starting motion."""

    arm_kp_scale: float = 6.0
    """Scale MuJoCo arm position-controller stiffness for trajectory debugging."""

    arm_kv_scale: float = 4.0
    """Scale MuJoCo arm position-controller damping for trajectory debugging."""

    gravity_scale: float = 1.0
    """Scale MuJoCo gravity. Use values <1 only as a visualization/debug aid."""


def read_env_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "task" in cfg and "env" in cfg["task"]:
        return cfg["task"]["env"]
    return cfg["env"]


def object_scales_for(name: str) -> np.ndarray:
    if name == "hammer":
        return np.array([0.141, 0.03025, 0.0271], dtype=np.float32)
    if name == "cube":
        return np.array([1.25, 1.25, 1.25], dtype=np.float32)
    raise ValueError(f"Unsupported object_name={name!r}; expected 'cube' or 'hammer'")


def object_start_pos_from_cfg(env_cfg: dict) -> np.ndarray | None:
    object_start_pose = env_cfg.get("objectStartPose")
    if object_start_pose is None:
        return None
    if len(object_start_pose) < 3:
        raise ValueError("objectStartPose must contain at least xyz")
    return np.array(object_start_pose[:3], dtype=np.float32)


def goal_object_start_pos_from_cfg(env_cfg: dict) -> np.ndarray | None:
    goal_object_pose = env_cfg.get("goalObjectPose")
    if goal_object_pose is None:
        object_start = object_start_pos_from_cfg(env_cfg)
        if object_start is None:
            return None
        goal_start = object_start.copy()
        goal_start[2] += 0.20
        return goal_start
    if len(goal_object_pose) < 3:
        raise ValueError("goalObjectPose must contain at least xyz")
    return np.array(goal_object_pose[:3], dtype=np.float32)


def initial_joint_pos_from_cfg(env_cfg: dict) -> np.ndarray:
    q = DEFAULT_JOINT_POS.copy()
    default_arm = np.array(env_cfg.get("defaultArmDofPos", q[:ARM_DOF]), dtype=np.float32)
    if default_arm.shape != (ARM_DOF,):
        raise ValueError(f"defaultArmDofPos has shape {default_arm.shape}; expected {(ARM_DOF,)}")
    q[:ARM_DOF] = default_arm
    return q


def scripted_waypoints_from_cfg(env_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    trajectory_cfg = env_cfg.get("scriptedArmTrajectory", {})
    waypoints = trajectory_cfg.get("waypoints", [])
    if len(waypoints) < 2:
        raise ValueError("scriptedArmTrajectory requires at least 2 waypoints")

    steps = []
    poses = []
    for waypoint in waypoints:
        step = int(waypoint["step"])
        pose = np.array(waypoint["q"], dtype=np.float32)
        if pose.shape != (ARM_DOF,):
            raise ValueError(f"Waypoint q has shape {pose.shape}; expected {(ARM_DOF,)}")
        steps.append(step)
        poses.append(pose)

    order = np.argsort(np.array(steps))
    sorted_steps = np.array([steps[i] for i in order], dtype=np.float32)
    sorted_poses = np.array([poses[i] for i in order], dtype=np.float32)
    return sorted_steps, sorted_poses


def scripted_arm_target(step: int, steps: np.ndarray, poses: np.ndarray) -> np.ndarray:
    if step <= steps[0]:
        return poses[0].copy()
    if step >= steps[-1]:
        return poses[-1].copy()

    segment_idx = np.searchsorted(steps, step, side="right") - 1
    step_a = steps[segment_idx]
    step_b = steps[segment_idx + 1]
    pose_a = poses[segment_idx]
    pose_b = poses[segment_idx + 1]
    alpha = np.clip((step - step_a) / (step_b - step_a), 0.0, 1.0)
    return pose_a + alpha * (pose_b - pose_a)


def scene_config_from_env_cfg(env_cfg: dict) -> dict:
    # Keep the low-table top at z=0.0 when tableResetZ=-0.125.
    # The generic MuJoCo default table thickness is 0.30, whose top would be
    # z=0.025 here and can visually intersect the cube.
    table_size_z = float(env_cfg.get("tableSizeZ", 0.251))
    return {
        "table_center_z": float(env_cfg.get("tableResetZ", -0.125)),
        "table_object_z_offset": float(env_cfg.get("tableObjectZOffset", 0.25)),
        "table_size_x": 0.8,
        "table_size_y": 0.8,
        "table_size_z": table_size_z,
        "initial_joint_pos": initial_joint_pos_from_cfg(env_cfg),
        "object_start_pos": object_start_pos_from_cfg(env_cfg),
        "goal_object_start_pos": goal_object_start_pos_from_cfg(env_cfg),
    }


def apply_debug_dynamics_overrides(sim: Ur5eDeltoMujocoSim, args: Args) -> None:
    if args.arm_kp_scale <= 0.0:
        raise ValueError("arm_kp_scale must be positive")
    if args.arm_kv_scale <= 0.0:
        raise ValueError("arm_kv_scale must be positive")
    if args.gravity_scale < 0.0:
        raise ValueError("gravity_scale must be non-negative")

    arm_actuator_ids = sim._actuator_ids[:ARM_DOF]
    sim.model.actuator_gainprm[arm_actuator_ids, 0] *= args.arm_kp_scale
    sim.model.actuator_biasprm[arm_actuator_ids, 1] *= args.arm_kp_scale
    sim.model.actuator_biasprm[arm_actuator_ids, 2] *= args.arm_kv_scale
    sim.model.opt.gravity[:] *= args.gravity_scale


def main() -> None:
    args = tyro.cli(Args)
    env_cfg = read_env_cfg(args.config_path)
    steps, arm_poses = scripted_waypoints_from_cfg(env_cfg)
    control_dt = 1.0 / args.control_hz

    initial_joint_pos = initial_joint_pos_from_cfg(env_cfg)
    initial_joint_pos[:ARM_DOF] = arm_poses[0]
    initial_joint_pos = np.clip(initial_joint_pos, LOWER_LIMITS, UPPER_LIMITS)

    scene_cfg = scene_config_from_env_cfg(env_cfg)
    scene_cfg["initial_joint_pos"] = initial_joint_pos

    sim = Ur5eDeltoMujocoSim(
        Ur5eDeltoMujocoConfig(
            enable_viewer=args.enable_viewer,
            object_name=args.object_name,
            object_scales=object_scales_for(args.object_name),
            **scene_cfg,
        )
    )
    apply_debug_dynamics_overrides(sim, args)

    targets = initial_joint_pos.copy()
    print(
        "Running scripted hand-only MuJoCo trajectory: "
        f"config={args.config_path}, object={args.object_name}, "
        f"steps={steps.astype(int).tolist()}"
    )
    print("Close the MuJoCo viewer or press Ctrl+C to stop.")

    if args.wait_for_enter:
        sim.step_for(control_dt)
        input("Press Enter to start the scripted arm trajectory...")

    step = 0
    try:
        while args.max_steps <= 0 or step < args.max_steps:
            start = time.time()
            targets[:ARM_DOF] = scripted_arm_target(step, steps, arm_poses)
            targets[:ARM_DOF] = np.clip(
                targets[:ARM_DOF],
                LOWER_LIMITS[:ARM_DOF],
                UPPER_LIMITS[:ARM_DOF],
            )
            if targets.shape != (N_ACT,):
                raise RuntimeError(f"targets.shape={targets.shape}, expected {(N_ACT,)}")

            sim.set_robot_joint_pos_targets(targets)
            sim.step_for(control_dt)

            if step % int(max(1, args.control_hz)) == 0:
                sim_state = sim.get_sim_state()
                palm_z = sim_state["palm_pos"][2]
                object_z = sim_state["object_pos"][2]
                print(
                    f"step={step:04d} "
                    f"arm_q={np.round(targets[:ARM_DOF], 4).tolist()} "
                    f"palm_z={palm_z:.3f} object_z={object_z:.3f}"
                )

            if args.realtime:
                sleep_dt = control_dt - (time.time() - start)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
