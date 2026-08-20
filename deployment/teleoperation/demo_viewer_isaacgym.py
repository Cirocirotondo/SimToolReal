#!/usr/bin/env python3
"""Replay a processed teleoperation demonstration in Isaac Gym."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple


def _configure_isaac_gym_graphics_environment() -> None:
    """Avoid ROS/Gazebo graphics libraries shadowing the NVIDIA driver.

    Isaac Gym Preview 4 loads its graphics backend when ``gymapi`` is imported.
    On this workstation a ROS 2 shell adds Gazebo/OGRE libraries to
    ``LD_LIBRARY_PATH``; those make Isaac Gym's viewer hang in
    ``step_graphics``.  Keep unrelated custom library paths, but remove those
    known incompatible ROS/Gazebo entries before importing Isaac Gym.

    Set ``ISAACGYM_PRESERVE_ROS_GRAPHICS_PATH=1`` to opt out deliberately.
    """
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

    # Preview 4's graphics stack may otherwise select Mesa's software Vulkan
    # implementation when several ICDs are installed.
    nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    if os.path.isfile(nvidia_icd):
        os.environ.setdefault("VK_ICD_FILENAMES", nvidia_icd)
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


_configure_isaac_gym_graphics_environment()

# Isaac Gym must be imported before PyTorch when both are used in a process.
from isaacgym import gymapi


def _configure_qt_environment() -> None:
    """Point matplotlib at the venv's Qt plugins for the diagnostics plot.

    This venv ships two Qt5 copies (PyQt5 and OpenCV's); the wrong one can
    make Qt fail to load the "xcb" platform plugin at ``plt.show()``.
    """
    os.environ.pop("QT_PLUGIN_PATH", None)
    pyqt_platform_plugins = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "PyQt5"
        / "Qt5"
        / "plugins"
        / "platforms"
    )
    if pyqt_platform_plugins.is_dir():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(pyqt_platform_plugins)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-simtoolreal")


_configure_qt_environment()

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DEMO_PATH = (
    HERE
    / "demonstrations_good"
    / "demo_20260727_152551_335339_60hz.npz"
)
TARGET_HZ = 60.0
HAND_LOWER_LIMIT_TOLERANCE_RAD = 0.02
ENV_SPACING = 1.2
GROUND_PLANE_Z = 0.0
ROBOT_BASE_Y = 0.6
ROBOT_BASE_Z = 0.55
ROBOT_ASSET_ROOT = REPO_ROOT / "assets"
ROBOT_ASSET_FILE = (
    "urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
)

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HAND_JOINT_NAMES = tuple(
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
TARGET_JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES

# Tesollo Delto DG5F position gains, in HAND_JOINT_NAMES order. Keep these in
# sync with the `hand_dofs == 20` branch of populate_dof_properties in
# isaacgymenvs/tasks/simtoolreal/utils.py, which documents the derivation:
# the real driver's p=1.5 gain is not Nm/rad (it feeds a closed-source
# current-control/PWM pipeline), so these are instead estimated from URDF
# mass/inertia data with a uniform stiffness (sized for 10 deg of sag at the
# 7.5 Nm effort limit - a judgment call, tune if needed) and per-joint
# critical damping, except the first two thumb joints, whose gains were tuned
# after removing spurious wrist/hand self-collisions (see create_scene()).
HAND_PD_STIFFNESS = (
    42.9718, 400.0, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
)
HAND_PD_DAMPING = (
    0.1, 0.9475, 0.3012, 0.1821,
    0.7523, 0.4126, 0.2856, 0.1365,
    0.7587, 0.4126, 0.2856, 0.1365,
    0.7274, 0.4126, 0.2856, 0.1365,
    0.2662, 0.4796, 0.3012, 0.1821,
)

WRIST_BODY_NAME = "wrist_3_link"
WRIST_COLLISION_HAND_BODY_NAMES = (
    "rl_dg_1_2",
    "rl_dg_4_2",
)


@dataclass(frozen=True)
class DemoTrajectory:
    """Validated timestamped targets in UR5e + right DG5F joint order."""

    path: Path
    times_s: np.ndarray
    arm_q: np.ndarray
    hand_q: np.ndarray
    joint_targets: np.ndarray
    joint_names: tuple[str, ...] = TARGET_JOINT_NAMES

    @property
    def sample_count(self) -> int:
        return len(self.times_s)

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])

    @property
    def nominal_hz(self) -> float:
        """Robust estimate of the source sampling frequency."""
        return float(1.0 / np.median(np.diff(self.times_s)))


@dataclass(frozen=True)
class RobotAsset:
    """Loaded combined robot asset and its demonstration-to-asset DOF map."""

    handle: object
    dof_names: tuple[str, ...]
    demo_to_asset_indices: np.ndarray
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    effort_limits: np.ndarray


@dataclass(frozen=True)
class PdControllerProperties:
    """Isaac Gym DOF properties configured exactly as in training."""

    dof_properties: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limits: np.ndarray


@dataclass
class IsaacGymScene:
    """Single-environment scene used to replay one demonstration."""

    gym: object
    sim: object
    env: object
    robot_actor: int
    robot: RobotAsset
    viewer: Optional[object]
    use_gpu_pipeline: bool
    sim_prepared: bool = False


@dataclass(frozen=True)
class ReplayDiagnostics:
    """Per-frame PD target vs. measured joint positions from one replay run."""

    times_s: np.ndarray
    target_positions: np.ndarray
    actual_positions: np.ndarray


@dataclass
class CollisionStats:
    """Aggregated active PhysX contacts for one rigid-body pair."""

    active_frames: int = 0
    contact_records: int = 0
    first_time_s: float = float("inf")
    last_time_s: float = float("-inf")
    max_impulse: float = 0.0


def load_demo(path: Path, *, hand_source: str = "measured") -> DemoTrajectory:
    """Load finite timestamped joint targets for the Isaac Gym replay loop."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Demonstration not found: {path}")

    hand_key = f"hand_q_{hand_source}"
    valid_key = f"{hand_key}_valid"

    with np.load(path, allow_pickle=False) as arrays:
        required_keys = {"timestamp", "arm_q", hand_key}
        missing_keys = sorted(required_keys.difference(arrays.files))
        if missing_keys:
            raise ValueError(f"{path.name} is missing fields: {missing_keys}")

        arm_q = np.asarray(arrays["arm_q"], dtype=np.float64).copy()
        hand_q = np.asarray(arrays[hand_key], dtype=np.float64).copy()
        timestamp_key = (
            "monotonic_timestamp"
            if "monotonic_timestamp" in arrays
            else "timestamp"
        )
        timestamps = np.asarray(arrays[timestamp_key], dtype=np.float64).copy()

        if valid_key in arrays:
            hand_valid = np.asarray(arrays[valid_key], dtype=bool)
            if hand_valid.shape != (len(timestamps),):
                raise ValueError(
                    f"Expected {valid_key} shape ({len(timestamps)},), "
                    f"got {hand_valid.shape}"
                )
            invalid_count = int(np.count_nonzero(~hand_valid))
            if invalid_count:
                raise ValueError(
                    f"{path.name} contains {invalid_count} invalid {hand_key} samples"
                )

    sample_count = len(timestamps)
    expected_shapes = {
        "arm_q": (sample_count, len(ARM_JOINT_NAMES)),
        hand_key: (sample_count, len(HAND_JOINT_NAMES)),
    }
    for name, values in (("arm_q", arm_q), (hand_key, hand_q)):
        if values.shape != expected_shapes[name]:
            raise ValueError(
                f"Expected {name} shape {expected_shapes[name]}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path.name} contains non-finite {name} values")

    if sample_count < 2 or timestamps.shape != (sample_count,):
        raise ValueError("A demonstration must contain at least two timestamps")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("Demonstration timestamps must be finite and increasing")

    times_s = timestamps - timestamps[0]
    return DemoTrajectory(
        path=path,
        times_s=times_s,
        arm_q=arm_q,
        hand_q=hand_q,
        joint_targets=np.concatenate((arm_q, hand_q), axis=1),
    )


def create_sim(
    *,
    compute_device_id: int = 0,
    graphics_device_id: int = 0,
    use_gpu_physx: bool = True,
    use_gpu_pipeline: bool = False,
    collect_contacts: bool = False,
) -> tuple[object, object]:
    """Create a PhysX simulation with the training-time physics settings."""
    print(
        "Creating sim "
        f"(compute_device_id={compute_device_id}, "
        f"graphics_device_id={graphics_device_id}, "
        f"use_gpu_physx={use_gpu_physx}, "
        f"use_gpu_pipeline={use_gpu_pipeline})",
        flush=True,
    )
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()

    sim_params.dt = 1.0 / TARGET_HZ
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.num_client_threads = 8
    sim_params.use_gpu_pipeline = use_gpu_pipeline

    sim_params.physx.use_gpu = use_gpu_physx
    sim_params.physx.num_threads = 6
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.physx.num_subscenes = 4
    sim_params.physx.contact_offset = 0.002
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.2
    sim_params.physx.max_depenetration_velocity = 2.0
    sim_params.physx.default_buffer_size_multiplier = 25.0
    sim_params.physx.contact_collection = gymapi.ContactCollection(
        gymapi.CC_ALL_SUBSTEPS if collect_contacts else gymapi.CC_NEVER
    )

    sim = gym.create_sim(
        compute_device_id,
        graphics_device_id,
        gymapi.SIM_PHYSX,
        sim_params,
    )
    if sim is None:
        raise RuntimeError(
            "Isaac Gym failed to create the PhysX simulation. Check the CUDA "
            "device and graphics-device settings."
        )
    print("Sim created", flush=True)
    return gym, sim


def load_robot_asset(gym, sim) -> RobotAsset:
    """Load the training UR5e+DG5F asset and validate its 26 DOF names."""
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True
    asset_options.thickness = 0.001
    asset_options.angular_damping = 0.01
    asset_options.linear_damping = 0.01
    asset_options.use_physx_armature = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

    handle = gym.load_asset(
        sim,
        str(ROBOT_ASSET_ROOT),
        ROBOT_ASSET_FILE,
        asset_options,
    )
    if handle is None:
        raise RuntimeError(
            f"Isaac Gym failed to load {ROBOT_ASSET_ROOT / ROBOT_ASSET_FILE}"
        )

    dof_names = tuple(gym.get_asset_dof_names(handle))
    expected_names = set(TARGET_JOINT_NAMES)
    actual_names = set(dof_names)
    missing_names = sorted(expected_names - actual_names)
    extra_names = sorted(actual_names - expected_names)
    if missing_names or extra_names or len(dof_names) != len(TARGET_JOINT_NAMES):
        raise ValueError(
            "Robot asset DOFs do not match the demonstration. "
            f"Missing: {missing_names}; extra: {extra_names}; "
            f"expected {len(TARGET_JOINT_NAMES)}, got {len(dof_names)}"
        )

    demo_to_asset_indices = np.asarray(
        [dof_names.index(name) for name in TARGET_JOINT_NAMES],
        dtype=np.int32,
    )
    dof_properties = gym.get_asset_dof_properties(handle)
    lower_limits = np.asarray(dof_properties["lower"], dtype=np.float64).copy()
    upper_limits = np.asarray(dof_properties["upper"], dtype=np.float64).copy()
    effort_limits = np.asarray(dof_properties["effort"], dtype=np.float64).copy()

    if (
        lower_limits.shape != (len(dof_names),)
        or upper_limits.shape != (len(dof_names),)
        or effort_limits.shape != (len(dof_names),)
        or not np.all(np.isfinite(lower_limits))
        or not np.all(np.isfinite(upper_limits))
        or not np.all(np.isfinite(effort_limits))
        or np.any(lower_limits > upper_limits)
        or np.any(effort_limits <= 0.0)
    ):
        raise ValueError("Robot asset contains invalid DOF limits")

    return RobotAsset(
        handle=handle,
        dof_names=dof_names,
        demo_to_asset_indices=demo_to_asset_indices,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        effort_limits=effort_limits,
    )


def configure_pd_controller(gym, robot_asset: RobotAsset) -> PdControllerProperties:
    """Build the same position-drive DOF properties used during training."""
    dof_properties = gym.get_asset_dof_properties(robot_asset.handle)
    expected_count = len(TARGET_JOINT_NAMES)
    if len(dof_properties) != expected_count:
        raise ValueError(
            f"Expected {expected_count} robot DOFs, got {len(dof_properties)}"
        )

    # SimToolReal.populate_dof_properties keeps the asset-provided gains for
    # the six-DOF arm. Importing that training helper here pulls in the whole
    # task registry and gymtorch, which is inappropriate for this standalone
    # viewer, so the arm gains are left at the asset defaults (matching the
    # arm_dofs != 7 branch there) and only the DG5F hand gains below are
    # reproduced explicitly. Keep these in sync with the `hand_dofs == 20`
    # branch of populate_dof_properties in
    # isaacgymenvs/tasks/simtoolreal/utils.py.
    dof_properties["driveMode"].fill(int(gymapi.DOF_MODE_POS))
    hand_asset_indices = robot_asset.demo_to_asset_indices[len(ARM_JOINT_NAMES):]
    dof_properties["stiffness"][hand_asset_indices] = HAND_PD_STIFFNESS
    dof_properties["damping"][hand_asset_indices] = HAND_PD_DAMPING

    stiffness = np.asarray(dof_properties["stiffness"], dtype=np.float64).copy()
    damping = np.asarray(dof_properties["damping"], dtype=np.float64).copy()
    effort_limits = np.asarray(dof_properties["effort"], dtype=np.float64).copy()
    lower_limits = np.asarray(dof_properties["lower"], dtype=np.float64)
    upper_limits = np.asarray(dof_properties["upper"], dtype=np.float64)

    if (
        stiffness.shape != (expected_count,)
        or damping.shape != (expected_count,)
        or effort_limits.shape != (expected_count,)
        or not np.all(np.isfinite(stiffness))
        or not np.all(np.isfinite(damping))
        or not np.all(np.isfinite(effort_limits))
        or np.any(stiffness < 0.0)
        or np.any(damping < 0.0)
        or np.any(effort_limits <= 0.0)
        or not np.array_equal(
            dof_properties["driveMode"],
            np.full(expected_count, int(gymapi.DOF_MODE_POS)),
        )
        or not np.allclose(lower_limits, robot_asset.lower_limits)
        or not np.allclose(upper_limits, robot_asset.upper_limits)
        or not np.allclose(effort_limits, robot_asset.effort_limits)
    ):
        raise ValueError("Training PD controller properties are invalid")

    return PdControllerProperties(
        dof_properties=dof_properties,
        stiffness=stiffness,
        damping=damping,
        effort_limits=effort_limits,
    )


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


def disable_wrist_hand_self_collisions(gym, env, actor: int) -> tuple[int, ...]:
    """Filter known wrist/hand intersections but preserve finger collisions."""
    return tuple(
        disable_actor_self_collision_pair(
            gym,
            env,
            actor,
            WRIST_BODY_NAME,
            hand_body_name,
        )
        for hand_body_name in WRIST_COLLISION_HAND_BODY_NAMES
    )


def create_scene(
    gym,
    sim,
    robot_asset: RobotAsset,
    pd_controller: PdControllerProperties,
    *,
    enable_viewer: bool = True,
    use_gpu_pipeline: bool = False,
) -> IsaacGymScene:
    """Create one Isaac Gym environment with the robot actor and optional viewer."""
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.distance = GROUND_PLANE_Z
    gym.add_ground(sim, plane_params)

    lower = gymapi.Vec3(-ENV_SPACING, -ENV_SPACING, 0.0)
    upper = gymapi.Vec3(ENV_SPACING, ENV_SPACING, ENV_SPACING)
    env = gym.create_env(sim, lower, upper, 1)
    if env is None:
        raise RuntimeError("Isaac Gym failed to create the replay environment")

    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(0.0, ROBOT_BASE_Y, ROBOT_BASE_Z)
    robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    robot_actor = gym.create_actor(
        env,
        robot_asset.handle,
        robot_pose,
        "robot",
        0,
        -1,
        0,
    )
    if robot_actor < 0:
        raise RuntimeError("Isaac Gym failed to create the robot actor")

    # Filter the two observed non-physical wrist/hand mesh intersections. Use
    # one bit per pair so all finger/finger collisions stay active.
    disable_wrist_hand_self_collisions(gym, env, robot_actor)

    gym.set_actor_dof_properties(
        env,
        robot_actor,
        pd_controller.dof_properties,
    )

    viewer = None
    if enable_viewer:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError(
                "Isaac Gym failed to create a viewer. If you are running headless, "
                "call create_scene(..., enable_viewer=False)."
            )
        camera_position = gymapi.Vec3(-1, -1, 1.1)
        camera_target = gymapi.Vec3(0.0, ROBOT_BASE_Y, ROBOT_BASE_Z + 0.35)
        gym.viewer_camera_look_at(viewer, env, camera_position, camera_target)

    return IsaacGymScene(
        gym=gym,
        sim=sim,
        env=env,
        robot_actor=robot_actor,
        robot=robot_asset,
        viewer=viewer,
        use_gpu_pipeline=use_gpu_pipeline,
    )


def prepare_scene(scene: IsaacGymScene) -> None:
    """Prepare the simulation once all actor-level initialization is complete."""
    if scene.sim_prepared:
        return
    scene.gym.prepare_sim(scene.sim)
    scene.sim_prepared = True


def demo_targets_to_asset_order(
    demo: DemoTrajectory,
    robot_asset: RobotAsset,
    sample_idx: int,
) -> np.ndarray:
    """Return one demonstration target vector reordered to the asset DOF order."""
    if sample_idx < 0 or sample_idx >= demo.sample_count:
        raise IndexError(
            f"sample_idx must be in [0, {demo.sample_count - 1}], got {sample_idx}"
        )

    return targets_to_asset_order(
        demo.joint_targets[sample_idx],
        robot_asset,
        description=f"Demo sample {sample_idx}",
    )


def targets_to_asset_order(
    target_in_demo_order: np.ndarray,
    robot_asset: RobotAsset,
    *,
    description: str,
) -> np.ndarray:
    """Reorder one UR5e + DG5F target vector and validate its joint limits."""
    target_in_demo_order = np.asarray(target_in_demo_order, dtype=np.float64)
    if target_in_demo_order.shape != (len(TARGET_JOINT_NAMES),):
        raise ValueError(
            f"{description} has target shape {target_in_demo_order.shape}; expected "
            f"({len(TARGET_JOINT_NAMES)},)"
        )

    target_in_asset_order = np.empty(len(robot_asset.dof_names), dtype=np.float32)
    target_in_asset_order[robot_asset.demo_to_asset_indices] = target_in_demo_order

    # Raw demonstrations can contain tiny negative encoder offsets for hand
    # joints whose URDF lower limit is zero.  Match postprocessing behavior by
    # treating offsets up to 0.02 rad below that limit as the closed position.
    hand_asset_indices = robot_asset.demo_to_asset_indices[len(ARM_JOINT_NAMES) :]
    hand_targets = target_in_asset_order[hand_asset_indices]
    hand_lower_limits = robot_asset.lower_limits[hand_asset_indices]
    small_hand_limit_violations = (
        (hand_targets < hand_lower_limits)
        & (hand_targets >= -HAND_LOWER_LIMIT_TOLERANCE_RAD)
    )
    target_in_asset_order[hand_asset_indices[small_hand_limit_violations]] = 0.0

    if (
        np.any(target_in_asset_order < robot_asset.lower_limits - 1e-6)
        or np.any(target_in_asset_order > robot_asset.upper_limits + 1e-6)
    ):
        offending_indices = np.flatnonzero(
            (target_in_asset_order < robot_asset.lower_limits - 1e-6)
            | (target_in_asset_order > robot_asset.upper_limits + 1e-6)
        )
        offending_names = [robot_asset.dof_names[i] for i in offending_indices[:8]]
        raise ValueError(
            f"{description} contains joint targets outside the asset "
            f"limits. First offending joints: {offending_names}"
        )

    return target_in_asset_order


def interpolated_targets_to_asset_order(
    demo: DemoTrajectory,
    robot_asset: RobotAsset,
    playback_time_s: float,
) -> np.ndarray:
    """Linearly interpolate source targets at a 60 Hz simulation instant."""
    if playback_time_s < demo.times_s[0] or playback_time_s > demo.times_s[-1]:
        raise ValueError(
            f"playback_time_s={playback_time_s:.6f} is outside the demo interval "
            f"[{demo.times_s[0]:.6f}, {demo.times_s[-1]:.6f}]"
        )
    interpolated_targets = np.asarray(
        [
            np.interp(playback_time_s, demo.times_s, demo.joint_targets[:, joint_idx])
            for joint_idx in range(len(TARGET_JOINT_NAMES))
        ],
        dtype=np.float64,
    )
    return targets_to_asset_order(
        interpolated_targets,
        robot_asset,
        description=f"Interpolated demo target at t={playback_time_s:.3f} s",
    )


def initialize_robot_from_demo(
    scene: IsaacGymScene,
    demo: DemoTrajectory,
    *,
    sample_idx: int = 0,
) -> np.ndarray:
    """Set the robot DOF state and PD target to a demonstration sample."""
    if scene.sim_prepared and scene.use_gpu_pipeline:
        raise RuntimeError(
            "Cannot initialize with actor-level DOF state APIs after prepare_sim "
            "when the GPU pipeline is enabled. Initialize before replay_demo()."
        )

    target_positions = demo_targets_to_asset_order(
        demo,
        scene.robot,
        sample_idx,
    )

    dof_states = scene.gym.get_actor_dof_states(
        scene.env,
        scene.robot_actor,
        gymapi.STATE_ALL,
    )
    if dof_states.shape != (len(scene.robot.dof_names),):
        raise RuntimeError(
            f"Expected {len(scene.robot.dof_names)} actor DOF states, "
            f"got {dof_states.shape}"
        )

    dof_states["pos"] = target_positions
    dof_states["vel"].fill(0.0)

    scene.gym.set_actor_dof_states(
        scene.env,
        scene.robot_actor,
        dof_states,
        gymapi.STATE_ALL,
    )
    scene.gym.set_actor_dof_position_targets(
        scene.env,
        scene.robot_actor,
        target_positions,
    )

    return target_positions


def _rigid_body_name(body_index: int, body_names: tuple[str, ...]) -> str:
    if body_index < 0:
        return "ground"
    if body_index < len(body_names):
        return body_names[body_index]
    return f"rigid_body_{body_index}"


def _update_collision_report(
    scene: IsaacGymScene,
    playback_time_s: float,
    body_names: tuple[str, ...],
    collision_stats: Dict[Tuple[str, str], CollisionStats],
    *,
    minimum_impulse: float = 1.0e-8,
) -> None:
    """Accumulate contacts that produced a nonzero solver impulse this frame."""
    active_pairs_this_frame = set()
    for contact in scene.gym.get_env_rigid_contacts(scene.env):
        impulse = abs(float(contact["lambda"]))
        if not np.isfinite(impulse) or impulse <= minimum_impulse:
            continue

        body_a = _rigid_body_name(int(contact["body0"]), body_names)
        body_b = _rigid_body_name(int(contact["body1"]), body_names)
        pair = tuple(sorted((body_a, body_b)))
        stats = collision_stats.setdefault(pair, CollisionStats())
        stats.contact_records += 1
        stats.first_time_s = min(stats.first_time_s, playback_time_s)
        stats.last_time_s = max(stats.last_time_s, playback_time_s)
        stats.max_impulse = max(stats.max_impulse, impulse)
        active_pairs_this_frame.add(pair)

    for pair in active_pairs_this_frame:
        collision_stats[pair].active_frames += 1


def print_collision_report(
    collision_stats: Dict[Tuple[str, str], CollisionStats],
) -> None:
    """Print active rigid-body contacts ordered by number of affected frames."""
    print("\nCollision report (contacts with impulse > 1e-8):", flush=True)
    if not collision_stats:
        print("  No active collisions detected.", flush=True)
        return

    ordered_pairs = sorted(
        collision_stats.items(),
        key=lambda item: (item[1].active_frames, item[1].max_impulse),
        reverse=True,
    )
    for (body_a, body_b), stats in ordered_pairs:
        print(
            f"  {body_a} <-> {body_b}: "
            f"active_frames={stats.active_frames}, "
            f"records={stats.contact_records}, "
            f"time=[{stats.first_time_s:.3f}, {stats.last_time_s:.3f}] s, "
            f"max_impulse={stats.max_impulse:.6g}",
            flush=True,
        )


def replay_demo(
    scene: IsaacGymScene,
    demo: DemoTrajectory,
    *,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    realtime: bool = True,
    print_every: int = 60,
    debug_render: bool = False,
    render_mode: str = "full",
    collect_diagnostics: bool = True,
    report_collisions: bool = False,
) -> tuple[int, Optional[ReplayDiagnostics]]:
    """Replay timestamped positions at Isaac Gym's fixed 60 Hz timestep."""
    if start_idx < 0 or start_idx >= demo.sample_count:
        raise IndexError(
            f"start_idx must be in [0, {demo.sample_count - 1}], got {start_idx}"
        )
    if end_idx is None:
        end_idx = demo.sample_count
    if end_idx <= start_idx or end_idx > demo.sample_count:
        raise IndexError(
            f"end_idx must be in ({start_idx}, {demo.sample_count}], got {end_idx}"
        )
    if print_every < 0:
        raise ValueError(f"print_every must be >= 0, got {print_every}")
    if render_mode not in {"full", "no-draw", "poll-only"}:
        raise ValueError(f"Unsupported render_mode: {render_mode}")

    replay_start_s = float(demo.times_s[start_idx])
    replay_end_s = float(demo.times_s[end_idx - 1])
    simulation_times_s = np.arange(
        replay_start_s,
        replay_end_s + 0.5 / TARGET_HZ,
        1.0 / TARGET_HZ,
    )
    # Preserve the exact final target even when the source duration is not an
    # integer multiple of Isaac Gym's 60 Hz simulation timestep.
    if simulation_times_s[-1] < replay_end_s:
        simulation_times_s = np.append(simulation_times_s, replay_end_s)
    else:
        simulation_times_s[-1] = replay_end_s

    diagnostics_enabled = collect_diagnostics and not scene.use_gpu_pipeline
    if collect_diagnostics and not diagnostics_enabled:
        print(
            "Diagnostics plot skipped: not supported with the GPU pipeline.",
            flush=True,
        )
    target_history = (
        np.empty((len(simulation_times_s), len(TARGET_JOINT_NAMES)), dtype=np.float64)
        if diagnostics_enabled
        else None
    )
    actual_history = (
        np.empty((len(simulation_times_s), len(TARGET_JOINT_NAMES)), dtype=np.float64)
        if diagnostics_enabled
        else None
    )
    body_names = (
        tuple(
            scene.gym.get_actor_rigid_body_names(
                scene.env,
                scene.robot_actor,
            )
        )
        if report_collisions
        else ()
    )
    collision_stats = {}  # type: Dict[Tuple[str, str], CollisionStats]

    prepare_scene(scene)
    played_samples = 0
    for frame_idx, playback_time_s in enumerate(simulation_times_s):
        if scene.viewer is not None:
            if debug_render:
                print(f"[render-debug] frame {frame_idx}: poll events", flush=True)
            scene.gym.poll_viewer_events(scene.viewer)
            if debug_render:
                print(f"[render-debug] frame {frame_idx}: query closed", flush=True)
            if scene.gym.query_viewer_has_closed(scene.viewer):
                break

        target_positions = interpolated_targets_to_asset_order(
            demo,
            scene.robot,
            float(playback_time_s),
        )
        if debug_render:
            print(f"[render-debug] frame {frame_idx}: set targets", flush=True)
        scene.gym.set_actor_dof_position_targets(
            scene.env,
            scene.robot_actor,
            target_positions,
        )

        if debug_render:
            print(f"[render-debug] frame {frame_idx}: simulate", flush=True)
        scene.gym.simulate(scene.sim)
        if debug_render:
            print(f"[render-debug] frame {frame_idx}: fetch results", flush=True)
        scene.gym.fetch_results(scene.sim, True)

        if report_collisions:
            _update_collision_report(
                scene,
                float(playback_time_s),
                body_names,
                collision_stats,
            )

        if diagnostics_enabled:
            actual_positions = scene.gym.get_actor_dof_states(
                scene.env,
                scene.robot_actor,
                gymapi.STATE_POS,
            )["pos"]
            target_history[played_samples] = target_positions[
                scene.robot.demo_to_asset_indices
            ]
            actual_history[played_samples] = actual_positions[
                scene.robot.demo_to_asset_indices
            ]

        if scene.viewer is not None and render_mode != "poll-only":
            if debug_render:
                print(f"[render-debug] frame {frame_idx}: step graphics", flush=True)
            scene.gym.step_graphics(scene.sim)

            if render_mode == "full":
                if debug_render:
                    print(
                        f"[render-debug] frame {frame_idx}: draw viewer",
                        flush=True,
                    )
                scene.gym.draw_viewer(scene.viewer, scene.sim, True)
                if debug_render:
                    print(f"[render-debug] frame {frame_idx}: draw done", flush=True)

        if realtime and render_mode == "full":
            if debug_render:
                print(f"[render-debug] frame {frame_idx}: sync frame", flush=True)
            scene.gym.sync_frame_time(scene.sim)
            if debug_render:
                print(f"[render-debug] frame {frame_idx}: sync done", flush=True)

        played_samples += 1
        if print_every and (
            played_samples == 1
            or played_samples % print_every == 0
            or frame_idx == len(simulation_times_s) - 1
        ):
            source_idx = min(
                int(np.searchsorted(demo.times_s, playback_time_s, side="right")),
                end_idx,
            ) - 1
            message = (
                f"frame {frame_idx:04d}/{len(simulation_times_s) - 1:04d} "
                f"t={playback_time_s:.3f}s source_sample={source_idx:04d}"
            )
            if scene.use_gpu_pipeline:
                print(message, flush=True)
            else:
                dof_states = scene.gym.get_actor_dof_states(
                    scene.env,
                    scene.robot_actor,
                    gymapi.STATE_POS,
                )
                position_error = target_positions - dof_states["pos"]
                print(
                    f"{message} "
                    f"max_abs_error={np.max(np.abs(position_error)):.5f} rad",
                    flush=True,
                )

    if report_collisions:
        print_collision_report(collision_stats)

    diagnostics = None
    if diagnostics_enabled:
        diagnostics = ReplayDiagnostics(
            times_s=simulation_times_s[:played_samples].copy(),
            target_positions=target_history[:played_samples].copy(),
            actual_positions=actual_history[:played_samples].copy(),
        )
    return played_samples, diagnostics


def plot_joint_tracking(diagnostics: ReplayDiagnostics, *, title: str) -> None:
    """Show selectable PD target vs. measured joint trajectories."""
    expected_shape = (len(diagnostics.times_s), len(TARGET_JOINT_NAMES))
    if (
        diagnostics.target_positions.shape != expected_shape
        or diagnostics.actual_positions.shape != expected_shape
    ):
        raise ValueError(
            f"Expected target and actual trajectories with shape {expected_shape}, "
            f"got {diagnostics.target_positions.shape} and "
            f"{diagnostics.actual_positions.shape}"
        )

    figure, axes = plt.subplots(figsize=(13, 7))
    figure.subplots_adjust(left=0.10, right=0.71, bottom=0.11, top=0.91)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(TARGET_JOINT_NAMES)))
    joint_lines: list[tuple[object, object]] = []

    for index, joint_name in enumerate(TARGET_JOINT_NAMES):
        (target_line,) = axes.plot(
            diagnostics.times_s,
            diagnostics.target_positions[:, index],
            color=colors[index],
            linewidth=1.4,
            label=f"{joint_name} target",
        )
        (actual_line,) = axes.plot(
            diagnostics.times_s,
            diagnostics.actual_positions[:, index],
            color=colors[index],
            linewidth=1.2,
            linestyle="--",
            label=f"{joint_name} actual",
        )
        joint_lines.append((target_line, actual_line))

    axes.set_title(title)
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Joint position [rad]")
    axes.grid(True, alpha=0.35)

    checkbox_axes = figure.add_axes((0.74, 0.11, 0.24, 0.80))
    checkboxes = CheckButtons(
        checkbox_axes, TARGET_JOINT_NAMES, [True] * len(TARGET_JOINT_NAMES)
    )
    checkbox_axes.set_title("Visible joints", fontsize="medium")

    def visible_lines() -> list[object]:
        return [
            line
            for target_line, actual_line in joint_lines
            if target_line.get_visible()
            for line in (target_line, actual_line)
        ]

    def refresh_legend() -> None:
        if axes.legend_ is not None:
            axes.legend_.remove()
        lines = visible_lines()
        if lines:
            axes.legend(handles=lines, loc="upper left", fontsize="x-small", ncol=2)

    def refresh_y_limits() -> None:
        lines = visible_lines()
        if not lines:
            return
        values = np.concatenate([line.get_ydata() for line in lines])
        lower = float(np.min(values))
        upper = float(np.max(values))
        padding = max(0.02, 0.05 * (upper - lower))
        axes.set_ylim(lower - padding, upper + padding)

    def toggle_joint(joint_name: str) -> None:
        index = TARGET_JOINT_NAMES.index(joint_name)
        target_line, actual_line = joint_lines[index]
        visible = not target_line.get_visible()
        target_line.set_visible(visible)
        actual_line.set_visible(visible)
        refresh_legend()
        refresh_y_limits()
        figure.canvas.draw_idle()

    def set_all_visible(visible: bool) -> None:
        for index, (target_line, _actual_line) in enumerate(joint_lines):
            if target_line.get_visible() != visible:
                checkboxes.set_active(index)
        refresh_legend()
        refresh_y_limits()
        figure.canvas.draw_idle()

    checkboxes.on_clicked(toggle_joint)
    select_all_axes = figure.add_axes((0.74, 0.025, 0.11, 0.05))
    clear_all_axes = figure.add_axes((0.87, 0.025, 0.11, 0.05))
    # Buttons must be kept referenced: an unassigned Button() is garbage
    # collected immediately and its clicks then silently do nothing.
    select_all_button = Button(select_all_axes, "All")
    select_all_button.on_clicked(lambda _event: set_all_visible(True))
    clear_all_button = Button(clear_all_axes, "None")
    clear_all_button.on_clicked(lambda _event: set_all_visible(False))
    refresh_legend()
    refresh_y_limits()
    plt.show()


def destroy_scene(scene: IsaacGymScene) -> None:
    """Release viewer resources before the process exits."""
    if scene.viewer is not None:
        scene.gym.destroy_viewer(scene.viewer)


def run_viewer_smoke_test(
    *,
    compute_device_id: int,
    graphics_device_id: int,
    use_gpu_physx: bool,
    frames: int,
    debug_render: bool,
) -> None:
    """Open the minimal Isaac Gym viewer and draw a ground plane for diagnostics."""
    if frames <= 0:
        raise ValueError(f"frames must be > 0, got {frames}")

    print("Starting minimal Isaac Gym viewer smoke test", flush=True)
    gym, sim = create_sim(
        compute_device_id=compute_device_id,
        graphics_device_id=graphics_device_id,
        use_gpu_physx=use_gpu_physx,
        use_gpu_pipeline=False,
    )
    print("Adding smoke-test ground plane", flush=True)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    print("Creating smoke-test environment", flush=True)
    lower = gymapi.Vec3(-ENV_SPACING, -ENV_SPACING, 0.0)
    upper = gymapi.Vec3(ENV_SPACING, ENV_SPACING, ENV_SPACING)
    env = gym.create_env(sim, lower, upper, 1)
    print("Creating smoke-test viewer", flush=True)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Isaac Gym failed to create the smoke-test viewer")

    print("Positioning smoke-test camera", flush=True)
    gym.viewer_camera_look_at(
        viewer,
        env,
        gymapi.Vec3(1.2, -1.2, 0.8),
        gymapi.Vec3(0.0, 0.0, 0.0),
    )
    print("Preparing smoke-test sim", flush=True)
    gym.prepare_sim(sim)
    print("Smoke-test sim prepared", flush=True)

    try:
        for frame_idx in range(frames):
            if debug_render:
                print(f"[smoke-test] frame {frame_idx}: poll events", flush=True)
            gym.poll_viewer_events(viewer)
            if gym.query_viewer_has_closed(viewer):
                print(f"Viewer closed after {frame_idx} smoke-test frames", flush=True)
                return

            if debug_render:
                print(f"[smoke-test] frame {frame_idx}: simulate", flush=True)
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            if debug_render:
                print(f"[smoke-test] frame {frame_idx}: step graphics", flush=True)
            gym.step_graphics(sim)

            if debug_render:
                print(f"[smoke-test] frame {frame_idx}: draw viewer", flush=True)
            gym.draw_viewer(viewer, sim, True)

            if debug_render:
                print(f"[smoke-test] frame {frame_idx}: sync frame", flush=True)
            gym.sync_frame_time(sim)

        print(f"Smoke test drew {frames} frames", flush=True)
    finally:
        gym.destroy_viewer(viewer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a timestamped teleoperation demo in Isaac Gym.",
    )
    parser.add_argument(
        "demo_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DEMO_PATH,
        help=f"Processed .npz demo path. Default: {DEFAULT_DEMO_PATH}",
    )
    parser.add_argument(
        "--hand-source",
        choices=("measured", "commanded"),
        default="measured",
        help="Which hand trajectory to replay from the demo.",
    )
    parser.add_argument(
        "--compute-device-id",
        type=int,
        default=0,
        help="Isaac Gym compute device id.",
    )
    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=0,
        help="Isaac Gym graphics device id.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Disable GPU PhysX.",
    )
    parser.add_argument(
        "--gpu-pipeline",
        action="store_true",
        help=(
            "Enable Isaac Gym GPU pipeline. This viewer currently uses actor-level "
            "APIs, so this mode is rejected until the replay loop uses tensor APIs."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the Isaac Gym viewer.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="First demo sample to replay.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="One-past-last demo sample to replay.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Replay at most this many samples from start-idx.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not throttle the simulation to wall-clock 60 Hz.",
    )
    parser.add_argument(
        "--debug-render",
        action="store_true",
        help="Print before/after each Isaac Gym viewer and simulation call.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("full", "no-draw", "poll-only"),
        default="full",
        help=(
            "Viewer diagnostic mode: full renders normally, no-draw skips "
            "draw_viewer, poll-only only processes viewer events."
        ),
    )
    parser.add_argument(
        "--viewer-smoke-test",
        action="store_true",
        help="Open a minimal Isaac Gym viewer without loading the robot or demo.",
    )
    parser.add_argument(
        "--smoke-test-frames",
        type=int,
        default=120,
        help="Number of frames for --viewer-smoke-test.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=60,
        help="Print tracking diagnostics every N simulated frames. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-diagnostics-plot",
        action="store_true",
        help="Skip collecting and plotting the PD target-vs-measured tracking plot.",
    )
    parser.add_argument(
        "--report-collisions",
        action="store_true",
        help=(
            "Collect active PhysX contacts during replay and print a rigid-body "
            "pair summary at the end."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = load_demo(args.demo_path, hand_source=args.hand_source)

    end_idx = args.end_idx
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(f"--max-samples must be > 0, got {args.max_samples}")
        max_samples_end_idx = args.start_idx + args.max_samples
        end_idx = (
            min(max_samples_end_idx, demo.sample_count)
            if end_idx is None
            else min(end_idx, max_samples_end_idx)
        )
    if args.gpu_pipeline:
        raise NotImplementedError(
            "--gpu-pipeline requires Isaac Gym tensor APIs for setting DOF targets. "
            "Run without it for the current actor-level viewer."
        )

    graphics_device_id = -1 if args.headless else args.graphics_device_id
    if args.viewer_smoke_test:
        if args.headless:
            raise ValueError("--viewer-smoke-test cannot be used with --headless")
        run_viewer_smoke_test(
            compute_device_id=args.compute_device_id,
            graphics_device_id=graphics_device_id,
            use_gpu_physx=not args.cpu,
            frames=args.smoke_test_frames,
            debug_render=args.debug_render,
        )
        return

    gym, sim = create_sim(
        compute_device_id=args.compute_device_id,
        graphics_device_id=graphics_device_id,
        use_gpu_physx=not args.cpu,
        use_gpu_pipeline=args.gpu_pipeline,
        collect_contacts=args.report_collisions,
    )
    robot_asset = load_robot_asset(gym, sim)
    pd_controller = configure_pd_controller(gym, robot_asset)
    scene = create_scene(
        gym,
        sim,
        robot_asset,
        pd_controller,
        enable_viewer=not args.headless,
        use_gpu_pipeline=args.gpu_pipeline,
    )

    diagnostics = None
    try:
        initialize_robot_from_demo(scene, demo, sample_idx=args.start_idx)
        print(
            f"Replaying {demo.path} "
            f"({demo.sample_count} source samples at {demo.nominal_hz:.2f} Hz, "
            f"{demo.duration_s:.2f} s; simulated at {TARGET_HZ:.0f} Hz)",
            flush=True,
        )
        played_samples, diagnostics = replay_demo(
            scene,
            demo,
            start_idx=args.start_idx,
            end_idx=end_idx,
            realtime=not args.no_realtime,
            print_every=args.print_every,
            debug_render=args.debug_render,
            render_mode=args.render_mode,
            collect_diagnostics=not args.no_diagnostics_plot,
            report_collisions=args.report_collisions,
        )
        print(f"Played {played_samples} simulation frames", flush=True)
    finally:
        destroy_scene(scene)

    if diagnostics is not None:
        plot_joint_tracking(
            diagnostics,
            title=f"Isaac Gym PD tracking — {demo.path.name}",
        )


if __name__ == "__main__":
    main()
