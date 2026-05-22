"""Shared DexToolBench / training-object settings for eval scripts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent

CUBE_CATEGORY = "cube"
CUBE_OBJECT_NAME = "training_cube"
CUBE_TASK_LIFT_DELTA = "lift_delta"

CUBE_EVAL_URDF = REPO_ROOT / "assets/urdf/eval_cube/cube_5cm.urdf"
CUBE_FIXED_SIZE = [0.05, 0.05, 0.05]

# Matches SimToolReal.yaml robotBaseY / tablePoseDy (60 cm spacing, table at y=0).
ISAAC_ROBOT_BASE_Y = 0.6
ISAAC_TABLE_POSE_DY = -0.6
EVAL_TABLE_RESET_Z = 0.28  # eval-only; training uses tableResetZ: 0.38
ISAAC_ROBOT_BASE_POS: Tuple[float, float, float] = (0.0, ISAAC_ROBOT_BASE_Y, 0.0)
ISAAC_TABLE_CENTER_POS: Tuple[float, float, float] = (
    0.0,
    ISAAC_ROBOT_BASE_Y + ISAAC_TABLE_POSE_DY,
    EVAL_TABLE_RESET_Z,
)


def eval_viser_default_arm_dof(num_arm_dofs: int = 6) -> List[float]:
    """Static Viser arm pose: defaultArmDofPos + startArmHigher (eval overrides)."""
    arm = [-1.5708, -1.571, 1.0, 0.5, 1.571, -1.571]
    arm[1] -= math.radians(10)
    arm[3] += math.radians(10)
    return arm[:num_arm_dofs]


def is_cube_eval(category: str, object_name: str) -> bool:
    return category == CUBE_CATEGORY or object_name == CUBE_OBJECT_NAME


def ensure_cube_eval_urdf() -> Path:
    if not CUBE_EVAL_URDF.exists():
        CUBE_EVAL_URDF.parent.mkdir(parents=True, exist_ok=True)
        from isaacgymenvs.tasks.simtoolreal.generate_objects import (
            generate_cuboid_urdf_constant_density,
        )

        generate_cuboid_urdf_constant_density(
            filepath=CUBE_EVAL_URDF,
            scale=tuple(CUBE_FIXED_SIZE),
            per_face_colors=True,
        )
    return CUBE_EVAL_URDF


def viser_object_urdf_path(category: str, object_name: str) -> Path:
    if is_cube_eval(category, object_name):
        return ensure_cube_eval_urdf()
    from dextoolbench.objects import NAME_TO_OBJECT

    return NAME_TO_OBJECT[object_name].urdf_path


def table_urdf_rel_for_eval(
    category: str,
    object_name: str,
    task_name: str,
    *,
    force_default_table: bool = False,
) -> str:
    if force_default_table or is_cube_eval(category, object_name):
        return "urdf/table_narrow.urdf"

    _TABLE_BY_CATEGORY = {
        "hammer": "urdf/table_narrow_nail.urdf",
        "spatula": "urdf/table_narrow_bowl_plate.urdf",
        "eraser": "urdf/table_narrow_whiteboard.urdf",
        "screwdriver": "urdf/table_narrow.urdf",
        "marker": "urdf/table_narrow_whiteboard.urdf",
        "brush": "urdf/table_narrow.urdf",
    }
    return _TABLE_BY_CATEGORY[category]


def trajectory_path(category: str, object_name: str, task_name: str) -> Path:
    from isaacgymenvs.utils.utils import get_repo_root_dir

    return (
        get_repo_root_dir()
        / "dextoolbench"
        / "trajectories"
        / category
        / object_name
        / f"{task_name}.json"
    )


def build_eval_env_overrides(
    category: str,
    object_name: str,
    table_urdf: Union[str, Path],
    traj_data: Dict[str, Any],
    *,
    z_offset: float = 0.03,
) -> Dict[str, Any]:
    """Hydra overrides shared by eval.py and eval_interactive sim_worker."""
    overrides: Dict[str, Any] = {
        "task.env.resetPositionNoiseX": 0.0,
        "task.env.resetPositionNoiseY": 0.0,
        "task.env.resetPositionNoiseZ": 0.0,
        "task.env.randomizeObjectRotation": False,
        "task.env.resetDofPosRandomIntervalFingers": 0.0,
        "task.env.resetDofPosRandomIntervalArm": 0.0,
        "task.env.resetDofVelRandomInterval": 0.0,
        "task.env.tableResetZRange": 0.0,
        "task.env.numEnvs": 1,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": traj_data["goals"],
        "task.env.useActionDelay": False,
        "task.env.useObsDelay": False,
        "task.env.useObjectStateDelayNoise": False,
        "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
        "task.env.resetWhenDropped": False,
        "task.env.armMovingAverage": 0.1,
        "task.env.evalSuccessTolerance": 0.01,
        "task.env.successSteps": 1,
        "task.env.fixedSizeKeypointReward": True,
        "task.env.fingertipMultiContactDistThresholdM": 0.06,
        "task.env.fingertipMultiContactMinFingers": 5,
        "task.env.fingertipSpreadPenaltyScale": 0.25,
        "task.env.fingertipMultiContactBonusScale": 0.1,
        "task.env.fingertipThumbBonusScale": 0.05,
        "task.env.asset.table": str(table_urdf),
        "task.env.robotBaseY": ISAAC_ROBOT_BASE_Y,
        "task.env.tablePoseDy": ISAAC_TABLE_POSE_DY,
        "task.env.tableResetZ": EVAL_TABLE_RESET_Z,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": traj_data["start_pose"],
        "task.env.startArmHigher": True,
        "task.env.defaultArmDofPos": [
            -1.5708,
            -1.571,
            1.0,
            0.5,
            1.571,
            -1.571,
        ],
        "task.env.forceScale": 0.0,
        "task.env.torqueScale": 0.0,
        "task.env.linVelImpulseScale": 0.0,
        "task.env.angVelImpulseScale": 0.0,
        "task.env.forceOnlyWhenLifted": True,
        "task.env.torqueOnlyWhenLifted": True,
        "task.env.linVelImpulseOnlyWhenLifted": True,
        "task.env.angVelImpulseOnlyWhenLifted": True,
        "task.env.forceProbRange": [0.0001, 0.0001],
        "task.env.torqueProbRange": [0.0001, 0.0001],
        "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
        "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
    }

    if is_cube_eval(category, object_name):
        overrides.update(
            {
                "task.env.objectName": "handle_head_primitives",
                "task.env.handleHeadTypes": ["cube"],
                "task.env.useSingleHandleHeadTemplate": True,
                "task.env.numObjectsPerType": 1,
                "task.env.fixedSize": list(CUBE_FIXED_SIZE),
                "task.env.use_hack_object_pos_offset": False,
            }
        )
    else:
        overrides["task.env.objectName"] = object_name

    return overrides


def load_trajectory(
    category: str, object_name: str, task_name: str, *, z_offset: float = 0.03
) -> Dict[str, Any]:
    import json

    path = trajectory_path(category, object_name, task_name)
    assert path.exists(), f"Trajectory file not found: {path}"
    with open(path) as f:
        traj_data = json.load(f)
    traj_data = dict(traj_data)
    traj_data["start_pose"] = list(traj_data["start_pose"])
    traj_data["start_pose"][2] += z_offset
    traj_data["goals"] = [list(g) for g in traj_data["goals"]]
    return traj_data
