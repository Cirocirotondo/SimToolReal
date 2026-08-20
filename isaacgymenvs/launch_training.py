import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import tyro

_TRAINING_PRESETS = (
    "default",
    "clean_dr",
    "real_dr",
    "train_5_low_table",
    "train_10_real_mid_combined",
    "train_11_simple",
    "train_b1_simple",
    "train_b5",
    "train_b6",
    "train_b61",
    "train_b61_right",
    "train_b62",
    "train_b62_linear",
    "train_b62_right",
    "train_b63",
    "train_b63_right",
    "train_b7",
    "train_b8",
    "train_b81",
    "train_b82",
    "train_b83",
    "train_c1",
    "train_d1",
    "train_d2",
    "train_d3",
    "train_d4",
    "train_d5",
    "train_d6",
    "motion_imitation",
    "motion_imitation_mi00",
    "motion_imitation_mi01",
    "motion_imitation_mi02",
    "motion_imitation_mi03",
    "motion_imitation_mi04",
    "motion_imitation_mi05",
    "motion_imitation_mi06",
    "motion_imitation_mi07",
    "motion_imitation_mi08_positive_gaussian_regularization",
    "motion_imitation_mi09_delta_gaussian_2x",
    "motion_imitation_mi10_target_input_smooth_gaussian",
    "motion_imitation_mi10_arm_dynamics_gaussian_2x",
    "motion_imitation_mi11_combined_gaussian_2x",
    "motion_imitation_llcfix_00",
    "motion_imitation_sapg02_precision",
    "motion_imitation_sapg03_triangular_target_input",
    "motion_imitation_sapg04_joint_regularized",
    "motion_imitation_sapg05_strong_regularization",
    "motion_imitation_sapg06_regularization_curriculum",
    "motion_imitation_sapg07_intermediate_precision",
    "motion_imitation_sapg08_positive_gaussian_regularization",
    "motion_imitation_sapg09_broad_pose_reward",
    "motion_imitation_sapg10_fingertip_tracking",
    "motion_imitation_sapg11_phase_055_085",
    "motion_imitation_sapg12_phase_055_085_uniform_rsi",
    "motion_imitation_sapg13_no_rsi_joint_smoothness",
    "motion_imitation_sapg_obj01_keypoint_tracking",
    "motion_imitation_sapg_obj02_pregrasp_object_priority",
    "motion_imitation_sapg_obj03_object66_imitation33",
    "motion_imitation_sapg_obj04_object50_imitation50",
    "motion_imitation_sapg_obj05_object33_imitation66",
)

_VALID_HANDLE_HEAD_TYPES = frozenset(
    ("hammer", "screwdriver", "marker", "spatula", "eraser", "brush", "cube")
)


@dataclass
class LaunchTrainingArgs:
    """Launch isaacgymenvs training with configurable parameters."""

    # === Experiment ===
    custom_experiment_name: str = "my_experiment"
    """Custom experiment name (datetime will be appended)."""

    seed: int = 0
    """Random seed. Set to -1 to choose random seed."""

    checkpoint: Optional[Path] = None
    """Path to checkpoint .pth file for finetuning. If None, trains from scratch."""

    hand_side: Literal["right", "left"] = "right"
    """Delto hand used by the task. Right is the default; use left for legacy runs."""

    algorithm: Literal["auto", "ppo", "sapg"] = "auto"
    """Policy optimizer/exploration setup.

    Auto preserves the established presets: SAPG is used except for TrainB7
    and the versioned MIxx presets, whose train profiles are PPO. The legacy
    MotionImitation preset retains its previous automatic SAPG behavior.
    """

    training_preset: Literal[
        "default",
        "clean_dr",
        "real_dr",
        "train_5_low_table",
        "train_10_real_mid_combined",
        "train_11_simple",
        "train_b1_simple",
        "train_b5",
        "train_b6",
        "train_b61",
        "train_b61_right",
        "train_b62",
        "train_b62_linear",
        "train_b62_right",
        "train_b63",
        "train_b63_right",
        "train_b7",
        "train_b8",
        "train_b81",
        "train_b82",
        "train_b83",
        "train_c1",
        "train_d1",
        "train_d2",
        "train_d3",
        "train_d4",
        "train_d5",
        "train_d6",
        "motion_imitation",
        "motion_imitation_mi00",
        "motion_imitation_mi01",
        "motion_imitation_mi02",
        "motion_imitation_mi03",
        "motion_imitation_mi04",
        "motion_imitation_mi05",
        "motion_imitation_mi06",
        "motion_imitation_mi07",
        "motion_imitation_mi08_positive_gaussian_regularization",
        "motion_imitation_mi09_delta_gaussian_2x",
        "motion_imitation_mi10_target_input_smooth_gaussian",
        "motion_imitation_mi10_arm_dynamics_gaussian_2x",
        "motion_imitation_mi11_combined_gaussian_2x",
        "motion_imitation_llcfix_00",
        "motion_imitation_sapg02_precision",
        "motion_imitation_sapg03_triangular_target_input",
        "motion_imitation_sapg04_joint_regularized",
        "motion_imitation_sapg05_strong_regularization",
        "motion_imitation_sapg06_regularization_curriculum",
        "motion_imitation_sapg07_intermediate_precision",
        "motion_imitation_sapg08_positive_gaussian_regularization",
        "motion_imitation_sapg09_broad_pose_reward",
        "motion_imitation_sapg10_fingertip_tracking",
        "motion_imitation_sapg11_phase_055_085",
        "motion_imitation_sapg12_phase_055_085_uniform_rsi",
        "motion_imitation_sapg13_no_rsi_joint_smoothness",
        "motion_imitation_sapg_obj01_keypoint_tracking",
        "motion_imitation_sapg_obj02_pregrasp_object_priority",
        "motion_imitation_sapg_obj03_object66_imitation33",
        "motion_imitation_sapg_obj04_object50_imitation50",
        "motion_imitation_sapg_obj05_object33_imitation66",
    ] = "default"
    """Select the named training preset.

    B8-B83 are the right-hand 5 x 5 x 15 cm cuboid curricula. B63Right is
    the right-hand command-randomization curriculum. D1-D2 use
    operational-space arm actions, D3-D5 add curated reference-state
    initialization, and D6 adapts the task to the 20 x 9 x 9 cm dumbbell.
    MotionImitation is the legacy DeepMimic-style preset. MI00 is the original
    adaptive-PPO baseline, MI01 keeps the same task with fixed-LR PPO, MI02
    adds triangular phase RSI, no action penalties, and linear LR decay, and
    MI03 augments MI02 with filtered reference-velocity tracking. MI04 uses
    matched-window velocities, reset warm-up, and action-delta regularization.
    MI05 strengthens that action-delta regularization.
    MI06 adds the desired palm pose and finger configuration to the MI05 input.
    MI07 removes phase from MI06 and adds target palm and finger velocities.
    MI08 branches from MI04 and replaces negative quadratic regularizers with
    bounded positive Gaussian rewards using explicit scale/sigma pairs.
    MI09 doubles only action-delta Gaussian strength. MI10 uses exactly the
    SAPG08 task configuration with PPO, isolating the optimization algorithm
    as the intended experimental variable. The old MI10 arm dynamics preset
    name remains as a compatibility alias. MI11 retains the earlier factorial
    delta-plus-arm-dynamics experiment.
    SAPG02Precision applies tighter pose rewards to the SAPG01 baseline.
    SAPG03TriangularTargetInput trains from zero with those precision rewards,
    triangular RSI, and the desired palm position in the policy observation.
    SAPG04JointRegularized adds weak command costs and measured arm-joint
    velocity/acceleration regularization to SAPG03.
    SAPG05StrongRegularization increases those costs using SAPG04 measurements
    and temporarily disables measured hand-joint acceleration regularization.
    SAPG06RegularizationCurriculum fine-tunes SAPG04 with a warm-up and a
    smooth gradual increase of only the vibration-related regularizers.
    SAPG07 returns to SAPG04's regularization setup but uses intermediate
    pose-reward kernels between SAPG01 and SAPG02 Precision.
    SAPG08 broadens SAPG07 orientation tracking, replaces the negative
    quadratic regularizers with calibrated bounded positive Gaussians, and
    lightly smooths controller targets to suppress bang-bang vibration.
    SAPG09 isolates reward sharpness by changing only SAPG07's pose kernels
    from 200/5.477/1 to the broader 100/2/0.5 values.
    SAPG10 derives Cartesian fingertip targets by FK and adds them to SAPG08's
    pose reward with weights 0.35/0.25/0.25/0.15.
    SAPG11 keeps SAPG08 unchanged but restricts episodes and RSI to the
    original demonstration phase interval [0.55, 0.85].
    SAPG12 changes only SAPG11's RSI distribution from triangular to uniform
    over the same [0.55, 0.85] phase interval.
    SAPG13 disables RSI on that segment and keeps only pose/hand tracking plus
    bounded arm/hand measured-joint acceleration smoothness rewards.
    LLCFix00 starts the post-low-level-controller-fix family: discrete 60 Hz
    joint-only references, uniform RSI, absolute joint-angle actions, and PPO.
    SAPG-OBJ01 derives from SAPG05 and adds the recorded physical object,
    four-keypoint object tracking reward, and object-aware observations.
    SAPG-OBJ02 makes object tracking dominant, terminates beyond 6 cm, and
    anchors half of the RSI resets at the grounded pre-grasp phase 0.6.
    SAPG-OBJ03/04/05 add bounded lift/fingertip grasp shaping and compare
    object/imitation primary-reward ratios of 2:1, 1:1, and 1:2.
    The remaining names preserve the established cube and hand-only
    curricula.
    """

    # === Forces/Torques : sim2real disturbances on object (when lifted). ===
    force_scale: Optional[float] = None
    """Force scale override. If unset: default/train_5_low_table/train_11_simple=6.0, train_10_real_mid_combined=12.0, real_dr=20.0."""

    torque_scale: Optional[float] = None
    """Torque scale override. If unset: default/train_5_low_table/train_11_simple=0.5, train_10_real_mid_combined=1.0, real_dr=2.0."""

    handle_head_type: Optional[str] = None
    """If set, only this procedural tool family is used (see task env handleHeadTypes)."""

    # === Penalty ===
    object_ang_vel_penalty_scale: float = 0.0
    """Object angular velocity penalty scale."""

    # === SAPG ===
    num_envs: int = 12288
    """Number of environments (from_zero default in SimToolReal.yaml). Increase if you have GPU headroom."""

    num_blocks: int = 6
    """SAPG block count (must match checkpoint when fine-tuning). Without checkpoint, may be lowered so num_envs divides evenly."""

    show_viewer: bool = False
    """If True, headless=False (finestra Isaac Gym). Con pochi env, minibatch viene ridotto automaticamente."""

    disable_video: bool = False
    """Disable native/W&B videos, camera sensors, and periodic video evaluation for renderless servers."""

    max_frames: Optional[int] = None
    """Maximum aggregate environment transitions. None keeps the train profile default."""

    # === Wandb ===
    wandb_entity: str = "simonecirelli-eth"
    """Wandb entity (user or team)."""

    wandb_project: str = "simtoolreal"
    """Wandb project name."""

    wandb_group: str = f"{datetime.now().strftime('%Y-%m-%d')}"
    """Wandb group name."""

    wandb_activate: bool = True
    """Whether to activate wandb logging."""

    wandb_tags: List[str] = field(default_factory=list)
    """Wandb tags."""

    wandb_notes: str = ""
    """Wandb notes."""

    @property
    def sapg_block_size(self) -> int:
        return self.num_envs // self.num_blocks

    def __post_init__(self) -> None:
        preferred_blocks = max(1, min(self.num_blocks, self.num_envs))

        if self.checkpoint is not None:
            # Weights include tensors shaped for num_blocks (e.g. extra_params [num_blocks, 32]);
            # do not reduce num_blocks when resuming — only bump num_envs to a multiple.
            if self.num_envs % preferred_blocks != 0:
                old_n = self.num_envs
                self.num_envs = (
                    (self.num_envs + preferred_blocks - 1) // preferred_blocks
                ) * preferred_blocks
                print(
                    f"[launch_training] num_envs {old_n} -> {self.num_envs} "
                    f"(multiple of num_blocks={preferred_blocks} required when loading a checkpoint)"
                )
            resolved = preferred_blocks
        else:
            resolved = None
            for nb in range(preferred_blocks, 0, -1):
                if self.num_envs % nb == 0:
                    resolved = nb
                    break
            assert resolved is not None
            if resolved != self.num_blocks:
                print(
                    f"[launch_training] num_blocks: {self.num_blocks} -> {resolved} "
                    f"(num_envs={self.num_envs} must be divisible by num_blocks)"
                )
        self.num_blocks = resolved
        if self.handle_head_type is not None:
            if self.handle_head_type not in _VALID_HANDLE_HEAD_TYPES:
                raise ValueError(
                    f"handle_head_type must be one of {sorted(_VALID_HANDLE_HEAD_TYPES)}, "
                    f"got {self.handle_head_type!r}"
                )
        if self.training_preset not in _TRAINING_PRESETS:
            raise ValueError(
                f"training_preset must be one of {_TRAINING_PRESETS}, "
                f"got {self.training_preset!r}"
            )
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError(
                f"max_frames must be positive when set, got {self.max_frames}"
            )


def launch_training(args: LaunchTrainingArgs) -> None:
    if args.checkpoint is not None:
        assert args.checkpoint.exists(), f"Checkpoint not found: {args.checkpoint}"
    if (
        args.training_preset
        == "motion_imitation_sapg06_regularization_curriculum"
        and args.checkpoint is None
    ):
        raise ValueError(
            "SAPG06 is a fine-tuning curriculum and requires --checkpoint "
            "pointing to the SAPG04 model"
        )

    now = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )  # Add this to avoid overwriting existing experiments
    experiment_name = f"{args.custom_experiment_name}_{now}"
    hydra_run_dir = (
        f"./train_dir/{args.wandb_project}/{args.wandb_group}/{experiment_name}"
    )

    wandb_tags_str = "[" + ",".join(args.wandb_tags) + "]"

    # Deve coincidere con train.params.config.horizon_length in SimToolRealPPO.yaml
    horizon_length = 16
    rollout_size = args.num_envs * horizon_length
    default_minibatch = 98304
    minibatch = default_minibatch if rollout_size >= default_minibatch else rollout_size
    if minibatch < 1:
        raise ValueError("num_envs troppo piccolo per un rollout valido")

    use_clean_dr = args.training_preset == "clean_dr"
    task_name = {
        "default": "SimToolRealLSTMAsymmetric",
        "clean_dr": "SimToolRealLSTMAsymmetricCleanDR",
        "real_dr": "SimToolRealLSTMAsymmetricRealDR",
        "train_5_low_table": "SimToolRealLSTMAsymmetricTrain5LowTable",
        "train_10_real_mid_combined": "SimToolRealLSTMAsymmetricTrain10RealMidCombined",
        "train_11_simple": "SimToolRealLSTMAsymmetricTrain11Simple",
        "train_b1_simple": "SimToolRealLSTMAsymmetricTrainB1Simple",
        "train_b5": "SimToolRealLSTMAsymmetricTrainB5",
        "train_b6": "SimToolRealLSTMAsymmetricTrainB6",
        "train_b61": "SimToolRealLSTMAsymmetricTrainB61",
        "train_b61_right": "SimToolRealLSTMAsymmetricTrainB61Right",
        "train_b62": "SimToolRealLSTMAsymmetricTrainB62",
        "train_b62_linear": "SimToolRealLSTMAsymmetricTrainB62Linear",
        "train_b62_right": "SimToolRealLSTMAsymmetricTrainB62Right",
        "train_b63": "SimToolRealLSTMAsymmetricTrainB63",
        "train_b63_right": "SimToolRealLSTMAsymmetricTrainB63Right",
        "train_b7": "SimToolRealLSTMAsymmetricTrainB7",
        "train_b8": "SimToolRealLSTMAsymmetricTrainB8",
        "train_b81": "SimToolRealLSTMAsymmetricTrainB81",
        "train_b82": "SimToolRealLSTMAsymmetricTrainB82",
        "train_b83": "SimToolRealLSTMAsymmetricTrainB83",
        "train_c1": "SimToolRealLSTMAsymmetricTrainC1",
        "train_d1": "SimToolRealLSTMAsymmetricTrainD1",
        "train_d2": "SimToolRealLSTMAsymmetricTrainD2",
        "train_d3": "SimToolRealLSTMAsymmetricTrainD3",
        "train_d4": "SimToolRealLSTMAsymmetricTrainD4",
        "train_d5": "SimToolRealLSTMAsymmetricTrainD5",
        "train_d6": "SimToolRealLSTMAsymmetricTrainD6",
        "motion_imitation": "SimToolRealMotionImitation",
        "motion_imitation_mi00": "SimToolRealMotionImitationMI00",
        "motion_imitation_mi01": "SimToolRealMotionImitationMI00",
        "motion_imitation_mi02": "SimToolRealMotionImitationMI02",
        "motion_imitation_mi03": "SimToolRealMotionImitationMI03",
        "motion_imitation_mi04": "SimToolRealMotionImitationMI04",
        "motion_imitation_mi05": "SimToolRealMotionImitationMI05",
        "motion_imitation_mi06": "SimToolRealMotionImitationMI06",
        "motion_imitation_mi07": "SimToolRealMotionImitationMI07",
        "motion_imitation_mi08_positive_gaussian_regularization": "SimToolRealMotionImitationMI08PositiveGaussianRegularization",
        "motion_imitation_mi09_delta_gaussian_2x": "SimToolRealMotionImitationMI09DeltaGaussian2x",
        "motion_imitation_mi10_target_input_smooth_gaussian": "SimToolRealMotionImitationMI10TargetInputSmoothGaussian",
        "motion_imitation_mi10_arm_dynamics_gaussian_2x": "SimToolRealMotionImitationMI10TargetInputSmoothGaussian",
        "motion_imitation_mi11_combined_gaussian_2x": "SimToolRealMotionImitationMI11CombinedGaussian2x",
        "motion_imitation_llcfix_00": "SimToolRealMotionImitationLLCFix00",
        "motion_imitation_sapg02_precision": "SimToolRealMotionImitationSAPG02Precision",
        "motion_imitation_sapg03_triangular_target_input": "SimToolRealMotionImitationSAPG03TriangularTargetInput",
        "motion_imitation_sapg04_joint_regularized": "SimToolRealMotionImitationSAPG04JointRegularized",
        "motion_imitation_sapg05_strong_regularization": "SimToolRealMotionImitationSAPG05StrongRegularization",
        "motion_imitation_sapg06_regularization_curriculum": "SimToolRealMotionImitationSAPG06RegularizationCurriculum",
        "motion_imitation_sapg07_intermediate_precision": "SimToolRealMotionImitationSAPG07IntermediatePrecision",
        "motion_imitation_sapg08_positive_gaussian_regularization": "SimToolRealMotionImitationSAPG08PositiveGaussianRegularization",
        "motion_imitation_sapg09_broad_pose_reward": "SimToolRealMotionImitationSAPG09BroadPoseReward",
        "motion_imitation_sapg10_fingertip_tracking": "SimToolRealMotionImitationSAPG10FingertipTracking",
        "motion_imitation_sapg11_phase_055_085": "SimToolRealMotionImitationSAPG11Phase055To085",
        "motion_imitation_sapg12_phase_055_085_uniform_rsi": "SimToolRealMotionImitationSAPG12Phase055To085UniformRSI",
        "motion_imitation_sapg13_no_rsi_joint_smoothness": "SimToolRealMotionImitationSAPG13NoRSIJointSmoothness",
        "motion_imitation_sapg_obj01_keypoint_tracking": "SimToolRealMotionImitationSAPGOBJ01KeypointTracking",
        "motion_imitation_sapg_obj02_pregrasp_object_priority": "SimToolRealMotionImitationSAPGOBJ02PregraspObjectPriority",
        "motion_imitation_sapg_obj03_object66_imitation33": "SimToolRealMotionImitationSAPGOBJ03Object66Imitation33",
        "motion_imitation_sapg_obj04_object50_imitation50": "SimToolRealMotionImitationSAPGOBJ04Object50Imitation50",
        "motion_imitation_sapg_obj05_object33_imitation66": "SimToolRealMotionImitationSAPGOBJ05Object33Imitation66",
    }[args.training_preset]
    force_scale = args.force_scale
    torque_scale = args.torque_scale
    if args.training_preset in {
        "default",
        "train_5_low_table",
        "train_11_simple",
        "train_b1_simple",
        "train_b6",
        "train_b61",
        "train_b61_right",
        "train_b62",
        "train_b62_linear",
        "train_b62_right",
        "train_b63",
        "train_b63_right",
        "train_b7",
        "train_b8",
        "train_b81",
        "train_b82",
        "train_b83",
        "train_c1",
    }:
        force_scale = 6.0 if force_scale is None else force_scale
        torque_scale = 0.5 if torque_scale is None else torque_scale
    elif args.training_preset in {
        "train_b5",
        "train_d1",
        "train_d2",
        "train_d3",
        "train_d4",
        "train_d5",
        "train_d6",
        "motion_imitation",
        "motion_imitation_mi00",
        "motion_imitation_mi01",
        "motion_imitation_mi02",
        "motion_imitation_mi03",
        "motion_imitation_mi04",
        "motion_imitation_mi05",
        "motion_imitation_mi06",
        "motion_imitation_mi07",
        "motion_imitation_mi08_positive_gaussian_regularization",
        "motion_imitation_mi09_delta_gaussian_2x",
        "motion_imitation_mi10_target_input_smooth_gaussian",
        "motion_imitation_mi10_arm_dynamics_gaussian_2x",
        "motion_imitation_mi11_combined_gaussian_2x",
        "motion_imitation_llcfix_00",
        "motion_imitation_sapg02_precision",
        "motion_imitation_sapg03_triangular_target_input",
        "motion_imitation_sapg04_joint_regularized",
        "motion_imitation_sapg05_strong_regularization",
        "motion_imitation_sapg06_regularization_curriculum",
        "motion_imitation_sapg07_intermediate_precision",
        "motion_imitation_sapg08_positive_gaussian_regularization",
        "motion_imitation_sapg09_broad_pose_reward",
        "motion_imitation_sapg10_fingertip_tracking",
        "motion_imitation_sapg11_phase_055_085",
        "motion_imitation_sapg12_phase_055_085_uniform_rsi",
        "motion_imitation_sapg13_no_rsi_joint_smoothness",
        "motion_imitation_sapg_obj01_keypoint_tracking",
        "motion_imitation_sapg_obj02_pregrasp_object_priority",
        "motion_imitation_sapg_obj03_object66_imitation33",
        "motion_imitation_sapg_obj04_object50_imitation50",
        "motion_imitation_sapg_obj05_object33_imitation66",
    }:
        force_scale = 0.0 if force_scale is None else force_scale
        torque_scale = 0.0 if torque_scale is None else torque_scale
    elif args.training_preset == "train_10_real_mid_combined":
        force_scale = 12.0 if force_scale is None else force_scale
        torque_scale = 1.0 if torque_scale is None else torque_scale
    elif args.training_preset == "real_dr":
        force_scale = 20.0 if force_scale is None else force_scale
        torque_scale = 2.0 if torque_scale is None else torque_scale

    is_motion_imitation = args.training_preset in {
        "motion_imitation",
        "motion_imitation_mi00",
        "motion_imitation_mi01",
        "motion_imitation_mi02",
        "motion_imitation_mi03",
        "motion_imitation_mi04",
        "motion_imitation_mi05",
        "motion_imitation_mi06",
        "motion_imitation_mi07",
        "motion_imitation_mi08_positive_gaussian_regularization",
        "motion_imitation_mi09_delta_gaussian_2x",
        "motion_imitation_mi10_target_input_smooth_gaussian",
        "motion_imitation_mi10_arm_dynamics_gaussian_2x",
        "motion_imitation_mi11_combined_gaussian_2x",
        "motion_imitation_llcfix_00",
        "motion_imitation_sapg02_precision",
        "motion_imitation_sapg03_triangular_target_input",
        "motion_imitation_sapg04_joint_regularized",
        "motion_imitation_sapg05_strong_regularization",
        "motion_imitation_sapg06_regularization_curriculum",
        "motion_imitation_sapg07_intermediate_precision",
        "motion_imitation_sapg08_positive_gaussian_regularization",
        "motion_imitation_sapg09_broad_pose_reward",
        "motion_imitation_sapg10_fingertip_tracking",
        "motion_imitation_sapg11_phase_055_085",
        "motion_imitation_sapg12_phase_055_085_uniform_rsi",
        "motion_imitation_sapg13_no_rsi_joint_smoothness",
        "motion_imitation_sapg_obj01_keypoint_tracking",
        "motion_imitation_sapg_obj02_pregrasp_object_priority",
        "motion_imitation_sapg_obj03_object66_imitation33",
        "motion_imitation_sapg_obj04_object50_imitation50",
        "motion_imitation_sapg_obj05_object33_imitation66",
    }
    auto_ppo_presets = {
        "train_b7",
        "motion_imitation_mi00",
        "motion_imitation_mi01",
        "motion_imitation_mi02",
        "motion_imitation_mi03",
        "motion_imitation_mi04",
        "motion_imitation_mi05",
        "motion_imitation_mi06",
        "motion_imitation_mi07",
        "motion_imitation_mi08_positive_gaussian_regularization",
        "motion_imitation_mi09_delta_gaussian_2x",
        "motion_imitation_mi10_target_input_smooth_gaussian",
        "motion_imitation_mi10_arm_dynamics_gaussian_2x",
        "motion_imitation_mi11_combined_gaussian_2x",
        "motion_imitation_llcfix_00",
    }
    use_sapg = args.algorithm == "sapg" or (
        args.algorithm == "auto"
        and args.training_preset not in auto_ppo_presets
    )

    cmd_parts = [
        shlex.quote(sys.executable),
        "-m",
        "isaacgymenvs.train",
        f"headless={not args.show_viewer}",
        f"task.env.numEnvs={args.num_envs}",
        # === Training ===
        f"train.params.config.minibatch_size={minibatch}",
        "multi_gpu=False",
        # === Wandb ===
        f"wandb_project={args.wandb_project}",
        f"wandb_entity={args.wandb_entity}",
        f"wandb_activate={args.wandb_activate}",
        f"wandb_group={args.wandb_group}",
        f"wandb_tags={wandb_tags_str}",
        f"++wandb_notes='{args.wandb_notes}'",
        # === Seed ===
        f"seed={args.seed}",
        # === Experiment ===
        f"experiment=00_{experiment_name}",
        f"hydra.run.dir={hydra_run_dir}",
        f"task={task_name}",
        f"task.env.handSide={args.hand_side}",
        f"task.env.objectAngVelPenaltyScale={args.object_ang_vel_penalty_scale}",
    ]

    if not is_motion_imitation:
        cmd_parts.extend(
            [
                "++task.env.useSparseReward=False",
                "train.params.config.good_reset_boundary=0",
                "task.env.goodResetBoundary=0",
                f"train.params.config.central_value_config.minibatch_size={minibatch}",
            ]
        )

    motion_imitation_train_profiles = {
        "motion_imitation_mi00": "SimToolRealMotionImitationMI00PPO",
        "motion_imitation_mi01": "SimToolRealMotionImitationMI01PPO",
        "motion_imitation_mi02": "SimToolRealMotionImitationMI02PPO",
        "motion_imitation_mi03": "SimToolRealMotionImitationMI03PPO",
        "motion_imitation_mi04": "SimToolRealMotionImitationMI04PPO",
        "motion_imitation_mi05": "SimToolRealMotionImitationMI05PPO",
        "motion_imitation_mi06": "SimToolRealMotionImitationMI06PPO",
        "motion_imitation_mi07": "SimToolRealMotionImitationMI07PPO",
        "motion_imitation_mi08_positive_gaussian_regularization": "SimToolRealMotionImitationMI08PPO",
        "motion_imitation_mi09_delta_gaussian_2x": "SimToolRealMotionImitationMI09PPO",
        "motion_imitation_mi10_target_input_smooth_gaussian": "SimToolRealMotionImitationMI10PPO",
        "motion_imitation_mi10_arm_dynamics_gaussian_2x": "SimToolRealMotionImitationMI10PPO",
        "motion_imitation_mi11_combined_gaussian_2x": "SimToolRealMotionImitationMI11PPO",
        "motion_imitation_llcfix_00": "SimToolRealMotionImitationLLCFix00PPO",
        "motion_imitation_sapg02_precision": "SimToolRealMotionImitationPPO",
        # SAPG03 starts from zero but keeps the established six-block SAPG
        # optimizer/network profile, now with a 104-dimensional observation.
        "motion_imitation_sapg03_triangular_target_input": "SimToolRealMotionImitationPPO",
        "motion_imitation_sapg04_joint_regularized": "SimToolRealMotionImitationSAPG04PPO",
        "motion_imitation_sapg05_strong_regularization": "SimToolRealMotionImitationSAPG05PPO",
        "motion_imitation_sapg06_regularization_curriculum": "SimToolRealMotionImitationSAPG06PPO",
        "motion_imitation_sapg07_intermediate_precision": "SimToolRealMotionImitationSAPG07PPO",
        "motion_imitation_sapg08_positive_gaussian_regularization": "SimToolRealMotionImitationSAPG08PPO",
        "motion_imitation_sapg09_broad_pose_reward": "SimToolRealMotionImitationSAPG09PPO",
        "motion_imitation_sapg10_fingertip_tracking": "SimToolRealMotionImitationSAPG10PPO",
        "motion_imitation_sapg11_phase_055_085": "SimToolRealMotionImitationSAPG11PPO",
        "motion_imitation_sapg12_phase_055_085_uniform_rsi": "SimToolRealMotionImitationSAPG12PPO",
        "motion_imitation_sapg13_no_rsi_joint_smoothness": "SimToolRealMotionImitationSAPG13PPO",
        "motion_imitation_sapg_obj01_keypoint_tracking": "SimToolRealMotionImitationSAPGOBJ01PPO",
        "motion_imitation_sapg_obj02_pregrasp_object_priority": "SimToolRealMotionImitationSAPGOBJ02PPO",
        "motion_imitation_sapg_obj03_object66_imitation33": "SimToolRealMotionImitationSAPGOBJ03PPO",
        "motion_imitation_sapg_obj04_object50_imitation50": "SimToolRealMotionImitationSAPGOBJ04PPO",
        "motion_imitation_sapg_obj05_object33_imitation66": "SimToolRealMotionImitationSAPGOBJ05PPO",
    }
    train_profile = motion_imitation_train_profiles.get(args.training_preset)
    if train_profile is not None:
        cmd_parts.append(f"train={train_profile}")

    # Match the established grasping SAPG setup: six (by default) exploration
    # populations, lead/follower experience sharing, entropy-conditioned
    # exploration, and one learned sigma vector per population.
    if use_sapg:
        cmd_parts.extend(
            [
                f"train.params.config.expl_coef_block_size={args.sapg_block_size}",
                "train.params.config.use_others_experience=lf",
                "train.params.config.off_policy_ratio=1.0",
                "train.params.config.expl_type=mixed_expl_learn_param",
                "train.params.config.expl_reward_type=entropy",
                "train.params.config.expl_reward_coef_scale=0.005",
                "train.params.network.space.continuous.fixed_sigma=coef_cond",
            ]
        )

    if not use_clean_dr:
        cmd_parts.extend(
            [
                f"task.env.forceScale={force_scale}",
                f"task.env.torqueScale={torque_scale}",
            ]
        )

    if args.disable_video:
        # Some remote compute nodes have no graphics device or video encoder.
        # Disable both recording paths: the generic wrapper/native camera path
        # and the isolated periodic evaluator, which always renders an MP4.
        cmd_parts.extend(
            [
                "capture_video=False",
                "force_render=False",
                "task.env.capture_video=False",
                "task.env.enableCameraSensors=False",
                "task.env.visualizeReferenceRobotInVideo=False",
                "task.env.referenceVisualizationActorEnabled=False",
                "task.env.periodicEvaluation=False",
            ]
        )

    if args.show_viewer:
        # Ensure draw_viewer runs each sub-step; disable long wandb video capture (it toggles sync).
        cmd_parts.extend(
            [
                "force_render=True",
                "capture_video=False",
                "task.env.capture_video=False",
            ]
        )

    if args.handle_head_type is not None:
        cmd_parts.append(
            f"task.env.handleHeadTypes=['{args.handle_head_type}']"
        )

    if args.max_frames is not None:
        # rl_games checks this after each rollout/update epoch. One frame is
        # one aggregate environment transition (not one optimizer update).
        cmd_parts.append(f"train.params.config.max_frames={args.max_frames}")

    if args.checkpoint is not None:
        cmd_parts.append(f"checkpoint={args.checkpoint}")

    cmd = " ".join(cmd_parts)
    print(f"Running command:\n{cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    args: LaunchTrainingArgs = tyro.cli(LaunchTrainingArgs)
    launch_training(args)


if __name__ == "__main__":
    main()
