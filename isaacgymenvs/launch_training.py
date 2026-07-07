import subprocess
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
    "train_b7",
    "train_c1",
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
        "train_b7",
        "train_c1",
    ] = "default"
    """default = prior disturbed setup. clean_dr = reduced disturbances. real_dr = heavier sim-to-real DR. train_5_low_table = train-5-like settings on the low-table scene. train_11_simple = train_5_low_table without action delay. train_b1_simple = simple low-table setup with relative hand actions. train_b5 = B1-derived setup with precision curriculum and no lifted-grasp shaping. train_b6 = B5-style setup with mild cube disturbances and looser target precision. train_b61 = first B6 sim-to-real curriculum stage with immediate reset and calibration variation. train_b61_right = right-hand B61 whose reset variation is introduced through a linear curriculum. train_b62 = B61 plus immediate moderate contact and physics randomization. train_b62_linear = B61 plus the same contact DR introduced through a linear curriculum. train_b62_right = B61Right plus the same contact DR introduced through a linear curriculum. train_b63 = B62 plus moderate action delay and Gaussian action noise. train_b7 = B6 task trained from scratch with standard PPO instead of SAPG. train_10_real_mid_combined = intermediate sim2real preset with moderate delay/noise/contact DR. train_c1 = simple grasping task with only hand control (no arm)."""

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


def launch_training(args: LaunchTrainingArgs) -> None:
    if args.checkpoint is not None:
        assert args.checkpoint.exists(), f"Checkpoint not found: {args.checkpoint}"

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
        "train_b7": "SimToolRealLSTMAsymmetricTrainB7",
        "train_c1": "SimToolRealLSTMAsymmetricTrainC1",
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
        "train_b7",
        "train_c1",
    }:
        force_scale = 6.0 if force_scale is None else force_scale
        torque_scale = 0.5 if torque_scale is None else torque_scale
    elif args.training_preset == "train_b5":
        force_scale = 0.0 if force_scale is None else force_scale
        torque_scale = 0.0 if torque_scale is None else torque_scale
    elif args.training_preset == "train_10_real_mid_combined":
        force_scale = 12.0 if force_scale is None else force_scale
        torque_scale = 1.0 if torque_scale is None else torque_scale
    elif args.training_preset == "real_dr":
        force_scale = 20.0 if force_scale is None else force_scale
        torque_scale = 2.0 if torque_scale is None else torque_scale

    cmd_parts = [
        "python",
        "-m",
        "isaacgymenvs.train",
        "++task.env.useSparseReward=False",
        f"headless={not args.show_viewer}",
        f"task.env.numEnvs={args.num_envs}",
        # === Training ===
        f"train.params.config.minibatch_size={minibatch}",
        "multi_gpu=False",
        "train.params.config.good_reset_boundary=0",
        "task.env.goodResetBoundary=0",
        f"train.params.config.central_value_config.minibatch_size={minibatch}",
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

    if args.training_preset != "train_b7":
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

    if args.show_viewer:
        # Ensure draw_viewer runs each sub-step; disable long wandb video capture (it toggles sync).
        cmd_parts.extend(
            ["force_render=True", "task.env.capture_video=False"]
        )

    if args.handle_head_type is not None:
        cmd_parts.append(
            f"task.env.handleHeadTypes=['{args.handle_head_type}']"
        )

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
