import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import tyro

_TRAINING_PRESETS = ("default", "clean_dr")

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

    training_preset: Literal["default", "clean_dr"] = "default"
    """default = SimToolRealLSTMAsymmetric (delays, noise, pushes). clean_dr = no disturbances, mild reset/URDF DR only (see SimToolRealCleanDR.yaml)."""

    # === Forces/Torques : sim2real disturbances on object (when lifted). Strong legacy: 20 / 2; default below is milder. ===
    force_scale: float = 6.0
    """Force scale (see task env forceScale)."""

    torque_scale: float = 0.5
    """Torque scale (see task env torqueScale)."""

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
    task_name = (
        "SimToolRealLSTMAsymmetricCleanDR"
        if use_clean_dr
        else "SimToolRealLSTMAsymmetric"
    )

    cmd_parts = [
        "python",
        "-m",
        "isaacgymenvs.train",
        "++task.env.useSparseReward=False",
        f"headless={not args.show_viewer}",
        f"task.env.numEnvs={args.num_envs}",
        # === Training ===
        f"train.params.config.minibatch_size={minibatch}",
        f"train.params.config.expl_coef_block_size={args.sapg_block_size}",
        "train.params.config.use_others_experience=lf",
        "train.params.config.off_policy_ratio=1.0",
        "train.params.config.expl_type=mixed_expl_learn_param",
        "train.params.config.expl_reward_type=entropy",
        "train.params.config.expl_reward_coef_scale=0.005",
        "train.params.network.space.continuous.fixed_sigma=coef_cond",
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
        f"task.env.objectAngVelPenaltyScale={args.object_ang_vel_penalty_scale}",
    ]

    if not use_clean_dr:
        cmd_parts.extend(
            [
                f"task.env.forceScale={args.force_scale}",
                f"task.env.torqueScale={args.torque_scale}",
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
