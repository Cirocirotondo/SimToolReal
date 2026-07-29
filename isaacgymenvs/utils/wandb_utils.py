import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from rl_games.common.algo_observer import AlgoObserver

from isaacgymenvs.utils.utils import retry
from isaacgymenvs.utils.reformat import omegaconf_to_dict


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.algo = None
        self._last_periodic_eval_epoch = None

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb

        wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                notes=cfg.wandb_notes if hasattr(cfg, 'wandb_notes') else '',
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method='fork'),
            )
       
            wandb.run.log_code(root=cfg.wandb_logcode_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            print('wandb running directory........', wandb.run.dir)

        print('Initializing WandB...')
        try:
            init_wandb()
            wandb.define_metric("*", step_metric="global_step")
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')

        with open(os.path.join(wandb.run.dir, 'diff.patch'), 'w') as f:
            os.system(f'cd {os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))} && git diff > {f.name}')

        diff_artifact = wandb.Artifact("diff", type="file", description=f"Git diff")
        diff_artifact.add_file(os.path.join(wandb.run.dir, 'diff.patch'))
        wandb.run.log_artifact(diff_artifact)

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    def after_init(self, algo):
        self.algo = algo

    def _periodic_evaluation_enabled(self) -> bool:
        if self.algo is None:
            return False
        env_cfg = self.cfg.task.env
        return bool(
            env_cfg.get("periodicEvaluation", False)
            and str(self.cfg.task_name) == "SimToolRealMotionImitation"
        )

    def after_print_stats(self, frame, epoch_num, total_time):
        if not self._periodic_evaluation_enabled():
            return

        env_cfg = self.cfg.task.env
        frequency = int(env_cfg.get("evaluationFrequencyEpochs", 25))
        if frequency <= 0:
            raise ValueError("evaluationFrequencyEpochs must be positive")
        if (
            epoch_num <= 0
            or epoch_num % frequency != 0
            or epoch_num == self._last_periodic_eval_epoch
        ):
            return
        self._last_periodic_eval_epoch = epoch_num
        try:
            self._run_periodic_evaluation(frame, epoch_num)
        except Exception as exc:
            # Evaluation is diagnostic and must never terminate a training run.
            print(f"Could not start periodic evaluation: {exc}")
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/failed": 1,
                        "eval/epoch": epoch_num,
                        "global_step": frame,
                    }
                )

    def _run_periodic_evaluation(self, frame: int, epoch_num: int) -> None:
        import wandb

        if wandb.run is None:
            print("Skipping periodic evaluation because W&B is not active")
            return

        experiment_dir = Path(self.algo.experiment_dir).resolve()
        config_path = experiment_dir / "config.yaml"
        eval_root = experiment_dir / "periodic_evaluation"
        eval_dir = eval_root / f"epoch_{epoch_num:06d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_stem = eval_root / "latest_policy"
        checkpoint_path = checkpoint_stem.with_suffix(".pth")
        result_path = eval_dir / "metrics.json"
        log_path = eval_dir / "evaluation.log"

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Periodic-evaluation config not found: {config_path}"
            )

        print(
            f"Starting deterministic phase-zero evaluation at epoch {epoch_num}"
        )
        self.algo.save(str(checkpoint_stem))

        repo_root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "-m",
            "dextoolbench.eval_imitation_periodic",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(eval_dir),
            "--result-json",
            str(result_path),
        ]
        timeout_s = int(
            self.cfg.task.env.get("evaluationTimeoutSeconds", 600)
        )

        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                subprocess.run(
                    command,
                    cwd=str(repo_root),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout_s,
                )

            metrics = json.loads(result_path.read_text(encoding="utf-8"))
            video_path = Path(metrics.pop("video_path"))
            payload = {
                f"eval/{name}": value for name, value in metrics.items()
            }
            payload["eval/video"] = wandb.Video(
                str(video_path),
                fps=60,
                format="mp4",
            )
            payload["eval/epoch"] = epoch_num
            payload["eval/failed"] = 0
            payload["global_step"] = frame
            wandb.log(payload)
            print(
                "Periodic evaluation complete: "
                f"reward={metrics['episode_reward']:.4f}, "
                f"phase={metrics['final_phase']:.4f}"
            )
        except Exception as exc:
            print(f"Periodic evaluation failed: {exc}")
            if log_path.is_file():
                tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-4000:]
                print(tail)
            wandb.log(
                {
                    "eval/failed": 1,
                    "eval/epoch": epoch_num,
                    "global_step": frame,
                }
            )
