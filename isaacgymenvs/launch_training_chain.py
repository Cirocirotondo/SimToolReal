"""Run independent Isaac Gym training experiments sequentially.

Each child run uses :mod:`isaacgymenvs.launch_training`, so naming, Hydra,
W&B, checkpointing, and preset selection remain identical to a manual launch.
The chain-level ``max_frames`` value is injected into every run.
"""

import json
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tyro

from isaacgymenvs.launch_training import LaunchTrainingArgs, launch_training


@dataclass
class LaunchTrainingChainArgs:
    """Launch a sequence of training runs described by a JSON manifest."""

    config: Path
    """Path to the chain JSON manifest."""

    dry_run: bool = False
    """Validate and print the resolved runs without launching training."""


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training-chain manifest not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    if not isinstance(manifest, dict):
        raise ValueError("The training-chain manifest root must be a JSON object")
    return manifest


def _resolve_runs(
    manifest: Dict[str, Any], manifest_path: Path
) -> Tuple[List[LaunchTrainingArgs], bool]:
    allowed_top_level = {
        "max_frames",
        "continue_on_error",
        "common",
        "runs",
        "summary_directory",
    }
    unknown_top_level = set(manifest) - allowed_top_level
    if unknown_top_level:
        raise ValueError(
            f"Unknown chain settings: {sorted(unknown_top_level)}"
        )

    max_frames = manifest.get("max_frames", 600_000_000)
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")

    common = manifest.get("common", {})
    runs = manifest.get("runs")
    if not isinstance(common, dict):
        raise ValueError("common must be a JSON object")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty JSON array")

    allowed_run_keys = {field.name for field in fields(LaunchTrainingArgs)}
    resolved_runs: List[LaunchTrainingArgs] = []

    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index - 1}] must be a JSON object")

        merged = {**common, **run}
        unknown_keys = set(merged) - allowed_run_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown settings in run {index}: {sorted(unknown_keys)}"
            )
        if "max_frames" in merged:
            raise ValueError(
                "max_frames is a chain-level setting and cannot be overridden "
                f"inside run {index}"
            )
        if "custom_experiment_name" not in merged:
            raise ValueError(
                f"Run {index} must define custom_experiment_name explicitly"
            )

        checkpoint = merged.get("checkpoint")
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = manifest_path.parent / checkpoint_path
            merged["checkpoint"] = checkpoint_path.resolve()

        merged["max_frames"] = max_frames
        resolved_runs.append(LaunchTrainingArgs(**merged))

    experiment_names = [run.custom_experiment_name for run in resolved_runs]
    duplicate_names = sorted(
        {name for name in experiment_names if experiment_names.count(name) > 1}
    )
    if duplicate_names:
        raise ValueError(
            f"Every run needs a unique custom_experiment_name; duplicates: {duplicate_names}"
        )

    continue_on_error = manifest.get("continue_on_error", True)
    if not isinstance(continue_on_error, bool):
        raise ValueError("continue_on_error must be true or false")
    return resolved_runs, continue_on_error


def _summary_path(manifest: Dict[str, Any], manifest_path: Path) -> Path:
    directory_value = manifest.get(
        "summary_directory", "train_dir/training_chains"
    )
    directory = Path(directory_value).expanduser()
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return directory / f"{manifest_path.stem}_{timestamp}.json"


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def launch_chain(args: LaunchTrainingChainArgs) -> int:
    manifest_path = args.config.expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    runs, continue_on_error = _resolve_runs(manifest, manifest_path)

    print(
        f"Validated {len(runs)} run(s); each is limited to "
        f"{runs[0].max_frames:,} aggregate frames."
    )
    for index, run in enumerate(runs, start=1):
        print(
            f"  {index}. {run.custom_experiment_name}: "
            f"preset={run.training_preset}, algorithm={run.algorithm}, "
            f"num_envs={run.num_envs}"
        )

    if args.dry_run:
        print("Dry run complete; no training was launched.")
        return 0

    summary_path = _summary_path(manifest, manifest_path)
    summary: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "started_at": datetime.now().astimezone().isoformat(),
        "absolute_max_frames_per_run": runs[0].max_frames,
        "continue_on_error": continue_on_error,
        "runs": [],
    }
    _write_summary(summary_path, summary)
    print(f"Chain summary: {summary_path}")

    for index, run in enumerate(runs, start=1):
        result: Dict[str, Any] = {
            "index": index,
            "name": run.custom_experiment_name,
            "settings": asdict(run),
            "started_at": datetime.now().astimezone().isoformat(),
            "status": "running",
        }
        summary["runs"].append(result)
        _write_summary(summary_path, summary)
        started = time.monotonic()

        print(
            f"\n=== Starting run {index}/{len(runs)}: "
            f"{run.custom_experiment_name} ==="
        )
        try:
            launch_training(run)
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            result["exit_code"] = 130
            summary["status"] = "interrupted"
            print("Training chain interrupted; the next run will not start.")
        except subprocess.CalledProcessError as error:
            result["status"] = "failed"
            result["exit_code"] = error.returncode
            print(
                f"Run {index} failed with exit code {error.returncode}: "
                f"{run.custom_experiment_name}"
            )
        else:
            result["status"] = "completed"
            result["exit_code"] = 0
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            result["duration_seconds"] = round(time.monotonic() - started, 3)
            _write_summary(summary_path, summary)

        if result["status"] == "interrupted":
            summary["finished_at"] = datetime.now().astimezone().isoformat()
            _write_summary(summary_path, summary)
            return 130
        if result["status"] == "failed" and not continue_on_error:
            summary["status"] = "failed"
            summary["finished_at"] = datetime.now().astimezone().isoformat()
            _write_summary(summary_path, summary)
            return int(result["exit_code"])

    failed_runs = [run for run in summary["runs"] if run["status"] == "failed"]
    summary["status"] = "completed_with_errors" if failed_runs else "completed"
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    _write_summary(summary_path, summary)
    print(
        f"Training chain finished with status {summary['status']}. "
        f"Summary: {summary_path}"
    )
    return 1 if failed_runs else 0


def main() -> None:
    args: LaunchTrainingChainArgs = tyro.cli(LaunchTrainingChainArgs)
    raise SystemExit(launch_chain(args))


if __name__ == "__main__":
    main()
