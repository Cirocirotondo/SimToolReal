"""Log and plot SimToolReal reward sub-terms during / after eval episodes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _configure_interactive_matplotlib() -> str:
    """Use a GUI backend for live plots; Agg only saves files, no window."""
    import matplotlib

    current = matplotlib.get_backend().lower()
    if current not in ("agg", "svg", "pdf", "ps", "cairo"):
        return matplotlib.get_backend()

    if not os.environ.get("DISPLAY"):
        print(
            "[reward-plot] No DISPLAY: live window disabled; "
            "PNG/NPZ will still be saved after the episode.",
            flush=True,
        )
        return matplotlib.get_backend()

    for backend in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt  # noqa: F401

            print(f"[reward-plot] Using interactive backend: {backend}", flush=True)
            return backend
        except Exception:
            continue

    print(
        "[reward-plot] Could not enable a GUI backend (still on Agg). "
        "Install tk: `sudo apt install python3-tk` or run with "
        "`MPLBACKEND=TkAgg`. Post-episode PNGs still work.",
        flush=True,
    )
    return matplotlib.get_backend()

# Main terms from env.py compute_reward (order preserved for stable legends).
STEP_REWARD_KEYS: Tuple[str, ...] = (
    "fingertip_delta_rew",
    "hand_delta_penalty",
    "lifting_rew",
    "lift_bonus_rew",
    "keypoint_rew",
    "kuka_actions_penalty",
    "hand_actions_penalty",
    "bonus_rew",
    "object_lin_vel_penalty",
    "object_ang_vel_penalty",
    "fingertip_spread_penalty",
    "fingertip_multi_contact_bonus",
    "fingertip_thumb_bonus",
    "total_reward",
)


def _is_raw_reward_term(key: str) -> bool:
    """Pre-scale logging only; excluded from eval plots."""
    return key.startswith("raw_")


def _scalar_from_extra(value: Any, env_idx: int = 0) -> float:
    if value is None:
        return float("nan")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        arr = value.numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        return float("nan")
    idx = min(env_idx, arr.size - 1)
    return float(arr[idx])


def _filter_plotted_reward_dict(data: Dict[str, float]) -> Dict[str, float]:
    """Drop pre-scale raw_* terms; eval plots use scaled terms only."""
    return {k: v for k, v in data.items() if not _is_raw_reward_term(k)}


def extract_reward_dicts(
    extras: Dict[str, Any], env_idx: int = 0
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-step terms (episode_cumulative) and episode sums (rewards_episode)."""
    per_step: Dict[str, float] = {}
    cumulative: Dict[str, float] = {}
    if "episode_cumulative" in extras:
        for key, val in extras["episode_cumulative"].items():
            per_step[str(key)] = _scalar_from_extra(val, env_idx)
    if "rewards_episode" in extras:
        for key, val in extras["rewards_episode"].items():
            cumulative[str(key)] = _scalar_from_extra(val, env_idx)
    return _filter_plotted_reward_dict(per_step), _filter_plotted_reward_dict(
        cumulative
    )


class RewardEpisodePlotter:
    """Live matplotlib window during episode + PNG/CSV after."""

    def __init__(
        self,
        save_dir: Path,
        *,
        live: bool = False,
        live_every: int = 5,
        env_idx: int = 0,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.live = live
        self.live_every = max(1, int(live_every))
        self.env_idx = env_idx

        self._slug = "episode"
        self._steps: List[int] = []
        self._per_step: Dict[str, List[float]] = {}
        self._cumulative: Dict[str, List[float]] = {}

        self._fig = None
        self._axes = None
        self._live_enabled = False

    def start_episode(self, slug: str) -> None:
        self._slug = slug
        self._steps.clear()
        self._per_step.clear()
        self._cumulative.clear()
        if self.live:
            self._init_live_figure()

    def record(self, extras: Dict[str, Any], step: int) -> None:
        per_step, cumulative = extract_reward_dicts(extras, self.env_idx)
        self._steps.append(step)
        for key, val in per_step.items():
            self._per_step.setdefault(key, []).append(val)
        for key, val in cumulative.items():
            self._cumulative.setdefault(key, []).append(val)

        if self._live_enabled and step % self.live_every == 0:
            self._update_live()

    def finalize(self, reason: str = "done") -> Dict[str, str]:
        paths: Dict[str, str] = {}
        if not self._steps:
            self._close_live()
            return paths

        # Fixed folder per task slug; files are overwritten each run.
        episode_dir = self.save_dir / self._slug
        episode_dir.mkdir(parents=True, exist_ok=True)
        per_term_dir = episode_dir / "per_term"
        per_term_dir.mkdir(parents=True, exist_ok=True)
        for old_png in per_term_dir.glob("*.png"):
            old_png.unlink()

        paths["episode_dir"] = str(episode_dir)
        paths["per_term_dir"] = str(per_term_dir)
        paths.update(self._save_csv(episode_dir))
        paths.update(self._save_combined_plots(episode_dir))
        paths.update(self._save_per_term_plots(per_term_dir))

        print(
            f"[reward-plot] Episode folder: {episode_dir} "
            f"({len(list(per_term_dir.glob('*.png')))} per-term plots)",
            flush=True,
        )

        self._close_live()
        return paths

    def _init_live_figure(self) -> None:
        try:
            _configure_interactive_matplotlib()
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "[reward-plot] matplotlib not installed; "
                "post-episode plots only if numpy data was recorded.",
                flush=True,
            )
            self.live = False
            return

        if plt.get_backend().lower() in ("agg", "svg", "pdf", "ps"):
            print(
                "[reward-plot] Non-interactive backend "
                f"({plt.get_backend()}): skipping live window.",
                flush=True,
            )
            self.live = False
            return

        plt.ion()
        self._fig, self._axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        self._fig.suptitle(f"Reward terms — {self._slug} (live)")
        self._axes[0].set_ylabel("Per-step")
        self._axes[1].set_ylabel("Episode cumulative")
        self._axes[1].set_xlabel("Step")
        self._fig.tight_layout()
        self._live_enabled = True

    def _ordered_keys(
        self, data: Dict[str, List[float]], preferred: Sequence[str] = STEP_REWARD_KEYS
    ) -> List[str]:
        """Only scaled terms from STEP_REWARD_KEYS; never raw_* or unknown keys."""
        return [k for k in preferred if k in data and not _is_raw_reward_term(k)]

    def _update_live(self) -> None:
        if not self._live_enabled or self._fig is None:
            return
        import matplotlib.pyplot as plt

        steps = np.asarray(self._steps, dtype=np.int32)
        for ax in self._axes:
            ax.cla()
        self._axes[0].set_ylabel("Per-step")
        self._axes[1].set_ylabel("Episode cumulative")
        self._axes[1].set_xlabel("Step")

        for key in self._ordered_keys(self._per_step, STEP_REWARD_KEYS):
            y = np.asarray(self._per_step[key], dtype=np.float64)
            if y.size == steps.size:
                self._axes[0].plot(steps, y, label=key, linewidth=1.2)
        for key in self._ordered_keys(self._cumulative, STEP_REWARD_KEYS):
            y = np.asarray(self._cumulative[key], dtype=np.float64)
            if y.size == steps.size:
                self._axes[1].plot(steps, y, label=key, linewidth=1.2)

        for ax in self._axes:
            ax.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def _save_csv(self, base: Path) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        steps = np.asarray(self._steps, dtype=np.int32)
        out = base / "reward_log.npz"
        payload = {"steps": steps}
        for key, series in self._per_step.items():
            if not _is_raw_reward_term(key):
                payload[f"step__{key}"] = np.asarray(series, dtype=np.float64)
        for key, series in self._cumulative.items():
            if not _is_raw_reward_term(key):
                payload[f"cumul__{key}"] = np.asarray(series, dtype=np.float64)
        np.savez_compressed(out, **payload)
        paths["npz"] = str(out)
        return paths

    def _save_combined_plots(self, episode_dir: Path) -> Dict[str, str]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[reward-plot] matplotlib missing; saved .npz only.")
            return {}

        steps = np.asarray(self._steps, dtype=np.int32)
        paths: Dict[str, str] = {}

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        for key in self._ordered_keys(self._per_step, STEP_REWARD_KEYS):
            y = np.asarray(self._per_step[key], dtype=np.float64)
            if y.size == steps.size:
                ax1.plot(steps, y, label=key, linewidth=1.5)
        ax1.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
        ax1.set_title(f"Per-step reward terms — {self._slug}")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Reward contribution")
        ax1.legend(loc="upper right", fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        p1 = episode_dir / "per_step_all.png"
        fig1.savefig(p1, dpi=150)
        plt.close(fig1)
        paths["per_step_all_png"] = str(p1)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for key in self._ordered_keys(self._cumulative, STEP_REWARD_KEYS):
            y = np.asarray(self._cumulative[key], dtype=np.float64)
            if y.size == steps.size:
                ax2.plot(steps, y, label=key, linewidth=1.5)
        ax2.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
        ax2.set_title(f"Episode cumulative reward terms — {self._slug}")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Cumulative sum")
        ax2.legend(loc="upper right", fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        p2 = episode_dir / "cumulative_all.png"
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        paths["cumulative_all_png"] = str(p2)

        # Stacked overview: scaled per-step terms (top) + total only (bottom)
        fig3, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        for key in self._ordered_keys(self._per_step):
            if key == "total_reward":
                continue
            y = np.asarray(self._per_step[key], dtype=np.float64)
            if y.size == steps.size:
                axes[0].plot(steps, y, label=key, linewidth=1.2)
        if "total_reward" in self._per_step:
            y = np.asarray(self._per_step["total_reward"], dtype=np.float64)
            if y.size == steps.size:
                axes[1].plot(steps, y, label="total_reward", color="black", linewidth=2)
        if "total_reward" in self._cumulative:
            y = np.asarray(self._cumulative["total_reward"], dtype=np.float64)
            if y.size == steps.size:
                axes[1].plot(
                    steps,
                    y,
                    label="cumul_total_reward",
                    linestyle="--",
                    color="C1",
                    linewidth=1.2,
                )
        for ax in axes:
            ax.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Per-step (excl. total)")
        axes[1].set_ylabel("Total / cumulative")
        axes[1].set_xlabel("Step")
        fig3.suptitle(f"Reward overview — {self._slug}")
        fig3.tight_layout()
        p3 = episode_dir / "overview.png"
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)
        paths["overview_png"] = str(p3)

        return paths

    def _save_per_term_plots(self, per_term_dir: Path) -> Dict[str, str]:
        """One PNG per reward term (per-step + cumulative) under per_term/."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return {}

        steps = np.asarray(self._steps, dtype=np.int32)
        paths: Dict[str, str] = {}

        merged = {**self._per_step, **self._cumulative}
        all_keys = self._ordered_keys(merged)

        for key in all_keys:
            has_step = key in self._per_step
            has_cumul = key in self._cumulative
            if not has_step and not has_cumul:
                continue

            nrows = (1 if has_step else 0) + (1 if has_cumul else 0)
            fig, axes = plt.subplots(
                nrows, 1, figsize=(10, 3.5 * nrows), sharex=True, squeeze=False
            )
            axes_flat = axes.flatten()
            row = 0

            if has_step:
                y = np.asarray(self._per_step[key], dtype=np.float64)
                if y.size == steps.size:
                    axes_flat[row].plot(steps, y, color="C0", linewidth=1.5)
                axes_flat[row].axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
                axes_flat[row].set_ylabel("Per-step")
                axes_flat[row].set_title(f"{key} — per-step")
                axes_flat[row].grid(True, alpha=0.3)
                row += 1

            if has_cumul:
                y = np.asarray(self._cumulative[key], dtype=np.float64)
                if y.size == steps.size:
                    axes_flat[row].plot(steps, y, color="C1", linewidth=1.5)
                axes_flat[row].axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
                axes_flat[row].set_ylabel("Cumulative")
                axes_flat[row].set_title(f"{key} — episode cumulative")
                axes_flat[row].grid(True, alpha=0.3)

            axes_flat[-1].set_xlabel("Step")
            fig.suptitle(f"{self._slug} — {key}", fontsize=11)
            fig.tight_layout()

            safe_name = key.replace("/", "_").replace(" ", "_")
            out = per_term_dir / f"{safe_name}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            paths[f"per_term/{safe_name}"] = str(out)

        return paths

    def _close_live(self) -> None:
        if not self._live_enabled:
            return
        try:
            import matplotlib.pyplot as plt

            if self._fig is not None:
                plt.close(self._fig)
            plt.ioff()
        except Exception:
            pass
        self._fig = None
        self._axes = None
        self._live_enabled = False
