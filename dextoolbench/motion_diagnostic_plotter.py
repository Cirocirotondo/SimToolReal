"""Record and plot motion signals from an imitation-learning evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _env_vector(value: Any, env_idx: int = 0) -> np.ndarray:
    array = _numpy(value)
    if array.ndim > 1:
        array = array[env_idx]
    return np.asarray(array, dtype=np.float64).copy()


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    result = np.zeros_like(values)
    if values.shape[0] > 1:
        result[1:] = np.diff(values, axis=0) / dt
    return result


def _difference(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    if values.shape[0] > 1:
        result[1:] = np.diff(values, axis=0)
    return result


def _quaternion_angular_velocity_xyzw(quaternions: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference angular velocity from an XYZW quaternion series."""
    angular_velocity = np.zeros((quaternions.shape[0], 3), dtype=np.float64)
    if quaternions.shape[0] < 2:
        return angular_velocity

    previous_conjugate = quaternions[:-1].copy()
    previous_conjugate[:, :3] *= -1.0
    current = quaternions[1:]
    current_xyz = current[:, :3]
    previous_xyz = previous_conjugate[:, :3]
    current_w = current[:, 3:4]
    previous_w = previous_conjugate[:, 3:4]
    delta_xyz = (
        current_w * previous_xyz
        + previous_w * current_xyz
        + np.cross(current_xyz, previous_xyz)
    )
    delta_w = current_w[:, 0] * previous_w[:, 0] - np.sum(
        current_xyz * previous_xyz, axis=1
    )
    negative = delta_w < 0.0
    delta_xyz[negative] *= -1.0
    delta_w[negative] *= -1.0
    vector_norm = np.linalg.norm(delta_xyz, axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta_w, 0.0, None))
    rotation_vector = np.zeros_like(delta_xyz)
    nonzero = vector_norm > 1e-8
    rotation_vector[nonzero] = (
        delta_xyz[nonzero] * (angle[nonzero] / vector_norm[nonzero])[:, None]
    )
    rotation_vector[~nonzero] = 2.0 * delta_xyz[~nonzero]
    angular_velocity[1:] = rotation_vector / dt
    return angular_velocity


def _quaternion_error_rad(
    actual_xyzw: np.ndarray, reference_xyzw: np.ndarray
) -> np.ndarray:
    dot = np.abs(np.sum(actual_xyzw * reference_xyzw, axis=1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1)


class MotionDiagnosticPlotter:
    """Save synchronized target, action, and measured-joint diagnostics."""

    def __init__(self, save_dir: Path, *, env_idx: int = 0) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.env_idx = int(env_idx)
        self._slug = "episode"
        self._control_dt = 1.0
        self._num_arm_dofs = 6
        self._records: Dict[str, List[Any]] = {}

    def start_episode(self, slug: str, env) -> None:
        self._slug = slug
        self._control_dt = float(env.control_dt)
        self._num_arm_dofs = int(env.num_arm_dofs)
        self._records.clear()

    def _append(self, key: str, value: Any) -> None:
        self._records.setdefault(key, []).append(value)

    def record(
        self,
        env,
        step: int,
        raw_policy_action: Any,
        submitted_action: Any,
    ) -> None:
        """Record the post-step state and the action responsible for it."""
        env_idx = self.env_idx
        reference = env.current_reference
        palm_linear_velocity, palm_angular_velocity = env._palm_center_velocity()

        self._append("steps", int(step))
        self._append("time_s", float(env.phase[env_idx]) * env.reference.duration_s)
        self._append("phase", float(env.phase[env_idx]))
        self._append("raw_policy_action", _env_vector(raw_policy_action, env_idx))
        self._append("submitted_action", _env_vector(submitted_action, env_idx))
        self._append("effective_action", _env_vector(env.actions, env_idx))
        self._append("effective_action_delta", _env_vector(env.action_deltas, env_idx))

        self._append("actual_joint_q", _env_vector(env.arm_hand_dof_pos, env_idx))
        self._append("actual_joint_qd", _env_vector(env.arm_hand_dof_vel, env_idx))
        self._append("reference_arm_q", _env_vector(reference.arm_q, env_idx))
        self._append("reference_arm_qd", _env_vector(reference.arm_dq, env_idx))
        self._append("reference_hand_q", _env_vector(reference.hand_q, env_idx))
        self._append("reference_hand_qd", _env_vector(reference.hand_dq, env_idx))

        self._append("actual_palm_pos", _env_vector(env.palm_center_pos, env_idx))
        self._append("actual_palm_quat_xyzw", _env_vector(env._palm_rot, env_idx))
        self._append("actual_palm_lin_vel", _env_vector(palm_linear_velocity, env_idx))
        self._append("actual_palm_ang_vel", _env_vector(palm_angular_velocity, env_idx))
        self._append("reference_palm_pos", _env_vector(reference.palm_pos, env_idx))
        self._append(
            "reference_palm_quat_xyzw",
            _env_vector(reference.palm_quat_xyzw, env_idx),
        )
        self._append(
            "reference_palm_lin_vel",
            _env_vector(reference.palm_lin_vel, env_idx),
        )
        self._append(
            "reference_palm_ang_vel",
            _env_vector(reference.palm_ang_vel, env_idx),
        )

    def _arrays(self) -> Dict[str, np.ndarray]:
        arrays = {key: np.asarray(values) for key, values in self._records.items()}
        if not arrays:
            return arrays
        dt = self._control_dt
        arrays["raw_policy_action_delta"] = _difference(arrays["raw_policy_action"])
        arrays["submitted_action_delta"] = _difference(arrays["submitted_action"])
        arrays["actual_joint_qdd"] = _finite_difference(arrays["actual_joint_qd"], dt)
        arrays["reference_palm_lin_vel_discrete"] = _finite_difference(
            arrays["reference_palm_pos"], dt
        )
        arrays["reference_palm_ang_vel_discrete"] = _quaternion_angular_velocity_xyzw(
            arrays["reference_palm_quat_xyzw"], dt
        )
        arrays["reference_arm_qd_discrete"] = _finite_difference(
            arrays["reference_arm_q"], dt
        )
        arrays["reference_hand_qd_discrete"] = _finite_difference(
            arrays["reference_hand_q"], dt
        )
        arrays["palm_position_error_m"] = _norm(
            arrays["actual_palm_pos"] - arrays["reference_palm_pos"]
        )
        arrays["palm_orientation_error_rad"] = _quaternion_error_rad(
            arrays["actual_palm_quat_xyzw"],
            arrays["reference_palm_quat_xyzw"],
        )
        return arrays

    def finalize(self, reason: str = "done") -> Dict[str, str]:
        arrays = self._arrays()
        if not arrays:
            return {}
        episode_dir = self.save_dir / self._slug
        episode_dir.mkdir(parents=True, exist_ok=True)
        npz_path = episode_dir / "motion_diagnostics.npz"
        np.savez_compressed(
            npz_path,
            **arrays,
            control_dt=np.asarray(self._control_dt),
            termination_reason=np.asarray(reason),
        )
        paths = {
            "diagnostic_episode_dir": str(episode_dir),
            "diagnostic_npz": str(npz_path),
        }
        paths.update(self._save_plots(episode_dir, arrays))
        print(f"[motion-diagnostics] Episode folder: {episode_dir}", flush=True)
        return paths

    @staticmethod
    def _plot_components(
        ax, time_s, values, label_prefix, component_names=None, **kwargs
    ) -> None:
        for index in range(values.shape[1]):
            suffix = (
                component_names[index]
                if component_names is not None and index < len(component_names)
                else str(index)
            )
            ax.plot(
                time_s,
                values[:, index],
                label=f"{label_prefix}_{suffix}",
                **kwargs,
            )

    @staticmethod
    def _finish_axis(ax, ylabel: str) -> None:
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2)

    def _save_plots(
        self, episode_dir: Path, data: Dict[str, np.ndarray]
    ) -> Dict[str, str]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[motion-diagnostics] matplotlib missing; saved .npz only.")
            return {}

        time_s = data["time_s"]
        split = self._num_arm_dofs
        paths: Dict[str, str] = {}

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
        axes[0].plot(
            time_s,
            _norm(data["reference_palm_lin_vel_discrete"]),
            label="target palm linear (pose finite diff)",
        )
        axes[0].plot(
            time_s,
            _norm(data["reference_palm_lin_vel"]),
            label="target palm linear (loader filtered)",
        )
        self._finish_axis(axes[0], "m/s")
        axes[1].plot(
            time_s,
            _norm(data["reference_palm_ang_vel_discrete"]),
            label="target palm angular (pose finite diff)",
        )
        axes[1].plot(
            time_s,
            _norm(data["reference_palm_ang_vel"]),
            label="target palm angular (loader filtered)",
        )
        self._finish_axis(axes[1], "rad/s")
        for key, label, style in (
            ("raw_policy_action_delta", "raw policy", "-"),
            ("submitted_action_delta", "submitted/filtered", "--"),
            ("effective_action_delta", "effective post-delay", ":"),
        ):
            axes[2].plot(
                time_s,
                _norm(data[key][:, :split]),
                linestyle=style,
                label=f"arm {label}",
            )
            axes[2].plot(
                time_s,
                _norm(data[key][:, split:]),
                linestyle=style,
                alpha=0.8,
                label=f"hand {label}",
            )
        self._finish_axis(axes[2], "action delta")
        axes[3].plot(
            time_s,
            _norm(data["actual_joint_qdd"][:, :split]),
            label="arm measured qdd",
        )
        axes[3].plot(
            time_s,
            _norm(data["actual_joint_qdd"][:, split:]),
            label="hand measured qdd",
        )
        self._finish_axis(axes[3], "rad/s²")
        axes[3].set_xlabel("Demonstration time [s]")
        fig.suptitle("Vibration diagnosis overview")
        fig.tight_layout()
        path = episode_dir / "diagnostic_overview.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["diagnostic_overview_png"] = str(path)

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
        self._plot_components(
            axes[0],
            time_s,
            data["reference_palm_pos"],
            "target",
            component_names=("x", "y", "z"),
            linewidth=1.4,
        )
        self._plot_components(
            axes[0],
            time_s,
            data["actual_palm_pos"],
            "actual",
            component_names=("x", "y", "z"),
            linestyle="--",
        )
        self._finish_axis(axes[0], "Palm position [m]")
        axes[1].plot(
            time_s,
            100.0 * data["palm_position_error_m"],
            label="position error [cm]",
        )
        axes[1].plot(
            time_s,
            np.degrees(data["palm_orientation_error_rad"]),
            label="orientation error [degree]",
        )
        self._finish_axis(axes[1], "cm / degree")
        self._plot_components(
            axes[2],
            time_s,
            data["reference_palm_lin_vel"],
            "target",
            component_names=("x", "y", "z"),
            linewidth=1.4,
        )
        self._plot_components(
            axes[2],
            time_s,
            data["actual_palm_lin_vel"],
            "actual",
            component_names=("x", "y", "z"),
            linestyle="--",
        )
        self._finish_axis(axes[2], "Linear velocity [m/s]")
        self._plot_components(
            axes[3],
            time_s,
            data["reference_palm_ang_vel"],
            "target",
            component_names=("x", "y", "z"),
            linewidth=1.4,
        )
        self._plot_components(
            axes[3],
            time_s,
            data["actual_palm_ang_vel"],
            "actual",
            component_names=("x", "y", "z"),
            linestyle="--",
        )
        self._finish_axis(axes[3], "Angular velocity [rad/s]")
        axes[3].set_xlabel("Demonstration time [s]")
        fig.suptitle("Palm target versus simulated motion")
        fig.tight_layout()
        path = episode_dir / "palm_target_vs_actual.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["diagnostic_palm_png"] = str(path)

        paths.update(
            self._save_joint_plot(
                episode_dir,
                time_s,
                data["reference_arm_q"],
                data["reference_arm_qd"],
                data["reference_arm_qd_discrete"],
                data["actual_joint_q"][:, :split],
                data["actual_joint_qd"][:, :split],
                data["actual_joint_qdd"][:, :split],
                "arm",
            )
        )
        paths.update(
            self._save_joint_plot(
                episode_dir,
                time_s,
                data["reference_hand_q"],
                data["reference_hand_qd"],
                data["reference_hand_qd_discrete"],
                data["actual_joint_q"][:, split:],
                data["actual_joint_qd"][:, split:],
                data["actual_joint_qdd"][:, split:],
                "hand",
            )
        )

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
        self._plot_components(
            axes[0], time_s, data["raw_policy_action"][:, :split], "raw"
        )
        self._plot_components(
            axes[0],
            time_s,
            data["effective_action"][:, :split],
            "effective",
            linestyle="--",
        )
        self._finish_axis(axes[0], "Arm action")
        self._plot_components(
            axes[1], time_s, data["effective_action_delta"][:, :split], "arm"
        )
        self._finish_axis(axes[1], "Arm action delta")
        self._plot_components(
            axes[2], time_s, data["raw_policy_action"][:, split:], "raw"
        )
        self._plot_components(
            axes[2],
            time_s,
            data["effective_action"][:, split:],
            "effective",
            linestyle="--",
        )
        self._finish_axis(axes[2], "Hand action")
        self._plot_components(
            axes[3], time_s, data["effective_action_delta"][:, split:], "hand"
        )
        self._finish_axis(axes[3], "Hand action delta")
        axes[3].set_xlabel("Demonstration time [s]")
        fig.suptitle("Policy and effective controller actions")
        fig.tight_layout()
        path = episode_dir / "action_diagnostics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["diagnostic_actions_png"] = str(path)
        return paths

    def _save_joint_plot(
        self,
        episode_dir: Path,
        time_s: np.ndarray,
        reference_q: np.ndarray,
        reference_qd: np.ndarray,
        reference_qd_discrete: np.ndarray,
        actual_q: np.ndarray,
        actual_qd: np.ndarray,
        actual_qdd: np.ndarray,
        group: str,
    ) -> Dict[str, str]:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
        self._plot_components(axes[0], time_s, reference_q, "target")
        self._plot_components(axes[0], time_s, actual_q, "actual", linestyle="--")
        self._finish_axis(axes[0], "Joint position [rad]")
        self._plot_components(axes[1], time_s, reference_qd_discrete, "target_raw")
        self._plot_components(
            axes[1], time_s, reference_qd, "target_filtered", linestyle="--"
        )
        self._finish_axis(axes[1], "Target velocity [rad/s]")
        self._plot_components(axes[2], time_s, actual_qd, "actual")
        self._finish_axis(axes[2], "Measured velocity [rad/s]")
        self._plot_components(axes[3], time_s, actual_qdd, "actual")
        self._finish_axis(axes[3], "Measured acceleration [rad/s²]")
        axes[3].set_xlabel("Demonstration time [s]")
        fig.suptitle(f"{group.capitalize()} joint diagnostics")
        fig.tight_layout()
        path = episode_dir / f"{group}_joint_diagnostics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return {f"diagnostic_{group}_joints_png": str(path)}
