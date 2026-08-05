"""Interactive evaluation for SimToolReal motion-imitation policies.

The Viser UI runs in the main process. Isaac Gym and the policy run in a fresh
subprocess so an evaluation environment can be restarted safely.
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np
import yaml

from dextoolbench.interactive_eval_common import (
    checkpoint_payload,
    install_path_is_relative_to_backport,
    quat_xyzw_to_wxyz,
)
from dextoolbench.eval_env_config import (
    motion_imitation_robot_urdf_path_for_hand,
    policy_config_hand_side,
)
from dextoolbench.motion_diagnostic_plotter import MotionDiagnosticPlotter
from dextoolbench.reward_episode_plotter import RewardEpisodePlotter

install_path_is_relative_to_backport()

import viser  # noqa: E402
from viser.extras import ViserUrdf  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_SIZE = (0.475, 0.4, 0.3)


class MovingAverageActionFilter:
    """Causal finite-window mean for batched policy actions."""

    def __init__(self, window_size: int) -> None:
        if window_size < 2:
            raise ValueError(
                f"Action moving-average window must be at least 2, got {window_size}"
            )
        self.window_size = int(window_size)
        self._history: Deque[Any] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._history.clear()

    def __call__(self, action):
        import torch

        self._history.append(action.detach().clone())
        return torch.stack(tuple(self._history), dim=0).mean(dim=0)


def _read_env_config(config_path: str) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    task = config.get("task", {})
    return task.get("env", config.get("env", {}))


def _uses_sapg_exploration_observation(config_path: str) -> bool:
    """Return whether the saved policy expects the SAPG population identifier."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    expl_type = (
        config.get("train", {})
        .get("params", {})
        .get("config", {})
        .get("expl_type", "none")
    )
    return str(expl_type).startswith("mixed_expl")


def _checkpoint_env_state(checkpoint) -> Optional[Dict[str, Any]]:
    payload = checkpoint_payload(checkpoint)
    env_state = payload.get("env_state")
    return env_state if isinstance(env_state, dict) else None


def _scalar(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _component(extras: Dict[str, Any], name: str) -> float:
    values = extras.get("episode_cumulative", {}).get(name)
    if values is None:
        return 0.0
    return _scalar(values[0])


def _sim_state(env, extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    extras = extras or {}
    actual_q = env.arm_hand_dof_pos[0, : env.num_hand_arm_dofs]
    reference = env.current_reference
    reference_q = np.concatenate(
        (
            reference.arm_q[0].detach().cpu().numpy(),
            reference.hand_q[0].detach().cpu().numpy(),
        )
    )
    phase = float(env.phase[0].item())
    velocity_tracking_enabled = bool(
        getattr(env, "velocity_tracking_enabled", False)
    )
    object_tracking_enabled = bool(
        getattr(env, "object_tracking_enabled", False)
    )
    actual_object_pose = None
    reference_object_pose = None
    if object_tracking_enabled:
        if reference.object_pos is None or reference.object_quat_xyzw is None:
            raise RuntimeError("Object-aware evaluation requires a reference pose")
        actual_object_pose = np.concatenate(
            (
                env.object_pos[0].detach().cpu().numpy(),
                env.object_rot[0].detach().cpu().numpy(),
            )
        )
        reference_object_pose = np.concatenate(
            (
                reference.object_pos[0].detach().cpu().numpy(),
                reference.object_quat_xyzw[0].detach().cpu().numpy(),
            )
        )
    metrics = {
        "position_error_m": _scalar(
            extras.get("imitation/position_error_m", 0.0)
        ),
        "rotation_error_rad": _scalar(
            extras.get("imitation/rotation_error_rad", 0.0)
        ),
        "hand_error_rad": _scalar(
            extras.get("imitation/hand_error_rad", 0.0)
        ),
        "linear_velocity_error_mps": _scalar(
            extras.get("imitation/linear_velocity_error_mps", 0.0)
        ),
        "angular_velocity_error_radps": _scalar(
            extras.get("imitation/angular_velocity_error_radps", 0.0)
        ),
        "hand_velocity_error_radps": _scalar(
            extras.get("imitation/hand_velocity_error_radps", 0.0)
        ),
        "ee_position_reward": _component(extras, "ee_position_reward"),
        "ee_rotation_reward": _component(extras, "ee_rotation_reward"),
        "hand_pose_reward": _component(extras, "hand_pose_reward"),
        "pose_imitation_reward": _component(extras, "pose_imitation_reward"),
        "palm_linear_velocity_reward": _component(
            extras, "palm_linear_velocity_reward"
        ),
        "palm_angular_velocity_reward": _component(
            extras, "palm_angular_velocity_reward"
        ),
        "hand_velocity_reward": _component(extras, "hand_velocity_reward"),
        "velocity_imitation_reward": _component(
            extras, "velocity_imitation_reward"
        ),
        "imitation_reward": _component(extras, "imitation_reward"),
        "object_keypoint_reward": _component(
            extras, "object_keypoint_reward"
        ),
        "object_keypoint_mean_error_m": _scalar(
            extras.get("object_tracking/keypoint_mean_error_m", 0.0)
        ),
        "object_keypoint_max_error_m": _scalar(
            extras.get("object_tracking/keypoint_max_error_m", 0.0)
        ),
        "object_position_error_m": _scalar(
            extras.get("object_tracking/position_error_m", 0.0)
        ),
        "action_penalty": sum(
            _component(extras, key)
            for key in (
                "kuka_actions_penalty",
                "hand_actions_penalty",
                "arm_action_delta_penalty",
                "hand_action_delta_penalty",
                "arm_joint_velocity_penalty",
                "arm_joint_acceleration_penalty",
                "hand_joint_acceleration_penalty",
            )
        ),
        "total_reward": _component(extras, "total_reward"),
    }
    return {
        "actual_q": actual_q.detach().cpu().numpy(),
        "reference_q": reference_q,
        "phase": phase,
        "time_s": phase * env.reference.duration_s,
        "duration_s": env.reference.duration_s,
        "velocity_tracking_enabled": velocity_tracking_enabled,
        "object_tracking_enabled": object_tracking_enabled,
        "actual_object_pose": actual_object_pose,
        "reference_object_pose": reference_object_pose,
        "metrics": metrics,
    }


def _reset_at_phase(env, phase: float):
    import torch

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    env.reset_idx(env_ids, tensor_reset=True)
    env.set_reference_phase(env_ids, float(phase), flush=True)
    env.set_actor_root_state_tensor_indexed()
    return env.obs_buf.to(env.rl_device)


def _termination_reason(env, state: Dict[str, Any]) -> str:
    if state["phase"] >= 1.0:
        return "completed"
    metrics = state["metrics"]
    reasons = []
    if metrics["position_error_m"] > env.ee_position_termination_distance:
        reasons.append("EE position")
    if metrics["rotation_error_rad"] > env.ee_rotation_termination_angle:
        reasons.append("EE orientation")
    if metrics["hand_error_rad"] > env.hand_termination_error:
        reasons.append("hand pose")
    if (
        getattr(env, "object_early_termination_enabled", False)
        and metrics["object_position_error_m"]
        > env.object_position_termination_distance
    ):
        reasons.append("object position")
    return "early termination: " + ", ".join(reasons or ["unknown"])


def _run_episode(
    conn,
    env,
    policy,
    *,
    initial_phase: float,
    realtime: bool,
    plot_rewards: bool,
    reward_plot_dir: Optional[Path],
    action_moving_average: bool,
    action_moving_average_window: int,
    plot_motion_diagnostics: bool,
    diagnostic_plot_dir: Optional[Path],
) -> bool:
    """Run one rollout; return True when the worker should exit."""
    control_dt = float(env.control_dt)
    policy.reset()
    obs = _reset_at_phase(env, initial_phase)
    conn.send(("state", _sim_state(env), 0, 0.0))
    action_filter = (
        MovingAverageActionFilter(action_moving_average_window)
        if action_moving_average
        else None
    )

    plotter = None
    if plot_rewards and reward_plot_dir is not None:
        plotter = RewardEpisodePlotter(reward_plot_dir, live=False)
        plotter.start_episode(f"imitation_phase_{initial_phase:.3f}")
    diagnostic_plotter = None
    if plot_motion_diagnostics and diagnostic_plot_dir is not None:
        diagnostic_plotter = MotionDiagnosticPlotter(diagnostic_plot_dir)
        action_mode = (
            f"moving_average_w{action_moving_average_window}"
            if action_moving_average
            else "raw_actions"
        )
        diagnostic_plotter.start_episode(
            f"imitation_phase_{initial_phase:.3f}_{action_mode}", env
        )

    def finalize_plotters(reason: str) -> Dict[str, str]:
        paths = plotter.finalize(reason) if plotter else {}
        if diagnostic_plotter:
            paths.update(diagnostic_plotter.finalize(reason))
        return paths

    step = 0
    total_reward = 0.0
    paused = False
    while True:
        while conn.poll(0):
            command = conn.recv()
            if command == "pause":
                paused = True
            elif command == "resume":
                paused = False
            elif command == "quit":
                finalize_plotters("quit")
                return True
            elif command == "stop":
                paths = finalize_plotters("stopped")
                conn.send(("stopped", paths))
                return False

        if paused:
            time.sleep(0.05)
            continue

        started = time.time()
        raw_action = policy.get_normalized_action(obs, deterministic_actions=True)
        action = raw_action
        if action_filter is not None:
            action = action_filter(action)
        obs_dict, reward, done, extras = env.step(action)
        obs = obs_dict["obs"]
        step += 1
        total_reward += float(reward[0].item())
        if plotter is not None:
            plotter.record(extras, step)
        if diagnostic_plotter is not None:
            diagnostic_plotter.record(env, step, raw_action, action)

        state = _sim_state(env, extras)
        conn.send(("state", state, step, total_reward))
        if bool(done[0].item()):
            reason = _termination_reason(env, state)
            paths = finalize_plotters(reason)
            conn.send(("done", reason, step, total_reward, paths))
            return False

        if realtime:
            remaining = control_dt - (time.time() - started)
            if remaining > 0.0:
                time.sleep(remaining)


def sim_worker(
    conn,
    config_path: str,
    checkpoint_path: str,
    use_cpu: bool,
    plot_rewards: bool,
    reward_plot_dir: Optional[str],
    action_moving_average: bool,
    action_moving_average_window: int,
    hand_termination_error: Optional[float],
    plot_motion_diagnostics: bool,
    diagnostic_plot_dir: Optional[str],
) -> None:
    """Create one imitation environment and serve evaluation commands."""
    try:
        from isaacgym import gymapi  # noqa: F401
        import torch

        from deployment.isaac.isaac_env import create_env
        from deployment.rl_player import RlPlayer
        from isaacgymenvs.tasks.simtoolreal.env_motion_imitation import (
            SimToolRealMotionImitation,
        )

        device = (
            "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        env_overrides = {
            "task.env.capture_video": False,
            "task.env.enableCameraSensors": False,
            "task.env.useReferenceStateInitialization": False,
            "task.env.referenceStateInitProbability": 0.0,
        }
        # The anchored-RSI option exists only in object/pre-grasp task
        # configurations. create_env() overrides may update existing keys but
        # cannot introduce a missing structured-config key, so add this
        # override only when the saved training configuration defines it.
        saved_env_cfg = _read_env_config(config_path)
        if "referenceInitAnchorProbability" in saved_env_cfg:
            env_overrides["task.env.referenceInitAnchorProbability"] = 0.0
        if hand_termination_error is not None:
            env_overrides["task.env.handTerminationError"] = (
                hand_termination_error
            )
        env = create_env(
            config_path=config_path,
            headless=True,
            device=device,
            overrides=env_overrides,
        )
        if not isinstance(env, SimToolRealMotionImitation):
            raise TypeError(
                "The supplied config does not create "
                f"SimToolRealMotionImitation (got {type(env).__name__})"
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        env_state = _checkpoint_env_state(checkpoint)
        if env_state is not None:
            env.set_env_state(env_state)

        policy = RlPlayer(
            int(env.num_obs),
            int(env.num_acts),
            config_path,
            checkpoint_path,
            device,
            env.num_envs,
            append_exploration_observation=(
                _uses_sapg_exploration_observation(config_path)
            ),
        )
        obs = _reset_at_phase(env, 0.0)
        del obs
        conn.send(("ready", _sim_state(env)))

        while True:
            command = conn.recv()
            if command == "quit":
                return
            if isinstance(command, tuple) and command[0] == "run":
                options = command[1]
                phase = options.get("phase")
                if phase is None:
                    phase = float(
                        np.random.uniform(0.0, env.reference_init_max_phase)
                    )
                should_quit = _run_episode(
                    conn,
                    env,
                    policy,
                    initial_phase=float(phase),
                    realtime=bool(options.get("realtime", True)),
                    plot_rewards=plot_rewards,
                    reward_plot_dir=(
                        Path(reward_plot_dir) if reward_plot_dir else None
                    ),
                    action_moving_average=action_moving_average,
                    action_moving_average_window=action_moving_average_window,
                    plot_motion_diagnostics=plot_motion_diagnostics,
                    diagnostic_plot_dir=(
                        Path(diagnostic_plot_dir) if diagnostic_plot_dir else None
                    ),
                )
                if should_quit:
                    return
    except ImportError as exc:
        conn.send(("error", f"Missing evaluation dependency: {exc}"))
    except Exception as exc:
        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
    finally:
        conn.close()


class ImitationInteractiveDemo:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        *,
        port: int,
        use_cpu: bool,
        plot_rewards: bool,
        reward_plot_dir: Optional[str],
        action_moving_average: bool,
        action_moving_average_window: int,
        hand_termination_error: Optional[float],
        plot_motion_diagnostics: bool,
        diagnostic_plot_dir: Optional[str],
    ) -> None:
        self.config_path = str(Path(config_path).resolve())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.use_cpu = use_cpu
        self.plot_rewards = plot_rewards
        self.action_moving_average = action_moving_average
        self.action_moving_average_window = action_moving_average_window
        self.hand_termination_error = hand_termination_error
        if (
            self.hand_termination_error is not None
            and self.hand_termination_error <= 0.0
        ):
            raise ValueError("hand_termination_error must be positive")
        self.plot_motion_diagnostics = plot_motion_diagnostics
        self.reward_plot_dir = str(
            Path(reward_plot_dir).resolve()
            if reward_plot_dir
            else REPO_ROOT / "eval_reward_plots" / "motion_imitation"
        )
        self.diagnostic_plot_dir = str(
            Path(diagnostic_plot_dir).resolve()
            if diagnostic_plot_dir
            else REPO_ROOT / "eval_diagnostic_plots" / "motion_imitation"
        )
        if not Path(self.config_path).is_file():
            raise FileNotFoundError(self.config_path)
        if not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(self.checkpoint_path)

        env_cfg = _read_env_config(self.config_path)
        self.hand_side = policy_config_hand_side(self.config_path) or "right"
        self.demonstration = str(env_cfg.get("demonstration", "unknown"))
        self.object_tracking_enabled = bool(
            env_cfg.get("objectTrackingEnabled", False)
        )
        self.object_dimensions = None
        if self.object_tracking_enabled:
            dimensions = np.asarray(
                env_cfg.get("cuboidSize", env_cfg.get("fixedSize")),
                dtype=np.float64,
            )
            if dimensions.shape != (3,) or not np.all(dimensions > 0.0):
                raise ValueError(
                    "Object-aware evaluation requires positive cuboidSize or "
                    f"fixedSize dimensions, got {dimensions}"
                )
            self.object_dimensions = tuple(float(value) for value in dimensions)
        self.robot_base = (
            0.0,
            float(env_cfg.get("robotBaseY", 0.6)),
            0.0,
        )
        table_reset_z = env_cfg.get("tableResetZ")
        table_center_z = (
            float(table_reset_z)
            if table_reset_z is not None
            else float(env_cfg.get("tableSurfaceZ", -0.05))
            - TABLE_SIZE[2] / 2.0
        )
        self.table_center = (
            0.0,
            self.robot_base[1] + float(env_cfg.get("tablePoseDy", -0.6)),
            table_center_z,
        )

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.port = int(self.server.get_port())
        self._proc = None
        self._conn = None
        self._ready = False
        self._running = False
        self._paused = False
        self._episodes = 0
        self._completed = 0
        self._returns = []

        self._build_scene()
        self._build_gui()

    def _build_scene(self) -> None:
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (0.0, -1.1, 1.05)
            client.camera.look_at = (0.0, 0.1, 0.48)

        self.server.scene.add_grid(
            "/ground", width=2.0, height=2.0, cell_size=0.1, position=(0, 0, -0.2)
        )
        self.server.scene.add_box(
            "/table",
            dimensions=TABLE_SIZE,
            position=self.table_center,
            color=(180, 130, 70),
            opacity=0.85,
        )
        robot_urdf = motion_imitation_robot_urdf_path_for_hand(self.hand_side)
        self.server.scene.add_frame(
            "/actual", position=self.robot_base, show_axes=False
        )
        self.server.scene.add_frame(
            "/reference", position=self.robot_base, show_axes=False
        )
        self.actual_robot = ViserUrdf(
            self.server, robot_urdf, root_node_name="/actual"
        )
        self.reference_robot = ViserUrdf(
            self.server,
            robot_urdf,
            root_node_name="/reference",
            mesh_color_override=(40, 210, 90, 0.35),
        )
        zeros = np.zeros(26, dtype=np.float32)
        self.actual_robot.update_cfg(zeros)
        self.reference_robot.update_cfg(zeros)
        self.actual_object = None
        self.reference_object = None
        if self.object_tracking_enabled:
            assert self.object_dimensions is not None
            hidden_position = (0.0, 0.0, -10.0)
            self.actual_object = self.server.scene.add_box(
                "/actual_object",
                dimensions=self.object_dimensions,
                position=hidden_position,
                color=(65, 115, 210),
                opacity=0.95,
                side="double",
            )
            self.reference_object = self.server.scene.add_box(
                "/reference_object",
                dimensions=self.object_dimensions,
                position=hidden_position,
                color=(40, 210, 90),
                opacity=0.35,
                side="double",
            )

    def _build_gui(self) -> None:
        action_filter_description = (
            f"enabled, window {self.action_moving_average_window}"
            if self.action_moving_average
            else "disabled"
        )
        legend = "**Solid:** policy robot &nbsp; **Green:** reference motion"
        if self.object_tracking_enabled:
            legend += (
                "  \n**Blue cuboid:** simulated object &nbsp; "
                "**Green cuboid:** demonstration object"
            )
        self.server.gui.add_markdown(
            "# Motion Imitation\n"
            "Interactive policy evaluation\n\n"
            f"{legend}\n\n"
            f"**Demonstration:** `{Path(self.demonstration).name}`\n\n"
            f"**Policy action moving average:** {action_filter_description}"
        )
        with self.server.gui.add_folder("Environment", expand_by_default=True):
            self._btn_load = self.server.gui.add_button("Load Policy")
            self._btn_load.on_click(lambda _: self._load())
            self._status = self.server.gui.add_markdown("**Status:** Not loaded")

        with self.server.gui.add_folder("Episode", expand_by_default=True):
            self._random_phase = self.server.gui.add_checkbox(
                "Random RSI phase", initial_value=False
            )
            self._phase_slider = self.server.gui.add_slider(
                "Initial phase",
                min=0.0,
                max=0.99,
                step=0.01,
                initial_value=0.0,
            )
            self._realtime = self.server.gui.add_checkbox(
                "Run at 60 Hz", initial_value=True
            )
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._run_episode())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._toggle_pause())
            self._btn_stop = self.server.gui.add_button("Stop")
            self._btn_stop.on_click(lambda _: self._send("stop"))

        with self.server.gui.add_folder("Live Metrics", expand_by_default=True):
            self._progress = self.server.gui.add_markdown("**Phase:** --")
            self._errors = self.server.gui.add_markdown("**Tracking errors:** --")
            self._rewards = self.server.gui.add_markdown("**Reward:** --")
            self._stats = self.server.gui.add_markdown("**Episodes:** 0")

    def _load(self) -> None:
        self._kill_worker()
        self._status.content = "**Status:** Loading Isaac Gym and policy..."
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._conn = parent
        self._proc = context.Process(
            target=sim_worker,
            args=(
                child,
                self.config_path,
                self.checkpoint_path,
                self.use_cpu,
                self.plot_rewards,
                self.reward_plot_dir,
                self.action_moving_average,
                self.action_moving_average_window,
                self.hand_termination_error,
                self.plot_motion_diagnostics,
                self.diagnostic_plot_dir,
            ),
            daemon=True,
        )
        self._proc.start()
        child.close()
        print(f"[imitation-eval] Spawned worker pid={self._proc.pid}")

    def _kill_worker(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send("quit")
            except (BrokenPipeError, OSError):
                pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
            self._proc = None
        self._ready = self._running = self._paused = False

    def _send(self, message) -> None:
        if self._conn is not None:
            try:
                self._conn.send(message)
            except (BrokenPipeError, OSError):
                pass

    def _run_episode(self) -> None:
        if not self._ready:
            self._status.content = "**Status:** Load the policy first."
            return
        if self._running:
            return
        phase = None if self._random_phase.value else self._phase_slider.value
        self._running = True
        self._paused = False
        self._btn_pause.name = "Pause"
        self._status.content = "**Status:** Running..."
        self._send(
            (
                "run",
                {"phase": phase, "realtime": bool(self._realtime.value)},
            )
        )

    def _toggle_pause(self) -> None:
        if not self._running:
            return
        self._paused = not self._paused
        self._send("pause" if self._paused else "resume")
        self._btn_pause.name = "Resume" if self._paused else "Pause"

    def _update_state(
        self, state: Dict[str, Any], step: int, episode_return: float
    ) -> None:
        self.actual_robot.update_cfg(state["actual_q"])
        self.reference_robot.update_cfg(state["reference_q"])
        if state["object_tracking_enabled"]:
            if self.actual_object is None or self.reference_object is None:
                raise RuntimeError("Object state received without Viser object handles")
            actual_object_pose = state["actual_object_pose"]
            reference_object_pose = state["reference_object_pose"]
            self.actual_object.position = tuple(actual_object_pose[:3])
            self.actual_object.wxyz = quat_xyzw_to_wxyz(actual_object_pose[3:7])
            self.reference_object.position = tuple(reference_object_pose[:3])
            self.reference_object.wxyz = quat_xyzw_to_wxyz(
                reference_object_pose[3:7]
            )
        metrics = state["metrics"]
        self._progress.content = (
            f"**Phase:** {state['phase']:.3f} &nbsp;|&nbsp; "
            f"**Reference time:** {state['time_s']:.2f} / "
            f"{state['duration_s']:.2f} s &nbsp;|&nbsp; **Step:** {step}"
        )
        self._errors.content = (
            f"**Position:** {100.0 * metrics['position_error_m']:.2f} cm  \n"
            f"**Orientation:** {np.degrees(metrics['rotation_error_rad']):.2f}°  \n"
            f"**Hand L2:** {metrics['hand_error_rad']:.3f} rad"
        )
        if state["velocity_tracking_enabled"]:
            self._errors.content += (
                "  \n"
                f"**Palm linear velocity L2:** "
                f"{metrics['linear_velocity_error_mps']:.3f} m/s  \n"
                f"**Palm angular velocity L2:** "
                f"{metrics['angular_velocity_error_radps']:.3f} rad/s  \n"
                f"**Hand velocity L2:** "
                f"{metrics['hand_velocity_error_radps']:.3f} rad/s"
            )
        if state["object_tracking_enabled"]:
            self._errors.content += (
                "  \n"
                f"**Object position:** "
                f"{100.0 * metrics['object_position_error_m']:.2f} cm  \n"
                f"**Object keypoints mean/max:** "
                f"{100.0 * metrics['object_keypoint_mean_error_m']:.2f} / "
                f"{100.0 * metrics['object_keypoint_max_error_m']:.2f} cm"
            )
        self._rewards.content = (
            f"**Step reward:** {metrics['total_reward']:.4f} "
            f"(imitation {metrics['imitation_reward']:.4f}, "
            f"penalties {metrics['action_penalty']:.4f})  \n"
            f"Position {metrics['ee_position_reward']:.4f} &nbsp;|&nbsp; "
            f"Rotation {metrics['ee_rotation_reward']:.4f} &nbsp;|&nbsp; "
            f"Hand {metrics['hand_pose_reward']:.4f}  \n"
        )
        if state["velocity_tracking_enabled"]:
            velocity_contribution = (
                metrics["palm_linear_velocity_reward"]
                + metrics["palm_angular_velocity_reward"]
                + metrics["hand_velocity_reward"]
            )
            self._rewards.content += (
                f"Palm linear velocity "
                f"{metrics['palm_linear_velocity_reward']:.4f} "
                f"&nbsp;|&nbsp; Palm angular velocity "
                f"{metrics['palm_angular_velocity_reward']:.4f} "
                f"&nbsp;|&nbsp; Hand velocity "
                f"{metrics['hand_velocity_reward']:.4f}  \n"
                f"Pose contribution {metrics['pose_imitation_reward']:.4f} "
                f"&nbsp;|&nbsp; Velocity contribution "
                f"{velocity_contribution:.4f}  \n"
            )
        if state["object_tracking_enabled"]:
            self._rewards.content += (
                f"Object keypoints {metrics['object_keypoint_reward']:.4f}  \n"
            )
        self._rewards.content += f"**Episode return:** {episode_return:.2f}"

    def _handle(self, message) -> None:
        tag = message[0]
        if tag == "ready":
            self._ready = True
            self._update_state(message[1], 0, 0.0)
            self._status.content = "**Status:** Ready"
        elif tag == "state":
            self._update_state(message[1], message[2], message[3])
        elif tag == "done":
            reason, steps, episode_return = message[1:4]
            paths = message[4] if len(message) > 4 else {}
            self._running = self._paused = False
            self._episodes += 1
            self._completed += int(reason == "completed")
            self._returns.append(float(episode_return))
            plot_dir = paths.get("diagnostic_episode_dir") or paths.get("episode_dir")
            self._status.content = f"**Status:** {reason} after {steps} steps" + (
                f" — plots: `{plot_dir}`" if plot_dir else ""
            )
            self._stats.content = (
                f"**Episodes:** {self._episodes} &nbsp;|&nbsp; "
                f"**Completed:** {self._completed}/{self._episodes} &nbsp;|&nbsp; "
                f"**Mean return:** {np.mean(self._returns):.2f}"
            )
        elif tag == "stopped":
            self._running = self._paused = False
            paths = message[1] if len(message) > 1 else {}
            plot_dir = paths.get("diagnostic_episode_dir") or paths.get("episode_dir")
            self._status.content = "**Status:** Episode stopped" + (
                f" — plots: `{plot_dir}`" if plot_dir else ""
            )
        elif tag == "error":
            self._ready = self._running = self._paused = False
            self._status.content = f"**Status:** Error — {message[1][:240]}"
            print(f"[imitation-eval] Worker error:\n{message[1]}")

    def _poll(self) -> None:
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, ConnectionResetError, OSError):
            self._status.content = "**Status:** Worker exited unexpectedly"
            self._ready = self._running = False

    def run(self) -> None:
        print()
        print(f"Motion imitation evaluator: http://localhost:{self.port}")
        if self.action_moving_average:
            print(
                "Policy action moving average: enabled "
                f"(window={self.action_moving_average_window})"
            )
        print()
        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[imitation-eval] Shutting down...")
        finally:
            self._kill_worker()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive SimToolReal motion-imitation evaluation"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--plot-rewards", action="store_true")
    parser.add_argument("--reward-plot-dir", default=None)
    parser.add_argument(
        "--plot-motion-diagnostics",
        action="store_true",
        help=(
            "Save synchronized target pose/velocity, policy-action delta, and "
            "measured joint velocity/acceleration plots after each episode."
        ),
    )
    parser.add_argument(
        "--diagnostic-plot-dir",
        default=None,
        help=(
            "Diagnostic output root (default: eval_diagnostic_plots/motion_imitation)."
        ),
    )
    parser.add_argument(
        "--action-moving-average",
        action="store_true",
        help="Apply a causal moving average to policy actions before env.step().",
    )
    parser.add_argument(
        "--action-moving-average-window",
        type=int,
        default=5,
        help="Number of policy actions in the moving-average window (default: 5).",
    )
    parser.add_argument(
        "--hand-termination-error",
        type=float,
        default=None,
        help=(
            "override the saved 20-joint hand-pose L2 termination threshold "
            "in radians"
        ),
    )
    args = parser.parse_args()
    if args.action_moving_average_window < 2:
        parser.error("--action-moving-average-window must be at least 2")
    ImitationInteractiveDemo(
        args.config_path,
        args.checkpoint_path,
        port=args.port,
        use_cpu=args.use_cpu,
        plot_rewards=args.plot_rewards,
        reward_plot_dir=args.reward_plot_dir,
        action_moving_average=args.action_moving_average,
        action_moving_average_window=args.action_moving_average_window,
        hand_termination_error=args.hand_termination_error,
        plot_motion_diagnostics=args.plot_motion_diagnostics,
        diagnostic_plot_dir=args.diagnostic_plot_dir,
    ).run()


if __name__ == "__main__":
    main()
