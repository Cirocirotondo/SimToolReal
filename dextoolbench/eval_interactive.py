"""DexToolBench Interactive Policy Demo
========================================
Single viser server with full IsaacGym policy evaluation.

Architecture:
  Main process  -- viser GUI + scene rendering (lightweight, no isaacgym)
  Subprocess    -- IsaacGym env + policy (sends state back via pipe)

Each "Load Environment" kills the old subprocess and spawns a fresh one,
sidestepping the fact that IsaacGym cannot cleanly reset within a process.

Usage:
    python dextoolbench/eval_interactive.py \
        --config-path pretrained_policy/config.yaml \
        --checkpoint-path pretrained_policy/model.pth
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
import platform
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Python 3.8 compatibility:
# viser's HTTP static-file path check relies on Path.is_relative_to(), which is
# available in Python 3.9+. On Python 3.8 this causes HTTP 500 at "/" while
# websocket handshake still works. Provide a backport shim before importing viser.
if not hasattr(Path, "is_relative_to"):
    def _is_relative_to(self: Path, *other: Path) -> bool:
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    Path.is_relative_to = _is_relative_to  # type: ignore[attr-defined]

import viser
from viser.extras import ViserUrdf

from dextoolbench.eval_env_config import (
    CUBE_FIXED_SIZE,
    ISAAC_ROBOT_BASE_POS,
    build_eval_env_overrides,
    eval_table_center_pos,
    eval_viser_default_arm_dof,
    is_cube_eval,
    load_trajectory,
    table_urdf_rel_for_eval,
)
from dextoolbench.viser_colored_cube import add_colored_cube_viser
from dextoolbench.reward_episode_plotter import RewardEpisodePlotter
from dextoolbench.metadata import DEXTOOLBENCH_DATA_STRUCTURE, OBJECT_NAME_TO_CATEGORY

# Pre-load the sidebar overview image as a numpy array (once, at import time)
_SIDEBAR_IMG_PATH = Path(__file__).resolve().parent.parent / "assets" / "urdf" / "dextoolbench" / "dextoolbench_objects_sidebar.png"
_SIDEBAR_IMG = None
if _SIDEBAR_IMG_PATH.exists():
    from PIL import Image as _PILImage
    _SIDEBAR_IMG = np.asarray(_PILImage.open(_SIDEBAR_IMG_PATH).convert("RGB"))

# ═══════════════════════════════════════════════════════════════════
# Constants  (lightweight -- no isaacgym imports)
# ═══════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parent.parent
Z_OFFSET = 0.03

DEFAULT_DOF_POS = np.zeros(26)
DEFAULT_DOF_POS[:6] = eval_viser_default_arm_dof()

# ── Per-task environment URDFs ─────────────────────────────────────

ENV_DIR = REPO_ROOT / "assets" / "urdf" / "dextoolbench" / "environments"


def _setup_network_debug_logging():
    """Enable verbose logs from websockets/viser internals."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("websockets").setLevel(logging.DEBUG)
    logging.getLogger("websockets.server").setLevel(logging.DEBUG)
    logging.getLogger("viser").setLevel(logging.DEBUG)


def _http_probe(port: int):
    url = f"http://127.0.0.1:{port}/"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            body = resp.read(160).decode("utf-8", errors="replace")
            body = " ".join(body.splitlines())[:160]
            print(f"[net-debug] HTTP probe OK: status={resp.status} body='{body}'")
    except urllib.error.HTTPError as err:
        body = err.read(160).decode("utf-8", errors="replace")
        body = " ".join(body.splitlines())[:160]
        print(
            f"[net-debug] HTTP probe HTTPError: status={err.code} body='{body}'"
        )
    except Exception as err:
        print(f"[net-debug] HTTP probe failed: {type(err).__name__}: {err}")


def _ws_probe(port: int):
    async def _run():
        import websockets

        url = f"ws://127.0.0.1:{port}"
        try:
            async with websockets.connect(url, open_timeout=2.0):
                print(f"[net-debug] WS probe OK: {url}")
        except Exception as err:
            print(f"[net-debug] WS probe failed: {type(err).__name__}: {err}")

    try:
        asyncio.run(_run())
    except RuntimeError:
        # Fallback for environments where an event loop is already running.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()
    except Exception as err:
        print(f"[net-debug] WS probe launcher failed: {type(err).__name__}: {err}")


def _print_network_debug(port: int):
    """Print environment + local connectivity diagnostics."""
    print("[net-debug] ---- environment ----")
    print(f"[net-debug] python={sys.version.split()[0]} platform={platform.platform()}")
    print(f"[net-debug] host={socket.gethostname()} port={port}")
    print(
        "[net-debug] proxy env: "
        f"HTTP_PROXY={os.getenv('HTTP_PROXY')} "
        f"HTTPS_PROXY={os.getenv('HTTPS_PROXY')} "
        f"NO_PROXY={os.getenv('NO_PROXY')} "
        f"no_proxy={os.getenv('no_proxy')}"
    )
    try:
        import websockets

        print(f"[net-debug] websockets={websockets.__version__}")
    except Exception as err:
        print(f"[net-debug] websockets import failed: {type(err).__name__}: {err}")
    print(f"[net-debug] viser={viser.__version__}")
    print("[net-debug] ---- probes ----")
    _http_probe(port)
    _ws_probe(port)
    print("[net-debug] -----------------")


def _get_task_table_urdf(category, object_name, task_name):
    # type: (str, str, str) -> str
    """Return the URDF path relative to the assets/ dir (for IsaacGym)."""
    if is_cube_eval(category, object_name):
        return table_urdf_rel_for_eval(category, object_name, task_name)
    return (
        "urdf/dextoolbench/environments/"
        + category
        + "/"
        + object_name
        + "/"
        + task_name
        + ".urdf"
    )


def _get_task_table_urdf_abs(category, object_name, task_name):
    # type: (str, str, str) -> str
    """Return the absolute URDF path (for viser parsing)."""
    if is_cube_eval(category, object_name):
        return str(REPO_ROOT / "assets" / table_urdf_rel_for_eval(
            category, object_name, task_name
        ))
    return str(ENV_DIR / category / object_name / (task_name + ".urdf"))


def _parse_table_urdf(urdf_path):
    # type: (str) -> Tuple[list, list, list]
    """Parse a table URDF and extract props (nail, whiteboard, bowl/plate meshes).

    Returns (boxes, whiteboards, meshes) where:
      boxes:       [(name, (x,y,z), (sx,sy,sz), material_name), ...]
      whiteboards: [(x, y, z, w, h), ...]  -- whiteboard surfaces
      meshes:      [(name, (x,y,z), mesh_filename), ...]  -- bowl/plate meshes
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    boxes = []
    whiteboards = []
    meshes = []

    for visual in root.iter("visual"):
        origin = visual.find("origin")
        if origin is None:
            continue
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]

        geom = visual.find("geometry")
        if geom is None:
            continue

        mat = visual.find("material")
        mat_name = mat.get("name", "") if mat is not None else ""

        box = geom.find("box")
        mesh = geom.find("mesh")

        if box is not None:
            size = [float(v) for v in box.get("size", "0 0 0").split()]
            if mat_name == "wood":
                # This is the main table body -- skip, we always draw it
                continue
            elif mat_name == "whiteboard":
                whiteboards.append((xyz[0], xyz[1], xyz[2], size[1], size[2]))
            else:
                # Nail or other box prop
                boxes.append((mat_name, tuple(xyz), tuple(size)))
        elif mesh is not None:
            filename = mesh.get("filename", "")
            meshes.append((mat_name, tuple(xyz), filename))

    return boxes, whiteboards, meshes

# ── Dataset catalogue (built from metadata.py) ───────────────────

CATEGORY_DESCRIPTIONS = {
    "hammer": "Swing a hammer to hit a nail.",
    "spatula": "Flip or serve food with a spatula.",
    "eraser": "Wipe a whiteboard with an eraser.",
    "screwdriver": "Drive a screw from the top or side.",
    "marker": "Write shapes on a whiteboard.",
    "brush": "Sweep debris forward across the table.",
    "cube": "Pick up a training cube and follow a short lift-and-move trajectory.",
}


def _snake_to_title(s: str) -> str:
    """Convert snake_case to Title Case for display."""
    return s.replace("_", " ").title()


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


# ═══════════════════════════════════════════════════════════════════
# SUBPROCESS  -- IsaacGym simulation (all heavy imports stay here)
# ═══════════════════════════════════════════════════════════════════

def _sim_get_state(env, obs, joint_lower, joint_upper, n_act):
    """Extract visualisation state from the env."""
    obs_np = obs[0].cpu().numpy()
    joint_pos = 0.5 * (obs_np[:n_act] + 1.0) * (joint_upper - joint_lower) + joint_lower
    return (
        joint_pos,
        env.object_state[0, :7].cpu().numpy(),
        env.goal_pose[0].cpu().numpy(),
    )


def _sim_reset(env, n_act, device):
    import torch
    env.reset_idx(
        torch.arange(env.num_envs, dtype=torch.long, device=env.device),
        tensor_reset=True,
    )
    obs, _, _, _ = env.step(torch.zeros((env.num_envs, n_act), device=device))
    return obs["obs"]


def _sim_episode(
    conn,
    env,
    policy,
    joint_lower,
    joint_upper,
    n_act,
    device,
    *,
    plot_rewards: bool = False,
    reward_plot_dir: Optional[Path] = None,
    plot_live_every: int = 5,
    episode_slug: str = "episode",
):
    """Run one episode, streaming state to the parent via *conn*."""
    import time, torch  # noqa: E401

    control_dt = 1.0 / 60.0

    policy.reset()
    obs = _sim_reset(env, n_act, device)

    reward_plotter: Optional[RewardEpisodePlotter] = None
    if plot_rewards and reward_plot_dir is not None:
        reward_plotter = RewardEpisodePlotter(
            reward_plot_dir,
            live=False,
            live_every=plot_live_every,
        )
        reward_plotter.start_episode(episode_slug)
        print(f"[reward-plot] Logging reward terms → {reward_plot_dir}", flush=True)

    step, done, paused = 0, False, False

    def _finish(reason: str):
        paths = {}
        if reward_plotter is not None:
            paths = reward_plotter.finalize(reason)
            if paths:
                print("[reward-plot] Saved:", flush=True)
                for k, p in paths.items():
                    print(f"  {k}: {p}", flush=True)
        return paths

    while not done:
        # Drain commands (non-blocking)
        while conn.poll(0):
            cmd = conn.recv()
            if cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                plot_paths = _finish("stopped")
                conn.send(("stopped", plot_paths))
                return obs

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()

        state = _sim_get_state(env, obs, joint_lower, joint_upper, n_act)
        #print(f"obs: {obs}", flush=True)
        action = policy.get_normalized_action(obs, deterministic_actions=True)
        #print(f"action: {action}", flush=True)
        #time.sleep(1)
        obs_dict, _, done_tensor, extras = env.step(action)
        obs = obs_dict["obs"]
        done = done_tensor[0].item()
        step += 1
        if reward_plotter is not None:
            reward_plotter.record(extras, step)

        conn.send((
            "state",
            state,
            int(env.successes[0].item()),
            env.max_consecutive_successes,
            step,
        ))

        elapsed = time.time() - t0
        if (sleep := control_dt - elapsed) > 0:
            time.sleep(sleep)

    goal_pct = 100 * int(env.successes[0].item()) / env.max_consecutive_successes
    plot_paths = _finish("done")
    conn.send(("done", goal_pct, step, plot_paths))
    return obs


def sim_worker(
    conn,
    category,
    object_name,
    task_name,
    table_urdf,
    config_path,
    checkpoint_path,
    plot_rewards: bool = False,
    reward_plot_dir: Optional[str] = None,
    plot_live_every: int = 5,
):
    """Child process entry-point.  Creates the env, then waits for commands."""
    # ── Heavy imports (only in the subprocess) ────────────────
    try:
        from isaacgym import gymapi  # noqa: F401 isort:skip
    except ImportError:
        conn.send(("error",
                   "Isaac Gym is not installed. Download Isaac Gym Preview 4 "
                   "from https://developer.nvidia.com/isaac-gym-preview-4 and "
                   "install with: cd isaacgym/python && uv pip install -e ."))
        return
    import torch  # noqa: E401
    from deployment.rl_player import RlPlayer
    from deployment.isaac.isaac_env import create_env

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        traj_data = load_trajectory(category, object_name, task_name, z_offset=Z_OFFSET)

        env = create_env(
            config_path=str(config_path),
            headless=True,
            device=device,
            overrides=build_eval_env_overrides(
                category, object_name, table_urdf, traj_data, z_offset=0.0
            ),
        )
        n_act = int(env.num_acts)

        joint_lower = env.arm_hand_dof_lower_limits[:n_act].cpu().numpy()
        joint_upper = env.arm_hand_dof_upper_limits[:n_act].cpu().numpy()

        # Load policy
        env.set_env_state(torch.load(checkpoint_path)[0]["env_state"])
        policy = RlPlayer(
            int(env.num_obs), n_act, config_path, checkpoint_path, device, env.num_envs
        )

        # Initial reset
        obs = _sim_reset(env, n_act, device)
        init_state = _sim_get_state(env, obs, joint_lower, joint_upper, n_act)

        conn.send(("ready", init_state))

        # ── Command loop ─────────────────────────────────────
        while True:
            cmd = conn.recv()
            if cmd == "run":
                slug = f"{category}_{object_name}_{task_name}"
                obs = _sim_episode(
                    conn,
                    env,
                    policy,
                    joint_lower,
                    joint_upper,
                    n_act,
                    device,
                    plot_rewards=plot_rewards,
                    reward_plot_dir=(
                        Path(reward_plot_dir) if reward_plot_dir else None
                    ),
                    plot_live_every=plot_live_every,
                    episode_slug=slug,
                )
            elif cmd == "quit":
                break

    except Exception as exc:
        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))

    conn.close()


# ═══════════════════════════════════════════════════════════════════
# MAIN PROCESS  -- single viser server with all GUI + rendering
# ═══════════════════════════════════════════════════════════════════

class InteractiveDemo:

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        port: int = 8080,
        debug_network: bool = False,
        plot_rewards: bool = False,
        reward_plot_dir: Optional[str] = None,
        plot_live_every: int = 5,
    ):
        self.port = port
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.debug_network = debug_network
        self.plot_rewards = plot_rewards
        self.reward_plot_dir = (
            Path(reward_plot_dir)
            if reward_plot_dir
            else REPO_ROOT / "eval_reward_plots"
        )
        self.plot_live_every = plot_live_every
        if self.debug_network:
            _setup_network_debug_logging()
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        # Viser may auto-select a different free port if requested one is busy.
        self.port = int(self.server.get_port())

        # Subprocess
        self._proc = None  # type: Optional[multiprocessing.Process]
        self._conn = None  # type: Optional[multiprocessing.connection.Connection]
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

        # Pending config (set in _load_env, consumed in _handle_ready)
        self._pending_obj_name: str = ""
        self._pending_cat_key: str = ""

        # Stats
        self.ep_count = 0
        self.ep_goals = []  # type: List[float]
        self.ep_lengths = []  # type: List[int]

        # Scene handles
        self.robot = None  # type: Optional[ViserUrdf]
        self._dyn = []  # type: list
        self._obj_frame = None
        self._goal_frame = None

        self._build_gui()
        self._setup_static_scene()

    # ── GUI ────────────────────────────────────────────────────

    def _build_gui(self):
        self.server.gui.add_markdown(
            "# DexToolBench\n### Interactive Policy Demo"
        )

        if _SIDEBAR_IMG is not None:
            with self.server.gui.add_folder(
                "DexToolBench Objects", expand_by_default=True,
            ):
                self.server.gui.add_image(
                    _SIDEBAR_IMG,
                    label="Tool objects in the benchmark",
                    format="jpeg",
                )

        _PH = "-- Select --"
        with self.server.gui.add_folder("Dataset Selection", expand_by_default=True):
            cats = [_PH] + [_snake_to_title(c) for c in sorted(DEXTOOLBENCH_DATA_STRUCTURE.keys())]
            self._dd_cat = self.server.gui.add_dropdown(
                "Tool Category", options=cats, initial_value=_PH,
            )
            self._dd_obj = self.server.gui.add_dropdown(
                "Object Instance", options=[_PH], initial_value=_PH,
            )
            self._dd_task = self.server.gui.add_dropdown(
                "Task", options=[_PH], initial_value=_PH,
            )
            self._md_desc = self.server.gui.add_markdown(
                "*Select a tool category to begin.*"
            )
            self._btn_load = self.server.gui.add_button("Load Environment")
            self._btn_load.on_click(lambda _: self._load_env())
            self._md_status = self.server.gui.add_markdown("**Status:** Ready")
            self._dd_cat.on_update(lambda _: self._on_cat_change())

        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._cmd_run())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._cmd_pause())
            self._btn_stop = self.server.gui.add_button("Stop Episode")
            self._btn_stop.on_click(lambda _: self._cmd_stop())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_task = self.server.gui.add_markdown("**Task:** --")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")
            self._md_obj = self.server.gui.add_markdown("**Object Pos:** --")

    # ── Static scene ───────────────────────────────────────────

    def _setup_static_scene(self):
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        robot_urdf = (
            REPO_ROOT
            / "assets"
            / "urdf"
            / "ur5e_delto_description"
            / "ur5e_left_dg5f.urdf"
        )
        self.server.scene.add_frame(
            "/robot",
            position=ISAAC_ROBOT_BASE_POS,
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        self.robot = ViserUrdf(self.server, robot_urdf, root_node_name="/robot")
        self.robot.update_cfg(DEFAULT_DOF_POS)

        # Show a default table before any environment is loaded
        self._setup_default_table()

    def _setup_default_table(self):
        """Show the cube setup table before any environment is loaded."""
        self._clear_dynamic()
        t = self.server.scene.add_frame(
            "/table",
            position=eval_table_center_pos("cube", "training_cube"),
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        self._dyn.append(t)
        self._add_box(
            "/table/wood", (0.475, 0.4, 0.3), (0, 0, 0),
            color=(180, 130, 70), opacity=0.9,
        )

    # ── Cascading dropdown ─────────────────────────────────────

    def _get_category_key(self) -> Optional[str]:
        """Convert display name back to snake_case category key."""
        display = self._dd_cat.value
        for cat_key in DEXTOOLBENCH_DATA_STRUCTURE:
            if _snake_to_title(cat_key) == display:
                return cat_key
        return None

    def _on_cat_change(self):
        cat_key = self._get_category_key()
        if cat_key is None:
            return
        objects = DEXTOOLBENCH_DATA_STRUCTURE[cat_key]
        obj_display = [_snake_to_title(o) for o in objects.keys()]
        self._dd_obj.options = obj_display
        self._dd_obj.value = obj_display[0]

        # Tasks for the first object
        first_obj = list(objects.keys())[0]
        task_display = [_snake_to_title(t) for t in objects[first_obj]]
        self._dd_task.options = task_display
        self._dd_task.value = task_display[0]

        desc = CATEGORY_DESCRIPTIONS.get(cat_key, "")
        self._md_desc.content = f"*{desc}*" if desc else ""

        # Update tasks when object changes
        self._dd_obj.on_update(lambda _: self._on_obj_change())

    def _on_obj_change(self):
        cat_key = self._get_category_key()
        if cat_key is None:
            return
        obj_name = self._display_to_snake(self._dd_obj.value)
        objects = DEXTOOLBENCH_DATA_STRUCTURE[cat_key]
        if obj_name in objects:
            task_display = [_snake_to_title(t) for t in objects[obj_name]]
            self._dd_task.options = task_display
            self._dd_task.value = task_display[0]

    @staticmethod
    def _display_to_snake(display: str) -> str:
        """Convert Title Case display name back to snake_case."""
        return display.lower().replace(" ", "_")

    # ── Dynamic scene (rebuilt per config) ─────────────────────

    def _clear_dynamic(self):
        for h in self._dyn:
            try:
                h.remove()
            except Exception:
                pass
        self._dyn.clear()
        self._obj_frame = self._goal_frame = None

    def _add_box(self, name, dimensions, position, color, opacity=None):
        """Add a coloured box to the viser scene and track it in _dyn."""
        kwargs = dict(color=color, dimensions=dimensions, position=position, side="double")
        if opacity is not None:
            kwargs["opacity"] = opacity
        h = self.server.scene.add_box(name, **kwargs)
        self._dyn.append(h)
        return h

    def _setup_table(self, table_urdf_path, category: str, object_name: str):
        """Parse the per-task URDF and render coloured boxes in viser."""
        self._clear_dynamic()
        t = self.server.scene.add_frame(
            "/table",
            position=eval_table_center_pos(category, object_name),
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        self._dyn.append(t)

        # Always draw the wooden table body
        self._add_box(
            "/table/wood", (0.475, 0.4, 0.3), (0, 0, 0),
            color=(180, 130, 70), opacity=0.9,
        )

        # Parse URDF for additional props
        if Path(table_urdf_path).exists():
            boxes, whiteboards, meshes = _parse_table_urdf(table_urdf_path)

            # Render nail / wall / other box props with material-appropriate colors
            _MAT_COLORS = {
                "grey": (170, 175, 180),   # metallic grey (nail)
                "wall": (184, 122, 72),    # wooden wall
            }
            for i, (mat_name, xyz, size) in enumerate(boxes):
                color = _MAT_COLORS.get(mat_name, (170, 175, 180))
                self._add_box(
                    f"/table/prop_{i}", size, xyz, color=color,
                )

            # Render whiteboards
            for i, (bx, by, bz, bw, bh) in enumerate(whiteboards):
                fw = 0.03  # frame strip width
                fd = 0.03  # frame depth
                # White surface
                self._add_box(
                    f"/table/wb_surface_{i}", (0.02, bw, bh), (bx, by, bz),
                    color=(240, 240, 240),
                )
                # Wooden frame: 4 border strips
                self._add_box(
                    f"/table/wb_ft_{i}", (fd, bw + 2 * fw, fw),
                    (bx, by, bz + bh / 2 + fw / 2),
                    color=(140, 90, 45),
                )
                self._add_box(
                    f"/table/wb_fb_{i}", (fd, bw + 2 * fw, fw),
                    (bx, by, bz - bh / 2 - fw / 2),
                    color=(140, 90, 45),
                )
                self._add_box(
                    f"/table/wb_fl_{i}", (fd, fw, bh),
                    (bx, by - bw / 2 - fw / 2, bz),
                    color=(140, 90, 45),
                )
                self._add_box(
                    f"/table/wb_fr_{i}", (fd, fw, bh),
                    (bx, by + bw / 2 + fw / 2, bz),
                    color=(140, 90, 45),
                )

            # Render bowl/plate meshes using ViserUrdf (needs the actual URDF)
            if meshes:
                ViserUrdf(self.server, table_urdf_path,
                          root_node_name="/table")

        # Reset robot to default pose while we wait
        self.robot.update_cfg(DEFAULT_DOF_POS)

    def _setup_object_goal(self, object_name, category: str):
        """Add the object + goal URDFs (called once IsaacGym reports ready)."""
        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._obj_frame)

        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._goal_frame)

        if is_cube_eval(category, object_name):
            add_colored_cube_viser(
                self.server, "/object", self._dyn, scale=CUBE_FIXED_SIZE
            )
            add_colored_cube_viser(
                self.server,
                "/goal",
                self._dyn,
                scale=CUBE_FIXED_SIZE,
                opacity=0.85,
            )
        else:
            from dextoolbench.objects import NAME_TO_OBJECT

            obj_urdf = NAME_TO_OBJECT[object_name].urdf_path
            ViserUrdf(self.server, obj_urdf, root_node_name="/object")
            ViserUrdf(
                self.server,
                obj_urdf,
                root_node_name="/goal",
                mesh_color_override=(0, 255, 0, 0.5),
            )

    # ── Subprocess management ──────────────────────────────────

    def _kill_subprocess(self):
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
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

    def _load_env(self):
        cat_key = self._get_category_key()
        if cat_key is None:
            self._md_status.content = "**Status:** Please select a tool category first."
            return
        self._kill_subprocess()

        object_name = self._display_to_snake(self._dd_obj.value)
        task_name = self._display_to_snake(self._dd_task.value)
        table_urdf_rel = _get_task_table_urdf(cat_key, object_name, task_name)
        table_urdf_abs = _get_task_table_urdf_abs(cat_key, object_name, task_name)

        self._pending_obj_name = object_name
        self._pending_cat_key = cat_key

        # Show table + default robot pose immediately while IsaacGym loads
        self._setup_table(table_urdf_abs, cat_key, object_name)

        label = f"{_snake_to_title(cat_key)} / {self._dd_obj.value} / {self._dd_task.value}"
        self._md_status.content = f"**Status:** Loading *{label}* ..."
        self._md_task.content = f"**Task:** {label}"

        # Reset stats
        self.ep_count = 0
        self.ep_goals.clear()
        self.ep_lengths.clear()
        self._md_stats.content = "**Stats:** No episodes yet"

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=sim_worker,
            args=(
                child_conn,
                cat_key,
                object_name,
                task_name,
                table_urdf_rel,
                self.config_path,
                self.checkpoint_path,
                self.plot_rewards,
                str(self.reward_plot_dir),
                self.plot_live_every,
            ),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[launcher] Spawned subprocess pid={self._proc.pid}")

    # ── Commands to subprocess ─────────────────────────────────

    def _send(self, msg):
        if self._conn is not None:
            try:
                self._conn.send(msg)
            except (BrokenPipeError, OSError):
                pass

    def _cmd_run(self):
        if not self._env_ready:
            self._md_status.content = "**Status:** Load an environment first."
            return
        if self._episode_running:
            return
        self._episode_running = True
        self._is_paused = False
        self._btn_pause.name = "Pause"
        self._md_status.content = "**Status:** Running episode..."
        self._send("run")

    def _cmd_pause(self):
        if not self._episode_running:
            return
        self._is_paused = not self._is_paused
        self._send("pause" if self._is_paused else "resume")
        self._btn_pause.name = "Resume" if self._is_paused else "Pause"

    def _cmd_stop(self):
        if self._episode_running:
            self._send("stop")

    # ── Scene update ───────────────────────────────────────────

    def _update_viz(self, state_tuple):
        joint_pos, obj_pose, goal_pose = state_tuple
        self.robot.update_cfg(joint_pos)
        if self._obj_frame is not None:
            self._obj_frame.position = tuple(obj_pose[:3])
            self._obj_frame.wxyz = quat_xyzw_to_wxyz(obj_pose[3:7])
        if self._goal_frame is not None:
            self._goal_frame.position = tuple(goal_pose[:3])
            self._goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])

    # ── Message handling ───────────────────────────────────────

    def _handle(self, msg):
        tag = msg[0]

        if tag == "ready":
            init_state = msg[1]
            self._setup_object_goal(self._pending_obj_name, self._pending_cat_key)
            self._update_viz(init_state)
            self._env_ready = True
            self._md_status.content = "**Status:** Ready -- click **Run Episode**"
            print("[launcher] Environment ready")

        elif tag == "state":
            state, successes, max_succ, step = msg[1], msg[2], msg[3], msg[4]
            self._update_viz(state)
            pct = 100 * successes / max_succ if max_succ > 0 else 0
            self._md_prog.content = (
                f"**Time:** {step / 60.0:.1f}s &nbsp;|&nbsp; "
                f"**Goal:** {successes}/{max_succ} ({pct:.0f}%)"
            )
            obj_pos = state[1][:3]
            self._md_obj.content = (
                f"**Object Pos:** {obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}"
            )

        elif tag == "done":
            goal_pct, steps = msg[1], msg[2]
            plot_paths = msg[3] if len(msg) > 3 else {}
            self._episode_running = False
            self.ep_goals.append(goal_pct)
            self.ep_lengths.append(steps)
            self.ep_count += 1
            avg_g = np.mean(self.ep_goals)
            avg_t = np.mean(self.ep_lengths) / 60.0
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Avg Goal:** {avg_g:.1f}% &nbsp;|&nbsp; "
                f"**Avg Time:** {avg_t:.1f}s"
            )
            plot_note = ""
            if plot_paths:
                ep_dir = plot_paths.get("episode_dir", "")
                if ep_dir:
                    plot_note = f" &nbsp;|&nbsp; plots: `{ep_dir}/per_term/`"
            self._md_status.content = (
                f"**Status:** Done -- {steps / 60.0:.1f}s, {goal_pct:.0f}% goals"
                f"{plot_note}"
            )
            print(f"[launcher] Episode done: {goal_pct:.0f}% goals in {steps / 60.0:.1f}s")

        elif tag == "stopped":
            plot_paths = msg[1] if len(msg) > 1 else {}
            self._episode_running = False
            note = ""
            if plot_paths:
                ep_dir = plot_paths.get("episode_dir", "")
                if ep_dir:
                    note = f" (plots: {ep_dir}/per_term/)"
            self._md_status.content = f"**Status:** Episode stopped.{note}"

        elif tag == "error":
            self._env_ready = False
            self._episode_running = False
            self._md_status.content = f"**Status:** Error -- {msg[1][:200]}"
            print(f"[launcher] Subprocess error:\n{msg[1]}")

    def _poll(self):
        """Drain all pending messages from the subprocess pipe."""
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, ConnectionResetError, OSError):
            self._conn = None
            if self._proc is not None and not self._proc.is_alive():
                self._md_status.content = "**Status:** Subprocess exited unexpectedly."
                self._proc = None
                self._env_ready = False
                self._episode_running = False

    # ── Main loop ──────────────────────────────────────────────

    def run(self):
        print()
        print("  +-------------------------------------------------+")
        print("  |     DexToolBench Interactive Policy Demo         |")
        print(f"  |     http://localhost:{self.port:<26}|")
        print(f"  |     http://127.0.0.1:{self.port:<26}|")
        print("  +-------------------------------------------------+")
        print()
        if self.debug_network:
            # Give the server a brief moment before probing.
            time.sleep(0.2)
            _print_network_debug(self.port)

        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            self._kill_subprocess()


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DexToolBench Interactive Policy Demo",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--debug-network",
        action="store_true",
        help="Print extra diagnostics for HTTP/WebSocket connectivity.",
    )
    parser.add_argument(
        "--config-path", type=str, default="pretrained_policy/config.yaml",
        help="Path to the policy config YAML",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="pretrained_policy/model.pth",
        help="Path to the policy checkpoint",
    )
    parser.add_argument(
        "--plot-rewards",
        action="store_true",
        help="Plot reward sub-terms live during the episode and save PNG/NPZ after.",
    )
    parser.add_argument(
        "--reward-plot-dir",
        type=str,
        default=None,
        help="Directory for reward plots (default: <repo>/eval_reward_plots).",
    )
    parser.add_argument(
        "--plot-live-every",
        type=int,
        default=5,
        help="Update live matplotlib window every N sim steps (with --plot-rewards).",
    )
    args = parser.parse_args()
    InteractiveDemo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        port=args.port,
        debug_network=args.debug_network,
        plot_rewards=args.plot_rewards,
        reward_plot_dir=args.reward_plot_dir,
        plot_live_every=args.plot_live_every,
    ).run()
