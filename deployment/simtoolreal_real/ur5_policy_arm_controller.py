from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zmq


HERE = Path(__file__).parent
SIMTOOLREAL_ROOT = Path(
    os.environ.get("SIMTOOLREAL_ROOT", HERE.parent.parent)
).expanduser()
REPO_ROOT = SIMTOOLREAL_ROOT
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "train_dir/simtoolreal/2026-06-10/train_07_sim2real_resume_resume_2026-06-10_15-00-42"
    / "runs/00_train_07_sim2real_resume_resume_2026-06-10_15-00-42"
)
RUN_DIR = Path(os.environ.get("SIMTOOLREAL_RUN_DIR", DEFAULT_RUN_DIR)).expanduser()
CONFIG_PATH = RUN_DIR / "config.yaml"
NN_DIR = RUN_DIR / "nn"
LOW_LEVEL_CONFIG_PATH = HERE / "pc_ur_new.json"

CONTROL_HZ = 60.0
STATE_TIMEOUT_S = 2.0
PRINT_PERIOD_S = 1.0
MAX_ARM_TARGET_ERROR_DEG = 10.0
ARM_DOF = 6
N_ACT = 26
N_OBS = 131

# Policy-side cube scale used by the existing UR5e+Delto MuJoCo adapter.
CUBE_OBJECT_SCALES = np.array([1.25, 1.25, 1.25], dtype=np.float32)
OBJECT_BASE_SIZE_M = 0.04


class SimpleRateLimiter:
    def __init__(self, frequency: float) -> None:
        self.period = 1.0 / frequency
        self.next_time = time.monotonic()

    def sleep(self) -> None:
        self.next_time += self.period
        sleep_dt = self.next_time - time.monotonic()
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            self.next_time = time.monotonic()


def best_checkpoint_path(nn_dir: Path) -> Path:
    best_path = RUN_DIR / "best" / "model.pth"
    if best_path.exists():
        return best_path
    if best_path.is_symlink():
        raise FileNotFoundError(
            f"Default checkpoint symlink is broken: {best_path} -> {best_path.readlink()}"
        )

    candidates = []
    pattern = re.compile(r"_rew_([-+]?\d+(?:\.\d+)?)\.pth$")
    for path in nn_dir.glob("*.pth"):
        match = pattern.search(path.name)
        if match is not None:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        fallback = nn_dir / f"{RUN_DIR.name}.pth"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            "No policy checkpoint found. Expected one of:\n"
            f"  - {best_path}\n"
            f"  - {fallback}\n"
            f"  - any '*_rew_*.pth' file in {nn_dir}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def make_zmq_sockets(config: dict) -> tuple[zmq.Context, zmq.Socket, zmq.Socket]:
    context = zmq.Context()

    command_socket = context.socket(zmq.PUB)
    command_socket.bind(f"tcp://*:{config['socket_port']}")

    state_socket = context.socket(zmq.SUB)
    state_socket.setsockopt(zmq.CONFLATE, 1)
    state_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    state_socket.connect(f"tcp://127.0.0.1:{config['publisher_port']}")

    return context, command_socket, state_socket


def receive_latest_robot_state(state_socket: zmq.Socket) -> Optional[dict]:
    state = None
    while True:
        try:
            state = state_socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return state


def robot_state_to_policy_q(
    state: dict,
    previous_q: Optional[np.ndarray],
    default_joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = default_joint_pos.copy()
    qd = np.zeros(N_ACT, dtype=np.float32)

    if previous_q is not None:
        q[:] = previous_q

    arm_q = np.asarray(state.get("Q", []), dtype=np.float32)
    arm_qd = np.asarray(state.get("Qd", []), dtype=np.float32)
    if arm_q.shape[0] < ARM_DOF:
        raise ValueError(f"Robot state Q has shape {arm_q.shape}; expected at least {ARM_DOF}")

    q[:ARM_DOF] = arm_q[:ARM_DOF]
    if arm_qd.shape[0] >= ARM_DOF:
        qd[:ARM_DOF] = arm_qd[:ARM_DOF]
    return q, qd


def format_deg(values: np.ndarray) -> list[float]:
    return np.rad2deg(values).round(3).tolist()


def print_model_pose_debug(label: str, model, data) -> None:
    print(f"[debug] {label}", flush=True)
    print(f"[debug]   data.qpos_deg: {format_deg(data.qpos[:ARM_DOF])}", flush=True)
    if model.nu >= ARM_DOF:
        print(f"[debug]   data.ctrl_deg: {format_deg(data.ctrl[:ARM_DOF])}", flush=True)
    if model.nkey > 0:
        print(
            f"[debug]   home.qpos_deg: {format_deg(model.key_qpos[0, :ARM_DOF])}",
            flush=True,
        )


def wait_for_debug_step() -> None:
    print("Press Space/Enter for next step, or q to stop: ", end="", flush=True)
    if sys.stdin.isatty():
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()
    else:
        key = input().strip().lower()

    if key.lower() == "q":
        raise KeyboardInterrupt


def set_viewer_box(
    viewer,
    geom_index: int,
    pos: np.ndarray,
    size: np.ndarray,
    rgba: np.ndarray,
    geom_type,
) -> None:
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    mujoco = sys.modules["mujoco"]
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_index],
        geom_type,
        size.astype(np.float64),
        pos.astype(np.float64),
        mat,
        rgba.astype(np.float32),
    )


def update_viewer_markers(
    viewer,
    object_pos: np.ndarray,
    goal_object_pos: np.ndarray,
    table_z: float,
    target_palm_pos: Optional[np.ndarray] = None,
) -> None:
    mujoco = sys.modules["mujoco"]
    viewer.user_scn.ngeom = 4 if target_palm_pos is not None else 3
    set_viewer_box(
        viewer=viewer,
        geom_index=0,
        pos=np.array([0.0, -0.6, table_z], dtype=np.float32),
        size=np.array([0.475 / 2.0, 0.4 / 2.0, 0.3 / 2.0], dtype=np.float32),
        rgba=np.array([0.85, 0.85, 0.85, 0.35], dtype=np.float32),
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
    )
    set_viewer_box(
        viewer=viewer,
        geom_index=1,
        pos=object_pos,
        size=OBJECT_BASE_SIZE_M * CUBE_OBJECT_SCALES / 2.0,
        rgba=np.array([0.45, 0.45, 0.45, 1.0], dtype=np.float32),
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
    )
    set_viewer_box(
        viewer=viewer,
        geom_index=2,
        pos=goal_object_pos,
        size=OBJECT_BASE_SIZE_M * CUBE_OBJECT_SCALES / 2.0,
        rgba=np.array([0.1, 0.9, 0.2, 0.35], dtype=np.float32),
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
    )
    if target_palm_pos is not None:
        set_viewer_box(
            viewer=viewer,
            geom_index=3,
            pos=target_palm_pos,
            size=np.array([0.035, 0.035, 0.035], dtype=np.float32),
            rgba=np.array([1.0, 0.75, 0.05, 0.85], dtype=np.float32),
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the trained SimToolReal UR5 arm policy on the real low-level arm controller."
    )
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--low-level-config-path", type=Path, default=LOW_LEVEL_CONFIG_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--send-to-robot", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--debug-step",
        action="store_true",
        help="Print every step and wait for Space/Enter before sending/continuing.",
    )
    parser.add_argument(
        "--ignore-robot-state",
        action="store_true",
        help="Use the config default joint pose instead of Q from the low-level state publisher.",
    )
    parser.add_argument("--print-every", type=float, default=PRINT_PERIOD_S)
    args = parser.parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch

    import mujoco
    import mujoco.viewer
    from scipy.spatial.transform import Rotation

    from deployment.mujoco_ur5e_delto.policy_adapter import (
        DEFAULT_JOINT_POS,
        FINGERTIP_LOCAL_OFFSETS,
        LOWER_LIMITS,
        UPPER_LIMITS,
        build_observation,
        compute_targets,
        create_rl_player,
        read_policy_cfg,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    checkpoint_path = args.checkpoint_path or best_checkpoint_path(NN_DIR)
    cfg = read_policy_cfg(args.config_path)
    env_cfg = cfg["task"]["env"]
    num_obs = int(env_cfg.get("numObservations", N_OBS) or N_OBS)
    num_act = int(env_cfg.get("numActions", N_ACT) or N_ACT)
    if (num_obs, num_act) != (N_OBS, N_ACT):
        raise RuntimeError(
            f"This controller expects a UR5e+Delto policy with {N_OBS}/{N_ACT} "
            f"obs/actions, got {num_obs}/{num_act}"
        )
    default_joint_pos = DEFAULT_JOINT_POS.copy()
    default_arm_dof_pos = np.asarray(
        env_cfg.get("defaultArmDofPos", default_joint_pos[:ARM_DOF]),
        dtype=np.float32,
    )
    if default_arm_dof_pos.shape[0] != ARM_DOF:
        raise ValueError(
            f"defaultArmDofPos has shape {default_arm_dof_pos.shape}; expected {ARM_DOF}"
        )
    default_joint_pos[:ARM_DOF] = default_arm_dof_pos

    with args.low_level_config_path.open(encoding="utf-8") as f:
        low_level_cfg = json.load(f)

    context, command_socket, state_socket = make_zmq_sockets(low_level_cfg)
    print(f"Policy config:     {args.config_path}")
    print(f"Policy checkpoint: {checkpoint_path}")
    print(f"Listening state:   tcp://127.0.0.1:{low_level_cfg['publisher_port']}")
    print(f"Publishing target: tcp://*:{low_level_cfg['socket_port']}")
    print("Robot state input: IGNORED." if args.ignore_robot_state else "Robot state input: ENABLED.")
    print("Robot publishing is ENABLED." if args.send_to_robot else "Dry run: not sending robot commands.")

    model_path = HERE / "assets" / "universal_robots_ur5e" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    target_data = mujoco.MjData(model)
    data.qpos[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    target_data.qpos[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    data.ctrl[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    target_data.ctrl[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, target_data)
    if args.debug_step:
        print(f"[debug] defaultArmDofPos_deg: {format_deg(default_joint_pos[:ARM_DOF])}", flush=True)
        print_model_pose_debug("after initial qpos/ctrl setup", model, data)
    wrist_3_body_id = model.body("wrist_3_link").id
    table_z = float(env_cfg.get("tableResetZ", 0.38))
    object_pos = np.array(
        [0.0, -0.6, table_z + float(env_cfg.get("tableObjectZOffset", 0.25))],
        dtype=np.float32,
    )
    goal_object_pos = np.array(
        [0.12, -0.6, table_z + 0.35],
        dtype=np.float32,
    )
    object_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        update_viewer_markers(viewer, object_pos, goal_object_pos, table_z)
        viewer.sync()
        if args.debug_step:
            print_model_pose_debug("after viewer launch/sync", model, data)

    player = create_rl_player(
        simtoolreal_root=REPO_ROOT,
        config_path=args.config_path,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    data.qpos[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    data.ctrl[:ARM_DOF] = default_joint_pos[:ARM_DOF]
    mujoco.mj_forward(model, data)
    if viewer is not None:
        update_viewer_markers(viewer, object_pos, goal_object_pos, table_z)
        viewer.sync()
    if args.debug_step:
        print_model_pose_debug("after policy load and resync", model, data)

    obs_list = env_cfg["obsList"]
    if args.control_hz <= 0.0:
        raise ValueError(f"--control-hz must be positive, got {args.control_hz}")
    control_dt = 1.0 / args.control_hz
    dof_speed_scale = float(env_cfg["dofSpeedScale"])
    arm_moving_average = float(env_cfg["armMovingAverage"])
    hand_moving_average = float(env_cfg["handMovingAverage"])

    prev_q = None
    prev_targets = None
    latest_state = {
        "Q": default_joint_pos[:ARM_DOF].tolist(),
        "Qd": [0.0] * ARM_DOF,
        "source": "config_default",
    }
    if not args.ignore_robot_state:
        latest_state = None
        deadline = time.monotonic() + STATE_TIMEOUT_S
        while time.monotonic() < deadline:
            latest_state = receive_latest_robot_state(state_socket)
            if latest_state is not None and "Q" in latest_state:
                break
            time.sleep(0.01)
        if latest_state is None or "Q" not in latest_state:
            raise TimeoutError("No robot state received. Is the low-level controller running?")
        if args.debug_step:
            print(f"[debug] initial robot state keys: {sorted(latest_state.keys())}", flush=True)
            print(
                f"[debug] initial robot Q_deg: {format_deg(np.asarray(latest_state['Q'])[:ARM_DOF])}",
                flush=True,
            )

    # Give the C++ subscriber a moment to connect before sending targets.
    time.sleep(1.0)
    rate = SimpleRateLimiter(frequency=args.control_hz)
    last_print = 0.0
    step = 0

    try:
        while args.max_steps <= 0 or step < args.max_steps:
            if viewer is not None and not viewer.is_running():
                break

            state = None if args.ignore_robot_state else receive_latest_robot_state(state_socket)
            if state is not None:
                latest_state = state

            q, qd = robot_state_to_policy_q(
                latest_state,
                previous_q=prev_q,
                default_joint_pos=default_joint_pos,
            )
            prev_q = q

            data.qpos[:ARM_DOF] = q[:ARM_DOF]
            data.qvel[:ARM_DOF] = qd[:ARM_DOF]
            data.ctrl[:ARM_DOF] = q[:ARM_DOF]
            mujoco.mj_forward(model, data)
            if viewer is not None:
                update_viewer_markers(viewer, object_pos, goal_object_pos, table_z)
                viewer.sync()

            palm_quat_wxyz = data.xquat[wrist_3_body_id].copy()
            palm_rot = Rotation.from_quat(palm_quat_wxyz[[1, 2, 3, 0]])
            palm_pos = data.xpos[wrist_3_body_id].copy() + palm_rot.apply(
                np.array([0.0, 0.0, 0.16])
            )
            fingertip_positions = palm_pos[None, :] + palm_rot.apply(
                FINGERTIP_LOCAL_OFFSETS
            )

            sim_state = {
                "joint_positions": q,
                "joint_velocities": qd,
                "palm_pos": palm_pos.astype(np.float32),
                "palm_quat_wxyz": palm_quat_wxyz.astype(np.float32),
                "fingertip_positions": fingertip_positions.astype(np.float32),
                "object_pos": object_pos,
                "object_quat_wxyz": object_quat_wxyz,
                "goal_object_pos": goal_object_pos,
                "goal_object_quat_wxyz": object_quat_wxyz,
            }

            obs = build_observation(
                sim_state=sim_state,
                object_scales=CUBE_OBJECT_SCALES,
                obs_list=obs_list,
                prev_targets=prev_targets,
            )
            obs_t = torch.from_numpy(obs).float().to(args.device)
            with torch.no_grad():
                action = player.get_normalized_action(obs_t, deterministic_actions=True)

            targets = compute_targets(
                actions=action.cpu().numpy()[0],
                q=q,
                prev_targets=prev_targets,
                control_dt=control_dt,
                dof_speed_scale=dof_speed_scale,
                arm_moving_average=arm_moving_average,
                hand_moving_average=hand_moving_average,
            )
            targets[:ARM_DOF] = np.clip(
                targets[:ARM_DOF], LOWER_LIMITS[:ARM_DOF], UPPER_LIMITS[:ARM_DOF]
            )

            arm_error_deg = np.rad2deg(np.abs(targets[:ARM_DOF] - q[:ARM_DOF]))
            if arm_error_deg.max() > MAX_ARM_TARGET_ERROR_DEG:
                raise RuntimeError(
                    "Refusing to send arm target too far from current state: "
                    f"{arm_error_deg.round(2).tolist()} deg"
                )

            action_arm = action.cpu().numpy()[0, :ARM_DOF]
            target_palm_pos = None
            if viewer is not None:
                target_data.qpos[:ARM_DOF] = targets[:ARM_DOF]
                target_data.qvel[:ARM_DOF] = 0.0
                target_data.ctrl[:ARM_DOF] = targets[:ARM_DOF]
                mujoco.mj_forward(model, target_data)
                target_palm_quat_wxyz = target_data.xquat[wrist_3_body_id].copy()
                target_palm_rot = Rotation.from_quat(
                    target_palm_quat_wxyz[[1, 2, 3, 0]]
                )
                target_palm_pos = target_data.xpos[
                    wrist_3_body_id
                ].copy() + target_palm_rot.apply(np.array([0.0, 0.0, 0.16]))
                update_viewer_markers(
                    viewer,
                    object_pos,
                    goal_object_pos,
                    table_z,
                    target_palm_pos=target_palm_pos,
                )
                viewer.sync()

            if args.debug_step:
                print()
                print(f"step:             {step}")
                print(f"send_to_robot:    {args.send_to_robot}")
                print(f"q_rad:            {q[:ARM_DOF].round(6).tolist()}")
                print(f"q_deg:            {format_deg(q[:ARM_DOF])}")
                print(f"target_q_rad:     {targets[:ARM_DOF].round(6).tolist()}")
                print(f"target_q_deg:     {format_deg(targets[:ARM_DOF])}")
                print(f"target_delta_deg: {format_deg(targets[:ARM_DOF] - q[:ARM_DOF])}")
                print(f"qd_rad_s:         {qd[:ARM_DOF].round(6).tolist()}")
                print(f"action_arm:       {action_arm.round(6).tolist()}")
                print(f"object_pos:       {object_pos.round(6).tolist()}")
                print(f"state_source:     {latest_state.get('source', 'robot_state_publisher')}")
                if "timestamp_ms" in latest_state:
                    print(f"state_timestamp:  {latest_state['timestamp_ms']}")
                if target_palm_pos is not None:
                    print(f"target_palm_pos:  {target_palm_pos.round(6).tolist()}")
                wait_for_debug_step()

            if args.send_to_robot:
                command_socket.send_json({"target_q": targets[:ARM_DOF].tolist()})

            prev_targets = targets
            now = time.monotonic()
            if not args.debug_step and now - last_print >= args.print_every:
                last_print = now
                print(
                    f"step={step:06d} "
                    f"q_deg={np.rad2deg(q[:ARM_DOF]).round(2).tolist()} "
                    f"target_deg={np.rad2deg(targets[:ARM_DOF]).round(2).tolist()} "
                    f"action_arm={action_arm.round(3).tolist()}"
                )

            step += 1
            if not args.debug_step:
                rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        command_socket.close()
        state_socket.close()
        context.term()
        print("Stopped.")


if __name__ == "__main__":
    main()
