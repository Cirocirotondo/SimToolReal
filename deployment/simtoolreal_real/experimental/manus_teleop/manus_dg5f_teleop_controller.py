#!/usr/bin/env python3
"""Safely stream the right MANUS glove posture to the real DG5F hand.

This process deliberately contains no ROS dependency.  It receives MANUS
ergonomics data over ZMQ and exchanges measured states / position targets with
``dg5f_policy_ros_bridge.py`` over localhost UDP.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import zmq


HAND_DOF = 20
JOINT_NAMES = [
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]

# Same conservative right-hand limits enforced by dg5f_policy_ros_bridge.py.
RIGHT_LOWER = np.array(
    [
        -0.3839724354, -3.1415926536, 0.0, 0.0,
        -0.4188790205, 0.0, 0.0, 0.0,
        -0.6108652382, 0.0, 0.0, 0.0,
        -0.6108652382, 0.0, 0.0, 0.0,
        -0.0174532925, -0.4188790205, 0.0, 0.0,
    ],
    dtype=np.float64,
)
RIGHT_UPPER = np.array(
    [
        0.8901179185, 0.0, 1.5707963268, 1.5707963268,
        0.6108652382, 2.0071286398, 1.5707963268, 1.5707963268,
        0.6108652382, 1.9547687622, 1.5707963268, 1.5707963268,
        0.4188790205, 1.9024088847, 1.5707963268, 1.5707963268,
        1.0471975512, 0.6108652382, 1.5707963268, 1.5707963268,
    ],
    dtype=np.float64,
)
LONG_FINGER_FLEX_INDICES = np.array(
    [5, 6, 7, 9, 10, 11, 13, 14, 15], dtype=int
)
PINKY_OPPOSITION_INDEX = 16
DEFAULT_MODEL_PATH = (
    Path("/home/duplo/git/robohand-robohand2/assets/output.xml")
)


def manus_right_to_dg5f(
    raw_degrees: np.ndarray,
    *,
    flex_gain: float,
    pinky_opposition_degrees: float,
    pinky_j2_source: str,
    pinky_j2_gain: float,
    pinky_j2_min_degrees: float,
    thumb_j1_offset_degrees: float,
) -> np.ndarray:
    """Apply the direct-joints mapping validated in the MuJoCo viewer."""
    raw = np.asarray(raw_degrees, dtype=np.float64)
    if raw.shape != (HAND_DOF,) or not np.all(np.isfinite(raw)):
        raise ValueError("MANUS sample must contain 20 finite angles")

    mapped = raw.copy()

    # Thumb: exchange CMC stretch/spread to match the first two DG5F axes.
    mapped[0] = raw[1] + thumb_j1_offset_degrees
    mapped[1] = -raw[0] - 45.0

    # Index, middle and ring spread axes have the opposite sign.
    mapped[[4, 8, 12]] *= -1.0

    # DG5F pinky opposition/cupping has no direct MANUS equivalent.  Keep it
    # at an explicit fixed posture.  Joint 2 can be tested against either raw
    # MANUS MCP spread (the anatomical match) or stretch (the legacy synergy).
    mapped[16] = pinky_opposition_degrees
    if pinky_j2_source == "spread":
        pinky_source_value = raw[16]
    elif pinky_j2_source == "stretch":
        pinky_source_value = raw[17]
    else:
        raise ValueError(
            "pinky_j2_source must be either 'spread' or 'stretch'"
        )
    mapped[17] = pinky_j2_gain * pinky_source_value
    mapped[17] = max(mapped[17], pinky_j2_min_degrees)

    mapped = np.deg2rad(mapped)
    mapped[LONG_FINGER_FLEX_INDICES] *= flex_gain
    return np.clip(mapped, RIGHT_LOWER, RIGHT_UPPER)


class ManusReceiver:
    def __init__(self, address: str) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(address)
        self.latest: Optional[np.ndarray] = None
        self.received_at: Optional[float] = None

    def receive_latest(self) -> Optional[np.ndarray]:
        while True:
            try:
                payload = self.socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                return None if self.latest is None else self.latest.copy()
            fields = payload.split(",")
            if len(fields) != 40:
                continue
            try:
                right = np.asarray(fields[20:40], dtype=np.float64)
            except ValueError:
                continue
            if not np.all(np.isfinite(right)) or np.max(np.abs(right)) <= 0.5:
                continue
            self.latest = right
            self.received_at = time.monotonic()

    def close(self) -> None:
        self.socket.close()
        self.context.term()


class HandBridgeClient:
    def __init__(
        self,
        *,
        state_address: str,
        state_port: int,
        command_address: str,
        command_port: int,
    ) -> None:
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_socket.setblocking(False)
        self.state_socket.bind((state_address, state_port))
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_endpoint = (command_address, command_port)
        self.latest_state: Optional[dict] = None
        self.sequence = 0

    def receive_latest_state(self) -> Optional[dict]:
        while True:
            try:
                payload, _ = self.state_socket.recvfrom(65535)
            except BlockingIOError:
                return self.latest_state
            try:
                message = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if message.get("type") != "hand_state":
                continue
            q = np.asarray(message.get("positions"), dtype=np.float64)
            qd = np.asarray(message.get("velocities"), dtype=np.float64)
            if q.shape != (HAND_DOF,) or qd.shape != (HAND_DOF,):
                continue
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                continue
            currents_ma = message.get("currents_ma")
            if currents_ma is not None:
                currents_ma = np.asarray(currents_ma, dtype=np.float64)
                if (
                    currents_ma.shape != (HAND_DOF,)
                    or not np.all(np.isfinite(currents_ma))
                ):
                    currents_ma = None
            self.latest_state = {
                "q": q,
                "qd": qd,
                "currents_ma": currents_ma,
                "received_at": time.monotonic(),
            }

    def send_target(self, target: np.ndarray) -> None:
        self.sequence += 1
        payload = {
            "type": "hand_target",
            "sequence": self.sequence,
            "positions": np.asarray(target, dtype=float).tolist(),
        }
        self.command_socket.sendto(
            json.dumps(payload, separators=(",", ":")).encode(),
            self.command_endpoint,
        )

    def close(self) -> None:
        self.state_socket.close()
        self.command_socket.close()


class FixedWristPreview:
    """MuJoCo preview that contains only the fixed-base DG5F hand."""

    def __init__(self, model_path: Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.qpos_addresses = np.array(
            [int(self.model.joint(name).qposadr[0]) for name in JOINT_NAMES],
            dtype=int,
        )
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

    def update(self, target: np.ndarray) -> bool:
        if not self.viewer.is_running():
            return False
        # The selected model has no free joint: its wrist/base is fixed by
        # construction.  Only the 20 finger qpos entries are ever modified.
        self.data.qpos[self.qpos_addresses] = np.asarray(target, dtype=float)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.viewer.sync()
        return True

    def close(self) -> None:
        self.viewer.close()


def wait_for_glove(receiver: ManusReceiver, timeout: float) -> np.ndarray:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sample = receiver.receive_latest()
        if sample is not None and receiver.received_at is not None:
            return sample
        time.sleep(0.01)
    raise TimeoutError("No valid right-glove MANUS sample received")


def wait_for_hand_state(bridge: HandBridgeClient, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = bridge.receive_latest_state()
        if state is not None:
            return state
        time.sleep(0.01)
    raise TimeoutError(
        "No DG5F state received; check the ROS driver and "
        "dg5f_policy_ros_bridge.py"
    )


def wait_for_fresh_hand_state(
    bridge: HandBridgeClient,
    *,
    newer_than: float,
    timeout: float,
) -> dict:
    """Wait for a state received after an interactive arming prompt."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = bridge.receive_latest_state()
        if state is not None and state["received_at"] > newer_than:
            return state
        time.sleep(0.01)
    raise TimeoutError("No fresh DG5F state received after SEND confirmation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct right-MANUS teleoperation of the real DG5F hand."
    )
    parser.add_argument("--manus-address", default="tcp://127.0.0.1:8001")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Fixed-wrist DG5F MuJoCo model used by the preview.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Disable the MuJoCo preview window.",
    )
    parser.add_argument("--state-address", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=5563)
    parser.add_argument("--command-address", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=5562)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--flex-gain", type=float, default=1.15)
    parser.add_argument(
        "--thumb-j1-offset-deg",
        type=float,
        default=-7.0,
        help=(
            "Offset in DG5F thumb-j1 = MANUS CMC stretch + offset "
            "(default: -7 deg)."
        ),
    )
    parser.add_argument("--pinky-opposition-deg", type=float, default=0.0)
    parser.add_argument(
        "--pinky-j2-source",
        choices=("spread", "stretch"),
        default="spread",
        help=(
            "MANUS channel used for DG5F pinky joint 2. Use 'spread' for "
            "abduction/adduction; 'stretch' reproduces the legacy synergy."
        ),
    )
    parser.add_argument(
        "--pinky-j2-gain",
        type=float,
        default=-3.0,
        help=(
            "Signed gain from the selected MANUS channel to DG5F pinky "
            "joint 2 (default: -3.0)."
        ),
    )
    parser.add_argument(
        "--pinky-j2-min-deg",
        type=float,
        default=5.0,
        help=(
            "Minimum DG5F pinky-j2 angle, keeping the resting pinky away "
            "from the other fingers (default: 5 deg)."
        ),
    )
    parser.add_argument("--glove-timeout", type=float, default=0.25)
    parser.add_argument("--state-timeout", type=float, default=0.25)
    parser.add_argument(
        "--max-velocity-rad-s",
        type=float,
        default=0.5,
        help="Per-joint target slew-rate limit (default: 0.5 rad/s).",
    )
    parser.add_argument(
        "--max-tracking-error-rad",
        type=float,
        default=0.5,
        help="Stop if any measured joint trails the streamed target by more.",
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        default=None,
        help="Optional automatic stop time in seconds.",
    )
    parser.add_argument(
        "--send-to-hand",
        action="store_true",
        help="Actually transmit targets. Without this flag the run is dry.",
    )
    parser.add_argument("--skip-confirmation", action="store_true")
    args = parser.parse_args()

    for name in (
        "control_hz",
        "flex_gain",
        "glove_timeout",
        "state_timeout",
        "max_velocity_rad_s",
        "max_tracking_error_rad",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.pinky_j2_gain == 0.0:
        parser.error("--pinky-j2-gain must be non-zero")
    pinky_min_rad = np.deg2rad(args.pinky_j2_min_deg)
    if not RIGHT_LOWER[17] <= pinky_min_rad <= RIGHT_UPPER[17]:
        parser.error(
            "--pinky-j2-min-deg must be inside the physical joint range "
            f"[{np.degrees(RIGHT_LOWER[17]):.1f}, "
            f"{np.degrees(RIGHT_UPPER[17]):.1f}] deg"
        )
    if args.max_runtime is not None and args.max_runtime <= 0.0:
        parser.error("--max-runtime must be positive")
    return args


def main() -> None:
    args = parse_args()
    receiver = ManusReceiver(args.manus_address)
    bridge: Optional[HandBridgeClient] = None
    preview: Optional[FixedWristPreview] = None
    latest_state: Optional[dict] = None
    command_was_sent = False

    try:
        print(f"Waiting for right MANUS glove on {args.manus_address}...")
        raw = wait_for_glove(receiver, 5.0)
        requested = manus_right_to_dg5f(
            raw,
            flex_gain=args.flex_gain,
            pinky_opposition_degrees=args.pinky_opposition_deg,
            pinky_j2_source=args.pinky_j2_source,
            pinky_j2_gain=args.pinky_j2_gain,
            pinky_j2_min_degrees=args.pinky_j2_min_deg,
            thumb_j1_offset_degrees=args.thumb_j1_offset_deg,
        )
        print(
            "MANUS ready. Target degrees: "
            f"{np.round(np.rad2deg(requested), 1).tolist()}"
        )
        print(
            "Initial thumb diagnostic: MANUS raw "
            f"spread={raw[0]:.1f} deg, stretch={raw[1]:.1f} deg; "
            f"DG5F j1 before clipping="
            f"{raw[1] + args.thumb_j1_offset_deg:.1f} deg, "
            f"after clipping={np.degrees(requested[0]):.1f} deg"
        )
        print(
            "Initial pinky diagnostic: MANUS raw "
            f"spread={raw[16]:.1f} deg, stretch={raw[17]:.1f} deg; "
            "DG5F j2 target="
            f"{np.degrees(requested[17]):.1f} deg "
            f"(from MCP {args.pinky_j2_source}, "
            f"gain={args.pinky_j2_gain:g}, "
            f"minimum={args.pinky_j2_min_deg:g} deg)"
        )
        print(
            "Do not run another MANUS PULL receiver at the same time; "
            "ZMQ PUSH distributes samples between receivers."
        )
        if not args.no_viewer:
            preview = FixedWristPreview(args.model_path)
            preview.update(requested)
            print(
                "MuJoCo preview ready: fixed wrist/base; only the 20 finger "
                "joints are updated. Close the window to stop."
            )

        if args.send_to_hand:
            bridge = HandBridgeClient(
                state_address=args.state_address,
                state_port=args.state_port,
                command_address=args.command_address,
                command_port=args.command_port,
            )
            print(
                f"Waiting for DG5F state on udp://{args.state_address}:"
                f"{args.state_port}..."
            )
            latest_state = wait_for_hand_state(bridge, 5.0)
            initial_error = float(np.max(np.abs(requested - latest_state["q"])))
            print(
                "DG5F ready. Initial max target difference: "
                f"{initial_error:.3f} rad ({np.degrees(initial_error):.1f} deg)."
            )
            print(
                "The target starts from the measured hand and is limited to "
                f"{args.max_velocity_rad_s:g} rad/s per joint."
            )
            if not args.skip_confirmation:
                answer = input(
                    "REAL HAND COMMANDS ARE ENABLED. Keep the workspace clear "
                    "and type SEND to start: "
                )
                if answer.strip() != "SEND":
                    print("Aborted; no command was sent.")
                    return
            # The hand may move while the user is reading the prompt.  Never
            # use the state captured before SEND as the first slew reference.
            confirmed_at = time.monotonic()
            latest_state = wait_for_fresh_hand_state(
                bridge,
                newer_than=confirmed_at,
                timeout=2.0,
            )
            streamed = latest_state["q"].copy()
            print(
                "Armed from fresh measured DG5F state; first target delta is "
                "zero."
            )
            outside = np.flatnonzero(
                (streamed < RIGHT_LOWER) | (streamed > RIGHT_UPPER)
            )
            if outside.size:
                details = ", ".join(
                    f"{JOINT_NAMES[index]}="
                    f"{np.degrees(streamed[index]):.1f} deg"
                    for index in outside
                )
                print(
                    "Measured joints outside the conservative target limits; "
                    "they will recover gradually without an initial jump: "
                    + details
                )
        else:
            print("DRY RUN: no command will be sent to the real hand.")
            streamed = requested.copy()

        period = 1.0 / args.control_hz
        max_step = args.max_velocity_rad_s * period
        started_at = time.monotonic()
        next_tick = started_at
        last_log_at = 0.0

        while True:
            now = time.monotonic()
            raw = receiver.receive_latest()
            if (
                raw is None
                or receiver.received_at is None
                or now - receiver.received_at > args.glove_timeout
            ):
                raise RuntimeError(
                    "MANUS glove stream is stale; stopping commands so the "
                    "bridge watchdog holds the measured hand position."
                )

            requested = manus_right_to_dg5f(
                raw,
                flex_gain=args.flex_gain,
                pinky_opposition_degrees=args.pinky_opposition_deg,
                pinky_j2_source=args.pinky_j2_source,
                pinky_j2_gain=args.pinky_j2_gain,
                pinky_j2_min_degrees=args.pinky_j2_min_deg,
                thumb_j1_offset_degrees=args.thumb_j1_offset_deg,
            )
            streamed += np.clip(requested - streamed, -max_step, max_step)
            # requested is already inside the target limits.  Since this is a
            # convex step toward requested, do not hard-clip here: a measured
            # initial position may legitimately be outside the conservative
            # command interval and must return gradually rather than jump to
            # the nearest boundary in one cycle.

            if preview is not None and not preview.update(streamed):
                print("MuJoCo window closed; stopping teleoperation.")
                break

            if args.send_to_hand:
                assert bridge is not None
                state = bridge.receive_latest_state()
                if state is not None:
                    latest_state = state
                if (
                    latest_state is None
                    or now - latest_state["received_at"] > args.state_timeout
                ):
                    raise RuntimeError(
                        "DG5F state is stale; stopping commands so the bridge "
                        "watchdog holds position."
                    )
                tracking_error = float(
                    np.max(np.abs(streamed - latest_state["q"]))
                )
                if tracking_error > args.max_tracking_error_rad:
                    worst = int(np.argmax(np.abs(streamed - latest_state["q"])))
                    currents_ma = latest_state.get("currents_ma")
                    current_detail = (
                        ""
                        if currents_ma is None
                        else f", current={currents_ma[worst]:.0f} mA"
                    )
                    raise RuntimeError(
                        "DG5F tracking error exceeded the safety limit: "
                        f"{JOINT_NAMES[worst]}={tracking_error:.3f} rad; "
                        f"target={np.degrees(streamed[worst]):.1f} deg, "
                        "measured="
                        f"{np.degrees(latest_state['q'][worst]):.1f} deg"
                        f"{current_detail}"
                    )
                bridge.send_target(streamed)
                command_was_sent = True
            else:
                tracking_error = 0.0

            if now - last_log_at >= 1.0:
                mode = "SEND" if args.send_to_hand else "DRY"
                if args.send_to_hand and latest_state is not None:
                    pinky_measured_deg = np.degrees(latest_state["q"][17])
                    pinky_error_deg = np.degrees(
                        streamed[17] - latest_state["q"][17]
                    )
                    currents_ma = latest_state.get("currents_ma")
                    pinky_current = (
                        "n/a"
                        if currents_ma is None
                        else f"{currents_ma[17]:.0f} mA"
                    )
                    pinky_feedback = (
                        f" | pinky_j2_measured={pinky_measured_deg:.1f} deg"
                        f" | pinky_j2_error={pinky_error_deg:.1f} deg"
                        f" | pinky_j2_current={pinky_current}"
                    )
                else:
                    pinky_feedback = ""
                print(
                    f"[{mode}] target_deg[0:4]="
                    f"{np.round(np.rad2deg(streamed[:4]), 1).tolist()} | "
                    "thumb_raw_spread/stretch="
                    f"{raw[0]:.1f}/{raw[1]:.1f} deg | "
                    "thumb_j1_preclip="
                    f"{raw[1] + args.thumb_j1_offset_deg:.1f} deg | "
                    "pinky_raw_spread/stretch="
                    f"{raw[16]:.1f}/{raw[17]:.1f} deg | "
                    "pinky_j2_target="
                    f"{np.degrees(streamed[17]):.1f} deg | "
                    f"tracking_error={tracking_error:.3f} rad"
                    f"{pinky_feedback}"
                )
                last_log_at = now

            if args.max_runtime is not None and now - started_at >= args.max_runtime:
                print(f"Reached --max-runtime={args.max_runtime:g}s.")
                break

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    except (RuntimeError, TimeoutError) as error:
        print(f"\nSAFETY STOP: {error}")
    finally:
        if bridge is not None and args.send_to_hand and command_was_sent:
            print("Holding the latest measured DG5F position before shutdown...")
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                state = bridge.receive_latest_state()
                if state is not None:
                    latest_state = state
                if latest_state is not None:
                    bridge.send_target(latest_state["q"])
                time.sleep(0.02)
        if preview is not None:
            preview.close()
            # Give GLFW's passive-viewer thread time to terminate before the
            # Python interpreter unloads MuJoCo.
            time.sleep(0.1)
        if bridge is not None:
            bridge.close()
        receiver.close()


if __name__ == "__main__":
    main()
