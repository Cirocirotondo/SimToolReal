#!/usr/bin/env python3
"""Slow, isolated physical-motion test for DG5F joint rj_dg_5_2.

The script talks to dg5f_policy_ros_bridge.py over localhost UDP.  It holds
the 19 other joints at their measured starting positions, ramps rj_dg_5_2 by
a small relative angle, holds it briefly, and then returns to the start.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Optional

import numpy as np


HAND_DOF = 20
PINKY_J2_INDEX = 17
PINKY_J2_NAME = "rj_dg_5_2"


class BridgeClient:
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
        self.latest: Optional[dict] = None
        self.sequence = 0

    def receive_latest(self) -> Optional[dict]:
        while True:
            try:
                payload, _ = self.state_socket.recvfrom(65535)
            except BlockingIOError:
                return self.latest
            try:
                message = json.loads(payload)
                q = np.asarray(message["positions"], dtype=np.float64)
                qd = np.asarray(message["velocities"], dtype=np.float64)
            except (KeyError, TypeError, ValueError, UnicodeDecodeError,
                    json.JSONDecodeError):
                continue
            if message.get("type") != "hand_state":
                continue
            if q.shape != (HAND_DOF,) or qd.shape != (HAND_DOF,):
                continue
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                continue

            currents = message.get("currents_ma")
            if currents is not None:
                currents = np.asarray(currents, dtype=np.float64)
                if currents.shape != (HAND_DOF,) or not np.all(np.isfinite(currents)):
                    currents = None
            self.latest = {
                "q": q,
                "qd": qd,
                "currents_ma": currents,
                "received_at": time.monotonic(),
            }

    def send(self, positions: np.ndarray) -> None:
        self.sequence += 1
        message = {
            "type": "hand_target",
            "sequence": self.sequence,
            "positions": np.asarray(positions, dtype=np.float64).tolist(),
        }
        self.command_socket.sendto(
            json.dumps(message, separators=(",", ":")).encode(),
            self.command_endpoint,
        )

    def close(self) -> None:
        self.state_socket.close()
        self.command_socket.close()


def wait_for_state(client: BridgeClient, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.receive_latest()
        if state is not None:
            return state
        time.sleep(0.01)
    raise TimeoutError(
        "No DG5F state received. Start dg5f_policy_ros_bridge.py first."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move only physical DG5F joint rj_dg_5_2 by a small angle."
    )
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=5.0,
        help="Relative test displacement; signed, default: +5 degrees.",
    )
    parser.add_argument(
        "--velocity-deg-s",
        type=float,
        default=5.0,
        help="Target ramp speed, default: 5 degrees/second.",
    )
    parser.add_argument(
        "--hold-seconds", type=float, default=1.0,
        help="Time at the displaced target before returning, default: 1 second.",
    )
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--state-address", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=5563)
    parser.add_argument("--command-address", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=5562)
    parser.add_argument("--state-timeout", type=float, default=0.25)
    parser.add_argument(
        "--max-tracking-error-deg", type=float, default=8.0,
        help="Abort threshold for rj_dg_5_2, default: 8 degrees.",
    )
    parser.add_argument(
        "--max-current-ma", type=float, default=220.0,
        help="Abort when |rj_dg_5_2 current| exceeds this value.",
    )
    parser.add_argument(
        "--skip-confirmation", action="store_true",
        help="Skip the SEND prompt; intended only for deliberate repeat tests.",
    )
    args = parser.parse_args()
    if not 0.0 < abs(args.delta_deg) <= 10.0:
        parser.error("--delta-deg magnitude must be in (0, 10] degrees")
    for name in (
        "velocity_deg_s", "hold_seconds", "control_hz", "state_timeout",
        "max_tracking_error_deg", "max_current_ma",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    client = BridgeClient(
        state_address=args.state_address,
        state_port=args.state_port,
        command_address=args.command_address,
        command_port=args.command_port,
    )
    command_sent = False
    latest: Optional[dict] = None

    try:
        print(
            f"Waiting for DG5F state on udp://{args.state_address}:"
            f"{args.state_port}..."
        )
        latest = wait_for_state(client, 5.0)
        start = latest["q"].copy()
        start_deg = float(np.degrees(start[PINKY_J2_INDEX]))
        target = start.copy()
        target[PINKY_J2_INDEX] += np.radians(args.delta_deg)
        target_deg = float(np.degrees(target[PINKY_J2_INDEX]))

        print(f"Measured {PINKY_J2_NAME}: {start_deg:.2f} deg")
        print(
            f"Test target: {target_deg:.2f} deg "
            f"(relative change {args.delta_deg:+.2f} deg)"
        )
        print(
            "Only rj_dg_5_2 will change; the other 19 joints will be held "
            "at their measured starting positions."
        )
        print(
            f"Motion: {args.velocity_deg_s:g} deg/s, hold: "
            f"{args.hold_seconds:g} s, then return."
        )
        if not args.skip_confirmation:
            answer = input(
                "Keep the hand clear and type SEND to run this physical test: "
            )
            if answer.strip() != "SEND":
                print("Aborted; no command was sent.")
                return

        period = 1.0 / args.control_hz
        step = np.radians(args.velocity_deg_s) * period
        commanded = start.copy()
        phase = "out"
        hold_started: Optional[float] = None
        last_log = 0.0

        while True:
            now = time.monotonic()
            state = client.receive_latest()
            if state is not None:
                latest = state
            if latest is None or now - latest["received_at"] > args.state_timeout:
                raise RuntimeError("DG5F state became stale")

            if phase == "out":
                remaining = target[PINKY_J2_INDEX] - commanded[PINKY_J2_INDEX]
                increment = float(np.clip(remaining, -step, step))
                commanded[PINKY_J2_INDEX] += increment
                if abs(remaining) <= step:
                    commanded[PINKY_J2_INDEX] = target[PINKY_J2_INDEX]
                    phase = "hold"
                    hold_started = now
            elif phase == "hold":
                if now - float(hold_started) >= args.hold_seconds:
                    phase = "return"
            elif phase == "return":
                remaining = start[PINKY_J2_INDEX] - commanded[PINKY_J2_INDEX]
                increment = float(np.clip(remaining, -step, step))
                commanded[PINKY_J2_INDEX] += increment
                if abs(remaining) <= step:
                    commanded[PINKY_J2_INDEX] = start[PINKY_J2_INDEX]
                    client.send(commanded)
                    command_sent = True
                    print("Returned to the measured starting target.")
                    break

            measured = float(latest["q"][PINKY_J2_INDEX])
            error = float(commanded[PINKY_J2_INDEX] - measured)
            current = None
            if latest["currents_ma"] is not None:
                current = float(latest["currents_ma"][PINKY_J2_INDEX])

            if abs(error) > np.radians(args.max_tracking_error_deg):
                raise RuntimeError(
                    f"{PINKY_J2_NAME} tracking error {np.degrees(error):+.2f} "
                    f"deg exceeds {args.max_tracking_error_deg:g} deg"
                )
            if current is not None and abs(current) > args.max_current_ma:
                raise RuntimeError(
                    f"{PINKY_J2_NAME} current {current:+.0f} mA exceeds "
                    f"{args.max_current_ma:g} mA"
                )

            client.send(commanded)
            command_sent = True
            if now - last_log >= 0.2:
                current_text = "n/a" if current is None else f"{current:+.0f} mA"
                command_deg = np.degrees(commanded[PINKY_J2_INDEX])
                print(
                    f"[{phase:6s}] target={command_deg:+.2f} deg | "
                    f"measured={np.degrees(measured):+.2f} deg | "
                    f"error={np.degrees(error):+.2f} deg | current={current_text}"
                )
                last_log = now
            time.sleep(period)

    except (KeyboardInterrupt, RuntimeError, TimeoutError) as exc:
        print(f"\nSAFETY STOP: {exc}")
    finally:
        if command_sent and latest is not None:
            # Ask the bridge to hold the newest measured posture rather than
            # leaving a displaced target active while this process exits.
            hold = latest["q"].copy()
            for _ in range(5):
                client.send(hold)
                time.sleep(0.02)
            print("Holding the latest measured DG5F posture.")
        client.close()


if __name__ == "__main__":
    main()
