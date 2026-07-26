from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Optional

import numpy as np
import zmq


class RobotIo:
    """ZMQ interface to the existing UR low-level controller."""

    def __init__(self, low_level_config_path: str | Path) -> None:
        with Path(low_level_config_path).open(encoding="utf-8") as stream:
            config = json.load(stream)
        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.PUB)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.command_socket.bind(f"tcp://*:{config['socket_port']}")
        self.state_socket = self.context.socket(zmq.SUB)
        self.state_socket.setsockopt(zmq.LINGER, 0)
        self.state_socket.setsockopt(zmq.CONFLATE, 1)
        self.state_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.state_socket.connect(
            f"tcp://127.0.0.1:{config['publisher_port']}"
        )
        self.latest_state: Optional[dict] = None

    def poll_state(self) -> Optional[dict]:
        while True:
            try:
                state = self.state_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return self.latest_state
            q = np.asarray(state.get("Q", []), dtype=np.float64)
            if q.shape[0] >= 6 and np.all(np.isfinite(q[:6])):
                state["_received_at"] = time.monotonic()
                self.latest_state = state

    def request_stop(
        self,
        *,
        repetitions: int = 5,
        interval_s: float = 0.02,
    ) -> None:
        """Reliably request speedStop before closing the PUB socket."""
        for _ in range(repetitions):
            self.command_socket.send_json({"stop": True})
            time.sleep(interval_s)

    def close(self) -> None:
        self.command_socket.close()
        self.state_socket.close()
        self.context.term()


class CommandStreamer:
    def __init__(
        self,
        *,
        command_socket: zmq.Socket,
        frequency_hz: float,
    ) -> None:
        self.command_socket = command_socket
        self.period = 1.0 / frequency_hz
        self.target: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def set_target(self, target: np.ndarray) -> None:
        with self.lock:
            self.target = np.asarray(target, dtype=np.float64).copy()

    def _run(self) -> None:
        next_send = time.monotonic()
        while not self.stop_event.is_set():
            with self.lock:
                target = None if self.target is None else self.target.copy()
            if target is not None:
                self.command_socket.send_json({"target_q": target.tolist()})
            next_send += self.period
            self.stop_event.wait(max(0.0, next_send - time.monotonic()))
            if next_send < time.monotonic() - self.period:
                next_send = time.monotonic()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
