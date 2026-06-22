from pathlib import Path
import json
import time

import numpy as np
import zmq


HERE = Path(__file__).parent
REPO_ROOT = HERE.parent


def first_existing_path(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


CONFIG_PATH = first_existing_path(
    HERE / "pc_ur_new.json",
    REPO_ROOT / "src" / "robot_ipc_control" / "controller" / "pc_ur_new.json",
)

PRINT_PERIOD_S = 0.5


def format_array(values, precision=4):
    array = np.array(values, dtype=float)
    return np.array2string(array, precision=precision, suppress_small=True)


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(f"tcp://127.0.0.1:{config['publisher_port']}")

    print(f"Listening for robot state on tcp://127.0.0.1:{config['publisher_port']}")
    print("Press Ctrl+C to stop.")

    last_print_time = 0.0
    last_receive_time = None

    try:
        while True:
            try:
                state = socket.recv_json(flags=zmq.NOBLOCK)
                last_receive_time = time.monotonic()
            except zmq.Again:
                state = None

            now = time.monotonic()
            if now - last_print_time >= PRINT_PERIOD_S:
                last_print_time = now

                if state is None:
                    if last_receive_time is None:
                        print("Waiting for first robot state...")
                    else:
                        age = now - last_receive_time
                        print(f"No new robot state for {age:.2f} s")
                else:
                    q = state.get("Q", [])
                    print()
                    print(f"timestamp_ms: {state.get('timestamp_ms')}")
                    print(f"is_truncated: {state.get('is_truncated')}")
                    print(f"Q rad: {format_array(q)}")
                    print(f"Q deg: {format_array(np.rad2deg(q), precision=2)}")
                    print(f"Qd:    {format_array(state.get('Qd', []))}")
                    print(f"pos:   {format_array(state.get('pos', []))}")
                    print(f"vel:   {format_array(state.get('vel', []))}")
                    print(f"force: {format_array(state.get('force', []))}")

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()
        print("Stopped.")


if __name__ == "__main__":
    main()
