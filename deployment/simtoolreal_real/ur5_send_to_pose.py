from pathlib import Path
import json
import time

import mujoco
import mujoco.viewer
import numpy as np
import zmq
from loop_rate_limiters import RateLimiter


HERE = Path(__file__).parent
REPO_ROOT = HERE.parent


def first_existing_path(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


MODEL_PATH = first_existing_path(
    HERE / "assets" / "universal_robots_ur5e" / "scene.xml",
    REPO_ROOT / "assets" / "universal_robots_ur5e" / "scene.xml",
)
CONFIG_PATH = first_existing_path(
    HERE / "pc_ur_new.json",
    REPO_ROOT / "src" / "robot_ipc_control" / "controller" / "pc_ur_new.json",
)

SEND_RATE_HZ = 50.0
USE_REAL_ROBOT = True

# Joint order:
# shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
HOME_DEG = np.array([-90.0, -68.755, 103.132, -34.377, 90.012, -150])



def make_zmq_sockets(config):
    context = zmq.Context()

    command_socket = context.socket(zmq.PUB)
    command_socket.bind(f"tcp://*:{config['socket_port']}")

    state_socket = context.socket(zmq.SUB)
    state_socket.setsockopt(zmq.CONFLATE, 1)
    state_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    state_socket.connect(f"tcp://127.0.0.1:{config['publisher_port']}")

    return context, command_socket, state_socket


def receive_robot_q(state_socket):
    try:
        state = state_socket.recv_json(flags=zmq.NOBLOCK)
        if "Q" in state:
            return np.array(state["Q"], dtype=float)
    except zmq.Again:
        pass
    return None


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    home_q = np.deg2rad(HOME_DEG)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    data.qpos[:6] = home_q
    mujoco.mj_forward(model, data)

    if USE_REAL_ROBOT:
        context, command_socket, state_socket = make_zmq_sockets(config)
        print(f"Publishing target_q on tcp://*:{config['socket_port']}")
        print(f"Listening for robot state on tcp://127.0.0.1:{config['publisher_port']}")
    else:
        context = None
        command_socket = None
        state_socket = None
        print("Simulation-only mode: not sending commands to the real robot.")

    print(f"Home deg: {HOME_DEG.tolist()}")
    print("Close the MuJoCo viewer or press Ctrl+C to stop.")

    if USE_REAL_ROBOT:
        # Give the C++ subscriber a moment to connect before first command.
        time.sleep(1.0)

    display_q = home_q.copy()
    rate = RateLimiter(frequency=SEND_RATE_HZ, warn=False)

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            while viewer.is_running():

                if USE_REAL_ROBOT:
                    command_socket.send_json({"target_q": home_q.tolist()})

                robot_q = receive_robot_q(state_socket) if USE_REAL_ROBOT else None
                if robot_q is not None and robot_q.shape[0] >= 6:
                    display_q = robot_q[:6]
                else:
                    # If the controller is not publishing state, still show the
                    # commanded motion smoothly in the viewer.
                    display_q = 0.92 * display_q + 0.08 * home_q

                data.qpos[:6] = display_q
                mujoco.mj_forward(model, data)
                viewer.sync()
                rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        if command_socket is not None:
            command_socket.close()
        if state_socket is not None:
            state_socket.close()
        if context is not None:
            context.term()
        print("Stopped.")


if __name__ == "__main__":
    main()
