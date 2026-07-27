#!/usr/bin/env python3
"""Passively mirror DG5F ROS state and commands to a local UDP socket."""

from __future__ import annotations

import argparse
import json
import socket
import time

import rclpy
from control_msgs.msg import MultiDOFCommand
from rclpy.node import Node
from sensor_msgs.msg import JointState


RIGHT_JOINT_NAMES = [
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]


class HandRosBridge(Node):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        state_topic: str,
        command_topic: str,
    ) -> None:
        super().__init__("dg5f_demonstration_bridge")
        self.destination = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.measured = [float("nan")] * len(RIGHT_JOINT_NAMES)
        self.commanded = [float("nan")] * len(RIGHT_JOINT_NAMES)
        self.measured_valid = False
        self.commanded_valid = False
        self.create_subscription(
            JointState,
            state_topic,
            self._on_state,
            10,
        )
        self.create_subscription(
            MultiDOFCommand,
            command_topic,
            self._on_command,
            10,
        )
        self.get_logger().info(
            f"Mirroring {state_topic} and {command_topic} to "
            f"udp://{host}:{port}"
        )

    @staticmethod
    def _ordered_values(names, values):
        by_name = dict(zip(names, values))
        return [
            float(by_name.get(name, float("nan")))
            for name in RIGHT_JOINT_NAMES
        ]

    def _publish(self) -> None:
        message = {
            "timestamp": time.time(),
            "monotonic_timestamp": time.monotonic(),
            "joint_names": RIGHT_JOINT_NAMES,
            "hand_q_measured": self.measured,
            "hand_q_commanded": self.commanded,
            "hand_q_measured_valid": self.measured_valid,
            "hand_q_commanded_valid": self.commanded_valid,
        }
        self.socket.sendto(
            json.dumps(message, allow_nan=True).encode("utf-8"),
            self.destination,
        )

    def _on_state(self, message: JointState) -> None:
        self.measured = self._ordered_values(
            message.name,
            message.position,
        )
        self.measured_valid = all(
            value == value for value in self.measured
        )
        self._publish()

    def _on_command(self, message: MultiDOFCommand) -> None:
        self.commanded = self._ordered_values(
            message.dof_names,
            message.values,
        )
        self.commanded_valid = all(
            value == value for value in self.commanded
        )
        self._publish()

    def destroy_node(self) -> bool:
        self.socket.close()
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5564)
    parser.add_argument(
        "--state-topic",
        default="/dg5f_right/joint_states",
    )
    parser.add_argument(
        "--command-topic",
        default="/dg5f_right/rj_dg_pospid/reference",
    )
    args = parser.parse_args()

    rclpy.init()
    node = HandRosBridge(
        host=args.host,
        port=args.port,
        state_topic=args.state_topic,
        command_topic=args.command_topic,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
