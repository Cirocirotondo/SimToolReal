from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ik_solver import Ur5DampedLeastSquaresIk
from arm_teleop import move_robot_to_home
from transforms import (
    RelativeBoardMapper,
    apply_local_origin_offset,
    make_transform,
)


class FakeRobot:
    def __init__(self, initial_q: np.ndarray, home_q: np.ndarray) -> None:
        self.initial_q = initial_q
        self.home_q = home_q
        self.poll_count = 0

    def poll_state(self) -> dict:
        import time

        self.poll_count += 1
        q = self.initial_q if self.poll_count < 3 else self.home_q
        return {"Q": q.tolist(), "_received_at": time.monotonic()}


class FakeStreamer:
    def __init__(self) -> None:
        self.targets: list[np.ndarray] = []

    def set_target(self, target: np.ndarray) -> None:
        self.targets.append(np.asarray(target, dtype=np.float64).copy())


def main() -> None:
    identity = np.eye(4)
    home = make_transform(
        Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix(),
        np.array([0.2, -0.4, 0.3]),
    )
    axis_map = np.diag([-1.0, -1.0, 1.0])
    orientation_axis_map = np.array(
        [
            [0.0, -np.sqrt(0.5), -np.sqrt(0.5)],
            [-1.0, 0.0, 0.0],
            [0.0, np.sqrt(0.5), -np.sqrt(0.5)],
        ]
    )
    np.testing.assert_allclose(
        orientation_axis_map.T @ orientation_axis_map,
        np.eye(3),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.det(orientation_axis_map),
        1.0,
        atol=1e-10,
    )
    origin_offset_board_m = np.array(
        [-0.04548951, 0.05457156, -0.02523359]
    )
    shifted_identity = apply_local_origin_offset(
        identity,
        origin_offset_board_m,
    )
    np.testing.assert_allclose(
        shifted_identity[:3, 3],
        origin_offset_board_m,
        atol=1e-10,
    )
    rotated_board = make_transform(
        Rotation.from_euler("z", np.pi / 2.0).as_matrix(),
        np.array([0.1, 0.2, 0.3]),
    )
    shifted_rotated_board = apply_local_origin_offset(
        rotated_board,
        origin_offset_board_m,
    )
    np.testing.assert_allclose(
        shifted_rotated_board[:3, 3],
        (
            rotated_board[:3, 3]
            + rotated_board[:3, :3] @ origin_offset_board_m
        ),
        atol=1e-10,
    )
    fixed_world_wrist = np.array([0.4, -0.2, 0.6])
    board_rotating_about_wrist = rotated_board.copy()
    board_rotating_about_wrist[:3, 3] = (
        fixed_world_wrist
        - rotated_board[:3, :3] @ origin_offset_board_m
    )
    recovered_world_wrist = apply_local_origin_offset(
        board_rotating_about_wrist,
        origin_offset_board_m,
    )
    np.testing.assert_allclose(
        recovered_world_wrist[:3, 3],
        fixed_world_wrist,
        atol=1e-10,
    )
    mapper = RelativeBoardMapper(
        initial_world_board=identity,
        home_model_ee=home,
        translation_axis_map=axis_map,
        orientation_axis_map=orientation_axis_map,
        position_scale=1.0,
        track_orientation=True,
    )
    np.testing.assert_allclose(mapper.target(identity), home, atol=1e-10)
    moved = identity.copy()
    moved[:3, 3] = [0.03, -0.02, 0.01]
    moved[:3, :3] = Rotation.from_euler("z", 0.1).as_matrix()
    mapped = mapper.target(moved)
    np.testing.assert_allclose(
        mapped[:3, 3],
        home[:3, 3] + axis_map @ moved[:3, 3],
        atol=1e-10,
    )
    forward = identity.copy()
    forward[0, 3] = -0.01
    np.testing.assert_allclose(
        mapper.target(forward)[:3, 3],
        home[:3, 3] + np.array([0.01, 0.0, 0.0]),
        atol=1e-10,
    )
    left = identity.copy()
    left[1, 3] = 0.01
    np.testing.assert_allclose(
        mapper.target(left)[:3, 3],
        home[:3, 3] + np.array([0.0, -0.01, 0.0]),
        atol=1e-10,
    )
    for axis_index in range(3):
        rotated = identity.copy()
        rotation_vector = np.zeros(3)
        rotation_vector[axis_index] = 0.1
        rotated[:3, :3] = Rotation.from_rotvec(
            rotation_vector
        ).as_matrix()
        mapped_relative = (
            home[:3, :3].T @ mapper.target(rotated)[:3, :3]
        )
        expected_rotation_vector = orientation_axis_map @ rotation_vector
        np.testing.assert_allclose(
            Rotation.from_matrix(mapped_relative).as_rotvec(),
            expected_rotation_vector,
            atol=1e-10,
        )

    model_path = (
        ROOT.parent
        / "simtoolreal_real"
        / "assets"
        / "universal_robots_ur5e"
        / "scene.xml"
    )
    ik = Ur5DampedLeastSquaresIk(
        model_path=model_path,
        end_effector_body="wrist_3_link",
        damping=0.03,
        position_gain=8.0,
        orientation_gain=8.0,
        maximum_joint_velocity_rad_s=0.8,
    )
    q = np.array([-1.5708, -1.05, 1.95, -0.9, 1.571, -1.571])
    fake_robot = FakeRobot(q + 0.1, q)
    fake_streamer = FakeStreamer()
    final_state = move_robot_to_home(
        fake_robot,
        fake_streamer,
        q,
        timeout_s=0.5,
        tolerance_deg=0.1,
        settle_s=0.02,
        maximum_state_age_s=0.1,
    )
    np.testing.assert_allclose(final_state["Q"], q)
    np.testing.assert_allclose(fake_streamer.targets[-1], q)

    target = ik.forward(q)
    target[:3, 3] += np.array([0.01, 0.0, 0.0])
    initial_error = np.linalg.norm(target[:3, 3] - ik.forward(q)[:3, 3])
    for _ in range(60):
        q, _ = ik.step(q, target, 1.0 / 60.0)
    final_error = np.linalg.norm(target[:3, 3] - ik.forward(q)[:3, 3])
    if not final_error < initial_error * 0.2:
        raise AssertionError(
            f"IK did not converge enough: {initial_error} -> {final_error}"
        )

    q_orientation = np.array(
        [-1.5708, -1.05, 1.95, -0.9, 1.571, -1.571]
    )
    orientation_target = ik.forward(q_orientation)
    orientation_target[:3, :3] = (
        Rotation.from_rotvec(np.deg2rad([0.0, 0.0, 20.0])).as_matrix()
        @ orientation_target[:3, :3]
    )
    initial_orientation_error = np.deg2rad(20.0)
    for _ in range(25):
        q_orientation, diagnostics = ik.step(
            q_orientation, orientation_target, 1.0 / 50.0
        )
    final_orientation_error = diagnostics.orientation_error_rad
    if not final_orientation_error < np.deg2rad(2.0):
        raise AssertionError(
            "IK orientation did not converge enough in 0.5 s: "
            f"{np.rad2deg(initial_orientation_error):.2f} -> "
            f"{np.rad2deg(final_orientation_error):.2f} deg"
        )
    print(
        "teleoperation self-test: OK "
        f"(IK position error {initial_error * 1000:.2f} -> "
        f"{final_error * 1000:.2f} mm; orientation error "
        f"{np.rad2deg(initial_orientation_error):.2f} -> "
        f"{np.rad2deg(final_orientation_error):.2f} deg in 0.5 s)"
    )


if __name__ == "__main__":
    main()
