from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ik_solver import Ur5DampedLeastSquaresIk
from transforms import RelativeBoardMapper, make_transform


def main() -> None:
    identity = np.eye(4)
    home = make_transform(
        Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix(),
        np.array([0.2, -0.4, 0.3]),
    )
    mapper = RelativeBoardMapper(
        initial_world_board=identity,
        home_model_ee=home,
        translation_axis_map=np.eye(3),
        position_scale=1.0,
        track_orientation=True,
    )
    np.testing.assert_allclose(mapper.target(identity), home, atol=1e-10)
    moved = identity.copy()
    moved[:3, 3] = [0.03, -0.02, 0.01]
    moved[:3, :3] = Rotation.from_euler("z", 0.1).as_matrix()
    mapped = mapper.target(moved)
    np.testing.assert_allclose(
        mapped[:3, 3], home[:3, 3] + moved[:3, 3], atol=1e-10
    )
    forward = identity.copy()
    forward[0, 3] = -0.01
    np.testing.assert_allclose(
        mapper.target(forward)[:3, 3],
        home[:3, 3] + np.array([-0.01, 0.0, 0.0]),
        atol=1e-10,
    )
    left = identity.copy()
    left[1, 3] = 0.01
    np.testing.assert_allclose(
        mapper.target(left)[:3, 3],
        home[:3, 3] + np.array([0.0, 0.01, 0.0]),
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
        position_gain=4.0,
        orientation_gain=2.0,
        maximum_joint_velocity_rad_s=0.5,
    )
    q = np.array([-1.5708, -1.05, 1.95, -0.9, 1.571, -1.571])
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
    print(
        "teleoperation self-test: OK "
        f"(IK position error {initial_error * 1000:.2f} -> "
        f"{final_error * 1000:.2f} mm)"
    )


if __name__ == "__main__":
    main()
