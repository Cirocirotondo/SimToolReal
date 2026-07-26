# Real-world policy deployment

This directory contains the runtime used to apply a trained SimToolReal policy
to the physical UR5e and Tesollo DG5F.

## Main components

- `full_policy_controller.py`: runs one policy inference and splits the six arm
  targets from the twenty hand targets.
- `dg5f_policy_ros_bridge.py`: exchanges DG5F state and targets between the
  Python policy process and ROS 2.
- `ur5_policy_arm_controller.py`: UR5 communication and safety utilities used
  by the full controller.
- `hand_policy_controller.py`: hand-only policy test utility.
- `pc_ur_new.json`: low-level UR5 controller configuration.

The policy controllers also use:

- `deployment/mujoco_ur5e_delto/`
- `deployment/rl_player.py`
- `rl_games/`
- the UR5e/DG5F descriptions under `assets/urdf/`

Training configurations and checkpoints remain under the main repository's
`train_dir/`; they are passed to the controller with `--config-path` and
`--checkpoint-path`.

## Typical dry run

From the repository root:

```bash
cd deployment/simtoolreal_real

CONFIG=/absolute/path/to/config.yaml
CHECKPOINT=/absolute/path/to/model.pth

uv run python full_policy_controller.py \
  --config-path "$CONFIG" \
  --checkpoint-path "$CHECKPOINT"
```

This does not authorize physical commands. Enable arm and hand output only
after the corresponding low-level controllers and state streams have been
verified.

## Optional components

- `pose_estimation/` contains the camera/robot calibration utilities.
- `calibration/` contains the current machine-specific transforms.
- `experimental/manus_teleop/` contains the earlier custom MANUS-to-DG5F
  controller and the isolated pinky-joint physical test. These are not part of
  policy inference.

The active official Tesollo ROS workspace is:

```text
/home/duplo/git/tesollo_ros2
```

Do not use commands referring to the older `robohand/ros_ws` without first
checking which DG5F generation is connected.
