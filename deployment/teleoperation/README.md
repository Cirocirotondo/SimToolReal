# MANUS wrist to UR5e arm teleoperation

This package controls **only the UR5e arm**. Finger motion remains under the
official Tesollo ROS 2 MANUS retargeting pipeline in
`/home/duplo/git/tesollo_ros2`.

The MANUS glove does not provide an absolute room position by itself. Absolute
wrist tracking therefore comes from the rigid three-AprilTag fixture already
mounted on the right glove (IDs 0, 2 and 4). MANUS still supplies the finger
data to Tesollo's official software.

## Frames and transforms

The runtime uses this chain:

```text
camera image
    -> T_world_camera                 camera extrinsic calibration
    -> T_world_board                  AprilTag pose estimator
    -> T_board_wrist                  existing glove calibration
    -> T_world_wrist                  tracked human wrist

T_world_robot_base                    robot/world calibration
T_model_robot_base                    MuJoCo/physical-base convention
```

When tracking is armed, the captured wrist pose becomes the UR5 home pose:

```text
p_target = p_home + scale * R_model_world * (p_wrist - p_wrist_initial)
R_target = R_home * (R_wrist_initial^-1 * R_wrist)
```

The first equation prevents the absolute camera position from producing a
robot jump. The second maps relative human wrist rotation onto the robot home
orientation. `--position-only` keeps `R_target = R_home`.

## 1. Mechanical preparation

1. Mount one RealSense rigidly above the workspace.
2. Ensure its color image sees the complete operator wrist workspace.
3. Keep the AprilTag fixture rigid on the glove; changing its geometry
   invalidates `calibration/right_wrist2board_calibration.json`.
4. Fix the 58 mm-square / 45 mm-marker ChArUco board in the world-frame
   location. Do not use the 80/60 mm board with the supplied JSON.

The camera must not move after extrinsic calibration.

## 2. Camera extrinsic calibration

Create the active camera configuration:

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation
cp config/overhead_camera.example.json config/overhead_camera.json
```

Edit `serial_number`. The generic RealSense color wrapper is historically
named `D405`; leave that value even when the connected camera is a D435I.

Run the single-camera calibration using the exact printed-board JSON:

```bash
cd /home/duplo/git/robohand/src/tag-pose-estimation
source /home/duplo/git/robohand-robohand2/.venv/bin/activate

python scripts/calibrate_extrinsics_single_camera.py \
  --camera_config_path \
  /home/duplo/simone/SimToolReal/deployment/teleoperation/config/overhead_camera.json \
  --board_config_path \
  config/calibration_boards/default_intrinsic_calibration_board/charuco_board.json
```

Capture several frames with the board well distributed in the field of view
if the script supports multiple samples. The exported matrix is
`T_world_camera`. Copy the selected result to:

```text
/home/duplo/simone/SimToolReal/deployment/teleoperation/calibration/overhead_T_world_camera.npy
```

The active `overhead_camera.json` already points to this filename.

## 3. World to UR5 base

`config/arm_teleop.json` currently uses:

```text
../simtoolreal_real/calibration/base_pose_robot_ur5e.npy
```

This file contains `T_world_robot_base`. It is reusable only if the world
ChArUco board has not moved relative to the UR5 base. Moving the overhead
camera alone requires only a new camera extrinsic. Moving the world board or
the robot requires repeating robot-base calibration with the end-effector
board.

## 4. Start wrist pose estimation

Create the active pose-estimation configuration:

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation
cp config/manus_overhead_pose_estimation.example.json \
   config/manus_overhead_pose_estimation.json
```

Then:

```bash
cd /home/duplo/git/robohand/src/tag-pose-estimation
source /home/duplo/git/robohand-robohand2/.venv/bin/activate

python scripts/run_pose_estimation.py \
  --config \
  /home/duplo/simone/SimToolReal/deployment/teleoperation/config/manus_overhead_pose_estimation.json
```

It publishes the rigid glove-board pose on `tcp://127.0.0.1:5557`. Before
using IK, move the wrist through the intended workspace and verify:

- no tag-ID swaps;
- no position discontinuities;
- axes move in the expected world directions;
- the pose remains available when one or two tags are occluded.

## 5. IK dry-run

No UR controller is required:

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation

../simtoolreal_real/.venv/bin/python arm_teleop.py \
  --position-only \
  --max-runtime 20
```

Hold the wrist still during the initial 30-sample capture. That position is
assigned to the configured UR5 home. Move slowly and inspect target
coordinates, IK error and joint targets. Then repeat without
`--position-only` to test orientation.

The controller uses damped least-squares IK with MuJoCo Jacobians. No separate
Mink/DAQP installation is required.

## 6. Physical arm, progressively

1. Start the existing low-level UR controller.
2. Send the UR5 to the `home_q_rad` configured in `config/arm_teleop.json`.
3. Keep finger control running in the official Tesollo terminals.
4. For the first physical run, set `position_scale` to `0.3`,
   `maximum_target_speed_m_s` to `0.05`, and use `--position-only`.
5. Keep the physical and human workspaces clear.

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation

../simtoolreal_real/.venv/bin/python arm_teleop.py \
  --send-to-robot \
  --position-only \
  --max-runtime 10
```

The program requires the robot to already be close to home. It captures a
stable initial wrist pose before accepting the exact confirmation
`START SEND`.

After position tracking is validated:

1. increase runtime;
2. increase `position_scale` toward `1.0`;
3. raise Cartesian/joint speed limits gradually;
4. finally remove `--position-only`.

## Safety behavior

Physical commands are disabled by default. When enabled, the controller stops
and holds the latest measured joint position if:

- camera wrist data becomes stale;
- robot state becomes stale;
- a camera pose jumps beyond the configured threshold;
- IK requests a point outside the configured robot-base workspace;
- joint tracking error exceeds the configured limit.

All thresholds are in `config/arm_teleop.json`.

## Offline self-test

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation
../simtoolreal_real/.venv/bin/python tests/self_test.py
```

This tests the initial-pose mapping and a small MuJoCo IK displacement without
opening sockets or commanding hardware.
