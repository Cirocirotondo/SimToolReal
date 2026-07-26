# MANUS wrist to UR5e arm teleoperation

This package controls **only the UR5e arm**. Finger motion remains under the
official Tesollo ROS 2 MANUS retargeting pipeline in
`/home/duplo/git/tesollo_ros2`.

- This pipeline tracks the position of the right MANUS glove, which mounts a three-AprilTag board (IDs 0,2 and 4).
- The world frame is given by the 5x7 Charuco Board `/home/duplo/simone/SimToolReal/deployment/teleoperation/calibration/world_charuco_5x7_35mm_26mm/charuco_board.json`.
- The board should be used with the x (red axes) pointing toward the glove user and the y (green axes) pointing toward left. Indeed, this is the mapping used in the configuration.
- The camera used in the scripts has ID 242322072500. I suggest to put the camera above the scene in order to avoid occlusions.


## Camera extrinsic calibration

Run the single-camera calibration using the exact 5 x 7 printed-board JSON:

```bash
cd /home/duplo/git/robohand/src/tag-pose-estimation
source /home/duplo/git/robohand-robohand2/.venv/bin/activate

python scripts/calibrate_extrinsics_single_camera.py \
  --camera_config_path \
  /home/duplo/simone/SimToolReal/deployment/teleoperation/config/overhead_camera.json \
  --board_config_path \
  /home/duplo/simone/SimToolReal/deployment/teleoperation/calibration/world_charuco_5x7_35mm_26mm/charuco_board.json
```
Capture a frame by pressing `c` and then `Enter`. The calibration will be saved here:

```text
/home/duplo/simone/SimToolReal/deployment/teleoperation/calibration/overhead_T_world_camera.npy
```

The active `overhead_camera.json` already points to this filename.



## How the relative mapping works

No MANUS-board-to-wrist or world-to-UR5-base calibration is used. At startup,
the script detects the glove board and starts a three-second countdown. At the
end of the countdown:

- the current MANUS board pose is assigned to the UR5 home configuration
  `[-1.5708, -1.05, 1.95, -0.9, 1.571, -1.571]`;
- subsequent translations are applied one-to-one to the robot end effector;
- MANUS forward is world `-X` and produces robot model `-X`;
- MANUS left is world `+Y` and produces robot model `+Y`.

The fixed camera extrinsic is still required because it defines these world
axes. The absolute camera position and the initial MANUS position cancel from
the relative motion.

## IK dry-run

No UR controller is required:

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation

../simtoolreal_real/.venv/bin/python arm_teleop.py \
  --position-only \
  --max-runtime 20
```

The script automatically launches the camera pose estimator. Hold the glove
still at the desired neutral pose during the three-second countdown. Move
slowly and inspect target coordinates, IK error and joint targets. Repeat
without `--position-only` to test relative orientation as well.

The controller uses damped least-squares IK with MuJoCo Jacobians. No separate
Mink/DAQP installation is required.

## Physical arm

The normal workflow needs two terminals.

In terminal 1, start the existing UR5 low-level arm controller and move the
robot to the home joint configuration:

```text
[-1.5708, -1.05, 1.95, -0.9, 1.571, -1.571] rad
```

In terminal 2:

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation

../simtoolreal_real/.venv/bin/python arm_teleop.py --send-to-robot
```

Keep the glove at the desired neutral position during the countdown. When
`GO` appears, relative motion begins. `Ctrl+C` stops and holds the arm. Finger
control is independent and can remain in the official Tesollo ROS terminals.

For an initial physical check, use:

```bash
../simtoolreal_real/.venv/bin/python arm_teleop.py \
  --send-to-robot \
  --position-only \
  --max-runtime 10
```

If a pose estimator is already running, add
`--no-start-pose-estimator`; otherwise it is started and stopped
automatically.

## Safety behavior

Physical commands are disabled by default. When enabled, the controller stops
and holds the latest measured joint position if:

- camera board data becomes stale;
- robot state becomes stale;
- a camera pose jumps beyond the configured threshold;
- IK requests a point outside the configured MuJoCo-model workspace;
- joint tracking error exceeds the configured limit.

All thresholds are in `config/arm_teleop.json`.

## Offline self-test

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation
../simtoolreal_real/.venv/bin/python tests/self_test.py
```

This tests the initial-pose mapping and a small MuJoCo IK displacement without
opening sockets or commanding hardware.
