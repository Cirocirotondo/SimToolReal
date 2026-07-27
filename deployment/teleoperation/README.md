# MANUS wrist to UR5e arm teleoperation

This package controls **only the UR5e arm**. Finger motion remains under the
official Tesollo ROS 2 MANUS retargeting pipeline in
`/home/duplo/git/tesollo_ros2`.

- This pipeline tracks the position of the right MANUS glove, which mounts a three-AprilTag board (IDs 0,2 and 4).
- The world frame is given by the 5x7 Charuco Board `/home/duplo/simone/SimToolReal/deployment/teleoperation/calibration/world_charuco_5x7_35mm_26mm/charuco_board.json`.
- The board should be used with the x (red axes) pointing toward left of the glove user and the y (green axes) pointing toward the glove user. Indeed, this is the mapping used in the configuration.
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

<details>
<summary>Manus board-wrist calibration</summary>
This calibration is needed to know what is the relative pose between the Aruco board on the wrist and the hand skeleton.
Without it, the rotation of the hand is not correctly mapped to the robot.
By default the script uses this one `/home/duplo/git/robohand-robohand2/src/tag-pose-estimation/config/calibration_gloves/right_wrist2board_calibration.json`.
You can easily continue to use this, as long as the board on the wrist is not removed or changes position by a lot.
</details>


## Physical arm

The normal workflow needs two terminals.

In terminal 1, start the existing UR5 low-level arm controller.
```bash
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real
./impedance_controller pc_ur_new.json
```

In terminal 2:

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation

../simtoolreal_real/.venv/bin/python arm_teleop.py --send-to-robot
```

The script first moves the arm to the configured home joint position
`[-1.5708, -1.05, 1.95, -0.9, 1.571, -1.571]`. It waits until the measured
joints have remained within the configured tolerance before starting the
camera pose estimator and the MANUS countdown.

Keep the glove at the desired neutral position during that countdown. When
`GO` appears, relative motion begins. `Ctrl+C` stops and holds the arm. Finger
control is independent and can remain in the official Tesollo ROS terminals.

If a pose estimator is already running, add
`--no-start-pose-estimator`; otherwise it is started and stopped
automatically.

<details>
<summary>Offline self-test</summary>

Physical commands are disabled by default. When enabled, the controller stops
and holds the latest measured joint position if:

- the robot does not reach home within the configured timeout;
- camera board data becomes stale;
- robot state becomes stale;
- a camera pose jumps beyond the configured threshold;
- IK requests a point outside the configured MuJoCo-model workspace;
- joint tracking error exceeds the configured limit.

All thresholds are in `config/arm_teleop.json`.
</details>

<details>
<summary>Safety behavior</summary>

```bash
cd /home/duplo/simone/SimToolReal/deployment/teleoperation
../simtoolreal_real/.venv/bin/python tests/self_test.py
```

This tests the initial-pose mapping and a small MuJoCo IK displacement without
opening sockets or commanding hardware.
</details>

<details>
<summary>IK dry-run</summary>

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
</details>
