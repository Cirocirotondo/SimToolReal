# Demonstration motion imitation

`SimToolRealMotionImitation` is a separate DeepMimic-style task for the UR5e
and right DG5F. It tracks the robot motion in one teleoperation recording and
does not observe or reward the object.

## Start training

From the repository root:

```bash
.venv/bin/python isaacgymenvs/launch_training.py \
  --training-preset motion_imitation \
  --custom-experiment-name motion_imitation_demo_20260727_152551 \
  --num-envs 4800
```

Headless execution, Reference State Initialization, Weights & Biases logging,
and periodic video capture are enabled by default. The default W&B project is
`simtoolreal`, the entity is `simonecirelli-eth`, and the group is
`motion_imitation`. Training uses 4800 parallel environments, a rollout horizon
of 16 steps, and a minibatch of 76800 samples, matching the previous
SimToolReal training setup. The launcher appends a timestamp to the experiment
name and creates its output directory automatically.

Select another recording with a Hydra override:

```bash
.venv/bin/python isaacgymenvs/train.py \
  task=SimToolRealMotionImitation \
  task.env.demonstration=deployment/teleoperation/demonstrations/demo_NAME.npz
```

The default task configuration is
`isaacgymenvs/cfg/task/SimToolRealMotionImitation.yaml`.

## Time and interpolation

Each environment owns a normalized phase in `[0, 1]`. At 60 Hz it advances by

```text
phase_delta = (1 / 60) / demonstration_duration
```

The loader uses the recording's actual timestamps rather than assuming exact
50 Hz spacing. Joint positions and palm positions use linear interpolation;
palm orientation uses shortest-path quaternion SLERP.

At reset, Reference State Initialization samples a phase uniformly and sets
the arm and hand joint positions and velocities from the interpolated
reference. Values marginally outside the simulation URDF limits are clamped.

## Observation and action

The 101-dimensional observation is:

```text
joint_pos (26), joint_vel (26), previous_targets (26),
palm_pos (3), palm_rot_xyzw (4),
fingertips_relative_to_palm (15), phase (1)
```

The first six actions request palm-center translation and rotation. The
operational-space Jacobian includes the configured 16 cm wrist-to-palm offset.
The other twenty actions are relative DG5F joint commands.

## Reference frames

The physical demonstration is transformed into the Isaac world using the
configurable yaw and position offsets. The measured EE pose is then translated
to the virtual palm center. Defaults are:

```yaml
demonstrationWorldYawOffsetDeg: 180.0
demonstrationWorldPositionOffset: [0.0, 0.6, 0.0]
demonstrationEeToPalmOffset: [0.0, 0.0, 0.16]
handMountYawOffsetDeg: 60.0
palmOrientationOffsetQuatXyzw: [0.0, 0.0, 0.5, 0.8660254037844386]
demonstrationEeToPalmQuatXyzw: [0.0, 0.0, 0.5, 0.8660254037844386]
```

The quaternion is the physical `+60°` DG5F mounting yaw around the UR wrist's
local Z axis. Motion imitation uses a dedicated combined URDF carrying the same
fixed transform. The legacy combined URDF remains unchanged so existing
non-imitation policies preserve their original kinematics.

Legacy motion-imitation run configs are migrated to this transform when they
are loaded for evaluation. Their checkpoints can still be inspected, but a
policy trained before this correction saw the wrist quaternion and identity
hand mount; retraining is recommended for a physically consistent policy.

## Reward and termination

The positive imitation reward is

```text
r = wp exp(-kp ||p - p*||²)
  + wR exp(-kR angle(R, R*)²)
  + wh exp(-kh ||q_hand - q_hand*||²)
```

Action-magnitude and consecutive-action-difference penalties are added
separately for arm and hand. An episode ends at phase 1 or early when palm
position, palm orientation, or hand pose exceeds its configured threshold.

## W&B metrics

The task reports the weighted position, orientation, hand-pose, and combined
imitation rewards, all four action penalties, and total reward under
`reward_step/*`. Episode totals use `episode_cumulative/*`; because RSI produces
episodes of different lengths, `episode_mean_per_step/*` is the preferred
episode-level comparison.

Tracking errors and termination causes are reported under `imitation/*`.
Action and action-delta RMS values, plus the operational-space joint-delta
clipping fraction, are reported under `control/*`.

## Interactive evaluation

Use the dedicated evaluator with the `config.yaml` and a policy checkpoint from
the same training run:

```bash
.venv/bin/python dextoolbench/eval_imitation_interactive.py \
  --config-path train_dir/.../runs/00_EXPERIMENT/config.yaml \
  --checkpoint-path train_dir/.../runs/00_EXPERIMENT/nn/00_EXPERIMENT.pth
```

Open the URL printed in the terminal (by default
`http://localhost:8080`), click **Load Policy**, then run an episode. The solid
robot is the deterministic policy rollout; the translucent green robot is the
interpolated demonstration reference at the same phase. The panel shows phase,
reference time, tracking errors, reward components, episode return, and the
termination reason.

The initial phase can be chosen with the slider or sampled randomly from the
configured RSI interval. Disable **Run at 60 Hz** to evaluate as fast as the
simulator allows. Optional per-episode reward CSVs and plots can be enabled
with:

```bash
  --plot-rewards --reward-plot-dir eval_reward_plots/motion_imitation
```
