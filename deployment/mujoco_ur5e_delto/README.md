# MuJoCo UR5e + Delto Deployment

No-ROS MuJoCo runner for a SimToolReal policy trained on the UR5e + Tesollo
Delto DG5F setup.

The right hand is used by default. Add `--hand-side left` when running a
legacy policy trained with the left hand.

Run from the SimToolReal repo root with the MuJoCo environment:

```bash
source /home/simone/.venv/bin/activate
python deployment/mujoco_ur5e_delto/run_policy_no_ros.py \
  --config-path /path/to/config.yaml \
  --checkpoint-path /path/to/model.pth \
  --object-name cube
```

Use `--object-name hammer` for the hammer primitive, or `--object-name cuboid_5x5x15`
for the 5 x 5 x 15 cm training parallelepiped. Add `--max-steps 300` for a bounded
smoke test, and `--no-enable-viewer` for headless execution.

By default, the table height is read from the policy config. For train_7 and the
new evaluation setup this is the near-zero table convention:

```bash
source /home/simone/.venv/bin/activate
python deployment/mujoco_ur5e_delto/run_policy_no_ros.py \
  --config-path train_dir/simtoolreal/2026-06-10/train_07_sim2real_resume_resume_2026-06-10_15-00-42/runs/00_train_07_sim2real_resume_resume_2026-06-10_15-00-42/config.yaml \
  --checkpoint-path train_dir/simtoolreal/2026-06-10/train_07_sim2real_resume_resume_2026-06-10_15-00-42/runs/00_train_07_sim2real_resume_resume_2026-06-10_15-00-42/best/model.pth \
  --object-name cube
```

For the B8 5 x 5 x 15 cm parallelepiped policy:

```bash
python deployment/mujoco_ur5e_delto/run_policy_no_ros.py \
  --config-path train_dir/simtoolreal/2026-07-03/training_b8_parallelepiped_2026-07-03_18-13-35/runs/00_training_b8_parallelepiped_2026-07-03_18-13-35/config.yaml \
  --checkpoint-path train_dir/simtoolreal/2026-07-03/training_b8_parallelepiped_2026-07-03_18-13-35/runs/00_training_b8_parallelepiped_2026-07-03_18-13-35/last/model.pth \
  --object-name cuboid_5x5x15
```

This places the table body center at `z=-0.125`, so the table top is near
`z=0.025`, matching the train_7 / evaluation convention. The object starts at
`z=0.125`, using `tableObjectZOffset=0.25` from the config.

The MuJoCo viewer displays local red/green/blue axes on both the real cube and
the visual-only translucent green goal cube, matching the Gazebo sim2sim
goal-marker idea while making the cube orientation easier to inspect.

To evaluate with the original high-table scene instead, pass
`--scene-height high_table`:

```bash
python deployment/mujoco_ur5e_delto/run_policy_no_ros.py \
  --config-path train_dir/simtoolreal/2026-05-22/train_5_cube_2026-05-22_17-27-47/runs/00_train_5_cube_2026-05-22_17-27-47/config.yaml \
  --checkpoint-path train_dir/simtoolreal/2026-05-22/train_5_cube_2026-05-22_17-27-47/runs/00_train_5_cube_2026-05-22_17-27-47/nn/last_00_train_5_cube_2026-05-22_17-27-47_ep_120000_rew_10669.215.pth \
  --object-name cube \
  --scene-height high_table
```

`--scene-height default`, `--scene-height train7`, and `--scene-height
from-config` all read `tableResetZ` from the policy config, falling back to
`-0.125` if the config does not define it.

This backend reads MuJoCo state directly, builds the same 131-D observation used
by the Gazebo ROS 2 policy node, runs `deployment.rl_player.RlPlayer`, converts
the 26-D normalized action into joint position targets, and writes those targets
to MuJoCo position actuators.
