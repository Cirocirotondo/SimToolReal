# FILES TO BE CHANGED:

### SimToolReal.yaml - isaacgymenvs/cfg/task/SimToolReal.yaml

In SimToolReal.yaml, the main robot switch point is:

env.asset.robot
That path currently points to the iiwa+sharpa URDF.
For a new robot, this is the first line you change.

But in this repo, for a real robot swap you’ll likely also revisit these nearby settings in the same file:

env.dofSpeedScale
env.handMovingAverage
env.armMovingAverage
env.stiffnessScale
env.forceLimitScale
env.startArmHigher
env.kukaActionsPenaltyScale (name says kuka, but used generically in reward)
env.handActionsPenaltyScale
env.stateList / env.obsList (only if obs structure changes)
Important caveat: changing only SimToolReal.yaml won’t be enough here. Code is still hardcoded for iiwa/sharpa in:

isaacgymenvs/tasks/simtoolreal/env.py
isaacgymenvs/utils/observation_action_utils_sharpa.py
evaluation scripts (dextoolbench/eval.py, dextoolbench/eval_interactive.py)
So treat env.asset.robot as the entry change, not the full migration.

---

## ROBOT MIGRATION CHECKLIST (IN ORDER)

Use this as a practical reminder when switching robot arm/hand in this repo.

### 1) Asset Layer (URDF + meshes)

Files/folders:
- `assets/urdf/...` (robot URDF and mesh paths)

What to change/check:
- Add a standalone `.urdf` that Isaac Gym can load directly.
- Ensure mesh paths are valid from repo-local assets (avoid unresolved `package://` references unless remapped).
- If using an arm+hand combo, ensure the fixed mount joint is correct (parent link, xyz/rpy offset).
- Verify the URDF contains:
  - expected actuated joints,
  - expected end-effector link,
  - valid inertials and collision meshes.

Quick sanity checks:
- Joint count in URDF matches what code/config expects.
- End-effector link name exists in URDF body names.

### 2) Task Config Layer

File:
- `isaacgymenvs/cfg/task/SimToolReal.yaml`

What to change/check:
- `env.asset.robot` -> new URDF path.
- Revisit robot-sensitive tuning:
  - `env.dofSpeedScale`
  - `env.handMovingAverage`
  - `env.armMovingAverage`
  - `env.stiffnessScale`
  - `env.forceLimitScale`
  - `env.startArmHigher`
  - `env.kukaActionsPenaltyScale`
  - `env.handActionsPenaltyScale`
- If observation layout changed, revisit:
  - `env.stateList`
  - `env.obsList`

### 3) Core Task Logic Layer (hardcoded assumptions)

File:
- `isaacgymenvs/tasks/simtoolreal/env.py`

What to change/check:
- Arm DOF assumptions (currently iiwa-oriented).
- End-effector link checks (currently `iiwa14_link_7` oriented).
- Finger link naming assumptions for fingertip extraction.
- Action slicing assumptions (arm index range vs hand index range).
- Reset/control code using fixed arm-hand split boundaries.
- Any direct calls to robot-specific helper functions.

Common failure signatures:
- Asset DOF count assertion mismatch.
- Missing link assertion (EE/fingertip names not found).
- Shape mismatch in target/action tensors.

### 4) Observation/Action Utility Layer

File:
- `isaacgymenvs/utils/observation_action_utils_sharpa.py`

What to change/check:
- Joint name list and order.
- Joint limit arrays and expected shapes.
- Robot name handling in `create_urdf_object(...)`.
- FK link list and palm/EE frame link references.
- Observation assembly assumptions on total DOFs.
- Action-to-target logic (arm slice, hand slice, EMA/clamp boundaries).

Common failure signatures:
- Observation shape mismatch vs policy input.
- Action shape mismatch vs env action space.
- FK link lookup errors.

### 5) Self-Collision Adjacency Layer

File:
- `isaacgymenvs/tasks/simtoolreal/adjacent_links.py`

What to change/check:
- Update link adjacency maps to match new link names/topology.
- Ensure all referenced links exist in new robot asset.

Common failure signatures:
- Assertions on missing links in adjacency mapping.
- Incorrect collision behavior if adjacency is stale.

### 6) Evaluation and Visualization Layer

Files:
- `dextoolbench/eval.py`
- `dextoolbench/eval_interactive.py`
- `deployment/rl_player.py` (dimension expectations)

What to change/check:
- Hardcoded robot URDF path for Viser rendering.
- Hardcoded dimensions (e.g. 29 -> new total DOF).
- End-effector link/frame assumptions in visual/debug code.
- Any hardcoded action/observation dimension passed to player.

### 7) Optional Deployment Nodes (if used)

Files:
- `deployment/rl_policy_node.py`
- `deployment/visualization_node.py`
- other `deployment/*.py` that parse joint states/targets

What to change/check:
- Joint ordering contracts with ROS topics.
- Robot-specific topic assumptions and kinematic references.
- Any arm-specific names in transforms.

---

## QUICK VALIDATION PLAN AFTER CHANGES

1. Load env with `numEnvs=1` and random actions.
2. Assert no missing-link or DOF-count errors.
3. Run one deterministic policy rollout in `dextoolbench/eval.py`.
4. Run interactive eval (`dextoolbench/eval_interactive.py`) and confirm rendering + controls.
5. Launch training with reduced `num_envs` first to validate tensor shapes.

---

## IMPORTANT NOTES

- This repo is currently optimized for iiwa+Sharpa assumptions in several files.
- Changing only `SimToolReal.yaml` is not sufficient for a true robot migration.
- Prefer freezing a standalone URDF for training (avoid runtime xacro/ROS dependency in training scripts).