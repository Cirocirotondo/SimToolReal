# Interactive Evaluation:
Launch: 
```bash 
python dextoolbench/eval_interactive.py   --config-path pretrained_policy/config.yaml   --checkpoint-path pretrained_policy/model.pth  
```
Add `--debug-network` to see all the debug messages (shouldn't be necessary anymore). 


### Troubleshoot: 
The original version had a problem. If you would launch the code you would open the browser window and obtain the error message: `Failed to open a WebSocket connection. See server log for more information.`
The issue was a combination of two problems:

websockets was on an incompatible version (12.0) for viser 1.0.26, which caused:
ModuleNotFoundError: No module named 'websockets.asyncio'
Even after fixing that, on Python 3.8 viser uses Path.is_relative_to() (available in Python 3.9+), so HTTP / serving failed and the browser showed:
“Failed to open a WebSocket connection”
So the server looked alive, but the web UI could not load correctly.

What fixed it
Updated websockets to 13.1
Added a Python 3.8 compatibility shim in eval_interactive.py (for Path.is_relative_to, before importing viser)
After that, logs show HTTP 200 and Connection opened, so it is resolved.

# Launch Training
```bash 
source ../.venv/bin/activate
python isaacgymenvs/launch_training.py \
  --custom_experiment_name my_experiment \
  --num_envs 12288
```

## Training “sim pulita” + domain randomization leggera

Preset `clean_dr`: niente delay su obs/action, niente rumore su stato oggetto / velocità giunti, niente forze/coppie/impulsi sul cubo sollevato; in cambio DR su altezza tavolo, pose di reset e pool di cubi procedurali (4–6 cm, densità variabile). PhysX DR (`task.randomize`) resta **OFF**.

Config: `isaacgymenvs/cfg/task/SimToolRealCleanDR.yaml`

```bash
python isaacgymenvs/launch_training.py \
  --training-preset clean_dr \
  --custom_experiment_name cube_clean_dr \
  --handle-head-type cube \
  --num_envs 12288
```

Fine-tune da un checkpoint precedente (stesso `num_blocks` del run sorgente):

```bash
python isaacgymenvs/launch_training.py \
  --training-preset clean_dr \
  --custom_experiment_name cube_clean_dr_ft \
  --checkpoint train_dir/simtoolreal/.../best/model.pth \
  --num_envs 12288
```

## Fine Tune:
```bash 
python isaacgymenvs/launch_training.py \
  --custom_experiment_name fine_tuning \
  --checkpoint train_dir/simtoolreal/2026-05-06/my_experiment_2026-05-06_17-51-37/runs/00_my_experiment_2026-05-06_17-51-37/best/model.pth \
  --num_envs 12288
```

## SimToolReal — nuovi termini di reward (grasp shaping)

Attivi **solo mentre l’oggetto è sollevato** (`lifted_object`). Sommati alla reward totale in `compute_kuka_reward` e accumulati in `rewards_episode` / `episode_cumulative` per il logging.

| Chiave (accumulo episodio) | Effetto |
|----------------------------|---------|
| **`fingertip_spread_penalty`** | Penalità **≤ 0**: `-fingertipSpreadPenaltyScale × std(d₁,…,dₙ)` sulle distanze punta–oggetto. Sfavorisce pinze con poche dita vicine e altre lontane. |
| **`fingertip_multi_contact_bonus`** | Bonus **≥ 0**: `fingertipMultiContactBonusScale × min(1, N_close / K)` con `N_close` = numero di dita con distanza sotto la soglia (`fingertipMultiContactDistThresholdM`). |
| **`fingertip_thumb_bonus`** | Bonus **≥ 0** (solo sollevato): `fingertipThumbBonusScale` se il pollice è sotto la stessa soglia; indice auto (`ll_dg_1_4` su Delto) o `fingertipThumbIndex`. |

Parametri in `isaacgymenvs/cfg/task/SimToolReal.yaml` (metti `0` per disattivare lo scale corrispondente):

- `fingertipSpreadPenaltyScale` — intensità penalità spread (`0.25`)
- `fingertipMultiContactBonusScale` — intensità bonus multi-dito (`0.1`, 50× rispetto a `0.002`)
- `fingertipMultiContactDistThresholdM` — soglia in metri per contare una punta come “vicina” (`0.06`)
- `fingertipMultiContactMinFingers` — `K` per bonus pieno, limitato a `num_fingertips` (`5` su Delto)
- `fingertipThumbBonusScale` — bonus uso pollice (`0.05`; `0` disattiva)
- `fingertipThumbIndex` — indice punta pollice (`null` = auto)

### Training troubleshoot (WandB, GPU, …)
- you have to use WandB. Note there is a bug: you have to insert the API key, but you if you do `wandb login` you will get the error `wandb api key must be 40 characters long yours was 86`. To avoid the bug directly add the key to the env variables: ```export WANDB_API_KEY="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"```
- If you launch the training without `num_envs`, you will get an error like this: ```torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 320.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 23.25 MiB is free. Process 203706 has 2.73 GiB memory in use. Including non-PyTorch memory, this process has 19.69 GiB memory in use. Of the allocated memory 9.10 GiB is allocated by PyTorch, and 898.08 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)```. 
This is due to the fact that by default the envs are 24576, and they don't fit in the VRAM of the GPU I have.

- Fun fact: iter 3801: RL policies finds a bug in the simulator
- Good iteration: 48924 


Training with the UR5e:
- iter 22610: when the object falls, the arm goes anyway toward the goal
- iter 48193: when the object falls, the robot repicks it up




# Applying the code to a new robot - UR5e

## Re-run checklist (UR5e + Sharpa)

### 0) Go to repo + activate env
```bash
cd /home/simone/simtoolreal
source /home/simone/.venv/bin/activate
```

### 1) Headless env smoke test
This is a test that you can do after having changed some configurations or robot asset, in order to see if there is a failure in a matter of seconds. 
It's enough to launch it once after a modification, it's not needed to repeat it every time you want to launch a new training/evaluation.

```bash
/home/simone/.venv/bin/python - <<'PY'
from deployment.isaac.isaac_env import create_env

env = create_env(
    config_path='pretrained_policy/config.yaml',
    device='cuda',
    headless=True,
    overrides={
        'task.env.asset.robot': 'urdf/ur5e_delto_description/ur5e_left_dg5f.urdf',
        'task.env.armDofs': 6,
        'task.env.armEndEffectorLinkName': 'wrist_3_link',
        'task.env.defaultArmDofPos': [-1.5708, -1.571, 1, 0.5, 1.571, -1.571],
        'task.env.numEnvs': 1,
        'task.env.capture_video': False,
    }
)
obs = env.reset()
print('env ok', env.num_envs, env.num_acts, env.num_obs, obs['obs'].shape)
PY
```

Expected final output:
```text
env ok 1 28 137 torch.Size([1, 137])
```

### 2) Visual interactive run
```bash
/home/simone/.venv/bin/python dextoolbench/eval_interactive.py \
  --config-path pretrained_policy/config.yaml \
  --checkpoint-path pretrained_policy/model.pth \
  --port 8081
```

NOTE: change the two arguments with the model that you trained

Then open:
```text
http://localhost:8081
```

In the UI:
1. Select category/object/task
2. Click `Load Environment`
3. Click `Run Episode`

### 3) Training launch (UR5e + Sharpa)
```bash
/home/simone/.venv/bin/python isaacgymenvs/launch_training.py \
  --custom_experiment_name ur5e_sharpa_experiment \
  --num_envs 12288
```

If OOM, try:
```bash
/home/simone/.venv/bin/python isaacgymenvs/launch_training.py \
  --custom_experiment_name ur5e_sharpa_experiment \
  --num_envs 8192
```

## Training hierarchy tree (storico run)

Dettaglio reward / disturbi / tabella parametri: [`REWARD_SIMTOOLREAL.md`](REWARD_SIMTOOLREAL.md).  
Grafici WandB: `reward_step/<termine>` (un solo tag per termine; asse X = `global_step` / frame).

### Hammer / single tool

```text
00_train_single_tool_from_zero_2026-05-11_19-04-34  → spesso presa a due dita
├── 00_train_single_tool_from_chkpt1_2026-05-12_16-55-11  → shaping presa + disturbi; braccio fermo
│   └── 00_train_single_tool_from_chkpt2_2026-05-12_22-28-37  → meno penalità braccio; tardi
├── 00_train_01_st_2026-05-13_12-16-23
└── 00_train_02_st_2026-05-13_19-18-47  → kukaActionsPenaltyScale: 0
```

### Cubo

```text
00_train_2_cube_2026-05-16_11-58-42  → cubo; rischio “pinch & launch”
├── 00_train_3_cube_2026-05-18_10-45-49  → restituzione 0, max_depenetration 2 m/s
│   └── 00_train_30_cube_2026-05-21_11-57-27
│       Fine-tune da train_3 best. Shaping dita ↑ (spread 0.25, multi-contact 0.1, pollice 0.05).
│       WandB: stesso gruppo logico di train_3; confronta reward_step/fingertip_*.
│
└── 00_train_4_cube_2026-05-21_19-06-51  (sorella di train_3, non figlia)
    Da zero. Sim pulita: no delay/rumore obs (preset clean_dr), push sul cubo off;
    DR leggera su tavolo + cubi procedurali (size/massa). Vedi sezione “Training sim pulita” sopra.
```

**Eval + grafici reward locali** (es. train_3 / train_30 checkpoint):

```bash
python dextoolbench/eval_interactive.py \
  --config-path train_dir/simtoolreal/2026-05-21/train_30_cube_2026-05-21_11-57-27/runs/00_train_30_cube_2026-05-21_11-57-27/config.yaml \
  --checkpoint-path train_dir/simtoolreal/2026-05-21/train_30_cube_2026-05-21_11-57-27/runs/00_train_30_cube_2026-05-21_11-57-27/best/model.pth \
  --plot-rewards
# → eval_reward_plots/cube_training_cube_lift_delta/
```