# SimToolReal — termini della reward (`compute_kuka_reward`)

Somma in `isaacgymenvs/tasks/simtoolreal/env.py` (`compute_kuka_reward`).  
**Originale** = valore di riferimento **prima delle modifiche recenti** (tipico del task / commenti nel YAML / tuning nelle chat): non è necessariamente il primo commit del repository.


| Elemento (logging / variabile)  | Parametro YAML principale                                                                                   | Originale (riferimento)     | Attuale (`SimToolReal.yaml`) | Descrizione breve                                                                                                                                                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fingertip_delta_rew`           | `distanceDeltaRewScale`                                                                                     | `50.0`                      | `100.0`                      | Reward **pre-sollevamento**: progresso delle punte verso l’oggetto (delta sulla distanza minima dito–oggetto), pesato per dito.                                                                                            |
| `hand_delta_penalty`            | *(stesso scale, ma moltiplicato per 0 in codice)*                                                           | `50.0` (se attivo)          | `0` in pratica               | Penalità “mano che si allontana” — **disattivata** in `env.py` (`× distanceDeltaRewScale * 0`).                                                                                                                            |
| `lifting_rew`                   | `liftingRewScale`                                                                                           | `20.0`                      | `20.0`                       | Reward di **altezza** dell’oggetto sopra il tavolo finché non è considerato sollevato.                                                                                                                                     |
| `lift_bonus_rew`                | `liftingBonus`, `liftingBonusThreshold`                                                                     | `300.0`, `0.15` m           | `300.0`, `0.15` m            | **Bonus one-shot** quando l’oggetto supera la soglia di sollevamento.                                                                                                                                                      |
| `keypoint_rew`                  | `keypointRewScale`, `fixedSizeKeypointReward`                                                               | `200.0`, `True`             | `260.0`, `True`              | **Post-sollevamento**: avvicinamento keypoint oggetto ↔ goal (qui usa la variante fixed-size).                                                                                                                             |
| `bonus_rew`                     | `reachGoalBonus`, `successSteps`, `forceConsecutiveNearGoalSteps`                                           | `1000.0`, `10`, `False`*    | `1000.0`, `10`, `False`      | Bonus per stare **vicino al goal** (orientamento/keypoint entro tolleranza); con `forceConsecutiveNearGoalSteps=True` diventa bonus solo al successo. *Il launcher può sovrascrivere `forceConsecutiveNearGoalSteps=True`. |
| `kuka_actions_penalty`          | `kukaActionsPenaltyScale`                                                                                   | `0.03`                      | `0.0`                        | Penalità **−scale × Σ                                                                                                                                                                                                      |
| `hand_actions_penalty`          | `handActionsPenaltyScale`                                                                                   | `0.003`                     | `0.003`                      | Penalità analoga sulla **mano** (Σ                                                                                                                                                                                         |
| `arm_action_delta_penalty`      | `armActionDeltaPenaltyScale`, `actionDeltaPenaltyLiftedMultiplier`                                          | —                           | `0.0` (`real_dr`: `0.003`, `2.0×` lifted) | Penalità su `||action_t - action_(t-1)||²` del braccio dopo il delay simulato; riduce inversioni rapide del comando. |
| `hand_action_delta_penalty`     | `handActionDeltaPenaltyScale`, `actionDeltaPenaltyLiftedMultiplier`                                         | —                           | `0.0` (`real_dr`: `0.0003`, `2.0×` lifted) | Stessa penalità per le dita, mantenuta più debole per lasciare libertà agli assestamenti del grasp. |
| `object_lin_vel_penalty`        | `objectLinVelPenaltyScale`                                                                                  | variabile / spesso piccolo  | `0.0`                        | Penalità sulla **velocità lineare** dell’oggetto (stabilità). `0` = disattivata.                                                                                                                                           |
| `object_ang_vel_penalty`        | `objectAngVelPenaltyScale`                                                                                  | variabile / spesso piccolo  | `0.0`                        | Penalità sulla **velocità angolare** dell’oggetto. `0` = disattivata.                                                                                                                                                      |
| `fingertip_spread_penalty`      | `fingertipSpreadPenaltyScale`                                                                               | — (non in reward originale) | `0.25`                       | **Solo con oggetto sollevato**: `−scale × std(dist dito–oggetto)` per scoraggiare pinze “a due dita”. `0` disattiva.                                                                                                       |
| `fingertip_multi_contact_bonus` | `fingertipMultiContactBonusScale`, `fingertipMultiContactDistThresholdM`, `fingertipMultiContactMinFingers` | —                           | `0.1`, `0.06` m, `5`         | **Solo sollevato**: bonus se molte punte sono vicine all’oggetto (sotto soglia, fino a K dita).                                                                                                                            |
| `fingertip_thumb_bonus`         | `fingertipThumbBonusScale`, `fingertipThumbIndex` (+ stessa soglia distanza del multi-contact)              | —                           | `0.05`, auto (`ll_dg_1_4`)   | **Solo sollevato**: bonus se il **pollice** è sotto soglia (complementa il bonus 5 dita).                                                                                                                                 |
| `total_reward`                  | —                                                                                                           | —                           | —                            | Somma di tutti i termini sopra (reward di step in `rew_buf`).                                                                                                                                                              |


## Parametri correlati (non moltiplicatori diretti nella somma)


| Parametro                       | Originale (riferimento) | Attuale   | Ruolo                                                                                                                                     |
| ------------------------------- | ----------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `keypointScale`                 | `1.5`                   | `1.5`     | Moltiplica la **tolleranza** keypoint per `near_goal` / successo (stringe o allenta il criterio), non il valore grezzo di `keypoint_rew`. |
| `successTolerance` / curriculum | tipico `0.075` …        | vedi YAML | Curriculum sulla **tolleranza** di successo (effetto su metriche tipo `true_objective`, non su ogni termine reward sopra).                |
| `fallPenalty`                   | `0.0`                   | `0.0`     | Config presente; **reset** su caduta in `_compute_resets`, non sommato nella reward step in `compute_kuka_reward`.                        |
| `finger_rew_coeffs`             | `1` per dito (codice)   | `1`       | Pesi per dito sul `fingertip_delta_rew` (tensor di soli uno nel codice attuale).                                                          |


## File di riferimento

- Logica: `isaacgymenvs/tasks/simtoolreal/env.py` → `compute_kuka_reward`, `_distance_delta_rewards`, `_lifting_reward`, `_keypoint_reward`, `_action_penalties`, `_fingertip_grasp_shaping` (spread, multi-contact, thumb).
- Scala / default: `isaacgymenvs/cfg/task/SimToolReal.yaml` (`env:`).

---

## Storico training — gerarchia (reward, disturbi, checkpoint)

Nome run W&B / cartella sotto `train_dir/.../runs/` (prefisso `00_` aggiunto da `launch_training.py`).  
**Come aggiornare:** dopo ogni run aggiungi una riga o un ramo e, se serve, una riga nella tabella reward in alto quando cambi `SimToolReal.yaml`.

```
00_train_single_tool_from_zero_2026-05-11_19-04-34
│   Solo un tool (hammer + template singolo). Spesso presa instabile a due dita.
│
├── 00_train_single_tool_from_chkpt1_2026-05-12_16-55-11
│   │   Da checkpoint precedente: reward shaping presa stabile (spread + multi-contact),
│   │   disturbi leggeri sull’oggetto quando sollevato.
│   │   Problema osservato: il braccio smette di muoversi.
│   │
│   └── 00_train_single_tool_from_chkpt2_2026-05-12_22-28-37
│       Da chkpt1: meno penalità sul movimento braccio, più reward avvicinamento (distance delta / keypoint).
│       Problema: insufficiente e tardivo — comportamento a due dita troppo radicato, braccio “bloccato”.
│
├── 00_train_01_st_2026-05-13_12-16-23   (st = single tool)
│   Da zero: shaping presa + disturbi oggetto **e** stesso bilanciamento meno penalità braccio / più avvicinamento.
│   Problema: il braccio finisce comunque fermo.
│
└── 00_train_02_st_2026-05-13_19-18-47
    Come train_01, ma **nessuna penalità** sui giunti del braccio (`kukaActionsPenaltyScale: 0`).

00_train_2_cube_2026-05-16_11-58-42 
   Come train_02, usando un cubo
   Problema: il robot impara a pizzicare il cubo e farlo saltare anziche' ad afferrarlo. Il cubo balza lontanissimo non appena toccato

00_train_3_cube_2026-05-18_10-45-49
   Stessi parametri di train_2, ma restituzione a 0.0 e max_depenetration_velocity 1000 → 2.0 m/s
   │
   └── 00_train_30_cube_2026-05-21_11-57-27
       **Figlio di train_3** (fine-tune da `train_3_cube/.../best/model.pth`).
       Shaping presa più forte: `fingertipSpreadPenaltyScale` 0.0025 → **0.25**,
       `fingertipMultiContactBonusScale` 0.002 → **0.1**, soglia 0.042 → **0.06**, min dita 4 → **5**;
       aggiunto **`fingertipThumbBonusScale: 0.05`**. Resto allineato a train_3 (disturbi obs/action, DR cubo ±10%).

00_train_4_cube_2026-05-21_19-06-51
   training da zero con intento **sim pulita**.
   Niente disturbi su osservazioni/azioni/stato oggetto (preset `clean_dr` → `SimToolRealCleanDR.yaml`);
   niente push/impulsi sul cubo sollevato; **DR leggera**: altezza tavolo (`tableResetZRange`), pose di reset,
   pool cubi procedurali (dimensioni/massa). Shaping dita come repo attuale (0.25 / 0.1 / pollice 0.05).
   Per rilanciare: `launch_training.py --training-preset clean_dr --handle-head-type cube`.

00_train_5_cube_2026-05-22_17-27-47
   Nuova run cubo dopo fix URDF Delto: limiti corretti sui giunti del pollice
   `lj_dg_1_3` e `lj_dg_1_4` a [-90°, 0°] invece di [0°, 90°].
   --> BEST BY NOW! Ora il robot afferra per bene, ha raggiunto reward media per episodio > 10000, la reward sta crescendo bene!
   │
   └── 00_train_50_cube_from_50h_
         Riparto da 50h, con tutti gli stessi valori per controllare che il training risegua la stessa strada. Questo per controllare che tutti i valori per fare ripresa del training siano stati salvati correttamente.

00_train_06_sim2real_2026-05-27_14-20-50
   Prima run esplicitamente orientata al sim2real (`training-preset real_dr`).
   Include: tavolo abbassato con top a z≈[0, 5] cm (`tableResetZ=-0.125`, `tableResetZRange=0.025`),
   cubo procedurale 5–7 cm con COM offset fino a 5 mm, DR su massa / friction / restitution di robot-oggetto-tavolo,
   noise osservazioni/azioni con ramp lineare lunga, delay su osservazioni/azioni/stato oggetto,
   curriculum dei disturbi sull’oggetto, shaping antioscillazione sulle action delta,
   posa iniziale braccio aggiornata a `[-1.5708, -1.2, 1.8, -0.6, 1.571, -1.571]`.
   Note di debug importanti: DR dei colori del robot disattivata (`color: null`);
   DR delle `robot.dof_properties` disattivata perché causava il segmentation fault di PhysX / GPU.

00_train_07_sim2real_2026-05-28_15-39-00
   Nuovo training da zero, separato da train_06 ma con la stessa famiglia `real_dr`.
   Differenza principale: errore iniziale del braccio progressivo, con reset gaussiano attorno alla posa di default
   e std del braccio in curriculum da 0.03 a 0.10, con aumento lineare al crescere della reward media di episodio
   da 0 a 10000. Dita invariate (`resetDofPosRandomIntervalFingers: 0.08`).
   Dai grafici reward il training sembrava andare bene, ma dai video si vedeva che il braccio stava diventando
   sempre piu' lento.
   │
   └── 00_train_07_sim2real_resume_2026-06-10_11-54-51
       Resume da `.../train_07_sim2real_2026-05-28_15-39-00/.../last/model.pth` dopo il crash tardivo
       con `CUDA error: unspecified launch failure`.
       │
       └── 00_train_07_sim2real_resume_resume_2026-06-10_15-00-42
           Secondo resume della catena di train_07, lanciato dopo uno stop manuale accidentale
           della prima run `resume`.

00_train_10_real_mid_combined_2026-06-15_14-42-50
   Nuovo training da zero con preset `train_10_real_mid_combined`.
   Configurazione intermedia per sim2real: piu' robusta di train_5 / setup leggero, ma meno aggressiva del
   `real_dr` completo di train_07. Mantiene tavolo basso, cubi procedurali 5-7 cm, posa iniziale UR5e aggiornata,
   `fixedSizeKeypointReward: False`, 20 cubi template e COM offset del cubo ridotto a 3 mm.
   Delay e rumori sono moderati: `obsDelayMax: 2`, `actionDelayMax: 2`, `objectStateDelayMax: 6`,
   `objectStateXyzNoiseStd: 0.0075`, rotazione oggetto 3 gradi, rumore velocita' giunti 0.05.
   Il curriculum dei disturbi parte a reward media 7000 e arriva pieno a 15000:
   forza 4.0 -> 12.0, torque 0.30 -> 1.00, impulsi lineari/angolari 0.0 -> 0.01.
   DR fisica moderata: massa robot [0.85, 1.15], massa cubo [0.8, 1.2],
   friction cubo/tavolo [0.7, 1.6], restitution cubo/tavolo [0.0, 0.08],
   robot restitution [0.0, 0.05], senza DR sui colori e senza DR sulle `robot.dof_properties`.
   Shaping piu' morbido del full real_dr: `fingertipSpreadPenaltyScale: 0.9`,
   `fingertipMultiContactBonusScale: 0.15`, action-delta penalty arm/hand 0.0015 / 0.00015.

00_train_11_simple_2026-06-19_...
   Nuovo training da zero con preset `train_11_simple`.
   Versione semplice/controllata per riallineare il training all'eval: eredita il setup train_5-like con
   tavolo basso e cubo 5-7 cm, ma usa posa iniziale UR5e `[-1.5708, -1, 2, -1, 1.571, -1.571]`.
   La posa iniziale del cubo e' fissa e uguale all'eval:
   `objectStartPose=[0.09, -0.12, 0.03, 0, 0, 0, 1]`, con `useFixedInitObjectPose=True`.
   Randomizzazione della posa iniziale del cubo rimossa: `resetPositionNoiseX/Y/Z=0`,
   `tableResetZRange=0`, `randomizeObjectRotation=False`.
   Delay osservazioni/azioni disattivati (`useObsDelay=False`, `useActionDelay=False`) ma code minime
   a lunghezza 1 per evitare crash runtime (`obsDelayMax=1`, `actionDelayMax=1`).
   Mantiene disturbi leggeri train_5-like (`forceScale=6.0`, `torqueScale=0.5`) e senza PhysX DR pesante.

00_train_b1_simple_2026-06-...
   Nuovo training da zero con preset `train_b1_simple`.
   Variante B del setup semplice: mantiene cubo fisso e tavolo basso, ma cambia il controllo dita in
   target relativi/delta (`useRelativeHandControl=True`, `handDofSpeedScale=50`) con penalty sulle
   variazioni mano molto ridotta (`handActionDeltaPenaltyScale=0.00003`).
   Cubo iniziale fisso: `objectStartPose=[0.12, -0.12, 0.03, 0, 0, 0, 1]`; cubo 5 cm; no obs/action delay.

   └── 00_train_b2_no_omnireset_early_terminations_2026-06-...
       Run derivata da B1 per testare la stessa configurazione senza omnireset/good-reset e con le
       early termination esplicite attive (`resetWhenDropped`, `resetIfNotLiftedByDeadline`,
       `resetIfFingertipsFarAfterGrasp`).
       │
       └── 00_train_b3_no_cube_disturbances_early_termination_logging_2026-06-...
           Uguale a B2 come struttura del task, ma aggiunge debug/logging delle ragioni delle early
           termination (`reset_count/*`, `reset_current/*`, `reset/*`). Differenza pratica rilevante:
           disturbi sul cubo disattivati (`forceScale=0`, `torqueScale=0`).
           │
           └── 00_train_b4_success_tol_7p5cm_keypoint_scale_1_dropped_at_start_2026-06-...
               Derivato da B3. Rende il criterio di successo molto piu' permissivo:
               `successTolerance=0.075` m invece di `0.0075` m, con `keypointScale=1.0`.
               Inoltre abbassa la soglia del reset `dropped`: da `object_start_z + 0.01` a
               `object_start_z`, quindi il cubo deve tornare alla quota iniziale per essere contato
               come dropped.



```

### Tabella storico: **colonne = training** (+ colonna di riferimento), **righe = parametri**

Oltre ai coefficienti che entrano nella somma in `compute_kuka_reward` e al curriculum di tolleranza, sono inclusi i parametri che **influenzano il training** e che hai modificato tra un esperimento e l’altro (es. **disturbi sul tool**), più **domain randomization** (reset, scala oggetto), **definizione del tool** (`handleHeadTypes`, …), **ritardi/rumore** su osservazioni e stato oggetto, **attrito**, `**numEnvs`**, e una sintesi **RL-Games** (`train.params.config.*`, `fixed_sigma`) — in queste 5 run spesso costanti, ma da aggiornare se cambi `launch_training.py` o il train YAML. La seconda colonna (**Originale**) è fissa: aggiornala solo se cambi il baseline del repo che vuoi usare come confronto.

Valori letti dai `**config.yaml`** risolti in ciascuna cartella `train_dir/.../runs/<00_nome>/` (generati da `train.py` all’avvio).  
`—` = chiave **assente** in quel file (spesso equivale al default nel codice, es. shaping dita a 0).

Percorsi usati: `train_dir/simtoolreal/2026-05-11/train_single_tool_from_zero_.../runs/...`, `.../2026-05-12/train_single_tool_from_chkpt1...`, `.../chkpt2...`, `.../2026-05-13/train_01_st...`, `.../train_02_st...`, `.../2026-05-16/train_2_cube_.../runs/00_train_2_cube_2026-05-16_11-58-42/`, `.../2026-05-18/train_3_cube_.../runs/00_train_3_cube_2026-05-18_10-45-49/`, `.../2026-05-21/train_30_cube_2026-05-21_11-57-27/runs/00_train_30_cube_2026-05-21_11-57-27/`, `.../2026-05-21/train_4_cube_2026-05-21_19-06-51/runs/00_train_4_cube_2026-05-21_19-06-51/`, `.../2026-05-22/train_5_cube_2026-05-22_17-27-47/runs/00_train_5_cube_2026-05-22_17-27-47/`, `.../2026-05-27/train_06_sim2real_2026-05-27_14-20-50/runs/00_train_06_sim2real_2026-05-27_14-20-50/`, `.../2026-05-28/train_07_sim2real_2026-05-28_15-39-00/runs/00_train_07_sim2real_2026-05-28_15-39-00/`, `.../2026-06-15/train_10_real_mid_combined_2026-06-15_14-42-50/runs/00_train_10_real_mid_combined_2026-06-15_14-42-50/`.

### Tabella run **Sim2Real / cubo** (train_5 baseline, train_06/07 full RealDR, train_10 intermedio)

Questa tabella tiene insieme le modifiche che contano per il passaggio da train_5, che imparava bene ma aveva poca robustezza sim2real, ai preset piu' robusti. Valori letti dai `config.yaml` salvati nelle cartelle `runs/`.

| Parametro | `00_train_5_cube_…_17-27-47` | `00_train_06_sim2real_…_14-20-50` | `00_train_07_sim2real_…_15-39-00` | `00_train_10_real_mid_combined_…_14-42-50` | `00_train_b2_no_omnireset_early_terminations_2026-06-...` | `00_train_b3_no_cube_disturbances_early_termination_logging_2026-06-...` | `00_train_b4_success_tol_7p5cm_keypoint_scale_1_dropped_at_start_2026-06-...` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Parent / checkpoint | da zero | da zero | da zero, separato da train_06 | da zero | da zero, famiglia B1 | da zero / stessa famiglia B2 | derivato da B3 |
| Preset / intent | cube baseline robusto in sim | `real_dr`, full sim2real | `real_dr` + reset arm progressivo | `train_10_real_mid_combined`, sim2real intermedio | `train_b1_simple`, controllo mano relativo + early termination | come B2, senza disturbi sul cubo e con logging ragioni early termination | come B3, ma successo piu' largo e dropped meno sensibile |
| Tavolo | alto: `tableResetZ=0.38`, range 1 cm | basso: `-0.125`, range 2.5 cm | basso: `-0.125`, range 2.5 cm | basso: `-0.125`, range 2.5 cm | basso: `tableResetZ=-0.149`, range 0 | stessa B2 | stessa B3 |
| Posa iniziale UR5e | `[-1.5708, -1.571, 1.0, 0.5, 1.571, -1.571]` | `[-1.5708, -1.2, 1.8, -0.6, 1.571, -1.571]` | stessa train_06 | stessa train_06 | `[-1.5708, -1.05, 1.95, -0.9, 1.571, -1.571]` | stessa B2 | stessa B3 |
| `objectFallResetZ` | default alto / non abbassato | `-0.05` | `-0.05` | `-0.05` | `-0.05` | `-0.05` | `-0.05` |
| `dropped` reset threshold | legacy/default | `object_start_z + 0.01` | `object_start_z + 0.01` | `object_start_z + 0.01` | `object_start_z + 0.01` | `object_start_z + 0.01` | `object_start_z` |
| `fixedSizeKeypointReward` | `true` | `false` | `false` | `false` | `true` | `true` | `true` |
| `successTolerance` / `keypointScale` | `0.075 / 1.5` | `0.075 / 1.5` | `0.075 / 1.5` | `0.075 / 1.5` | `0.0075 / 1.5` | `0.0075 / 1.5` | `0.075 / 1.0` |
| Cubi procedurali | 1 cubo | 20 cubi | 20 cubi | 20 cubi | 1 cubo fisso | 1 cubo fisso | 1 cubo fisso |
| `cubeSizeRange` | assente nel config salvato | `[0.05, 0.07]` | `[0.05, 0.07]` | `[0.05, 0.07]` | `[0.05, 0.051]` | `[0.05, 0.051]` | `[0.05, 0.051]` |
| `cubeComOffsetRange` | assente | `[-0.005, 0.005]` | `[-0.005, 0.005]` | `[-0.003, 0.003]` | assente / default | assente / default | assente / default |
| Densita' cubo procedurale | singolo template | `[300, 600] kg/m^3` | `[300, 600] kg/m^3` | `[300, 600] kg/m^3` | singolo template | singolo template | singolo template |
| `objectScaleNoiseMultiplierRange` | `[0.9, 1.1]` | `[0.9, 1.1]` | `[0.9, 1.1]` | `[0.95, 1.05]` | default/fisso | default/fisso | default/fisso |
| Reset pos oggetto XYZ | `0.1 / 0.1 / 0.02` | `0.05 / 0.05 / 0.02` | `0.05 / 0.05 / 0.02` | `0.05 / 0.05 / 0.02` | `0.0 / 0.0 / 0.0`, `objectStartPose=[0.12,-0.12,0.03,...]` | stessa B2 | stessa B3 |
| Reset DOF dita / arm | `0.03 / 0.03` | `0.1 / 0.5` | `0.08 / 0.1` | `0.06 / 0.06` | `0.0 / 0.03`, vel `0.0` | stessa B2 | stessa B3 |
| Curriculum reset arm | no | no | `0.03 -> 0.1`, reward 0 -> 10000 | `0.03 -> 0.06`, reward 0 -> 7000 | no | no | no |
| Obs/action delay | `3 / 3` | `3 / 3` | `3 / 3` | `2 / 2` | off: `useObsDelay=False`, `useActionDelay=False`, max `1 / 1` | stessa B2 | stessa B3 |
| Object-state delay | 10 | 10 | 10 | 6 | off: `useObjectStateDelayNoise=False` | stessa B2 | stessa B3 |
| Object-state noise | 1 cm, 5 deg | 1 cm, 5 deg | 1 cm, 5 deg | 0.75 cm, 3 deg | off | off | off |
| Joint velocity obs noise | 0.1 | 0.1 | 0.1 | 0.05 | `0.0` | `0.0` | `0.0` |
| Obs/action generic DR | `task.randomize=false`; schedule presente ma non attivo | gaussian `[0, 0.01]`, 10G transitions | gaussian `[0, 0.01]`, 10G transitions | gaussian `[0, 0.01]`, ~5G transitions | no delay/noise DR nel task; launcher standard | stessa B2 | stessa B3 |
| `forceScale` / `torqueScale` | `6.0 / 0.5` | `20.0 / 2.0` | `20.0 / 2.0` | `12.0 / 1.0` | `6.0 / 0.5` | `0.0 / 0.0` | `0.0 / 0.0` |
| Linear/angular impulse | `0.0 / 0.0` | `0.02 / 0.02` | `0.02 / 0.02` | `0.01 / 0.01` | default / assente | default / assente | default / assente |
| Disturbance curriculum | no esplicito | reward 10000 -> 19000 | reward 10000 -> 19000 | reward 7000 -> 15000 | no esplicito | no disturbi cubo | no disturbi cubo |
| `fingertipSpreadPenaltyScale` | 0.25 | 1.25 | 1.25 | 0.9 | default/inherited | default/inherited | default/inherited |
| `fingertipMultiContactBonusScale` | 0.1 | 0.2 | 0.2 | 0.15 | default/inherited | default/inherited | default/inherited |
| Action-delta penalty arm / hand | assente o 0 nel vecchio config | `0.003 / 0.0003` | `0.003 / 0.0003` | `0.0015 / 0.00015` | arm default/0, hand `0.00003`; `useRelativeHandControl=True`, `handDofSpeedScale=50` | stessa B2 | stessa B3 |
| Robot mass DR | `task.randomize=false`; config legacy `[0.7, 1.3]` non attivo | `[0.7, 1.3]` | `[0.7, 1.3]` | `[0.85, 1.15]` | non attivo | non attivo | non attivo |
| Robot rigid-shape friction DR | non attivo | `[0.7, 1.3]` scaling | `[0.7, 1.3]` scaling | `[0.85, 1.15]` scaling | non attivo | non attivo | non attivo |
| Robot restitution DR | non attivo | `[0.0, 0.1]` additive | `[0.0, 0.1]` additive | `[0.0, 0.05]` additive | non attivo | non attivo | non attivo |
| Robot DOF-property DR | non attivo | disattivato (`null`) | disattivato (`null`) | disattivato (`null`) | non attivo | non attivo | non attivo |
| Object mass DR | non attivo | `[0.7, 1.3]` | `[0.7, 1.3]` | `[0.8, 1.2]` | non attivo | non attivo | non attivo |
| Object/table friction DR | non attivo | `[0.5, 2.0]` scaling | `[0.5, 2.0]` scaling | `[0.7, 1.6]` scaling | non attivo | non attivo | non attivo |
| Object/table restitution DR | non attivo | `[0.0, 0.1]` additive | `[0.0, 0.1]` additive | `[0.0, 0.08]` additive | non attivo | non attivo | non attivo |
| Colori asset | robot color DR nel config legacy ma non attivo | `color: true` nel config salvato | `color: null` nella versione corretta del preset | `color: null` | non attivo | non attivo | non attivo |
| Early termination | standard | standard | standard + reset arm curriculum | standard | `resetWhenDropped=True`, not lifted by step 420, fingertips >12 cm for 30 steps after step 120 | stessa B2 | stessa B3, ma `dropped` a quota iniziale invece di +1 cm |
| Early termination logging | no extra logging | no extra logging | no extra logging | no extra logging | base fractions `reset/*` | `reset_count/*`, `reset_current/*`, `reset/*` | stessa B3 |
| Good reset / omnireset | dipende dal launcher storico | dipende dal launcher storico | curriculum/good reset storico | launcher con `good_reset_boundary=0` nelle run recenti | no omnireset/good-reset: `good_reset_boundary=0`, `task.env.goodResetBoundary=0` | stessa B2 | stessa B3 |
| Note comportamento | buon grasp in sim; poca robustezza sim2real | full DR molto difficile | reward bene, video: braccio sempre piu' lento | intermedio per ridurre difficolta' mantenendo robustezza | setup semplice B: cubo fisso, mano relativa, terminate early sugli episodi morti | come B2 ma senza push/torque sul cubo e con diagnosi reset piu' leggibile | prova per sbloccare successi: tolleranza 7.5 cm, keypoint scale neutro, dropped meno anticipato |

### Tabella run **cubo** (WandB `reward_step/*`, `episode_cumulative/*`)

Confronto rapido per leggere i grafici WandB / TensorBoard. Nomi run = cartella sotto `runs/` (prefisso `00_`).

| Parametro | `00_train_2_cube_…_11-58-42` | `00_train_3_cube_…_10-45-49` | `00_train_30_cube_…_11-57-27` | `00_train_4_cube_…_19-06-51` | `00_train_5_cube_…_17-27-47` |
| --- | --- | --- | --- | --- | --- |
| Parent / checkpoint | da `train_02_st` | da `train_2_cube` | **da train_3 `best`** | **da zero** (sorella di train_3) | **da zero** |
| `fingertipSpreadPenaltyScale` | 0.0025 | 0.0025 | **0.25** | **0.25** (repo / clean_dr) | **0.25** |
| `fingertipMultiContactBonusScale` | 0.002 | 0.002 | **0.1** | **0.1** | **0.1** |
| `fingertipMultiContactDistThresholdM` | 0.042 | 0.042 | **0.06** | **0.06** | **0.06** |
| `fingertipMultiContactMinFingers` | 4 | 4 | **5** | **5** | **5** |
| `fingertipThumbBonusScale` | — | — | **0.05** | **0.05** | **0.05** |
| `lj_dg_1_3` / `lj_dg_1_4` limiti URDF | `[0°, 90°]` | `[0°, 90°]` | `[0°, 90°]` | `[0°, 90°]` | **`[-90°, 0°]`** |
| `objectRestitution` / depenetration | default | **0.0**, max depen **2.0** | come train_3 | come train_3 | come train_4 |
| Disturbi obs / action / object state | on | on | on (eredita train_3) | **off** (preset `clean_dr`) | **on** |
| DR tavolo / oggetto | `tableResetZRange` 0.01, scala ±10% | idem | idem | **tavolo ±3 cm**, pool cubi procedurali | `tableResetZRange` 0.01, scala ±10%, 1 cubo |
| Push su cubo sollevato (`forceScale`, impulsi) | 6.0 / 0 impulsi | idem | idem | **0** (clean_dr) | 6.0 / 0 impulsi |

**Nota train_4:** l’intento della run è `SimToolRealCleanDR.yaml` (vedi `README_MINE.md`). Se in un `config.yaml` salvato vedi ancora `useObsDelay: true`, il lancio non aveva `--training-preset clean_dr` — usa quel flag per allineare log e comportamento.

**Nota:** `hand_delta_penalty` è sempre moltiplicato per `0` in `env.py` (disattivato); non compare come chiave dedicata.

**Colonna «Originale (repo pre-modifica)»:** valori di riferimento del task **prima delle tue modifiche** (default da `SimToolReal.yaml` + `SimToolRealPPO.yaml` nel repo; disturbi forti `20` / `2` e impulsi `0.02` come nei commenti legacy del YAML; sei famiglie tool e `numObjectsPerType: 100` come in `launch_training.py`). Non coincide necessariamente con il paper accademico.

**Grassetto:** nelle colonne delle run, valore **diverso** da «Originale (repo pre-modifica)». `—` = chiave assente (per lo shaping dita l’originale è assenza di chiave).


| Parametro                                                               | Originale (repo pre-modifica)                                       | `00_train_single_tool_from_zero_2026-05-11_19-04-34` | `00_train_single_tool_from_chkpt1_2026-05-12_16-55-11` | `00_train_single_tool_from_chkpt2_2026-05-12_22-28-37` | `00_train_01_st_2026-05-13_12-16-23` | `00_train_02_st_2026-05-13_19-18-47` |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------ | ------------------------------------ |
| `distanceDeltaRewScale`                                                 | 50.0                                                                | 50.0                                                 | 50.0                                                   | **70.0**                                               | **100.0**                            | **100.0**                            |
| `liftingRewScale`                                                       | 20.0                                                                | 20.0                                                 | 20.0                                                   | 20.0                                                   | 20.0                                 | 20.0                                 |
| `liftingBonus`                                                          | 300.0                                                               | 300.0                                                | 300.0                                                  | 300.0                                                  | 300.0                                | 300.0                                |
| `liftingBonusThreshold`                                                 | 0.15                                                                | 0.15                                                 | 0.15                                                   | 0.15                                                   | 0.15                                 | 0.15                                 |
| `keypointRewScale`                                                      | 200.0                                                               | 200.0                                                | 200.0                                                  | 200.0                                                  | **260.0**                            | **260.0**                            |
| `reachGoalBonus`                                                        | 1000.0                                                              | 1000.0                                               | 1000.0                                                 | 1000.0                                                 | 1000.0                               | 1000.0                               |
| `successSteps`                                                          | 10                                                                  | 10                                                   | 10                                                     | 10                                                     | 10                                   | 10                                   |
| `forceConsecutiveNearGoalSteps`                                         | False                                                               | **True**                                             | **True**                                               | **True**                                               | **True**                             | **True**                             |
| `keypointScale`                                                         | 1.5                                                                 | 1.5                                                  | 1.5                                                    | 1.5                                                    | 1.5                                  | 1.5                                  |
| `fixedSizeKeypointReward`                                               | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `kukaActionsPenaltyScale`                                               | 0.03                                                                | 0.03                                                 | 0.03                                                   | **0.012**                                              | **0.001**                            | **0.0**                              |
| `handActionsPenaltyScale`                                               | 0.003                                                               | 0.003                                                | 0.003                                                  | 0.003                                                  | 0.003                                | 0.003                                |
| `armActionDeltaPenaltyScale`                                            | 0.0                                                                 | —                                                    | —                                                      | —                                                      | —                                    | 0.0                                  |
| `handActionDeltaPenaltyScale`                                           | 0.0                                                                 | —                                                    | —                                                      | —                                                      | —                                    | 0.0                                  |
| `actionDeltaPenaltyLiftedMultiplier`                                    | 1.0                                                                 | —                                                    | —                                                      | —                                                      | —                                    | 1.0                                  |
| `objectLinVelPenaltyScale`                                              | 0.0                                                                 | 0.0                                                  | 0.0                                                    | 0.0                                                    | 0.0                                  | 0.0                                  |
| `objectAngVelPenaltyScale`                                              | 0.0                                                                 | 0.0                                                  | 0.0                                                    | 0.0                                                    | 0.0                                  | 0.0                                  |
| `fingertipSpreadPenaltyScale`                                           | —                                                                   | —                                                    | **0.0025**                                             | **0.0025**                                             | **0.0025**                           | **0.25** (repo attuale)              |
| `fingertipMultiContactBonusScale`                                       | —                                                                   | —                                                    | **0.002**                                              | **0.002**                                              | **0.002**                            | **0.1** (repo attuale)               |
| `fingertipMultiContactDistThresholdM`                                   | 0.042                                                               | —                                                    | **0.042**                                              | **0.042**                                              | **0.042**                            | **0.06** (repo attuale)              |
| `fingertipMultiContactMinFingers`                                       | 4                                                                   | —                                                    | **4**                                                  | **4**                                                  | **4**                                | **5** (repo attuale)                 |
| `fallPenalty`                                                           | 0.0                                                                 | 0.0                                                  | 0.0                                                    | 0.0                                                    | 0.0                                  | 0.0                                  |
| `fallDistance`                                                          | 0.24                                                                | 0.24                                                 | 0.24                                                   | 0.24                                                   | 0.24                                 | 0.24                                 |
| `jointVelocityPenaltyScale`                                             | 0.0                                                                 | 0.0                                                  | 0.0                                                    | 0.0                                                    | 0.0                                  | 0.0                                  |
| `jointAccelerationPenaltyScale`                                         | 0.0                                                                 | 0.0                                                  | 0.0                                                    | 0.0                                                    | 0.0                                  | 0.0                                  |
| `successTolerance`                                                      | 0.075                                                               | 0.075                                                | 0.075                                                  | 0.075                                                  | 0.075                                | 0.075                                |
| `targetSuccessTolerance`                                                | 0.01                                                                | 0.01                                                 | 0.01                                                   | 0.01                                                   | 0.01                                 | 0.01                                 |
| `toleranceCurriculumIncrement`                                          | 0.9                                                                 | 0.9                                                  | 0.9                                                    | 0.9                                                    | 0.9                                  | 0.9                                  |
| `toleranceCurriculumInterval`                                           | 3000                                                                | 3000                                                 | 3000                                                   | 3000                                                   | 3000                                 | 3000                                 |
| *↓ Disturbi sul tool, domain randomization, ritardi, attrito, RL-Games* |                                                                     |                                                      |                                                        |                                                        |                                      |                                      |
| `forceScale`                                                            | 20.0                                                                | **0.0**                                              | **6.0**                                                | **6.0**                                                | **6.0**                              | **6.0**                              |
| `torqueScale`                                                           | 2.0                                                                 | **0.0**                                              | **0.5**                                                | **0.5**                                                | **0.5**                              | **0.5**                              |
| `forceProbRange`                                                        | `[0.001, 0.1]`                                                      | `[0.001, 0.1]`                                       | `[0.001, 0.1]`                                         | `[0.001, 0.1]`                                         | `[0.001, 0.1]`                       | `[0.001, 0.1]`                       |
| `torqueProbRange`                                                       | `[0.001, 0.1]`                                                      | `[0.001, 0.1]`                                       | `[0.001, 0.1]`                                         | `[0.001, 0.1]`                                         | `[0.001, 0.1]`                       | `[0.001, 0.1]`                       |
| `linVelImpulseScale`                                                    | 0.02                                                                | **0.0**                                              | **0.005**                                              | **0.005**                                              | **0.005**                            | **0.005**                            |
| `angVelImpulseScale`                                                    | 0.02                                                                | **0.0**                                              | **0.005**                                              | **0.005**                                              | **0.005**                            | **0.005**                            |
| `linVelImpulseProbRange`                                                | `[0.001, 0.1]`                                                      | `[0.001, 0.1]`                                       | `[0.001, 0.1]`                                         | `[0.001, 0.1]`                                         | `[0.001, 0.1]`                       | `[0.001, 0.1]`                       |
| `angVelImpulseProbRange`                                                | `[0.001, 0.1]`                                                      | `[0.001, 0.1]`                                       | `[0.001, 0.1]`                                         | `[0.001, 0.1]`                                         | `[0.001, 0.1]`                       | `[0.001, 0.1]`                       |
| `useSparseReward`                                                       | False                                                               | False                                                | False                                                  | False                                                  | False                                | False                                |
| `forceDecay`                                                            | 0.0                                                                 | 0.0                                                  | 0.0                                                    | 0.0                                                    | 0.0                                  | 0.0                                  |
| `forceDecayInterval`                                                    | 0.08                                                                | 0.08                                                 | 0.08                                                   | 0.08                                                   | 0.08                                 | 0.08                                 |
| `forceOnlyWhenLifted`                                                   | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `torqueDecay`                                                           | 0.0                                                                 | 0.0                                                  | 0.0                                                    | 0.0                                                    | 0.0                                  | 0.0                                  |
| `torqueDecayInterval`                                                   | 0.08                                                                | 0.08                                                 | 0.08                                                   | 0.08                                                   | 0.08                                 | 0.08                                 |
| `torqueOnlyWhenLifted`                                                  | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `linVelImpulseOnlyWhenLifted`                                           | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `angVelImpulseOnlyWhenLifted`                                           | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `handleHeadTypes`                                                       | `['hammer', 'screwdriver', 'marker', 'spatula', 'eraser', 'brush']` | `**['hammer']`**                                     | `**['hammer']`**                                       | `**['hammer']**`                                       | `**['hammer']**`                     | `**['hammer']**`                     |
| `numObjectsPerType`                                                     | 100                                                                 | **1**                                                | **1**                                                  | **1**                                                  | **1**                                | **1**                                |
| `useSingleHandleHeadTemplate`                                           | False                                                               | **True**                                             | **True**                                               | **True**                                               | **True**                             | **True**                             |
| `objectScaleNoiseMultiplierRange`                                       | `[1.0, 1.0]`                                                        | `**[0.9, 1.1]`**                                     | `**[0.9, 1.1]`**                                       | `**[0.9, 1.1]**`                                       | `**[0.9, 1.1]**`                     | `**[0.9, 1.1]**`                     |
| `resetPositionNoiseX`                                                   | 0.1                                                                 | 0.1                                                  | 0.1                                                    | 0.1                                                    | 0.1                                  | 0.1                                  |
| `resetPositionNoiseY`                                                   | 0.1                                                                 | 0.1                                                  | 0.1                                                    | 0.1                                                    | 0.1                                  | 0.1                                  |
| `resetPositionNoiseZ`                                                   | 0.02                                                                | 0.02                                                 | 0.02                                                   | 0.02                                                   | 0.02                                 | 0.02                                 |
| `resetDofPosRandomIntervalFingers`                                      | 0.1                                                                 | **0.03**                                             | **0.03**                                               | **0.03**                                               | **0.03**                             | **0.03**                             |
| `resetDofPosRandomIntervalArm`                                          | 0.5                                                                 | **0.03**                                             | **0.03**                                               | **0.03**                                               | **0.03**                             | **0.03**                             |
| `resetDofVelRandomInterval`                                             | 0.15                                                                | 0.15                                                 | 0.15                                                   | 0.15                                                   | 0.15                                 | 0.15                                 |
| `randomizeObjectRotation`                                               | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `numEnvs`                                                               | 8192                                                                | **12288**                                            | **12288**                                              | **12288**                                              | **12288**                            | **12288**                            |
| `useObsDelay`                                                           | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `obsDelayMax`                                                           | 3                                                                   | 3                                                    | 3                                                      | 3                                                      | 3                                    | 3                                    |
| `useActionDelay`                                                        | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `actionDelayMax`                                                        | 3                                                                   | 3                                                    | 3                                                      | 3                                                      | 3                                    | 3                                    |
| `useObjectStateDelayNoise`                                              | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `objectStateDelayMax`                                                   | 10                                                                  | 10                                                   | 10                                                     | 10                                                     | 10                                   | 10                                   |
| `objectStateXyzNoiseStd`                                                | 0.01                                                                | 0.01                                                 | 0.01                                                   | 0.01                                                   | 0.01                                 | 0.01                                 |
| `objectStateRotationNoiseDegrees`                                       | 5.0                                                                 | 5.0                                                  | 5.0                                                    | 5.0                                                    | 5.0                                  | 5.0                                  |
| `dofSpeedScale`                                                         | 1.5                                                                 | 1.5                                                  | 1.5                                                    | 1.5                                                    | 1.5                                  | 1.5                                  |
| `handMovingAverage`                                                     | 0.1                                                                 | 0.1                                                  | 0.1                                                    | 0.1                                                    | 0.1                                  | 0.1                                  |
| `armMovingAverage`                                                      | 0.1                                                                 | 0.1                                                  | 0.1                                                    | 0.1                                                    | 0.1                                  | 0.1                                  |
| `modifyAssetFrictions`                                                  | True                                                                | True                                                 | True                                                   | True                                                   | True                                 | True                                 |
| `robotFriction`                                                         | 0.5                                                                 | 0.5                                                  | 0.5                                                    | 0.5                                                    | 0.5                                  | 0.5                                  |
| `fingerTipFriction`                                                     | 1.5                                                                 | 1.5                                                  | 1.5                                                    | 1.5                                                    | 1.5                                  | 1.5                                  |
| `objectFriction`                                                        | 0.5                                                                 | 0.5                                                  | 0.5                                                    | 0.5                                                    | 0.5                                  | 0.5                                  |
| `tableFriction`                                                         | 0.5                                                                 | 0.5                                                  | 0.5                                                    | 0.5                                                    | 0.5                                  | 0.5                                  |
| `train.params.config.expl_type`                                         | none                                                                | **mixed_expl_learn_param**                           | **mixed_expl_learn_param**                             | **mixed_expl_learn_param**                             | **mixed_expl_learn_param**           | **mixed_expl_learn_param**           |
| `train.params.config.expl_reward_type`                                  | rnd                                                                 | **entropy**                                          | **entropy**                                            | **entropy**                                            | **entropy**                          | **entropy**                          |
| `train.params.config.expl_reward_coef_scale`                            | 1.0                                                                 | **0.005**                                            | **0.005**                                              | **0.005**                                              | **0.005**                            | **0.005**                            |
| `train.params.config.expl_coef_block_size`                              | 4096                                                                | **2048**                                             | **2048**                                               | **2048**                                               | **2048**                             | **2048**                             |
| `train.params.config.learning_rate`                                     | 0.0001                                                              | 0.0001                                               | 0.0001                                                 | 0.0001                                                 | 0.0001                               | 0.0001                               |
| `train.params.config.entropy_coef`                                      | 0.0                                                                 | 0.0                                                  | 0.0                                                    | 0.0                                                    | 0.0                                  | 0.0                                  |
| `train.params.config.horizon_length`                                    | 16                                                                  | 16                                                   | 16                                                     | 16                                                     | 16                                   | 16                                   |
| `train.params.config.minibatch_size`                                    | 32768                                                               | **98304**                                            | **98304**                                              | **98304**                                              | **98304**                            | **98304**                            |
| `train.params.network.space.continuous.fixed_sigma`                     | fixed                                                               | **coef_cond**                                        | **coef_cond**                                          | **coef_cond**                                          | **coef_cond**                        | **coef_cond**                        |


**Come aggiornare:** dopo una nuova run, aggiungi una **colonna** (nome = run sotto `runs/`) e compila le celle da `train_dir/.../runs/00_<nome>/config.yaml` → `task.env` e, per le righe `train.params.`*, dalla stessa radice del file (`train.params.config`, `train.params.network`). Ricalcola il **grassetto** confrontando ogni cella con la colonna **Originale (repo pre-modifica)**.

### Note

- Per confrontare i pesi: `train_dir/<project>/<data>/<custom_name>/runs/<00_run...>/` → `last/`, `best/`, `nn/`.
- Allineare questo albero ai **commit git** o a un export di `SimToolReal.yaml` salvato per run se vuoi riproducibilità assoluta.
