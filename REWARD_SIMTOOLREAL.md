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
               │
               └── 00_train_b5_no_lifted_grasp_shaping_precision_curriculum_2026-06-...
                   Variante B5 che eredita da `SimToolRealTrainB1Simple`: rimuove i termini di
                   reward lifted-grasp (`fingertip_thumb_bonus`, `fingertip_spread_penalty`,
                   `fingertip_multi_contact_bonus`) e riattiva il curriculum sulla precisione del
                   reaching: `successTolerance=0.075` m -> `targetSuccessTolerance=0.0075` m,
                   con `keypointScale=1.0`.
                   │
                   └── 00_train_b6_cube_physics_2026-06-26_16-11-34
                       Variante B6 preparata da B5: mantiene rimossi i reward lifted-grasp e usa
                       curriculum di precisione piu' morbido (`targetSuccessTolerance=0.075`).
                       Aggiunge una fisica del cubo meno reattiva: massa cubo fissata a 200 g
                       (`objectMassOverride=0.2`, inerzia coerente per cubo 5 cm) e
                       `sim.physx.num_velocity_iterations=2`.
                       │
                       ├── 00_train_b61_reset_dr_2026-06-...
                           Primo stadio del curriculum sim2real, fine-tuning dal best checkpoint B6.
                           Introduce piccole variazioni di reset tutte attive fin dall'inizio:
                           posizione cubo `±1.5 cm` in X/Y e `±3 mm` in Z, altezza tavolo `±5 mm`,
                           posizione iniziale dita `±0.02 rad`, braccio `±0.04 rad` e velocita'
                           iniziali `±0.03`. Mantiene invariati fisica, reward, delay e observation
                           noise di B6. La rotazione iniziale del cubo resta fissa per evitare
                           intersezioni con il tavolo alla quota di spawn bassa.
                           │
                           ├── 00_train_b62_contact_dr_2026-06-...
                           │   Secondo stadio storico del curriculum sim2real, fine-tuning dal
                           │   best checkpoint B61. Mantiene la variabilita' dei reset di B61 e
                           │   applica subito la DR fisica moderata: massa cubo `x[0.9, 1.1]`,
                           │   friction cubo `x[0.7, 1.4]`, friction tavolo `x[0.8, 1.3]`,
                           │   restitution cubo e tavolo `[0.0, 0.03]` e rumore gaussiano sulla
                           │   gravita' con sigma `0.1 m/s^2`.
                           │   │
                           │   └── 00_train_b63_command_uncertainty_2026-07-...
                           │       Terzo stadio del curriculum sim2real, fine-tuning dal best
                           │       checkpoint B62. Mantiene reset e contact DR di B62 e aggiunge
                           │       incertezza moderata sui comandi: action delay stocastico con
                           │       `actionDelayMax=2` e rumore gaussiano additivo sulle azioni con
                           │       media `0` e deviazione standard `0.003`.
                           │
                           └── 00_train_b62_linear_contact_dr_2026-07-...
                               Ramo alternativo, anch'esso fine-tuning diretto dal best checkpoint
                               B61. Introduce gli stessi range fisici di B62 con una rampa lineare
                               di `406901` simulation step, circa `5G` transizioni o `25.4k`
                               epoche con 12288 env. Observation/action noise e DR fisica del
                               robot restano disattivati.
                       │
                       └── 00_train_b61_right_2026-07-...
                           Ramo per mano destra, fine-tuning dal best checkpoint B6Right. Introduce
                           la variabilita' di reset/calibrazione con una rampa lineare fino ai range
                           di B61 e usa split Gaussian per le posizioni iniziali delle dita.
                           │
                           └── 00_train_b62_right_2026-07-...
                               Secondo stadio per mano destra, fine-tuning dal best checkpoint
                               B61Right. Mantiene la rampa/reset DR di B61Right e introduce la DR
                               di contatto/fisica di B62 con una rampa lineare di `406901` step.
                               │
                               └── 00_train_b63_right_2026-07-...
                                   Terzo stadio per mano destra, fine-tuning dal checkpoint B62Right.
                                   Mantiene reset e contact DR di B62Right al valore massimo e
                                   aggiunge linearmente la command uncertainty: action delay
                                   stocastico con `actionDelayMax=2` e rumore gaussiano additivo
                                   sulle azioni fino a media `0` e deviazione standard `0.003`.

00_train_d1_operational_space_2026-07-...
   Variante con controllo del braccio nello spazio operativo: la policy produce delta cartesiani
   dell'end effector, convertiti in target articolari tramite inverse kinematics.
   │
   └── 00_train_d2_conservative_operational_space_2026-07-...
       Riduce velocita' cartesiane e massimo delta articolare rispetto a D1, con una penalita'
       leggermente maggiore sulle azioni del braccio.
       │
       └── train_d3 (preset preparato, non ancora lanciato)
           Reference State Initialization basata su D2 e mano destra. A ogni reset, con probabilita'
           `0.5` usa il reset standard di D2; altrimenti campiona uniformemente uno degli otto grasp
           curati `grasp_candidate_5` ... `grasp_candidate_5_7`, con cubo da 5 cm gia' afferrato e
           sollevato. Gli stati RSI non ricevono nuovamente il bonus one-shot di sollevamento.
           │
           └── train_d4 (preset preparato, non ancora lanciato)
               Usa `58%` reset standard e `42%` RSI, concentrando la parte RSI su sei grasp
               campionati uniformemente: `grasp_candidate_5_3`, `grasp_candidate_6` e
               `grasp_candidate_6_1` ... `grasp_candidate_6_4`. Imposta inoltre
               `kukaActionsPenaltyScale=0.07` e `armMovingAverage=1.0`.
               │
               └── train_d5 (preset preparato, non ancora lanciato)
                   Porta il mix a `50%` reset standard / `50%` RSI e amplia il pool a dieci grasp, aggiungendo
                   `grasp_candidate_5`, `grasp_candidate_5_1`, `grasp_candidate_5_2` e
                   `grasp_candidate_5_4`. Ogni riferimento pesa circa il `5%` dei reset totali.
                   Riduce inoltre la durata massima dell'episodio da 600 a 420 step (`7 s` a 60 Hz)
                   e anticipa la deadline di mancato sollevamento a 210 step (`3.5 s`).


Motion imitation — lineage logica PPO/MIxx (tutte le run principali sotto sono da zero,
non fine-tuning, salvo indicazione esplicita):

00_motion_imitation_demo_20260727_152551_2026-07-28_16-36-33   (legacy / MI00)
│   PPO robot-only sulla dimostrazione `demo_20260727_152551_335339.npz`, RSI uniforme
│   e scheduler adattivo. Dopo circa 1.2G frame la reward collassa; il learning rate
│   era cresciuto fino a oltre 25 volte il valore iniziale.
│
└── MI01 — PPO con learning rate fisso `5e-5`
    │   Diversi tentativi di avvio a 12k/6k env il 2026-07-29 hanno evidenziato limiti
    │   di creazione PhysX del task. Il setup stabile usa 4800 env.
    │
    └── 00_debug_ppo_fixed_lr_4800_2026-07-29_17-13-08
        │   Prima run PPO fixed-LR completa/stabile a 4800 env. Il tracking migliora,
        │   ma la reward successivamente tende a diminuire e la policy non segue ancora
        │   la dimostrazione con sufficiente precisione.
        │
        └── 00_mi02_ppo_linear_triangular_rsi_no_action_penalties_2026-07-30_10-35-02
            │   Da zero. RSI triangolare con maggiore probabilita' per le fasi iniziali;
            │   rimosse tutte le action/action-delta penalties. LR lineare `5e-5 -> 1e-6`
            │   nei primi 6000 epoch e poi mantenuto al floor, senza arresto automatico
            │   (`max_epochs=-1`, stop manuale con Ctrl+C).
            │   Risultato: prima policy PPO giudicata buona, ottenuta in circa 2 ore.
            │
            └── 00_mi03_ppo_filtered_velocity_tracking_2026-07-30_16-11-35
                │   Da zero. MI02 + tracking di velocita' lineare palm, velocita' angolare
                │   palm e velocita' dei 20 joint mano. Velocita' derivate nel loader dalla
                │   demo originale, con filtro triangolare centrato da `0.15 s`; observation
                │   invariata a 101D. Reward `0.8 pose + 0.2 velocity`.
                │   Risultato: target filtrati ragionevoli, ma velocity reward istantaneo
                │   ancora molto rumoroso.
                │
                ├── 00_mi04_ppo_matched_window_velocity_2026-07-31_10-27-33
                │   Tentativo a 12.288 env: segfault durante la creazione degli actor PhysX,
                │   prima del caricamento demo e dei buffer MI04; nessun checkpoint prodotto.
                │
                └── 00_mi04_ppo_matched_window_velocity_2026-07-31_10-44-47
                    │   Da zero, 4800 env. Le velocita' simulate e reference sono calcolate
                    │   con lo stesso intervallo di 5 step (~83 ms a 60 Hz). Warm-up di 5
                    │   step dopo reset: peso velocity `0 -> 0.2`, peso pose `1 -> 0.8`.
                    │   Reintrodotte solo le delta penalties: arm `0.001`, hand `0.0001`.
                    │   Risultato osservato: tracking nettamente migliore di MI03; resta un
                    │   leggero shaking. A epoch 3500, periodic eval completa la clip con
                    │   reward medio/step ~0.892, errore posizione medio ~2.8 cm,
                    │   orientamento ~0.070 rad e hand L2 ~0.62 rad.
                    │
                    ├── 00_mi05_ppo_stronger_action_delta_2026-07-31_14-08-06
                    │   │   Da zero, 4800 env, avviato. Identico a MI04 tranne delta-action
                    │   │   penalties moltiplicate per 3: arm `0.003`, hand `0.0003`.
                    │   │   Obiettivo: ridurre lo shaking residuo senza penalizzare i comandi
                    │   │   sostenuti; le action-magnitude penalties restano a zero.
                    │   │
                    │   └── MI06 (preset preparato, non ancora lanciato)
                    │       Da zero. MI05 + target pose istantanea dalla demonstration:
                    │       posizione palm (3D), orientamento palm quaternion (4D) e pose
                    │       normalizzate dei 20 joint delle dita. Observation 101D -> 128D.
                    │       Mantiene velocity reward matched-window, warm-up, RSI triangolare,
                    │       delta penalties e profilo PPO di MI05; target velocity non in input.
                    │       │
                    │       └── MI07 (preset preparato, non ancora lanciato)
                    │           Da zero. Mantiene la target pose completa di MI06, rimuove la
                    │           phase dall'osservazione e aggiunge target palm linear/angular
                    │           velocity (3D + 3D) e target finger velocity (20D). Observation
                    │           128D -> 153D. Obiettivo: evitare l'anticipo temporale osservato
                    │           in MI06, fornendo direzione/velocita' istantanee senza un clock.
                    │
                    └── MI08-PositiveGaussianRegularization (preset preparato)
                        Ramo diretto da MI04. Sostituisce le penalty quadratiche negative
                        con reward gaussiane positive e limitate. Tutti i sei termini
                        attivi hanno massimo `0.05`; le sigma sono `500 / 5000` per
                        action arm/hand, `500 / 5000` per action-delta arm/hand,
                        `50` per arm-joint velocity e `50000` per arm-joint acceleration.
                        I rapporti `SCALE/SIGMA` riproducono localmente tutti i
                        coefficienti SAPG04. Solo hand joint acceleration resta
                        disattivato. Il launcher supporta `--disable-video` per TARS/CASE.
                        │
                        ├── MI09-DeltaGaussian2x (preset 600M preparato)
                        │   Mantiene tutte le SCALE di MI08 e dimezza solo le SIGMA dei
                        │   delta action arm/hand: `500 -> 250`, `5000 -> 2500`.
                        │   Raddoppia quindi la regolarizzazione locale sui cambi di comando.
                        │
                        ├── MI10-TargetInputSmoothGaussian (preset 600M preparato)
                        │   Controparte PPO controllata di SAPG08: eredita direttamente
                        │   il suo task, quindi reward, observation 104D, RSI triangolare,
                        │   smoothing e dinamica dell'environment coincidono. Cambia
                        │   soltanto il profilo di ottimizzazione, da SAPG a PPO.
                        │   Richiede training da zero.
                        │
                        └── MI11-CombinedGaussian2x (preset 600M preparato)
                            Resta il precedente ramo fattoriale da MI09: delta action e
                            dinamica misurata del braccio hanno coefficiente locale 2x,
                            senza il nuovo target input o smoothing introdotto da MI10.


Motion imitation — lineage logica SAPG:

00_motion_imitation_sapg_2026-07-29_12-14-47   (SAPG01-Base)
│   Da zero, 4800 env, SAPG con 6 blocchi da 800 env. RSI uniforme, observation
│   101D e reward pose con scale position/orientation/hand `100 / 2 / 0.5`.
│
└── 00_motion_imitation_sapg_precision_2026-07-30_12-14-50   (SAPG02-Precision)
    │   Fine-tuning dal checkpoint `last/model.pth` di SAPG01. Mantiene RSI uniforme
    │   e observation 101D, ma aumenta le scale position/orientation/hand a
    │   `400 / 15 / 2`. Robot reference verde escluso dal training e usato solo
    │   nell'eval periodico isolato.
    │
    └── 00_sapg03_triangular_target_input_2026-07-31_15-42-06   (SAPG03-Triangular-TargetInput)
        Successore logico/configurativo di SAPG02, ma avviato **da zero** senza
        checkpoint. SAPG a 6 blocchi, RSI triangolare con mode 0 e scale reward
        `400 / 15 / 2`. Aggiunge la target palm position (`reference_palm_pos`, 3D)
        all'input della policy: observation 101D -> 104D. Velocity tracking
        disabilitato per isolare triangular RSI e target conditioning.
        │
        └── 00_sapg04_joint_regularized_2026-08-02_18-03-08
            │   Da zero. Mantiene observation, reward pose e RSI di SAPG03. Riduce
            │   di 10x i costi sui comandi EE/mano e sui loro delta; aggiunge costi
            │   sulle velocita' reali dei 6 giunti UR e sulle accelerazioni reali
            │   dei giunti arm/hand. A epoch 19500: eval completa, reward medio
            │   ~0.885, errore posizione medio ~1.13 cm, nessuna soglia violata.
            │   Tracking soddisfacente, ma restano vibrazioni e penalita' troppo deboli.
            │
            ├── SAPG05-StrongRegularization
            │   │   Da zero. Aumenta rispetto a SAPG04 i costi action/rate EE a
            │   │   `5e-4 / 5e-4`, hand a `1e-4 / 5e-4` e arm joint
            │   │   velocity/acceleration a `2e-3 / 2e-6`; disabilita hand joint
            │   │   acceleration (`0`). Il tracking iniziale e' buono, ma nella eval
            │   │   da phase zero la hand pose degrada stabilmente dopo circa lo step
            │   │   650. Policy giudicata meno gradevole di SAPG04.
            │   │
            │   └── SAPG-OBJ01-KeypointTracking (nuova famiglia)
            │       Derivato logicamente da SAPG05 e avviato **da zero** per il
            │       cambio observation 104D -> 138D. Attiva il cuboide fisico registrato
            │       5x5x15 cm e aggiunge il tracking denso di quattro keypoint
            │       corrispondenti object/reference. Reward object:
            │       `0.25 * exp(-400 * max_i(||k_obj_i-k_ref_i||)^2)`.
            │       La policy osserva object rot/vel, keypoint rispetto alla palm e
            │       delta dei keypoint rispetto alla posa object della demonstration.
            │       Come in `demo_viewer.py`, il modello usa dimensioni locali
            │       `[0.15, 0.05, 0.05]` (asse lungo X), senza offset locale sul
            │       quaternion; il piano del tavolo e' a `z=-0.03 m`. Risultato:
            │       il robot non ha imparato a sollevare l'oggetto; il termine object
            │       massimo `0.25` era troppo debole rispetto al robot reward `1.0`.
            │       La soglia early termination hand L2 della famiglia object-aware
            │       e' rilassata da `2.0` a `2.2 rad`.
            │       │
            │       └── SAPG-OBJ02-PregraspObjectPriority (preset preparato)
            │           Stessa observation/rete 138D. Porta il massimo object reward
            │           `0.25 -> 2.0` (8x), abbassa l'early termination object-position
            │           da `15 cm` a `6 cm` dopo 10 step di grace period e usa RSI misto:
            │           `50%` dei reset esattamente al pre-grasp grounded a phase `0.6`
            │           (~`11.08 s` nella demo), `50%` dalla precedente
            │           distribuzione triangolare. Puo' partire da zero o da un
            │           checkpoint OBJ01 compatibile.
            │           │
            │           ├── SAPG-OBJ03-Object66Imitation33 (900M, da zero)
            │           ├── SAPG-OBJ04-Object50Imitation50 (900M, da zero)
            │           └── SAPG-OBJ05-Object33Imitation66 (900M, da zero)
            │               Sweep controllato dei rapporti object/imitation `2:1`,
            │               `1:1`, `1:2`; in tutti i casi i massimi primari sommano
            │               a `1.0`. Aggiunge shaping comune: lifting lineare e
            │               saturato a `0.10` dopo 10 cm; fingertip-delta SimToolReal
            │               con scala `20` e cap `0.10` per step. Mantiene RSI
            │               pre-grasp 50% ed early termination object a 6 cm. Ripristina
            │               inoltre le kernel imitation piu' larghe di SAPG01:
            │               position/orientation/hand `100 / 2 / 0.5`, evitando i
            │               valori precision `400 / 15 / 2` ereditati da SAPG02.
            │
            ├── SAPG06-RegularizationCurriculum (preset preparato)
                Fine-tuning obbligatorio dal checkpoint SAPG04, con LR fisso `1e-5`.
                Mantiene esattamente le scale SAPG04 per 8000 control step (500 epoch),
                poi usa una rampa smoothstep di 96000 step (6000 epoch) e continua
                indefinitamente ai valori finali. Aumenta solo arm/hand action delta a
                `2e-4 / 5e-5`, arm joint velocity a `1.25e-3` e arm joint acceleration
                a `1.5e-6`; action magnitude invariata e hand joint acceleration a zero.
                Obiettivo: ridurre le vibrazioni preservando il comportamento SAPG04.
            │
            └── SAPG07-IntermediatePrecision (preset preparato)
                Training da zero con observation, RSI, SAPG e regolarizzazioni di
                SAPG04. Sostituisce soltanto le kernel pose precision `400 / 15 / 2`
                con i valori geometricamente intermedi `200 / 5.4772255751 / 1.0`.
                Obiettivo: mantenere piu' precisione di SAPG01 senza il bacino stretto
                e le correzioni brusche osservate in SAPG04. Nessun limite automatico.
                │
                ├── SAPG08-PositiveGaussianRegularization (preset preparato)
                    Mantiene position/hand tracking, observation, RSI e profilo SAPG
                    di SAPG07; allarga orientation a `scale=2`. Usa sei bonus
                    `0.05 * exp(-||x||^2 / sigma)` con sigma `50 / 166.667 / 25 /
                    50 / 25 / 16666.667` per action arm/hand, delta arm/hand e arm
                    qd/qdd. I coefficienti locali sono `1e-3 / 3e-4 / 2e-3 /
                    1e-3 / 2e-3 / 3e-6`; hand qdd e' disattivato. La calibrazione
                    sulla eval SAPG07 porta la perdita di bonus associata alle
                    oscillazioni osservate da ~`1.7%` a ~`4.8%`. Aggiunge smoothing
                    leggero dei target controller arm/hand (`0.8 / 0.8`). Bonus
                    massimo `0.30`, reward teorica massima `1.30`. Nessun limite
                    automatico.
                    │
                    ├── SAPG10-FingertipTracking (preset preparato)
                        Deriva da SAPG08 e mantiene observation 104D, RSI,
                        smoothing, regolarizzazioni gaussiane e profilo SAPG.
                        Il loader applica una sola volta la FK alla traiettoria
                        articolare originale e ricava i cinque punti fingertip nel
                        frame locale del palmo virtuale, senza modificare il file
                        demonstration. La pose reward usa pesi `0.35 / 0.25 /
                        0.25 / 0.15` per EE position, EE orientation, finger joints
                        e fingertip; l'ultimo termine e' `exp(-500 * RMS_tip^2)`.
                        Serve a premiare direttamente la geometria utile al grasp.
                    │
                    ├── SAPG11-Phase055To085 (preset preparato)
                        Confronto controllato che eredita integralmente SAPG08 e
                        cambia soltanto i limiti temporali. La fase resta globale:
                        gli episodi partono da `0.55`, terminano a `0.85` e durano
                        quindi il 30% della demonstration. La RSI triangolare viene
                        rimappata su `[0.55, 0.85]`, con mode a `0.55`; reward,
                        observation 104D, smoothing, regolarizzazioni e SAPG non
                        cambiano.
                    │   │
                    │   ├── SAPG12-Phase055To085-UniformRSI (preset preparato)
                    │       Deriva integralmente da SAPG11 e modifica soltanto la
                    │       distribuzione RSI: le fasi iniziali sono uniformi su
                    │       `[0.55, 0.85]` anziche' triangolari con mode a `0.55`.
                    │       Segmento, reward, observation, smoothing,
                    │       regolarizzazioni e profilo SAPG restano identici.
                    │       │
                    │       └── SAPG13-NoRSI-JointSmoothness (preset preparato)
                    │           Mantiene il segmento globale `[0.55, 0.85]` ma
                    │           disabilita completamente RSI, quindi ogni episodio
                    │           parte esattamente da `0.55`. La reward contiene solo
                    │           EE position/orientation e hand joint pose con pesi
                    │           `0.4 / 0.3 / 0.3`, piu' due bonus gaussiani bounded
                    │           da `0.05` per la smoothness delle accelerazioni fisiche
                    │           di arm e hand (`sigma=16666.67 / 5000000`). Tutti i
                    │           termini action, action-delta e arm velocity sono
                    │           disattivati. Reward massima teorica `1.10`.
                │
                └── SAPG09-BroadPoseReward (preset preparato)
                    Esperimento causale da zero che modifica soltanto le kernel pose
                    di SAPG07 da `200 / 5.4772255751 / 1` a `100 / 2 / 0.5`.
                    Observation 104D, RSI triangolare, penalty quadratiche SAPG04,
                    controller senza smoothing aggiuntivo e profilo SAPG restano
                    identici a SAPG07. Velocity tracking disattivato e nessun limite
                    automatico. Serve a isolare la rigidita' della reward come causa
                    delle vibrazioni.



```

### Tabella lineage motion imitation MI00–MI11

Questa famiglia usa `SimToolRealMotionImitation.compute_imitation_reward`, separata dalla
reward object-manipulation descritta nella tabella iniziale del documento.

| ID | RSI | LR PPO | Reward velocity | Action penalty arm/hand | Delta penalty arm/hand | Stato / risultato |
| --- | --- | --- | --- | --- | --- | --- |
| `MI00` | uniforme | adattivo | no | `0.001 / 0.0001` | `0.001 / 0.0001` | Collasso dopo ~1.2G frame, LR >25x iniziale |
| `MI01` | uniforme | fisso `5e-5` | no | `0.001 / 0.0001` | `0.001 / 0.0001` | Run stabile a 4800 env, tracking ancora insufficiente |
| `MI02` | triangolare, mode 0 | lineare `5e-5 -> 1e-6`, poi floor | no | `0 / 0` | `0 / 0` | Buon PPO in circa 2 ore |
| `MI03` | come MI02 | come MI02 | demo derivative filtrate `0.15 s`; peso `0.2` | `0 / 0` | `0 / 0` | Velocity reward troppo rumoroso |
| `MI04` | come MI02 | come MI02 | matched window 5 step + warm-up 5 step; peso `0.2` | `0 / 0` | `0.001 / 0.0001` | Tracking migliore; leggero shaking residuo |
| `MI05` | come MI02 | come MI02 | come MI04 | `0 / 0` | `0.003 / 0.0003` | Avviato; verifica riduzione shaking in corso |
| `MI06` | come MI02 | come MI02 | come MI04; target palm pose + finger pose in input (128D) | `0 / 0` | `0.003 / 0.0003` | Preset preparato; training da zero richiesto |
| `MI07` | come MI02 | come MI02 | come MI04; target pose + target velocities, senza phase (153D) | `0 / 0` | `0.003 / 0.0003` | Preset preparato; training da zero richiesto |
| `MI08` | come MI04 | come MI04 | come MI04 | Gaussiani action arm/hand: `S=.05`, `sigma=500/5000` | Gaussiani delta arm/hand: `S=.05`, `sigma=500/5000`; anche arm qd/qdd gaussiani | Coefficienti locali SAPG04; hand qdd disattivato; video disattivabile |
| `MI09` | come MI08 | come MI08 | come MI08 | come MI08 | Delta gaussiani arm/hand: `sigma=250/2500`; arm qd/qdd come MI08 | Isola delta-action 2x; run da zero, limite 600M |
| `MI10` | triangolare come SAPG08 | PPO lineare `5e-5 -> 1e-6` | disattivata; `reference_palm_pos` nell'observation (104D) | Gaussiani arm/hand: `sigma=50/166.67` | Delta `sigma=25/50`; arm qd/qdd `sigma=25/16666.67`; smoothing arm/hand `0.8` | Task identico a SAPG08; cambia soltanto SAPG → PPO; run da zero, limite 600M |
| `MI11` | come MI08 | come MI08 | come MI08 | come MI08 | Delta `sigma=250/2500`; arm qd/qdd `sigma=25/25000` | Precedente ramo fattoriale delta+dynamics, senza target input; run da zero, limite 600M |

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















# Proposed Curriculum
| Training	| Add only	| Suggested values |
| --- | --- | --- |
| B61	| Reset/calibration variation |	useFixedInitObjectPose: false, XYZ noise [0.015, 0.015, 0.003], table range 0.005, finger reset 0.02, arm reset 0.04 |
| B61Right	| Right-hand reset/calibration variation, linear ramp |	from B6 values to the B61 reset ranges in 406901 control steps; split Gaussian for finger joint positions |
| B62	| Contact physics, immediate	| object mass scaling [0.9,1.1], object friction [0.7,1.4], table friction [0.8,1.3], restitution [0,0.03], gravity noise up to 0.1 |
| B62Linear	| Contact physics, linear ramp	| reach the same B62 ranges in 406901 simulation steps |
| B62Right	| Right-hand contact physics, linear ramp	| fine-tune from B61Right; reach the same B62 contact/physics ranges in 406901 simulation steps |
| B63	| Command uncertainty	| useActionDelay: true, actionDelayMax: 2, action Gaussian noise std around 0.003|
| B63Right	| Right-hand command uncertainty, linear ramp | fine-tune from B62Right; keep B62Right DR at max and linearly add the B63 action delay/noise |
| B64	| Perception uncertainty	| object delay max 4, XYZ noise 0.003 m, rotation noise 1.5°, velocity noise 0.02, observation delay max  2 |
| B65	| Stronger combined DR	| object delay max 6, XYZ 0.005 m, rotation 3°, action delay max 3, force 8–10, torque 0.7–1.0, small impulses |
| B66	| Precision curriculum	| successTolerance: 0.075, targetSuccessTolerance: 0.04; introduce no other new difficulty |
| B67	| Final consolidation	| Slightly wider versions of all previous ranges, but still based on measured real-system uncertainty |
