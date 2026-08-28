# DATA INVENTORY: what exists, where, and whether it was verified

**Single source of truth for artifact locations.** Read this BEFORE assuming any
dataset, tensor, checkpoint or script exists. Update it at the END of every work
session, per Ninon 2026-08-27 after a session where four separate artifacts were
believed present and were not.

Every row carries a **verified date**. A row without one is hearsay. If you rely on a
row older than a week, re-check it and update the date rather than trusting it.

---

## 1. Cluster: /orcd/scratch/orcd/010/ninon/reaDream

| artifact | status | verified |
|---|---|---|
| `dream_dense_training_pairs_v3_256ch_fp16.pt` (1.19 GB) | PRESENT, REM block_data | 2026-08-27 |
| `dream_dense_256ch_blocks.pt` (1.25 GB) | PRESENT | 2026-08-27 |
| `dream_block_stages_v3.pt` (2.3 MB) | PRESENT, dict{night: int8 tensor}, 71 nights | 2026-08-27 |
| `pretrain_allstage_150k.pt` (30.3 GB) | PRESENT | 2026-08-27 |
| `pretrain_rem_allnights.pt` (12.8 GB) | PRESENT | 2026-08-27 |
| `pretrain_allstage/allstage_night*.pt` | PRESENT, 54 nights, bare Tensor (N,256,400) fp16 | 2026-08-27 |
| `nrem_blocks/night_*_meta.json` + `extraction_summary.json` | PRESENT, 81 nights | 2026-08-27 |
| **`nrem_blocks/per_night_pt/`** | **ABSENT.** Only meta JSONs survive. | 2026-08-27 |
| `dream_db/zhang_wamsley_2019.zip` (1049 MB) | PRESENT, md5 5854cfea4925f57d4d0a440518f4b72a | 2026-08-27 |
| `lbl/zw_labels.json` (314 KB) | PRESENT, Cortal + DReAMy + NRC-VAD on 308 reports | 2026-08-27 |
| REM_Turku | **ABSENT from cluster.** Local copy was purged. Re-download needed. | 2026-08-27 |

### Traps, all cost time today

- **`nrem_blocks/per_night_pt` is referenced by live code** (`positive_control_rem_vs_nrem.py`,
  `train_phase_decoding_v2_cv.py::_iter_nrem_nights`) but does not exist. Any Dehaene /
  H-S2 rerun needs NREM re-extraction first.
- **`allstage_night*.pt` carry NO stage labels.** Bare tensors. And
  `dream_block_stages_v3.pt` does NOT align to them: lengths are 30000/96000 (round)
  against 5119-8189 allstage rows. It is the v3 *extraction's* stages, not a full-night
  hypnogram. **There is no per-window hypnogram aligned to allstage.**
- 101-Nights has **no ground-truth hypnogram at all**. Stages were derived in-house with
  YASA (`data/preprocessing/stage_full_nights.py`). Ninon confirmed 2026-08-27.

## 2. Cluster environments

| env | torch | transformers | mne | pyriemann | use for |
|---|---|---|---|---|---|
| `scaledavde` | 2.4.0 | 5.9.0 **BROKEN pair** | 1.12.1 | 0.12 | Riemannian, NOT transformers |
| `ds4d` | 2.12.1 | 4.44.2 OK | - | - | **transformers jobs** |
| `dance_py312` | 2.12.1 | - | 1.12.1 | - | torch + mne |
| `vllmenv` | 2.11.0 | 5.12.1 OK | - | - | |

**Partition: `mit_normal_gpu` (6h cap), account `mit_general`.**
`sched_mit_psfc_gpu_r8` in `mit_cluster/sbatch/*.sbatch` is DEAD: returns
`User's group not permitted`. All three of those sbatch files fail as written.

**SSH:** gate every call behind `ssh -O check orcd-login`. Dropped 3x on 2026-08-27.
Needs interactive 2FA to re-establish. Never loop-retry (fail2ban locks Ninon out ~1h).

## 3. Authoritative code (never reimplement)

| what | path | note |
|---|---|---|
| EEGNet | `~/Downloads/EEGNet_Embedding_version.py` | 358 L, ends in LogSoftmax |
| ShallowConvNet | `~/imagination_decoding/code/crossmodal_bridge/ShallowConv_Embedding_version.py` | 127 L |
| sweep grid + `train()`/`test()` | `~/Downloads/tuning_p10_v3.py` | **CANNOT be imported**, loads data at import (line 24). Copy verbatim. |
| `set_all_seeds`, corrected loss | `~/Downloads/tuning_p12_stable.py` | **Use this NLLLoss.** p10_v3 uses CrossEntropyLoss on a LogSoftmax output = double softmax. |
| `base_model` | `~/Downloads/base_model.py` | Lightning. Converted copy at `mit_cluster/riemann/dc_ldm/models/`, `.orig` kept. **No converted copy ever existed on the cluster** despite being reported there. |
| phase decoding + NREM/REM loaders | cluster `projects/reaDream/scripts/train_phase_decoding_v2_cv.py` | `--nrem-mode` only has `last_preceding`/`all`. **Superseded**: CLAUDE.md demoted `last_preceding`, main modes should be `sws_night`/`sws_early`. |
| REM_Turku preprocessing | `data/preprocessing/prepare_remturku.py` | v `remturku-v1-sikka2019`, 133/133 OK |

**`bridge_nn.py` is NOT a model dispatcher.** It is Elias's perception->imagery bridge
(E28-NN), a different paper. Do not import it for dream work. Ninon confirmed 2026-08-27.

## 4. Experiments ALREADY RUN (do not re-run blind)

| experiment | verdict | where |
|---|---|---|
| **E-S1** (windowed phase-decode, structure vs semantics) | **UNINTERPRETABLE.** Every label below its 63-95% detectable-effect floor. Null caused by power, not absence. | `evaluation/E_S1_handoff_for_neuroscience_review.md` |
| **E-S2** (SO-spindle coupling vs speech-graph) | **CLEAN, POWERED NULL.** CIs exclude r>=0.33 and any positive assoc above ~+0.17. Marker validated. | `evaluation/E_S2_results.md` |
| **Stage 0a** REM-vs-NREM decode | PASSED, 0.795 | CLAUDE.md |
| **Stage 0b** reproduce content decode | **FAILED.** This is why E-S1 is uninterpretable. | CLAUDE.md |
| **FAA gate** (Sikka 2019 on REM_Turku) | **PARTIAL PASS.** Descriptives match to 1% (anger 40 vs 41%, interest 88 vs 88%). Effect right sign and frontal topography, rho=+0.335 p=0.19 at N=17. **Vanishes at awakening level (rho=+0.040): it is a TRAIT, not a state.** | Google Doc, `analysis/faa_gate.py` |

**Any 5-arm / Dehaene design on 101-Nights is E-S1 with a different target.** Same corpus,
same night-level granularity, same binary CV, 81 nights, one retrospective report per night.
Compute the detectable-effect floor BEFORE spending GPU.


## 4b. POWER FLOOR: 101-Nights night-level CV (computed 2026-08-27, do not re-propose)

Class balance, 81 nights, `evaluation/dreamy_enriched_labels.json`:

| label | positives / 81 | rate | majority baseline | lag-1 autocorr |
|---|---|---|---|---|
| emo_happiness | 40 | 49.4% | 0.506 | -0.125 |
| emo_apprehension | 12 | 14.8% | 0.852 | -0.078 |
| emo_anger | **4** | 4.9% | 0.951 | +0.211 |
| emo_sadness | **5** | 6.2% | 0.938 | +0.147 |
| emo_confusion | **3** | 3.7% | 0.963 | -0.039 |

**Four of five labels have 3-12 positives.** Leave-one-night-out is not underpowered
there, it is undefined: most folds contain zero positive test cases. Only `happiness`
is usable, and it is the only balanced one.

**Detectable-effect floor for happiness: 0.68 balanced accuracy at 80% power.**
(0.60 -> 46% power, 0.65 -> 76%, 0.68 -> 91%.)

Two counter-intuitive results worth keeping:
- The **circular-shift null is NARROWER than i.i.d.**, 95th pct 0.568 vs 0.593. Preserving
  autocorrelation only costs power when the autocorrelation is POSITIVE; happiness is at
  -0.125. Floor is 0.68 against either null.
- The circular-shift null has a **hard p floor of 1/81 = 0.0123** (only 80 distinct
  shifts). Under Bonferroni alpha=0.0167 exactly one p-value lies below threshold. That is
  a constraint of the null, not of the effect.

**Verdict: the Dehaene / 5-arm design cannot produce an interpretable result on
101-Nights.** 0.68 on night-level emotional content is close to the 0.795 that Stage 0a
achieves on REM-vs-NREM, one of the easiest discriminations in sleep EEG. And Stage 0b
(reproduce the known content decode through this CV chain) has ALREADY FAILED, which is
why E-S1 is uninterpretable. Running it again with a valence target reproduces a known
failure.


## 4c. INCIDENT 2026-08-27: overwritten body_action record, and what it cost

**What happened.** SLURM array `21408539` ran `train_shallowconv_emotion_cv.py` with 4
tasks (1 real + 3 permutation nulls). The script wrote to a FIXED path,
`logs/shallowconv_{target}_cv_result.json`, with no shuffle seed in the name. All four
tasks wrote to the same file and raced. The last to finish won, and it was a shuffled run,
so the surviving file contained a permutation null presented as a result. It also
destroyed the pre-existing 2026-07-01 record.

**Restored 2026-08-28.** `logs/shallowconv_body_action_cv_result.json` was rebuilt from
`logs/shallowconv_body_action_cv_16926807.log`, which survived. The restored record carries
a `_provenance` block saying so. The overwriting file is preserved as
`...cv_result.OVERWRITTEN_BY_21408539.json` rather than deleted.

**Fixed at source.** The output path now carries the shuffle seed.

### The authoritative body_action numbers

| run | date | architecture | per-fold night_acc | ensemble |
|---|---|---|---|---|
| `16920497` non-CV | 2026-07-01 | ShallowConvNetBackbone | n/a | 0.700 |
| `16926807` 5-fold CV | 2026-07-01 | ShallowConvNetBackbone | 0.6 / 0.6 / 0.5 / 0.8 / 0.7 | **0.800** (8/10) |
| `21408539_0` re-run | 2026-08-27 | same | 0.6 / 0.6 / 0.5 / 0.8 / 0.7 | **0.800** (8/10) |

The August re-run **reproduces July exactly**, so the pipeline is deterministic and 0.800
is solid. July used **ShallowConv, not EEGNet**.

Permutation nulls on body_action (`21408539_1..3`): **0.600, 0.300, 0.600**, null mean 0.50.
Real 0.800 exceeds all three, but with 3 permutations the minimum attainable p is 0.25.
8/10 nights gives a binomial p of 0.055 against chance. Encouraging, not established.

### Unresolved

A THIRD body_action JSON existed before 2026-08-27 showing per-fold `[1.0, 0.7, 0.6, 0.5, 0.6]`,
mean 0.680, `pool_prevalence` 0.503. It matches neither July log. Its source is unknown and
it is the number that was quoted in this project as "body_action 0.680". **Treat 0.800 as
the defensible figure and 0.680 as unsourced** until someone identifies that run.

## 5. The structural bind (why this keeps failing)

The corpus with sleep stages has **one label per night**; the corpus with many labels has
**only REM**. 101-Nights: W/N1/N2/N3/REM but 1 retrospective report/night, n=1 subject.
REM_Turku: 7 awakenings/subject median with per-awakening self-report, but REM only, 2 min
per file, no NREM and no wake. Zhang & Wamsley: pre-awakening segments only.

This is the "wall-breaker" line in CLAUDE.md: it needs serial-awakening within-night data.

## 6. Dataset label status

| dataset | labels | usable |
|---|---|---|
| 101-Nights | 5 HVdC binary, 81 nights, `evaluation/dreamy_enriched_labels.json` | YES, but n=1 subject |
| REM_Turku | mDES 20 items 0-4 self-rated, 121 awakenings joined, 17 subjects | YES, best power |
| Zhang & Wamsley | derived from 308 reports | **NO for categorical.** DReAMy and Cortal independently agree ~90% emotionless. anger 3.2%/2.3%, sadness 2.6%/2.3%. DReAMy "confusion" 42.5% is a length artifact (Cortal says 1.0%). VA marginal: sd 0.073 vs REM_Turku 0.175, length-confounded. |

## 7. Findings filed

- **Negative dream affect predicts recall length.** REM_Turku, n=114: negative affect
  rho=+0.226 p=0.0148, positive rho=+0.079 p=0.40, fear rho=+0.265 p=0.0051. Corroborated
  on Zhang by a different instrument (corr(valence,wordcount)=-0.194). Google Doc.
- **VAD is 2D not 3D.** Dominance collinear with valence at r=0.923 at mDES ITEM level, so
  it is the instrument, not averaging. Google Doc.

---

*Update this file at the end of every session. A stale inventory is worse than none.*
