# Riemannian dream-emotion decoding: baselines, nulls, and what they say

Handoff for Paul Barbaste. Everything here runs end to end on the MIT ORCD cluster.
Read `PREREG.md` first: it fixes every analysis choice, and it was committed before the
corresponding numbers existed. Where a number contradicts an earlier claim of ours, the
prereg is why we did not quietly switch designs.

## The one-paragraph state

Every arm we ran on REM_Turku is null, and the one performing number we own
(ShallowConv 0.800 on 101-Nights body_action) is still waiting on its permutation null.
Nothing in here is a decoding result yet. What IS solid is the harness: matched folds,
within-subject permutation nulls, and empirical floors next to every number.

## Results, with their nulls

### REM_Turku, cross-subject LOSO, 17 subjects
| arm | bal.acc | null | p |
|---|---|---|---|
| ShallowFBCSPNet (braindecode defaults) | 0.5413 | 0.5005 | 0.161 |
| EEGNet | 0.5507 | | |
| ShallowConv_Embedding (1.29M params) | 0.5347 | | |
| tangent space, global reference | 0.5701 | 0.4953 | **0.055** (199 draws) |
| tangent + per-subject recentring (prereg PRIMARY) | 0.5117 | 0.4950 | 0.398 |

Paired tangent vs DL on the same 14 subjects: **+0.0288, 11/14 positive, Wilcoxon
p=0.012** — but neither method separates from chance, so this is a method comparison,
not a decoding claim.

**anger, AUC, threshold-free: 0.3341 against a null centred at 0.5022, p=0.0100, passes
Bonferroni over 3 labels.** The ranking is significantly INVERTED. Provisional; the
mechanism test (`signtest.py`) has not completed.

### REM_Turku, within-subject (leave-one-awakening-out)
apprehension 0.4477, anger 0.3694, confusion 0.3604.
**Report these against their own null, never against 0.5.** The within-subject null
centres at **0.3856** (tangent) and **0.4114** (DL) at this data scale, because 5-11
training awakenings per fold is not a regime where balanced accuracy centres at chance.
apprehension is ABOVE its null (p=0.205), not below chance.

### 101-Nights body_action, 1 subject, 5-fold night CV
| arm | bal.acc | null | p |
|---|---|---|---|
| tangent + LDA, 7 motor ch 20-40 Hz (prereg primary) | 0.5723 | 0.5080 | 0.198 |
| tangent + LDA, PCA-32 over 256 ch | 0.4453 | | |
| ShallowConv | **0.800** | running | **PENDING** |

Per-fold spread for the tangent arm is 0.261 to 0.827. With one subject and five night
folds the variance is enormous, which is exactly why the ShallowConv 0.800 on three null
draws was never safe to quote.

## Running it

```bash
# REM_Turku cross-subject, one arm
python baseline_remturku.py --npz remturku_epochs.npz --zip REM_Turku.zip \
  --target apprehension --arch shallow_bd --defaults --n-dev 0 --seeds 0 --out out.json

# add --shuffle-labels N for a null draw (N >= 1; N=0 means NO shuffle, see below)
# add --within-subject for leave-one-awakening-out inside each dreamer

# Riemannian alignment ladder, rung 1 = global reference, rung 2 = per-subject recentring
python ladder.py apprehension 2 0 1        # observed
python ladder.py apprehension 2 1 201      # 200 null draws

# 101-Nights body_action tangent arm, same folds as the ShallowConv baseline
python ba_tangent.py --variant motor --out out.json
python test_guard.py                        # must PASS before any null run
```

## Four traps this harness exists to avoid

1. **Permute labels WITHIN subject, not across.** A global shuffle redraws each subject's
   base rate toward the grand mean. It inflated our detectable-effect floor from 0.598 to
   0.655 and manufactured a fold dependence (ratio 1.50) that does not exist (true 1.15).
2. **Seed 0 means NO shuffle.** A null loop starting at 0 puts the unshuffled observed
   inside its own null. `test_guard.py` asserts every null loop guards its lower bound;
   run it before any null.
3. **A permutation p at its floor is not evidence.** Ours read 0.0175 at 56 draws and
   0.0550 at 199. Report the draw count next to every p.
4. **One output path per seed.** A SLURM array writing a fixed filename raced and
   destroyed a result record here on 2026-08-27. Every script names outputs by seed.

## Files
`baseline_remturku.py` DL arms and within-subject mode · `ladder.py` alignment ladder ·
`ws_tangent.py` / `within_tangent.py` two independent within-subject implementations that
agree to four decimals on apprehension (0.4477) · `ba_tangent.py` 101-Nights body_action ·
`auc.py`, `aucnull.py` threshold-free metric and its null · `signtest.py` sign-inversion
mechanism test · `test_guard.py` the seed-0 regression test · `tuning_core.py` optimizer and
train loop copied verbatim from `tuning_p10_v3.py`.

## Screened corpora
`DREAM_SCREEN.md`: 20 DREAM-database deposits, 7 of which carry a LOSO claim alone
(208 subjects, 1696 awakenings, 7 to 257 channels across labs). Metadata only — nobody has
yet confirmed the EEG files load.
