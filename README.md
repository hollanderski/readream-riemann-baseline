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
Bonferroni over 3 labels.** The ranking is significantly INVERTED, and the mechanism is
measured: a held-out Simpson decomposition gives r = **+0.180 BETWEEN** dreamers and
r = **-0.236 WITHIN** a dreamer. The pooled direction tracks anger positively across
dreamers and negatively inside one, so a cross-subject decoder learns the between-subject
axis and applies it within the held-out dreamer, where it ranks anger backwards. This also
explains why per-subject recentring made it worse (0.334 -> 0.318): recentring removes
subject MEANS, and the between-subject axis survives mean removal.
Supported, not proven: one label of three, and the mechanism test has no null of its own.

### REM_Turku, within-subject (leave-one-awakening-out), all four labels closed
| target | observed | its null (mean, sd) | p |
|---|---|---|---|
| apprehension | 0.4477 | 0.3856, sd 0.0662, n=82 | 0.205 |
| negaff (composite) | 0.4408 | 0.4241, sd 0.0644, n=52 | 0.434 |
| anger | 0.3694 | 0.4036, sd 0.0497, n=61 | 0.758 |
| confusion | 0.3604 | 0.3917, sd 0.0610, n=62 | 0.667 |
**Report these against their own null, never against 0.5.** The within-subject null
centres at **0.3856** (tangent) and **0.4114** (DL) at this data scale, because 5-11
training awakenings per fold is not a regime where balanced accuracy centres at chance.
apprehension is ABOVE its null (p=0.205), not below chance.

### 101-Nights body_action, 1 subject
**Protocol: train on the 55-night pool with `stratified_kfold_by_night(seed=42)`, evaluate
on the 10 held-out Set B nights at NIGHT level (majority over that night's windows),
ensembling allowed for every method.** An earlier version of this table compared
ensemble-ShallowConv on Set B against single-model tangent inside the pool at window level.
That was not a comparison; it is fixed here.

| arm | val night | Set B single-model | Set B ENSEMBLE |
|---|---|---|---|
| ShallowConv | 70.6% | 64.0% | **80.0%** |
| tangent + LDA, 7 motor ch 20-40 Hz | 59.9% | 60.0% | 60.0% |

**The tangent 60.0% is not a score.** Set B is 6 negative / 4 positive nights, so the
majority-class night accuracy IS 0.600, and all five independently fitted models return
byte-identical Set B accuracy: they predict one class for every night. Tangent + LDA does
not discriminate on Set B at all.

**The ShallowConv 80.0% does NOT survive its null: p=0.1765, resolved, not floor-limited.**
Two of eighteen shuffles reached 0.8 and 0.9. The cause is the evaluation, not the model:
Set B is **10 nights**, so night accuracy moves in steps of 0.1 and the permutation
distribution spans 0.2 to 0.9 with sd 0.21. The observed sits 1.4 sd above a null centred
at 0.506. No number of extra draws fixes a granularity problem. The null was stopped at 18
of 48 once resolved, to free GPU for the protocol below.

Inside the pool at window level the tangent arm reaches 0.5723 (AUC 0.6122) against a
100-draw null centred at 0.5080, p=0.198, with per-fold values from 0.261 to 0.827. That
spread is why one subject and five night folds cannot carry a claim.

### 101-Nights body_action, repeated CV over ALL 65 nights (the protocol that replaces Set B)
Prereg addendum 4. Every night held out exactly once per repeat, mean across folds, **no
ensemble vote anywhere**. Majority baseline of the 65 nights = 0.5385.

| arm | observed | null (mean, sd, 95th) | p |
|---|---|---|---|
| tangent + LDA, 7 motor ch | 0.5754 (sd 0.0345, 10 repeats) | 0.4988, 0.0532, **0.5846** (n=100) | **0.0792** |
| ShallowConv | **0.4923** (sd 0.0126, 3 repeats) | null running | below baseline |

**ShallowConv does not reach the 0.5385 majority baseline under this protocol**, against
0.800 on the Set B protocol. Three things differ and no single-cause claim is made without
an ablation, but one of them is a validity issue rather than a power one: the project's
`train_one_fold` selects its best epoch on `val_night_acc`, so in the Set B protocol the
epoch was chosen on the fold being scored. The driver here passes the outer fold as the
test set and carves an inner validation split out of the training nights. The other two
differences are the ensemble vote and the 10-night test set, both removed here.

**The protocol works as intended: the null sd falls from 0.2106 on the 10-night Set B to
0.0532 here, 4x tighter.** The evaluation was the problem. The tangent arm is still not
significant, and note it read p=0.0816 and cleared its 95th percentile at 48 draws but
does NOT clear it at 100 (0.5754 < 0.5846) — the same drift that killed two earlier
numbers, and the reason interim p's are never quoted here.

### Positive controls on REM_Turku: is the pipeline destroying signal?
Label-free controls through the identical features, epochs and CV as every emotion arm.

| control | observed | baseline | null mean, sd, MAX | p |
|---|---|---|---|---|
| subject identity, 17-way | **0.8947** | 0.0902 | 0.0576, 0.0245, max 0.1278 (n=84) | floor |
| night 1 vs 2 | **0.8120** | 0.5338 | 0.4652, 0.0557, max 0.5714 (n=100) | floor |
| subject sex | 0.7222 | 0.6241 | 0.5006, 0.1262, max 0.8333 (n=100) | 0.089 |
| early vs late awakening | 0.5038 | 0.5188 | 0.4312, 0.0550, max 0.5940 (n=90) | 0.088 |

Two controls pass decisively: the observed lies outside the ENTIRE null range in both
cases, so their floor-limited p is a resolution limit, not uncertainty. **The emotion nulls
cannot be attributed to a pipeline that destroys signal.** Sex and circadian position are
non-significant, and early/late is non-significant rather than absent (0.5038 is above its
own null mean of 0.4312). The night control carries a caveat: leave-one-awakening-out puts
other awakenings of the same night in training, so part of 0.812 is session fingerprinting.

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
