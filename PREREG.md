# Prereg-lite: REM_Turku arm, analysis decisions declared before the numbers

Written 2026-08-28, verified at the time of writing that zero `ref17_*.json` result files
existed on the cluster. Recorded in git so the timestamp is checkable.

## Primary reference

**Braindecode `ShallowFBCSPNet` at library defaults, LOSO over all 17 subjects, no dev
split.** `--defaults --n-dev 0`, 5 seeds.

`--defaults` performs no selection, so holding out 6 subjects protected nothing while
costing 6 outer folds and 6 of 16 training subjects per fold. The originally planned
6-dev / 11-outer design is reported as a secondary variant, not as the reference.

This is declared before the all-17 numbers exist. Declared afterwards it would be
p-hacking, and it is recorded here for that reason.

## Primary estimator

**Mean over subjects of per-subject balanced accuracy** (the LOSO subject mean). The unit
of inference for a cross-subject claim is the subject, and the subject mean weights
subjects equally. Pooling held-out awakenings would weight by awakening count and let a
talkative subject dominate; it is not the primary.

Supplementary if it costs under an hour: a mixed model, balanced accuracy ~ 1 + (1|subject),
tested against the permutation distribution of its intercept. Plus the paired per-subject
test against chance.

## Null

**Permutation, WITHIN SUBJECT.** Each subject's awakening labels are permuted among
themselves so every subject keeps its own class prior. 30 permutations, p floor 1/31.

Corrected 2026-08-28 from a global across-subject shuffle, which redrew every subject's
base rate toward the grand mean. That changed the between-subject prior structure and
changed which folds came out degenerate: the across-subject nulls scored 10 or 11
subjects where the observed run scored 9. A null averaging a different subject set than
the observed is biased, not merely wide. The 11 across-subject draws already collected
are discarded for inference and kept only as the record of the defect.

## Detectable-effect floor, reported next to every number

Measured before the result, from the across-subject draws (the floor is a property of the
design, and this estimate carries their caveat):

- group-mean sd across shuffles, measured: 0.0820
- group-mean sd predicted if folds were independent: 0.0545
- ratio 1.50, mean inter-fold correlation +0.158

LOSO folds are not independent: a label shuffle perturbs the shared training set and moves
every fold together. Floor for one-sided p<0.05, null mean 0.532: **0.668 at 9 scored
subjects, 0.655 at 17**.

Every reported accuracy is printed next to its floor, with no adjectives.

## What the arm may conclude

Declared in advance so the conclusion is not chosen after seeing which way it went.

1. The verdict on the Riemannian claim comes from the **paired DL vs Riemannian
   comparison under matched folds**, not from the baseline's own p-value.
2. If neither arm clears the floor, REM_Turku is reported as a **power-limited arm** in
   one sentence in the results and one in the limitations, and the cross-subject claim
   rests on the datasets that have power.
3. The label-selectivity contrast (apprehension's CI excluding anger and confusion) is
   **not** a headline. It was noticed after seeing the numbers and was not pre-declared.
   Confusion scoring below chance is treated as a warning sign of prior shift until the
   balanced-accuracy check settles it, not as a finding.

## Methods finding, reported regardless of outcome

LOSO permutation nulls are about 1.5x wider than an independent-fold assumption predicts,
so detectable-effect floors computed under independence are optimistic. This holds whether
or not our own effect clears its floor.

---

# Addendum: learning curve, declared 2026-08-28 before any result

Verified at the time of writing that zero `lc17_*.json` files existed and the array was
still `PD`. Disclosure: the job (21470648) was submitted a few minutes before this
paragraph was written. Nothing had run, so no result informed anything below, but the
ordering is recorded rather than smoothed over.

## What it tests

On the same 9 subjects both designs scored, training on 16 subjects instead of 10 dropped
apprehension balanced accuracy by 0.071. Training on more data made it worse. The curve
varies ONLY the training-set size (k = 4, 8, 16 subjects), holding the 17 outer test folds
fixed, 3 seeds each.

## Predictions, committed in advance

| mechanism | signature |
|---|---|
| subject heterogeneity the model cannot absorb | curve FALLS as k rises |
| early stopping drifting with training-set size | curve FLAT, epochs-to-stop differs across k |
| the 11-subject number was noise flattered by its split | curve FLAT, no trend, all k at chance |

## Why it is worth running on a null result

Not to rescue apprehension; apprehension is reported as a clean null regardless of what
this shows. The curve tests the mechanism the paper's thesis rests on: that alignment is
needed because pooling heterogeneous subjects hurts. A DL curve that falls with k beside a
tangent-space curve that does not would carry that argument on a corpus where neither
method reaches significance. If both curves are flat at chance, the conclusion is that
REM_Turku carries no usable signal for either method, stated in one line.

The matching tangent-space curve is run through the same folds and the same k values, or
the comparison is not made at all.

## Reporting rule

The curve is reported whichever way it comes out, including flat. It does not become a
finding only if it falls.

---

# Withdrawn, kept in the supplement as a warning

The permutation null shuffled labels across subjects instead of within until 2026-08-28.
The measured cost of that single error, on this corpus:

| quantity | across-subject shuffle | within-subject shuffle |
|---|---|---|
| apparent fold dependence (measured/independent sd) | 1.50 | 1.15 |
| apparent inter-fold correlation | +0.158 | +0.047 |
| detectable-effect floor, p<.05 | 0.655 | 0.598 |

The global shuffle moved every subject's base rate together, which is itself a shared
perturbation across folds, so it manufactured the dependence it appeared to measure. The
"LOSO nulls are 1.5x wider than independence predicts" claim is withdrawn. The table stays
because it is a reusable warning for anyone running LOSO permutation tests.

---

# Addendum 2: alignment ladder on the tangent arm, declared 2026-08-29 before any fit

Written before the first ladder fit. The cluster was unreachable at the time of writing
(ssh control socket dead), so no ladder result could exist.

## The fact that motivates it

`tg_matched.py` built `Covariances(oas) -> TangentSpace(riemann) -> LogisticRegression`
with NO recentring: a single Fréchet mean over the pooled training covariances, applied to
the held-out subject as well. So the 0.570 / 0.390 / 0.491 reported for the "Riemannian
arm" is the UN-ALIGNED tangent space. **The paper's actual method, domain alignment across
subjects, has not been run.** Yesterday's null is a null for unaligned tangent space, which
is a weaker and different statement than "Riemannian fails on REM_Turku".

## The ladder, one ordered family, same 14 LOSO folds as the DL reference

1. tangent, global reference (already run: 0.570 / 0.390 / 0.491)
2. **tangent + per-subject recentring** (each subject's covariances whitened by its own
   Riemannian mean; the held-out subject recentred with its OWN UNLABELLED mean, which is
   legal under LOSO because it uses no test labels). **PRIMARY.**
3. + stretch (RPA re-scale)
4. + rotation (full RPA, Procrustes on training-subject class means)
5. filter-bank tangent + recentring: theta 4-8, alpha 8-13, beta 13-30 concatenated
6. late fusion: mean of DL and rung-2 probabilities

## Declared in advance

- **Primary: rung 2 on apprehension.** anger and confusion reported at the same rung,
  Holm across the three labels.
- **The paper's number is the paired rung-2 minus rung-1 difference**, the alignment gain,
  on the same subjects. It does not depend on either arm's null centre.
- Within-subject label-shuffle null per rung, 200 draws, empirical 95th percentile floor
  printed next to every number.
- Every rung is reported whichever way it comes out, including rungs that lose to rung 1.

## Standing hazard, recorded because it has bitten three times in two days

A permutation p at its floor is not evidence. Yesterday the tangent p read 0.0175 at 56
draws (0 exceedances), 0.0333 at 59, and **0.0550 at 199**. Two other numbers dissolved the
same way: apprehension 0.646 (favourable split) and "confusion below chance" (same split).
No ladder number is quotable until its 200-draw null exists.

## Prediction filed for the anger case

anger sits at 0.390 unaligned, below chance, and its per-subject class priors are the most
skewed of the three labels. If recentring is doing what it should, anger should move
toward chance or above. If anger stays at 0.39 after recentring, the prior-shift
explanation for it is wrong.
