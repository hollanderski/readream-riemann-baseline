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
