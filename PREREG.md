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

---

# Addendum 3: 101-Nights body_action Riemannian arm + REM_Turku within-subject tangent
Declared 2026-08-29 before the first fit of either. Deadline moved to today by Ninon.

## Job B: 101-Nights body_action, tangent space next to ShallowConv 0.800

Same data and same folds as the existing ShallowConv result, no re-splitting:
`Dance101NightsBodyActionDataset` train+val merged as the CV pool, and
`stratified_kfold_by_night(night_pool, y_pool, n_folds=5, seed=42)` reused verbatim.
n=1 subject, so this is a WITHIN-DREAMER baseline by construction and is labelled as such.

**Dimensionality, declared because it is a real choice.** Windows are 256 ch x 3200
samples; a 256x256 covariance gives 32,896 tangent dimensions against a few hundred
windows, which is hopeless. Two variants, both fixed now:
- **PRIMARY: the 7 motor-strip channels** already defined in the project's own loader
  (`MOTOR_STRIP_CHANS`), band-passed to its own `MOTOR_BAND_HZ` = 20-40 Hz. Motor cortex,
  beta/low-gamma, for a body-action label: motivated before seeing any number.
- Secondary: 32-component spatial PCA over all 256 channels.

Pipeline: `Covariances(oas)` -> `TangentSpace(riemann)` -> `LDA(lsqr, shrinkage=auto)`.
Metric: balanced accuracy, plus AUC, per fold and pooled.

**Nulls: 100 label-shuffle draws for BOTH arms**, the tangent arm and ShallowConv, because
the existing ShallowConv 0.800 has only 3 null draws (p floor 0.25) and cannot support a
claim. Empirical 5th/95th percentile floor next to every number.

## Job A: REM_Turku within-subject, tangent + shrinkage LDA

Per subject: `Covariances(oas)` per epoch, `TangentSpace(riemann)` fit on that subject's
TRAINING epochs only, `LDA(lsqr, shrinkage=auto)`. Leave-one-AWAKENING-out inside the
subject; awakening-level prediction = mean of the held-out awakening's epoch decision
values. Report awakening-level AND epoch-level balanced accuracy, then the subject mean.
Skip subjects with fewer than 4 awakenings or a single class; state how many remain.

**Composite label, defined before running** because single mDES items are too rare per
dreamer for within-subject CV:
  NEGATIVE = max over the negative items (NA1 anger, NA7 hate, NA8 sad, NA9 scared,
             NA10 stressed)
  POSITIVE = max over the positive items (PA1, PA4, PA7, PA8, PA10)
  label = 1 if NEGATIVE > POSITIVE, 0 if POSITIVE > NEGATIVE, awakening DROPPED if equal.
Reported alongside apprehension, anger and confusion. Class counts per subject reported.

Null: within-subject label shuffle at AWAKENING level, 100 draws. Group test: Wilcoxon of
per-subject accuracies against 0.5.

## Standing rule, unchanged
No number leaves this repo while its permutation p is at its floor. It has bitten three
times: apprehension 0.646, the tangent p (0.0175 at 56 draws -> 0.055 at 199), and the
seed-0 contamination that put the unshuffled observed inside its own null.

---

# Addendum 4: repeated CV over all 65 nights, declared before the first fit

Ninon's protocol (16:48Z), written after the 10-night Set B was shown to be untestable:
the ShallowConv 0.800 gave p=0.1765 against a null centred at 0.506 with sd 0.21, because
a night-level accuracy over 10 nights moves in steps of 0.1 and its permutation
distribution spans 0.2 to 0.9. The fix is a larger evaluation, not another model.

- **Nights:** all 65 labelled nights as one pool (the former 55-night pool + the 10 Set B
  nights). The majority baseline of the 65 is reported next to every number.
- **CV:** stratified 5-fold by night, repeated with fold seeds 0..R-1. Every night is held
  out exactly once per repeat.
- **Reported:** night-level held-out accuracy per fold; mean and sd across the 5 folds
  within a repeat; mean across repeats. Balanced accuracy and window level as secondary.
- **No ensemble vote anywhere.** The vote is what made the Set B comparison unfair and it
  is removed from every arm.
- **Null:** night-level label shuffle under the identical repeated CV, with the empirical
  95th percentile of the null mean-across-folds printed next to the observed.
- Tangent + shrinkage LDA: R=10 repeats, 100 null draws, CPU. ShallowConv: R=5, GPU, only
  after the current null array drains, with its own 20-draw null under this protocol (the
  48-draw Set B null is NOT reusable, different protocol).

Every method added later, SPDNet and TSMNet included, runs through exactly this, at either
library defaults for all or the same number of sampled configs for all. Not one tuned and
one not.

---

# Addendum 5: nested sweep on REM_Turku, matched selection budget. Declared before the first fit.

fundamental_ai's ruling (2026-08-30 11:45Z), on Ninon's instruction to tune every DL
architecture with her sweep code.

## Budget, identical for every method
**12 sampled configs per method**, drawn from that method's OWN declared grid, selected on
the SAME inner splits. Not one tuned and one not: that asymmetry is the mirror image of the
ensemble-vs-single-LDA problem Ninon caught on body_action, and it is what makes the
existing SPDNet-on-Gao comparison unusable.

## Nesting, which is the point
Selection happens INSIDE each outer fold. For each of the 17 LOSO outer folds: sample the
12 configs, score each on inner splits of that fold's TRAINING subjects only, take the
winner, retrain it at full budget on all of that fold's training subjects, score once on
the held-out subject. The outer subject never touches selection. A single sweep on one dev
split followed by a frozen config, which is what our earlier arms did, is NOT nested and
its number is optimistic by an unknown amount.

## Grids, declared here
- **DL (`shallow_bd`, `eegnet`, `shallow`)**: the project's `grid_for(n_ch)` from
  `baseline_remturku.py`, 29 swept dimensions, with Ninon's constraints as already coded:
  `F1*D <= n_channels` (NOT <= 64), `depthwise_kernel_length` genuinely varying, real grid
  sampling rather than a hand-written config list.
- **tangent + LDA**: `shrinkage` in {auto, 0.0, 0.1, 0.3, 0.5}, `metric` in
  {riemann, logeuclid}, `estimator` in {oas, lwf, scm}, epoch decimation `sub` in {1, 2, 4}.
  12 sampled from that product.
- **SPDNet / TSMNet**: the repo's own hyperparameters, not invented ones, declared in a
  further addendum before that arm launches.

## Reported
- Headline: **tuned vs tuned**, nested, per target.
- One **defaults reference row per method**, clearly labelled, one run each.
- Selection metric: inner-split balanced accuracy at awakening level (REM_Turku's unit),
  not night accuracy, which is the 101-Nights unit.
- Grid as executed, with singleton dimensions listed as fixed.
- Within-subject permutation null on the winning arm; no p quoted while at its floor.

## Declared in advance
The nested number is expected to be LOWER than our earlier frozen-config numbers, because
those selected once on a dev split and reported the same split family. If it comes out
lower, that is the correction working, not a regression, and it gets reported either way.

---

# Addendum 6: nested sweep on 101-Nights body_action. Declared before the first fit.

fundamental_ai's order (2026-08-30 16:42Z), restoring Ninon's binding instruction to tune
every DL architecture we use. It was not applied to body_action: that lane was closed on
**defaults-only** DL, which conflates two different findings.

## What this run distinguishes, and why it has to exist
The lane currently reads: ShallowConv 0.4923 against a 0.5385 majority baseline, tangent
0.5754 with p=0.0792, no performing baseline. But 0.4923 is a DEFAULTS number. "This model
family cannot do the task" and "the library defaults cannot do the task" are different
claims and only the second one is currently supported. Either outcome of this run is a
result and both get reported.

## Protocol, identical to addendum 4 except for the nesting
All 65 nights, stratified 5-fold by night, R=3 repeats, night-level held-out accuracy, mean
across folds, **no ensemble vote**. Majority baseline of the 65 nights = 0.5385.

**Nested selection, per outer fold:** 12 configs sampled from `grid_for(n_channels)` with
Ninon's constraints (`F1*D <= n_channels` not 64, `depthwise_kernel_length` genuinely
varying, real sampling), each scored on 3 inner splits of that fold's TRAINING nights,
winner retrained on all of that fold's training nights, scored once on the held-out fold.
The held-out nights never enter selection.

Architectures: **ShallowConv and EEGNet**, the same two families as REM_Turku, same 12-config
budget, so the comparison across corpora is budget-matched.

## Reference rows, clearly labelled as defaults
  ShallowConv, defaults, 65-night repeated CV : 0.4923 (sd 0.0126, R=3), p=0.5714 (20 draws)
  tangent + LDA, 7 motor ch                   : 0.5754, p=0.0792 (100 draws)
  majority baseline                            : 0.5385

## Declared in advance
- The tuned number is expected to be at or below the defaults number, for the same reason
  declared in addendum 5: selection inside each fold removes the optimism that a single
  frozen config carries. If it comes out higher, that is informative about the defaults,
  not about the protocol.
- **A 20-draw night-level permutation null runs ONLY if the tuned observed beats 0.5385.**
  Spending 20 GPU-draws to null a number below its own majority baseline buys nothing, and
  the ShallowConv defaults null already showed 11 of 20 shuffles beating an observed that
  sat below baseline.

## Addendum 7 (2026-08-30T17:45Z, fundamental_ai, committed while job 21597446 is PENDING, before any fit)

SPD-network DEFAULTS row on REM_Turku: TSMNet (Kobler et al., NeurIPS 2022), the
authoritative rkobler/TSMNet repo imported unmodified (spdnets.models.TSMNet). Isolated
venv on scratch (geoopt 0.5.1, pyriemann 0.12, torch 2.12.1); no shared env touched.
Cross-subject LOSO, same 17-subject outer fold order as every other arm (ref17), same
scorability rule (folds with a single test class skipped). Defaults declared: 4 temporal
filters, 40 spatial, subspace 20, SPD domain-specific momentum-free batch norm (the
method's core), RiemannianAdam lr 1e-3, 40 epochs, batch 256, seed 0, NO early stopping,
no tuning. Test-subject batch norm adapts on unlabelled test data only (label-free, the
same legality class as Zanini recentring). Reported: epoch-level and awakening-level
(majority) balanced accuracy, per fold and mean. This is the DEFAULTS reference row; the
12-config matched-budget arm follows separately if time allows and is declared before it
runs. Expected under the campaign's evidence: near chance; a positive result must clear
a within-subject permutation null before it is called signal.

---

# Deviation log: arm cancelled mid-run on interim grounds, 2026-08-30

Recording this because addendum 5 declares 9 arms and the delivered table will have 8.

**What happened.** `21588621_1`, the shallow_bd / anger nested arm, was cancelled at 10 of
17 outer folds to free a GPU slot for the SPDNet defaults row, on Ninon's request for SPD
results immediately. The stated selection criterion was that arm's own interim inner
validation sitting at chance, i.e. "lowest information of the six".

**The objection, raised before the cancel and recorded here.** Choosing which arm to
complete on the basis of that arm's interim results makes the missing table entry not
missing at random: arms that looked promising were kept (arm _3 was named "the 60.4%
candidate"), the flat one was dropped. anger is also the target carrying the campaign's
only significant result, the inverted cross-subject AUC (0.3341, p=0.0100). A
result-independent alternative was offered at the same time (cancel by work lost: either
`nb` job stood at 2 of 17 folds against this arm's 10) and was not taken.

**Why "inner val at chance" is not a defect.** Across all 10 scored folds the 12 sampled
configs spanned 0.433-0.531 on inner validation. That is the direct measurement that
selection has almost nothing to rank on this target at this data scale, which is the same
finding the defaults-vs-swept comparison produced on 08-28 (0.6462 vs 0.6456). It is
information, not its absence.

**What was preserved.** All 10 scored folds are in
`results/PARTIAL_CANCELLED_nested_shallow_bd_anger.json`, labelled `PARTIAL_CANCELLED`,
partial mean 0.5417 (sd 0.1091, n=10 of 17). **It is not comparable to the 17-fold arms and
must never be pooled with them.** The raw log is kept at
`PARTIAL_CANCELLED_ns_21588621_1.out`.

**How the table must read.** shallow_bd / anger is reported as incomplete with n stated and
this deviation cited, not silently omitted and not filled with the partial mean.

## Amendment to addendum 7 (2026-08-30T17:55Z, fundamental_ai): arm _1 cancellation caveat

Arm 21588621_1 (shallow_bd/anger) was cancelled at 10/17 folds by fundamental_ai to free
a GPU slot for the TSMNet defaults row, on the criterion "inner val at chance = lowest
information". riemannian_geometry objected BEFORE the cancellation was executed (objection
posted 15:14Z, cancel ~15:43Z) that this criterion is RESULTS-BASED: completing arms that
look promising and dropping one that looks flat leaves a table whose missing entry is not
missing at random, on the very target (anger) carrying the campaign's only significant
result. The objection is correct and was missed because the cancel was executed without
re-reading the inbox. Remedies, binding: (1) this caveat travels with the table; (2) arm
_1 is resubmitted to full completion as soon as a slot frees, so the final 9-arm table has
NO missing entry and the cancellation is a pause, not a hole; (3) its 10 completed folds
are preserved (PARTIAL_CANCELLED_ns_21588621_1.out) and reported as partial only until
the full rerun lands; (4) any future mid-flight cancellation uses a RESULT-INDEPENDENT
criterion (work lost / declared coin flip), per riemannian's rule.

## Amendment to addendum 7 (2026-08-31T12:00Z, fundamental_ai): primary statistic pinned, null RESOLVED, extension misfire recorded

Addendum 7 said "epoch-level and awakening-level balanced accuracy" without naming the
primary. That ambiguity produced a real error: fundamental_ai read the EPOCH-level null
(0/20 draws >= 0.5834) and submitted the 100-draw extension under the standing order,
while the declared test statistic in the null job itself was AWAKENING-level (observed
0.5852), where 2/20 draws reach the observed (max 0.6106): p = 3/21 = 0.1429, RESOLVED,
not significant, no extension warranted. The extension (21643409) was cancelled within
minutes; 4 draws had started.
RULING, binding henceforth: the PRIMARY statistic for every REM_Turku arm is
AWAKENING-level balanced accuracy (the awakening is the labelled unit); epoch-level is
secondary/diagnostic. TSMNet defaults on apprehension is therefore NOT significant vs
chance (p=0.1429). The paired comparisons (all computed on epoch-level) will be re-run on
awakening-level for the paper table.

Addendum to the 12:00Z amendment (12:10Z): the causal basis for the primary-statistic
ruling, from riemannian_geometry: epochs within an awakening share its label and are
temporally correlated, so an epoch-level permutation null is narrower than the sampling
distribution of the statistic by construction; this is why the two levels disagree
(0/20 vs 2/20 on identical draws). Awakening-level was also the pre-registered unit in
addendum 5 ("inner-split balanced accuracy at awakening level, REM_Turku's unit").

## Addendum 8 (2026-09-01T13:50Z, fundamental_ai, committed BEFORE any battery fit): anger-inversion robustness battery, from external review

Pipeline: identical to the reported arm (OAS covariances, tangent space, shrinkage LDA,
LOSO over subjects, epoch level). Items and predictions, in run order with stop rules:
1. Per-subject sign plot: per held-out subject AUC with n. Stop rule: if the 2 most
   extreme subjects' removal moves p past 0.05, claim softens to subject-heterogeneous.
2. Literal demeaning: tangent features AND label demeaned within subject; LOSO linear
   readout; held-out within-subject correlation vs within-subject-shuffled null.
   Prediction (from the recentred arm's surviving inversion): r stays negative.
3. Mediator regression: within-subject anger~projection relation partialled on case
   order, time of awakening, last sleep stage, duration, artifact proportion. If the
   negative relation is absorbed by mediators, the paper reports nuisance, not anger.
4. Null extension to 1999 within-subject-shuffled draws, lower-tail AND two-sided
   reported; the lower-tail reading is acknowledged as post hoc if the ledger (item 6)
   finds no prior declaration.
5. Mixed effects with random SLOPES: label ~ proj + (proj | subject); the slope
   distribution is the within-subject claim.
6. Multiplicity ledger: exhaustive fork count on REM_Turku emotion decoding with dates;
   anger p corrected against it; prereg-timeline check for the AUC test. If the
   corrected p dies, the paper's framing becomes hypothesis-generating with the Tononi
   replication pre-registered as the confirmatory test.
