# Multiplicity ledger: the anger inversion under honest correction (2026-09-01)

## Timeline facts, from artefact mtimes and prereg commit history

1. The anger AUC null draws began 08-29 ~08:07 (cluster mtime, aucnull_r1_anger_*):
   the analysis and its result predate every prereg text that mentions AUC.
2. The only pre-battery "AUC" declaration (addendum 3, commit 3143ca8, 08-29 15:03)
   sits in the 101-Nights body_action section (motor-strip arm) and does not cover the
   REM_Turku cross-subject anger test.
3. The word "lower" (lower-tail) appears in no prereg text before addendum 8
   (09-01, the battery itself). The lower-tail reading of the anger AUC is POST HOC.

## Fork count (observed cross-subject and within-subject emotion tests on REM_Turku)

From the results directory: ref17 x3 labels; arch17 x2 archs (apprehension); tangent
logistic x1; learning curve x3; ladder r1/r2 balanced accuracy x6; ladder r1/r2 AUC x6;
nested x5 arms; TSMNet x3 (epoch) +3 (awakening); within-subject x7; negaff composite x1.
Conservative count: ~38 observed test statistics, before counting the two bias artefacts
that were caught and corrected (each a fork in the garden even when fixed honestly).

## The math

- FINAL null, 2000 draws (array 21740825, closed 2026-09-01 ~16:26Z, 50/50 tasks):
  observed mean AUC **0.3353**, null mean 0.5002 (sd 0.0703, 95% [0.3609, 0.6388]),
  **p_lower = 0.0095 (18 exceedances), two-sided p = 0.0190**, floor 0.0005.
  18 exceedances means this is a measured p, not a floor-limited bound.
- The 10x larger null did NOT change the verdict. At 199 draws the lower tail read
  0.010; at 2000 it reads 0.0095. The conclusion below was never resting on the
  draw count, and now that is on the record rather than assumed.
- Bonferroni over the 3 labels alone: 0.0190 x 3 = **0.057**. Above 0.05 already.
- Corrected against the honest campaign-wide count (~38): dead by an order of magnitude.
- Caveat carried everywhere this number appears: the observed statistic is a **mean
  over 16 scorable subjects of 17** (one subject single-class, skipped by the LOSO
  scorability rule for the observed AND every null draw alike, measured over 400 draws,
  so it biases nothing); awakenings per subject range 2-12, so the per-subject AUCs the
  mean is taken over are not equally precise.
- Battery item 5 (mixed-model slopes) returned **fixed_proj_p = NaN** despite reporting
  converged. That item supports nothing until the NaN is explained; it is not counted
  as descriptive support below.

## Verdict, binding for the workshop paper

The anger inversion is NOT a confirmatory finding. It is reported as
HYPOTHESIS-GENERATING: an observed, mechanistically coherent pattern (between-subject
r = +0.18, within-subject r = -0.24, surviving per-subject recentring) whose
confirmatory test is PRE-REGISTERED on the DREAM/Tononi Serial Awakenings deposit
(39 subjects): exact statistic declared in advance: two-sided, awakening-level,
within-subject-shuffled null at >= 1999 draws, single label (anger), single arm
(tangent + shrinkage LDA, global reference), no alternatives. The robustness battery
(addendum 8) is reported as descriptive support, not as inference.

Every null in the campaign was reported, and this ledger exists because we counted our
own forks before a reviewer had to.

## Reporting frame, final (2026-09-01T16:50Z, per external review round 3; prereg f98299d declared the item-5b terms before its result)

- HEADLINE STATISTIC: the sign test. 14 of 16 held-out subjects below chance, exact
  binomial two-sided p = 0.0042. Distribution-free, independent of the permutation
  null's construction.
- EFFECT SIZE: mean AUC 0.3353, bootstrap-over-subjects 95% CI [0.2415, 0.4348]
  (excludes 0.5); permutation p = 0.0095 +/- 0.002 (SE at 2000 draws): a measurement.
  The 199-vs-2000 agreement is expected, not evidential, and is not cited as support.
- DEPENDENCE, stated: 14/16, AUC 0.335, and demeaned r = -0.278 are three views of one
  quantity computed on one set of held-out predictions; they are never presented as
  independent converging evidence.
- SUBJECT 16: excluded because all 3 of its scorable awakenings carry anger = 0; AUC
  is mathematically undefined on a single-class test set (the metric raises, no number
  ever existed to see). The n = 16 count appears in the first anger arm (ref17,
  2026-08-28) which predates the AUC analysis (nulls begun 08-29 08:07): the exclusion
  is forced by definition and predates the result. No fork.
- DEPLOYMENT IMPLICATION: demeaning failing to repair the sign closes the easy path
  (per-subject normalization + fixed sign). Any cross-subject anger decoder requires an
  explicitly learned sign inversion, which is precisely what the pre-registered Tononi
  test exists to validate.
