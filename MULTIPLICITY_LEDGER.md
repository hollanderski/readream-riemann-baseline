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

- p = 0.010 lower-tail at 199 draws, post hoc tail: two-sided ~ 0.02.
- Bonferroni over the 3 labels alone: ~ 0.06. Above 0.05 already.
- Corrected against the honest campaign-wide count (~38): dead by an order of magnitude.

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
