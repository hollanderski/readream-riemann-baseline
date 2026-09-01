"""Battery item 5, redone without convergence theater.

The MixedLM version returned fixed_proj_p = NaN while reporting converged=True: a
singular random-effects fit dressed as success. Dropped on fundamental_ai's ruling
(2026-09-01 16:42Z). Replaced by the version that needs no optimizer to be trusted:

  per subject, OLS slope of label on the LOSO projection (within-subject),
  then a one-sample Wilcoxon on those slopes against zero.

The projection block is copied VERBATIM from battery_anger.py rather than rewritten,
so this is the same code path producing the same proj_awk / lab_awk / subj_awk that
items 3 and 5 used. Rewriting it would have made the comparison meaningless.

Usage: python3 item5b.py --target anger --out item5b_anger.json
"""
import sys, json, argparse
import numpy as np
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

T = "/orcd/scratch/orcd/010/ninon/reaDream/turku"
sys.path.insert(0, T)
import ladder
import safe_json

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="anger")
ap.add_argument("--out", required=True)
a = ap.parse_args()

Xf, y_awk, s_awk = ladder.load(a.target)

Xe, ye, se, ae = [], [], [], []
for i, (x, yy, ss) in enumerate(zip(Xf, y_awk, s_awk)):
    Xe.append(np.asarray(x, np.float64)); ye += [int(yy)] * len(x)
    se += [ss] * len(x); ae += [i] * len(x)
Xe = np.concatenate(Xe); ye = np.array(ye); se = np.array(se); ae = np.array(ae)
C = Covariances(estimator="oas").transform(Xe)

# ---- verbatim from battery_anger.py (items 3 & 5 projection)
proj_awk, lab_awk, subj_awk = [], [], []
for held in np.unique(se):
    tr, te = se != held, se == held
    if len(np.unique(ye[te])) < 2:
        continue
    ts = TangentSpace().fit(C[tr])
    clf = LDA(solver="lsqr", shrinkage="auto").fit(ts.transform(C[tr]), ye[tr])
    sc = clf.decision_function(ts.transform(C[te]))
    for ai in np.unique(ae[te]):
        m = ae[te] == ai
        proj_awk.append(float(np.mean(sc[m])))
        lab_awk.append(int(ye[te][m][0]))
        subj_awk.append(str(held))
proj_awk = np.array(proj_awk); lab_awk = np.array(lab_awk); subj_awk = np.array(subj_awk)
# ---- end verbatim

slopes, skipped = {}, {}
for u in np.unique(subj_awk):
    m = subj_awk == u
    x, y = proj_awk[m].astype(float), lab_awk[m].astype(float)
    if len(x) < 2:
        skipped[u] = f"only {len(x)} awakening(s)"; continue
    if np.std(x) == 0:
        skipped[u] = "no variance in projection"; continue
    if np.std(y) == 0:
        skipped[u] = "single-class within subject"; continue
    # OLS slope of label on projection, within subject
    slopes[u] = float(np.polyfit(x, y, 1)[0])

sl = np.array(list(slopes.values()))
n = len(sl)
if n < 3:
    raise SystemExit(f"only {n} usable slopes: a Wilcoxon here would be theater of a "
                     "different kind. Report the slopes and stop.")

stat, p = wilcoxon(sl, alternative="two-sided")
n_neg = int((sl < 0).sum())
# distribution-free CI for the median (order statistics, ~95%)
srt = np.sort(sl)
from math import comb
lo_i = next((k for k in range(n) if sum(comb(n, j) for j in range(k + 1)) / 2 ** n > 0.025), 0)
ci = (float(srt[lo_i]), float(srt[n - 1 - lo_i]))

out = {
    "target": a.target,
    "method": ("per-subject OLS slope of label on within-subject LOSO projection, "
               "one-sample Wilcoxon against zero. Replaces the MixedLM whose Wald p "
               "was NaN with converged=True."),
    "n_subjects_with_slope": n,
    "slopes": slopes,
    "skipped": skipped,
    "median_slope": float(np.median(sl)),
    "mean_slope": float(np.mean(sl)),
    "n_negative": f"{n_neg}/{n}",
    "median_ci95_orderstat": ci,
    "wilcoxon_stat": float(stat),
    "wilcoxon_p_two_sided": float(p),
}
safe_json.dump(out, a.out)
print(f"ITEM5b {a.target}: n={n} slopes, {n_neg}/{n} negative, "
      f"median {np.median(sl):+.4f} CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}], "
      f"Wilcoxon p={p:.4f}")
if skipped:
    print("  skipped:", skipped)
