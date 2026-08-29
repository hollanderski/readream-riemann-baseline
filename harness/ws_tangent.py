"""REM_Turku WITHIN-SUBJECT: tangent space + shrinkage LDA, leave-one-awakening-out.

Per PREREG addendum 3. Train and test inside one dreamer, never across. The tangent space
is fitted on that subject's TRAINING epochs only. A held-out awakening carries one label,
so its prediction is the MEAN of its epoch decision values and balanced accuracy is scored
once per subject over its awakenings.
"""
import sys, json, csv, io, zipfile, argparse, numpy as np, statistics as st
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

T = "/orcd/scratch/orcd/010/ninon/reaDream/turku"
HV = {"anger": ["SR_NA1", "SR_NA7"], "apprehension": ["SR_NA9", "SR_NA10"],
      "confusion": ["SR_PA2"]}
NEG = ["SR_NA1", "SR_NA7", "SR_NA8", "SR_NA9", "SR_NA10"]
POS = ["SR_PA1", "SR_PA4", "SR_PA7", "SR_PA8", "SR_PA10"]

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--shuffle", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

z = zipfile.ZipFile(f"{T}/REM_Turku.zip")
rat = {r["Filename"]: r for r in csv.DictReader(io.StringIO(z.read("REM_Turku/Data/Ratings.csv").decode("utf-8-sig")))}
rec = {r["Filename"]: r for r in csv.DictReader(io.StringIO(z.read("REM_Turku/Records.csv").decode("utf-8-sig")))}
npz = np.load(f"{T}/remturku_epochs.npz")
def f(v):
    try: return float(v)
    except Exception: return None

X, y, s = [], [], []
for fn, r in rat.items():
    k = f"{fn}|raw"
    if k not in npz or fn not in rec: continue
    if not any((f(r[c]) or 0) > 0 for c in r if c.startswith("SR_")): continue
    if a.target == "negpos":
        n = max([f(r[c]) or 0 for c in NEG]); p = max([f(r[c]) or 0 for c in POS])
        if n == p: continue                      # PREREG: ties are DROPPED
        lab = int(n > p)
    else:
        lab = int(any((f(r[c]) or 0) > 0 for c in HV[a.target]))
    X.append(npz[k]); y.append(lab); s.append(rec[fn]["Subject ID"])
y = np.array(y); s = np.array(s)
print("target=%s  %d awakenings, %d subjects, %d positive" % (a.target, len(y), len(set(s)), y.sum()), flush=True)

def bal(p, t):
    tp=((p==1)&(t==1)).sum(); fn=((p==0)&(t==1)).sum()
    tn=((p==0)&(t==0)).sum(); fp=((p==1)&(t==0)).sum()
    return float(((tp/(tp+fn) if tp+fn else 0)+(tn/(tn+fp) if tn+fp else 0))/2)

if a.shuffle:
    rng = np.random.default_rng(a.shuffle)
    for u in np.unique(s):
        m = s == u; y[m] = rng.permutation(y[m])

out, skipped = {}, []
for sid in sorted(set(s), key=int):
    idx = np.where(s == sid)[0]; ys = y[idx]
    if len(idx) < 4 or ys.sum() < 2 or (len(ys) - ys.sum()) < 2:
        skipped.append((sid, len(idx), int(ys.sum()))); continue
    preds, truth = [], []
    for i in idx:
        tr = np.array([j for j in idx if j != i])
        if len(set(y[tr].tolist())) < 2: continue
        def win(ix): return np.concatenate([X[j][::2] for j in ix]) * 1e6
        Xtr, Xte = win(tr), X[i][::2] * 1e6
        ytr = np.concatenate([[y[j]] * len(X[j][::2]) for j in tr])
        cov = Covariances(estimator="oas")
        Ctr, Cte = cov.fit_transform(Xtr), cov.transform(Xte)
        ts = TangentSpace(metric="riemann")
        Ttr, Tte = ts.fit_transform(Ctr), ts.transform(Cte)
        m = LDA(solver="lsqr", shrinkage="auto").fit(Ttr, ytr)
        preds.append(int(m.decision_function(Tte).mean() > 0)); truth.append(int(y[i]))
    if len(set(truth)) == 2:
        out[sid] = bal(np.array(preds), np.array(truth))
        print("  subj %-3s n_awak=%2d pos=%2d awakening-level bal.acc=%.3f"
              % (sid, len(idx), int(ys.sum()), out[sid]), flush=True)

res = {"target": a.target, "shuffle": a.shuffle, "per_subject": out,
       "skipped": skipped, "n_scored": len(out)}
json.dump(res, open(a.out, "w"), indent=1)
if out:
    v = list(out.values())
    print("\nWITHIN-SUBJECT tangent %s: mean %.4f sd %.4f over %d subjects (%d skipped)"
          % (a.target, st.mean(v), st.pstdev(v), len(v), len(skipped)), flush=True)
