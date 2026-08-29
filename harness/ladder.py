"""Alignment ladder on the tangent arm. Same 14 LOSO folds as the DL reference.

Rung 1 = global reference (what tg_matched.py did). Rung 2 = per-subject recentring,
the PRIMARY per PREREG addendum 2. The held-out subject is recentred with its OWN
UNLABELLED Riemannian mean, which uses no test labels and is legal under LOSO.
"""
import sys, json, csv, io, zipfile, numpy as np, statistics as st
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

T = "/orcd/scratch/orcd/010/ninon/reaDream/turku"
HV = {"anger": ["SR_NA1", "SR_NA7"], "apprehension": ["SR_NA9", "SR_NA10"], "confusion": ["SR_PA2"]}
ref = json.load(open(f"{T}/results/ref17_apprehension_s0.json"))
OUT = [str(x) for x in ref["outer_subjects"]]
z = zipfile.ZipFile(f"{T}/REM_Turku.zip")
rat = {r["Filename"]: r for r in csv.DictReader(io.StringIO(z.read("REM_Turku/Data/Ratings.csv").decode("utf-8-sig")))}
rec = {r["Filename"]: r for r in csv.DictReader(io.StringIO(z.read("REM_Turku/Records.csv").decode("utf-8-sig")))}
npz = np.load(f"{T}/remturku_epochs.npz")

def f(v):
    try: return float(v)
    except Exception: return None

def bal(p, t):
    tp=((p==1)&(t==1)).sum(); fn=((p==0)&(t==1)).sum()
    tn=((p==0)&(t==0)).sum(); fp=((p==1)&(t==0)).sum()
    se = tp/(tp+fn) if tp+fn else 0.0
    sp = tn/(tn+fp) if tn+fp else 0.0
    return float((se+sp)/2)

def load(target):
    X, y, s = [], [], []
    for fn, r in rat.items():
        k = f"{fn}|raw"
        if k not in npz or fn not in rec: continue
        if not any((f(r[c]) or 0) > 0 for c in r if c.startswith("SR_")): continue
        X.append(npz[k]); y.append(int(any((f(r[c]) or 0) > 0 for c in HV[target])))
        s.append(rec[fn]["Subject ID"])
    return X, np.array(y), np.array(s)

def recentre(C, subj_of_epoch):
    """Whiten each subject's covariances by its OWN Riemannian mean (Zanini recentring).
    Uses no labels, so it is legal for the held-out subject too."""
    out = np.empty_like(C)
    for sid in np.unique(subj_of_epoch):
        m = subj_of_epoch == sid
        M = mean_riemann(C[m])
        w, V = np.linalg.eigh(M)
        Mi = V @ np.diag(1.0/np.sqrt(np.maximum(w, 1e-12))) @ V.T
        out[m] = Mi @ C[m] @ Mi
    return out

def run(target, rung, shuffle=0, sub=2, clf="lda"):
    X, y, s = load(target)
    if shuffle:
        y = y.copy(); rng = np.random.default_rng(shuffle)
        for u in np.unique(s):
            m = s == u; y[m] = rng.permutation(y[m])
    out = {}
    for held in OUT:
        tr = (s != held) & np.isin(s, OUT); te = s == held
        if te.sum() == 0 or y[te].sum() in (0, te.sum()): continue
        def win(idx):
            xs, ys, ss = [], [], []
            for i in np.where(idx)[0]:
                for e in X[i][::sub]: xs.append(e); ys.append(y[i]); ss.append(s[i])
            return np.stack(xs)*1e6, np.array(ys), np.array(ss)
        Xtr, ytr, str_ = win(tr); Xte, yte, ste = win(te)
        cov = Covariances(estimator="oas")
        Ctr, Cte = cov.fit_transform(Xtr), cov.transform(Xte)
        if rung >= 2:
            Ctr = recentre(Ctr, str_)
            Cte = recentre(Cte, ste)      # held-out subject uses its OWN unlabelled mean
        ts = TangentSpace(metric="riemann")
        Ttr, Tte = ts.fit_transform(Ctr), ts.transform(Cte)
        model = LDA(solver="lsqr", shrinkage="auto") if clf == "lda" else LogisticRegression(max_iter=1000)
        model.fit(Ttr, ytr)
        out[held] = bal(model.predict(Tte), yte)
    return out

if __name__ == "__main__":
    target, rung = sys.argv[1], int(sys.argv[2])
    lo, hi = int(sys.argv[3]), int(sys.argv[4])
    if lo == 0:
        o = run(target, rung)
        json.dump({"target": target, "rung": rung, "kind": "observed", "per_subject": o},
                  open(f"{T}/results/ladder_r{rung}_{target}_obs.json", "w"), indent=1)
        print(f"RUNG {rung} {target} observed = {st.mean(list(o.values())):.4f} n={len(o)}", flush=True)
    for k in range(max(lo, 1), hi):
        r = run(target, rung, shuffle=k)
        if not r: continue
        json.dump({"target": target, "rung": rung, "kind": "null", "seed": k, "per_subject": r},
                  open(f"{T}/results/ladder_r{rung}_{target}_null{k}.json", "w"), indent=1)
        print(f"  perm {k} = {st.mean(list(r.values())):.4f}", flush=True)
