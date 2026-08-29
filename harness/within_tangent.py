"""REM_Turku WITHIN-SUBJECT Riemannian baseline.

Per dreamer: OAS covariance per 2 s epoch -> tangent space (Riemannian mean of the
subject's TRAINING epochs) -> shrinkage LDA. Leave-one-AWAKENING-out inside the subject.
Awakening-level prediction = sign of the mean LDA decision value over the held-out
awakening's epochs. Epoch-level balanced accuracy reported alongside.

Data loading, label definitions (HV) and balanced accuracy are IMPORTED from ladder.py
so the features and labels are byte-identical to the cross-subject arms.

Labels: apprehension, anger, confusion (HV, unchanged) + one PRE-DECLARED composite
`negaff`: 1 if the awakening's summed self-rated negative-affect items (SR_NA*) exceed
the summed positive-affect items (SR_PA*), else 0. Declared before any fit.

Null: within-subject label shuffle at AWAKENING level, N draws, same CV.
Usage: within_tangent.py <target> <lo> <hi>   (lo==0 also writes the observed run)
Outputs: results/within_tg_<target>_obs.json / _null<k>.json
"""
import sys, json, numpy as np, statistics as st
sys.path.insert(0, "/orcd/scratch/orcd/010/ninon/reaDream/turku")
import ladder  # reuse load(), HV, bal, rat, rec, npz, T
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

T = ladder.T
SUB = 2            # same epoch decimation as ladder.py
MIN_AWAK = 4       # subjects with fewer scorable awakenings are skipped and counted


def load_composite():
    """negaff composite: sum(SR_NA*) > sum(SR_PA*) per awakening. Same file filter as ladder.load."""
    X, y, s = [], [], []
    for fn, r in ladder.rat.items():
        k = f"{fn}|raw"
        if k not in ladder.npz or fn not in ladder.rec:
            continue
        if not any((ladder.f(r[c]) or 0) > 0 for c in r if c.startswith("SR_")):
            continue
        na = sum((ladder.f(r[c]) or 0) for c in r if c.startswith("SR_NA"))
        pa = sum((ladder.f(r[c]) or 0) for c in r if c.startswith("SR_PA"))
        X.append(ladder.npz[k]); y.append(int(na > pa)); s.append(ladder.rec[fn]["Subject ID"])
    return X, np.array(y), np.array(s)


def load(target):
    return load_composite() if target == "negaff" else ladder.load(target)


def run(target, shuffle=0):
    X, y, s = load(target)
    if shuffle:
        y = y.copy(); rng = np.random.default_rng(shuffle)
        for u in np.unique(s):
            m = s == u; y[m] = rng.permutation(y[m])
    out, skipped = {}, {}
    for sid in np.unique(s):
        idx = np.where(s == sid)[0]
        if len(idx) < MIN_AWAK or y[idx].sum() in (0, len(idx)):
            skipped[str(sid)] = {"n_awak": int(len(idx)), "n_pos": int(y[idx].sum())}
            continue
        awak_pred, awak_true, ep_pred, ep_true = [], [], [], []
        for held in idx:
            tr = [i for i in idx if i != held]
            ytr_a = y[tr]
            if ytr_a.sum() in (0, len(tr)):
                continue  # training fold single-class, skip this held-out awakening
            Xtr = np.concatenate([X[i][::SUB] for i in tr]) * 1e6
            ytr = np.concatenate([[y[i]] * len(X[i][::SUB]) for i in tr])
            Xte = X[held][::SUB] * 1e6
            cov = Covariances(estimator="oas")
            Ctr, Cte = cov.fit_transform(Xtr), cov.transform(Xte)
            ts = TangentSpace(metric="riemann")
            Ttr, Tte = ts.fit_transform(Ctr), ts.transform(Cte)
            clf = LDA(solver="lsqr", shrinkage="auto").fit(Ttr, ytr)
            d = clf.decision_function(Tte)
            ep_pred.extend((d > 0).astype(int)); ep_true.extend([y[held]] * len(d))
            awak_pred.append(int(d.mean() > 0)); awak_true.append(int(y[held]))
        if len(awak_true) < MIN_AWAK or sum(awak_true) in (0, len(awak_true)):
            skipped[str(sid)] = {"n_awak": int(len(idx)), "n_scored": len(awak_true)}
            continue
        out[str(sid)] = {
            "awak_bal_acc": ladder.bal(np.array(awak_pred), np.array(awak_true)),
            "epoch_bal_acc": ladder.bal(np.array(ep_pred), np.array(ep_true)),
            "n_awak": int(len(awak_true)), "n_pos": int(sum(awak_true)),
        }
    return out, skipped


def summary(o):
    a = [v["awak_bal_acc"] for v in o.values()]; e = [v["epoch_bal_acc"] for v in o.values()]
    return {"n_subjects": len(o),
            "awak_mean": float(st.mean(a)) if a else None,
            "awak_sd": float(st.pstdev(a)) if len(a) > 1 else None,
            "epoch_mean": float(st.mean(e)) if e else None}


if __name__ == "__main__":
    target, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if lo == 0:
        o, sk = run(target)
        json.dump({"target": target, "kind": "observed", "cv": "within-subject leave-one-awakening-out",
                   "features": "OAS cov, tangent(riemann), LDA lsqr shrinkage=auto, sub=2",
                   "per_subject": o, "skipped": sk, "summary": summary(o)},
                  open(f"{T}/results/within_tg_{target}_obs.json", "w"), indent=1)
        print(f"WITHIN {target} observed: {summary(o)} skipped={len(sk)}", flush=True)
    for k in range(max(lo, 1), hi):
        r, _ = run(target, shuffle=k)
        if not r:
            continue
        json.dump({"target": target, "kind": "null", "seed": k, "per_subject": r, "summary": summary(r)},
                  open(f"{T}/results/within_tg_{target}_null{k}.json", "w"), indent=1)
        print(f"  perm {k}: awak_mean={summary(r)['awak_mean']:.4f}", flush=True)
