"""AUC alongside balanced accuracy. AUC is threshold-free, so it is immune to the
per-subject calibration shift measured on anger: high-prevalence subjects were
under-predicted by 0.310 and low-prevalence ones over-predicted by 0.301, i.e. every
subject regressed toward ~0.45 regardless of its own prior. If AUC sits at 0.5 the seven
null arms stand. If AUC is above 0.5 while balanced accuracy is below, the signal is real
and the DECISION THRESHOLD is what is broken."""
import sys, json, numpy as np, statistics as st
sys.path.insert(0, "/orcd/scratch/orcd/010/ninon/reaDream/turku")
from ladder import load, OUT, bal, recentre, T
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

def arm(target, rung, sub=2):
    X, y, s = load(target)
    B, A = {}, {}
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
        if rung >= 2: Ctr = recentre(Ctr, str_); Cte = recentre(Cte, ste)
        ts = TangentSpace(metric="riemann")
        Ttr, Tte = ts.fit_transform(Ctr), ts.transform(Cte)
        m = LDA(solver="lsqr", shrinkage="auto").fit(Ttr, ytr)
        d = m.decision_function(Tte)
        B[held] = bal(m.predict(Tte), yte)
        A[held] = float(roc_auc_score(yte, d)) if len(set(yte.tolist())) == 2 else None
    return B, A

for target in ("apprehension", "anger", "confusion"):
    for rung in (1, 2):
        B, A = arm(target, rung)
        av = [v for v in A.values() if v is not None]
        print("%-13s rung %d  bal.acc %.4f | AUC %.4f  (n=%d)"
              % (target, rung, st.mean(list(B.values())), st.mean(av), len(av)), flush=True)
        print("%-13s          per-subject AUC %s" % ("", [round(x,2) for x in sorted(av)]), flush=True)
    print(flush=True)
