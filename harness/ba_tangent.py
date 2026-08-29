"""101-Nights body_action: tangent space + shrinkage LDA on the SAME folds as ShallowConv.

Reuses the project's own dataset class and fold function verbatim, so the comparison to
the ShallowConv 0.800 is fold-matched by construction rather than by re-implementation.
PREREG addendum 3: primary = the 7 MOTOR_STRIP_CHANS band-passed to MOTOR_BAND_HZ.
"""
import sys, json, argparse, numpy as np
sys.path.insert(0, "/home/ninon/projects/reaDream/scripts")
from dance_101nights_body_action_data import (
    Dance101NightsBodyActionDataset, MOTOR_STRIP_CHANS, MOTOR_BAND_HZ, SF)
from train_shallowconv_body_action_cv import stratified_kfold_by_night, build_tensors
from scipy.signal import butter, filtfilt
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="motor", choices=["motor", "pca32"])
ap.add_argument("--shuffle", type=int, default=0)
ap.add_argument("--n-folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", required=True)
a = ap.parse_args()

S = "/orcd/scratch/orcd/010/ninon/reaDream"
R = "/home/ninon/projects/reaDream"
dense = f"{S}/dream_dense_training_pairs_v3_256ch_fp16.pt"
csv = f"{R}/evaluation/101-Nights-Final-dreamsheet_LABELED_filter.csv"
Xs, ys, ns = [], [], []
for sp in ("train", "val"):
    ds = Dance101NightsBodyActionDataset(splits=(sp,), dense_pt=dense, labeled_csv=csv)
    x, y, n = build_tensors(ds)
    Xs.append(x.numpy().astype(np.float64)); ys.append(y.numpy()); ns.append(n.numpy())
X = np.concatenate(Xs); y = np.concatenate(ys).astype(int); nights = np.concatenate(ns)
print("CV pool: %d windows, %d nights, %d pos" % (len(X), len(np.unique(nights)), y.sum()), flush=True)

if a.shuffle:
    # shuffle labels at NIGHT level: nights are the CV unit, so shuffling windows would
    # leak the night structure and give a null that is far too tight.
    un = np.unique(nights)
    lab = {n: int(y[nights == n].max()) for n in un}
    perm = np.random.default_rng(a.shuffle).permutation([lab[n] for n in un])
    lab = dict(zip(un, perm))
    y = np.array([lab[n] for n in nights])
    print("*** NULL: night-level label shuffle, seed %d ***" % a.shuffle, flush=True)

if a.variant == "motor":
    Xr = X[:, MOTOR_STRIP_CHANS, :]
    b, aa = butter(4, [MOTOR_BAND_HZ[0]/(SF/2), MOTOR_BAND_HZ[1]/(SF/2)], btype="band")
    Xr = filtfilt(b, aa, Xr, axis=-1)
else:
    flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    p = PCA(n_components=32, random_state=0).fit(flat[::37])
    Xr = p.transform(flat).reshape(X.shape[0], X.shape[2], 32).transpose(0, 2, 1)
print("reduced to %s" % (Xr.shape,), flush=True)

folds = stratified_kfold_by_night(nights, y, a.n_folds, a.seed)
def bal(p, t):
    tp=((p==1)&(t==1)).sum(); fn=((p==0)&(t==1)).sum()
    tn=((p==0)&(t==0)).sum(); fp=((p==1)&(t==0)).sum()
    return float(((tp/(tp+fn) if tp+fn else 0)+(tn/(tn+fp) if tn+fp else 0))/2)

accs, aucs = [], []
for k, vn in enumerate(folds):
    te = np.isin(nights, [int(v) for v in vn]); tr = ~te
    if len(set(y[te].tolist())) < 2 or len(set(y[tr].tolist())) < 2: continue
    cov = Covariances(estimator="oas")
    Ctr, Cte = cov.fit_transform(Xr[tr]), cov.transform(Xr[te])
    ts = TangentSpace(metric="riemann")
    Ttr, Tte = ts.fit_transform(Ctr), ts.transform(Cte)
    m = LDA(solver="lsqr", shrinkage="auto").fit(Ttr, y[tr])
    accs.append(bal(m.predict(Tte), y[te]))
    aucs.append(float(roc_auc_score(y[te], m.decision_function(Tte))))
    print("  fold %d n_test=%d bal=%.4f auc=%.4f" % (k, te.sum(), accs[-1], aucs[-1]), flush=True)

res = {"variant": a.variant, "shuffle": a.shuffle, "n_folds": len(accs),
       "bal_acc": float(np.mean(accs)), "auc": float(np.mean(aucs)),
       "per_fold_bal": accs, "per_fold_auc": aucs}
json.dump(res, open(a.out, "w"), indent=1)
print("\nTANGENT body_action (%s): bal.acc %.4f | AUC %.4f  over %d folds"
      % (a.variant, res["bal_acc"], res["auc"], len(accs)), flush=True)
