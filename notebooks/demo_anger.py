"""Standalone Demo 1: the inverted anger pattern. Immune to stale notebooks.
Run in Colab:  %run demo_anger.py   (from the notebooks/ directory of a fresh checkout)
"""
import subprocess, sys, hashlib, csv, json
from pathlib import Path
import numpy as np

for pkg in ("pyriemann==0.12", "gdown"):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", pkg], check=True)
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

NPZ_URL = "https://drive.google.com/file/d/1i96wgL6mapOl5DfIxfcZITBmPhbKfWRL/view"
NPZ_MD5 = "c3bb5bfad1312f3c898c126df0ae9e1a"
DATA = Path("remturku_data"); DATA.mkdir(exist_ok=True)
npz_path = DATA / "remturku_epochs.npz"
if not npz_path.exists():
    import gdown
    gdown.download(NPZ_URL, str(npz_path), fuzzy=True, quiet=False)
    assert hashlib.md5(npz_path.read_bytes()).hexdigest() == NPZ_MD5, "bad download, rerun"

HV = {"anger": ["SR_NA1", "SR_NA7"]}
rat = {r["Filename"]: r for r in csv.DictReader(open("data/Ratings.csv", encoding="utf-8-sig"))}
rec = {r["Filename"]: r for r in csv.DictReader(open("data/Records.csv", encoding="utf-8-sig"))}
npz = np.load(npz_path)

def f(v):
    try: return float(v)
    except Exception: return None

Xe, ye, se = [], [], []
for fn, r in rat.items():
    if f"{fn}|raw" not in npz.files or fn not in rec: continue
    if not any((f(r[c]) or 0) > 0 for c in r if c.startswith("SR_")): continue
    x = np.asarray(npz[f"{fn}|raw"], np.float64)
    y = int(any((f(r[c]) or 0) > 0 for c in HV["anger"]))
    Xe.append(x); ye += [y] * len(x); se += [rec[fn]["Subject ID"]] * len(x)
Xe = np.concatenate(Xe); ye = np.array(ye); se = np.array(se)
print(f"loaded: {Xe.shape[0]} epochs, {len(np.unique(se))} subjects")

C = Covariances(estimator="oas").transform(Xe)
aucs = {}
for held in np.unique(se):
    tr, te = se != held, se == held
    if len(np.unique(ye[te])) < 2: continue
    ts = TangentSpace().fit(C[tr])
    clf = LDA(solver="lsqr", shrinkage="auto").fit(ts.transform(C[tr]), ye[tr])
    aucs[held] = roc_auc_score(ye[te], clf.decision_function(ts.transform(C[te])))
    print(f"held-out subject {held}: AUC = {aucs[held]:.3f}")
below = sum(1 for v in aucs.values() if v < 0.5)
print(f"\nmean cross-subject AUC (anger): {np.mean(list(aucs.values())):.4f}  chance 0.50")
print(f"subjects below chance: {below}/{len(aucs)}  (frozen record: 0.335, 14/16, "
      f"hypothesis-generating per MULTIPLICITY_LEDGER.md)")
