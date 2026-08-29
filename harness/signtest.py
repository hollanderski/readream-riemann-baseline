"""Direct test of the sign-inversion reading for anger.

anger's pooled cross-subject AUC is 0.334, significantly BELOW chance (p=0.010, null
centred 0.5001). Inference so far: the EEG signature exists but its SIGN is subject-
dependent. That inference is NOT established, and one thing about it is geometrically
suspicious: if per-subject directions merely DISAGREED at random, the pooled decoder would
land at AUC 0.5, not 0.334. Systematic inversion needs each held-out subject to be
anti-aligned with the mean of the other 15, which cannot hold for all 16 at once unless
something structured is going on.

This fits one LDA direction per subject in a COMMON tangent space and measures:
  (a) pairwise cosine between per-subject directions
  (b) cosine between each subject's own direction and the LOSO direction trained on the
      other 15, which is the quantity that actually predicts the sign of its AUC
  (c) correlation between that cosine and the subject's held-out AUC
If (b) is systematically negative and (c) is positive, the mechanism is established.
If (b) scatters around 0, the inversion is NOT explained by per-subject sign flips and I
have the wrong story.
"""
import sys, json, numpy as np, statistics as st
sys.path.insert(0, "/orcd/scratch/orcd/010/ninon/reaDream/turku")
from ladder import load, OUT, bal, T
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

TARGET = sys.argv[1] if len(sys.argv) > 1 else "anger"
SUB = 2
X, y, s = load(TARGET)

# ONE common tangent space fitted on all subjects, so directions are comparable vectors.
xs, ys, ss = [], [], []
for i in range(len(X)):
    for e in X[i][::SUB]: xs.append(e); ys.append(y[i]); ss.append(s[i])
Xa, ya, sa = np.stack(xs)*1e6, np.array(ys), np.array(ss)
C = Covariances(estimator="oas").fit_transform(Xa)
Tall = TangentSpace(metric="riemann").fit_transform(C)
print("common tangent space: %d epochs x %d dims, %d subjects\n" % (*Tall.shape, len(set(sa))))

def direction(mask):
    m = LDA(solver="lsqr", shrinkage="auto").fit(Tall[mask], ya[mask])
    w = np.ravel(m.coef_); n = np.linalg.norm(w)
    return w / n if n else w

own, loso, auc = {}, {}, {}
for h in OUT:
    te = sa == h; tr = (sa != h) & np.isin(sa, OUT)
    if te.sum() == 0 or len(set(ya[te].tolist())) < 2: continue
    if len(set(ya[te].tolist())) == 2 and len(set(ya[tr].tolist())) == 2:
        own[h] = direction(te)
        loso[h] = direction(tr)
        d = Tall[te] @ loso[h]
        auc[h] = float(roc_auc_score(ya[te], d))

ks = sorted(own, key=int)
print("(b) cosine(subject's OWN direction, LOSO direction from the other 15) and its AUC")
print("  subj    cos(own, loso)     held-out AUC")
cs, au = [], []
for h in ks:
    c = float(own[h] @ loso[h]); cs.append(c); au.append(auc[h])
    print("  %4s      %+.3f            %.3f" % (h, c, auc[h]))
print("\n  mean cosine = %+.3f   (negative on most subjects = sign inversion)" % st.mean(cs))
print("  subjects with NEGATIVE cosine: %d / %d" % (sum(1 for c in cs if c < 0), len(cs)))

mx, my = st.mean(cs), st.mean(au)
num = sum((a-mx)*(b-my) for a, b in zip(cs, au))
den = (sum((a-mx)**2 for a in cs)*sum((b-my)**2 for b in au))**0.5
print("  (c) corr(cosine, AUC) = %+.3f  (positive = the cosine PREDICTS the AUC sign)" % (num/den if den else 0))

P = np.array([own[h] for h in ks])
G = P @ P.T
off = G[np.triu_indices_from(G, 1)]
print("\n(a) pairwise cosine between per-subject directions: mean %+.3f, sd %.3f, %d of %d negative"
      % (off.mean(), off.std(), int((off < 0).sum()), len(off)))
print("\nVERDICT: %s" % ("mechanism SUPPORTED: held-out subjects are anti-aligned with the pooled direction"
                         if st.mean(cs) < -0.05 else
                         "mechanism NOT supported: cosines scatter around 0, the inversion needs another explanation"))

# ---------------------------------------------------------------------------
# (b) Simpson decomposition, per fundamental_ai 14:32Z.
# For the pooled discriminative direction, correlate the projection with anger
#   (i)  ACROSS subject means, and
#   (ii) WITHIN subject after removing each subject's mean.
# Opposite signs = a between-subject covariate that reverses within subjects, i.e.
# Simpson's paradox. Same signs = the between/within story is wrong and the 0.334 stays
# an unexplained number with its null.
print("\n" + "="*72)
print("(b) SIMPSON DECOMPOSITION on the pooled direction")
wp = direction(np.isin(sa, OUT))
proj = Tall @ wp
subs_all = sorted(set(sa[np.isin(sa, OUT)]), key=int)
bm_x, bm_y = [], []
wi_x, wi_y = [], []
for h in subs_all:
    m = sa == h
    if m.sum() == 0: continue
    bm_x.append(float(proj[m].mean())); bm_y.append(float(ya[m].mean()))
    wi_x.extend((proj[m] - proj[m].mean()).tolist())
    wi_y.extend((ya[m] - ya[m].mean()).tolist())
def pear(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    den = (sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))**0.5
    return num/den if den else 0.0
rb = pear(bm_x, bm_y); rw = pear(wi_x, wi_y)
print("  (i)  BETWEEN subjects, %d subject means : r = %+.3f" % (len(bm_x), rb))
print("  (ii) WITHIN subjects, subject-mean removed: r = %+.3f" % rw)
print("  signs %s" % ("OPPOSE -> Simpson: a between-subject covariate reverses within subjects"
                      if rb*rw < 0 else
                      "AGREE -> not Simpson; the inverted AUC is not explained by this decomposition"))
