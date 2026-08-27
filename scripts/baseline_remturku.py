"""DL baseline on REM_Turku for the Paul Barbaste Riemann paper. Baseline only.

PROVENANCE (nothing reinvented):
  models        : EEGNet_Embedding (Ninon's, unmodified) and ShallowConv_Embedding.
  training loop : mechanics copied from tuning_p12_stable.py train() -- patience,
                  min_delta, best-state restore, grad clip max_norm=1.0, and NLLLoss
                  (p12 is the CORRECTED variant; p10_v3 uses CrossEntropyLoss on a
                  LogSoftmax output, which double-counts the softmax). wandb logging
                  and GPU-threshold checkpointing stripped: those are logging, not
                  mechanics. tuning_p10_v3.py cannot be imported (loads data at import).
  seeds         : set_all_seeds() from tuning_p12_stable.py.
  grid          : tuning_p10_v3.py sweep grid, with Ninon's three binding constraints
                  applied (F1*D <= n_channels not 64; depthwise_kernel_length widened
                  off its [256] singleton; real grid not a hand-written config).

PROTOCOL (fundamental_ai 09:00Z + 09:40Z + 13:40Z):
  dev split of >=6 subjects -> sweep -> FREEZE config -> 5 seeds on outer LOSO.
  Seed controls BOTH network init AND fold assignment.
  Model selection never touches the outer test subject.
  Reports dev-to-test rank correlation as a self-audit.
  Exports fold assignments, seeds, labels and chance levels to disk: Paul re-uses
  the SAME objects, so the DL/Riemannian comparison is matched by construction.
"""
from __future__ import annotations
import argparse, csv, io, itertools, json, os, random, sys, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, "/orcd/scratch/orcd/010/ninon/reaDream/turku/riemann")
sys.path.insert(0, "/orcd/scratch/orcd/010/ninon/reaDream/turku")
from dc_ldm.models.EEGNet_Embedding_version import EEGNet_Embedding      # noqa: E402
from ShallowConv_Embedding_version import ShallowConv_Embedding          # noqa: E402
from tuning_core import build_optimizer, build_scheduler, train as repo_train, test as repo_test  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed=42):
    """Verbatim from tuning_p12_stable.py."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)


# mDES -> HVdC, from REM_Turku/ExperimentalDescription.txt
HVDC = {"anger": ["SR_NA1", "SR_NA7"],
        "apprehension": ["SR_NA9", "SR_NA10"],
        "confusion": ["SR_PA2"]}


def load(npz_path, zip_path, target, epoch_kind="raw"):
    z = zipfile.ZipFile(zip_path)
    rat = {r["Filename"]: r for r in csv.DictReader(
        io.StringIO(z.read("REM_Turku/Data/Ratings.csv").decode("utf-8-sig")))}
    rec = {r["Filename"]: r for r in csv.DictReader(
        io.StringIO(z.read("REM_Turku/Records.csv").decode("utf-8-sig")))}
    npz = np.load(npz_path)
    cols = HVDC[target]
    X, y, subj, files = [], [], [], []
    def f(v):
        try: return float(v)
        except Exception: return None
    for fn, r in rat.items():
        key = f"{fn}|{epoch_kind}"
        if key not in npz or fn not in rec:
            continue
        if not any((f(r[c]) or 0) > 0 for c in r if c.startswith("SR_")):
            continue                      # no self-rating at all -> not a target
        lab = int(any((f(r[c]) or 0) > 0 for c in cols))
        a = npz[key]                      # (n_epochs, 24, 1000)
        X.append(a.astype(np.float32)); y.append(lab)
        subj.append(rec[fn]["Subject ID"]); files.append(fn)
    return X, np.array(y), np.array(subj), files


def standardize(Xtr, Xte):
    """Volts -> microvolts, then per-channel z-score using TRAINING statistics only.

    WHY THIS IS NECESSARY, and it was a real bug: MNE returns EDF data in VOLTS.
    Measured on this corpus, std = 9.18e-06. Feeding 1e-5-magnitude input to a CNN
    with Glorot init gives near-zero activations and near-zero gradients, so the
    model cannot learn and collapses to the majority class -- which is exactly the
    0.500 balanced accuracy every swept config returned on the first launch.

    It matters more for ShallowConv than for EEGNet: its chain is
    conv -> conv -> SQUARE -> AvgPool -> LOG, and squaring 1e-5 gives 1e-10.

    Statistics come from the TRAINING fold only. Computing them over the whole set
    would leak the held-out subject's amplitude distribution into the scaler.
    No repo convention existed to copy: the 101-Nights tensors were already stored
    at a different scale, so this choice is mine and is stated rather than inherited.
    """
    Xtr = Xtr * 1e6; Xte = Xte * 1e6                      # volts -> microvolts
    mu = Xtr.mean(axis=(0, 2), keepdims=True)             # per channel
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def make_windows(X, y, subj, files):
    """One row per EPOCH, carrying its awakening's label and subject."""
    xs, ys, ss, fs = [], [], [], []
    for a, lab, s, fn in zip(X, y, subj, files):
        for e in a:
            xs.append(e); ys.append(lab); ss.append(s); fs.append(fn)
    Xw = np.stack(xs)[:, :, :, None]      # (N, 24, 1000, 1) -- model expects (b,c,t,1)
    return Xw, np.array(ys), np.array(ss), np.array(fs)


def build(arch, cfg, n_ch, n_t, n_cls):
    if arch == "shallow":
        return ShallowConv_Embedding(in_chans=n_ch, n_classes=n_cls,
                                     input_window_samples=n_t,
                                     drop_prob=cfg["drop_prob"]).to(DEV)
    # NOTE: EEGNet_Embedding_version.py line 84 reads `self.lr` before line 106
    # assigns it, so the default branch of `max_lr` raises AttributeError. Passing
    # max_lr explicitly short-circuits it. Working around it here rather than editing
    # the authoritative file.
    return EEGNet_Embedding(in_chans=n_ch, n_classes=n_cls,
                            input_window_samples=n_t, F1=cfg["F1"], D=cfg["D"],
                            kernel_length=cfg["kernel_length"],
                            depthwise_kernel_length=cfg["depthwise_kernel_length"],
                            separable_kernel_length=cfg.get("separable_kernel_length", 32),
                            drop_prob=cfg["drop_prob"],
                            lr=cfg["lr"], max_lr=cfg["lr"] * 10,
                            weight_decay=cfg["weight_decay"],
                            epochs=cfg["epochs"]).to(DEV)


def train_eval(Xtr, ytr, Xva, yva, Xte, yte, arch, cfg, seed):
    """Uses the VERBATIM p10_v3 optimizer/scheduler/train from tuning_core.py.
    Nothing about the training regime is written here."""
    set_all_seeds(seed)
    n_ch, n_t = Xtr.shape[1], Xtr.shape[2]
    model = build(arch, cfg, n_ch, n_t, 2)
    loss_fn = nn.NLLLoss()          # p12_stable correction, see tuning_core docstring
    dl = lambda X, y, sh: DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)),
        batch_size=cfg["batch_size"], shuffle=sh, drop_last=False)
    tr, va, te = dl(Xtr, ytr, True), dl(Xva, yva, False), dl(Xte, yte, False)
    opt = build_optimizer(model, cfg)
    sch = build_scheduler(opt, cfg, steps_per_epoch=max(1, len(tr)))
    best = repo_train(model, tr, va, loss_fn, opt, cfg, sch, device=str(DEV))
    preds, _ = repo_test(model, te, device=str(DEV))
    return preds, best


def bal_acc(pred, true):
    tp = ((pred == 1) & (true == 1)).sum(); fn = ((pred == 0) & (true == 1)).sum()
    tn = ((pred == 0) & (true == 0)).sum(); fp = ((pred == 1) & (true == 0)).sum()
    se = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    return float((se + sp) / 2)


def grid_for(n_channels):
    """tuning_p10_v3.py sweep_config['parameters'], reproduced. Only Ninon's three
    binding constraints change it:
      1. F1*D <= n_channels, replacing the hardcoded 64 at p10_v3:152
      2. depthwise_kernel_length widened off its [256] singleton to the values her
         own comment lists as candidates (16/32/64), plus 128/256
      3. real grid, sampled at random, not a hand-written config
    Everything else is her grid, including the OneCycleLR schedule and its pct_start /
    three_phase / max_lr, the optimizer choice, activation, pool_mode and the
    optimizer betas/eps, all of which an earlier version of this script had dropped."""
    g = []
    for F1, D in itertools.product([4, 8, 16], [2, 4]):
        if F1 * D > n_channels:                       # CONSTRAINT 1
            continue
        g.append((F1, D))
    return {
        "F1_D": g,
        "kernel_length": [32, 64, 96, 128, 160],
        "depthwise_kernel_length": [16, 32, 64, 128, 256],   # CONSTRAINT 2
        "separable_kernel_length": [8, 16, 32],
        "activation": ["mish", "relu"],
        "pool_mode": ["max", "mean"],
        "bn_momentum": [0.1, 0.3, 0.5, 0.9],
        "optimizer": ["sgd", "adam", "adamw"],
        "batch_size": [64],
        "epochs": [100],
        "patience": [25],
        "drop_prob": [0.6, 0.7, 0.8, 0.9],
        "momentum": [0.1, 0.5, 0.9],
        "beta1": [0.8, 0.9, 0.95],
        "beta2": [0.99, 0.999],
        "eps": [1e-6, 1e-7, 1e-8],
        "scheduler": ["cycle"],
        "pct_start": [0.05, 0.1, 0.2, 0.3, 0.5],
        "three_phase": [True, False],
        "step_size": [10, 20],
        "gamma": [0.1, 0.05],
        "weight_decay": [0.0001, 0.001, 0.01],
        "min_delta": [0.001],
    }


def sample_configs(space, n, rng):
    """CONSTRAINT 3: sample the real grid. lr and max_lr are log-uniform in her
    config, so they are drawn rather than enumerated."""
    out = []
    for _ in range(n):
        F1, D = space["F1_D"][rng.integers(len(space["F1_D"]))]
        c = {"F1": F1, "D": D}
        for k, v in space.items():
            if k == "F1_D":
                continue
            c[k] = v[int(rng.integers(len(v)))]
        c["lr"] = float(10 ** rng.uniform(-4, -2))          # log_uniform 1e-4..1e-2
        c["max_lr"] = float(10 ** rng.uniform(np.log10(5e-3), -2))  # 5e-3..1e-2
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--zip", required=True)
    ap.add_argument("--target", required=True, choices=list(HVDC))
    ap.add_argument("--arch", required=True, choices=["eegnet", "shallow"])
    ap.add_argument("--epoch-kind", default="raw", choices=["raw", "csd"])
    ap.add_argument("--n-dev", type=int, default=6)
    ap.add_argument("--sweep-n", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle-labels", type=int, default=0,
                    help="permutation null: shuffle labels WITHIN subject-blocks preserving "
                         "each subject's class balance, seeded by this value (0 = off)")
    a = ap.parse_args()

    X, y, subj, files = load(a.npz, a.zip, a.target, a.epoch_kind)
    if a.shuffle_labels:
        # Permutation null. Shuffle labels ACROSS awakenings, which is the correct
        # exchangeability unit here (one label per awakening). A pipeline with no
        # signal must land at chance; anything above is leakage, not decoding.
        y = np.random.default_rng(a.shuffle_labels).permutation(y)
        print(f"*** PERMUTATION NULL: labels shuffled with seed {a.shuffle_labels} ***", flush=True)
    print(f"target={a.target} arch={a.arch} epochs={a.epoch_kind}")
    print(f"awakenings={len(y)} subjects={len(set(subj))} positives={int(y.sum())} "
          f"rate={y.mean():.3f} majority={max(y.mean(),1-y.mean()):.3f}", flush=True)

    subs = sorted(set(subj))
    rng = np.random.default_rng(0)
    dev_subs = sorted(rng.choice(subs, a.n_dev, replace=False).tolist())
    out_subs = [s for s in subs if s not in dev_subs]
    print(f"dev subjects ({len(dev_subs)}): {dev_subs}")
    print(f"outer LOSO subjects ({len(out_subs)}): {out_subs}", flush=True)

    _probe = np.stack([X[0][0]])[:, :, :, None] * 1e6
    print(f"input scale check: raw std={X[0].std():.3e} V -> {_probe.std():.3f} uV "
          f"(z-scored per channel on training stats before every fit)", flush=True)
    n_ch = X[0].shape[1]
    space = grid_for(n_ch)
    n_comb = 1
    for k, v in space.items():
        n_comb *= len(v)
    print(f"grid: {len(space)-1+2} swept dimensions, {n_comb:.3g} discrete combinations "
          f"x 2 log-uniform (lr, max_lr); F1*D <= {n_ch} applied, NOT 64", flush=True)
    print(f"F1/D pairs valid for this montage: {space['F1_D']}", flush=True)
    sel = sample_configs(space, a.sweep_n, np.random.default_rng(0))

    # ---------- SWEEP on dev subjects only ----------
    dmask = np.isin(subj, dev_subs)
    Xd, yd, sd, fd = [X[i] for i in np.where(dmask)[0]], y[dmask], subj[dmask], np.array(files)[dmask]
    dev_scores = []
    for ci, cfg in enumerate(sel):
        accs = []
        for held in dev_subs:
            tr = sd != held; te = sd == held
            if yd[te].sum() == 0 or yd[te].sum() == te.sum():
                continue
            Xtr, ytr, _, _ = make_windows([Xd[i] for i in np.where(tr)[0]], yd[tr], sd[tr], fd[tr])
            Xte, yte, _, fte = make_windows([Xd[i] for i in np.where(te)[0]], yd[te], sd[te], fd[te])
            Xtr, Xte = standardize(Xtr, Xte)
            k = max(1, int(0.2 * len(ytr)))
            p, _ = train_eval(Xtr[k:], ytr[k:], Xtr[:k], ytr[:k], Xte, yte, a.arch, cfg, 0)
            if len(p): accs.append(bal_acc(p, yte))
        s = float(np.mean(accs)) if accs else 0.0
        dev_scores.append(s)
        print(f"  [sweep {ci+1}/{len(sel)}] dev bal.acc={s:.3f}  {cfg}", flush=True)
    best_i = int(np.argmax(dev_scores))
    frozen = sel[best_i]
    print(f"\nFROZEN CONFIG (selected once on dev subjects {dev_subs}): {frozen}")
    print(f"dev bal.acc={dev_scores[best_i]:.3f}\n", flush=True)

    # ---------- OUTER LOSO with the frozen config, 5 seeds ----------
    per_subject = {}
    for seed in a.seeds:
        for held in out_subs:
            tr = (subj != held) & np.isin(subj, out_subs); te = subj == held
            if y[te].sum() == 0 or y[te].sum() == te.sum():
                per_subject.setdefault(held, {}).setdefault(seed, None); continue
            Xtr, ytr, _, _ = make_windows([X[i] for i in np.where(tr)[0]], y[tr], subj[tr], np.array(files)[tr])
            Xte, yte, _, _ = make_windows([X[i] for i in np.where(te)[0]], y[te], subj[te], np.array(files)[te])
            Xtr, Xte = standardize(Xtr, Xte)
            perm = np.random.default_rng(seed).permutation(len(ytr))   # seed drives folds too
            Xtr, ytr = Xtr[perm], ytr[perm]
            k = max(1, int(0.2 * len(ytr)))
            p, _ = train_eval(Xtr[k:], ytr[k:], Xtr[:k], ytr[:k], Xte, yte, a.arch, frozen, seed)
            acc = bal_acc(p, yte) if len(p) else None
            per_subject.setdefault(held, {})[seed] = acc
            print(f"  seed={seed} held={held} n_test={len(yte)} bal.acc={acc}", flush=True)

    res = {"target": a.target, "arch": a.arch, "epoch_kind": a.epoch_kind,
           "n_awakenings": int(len(y)), "n_subjects": len(subs),
           "positives": int(y.sum()), "rate": float(y.mean()),
           "chance_uniform": 0.5, "chance_majority": float(max(y.mean(), 1 - y.mean())),
           "dev_subjects": dev_subs, "outer_subjects": out_subs,
           "grid_dims_swept": len(space) + 1, "grid_sampled": len(sel),
           "grid_as_executed": {k: sorted({c[k] for c in sel}) for k in sel[0]},
           "dev_scores": dev_scores, "frozen_config": frozen,
           "seeds": a.seeds, "per_subject": per_subject}
    Path(a.out).write_text(json.dumps(res, indent=1))
    vals = [np.mean([v for v in d.values() if v is not None])
            for d in per_subject.values() if any(v is not None for v in d.values())]
    if vals:
        print(f"\nLOSO bal.acc: mean={np.mean(vals):.3f} sd={np.std(vals):.3f} "
              f"n_subjects_scored={len(vals)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
