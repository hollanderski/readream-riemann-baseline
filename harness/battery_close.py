"""Close pass for the anger robustness battery: merge items 1-5 + the null, final p.

Run only once the null array has CLOSED. A p computed while draws are still landing
is a statement about how many jobs finished, not about the effect.

What this refuses to do:
  - report a p whose exceedance count is 0 (it is at the floor 1/(n+1), which is a
    bound, not a measurement)
  - silently accept overlapping seed ranges (two chunks sharing seeds double-count
    draws, inflating n and understating p)
  - accept seed 0 anywhere (seed 0 means NO shuffle: the "null" draw is the observed)

Usage: python3 battery_close.py --results-dir <dir> --out merged.json
"""
import argparse, glob, json, os, sys

CAVEAT = ("mean over 16 scorable subjects of 17 (one subject single-class); "
          "awakenings per subject 2-12, so per-subject AUCs differ in precision")


def collect_null(results_dir, chunk_size=40):
    """Concatenate the null chunks, refusing any seed collision or seed 0."""
    files = sorted(glob.glob(os.path.join(results_dir, "battery_null_*.json")))
    if not files:
        raise SystemExit("no null chunks found: refusing to report a p over an empty null")
    seen, draws, chunks = {}, [], []
    for f in files:
        d = json.load(open(f))
        off, dr = int(d["offset"]), d["draws"]
        if off <= 0:
            raise SystemExit(f"{os.path.basename(f)}: offset {off} includes seed 0, "
                             "which applies NO shuffle. This is the contamination that "
                             "already invalidated one result.")
        for k in range(len(dr)):
            s = off + k
            if s in seen:
                raise SystemExit(f"seed {s} appears in both {seen[s]} and "
                                 f"{os.path.basename(f)}: overlapping chunks would "
                                 "double-count draws and understate p")
            seen[s] = os.path.basename(f)
        bad = [x for x in dr if x != x]
        if bad:
            raise SystemExit(f"{os.path.basename(f)}: {len(bad)} NaN draws")
        if len(dr) != chunk_size:
            print(f"  note: {os.path.basename(f)} has {len(dr)} draws, not {chunk_size}")
        draws += list(dr)
        chunks.append((off, len(dr)))
    return draws, chunks


def permutation_p(observed, draws, side="lower"):
    """(exceedances + 1) / (n + 1). Returns the count too: a p is not interpretable
    without knowing whether its exceedance count is 0."""
    n = len(draws)
    if side == "lower":
        exc = sum(1 for d in draws if d <= observed)
    else:
        exc = sum(1 for d in draws if d >= observed)
    return (exc + 1) / (n + 1), exc, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-draws", type=int, default=0,
                    help="refuse to close if fewer draws than this landed")
    a = ap.parse_args()

    items_path = os.path.join(a.results_dir, "battery_anger.json")
    if not os.path.exists(items_path):
        raise SystemExit("battery_anger.json missing: items 1-5 have not landed")
    items = json.load(open(items_path))
    obs = float(items["item1_mean_auc"])

    draws, chunks = collect_null(a.results_dir)
    n = len(draws)
    if a.expect_draws and n < a.expect_draws:
        raise SystemExit(f"only {n} draws of an expected {a.expect_draws}: the array has "
                         "not closed. A p now would describe job completion, not the effect.")

    p_lo, exc_lo, _ = permutation_p(obs, draws, "lower")
    p_hi, exc_hi, _ = permutation_p(obs, draws, "upper")
    p_two = min(1.0, 2 * min(p_lo, p_hi))
    floor = 1.0 / (n + 1)
    at_floor = min(exc_lo, exc_hi) == 0

    mean = sum(draws) / n
    sd = (sum((d - mean) ** 2 for d in draws) / (n - 1)) ** 0.5
    srt = sorted(draws)
    pct = lambda q: srt[max(0, min(n - 1, int(round(q * (n - 1)))))]

    out = {
        "prereg": items.get("prereg"),
        "target": items.get("target"),
        "caveat": CAVEAT,
        "observed_mean_auc": obs,
        "null": {"n_draws": n, "n_chunks": len(chunks), "mean": mean, "sd": sd,
                 "p2.5": pct(0.025), "p50": pct(0.5), "p97.5": pct(0.975),
                 "min": srt[0], "max": srt[-1]},
        "p_lower": p_lo, "exceedances_lower": exc_lo,
        "p_upper": p_hi, "exceedances_upper": exc_hi,
        "p_two_sided": p_two,
        "p_floor": floor,
        "at_floor": at_floor,
        "items": {k: v for k, v in items.items() if k.startswith("item")},
    }
    if at_floor:
        out["floor_warning"] = (
            f"exceedance count is 0: p is pinned at the floor {floor:.4g} = 1/(n+1). "
            "This is an upper bound on how small p could be shown to be with n draws, "
            "not a measured p. Report it as '< {:.3g}' with n, never as a value.".format(floor))

    with open(a.out + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    json.load(open(a.out))

    print(f"observed mean AUC {obs:.4f}   ({CAVEAT})")
    print(f"null n={n} over {len(chunks)} chunks: mean {mean:.4f} sd {sd:.4f} "
          f"[{pct(0.025):.4f}, {pct(0.975):.4f}]")
    print(f"p_lower {p_lo:.4g} (exceedances {exc_lo})   "
          f"p_two-sided {p_two:.4g}   floor {floor:.4g}")
    if at_floor:
        print("AT FLOOR: " + out["floor_warning"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
