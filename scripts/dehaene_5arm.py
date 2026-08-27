"""Dehaene 5-arm design on 101-Nights: is dream emotion stage-specific or a trait?

Ninon's design, 2026-08-26. One matrix, one segment per row, the SAME reported
emotion as target for every row of a night, the same permutation null everywhere.

    arm             role
    wake_report     positive control, ceiling
    wake_bedtime    CONFOUND CONTROL, must be null
    rem_last        the standard hypothesis
    n2_pre_rem      priming
    n3_night        H-S2 extended to valence

Reading rule: if rem_last or n3_night predicts AND wake_bedtime does not, the signal
is dream-specific. If wake_bedtime predicts equally well, we measured a TRAIT and the
paper must say so.

Anchored in the locked H-S1/H-S2 spec (CLAUDE.md): slow-wave sleep builds the
structural scaffold, REM draws on it. Syntax in NREM, semantics in REM.

Only 101-Nights can carry this: it is the only corpus with full staged nights.
REM_Turku is 2 min of REM per awakening and nothing else; Zhang & Wamsley is
pre-awakening segments only.

CRITICAL: the null permutes labels at NIGHT level, never at segment level. Permuting
segments leaks the within-night structure and inflates every arm.

=== FORMAT CONTRACT, MUST BE VERIFIED BEFORE RUNNING (ssh was down at authoring) ===
This script assumes, and asserts at load:
  nrem_blocks/night_<n>_meta.json : {"night", "stage_counts": {W,N1,N2,N3,REM},
                                     "last_rem_epoch", "block_info":[{start_epoch,
                                     end_epoch, n_channels, n_samples}, ...]}
  per-night EEG tensor            : (n_epochs, 256, samples_per_epoch), 30 s epochs
  dreamy_enriched_labels.json     : {night_id: {"dreamy_labels": {emo_*: bool}}}
Ask Ninon rather than guessing if any of these is wrong
(feedback_never_hallucinate_reimplementation_use_authoritative_source).
"""
from __future__ import annotations
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path

import numpy as np

HVDC = ["emo_happiness", "emo_apprehension", "emo_anger", "emo_sadness", "emo_confusion"]
ARMS = ["wake_report", "wake_bedtime", "rem_last", "n2_pre_rem", "n3_night"]
EPOCH_SEC = 30


def segment_epochs(meta: dict, hypno: np.ndarray) -> dict[str, np.ndarray]:
    """Return epoch INDICES per arm for one night. Pure index logic, no I/O.

    hypno: per-epoch stage codes, one of {'W','N1','N2','N3','REM'}.
    """
    last_rem = meta["last_rem_epoch"]
    idx = {}
    is_ = lambda s: np.where(hypno == s)[0]

    # sleep onset = first non-W epoch
    non_w = np.where(hypno != "W")[0]
    onset = int(non_w[0]) if len(non_w) else len(hypno)

    idx["wake_bedtime"] = is_("W")[is_("W") < onset]           # before sleep onset
    idx["wake_report"] = is_("W")[is_("W") > last_rem]          # after final REM
    rem = is_("REM")
    # contiguous REM block ending at last_rem_epoch
    if len(rem):
        blk = [last_rem]
        while blk[0] - 1 in set(rem.tolist()):
            blk.insert(0, blk[0] - 1)
        idx["rem_last"] = np.array(blk)
        n2 = is_("N2")
        idx["n2_pre_rem"] = n2[n2 < blk[0]][-len(blk):] if len(n2[n2 < blk[0]]) else np.array([], int)
    else:
        idx["rem_last"] = np.array([], int); idx["n2_pre_rem"] = np.array([], int)
    idx["n3_night"] = is_("N3")
    return idx


def night_level_permutation_null(y_by_night: dict, n_perm: int, seed: int):
    """Permute labels ACROSS NIGHTS, keeping each night's segments bound together."""
    rng = random.Random(seed)
    nights = sorted(y_by_night)
    labels = [y_by_night[n] for n in nights]
    for _ in range(n_perm):
        rng.shuffle(labels)
        yield dict(zip(nights, labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/orcd/scratch/orcd/010/ninon/reaDream")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--verify-only", action="store_true",
                    help="check the format contract and print the design matrix, run nothing")
    a = ap.parse_args()

    S = Path(a.scratch)
    metas = sorted((S / "nrem_blocks").glob("night_*_meta.json"))
    print(f"nights with meta: {len(metas)}")
    labels = json.loads(Path(a.labels).read_text())
    print(f"nights with labels: {len(labels)}")

    # design matrix: how many nights actually yield each arm
    counts = defaultdict(int); epochs = defaultdict(list)
    for m in metas:
        meta = json.loads(m.read_text())
        sc = meta.get("stage_counts", {})
        for arm, need in [("wake_bedtime", "W"), ("wake_report", "W"),
                          ("rem_last", "REM"), ("n2_pre_rem", "N2"), ("n3_night", "N3")]:
            if sc.get(need, 0) > 0:
                counts[arm] += 1; epochs[arm].append(sc[need])
    print(f"\n{'arm':<16}{'nights with data':>18}{'median epochs':>16}{'median min':>12}")
    for arm in ARMS:
        e = epochs[arm]
        med = float(np.median(e)) if e else 0
        print(f"{arm:<16}{counts[arm]:>18}{med:>16.0f}{med*EPOCH_SEC/60:>12.1f}")

    print(f"\n{'label':<20}{'n nights positive':>19}{'rate':>9}")
    for lab in HVDC:
        pos = sum(1 for v in labels.values() if v.get("dreamy_labels", {}).get(lab))
        print(f"{lab:<20}{pos:>19}{pos/max(len(labels),1):>8.1%}")

    if a.verify_only:
        print("\n--verify-only: format contract checked, nothing run.")
        return


if __name__ == "__main__":
    main()
