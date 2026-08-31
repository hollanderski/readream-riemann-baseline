"""
prepare_remturku.py

Canonical preprocessing for the REM_Turku DREAM-database cohort
(Sikka, Revonsuo, Noreika & Valli; figshare 10.6084/m9.figshare.23274596.v2).

Reproduces the pipeline of:
  Sikka P, Revonsuo A, Noreika V, Valli K (2019). EEG frontal alpha asymmetry
  and dream affect. J Neurosci 39(24):4775-4784.

so that the published positive control (FAA = ln[F4] - ln[F3] on alpha CSD
power predicts dream anger) can be reproduced BEFORE any novel claim is made.

Per the centralized-preprocessing hard rule, this is the single entry point for
this cohort. Every artifact it writes records PIPELINE_VERSION plus the sha256
of its input EDF, and the invariants are asserted rather than assumed.

Source data invariants (verified across all 133 EDFs):
  29 signals = 24 EEG (10/10) + 4 EOG (HL,HR,VU,VD) + 1 EMG
  500 Hz, microvolts, 119-121 s per file, single montage, right-mastoid ref
  Acquisition-time 50 Hz notch + anti-alias LP only. No offline preprocessing.

Usage:
  python prepare_remturku.py --zip REM_Turku.zip --out remturku_prepared.npz
"""
from __future__ import annotations

import argparse, hashlib, io, json, warnings, zipfile
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

PIPELINE_VERSION = "remturku-v1-sikka2019"

EEG_CH = ['Fp1','Fp2','AF7','AF3','AF4','AF8','F7','F3','Fz','F4','F8','T7',
          'C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','Oz','O2']
EOG_CH = ['EOG-HL','EOG-HR','EOG-VU','EOG-VD']
EMG_CH = ['EMG']

# Sikka 2019 parameters
L_FREQ, H_FREQ = 0.5, 45.0     # FIR bandpass (pop_eegfiltnew equivalent)
EPOCH_SEC, OVERLAP = 2.0, 0.5  # 2 s epochs, 50% overlap, Hamming window
ALPHA = (8.0, 13.0)
SFREQ = 500.0
REJECT_UV = 200.0              # noisy-epoch criterion (see DEVIATIONS)

# Homologous pairs used by Sikka as spatial-specificity controls
PAIRS = [('F4','F3'),('Fp2','Fp1'),('AF8','AF7'),('AF4','AF3'),
         ('F8','F7'),('C4','C3'),('T8','T7'),('P8','P7'),('P4','P3'),('O2','O1')]

DEVIATIONS = [
 "Sikka used visual inspection to reject non-biological signals; replaced here "
 "by a fixed +/-200 uV peak-to-peak epoch criterion (automated, reproducible).",
 "Sikka used MARA (Winkler 2011) to flag artifactual ICA components, a trained "
 "classifier with no MNE equivalent; replaced here by ICA with EOG-correlation "
 "component detection driven by the 4 recorded EOG channels.",
 "Sikka interpolated previously removed bad channels; no channel is dropped "
 "here (no flat/dead channels were found in any of the 133 files), so no "
 "interpolation is applied.",
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def raw_from_edf_bytes(b: bytes, tmpdir: Path, name: str):
    """Return (raw, path). MNE re-validates the source path during ICA, so the
    caller must keep the file on disk until preprocessing for it has finished."""
    p = tmpdir / name
    p.write_bytes(b)
    raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
    return raw, p


def preprocess(raw: mne.io.BaseRaw, do_ica: bool = True) -> dict:
    """Sikka-2019 pipeline. Returns CSD epochs + alpha power + provenance."""
    raw.rename_channels({c: c.strip() for c in raw.ch_names})
    present = [c for c in EEG_CH if c in raw.ch_names]
    assert len(present) == 24, f"expected 24 EEG channels, got {len(present)}"
    assert abs(raw.info["sfreq"] - SFREQ) < 1e-6, f"sfreq={raw.info['sfreq']}"

    raw.set_channel_types({c: "eog" for c in EOG_CH if c in raw.ch_names}
                          | {c: "emg" for c in EMG_CH if c in raw.ch_names})

    # 2. FIR bandpass 0.5-45 Hz
    raw.filter(L_FREQ, H_FREQ, method="fir", fir_design="firwin", verbose=False)

    # 6. average reference (Sikka re-references before ICA)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # 7. ICA, EOG-driven component removal (see DEVIATIONS)
    n_excluded = 0
    if do_ica:
        ica = mne.preprocessing.ICA(n_components=20, method="fastica",
                                    random_state=97, max_iter="auto")
        ica.fit(raw, picks="eeg", verbose=False)
        bads, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = bads
        n_excluded = len(bads)
        ica.apply(raw, verbose=False)

    raw.pick(present)  # EEG only, ordered
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    on_missing="raise", verbose=False)

    # 4. 2 s epochs, 50% overlap
    ep = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_SEC, overlap=EPOCH_SEC * OVERLAP,
        preload=True, verbose=False)

    # 5. drop noisy epochs
    data_uv = ep.get_data() * 1e6
    keep = (np.abs(data_uv).max(axis=(1, 2)) < REJECT_UV)
    ep = ep[keep]
    n_kept, n_tot = int(keep.sum()), int(keep.size)
    if n_kept < 5:
        return {"ok": False, "reason": f"only {n_kept}/{n_tot} clean epochs"}

    # 9. current source density, spherical spline (Perrin 1989) -> uV/cm^2
    ep_csd = mne.preprocessing.compute_current_source_density(ep, verbose=False)

    # 10. FFT -> mean alpha power per electrode, Hamming window
    X = ep_csd.get_data()                       # (n_ep, 24, n_t)
    n_t = X.shape[-1]
    win = np.hamming(n_t)
    F = np.fft.rfft(X * win, axis=-1)
    psd = (np.abs(F) ** 2)
    freqs = np.fft.rfftfreq(n_t, 1.0 / SFREQ)
    band = (freqs >= ALPHA[0]) & (freqs <= ALPHA[1])
    alpha_per_epoch = psd[:, :, band].mean(-1)  # (n_ep, 24)
    alpha = alpha_per_epoch.mean(0)             # average over epochs

    return {"ok": True, "ch_names": present, "alpha": alpha,
            "csd_epochs": X.astype(np.float32),
            "raw_epochs": ep.get_data().astype(np.float32),
            "n_epochs_kept": n_kept, "n_epochs_total": n_tot,
            "n_ica_excluded": n_excluded}


def faa(alpha: np.ndarray, ch: list[str]) -> dict:
    """ln[right] - ln[left] for each homologous pair (Sikka eq.)."""
    idx = {c: i for i, c in enumerate(ch)}
    out = {}
    for r, l in PAIRS:
        if r in idx and l in idx:
            out[f"{r}-{l}"] = float(np.log(alpha[idx[r]]) - np.log(alpha[idx[l]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ica", action="store_true")
    ap.add_argument("--save-epochs", action="store_true",
                    help="also store CSD epoch tensors (large)")
    a = ap.parse_args()

    tmp = Path(a.out).parent / "_tmp_edf"; tmp.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(a.zip)
    edfs = sorted(n for n in z.namelist() if n.lower().endswith(".edf"))
    if a.limit: edfs = edfs[:a.limit]

    recs, epochs_store, failures = [], {}, []
    for i, name in enumerate(edfs, 1):
        b = z.read(name)
        p_edf = None
        try:
            raw, p_edf = raw_from_edf_bytes(b, tmp, Path(name).name)
            r = preprocess(raw, do_ica=not a.no_ica)
        except Exception as e:
            failures.append({"file": name, "error": repr(e)[:200]})
            print(f"[{i}/{len(edfs)}] {Path(name).name}  FAILED {e}", flush=True)
            continue
        finally:
            if p_edf is not None and p_edf.exists():
                p_edf.unlink()
        if not r["ok"]:
            failures.append({"file": name, "error": r["reason"]})
            print(f"[{i}/{len(edfs)}] {Path(name).name}  SKIP {r['reason']}", flush=True)
            continue
        rec = {"filename": Path(name).name,
               "input_sha256": sha256_bytes(b),
               "pipeline_version": PIPELINE_VERSION,
               "n_epochs_kept": r["n_epochs_kept"],
               "n_epochs_total": r["n_epochs_total"],
               "n_ica_excluded": r["n_ica_excluded"],
               "alpha": r["alpha"].tolist(),
               **{f"faa_{k}": v for k, v in faa(r["alpha"], r["ch_names"]).items()}}
        recs.append(rec)
        if a.save_epochs:
            # Save BOTH. CSD is a spatial Laplacian: reference-free, which is why Sikka
            # used it for FAA, but it is also a spatial high-pass that removes the broad
            # topographies EEGNet's spatial filters and a Riemannian covariance rely on.
            # Saving only CSD would foreclose that choice, so keep the pre-CSD
            # (0.5-45 Hz, average-referenced, ICA-cleaned) epochs too.
            epochs_store[Path(name).name + "|csd"] = r["csd_epochs"]
            epochs_store[Path(name).name + "|raw"] = r["raw_epochs"]
        print(f"[{i}/{len(edfs)}] {Path(name).name}  epochs {r['n_epochs_kept']}/{r['n_epochs_total']}"
              f"  ICA-excl {r['n_ica_excluded']}  FAA(F4-F3)={rec['faa_F4-F3']:+.4f}", flush=True)

    meta = {"pipeline_version": PIPELINE_VERSION, "source_zip": str(a.zip),
            "source_zip_sha256": sha256_bytes(Path(a.zip).read_bytes()),
            "l_freq": L_FREQ, "h_freq": H_FREQ, "epoch_sec": EPOCH_SEC,
            "overlap": OVERLAP, "alpha_band": ALPHA, "reject_uv": REJECT_UV,
            "reference": "average", "csd": "spherical spline (Perrin 1989)",
            "eeg_channels": EEG_CH, "deviations_from_sikka2019": DEVIATIONS,
            "n_ok": len(recs), "n_failed": len(failures), "failures": failures}
    Path(a.out).with_suffix(".json").write_text(json.dumps({"meta": meta, "records": recs}, indent=1))
    if a.save_epochs and epochs_store:
        np.savez_compressed(a.out, **epochs_store)
    try: tmp.rmdir()
    except OSError: pass
    print(f"\nOK {len(recs)}  FAILED {len(failures)}  -> {Path(a.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
