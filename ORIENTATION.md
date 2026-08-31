# Orientation: your first hour

1. Read `README.md` (5 min): the state, the publishable results, the four traps.
2. Open `notebooks/REM_Turku_handoff.ipynb` on Colab from the notebooks/ directory.
   Run cells 0-2 (installs), then the data cell: it downloads the public figshare
   deposit and rebuilds the epochs (~15-20 min, cached). No credentials.
3. While it builds, read the frozen results table (section 4). The number your method
   must beat: nothing clears chance; best arms are TSMNet 58.5% (p=0.14) and tangent
   57.5% on body_action (p=0.079). A method clearing its own permutation null on these
   folds is the paper's headline.
4. Run the tangent arm (section 5, few minutes on CPU) to see the harness end to end.
5. Write your factory (section 6), run `evaluate`, compare with `paired`, then the
   permutation null (`shuffle_seed = 1..20`), reading trap 3 before quoting any p.
6. `notebooks/tuning_and_noise_floor.ipynb`: read its first cell BEFORE spending GPU on
   tuning; the measured answer is that selection cannot help at this N.
7. body_action data is private (single-subject dream records): Ninon supplies a Drive
   link separately; set `DATA_DIR` and the same harness applies.
8. `PREREG.md` = every pre-registration, committed before its numbers, plus deviations.
9. `COMMIT_MAP_2026-08-31.txt` maps older hashes cited in PREREG to current ones.
10. Anything unclear: the notebook cells are the authority; the cluster scripts in
    `harness/` are what actually ran.
