# Riemannian dream-affect decoding: baselines, nulls, and a measured noise floor

Handoff for **Paul Barbaste**. Two corpora, every arm null, and a harness where a new
method is ~20 lines and gets scored on exactly the folds every existing arm used.

## Start here

| file | what it is |
|---|---|
| `notebooks/REM_Turku_handoff.ipynb` | **Read first.** Harness, six arms, frozen results, your entry point. Colab-ready. |
| `notebooks/tuning_and_noise_floor.ipynb` | Nested selection, and the measurement showing it cannot help at this N. |
| `PREREG.md` | Six pre-registrations, each committed before its numbers existed, plus a deviation log. |
| `DREAM_SCREEN.md` | 20 DREAM-database deposits screened; 7 carry a LOSO claim alone (208 subjects, 1696 awakenings). |
| `harness/` | The scripts as run on the cluster, including `test_guard.py`. |

## The state, in one paragraph

Every arm on both corpora is null. REM_Turku: three DL architectures, tangent space with
and without per-subject recentring, within-subject leave-one-awakening-out, TSMNet, and a
12-config nested sweep. 101-Nights body_action: tangent and ShallowConv under a 65-night
repeated CV, neither reaching the 0.5385 majority baseline. **The one significant result is
anger's cross-subject ranking being significantly INVERTED** (AUC 0.3341, p=0.0100,
Bonferroni-passing) with a held-out Simpson decomposition as its mechanism.

The pipeline is not the explanation: through the identical features and CV, subject
identity decodes at 0.895 on 17 classes and recording night at 0.812.

## Four traps, each of which cost us a day

1. **Permute labels WITHIN subject.** A global shuffle redraws each subject's base rate
   toward the grand mean; it inflated our detectable-effect floor from 0.598 to 0.655 and
   manufactured a fold dependence that does not exist (1.50 apparent, 1.15 real).
2. **Seed 0 means NO shuffle.** A null loop starting at 0 puts the unshuffled observed
   inside its own null. `harness/test_guard.py` asserts every loop guards its lower bound.
3. **A permutation p at its floor is not a measurement.** Ours read 0.0175 at 56 draws and
   0.0550 at 199; another read 0.0816 and cleared its 95th percentile at 48 draws, then
   0.0792 and did not at 100.
4. **Metric level is not cosmetic.** The same 20 draws gave p<=0.048 at epoch level and
   p=0.1429 at awakening level for one observation, because epochs inside an awakening
   share its label. Every table row here is labelled.

## Running it

```bash
git clone https://github.com/hollanderski/readream-riemann-baseline
cd readream-riemann-baseline
python harness/test_guard.py        # must PASS before any null
```
Then open `notebooks/REM_Turku_handoff.ipynb`. Its data cell rebuilds from the public
figshare deposit (10.6084/m9.figshare.23274596.v2); we deliberately ship no preprocessed
archive so your environment is verified rather than trusted. body_action data is private
and staged separately.
