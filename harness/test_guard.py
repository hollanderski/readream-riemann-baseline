"""Guard against the seed-0 contamination returning a third time.
seed 0 means NO shuffle (`if shuffle:` is falsy), so a null loop must never start at 0."""
import re, pathlib, sys
bad = []
for f in ("aucnull.py", "ladder.py"):
    src = pathlib.Path(f).read_text()
    for m in re.finditer(r"for\s+\w+\s+in\s+range\(([^)]*)\)", src):
        args = m.group(1)
        # a null loop is one whose body writes a *null*.json; check the low bound is guarded
        if "LO" in args or "lo" in args:
            if "max(" not in args:
                bad.append("%s: unguarded null loop `range(%s)` can start at seed 0" % (f, args))
if bad:
    print("FAIL"); [print("  ", b) for b in bad]; sys.exit(1)
print("PASS: every null loop guards its lower bound against seed 0")


def test_notebook_cells_are_not_jammed():
    """Every notebook in the repo must survive the jam detector.

    A jammed cell is valid JSON and opens fine; it only fails when a human runs
    it. This shipped to Ninon twice, from `.split("\\n")` dropping the line
    terminators nbformat requires. Build source with nbsafe.lines(), write with
    nbsafe.dump(), and this test is the backstop for both.
    """
    import nbsafe, pathlib as _pl
    nbs = sorted(_pl.Path(__file__).resolve().parent.parent.glob("notebooks/*.ipynb"))
    assert nbs, "no notebooks found: the guard would pass vacuously"
    for nb in nbs:
        problems = nbsafe.verify(nbsafe.load(nb), nb)
        assert not problems, f"{nb.name} is jammed:\n  " + "\n  ".join(problems)
    print(f"  notebooks not jammed: {len(nbs)} checked, all clean")
