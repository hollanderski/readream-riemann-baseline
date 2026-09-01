"""Repo guards. Run from this directory: `python3 test_guard.py`.

Every check is a `test_*` function and the dispatcher at the bottom runs all of
them. Do NOT append a bare `def test_...` and assume it runs: an earlier version
of this file was a flat script, a guard was appended as an uncalled function, and
it sat there passing vacuously. The dispatcher now fails if it discovers nothing.
"""
import re, pathlib, sys


def test_null_loops_guard_seed_zero():
    """seed 0 means NO shuffle (`if shuffle:` is falsy), so a null loop must never start at 0."""
    bad = []
    here = pathlib.Path(__file__).resolve().parent
    checked = []
    for f in ("aucnull.py", "ladder.py"):
        p = here / f
        if not p.exists():
            raise AssertionError(f"{f} missing: this guard would pass vacuously")
        checked.append(f)
        src = p.read_text()
        for m in re.finditer(r"for\s+\w+\s+in\s+range\(([^)]*)\)", src):
            args = m.group(1)
            # a null loop is one whose body writes a *null*.json; check the low bound is guarded
            if "LO" in args or "lo" in args:
                if "max(" not in args:
                    bad.append("%s: unguarded null loop `range(%s)` can start at seed 0" % (f, args))
    assert not bad, "\n  " + "\n  ".join(bad)
    print(f"  null loops guard seed 0: {len(checked)} files checked")


def test_notebook_cells_are_not_jammed():
    """Every notebook in the repo must survive the jam detector.

    A jammed cell is valid JSON and opens fine; it only fails when a human runs
    it. This shipped to Ninon twice, from `.split("\\n")` dropping the line
    terminators nbformat requires. Build source with nbsafe.lines(), write with
    nbsafe.dump(), and this test is the backstop for both.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import nbsafe
    nbs = sorted(pathlib.Path(__file__).resolve().parent.parent.glob("notebooks/*.ipynb"))
    assert nbs, "no notebooks found: the guard would pass vacuously"
    for nb in nbs:
        problems = nbsafe.verify(nbsafe.load(nb), nb)
        assert not problems, f"{nb.name} is jammed:\n  " + "\n  ".join(problems)
    print(f"  notebooks not jammed: {len(nbs)} checked")


def test_jam_detector_actually_detects_jams():
    """Negative control for the detector itself.

    test_notebook_cells_are_not_jammed only proves the repo is clean. If
    nbsafe.verify ever regressed to returning [] unconditionally, that test would
    keep passing and we would be back to shipping jams with a green run. So the
    detector has to be shown FAILING on each defect it claims to catch.

    Same principle as the dead guard: a check that has never been observed failing
    is not evidence. Every case below is a form that actually reached disk or is
    one edit away from it.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import nbsafe

    def cell(src, kind="code"):
        c = {"cell_type": kind, "source": src, "metadata": {}}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        return {"cells": [c], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

    body = "import os\nx = 1\nprint(x)"
    cases = [
        # the exact defect that shipped: split() drops the terminators
        ("split without keepends", cell(body.split("\n"))),
        # what it looks like once nbformat has rejoined it
        ("single-line jam", cell(["".join(body.split("\n"))])),
        # a whole-string source: legal-ish JSON, breaks line-wise editing
        ("source is a str", cell(body)),
        # the detector must not go blind on markdown
        ("markdown jam", cell(["a long markdown line " * 8], kind="markdown")),
        # ast.parse arm, independent of the newline arm
        ("syntax error", cell(nbsafe.lines("def f(:\n    pass"))),
    ]
    for label, nb in cases:
        problems = nbsafe.verify(nb)
        assert problems, f"detector MISSED {label!r}: it is not detecting what it claims"

    # and it must stay quiet on correctly built source, or it is just noise
    assert not nbsafe.verify(cell(nbsafe.lines(body))), "detector fires on valid source"

    # dump() must refuse rather than write a jam to disk
    import tempfile, os as _os
    fd, tmp = tempfile.mkstemp(suffix=".ipynb"); _os.close(fd)
    try:
        try:
            nbsafe.dump(cell(body.split("\n")), tmp)
        except SystemExit:
            pass
        else:
            raise AssertionError("dump() wrote a jammed notebook instead of refusing")
    finally:
        _os.path.exists(tmp) and _os.unlink(tmp)
    print(f"  jam detector fails on all {len(cases)} defect forms, quiet on valid source")


if __name__ == "__main__":
    tests = sorted(k for k, v in list(globals().items())
                   if k.startswith("test_") and callable(v))
    if not tests:
        print("FAIL: dispatcher found no test_* functions"); sys.exit(1)
    failed = []
    for name in tests:
        try:
            globals()[name]()
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  FAIL {name}: {e}")
    if failed:
        print(f"FAIL: {len(failed)}/{len(tests)} guards failed"); sys.exit(1)
    print(f"PASS: {len(tests)} guards")
