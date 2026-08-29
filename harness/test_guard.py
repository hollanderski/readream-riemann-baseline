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
