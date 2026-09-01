"""Notebook read/write that cannot produce jammed cells.

The disease: nbformat stores `source` as a list of lines that must each keep its
trailing "\n" (all but the last). `s.split("\n")` strips them, so every line gets
re-joined edge to edge and the whole cell collapses to one line. It stays valid
JSON, so a JSON round-trip check passes and the notebook still opens; the failure
only appears when a human runs the cell. It reached Ninon's screen twice.

Use `lines()` to build source, and `dump()` to write. `verify()` is not optional:
`dump()` calls it and refuses to leave a jammed file on disk.
"""
import ast, json, os, pathlib


def lines(text):
    """Split into newline-TERMINATED lines, the way nbformat wants them."""
    out = text.split("\n")
    return [x + "\n" for x in out[:-1]] + ([out[-1]] if out[-1] else [])


def verify(nb, path="<mem>"):
    """Return a list of problems. Empty list means the notebook is safe to ship."""
    bad = []
    for i, c in enumerate(nb["cells"]):
        src = c["source"]
        if isinstance(src, str):
            bad.append(f"cell {i}: source is a str, not a list of lines")
            continue
        for line in src[:-1]:
            if not line.endswith("\n"):
                bad.append(f"cell {i} ({c['cell_type']}): line without terminator -> jam")
                break
        joined = "".join(src)
        if len(src) == 1 and len(joined) > 120:
            bad.append(f"cell {i} ({c['cell_type']}): single line of {len(joined)} chars -> jam")
        if c["cell_type"] == "code":
            code = "".join(l for l in src if not l.lstrip().startswith(("!", "%")))
            try:
                ast.parse(code)
            except SyntaxError as e:
                bad.append(f"cell {i}: SyntaxError line {e.lineno}: {e.msg}")
    return bad


def dump(nb, path):
    """Write only if the notebook verifies. Atomic, and re-read from disk."""
    problems = verify(nb, path)
    if problems:
        raise SystemExit("refusing to write a jammed notebook:\n  " + "\n  ".join(problems))
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    reread = json.load(open(path))
    problems = verify(reread, path)
    if problems:
        raise SystemExit("wrote a jammed notebook, on-disk check failed:\n  " + "\n  ".join(problems))
    return len(reread["cells"])


def load(path):
    return json.load(pathlib.Path(path).open())
