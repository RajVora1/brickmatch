"""Let the user check and fix the model's piece counts before builds are matched.

The model makes mistakes (a 1x4 brick seen from the side can look like a 2x4), so the demo shows one row per
piece type with what the model found and an editable "You have" column. Fixing a wrong piece type means taking
one off the type the model said and adding one to the right type. Whatever is in "You have" is what the
builds, the 3D view and the booklets are matched against.

No Gradio here, so it can be tested on its own (tests/test_counts.py).
"""
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PIECES = yaml.safe_load((ROOT / "configs/pieces.yaml").read_text())["pieces"]
NAMES = {p["id"]: p["name"] for p in PIECES}
HEADERS = ["Piece", "Model found", "You have", "Colors the model saw"]
MAX_COUNT = 999


def table(found_counts, colors=None, have=None):
    """One row per piece type: types the model found first (most first), then the other tier-1 types, then
    tier 2. `have` (the user's counts) defaults to what the model found."""
    found_counts = Counter(found_counts or {})
    have = found_counts if have is None else Counter(have)
    colors = colors or {}
    found_order = [p for p, _ in found_counts.most_common()]
    rest = [p["id"] for p in sorted(PIECES, key=lambda p: p["tier"]) if p["id"] not in found_counts]
    extra = [p for p in have if p not in NAMES and p not in found_counts]  # e.g. a class not in pieces.yaml
    return [[NAMES.get(p, p), found_counts.get(p, 0), have.get(p, 0), colors.get(p, "")]
            for p in found_order + rest + extra]


def _rows(rows):
    """Accept a pandas DataFrame (what Gradio passes back) or a list of rows."""
    if hasattr(rows, "values") and hasattr(rows, "columns"):
        return rows.values.tolist()
    return list(rows or [])


def _piece_id(name):
    key = " ".join(str(name).split()).lower()
    for pid, pname in NAMES.items():
        if key in (pid.lower(), pname.lower()):
            return pid
    return None


def _count(value, name):
    if value is None or (isinstance(value, float) and value != value) or str(value).strip() == "":
        return 0  # an emptied cell means none
    try:
        n = float(str(value).strip())
    except ValueError:
        raise ValueError(f"'{value}' isn't a number (row {name}). Type how many you have, like 3.") from None
    if n < 0 or n != int(n):
        raise ValueError(f"{name}: the count has to be a whole number, 0 or more (got {value}).")
    if n > MAX_COUNT:
        raise ValueError(f"{name}: {int(n)} is more than {MAX_COUNT}. Is that a typo?")
    return int(n)


def read_table(rows):
    """The user's counts from the edited table, as {piece_id: count}, leaving out zeros.
    Raises ValueError with a message meant for the user if a count can't be read."""
    counts = Counter()
    for row in _rows(rows):
        if not row or all(str(v).strip() in ("", "nan", "None") for v in row):
            continue
        name = row[0]
        if name is None or str(name).strip() in ("", "nan"):
            continue  # a blank row someone added
        pid = _piece_id(name) or str(name).strip()
        counts[pid] += _count(row[2] if len(row) > 2 else None, name)
    return {p: n for p, n in counts.items() if n > 0}


def changes(found_counts, have):
    """What the user changed compared with the model, one line per piece type, e.g.
    'Brick 2 x 4: 6 → 4'. Empty list if nothing changed."""
    found_counts, have = Counter(found_counts or {}), Counter(have or {})
    out = []
    for p in sorted(set(found_counts) | set(have), key=lambda p: NAMES.get(p, p)):
        if found_counts[p] != have[p]:
            out.append(f"{NAMES.get(p, p)}: {found_counts[p]} → {have[p]}")
    return out
