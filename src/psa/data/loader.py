"""Load power flow cases and time series profiles from data/.

Case files are JSON with the format documented in data/README.md: base_mva,
a source note, and bus, gen and branch tables as lists of dicts, using the
same keys the solvers expect. Profile files are CSV with a header row and a
comment block on top describing how the series was made.

`load_case` accepts either a path to a JSON file or a bare case name such as
'ieee30', which is resolved against the repository data/cases directory. The
name form works from a source checkout, which is how this project is meant
to be used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

REQUIRED_BUS_KEYS = {"bus", "type", "Pd", "Qd", "Gs", "Bs", "Vm", "Va"}
REQUIRED_GEN_KEYS = {"bus", "Pg", "Qmax", "Qmin", "Pmax", "Pmin"}
REQUIRED_BRANCH_KEYS = {"from", "to", "r", "x", "b"}


def load_case(name_or_path: str | Path) -> dict:
    """Return a case dict ready for psa.loadflow.solve."""
    p = Path(name_or_path)
    if p.suffix != ".json":
        p = _DATA_DIR / "cases" / f"{p.name}.json"
    if not p.exists():
        raise FileNotFoundError(f"case file not found: {p}")
    case = json.loads(p.read_text())
    _validate_case(case, p)
    return case


def _validate_case(case: dict, path: Path) -> None:
    for key in ("base_mva", "buses", "gens", "branches"):
        if key not in case:
            raise ValueError(f"{path}: missing '{key}'")
    n = len(case["buses"])
    numbers = {b["bus"] for b in case["buses"]}
    if numbers != set(range(1, n + 1)):
        raise ValueError(f"{path}: bus numbers must be consecutive 1..{n}")
    for b in case["buses"]:
        missing = REQUIRED_BUS_KEYS - set(b)
        if missing:
            raise ValueError(f"{path}: bus {b.get('bus')} missing {sorted(missing)}")
    slack = [b for b in case["buses"] if b["type"] == 3]
    if len(slack) != 1:
        raise ValueError(f"{path}: expected exactly one slack bus, found {len(slack)}")
    for g in case["gens"]:
        missing = REQUIRED_GEN_KEYS - set(g)
        if missing:
            raise ValueError(f"{path}: gen at bus {g.get('bus')} missing {sorted(missing)}")
        if not 1 <= g["bus"] <= n:
            raise ValueError(f"{path}: gen bus {g['bus']} out of range 1..{n}")
    for br in case["branches"]:
        missing = REQUIRED_BRANCH_KEYS - set(br)
        if missing:
            raise ValueError(f"{path}: branch missing {sorted(missing)}")
        if not (1 <= br["from"] <= n and 1 <= br["to"] <= n):
            raise ValueError(f"{path}: branch {br['from']}-{br['to']} out of range 1..{n}")


def load_profile(name_or_path: str | Path, column: str) -> list[float]:
    """Return one numeric column from a profile CSV in data/profiles.

    Lines starting with '#' are treated as comments. The first non-comment
    line is the header.
    """
    p = Path(name_or_path)
    if p.suffix != ".csv":
        p = _DATA_DIR / "profiles" / f"{p.name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"profile file not found: {p}")
    with open(p, newline="") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    header = rows[0]
    if column not in header:
        raise ValueError(f"{p}: no column '{column}', has {header}")
    idx = header.index(column)
    return [float(r[idx]) for r in rows[1:]]
