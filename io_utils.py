"""JSON + Excel logging for every intermediate result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return str(obj)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row)) + "\n")


def mean_std(values: Sequence[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "n": int(arr.size),
    }


def rows_to_excel(path: Path, sheets: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    """Write one or more list-of-dict sheets. Falls back to CSV if openpyxl is missing."""
    ensure_dir(path.parent)
    try:
        import pandas as pd
    except ImportError:
        for name, rows in sheets.items():
            csv_path = path.with_name(f"{path.stem}_{name}.csv")
            rows = list(rows)
            if not rows:
                csv_path.write_text("", encoding="utf-8")
                continue
            keys = list(rows[0].keys())
            lines = [",".join(keys)]
            for row in rows:
                lines.append(",".join(_csv_cell(row.get(k)) for k in keys))
            csv_path.write_text("\n".join(lines), encoding="utf-8")
        return

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, rows in sheets.items():
            df = pd.DataFrame(list(rows))
            if df.empty:
                df = pd.DataFrame({"_empty": []})
            df.to_excel(writer, sheet_name=name[:31], index=False)


def _csv_cell(v: Any) -> str:
    s = "" if v is None else str(v)
    if any(c in s for c in ",\"\n"):
        return '"' + s.replace('"', '""') + '"'
    return s
