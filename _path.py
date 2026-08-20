"""Put this folder on sys.path so `python section1_ablation.py` works on Kaggle.

The GitHub repo is often cloned as `exp-cam`, so `python -m expcam` and
`from expcam.cli import ...` both fail there. Inserting this directory lets
every module import siblings as `from cli import ...`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_pkg_path() -> Path:
    pkg = Path(__file__).resolve().parent
    s = str(pkg)
    if s not in sys.path:
        sys.path.insert(0, s)
    return pkg
