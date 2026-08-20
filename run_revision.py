"""Convenience wrapper: python -m expcam.run_revision 1 --quick"""

from __future__ import annotations

from pathlib import Path
import importlib.util

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

try:
    from .__main__ import main
except ImportError:
    _p = Path(__file__).resolve().parent / "__main__.py"
    _spec = importlib.util.spec_from_file_location("expcam_cli_main", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    main = _mod.main

if __name__ == "__main__":
    main()
