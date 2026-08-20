"""python -m expcam <section|eval-set|all> [args...]

Also works as a script on Kaggle:
    python __main__.py 1 --where kaggle
    python section1_ablation.py --where kaggle
"""

from __future__ import annotations

import sys

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

USAGE = """EXP-CAM revision experiments

Local (from the folder that contains trainx/ and expcam/):
  python -m expcam eval-set --where local
  python -m expcam 1 --where local [--quick]

Kaggle (from the cloned exp-cam folder):
  python __main__.py eval-set --where kaggle
  python section1_ablation.py --where kaggle
  python __main__.py 1 --where kaggle [--quick]

  --where kaggle uses
    /kaggle/input/datasets/meashish2003/trainx/trainx
    /kaggle/working/expcam_runs
  Override with --data-root / --out if those paths differ.

--where auto (default): kaggle if /kaggle exists, else local.

Every section reuses eval_set.json under the runs directory.
JSON + Excel intermediates are written under <runs>/sectionN/.
"""


_SECTION_MODS = {
    "1": "section1_ablation",
    "2": "section2_minimality",
    "3": "section3_baselines",
    "4": "section4_robustness",
    "5": "section5_auto_weights",
    "6": "section6_llm",
    "section1": "section1_ablation",
    "section2": "section2_minimality",
    "section3": "section3_baselines",
    "section4": "section4_robustness",
    "section5": "section5_auto_weights",
    "section6": "section6_llm",
}


def _load(modname: str):
    return __import__(modname, fromlist=["main"])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd in ("eval-set", "eval_set", "0"):
        from data import main as data_main

        data_main(rest)
        return
    if cmd == "all":
        from data import main as data_main

        data_main(rest)
        for key in ("1", "2", "3", "4", "5", "6"):
            modname = _SECTION_MODS[key]
            print(f"\n===== Section {key} =====")
            getattr(_load(modname), "main")(rest)
        return
    if cmd not in _SECTION_MODS:
        print(USAGE)
        raise SystemExit(f"Unknown command: {cmd}")
    getattr(_load(_SECTION_MODS[cmd]), "main")(rest)


if __name__ == "__main__":
    main()
