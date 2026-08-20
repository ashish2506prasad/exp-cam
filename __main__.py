"""python -m expcam <section|eval-set|all> [args...]"""

from __future__ import annotations

import sys

USAGE = """EXP-CAM revision experiments

Usage (from repo root, after installing requirements-expcam.txt):

  python -m expcam eval-set
  python -m expcam 1 [--quick]
  python -m expcam 2
  python -m expcam 3
  python -m expcam 4
  python -m expcam 5
  python -m expcam 6 [--model distilgpt2]
  python -m expcam all --quick

Every section reuses expcam_runs/eval_set.json (sampled from trainx/).
JSON + Excel intermediates are written under expcam_runs/sectionN/.
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd in ("eval-set", "eval_set", "0"):
        from .data import main as data_main

        data_main(rest)
        return
    mapping = {
        "1": ("expcam.section1_ablation", "main"),
        "2": ("expcam.section2_minimality", "main"),
        "3": ("expcam.section3_baselines", "main"),
        "4": ("expcam.section4_robustness", "main"),
        "5": ("expcam.section5_auto_weights", "main"),
        "6": ("expcam.section6_llm", "main"),
        "section1": ("expcam.section1_ablation", "main"),
        "section2": ("expcam.section2_minimality", "main"),
        "section3": ("expcam.section3_baselines", "main"),
        "section4": ("expcam.section4_robustness", "main"),
        "section5": ("expcam.section5_auto_weights", "main"),
        "section6": ("expcam.section6_llm", "main"),
    }
    if cmd == "all":
        from .data import main as data_main

        data_main([])
        for key in ("1", "2", "3", "4", "5", "6"):
            mod, fn = mapping[key]
            print(f"\n===== Section {key} =====")
            __import__(mod, fromlist=[fn])
            getattr(sys.modules[mod], fn)(rest)
        return
    if cmd not in mapping:
        print(USAGE)
        raise SystemExit(f"Unknown command: {cmd}")
    mod, fn = mapping[cmd]
    __import__(mod, fromlist=[fn])
    getattr(sys.modules[mod], fn)(rest)


if __name__ == "__main__":
    main()
