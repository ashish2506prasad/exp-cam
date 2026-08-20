"""Shared CLI helpers used by every section script."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch

from .config import DEFAULT_BACKBONES, N_IMAGES, RUNS_DIR, SEED, STEPS
from .data import load_eval_set


def device_from_flag(s: str) -> torch.device:
    if s == "cpu":
        return torch.device("cpu")
    if s.startswith("cuda"):
        return torch.device(s if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--n-images", type=int, default=None, help="Subset of the cached eval set (still stratified order).")
    p.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    p.add_argument("--out", type=Path, default=RUNS_DIR)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--quick", action="store_true", help="Tiny debug run: 1 image/class, 40 steps, resnet18 only.")
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print loss every N U-Net steps (0 = silent). Notebooks used 10.",
    )
    return p


def resolve_run(args: argparse.Namespace):
    if args.quick:
        args.steps = min(args.steps, 40)
        args.backbones = ["resnet18"]
        args.n_images = args.n_images or 7
    eval_set = load_eval_set()
    images = _stratified_subset(eval_set["images"], args.n_images)
    device = device_from_flag("cpu" if args.device == "auto" else args.device)
    if args.device == "auto":
        device = device_from_flag("cuda")
    return eval_set, images, device


def _stratified_subset(images, n):
    """Keep class balance when taking a prefix of the cached set."""
    if n is None or n >= len(images):
        return list(images)
    by_cls = {}
    for rec in images:
        by_cls.setdefault(rec["wnid"], []).append(rec)
    classes = list(by_cls.keys())
    out = []
    i = 0
    while len(out) < n:
        wnid = classes[i % len(classes)]
        slot = i // len(classes)
        if slot < len(by_cls[wnid]):
            out.append(by_cls[wnid][slot])
        i += 1
        if i > n * len(classes):
            break
    return out
