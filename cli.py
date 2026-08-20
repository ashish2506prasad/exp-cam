"""Shared CLI helpers used by every section script."""

from __future__ import annotations

import argparse

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

import torch

from config import DEFAULT_BACKBONES, SEED, STEPS, add_where_args
from data import load_eval_set

_CUDA_OK = None
_CUDA_NOTE = ""


def cuda_is_runnable() -> bool:
    """True only if a CUDA kernel can actually run, not merely if a GPU is present.

    Kaggle P100 is sm_60; current PyTorch wheels are sm_70+ and will crash at conv2d
    with cudaErrorNoKernelImageForDevice even though torch.cuda.is_available() is True.
    """
    global _CUDA_OK, _CUDA_NOTE
    if _CUDA_OK is not None:
        return _CUDA_OK
    if not torch.cuda.is_available():
        _CUDA_OK, _CUDA_NOTE = False, "CUDA not available"
        return False
    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        getter = getattr(torch.cuda, "get_arch_list", None)
        archs = [a for a in getter() if str(a).startswith("sm_")] if callable(getter) else []
        if archs and sm not in archs:
            _CUDA_OK = False
            _CUDA_NOTE = (
                f"{name} is {sm}; this PyTorch build only has {', '.join(archs)}. "
                "On Kaggle switch the accelerator from P100 to T4 (or any Volta+ GPU), "
                "or pass --device cpu."
            )
            return False
        x = torch.zeros(1, 3, 8, 8, device="cuda")
        w = torch.zeros(1, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        _CUDA_OK, _CUDA_NOTE = True, f"{name} ({sm})"
        return True
    except Exception as e:
        _CUDA_OK = False
        _CUDA_NOTE = f"CUDA probe failed ({type(e).__name__}: {e}). Pass --device cpu or use a T4 GPU."
        return False


def device_from_flag(s: str) -> torch.device:
    s = (s or "auto").strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    want_cuda = s in ("auto", "cuda") or s.startswith("cuda")
    if want_cuda and cuda_is_runnable():
        return torch.device(s if s.startswith("cuda:") else "cuda")
    if want_cuda and s != "cpu":
        print(f"[device] {_CUDA_NOTE}", flush=True)
        print("[device] using CPU", flush=True)
    return torch.device("cpu")


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_where_args(p)
    p.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA if the GPU can actually run kernels, else CPU), cuda, or cpu. "
        "Kaggle P100 is often unusable with current PyTorch; pick a T4 or pass cpu.",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--n-images", type=int, default=None, help="Subset of the cached eval set (still stratified order).")
    p.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
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
    import config as C

    C.apply_where(
        getattr(args, "where", "auto"),
        data_root=getattr(args, "data_root", None),
        runs_dir=getattr(args, "out", None),
    )
    if getattr(args, "out", None) is None:
        args.out = C.RUNS_DIR
    if args.quick:
        args.steps = min(args.steps, 40)
        args.backbones = ["resnet18"]
        args.n_images = args.n_images or 7
    eval_set = load_eval_set()
    images = _stratified_subset(eval_set["images"], args.n_images)
    device = device_from_flag(args.device)
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
