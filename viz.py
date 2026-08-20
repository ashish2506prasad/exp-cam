"""Figures specified in the revision plan. Grey-background convention throughout."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import GREY_BG
from .io_utils import ensure_dir
from .metrics import denormalize, explanation_on_grey


def caption(p_pct: float, conf: float, top1: float) -> str:
    mark = "✓" if top1 >= 0.5 else "✗"
    return f"p={p_pct:.1f}%  conf={conf:.2f}  top1={mark}"


def _save(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grouped_bars_dual(
    path: Path,
    labels: Sequence[str],
    left_mean: Sequence[float],
    left_std: Sequence[float],
    right_mean: Sequence[float],
    right_std: Sequence[float],
    left_ylabel: str,
    right_ylabel: str,
    title: str,
    left_color: str = "#4C78A8",
    right_color: str = "#F58518",
) -> None:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(labels)), 4.5))
    ax.bar(x, left_mean, yerr=left_std, color=left_color, alpha=0.85, capsize=3, label=left_ylabel)
    ax.set_ylabel(left_ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax2 = ax.twinx()
    ax2.errorbar(x, right_mean, yerr=right_std, color=right_color, marker="o", lw=2, capsize=3, label=right_ylabel)
    ax2.set_ylabel(right_ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def small_multiples_bars(
    path: Path,
    labels: Sequence[str],
    series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    title: str,
) -> None:
    keys = list(series.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = np.arange(len(labels))
    for ax, key in zip(axes.ravel(), keys):
        mean, std = series[key]
        ax.bar(x, mean, yerr=std, capsize=3, color="#4C78A8")
        ax.set_title(key)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)


def sweep_dual_axis(
    path: Path,
    xs: Sequence[float],
    left_mean: Sequence[float],
    left_std: Sequence[float],
    right_mean: Sequence[float],
    right_std: Sequence[float],
    left_ylabel: str,
    right_ylabel: str,
    title: str,
    xlabel: str = "coefficient (× default)",
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = np.asarray(xs, dtype=float)
    ax.plot(xs, left_mean, color="#4C78A8", marker="o", label=left_ylabel)
    ax.fill_between(xs, np.asarray(left_mean) - left_std, np.asarray(left_mean) + left_std, color="#4C78A8", alpha=0.2)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(left_ylabel, color="#4C78A8")
    ax2 = ax.twinx()
    ax2.plot(xs, right_mean, color="#F58518", marker="s", label=right_ylabel)
    ax2.fill_between(xs, np.asarray(right_mean) - right_std, np.asarray(right_mean) + right_std, color="#F58518", alpha=0.2)
    ax2.set_ylabel(right_ylabel, color="#F58518")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def training_curves(path: Path, curve: List[Dict[str, float]]) -> None:
    steps = [r["step"] for r in curve]
    am = ["act", "ce", "kl"]
    rest = ["area", "bin", "tv", "rob"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, keys, title in (
        (axes[0], am, "L_AM components"),
        (axes[1], rest, "L_MIN / L_ROB components"),
    ):
        for k in keys:
            ax.plot(steps, [r[k] for r in curve], label=k)
        ax.plot(steps, [r["total"] for r in curve], color="k", lw=2, label="total")
        ax.set_yscale("log")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend(ncol=4, fontsize=8)
    axes[1].set_xlabel("iteration")
    fig.tight_layout()
    _save(fig, path)


def fidelity_vs_budget(
    path: Path,
    p: Sequence[float],
    series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    p_actual_mean: Sequence[float],
    p_actual_std: Sequence[float],
    p_star: Optional[float],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    p = np.asarray(p) * 100
    colors = {"top1_agreement": "#4C78A8", "one_minus_normalized_kl": "#54A24B"}
    for name, (mean, std) in series.items():
        ax.plot(p, mean, marker="o", label=name, color=colors.get(name))
        ax.fill_between(p, np.asarray(mean) - std, np.asarray(mean) + std, alpha=0.18, color=colors.get(name))
    ax.set_xscale("log")
    ax.set_xlabel("mask budget p (% pixels)")
    ax.set_ylabel("Fidelity(p) components")
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(p, np.asarray(p_actual_mean) * 100, "k--", label="p_actual")
    ax2.plot(p, p, color="0.5", ls=":", label="p_actual = p_budget")
    ax2.set_ylabel("p_actual (%)")
    if p_star is not None:
        ax.axvline(p_star * 100, color="crimson", ls="--", label=f"p* (τ=0.95) = {p_star*100:.2f}%")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def overlay_two_families(
    path: Path,
    p: Sequence[float],
    family_a: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    family_b: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    label_a: str,
    label_b: str,
    title: str,
    p_b: Optional[Sequence[float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xa = np.asarray(p) * 100
    xb = np.asarray(p_b if p_b is not None else p) * 100
    styles = {label_a: "-", label_b: "--"}
    for fam, lab, xs in ((family_a, label_a, xa), (family_b, label_b, xb)):
        for name, (mean, std) in fam.items():
            ax.plot(xs, mean, ls=styles[lab], marker="o", label=f"{name} ({lab})")
            ax.fill_between(xs, np.asarray(mean) - std, np.asarray(mean) + std, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("mask size / budget p (% pixels)")
    ax.set_ylabel("Fidelity components")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, path)


def boxplot(path: Path, groups: Dict[str, Sequence[float]], ylabel: str, title: str, hline: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(groups)), 4.5))
    labels = list(groups.keys())
    data = [list(groups[k]) for k in labels]
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    if hline is not None:
        ax.axhline(hline, color="crimson", ls="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, path)


def scatter_parity(path: Path, xs: Sequence[float], ys: Sequence[float], xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys, c="#4C78A8", alpha=0.75)
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def line_auc(path: Path, curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]], xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, (xs, ys, auc) in curves.items():
        ax.plot(xs, ys, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def grouped_bars(
    path: Path,
    labels: Sequence[str],
    groups: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(labels))
    n = len(groups)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.5))
    for i, (name, (mean, std)) in enumerate(groups.items()):
        ax.bar(x + (i - (n - 1) / 2) * width, mean, width, yerr=std, capsize=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def image_grid(
    path: Path,
    rows: List[List[Tuple[np.ndarray, str]]],
    suptitle: str = "",
) -> None:
    """rows[i][j] = (H,W,3 or H,W float image, caption). Grey bg already baked in."""
    n_r, n_c = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.4 * n_c, 2.8 * n_r))
    if n_r == 1:
        axes = np.array([axes])
    if n_c == 1:
        axes = axes.reshape(n_r, 1)
    for i in range(n_r):
        for j in range(n_c):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(rows[i]):
                continue
            img, cap = rows[i][j]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))
            ax.set_title(cap, fontsize=7)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def to_numpy_image(t) -> np.ndarray:
    import torch

    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu()
        if arr.dim() == 4:
            arr = arr[0]
        if arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        return np.clip(arr, 0, 1)
    return np.asarray(t)


def panel_original(x) -> np.ndarray:
    return to_numpy_image(denormalize(x))


def panel_mask(mask) -> np.ndarray:
    return to_numpy_image(mask)


def panel_expl(x, mask, grey: float = GREY_BG) -> np.ndarray:
    return to_numpy_image(explanation_on_grey(x, mask, grey=grey))
