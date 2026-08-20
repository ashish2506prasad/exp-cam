"""Train/eval one (image, backbone, config) with JSON + mask caching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from .classifiers import ClassifierBundle
from .data import load_tensor
from .io_utils import load_json, save_json
from .metrics import evaluate_explanation
from .train import TrainConfig, TrainResult, image_seed, log, save_mask_tensors, train_mask


def run_dir(root: Path, section: str, backbone: str, config_name: str, image_id: str) -> Path:
    return root / section / backbone / _safe(config_name) / image_id


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def maybe_train_and_eval(
    bundle: ClassifierBundle,
    rec: dict,
    eval_root: Path,
    out_dir: Path,
    cfg: TrainConfig,
    skip_existing: bool,
    device: torch.device,
    extra_metrics: Optional[Dict] = None,
) -> Dict:
    metrics_path = out_dir / "metrics.json"
    tag = ""
    if extra_metrics:
        tag = extra_metrics.get("config") or extra_metrics.get("protocol") or ""
    who = f"{bundle.name}  img={rec['id']} {rec['class_name']}  {tag}".strip()
    if skip_existing and metrics_path.exists():
        log(f"[skip] {who}")
        return load_json(metrics_path)

    log(f"[train] {who}  steps={cfg.steps}")
    x = load_tensor(rec, bundle.preprocess, device, root=eval_root)
    result: TrainResult = train_mask(bundle, x, cfg)
    metrics = evaluate_explanation(
        bundle,
        x,
        result.mask_hard,
        mask_prob=result.mask_prob,
    )
    payload = {
        "image_id": rec["id"],
        "relpath": rec["relpath"],
        "wnid": rec["wnid"],
        "class_name": rec["class_name"],
        "backbone": bundle.name,
        "seed": result.seed,
        "steps": result.steps,
        "lr": cfg.lr,
        "optimizer": "Adam",
        "seconds": result.seconds,
        "lambdas": result.lambdas,
        "constraint": cfg.constraint,
        "p_target": cfg.p_target,
        "mu_final": result.mu_final,
        "auto_balance": cfg.auto_balance,
        "p_actual": result.p_actual,
        **metrics,
        **(extra_metrics or {}),
    }
    save_json(metrics_path, payload)
    log(
        f"[done]  img={rec['id']}  {result.seconds:.1f}s  "
        f"p={metrics['p_actual_pct']:.1f}%  top1={int(metrics['top1_agreement'])}  "
        f"conf={metrics['conf_e']:.3f}  kl={metrics['kl_divergence']:.3f}"
    )
    if cfg.record_curve and result.loss_curve:
        save_json(out_dir / "loss_curve.json", result.loss_curve)
    if result.weight_curve:
        save_json(out_dir / "weight_curve.json", result.weight_curve)
    save_mask_tensors(out_dir, result)
    save_json(
        out_dir / "train_meta.json",
        {
            "seconds": result.seconds,
            "steps": result.steps,
            "seed": result.seed,
            "optimizer": "Adam",
            "lr": cfg.lr,
            "per_image": True,
            "architecture": "UNetCSAE",
        },
    )
    return payload


def load_masks(out_dir: Path, device: torch.device):
    blob = torch.load(out_dir / "masks.pt", map_location=device)
    return blob["mask_hard"].to(device), blob["mask_prob"].to(device)
