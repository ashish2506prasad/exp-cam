"""Per-image U-Net training. Architecture, optimizer, steps, and seed are held
fixed across ablations unless a section explicitly changes the loss weights
or adds a mask-budget constraint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from classifiers import ClassifierBundle, run_with_acts
from config import DEFAULT_LAMBDAS, LR, MASK_THRESH, ROB_K_BACKGROUNDS, SEED, STEPS, TEMPERATURE
from losses import (
    abductive_loss,
    area_loss,
    binarization_loss,
    ce_loss,
    hard_st,
    kl_loss,
    lagrangian_area_penalty,
    layer_act_loss,
    project_topk_mask,
    sample_gaussian_backgrounds,
    tv8_loss,
)
from models import UNetCSAE


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_all(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_seed(image_id: str, backbone: str, base: int = SEED) -> int:
    """Same init for every config of (image, backbone) — required for fair LOO."""
    return int(base + 17 * int(image_id) + 1009 * (sum(map(ord, backbone)) % 97))


@dataclass
class TrainConfig:
    lambdas: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LAMBDAS))
    steps: int = STEPS
    lr: float = LR
    seed: int = SEED
    temperature: float = TEMPERATURE
    mask_thresh: float = MASK_THRESH
    constraint: Optional[str] = None  # None | "project" | "lagrangian"
    p_target: Optional[float] = None
    mu_init: float = 0.0
    mu_eta: float = 5.0
    auto_balance: Optional[str] = None  # None | "uncertainty"
    record_curve: bool = False
    curve_every: int = 1
    rob_k: int = ROB_K_BACKGROUNDS
    extra_backgrounds: Optional[Sequence[torch.Tensor]] = None
    log_every: int = 50  # 0 = silent; notebooks printed every 10 steps


@dataclass
class TrainResult:
    mask_prob: torch.Tensor
    mask_hard: torch.Tensor
    e: torch.Tensor
    seconds: float
    steps: int
    lambdas: Dict[str, float]
    seed: int
    loss_curve: List[Dict[str, float]]
    weight_curve: List[Dict[str, float]]
    mu_final: float
    p_actual: float


def train_mask(
    bundle: ClassifierBundle,
    x: torch.Tensor,
    cfg: TrainConfig,
) -> TrainResult:
    device = x.device
    seed_all(cfg.seed)
    mask_net = UNetCSAE().to(device)
    mask_net.train()
    opt = torch.optim.Adam(mask_net.parameters(), lr=cfg.lr)

    s_params: Optional[nn.ParameterDict] = None
    if cfg.auto_balance == "uncertainty":
        # Kendall et al.: L = Σ exp(-s_i) L_i + s_i  (s_i = log-variance)
        keys = [k for k in cfg.lambdas if not (cfg.constraint is not None and k == "area")]
        s_params = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros((), device=device)) for k in keys}
        )
        opt = torch.optim.Adam(list(mask_net.parameters()) + list(s_params.parameters()), lr=cfg.lr)

    acts_x, logits_x = run_with_acts(bundle, x, require_grad=False)
    shallow = set(list(bundle.tap_names)[:2])
    mu = float(cfg.mu_init)
    curve: List[Dict[str, float]] = []
    wcurve: List[Dict[str, float]] = []

    t0 = time.perf_counter()
    if cfg.log_every:
        log(
            f"  training U-Net  steps={cfg.steps}  lr={cfg.lr}  seed={cfg.seed}"
            + (f"  constraint={cfg.constraint} p={cfg.p_target}" if cfg.constraint else "")
        )
    for t in range(1, cfg.steps + 1):
        opt.zero_grad()
        mask_logits = mask_net(x)
        mask_prob = torch.sigmoid(mask_logits)
        if cfg.constraint == "project" and cfg.p_target is not None:
            mask_prob = project_topk_mask(mask_prob, cfg.p_target)
        mask_st = hard_st(mask_prob, cfg.mask_thresh)
        e = mask_st.expand_as(x) * x

        acts_e, logits_e = run_with_acts(bundle, e, require_grad=True)
        loss_ce = ce_loss(logits_x, logits_e)
        loss_kl = kl_loss(logits_x, logits_e, temperature=cfg.temperature)
        loss_act = x.new_zeros(())
        for name, w in bundle.layer_weights.items():
            if name not in acts_x or name not in acts_e:
                continue
            loss_act = loss_act + w * layer_act_loss(name, acts_x[name], acts_e[name], shallow)
        loss_area = area_loss(mask_prob)
        loss_bin = binarization_loss(mask_prob)
        loss_tv = tv8_loss(mask_prob)
        bgs = list(cfg.extra_backgrounds) if cfg.extra_backgrounds else sample_gaussian_backgrounds(x, cfg.rob_k)
        loss_rob = abductive_loss(bundle.model, x, mask_prob, bgs, temperature=cfg.temperature)

        components = {
            "act": loss_act,
            "ce": loss_ce,
            "kl": loss_kl,
            "area": loss_area,
            "bin": loss_bin,
            "tv": loss_tv,
            "rob": loss_rob,
        }

        if cfg.constraint in ("project", "lagrangian"):
            # Experiment 2B / 5a: do not minimize area — only constrain it.
            components_for_sum = {k: v for k, v in components.items() if k != "area"}
        else:
            components_for_sum = components

        if s_params is not None:
            loss_total = x.new_zeros(())
            for k, L in components_for_sum.items():
                if k not in s_params:
                    loss_total = loss_total + cfg.lambdas.get(k, 0.0) * L
                    continue
                s = s_params[k]
                loss_total = loss_total + torch.exp(-s) * L + s
        else:
            loss_total = x.new_zeros(())
            for k, L in components_for_sum.items():
                loss_total = loss_total + cfg.lambdas.get(k, 0.0) * L

        if cfg.constraint == "lagrangian" and cfg.p_target is not None:
            p_now = mask_prob.mean()
            loss_total = loss_total + lagrangian_area_penalty(p_now, cfg.p_target, mu)

        loss_total.backward()
        opt.step()

        if cfg.constraint == "lagrangian" and cfg.p_target is not None:
            with torch.no_grad():
                viol = float(torch.relu(mask_prob.mean() - cfg.p_target).item())
            mu = max(0.0, mu + cfg.mu_eta * viol)

        if cfg.record_curve and (t % cfg.curve_every == 0 or t == 1 or t == cfg.steps):
            row = {k: float(v.detach().item()) for k, v in components.items()}
            row["total"] = float(loss_total.detach().item())
            row["step"] = t
            row["p_actual"] = float(mask_prob.mean().item())
            row["mu"] = mu
            curve.append(row)
            if s_params is not None:
                wcurve.append(
                    {"step": t, **{k: float(torch.exp(-s_params[k]).detach().item()) for k in s_params}}
                )

        if cfg.log_every and (t == 1 or t % cfg.log_every == 0 or t == cfg.steps):
            with torch.no_grad():
                top1 = int(logits_e.argmax(dim=1).item() == logits_x.argmax(dim=1).item())
                p_now = float((mask_prob > cfg.mask_thresh).float().mean().item())
            log(
                f"  step {t:>4d}/{cfg.steps}  total={loss_total.item():.3f}  "
                f"act={loss_act.item():.3f}  ce={loss_ce.item():.3f}  kl={loss_kl.item():.3f}  "
                f"area={loss_area.item():.3f}  bin={loss_bin.item():.3f}  tv={loss_tv.item():.4f}  "
                f"rob={loss_rob.item():.3f}  p={100.0 * p_now:.1f}%  top1={'Y' if top1 else 'N'}"
            )

    seconds = time.perf_counter() - t0
    mask_net.eval()
    with torch.no_grad():
        mask_prob = torch.sigmoid(mask_net(x))
        if cfg.constraint == "project" and cfg.p_target is not None:
            mask_prob = project_topk_mask(mask_prob, cfg.p_target)
        mask_hard = (mask_prob > cfg.mask_thresh).float()
        e = mask_hard.expand_as(x) * x
        p_actual = float(mask_hard.mean().item())

    if cfg.log_every:
        log(f"  finished in {seconds:.1f}s  p_actual={100.0 * p_actual:.1f}%")

    return TrainResult(
        mask_prob=mask_prob.detach(),
        mask_hard=mask_hard.detach(),
        e=e.detach(),
        seconds=seconds,
        steps=cfg.steps,
        lambdas=dict(cfg.lambdas),
        seed=cfg.seed,
        loss_curve=curve,
        weight_curve=wcurve,
        mu_final=mu,
        p_actual=p_actual,
    )


def save_mask_tensors(out_dir, result: TrainResult) -> None:
    from pathlib import Path
    from io_utils import ensure_dir

    out_dir = ensure_dir(Path(out_dir))
    torch.save(
        {
            "mask_hard": result.mask_hard.cpu(),
            "mask_prob": result.mask_prob.cpu(),
        },
        out_dir / "masks.pt",
    )
