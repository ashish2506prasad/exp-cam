"""Training losses ported from AM0/AM6, plus Section 2/5 constraint helpers.

Paper groups:
    L_AM  = λ_act L_act + λ_CE L_CE + λ_KL L_KL
    L_MIN = λ_area L_area + λ_bin L_bin + λ_tv L_tv
    L_ROB = λ_rob L_rob
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F


def hard_st(p: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """Straight-through binarizer (AM0/AM6)."""
    m_hard = (p > thresh).float()
    return p + (m_hard - p).detach()


def ce_loss(logits_x: torch.Tensor, logits_e: torch.Tensor) -> torch.Tensor:
    y_star = logits_x.argmax(dim=1)
    return F.cross_entropy(logits_e, y_star)


def kl_loss(logits_x: torch.Tensor, logits_e: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    p_x = F.softmax(logits_x / temperature, dim=1)
    log_p_e = F.log_softmax(logits_e / temperature, dim=1)
    return F.kl_div(log_p_e, p_x, reduction="batchmean") * (temperature ** 2)


def cosine_loss(feat_x: torch.Tensor, feat_e: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if feat_x.dim() == 4:
        b, c, h, w = feat_x.shape
        feat_x = feat_x.view(b, c, -1)
        feat_e = feat_e.view(b, c, -1)
    elif feat_x.dim() == 3:
        feat_x = feat_x.transpose(1, 2)
        feat_e = feat_e.transpose(1, 2)
    feat_x = F.normalize(feat_x, dim=1, eps=eps)
    feat_e = F.normalize(feat_e, dim=1, eps=eps)
    cos = (feat_x * feat_e).sum(dim=1)
    return 1 - cos.mean()


def mse_loss(feat_x: torch.Tensor, feat_e: torch.Tensor) -> torch.Tensor:
    def norm(t):
        b, c, *spatial = t.shape
        flat = t.reshape(b, c, -1)
        mean = flat.mean(dim=-1, keepdim=True)
        std = flat.std(dim=-1, keepdim=True) + 1e-6
        return ((flat - mean) / std).view_as(t)

    return F.mse_loss(norm(feat_x), norm(feat_e))


def area_loss(mask_prob: torch.Tensor) -> torch.Tensor:
    return mask_prob.mean()


def binarization_loss(mask_prob: torch.Tensor) -> torch.Tensor:
    return (mask_prob * (1.0 - mask_prob)).mean()


def tv_loss(mask_prob: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(mask_prob[:, :, 1:, :] - mask_prob[:, :, :-1, :]).mean()
    dw = torch.abs(mask_prob[:, :, :, 1:] - mask_prob[:, :, :, :-1]).mean()
    return dh + dw


def tv8_loss(p: torch.Tensor, w_ax: float = 1.0, w_diag: float = 1.0 / 1.41421356237, eps: float = 0.0) -> torch.Tensor:
    """8-connected TV used in the later AM notebooks."""

    def diff(a, b):
        d = (a - b).abs()
        return d if eps == 0 else torch.sqrt(d * d + eps)

    tv = w_ax * diff(p[:, :, 1:, :], p[:, :, :-1, :]).mean()
    tv = tv + w_ax * diff(p[:, :, :, 1:], p[:, :, :, :-1]).mean()
    center = p[:, :, 1:, 1:]
    tv = tv + w_diag * diff(center, p[:, :, :-1, :-1]).mean()
    tv = tv + w_diag * diff(p[:, :, 1:, :-1], p[:, :, :-1, 1:]).mean()
    return tv


def abductive_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask_prob: torch.Tensor,
    backgrounds: Sequence[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """L_rob: prediction KL under background replacement (AM0/AM6)."""
    # STE so L_rob still trains the generator (notebook used a hard threshold with no grad)
    mask = hard_st(mask_prob).expand_as(x)
    with torch.no_grad():
        px = F.softmax(model(x) / temperature, dim=1)
    loss = x.new_zeros(())
    n = 0
    for r in backgrounds:
        r = r.to(x.device)
        e_r = mask * x + (1.0 - mask) * r
        log_pe = F.log_softmax(model(e_r) / temperature, dim=1)
        loss = loss + F.kl_div(log_pe, px, reduction="batchmean") * (temperature ** 2)
        n += 1
    return loss / max(1, n)


def sample_gaussian_backgrounds(x: torch.Tensor, k: int = 3) -> List[torch.Tensor]:
    """Original notebook L_rob perturbations (grey / Gaussian / uniform)."""
    bgs = [
        torch.zeros_like(x),
        torch.randn_like(x),
        torch.empty_like(x).uniform_(-1, 1),
    ]
    return bgs[:k]


def project_topk_mask(mask_prob: torch.Tensor, p_budget: float) -> torch.Tensor:
    """Keep the top p·N scores; zero the rest (Section 2B projected gradient)."""
    b, _, h, w = mask_prob.shape
    n = h * w
    k = max(1, int(round(p_budget * n)))
    flat = mask_prob.reshape(b, -1)
    if k >= n:
        return mask_prob
    thresh = torch.topk(flat, k, dim=1).values[:, -1:]
    projected = torch.where(flat >= thresh, flat, torch.zeros_like(flat))
    return projected.view_as(mask_prob)


def lagrangian_area_penalty(p_actual: torch.Tensor, p_target: float, mu: float) -> torch.Tensor:
    return mu * torch.relu(p_actual - p_target)


def layer_act_loss(name: str, fx: torch.Tensor, fe: torch.Tensor, shallow_names: Iterable[str]) -> torch.Tensor:
    if name in set(shallow_names):
        return cosine_loss(fx, fe)
    return mse_loss(fx, fe)
