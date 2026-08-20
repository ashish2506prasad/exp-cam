"""Size-matched CAM baselines via jacobgil/pytorch-grad-cam (not from AM0–AM6).

The notebooks never implemented Grad-CAM, Grad-CAM++, LayerCAM, or attention
heatmaps. Those were custom code; this module uses the standard library instead.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from classifiers import ClassifierBundle


def _otsu(heatmap: np.ndarray) -> float:
    h = np.clip(heatmap, 0, None)
    h = h / (h.max() + 1e-12)
    hist, bins = np.histogram(h.ravel(), bins=256, range=(0, 1))
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma))
    return float(bins[idx])


def otsu_binarize(heatmap: torch.Tensor) -> torch.Tensor:
    h = heatmap.detach().cpu().numpy()
    t = _otsu(h)
    return (heatmap >= t).float()


def topk_binarize(heatmap: torch.Tensor, k: int) -> torch.Tensor:
    flat = heatmap.reshape(-1)
    k = int(max(1, min(k, flat.numel())))
    idx = torch.topk(flat, k).indices
    m = torch.zeros_like(flat)
    m[idx] = 1.0
    return m.view_as(heatmap)


def reshape_transform_vit(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Drop CLS token and fold ViT tokens into a spatial feature map."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def cam_target_layers(bundle: ClassifierBundle):
    """Library target layers, matching the pytorch_grad_cam README convention."""
    model = bundle.model
    if bundle.name == "resnet18":
        return [model.layer4[-1]], None
    if bundle.name == "efficientnet_b0":
        return [model.features[-1]], None
    return [model.encoder.layers[-1].ln_1], reshape_transform_vit


def _library_cam(cam_cls, bundle: ClassifierBundle, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
    try:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as e:
        raise ImportError(
            "Section 3 needs `grad-cam` (import pytorch_grad_cam). "
            "Install with: pip install grad-cam"
        ) from e

    layers, reshape = cam_target_layers(bundle)
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(bundle.model(x).argmax(dim=1)[0].item())
    targets = [ClassifierOutputTarget(class_idx)]
    kwargs = {"model": bundle.model, "target_layers": layers}
    if reshape is not None:
        kwargs["reshape_transform"] = reshape
    was = [p.requires_grad for p in bundle.model.parameters()]
    for p in bundle.model.parameters():
        p.requires_grad_(True)
    try:
        with cam_cls(**kwargs) as cam:
            grayscale = cam(input_tensor=x, targets=targets)
    finally:
        for p, flag in zip(bundle.model.parameters(), was):
            p.requires_grad_(flag)
    t = torch.from_numpy(np.asarray(grayscale[0])).to(device=x.device, dtype=x.dtype)
    return t / (t.max() + 1e-12)


def grad_cam(bundle: ClassifierBundle, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
    from pytorch_grad_cam import GradCAM

    return _library_cam(GradCAM, bundle, x, class_idx)


def grad_cam_pp(bundle: ClassifierBundle, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
    from pytorch_grad_cam import GradCAMPlusPlus

    return _library_cam(GradCAMPlusPlus, bundle, x, class_idx)


def layer_cam(bundle: ClassifierBundle, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
    from pytorch_grad_cam import LayerCAM

    return _library_cam(LayerCAM, bundle, x, class_idx)


def eigen_cam(bundle: ClassifierBundle, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
    from pytorch_grad_cam import EigenCAM

    return _library_cam(EigenCAM, bundle, x, class_idx)


@torch.no_grad()
def vit_attention_map(bundle: ClassifierBundle, x: torch.Tensor) -> torch.Tensor:
    """CLS-to-patch attention of the last ViT block (ViT-only attention baseline)."""
    model = bundle.model
    z = model._process_input(x)
    n = z.shape[0]
    z = torch.cat([model.class_token.expand(n, -1, -1), z], dim=1)
    z = z + model.encoder.pos_embedding
    z = model.encoder.dropout(z)
    layers = model.encoder.layers
    for i, blk in enumerate(layers):
        if i == len(layers) - 1:
            x_ln = blk.ln_1(z)
            _out, w = blk.self_attention(
                x_ln, x_ln, x_ln, need_weights=True, average_attn_weights=True
            )
            cls_to_patch = w[:, 0, 1:]
            g = int(round(cls_to_patch.shape[-1] ** 0.5))
            cam = cls_to_patch.reshape(n, 1, g, g)
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            return cam / (cam.max() + 1e-12)
        z = blk(z)
    raise RuntimeError("ViT attention extraction failed")


@torch.no_grad()
def rise(
    bundle: ClassifierBundle,
    x: torch.Tensor,
    n_masks: int = 800,
    grid: int = 7,
    p1: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """RISE (Petsiuk et al.): random mask weighted by class score."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    b, c, h, w = x.shape
    y = bundle.model(x).argmax(dim=1)
    scores = []
    masks = []
    for _ in range(n_masks):
        small = (torch.rand(1, 1, grid, grid, generator=g) < p1).float()
        up = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).to(x.device)
        masked = x * up
        conf = torch.softmax(bundle.model(masked), dim=1)[0, y[0]]
        scores.append(conf)
        masks.append(up[0, 0])
    scores_t = torch.stack(scores)
    masks_t = torch.stack(masks)
    cam = (scores_t[:, None, None] * masks_t).sum(0) / (masks_t.sum(0) + 1e-12)
    return cam / (cam.max() + 1e-12)


BASELINE_NAMES = ("gradcam", "gradcam++", "layercam", "attention", "rise")


def compute_heatmap(name: str, bundle: ClassifierBundle, x: torch.Tensor, rise_masks: int = 800) -> Optional[torch.Tensor]:
    if name == "gradcam":
        return grad_cam(bundle, x)
    if name == "gradcam++":
        return grad_cam_pp(bundle, x)
    if name == "layercam":
        return layer_cam(bundle, x)
    if name == "attention":
        if not bundle.cam_is_vit:
            return None
        return vit_attention_map(bundle, x)
    if name == "rise":
        return rise(bundle, x, n_masks=rise_masks)
    raise ValueError(name)
