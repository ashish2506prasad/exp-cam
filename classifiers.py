"""Frozen ImageNet classifiers + activation hooks (ResNet-18, EfficientNet-B0, ViT-B/16)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from config import LAYER_WEIGHTS_RESNET


TapSpec = Dict[str, nn.Module]


@dataclass
class ClassifierBundle:
    name: str
    model: nn.Module
    preprocess: Callable
    labels: List[str]
    tap_names: Tuple[str, ...]
    layer_weights: Dict[str, float]
    cam_layer: nn.Module
    cam_is_vit: bool = False


def _resnet18(device: torch.device) -> ClassifierBundle:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    taps = {
        "relu": model.relu,
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
    }
    return ClassifierBundle(
        name="resnet18",
        model=model,
        preprocess=weights.transforms(),
        labels=list(weights.meta["categories"]),
        tap_names=tuple(taps.keys()),
        layer_weights=dict(LAYER_WEIGHTS_RESNET),
        cam_layer=model.layer4[-1],
        cam_is_vit=False,
    )


def _efficientnet_b0(device: torch.device) -> ClassifierBundle:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    taps = {
        "f1": model.features[1],
        "f3": model.features[3],
        "f5": model.features[5],
        "f7": model.features[7],
        "f8": model.features[8],
    }
    weights_l = {"f1": 1.0, "f3": 2.0, "f5": 2.0, "f7": 4.0, "f8": 8.0}
    return ClassifierBundle(
        name="efficientnet_b0",
        model=model,
        preprocess=weights.transforms(),
        labels=list(weights.meta["categories"]),
        tap_names=tuple(taps.keys()),
        layer_weights=weights_l,
        cam_layer=model.features[-1],
        cam_is_vit=False,
    )


def _vit_b_16(device: torch.device) -> ClassifierBundle:
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.encoder.layers
    taps = {
        "block3": layers[3],
        "block7": layers[7],
        "block11": layers[11],
    }
    weights_l = {"block3": 1.0, "block7": 2.0, "block11": 8.0}
    return ClassifierBundle(
        name="vit_b_16",
        model=model,
        preprocess=weights.transforms(),
        labels=list(weights.meta["categories"]),
        tap_names=tuple(taps.keys()),
        layer_weights=weights_l,
        cam_layer=layers[11],
        cam_is_vit=True,
    )


_BUILDERS = {
    "resnet18": _resnet18,
    "efficientnet_b0": _efficientnet_b0,
    "vit_b_16": _vit_b_16,
}


def load_classifier(name: str, device: torch.device) -> ClassifierBundle:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown backbone {name}; choose from {list(_BUILDERS)}")
    return _BUILDERS[name](device)


def tap_modules(bundle: ClassifierBundle) -> TapSpec:
    model = bundle.model
    if bundle.name == "resnet18":
        return {
            "relu": model.relu,
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        }
    if bundle.name == "efficientnet_b0":
        return {
            "f1": model.features[1],
            "f3": model.features[3],
            "f5": model.features[5],
            "f7": model.features[7],
            "f8": model.features[8],
        }
    return {
        "block3": model.encoder.layers[3],
        "block7": model.encoder.layers[7],
        "block11": model.encoder.layers[11],
    }


def run_with_acts(
    bundle: ClassifierBundle,
    x: torch.Tensor,
    require_grad: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Forward pass that returns named activations and logits."""
    model = bundle.model
    model.eval()
    acts: Dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def fn(_m, _inp, out):
            acts[name] = out if require_grad else out.detach()

        return fn

    for name, module in tap_modules(bundle).items():
        handles.append(module.register_forward_hook(hook(name)))
    with torch.set_grad_enabled(require_grad):
        logits = model(x)
    for h in handles:
        h.remove()
    return acts, logits


def spatial_feat(feat: torch.Tensor) -> torch.Tensor:
    """Map ViT token maps (B, T, C) to (B, C, H, W) when possible; leave CNN maps alone."""
    if feat.dim() == 4:
        return feat
    if feat.dim() == 3:
        b, t, c = feat.shape
        # drop CLS if present
        grid = int(round((t - 1) ** 0.5))
        if grid * grid == t - 1:
            patches = feat[:, 1:, :].transpose(1, 2).reshape(b, c, grid, grid)
            return patches
        grid = int(round(t ** 0.5))
        if grid * grid == t:
            return feat.transpose(1, 2).reshape(b, c, grid, grid)
    return feat
