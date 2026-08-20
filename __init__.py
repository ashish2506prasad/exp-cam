"""EXP-CAM revision experiments.

Shared evaluation module, per-image U-Net training, and one script per
revision section. Circuit-discovery code from AM0–AM6 is intentionally
not ported.

Canonical training defaults (from AM4–AM6 full-model cells):
    optimizer = Adam, lr = 5e-5, steps = 880, per-image U-Net from scratch
    λ_act=10, λ_CE=15, λ_KL=5, λ_area=100, λ_bin=1, λ_tv=500, λ_rob=100
"""

__all__ = ["config"]
