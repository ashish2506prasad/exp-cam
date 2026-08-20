"""Canonical hyperparameters, paths, and experiment constants.

Values below are the paper's later full-model defaults, taken from the
AM4/AM5/AM6 training cells (not the earlier AM1/AM2 hand-tuning grid).
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
TRAINX_DIR = PROJECT_ROOT / "trainx"
RUNS_DIR = PROJECT_ROOT / "expcam_runs"
EVAL_SET_PATH = RUNS_DIR / "eval_set.json"
WHERE = "local"

# Kaggle dataset layout (class folders such as n01440764 live under this root).
KAGGLE_TRAINX = Path("/kaggle/input/datasets/meashish2003/trainx/trainx")
KAGGLE_RUNS = Path("/kaggle/working/expcam_runs")
KAGGLE_WORKING = Path("/kaggle/working")

SEED = 0
N_IMAGES = 70  # 10 per class × 7 trainx classes (within the recommended 50–100)
N_PER_CLASS = 10
IMAGE_SIZE = 224

DEFAULT_BACKBONES = ("resnet18", "efficientnet_b0", "vit_b_16")

# Per-image U-Net training (AM6 cell 15 / AM4–AM5 full-model cells)
STEPS = 880
LR = 5e-5
OPTIMIZER_NAME = "adam"
TEMPERATURE = 2.0
MASK_THRESH = 0.5
ROB_K_BACKGROUNDS = 3

DEFAULT_LAMBDAS = {
    "act": 10.0,
    "ce": 15.0,
    "kl": 5.0,
    "area": 100.0,
    "bin": 1.0,
    "tv": 500.0,
    "rob": 100.0,  # abductive / L_rob in the notebooks (lam_abd)
}

LOSS_TERMS = ("act", "ce", "kl", "area", "bin", "tv", "rob")
SWEEP_TERMS = ("ce", "kl", "area", "bin", "tv", "rob")  # L_act stays fixed
SWEEP_MULTS = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)

BUDGETS = (0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
FIDELITY_TAUS = (0.95, 0.90)
P_TARGETS = (0.05, 0.10, 0.20)

LAYER_WEIGHTS_RESNET = {
    "relu": 1.0,
    "layer1": 2.0,
    "layer2": 2.0,
    "layer3": 4.0,
    "layer4": 8.0,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GREY_BG = 0.5

# ImageNet-1K synsets present in trainx/
TRAINX_CLASSES = {
    "n01440764": {"name": "tench", "imagenet_idx": 0},
    "n01484850": {"name": "great_white_shark", "imagenet_idx": 2},
    "n01494475": {"name": "hammerhead", "imagenet_idx": 4},
    "n01531178": {"name": "goldfinch", "imagenet_idx": 11},
    "n01632777": {"name": "axolotl", "imagenet_idx": 29},
    "n01665541": {"name": "leatherback_turtle", "imagenet_idx": 34},
    "n01687978": {"name": "agama", "imagenet_idx": 42},
}

# Dual-axis metric chosen per swept term (Section 1b / Figs A3–A8)
SWEEP_SECONDARY_METRIC = {
    "ce": "top1_agreement",
    "kl": "kl_divergence",
    "area": "top1_agreement",
    "bin": "binarization_sharpness",
    "tv": "tv_energy_per_pixel",
    "rob": "top1_agreement",
}

N_BG_DRAWS = 5
N_INSERT_STEPS = 50
N_RISE_MASKS = 800
N_INSERT_SEEDS = 5
AUG_ROT_DEG = 5.0
AUG_TRANSLATE_FRAC = 0.05
AUG_BRIGHTNESS = 0.10

# Retrospective count of distinct λ tuples in AM0–AM6 (Section 5 "tuning cost")
MANUAL_TRIAL_CONFIGS = 18


def detect_where() -> str:
    if Path("/kaggle/input").exists() or Path("/kaggle/working").exists():
        return "kaggle"
    return "local"


def apply_where(where: str = "auto", data_root=None, runs_dir=None) -> str:
    """Switch trainx/ and output roots for local vs Kaggle.

    local:  <repo>/trainx  and  <repo>/expcam_runs
    kaggle: /kaggle/input/datasets/meashish2003/trainx/trainx
            and /kaggle/working/expcam_runs
    """
    global PROJECT_ROOT, TRAINX_DIR, RUNS_DIR, EVAL_SET_PATH, WHERE
    if where in (None, "auto"):
        where = detect_where()
    WHERE = where
    if where == "kaggle":
        PROJECT_ROOT = KAGGLE_WORKING
        TRAINX_DIR = KAGGLE_TRAINX
        RUNS_DIR = KAGGLE_RUNS
    elif where == "local":
        PROJECT_ROOT = PACKAGE_DIR.parent
        TRAINX_DIR = PROJECT_ROOT / "trainx"
        RUNS_DIR = PROJECT_ROOT / "expcam_runs"
    else:
        raise ValueError(f"Unknown --where {where!r}; expected local, kaggle, or auto")
    if data_root is not None:
        TRAINX_DIR = Path(data_root)
    if runs_dir is not None:
        RUNS_DIR = Path(runs_dir)
    EVAL_SET_PATH = RUNS_DIR / "eval_set.json"
    print(f"[paths] where={WHERE} trainx={TRAINX_DIR} runs={RUNS_DIR}", flush=True)
    return WHERE


def add_where_args(p):
    p.add_argument(
        "--where",
        choices=("local", "kaggle", "auto"),
        default="auto",
        help=(
            "Path layout. local: <repo>/trainx and <repo>/expcam_runs. "
            "kaggle: /kaggle/input/datasets/meashish2003/trainx/trainx and "
            "/kaggle/working/expcam_runs. auto: kaggle if /kaggle exists."
        ),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the image folder (must contain n01440764 and the other synsets).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run directory. Default: expcam_runs (local) or /kaggle/working/expcam_runs (kaggle).",
    )
    return p


def lambdas_with_overrides(**overrides) -> dict:
    out = dict(DEFAULT_LAMBDAS)
    out.update(overrides)
    return out


def loo_configs() -> dict:
    """Full model plus seven leave-one-out coefficient sets."""
    cfgs = {"Full": dict(DEFAULT_LAMBDAS)}
    for term in LOSS_TERMS:
        lam = dict(DEFAULT_LAMBDAS)
        lam[term] = 0.0
        cfgs[f"-L_{term}"] = lam
    return cfgs
