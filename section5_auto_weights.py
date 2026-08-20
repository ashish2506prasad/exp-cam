"""Section 5 — Automatic hyperparameter weighting.

5a. Lagrangian / projection replaces λ_area with a target budget p_target.
5b. Kendall uncertainty weighting for the remaining terms (chosen over GradNorm
    as the simpler, more stable option).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cli import add_common_args, resolve_run
from .classifiers import load_classifier
from .config import DEFAULT_LAMBDAS, MANUAL_TRIAL_CONFIGS, P_TARGETS
from .io_utils import load_json, mean_std, rows_to_excel, save_json
from .runner import maybe_train_and_eval
from .train import TrainConfig, image_seed, log
from .viz import boxplot, grouped_bars, scatter_parity


def run_section5(args) -> None:
    eval_set, images, device = resolve_run(args)
    out_root = args.out / "section5"
    p_targets = P_TARGETS if not args.quick else (0.10,)
    rows = []

    log(
        f"Section 5  device={device}  images={len(images)}  p_targets={list(p_targets)}  "
        f"constraint={args.constraint}  steps={args.steps}"
    )
    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        for p in p_targets:
            log(f"--- auto p_target={p:g} / {backbone} ---")
            for rec in images:
                # 5a+5b together: constraint on area, uncertainty on the rest
                lam = dict(DEFAULT_LAMBDAS)
                lam["area"] = 0.0
                cfg = TrainConfig(
                    lambdas=lam,
                    steps=args.steps,
                    seed=image_seed(rec["id"], backbone, args.seed),
                    constraint=args.constraint,
                    p_target=p,
                    auto_balance="uncertainty",
                    log_every=args.log_every,
                    record_curve=(
                        rec["id"] == eval_set["curve_image_id"]
                        and backbone == args.backbones[0]
                        and abs(p - 0.10) < 1e-9
                    ),
                )
                out_dir = out_root / backbone / f"auto_p{p:g}" / rec["id"]
                row = maybe_train_and_eval(
                    bundle,
                    rec,
                    Path(eval_set["root"]),
                    out_dir,
                    cfg,
                    skip_existing=args.skip_existing,
                    device=device,
                    extra_metrics={"protocol": "auto", "p_target": p},
                )
                row["p_target"] = p
                row["p_gap"] = abs(p - row["p_actual"])
                rows.append(row)

    # Manual comparison: Section 1 Full model (fixed λ_area)
    manual = []
    s1 = args.out / "section1" / "all_runs.json"
    if s1.exists():
        manual = [r for r in load_json(s1) if r.get("protocol") == "loo" and r.get("config") == "Full"]

    save_json(out_root / "auto_runs.json", rows)
    save_json(
        out_root / "tuning_cost.json",
        {
            "manual_trial_configs_from_AM_notebooks": MANUAL_TRIAL_CONFIGS,
            "automatic_runs_per_image": 1,
            "note": "AM0–AM6 contain ~18 distinct λ tuples; auto-tuning is a single run per image.",
        },
    )
    rows_to_excel(out_root / "section5_results.xlsx", {"auto": rows, "manual_full": manual})
    _figures(eval_set, images, args, rows, manual)


def _figures(eval_set, images, args, rows, manual) -> None:
    fig_dir = args.out / "section5" / "figures"
    backbone = args.backbones[0]
    p_targets = sorted({r["p_target"] for r in rows})
    # Fig F1: variance of p_actual, manual vs auto, one panel group per p_target
    # Manual has a single natural size; compare std at each auto p_target vs that global manual distribution
    groups = {}
    man_sizes = [100 * r["p_actual"] for r in manual if r["backbone"] == backbone]
    if man_sizes:
        groups["Manual fixed λ_area"] = man_sizes
    for p in p_targets:
        groups[f"Auto p={p*100:g}%"] = [
            100 * r["p_actual"] for r in rows if r["backbone"] == backbone and abs(r["p_target"] - p) < 1e-9
        ]
    if groups:
        # draw one boxplot; mark the first p_target
        boxplot(
            fig_dir / "FigF1_variance_reduction.png",
            groups,
            "achieved mask size %",
            "Fig F1 — mask-size variance, manual vs automatic constraint",
            hline=100 * p_targets[0] if p_targets else None,
        )

    # Fig F2 fidelity parity at closest matched size
    if manual and rows:
        man_by_id = {(r["backbone"], r["image_id"]): r for r in manual}
        xs, ys = [], []
        for r in rows:
            key = (r["backbone"], r["image_id"])
            if key not in man_by_id:
                continue
            if abs(r["p_target"] - 0.10) > 1e-9 and 0.10 in p_targets:
                continue
            xs.append(man_by_id[key]["top1_agreement"])
            ys.append(r["top1_agreement"])
        if xs:
            scatter_parity(
                fig_dir / "FigF2_fidelity_parity.png",
                xs,
                ys,
                "top-1 agreement (manual)",
                "top-1 agreement (automatic)",
                "Fig F2 — per-image fidelity parity",
            )
            diffs = np.asarray(ys) - np.asarray(xs)
            save_json(
                args.out / "section5" / "fidelity_parity_stats.json",
                {
                    "mean_diff": float(diffs.mean()),
                    "std_diff": float(diffs.std(ddof=1) if len(diffs) > 1 else 0.0),
                    "n": int(len(diffs)),
                    "note": "paired per-image difference auto − manual (top-1 agreement)",
                },
            )

    curve = args.out / "section5" / backbone / "auto_p0.1" / eval_set["curve_image_id"] / "weight_curve.json"
    if not curve.exists():
        curve = args.out / "section5" / backbone / "auto_p0.10" / eval_set["curve_image_id"] / "weight_curve.json"
    if curve.exists():
        data = load_json(curve)
        # reuse training_curves-style plot for weights
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        steps = [r["step"] for r in data]
        keys = [k for k in data[0] if k != "step"]
        for k in keys:
            ax.plot(steps, [r[k] for r in data], label=k)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("effective weight exp(-s_i)")
        ax.set_title("Fig F3 — uncertainty-weight trajectories")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "FigF3_weight_trajectories.png", dpi=150)
        plt.close(fig)

    # Fig F4: transfer across backbones (proxy for domains; trainx is one domain)
    labels = list(args.backbones)
    means_auto, stds_auto, means_man, stds_man = [], [], [], []
    for bb in labels:
        a = [r["top1_agreement"] for r in rows if r["backbone"] == bb]
        m = [r["top1_agreement"] for r in manual if r["backbone"] == bb]
        sa, sm = mean_std(a), mean_std(m)
        means_auto.append(sa["mean"] or 0)
        stds_auto.append(sa["std"] or 0)
        means_man.append(sm["mean"] or 0)
        stds_man.append(sm["std"] or 0)
    grouped_bars(
        fig_dir / "FigF4_backbone_transfer.png",
        labels,
        {
            "manual λ": (means_man, stds_man),
            "automatic": (means_auto, stds_auto),
        },
        "top-1 agreement",
        "Fig F4 — same auto scheme across backbones (trainx ImageNet subset)",
    )


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 5: automatic weights"))
    p.add_argument("--constraint", choices=("lagrangian", "project"), default="lagrangian")
    args = p.parse_args(argv)
    run_section5(args)


if __name__ == "__main__":
    main()
