"""Section 1 — Hyperparameter ablation (LOO + single-term sweeps).

Must run first: later sections reuse the Full-model masks and the λ_area sweep.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .cli import add_common_args, resolve_run
from .classifiers import load_classifier
from .config import (
    DEFAULT_LAMBDAS,
    LOSS_TERMS,
    SWEEP_MULTS,
    SWEEP_SECONDARY_METRIC,
    SWEEP_TERMS,
    loo_configs,
)
from .data import load_tensor
from .io_utils import mean_std, rows_to_excel, save_json
from .runner import maybe_train_and_eval, run_dir
from .train import TrainConfig, image_seed, log
from .viz import (
    caption,
    grouped_bars_dual,
    image_grid,
    panel_expl,
    panel_mask,
    panel_original,
    small_multiples_bars,
    sweep_dual_axis,
    training_curves,
)


def _agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean_std(vals)


def run_section1(args) -> None:
    eval_set, images, device = resolve_run(args)
    out_root = args.out
    section = "section1"
    all_rows = []

    loo = loo_configs()
    if args.quick:
        # keep Full + a couple of LOO terms so the pipeline is testable
        loo = {k: loo[k] for k in ("Full", "-L_area", "-L_tv") if k in loo}

    log(
        f"Section 1  device={device}  images={len(images)}  backbones={list(args.backbones)}  "
        f"steps={args.steps}  loo_cfgs={len(loo)}  log_every={args.log_every}"
    )

    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        for cfg_name, lambdas in loo.items():
            log(f"--- LOO {cfg_name} / {backbone} ---")
            for rec in images:
                out_dir = run_dir(out_root, section, backbone, f"loo_{cfg_name}", rec["id"])
                cfg = TrainConfig(
                    lambdas=lambdas,
                    steps=args.steps,
                    seed=image_seed(rec["id"], backbone, args.seed),
                    log_every=args.log_every,
                    record_curve=(
                        cfg_name == "Full"
                        and rec["id"] == eval_set["curve_image_id"]
                        and backbone == args.backbones[0]
                    ),
                )
                row = maybe_train_and_eval(
                    bundle,
                    rec,
                    Path(eval_set["root"]),
                    out_dir,
                    cfg,
                    skip_existing=args.skip_existing,
                    device=device,
                    extra_metrics={"protocol": "loo", "config": cfg_name},
                )
                all_rows.append(row)

        # 1b. single-term sweeps (L_act held at default)
        sweep_terms = SWEEP_TERMS if not args.quick else ("area", "tv")
        sweep_mults = SWEEP_MULTS if not args.quick else (0.0, 1.0, 10.0)
        for term in sweep_terms:
            for mult in sweep_mults:
                lambdas = dict(DEFAULT_LAMBDAS)
                lambdas[term] = DEFAULT_LAMBDAS[term] * mult
                cfg_name = f"sweep_{term}_x{mult:g}"
                log(f"--- sweep {cfg_name} / {backbone} ---")
                for rec in images:
                    out_dir = run_dir(out_root, section, backbone, cfg_name, rec["id"])
                    cfg = TrainConfig(
                        lambdas=lambdas,
                        steps=args.steps,
                        seed=image_seed(rec["id"], backbone, args.seed),
                        log_every=args.log_every,
                    )
                    row = maybe_train_and_eval(
                        bundle,
                        rec,
                        Path(eval_set["root"]),
                        out_dir,
                        cfg,
                        skip_existing=args.skip_existing,
                        device=device,
                        extra_metrics={
                            "protocol": "sweep",
                            "term": term,
                            "mult": mult,
                            "lambda_value": lambdas[term],
                            "config": cfg_name,
                        },
                    )
                    all_rows.append(row)

    save_json(out_root / section / "all_runs.json", all_rows)
    rows_to_excel(
        out_root / section / "section1_results.xlsx",
        {
            "loo": [r for r in all_rows if r.get("protocol") == "loo"],
            "sweep": [r for r in all_rows if r.get("protocol") == "sweep"],
            "all": all_rows,
        },
    )
    _make_figures(eval_set, images, args, all_rows, device)


def _make_figures(eval_set, images, args, all_rows, device) -> None:
    fig_dir = args.out / "section1" / "figures"
    loo_order = ["Full"] + [f"-L_{t}" for t in LOSS_TERMS]
    # Fig A1 / A2: average over images, one backbone panel-set (resnet18 if present)
    backbone = args.backbones[0]
    loo_rows = [r for r in all_rows if r.get("protocol") == "loo" and r["backbone"] == backbone]
    by_cfg = defaultdict(list)
    for r in loo_rows:
        by_cfg[r["config"]].append(r)
    labels = [c for c in loo_order if c in by_cfg]
    size_m, size_s, agr_m, agr_s = [], [], [], []
    for c in labels:
        a = _agg(by_cfg[c], "p_actual_pct")
        b = _agg(by_cfg[c], "top1_agreement")
        size_m.append(a["mean"] or 0)
        size_s.append(a["std"] or 0)
        agr_m.append(100.0 * (b["mean"] or 0))
        agr_s.append(100.0 * (b["std"] or 0))
    grouped_bars_dual(
        fig_dir / "FigA1_loo_summary.png",
        labels,
        size_m,
        size_s,
        agr_m,
        agr_s,
        "mask size %",
        "top-1 agreement %",
        f"Fig A1 — LOO sparsity/fidelity ({backbone}, mean±std over {len(images)} images)",
    )
    detail = {}
    for key, title in (
        ("kl_divergence", "KL divergence"),
        ("confidence_delta", "confidence delta"),
        ("tv_energy_per_pixel", "TV energy / pixel"),
        ("binarization_sharpness", "L_bin"),
    ):
        means, stds = [], []
        for c in labels:
            s = _agg(by_cfg[c], key)
            means.append(s["mean"] or 0)
            stds.append(s["std"] or 0)
        detail[title] = (means, stds)
    small_multiples_bars(
        fig_dir / "FigA2_loo_detail.png",
        labels,
        detail,
        f"Fig A2 — LOO detail ({backbone})",
    )

    # Figs A3–A8
    for i, term in enumerate(SWEEP_TERMS, start=3):
        rows = [
            r
            for r in all_rows
            if r.get("protocol") == "sweep" and r.get("term") == term and r["backbone"] == backbone
        ]
        if not rows:
            continue
        by_mult = defaultdict(list)
        for r in rows:
            by_mult[float(r["mult"])].append(r)
        xs = sorted(by_mult)
        left, ls, right, rs = [], [], [], []
        sec = SWEEP_SECONDARY_METRIC[term]
        for m in xs:
            a = _agg(by_mult[m], "p_actual_pct")
            b = _agg(by_mult[m], sec)
            left.append(a["mean"] or 0)
            ls.append(a["std"] or 0)
            scale = 100.0 if sec == "top1_agreement" else 1.0
            right.append(scale * (b["mean"] or 0))
            rs.append(scale * (b["std"] or 0))
        sweep_dual_axis(
            fig_dir / f"FigA{i}_sweep_L_{term}.png",
            xs,
            left,
            ls,
            right,
            rs,
            "mask size %",
            sec,
            f"Fig A{i} — λ_{term} sweep ({backbone})",
        )

    # Fig A10 training curves
    curve_path = (
        args.out
        / "section1"
        / args.backbones[0]
        / "loo_Full"
        / eval_set["curve_image_id"]
        / "loss_curve.json"
    )
    if curve_path.exists():
        from .io_utils import load_json

        training_curves(fig_dir / "FigA10_training_curves.png", load_json(curve_path))

    # Fig A11 qualitative grid — same images for every ablated term
    _fig_a11(eval_set, args, device, fig_dir)


def _fig_a11(eval_set, args, device, fig_dir: Path) -> None:
    from .classifiers import load_classifier
    from .runner import load_masks

    qids = eval_set["qualitative_ids"][:4]
    id_to_rec = {r["id"]: r for r in eval_set["images"]}
    backbone = args.backbones[0]
    bundle = load_classifier(backbone, device)
    cols = [
        ("Full", "loo_Full"),
        ("-L_area", "loo_-L_area"),
        ("-L_bin", "loo_-L_bin"),
        ("-L_tv", "loo_-L_tv"),
        ("-L_rob", "loo_-L_rob"),
    ]
    rows = []
    for qid in qids:
        rec = id_to_rec[qid]
        x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
        row = [(panel_original(x), f"{rec['class_name']}\noriginal")]
        for label, cfg_name in cols:
            d = run_dir(args.out, "section1", backbone, cfg_name, qid)
            if not (d / "masks.pt").exists():
                continue
            from .io_utils import load_json

            m_hard, _ = load_masks(d, device)
            met = load_json(d / "metrics.json")
            cap = caption(met["p_actual_pct"], met["conf_e"], met["top1_agreement"])
            if label == "Full":
                row.append((panel_mask(m_hard), f"Full mask\n{cap}"))
                row.append((panel_expl(x, m_hard), f"Full expl.\n{cap}"))
            else:
                row.append((panel_expl(x, m_hard), f"{label}\n{cap}"))
        rows.append(row)
    if rows:
        image_grid(
            fig_dir / "FigA11_loo_qualitative.png",
            rows,
            suptitle="Fig A11 — same images across ablated terms (grey background)",
        )


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 1: hyperparameter ablation"))
    args = p.parse_args(argv)
    run_section1(args)


if __name__ == "__main__":
    main()
