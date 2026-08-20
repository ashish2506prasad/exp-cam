"""Section 2 — Minimality quantification.

2A reuses the Section 1 λ_area sweep.
2B trains with an explicit mask-budget constraint (projection by default).
p* is the smallest budget *found by the optimizer*, not a provable global min.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from cli import add_common_args, resolve_run
from classifiers import load_classifier
from config import BUDGETS, DEFAULT_LAMBDAS, FIDELITY_TAUS
from data import load_tensor
from io_utils import load_json, mean_std, rows_to_excel, save_json
from metrics import min_sufficient_budget
from runner import maybe_train_and_eval, run_dir
from train import TrainConfig, image_seed, log
from viz import (
    boxplot,
    caption,
    fidelity_vs_budget,
    image_grid,
    overlay_two_families,
    panel_expl,
    panel_original,
)


def run_section2(args) -> None:
    eval_set, images, device = resolve_run(args)
    out_root = args.out
    section = "section2"
    budgets = BUDGETS if not args.quick else (0.50, 0.10, 0.02)
    constraint = args.constraint
    rows_2b = []

    log(
        f"Section 2  device={device}  images={len(images)}  budgets={list(budgets)}  "
        f"constraint={constraint}  steps={args.steps}"
    )
    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        for p in budgets:
            log(f"--- budget p={p:g} / {backbone} ---")
            for rec in images:
                cfg_name = f"budget_p{p:g}"
                out_dir = run_dir(out_root, section, backbone, cfg_name, rec["id"])
                lam = dict(DEFAULT_LAMBDAS)
                lam["area"] = 0.0
                cfg = TrainConfig(
                    lambdas=lam,
                    steps=args.steps,
                    seed=image_seed(rec["id"], backbone, args.seed),
                    constraint=constraint,
                    p_target=p,
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
                        "protocol": "budget",
                        "p_budget": p,
                        "p_gap": None,
                        "config": cfg_name,
                    },
                )
                row["p_budget"] = p
                row["p_gap"] = abs(p - row["p_actual"])
                save_json(out_dir / "metrics.json", row)
                rows_2b.append(row)

    # 2A: load λ_area sweep from section 1 if present
    rows_2a = []
    s1 = out_root / "section1" / "all_runs.json"
    if s1.exists():
        for r in load_json(s1):
            if r.get("protocol") == "sweep" and r.get("term") == "area":
                r = dict(r)
                r["p_budget"] = r["p_actual"]
                rows_2a.append(r)

    save_json(out_root / section / "budget_runs.json", rows_2b)
    save_json(out_root / section / "area_sweep_reuse.json", rows_2a)

    pstar_rows = []
    for backbone in args.backbones:
        for rec in images:
            subset = [r for r in rows_2b if r["backbone"] == backbone and r["image_id"] == rec["id"]]
            for tau in FIDELITY_TAUS:
                pstar = min_sufficient_budget(subset, tau)
                pstar_rows.append(
                    {
                        "image_id": rec["id"],
                        "wnid": rec["wnid"],
                        "class_name": rec["class_name"],
                        "backbone": backbone,
                        "tau": tau,
                        "p_star": pstar,
                        "note": "smallest mask budget at which the optimizer found a sufficiently faithful explanation; not a provable global minimum",
                    }
                )
    rows_to_excel(
        out_root / section / "section2_results.xlsx",
        {"budget": rows_2b, "area_sweep_2A": rows_2a, "p_star": pstar_rows},
    )
    save_json(out_root / section / "p_star.json", pstar_rows)
    _figures(eval_set, images, args, rows_2a, rows_2b, pstar_rows, device, budgets)


def _series_for(rows, backbone, p_values, key, p_field="p_budget"):
    means, stds = [], []
    for p in p_values:
        vals = [r[key] for r in rows if r["backbone"] == backbone and abs(r[p_field] - p) < 1e-9]
        s = mean_std(vals)
        means.append(s["mean"] if s["mean"] is not None else 0.0)
        stds.append(s["std"] if s["std"] is not None else 0.0)
    return means, stds


def _figures(eval_set, images, args, rows_2a, rows_2b, pstar_rows, device, budgets) -> None:
    fig_dir = args.out / "section2" / "figures"
    backbone = args.backbones[0]
    p_list = list(budgets)
    fid_keys = ("top1_agreement", "one_minus_normalized_kl")
    series = {k: _series_for(rows_2b, backbone, p_list, k) for k in fid_keys}
    pa_m, pa_s = _series_for(rows_2b, backbone, p_list, "p_actual")
    pstars = [r["p_star"] for r in pstar_rows if r["backbone"] == backbone and r["tau"] == 0.95 and r["p_star"] is not None]
    p_star_mean = float(sum(pstars) / len(pstars)) if pstars else None
    fidelity_vs_budget(
        fig_dir / "FigB1_minimality_curve.png",
        p_list,
        series,
        pa_m,
        pa_s,
        p_star_mean,
        f"Fig B1 — Fidelity vs budget ({backbone}). p* is optimizer-found, not a global min.",
    )

    if rows_2a:
        from collections import defaultdict as dd

        by_mult = dd(list)
        for r in rows_2a:
            if r["backbone"] == backbone:
                by_mult[float(r.get("mult", r.get("lambda_value", 0)))].append(r)
        fam_a = {k: ([], []) for k in fid_keys}
        p_a = []
        for mult in sorted(by_mult):
            grp = by_mult[mult]
            p_a.append(float(np.mean([g["p_actual"] for g in grp])))
            for k in fid_keys:
                s = mean_std([g[k] for g in grp])
                fam_a[k][0].append(s["mean"] or 0)
                fam_a[k][1].append(s["std"] or 0)
        overlay_two_families(
            fig_dir / "FigB2_2A_vs_2B.png",
            p_list,
            {k: _series_for(rows_2b, backbone, p_list, k) for k in fid_keys},
            fam_a,
            "2B budget constraint",
            "2A λ_area sweep (x = achieved p_actual)",
            "Fig B2 — 2A vs 2B fidelity trade-off",
            p_b=p_a,
        )
        save_json(
            args.out / "section2" / "figB2_2A_points.json",
            {"p_actual_mean": p_a, "family_2A": {k: {"mean": fam_a[k][0], "std": fam_a[k][1]} for k in fid_keys}},
        )

    groups = defaultdict(list)
    for r in pstar_rows:
        if r["backbone"] == backbone and r["tau"] == 0.95 and r["p_star"] is not None:
            groups[r["class_name"]].append(100.0 * r["p_star"])
    if groups:
        boxplot(
            fig_dir / "FigB4_pstar_per_class.png",
            groups,
            "p* (% pixels) at τ=0.95",
            "Fig B4 — per-class minimum sufficient mask size (optimizer-found)",
        )

    _fig_b5(eval_set, args, device, fig_dir, budgets)


def _fig_b5(eval_set, args, device, fig_dir, budgets) -> None:
    from runner import load_masks

    qids = eval_set["qualitative_ids"][:3]
    id_to_rec = {r["id"]: r for r in eval_set["images"]}
    backbone = args.backbones[0]
    bundle = load_classifier(backbone, device)
    show_p = [p for p in (0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005) if p in set(budgets)]
    rows = []
    for qid in qids:
        rec = id_to_rec[qid]
        x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
        row = [(panel_original(x), f"{rec['class_name']}\noriginal")]
        for p in show_p:
            d = run_dir(args.out, "section2", backbone, f"budget_p{p:g}", qid)
            if not (d / "masks.pt").exists():
                continue
            m, _ = load_masks(d, device)
            met = load_json(d / "metrics.json")
            cap = caption(met["p_actual_pct"], met["conf_e"], met["top1_agreement"])
            row.append((panel_expl(x, m), f"p={p*100:g}%\n{cap}"))
        rows.append(row)
    if rows:
        image_grid(fig_dir / "FigB5_budget_strip.png", rows, suptitle="Fig B5 — explanation vs mask budget")


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 2: minimality"))
    p.add_argument("--constraint", choices=("project", "lagrangian"), default="project")
    args = p.parse_args(argv)
    run_section2(args)


if __name__ == "__main__":
    main()
