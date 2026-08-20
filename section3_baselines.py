"""Section 3 — Fair baseline comparison at size-matched sparsity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .baselines import BASELINE_NAMES, compute_heatmap, otsu_binarize, topk_binarize
from .cli import add_common_args, resolve_run
from .classifiers import load_classifier
from .config import N_INSERT_SEEDS, N_RISE_MASKS
from .data import load_tensor
from .io_utils import ensure_dir, load_json, mean_std, rows_to_excel, save_json
from .metrics import evaluate_explanation, insertion_curve
from .runner import load_masks, run_dir
from .train import log
from .viz import caption, grouped_bars, image_grid, line_auc, panel_expl, panel_mask, panel_original


def _exp_dir(args, backbone, image_id) -> Path:
    return run_dir(args.out, "section1", backbone, "loo_Full", image_id)


def run_section3(args) -> None:
    eval_set, images, device = resolve_run(args)
    out_root = args.out / "section3"
    rows = []
    curves = defaultdict(list)

    methods = list(BASELINE_NAMES) if not args.quick else ["gradcam", "layercam"]
    rise_masks = 64 if args.quick else args.rise_masks

    log(f"Section 3  device={device}  images={len(images)}  methods={methods}")
    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        for rec in images:
            x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
            exp_d = _exp_dir(args, backbone, rec["id"])
            if not (exp_d / "masks.pt").exists():
                log(f"[skip] {rec['id']} {backbone}: run Section 1 Full model first")
                continue
            log(f"[baselines] {backbone} img={rec['id']} {rec['class_name']}")
            m_exp, p_exp = load_masks(exp_d, device)
            k = int(m_exp.sum().item())
            exp_metrics = load_json(exp_d / "metrics.json")

            # EXP-CAM insertion (rank by soft mask)
            xs, ys, auc = insertion_curve(bundle, x, p_exp[0, 0], binary_mask=False)
            rows.append(
                {
                    "image_id": rec["id"],
                    "wnid": rec["wnid"],
                    "class_name": rec["class_name"],
                    "backbone": backbone,
                    "method": "EXP-CAM",
                    "mode": "size_matched",
                    "p_actual": exp_metrics["p_actual"],
                    "conf_e": exp_metrics["conf_e"],
                    "top1_agreement": exp_metrics["top1_agreement"],
                    "insertion_auc": auc,
                }
            )
            curves[("EXP-CAM", backbone)].append((xs, ys, auc))

            heatmaps = {}
            for name in methods:
                if name == "attention" and not bundle.cam_is_vit:
                    continue
                cam = compute_heatmap(name, bundle, x, rise_masks=rise_masks)
                if cam is None:
                    continue
                log(f"  {name} done")
                heatmaps[name] = cam.detach()
                matched = topk_binarize(cam, k).view(1, 1, *cam.shape[-2:])
                own = otsu_binarize(cam).view(1, 1, *cam.shape[-2:])
                for mode, mask in (("size_matched", matched), ("own_threshold", own)):
                    met = evaluate_explanation(bundle, x, mask, mask_prob=cam.view(1, 1, *cam.shape))
                    xs_i, ys_i, auc_i = insertion_curve(bundle, x, cam, binary_mask=False)
                    rows.append(
                        {
                            "image_id": rec["id"],
                            "wnid": rec["wnid"],
                            "class_name": rec["class_name"],
                            "backbone": backbone,
                            "method": name,
                            "mode": mode,
                            "p_actual": met["p_actual"],
                            "conf_e": met["conf_e"],
                            "top1_agreement": met["top1_agreement"],
                            "insertion_auc": auc_i,
                        }
                    )
                    if mode == "size_matched":
                        curves[(name, backbone)].append((xs_i, ys_i, auc_i))
                dest = ensure_dir(out_root / backbone / rec["id"])
                save_json(dest / f"{name}.json", {"k": k, "cam_max": float(cam.max().item())})
                torch.save(
                    {"cam": cam.cpu(), "matched": matched.cpu(), "own": own.cpu()},
                    dest / f"{name}.pt",
                )

    save_json(out_root / "all_runs.json", rows)
    rows_to_excel(out_root / "section3_results.xlsx", {"baselines": rows})
    _figures(eval_set, images, args, rows, curves, device)


def _mean_curve(items):
    import numpy as np

    xs = items[0][0]
    ys = np.mean([it[1] for it in items], axis=0)
    auc = float(np.mean([it[2] for it in items]))
    return xs, ys, auc


def _figures(eval_set, images, args, rows, curves, device) -> None:
    fig_dir = args.out / "section3" / "figures"
    backbone = args.backbones[0]
    matched = [r for r in rows if r["mode"] == "size_matched" and r["backbone"] == backbone]
    methods = []
    for r in matched:
        if r["method"] not in methods:
            methods.append(r["method"])
    by_cls = sorted({r["class_name"] for r in matched})
    groups = {}
    for cls in by_cls:
        means, stds = [], []
        for m in methods:
            vals = [r["insertion_auc"] for r in matched if r["method"] == m and r["class_name"] == cls]
            s = mean_std(vals)
            means.append(s["mean"] or 0)
            stds.append(s["std"] or 0)
        groups[cls] = (means, stds)
    if groups:
        grouped_bars(
            fig_dir / "FigC1_insertion_auc.png",
            methods,
            groups,
            "insertion AUC",
            f"Fig C1 — size-matched insertion AUC ({backbone})",
        )

    curve_plot = {}
    for (name, bb), items in curves.items():
        if bb != backbone or not items:
            continue
        curve_plot[name] = _mean_curve(items)
    if curve_plot:
        line_auc(
            fig_dir / "FigC2_insertion.png",
            curve_plot,
            "% pixels inserted",
            "normalized confidence of original class",
            f"Fig C2 — insertion curves ({backbone})",
        )

    # Fig C3: own vs matched, confidence
    labels = methods
    groups = {}
    for mode in ("own_threshold", "size_matched"):
        means, stds = [], []
        for m in labels:
            vals = [
                r["conf_e"]
                for r in rows
                if r["backbone"] == backbone and r["method"] == m and r["mode"] == mode
            ]
            s = mean_std(vals)
            means.append(s["mean"] or 0)
            stds.append(s["std"] or 0)
        groups[mode] = (means, stds)
    grouped_bars(
        fig_dir / "FigC3_own_vs_matched.png",
        labels,
        groups,
        "confidence preservation",
        f"Fig C3 — own-threshold vs size-matched ({backbone})",
    )

    _fig_c4(eval_set, args, device, fig_dir)


def _fig_c4(eval_set, args, device, fig_dir) -> None:
    qids = eval_set["qualitative_ids"][:6]
    id_to_rec = {r["id"]: r for r in eval_set["images"]}
    backbone = args.backbones[0]
    bundle = load_classifier(backbone, device)
    method_cols = ["EXP-CAM", "gradcam", "gradcam++", "layercam"]
    if bundle.cam_is_vit:
        method_cols.append("attention")
    rows = []
    for qid in qids:
        rec = id_to_rec.get(qid)
        if rec is None:
            continue
        x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
        row = [(panel_original(x), f"{rec['class_name']}\noriginal")]
        exp_d = _exp_dir(args, backbone, qid)
        if not (exp_d / "masks.pt").exists():
            continue
        m_exp, _ = load_masks(exp_d, device)
        met = load_json(exp_d / "metrics.json")
        cap = caption(met["p_actual_pct"], met["conf_e"], met["top1_agreement"])
        row.append((panel_expl(x, m_exp), f"EXP-CAM\n{cap}"))
        for name in method_cols[1:]:
            pt = args.out / "section3" / backbone / qid / f"{name}.pt"
            js = args.out / "section3" / "all_runs.json"
            if not pt.exists():
                continue
            blob = torch.load(pt, map_location=device)
            mask = blob["matched"].to(device)
            # find metrics
            all_rows = load_json(js) if js.exists() else []
            hit = next(
                (
                    r
                    for r in all_rows
                    if r["image_id"] == qid and r["method"] == name and r["mode"] == "size_matched" and r["backbone"] == backbone
                ),
                None,
            )
            cap_m = caption(
                100 * (hit["p_actual"] if hit else float(mask.mean())),
                hit["conf_e"] if hit else 0,
                hit["top1_agreement"] if hit else 0,
            )
            row.append((panel_mask(mask), f"{name}\n{cap_m}"))
        rows.append(row)
    if rows:
        image_grid(fig_dir / "FigC4_fair_grid.png", rows, suptitle="Fig C4 — size-matched binary masks, same images")


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 3: fair baselines"))
    p.add_argument("--rise-masks", type=int, default=N_RISE_MASKS)
    args = p.parse_args(argv)
    run_section3(args)


if __name__ == "__main__":
    main()
