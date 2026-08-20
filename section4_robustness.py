"""Section 4 — Robustness evaluation.

4a. Cross-class / same-class / Gaussian background substitution (post-hoc on Full masks).
4b. Retrain on geometric and color-jittered inputs; report fidelity retention.
Mask-stability IoU (Fig E2/E3/E5) is on hold and is not computed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

from .cli import add_common_args, resolve_run
from .classifiers import load_classifier, run_with_acts
from .config import AUG_BRIGHTNESS, AUG_ROT_DEG, AUG_TRANSLATE_FRAC, N_BG_DRAWS, DEFAULT_LAMBDAS
from .data import load_tensor, other_class_records, same_class_records
from .io_utils import mean_std, rows_to_excel, save_json
from .metrics import (
    class_confidence,
    confidence_delta,
    evaluate_explanation,
    robustness_accuracy,
    top1_agreement,
)
from .runner import load_masks, run_dir
from .train import TrainConfig, image_seed, log
from .viz import caption, grouped_bars, image_grid, panel_expl, panel_mask, panel_original, to_numpy_image


def _sample_bg(bundle, eval_set, rec, pool, rng, device):
    """Pick one other eval-set image and return (tensor, record)."""
    other = pool[int(rng.randint(0, len(pool)))]
    tensor = load_tensor(other, bundle.preprocess, device, root=Path(eval_set["root"]))
    return tensor, other


def run_section4(args) -> None:
    eval_set, images, device = resolve_run(args)
    out_root = args.out / "section4"
    n_draws = 2 if args.quick else args.n_draws
    rows_bg = []
    rows_aug = []
    rng = np.random.RandomState(args.seed)

    log(f"Section 4  device={device}  images={len(images)}  n_draws={n_draws}  steps={args.steps}")
    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        for rec in images:
            exp_d = run_dir(args.out, "section1", backbone, "loo_Full", rec["id"])
            if not (exp_d / "masks.pt").exists():
                log(f"[skip] 4a {rec['id']} {backbone}: need Section 1 Full masks")
                continue
            log(f"[robust] {backbone} img={rec['id']} {rec['class_name']}")
            x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
            m_hard, m_prob = load_masks(exp_d, device)
            with torch.no_grad():
                y = bundle.model(x).argmax(dim=1)

            # 4a. Post-hoc background swap on the *already trained* Full mask:
            #     e_tilde = m ⊙ x + (1-m) ⊙ r
            # r is (i) Gaussian noise, (ii) a same-class eval image, or
            # (iii) a *different-class* eval image (cross-class substitution).
            kinds = {
                "gaussian": [
                    (torch.randn_like(x), {"id": f"gauss_{i}", "wnid": None, "relpath": "N(0,1)", "class_name": "noise"})
                    for i in range(n_draws)
                ],
                "same_class": [],
                "cross_class": [],
            }
            same_pool = same_class_records(eval_set, rec)
            cross_pool = other_class_records(eval_set, rec["wnid"])
            for _ in range(n_draws):
                if same_pool:
                    kinds["same_class"].append(_sample_bg(bundle, eval_set, rec, same_pool, rng, device))
                if cross_pool:
                    kinds["cross_class"].append(_sample_bg(bundle, eval_set, rec, cross_pool, rng, device))

            for kind, draws in kinds.items():
                if not draws:
                    continue
                bgs = [t for t, _meta in draws]
                acc = robustness_accuracy(bundle.model, x, m_hard, bgs, y=y)
                m3 = m_hard.expand_as(x)
                confs = []
                distractors = []
                with torch.no_grad():
                    for r, meta in draws:
                        e_r = m3 * x + (1 - m3) * r
                        logits = bundle.model(e_r)
                        confs.append(class_confidence(logits, y))
                        distractors.append(
                            {
                                "id": meta.get("id"),
                                "wnid": meta.get("wnid"),
                                "class_name": meta.get("class_name"),
                                "relpath": meta.get("relpath"),
                            }
                        )
                row = {
                    "image_id": rec["id"],
                    "wnid": rec["wnid"],
                    "class_name": rec["class_name"],
                    "backbone": backbone,
                    "bg_type": kind,
                    "robustness_accuracy": acc,
                    "conf_e_mean": float(np.mean(confs)),
                    "n_draws": len(bgs),
                    "distractors": distractors,
                    "formula": "e_tilde = m * x + (1-m) * r",
                }
                rows_bg.append(row)
                save_json(out_root / "background" / backbone / rec["id"] / f"{kind}.json", row)

            # 4b. retrain on perturbed inputs
            h, w = x.shape[-2:]
            angle = AUG_ROT_DEG if rng.rand() > 0.5 else -AUG_ROT_DEG
            tx = int(AUG_TRANSLATE_FRAC * w) * (1 if rng.rand() > 0.5 else -1)
            ty = int(AUG_TRANSLATE_FRAC * h) * (1 if rng.rand() > 0.5 else -1)
            x_geo = TF.affine(x[0], angle=angle, translate=[tx, ty], scale=1.0, shear=0).unsqueeze(0)
            x_jit = TF.adjust_brightness(x[0], 1.0 + (AUG_BRIGHTNESS if rng.rand() > 0.5 else -AUG_BRIGHTNESS)).unsqueeze(0)

            # original explanation metrics as e_0
            e0_logits = run_with_acts(bundle, m_hard.expand_as(x) * x, require_grad=False)[1]

            for tag, x_k in (("geometric", x_geo), ("color_jitter", x_jit)):
                cfg = TrainConfig(
                    lambdas=dict(DEFAULT_LAMBDAS),
                    steps=args.steps,
                    seed=image_seed(rec["id"], backbone, args.seed) + (1 if tag == "geometric" else 2),
                    log_every=args.log_every,
                )
                rec_k = dict(rec)
                # train against the perturbed tensor by a tiny local wrapper: write temp via maybe_train? 
                # Direct train_mask call:
                from .train import train_mask
                from .metrics import evaluate_explanation as ev

                log(f"  retraining on {tag} ...")
                result = train_mask(bundle, x_k, cfg)
                met_k = ev(bundle, x_k, result.mask_hard, mask_prob=result.mask_prob)
                e_k = result.e
                logits_ek = run_with_acts(bundle, e_k, require_grad=False)[1]
                row = {
                    "image_id": rec["id"],
                    "wnid": rec["wnid"],
                    "class_name": rec["class_name"],
                    "backbone": backbone,
                    "aug": tag,
                    "top1_vs_e0": top1_agreement(e0_logits, logits_ek),
                    "conf_delta_vs_e0": confidence_delta(e0_logits, logits_ek),
                    "top1_on_perturbed": met_k["top1_agreement"],
                    "conf_e": met_k["conf_e"],
                    "p_actual": met_k["p_actual"],
                    "seconds": result.seconds,
                    "note": "mask-stability IoU on hold; not computed",
                }
                rows_aug.append(row)
                save_json(out_root / "aug" / backbone / rec["id"] / f"{tag}.json", row)
                torch.save(
                    {"mask_hard": result.mask_hard.cpu(), "x_k": x_k.cpu(), "angle": angle, "tx": tx, "ty": ty},
                    out_root / "aug" / backbone / rec["id"] / f"{tag}.pt",
                )

    save_json(out_root / "background_runs.json", rows_bg)
    save_json(out_root / "aug_runs.json", rows_aug)
    rows_to_excel(out_root / "section4_results.xlsx", {"background": rows_bg, "augmentation": rows_aug})
    _figures(eval_set, images, args, rows_bg, device)


def _figures(eval_set, images, args, rows_bg, device) -> None:
    fig_dir = args.out / "section4" / "figures"
    backbone = args.backbones[0]
    order = ["gaussian", "same_class", "cross_class"]
    means, stds = [], []
    for kind in order:
        vals = [100 * r["robustness_accuracy"] for r in rows_bg if r["backbone"] == backbone and r["bg_type"] == kind]
        s = mean_std(vals)
        means.append(s["mean"] or 0)
        stds.append(s["std"] or 0)
    grouped_bars(
        fig_dir / "FigE1_background_types.png",
        order,
        {"robustness acc. %": (means, stds)},
        "robustness accuracy %",
        f"Fig E1 — background substitution ({backbone}, mean±std)",
    )
    _fig_e4(eval_set, args, device, fig_dir)


def _fig_e4(eval_set, args, device, fig_dir) -> None:
    qids = eval_set["qualitative_ids"][:4]
    id_to_rec = {r["id"]: r for r in eval_set["images"]}
    backbone = args.backbones[0]
    bundle = load_classifier(backbone, device)
    rng = np.random.RandomState(0)
    rows = []
    for qid in qids:
        rec = id_to_rec[qid]
        x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_set["root"]))
        d = run_dir(args.out, "section1", backbone, "loo_Full", qid)
        if not (d / "masks.pt").exists():
            continue
        m, _ = load_masks(d, device)
        from .io_utils import load_json
        met = load_json(d / "metrics.json")
        cap0 = caption(met["p_actual_pct"], met["conf_e"], met["top1_agreement"])
        m3 = m.expand_as(x)
        gauss = torch.randn_like(x)
        same_pool = same_class_records(eval_set, rec)
        cross_pool = other_class_records(eval_set, rec["wnid"])
        same = load_tensor(same_pool[0], bundle.preprocess, device, root=Path(eval_set["root"])) if same_pool else gauss
        cross = load_tensor(cross_pool[0], bundle.preprocess, device, root=Path(eval_set["root"])) if cross_pool else gauss

        def expl_bg(r):
            e = m3 * x + (1 - m3) * r
            vis = to_numpy_image(torch.clamp(
                (e * x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                 + x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]),
                0, 1,
            ))
            with torch.no_grad():
                logits_x = bundle.model(x)
                logits_e = bundle.model(e)
            cap = caption(
                100 * float(m.mean()),
                class_confidence(logits_e, logits_x.argmax(dim=1)),
                top1_agreement(logits_x, logits_e),
            )
            return vis, cap

        g_img, g_cap = expl_bg(gauss)
        s_img, s_cap = expl_bg(same)
        c_img, c_cap = expl_bg(cross)
        rows.append(
            [
                (panel_original(x), f"{rec['class_name']}\noriginal"),
                (panel_mask(m), "mask"),
                (panel_expl(x, m), f"grey bg\n{cap0}"),
                (g_img, f"Gaussian bg\n{g_cap}"),
                (s_img, f"same-class bg\n{s_cap}"),
                (c_img, f"cross-class bg\n{c_cap}"),
            ]
        )
    if rows:
        image_grid(fig_dir / "FigE4_background_grid.png", rows, suptitle="Fig E4 — background substitution")


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 4: robustness"))
    p.add_argument("--n-draws", type=int, default=N_BG_DRAWS)
    args = p.parse_args(argv)
    run_section4(args)


if __name__ == "__main__":
    main()
