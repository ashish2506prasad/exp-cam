"""Section 6 — Exploratory token-mask ablation on a small causal LM.

Illustrative only: 10–15 prompts, no baseline comparison. Frame in the paper
as a short 'Extension to Text' note, not a validated text-explanation method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from cli import device_from_flag
from config import SEED, TEMPERATURE, add_where_args
from io_utils import ensure_dir, rows_to_excel, save_json
from losses import hard_st


PROMPTS = [
    ("The movie was surprisingly well acted but the script was weak. Sentiment:", "Negative"),
    ("I loved every minute of this delightful film. Sentiment:", "Positive"),
    ("The food arrived cold and the waiter ignored us. Sentiment:", "Negative"),
    ("A warm, generous performance that lifted the whole play. Sentiment:", "Positive"),
    ("This gadget broke on the second day. Sentiment:", "Negative"),
    ("Battery life exceeded every claim in the manual. Sentiment:", "Positive"),
    ("The plot twists felt cheap and unearned. Sentiment:", "Negative"),
    ("Quietly moving and beautifully shot. Sentiment:", "Positive"),
    ("Customer support hung up twice. Sentiment:", "Negative"),
    ("The hotel room was spotless and the staff kind. Sentiment:", "Positive"),
    ("I would not recommend this to anyone. Sentiment:", "Negative"),
    ("Best concert I have attended this year. Sentiment:", "Positive"),
]


def contig_loss(m: torch.Tensor) -> torch.Tensor:
    return (m[1:] - m[:-1]).abs().sum()


def train_token_mask(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    steps: int,
    lr: float,
    lambdas: dict,
) -> dict:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    t = input_ids.shape[1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        out0 = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        logits0 = out0.logits[:, -1, :]
        y = logits0.argmax(dim=-1)
        hidden0 = out0.hidden_states

    z = torch.nn.Parameter(torch.zeros(t, device=device))
    opt = torch.optim.Adam([z], lr=lr)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.eos_token_id

    for _ in range(steps):
        opt.zero_grad()
        m_prob = torch.sigmoid(z)
        m = hard_st(m_prob, 0.5)
        keep = m > 0.5
        # drop tokens from the attention mask (cleaner than a placeholder token)
        attn_m = attn * keep.float().unsqueeze(0)
        # keep at least the last token so next-token logits remain defined
        attn_m[0, -1] = 1
        out = model(input_ids=input_ids, attention_mask=attn_m, output_hidden_states=True)
        logits_e = out.logits[:, -1, :]
        hidden_e = out.hidden_states

        loss_ce = F.cross_entropy(logits_e, y)
        p0 = F.softmax(logits0 / TEMPERATURE, dim=-1)
        log_pe = F.log_softmax(logits_e / TEMPERATURE, dim=-1)
        loss_kl = F.kl_div(log_pe, p0, reduction="batchmean") * (TEMPERATURE ** 2)

        # L_act: cosine on last-token hidden states, every 4th layer
        loss_act = logits_e.new_zeros(())
        n_h = 0
        for i, (h0, he) in enumerate(zip(hidden0, hidden_e)):
            if i % 4 != 0:
                continue
            a = F.normalize(h0[:, -1, :], dim=-1)
            b = F.normalize(he[:, -1, :], dim=-1)
            loss_act = loss_act + (1 - (a * b).sum())
            n_h += 1
        if n_h:
            loss_act = loss_act / n_h

        loss_area = m_prob.mean()
        loss_bin = (m_prob * (1 - m_prob)).mean()
        loss_contig = contig_loss(m_prob)

        # L_rob: substitute dropped tokens with tokens from an unrelated prompt
        other = tokenizer(PROMPTS[(hash(prompt) % (len(PROMPTS) - 1)) + 1][0], return_tensors="pt")
        other_ids = other["input_ids"].to(device)
        mixed = input_ids.clone()
        L = min(t, other_ids.shape[1])
        replace = ~keep
        replace[-1] = False
        mixed[0, :L] = torch.where(replace[:L], other_ids[0, :L], mixed[0, :L])
        logits_r = model(input_ids=mixed, attention_mask=attn).logits[:, -1, :]
        loss_rob = F.cross_entropy(logits_r, y)

        loss = (
            lambdas["act"] * loss_act
            + lambdas["ce"] * loss_ce
            + lambdas["kl"] * loss_kl
            + lambdas["area"] * loss_area
            + lambdas["bin"] * loss_bin
            + lambdas["contig"] * loss_contig
            + lambdas["rob"] * loss_rob
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        m_prob = torch.sigmoid(z)
        keep = (m_prob > 0.5)
        keep[-1] = True
        attn_m = attn * keep.float().unsqueeze(0)
        out = model(input_ids=input_ids, attention_mask=attn_m)
        logits_e = out.logits[:, -1, :]
        pred_e = int(logits_e.argmax(dim=-1)[0].item())
        pred_x = int(y[0].item())
        p0 = F.softmax(logits0, dim=-1)[0]
        pe = F.softmax(logits_e, dim=-1)[0]
        kl = float(F.kl_div(F.log_softmax(logits_e, dim=-1), F.softmax(logits0, dim=-1), reduction="batchmean").item())

    kept = [tok if bool(keep[i]) else None for i, tok in enumerate(tokens)]
    return {
        "prompt": prompt,
        "tokens": tokens,
        "keep": [bool(keep[i]) for i in range(t)],
        "kept_render": _render(tokens, keep),
        "pred_x_id": pred_x,
        "pred_e_id": pred_e,
        "pred_x": tokenizer.decode([pred_x]).strip(),
        "pred_e": tokenizer.decode([pred_e]).strip(),
        "conf_x": float(p0[pred_x].item()),
        "conf_e": float(pe[pred_x].item()),
        "top1_agreement": int(pred_x == pred_e),
        "kl_divergence": kl,
        "pct_tokens_kept": 100.0 * float(keep.float().mean().item()),
        "n_tokens": t,
    }


def _render(tokens, keep) -> str:
    parts = []
    for tok, k in zip(tokens, keep):
        piece = tok.replace("Ġ", " ").replace("▁", " ")
        if k:
            parts.append(piece)
        else:
            parts.append(f"[{piece}]")
    return "".join(parts)


def run_section6(args) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "Section 6 needs `transformers`. Install it, then rerun. "
            f"Original error: {e}"
        )

    device = device_from_flag(args.device)
    print(f"Loading {args.model} on {device} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    except Exception as e:
        raise SystemExit(
            f"Could not load {args.model}. Pass --model distilgpt2 for a local-small run.\n{e}"
        )

    lambdas = {
        "act": 1.0,
        "ce": 5.0,
        "kl": 1.0,
        "area": 2.0,
        "bin": 0.5,
        "contig": 1.0,
        "rob": 1.0,
    }
    prompts = PROMPTS[: args.n_prompts]
    rows = []
    out = Path(args.out) / "section6"
    ensure_dir(out)
    for i, (prompt, _hint) in enumerate(prompts):
        torch.manual_seed(args.seed + i)
        row = train_token_mask(
            model, tokenizer, prompt, device, steps=args.steps, lr=args.lr, lambdas=lambdas
        )
        row["id"] = f"{i:02d}"
        row["hint_label"] = _hint
        rows.append(row)
        save_json(out / f"example_{i:02d}.json", row)
        print(f"[{i}] keep={row['pct_tokens_kept']:.1f}%  {row['pred_x']} -> {row['pred_e']}  {row['kept_render'][:80]}")

    summary = {
        "model": args.model,
        "n": len(rows),
        "mean_pct_kept": float(sum(r["pct_tokens_kept"] for r in rows) / len(rows)),
        "top1_agreement_rate": float(sum(r["top1_agreement"] for r in rows) / len(rows)),
        "mean_kl": float(sum(r["kl_divergence"] for r in rows) / len(rows)),
        "scope": "exploratory ablation only; not a validated text-explanation method",
    }
    save_json(out / "summary.json", summary)
    table = [
        {
            "prompt": r["prompt"],
            "kept_tokens": r["kept_render"],
            "orig_pred": r["pred_x"],
            "masked_pred": r["pred_e"],
            "conf_orig": r["conf_x"],
            "conf_masked": r["conf_e"],
            "pct_kept": r["pct_tokens_kept"],
        }
        for r in rows
    ]
    rows_to_excel(out / "TableG1.xlsx", {"TableG1": table})
    # markdown-ish dump for the paper table
    md = ["| Prompt | Kept tokens (dropped in [brackets]) | Orig → Masked | conf orig→masked | % kept |",
          "|---|---|---|---|---|"]
    for r in table:
        md.append(
            f"| {r['prompt']} | {r['kept_tokens']} | {r['orig_pred']} → {r['masked_pred']} | "
            f"{r['conf_orig']:.2f}→{r['conf_masked']:.2f} | {r['pct_kept']:.1f} |"
        )
    (out / "TableG1.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description="Section 6: exploratory LLM token masks")
    add_where_args(p)
    p.add_argument("--model", default="distilgpt2", help="Causal LM id. Paper target: meta-llama/Llama-3.2-3B")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--n-prompts", type=int, default=12)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    import config as C

    C.apply_where(args.where, data_root=args.data_root, runs_dir=args.out)
    if args.out is None:
        args.out = C.RUNS_DIR
    run_section6(args)


if __name__ == "__main__":
    main()
