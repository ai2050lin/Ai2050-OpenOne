#!/usr/bin/env python3
"""
Phase 674: Synthetic Value Readout Competitor Source Localization.

Focuses on the Phase 673 top entry point: DS7B same_format_random_value
readout_competitor_failure. The same synthetic-value controls are run across
qwen3, GLM4, and DS7B for comparison. No patching is used.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase584_gate_repair import load_model_flash  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402


CONTROL_PATH = Path(
    "results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.json"
)
OUT_ROOT = Path("results/glm5_phase674_synthetic_value_readout_competitor_source_localization")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def expected_variants(text: str) -> list[str]:
    return [text, " " + text, "\n" + text]


def encode_expected(tokenizer: Any, text: str) -> list[list[int]]:
    out = []
    seen = set()
    for variant in expected_variants(text):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids and tuple(ids) not in seen:
            seen.add(tuple(ids))
            out.append(ids)
    return out


def token_category(text: str, expected_ids: set[int], tid: int) -> str:
    s = text.strip()
    if tid in expected_ids:
        return "expected"
    if text == " " or (text.isspace() and "\n" not in text):
        return "space"
    if "\n" in text:
        return "newline"
    if s.startswith(("{", "[", '"')):
        return "json_or_quote"
    if s.startswith(("The", "the", "Record", "You", "I", "It", "This")):
        return "word_or_explanation"
    if s in {":", ".", ",", ";"}:
        return "punctuation"
    if not s:
        return "blank"
    return "other"


def capture_final_state(model, tokenizer, device, prompt: str) -> dict:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    final_norm = get_final_norm(model)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    if final_norm is not None:
        def pre_hook(_module, inputs):
            x = inputs[0]
            captured["final_norm_input"] = x[0, -1].detach().float().cpu()
            return None

        def out_hook(_module, _inputs, output):
            y = extract_tensor(output)
            captured["final_norm_output"] = y[0, -1].detach().float().cpu()
            return None

        handles.append(final_norm.register_forward_pre_hook(pre_hook))
        handles.append(final_norm.register_forward_hook(out_hook))
    try:
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
        return {
            "logits": logits.detach().cpu(),
            "final_norm_input": captured.get("final_norm_input"),
            "final_norm_output": captured.get("final_norm_output"),
        }
    finally:
        for h in handles:
            h.remove()


def logits_from_state(model, state: torch.Tensor | None) -> torch.Tensor | None:
    if state is None:
        return None
    emb = model.get_output_embeddings()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        return emb(state.to(device=device, dtype=dtype).unsqueeze(0)).squeeze(0).float().detach().cpu()


def best_expected(logits: torch.Tensor, expected_ids: set[int]) -> tuple[int, float, int]:
    best_id = max(expected_ids, key=lambda tid: float(logits[tid].item()))
    best_score = float(logits[best_id].item())
    rank = int((logits > logits[best_id]).sum().item()) + 1
    return int(best_id), best_score, rank


def top_competitor(tokenizer, logits: torch.Tensor, expected_ids: set[int], top_k: int) -> dict:
    vals, ids = torch.topk(logits, k=min(top_k, logits.numel()))
    top = []
    comp = None
    for rank, (score, tid_t) in enumerate(zip(vals.tolist(), ids.tolist()), start=1):
        tid = int(tid_t)
        text = tokenizer.decode([tid])
        cat = token_category(text, expected_ids, tid)
        item = {"rank": rank, "id": tid, "text": text, "score": float(score), "category": cat}
        top.append(item)
        if cat != "expected" and comp is None:
            comp = item
    return {"top": top[:10], "competitor": comp or top[0]}


def projection_diag(state: torch.Tensor, unembed: torch.Tensor, expected_id: int, competitor_id: int, logits: torch.Tensor) -> dict:
    h = state.float().cpu()
    we = unembed[expected_id].float().cpu()
    wc = unembed[competitor_id].float().cpu()
    e_logit = float(logits[expected_id].item())
    c_logit = float(logits[competitor_id].item())
    h_norm = float(h.norm().item())
    e_norm = float(we.norm().item())
    c_norm = float(wc.norm().item())
    e_cos = float(F.cosine_similarity(h, we, dim=0).item())
    c_cos = float(F.cosine_similarity(h, wc, dim=0).item())
    e_unit_score = float(torch.dot(h, we / max(e_norm, 1e-8)).item())
    c_unit_score = float(torch.dot(h, wc / max(c_norm, 1e-8)).item())
    actual_gap = c_logit - e_logit
    unit_gap = c_unit_score - e_unit_score
    if actual_gap <= 0:
        source = "expected_wins"
    elif unit_gap > 0:
        source = "direction_alignment"
    elif c_norm > e_norm:
        source = "projection_norm_advantage"
    else:
        source = "bias_or_other"
    return {
        "expected_logit": e_logit,
        "competitor_logit": c_logit,
        "competitor_minus_expected": actual_gap,
        "state_norm": h_norm,
        "expected_weight_norm": e_norm,
        "competitor_weight_norm": c_norm,
        "competitor_norm_advantage": c_norm - e_norm,
        "expected_cos": e_cos,
        "competitor_cos": c_cos,
        "competitor_cos_advantage": c_cos - e_cos,
        "expected_unit_score": e_unit_score,
        "competitor_unit_score": c_unit_score,
        "unit_gap": unit_gap,
        "diagnosed_source": source,
    }


def select_cases(max_cases: int) -> list[dict]:
    data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if c["family"] == "same_format_random_value"]
    return cases[:max_cases] if max_cases > 0 else cases


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "expected_top1": 0,
        "rank_sum": 0.0,
        "margin_sum": 0.0,
        "post_gap_sum": 0.0,
        "pre_gap_sum": 0.0,
        "norm_shift_sum": 0.0,
        "top1_category": {},
        "diagnosed_source": {},
        "top1_text": {},
    })
    for row in rows:
        for key in ["overall", row["relation"], row["top1_category"]]:
            g = groups[key]
            g["n"] += 1
            g["expected_top1"] += int(row["expected_rank"] == 1)
            g["rank_sum"] += row["expected_rank"]
            g["margin_sum"] += row["expected_minus_competitor"]
            g["post_gap_sum"] += row["post_diag"]["competitor_minus_expected"]
            if row.get("pre_diag"):
                g["pre_gap_sum"] += row["pre_diag"]["competitor_minus_expected"]
                g["norm_shift_sum"] += row["post_diag"]["competitor_minus_expected"] - row["pre_diag"]["competitor_minus_expected"]
            cat = row["top1_category"]
            src = row["post_diag"]["diagnosed_source"]
            text = row["top1_text"].replace("\n", "\\n")
            g["top1_category"][cat] = g["top1_category"].get(cat, 0) + 1
            g["diagnosed_source"][src] = g["diagnosed_source"].get(src, 0) + 1
            g["top1_text"][text] = g["top1_text"].get(text, 0) + 1
    out = {}
    for key, g in groups.items():
        n = max(1, g["n"])
        out[key] = {
            "n": g["n"],
            "expected_top1_rate": g["expected_top1"] / n,
            "mean_expected_rank": g["rank_sum"] / n,
            "mean_expected_minus_competitor": g["margin_sum"] / n,
            "mean_post_competitor_gap": g["post_gap_sum"] / n,
            "mean_pre_competitor_gap": g["pre_gap_sum"] / n,
            "mean_final_norm_gap_shift": g["norm_shift_sum"] / n,
            "top1_category": dict(sorted(g["top1_category"].items(), key=lambda kv: kv[1], reverse=True)),
            "diagnosed_source": dict(sorted(g["diagnosed_source"].items(), key=lambda kv: kv[1], reverse=True)),
            "top1_text": dict(sorted(g["top1_text"].items(), key=lambda kv: kv[1], reverse=True)[:10]),
        }
    return out


def run_model(args) -> dict:
    cases = select_cases(args.max_cases)
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        unembed = model.get_output_embeddings().weight.detach().float().cpu()
        for i, case in enumerate(cases):
            expected_seqs = encode_expected(tokenizer, case["expected_output"])
            expected_ids = {seq[0] for seq in expected_seqs if seq}
            probe = capture_final_state(model, tokenizer, device, case["prompt"])
            logits = probe["logits"]
            expected_id, expected_score, expected_rank = best_expected(logits, expected_ids)
            comp = top_competitor(tokenizer, logits, expected_ids, args.top_k)["competitor"]
            raw_top1_id = int(torch.argmax(logits).item())
            raw_top1_text = tokenizer.decode([raw_top1_id])
            raw_top1_category = token_category(raw_top1_text, expected_ids, raw_top1_id)
            post_state = probe["final_norm_output"]
            pre_state = probe["final_norm_input"]
            post_diag = projection_diag(post_state, unembed, expected_id, comp["id"], logits)
            pre_logits = logits_from_state(model, pre_state)
            pre_diag = projection_diag(pre_state, unembed, expected_id, comp["id"], pre_logits) if pre_logits is not None else None
            row = {
                "case_id": case["case_id"],
                "object_name": case["object_name"],
                "relation": case["relation"],
                "expected_output": case["expected_output"],
                "expected_id": expected_id,
                "expected_text": tokenizer.decode([expected_id]),
                "expected_rank": expected_rank,
                "top1_id": raw_top1_id,
                "top1_text": raw_top1_text,
                "top1_category": raw_top1_category,
                "competitor": comp,
                "competitor_is_top1": comp["id"] == raw_top1_id,
                "expected_minus_competitor": expected_score - comp["score"],
                "post_diag": post_diag,
                "pre_diag": pre_diag,
            }
            rows.append(row)
            if (i + 1) % 12 == 0 or i + 1 == len(cases):
                log(f"{args.model}: {i + 1}/{len(cases)} random-value cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "phase": 674,
        "title": "Synthetic Value Readout Competitor Source Localization",
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(rows),
        "source_control": str(CONTROL_PATH),
        "summary": summarize(rows),
        "rows": rows,
    }


def write_model(payload: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = payload["model"]
    full = OUT_ROOT / f"phase674_{model}_synthetic_value_readout_source_confirm.json"
    rows = OUT_ROOT / f"phase674_{model}_synthetic_value_readout_source_rows.jsonl"
    summary = OUT_ROOT / f"phase674_{model}_synthetic_value_readout_source_summary.json"
    full.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with rows.open("w", encoding="utf-8") as f:
        for row in payload["rows"]:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    slim = {k: v for k, v in payload.items() if k != "rows"}
    summary.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {full}")


def write_cross_summary() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase674_*_synthetic_value_readout_source_summary.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        models.append({"model": d["model"], "n_cases": d["n_cases"], "overall": d["summary"]["overall"]})
    payload = {
        "phase": 674,
        "title": "Synthetic Value Readout Competitor Source Localization Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    lines = [
        "# Phase 674 Synthetic Value Readout Competitor Source Localization",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | cases | top1_rate | mean_rank | expected_minus_comp | norm_shift | top1_category | source_diag |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for m in models:
        o = m["overall"]
        lines.append(
            f"| {m['model']} | {m['n_cases']} | {o['expected_top1_rate']:.3f} | "
            f"{o['mean_expected_rank']:.2f} | {o['mean_expected_minus_competitor']:.3f} | "
            f"{o['mean_final_norm_gap_shift']:.3f} | {o['top1_category']} | {o['diagnosed_source']} |"
        )
    (OUT_ROOT / "phase674_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "phase674_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--max-cases", type=int, default=72)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.summarize_only:
        payload = write_cross_summary()
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is set")
    payload = run_model(args)
    write_model(payload)
    print(json.dumps({k: payload[k] for k in ["model", "n_cases", "summary"]}, ensure_ascii=False, indent=2), flush=True)
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
