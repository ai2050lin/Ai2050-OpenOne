#!/usr/bin/env python3
"""
Phase 676: Late Readout Competitor Causal Suppression Audit.

Tests whether the Phase 675 late attention/final-norm trajectory attribution has
causal bite. The intervention is intentionally simple: remove or cancel the
case-specific competitor-vs-expected readout direction at selected sites.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn, get_final_norm  # noqa: E402


PHASE674_ROOT = Path("results/glm5_phase674_synthetic_value_readout_competitor_source_localization")
CONTROL_PATH = Path("results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.json")
OUT_ROOT = Path("results/glm5_phase676_late_readout_competitor_causal_suppression_audit")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows(model_name: str, max_cases: int) -> list[dict]:
    path = PHASE674_ROOT / f"phase674_{model_name}_synthetic_value_readout_source_rows.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_cases] if max_cases > 0 else rows


def prompt_map() -> dict[str, str]:
    data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    return {case["case_id"]: case["prompt"] for case in data["cases"]}


def normalize(v: torch.Tensor) -> torch.Tensor:
    v = v.float()
    return v / max(float(v.norm().item()), 1e-8)


def random_unit_like(v: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    r = torch.randn(v.shape, generator=gen)
    return normalize(r)


def replace_tensor_output(output: Any, new_first: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_first,) + tuple(output[1:])
    return new_first


def remove_projection_at_pos(tensor: torch.Tensor, pos: int, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    out = tensor.clone()
    d = direction.to(device=out.device, dtype=torch.float32)
    h = out[0, pos, :].float()
    comp = torch.dot(h, d) * d
    out[0, pos, :] = (h - alpha * comp).to(dtype=out.dtype)
    return out


def cancel_gap_at_pos(tensor: torch.Tensor, pos: int, direction: torch.Tensor, gap: float, alpha: float) -> torch.Tensor:
    out = tensor.clone()
    d = direction.to(device=out.device, dtype=torch.float32)
    h = out[0, pos, :].float()
    denom = max(float(torch.dot(d, d).item()), 1e-8)
    out[0, pos, :] = (h - alpha * (gap / denom) * d).to(dtype=out.dtype)
    return out


def scale_at_pos(tensor: torch.Tensor, pos: int, alpha: float) -> torch.Tensor:
    out = tensor.clone()
    out[0, pos, :] = out[0, pos, :] * (1.0 - alpha)
    return out


def score_logits(logits: torch.Tensor, expected_id: int, competitor_id: int, tokenizer) -> dict:
    logits = logits.float().cpu()
    e = float(logits[expected_id].item())
    c = float(logits[competitor_id].item())
    top1_id = int(torch.argmax(logits).item())
    return {
        "expected_logit": e,
        "competitor_logit": c,
        "gap": c - e,
        "expected_rank": int((logits > logits[expected_id]).sum().item()) + 1,
        "top1_id": top1_id,
        "top1_text": tokenizer.decode([top1_id]),
        "expected_top1": top1_id == expected_id,
    }


def forward_with_condition(
    model,
    tokenizer,
    device,
    prompt: str,
    expected_id: int,
    competitor_id: int,
    condition: dict,
    readout_dir: torch.Tensor,
    random_dir: torch.Tensor,
    baseline_gap: float,
    target_layer: int,
    prev_layer: int,
) -> dict:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos = len(ids) - 1
    input_ids = torch.tensor([ids], device=device)
    handles = []
    layers = get_layers(model)
    final_norm = get_final_norm(model)

    site = condition["site"]
    mode = condition["mode"]
    alpha = float(condition.get("alpha", 1.0))
    direction = random_dir if condition.get("direction") == "random" else readout_dir

    def apply_patch(tensor: torch.Tensor) -> torch.Tensor:
        if mode == "remove_projection":
            return remove_projection_at_pos(tensor, pos, direction, alpha)
        if mode == "cancel_gap":
            return cancel_gap_at_pos(tensor, pos, readout_dir, baseline_gap, alpha)
        if mode == "scale_zero":
            return scale_at_pos(tensor, pos, alpha)
        return tensor

    if site == "final_norm_output":
        if final_norm is None:
            raise RuntimeError("final norm not found")

        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            return replace_tensor_output(output, apply_patch(y))

        handles.append(final_norm.register_forward_hook(hook))
    elif site == "final_norm_input":
        if final_norm is None:
            raise RuntimeError("final norm not found")

        def pre_hook(_module, inputs):
            x = inputs[0]
            x_new = apply_patch(x)
            return (x_new,) + tuple(inputs[1:])

        handles.append(final_norm.register_forward_pre_hook(pre_hook))
    elif site in {"attn_output_last", "attn_output_prev"}:
        li = target_layer if site == "attn_output_last" else prev_layer
        attn = get_attn(layers[li])
        if attn is None:
            raise RuntimeError(f"attention module not found at layer {li}")

        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            return replace_tensor_output(output, apply_patch(y))

        handles.append(attn.register_forward_hook(hook))
    elif site == "none":
        pass
    else:
        raise RuntimeError(f"unknown site: {site}")

    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, return_dict=True)
        logits = out.logits[0, pos]
        return score_logits(logits, expected_id, competitor_id, tokenizer)
    finally:
        for h in handles:
            h.remove()


def conditions() -> list[dict]:
    return [
        {"name": "baseline", "site": "none", "mode": "none"},
        {"name": "final_output_remove_comp_a1", "site": "final_norm_output", "mode": "remove_projection", "direction": "comp", "alpha": 1.0},
        {"name": "final_output_remove_random_a1", "site": "final_norm_output", "mode": "remove_projection", "direction": "random", "alpha": 1.0},
        {"name": "final_output_cancel_gap_a1", "site": "final_norm_output", "mode": "cancel_gap", "direction": "comp", "alpha": 1.0},
        {"name": "final_input_remove_comp_a1", "site": "final_norm_input", "mode": "remove_projection", "direction": "comp", "alpha": 1.0},
        {"name": "attn_last_remove_comp_a1", "site": "attn_output_last", "mode": "remove_projection", "direction": "comp", "alpha": 1.0},
        {"name": "attn_prev_remove_comp_a1", "site": "attn_output_prev", "mode": "remove_projection", "direction": "comp", "alpha": 1.0},
        {"name": "attn_last_zero_a1", "site": "attn_output_last", "mode": "scale_zero", "direction": "comp", "alpha": 1.0},
    ]


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(lambda: {
        "n": 0,
        "top1": 0,
        "rank_sum": 0.0,
        "gap_sum": 0.0,
        "gap_delta_sum": 0.0,
        "switch_to_expected": 0,
        "damage_success": 0,
        "top1_text": {},
    })
    baseline_by_case = {r["case_id"]: r for r in rows if r["condition"] == "baseline"}
    for row in rows:
        base = baseline_by_case[row["case_id"]]
        base_success = base["expected_top1"]
        base_failure = not base_success
        for scope in ["overall", row["base_top1_category"], row["relation"]]:
            key = f"{scope}|{row['condition']}"
            g = groups[key]
            g["n"] += 1
            g["top1"] += int(row["expected_top1"])
            g["rank_sum"] += row["expected_rank"]
            g["gap_sum"] += row["gap"]
            g["gap_delta_sum"] += row["gap"] - base["gap"]
            g["switch_to_expected"] += int(base_failure and row["expected_top1"])
            g["damage_success"] += int(base_success and not row["expected_top1"])
            text = row["top1_text"].replace("\n", "\\n")
            g["top1_text"][text] = g["top1_text"].get(text, 0) + 1
    out = {}
    for key, g in groups.items():
        scope, condition = key.split("|", 1)
        n = max(1, g["n"])
        out[key] = {
            "scope": scope,
            "condition": condition,
            "n": g["n"],
            "expected_top1_rate": g["top1"] / n,
            "mean_expected_rank": g["rank_sum"] / n,
            "mean_gap": g["gap_sum"] / n,
            "mean_gap_delta_vs_baseline": g["gap_delta_sum"] / n,
            "switch_to_expected_rate": g["switch_to_expected"] / n,
            "damage_success_rate": g["damage_success"] / n,
            "top1_text": dict(sorted(g["top1_text"].items(), key=lambda kv: kv[1], reverse=True)[:8]),
        }
    return out


def run_model(args) -> dict:
    rows674 = load_rows(args.model, args.max_cases)
    prompts = prompt_map()
    model, tokenizer, device = load_model_flash(args.model)
    out_rows = []
    try:
        layers = get_layers(model)
        target_layer = len(layers) - 1
        prev_layer = max(0, target_layer - 1)
        unembed = model.get_output_embeddings().weight.detach().float().cpu()
        conds = conditions()
        for i, row674 in enumerate(rows674):
            case_id = row674["case_id"]
            expected_id = int(row674["expected_id"])
            competitor_id = int(row674["competitor"]["id"])
            readout_dir = normalize(unembed[competitor_id] - unembed[expected_id])
            rand_dir = random_unit_like(readout_dir, seed=676000 + i)
            prompt = prompts[case_id]

            baseline = forward_with_condition(
                model, tokenizer, device, prompt, expected_id, competitor_id,
                conds[0], readout_dir, rand_dir, 0.0, target_layer, prev_layer
            )
            baseline_gap = baseline["gap"]
            for cond in conds:
                stats = baseline if cond["name"] == "baseline" else forward_with_condition(
                    model, tokenizer, device, prompt, expected_id, competitor_id,
                    cond, readout_dir, rand_dir, baseline_gap, target_layer, prev_layer
                )
                out_rows.append({
                    "case_id": case_id,
                    "relation": row674["relation"],
                    "base_top1_category": row674["top1_category"],
                    "base_expected_rank": row674["expected_rank"],
                    "expected_id": expected_id,
                    "competitor_id": competitor_id,
                    "competitor_text": row674["competitor"]["text"],
                    "condition": cond["name"],
                    **stats,
                })
            if (i + 1) % 12 == 0 or i + 1 == len(rows674):
                log(f"{args.model}: {i + 1}/{len(rows674)} cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(out_rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_ROOT / f"phase676_{args.model}_causal_suppression_rows.jsonl"
    rows_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in out_rows) + "\n",
        encoding="utf-8",
    )
    result = {
        "phase": 676,
        "title": "Late Readout Competitor Causal Suppression Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "n_cases": len(rows674),
        "n_rows": len(out_rows),
        "conditions": [c["name"] for c in conditions()],
        "summary": summary,
    }
    out_path = OUT_ROOT / f"phase676_{args.model}_causal_suppression_summary.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    log(f"Wrote {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def write_cross_summary() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase676_*_causal_suppression_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    result = {
        "phase": 676,
        "title": "Late Readout Competitor Causal Suppression Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase676_cross_model_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    preferred = [
        "baseline",
        "final_output_remove_comp_a1",
        "final_output_remove_random_a1",
        "final_output_cancel_gap_a1",
        "final_input_remove_comp_a1",
        "attn_last_remove_comp_a1",
        "attn_prev_remove_comp_a1",
        "attn_last_zero_a1",
    ]
    lines = [
        "# Phase 676 Late Readout Competitor Causal Suppression",
        "",
        f"- generated: `{result['timestamp']}`",
        "",
        "| model | condition | top1_rate | mean_rank | mean_gap | gap_delta | switch_to_expected | damage_success |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        summary = item["summary"]
        for cond in preferred:
            row = summary.get(f"overall|{cond}")
            if not row:
                continue
            lines.append(
                f"| {item['model']} | {cond} | {row['expected_top1_rate']:.3f} | "
                f"{row['mean_expected_rank']:.2f} | {row['mean_gap']:.3f} | "
                f"{row['mean_gap_delta_vs_baseline']:.3f} | {row['switch_to_expected_rate']:.3f} | "
                f"{row['damage_success_rate']:.3f} |"
            )
    (OUT_ROOT / "phase676_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--max-cases", type=int, default=72)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
