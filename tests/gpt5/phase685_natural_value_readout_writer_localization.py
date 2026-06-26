#!/usr/bin/env python3
"""
Phase 685: Natural Value-Readout Writer Localization.

Phase 684 showed that adding a value-minus-prose readout direction can repair
DS7B short_only failures. This phase asks where that direction appears
naturally by comparing paired prompts:
  short_only fails, terse_no_explain succeeds, same case.

No PCA or learned classifier is used. The only measurement is a direct
projection of component outputs onto the lm_head value-minus-prose direction.
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

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    OUT_ROOT as PHASE683_OUT,
    expected_first_ids,
    expected_for,
    prompt_for,
    route_diag,
    route_id_sets,
    route_scores,
    select_base_cases,
    value_phrase,
)


OUT_ROOT = Path("results/glm5_phase685_natural_value_readout_writer_localization")
SHORT_VARIANT = {"name": "short_only", "instruction": "Answer with only the value.", "target_route": "value", "expected_mode": "value"}
TERSE_VARIANT = {"name": "terse_no_explain", "instruction": "Return exactly the value. Do not explain.", "target_route": "value", "expected_mode": "value"}
COMPONENTS = ["layer_out", "attn_out", "mlp_out"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_phase683_rows(model_name: str) -> dict[str, dict[str, dict[str, Any]]]:
    path = PHASE683_OUT / f"phase683_{model_name}_prose_bias_rows.jsonl"
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_case[row["case_id"]][row["variant"]] = row
    return by_case


def select_paired_cases(model_name: str, limit: int | None = None) -> list[str]:
    by_case = load_phase683_rows(model_name)
    paired = []
    for case_id, variants in by_case.items():
        short = variants.get("short_only")
        terse = variants.get("terse_no_explain")
        if short and terse and (not short.get("expected_top1")) and terse.get("expected_top1"):
            paired.append(case_id)
    paired.sort()
    if limit is not None:
        paired = paired[:limit]
    return paired


def mean_lm_direction(model, ids: set[int], device, dtype) -> torch.Tensor:
    valid = sorted(ids)
    weight = model.get_output_embeddings().weight
    vec = weight[torch.tensor(valid, device=device)].mean(dim=0).float()
    vec = vec / (vec.norm() + 1e-8)
    return vec.to(device=device, dtype=dtype)


def value_minus_prose_direction(model, routes: dict[str, set[int]], expected_ids: set[int], device, dtype) -> torch.Tensor:
    value_dir = mean_lm_direction(model, routes["value"] | expected_ids, device, dtype)
    prose_dir = mean_lm_direction(model, routes["prose"], device, dtype)
    d = value_dir - prose_dir
    return d / (d.float().norm() + 1e-8)


def best_expected_rank(logits: torch.Tensor, expected_ids: set[int]) -> tuple[int, int, bool]:
    logits_cpu = logits.detach().float().cpu()
    valid = [tid for tid in expected_ids if 0 <= tid < logits_cpu.numel()]
    best_id = max(valid, key=lambda tid: float(logits_cpu[tid].item()))
    rank = int((logits_cpu > logits_cpu[best_id]).sum().item()) + 1
    return int(best_id), rank, rank == 1


def final_diag(logits: torch.Tensor, routes: dict[str, set[int]], target_route: str, expected_ids: set[int]) -> dict[str, Any]:
    logits_cpu = logits.detach().float().cpu()
    expected_id, expected_rank, expected_top1 = best_expected_rank(logits_cpu, expected_ids)
    diag = route_diag(route_scores(logits_cpu, routes), target_route)
    return {
        "expected_id": expected_id,
        "expected_rank": expected_rank,
        "expected_top1": expected_top1,
        "top1_id": int(torch.argmax(logits_cpu).item()),
        "prose_minus_value": diag["prose_minus_value"],
        "target_margin": diag["target_margin"],
        "best_other_route": diag["best_other_route"],
    }


def capture_component_outputs(model, tokenizer, device, prompt: str) -> tuple[torch.Tensor, dict[tuple[int, str], torch.Tensor]]:
    layers = get_layers(model)
    captured: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    def save_output(layer_idx: int, component: str):
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            captured[(layer_idx, component)] = y[0, -1].detach()
        return hook

    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(save_output(li, "layer_out")))
        attn = get_attn(layer)
        if attn is not None:
            handles.append(attn.register_forward_hook(save_output(li, "attn_out")))
        mlp = get_mlp(layer)
        if mlp is not None:
            handles.append(mlp.register_forward_hook(save_output(li, "mlp_out")))

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        logits = out.logits[0, -1].detach()
    finally:
        for h in handles:
            h.remove()
    return logits, captured


def projection(vec: torch.Tensor, direction: torch.Tensor) -> float:
    return float(torch.dot(vec.float(), direction.float()).detach().cpu().item())


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    pair_rows = []
    try:
        dtype = next(model.parameters()).dtype
        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype)
            captures = {}
            finals = {}
            for variant in [SHORT_VARIANT, TERSE_VARIANT]:
                prompt = prompt_for(case, variant)
                logits, comp = capture_component_outputs(model, tokenizer, device, prompt)
                captures[variant["name"]] = comp
                finals[variant["name"]] = final_diag(logits, routes, variant["target_route"], expected_ids)

            all_keys = sorted(set(captures["short_only"]) | set(captures["terse_no_explain"]))
            for li, component in all_keys:
                short_vec = captures["short_only"].get((li, component))
                terse_vec = captures["terse_no_explain"].get((li, component))
                if short_vec is None or terse_vec is None:
                    continue
                short_proj = projection(short_vec, direction)
                terse_proj = projection(terse_vec, direction)
                rows.append({
                    "case_id": case_id,
                    "family": case["family"],
                    "object_name": case.get("object_name"),
                    "relation": case.get("relation"),
                    "value": value_phrase(case),
                    "layer": li,
                    "component": component,
                    "short_proj": short_proj,
                    "terse_proj": terse_proj,
                    "delta_terse_minus_short": terse_proj - short_proj,
                    "short_rank": finals["short_only"]["expected_rank"],
                    "terse_rank": finals["terse_no_explain"]["expected_rank"],
                    "rank_delta": finals["short_only"]["expected_rank"] - finals["terse_no_explain"]["expected_rank"],
                    "short_pmv": finals["short_only"]["prose_minus_value"],
                    "terse_pmv": finals["terse_no_explain"]["prose_minus_value"],
                    "pmv_delta": finals["short_only"]["prose_minus_value"] - finals["terse_no_explain"]["prose_minus_value"],
                })
            pair_rows.append({
                "case_id": case_id,
                "family": case["family"],
                "short_final": finals["short_only"],
                "terse_final": finals["terse_no_explain"],
            })
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, rows, pair_rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase685_{args.model}_writer_projection_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / f"phase685_{args.model}_paired_final_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in pair_rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 685,
        "title": "Natural Value-Readout Writer Localization",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "n_paired_cases": len(paired_ids),
        "n_projection_rows": len(rows),
        "summary": summary,
    }
    (OUT_ROOT / f"phase685_{args.model}_writer_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "mean_short_proj": sum(r["short_proj"] for r in rows) / n,
        "mean_terse_proj": sum(r["terse_proj"] for r in rows) / n,
        "mean_delta": sum(r["delta_terse_minus_short"] for r in rows) / n,
        "positive_delta_rate": sum(1 for r in rows if r["delta_terse_minus_short"] > 0) / n,
        "mean_rank_delta": sum(r["rank_delta"] for r in rows) / n,
        "mean_pmv_delta": sum(r["pmv_delta"] for r in rows) / n,
    }


def summarize_model(model_name: str, paired_ids: list[str], rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_site = {}
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["layer"], r["component"])].append(r)
    for (layer, component), vals in grouped.items():
        by_site[f"L{layer}_{component}"] = {
            "layer": layer,
            "component": component,
            **summarize_rows(vals),
        }
    top_positive = sorted(
        by_site.values(),
        key=lambda x: (x.get("mean_delta", 0.0), x.get("positive_delta_rate", 0.0)),
        reverse=True,
    )[:20]
    top_negative = sorted(
        by_site.values(),
        key=lambda x: (x.get("mean_delta", 0.0), x.get("positive_delta_rate", 0.0)),
    )[:12]
    by_component = {}
    for comp in COMPONENTS:
        by_component[comp] = summarize_rows([r for r in rows if r["component"] == comp])
    final_short_ranks = [r["short_final"]["expected_rank"] for r in pair_rows]
    final_terse_ranks = [r["terse_final"]["expected_rank"] for r in pair_rows]
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "mean_short_rank": sum(final_short_ranks) / max(1, len(final_short_ranks)),
        "mean_terse_rank": sum(final_terse_ranks) / max(1, len(final_terse_ranks)),
        "mean_final_rank_delta": sum(a - b for a, b in zip(final_short_ranks, final_terse_ranks)) / max(1, len(final_short_ranks)),
        "by_component": by_component,
        "top_positive_sites": top_positive,
        "top_negative_sites": top_negative,
    }


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase685_*_writer_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 685,
        "title": "Natural Value-Readout Writer Localization Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase685_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 685 Natural Value-Readout Writer Localization",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | paired_cases | short_rank | terse_rank | rank_delta | top_site | top_delta | top_pos_rate |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for item in models:
        s = item["summary"]
        top = s["top_positive_sites"][0] if s["top_positive_sites"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | "
            f"{s['mean_short_rank']:.2f} | {s['mean_terse_rank']:.2f} | {s['mean_final_rank_delta']:.2f} | "
            f"L{top.get('layer', '')}_{top.get('component', '')} | {top.get('mean_delta', 0.0):.3f} | "
            f"{top.get('positive_delta_rate', 0.0):.3f} |"
        )
    lines.extend(["", "## Top Positive Sites", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| site | component | mean_delta | positive_rate | short_proj | terse_proj |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in item["summary"]["top_positive_sites"][:12]:
            lines.append(
                f"| L{row['layer']} | {row['component']} | {row['mean_delta']:.3f} | "
                f"{row['positive_delta_rate']:.3f} | {row['mean_short_proj']:.3f} | {row['mean_terse_proj']:.3f} |"
            )
        lines.append("")
    (OUT_ROOT / "phase685_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=24)
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
