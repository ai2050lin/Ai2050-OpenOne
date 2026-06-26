#!/usr/bin/env python3
"""
Phase 684: Late Readout Prose Amplification Causal Audit.

Phase 683 showed that DS7B short_only failures are not prose-dominant at
L17-L22 or final_norm_input, but become prose-dominant at final logits.
This phase tests whether the failure can be changed by small readout-side
interventions, without searching for a new global gate.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    OUT_ROOT as PHASE683_OUT,
    VARIANTS,
    expected_first_ids,
    expected_for,
    prompt_for,
    route_diag,
    route_id_sets,
    route_scores,
    select_base_cases,
    value_phrase,
)


OUT_ROOT = Path("results/glm5_phase684_late_readout_prose_amplification_causal_audit")
TARGET_VARIANT = "short_only"
LOGIT_ALPHAS = [0.5, 1.0, 2.0]
HIDDEN_RATIOS = [0.02, 0.05, 0.10]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_variant(name: str) -> dict[str, Any]:
    for v in VARIANTS:
        if v["name"] == name:
            return v
    raise KeyError(name)


def load_phase683_short_failures(model_name: str, limit: int | None = None) -> list[dict[str, Any]]:
    path = PHASE683_OUT / f"phase683_{model_name}_prose_bias_rows.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("variant") == TARGET_VARIANT and not row.get("expected_top1"):
                rows.append(row)
    rows.sort(key=lambda r: (r.get("family", ""), r.get("case_id", "")))
    if limit is not None:
        rows = rows[:limit]
    return rows


def capture_readout_states(model, tokenizer, device, prompt: str) -> dict[str, torch.Tensor]:
    final_norm = get_final_norm(model)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    if final_norm is None:
        raise RuntimeError("final norm not found")

    def norm_pre(_module, inputs):
        captured["final_norm_input"] = inputs[0][0, -1].detach()

    def norm_out(_module, _inputs, output):
        y = extract_tensor(output)
        captured["final_norm_output"] = y[0, -1].detach()

    handles.append(final_norm.register_forward_pre_hook(norm_pre))
    handles.append(final_norm.register_forward_hook(norm_out))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        captured["logits"] = out.logits[0, -1].detach()
    finally:
        for h in handles:
            h.remove()
    return captured


def lm_logits(model, state: torch.Tensor) -> torch.Tensor:
    emb = model.get_output_embeddings()
    with torch.inference_mode():
        return emb(state.unsqueeze(0)).squeeze(0)


def norm_then_lm_logits(model, state: torch.Tensor) -> torch.Tensor:
    final_norm = get_final_norm(model)
    if final_norm is None:
        return lm_logits(model, state)
    with torch.inference_mode():
        normed = final_norm(state.view(1, 1, -1)).view(-1)
        return lm_logits(model, normed)


def mean_lm_direction(model, ids: set[int], device, dtype) -> torch.Tensor:
    valid = sorted(ids)
    if not valid:
        raise ValueError("empty id set")
    weight = model.get_output_embeddings().weight
    vec = weight[torch.tensor(valid, device=device)].mean(dim=0).to(dtype=torch.float32)
    vec = vec / (vec.norm() + 1e-8)
    return vec.to(device=device, dtype=dtype)


def best_expected_rank(logits: torch.Tensor, expected_ids: set[int]) -> tuple[int, int, bool]:
    valid = [tid for tid in expected_ids if 0 <= tid < logits.numel()]
    best_id = max(valid, key=lambda tid: float(logits[tid].item()))
    rank = int((logits > logits[best_id]).sum().item()) + 1
    return int(best_id), rank, rank == 1


def classify(logits: torch.Tensor, routes: dict[str, set[int]], target_route: str, expected_ids: set[int]) -> dict[str, Any]:
    logits_cpu = logits.detach().float().cpu()
    expected_id, expected_rank, expected_top1 = best_expected_rank(logits_cpu, expected_ids)
    diag = route_diag(route_scores(logits_cpu, routes), target_route)
    top1_id = int(torch.argmax(logits_cpu).item())
    return {
        "expected_id": expected_id,
        "expected_rank": expected_rank,
        "expected_top1": expected_top1,
        "top1_id": top1_id,
        "diag": {
            "target_margin": diag["target_margin"],
            "best_other_route": diag["best_other_route"],
            "prose_minus_value": diag["prose_minus_value"],
            "target_rank": diag["target_rank"],
        },
    }


def logit_patch(
    logits: torch.Tensor,
    routes: dict[str, set[int]],
    expected_ids: set[int],
    mode: str,
    alpha: float,
) -> torch.Tensor:
    out = logits.clone()
    scores = route_scores(out.detach().float().cpu(), routes)
    gap = max(1.0, float(scores["prose"] - scores["value"]))
    prose_ids = [tid for tid in routes["prose"] if 0 <= tid < out.numel()]
    value_ids = [tid for tid in (routes["value"] | expected_ids) if 0 <= tid < out.numel()]
    if mode in {"remove_prose", "remove_prose_add_value"}:
        out[torch.tensor(prose_ids, device=out.device)] -= alpha * gap
    if mode in {"add_value", "remove_prose_add_value"}:
        out[torch.tensor(value_ids, device=out.device)] += alpha * gap
    return out


def hidden_patch_logits(
    model,
    state: torch.Tensor,
    direction: torch.Tensor,
    ratio: float,
    site: str,
) -> torch.Tensor:
    scale = state.float().norm().to(state.device) * ratio
    patched = state + direction.to(dtype=state.dtype) * scale.to(dtype=state.dtype)
    if site == "final_norm_input":
        return norm_then_lm_logits(model, patched)
    if site == "final_norm_output":
        return lm_logits(model, patched)
    raise ValueError(site)


def run_model(args) -> dict[str, Any]:
    failures = load_phase683_short_failures(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    variant = get_variant(TARGET_VARIANT)
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        dtype = next(model.parameters()).dtype
        for idx, prev in enumerate(failures, 1):
            case = case_map[prev["case_id"]]
            prompt = prompt_for(case, variant)
            expected_text = expected_for(case, variant)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            captured = capture_readout_states(model, tokenizer, device, prompt)
            baseline = classify(captured["logits"], routes, variant["target_route"], expected_ids)
            value_dir = mean_lm_direction(model, routes["value"] | expected_ids, device, dtype)
            prose_dir = mean_lm_direction(model, routes["prose"], device, dtype)
            direction = value_dir - prose_dir
            direction = direction / (direction.float().norm() + 1e-8)

            row_base = {
                "case_id": case["case_id"],
                "family": case["family"],
                "object_name": case.get("object_name"),
                "relation": case.get("relation"),
                "value": value_phrase(case),
                "expected_text": expected_text,
                "phase683_expected_rank": prev["expected_rank"],
                "phase683_final_pmv": prev["features"]["final_prose_minus_value"],
                "baseline_expected_rank": baseline["expected_rank"],
                "baseline_expected_top1": baseline["expected_top1"],
                "baseline_final_pmv": baseline["diag"]["prose_minus_value"],
                "baseline_best_other_route": baseline["diag"]["best_other_route"],
            }

            for mode in ["remove_prose", "add_value", "remove_prose_add_value"]:
                for alpha in LOGIT_ALPHAS:
                    patched = logit_patch(captured["logits"], routes, expected_ids, mode, alpha)
                    diag = classify(patched, routes, variant["target_route"], expected_ids)
                    rows.append({
                        **row_base,
                        "intervention": "logit",
                        "mode": mode,
                        "site": "final_logits",
                        "alpha": alpha,
                        "ratio": None,
                        "patched_expected_rank": diag["expected_rank"],
                        "patched_expected_top1": diag["expected_top1"],
                        "patched_final_pmv": diag["diag"]["prose_minus_value"],
                        "patched_best_other_route": diag["diag"]["best_other_route"],
                    })

            for site, state_key in [
                ("final_norm_input", "final_norm_input"),
                ("final_norm_output", "final_norm_output"),
            ]:
                state = captured[state_key]
                for ratio in HIDDEN_RATIOS:
                    patched_logits = hidden_patch_logits(model, state, direction, ratio, site)
                    diag = classify(patched_logits, routes, variant["target_route"], expected_ids)
                    rows.append({
                        **row_base,
                        "intervention": "hidden_direction",
                        "mode": "add_value_minus_prose",
                        "site": site,
                        "alpha": None,
                        "ratio": ratio,
                        "patched_expected_rank": diag["expected_rank"],
                        "patched_expected_top1": diag["expected_top1"],
                        "patched_final_pmv": diag["diag"]["prose_minus_value"],
                        "patched_best_other_route": diag["diag"]["best_other_route"],
                    })

            if idx % args.log_every == 0 or idx == len(failures):
                log(f"{args.model}: {idx}/{len(failures)} short_only failures audited")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, failures, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase684_{args.model}_late_readout_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 684,
        "title": "Late Readout Prose Amplification Causal Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "target_variant": TARGET_VARIANT,
        "n_failure_cases": len(failures),
        "n_rows": len(rows),
        "summary": summary,
    }
    (OUT_ROOT / f"phase684_{args.model}_late_readout_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "baseline_top1_rate": sum(1 for r in rows if r["baseline_expected_top1"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_expected_top1"]) / n,
        "repair_rate": sum(1 for r in rows if (not r["baseline_expected_top1"]) and r["patched_expected_top1"]) / n,
        "mean_baseline_rank": sum(r["baseline_expected_rank"] for r in rows) / n,
        "mean_patched_rank": sum(r["patched_expected_rank"] for r in rows) / n,
        "mean_rank_delta": sum(r["baseline_expected_rank"] - r["patched_expected_rank"] for r in rows) / n,
        "mean_baseline_pmv": sum(r["baseline_final_pmv"] for r in rows) / n,
        "mean_patched_pmv": sum(r["patched_final_pmv"] for r in rows) / n,
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, failures: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        scale = f"a{r['alpha']}" if r["alpha"] is not None else f"r{r['ratio']}"
        grouped[(r["intervention"], r["site"], r["mode"], scale)].append(r)
    for key, vals in grouped.items():
        by_condition["|".join(key)] = summarize_group(vals)
    best = sorted(
        ((k, v) for k, v in by_condition.items()),
        key=lambda kv: (kv[1].get("repair_rate", 0.0), -kv[1].get("mean_patched_rank", 1e9)),
        reverse=True,
    )[:12]
    return {
        "model": model_name,
        "n_phase683_short_failures": len(failures),
        "baseline_failure_best_other_route": dict(Counter(r["final_diag"]["best_other_route"] for r in failures).most_common()),
        "by_condition": by_condition,
        "best_conditions": [{"condition": k, **v} for k, v in best],
    }


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase684_*_late_readout_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 684,
        "title": "Late Readout Prose Amplification Causal Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase684_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 684 Late Readout Prose Amplification Causal Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | short_failures | best_condition | repair_rate | patched_top1 | mean_rank_delta | patched_pmv | patched_best_other |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in models:
        best = item["summary"]["best_conditions"][0] if item["summary"]["best_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_failure_cases']} | {best.get('condition', '')} | "
            f"{best.get('repair_rate', 0.0):.3f} | {best.get('patched_top1_rate', 0.0):.3f} | "
            f"{best.get('mean_rank_delta', 0.0):.2f} | {best.get('mean_patched_pmv', 0.0):.3f} | "
            f"{best.get('patched_best_other_route', {})} |"
        )
    lines.extend(["", "## Best Conditions", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| condition | repair_rate | patched_top1 | mean_patched_rank | rank_delta | baseline_pmv | patched_pmv |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for b in item["summary"]["best_conditions"][:10]:
            lines.append(
                f"| {b['condition']} | {b['repair_rate']:.3f} | {b['patched_top1_rate']:.3f} | "
                f"{b['mean_patched_rank']:.2f} | {b['mean_rank_delta']:.2f} | "
                f"{b['mean_baseline_pmv']:.3f} | {b['mean_patched_pmv']:.3f} |"
            )
        lines.append("")
    (OUT_ROOT / "phase684_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
