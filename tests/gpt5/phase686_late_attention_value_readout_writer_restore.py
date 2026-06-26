#!/usr/bin/env python3
"""
Phase 686: Late Attention Value-Readout Writer Restore.

Use Phase 685 top writer candidates. For paired cases where short_only fails
and terse_no_explain succeeds, patch the short prompt with the corresponding
terse component output at the final readout position.
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

from model_utils import get_layers, release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    expected_first_ids,
    expected_for,
    prompt_for,
    route_diag,
    route_id_sets,
    route_scores,
    select_base_cases,
    value_phrase,
)
from phase685_natural_value_readout_writer_localization import (  # noqa: E402
    OUT_ROOT as PHASE685_OUT,
    SHORT_VARIANT,
    TERSE_VARIANT,
    select_paired_cases,
)


OUT_ROOT = Path("results/glm5_phase686_late_attention_value_readout_writer_restore")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_candidate_sites(model_name: str, max_layer_sites: int = 2, max_attn_sites: int = 2) -> list[tuple[int, str]]:
    path = PHASE685_OUT / f"phase685_{model_name}_writer_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tops = data["summary"]["top_positive_sites"]
    sites = []
    for component, limit in [("attn_out", max_attn_sites), ("layer_out", max_layer_sites)]:
        picked = [(int(r["layer"]), r["component"]) for r in tops if r["component"] == component][:limit]
        sites.extend(picked)
    return sites


def get_module(model, layer_idx: int, component: str):
    layer = get_layers(model)[layer_idx]
    if component == "layer_out":
        return layer
    if component == "attn_out":
        return get_attn(layer)
    if component == "mlp_out":
        return get_mlp(layer)
    raise ValueError(component)


def capture_sites(model, tokenizer, device, prompt: str, sites: list[tuple[int, str]]) -> tuple[torch.Tensor, dict[tuple[int, str], torch.Tensor]]:
    captured: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    def save(site):
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            captured[site] = y[0, -1].detach()
        return hook

    for site in sites:
        module = get_module(model, *site)
        if module is not None:
            handles.append(module.register_forward_hook(save(site)))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        logits = out.logits[0, -1].detach()
    finally:
        for h in handles:
            h.remove()
    return logits, captured


def install_patch_hooks(
    model,
    sites: list[tuple[int, str]],
    short_states: dict[tuple[int, str], torch.Tensor],
    terse_states: dict[tuple[int, str], torch.Tensor],
    mode: str,
    seed: int,
):
    handles = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def make_patch(site):
        donor = terse_states[site]
        short = short_states[site]
        delta = donor - short
        if mode == "random_delta":
            noise = torch.randn(delta.shape, generator=gen, dtype=torch.float32)
            noise = noise / (noise.norm() + 1e-8) * float(delta.float().norm().detach().cpu().item())
            delta_to_add = noise.to(device=delta.device, dtype=delta.dtype)
        else:
            delta_to_add = delta

        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            if mode == "replace":
                y_new[0, -1] = donor.to(device=y_new.device, dtype=y_new.dtype)
            elif mode in {"add_delta", "random_delta"}:
                y_new[0, -1] = y_new[0, -1] + delta_to_add.to(device=y_new.device, dtype=y_new.dtype)
            else:
                raise ValueError(mode)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        return hook

    for site in sites:
        if site not in short_states or site not in terse_states:
            continue
        module = get_module(model, *site)
        if module is not None:
            handles.append(module.register_forward_hook(make_patch(site)))
    return handles


def best_expected_rank(logits: torch.Tensor, expected_ids: set[int]) -> tuple[int, int, bool]:
    logits_cpu = logits.detach().float().cpu()
    valid = [tid for tid in expected_ids if 0 <= tid < logits_cpu.numel()]
    best_id = max(valid, key=lambda tid: float(logits_cpu[tid].item()))
    rank = int((logits_cpu > logits_cpu[best_id]).sum().item()) + 1
    return int(best_id), rank, rank == 1


def classify(logits: torch.Tensor, routes: dict[str, set[int]], expected_ids: set[int]) -> dict[str, Any]:
    logits_cpu = logits.detach().float().cpu()
    expected_id, expected_rank, expected_top1 = best_expected_rank(logits_cpu, expected_ids)
    diag = route_diag(route_scores(logits_cpu, routes), "value")
    return {
        "expected_id": expected_id,
        "expected_rank": expected_rank,
        "expected_top1": expected_top1,
        "top1_id": int(torch.argmax(logits_cpu).item()),
        "prose_minus_value": diag["prose_minus_value"],
        "target_margin": diag["target_margin"],
        "best_other_route": diag["best_other_route"],
    }


def condition_specs(candidate_sites: list[tuple[int, str]]) -> list[dict[str, Any]]:
    specs = []
    for site in candidate_sites:
        specs.append({"name": f"L{site[0]}_{site[1]}", "sites": [site]})
    attn = [s for s in candidate_sites if s[1] == "attn_out"]
    layer = [s for s in candidate_sites if s[1] == "layer_out"]
    if len(attn) >= 2:
        specs.append({"name": "top2_attn_out", "sites": attn[:2]})
    if len(layer) >= 2:
        specs.append({"name": "top2_layer_out", "sites": layer[:2]})
    if attn and layer:
        specs.append({"name": "best_attn_plus_best_layer", "sites": [attn[0], layer[0]]})
    return specs


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    candidate_sites = load_candidate_sites(args.model, args.max_layer_sites, args.max_attn_sites)
    all_sites = sorted(set(candidate_sites))
    case_map = {c["case_id"]: c for c in select_base_cases()}
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        specs = condition_specs(candidate_sites)
        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            short_prompt = prompt_for(case, SHORT_VARIANT)
            terse_prompt = prompt_for(case, TERSE_VARIANT)
            short_logits, short_states = capture_sites(model, tokenizer, device, short_prompt, all_sites)
            terse_logits, terse_states = capture_sites(model, tokenizer, device, terse_prompt, all_sites)
            short_diag = classify(short_logits, routes, expected_ids)
            terse_diag = classify(terse_logits, routes, expected_ids)
            ids = tokenizer.encode(short_prompt, add_special_tokens=False)
            for spec in specs:
                for mode in ["add_delta", "replace", "random_delta"]:
                    handles = install_patch_hooks(
                        model, spec["sites"], short_states, terse_states, mode, seed=idx * 1009 + len(spec["sites"])
                    )
                    try:
                        with torch.inference_mode():
                            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
                        patched_diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
                    finally:
                        for h in handles:
                            h.remove()
                    rows.append({
                        "case_id": case_id,
                        "family": case["family"],
                        "value": value_phrase(case),
                        "condition": spec["name"],
                        "sites": [f"L{li}_{comp}" for li, comp in spec["sites"]],
                        "mode": mode,
                        "short_rank": short_diag["expected_rank"],
                        "terse_rank": terse_diag["expected_rank"],
                        "patched_rank": patched_diag["expected_rank"],
                        "short_top1": short_diag["expected_top1"],
                        "terse_top1": terse_diag["expected_top1"],
                        "patched_top1": patched_diag["expected_top1"],
                        "short_pmv": short_diag["prose_minus_value"],
                        "terse_pmv": terse_diag["prose_minus_value"],
                        "patched_pmv": patched_diag["prose_minus_value"],
                        "patched_best_other_route": patched_diag["best_other_route"],
                    })
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: {idx}/{len(paired_ids)} paired cases patched")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, candidate_sites, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase686_{args.model}_writer_restore_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 686,
        "title": "Late Attention Value-Readout Writer Restore",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "n_paired_cases": len(paired_ids),
        "candidate_sites": [f"L{li}_{comp}" for li, comp in candidate_sites],
        "n_rows": len(rows),
        "summary": summary,
    }
    (OUT_ROOT / f"phase686_{args.model}_writer_restore_summary.json").write_text(
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
        "repair_rate": sum(1 for r in rows if (not r["short_top1"]) and r["patched_top1"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "mean_short_rank": sum(r["short_rank"] for r in rows) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_terse_rank": sum(r["terse_rank"] for r in rows) / n,
        "mean_rank_delta": sum(r["short_rank"] - r["patched_rank"] for r in rows) / n,
        "mean_short_pmv": sum(r["short_pmv"] for r in rows) / n,
        "mean_patched_pmv": sum(r["patched_pmv"] for r in rows) / n,
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, paired_ids: list[str], candidate_sites: list[tuple[int, str]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["mode"])].append(r)
    for (condition, mode), vals in grouped.items():
        by_condition[f"{condition}|{mode}"] = summarize_group(vals)
    best = sorted(
        ((k, v) for k, v in by_condition.items()),
        key=lambda kv: (kv[1].get("repair_rate", 0.0), -kv[1].get("mean_patched_rank", 1e9)),
        reverse=True,
    )[:12]
    controls = {k: v for k, v in by_condition.items() if k.endswith("|random_delta")}
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "candidate_sites": [f"L{li}_{comp}" for li, comp in candidate_sites],
        "by_condition": by_condition,
        "best_conditions": [{"condition": k, **v} for k, v in best],
        "random_controls": controls,
    }


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase686_*_writer_restore_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 686,
        "title": "Late Attention Value-Readout Writer Restore Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase686_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 686 Late Attention Value-Readout Writer Restore",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | candidate_sites | best_condition | repair_rate | patched_top1 | rank_delta | patched_pmv | patched_best_other |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for item in models:
        best = item["summary"]["best_conditions"][0] if item["summary"]["best_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | {', '.join(item['candidate_sites'])} | "
            f"{best.get('condition', '')} | {best.get('repair_rate', 0.0):.3f} | "
            f"{best.get('patched_top1_rate', 0.0):.3f} | {best.get('mean_rank_delta', 0.0):.2f} | "
            f"{best.get('mean_patched_pmv', 0.0):.3f} | {best.get('patched_best_other_route', {})} |"
        )
    lines.extend(["", "## Best Conditions", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| condition | repair_rate | patched_top1 | patched_rank | rank_delta | patched_pmv | best_other |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for b in item["summary"]["best_conditions"][:10]:
            lines.append(
                f"| {b['condition']} | {b['repair_rate']:.3f} | {b['patched_top1_rate']:.3f} | "
                f"{b['mean_patched_rank']:.2f} | {b['mean_rank_delta']:.2f} | "
                f"{b['mean_patched_pmv']:.3f} | {b['patched_best_other_route']} |"
            )
        lines.append("")
    (OUT_ROOT / "phase686_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-layer-sites", type=int, default=2)
    parser.add_argument("--max-attn-sites", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=12)
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
