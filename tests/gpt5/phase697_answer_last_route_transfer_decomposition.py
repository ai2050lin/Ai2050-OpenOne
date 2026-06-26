#!/usr/bin/env python3
"""
Phase 697: Answer-Last Route Transfer Path Decomposition.

Phase 696 showed that source-token value signals exist early and are nearly
identical between short_only and terse_no_explain; the strong split appears at
answer_last near-readout layers. This phase decomposes that answer_last split.

For paired short_only failures and terse_no_explain successes, capture
answer-last states in the near-readout transfer window and run same-case
component transplants:
  - layer_input
  - attn_out
  - mlp_out
  - layer_out
  - algebraic carry_est = layer_out_delta - attn_delta - mlp_delta
  - transfer-window attn/mlp/attn+mlp/layer patches

This is a component-level route-transfer audit, not a source-token or
head-specific path proof.
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
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    expected_first_ids,
    expected_for,
    prompt_for,
    route_id_sets,
    select_base_cases,
)
from phase685_natural_value_readout_writer_localization import (  # noqa: E402
    SHORT_VARIANT,
    TERSE_VARIANT,
    projection,
    select_paired_cases,
    value_minus_prose_direction,
)
from phase687_l26_l27_value_support_state_decomposition import (  # noqa: E402
    classify,
    get_module,
    install_patch_hooks,
    paired_case_metadata,
    random_same_norm,
)


OUT_ROOT = Path("results/glm5_phase697_answer_last_route_transfer_decomposition")
COMPONENTS = ["layer_input", "attn_out", "mlp_out", "layer_out"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def transfer_layers(model_name: str, n_layers: int) -> list[int]:
    if model_name == "deepseek7b":
        raw = list(range(23, 28))
    elif model_name == "glm4":
        raw = list(range(34, 40))
    else:
        raw = list(range(30, 36))
    return [li for li in raw if 0 <= li < n_layers]


def capture_components_and_final(
    model,
    tokenizer,
    device,
    prompt: str,
    sites: list[tuple[int, str]],
    direction: torch.Tensor,
    routes,
    expected_ids,
) -> tuple[dict[str, Any], dict[tuple[int, str], torch.Tensor]]:
    captured: dict[tuple[int, str], torch.Tensor] = {}
    final_box: dict[str, torch.Tensor] = {}
    handles = []
    for li, component in sites:
        module = get_module(model, li, component)
        if module is None:
            continue
        if component == "layer_input":
            def pre_hook(_module, inputs, site=(li, component)):
                captured[site] = inputs[0][0, -1].detach()
            handles.append(module.register_forward_pre_hook(pre_hook))
        else:
            def out_hook(_module, _inputs, output, site=(li, component)):
                y = extract_tensor(output)
                captured[site] = y[0, -1].detach()
            handles.append(module.register_forward_hook(out_hook))
    final_norm = get_final_norm(model)
    if final_norm is not None:
        def final_pre(_module, inputs):
            final_box["final"] = inputs[0][0, -1].detach()
        handles.append(final_norm.register_forward_pre_hook(final_pre))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for h in handles:
            h.remove()
    if "final" in final_box:
        diag["final_proj"] = projection(final_box["final"], direction)
    else:
        diag["final_proj"] = None
    return diag, captured


def run_patched(
    model,
    tokenizer,
    device,
    prompt: str,
    patches: list[dict[str, Any]],
    direction: torch.Tensor,
    routes,
    expected_ids,
) -> dict[str, Any]:
    final_box: dict[str, torch.Tensor] = {}
    handles = install_patch_hooks(model, patches)
    final_norm = get_final_norm(model)
    if final_norm is not None:
        def final_pre(_module, inputs):
            final_box["final"] = inputs[0][0, -1].detach()
        handles.append(final_norm.register_forward_pre_hook(final_pre))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for h in handles:
            h.remove()
    if "final" in final_box:
        diag["final_proj"] = projection(final_box["final"], direction)
    else:
        diag["final_proj"] = None
    return diag


def make_single_patch(short_states, terse_states, li: int, mode: str, phase_kind: str, seed: int) -> tuple[list[dict[str, Any]], str]:
    if mode in COMPONENTS:
        site = (li, mode)
        donor = terse_states[site] if phase_kind == "restore" else short_states[site]
        return [{"layer": li, "component": mode, "new_vec": donor}], mode
    layer_site = (li, "layer_out")
    attn_site = (li, "attn_out")
    mlp_site = (li, "mlp_out")
    dl = terse_states[layer_site] - short_states[layer_site]
    da = terse_states[attn_site] - short_states[attn_site]
    dm = terse_states[mlp_site] - short_states[mlp_site]
    carry = dl - da - dm
    if mode == "carry_est_layerout":
        new_vec = (short_states[layer_site] + carry) if phase_kind == "restore" else (terse_states[layer_site] - carry)
        return [{"layer": li, "component": "layer_out", "new_vec": new_vec}], "layer_out_carry_est"
    if mode == "random_layer_same_norm":
        noise = random_same_norm(dl, seed=seed)
        new_vec = (short_states[layer_site] + noise) if phase_kind == "restore" else (terse_states[layer_site] + noise)
        return [{"layer": li, "component": "layer_out", "new_vec": new_vec}], "random_layer_out"
    raise ValueError(mode)


def make_window_patch(short_states, terse_states, layers: list[int], mode: str, phase_kind: str) -> tuple[list[dict[str, Any]], list[str]]:
    if mode == "attn_window":
        comps = ["attn_out"]
    elif mode == "mlp_window":
        comps = ["mlp_out"]
    elif mode == "attn_mlp_window":
        comps = ["attn_out", "mlp_out"]
    elif mode == "layer_window":
        comps = ["layer_out"]
    elif mode == "input_window":
        comps = ["layer_input"]
    else:
        raise ValueError(mode)
    patches = []
    for li in layers:
        for comp in comps:
            site = (li, comp)
            donor = terse_states[site] if phase_kind == "restore" else short_states[site]
            patches.append({"layer": li, "component": comp, "new_vec": donor})
    return patches, comps


def make_row(meta, case_id, phase_kind, condition, patch_components, layers, short_diag, terse_diag, patched):
    if phase_kind == "restore":
        final_success_change = (not short_diag["expected_top1"]) and patched["expected_top1"]
        rank_effect = short_diag["expected_rank"] - patched["expected_rank"]
        pmv_effect = short_diag["prose_minus_value"] - patched["prose_minus_value"]
        final_proj_effect = None if patched["final_proj"] is None or short_diag["final_proj"] is None else patched["final_proj"] - short_diag["final_proj"]
    else:
        final_success_change = terse_diag["expected_top1"] and not patched["expected_top1"]
        rank_effect = patched["expected_rank"] - terse_diag["expected_rank"]
        pmv_effect = patched["prose_minus_value"] - terse_diag["prose_minus_value"]
        final_proj_effect = None if patched["final_proj"] is None or terse_diag["final_proj"] is None else terse_diag["final_proj"] - patched["final_proj"]
    return {
        "case_id": case_id,
        "family": meta[case_id]["family"],
        "relation": meta[case_id]["relation"],
        "value": meta[case_id]["value"],
        "phase_kind": phase_kind,
        "condition": condition,
        "patch_components": patch_components,
        "layers": layers,
        "short_rank": short_diag["expected_rank"],
        "terse_rank": terse_diag["expected_rank"],
        "patched_rank": patched["expected_rank"],
        "short_top1": short_diag["expected_top1"],
        "terse_top1": terse_diag["expected_top1"],
        "patched_top1": patched["expected_top1"],
        "final_success_change": final_success_change,
        "rank_effect": rank_effect,
        "short_pmv": short_diag["prose_minus_value"],
        "terse_pmv": terse_diag["prose_minus_value"],
        "patched_pmv": patched["prose_minus_value"],
        "pmv_effect": pmv_effect,
        "short_final_proj": short_diag["final_proj"],
        "terse_final_proj": terse_diag["final_proj"],
        "patched_final_proj": patched["final_proj"],
        "final_proj_effect": final_proj_effect,
        "patched_best_other_route": patched["best_other_route"],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    final_effects = [r["final_proj_effect"] for r in rows if r["final_proj_effect"] is not None]
    return {
        "n": n,
        "success_change_rate": sum(1 for r in rows if r["final_success_change"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "mean_short_rank": sum(r["short_rank"] for r in rows) / n,
        "mean_terse_rank": sum(r["terse_rank"] for r in rows) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_rank_effect": sum(r["rank_effect"] for r in rows) / n,
        "mean_pmv_effect": sum(r["pmv_effect"] for r in rows) / n,
        "mean_final_proj_effect": sum(final_effects) / max(1, len(final_effects)),
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, paired_ids: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["phase_kind"], r["condition"])].append(r)
    by_condition = {f"{k}|{c}": summarize_group(vals) for (k, c), vals in grouped.items()}
    best_restore = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("restore|")),
        key=lambda kv: (kv[1]["success_change_rate"], kv[1]["mean_final_proj_effect"], kv[1]["mean_rank_effect"]),
        reverse=True,
    )
    best_degradation = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("degradation|")),
        key=lambda kv: (kv[1]["success_change_rate"], kv[1]["mean_final_proj_effect"], kv[1]["mean_rank_effect"]),
        reverse=True,
    )
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "by_condition": by_condition,
        "best_restore_conditions": [{"condition": k, **v} for k, v in best_restore],
        "best_degradation_conditions": [{"condition": k, **v} for k, v in best_degradation],
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    try:
        dtype = next(model.parameters()).dtype
        layers = transfer_layers(args.model, len(get_layers(model)))
        sites = [(li, comp) for li in layers for comp in COMPONENTS]
        single_modes = ["layer_input", "attn_out", "mlp_out", "layer_out", "carry_est_layerout", "random_layer_same_norm"]
        window_modes = ["input_window", "attn_window", "mlp_window", "attn_mlp_window", "layer_window"]
        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype)
            short_prompt = prompt_for(case, SHORT_VARIANT)
            terse_prompt = prompt_for(case, TERSE_VARIANT)
            short_diag, short_states = capture_components_and_final(
                model, tokenizer, device, short_prompt, sites, direction, routes, expected_ids
            )
            terse_diag, terse_states = capture_components_and_final(
                model, tokenizer, device, terse_prompt, sites, direction, routes, expected_ids
            )

            for li in layers:
                for mode in single_modes:
                    for phase_kind, prompt in [("restore", short_prompt), ("degradation", terse_prompt)]:
                        patches, comp_name = make_single_patch(
                            short_states, terse_states, li, mode, phase_kind, seed=idx * 69701 + li * 97
                        )
                        patched = run_patched(model, tokenizer, device, prompt, patches, direction, routes, expected_ids)
                        rows.append(make_row(
                            meta, case_id, phase_kind, f"L{li}_{mode}", [comp_name], [li],
                            short_diag, terse_diag, patched,
                        ))
            for mode in window_modes:
                for phase_kind, prompt in [("restore", short_prompt), ("degradation", terse_prompt)]:
                    patches, comps = make_window_patch(short_states, terse_states, layers, mode, phase_kind)
                    patched = run_patched(model, tokenizer, device, prompt, patches, direction, routes, expected_ids)
                    rows.append(make_row(
                        meta, case_id, phase_kind, mode, comps, layers, short_diag, terse_diag, patched
                    ))
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: answer-last transfer patched {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase697_{args.model}_route_transfer_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 697,
        "title": "Answer-Last Route Transfer Path Decomposition",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "transfer_layers": layers,
        "n_paired_cases": len(paired_ids),
        "summary": summary,
    }
    (OUT_ROOT / f"phase697_{args.model}_route_transfer_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase697_*_route_transfer_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 697,
        "title": "Answer-Last Route Transfer Path Decomposition Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase697_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 697 Answer-Last Route Transfer Path Decomposition",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | layers | best_restore | repair | patched_top1 | rank_effect | final_proj_effect | best_degrade | drop | patched_top1 | rank_effect | final_proj_effect |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in models:
        br = item["summary"]["best_restore_conditions"][0] if item["summary"]["best_restore_conditions"] else {}
        bd = item["summary"]["best_degradation_conditions"][0] if item["summary"]["best_degradation_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | {item['transfer_layers']} | "
            f"{br.get('condition', '')} | {br.get('success_change_rate', 0.0):.3f} | {br.get('patched_top1_rate', 0.0):.3f} | "
            f"{br.get('mean_rank_effect', 0.0):.2f} | {br.get('mean_final_proj_effect', 0.0):.3f} | "
            f"{bd.get('condition', '')} | {bd.get('success_change_rate', 0.0):.3f} | {bd.get('patched_top1_rate', 0.0):.3f} | "
            f"{bd.get('mean_rank_effect', 0.0):.2f} | {bd.get('mean_final_proj_effect', 0.0):.3f} |"
        )
    for section, key in [("Best Restore", "best_restore_conditions"), ("Best Degradation", "best_degradation_conditions")]:
        lines.extend(["", f"## {section}", ""])
        for item in models:
            lines.append(f"### {item['model']}")
            lines.append("")
            lines.append("| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
            for row in item["summary"][key][:24]:
                lines.append(
                    f"| {row['condition']} | {row['success_change_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                    f"{row['mean_patched_rank']:.2f} | {row['mean_rank_effect']:.2f} | {row['mean_pmv_effect']:.3f} | "
                    f"{row['mean_final_proj_effect']:.3f} | {row['patched_best_other_route']} |"
                )
            lines.append("")
    (OUT_ROOT / "phase697_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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
