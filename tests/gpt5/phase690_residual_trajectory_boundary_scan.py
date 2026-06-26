#!/usr/bin/env python3
"""
Phase 690: Residual Trajectory Boundary Scan.

Phase 689 showed that pre-L26 layer_out sites from L18-L25 can almost fully
restore the L26 value-support state in DS7B. This phase scans earlier
layer_out sites to find the visible boundary of that residual trajectory.

It records both an early checkpoint projection and the final target
layer_input projection, plus final readout behavior.
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
    capture_states,
    classify,
    get_module,
    install_patch_hooks,
    model_layers,
    paired_case_metadata,
    random_same_norm,
)


OUT_ROOT = Path("results/glm5_phase690_residual_trajectory_boundary_scan")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_config(model_name: str, final_target_layer: int) -> tuple[int, list[int], list[tuple[int, str]]]:
    if model_name == "deepseek7b":
        early_target = 18
        layer_out_layers = list(range(8, 19))
        component_layers = [16, 17, 18]
    elif model_name == "glm4":
        early_target = 30
        layer_out_layers = list(range(20, 31))
        component_layers = [28, 29, 30]
    else:
        early_target = 25
        layer_out_layers = list(range(15, 26))
        component_layers = [23, 24, 25]
    layer_out_layers = [li for li in layer_out_layers if 0 <= li < final_target_layer]
    component_layers = [li for li in component_layers if 0 <= li < final_target_layer]
    sites = [(li, "layer_out") for li in layer_out_layers]
    for li in component_layers:
        sites.extend([(li, "attn_out"), (li, "mlp_out")])
    # Preserve order while removing duplicates.
    seen = set()
    unique = []
    for site in sites:
        if site not in seen:
            unique.append(site)
            seen.add(site)
    return early_target, layer_out_layers, unique


def run_patched_with_checkpoints(
    model,
    tokenizer,
    device,
    prompt: str,
    patches: list[dict[str, Any]],
    early_target: int,
    final_target: int,
    direction: torch.Tensor,
    routes,
    expected_ids,
) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}
    handles = install_patch_hooks(model, patches)

    def make_pre_hook(name: str):
        def pre_hook(_module, inputs):
            captured[name] = inputs[0][0, -1].detach()
        return pre_hook

    handles.append(get_module(model, early_target, "layer_input").register_forward_pre_hook(make_pre_hook("early")))
    handles.append(get_module(model, final_target, "layer_input").register_forward_pre_hook(make_pre_hook("final")))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for h in handles:
            h.remove()
    diag["early_target_proj"] = projection(captured["early"], direction)
    diag["final_target_proj"] = projection(captured["final"], direction)
    return diag


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    try:
        dtype = next(model.parameters()).dtype
        final_target = model_layers(args.model, len(get_layers(model)))[0]
        early_target, layer_out_layers, source_sites = scan_config(args.model, final_target)
        checkpoint_sites = [(early_target, "layer_input"), (final_target, "layer_input")]
        capture_sites_all = source_sites + checkpoint_sites

        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype)
            short_prompt = prompt_for(case, SHORT_VARIANT)
            terse_prompt = prompt_for(case, TERSE_VARIANT)

            short_logits, short_states = capture_states(model, tokenizer, device, short_prompt, capture_sites_all)
            terse_logits, terse_states = capture_states(model, tokenizer, device, terse_prompt, capture_sites_all)
            short_diag = classify(short_logits, routes, expected_ids)
            terse_diag = classify(terse_logits, routes, expected_ids)
            short_early = projection(short_states[(early_target, "layer_input")], direction)
            terse_early = projection(terse_states[(early_target, "layer_input")], direction)
            short_final = projection(short_states[(final_target, "layer_input")], direction)
            terse_final = projection(terse_states[(final_target, "layer_input")], direction)

            for li, component in source_sites:
                site = (li, component)
                if site not in short_states or site not in terse_states:
                    continue
                delta = terse_states[site] - short_states[site]
                conditions = [
                    ("restore", "add_delta", short_prompt, short_states[site] + delta),
                    ("restore", "random_same_norm", short_prompt, short_states[site] + random_same_norm(delta, seed=idx * 5021 + li * 53)),
                    ("degradation", "replace_short", terse_prompt, short_states[site]),
                    ("degradation", "random_same_norm", terse_prompt, terse_states[site] + random_same_norm(delta, seed=idx * 6037 + li * 59)),
                ]
                for phase_kind, mode, prompt, new_vec in conditions:
                    patched = run_patched_with_checkpoints(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        [{"layer": li, "component": component, "new_vec": new_vec}],
                        early_target,
                        final_target,
                        direction,
                        routes,
                        expected_ids,
                    )
                    rows.append(make_row(
                        meta,
                        case_id,
                        phase_kind,
                        mode,
                        f"L{li}_{component}",
                        early_target,
                        final_target,
                        short_diag,
                        terse_diag,
                        patched,
                        short_early,
                        terse_early,
                        short_final,
                        terse_final,
                    ))
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: patched {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase690_{args.model}_boundary_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 690,
        "title": "Residual Trajectory Boundary Scan",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "early_target_layer": early_target,
        "final_target_layer": final_target,
        "layer_out_layers": layer_out_layers,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "summary": summary,
    }
    (OUT_ROOT / f"phase690_{args.model}_boundary_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def make_row(meta, case_id, phase_kind, mode, site, early_target, final_target, short_diag, terse_diag, patched, short_early, terse_early, short_final, terse_final):
    if phase_kind == "restore":
        success_change = (not short_diag["expected_top1"]) and patched["expected_top1"]
        rank_effect = short_diag["expected_rank"] - patched["expected_rank"]
        pmv_effect = short_diag["prose_minus_value"] - patched["prose_minus_value"]
        early_effect = patched["early_target_proj"] - short_early
        final_effect = patched["final_target_proj"] - short_final
    else:
        success_change = terse_diag["expected_top1"] and not patched["expected_top1"]
        rank_effect = patched["expected_rank"] - terse_diag["expected_rank"]
        pmv_effect = patched["prose_minus_value"] - terse_diag["prose_minus_value"]
        early_effect = terse_early - patched["early_target_proj"]
        final_effect = terse_final - patched["final_target_proj"]
    early_delta = terse_early - short_early
    final_delta = terse_final - short_final
    return {
        "case_id": case_id,
        "family": meta[case_id]["family"],
        "relation": meta[case_id]["relation"],
        "value": meta[case_id]["value"],
        "phase_kind": phase_kind,
        "mode": mode,
        "site": site,
        "early_target_site": f"L{early_target}_layer_input",
        "final_target_site": f"L{final_target}_layer_input",
        "short_rank": short_diag["expected_rank"],
        "terse_rank": terse_diag["expected_rank"],
        "patched_rank": patched["expected_rank"],
        "short_top1": short_diag["expected_top1"],
        "terse_top1": terse_diag["expected_top1"],
        "patched_top1": patched["expected_top1"],
        "success_change": success_change,
        "rank_effect": rank_effect,
        "short_pmv": short_diag["prose_minus_value"],
        "terse_pmv": terse_diag["prose_minus_value"],
        "patched_pmv": patched["prose_minus_value"],
        "pmv_effect": pmv_effect,
        "short_early_proj": short_early,
        "terse_early_proj": terse_early,
        "patched_early_proj": patched["early_target_proj"],
        "early_effect": early_effect,
        "early_delta": early_delta,
        "early_fraction": early_effect / early_delta if abs(early_delta) > 1e-8 else None,
        "short_final_proj": short_final,
        "terse_final_proj": terse_final,
        "patched_final_proj": patched["final_target_proj"],
        "final_effect": final_effect,
        "final_delta": final_delta,
        "final_fraction": final_effect / final_delta if abs(final_delta) > 1e-8 else None,
        "patched_best_other_route": patched["best_other_route"],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    early_fracs = [r["early_fraction"] for r in rows if r["early_fraction"] is not None]
    final_fracs = [r["final_fraction"] for r in rows if r["final_fraction"] is not None]
    return {
        "n": n,
        "success_change_rate": sum(1 for r in rows if r["success_change"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_rank_effect": sum(r["rank_effect"] for r in rows) / n,
        "mean_patched_pmv": sum(r["patched_pmv"] for r in rows) / n,
        "mean_pmv_effect": sum(r["pmv_effect"] for r in rows) / n,
        "mean_early_effect": sum(r["early_effect"] for r in rows) / n,
        "mean_early_delta": sum(r["early_delta"] for r in rows) / n,
        "mean_early_fraction": sum(early_fracs) / max(1, len(early_fracs)),
        "mean_final_effect": sum(r["final_effect"] for r in rows) / n,
        "mean_final_delta": sum(r["final_delta"] for r in rows) / n,
        "mean_final_fraction": sum(final_fracs) / max(1, len(final_fracs)),
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, paired_ids: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["phase_kind"], r["mode"], r["site"])].append(r)
    by_condition = {f"{k}|{m}|{s}": summarize_group(v) for (k, m, s), v in grouped.items()}
    best_restore = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("restore|add_delta|")),
        key=lambda kv: (kv[1].get("success_change_rate", 0.0), kv[1].get("mean_final_effect", 0.0), kv[1].get("mean_rank_effect", 0.0)),
        reverse=True,
    )[:24]
    best_degrade = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("degradation|replace_short|")),
        key=lambda kv: (kv[1].get("success_change_rate", 0.0), kv[1].get("mean_final_effect", 0.0), kv[1].get("mean_rank_effect", 0.0)),
        reverse=True,
    )[:24]
    random_layer_out = {
        k: v for k, v in by_condition.items()
        if "|random_same_norm|" in k and k.endswith("_layer_out")
    }
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "by_condition": by_condition,
        "best_restore_conditions": [{"condition": k, **v} for k, v in best_restore],
        "best_degradation_conditions": [{"condition": k, **v} for k, v in best_degrade],
        "random_layer_out_controls": random_layer_out,
    }


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase690_*_boundary_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 690,
        "title": "Residual Trajectory Boundary Scan Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase690_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 690 Residual Trajectory Boundary Scan",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | early_target | final_target | layer_out_scan | best_restore | repair | final_gain | rank_effect | best_degrade | drop | final_loss | rank_effect |",
        "|---|---:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for item in models:
        br = item["summary"]["best_restore_conditions"][0] if item["summary"]["best_restore_conditions"] else {}
        bd = item["summary"]["best_degradation_conditions"][0] if item["summary"]["best_degradation_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | L{item['early_target_layer']}_layer_input | L{item['final_target_layer']}_layer_input | {item['layer_out_layers']} | "
            f"{br.get('condition', '')} | {br.get('success_change_rate', 0.0):.3f} | {br.get('mean_final_effect', 0.0):.3f} | {br.get('mean_rank_effect', 0.0):.2f} | "
            f"{bd.get('condition', '')} | {bd.get('success_change_rate', 0.0):.3f} | {bd.get('mean_final_effect', 0.0):.3f} | {bd.get('mean_rank_effect', 0.0):.2f} |"
        )
    for section, key in [("Best Restore", "best_restore_conditions"), ("Best Degradation", "best_degradation_conditions")]:
        lines.extend(["", f"## {section}", ""])
        for item in models:
            lines.append(f"### {item['model']}")
            lines.append("")
            lines.append("| condition | change | patched_top1 | patched_rank | rank_effect | early_effect | early_frac | final_effect | final_frac | pmv_effect | best_other |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for row in item["summary"][key][:18]:
                lines.append(
                    f"| {row['condition']} | {row['success_change_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                    f"{row['mean_patched_rank']:.2f} | {row['mean_rank_effect']:.2f} | "
                    f"{row['mean_early_effect']:.3f} | {row['mean_early_fraction']:.3f} | "
                    f"{row['mean_final_effect']:.3f} | {row['mean_final_fraction']:.3f} | "
                    f"{row['mean_pmv_effect']:.3f} | {row['patched_best_other_route']} |"
                )
            lines.append("")
    (OUT_ROOT / "phase690_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
