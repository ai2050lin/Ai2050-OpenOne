#!/usr/bin/env python3
"""
Phase 689: Pre-L26 Source Path Localization.

Phase 688 showed that the L26/L27 residual state is near-necessary for
DS7B terse_no_explain success. This phase traces where the L26 layer_input
state comes from by scanning earlier layer_out / attn_out / mlp_out sites.

Two effects are measured for each upstream patch:
1. upstream_to_target: change in target layer_input value-minus-prose projection.
2. final_readout: expected token rank/top1/prose-minus-value after patch.
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
    value_phrase,
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


OUT_ROOT = Path("results/glm5_phase689_pre_l26_source_path_localization")
COMPONENTS = ["layer_out", "attn_out", "mlp_out"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def source_layers(model_name: str, target_layer: int) -> list[int]:
    if model_name == "deepseek7b":
        raw = list(range(18, 26))
    elif model_name == "glm4":
        raw = list(range(30, 38))
    else:
        raw = list(range(25, 33))
    return [li for li in raw if 0 <= li < target_layer]


def run_patched_with_target_capture(
    model,
    tokenizer,
    device,
    prompt: str,
    patches: list[dict[str, Any]],
    target_layer: int,
    direction: torch.Tensor,
    routes,
    expected_ids,
) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}
    handles = install_patch_hooks(model, patches)
    target_module = get_module(model, target_layer, "layer_input")

    def target_pre_hook(_module, inputs):
        captured["target"] = inputs[0][0, -1].detach()

    handles.append(target_module.register_forward_pre_hook(target_pre_hook))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for h in handles:
            h.remove()
    target_vec = captured["target"]
    diag["target_proj"] = projection(target_vec, direction)
    return diag


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    try:
        dtype = next(model.parameters()).dtype
        target_layer = model_layers(args.model, len(get_layers(model)))[0]
        scan_layers = source_layers(args.model, target_layer)
        source_sites = [(li, comp) for li in scan_layers for comp in COMPONENTS]
        target_site = (target_layer, "layer_input")
        capture_sites_all = source_sites + [target_site]

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
            short_target_proj = projection(short_states[target_site], direction)
            terse_target_proj = projection(terse_states[target_site], direction)
            target_delta = terse_target_proj - short_target_proj

            for li, component in source_sites:
                site = (li, component)
                if site not in short_states or site not in terse_states:
                    continue
                delta = terse_states[site] - short_states[site]
                conditions = [
                    ("restore", "add_delta", short_prompt, short_states[site] + delta),
                    ("restore", "random_same_norm", short_prompt, short_states[site] + random_same_norm(delta, seed=idx * 3023 + li * 43)),
                    ("degradation", "replace_short", terse_prompt, short_states[site]),
                    ("degradation", "random_same_norm", terse_prompt, terse_states[site] + random_same_norm(delta, seed=idx * 4019 + li * 47)),
                ]
                for phase_kind, mode, prompt, new_vec in conditions:
                    patched = run_patched_with_target_capture(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        [{"layer": li, "component": component, "new_vec": new_vec}],
                        target_layer,
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
                        target_layer,
                        short_diag,
                        terse_diag,
                        patched,
                        short_target_proj,
                        terse_target_proj,
                        target_delta,
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
    (OUT_ROOT / f"phase689_{args.model}_source_path_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 689,
        "title": "Pre-L26 Source Path Localization",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "target_layer": target_layer,
        "source_layers": scan_layers,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "summary": summary,
    }
    (OUT_ROOT / f"phase689_{args.model}_source_path_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def make_row(meta, case_id, phase_kind, mode, site, target_layer, short_diag, terse_diag, patched, short_target_proj, terse_target_proj, target_delta):
    if phase_kind == "restore":
        final_success_change = (not short_diag["expected_top1"]) and patched["expected_top1"]
        rank_effect = short_diag["expected_rank"] - patched["expected_rank"]
        pmv_effect = short_diag["prose_minus_value"] - patched["prose_minus_value"]
        target_effect = patched["target_proj"] - short_target_proj
    else:
        final_success_change = terse_diag["expected_top1"] and not patched["expected_top1"]
        rank_effect = patched["expected_rank"] - terse_diag["expected_rank"]
        pmv_effect = patched["prose_minus_value"] - terse_diag["prose_minus_value"]
        target_effect = terse_target_proj - patched["target_proj"]
    return {
        "case_id": case_id,
        "family": meta[case_id]["family"],
        "relation": meta[case_id]["relation"],
        "value": meta[case_id]["value"],
        "phase_kind": phase_kind,
        "mode": mode,
        "site": site,
        "target_site": f"L{target_layer}_layer_input",
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
        "short_target_proj": short_target_proj,
        "terse_target_proj": terse_target_proj,
        "target_delta_terse_minus_short": target_delta,
        "patched_target_proj": patched["target_proj"],
        "target_effect": target_effect,
        "target_delta_fraction": target_effect / target_delta if abs(target_delta) > 1e-8 else None,
        "patched_best_other_route": patched["best_other_route"],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "success_change_rate": sum(1 for r in rows if r["final_success_change"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "mean_short_rank": sum(r["short_rank"] for r in rows) / n,
        "mean_terse_rank": sum(r["terse_rank"] for r in rows) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_rank_effect": sum(r["rank_effect"] for r in rows) / n,
        "mean_short_pmv": sum(r["short_pmv"] for r in rows) / n,
        "mean_terse_pmv": sum(r["terse_pmv"] for r in rows) / n,
        "mean_patched_pmv": sum(r["patched_pmv"] for r in rows) / n,
        "mean_pmv_effect": sum(r["pmv_effect"] for r in rows) / n,
        "mean_short_target_proj": sum(r["short_target_proj"] for r in rows) / n,
        "mean_terse_target_proj": sum(r["terse_target_proj"] for r in rows) / n,
        "mean_patched_target_proj": sum(r["patched_target_proj"] for r in rows) / n,
        "mean_target_effect": sum(r["target_effect"] for r in rows) / n,
        "mean_target_delta": sum(r["target_delta_terse_minus_short"] for r in rows) / n,
        "mean_target_delta_fraction": sum(
            r["target_delta_fraction"] for r in rows if r["target_delta_fraction"] is not None
        ) / max(1, sum(1 for r in rows if r["target_delta_fraction"] is not None)),
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, paired_ids: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["phase_kind"], r["mode"], r["site"])].append(r)
    by_condition = {f"{k}|{m}|{s}": summarize_group(v) for (k, m, s), v in grouped.items()}
    restore_best = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("restore|add_delta|")),
        key=lambda kv: (kv[1].get("success_change_rate", 0.0), kv[1].get("mean_target_effect", 0.0), kv[1].get("mean_rank_effect", 0.0)),
        reverse=True,
    )[:24]
    degradation_best = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("degradation|replace_short|")),
        key=lambda kv: (kv[1].get("success_change_rate", 0.0), kv[1].get("mean_target_effect", 0.0), kv[1].get("mean_rank_effect", 0.0)),
        reverse=True,
    )[:24]
    target_best = sorted(
        by_condition.items(),
        key=lambda kv: abs(kv[1].get("mean_target_effect", 0.0)),
        reverse=True,
    )[:24]
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "by_condition": by_condition,
        "best_restore_conditions": [{"condition": k, **v} for k, v in restore_best],
        "best_degradation_conditions": [{"condition": k, **v} for k, v in degradation_best],
        "largest_target_effect_conditions": [{"condition": k, **v} for k, v in target_best],
    }


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase689_*_source_path_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 689,
        "title": "Pre-L26 Source Path Localization Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase689_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 689 Pre-L26 Source Path Localization",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | target | source_layers | best_restore | repair | target_gain | rank_effect | best_degrade | drop | target_loss | rank_effect |",
        "|---|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for item in models:
        br = item["summary"]["best_restore_conditions"][0] if item["summary"]["best_restore_conditions"] else {}
        bd = item["summary"]["best_degradation_conditions"][0] if item["summary"]["best_degradation_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | L{item['target_layer']}_layer_input | {item['source_layers']} | "
            f"{br.get('condition', '')} | {br.get('success_change_rate', 0.0):.3f} | {br.get('mean_target_effect', 0.0):.3f} | {br.get('mean_rank_effect', 0.0):.2f} | "
            f"{bd.get('condition', '')} | {bd.get('success_change_rate', 0.0):.3f} | {bd.get('mean_target_effect', 0.0):.3f} | {bd.get('mean_rank_effect', 0.0):.2f} |"
        )
    for section, key in [("Best Restore", "best_restore_conditions"), ("Best Degradation", "best_degradation_conditions"), ("Largest Target Effects", "largest_target_effect_conditions")]:
        lines.extend(["", f"## {section}", ""])
        for item in models:
            lines.append(f"### {item['model']}")
            lines.append("")
            lines.append("| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
            for row in item["summary"][key][:16]:
                lines.append(
                    f"| {row['condition']} | {row['success_change_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                    f"{row['mean_patched_rank']:.2f} | {row['mean_rank_effect']:.2f} | "
                    f"{row['mean_target_effect']:.3f} | {row['mean_target_delta_fraction']:.3f} | "
                    f"{row['mean_pmv_effect']:.3f} | {row['patched_best_other_route']} |"
                )
            lines.append("")
    (OUT_ROOT / "phase689_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
