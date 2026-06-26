#!/usr/bin/env python3
"""
Phase 698: Answer-Last Attention Head and Source-Token Path Audit.

Phase 697 showed that the answer_last L23-L27 attention window is causal in
DS7B, but it remained component-level. This phase narrows that path:

1. Capture answer_last o_proj inputs in the near-readout transfer window for
   paired short_only failures and terse_no_explain successes.
2. Rank each head slot by the direct o_proj(delta_head) projection onto the
   case-specific value-minus-prose readout direction.
3. Causally transplant ranked head sets from terse->short and short->terse.
4. Observationally map the selected heads' attention mass to source-token
   groups. This maps source paths but does not yet patch individual sources.
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
from phase599_final_layer_washout_decomposition import get_final_norm  # noqa: E402
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
    classify,
    paired_case_metadata,
)
from phase693_boundary_attention_head_candidate_audit import (  # noqa: E402
    head_meta,
    install_head_patch_hooks,
    oproj_head_contribution,
    random_heads,
)
from phase694_boundary_head_source_token_attention_audit import (  # noqa: E402
    GROUPS,
    group_mass,
    token_groups,
)
from phase697_answer_last_route_transfer_decomposition import transfer_layers  # noqa: E402


OUT_ROOT = Path("results/glm5_phase698_answer_last_attention_head_source_audit")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def capture_oproj_and_final(
    model,
    tokenizer,
    device,
    prompt: str,
    scan_layers: list[int],
    direction: torch.Tensor,
    routes,
    expected_ids,
) -> tuple[dict[str, Any], dict[int, torch.Tensor], torch.Tensor | None]:
    captured_heads: dict[int, torch.Tensor] = {}
    final_box: dict[str, torch.Tensor] = {}
    handles = []
    for li in scan_layers:
        o_proj, _n, _d = head_meta(model, li)

        def pre_hook(_module, inputs, li=li):
            captured_heads[li] = inputs[0][0, -1].detach().cpu()

        handles.append(o_proj.register_forward_pre_hook(pre_hook))

    final_norm = get_final_norm(model)
    if final_norm is not None:
        def final_pre(_module, inputs):
            final_box["final"] = inputs[0][0, -1].detach().cpu()

        handles.append(final_norm.register_forward_pre_hook(final_pre))

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for h in handles:
            h.remove()
    final = final_box.get("final")
    diag["final_proj"] = projection(final, direction.detach().cpu()) if final is not None else None
    return diag, captured_heads, final


def run_head_patched_with_final(
    model,
    tokenizer,
    device,
    prompt: str,
    patch_sets: list[dict[str, Any]],
    direction: torch.Tensor,
    routes,
    expected_ids,
) -> dict[str, Any]:
    final_box: dict[str, torch.Tensor] = {}
    handles = install_head_patch_hooks(model, patch_sets)
    final_norm = get_final_norm(model)
    if final_norm is not None:
        def final_pre(_module, inputs):
            final_box["final"] = inputs[0][0, -1].detach().cpu()

        handles.append(final_norm.register_forward_pre_hook(final_pre))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for h in handles:
            h.remove()
    final = final_box.get("final")
    diag["final_proj"] = projection(final, direction.detach().cpu()) if final is not None else None
    return diag


def make_row(
    meta,
    case_id: str,
    phase_kind: str,
    condition: str,
    head_sets: dict[int, list[int]],
    short_diag: dict[str, Any],
    terse_diag: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    if phase_kind == "restore":
        final_success_change = (not short_diag["expected_top1"]) and patched["expected_top1"]
        rank_effect = short_diag["expected_rank"] - patched["expected_rank"]
        pmv_effect = short_diag["prose_minus_value"] - patched["prose_minus_value"]
        final_proj_effect = None if short_diag["final_proj"] is None or patched["final_proj"] is None else patched["final_proj"] - short_diag["final_proj"]
    else:
        final_success_change = terse_diag["expected_top1"] and not patched["expected_top1"]
        rank_effect = patched["expected_rank"] - terse_diag["expected_rank"]
        pmv_effect = patched["prose_minus_value"] - terse_diag["prose_minus_value"]
        final_proj_effect = None if terse_diag["final_proj"] is None or patched["final_proj"] is None else terse_diag["final_proj"] - patched["final_proj"]
    return {
        "case_id": case_id,
        "family": meta[case_id]["family"],
        "relation": meta[case_id]["relation"],
        "value": meta[case_id]["value"],
        "phase_kind": phase_kind,
        "condition": condition,
        "head_sets": {str(k): v for k, v in head_sets.items()},
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


def summarize_model(
    model_name: str,
    paired_ids: list[str],
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
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
        "n_patch_rows": len(rows),
        "n_source_rows": len(source_rows),
        "by_condition": by_condition,
        "best_restore_conditions": [{"condition": k, **v} for k, v in best_restore],
        "best_degradation_conditions": [{"condition": k, **v} for k, v in best_degradation],
        "top_candidate_heads": sorted(candidate_rows, key=lambda r: r["mean_direct_effect"], reverse=True)[:48],
        "source_attention_summary": summarize_source_rows(source_rows),
    }


def build_conditions(
    scan_layers: list[int],
    layer_meta: dict[int, tuple[Any, int, int]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, dict[int, list[int]]]:
    by_layer_ranked = {
        li: sorted([r for r in candidate_rows if r["layer"] == li], key=lambda r: r["mean_direct_effect"], reverse=True)
        for li in scan_layers
    }
    global_ranked = sorted(candidate_rows, key=lambda r: r["mean_direct_effect"], reverse=True)
    mid = len(scan_layers) // 2
    conds: dict[str, dict[int, list[int]]] = {}
    for li in scan_layers:
        ranked = by_layer_ranked[li]
        for k in [1, 4]:
            conds[f"L{li}_top{k}"] = {li: [int(r["head"]) for r in ranked[: min(k, len(ranked))]]}
    for k in [8, 16, 32]:
        selected: dict[int, list[int]] = defaultdict(list)
        for r in global_ranked[: min(k, len(global_ranked))]:
            selected[int(r["layer"])].append(int(r["head"]))
        conds[f"global_top{k}"] = dict(selected)
    for name, layers in [("early", scan_layers[:mid]), ("late", scan_layers[mid:])]:
        selected = defaultdict(list)
        subset = [r for r in global_ranked if r["layer"] in layers]
        for r in subset[: min(16, len(subset))]:
            selected[int(r["layer"])].append(int(r["head"]))
        conds[f"{name}_top16"] = dict(selected)
    selected = defaultdict(list)
    for li in scan_layers:
        n_heads = layer_meta[li][1]
        selected[li].extend(random_heads(n_heads, min(4, n_heads), 69800 + li))
    conds["global_random_window"] = dict(selected)
    conds["window_all_heads"] = {li: list(range(layer_meta[li][1])) for li in scan_layers}
    return conds


def run_attention_for_sources(model, tokenizer, device, prompt: str):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([ids], device=device),
            return_dict=True,
            use_cache=False,
            output_attentions=True,
        )
    if out.attentions is None:
        raise RuntimeError("model returned no attentions")
    return ids, out.attentions


def source_rows_for_prompt(
    model,
    tokenizer,
    device,
    case: dict[str, Any],
    variant_name: str,
    prompt: str,
    selected_heads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ids, attentions = run_attention_for_sources(model, tokenizer, device, prompt)
    groups = token_groups(tokenizer, prompt, case, ids)
    answer_pos = len(ids) - 1
    rows = []
    try:
        for h in selected_heads:
            li = int(h["layer"])
            head = int(h["head"])
            if li >= len(attentions) or head >= attentions[li].shape[1]:
                continue
            row = attentions[li][0, head, answer_pos, :].detach()
            top_pos = int(torch.argmax(row).detach().cpu().item())
            rows.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "relation": case["relation"],
                "value": value_phrase(case),
                "variant": variant_name,
                "layer": li,
                "head": head,
                "head_key": h["head_key"],
                "seq_len": len(ids),
                "answer_pos": answer_pos,
                "top_attn_pos": top_pos,
                "top_attn_token": tokenizer.decode([ids[top_pos]]),
                "top_attn_mass": float(row[top_pos].detach().float().cpu().item()),
                **group_mass(row, groups),
            })
    finally:
        del attentions
    return rows


def summarize_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0, "by_variant": {}, "by_head_variant": {}, "heads_high_value_mass": []}
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_head_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
        by_head_variant[(r["variant"], r["head_key"])].append(r)

    def summarize(vals: list[dict[str, Any]]) -> dict[str, Any]:
        rec = {"n": len(vals)}
        metric_keys = [k for k in vals[0] if k.startswith("mass_") or k == "target_value_in_record_mass"]
        for key in metric_keys:
            rec[f"mean_{key}"] = sum(v[key] for v in vals) / len(vals)
        rec["top_attn_token_counts"] = dict(Counter(v["top_attn_token"] for v in vals).most_common(10))
        return rec

    hv = [{"condition": f"{variant}|{head}", **summarize(vals)} for (variant, head), vals in by_head_variant.items()]
    return {
        "n_rows": len(rows),
        "by_variant": {variant: summarize(vals) for variant, vals in by_variant.items()},
        "by_head_variant": {f"{variant}|{head}": summarize(vals) for (variant, head), vals in by_head_variant.items()},
        "heads_high_value_mass": sorted(hv, key=lambda x: x.get("mean_target_value_in_record_mass", 0.0), reverse=True)[:32],
        "heads_high_instruction_mass": sorted(hv, key=lambda x: x.get("mean_mass_instruction_line", 0.0), reverse=True)[:32],
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    candidate_acc: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    case_cache: dict[str, dict[str, Any]] = {}
    try:
        dtype = next(model.parameters()).dtype
        scan_layers = transfer_layers(args.model, len(get_layers(model)))
        layer_meta = {li: head_meta(model, li) for li in scan_layers}
        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype)
            direction_cpu = direction.detach().cpu()
            short_prompt = prompt_for(case, SHORT_VARIANT)
            terse_prompt = prompt_for(case, TERSE_VARIANT)

            short_diag, short_heads, short_final = capture_oproj_and_final(
                model, tokenizer, device, short_prompt, scan_layers, direction_cpu, routes, expected_ids
            )
            terse_diag, terse_heads, terse_final = capture_oproj_and_final(
                model, tokenizer, device, terse_prompt, scan_layers, direction_cpu, routes, expected_ids
            )
            case_cache[case_id] = {
                "short_prompt": short_prompt,
                "terse_prompt": terse_prompt,
                "routes": routes,
                "expected_ids": expected_ids,
                "direction": direction_cpu,
                "short_heads": short_heads,
                "terse_heads": terse_heads,
                "short_diag": short_diag,
                "terse_diag": terse_diag,
                "short_final": short_final,
                "terse_final": terse_final,
            }
            for li in scan_layers:
                o_proj, n_heads, head_dim = layer_meta[li]
                delta = terse_heads[li] - short_heads[li]
                for head in range(n_heads):
                    contrib = oproj_head_contribution(o_proj, delta, head, n_heads, head_dim)
                    candidate_acc[(li, head)].append({
                        "direct_effect": projection(contrib, direction_cpu),
                        "delta_norm": float(delta[head * head_dim:(head + 1) * head_dim].float().norm().item()),
                    })
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: cached/scored {idx}/{len(paired_ids)} paired cases")

        candidate_rows = []
        for (li, head), vals in candidate_acc.items():
            candidate_rows.append({
                "layer": li,
                "head": head,
                "head_key": f"L{li}H{head}",
                "mean_direct_effect": sum(v["direct_effect"] for v in vals) / len(vals),
                "mean_delta_norm": sum(v["delta_norm"] for v in vals) / len(vals),
            })
        candidate_rows = sorted(candidate_rows, key=lambda r: r["mean_direct_effect"], reverse=True)
        conds = build_conditions(scan_layers, layer_meta, candidate_rows)

        for idx, case_id in enumerate(paired_ids, 1):
            cur = case_cache[case_id]
            for cond_name, head_sets in conds.items():
                for phase_kind, prompt, donor_key in [
                    ("restore", cur["short_prompt"], "terse_heads"),
                    ("degradation", cur["terse_prompt"], "short_heads"),
                ]:
                    patches = [
                        {"layer": li, "heads": heads, "new_full": cur[donor_key][li]}
                        for li, heads in head_sets.items()
                        if heads
                    ]
                    patched = run_head_patched_with_final(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        patches,
                        cur["direction"],
                        cur["routes"],
                        cur["expected_ids"],
                    )
                    rows.append(make_row(
                        meta,
                        case_id,
                        phase_kind,
                        cond_name,
                        head_sets,
                        cur["short_diag"],
                        cur["terse_diag"],
                        patched,
                    ))
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: causal head patched {idx}/{len(paired_ids)} paired cases")

        selected_heads = candidate_rows[: min(args.source_top_heads, len(candidate_rows))]
        if not args.skip_source_attention:
            for idx, case_id in enumerate(paired_ids, 1):
                case = case_map[case_id]
                for variant_name, prompt in [
                    ("short_only", case_cache[case_id]["short_prompt"]),
                    ("terse_no_explain", case_cache[case_id]["terse_prompt"]),
                ]:
                    source_rows.extend(source_rows_for_prompt(
                        model, tokenizer, device, case, variant_name, prompt, selected_heads
                    ))
                if idx % args.log_every == 0 or idx == len(paired_ids):
                    log(f"{args.model}: source attention mapped {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, rows, candidate_rows, source_rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase698_{args.model}_head_patch_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / f"phase698_{args.model}_candidate_scores.json").write_text(
        json.dumps(candidate_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (OUT_ROOT / f"phase698_{args.model}_source_attention_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in source_rows) + ("\n" if source_rows else ""),
        encoding="utf-8",
    )
    payload = {
        "phase": 698,
        "title": "Answer-Last Attention Head and Source-Token Path Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "transfer_layers": scan_layers,
        "n_paired_cases": len(paired_ids),
        "summary": summary,
    }
    (OUT_ROOT / f"phase698_{args.model}_head_source_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase698_*_head_source_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 698,
        "title": "Answer-Last Attention Head and Source-Token Path Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase698_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 698 Answer-Last Attention Head and Source-Token Path Audit",
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
    for section, key in [("Top Candidate Heads", "top_candidate_heads"), ("Best Restore", "best_restore_conditions"), ("Best Degradation", "best_degradation_conditions")]:
        lines.extend(["", f"## {section}", ""])
        for item in models:
            lines.append(f"### {item['model']}")
            lines.append("")
            if key == "top_candidate_heads":
                lines.append("| head | mean_direct_effect | mean_delta_norm |")
                lines.append("|---|---:|---:|")
                for row in item["summary"][key][:32]:
                    lines.append(f"| {row['head_key']} | {row['mean_direct_effect']:.3f} | {row['mean_delta_norm']:.3f} |")
            else:
                lines.append("| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
                for row in item["summary"][key][:20]:
                    lines.append(
                        f"| {row['condition']} | {row['success_change_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                        f"{row['mean_patched_rank']:.2f} | {row['mean_rank_effect']:.2f} | {row['mean_pmv_effect']:.3f} | "
                        f"{row['mean_final_proj_effect']:.3f} | {row['patched_best_other_route']} |"
                    )
            lines.append("")
    lines.extend(["", "## Source Attention", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        s = item["summary"].get("source_attention_summary", {})
        lines.append("| variant | rows | value_in_record | record | question | instruction | answer | self | object | relation |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for variant, rec in s.get("by_variant", {}).items():
            lines.append(
                f"| {variant} | {rec.get('n', 0)} | {rec.get('mean_target_value_in_record_mass', 0.0):.3f} | "
                f"{rec.get('mean_mass_record_line', 0.0):.3f} | {rec.get('mean_mass_question_line', 0.0):.3f} | "
                f"{rec.get('mean_mass_instruction_line', 0.0):.3f} | {rec.get('mean_mass_answer_line', 0.0):.3f} | "
                f"{rec.get('mean_mass_self_last', 0.0):.3f} | {rec.get('mean_mass_object_name', 0.0):.3f} | "
                f"{rec.get('mean_mass_relation', 0.0):.3f} |"
            )
        lines.append("")
        lines.append("| high-value head | value_in_record | record | instruction | answer | self | top_tokens |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in s.get("heads_high_value_mass", [])[:16]:
            lines.append(
                f"| {row['condition']} | {row.get('mean_target_value_in_record_mass', 0.0):.3f} | "
                f"{row.get('mean_mass_record_line', 0.0):.3f} | {row.get('mean_mass_instruction_line', 0.0):.3f} | "
                f"{row.get('mean_mass_answer_line', 0.0):.3f} | {row.get('mean_mass_self_last', 0.0):.3f} | "
                f"{row.get('top_attn_token_counts', {})} |"
            )
        lines.append("")
    (OUT_ROOT / "phase698_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-top-heads", type=int, default=32)
    parser.add_argument("--skip-source-attention", action="store_true")
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
