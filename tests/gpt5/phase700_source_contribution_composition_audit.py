#!/usr/bin/env python3
"""
Phase 700: Source Contribution Composition and Scaling Audit.

Phase 699 showed that target_value source contribution inside DS7B answer_last
top heads is strongly necessary and partially sufficient, but it does not fully
match the full top-head transplant.

This phase asks whether the missing gap is explained by:
  - composition with nearby source contributions; or
  - insufficient target_value contribution magnitude.

The patch site is still o_proj input at answer_last. The source contribution is
linear in the attention output, so this is a local causal source patch, not a
full recomputation of the attention pattern.
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
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_o_proj  # noqa: E402
from phase132_source_value_contribution_cuda import (  # noqa: E402
    compute_source_contribution,
    get_num_kv_heads,
    get_v_proj,
)
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
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase694_boundary_head_source_token_attention_audit import token_groups  # noqa: E402
from phase697_answer_last_route_transfer_decomposition import transfer_layers  # noqa: E402


OUT_ROOT = Path("results/glm5_phase700_source_contribution_composition_audit")
PHASE698_ROOT = Path("results/glm5_phase698_answer_last_attention_head_source_audit")
SOURCE_GROUPS = [
    "target_value",
    "record_line",
    "record_non_value",
    "relation",
    "object_name",
    "instruction_line",
    "question_line",
    "answer_line",
    "self_last",
]
COMBO_GROUPS = [
    ("target_value+answer_line", ["target_value", "answer_line"]),
    ("target_value+self_last", ["target_value", "self_last"]),
    ("target_value+record_non_value", ["target_value", "record_non_value"]),
    ("target_value+answer_line+self_last", ["target_value", "answer_line", "self_last"]),
    ("record_line", ["record_line"]),
]
DEFAULT_ALPHAS = [0.5, 1.0, 1.5, 2.0]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_top_head_sets(model_name: str, top_k: int) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    path = PHASE698_ROOT / f"phase698_{model_name}_candidate_scores.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase698 candidate scores: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))[:top_k]
    head_sets: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        head_sets[int(r["layer"])].append(int(r["head"]))
    return rows, dict(head_sets)


def source_positions(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    groups = token_groups(tokenizer, prompt, case, ids)
    record = set(groups.get("record_line", []))
    target = set(groups.get("target_value", []))
    out = {
        "target_value": sorted(target),
        "record_line": sorted(record),
        "record_non_value": sorted(record - target),
        "relation": groups.get("relation", []),
        "object_name": groups.get("object_name", []),
        "instruction_line": groups.get("instruction_line", []),
        "question_line": groups.get("question_line", []),
        "answer_line": groups.get("answer_line", []),
        "self_last": groups.get("self_last", []),
    }
    return {k: [p for p in v if 0 <= p < len(ids)] for k, v in out.items()}


def capture_prompt_state(
    model,
    tokenizer,
    device,
    prompt: str,
    case: dict[str, Any],
    scan_layers: list[int],
    direction: torch.Tensor,
    routes,
    expected_ids,
    layer_meta: dict[int, tuple[Any, int, int, int]],
) -> dict[str, Any]:
    captured_heads: dict[int, torch.Tensor] = {}
    captured_values: dict[int, torch.Tensor] = {}
    final_box: dict[str, torch.Tensor] = {}
    handles = []
    for li in scan_layers:
        o_proj, _n_heads, _head_dim, _n_kv = layer_meta[li]
        attn = get_attention_module(get_layers(model)[li])
        v_proj = get_v_proj(attn)

        def o_pre(_module, inputs, li=li):
            captured_heads[li] = inputs[0][0, -1].detach().cpu()

        def v_out(_module, _inputs, output, li=li):
            captured_values[li] = output.detach().cpu()

        handles.append(o_proj.register_forward_pre_hook(o_pre))
        handles.append(v_proj.register_forward_hook(v_out))

    final_norm = get_final_norm(model)
    if final_norm is not None:
        def final_pre(_module, inputs):
            final_box["final"] = inputs[0][0, -1].detach().cpu()

        handles.append(final_norm.register_forward_pre_hook(final_pre))

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )
        if out.attentions is None:
            raise RuntimeError("model returned no attentions")
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
        attentions = {li: out.attentions[li].detach().float().cpu().numpy() for li in scan_layers}
    finally:
        for h in handles:
            h.remove()

    groups = source_positions(tokenizer, prompt, case, ids)
    answer_pos = len(ids) - 1
    contributions: dict[int, dict[str, torch.Tensor]] = {}
    for li in scan_layers:
        _o_proj, n_heads, _head_dim, n_kv_heads = layer_meta[li]
        contributions[li] = {}
        for group_name, positions in groups.items():
            contributions[li][group_name] = compute_source_contribution(
                attentions[li],
                captured_values[li],
                [answer_pos],
                [positions],
                n_heads,
                n_kv_heads,
            )[0].detach().cpu()
    final = final_box.get("final")
    diag["final_proj"] = projection(final, direction.detach().cpu()) if final is not None else None
    return {
        "ids": ids,
        "diag": diag,
        "heads": captured_heads,
        "contributions": contributions,
        "group_lengths": {k: len(v) for k, v in groups.items()},
    }


def install_full_head_patch_hooks(model, patches: list[dict[str, Any]], layer_meta: dict[int, tuple[Any, int, int, int]]):
    handles = []
    for patch in patches:
        li = int(patch["layer"])
        heads = sorted(set(int(h) for h in patch["heads"]))
        donor_full = patch["new_full"]
        o_proj, n_heads, head_dim, _n_kv = layer_meta[li]

        def pre_hook(_module, inputs, heads=heads, donor_full=donor_full, n_heads=n_heads, head_dim=head_dim):
            x = inputs[0]
            y = x.clone()
            yv = y.view(y.shape[0], y.shape[1], n_heads, head_dim)
            donor = donor_full.to(device=y.device, dtype=y.dtype).view(n_heads, head_dim)
            for h in heads:
                if 0 <= h < n_heads:
                    yv[0, -1, h, :] = donor[h]
            return (y,) + tuple(inputs[1:])

        handles.append(o_proj.register_forward_pre_hook(pre_hook))
    return handles


def install_source_delta_hooks(model, patches: list[dict[str, Any]], layer_meta: dict[int, tuple[Any, int, int, int]]):
    handles = []
    for patch in patches:
        li = int(patch["layer"])
        heads = sorted(set(int(h) for h in patch["heads"]))
        delta_full = patch["delta_full"]
        o_proj, n_heads, head_dim, _n_kv = layer_meta[li]

        def pre_hook(_module, inputs, heads=heads, delta_full=delta_full, n_heads=n_heads, head_dim=head_dim):
            x = inputs[0]
            y = x.clone()
            yv = y.view(y.shape[0], y.shape[1], n_heads, head_dim)
            delta = delta_full.to(device=y.device, dtype=y.dtype)
            for h in heads:
                if 0 <= h < n_heads:
                    yv[0, -1, h, :] = yv[0, -1, h, :] + delta[h]
            return (y,) + tuple(inputs[1:])

        handles.append(o_proj.register_forward_pre_hook(pre_hook))
    return handles


def run_patched(
    model,
    tokenizer,
    device,
    prompt: str,
    hook_kind: str,
    patches: list[dict[str, Any]],
    direction: torch.Tensor,
    routes,
    expected_ids,
    layer_meta: dict[int, tuple[Any, int, int, int]],
) -> dict[str, Any]:
    if hook_kind == "full_head":
        handles = install_full_head_patch_hooks(model, patches, layer_meta)
    elif hook_kind == "source_delta":
        handles = install_source_delta_hooks(model, patches, layer_meta)
    else:
        raise ValueError(hook_kind)
    final_box: dict[str, torch.Tensor] = {}
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


def make_full_patches(donor_state: dict[str, Any], head_sets: dict[int, list[int]]) -> list[dict[str, Any]]:
    return [
        {"layer": li, "heads": heads, "new_full": donor_state["heads"][li]}
        for li, heads in head_sets.items()
        if heads
    ]


def make_source_delta_patches(
    src_state: dict[str, Any],
    dst_state: dict[str, Any],
    head_sets: dict[int, list[int]],
    group_name: str,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    return make_multi_source_delta_patches(src_state, dst_state, head_sets, [group_name], scale)


def make_multi_source_delta_patches(
    src_state: dict[str, Any],
    dst_state: dict[str, Any],
    head_sets: dict[int, list[int]],
    group_names: list[str],
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    patches = []
    for li, heads in head_sets.items():
        delta = None
        for group_name in group_names:
            group_delta = src_state["contributions"][li][group_name] - dst_state["contributions"][li][group_name]
            delta = group_delta if delta is None else delta + group_delta
        patches.append({"layer": li, "heads": heads, "delta_full": scale * delta})
    return patches


def make_erase_patches(state: dict[str, Any], head_sets: dict[int, list[int]], group_name: str, scale: float = 1.0) -> list[dict[str, Any]]:
    return make_multi_erase_patches(state, head_sets, [group_name], scale)


def make_multi_erase_patches(
    state: dict[str, Any],
    head_sets: dict[int, list[int]],
    group_names: list[str],
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    patches = []
    for li, heads in head_sets.items():
        contrib = None
        for group_name in group_names:
            group_contrib = state["contributions"][li][group_name]
            contrib = group_contrib if contrib is None else contrib + group_contrib
        delta = -scale * contrib
        patches.append({"layer": li, "heads": heads, "delta_full": delta})
    return patches


def source_group_len(group_lengths: dict[str, int], group_name: str) -> int | None:
    if group_name in group_lengths:
        return group_lengths[group_name]
    if "+" in group_name:
        return sum(group_lengths.get(g, 0) for g in group_name.split("+"))
    return None


def row_from_diag(
    meta,
    case_id: str,
    phase_kind: str,
    condition: str,
    group_name: str,
    short_diag: dict[str, Any],
    terse_diag: dict[str, Any],
    patched: dict[str, Any],
    group_lengths: dict[str, int],
) -> dict[str, Any]:
    if phase_kind == "restore":
        success_change = (not short_diag["expected_top1"]) and patched["expected_top1"]
        rank_effect = short_diag["expected_rank"] - patched["expected_rank"]
        pmv_effect = short_diag["prose_minus_value"] - patched["prose_minus_value"]
        final_effect = None if short_diag["final_proj"] is None or patched["final_proj"] is None else patched["final_proj"] - short_diag["final_proj"]
    else:
        success_change = terse_diag["expected_top1"] and not patched["expected_top1"]
        rank_effect = patched["expected_rank"] - terse_diag["expected_rank"]
        pmv_effect = patched["prose_minus_value"] - terse_diag["prose_minus_value"]
        final_effect = None if terse_diag["final_proj"] is None or patched["final_proj"] is None else terse_diag["final_proj"] - patched["final_proj"]
    return {
        "case_id": case_id,
        "family": meta[case_id]["family"],
        "relation": meta[case_id]["relation"],
        "value": meta[case_id]["value"],
        "phase_kind": phase_kind,
        "condition": condition,
        "source_group": group_name,
        "source_group_len": source_group_len(group_lengths, group_name),
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
        "short_final_proj": short_diag["final_proj"],
        "terse_final_proj": terse_diag["final_proj"],
        "patched_final_proj": patched["final_proj"],
        "final_proj_effect": final_effect,
        "patched_best_other_route": patched["best_other_route"],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    finals = [r["final_proj_effect"] for r in rows if r["final_proj_effect"] is not None]
    return {
        "n": n,
        "success_change_rate": sum(1 for r in rows if r["success_change"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_rank_effect": sum(r["rank_effect"] for r in rows) / n,
        "mean_pmv_effect": sum(r["pmv_effect"] for r in rows) / n,
        "mean_final_proj_effect": sum(finals) / max(1, len(finals)),
        "mean_source_group_len": sum((r["source_group_len"] or 0) for r in rows) / n,
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, rows: list[dict[str, Any]], paired_ids: list[str], head_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["phase_kind"], r["condition"])].append(r)
    by_condition = {f"{k}|{c}": summarize_group(v) for (k, c), v in grouped.items()}
    restore = sorted(
        [{"condition": k, **v} for k, v in by_condition.items() if k.startswith("restore|")],
        key=lambda r: (r["success_change_rate"], r["mean_final_proj_effect"], r["mean_rank_effect"]),
        reverse=True,
    )
    degradation = sorted(
        [{"condition": k, **v} for k, v in by_condition.items() if k.startswith("degradation|")],
        key=lambda r: (r["success_change_rate"], r["mean_final_proj_effect"], r["mean_rank_effect"]),
        reverse=True,
    )
    erase = sorted(
        [{"condition": k, **v} for k, v in by_condition.items() if k.startswith("erase|")],
        key=lambda r: (r["success_change_rate"], r["mean_final_proj_effect"], r["mean_rank_effect"]),
        reverse=True,
    )
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "top_heads": head_rows,
        "by_condition": by_condition,
        "best_restore_conditions": restore,
        "best_degradation_conditions": degradation,
        "best_erase_conditions": erase,
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    top_rows, head_sets = load_top_head_sets(args.model, args.top_heads)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    try:
        dtype = next(model.parameters()).dtype
        scan_layers = transfer_layers(args.model, len(get_layers(model)))
        layer_meta: dict[int, tuple[Any, int, int, int]] = {}
        for li in scan_layers:
            o_proj, n_heads, head_dim = head_meta(model, li)
            attn = get_attention_module(get_layers(model)[li])
            layer_meta[li] = (o_proj, n_heads, head_dim, get_num_kv_heads(model, attn, n_heads))
        head_sets = {li: heads for li, heads in head_sets.items() if li in layer_meta}
        combo_specs = COMBO_GROUPS
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype).detach().cpu()
            short_prompt = prompt_for(case, SHORT_VARIANT)
            terse_prompt = prompt_for(case, TERSE_VARIANT)
            short_state = capture_prompt_state(
                model, tokenizer, device, short_prompt, case, scan_layers, direction, routes, expected_ids, layer_meta
            )
            terse_state = capture_prompt_state(
                model, tokenizer, device, terse_prompt, case, scan_layers, direction, routes, expected_ids, layer_meta
            )
            short_diag = short_state["diag"]
            terse_diag = terse_state["diag"]

            # Positive/negative controls: full selected head-slot transplant.
            for phase_kind, prompt, donor, group_lengths in [
                ("restore", short_prompt, terse_state, short_state["group_lengths"]),
                ("degradation", terse_prompt, short_state, terse_state["group_lengths"]),
            ]:
                patched = run_patched(
                    model, tokenizer, device, prompt, "full_head", make_full_patches(donor, head_sets),
                    direction, routes, expected_ids, layer_meta,
                )
                rows.append(row_from_diag(
                    meta, case_id, phase_kind, f"full_top{args.top_heads}_head_slot", "all_head_slot",
                    short_diag, terse_diag, patched, group_lengths,
                ))

            for combo_name, group_names in combo_specs:
                # Composition delta transplant.
                patched = run_patched(
                    model, tokenizer, device, short_prompt, "source_delta",
                    make_multi_source_delta_patches(terse_state, short_state, head_sets, group_names),
                    direction, routes, expected_ids, layer_meta,
                )
                rows.append(row_from_diag(
                    meta, case_id, "restore", f"combo_delta_{combo_name}", combo_name,
                    short_diag, terse_diag, patched, short_state["group_lengths"],
                ))

                patched = run_patched(
                    model, tokenizer, device, terse_prompt, "source_delta",
                    make_multi_source_delta_patches(short_state, terse_state, head_sets, group_names),
                    direction, routes, expected_ids, layer_meta,
                )
                rows.append(row_from_diag(
                    meta, case_id, "degradation", f"combo_delta_{combo_name}", combo_name,
                    short_diag, terse_diag, patched, terse_state["group_lengths"],
                ))

                patched = run_patched(
                    model, tokenizer, device, terse_prompt, "source_delta",
                    make_multi_erase_patches(terse_state, head_sets, group_names),
                    direction, routes, expected_ids, layer_meta,
                )
                rows.append(row_from_diag(
                    meta, case_id, "erase", f"combo_erase_{combo_name}", combo_name,
                    short_diag, terse_diag, patched, terse_state["group_lengths"],
                ))

            for alpha in alphas:
                alpha_label = f"{alpha:g}"
                patched = run_patched(
                    model, tokenizer, device, short_prompt, "source_delta",
                    make_source_delta_patches(terse_state, short_state, head_sets, "target_value", scale=alpha),
                    direction, routes, expected_ids, layer_meta,
                )
                rows.append(row_from_diag(
                    meta, case_id, "restore", f"alpha_target_value_{alpha_label}", "target_value",
                    short_diag, terse_diag, patched, short_state["group_lengths"],
                ))

                patched = run_patched(
                    model, tokenizer, device, terse_prompt, "source_delta",
                    make_source_delta_patches(short_state, terse_state, head_sets, "target_value", scale=alpha),
                    direction, routes, expected_ids, layer_meta,
                )
                rows.append(row_from_diag(
                    meta, case_id, "degradation", f"alpha_target_value_{alpha_label}", "target_value",
                    short_diag, terse_diag, patched, terse_state["group_lengths"],
                ))

                patched = run_patched(
                    model, tokenizer, device, terse_prompt, "source_delta",
                    make_erase_patches(terse_state, head_sets, "target_value", scale=alpha),
                    direction, routes, expected_ids, layer_meta,
                )
                rows.append(row_from_diag(
                    meta, case_id, "erase", f"alpha_erase_target_value_{alpha_label}", "target_value",
                    short_diag, terse_diag, patched, terse_state["group_lengths"],
                ))

            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: composition patched {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, rows, paired_ids, top_rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase700_{args.model}_composition_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 700,
        "title": "Source Contribution Composition and Scaling Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "transfer_layers": scan_layers,
        "top_heads": args.top_heads,
        "combo_groups": combo_specs,
        "alphas": alphas,
        "n_paired_cases": len(paired_ids),
        "summary": summary,
    }
    (OUT_ROOT / f"phase700_{args.model}_composition_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase700_*_composition_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 700,
        "title": "Source Contribution Composition and Scaling Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase700_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 700 Source Contribution Composition and Scaling Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | layers | top_heads | best_restore | repair | rank_effect | final_effect | best_degrade | drop | rank_effect | final_effect | best_erase | drop | final_effect |",
        "|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for item in models:
        s = item["summary"]
        br = s["best_restore_conditions"][0] if s["best_restore_conditions"] else {}
        bd = s["best_degradation_conditions"][0] if s["best_degradation_conditions"] else {}
        be = s["best_erase_conditions"][0] if s["best_erase_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | {item['transfer_layers']} | {item['top_heads']} | "
            f"{br.get('condition','')} | {br.get('success_change_rate',0.0):.3f} | {br.get('mean_rank_effect',0.0):.2f} | {br.get('mean_final_proj_effect',0.0):.3f} | "
            f"{bd.get('condition','')} | {bd.get('success_change_rate',0.0):.3f} | {bd.get('mean_rank_effect',0.0):.2f} | {bd.get('mean_final_proj_effect',0.0):.3f} | "
            f"{be.get('condition','')} | {be.get('success_change_rate',0.0):.3f} | {be.get('mean_final_proj_effect',0.0):.3f} |"
        )
    for section, key in [
        ("Best Restore", "best_restore_conditions"),
        ("Best Degradation", "best_degradation_conditions"),
        ("Best Erase", "best_erase_conditions"),
    ]:
        lines.extend(["", f"## {section}", ""])
        for item in models:
            lines.append(f"### {item['model']}")
            lines.append("")
            lines.append("| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
            for row in item["summary"][key][:24]:
                lines.append(
                    f"| {row['condition']} | {row['success_change_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                    f"{row['mean_patched_rank']:.2f} | {row['mean_rank_effect']:.2f} | {row['mean_pmv_effect']:.3f} | "
                    f"{row['mean_final_proj_effect']:.3f} | {row['mean_source_group_len']:.2f} | {row['patched_best_other_route']} |"
                )
            lines.append("")
    (OUT_ROOT / "phase700_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-heads", type=int, default=32)
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
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
