#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import (  # noqa: E402
    MODELS,
    bare_phrase_positions,
    line_span_positions,
    load_model_bf16_eager,
    phrase_positions,
    select_evenly,
)
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    install_source_contribution_removal,
    project_source_contribution,
)
from phase752_natural_writer_stability_path_chain import attention_mass_for_group  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402
from phase756_cross_domain_writer_control_downstream_carrier import run_logits  # noqa: E402
from phase762_semantic_numeric_fiber_atlas import VALUE_POOLS  # noqa: E402
from phase765_commonsense_context_identity_closure_test import (  # noqa: E402
    capture_state,
    relation_label,
    route_ids_for_case,
)
from phase771_matched_causal_intervention_reliability_test import (  # noqa: E402
    case_map_for,
    margin,
    pair_info_map,
    phase770_path,
    phase767_path,
    select_matched_case_ids,
    semantic_label,
)


OUT_ROOT = Path("results/glm5_phase773_instruction_source_disentanglement")
RESULT_ROOT = Path("tests/result/phase773_instruction_source_disentanglement")

DEFAULT_SOURCE_GROUPS = [
    "instruction_core",
    "format_cue",
    "candidate_list",
    "answer_prefix",
    "object_tokens",
    "relation_tokens",
    "semantic_pair",
    "task_frame_without_semantic",
]

DEFAULT_SCAN_LAYERS = {
    "qwen3": [30, 32, 33, 35],
    "glm4": [34, 35, 38, 39],
    "deepseek7b": [22, 23, 25, 27],
}

SOURCE_FAMILY = {
    "instruction_core": "protocol_instruction",
    "format_cue": "protocol_format",
    "candidate_list": "candidate_protocol",
    "answer_prefix": "output_prefix",
    "object_tokens": "semantic_object",
    "relation_tokens": "semantic_relation",
    "semantic_pair": "semantic_mixed",
    "task_frame_without_semantic": "query_frame",
    "target_value_tokens": "candidate_target_value",
    "contrast_value_tokens": "candidate_contrast_value",
    "instruction_no_candidate": "protocol_instruction",
    "protocol_all": "protocol_all",
    "all_pre_answer": "full_context",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sorted_positions(values: list[int], n_tokens: int) -> list[int]:
    return [p for p in sorted(set(int(v) for v in values)) if 0 <= p < n_tokens]


def source_family(name: str) -> str:
    return SOURCE_FAMILY.get(name, "other")


def source_groups_for(args: argparse.Namespace) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return DEFAULT_SOURCE_GROUPS[: args.max_source_groups]


def scan_layers_for(args: argparse.Namespace, n_layers: int) -> list[int]:
    if args.scan_layers:
        layers = [int(x.strip()) for x in args.scan_layers.split(",") if x.strip()]
    else:
        layers = DEFAULT_SCAN_LAYERS[args.model]
    layers = [li for li in layers if 0 <= li < n_layers]
    if args.max_scan_layers and len(layers) > args.max_scan_layers:
        layers = layers[: args.max_scan_layers]
    if not layers:
        raise ValueError(f"no valid scan layers for {args.model}")
    return layers


def focus_filter(selected: list[dict[str, Any]], pair_info: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.focus == "all_matched":
        out = selected
    elif args.focus == "clean":
        out = [x for x in selected if x["matched_arm"] == "clean"]
    elif args.focus == "fiber_high":
        out = [x for x in selected if pair_info.get(x["case_id"], {}).get("fiber_bucket") == "fiber_high"]
    elif args.focus == "clean_fiber_high":
        out = [
            x
            for x in selected
            if x["matched_arm"] == "clean" and pair_info.get(x["case_id"], {}).get("fiber_bucket") == "fiber_high"
        ]
    else:
        raise ValueError(args.focus)
    if args.max_cases and len(out) > args.max_cases:
        out = [out[i] for i in select_evenly(len(out), args.max_cases)]
    return out


def build_disentangled_source_groups(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    answer_pos = len(ids) - 1
    obj = case["object"]
    relation = case["relation"]
    n_tokens = len(ids)
    instruction_core = line_span_positions(
        tokenizer,
        prompt,
        lambda s: s.startswith("Answer using common everyday knowledge.") or s.startswith("Use common everyday knowledge."),
    )
    format_cue = line_span_positions(
        tokenizer,
        prompt,
        lambda s: s.startswith("Use exactly one short value.") or s.startswith("Answer with exactly one short value."),
    )
    candidate_list = line_span_positions(tokenizer, prompt, lambda s: s.startswith("Allowed values:"))
    task_line = line_span_positions(tokenizer, prompt, lambda s: s.startswith("Question:") or s.startswith("Task:"))
    answer_prefix = line_span_positions(tokenizer, prompt, lambda s: s.startswith("Answer:"))
    relation_phrases = [relation, relation.replace("_", " "), relation_label(relation)]
    if relation == "grows_on_tree":
        relation_phrases += ["tree", "grow", "grows"]
    if relation == "edible":
        relation_phrases += ["edible", "eat"]
    object_tokens = phrase_positions(tokenizer, ids, [obj, obj.capitalize()])
    relation_tokens = phrase_positions(tokenizer, ids, relation_phrases)
    target_value_tokens = [p for p in bare_phrase_positions(tokenizer, ids, [case["answer"]]) if p < answer_pos]
    contrast_value_tokens = [p for p in bare_phrase_positions(tokenizer, ids, [case["contrast_answer"]]) if p < answer_pos]
    semantic_pair = object_tokens + relation_tokens
    task_frame_without_semantic = sorted(set(task_line) - set(object_tokens) - set(relation_tokens))
    instruction_no_candidate = instruction_core + format_cue
    protocol_all = instruction_core + format_cue + candidate_list + answer_prefix
    groups = {
        "instruction_core": instruction_core,
        "format_cue": format_cue,
        "candidate_list": candidate_list,
        "answer_prefix": [p for p in answer_prefix if p < answer_pos],
        "object_tokens": object_tokens,
        "relation_tokens": relation_tokens,
        "semantic_pair": semantic_pair,
        "task_frame_without_semantic": task_frame_without_semantic,
        "target_value_tokens": target_value_tokens,
        "contrast_value_tokens": contrast_value_tokens,
        "instruction_no_candidate": instruction_no_candidate,
        "protocol_all": protocol_all,
        "all_pre_answer": list(range(0, max(0, answer_pos))),
    }
    return {k: sorted_positions(v, n_tokens) for k, v in groups.items()}


def scan_score(direct: dict[str, Any], attention_mass: float, args: argparse.Namespace) -> float:
    target = max(0.0, float(direct.get("direct_target_boost") or 0.0))
    route = max(0.0, float(direct.get("direct_total_route_suppression") or 0.0))
    margin_gain = max(0.0, float(direct.get("direct_mean_margin_gain") or 0.0))
    attn = max(0.0, float(attention_mass))
    return args.target_weight * target + args.route_weight * route + args.margin_weight * margin_gain + args.attention_weight * attn


def control_head(head: int, n_heads: int, offset: int) -> int:
    if n_heads <= 1:
        return head
    cand = (int(head) + int(offset)) % n_heads
    if cand == head:
        cand = (cand + 1) % n_heads
    return cand


def direct_candidates_for_case(
    model,
    state: dict[str, Any],
    source_groups: list[str],
    scan_layers: list[int],
    target_id: int,
    route_ids: list[int],
    args: argparse.Namespace,
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    answer_pos = state["answer_pos"]
    for layer in scan_layers:
        site = f"L{layer}:attn_out"
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
        for source_group in source_groups:
            src_positions = [int(p) for p in state["source_groups"].get(source_group, [])]
            if not src_positions:
                continue
            contribution = compute_source_contribution(
                state["attentions"][layer],
                state["values"][layer],
                [answer_pos],
                [src_positions],
                n_heads,
                num_kv_heads,
            )
            for head in range(n_heads):
                projected = project_source_contribution(model, layer, [head], contribution)
                direct = direct_delta_score(projected, unembed, target_id, route_ids)
                attn_mass = attention_mass_for_group(state["attentions"][layer], head, answer_pos, src_positions)
                rows.append(
                    {
                        "site": site,
                        "layer": layer,
                        "head": head,
                        "source_group": source_group,
                        "source_family": source_family(source_group),
                        "source_positions_n": len(src_positions),
                        "attention_mass_to_source": attn_mass,
                        "source_direct_score": direct,
                        "scan_score": scan_score(direct, attn_mass, args),
                        "candidate_kind": "source_group_top_component",
                        "selection": "source_group_direct_score_scan",
                    }
                )
    rows.sort(key=lambda r: (r["scan_score"], r["source_direct_score"].get("direct_target_boost") or 0.0), reverse=True)
    return rows


def select_candidates(direct_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for row in direct_rows[: args.top_global_components_per_case]:
        key = (row["layer"], row["head"], row["source_group"])
        if key in seen:
            continue
        item = dict(row)
        item["candidate_kind"] = "global_top_component"
        item["selection"] = "global_direct_score_scan"
        out.append(item)
        seen.add(key)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in direct_rows:
        by_source[row["source_group"]].append(row)
    for source_group in sorted(by_source):
        for row in by_source[source_group][: args.top_components_per_source]:
            key = (row["layer"], row["head"], row["source_group"])
            if key in seen:
                continue
            out.append(dict(row))
            seen.add(key)
    return out


def add_controls(model, selected: list[dict[str, Any]], direct_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.include_controls:
        return selected
    by_key = {(r["layer"], r["head"], r["source_group"]): r for r in direct_rows}
    out = list(selected)
    seen = {(r["layer"], r["head"], r["source_group"], r["candidate_kind"]) for r in out}
    for row in selected:
        attn = get_attention_module(get_layers(model)[row["layer"]])
        n_heads = get_num_heads(model, attn)
        h = control_head(row["head"], n_heads, args.control_offset)
        key = (row["layer"], h, row["source_group"])
        if key not in by_key:
            continue
        ctrl = dict(by_key[key])
        ctrl["candidate_kind"] = "same_layer_control_head"
        ctrl["selection"] = f"control_offset_{args.control_offset}"
        ctrl["control_of"] = f"L{row['layer']}:H{row['head']}:{row['source_group']}"
        seen_key = (ctrl["layer"], ctrl["head"], ctrl["source_group"], ctrl["candidate_kind"])
        if seen_key in seen:
            continue
        out.append(ctrl)
        seen.add(seen_key)
    return out


def causal_test_candidate(model, device, state: dict[str, Any], row: dict[str, Any], target_id: int, contrast_id: int) -> dict[str, Any]:
    layer = int(row["layer"])
    head = int(row["head"])
    source_group = row["source_group"]
    src_positions = [int(p) for p in state["source_groups"].get(source_group, [])]
    answer_pos = state["answer_pos"]
    attn = get_attention_module(get_layers(model)[layer])
    n_heads = get_num_heads(model, attn)
    num_kv_heads = get_num_kv_heads(model, attn, n_heads)
    contribution = compute_source_contribution(
        state["attentions"][layer],
        state["values"][layer],
        [answer_pos],
        [src_positions],
        n_heads,
        num_kv_heads,
    )
    install = install_source_contribution_removal(model, row["site"], [head], contribution)
    after_logits = run_logits(model, device, state["ids"], install)
    base_logits = state["logits"]
    base_target_diag = logit_diag(base_logits, target_id)
    after_target_diag = logit_diag(after_logits, target_id)
    base_margin = margin(base_logits, target_id, contrast_id)
    after_margin = margin(after_logits, target_id, contrast_id)
    return {
        "base_target_rank": base_target_diag["target_rank"],
        "base_target_top1": base_target_diag["target_top1"],
        "after_target_rank": after_target_diag["target_rank"],
        "after_target_top1": after_target_diag["target_top1"],
        "top1_loss": bool(base_target_diag["target_top1"]) and not bool(after_target_diag["target_top1"]),
        "target_logit_drop": float(base_logits[target_id].item() - after_logits[target_id].item()),
        "contrast_logit_gain": float(after_logits[contrast_id].item() - base_logits[contrast_id].item()),
        "margin_drop_target_vs_contrast": float(base_margin - after_margin),
    }


def audit_case(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
    case: dict[str, Any],
    case_label: dict[str, Any],
    pair_info: dict[str, Any],
    scan_layers: list[int],
    source_groups: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    state = capture_state(model, tokenizer, device, case, scan_layers)
    state["source_groups"] = build_disentangled_source_groups(tokenizer, state["prompt"], case, state["ids"])
    target_id = get_first_token_id(tokenizer, case["answer"])
    contrast_id = get_first_token_id(tokenizer, case["contrast_answer"])
    route_ids = route_ids_for_case(tokenizer, case, target_id)
    direct_rows = direct_candidates_for_case(model, state, source_groups, scan_layers, target_id, route_ids, args, unembed)
    selected = select_candidates(direct_rows, args)
    selected = add_controls(model, selected, direct_rows, args)
    phase767 = case_label["phase767"]
    out = []
    for rank, row in enumerate(selected, 1):
        causal = causal_test_candidate(model, device, state, row, target_id, contrast_id)
        out.append(
            {
                "row_kind": "instruction_source_disentanglement",
                "case_id": case["case_id"],
                "pair_index": case_label["pair_index"],
                "matched_arm": case_label["matched_arm"],
                "stratum": case_label["stratum"],
                "object": case["object"],
                "domain": case["domain"],
                "relation": case["relation"],
                "context_format": case["context_format"],
                "target_answer": case["answer"],
                "contrast_answer": case["contrast_answer"],
                "phase767_exact_top1": bool(phase767.get("exact_target_top1")),
                "phase767_semantic_top1": bool(phase767.get("target_top1")),
                "semantic_label": semantic_label(phase767),
                **pair_info,
                "selection_rank": rank,
                "site": row["site"],
                "layer": row["layer"],
                "head": row["head"],
                "component_key": f"{row['site']}:H{row['head']}:{row['source_group']}",
                "source_group": row["source_group"],
                "source_family": row["source_family"],
                "source_positions_n": row["source_positions_n"],
                "candidate_kind": row["candidate_kind"],
                "selection": row["selection"],
                "control_of": row.get("control_of"),
                "scan_score": row["scan_score"],
                "attention_mass_to_source": row["attention_mass_to_source"],
                "source_direct_score": row["source_direct_score"],
                **causal,
            }
        )
    return out


def group_summary(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        direct = [v.get("source_direct_score") or {} for v in vals]
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v["case_id"] for v in vals}),
                "mean_scan_score": safe_mean([v.get("scan_score") for v in vals]),
                "mean_target_logit_drop": safe_mean([v.get("target_logit_drop") for v in vals]),
                "mean_margin_drop_target_vs_contrast": safe_mean([v.get("margin_drop_target_vs_contrast") for v in vals]),
                "mean_attention_mass": safe_mean([v.get("attention_mass_to_source") for v in vals]),
                "mean_direct_target_boost": safe_mean([d.get("direct_target_boost") for d in direct]),
                "mean_direct_route_suppression": safe_mean([d.get("direct_total_route_suppression") for d in direct]),
                "top1_loss_rate": sum(1 for v in vals if v.get("top1_loss")) / len(vals) if vals else None,
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("top1_loss_rate") or 0.0,
            r.get("mean_target_logit_drop") or 0.0,
            r.get("mean_margin_drop_target_vs_contrast") or 0.0,
            r.get("mean_scan_score") or 0.0,
        ),
        reverse=True,
    )
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, model_name: str, attn_impl: str, scan_layers: list[int]) -> dict[str, Any]:
    return {
        "phase": 773,
        "title": "Instruction Source Disentanglement and Object-Route Reweighting",
        "model": model_name,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "focus": args.focus,
        "scan_layers": scan_layers,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_pairs": len({r["pair_index"] for r in rows}),
        "source_groups": source_groups_for(args),
        "by_candidate_kind": group_summary(rows, ["candidate_kind"]),
        "by_source_family": group_summary(rows, ["source_family", "candidate_kind"]),
        "by_source_group": group_summary(rows, ["source_group", "candidate_kind"]),
        "by_matched_arm": group_summary(rows, ["matched_arm", "candidate_kind"]),
        "by_fiber_bucket": group_summary(rows, ["fiber_bucket", "candidate_kind"]),
        "top_components": group_summary(rows, ["component_key", "candidate_kind"])[:40],
        "strict_interpretation": "This phase separates protocol, candidate-list, object, and relation sources. It is still head/source-level.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows767 = load_jsonl(phase767_path(args.model, args.phase767_round))
    phase770 = load_json(phase770_path(args.phase770_round))
    selected = select_matched_case_ids(rows767, args)
    cmap = case_map_for(args)
    pinfo = pair_info_map(phase770, args.model)
    selected = focus_filter(selected, pinfo, args)
    source_groups = source_groups_for(args)
    log(f"{args.model}/{args.round_name}: focus={args.focus} cases={len(selected)} sources={source_groups}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        scan_layers = scan_layers_for(args, len(get_layers(model)))
        unembed = get_unembed(model)
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(selected, 1):
            case = cmap[item["case_id"]]
            rows.extend(audit_case(model, tokenizer, device, args, case, item, pinfo.get(case["case_id"], {}), scan_layers, source_groups, unembed))
            if idx % args.log_every == 0 or idx == len(selected):
                log(f"{args.model}: source disentanglement {idx}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, args.model, attn_impl, scan_layers)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase773_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase773_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "n_cases": summary["n_cases"], "by_source_family": summary["by_source_family"][:8]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 773 Instruction Source Disentanglement ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: split instruction/protocol sources from object/relation semantic sources, then scan and causally remove per-source components.",
        "- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.",
        "",
        "## Candidate Kind Summary",
        "",
        "| model | kind | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_candidate_kind"]:
            lines.append(
                f"| {model} | `{row['candidate_kind']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['mean_target_logit_drop'])} | {fmt(row['mean_margin_drop_target_vs_contrast'])} | "
                f"{fmt(row['top1_loss_rate'])} | {fmt(row['mean_direct_target_boost'])} | {fmt(row['mean_direct_route_suppression'])} |"
            )
    lines += [
        "",
        "## Source Family Summary",
        "",
        "| model | family | kind | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_source_family"]:
            lines.append(
                f"| {model} | `{row['source_family']}` | `{row['candidate_kind']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['mean_target_logit_drop'])} | {fmt(row['mean_margin_drop_target_vs_contrast'])} | "
                f"{fmt(row['top1_loss_rate'])} | {fmt(row['mean_direct_target_boost'])} | {fmt(row['mean_direct_route_suppression'])} |"
            )
    lines += [
        "",
        "## Source Group Summary",
        "",
        "| model | source | kind | rows | cases | target drop | margin drop | top1 loss | attention |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_source_group"]:
            lines.append(
                f"| {model} | `{row['source_group']}` | `{row['candidate_kind']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['mean_target_logit_drop'])} | {fmt(row['mean_margin_drop_target_vs_contrast'])} | "
                f"{fmt(row['top1_loss_rate'])} | {fmt(row['mean_attention_mass'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- This phase tests whether Phase 772's instruction dominance is protocol/candidate-list driven or semantic-source driven.",
        "- Per-source selection prevents instruction from hiding object/relation candidates in global ranking.",
        "- Causal removal remains the evidence; direct score remains a filter.",
        "- This is still not neuron/channel-level evidence.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase773_{model}_summary.json"
        if path.exists():
            by_model[model] = load_json(path)
    payload = {
        "phase": 773,
        "title": "Instruction Source Disentanglement and Object-Route Reweighting",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase773_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase773_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "focus": args.focus, "models": {}}
    for model in MODELS:
        args.model = model
        rows767 = load_jsonl(phase767_path(model, args.phase767_round))
        phase770 = load_json(phase770_path(args.phase770_round))
        selected = select_matched_case_ids(rows767, args)
        pinfo = pair_info_map(phase770, model)
        selected = focus_filter(selected, pinfo, args)
        payload["models"][model] = {
            "selected_cases": len(selected),
            "pairs": len({x["pair_index"] for x in selected}),
            "arms": dict(Counter(x["matched_arm"] for x in selected)),
            "fiber": dict(Counter(pinfo.get(x["case_id"], {}).get("fiber_bucket", "missing") for x in selected)),
            "scan_layers": DEFAULT_SCAN_LAYERS[model][: args.max_scan_layers] if args.max_scan_layers else DEFAULT_SCAN_LAYERS[model],
            "source_groups": source_groups_for(args),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--phase767-round", default="main")
    parser.add_argument("--phase770-round", default="confirm_x_main")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--focus", choices=["all_matched", "clean", "fiber_high", "clean_fiber_high"], default="clean_fiber_high")
    parser.add_argument("--max-per-stratum", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--max-scan-layers", type=int, default=4)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--max-source-groups", type=int, default=8)
    parser.add_argument("--top-global-components-per-case", type=int, default=0)
    parser.add_argument("--top-components-per-source", type=int, default=1)
    parser.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--control-offset", type=int, default=5)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--attention-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.summarize_only:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --dry-run or --summarize-only")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
