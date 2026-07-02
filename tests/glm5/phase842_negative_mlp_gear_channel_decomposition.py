#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase828_cross_component_consistency_fiber_composition as p828  # noqa: E402
import phase834_blocker_aware_internal_route_boundary_predictor as p834  # noqa: E402
import phase837_global_gear_response_atlas_pilot as p837  # noqa: E402
import phase838_gear_response_decomposition_prediction as p838  # noqa: E402
import phase839_gear_interaction_edge_minimal_set as p839  # noqa: E402
import phase840_strict_target_interaction_natural_coactivation as p840  # noqa: E402
import phase841_signed_complementary_gear_role_validation as p841  # noqa: E402


PHASE = 842
RESULT_ROOT = Path("tests/result/phase842_negative_mlp_gear_channel_decomposition")
TARGET_CLASS = "target_equivalent"


def log(msg: str) -> None:
    p837.log(msg)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    return p838.finite(value, default)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def negative_role_labels(model_name: str, args: argparse.Namespace) -> list[str]:
    roles = p841.infer_role_signs(model_name)
    labels = [
        label
        for label, role in roles.items()
        if int(role.get("role_sign", 0)) < 0 and str(role.get("component_kind")) == "mlp_channel_group"
    ]
    labels.sort(key=lambda label: finite((roles.get(label) or {}).get("mean_strict_selected_signed_sum")))
    if int(args.max_negative_components) > 0:
        labels = labels[: int(args.max_negative_components)]
    return labels


def selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = p840.load_strict_candidates(args.model, args)
    return p840.selected_cases(args, candidates)


def channel_id_for(comp_data: dict[str, Any], local_idx: int) -> int | None:
    channel_ids = comp_data.get("spec", {}).get("channel_ids") or []
    if 0 <= int(local_idx) < len(channel_ids):
        return int(channel_ids[int(local_idx)])
    return None


def subset_item(item: dict[str, Any], selected: list[int], alpha: float = 1.0, zero: bool = False) -> dict[str, Any]:
    out = dict(item)
    out["selected_indices"] = [int(x) for x in selected]
    out["alpha"] = float(alpha)
    if zero:
        donor_vec = out["donor_vec"].clone()
        for idx in out["selected_indices"]:
            if 0 <= int(idx) < int(donor_vec.numel()):
                donor_vec[int(idx)] = 0.0
        out["donor_vec"] = donor_vec
    return out


def channel_mode_specs(selected: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [int(x) for x in selected]
    if int(args.max_channels) > 0:
        selected = selected[: int(args.max_channels)]
    specs: list[dict[str, Any]] = [
        {"mode": "full_original", "mode_family": "full", "channel_local_index": None, "selected": selected, "alpha": 1.0},
        {"mode": "full_flip", "mode_family": "full", "channel_local_index": None, "selected": selected, "alpha": -1.0},
        {"mode": "full_zero", "mode_family": "full", "channel_local_index": None, "selected": selected, "alpha": 1.0, "zero": True},
    ]
    for idx in selected:
        rest = [x for x in selected if x != idx]
        specs.extend(
            [
                {"mode": f"single_original_{idx}", "mode_family": "single_original", "channel_local_index": idx, "selected": [idx], "alpha": 1.0},
                {"mode": f"leave_one_out_{idx}", "mode_family": "leave_one_out", "channel_local_index": idx, "selected": rest, "alpha": 1.0},
                {"mode": f"flip_one_{idx}", "mode_family": "flip_one", "channel_local_index": idx, "selected": selected, "flip_index": idx},
                {"mode": f"zero_one_{idx}", "mode_family": "zero_one", "channel_local_index": idx, "selected": selected, "zero_index": idx},
            ]
        )
    return specs


def patch_items_for_spec(base_item: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [int(x) for x in spec.get("selected") or []]
    mode_family = str(spec.get("mode_family"))
    if mode_family == "flip_one":
        idx = int(spec["flip_index"])
        rest = [x for x in selected if x != idx]
        out = []
        if rest:
            out.append(subset_item(base_item, rest, 1.0))
        out.append(subset_item(base_item, [idx], -1.0))
        return out
    if mode_family == "zero_one":
        idx = int(spec["zero_index"])
        rest = [x for x in selected if x != idx]
        out = []
        if rest:
            out.append(subset_item(base_item, rest, 1.0))
        out.append(subset_item(base_item, [idx], 1.0, zero=True))
        return out
    return [subset_item(base_item, selected, finite(spec.get("alpha"), 1.0), bool(spec.get("zero")))]


def eval_case(
    model,
    tokenizer,
    device: torch.device,
    standards: list[dict[str, Any]],
    case: dict[str, Any],
    neg_labels: list[str],
    groups: dict[str, dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lookup = p828.p820.standard_lookup(standards)
    recipient_prompt = p828.p825.natural_prompt(case, args.recipient_prompt)
    recipient_ids = p828.p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p841.greedy_generate_per_item(
        model, tokenizer, device, recipient_ids, [], int(args.max_new_tokens), float(args.alpha)
    )
    baseline_boundary = p828.p825.boundary_for(lookup, case["case_id"], baseline_text)
    candidates_span = p837.gear_candidates(tokenizer, case, args)
    baseline_scored = p837.p816.score_candidates(
        model, tokenizer, device, recipient_ids, candidates_span, int(args.batch_size), int(args.top_k)
    )
    baseline_span = p837.gear_span_profile(baseline_scored)
    target_id = p837.target_first_id(tokenizer, case)
    baseline_id = int(baseline_ids[0]) if baseline_ids else None
    no_patch_logits = p841.first_step_logits_per_item(model, device, recipient_ids, [], float(args.alpha))
    no_patch_rank_profile = p834.rank_profile(no_patch_logits, target_id, baseline_id)
    no_patch_rank_profile["top_token"] = (
        tokenizer.decode([int(no_patch_rank_profile["top_token_id"])])
        if no_patch_rank_profile.get("top_token_id") is not None
        else None
    )

    comp_cache: dict[str, dict[str, Any]] = {}
    for label in neg_labels:
        group = groups.get(label)
        source = source_rows.get(label)
        if not group or not source:
            continue
        comp = p840.prepare_component_data(model, tokenizer, device, case, group, source, recipient_prompt, baseline_ids)
        if comp is not None:
            comp_cache[label] = comp

    rows: list[dict[str, Any]] = []
    for donor_variant in parse_csv(args.search_donor_prompts):
        for label, comp_data in comp_cache.items():
            group = groups[label]
            base_item, natural_features = p840.donor_patch_item_with_features(
                model, tokenizer, device, case, donor_variant, group, comp_data
            )
            if base_item is None:
                continue
            selected = [int(x) for x in group.get("selected_indices") or []]
            for mode_spec in channel_mode_specs(selected, args):
                patch_items = patch_items_for_spec(base_item, mode_spec)
                patched_text, patched_ids = p841.greedy_generate_per_item(
                    model,
                    tokenizer,
                    device,
                    recipient_ids,
                    patch_items,
                    int(args.max_new_tokens),
                    float(args.alpha),
                )
                patched_boundary = p828.p825.boundary_for(lookup, case["case_id"], patched_text)
                patch_logits = p841.first_step_logits_per_item(model, device, recipient_ids, patch_items, float(args.alpha))
                rank_features = p834.rank_profile(patch_logits, target_id, baseline_id)
                top_id = rank_features.get("top_token_id")
                rank_features["top_token"] = tokenizer.decode([int(top_id)]) if top_id is not None else None
                base_rank = no_patch_rank_profile.get("target_rank")
                base_above = no_patch_rank_profile.get("above_target_count")
                rank = rank_features.get("target_rank")
                above = rank_features.get("above_target_count")
                rank_features["target_rank_improved"] = bool(rank is not None and base_rank is not None and int(rank) < int(base_rank))
                rank_features["above_target_decreased"] = bool(
                    above is not None and base_above is not None and int(above) < int(base_above)
                )
                patched_scored = p837.p835.score_candidates_with_first_logits(
                    tokenizer, candidates_span, baseline_scored, patch_logits, int(args.top_k)
                )
                patched_span = p837.gear_span_profile(patched_scored)
                local_idx = mode_spec.get("channel_local_index")
                row: dict[str, Any] = {
                    "row_kind": "phase842_negative_mlp_gear_channel_decomposition",
                    "phase": PHASE,
                    "model": args.model,
                    "round": args.round_name,
                    "source_phase": "phase841_negative_role_needed",
                    "case_id": case["case_id"],
                    "object": case["object"],
                    "target_answer": case["answer"],
                    "donor_variant": donor_variant,
                    "recipient_prompt": args.recipient_prompt,
                    "negative_component_label": label,
                    "component_kind": group.get("component_kind"),
                    "layer_idx": int(group.get("layer_idx", -1)),
                    "mode": mode_spec.get("mode"),
                    "mode_family": mode_spec.get("mode_family"),
                    "channel_local_index": None if local_idx is None else int(local_idx),
                    "channel_id": None if local_idx is None else channel_id_for(comp_data, int(local_idx)),
                    "selected_indices_for_patch": [int(x) for x in mode_spec.get("selected") or []],
                    "baseline_generated": p828.p825.clean_generated(baseline_text),
                    "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
                    "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
                    "baseline_protocol_valid": bool(baseline_boundary.get("protocol_valid")),
                    "patched_generated": p828.p825.clean_generated(patched_text),
                    "patched_token_ids": patched_ids,
                    "patched_boundary_class": patched_boundary.get("final_boundary_class"),
                    "patched_boundary_rank": int(patched_boundary["boundary_rank"]),
                    "patched_protocol_valid": bool(patched_boundary.get("protocol_valid")),
                    "delta_boundary_rank": int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"]),
                    "improved_boundary": int(patched_boundary["boundary_rank"]) > int(baseline_boundary["boundary_rank"]),
                    "degraded_boundary": int(patched_boundary["boundary_rank"]) < int(baseline_boundary["boundary_rank"]),
                    "target_transition": patched_boundary.get("final_boundary_class") == TARGET_CLASS,
                    "baseline_span": baseline_span,
                    "baseline_rank_profile": no_patch_rank_profile,
                    "natural_selected_signed_sum": (natural_features or {}).get("selected_signed_sum"),
                    "natural_selected_positive_ratio": (natural_features or {}).get("selected_positive_ratio"),
                    **rank_features,
                    **p837.profile_delta("patch", baseline_span, patched_span),
                }
                row["response_type"] = p837.classify_response(row)
                row["response_vector"] = p838.row_vector(row)
                rows.append(row)
    add_channel_comparison_features(rows)
    return rows


def full_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("case_id")), str(row.get("donor_variant")), str(row.get("negative_component_label")))


def add_channel_comparison_features(rows: list[dict[str, Any]]) -> None:
    full_rows = {full_key(row): row for row in rows if row.get("mode") == "full_original"}
    for row in rows:
        full = full_rows.get(full_key(row))
        if not full:
            continue
        full_vec = full.get("response_vector") or {}
        vec = row.get("response_vector") or {}
        row["full_original_target_transition"] = bool(full.get("target_transition"))
        row["full_original_generated"] = full.get("patched_generated")
        row["full_original_boundary_class"] = full.get("patched_boundary_class")
        row["delta_quality_vs_full"] = finite(vec.get("target_quality_score")) - finite(full_vec.get("target_quality_score"))
        row["target_lost_vs_full"] = bool(full.get("target_transition") and not row.get("target_transition"))
        row["target_gained_vs_full"] = bool(row.get("target_transition") and not full.get("target_transition"))


def avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def avg_vec(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [finite((row.get("response_vector") or {}).get(key)) for row in rows]
    return avg(vals)


def avg_field(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [finite(row.get(key)) for row in rows if row.get(key) is not None]
    return avg(vals)


def compact_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "target_rows": sum(1 for row in rows if row.get("target_transition")),
        "target_lost_vs_full_rows": sum(1 for row in rows if row.get("target_lost_vs_full")),
        "target_gained_vs_full_rows": sum(1 for row in rows if row.get("target_gained_vs_full")),
        "object_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "object_echo"),
        "format_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "format_echo"),
        "mean_quality": avg_vec(rows, "target_quality_score"),
        "mean_delta_quality_vs_full": avg_field(rows, "delta_quality_vs_full"),
        "classes": dict(Counter(str(row.get("patched_boundary_class")) for row in rows)),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    neg_labels: list[str],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    attn_impl: str | None,
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("mode_family"))].append(row)
        if row.get("channel_local_index") is not None:
            by_channel[f"{row.get('channel_local_index')}:{row.get('channel_id')}"].append(row)

    channel_records = []
    for key, vals in by_channel.items():
        local, channel_id = key.split(":", 1)
        channel_records.append(
            {
                "channel_local_index": int(local),
                "channel_id": None if channel_id == "None" else int(channel_id),
                "n_rows": len(vals),
                "single_target_rows": sum(
                    1 for row in vals if row.get("mode_family") == "single_original" and row.get("target_transition")
                ),
                "leave_one_out_loss_rows": sum(
                    1 for row in vals if row.get("mode_family") == "leave_one_out" and row.get("target_lost_vs_full")
                ),
                "flip_one_loss_rows": sum(
                    1 for row in vals if row.get("mode_family") == "flip_one" and row.get("target_lost_vs_full")
                ),
                "zero_one_loss_rows": sum(
                    1 for row in vals if row.get("mode_family") == "zero_one" and row.get("target_lost_vs_full")
                ),
                "mean_delta_quality_vs_full": avg_field(vals, "delta_quality_vs_full"),
                "classes": dict(Counter(str(row.get("patched_boundary_class")) for row in vals)),
            }
        )
    channel_records.sort(
        key=lambda item: (
            int(item["leave_one_out_loss_rows"]),
            int(item["flip_one_loss_rows"]),
            int(item["single_target_rows"]),
            finite(item["mean_delta_quality_vs_full"]),
        ),
        reverse=True,
    )
    top_rows = sorted(
        rows,
        key=lambda row: (
            int(bool(row.get("target_lost_vs_full"))),
            int(bool(row.get("target_transition"))),
            abs(finite(row.get("delta_quality_vs_full"))),
        ),
        reverse=True,
    )[:100]
    skipped = not neg_labels
    return {
        "phase": PHASE,
        "title": "Negative MLP Gear Channel Decomposition",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "skipped_model_load": skipped,
        "skip_reason": "no Phase 841 negative MLP role candidates" if skipped else None,
        "n_rows": len(rows),
        "n_negative_components": len(neg_labels),
        "negative_component_labels": neg_labels,
        "n_cases": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "donor_variants": parse_csv(args.search_donor_prompts),
        "target_rows": sum(1 for row in rows if row.get("target_transition")),
        "full_original_target_rows": sum(1 for row in rows if row.get("mode") == "full_original" and row.get("target_transition")),
        "target_lost_vs_full_rows": sum(1 for row in rows if row.get("target_lost_vs_full")),
        "target_gained_vs_full_rows": sum(1 for row in rows if row.get("target_gained_vs_full")),
        "object_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "object_echo"),
        "format_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "format_echo"),
        "mode_family_summary": {mode: compact_rows(vals) for mode, vals in sorted(by_family.items())},
        "channel_records": channel_records,
        "top_channel_rows": [
            {
                "case_id": row.get("case_id"),
                "donor_variant": row.get("donor_variant"),
                "mode": row.get("mode"),
                "mode_family": row.get("mode_family"),
                "channel_local_index": row.get("channel_local_index"),
                "channel_id": row.get("channel_id"),
                "patched_boundary_class": row.get("patched_boundary_class"),
                "patched_generated": row.get("patched_generated"),
                "target_transition": bool(row.get("target_transition")),
                "full_original_target_transition": bool(row.get("full_original_target_transition")),
                "target_lost_vs_full": bool(row.get("target_lost_vs_full")),
                "target_gained_vs_full": bool(row.get("target_gained_vs_full")),
                "target_quality_score": finite((row.get("response_vector") or {}).get("target_quality_score")),
                "delta_quality_vs_full": row.get("delta_quality_vs_full"),
            }
            for row in top_rows
        ],
        "boundary": (
            "This phase decomposes a negative MLP channel group under patch intervention. It does not prove "
            "natural unpatched causal use or global geometry reuse."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    neg_labels = negative_role_labels(args.model, args)
    cases = selected_cases(args)
    groups = p839.load_phase837_groups(args.model)
    source_rows = {}
    for label in neg_labels:
        group = groups.get(label)
        if not group:
            continue
        source = p837.source_row_for_group(args.model, group, args)
        if source is not None:
            source_rows[label] = source
    log(
        f"{args.model}/{args.round_name}: negative_labels={len(neg_labels)} source_rows={len(source_rows)} "
        f"cases={len(cases)} donors={parse_csv(args.search_donor_prompts)}"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "negative_labels": neg_labels,
                    "cases": [case["case_id"] for case in cases],
                    "source_rows": len(source_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"negative_labels": neg_labels, "cases": cases}
    if not neg_labels or not source_rows or not cases:
        summary = summarize_rows([], neg_labels, cases, args, None)
        if neg_labels and not source_rows:
            summary["skip_reason"] = "negative role candidates exist but source rows are missing"
        p828.write_jsonl(out_dir / f"phase842_{args.model}_rows.jsonl", [])
        p828.write_json(out_dir / f"phase842_{args.model}_summary.json", summary)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "round": args.round_name,
                    "rows": 0,
                    "skipped_model_load": True,
                    "skip_reason": summary.get("skip_reason"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return summary

    model, tokenizer, device, attn_impl = p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    standards = p828.p820.standard_rows()
    rows: list[dict[str, Any]] = []
    try:
        for idx, case in enumerate(cases, 1):
            rows.extend(eval_case(model, tokenizer, device, standards, case, neg_labels, groups, source_rows, args))
            if idx % int(args.log_every) == 0 or idx == len(cases):
                log(f"{args.model}: evaluated cases {idx}/{len(cases)} rows={len(rows)}")
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, neg_labels, cases, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase842_{args.model}_rows.jsonl", rows)
    p828.write_json(out_dir / f"phase842_{args.model}_summary.json", summary)
    printable = {
        "model": args.model,
        "round": args.round_name,
        "rows": summary["n_rows"],
        "negative_components": summary["n_negative_components"],
        "full_original_target_rows": summary["full_original_target_rows"],
        "target_lost_vs_full_rows": summary["target_lost_vs_full_rows"],
        "target_gained_vs_full_rows": summary["target_gained_vs_full_rows"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{finite(value):.4f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 842 Negative MLP Gear Channel Decomposition ({payload['round']})",
        "",
        "- Source: Phase 841 negative MLP role candidate.",
        "- Boundary: channel-level patch decomposition; not natural ablation.",
        "",
        "## Model Summary",
        "",
        "| model | skipped | neg comps | rows | cases | full-original target | lost vs full | gained vs full | object_echo | format_echo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {int(bool(data.get('skipped_model_load')))} | {data.get('n_negative_components', 0)} | "
            f"{data.get('n_rows', 0)} | {data.get('n_cases', 0)} | {data.get('full_original_target_rows', 0)} | "
            f"{data.get('target_lost_vs_full_rows', 0)} | {data.get('target_gained_vs_full_rows', 0)} | "
            f"{data.get('object_echo_rows', 0)} | {data.get('format_echo_rows', 0)} |"
        )
    lines += ["", "## Mode Family Summary", ""]
    lines += ["| model | mode family | n | target | lost vs full | gained vs full | mean quality | mean delta quality | classes |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for mode, row in (data.get("mode_family_summary") or {}).items():
            lines.append(
                f"| {model_name} | `{mode}` | {row.get('n', 0)} | {row.get('target_rows', 0)} | "
                f"{row.get('target_lost_vs_full_rows', 0)} | {row.get('target_gained_vs_full_rows', 0)} | "
                f"{fmt(row.get('mean_quality'))} | {fmt(row.get('mean_delta_quality_vs_full'))} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Channel Records", ""]
    lines += ["| model | local | channel id | single target | leave-one-out loss | flip-one loss | zero-one loss | mean delta quality | classes |"]
    lines += ["|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in (data.get("channel_records") or [])[:40]:
            lines.append(
                f"| {model_name} | {row.get('channel_local_index')} | {row.get('channel_id')} | "
                f"{row.get('single_target_rows', 0)} | {row.get('leave_one_out_loss_rows', 0)} | "
                f"{row.get('flip_one_loss_rows', 0)} | {row.get('zero_one_loss_rows', 0)} | "
                f"{fmt(row.get('mean_delta_quality_vs_full'))} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Top Channel Rows", ""]
    lines += ["| model | case | donor | mode | local | channel id | class | output | target | full target | lost | gained | quality | delta |"]
    lines += ["|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in data.get("top_channel_rows") or []:
            output = str(row.get("patched_generated") or "").replace("|", "/")[:60]
            lines.append(
                f"| {model_name} | `{row.get('case_id')}` | `{row.get('donor_variant')}` | `{row.get('mode')}` | "
                f"{row.get('channel_local_index')} | {row.get('channel_id')} | `{row.get('patched_boundary_class')}` | "
                f"{output} | {int(bool(row.get('target_transition')))} | "
                f"{int(bool(row.get('full_original_target_transition')))} | {int(bool(row.get('target_lost_vs_full')))} | "
                f"{int(bool(row.get('target_gained_vs_full')))} | {fmt(row.get('target_quality_score'))} | "
                f"{fmt(row.get('delta_quality_vs_full'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_summaries": {},
        "models": [],
    }
    for model_name in p828.MODELS:
        path = out_dir / f"phase842_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase842_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase842_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=p828.MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--recipient-prompt", default="no_choices")
    parser.add_argument("--case-scope", choices=["candidate", "holdout", "candidate_plus_holdout", "all"], default="candidate")
    parser.add_argument("--candidate-combo-kinds", default="pair")
    parser.add_argument("--minimal-only", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--max-negative-components", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--max-channels", type=int, default=16)
    parser.add_argument("--search-donor-prompts", default="natural_question,object_only")
    parser.add_argument("--budgets", default="16,32")
    parser.add_argument("--component-kinds", default="layer_residual,attention_output,mlp_output,attention_head,mlp_channel_group")
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-span-candidates", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(json.dumps({"round": args.round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    run_model(args)


if __name__ == "__main__":
    main()
