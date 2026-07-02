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


PHASE = 840
RESULT_ROOT = Path("tests/result/phase840_strict_target_interaction_natural_coactivation")
SOURCE_839 = Path("tests/result/phase839_gear_interaction_edge_minimal_set/confirm")
TARGET_CLASS = "target_equivalent"


def log(msg: str) -> None:
    p837.log(msg)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    return p838.finite(value, default)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def combo_label(labels: list[str]) -> str:
    return " + ".join(labels)


def normalize_labels(labels: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in labels or [])


def candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, int, float, float, str]:
    return (
        int(item.get("minimal_seed_rows", 0)),
        int(item.get("strict_seed_rows", 0)),
        -int(item.get("n_components", 0)),
        finite(item.get("max_interaction_gain")),
        finite(item.get("mean_quality")),
        str(item.get("combo_label")),
    )


def load_strict_candidates(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_839 / f"phase839_{model_name}_rows.jsonl")
    allowed_kinds = set(parse_csv(args.candidate_combo_kinds))
    by_labels: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = normalize_labels(row.get("combo_labels"))
        if len(labels) <= 1:
            continue
        if allowed_kinds and str(row.get("combo_kind")) not in allowed_kinds:
            continue
        if args.minimal_only and not row.get("minimal_sufficient_candidate"):
            continue
        if row.get("patched_boundary_class") != TARGET_CLASS:
            continue
        if not row.get("target_transition"):
            continue
        if not row.get("positive_interaction_edge"):
            continue
        by_labels[labels].append(row)

    out: list[dict[str, Any]] = []
    for labels, vals in by_labels.items():
        out.append(
            {
                "combo_labels": list(labels),
                "combo_label": combo_label(list(labels)),
                "combo_kind": str(vals[0].get("combo_kind")),
                "n_components": len(labels),
                "strict_seed_rows": len(vals),
                "minimal_seed_rows": sum(1 for row in vals if row.get("minimal_sufficient_candidate")),
                "seed_case_ids": sorted({str(row.get("case_id")) for row in vals}),
                "seed_donor_variants": sorted({str(row.get("donor_variant")) for row in vals}),
                "seed_outputs": sorted({str(row.get("patched_generated")) for row in vals}),
                "max_interaction_gain": max(finite(row.get("interaction_quality_gain")) for row in vals),
                "mean_quality": sum(finite((row.get("response_vector") or {}).get("target_quality_score")) for row in vals) / len(vals),
                "seed_rows": [
                    {
                        "case_id": row.get("case_id"),
                        "donor_variant": row.get("donor_variant"),
                        "patched_generated": row.get("patched_generated"),
                        "interaction_quality_gain": row.get("interaction_quality_gain"),
                        "minimal_sufficient_candidate": row.get("minimal_sufficient_candidate"),
                    }
                    for row in vals
                ],
            }
        )
    out.sort(key=candidate_sort_key, reverse=True)
    if int(args.max_candidates) > 0:
        out = out[: int(args.max_candidates)]
    return out


def labels_from_candidates(candidates: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    labels: list[str] = []
    for cand in candidates:
        for label in cand.get("combo_labels") or []:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def selected_cases(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cmap = p828.p825.case_map()
    if args.case_scope == "candidate":
        case_ids = []
        seen: set[str] = set()
        for cand in candidates:
            for case_id in cand.get("seed_case_ids") or []:
                if case_id in cmap and case_id not in seen:
                    seen.add(case_id)
                    case_ids.append(case_id)
        out = [cmap[x] for x in case_ids]
    elif args.case_scope == "candidate_plus_holdout":
        holdout_args = argparse.Namespace(**vars(args))
        holdout_args.case_scope = "holdout"
        base_cases = p839.selected_cases(holdout_args)
        case_ids = []
        seen: set[str] = set()
        for cand in candidates:
            for case_id in cand.get("seed_case_ids") or []:
                if case_id in cmap and case_id not in seen:
                    seen.add(case_id)
                    case_ids.append(case_id)
        for case in base_cases:
            if case["case_id"] not in seen:
                seen.add(case["case_id"])
                case_ids.append(case["case_id"])
        out = [cmap[x] for x in case_ids if x in cmap]
    else:
        case_args = argparse.Namespace(**vars(args))
        case_args.case_scope = args.case_scope
        out = p839.selected_cases(case_args)
    if 0 < int(args.max_cases) < len(out):
        out = out[: int(args.max_cases)]
    return out


def combo_specs(candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for label in labels_from_candidates(candidates):
        key = ("single", (label,))
        specs.append({"combo_kind": "single_control", "labels": [label], "source": "phase840_single_control"})
        seen.add(key)
    for cand in candidates:
        labels = normalize_labels(cand.get("combo_labels"))
        key = (str(cand.get("combo_kind")), labels)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "combo_kind": cand.get("combo_kind"),
                "labels": list(labels),
                "source": "phase839_strict_candidate",
                "seed_summary": {k: v for k, v in cand.items() if k != "seed_rows"},
            }
        )
    return specs


def prepare_component_data(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    group: dict[str, Any],
    source_row: dict[str, Any],
    recipient_prompt: str,
    baseline_ids: list[int],
) -> dict[str, Any] | None:
    return p837.component_data_for_case(
        model,
        tokenizer,
        device,
        group,
        source_row,
        recipient_prompt,
        case,
        baseline_ids,
    )


def donor_patch_item_with_features(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    donor_variant: str,
    group: dict[str, Any],
    comp_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | tuple[None, None]:
    donor_prompt = p828.p825.natural_prompt(case, donor_variant)
    donor_state = p828.p822.capture_component_state(model, tokenizer, device, donor_prompt, int(group["layer_idx"]))
    donor_vec = p828.p823.component_vector(donor_state, comp_data["spec"])
    if donor_vec is None:
        return None, None
    donor_vec = donor_vec.float().cpu()
    patch_item = {
        "layer_idx": int(group["layer_idx"]),
        "spec": comp_data["spec"],
        "recipient_vec": comp_data["recipient_vec"],
        "donor_vec": donor_vec,
        "selected_indices": comp_data["selected_indices"],
    }
    features = p837.signed_features(comp_data, donor_vec)
    features.update(
        {
            "layer_idx": int(group["layer_idx"]),
            "component_kind": group.get("component_kind"),
            "component_label": group.get("component_label"),
            "budget": int(group.get("budget", 0)),
            "donor_prompt": donor_prompt,
            "has_effective_readout_dir": comp_data.get("effective_dir") is not None,
        }
    )
    return patch_item, features


def natural_combo_features(features_by_label: dict[str, dict[str, Any]], labels: list[str], args: argparse.Namespace) -> dict[str, Any]:
    vals = [features_by_label.get(label) or {} for label in labels]
    signed = [finite(item.get("selected_signed_sum")) for item in vals if item.get("selected_signed_sum") is not None]
    donor_gain = [
        finite(item.get("selected_donor_minus_recipient_score"))
        for item in vals
        if item.get("selected_donor_minus_recipient_score") is not None
    ]
    threshold = float(args.natural_positive_threshold)
    n = len(vals)
    pos = sum(1 for value in signed if value > threshold)
    neg = sum(1 for value in signed if value < -threshold)
    gain_pos = sum(1 for value in donor_gain if value > threshold)
    return {
        "natural_component_count": n,
        "natural_observed_component_count": len(signed),
        "natural_selected_signed_sum_total": sum(signed) if signed else None,
        "natural_selected_signed_sum_min": min(signed) if signed else None,
        "natural_selected_signed_sum_max": max(signed) if signed else None,
        "natural_positive_component_count": pos,
        "natural_negative_component_count": neg,
        "natural_positive_component_ratio": (pos / n) if n else 0.0,
        "natural_donor_gain_positive_count": gain_pos,
        "natural_donor_gain_positive_ratio": (gain_pos / n) if n else 0.0,
        "natural_all_selected_positive": bool(n > 0 and len(signed) == n and pos == n),
        "natural_any_selected_negative": bool(neg > 0),
    }


def eval_case(
    model,
    tokenizer,
    device: torch.device,
    standards: list[dict[str, Any]],
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lookup = p828.p820.standard_lookup(standards)
    recipient_prompt = p828.p825.natural_prompt(case, args.recipient_prompt)
    recipient_ids = p828.p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p828.greedy_generate_with_multi_patch(
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
    no_patch_logits = p834.first_step_logits(model, device, recipient_ids, [], float(args.alpha))
    no_patch_rank_profile = p834.rank_profile(no_patch_logits, target_id, baseline_id)
    no_patch_rank_profile["top_token"] = (
        tokenizer.decode([int(no_patch_rank_profile["top_token_id"])])
        if no_patch_rank_profile.get("top_token_id") is not None
        else None
    )

    labels = labels_from_candidates(candidates)
    comp_cache: dict[str, dict[str, Any]] = {}
    for label in labels:
        group = groups.get(label)
        source = source_rows.get(label)
        if not group or not source:
            continue
        comp = prepare_component_data(model, tokenizer, device, case, group, source, recipient_prompt, baseline_ids)
        if comp is not None:
            comp_cache[label] = comp

    specs = combo_specs(candidates, args)
    rows: list[dict[str, Any]] = []
    for donor_variant in parse_csv(args.search_donor_prompts):
        patch_cache: dict[str, dict[str, Any] | None] = {}
        feature_cache: dict[str, dict[str, Any]] = {}
        for label in comp_cache:
            item, features = donor_patch_item_with_features(
                model,
                tokenizer,
                device,
                case,
                donor_variant,
                groups[label],
                comp_cache[label],
            )
            patch_cache[label] = item
            if features is not None:
                feature_cache[label] = features

        for spec in specs:
            labels_for_combo = list(spec["labels"])
            patch_items = []
            missing = False
            for label in labels_for_combo:
                item = patch_cache.get(label)
                if item is None:
                    missing = True
                    break
                patch_items.append(item)
            if missing or not patch_items:
                continue

            patched_text, patched_ids = p828.greedy_generate_with_multi_patch(
                model,
                tokenizer,
                device,
                recipient_ids,
                patch_items,
                int(args.max_new_tokens),
                float(args.alpha),
            )
            patched_boundary = p828.p825.boundary_for(lookup, case["case_id"], patched_text)
            patch_logits = p834.first_step_logits(model, device, recipient_ids, patch_items, float(args.alpha))
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
            natural_features = natural_combo_features(feature_cache, labels_for_combo, args)
            row: dict[str, Any] = {
                "row_kind": "phase840_strict_target_interaction_natural_coactivation",
                "phase": PHASE,
                "model": args.model,
                "round": args.round_name,
                "source_phase": "phase839_strict_target_positive_interaction",
                "case_id": case["case_id"],
                "object": case["object"],
                "target_answer": case["answer"],
                "donor_variant": donor_variant,
                "recipient_prompt": args.recipient_prompt,
                "combo_kind": spec["combo_kind"],
                "combo_source": spec.get("source"),
                "combo_labels": labels_for_combo,
                "combo_label": combo_label(labels_for_combo),
                "n_components": len(labels_for_combo),
                "seed_summary": spec.get("seed_summary"),
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
                "component_natural_features": {label: feature_cache.get(label) for label in labels_for_combo},
                **natural_features,
                **rank_features,
                **p837.profile_delta("patch", baseline_span, patched_span),
            }
            row["response_type"] = p837.classify_response(row)
            row["response_vector"] = p838.row_vector(row)
            rows.append(row)
    p839.add_interaction_features(rows, args)
    for row in rows:
        row["strict_target_positive_interaction"] = bool(row.get("target_transition") and row.get("positive_interaction_edge"))
        row["natural_supported_strict_interaction"] = bool(
            row.get("strict_target_positive_interaction") and row.get("natural_all_selected_positive")
        )
    return rows


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
        "strict_positive_rows": sum(1 for row in rows if row.get("strict_target_positive_interaction")),
        "natural_supported_strict_rows": sum(1 for row in rows if row.get("natural_supported_strict_interaction")),
        "object_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "object_echo"),
        "format_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "format_echo"),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "minimal_sufficient_rows": sum(1 for row in rows if row.get("minimal_sufficient_candidate")),
        "mean_quality": avg_vec(rows, "target_quality_score"),
        "mean_echo_risk": avg_vec(rows, "echo_risk_score"),
        "mean_natural_positive_ratio": avg_field(rows, "natural_positive_component_ratio"),
        "classes": dict(Counter(str(row.get("patched_boundary_class")) for row in rows)),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    labels: list[str],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    attn_impl: str | None,
) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_combo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[str(row.get("combo_kind"))].append(row)
        by_combo[str(row.get("combo_label"))].append(row)

    combo_records = []
    for label, vals in by_combo.items():
        combo_records.append(
            {
                "combo_label": label,
                "combo_kind": vals[0].get("combo_kind"),
                "combo_source": vals[0].get("combo_source"),
                "n_components": vals[0].get("n_components"),
                "n_rows": len(vals),
                "target_rows": sum(1 for row in vals if row.get("target_transition")),
                "strict_positive_rows": sum(1 for row in vals if row.get("strict_target_positive_interaction")),
                "natural_supported_strict_rows": sum(1 for row in vals if row.get("natural_supported_strict_interaction")),
                "minimal_sufficient_rows": sum(1 for row in vals if row.get("minimal_sufficient_candidate")),
                "mean_quality": avg_vec(vals, "target_quality_score"),
                "mean_echo_risk": avg_vec(vals, "echo_risk_score"),
                "mean_interaction_gain": avg_field(vals, "interaction_quality_gain"),
                "mean_natural_positive_ratio": avg_field(vals, "natural_positive_component_ratio"),
                "classes": dict(Counter(str(row.get("patched_boundary_class")) for row in vals)),
            }
        )
    combo_records.sort(
        key=lambda item: (
            int(item["strict_positive_rows"]),
            int(item["natural_supported_strict_rows"]),
            int(item["minimal_sufficient_rows"]),
            finite(item["mean_interaction_gain"]),
        ),
        reverse=True,
    )
    top_rows = sorted(
        [row for row in rows if row.get("strict_target_positive_interaction") or row.get("natural_supported_strict_interaction")],
        key=lambda row: (
            int(bool(row.get("natural_supported_strict_interaction"))),
            finite(row.get("interaction_quality_gain")),
            finite(row.get("natural_positive_component_ratio")),
        ),
        reverse=True,
    )[:60]
    skipped = not candidates
    summary = {
        "phase": PHASE,
        "title": "Strict Target Interaction Expansion and Natural Co-activation Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "skipped_model_load": skipped,
        "skip_reason": "no strict Phase 839 target-positive interaction candidates" if skipped else None,
        "n_rows": len(rows),
        "n_candidates": len(candidates),
        "n_component_labels": len(labels),
        "n_cases": len(cases),
        "strict_candidates": candidates,
        "component_labels": labels,
        "case_ids": [case["case_id"] for case in cases],
        "donor_variants": parse_csv(args.search_donor_prompts),
        "combo_kind_summary": {kind: compact_rows(vals) for kind, vals in sorted(by_kind.items())},
        "target_rows": sum(1 for row in rows if row.get("target_transition")),
        "strict_positive_rows": sum(1 for row in rows if row.get("strict_target_positive_interaction")),
        "natural_supported_strict_rows": sum(1 for row in rows if row.get("natural_supported_strict_interaction")),
        "minimal_sufficient_rows": sum(1 for row in rows if row.get("minimal_sufficient_candidate")),
        "object_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "object_echo"),
        "format_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "format_echo"),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "mean_quality": avg_vec(rows, "target_quality_score"),
        "mean_echo_risk": avg_vec(rows, "echo_risk_score"),
        "mean_harm_risk": avg_vec(rows, "harm_risk_score"),
        "mean_natural_positive_ratio": avg_field(rows, "natural_positive_component_ratio"),
        "combo_records": combo_records[:120],
        "top_strict_rows": [
            {
                "case_id": row.get("case_id"),
                "donor_variant": row.get("donor_variant"),
                "combo_kind": row.get("combo_kind"),
                "combo_label": row.get("combo_label"),
                "patched_boundary_class": row.get("patched_boundary_class"),
                "patched_generated": row.get("patched_generated"),
                "target_quality_score": finite((row.get("response_vector") or {}).get("target_quality_score")),
                "echo_risk_score": finite((row.get("response_vector") or {}).get("echo_risk_score")),
                "interaction_quality_gain": finite(row.get("interaction_quality_gain")),
                "minimal_sufficient_candidate": bool(row.get("minimal_sufficient_candidate")),
                "natural_positive_component_ratio": finite(row.get("natural_positive_component_ratio")),
                "natural_selected_signed_sum_total": row.get("natural_selected_signed_sum_total"),
                "natural_all_selected_positive": bool(row.get("natural_all_selected_positive")),
                "natural_supported_strict_interaction": bool(row.get("natural_supported_strict_interaction")),
            }
            for row in top_rows
        ],
        "boundary": (
            "This phase expands only strict target-positive Phase 839 interaction edges and audits no-patch "
            "natural donor-state co-activation. Natural co-activation is correlational support only, not causal "
            "natural-route proof without ablation."
        ),
    }
    return summary


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_strict_candidates(args.model, args)
    labels = labels_from_candidates(candidates)
    cases = selected_cases(args, candidates)
    groups = p839.load_phase837_groups(args.model)
    source_rows = {}
    for label in labels:
        group = groups.get(label)
        if not group:
            continue
        source = p837.source_row_for_group(args.model, group, args)
        if source is not None:
            source_rows[label] = source
    log(
        f"{args.model}/{args.round_name}: strict_candidates={len(candidates)} labels={len(labels)} "
        f"source_rows={len(source_rows)} cases={len(cases)} donors={parse_csv(args.search_donor_prompts)}"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "candidates": candidates,
                    "labels": labels,
                    "cases": [case["case_id"] for case in cases],
                    "source_rows": len(source_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"candidates": candidates, "labels": labels, "cases": cases}
    if not candidates or not labels or not source_rows or not cases:
        summary = summarize_rows([], candidates, labels, cases, args, None)
        if candidates and not source_rows:
            summary["skip_reason"] = "strict candidates exist but source rows are missing"
        p828.write_jsonl(out_dir / f"phase840_{args.model}_rows.jsonl", [])
        p828.write_json(out_dir / f"phase840_{args.model}_summary.json", summary)
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
            rows.extend(eval_case(model, tokenizer, device, standards, case, candidates, groups, source_rows, args))
            if idx % int(args.log_every) == 0 or idx == len(cases):
                log(f"{args.model}: evaluated cases {idx}/{len(cases)} rows={len(rows)}")
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, candidates, labels, cases, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase840_{args.model}_rows.jsonl", rows)
    p828.write_json(out_dir / f"phase840_{args.model}_summary.json", summary)
    printable = {
        "model": args.model,
        "round": args.round_name,
        "rows": summary["n_rows"],
        "candidates": summary["n_candidates"],
        "target_rows": summary["target_rows"],
        "strict_positive_rows": summary["strict_positive_rows"],
        "natural_supported_strict_rows": summary["natural_supported_strict_rows"],
        "minimal_sufficient_rows": summary["minimal_sufficient_rows"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{finite(value):.4f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 840 Strict Target Interaction and Natural Co-activation ({payload['round']})",
        "",
        "- Source: Phase 839 confirm strict target-positive interaction rows only.",
        "- Boundary: patch expansion plus natural donor-state co-activation audit; not natural causal ablation.",
        "",
        "## Model Summary",
        "",
        "| model | skipped | candidates | rows | cases | target | strict positive | natural-supported strict | minimal | object_echo | format_echo | mean quality | mean echo risk | mean natural positive ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {int(bool(data.get('skipped_model_load')))} | {data.get('n_candidates', 0)} | "
            f"{data.get('n_rows', 0)} | {data.get('n_cases', 0)} | {data.get('target_rows', 0)} | "
            f"{data.get('strict_positive_rows', 0)} | {data.get('natural_supported_strict_rows', 0)} | "
            f"{data.get('minimal_sufficient_rows', 0)} | {data.get('object_echo_rows', 0)} | "
            f"{data.get('format_echo_rows', 0)} | {fmt(data.get('mean_quality'))} | "
            f"{fmt(data.get('mean_echo_risk'))} | {fmt(data.get('mean_natural_positive_ratio'))} |"
        )
    lines += ["", "## Combo Records", ""]
    lines += ["| model | kind | combo | rows | target | strict | natural-supported | minimal | mean quality | mean echo | natural ratio | classes |"]
    lines += ["|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in data.get("combo_records") or []:
            lines.append(
                f"| {model_name} | `{row.get('combo_kind')}` | `{row.get('combo_label')}` | {row.get('n_rows', 0)} | "
                f"{row.get('target_rows', 0)} | {row.get('strict_positive_rows', 0)} | "
                f"{row.get('natural_supported_strict_rows', 0)} | {row.get('minimal_sufficient_rows', 0)} | "
                f"{fmt(row.get('mean_quality'))} | {fmt(row.get('mean_echo_risk'))} | "
                f"{fmt(row.get('mean_natural_positive_ratio'))} | `{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Top Strict Rows", ""]
    lines += ["| model | case | donor | kind | combo | class | output | quality | gain | echo | natural ratio | natural all+ | natural-supported | minimal |"]
    lines += ["|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in data.get("top_strict_rows") or []:
            output = str(row.get("patched_generated") or "").replace("|", "/")[:60]
            lines.append(
                f"| {model_name} | `{row.get('case_id')}` | `{row.get('donor_variant')}` | `{row.get('combo_kind')}` | "
                f"`{row.get('combo_label')}` | `{row.get('patched_boundary_class')}` | {output} | "
                f"{fmt(row.get('target_quality_score'))} | {fmt(row.get('interaction_quality_gain'))} | "
                f"{fmt(row.get('echo_risk_score'))} | {fmt(row.get('natural_positive_component_ratio'))} | "
                f"{int(bool(row.get('natural_all_selected_positive')))} | "
                f"{int(bool(row.get('natural_supported_strict_interaction')))} | "
                f"{int(bool(row.get('minimal_sufficient_candidate')))} |"
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
        path = out_dir / f"phase840_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase840_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase840_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=p828.MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--recipient-prompt", default="no_choices")
    parser.add_argument("--case-scope", choices=["candidate", "holdout", "candidate_plus_holdout", "all"], default="candidate_plus_holdout")
    parser.add_argument("--candidate-combo-kinds", default="pair")
    parser.add_argument("--minimal-only", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=4)
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
    parser.add_argument("--interaction-quality-threshold", type=float, default=0.05)
    parser.add_argument("--echo-tolerance", type=float, default=0.05)
    parser.add_argument("--harm-tolerance", type=float, default=0.05)
    parser.add_argument("--max-minimal-echo-risk", type=float, default=0.25)
    parser.add_argument("--max-minimal-harm-risk", type=float, default=0.05)
    parser.add_argument("--natural-positive-threshold", type=float, default=0.0)
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
