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


PHASE = 841
RESULT_ROOT = Path("tests/result/phase841_signed_complementary_gear_role_validation")
SOURCE_840 = Path("tests/result/phase840_strict_target_interaction_natural_coactivation/confirm")
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


def sign_name(value: int) -> str:
    if value > 0:
        return "positive_carrier"
    if value < 0:
        return "negative_suppressor_or_rewriter"
    return "unknown_role"


def infer_role_signs(model_name: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(SOURCE_840 / f"phase840_{model_name}_rows.jsonl")
    strict_rows = [row for row in rows if row.get("strict_target_positive_interaction")]
    buckets: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    for row in strict_rows:
        for label, feat in (row.get("component_natural_features") or {}).items():
            if not isinstance(feat, dict):
                continue
            if feat.get("selected_signed_sum") is None:
                continue
            buckets[str(label)].append(finite(feat.get("selected_signed_sum")))
            meta.setdefault(
                str(label),
                {
                    "component_kind": feat.get("component_kind"),
                    "component_label": feat.get("component_label"),
                    "layer_idx": feat.get("layer_idx"),
                    "budget": feat.get("budget"),
                },
            )
    roles: dict[str, dict[str, Any]] = {}
    for label, vals in buckets.items():
        mean_signed = sum(vals) / len(vals)
        role_sign = 1 if mean_signed > 0 else -1 if mean_signed < 0 else 0
        roles[label] = {
            **meta.get(label, {}),
            "role_sign": role_sign,
            "role_name": sign_name(role_sign),
            "mean_strict_selected_signed_sum": mean_signed,
            "strict_observation_count": len(vals),
            "strict_signed_values": vals,
        }
    return roles


def combo_label(labels: list[str]) -> str:
    return " + ".join(labels)


def labels_from_candidates(candidates: list[dict[str, Any]]) -> list[str]:
    return p840.labels_from_candidates(candidates)


def selected_cases(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return p840.selected_cases(args, candidates)


def clone_item(item: dict[str, Any], alpha: float = 1.0) -> dict[str, Any]:
    out = dict(item)
    out["alpha"] = float(alpha)
    return out


def zero_item(item: dict[str, Any]) -> dict[str, Any]:
    out = clone_item(item, 1.0)
    donor_vec = out["donor_vec"].clone()
    for idx in out.get("selected_indices") or []:
        idx = int(idx)
        if 0 <= idx < int(donor_vec.numel()):
            donor_vec[idx] = 0.0
    out["donor_vec"] = donor_vec
    out["alpha"] = 1.0
    return out


def install_multi_patch_per_item(model, patch_items: list[dict[str, Any]], default_alpha: float) -> list[Any]:
    handles = []
    for item in patch_items:
        handles.append(
            p828.p823.install_subspace_patch(
                model,
                int(item["layer_idx"]),
                item["spec"],
                item["recipient_vec"],
                item["donor_vec"],
                item["selected_indices"],
                float(item.get("alpha", default_alpha)),
            )
        )
    return handles


def first_step_logits_per_item(
    model,
    device: torch.device,
    prompt_ids: list[int],
    patch_items: list[dict[str, Any]],
    default_alpha: float,
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles: list[Any] = []
    if patch_items:
        handles = install_multi_patch_per_item(model, patch_items, default_alpha)
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
    finally:
        for handle in handles:
            handle.remove()


def greedy_generate_per_item(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    patch_items: list[dict[str, Any]],
    max_new_tokens: int,
    default_alpha: float,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handles: list[Any] = []
        if step == 0 and patch_items:
            handles = install_multi_patch_per_item(model, patch_items, default_alpha)
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
        finally:
            for handle in handles:
                handle.remove()
        next_id = int(torch.argmax(logits).item())
        new_ids.append(next_id)
        current.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return tokenizer.decode(new_ids, skip_special_tokens=True), new_ids


def mode_patch_items(
    labels: list[str],
    patch_cache: dict[str, dict[str, Any]],
    roles: dict[str, dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    positives = [label for label in labels if int((roles.get(label) or {}).get("role_sign", 0)) > 0]
    negatives = [label for label in labels if int((roles.get(label) or {}).get("role_sign", 0)) < 0]

    def item(label: str, alpha: float = 1.0, zero: bool = False) -> dict[str, Any] | None:
        base = patch_cache.get(label)
        if base is None:
            return None
        return zero_item(base) if zero else clone_item(base, alpha)

    specs: list[tuple[str, float, bool]] = []
    if mode == "pair_original":
        specs = [(label, 1.0, False) for label in labels]
    elif mode == "positive_only":
        specs = [(label, 1.0, False) for label in positives]
    elif mode == "negative_only":
        specs = [(label, 1.0, False) for label in negatives]
    elif mode == "flip_positive":
        specs = [(label, -1.0 if label in positives else 1.0, False) for label in labels]
    elif mode == "flip_negative":
        specs = [(label, -1.0 if label in negatives else 1.0, False) for label in labels]
    elif mode == "zero_positive":
        specs = [(label, 1.0, label in positives) for label in labels]
    elif mode == "zero_negative":
        specs = [(label, 1.0, label in negatives) for label in labels]
    elif mode == "zero_all":
        specs = [(label, 1.0, True) for label in labels]
    else:
        raise ValueError(f"unknown mode: {mode}")

    out: list[dict[str, Any]] = []
    for label, alpha, zero in specs:
        patched = item(label, alpha, zero)
        if patched is not None:
            out.append(patched)
    return out


def natural_feature_cache_for_donor(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    donor_variant: str,
    labels: list[str],
    groups: dict[str, dict[str, Any]],
    comp_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    patch_cache: dict[str, dict[str, Any]] = {}
    feature_cache: dict[str, dict[str, Any]] = {}
    for label in labels:
        if label not in comp_cache or label not in groups:
            continue
        item, features = p840.donor_patch_item_with_features(
            model,
            tokenizer,
            device,
            case,
            donor_variant,
            groups[label],
            comp_cache[label],
        )
        if item is not None:
            patch_cache[label] = item
        if features is not None:
            feature_cache[label] = features
    return patch_cache, feature_cache


def eval_case(
    model,
    tokenizer,
    device: torch.device,
    standards: list[dict[str, Any]],
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    roles: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lookup = p828.p820.standard_lookup(standards)
    recipient_prompt = p828.p825.natural_prompt(case, args.recipient_prompt)
    recipient_ids = p828.p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = greedy_generate_per_item(
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
    no_patch_logits = first_step_logits_per_item(model, device, recipient_ids, [], float(args.alpha))
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
        comp = p840.prepare_component_data(model, tokenizer, device, case, group, source, recipient_prompt, baseline_ids)
        if comp is not None:
            comp_cache[label] = comp

    rows: list[dict[str, Any]] = []
    modes = parse_csv(args.modes)
    for donor_variant in parse_csv(args.search_donor_prompts):
        patch_cache, feature_cache = natural_feature_cache_for_donor(
            model, tokenizer, device, case, donor_variant, labels, groups, comp_cache
        )
        for cand_idx, cand in enumerate(candidates):
            combo_labels = [str(x) for x in cand.get("combo_labels") or []]
            role_pattern = {
                label: {
                    "role_sign": (roles.get(label) or {}).get("role_sign"),
                    "role_name": (roles.get(label) or {}).get("role_name"),
                    "strict_mean_signed_sum": (roles.get(label) or {}).get("mean_strict_selected_signed_sum"),
                    "current_selected_signed_sum": (feature_cache.get(label) or {}).get("selected_signed_sum"),
                    "current_selected_donor_minus_recipient_score": (feature_cache.get(label) or {}).get(
                        "selected_donor_minus_recipient_score"
                    ),
                }
                for label in combo_labels
            }
            natural_features = p840.natural_combo_features(feature_cache, combo_labels, args)
            for mode in modes:
                patch_items = mode_patch_items(combo_labels, patch_cache, roles, mode)
                if mode != "zero_all" and not patch_items:
                    continue
                patched_text, patched_ids = greedy_generate_per_item(
                    model,
                    tokenizer,
                    device,
                    recipient_ids,
                    patch_items,
                    int(args.max_new_tokens),
                    float(args.alpha),
                )
                patched_boundary = p828.p825.boundary_for(lookup, case["case_id"], patched_text)
                patch_logits = first_step_logits_per_item(model, device, recipient_ids, patch_items, float(args.alpha))
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
                row: dict[str, Any] = {
                    "row_kind": "phase841_signed_complementary_gear_role_validation",
                    "phase": PHASE,
                    "model": args.model,
                    "round": args.round_name,
                    "source_phase": "phase840_strict_pair_role_pattern",
                    "case_id": case["case_id"],
                    "object": case["object"],
                    "target_answer": case["answer"],
                    "donor_variant": donor_variant,
                    "recipient_prompt": args.recipient_prompt,
                    "candidate_index": cand_idx,
                    "combo_labels": combo_labels,
                    "combo_label": combo_label(combo_labels),
                    "combo_kind": cand.get("combo_kind"),
                    "mode": mode,
                    "mode_patch_item_count": len(patch_items),
                    "role_pattern": role_pattern,
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
                    "component_natural_features": {label: feature_cache.get(label) for label in combo_labels},
                    **natural_features,
                    **rank_features,
                    **p837.profile_delta("patch", baseline_span, patched_span),
                }
                row["response_type"] = p837.classify_response(row)
                row["response_vector"] = p838.row_vector(row)
                rows.append(row)

    add_mode_comparison_features(rows)
    return rows


def mode_group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("case_id")),
        str(row.get("donor_variant")),
        str(row.get("candidate_index")),
        str(row.get("combo_label")),
    )


def add_mode_comparison_features(rows: list[dict[str, Any]]) -> None:
    by_group: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_group[mode_group_key(row)][str(row.get("mode"))] = row
    for row in rows:
        original = by_group.get(mode_group_key(row), {}).get("pair_original")
        if not original:
            continue
        orig_vec = original.get("response_vector") or {}
        vec = row.get("response_vector") or {}
        row["original_target_transition"] = bool(original.get("target_transition"))
        row["original_patched_generated"] = original.get("patched_generated")
        row["original_patched_boundary_class"] = original.get("patched_boundary_class")
        row["delta_quality_vs_original"] = finite(vec.get("target_quality_score")) - finite(orig_vec.get("target_quality_score"))
        row["target_lost_vs_original"] = bool(original.get("target_transition") and not row.get("target_transition"))
        row["target_gained_vs_original"] = bool(row.get("target_transition") and not original.get("target_transition"))
        row["negative_role_needed_signal"] = bool(
            str(row.get("mode")) in {"positive_only", "flip_negative", "zero_negative", "zero_all"}
            and row.get("target_lost_vs_original")
        )
        row["positive_role_needed_signal"] = bool(
            str(row.get("mode")) in {"negative_only", "flip_positive", "zero_positive", "zero_all"}
            and row.get("target_lost_vs_original")
        )


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
        "target_lost_vs_original_rows": sum(1 for row in rows if row.get("target_lost_vs_original")),
        "negative_role_needed_rows": sum(1 for row in rows if row.get("negative_role_needed_signal")),
        "positive_role_needed_rows": sum(1 for row in rows if row.get("positive_role_needed_signal")),
        "object_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "object_echo"),
        "format_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "format_echo"),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "mean_quality": avg_vec(rows, "target_quality_score"),
        "mean_delta_quality_vs_original": avg_field(rows, "delta_quality_vs_original"),
        "classes": dict(Counter(str(row.get("patched_boundary_class")) for row in rows)),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    roles: dict[str, dict[str, Any]],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    attn_impl: str | None,
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("mode"))].append(row)
        by_case[str(row.get("case_id"))].append(row)
    top_rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("target_lost_vs_original") or False),
            finite(abs(row.get("delta_quality_vs_original") or 0.0)),
            finite((row.get("response_vector") or {}).get("target_quality_score")),
        ),
        reverse=True,
    )[:80]
    skipped = not candidates or not roles
    return {
        "phase": PHASE,
        "title": "Signed Complementary Gear Role Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "skipped_model_load": skipped,
        "skip_reason": "no Phase 840 strict role candidates" if skipped else None,
        "n_rows": len(rows),
        "n_candidates": len(candidates),
        "n_roles": len(roles),
        "n_cases": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "donor_variants": parse_csv(args.search_donor_prompts),
        "modes": parse_csv(args.modes),
        "strict_candidates": candidates,
        "role_signs": roles,
        "target_rows": sum(1 for row in rows if row.get("target_transition")),
        "pair_original_target_rows": sum(1 for row in rows if row.get("mode") == "pair_original" and row.get("target_transition")),
        "target_lost_vs_original_rows": sum(1 for row in rows if row.get("target_lost_vs_original")),
        "negative_role_needed_rows": sum(1 for row in rows if row.get("negative_role_needed_signal")),
        "positive_role_needed_rows": sum(1 for row in rows if row.get("positive_role_needed_signal")),
        "object_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "object_echo"),
        "format_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "format_echo"),
        "mode_summary": {mode: compact_rows(vals) for mode, vals in sorted(by_mode.items())},
        "case_summary": {case_id: compact_rows(vals) for case_id, vals in sorted(by_case.items())},
        "top_mode_rows": [
            {
                "case_id": row.get("case_id"),
                "donor_variant": row.get("donor_variant"),
                "mode": row.get("mode"),
                "combo_label": row.get("combo_label"),
                "patched_boundary_class": row.get("patched_boundary_class"),
                "patched_generated": row.get("patched_generated"),
                "target_transition": bool(row.get("target_transition")),
                "original_target_transition": bool(row.get("original_target_transition")),
                "target_lost_vs_original": bool(row.get("target_lost_vs_original")),
                "negative_role_needed_signal": bool(row.get("negative_role_needed_signal")),
                "positive_role_needed_signal": bool(row.get("positive_role_needed_signal")),
                "target_quality_score": finite((row.get("response_vector") or {}).get("target_quality_score")),
                "delta_quality_vs_original": row.get("delta_quality_vs_original"),
            }
            for row in top_rows
        ],
        "boundary": (
            "This phase tests role-signed necessity with patch-mode perturbations. It is still patch evidence, "
            "not natural-route ablation of an unpatched model."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = p840.load_strict_candidates(args.model, args)
    roles = infer_role_signs(args.model)
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
        f"{args.model}/{args.round_name}: candidates={len(candidates)} roles={len(roles)} "
        f"labels={len(labels)} source_rows={len(source_rows)} cases={len(cases)} modes={parse_csv(args.modes)}"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "candidates": candidates,
                    "roles": roles,
                    "labels": labels,
                    "cases": [case["case_id"] for case in cases],
                    "source_rows": len(source_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"candidates": candidates, "roles": roles, "labels": labels, "cases": cases}
    if not candidates or not roles or not labels or not source_rows or not cases:
        summary = summarize_rows([], candidates, roles, cases, args, None)
        if candidates and not source_rows:
            summary["skip_reason"] = "strict role candidates exist but source rows are missing"
        p828.write_jsonl(out_dir / f"phase841_{args.model}_rows.jsonl", [])
        p828.write_json(out_dir / f"phase841_{args.model}_summary.json", summary)
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
            rows.extend(eval_case(model, tokenizer, device, standards, case, candidates, roles, groups, source_rows, args))
            if idx % int(args.log_every) == 0 or idx == len(cases):
                log(f"{args.model}: evaluated cases {idx}/{len(cases)} rows={len(rows)}")
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, candidates, roles, cases, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase841_{args.model}_rows.jsonl", rows)
    p828.write_json(out_dir / f"phase841_{args.model}_summary.json", summary)
    printable = {
        "model": args.model,
        "round": args.round_name,
        "rows": summary["n_rows"],
        "candidates": summary["n_candidates"],
        "roles": summary["n_roles"],
        "pair_original_target_rows": summary["pair_original_target_rows"],
        "target_lost_vs_original_rows": summary["target_lost_vs_original_rows"],
        "negative_role_needed_rows": summary["negative_role_needed_rows"],
        "positive_role_needed_rows": summary["positive_role_needed_rows"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{finite(value):.4f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 841 Signed Complementary Gear Role Validation ({payload['round']})",
        "",
        "- Source: Phase 840 strict pair rows and inferred role signs.",
        "- Boundary: patch-mode perturbation evidence; not full natural ablation.",
        "",
        "## Model Summary",
        "",
        "| model | skipped | candidates | roles | rows | cases | pair-original target | target lost vs original | negative-role needed | positive-role needed | object_echo | format_echo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {int(bool(data.get('skipped_model_load')))} | {data.get('n_candidates', 0)} | "
            f"{data.get('n_roles', 0)} | {data.get('n_rows', 0)} | {data.get('n_cases', 0)} | "
            f"{data.get('pair_original_target_rows', 0)} | {data.get('target_lost_vs_original_rows', 0)} | "
            f"{data.get('negative_role_needed_rows', 0)} | {data.get('positive_role_needed_rows', 0)} | "
            f"{data.get('object_echo_rows', 0)} | {data.get('format_echo_rows', 0)} |"
        )
    lines += ["", "## Role Signs", ""]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        roles = data.get("role_signs") or {}
        if not roles:
            continue
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| component | role | mean signed sum | observations |")
        lines.append("|---|---|---:|---:|")
        for label, role in roles.items():
            lines.append(
                f"| `{label}` | `{role.get('role_name')}` | {fmt(role.get('mean_strict_selected_signed_sum'))} | "
                f"{role.get('strict_observation_count', 0)} |"
            )
        lines.append("")
    lines += ["## Mode Summary", ""]
    lines += ["| model | mode | n | target | lost vs original | negative needed | positive needed | mean quality | mean delta quality | classes |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for mode, row in (data.get("mode_summary") or {}).items():
            lines.append(
                f"| {model_name} | `{mode}` | {row.get('n', 0)} | {row.get('target_rows', 0)} | "
                f"{row.get('target_lost_vs_original_rows', 0)} | {row.get('negative_role_needed_rows', 0)} | "
                f"{row.get('positive_role_needed_rows', 0)} | {fmt(row.get('mean_quality'))} | "
                f"{fmt(row.get('mean_delta_quality_vs_original'))} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Top Mode Rows", ""]
    lines += ["| model | case | donor | mode | class | output | target | original target | lost | neg needed | pos needed | quality | delta quality |"]
    lines += ["|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in data.get("top_mode_rows") or []:
            output = str(row.get("patched_generated") or "").replace("|", "/")[:60]
            lines.append(
                f"| {model_name} | `{row.get('case_id')}` | `{row.get('donor_variant')}` | `{row.get('mode')}` | "
                f"`{row.get('patched_boundary_class')}` | {output} | {int(bool(row.get('target_transition')))} | "
                f"{int(bool(row.get('original_target_transition')))} | {int(bool(row.get('target_lost_vs_original')))} | "
                f"{int(bool(row.get('negative_role_needed_signal')))} | {int(bool(row.get('positive_role_needed_signal')))} | "
                f"{fmt(row.get('target_quality_score'))} | {fmt(row.get('delta_quality_vs_original'))} |"
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
        path = out_dir / f"phase841_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase841_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase841_cross_model_summary.md", payload)
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
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--search-donor-prompts", default="natural_question,object_only")
    parser.add_argument("--modes", default="pair_original,positive_only,negative_only,flip_positive,flip_negative,zero_positive,zero_negative")
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
