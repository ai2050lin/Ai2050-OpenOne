#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402
import phase903_protocol_continuation_field_mapping as p903  # noqa: E402
import phase909_l0_attention_source_span_eos_boundary_audit as p909  # noqa: E402
import phase910_prompt_preserving_termination_route_reconstruction as p910  # noqa: E402
import phase911_full_vocab_blocker_displacement_audit as p911  # noqa: E402
import phase912_finite_blocker_band_source_localization as p912  # noqa: E402
import phase913_route_preserving_blocker_band_disentanglement as p913  # noqa: E402
import phase918_l39_mlp_channel_a_blocker_suppressor_localization as p918  # noqa: E402


PHASE = 919
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase919_frozen_l39_signed_margin_group_transfer_validation")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(part) for part in parse_csv(raw)]


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def state_key(row: dict[str, Any]) -> str:
    parts = [
        row.get("case_id"),
        row.get("prompt_variant"),
        row.get("source_subset_key"),
        row.get("edit_mode"),
        row.get("eval_kind"),
        row.get("boundary_group_kind"),
        row.get("boundary_factor"),
    ]
    return "|".join(str(part) for part in parts)


def build_transfer_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for group_kind in parse_csv(args.up_groups):
        factors = parse_floats(args.eos_factors if group_kind.startswith("eos_") else args.margin_pos_factors)
        for factor in factors:
            specs.append(
                {
                    "control_label": f"frozen_L{args.target_layer}_{group_kind}_scale_{factor:g}",
                    "control_family": "frozen_l39_channel_amplify",
                    "control_kind": "frozen_source_channel_group_scale",
                    "layer_idx": int(args.target_layer),
                    "group_kind": group_kind,
                    "factor": float(factor),
                }
            )
    for group_kind in parse_csv(args.down_groups):
        factors = parse_floats(args.down_factors)
        for factor in factors:
            specs.append(
                {
                    "control_label": f"frozen_L{args.target_layer}_{group_kind}_scale_{factor:g}",
                    "control_family": "frozen_l39_channel_suppress",
                    "control_kind": "frozen_source_channel_group_scale",
                    "layer_idx": int(args.target_layer),
                    "group_kind": group_kind,
                    "factor": float(factor),
                }
            )
    return specs


def reconstruct_state(
    model,
    tokenizer,
    device: torch.device,
    groups: dict[str, list[int]],
    case_map: dict[str, dict[str, Any]],
    source_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    case = case_map.get(str(source_row.get("case_id")))
    if not case:
        return None
    prompt = p885.prompt_for_case(case, str(source_row.get("prompt_variant")))
    prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
    gears = p903.parse_gears(str(source_row.get("source_subset_key")))
    _prefix_logits, prefix_ids, prefix_text, _answer_seen = p901.logits_after_answer_prefix(
        model,
        tokenizer,
        device,
        prompt_ids,
        gears,
        str(source_row.get("edit_mode")),
        case,
        int(args.max_prefix_tokens),
        float(args.scale_up_factor),
    )
    current_ids = [int(x) for x in prompt_ids] + [int(x) for x in prefix_ids]
    answer_logits = p903.logits_plain(model, device, current_ids)
    answer_metrics = p903.state_metrics(tokenizer, answer_logits, groups)
    period_id = answer_metrics.get("period_best_id") or ((groups.get("period") or [None])[0])
    if period_id is None:
        return None
    period_ids = current_ids + [int(period_id)]
    _baseline_logits, base_vec = p910.logits_and_l0_vector(model, device, period_ids)
    prompt_zero_handles = p909.install_attention_input_span_scale(model, 0, 0, len(prompt_ids), 0.0)
    _prompt_zero_logits, prompt_zero_vec = p910.logits_and_l0_vector(model, device, period_ids, prompt_zero_handles)
    if base_vec is None or prompt_zero_vec is None:
        return None
    route_delta = prompt_zero_vec - base_vec
    route_delta_norm = float(torch.linalg.vector_norm(route_delta).item())
    route_logits, l4_activation = p913.capture_route_logits_and_mlp_activation(model, device, period_ids, route_delta, 4)
    if route_logits is None:
        return None
    route_metrics = p903.state_metrics(tokenizer, route_logits, groups)
    route_top_rows = p910.topk_tokens(tokenizer, route_logits, groups, max(64, int(args.band_size)))
    route_band32_ids = p911.top_non_eos_ids(route_top_rows, int(args.band_size))
    route_band16_ids = route_band32_ids[: min(16, len(route_band32_ids))]
    l4_mlp_groups, l4_mlp_diag = p913.mlp_channel_groups_for_case(
        model,
        device,
        l4_activation,
        route_metrics.get("eos_best_id"),
        route_band16_ids,
        route_band32_ids,
        int(args.l4_candidate_pool),
    )
    boundary_factor = source_row.get("boundary_factor")
    if boundary_factor is None:
        return None
    boundary_spec = {
        "control_label": f"L4_mlp_channels_top_abs_64_scale_{float(boundary_factor):g}",
        "control_kind": "mlp_channel_group_scale",
        "layer_idx": 4,
        "group_kind": "top_abs_64",
        "factor": float(boundary_factor),
    }
    boundary_logits, l39_activation = p918.capture_boundary_logits_and_mlp_activation(
        model,
        device,
        period_ids,
        route_delta,
        boundary_spec,
        len(prompt_ids),
        len(prefix_ids),
        l4_mlp_groups,
        int(args.target_layer),
    )
    if boundary_logits is None:
        return None
    boundary_metrics = p903.state_metrics(tokenizer, boundary_logits, groups)
    boundary_top_rows = p910.topk_tokens(tokenizer, boundary_logits, groups, max(64, int(args.band_size)))
    boundary_blocker = p910.first_non_eos_top(boundary_top_rows)
    boundary_blocker_ids = p911.top_non_eos_ids(boundary_top_rows, int(args.band_size))
    channel_groups, channel_diag = p918.channel_groups_for_boundary_case(
        model,
        device,
        int(args.target_layer),
        l39_activation,
        boundary_metrics.get("eos_best_id"),
        boundary_blocker.get("token_id") if boundary_blocker else None,
        boundary_blocker_ids,
        int(args.channel_candidate_pool),
    )
    return {
        "state_key": state_key(source_row),
        "case": case,
        "source_row": dict(source_row),
        "prompt_ids": prompt_ids,
        "prefix_ids": [int(x) for x in prefix_ids],
        "prefix_text": prefix_text,
        "period_ids": period_ids,
        "route_delta": route_delta,
        "route_delta_norm": route_delta_norm,
        "route_metrics": route_metrics,
        "route_top_rows": route_top_rows,
        "route_band32_ids": route_band32_ids,
        "route_band16_ids": route_band16_ids,
        "l4_mlp_groups": l4_mlp_groups,
        "l4_mlp_diag": l4_mlp_diag,
        "boundary_spec": boundary_spec,
        "boundary_logits": boundary_logits,
        "boundary_metrics": boundary_metrics,
        "boundary_top_rows": boundary_top_rows,
        "boundary_blocker_ids": boundary_blocker_ids,
        "boundary_blocker": boundary_blocker,
        "channel_groups": channel_groups,
        "channel_diag": channel_diag,
    }


def logits_with_target_boundary_and_frozen_group(
    model,
    device: torch.device,
    target_state: dict[str, Any],
    source_group: list[int],
    layer_idx: int,
    factor: float,
) -> torch.Tensor | None:
    handles = p913.install_route_and_disentangle_hooks(
        model,
        target_state["route_delta"],
        target_state["boundary_spec"],
        len(target_state["prompt_ids"]),
        len(target_state["prefix_ids"]),
        len(target_state["period_ids"]),
        target_state["l4_mlp_groups"],
    )
    handles.extend(p913.install_mlp_channel_group_scale(model, int(layer_idx), source_group, float(factor)))
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, target_state["period_ids"])
    finally:
        for handle in handles:
            handle.remove()


def transfer_kind(source_state: dict[str, Any], target_state: dict[str, Any]) -> str:
    source_row = source_state["source_row"]
    target_row = target_state["source_row"]
    if source_state["state_key"] == target_state["state_key"]:
        return "self"
    if str(source_row.get("case_id")) == str(target_row.get("case_id")):
        return "cross_same_case"
    if str(source_row.get("eval_domain")) == str(target_row.get("eval_domain")):
        return "cross_same_domain"
    return "cross_domain"


def row_from_transfer(
    tokenizer,
    source_state: dict[str, Any],
    target_state: dict[str, Any],
    spec: dict[str, Any],
    source_group: list[int],
    patched_logits: torch.Tensor,
    groups: dict[str, list[int]],
) -> dict[str, Any]:
    source_row = source_state["source_row"]
    target_row = target_state["source_row"]
    boundary_logits = target_state["boundary_logits"]
    boundary_metrics = target_state["boundary_metrics"]
    boundary_top_rows = target_state["boundary_top_rows"]
    patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
    patched_top_rows = p910.topk_tokens(tokenizer, patched_logits, groups, 16)
    boundary_blocker = p910.first_non_eos_top(boundary_top_rows)
    patched_blocker = p910.first_non_eos_top(patched_top_rows)
    boundary_rank = boundary_metrics.get("eos_rank")
    patched_rank = patched_metrics.get("eos_rank")
    boundary_eos_logit = boundary_metrics.get("eos_best_logit")
    patched_eos_logit = patched_metrics.get("eos_best_logit")
    boundary_margin = p911.eos_margin_vs_blocker(boundary_metrics, boundary_blocker)
    patched_margin = p911.eos_margin_vs_blocker(patched_metrics, patched_blocker)
    boundary_blocker_id = boundary_blocker.get("token_id") if boundary_blocker else None
    boundary_blocker_logit = p911.token_logit(boundary_logits, boundary_blocker_id)
    boundary_blocker_after = p911.token_logit(patched_logits, boundary_blocker_id)
    rank_delta = None if boundary_rank is None or patched_rank is None else int(patched_rank) - int(boundary_rank)
    margin_delta = None if boundary_margin is None or patched_margin is None else float(patched_margin - boundary_margin)
    eos_delta = None if boundary_eos_logit is None or patched_eos_logit is None else float(patched_eos_logit - boundary_eos_logit)
    blocker_delta = None if boundary_blocker_logit is None or boundary_blocker_after is None else float(boundary_blocker_after - boundary_blocker_logit)
    band_before = p912.stats_for_ids(boundary_logits, target_state["boundary_blocker_ids"][:16])
    band_after = p912.stats_for_ids(patched_logits, target_state["boundary_blocker_ids"][:16])
    band16_delta = None if band_before["mean"] is None or band_after["mean"] is None else float(band_after["mean"] - band_before["mean"])
    eos_top1 = bool(patched_rank == 1)
    kind = transfer_kind(source_state, target_state)
    source_case = source_state["case"]
    target_case = target_state["case"]
    return {
        "phase": PHASE,
        "row_kind": "phase919_frozen_l39_signed_margin_group_transfer_row",
        "model": target_row.get("model"),
        "source_phase": 915,
        "source_state_key": source_state["state_key"],
        "target_state_key": target_state["state_key"],
        "transfer_kind": kind,
        "is_cross_transfer": bool(kind != "self"),
        "source_case_id": source_row.get("case_id"),
        "source_eval_domain": source_row.get("eval_domain"),
        "source_prompt_variant": source_row.get("prompt_variant"),
        "source_source_subset_key": source_row.get("source_subset_key"),
        "source_edit_mode": source_row.get("edit_mode"),
        "source_object": source_case.get("object"),
        "source_canonical_answer": source_case.get("canonical_answer"),
        "target_case_id": target_row.get("case_id"),
        "target_eval_domain": target_row.get("eval_domain"),
        "target_prompt_variant": target_row.get("prompt_variant"),
        "target_source_subset_key": target_row.get("source_subset_key"),
        "target_edit_mode": target_row.get("edit_mode"),
        "target_object": target_case.get("object"),
        "target_canonical_answer": target_case.get("canonical_answer"),
        "target_prefix_text": target_state["prefix_text"],
        "source_boundary_factor": source_row.get("boundary_factor"),
        "target_boundary_factor": target_row.get("boundary_factor"),
        "source_l39_source_control": source_row.get("control_label"),
        "target_boundary_group_kind": target_row.get("boundary_group_kind"),
        "control_label": spec.get("control_label"),
        "control_family": spec.get("control_family"),
        "control_kind": spec.get("control_kind"),
        "layer_idx": spec.get("layer_idx"),
        "group_kind": spec.get("group_kind"),
        "factor": spec.get("factor"),
        "neural_intervention": True,
        "prompt_input_intact": True,
        "prompt_all_zero_used_as_test_control": False,
        "target_route_delta_norm": target_state["route_delta_norm"],
        "target_boundary_eos_rank": boundary_rank,
        "target_boundary_eos_logit": boundary_eos_logit,
        "target_boundary_eos_margin_vs_blocker": boundary_margin,
        "target_boundary_blocker_id": boundary_blocker_id,
        "target_boundary_blocker_token": boundary_blocker.get("token") if boundary_blocker else None,
        "target_boundary_blocker_logit": boundary_blocker_logit,
        "patched_eos_rank": patched_rank,
        "patched_eos_logit": patched_eos_logit,
        "patched_eos_top1": eos_top1,
        "patched_eos_top5": bool(patched_rank is not None and int(patched_rank) <= 5),
        "patched_eos_top10": bool(patched_rank is not None and int(patched_rank) <= 10),
        "patched_eos_margin_vs_blocker": patched_margin,
        "patched_eos_margin_nonnegative": bool(patched_margin is not None and patched_margin >= 0),
        "patched_blocker_token": patched_blocker.get("token") if patched_blocker else None,
        "patched_blocker_logit": patched_blocker.get("logit") if patched_blocker else None,
        "eos_rank_delta_vs_target_boundary": rank_delta,
        "eos_logit_delta_vs_target_boundary": eos_delta,
        "eos_margin_delta_vs_target_boundary": margin_delta,
        "target_boundary_blocker_logit_after_patch": boundary_blocker_after,
        "target_boundary_blocker_logit_delta": blocker_delta,
        "target_boundary_band16_mean_delta": band16_delta,
        "target_boundary_blocker_suppressed": bool(blocker_delta is not None and blocker_delta < 0),
        "promoted_margin_from_negative": bool(
            boundary_margin is not None and boundary_margin < 0 and patched_margin is not None and patched_margin >= 0
        ),
        "promoted_top1_from_non_top1": bool(boundary_rank is not None and int(boundary_rank) > 1 and patched_rank == 1),
        "promoted_top5_from_non_top5": bool(
            boundary_rank is not None and int(boundary_rank) > 5 and patched_rank is not None and int(patched_rank) <= 5
        ),
        "rank_improved": bool(rank_delta is not None and rank_delta < 0),
        "weak_transfer_candidate": bool(
            rank_delta is not None
            and rank_delta < 0
            and eos_delta is not None
            and eos_delta >= 0
            and margin_delta is not None
            and margin_delta > 0
        ),
        "strict_clean_candidate": p911.strict_clean_candidate(tokenizer, target_case, target_state["prefix_ids"], eos_top1),
        "source_channel_group_size": len(source_group),
        "source_channel_group_preview": [int(x) for x in source_group[:16]],
        "target_native_group_overlap": len(
            set(int(x) for x in source_group)
            & set(int(x) for x in target_state["channel_groups"].get(str(spec.get("group_kind")), []))
        ),
        "target_native_group_size": len(target_state["channel_groups"].get(str(spec.get("group_kind")), [])),
        "source_channel_diag": source_state["channel_diag"],
        "target_channel_diag": target_state["channel_diag"],
        "target_boundary_top8": boundary_top_rows[:8],
        "patched_top8": patched_top_rows[:8],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "top1": sum(1 for row in rows if row.get("patched_eos_top1")),
        "top5": sum(1 for row in rows if row.get("patched_eos_top5")),
        "top10": sum(1 for row in rows if row.get("patched_eos_top10")),
        "margin_nonnegative": sum(1 for row in rows if row.get("patched_eos_margin_nonnegative")),
        "promoted_margin": sum(1 for row in rows if row.get("promoted_margin_from_negative")),
        "promoted_top1": sum(1 for row in rows if row.get("promoted_top1_from_non_top1")),
        "promoted_top5": sum(1 for row in rows if row.get("promoted_top5_from_non_top5")),
        "rank_improved": sum(1 for row in rows if row.get("rank_improved")),
        "weak_transfer_candidate": sum(1 for row in rows if row.get("weak_transfer_candidate")),
        "blocker_suppressed": sum(1 for row in rows if row.get("target_boundary_blocker_suppressed")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "median_margin_delta": median([row.get("eos_margin_delta_vs_target_boundary") for row in rows]),
        "mean_margin_delta": mean([row.get("eos_margin_delta_vs_target_boundary") for row in rows]),
        "mean_eos_delta": mean([row.get("eos_logit_delta_vs_target_boundary") for row in rows]),
        "median_blocker_delta": median([row.get("target_boundary_blocker_logit_delta") for row in rows]),
        "median_native_group_overlap": median([row.get("target_native_group_overlap") for row in rows]),
        "boundary_blocker_tokens_top12": dict(Counter(str(row.get("target_boundary_blocker_token")) for row in rows).most_common(12)),
        "patched_blocker_tokens_top12": dict(Counter(str(row.get("patched_blocker_token")) for row in rows).most_common(12)),
    }


def target_state_coverage(rows: list[dict[str, Any]], key: str) -> int:
    return len({str(row.get("target_state_key")) for row in rows if row.get(key)})


def summarize_by(rows: list[dict[str, Any]], keys: list[str], limit: int = 200) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key)) for key in keys)].append(row)
    out = []
    for key_tuple, vals in buckets.items():
        summary = summarize_rows(vals)
        first = vals[0]
        for name, value in zip(keys, key_tuple):
            summary[name] = value
        for meta_key in ["control_label", "control_family", "control_kind", "group_kind", "factor", "transfer_kind"]:
            summary.setdefault(meta_key, first.get(meta_key))
        summary["target_states_with_strict"] = target_state_coverage(vals, "strict_clean_candidate")
        summary["target_states_with_top1"] = target_state_coverage(vals, "patched_eos_top1")
        summary["target_states_with_margin"] = target_state_coverage(vals, "patched_eos_margin_nonnegative")
        summary["target_states_with_weak"] = target_state_coverage(vals, "weak_transfer_candidate")
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("strict_clean_candidate") or 0,
            row.get("top1") or 0,
            row.get("margin_nonnegative") or 0,
            row.get("promoted_top5") or 0,
            row.get("weak_transfer_candidate") or 0,
            row.get("rank_improved") or 0,
            row.get("median_margin_delta") or -9999,
        ),
        reverse=True,
    )
    return out[:limit]


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, specs_count: int, attn_impl: str | None) -> dict[str, Any]:
    self_rows = [row for row in rows if row.get("transfer_kind") == "self"]
    cross_rows = [row for row in rows if row.get("transfer_kind") != "self"]
    cross_same_case = [row for row in cross_rows if row.get("transfer_kind") == "cross_same_case"]
    cross_same_domain = [row for row in cross_rows if row.get("transfer_kind") == "cross_same_domain"]
    cross_domain = [row for row in cross_rows if row.get("transfer_kind") == "cross_domain"]
    overall = {
        "all": summarize_rows(rows),
        "self": summarize_rows(self_rows),
        "cross": summarize_rows(cross_rows),
        "cross_same_case": summarize_rows(cross_same_case),
        "cross_same_domain": summarize_rows(cross_same_domain),
        "cross_domain": summarize_rows(cross_domain),
        "target_state_count": len({str(row.get("target_state_key")) for row in rows}),
        "cross_target_states_with_strict": target_state_coverage(cross_rows, "strict_clean_candidate"),
        "cross_target_states_with_top1": target_state_coverage(cross_rows, "patched_eos_top1"),
        "cross_target_states_with_margin": target_state_coverage(cross_rows, "patched_eos_margin_nonnegative"),
        "cross_target_states_with_weak": target_state_coverage(cross_rows, "weak_transfer_candidate"),
    }
    if selected_count == 0:
        evidence = "no_phase915_l39_candidates"
    elif overall["cross"]["strict_clean_candidate"] > 0:
        evidence = "frozen_cross_strict_clean_transfer_found"
    elif overall["cross"]["top1"] > 0 or overall["cross"]["margin_nonnegative"] > 0:
        evidence = "frozen_cross_closure_transfer_found"
    elif overall["cross"]["weak_transfer_candidate"] > 0:
        evidence = "frozen_cross_weak_boundary_transfer_only"
    elif overall["self"]["strict_clean_candidate"] > 0 or overall["self"]["margin_nonnegative"] > 0:
        evidence = "self_only_case_conditioned_channel_effect"
    else:
        evidence = "no_frozen_channel_transfer_effect"
    return {
        "phase": PHASE,
        "title": "Frozen L39 Signed Margin Group Transfer Validation",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_phase915_l39_candidates": int(selected_count),
        "transfer_spec_count": int(specs_count),
        "overall": overall,
        "by_transfer_kind": summarize_by(rows, ["transfer_kind"]),
        "by_control": summarize_by(rows, ["control_label", "transfer_kind"]),
        "by_group": summarize_by(rows, ["group_kind", "transfer_kind"]),
        "by_source_state": summarize_by(rows, ["source_state_key", "transfer_kind"], limit=300),
        "evidence_label": evidence,
        "boundary": (
            "Phase919 freezes source-case L39 channel IDs from Phase918 and applies them to target-case "
            "route+L4 boundary states. It preserves target route/L4 reconstruction while testing whether "
            "the signed L39 channel group is transferable across states."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p918.select_phase915_candidates(args.model, args)
    specs = build_transfer_specs(args)
    if args.dry_run or not selected:
        payload = summarize_model(args.model, [], len(selected), len(specs), None)
        payload["status"] = "dry_run" if args.dry_run else "no_phase915_l39_candidates"
        payload["preview"] = selected[:20]
        p846.write_json(out_dir / f"phase919_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase919_{args.model}_rows.jsonl", [])
        print(json.dumps({"phase": PHASE, "model": args.model, "status": payload["status"], "selected": len(selected)}, ensure_ascii=False, indent=2), flush=True)
        return payload
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    states: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p903.protocol_category_groups(tokenizer)
        for idx, source_row in enumerate(selected, 1):
            state = reconstruct_state(model, tokenizer, device, groups, case_map, source_row, args)
            if state is not None:
                states.append(state)
            log(f"{args.model}/{args.round_name}: reconstructed_state={idx}/{len(selected)} kept={len(states)}")
        for source_idx, source_state in enumerate(states, 1):
            for target_idx, target_state in enumerate(states, 1):
                if not args.include_self_transfer and source_state["state_key"] == target_state["state_key"]:
                    continue
                for spec in specs:
                    group = source_state["channel_groups"].get(str(spec.get("group_kind"))) or []
                    if not group:
                        continue
                    patched_logits = logits_with_target_boundary_and_frozen_group(
                        model,
                        device,
                        target_state,
                        group,
                        int(spec.get("layer_idx") or args.target_layer),
                        float(spec.get("factor")),
                    )
                    if patched_logits is None:
                        continue
                    rows.append(row_from_transfer(tokenizer, source_state, target_state, spec, group, patched_logits, groups))
            if source_idx % max(1, int(args.log_every)) == 0 or source_idx == len(states):
                log(f"{args.model}/{args.round_name}: source={source_idx}/{len(states)} target_states={len(states)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        del states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = summarize_model(args.model, rows, len(selected), len(specs), attn_impl)
    p846.write_json(out_dir / f"phase919_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase919_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    evidence = Counter()
    scalar = Counter()
    controls = []
    groups = []
    transfers = []
    for model_name in MODELS:
        path = out_dir / f"phase919_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        scalar["selected_phase915_l39_candidates"] += int(summary.get("selected_phase915_l39_candidates") or 0)
        overall = summary.get("overall") or {}
        for scope in ["all", "self", "cross", "cross_same_case", "cross_same_domain", "cross_domain"]:
            scoped = overall.get(scope) or {}
            for key in ["rows", "top1", "margin_nonnegative", "weak_transfer_candidate", "strict_clean_candidate"]:
                scalar[f"{scope}_{key}"] += int(scoped.get(key) or 0)
        for key in ["target_state_count", "cross_target_states_with_strict", "cross_target_states_with_top1", "cross_target_states_with_margin", "cross_target_states_with_weak"]:
            scalar[key] += int(overall.get(key) or 0)
        for row in summary.get("by_control") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            controls.append(item)
        for row in summary.get("by_group") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            groups.append(item)
        for row in summary.get("by_transfer_kind") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            transfers.append(item)
    sort_keys = lambda row: (
        row.get("strict_clean_candidate") or 0,
        row.get("top1") or 0,
        row.get("margin_nonnegative") or 0,
        row.get("weak_transfer_candidate") or 0,
        row.get("median_margin_delta") or -9999,
    )
    controls.sort(key=sort_keys, reverse=True)
    groups.sort(key=sort_keys, reverse=True)
    transfers.sort(key=sort_keys, reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_transfers": transfers[:80],
        "top_controls": controls[:160],
        "top_groups": groups[:80],
    }
    p846.write_json(out_dir / "phase919_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase919_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 919 frozen L39 signed margin group transfer validation",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | selected | target states | cross rows | cross top1 | cross margin>=0 | cross weak | cross strict | cross targets top1 | cross targets margin | cross targets weak | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        cross = overall.get("cross") or {}
        lines.append(
            "| {model} | {selected} | {states} | {rows} | {top1} | {margin} | {weak} | {strict} | {targets_top1} | {targets_margin} | {targets_weak} | {evidence} |".format(
                model=summary.get("model"),
                selected=summary.get("selected_phase915_l39_candidates"),
                states=overall.get("target_state_count"),
                rows=cross.get("rows"),
                top1=cross.get("top1"),
                margin=cross.get("margin_nonnegative"),
                weak=cross.get("weak_transfer_candidate"),
                strict=cross.get("strict_clean_candidate"),
                targets_top1=overall.get("cross_target_states_with_top1"),
                targets_margin=overall.get("cross_target_states_with_margin"),
                targets_weak=overall.get("cross_target_states_with_weak"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Transfer Kinds", ""])
    lines.append(
        "| model | kind | rows | top1 | margin>=0 | weak | strict | median margin delta | mean eos delta | overlap median |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_transfers") or []:
        row = {
            "model": "",
            "transfer_kind": "",
            "rows": 0,
            "top1": 0,
            "margin_nonnegative": 0,
            "weak_transfer_candidate": 0,
            "strict_clean_candidate": 0,
            "median_margin_delta": None,
            "mean_eos_delta": None,
            "median_native_group_overlap": None,
            **row,
        }
        lines.append(
            "| {model} | {transfer_kind} | {rows} | {top1} | {margin_nonnegative} | {weak_transfer_candidate} | {strict_clean_candidate} | {median_margin_delta} | {mean_eos_delta} | {median_native_group_overlap} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Controls", ""])
    lines.append(
        "| model | control | kind | group | rows | top1 | margin>=0 | weak | strict | target states top1 | target states margin | median margin delta | mean eos delta | overlap median |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_controls") or []:
        row = {
            "model": "",
            "control_label": "",
            "transfer_kind": "",
            "group_kind": "",
            "rows": 0,
            "top1": 0,
            "margin_nonnegative": 0,
            "weak_transfer_candidate": 0,
            "strict_clean_candidate": 0,
            "target_states_with_top1": 0,
            "target_states_with_margin": 0,
            "median_margin_delta": None,
            "mean_eos_delta": None,
            "median_native_group_overlap": None,
            **row,
        }
        lines.append(
            "| {model} | {control_label} | {transfer_kind} | {group_kind} | {rows} | {top1} | {margin_nonnegative} | {weak_transfer_candidate} | {strict_clean_candidate} | {target_states_with_top1} | {target_states_with_margin} | {median_margin_delta} | {mean_eos_delta} | {median_native_group_overlap} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="frozen_l39_signed_margin_group_transfer_validation")
    parser.add_argument("--phase915-round", default="near_boundary_action_gate_search")
    parser.add_argument("--source-control-label", default="L39_mlp_output_scale_1.5")
    parser.add_argument("--boundary-blocker-token", default="a")
    parser.add_argument("--max-candidates-per-model", type=int, default=12)
    parser.add_argument("--target-layer", type=int, default=39)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--l4-candidate-pool", type=int, default=512)
    parser.add_argument("--channel-candidate-pool", type=int, default=768)
    parser.add_argument("--band-size", type=int, default=32)
    parser.add_argument("--up-groups", default="margin_support_pos_64,eos_support_64")
    parser.add_argument("--down-groups", default="a_blocker_support_64,margin_support_neg_64,a_logit_support_64")
    parser.add_argument("--margin-pos-factors", default="1.375,1.5,1.75,2.0")
    parser.add_argument("--eos-factors", default="1.75,2.0")
    parser.add_argument("--down-factors", default="0.0,0.125,0.25,0.375,0.5")
    parser.add_argument("--include-self-transfer", action="store_true")
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall_scalar"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
