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
import phase903_protocol_continuation_field_mapping as p903  # noqa: E402
import phase909_l0_attention_source_span_eos_boundary_audit as p909  # noqa: E402
import phase910_prompt_preserving_termination_route_reconstruction as p910  # noqa: E402
import phase911_full_vocab_blocker_displacement_audit as p911  # noqa: E402
import phase912_finite_blocker_band_source_localization as p912  # noqa: E402
import phase913_route_preserving_blocker_band_disentanglement as p913  # noqa: E402
import phase918_l39_mlp_channel_a_blocker_suppressor_localization as p918  # noqa: E402
import phase919_frozen_l39_signed_margin_group_transfer_validation as p919  # noqa: E402
import phase920_consensus_l39_signed_margin_gear_holdout_controls as p920  # noqa: E402


PHASE = 922
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase922_candidate_gate_variable_causal_coupling_test")


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


def make_intervention_specs() -> list[dict[str, Any]]:
    return [
        {
            "control_label": "l39_only",
            "control_family": "l39_low_factor_baseline",
            "control_class": "baseline",
            "route_alpha": 1.0,
            "l4_factor_multiplier": 1.0,
        },
        {
            "control_label": "route_alpha_1.125",
            "control_family": "route_candidate_plus",
            "control_class": "candidate_plus",
            "route_alpha": 1.125,
            "l4_factor_multiplier": 1.0,
        },
        {
            "control_label": "route_alpha_1.25",
            "control_family": "route_candidate_plus",
            "control_class": "candidate_plus",
            "route_alpha": 1.25,
            "l4_factor_multiplier": 1.0,
        },
        {
            "control_label": "l4_boundary_1.05",
            "control_family": "l4_boundary_candidate_plus",
            "control_class": "candidate_plus",
            "route_alpha": 1.0,
            "l4_factor_multiplier": 1.05,
        },
        {
            "control_label": "l4_boundary_1.10",
            "control_family": "l4_boundary_candidate_plus",
            "control_class": "candidate_plus",
            "route_alpha": 1.0,
            "l4_factor_multiplier": 1.10,
        },
        {
            "control_label": "protocol_last8_0.90",
            "control_family": "protocol_pressure_candidate_suppress",
            "control_class": "candidate_plus",
            "route_alpha": 1.0,
            "l4_factor_multiplier": 1.0,
            "protocol_span_kind": "last8_before_period",
            "protocol_span_factor": 0.90,
        },
        {
            "control_label": "protocol_answer_last_0.90",
            "control_family": "protocol_pressure_candidate_suppress",
            "control_class": "candidate_plus",
            "route_alpha": 1.0,
            "l4_factor_multiplier": 1.0,
            "protocol_span_kind": "answer_prefix_last",
            "protocol_span_factor": 0.90,
        },
        {
            "control_label": "route_1.125_l4_1.05",
            "control_family": "route_l4_combo_candidate_plus",
            "control_class": "candidate_plus_combo",
            "route_alpha": 1.125,
            "l4_factor_multiplier": 1.05,
        },
        {
            "control_label": "route_1.125_protocol_last8_0.90",
            "control_family": "route_protocol_combo_candidate_plus",
            "control_class": "candidate_plus_combo",
            "route_alpha": 1.125,
            "l4_factor_multiplier": 1.0,
            "protocol_span_kind": "last8_before_period",
            "protocol_span_factor": 0.90,
        },
        {
            "control_label": "l4_1.05_protocol_last8_0.90",
            "control_family": "l4_protocol_combo_candidate_plus",
            "control_class": "candidate_plus_combo",
            "route_alpha": 1.0,
            "l4_factor_multiplier": 1.05,
            "protocol_span_kind": "last8_before_period",
            "protocol_span_factor": 0.90,
        },
        {
            "control_label": "route_1.125_l4_1.05_protocol_last8_0.90",
            "control_family": "route_l4_protocol_combo_candidate_plus",
            "control_class": "candidate_plus_combo",
            "route_alpha": 1.125,
            "l4_factor_multiplier": 1.05,
            "protocol_span_kind": "last8_before_period",
            "protocol_span_factor": 0.90,
        },
        {
            "control_label": "route_alpha_0.875_direction_control",
            "control_family": "route_direction_control",
            "control_class": "direction_control",
            "route_alpha": 0.875,
            "l4_factor_multiplier": 1.0,
        },
        {
            "control_label": "l4_boundary_0.95_direction_control",
            "control_family": "l4_boundary_direction_control",
            "control_class": "direction_control",
            "route_alpha": 1.0,
            "l4_factor_multiplier": 0.95,
        },
        {
            "control_label": "protocol_last8_1.10_direction_control",
            "control_family": "protocol_pressure_direction_control",
            "control_class": "direction_control",
            "route_alpha": 1.0,
            "l4_factor_multiplier": 1.0,
            "protocol_span_kind": "last8_before_period",
            "protocol_span_factor": 1.10,
        },
    ]


def adjusted_boundary_spec(state: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    boundary_spec = dict(state["boundary_spec"])
    multiplier = float(spec.get("l4_factor_multiplier") or 1.0)
    boundary_spec["factor"] = float(boundary_spec.get("factor") or 1.0) * multiplier
    boundary_spec["control_label"] = (
        f"L4_mlp_channels_{boundary_spec.get('group_kind')}_scale_{float(boundary_spec['factor']):g}"
    )
    return boundary_spec


def protocol_span_bounds(state: dict[str, Any], span_kind: str | None) -> tuple[int, int]:
    if not span_kind:
        return 0, 0
    return p909.span_bounds(
        str(span_kind),
        len(state["prompt_ids"]),
        len(state["prefix_ids"]),
        len(state["period_ids"]),
    )


def logits_with_coupled_intervention(
    model,
    device: torch.device,
    state: dict[str, Any],
    consensus_group: list[int],
    l39_factor: float,
    spec: dict[str, Any],
    target_layer: int,
) -> torch.Tensor | None:
    route_alpha = float(spec.get("route_alpha") or 1.0)
    route_delta = state["route_delta"] * route_alpha
    handles = p913.install_route_and_disentangle_hooks(
        model,
        route_delta,
        adjusted_boundary_spec(state, spec),
        len(state["prompt_ids"]),
        len(state["prefix_ids"]),
        len(state["period_ids"]),
        state["l4_mlp_groups"],
    )
    handles.extend(p913.install_mlp_channel_group_scale(model, int(target_layer), consensus_group, float(l39_factor)))
    span_kind = spec.get("protocol_span_kind")
    if span_kind is not None:
        span_start, span_end = protocol_span_bounds(state, str(span_kind))
        handles.extend(
            p909.install_attention_input_span_scale(
                model,
                0,
                int(span_start),
                int(span_end),
                float(spec.get("protocol_span_factor") or 1.0),
            )
        )
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, state["period_ids"])
    finally:
        for handle in handles:
            handle.remove()


def row_from_logits(
    tokenizer,
    state: dict[str, Any],
    consensus_group: list[int],
    l39_factor: float,
    spec: dict[str, Any],
    patched_logits: torch.Tensor,
    groups: dict[str, list[int]],
) -> dict[str, Any]:
    source_row = state["source_row"]
    case = state["case"]
    boundary_logits = state["boundary_logits"]
    boundary_metrics = state["boundary_metrics"]
    boundary_top_rows = state["boundary_top_rows"]
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
    band_before = p912.stats_for_ids(boundary_logits, state["boundary_blocker_ids"][:16])
    band_after = p912.stats_for_ids(patched_logits, state["boundary_blocker_ids"][:16])
    band16_delta = None if band_before["mean"] is None or band_after["mean"] is None else float(band_after["mean"] - band_before["mean"])
    eos_top1 = bool(patched_rank == 1)
    native_group = state["channel_groups"].get("margin_support_pos_64", [])
    span_start, span_end = protocol_span_bounds(state, spec.get("protocol_span_kind"))
    l4_original = float(state["boundary_spec"].get("factor") or 1.0)
    l4_multiplier = float(spec.get("l4_factor_multiplier") or 1.0)
    return {
        "phase": PHASE,
        "row_kind": "phase922_candidate_gate_variable_causal_coupling_row",
        "model": source_row.get("model"),
        "target_state_key": state["state_key"],
        "target_case_id": source_row.get("case_id"),
        "target_eval_domain": source_row.get("eval_domain"),
        "target_prompt_variant": source_row.get("prompt_variant"),
        "target_source_subset_key": source_row.get("source_subset_key"),
        "target_edit_mode": source_row.get("edit_mode"),
        "target_object": case.get("object"),
        "target_canonical_answer": case.get("canonical_answer"),
        "target_prefix_text": state["prefix_text"],
        "control_label": spec.get("control_label"),
        "control_family": spec.get("control_family"),
        "control_class": spec.get("control_class"),
        "l39_factor": float(l39_factor),
        "target_layer": 39,
        "route_alpha": float(spec.get("route_alpha") or 1.0),
        "l4_factor_original": l4_original,
        "l4_factor_multiplier": l4_multiplier,
        "l4_factor_effective": float(l4_original * l4_multiplier),
        "protocol_span_kind": spec.get("protocol_span_kind"),
        "protocol_span_factor": spec.get("protocol_span_factor"),
        "protocol_span_start": int(span_start),
        "protocol_span_end": int(span_end),
        "protocol_span_len": int(span_end) - int(span_start),
        "neural_intervention": True,
        "prompt_input_intact": True,
        "prompt_all_zero_used_as_test_control": False,
        "target_route_delta_norm": state["route_delta_norm"],
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
        "rank_improved_vs_target_boundary": bool(rank_delta is not None and rank_delta < 0),
        "weak_transfer_candidate": bool(
            rank_delta is not None
            and rank_delta < 0
            and eos_delta is not None
            and eos_delta >= 0
            and margin_delta is not None
            and margin_delta > 0
        ),
        "strict_clean_candidate": p911.strict_clean_candidate(tokenizer, case, state["prefix_ids"], eos_top1),
        "consensus_group_size": len(consensus_group),
        "consensus_group_preview": [int(x) for x in consensus_group[:16]],
        "target_native_margin_group_overlap": len(set(int(x) for x in consensus_group) & set(int(x) for x in native_group)),
        "target_native_margin_group_size": len(native_group),
        "target_boundary_top8": boundary_top_rows[:8],
        "patched_top8": patched_top_rows[:8],
    }


def annotate_vs_l39_only(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, float], dict[str, Any]] = {}
    for row in rows:
        if row.get("control_label") == "l39_only":
            baselines[(str(row.get("target_state_key")), float(row.get("l39_factor")))] = row
    for row in rows:
        base = baselines.get((str(row.get("target_state_key")), float(row.get("l39_factor"))))
        if base is None:
            continue
        row["l39_only_margin"] = base.get("patched_eos_margin_vs_blocker")
        row["l39_only_eos_rank"] = base.get("patched_eos_rank")
        row["l39_only_top1"] = base.get("patched_eos_top1")
        row["l39_only_margin_nonnegative"] = base.get("patched_eos_margin_nonnegative")
        row["l39_only_strict_clean_candidate"] = base.get("strict_clean_candidate")
        row_margin = row.get("patched_eos_margin_vs_blocker")
        base_margin = base.get("patched_eos_margin_vs_blocker")
        row_rank = row.get("patched_eos_rank")
        base_rank = base.get("patched_eos_rank")
        row["margin_delta_vs_l39_only"] = None if row_margin is None or base_margin is None else float(row_margin - base_margin)
        row["rank_delta_vs_l39_only"] = None if row_rank is None or base_rank is None else int(row_rank) - int(base_rank)
        row["improved_margin_vs_l39_only"] = bool(
            row["margin_delta_vs_l39_only"] is not None and row["margin_delta_vs_l39_only"] > 0
        )
        row["worsened_margin_vs_l39_only"] = bool(
            row["margin_delta_vs_l39_only"] is not None and row["margin_delta_vs_l39_only"] < 0
        )
        row["new_margin_closure_vs_l39_only"] = bool(
            not base.get("patched_eos_margin_nonnegative") and row.get("patched_eos_margin_nonnegative")
        )
        row["lost_margin_closure_vs_l39_only"] = bool(
            base.get("patched_eos_margin_nonnegative") and not row.get("patched_eos_margin_nonnegative")
        )
        row["new_top1_vs_l39_only"] = bool(not base.get("patched_eos_top1") and row.get("patched_eos_top1"))
        row["lost_top1_vs_l39_only"] = bool(base.get("patched_eos_top1") and not row.get("patched_eos_top1"))
        row["new_strict_vs_l39_only"] = bool(
            not base.get("strict_clean_candidate") and row.get("strict_clean_candidate")
        )
        row["lost_strict_vs_l39_only"] = bool(
            base.get("strict_clean_candidate") and not row.get("strict_clean_candidate")
        )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "top1": sum(1 for row in rows if row.get("patched_eos_top1")),
        "top5": sum(1 for row in rows if row.get("patched_eos_top5")),
        "margin_nonnegative": sum(1 for row in rows if row.get("patched_eos_margin_nonnegative")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "weak_transfer_candidate": sum(1 for row in rows if row.get("weak_transfer_candidate")),
        "rank_improved_vs_target_boundary": sum(1 for row in rows if row.get("rank_improved_vs_target_boundary")),
        "blocker_suppressed": sum(1 for row in rows if row.get("target_boundary_blocker_suppressed")),
        "improved_margin_vs_l39_only": sum(1 for row in rows if row.get("improved_margin_vs_l39_only")),
        "worsened_margin_vs_l39_only": sum(1 for row in rows if row.get("worsened_margin_vs_l39_only")),
        "new_margin_closure_vs_l39_only": sum(1 for row in rows if row.get("new_margin_closure_vs_l39_only")),
        "lost_margin_closure_vs_l39_only": sum(1 for row in rows if row.get("lost_margin_closure_vs_l39_only")),
        "new_top1_vs_l39_only": sum(1 for row in rows if row.get("new_top1_vs_l39_only")),
        "lost_top1_vs_l39_only": sum(1 for row in rows if row.get("lost_top1_vs_l39_only")),
        "new_strict_vs_l39_only": sum(1 for row in rows if row.get("new_strict_vs_l39_only")),
        "lost_strict_vs_l39_only": sum(1 for row in rows if row.get("lost_strict_vs_l39_only")),
        "median_margin_delta_vs_target_boundary": median([row.get("eos_margin_delta_vs_target_boundary") for row in rows]),
        "mean_margin_delta_vs_target_boundary": mean([row.get("eos_margin_delta_vs_target_boundary") for row in rows]),
        "median_margin_delta_vs_l39_only": median([row.get("margin_delta_vs_l39_only") for row in rows]),
        "mean_margin_delta_vs_l39_only": mean([row.get("margin_delta_vs_l39_only") for row in rows]),
        "median_patched_margin": median([row.get("patched_eos_margin_vs_blocker") for row in rows]),
        "median_blocker_delta": median([row.get("target_boundary_blocker_logit_delta") for row in rows]),
        "target_state_coverage_top1": len({row.get("target_state_key") for row in rows if row.get("patched_eos_top1")}),
        "target_state_coverage_margin": len(
            {row.get("target_state_key") for row in rows if row.get("patched_eos_margin_nonnegative")}
        ),
        "target_state_coverage_strict": len({row.get("target_state_key") for row in rows if row.get("strict_clean_candidate")}),
    }


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
        for meta_key in [
            "control_label",
            "control_family",
            "control_class",
            "l39_factor",
            "route_alpha",
            "l4_factor_multiplier",
            "protocol_span_kind",
            "protocol_span_factor",
        ]:
            summary.setdefault(meta_key, first.get(meta_key))
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("new_strict_vs_l39_only") or 0,
            row.get("new_top1_vs_l39_only") or 0,
            row.get("new_margin_closure_vs_l39_only") or 0,
            row.get("improved_margin_vs_l39_only") or 0,
            row.get("mean_margin_delta_vs_l39_only") or -9999,
        ),
        reverse=True,
    )
    return out[:limit]


def summarize_model(
    model_name: str,
    rows: list[dict[str, Any]],
    selected_count: int,
    spec_count: int,
    factor_count: int,
    consensus_diag: dict[str, Any] | None,
    attn_impl: str | None,
) -> dict[str, Any]:
    baseline_rows = [row for row in rows if row.get("control_label") == "l39_only"]
    coupled_rows = [row for row in rows if row.get("control_label") != "l39_only"]
    candidate_rows = [row for row in coupled_rows if str(row.get("control_class")).startswith("candidate")]
    direction_rows = [row for row in coupled_rows if row.get("control_class") == "direction_control"]
    overall = {
        "all": summarize_rows(rows),
        "l39_only": summarize_rows(baseline_rows),
        "coupled_nonbaseline": summarize_rows(coupled_rows),
        "candidate_plus": summarize_rows(candidate_rows),
        "direction_control": summarize_rows(direction_rows),
        "target_state_count": len({row.get("target_state_key") for row in rows}),
    }
    best_controls = summarize_by(rows, ["control_label"], limit=20)
    candidate_overall = overall["candidate_plus"]
    direction_overall = overall["direction_control"]
    candidate_mean_delta = candidate_overall.get("mean_margin_delta_vs_l39_only") or -9999
    candidate_new_margin = candidate_overall.get("new_margin_closure_vs_l39_only") or 0
    candidate_new_top1 = candidate_overall.get("new_top1_vs_l39_only") or 0
    direction_new_margin = direction_overall.get("new_margin_closure_vs_l39_only") or 0
    direction_new_top1 = direction_overall.get("new_top1_vs_l39_only") or 0
    if selected_count == 0:
        evidence = "no_phase915_l39_candidates"
    elif candidate_new_top1 > 0 or candidate_new_margin > 0:
        evidence = "candidate_gate_intervention_adds_low_factor_closure"
    elif (direction_new_top1 > 0 or direction_new_margin > 0) and candidate_mean_delta > 0:
        evidence = "candidate_moves_margin_but_direction_control_only_adds_closure"
    elif candidate_mean_delta > 0:
        evidence = "candidate_gate_intervention_moves_margin_without_extra_closure"
    else:
        evidence = "no_positive_coupling_over_l39_low_factor"
    return {
        "phase": PHASE,
        "title": "Candidate Gate Variable Causal Coupling Test",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_phase915_l39_candidates": int(selected_count),
        "intervention_spec_count": int(spec_count),
        "low_factor_count": int(factor_count),
        "consensus_diag": consensus_diag or {},
        "overall": overall,
        "by_control": best_controls,
        "by_factor": summarize_by(rows, ["l39_factor"], limit=50),
        "by_control_factor": summarize_by(rows, ["control_label", "l39_factor"], limit=200),
        "by_family": summarize_by(rows, ["control_family"], limit=80),
        "evidence_label": evidence,
        "boundary": (
            "Phase922 fixes the Phase920 consensus L39 margin gear and tests whether simple candidate "
            "gate variables from Phase921 causally couple with low-factor L39 intervention. It is not a "
            "natural gate closure test; it is a causal-coupling screen around route, L4 boundary, and "
            "protocol pressure variables."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p918.select_phase915_candidates(args.model, args)
    specs = make_intervention_specs()
    low_factors = parse_floats(args.low_factors)
    if args.dry_run or not selected:
        payload = summarize_model(args.model, [], len(selected), len(specs), len(low_factors), {}, None)
        payload["status"] = "dry_run" if args.dry_run else "no_phase915_l39_candidates"
        p846.write_json(out_dir / f"phase922_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase922_{args.model}_rows.jsonl", [])
        print(
            json.dumps(
                {"phase": PHASE, "model": args.model, "status": payload["status"], "selected": len(selected)},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return payload
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    states: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    consensus_diag: dict[str, Any] = {}
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p903.protocol_category_groups(tokenizer)
        for idx, source_row in enumerate(selected, 1):
            state = p919.reconstruct_state(model, tokenizer, device, groups, case_map, source_row, args)
            if state is not None:
                states.append(state)
            log(f"{args.model}/{args.round_name}: reconstructed_state={idx}/{len(selected)} kept={len(states)}")
        consensus_group, consensus_diag = p920.consensus_group(states, "margin_support_pos_64", int(args.group_budget))
        for state_idx, state in enumerate(states, 1):
            for l39_factor in low_factors:
                for spec in specs:
                    patched_logits = logits_with_coupled_intervention(
                        model,
                        device,
                        state,
                        consensus_group,
                        float(l39_factor),
                        spec,
                        int(args.target_layer),
                    )
                    if patched_logits is None:
                        continue
                    rows.append(row_from_logits(tokenizer, state, consensus_group, float(l39_factor), spec, patched_logits, groups))
            if state_idx % max(1, int(args.log_every)) == 0 or state_idx == len(states):
                log(f"{args.model}/{args.round_name}: coupled_state={state_idx}/{len(states)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        del states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    annotate_vs_l39_only(rows)
    payload = summarize_model(args.model, rows, len(selected), len(specs), len(low_factors), consensus_diag, attn_impl)
    p846.write_json(out_dir / f"phase922_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase922_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    evidence = Counter()
    scalar = Counter()
    controls = []
    factors = []
    control_factors = []
    families = []
    for model_name in MODELS:
        path = out_dir / f"phase922_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        scalar["selected_phase915_l39_candidates"] += int(summary.get("selected_phase915_l39_candidates") or 0)
        overall = summary.get("overall") or {}
        for scope in ["all", "l39_only", "coupled_nonbaseline", "candidate_plus", "direction_control"]:
            scoped = overall.get(scope) or {}
            for key in [
                "rows",
                "top1",
                "margin_nonnegative",
                "strict_clean_candidate",
                "improved_margin_vs_l39_only",
                "new_margin_closure_vs_l39_only",
                "new_top1_vs_l39_only",
                "new_strict_vs_l39_only",
            ]:
                scalar[f"{scope}_{key}"] += int(scoped.get(key) or 0)
        scalar["target_state_count"] += int(overall.get("target_state_count") or 0)
        for source_key, target in [
            ("by_control", controls),
            ("by_factor", factors),
            ("by_control_factor", control_factors),
            ("by_family", families),
        ]:
            for row in summary.get(source_key) or []:
                item = dict(row)
                item["model"] = summary.get("model")
                target.append(item)
    sort_keys = lambda row: (
        row.get("new_strict_vs_l39_only") or 0,
        row.get("new_top1_vs_l39_only") or 0,
        row.get("new_margin_closure_vs_l39_only") or 0,
        row.get("improved_margin_vs_l39_only") or 0,
        row.get("mean_margin_delta_vs_l39_only") or -9999,
    )
    controls.sort(key=sort_keys, reverse=True)
    factors.sort(key=sort_keys, reverse=True)
    control_factors.sort(key=sort_keys, reverse=True)
    families.sort(key=sort_keys, reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_controls": controls[:120],
        "top_factors": factors[:40],
        "top_control_factors": control_factors[:160],
        "top_families": families[:80],
    }
    p846.write_json(out_dir / "phase922_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase922_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 922 candidate gate variable causal coupling test",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | selected | states | l39 rows | l39 top1 | l39 margin | l39 strict | candidate rows | candidate new margin | candidate new top1 | candidate mean delta | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        l39 = overall.get("l39_only") or {}
        cand = overall.get("candidate_plus") or {}
        lines.append(
            "| {model} | {selected} | {states} | {l39_rows} | {l39_top1} | {l39_margin} | {l39_strict} | {cand_rows} | {cand_new_margin} | {cand_new_top1} | {cand_mean_delta} | {evidence} |".format(
                model=summary.get("model"),
                selected=summary.get("selected_phase915_l39_candidates"),
                states=overall.get("target_state_count"),
                l39_rows=l39.get("rows"),
                l39_top1=l39.get("top1"),
                l39_margin=l39.get("margin_nonnegative"),
                l39_strict=l39.get("strict_clean_candidate"),
                cand_rows=cand.get("rows"),
                cand_new_margin=cand.get("new_margin_closure_vs_l39_only"),
                cand_new_top1=cand.get("new_top1_vs_l39_only"),
                cand_mean_delta=cand.get("mean_margin_delta_vs_l39_only"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Controls", ""])
    lines.append(
        "| model | control | class | family | rows | top1 | margin | strict | improved | new margin | new top1 | new strict | lost margin | mean delta vs l39 | median patched margin |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_controls") or []:
        row = {
            "model": "",
            "control_label": "",
            "control_class": "",
            "control_family": "",
            "rows": 0,
            "top1": 0,
            "margin_nonnegative": 0,
            "strict_clean_candidate": 0,
            "improved_margin_vs_l39_only": 0,
            "new_margin_closure_vs_l39_only": 0,
            "new_top1_vs_l39_only": 0,
            "new_strict_vs_l39_only": 0,
            "lost_margin_closure_vs_l39_only": 0,
            "mean_margin_delta_vs_l39_only": None,
            "median_patched_margin": None,
            **row,
        }
        lines.append(
            "| {model} | {control_label} | {control_class} | {control_family} | {rows} | {top1} | {margin_nonnegative} | {strict_clean_candidate} | {improved_margin_vs_l39_only} | {new_margin_closure_vs_l39_only} | {new_top1_vs_l39_only} | {new_strict_vs_l39_only} | {lost_margin_closure_vs_l39_only} | {mean_margin_delta_vs_l39_only} | {median_patched_margin} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Control Factors", ""])
    lines.append(
        "| model | control | factor | rows | top1 | margin | strict | improved | new margin | new top1 | mean delta vs l39 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_control_factors") or []:
        row = {
            "model": "",
            "control_label": "",
            "l39_factor": "",
            "rows": 0,
            "top1": 0,
            "margin_nonnegative": 0,
            "strict_clean_candidate": 0,
            "improved_margin_vs_l39_only": 0,
            "new_margin_closure_vs_l39_only": 0,
            "new_top1_vs_l39_only": 0,
            "mean_margin_delta_vs_l39_only": None,
            **row,
        }
        lines.append(
            "| {model} | {control_label} | {l39_factor} | {rows} | {top1} | {margin_nonnegative} | {strict_clean_candidate} | {improved_margin_vs_l39_only} | {new_margin_closure_vs_l39_only} | {new_top1_vs_l39_only} | {mean_margin_delta_vs_l39_only} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="candidate_gate_variable_causal_coupling_test")
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
    parser.add_argument("--group-budget", type=int, default=64)
    parser.add_argument("--low-factors", default="1.125,1.25,1.375")
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
