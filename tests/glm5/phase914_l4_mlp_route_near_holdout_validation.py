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
import phase906_eos_action_boundary_test as p906  # noqa: E402
import phase909_l0_attention_source_span_eos_boundary_audit as p909  # noqa: E402
import phase910_prompt_preserving_termination_route_reconstruction as p910  # noqa: E402
import phase911_full_vocab_blocker_displacement_audit as p911  # noqa: E402
import phase912_finite_blocker_band_source_localization as p912  # noqa: E402
import phase913_route_preserving_blocker_band_disentanglement as p913  # noqa: E402


PHASE = 914
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase914_l4_mlp_route_near_holdout_validation")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_factors(raw: str) -> list[float]:
    values = []
    for part in parse_csv(raw):
        values.append(float(part))
    return values or [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def eval_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("case_id"),
        row.get("prompt_variant"),
        row.get("source_subset_key"),
        row.get("edit_mode"),
        row.get("eval_kind"),
    )


def diverse_limit(rows: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if max_items <= 0 or len(rows) <= max_items:
        return rows
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                row.get("eval_domain"),
                row.get("eval_kind"),
                row.get("source_subset_key"),
                row.get("edit_mode"),
            )
        ].append(row)
    for vals in buckets.values():
        vals.sort(
            key=lambda row: (
                str(row.get("case_id")),
                str(row.get("prompt_variant")),
            )
        )
    ordered_keys = sorted(buckets, key=lambda key: tuple(str(part) for part in key))
    selected: list[dict[str, Any]] = []
    while len(selected) < max_items and ordered_keys:
        next_keys = []
        for key in ordered_keys:
            vals = buckets[key]
            if vals:
                selected.append(vals.pop(0))
                if len(selected) >= max_items:
                    break
            if vals:
                next_keys.append(key)
        ordered_keys = next_keys
    selected.sort(
        key=lambda row: (
            str(row.get("eval_domain")),
            str(row.get("eval_kind")),
            str(row.get("source_subset_key")),
            str(row.get("edit_mode")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
        )
    )
    return selected


def expand_eval_rows(model_name: str, args: argparse.Namespace, case_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base_rows = p906.selected_phase899_rows(model_name, args)
    prompt_variants = parse_csv(args.prompt_variants)
    holdout_prompt_variants = parse_csv(args.holdout_prompt_variants)
    all_cases = list(case_map.values())
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add_row(source_row: dict[str, Any], case: dict[str, Any], prompt_variant: str, eval_kind: str) -> None:
        item = dict(source_row)
        item.update(
            {
                "case_id": case.get("case_id"),
                "object": case.get("object"),
                "canonical_answer": case.get("canonical_answer"),
                "eval_domain": case.get("domain"),
                "prompt_variant": prompt_variant,
                "eval_kind": eval_kind,
                "case_split": eval_kind,
            }
        )
        key = eval_key(item)
        if key in seen:
            return
        seen.add(key)
        out.append(item)

    for source_row in base_rows:
        case = case_map.get(str(source_row.get("case_id")))
        if not case:
            continue
        variants = prompt_variants or [str(source_row.get("prompt_variant"))]
        for variant in variants:
            add_row(source_row, case, variant, "source_case_prompt_variant")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for source_row in base_rows:
        grouped[(str(source_row.get("eval_domain")), str(source_row.get("source_subset_key")), str(source_row.get("edit_mode")))].append(source_row)

    max_holdout = int(args.same_domain_holdout_per_domain)
    for (domain, _subset_key, _edit_mode), source_rows in grouped.items():
        source_row = source_rows[0]
        source_case_ids = {str(row.get("case_id")) for row in source_rows}
        cases = [
            case
            for case in all_cases
            if str(case.get("domain")) == domain and str(case.get("case_id")) not in source_case_ids
        ]
        cases.sort(key=lambda case: (0 if str(case.get("split_source")) == "phase856_base" else 1, str(case.get("case_id"))))
        for case in cases[:max_holdout]:
            for variant in holdout_prompt_variants:
                add_row(source_row, case, variant, "same_domain_holdout_case")

    max_items = int(args.max_eval_items_per_model)
    out.sort(
        key=lambda row: (
            str(row.get("eval_domain")),
            str(row.get("eval_kind")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("source_subset_key")),
            str(row.get("edit_mode")),
        )
    )
    return diverse_limit(out, max_items)


def mlp_specs(factors: list[float], group_kinds: list[str]) -> list[dict[str, Any]]:
    specs = [
        {
            "control_label": "route_only_alpha_1",
            "control_kind": "route_only",
            "subunit_family": "route_only",
            "factor": None,
            "group_kind": None,
        }
    ]
    for factor in factors:
        for group_kind in group_kinds:
            specs.append(
                {
                    "control_label": f"L4_mlp_channels_{group_kind}_scale_{factor:g}",
                    "control_kind": "mlp_channel_group_scale",
                    "subunit_family": "l4_mlp_channel_group",
                    "layer_idx": 4,
                    "group_kind": group_kind,
                    "factor": float(factor),
                }
            )
    return specs


def row_from_logits(
    tokenizer,
    source_row: dict[str, Any],
    case: dict[str, Any],
    spec: dict[str, Any],
    prefix_ids: list[int],
    prefix_text: str,
    route_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    route_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    route_top_rows: list[dict[str, Any]],
    patched_top_rows: list[dict[str, Any]],
    band16_ids: list[int],
    band32_ids: list[int],
    route_delta_norm: float,
    mlp_groups: dict[str, list[int]],
    mlp_diag: dict[str, Any],
) -> dict[str, Any]:
    route_band16 = p912.stats_for_ids(route_logits, band16_ids)
    route_band32 = p912.stats_for_ids(route_logits, band32_ids)
    patched_band16 = p912.stats_for_ids(patched_logits, band16_ids)
    patched_band32 = p912.stats_for_ids(patched_logits, band32_ids)
    route_eos_logit = route_metrics.get("eos_best_logit")
    patched_eos_logit = patched_metrics.get("eos_best_logit")
    route_eos_rank = route_metrics.get("eos_rank")
    patched_eos_rank = patched_metrics.get("eos_rank")
    route_blocker = p910.first_non_eos_top(route_top_rows)
    patched_blocker = p910.first_non_eos_top(patched_top_rows)
    patched_margin = p911.eos_margin_vs_blocker(patched_metrics, patched_blocker)
    route_margin = p911.eos_margin_vs_blocker(route_metrics, route_blocker)
    band16_mean_delta = None if patched_band16["mean"] is None or route_band16["mean"] is None else patched_band16["mean"] - route_band16["mean"]
    band32_mean_delta = None if patched_band32["mean"] is None or route_band32["mean"] is None else patched_band32["mean"] - route_band32["mean"]
    band16_max_delta = None if patched_band16["max"] is None or route_band16["max"] is None else patched_band16["max"] - route_band16["max"]
    band32_max_delta = None if patched_band32["max"] is None or route_band32["max"] is None else patched_band32["max"] - route_band32["max"]
    eos_delta = None if route_eos_logit is None or patched_eos_logit is None else float(patched_eos_logit - route_eos_logit)
    rank_delta = None if route_eos_rank is None or patched_eos_rank is None else int(patched_eos_rank) - int(route_eos_rank)
    route_near_top50 = bool(route_eos_rank is not None and int(route_eos_rank) <= 50)
    group_ids = mlp_groups.get(str(spec.get("group_kind"))) if spec.get("group_kind") else []
    weak = bool(
        route_near_top50
        and band16_mean_delta is not None
        and band16_mean_delta <= -0.25
        and eos_delta is not None
        and eos_delta >= 0.0
        and rank_delta is not None
        and rank_delta <= 0
    )
    strong = bool(
        weak
        and band16_mean_delta is not None
        and band16_mean_delta <= -0.35
        and patched_eos_rank is not None
        and route_eos_rank is not None
        and int(patched_eos_rank) <= max(1, int(route_eos_rank) - 3)
    )
    eos_top1 = bool(patched_eos_rank == 1)
    return {
        "phase": PHASE,
        "row_kind": "phase914_l4_mlp_route_near_holdout_validation_row",
        "model": source_row.get("model"),
        "eval_kind": source_row.get("eval_kind"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "edit_mode": source_row.get("edit_mode"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "canonical_answer": source_row.get("canonical_answer"),
        "prompt_variant": source_row.get("prompt_variant"),
        "prefix_text": prefix_text,
        "control_label": spec.get("control_label"),
        "control_kind": spec.get("control_kind"),
        "subunit_family": spec.get("subunit_family"),
        "group_kind": spec.get("group_kind"),
        "factor": spec.get("factor"),
        "route_near_top50": route_near_top50,
        "prompt_input_intact": True,
        "prompt_all_zero_used_as_test_control": False,
        "route_delta_norm": route_delta_norm,
        "route_eos_rank": route_eos_rank,
        "route_eos_logit": route_eos_logit,
        "route_eos_margin_vs_blocker": route_margin,
        "route_blocker_token": route_blocker.get("token") if route_blocker else None,
        "patched_eos_rank": patched_eos_rank,
        "patched_eos_logit": patched_eos_logit,
        "patched_eos_top1": eos_top1,
        "patched_eos_top5": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 5),
        "patched_eos_top10": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 10),
        "patched_eos_top50": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 50),
        "patched_eos_margin_vs_blocker": patched_margin,
        "patched_eos_margin_nonnegative": bool(patched_margin is not None and patched_margin >= 0),
        "patched_blocker_token": patched_blocker.get("token") if patched_blocker else None,
        "eos_logit_delta_vs_route": eos_delta,
        "eos_rank_delta_vs_route": rank_delta,
        "band16_mean_logit_delta": band16_mean_delta,
        "band32_mean_logit_delta": band32_mean_delta,
        "band16_max_logit_delta": band16_max_delta,
        "band32_max_logit_delta": band32_max_delta,
        "weak_holdout_candidate": weak,
        "strong_holdout_candidate": strong,
        "strict_clean_candidate": p911.strict_clean_candidate(tokenizer, case, prefix_ids, eos_top1),
        "mlp_group_size": len(group_ids or []),
        "mlp_group_preview": [int(x) for x in (group_ids or [])[:16]],
        "mlp_diag": mlp_diag if spec.get("control_kind") == "mlp_channel_group_scale" else {},
        "band16_tokens": [p903.decode_token(tokenizer, token_id) for token_id in band16_ids],
        "route_top8": route_top_rows[:8],
        "patched_top8": patched_top_rows[:8],
    }


def monotonic_summaries(rows: list[dict[str, Any]], factors: list[float]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("control_kind") != "mlp_channel_group_scale" or not row.get("route_near_top50"):
            continue
        groups[
            (
                row.get("model"),
                row.get("case_id"),
                row.get("prompt_variant"),
                row.get("source_subset_key"),
                row.get("edit_mode"),
                row.get("eval_kind"),
                row.get("group_kind"),
            )
        ].append(row)
    out = []
    expected = sorted(float(x) for x in factors)
    for key, vals in groups.items():
        by_factor = {float(row.get("factor")): row for row in vals if row.get("factor") is not None}
        ordered = [by_factor[factor] for factor in expected if factor in by_factor]
        if len(ordered) < 3:
            continue
        band = [row.get("band16_mean_logit_delta") for row in ordered]
        eos = [row.get("eos_logit_delta_vs_route") for row in ordered]
        ranks = [row.get("patched_eos_rank") for row in ordered]
        valid_band = all(value is not None for value in band)
        valid_eos = all(value is not None for value in eos)
        # factors ascend from strong suppression to weak suppression. Stronger suppression should produce no weaker band reduction.
        band_monotonic = bool(valid_band and all(float(band[i]) <= float(band[i + 1]) + 1e-6 for i in range(len(band) - 1)))
        eos_nonnegative_all = bool(valid_eos and all(float(value) >= 0.0 for value in eos))
        any_weak = any(row.get("weak_holdout_candidate") for row in ordered)
        any_strong = any(row.get("strong_holdout_candidate") for row in ordered)
        best = min(ordered, key=lambda row: row.get("patched_eos_rank") or 10**9)
        out.append(
            {
                "model": key[0],
                "case_id": key[1],
                "prompt_variant": key[2],
                "source_subset_key": key[3],
                "edit_mode": key[4],
                "eval_kind": key[5],
                "group_kind": key[6],
                "n_factors": len(ordered),
                "band_monotonic": band_monotonic,
                "eos_nonnegative_all": eos_nonnegative_all,
                "any_weak_holdout_candidate": any_weak,
                "any_strong_holdout_candidate": any_strong,
                "best_factor": best.get("factor"),
                "best_patched_rank": best.get("patched_eos_rank"),
                "best_band16_mean_delta": best.get("band16_mean_logit_delta"),
                "best_eos_delta": best.get("eos_logit_delta_vs_route"),
                "best_margin": best.get("patched_eos_margin_vs_blocker"),
                "factors": [row.get("factor") for row in ordered],
                "band16_mean_deltas": band,
                "eos_deltas": eos,
                "patched_ranks": ranks,
            }
        )
    out.sort(
        key=lambda row: (
            row.get("any_strong_holdout_candidate"),
            row.get("any_weak_holdout_candidate"),
            row.get("band_monotonic"),
            row.get("eos_nonnegative_all"),
            -(row.get("best_band16_mean_delta") or 9999),
        ),
        reverse=True,
    )
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = [row for row in rows if row.get("control_kind") != "route_only"]
    route_rows = [row for row in rows if row.get("control_kind") == "route_only"]
    route_near_sources = [row for row in source_rows if row.get("route_near_top50")]
    return {
        "rows": len(rows),
        "route_rows": len(route_rows),
        "source_rows": len(source_rows),
        "route_near_route_rows": sum(1 for row in route_rows if row.get("route_near_top50")),
        "route_near_source_rows": len(route_near_sources),
        "route_eos_top10": sum(1 for row in route_rows if row.get("patched_eos_top10")),
        "route_eos_top50": sum(1 for row in route_rows if row.get("patched_eos_top50")),
        "route_eos_top5": sum(1 for row in route_rows if row.get("patched_eos_top5")),
        "source_eos_top1": sum(1 for row in route_near_sources if row.get("patched_eos_top1")),
        "source_eos_top5": sum(1 for row in route_near_sources if row.get("patched_eos_top5")),
        "source_eos_top10": sum(1 for row in route_near_sources if row.get("patched_eos_top10")),
        "source_eos_top50": sum(1 for row in route_near_sources if row.get("patched_eos_top50")),
        "source_margin_nonnegative": sum(1 for row in route_near_sources if row.get("patched_eos_margin_nonnegative")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "source_strict_clean_candidate": sum(1 for row in route_near_sources if row.get("strict_clean_candidate")),
        "weak_holdout_candidate": sum(1 for row in route_near_sources if row.get("weak_holdout_candidate")),
        "strong_holdout_candidate": sum(1 for row in route_near_sources if row.get("strong_holdout_candidate")),
        "median_band16_mean_delta": median([row.get("band16_mean_logit_delta") for row in route_near_sources]),
        "median_eos_logit_delta": median([row.get("eos_logit_delta_vs_route") for row in route_near_sources]),
        "mean_eos_logit_delta": mean([row.get("eos_logit_delta_vs_route") for row in route_near_sources]),
        "route_blocker_tokens_top12": dict(Counter(str(row.get("route_blocker_token")) for row in route_rows).most_common(12)),
        "patched_blocker_tokens_top12": dict(Counter(str(row.get("patched_blocker_token")) for row in route_near_sources).most_common(12)),
    }


def posthoc_boundary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = [row for row in rows if row.get("control_kind") != "route_only" and row.get("route_near_top50")]
    promoted_top5 = [
        row
        for row in source_rows
        if row.get("patched_eos_top5") and row.get("route_eos_rank") is not None and int(row.get("route_eos_rank")) > 5
    ]
    already_top5 = [
        row
        for row in source_rows
        if row.get("patched_eos_top5") and row.get("route_eos_rank") is not None and int(row.get("route_eos_rank")) <= 5
    ]
    promoted_top10 = [
        row
        for row in source_rows
        if row.get("patched_eos_top10") and row.get("route_eos_rank") is not None and int(row.get("route_eos_rank")) > 10
    ]
    improved = [row for row in source_rows if row.get("eos_rank_delta_vs_route") is not None and int(row.get("eos_rank_delta_vs_route")) < 0]
    keys = {
        (
            row.get("case_id"),
            row.get("prompt_variant"),
            row.get("source_subset_key"),
            row.get("edit_mode"),
            row.get("eval_kind"),
        )
        for row in promoted_top5
    }
    return {
        "source_promoted_top5_from_non_top5": len(promoted_top5),
        "source_promoted_top5_unique_eval_keys": len(keys),
        "source_top5_already_route_top5": len(already_top5),
        "source_promoted_top10_from_non_top10": len(promoted_top10),
        "source_rank_improved": len(improved),
        "promoted_top5_by_group_factor": {
            f"{group}|{factor:g}": count
            for (group, factor), count in Counter(
                (str(row.get("group_kind")), float(row.get("factor"))) for row in promoted_top5 if row.get("factor") is not None
            ).most_common()
        },
        "promoted_top5_cases_top12": {
            f"{case}|{prompt}|{kind}|route_rank={rank}": count
            for (case, prompt, kind, rank), count in Counter(
                (
                    str(row.get("case_id")),
                    str(row.get("prompt_variant")),
                    str(row.get("eval_kind")),
                    int(row.get("route_eos_rank")),
                )
                for row in promoted_top5
            ).most_common(12)
        },
    }


def summarize_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("control_kind") == "mlp_channel_group_scale" and row.get("route_near_top50"):
            buckets[(str(row.get("group_kind")), float(row.get("factor")))].append(row)
    out = []
    for (group_kind, factor), vals in buckets.items():
        summary = summarize_rows(vals)
        summary.update({"group_kind": group_kind, "factor": factor})
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("source_eos_top1") or 0,
            row.get("source_margin_nonnegative") or 0,
            row.get("source_eos_top5") or 0,
            row.get("strong_holdout_candidate") or 0,
            row.get("weak_holdout_candidate") or 0,
            -(row.get("median_band16_mean_delta") or 9999),
        ),
        reverse=True,
    )
    return out


def summarize_model(model_name: str, rows: list[dict[str, Any]], eval_count: int, factors: list[float], attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_rows(rows)
    monotonic = monotonic_summaries(rows, factors)
    if overall["source_eos_top1"] > 0:
        evidence = "l4_holdout_reaches_eos_top1"
    elif overall["source_margin_nonnegative"] > 0:
        evidence = "l4_holdout_crosses_margin"
    elif overall["source_eos_top5"] > 0:
        evidence = "l4_holdout_reaches_eos_top5"
    elif overall["strong_holdout_candidate"] > 0:
        evidence = "strong_l4_route_near_holdout_candidates_found"
    elif overall["weak_holdout_candidate"] > 0:
        evidence = "weak_l4_route_near_holdout_candidates_found"
    elif overall["route_near_route_rows"] > 0:
        evidence = "route_near_but_no_l4_holdout_candidate"
    else:
        evidence = "no_route_near_samples_for_l4_holdout"
    return {
        "phase": PHASE,
        "title": "GLM4 Route-near L4 MLP Channel Group Holdout Validation",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "eval_items": eval_count,
        "overall": overall,
        "group_summaries": summarize_by_group(rows),
        "monotonic_summaries": monotonic[:120],
        "monotonic_overall": {
            "rows": len(monotonic),
            "band_monotonic": sum(1 for row in monotonic if row.get("band_monotonic")),
            "eos_nonnegative_all": sum(1 for row in monotonic if row.get("eos_nonnegative_all")),
            "any_weak_holdout_candidate": sum(1 for row in monotonic if row.get("any_weak_holdout_candidate")),
            "any_strong_holdout_candidate": sum(1 for row in monotonic if row.get("any_strong_holdout_candidate")),
        },
        "posthoc_boundary_metrics": posthoc_boundary_metrics(rows),
        "evidence_label": evidence,
        "boundary": (
            "Phase914 validates Phase913 L4 MLP channel-group candidates only on route-top50 samples. "
            "Rows outside route-near are not counted as closure evidence."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    eval_rows = expand_eval_rows(args.model, args, case_map)
    if args.dry_run or not eval_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if eval_rows else "no_rows",
            "eval_rows": len(eval_rows),
            "preview": eval_rows[:20],
        }
        p846.write_json(out_dir / f"phase914_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase914_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    factors = parse_factors(args.factors)
    group_kinds = parse_csv(args.mlp_group_kinds)
    specs = mlp_specs(factors, group_kinds)
    rows: list[dict[str, Any]] = []
    model = None
    tokenizer = None
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p903.protocol_category_groups(tokenizer)
        prompt_cache: dict[tuple[str, str], list[int]] = {}
        for idx, source_row in enumerate(eval_rows, 1):
            case = case_map.get(str(source_row.get("case_id")))
            if not case:
                continue
            prompt_key = (str(source_row.get("case_id")), str(source_row.get("prompt_variant")))
            if prompt_key not in prompt_cache:
                prompt = p885.prompt_for_case(case, str(source_row.get("prompt_variant")))
                prompt_cache[prompt_key] = p862.p844.encode_prompt(tokenizer, prompt)
            prompt_ids = prompt_cache[prompt_key]
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
                continue
            period_ids = current_ids + [int(period_id)]
            _baseline_logits, base_vec = p910.logits_and_l0_vector(model, device, period_ids)
            prompt_zero_handles = p909.install_attention_input_span_scale(model, 0, 0, len(prompt_ids), 0.0)
            _prompt_zero_logits, prompt_zero_vec = p910.logits_and_l0_vector(model, device, period_ids, prompt_zero_handles)
            if base_vec is None or prompt_zero_vec is None:
                continue
            route_delta = prompt_zero_vec - base_vec
            route_delta_norm = float(torch.linalg.vector_norm(route_delta).item())
            if route_delta_norm <= 0:
                continue
            route_logits, mlp_activation = p913.capture_route_logits_and_mlp_activation(model, device, period_ids, route_delta, 4)
            if route_logits is None:
                continue
            route_metrics = p903.state_metrics(tokenizer, route_logits, groups)
            route_top_rows = p910.topk_tokens(tokenizer, route_logits, groups, max(64, int(args.band_size)))
            band32_ids = p911.top_non_eos_ids(route_top_rows, int(args.band_size))
            band16_ids = band32_ids[: min(16, len(band32_ids))]
            mlp_groups, mlp_diag = p913.mlp_channel_groups_for_case(
                model,
                device,
                mlp_activation,
                route_metrics.get("eos_best_id"),
                band16_ids,
                band32_ids,
                int(args.mlp_candidate_pool),
            )
            route_near = bool(route_metrics.get("eos_rank") is not None and int(route_metrics["eos_rank"]) <= int(args.route_topk_filter))
            for spec in specs:
                if spec.get("control_kind") == "route_only":
                    patched_logits = route_logits
                else:
                    if not route_near and not args.evaluate_non_route_near:
                        continue
                    if not mlp_groups.get(str(spec.get("group_kind"))):
                        continue
                    patched_logits = p913.logits_with_spec(
                        model,
                        device,
                        period_ids,
                        route_delta,
                        spec,
                        len(prompt_ids),
                        len(prefix_ids),
                        mlp_groups,
                    )
                    if patched_logits is None:
                        continue
                patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
                patched_top_rows = p910.topk_tokens(tokenizer, patched_logits, groups, 16)
                rows.append(
                    row_from_logits(
                        tokenizer,
                        source_row,
                        case,
                        spec,
                        prefix_ids,
                        prefix_text,
                        route_metrics,
                        patched_metrics,
                        route_logits,
                        patched_logits,
                        route_top_rows,
                        patched_top_rows,
                        band16_ids,
                        band32_ids,
                        route_delta_norm,
                        mlp_groups,
                        mlp_diag,
                    )
                )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(eval_rows):
                log(f"{args.model}/{args.round_name}: item={idx}/{len(eval_rows)} rows={len(rows)} route_near={route_near}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = summarize_model(args.model, rows, len(eval_rows), factors, attn_impl)
    p846.write_json(out_dir / f"phase914_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase914_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "monotonic": payload["monotonic_overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    top_groups = []
    top_mono = []
    for model_name in MODELS:
        path = out_dir / f"phase914_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        row_path = out_dir / f"phase914_{model_name}_rows.jsonl"
        rows = read_jsonl(row_path)
        if rows:
            summary["posthoc_boundary_metrics"] = posthoc_boundary_metrics(rows)
            if isinstance(summary.get("overall"), dict):
                summary["overall"]["route_eos_top5"] = sum(
                    1 for row in rows if row.get("control_kind") == "route_only" and row.get("patched_eos_top5")
                )
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        for key in [
            "rows",
            "route_rows",
            "source_rows",
            "route_near_route_rows",
            "route_near_source_rows",
            "route_eos_top5",
            "route_eos_top10",
            "route_eos_top50",
            "source_eos_top1",
            "source_eos_top5",
            "source_eos_top10",
            "source_eos_top50",
            "source_margin_nonnegative",
            "strict_clean_candidate",
            "source_strict_clean_candidate",
            "weak_holdout_candidate",
            "strong_holdout_candidate",
        ]:
            scalar[key] += int(overall.get(key) or 0)
        posthoc = summary.get("posthoc_boundary_metrics") or {}
        for key in [
            "source_promoted_top5_from_non_top5",
            "source_promoted_top5_unique_eval_keys",
            "source_top5_already_route_top5",
            "source_promoted_top10_from_non_top10",
            "source_rank_improved",
        ]:
            scalar[key] += int(posthoc.get(key) or 0)
        for row in summary.get("group_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_groups.append(item)
        for row in summary.get("monotonic_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_mono.append(item)
    top_groups.sort(
        key=lambda row: (
            row.get("source_eos_top1") or 0,
            row.get("source_margin_nonnegative") or 0,
            row.get("source_eos_top5") or 0,
            row.get("strong_holdout_candidate") or 0,
            row.get("weak_holdout_candidate") or 0,
            -(row.get("median_band16_mean_delta") or 9999),
        ),
        reverse=True,
    )
    top_mono.sort(
        key=lambda row: (
            row.get("any_strong_holdout_candidate"),
            row.get("any_weak_holdout_candidate"),
            row.get("band_monotonic"),
            row.get("eos_nonnegative_all"),
            -(row.get("best_band16_mean_delta") or 9999),
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_groups": top_groups[:120],
        "top_monotonic": top_mono[:120],
    }
    p846.write_json(out_dir / "phase914_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase914_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 914 GLM4 route-near L4 MLP channel group holdout validation",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | eval items | route rows | route top5 | route top50 | near source rows | top5 | promoted top5 | top10 | margin>=0 | weak | strong | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        posthoc = summary.get("posthoc_boundary_metrics") or {}
        lines.append(
            "| {model} | {items} | {route_rows} | {route5} | {route50} | {near_sources} | {top5} | {promoted_top5} | {top10} | {margin} | {weak} | {strong} | {evidence} |".format(
                model=summary.get("model"),
                items=summary.get("eval_items"),
                route_rows=overall.get("route_rows"),
                route5=overall.get("route_eos_top5"),
                route50=overall.get("route_eos_top50"),
                near_sources=overall.get("route_near_source_rows"),
                top5=overall.get("source_eos_top5"),
                promoted_top5=posthoc.get("source_promoted_top5_from_non_top5"),
                top10=overall.get("source_eos_top10"),
                margin=overall.get("source_margin_nonnegative"),
                weak=overall.get("weak_holdout_candidate"),
                strong=overall.get("strong_holdout_candidate"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Groups", ""])
    lines.append(
        "| model | group | factor | rows | top5 | top10 | margin>=0 | weak | strong | median band16 delta | median eos delta | blockers |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("top_groups") or []:
        lines.append(
            "| {model} | {group_kind} | {factor} | {rows} | {source_eos_top5} | {source_eos_top10} | {source_margin_nonnegative} | {weak_holdout_candidate} | {strong_holdout_candidate} | {median_band16_mean_delta} | {median_eos_logit_delta} | {route_blocker_tokens_top12} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Monotonic", ""])
    lines.append(
        "| model | case | prompt | group | weak | strong | band mono | eos nonneg all | best factor | best rank | best band16 delta | best eos delta |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_monotonic") or []:
        lines.append(
            "| {model} | {case_id} | {prompt_variant} | {group_kind} | {any_weak_holdout_candidate} | {any_strong_holdout_candidate} | {band_monotonic} | {eos_nonnegative_all} | {best_factor} | {best_patched_rank} | {best_band16_mean_delta} | {best_eos_delta} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="l4_mlp_route_near_holdout_validation")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-eval-items-per-model", type=int, default=96)
    parser.add_argument("--prompt-variants", default="natural_question,natural_category,classification,question_plain,type_of_completion")
    parser.add_argument("--holdout-prompt-variants", default="natural_question,classification")
    parser.add_argument("--same-domain-holdout-per-domain", type=int, default=4)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--factors", default="0.9,0.8,0.7,0.6,0.5,0.4,0.3")
    parser.add_argument("--mlp-group-kinds", default="top_abs_64,band16_support_32,band16_support_64,band32_support_64,low_abs_64")
    parser.add_argument("--mlp-candidate-pool", type=int, default=512)
    parser.add_argument("--band-size", type=int, default=32)
    parser.add_argument("--route-topk-filter", type=int, default=50)
    parser.add_argument("--evaluate-non-route-near", action="store_true")
    parser.add_argument("--log-every", type=int, default=8)
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
