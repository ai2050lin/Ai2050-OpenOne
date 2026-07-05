#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
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
import phase910_prompt_preserving_termination_route_reconstruction as p910  # noqa: E402
import phase911_full_vocab_blocker_displacement_audit as p911  # noqa: E402
import phase912_finite_blocker_band_source_localization as p912  # noqa: E402
import phase913_route_preserving_blocker_band_disentanglement as p913  # noqa: E402
import phase918_l39_mlp_channel_a_blocker_suppressor_localization as p918  # noqa: E402
import phase919_frozen_l39_signed_margin_group_transfer_validation as p919  # noqa: E402


PHASE = 920
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase920_consensus_l39_signed_margin_gear_holdout_controls")


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


def consensus_group(states: list[dict[str, Any]], group_kind: str, budget: int) -> tuple[list[int], dict[str, Any]]:
    counts: Counter[int] = Counter()
    contributing = 0
    for state in states:
        group = state["channel_groups"].get(group_kind) or []
        if group:
            contributing += 1
            counts.update(int(x) for x in group)
    chosen = [int(channel) for channel, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: int(budget)]]
    freqs = [int(counts[channel]) for channel in chosen]
    return chosen, {
        "train_state_count": len(states),
        "contributing_state_count": contributing,
        "unique_channel_count": len(counts),
        "chosen_size": len(chosen),
        "chosen_min_frequency": min(freqs) if freqs else None,
        "chosen_median_frequency": median(freqs),
        "chosen_max_frequency": max(freqs) if freqs else None,
    }


def rotated_group(group: list[int], width: int, offset: int) -> list[int]:
    if width <= 0:
        return []
    seen = set()
    out = []
    for value in group:
        candidate = (int(value) + int(offset)) % int(width)
        while candidate in seen:
            candidate = (candidate + 1) % int(width)
        seen.add(candidate)
        out.append(candidate)
    return out


def random_group(width: int, budget: int, seed: int) -> list[int]:
    if width <= 0 or budget <= 0:
        return []
    rng = random.Random(int(seed))
    return sorted(rng.sample(range(int(width)), k=min(int(budget), int(width))))


def state_filter(states: list[dict[str, Any]], target_state: dict[str, Any], fold_kind: str) -> list[dict[str, Any]]:
    target_row = target_state["source_row"]
    if fold_kind == "all_train":
        return list(states)
    if fold_kind == "leave_one_case":
        return [state for state in states if str(state["source_row"].get("case_id")) != str(target_row.get("case_id"))]
    if fold_kind == "leave_one_domain":
        return [state for state in states if str(state["source_row"].get("eval_domain")) != str(target_row.get("eval_domain"))]
    return list(states)


def make_control_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for factor in parse_floats(args.margin_pos_factors):
        for fold_kind in parse_csv(args.fold_kinds):
            specs.append(
                {
                    "control_label": f"consensus_margin_support_pos_64_{fold_kind}_scale_{factor:g}",
                    "control_family": "consensus_positive_margin",
                    "control_class": "positive",
                    "group_source": "consensus",
                    "fold_kind": fold_kind,
                    "group_kind": "margin_support_pos_64",
                    "factor": float(factor),
                }
            )
    for factor in parse_floats(args.suppress_factors):
        for group_kind in ["a_blocker_support_64", "margin_support_neg_64"]:
            for fold_kind in parse_csv(args.fold_kinds):
                specs.append(
                    {
                        "control_label": f"consensus_{group_kind}_{fold_kind}_scale_{factor:g}",
                        "control_family": "consensus_suppress_margin_blocker",
                        "control_class": "positive",
                        "group_source": "consensus",
                        "fold_kind": fold_kind,
                        "group_kind": group_kind,
                        "factor": float(factor),
                    }
                )
    for factor in parse_floats(args.negative_scale_factors):
        specs.extend(
            [
                {
                    "control_label": f"random_all_64_scale_{factor:g}",
                    "control_family": "negative_random_all",
                    "control_class": "negative",
                    "group_source": "random_all",
                    "fold_kind": "none",
                    "group_kind": "random_all_64",
                    "factor": float(factor),
                },
                {
                    "control_label": f"rotated_consensus_margin_support_pos_64_scale_{factor:g}",
                    "control_family": "negative_rotated_consensus",
                    "control_class": "negative",
                    "group_source": "rotated_consensus",
                    "fold_kind": "all_train",
                    "group_kind": "margin_support_pos_64",
                    "factor": float(factor),
                },
            ]
        )
    for factor in parse_floats(args.negative_suppress_factors):
        specs.append(
            {
                "control_label": f"consensus_a_logit_support_64_scale_{factor:g}",
                "control_family": "negative_a_logit_only",
                "control_class": "negative",
                "group_source": "consensus",
                "fold_kind": "all_train",
                "group_kind": "a_logit_support_64",
                "factor": float(factor),
            }
        )
    return specs


def resolve_group(
    states: list[dict[str, Any]],
    target_state: dict[str, Any],
    spec: dict[str, Any],
    layer_width: int,
    budget: int,
    seed_base: int,
) -> tuple[list[int], dict[str, Any]]:
    source = str(spec.get("group_source"))
    group_kind = str(spec.get("group_kind"))
    if source == "random_all":
        group = random_group(layer_width, budget, seed_base)
        return group, {"chosen_size": len(group), "train_state_count": 0, "negative_seed": seed_base}
    train_states = state_filter(states, target_state, str(spec.get("fold_kind")))
    base_group, diag = consensus_group(train_states, group_kind, budget)
    if source == "rotated_consensus":
        group = rotated_group(base_group, layer_width, max(97, layer_width // 7))
        diag = dict(diag)
        diag["rotated_offset"] = max(97, layer_width // 7)
        diag["base_group_preview"] = base_group[:16]
        diag["chosen_size"] = len(group)
        return group, diag
    return base_group, diag


def row_from_patch(
    tokenizer,
    target_state: dict[str, Any],
    spec: dict[str, Any],
    group: list[int],
    group_diag: dict[str, Any],
    patched_logits: torch.Tensor,
    groups: dict[str, list[int]],
) -> dict[str, Any]:
    target_row = target_state["source_row"]
    target_case = target_state["case"]
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
    native_group = target_state["channel_groups"].get(str(spec.get("group_kind")), [])
    return {
        "phase": PHASE,
        "row_kind": "phase920_consensus_l39_signed_margin_gear_control_row",
        "model": target_row.get("model"),
        "target_state_key": target_state["state_key"],
        "target_case_id": target_row.get("case_id"),
        "target_eval_domain": target_row.get("eval_domain"),
        "target_prompt_variant": target_row.get("prompt_variant"),
        "target_source_subset_key": target_row.get("source_subset_key"),
        "target_edit_mode": target_row.get("edit_mode"),
        "target_object": target_case.get("object"),
        "target_canonical_answer": target_case.get("canonical_answer"),
        "target_prefix_text": target_state["prefix_text"],
        "control_label": spec.get("control_label"),
        "control_family": spec.get("control_family"),
        "control_class": spec.get("control_class"),
        "group_source": spec.get("group_source"),
        "fold_kind": spec.get("fold_kind"),
        "group_kind": spec.get("group_kind"),
        "factor": spec.get("factor"),
        "layer_idx": 39,
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
        "channel_group_size": len(group),
        "channel_group_preview": [int(x) for x in group[:16]],
        "target_native_group_overlap": len(set(int(x) for x in group) & set(int(x) for x in native_group)),
        "target_native_group_size": len(native_group),
        "group_diag": group_diag,
        "target_boundary_top8": boundary_top_rows[:8],
        "patched_top8": patched_top_rows[:8],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "top1": sum(1 for row in rows if row.get("patched_eos_top1")),
        "top5": sum(1 for row in rows if row.get("patched_eos_top5")),
        "margin_nonnegative": sum(1 for row in rows if row.get("patched_eos_margin_nonnegative")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "weak_transfer_candidate": sum(1 for row in rows if row.get("weak_transfer_candidate")),
        "rank_improved": sum(1 for row in rows if row.get("rank_improved")),
        "blocker_suppressed": sum(1 for row in rows if row.get("target_boundary_blocker_suppressed")),
        "median_margin_delta": median([row.get("eos_margin_delta_vs_target_boundary") for row in rows]),
        "mean_margin_delta": mean([row.get("eos_margin_delta_vs_target_boundary") for row in rows]),
        "mean_eos_delta": mean([row.get("eos_logit_delta_vs_target_boundary") for row in rows]),
        "median_blocker_delta": median([row.get("target_boundary_blocker_logit_delta") for row in rows]),
        "median_native_group_overlap": median([row.get("target_native_group_overlap") for row in rows]),
        "target_state_coverage_top1": len({row.get("target_state_key") for row in rows if row.get("patched_eos_top1")}),
        "target_state_coverage_margin": len({row.get("target_state_key") for row in rows if row.get("patched_eos_margin_nonnegative")}),
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
        for meta_key in ["control_label", "control_family", "control_class", "group_source", "fold_kind", "group_kind", "factor"]:
            summary.setdefault(meta_key, first.get(meta_key))
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("strict_clean_candidate") or 0,
            row.get("top1") or 0,
            row.get("margin_nonnegative") or 0,
            row.get("weak_transfer_candidate") or 0,
            row.get("median_margin_delta") or -9999,
        ),
        reverse=True,
    )
    return out[:limit]


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, spec_count: int, attn_impl: str | None) -> dict[str, Any]:
    positive_rows = [row for row in rows if row.get("control_class") == "positive"]
    negative_rows = [row for row in rows if row.get("control_class") == "negative"]
    consensus_rows = [row for row in rows if row.get("group_source") == "consensus" and row.get("control_class") == "positive"]
    loo_case_rows = [row for row in consensus_rows if row.get("fold_kind") == "leave_one_case"]
    loo_domain_rows = [row for row in consensus_rows if row.get("fold_kind") == "leave_one_domain"]
    overall = {
        "all": summarize_rows(rows),
        "positive": summarize_rows(positive_rows),
        "negative": summarize_rows(negative_rows),
        "consensus_positive": summarize_rows(consensus_rows),
        "leave_one_case": summarize_rows(loo_case_rows),
        "leave_one_domain": summarize_rows(loo_domain_rows),
        "target_state_count": len({row.get("target_state_key") for row in rows}),
    }
    if selected_count == 0:
        evidence = "no_phase915_l39_candidates"
    elif overall["leave_one_case"]["strict_clean_candidate"] > 0 and overall["negative"]["strict_clean_candidate"] == 0:
        evidence = "consensus_holdout_positive_beats_negative_controls"
    elif overall["consensus_positive"]["margin_nonnegative"] > overall["negative"]["margin_nonnegative"]:
        evidence = "consensus_positive_above_negative_controls_without_strict_separation"
    else:
        evidence = "consensus_not_separated_from_negative_controls"
    return {
        "phase": PHASE,
        "title": "Consensus L39 Signed Margin Gear Holdout Controls",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_phase915_l39_candidates": int(selected_count),
        "control_spec_count": int(spec_count),
        "overall": overall,
        "by_control": summarize_by(rows, ["control_label"]),
        "by_family": summarize_by(rows, ["control_family", "fold_kind"]),
        "by_class": summarize_by(rows, ["control_class"]),
        "evidence_label": evidence,
        "boundary": (
            "Phase920 compresses Phase919 source-frozen channel groups into consensus groups and tests "
            "leave-one-case/domain holdout plus random, rotated, and a-logit-only negative controls."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p918.select_phase915_candidates(args.model, args)
    specs = make_control_specs(args)
    if args.dry_run or not selected:
        payload = summarize_model(args.model, [], len(selected), len(specs), None)
        payload["status"] = "dry_run" if args.dry_run else "no_phase915_l39_candidates"
        p846.write_json(out_dir / f"phase920_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase920_{args.model}_rows.jsonl", [])
        print(json.dumps({"phase": PHASE, "model": args.model, "status": payload["status"], "selected": len(selected)}, ensure_ascii=False, indent=2), flush=True)
        return payload
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
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
        down_proj = p913.mlp_down_proj(model, int(args.target_layer))
        layer_width = int(getattr(down_proj, "in_features", 0) or 0)
        for target_idx, target_state in enumerate(states, 1):
            for spec_idx, spec in enumerate(specs, 1):
                group, group_diag = resolve_group(
                    states,
                    target_state,
                    spec,
                    layer_width,
                    int(args.group_budget),
                    seed_base=PHASE * 100000 + target_idx * 1000 + spec_idx,
                )
                if not group:
                    continue
                patched_logits = p919.logits_with_target_boundary_and_frozen_group(
                    model,
                    device,
                    target_state,
                    group,
                    int(args.target_layer),
                    float(spec.get("factor")),
                )
                if patched_logits is None:
                    continue
                rows.append(row_from_patch(tokenizer, target_state, spec, group, group_diag, patched_logits, groups))
            if target_idx % max(1, int(args.log_every)) == 0 or target_idx == len(states):
                log(f"{args.model}/{args.round_name}: target={target_idx}/{len(states)} rows={len(rows)}")
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
    p846.write_json(out_dir / f"phase920_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase920_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    evidence = Counter()
    scalar = Counter()
    controls = []
    families = []
    classes = []
    for model_name in MODELS:
        path = out_dir / f"phase920_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        scalar["selected_phase915_l39_candidates"] += int(summary.get("selected_phase915_l39_candidates") or 0)
        overall = summary.get("overall") or {}
        for scope in ["all", "positive", "negative", "consensus_positive", "leave_one_case", "leave_one_domain"]:
            scoped = overall.get(scope) or {}
            for key in ["rows", "top1", "margin_nonnegative", "strict_clean_candidate", "weak_transfer_candidate"]:
                scalar[f"{scope}_{key}"] += int(scoped.get(key) or 0)
        scalar["target_state_count"] += int(overall.get("target_state_count") or 0)
        for source_key, target in [("by_control", controls), ("by_family", families), ("by_class", classes)]:
            for row in summary.get(source_key) or []:
                item = dict(row)
                item["model"] = summary.get("model")
                target.append(item)
    sort_keys = lambda row: (
        row.get("strict_clean_candidate") or 0,
        row.get("top1") or 0,
        row.get("margin_nonnegative") or 0,
        row.get("weak_transfer_candidate") or 0,
        row.get("median_margin_delta") or -9999,
    )
    controls.sort(key=sort_keys, reverse=True)
    families.sort(key=sort_keys, reverse=True)
    classes.sort(key=sort_keys, reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_controls": controls[:160],
        "top_families": families[:80],
        "top_classes": classes[:40],
    }
    p846.write_json(out_dir / "phase920_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase920_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 920 consensus L39 signed margin gear holdout controls",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | selected | target states | positive top1 | positive margin | positive strict | negative top1 | negative margin | negative strict | loo-case strict | loo-domain strict | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        pos = overall.get("positive") or {}
        neg = overall.get("negative") or {}
        loo_case = overall.get("leave_one_case") or {}
        loo_domain = overall.get("leave_one_domain") or {}
        lines.append(
            "| {model} | {selected} | {states} | {pos_top1} | {pos_margin} | {pos_strict} | {neg_top1} | {neg_margin} | {neg_strict} | {case_strict} | {domain_strict} | {evidence} |".format(
                model=summary.get("model"),
                selected=summary.get("selected_phase915_l39_candidates"),
                states=overall.get("target_state_count"),
                pos_top1=pos.get("top1"),
                pos_margin=pos.get("margin_nonnegative"),
                pos_strict=pos.get("strict_clean_candidate"),
                neg_top1=neg.get("top1"),
                neg_margin=neg.get("margin_nonnegative"),
                neg_strict=neg.get("strict_clean_candidate"),
                case_strict=loo_case.get("strict_clean_candidate"),
                domain_strict=loo_domain.get("strict_clean_candidate"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Controls", ""])
    lines.append(
        "| model | control | class | family | fold | group | factor | rows | top1 | margin | strict | weak | targets top1 | targets margin | median delta | overlap |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_controls") or []:
        row = {
            "model": "",
            "control_label": "",
            "control_class": "",
            "control_family": "",
            "fold_kind": "",
            "group_kind": "",
            "factor": "",
            "rows": 0,
            "top1": 0,
            "margin_nonnegative": 0,
            "strict_clean_candidate": 0,
            "weak_transfer_candidate": 0,
            "target_state_coverage_top1": 0,
            "target_state_coverage_margin": 0,
            "median_margin_delta": None,
            "median_native_group_overlap": None,
            **row,
        }
        lines.append(
            "| {model} | {control_label} | {control_class} | {control_family} | {fold_kind} | {group_kind} | {factor} | {rows} | {top1} | {margin_nonnegative} | {strict_clean_candidate} | {weak_transfer_candidate} | {target_state_coverage_top1} | {target_state_coverage_margin} | {median_margin_delta} | {median_native_group_overlap} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="consensus_l39_signed_margin_gear_holdout_controls")
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
    parser.add_argument("--fold-kinds", default="all_train,leave_one_case,leave_one_domain")
    parser.add_argument("--margin-pos-factors", default="1.125,1.25,1.375,1.5,1.75,2.0")
    parser.add_argument("--suppress-factors", default="0.0,0.25,0.5")
    parser.add_argument("--negative-scale-factors", default="1.375,1.75,2.0")
    parser.add_argument("--negative-suppress-factors", default="0.0,0.25,0.5")
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
