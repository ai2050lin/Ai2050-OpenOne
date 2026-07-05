#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter
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
import phase913_route_preserving_blocker_band_disentanglement as p913  # noqa: E402
import phase918_l39_mlp_channel_a_blocker_suppressor_localization as p918  # noqa: E402
import phase920_consensus_l39_signed_margin_gear_holdout_controls as p920  # noqa: E402
import phase922_candidate_gate_variable_causal_coupling_test as p922  # noqa: E402
import phase924_route_protocol_response_surface_audit as p924  # noqa: E402


PHASE = 926
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase926_generalized_route_protocol_surface_validation")
PHASE925_ROOT = Path("tests/result/phase925_response_surface_generalization_dataset_expansion")


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


def parse_floats(raw: str) -> list[float]:
    return [float(part) for part in parse_csv(raw)]


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def blocker_class(token: Any) -> str:
    text = str(token)
    if text == "a":
        return "article_a"
    if text.strip() in {".", "。"}:
        return "punctuation_period"
    if not text.strip():
        return "blank_or_space"
    return "other"


def seed_sort_key(row: dict[str, Any]) -> tuple[float, int, int, str]:
    return (
        float(row.get("score_scalar") or 0.0),
        int(bool(row.get("top5"))),
        int(bool(row.get("top10"))),
        str(row.get("surface_state_key")),
    )


def select_phase925_seeds(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE925_ROOT / args.phase925_round / f"phase925_{model_name}_selected_surface_seeds.jsonl"
    rows = read_jsonl(path)
    rows.sort(key=seed_sort_key, reverse=True)
    selected: list[dict[str, Any]] = []
    case_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    for row in rows:
        case_id = str(row.get("case_id"))
        domain = str(row.get("eval_domain"))
        group = str(row.get("group_kind"))
        blocker = blocker_class(row.get("patched_blocker_token"))
        if case_counts[case_id] >= int(args.max_per_case):
            continue
        if domain_counts[domain] >= int(args.max_per_domain):
            continue
        if group_counts[group] >= int(args.max_per_group):
            continue
        if blocker_counts[blocker] >= int(args.max_per_blocker_class):
            continue
        selected.append(dict(row))
        case_counts[case_id] += 1
        domain_counts[domain] += 1
        group_counts[group] += 1
        blocker_counts[blocker] += 1
        if len(selected) >= int(args.max_seeds_per_model):
            break
    if len(selected) < int(args.min_seeds_per_model):
        used = {str(row.get("surface_state_key")) for row in selected}
        for row in rows:
            key = str(row.get("surface_state_key"))
            if key in used:
                continue
            selected.append(dict(row))
            used.add(key)
            if len(selected) >= min(int(args.max_seeds_per_model), int(args.min_seeds_per_model)):
                break
    return selected[: max(0, int(args.max_seeds_per_model))]


def reconstruct_seed_state(
    model,
    tokenizer,
    device: torch.device,
    groups: dict[str, list[int]],
    case_map: dict[str, dict[str, Any]],
    seed: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    case = case_map.get(str(seed.get("case_id")))
    if not case:
        return None
    prompt = p885.prompt_for_case(case, str(seed.get("prompt_variant")))
    prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
    gears = p903.parse_gears(str(seed.get("source_subset_key")))
    _prefix_logits, prefix_ids, prefix_text, _answer_seen = p901.logits_after_answer_prefix(
        model,
        tokenizer,
        device,
        prompt_ids,
        gears,
        str(seed.get("edit_mode")),
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
    boundary_factor = seed.get("factor")
    group_kind = seed.get("group_kind")
    if boundary_factor is None or group_kind is None:
        return None
    boundary_spec = {
        "control_label": f"L4_mlp_channels_{group_kind}_scale_{float(boundary_factor):g}",
        "control_kind": "mlp_channel_group_scale",
        "layer_idx": 4,
        "group_kind": str(group_kind),
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
        "state_key": str(seed.get("surface_state_key")),
        "case": case,
        "source_row": dict(seed),
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


def counter_rows(rows: list[dict[str, Any]], key: str, limit: int = 80) -> list[dict[str, Any]]:
    counter = Counter(str(row.get(key)) for row in rows)
    return [{"key": item, "count": int(count)} for item, count in counter.most_common(limit)]


def summarize_surface_rows(rows: list[dict[str, Any]], surfaces: list[dict[str, Any]]) -> dict[str, Any]:
    surface_summary = p924.summarize_surfaces(surfaces)
    return {
        "overall": {
            "all": p924.summarize_rows(rows),
            "surface_base": p924.summarize_rows(
                [
                    row
                    for row in rows
                    if abs(float(row.get("route_alpha") or 0.0) - 1.0) <= 1e-9
                    and abs(float(row.get("protocol_span_factor") or 0.0) - 1.0) <= 1e-9
                ]
            ),
            "non_base": p924.summarize_rows(
                [
                    row
                    for row in rows
                    if not (
                        abs(float(row.get("route_alpha") or 0.0) - 1.0) <= 1e-9
                        and abs(float(row.get("protocol_span_factor") or 0.0) - 1.0) <= 1e-9
                    )
                ]
            ),
            "target_state_count": len({row.get("target_state_key") for row in rows}),
        },
        "surface_summary": surface_summary,
        "by_alpha_protocol": p924.summarize_by(rows, ["route_alpha", "protocol_span_factor"], limit=200),
        "by_domain": p924.summarize_by(rows, ["target_eval_domain"], limit=80),
        "by_case": p924.summarize_by(rows, ["target_case_id"], limit=120),
        "by_l4_group": p924.summarize_by(rows, ["phase925_group_kind"], limit=80),
        "by_seed_blocker_class": p924.summarize_by(rows, ["phase925_seed_blocker_class"], limit=40),
    }


def evidence_label(selected_count: int, surfaces: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if selected_count == 0:
        return "no_phase925_surface_seeds"
    if not surfaces:
        return "no_reconstructable_surface_states"
    ssum = summary.get("surface_summary") or {}
    overall = summary.get("overall") or {}
    non_base = overall.get("non_base") or {}
    if int(non_base.get("new_strict_vs_surface_base") or 0) > 0:
        return "generalized_surface_adds_strict_closure"
    if int(non_base.get("new_top1_vs_surface_base") or 0) > 0:
        return "generalized_surface_adds_top1_closure"
    if int(ssum.get("best_coord_is_base") or 0) < int(ssum.get("surface_count") or 0):
        return "generalized_surface_changes_best_coordinate"
    return "surface_base_remains_best_on_generalized_seeds"


def summarize_model(
    model_name: str,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    surfaces: list[dict[str, Any]],
    consensus_diag: dict[str, Any] | None,
    args: argparse.Namespace,
    attn_impl: str | None,
) -> dict[str, Any]:
    surface_payload = summarize_surface_rows(rows, surfaces)
    label = evidence_label(len(selected), surfaces, surface_payload)
    return {
        "phase": PHASE,
        "title": "Generalized Route-Protocol Surface Validation",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "phase925_selected_seed_count": len(selected),
        "alpha_count": len(parse_floats(args.route_alphas)),
        "protocol_factor_count": len(parse_floats(args.protocol_factors)),
        "l39_factor_count": len(parse_floats(args.l39_factors)),
        "expected_rows_if_all_reconstructed": len(selected)
        * len(parse_floats(args.route_alphas))
        * len(parse_floats(args.protocol_factors))
        * len(parse_floats(args.l39_factors)),
        "consensus_diag": consensus_diag or {},
        "selected_seed_summary": {
            "rows": len(selected),
            "unique_cases": len({row.get("case_id") for row in selected}),
            "unique_domains": len({row.get("eval_domain") for row in selected}),
            "unique_groups": len({row.get("group_kind") for row in selected}),
            "blocker_classes": dict(Counter(blocker_class(row.get("patched_blocker_token")) for row in selected)),
            "new_vs_phase924": sum(1 for row in selected if row.get("new_surface_seed_vs_phase924")),
            "median_seed_margin": median([row.get("patched_eos_margin_vs_blocker") for row in selected]),
            "median_seed_rank": median([row.get("patched_eos_rank") for row in selected]),
        },
        **surface_payload,
        "top_surfaces": surfaces[:120],
        "evidence_label": label,
        "boundary": (
            "Phase926 runs new model forwards on a balanced subset of Phase925 response-surface seeds. "
            "It tests whether route_alpha by protocol_span_factor surfaces generalize beyond the Phase924 fish-heavy set."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_phase925_seeds(args.model, args)
    if args.dry_run or not selected:
        payload = summarize_model(args.model, selected, [], [], {}, args, None)
        payload["status"] = "dry_run" if args.dry_run else "no_phase925_surface_seeds"
        p846.write_json(out_dir / f"phase926_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase926_{args.model}_selected_seeds.jsonl", selected)
        p846.write_jsonl(out_dir / f"phase926_{args.model}_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase926_{args.model}_surfaces.jsonl", [])
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
    alpha_values = parse_floats(args.route_alphas)
    protocol_factors = parse_floats(args.protocol_factors)
    l39_factors = parse_floats(args.l39_factors)
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
        for idx, seed in enumerate(selected, 1):
            state = reconstruct_seed_state(model, tokenizer, device, groups, case_map, seed, args)
            if state is not None:
                states.append(state)
            log(f"{args.model}/{args.round_name}: reconstructed_seed={idx}/{len(selected)} kept={len(states)}")
        consensus_group, consensus_diag = p920.consensus_group(states, "margin_support_pos_64", int(args.group_budget))
        for state_idx, state in enumerate(states, 1):
            seed = state["source_row"]
            for l39_factor in l39_factors:
                for alpha in alpha_values:
                    for protocol_factor in protocol_factors:
                        spec = p924.surface_spec(float(alpha), float(protocol_factor), args.protocol_span_kind)
                        patched_logits = p922.logits_with_coupled_intervention(
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
                        row = p922.row_from_logits(
                            tokenizer, state, consensus_group, float(l39_factor), spec, patched_logits, groups
                        )
                        row["phase"] = PHASE
                        row["row_kind"] = "phase926_generalized_route_protocol_surface_row"
                        row["phase925_surface_state_key"] = seed.get("surface_state_key")
                        row["phase925_group_kind"] = seed.get("group_kind")
                        row["phase925_factor"] = seed.get("factor")
                        row["phase925_seed_score"] = seed.get("score_scalar")
                        row["phase925_seed_blocker_token"] = seed.get("patched_blocker_token")
                        row["phase925_seed_blocker_class"] = blocker_class(seed.get("patched_blocker_token"))
                        row["phase925_seed_margin"] = seed.get("patched_eos_margin_vs_blocker")
                        row["phase925_seed_rank"] = seed.get("patched_eos_rank")
                        row["phase925_new_surface_seed_vs_phase924"] = seed.get("new_surface_seed_vs_phase924")
                        rows.append(row)
            if state_idx % max(1, int(args.log_every)) == 0 or state_idx == len(states):
                log(f"{args.model}/{args.round_name}: generalized_surface_state={state_idx}/{len(states)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        del states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    p924.annotate_vs_surface_base(rows)
    surfaces = p924.surface_summaries(rows)
    for surface in surfaces:
        related = next((row for row in rows if row.get("target_state_key") == surface.get("target_state_key")), {})
        surface["phase925_group_kind"] = related.get("phase925_group_kind")
        surface["phase925_seed_blocker_class"] = related.get("phase925_seed_blocker_class")
        surface["phase925_seed_blocker_token"] = related.get("phase925_seed_blocker_token")
        surface["phase925_new_surface_seed_vs_phase924"] = related.get("phase925_new_surface_seed_vs_phase924")
    payload = summarize_model(args.model, selected, rows, surfaces, consensus_diag, args, attn_impl)
    p846.write_json(out_dir / f"phase926_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase926_{args.model}_selected_seeds.jsonl", selected)
    p846.write_jsonl(out_dir / f"phase926_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase926_{args.model}_surfaces.jsonl", surfaces)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "selected": len(selected),
                "rows": len(rows),
                "surface_summary": payload["surface_summary"],
                "evidence_label": payload["evidence_label"],
            },
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
    top_surfaces = []
    alpha_protocol = []
    by_domain = []
    by_blocker = []
    for model_name in MODELS:
        path = out_dir / f"phase926_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        scalar["phase925_selected_seed_count"] += int(summary.get("phase925_selected_seed_count") or 0)
        scalar["expected_rows_if_all_reconstructed"] += int(summary.get("expected_rows_if_all_reconstructed") or 0)
        overall = summary.get("overall") or {}
        for scope in ["all", "surface_base", "non_base"]:
            scoped = overall.get(scope) or {}
            for key in [
                "rows",
                "top1",
                "margin_nonnegative",
                "strict_clean_candidate",
                "improved_margin_vs_surface_base",
                "new_margin_closure_vs_surface_base",
                "new_top1_vs_surface_base",
                "new_strict_vs_surface_base",
                "lost_margin_closure_vs_surface_base",
            ]:
                scalar[f"{scope}_{key}"] += int(scoped.get(key) or 0)
        ssum = summary.get("surface_summary") or {}
        for key in [
            "surface_count",
            "best_coord_is_base",
            "best_alpha_lt_1",
            "best_alpha_eq_1",
            "best_alpha_gt_1",
            "best_protocol_lt_1",
            "best_protocol_eq_1",
            "best_protocol_gt_1",
            "with_closure_coord",
        ]:
            scalar[f"surfaces_{key}"] += int(ssum.get(key) or 0)
        scalar["target_state_count"] += int(overall.get("target_state_count") or 0)
        for source_key, target in [
            ("top_surfaces", top_surfaces),
            ("by_alpha_protocol", alpha_protocol),
            ("by_domain", by_domain),
            ("by_seed_blocker_class", by_blocker),
        ]:
            for row in summary.get(source_key) or []:
                item = dict(row)
                item["model"] = summary.get("model")
                target.append(item)
    sort_rows = lambda row: (
        row.get("new_strict_vs_surface_base") or 0,
        row.get("new_top1_vs_surface_base") or 0,
        row.get("new_margin_closure_vs_surface_base") or 0,
        row.get("improved_margin_vs_surface_base") or 0,
        row.get("mean_margin_delta_vs_surface_base") or -9999,
    )
    top_surfaces.sort(key=lambda row: (row.get("closure_coord_count") or 0, row.get("best_margin_delta_vs_surface_base") or -9999), reverse=True)
    alpha_protocol.sort(key=sort_rows, reverse=True)
    by_domain.sort(key=sort_rows, reverse=True)
    by_blocker.sort(key=sort_rows, reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_surfaces": top_surfaces[:160],
        "top_alpha_protocol": alpha_protocol[:160],
        "top_domain_rows": by_domain[:80],
        "top_blocker_class_rows": by_blocker[:80],
    }
    p846.write_json(out_dir / "phase926_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase926_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 926 generalized route-protocol surface validation",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | selected | states | rows | surfaces | best base | best alpha<1 | best alpha>1 | best prot<1 | best prot>1 | non-base new top1 | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        all_rows = overall.get("all") or {}
        non_base = overall.get("non_base") or {}
        surfaces = summary.get("surface_summary") or {}
        lines.append(
            "| {model} | {selected} | {states} | {rows} | {surface_count} | {best_base} | {alt} | {agt} | {plt} | {pgt} | {new_top1} | {evidence} |".format(
                model=summary.get("model"),
                selected=summary.get("phase925_selected_seed_count"),
                states=overall.get("target_state_count"),
                rows=all_rows.get("rows"),
                surface_count=surfaces.get("surface_count"),
                best_base=surfaces.get("best_coord_is_base"),
                alt=surfaces.get("best_alpha_lt_1"),
                agt=surfaces.get("best_alpha_gt_1"),
                plt=surfaces.get("best_protocol_lt_1"),
                pgt=surfaces.get("best_protocol_gt_1"),
                new_top1=non_base.get("new_top1_vs_surface_base"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Surfaces", ""])
    lines.append(
        "| model | state | domain | blocker class | group | factor | best alpha | best protocol | best margin | base margin | delta | closures |"
    )
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("top_surfaces") or []:
        row = {
            "model": "",
            "target_state_key": "",
            "target_eval_domain": "",
            "phase925_seed_blocker_class": "",
            "phase925_group_kind": "",
            "l39_factor": "",
            "best_alpha": "",
            "best_protocol_factor": "",
            "best_margin": "",
            "base_margin": "",
            "best_margin_delta_vs_surface_base": "",
            "closure_coords": [],
            **row,
        }
        lines.append(
            "| {model} | {target_state_key} | {target_eval_domain} | {phase925_seed_blocker_class} | {phase925_group_kind} | {l39_factor} | {best_alpha} | {best_protocol_factor} | {best_margin} | {base_margin} | {best_margin_delta_vs_surface_base} | {closure_coords} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Alpha Protocol Coordinates", ""])
    lines.append("| model | alpha | protocol | rows | top1 | margin | strict | improved | new margin | new top1 | mean delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_alpha_protocol") or []:
        row = {
            "model": "",
            "route_alpha": "",
            "protocol_span_factor": "",
            "rows": 0,
            "top1": 0,
            "margin_nonnegative": 0,
            "strict_clean_candidate": 0,
            "improved_margin_vs_surface_base": 0,
            "new_margin_closure_vs_surface_base": 0,
            "new_top1_vs_surface_base": 0,
            "mean_margin_delta_vs_surface_base": None,
            **row,
        }
        lines.append(
            "| {model} | {route_alpha} | {protocol_span_factor} | {rows} | {top1} | {margin_nonnegative} | {strict_clean_candidate} | {improved_margin_vs_surface_base} | {new_margin_closure_vs_surface_base} | {new_top1_vs_surface_base} | {mean_margin_delta_vs_surface_base} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="generalized_route_protocol_surface_validation")
    parser.add_argument("--phase925-round", default="response_surface_generalization_dataset_expansion")
    parser.add_argument("--max-seeds-per-model", type=int, default=30)
    parser.add_argument("--min-seeds-per-model", type=int, default=24)
    parser.add_argument("--max-per-case", type=int, default=4)
    parser.add_argument("--max-per-domain", type=int, default=12)
    parser.add_argument("--max-per-group", type=int, default=12)
    parser.add_argument("--max-per-blocker-class", type=int, default=18)
    parser.add_argument("--target-layer", type=int, default=39)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--l4-candidate-pool", type=int, default=512)
    parser.add_argument("--channel-candidate-pool", type=int, default=768)
    parser.add_argument("--band-size", type=int, default=32)
    parser.add_argument("--group-budget", type=int, default=64)
    parser.add_argument("--l39-factors", default="1.25,1.375")
    parser.add_argument("--route-alphas", default="0.75,0.875,1.0,1.125,1.25,1.375")
    parser.add_argument("--protocol-span-kind", default="last8_before_period")
    parser.add_argument("--protocol-factors", default="0.85,0.9,1.0,1.1")
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
