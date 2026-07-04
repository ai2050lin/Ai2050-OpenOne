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

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase888_direction_set_internal_subspace_probe as p888  # noqa: E402
import phase893_attention_head_complementarity_holdout_probe as p893  # noqa: E402
import phase894_weak_no_single_closure_rollout_probe as p894  # noqa: E402


PHASE = 895
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase895_no_single_minimality_head_pathway_split")
PHASE894_ROOT = Path("tests/result/phase894_weak_no_single_closure_rollout_probe")
PHASE894_ROUND = "weak_no_single_closure_rollout"
FOCUS_SUBSET = {
    "qwen3": "L31C2257",
    "glm4": "L31C6437",
    "deepseek7b": "L26C8587+L27C15369",
}
HEAD_SETS = {
    "qwen3": [
        "none",
        "L31H19",
        "L31H26",
        "L31H30",
        "L31H12",
        "L31H17",
        "L31H19+L31H26+L31H30+L31H12+L31H17",
    ],
    "glm4": ["none"],
    "deepseek7b": [
        "none",
        "L26H3",
        "L26H7",
        "L26H11",
        "L26H14",
        "L26H3+L26H7",
        "L26H3+L26H11",
        "L26H7+L26H11",
        "L26H3+L26H7+L26H11+L26H14",
    ],
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def counter_values(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def phase894_path(round_name: str, model: str, suffix: str) -> Path:
    return PHASE894_ROOT / round_name / f"phase894_{model}_{suffix}.jsonl"


def condition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))


def condition_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    case = dict(p894.case_map().get(str(row.get("case_id"))) or {})
    if not case:
        return None
    case["prompt_variant"] = row.get("prompt_variant")
    case["edit_mode"] = row.get("edit_mode")
    case["condition_source"] = row.get("condition_source")
    case["case_split"] = row.get("case_split")
    case["phase894_subset_key"] = row.get("subset_key")
    return case


def phase894_selected_conditions(model_name: str, round_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = p846.read_jsonl(phase894_path(round_name, model_name, "first_rows"))
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    if model_name == "deepseek7b":
        sources = [
            row
            for row in rows
            if str(row.get("subset_key")) == "L26C8587+L27C15369" and row.get("closure_without_single_axis_closure")
        ]
        sources.sort(
            key=lambda row: (
                str(row.get("condition_source")) == "phase893_exact_no_single",
                finite(row.get("complementarity_over_best_single")),
                finite(row.get("target_lift")),
            ),
            reverse=True,
        )
    elif model_name == "qwen3":
        sources = [row for row in rows if str(row.get("subset_key")) == "L31C2257" and row.get("closure_from_open")]
        sources.sort(key=lambda row: finite(row.get("target_lift")), reverse=True)
        sources = sources[: int(args.max_qwen_conditions)]
    else:
        sources = [row for row in rows if str(row.get("subset_key")) == "L31C6437"]
        sources = sources[: int(args.max_glm4_conditions)]
    for row in sources:
        key = condition_key(row)
        if key in seen:
            continue
        condition = condition_from_row(row)
        if condition is None:
            continue
        seen.add(key)
        out.append(condition)
    return out


def subset_specs(model_name: str, model_gears: list[dict[str, Any]], max_subset_size: int) -> list[dict[str, Any]]:
    specs = p893.subset_specs(model_name, model_gears, max_subset_size)
    if model_name == "deepseek7b":
        wanted = {
            "L26C8587",
            "L27C15369",
            "L27C16651",
            "L26C8587+L27C15369",
            "L26C8587+L27C16651",
            "L27C15369+L27C16651",
            "L26C8587+L27C15369+L27C16651",
        }
        specs = [spec for spec in specs if str(spec.get("subset_key")) in wanted]
    return specs


def target_lift(base_metrics: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    base_class = base_metrics.get("class_best_logit")
    current_class = metrics.get("class_best_logit")
    if base_class is None or current_class is None:
        return None
    return finite(current_class) - finite(base_class)


def class_blocker_summary(blocker: dict[str, Any], max_items: int = 8) -> tuple[list[dict[str, Any]], dict[str, int]]:
    blockers = []
    roles: Counter[str] = Counter()
    for item in (blocker.get("class_top_blockers") or [])[: int(max_items)]:
        roles[str(item.get("role") or "unknown")] += 1
        blockers.append(
            {
                "token_id": item.get("token_id"),
                "token": item.get("token"),
                "role": item.get("role"),
                "gap": item.get("gap_vs_threshold"),
            }
        )
    return blockers, counter_values(roles)


def metrics_for_logits(tokenizer, logits: torch.Tensor, token_sets: dict[str, Any], topk_tokens: int, topk_blockers: int) -> dict[str, Any]:
    first = p888.metrics_for_logits(tokenizer, logits, token_sets, int(topk_tokens))
    blocker = p862.p854.blocker_metrics(tokenizer, logits, token_sets, int(topk_blockers))
    blockers, role_counts = class_blocker_summary(blocker)
    return {
        **first,
        "full_class_rank": blocker.get("class_best_target_rank"),
        "full_class_blocker_count": blocker.get("class_blocker_count"),
        "full_class_top_blocker_token": blocker.get("class_top_blocker_token"),
        "full_class_top_blocker_role": blocker.get("class_top_blocker_role"),
        "full_class_top_blocker_gap": blocker.get("class_top_blocker_gap"),
        "full_class_top_blockers_compact": blockers,
        "full_class_top_blocker_role_counts": role_counts,
        "full_class_minus_object_logit": blocker.get("class_minus_object_logit"),
        "full_top_tokens_compact": [
            {
                "token_id": item.get("token_id"),
                "token": item.get("token"),
                "role": item.get("role"),
                "logit": item.get("logit"),
            }
            for item in (blocker.get("top_tokens") or [])[: int(topk_tokens)]
        ],
    }


def blocker_reduction(base_metrics: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    before = base_metrics.get("full_class_blocker_count")
    after = metrics.get("full_class_blocker_count")
    if before is None or after is None:
        return None
    return finite(before) - finite(after)


def make_minimality_row(
    model_name: str,
    condition: dict[str, Any],
    spec: dict[str, Any],
    base_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    lift = target_lift(base_metrics, metrics)
    reduction = blocker_reduction(base_metrics, metrics)
    return {
        "phase": PHASE,
        "row_kind": "phase895_minimality_row",
        "model": model_name,
        "case_id": condition.get("case_id"),
        "case_split": condition.get("case_split"),
        "condition_source": condition.get("condition_source"),
        "eval_domain": condition.get("domain"),
        "object": condition.get("object"),
        "prompt_variant": condition.get("prompt_variant"),
        "edit_mode": condition.get("edit_mode"),
        "subset_key": spec.get("subset_key"),
        "subset_size": spec.get("subset_size"),
        "subset_relation": spec.get("subset_relation"),
        "gear_keys": spec.get("gear_keys"),
        "base_boundary_closed": bool(base_metrics.get("class_boundary_closed")),
        "boundary_closed": bool(metrics.get("class_boundary_closed")),
        "closure_from_open": bool((not base_metrics.get("class_boundary_closed")) and metrics.get("class_boundary_closed")),
        "target_lift": lift,
        "base_class_rank": base_metrics.get("class_best_rank"),
        "class_rank": metrics.get("class_best_rank"),
        "base_full_class_blocker_count": base_metrics.get("full_class_blocker_count"),
        "full_class_blocker_count": metrics.get("full_class_blocker_count"),
        "full_blocker_reduction": reduction,
        "full_top_blocker_token": metrics.get("full_class_top_blocker_token"),
        "full_top_blocker_role": metrics.get("full_class_top_blocker_role"),
        "full_top_blocker_gap": metrics.get("full_class_top_blocker_gap"),
        "full_top_blockers_compact": metrics.get("full_class_top_blockers_compact"),
        "full_top_blocker_role_counts": metrics.get("full_class_top_blocker_role_counts"),
        "class_minus_object_logit": metrics.get("full_class_minus_object_logit"),
    }


def add_condition_minimality(rows: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        groups[key].append(row)
    summaries: list[dict[str, Any]] = []
    focus_key = FOCUS_SUBSET.get(model_name, "")
    for key, vals in groups.items():
        by_subset = {str(row.get("subset_key")): row for row in vals}
        focus = by_subset.get(focus_key)
        singles = [row for row in vals if int(row.get("subset_size") or 0) == 1]
        pairs = [row for row in vals if int(row.get("subset_size") or 0) == 2]
        alternatives = [row for row in pairs if str(row.get("subset_key")) != focus_key]
        model_u = next((row for row in vals if str(row.get("subset_relation")) == "model_U" or int(row.get("subset_size") or 0) >= 3), None)
        any_single = any(row.get("closure_from_open") for row in singles)
        any_alt_pair = any(row.get("closure_from_open") for row in alternatives)
        focus_closure = bool(focus and focus.get("closure_from_open"))
        known_axis_minimal = bool(focus_closure and not any_single and not any_alt_pair)
        model_u_not_required = bool(focus_closure and (not model_u or model_u.get("closure_from_open")))
        for row in vals:
            row["focus_subset_key"] = focus_key
            row["focus_closure_from_open"] = focus_closure
            row["any_single_axis_closure"] = bool(any_single)
            row["any_alternative_pair_closure"] = bool(any_alt_pair)
            row["known_axis_minimal_candidate"] = known_axis_minimal
            row["model_u_not_required_for_focus_closure"] = model_u_not_required
        if focus:
            summaries.append(
                {
                    "phase": PHASE,
                    "row_kind": "phase895_condition_summary",
                    "model": model_name,
                    "case_id": key[1],
                    "prompt_variant": key[2],
                    "edit_mode": key[3],
                    "eval_domain": focus.get("eval_domain"),
                    "object": focus.get("object"),
                    "condition_source": focus.get("condition_source"),
                    "focus_subset_key": focus_key,
                    "focus_closure_from_open": focus_closure,
                    "known_axis_minimal_candidate": known_axis_minimal,
                    "any_single_axis_closure": bool(any_single),
                    "single_closure_keys": sorted(str(row.get("subset_key")) for row in singles if row.get("closure_from_open")),
                    "any_alternative_pair_closure": bool(any_alt_pair),
                    "alternative_pair_closure_keys": sorted(str(row.get("subset_key")) for row in alternatives if row.get("closure_from_open")),
                    "model_u_closure_from_open": bool(model_u and model_u.get("closure_from_open")),
                    "model_u_not_required_for_focus_closure": model_u_not_required,
                    "focus_target_lift": focus.get("target_lift"),
                    "focus_blocker_reduction": focus.get("full_blocker_reduction"),
                    "focus_full_class_blocker_count": focus.get("full_class_blocker_count"),
                    "focus_top_blocker_role": focus.get("full_top_blocker_role"),
                    "focus_top_blocker_token": focus.get("full_top_blocker_token"),
                }
            )
    return summaries


def function_label(row: dict[str, Any]) -> str:
    if str(row.get("head_set")) == "none":
        return "none_control"
    target_damage = finite(row.get("target_lift_damage_vs_none"))
    blocker_damage = finite(row.get("blocker_reduction_damage_vs_none"))
    closure_lost = bool(row.get("closure_lost_vs_none"))
    if closure_lost and target_damage > 0.25 and blocker_damage > 0:
        return "target_and_blocker_boundary_candidate"
    if closure_lost and target_damage > 0.25:
        return "target_lift_boundary_candidate"
    if closure_lost and blocker_damage > 0:
        return "blocker_boundary_candidate"
    if target_damage > 0.25:
        return "target_lift_damage_candidate"
    if blocker_damage > 0:
        return "blocker_reduction_damage_candidate"
    return "weak_or_no_damage"


def make_head_row(
    model_name: str,
    condition: dict[str, Any],
    subset_key: str,
    head_set: str,
    base_metrics: dict[str, Any],
    none_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    none_lift = target_lift(base_metrics, none_metrics)
    lift = target_lift(base_metrics, metrics)
    none_reduction = blocker_reduction(base_metrics, none_metrics)
    reduction = blocker_reduction(base_metrics, metrics)
    row = {
        "phase": PHASE,
        "row_kind": "phase895_head_split_row",
        "model": model_name,
        "case_id": condition.get("case_id"),
        "case_split": condition.get("case_split"),
        "condition_source": condition.get("condition_source"),
        "eval_domain": condition.get("domain"),
        "object": condition.get("object"),
        "prompt_variant": condition.get("prompt_variant"),
        "edit_mode": condition.get("edit_mode"),
        "subset_key": subset_key,
        "head_set": head_set,
        "head_count": 0 if head_set == "none" else len(head_set.split("+")),
        "none_boundary_closed": bool(none_metrics.get("class_boundary_closed")),
        "head_boundary_closed": bool(metrics.get("class_boundary_closed")),
        "closure_lost_vs_none": bool(none_metrics.get("class_boundary_closed") and not metrics.get("class_boundary_closed")),
        "none_target_lift": none_lift,
        "head_target_lift": lift,
        "target_lift_damage_vs_none": None if none_lift is None or lift is None else none_lift - lift,
        "none_blocker_reduction": none_reduction,
        "head_blocker_reduction": reduction,
        "blocker_reduction_damage_vs_none": None if none_reduction is None or reduction is None else none_reduction - reduction,
        "none_full_class_blocker_count": none_metrics.get("full_class_blocker_count"),
        "head_full_class_blocker_count": metrics.get("full_class_blocker_count"),
        "blocker_count_delta_vs_none": None
        if none_metrics.get("full_class_blocker_count") is None or metrics.get("full_class_blocker_count") is None
        else finite(metrics.get("full_class_blocker_count")) - finite(none_metrics.get("full_class_blocker_count")),
        "none_class_rank": none_metrics.get("class_best_rank"),
        "head_class_rank": metrics.get("class_best_rank"),
        "class_rank_delta_vs_none": None
        if none_metrics.get("class_best_rank") is None or metrics.get("class_best_rank") is None
        else finite(metrics.get("class_best_rank")) - finite(none_metrics.get("class_best_rank")),
        "head_top_blocker_role": metrics.get("full_class_top_blocker_role"),
        "head_top_blocker_token": metrics.get("full_class_top_blocker_token"),
        "head_top_blocker_role_counts": metrics.get("full_class_top_blocker_role_counts"),
    }
    row["pathway_label"] = function_label(row)
    return row


def protocol_drift(text: str) -> bool:
    norm = p894.normalize(text)
    markers = ["answer:", "category:", "item:", "{", "}", "\n", "another example"]
    return any(marker in norm for marker in markers)


def make_rollout_row(
    model_name: str,
    condition: dict[str, Any],
    subset_key: str,
    head_set: str,
    text: str,
    ids: list[int],
    class_hit: bool,
    object_echo: bool,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "row_kind": "phase895_rollout_pathway_row",
        "model": model_name,
        "case_id": condition.get("case_id"),
        "case_split": condition.get("case_split"),
        "condition_source": condition.get("condition_source"),
        "eval_domain": condition.get("domain"),
        "object": condition.get("object"),
        "prompt_variant": condition.get("prompt_variant"),
        "edit_mode": condition.get("edit_mode"),
        "subset_key": subset_key,
        "head_set": head_set,
        "generated_text": text,
        "generated_ids": ids,
        "rollout_class_hit": bool(class_hit),
        "rollout_object_echo": bool(object_echo),
        "rollout_protocol_drift": protocol_drift(text),
        "answer_like_no_echo": bool(class_hit and not object_echo and not protocol_drift(text)),
    }


def add_rollout_deltas(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("edit_mode")),
            str(row.get("subset_key")),
        )
        groups[key].append(row)
    for vals in groups.values():
        none = next((row for row in vals if str(row.get("head_set")) == "none"), None)
        if not none:
            continue
        for row in vals:
            row["rollout_class_lost_vs_none"] = bool(none.get("rollout_class_hit") and not row.get("rollout_class_hit"))
            row["rollout_object_echo_added_vs_none"] = bool((not none.get("rollout_object_echo")) and row.get("rollout_object_echo"))
            row["rollout_protocol_drift_added_vs_none"] = bool((not none.get("rollout_protocol_drift")) and row.get("rollout_protocol_drift"))


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_gears = p893.load_model_gears(args.phase893_round, args.model)
    specs = subset_specs(args.model, model_gears, int(args.max_subset_size))
    conditions = phase894_selected_conditions(args.model, args.phase894_round, args)
    if args.dry_run or not model_gears or not specs or not conditions:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run" if model_gears and specs and conditions else "no_gears_or_conditions",
            "model_gear_keys": [p894.gear_key(gear) for gear in model_gears],
            "selected_conditions": len(conditions),
            "subset_specs": [spec.get("subset_key") for spec in specs],
        }
        p846.write_json(out_dir / f"phase895_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase895_{args.model}_minimality_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase895_{args.model}_condition_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase895_{args.model}_head_split_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase895_{args.model}_rollout_rows.jsonl", [])
        return payload

    model = None
    tokenizer = None
    minimality_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        spec_by_key = {str(spec.get("subset_key")): spec for spec in specs}
        focus_key = FOCUS_SUBSET.get(args.model, "")
        for idx, condition in enumerate(conditions, 1):
            prompt = p885.prompt_for_case(condition, str(condition.get("prompt_variant")))
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            token_sets = p856.token_sets(tokenizer, condition)
            base_logits = p894.first_logits_with_gears_and_heads(
                model, device, prompt_ids, [], "original", float(args.scale_up_factor), []
            )
            base_metrics = metrics_for_logits(tokenizer, base_logits, token_sets, int(args.topk_tokens), int(args.topk_blockers))
            condition_metrics: dict[str, dict[str, Any]] = {}
            for spec in specs:
                logits = p894.first_logits_with_gears_and_heads(
                    model,
                    device,
                    prompt_ids,
                    spec["gears"],
                    str(condition.get("edit_mode")),
                    float(args.scale_up_factor),
                    [],
                )
                metrics = metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens), int(args.topk_blockers))
                condition_metrics[str(spec.get("subset_key"))] = metrics
                minimality_rows.append(make_minimality_row(args.model, condition, spec, base_metrics, metrics))
            condition_rows = add_condition_minimality(minimality_rows, args.model)

            focus_spec = spec_by_key.get(focus_key)
            focus_metrics = condition_metrics.get(focus_key)
            if focus_spec is not None and focus_metrics is not None:
                for head_set in HEAD_SETS.get(args.model, ["none"]):
                    heads = p894.parse_head_set(head_set)
                    logits = p894.first_logits_with_gears_and_heads(
                        model,
                        device,
                        prompt_ids,
                        focus_spec["gears"],
                        str(condition.get("edit_mode")),
                        float(args.scale_up_factor),
                        heads,
                    )
                    metrics = metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens), int(args.topk_blockers))
                    head_rows.append(make_head_row(args.model, condition, focus_key, p894.head_set_key(heads), base_metrics, focus_metrics, metrics))
                    if args.model != "glm4":
                        text, ids = p894.greedy_with_gears_and_heads(
                            model,
                            tokenizer,
                            device,
                            prompt_ids,
                            focus_spec["gears"],
                            str(condition.get("edit_mode")),
                            float(args.scale_up_factor),
                            heads,
                            int(args.max_new_tokens),
                        )
                        class_hit = p894.text_hits_any(text, list(condition.get("answer_aliases") or []))
                        object_echo = p894.text_hits_any(text, [str(condition.get("object"))])
                        rollout_rows.append(make_rollout_row(args.model, condition, focus_key, p894.head_set_key(heads), text, ids, class_hit, object_echo))
            log(
                f"{args.model}/{args.round_name}: condition={idx}/{len(conditions)} "
                f"minimality_rows={len(minimality_rows)} head_rows={len(head_rows)} rollout_rows={len(rollout_rows)}"
            )
        condition_rows = add_condition_minimality(minimality_rows, args.model)
        add_rollout_deltas(rollout_rows)
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, minimality_rows, condition_rows, head_rows, rollout_rows, conditions, model_gears, specs, attn_impl)
    p846.write_json(out_dir / f"phase895_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase895_{args.model}_minimality_rows.jsonl", minimality_rows)
    p846.write_jsonl(out_dir / f"phase895_{args.model}_condition_rows.jsonl", condition_rows)
    p846.write_jsonl(out_dir / f"phase895_{args.model}_head_split_rows.jsonl", head_rows)
    p846.write_jsonl(out_dir / f"phase895_{args.model}_rollout_rows.jsonl", rollout_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_model(
    model_name: str,
    minimality_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    model_gears: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    attn_impl: str | None,
) -> dict[str, Any]:
    focus_key = FOCUS_SUBSET.get(model_name, "")
    focus_rows = [row for row in minimality_rows if str(row.get("subset_key")) == focus_key]
    known_minimal = [row for row in condition_rows if row.get("known_axis_minimal_candidate")]
    alt_pair = [row for row in condition_rows if row.get("any_alternative_pair_closure")]
    any_single = [row for row in condition_rows if row.get("any_single_axis_closure")]
    model_u_not_required = [row for row in condition_rows if row.get("model_u_not_required_for_focus_closure")]
    head_lost = [row for row in head_rows if row.get("closure_lost_vs_none")]
    target_damage = [
        row
        for row in head_rows
        if row.get("target_lift_damage_vs_none") is not None and finite(row.get("target_lift_damage_vs_none")) > 0.25
    ]
    blocker_damage = [
        row
        for row in head_rows
        if row.get("blocker_reduction_damage_vs_none") is not None and finite(row.get("blocker_reduction_damage_vs_none")) > 0
    ]
    rollout_loss = [row for row in rollout_rows if row.get("rollout_class_lost_vs_none")]
    protocol_added = [row for row in rollout_rows if row.get("rollout_protocol_drift_added_vs_none")]
    object_echo_added = [row for row in rollout_rows if row.get("rollout_object_echo_added_vs_none")]

    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in minimality_rows:
        by_subset[str(row.get("subset_key"))].append(row)
    subset_groups = []
    for key, vals in by_subset.items():
        subset_groups.append(
            {
                "model": model_name,
                "subset_key": key,
                "n_rows": len(vals),
                "closure_from_open": sum(1 for row in vals if row.get("closure_from_open")),
                "mean_target_lift": mean([finite(row.get("target_lift")) for row in vals if row.get("target_lift") is not None]) or 0.0,
                "mean_blocker_reduction": mean(
                    [finite(row.get("full_blocker_reduction")) for row in vals if row.get("full_blocker_reduction") is not None]
                )
                or 0.0,
                "top_blocker_roles": counter_values(Counter(str(row.get("full_top_blocker_role") or "none") for row in vals)),
            }
        )
    subset_groups.sort(key=lambda row: (row.get("closure_from_open") or 0, row.get("mean_target_lift") or 0.0), reverse=True)

    by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in head_rows:
        by_head[str(row.get("head_set"))].append(row)
    head_groups = []
    for key, vals in by_head.items():
        head_groups.append(
            {
                "model": model_name,
                "head_set": key,
                "n_rows": len(vals),
                "closure_lost": sum(1 for row in vals if row.get("closure_lost_vs_none")),
                "target_damage_gt_0_25": sum(
                    1
                    for row in vals
                    if row.get("target_lift_damage_vs_none") is not None and finite(row.get("target_lift_damage_vs_none")) > 0.25
                ),
                "blocker_damage_gt_0": sum(
                    1
                    for row in vals
                    if row.get("blocker_reduction_damage_vs_none") is not None and finite(row.get("blocker_reduction_damage_vs_none")) > 0
                ),
                "mean_target_damage": mean(
                    [finite(row.get("target_lift_damage_vs_none")) for row in vals if row.get("target_lift_damage_vs_none") is not None]
                )
                or 0.0,
                "mean_blocker_damage": mean(
                    [
                        finite(row.get("blocker_reduction_damage_vs_none"))
                        for row in vals
                        if row.get("blocker_reduction_damage_vs_none") is not None
                    ]
                )
                or 0.0,
                "pathway_labels": counter_values(Counter(str(row.get("pathway_label")) for row in vals)),
            }
        )
    head_groups.sort(key=lambda row: (row.get("closure_lost") or 0, row.get("mean_target_damage") or 0.0), reverse=True)

    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        by_rollout[str(row.get("head_set"))].append(row)
    rollout_groups = []
    for key, vals in by_rollout.items():
        rollout_groups.append(
            {
                "model": model_name,
                "head_set": key,
                "n_rows": len(vals),
                "class_hit": sum(1 for row in vals if row.get("rollout_class_hit")),
                "answer_like_no_echo": sum(1 for row in vals if row.get("answer_like_no_echo")),
                "object_echo": sum(1 for row in vals if row.get("rollout_object_echo")),
                "protocol_drift": sum(1 for row in vals if row.get("rollout_protocol_drift")),
                "class_lost_vs_none": sum(1 for row in vals if row.get("rollout_class_lost_vs_none")),
                "object_echo_added_vs_none": sum(1 for row in vals if row.get("rollout_object_echo_added_vs_none")),
                "protocol_drift_added_vs_none": sum(1 for row in vals if row.get("rollout_protocol_drift_added_vs_none")),
            }
        )
    rollout_groups.sort(key=lambda row: (row.get("class_lost_vs_none") or 0, row.get("protocol_drift_added_vs_none") or 0), reverse=True)

    evidence_label = "negative_or_control"
    if model_name == "deepseek7b" and known_minimal and head_lost:
        evidence_label = "known_axis_minimal_pair_with_head_pathway_split_candidate"
    elif known_minimal:
        evidence_label = "known_axis_minimal_pair_candidate"
    elif head_lost:
        evidence_label = "head_pathway_fragility_without_pair_minimality"
    elif focus_rows and any(row.get("closure_from_open") for row in focus_rows):
        evidence_label = "focus_closure_without_minimality"

    return {
        "phase": PHASE,
        "title": "No-Single Closure Minimality and Multi-Head Pathway Split",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "model_gear_keys": [p894.gear_key(gear) for gear in model_gears],
        "subset_specs": [spec.get("subset_key") for spec in specs],
        "focus_subset_key": focus_key,
        "selected_conditions": len(conditions),
        "output_minimality_rows": len(minimality_rows),
        "output_condition_rows": len(condition_rows),
        "output_head_split_rows": len(head_rows),
        "output_rollout_rows": len(rollout_rows),
        "overall": {
            "focus_closure_from_open": sum(1 for row in focus_rows if row.get("closure_from_open")),
            "known_axis_minimal_candidate": len(known_minimal),
            "any_single_axis_closure": len(any_single),
            "any_alternative_pair_closure": len(alt_pair),
            "model_u_not_required_for_focus_closure": len(model_u_not_required),
            "head_closure_lost": len(head_lost),
            "head_target_damage_gt_0_25": len(target_damage),
            "head_blocker_damage_gt_0": len(blocker_damage),
            "rollout_class_hit": sum(1 for row in rollout_rows if row.get("rollout_class_hit")),
            "rollout_answer_like_no_echo": sum(1 for row in rollout_rows if row.get("answer_like_no_echo")),
            "rollout_class_lost_vs_none": len(rollout_loss),
            "rollout_protocol_drift_added_vs_none": len(protocol_added),
            "rollout_object_echo_added_vs_none": len(object_echo_added),
        },
        "subset_groups": subset_groups,
        "head_groups": head_groups,
        "rollout_groups": rollout_groups,
        "evidence_label": evidence_label,
        "boundary": (
            "Phase895 checks known-axis minimality, full-vocabulary blocker ranks, "
            "and whether multi-head zero affects target lift, blocker reduction, or rollout protocol. "
            "It is still not full long-horizon language closure."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 895 no-single closure minimality and multi-head pathway split",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
        f"- selected_conditions: {payload.get('selected_conditions')}",
        f"- output_minimality_rows: {payload.get('output_minimality_rows')}",
        f"- output_condition_rows: {payload.get('output_condition_rows')}",
        f"- output_head_split_rows: {payload.get('output_head_split_rows')}",
        f"- output_rollout_rows: {payload.get('output_rollout_rows')}",
    ]
    for key, value in (payload.get("overall") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Subset groups", "", "| model | subset | rows | closure | mean lift | mean blocker reduction | top roles |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
    for row in payload.get("subset_groups", [])[:40]:
        lines.append(
            "| {model} | {subset} | {rows} | {closure} | {lift:.3f} | {reduction:.3f} | {roles} |".format(
                model=row.get("model"),
                subset=row.get("subset_key"),
                rows=row.get("n_rows"),
                closure=row.get("closure_from_open"),
                lift=finite(row.get("mean_target_lift")),
                reduction=finite(row.get("mean_blocker_reduction")),
                roles=json.dumps(row.get("top_blocker_roles") or {}, ensure_ascii=False),
            )
        )
    lines.extend(["", "## Head groups", "", "| model | head set | rows | closure lost | target damage | blocker damage | mean target damage | mean blocker damage | labels |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in payload.get("head_groups", [])[:40]:
        lines.append(
            "| {model} | {head_set} | {rows} | {lost} | {td} | {bd} | {mtd:.3f} | {mbd:.3f} | {labels} |".format(
                model=row.get("model"),
                head_set=row.get("head_set"),
                rows=row.get("n_rows"),
                lost=row.get("closure_lost"),
                td=row.get("target_damage_gt_0_25"),
                bd=row.get("blocker_damage_gt_0"),
                mtd=finite(row.get("mean_target_damage")),
                mbd=finite(row.get("mean_blocker_damage")),
                labels=json.dumps(row.get("pathway_labels") or {}, ensure_ascii=False),
            )
        )
    lines.extend(["", "## Rollout groups", "", "| model | head set | rows | class hit | answer-like | object echo | protocol drift | class lost | echo added | drift added |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in payload.get("rollout_groups", [])[:40]:
        lines.append(
            f"| {row.get('model')} | {row.get('head_set')} | {row.get('n_rows')} | {row.get('class_hit')} | {row.get('answer_like_no_echo')} | {row.get('object_echo')} | {row.get('protocol_drift')} | {row.get('class_lost_vs_none')} | {row.get('object_echo_added_vs_none')} | {row.get('protocol_drift_added_vs_none')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [
        read_json(out_dir / f"phase895_{model}_summary.json")
        for model in MODELS
        if (out_dir / f"phase895_{model}_summary.json").exists()
    ]
    summaries = [item for item in summaries if item and item.get("status") == "complete"]
    overall = Counter()
    subset_groups: list[dict[str, Any]] = []
    head_groups: list[dict[str, Any]] = []
    rollout_groups: list[dict[str, Any]] = []
    selected_conditions = output_minimality_rows = output_condition_rows = output_head_split_rows = output_rollout_rows = 0
    for summary in summaries:
        selected_conditions += int(summary.get("selected_conditions") or 0)
        output_minimality_rows += int(summary.get("output_minimality_rows") or 0)
        output_condition_rows += int(summary.get("output_condition_rows") or 0)
        output_head_split_rows += int(summary.get("output_head_split_rows") or 0)
        output_rollout_rows += int(summary.get("output_rollout_rows") or 0)
        subset_groups.extend(summary.get("subset_groups") or [])
        head_groups.extend(summary.get("head_groups") or [])
        rollout_groups.extend(summary.get("rollout_groups") or [])
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall[key] += value
    subset_groups.sort(key=lambda row: (row.get("closure_from_open") or 0, row.get("mean_target_lift") or 0.0), reverse=True)
    head_groups.sort(key=lambda row: (row.get("closure_lost") or 0, row.get("mean_target_damage") or 0.0), reverse=True)
    rollout_groups.sort(key=lambda row: (row.get("class_lost_vs_none") or 0, row.get("protocol_drift_added_vs_none") or 0), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "selected_conditions": selected_conditions,
        "output_minimality_rows": output_minimality_rows,
        "output_condition_rows": output_condition_rows,
        "output_head_split_rows": output_head_split_rows,
        "output_rollout_rows": output_rollout_rows,
        "overall": {key: int(value) for key, value in sorted(overall.items())},
        "subset_groups": subset_groups,
        "head_groups": head_groups,
        "rollout_groups": rollout_groups,
        "evidence_label_counts": counter_values(Counter(str(summary.get("evidence_label")) for summary in summaries)),
    }
    p846.write_json(out_dir / "phase895_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase895_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="minimality_head_pathway_split")
    parser.add_argument("--phase893-round", default=p894.PHASE893_ROUND)
    parser.add_argument("--phase894-round", default=PHASE894_ROUND)
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--max-qwen-conditions", type=int, default=11)
    parser.add_argument("--max-glm4-conditions", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--topk-tokens", type=int, default=30)
    parser.add_argument("--topk-blockers", type=int, default=50)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
