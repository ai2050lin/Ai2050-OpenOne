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
import phase893_attention_head_complementarity_holdout_probe as p893  # noqa: E402
import phase894_weak_no_single_closure_rollout_probe as p894  # noqa: E402
import phase895_no_single_minimality_head_pathway_split as p895  # noqa: E402


PHASE = 896
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase896_cross_domain_pair_search_long_rollout")
PHASE895_ROOT = Path("tests/result/phase895_no_single_minimality_head_pathway_split")
PHASE895_ROUND = "minimality_head_pathway_split"
FOCUS_SUBSET = {
    "qwen3": "L31C2257",
    "glm4": "L31C6437",
    "deepseek7b": "L26C8587+L27C15369",
}
HEAD_SETS = {
    "qwen3": [
        "none",
        "L31H19+L31H26+L31H30+L31H12+L31H17",
    ],
    "glm4": ["none"],
    "deepseek7b": [
        "none",
        "L26H7",
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


def condition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))


def all_cases() -> list[dict[str, Any]]:
    rows = p894.all_cases()
    rows.sort(key=lambda case: (str(case.get("domain")), str(case.get("split_source")), str(case.get("object"))))
    return rows


def selected_conditions(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    domains = set(parse_csv(args.domains))
    prompts = parse_csv(args.prompt_variants if model_name == "deepseek7b" else args.control_prompt_variants)
    modes = parse_csv(args.edit_modes)
    max_per_domain = int(args.max_cases_per_domain)
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for case in all_cases():
        domain = str(case.get("domain"))
        if domains and domain not in domains:
            continue
        if max_per_domain > 0 and counts[domain] >= max_per_domain:
            continue
        counts[domain] += 1
        for prompt_variant in prompts:
            for mode in modes:
                item = dict(case)
                item["prompt_variant"] = prompt_variant
                item["edit_mode"] = mode
                item["case_split"] = case.get("split_source", "phase856_base")
                item["condition_source"] = case.get("split_source", "phase856_base")
                out.append(item)
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


def phase895_known_keys(model_name: str, round_name: str) -> set[tuple[str, str, str]]:
    path = PHASE895_ROOT / round_name / f"phase895_{model_name}_condition_rows.jsonl"
    rows = p846.read_jsonl(path) if path.exists() else []
    return {
        condition_key(row)
        for row in rows
        if row.get("known_axis_minimal_candidate") and str(row.get("focus_subset_key")) == FOCUS_SUBSET.get(model_name, "")
    }


def make_search_row(
    model_name: str,
    condition: dict[str, Any],
    spec: dict[str, Any],
    base_metrics: dict[str, Any],
    metrics: dict[str, Any],
    phase895_keys: set[tuple[str, str, str]],
) -> dict[str, Any]:
    lift = p895.target_lift(base_metrics, metrics)
    reduction = p895.blocker_reduction(base_metrics, metrics)
    return {
        "phase": PHASE,
        "row_kind": "phase896_cross_domain_search_row",
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
        "phase895_known_axis_condition": condition_key(condition) in phase895_keys,
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
        "full_top_blocker_role_counts": metrics.get("full_class_top_blocker_role_counts"),
        "class_minus_object_logit": metrics.get("full_class_minus_object_logit"),
    }


def add_pair_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        groups[key].append(row)
    condition_rows: list[dict[str, Any]] = []
    for key, vals in groups.items():
        singles = [row for row in vals if int(row.get("subset_size") or 0) == 1]
        pairs = [row for row in vals if int(row.get("subset_size") or 0) == 2]
        model_u = next((row for row in vals if str(row.get("subset_relation")) == "model_U" or int(row.get("subset_size") or 0) >= 3), None)
        single_closure = {str(row.get("subset_key")): bool(row.get("closure_from_open")) for row in singles}
        pair_closure_keys = [str(row.get("subset_key")) for row in pairs if row.get("closure_from_open")]
        no_single_pair_keys: list[str] = []
        for row in pairs:
            keys = [str(item) for item in row.get("gear_keys") or []]
            any_component = any(single_closure.get(item, False) for item in keys)
            row["any_component_single_closure"] = bool(any_component)
            row["no_single_pair_closure"] = bool(row.get("closure_from_open") and not any_component)
            if row["no_single_pair_closure"]:
                no_single_pair_keys.append(str(row.get("subset_key")))
        known_minimal_pair_keys = [
            str(row.get("subset_key"))
            for row in pairs
            if row.get("no_single_pair_closure") and not any(key != str(row.get("subset_key")) for key in pair_closure_keys)
        ]
        for row in vals:
            row["condition_any_single_axis_closure"] = any(single_closure.values())
            row["condition_pair_closure_keys"] = sorted(pair_closure_keys)
            row["condition_no_single_pair_keys"] = sorted(no_single_pair_keys)
            row["condition_known_axis_minimal_pair_keys"] = sorted(known_minimal_pair_keys)
            row["condition_model_u_closure"] = bool(model_u and model_u.get("closure_from_open"))
            if int(row.get("subset_size") or 0) == 2:
                row["known_axis_minimal_pair_closure"] = str(row.get("subset_key")) in known_minimal_pair_keys
        focus_key = FOCUS_SUBSET.get(str(key[0]), "")
        focus = next((row for row in vals if str(row.get("subset_key")) == focus_key), None)
        condition_rows.append(
            {
                "phase": PHASE,
                "row_kind": "phase896_condition_summary",
                "model": key[0],
                "case_id": key[1],
                "prompt_variant": key[2],
                "edit_mode": key[3],
                "eval_domain": vals[0].get("eval_domain") if vals else None,
                "object": vals[0].get("object") if vals else None,
                "condition_source": vals[0].get("condition_source") if vals else None,
                "phase895_known_axis_condition": bool(focus and focus.get("phase895_known_axis_condition")),
                "focus_subset_key": focus_key,
                "focus_closure_from_open": bool(focus and focus.get("closure_from_open")),
                "any_single_axis_closure": any(single_closure.values()),
                "pair_closure_keys": sorted(pair_closure_keys),
                "no_single_pair_keys": sorted(no_single_pair_keys),
                "known_axis_minimal_pair_keys": sorted(known_minimal_pair_keys),
                "model_u_closure": bool(model_u and model_u.get("closure_from_open")),
                "focus_target_lift": focus.get("target_lift") if focus else None,
                "focus_blocker_reduction": focus.get("full_blocker_reduction") if focus else None,
            }
        )
    return condition_rows


def select_rollout_sources(
    model_name: str,
    search_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    max_sources: int,
) -> list[dict[str, Any]]:
    by_condition = {
        (str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode"))): row
        for row in condition_rows
    }
    sources = []
    if model_name == "deepseek7b":
        for row in search_rows:
            if int(row.get("subset_size") or 0) != 2:
                continue
            key = (str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
            cond = by_condition.get(key) or {}
            if row.get("known_axis_minimal_pair_closure") or row.get("phase895_known_axis_condition"):
                item = dict(row)
                item["condition_known_axis_minimal_pair_keys"] = cond.get("known_axis_minimal_pair_keys") or []
                sources.append(item)
    else:
        focus_key = FOCUS_SUBSET.get(model_name, "")
        for row in search_rows:
            if str(row.get("subset_key")) == focus_key and row.get("closure_from_open"):
                sources.append(dict(row))
    sources.sort(
        key=lambda row: (
            bool(row.get("phase895_known_axis_condition")),
            bool(row.get("known_axis_minimal_pair_closure")),
            str(row.get("eval_domain")) == "color",
            finite(row.get("target_lift")),
        ),
        reverse=True,
    )
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in sources:
        key = (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")), str(row.get("subset_key")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= int(max_sources):
            break
    return out


def protocol_drift(text: str) -> bool:
    return p895.protocol_drift(text)


def make_rollout_row(
    model_name: str,
    source: dict[str, Any],
    head_set: str,
    text: str,
    ids: list[int],
    rollout: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "row_kind": "phase896_long_rollout_row",
        "model": model_name,
        "case_id": source.get("case_id"),
        "case_split": source.get("case_split"),
        "condition_source": source.get("condition_source"),
        "eval_domain": source.get("eval_domain"),
        "object": source.get("object"),
        "prompt_variant": source.get("prompt_variant"),
        "edit_mode": source.get("edit_mode"),
        "subset_key": source.get("subset_key"),
        "phase895_known_axis_condition": bool(source.get("phase895_known_axis_condition")),
        "known_axis_minimal_pair_closure": bool(source.get("known_axis_minimal_pair_closure")),
        "head_set": head_set,
        "max_new_tokens": len(ids),
        "generated_text": text,
        "generated_clean": rollout.get("generated_clean"),
        "generated_ids": ids,
        "rollout_label": rollout.get("rollout_label"),
        "rollout_class_hit": bool(rollout.get("rollout_answer_class")),
        "rollout_clear_answer_class": bool(rollout.get("rollout_clear_answer_class")),
        "rollout_strict_canonical": bool(rollout.get("rollout_strict_canonical")),
        "rollout_object_echo": bool(rollout.get("rollout_object_echo")),
        "rollout_other_or_format": bool(rollout.get("rollout_other_or_format")),
        "rollout_protocol_drift": protocol_drift(text),
        "answer_like_no_echo": bool(rollout.get("rollout_clear_answer_class") and not protocol_drift(text)),
        "intervention_scope": "first_step_only",
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
            row["rollout_clear_lost_vs_none"] = bool(none.get("rollout_clear_answer_class") and not row.get("rollout_clear_answer_class"))
            row["rollout_echo_added_vs_none"] = bool((not none.get("rollout_object_echo")) and row.get("rollout_object_echo"))
            row["rollout_protocol_drift_added_vs_none"] = bool((not none.get("rollout_protocol_drift")) and row.get("rollout_protocol_drift"))


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_gears = p893.load_model_gears(args.phase893_round, args.model)
    specs = subset_specs(args.model, model_gears, int(args.max_subset_size))
    conditions = selected_conditions(args.model, args)
    phase895_keys = phase895_known_keys(args.model, args.phase895_round)
    if args.dry_run or not model_gears or not specs or not conditions:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run" if model_gears and specs and conditions else "no_gears_or_conditions",
            "model_gear_keys": [p894.gear_key(gear) for gear in model_gears],
            "selected_conditions": len(conditions),
            "phase895_known_keys": len(phase895_keys),
            "subset_specs": [spec.get("subset_key") for spec in specs],
        }
        p846.write_json(out_dir / f"phase896_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase896_{args.model}_search_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase896_{args.model}_condition_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase896_{args.model}_long_rollout_rows.jsonl", [])
        return payload

    model = None
    tokenizer = None
    search_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], list[int]]] = {}
        for idx, condition in enumerate(conditions, 1):
            prompt = p885.prompt_for_case(condition, str(condition.get("prompt_variant")))
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            token_sets = p856.token_sets(tokenizer, condition)
            base_key = (str(condition.get("case_id")), str(condition.get("prompt_variant")))
            if base_key not in base_cache:
                base_logits = p894.first_logits_with_gears_and_heads(
                    model, device, prompt_ids, [], "original", float(args.scale_up_factor), []
                )
                base_cache[base_key] = (
                    p895.metrics_for_logits(tokenizer, base_logits, token_sets, int(args.topk_tokens), int(args.topk_blockers)),
                    token_sets,
                    prompt_ids,
                )
            base_metrics, token_sets, prompt_ids = base_cache[base_key]
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
                metrics = p895.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens), int(args.topk_blockers))
                search_rows.append(make_search_row(args.model, condition, spec, base_metrics, metrics, phase895_keys))
            if idx % int(args.log_every) == 0 or idx == len(conditions):
                log(f"{args.model}/{args.round_name}: search_condition={idx}/{len(conditions)} rows={len(search_rows)}")
        condition_rows = add_pair_fields(search_rows)

        spec_by_key = {str(spec.get("subset_key")): spec for spec in specs}
        condition_by_key = {
            (str(item.get("case_id")), str(item.get("prompt_variant")), str(item.get("edit_mode"))): item for item in conditions
        }
        rollout_sources = select_rollout_sources(args.model, search_rows, condition_rows, int(args.max_rollout_sources_per_model))
        for ridx, source in enumerate(rollout_sources, 1):
            condition = condition_by_key[(str(source.get("case_id")), str(source.get("prompt_variant")), str(source.get("edit_mode")))]
            prompt = p885.prompt_for_case(condition, str(condition.get("prompt_variant")))
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            spec = spec_by_key.get(str(source.get("subset_key")))
            if spec is None:
                continue
            for head_set in HEAD_SETS.get(args.model, ["none"]):
                heads = p894.parse_head_set(head_set)
                text, ids = p894.greedy_with_gears_and_heads(
                    model,
                    tokenizer,
                    device,
                    prompt_ids,
                    spec["gears"],
                    str(condition.get("edit_mode")),
                    float(args.scale_up_factor),
                    heads,
                    int(args.max_new_tokens),
                )
                rollout = p856.classify_rollout(text, condition)
                rollout_rows.append(make_rollout_row(args.model, source, p894.head_set_key(heads), text, ids, rollout))
            log(f"{args.model}/{args.round_name}: rollout_source={ridx}/{len(rollout_sources)} rows={len(rollout_rows)}")
        add_rollout_deltas(rollout_rows)
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, search_rows, condition_rows, rollout_rows, conditions, model_gears, specs, attn_impl)
    p846.write_json(out_dir / f"phase896_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase896_{args.model}_search_rows.jsonl", search_rows)
    p846.write_jsonl(out_dir / f"phase896_{args.model}_condition_rows.jsonl", condition_rows)
    p846.write_jsonl(out_dir / f"phase896_{args.model}_long_rollout_rows.jsonl", rollout_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_model(
    model_name: str,
    search_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    model_gears: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    attn_impl: str | None,
) -> dict[str, Any]:
    focus_key = FOCUS_SUBSET.get(model_name, "")
    no_single_conditions = [row for row in condition_rows if row.get("no_single_pair_keys")]
    known_minimal = [row for row in condition_rows if row.get("known_axis_minimal_pair_keys")]
    cross_domain_known = [row for row in known_minimal if str(row.get("eval_domain")) != "color"]
    phase895_replicated = [
        row
        for row in condition_rows
        if row.get("phase895_known_axis_condition") and focus_key in set(row.get("known_axis_minimal_pair_keys") or [])
    ]
    focus_closure = [
        row for row in search_rows if str(row.get("subset_key")) == focus_key and row.get("closure_from_open")
    ]
    by_domain: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        by_domain[(str(row.get("model")), str(row.get("eval_domain")))].append(row)
    domain_groups = []
    for (model, domain), vals in by_domain.items():
        domain_groups.append(
            {
                "model": model,
                "domain": domain,
                "conditions": len(vals),
                "focus_closure_from_open": sum(1 for row in vals if row.get("focus_closure_from_open")),
                "no_single_pair_conditions": sum(1 for row in vals if row.get("no_single_pair_keys")),
                "known_axis_minimal_pair_conditions": sum(1 for row in vals if row.get("known_axis_minimal_pair_keys")),
                "phase895_replicated": sum(1 for row in vals if row.get("phase895_known_axis_condition")),
                "pair_keys": counter_values(Counter(key for row in vals for key in (row.get("known_axis_minimal_pair_keys") or []))),
            }
        )
    domain_groups.sort(
        key=lambda row: (row.get("known_axis_minimal_pair_conditions") or 0, row.get("no_single_pair_conditions") or 0),
        reverse=True,
    )

    by_pair_domain: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in search_rows:
        if int(row.get("subset_size") or 0) == 2:
            by_pair_domain[(str(row.get("model")), str(row.get("eval_domain")), str(row.get("subset_key")))].append(row)
    pair_domain_groups = []
    for (model, domain, subset), vals in by_pair_domain.items():
        pair_domain_groups.append(
            {
                "model": model,
                "domain": domain,
                "subset_key": subset,
                "rows": len(vals),
                "closure_from_open": sum(1 for row in vals if row.get("closure_from_open")),
                "no_single_pair_closure": sum(1 for row in vals if row.get("no_single_pair_closure")),
                "known_axis_minimal_pair_closure": sum(1 for row in vals if row.get("known_axis_minimal_pair_closure")),
                "mean_target_lift": mean([finite(row.get("target_lift")) for row in vals if row.get("target_lift") is not None]) or 0.0,
                "mean_blocker_reduction": mean(
                    [finite(row.get("full_blocker_reduction")) for row in vals if row.get("full_blocker_reduction") is not None]
                )
                or 0.0,
            }
        )
    pair_domain_groups.sort(
        key=lambda row: (
            row.get("known_axis_minimal_pair_closure") or 0,
            row.get("no_single_pair_closure") or 0,
            row.get("closure_from_open") or 0,
        ),
        reverse=True,
    )

    by_rollout: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        by_rollout[(str(row.get("model")), str(row.get("eval_domain")), str(row.get("subset_key")), str(row.get("head_set")))].append(row)
    rollout_groups = []
    for (model, domain, subset, head_set), vals in by_rollout.items():
        rollout_groups.append(
            {
                "model": model,
                "domain": domain,
                "subset_key": subset,
                "head_set": head_set,
                "rows": len(vals),
                "class_hit": sum(1 for row in vals if row.get("rollout_class_hit")),
                "clear_answer": sum(1 for row in vals if row.get("rollout_clear_answer_class")),
                "strict_canonical": sum(1 for row in vals if row.get("rollout_strict_canonical")),
                "answer_like_no_echo": sum(1 for row in vals if row.get("answer_like_no_echo")),
                "object_echo": sum(1 for row in vals if row.get("rollout_object_echo")),
                "other_or_format": sum(1 for row in vals if row.get("rollout_other_or_format")),
                "protocol_drift": sum(1 for row in vals if row.get("rollout_protocol_drift")),
                "class_lost_vs_none": sum(1 for row in vals if row.get("rollout_class_lost_vs_none")),
                "clear_lost_vs_none": sum(1 for row in vals if row.get("rollout_clear_lost_vs_none")),
                "echo_added_vs_none": sum(1 for row in vals if row.get("rollout_echo_added_vs_none")),
                "drift_added_vs_none": sum(1 for row in vals if row.get("rollout_protocol_drift_added_vs_none")),
            }
        )
    rollout_groups.sort(key=lambda row: (row.get("class_lost_vs_none") or 0, row.get("class_hit") or 0), reverse=True)

    evidence_label = "negative_or_control"
    if model_name == "deepseek7b" and phase895_replicated and cross_domain_known:
        evidence_label = "color_minimal_replicated_with_cross_domain_pair_candidates"
    elif model_name == "deepseek7b" and phase895_replicated:
        evidence_label = "color_known_axis_minimal_replicated_no_cross_domain_extension"
    elif known_minimal:
        evidence_label = "known_axis_minimal_pair_candidates"
    elif focus_closure:
        evidence_label = "focus_closure_without_pair_minimality"

    return {
        "phase": PHASE,
        "title": "Cross-Domain No-Single Pair Search and Long Rollout Stability",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "model_gear_keys": [p894.gear_key(gear) for gear in model_gears],
        "subset_specs": [spec.get("subset_key") for spec in specs],
        "focus_subset_key": focus_key,
        "selected_conditions": len(conditions),
        "output_search_rows": len(search_rows),
        "output_condition_rows": len(condition_rows),
        "output_long_rollout_rows": len(rollout_rows),
        "overall": {
            "focus_closure_from_open": len(focus_closure),
            "no_single_pair_conditions": len(no_single_conditions),
            "known_axis_minimal_pair_conditions": len(known_minimal),
            "cross_domain_known_axis_minimal_pair_conditions": len(cross_domain_known),
            "phase895_known_axis_replicated": len(phase895_replicated),
            "rollout_class_hit": sum(1 for row in rollout_rows if row.get("rollout_class_hit")),
            "rollout_clear_answer": sum(1 for row in rollout_rows if row.get("rollout_clear_answer_class")),
            "rollout_answer_like_no_echo": sum(1 for row in rollout_rows if row.get("answer_like_no_echo")),
            "rollout_object_echo": sum(1 for row in rollout_rows if row.get("rollout_object_echo")),
            "rollout_other_or_format": sum(1 for row in rollout_rows if row.get("rollout_other_or_format")),
            "rollout_protocol_drift": sum(1 for row in rollout_rows if row.get("rollout_protocol_drift")),
            "rollout_class_lost_vs_none": sum(1 for row in rollout_rows if row.get("rollout_class_lost_vs_none")),
            "rollout_clear_lost_vs_none": sum(1 for row in rollout_rows if row.get("rollout_clear_lost_vs_none")),
        },
        "domain_groups": domain_groups,
        "pair_domain_groups": pair_domain_groups,
        "rollout_groups": rollout_groups,
        "evidence_label": evidence_label,
        "boundary": (
            "Phase896 searches known-axis no-single pairs across existing domains and extends rollout length. "
            "Rollout remains first-step intervention rollout, not natural long-horizon closure."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 896 cross-domain no-single pair search and long rollout stability",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
        f"- selected_conditions: {payload.get('selected_conditions')}",
        f"- output_search_rows: {payload.get('output_search_rows')}",
        f"- output_condition_rows: {payload.get('output_condition_rows')}",
        f"- output_long_rollout_rows: {payload.get('output_long_rollout_rows')}",
    ]
    for key, value in (payload.get("overall") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Domain groups",
            "",
            "| model | domain | conditions | focus closure | no-single pair | known minimal pair | phase895 replicated | pair keys |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload.get("domain_groups", [])[:50]:
        lines.append(
            "| {model} | {domain} | {conditions} | {focus} | {no_single} | {known} | {rep} | {pairs} |".format(
                model=row.get("model"),
                domain=row.get("domain"),
                conditions=row.get("conditions"),
                focus=row.get("focus_closure_from_open"),
                no_single=row.get("no_single_pair_conditions"),
                known=row.get("known_axis_minimal_pair_conditions"),
                rep=row.get("phase895_replicated"),
                pairs=json.dumps(row.get("pair_keys") or {}, ensure_ascii=False),
            )
        )
    lines.extend(
        [
            "",
            "## Pair-domain groups",
            "",
            "| model | domain | subset | rows | closure | no-single | known minimal | mean lift | mean blocker reduction |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("pair_domain_groups", [])[:60]:
        lines.append(
            "| {model} | {domain} | {subset} | {rows} | {closure} | {no_single} | {known} | {lift:.3f} | {reduction:.3f} |".format(
                model=row.get("model"),
                domain=row.get("domain"),
                subset=row.get("subset_key"),
                rows=row.get("rows"),
                closure=row.get("closure_from_open"),
                no_single=row.get("no_single_pair_closure"),
                known=row.get("known_axis_minimal_pair_closure"),
                lift=finite(row.get("mean_target_lift")),
                reduction=finite(row.get("mean_blocker_reduction")),
            )
        )
    lines.extend(
        [
            "",
            "## Long rollout groups",
            "",
            "| model | domain | subset | head set | rows | class hit | clear | answer-like | object echo | other/format | drift | class lost | clear lost |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("rollout_groups", [])[:60]:
        lines.append(
            "| {model} | {domain} | {subset} | {head} | {rows} | {class_hit} | {clear} | {answer_like} | {echo} | {other} | {drift} | {class_lost} | {clear_lost} |".format(
                model=row.get("model"),
                domain=row.get("domain"),
                subset=row.get("subset_key"),
                head=row.get("head_set"),
                rows=row.get("rows"),
                class_hit=row.get("class_hit"),
                clear=row.get("clear_answer"),
                answer_like=row.get("answer_like_no_echo"),
                echo=row.get("object_echo"),
                other=row.get("other_or_format"),
                drift=row.get("protocol_drift"),
                class_lost=row.get("class_lost_vs_none"),
                clear_lost=row.get("clear_lost_vs_none"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [
        read_json(out_dir / f"phase896_{model}_summary.json")
        for model in MODELS
        if (out_dir / f"phase896_{model}_summary.json").exists()
    ]
    summaries = [item for item in summaries if item and item.get("status") == "complete"]
    overall = Counter()
    domain_groups: list[dict[str, Any]] = []
    pair_domain_groups: list[dict[str, Any]] = []
    rollout_groups: list[dict[str, Any]] = []
    selected_conditions = output_search_rows = output_condition_rows = output_long_rollout_rows = 0
    for summary in summaries:
        selected_conditions += int(summary.get("selected_conditions") or 0)
        output_search_rows += int(summary.get("output_search_rows") or 0)
        output_condition_rows += int(summary.get("output_condition_rows") or 0)
        output_long_rollout_rows += int(summary.get("output_long_rollout_rows") or 0)
        domain_groups.extend(summary.get("domain_groups") or [])
        pair_domain_groups.extend(summary.get("pair_domain_groups") or [])
        rollout_groups.extend(summary.get("rollout_groups") or [])
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall[key] += value
    domain_groups.sort(key=lambda row: (row.get("known_axis_minimal_pair_conditions") or 0, row.get("no_single_pair_conditions") or 0), reverse=True)
    pair_domain_groups.sort(
        key=lambda row: (
            row.get("known_axis_minimal_pair_closure") or 0,
            row.get("no_single_pair_closure") or 0,
            row.get("closure_from_open") or 0,
        ),
        reverse=True,
    )
    rollout_groups.sort(key=lambda row: (row.get("class_lost_vs_none") or 0, row.get("class_hit") or 0), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "selected_conditions": selected_conditions,
        "output_search_rows": output_search_rows,
        "output_condition_rows": output_condition_rows,
        "output_long_rollout_rows": output_long_rollout_rows,
        "overall": {key: int(value) for key, value in sorted(overall.items())},
        "domain_groups": domain_groups,
        "pair_domain_groups": pair_domain_groups,
        "rollout_groups": rollout_groups,
        "evidence_label_counts": counter_values(Counter(str(summary.get("evidence_label")) for summary in summaries)),
    }
    p846.write_json(out_dir / "phase896_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase896_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="cross_domain_pair_search_long_rollout")
    parser.add_argument("--phase893-round", default=p894.PHASE893_ROUND)
    parser.add_argument("--phase895-round", default=PHASE895_ROUND)
    parser.add_argument("--domains", default="animal,color,material,geometry,tool,abstract,plant,object")
    parser.add_argument("--prompt-variants", default="natural_question,natural_category,question_plain")
    parser.add_argument("--control-prompt-variants", default="natural_question,natural_category")
    parser.add_argument("--edit-modes", default="zero,flip")
    parser.add_argument("--max-cases-per-domain", type=int, default=24)
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--max-rollout-sources-per-model", type=int, default=36)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--topk-tokens", type=int, default=30)
    parser.add_argument("--topk-blockers", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=24)
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
