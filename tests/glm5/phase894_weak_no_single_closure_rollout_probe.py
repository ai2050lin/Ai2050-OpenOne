#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
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


PHASE = 894
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase894_weak_no_single_closure_rollout_probe")
PHASE893_ROOT = Path("tests/result/phase893_attention_head_complementarity_holdout_probe")
PHASE893_ROUND = "attention_head_complementarity_holdout"
EXTRA_COLOR_OBJECTS = [
    "pink",
    "gray",
    "grey",
    "cyan",
    "magenta",
    "teal",
    "navy",
    "maroon",
    "gold",
    "silver",
    "beige",
    "turquoise",
]
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


def gear_key(gear: dict[str, Any]) -> str:
    return p862.gear_key(gear)


def parse_head_key(key: str) -> tuple[int, int] | None:
    text = str(key or "").strip()
    if not text or text == "none":
        return None
    if not (text.startswith("L") and "H" in text):
        return None
    layer_text, head_text = text[1:].split("H", 1)
    try:
        return int(layer_text), int(head_text)
    except ValueError:
        return None


def parse_head_set(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in str(text or "").split("+"):
        parsed = parse_head_key(part)
        if parsed is not None:
            out.append(parsed)
    return out


def head_set_key(heads: list[tuple[int, int]]) -> str:
    return "+".join(f"L{layer}H{head}" for layer, head in heads) if heads else "none"


def install_attention_head_set_zero(model, heads: list[tuple[int, int]]) -> list[Any]:
    handles: list[Any] = []
    for layer_idx, head_idx in heads:
        handles.extend(p893.install_attention_head_zero(model, int(layer_idx), int(head_idx)))
    return handles


def first_logits_with_gears_and_heads(
    model,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    mode: str,
    scale_up_factor: float,
    heads: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles: list[Any] = []
    try:
        if heads:
            handles.extend(install_attention_head_set_zero(model, heads))
        if mode != "original" and gears:
            handles.extend(p862.install_scaled_gear_edit(model, gears, mode, scale_up_factor))
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
    finally:
        for handle in handles:
            handle.remove()


def greedy_with_gears_and_heads(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    mode: str,
    scale_up_factor: float,
    heads: list[tuple[int, int]] | None,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handles: list[Any] = []
        try:
            if step == 0:
                if heads:
                    handles.extend(install_attention_head_set_zero(model, heads))
                if mode != "original" and gears:
                    handles.extend(p862.install_scaled_gear_edit(model, gears, mode, scale_up_factor))
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


def normalize(text: str) -> str:
    return p856.normalize(text)


def text_hits_any(text: str, phrases: list[str]) -> bool:
    norm = normalize(text)
    for phrase in phrases:
        item = normalize(phrase)
        if item and item in norm:
            return True
    return False


def extra_color_cases() -> list[dict[str, Any]]:
    start = 940
    rows = []
    for idx, obj in enumerate(EXTRA_COLOR_OBJECTS, start):
        rows.append(
            {
                "case_id": f"p894_{idx:03d}_color_{obj}",
                "domain": "color",
                "object": obj,
                "answer_aliases": ["color", "hue", "colour"],
                "canonical_answer": "color",
                "overlap_kind": "phase894_extra_color_holdout",
                "split_source": "phase894_extra_color_holdout",
            }
        )
    return rows


def all_cases() -> list[dict[str, Any]]:
    return [dict(case, split_source=case.get("split_source", "phase856_base")) for case in p885.extended_cases()] + extra_color_cases()


def case_map() -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): dict(case) for case in all_cases()}


def phase893_no_single_conditions() -> list[dict[str, Any]]:
    path = PHASE893_ROOT / PHASE893_ROUND / "phase893_deepseek7b_subset_rows.jsonl"
    rows = p846.read_jsonl(path) if path.exists() else []
    out = []
    seen: set[tuple[str, str, str]] = set()
    cases = case_map()
    for row in rows:
        if str(row.get("subset_key")) != "L26C8587+L27C15369":
            continue
        if not row.get("closure_without_single_axis_closure"):
            continue
        key = (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        if key in seen:
            continue
        seen.add(key)
        case = dict(cases.get(str(row.get("case_id"))) or {})
        if not case:
            continue
        case["prompt_variant"] = row.get("prompt_variant")
        case["edit_mode"] = row.get("edit_mode")
        case["condition_source"] = "phase893_exact_no_single"
        case["case_split"] = row.get("case_split")
        out.append(case)
    return out


def selected_conditions(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if model_name == "deepseek7b":
        exact = phase893_no_single_conditions()
        all_color = [case for case in all_cases() if str(case.get("domain")) == "color"]
        prompts = parse_csv(args.color_prompt_variants)
        modes = parse_csv(args.color_edit_modes)
        max_cases = int(args.max_color_cases)
        selected = []
        counts = 0
        for case in all_color:
            if max_cases > 0 and counts >= max_cases:
                break
            counts += 1
            for prompt_variant in prompts:
                for mode in modes:
                    item = dict(case)
                    item["prompt_variant"] = prompt_variant
                    item["edit_mode"] = mode
                    item["condition_source"] = (
                        "phase894_extra_color_holdout"
                        if str(case.get("split_source")) == "phase894_extra_color_holdout"
                        else str(case.get("split_source", "phase856_base"))
                    )
                    item["case_split"] = case.get("split_source", "phase856_base")
                    selected.append(item)
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in selected:
            merged[(str(item.get("case_id")), str(item.get("prompt_variant")), str(item.get("edit_mode")))] = item
        for item in exact:
            merged[(str(item.get("case_id")), str(item.get("prompt_variant")), str(item.get("edit_mode")))] = item
        return list(merged.values())
    if model_name == "qwen3":
        cases = [case for case in all_cases() if str(case.get("domain")) == "material"][: int(args.max_material_cases)]
        out = []
        for case in cases:
            for prompt_variant in parse_csv(args.material_prompt_variants):
                for mode in parse_csv(args.material_edit_modes):
                    item = dict(case)
                    item["prompt_variant"] = prompt_variant
                    item["edit_mode"] = mode
                    item["condition_source"] = str(case.get("split_source", "phase856_base"))
                    item["case_split"] = case.get("split_source", "phase856_base")
                    out.append(item)
        return out
    cases = [case for case in all_cases() if str(case.get("domain")) in {"color", "material"}][: int(args.max_glm4_cases)]
    out = []
    for case in cases:
        for prompt_variant in parse_csv(args.control_prompt_variants):
            for mode in parse_csv(args.control_edit_modes):
                item = dict(case)
                item["prompt_variant"] = prompt_variant
                item["edit_mode"] = mode
                item["condition_source"] = str(case.get("split_source", "phase856_base"))
                item["case_split"] = case.get("split_source", "phase856_base")
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


def make_first_row(
    model_name: str,
    condition: dict[str, Any],
    spec: dict[str, Any],
    base_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "row_kind": "phase894_first_token_row",
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
        "target_lift": target_lift(base_metrics, metrics),
        "base_class_rank": base_metrics.get("class_best_rank"),
        "class_rank": metrics.get("class_best_rank"),
        "base_blocker_count": base_metrics.get("class_blocker_count"),
        "blocker_count": metrics.get("class_blocker_count"),
    }


def add_no_single_fields(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        groups[key].append(row)
    for vals in groups.values():
        single_closure = {
            str(row.get("subset_key")): bool(row.get("closure_from_open"))
            for row in vals
            if int(row.get("subset_size") or 0) == 1
        }
        single_lift = {
            str(row.get("subset_key")): finite(row.get("target_lift"))
            for row in vals
            if int(row.get("subset_size") or 0) == 1 and row.get("target_lift") is not None
        }
        for row in vals:
            keys = [str(key) for key in row.get("gear_keys") or []]
            component_lifts = [single_lift[key] for key in keys if key in single_lift]
            any_single = any(single_closure.get(key, False) for key in keys)
            best_single = max(component_lifts) if component_lifts else None
            lift = finite(row.get("target_lift")) if row.get("target_lift") is not None else None
            row["any_single_axis_closure"] = bool(any_single)
            row["closure_without_single_axis_closure"] = bool(
                int(row.get("subset_size") or 0) > 1 and row.get("closure_from_open") and not any_single
            )
            row["max_single_target_lift"] = best_single
            row["complementarity_over_best_single"] = None if lift is None or best_single is None else lift - best_single


def make_rollout_row(
    model_name: str,
    condition: dict[str, Any],
    spec: dict[str, Any],
    text: str,
    ids: list[int],
    class_hit: bool,
    object_echo: bool,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "row_kind": "phase894_rollout_row",
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
        "generated_text": text,
        "generated_ids": ids,
        "rollout_class_hit": bool(class_hit),
        "rollout_object_echo": bool(object_echo),
    }


def add_rollout_no_single(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        groups[key].append(row)
    for vals in groups.values():
        single_hits = {
            str(row.get("subset_key")): bool(row.get("rollout_class_hit"))
            for row in vals
            if int(row.get("subset_size") or 0) == 1
        }
        for row in vals:
            keys = [str(key) for key in str(row.get("subset_key")).split("+") if key]
            any_single = any(single_hits.get(key, False) for key in keys)
            row["rollout_without_single_axis_hit"] = bool(
                int(row.get("subset_size") or 0) > 1 and row.get("rollout_class_hit") and not any_single
            )


def make_head_row(
    model_name: str,
    source: dict[str, Any],
    head_key_value: str,
    base_metrics: dict[str, Any],
    none_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    none_lift = target_lift(base_metrics, none_metrics)
    lift = target_lift(base_metrics, metrics)
    return {
        "phase": PHASE,
        "row_kind": "phase894_head_combo_row",
        "model": model_name,
        "case_id": source.get("case_id"),
        "case_split": source.get("case_split"),
        "condition_source": source.get("condition_source"),
        "eval_domain": source.get("eval_domain"),
        "object": source.get("object"),
        "prompt_variant": source.get("prompt_variant"),
        "edit_mode": source.get("edit_mode"),
        "subset_key": source.get("subset_key"),
        "head_set": head_key_value,
        "head_count": 0 if head_key_value == "none" else len(head_key_value.split("+")),
        "none_boundary_closed": bool(none_metrics.get("class_boundary_closed")),
        "head_boundary_closed": bool(metrics.get("class_boundary_closed")),
        "none_closure_from_open": bool((not base_metrics.get("class_boundary_closed")) and none_metrics.get("class_boundary_closed")),
        "closure_lost_vs_none": bool(none_metrics.get("class_boundary_closed") and not metrics.get("class_boundary_closed")),
        "none_target_lift": none_lift,
        "head_target_lift": lift,
        "target_lift_damage_vs_none": None if none_lift is None or lift is None else none_lift - lift,
        "none_blocker_count": none_metrics.get("class_blocker_count"),
        "head_blocker_count": metrics.get("class_blocker_count"),
    }


def select_rollout_sources(model_name: str, first_rows: list[dict[str, Any]], max_sources: int) -> list[dict[str, Any]]:
    preferred_subset = "L26C8587+L27C15369" if model_name == "deepseek7b" else "L31C2257"
    rows = [
        row
        for row in first_rows
        if str(row.get("subset_key")) == preferred_subset
        and (row.get("closure_without_single_axis_closure") or row.get("closure_from_open"))
    ]
    rows.sort(
        key=lambda row: (
            bool(row.get("closure_without_single_axis_closure")),
            str(row.get("condition_source")) == "phase893_exact_no_single",
            finite(row.get("complementarity_over_best_single")),
            finite(row.get("target_lift")),
        ),
        reverse=True,
    )
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= int(max_sources):
            break
    return out


def select_head_sources(model_name: str, first_rows: list[dict[str, Any]], max_sources: int) -> list[dict[str, Any]]:
    if model_name == "deepseek7b":
        rows = [
            row
            for row in first_rows
            if str(row.get("subset_key")) == "L26C8587+L27C15369"
            and (row.get("closure_without_single_axis_closure") or row.get("condition_source") == "phase893_exact_no_single")
        ]
    elif model_name == "qwen3":
        rows = [row for row in first_rows if str(row.get("subset_key")) == "L31C2257" and row.get("closure_from_open")]
    else:
        rows = []
    rows.sort(
        key=lambda row: (
            bool(row.get("closure_without_single_axis_closure")),
            bool(row.get("closure_from_open")),
            finite(row.get("target_lift")),
        ),
        reverse=True,
    )
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")), str(row.get("subset_key")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= int(max_sources):
            break
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_gears = p893.load_model_gears(args.phase893_round, args.model)
    specs = subset_specs(args.model, model_gears, int(args.max_subset_size))
    conditions = selected_conditions(args.model, args)
    if args.dry_run or not model_gears or not specs or not conditions:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run" if model_gears and conditions else "no_gears_or_conditions",
            "model_gear_keys": [gear_key(gear) for gear in model_gears],
            "selected_conditions": len(conditions),
            "subset_specs": [spec.get("subset_key") for spec in specs],
        }
        p846.write_json(out_dir / f"phase894_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase894_{args.model}_first_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase894_{args.model}_rollout_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase894_{args.model}_head_rows.jsonl", [])
        return payload

    model = None
    tokenizer = None
    first_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    attn_impl = None
    condition_index = {(str(c.get("case_id")), str(c.get("prompt_variant")), str(c.get("edit_mode"))): c for c in conditions}
    base_cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], list[int]]] = {}
    metrics_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for idx, condition in enumerate(conditions, 1):
            prompt = p885.prompt_for_case(condition, str(condition.get("prompt_variant")))
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            token_sets = p856.token_sets(tokenizer, condition)
            cache_key = (str(condition.get("case_id")), str(condition.get("prompt_variant")))
            if cache_key not in base_cache:
                base_logits = first_logits_with_gears_and_heads(
                    model, device, prompt_ids, [], "original", float(args.scale_up_factor), []
                )
                base_cache[cache_key] = (
                    p888.metrics_for_logits(tokenizer, base_logits, token_sets, int(args.topk_tokens)),
                    token_sets,
                    prompt_ids,
                )
            base_metrics, token_sets, prompt_ids = base_cache[cache_key]
            mode = str(condition.get("edit_mode"))
            for spec in specs:
                logits = first_logits_with_gears_and_heads(
                    model, device, prompt_ids, spec["gears"], mode, float(args.scale_up_factor), []
                )
                metrics = p888.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens))
                row = make_first_row(args.model, condition, spec, base_metrics, metrics)
                first_rows.append(row)
                metrics_cache[(str(condition.get("case_id")), str(condition.get("prompt_variant")), mode, str(spec.get("subset_key")))] = metrics
            log(f"{args.model}/{args.round_name}: first_condition={idx}/{len(conditions)} rows={len(first_rows)}")

        add_no_single_fields(first_rows)

        rollout_sources = select_rollout_sources(args.model, first_rows, int(args.max_rollout_sources_per_model))
        spec_by_key = {str(spec.get("subset_key")): spec for spec in specs}
        rollout_subset_keys = parse_csv(args.rollout_subset_keys)
        for ridx, source in enumerate(rollout_sources, 1):
            condition = condition_index[(str(source.get("case_id")), str(source.get("prompt_variant")), str(source.get("edit_mode")))]
            prompt = p885.prompt_for_case(condition, str(condition.get("prompt_variant")))
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            for subset_key in rollout_subset_keys:
                spec = spec_by_key.get(subset_key)
                if spec is None:
                    continue
                text, ids = greedy_with_gears_and_heads(
                    model,
                    tokenizer,
                    device,
                    prompt_ids,
                    spec["gears"],
                    str(condition.get("edit_mode")),
                    float(args.scale_up_factor),
                    [],
                    int(args.max_new_tokens),
                )
                class_hit = text_hits_any(text, list(condition.get("answer_aliases") or []))
                object_echo = text_hits_any(text, [str(condition.get("object"))])
                rollout_rows.append(make_rollout_row(args.model, condition, spec, text, ids, class_hit, object_echo))
            log(f"{args.model}/{args.round_name}: rollout_source={ridx}/{len(rollout_sources)} rows={len(rollout_rows)}")
        add_rollout_no_single(rollout_rows)

        head_sources = select_head_sources(args.model, first_rows, int(args.max_head_sources_per_model))
        for hidx, source in enumerate(head_sources, 1):
            condition = condition_index[(str(source.get("case_id")), str(source.get("prompt_variant")), str(source.get("edit_mode")))]
            prompt = p885.prompt_for_case(condition, str(condition.get("prompt_variant")))
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            base_metrics, token_sets, prompt_ids = base_cache[(str(condition.get("case_id")), str(condition.get("prompt_variant")))]
            spec = spec_by_key[str(source.get("subset_key"))]
            none_metrics = metrics_cache[
                (str(condition.get("case_id")), str(condition.get("prompt_variant")), str(condition.get("edit_mode")), str(source.get("subset_key")))
            ]
            for head_spec in HEAD_SETS.get(args.model, ["none"]):
                heads = parse_head_set(head_spec)
                logits = first_logits_with_gears_and_heads(
                    model,
                    device,
                    prompt_ids,
                    spec["gears"],
                    str(condition.get("edit_mode")),
                    float(args.scale_up_factor),
                    heads,
                )
                metrics = p888.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens))
                head_rows.append(make_head_row(args.model, source, head_set_key(heads), base_metrics, none_metrics, metrics))
            log(f"{args.model}/{args.round_name}: head_source={hidx}/{len(head_sources)} rows={len(head_rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, first_rows, rollout_rows, head_rows, conditions, model_gears, specs, attn_impl)
    p846.write_json(out_dir / f"phase894_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase894_{args.model}_first_rows.jsonl", first_rows)
    p846.write_jsonl(out_dir / f"phase894_{args.model}_rollout_rows.jsonl", rollout_rows)
    p846.write_jsonl(out_dir / f"phase894_{args.model}_head_rows.jsonl", head_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_model(
    model_name: str,
    first_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    model_gears: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    attn_impl: str | None,
) -> dict[str, Any]:
    closures = [row for row in first_rows if row.get("closure_from_open")]
    no_single = [row for row in first_rows if row.get("closure_without_single_axis_closure")]
    exact_conditions = {
        (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        for row in first_rows
        if row.get("condition_source") == "phase893_exact_no_single"
    }
    exact_replicated = {
        (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        for row in no_single
        if row.get("condition_source") == "phase893_exact_no_single" and str(row.get("subset_key")) == "L26C8587+L27C15369"
    }
    expanded_no_single = [
        row
        for row in no_single
        if row.get("condition_source") != "phase893_exact_no_single" and str(row.get("subset_key")) == "L26C8587+L27C15369"
    ]

    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in first_rows:
        by_subset[str(row.get("subset_key"))].append(row)
    subset_groups = []
    for key, vals in by_subset.items():
        subset_groups.append(
            {
                "model": model_name,
                "subset_key": key,
                "subset_relation": vals[0].get("subset_relation") if vals else None,
                "n_rows": len(vals),
                "closure_from_open": sum(1 for row in vals if row.get("closure_from_open")),
                "closure_without_single_axis_closure": sum(1 for row in vals if row.get("closure_without_single_axis_closure")),
                "exact_no_single_replicated": sum(
                    1 for row in vals if row.get("condition_source") == "phase893_exact_no_single" and row.get("closure_without_single_axis_closure")
                ),
                "expanded_no_single": sum(
                    1 for row in vals if row.get("condition_source") != "phase893_exact_no_single" and row.get("closure_without_single_axis_closure")
                ),
                "mean_target_lift_on_closure": mean([finite(row.get("target_lift")) for row in vals if row.get("closure_from_open") and row.get("target_lift") is not None]) or 0.0,
                "mean_complementarity_over_best": mean(
                    [
                        finite(row.get("complementarity_over_best_single"))
                        for row in vals
                        if row.get("complementarity_over_best_single") is not None
                    ]
                )
                or 0.0,
                "modes": counter_values(Counter(str(row.get("edit_mode")) for row in vals if row.get("closure_from_open"))),
                "objects_no_single": sorted(set(str(row.get("object")) for row in vals if row.get("closure_without_single_axis_closure"))),
            }
        )
    subset_groups.sort(
        key=lambda row: (
            row.get("closure_without_single_axis_closure") or 0,
            row.get("closure_from_open") or 0,
        ),
        reverse=True,
    )

    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        by_rollout[str(row.get("subset_key"))].append(row)
    rollout_groups = []
    for key, vals in by_rollout.items():
        rollout_groups.append(
            {
                "model": model_name,
                "subset_key": key,
                "n_rows": len(vals),
                "rollout_class_hit": sum(1 for row in vals if row.get("rollout_class_hit")),
                "rollout_without_single_axis_hit": sum(1 for row in vals if row.get("rollout_without_single_axis_hit")),
                "object_echo": sum(1 for row in vals if row.get("rollout_object_echo")),
            }
        )
    rollout_groups.sort(key=lambda row: (row.get("rollout_without_single_axis_hit") or 0, row.get("rollout_class_hit") or 0), reverse=True)

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
                "closure_lost_vs_none": sum(1 for row in vals if row.get("closure_lost_vs_none")),
                "damage_gt_0_25": sum(
                    1
                    for row in vals
                    if row.get("target_lift_damage_vs_none") is not None and finite(row.get("target_lift_damage_vs_none")) > 0.25
                ),
                "mean_damage": mean(
                    [finite(row.get("target_lift_damage_vs_none")) for row in vals if row.get("target_lift_damage_vs_none") is not None]
                )
                or 0.0,
                "max_damage": max(
                    [finite(row.get("target_lift_damage_vs_none")) for row in vals if row.get("target_lift_damage_vs_none") is not None]
                    or [0.0]
                ),
            }
        )
    head_groups.sort(key=lambda row: (row.get("closure_lost_vs_none") or 0, row.get("mean_damage") or 0.0), reverse=True)

    evidence_label = "negative_under_phase894_candidates"
    if model_name == "deepseek7b" and exact_replicated and expanded_no_single:
        evidence_label = "replicated_and_expanded_weak_no_single_closure"
    elif model_name == "deepseek7b" and exact_replicated:
        evidence_label = "replicated_phase893_weak_no_single_closure"
    elif closures:
        evidence_label = "first_token_closure_without_no_single_extension"

    return {
        "phase": PHASE,
        "title": "Weak No-Single Closure Replication and Rollout Boundary Probe",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "model_gear_keys": [gear_key(gear) for gear in model_gears],
        "subset_specs": [spec.get("subset_key") for spec in specs],
        "selected_conditions": len(conditions),
        "output_first_rows": len(first_rows),
        "output_rollout_rows": len(rollout_rows),
        "output_head_rows": len(head_rows),
        "overall": {
            "closure_from_open": len(closures),
            "closure_without_single_axis_closure": len(no_single),
            "unique_no_single_conditions": len(
                set((str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode"))) for row in no_single)
            ),
            "phase893_exact_conditions": len(exact_conditions),
            "phase893_exact_no_single_replicated": len(exact_replicated),
            "expanded_pair_no_single_conditions": len(
                set((str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode"))) for row in expanded_no_single)
            ),
            "rollout_class_hit": sum(1 for row in rollout_rows if row.get("rollout_class_hit")),
            "rollout_without_single_axis_hit": sum(1 for row in rollout_rows if row.get("rollout_without_single_axis_hit")),
            "head_combo_closure_lost": sum(1 for row in head_rows if row.get("closure_lost_vs_none")),
            "head_combo_damage_gt_0_25": sum(
                1
                for row in head_rows
                if row.get("target_lift_damage_vs_none") is not None and finite(row.get("target_lift_damage_vs_none")) > 0.25
            ),
        },
        "subset_groups": subset_groups,
        "rollout_groups": rollout_groups,
        "head_groups": head_groups,
        "evidence_label": evidence_label,
        "condition_source_counts": counter_values(Counter(str(row.get("condition_source")) for row in conditions)),
        "boundary": (
            "Phase894 replicates weak no-single closure candidates and checks short rollout. "
            "Short rollout is still first-step intervention rollout, not full long-horizon generation closure."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 894 weak no-single closure replication and rollout boundary probe",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
        f"- selected_conditions: {payload.get('selected_conditions')}",
        f"- output_first_rows: {payload.get('output_first_rows')}",
        f"- output_rollout_rows: {payload.get('output_rollout_rows')}",
        f"- output_head_rows: {payload.get('output_head_rows')}",
        f"- closure_from_open: {payload.get('overall', {}).get('closure_from_open')}",
        f"- closure_without_single_axis_closure: {payload.get('overall', {}).get('closure_without_single_axis_closure')}",
        f"- phase893_exact_no_single_replicated: {payload.get('overall', {}).get('phase893_exact_no_single_replicated')}",
        f"- expanded_pair_no_single_conditions: {payload.get('overall', {}).get('expanded_pair_no_single_conditions')}",
        f"- rollout_class_hit: {payload.get('overall', {}).get('rollout_class_hit')}",
        f"- rollout_without_single_axis_hit: {payload.get('overall', {}).get('rollout_without_single_axis_hit')}",
        f"- head_combo_closure_lost: {payload.get('overall', {}).get('head_combo_closure_lost')}",
        f"- head_combo_damage_gt_0_25: {payload.get('overall', {}).get('head_combo_damage_gt_0_25')}",
        "",
        "## Subset groups",
        "",
        "| model | subset | closure | no-single | exact replicated | expanded no-single | mean lift | mean comp | no-single objects |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("subset_groups", [])[:30]:
        lines.append(
            "| {model} | {subset} | {closure} | {no_single} | {exact} | {expanded} | {lift:.3f} | {comp:.3f} | {objects} |".format(
                model=row.get("model"),
                subset=row.get("subset_key"),
                closure=row.get("closure_from_open"),
                no_single=row.get("closure_without_single_axis_closure"),
                exact=row.get("exact_no_single_replicated"),
                expanded=row.get("expanded_no_single"),
                lift=finite(row.get("mean_target_lift_on_closure")),
                comp=finite(row.get("mean_complementarity_over_best")),
                objects=",".join(row.get("objects_no_single") or []),
            )
        )
    lines.extend(
        [
            "",
            "## Rollout groups",
            "",
            "| model | subset | rows | class hit | no-single hit | object echo |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("rollout_groups", [])[:30]:
        lines.append(
            f"| {row.get('model')} | {row.get('subset_key')} | {row.get('n_rows')} | {row.get('rollout_class_hit')} | {row.get('rollout_without_single_axis_hit')} | {row.get('object_echo')} |"
        )
    lines.extend(
        [
            "",
            "## Head groups",
            "",
            "| model | head set | rows | closure lost | damage > 0.25 | mean damage | max damage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("head_groups", [])[:30]:
        lines.append(
            "| {model} | {head_set} | {rows} | {lost} | {damage_rows} | {mean_damage:.3f} | {max_damage:.3f} |".format(
                model=row.get("model"),
                head_set=row.get("head_set"),
                rows=row.get("n_rows"),
                lost=row.get("closure_lost_vs_none"),
                damage_rows=row.get("damage_gt_0_25"),
                mean_damage=finite(row.get("mean_damage")),
                max_damage=finite(row.get("max_damage")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [
        read_json(out_dir / f"phase894_{model}_summary.json")
        for model in MODELS
        if (out_dir / f"phase894_{model}_summary.json").exists()
    ]
    summaries = [item for item in summaries if item and item.get("status") == "complete"]
    overall = Counter()
    subset_groups: list[dict[str, Any]] = []
    rollout_groups: list[dict[str, Any]] = []
    head_groups: list[dict[str, Any]] = []
    selected_conditions = output_first_rows = output_rollout_rows = output_head_rows = 0
    for summary in summaries:
        selected_conditions += int(summary.get("selected_conditions") or 0)
        output_first_rows += int(summary.get("output_first_rows") or 0)
        output_rollout_rows += int(summary.get("output_rollout_rows") or 0)
        output_head_rows += int(summary.get("output_head_rows") or 0)
        subset_groups.extend(summary.get("subset_groups") or [])
        rollout_groups.extend(summary.get("rollout_groups") or [])
        head_groups.extend(summary.get("head_groups") or [])
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall[key] += value
    subset_groups.sort(
        key=lambda row: (
            row.get("closure_without_single_axis_closure") or 0,
            row.get("closure_from_open") or 0,
        ),
        reverse=True,
    )
    rollout_groups.sort(key=lambda row: (row.get("rollout_without_single_axis_hit") or 0, row.get("rollout_class_hit") or 0), reverse=True)
    head_groups.sort(key=lambda row: (row.get("closure_lost_vs_none") or 0, row.get("mean_damage") or 0.0), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "selected_conditions": selected_conditions,
        "output_first_rows": output_first_rows,
        "output_rollout_rows": output_rollout_rows,
        "output_head_rows": output_head_rows,
        "overall": {key: int(value) for key, value in sorted(overall.items())},
        "subset_groups": subset_groups,
        "rollout_groups": rollout_groups,
        "head_groups": head_groups,
        "evidence_label_counts": counter_values(Counter(str(summary.get("evidence_label")) for summary in summaries)),
    }
    p846.write_json(out_dir / "phase894_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase894_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="weak_no_single_closure_rollout")
    parser.add_argument("--phase893-round", default=PHASE893_ROUND)
    parser.add_argument("--color-prompt-variants", default="natural_question,natural_category,question_plain")
    parser.add_argument("--color-edit-modes", default="zero,flip")
    parser.add_argument("--material-prompt-variants", default="natural_question,natural_category,classification")
    parser.add_argument("--material-edit-modes", default="zero,flip,half")
    parser.add_argument("--control-prompt-variants", default="natural_question,natural_category,classification")
    parser.add_argument("--control-edit-modes", default="zero,flip")
    parser.add_argument("--rollout-subset-keys", default="L26C8587,L27C15369,L26C8587+L27C15369,L26C8587+L27C15369+L27C16651,L31C2257,L31C6437")
    parser.add_argument("--max-color-cases", type=int, default=24)
    parser.add_argument("--max-material-cases", type=int, default=12)
    parser.add_argument("--max-glm4-cases", type=int, default=24)
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--max-rollout-sources-per-model", type=int, default=28)
    parser.add_argument("--max-head-sources-per-model", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--topk-tokens", type=int, default=20)
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
