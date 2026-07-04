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
import phase892_channel_complementarity_coordinate_basis_probe as p892  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 893
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE892_ROOT = Path("tests/result/phase892_channel_complementarity_coordinate_basis_probe")
RESULT_ROOT = Path("tests/result/phase893_attention_head_complementarity_holdout_probe")
DEFAULT_GEAR_KEYS = {
    "qwen3": ["L31C2257"],
    "glm4": ["L31C6437"],
    "deepseek7b": ["L26C8587", "L27C15369", "L27C16651"],
}
MODEL_DOMAINS = {
    "qwen3": ["material"],
    "glm4": ["color", "material"],
    "deepseek7b": ["color", "animal"],
}
PREFERRED_HEAD_SUBSETS = {
    "qwen3": ["L31C2257"],
    "glm4": ["L31C6437"],
    "deepseek7b": ["L26C8587+L27C15369", "L27C16651"],
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def counter_values(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def gear_tuple(gear: dict[str, Any]) -> tuple[int, int]:
    return (int(gear["layer_idx"]), int(gear["channel_id"]))


def gear_key_from_tuple(item: tuple[int, int]) -> str:
    return f"L{int(item[0])}C{int(item[1])}"


def dedupe_gears(gears: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    for gear in sorted(gears, key=gear_tuple):
        key = gear_tuple(gear)
        if key in seen:
            continue
        seen.add(key)
        out.append({"layer_idx": key[0], "channel_id": key[1]})
    return out


def load_model_gears(round_name: str, model: str) -> list[dict[str, Any]]:
    summary = read_json(PHASE892_ROOT / round_name / f"phase892_{model}_summary.json")
    keys = summary.get("model_gear_keys") or DEFAULT_GEAR_KEYS.get(model, [])
    gears = []
    for key in keys:
        gear = p862.parse_gear_key(str(key))
        if gear is not None:
            gears.append(gear)
    return dedupe_gears(gears)


def selected_cases(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    domains = parse_csv(args.domains) or MODEL_DOMAINS.get(model_name, [])
    prompts = parse_csv(args.prompt_variants)
    max_per_domain = int(args.max_cases_per_domain)
    rows = [dict(case) for case in p885.extended_cases() if str(case.get("domain")) in set(domains)]
    rows.sort(key=lambda case: (str(case.get("domain")), str(case.get("split_source", "phase856_base")), str(case.get("object"))))
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for case in rows:
        domain = str(case.get("domain"))
        if max_per_domain > 0 and counts[domain] >= max_per_domain:
            continue
        counts[domain] += 1
        for prompt_variant in prompts:
            item = dict(case)
            item["prompt_variant"] = prompt_variant
            item["case_split"] = case.get("split_source", "phase856_base")
            out.append(item)
    return out


def subset_specs(model_name: str, gears: list[dict[str, Any]], max_subset_size: int) -> list[dict[str, Any]]:
    specs = []
    cleaned = dedupe_gears(gears)
    max_size = min(int(max_subset_size), len(cleaned))
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(cleaned, size):
            tuples = tuple(gear_tuple(gear) for gear in combo)
            keys = [gear_key_from_tuple(item) for item in tuples]
            subset_key = "+".join(keys)
            relation = "single_axis" if size == 1 else "multi_axis"
            if model_name == "deepseek7b" and subset_key == "L26C8587+L27C15369":
                relation = "ds7b_color_complementary_pair"
            elif model_name == "deepseek7b" and subset_key == "L27C16651":
                relation = "ds7b_animal_single_axis"
            elif size == len(cleaned):
                relation = "model_U"
            specs.append(
                {
                    "subset_key": subset_key,
                    "subset_size": size,
                    "subset_relation": relation,
                    "gear_keys": keys,
                    "gears": [{"layer_idx": item[0], "channel_id": item[1]} for item in tuples],
                }
            )
    return specs


def install_attention_head_zero(model, layer_idx: int, head_idx: int) -> list[Any]:
    layers = get_layers(model)
    if not (0 <= int(layer_idx) < len(layers)):
        return []
    attn = getattr(layers[int(layer_idx)], "self_attn", None)
    if attn is None:
        return []
    out_proj = None
    for name in ("o_proj", "dense", "out_proj"):
        if hasattr(attn, name):
            out_proj = getattr(attn, name)
            break
    if out_proj is None:
        return []
    n_heads = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
    if n_heads is None:
        n_heads = getattr(getattr(model, "config", object()), "num_attention_heads", None)
    if n_heads is None:
        return []
    n_heads = int(n_heads)
    if not (0 <= int(head_idx) < n_heads):
        return []

    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        tensor = inputs[0]
        if tensor.shape[-1] < n_heads:
            return inputs
        head_dim = int(tensor.shape[-1]) // n_heads
        start = int(head_idx) * head_dim
        end = start + head_dim
        if start >= int(tensor.shape[-1]):
            return inputs
        patched = tensor.clone()
        end = min(end, int(patched.shape[-1]))
        if patched.ndim >= 3:
            patched[:, -1, start:end] = 0
        elif patched.ndim >= 2:
            patched[:, start:end] = 0
        return (patched, *inputs[1:])

    return [out_proj.register_forward_pre_hook(hook)]


def attention_head_count(model, layer_idx: int) -> int:
    layers = get_layers(model)
    if not (0 <= int(layer_idx) < len(layers)):
        return 0
    attn = getattr(layers[int(layer_idx)], "self_attn", None)
    if attn is None:
        return 0
    n_heads = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
    if n_heads is None:
        n_heads = getattr(getattr(model, "config", object()), "num_attention_heads", 0)
    return int(n_heads or 0)


def first_logits_with_gears_and_head_zero(
    model,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    mode: str,
    scale_up_factor: float,
    head_layer: int | None = None,
    head_idx: int | None = None,
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles: list[Any] = []
    try:
        if head_layer is not None and head_idx is not None:
            handles.extend(install_attention_head_zero(model, int(head_layer), int(head_idx)))
        if mode != "original" and gears:
            handles.extend(p862.install_scaled_gear_edit(model, gears, mode, scale_up_factor))
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
    finally:
        for handle in handles:
            handle.remove()


def metric_delta(base_metrics: dict[str, Any], metrics: dict[str, Any]) -> tuple[float | None, float | None]:
    base_class = base_metrics.get("class_best_logit")
    current_class = metrics.get("class_best_logit")
    target_lift = None if base_class is None or current_class is None else finite(current_class) - finite(base_class)
    blocker_reduction = None
    if base_metrics.get("class_blocker_count") is not None and metrics.get("class_blocker_count") is not None:
        blocker_reduction = finite(base_metrics.get("class_blocker_count")) - finite(metrics.get("class_blocker_count"))
    return target_lift, blocker_reduction


def make_subset_row(
    model_name: str,
    case: dict[str, Any],
    spec: dict[str, Any],
    mode: str,
    base_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    target_lift, blocker_reduction = metric_delta(base_metrics, metrics)
    return {
        "phase": PHASE,
        "row_kind": "phase893_subset_holdout_row",
        "model": model_name,
        "case_id": case.get("case_id"),
        "case_split": case.get("case_split"),
        "is_holdout_case": bool(str(case.get("case_split")) == "phase885_extended_holdout"),
        "eval_domain": case.get("domain"),
        "object": case.get("object"),
        "prompt_variant": case.get("prompt_variant"),
        "edit_mode": mode,
        "subset_key": spec.get("subset_key"),
        "subset_size": spec.get("subset_size"),
        "subset_relation": spec.get("subset_relation"),
        "gear_keys": spec.get("gear_keys"),
        "base_boundary_closed": bool(base_metrics.get("class_boundary_closed")),
        "boundary_closed": bool(metrics.get("class_boundary_closed")),
        "closure_from_open": bool((not base_metrics.get("class_boundary_closed")) and metrics.get("class_boundary_closed")),
        "base_class_logit": base_metrics.get("class_best_logit"),
        "class_logit": metrics.get("class_best_logit"),
        "target_lift": target_lift,
        "base_blocker_count": base_metrics.get("class_blocker_count"),
        "blocker_count": metrics.get("class_blocker_count"),
        "blocker_reduction": blocker_reduction,
    }


def add_subset_complementarity(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        groups[key].append(row)
    for vals in groups.values():
        singles = {
            str(row.get("subset_key")): finite(row.get("target_lift"))
            for row in vals
            if int(row.get("subset_size") or 0) == 1 and row.get("target_lift") is not None
        }
        single_closures = {
            str(row.get("subset_key")): bool(row.get("closure_from_open"))
            for row in vals
            if int(row.get("subset_size") or 0) == 1
        }
        for row in vals:
            keys = [str(key) for key in row.get("gear_keys") or []]
            component_lifts = [singles[key] for key in keys if key in singles]
            target_lift = finite(row.get("target_lift")) if row.get("target_lift") is not None else None
            additive = sum(component_lifts) if component_lifts else None
            best_single = max(component_lifts) if component_lifts else None
            any_single_closure = any(single_closures.get(key, False) for key in keys)
            row["additive_expected_target_lift"] = additive
            row["max_single_target_lift"] = best_single
            row["complementarity_over_best_single"] = None if target_lift is None or best_single is None else target_lift - best_single
            row["interaction_residual_vs_additive"] = None if target_lift is None or additive is None else target_lift - additive
            row["any_single_axis_closure"] = bool(any_single_closure)
            row["closure_without_single_axis_closure"] = bool(
                int(row.get("subset_size") or 0) > 1 and row.get("closure_from_open") and not any_single_closure
            )


def select_head_sources(model_name: str, subset_rows: list[dict[str, Any]], max_sources: int) -> list[dict[str, Any]]:
    preferred = set(PREFERRED_HEAD_SUBSETS.get(model_name, []))
    rows = [
        row
        for row in subset_rows
        if row.get("target_lift") is not None
        and (not preferred or str(row.get("subset_key")) in preferred)
        and finite(row.get("target_lift")) > 0
    ]
    rows.sort(
        key=lambda row: (
            bool(row.get("closure_from_open")),
            finite(row.get("complementarity_over_best_single")),
            finite(row.get("target_lift")),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("subset_key")), str(row.get("edit_mode")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= int(max_sources):
            break
    return selected


def make_head_row(
    model_name: str,
    source: dict[str, Any],
    head_layer: int,
    head_idx: int,
    base_metrics: dict[str, Any],
    none_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    none_lift, _none_blocker_reduction = metric_delta(base_metrics, none_metrics)
    lift, blocker_reduction = metric_delta(base_metrics, metrics)
    return {
        "phase": PHASE,
        "row_kind": "phase893_attention_head_attribution_row",
        "model": model_name,
        "case_id": source.get("case_id"),
        "case_split": source.get("case_split"),
        "is_holdout_case": bool(source.get("is_holdout_case")),
        "eval_domain": source.get("eval_domain"),
        "object": source.get("object"),
        "prompt_variant": source.get("prompt_variant"),
        "edit_mode": source.get("edit_mode"),
        "subset_key": source.get("subset_key"),
        "subset_relation": source.get("subset_relation"),
        "head_layer": int(head_layer),
        "head_idx": int(head_idx),
        "head_key": f"L{int(head_layer)}H{int(head_idx)}",
        "base_boundary_closed": bool(base_metrics.get("class_boundary_closed")),
        "none_boundary_closed": bool(none_metrics.get("class_boundary_closed")),
        "head_zero_boundary_closed": bool(metrics.get("class_boundary_closed")),
        "none_closure_from_open": bool((not base_metrics.get("class_boundary_closed")) and none_metrics.get("class_boundary_closed")),
        "closure_lost_vs_none": bool(none_metrics.get("class_boundary_closed") and not metrics.get("class_boundary_closed")),
        "none_target_lift": none_lift,
        "head_zero_target_lift": lift,
        "target_lift_damage_vs_none": None if none_lift is None or lift is None else none_lift - lift,
        "head_zero_blocker_reduction": blocker_reduction,
        "none_blocker_count": none_metrics.get("class_blocker_count"),
        "head_zero_blocker_count": metrics.get("class_blocker_count"),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_gears = load_model_gears(args.phase892_round, args.model)
    cases = selected_cases(args.model, args)
    modes = parse_csv(args.edit_modes)
    specs = subset_specs(args.model, model_gears, int(args.max_subset_size))
    if args.dry_run or not model_gears or not cases:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run" if model_gears and cases else "no_gears_or_cases",
            "model_gear_keys": [p862.gear_key(gear) for gear in model_gears],
            "selected_cases": len(cases),
            "edit_modes": modes,
        }
        p846.write_json(out_dir / f"phase893_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase893_{args.model}_subset_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase893_{args.model}_head_rows.jsonl", [])
        return payload

    model = None
    tokenizer = None
    subset_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    attn_impl = None
    base_cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], list[int]]] = {}
    none_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for idx, case in enumerate(cases, 1):
            prompt = p885.prompt_for_case(case, str(case.get("prompt_variant")))
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            token_sets = p856.token_sets(tokenizer, case)
            cache_key = (str(case.get("case_id")), str(case.get("prompt_variant")))
            if cache_key not in base_cache:
                base_logits = first_logits_with_gears_and_head_zero(
                    model, device, prompt_ids, [], "original", float(args.scale_up_factor)
                )
                base_cache[cache_key] = (
                    p888.metrics_for_logits(tokenizer, base_logits, token_sets, int(args.topk_tokens)),
                    token_sets,
                    prompt_ids,
                )
            base_metrics, token_sets, prompt_ids = base_cache[cache_key]
            for spec in specs:
                for mode in modes:
                    logits = first_logits_with_gears_and_head_zero(
                        model, device, prompt_ids, spec["gears"], mode, float(args.scale_up_factor)
                    )
                    metrics = p888.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens))
                    row = make_subset_row(args.model, case, spec, mode, base_metrics, metrics)
                    subset_rows.append(row)
                    none_cache[(str(case.get("case_id")), str(case.get("prompt_variant")), str(spec.get("subset_key")), mode)] = metrics
            log(f"{args.model}/{args.round_name}: subset_case={idx}/{len(cases)} rows={len(subset_rows)}")

        add_subset_complementarity(subset_rows)
        head_sources = select_head_sources(args.model, subset_rows, int(args.max_head_sources_per_model))
        head_layers = sorted({int(gear["layer_idx"]) for gear in model_gears})
        for sidx, source in enumerate(head_sources, 1):
            case = next((item for item in cases if str(item.get("case_id")) == str(source.get("case_id")) and str(item.get("prompt_variant")) == str(source.get("prompt_variant"))), None)
            if case is None:
                continue
            prompt = p885.prompt_for_case(case, str(case.get("prompt_variant")))
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            cache_key = (str(case.get("case_id")), str(case.get("prompt_variant")))
            base_metrics, token_sets, prompt_ids = base_cache[cache_key]
            spec = next((item for item in specs if str(item.get("subset_key")) == str(source.get("subset_key"))), None)
            if spec is None:
                continue
            none_metrics = none_cache[(str(source.get("case_id")), str(source.get("prompt_variant")), str(source.get("subset_key")), str(source.get("edit_mode")))]
            for layer_idx in head_layers:
                n_heads = min(attention_head_count(model, layer_idx), int(args.max_heads_per_layer))
                for head_idx in range(n_heads):
                    logits = first_logits_with_gears_and_head_zero(
                        model,
                        device,
                        prompt_ids,
                        spec["gears"],
                        str(source.get("edit_mode")),
                        float(args.scale_up_factor),
                        layer_idx,
                        head_idx,
                    )
                    metrics = p888.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens))
                    head_rows.append(make_head_row(args.model, source, layer_idx, head_idx, base_metrics, none_metrics, metrics))
            log(f"{args.model}/{args.round_name}: head_source={sidx}/{len(head_sources)} head_rows={len(head_rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, subset_rows, head_rows, cases, model_gears, modes, attn_impl)
    p846.write_json(out_dir / f"phase893_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase893_{args.model}_subset_rows.jsonl", subset_rows)
    p846.write_jsonl(out_dir / f"phase893_{args.model}_head_rows.jsonl", head_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def evidence_label(model_name: str, subset_rows: list[dict[str, Any]], head_rows: list[dict[str, Any]]) -> str:
    closures = [row for row in subset_rows if row.get("closure_from_open")]
    positive_comp = [
        row
        for row in closures
        if int(row.get("subset_size") or 0) > 1
        and row.get("complementarity_over_best_single") is not None
        and finite(row.get("complementarity_over_best_single")) > 0.25
    ]
    holdout_comp = [row for row in positive_comp if row.get("is_holdout_case")]
    head_damage = [
        row
        for row in head_rows
        if row.get("target_lift_damage_vs_none") is not None and finite(row.get("target_lift_damage_vs_none")) > 0.25
    ]
    if model_name == "deepseek7b" and holdout_comp and head_damage:
        return "holdout_pairwise_complementarity_with_attention_head_damage_candidates"
    if holdout_comp:
        return "holdout_pairwise_target_lift_complementarity"
    if positive_comp:
        return "in_sample_pairwise_target_lift_complementarity"
    if closures:
        return "single_axis_or_noncomplementary_target_lift"
    return "negative_under_phase893_candidates"


def summarize_model(
    model_name: str,
    subset_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    model_gears: list[dict[str, Any]],
    modes: list[str],
    attn_impl: str | None,
) -> dict[str, Any]:
    closures = [row for row in subset_rows if row.get("closure_from_open")]
    multi = [row for row in closures if int(row.get("subset_size") or 0) > 1]
    positive_comp = [
        row
        for row in multi
        if row.get("complementarity_over_best_single") is not None and finite(row.get("complementarity_over_best_single")) > 0.25
    ]
    holdout_positive_comp = [row for row in positive_comp if row.get("is_holdout_case")]
    closure_without_single = [row for row in multi if row.get("closure_without_single_axis_closure")]
    damage_rows = [
        row
        for row in head_rows
        if row.get("target_lift_damage_vs_none") is not None and finite(row.get("target_lift_damage_vs_none")) > 0.25
    ]
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset_rows:
        by_subset[str(row.get("subset_key"))].append(row)
    subset_groups = []
    for key, vals in by_subset.items():
        vals_closure = [row for row in vals if row.get("closure_from_open")]
        vals_multi = [row for row in vals_closure if int(row.get("subset_size") or 0) > 1]
        vals_comp = [
            row
            for row in vals_multi
            if row.get("complementarity_over_best_single") is not None and finite(row.get("complementarity_over_best_single")) > 0.25
        ]
        subset_groups.append(
            {
                "model": model_name,
                "subset_key": key,
                "subset_relation": vals[0].get("subset_relation") if vals else None,
                "subset_size": vals[0].get("subset_size") if vals else None,
                "n_rows": len(vals),
                "closure_from_open": len(vals_closure),
                "holdout_closure_from_open": sum(1 for row in vals_closure if row.get("is_holdout_case")),
                "positive_complementarity_rows": len(vals_comp),
                "holdout_positive_complementarity_rows": sum(1 for row in vals_comp if row.get("is_holdout_case")),
                "closure_without_single_axis_closure": sum(
                    1 for row in vals_multi if row.get("closure_without_single_axis_closure")
                ),
                "mean_target_lift_on_closure": mean([finite(row.get("target_lift")) for row in vals_closure if row.get("target_lift") is not None]) or 0.0,
                "mean_complementarity_over_best": mean(
                    [
                        finite(row.get("complementarity_over_best_single"))
                        for row in vals_multi
                        if row.get("complementarity_over_best_single") is not None
                    ]
                )
                or 0.0,
                "objects": sorted(set(str(row.get("object")) for row in vals)),
                "domains": sorted(set(str(row.get("eval_domain")) for row in vals)),
                "modes": counter_values(Counter(str(row.get("edit_mode")) for row in vals_closure)),
            }
        )
    subset_groups.sort(
        key=lambda row: (
            row.get("holdout_positive_complementarity_rows") or 0,
            row.get("positive_complementarity_rows") or 0,
            row.get("closure_from_open") or 0,
        ),
        reverse=True,
    )

    by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in head_rows:
        by_head[str(row.get("head_key"))].append(row)
    head_groups = []
    for key, vals in by_head.items():
        head_groups.append(
            {
                "model": model_name,
                "head_key": key,
                "n_rows": len(vals),
                "closure_lost_vs_none": sum(1 for row in vals if row.get("closure_lost_vs_none")),
                "damage_gt_0_25": sum(
                    1
                    for row in vals
                    if row.get("target_lift_damage_vs_none") is not None
                    and finite(row.get("target_lift_damage_vs_none")) > 0.25
                ),
                "mean_target_lift_damage_vs_none": mean(
                    [
                        finite(row.get("target_lift_damage_vs_none"))
                        for row in vals
                        if row.get("target_lift_damage_vs_none") is not None
                    ]
                )
                or 0.0,
                "max_target_lift_damage_vs_none": max(
                    [finite(row.get("target_lift_damage_vs_none")) for row in vals if row.get("target_lift_damage_vs_none") is not None]
                    or [0.0]
                ),
                "subsets": sorted(set(str(row.get("subset_key")) for row in vals)),
            }
        )
    head_groups.sort(
        key=lambda row: (
            row.get("closure_lost_vs_none") or 0,
            row.get("mean_target_lift_damage_vs_none") or 0.0,
            row.get("damage_gt_0_25") or 0,
        ),
        reverse=True,
    )

    return {
        "phase": PHASE,
        "title": "Attention-Head Attribution and Pairwise Complementarity Holdout Probe",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "model_gear_keys": [p862.gear_key(gear) for gear in model_gears],
        "selected_case_prompts": len(cases),
        "output_subset_rows": len(subset_rows),
        "output_head_rows": len(head_rows),
        "edit_modes": modes,
        "overall": {
            "closure_from_open": len(closures),
            "multi_axis_closure": len(multi),
            "positive_complementarity_rows": len(positive_comp),
            "holdout_positive_complementarity_rows": len(holdout_positive_comp),
            "closure_without_single_axis_closure": len(closure_without_single),
            "mean_multi_complementarity_over_best": mean(
                [
                    finite(row.get("complementarity_over_best_single"))
                    for row in multi
                    if row.get("complementarity_over_best_single") is not None
                ]
            )
            or 0.0,
            "head_test_sources": len(set((str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("subset_key")), str(row.get("edit_mode"))) for row in head_rows)),
            "head_zero_closure_lost": sum(1 for row in head_rows if row.get("closure_lost_vs_none")),
            "head_zero_damage_gt_0_25": len(damage_rows),
            "mean_head_target_lift_damage_vs_none": mean(
                [
                    finite(row.get("target_lift_damage_vs_none"))
                    for row in head_rows
                    if row.get("target_lift_damage_vs_none") is not None
                ]
            )
            or 0.0,
        },
        "subset_groups": subset_groups,
        "head_groups": head_groups,
        "evidence_label": evidence_label(model_name, subset_rows, head_rows),
        "domain_counts": counter_values(Counter(str(case.get("domain")) for case in cases)),
        "case_split_counts": counter_values(Counter(str(case.get("case_split")) for case in cases)),
        "boundary": (
            "Phase893 tests holdout stability of coordinate-axis complementarity and performs head-level "
            "zeroing on high-signal subset rows. Head zeroing is attribution, not full causal pathway closure."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 893 attention-head attribution and pairwise complementarity holdout probe",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
        f"- selected_case_prompts: {payload.get('selected_case_prompts')}",
        f"- output_subset_rows: {payload.get('output_subset_rows')}",
        f"- output_head_rows: {payload.get('output_head_rows')}",
        f"- closure_from_open: {payload.get('overall', {}).get('closure_from_open')}",
        f"- positive_complementarity_rows: {payload.get('overall', {}).get('positive_complementarity_rows')}",
        f"- holdout_positive_complementarity_rows: {payload.get('overall', {}).get('holdout_positive_complementarity_rows')}",
        f"- closure_without_single_axis_closure: {payload.get('overall', {}).get('closure_without_single_axis_closure')}",
        f"- mean_multi_complementarity_over_best: {finite(payload.get('overall', {}).get('mean_multi_complementarity_over_best')):.3f}",
        f"- head_zero_closure_lost: {payload.get('overall', {}).get('head_zero_closure_lost')}",
        f"- head_zero_damage_gt_0_25: {payload.get('overall', {}).get('head_zero_damage_gt_0_25')}",
        f"- mean_head_target_lift_damage_vs_none: {finite(payload.get('overall', {}).get('mean_head_target_lift_damage_vs_none')):.3f}",
        "",
        "## Subset groups",
        "",
        "| model | subset | relation | closure | holdout closure | comp rows | holdout comp | no-single closure | mean lift | mean comp | modes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("subset_groups", [])[:30]:
        lines.append(
            "| {model} | {subset} | {relation} | {closure} | {holdout_closure} | {comp} | {holdout_comp} | {no_single} | {lift:.3f} | {mean_comp:.3f} | {modes} |".format(
                model=row.get("model"),
                subset=row.get("subset_key"),
                relation=row.get("subset_relation"),
                closure=row.get("closure_from_open"),
                holdout_closure=row.get("holdout_closure_from_open"),
                comp=row.get("positive_complementarity_rows"),
                holdout_comp=row.get("holdout_positive_complementarity_rows"),
                no_single=row.get("closure_without_single_axis_closure"),
                lift=finite(row.get("mean_target_lift_on_closure")),
                mean_comp=finite(row.get("mean_complementarity_over_best")),
                modes=json.dumps(row.get("modes") or {}, ensure_ascii=False),
            )
        )
    lines.extend(
        [
            "",
            "## Head groups",
            "",
            "| model | head | closure lost | damage > 0.25 | mean damage | max damage | subsets |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload.get("head_groups", [])[:30]:
        lines.append(
            "| {model} | {head} | {lost} | {damage_rows} | {mean_damage:.3f} | {max_damage:.3f} | {subsets} |".format(
                model=row.get("model"),
                head=row.get("head_key"),
                lost=row.get("closure_lost_vs_none"),
                damage_rows=row.get("damage_gt_0_25"),
                mean_damage=finite(row.get("mean_target_lift_damage_vs_none")),
                max_damage=finite(row.get("max_target_lift_damage_vs_none")),
                subsets=",".join(row.get("subsets") or []),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [
        read_json(out_dir / f"phase893_{model}_summary.json")
        for model in MODELS
        if (out_dir / f"phase893_{model}_summary.json").exists()
    ]
    summaries = [item for item in summaries if item and item.get("status") == "complete"]
    overall = Counter()
    float_values: dict[str, list[float]] = defaultdict(list)
    subset_groups: list[dict[str, Any]] = []
    head_groups: list[dict[str, Any]] = []
    selected_case_prompts = 0
    output_subset_rows = 0
    output_head_rows = 0
    for summary in summaries:
        selected_case_prompts += int(summary.get("selected_case_prompts") or 0)
        output_subset_rows += int(summary.get("output_subset_rows") or 0)
        output_head_rows += int(summary.get("output_head_rows") or 0)
        subset_groups.extend(summary.get("subset_groups") or [])
        head_groups.extend(summary.get("head_groups") or [])
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall[key] += value
            elif isinstance(value, float):
                float_values[key].append(float(value))
    for key, values in float_values.items():
        overall[key] = finite(mean(values))
    subset_groups.sort(
        key=lambda row: (
            row.get("holdout_positive_complementarity_rows") or 0,
            row.get("positive_complementarity_rows") or 0,
            row.get("closure_from_open") or 0,
        ),
        reverse=True,
    )
    head_groups.sort(
        key=lambda row: (
            row.get("closure_lost_vs_none") or 0,
            row.get("mean_target_lift_damage_vs_none") or 0.0,
            row.get("damage_gt_0_25") or 0,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "selected_case_prompts": selected_case_prompts,
        "output_subset_rows": output_subset_rows,
        "output_head_rows": output_head_rows,
        "overall": {key: (float(value) if isinstance(value, float) else int(value)) for key, value in sorted(overall.items())},
        "subset_groups": subset_groups,
        "head_groups": head_groups,
        "evidence_label_counts": counter_values(Counter(str(summary.get("evidence_label")) for summary in summaries)),
    }
    p846.write_json(out_dir / "phase893_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase893_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="attention_head_complementarity_holdout")
    parser.add_argument("--phase892-round", default="channel_complementarity_coordinate_basis")
    parser.add_argument("--domains", default="")
    parser.add_argument("--prompt-variants", default="natural_question,natural_category,classification,question_plain,type_of_completion")
    parser.add_argument("--edit-modes", default="zero,flip,half")
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--max-cases-per-domain", type=int, default=12)
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--max-head-sources-per-model", type=int, default=8)
    parser.add_argument("--max-heads-per-layer", type=int, default=32)
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
