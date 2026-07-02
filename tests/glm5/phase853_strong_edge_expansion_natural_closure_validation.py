#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase845_geometry_gear_interaction_edge_atlas as p845  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase849_residual_blocker_route_gate_expansion as p849  # noqa: E402
import phase850_strong_edge_balanced_route_gate_validation as p850  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 853
MODELS = p846.MODELS
TARGET_CLASS = "target_equivalent"
RESULT_ROOT = Path("tests/result/phase853_strong_edge_expansion_natural_closure_validation")
PHASE844_ROOT = Path("tests/result/phase844_geometry_route_natural_gear_set_search")
PHASE851_ROOT = Path("tests/result/phase851_global_atlas_schema_orthogonality_audit")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return p844.parse_csv(text)


def gear_key(gear: dict[str, Any]) -> str:
    return str(gear.get("gear_key") or f"L{int(gear['layer_idx'])}C{int(gear['channel_id'])}")


def parse_gear_key(text: str) -> tuple[int, int] | None:
    key = str(text)
    if not key.startswith("L") or "C" not in key:
        return None
    try:
        layer_text, channel_text = key[1:].split("C", 1)
        return int(layer_text), int(channel_text)
    except ValueError:
        return None


def load_phase844_gears(model_name: str, round_name: str, top_n: int) -> list[dict[str, Any]]:
    path = PHASE844_ROOT / round_name / f"phase844_{model_name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 844 summary: {path}")
    data = p846.read_json(path)
    gears = [dict(row) for row in data.get("top_gears", [])[: int(top_n)]]
    for idx, gear in enumerate(gears, 1):
        gear["gear_rank"] = idx
        gear["gear_key"] = gear_key(gear)
        gear["source"] = "phase844_top_gear"
    return gears


def load_phase851_candidate_gears(model_name: str, round_name: str, max_candidates: int) -> list[dict[str, Any]]:
    if max_candidates <= 0:
        return []
    path = PHASE851_ROOT / round_name / f"phase851_{model_name}_atlas_audit.json"
    if not path.exists():
        return []
    data = p846.read_json(path)
    out: list[dict[str, Any]] = []
    for row in data.get("gear_min_cut_candidates") or []:
        if row.get("audit_status") != "counterfactual_min_cut_candidate":
            continue
        parsed = parse_gear_key(str(row.get("gear")))
        if not parsed:
            continue
        layer_idx, channel_id = parsed
        out.append(
            {
                "layer_idx": layer_idx,
                "channel_id": channel_id,
                "gear_key": f"L{layer_idx}C{channel_id}",
                "source": "phase851_min_cut_candidate",
                "phase851_total": row.get("total"),
                "phase851_strong": row.get("strong"),
                "phase851_lift": row.get("lift_vs_baseline"),
            }
        )
        if len(out) >= max_candidates:
            break
    return out


def merge_gears(base: list[dict[str, Any]], extra: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gear in [*base, *extra]:
        key = gear_key(gear)
        parsed = parse_gear_key(key)
        if not parsed:
            continue
        layer_idx, channel_id = parsed
        if layer_idx < 0 or layer_idx >= n_layers or channel_id < 0:
            continue
        if key in seen:
            continue
        clean = dict(gear)
        clean["layer_idx"] = layer_idx
        clean["channel_id"] = channel_id
        clean["gear_key"] = key
        clean["gear_rank"] = len(merged) + 1
        merged.append(clean)
        seen.add(key)
    return merged


def combo_key(gears: list[dict[str, Any]]) -> str:
    return "+".join(gear_key(gear) for gear in gears)


def load_phase851_focus_combos(model_name: str, round_name: str, max_combos: int) -> list[list[str]]:
    if max_combos <= 0:
        return []
    path = PHASE851_ROOT / round_name / f"phase851_{model_name}_atlas_audit.json"
    if not path.exists():
        return []
    data = p846.read_json(path)
    combos: list[list[str]] = []
    for row in data.get("combo_strong_edge_candidates") or []:
        keys = [str(x) for x in row.get("gear_keys") or p846.split_combo_key(str(row.get("combo_key")))]
        if len(keys) < 2:
            continue
        combos.append(keys)
        if len(combos) >= max_combos:
            break
    return combos


def combo_specs(
    gears: list[dict[str, Any]],
    args: argparse.Namespace,
    focus_combos: list[list[str]],
) -> list[dict[str, Any]]:
    modes = [mode for mode in parse_csv(args.edit_modes) if mode != "original"]
    specs: list[dict[str, Any]] = [{"spec_name": "original", "combo_type": "original", "mode": "original", "gears": []}]
    key_to_gear = {gear_key(gear): gear for gear in gears}
    seen: set[tuple[str, str]] = set()

    def add_spec(combo_type: str, group: tuple[dict[str, Any], ...] | list[dict[str, Any]], focus: bool = False) -> None:
        group_list = [dict(gear) for gear in group]
        key = combo_key(group_list)
        for mode in modes:
            dedupe = (mode, key)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            specs.append(
                {
                    "spec_name": f"{combo_type}_{mode}_{key}",
                    "combo_type": combo_type,
                    "mode": mode,
                    "gears": group_list,
                    "combo_key": key,
                    "gear_keys": [gear_key(gear) for gear in group_list],
                    "focus_combo": bool(focus),
                }
            )

    if "single" in parse_csv(args.combo_types):
        for gear in gears:
            add_spec("single", [gear])
    if "pair" in parse_csv(args.combo_types):
        for group in list(itertools.combinations(gears, 2))[: int(args.max_pairs)]:
            add_spec("pair", group)
    if "triplet" in parse_csv(args.combo_types):
        for group in list(itertools.combinations(gears, 3))[: int(args.max_triplets)]:
            add_spec("triplet", group)
    for keys in focus_combos:
        group = [key_to_gear[key] for key in keys if key in key_to_gear]
        if len(group) >= 2:
            add_spec("focus", group, focus=True)
    return specs


def clean_margin(scores: dict[str, Any]) -> float | None:
    value = scores.get("target_minus_object_logit")
    if value is None:
        return None
    out = finite(value, float("nan"))
    return out if math.isfinite(out) else None


def role_from_scores(scores: dict[str, Any]) -> str:
    best_rank = scores.get("best_target_rank")
    object_rank = scores.get("object_rank")
    top_tokens = scores.get("top_tokens") or []
    if best_rank == 1:
        return "target"
    if object_rank == 1:
        return "object_echo"
    if top_tokens:
        first = top_tokens[0]
        if isinstance(first, dict):
            text = str(first.get("text", ""))
        elif isinstance(first, (list, tuple)) and first:
            text = str(first[0])
        else:
            text = str(first)
        if not text.strip():
            return "format"
    return "blocker_or_other"


def exact_natural_consistency(row: dict[str, Any]) -> bool:
    return bool(row.get("target_transition")) and int(row.get("best_target_rank") or 999999) == 1


def eval_forward_rows(model, tokenizer, device: torch.device, cases: list[dict[str, Any]], gears: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    standards = p844.p828.p820.standard_rows()
    prompt_variants = parse_csv(args.prompt_variants)
    focus_combos = load_phase851_focus_combos(args.model, args.phase851_round, int(args.max_focus_combos))
    specs = combo_specs(gears, args, focus_combos)
    rows: list[dict[str, Any]] = []
    for case_idx, case in enumerate(cases, 1):
        for variant in prompt_variants:
            prompt = p844.prompt_for_case(case, variant)
            prompt_ids = p844.encode_prompt(tokenizer, prompt)
            original_text, original_ids = p844.greedy_with_gears(
                model, tokenizer, device, prompt_ids, [], "original", int(args.max_new_tokens)
            )
            original_boundary = p844.classify_output(case, original_text, standards)
            baseline_id = int(original_ids[0]) if original_ids else None
            original_logits = p844.first_logits_with_gears(model, device, prompt_ids, [], "original")
            original_scores = p844.token_scores(tokenizer, original_logits, case, baseline_id)
            original_margin = clean_margin(original_scores)
            single_delta_by_mode_key: dict[tuple[str, str], float] = {}
            local_rows: list[dict[str, Any]] = []
            for spec in specs:
                if spec["combo_type"] == "original":
                    logits = original_logits
                    generated = original_text
                    token_ids = original_ids
                    boundary = original_boundary
                else:
                    logits = p844.first_logits_with_gears(model, device, prompt_ids, spec["gears"], spec["mode"])
                    if spec["combo_type"] == "single" and not args.generate_singles:
                        generated = ""
                        token_ids = []
                        boundary = {"final_boundary_class": "not_generated_single", "boundary_rank": -1}
                    else:
                        generated, token_ids = p844.greedy_with_gears(
                            model,
                            tokenizer,
                            device,
                            prompt_ids,
                            spec["gears"],
                            spec["mode"],
                            int(args.max_new_tokens),
                        )
                        boundary = p844.classify_output(case, generated, standards)
                scores = p844.token_scores(tokenizer, logits, case, baseline_id)
                margin = clean_margin(scores)
                delta = None if margin is None or original_margin is None else float(margin - original_margin)
                gear_keys = [gear_key(gear) for gear in spec.get("gears", [])]
                if spec["combo_type"] == "single" and delta is not None and gear_keys:
                    single_delta_by_mode_key[(str(spec["mode"]), gear_keys[0])] = delta
                row = {
                    "row_kind": "phase853_strong_edge_expansion_natural_closure",
                    "phase": PHASE,
                    "model": args.model,
                    "round": args.round_name,
                    "phase844_round": args.phase844_round,
                    "phase851_round": args.phase851_round,
                    "case_id": case["case_id"],
                    "object": case.get("object"),
                    "target_answer": case.get("answer"),
                    "synthetic_case": bool(case.get("synthetic_case")),
                    "prompt_variant": variant,
                    "prompt": prompt,
                    "combo_type": spec["combo_type"],
                    "spec_name": spec["spec_name"],
                    "edit_mode": spec["mode"],
                    "gear_count": len(gear_keys),
                    "gear_keys": gear_keys,
                    "combo_key": spec.get("combo_key") or "original",
                    "focus_combo": bool(spec.get("focus_combo")),
                    "generated": p844.p828.p825.clean_generated(generated),
                    "token_ids": token_ids,
                    "boundary_class": boundary.get("final_boundary_class"),
                    "boundary_rank": int(boundary.get("boundary_rank", 0)),
                    "target_transition": boundary.get("final_boundary_class") == TARGET_CLASS,
                    "original_generated": p844.p828.p825.clean_generated(original_text),
                    "original_boundary_class": original_boundary.get("final_boundary_class"),
                    "original_boundary_rank": int(original_boundary.get("boundary_rank", 0)),
                    "original_target_transition": original_boundary.get("final_boundary_class") == TARGET_CLASS,
                    "target_lost_vs_original": bool(
                        original_boundary.get("final_boundary_class") == TARGET_CLASS
                        and boundary.get("final_boundary_class") != TARGET_CLASS
                    ),
                    "target_gained_vs_original": bool(
                        original_boundary.get("final_boundary_class") != TARGET_CLASS
                        and boundary.get("final_boundary_class") == TARGET_CLASS
                    ),
                    "original_target_minus_object_logit": original_margin,
                    "margin_delta_vs_original": delta,
                    "expected_additive_delta": None,
                    "interaction_residual": None,
                    "top1_role_proxy": role_from_scores(scores),
                    **scores,
                }
                row["exact_natural_consistency"] = exact_natural_consistency(row)
                local_rows.append(row)

            for row in local_rows:
                if row["combo_type"] not in {"pair", "triplet", "focus"}:
                    continue
                expected_terms = [
                    single_delta_by_mode_key.get((str(row["edit_mode"]), key))
                    for key in row.get("gear_keys", [])
                ]
                if not expected_terms or any(value is None for value in expected_terms):
                    continue
                expected = float(sum(float(value) for value in expected_terms if value is not None))
                actual = row.get("margin_delta_vs_original")
                if actual is not None:
                    row["expected_additive_delta"] = expected
                    row["interaction_residual"] = float(actual) - expected
            rows.extend(local_rows)
        if case_idx % max(1, int(args.log_every)) == 0 or case_idx == len(cases):
            log(f"{args.model}/{args.round_name}: forward cases {case_idx}/{len(cases)} rows={len(rows)} specs={len(specs)}")
    return rows


def residual_class(value: float, threshold: float) -> str:
    return p849.residual_class(value, threshold)


def class_is_strong(label: str) -> bool:
    return p850.class_is_strong(label)


def interaction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("combo_type") in {"pair", "triplet", "focus"} and row.get("interaction_residual") is not None
    ]


def closure_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    usable = interaction_rows(rows)
    strong = [row for row in usable if class_is_strong(residual_class(finite(row.get("interaction_residual")), threshold))]
    by_class = Counter(residual_class(finite(row.get("interaction_residual")), threshold) for row in usable)
    def compact(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(group),
            "target_transition": sum(1 for row in group if row.get("target_transition")),
            "target_gained_vs_original": sum(1 for row in group if row.get("target_gained_vs_original")),
            "target_lost_vs_original": sum(1 for row in group if row.get("target_lost_vs_original")),
            "exact_natural_consistency": sum(1 for row in group if row.get("exact_natural_consistency")),
            "top1_role_proxy": dict(Counter(str(row.get("top1_role_proxy")) for row in group)),
            "boundary_classes": dict(Counter(str(row.get("boundary_class")) for row in group)),
            "mean_abs_residual": mean([abs(finite(row.get("interaction_residual"))) for row in group]),
            "mean_target_minus_object_logit": mean(
                [finite(row.get("target_minus_object_logit")) for row in group if row.get("target_minus_object_logit") is not None]
            ),
        }
    by_combo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_combo[str(row.get("combo_type"))].append(row)
        by_prompt[str(row.get("prompt_variant"))].append(row)
        by_object[str(row.get("object"))].append(row)
    return {
        "usable_interaction_rows": len(usable),
        "classes": dict(by_class),
        "strong_rows": len(strong),
        "all_rows": compact(usable),
        "strong_rows_summary": compact(strong),
        "by_combo_type": {key: compact(group) for key, group in sorted(by_combo.items())},
        "by_prompt": {key: compact(group) for key, group in sorted(by_prompt.items())},
        "by_object": {key: compact(group) for key, group in sorted(by_object.items())},
    }


def enrich_feature_rows(feature_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_by_key = {
        "|".join(
            str(row.get(k, ""))
            for k in ["case_id", "object", "prompt_variant", "combo_type", "edit_mode", "combo_key"]
        ): row
        for row in source_rows
    }
    out: list[dict[str, Any]] = []
    for row in feature_rows:
        key = "|".join(
            str(row.get(k, ""))
            for k in ["case_id", "object", "prompt_variant", "combo_type", "edit_mode", "combo_key"]
        )
        source = source_by_key.get(key, {})
        merged = dict(row)
        for name in [
            "generated",
            "boundary_class",
            "boundary_rank",
            "target_transition",
            "target_gained_vs_original",
            "target_lost_vs_original",
            "exact_natural_consistency",
            "top1_role_proxy",
            "focus_combo",
        ]:
            merged[name] = source.get(name)
        out.append(merged)
    return out


def gate_validation(feature_rows: list[dict[str, Any]], model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    split_types = [part.strip() for part in args.split_types.split(",") if part.strip()]
    specs = p850.split_specs(feature_rows, split_types)
    split_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for idx, (split_type, split_key, train_rows, test_rows) in enumerate(specs, 1):
        result = p850.evaluate_split(
            train_rows,
            test_rows,
            split_type,
            split_key,
            model_name,
            float(args.interaction_threshold),
        )
        split_results.append(result)
        all_predictions.extend(result["predictions"])
        if idx % max(1, int(args.log_every)) == 0 or idx == len(specs):
            log(f"{model_name}/{args.round_name}: gate split {idx}/{len(specs)} {split_type}:{split_key}")
    return {
        "split_results_compact": [{k: v for k, v in result.items() if k != "predictions"} for result in split_results],
        "split_summary": p850.aggregate(split_results),
        "predictions": all_predictions,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = p844.selected_cases(args)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "cases": [case["case_id"] for case in cases],
            "prompt_variants": parse_csv(args.prompt_variants),
            "phase844_round": args.phase844_round,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        gears = merge_gears(
            load_phase844_gears(args.model, args.phase844_round, int(args.top_gears)),
            load_phase851_candidate_gears(args.model, args.phase851_round, int(args.max_min_cut_gears)),
            n_layers,
        )
        rows = eval_forward_rows(model, tokenizer, device, cases, gears, args)
        usable_rows = interaction_rows(rows)
        prompt_states = p849.capture_internal_states(model, tokenizer, device, usable_rows, gears, args)
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    feature_rows = p849.make_feature_rows(usable_rows, prompt_states)
    feature_rows = enrich_feature_rows(feature_rows, usable_rows)
    gate = gate_validation(feature_rows, args.model, args)
    summary = {
        "phase": PHASE,
        "title": "Strong-edge Expansion and Natural Closure Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "phase844_round": args.phase844_round,
        "phase851_round": args.phase851_round,
        "n_cases": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "prompt_variants": parse_csv(args.prompt_variants),
        "n_gears": len(gears),
        "gears": gears,
        "n_rows": len(rows),
        "n_interaction_rows": len(usable_rows),
        "n_feature_rows": len(feature_rows),
        "closure_summary": closure_summary(rows, float(args.interaction_threshold)),
        "feature_summary": p850.feature_summary(feature_rows, float(args.interaction_threshold)),
        "gate_split_summary": gate["split_summary"],
        "boundary": (
            "This phase runs new forward passes and evaluates whether expanded gear interactions create more "
            "strong-edge rows and whether those strong edges reach natural output closure. It is not final language closure."
        ),
    }
    p846.write_jsonl(out_dir / f"phase853_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase853_{args.model}_feature_rows.jsonl", feature_rows)
    p846.write_jsonl(out_dir / f"phase853_{args.model}_predictions.jsonl", gate["predictions"])
    p846.write_json(out_dir / f"phase853_{args.model}_split_results.json", gate["split_results_compact"])
    p846.write_json(out_dir / f"phase853_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "rows": len(rows),
                "interaction_rows": len(usable_rows),
                "strong_rows": summary["closure_summary"]["strong_rows"],
                "exact_natural_strong": summary["closure_summary"]["strong_rows_summary"]["exact_natural_consistency"],
                "classes": summary["closure_summary"]["classes"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def metric(summary: dict[str, Any], split: str, predictor: str, mode: str) -> dict[str, Any] | None:
    split_row = (summary.get("gate_split_summary") or {}).get(split) or {}
    source = split_row.get(f"{mode}_summary") or {}
    return source.get(predictor)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 853 Strong-edge Expansion and Natural Closure Validation ({payload['round']})",
        "",
        "- Source: new BF16 forward passes over expanded Phase 844/851 gear sets.",
        "- Boundary: strong-edge expansion + natural closure audit, not final language closure.",
        "",
        "## Expansion / Closure",
        "",
        "| model | rows | interaction rows | strong rows | target in strong | exact natural in strong | classes | strong boundaries |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        closure = data.get("closure_summary") or {}
        strong = closure.get("strong_rows_summary") or {}
        lines.append(
            f"| {model_name} | {data.get('n_rows', 0)} | {data.get('n_interaction_rows', 0)} | "
            f"{closure.get('strong_rows', 0)} | {strong.get('target_transition', 0)} | "
            f"{strong.get('exact_natural_consistency', 0)} | "
            f"`{json.dumps(closure.get('classes') or {}, ensure_ascii=False)}` | "
            f"`{json.dumps(strong.get('boundary_classes') or {}, ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "## Gate Holdout Summary",
        "",
        "| model | predictor | in F1 | object F1 | prompt F1 | object balanced F1 | prompt balanced F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for predictor in ["global_combo", "residual_projection_combo", "blocker_field_combo", "model_default_gate", "train_selected_gate"]:
            in_stats = metric(data, "in_sample", predictor, "raw") or {}
            obj_stats = metric(data, "object_holdout", predictor, "raw") or {}
            prompt_stats = metric(data, "prompt_holdout", predictor, "raw") or {}
            obj_bal = metric(data, "object_holdout", predictor, "balanced") or {}
            prompt_bal = metric(data, "prompt_holdout", predictor, "balanced") or {}
            lines.append(
                f"| {model_name} | `{predictor}` | {fmt((in_stats.get('strong') or {}).get('f1'))} | "
                f"{fmt((obj_stats.get('strong') or {}).get('f1'))} | "
                f"{fmt((prompt_stats.get('strong') or {}).get('f1'))} | "
                f"{fmt((obj_bal.get('strong') or {}).get('f1'))} | "
                f"{fmt((prompt_bal.get('strong') or {}).get('f1'))} |"
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
        "models": [],
        "model_summaries": {},
    }
    for model_name in MODELS:
        path = out_dir / f"phase853_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = p846.read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase853_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase853_cross_model_summary.md", payload)
    print(json.dumps({"status": payload["status"], "round": round_name, "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase844-round", default="confirm")
    parser.add_argument("--phase851-round", default="confirm")
    parser.add_argument("--include-seed-triangle", action="store_true")
    parser.add_argument("--geometry-objects", default="triangle")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--prompt-variants", default="natural_question")
    parser.add_argument("--top-gears", type=int, default=4)
    parser.add_argument("--max-min-cut-gears", type=int, default=2)
    parser.add_argument("--max-focus-combos", type=int, default=4)
    parser.add_argument("--combo-types", default="single,pair")
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--max-triplets", type=int, default=0)
    parser.add_argument("--edit-modes", default="zero,flip")
    parser.add_argument("--split-types", default="in_sample,object_holdout,prompt_holdout")
    parser.add_argument("--interaction-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--topk-entropy", type=int, default=20)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--generate-singles", action="store_true")
    parser.add_argument("--no-neighbor-layers", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)


if __name__ == "__main__":
    main()
