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


PHASE = 892
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE891_ROOT = Path("tests/result/phase891_target_lift_source_pathway_projection_audit")
RESULT_ROOT = Path("tests/result/phase892_channel_complementarity_coordinate_basis_probe")


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


def gear_tuple(gear: dict[str, Any]) -> tuple[int, int]:
    return (int(gear["layer_idx"]), int(gear["channel_id"]))


def gear_tuple_key(item: tuple[int, int]) -> str:
    return f"L{int(item[0])}C{int(item[1])}"


def gear_sort_key(gear: dict[str, Any]) -> tuple[int, int]:
    return gear_tuple(gear)


def dedupe_gears(gears: list[dict[str, Any]], max_gears: int | None = None) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    for gear in sorted(gears, key=gear_sort_key):
        key = gear_tuple(gear)
        if key in seen:
            continue
        seen.add(key)
        out.append({"layer_idx": key[0], "channel_id": key[1]})
        if max_gears is not None and len(out) >= int(max_gears):
            break
    return out


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_phase891_rows(round_name: str, model: str) -> list[dict[str, Any]]:
    path = PHASE891_ROOT / round_name / f"phase891_{model}_rows.jsonl"
    return p846.read_jsonl(path) if path.exists() else []


def load_model_gears(round_name: str, model: str) -> list[dict[str, Any]]:
    summary = read_json(PHASE891_ROOT / round_name / f"phase891_{model}_summary.json")
    gears = []
    for key in summary.get("model_gear_keys") or []:
        gear = p862.parse_gear_key(str(key))
        if gear is not None:
            gears.append(gear)
    if gears:
        return dedupe_gears(gears)
    rows = load_phase891_rows(round_name, model)
    for row in rows:
        for key in row.get("gear_keys") or []:
            gear = p862.parse_gear_key(str(key))
            if gear is not None:
                gears.append(gear)
    return dedupe_gears(gears)


def source_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("parent_candidate_key")), str(row.get("case_id")), str(row.get("prompt_variant")))


def source_from_rows(vals: list[dict[str, Any]]) -> dict[str, Any]:
    row = vals[0]
    none_rows = [item for item in vals if item.get("control_type") == "none"]
    closures = [item for item in none_rows if item.get("none_closure_from_open")]
    return {
        "parent_candidate_key": row.get("parent_candidate_key"),
        "case_id": row.get("case_id"),
        "case_split": row.get("case_split"),
        "eval_domain": row.get("eval_domain"),
        "object": row.get("object"),
        "prompt_variant": row.get("prompt_variant"),
        "phase891_none_closure_count": len(closures),
        "phase891_had_closure": bool(closures),
    }


def select_sources(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[source_key(row)].append(row)
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vals in buckets.values():
        item = source_from_rows(vals)
        by_candidate[str(item["parent_candidate_key"])].append(item)

    selected: list[dict[str, Any]] = []
    for _candidate, vals in sorted(by_candidate.items()):
        positives = [row for row in vals if row.get("phase891_had_closure")]
        controls = [row for row in vals if not row.get("phase891_had_closure")]
        positives.sort(
            key=lambda row: (
                int(row.get("phase891_none_closure_count") or 0),
                str(row.get("object")),
                str(row.get("prompt_variant")),
            ),
            reverse=True,
        )
        controls.sort(key=lambda row: (str(row.get("object")), str(row.get("prompt_variant"))))
        selected.extend(positives[: int(args.max_closure_cases_per_candidate)])
        selected.extend(controls[: int(args.max_control_cases_per_candidate)])
    return selected


def subset_specs(model_gears: list[dict[str, Any]], parent_gears: list[dict[str, Any]], max_subset_size: int) -> list[dict[str, Any]]:
    gears = dedupe_gears(model_gears)
    parent = set(gear_tuple(gear) for gear in parent_gears)
    out: list[dict[str, Any]] = []
    for size in range(1, min(int(max_subset_size), len(gears)) + 1):
        for combo in itertools.combinations(gears, size):
            combo_gears = list(combo)
            tuples = tuple(gear_tuple(gear) for gear in combo_gears)
            keys = [gear_tuple_key(item) for item in tuples]
            combo_set = set(tuples)
            if size == 1:
                relation = "parent_axis" if combo_set <= parent else "other_axis"
            elif combo_set == parent and len(parent) > 1:
                relation = "parent_multi_axis"
            elif parent and combo_set & parent:
                relation = "with_parent_axis"
            else:
                relation = "without_parent_axis"
            if size == len(gears):
                relation = "model_U"
            out.append(
                {
                    "subset_type": f"{size}_axis",
                    "subset_relation": relation,
                    "subset_key": "+".join(keys),
                    "subset_size": size,
                    "gear_keys": keys,
                    "gears": [{"layer_idx": item[0], "channel_id": item[1]} for item in tuples],
                }
            )
    return out


def make_row(
    model_name: str,
    source: dict[str, Any],
    spec: dict[str, Any],
    mode: str,
    base_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    base_class = base_metrics.get("class_best_logit")
    current_class = metrics.get("class_best_logit")
    target_lift = None if base_class is None or current_class is None else finite(current_class) - finite(base_class)
    blocker_reduction = None
    if base_metrics.get("class_blocker_count") is not None and metrics.get("class_blocker_count") is not None:
        blocker_reduction = finite(base_metrics.get("class_blocker_count")) - finite(metrics.get("class_blocker_count"))
    return {
        "phase": PHASE,
        "row_kind": "phase892_channel_complementarity_row",
        "model": model_name,
        "parent_candidate_key": source.get("parent_candidate_key"),
        "case_id": source.get("case_id"),
        "case_split": source.get("case_split"),
        "eval_domain": source.get("eval_domain"),
        "object": source.get("object"),
        "prompt_variant": source.get("prompt_variant"),
        "edit_mode": mode,
        "subset_type": spec.get("subset_type"),
        "subset_relation": spec.get("subset_relation"),
        "subset_key": spec.get("subset_key"),
        "subset_size": spec.get("subset_size"),
        "gear_keys": spec.get("gear_keys"),
        "base_boundary_closed": bool(base_metrics.get("class_boundary_closed")),
        "boundary_closed": bool(metrics.get("class_boundary_closed")),
        "closure_from_open": bool((not base_metrics.get("class_boundary_closed")) and metrics.get("class_boundary_closed")),
        "base_class_logit": base_class,
        "class_logit": current_class,
        "target_lift": target_lift,
        "base_blocker_count": base_metrics.get("class_blocker_count"),
        "blocker_count": metrics.get("class_blocker_count"),
        "blocker_reduction": blocker_reduction,
    }


def add_complementarity_fields(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("parent_candidate_key")),
            str(row.get("edit_mode")),
        )
        groups[key].append(row)
    for vals in groups.values():
        singles = {
            str(row.get("subset_key")): row
            for row in vals
            if int(row.get("subset_size") or 0) == 1 and row.get("target_lift") is not None
        }
        single_lifts = {key: finite(row.get("target_lift")) for key, row in singles.items()}
        single_closures = {key: bool(row.get("closure_from_open")) for key, row in singles.items()}
        for row in vals:
            keys = [str(key) for key in row.get("gear_keys") or []]
            component_lifts = [single_lifts[key] for key in keys if key in single_lifts]
            additive = sum(component_lifts) if component_lifts else None
            max_single = max(component_lifts) if component_lifts else None
            any_single_closure = any(single_closures.get(key, False) for key in keys)
            target_lift = finite(row.get("target_lift")) if row.get("target_lift") is not None else None
            row["additive_expected_target_lift"] = additive
            row["max_single_target_lift"] = max_single
            row["any_single_axis_closure"] = bool(any_single_closure)
            row["interaction_residual_vs_additive"] = None if additive is None or target_lift is None else target_lift - additive
            row["complementarity_over_best_single"] = None if max_single is None or target_lift is None else target_lift - max_single
            row["closure_without_single_axis_closure"] = bool(row.get("closure_from_open") and not any_single_closure and int(row.get("subset_size") or 0) > 1)


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    phase891_rows = load_phase891_rows(args.phase891_round, args.model)
    selected = select_sources(phase891_rows, args)
    model_gears = load_model_gears(args.phase891_round, args.model)
    modes = parse_csv(args.edit_modes)
    if args.dry_run or not selected or not model_gears:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run" if selected and model_gears else "no_sources_or_gears",
            "selected_sources": selected,
            "model_gear_keys": [p862.gear_key(gear) for gear in model_gears],
            "edit_modes": modes,
        }
        p846.write_json(out_dir / f"phase892_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase892_{args.model}_rows.jsonl", [])
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cases = p888.case_map()
        cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], list[int]]] = {}
        for idx, source in enumerate(selected, 1):
            case = cases.get(str(source.get("case_id")))
            if not case:
                continue
            prompt_variant = str(source.get("prompt_variant"))
            prompt = p885.prompt_for_case(case, prompt_variant)
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            token_sets = p856.token_sets(tokenizer, case)
            cache_key = (str(source.get("case_id")), prompt_variant)
            if cache_key not in cache:
                base_logits = p862.first_logits_with_scaled_gears(
                    model, device, prompt_ids, [], "original", float(args.scale_up_factor)
                )
                base_metrics = p888.metrics_for_logits(tokenizer, base_logits, token_sets, int(args.topk_tokens))
                cache[cache_key] = (base_metrics, token_sets, prompt_ids)
            base_metrics, token_sets, prompt_ids = cache[cache_key]
            parent_gears = p885.parse_gears_from_candidate_key(str(source.get("parent_candidate_key")))
            specs = subset_specs(model_gears, parent_gears, int(args.max_subset_size))
            for spec in specs:
                for mode in modes:
                    logits = p862.first_logits_with_scaled_gears(
                        model,
                        device,
                        prompt_ids,
                        spec["gears"],
                        mode,
                        float(args.scale_up_factor),
                    )
                    metrics = p888.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens))
                    rows.append(make_row(args.model, source, spec, mode, base_metrics, metrics))
            log(f"{args.model}/{args.round_name}: source={idx}/{len(selected)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    add_complementarity_fields(rows)
    payload = summarize_model(args.model, rows, selected, model_gears, modes, attn_impl)
    p846.write_json(out_dir / f"phase892_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase892_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def evidence_label(vals: list[dict[str, Any]]) -> str:
    closures = [row for row in vals if row.get("closure_from_open")]
    if not closures:
        return "negative_no_channel_complementarity"
    multi = [row for row in closures if int(row.get("subset_size") or 0) > 1]
    positive_comp = [
        row
        for row in multi
        if row.get("complementarity_over_best_single") is not None
        and finite(row.get("complementarity_over_best_single")) > 0.25
    ]
    closure_without_single = [row for row in multi if row.get("closure_without_single_axis_closure")]
    if closure_without_single:
        return "multi_axis_closure_without_single_axis"
    if positive_comp:
        return "multi_axis_target_lift_complementarity"
    return "single_axis_dominant_target_lift"


def summarize_model(
    model_name: str,
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    model_gears: list[dict[str, Any]],
    modes: list[str],
    attn_impl: str | None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("parent_candidate_key"))].append(row)

    candidate_groups = []
    for key, vals in groups.items():
        closures = [row for row in vals if row.get("closure_from_open")]
        multi = [row for row in closures if int(row.get("subset_size") or 0) > 1]
        single = [row for row in closures if int(row.get("subset_size") or 0) == 1]
        positive_comp = [
            row
            for row in multi
            if row.get("complementarity_over_best_single") is not None
            and finite(row.get("complementarity_over_best_single")) > 0.25
        ]
        closure_without_single = [row for row in multi if row.get("closure_without_single_axis_closure")]
        subset_means: dict[str, list[float]] = defaultdict(list)
        for row in closures:
            if row.get("target_lift") is not None:
                subset_means[str(row.get("subset_key"))].append(finite(row.get("target_lift")))
        best_subset = None
        if subset_means:
            best_key, best_vals = max(subset_means.items(), key=lambda item: finite(mean(item[1])))
            best_subset = {"subset_key": best_key, "mean_target_lift": finite(mean(best_vals)), "n": len(best_vals)}
        candidate_groups.append(
            {
                "model": model_name,
                "parent_candidate_key": key,
                "evidence_label": evidence_label(vals),
                "n_rows": len(vals),
                "n_source_cases": len(set((str(row.get("case_id")), str(row.get("prompt_variant"))) for row in vals)),
                "closure_from_open": len(closures),
                "single_axis_closure": len(single),
                "multi_axis_closure": len(multi),
                "positive_complementarity_rows": len(positive_comp),
                "closure_without_single_axis_closure": len(closure_without_single),
                "mean_target_lift": mean([finite(row.get("target_lift")) for row in closures if row.get("target_lift") is not None])
                or 0.0,
                "mean_multi_complementarity_over_best": mean(
                    [
                        finite(row.get("complementarity_over_best_single"))
                        for row in multi
                        if row.get("complementarity_over_best_single") is not None
                    ]
                )
                or 0.0,
                "mean_interaction_residual_vs_additive": mean(
                    [
                        finite(row.get("interaction_residual_vs_additive"))
                        for row in multi
                        if row.get("interaction_residual_vs_additive") is not None
                    ]
                )
                or 0.0,
                "closure_subset_types": counter_values(Counter(str(row.get("subset_type")) for row in closures)),
                "closure_subset_relations": counter_values(Counter(str(row.get("subset_relation")) for row in closures)),
                "closure_modes": counter_values(Counter(str(row.get("edit_mode")) for row in closures)),
                "best_subset": best_subset,
                "objects": sorted(set(str(row.get("object")) for row in vals)),
            }
        )
    candidate_groups.sort(
        key=lambda row: (
            row.get("positive_complementarity_rows") or 0,
            row.get("multi_axis_closure") or 0,
            row.get("closure_from_open") or 0,
        ),
        reverse=True,
    )
    closures = [row for row in rows if row.get("closure_from_open")]
    multi = [row for row in closures if int(row.get("subset_size") or 0) > 1]
    return {
        "phase": PHASE,
        "title": "Channel Complementarity and Coordinate Basis Probe",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_sources": len(selected),
        "output_rows": len(rows),
        "model_gear_keys": [p862.gear_key(gear) for gear in model_gears],
        "edit_modes": modes,
        "overall": {
            "closure_from_open": len(closures),
            "single_axis_closure": sum(1 for row in closures if int(row.get("subset_size") or 0) == 1),
            "multi_axis_closure": len(multi),
            "positive_complementarity_rows": sum(
                1
                for row in multi
                if row.get("complementarity_over_best_single") is not None
                and finite(row.get("complementarity_over_best_single")) > 0.25
            ),
            "closure_without_single_axis_closure": sum(1 for row in multi if row.get("closure_without_single_axis_closure")),
            "mean_target_lift": mean([finite(row.get("target_lift")) for row in closures if row.get("target_lift") is not None])
            or 0.0,
            "mean_multi_complementarity_over_best": mean(
                [
                    finite(row.get("complementarity_over_best_single"))
                    for row in multi
                    if row.get("complementarity_over_best_single") is not None
                ]
            )
            or 0.0,
            "mean_interaction_residual_vs_additive": mean(
                [
                    finite(row.get("interaction_residual_vs_additive"))
                    for row in multi
                    if row.get("interaction_residual_vs_additive") is not None
                ]
            )
            or 0.0,
        },
        "candidate_groups": candidate_groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in candidate_groups)),
        "boundary": (
            "Phase892 tests coordinate-axis channel subsets. It is finer than component ablation, "
            "but it is not an arbitrary learned projection basis."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 892 channel complementarity and coordinate basis probe",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
        f"- selected_sources: {payload.get('selected_sources')}",
        f"- output_rows: {payload.get('output_rows')}",
        f"- closure_from_open: {payload.get('overall', {}).get('closure_from_open')}",
        f"- single_axis_closure: {payload.get('overall', {}).get('single_axis_closure')}",
        f"- multi_axis_closure: {payload.get('overall', {}).get('multi_axis_closure')}",
        f"- positive_complementarity_rows: {payload.get('overall', {}).get('positive_complementarity_rows')}",
        f"- closure_without_single_axis_closure: {payload.get('overall', {}).get('closure_without_single_axis_closure')}",
        f"- mean_multi_complementarity_over_best: {finite(payload.get('overall', {}).get('mean_multi_complementarity_over_best')):.3f}",
        f"- mean_interaction_residual_vs_additive: {finite(payload.get('overall', {}).get('mean_interaction_residual_vs_additive')):.3f}",
        "",
        "## Candidate groups",
        "",
        "| model | candidate | label | closures | single | multi | comp rows | no-single closures | mean comp | best subset | modes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload.get("candidate_groups", [])[:30]:
        lines.append(
            "| {model} | {key} | {label} | {closures} | {single} | {multi} | {comp} | {no_single} | {mean_comp:.3f} | {best} | {modes} |".format(
                model=row.get("model"),
                key=row.get("parent_candidate_key"),
                label=row.get("evidence_label"),
                closures=row.get("closure_from_open"),
                single=row.get("single_axis_closure"),
                multi=row.get("multi_axis_closure"),
                comp=row.get("positive_complementarity_rows"),
                no_single=row.get("closure_without_single_axis_closure"),
                mean_comp=finite(row.get("mean_multi_complementarity_over_best")),
                best=json.dumps(row.get("best_subset") or {}, ensure_ascii=False),
                modes=json.dumps(row.get("closure_modes") or {}, ensure_ascii=False),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [
        read_json(out_dir / f"phase892_{model}_summary.json")
        for model in MODELS
        if (out_dir / f"phase892_{model}_summary.json").exists()
    ]
    summaries = [item for item in summaries if item and item.get("status") == "complete"]
    overall = Counter()
    float_values: dict[str, list[float]] = defaultdict(list)
    groups: list[dict[str, Any]] = []
    selected_sources = 0
    output_rows = 0
    for summary in summaries:
        selected_sources += int(summary.get("selected_sources") or 0)
        output_rows += int(summary.get("output_rows") or 0)
        groups.extend(summary.get("candidate_groups") or [])
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall[key] += value
            elif isinstance(value, float):
                float_values[key].append(float(value))
    for key, values in float_values.items():
        overall[key] = finite(mean(values))
    groups.sort(
        key=lambda row: (
            row.get("positive_complementarity_rows") or 0,
            row.get("multi_axis_closure") or 0,
            row.get("closure_from_open") or 0,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "selected_sources": selected_sources,
        "output_rows": output_rows,
        "overall": {key: (float(value) if isinstance(value, float) else int(value)) for key, value in sorted(overall.items())},
        "candidate_groups": groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in groups)),
    }
    p846.write_json(out_dir / "phase892_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase892_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="channel_complementarity_coordinate_basis")
    parser.add_argument("--phase891-round", default="target_lift_source_projection")
    parser.add_argument("--edit-modes", default="zero,flip,half")
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--max-closure-cases-per-candidate", type=int, default=8)
    parser.add_argument("--max-control-cases-per-candidate", type=int, default=1)
    parser.add_argument("--max-subset-size", type=int, default=3)
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
