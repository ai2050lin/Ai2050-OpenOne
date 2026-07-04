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
from model_utils import get_layers  # noqa: E402


PHASE = 891
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE890_ROOT = Path("tests/result/phase890_distributed_restore_projection_subspace")
RESULT_ROOT = Path("tests/result/phase891_target_lift_source_pathway_projection_audit")


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


def dedupe_gears(gears: list[dict[str, Any]], max_gears: int | None = None) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    for gear in gears:
        key = gear_tuple(gear)
        if key in seen:
            continue
        seen.add(key)
        out.append({"layer_idx": key[0], "channel_id": key[1]})
        if max_gears is not None and len(out) >= int(max_gears):
            break
    return out


def load_phase890_rows(round_name: str, model: str) -> list[dict[str, Any]]:
    path = PHASE890_ROOT / round_name / f"phase890_{model}_rows.jsonl"
    return p846.read_jsonl(path) if path.exists() else []


def source_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("parent_candidate_key")), str(row.get("case_id")), str(row.get("prompt_variant")))


def source_from_rows(vals: list[dict[str, Any]]) -> dict[str, Any]:
    row = vals[0]
    closure_rows = [item for item in vals if item.get("mode_closure_from_open")]
    projection_rows = [item for item in closure_rows if item.get("is_projection_style")]
    restore_rows = [item for item in vals if item.get("restore_reopens_boundary")]
    return {
        "parent_candidate_key": row.get("parent_candidate_key"),
        "case_id": row.get("case_id"),
        "case_split": row.get("case_split"),
        "eval_domain": row.get("eval_domain"),
        "object": row.get("object"),
        "prompt_variant": row.get("prompt_variant"),
        "phase890_closure_count": len(closure_rows),
        "phase890_projection_closure_count": len(projection_rows),
        "phase890_restore_reopen_count": len(restore_rows),
        "phase890_had_closure": bool(closure_rows),
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
        positives = [row for row in vals if row.get("phase890_had_closure")]
        controls = [row for row in vals if not row.get("phase890_had_closure")]
        positives.sort(
            key=lambda row: (
                int(row.get("phase890_closure_count") or 0),
                int(row.get("phase890_projection_closure_count") or 0),
                str(row.get("object")),
                str(row.get("prompt_variant")),
            ),
            reverse=True,
        )
        controls.sort(key=lambda row: (str(row.get("object")), str(row.get("prompt_variant"))))
        selected.extend(positives[: int(args.max_closure_cases_per_candidate)])
        selected.extend(controls[: int(args.max_control_cases_per_candidate)])
    return selected


def all_model_gears(rows: list[dict[str, Any]], max_gears: int) -> list[dict[str, Any]]:
    gears: list[dict[str, Any]] = []
    for row in rows:
        gears.extend(p885.parse_gears_from_candidate_key(str(row.get("parent_candidate_key"))))
    return dedupe_gears(gears, int(max_gears))


def gear_set_specs(source: dict[str, Any], model_gears: list[dict[str, Any]], max_gears: int) -> list[dict[str, Any]]:
    candidate_gears = dedupe_gears(p885.parse_gears_from_candidate_key(str(source.get("parent_candidate_key"))), max_gears)
    specs: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    def add(name: str, gears: list[dict[str, Any]]) -> None:
        cleaned = dedupe_gears(gears, max_gears)
        if not cleaned:
            return
        key = tuple(sorted(gear_tuple(gear) for gear in cleaned))
        if key in seen:
            return
        seen.add(key)
        specs.append(
            {
                "gear_set_type": name,
                "gears": cleaned,
                "gear_keys": [p862.gear_key(gear) for gear in cleaned],
                "gear_count": len(cleaned),
                "target_layers": sorted(set(int(gear["layer_idx"]) for gear in cleaned)),
                "is_multi_axis": len(cleaned) > 1,
            }
        )

    add("candidate_axis", candidate_gears)
    candidate_layers = sorted(set(int(gear["layer_idx"]) for gear in candidate_gears))
    if candidate_layers:
        same_layer = [gear for gear in model_gears if int(gear["layer_idx"]) in set(candidate_layers)]
        add("same_layer_U", same_layer)
    add("model_U", model_gears)
    return specs


def patch_module_output(output: Any, scale: float) -> Any:
    def patch_tensor(tensor: torch.Tensor) -> torch.Tensor:
        patched = tensor.clone()
        if patched.ndim >= 3:
            patched[:, -1, :] = patched[:, -1, :] * float(scale)
        elif patched.ndim >= 2:
            patched[:, -1] = patched[:, -1] * float(scale)
        else:
            patched = patched * float(scale)
        return patched

    if torch.is_tensor(output):
        return patch_tensor(output)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (patch_tensor(output[0]), *output[1:])
    return output


def install_component_scale(model, layer_ids: list[int], component: str, scale: float) -> list[Any]:
    handles: list[Any] = []
    layers = get_layers(model)

    def make_hook():
        def hook(_module, _inputs, output):
            return patch_module_output(output, float(scale))

        return hook

    for layer_id in sorted(set(int(x) for x in layer_ids)):
        if not (0 <= layer_id < len(layers)):
            continue
        layer = layers[layer_id]
        if component == "mlp" and hasattr(layer, "mlp"):
            handles.append(layer.mlp.register_forward_hook(make_hook()))
        elif component == "attn" and hasattr(layer, "self_attn"):
            handles.append(layer.self_attn.register_forward_hook(make_hook()))
    return handles


def first_logits_with_pathway_controls(
    model,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    mode: str,
    scale_up_factor: float,
    component: str,
    component_scale: float,
    target_layers: list[int],
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles: list[Any] = []
    try:
        if component != "none":
            handles.extend(install_component_scale(model, target_layers, component, component_scale))
        if mode != "original" and gears:
            handles.extend(p862.install_scaled_gear_edit(model, gears, mode, scale_up_factor))
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
    finally:
        for handle in handles:
            handle.remove()


def control_spec(control_type: str) -> tuple[str, float]:
    mapping = {
        "none": ("none", 1.0),
        "mlp_zero": ("mlp", 0.0),
        "attn_zero": ("attn", 0.0),
        "mlp_half": ("mlp", 0.5),
        "attn_half": ("attn", 0.5),
    }
    if control_type not in mapping:
        raise ValueError(f"unknown control type: {control_type}")
    return mapping[control_type]


def make_result_row(
    model_name: str,
    source: dict[str, Any],
    gear_spec: dict[str, Any],
    edit_mode: str,
    control_type: str,
    base_metrics: dict[str, Any],
    metrics: dict[str, Any],
    none_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    base_class = base_metrics.get("class_best_logit")
    current_class = metrics.get("class_best_logit")
    none_class = none_metrics.get("class_best_logit") if none_metrics else None
    target_lift = None if base_class is None or current_class is None else finite(current_class) - finite(base_class)
    none_target_lift = None if base_class is None or none_class is None else finite(none_class) - finite(base_class)
    blocker_reduction = None
    if base_metrics.get("class_blocker_count") is not None and metrics.get("class_blocker_count") is not None:
        blocker_reduction = finite(base_metrics.get("class_blocker_count")) - finite(metrics.get("class_blocker_count"))
    none_boundary_closed = bool(none_metrics.get("class_boundary_closed")) if none_metrics else bool(metrics.get("class_boundary_closed"))
    return {
        "phase": PHASE,
        "row_kind": "phase891_target_lift_source_pathway_row",
        "model": model_name,
        "parent_candidate_key": source.get("parent_candidate_key"),
        "case_id": source.get("case_id"),
        "case_split": source.get("case_split"),
        "eval_domain": source.get("eval_domain"),
        "object": source.get("object"),
        "prompt_variant": source.get("prompt_variant"),
        "gear_set_type": gear_spec.get("gear_set_type"),
        "gear_keys": gear_spec.get("gear_keys"),
        "gear_count": gear_spec.get("gear_count"),
        "target_layers": gear_spec.get("target_layers"),
        "is_multi_axis": bool(gear_spec.get("is_multi_axis")),
        "edit_mode": edit_mode,
        "control_type": control_type,
        "base_boundary_closed": bool(base_metrics.get("class_boundary_closed")),
        "boundary_closed": bool(metrics.get("class_boundary_closed")),
        "none_boundary_closed": none_boundary_closed,
        "closure_from_open": bool((not base_metrics.get("class_boundary_closed")) and metrics.get("class_boundary_closed")),
        "none_closure_from_open": bool((not base_metrics.get("class_boundary_closed")) and none_boundary_closed),
        "closure_lost_vs_none": bool(control_type != "none" and none_boundary_closed and not metrics.get("class_boundary_closed")),
        "base_class_logit": base_class,
        "class_logit": current_class,
        "target_lift": target_lift,
        "none_target_lift": none_target_lift,
        "target_lift_removed_vs_none": None if none_target_lift is None or target_lift is None else none_target_lift - target_lift,
        "target_lift_retention_vs_none": None
        if none_target_lift is None or abs(none_target_lift) < 1.0e-9 or target_lift is None
        else target_lift / none_target_lift,
        "base_blocker_count": base_metrics.get("class_blocker_count"),
        "blocker_count": metrics.get("class_blocker_count"),
        "blocker_reduction": blocker_reduction,
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    phase890_rows = load_phase890_rows(args.phase890_round, args.model)
    selected = select_sources(phase890_rows, args)
    model_gears = all_model_gears(phase890_rows, int(args.max_u_gears))
    modes = parse_csv(args.edit_modes)
    controls = parse_csv(args.control_types)
    if args.dry_run or not selected:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run" if selected else "no_sources",
            "selected_sources": selected,
            "model_gears": model_gears,
            "edit_modes": modes,
            "control_types": controls,
        }
        p846.write_json(out_dir / f"phase891_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase891_{args.model}_rows.jsonl", [])
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
        cache: dict[tuple[str, str], tuple[torch.Tensor, dict[str, Any], dict[str, Any], list[int]]] = {}
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
                base_logits = first_logits_with_pathway_controls(
                    model, device, prompt_ids, [], "original", float(args.scale_up_factor), "none", 1.0, []
                )
                base_metrics = p888.metrics_for_logits(tokenizer, base_logits, token_sets, int(args.topk_tokens))
                cache[cache_key] = (base_logits, base_metrics, token_sets, prompt_ids)
            _base_logits, base_metrics, token_sets, prompt_ids = cache[cache_key]
            for gear_spec in gear_set_specs(source, model_gears, int(args.max_u_gears)):
                for mode in modes:
                    none_metrics: dict[str, Any] | None = None
                    none_logits = first_logits_with_pathway_controls(
                        model,
                        device,
                        prompt_ids,
                        gear_spec["gears"],
                        mode,
                        float(args.scale_up_factor),
                        "none",
                        1.0,
                        gear_spec["target_layers"],
                    )
                    none_metrics = p888.metrics_for_logits(tokenizer, none_logits, token_sets, int(args.topk_tokens))
                    for control in controls:
                        component, scale = control_spec(control)
                        if control == "none":
                            metrics = none_metrics
                        else:
                            logits = first_logits_with_pathway_controls(
                                model,
                                device,
                                prompt_ids,
                                gear_spec["gears"],
                                mode,
                                float(args.scale_up_factor),
                                component,
                                scale,
                                gear_spec["target_layers"],
                            )
                            metrics = p888.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens))
                        rows.append(make_result_row(args.model, source, gear_spec, mode, control, base_metrics, metrics, none_metrics))
            log(f"{args.model}/{args.round_name}: source={idx}/{len(selected)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, rows, selected, model_gears, modes, controls, attn_impl)
    p846.write_json(out_dir / f"phase891_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase891_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def control_label(vals: list[dict[str, Any]]) -> str:
    none_rows = [row for row in vals if row.get("control_type") == "none" and row.get("none_closure_from_open")]
    if not none_rows:
        return "negative_no_target_lift_pathway"
    mlp_zero = [row for row in vals if row.get("control_type") == "mlp_zero" and row.get("none_closure_from_open")]
    attn_zero = [row for row in vals if row.get("control_type") == "attn_zero" and row.get("none_closure_from_open")]
    mlp_lost = sum(1 for row in mlp_zero if row.get("closure_lost_vs_none") or finite(row.get("target_lift_retention_vs_none"), 1.0) < 0.35)
    attn_lost = sum(1 for row in attn_zero if row.get("closure_lost_vs_none") or finite(row.get("target_lift_retention_vs_none"), 1.0) < 0.35)
    if mlp_lost and mlp_lost >= max(1, 2 * attn_lost):
        return "mlp_output_required_target_lift"
    if attn_lost and attn_lost >= max(1, 2 * mlp_lost):
        return "attn_output_required_target_lift"
    if mlp_lost and attn_lost:
        return "mixed_component_target_lift"
    return "target_lift_survives_component_ablation"


def summarize_model(
    model_name: str,
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    model_gears: list[dict[str, Any]],
    modes: list[str],
    controls: list[str],
    attn_impl: str | None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("parent_candidate_key"))].append(row)
    candidate_groups = []
    for key, vals in groups.items():
        none_rows = [row for row in vals if row.get("control_type") == "none"]
        none_closure = [row for row in none_rows if row.get("none_closure_from_open")]
        multi_none_closure = [row for row in none_closure if row.get("is_multi_axis")]
        mlp_zero = [row for row in vals if row.get("control_type") == "mlp_zero" and row.get("none_closure_from_open")]
        attn_zero = [row for row in vals if row.get("control_type") == "attn_zero" and row.get("none_closure_from_open")]
        candidate_groups.append(
            {
                "model": model_name,
                "parent_candidate_key": key,
                "evidence_label": control_label(vals),
                "n_rows": len(vals),
                "n_source_cases": len(set((str(row.get("case_id")), str(row.get("prompt_variant"))) for row in vals)),
                "none_closure_from_open": len(none_closure),
                "multi_axis_none_closure": len(multi_none_closure),
                "mean_none_target_lift": mean([finite(row.get("target_lift")) for row in none_closure]) or 0.0,
                "mean_none_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in none_closure]) or 0.0,
                "mlp_zero_closure_lost": sum(1 for row in mlp_zero if row.get("closure_lost_vs_none")),
                "attn_zero_closure_lost": sum(1 for row in attn_zero if row.get("closure_lost_vs_none")),
                "mean_mlp_zero_lift_retention": mean(
                    [
                        finite(row.get("target_lift_retention_vs_none"))
                        for row in mlp_zero
                        if row.get("target_lift_retention_vs_none") is not None
                    ]
                )
                or 0.0,
                "mean_attn_zero_lift_retention": mean(
                    [
                        finite(row.get("target_lift_retention_vs_none"))
                        for row in attn_zero
                        if row.get("target_lift_retention_vs_none") is not None
                    ]
                )
                or 0.0,
                "closure_modes": counter_values(Counter(str(row.get("edit_mode")) for row in none_closure)),
                "gear_set_types": counter_values(Counter(str(row.get("gear_set_type")) for row in none_closure)),
                "objects": sorted(set(str(row.get("object")) for row in vals)),
            }
        )
    candidate_groups.sort(
        key=lambda row: (
            row.get("none_closure_from_open") or 0,
            row.get("mlp_zero_closure_lost") or 0,
            row.get("multi_axis_none_closure") or 0,
        ),
        reverse=True,
    )
    none_rows = [row for row in rows if row.get("control_type") == "none"]
    mlp_zero = [row for row in rows if row.get("control_type") == "mlp_zero" and row.get("none_closure_from_open")]
    attn_zero = [row for row in rows if row.get("control_type") == "attn_zero" and row.get("none_closure_from_open")]
    return {
        "phase": PHASE,
        "title": "Target-Lift Source Pathway and True Projection Subspace Audit",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_sources": len(selected),
        "output_rows": len(rows),
        "model_gear_keys": [p862.gear_key(gear) for gear in model_gears],
        "edit_modes": modes,
        "control_types": controls,
        "overall": {
            "none_closure_from_open": sum(1 for row in none_rows if row.get("none_closure_from_open")),
            "multi_axis_none_closure": sum(1 for row in none_rows if row.get("none_closure_from_open") and row.get("is_multi_axis")),
            "mlp_zero_closure_lost": sum(1 for row in mlp_zero if row.get("closure_lost_vs_none")),
            "attn_zero_closure_lost": sum(1 for row in attn_zero if row.get("closure_lost_vs_none")),
            "mean_none_target_lift": mean([finite(row.get("target_lift")) for row in none_rows if row.get("none_closure_from_open")]) or 0.0,
            "mean_mlp_zero_lift_retention": mean(
                [
                    finite(row.get("target_lift_retention_vs_none"))
                    for row in mlp_zero
                    if row.get("target_lift_retention_vs_none") is not None
                ]
            )
            or 0.0,
            "mean_attn_zero_lift_retention": mean(
                [
                    finite(row.get("target_lift_retention_vs_none"))
                    for row in attn_zero
                    if row.get("target_lift_retention_vs_none") is not None
                ]
            )
            or 0.0,
        },
        "candidate_groups": candidate_groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in candidate_groups)),
        "boundary": (
            "Phase891 audits target-lift source using broad component scaling. "
            "MLP/attention output ablation is coarse and should not be read as neuron-level closure."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 891 target-lift source pathway and projection subspace audit",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
        f"- selected_sources: {payload.get('selected_sources')}",
        f"- output_rows: {payload.get('output_rows')}",
        f"- none_closure_from_open: {payload.get('overall', {}).get('none_closure_from_open')}",
        f"- multi_axis_none_closure: {payload.get('overall', {}).get('multi_axis_none_closure')}",
        f"- mlp_zero_closure_lost: {payload.get('overall', {}).get('mlp_zero_closure_lost')}",
        f"- attn_zero_closure_lost: {payload.get('overall', {}).get('attn_zero_closure_lost')}",
        f"- mean_mlp_zero_lift_retention: {finite(payload.get('overall', {}).get('mean_mlp_zero_lift_retention')):.3f}",
        f"- mean_attn_zero_lift_retention: {finite(payload.get('overall', {}).get('mean_attn_zero_lift_retention')):.3f}",
        "",
        "## Candidate groups",
        "",
        "| model | candidate | label | none closure | multi-axis closure | mlp lost | attn lost | mlp retention | attn retention | gear sets |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("candidate_groups", [])[:30]:
        lines.append(
            "| {model} | {key} | {label} | {closure} | {multi} | {mlp_lost} | {attn_lost} | {mlp_ret:.3f} | {attn_ret:.3f} | {sets} |".format(
                model=row.get("model"),
                key=row.get("parent_candidate_key"),
                label=row.get("evidence_label"),
                closure=row.get("none_closure_from_open"),
                multi=row.get("multi_axis_none_closure"),
                mlp_lost=row.get("mlp_zero_closure_lost"),
                attn_lost=row.get("attn_zero_closure_lost"),
                mlp_ret=finite(row.get("mean_mlp_zero_lift_retention")),
                attn_ret=finite(row.get("mean_attn_zero_lift_retention")),
                sets=json.dumps(row.get("gear_set_types") or {}, ensure_ascii=False),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [p846.read_json(out_dir / f"phase891_{model}_summary.json") for model in MODELS if (out_dir / f"phase891_{model}_summary.json").exists()]
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
            row.get("none_closure_from_open") or 0,
            row.get("mlp_zero_closure_lost") or 0,
            row.get("multi_axis_none_closure") or 0,
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
    p846.write_json(out_dir / "phase891_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase891_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="target_lift_source_projection")
    parser.add_argument("--phase890-round", default="distributed_restore_projection")
    parser.add_argument("--edit-modes", default="zero,flip,half")
    parser.add_argument("--control-types", default="none,mlp_zero,attn_zero,mlp_half,attn_half")
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--max-closure-cases-per-candidate", type=int, default=8)
    parser.add_argument("--max-control-cases-per-candidate", type=int, default=2)
    parser.add_argument("--max-u-gears", type=int, default=8)
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
