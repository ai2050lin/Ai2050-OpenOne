#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE = 851
MODELS = ("qwen3", "glm4", "deepseek7b")
PREDICTORS = (
    "global_combo",
    "internal_strength_combo",
    "residual_projection_combo",
    "blocker_field_combo",
    "route_competition_combo",
    "compact_joint_gate_combo",
    "joint_gate_combo",
    "model_default_gate",
    "train_selected_gate",
)
CORE_PREDICTORS = (
    "global_combo",
    "residual_projection_combo",
    "blocker_field_combo",
    "model_default_gate",
    "train_selected_gate",
)
NUMERIC_FEATURES = (
    "actual_residual",
    "actual_delta",
    "expected_additive_delta",
    "abs_mean",
    "abs_sum",
    "signed_mean",
    "signed_sum",
    "max_abs",
    "min_abs",
    "pos_count",
    "neg_count",
    "zero_count",
    "blocker_pressure",
    "route_gap",
    "topk_entropy",
    "original_margin",
    "target_minus_blocker_logit",
    "target_minus_object_logit",
    "object_minus_blocker_logit",
    "object_echo_pressure",
    "target_blocker_resid_final",
    "object_blocker_resid_final",
    "best_target_blocker_resid_final",
    "best_target_object_resid_final",
    "resid_target_blocker_span",
    "resid_polygon_blocker_span",
    "object_rank",
    "blocker_rank",
    "best_target_rank",
)
RESULT_ROOT = Path("tests/result/phase851_global_atlas_schema_orthogonality_audit")
PHASE849_ROOT = Path("tests/result/phase849_residual_blocker_route_gate_expansion")
PHASE850_ROOT = Path("tests/result/phase850_strong_edge_balanced_route_gate_validation")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(out):
        return out
    return default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def actual_class(row: dict[str, Any], threshold: float) -> str:
    value = finite(row.get("actual_residual"))
    if value >= threshold:
        return "synergy"
    if value <= -threshold:
        return "antagonistic"
    return "additive"


def is_strong_label(label: str) -> bool:
    return label in {"synergy", "antagonistic"}


def is_strong_row(row: dict[str, Any], threshold: float) -> bool:
    return is_strong_label(actual_class(row, threshold))


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def rate_table(rows: list[dict[str, Any]], group_key: str, threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    out: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        strong = sum(1 for row in group_rows if is_strong_row(row, threshold))
        out.append(
            {
                "group": key,
                "n": len(group_rows),
                "strong": strong,
                "strong_rate": strong / len(group_rows) if group_rows else None,
                "class_counts": counter_dict(Counter(actual_class(row, threshold) for row in group_rows)),
            }
        )
    return out


def rate_range(table: list[dict[str, Any]]) -> float | None:
    values = [finite(row.get("strong_rate"), float("nan")) for row in table if row.get("strong_rate") is not None]
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return None
    return max(values) - min(values)


def eta_squared(rows: list[dict[str, Any]], feature: str, group_key: str) -> float | None:
    values: list[tuple[str, float]] = []
    for row in rows:
        if row.get(feature) is None:
            continue
        values.append((str(row.get(group_key, "")), finite(row.get(feature))))
    if len(values) < 3:
        return None
    all_values = [v for _, v in values]
    grand_mean = sum(all_values) / len(all_values)
    total_ss = sum((v - grand_mean) ** 2 for v in all_values)
    if total_ss <= 1e-12:
        return None
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in values:
        grouped[group].append(value)
    between_ss = 0.0
    for group_values in grouped.values():
        group_mean = sum(group_values) / len(group_values)
        between_ss += len(group_values) * (group_mean - grand_mean) ** 2
    return between_ss / total_ss


def classify_axis(object_eta: float | None, prompt_eta: float | None) -> str:
    obj = object_eta or 0.0
    prompt = prompt_eta or 0.0
    if max(obj, prompt) < 0.05:
        return "low_axis_signal"
    if prompt >= obj * 1.5 and prompt >= 0.05:
        return "protocol_like"
    if obj >= prompt * 1.5 and obj >= 0.05:
        return "semantic_like"
    return "entangled_or_shared"


def orthogonality_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        object_eta = eta_squared(rows, feature, "object")
        prompt_eta = eta_squared(rows, feature, "prompt_variant")
        if object_eta is None and prompt_eta is None:
            continue
        out.append(
            {
                "feature": feature,
                "object_eta": object_eta,
                "prompt_eta": prompt_eta,
                "axis_class": classify_axis(object_eta, prompt_eta),
                "mean": mean([finite(row.get(feature)) for row in rows if row.get(feature) is not None]),
            }
        )
    return sorted(out, key=lambda row: max(row.get("object_eta") or 0.0, row.get("prompt_eta") or 0.0), reverse=True)


def gear_candidates(rows: list[dict[str, Any]], threshold: float, min_total: int) -> list[dict[str, Any]]:
    total = Counter()
    strong = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        label = actual_class(row, threshold)
        for gear in row.get("gear_keys") or []:
            gear_key = str(gear)
            total[gear_key] += 1
            class_counts[gear_key][label] += 1
            if is_strong_label(label):
                strong[gear_key] += 1
    baseline_rate = sum(strong.values()) / sum(total.values()) if total else 0.0
    candidates: list[dict[str, Any]] = []
    for gear, n in total.items():
        strong_count = strong[gear]
        strong_rate = strong_count / n if n else 0.0
        lift = strong_rate / baseline_rate if baseline_rate > 0 else None
        status = "low_support"
        if n >= min_total and strong_count >= 2 and lift is not None and lift >= 1.5:
            status = "counterfactual_min_cut_candidate"
        elif n >= min_total and strong_count:
            status = "weak_candidate"
        candidates.append(
            {
                "gear": gear,
                "total": int(n),
                "strong": int(strong_count),
                "strong_rate": strong_rate,
                "lift_vs_baseline": lift,
                "class_counts": counter_dict(class_counts[gear]),
                "audit_status": status,
            }
        )
    return sorted(candidates, key=lambda row: (row["strong"], row["lift_vs_baseline"] or 0.0, row["total"]), reverse=True)


def combo_candidates(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    total = Counter()
    strong = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("combo_key"))
        label = actual_class(row, threshold)
        total[key] += 1
        class_counts[key][label] += 1
        if is_strong_label(label):
            strong[key] += 1
        meta.setdefault(
            key,
            {
                "combo_type": row.get("combo_type"),
                "gear_count": row.get("gear_count"),
                "gear_keys": row.get("gear_keys"),
            },
        )
    out: list[dict[str, Any]] = []
    for key, n in total.items():
        strong_count = strong[key]
        out.append(
            {
                "combo_key": key,
                "total": int(n),
                "strong": int(strong_count),
                "strong_rate": strong_count / n if n else 0.0,
                "class_counts": counter_dict(class_counts[key]),
                **meta.get(key, {}),
            }
        )
    return sorted(out, key=lambda row: (row["strong"], row["strong_rate"], row["total"]), reverse=True)


def metric(summary: dict[str, Any], split: str, predictor: str, mode: str) -> dict[str, Any] | None:
    split_row = (summary.get("split_summary") or {}).get(split) or {}
    source = split_row.get(f"{mode}_summary") or {}
    return source.get(predictor)


def f1(summary: dict[str, Any], split: str, predictor: str, mode: str) -> float | None:
    stats = metric(summary, split, predictor, mode)
    if not stats:
        return None
    return (stats.get("strong") or {}).get("f1")


def recall(summary: dict[str, Any], split: str, predictor: str, mode: str) -> float | None:
    stats = metric(summary, split, predictor, mode)
    if not stats:
        return None
    return (stats.get("strong") or {}).get("recall")


def gate_evidence_level(summary: dict[str, Any], predictor: str) -> str:
    strong_rows = int((summary.get("feature_summary") or {}).get("strong_rows") or 0)
    if strong_rows <= 0:
        return "L0_untriggered"
    raw_obj = f1(summary, "object_holdout", predictor, "raw")
    raw_prompt = f1(summary, "prompt_holdout", predictor, "raw")
    bal_obj = f1(summary, "object_holdout", predictor, "balanced")
    bal_prompt = f1(summary, "prompt_holdout", predictor, "balanced")
    global_raw_obj = f1(summary, "object_holdout", "global_combo", "raw")
    global_raw_prompt = f1(summary, "prompt_holdout", "global_combo", "raw")
    global_bal_obj = f1(summary, "object_holdout", "global_combo", "balanced")
    global_bal_prompt = f1(summary, "prompt_holdout", "global_combo", "balanced")
    in_f1 = f1(summary, "in_sample", predictor, "raw") or 0.0
    raw_pass = (
        raw_obj is not None
        and raw_prompt is not None
        and global_raw_obj is not None
        and global_raw_prompt is not None
        and raw_obj > global_raw_obj
        and raw_prompt > global_raw_prompt
    )
    balanced_pass = (
        bal_obj is not None
        and bal_prompt is not None
        and global_bal_obj is not None
        and global_bal_prompt is not None
        and bal_obj > global_bal_obj
        and bal_prompt > global_bal_prompt
    )
    if strong_rows >= 10 and raw_pass and balanced_pass:
        return "L5_strong_edge_holdout_candidate"
    if raw_pass or balanced_pass:
        return "L4_partial_holdout_candidate"
    if in_f1 >= 0.5:
        return "L3_in_sample_only"
    if strong_rows > 0:
        return "L2_weak_strong_edge_signal"
    return "L0_untriggered"


def gate_evidence(summary: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for predictor in PREDICTORS:
        row = {
            "predictor": predictor,
            "evidence_level": gate_evidence_level(summary, predictor),
            "in_sample_raw_f1": f1(summary, "in_sample", predictor, "raw"),
            "object_holdout_raw_f1": f1(summary, "object_holdout", predictor, "raw"),
            "prompt_holdout_raw_f1": f1(summary, "prompt_holdout", predictor, "raw"),
            "object_holdout_balanced_f1": f1(summary, "object_holdout", predictor, "balanced"),
            "prompt_holdout_balanced_f1": f1(summary, "prompt_holdout", predictor, "balanced"),
            "object_holdout_raw_recall": recall(summary, "object_holdout", predictor, "raw"),
            "prompt_holdout_raw_recall": recall(summary, "prompt_holdout", predictor, "raw"),
        }
        out.append(row)
    return out


def hierarchy_readiness(rows: list[dict[str, Any]], summary: dict[str, Any], threshold: float) -> dict[str, Any]:
    object_rates = rate_table(rows, "object", threshold)
    prompt_rates = rate_table(rows, "prompt_variant", threshold)
    sign_rates = rate_table(rows, "sign_pattern", threshold)
    combo_rates = rate_table(rows, "combo_type", threshold)
    predictor_levels = gate_evidence(summary)
    best_holdout = max(
        (
            row
            for row in predictor_levels
            if row.get("object_holdout_raw_f1") is not None and row.get("prompt_holdout_raw_f1") is not None
        ),
        key=lambda row: min(row.get("object_holdout_raw_f1") or 0.0, row.get("prompt_holdout_raw_f1") or 0.0),
        default=None,
    )
    return {
        "macro_prompt_strong_rate_range": rate_range(prompt_rates),
        "semantic_object_strong_rate_range": rate_range(object_rates),
        "meso_sign_pattern_strong_rate_range": rate_range(sign_rates),
        "meso_combo_type_strong_rate_range": rate_range(combo_rates),
        "best_micro_gate_by_min_holdout_f1": best_holdout,
        "macro_groups": prompt_rates,
        "semantic_groups": object_rates,
        "meso_sign_groups": sign_rates,
        "meso_combo_groups": combo_rates,
    }


def atlas_nodes(model: str, summary: dict[str, Any], gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    nodes.append(
        {
            "node_id": f"{model}:state:residual_stream",
            "node_type": "state",
            "evidence": "observed_from_phase849_feature_rows",
        }
    )
    nodes.append(
        {
            "node_id": f"{model}:boundary:strong_edge",
            "node_type": "boundary",
            "strong_rows": (summary.get("feature_summary") or {}).get("strong_rows"),
        }
    )
    for gate in gates:
        level = str(gate.get("evidence_level"))
        if level.startswith(("L5", "L4", "L3")):
            nodes.append(
                {
                    "node_id": f"{model}:gate:{gate['predictor']}",
                    "node_type": "gate",
                    "predictor": gate["predictor"],
                    "evidence_level": level,
                    "object_holdout_raw_f1": gate.get("object_holdout_raw_f1"),
                    "prompt_holdout_raw_f1": gate.get("prompt_holdout_raw_f1"),
                }
            )
    return nodes


def atlas_edges(model: str, gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for gate in gates:
        level = str(gate.get("evidence_level"))
        if level.startswith(("L5", "L4", "L3")):
            edges.append(
                {
                    "source": f"{model}:gate:{gate['predictor']}",
                    "target": f"{model}:boundary:strong_edge",
                    "edge_type": "interaction_prediction",
                    "evidence_level": level,
                    "object_holdout_raw_f1": gate.get("object_holdout_raw_f1"),
                    "prompt_holdout_raw_f1": gate.get("prompt_holdout_raw_f1"),
                }
            )
    return edges


def audit_model(model: str, phase849_round: str, phase850_round: str, threshold: float, min_gear_total: int) -> dict[str, Any]:
    rows_path = PHASE849_ROOT / phase849_round / f"phase849_{model}_feature_rows.jsonl"
    summary_path = PHASE850_ROOT / phase850_round / f"phase850_{model}_summary.json"
    rows = read_jsonl(rows_path)
    summary = read_json(summary_path)
    gates = gate_evidence(summary)
    orth = orthogonality_audit(rows)
    payload = {
        "phase": PHASE,
        "title": "Global Atlas Schema and Orthogonality Audit",
        "model": model,
        "phase849_round": phase849_round,
        "phase850_round": phase850_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_rows": len(rows),
        "feature_summary": summary.get("feature_summary"),
        "gate_evidence": gates,
        "hierarchical_gating_readiness": hierarchy_readiness(rows, summary, threshold),
        "orthogonality_audit": orth,
        "top_protocol_like_features": [row for row in orth if row["axis_class"] == "protocol_like"][:10],
        "top_semantic_like_features": [row for row in orth if row["axis_class"] == "semantic_like"][:10],
        "top_entangled_features": [row for row in orth if row["axis_class"] == "entangled_or_shared"][:10],
        "gear_min_cut_candidates": gear_candidates(rows, threshold, min_gear_total)[:30],
        "combo_strong_edge_candidates": combo_candidates(rows, threshold)[:30],
        "atlas_nodes": atlas_nodes(model, summary, gates),
        "atlas_edges": atlas_edges(model, gates),
        "boundary": (
            "This phase is an atlas-schema audit over existing Phase 849/850 rows. "
            "It does not run new model forward passes and does not prove closure."
        ),
    }
    return payload


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 851 Global Atlas Schema and Orthogonality Audit ({payload['round']})",
        "",
        "- Source: Phase 849 feature rows and Phase 850 strong-edge summaries.",
        "- Boundary: schema audit and candidate ranking, not new forward-pass mechanism discovery.",
        "",
        "## Gate Evidence",
        "",
        "| model | rows | strong rows | predictor | evidence | in F1 | object F1 | prompt F1 | object balanced F1 | prompt balanced F1 |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, data in payload.get("model_audits", {}).items():
        fs = data.get("feature_summary") or {}
        for row in data.get("gate_evidence") or []:
            if row.get("predictor") not in CORE_PREDICTORS:
                continue
            lines.append(
                f"| {model} | {data.get('source_rows', 0)} | {fs.get('strong_rows', 0)} | "
                f"`{row.get('predictor')}` | `{row.get('evidence_level')}` | "
                f"{fmt(row.get('in_sample_raw_f1'))} | {fmt(row.get('object_holdout_raw_f1'))} | "
                f"{fmt(row.get('prompt_holdout_raw_f1'))} | {fmt(row.get('object_holdout_balanced_f1'))} | "
                f"{fmt(row.get('prompt_holdout_balanced_f1'))} |"
            )
    lines += [
        "",
        "## Orthogonality Audit",
        "",
        "| model | class | feature | object eta | prompt eta | mean |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model, data in payload.get("model_audits", {}).items():
        picks = (data.get("top_protocol_like_features") or [])[:5]
        picks += (data.get("top_semantic_like_features") or [])[:5]
        picks += (data.get("top_entangled_features") or [])[:5]
        for row in picks:
            lines.append(
                f"| {model} | `{row.get('axis_class')}` | `{row.get('feature')}` | "
                f"{fmt(row.get('object_eta'))} | {fmt(row.get('prompt_eta'))} | {fmt(row.get('mean'))} |"
            )
    lines += [
        "",
        "## Counterfactual Min-Cut Pre-Candidates",
        "",
        "| model | gear | total | strong | strong rate | lift | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for model, data in payload.get("model_audits", {}).items():
        for row in (data.get("gear_min_cut_candidates") or [])[:10]:
            lines.append(
                f"| {model} | `{row.get('gear')}` | {row.get('total')} | {row.get('strong')} | "
                f"{fmt(row.get('strong_rate'))} | {fmt(row.get('lift_vs_baseline'))} | `{row.get('audit_status')}` |"
            )
    lines += [
        "",
        "## Atlas Edges",
        "",
        "| model | source | target | type | evidence | object F1 | prompt F1 |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for model, data in payload.get("model_audits", {}).items():
        for row in data.get("atlas_edges") or []:
            lines.append(
                f"| {model} | `{row.get('source')}` | `{row.get('target')}` | `{row.get('edge_type')}` | "
                f"`{row.get('evidence_level')}` | {fmt(row.get('object_holdout_raw_f1'))} | "
                f"{fmt(row.get('prompt_holdout_raw_f1'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_audits: dict[str, Any] = {}
    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    for model in MODELS:
        log(f"phase851 audit start model={model} round={args.round_name}")
        audit = audit_model(model, args.phase849_round, args.phase850_round, args.interaction_threshold, args.min_gear_total)
        model_audits[model] = audit
        all_nodes.extend(audit["atlas_nodes"])
        all_edges.extend(audit["atlas_edges"])
        write_json(out_dir / f"phase851_{model}_atlas_audit.json", audit)
        write_jsonl(out_dir / f"phase851_{model}_atlas_nodes.jsonl", audit["atlas_nodes"])
        write_jsonl(out_dir / f"phase851_{model}_atlas_edges.jsonl", audit["atlas_edges"])
        log(f"phase851 audit done model={model} nodes={len(audit['atlas_nodes'])} edges={len(audit['atlas_edges'])}")
    payload = {
        "phase": PHASE,
        "round": args.round_name,
        "phase849_round": args.phase849_round,
        "phase850_round": args.phase850_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete",
        "model_audits": model_audits,
        "atlas_nodes": all_nodes,
        "atlas_edges": all_edges,
        "next_phase": {
            "phase": 852,
            "title": "Strong-edge Expansion Forward Test",
            "reason": "GLM4 has no strong rows and DS7B has too few holdout-usable strong rows; new forward data is required before causal closure claims.",
        },
    }
    write_json(out_dir / "phase851_cross_model_atlas_audit.json", payload)
    write_jsonl(out_dir / "phase851_cross_model_atlas_nodes.jsonl", all_nodes)
    write_jsonl(out_dir / "phase851_cross_model_atlas_edges.jsonl", all_edges)
    write_markdown(out_dir / "phase851_cross_model_atlas_audit.md", payload)
    print(json.dumps({"status": "complete", "round": args.round_name, "nodes": len(all_nodes), "edges": len(all_edges)}, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default="confirm")
    parser.add_argument("--phase849-round", default="confirm")
    parser.add_argument("--phase850-round", default="confirm")
    parser.add_argument("--interaction-threshold", type=float, default=0.5)
    parser.add_argument("--min-gear-total", type=int, default=10)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
