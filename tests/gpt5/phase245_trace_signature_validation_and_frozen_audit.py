#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PHASE = 245
SOURCE_PHASE = 244
SCHEMA_VERSION = "1.0.0"
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE244_DIR = Path("tests/result/phase244_first_internal_trace_batch/first_internal_trace_batch")
RESULT_ROOT = Path("tests/result/phase245_trace_signature_validation_and_frozen_audit")
ROUND_DEFAULT = "trace_signature_validation_and_frozen_audit"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_unique_jsonl(path: Path, new_rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + new_rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / denom


def sigmoid01(value: float, scale: float = 5.0) -> float:
    return 1.0 / (1.0 + math.exp(-value / max(scale, 1e-6)))


def load_phase244() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    component = read_jsonl(PHASE244_DIR / "phase244_cross_model_component_trace_rows.jsonl")
    residual = read_jsonl(PHASE244_DIR / "phase244_cross_model_residual_trace_rows.jsonl")
    readout = read_jsonl(PHASE244_DIR / "phase244_cross_model_readout_trace_rows.jsonl")
    rollout = read_jsonl(PHASE244_DIR / "phase244_cross_model_stepwise_rollout_rows.jsonl")
    summary = read_json(PHASE244_DIR / "phase244_cross_model_summary.json")
    if not component or not residual or not readout:
        raise FileNotFoundError(f"missing Phase244 trace rows under {PHASE244_DIR}")
    return component, residual, readout, rollout, summary


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def mean_by_component(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    by_component: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_component[str(row.get("component"))].append(safe_float(row.get("relative_delta_vs_full")))
    for component, values in by_component.items():
        out[f"{component}_relative_delta"] = round(mean(values), 6)
    return out


def build_signature_rows(
    component_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    readout_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comp_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    residual_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rollout_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        comp_by_key[row_key(row)].append(row)
    for row in residual_rows:
        residual_by_key[row_key(row)].append(row)
    for row in rollout_rows:
        rollout_by_key[row_key(row)].append(row)

    now = utc_now()
    out = []
    for readout in readout_rows:
        key = row_key(readout)
        comp_rows = comp_by_key.get(key, [])
        res_rows = residual_by_key.get(key, [])
        step_rows = sorted(rollout_by_key.get(key, []), key=lambda x: int(x.get("step_index") or 0))
        comp_vals = [safe_float(x.get("relative_delta_vs_full")) for x in comp_rows]
        product_vals = [
            safe_float(x.get("relative_delta_vs_full"))
            for x in comp_rows
            if str(x.get("component")) in {"product", "down_out", "recomputed_product"}
        ]
        residual_vals = [safe_float(x.get("relative_delta_vs_full")) for x in res_rows]
        margin_delta = safe_float(readout.get("target_margin_delta_vs_full"))
        component_mean = mean(comp_vals) if comp_vals else 0.0
        product_mean = mean(product_vals) if product_vals else 0.0
        residual_mean = mean(residual_vals) if residual_vals else 0.0
        first_winner = str(step_rows[0].get("winning_regime")) if step_rows else str(readout.get("winning_regime"))
        winner_sequence = [str(x.get("winning_regime")) for x in step_rows]
        token_sequence = [str(x.get("generated_token")) for x in step_rows]
        if component_mean >= 0.55 and margin_delta >= 8.0:
            signature_class = "high_component_high_readout"
        elif component_mean >= 0.55 and margin_delta < 4.0:
            signature_class = "high_component_low_readout"
        elif component_mean < 0.35 and margin_delta >= 8.0:
            signature_class = "low_component_high_readout"
        elif abs(margin_delta) < 1.0:
            signature_class = "readout_boundary_weak_change"
        else:
            signature_class = "mixed_signature"
        continuation_hits = sum(1 for x in winner_sequence if x in {"the_continuation", "be_continuation", "for_continuation"})
        closure_hits = sum(1 for x in winner_sequence if x in {"period_stop", "newline_boundary", "answer_boundary"})
        rollout_drift_score = continuation_hits / max(1, len(winner_sequence))
        closure_proxy_score = closure_hits / max(1, len(winner_sequence))
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase245",
                "source_phase": "Phase244",
                "created_at": now,
                "signature_id": f"phase245:{key[0]}:{key[1]}:{key[2]}",
                "model": key[0],
                "case_id": key[1],
                "variant_id": key[2],
                "family_id": readout.get("family_id"),
                "mode_id": readout.get("mode_id"),
                "recommended_next_test": readout.get("recommended_next_test"),
                "data_split": readout.get("data_split"),
                "cluster_key": readout.get("cluster_key"),
                "stable_winner_regime": readout.get("stable_winner_regime"),
                "winning_regime": readout.get("winning_regime"),
                "second_competitor": readout.get("second_competitor"),
                "component_mean_delta": round(component_mean, 6),
                "product_down_mean_delta": round(product_mean, 6),
                "residual_mean_delta": round(residual_mean, 6),
                "readout_margin_delta_vs_full": round(margin_delta, 6),
                "target_rank": readout.get("target_rank"),
                "stable_winner_match": bool(readout.get("stable_winner_match")),
                "winner_changed_vs_full": bool(readout.get("winner_changed_vs_full")),
                "signature_class": signature_class,
                "component_details": mean_by_component(comp_rows),
                "winner_sequence": winner_sequence,
                "generated_token_sequence": token_sequence,
                "first_step_winner": first_winner,
                "rollout_drift_score": round(rollout_drift_score, 6),
                "closure_proxy_score": round(closure_proxy_score, 6),
            }
        )
    out.sort(key=lambda x: (x["model"], x["recommended_next_test"], -abs(safe_float(x["readout_margin_delta_vs_full"]))))
    return out


def group_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field)) for field in fields)


def correlation_rows(signature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    fields_list = [
        ("model",),
        ("model", "recommended_next_test"),
        ("model", "family_id"),
        ("model", "family_id", "recommended_next_test"),
        ("model", "winning_regime"),
        ("model", "signature_class"),
    ]
    rows = []
    for fields in fields_list:
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in signature_rows:
            groups[group_key(row, fields)].append(row)
        for key, items in groups.items():
            if len(items) < 3:
                continue
            comp = [safe_float(x.get("component_mean_delta")) for x in items]
            product = [safe_float(x.get("product_down_mean_delta")) for x in items]
            residual = [safe_float(x.get("residual_mean_delta")) for x in items]
            margin = [safe_float(x.get("readout_margin_delta_vs_full")) for x in items]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase245",
                    "created_at": now,
                    "correlation_id": f"phase245:corr:{':'.join(fields)}:{':'.join(key)}",
                    "scope_fields": list(fields),
                    "scope_values": list(key),
                    "row_count": len(items),
                    "component_readout_corr": round(pearson(comp, margin), 6),
                    "product_readout_corr": round(pearson(product, margin), 6),
                    "residual_readout_corr": round(pearson(residual, margin), 6),
                    "mean_component_delta": round(mean(comp), 6),
                    "mean_product_down_delta": round(mean(product), 6),
                    "mean_residual_delta": round(mean(residual), 6),
                    "mean_readout_margin_delta": round(mean(margin), 6),
                }
            )
    rows.sort(key=lambda x: (x["scope_fields"], x["scope_values"]))
    return rows


def split_signature(values: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out = {}
    for split in ["explore", "validate", "frozen"]:
        items = [x for x in values if str(x.get("data_split")) == split]
        if not items:
            out[split] = {"rows": 0, "component": 0.0, "readout": 0.0, "residual": 0.0}
            continue
        out[split] = {
            "rows": len(items),
            "component": round(mean(safe_float(x.get("component_mean_delta")) for x in items), 6),
            "readout": round(mean(safe_float(x.get("readout_margin_delta_vs_full")) for x in items), 6),
            "residual": round(mean(safe_float(x.get("residual_mean_delta")) for x in items), 6),
        }
    return out


def stability_score(split_stats: dict[str, dict[str, float]]) -> float:
    if not split_stats["validate"]["rows"] or not split_stats["frozen"]["rows"]:
        return 0.0
    gaps = []
    for metric in ["component", "readout", "residual"]:
        gaps.append(abs(split_stats["validate"][metric] - split_stats["frozen"][metric]))
    return round(1.0 / (1.0 + mean(gaps)), 6)


def validate_frozen_audit_rows(signature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    fields_list = [
        ("model", "recommended_next_test"),
        ("model", "family_id"),
        ("model", "signature_class"),
        ("model", "winning_regime"),
        ("family_id", "recommended_next_test"),
    ]
    rows = []
    for fields in fields_list:
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in signature_rows:
            groups[group_key(row, fields)].append(row)
        for key, items in groups.items():
            split_stats = split_signature(items)
            if split_stats["validate"]["rows"] == 0 and split_stats["frozen"]["rows"] == 0:
                continue
            score = stability_score(split_stats)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase245",
                    "created_at": now,
                    "audit_id": f"phase245:audit:{':'.join(fields)}:{':'.join(key)}",
                    "scope_fields": list(fields),
                    "scope_values": list(key),
                    "row_count": len(items),
                    "split_stats": split_stats,
                    "validate_frozen_stability": score,
                    "audit_status": "stable_candidate" if score >= 0.35 and split_stats["frozen"]["rows"] >= 2 else "needs_more_frozen_rows",
                }
            )
    rows.sort(key=lambda x: safe_float(x.get("validate_frozen_stability")), reverse=True)
    return rows


def factor_projection_rows(signature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in signature_rows:
        winner = str(row.get("winning_regime"))
        second = str(row.get("second_competitor"))
        target_score = sigmoid01(safe_float(row.get("readout_margin_delta_vs_full")), 8.0)
        protocol_score = 1.0 if row.get("recommended_next_test") == "protocol_gate_product_residual_trace" else 0.35
        boundary_score = 1.0 if winner in {"answer_boundary", "newline_boundary", "period_stop"} else 0.25
        competitor_score = 1.0 if winner in {"the_continuation", "be_continuation", "for_continuation", "comma_repeat"} else 0.4
        closure_score = safe_float(row.get("closure_proxy_score"))
        rollout_score = safe_float(row.get("rollout_drift_score"))
        component_score = min(1.0, safe_float(row.get("product_down_mean_delta")))
        factors = {
            "target_proxy": round(target_score, 6),
            "protocol_proxy": round(protocol_score * component_score, 6),
            "boundary_proxy": round(boundary_score * component_score, 6),
            "competitor_proxy": round(competitor_score * (0.5 + 0.5 * component_score), 6),
            "closure_proxy": round(closure_score, 6),
            "rollout_drift_proxy": round(rollout_score, 6),
        }
        dominant = max(factors.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase245",
                "created_at": now,
                "projection_id": f"phase245:projection:{row['model']}:{row['case_id']}:{row['variant_id']}",
                "signature_id": row["signature_id"],
                "model": row["model"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "recommended_next_test": row["recommended_next_test"],
                "winning_regime": winner,
                "second_competitor": second,
                "factor_scores": factors,
                "dominant_proxy_factor": dominant,
                "projection_note": "proxy_only_no_raw_vector_orthogonalization",
            }
        )
    return rows


def causal_candidate_rows(signature_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    stable_scopes = set()
    for row in audit_rows:
        if row.get("audit_status") == "stable_candidate":
            stable_scopes.add(tuple(row.get("scope_values", [])))
    out = []
    for row in signature_rows:
        component = safe_float(row.get("component_mean_delta"))
        product = safe_float(row.get("product_down_mean_delta"))
        readout = safe_float(row.get("readout_margin_delta_vs_full"))
        residual = safe_float(row.get("residual_mean_delta"))
        decouple = abs(component - sigmoid01(readout, 8.0))
        score = 0.28 * min(1.0, product) + 0.24 * sigmoid01(readout, 8.0) + 0.18 * min(1.0, residual)
        score += 0.16 * (1.0 if row.get("data_split") in {"validate", "frozen"} else 0.65)
        score += 0.14 * min(1.0, decouple)
        if row.get("signature_class") in {"high_component_high_readout", "high_component_low_readout", "low_component_high_readout"}:
            score += 0.08
        reason = []
        if product >= 0.55:
            reason.append("high_product_down_delta")
        if readout >= 8.0:
            reason.append("high_readout_margin_delta")
        if row.get("signature_class") in {"high_component_low_readout", "low_component_high_readout"}:
            reason.append("component_readout_decoupling")
        if row.get("data_split") in {"validate", "frozen"}:
            reason.append("non_explore_split")
        if row.get("winner_changed_vs_full"):
            reason.append("winner_changed")
        if not reason:
            continue
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase245",
                "created_at": now,
                "candidate_id": f"phase245:causal:{row['model']}:{row['case_id']}:{row['variant_id']}",
                "signature_id": row["signature_id"],
                "model": row["model"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "recommended_next_test": row["recommended_next_test"],
                "data_split": row["data_split"],
                "signature_class": row["signature_class"],
                "component_mean_delta": row["component_mean_delta"],
                "product_down_mean_delta": row["product_down_mean_delta"],
                "residual_mean_delta": row["residual_mean_delta"],
                "readout_margin_delta_vs_full": row["readout_margin_delta_vs_full"],
                "candidate_score": round(score, 6),
                "selection_reasons": reason,
                "recommended_causal_test": "target_injection_vs_competitor_suppression"
                if row.get("signature_class") == "low_component_high_readout"
                else "component_ablation_and_readout_margin_test",
            }
        )
    out.sort(key=lambda x: safe_float(x.get("candidate_score")), reverse=True)
    return out[:30]


def observation_rows(signature_rows: list[dict[str, Any]], projection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in signature_rows:
        for metric_name in ["component_mean_delta", "product_down_mean_delta", "residual_mean_delta", "readout_margin_delta_vs_full"]:
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase245",
                    "created_at": now,
                    "observation_id": f"phase245:{row['model']}:{row['case_id']}:{row['variant_id']}:{metric_name}",
                    "case_id": row["case_id"],
                    "model": row["model"],
                    "family_id": row["family_id"],
                    "mode_id": row["mode_id"],
                    "variant_id": row["variant_id"],
                    "level": "trace_signature_validation",
                    "component": metric_name,
                    "metric_name": metric_name,
                    "metric_value": safe_float(row.get(metric_name)),
                    "metric_unit": "ratio_or_logit",
                    "recommended_next_test": row["recommended_next_test"],
                    "data_split": row["data_split"],
                    "signature_class": row["signature_class"],
                }
            )
    for row in projection_rows:
        for factor, value in row.get("factor_scores", {}).items():
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase245",
                    "created_at": now,
                    "observation_id": f"phase245:{row['model']}:{row['case_id']}:{row['variant_id']}:{factor}",
                    "case_id": row["case_id"],
                    "model": row["model"],
                    "family_id": row["family_id"],
                    "mode_id": row["mode_id"],
                    "variant_id": row["variant_id"],
                    "level": "proxy_factor_projection",
                    "component": factor,
                    "metric_name": factor,
                    "metric_value": safe_float(value),
                    "metric_unit": "proxy_score",
                    "recommended_next_test": row["recommended_next_test"],
                    "dominant_proxy_factor": row["dominant_proxy_factor"],
                }
            )
    return rows


def metric_rows(corr_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], signature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in corr_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase245",
                "created_at": now,
                "metric_id": row["correlation_id"],
                "scope": "component_readout_correlation",
                "metric_name": "component_readout_corr",
                "metric_value": row["component_readout_corr"],
                "scope_fields": row["scope_fields"],
                "scope_values": row["scope_values"],
                "rows": row["row_count"],
            }
        )
    for row in audit_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase245",
                "created_at": now,
                "metric_id": row["audit_id"],
                "scope": "validate_frozen_stability",
                "metric_name": "validate_frozen_stability",
                "metric_value": row["validate_frozen_stability"],
                "scope_fields": row["scope_fields"],
                "scope_values": row["scope_values"],
                "rows": row["row_count"],
                "audit_status": row["audit_status"],
            }
        )
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in signature_rows:
        by_class[str(row.get("signature_class"))].append(row)
    for cls, items in by_class.items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase245",
                "created_at": now,
                "metric_id": f"phase245:signature_class:{cls}",
                "scope": "signature_class",
                "metric_name": "signature_class_rate",
                "metric_value": round(len(items) / max(1, len(signature_rows)), 6),
                "signature_class": cls,
                "rows": len(items),
                "mean_component_delta": round(mean(safe_float(x.get("component_mean_delta")) for x in items), 6),
                "mean_readout_margin_delta": round(mean(safe_float(x.get("readout_margin_delta_vs_full")) for x in items), 6),
            }
        )
    return rows


def graph_edges(causal_rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    strongest = sorted(corr_rows, key=lambda x: abs(safe_float(x.get("component_readout_corr"))), reverse=True)[:20]
    for row in strongest:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase245",
                "created_at": now,
                "edge_id": f"phase245:corr:{row['correlation_id']}",
                "source": "node:ComponentDelta",
                "target": "node:ReadoutMarginDelta",
                "edge_type": "trace_signature_correlation",
                "evidence_type": "grouped_pearson_correlation",
                "effect_direction": "positive" if safe_float(row.get("component_readout_corr")) > 0 else "negative_or_decoupled",
                "effect_size": row["component_readout_corr"],
                "confidence": round(0.35 + min(0.25, abs(safe_float(row.get("component_readout_corr"))) / 4), 4),
                "supporting_phases": ["Phase244", "Phase245"],
                "scope_fields": row["scope_fields"],
                "scope_values": row["scope_values"],
                "status": "trace_validation_not_causal_closure",
            }
        )
    for row in causal_rows[:20]:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase245",
                "created_at": now,
                "edge_id": f"phase245:causal_candidate:{row['candidate_id']}",
                "source": f"signature:{row['signature_class']}",
                "target": f"causal_test:{row['recommended_causal_test']}",
                "edge_type": "causal_test_candidate",
                "evidence_type": "trace_signature_score",
                "effect_direction": "candidate",
                "effect_size": row["candidate_score"],
                "confidence": 0.48,
                "supporting_phases": ["Phase241", "Phase242", "Phase243", "Phase244", "Phase245"],
                "status": "candidate_not_validated_causally",
            }
        )
    return rows


def update_pattern_atlas(summary: dict[str, Any], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    ATLAS_ROOT.mkdir(parents=True, exist_ok=True)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    progress_path = ATLAS_ROOT / "progress.json"
    progress = read_json(progress_path)
    progress.update(
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
            "latest_phase": "Phase245",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "run focused causal tests only after trace signatures are stable",
            "small_model_bias_warning": "Phase245 is a no-new-forward audit over qwen3/glm4/deepseek7b traces; proxy factors are not raw vector orthogonalization.",
        }
    )
    write_json(progress_path, progress)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase245 trace signature validation and frozen audit",
        "",
        "## Core result",
        "",
        "Phase245 reuses Phase244 trace rows to classify trace signatures, audit validate/frozen stability, and select causal-test candidates.",
        "It does not run new model forwards and does not claim causal closure.",
        "",
        "## Counts",
        "",
        f"- signature_rows: {summary['signature_rows']}",
        f"- correlation_rows: {summary['correlation_rows']}",
        f"- validate_frozen_audit_rows: {summary['validate_frozen_audit_rows']}",
        f"- proxy_factor_projection_rows: {summary['proxy_factor_projection_rows']}",
        f"- causal_test_candidate_rows: {summary['causal_test_candidate_rows']}",
        "",
        "## Signature classes",
        "",
        "```json",
        json.dumps(summary["signature_class_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Correlation summary",
        "",
        f"- global_component_readout_corr_by_model: {json.dumps(summary['global_component_readout_corr_by_model'], ensure_ascii=False)}",
        f"- strongest_abs_corr: {summary['strongest_abs_component_readout_corr']}",
        "",
        "## Progress",
        "",
        "```json",
        json.dumps(summary["pattern_atlas_progress"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    component_rows, residual_rows, readout_rows, rollout_rows, phase244_summary = load_phase244()
    signature = build_signature_rows(component_rows, residual_rows, readout_rows, rollout_rows)
    corr = correlation_rows(signature)
    audit = validate_frozen_audit_rows(signature)
    projection = factor_projection_rows(signature)
    causal = causal_candidate_rows(signature, audit)
    observations = observation_rows(signature, projection)
    metrics = metric_rows(corr, audit, signature)
    edges = graph_edges(causal, corr)

    by_model_corr = {
        row["scope_values"][0]: row["component_readout_corr"]
        for row in corr
        if row.get("scope_fields") == ["model"] and row.get("scope_values")
    }
    strongest_abs = max([abs(safe_float(x.get("component_readout_corr"))) for x in corr], default=0.0)
    progress = {
        "pattern_family_atlas": 0.73,
        "candidate_clustering": 0.42,
        "case_bank_calibration": 0.38,
        "high_value_trace_selection": 0.58,
        "first_internal_trace_batch": 0.36,
        "trace_signature_validation": 0.32,
        "gate_up_product_signature": 0.42,
        "residual_state_signature": 0.40,
        "readout_competition_trace": 0.61,
        "stepwise_rollout_trace": 0.21,
        "proxy_factor_decomposition": 0.16,
        "causal_closure": 0.10,
        "general_language_mechanism_confidence": 0.54,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Trace signature validation and frozen audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "phase244_status": phase244_summary.get("status"),
        "signature_rows": len(signature),
        "correlation_rows": len(corr),
        "validate_frozen_audit_rows": len(audit),
        "proxy_factor_projection_rows": len(projection),
        "causal_test_candidate_rows": len(causal),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "signature_class_counts": dict(Counter(str(x.get("signature_class")) for x in signature).most_common()),
        "data_split_counts": dict(Counter(str(x.get("data_split")) for x in signature).most_common()),
        "global_component_readout_corr_by_model": by_model_corr,
        "strongest_abs_component_readout_corr": round(strongest_abs, 6),
        "top_causal_candidate_ids": [x["candidate_id"] for x in causal[:10]],
        "pattern_atlas_progress": progress,
        "judgement": "trace_signature_audit_not_causal_closure",
        "limitations": [
            "No new model forwards were run in Phase245.",
            "Factor projection rows are proxy scores from trace/readout/rollout metadata, not true raw-vector orthogonal projections.",
            "Validate/frozen audit is limited by Phase243 selection balance and should guide, not replace, future causal tests.",
        ],
    }
    write_json(out_dir / "phase245_summary.json", payload)
    write_jsonl(out_dir / "phase245_trace_signature_rows.jsonl", signature)
    write_jsonl(out_dir / "phase245_component_readout_correlation_rows.jsonl", corr)
    write_jsonl(out_dir / "phase245_validate_frozen_audit_rows.jsonl", audit)
    write_jsonl(out_dir / "phase245_factor_projection_rows.jsonl", projection)
    write_jsonl(out_dir / "phase245_causal_test_candidate_rows.jsonl", causal)
    write_jsonl(out_dir / "phase245_observations.jsonl", observations)
    write_jsonl(out_dir / "phase245_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase245_graph_edges.jsonl", edges)
    write_report(out_dir / "phase245_trace_signature_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase245 trace signature validation and frozen audit")
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.round_name)


if __name__ == "__main__":
    main()
