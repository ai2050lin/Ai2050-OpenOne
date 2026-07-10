#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
PHASE = "Phase299"
SCHEMA_VERSION = "2.26.0"


def now() -> str:
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def grouped(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[(str(row.get("model")), str(row.get("family_id")))].append(row)
    return out


def main() -> None:
    prev_rows = read_jsonl(V2 / "phase297_feature_matrix_v4_rows.jsonl")
    audit_rows = read_jsonl(V2 / "phase298_mlp_causal_audit_rows.jsonl")
    summary_298 = read_json(V2 / "phase298_cross_model_summary.json")
    audit_by_cell = grouped(audit_rows)
    rows: list[dict[str, Any]] = []
    for prev in prev_rows:
        key = (str(prev.get("model")), str(prev.get("family_id")))
        audits = audit_by_cell.get(key, [])
        case_count = len({str(r.get("case_id")) for r in audits})
        audit_rows_count = len(audits)
        weak_or_strong = [r for r in audits if r.get("necessity_supported")]
        strong = [r for r in audits if r.get("causal_support_level") == "strong"]
        mean_delta = mean_safe([safe_float(r.get("delta_continue_stop_margin")) for r in audits])
        support_rate = round(len(weak_or_strong) / audit_rows_count, 6) if audit_rows_count else 0.0
        strong_rate = round(len(strong) / audit_rows_count, 6) if audit_rows_count else 0.0
        coverage_score = clamp(case_count / 3.0)
        causal_score = clamp(0.45 * coverage_score + 0.30 * support_rate + 0.15 * strong_rate + 0.10 * min(abs(mean_delta) / 8.0, 1.0))
        prev_completion = safe_float(prev.get("atlas_completion_v4"))
        atlas_completion_v5 = clamp(prev_completion + 0.07 * causal_score)
        if case_count == 0:
            priority = str(prev.get("next_priority_v4") or "mlp_continue_path_causal_audit")
        elif support_rate >= 0.5 and strong_rate >= 0.2:
            priority = "causal_path_expand_and_stop_source_search"
        elif support_rate > 0.0:
            priority = "weak_causal_path_expand"
        else:
            priority = "side_effect_or_noncausal_path_recheck"
        rows.append(
            {
                **prev,
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "atlas_completion_v5": round(atlas_completion_v5, 6),
                "delta_v5_minus_v4": round(atlas_completion_v5 - prev_completion, 6),
                "expanded_mlp_causal_case_count": case_count,
                "expanded_mlp_causal_audit_rows": audit_rows_count,
                "expanded_mlp_causal_support_rate": support_rate,
                "expanded_mlp_causal_strong_rate": strong_rate,
                "expanded_mlp_mean_delta_continue_stop_margin": mean_delta,
                "expanded_mlp_causal_score": round(causal_score, 6),
                "next_priority_v5": priority,
            }
        )
    by_model = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model"))].append(row)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "input_cells": len(prev_rows),
        "feature_cells": len(rows),
        "phase298_audit_rows": len(audit_rows),
        "phase298_selected_mlp_dominant_cases": summary_298.get("selected_mlp_dominant_cases"),
        "mean_atlas_completion_v5": mean_safe([safe_float(r.get("atlas_completion_v5")) for r in rows]),
        "mean_completion_delta_v5_minus_v4": mean_safe([safe_float(r.get("delta_v5_minus_v4")) for r in rows]),
        "mean_expanded_mlp_causal_support_rate": mean_safe([safe_float(r.get("expanded_mlp_causal_support_rate")) for r in rows]),
        "mean_expanded_mlp_causal_strong_rate": mean_safe([safe_float(r.get("expanded_mlp_causal_strong_rate")) for r in rows]),
        "mean_expanded_mlp_causal_score": mean_safe([safe_float(r.get("expanded_mlp_causal_score")) for r in rows]),
        "next_priority_counts": dict(Counter(str(r.get("next_priority_v5")) for r in rows)),
        "model_completion_v5": {
            model: mean_safe([safe_float(r.get("atlas_completion_v5")) for r in vals]) for model, vals in sorted(by_model.items())
        },
        "progress": {
            "language_pattern_family_atlas": 0.78,
            "sample_type_coverage": 0.70,
            "large_data_feature_mining": 0.66,
            "physical_distribution_puzzle": 0.72,
            "mechanism_causal_audit": 0.50,
            "closure": 0.21,
        },
    }
    write_jsonl(V2 / "phase299_feature_matrix_v5_rows.jsonl", rows)
    write_json(V2 / "phase299_summary.json", payload)
    write_report(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Phase299 Feature Algorithm v5 With Causal Audit",
        "",
        f"- feature_cells: {payload['feature_cells']}",
        f"- phase298_audit_rows: {payload['phase298_audit_rows']}",
        f"- mean_atlas_completion_v5: {payload['mean_atlas_completion_v5']}",
        f"- mean_completion_delta_v5_minus_v4: {payload['mean_completion_delta_v5_minus_v4']}",
        f"- mean_expanded_mlp_causal_support_rate: {payload['mean_expanded_mlp_causal_support_rate']}",
        f"- mean_expanded_mlp_causal_strong_rate: {payload['mean_expanded_mlp_causal_strong_rate']}",
        f"- next_priority_counts: {json.dumps(payload['next_priority_counts'], ensure_ascii=False)}",
        "",
        "v5 adds causal audit evidence to the expanded pattern atlas feature matrix.",
    ]
    (V2 / "phase299_feature_algorithm_v5_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
