#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

PHASE = 292
SCHEMA_VERSION = "2.19.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase292_feature_analysis_algorithm_v2"


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


def rate(count: float, total: float) -> float:
    return round(count / total, 6) if total else 0.0


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, value)), 6)


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    h = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        h -= p * math.log(p)
    return round(h / math.log(len(counts)), 6)


def group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(k) or "") for k in keys)].append(row)
    return out


def confidence_flag(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    if value > 0:
        return "low"
    return "none"


def build_feature_rows() -> list[dict[str, Any]]:
    family_features = {(str(r.get("family_id")),): r for r in read_jsonl(V2 / "phase288_family_feature_matrix.jsonl")}
    model_features = {(str(r.get("model")),): r for r in read_jsonl(V2 / "phase288_model_feature_matrix.jsonl")}
    readout_rows = read_jsonl(V2 / "phase290_family_model_readout_summary.jsonl")
    heat_rows = {(str(r.get("family_id")), str(r.get("model"))): r for r in read_jsonl(V2 / "phase288_gap_heatmap_rows.jsonl")}
    selected = group(read_jsonl(V2 / "phase291_selected_large_batch_queue.jsonl"), "family_id", "model")
    full_plan = group(read_jsonl(V2 / "phase291_full_model_test_plan_rows.jsonl"), "family_id", "model")
    output: list[dict[str, Any]] = []
    for row in readout_rows:
        family = str(row.get("family_id"))
        model = str(row.get("model"))
        family_row = family_features.get((family,), {})
        model_row = model_features.get((model,), {})
        heat = heat_rows.get((family, model), {})
        selected_rows = selected.get((family, model), [])
        full_rows = full_plan.get((family, model), [])
        signature_rows = int(row.get("rows") or 0)
        channel_counts = {str(k): int(v) for k, v in (row.get("channel_family_counts") or {}).items()}
        bottleneck_counts = {str(k): int(v) for k, v in (row.get("bottleneck_counts") or {}).items()}
        structure_count = sum(channel_counts.get(k, 0) for k in ["list_structure_continue", "protocol_json_continue", "protocol_format_continue"])
        stop_not_winner = bottleneck_counts.get("stop_not_winner", 0)
        continue_not_suppressed = bottleneck_counts.get("continue_not_suppressed", 0)
        target_weak = bottleneck_counts.get("target_readout_weak", 0)
        component_coverage = rate(safe_float(family_row.get("component_summary_rows")), safe_float(family_row.get("signature_rows")))
        causal_coverage = rate(safe_float(family_row.get("causal_rows")), safe_float(family_row.get("signature_rows")))
        closure_coverage = rate(safe_float(family_row.get("closure_quality_rows")), safe_float(family_row.get("signature_rows")))
        measurement_coverage = mean_safe([component_coverage, causal_coverage, closure_coverage, 1.0])
        expanded_selection_rate = rate(len(selected_rows), len(full_rows))
        gap_norm = clamp(safe_float(heat.get("open_gap_total")) / 180.0)
        channel_entropy = entropy_from_counts(channel_counts)
        structure_continue_rate = rate(structure_count, signature_rows)
        bottleneck_pressure = rate(stop_not_winner + continue_not_suppressed, 2 * signature_rows)
        target_weak_rate = rate(target_weak, signature_rows)
        side_effect_risk = safe_float(model_row.get("side_effect_risk_rate"))
        behavior = safe_float(family_row.get("mean_behavior_score"))
        readout_score = safe_float(family_row.get("mean_readout_score"))
        rollout = safe_float(family_row.get("mean_rollout_score"))
        completion = clamp(
            0.14 * behavior
            + 0.10 * readout_score
            + 0.09 * rollout
            + 0.12 * measurement_coverage
            + 0.10 * expanded_selection_rate
            + 0.10 * channel_entropy
            + 0.08 * (1.0 - structure_continue_rate)
            + 0.10 * (1.0 - target_weak_rate)
            + 0.09 * (1.0 - gap_norm)
            + 0.08 * (1.0 - side_effect_risk)
            - 0.10 * bottleneck_pressure
        )
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase292",
                "created_at": now(),
                "feature_v2_id": f"phase292:feature_v2:{family}:{model}",
                "family_id": family,
                "model": model,
                "signature_rows": signature_rows,
                "behavior_score": round(behavior, 6),
                "readout_score": round(readout_score, 6),
                "rollout_score": round(rollout, 6),
                "component_coverage": component_coverage,
                "causal_coverage": causal_coverage,
                "closure_quality_coverage": closure_coverage,
                "measurement_coverage": measurement_coverage,
                "expanded_full_plan_rows": len(full_rows),
                "expanded_selected_rows": len(selected_rows),
                "expanded_selection_rate": expanded_selection_rate,
                "channel_entropy": channel_entropy,
                "channel_family_counts": channel_counts,
                "structure_continue_rate": structure_continue_rate,
                "bottleneck_pressure": bottleneck_pressure,
                "target_weak_rate": target_weak_rate,
                "side_effect_risk_rate": round(side_effect_risk, 6),
                "open_gap_total": int(heat.get("open_gap_total") or 0),
                "gap_pressure_norm": gap_norm,
                "atlas_completion_v2": completion,
                "completion_confidence": confidence_flag(measurement_coverage),
                "next_priority": priority_label(completion, bottleneck_pressure, gap_norm, structure_continue_rate),
            }
        )
    output.sort(key=lambda r: (safe_float(r.get("atlas_completion_v2")), -safe_float(r.get("bottleneck_pressure")), str(r.get("model")), str(r.get("family_id"))))
    return output


def priority_label(completion: float, bottleneck: float, gap_norm: float, structure_rate: float) -> str:
    if completion < 0.45 and bottleneck > 0.8:
        return "urgent_readout_bottleneck"
    if structure_rate > 0.55:
        return "protocol_structure_channel_audit"
    if gap_norm > 0.75:
        return "large_gap_physical_path_fill"
    if completion < 0.55:
        return "balanced_large_batch_measurement"
    return "monitor_and_validate"


def channel_entropy_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in feature_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase292",
                "created_at": now(),
                "entropy_id": f"phase292:channel_entropy:{row['family_id']}:{row['model']}",
                "family_id": row["family_id"],
                "model": row["model"],
                "channel_entropy": row["channel_entropy"],
                "channel_family_counts": row["channel_family_counts"],
                "structure_continue_rate": row["structure_continue_rate"],
                "interpretation": "higher entropy means multiple continue families compete; lower entropy means one dominant channel family",
            }
        )
    return rows


def gap_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in feature_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase292",
                "created_at": now(),
                "gap_v2_id": f"phase292:coverage_gap:{row['family_id']}:{row['model']}",
                "family_id": row["family_id"],
                "model": row["model"],
                "open_gap_total": row["open_gap_total"],
                "gap_pressure_norm": row["gap_pressure_norm"],
                "measurement_coverage": row["measurement_coverage"],
                "expanded_selection_rate": row["expanded_selection_rate"],
                "bottleneck_pressure": row["bottleneck_pressure"],
                "atlas_completion_v2": row["atlas_completion_v2"],
                "next_priority": row["next_priority"],
            }
        )
    return rows


def priority_queue(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = read_jsonl(V2 / "phase291_selected_large_batch_queue.jsonl")
    feature_by_cell = {(str(r["family_id"]), str(r["model"])): r for r in feature_rows}
    rows = []
    for item in selected:
        feature = feature_by_cell.get((str(item.get("family_id")), str(item.get("model"))), {})
        score = (
            safe_float(item.get("priority_score"))
            + 5.0 * (1.0 - safe_float(feature.get("atlas_completion_v2")))
            + 2.0 * safe_float(feature.get("bottleneck_pressure"))
            + 2.0 * safe_float(feature.get("gap_pressure_norm"))
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase292",
                "created_at": now(),
                "priority_id": f"phase292:priority:{item.get('model')}:{item.get('case_id')}",
                "model": item.get("model"),
                "family_id": item.get("family_id"),
                "case_id": item.get("case_id"),
                "variant_id": item.get("variant_id"),
                "channel_focus": item.get("channel_focus"),
                "prompt": item.get("prompt"),
                "target": item.get("target"),
                "expected_pattern": item.get("expected_pattern"),
                "atlas_completion_v2": feature.get("atlas_completion_v2"),
                "cell_next_priority": feature.get("next_priority"),
                "phase292_priority_score": round(score, 6),
                "recommended_execution": "sequential_cuda_behavior_readout_first",
            }
        )
    rows.sort(key=lambda r: (-safe_float(r.get("phase292_priority_score")), str(r.get("model")), str(r.get("family_id")), str(r.get("case_id"))))
    for rank, row in enumerate(rows, 1):
        row["phase292_rank"] = rank
    return rows


def completion_summary(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = defaultdict(list)
    by_model = defaultdict(list)
    for row in feature_rows:
        by_family[str(row["family_id"])].append(safe_float(row["atlas_completion_v2"]))
        by_model[str(row["model"])].append(safe_float(row["atlas_completion_v2"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase292",
        "created_at": now(),
        "feature_matrix_rows": len(feature_rows),
        "mean_atlas_completion_v2": mean_safe([safe_float(r.get("atlas_completion_v2")) for r in feature_rows]),
        "min_atlas_completion_v2": round(min(safe_float(r.get("atlas_completion_v2")) for r in feature_rows), 6) if feature_rows else 0.0,
        "max_atlas_completion_v2": round(max(safe_float(r.get("atlas_completion_v2")) for r in feature_rows), 6) if feature_rows else 0.0,
        "mean_bottleneck_pressure": mean_safe([safe_float(r.get("bottleneck_pressure")) for r in feature_rows]),
        "mean_channel_entropy": mean_safe([safe_float(r.get("channel_entropy")) for r in feature_rows]),
        "mean_measurement_coverage": mean_safe([safe_float(r.get("measurement_coverage")) for r in feature_rows]),
        "family_completion": {k: mean_safe(v) for k, v in sorted(by_family.items())},
        "model_completion": {k: mean_safe(v) for k, v in sorted(by_model.items())},
        "next_priority_counts": dict(Counter(str(r.get("next_priority")) for r in feature_rows)),
        "progress_estimate": {
            "pattern_family_atlas": 0.71,
            "sample_type_coverage": 0.62,
            "feature_mining": 0.50,
            "physical_distribution_puzzle": 0.64,
            "mechanism_audit": 0.42,
            "closure": 0.20,
        },
    }


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    for name in [
        "phase292_feature_matrix_v2_rows",
        "phase292_channel_entropy_rows",
        "phase292_coverage_normalized_gap_rows",
        "phase292_feature_priority_queue_rows",
    ]:
        files[name] = f"{name}.jsonl"
    files["phase292_global_atlas_completion"] = "phase292_global_atlas_completion.json"
    files["phase292_report"] = "phase292_report.md"
    manifest["latest_feature_algorithm_phase"] = "Phase292"
    manifest["phase292_summary"] = summary
    write_json(V2 / "manifest.json", manifest)

    client = read_json(V2 / "client_index.json")
    for item in [
        "phase292_global_atlas_completion.json",
        "phase292_feature_matrix_v2_rows.jsonl",
        "phase292_feature_priority_queue_rows.jsonl",
    ]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase292_summary_ref"] = "phase292_global_atlas_completion.json"
    client["phase292_priority_queue_ref"] = "phase292_feature_priority_queue_rows.jsonl"
    write_json(V2 / "client_index.json", client)

    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase292_feature_matrix_v2_rows"] = "coverage-normalized feature matrix with channel entropy, bottleneck pressure, gap pressure, and completion score"
    tables["phase292_channel_entropy_rows"] = "channel-family entropy and structure continuation distribution"
    tables["phase292_coverage_normalized_gap_rows"] = "coverage-normalized gap and bottleneck pressure by family-model cell"
    tables["phase292_feature_priority_queue_rows"] = "Phase291 large-batch queue rescored by feature algorithm v2"
    write_json(V2 / "schema.json", schema)


def main() -> None:
    feature = build_feature_rows()
    entropy = channel_entropy_rows(feature)
    gaps = gap_rows(feature)
    queue = priority_queue(feature)
    summary = completion_summary(feature)
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "phase292_feature_matrix_v2_rows.jsonl": feature,
        "phase292_channel_entropy_rows.jsonl": entropy,
        "phase292_coverage_normalized_gap_rows.jsonl": gaps,
        "phase292_feature_priority_queue_rows.jsonl": queue,
    }
    for name, rows in outputs.items():
        write_jsonl(OUT / name, rows)
        write_jsonl(V2 / name, rows)
    write_json(OUT / "phase292_global_atlas_completion.json", summary)
    write_json(V2 / "phase292_global_atlas_completion.json", summary)
    report = "\n".join(
        [
            "# Phase292 Feature Analysis Algorithm V2",
            "",
            f"- feature_matrix_rows: {summary['feature_matrix_rows']}",
            f"- mean_atlas_completion_v2: {summary['mean_atlas_completion_v2']}",
            f"- mean_bottleneck_pressure: {summary['mean_bottleneck_pressure']}",
            f"- mean_channel_entropy: {summary['mean_channel_entropy']}",
            f"- mean_measurement_coverage: {summary['mean_measurement_coverage']}",
            f"- next_priority_counts: {json.dumps(summary['next_priority_counts'], ensure_ascii=False)}",
            f"- family_completion: {json.dumps(summary['family_completion'], ensure_ascii=False)}",
            f"- model_completion: {json.dumps(summary['model_completion'], ensure_ascii=False)}",
            "",
            "This algorithm separates measured atlas evidence from planned sample expansion and uses coverage-aware completion scoring.",
        ]
    ) + "\n"
    (OUT / "phase292_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase292_report.md").write_text(report, encoding="utf-8")
    update_v2(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
