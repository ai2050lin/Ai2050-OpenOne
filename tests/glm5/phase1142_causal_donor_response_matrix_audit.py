#!/usr/bin/env python3
"""Independent raw-record audit for Phase1142."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1142_causal_donor_response_matrix"
SCRIPT = ROOT / "tests/glm5/phase1142_causal_donor_response_matrix.py"
MODELS = ("qwen3_4b", "qwen3_14b")
SPLITS = ("discovery", "confirmation")
PROPERTIES = ("P54", "P286", "P6")
COHORT_SIZE = 12
ITEMS_PER_PROPERTY = 4
EXPECTED_RECORDS = 1152
EXPECTED_COMPARISONS = 288
EXPECTED_MATRIX = 264
ALPHAS = (0.0, 1.0)
EPSILON = 1e-8


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def median(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(finite) if finite else None


def fraction(rows: Iterable[dict[str, Any]], key: str) -> float:
    rows = list(rows)
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def independent_curves(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["curve_id"])].append(row)
    result = []
    for curve_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        if [float(row["alpha"]) for row in ordered] != list(ALPHAS):
            raise RuntimeError(f"alpha drift for {curve_id}")
        base, endpoint = ordered
        finite = bool(base["finite"] and endpoint["finite"])
        before = base["full_oriented_margin"]
        after = endpoint["full_oriented_margin"]
        change = (
            float(after) - float(before)
            if finite and before is not None and after is not None
            else None
        )
        result.append(
            {
                "curve_id": curve_id,
                "comparison_id": base["comparison_id"],
                "comparison_kind": base["comparison_kind"],
                "arm": base["arm"],
                "model": base["model"],
                "split": base["split"],
                "item_id": base["item_id"],
                "source_item_id": base["source_item_id"],
                "property_id": base["property_id"],
                "source_property_id": base["source_property_id"],
                "same_relation": bool(base["same_relation"]),
                "panel": base["panel"],
                "finite": finite,
                "baseline_margin": before,
                "endpoint_margin": after,
                "margin_change": change,
                "baseline_valid": bool(
                    finite and before is not None and float(before) < 0.0
                ),
                "endpoint_flip": bool(
                    finite and after is not None and float(after) > 0.0
                ),
                "positive_change": bool(change is not None and change > 0.0),
            }
        )
    return result


def independent_comparisons(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in curves:
        grouped[str(row["comparison_id"])][str(row["arm"])] = row
    result = []
    for comparison_id, arms in sorted(grouped.items()):
        if set(arms) != {"paired_correct_donor", "challenger_donor"}:
            raise RuntimeError(f"arm drift for {comparison_id}")
        reference = arms["paired_correct_donor"]
        challenger = arms["challenger_donor"]
        finite = bool(reference["finite"] and challenger["finite"])
        advantage = (
            float(reference["margin_change"]) - float(challenger["margin_change"])
            if finite
            and reference["margin_change"] is not None
            and challenger["margin_change"] is not None
            else None
        )
        alpha0 = (
            abs(float(reference["baseline_margin"]) - float(challenger["baseline_margin"]))
            if finite
            and reference["baseline_margin"] is not None
            and challenger["baseline_margin"] is not None
            else None
        )
        result.append(
            {
                "comparison_id": comparison_id,
                "comparison_kind": reference["comparison_kind"],
                "model": reference["model"],
                "split": reference["split"],
                "item_id": reference["item_id"],
                "source_item_id": challenger["source_item_id"],
                "property_id": reference["property_id"],
                "source_property_id": challenger["source_property_id"],
                "same_relation": bool(challenger["same_relation"]),
                "panel": reference["panel"],
                "finite": finite,
                "paired_alpha0_abs_margin_difference": alpha0,
                "diagonal_change": reference["margin_change"],
                "challenger_change": challenger["margin_change"],
                "diagonal_endpoint_flip": bool(reference["endpoint_flip"]),
                "challenger_endpoint_flip": bool(challenger["endpoint_flip"]),
                "diagonal_positive_change": bool(reference["positive_change"]),
                "challenger_positive_change": bool(challenger["positive_change"]),
                "diagonal_baseline_valid": bool(reference["baseline_valid"]),
                "challenger_baseline_valid": bool(challenger["baseline_valid"]),
                "diagonal_advantage": advantage,
                "diagonal_advantage_positive": bool(
                    advantage is not None and advantage > 0.0
                ),
            }
        )
    return result


def independent_target_panels(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in matrix:
        grouped[(str(row["item_id"]), str(row["panel"]))].append(row)
    result = []
    for (item_id, panel), rows in sorted(grouped.items()):
        same = [row for row in rows if row["same_relation"]]
        cross = [row for row in rows if not row["same_relation"]]
        if len(rows) != 11 or len(same) != 3 or len(cross) != 8:
            raise RuntimeError(f"matrix block drift for {item_id} {panel}")
        diagonal = median(row["diagonal_change"] for row in rows)
        donors = [float(row["challenger_change"]) for row in rows]
        rank = (
            1 + sum(value >= float(diagonal) for value in donors)
            if diagonal is not None
            else None
        )
        same_median = median(row["challenger_change"] for row in same)
        cross_median = median(row["challenger_change"] for row in cross)
        relation_advantage = (
            float(same_median) - float(cross_median)
            if same_median is not None and cross_median is not None
            else None
        )
        result.append(
            {
                "item_id": item_id,
                "panel": panel,
                "property_id": rows[0]["property_id"],
                "diagonal_change_median": diagonal,
                "same_relation_change_median": same_median,
                "cross_relation_change_median": cross_median,
                "relation_advantage": relation_advantage,
                "relation_advantage_positive": bool(
                    relation_advantage is not None and relation_advantage > 0.0
                ),
                "diagonal_rank": rank,
                "diagonal_top1": rank == 1,
                "diagonal_top3": rank is not None and rank <= 3,
            }
        )
    return result


def independent_metrics(
    model: str,
    split: str,
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    curves = independent_curves(records)
    comparisons = independent_comparisons(curves)
    matrix = [row for row in comparisons if row["comparison_kind"] == "matrix"]
    same = [row for row in matrix if row["same_relation"]]
    cross = [row for row in matrix if not row["same_relation"]]
    temporal = [
        row
        for row in comparisons
        if row["comparison_kind"] == "same_answer_temporal_control"
    ]
    self_controls = [
        row
        for row in comparisons
        if row["comparison_kind"] == "self_identity_control"
    ]
    target_panels = independent_target_panels(matrix)
    diagonal = median(row["diagonal_change"] for row in matrix)
    temporal_abs = median(
        abs(float(row["challenger_change"]))
        for row in temporal
        if row["challenger_change"] is not None
    )
    alpha0 = [
        float(row["paired_alpha0_abs_margin_difference"])
        for row in comparisons
        if row["paired_alpha0_abs_margin_difference"] is not None
    ]
    self_abs = [
        abs(float(row["challenger_change"]))
        for row in self_controls
        if row["challenger_change"] is not None
    ]
    per_item = {
        prop: median(
            row["diagonal_advantage"] for row in matrix if row["property_id"] == prop
        )
        for prop in PROPERTIES
    }
    per_relation = {
        prop: median(
            row["relation_advantage"]
            for row in target_panels
            if row["property_id"] == prop
        )
        for prop in PROPERTIES
    }
    donor_columns = {}
    for donor in sorted({str(row["source_item_id"]) for row in matrix}):
        rows = [row for row in matrix if str(row["source_item_id"]) == donor]
        donor_columns[donor] = {
            "property_id": str(rows[0]["source_property_id"]),
            "median_change": median(row["challenger_change"] for row in rows),
            "flip_fraction": fraction(rows, "challenger_endpoint_flip"),
        }
    metrics = {
        "model": model,
        "split": split,
        "record_count": len(records),
        "curve_count": len(curves),
        "comparison_count": len(comparisons),
        "matrix_comparison_count": len(matrix),
        "same_relation_matrix_count": len(same),
        "cross_relation_matrix_count": len(cross),
        "finite_fraction": sum(bool(row["finite"]) for row in records) / max(len(records), 1),
        "paired_alpha0_max_abs_margin_difference": max(alpha0) if alpha0 else None,
        "diagonal_baseline_valid_fraction": fraction(matrix, "diagonal_baseline_valid"),
        "diagonal_endpoint_flip_fraction": fraction(matrix, "diagonal_endpoint_flip"),
        "diagonal_positive_change_fraction": fraction(matrix, "diagonal_positive_change"),
        "diagonal_change_median": diagonal,
        "offdiagonal_change_median": median(row["challenger_change"] for row in matrix),
        "same_relation_change_median": median(row["challenger_change"] for row in same),
        "cross_relation_change_median": median(row["challenger_change"] for row in cross),
        "offdiagonal_endpoint_flip_fraction": fraction(matrix, "challenger_endpoint_flip"),
        "same_relation_endpoint_flip_fraction": fraction(same, "challenger_endpoint_flip"),
        "cross_relation_endpoint_flip_fraction": fraction(cross, "challenger_endpoint_flip"),
        "cross_relation_positive_change_fraction": fraction(cross, "challenger_positive_change"),
        "item_advantage_median": median(row["diagonal_advantage"] for row in matrix),
        "item_advantage_positive_fraction": fraction(matrix, "diagonal_advantage_positive"),
        "item_advantage_same_relation_median": median(row["diagonal_advantage"] for row in same),
        "item_advantage_cross_relation_median": median(row["diagonal_advantage"] for row in cross),
        "diagonal_minus_offdiagonal_flip_fraction": fraction(matrix, "diagonal_endpoint_flip") - fraction(matrix, "challenger_endpoint_flip"),
        "diagonal_top1_fraction": fraction(target_panels, "diagonal_top1"),
        "diagonal_top3_fraction": fraction(target_panels, "diagonal_top3"),
        "diagonal_rank_median": median(row["diagonal_rank"] for row in target_panels),
        "per_property_item_advantage_median": per_item,
        "relation_advantage_median": median(row["relation_advantage"] for row in target_panels),
        "relation_advantage_positive_fraction": fraction(target_panels, "relation_advantage_positive"),
        "same_minus_cross_relation_flip_fraction": fraction(same, "challenger_endpoint_flip") - fraction(cross, "challenger_endpoint_flip"),
        "per_property_relation_advantage_median": per_relation,
        "same_answer_endpoint_flip_fraction": fraction(temporal, "challenger_endpoint_flip"),
        "same_answer_abs_change_median": temporal_abs,
        "self_identity_max_abs_margin_change": max(self_abs) if self_abs else None,
        "diagonal_to_same_answer_abs_ratio": (
            float(diagonal) / max(float(temporal_abs), EPSILON)
            if diagonal is not None and temporal_abs is not None
            else None
        ),
        "donor_columns": donor_columns,
    }
    instrument = {
        "finite": metrics["finite_fraction"] >= thresholds["finite_fraction"],
        "paired_alpha0": metrics["paired_alpha0_max_abs_margin_difference"] is not None and metrics["paired_alpha0_max_abs_margin_difference"] <= thresholds["paired_alpha0_max_abs_margin_difference"],
        "diagonal_baseline": metrics["diagonal_baseline_valid_fraction"] >= thresholds["diagonal_baseline_valid_fraction"],
        "diagonal_endpoint": metrics["diagonal_endpoint_flip_fraction"] >= thresholds["diagonal_endpoint_flip_fraction"],
        "diagonal_positive": metrics["diagonal_positive_change_fraction"] >= thresholds["diagonal_positive_change_fraction"],
        "same_answer": metrics["same_answer_endpoint_flip_fraction"] <= thresholds["same_answer_endpoint_flip_fraction"],
        "self_identity": metrics["self_identity_max_abs_margin_change"] is not None and metrics["self_identity_max_abs_margin_change"] <= thresholds["self_identity_max_abs_margin_change"],
        "temporal_ratio": metrics["diagonal_to_same_answer_abs_ratio"] is not None and metrics["diagonal_to_same_answer_abs_ratio"] >= thresholds["diagonal_to_same_answer_abs_ratio"],
    }
    item = {
        "instrument": all(instrument.values()),
        "median": metrics["item_advantage_median"] is not None and metrics["item_advantage_median"] >= thresholds["item_advantage_median"],
        "positive_fraction": metrics["item_advantage_positive_fraction"] >= thresholds["item_advantage_positive_fraction"],
        "same_relation": metrics["item_advantage_same_relation_median"] is not None and metrics["item_advantage_same_relation_median"] >= thresholds["item_advantage_same_relation_median"],
        "cross_relation": metrics["item_advantage_cross_relation_median"] is not None and metrics["item_advantage_cross_relation_median"] >= thresholds["item_advantage_cross_relation_median"],
        "flip_advantage": metrics["diagonal_minus_offdiagonal_flip_fraction"] >= thresholds["diagonal_minus_offdiagonal_flip_fraction"],
        "top1": metrics["diagonal_top1_fraction"] >= thresholds["diagonal_top1_fraction"],
        "each_property": all(per_item[prop] is not None and float(per_item[prop]) >= thresholds["per_property_item_advantage_median"] for prop in PROPERTIES),
    }
    relation = {
        "instrument": all(instrument.values()),
        "median": metrics["relation_advantage_median"] is not None and metrics["relation_advantage_median"] >= thresholds["relation_advantage_median"],
        "positive_fraction": metrics["relation_advantage_positive_fraction"] >= thresholds["relation_advantage_positive_fraction"],
        "flip_advantage": metrics["same_minus_cross_relation_flip_fraction"] >= thresholds["same_minus_cross_relation_flip_fraction"],
        "each_property": all(per_relation[prop] is not None and float(per_relation[prop]) >= thresholds["per_property_relation_advantage_median"] for prop in PROPERTIES),
    }
    global_field = {
        "cross_relation_flip": metrics["cross_relation_endpoint_flip_fraction"] >= thresholds["global_cross_relation_flip_fraction"],
        "cross_relation_positive": metrics["cross_relation_positive_change_fraction"] >= thresholds["global_cross_relation_positive_fraction"],
        "item_route_failed": not all(item.values()),
        "relation_route_failed": not all(relation.values()),
    }
    metrics["instrument_gate_checks"] = instrument
    metrics["instrument_qualified"] = all(instrument.values())
    metrics["item_gate_checks"] = item
    metrics["item_selectivity_qualified"] = all(item.values())
    metrics["relation_gate_checks"] = relation
    metrics["relation_selectivity_qualified"] = all(relation.values())
    metrics["global_direction_checks"] = global_field
    metrics["global_role_compatible_field_observed"] = all(global_field.values())
    metrics["curve_digest"] = digest(curves)
    metrics["comparison_digest"] = digest(comparisons)
    return metrics, curves, comparisons


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    checks: dict[str, bool] = {
        "protocol_audit_passed": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest_chain": protocol_audit["protocol_digest"] == prereg["protocol_digest"],
        "primary_script_hash": sha256_file(SCRIPT) == prereg["source"]["script_sha256"],
        "cohorts_12": all(len(prereg["material"]["cohorts"][split]) == COHORT_SIZE for split in SPLITS),
        "cohorts_disjoint": set(prereg["material"]["cohorts"]["discovery"]).isdisjoint(prereg["material"]["cohorts"]["confirmation"]),
        "reserve_12": len(prereg["material"]["reserve_item_ids"]) == 12,
        "reserve_disjoint": set(prereg["material"]["reserve_item_ids"]).isdisjoint(set(prereg["material"]["cohorts"]["discovery"]) | set(prereg["material"]["cohorts"]["confirmation"])),
    }
    recomputed: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        for model in MODELS:
            summary_path = OUT_ROOT / "runs" / split / model / "summary.json"
            if not summary_path.exists():
                continue
            prefix = f"{split}_{model}"
            summary = read_json(summary_path)
            records = read_jsonl(summary_path.parent / "records.jsonl")
            stored_curves = read_jsonl(summary_path.parent / "curves.jsonl")
            stored_comparisons = read_jsonl(summary_path.parent / "comparisons.jsonl")
            metrics, curves, comparisons = independent_metrics(model, split, records, prereg["thresholds"])
            recomputed[prefix] = metrics
            checks[f"{prefix}_record_count"] = len(records) == EXPECTED_RECORDS == summary["record_count"]
            checks[f"{prefix}_record_digest"] = digest(records) == summary["record_digest"]
            checks[f"{prefix}_curve_exact"] = canonical(curves) == canonical(stored_curves)
            checks[f"{prefix}_comparison_exact"] = canonical(comparisons) == canonical(stored_comparisons)
            checks[f"{prefix}_metrics_exact"] = canonical(metrics) == canonical(summary["metrics"])
            checks[f"{prefix}_comparison_count"] = len(comparisons) == EXPECTED_COMPARISONS
            checks[f"{prefix}_matrix_count"] = sum(row["comparison_kind"] == "matrix" for row in comparisons) == EXPECTED_MATRIX
            checks[f"{prefix}_alpha_pairing"] = all(sorted(float(row["alpha"]) for row in records if row["curve_id"] == curve["curve_id"]) == list(ALPHAS) for curve in curves)
            checks[f"{prefix}_protocol_digest"] = summary["protocol_digest"] == prereg["protocol_digest"]

    discovery_item = {model: bool(recomputed[f"discovery_{model}"]["item_selectivity_qualified"]) for model in MODELS}
    discovery_relation = {model: bool(recomputed[f"discovery_{model}"]["relation_selectivity_qualified"]) for model in MODELS}
    item_authorized = all(discovery_item.values())
    relation_authorized = all(discovery_relation.values())
    selected_level = "item" if item_authorized else "relation" if relation_authorized else None
    confirmation_authorized = item_authorized or relation_authorized
    checks["selection_item"] = selection["item_qualified"] == discovery_item
    checks["selection_relation"] = selection["relation_qualified"] == discovery_relation
    checks["selection_level"] = selection["selected_claim_level"] == selected_level
    checks["selection_authorization"] = selection["confirmation_authorized"] == confirmation_authorized
    selection_core = {key: value for key, value in selection.items() if key != "selection_digest"}
    checks["selection_digest"] = digest(selection_core) == selection["selection_digest"]

    confirmation_run = any((OUT_ROOT / "runs" / "confirmation" / model / "summary.json").exists() for model in MODELS)
    confirmed = bool(
        confirmation_run
        and selected_level is not None
        and all(recomputed[f"confirmation_{model}"][f"{selected_level}_selectivity_qualified"] for model in MODELS)
    )
    expected_outcome = (
        "discovery_no_identity_or_relation_selectivity"
        if not confirmation_authorized
        else f"{selected_level}_selectivity_independently_confirmed"
        if confirmed
        else f"{selected_level}_selectivity_confirmation_failed"
    )
    checks["final_confirmation_run"] = final["confirmation_run"] == confirmation_run
    checks["final_confirmed"] = final["selected_route_confirmed"] == confirmed
    checks["final_outcome"] = final["outcome"] == expected_outcome
    checks["final_auto_continue"] = final["auto_continue"] == confirmed
    checks["final_no_component_search"] = final["component_search_authorized"] is False
    final_core = {key: value for key, value in final.items() if key != "final_digest"}
    checks["final_digest"] = digest(final_core) == final["final_digest"]

    audit_core = {
        "schema_version": "phase1142_independent_result_audit.v1",
        "phase": 1142,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": prereg["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "final_digest": final["final_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "independent_metrics": recomputed,
        "audit_script_sha256": sha256_file(Path(__file__)),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "audit/independent_result_audit.json", audit)
    print(
        json.dumps(
            {
                "phase": 1142,
                "checks": f"{audit['passed_count']}/{audit['check_count']}",
                "all_checks_passed": audit["all_checks_passed"],
                "audit_digest": audit["audit_digest"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
