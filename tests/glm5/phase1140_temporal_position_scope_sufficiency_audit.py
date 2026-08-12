#!/usr/bin/env python3
"""Independent result audit for Phase1140."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1140_temporal_position_scope_sufficiency"
MODELS = ("qwen3_4b", "qwen3_14b")
SPLITS = ("discovery", "confirmation")
SCOPES = ("answer_boundary", "candidate_prediction_span")
EXPECTED_RECORDS = 576
EXPECTED_CURVES = 144
EPSILON = 1e-8
THRESHOLDS = {
    "finite_fraction": 0.99,
    "identity_max_abs_margin_drift": 0.005,
    "baseline_valid_fraction": 0.99,
    "main_endpoint_flip_fraction": 0.95,
    "panel_endpoint_flip_fraction": 0.95,
    "main_positive_change_fraction": 0.99,
    "main_to_same_answer_span_ratio": 2.0,
    "same_answer_control_flip_fraction": 0.10,
    "span_rescue_min_improvement": 0.15,
}


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


def independent_scope(
    records: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scope_records = [row for row in records if row["scope"] == scope]
    for row in scope_records:
        grouped[str(row["curve_id"])].append(row)
    curves = []
    for curve_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        if [float(row["alpha"]) for row in ordered] != [0.0, 1.0]:
            raise RuntimeError(f"incomplete alpha grid for {curve_id}")
        base, endpoint = ordered
        base_margin = base["full_oriented_margin"]
        endpoint_margin = endpoint["full_oriented_margin"]
        change = (
            float(endpoint_margin) - float(base_margin)
            if base_margin is not None and endpoint_margin is not None
            else None
        )
        informative = bool(base["first_token_informative"])
        first_base = base["first_oriented_margin"]
        first_endpoint = endpoint["first_oriented_margin"]
        first_change = (
            float(first_endpoint) - float(first_base)
            if informative and first_base is not None and first_endpoint is not None
            else None
        )
        curves.append({
            "curve_kind": base["curve_kind"],
            "panel": base["panel"],
            "finite": bool(base["finite"] and endpoint["finite"]),
            "identity_full": base["identity_full_margin_drift"],
            "identity_first": base["identity_first_margin_drift"],
            "base": base_margin,
            "endpoint": endpoint_margin,
            "change": change,
            "informative": informative,
            "first_base": first_base,
            "first_endpoint": first_endpoint,
            "first_change": first_change,
        })
    main = [row for row in curves if row["curve_kind"] == "main"]
    controls = [
        row for row in curves if row["curve_kind"] == "same_answer_temporal_control"
    ]
    full_identity = [
        abs(float(row["identity_full"]))
        for row in curves
        if row["identity_full"] is not None
    ]
    first_identity = [
        abs(float(row["identity_first"]))
        for row in curves
        if row["identity_first"] is not None
    ]
    main_span = median(row["change"] for row in main)
    control_span = median(
        abs(float(row["change"]))
        for row in controls
        if row["change"] is not None
    )
    ratio = (
        main_span / max(control_span, EPSILON)
        if main_span is not None and control_span is not None
        else None
    )
    first_main = [row for row in main if row["informative"]]
    result = {
        "record_count": len(scope_records),
        "curve_count": len(curves),
        "main_count": len(main),
        "control_count": len(controls),
        "finite_fraction": sum(row["finite"] for row in scope_records) / max(len(scope_records), 1),
        "identity_full": max(full_identity) if full_identity else None,
        "identity_first": max(first_identity) if first_identity else None,
        "baseline_valid": sum(
            row["finite"] and row["base"] is not None and row["base"] < 0
            for row in main
        ) / max(len(main), 1),
        "endpoint_flip": sum(
            row["finite"] and row["endpoint"] is not None and row["endpoint"] > 0
            for row in main
        ) / max(len(main), 1),
        "original_flip": sum(
            row["endpoint"] is not None and row["endpoint"] > 0
            for row in main
            if row["panel"] == "original"
        ) / max(sum(row["panel"] == "original" for row in main), 1),
        "swapped_flip": sum(
            row["endpoint"] is not None and row["endpoint"] > 0
            for row in main
            if row["panel"] == "swapped"
        ) / max(sum(row["panel"] == "swapped" for row in main), 1),
        "positive_change": sum(
            row["change"] is not None and row["change"] > 0 for row in main
        ) / max(len(main), 1),
        "main_span": main_span,
        "control_span": control_span,
        "span_ratio": ratio,
        "control_flip": sum(
            row["endpoint"] is not None and row["endpoint"] > 0
            for row in controls
        ) / max(len(controls), 1),
        "first_count": len(first_main),
        "first_baseline_valid": sum(
            row["first_base"] is not None and row["first_base"] < 0
            for row in first_main
        ) / max(len(first_main), 1),
        "first_endpoint_flip": sum(
            row["first_endpoint"] is not None and row["first_endpoint"] > 0
            for row in first_main
        ) / max(len(first_main), 1),
        "first_positive_change": sum(
            row["first_change"] is not None and row["first_change"] > 0
            for row in first_main
        ) / max(len(first_main), 1),
    }
    result["qualified"] = bool(
        result["finite_fraction"] >= THRESHOLDS["finite_fraction"]
        and result["identity_full"] is not None
        and result["identity_full"] <= THRESHOLDS["identity_max_abs_margin_drift"]
        and (
            result["identity_first"] is None
            or result["identity_first"] <= THRESHOLDS["identity_max_abs_margin_drift"]
        )
        and result["baseline_valid"] >= THRESHOLDS["baseline_valid_fraction"]
        and result["endpoint_flip"] >= THRESHOLDS["main_endpoint_flip_fraction"]
        and result["original_flip"] >= THRESHOLDS["panel_endpoint_flip_fraction"]
        and result["swapped_flip"] >= THRESHOLDS["panel_endpoint_flip_fraction"]
        and result["positive_change"] >= THRESHOLDS["main_positive_change_fraction"]
        and result["span_ratio"] is not None
        and result["span_ratio"] >= THRESHOLDS["main_to_same_answer_span_ratio"]
        and result["control_flip"] <= THRESHOLDS["same_answer_control_flip_fraction"]
    )
    return result


def close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def compare_scope(
    prefix: str,
    independent: dict[str, Any],
    recorded: dict[str, Any],
    checks: dict[str, bool],
) -> None:
    mapping = {
        "record_count": "record_count",
        "curve_count": "curve_count",
        "finite_fraction": "finite_fraction",
        "identity_full": "identity_full_max_abs_margin_drift",
        "identity_first": "identity_first_max_abs_margin_drift",
        "baseline_valid": "full_baseline_valid_fraction",
        "endpoint_flip": "full_main_endpoint_flip_fraction",
        "original_flip": "full_original_endpoint_flip_fraction",
        "swapped_flip": "full_swapped_endpoint_flip_fraction",
        "positive_change": "full_main_positive_change_fraction",
        "main_span": "full_main_margin_change_median",
        "control_span": "full_control_abs_margin_change_median",
        "span_ratio": "full_main_to_control_span_ratio",
        "control_flip": "full_control_endpoint_flip_fraction",
        "first_count": "first_informative_main_count",
        "first_baseline_valid": "first_baseline_valid_fraction",
        "first_endpoint_flip": "first_endpoint_flip_fraction",
        "first_positive_change": "first_positive_change_fraction",
    }
    for independent_key, recorded_key in mapping.items():
        checks[f"{prefix}_{independent_key}"] = close(
            independent[independent_key],
            recorded[recorded_key],
        )
    checks[f"{prefix}_qualified"] = (
        independent["qualified"] == recorded["qualified"]
    )


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    checks: dict[str, bool] = {
        "protocol_audit_passed": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest_matches_audit": (
            prereg["protocol_digest"] == protocol_audit["protocol_digest"]
        ),
        "protocol_digest_recomputed": (
            prereg["protocol_digest"]
            == digest({key: value for key, value in prereg.items() if key != "protocol_digest"})
        ),
        "cohorts_disjoint": set(prereg["material"]["cohorts"]["discovery"]).isdisjoint(
            prereg["material"]["cohorts"]["confirmation"]
        ),
        "cohorts_24_each": all(
            len(prereg["material"]["cohorts"][split]) == 24 for split in SPLITS
        ),
        "reserve_63": len(prereg["material"]["reserve_item_ids"]) == 63,
        "selection_protocol_digest": (
            selection["protocol_digest"] == prereg["protocol_digest"]
        ),
        "final_protocol_digest": final["protocol_digest"] == prereg["protocol_digest"],
        "final_selection_digest": (
            final["selection_digest"] == selection["selection_digest"]
        ),
        "cross_architecture_claim_denied": not final["cross_architecture_claim_authorized"],
        "semantic_module_claim_denied": not final["semantic_module_claim_authorized"],
    }
    independent: dict[str, Any] = {}
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    splits_to_check = ["discovery"]
    if selection["confirmation_authorized"]:
        splits_to_check.append("confirmation")

    for split in splits_to_check:
        for model in MODELS:
            root = OUT_ROOT / "runs" / split / model
            summary = read_json(root / "summary.json")
            records = read_jsonl(root / "records.jsonl")
            summaries[(split, model)] = summary
            prefix = f"{split}_{model}"
            checks[f"{prefix}_record_count"] = len(records) == EXPECTED_RECORDS
            checks[f"{prefix}_summary_record_count"] = (
                summary["record_count"] == EXPECTED_RECORDS
            )
            checks[f"{prefix}_record_digest"] = (
                digest(records) == summary["record_digest"]
            )
            checks[f"{prefix}_summary_digest"] = (
                summary["summary_digest"]
                == digest({
                    key: value
                    for key, value in summary.items()
                    if key != "summary_digest"
                })
            )
            checks[f"{prefix}_protocol_digest"] = (
                summary["protocol_digest"] == prereg["protocol_digest"]
            )
            checks[f"{prefix}_scope_set"] = {
                row["scope"] for row in records
            } == set(SCOPES)
            checks[f"{prefix}_alpha_set"] = {
                float(row["alpha"]) for row in records
            } == {0.0, 1.0}
            independent[prefix] = {}
            for scope in SCOPES:
                result = independent_scope(records, scope)
                independent[prefix][scope] = result
                compare_scope(
                    f"{prefix}_{scope}",
                    result,
                    summary["metrics"][scope],
                    checks,
                )
            improvement = (
                independent[prefix]["candidate_prediction_span"]["endpoint_flip"]
                - independent[prefix]["answer_boundary"]["endpoint_flip"]
            )
            checks[f"{prefix}_improvement"] = close(
                improvement,
                summary["metrics"]["span_minus_boundary_endpoint_flip"],
            )
            rescue = bool(
                independent[prefix]["candidate_prediction_span"]["qualified"]
                and not independent[prefix]["answer_boundary"]["qualified"]
                and improvement >= THRESHOLDS["span_rescue_min_improvement"]
            )
            checks[f"{prefix}_rescue"] = rescue == summary["metrics"]["span_rescue"]

    discovery_boundary = all(
        independent[f"discovery_{model}"]["answer_boundary"]["qualified"]
        for model in MODELS
    )
    discovery_span = all(
        independent[f"discovery_{model}"]["candidate_prediction_span"]["qualified"]
        for model in MODELS
    )
    expected_scope = (
        "answer_boundary"
        if discovery_boundary
        else "candidate_prediction_span"
        if discovery_span
        else None
    )
    checks["selection_scope_recomputed"] = selection["selected_scope"] == expected_scope
    checks["selection_confirmation_recomputed"] = (
        selection["confirmation_authorized"] == (expected_scope is not None)
    )
    checks["selection_digest_recomputed"] = (
        selection["selection_digest"]
        == digest({
            key: value for key, value in selection.items() if key != "selection_digest"
        })
    )

    if expected_scope is None:
        expected_confirmed = False
    else:
        expected_confirmed = all(
            independent[f"confirmation_{model}"][expected_scope]["qualified"]
            for model in MODELS
        )
    checks["final_confirmed_recomputed"] = (
        final["minimal_scope_sufficiency_confirmed"] == expected_confirmed
    )
    checks["final_component_authorization_recomputed"] = (
        final["component_mediation_authorized"] == expected_confirmed
    )
    checks["final_auto_continue_recomputed"] = (
        final["auto_continue"] == expected_confirmed
    )
    checks["final_digest_recomputed"] = (
        final["final_digest"]
        == digest({key: value for key, value in final.items() if key != "final_digest"})
    )

    core = {
        "schema_version": "phase1140_independent_result_audit.v1",
        "phase": 1140,
        "protocol_digest": prereg["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "final_digest": final["final_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "independent_recomputation": independent,
    }
    audit = dict(core)
    audit["audit_digest"] = digest(core)
    write_json(OUT_ROOT / "audit/independent_result_audit.json", audit)
    print(json.dumps({
        "phase": 1140,
        "checks": f"{audit['passed_count']}/{audit['check_count']}",
        "all_checks_passed": audit["all_checks_passed"],
        "audit_digest": audit["audit_digest"],
    }), flush=True)
    if not audit["all_checks_passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Phase1140 independent audit failed: {failed}")


if __name__ == "__main__":
    main()
