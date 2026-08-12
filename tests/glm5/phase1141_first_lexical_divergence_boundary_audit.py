#!/usr/bin/env python3
"""Independent raw-record and decision audit for Phase1141."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1141_first_lexical_divergence_boundary"
SCRIPT = ROOT / "tests/glm5/phase1141_first_lexical_divergence_boundary.py"
MODELS = ("qwen3_4b", "qwen3_14b")
SPLITS = ("discovery", "confirmation")
SCOPES = ("answer_boundary", "first_lexical_divergence", "candidate_prediction_span")
EXPECTED_RECORDS = 1080
EPSILON = 1e-8
THRESHOLDS = {
    "finite_fraction": 0.99,
    "identity_max_abs_margin_drift": 0.005,
    "baseline_valid_fraction": 0.99,
    "main_endpoint_flip_fraction": 0.95,
    "stratum_endpoint_flip_fraction": 0.95,
    "panel_endpoint_flip_fraction": 0.95,
    "main_positive_change_fraction": 0.99,
    "same_answer_control_flip_fraction": 0.10,
    "cross_item_control_flip_fraction": 0.10,
    "main_to_each_control_ratio": 2.0,
    "shared_minus_boundary_min": 0.50,
    "shared_span_noninferiority_margin": 0.05,
    "lcp0_scope_equivalence_max_abs_margin": 0.005,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def raw_curves(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["scope"] == scope:
            grouped[str(row["curve_id"])].append(row)
    result = []
    for curve_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        if [float(row["alpha"]) for row in ordered] != [0.0, 1.0]:
            raise RuntimeError(f"incomplete alpha grid for {curve_id}")
        base, endpoint = ordered
        finite = bool(base["finite"] and endpoint["finite"])
        before = base["full_oriented_margin"]
        after = endpoint["full_oriented_margin"]
        change = float(after) - float(before) if finite and before is not None and after is not None else None
        result.append({
            "curve_id": curve_id,
            "kind": str(base["curve_kind"]),
            "panel": str(base["panel"]),
            "stratum": str(base["stratum"]),
            "item_id": str(base["item_id"]),
            "source_item_id": str(base["source_item_id"]),
            "lcp": int(base["common_prefix_length"]),
            "finite": finite,
            "identity_full": base["identity_full_margin_drift"],
            "identity_decision": base["identity_decision_margin_drift"],
            "before": before,
            "after": after,
            "change": change,
            "baseline_valid": bool(finite and before is not None and before < 0),
            "endpoint_flip": bool(finite and after is not None and after > 0),
            "positive": bool(change is not None and change > 0),
        })
    return result


def fraction(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def aggregate(records: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    curves = raw_curves(records, scope)
    main = [row for row in curves if row["kind"] == "main"]
    temporal = [row for row in curves if row["kind"] == "same_answer_temporal_control"]
    cross = [row for row in curves if row["kind"] == "cross_item_wrong_donor_control"]
    scope_records = [row for row in records if row["scope"] == scope]
    identity_full = [abs(float(row["identity_full"])) for row in curves if row["identity_full"] is not None]
    identity_decision = [abs(float(row["identity_decision"])) for row in curves if row["identity_decision"] is not None]
    main_change = median(row["change"] for row in main)
    temporal_change = median(abs(float(row["change"])) for row in temporal if row["change"] is not None)
    cross_change = median(abs(float(row["change"])) for row in cross if row["change"] is not None)
    return {
        "record_count": len(scope_records),
        "curve_count": len(curves),
        "main_count": len(main),
        "temporal_count": len(temporal),
        "cross_count": len(cross),
        "finite": sum(bool(row["finite"]) for row in scope_records) / max(len(scope_records), 1),
        "identity_full": max(identity_full) if identity_full else None,
        "identity_decision": max(identity_decision) if identity_decision else None,
        "baseline": fraction(main, "baseline_valid"),
        "endpoint": fraction(main, "endpoint_flip"),
        "original": fraction([row for row in main if row["panel"] == "original"], "endpoint_flip"),
        "swapped": fraction([row for row in main if row["panel"] == "swapped"], "endpoint_flip"),
        "positive": fraction(main, "positive"),
        "main_change": main_change,
        "temporal_change": temporal_change,
        "cross_change": cross_change,
        "main_temporal_ratio": main_change / max(temporal_change, EPSILON) if main_change is not None and temporal_change is not None else None,
        "main_cross_ratio": main_change / max(cross_change, EPSILON) if main_change is not None and cross_change is not None else None,
        "temporal_flip": fraction(temporal, "endpoint_flip"),
        "cross_flip": fraction(cross, "endpoint_flip"),
        "strata": {
            stratum: fraction([row for row in main if row["stratum"] == stratum], "endpoint_flip")
            for stratum in ("shared_prefix_p54", "immediate_p54", "immediate_other")
        },
    }


def independently_qualify(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = {scope: aggregate(records, scope) for scope in SCOPES}
    decision = values["first_lexical_divergence"]
    boundary = values["answer_boundary"]
    span = values["candidate_prediction_span"]
    answer_lcp0 = {
        (str(row["curve_id"]), float(row["alpha"])): row
        for row in records
        if row["scope"] == "answer_boundary" and row["curve_kind"] == "main" and int(row["common_prefix_length"]) == 0
    }
    decision_lcp0 = {
        (str(row["curve_id"]), float(row["alpha"])): row
        for row in records
        if row["scope"] == "first_lexical_divergence" and row["curve_kind"] == "main" and int(row["common_prefix_length"]) == 0
    }
    if set(answer_lcp0) != set(decision_lcp0):
        raise RuntimeError("LCP0 key mismatch")
    lcp0_max = max(
        abs(float(answer_lcp0[key]["full_oriented_margin"]) - float(decision_lcp0[key]["full_oriented_margin"]))
        for key in answer_lcp0
    )
    shared_gain = decision["strata"]["shared_prefix_p54"] - boundary["strata"]["shared_prefix_p54"]
    shared_minus_span = decision["strata"]["shared_prefix_p54"] - span["strata"]["shared_prefix_p54"]
    gates = {
        "finite": decision["finite"] >= THRESHOLDS["finite_fraction"],
        "identity_full": decision["identity_full"] is not None and decision["identity_full"] <= THRESHOLDS["identity_max_abs_margin_drift"],
        "identity_decision": decision["identity_decision"] is not None and decision["identity_decision"] <= THRESHOLDS["identity_max_abs_margin_drift"],
        "baseline_valid": decision["baseline"] >= THRESHOLDS["baseline_valid_fraction"],
        "main_endpoint": decision["endpoint"] >= THRESHOLDS["main_endpoint_flip_fraction"],
        "each_stratum": all(value >= THRESHOLDS["stratum_endpoint_flip_fraction"] for value in decision["strata"].values()),
        "original_panel": decision["original"] >= THRESHOLDS["panel_endpoint_flip_fraction"],
        "swapped_panel": decision["swapped"] >= THRESHOLDS["panel_endpoint_flip_fraction"],
        "positive_change": decision["positive"] >= THRESHOLDS["main_positive_change_fraction"],
        "same_answer_flip": decision["temporal_flip"] <= THRESHOLDS["same_answer_control_flip_fraction"],
        "cross_item_flip": decision["cross_flip"] <= THRESHOLDS["cross_item_control_flip_fraction"],
        "same_answer_ratio": decision["main_temporal_ratio"] is not None and decision["main_temporal_ratio"] >= THRESHOLDS["main_to_each_control_ratio"],
        "cross_item_ratio": decision["main_cross_ratio"] is not None and decision["main_cross_ratio"] >= THRESHOLDS["main_to_each_control_ratio"],
        "shared_boundary_improvement": shared_gain >= THRESHOLDS["shared_minus_boundary_min"],
        "shared_span_noninferiority": shared_minus_span >= -THRESHOLDS["shared_span_noninferiority_margin"],
        "lcp0_scope_equivalence": lcp0_max <= THRESHOLDS["lcp0_scope_equivalence_max_abs_margin"],
    }
    return {
        "scopes": values,
        "shared_gain": shared_gain,
        "shared_minus_span": shared_minus_span,
        "lcp0_max": lcp0_max,
        "gate_checks": gates,
        "qualified": all(gates.values()),
    }


def compare_run(prefix: str, independent: dict[str, Any], recorded: dict[str, Any], checks: dict[str, bool]) -> None:
    checks[f"{prefix}_qualified"] = independent["qualified"] == recorded["qualified"]
    checks[f"{prefix}_gates"] = independent["gate_checks"] == recorded["gate_checks"]
    checks[f"{prefix}_shared_gain"] = close(independent["shared_gain"], recorded["shared_first_divergence_minus_boundary"])
    checks[f"{prefix}_shared_span"] = close(independent["shared_minus_span"], recorded["shared_first_divergence_minus_span"])
    checks[f"{prefix}_lcp0"] = close(independent["lcp0_max"], recorded["lcp0_boundary_vs_divergence_max_abs_margin"])
    mapping = {
        "record_count": "record_count",
        "curve_count": "curve_count",
        "finite": "finite_fraction",
        "identity_full": "identity_full_max_abs_margin_drift",
        "identity_decision": "identity_decision_max_abs_margin_drift",
        "baseline": "full_baseline_valid_fraction",
        "endpoint": "full_main_endpoint_flip_fraction",
        "original": "full_original_endpoint_flip_fraction",
        "swapped": "full_swapped_endpoint_flip_fraction",
        "positive": "full_main_positive_change_fraction",
        "main_change": "full_main_margin_change_median",
        "temporal_change": "full_same_answer_abs_change_median",
        "cross_change": "full_cross_item_abs_change_median",
        "main_temporal_ratio": "full_main_to_same_answer_ratio",
        "main_cross_ratio": "full_main_to_cross_item_ratio",
        "temporal_flip": "full_same_answer_endpoint_flip_fraction",
        "cross_flip": "full_cross_item_endpoint_flip_fraction",
    }
    for scope in SCOPES:
        for left, right in mapping.items():
            checks[f"{prefix}_{scope}_{left}"] = close(independent["scopes"][scope][left], recorded[scope][right])
        checks[f"{prefix}_{scope}_strata"] = all(
            close(independent["scopes"][scope]["strata"][key], recorded[scope]["stratum_endpoint_flip_fraction"][key])
            for key in independent["scopes"][scope]["strata"]
        )


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    checks: dict[str, bool] = {
        "protocol_audit": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest_audit": prereg["protocol_digest"] == protocol_audit["protocol_digest"],
        "protocol_digest_recomputed": prereg["protocol_digest"] == digest({key: value for key, value in prereg.items() if key != "protocol_digest"}),
        "script_hash_frozen": prereg["source"]["script_sha256"] == sha256_file(SCRIPT),
        "cohorts_30": all(len(prereg["material"]["cohorts"][split]) == 30 for split in SPLITS),
        "cohorts_disjoint": set(prereg["material"]["cohorts"]["discovery"]).isdisjoint(prereg["material"]["cohorts"]["confirmation"]),
        "reserve_18": len(prereg["material"]["reserve_item_ids"]) == 18,
        "reserve_disjoint": set(prereg["material"]["reserve_item_ids"]).isdisjoint(set(prereg["material"]["cohorts"]["discovery"]) | set(prereg["material"]["cohorts"]["confirmation"])),
        "selection_protocol": selection["protocol_digest"] == prereg["protocol_digest"],
        "final_protocol": final["protocol_digest"] == prereg["protocol_digest"],
        "final_selection": final["selection_digest"] == selection["selection_digest"],
        "semantic_claim_denied": not final["semantic_boundary_claim_authorized"],
        "necessity_claim_denied": not final["necessity_claim_authorized"],
        "cross_architecture_claim_denied": not final["cross_architecture_claim_authorized"],
    }
    independent: dict[str, Any] = {}
    splits = ["discovery"] + (["confirmation"] if selection["confirmation_authorized"] else [])
    for split in splits:
        for model in MODELS:
            root = OUT_ROOT / "runs" / split / model
            records = read_jsonl(root / "records.jsonl")
            summary = read_json(root / "summary.json")
            prefix = f"{split}_{model}"
            checks[f"{prefix}_record_count"] = len(records) == EXPECTED_RECORDS == summary["record_count"]
            checks[f"{prefix}_record_digest"] = digest(records) == summary["record_digest"]
            checks[f"{prefix}_summary_digest"] = summary["summary_digest"] == digest({key: value for key, value in summary.items() if key != "summary_digest"})
            checks[f"{prefix}_protocol"] = summary["protocol_digest"] == prereg["protocol_digest"]
            checks[f"{prefix}_scopes"] = {str(row["scope"]) for row in records} == set(SCOPES)
            checks[f"{prefix}_alphas"] = {float(row["alpha"]) for row in records} == {0.0, 1.0}
            checks[f"{prefix}_cohort"] = {str(row["item_id"]) for row in records} == set(prereg["material"]["cohorts"][split])
            checks[f"{prefix}_wrong_donors_differ"] = all(
                row["item_id"] != row["source_item_id"]
                for row in records if row["curve_kind"] == "cross_item_wrong_donor_control"
            )
            checks[f"{prefix}_control_scope"] = all(
                row["scope"] == "first_lexical_divergence"
                for row in records if row["curve_kind"] != "main"
            )
            checks[f"{prefix}_span_same_item"] = all(
                row["item_id"] == row["source_item_id"]
                for row in records if row["scope"] == "candidate_prediction_span"
            )
            result = independently_qualify(records)
            independent[prefix] = result
            compare_run(prefix, result, summary["metrics"], checks)

    discovery_pass = {
        model: independent[f"discovery_{model}"]["qualified"] for model in MODELS
    }
    expected_authorized = all(discovery_pass.values())
    checks["selection_qualified"] = selection["qualified"] == discovery_pass
    checks["selection_authorized"] = selection["confirmation_authorized"] == expected_authorized
    checks["selection_scope"] = selection["selected_scope"] == ("first_lexical_divergence" if expected_authorized else None)
    checks["selection_digest"] = selection["selection_digest"] == digest({key: value for key, value in selection.items() if key != "selection_digest"})
    expected_confirmed = bool(
        expected_authorized
        and all(independent[f"confirmation_{model}"]["qualified"] for model in MODELS)
    )
    checks["final_confirmed"] = final["first_lexical_divergence_sufficiency_confirmed"] == expected_confirmed
    checks["final_component"] = final["component_mediation_authorized"] == expected_confirmed
    checks["final_auto_continue"] = final["auto_continue"] == expected_confirmed
    checks["confirmation_untouched_when_denied"] = expected_authorized or not (OUT_ROOT / "runs/confirmation").exists()
    checks["final_digest"] = final["final_digest"] == digest({key: value for key, value in final.items() if key != "final_digest"})

    core = {
        "schema_version": "phase1141_independent_result_audit.v1",
        "phase": 1141,
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
        "phase": 1141,
        "checks": f"{audit['passed_count']}/{audit['check_count']}",
        "all_checks_passed": audit["all_checks_passed"],
        "audit_digest": audit["audit_digest"],
    }), flush=True)


if __name__ == "__main__":
    main()
