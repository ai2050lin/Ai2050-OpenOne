#!/usr/bin/env python3
"""Aggregate and judge the frozen Phase1128 numerical-formation audit."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1128_fp16_numeric_formation_protocol as protocol


def median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def failure_type(row: dict[str, Any]) -> str:
    candidate = bool(row["source_candidate_finite"])
    suffix = bool(row["source_suffix_finite"])
    if candidate and suffix:
        return "none"
    if not candidate and not suffix:
        return "both"
    return "candidate_only" if not candidate else "suffix_only"


def model_result(model_name: str, prereg: dict[str, Any]) -> dict[str, Any]:
    root = protocol.OUT_ROOT / "scan" / model_name
    summary = protocol.read_json(root / "summary.json")
    cases = protocol.read_jsonl(root / "cases.jsonl")
    events = protocol.read_jsonl(root / "events.jsonl")
    if protocol.digest(cases) != summary["case_detail_digest"]:
        raise RuntimeError(f"Case digest mismatch: {model_name}")
    if protocol.digest(events) != summary["event_digest"]:
        raise RuntimeError(f"Event digest mismatch: {model_name}")

    source_nonfinite = [row for row in cases if not row["source_total_finite"]]
    source_finite = [row for row in cases if row["source_total_finite"]]
    first_exact = Counter(row["first_nonfinite_name"] for row in source_nonfinite if row["first_nonfinite_name"])
    first_class = Counter(row["first_nonfinite_class"] for row in source_nonfinite if row["first_nonfinite_class"])
    first_layers = [int(row["first_nonfinite_layer"]) for row in source_nonfinite if row["first_nonfinite_layer"] is not None]
    first_depths = [float(row["first_nonfinite_relative_depth"]) for row in source_nonfinite
                    if row["first_nonfinite_relative_depth"] is not None]
    dominant_name, dominant_count = first_exact.most_common(1)[0] if first_exact else (None, 0)
    dominant_class = first_class.most_common(1)[0][0] if first_class else None
    dominance = dominant_count / len(source_nonfinite) if source_nonfinite else None
    failure_types = Counter(failure_type(row) for row in cases)

    event_maxima: dict[str, list[float]] = defaultdict(list)
    event_nonfinite_cases: Counter[str] = Counter()
    for event in events:
        if event["max_abs_finite"] is not None:
            event_maxima[event["event_name"]].append(float(event["max_abs_finite"]))
        if int(event["nonfinite_count"]) > 0:
            event_nonfinite_cases[event["event_name"]] += 1
    profile = [{
        "event_name": name,
        "median_max_abs_finite": median_or_none(values),
        "nonfinite_case_count": event_nonfinite_cases[name],
    } for name, values in sorted(event_maxima.items(), key=lambda item: next(
        event["order"] for event in prereg["model_specs"][model_name]["event_registry"]
        if event["name"] == item[0]
    ))]

    gate = prereg["automatic_refinement_gate"]
    auto_refinement = (
        len(source_nonfinite) >= int(gate["minimum_source_nonfinite_cases"])
        and dominance is not None
        and dominance >= float(gate["minimum_same_exact_event_fraction"])
        and dominant_class in gate["allowed_event_classes"]
        and all(row["candidate_finite_parity"] and row["suffix_finite_parity"] and row["total_finite_parity"]
                for row in cases)
    )
    return {
        "source_finite_count": len(source_finite),
        "source_nonfinite_count": len(source_nonfinite),
        "rerun_finite_count": sum(bool(row["rerun_total_finite"]) for row in cases),
        "component_finite_parity": {
            component: sum(bool(row[f"{component}_finite_parity"]) for row in cases)
            for component in ("candidate", "suffix", "total")
        },
        "event_count_valid_cases": sum(bool(row["event_count_expected"]) for row in cases),
        "source_nonfinite_localized_count": sum(bool(row["first_nonfinite_name"]) for row in source_nonfinite),
        "source_finite_with_tracked_nonfinite": sum(bool(row["any_tracked_nonfinite"]) for row in source_finite),
        "failure_type_counts": dict(sorted(failure_types.items())),
        "first_event_exact_counts": dict(first_exact.most_common()),
        "first_event_class_counts": dict(first_class.most_common()),
        "dominant_first_event": dominant_name,
        "dominant_first_event_class": dominant_class,
        "dominant_first_event_count": dominant_count,
        "dominant_first_event_fraction": dominance,
        "first_event_layer_median": median_or_none([float(value) for value in first_layers]),
        "first_event_relative_depth_median": median_or_none(first_depths),
        "event_magnitude_profile": profile,
        "automatic_refinement_gate_passed": auto_refinement,
        "scan_summary": summary,
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not protocol_audit["passed"]:
        raise RuntimeError("Phase1128 protocol audit failed")

    results = {model: model_result(model, prereg) for model in protocol.MODELS}
    exact_parity = all(
        all(count == 320 for count in result["component_finite_parity"].values())
        for result in results.values()
    )
    localization_complete = all(
        result["source_nonfinite_localized_count"] == result["source_nonfinite_count"]
        for result in results.values()
    )
    qwen_healthy = (
        results["qwen3"]["source_nonfinite_count"] == 0
        and results["qwen3"]["source_finite_with_tracked_nonfinite"] == 0
    )
    auto_models = [model for model, result in results.items() if result["automatic_refinement_gate_passed"]]
    predictions = {
        "P1_identity_and_protocol": protocol_audit["passed"],
        "P2_exact_source_finite_parity": exact_parity,
        "P3_all_source_nonfinite_cases_localized": localization_complete,
        "P4_qwen3_healthy_reference": qwen_healthy,
        "P5_first_event_concentration_is_descriptive": True,
        "P6_no_semantic_or_behavioral_mechanism_claim": True,
    }
    auto_continue = {
        "value": bool(auto_models),
        "authorized_models": auto_models,
        "authorized_scope": "one separately frozen numerical subcomponent audit" if auto_models else None,
        "reason": (
            "The frozen exact-event concentration gate passed."
            if auto_models else
            "No failing model met the frozen large-sample, 90%-concentration, refinable-component gate."
        ),
    }
    final_core = {
        "schema_version": "phase1128_fp16_numeric_formation_final.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": protocol_audit["audit_digest"],
        "model_results": results,
        "predictions": predictions,
        "automatic_refinement": auto_continue,
        "interpretation_boundary": (
            "The result localizes FP16 numerical formation at previously scored positions only. It neither rescues "
            "Phase1126 nor identifies semantic content, computation, necessity, or a preferred model precision."
        ),
    }
    final = dict(final_core)
    final["final_digest"] = protocol.digest(final_core)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
