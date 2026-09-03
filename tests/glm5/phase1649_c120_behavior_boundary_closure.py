#!/usr/bin/env python3
"""Close C120 at its registered behavioral boundary without opening HiddenState data."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1647_c120_controlled_comparison_observation_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def accuracy(rows: list[dict]) -> float:
    return sum(row["correct"] for row in rows) / len(rows)


if __name__ == "__main__":
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    capture_audit = core.load(OUT / "audit/independent_capture_audit.json")
    rows = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    if not capture_audit["all_checks_passed"] or capture["behavior_gate_passed"]:
        raise RuntimeError("C120 behavior-boundary branch mismatch")
    cells = []
    for partition in ("discovery", "confirmation", "lockbox"):
        for dimension in ("length", "width", "weight"):
            for truth in (1, -1):
                for gap in (1, -1):
                    for surface in (1, -1):
                        for output_format in (1, -1):
                            selected = [
                                row for row in rows
                                if row["partition"] == partition
                                and row["dimension"] == dimension
                                and row["truth_factor"] == truth
                                and row["gap_factor"] == gap
                                and row["surface_factor"] == surface
                                and row["output_format"] == output_format
                            ]
                            cells.append({
                                "partition": partition,
                                "dimension": dimension,
                                "truth_factor": truth,
                                "gap_factor": gap,
                                "surface_factor": surface,
                                "output_format": output_format,
                                "n": len(selected),
                                "accuracy": accuracy(selected),
                                "mean_positive_minus_negative": sum(row["positive_minus_negative"] for row in selected) / len(selected),
                            })
    marginal = {}
    for field, values in (
        ("dimension", ("length", "width", "weight")),
        ("truth_factor", (1, -1)),
        ("gap_factor", (1, -1)),
        ("surface_factor", (1, -1)),
        ("output_format", (1, -1)),
        ("partition", ("discovery", "confirmation", "lockbox")),
    ):
        marginal[field] = {
            str(value): {
                "n": sum(row[field] == value for row in rows),
                "accuracy": accuracy([row for row in rows if row[field] == value]),
            }
            for value in values
        }
    diagnostic = {
        "phase": 1649,
        "campaign": "C120",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "behavior_failure_factorially_diagnosed_hidden_states_sealed",
        "marginal_accuracy": marginal,
        "factor_cells": cells,
        "strict_adjudication": {
            "observed": (
                "The controlled exact-score interface is above chance but fails every registered aggregate "
                "behavior gate. Far-gap cases exceed near-gap cases, weight exceeds width and length, and "
                "positive and negative truth cases differ."
            ),
            "not_identified": (
                "These behavioral asymmetries do not locate a neural cause and do not show that comparison "
                "information is absent from the sealed embedding/HiddenState archive."
            ),
            "forbidden_claims": [
                "Qwen3 lacks numeric comparison",
                "comparison relations have no HiddenState field",
                "length, width and weight share or do not share a comparator",
                "any activation coordinate is causal or semantic",
            ],
        },
        "input_hashes": {
            "protocol": core.sha(OUT / "protocol/preregistration.json"),
            "capture": core.sha(OUT / "analysis/capture_summary.json"),
            "behavior_index": core.sha(OUT / "raw/qwen3_behavior_index.jsonl"),
            "sealed_raw": capture["raw_sha256"],
        },
        "authorization": (
            "close_C120_without_hidden_state_analysis_and_execute_C121_fresh_structured_comparison_qualification"
        ),
    }
    core.save(OUT / "analysis/behavior_boundary_diagnostic.json", diagnostic)
    closure = {
        "phase": 1649,
        "campaign": "C120",
        "created_at_utc": diagnostic["created_at_utc"],
        "status": "behavior_gate_failed_hidden_state_route_closed",
        "headline": capture["behavior"],
        "gate_checks": capture["behavior_gate_checks"],
        "strict_conclusion": (
            "C120's controlled word-score prompt did not behaviorally qualify any registered comparison "
            "dimension. The run is a valid executor-interface boundary, not a HiddenState mechanism result."
        ),
        "raw_archive_status": (
            "The unified CUDA forward produced a reproducibility archive, but all embedding/HiddenState "
            "scientific analysis and heatmap export are sealed by the frozen behavior gate."
        ),
        "new_puzzles": {
            "K312-BOUNDARY": (
                "exact-score comparison performance is strongly conditioned by dimension, numeric gap and "
                "truth polarity under the C120 wording; no internal comparison-field claim was tested"
            )
        },
        "theory_update": (
            "No mechanism term is added. RDC keeps natural state H, researcher contrast R and intervention "
            "object Gamma separate; C120 never qualified construction of R."
        ),
        "problems": [
            "one Qwen3 and controlled English",
            "machine naturalness only",
            "number words rather than digits",
            "candidate selection rather than free generation",
            "near comparisons are especially weak",
            "all HiddenState claims are untested",
        ],
        "claim_boundary": protocol["claim_boundary"],
        "next_authorization": diagnostic["authorization"],
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "capture_audit": capture_audit["all_checks_passed"],
        "gate_failed": not capture["behavior_gate_passed"],
        "all_aggregate_checks_failed": not any(capture["behavior_gate_checks"].values()),
        "cells": len(cells) == 144 and all(row["n"] == 8 for row in cells),
        "marginals": all(sum(value["n"] for value in group.values()) == 1152 for group in marginal.values()),
        "sealed": "sealed" in closure["raw_archive_status"],
        "no_heatmap": not (OUT / "visualization").exists(),
        "authorization": closure["next_authorization"].startswith("close_C120"),
    }
    report = {
        "phase": 1649,
        "campaign": "C120",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": closure["next_authorization"],
    }
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"closure": closure, "audit": report}, indent=2))
