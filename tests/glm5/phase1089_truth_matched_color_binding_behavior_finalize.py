#!/usr/bin/env python3
"""Authorize Phase1089 hidden scan from panel-specific behavior."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1089_truth_matched_color_binding_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    static = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    models = {}
    passing_models = []
    for model_name in protocol.MODELS:
        pilot = protocol.read_json(
            protocol.OUT_ROOT / "pilot" / f"{model_name}.json"
        )
        cases = {
            int(row["case_index"]): row for row in protocol.read_jsonl(
                protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
            )
        }
        detail = protocol.read_jsonl(
            protocol.OUT_ROOT / "pilot" / f"candidate.{model_name}.jsonl"
        )
        totals: Counter = Counter()
        hits: Counter = Counter()
        finite: Counter = Counter()
        for row in detail:
            case = cases[int(row["case_index"])]
            key = (case["operation"], case["world"], case["panel"])
            totals[key] += 1
            hits[key] += int(row["hit"])
            finite[key] += int(row["finite"])
        passing_worlds: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        per_cell = {}
        for operation in protocol.OPERATIONS:
            for world in protocol.WORLDS:
                for panel in protocol.PANELS:
                    key = (operation, world, panel)
                    total = totals[key]
                    accuracy = hits[key] / total if total else 0.0
                    finite_fraction = finite[key] / total if total else 0.0
                    threshold = (
                        prereg["evidence_thresholds"]
                        ["candidate_accuracy_for_operation_behavior"]
                        if panel == "active" else
                        prereg["evidence_thresholds"]
                        ["minimum_null_candidate_accuracy"]
                    )
                    passed = accuracy >= threshold and finite_fraction >= 0.95
                    if passed:
                        passing_worlds[operation][panel].append(world)
                    per_cell[f"{operation}__{world}__{panel}"] = {
                        "count": total,
                        "accuracy": accuracy,
                        "finite_fraction": finite_fraction,
                        "passed": passed,
                    }
        minimum_worlds = int(
            prereg["evidence_thresholds"]
            ["minimum_behavior_worlds_per_operation"]
        )
        passing_operations = [
            operation for operation in protocol.OPERATIONS
            if all(
                len(passing_worlds[operation][panel]) >= minimum_worlds
                for panel in protocol.PANELS
            )
        ]
        passed = (
            len(passing_operations) >= int(
                prereg["evidence_thresholds"]["minimum_behavior_operations"]
            )
            and pilot["candidate_finite_fraction"] >= prereg[
                "evidence_thresholds"
            ]["minimum_candidate_finite_fraction"]
            and pilot["precision"]["has_fp16_parameters"]
            and not pilot["precision"]["has_bf16_parameters"]
            and not pilot["precision"]["has_quantized_modules"]
        )
        if passed:
            passing_models.append(model_name)
        models[model_name] = {
            "passed": passed,
            "passing_operations": passing_operations,
            "passing_operation_count": len(passing_operations),
            "passing_worlds": {
                operation: dict(panels)
                for operation, panels in passing_worlds.items()
            },
            "per_cell": per_cell,
            "candidate_finite_fraction": pilot["candidate_finite_fraction"],
            "precision": pilot["precision"],
            "elapsed_seconds": pilot["elapsed_seconds"],
            "result_digest": pilot["result_digest"],
        }
    minimum_models = int(
        prereg["evidence_thresholds"]["minimum_behavior_models"]
    )
    p1 = bool(static["all_checks_passed"])
    p2 = len(passing_models) >= minimum_models
    result = {
        "schema_version": "phase1089_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "predictions": {
            "P1": {"passed": p1},
            "P2": {"passed": p2, "passing_models": passing_models},
        },
        "models": models,
        "hidden_scan_authorized": p1 and p2,
        "full_atlas_authorized": False,
        "causal_authorized": False,
        "reason": (
            "Only the preregistered middle-band signed scan is authorized. "
            "Both active and truth-matched-null panels were behavior-gated."
        ),
    }
    result["authorization_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result
    )
    print({
        "phase": protocol.PHASE,
        "passing_models": passing_models,
        "hidden_scan_authorized": result["hidden_scan_authorized"],
        "authorization_digest": result["authorization_digest"],
    })


if __name__ == "__main__":
    main()
