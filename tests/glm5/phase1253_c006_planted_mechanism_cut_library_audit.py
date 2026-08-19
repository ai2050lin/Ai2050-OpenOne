#!/usr/bin/env python3
"""Independent audit for Phase1253."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import phase1253_c006_planted_mechanism_cut_library as main


def write(value: dict[str, Any], path: Path) -> None:
    main.atomic_json(path, value)
    print(main.canonical_json(value))


def preaudit() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    protocol = main.read_json(main.PROTOCOL)
    public = main.read_jsonl(main.PUBLIC_SYSTEMS)
    truth = main.read_jsonl(main.SEALED_TRUTH)
    expected = main.protocol_payload(public, truth)
    checks["source_hashes"] = protocol["source_hashes"] == expected["source_hashes"]
    checks["protocol_digest"] = protocol["protocol_digest"] == expected["protocol_digest"]
    checks["material_digests"] = protocol["public_digest"] == main.digest(public) and protocol["sealed_truth_digest"] == main.digest(truth)
    checks["system_count"] = len(public) == len(truth) == len(main.SPLITS) * main.TASKS_PER_SPLIT * main.GAUGES_PER_TASK * len(main.MECHANISMS)
    checks["split_balance"] = all(sum(row["split"] == split for row in public) == len(public) // 2 for split in main.SPLITS)
    checks["mechanism_balance"] = all(sum(row["mechanism"] == mechanism for row in truth) == len(truth) // len(main.MECHANISMS) for mechanism in main.MECHANISMS)
    checks["opaque_edge_bijection"] = all(len(set(row["edge_map"].values())) == len(main.EDGES) for row in truth)
    checks["public_role_sealed"] = all("mechanism" not in row and "edge_map" not in row and "planted_cuts" not in row for row in public)
    checks["mechanism_truth_complete"] = all(
        set(row["active_edges"]) == set(main.active_edges(row["mechanism"]))
        and {tuple(sorted(cut)) for cut in row["planted_cuts"]} == set(main.planted_cuts(row["mechanism"]))
        for row in truth
    )
    checks["one_shot_clean"] = not main.COMPLETE.exists() and not main.RAW.exists() and not main.DETAILS.exists()
    result = {
        "phase": main.PHASE,
        "audit_stage": "preaudit",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = main.digest(result)
    return result


def final_audit() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    protocol, public, truth = main.verify_protocol()
    raw = main.read_json(main.RAW)
    marker = main.read_json(main.COMPLETE)
    details = main.read_jsonl(main.DETAILS)
    analysis = main.read_json(main.ANALYSIS)
    final = main.read_json(main.FINAL)

    checks["one_shot_completion"] = marker["status"] == "formal_run_complete" and raw["system_count"] == len(details)
    checks["artifact_integrity"] = (
        raw["details_sha256"] == main.file_sha256(main.DETAILS)
        and marker["raw_sha256"] == main.file_sha256(main.RAW)
        and marker["details_sha256"] == main.file_sha256(main.DETAILS)
    )
    checks["system_identity"] = {row["system_id"] for row in details} == {row["system_id"] for row in truth}
    checks["no_pretrained_model"] = raw["pretrained_model_loaded"] is False
    checks["truth_not_public"] = all("mechanism" not in row for row in public)
    recomputed = {split: main.summarize_split(details, split) for split in main.SPLITS}
    gates = {
        "G-MECHANISM-TRUTH-BREADTH": all(summary["passed"] for summary in recomputed.values()),
        "G-CUT-RECOVERY": all(summary["checks"]["cut_recovery"] for summary in recomputed.values()),
        "G-ROLE-RECOVERY": all(summary["checks"]["role_recovery"] for summary in recomputed.values()),
        "G-IDENTITY-RECOVERY": all(summary["checks"]["identity_recovery"] for summary in recomputed.values()),
        "G-TWIN-ABSTENTION-SEPARATION": all(
            summary["checks"]["output_twin_abstention"] and summary["checks"]["edge_twin_separation"]
            for summary in recomputed.values()
        ),
        "G-GAUGE-INVARIANCE": all(summary["checks"]["gauge_invariance"] for summary in recomputed.values()),
    }
    verdict = "planted_component_edge_cut_camera_confirmed" if all(gates.values()) else "planted_component_edge_cut_camera_not_confirmed"
    checks["split_summaries"] = recomputed == analysis["splits"]
    checks["gates"] = gates == analysis["gates"] == final["gates"]
    checks["verdict"] = verdict == analysis["verdict"] == final["verdict"]
    checks["authorization"] = final["authorization"]["pretrained_model_contract"] is False and final["authorization"]["natural_language_mechanism_claim"] is False
    checks["final_hashes"] = all(
        final["artifact_hashes"][name] == main.file_sha256(path)
        for name, path in {
            "protocol": main.PROTOCOL,
            "public_material": main.PUBLIC_SYSTEMS,
            "sealed_truth": main.SEALED_TRUTH,
            "environment": main.ENVIRONMENT,
            "preaudit": main.PREAUDIT,
            "raw": main.RAW,
            "details": main.DETAILS,
            "complete": main.COMPLETE,
            "analysis": main.ANALYSIS,
        }.items()
    )
    result = {
        "phase": main.PHASE,
        "audit_stage": "final",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed_gates": gates,
        "recomputed_verdict": verdict,
    }
    result["audit_digest"] = main.digest(result)
    return result


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "final"))
    args = parser.parse_args()
    if args.stage == "preaudit":
        result = preaudit()
        write(result, main.PREAUDIT)
    else:
        result = final_audit()
        write(result, main.FINAL_AUDIT)
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
