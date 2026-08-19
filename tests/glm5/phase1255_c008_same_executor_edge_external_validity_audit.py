#!/usr/bin/env python3
"""Independent audit for Phase1255/C008."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import phase1254_c007_free_transformer_edge_external_validity as prior
import phase1255_c008_same_executor_edge_external_validity as main


def write(value: dict[str, Any], path: Path) -> None:
    main.atomic_json(path, value)
    print(main.canonical_json(value))


def preaudit() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    protocol = main.read_json(main.PROTOCOL)
    rows = main.read_jsonl(main.MATERIAL)
    expected = main.protocol_payload(rows)
    checks["source_hashes"] = protocol["source_hashes"] == expected["source_hashes"]
    checks["protocol_digest"] = protocol["protocol_digest"] == expected["protocol_digest"]
    checks["threshold_identity"] = protocol["thresholds"] == prior.THRESHOLDS == main.THRESHOLDS
    checks["prefix_identity"] = tuple(protocol["selection"]["candidate_prefix_sizes"]) == prior.PREFIX_SIZES == main.PREFIX_SIZES
    checks["architecture_identity"] = protocol["architectures"] == {name: vars(config) for name, config in prior.ARCHITECTURES.items()}
    checks["row_count"] = len(rows) == sum(main.WORLD_COUNTS.values())
    checks["partition_counts"] = {
        name: sum(row["partition"] == name for row in rows) for name in main.WORLD_COUNTS
    } == main.WORLD_COUNTS
    checks["row_digests"] = all(
        main.digest({key: value for key, value in row.items() if key != "row_digest"}) == row["row_digest"]
        for row in rows
    )
    checks["counterfactual_answers"] = all(
        row["answers"]["target"] != row["answers"]["base"]
        and row["answers"]["wrong"] != row["answers"]["target"]
        and row["answers"]["null"] == row["answers"]["base"]
        for row in rows
    )
    checks["one_shot_clean"] = not main.COMPLETE.exists() and not main.RAW.exists() and not main.MODELS.exists()
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
    main.verify_protocol()
    raw = main.read_json(main.RAW)
    marker = main.read_json(main.COMPLETE)
    rows = main.read_jsonl(main.MODELS)
    analysis = main.read_json(main.ANALYSIS)
    final = main.read_json(main.FINAL)
    checks["one_shot_completion"] = marker["status"] == "formal_run_complete" and raw["model_count"] == len(rows) == 6
    checks["artifact_integrity"] = (
        raw["models_sha256"] == main.file_sha256(main.MODELS)
        and marker["raw_sha256"] == main.file_sha256(main.RAW)
        and marker["models_sha256"] == main.file_sha256(main.MODELS)
    )
    checks["model_identity"] = {row["model_key"] for row in rows} == set(main.MODEL_SEEDS)
    checks["same_executor"] = all(
        row["native_explicit_logit_gap"] <= 1.0e-7 for row in rows if row["behavior_qualified"]
    )
    checks["selection_prefix"] = all(
        not row["behavior_qualified"]
        or row["selected_components"] == row["greedy_ranking"][: row["selected_size"]]
        for row in rows
    )
    checks["component_validity"] = all(
        not row["behavior_qualified"]
        or set(row["selected_components"]).issubset(set(main.component_ids(main.ARCHITECTURES[row["architecture"]])))
        for row in rows
    )
    recomputed = main.summarize(rows)
    verdict = "free_transformer_typed_edge_coalition_confirmed" if recomputed["passed_all"] else "free_transformer_typed_edge_coalition_not_confirmed"
    checks["summary"] = recomputed == analysis["summary"] == final["summary"]
    checks["verdict"] = verdict == analysis["verdict"] == final["verdict"]
    checks["authorization"] = (
        final["authorization"]["fresh_qwen_single_model_edge_contract"] == recomputed["passed_all"]
        and final["authorization"]["semantic_mechanism_claim"] is False
        and final["authorization"]["unique_minimal_algorithm_claim"] is False
    )
    checks["final_hashes"] = all(
        final["artifact_hashes"][name] == main.file_sha256(path)
        for name, path in {
            "protocol": main.PROTOCOL,
            "material": main.MATERIAL,
            "environment": main.ENVIRONMENT,
            "preaudit": main.PREAUDIT,
            "raw": main.RAW,
            "models": main.MODELS,
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
        "recomputed_summary": recomputed,
        "recomputed_verdict": verdict,
    }
    result["audit_digest"] = main.digest(result)
    return result


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "final"))
    args = parser.parse_args()
    result = preaudit() if args.stage == "preaudit" else final_audit()
    write(result, main.PREAUDIT if args.stage == "preaudit" else main.FINAL_AUDIT)
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
