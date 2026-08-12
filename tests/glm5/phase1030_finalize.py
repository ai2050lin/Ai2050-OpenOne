#!/usr/bin/env python3
"""Finalize the Phase1030 two-template replication."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1030_composition_replication_protocol as protocol


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_map(
    metrics: dict[str, Any],
    scope: str,
) -> dict[str, dict[str, Any]]:
    return {
        row["condition"]: row["metrics"][scope]
        for row in metrics["confirmation_conditions"]
    }


def scope_gate(
    clean: float,
    conditions: dict[str, dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    selected = conditions["selected_source_b"]
    unselected = conditions["unselected_source_b"]
    source = conditions["source_pair_b"]
    query_q = conditions["query_q"]
    query_bq = conditions["query_bq"]
    source_query_q = conditions["source_pair_plus_query_q"]
    source_query_bq = conditions["source_pair_plus_query_bq"]
    full_bq = conditions["full_bq"]
    scrambled = conditions["source_pair_scrambled"]
    source_wrong = conditions["source_pair_wrong_position"]
    query_wrong = conditions["query_q_wrong_position"]
    values = {
        "clean_expected_top1": clean,
        "selected_source_alternate_top1": float(
            selected["alternate_top1"]
        ),
        "unselected_source_alternate_top1": float(
            unselected["alternate_top1"]
        ),
        "source_pair_alternate_top1": float(
            source["alternate_top1"]
        ),
        "query_q_alternate_top1": float(
            query_q["alternate_top1"]
        ),
        "query_bq_base_top1": float(query_bq["base_top1"]),
        "source_query_q_base_top1": float(
            source_query_q["base_top1"]
        ),
        "source_query_bq_base_top1": float(
            source_query_bq["base_top1"]
        ),
        "full_bq_base_top1": float(full_bq["base_top1"]),
        "selected_minus_unselected_alternate_top1": float(
            selected["alternate_top1"]
            - unselected["alternate_top1"]
        ),
        "source_minus_scrambled_alternate_top1": float(
            source["alternate_top1"]
            - scrambled["alternate_top1"]
        ),
        "source_minus_wrong_alternate_top1": float(
            source["alternate_top1"]
            - source_wrong["alternate_top1"]
        ),
        "query_q_minus_wrong_alternate_top1": float(
            query_q["alternate_top1"]
            - query_wrong["alternate_top1"]
        ),
        "source_query_bq_base_minus_single_base_top1": float(
            source_query_bq["base_top1"]
            - max(source["base_top1"], query_q["base_top1"])
        ),
    }
    checks = {
        "clean": (
            values["clean_expected_top1"]
            >= thresholds["clean_expected_top1_minimum"]
        ),
        "selected_source": (
            values["selected_source_alternate_top1"]
            >= thresholds[
                "selected_source_alternate_top1_minimum"
            ]
        ),
        "source_pair": (
            values["source_pair_alternate_top1"]
            >= thresholds[
                "source_pair_alternate_top1_minimum"
            ]
        ),
        "query_q": (
            values["query_q_alternate_top1"]
            >= thresholds["query_q_alternate_top1_minimum"]
        ),
        "query_bq": (
            values["query_bq_base_top1"]
            >= thresholds["query_bq_base_top1_minimum"]
        ),
        "source_query_bq": (
            values["source_query_bq_base_top1"]
            >= thresholds[
                "source_plus_query_bq_base_top1_minimum"
            ]
        ),
        "full_bq": (
            values["full_bq_base_top1"]
            >= thresholds["full_bq_base_top1_minimum"]
        ),
        "selected_vs_unselected": (
            values["selected_minus_unselected_alternate_top1"]
            >= thresholds[
                "selected_minus_unselected_alternate_top1_minimum"
            ]
        ),
        "source_vs_scrambled": (
            values["source_minus_scrambled_alternate_top1"]
            >= thresholds[
                "source_minus_scrambled_alternate_top1_minimum"
            ]
        ),
        "source_vs_wrong": (
            values["source_minus_wrong_alternate_top1"]
            >= thresholds[
                "source_minus_wrong_alternate_top1_minimum"
            ]
        ),
        "query_vs_wrong": (
            values["query_q_minus_wrong_alternate_top1"]
            >= thresholds[
                "query_q_minus_wrong_alternate_top1_minimum"
            ]
        ),
        "causal_cancellation": (
            values[
                "source_query_bq_base_minus_single_base_top1"
            ]
            >= thresholds[
                "source_query_bq_base_minus_single_base_top1_minimum"
            ]
        ),
    }
    source_pass = all(
        checks[key]
        for key in (
            "clean",
            "selected_source",
            "source_pair",
            "selected_vs_unselected",
            "source_vs_scrambled",
            "source_vs_wrong",
        )
    )
    query_pass = all(
        checks[key]
        for key in (
            "clean",
            "query_q",
            "query_bq",
            "query_vs_wrong",
        )
    )
    composition_pass = all(
        checks[key]
        for key in (
            "clean",
            "source_pair",
            "query_q",
            "query_bq",
            "source_query_bq",
            "full_bq",
            "causal_cancellation",
        )
    )
    return {
        "values": values,
        "checks": checks,
        "source_pass": source_pass,
        "query_pass": query_pass,
        "composition_pass": composition_pass,
        "full_pass": source_pass and query_pass and composition_pass,
    }


def model_summary(
    model: str,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    atlas_dir = protocol.OUT_ROOT / "atlas" / model
    run = read_json(atlas_dir / "summary.json")
    metrics = read_json(atlas_dir / "metrics.json")
    gates = {}
    for scope in ("template_0", "template_1"):
        clean = float(
            metrics["clean_four_world_readout"][scope]
            ["all_worlds"]["expected_top1"]
        )
        gates[scope] = scope_gate(
            clean,
            condition_map(metrics, scope),
            prereg["replication_gate"],
        )
    return {
        "model": model,
        "selected_depths": run["selected_depths"],
        "template_gates": gates,
        "source_both_templates": all(
            row["source_pass"] for row in gates.values()
        ),
        "query_both_templates": all(
            row["query_pass"] for row in gates.values()
        ),
        "composition_both_templates": all(
            row["composition_pass"] for row in gates.values()
        ),
        "full_both_templates": all(
            row["full_pass"] for row in gates.values()
        ),
        "clean_four_world_readout": metrics[
            "clean_four_world_readout"
        ],
        "conditions": {
            scope: condition_map(metrics, scope)
            for scope in ("all", "template_0", "template_1")
        },
        "observational_role_depth": metrics[
            "observational_role_depth"
        ],
        "run": run,
    }


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def artifact_manifest() -> dict[str, Any]:
    manifest_path = protocol.OUT_ROOT / "final" / "artifact_manifest.json"
    files = []
    for path in sorted(
        item for item in protocol.OUT_ROOT.rglob("*")
        if item.is_file() and item != manifest_path
    ):
        files.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        })
    return {
        "schema_version": "phase1030_artifact_manifest.v1",
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def main() -> None:
    prereg = read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    models = {
        model: model_summary(model, prereg)
        for model in protocol.MODELS
    }
    source_models = [
        model
        for model, row in models.items()
        if row["source_both_templates"]
    ]
    query_models = [
        model
        for model, row in models.items()
        if row["query_both_templates"]
    ]
    composition_models = [
        model
        for model, row in models.items()
        if row["composition_both_templates"]
    ]
    full_models = [
        model
        for model, row in models.items()
        if row["full_both_templates"]
    ]
    cross_model = {
        "source_both_templates_models": source_models,
        "source_both_templates_count": len(source_models),
        "query_both_templates_models": query_models,
        "query_both_templates_count": len(query_models),
        "composition_both_templates_models": composition_models,
        "composition_both_templates_count": len(composition_models),
        "full_both_templates_models": full_models,
        "full_both_templates_count": len(full_models),
    }
    summary = {
        "schema_version": "phase1030_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": models,
        "cross_model": cross_model,
        "interpretation_policy": prereg["interpretation_rule"],
        "claim_limit": prereg["claim_limit"],
    }
    final_dir = protocol.OUT_ROOT / "final"
    protocol.write_json(final_dir / "summary.json", summary)

    if len(full_models) >= 2:
        route = "component_path_localization"
        decision = (
            "the two-template composition candidate repeated in at least "
            "two models; localize attention and MLP contributions while "
            "retaining the residual-state result as the reference map"
        )
    elif len(source_models) >= 2 and len(query_models) >= 2:
        route = "distributed_query_boundary_coalition"
        decision = (
            "source and query effects repeat, but full cancellation does "
            "not repeat across models/templates; stop singleton retries "
            "and map query-plus-boundary coalitions in the two-binding task"
        )
    elif len(source_models) >= 2:
        route = "distributed_query_selector"
        decision = (
            "source transport repeats but query transport does not; map "
            "the query span and late boundary jointly instead of another "
            "single-token query patch"
        )
    else:
        route = "protocol_reassessment"
        decision = (
            "even source transport failed the two-template replication; "
            "audit tokenization, prototype readout, and source position "
            "definitions before further causal expansion"
        )
    next_action = {
        "schema_version": "phase1030_automatic_next_action.v1",
        "automatic_next_execution_authorized": True,
        "route": route,
        "decision": decision,
        "authorization_basis": (
            "512 independent units, two templates, frozen Phase1029 "
            "depths, and separate Q-only/BQ query-state interventions"
        ),
    }
    protocol.write_json(
        final_dir / "automatic_next_action.json", next_action
    )

    checks = {
        "protocol_common_audit": read_json(
            protocol.OUT_ROOT / "protocol" / "audit.common.json"
        )["all_checks_passed"],
        "protocol_model_audits": read_json(
            protocol.OUT_ROOT / "protocol" / "audit.models.json"
        )["all_checks_passed"],
        "all_model_runs_present": all(
            (
                protocol.OUT_ROOT / "atlas" / model / "summary.json"
            ).exists()
            for model in protocol.MODELS
        ),
        "phase1029_frozen_selection": all(
            row["run"]["selection_source"] == "phase1029_frozen"
            for row in models.values()
        ),
        "all_arrays_finite": all(
            row["run"]["finiteness"]["all_finite"]
            for row in models.values()
        ),
        "fp16_no_quantization": all(
            row["run"]["precision"] == "fp16"
            and row["run"]["quantization"] == "none"
            for row in models.values()
        ),
    }
    audit = {
        "schema_version": "phase1030_final_audit.v1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    protocol.write_json(final_dir / "audit.json", audit)
    manifest = artifact_manifest()
    protocol.write_json(
        final_dir / "artifact_manifest.json", manifest
    )
    print(json.dumps({
        "cross_model": cross_model,
        "next_action": next_action,
        "audit": audit,
        "manifest": {
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
