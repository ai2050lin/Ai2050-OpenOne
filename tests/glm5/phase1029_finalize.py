#!/usr/bin/env python3
"""Finalize Phase1029 and choose the next non-blocking research route."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1029_multibinding_competition_protocol as protocol


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_map(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["condition"]: row["metrics"]
        for row in metrics["confirmation_conditions"]
    }


def model_gate(
    model: str,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    atlas_dir = protocol.OUT_ROOT / "atlas" / model
    run = read_json(atlas_dir / "summary.json")
    metrics = read_json(atlas_dir / "metrics.json")
    conditions = condition_map(metrics)
    thresholds = prereg["scale_free_confirmation_gate"]
    clean = float(
        metrics["clean_four_world_readout"]["confirmation"]
        ["all_worlds"]["expected_top1"]
    )
    selected = conditions["selected_source_b"]
    unselected = conditions["unselected_source_b"]
    source_pair = conditions["source_pair_b"]
    query = conditions["query_q"]
    combined = conditions["source_pair_plus_query_mixed"]
    full_bq = conditions["full_bq"]
    scrambled = conditions["source_pair_scrambled"]
    source_wrong = conditions["source_pair_wrong_position"]
    query_wrong = conditions["query_wrong_position"]
    values = {
        "clean_four_world_expected_top1": clean,
        "selected_source_alternate_top1": float(
            selected["alternate_top1"]
        ),
        "unselected_source_alternate_top1": float(
            unselected["alternate_top1"]
        ),
        "source_pair_alternate_top1": float(
            source_pair["alternate_top1"]
        ),
        "query_alternate_top1": float(query["alternate_top1"]),
        "combined_base_top1": float(combined["base_top1"]),
        "full_bq_base_top1": float(full_bq["base_top1"]),
        "source_pair_scrambled_alternate_top1": float(
            scrambled["alternate_top1"]
        ),
        "source_pair_wrong_position_alternate_top1": float(
            source_wrong["alternate_top1"]
        ),
        "query_wrong_position_alternate_top1": float(
            query_wrong["alternate_top1"]
        ),
        "selected_minus_unselected_alternate_top1": float(
            selected["alternate_top1"]
            - unselected["alternate_top1"]
        ),
        "source_pair_minus_scrambled_alternate_top1": float(
            source_pair["alternate_top1"]
            - scrambled["alternate_top1"]
        ),
        "source_pair_minus_wrong_position_alternate_top1": float(
            source_pair["alternate_top1"]
            - source_wrong["alternate_top1"]
        ),
        "query_minus_wrong_position_alternate_top1": float(
            query["alternate_top1"]
            - query_wrong["alternate_top1"]
        ),
        "combined_base_minus_single_base_top1": float(
            combined["base_top1"]
            - max(source_pair["base_top1"], query["base_top1"])
        ),
    }
    checks = {
        "clean": (
            values["clean_four_world_expected_top1"]
            >= thresholds[
                "clean_four_world_expected_top1_minimum"
            ]
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
        "query": (
            values["query_alternate_top1"]
            >= thresholds["query_alternate_top1_minimum"]
        ),
        "combined": (
            values["combined_base_top1"]
            >= thresholds["combined_base_top1_minimum"]
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
            values["source_pair_minus_scrambled_alternate_top1"]
            >= thresholds[
                "source_pair_minus_scrambled_alternate_top1_minimum"
            ]
        ),
        "source_vs_wrong_position": (
            values[
                "source_pair_minus_wrong_position_alternate_top1"
            ]
            >= thresholds[
                "source_pair_minus_wrong_position_alternate_top1_minimum"
            ]
        ),
        "query_vs_wrong_position": (
            values["query_minus_wrong_position_alternate_top1"]
            >= thresholds[
                "query_minus_wrong_position_alternate_top1_minimum"
            ]
        ),
        "causal_cancellation": (
            values["combined_base_minus_single_base_top1"]
            >= thresholds[
                "combined_base_minus_single_base_top1_minimum"
            ]
        ),
        "finite": bool(run["finiteness"]["all_finite"]),
    }
    source_pass = all(
        checks[key]
        for key in (
            "clean",
            "selected_source",
            "source_pair",
            "selected_vs_unselected",
            "source_vs_scrambled",
            "source_vs_wrong_position",
            "finite",
        )
    )
    query_pass = all(
        checks[key]
        for key in (
            "clean",
            "query",
            "query_vs_wrong_position",
            "finite",
        )
    )
    composition_pass = all(
        checks[key]
        for key in (
            "clean",
            "source_pair",
            "query",
            "combined",
            "full_bq",
            "causal_cancellation",
            "finite",
        )
    )
    full_pass = source_pass and query_pass and composition_pass
    return {
        "model": model,
        "selected_depths": {
            "source": run["selected_source_depth"],
            "query": run["selected_query_depth"],
            "pre_output": run["preoutput_depth"],
            "readout": run["readout_depth"],
        },
        "values": values,
        "checks": checks,
        "source_pass": source_pass,
        "query_pass": query_pass,
        "composition_pass": composition_pass,
        "full_binding_gate_pass": full_pass,
        "conditions": conditions,
        "run": run,
        "observational_role_depth": metrics[
            "observational_role_depth"
        ],
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
        "schema_version": "phase1029_artifact_manifest.v1",
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def main() -> None:
    prereg = read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    models = {
        model: model_gate(model, prereg)
        for model in protocol.MODELS
    }
    source_models = [
        model for model, row in models.items() if row["source_pass"]
    ]
    query_models = [
        model for model, row in models.items() if row["query_pass"]
    ]
    composition_models = [
        model
        for model, row in models.items()
        if row["composition_pass"]
    ]
    full_models = [
        model
        for model, row in models.items()
        if row["full_binding_gate_pass"]
    ]
    cross_model = {
        "source_pass_models": source_models,
        "source_pass_count": len(source_models),
        "query_pass_models": query_models,
        "query_pass_count": len(query_models),
        "composition_pass_models": composition_models,
        "composition_pass_count": len(composition_models),
        "full_binding_gate_pass_models": full_models,
        "full_binding_gate_pass_count": len(full_models),
    }
    summary = {
        "schema_version": "phase1029_final_summary.v1",
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
        route = "component_decomposition_replication"
        decision = (
            "binding selection and BQ cancellation repeated in at least "
            "two models; expand templates and localize attention/MLP "
            "components without claiming a minimal circuit"
        )
    elif len(source_models) >= 2 and len(query_models) < 2:
        route = "distributed_query_selector_mapping"
        decision = (
            "source transport repeated but query transport did not; map "
            "multi-position query and boundary coalitions in the same "
            "two-binding task before any binding closure claim"
        )
    elif len(source_models) < 2:
        route = "source_anchor_replication"
        decision = (
            "selected source transport did not repeat across two models; "
            "replicate with new concepts and inspect tokenization and "
            "wrong-position sensitivity"
        )
    else:
        route = "composition_cancellation_replication"
        decision = (
            "source and query transport repeated but BQ cancellation did "
            "not; replicate the interaction with new templates and "
            "separate mixed from coherent BQ donors"
        )
    next_action = {
        "schema_version": "phase1029_automatic_next_action.v1",
        "automatic_next_execution_authorized": True,
        "route": route,
        "decision": decision,
        "authorization_basis": (
            "independent confirmation of the four-world B/Q/BQ design "
            "with source, query, scrambled, wrong-position, and same-"
            "answer controls"
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
        "discovery_only_selection": all(
            row["run"]["selection_source"] == "discovery_only"
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
        "schema_version": "phase1029_final_audit.v1",
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
