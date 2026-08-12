#!/usr/bin/env python3
"""Finalize Phase1035 discovery/confirmation event mapping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import phase1035_native_family_routing_protocol as protocol


EVENT_ANCHORS = (
    "query_nonce",
    "suffix_first",
    "suffix_mid",
    "pre_output",
)


def manifest(root: Path) -> dict[str, Any]:
    excluded = {
        root / "final" / "artifact_manifest.json",
        root / "final" / "audit.json",
    }
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path in excluded:
            continue
        data = path.read_bytes()
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    return {
        "phase": protocol.PHASE,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def cell_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["component"]),
        int(row["depth_bin"]),
        str(row["anchor"]),
    )


def indexed_rows(
    metrics: dict[str, Any],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    return {
        (
            str(row["component"]),
            str(row["group"]),
            int(row["depth_bin"]),
            str(row["anchor"]),
        ): row
        for row in metrics["response_depth_bins"]
    }


def event_passes(
    row: dict[str, Any],
    rule: dict[str, Any],
) -> bool:
    metrics = row["metrics"]
    cosine = metrics["binding_query_cosine"]["median"]
    invariant = metrics["bq_member_invariance"]["median"]
    interaction = metrics["bq_interaction_rel_norm"]["median"]
    return (
        row["finite_row_rate"] >= 0.99
        and cosine is not None
        and cosine <= rule["binding_query_cosine_median_max"]
        and row["binding_query_negative_rate"]
        >= rule["binding_query_negative_rate_min"]
        and invariant is not None
        and invariant >= rule["bq_member_invariance_median_min"]
        and row["bq_member_positive_rate"]
        >= rule["bq_member_positive_rate_min"]
        and interaction is not None
        and interaction >= rule["bq_interaction_rel_norm_median_min"]
    )


def source_family_passes(row: dict[str, Any]) -> bool:
    metrics = row["metrics"]
    values = [
        metrics["binding_member_invariance_q0"]["median"],
        metrics["binding_member_invariance_q1"]["median"],
    ]
    return (
        row["finite_row_rate"] >= 0.99
        and all(value is not None and value > 0 for value in values)
    )


def summarize_model_events(
    model: str,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    rule: dict[str, Any],
) -> dict[str, Any]:
    index = indexed_rows(metrics)
    discovered = []
    confirmed = []
    for component in ("residual", "attention", "mlp"):
        if (
            component != "residual"
            and not summary["component_instrumentation_gate_passed"]
        ):
            continue
        for depth_bin in range(protocol.DEPTH_BIN_COUNT):
            for anchor in EVENT_ANCHORS:
                key = (component, depth_bin, anchor)
                discovery_rows = [
                    index[(component, f"template_{template}", depth_bin, anchor)]
                    for template in (0, 1)
                ]
                if not all(event_passes(row, rule) for row in discovery_rows):
                    continue
                discovery_entry = {
                    "component": component,
                    "depth_bin": depth_bin,
                    "anchor": anchor,
                    "discovery_templates": discovery_rows,
                }
                discovered.append(discovery_entry)
                confirmation_rows = [
                    index[(component, f"template_{template}", depth_bin, anchor)]
                    for template in (2, 3)
                ]
                if all(event_passes(row, rule) for row in confirmation_rows):
                    confirmed.append({
                        **discovery_entry,
                        "confirmation_templates": confirmation_rows,
                    })

    source_discovered = []
    source_confirmed = []
    for depth_bin in range(protocol.DEPTH_BIN_COUNT):
        for anchor in ("concept_a", "concept_b"):
            key = ("residual", depth_bin, anchor)
            discovery_rows = [
                index[("residual", f"template_{template}", depth_bin, anchor)]
                for template in (0, 1)
            ]
            if not all(source_family_passes(row) for row in discovery_rows):
                continue
            entry = {
                "component": "residual",
                "depth_bin": depth_bin,
                "anchor": anchor,
                "discovery_templates": discovery_rows,
            }
            source_discovered.append(entry)
            confirmation_rows = [
                index[("residual", f"template_{template}", depth_bin, anchor)]
                for template in (2, 3)
            ]
            if all(source_family_passes(row) for row in confirmation_rows):
                source_confirmed.append({
                    **entry,
                    "confirmation_templates": confirmation_rows,
                })

    behavior = metrics["behavior"]
    prototypes = metrics["heldout_internal_prototype_readout"]
    return {
        "model": model,
        "discovered_event_cells": discovered,
        "confirmed_event_cells": confirmed,
        "discovered_source_family_cells": source_discovered,
        "confirmed_source_family_cells": source_confirmed,
        "behavior": behavior,
        "heldout_internal_prototype_readout": prototypes,
        "instrumentation": {
            "component_gate_passed": summary[
                "component_instrumentation_gate_passed"
            ],
            "candidate_logit_gate_passed": summary[
                "candidate_logit_gate_passed"
            ],
            "array_finiteness": summary["array_finiteness"],
        },
    }


def cross_model_cells(
    model_rows: dict[str, dict[str, Any]],
    field: str,
    minimum_models: int = 2,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[str]] = {}
    details: dict[tuple[str, int, str], dict[str, Any]] = {}
    for model, row in model_rows.items():
        for entry in row[field]:
            key = cell_key(entry)
            grouped.setdefault(key, []).append(model)
            details.setdefault(key, {})[model] = entry
    result = []
    for key, models in sorted(grouped.items()):
        if len(models) < minimum_models:
            continue
        result.append({
            "component": key[0],
            "depth_bin": key[1],
            "anchor": key[2],
            "models": models,
            "model_details": details[key],
        })
    return result


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {}
    metrics = {}
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summaries[model] = protocol.read_json(atlas / "summary.json")
        metrics[model] = protocol.read_json(atlas / "metrics.json")

    rule = prereg["event_definition"]["discovery_rule"]
    model_events = {
        model: summarize_model_events(
            model, summaries[model], metrics[model], rule
        )
        for model in protocol.MODELS
    }
    conserved_events = cross_model_cells(
        model_events, "confirmed_event_cells"
    )
    conserved_sources = cross_model_cells(
        model_events, "confirmed_source_family_cells"
    )

    behavior_eligible = []
    for model, row in model_events.items():
        confirmation = row["behavior"]["confirmation"]
        if (
            row["instrumentation"]["candidate_logit_gate_passed"]
            and confirmation["candidate_set_accuracy"] is not None
            and confirmation["candidate_set_accuracy"] >= 0.50
        ):
            behavior_eligible.append(model)
    event_models = sorted({
        model
        for row in conserved_events
        for model in row["models"]
    })
    causal_models = sorted(set(behavior_eligible).intersection(event_models))
    causal_followup = (
        len(causal_models) >= 2
        and bool(conserved_events)
        and bool(conserved_sources)
    )

    if causal_followup:
        route = (
            "Run Phase1036 on held-out confirmation units. Compare full-span "
            "same-family lexical-member patches with cross-family patches at "
            "the queried and unqueried source roles, using only discovery-"
            "selected depth bins. Preserve self and wrong-role controls."
        )
    elif conserved_events:
        route = (
            "Preserve the repeated observational event map, but do not claim "
            "a transport path. Improve behavior or output finiteness before "
            "running a candidate-logit causal patch."
        )
    else:
        route = (
            "Do not run another source coalition patch. Inspect complete "
            "native curves and revise the event coordinate because no BxQ "
            "event survived disjoint lexical/template confirmation across "
            "models."
        )

    hypothesis_assessment = {
        "brain_language_near_optimal": (
            "Not tested. The experiment contains no brain, energy, evolution, "
            "or optimality comparator."
        ),
        "reuse_and_difference_exist": (
            "Supported only if source-family or BxQ structures repeat on "
            "held-out lexical members; efficiency remains untested."
        ),
        "relative_coding": (
            "The weak relational form is tested through role-conditioned "
            "family contrasts. The claim that a concept is fully defined by "
            "all of its relations is not tested."
        ),
        "language_as_pattern_collection": (
            "Useful as a protocol decomposition, not established as a full "
            "ontology of language."
        ),
        "unique_lexical_ecological_niche": (
            "The L factor measures retained lexical-member differences while "
            "holding family and answer fixed. It cannot establish a unique "
            "global niche for every vocabulary item."
        ),
        "small_model_roughness": (
            "Model-specific behavior and numerical stability are measured, "
            "but size remains confounded with architecture, tokenizer, and "
            "training data."
        ),
    }

    aggregate = {
        "schema_version": "phase1035_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": {
            "all_models_present": len(summaries) == len(protocol.MODELS),
            "protocol_digest_consistent": all(
                row["protocol_digest"] == prereg["protocol_digest"]
                for row in summaries.values()
            ),
            "all_fp16_no_quantization": all(
                row["precision"]["has_fp16_parameters"]
                and not row["precision"]["has_bf16_parameters"]
                and not row["precision"]["has_quantized_modules"]
                for row in summaries.values()
            ),
            "raw_protocol_audits_passed": all(
                row["all_checks_passed"]
                for row in prereg["model_audits"].values()
            ),
        },
        "model_events": model_events,
        "conserved_confirmed_event_cells": conserved_events,
        "conserved_confirmed_source_family_cells": conserved_sources,
        "hypothesis_assessment": hypothesis_assessment,
        "automatic_next_decision": {
            "causal_followup_needed": causal_followup,
            "eligible_models": causal_models,
            "route": route,
            "claim_limit": (
                "The gate chooses whether a controlled causal follow-up is "
                "worthwhile. It does not convert an observational event into "
                "a mechanism."
            ),
        },
        "model_summaries": summaries,
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)

    artifact = manifest(protocol.OUT_ROOT)
    protocol.write_json(
        protocol.OUT_ROOT / "final" / "artifact_manifest.json",
        artifact,
    )
    audit = {
        "phase": protocol.PHASE,
        "checks": aggregate["checks"],
        "confirmed_event_cells_by_model": {
            model: len(row["confirmed_event_cells"])
            for model, row in model_events.items()
        },
        "confirmed_source_cells_by_model": {
            model: len(row["confirmed_source_family_cells"])
            for model, row in model_events.items()
        },
        "conserved_event_cell_count": len(conserved_events),
        "conserved_source_cell_count": len(conserved_sources),
        "causal_followup_needed": causal_followup,
        "manifest_file_count": artifact["file_count"],
        "manifest_total_bytes": artifact["total_bytes"],
    }
    protocol.write_json(protocol.OUT_ROOT / "final" / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(
        aggregate["automatic_next_decision"],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
