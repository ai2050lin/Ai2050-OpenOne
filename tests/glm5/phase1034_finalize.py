#!/usr/bin/env python3
"""Finalize Phase1034 and identify only repeated descriptive atlas cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase1034_post_query_component_protocol as protocol


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
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "phase": protocol.PHASE,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def row_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(row["depth_bin"]),
        row["group"],
        row["component"],
        row["anchor"],
    )


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

    rule = prereg["descriptive_repeat_rule"]
    qualifying: dict[
        tuple[int, str, str], list[str]
    ] = {}
    details: dict[tuple[int, str, str], dict[str, Any]] = {}
    for model in protocol.MODELS:
        indexed = {
            row_key(row): row
            for row in metrics[model]["depth_bins"]
        }
        for depth_bin in range(protocol.DEPTH_BIN_COUNT):
            for component in protocol.COMPONENTS:
                for anchor in protocol.ANCHORS:
                    template_rows = [
                        indexed[
                            (
                                depth_bin,
                                f"template_{template}",
                                component,
                                anchor,
                            )
                        ]
                        for template in (0, 1)
                    ]
                    passed = all(
                        row["metrics"]["binding_context_cosine"][
                            "median"
                        ]
                        <= rule["binding_context_cosine_max"]
                        and row["binding_context_negative_rate"]
                        >= rule["negative_cosine_prevalence_min"]
                        for row in template_rows
                    )
                    if passed:
                        key = (depth_bin, component, anchor)
                        qualifying.setdefault(key, []).append(model)
                        details.setdefault(key, {})[model] = {
                            "template_0": template_rows[0],
                            "template_1": template_rows[1],
                        }

    conserved = []
    for key, models in sorted(qualifying.items()):
        if len(models) < int(rule["minimum_models"]):
            continue
        conserved.append(
            {
                "depth_bin": key[0],
                "component": key[1],
                "anchor": key[2],
                "models": models,
                "model_details": details[key],
            }
        )

    instrumentation_models = [
        model
        for model, row in summaries.items()
        if row["instrumentation_gate_passed"]
        and row["all_recorded_values_finite"]
    ]
    suffix_cells = [
        row
        for row in conserved
        if row["anchor"] != "query_end"
    ]
    causal_followup_needed = (
        len(instrumentation_models) >= 2 and bool(suffix_cells)
    )

    source_reuse = {}
    for model in protocol.MODELS:
        group_rows = {
            row["group"]: row
            for row in metrics[model]["source_summary"]
        }
        model_summary = {
            group: {
                metric: values["median"]
                for metric, values in group_rows[group][
                    "metrics"
                ].items()
            }
            for group in (
                "template_0",
                "template_1",
                "bank_single",
                "bank_double",
            )
        }
        values = np.load(
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "source_relative.fp32.npy",
            mmap_mode="r",
        )
        layer_rows = []
        for bin_index in range(protocol.DEPTH_BIN_COUNT):
            start = int(
                np.floor(
                    bin_index
                    * values.shape[1]
                    / protocol.DEPTH_BIN_COUNT
                )
            )
            end = int(
                np.floor(
                    (bin_index + 1)
                    * values.shape[1]
                    / protocol.DEPTH_BIN_COUNT
                )
            )
            end = max(end, start + 1)
            layer_rows.append(
                {
                    "depth_bin": bin_index,
                    "depth_start": start + 1,
                    "depth_end": end,
                    "relative_pair_difference_median": float(
                        np.median(values[:, start:end, 0])
                    ),
                    "future_query_leakage_median": float(
                        np.median(values[:, start:end, 1])
                    ),
                    "pair_opposition_median": float(
                        np.median(values[:, start:end, 2])
                    ),
                }
            )
        source_reuse[model] = {
            "groups": model_summary,
            "minimum_pair_opposition_across_template_and_bank": min(
                row["pair_opposition"] for row in model_summary.values()
            ),
            "depth_bins": layer_rows,
        }
    if causal_followup_needed:
        route = (
            "Preregister one band-level causal test over the repeated "
            "post-query component cells. Patch the whole repeated band and "
            "retain wrong-position, self, and scrambled-world controls; do "
            "not select individual high-activation neurons or heads."
        )
    else:
        route = (
            "Do not launch another coalition patch. Preserve the complete "
            "atlas and redesign the query-role controls because no "
            "instrumented post-query component band repeated across models."
        )

    aggregate = {
        "schema_version": "phase1034_aggregate.v1",
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
            "all_recorded_values_finite": all(
                row["all_recorded_values_finite"]
                for row in summaries.values()
            ),
        },
        "instrumentation_models": instrumentation_models,
        "source_relative_reuse_atlas": {
            "models": source_reuse,
            "descriptive_observation": (
                "The same binding swap produces strongly opposed differences "
                "at the two fact slots across models, templates, token-length "
                "banks, and most depths."
            ),
            "interpretation_limit": (
                "This is a repeated relative-difference structure, but the "
                "swap also changes lexical content. It does not by itself "
                "establish semantic-family knowledge, a natural causal path, "
                "or biological optimality."
            ),
        },
        "conserved_descriptive_cells": conserved,
        "conserved_post_query_cells": suffix_cells,
        "automatic_next_decision": {
            "causal_followup_needed": causal_followup_needed,
            "route": route,
            "claim_limit": (
                "A conserved cell is a repeated factorial response pattern, "
                "not yet a natural causal path or a language equation."
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
        "instrumentation_model_count": len(instrumentation_models),
        "conserved_cell_count": len(conserved),
        "conserved_post_query_cell_count": len(suffix_cells),
        "manifest_file_count": artifact["file_count"],
        "manifest_total_bytes": artifact["total_bytes"],
    }
    protocol.write_json(protocol.OUT_ROOT / "final" / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            aggregate["automatic_next_decision"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
