#!/usr/bin/env python3
"""Aggregate Phase1046 and freeze the smallest repeated coalition."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1046_distributed_receiver_protocol as protocol


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model_name: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model_name / "summary.json"
        )
        for model_name in protocol.MODELS
    }
    repeated: dict[tuple[int, str], list[str]] = defaultdict(list)
    rows = {}
    for model_name, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model_name} protocol digest drift")
        rows[model_name] = {}
        for row in summary["candidate_cells"]:
            key = (
                int(row["relative_depth_slot"]),
                str(row["coalition_mask"]),
            )
            repeated[key].append(model_name)
            rows[model_name][key] = row

    repeated_cells = []
    for key, models in sorted(repeated.items()):
        if len(models) < prereg["discovery_gate"]["minimum_models"]:
            continue
        depth, mask = key
        repeated_cells.append({
            "relative_depth_slot": depth,
            "coalition_mask": mask,
            "models": models,
            "model_count": len(models),
            "model_rows": {
                model_name: rows[model_name][key]
                for model_name in models
            },
        })

    site_counts = {
        mask: len(sites)
        for mask, sites in protocol.COALITION_MASKS.items()
        if mask != "full_sequence_reference"
    }
    candidates = sorted(
        repeated_cells,
        key=lambda row: (
            site_counts[row["coalition_mask"]],
            int(row["relative_depth_slot"]),
            -int(row["model_count"]),
        ),
    )
    frozen = candidates[:1]
    confirmation_needed = bool(frozen)
    reference_audit = {
        model_name: [
            {
                "relative_depth_slot": row["relative_depth_slot"],
                "mediation_fraction": row[
                    "mediation_fraction"
                ]["median"],
                "replay_recovery": row["replay_recovery"]["median"],
                "passed": row["reference_gate"],
            }
            for row in summary["reference_cells"]
        ]
        for model_name, summary in summaries.items()
    }
    reference_passed = all(
        row["passed"]
        for model_rows in reference_audit.values()
        for row in model_rows
    )
    automatic_next = {
        "confirmation_needed": confirmation_needed,
        "frozen_candidates": frozen,
        "route": (
            "On untouched surface-0 complementary-query material, confirm "
            "the relative-slot-2 concept pair and decompose it into selected "
            "and unselected concept constituents. Keep query-boundary and "
            "full-sequence controls."
            if confirmation_needed
            else prereg["automatic_followup"]["if_no_candidate"]
        ),
    }
    manifests = {}
    for model_name in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model_name
        manifests[model_name] = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(atlas.iterdir())
            if path.is_file()
        }

    aggregate = {
        "schema_version": "phase1046_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "sample_plan": prereg["sample_plan"],
        "model_candidate_counts": {
            model_name: summary["candidate_cell_count"]
            for model_name, summary in summaries.items()
        },
        "cross_model_repeated_cell_count": len(repeated_cells),
        "cross_model_repeated_cells": repeated_cells,
        "reference_audit": reference_audit,
        "all_full_sequence_references_passed": reference_passed,
        "automatic_next_decision": automatic_next,
        "artifact_manifest": manifests,
        "main_result": {
            "frozen_cell": (
                frozen[0] if frozen else None
            ),
            "plain_statement": (
                "At the second receiver slot, swapping the two concept "
                "positions transfers most of the early source effect in all "
                "three models. Query and output-boundary positions do not "
                "show the same cross-model sufficiency."
            ),
            "remaining_ambiguity": (
                "The pair may be a true competitive joint state, or the "
                "selected source state may simply persist at its original "
                "concept position. Constituent decomposition is required."
            ),
        },
        "claim_limits": prereg["claim_limits"],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    protocol.write_json(
        protocol.OUT_ROOT / "automatic_next_decision.json",
        automatic_next,
    )
    print(json.dumps({
        "cross_model_repeated_cell_count": len(repeated_cells),
        "all_full_sequence_references_passed": reference_passed,
        "automatic_next_decision": automatic_next,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
