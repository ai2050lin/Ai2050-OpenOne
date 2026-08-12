#!/usr/bin/env python3
"""Finalize Phase1007 gates from frozen behavior and source artifacts."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase1007_role_aligned_causal_source_protocol import (
    CONTRASTS,
    MODELS,
    OUT_ROOT,
    PHASE,
    read_json,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAMES = (
    "phase1007_role_aligned_causal_source_protocol.py",
    "phase1007_role_aligned_behavior.py",
    "phase1007_role_aligned_causal_source.py",
    "phase1007_role_aligned_causal_source_finalize.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition_view(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "donor_sequence_rate": condition["donor_sequence_rate"],
        "target_sequence_rate": condition["target_sequence_rate"],
        "median_normalized_transfer": condition[
            "median_normalized_transfer"
        ],
        "basic_transfer_gate": condition["basic_transfer_gate"],
    }


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    protocol_digest = protocol["preregistration_digest"]

    behavior_rows: list[dict[str, Any]] = []
    behavior_counts: dict[str, int] = {}
    for model in MODELS:
        summary = read_json(OUT_ROOT / "behavior" / model / "summary.json")
        if summary["protocol_digest"] != protocol_digest:
            raise RuntimeError(f"Behavior digest mismatch: {model}")
        behavior_counts[model] = summary["gate_pass_count"]
        behavior_rows.extend(summary["cells"])

    source_rows: list[dict[str, Any]] = []
    for model in ("qwen3", "glm4"):
        summary = read_json(
            OUT_ROOT / "source" / model / "discovery" / "summary.json"
        )
        if summary["protocol_digest"] != protocol_digest:
            raise RuntimeError(f"Source digest mismatch: {model}")
        for cell in summary["cells"]:
            if not cell["source_run"]:
                continue
            final = cell["final_conditions"]
            source_rows.append(
                {
                    "model": model,
                    "split": cell["split"],
                    "template": cell["template"],
                    "contrast": cell["contrast"],
                    "frozen_positions": cell["frozen_positions"],
                    "frozen_position_count": cell[
                        "frozen_position_count"
                    ],
                    "semantic_role_audit": cell["semantic_role_audit"],
                    "semantic_labels_used_for_selection": cell[
                        "semantic_labels_used_for_selection"
                    ],
                    "within_minimal_replace": condition_view(
                        final["within_minimal_replace"]
                    ),
                    "cross_world_whole": condition_view(
                        final["cross_world_whole"]
                    ),
                    "cross_world_same_answer_whole": condition_view(
                        final["cross_world_same_answer_whole"]
                    ),
                    "causal_delta": condition_view(final["causal_delta"]),
                    "nuisance_delta": condition_view(
                        final["nuisance_delta"]
                    ),
                    "target_noop": condition_view(final["target_noop"]),
                    "whole_source_gate": cell["whole_source_gate"],
                    "delta_source_gate": cell["delta_source_gate"],
                }
            )

    discovery_by_model_template: dict[
        tuple[str, int], set[str]
    ] = defaultdict(set)
    for row in source_rows:
        if row["delta_source_gate"]:
            discovery_by_model_template[
                (row["model"], row["template"])
            ].add(row["contrast"])

    discovery_models = sorted(
        {
            model
            for (model, _), passed in discovery_by_model_template.items()
            if set(CONTRASTS).issubset(passed)
        }
    )
    discovery_parent_pass = len(discovery_models) >= 2
    confirmation_authorized = discovery_parent_pass
    holdout_authorized = False
    downstream_authorized = False

    binding_rows = [
        row for row in source_rows if row["contrast"] == "binding_flip"
    ]
    query_rows = [
        row for row in source_rows if row["contrast"] == "query_flip"
    ]
    repeated_structure = {
        "binding_flip": {
            "models_observed": sorted(row["model"] for row in binding_rows),
            "frozen_position_count_by_model": {
                row["model"]: row["frozen_position_count"]
                for row in binding_rows
            },
            "post_hoc_roles": {
                row["model"]: row["semantic_role_audit"]
                for row in binding_rows
            },
            "interpretation_limit": (
                "Repeated four value-word positions at depth 1; this is a "
                "task-local causal prompt structure, not a permanent "
                "knowledge address or neuron-level mechanism."
            ),
        },
        "query_flip": {
            "models_observed": sorted(row["model"] for row in query_rows),
            "frozen_position_count_by_model": {
                row["model"]: row["frozen_position_count"]
                for row in query_rows
            },
            "post_hoc_roles": {
                row["model"]: row["semantic_role_audit"]
                for row in query_rows
            },
            "interpretation_limit": (
                "A single query-name position is sufficient within a fixed "
                "world, but its whole state and matched delta do not transfer "
                "cleanly across worlds."
            ),
        },
    }

    formal_behavior_n = sum(row["n"] for row in behavior_rows)
    payload = {
        "schema_version": "phase1007_final.v1",
        "phase": PHASE,
        "protocol_digest": protocol_digest,
        "protocol_revision": protocol["protocol_revision"],
        "formal_behavior_case_count": formal_behavior_n,
        "formal_behavior_cell_count": len(behavior_rows),
        "behavior_gate_pass_count_by_model": behavior_counts,
        "source_cell_count": len(source_rows),
        "whole_source_gate_pass_count": sum(
            int(row["whole_source_gate"]) for row in source_rows
        ),
        "delta_source_gate_pass_count": sum(
            int(row["delta_source_gate"]) for row in source_rows
        ),
        "discovery_models_passing_both_contrasts": discovery_models,
        "discovery_parent_pass": discovery_parent_pass,
        "confirmation_authorized": confirmation_authorized,
        "holdout_authorized": holdout_authorized,
        "downstream_temporal_kv_receiver_neuron_authorized": (
            downstream_authorized
        ),
        "automatic_decision": (
            "stop_before_confirmation_and_downstream_decomposition"
            if not discovery_parent_pass
            else "run_confirmation"
        ),
        "stop_reason": (
            "query_flip failed the clean cross-world whole and matched-delta "
            "source gates in both Qwen3 and GLM4"
            if not discovery_parent_pass
            else None
        ),
        "repeated_structure": repeated_structure,
        "source_cells": source_rows,
        "script_sha256": {
            name: sha256(ROOT / "tests" / "glm5" / name)
            for name in SCRIPT_NAMES
        },
        "claims_not_authorized": [
            "abstract native role variable",
            "autoregressive temporal aggregation mechanism",
            "KV-cache carrier mechanism",
            "attention/MLP/head/channel/neuron mechanism",
            "cross-template or cross-model language law",
        ],
    }
    final_root = OUT_ROOT / "final"
    write_jsonl(final_root / "source_cell_matrix.jsonl", source_rows)
    write_json(final_root / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
