#!/usr/bin/env python3
"""Posthoc, non-upgrading divergence ledger for the Phase1119 scale result."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from typing import Any

import phase1119_qwen3_scale_protocol as protocol


def paired(details: list[dict[str, Any]], case_by_index: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[row["pair_id"]].append(row)
    output: dict[str, dict[str, Any]] = {}
    for pair_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["sense"])
        source = case_by_index[rows[0]["case_index"]]
        true_d = rows[0]["true_z"] - rows[1]["true_z"]
        control_d = rows[0]["control_z"] - rows[1]["control_z"]
        output[pair_id] = {
            "concept_id": rows[0]["concept_id"],
            "split": rows[0]["split"],
            "template": rows[0]["template"],
            "source_phase": source["source_phase"],
            "true_d": true_d,
            "control_d": control_d,
            "direction": true_d > 0.0,
            "control_direction": control_d > 0.0,
            "bidirectional": rows[0]["true_z"] > 0.0 and rows[1]["true_z"] < 0.0,
        }
    return output


def transition(left: bool, right: bool) -> str:
    return f"{int(left)}->{int(right)}"


def accuracy(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def main() -> None:
    cases = list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.jsonl"))
    case_by_index = {row["case_index"]: row for row in cases}
    details = {
        model: list(
            protocol.read_jsonl(
                protocol.OUT_ROOT / "behavior" / model / "candidate_detail.jsonl"
            )
        )
        for model in protocol.MODEL_ROOTS
    }
    panels = {model: paired(rows, case_by_index) for model, rows in details.items()}
    if set(panels["qwen3_4b"]) != set(panels["qwen3_14b"]):
        raise RuntimeError("pair identity mismatch")

    pair_ids = sorted(panels["qwen3_4b"])
    pair_transitions = {
        key: Counter(
            transition(panels["qwen3_4b"][pair_id][key], panels["qwen3_14b"][pair_id][key])
            for pair_id in pair_ids
        )
        for key in ("direction", "control_direction", "bidirectional")
    }
    concept_values: dict[str, dict[str, list[float]]] = {
        model: defaultdict(list) for model in panels
    }
    for model, panel in panels.items():
        for row in panel.values():
            concept_values[model][row["concept_id"]].append(row["true_d"])
    concept_signs = {
        model: {
            concept: statistics.median(values) > 0.0
            for concept, values in values_by_concept.items()
        }
        for model, values_by_concept in concept_values.items()
    }
    concept_transitions = Counter(
        transition(concept_signs["qwen3_4b"][concept], concept_signs["qwen3_14b"][concept])
        for concept in sorted(concept_signs["qwen3_4b"])
    )

    panels_by_factor: dict[str, Any] = {}
    factors = {
        "split": list(protocol.SPLITS),
        "source_phase": [1114, 1115],
        "template": list(range(protocol.TEMPLATE_COUNT)),
    }
    for factor, levels in factors.items():
        panels_by_factor[factor] = {}
        for level in levels:
            small = [row for row in panels["qwen3_4b"].values() if row[factor] == level]
            large = [row for row in panels["qwen3_14b"].values() if row[factor] == level]
            panels_by_factor[factor][str(level)] = {
                "pair_count": len(small),
                "direction_4b": accuracy(small, "direction"),
                "direction_14b": accuracy(large, "direction"),
                "direction_gain": accuracy(large, "direction") - accuracy(small, "direction"),
                "bidirectional_4b": accuracy(small, "bidirectional"),
                "bidirectional_14b": accuracy(large, "bidirectional"),
                "bidirectional_gain": accuracy(large, "bidirectional")
                - accuracy(small, "bidirectional"),
            }

    core = {
        "schema_version": "phase1119_scale_divergence_diagnostic.v1",
        "phase": protocol.PHASE,
        "status": "posthoc_non_upgrading",
        "pair_transitions": {
            key: dict(sorted(value.items())) for key, value in pair_transitions.items()
        },
        "concept_transitions": dict(sorted(concept_transitions.items())),
        "factor_panels": panels_by_factor,
        "interpretation_limit": (
            "These counts localize heterogeneity in the frozen result; they do not alter "
            "the failed prospective scale gate or authorize subgroup claims."
        ),
    }
    result = dict(core)
    result["diagnostic_digest"] = protocol.digest(core)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "divergence_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
