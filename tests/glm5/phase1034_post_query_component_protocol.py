#!/usr/bin/env python3
"""Freeze the Phase1034 post-query component atlas protocol."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1033_alliance_replication_protocol as source


PHASE = 1034
MODELS = source.MODELS
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1034_post_query_component_atlas"
)
SOURCE_ROOT = source.OUT_ROOT
ANCHORS = (
    "query_end",
    "suffix_first",
    "suffix_quarter",
    "suffix_mid",
    "pre_output",
)
COMPONENTS = ("attention", "mlp", "residual")
RESPONSE_METRICS = (
    "binding_q0_relative",
    "binding_q1_relative",
    "query_b0_relative",
    "query_b1_relative",
    "interaction_relative",
    "binding_context_cosine",
    "query_context_cosine",
    "selected_source_alignment",
    "interaction_source_cosine",
)
SOURCE_METRICS = (
    "relative_pair_difference",
    "future_query_leakage",
    "pair_opposition",
)
WRITE_METRICS = ("alignment_to_residual_write", "signed_fraction")
DEPTH_BIN_COUNT = 8


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def anchor_positions(row: dict[str, Any]) -> list[int]:
    query_end = int(row["role_spans"]["query_nonce"][1])
    pre_output = int(row["role_spans"]["pre_output"][1])
    suffix_length = pre_output - query_end
    if suffix_length < 1:
        raise ValueError(
            f"no post-query suffix in {row['record_id']}: "
            f"{query_end} -> {pre_output}"
        )
    quarter = max(1, math.ceil(suffix_length / 4))
    middle = max(1, math.ceil(suffix_length / 2))
    return [
        query_end,
        query_end + 1,
        query_end + quarter,
        query_end + middle,
        pre_output,
    ]


def audit_model(model: str) -> dict[str, Any]:
    rows = read_jsonl(
        SOURCE_ROOT / "protocol" / f"cases.{model}.jsonl"
    )
    checks = {
        "case_count_2048": len(rows) == 2048,
        "four_world_order": True,
        "unit_rows_contiguous": True,
        "anchors_in_bounds": True,
        "anchors_monotonic": True,
        "concept_spans_at_most_two": True,
    }
    suffix_counts: dict[str, int] = {}
    duplicate_anchor_counts: dict[str, int] = {}
    expected_worlds = ["00", "10", "01", "11"]

    for start in range(0, len(rows), 4):
        group = rows[start:start + 4]
        checks["four_world_order"] &= (
            [row["world"] for row in group] == expected_worlds
        )
        checks["unit_rows_contiguous"] &= (
            len({int(row["unit_index"]) for row in group}) == 1
        )
    for row in rows:
        positions = anchor_positions(row)
        checks["anchors_in_bounds"] &= all(
            0 <= value < len(row["input_ids"]) for value in positions
        )
        checks["anchors_monotonic"] &= all(
            left <= right
            for left, right in zip(positions, positions[1:])
        )
        for role in ("concept_a", "concept_b"):
            start, end = (
                int(value) for value in row["role_spans"][role]
            )
            checks["concept_spans_at_most_two"] &= end - start + 1 <= 2
        suffix = (
            int(row["role_spans"]["pre_output"][1])
            - int(row["role_spans"]["query_nonce"][1])
        )
        suffix_counts[str(suffix)] = suffix_counts.get(str(suffix), 0) + 1
        duplicates = len(positions) - len(set(positions))
        duplicate_anchor_counts[str(duplicates)] = (
            duplicate_anchor_counts.get(str(duplicates), 0) + 1
        )
    return {
        "schema_version": "phase1034_model_audit.v1",
        "phase": PHASE,
        "model": model,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "suffix_length_counts": suffix_counts,
        "duplicate_anchor_counts": duplicate_anchor_counts,
    }


def main() -> None:
    source.main()
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    model_audits = {model: audit_model(model) for model in MODELS}
    if not all(
        row["all_checks_passed"] for row in model_audits.values()
    ):
        raise RuntimeError("Phase1034 protocol audit failed")

    frozen = {
        "phase": PHASE,
        "source_phase": source.PHASE,
        "source_protocol_digest": source_prereg["protocol_digest"],
        "models": list(MODELS),
        "precision": "float16",
        "quantization": False,
        "unit_count": 512,
        "case_count": 2048,
        "world_order": ["00", "10", "01", "11"],
        "anchors": list(ANCHORS),
        "anchor_rule": {
            "query_end": "last token of the complete query nonce",
            "suffix_first": "first causal token after the query nonce",
            "suffix_quarter": "ceil(25% of query-to-readout suffix)",
            "suffix_mid": "ceil(50% of query-to-readout suffix)",
            "pre_output": "last prompt token before next-token prediction",
        },
        "components": list(COMPONENTS),
        "response_metrics": list(RESPONSE_METRICS),
        "source_metrics": list(SOURCE_METRICS),
        "write_metrics": list(WRITE_METRICS),
        "depth_bin_count": DEPTH_BIN_COUNT,
        "analysis_principles": [
            "Scan every layer; do not select the largest neuron, head, or layer.",
            "Treat B, Q, and BxQ as measurements, not a mechanism law.",
            "Report both templates and both token-length banks separately.",
            "Use component outputs only when residual-addition closure is audited.",
            "Do not treat response similarity as a natural causal edge.",
        ],
        "descriptive_repeat_rule": {
            "binding_context_cosine_max": -0.15,
            "negative_cosine_prevalence_min": 0.65,
            "minimum_models": 2,
            "must_hold_in_each_template": True,
            "purpose": (
                "Label repeated query-conditioned reversal cells; this is "
                "an atlas annotation, not a closed mechanism equation."
            ),
        },
        "instrumentation_gate": {
            "residual_addition_relative_error_p95_max": 0.02,
            "all_recorded_values_finite": True,
        },
        "automatic_next_rule": (
            "Only if a suffix component/depth band repeats in at least two "
            "models and passes instrumentation, perform a separately "
            "preregistered causal test of the whole band. Otherwise preserve "
            "the atlas and redesign the role controls."
        ),
        "model_audits": model_audits,
    }
    frozen["protocol_digest"] = digest(frozen)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", frozen)
    for model, audit in model_audits.items():
        write_json(OUT_ROOT / "protocol" / f"audit.{model}.json", audit)
    write_json(
        OUT_ROOT / "protocol" / "audit.json",
        {
            "phase": PHASE,
            "all_models_passed": all(
                row["all_checks_passed"]
                for row in model_audits.values()
            ),
            "source_protocol_digest": source_prereg["protocol_digest"],
            "protocol_digest": frozen["protocol_digest"],
        },
    )
    print(
        json.dumps(
            {
                "phase": PHASE,
                "protocol_digest": frozen["protocol_digest"],
                "model_audits": model_audits,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
