#!/usr/bin/env python3
"""Freeze the Phase1036 semantic-family contrast controls.

Phase1035 found positive cross-lexical family contrast at fact slots, but a
positive cosine alone can arise from residual anisotropy or generic role
geometry.  Phase1036 preserves the actual contrast directions at uniformly
spaced depths and compares:

1. the same ordered family pair across disjoint lexical/template splits, and
2. a deterministic different-family-pair control.

No activation value is used to choose a depth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1035_native_family_routing_protocol as source


PHASE = 1036
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = source.PRECISION
QUANTIZATION = source.QUANTIZATION
SOURCE_ROOT = source.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1036_family_contrast_controls"
)
ROLE_ANCHORS = ("concept_a", "concept_b")
LEXICAL_MEMBERS = (0, 1)
NORMALIZED_DEPTH_DIVISIONS = 8


write_json = source.write_json
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest


def uniform_depths(n_layers: int) -> list[int]:
    return sorted({
        int(i * n_layers // NORMALIZED_DEPTH_DIVISIONS)
        for i in range(NORMALIZED_DEPTH_DIVISIONS + 1)
    } | {n_layers})


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    depths = {}
    for model in MODELS:
        summary = read_json(
            SOURCE_ROOT / "atlas" / model / "summary.json"
        )
        depths[model] = uniform_depths(
            int(summary["model_info"]["n_layers"])
        )
    prereg_core: dict[str, Any] = {
        "schema_version": "phase1036_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Ordered semantic-family contrast specificity and shuffled controls"
        ),
        "source_phase": source.PHASE,
        "source_protocol_digest": source_prereg["protocol_digest"],
        "source_phase_result": {
            "conserved_event_cell_count": len(
                source_aggregate["conserved_confirmed_event_cells"]
            ),
            "conserved_source_cell_count": len(
                source_aggregate[
                    "conserved_confirmed_source_family_cells"
                ]
            ),
            "causal_followup_needed": source_aggregate[
                "automatic_next_decision"
            ]["causal_followup_needed"],
        },
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "uniform_physical_depths": depths,
        "selection_rule": (
            "Depths are fixed by uniform normalized spacing only; no "
            "Phase1035 activation peak is used."
        ),
        "saved_directions": (
            "For each unit, depth, fact role, and lexical member, save the "
            "canonical donor-family minus target-family binding contrast."
        ),
        "canonical_role_sign": {
            "concept_a": (
                "+1 when q0_slot is a, otherwise -1, so the direction is "
                "always donor family minus target family."
            ),
            "concept_b": (
                "The opposite sign, making both physical roles comparable."
            ),
        },
        "comparisons": {
            "within_unit_member_invariance": (
                "Cosine between the two lexical-member family contrasts."
            ),
            "same_pair_cross_context": (
                "Cosine among contexts sharing the ordered target/donor "
                "family pair inside each split."
            ),
            "same_pair_cross_split": (
                "Cosine between discovery and confirmation centroids for "
                "the same ordered family pair; all words, nonces, and "
                "templates differ."
            ),
            "shuffled_pair_cross_split": (
                "Cosine between a discovery family-pair centroid and a "
                "deterministically rotated, different confirmation pair."
            ),
            "matched_advantage": (
                "Same-pair cross-split cosine minus shuffled-pair cosine."
            ),
            "output_bq_member_invariance": (
                "The same BxQ lexical-member invariance computed directly "
                "in the eight-candidate logit vector from Phase1035."
            ),
        },
        "descriptive_evidence_gate": {
            "same_pair_cross_split_median_min": 0.0,
            "same_pair_positive_rate_min": 0.75,
            "matched_minus_shuffled_median_min": 0.10,
            "required_nonfinal_depths_per_role": 4,
            "minimum_models": 2,
        },
        "causal_followup_gate": {
            "family_contrast_gate_passed": True,
            "confirmation_output_bq_member_invariance_median_min": 0.0,
            "confirmation_output_bq_member_positive_rate_min": 0.75,
            "confirmation_candidate_accuracy_min": 0.275,
            "minimum_models": 2,
        },
        "claim_limits": [
            (
                "Matched-over-shuffled directions support a category-pair-"
                "specific representational geometry, not a complete natural "
                "knowledge graph."
            ),
            (
                "A source direction can be readable and repeated without "
                "being a sufficient causal transporter."
            ),
            (
                "Positive evidence does not establish neural or biological "
                "optimality."
            ),
        ],
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_protocol_digest": prereg["source_protocol_digest"],
        "uniform_physical_depths": depths,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
