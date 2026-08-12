#!/usr/bin/env python3
"""Freeze the Phase1038 source-state computation-channel atlas.

Phase1037 showed that a complete queried-source state has strong causal
influence but also carries a large same-family lexical side effect.  This
phase does not assume that a pure semantic channel exists.  It measures the
actual additive terms at uniformly frozen Transformer depths:

    layer output = layer input + attention write + MLP write

The equation above is an instrumentation identity for the local block, not a
theory of language encoding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1035_native_family_routing_protocol as source
import phase1036_family_contrast_protocol as geometry
import phase1037_family_source_causal_protocol as causal


PHASE = 1038
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = source.PRECISION
QUANTIZATION = source.QUANTIZATION
SOURCE_ROOT = source.OUT_ROOT
GEOMETRY_ROOT = geometry.OUT_ROOT
CAUSAL_ROOT = causal.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1038_source_channel_atlas"
)
ROLE_ANCHORS = ("concept_a", "concept_b")
LEXICAL_MEMBERS = (0, 1)
CHANNELS = (
    "upstream_residual",
    "attention_write",
    "mlp_write",
    "layer_output",
)


write_json = source.write_json
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    geometry_prereg = read_json(
        GEOMETRY_ROOT / "protocol" / "preregistration.json"
    )
    causal_prereg = read_json(
        CAUSAL_ROOT / "protocol" / "preregistration.json"
    )
    causal_aggregate = read_json(CAUSAL_ROOT / "aggregate.json")

    model_depths: dict[str, list[int]] = {}
    normalized_depth_slots: dict[str, list[int]] = {}
    for model in MODELS:
        all_depths = [
            int(value)
            for value in geometry_prereg[
                "uniform_physical_depths"
            ][model]
        ]
        # Depth zero has no block write, and the final depth is excluded to
        # avoid the known DeepSeek FP16 boundary instability.
        model_depths[model] = all_depths[1:-1]
        normalized_depth_slots[model] = list(
            range(1, len(all_depths) - 1)
        )

    prereg_core: dict[str, Any] = {
        "schema_version": "phase1038_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Observed source-state computation channels and lexical-family "
            "separation atlas"
        ),
        "source_phase": source.PHASE,
        "source_protocol_digest": source_prereg["protocol_digest"],
        "geometry_phase": geometry.PHASE,
        "geometry_protocol_digest": geometry_prereg["protocol_digest"],
        "causal_phase": causal.PHASE,
        "causal_protocol_digest": causal_prereg["protocol_digest"],
        "causal_result": {
            "cross_model_gate_passed": causal_aggregate[
                "cross_model_causal_result"
            ]["passed"],
            "automatic_route": causal_aggregate[
                "automatic_next_decision"
            ]["route"],
        },
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "source_cases": {
            "units": 256,
            "worlds_per_unit": 8,
            "discovery_units": 128,
            "confirmation_units": 128,
            "factors": ["binding", "query", "lexical_member"],
        },
        "channels": list(CHANNELS),
        "channel_identity": {
            "upstream_residual": (
                "The residual stream entering the selected Transformer block."
            ),
            "attention_write": (
                "The complete projected self-attention output added by that "
                "block at the source position; no head ranking."
            ),
            "mlp_write": (
                "The complete MLP output added by that block at the source "
                "position; no neuron ranking."
            ),
            "layer_output": (
                "The complete residual stream after the selected block."
            ),
        },
        "instrumentation_identity": (
            "At each measured source position, audit layer_output minus "
            "upstream_residual minus attention_write minus mlp_write. This is "
            "only a Transformer block accounting identity."
        ),
        "model_physical_depths": model_depths,
        "normalized_depth_slots": normalized_depth_slots,
        "depth_selection_rule": (
            "Use every nonzero, nonfinal uniformly spaced depth frozen before "
            "this phase. No activation, cosine, causal effect, or output peak "
            "selects a depth."
        ),
        "observations": {
            "ordered_family_contrast": (
                "For each lexical member and fact role, preserve the actual "
                "donor-family minus target-family B contrast vector."
            ),
            "same_family_lexical_contrast": (
                "For each binding world and fact role, record the norm of the "
                "L1 minus L0 channel difference."
            ),
            "same_pair_cross_split": (
                "Compare discovery and confirmation centroids for the same "
                "ordered family pair; words, nonces, and templates are "
                "disjoint."
            ),
            "shuffled_pair_control": (
                "Compare against a deterministic different ordered-family "
                "pair."
            ),
            "family_to_lexical_norm_ratio": (
                "Per unit, divide mean family-contrast norm by mean "
                "same-family lexical-contrast norm. This is descriptive and "
                "not a purity law."
            ),
        },
        "descriptive_channel_gate": {
            "same_pair_cross_split_median_min": 0.0,
            "same_pair_positive_rate_min": 0.75,
            "matched_minus_shuffled_median_min": 0.10,
            "minimum_models": 2,
            "eligible_causal_channels": [
                "attention_write",
                "mlp_write",
            ],
        },
        "automatic_followup_rule": (
            "Run a new causal channel-delta phase only if at least one actual "
            "write channel passes the same-pair-over-shuffled gate in at "
            "least two models at a conserved normalized depth and role. "
            "Otherwise move to a separately preregistered multi-position "
            "source-span atlas."
        ),
        "claim_limits": [
            (
                "A repeated component contrast is an observed representational "
                "regularity, not proof that the component transports a pure "
                "semantic variable."
            ),
            (
                "The additive block identity does not imply that language "
                "itself is linearly decomposed."
            ),
            (
                "Low lexical norm does not establish context independence; "
                "high lexical norm does not refute distributed semantic use."
            ),
            (
                "No result in this phase establishes biological optimality, "
                "brain-model isomorphism, or a universal language equation."
            ),
        ],
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_physical_depths": model_depths,
        "channels": list(CHANNELS),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
