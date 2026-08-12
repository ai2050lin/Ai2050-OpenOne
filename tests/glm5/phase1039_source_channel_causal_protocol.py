#!/usr/bin/env python3
"""Freeze causal tests of repeated Attention and MLP source writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1035_native_family_routing_protocol as source
import phase1037_family_source_causal_protocol as baseline
import phase1038_source_channel_protocol as channel_atlas


PHASE = 1039
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = source.PRECISION
QUANTIZATION = source.QUANTIZATION
SOURCE_ROOT = source.OUT_ROOT
BASELINE_ROOT = baseline.OUT_ROOT
EVIDENCE_ROOT = channel_atlas.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1039_source_channel_causal"
)
CHANNELS = ("attention_write", "mlp_write")
CONDITIONS = (
    "same_family_selected",
    "cross_family_selected",
    "cross_family_unselected",
    "cross_family_wrong_target",
)


write_json = source.write_json
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    baseline_prereg = read_json(
        BASELINE_ROOT / "protocol" / "preregistration.json"
    )
    evidence_prereg = read_json(
        EVIDENCE_ROOT / "protocol" / "preregistration.json"
    )
    evidence = read_json(EVIDENCE_ROOT / "aggregate.json")
    decision = evidence["automatic_next_decision"]
    if not decision["causal_followup_needed"]:
        raise RuntimeError("Phase1038 did not authorize a causal follow-up")
    if set(decision["eligible_channels"]) != set(CHANNELS):
        raise RuntimeError("eligible channel set drift")

    model_depths = {
        model: [
            int(value)
            for value in baseline_prereg[
                "model_physical_depths"
            ][model]
        ]
        for model in MODELS
    }
    prereg_core: dict[str, Any] = {
        "schema_version": "phase1039_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Causal use of repeated Attention and MLP source-write "
            "differences"
        ),
        "source_phase": source.PHASE,
        "source_protocol_digest": source_prereg["protocol_digest"],
        "whole_state_baseline_phase": baseline.PHASE,
        "whole_state_baseline_protocol_digest": baseline_prereg[
            "protocol_digest"
        ],
        "evidence_phase": channel_atlas.PHASE,
        "evidence_protocol_digest": evidence_prereg["protocol_digest"],
        "evidence_candidate_cell_count": len(
            evidence["causal_candidate_cells"]
        ),
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "confirmation_target_count": baseline_prereg["target_count"],
        "channels": list(CHANNELS),
        "conditions": list(CONDITIONS),
        "model_physical_depths": model_depths,
        "normalized_depth_slots": list(
            baseline_prereg["normalized_depth_slots"]
        ),
        "selection_rule": (
            "Test both actual write-channel types at all three depths frozen "
            "in Phase1037. No component magnitude, causal peak, head, neuron, "
            "or fitted direction selects a row."
        ),
        "intervention": (
            "At the selected layer output and target source position, add "
            "the clean donor-channel value minus the clean target-channel "
            "value: h' = h + (c_donor - c_target). All other accumulated "
            "residual terms at that position remain from the target. The "
            "difference is not scaled or projected."
        ),
        "controls": {
            "same_family_selected": (
                "Same semantic family, different lexical member, queried "
                "source role."
            ),
            "cross_family_selected": (
                "Opposite family at the queried source role."
            ),
            "cross_family_unselected": (
                "Opposite family at the unqueried source role."
            ),
            "cross_family_wrong_target": (
                "Queried-role donor channel inserted at the unqueried target "
                "role."
            ),
            "zero_delta_identity": (
                "Audit donor=target channel deltas directly from the clean "
                "cache; a separate zero-effect forward is not used."
            ),
            "whole_state_baseline": (
                "Reuse the exact Phase1037 complete-state interventions on "
                "the same targets and depths; do not spend a new forward pass "
                "to reproduce an already frozen baseline."
            ),
        },
        "readouts": {
            "candidate_margin": (
                "Cross-family candidate logit minus target-family candidate "
                "logit, relative to the clean target."
            ),
            "candidate_top1": "Top family among the eight frozen candidates.",
            "internal_prototype": (
                "Discovery-only penultimate residual family prototypes, "
                "reported separately from output logits."
            ),
            "scale_free_audits": [
                "paired selected-minus-unselected shift",
                "paired selected-minus-wrong-target shift",
                "cross-effect to same-family absolute-effect ratio",
                "cross-effect retention relative to Phase1037 whole state",
            ],
        },
        "single_channel_gate": {
            "cross_selected_shift_median_min": 0.0,
            "cross_selected_positive_rate_min": 0.65,
            "selected_minus_unselected_median_min": 0.0,
            "selected_minus_wrong_target_median_min": 0.0,
            "cross_to_same_absolute_ratio_min": 2.0,
            "whole_state_effect_retention_min": 0.10,
            "both_confirmation_templates_required": True,
            "minimum_models_same_channel_and_depth_slot": 2,
        },
        "automatic_followup_rule": (
            "If no single write channel passes in at least two models at the "
            "same channel and normalized depth slot, run a separately frozen "
            "current-write alliance and upstream-residual decomposition. "
            "A failed single-channel gate must not erase Phase1038 geometry."
        ),
        "claim_limits": [
            (
                "A channel-delta effect establishes local use of that clean "
                "difference in this task; it does not identify a pure "
                "semantic code."
            ),
            (
                "Adding a clean component difference at the layer output is "
                "an intervention on an additive write, not a replay of the "
                "component's complete internal computation."
            ),
            (
                "Failure can mean insufficiency, nonlinear cooperation, or "
                "distributed redundancy; it does not refute the observed "
                "component geometry."
            ),
        ],
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "channels": list(CHANNELS),
        "conditions": list(CONDITIONS),
        "model_physical_depths": model_depths,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
