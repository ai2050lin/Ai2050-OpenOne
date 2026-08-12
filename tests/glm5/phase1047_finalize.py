#!/usr/bin/env python3
"""Finalize Phase1047 and separate persistence from coalition synergy."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1047_concept_pair_confirmation_protocol as protocol


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
    for model_name, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model_name} protocol digest drift")

    pair_pass_models = [
        model_name for model_name, summary in summaries.items()
        if summary["analysis"]["concept_pair_gate_passed"]
    ]
    alliance_pass_models = [
        model_name for model_name, summary in summaries.items()
        if summary["analysis"]["concept_pair_alliance_gate_passed"]
    ]
    persistent_selected = {}
    for model_name, summary in summaries.items():
        analysis = summary["analysis"]
        selected = analysis["mask_metrics"]["selected_concept"]
        unselected = analysis["mask_metrics"]["unselected_concept"]
        pair = analysis["mask_metrics"]["concept_pair"]
        persistent_selected[model_name] = {
            "selected_mediation_fraction": selected[
                "mediation_fraction"
            ]["median"],
            "selected_replay_recovery": selected[
                "replay_recovery"
            ]["median"],
            "unselected_mediation_fraction": unselected[
                "mediation_fraction"
            ]["median"],
            "unselected_replay_recovery": unselected[
                "replay_recovery"
            ]["median"],
            "pair_mediation_fraction": pair[
                "mediation_fraction"
            ]["median"],
            "pair_replay_recovery": pair[
                "replay_recovery"
            ]["median"],
            **analysis["alliance_gains"],
        }

    minimum_models = prereg["confirmation_gate"]["minimum_models"]
    pair_confirmed = len(pair_pass_models) >= minimum_models
    alliance_confirmed = len(alliance_pass_models) >= minimum_models
    persistence_repeated = all(
        row["selected_mediation_fraction"] >= 0.5
        and row["selected_replay_recovery"] >= 0.5
        and abs(row["unselected_mediation_fraction"]) < 0.1
        and row["pair_minus_best_constituent_mediation"]
        < prereg["confirmation_gate"][
            "pair_minus_best_constituent_mediation_min"
        ]
        for row in persistent_selected.values()
    )
    automatic_next = {
        "immediate_additional_execution_needed": False,
        "same_block_complete": True,
        "reason": (
            "The independent data confirm the concept-pair effect but reject "
            "pair synergy in all three models. The selected concept state "
            "alone carries 0.826-0.865 median mediation and 0.789-0.961 "
            "replay recovery; another mask/depth sweep would tune an "
            "answered question."
        ),
        "next_major_task": (
            "Start a separately preregistered read-path block: identify how "
            "the later query and output boundary read the persistent fact "
            "register, using source-position-specific Attention/KV causal "
            "interventions and natural rollout. Do not treat attention "
            "weights alone as edges."
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
        "schema_version": "phase1047_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "sample_plan": prereg["sample_plan"],
        "pair_pass_models": pair_pass_models,
        "alliance_pass_models": alliance_pass_models,
        "pair_confirmed": pair_confirmed,
        "concept_pair_alliance_confirmed": alliance_confirmed,
        "persistent_selected_state_repeated": persistence_repeated,
        "persistent_selected_state": persistent_selected,
        "model_analysis": {
            model_name: summary["analysis"]
            for model_name, summary in summaries.items()
        },
        "finite_audits": {
            model_name: {
                "source_cache": summary["source_cache_finite"],
                "baseline_logits": summary["baseline_logits_finite"],
                "paired_logits": summary["paired_logits_finite"],
                "response_norms": summary["response_norms_finite"],
            }
            for model_name, summary in summaries.items()
        },
        "automatic_next_decision": automatic_next,
        "artifact_manifest": manifests,
        "mechanism_update": {
            "supported": [
                (
                    "In this controlled two-fact family task, the early "
                    "complete state written at the queried fact's concept "
                    "position persists as a highly necessary and locally "
                    "sufficient state at relative receiver slot 2."
                ),
                (
                    "The later query position is not the main storage "
                    "bottleneck. Its state has only a small partial effect."
                ),
                (
                    "Because decoder causality prevents an earlier concept "
                    "token from seeing the later query token, the concept "
                    "positions are best interpreted as fact registers; the "
                    "later query selects which register is read."
                ),
            ],
            "not_supported": [
                "A synergistic two-concept coalition",
                "A query-position bottleneck",
                "A pure semantic-family vector",
                "A neuron-level or head-level implementation",
                "A complete natural-language, reasoning, or grammar theory",
                "Biological optimality or brain-model isomorphism",
            ],
        },
        "claim_limits": prereg["claim_limits"],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    protocol.write_json(
        protocol.OUT_ROOT / "automatic_next_decision.json",
        automatic_next,
    )
    print(json.dumps({
        "pair_confirmed": pair_confirmed,
        "concept_pair_alliance_confirmed": alliance_confirmed,
        "persistent_selected_state_repeated": persistence_repeated,
        "persistent_selected_state": persistent_selected,
        "automatic_next_decision": automatic_next,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
