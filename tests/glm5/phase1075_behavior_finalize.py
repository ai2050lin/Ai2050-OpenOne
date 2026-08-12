#!/usr/bin/env python3
"""Finalize Phase1075 relation-level behavior authorization."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1075_relation_polarity_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "summary.json"
        )
        for model in protocol.MODELS
    }
    authorized_models_by_relation = {}
    repeated_relations = []
    for relation in protocol.RELATIONS:
        models = [
            model
            for model, summary in summaries.items()
            if summary["relations"][relation][
                "relation_behavior_gate_passed"
            ]
        ]
        authorized_models_by_relation[relation] = models
        if (
            len(models)
            >= prereg["gates"]["minimum_models_per_relation"]
        ):
            repeated_relations.append(relation)

    selected_relations = [
        relation
        for relation in protocol.RELATION_PRIORITY
        if relation in repeated_relations
    ][:int(prereg["max_internal_relations"])]
    selected_models = [
        model
        for model in protocol.MODELS
        if any(
            model in authorized_models_by_relation[relation]
            for relation in selected_relations
        )
    ]
    should_continue = bool(selected_relations)
    payload = {
        "schema_version": "phase1075_behavior_decision.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "authorized_models_by_relation": (
            authorized_models_by_relation
        ),
        "repeated_relations": repeated_relations,
        "selected_relations": selected_relations,
        "selected_models": selected_models,
        "should_run_internal_mapping": should_continue,
        "route": (
            "run_heldout_relation_internal_mapping"
            if should_continue
            else "stop_at_heldout_behavior"
        ),
        "reason": (
            "At least one relation passed every fresh relation-level "
            "gate in two or more models."
            if should_continue
            else (
                "No relation passed every fresh relation-level gate in "
                "two or more models; internal operation claims are not "
                "authorized."
            )
        ),
        "model_relation_summary": {
            model: {
                relation: {
                    "candidate_finite_rate": summary["relations"][
                        relation
                    ]["candidate_finite_rate"],
                    "candidate_accuracy": summary["relations"][
                        relation
                    ]["candidate_accuracy"],
                    "by_task": summary["relations"][relation][
                        "by_task"
                    ],
                    "by_path": summary["relations"][relation][
                        "by_path"
                    ],
                    "confirmation_candidate_accuracy": summary[
                        "relations"
                    ][relation][
                        "confirmation_candidate_accuracy"
                    ],
                    "natural_semantic_first_rate": summary[
                        "relations"
                    ][relation]["natural_semantic_first_rate"],
                    "gate_passed": summary["relations"][relation][
                        "relation_behavior_gate_passed"
                    ],
                }
                for relation in protocol.RELATIONS
            }
            for model, summary in summaries.items()
        },
    }
    payload["decision_digest"] = protocol.digest(payload)
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "behavior_decision.json",
        payload,
    )
    internal_prereg = {
        "schema_version": "phase1075_internal_preregistration.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_decision_digest": payload["decision_digest"],
        "selected_relations": selected_relations,
        "authorized_models_by_relation": {
            relation: authorized_models_by_relation[relation]
            for relation in selected_relations
        },
        "internal_replicates": list(protocol.INTERNAL_REPLICATES),
        "candidate_selection": {
            "source_split": "discovery",
            "evaluation_split": "confirmation",
            "routing_metric": "attention_mass",
            "routing_destination": "answer_boundary",
            "routing_source_pair": "fact",
            "top_heads_per_relation_model": 8,
            "ranking": (
                "discovery mean positive route selectivity, then "
                "discovery positive fraction, then depth/head"
            ),
        },
        "claim_gates": {
            **{
                key: value
                for key, value in prereg["gates"].items()
                if key.startswith("internal_")
                or key.startswith("prebranch_")
                or key.startswith("local_")
                or key.startswith("raw_")
                or key.startswith("attention_")
                or key.startswith("minimum_internal_")
            },
            "cross_model_local_profile_cosine_min": 0.60,
        },
        "interpretation_limits": [
            "Discovery-selected heads are evaluated only on confirmation.",
            "Passing a routing gate is observational evidence, not causal proof.",
            "The local logit lens is an external observer coordinate.",
        ],
    }
    internal_prereg["internal_preregistration_digest"] = (
        protocol.digest(internal_prereg)
    )
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "internal_preregistration.json",
        internal_prereg,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
