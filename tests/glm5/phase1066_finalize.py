#!/usr/bin/env python3
"""Finalize Phase1066 and preserve causal claim boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1066_reasoning_role_causal_protocol as protocol


def result(
    summary: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    return summary["condition_results"][condition]


def finalize() -> dict[str, Any]:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1066 protocol audit failed")
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    for model, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"protocol drift for {model}")
        if (
            summary["precision"]["has_quantized_modules"]
            or summary["precision"]["has_bf16_parameters"]
            or not summary["precision"]["has_fp16_parameters"]
        ):
            raise RuntimeError(f"precision drift for {model}")
        if (
            summary["clean_candidate_replay_rate"]
            < prereg["gates"]["clean_candidate_replay_min"]
        ):
            raise RuntimeError(f"clean replay drift for {model}")

    model_topologies = {}
    for model, summary in summaries.items():
        premise = {
            "premise1": result(
                summary, "premise1_all_kv"
            )["bidirectional_donor_class_flip_rate"],
            "premise2": result(
                summary, "premise2_all_kv"
            )["bidirectional_donor_class_flip_rate"],
            "both": result(
                summary, "both_all_kv"
            )["bidirectional_donor_class_flip_rate"],
        }
        depth = {
            name: result(
                summary, f"both_{name}_kv"
            )["bidirectional_donor_class_flip_rate"]
            for name in (
                "q1", "q2", "q3", "q4", "q2_q4", "q3_q4"
            )
        }
        channel = {
            "k_only": result(
                summary, "both_all_k_only"
            )["bidirectional_donor_class_flip_rate"],
            "v_only": result(
                summary, "both_all_v_only"
            )["bidirectional_donor_class_flip_rate"],
            "kv": result(
                summary, "both_all_kv"
            )["bidirectional_donor_class_flip_rate"],
        }
        controls = {
            "operator": result(
                summary, "operator_all_kv"
            )["bidirectional_donor_class_flip_rate"],
            "query": result(
                summary, "query_all_kv"
            )["bidirectional_donor_class_flip_rate"],
        }
        groups = {
            name: row["bidirectional_donor_class_flip_rate"]
            for name, row in summary["condition_results"].items()
            if row["kind"] == "group"
        }
        model_topologies[model] = {
            "premise_role_rates": premise,
            "depth_rates": depth,
            "channel_rates": channel,
            "role_control_rates": controls,
            "group_rates": groups,
            "joint_premise_increment_over_best_single": (
                premise["both"]
                - max(premise["premise1"], premise["premise2"])
            ),
            "kv_increment_over_best_single_channel": (
                channel["kv"]
                - max(channel["k_only"], channel["v_only"])
            ),
            "best_group_rate": max(groups.values(), default=0.0),
        }

    passing_models = [
        model for model, summary in summaries.items()
        if summary["causal_gate_passed"]
    ]
    repeated = (
        len(passing_models)
        >= prereg["gates"]["minimum_repeated_models"]
    )
    automatic = {
        "schema_version": "phase1066_automatic_next.v1",
        "phase": protocol.PHASE,
        "should_continue_automatically": repeated,
        "passing_models": passing_models,
        "route": (
            "build_phase1067_reasoning_natural_necessity_and_component_localization"
            if repeated
            else "stop_at_response_atlas_without_causal_transport"
        ),
        "rationale": (
            "Continuation requires source-role donor transport in at least "
            "two models with low operator/query controls. A positive result "
            "still does not establish natural necessity."
        ),
    }
    aggregate = {
        "schema_version": "phase1066_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "model_summaries": summaries,
        "model_functional_topologies": model_topologies,
        "passing_models": passing_models,
        "cross_model_causal_repetition": repeated,
        "automatic_next_decision": automatic,
        "claim_boundary": {
            "supported_if_repeated": (
                "Role-endpoint K/V state exchange can conditionally move "
                "the binary transitive-answer competition."
            ),
            "not_supported": [
                "The endpoint is the complete reasoning representation.",
                "The intervention proves natural necessity.",
                "The model performs symbolic transitivity in a single K/V channel.",
                "The same topology applies to unrestricted reasoning.",
                "A measurement increment is a language law.",
            ],
        },
    }
    protocol.write_json(
        protocol.OUT_ROOT / "aggregate.json", aggregate
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json",
        automatic,
    )
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "model_functional_topologies.json",
        model_topologies,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "model_topologies": model_topologies,
        "automatic_next": automatic,
    }, ensure_ascii=False, indent=2))
    return aggregate


def main() -> None:
    finalize()


if __name__ == "__main__":
    main()
