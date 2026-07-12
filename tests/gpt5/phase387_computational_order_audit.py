#!/usr/bin/env python3
"""Audit Phase386 predictive relations against the decoder computation order."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/gpt5/result/phase386_multitime_relation_atlas"
    / "phase386_physical_candidate_rows.jsonl"
)
OUT = ROOT / "tests/gpt5/result/phase387_computational_order_audit"


COMPONENT_STAGE = {
    "layer_input": "block_input_before_attention",
    "attention_head_state": "attention_value_mix_before_output_projection",
    "attention_output": "attention_update_after_output_projection",
    "mlp_channel_product": "mlp_product_before_down_projection",
    "mlp_output": "mlp_update_after_down_projection",
    "layer_output": "block_output_after_attention_and_mlp",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def checksum_ids(rows: list[dict[str, Any]]) -> str:
    material = "\n".join(sorted(row["candidate_id"] for row in rows)).encode()
    return hashlib.sha256(material).hexdigest()


def audit_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    family = candidate["vector_family"]
    source = candidate["source_coordinate"]
    target = candidate["target_coordinate"]
    terminal = source == "target_encoded" and target == "post_decision_next_token"
    pre_answer = source == "source_encoded" and target == "query_integrated"

    if family == "layer_input":
        admissible_rewrite = (
            "source layer_input/K/V at layer L -> receiver attention state or "
            "attention output at layer L"
        )
    elif family in {"attention_head_state", "attention_output"}:
        admissible_rewrite = (
            "source layer_input/K/V at layer L -> receiver attention state at layer L, "
            "or source attention update at layer L -> receiver state at layer L+1 or later"
        )
    else:
        admissible_rewrite = (
            "source block output at layer L -> receiver state at layer L+1 or later"
        )

    return {
        "schema_version": "61.0.0",
        "phase_id": "Phase387-ComputationalOrderAudit",
        "candidate_id": candidate["candidate_id"],
        "mechanism_id": candidate["mechanism_id"],
        "vector_family": family,
        "component_stage": COMPONENT_STAGE[family],
        "source_coordinate": source,
        "target_coordinate": target,
        "depth_bin": candidate["depth_bin"],
        "model_layers": candidate["model_layers"],
        "physical_holdout_used": candidate["physical_holdout_used"],
        "physical_predictive_relation_pass": True,
        "semantic_temporal_precedence": True,
        "terminal_answer_continuation": terminal,
        "pre_answer_relation": pre_answer,
        "same_exact_layer_and_same_family_relation": True,
        "direct_computational_edge_admissible": False,
        "indirect_causal_reachability_proven": False,
        "registered_edge_kind": "physical_holdout_predictive_trajectory",
        "downgrade_reason": (
            "The stored source component is not the direct tensor consumed by the "
            "same-family receiver component at the same layer. Decoder attention at "
            "receiver position p consumes source layer-input-derived K/V at layer L. "
            "Source attention/MLP/block outputs from layer L can first alter source "
            "K/V at layer L+1 or later. Temporal precedence and prediction therefore "
            "do not establish the registered same-layer arrow."
        ),
        "causally_admissible_rewrite": admissible_rewrite,
        "causal_claim": False,
        "language_path_claim": False,
    }


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    candidates = [
        row
        for row in read_jsonl(SOURCE)
        if row.get("physical_predictive_relation_path_gate_pass") is True
    ]
    if len(candidates) != 10:
        raise RuntimeError(f"Expected 10 Phase386 physical survivors, found {len(candidates)}")
    if not all(row.get("physical_holdout_used") is True for row in candidates):
        raise RuntimeError("A physical survivor is not marked as physical-holdout evidence")

    rows = [audit_candidate(candidate) for candidate in candidates]
    write_jsonl(OUT / "phase387_candidate_order_rows.jsonl", rows)

    contract = {
        "schema_version": "61.0.0",
        "phase_id": "Phase387-ComputationalOrderContract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decoder_block_partial_order": [
            "x_L,p -> norm(x_L,p)",
            "norm(x_L,p) -> q_L,p",
            "norm(x_L,s) -> k_L,s and v_L,s",
            "q_L,p plus k_L,s plus v_L,s -> z_L,p for s <= p",
            "z_L,p -> attention_output_L,p",
            "x_L,p plus attention_output_L,p -> post_attention_L,p",
            "post_attention_L,p -> mlp_output_L,p",
            "post_attention_L,p plus mlp_output_L,p -> x_(L+1),p",
        ],
        "cross_position_rule": (
            "At the same layer, a later receiver attention state reads earlier-position "
            "K/V derived from that earlier position's layer input. It does not read the "
            "earlier position's attention-head output from the same layer."
        ),
        "incremental_cache_rule": (
            "The generation cache stores per-layer K/V derived before that layer's "
            "attention output. A target-token attention output at layer L can affect a "
            "later token through layer L+1 or deeper, not as the cached K/V at layer L."
        ),
        "registration_rule": (
            "A predictive trajectory may be displayed as a descriptive relation. A "
            "direct physical edge requires the source tensor to be an actual input of "
            "the registered receiver operation under the frozen layer and position map."
        ),
        "composite_score_used": False,
        "model_run_required": False,
    }
    write_json(OUT / "phase387_computational_order_contract.json", contract)

    summary = {
        "schema_version": "61.0.0",
        "phase_id": "Phase387-ComputationalOrderAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "phase386_physical_survivor_count": len(rows),
            "candidate_id_checksum": checksum_ids(rows),
            "terminal_trajectory_count": sum(
                row["terminal_answer_continuation"] for row in rows
            ),
            "pre_answer_trajectory_count": sum(row["pre_answer_relation"] for row in rows),
        },
        "results": {
            "semantic_temporal_precedence_count": sum(
                row["semantic_temporal_precedence"] for row in rows
            ),
            "direct_computational_edge_admissible_count": sum(
                row["direct_computational_edge_admissible"] for row in rows
            ),
            "indirect_causal_reachability_proven_count": sum(
                row["indirect_causal_reachability_proven"] for row in rows
            ),
            "predictive_trajectory_count": sum(
                row["registered_edge_kind"] == "physical_holdout_predictive_trajectory"
                for row in rows
            ),
            "counts_by_vector_family": dict(
                sorted(Counter(row["vector_family"] for row in rows).items())
            ),
            "upstream_direct_physical_edge_count": sum(
                row["pre_answer_relation"] and row["direct_computational_edge_admissible"]
                for row in rows
            ),
            "causal_language_path_count": 0,
            "single_neuron_causal_path_count": 0,
        },
        "decision": (
            "Retain all ten Phase386 survivors as physical-holdout predictive "
            "trajectories, but register none as a direct computational edge."
        ),
        "authorization": {
            "display_predictive_trajectories": True,
            "display_direct_physical_edge": False,
            "reuse_phase386_physical_holdout": False,
            "run_single_neuron_scan": False,
            "freeze_fresh_causally_admissible_intervention_protocol": True,
        },
        "next_stage": {
            "phase": 388,
            "objective": (
                "test source layer-input-derived K/V to receiver attention-state "
                "transport on a fresh relation-binding denominator"
            ),
            "required_controls": [
                "wrong_source_position",
                "wrong_layer",
                "K_only",
                "V_only",
                "matched_terminal_relation",
                "no_intervention",
            ],
            "required_outcomes": [
                "receiver_state_mediation",
                "downstream_target_logit_change",
                "behavior_change_when_the_natural_outputs_differ",
            ],
        },
    }
    write_json(OUT / "phase387_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
