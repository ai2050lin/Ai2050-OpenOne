#!/usr/bin/env python3
"""Freeze the coordinate-free Phase573 natural transition trace contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase573"
MODEL = "qwen3"
OUT_DIR = ROOT / "tests/gpt5/result/phase573_natural_transition"
BEHAVIOR_SUMMARY = OUT_DIR / f"phase573_{MODEL}_behavior_summary.json"
BEHAVIOR_REGISTRY = OUT_DIR / f"phase573_{MODEL}_behavior_registry.json"
FROZEN_PROTOCOL = OUT_DIR / "phase573_frozen_protocol.json"
OPEN_CASES = OUT_DIR / "phase573_open_cases.jsonl.gz"
TRACE_PROTOCOL = OUT_DIR / "phase573_natural_trace_protocol.json"

SPLITS = (
    "structure_discovery",
    "structure_confirmation",
    "heldout_recombination",
)
AXES = ("relation", "object", "order")
VARIANT_BY_AXIS = {
    "relation": "relation_swap",
    "object": "object_swap",
    "order": "order_swap",
}
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
SEMANTIC_ROLES = (
    "target_fact_object",
    "target_fact_relation",
    "target_fact_value",
    "other_fact_object",
    "other_fact_relation",
    "other_fact_value",
    "query_relation",
    "query_object",
    "query_terminal",
    "answer_boundary",
)
PHYSICAL_PREFIX_ROLES = (
    "anchor_target_fact_object",
    "anchor_target_fact_relation",
    "anchor_target_fact_value",
    "anchor_other_fact_object",
    "anchor_other_fact_relation",
    "anchor_other_fact_value",
)
ROUTING_RECEIVERS = ("query_terminal", "answer_boundary")
WORLDS_PER_SPLIT = 128
TRACE_BATCH_WORLDS = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze() -> dict[str, Any]:
    summary = read_json(BEHAVIOR_SUMMARY)
    registry = read_json(BEHAVIOR_REGISTRY)
    frozen = read_json(FROZEN_PROTOCOL)
    if not summary["authorized_for_natural_trace"]:
        raise RuntimeError("Phase573 Qwen3 behavior gate does not authorize a trace")
    if not registry["authorized_for_natural_trace"] or registry["sealed_split_read"]:
        raise RuntimeError("Phase573 Qwen3 trace registry is not authorized")
    selected = registry["selected_base_case_ids_by_split"]
    if set(selected) != set(SPLITS):
        raise RuntimeError("Phase573 trace split drift")
    if any(len(selected[split]) != WORLDS_PER_SPLIT for split in SPLITS):
        raise RuntimeError("Phase573 trace requires exactly 128 worlds per split")
    if len(set().union(*(set(selected[split]) for split in SPLITS))) != 384:
        raise RuntimeError("Phase573 trace worlds overlap across splits")

    payload = {
        "schema_version": "phase573_natural_trace_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "model": MODEL,
        "splits": list(SPLITS),
        "axes": list(AXES),
        "variant_by_axis": VARIANT_BY_AXIS,
        "worlds_per_split": WORLDS_PER_SPLIT,
        "selected_base_case_ids_by_split": selected,
        "components": list(COMPONENTS),
        "semantic_roles": list(SEMANTIC_ROLES),
        "physical_prefix_roles": list(PHYSICAL_PREFIX_ROLES),
        "routing_receivers": list(ROUTING_RECEIVERS),
        "trace_batch_worlds": TRACE_BATCH_WORLDS,
        "position_id_policy": (
            "right-pad observer-only forward batches and derive position_ids from the "
            "cumulative attention mask, keeping identical prefixes in identical tensor "
            "columns and rotary coordinates"
        ),
        "primary_state_metric": (
            "paired Euclidean change divided by the mean norm of the two full vectors"
        ),
        "output_embedding_direction_used_for_upstream_trace": False,
        "attention_routing_interpretation": (
            "observer-only source routing weight, not a source-specific value message "
            "and not causal evidence"
        ),
        "causal_mask_audit": {
            "fixed_context_axes": ["relation", "object"],
            "audited_roles": list(PHYSICAL_PREFIX_ROLES),
            "maximum_allowed_relative_prefix_delta": 1e-5,
            "reason": (
                "A future query cannot alter earlier fact-token states in a causal "
                "transformer; fact reading is evaluated only at later receivers."
            ),
            "first_invalid_run_observer_artifact": (
                "The initial batched trace omitted explicit position_ids; variable left "
                "padding moved identical prefixes. A second rejected audit showed that "
                "left-padded BF16 reductions still amplified layout differences for "
                "variable-length object names. Both runs are rejected and replaced by "
                "right-padded observer-only forwards."
            ),
        },
        "state_event_gate": {
            "minimum_relative_delta": 0.05,
            "minimum_world_rate_each_split": 0.75,
            "eligible_receivers": ["query_terminal", "answer_boundary"],
            "discovery_coordinate_must_replicate_in_confirmation_and_heldout": True,
        },
        "routing_event_gate": {
            "minimum_semantic_selection_pair_rate": 0.60,
            "minimum_mean_semantic_selection_margin": 0.0,
            "minimum_order_preservation_pair_rate": 0.60,
            "minimum_object_selection_pair_rate": 0.55,
            "discovery_coordinate_must_replicate_in_confirmation_and_heldout": True,
            "eligible_receivers": list(ROUTING_RECEIVERS),
        },
        "coarse_message_causal_authorization": (
            "Only a routing coordinate passing discovery, confirmation, heldout, "
            "object and order gates may authorize a later coarse message-edge test."
        ),
        "full_vectors_persisted": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "causal_intervention_executed": False,
        "causal_splits_read": False,
        "sealed_split_read": False,
        "behavior_summary_sha256": sha256_file(BEHAVIOR_SUMMARY),
        "behavior_registry_sha256": sha256_file(BEHAVIOR_REGISTRY),
        "frozen_protocol_sha256": sha256_file(FROZEN_PROTOCOL),
        "open_cases_sha256": sha256_file(OPEN_CASES),
        "sealed_commitment_sha256": frozen["sealed_commitment_sha256"],
    }
    write_json(TRACE_PROTOCOL, payload)
    print(json.dumps({
        "model": MODEL,
        "worlds": sum(len(ids) for ids in selected.values()),
        "sealed_split_read": False,
        "trace_protocol": str(TRACE_PROTOCOL.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    freeze()
