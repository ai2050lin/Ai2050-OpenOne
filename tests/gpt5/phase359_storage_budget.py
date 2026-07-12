#!/usr/bin/env python3
"""Freeze a storage budget before expanding Phase358 full-vector traces."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase359_full_vector_anchor"
MODEL_CONFIGS = {
    "qwen3": ROOT / "models/hf/qwen3-4b/config.json",
    "glm4": ROOT / "models/hf/glm4-9b-chat-hf/config.json",
    "deepseek7b": ROOT / "models/hf/deepseek-r1-distill-qwen-7b/config.json",
}
SCHEMA_VERSION = "35.0.0"
MODEL_CASES = 18 * 8
TRACE_PASSES_PER_CASE = 8
PLANNING_SEQUENCE_LENGTH = 256
ROLE_POSITION_COUNT = 4
MLP_SHARD_COUNT = 16


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tensor_bytes(*dimensions: int, dtype_bytes: int = 2) -> int:
    result = dtype_bytes
    for dimension in dimensions:
        result *= dimension
    return result


def model_budget(model: str, config: dict[str, Any]) -> dict[str, Any]:
    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    sequence = PLANNING_SEQUENCE_LENGTH

    raw_states = tensor_bytes(9, layers, sequence, hidden)
    raw_mlp_channels = tensor_bytes(layers, sequence, intermediate)
    attention_probabilities = tensor_bytes(layers, heads, sequence, sequence)
    projected_heads = tensor_bytes(layers, heads, sequence, hidden, dtype_bytes=4)
    projected_shards = tensor_bytes(layers, MLP_SHARD_COUNT, sequence, hidden, dtype_bytes=4)
    naive_per_pass = sum((raw_states, raw_mlp_channels, attention_probabilities, projected_heads, projected_shards))
    naive_all_cases = naive_per_pass * TRACE_PASSES_PER_CASE * MODEL_CASES

    r0_role_states = tensor_bytes(9, layers, ROLE_POSITION_COUNT, hidden) * TRACE_PASSES_PER_CASE * MODEL_CASES
    r0_scalar_ledgers = (
        layers * ROLE_POSITION_COUNT * (heads + MLP_SHARD_COUNT) * 4
        * TRACE_PASSES_PER_CASE * MODEL_CASES
    )
    r1_one_mlp_shard = tensor_bytes(
        layers, ROLE_POSITION_COUNT, max(1, intermediate // MLP_SHARD_COUNT)
    ) * TRACE_PASSES_PER_CASE * MODEL_CASES
    r2_one_anchor = naive_per_pass
    recommended = r0_role_states + r0_scalar_ledgers + r1_one_mlp_shard + r2_one_anchor
    return {
        "model": model,
        "architecture": {
            "layer_count": layers,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "attention_head_count": heads,
        },
        "assumptions": {
            "model_case_count": MODEL_CASES,
            "trace_passes_per_case": TRACE_PASSES_PER_CASE,
            "planning_sequence_length": sequence,
            "native_dtype_bytes": 2,
            "projected_contribution_dtype_bytes": 4,
        },
        "naive_full_trace": {
            "bytes_per_pass": naive_per_pass,
            "bytes_all_cases": naive_all_cases,
        },
        "recommended_multiresolution": {
            "r0_role_state_and_scalar_bytes": r0_role_states + r0_scalar_ledgers,
            "r1_balanced_single_shard_bytes": r1_one_mlp_shard,
            "r2_one_full_anchor_bytes": r2_one_anchor,
            "total_planned_bytes": recommended,
        },
    }


def main() -> None:
    rows = []
    for model, path in MODEL_CONFIGS.items():
        rows.append(model_budget(model, json.loads(path.read_text(encoding="utf-8"))))
    disk = shutil.disk_usage(ROOT)
    naive_total = sum(row["naive_full_trace"]["bytes_all_cases"] for row in rows)
    recommended_total = sum(row["recommended_multiresolution"]["total_planned_bytes"] for row in rows)
    reserve = 200 * 1024**3
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase359",
        "created_at": now(),
        "purpose": "storage_budget_freeze_before_blind_full_vector_discovery",
        "models": rows,
        "global": {
            "naive_full_trace_bytes": naive_total,
            "recommended_multiresolution_bytes": recommended_total,
            "disk_free_bytes_at_freeze": disk.free,
            "minimum_free_space_reserve_bytes": reserve,
            "naive_full_trace_fits_with_reserve": naive_total + reserve <= disk.free,
            "recommended_multiresolution_fits_with_reserve": recommended_total + reserve <= disk.free,
        },
        "frozen_policy": {
            "r0": "all cases; four pre-registered position roles; component states plus scalar head/shard ledgers",
            "r1": "balanced hash assignment; one raw MLP shard per case; no label-conditioned selection",
            "r2": "one sealed full-vector anchor per model until replay audit passes",
            "r3": "held-out and causal traces remain unopened",
            "frontend_raw_tensor_export": False,
            "semantic_label_used_for_selection": False,
        },
        "decision": (
            "allow_one_full_vector_anchor_per_model"
            if recommended_total + reserve <= disk.free
            else "block_anchor_capture_insufficient_storage"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase359_storage_budget.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["global"] | {"decision": summary["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
