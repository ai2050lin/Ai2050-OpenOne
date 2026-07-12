#!/usr/bin/env python3
"""Audit exact role-tree feasibility and all-token gaps on one existing case per model."""

from __future__ import annotations

import gc
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase365_dynamic_bundle_extraction import load_weight  # noqa: E402
from phase358_multiresolution_component_conservation import relative_error  # noqa: E402


PHASE369 = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
RAW = PHASE369 / "raw_collection"
OUT = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity/engineering_feasibility"
MODELS = ("qwen3", "glm4", "deepseek7b")
NUMERIC_GATE = 0.01
PARTITION_COUNT = 8


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def anchors(layer_count: int) -> tuple[int, int, int]:
    return 0, layer_count // 2, layer_count - 1


def contiguous_partitions(size: int, count: int) -> list[tuple[int, int]]:
    return [
        (math.floor(index * size / count), math.floor((index + 1) * size / count))
        for index in range(count)
    ]


def attention_tree_replay(
    payload: dict[str, Any],
    weight: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, int]:
    attention = payload["attention"]
    values = attention["value_states_all_sources"].to(device=device, dtype=torch.float32)
    probabilities = attention["probabilities_role_receivers_all_sources"].to(device=device, dtype=torch.float32)
    expected = payload["component_vectors"]["attention_output"].to(device=device, dtype=torch.float32)
    head_count = int(attention["head_count"])
    kv_count = int(attention["key_value_head_count"])
    head_dim = int(attention["head_dim"])
    if kv_count != head_count:
        values = values.repeat_interleave(head_count // kv_count, dim=1)
    weighted = torch.einsum("bhqs,bhsd->bhqd", probabilities, values)
    blocks = weight.to(device=device, dtype=torch.float32).view(weight.shape[0], head_count, head_dim)
    head_writes = torch.einsum("bhqd,ohd->bqho", weighted, blocks)
    direct = head_writes.sum(dim=2)
    _, direct_error = relative_error(expected, direct)
    partition_writes = [
        head_writes[:, :, start:end].sum(dim=2)
        for start, end in contiguous_partitions(head_count, min(PARTITION_COUNT, head_count))
    ]
    tree_parent = torch.stack(partition_writes).sum(dim=0)
    _, tree_error = relative_error(direct, tree_parent)
    return direct_error, tree_error, len(partition_writes)


def mlp_tree_replay(
    payload: dict[str, Any],
    weight: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, int]:
    product = payload["mlp"]["down_projection_input_product_at_roles"].to(device=device, dtype=torch.float32)
    expected = payload["component_vectors"]["mlp_output"].to(device=device, dtype=torch.float32)
    weight = weight.to(device=device, dtype=torch.float32)
    direct = F.linear(product, weight)
    _, direct_error = relative_error(expected, direct)
    channel_count = int(product.shape[-1])
    partition_writes = [
        F.linear(product[..., start:end], weight[:, start:end])
        for start, end in contiguous_partitions(channel_count, PARTITION_COUNT)
    ]
    tree_parent = torch.stack(partition_writes).sum(dim=0)
    _, tree_error = relative_error(direct, tree_parent)
    return direct_error, tree_error, len(partition_writes)


def storage_estimate(manifests: list[dict[str, Any]]) -> dict[str, int]:
    current_bytes = sum(int(item["total_byte_count"]) for item in manifests)
    additional = 0
    for manifest in manifests:
        model = manifest["model"]
        first_case = manifest["case_rows"][0]["blind_case_id"]
        sample = torch.load(
            RAW / "private/models" / model / first_case / "time_0/layer_000.pt",
            map_location="cpu", weights_only=True,
        )
        hidden = int(sample["component_vectors"]["layer_input"].shape[-1])
        channels = int(sample["mlp"]["channel_count"])
        heads = int(sample["attention"]["head_count"])
        kv_heads = int(sample["attention"]["key_value_head_count"])
        head_dim = int(sample["attention"]["head_dim"])
        layer_count = int(manifest["layer_count"])
        for file_row in manifest["files"]:
            if file_row["kind"] != "time_meta":
                continue
            meta = torch.load(RAW / file_row["relative_path"], map_location="cpu", weights_only=True)
            sequence = int(meta["sequence_length"])
            extra_positions = max(sequence - 4, 0)
            per_layer_elements = (
                extra_positions * hidden * 7
                + extra_positions * channels * 3
                + heads * extra_positions * sequence
                + (heads + kv_heads) * sequence * head_dim
            )
            additional += layer_count * per_layer_elements * 2
    return {
        "current_raw_bytes": current_bytes,
        "estimated_additional_all_token_qk_bytes": additional,
        "estimated_total_raw_bytes": current_bytes + additional,
        "free_disk_bytes_at_audit": shutil.disk_usage(ROOT).free,
    }


def run_model(model: str, device: torch.device) -> dict[str, Any]:
    manifest = read_json(RAW / "models" / model / "manifest.json")
    layer_count = int(manifest["layer_count"])
    case_id = sorted(row["blind_case_id"] for row in manifest["case_rows"])[0]
    rows = []
    for layer in anchors(layer_count):
        o_weight = load_weight(model, f"model.layers.{layer}.self_attn.o_proj.weight")
        down_weight = load_weight(model, f"model.layers.{layer}.mlp.down_proj.weight")
        for generation_time in range(3):
            payload = torch.load(
                RAW / "private/models" / model / case_id / f"time_{generation_time}" / f"layer_{layer:03d}.pt",
                map_location="cpu", weights_only=True,
            )
            attention_direct, attention_tree, attention_children = attention_tree_replay(payload, o_weight, device)
            mlp_direct, mlp_tree, mlp_children = mlp_tree_replay(payload, down_weight, device)
            sequence_length = int(payload["attention"]["value_states_all_sources"].shape[2])
            role_count = int(payload["component_vectors"]["layer_input"].shape[1])
            rows.append({
                "model": model,
                "anonymous_case_id": case_id,
                "generation_time": generation_time,
                "layer_index": layer,
                "sequence_length": sequence_length,
                "stored_receiver_position_count": role_count,
                "attention_direct_error": attention_direct,
                "attention_tree_error": attention_tree,
                "attention_tree_child_count": attention_children,
                "mlp_direct_error": mlp_direct,
                "mlp_tree_error": mlp_tree,
                "mlp_tree_child_count": mlp_children,
                "role_tree_numeric_gate_pass": max(attention_direct, attention_tree, mlp_direct, mlp_tree) <= NUMERIC_GATE,
                "all_token_receiver_states_available": role_count == sequence_length,
                "query_key_states_available": "query_states_all_positions" in payload["attention"] and "key_states_all_positions" in payload["attention"],
            })
            del payload
        del o_weight, down_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "model": model,
        "anonymous_case_id": case_id,
        "anchor_layer_count": 3,
        "generation_time_count": 3,
        "audit_row_count": len(rows),
        "max_attention_direct_error": max(row["attention_direct_error"] for row in rows),
        "max_attention_tree_error": max(row["attention_tree_error"] for row in rows),
        "max_mlp_direct_error": max(row["mlp_direct_error"] for row in rows),
        "max_mlp_tree_error": max(row["mlp_tree_error"] for row in rows),
        "role_tree_numeric_gate_pass": all(row["role_tree_numeric_gate_pass"] for row in rows),
        "all_token_receiver_states_available": all(row["all_token_receiver_states_available"] for row in rows),
        "query_key_states_available": all(row["query_key_states_available"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests = [read_json(RAW / "models" / model / "manifest.json") for model in MODELS]
    model_rows = []
    for model in MODELS:
        model_rows.append(run_model(model, device))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{model}] Phase371A role-tree feasibility complete", flush=True)
    storage = storage_estimate(manifests)
    role_gate = all(row["role_tree_numeric_gate_pass"] for row in model_rows)
    all_token = all(row["all_token_receiver_states_available"] for row in model_rows)
    qk = all(row["query_key_states_available"] for row in model_rows)
    summary = {
        "schema_version": "47.1.0",
        "phase_id": "Phase371A",
        "created_at": now(),
        "objective": "test_exact_conservation_tree_feasibility_on_existing_role_limited_ledger_and_audit_missing_all_token_state",
        "execution": {
            "model_execution_or_generation_used": False,
            "existing_private_ledger_only": True,
            "compute_device": str(device),
            "model_order": list(MODELS),
        },
        "denominator": {
            "model_count": 3,
            "existing_case_count": 3,
            "anchor_layer_count_per_model": 3,
            "generation_time_count": 3,
            "audit_row_count": sum(row["audit_row_count"] for row in model_rows),
            "tree_partition_count": PARTITION_COUNT,
        },
        "results": {
            "role_limited_exact_tree_numeric_gate_pass": role_gate,
            "all_token_receiver_states_available": all_token,
            "query_key_states_available": qk,
            "full_phase371a_path_object_gate_pass": role_gate and all_token and qk,
            "language_mechanism_claimed": False,
        },
        "storage": storage,
        "models": [{key: value for key, value in row.items() if key != "rows"} for row in model_rows],
        "claim_boundary": {
            "four_role_exact_tree_feasible": role_gate,
            "all_token_exact_tree_tested": False,
            "query_key_path_tested": False,
            "new_language_path_discovered": False,
            "physical_holdout_opened": False,
        },
        "authorization": {
            "extend_collector_schema_for_all_token_qk_anchor_replay": role_gate and not (all_token and qk),
            "full_case_collection": False,
            "new_discovery_or_calibration_cycle": False,
            "physical_holdout": False,
        },
        "next_decision": "implement_three_anchor_all_token_qk_collector_before_any_new_case_bank",
    }
    write_json(OUT / "phase371a_existing_ledger_tree_feasibility_summary.json", summary)
    write_json(
        OUT / "private/phase371a_existing_ledger_tree_rows.json",
        {"schema_version": "47.1.0", "phase_id": "Phase371A", "models": model_rows},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
