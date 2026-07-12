#!/usr/bin/env python3
"""Derive exploratory fixed-hash attention-head and MLP-neuron topology features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase365_dynamic_bundle_extraction import load_weight  # noqa: E402


BASE = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
RAW = BASE / "raw_collection"
BUNDLES = BASE / "dynamic_bundle_extraction/blind_bundles"
OUT = BASE / "head_neuron_topology_diagnostic"
MODELS = ("qwen3", "glm4", "deepseek7b")
SHARD_COUNTS = (8, 32, 128)
HASH_SEEDS = (17, 29, 43)
_ASSIGNMENT_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def assignment(unit_count: int, shard_count: int, seed: int) -> torch.Tensor:
    cache_key = (unit_count, shard_count, seed)
    if cache_key in _ASSIGNMENT_CACHE:
        return _ASSIGNMENT_CACHE[cache_key]
    values = []
    for unit_index in range(unit_count):
        digest = hashlib.sha256(f"phase369:{seed}:{unit_index}".encode()).digest()
        values.append(int.from_bytes(digest[:8], "big") % shard_count)
    result = torch.tensor(values, dtype=torch.long)
    _ASSIGNMENT_CACHE[cache_key] = result
    return result


def shard_signatures(contributions: torch.Tensor) -> torch.Tensor:
    if contributions.ndim != 2:
        raise ValueError(f"Expected [row, unit] contributions, got {tuple(contributions.shape)}")
    unit_count = int(contributions.shape[1])
    rows = []
    for shard_count in SHARD_COUNTS:
        for seed in HASH_SEEDS:
            indices = assignment(unit_count, shard_count, seed)
            shards = torch.zeros((contributions.shape[0], shard_count), dtype=torch.float32)
            shards.scatter_add_(1, indices.unsqueeze(0).expand(contributions.shape[0], -1), contributions.float().abs())
            shares = shards / shards.sum(dim=1, keepdim=True).clamp_min(1e-12)
            rows.append(shares.sort(dim=1, descending=True).values)
    return torch.cat(rows, dim=1)


def shard_slices() -> dict[str, list[int]]:
    result = {}
    offset = 0
    for shard_count in SHARD_COUNTS:
        length = shard_count * len(HASH_SEEDS)
        result[str(shard_count)] = [offset, offset + length]
        offset += length
    return result


def static_unit_norms(model: str, manifest: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    layer_count = int(manifest["layer_count"])
    first_case = manifest["case_rows"][0]["blind_case_id"]
    head_norms = []
    neuron_norms = []
    for layer in range(layer_count):
        sample = torch.load(
            RAW / "private/models" / model / first_case / "time_0" / f"layer_{layer:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        head_count = int(sample["attention"]["head_count"])
        head_dim = int(sample["attention"]["head_dim"])
        o_weight = load_weight(model, f"model.layers.{layer}.self_attn.o_proj.weight").float()
        head_norms.append(
            torch.linalg.vector_norm(o_weight.view(o_weight.shape[0], head_count, head_dim), dim=(0, 2))
        )
        down_weight = load_weight(model, f"model.layers.{layer}.mlp.down_proj.weight").float()
        neuron_norms.append(torch.linalg.vector_norm(down_weight, dim=0))
        del sample, o_weight, down_weight
    return head_norms, neuron_norms


def route_head_contribution(
    event: dict[str, Any],
    raw: dict[str, Any],
    head_projection_norm: torch.Tensor,
) -> torch.Tensor:
    roles = list(raw["role_names"])
    positions = [int(value) for value in raw["role_positions"]]
    receiver_role = event["receiver_role"].split("+")[0]
    receiver_index = roles.index(receiver_role)
    probabilities = raw["attention"]["probabilities_role_receivers_all_sources"][0].float()
    values = raw["attention"]["value_states_all_sources"][0].float()
    head_count = int(raw["attention"]["head_count"])
    kv_count = int(raw["attention"]["key_value_head_count"])
    if kv_count != head_count:
        values = values.repeat_interleave(head_count // kv_count, dim=0)
    value_norms = torch.linalg.vector_norm(values, dim=-1)
    if event["source_role"] == "other_sources":
        named_positions = set(positions)
        source_positions = [index for index in range(values.shape[1]) if index not in named_positions]
        if not source_positions:
            return torch.zeros(head_count)
        indices = torch.tensor(source_positions, dtype=torch.long)
        routed = (
            probabilities[:, receiver_index].index_select(1, indices)
            * value_norms.index_select(1, indices)
        ).sum(dim=1)
    else:
        source_position = int(event["source_position"])
        routed = probabilities[:, receiver_index, source_position] * value_norms[:, source_position]
    return routed * head_projection_norm


def extract_model(model: str) -> dict[str, Any]:
    manifest = read_json(RAW / "models" / model / "manifest.json")
    head_norms, neuron_norms = static_unit_norms(model, manifest)
    case_rows = []
    for case_index, bundle_path in enumerate(sorted((BUNDLES / model).glob("*.json")), 1):
        bundle = read_json(bundle_path)
        case_id = bundle["anonymous_case_id"]
        route_events_by_layer_time: dict[tuple[int, int], list[dict[str, Any]]] = {}
        mlp_events_by_layer_time: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for event in bundle["events"]:
            key = (int(event["generation_time"]), int(event["layer_index"]))
            if event["event_type"] == "attention_source_write":
                route_events_by_layer_time.setdefault(key, []).append(event)
            elif event["event_type"] == "mlp_merge":
                mlp_events_by_layer_time.setdefault(key, []).append(event)
        head_records = []
        head_features = []
        neuron_records = []
        neuron_features = []
        for generation_time in range(3):
            for layer in range(int(manifest["layer_count"])):
                raw = torch.load(
                    RAW / "private/models" / model / case_id / f"time_{generation_time}" / f"layer_{layer:03d}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                layer_route_events = sorted(
                    route_events_by_layer_time[(generation_time, layer)],
                    key=lambda item: (item["source_role"], item["receiver_role"]),
                )
                layer_head_contributions = []
                for event in layer_route_events:
                    layer_head_contributions.append(route_head_contribution(event, raw, head_norms[layer]))
                    head_records.append({
                        "generation_time": generation_time,
                        "layer_index": layer,
                        "source_role": event["source_role"],
                        "receiver_role": event["receiver_role"],
                    })
                head_features.extend(
                    shard_signatures(torch.stack(layer_head_contributions)).unbind(dim=0)
                )
                product = raw["mlp"]["down_projection_input_product_at_roles"][0].float()
                roles = list(raw["role_names"])
                layer_mlp_events = sorted(
                    mlp_events_by_layer_time[(generation_time, layer)],
                    key=lambda item: item["receiver_role"],
                )
                layer_neuron_contributions = []
                for event in layer_mlp_events:
                    role_index = roles.index(event["receiver_role"].split("+")[0])
                    layer_neuron_contributions.append(product[role_index].abs() * neuron_norms[layer])
                    neuron_records.append({
                        "generation_time": generation_time,
                        "layer_index": layer,
                        "receiver_role": event["receiver_role"],
                    })
                neuron_features.extend(
                    shard_signatures(torch.stack(layer_neuron_contributions)).unbind(dim=0)
                )
                del raw
        payload = {
            "schema_version": "46.3.0",
            "phase_id": "Phase369-Diagnostic",
            "anonymous_case_id": case_id,
            "anonymous_group_id": bundle["anonymous_group_id"],
            "anonymous_model_id": bundle["anonymous_model_id"],
            "head_records": head_records,
            "head_hash_topology": torch.stack(head_features).to(torch.float16),
            "neuron_records": neuron_records,
            "neuron_hash_topology": torch.stack(neuron_features).to(torch.float16),
            "shard_slices": shard_slices(),
            "hash_seeds": HASH_SEEDS,
            "multiple_seeds_are_sensitivity_checks_not_replications": True,
            "task_selected_top_k_used": False,
            "semantic_labels_used": False,
            "exact_single_neuron_write_magnitude_used": True,
            "within_shard_vector_cross_terms_retained": False,
        }
        output_path = OUT / "private/cases" / model / f"{case_id}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        case_rows.append({
            "anonymous_case_id": case_id,
            "head_route_count": len(head_records),
            "mlp_role_count": len(neuron_records),
            "relative_path": str(output_path.relative_to(OUT)),
            "byte_count": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        })
        if case_index % 8 == 0 or case_index == len(manifest["case_rows"]):
            print(f"[{model}] head/neuron topology {case_index}/{len(manifest['case_rows'])}", flush=True)
    result = {
        "model": model,
        "case_count": len(case_rows),
        "head_route_count": sum(row["head_route_count"] for row in case_rows),
        "mlp_role_count": sum(row["mlp_role_count"] for row in case_rows),
        "case_rows": case_rows,
        "valid": len(case_rows) == 112,
    }
    write_json(OUT / "models" / model / "manifest.json", result)
    return result


def freeze_protocol() -> None:
    protocol = {
        "schema_version": "46.3.0",
        "phase_id": "Phase369-Diagnostic",
        "created_at": now(),
        "reason": "phase369_raw_component_relations_failed_the_full_frozen_discovery_gate",
        "claim_status": "exploratory_diagnostic_only_cannot_rescue_phase369_or_open_calibration",
        "fixed_hash_shard_counts": SHARD_COUNTS,
        "fixed_hash_seeds": HASH_SEEDS,
        "multiple_hash_seeds_count_as_independent_replication": False,
        "task_score_top_k_used": False,
        "head_feature": "sorted_hash_shard_share_of_routing_weighted_value_norm_times_output_projection_block_norm",
        "neuron_feature": "sorted_hash_shard_share_of_exact_single_neuron_write_magnitude_abs_product_times_down_column_norm",
        "known_loss": "within_shard_vector_direction_and_cross_terms_not_retained",
        "success_only_authorizes": "new_independent_data_cycle_with_frozen_topology_object",
        "calibration_or_physical_holdout_authorized": False,
    }
    write_json(OUT / "phase369_head_neuron_diagnostic_protocol.json", protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    freeze_protocol()
    if args.model:
        result = extract_model(args.model)
        print(json.dumps({key: value for key, value in result.items() if key != "case_rows"}, ensure_ascii=False, indent=2))
        return
    if args.summarize:
        rows = [read_json(OUT / "models" / model / "manifest.json") for model in MODELS]
        summary = {
            "schema_version": "46.3.0",
            "phase_id": "Phase369-Diagnostic",
            "created_at": now(),
            "denominator": {
                "model_count": 3,
                "case_count": sum(row["case_count"] for row in rows),
                "head_route_count": sum(row["head_route_count"] for row in rows),
                "mlp_role_count": sum(row["mlp_role_count"] for row in rows),
                "shard_count_count": len(SHARD_COUNTS),
                "hash_seed_count": len(HASH_SEEDS),
            },
            "results": {
                "all_case_files_valid": all(row["valid"] for row in rows),
                "task_score_top_k_used": False,
                "semantic_labels_used": False,
                "single_units_causally_confirmed": False,
            },
            "models": [{key: value for key, value in row.items() if key != "case_rows"} for row in rows],
            "authorization": {
                "exploratory_blind_future_diagnostic": True,
                "phase369_calibration": False,
                "physical_holdout": False,
            },
        }
        write_json(OUT / "phase369_head_neuron_topology_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    raise SystemExit("Pass --model or --summarize")


if __name__ == "__main__":
    main()
