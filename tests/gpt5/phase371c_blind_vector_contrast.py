#!/usr/bin/env python3
"""Stream all blind condition-pair exact-vector route contrasts without candidate selection."""

from __future__ import annotations

import gc
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase358_multiresolution_component_conservation import relative_error  # noqa: E402
from phase361_r0_r1_component_trace import fragment_end_position  # noqa: E402
from phase365_dynamic_bundle_extraction import load_weight  # noqa: E402


PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
BASE = PHASE371 / "phase371c_internal_discovery"
ADJ = PHASE371 / "phase371c_adjacent_extension"
CASES = PHASE371 / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl"
PROTOCOL = PHASE371 / "phase371c_blind_vector_contrast_protocol.json"
OUT = PHASE371 / "phase371c_blind_vector_contrast"
MODELS = ("qwen3", "glm4", "deepseek7b")
ROLE_NAMES = ("source_end", "query_end", "answer_start", "current_generation")
DEPTH_NAMES = ("early", "middle", "late")
PARTITION_COUNT = 8


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def jsonl_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def prompt_token_ids(tokenizer: Any, case: dict[str, Any]) -> list[int]:
    return [int(value) for value in tokenizer(
        case["prompt"],
        add_special_tokens=bool(case["tokenization_add_special_tokens"]),
        truncation=True,
        max_length=256,
    )["input_ids"]]


def static_roles(tokenizer: Any, case: dict[str, Any]) -> tuple[list[int], int]:
    base_ids = prompt_token_ids(tokenizer, case)
    source, source_exact = fragment_end_position(
        tokenizer, case["prompt"], base_ids, case["source_fragment"], last=False,
    )
    query, query_exact = fragment_end_position(
        tokenizer, case["prompt"], base_ids, case["query_fragment"], last=True,
    )
    if not source_exact or not query_exact:
        raise RuntimeError(f"Exact role mapping failed for {case['blind_case_id']}")
    return [source, query, len(base_ids) - 1], len(base_ids)


def layer_file(root: Path, model: str, case_id: str, time: int, layer: int) -> Path:
    return root / "private/models" / model / case_id / f"time_{time}" / f"layer_{layer:03d}.pt"


def repeat_key_value(value: torch.Tensor, head_count: int) -> torch.Tensor:
    if value.shape[1] == head_count:
        return value
    return value.repeat_interleave(head_count // value.shape[1], dim=1)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    denominator = float(torch.linalg.vector_norm(left).item() * torch.linalg.vector_norm(right).item())
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).item() / denominator)


def inner_share(child: torch.Tensor, parent: torch.Tensor) -> float:
    parent = parent.float()
    denominator = float(torch.dot(parent, parent).item())
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(child.float(), parent).item() / denominator)


def route_family(route: str) -> str:
    if route.startswith("attention_partition_"):
        return "attention_head_partition_difference"
    if route.startswith("mlp_partition_"):
        return "mlp_neuron_partition_difference"
    return f"{route}_difference"


def derive_routes(
    payload: dict[str, Any],
    positions: list[int],
    o_weight: torch.Tensor,
    down_weight: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    position_tensor = torch.tensor(positions, dtype=torch.long)
    components = {
        key: value[0].index_select(0, position_tensor).float()
        for key, value in payload["component_vectors"].items()
    }
    attention = payload["attention"]
    probabilities = attention["probabilities_all_receivers_all_sources"]
    selected_probabilities = probabilities[0].index_select(1, position_tensor).to(device=device, dtype=torch.float32)
    values = attention["value_states_all_positions"].to(device=device, dtype=torch.float32)
    head_count = int(attention["head_count"])
    values = repeat_key_value(values, head_count)[0]
    weighted = torch.einsum("hqs,hsd->hqd", selected_probabilities, values)
    head_dim = int(attention["head_dim"])
    o_weight = o_weight.to(device=device, dtype=torch.float32)
    blocks = o_weight.view(o_weight.shape[0], head_count, head_dim)
    head_writes = torch.einsum("hqd,ohd->qho", weighted, blocks)
    attention_partitions = torch.stack([
        head_writes[:, start:end].sum(dim=1)
        for start, end in attention["head_partitions"]
    ])
    product = payload["mlp"]["down_projection_input_product_all_positions"][0]
    selected_product = product.index_select(0, position_tensor).to(device=device, dtype=torch.float32)
    down_weight = down_weight.to(device=device, dtype=torch.float32)
    mlp_partitions = torch.stack([
        F.linear(selected_product[:, start:end], down_weight[:, start:end])
        for start, end in payload["mlp"]["channel_partitions"]
    ])
    routes = {
        "layer_input": components["layer_input_all_positions"],
        "attention_merge": components["attention_output_all_positions"],
        "post_attention": components["post_attention_state_all_positions"],
        "mlp_merge": components["mlp_output_all_positions"],
        "layer_output": components["layer_output_all_positions"],
    }
    routes.update({f"attention_partition_{index}": value.cpu() for index, value in enumerate(attention_partitions)})
    routes.update({f"mlp_partition_{index}": value.cpu() for index, value in enumerate(mlp_partitions)})
    attention_sum = attention_partitions.sum(dim=0).cpu()
    mlp_sum = mlp_partitions.sum(dim=0).cpu()
    output_replay = (
        routes["layer_input"] + routes["attention_merge"] + routes["mlp_merge"]
    )
    errors = {
        "attention_partition_conservation": relative_error(routes["attention_merge"], attention_sum)[1],
        "mlp_partition_conservation": relative_error(routes["mlp_merge"], mlp_sum)[1],
        "block_difference_conservation": relative_error(routes["layer_output"], output_replay)[1],
    }
    return routes, errors


def group_cases(model: str) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(CASES):
        if row["private_execution_model"] == model:
            grouped[row["anonymous_group_id"]].append(row)
    groups = []
    for group_id, rows in sorted(grouped.items()):
        if len(rows) != 4:
            raise RuntimeError(f"Blind group {group_id} has {len(rows)} cases")
        groups.append(sorted(rows, key=lambda row: row["anonymous_condition_slot"]))
    if len(groups) != 22:
        raise RuntimeError(f"Expected 22 model groups for {model}, got {len(groups)}")
    return groups


def model_pairs(model: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_manifest = read_json(BASE / "models" / model / "manifest.json")
    adjacent_manifest = read_json(ADJ / "models" / model / "manifest.json")
    base_layers = list(base_manifest["anchor_layers"])
    adjacent_layers = list(adjacent_manifest["selected_layers"])
    pairs = [
        {"name": "early", "source_layer": base_layers[0], "source_root": BASE, "receiver_layer": adjacent_layers[0], "receiver_root": ADJ},
        {"name": "middle", "source_layer": base_layers[1], "source_root": BASE, "receiver_layer": adjacent_layers[1], "receiver_root": ADJ},
        {"name": "late", "source_layer": adjacent_layers[2], "source_root": ADJ, "receiver_layer": base_layers[2], "receiver_root": BASE},
    ]
    return pairs, [base_manifest, adjacent_manifest]


def process_model(model: str, route_handle: Any, vocab_handle: Any, device: torch.device) -> dict[str, Any]:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True, use_fast=False,
    )
    pairs, _manifests = model_pairs(model)
    weights = {}
    for pair in pairs:
        layer = pair["source_layer"]
        weights[pair["name"]] = (
            load_weight(model, f"model.layers.{layer}.self_attn.o_proj.weight").to(device=device, dtype=torch.float32),
            load_weight(model, f"model.layers.{layer}.mlp.down_proj.weight").to(device=device, dtype=torch.float32),
        )
    route_count = 0
    vocab_count = 0
    maxima = {
        "attention_partition_conservation": 0.0,
        "mlp_partition_conservation": 0.0,
        "block_difference_conservation": 0.0,
    }
    for group_index, cases in enumerate(group_cases(model), 1):
        role_static = {}
        for case in cases:
            static, base_length = static_roles(tokenizer, case)
            role_static[case["blind_case_id"]] = (static, base_length)
        data: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
        vocab: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        for pair in pairs:
            o_weight, down_weight = weights[pair["name"]]
            for generation_time in range(3):
                for case in cases:
                    case_id = case["blind_case_id"]
                    source_payload = torch.load(
                        layer_file(pair["source_root"], model, case_id, generation_time, pair["source_layer"]),
                        map_location="cpu", weights_only=True,
                    )
                    receiver_payload = torch.load(
                        layer_file(pair["receiver_root"], model, case_id, generation_time, pair["receiver_layer"]),
                        map_location="cpu", weights_only=True,
                    )
                    static, _base_length = role_static[case_id]
                    positions = [*static, int(source_payload["sequence_length"]) - 1]
                    routes, errors = derive_routes(source_payload, positions, o_weight, down_weight, device)
                    receiver_output = receiver_payload["component_vectors"]["layer_output_all_positions"][0]
                    receiver_roles = receiver_output.index_select(0, torch.tensor(positions)).float()
                    data[pair["name"]][generation_time][case_id] = {
                        "routes": routes,
                        "receiver_output": receiver_roles,
                        "source_layer": pair["source_layer"],
                        "receiver_layer": pair["receiver_layer"],
                    }
                    for key, value in errors.items():
                        maxima[key] = max(maxima[key], value)
        for generation_time in range(3):
            for case in cases:
                case_id = case["blind_case_id"]
                meta = torch.load(
                    BASE / "private/models" / model / case_id / f"time_{generation_time}/time_meta.pt",
                    map_location="cpu", weights_only=True,
                )
                vocab[generation_time][case_id] = meta["full_vocabulary_logits"].float()
        for left_index, right_index in itertools.combinations(range(4), 2):
            left_case = cases[left_index]
            right_case = cases[right_index]
            left_id = left_case["blind_case_id"]
            right_id = right_case["blind_case_id"]
            pair_id = f"slot{left_index}_slot{right_index}"
            for depth_index, depth_name in enumerate(DEPTH_NAMES):
                wrong_depth = DEPTH_NAMES[(depth_index + 1) % len(DEPTH_NAMES)]
                for generation_time in range(3):
                    wrong_time = (generation_time + 1) % 3
                    left_data = data[depth_name][generation_time][left_id]
                    right_data = data[depth_name][generation_time][right_id]
                    receiver_delta = left_data["receiver_output"] - right_data["receiver_output"]
                    wrong_depth_delta = (
                        data[wrong_depth][generation_time][left_id]["receiver_output"]
                        - data[wrong_depth][generation_time][right_id]["receiver_output"]
                    )
                    wrong_time_delta = (
                        data[depth_name][wrong_time][left_id]["receiver_output"]
                        - data[depth_name][wrong_time][right_id]["receiver_output"]
                    )
                    source_output_delta = (
                        left_data["routes"]["layer_output"]
                        - right_data["routes"]["layer_output"]
                    )
                    for role_index, role_name in enumerate(ROLE_NAMES):
                        wrong_role_index = (role_index + 1) % len(ROLE_NAMES)
                        parent = source_output_delta[role_index]
                        receiver = receiver_delta[role_index]
                        for route, left_vector in left_data["routes"].items():
                            delta = left_vector[role_index] - right_data["routes"][route][role_index]
                            route_handle.write(jsonl_line({
                                "schema_version": "47.17.0",
                                "phase_id": "Phase371C-Contrast",
                                "model": model,
                                "anonymous_group_id": left_case["anonymous_group_id"],
                                "anonymous_parallel_group_id": left_case["anonymous_parallel_group_id"],
                                "anonymous_pair_id": pair_id,
                                "anonymous_slot_left": left_case["anonymous_condition_slot"],
                                "anonymous_slot_right": right_case["anonymous_condition_slot"],
                                "generation_time": generation_time,
                                "depth_pair": depth_name,
                                "source_layer": left_data["source_layer"],
                                "receiver_layer": left_data["receiver_layer"],
                                "role": role_name,
                                "route": route,
                                "route_family": route_family(route),
                                "indices": {
                                    "exact_difference_norm": float(torch.linalg.vector_norm(delta.float()).item()),
                                    "signed_cosine_to_source_output_difference": cosine(delta, parent),
                                    "child_parent_inner_product_share": inner_share(delta, parent),
                                    "adjacent_output_direction_persistence": cosine(delta, receiver),
                                    "wrong_depth_control_cosine": cosine(delta, wrong_depth_delta[role_index]),
                                    "wrong_role_control_cosine": cosine(delta, receiver_delta[wrong_role_index]),
                                    "time_shuffle_control_cosine": cosine(delta, wrong_time_delta[role_index]),
                                },
                                "exact_vector_locator": {
                                    "left_case_id": left_id,
                                    "right_case_id": right_id,
                                    "source_ledger": "base" if left_data["source_layer"] in _manifests[0]["anchor_layers"] else "adjacent",
                                    "route": route,
                                    "role_index": role_index,
                                },
                                "candidate_selected": False,
                                "semantic_labels_available": False,
                            }))
                            route_count += 1
            for generation_time in range(3):
                wrong_time = (generation_time + 1) % 3
                delta = vocab[generation_time][left_id] - vocab[generation_time][right_id]
                wrong_delta = vocab[wrong_time][left_id] - vocab[wrong_time][right_id]
                vocab_handle.write(jsonl_line({
                    "schema_version": "47.17.0",
                    "phase_id": "Phase371C-Contrast",
                    "model": model,
                    "anonymous_group_id": left_case["anonymous_group_id"],
                    "anonymous_parallel_group_id": left_case["anonymous_parallel_group_id"],
                    "anonymous_pair_id": pair_id,
                    "anonymous_slot_left": left_case["anonymous_condition_slot"],
                    "anonymous_slot_right": right_case["anonymous_condition_slot"],
                    "generation_time": generation_time,
                    "route_family": "label_free_vocab_distribution_difference",
                    "indices": {
                        "exact_difference_norm": float(torch.linalg.vector_norm(delta).item()),
                        "time_persistence_cosine": cosine(delta, wrong_delta),
                    },
                    "exact_vector_locator": {
                        "left_case_id": left_id,
                        "right_case_id": right_id,
                        "tensor": "full_vocabulary_logits",
                    },
                    "candidate_selected": False,
                    "semantic_labels_available": False,
                }))
                vocab_count += 1
        print(f"[{model}] blind contrast groups {group_index}/22", flush=True)
        del data, vocab
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": model,
        "model_group_count": 22,
        "route_row_count": route_count,
        "vocab_row_count": vocab_count,
        "max_conservation_errors": maxima,
    }


def main() -> None:
    protocol = read_json(PROTOCOL)
    if not protocol["authorization"]["implement_and_hash_blind_contrast_extractor"]:
        raise RuntimeError("Blind contrast protocol is not authorized")
    OUT.mkdir(parents=True, exist_ok=True)
    route_path = OUT / "private/phase371c_blind_route_contrasts.jsonl"
    vocab_path = OUT / "private/phase371c_blind_vocab_contrasts.jsonl"
    route_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_rows = []
    with route_path.open("w", encoding="utf-8") as route_handle, vocab_path.open("w", encoding="utf-8") as vocab_handle:
        for model in MODELS:
            model_rows.append(process_model(model, route_handle, vocab_handle, device))
    route_count = sum(row["route_row_count"] for row in model_rows)
    vocab_count = sum(row["vocab_row_count"] for row in model_rows)
    summary = {
        "schema_version": "47.17.0",
        "phase_id": "Phase371C-Contrast",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "extract_all_blind_exact_route_pair_indices_without_candidate_selection",
        "execution": {
            "device": str(device),
            "model_execution": False,
            "model_order": list(MODELS),
            "semantic_condition_key_opened": False,
        },
        "denominator": {
            "model_group_count": 66,
            "unordered_pair_count": 396,
            "route_contrast_row_count": route_count,
            "vocab_contrast_row_count": vocab_count,
        },
        "models": model_rows,
        "quality": {
            "all_six_pairs_retained": route_count == 299376,
            "expected_route_row_count": 299376,
            "expected_vocab_row_count": 1188,
            "semantic_labels_available": False,
            "top_k_used": False,
            "candidate_selected": False,
            "weighted_scalar_score_used": False,
        },
        "results": {
            "blind_index_extraction_complete": route_count == 299376 and vocab_count == 1188,
            "candidate_language_path_count": 0,
            "language_mechanism_claimed": False,
        },
        "authorization": {
            "audit_and_hash_blind_rows": True,
            "open_condition_semantics_before_row_hash": False,
            "open_calibration": False,
            "open_physical": False,
        },
        "next_decision": "audit_row_denominator_and_hash_then_freeze_discovery_gate_before_semantic_mapping",
    }
    write_json(OUT / "phase371c_blind_vector_contrast_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
