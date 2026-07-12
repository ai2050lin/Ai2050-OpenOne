#!/usr/bin/env python3
"""Map projection mass of every attention head and MLP channel without Top-K."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase365_dynamic_bundle_extraction import load_weight  # noqa: E402
from phase383_signed_event_map import factorial_effect  # noqa: E402


SOURCE = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"
COLLECTION = SOURCE / "collection"
OUT = ROOT / "tests/gpt5/result/phase384_exact_subunit_mass_map"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
AXES = ("content", "operation", "interaction")
ROLES = ("source", "query", "answer_start", "current_generation")
SOURCE_PARTITIONS = (
    "source",
    "query",
    "answer_start",
    "current_generation",
    "other_sources",
)
PARTITION_PRIORITY = ("source", "query", "current_generation", "answer_start")
DEPTH_BIN_COUNT = 8
EPSILON = 1e-12


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def median(values: Iterable[float]) -> float:
    rows = list(values)
    return float(statistics.median(rows)) if rows else 0.0


def depth_bin(layer: int, layer_count: int) -> int:
    return min(
        DEPTH_BIN_COUNT - 1,
        int(layer / max(layer_count - 1, 1) * DEPTH_BIN_COUNT),
    )


def projection_mass(projections: torch.Tensor) -> dict[str, float]:
    values = projections.float()
    positive = float(torch.clamp(values, min=0).sum().item())
    negative = float(torch.clamp(-values, min=0).sum().item())
    absolute = positive + negative
    net = float(values.sum().item())
    cancellation = 1.0 - abs(net) / max(absolute, EPSILON) if absolute > EPSILON else 0.0
    return {
        "positive_projection_mass": positive,
        "negative_projection_mass": negative,
        "absolute_projection_mass": absolute,
        "net_projection": net,
        "cancellation_fraction": cancellation,
    }


def partition_positions(positions: list[int], sequence_length: int) -> dict[str, list[int]]:
    role_position = dict(zip(ROLES, positions, strict=True))
    claimed: set[int] = set()
    result: dict[str, list[int]] = {role: [] for role in ROLES}
    for role in PARTITION_PRIORITY:
        position = role_position[role]
        if position not in claimed:
            result[role] = [position]
            claimed.add(position)
    result["other_sources"] = [
        position for position in range(sequence_length) if position not in claimed
    ]
    return result


def attention_subunits(
    payload: dict[str, Any],
    o_weight: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    attention = payload["attention"]
    values = attention["value_states_all_sources"].to(device, dtype=torch.float32)
    probs = attention["probabilities_role_receivers_all_sources"].to(
        device, dtype=torch.float32
    )
    head_count = int(attention["head_count"])
    kv_count = int(attention["key_value_head_count"])
    head_dim = int(attention["head_dim"])
    repeated = values
    if kv_count != head_count:
        repeated = values.repeat_interleave(head_count // kv_count, dim=1)
    repeated = repeated[0]
    sequence_length = int(repeated.shape[1])
    partitions = partition_positions(
        [int(value) for value in payload["role_positions"]], sequence_length
    )
    weight = o_weight.view(o_weight.shape[0], head_count, head_dim)
    result = torch.zeros(
        (len(ROLES), len(SOURCE_PARTITIONS), head_count, o_weight.shape[0]),
        dtype=torch.float32,
        device=device,
    )
    counts = []
    for partition_index, partition in enumerate(SOURCE_PARTITIONS):
        source_positions = partitions[partition]
        counts.append(len(source_positions))
        if not source_positions:
            continue
        index = torch.tensor(source_positions, dtype=torch.long, device=device)
        selected_values = repeated.index_select(1, index)
        for receiver_index in range(len(ROLES)):
            selected_probs = probs[0, :, receiver_index].index_select(1, index)
            weighted = torch.einsum("hs,hsd->hd", selected_probs, selected_values)
            result[receiver_index, partition_index] = torch.einsum(
                "hd,ohd->ho", weighted, weight
            )
    return result, counts


def parent_projection(delta: torch.Tensor, terminal: torch.Tensor) -> torch.Tensor:
    denominator = max(float(torch.dot(terminal.float(), terminal.float()).item()), EPSILON)
    return torch.einsum("...h,h->...", delta.float(), terminal.float()) / denominator


def process_model(model: str, split: str, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = [
        row
        for row in read_jsonl(SOURCE / "protocol/private/phase383_execution_cases.jsonl")
        if row["private_execution_model"] == model and row["phase383_split"] == split
    ]
    manifest = read_json(COLLECTION / split / "models" / model / "manifest.json")
    if not manifest["valid"] or len(cases) != manifest["case_count"]:
        raise RuntimeError(f"Invalid Phase383 {split} manifest for {model}")
    file_map = {
        (row["phase383_case_id"], row["kind"], row["layer_index"]): COLLECTION
        / row["relative_path"]
        for row in manifest["files"]
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[case["phase383_public_parallel_group_id"]].append(case)
    if any(len(rows) != len(CONDITIONS) for rows in groups.values()):
        raise RuntimeError(f"Incomplete Phase384 groups for {model}/{split}")
    layer_count = int(manifest["layer_count"])
    o_weights: dict[int, torch.Tensor] = {}
    down_weights: dict[int, torch.Tensor] = {}
    for layer in range(layer_count):
        o_weights[layer] = load_weight(
            model, f"model.layers.{layer}.self_attn.o_proj.weight"
        ).to(device, dtype=torch.float32)
        down = load_weight(
            model, f"model.layers.{layer}.mlp.down_proj.weight"
        ).to(device, dtype=torch.float32)
        down_weights[layer] = down
    rows: list[dict[str, Any]] = []
    max_attention_conservation_error = 0.0
    max_mlp_conservation_error = 0.0
    exact_attention_subunit_count = 0
    exact_mlp_subunit_count = 0
    try:
        for group_index, (group_id, group_cases) in enumerate(sorted(groups.items()), 1):
            by_condition = {row["contrast_condition"]: row for row in group_cases}
            if set(by_condition) != set(CONDITIONS):
                raise RuntimeError(f"Condition mismatch in Phase384 {model}/{group_id}")
            final_payloads = {
                condition: torch.load(
                    file_map[(case["phase383_case_id"], "layer", layer_count - 1)],
                    map_location="cpu",
                    weights_only=True,
                )
                for condition, case in by_condition.items()
            }
            final_states = {
                condition: payload["component_vectors"]["layer_output"][0, 3].float()
                for condition, payload in final_payloads.items()
            }
            terminal_effects = {
                axis: factorial_effect(final_states, axis) for axis in AXES
            }
            for layer in range(layer_count):
                payloads = {
                    condition: torch.load(
                        file_map[(case["phase383_case_id"], "layer", layer)],
                        map_location="cpu",
                        weights_only=True,
                    )
                    for condition, case in by_condition.items()
                }
                attention_by_condition = {}
                partition_counts_by_condition = {}
                for condition, payload in payloads.items():
                    attention_values, partition_counts = attention_subunits(
                        payload, o_weights[layer], device
                    )
                    attention_by_condition[condition] = attention_values
                    partition_counts_by_condition[condition] = partition_counts
                mlp_products = {
                    condition: payload["mlp"][
                        "down_projection_input_product_at_roles"
                    ][0].to(device, dtype=torch.float32)
                    for condition, payload in payloads.items()
                }
                attention_parents = {
                    condition: payload["component_vectors"]["attention_output"][0].float()
                    for condition, payload in payloads.items()
                }
                mlp_parents = {
                    condition: payload["component_vectors"]["mlp_output"][0].float()
                    for condition, payload in payloads.items()
                }
                for axis in AXES:
                    terminal = terminal_effects[axis].to(device, dtype=torch.float32)
                    terminal_squared = max(float(torch.dot(terminal, terminal).item()), EPSILON)
                    attention_delta = factorial_effect(attention_by_condition, axis)
                    attention_projections = torch.einsum(
                        "rpho,o->rph", attention_delta, terminal
                    ) / terminal_squared
                    attention_parent_delta = factorial_effect(attention_parents, axis)
                    attention_parent = parent_projection(
                        attention_parent_delta.to(device), terminal
                    )
                    attention_net = attention_projections.sum(dim=(1, 2))
                    max_attention_conservation_error = max(
                        max_attention_conservation_error,
                        float((attention_net - attention_parent).abs().max().item()),
                    )
                    for receiver_index, receiver in enumerate(ROLES):
                        for partition_index, partition in enumerate(SOURCE_PARTITIONS):
                            projections = attention_projections[
                                receiver_index, partition_index
                            ]
                            mass = projection_mass(projections)
                            exact_attention_subunit_count += int(projections.numel())
                            rows.append(
                                {
                                    "schema_version": "58.1.0",
                                    "phase_id": "Phase384-ExactSubunitMass",
                                    "split": split,
                                    "model": model,
                                    "public_parallel_group_id": group_id,
                                    "mechanism_id": group_cases[0]["mechanism_id"],
                                    "contrast_axis": axis,
                                    "semantic_time": "target_decision",
                                    "layer_index": layer,
                                    "relative_depth": layer / max(layer_count - 1, 1),
                                    "depth_bin": depth_bin(layer, layer_count),
                                    "subunit_family": "attention_head_source",
                                    "receiver_role": receiver,
                                    "source_partition": partition,
                                    "subunit_count": int(projections.numel()),
                                    "source_position_count_a": partition_counts_by_condition[
                                        CONDITIONS[0]
                                    ][partition_index],
                                    "source_position_count_b": partition_counts_by_condition[
                                        CONDITIONS[1]
                                    ][partition_index],
                                    "source_position_count_c": partition_counts_by_condition[
                                        CONDITIONS[2]
                                    ][partition_index],
                                    "source_position_count_d": partition_counts_by_condition[
                                        CONDITIONS[3]
                                    ][partition_index],
                                    **mass,
                                    "top_k_used": False,
                                }
                            )
                    product_delta = factorial_effect(mlp_products, axis)
                    weight_terminal = down_weights[layer].transpose(0, 1).mv(terminal)
                    mlp_projections = product_delta * weight_terminal.unsqueeze(0) / terminal_squared
                    mlp_parent_delta = factorial_effect(mlp_parents, axis)
                    mlp_parent = parent_projection(mlp_parent_delta.to(device), terminal)
                    mlp_net = mlp_projections.sum(dim=1)
                    max_mlp_conservation_error = max(
                        max_mlp_conservation_error,
                        float((mlp_net - mlp_parent).abs().max().item()),
                    )
                    for receiver_index, receiver in enumerate(ROLES):
                        projections = mlp_projections[receiver_index]
                        mass = projection_mass(projections)
                        exact_mlp_subunit_count += int(projections.numel())
                        rows.append(
                            {
                                "schema_version": "58.1.0",
                                "phase_id": "Phase384-ExactSubunitMass",
                                "split": split,
                                "model": model,
                                "public_parallel_group_id": group_id,
                                "mechanism_id": group_cases[0]["mechanism_id"],
                                "contrast_axis": axis,
                                "semantic_time": "target_decision",
                                "layer_index": layer,
                                "relative_depth": layer / max(layer_count - 1, 1),
                                "depth_bin": depth_bin(layer, layer_count),
                                "subunit_family": "mlp_channel",
                                "receiver_role": receiver,
                                "source_partition": "",
                                "subunit_count": int(projections.numel()),
                                "source_position_count_a": 0,
                                "source_position_count_b": 0,
                                "source_position_count_c": 0,
                                "source_position_count_d": 0,
                                **mass,
                                "top_k_used": False,
                            }
                        )
                del payloads, attention_by_condition, mlp_products
            print(
                f"[{model}/{split}] exact subunit mass {group_index}/{len(groups)} "
                f"rows={len(rows)}",
                flush=True,
            )
    finally:
        del o_weights, down_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows, {
        "model": model,
        "group_count": len(groups),
        "case_count": len(cases),
        "layer_count": layer_count,
        "mass_row_count": len(rows),
        "exact_attention_head_event_count": exact_attention_subunit_count,
        "exact_mlp_channel_event_count": exact_mlp_subunit_count,
        "maximum_attention_parent_projection_error": max_attention_conservation_error,
        "maximum_mlp_parent_projection_error": max_mlp_conservation_error,
    }


def build_model_cells(rows: list[dict[str, Any]], group_count: int, contract: dict[str, Any]) -> list[dict[str, Any]]:
    group_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["subunit_family"],
            row["receiver_role"],
            row["source_partition"],
            row["depth_bin"],
            row["public_parallel_group_id"],
        )
        group_buckets[key].append(row)
    group_cells = []
    for key, bucket in sorted(group_buckets.items()):
        group_cells.append(
            {
                "model": key[0],
                "mechanism_id": key[1],
                "contrast_axis": key[2],
                "subunit_family": key[3],
                "receiver_role": key[4],
                "source_partition": key[5],
                "depth_bin": key[6],
                "public_parallel_group_id": key[7],
                "median_absolute_projection_mass": median(
                    row["absolute_projection_mass"] for row in bucket
                ),
                "median_net_projection": median(row["net_projection"] for row in bucket),
                "median_cancellation_fraction": median(
                    row["cancellation_fraction"] for row in bucket
                ),
            }
        )
    model_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in group_cells:
        key = (
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["subunit_family"],
            row["receiver_role"],
            row["source_partition"],
            row["depth_bin"],
        )
        model_buckets[key].append(row)
    gates = contract["frozen_pattern_gates"]
    result = []
    for key, bucket in sorted(model_buckets.items()):
        coherent_flags = [
            row["median_absolute_projection_mass"]
            >= gates["minimum_absolute_projection_mass"]
            and row["median_cancellation_fraction"]
            <= gates["coherent_maximum_cancellation_fraction"]
            and abs(row["median_net_projection"])
            >= gates["coherent_minimum_absolute_net_projection"]
            for row in bucket
        ]
        opposing_flags = [
            row["median_absolute_projection_mass"]
            >= gates["minimum_absolute_projection_mass"]
            and row["median_cancellation_fraction"]
            >= gates["opposing_minimum_cancellation_fraction"]
            for row in bucket
        ]
        result.append(
            {
                "schema_version": "58.1.0",
                "phase_id": "Phase384-ExactSubunitMass",
                "model": key[0],
                "mechanism_id": key[1],
                "contrast_axis": key[2],
                "subunit_family": key[3],
                "receiver_role": key[4],
                "source_partition": key[5],
                "depth_bin": key[6],
                "group_count": len(bucket),
                "median_absolute_projection_mass": median(
                    row["median_absolute_projection_mass"] for row in bucket
                ),
                "median_net_projection": median(
                    row["median_net_projection"] for row in bucket
                ),
                "median_cancellation_fraction": median(
                    row["median_cancellation_fraction"] for row in bucket
                ),
                "coherent_group_pass_count": sum(coherent_flags),
                "opposing_group_pass_count": sum(opposing_flags),
                "coherent_model_pass": len(bucket) == group_count and all(coherent_flags),
                "opposing_model_pass": len(bucket) == group_count and all(opposing_flags),
            }
        )
    return result


def crossmodel_patterns(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        key = (
            row["mechanism_id"],
            row["contrast_axis"],
            row["subunit_family"],
            row["receiver_role"],
            row["source_partition"],
            row["depth_bin"],
        )
        buckets[key].append(row)
    result = []
    for key, bucket in sorted(buckets.items()):
        for pattern in ("coherent", "opposing"):
            passing = sorted(
                row["model"] for row in bucket if row[f"{pattern}_model_pass"]
            )
            level2 = "glm4" in passing and bool(
                set(passing) & {"qwen3", "deepseek7b"}
            )
            level3 = set(passing) == set(MODELS)
            result.append(
                {
                    "schema_version": "58.1.0",
                    "phase_id": "Phase384-ExactSubunitMass",
                    "mechanism_id": key[0],
                    "contrast_axis": key[1],
                    "subunit_family": key[2],
                    "receiver_role": key[3],
                    "source_partition": key[4],
                    "depth_bin": key[5],
                    "pattern_type": pattern,
                    "passing_models": passing,
                    "heterogeneous_level2_pass": level2,
                    "level3_pass": level3,
                    "upstream_cell": key[5] <= 5,
                    "terminal_interface_cell": (
                        key[5] == 7 and key[3] == "current_generation"
                    ),
                    "median_absolute_projection_mass_across_models": median(
                        row["median_absolute_projection_mass"] for row in bucket
                    ),
                    "median_net_projection_across_models": median(
                        row["median_net_projection"] for row in bucket
                    ),
                    "median_cancellation_fraction_across_models": median(
                        row["median_cancellation_fraction"] for row in bucket
                    ),
                    "language_path_established": False,
                }
            )
    return result


def signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["mechanism_id"],
        row["contrast_axis"],
        row["subunit_family"],
        row["receiver_role"],
        row["source_partition"],
        row["depth_bin"],
        row["pattern_type"],
    )


def run(split: str) -> None:
    contract = read_json(OUT / "phase384_subunit_mass_contract.json")
    if split == "discovery":
        if not contract["authorization"]["discovery_subunit_mass_extraction"]:
            raise RuntimeError("Phase384 discovery is not authorized")
        group_count = 3
    else:
        freeze = read_json(OUT / "phase384_discovery_mass_freeze.json")
        if not freeze["authorization"]["calibration_subunit_mass_extraction"]:
            raise RuntimeError("Phase384 calibration is not authorized")
        group_count = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    model_summaries = []
    for model in MODELS:
        rows, summary = process_model(model, split, device)
        path = OUT / f"private/{split}/phase384_{model}_subunit_mass.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        summary["parquet_relative_path"] = str(path.relative_to(OUT))
        summary["parquet_byte_count"] = path.stat().st_size
        model_summaries.append(summary)
        all_rows.extend(rows)
    cells = build_model_cells(all_rows, group_count, contract)
    patterns = crossmodel_patterns(cells)
    write_jsonl(OUT / f"phase384_{split}_model_cells.jsonl", cells)
    write_jsonl(OUT / f"phase384_{split}_crossmodel_patterns.jsonl", patterns)
    conservation_limit = contract["projection_mass"][
        "maximum_parent_projection_absolute_error"
    ]
    conservation_pass = all(
        row["maximum_attention_parent_projection_error"] <= conservation_limit
        and row["maximum_mlp_parent_projection_error"] <= conservation_limit
        for row in model_summaries
    )
    passing = [row for row in patterns if row["heterogeneous_level2_pass"]]
    upstream = [row for row in passing if row["upstream_cell"]]
    coherent = [row for row in passing if row["pattern_type"] == "coherent"]
    opposing = [row for row in passing if row["pattern_type"] == "opposing"]
    if split == "discovery":
        summary = {
            "schema_version": "58.1.0",
            "phase_id": "Phase384-ExactSubunitMassDiscovery",
            "created_at": now(),
            "denominator": {
                "model_count": 3,
                "mechanism_count": 4,
                "parallel_group_count": 12,
                "case_count": sum(row["case_count"] for row in model_summaries),
                "mass_row_count": len(all_rows),
                "exact_attention_head_event_count": sum(
                    row["exact_attention_head_event_count"] for row in model_summaries
                ),
                "exact_mlp_channel_event_count": sum(
                    row["exact_mlp_channel_event_count"] for row in model_summaries
                ),
            },
            "models": model_summaries,
            "results": {
                "parent_projection_conservation_pass": conservation_pass,
                "heterogeneous_level2_pattern_count": len(passing),
                "level3_pattern_count": sum(row["level3_pass"] for row in passing),
                "coherent_pattern_count": len(coherent),
                "opposing_pattern_count": len(opposing),
                "upstream_level2_pattern_count": len(upstream),
                "upstream_coherent_pattern_count": sum(
                    row["pattern_type"] == "coherent" for row in upstream
                ),
                "upstream_opposing_pattern_count": sum(
                    row["pattern_type"] == "opposing" for row in upstream
                ),
                "top_k_used": False,
                "language_path_discovered": False,
            },
            "claim_boundary": {
                "projection_mass_pattern_is_causal_path": False,
                "same_functional_cell_is_same_unit_identity": False,
                "all_exact_subunits_included": True,
            },
            "authorization": {
                "calibration_subunit_mass_extraction": conservation_pass
                and len(passing) > 0,
                "physical_holdout": False,
                "causal_intervention": False,
            },
        }
        write_json(OUT / "phase384_discovery_summary.json", summary)
        freeze = {
            "schema_version": "58.1.1",
            "phase_id": "Phase384-DiscoveryMassFreeze",
            "created_at": now(),
            "threshold_retuned": False,
            "frozen_pattern_count": len(passing),
            "frozen_patterns": passing,
            "authorization": {
                "calibration_subunit_mass_extraction": summary["authorization"][
                    "calibration_subunit_mass_extraction"
                ],
                "physical_holdout": False,
                "causal_intervention": False,
            },
        }
        write_json(OUT / "phase384_discovery_mass_freeze.json", freeze)
    else:
        freeze = read_json(OUT / "phase384_discovery_mass_freeze.json")
        by_signature = {signature(row): row for row in patterns}
        replication = []
        for frozen in freeze["frozen_patterns"]:
            calibrated = by_signature.get(signature(frozen))
            replication.append(
                {
                    "schema_version": "58.2.0",
                    "phase_id": "Phase384-ExactSubunitMassCalibration",
                    **{
                        key: frozen[key]
                        for key in (
                            "mechanism_id",
                            "contrast_axis",
                            "subunit_family",
                            "receiver_role",
                            "source_partition",
                            "depth_bin",
                            "pattern_type",
                            "upstream_cell",
                            "terminal_interface_cell",
                        )
                    },
                    "calibration_level2_pass": bool(
                        calibrated and calibrated["heterogeneous_level2_pass"]
                    ),
                    "calibration_level3_pass": bool(
                        calibrated and calibrated["level3_pass"]
                    ),
                    "calibration_passing_models": (
                        calibrated["passing_models"] if calibrated else []
                    ),
                    "language_path_established": False,
                }
            )
        write_jsonl(OUT / "phase384_calibration_replication_rows.jsonl", replication)
        replicated = [row for row in replication if row["calibration_level2_pass"]]
        replicated_upstream = [row for row in replicated if row["upstream_cell"]]
        summary = {
            "schema_version": "58.2.0",
            "phase_id": "Phase384-ExactSubunitMassCalibration",
            "created_at": now(),
            "denominator": {
                "model_count": 3,
                "mechanism_count": 4,
                "parallel_group_count": 8,
                "case_count": sum(row["case_count"] for row in model_summaries),
                "mass_row_count": len(all_rows),
                "frozen_pattern_count": len(replication),
            },
            "models": model_summaries,
            "results": {
                "parent_projection_conservation_pass": conservation_pass,
                "level2_replication_count": len(replicated),
                "level3_replication_count": sum(
                    row["calibration_level3_pass"] for row in replicated
                ),
                "upstream_level2_replication_count": len(replicated_upstream),
                "upstream_coherent_replication_count": sum(
                    row["pattern_type"] == "coherent" for row in replicated_upstream
                ),
                "upstream_opposing_replication_count": sum(
                    row["pattern_type"] == "opposing" for row in replicated_upstream
                ),
                "replication_by_subunit_family": dict(
                    Counter(row["subunit_family"] for row in replicated)
                ),
                "top_k_used": False,
                "language_path_discovered": False,
            },
            "claim_boundary": {
                "replicated_opposing_mass_is_language_path": False,
                "replicated_coherent_mass_is_causal_path": False,
                "physical_holdout_unopened": True,
                "generation_time_relation_graph_completed": False,
            },
            "authorization": {
                "physical_holdout": False,
                "causal_intervention": False,
                "multi_time_relation_graph": True,
            },
            "next_decision": "build_relation_aware_multi_time_event_graph_without_opening_physical_holdout",
        }
        write_json(OUT / "phase384_calibration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("discovery", "calibration"), required=True)
    args = parser.parse_args()
    run(args.split)
