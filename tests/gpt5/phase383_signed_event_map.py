#!/usr/bin/env python3
"""Build signed target-decision component-event maps from Phase383 ledgers."""

from __future__ import annotations

import argparse
import json
import math
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


OUT = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"
COLLECTION = OUT / "collection"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
AXES = ("content", "operation", "interaction")
ROLE_NAMES = ("source", "query", "answer_start", "current_generation")
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


def factorial_effect(values: dict[str, torch.Tensor], axis: str) -> torch.Tensor:
    a, b, c, d = (values[name].float() for name in CONDITIONS)
    if axis == "content":
        return ((a - c) + (b - d)) * 0.5
    if axis == "operation":
        return ((a - b) + (c - d)) * 0.5
    if axis == "interaction":
        return ((a - b) - (c - d)) * 0.5
    raise ValueError(axis)


def descriptors(event_delta: torch.Tensor, terminal: torch.Tensor) -> tuple[float, float, float, float, float]:
    event = event_delta.float()
    target = terminal.float()
    event_norm = float(torch.linalg.vector_norm(event).item())
    terminal_norm = float(torch.linalg.vector_norm(target).item())
    dot = float(torch.dot(event, target).item())
    alignment = dot / max(event_norm * terminal_norm, EPSILON)
    amplitude = event_norm / max(terminal_norm, EPSILON)
    projection = dot / max(terminal_norm * terminal_norm, EPSILON)
    return event_norm, terminal_norm, alignment, amplitude, projection


def depth_bin(layer: int, layer_count: int) -> int:
    relative = layer / max(layer_count - 1, 1)
    return min(DEPTH_BIN_COUNT - 1, int(relative * DEPTH_BIN_COUNT))


def attention_role_events(
    payload: dict[str, Any],
    o_weight: torch.Tensor,
    device: torch.device,
) -> dict[tuple[str, str, str], torch.Tensor]:
    roles = list(payload["role_names"])
    if tuple(roles) != ROLE_NAMES:
        raise RuntimeError(f"Unexpected Phase383 roles: {roles}")
    positions = [int(value) for value in payload["role_positions"]]
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
    weight = o_weight.view(o_weight.shape[0], head_count, head_dim)
    selected_attention = payload["component_vectors"]["attention_output"].to(
        device, dtype=torch.float32
    )[0]
    unique_source_positions = sorted(set(positions))
    result: dict[tuple[str, str, str], torch.Tensor] = {}
    for receiver_index, receiver_role in enumerate(roles):
        unique_sum = torch.zeros_like(selected_attention[receiver_index])
        unique_vectors: dict[int, torch.Tensor] = {}
        for source_position in unique_source_positions:
            weighted = (
                probs[0, :, receiver_index, source_position].unsqueeze(-1)
                * repeated[:, source_position, :]
            )
            vector = torch.einsum("hd,ohd->o", weighted, weight)
            unique_vectors[source_position] = vector
            unique_sum += vector
        for source_role, source_position in zip(roles, positions, strict=True):
            result[(
                receiver_role,
                source_role,
                "source_role_position_retained_per_condition",
            )] = (
                unique_vectors[source_position].detach().cpu()
            )
        result[(receiver_role, "other_sources", "alias_count_0")] = (
            selected_attention[receiver_index] - unique_sum
        ).detach().cpu()
    return result


def event_vectors(
    payload: dict[str, Any],
    o_weight: torch.Tensor,
    device: torch.device,
) -> dict[tuple[str, str, str, str], torch.Tensor]:
    components = payload["component_vectors"]
    result: dict[tuple[str, str, str, str], torch.Tensor] = {}
    component_map = {
        "layer_input_state": "layer_input",
        "attention_output_write": "attention_output",
        "post_attention_state": "post_attention_state",
        "mlp_output_write": "mlp_output",
        "layer_output_state": "layer_output",
    }
    for event_type, component in component_map.items():
        tensor = components[component][0].float().cpu()
        for role_index, receiver_role in enumerate(payload["role_names"]):
            result[(event_type, receiver_role, "", "alias_count_0")] = tensor[
                role_index
            ]
    for (receiver, source, alias), vector in attention_role_events(
        payload, o_weight, device
    ).items():
        event_type = (
            "attention_other_sources_write"
            if source == "other_sources"
            else "attention_source_write"
        )
        result[(event_type, receiver, source, alias)] = vector
    return result


def process_model(model: str, split: str, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = [
        row
        for row in read_jsonl(OUT / "protocol/private/phase383_execution_cases.jsonl")
        if row["private_execution_model"] == model
        and row["phase383_split"] == split
    ]
    manifest = read_json(COLLECTION / split / "models" / model / "manifest.json")
    if not manifest["valid"] or len(cases) != manifest["case_count"]:
        raise RuntimeError(f"Invalid Phase383 collection for {model}/{split}")
    file_map = {
        (row["phase383_case_id"], row["kind"], row["layer_index"]): COLLECTION
        / row["relative_path"]
        for row in manifest["files"]
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[case["phase383_public_parallel_group_id"]].append(case)
    if any(len(rows) != len(CONDITIONS) for rows in groups.values()):
        raise RuntimeError(f"Incomplete Phase383 model groups for {model}/{split}")

    layer_count = int(manifest["layer_count"])
    o_weights = {
        layer: load_weight(
            model, f"model.layers.{layer}.self_attn.o_proj.weight"
        ).to(device, dtype=torch.float32)
        for layer in range(layer_count)
    }
    event_rows: list[dict[str, Any]] = []
    alias_counts = Counter()
    try:
        for group_index, (group_id, group_cases) in enumerate(sorted(groups.items()), 1):
            by_condition = {row["contrast_condition"]: row for row in group_cases}
            if set(by_condition) != set(CONDITIONS):
                raise RuntimeError(f"Condition mismatch in {model}/{group_id}")
            layer_payloads: dict[str, list[dict[str, Any]]] = {}
            for condition, case in by_condition.items():
                layer_payloads[condition] = [
                    torch.load(
                        file_map[(case["phase383_case_id"], "layer", layer)],
                        map_location="cpu",
                        weights_only=True,
                    )
                    for layer in range(layer_count)
                ]
            final_values = {
                condition: payloads[-1]["component_vectors"]["layer_output"][
                    0, ROLE_NAMES.index("current_generation")
                ].float()
                for condition, payloads in layer_payloads.items()
            }
            terminal_effects = {
                axis: factorial_effect(final_values, axis) for axis in AXES
            }
            for layer in range(layer_count):
                condition_events = {
                    condition: event_vectors(
                        payloads[layer], o_weights[layer], device
                    )
                    for condition, payloads in layer_payloads.items()
                }
                event_keys = set(condition_events[CONDITIONS[0]])
                if any(set(values) != event_keys for values in condition_events.values()):
                    raise RuntimeError(
                        f"Event ledger mismatch in {model}/{group_id}/layer{layer}"
                    )
                for axis in AXES:
                    terminal = terminal_effects[axis]
                    for event_type, receiver, source, alias in sorted(event_keys):
                        effect = factorial_effect(
                            {
                                condition: condition_events[condition][
                                    (event_type, receiver, source, alias)
                                ]
                                for condition in CONDITIONS
                            },
                            axis,
                        )
                        event_norm, terminal_norm, alignment, amplitude, projection = (
                            descriptors(effect, terminal)
                        )
                        alias_counts[alias] += 1
                        event_rows.append(
                            {
                                "schema_version": "57.4.0",
                                "phase_id": "Phase383-SignedEventMap",
                                "split": split,
                                "model": model,
                                "public_parallel_group_id": group_id,
                                "mechanism_id": group_cases[0]["mechanism_id"],
                                "contrast_axis": axis,
                                "semantic_time": "target_decision",
                                "layer_index": layer,
                                "layer_count": layer_count,
                                "relative_depth": layer / max(layer_count - 1, 1),
                                "depth_bin": depth_bin(layer, layer_count),
                                "event_type": event_type,
                                "receiver_role": receiver,
                                "source_role": source,
                                "alias_class": alias,
                                "event_norm": event_norm,
                                "terminal_norm": terminal_norm,
                                "signed_alignment": alignment,
                                "relative_amplitude": amplitude,
                                "signed_projection": projection,
                                "absolute_cosine_used": False,
                                "amplitude_clipped": False,
                                "composite_score_used": False,
                            }
                        )
            print(
                f"[{model}/{split}] event map {group_index}/{len(groups)} "
                f"rows={len(event_rows)}",
                flush=True,
            )
            del layer_payloads
    finally:
        del o_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return event_rows, {
        "model": model,
        "group_count": len(groups),
        "case_count": len(cases),
        "layer_count": layer_count,
        "event_row_count": len(event_rows),
        "alias_observation_counts": dict(alias_counts),
    }


def model_cells(event_rows: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    group_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        key = (
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["event_type"],
            row["receiver_role"],
            row["source_role"],
            row["depth_bin"],
            row["public_parallel_group_id"],
        )
        group_buckets[key].append(row)
    group_cells = []
    for key, rows in sorted(group_buckets.items()):
        group_cells.append(
            {
                "model": key[0],
                "mechanism_id": key[1],
                "contrast_axis": key[2],
                "event_type": key[3],
                "receiver_role": key[4],
                "source_role": key[5],
                "depth_bin": key[6],
                "public_parallel_group_id": key[7],
                "layer_count_in_bin": len(rows),
                "median_signed_alignment": median(
                    row["signed_alignment"] for row in rows
                ),
                "median_relative_amplitude": median(
                    row["relative_amplitude"] for row in rows
                ),
                "median_signed_projection": median(
                    row["signed_projection"] for row in rows
                ),
            }
        )
    model_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in group_cells:
        key = (
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["event_type"],
            row["receiver_role"],
            row["source_role"],
            row["depth_bin"],
        )
        model_buckets[key].append(row)
    floors = contract["candidate_gates"]
    cells = []
    for key, rows in sorted(model_buckets.items()):
        alignments = [row["median_signed_alignment"] for row in rows]
        amplitudes = [row["median_relative_amplitude"] for row in rows]
        cells.append(
            {
                "schema_version": "57.4.0",
                "phase_id": "Phase383-SignedEventMap",
                "model": key[0],
                "mechanism_id": key[1],
                "contrast_axis": key[2],
                "event_type": key[3],
                "receiver_role": key[4],
                "source_role": key[5],
                "depth_bin": key[6],
                "group_count": len(rows),
                "positive_direction_group_count": sum(value > 0 for value in alignments),
                "alignment_gate_group_count": sum(
                    value >= floors["minimum_group_median_signed_alignment"]
                    for value in alignments
                ),
                "amplitude_gate_group_count": sum(
                    value >= floors["minimum_group_median_relative_amplitude"]
                    for value in amplitudes
                ),
                "median_signed_alignment": median(alignments),
                "median_relative_amplitude": median(amplitudes),
                "median_signed_projection": median(
                    row["median_signed_projection"] for row in rows
                ),
            }
        )
    by_key = {
        (
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["event_type"],
            row["receiver_role"],
            row["source_role"],
            row["depth_bin"],
        ): row
        for row in cells
    }
    required = int(floors["minimum_group_count"])
    for row in cells:
        wrong_depth = by_key.get(
            (
                row["model"],
                row["mechanism_id"],
                row["contrast_axis"],
                row["event_type"],
                row["receiver_role"],
                row["source_role"],
                (row["depth_bin"] + DEPTH_BIN_COUNT // 2) % DEPTH_BIN_COUNT,
            )
        )
        other_receivers = [
            candidate["median_signed_alignment"]
            for candidate in cells
            if candidate["model"] == row["model"]
            and candidate["mechanism_id"] == row["mechanism_id"]
            and candidate["contrast_axis"] == row["contrast_axis"]
            and candidate["event_type"] == row["event_type"]
            and candidate["source_role"] == row["source_role"]
            and candidate["depth_bin"] == row["depth_bin"]
            and candidate["receiver_role"] != row["receiver_role"]
        ]
        wrong_depth_alignment = (
            wrong_depth["median_signed_alignment"] if wrong_depth else 0.0
        )
        wrong_receiver_alignment = max(other_receivers, default=0.0)
        row["wrong_depth_signed_alignment"] = wrong_depth_alignment
        row["wrong_receiver_signed_alignment"] = wrong_receiver_alignment
        row["wrong_depth_alignment_margin"] = (
            row["median_signed_alignment"] - wrong_depth_alignment
        )
        row["wrong_receiver_alignment_margin"] = (
            row["median_signed_alignment"] - wrong_receiver_alignment
        )
        row["group_count_gate"] = row["group_count"] >= required
        row["positive_direction_gate"] = (
            row["positive_direction_group_count"] == row["group_count"]
        )
        row["alignment_gate"] = (
            row["alignment_gate_group_count"] == row["group_count"]
        )
        row["amplitude_gate"] = (
            row["amplitude_gate_group_count"] == row["group_count"]
        )
        row["wrong_depth_gate"] = (
            row["wrong_depth_alignment_margin"]
            >= floors["minimum_wrong_depth_alignment_margin"]
        )
        row["wrong_receiver_gate"] = (
            row["wrong_receiver_alignment_margin"]
            >= floors["minimum_wrong_receiver_alignment_margin"]
        )
        row["model_cell_pass"] = all(
            row[name]
            for name in (
                "group_count_gate",
                "positive_direction_gate",
                "alignment_gate",
                "amplitude_gate",
                "wrong_depth_gate",
                "wrong_receiver_gate",
            )
        )
    return cells


def crossmodel_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        key = (
            row["mechanism_id"],
            row["contrast_axis"],
            row["event_type"],
            row["receiver_role"],
            row["source_role"],
            row["depth_bin"],
        )
        buckets[key].append(row)
    result = []
    for key, rows in sorted(buckets.items()):
        passing = sorted(row["model"] for row in rows if row["model_cell_pass"])
        models = {row["model"] for row in rows}
        level2 = "glm4" in passing and bool(set(passing) & {"qwen3", "deepseek7b"})
        level3 = set(passing) == set(MODELS)
        terminal = key[5] == DEPTH_BIN_COUNT - 1 and key[3] == "current_generation"
        upstream = key[5] <= 5
        result.append(
            {
                "schema_version": "57.4.0",
                "phase_id": "Phase383-SignedEventMap",
                "mechanism_id": key[0],
                "contrast_axis": key[1],
                "event_type": key[2],
                "receiver_role": key[3],
                "source_role": key[4],
                "depth_bin": key[5],
                "model_count": len(models),
                "passing_models": passing,
                "heterogeneous_level2_pass": level2,
                "level3_pass": level3,
                "terminal_interface_cell": terminal,
                "upstream_cell": upstream,
                "median_signed_alignment_across_models": median(
                    row["median_signed_alignment"] for row in rows
                ),
                "median_relative_amplitude_across_models": median(
                    row["median_relative_amplitude"] for row in rows
                ),
            }
        )
    return result


def reuse_rows(cross_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mechanisms = sorted({row["mechanism_id"] for row in cross_rows})
    sets: dict[str, set[tuple[Any, ...]]] = {}
    for mechanism in mechanisms:
        sets[mechanism] = {
            (
                row["contrast_axis"],
                row["event_type"],
                row["receiver_role"],
                row["source_role"],
                row["depth_bin"],
            )
            for row in cross_rows
            if row["mechanism_id"] == mechanism
            and row["heterogeneous_level2_pass"]
        }
    result = []
    for left in mechanisms:
        for right in mechanisms:
            union = sets[left] | sets[right]
            overlap = sets[left] & sets[right]
            result.append(
                {
                    "schema_version": "57.4.0",
                    "phase_id": "Phase383-SignedEventMap",
                    "left_mechanism": left,
                    "right_mechanism": right,
                    "left_cell_count": len(sets[left]),
                    "right_cell_count": len(sets[right]),
                    "overlap_cell_count": len(overlap),
                    "union_cell_count": len(union),
                    "descriptive_jaccard": len(overlap) / len(union) if union else 0.0,
                    "composite_weight_used": False,
                    "causal_reuse_established": False,
                }
            )
    return result


def main(split: str, device_name: str) -> None:
    contract = read_json(OUT / "phase383_signed_event_contract.json")
    if not contract["authorization"]["signed_discovery_map_extraction"]:
        raise RuntimeError("Signed Phase383 event extraction is not authorized")
    if split != "discovery":
        raise RuntimeError("This extractor freezes discovery only; calibration uses a separate evaluator")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if str(device) == "auto":
        device = torch.device("cpu")
    all_rows: list[dict[str, Any]] = []
    model_summaries = []
    private_root = OUT / "signed_event_map/private"
    for model in MODELS:
        rows, summary = process_model(model, split, device)
        path = private_root / f"phase383_{model}_event_rows.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        summary["parquet_relative_path"] = str(path.relative_to(OUT))
        summary["parquet_byte_count"] = path.stat().st_size
        model_summaries.append(summary)
        all_rows.extend(rows)
    cells = model_cells(all_rows, contract)
    cross_rows = crossmodel_cells(cells)
    reuse = reuse_rows(cross_rows)
    write_jsonl(OUT / "phase383_model_event_cells.jsonl", cells)
    write_jsonl(OUT / "phase383_crossmodel_event_cells.jsonl", cross_rows)
    write_jsonl(OUT / "phase383_reuse_difference_matrix.jsonl", reuse)
    candidates = [row for row in cross_rows if row["heterogeneous_level2_pass"]]
    upstream = [row for row in candidates if row["upstream_cell"]]
    terminal = [row for row in candidates if row["terminal_interface_cell"]]
    level3 = [row for row in candidates if row["level3_pass"]]
    summary = {
        "schema_version": "57.4.0",
        "phase_id": "Phase383-SignedEventMap",
        "created_at": now(),
        "denominator": {
            "model_count": len(MODELS),
            "mechanism_count": 4,
            "discovery_parallel_group_count": 12,
            "case_count": sum(row["case_count"] for row in model_summaries),
            "event_row_count": len(all_rows),
            "model_cell_count": len(cells),
            "crossmodel_cell_count": len(cross_rows),
        },
        "models": model_summaries,
        "results": {
            "model_cell_pass_count": sum(row["model_cell_pass"] for row in cells),
            "heterogeneous_level2_candidate_count": len(candidates),
            "level3_candidate_count": len(level3),
            "upstream_level2_candidate_count": len(upstream),
            "terminal_interface_level2_candidate_count": len(terminal),
            "candidate_counts_by_event_type": dict(
                Counter(row["event_type"] for row in candidates)
            ),
            "candidate_counts_by_depth_bin": dict(
                Counter(str(row["depth_bin"]) for row in candidates)
            ),
            "descriptive_reuse_matrix_built": True,
            "terminal_prediction_gain_computed": False,
            "language_path_discovered": False,
        },
        "claim_boundary": {
            "event_rows_are_independent_samples": False,
            "independent_units_are_parallel_groups": True,
            "discovery_group_count_per_mechanism": 3,
            "source_roles_are_aggregated_across_heads": True,
            "exact_head_events_remain_lazy_replayable": True,
            "exact_mlp_channel_events_remain_lazy_replayable": True,
            "candidate_is_causal_path": False,
            "reuse_matrix_is_descriptive_not_causal": True,
        },
        "authorization": {
            "freeze_discovery_candidates": len(candidates) > 0,
            "calibration_collection": len(candidates) > 0,
            "physical_holdout_collection": False,
            "causal_intervention": False,
        },
        "next_decision": (
            "freeze_all_threshold_passing_cells_and_run_independent_calibration"
            if candidates
            else "stop_candidate_route_and_expand_event_coordinates"
        ),
    }
    write_json(OUT / "phase383_signed_event_map_summary.json", summary)
    freeze = {
        "schema_version": "57.4.1",
        "phase_id": "Phase383-DiscoveryMapFreeze",
        "created_at": now(),
        "source_contract": "phase383_signed_event_contract.json",
        "threshold_retuned": False,
        "frozen_candidate_count": len(candidates),
        "frozen_candidates": candidates,
        "authorization": {
            "calibration_collection": len(candidates) > 0,
            "physical_holdout_collection": False,
            "causal_intervention": False,
        },
    }
    write_json(OUT / "phase383_discovery_map_freeze.json", freeze)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("discovery",), default="discovery")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    main(args.split, args.device)
