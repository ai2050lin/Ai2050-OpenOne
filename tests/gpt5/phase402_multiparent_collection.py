#!/usr/bin/env python3
"""Collect Phase402 direct-child responses for all four-parent subsets."""

from __future__ import annotations

import argparse
import gc
import gzip
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase358_multiresolution_component_conservation import (  # noqa: E402
    install_hooks,
    module_attr,
)
from phase371b_anchor_qk_collection import capture_actual_qkv  # noqa: E402
from phase401_local_edge_collection import (  # noqa: E402
    capture_case,
    shifted_layer,
    state_metrics,
    to_device_layer,
)
from phase402_behavior_protocol import parse_condition  # noqa: E402
from phase402_multiparent_protocol import (  # noqa: E402
    CONTROL_NAMES,
    MODELS,
    OUT,
    PARENT_CATEGORIES,
)


SOURCE = OUT / "trace/protocol/private/phase402_discovery_trace_cases.jsonl"
BASE_CONTROLS = tuple(
    control
    for control in CONTROL_NAMES
    if control != "same_absolute_mass_sign_permuted"
)
SUBSETS = tuple(
    {
        "subset_id": f"S{mask:04b}",
        "mask": mask,
        "categories": tuple(
            category
            for bit, category in enumerate(PARENT_CATEGORIES)
            if mask & (1 << bit)
        ),
    }
    for mask in range(16)
)
SOURCE_CONTENT_ROLES = (
    "source_entity_a",
    "source_entity_b",
    "source_value_a",
    "source_value_b",
)


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


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase402 non-finite discovery value: {value}")
    return round(value, 9)


def median(values: list[float]) -> float | None:
    return clean(float(statistics.median(values))) if values else None


def condition(axis: str, relation: int, order: int, query: int) -> str:
    return f"{axis}_R{relation}_O{order}_Q{query}"


def pair_specs(cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for recipient_condition in sorted(cases):
        axis, relation, order, query = parse_condition(recipient_condition)
        other_axis = "Y" if axis == "X" else "X"
        true_donor = condition(axis, 1 - relation, order, query)
        same_target_order = condition(axis, relation, 1 - order, query)
        same_content_wrong_structure = condition(
            axis, 1 - relation, 1 - order, query
        )
        random_donor = condition(other_axis, relation, order, query)
        if cases[recipient_condition]["target"] == cases[true_donor]["target"]:
            raise RuntimeError("Phase402 true donor did not change the target")
        if cases[recipient_condition]["target"] != cases[same_target_order]["target"]:
            raise RuntimeError("Phase402 order control changed the target")
        result.append(
            {
                "pair_id": f"{recipient_condition}->{true_donor}",
                "recipient_condition": recipient_condition,
                "true_donor_condition": true_donor,
                "controls": {
                    "true_relation": (true_donor, "query_end"),
                    "same_target_wrong_order": (same_target_order, "query_end"),
                    "wrong_receiver_role": (true_donor, "query_entity"),
                    "wrong_semantic_time": (true_donor, "answer_anchor"),
                    "wrong_depth_quarter_shift": (true_donor, "query_end"),
                    "source_content_role_permutation": (true_donor, "query_end"),
                    "same_content_wrong_structure": (
                        same_content_wrong_structure,
                        "query_end",
                    ),
                    "deterministic_random_natural_donor": (
                        random_donor,
                        "query_end",
                    ),
                },
            }
        )
    if len(result) != 16:
        raise RuntimeError(f"Phase402 expected 16 directed pairs, got {len(result)}")
    return result


def subset_applicable(control: str, categories: tuple[str, ...]) -> bool:
    if not categories:
        return False
    if control == "source_content_role_permutation":
        return "source_content" in categories
    if control == "same_content_wrong_structure":
        return "source_structure" in categories
    return True


def positions_before(values: list[int], receiver: int) -> list[int]:
    return [position for position in values if position <= receiver]


def aligned_donor(
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    receiver_role: str,
    permute_source_content: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = recipient_state["key"].clone()
    value = recipient_state["value"].clone()
    receiver = int(recipient_case["state_role_positions_private"][receiver_role][0])
    donor_receiver = int(donor_case["state_role_positions_private"][receiver_role][0])
    recipient_partition = recipient_case["parent_partitions_private"][receiver_role]
    donor_partition = donor_case["parent_partitions_private"][receiver_role]

    donor_role_map = {
        "source_entity_a": "source_entity_b",
        "source_entity_b": "source_entity_a",
        "source_value_a": "source_value_b",
        "source_value_b": "source_value_a",
    } if permute_source_content else {role: role for role in SOURCE_CONTENT_ROLES}
    claimed: set[int] = set()
    for recipient_role, donor_role in donor_role_map.items():
        recipient_positions = positions_before(
            recipient_case["state_role_positions_private"][recipient_role], receiver
        )
        donor_positions = positions_before(
            donor_case["state_role_positions_private"][donor_role], donor_receiver
        )
        if len(recipient_positions) != len(donor_positions):
            raise RuntimeError("Phase402 source-content role alignment mismatch")
        if recipient_positions:
            recipient_index = torch.tensor(
                recipient_positions, dtype=torch.long, device=key.device
            )
            donor_index = torch.tensor(
                donor_positions, dtype=torch.long, device=key.device
            )
            key.index_copy_(
                1, recipient_index, donor_state["key"].index_select(1, donor_index)
            )
            value.index_copy_(
                1, recipient_index, donor_state["value"].index_select(1, donor_index)
            )
            claimed.update(recipient_positions)
    if claimed != set(recipient_partition["source_content"]):
        raise RuntimeError("Phase402 source-content subroles do not conserve category")

    for category in PARENT_CATEGORIES[1:]:
        recipient_positions = recipient_partition[category]
        donor_positions = donor_partition[category]
        if len(recipient_positions) != len(donor_positions):
            raise RuntimeError(
                f"Phase402 category alignment mismatch for {category}"
            )
        recipient_index = torch.tensor(
            recipient_positions, dtype=torch.long, device=key.device
        )
        donor_index = torch.tensor(
            donor_positions, dtype=torch.long, device=key.device
        )
        key.index_copy_(
            1, recipient_index, donor_state["key"].index_select(1, donor_index)
        )
        value.index_copy_(
            1, recipient_index, donor_state["value"].index_select(1, donor_index)
        )
    return key, value


def subset_selectors(
    case: dict[str, Any], receiver_role: str, length: int, device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    partition = case["parent_partitions_private"][receiver_role]
    selectors = torch.zeros((len(SUBSETS), length), dtype=dtype, device=device)
    for subset_index, subset in enumerate(SUBSETS):
        positions = list(
            itertools.chain.from_iterable(
                partition[category] for category in subset["categories"]
            )
        )
        if positions:
            selectors[subset_index, positions] = 1
    return selectors


@torch.inference_mode()
def recompute_subset_batch(
    layer: Any,
    base_groups: list[dict[str, Any]],
) -> list[torch.Tensor]:
    key_variants: list[torch.Tensor] = []
    value_variants: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    scales: list[float] = []
    for group in base_groups:
        recipient_state = group["recipient_state"]
        receiver = recipient_state["receivers"][group["receiver_role"]]
        aligned_key, aligned_value = aligned_donor(
            recipient_state,
            group["patch_donor_state"],
            group["recipient_case"],
            group["donor_case"],
            group["receiver_role"],
            group["control"] == "source_content_role_permutation",
        )
        selectors = subset_selectors(
            group["recipient_case"],
            group["receiver_role"],
            int(recipient_state["key"].shape[1]),
            recipient_state["key"].device,
            recipient_state["key"].dtype,
        )
        selector = selectors[:, None, :, None]
        key_variants.append(
            recipient_state["key"].unsqueeze(0)
            + selector * (aligned_key - recipient_state["key"]).unsqueeze(0)
        )
        value_variants.append(
            recipient_state["value"].unsqueeze(0)
            + selector * (aligned_value - recipient_state["value"]).unsqueeze(0)
        )
        queries.append(receiver["query"].unsqueeze(0).expand(len(SUBSETS), -1, -1))
        if receiver["mask"] is None:
            masks.append(
                torch.zeros(
                    (len(SUBSETS), int(recipient_state["key"].shape[1])),
                    dtype=recipient_state["key"].dtype,
                    device=recipient_state["key"].device,
                )
            )
        else:
            masks.append(
                receiver["mask"].to(recipient_state["key"].dtype).unsqueeze(0).expand(
                    len(SUBSETS), -1
                )
            )
        scales.extend([float(recipient_state["scaling"])] * len(SUBSETS))
    key = torch.cat(key_variants, dim=0)
    value = torch.cat(value_variants, dim=0)
    query = torch.cat(queries, dim=0)
    mask = torch.cat(masks, dim=0)
    head_count = int(query.shape[1])
    if key.shape[1] != head_count:
        repeat = head_count // int(key.shape[1])
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("bhd,bhsd->bhs", query, key)
    scale = torch.tensor(scales, device=scores.device, dtype=scores.dtype)
    scores = scores * scale[:, None, None] + mask[:, None, :]
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weighted = torch.einsum("bhs,bhsd->bhd", probabilities, value)
    o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
    projected = o_proj(weighted.reshape(weighted.shape[0], 1, -1))[:, 0]
    return list(projected.split(len(SUBSETS), dim=0))


def aggregate_layer(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["control_name"], row["subset_id"])].append(row)
    result: list[dict[str, Any]] = []
    for (control, subset_id), rows in sorted(grouped.items()):
        metrics = [row["metrics"] for row in rows if row["control_applicable"]]
        informative = [metric for metric in metrics if metric["informative"]]
        result.append(
            {
                "schema_version": "76.7.0",
                "phase_id": "Phase402-GroupLayerSubset",
                "model": rows[0]["model"],
                "surface_private": rows[0]["surface_private"],
                "public_parallel_group_id": rows[0]["public_parallel_group_id"],
                "layer_index": rows[0]["layer_index"],
                "layer_count": rows[0]["layer_count"],
                "control_name": control,
                "control_applicable": bool(metrics),
                "subset_id": subset_id,
                "subset_categories": rows[0]["subset_categories"],
                "subset_size": len(rows[0]["subset_categories"]),
                "pair_count": len(rows),
                "applicable_pair_count": len(metrics),
                "informative_pair_rate": (
                    clean(len(informative) / len(metrics)) if metrics else None
                ),
                "pair_pass_rate": (
                    clean(sum(metric["pair_pass"] for metric in metrics) / len(metrics))
                    if metrics
                    else None
                ),
                "median_state_recovery": median(
                    [metric["state_recovery"] for metric in informative]
                ),
                "median_direction_cosine": median(
                    [metric["direction_cosine"] for metric in informative]
                ),
            }
        )
    return result


@torch.inference_mode()
def analyze_group(
    loaded: Any,
    layers: list[Any],
    collected: dict[str, dict[str, Any]],
    pair_handle: Any,
) -> tuple[list[dict[str, Any]], int]:
    freeze = read_json(OUT / "phase402_discovery_execution_freeze.json")
    pair_gate = freeze["frozen_gates"]["pair_gate"]
    pairs = pair_specs({key: item["case"] for key, item in collected.items()})
    all_group_rows: list[dict[str, Any]] = []
    pair_row_count = 0
    for layer_index, layer in enumerate(layers):
        current = {
            condition_id: to_device_layer(item["layers"][layer_index], loaded.input_device)
            for condition_id, item in collected.items()
        }
        wrong_index = shifted_layer(layer_index, len(layers))
        wrong_depth = {
            condition_id: to_device_layer(item["layers"][wrong_index], loaded.input_device)
            for condition_id, item in collected.items()
        }
        base_groups: list[dict[str, Any]] = []
        for pair in pairs:
            recipient_condition = pair["recipient_condition"]
            for control in BASE_CONTROLS:
                donor_condition, receiver_role = pair["controls"][control]
                base_groups.append(
                    {
                        "pair": pair,
                        "control": control,
                        "recipient_condition": recipient_condition,
                        "donor_condition": donor_condition,
                        "receiver_role": receiver_role,
                        "recipient_case": collected[recipient_condition]["case"],
                        "donor_case": collected[donor_condition]["case"],
                        "recipient_state": current[recipient_condition],
                        "patch_donor_state": (
                            wrong_depth[donor_condition]
                            if control == "wrong_depth_quarter_shift"
                            else current[donor_condition]
                        ),
                        "reference_donor_state": current[donor_condition],
                    }
                )

        pair_rows: list[dict[str, Any]] = []
        true_outputs: dict[tuple[str, str], torch.Tensor] = {}
        chunk_size = freeze["counterfactual_execution_contract"][
            "base_control_groups_per_vectorized_chunk"
        ]
        for start in range(0, len(base_groups), chunk_size):
            selected = base_groups[start : start + chunk_size]
            outputs = recompute_subset_batch(layer, selected)
            for group, subset_outputs in zip(selected, outputs, strict=True):
                recipient_receiver = group["recipient_state"]["receivers"][
                    group["receiver_role"]
                ]
                donor_receiver = group["reference_donor_state"]["receivers"][
                    group["receiver_role"]
                ]
                for subset, counterfactual in zip(SUBSETS, subset_outputs, strict=True):
                    applicable = subset_applicable(
                        group["control"], subset["categories"]
                    )
                    metrics = state_metrics(
                        counterfactual,
                        recipient_receiver["attention"],
                        donor_receiver["attention"],
                        pair_gate["minimum_informative_baseline_relative_norm"],
                        pair_gate["direction_cosine_min"],
                        pair_gate["state_recovery_min"],
                    )
                    row = {
                        "schema_version": "76.7.0",
                        "phase_id": "Phase402-MultiParentPair",
                        "model": loaded.key,
                        "surface_private": group["recipient_case"][
                            "task_surface_private"
                        ],
                        "public_parallel_group_id": group["recipient_case"][
                            "phase402_public_parallel_group_id"
                        ],
                        "pair_id_private": group["pair"]["pair_id"],
                        "control_name": group["control"],
                        "control_applicable": applicable,
                        "receiver_role": group["receiver_role"],
                        "layer_index": layer_index,
                        "layer_count": len(layers),
                        "subset_id": subset["subset_id"],
                        "subset_categories": list(subset["categories"]),
                        "metrics": metrics,
                    }
                    pair_rows.append(row)
                    if group["control"] == "true_relation":
                        true_outputs[(group["pair"]["pair_id"], subset["subset_id"])] = (
                            counterfactual.detach()
                        )
            del outputs

        for pair in pairs:
            recipient_condition = pair["recipient_condition"]
            donor_condition = pair["true_donor_condition"]
            recipient = current[recipient_condition]["receivers"]["query_end"]
            donor = current[donor_condition]["receivers"]["query_end"]
            for subset in SUBSETS:
                true_output = true_outputs[(pair["pair_id"], subset["subset_id"])]
                delta = true_output - recipient["attention"]
                signs = torch.where(
                    torch.arange(delta.numel(), device=delta.device) % 2 == 0,
                    torch.ones(delta.numel(), device=delta.device),
                    -torch.ones(delta.numel(), device=delta.device),
                ).to(delta.dtype)
                counterfactual = recipient["attention"] + delta.abs() * signs
                applicable = subset_applicable(
                    "same_absolute_mass_sign_permuted", subset["categories"]
                )
                pair_rows.append(
                    {
                        "schema_version": "76.7.0",
                        "phase_id": "Phase402-MultiParentPair",
                        "model": loaded.key,
                        "surface_private": collected[recipient_condition]["case"][
                            "task_surface_private"
                        ],
                        "public_parallel_group_id": collected[recipient_condition][
                            "case"
                        ]["phase402_public_parallel_group_id"],
                        "pair_id_private": pair["pair_id"],
                        "control_name": "same_absolute_mass_sign_permuted",
                        "control_applicable": applicable,
                        "receiver_role": "query_end",
                        "layer_index": layer_index,
                        "layer_count": len(layers),
                        "subset_id": subset["subset_id"],
                        "subset_categories": list(subset["categories"]),
                        "metrics": state_metrics(
                            counterfactual,
                            recipient["attention"],
                            donor["attention"],
                            pair_gate[
                                "minimum_informative_baseline_relative_norm"
                            ],
                            pair_gate["direction_cosine_min"],
                            pair_gate["state_recovery_min"],
                        ),
                    }
                )
        expected = len(pairs) * len(CONTROL_NAMES) * len(SUBSETS)
        if len(pair_rows) != expected:
            raise RuntimeError(
                f"Phase402 layer row count {len(pair_rows)} != {expected}"
            )
        for row in pair_rows:
            pair_handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
        pair_row_count += len(pair_rows)
        all_group_rows.extend(aggregate_layer(pair_rows))
        del current, wrong_depth, base_groups, pair_rows, true_outputs
        if (layer_index + 1) % 8 == 0 or layer_index + 1 == len(layers):
            print(
                f"[{loaded.key}/phase402] layer {layer_index + 1}/{len(layers)}",
                flush=True,
            )
    return all_group_rows, pair_row_count


@torch.inference_mode()
def run(model: str, smoke: bool = False) -> dict[str, Any]:
    freeze = read_json(OUT / "phase402_discovery_execution_freeze.json")
    if not freeze["authorization"]["run_discovery_models_sequentially"]:
        raise RuntimeError("Phase402 discovery is not authorized")
    cases = [
        row for row in read_jsonl(SOURCE) if row["private_execution_model"] == model
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["phase402_public_parallel_group_id"]].append(case)
    group_ids = sorted(grouped)
    if smoke:
        group_ids = group_ids[:1]
    expected_groups = (
        1
        if smoke
        else len(freeze["discovery_denominator"]["eligible_surfaces"])
        * freeze["discovery_denominator"]["groups_per_surface"]
    )
    if len(group_ids) != expected_groups:
        raise RuntimeError(
            f"Phase402 discovery groups for {model}: {len(group_ids)} != {expected_groups}"
        )

    loaded = None
    handles: list[Any] = []
    all_group_rows: list[dict[str, Any]] = []
    group_audits: list[dict[str, Any]] = []
    total_pair_rows = 0
    mode = "smoke" if smoke else "discovery"
    private_root = OUT / "multiparent" / mode / "private" / model
    private_root.mkdir(parents=True, exist_ok=True)
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        with gzip.open(
            private_root / "pair_rows.jsonl.gz", "wt", encoding="utf-8"
        ) as pair_handle:
            with capture_actual_qkv(model, tuple(range(len(layers))), captures):
                for group_index, group_id in enumerate(group_ids, 1):
                    group_cases = sorted(
                        grouped[group_id],
                        key=lambda row: row["anonymous_condition_slot"],
                    )
                    if len(group_cases) != 16:
                        raise RuntimeError(f"Phase402 incomplete group: {group_id}")
                    collected = {
                        case["anonymous_condition_slot"]: capture_case(
                            loaded, layers, captures, case
                        )
                        for case in group_cases
                    }
                    if len({item["sequence_length"] for item in collected.values()}) != 1:
                        raise RuntimeError("Phase402 group sequence lengths diverged")
                    replay = all(
                        item["first_prediction_matches_frozen"]
                        for item in collected.values()
                    )
                    group_rows, pair_count = analyze_group(
                        loaded, layers, collected, pair_handle
                    )
                    all_group_rows.extend(group_rows)
                    total_pair_rows += pair_count
                    group_audits.append(
                        {
                            "schema_version": "76.7.0",
                            "phase_id": "Phase402-MultiParentGroupAudit",
                            "model": model,
                            "surface_private": group_cases[0]["task_surface_private"],
                            "public_parallel_group_id": group_id,
                            "case_count": 16,
                            "layer_count": len(layers),
                            "pair_row_count": pair_count,
                            "group_layer_subset_row_count": len(group_rows),
                            "first_token_replay_all_match": replay,
                            "valid": replay
                            and pair_count
                            == len(layers)
                            * 16
                            * len(CONTROL_NAMES)
                            * len(SUBSETS),
                        }
                    )
                    del collected, group_rows
                    gc.collect()
                    print(
                        f"[{model}/phase402] group {group_index}/{len(group_ids)} "
                        f"valid={group_audits[-1]['valid']}",
                        flush=True,
                    )
        write_jsonl(private_root / "group_layer_subset_rows.jsonl", all_group_rows)
        write_jsonl(private_root / "group_audit_rows.jsonl", group_audits)
        payload = {
            "schema_version": "76.7.0",
            "phase_id": "Phase402-MultiParentCollection",
            "created_at": now(),
            "model": model,
            "mode": mode,
            "group_count": len(group_audits),
            "case_count": len(group_audits) * 16,
            "layer_count": len(layers),
            "subset_count": len(SUBSETS),
            "control_count_including_true": len(CONTROL_NAMES),
            "pair_row_count": total_pair_rows,
            "group_layer_subset_row_count": len(all_group_rows),
            "valid_group_count": sum(row["valid"] for row in group_audits),
            "valid": bool(group_audits) and all(row["valid"] for row in group_audits),
            "claim_boundary": {
                "direct_child_response_is_a_language_path": False,
                "terminal_prediction_measured": False,
                "natural_generation_intervened": False,
            },
        }
        write_json(OUT / "multiparent" / mode / model / "complete.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.model, smoke=args.smoke)
