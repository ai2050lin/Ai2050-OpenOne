#!/usr/bin/env python3
"""Collect Phase425 matched-literal role formation and legal source transport."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase358_multiresolution_component_conservation import install_hooks  # noqa: E402
from phase358_multiresolution_component_conservation import module_attr  # noqa: E402
from phase371b_anchor_qk_collection import capture_actual_qkv, repeat_key_value  # noqa: E402
from phase420_typed_path_trace import source_writes  # noqa: E402
from phase424_global_physical_collect import (  # noqa: E402
    collect_condition,
    normalized_delta,
    tensor_cosine,
)
from phase425_role_exchange_protocol import (  # noqa: E402
    HISTORIES,
    INTERFACES,
    MODELS,
    OUT,
    ROLES,
    SCHEMA_VERSION,
)


PHASE_ID = "Phase425-RoleExchangeCollection"
PAIR_FILE = OUT / "phase425_registered_pairs.jsonl"
OPEN_FILE = OUT / "phase425_registered_conditions_open.jsonl"
SEALED_FILE = OUT / "sealed" / "phase425_registered_conditions_sealed.jsonl"
LEDGER_THRESHOLD = 0.01
LAYER_FEATURES = (
    "formation_specificity",
    "transport_specificity",
    "formation_functional_specificity",
    "transport_functional_specificity",
    "formation_role_dominance",
    "transport_role_dominance",
    "competition_specificity",
    "role_delta_coherence",
    "transport_delta_coherence",
    "role_interaction_ratio",
    "transport_interaction_ratio",
    "source_mass_specificity",
    "source_write_coherence",
    "role_source_contrast",
    "lexical_source_contrast",
    "role_write_contrast",
    "lexical_write_contrast",
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
        raise RuntimeError(f"Phase425 non-finite scalar: {value}")
    return round(float(value), 10)


def hash_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.fmean(rows)) if rows else 0.0


def mean_vector(values: Iterable[torch.Tensor]) -> torch.Tensor:
    rows = [value.float() for value in values]
    return torch.stack(rows, dim=0).mean(dim=0)


def cosine_or_zero(left: torch.Tensor, right: torch.Tensor) -> float:
    value = tensor_cosine(left, right)
    return 0.0 if value is None else float(value)


def vector_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    top = float(torch.linalg.vector_norm(numerator.float()).item())
    bottom = float(torch.linalg.vector_norm(denominator.float()).item())
    return clean(top / max(bottom, 1e-8))


def add_transport_diagnostics(
    layers: list[Any],
    captures: dict[tuple[str, int], torch.Tensor],
    row: dict[str, Any],
    scalars: list[dict[str, Any]],
) -> None:
    """Add net, absolute and coherence measurements for the registered source."""
    scalar_by_layer = {int(value["layer"]): value for value in scalars}
    qpos = int(row["prediction_position"])
    source_positions = [int(value) for value in row["source_positions"]]
    for layer_index, layer in enumerate(layers):
        probabilities = captures[("attention_probabilities", layer_index)]
        value = captures[("value", layer_index)]
        head_count = int(probabilities.shape[1])
        repeated_value = repeat_key_value(value, head_count)
        head_dim = int(repeated_value.shape[-1])
        o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
        output_blocks = o_proj.weight.float().view(
            o_proj.weight.shape[0], head_count, head_dim
        )
        writes = []
        for position in source_positions:
            _, _, write = source_writes(
                probabilities,
                repeated_value,
                output_blocks,
                [position],
                qpos,
            )
            writes.append(write.float())
        net_write = torch.stack(writes, dim=0).sum(dim=0)
        net_norm = float(torch.linalg.vector_norm(net_write).item())
        absolute_norm = sum(
            float(torch.linalg.vector_norm(write).item()) for write in writes
        )
        scalar_by_layer[layer_index].update(
            {
                "source_net_write_norm": clean(net_norm),
                "source_absolute_write_norm": clean(absolute_norm),
                "source_write_coherence": clean(
                    net_norm / max(absolute_norm, 1e-8)
                ),
                "source_position_write_norms": [
                    clean(float(torch.linalg.vector_norm(write).item()))
                    for write in writes
                ],
            }
        )
        del repeated_value, output_blocks, writes, net_write


def condition_key(role: str, interface: str, history: str) -> str:
    return f"r{role}__i{interface}__h{history}"


def nuisance_pairs() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for role in ROLES:
        for history in HISTORIES:
            rows.append(
                (
                    condition_key(role, INTERFACES[0], history),
                    condition_key(role, INTERFACES[1], history),
                )
            )
        for interface in INTERFACES:
            rows.append(
                (
                    condition_key(role, interface, HISTORIES[0]),
                    condition_key(role, interface, HISTORIES[1]),
                )
            )
    return rows


def matched_role_pairs() -> list[tuple[str, str]]:
    return [
        (
            condition_key("a", interface, history),
            condition_key("b", interface, history),
        )
        for interface in INTERFACES
        for history in HISTORIES
    ]


def pair_layer_payload(
    entries: dict[str, dict[str, Any]], layer: int
) -> dict[str, Any]:
    role_pairs = matched_role_pairs()
    nuisance = nuisance_pairs()
    role_source = mean(
        normalized_delta(
            entries[left]["tensors"][layer]["source_state"],
            entries[right]["tensors"][layer]["source_state"],
        )
        for left, right in role_pairs
    )
    role_control = mean(
        normalized_delta(
            entries[left]["tensors"][layer]["control_state"],
            entries[right]["tensors"][layer]["control_state"],
        )
        for left, right in role_pairs
    )
    role_write = mean(
        normalized_delta(
            entries[left]["tensors"][layer]["source_write"],
            entries[right]["tensors"][layer]["source_write"],
        )
        for left, right in role_pairs
    )
    role_control_write = mean(
        normalized_delta(
            entries[left]["tensors"][layer]["control_write"],
            entries[right]["tensors"][layer]["control_write"],
        )
        for left, right in role_pairs
    )
    nuisance_source = mean(
        normalized_delta(
            entries[left]["tensors"][layer]["source_state"],
            entries[right]["tensors"][layer]["source_state"],
        )
        for left, right in nuisance
    )
    nuisance_write = mean(
        normalized_delta(
            entries[left]["tensors"][layer]["source_write"],
            entries[right]["tensors"][layer]["source_write"],
        )
        for left, right in nuisance
    )
    role_delta = mean_vector(
        entries[condition_key("a", interface, history)]["tensors"][layer][
            "source_state"
        ]
        for interface in INTERFACES
        for history in HISTORIES
    ) - mean_vector(
        entries[condition_key("b", interface, history)]["tensors"][layer][
            "source_state"
        ]
        for interface in INTERFACES
        for history in HISTORIES
    )
    transport_delta = mean_vector(
        entries[condition_key("a", interface, history)]["tensors"][layer][
            "source_write"
        ]
        for interface in INTERFACES
        for history in HISTORIES
    ) - mean_vector(
        entries[condition_key("b", interface, history)]["tensors"][layer][
            "source_write"
        ]
        for interface in INTERFACES
        for history in HISTORIES
    )
    competition = mean(
        cosine_or_zero(
            entry["tensors"][layer]["source_write"],
            entry["tensors"][layer]["direction"],
        )
        - cosine_or_zero(
            entry["tensors"][layer]["control_write"],
            entry["tensors"][layer]["direction"],
        )
        for entry in entries.values()
    )
    scalar_by_key = {
        key: {int(row["layer"]): row for row in entry["scalars"]}
        for key, entry in entries.items()
    }
    source_mass = mean(
        float(rows[layer]["source_attention_mass_per_token"])
        - float(rows[layer]["control_attention_mass_per_token"])
        for rows in scalar_by_key.values()
    )
    source_write_norm = mean(
        float(torch.linalg.vector_norm(entry["tensors"][layer]["source_write"]).item())
        for entry in entries.values()
    )
    source_write_coherence = mean(
        float(rows[layer]["source_write_coherence"])
        for rows in scalar_by_key.values()
    )
    margins = [
        float(entry["summary"]["final_target_branch_margin"])
        for entry in entries.values()
    ]
    return {
        "role_source_contrast": role_source,
        "matched_position_contrast": role_control,
        "interface_history_source_contrast": nuisance_source,
        "role_write_contrast": role_write,
        "matched_position_write_contrast": role_control_write,
        "interface_history_write_contrast": nuisance_write,
        "prelexical_formation_specificity": clean(
            role_source - max(role_control, nuisance_source)
        ),
        "prelexical_transport_specificity": clean(
            role_write - max(role_control_write, nuisance_write)
        ),
        "competition_specificity": competition,
        "source_mass_specificity": source_mass,
        "source_write_norm": source_write_norm,
        "source_write_coherence": source_write_coherence,
        "role_delta_vector": role_delta,
        "transport_delta_vector": transport_delta,
        "behavior_margin_mean": mean(margins),
        "condition_correct_fraction": clean(sum(value > 0 for value in margins) / len(margins)),
    }


def combine_replica_group(
    model: str,
    pair_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, dict[str, Any]]],
    layer_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(pair_rows) != 2:
        raise RuntimeError("A lexical replica group must contain exactly two pairs")
    pair_rows = sorted(pair_rows, key=lambda row: int(row["lexical_replica"]))
    left_pair, right_pair = pair_rows
    pair_layer_rows: list[dict[str, Any]] = []
    group_layer_rows: list[dict[str, Any]] = []
    group_id = left_pair["replica_group_id"]
    for layer in range(layer_count):
        left = pair_layer_payload(payloads[left_pair["pair_id"]], layer)
        right = pair_layer_payload(payloads[right_pair["pair_id"]], layer)
        left_entries = payloads[left_pair["pair_id"]]
        right_entries = payloads[right_pair["pair_id"]]
        lexical_source = mean(
            normalized_delta(
                left_entries[key]["tensors"][layer]["source_state"],
                right_entries[key]["tensors"][layer]["source_state"],
            )
            for key in left_entries
        )
        lexical_write = mean(
            normalized_delta(
                left_entries[key]["tensors"][layer]["source_write"],
                right_entries[key]["tensors"][layer]["source_write"],
            )
            for key in left_entries
        )
        role_coherence = cosine_or_zero(
            left["role_delta_vector"], right["role_delta_vector"]
        )
        transport_coherence = cosine_or_zero(
            left["transport_delta_vector"], right["transport_delta_vector"]
        )
        role_main = 0.5 * (left["role_delta_vector"] + right["role_delta_vector"])
        role_interaction = 0.5 * (
            left["role_delta_vector"] - right["role_delta_vector"]
        )
        transport_main = 0.5 * (
            left["transport_delta_vector"] + right["transport_delta_vector"]
        )
        transport_interaction = 0.5 * (
            left["transport_delta_vector"] - right["transport_delta_vector"]
        )
        role_interaction_ratio = vector_ratio(role_interaction, role_main)
        transport_interaction_ratio = vector_ratio(
            transport_interaction, transport_main
        )
        pair_feature_rows = []
        for pair, values in ((left_pair, left), (right_pair, right)):
            formation_functional = clean(
                float(values["role_source_contrast"])
                - max(
                    float(values["matched_position_contrast"]),
                    float(values["interface_history_source_contrast"]),
                )
            )
            transport_functional = clean(
                float(values["role_write_contrast"])
                - max(
                    float(values["matched_position_write_contrast"]),
                    float(values["interface_history_write_contrast"]),
                )
            )
            formation_dominance = clean(
                float(values["role_source_contrast"]) - lexical_source
            )
            transport_dominance = clean(
                float(values["role_write_contrast"]) - lexical_write
            )
            formation = min(formation_functional, formation_dominance)
            transport = min(transport_functional, transport_dominance)
            row = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase425-PairRolePath",
                "created_at": now(),
                "model": model,
                "block_id": pair["block_id"],
                "family_id": pair["family_id"],
                "mechanism_id": pair["mechanism_id"],
                "candidate": pair["candidate"],
                "pair_id": pair["pair_id"],
                "pair_index": pair["pair_index"],
                "replica_group_id": group_id,
                "lexical_replica": pair["lexical_replica"],
                "split": pair["split"],
                "layer": layer,
                "relative_depth": clean(layer / max(1, layer_count - 1)),
                "depth_bin": (
                    "early"
                    if layer / max(1, layer_count - 1) < 1 / 3
                    else "middle"
                    if layer / max(1, layer_count - 1) < 2 / 3
                    else "late"
                ),
                "lexical_source_contrast": lexical_source,
                "lexical_write_contrast": lexical_write,
                "role_delta_coherence": role_coherence,
                "transport_delta_coherence": transport_coherence,
                "formation_specificity": formation,
                "transport_specificity": transport,
                "formation_functional_specificity": formation_functional,
                "transport_functional_specificity": transport_functional,
                "formation_role_dominance": formation_dominance,
                "transport_role_dominance": transport_dominance,
                "role_interaction_ratio": role_interaction_ratio,
                "transport_interaction_ratio": transport_interaction_ratio,
                **{
                    key: clean(float(value))
                    for key, value in values.items()
                    if not key.endswith("_vector")
                },
                "physical": True,
                "observer": False,
                "predictive": False,
                "causal": False,
            }
            pair_layer_rows.append(row)
            pair_feature_rows.append(row)
        group_layer_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase425-ReplicaGroupRolePath",
                "created_at": now(),
                "model": model,
                "block_id": left_pair["block_id"],
                "family_id": left_pair["family_id"],
                "mechanism_id": left_pair["mechanism_id"],
                "candidate": left_pair["candidate"],
                "replica_group_id": group_id,
                "split": left_pair["split"],
                "layer": layer,
                "relative_depth": pair_feature_rows[0]["relative_depth"],
                "depth_bin": pair_feature_rows[0]["depth_bin"],
                **{
                    feature: mean(row[feature] for row in pair_feature_rows)
                    for feature in LAYER_FEATURES
                },
                "role_delta_coherence": role_coherence,
                "transport_delta_coherence": transport_coherence,
                "behavior_margin_mean": mean(
                    row["behavior_margin_mean"] for row in pair_feature_rows
                ),
                "condition_correct_fraction": mean(
                    row["condition_correct_fraction"] for row in pair_feature_rows
                ),
                "physical": True,
                "observer": False,
                "predictive": False,
                "causal": False,
            }
        )
    summaries: list[dict[str, Any]] = []
    for pair in pair_rows:
        rows = [row for row in pair_layer_rows if row["pair_id"] == pair["pair_id"]]
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase425-PairRoleSummary",
            "created_at": now(),
            "model": model,
            "block_id": pair["block_id"],
            "family_id": pair["family_id"],
            "mechanism_id": pair["mechanism_id"],
            "candidate": pair["candidate"],
            "pair_id": pair["pair_id"],
            "pair_index": pair["pair_index"],
            "replica_group_id": group_id,
            "lexical_replica": pair["lexical_replica"],
            "split": pair["split"],
            "condition_count": 8,
            "executed_token_count_mean": mean(
                entry["row"]["executed_token_count"]
                for entry in payloads[pair["pair_id"]].values()
            ),
            "source_token_count_mean": mean(
                entry["row"]["source_token_count"]
                for entry in payloads[pair["pair_id"]].values()
            ),
            "query_token_count_mean": mean(
                entry["row"]["query_token_count"]
                for entry in payloads[pair["pair_id"]].values()
            ),
            "behavior_margin_mean": mean(
                entry["summary"]["final_target_branch_margin"]
                for entry in payloads[pair["pair_id"]].values()
            ),
            "condition_correct_fraction": clean(
                sum(
                    entry["summary"]["branch_correct"]
                    for entry in payloads[pair["pair_id"]].values()
                )
                / 8
            ),
            "max_component_ledger_relative_error": max(
                float(entry["summary"]["max_component_ledger_relative_error"])
                for entry in payloads[pair["pair_id"]].values()
            ),
            "numerical_retry_count": sum(
                int(entry["summary"]["numerical_retry_count"])
                for entry in payloads[pair["pair_id"]].values()
            ),
            "physical": True,
            "observer": False,
            "predictive": False,
            "causal": False,
        }
        for depth in ("early", "middle", "late"):
            depth_rows = [row for row in rows if row["depth_bin"] == depth]
            for feature in LAYER_FEATURES:
                summary[f"{depth}_{feature}_median"] = clean(
                    statistics.median(float(row[feature]) for row in depth_rows)
                )
        summaries.append(summary)
    group_summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase425-ReplicaGroupSummary",
        "created_at": now(),
        "model": model,
        "block_id": left_pair["block_id"],
        "family_id": left_pair["family_id"],
        "mechanism_id": left_pair["mechanism_id"],
        "candidate": left_pair["candidate"],
        "replica_group_id": group_id,
        "split": left_pair["split"],
        "pair_count": 2,
        "condition_count": 16,
        "executed_token_count_mean": mean(
            row["executed_token_count_mean"] for row in summaries
        ),
        "source_token_count_mean": mean(
            row["source_token_count_mean"] for row in summaries
        ),
        "query_token_count_mean": mean(
            row["query_token_count_mean"] for row in summaries
        ),
        "behavior_margin_mean": mean(row["behavior_margin_mean"] for row in summaries),
        "condition_correct_fraction": mean(
            row["condition_correct_fraction"] for row in summaries
        ),
        "max_component_ledger_relative_error": max(
            float(row["max_component_ledger_relative_error"]) for row in summaries
        ),
        "physical": True,
        "observer": False,
        "predictive": False,
        "causal": False,
    }
    for depth in ("early", "middle", "late"):
        depth_rows = [row for row in group_layer_rows if row["depth_bin"] == depth]
        for feature in LAYER_FEATURES:
            group_summary[f"{depth}_{feature}_median"] = clean(
                statistics.median(float(row[feature]) for row in depth_rows)
            )
    return pair_layer_rows, group_layer_rows, [*summaries, group_summary]


def run_model(model: str, stage: str) -> dict[str, Any]:
    protocol = read_json(OUT / "phase425_protocol.json")
    if not protocol["validation"]["valid"]:
        raise RuntimeError("Phase425 protocol is invalid")
    if stage == "sealed":
        freeze_path = OUT / "phase425_gate_freeze.json"
        if not freeze_path.exists() or not read_json(freeze_path).get("sealed_unlock"):
            raise RuntimeError("Phase425 sealed split is not unlocked")
        gate_freeze = read_json(freeze_path)
        block_contract = {row["block_id"]: row for row in protocol["blocks"]}
        authorized_blocks = set(gate_freeze["sealed_unlock_blocks"])
        authorized_blocks.update(
            block_contract[block_id]["matched_control_block_id"]
            for block_id in tuple(authorized_blocks)
        )
        condition_file = SEALED_FILE
        split_names = {"sealed_physical_holdout"}
    else:
        authorized_blocks = {row["block_id"] for row in protocol["blocks"]}
        condition_file = OPEN_FILE
        split_names = {"discovery", "calibration", "behavior_holdout"}
    all_conditions = read_jsonl(condition_file)
    conditions = [
        row
        for row in all_conditions
        if row["model"] == model and row["block_id"] in authorized_blocks
    ]
    expected_conditions = len(authorized_blocks) * (192 if stage == "sealed" else 576)
    if len(conditions) != expected_conditions:
        raise RuntimeError(
            f"Expected {expected_conditions} {stage} conditions for {model}, got {len(conditions)}"
        )
    condition_index = {row["condition_id"]: row for row in conditions}
    pairs = [
        row
        for row in read_jsonl(PAIR_FILE)
        if row["split"] in split_names and row["block_id"] in authorized_blocks
    ]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_group[pair["replica_group_id"]].append(pair)
    groups = sorted(by_group)
    expected_groups = len(authorized_blocks) * (12 if stage == "sealed" else 36)
    if len(groups) != expected_groups:
        raise RuntimeError(f"Expected {expected_groups} groups, got {len(groups)}")
    model_root = OUT / "models" / model / stage
    complete_path = model_root / "phase425_collection_complete.json"
    if complete_path.exists():
        existing = read_json(complete_path)
        if existing.get("all_rows_complete"):
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return existing
    checkpoint_root = model_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    expected_dtype = protocol["execution_dtype_by_model"][model]
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    frozen_hash = protocol["implementation_commitments"][Path(__file__).name]
    if implementation_hash != frozen_hash:
        raise RuntimeError("Phase425 collector changed after protocol freeze")
    completed_groups: set[str] = set()
    for path in sorted(checkpoint_root.glob("phase425_group_summaries_*.jsonl")):
        completed_groups.update(row["replica_group_id"] for row in read_jsonl(path))
    loaded = None
    handles: list[Any] = []
    started = time.monotonic()
    shard_condition_rows: list[dict[str, Any]] = []
    shard_pair_layers: list[dict[str, Any]] = []
    shard_group_layers: list[dict[str, Any]] = []
    shard_pair_summaries: list[dict[str, Any]] = []
    shard_group_summaries: list[dict[str, Any]] = []
    shard_start: int | None = None

    def flush(end_index: int) -> None:
        nonlocal shard_start
        if shard_start is None or not shard_group_summaries:
            return
        stem = f"{shard_start:04d}_{end_index:04d}"
        write_jsonl(checkpoint_root / f"phase425_condition_rows_{stem}.jsonl", shard_condition_rows)
        write_jsonl(checkpoint_root / f"phase425_pair_layers_{stem}.jsonl", shard_pair_layers)
        write_jsonl(checkpoint_root / f"phase425_group_layers_{stem}.jsonl", shard_group_layers)
        write_jsonl(checkpoint_root / f"phase425_pair_summaries_{stem}.jsonl", shard_pair_summaries)
        write_jsonl(checkpoint_root / f"phase425_group_summaries_{stem}.jsonl", shard_group_summaries)
        shard_condition_rows.clear()
        shard_pair_layers.clear()
        shard_group_layers.clear()
        shard_pair_summaries.clear()
        shard_group_summaries.clear()
        shard_start = None

    try:
        print(
            f"[Phase425] loading {model}; stage={stage}; groups={len(groups)}; "
            f"checkpointed={len(completed_groups)}",
            flush=True,
        )
        loaded = load_probe_model(model)
        actual_dtype = str(next(loaded.model.parameters()).dtype).removeprefix("torch.")
        if actual_dtype != expected_dtype:
            raise RuntimeError(f"Execution dtype mismatch: {actual_dtype} != {expected_dtype}")
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], torch.Tensor] = {}
        handles = install_hooks(layers, captures)
        with capture_actual_qkv(loaded.key, tuple(range(len(layers))), captures):
            for group_number, group_id in enumerate(groups, start=1):
                if group_id in completed_groups:
                    continue
                if shard_start is None:
                    shard_start = group_number - 1
                group_pairs = sorted(
                    by_group[group_id], key=lambda row: int(row["lexical_replica"])
                )
                payloads: dict[str, dict[str, dict[str, Any]]] = {}
                for pair in group_pairs:
                    entries: dict[str, dict[str, Any]] = {}
                    for role in ROLES:
                        for interface in INTERFACES:
                            for history in HISTORIES:
                                key = condition_key(role, interface, history)
                                row = condition_index[
                                    f"{pair['pair_id']}__{key}__{model}"
                                ]
                                probe_row = {**row, "pair_identity": key}
                                scalars, tensors, summary = collect_condition(
                                    loaded, layers, captures, probe_row
                                )
                                add_transport_diagnostics(
                                    layers, captures, row, scalars
                                )
                                for scalar in scalars:
                                    scalar.update(
                                        {
                                            "schema_version": SCHEMA_VERSION,
                                            "phase_id": PHASE_ID,
                                            "block_id": pair["block_id"],
                                            "replica_group_id": group_id,
                                            "role": role,
                                            "interface": interface,
                                            "history": history,
                                        }
                                    )
                                shard_condition_rows.extend(scalars)
                                entries[key] = {
                                    "row": row,
                                    "scalars": scalars,
                                    "tensors": tensors,
                                    "summary": summary,
                                }
                                captures.clear()
                    payloads[pair["pair_id"]] = entries
                pair_layers, group_layers, summaries = combine_replica_group(
                    model, group_pairs, payloads, len(layers)
                )
                shard_pair_layers.extend(pair_layers)
                shard_group_layers.extend(group_layers)
                shard_pair_summaries.extend(
                    row for row in summaries if row["phase_id"] == "Phase425-PairRoleSummary"
                )
                shard_group_summaries.extend(
                    row for row in summaries if row["phase_id"] == "Phase425-ReplicaGroupSummary"
                )
                del payloads, pair_layers, group_layers, summaries
                captures.clear()
                if len(shard_group_summaries) == 4 or group_number == len(groups):
                    flush(group_number - 1)
                    print(
                        f"[Phase425:{model}:{stage}] groups={group_number}/{len(groups)}",
                        flush=True,
                    )
        condition_rows: list[dict[str, Any]] = []
        pair_layers: list[dict[str, Any]] = []
        group_layers: list[dict[str, Any]] = []
        pair_summaries: list[dict[str, Any]] = []
        group_summaries: list[dict[str, Any]] = []
        for path in sorted(checkpoint_root.glob("phase425_condition_rows_*.jsonl")):
            condition_rows.extend(read_jsonl(path))
        for path in sorted(checkpoint_root.glob("phase425_pair_layers_*.jsonl")):
            pair_layers.extend(read_jsonl(path))
        for path in sorted(checkpoint_root.glob("phase425_group_layers_*.jsonl")):
            group_layers.extend(read_jsonl(path))
        for path in sorted(checkpoint_root.glob("phase425_pair_summaries_*.jsonl")):
            pair_summaries.extend(read_jsonl(path))
        for path in sorted(checkpoint_root.glob("phase425_group_summaries_*.jsonl")):
            group_summaries.extend(read_jsonl(path))
        write_jsonl(model_root / "phase425_condition_layer_rows.jsonl", condition_rows)
        write_jsonl(model_root / "phase425_pair_layer_rows.jsonl", pair_layers)
        write_jsonl(model_root / "phase425_group_layer_rows.jsonl", group_layers)
        write_jsonl(model_root / "phase425_pair_summary_rows.jsonl", pair_summaries)
        write_jsonl(model_root / "phase425_group_summary_rows.jsonl", group_summaries)
        expected_pair_count = expected_groups * 2
        expected_condition_layers = expected_conditions * len(layers)
        expected_pair_layers = expected_pair_count * len(layers)
        expected_group_layers = expected_groups * len(layers)
        max_error = max(
            float(row["max_component_ledger_relative_error"])
            for row in pair_summaries
        )
        complete = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "stage": stage,
            "authorized_blocks": sorted(authorized_blocks),
            "layer_count": len(layers),
            "execution_dtype": actual_dtype,
            "protocol_schema_version": protocol["schema_version"],
            "collector_sha256": implementation_hash,
            "condition_count": expected_conditions,
            "pair_count": len(pair_summaries),
            "replica_group_count": len(group_summaries),
            "condition_layer_row_count": len(condition_rows),
            "pair_layer_row_count": len(pair_layers),
            "group_layer_row_count": len(group_layers),
            "expected_condition_layer_row_count": expected_condition_layers,
            "expected_pair_layer_row_count": expected_pair_layers,
            "expected_group_layer_row_count": expected_group_layers,
            "condition_correct_count": sum(
                round(float(row["condition_correct_fraction"]) * 8)
                for row in pair_summaries
            ),
            "numerical_retry_condition_count": sum(
                int(row["numerical_retry_count"]) for row in pair_summaries
            ),
            "max_component_ledger_relative_error": clean(max_error),
            "component_ledger_gate_pass": max_error <= LEDGER_THRESHOLD,
            "all_rows_complete": bool(
                len(condition_rows) == expected_condition_layers
                and len(pair_layers) == expected_pair_layers
                and len(group_layers) == expected_group_layers
                and len(pair_summaries) == expected_pair_count
                and len(group_summaries) == expected_groups
                and max_error <= LEDGER_THRESHOLD
            ),
            "condition_rows_sha256": hash_rows(condition_rows),
            "pair_layer_rows_sha256": hash_rows(pair_layers),
            "group_layer_rows_sha256": hash_rows(group_layers),
            "pair_summary_rows_sha256": hash_rows(pair_summaries),
            "group_summary_rows_sha256": hash_rows(group_summaries),
            "elapsed_seconds": clean(time.monotonic() - started),
            "vram_gb": list(vram_gb()),
            "pipeline_sealed_stage": stage == "sealed",
            "strict_human_double_blind": False,
            "physical": True,
            "observer_overlay": True,
            "predictive": False,
            "causal": False,
        }
        write_json(complete_path, complete)
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return complete
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--stage", choices=("open", "sealed"), default="open")
    args = parser.parse_args()
    run_model(args.model, args.stage)


if __name__ == "__main__":
    main()
