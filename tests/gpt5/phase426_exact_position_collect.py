#!/usr/bin/env python3
"""Collect Phase426 full-sequence behavior and exact-position physical paths."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
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
from phase371b_anchor_qk_collection import capture_actual_qkv  # noqa: E402
from phase424_global_physical_collect import (  # noqa: E402
    collect_condition,
    normalized_delta,
    tensor_cosine,
)
from phase425_role_exchange_collect import add_transport_diagnostics  # noqa: E402
from phase426_exact_position_protocol import (  # noqa: E402
    HISTORIES,
    INTERFACES,
    MODELS,
    OUT,
    ROLES,
    SCHEMA_VERSION,
    TIMINGS,
)


PHASE_ID = "Phase426-ExactPositionRoleCollection"
LEDGER_THRESHOLD = 0.01
BEHAVIOR_BATCH_SIZE = {"qwen3": 8, "glm4": 2, "deepseek7b": 4}
LAYER_FEATURES = (
    "formation_early_role_distance",
    "formation_late_role_distance",
    "formation_exact_specificity",
    "formation_radial_specificity",
    "formation_angular_specificity",
    "formation_role_covariance",
    "formation_conditional_covariance",
    "formation_replica_signal_ratio",
    "formation_role_dominance",
    "transport_early_role_distance",
    "transport_late_role_distance",
    "transport_exact_specificity",
    "transport_radial_specificity",
    "transport_angular_specificity",
    "transport_role_covariance",
    "transport_conditional_covariance",
    "transport_replica_signal_ratio",
    "transport_role_dominance",
    "source_to_write_identity_alignment",
    "competition_specificity",
    "competition_role_covariance",
    "source_write_coherence",
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
        raise RuntimeError(f"Phase426 non-finite scalar: {value}")
    return round(float(value), 10)


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.fmean(rows)) if rows else 0.0


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def mean_vector(values: Iterable[torch.Tensor]) -> torch.Tensor:
    rows = [value.float() for value in values]
    if not rows:
        raise RuntimeError("Cannot average an empty vector list")
    return torch.stack(rows, dim=0).mean(dim=0)


def cosine_or_zero(left: torch.Tensor, right: torch.Tensor) -> float:
    value = tensor_cosine(left, right)
    return 0.0 if value is None else float(value)


def pairwise_cosine(vectors: list[torch.Tensor]) -> float:
    values = [
        cosine_or_zero(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    return mean(values)


def vector_signal_ratio(left: torch.Tensor, right: torch.Tensor) -> float:
    signal = 0.5 * (left.float() + right.float())
    disagreement = 0.5 * (left.float() - right.float())
    numerator = float(torch.linalg.vector_norm(signal).item())
    denominator = float(torch.linalg.vector_norm(disagreement).item())
    return clean(numerator / max(denominator, 1e-8))


def radial_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(torch.linalg.vector_norm(left.float()).item())
    right_norm = float(torch.linalg.vector_norm(right.float()).item())
    return clean(abs(left_norm - right_norm) / max(0.5 * (left_norm + right_norm), 1e-8))


def angular_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return clean(1.0 - cosine_or_zero(left, right))


def condition_key(role: str, interface: str, history: str, timing: str) -> str:
    return f"r{role}__i{interface}__h{history}__t{timing}"


def depth_bin(layer: int, layer_count: int) -> str:
    relative = layer / max(1, layer_count - 1)
    if relative < 1 / 3:
        return "early"
    if relative < 2 / 3:
        return "middle"
    return "late"


def prompt_ids(loaded: Any, row: dict[str, Any]) -> list[int]:
    ids = [
        int(value)
        for value in loaded.tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
    ]
    if len(ids) != int(row["base_prompt_token_count"]):
        raise RuntimeError(f"Prompt token contract changed: {row['condition_id']}")
    return ids


def padded_batch(
    sequences: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    width = max(len(row) for row in sequences)
    input_ids = torch.full(
        (len(sequences), width), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    pads: list[int] = []
    for index, sequence in enumerate(sequences):
        pad = width - len(sequence)
        pads.append(pad)
        input_ids[index, pad:] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, pad:] = 1
    return input_ids, attention_mask, pads


def sequence_scores(
    loaded: Any,
    rows: list[dict[str, Any]],
    variant: str,
) -> list[tuple[float, float]]:
    ids_key = (
        "target_sequence_token_ids"
        if variant == "target"
        else "opposite_sequence_token_ids"
    )
    prompts = [prompt_ids(loaded, row) for row in rows]
    continuations = [[int(value) for value in row[ids_key]] for row in rows]
    sequences = [prompt + continuation for prompt, continuation in zip(prompts, continuations)]
    pad_id = int(loaded.tokenizer.pad_token_id)
    input_ids, attention_mask, pads = padded_batch(
        sequences, pad_id, loaded.input_device
    )
    with torch.inference_mode():
        result = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    logits = result.logits.float()
    output: list[tuple[float, float]] = []
    for batch_index, (prompt, continuation, pad) in enumerate(
        zip(prompts, continuations, pads)
    ):
        values: list[float] = []
        start = pad + len(prompt)
        for offset, token_id in enumerate(continuation):
            log_probs = torch.log_softmax(
                logits[batch_index, start + offset - 1], dim=-1
            )
            values.append(float(log_probs[token_id].item()))
        output.append((clean(sum(values)), clean(statistics.fmean(values))))
    del result, logits, input_ids, attention_mask
    return output


def parse_generation(
    text: str,
    generated_ids: list[int],
    row: dict[str, Any],
    eos_ids: set[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    lowered = text.strip().lower()
    target = str(row["target"]).lower()
    opposite = str(row["opposite_target"]).lower()
    target_at = lowered.find(target)
    opposite_at = lowered.find(opposite)
    target_seen = target_at >= 0
    opposite_seen = opposite_at >= 0
    target_first = target_seen and (not opposite_seen or target_at < opposite_at)
    opposite_first = opposite_seen and (not target_seen or opposite_at < target_at)
    first_atom = re.split(r"[\s,.;:!?\n]+", lowered.lstrip(" \t\"'`([{<"), maxsplit=1)[0]
    exact_target_first = first_atom == target
    eos_seen = any(token_id in eos_ids for token_id in generated_ids)
    boundary = bool(re.search(r"[.!?;\n]", text)) or eos_seen
    stop = eos_seen or len(generated_ids) < max_new_tokens
    return {
        "natural_text": text,
        "natural_generated_token_ids": generated_ids,
        "natural_generated_token_count": len(generated_ids),
        "natural_target_seen": target_seen,
        "natural_opposite_seen": opposite_seen,
        "natural_target_first": target_first,
        "natural_opposite_first": opposite_first,
        "natural_exact_target_first": exact_target_first,
        "natural_revision": target_seen and opposite_seen,
        "natural_boundary": boundary,
        "natural_stop": stop,
        "natural_censoring": len(generated_ids) >= max_new_tokens and not eos_seen,
    }


def natural_scores(
    loaded: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prompts = [prompt_ids(loaded, row) for row in rows]
    pad_id = int(loaded.tokenizer.pad_token_id)
    input_ids, attention_mask, _ = padded_batch(prompts, pad_id, loaded.input_device)
    eos_value = loaded.tokenizer.eos_token_id
    eos_ids = (
        {int(value) for value in eos_value}
        if isinstance(eos_value, list)
        else ({int(eos_value)} if eos_value is not None else set())
    )
    with torch.inference_mode():
        generated = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_value,
            use_cache=True,
        )
    prefix_width = int(input_ids.shape[1])
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        new_ids = [int(value) for value in generated[index, prefix_width:].tolist()]
        while new_ids and new_ids[-1] == pad_id and pad_id not in eos_ids:
            new_ids.pop()
        text = loaded.tokenizer.decode(new_ids, skip_special_tokens=True)
        output.append(
            parse_generation(text, new_ids, row, eos_ids, max_new_tokens)
        )
    del generated, input_ids, attention_mask
    return output


def collect_behavior(
    loaded: Any,
    rows: list[dict[str, Any]],
    output_path: Path,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if output_path.exists():
        existing = read_jsonl(output_path)
        if len(existing) == len(rows):
            return existing
    batch_size = BEHAVIOR_BATCH_SIZE[loaded.key]
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        target_scores = sequence_scores(loaded, batch, "target")
        opposite_scores = sequence_scores(loaded, batch, "opposite")
        natural = natural_scores(loaded, batch, max_new_tokens)
        for row, target_score, opposite_score, natural_row in zip(
            batch, target_scores, opposite_scores, natural
        ):
            target_sum, target_mean = target_score
            opposite_sum, opposite_mean = opposite_score
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase426-FullEventBehavior",
                    "created_at": now(),
                    "model": loaded.key,
                    "condition_id": row["condition_id"],
                    "replica_group_id": row["replica_group_id"],
                    "block_id": row["block_id"],
                    "candidate": row["candidate"],
                    "split": row["split"],
                    "role": row["role"],
                    "interface": row["interface"],
                    "history": row["history"],
                    "timing": row["timing"],
                    "target_sequence_logprob": target_sum,
                    "opposite_sequence_logprob": opposite_sum,
                    "teacher_sequence_logprob_margin": clean(
                        target_sum - opposite_sum
                    ),
                    "target_mean_token_logprob": target_mean,
                    "opposite_mean_token_logprob": opposite_mean,
                    "teacher_mean_token_logprob_margin": clean(
                        target_mean - opposite_mean
                    ),
                    "teacher_sequence_correct": target_sum > opposite_sum,
                    "target_sequence_token_count": len(
                        row["target_sequence_token_ids"]
                    ),
                    "opposite_sequence_token_count": len(
                        row["opposite_sequence_token_ids"]
                    ),
                    **natural_row,
                    "physical": False,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                }
            )
        if start == 0 or (start // batch_size + 1) % 64 == 0:
            print(
                f"[Phase426] {loaded.key} behavior {min(start + len(batch), len(rows))}/{len(rows)}",
                flush=True,
            )
    write_jsonl(output_path, output)
    return output


def role_vectors(
    entries: dict[str, dict[str, Any]],
    layer: int,
    tensor_name: str,
    timing: str,
) -> list[torch.Tensor]:
    return [
        entries[condition_key("a", interface, history, timing)]["tensors"][layer][
            tensor_name
        ]
        - entries[condition_key("b", interface, history, timing)]["tensors"][layer][
            tensor_name
        ]
        for interface in INTERFACES
        for history in HISTORIES
    ]


def role_distances(
    entries: dict[str, dict[str, Any]],
    layer: int,
    tensor_name: str,
    timing: str,
) -> list[float]:
    return [
        normalized_delta(
            entries[condition_key("a", interface, history, timing)]["tensors"][layer][
                tensor_name
            ],
            entries[condition_key("b", interface, history, timing)]["tensors"][layer][
                tensor_name
            ],
        )
        for interface in INTERFACES
        for history in HISTORIES
    ]


def role_radial_angular(
    entries: dict[str, dict[str, Any]],
    layer: int,
    tensor_name: str,
    timing: str,
) -> tuple[float, float]:
    radial: list[float] = []
    angular: list[float] = []
    for interface in INTERFACES:
        for history in HISTORIES:
            left = entries[condition_key("a", interface, history, timing)]["tensors"][
                layer
            ][tensor_name]
            right = entries[condition_key("b", interface, history, timing)]["tensors"][
                layer
            ][tensor_name]
            radial.append(radial_distance(left, right))
            angular.append(angular_distance(left, right))
    return mean(radial), mean(angular)


def lexical_distance(
    replicas: list[dict[str, dict[str, Any]]], layer: int, tensor_name: str
) -> float:
    values = []
    for role in ROLES:
        for interface in INTERFACES:
            for history in HISTORIES:
                for timing in TIMINGS:
                    key = condition_key(role, interface, history, timing)
                    values.append(
                        normalized_delta(
                            replicas[0][key]["tensors"][layer][tensor_name],
                            replicas[1][key]["tensors"][layer][tensor_name],
                        )
                    )
    return mean(values)


def group_layer_payload(
    replicas: list[dict[str, dict[str, Any]]], layer: int, layer_count: int
) -> dict[str, Any]:
    formation_early = [
        role_vectors(replica, layer, "source_state", "early_role")
        for replica in replicas
    ]
    formation_late = [
        role_vectors(replica, layer, "source_state", "late_role")
        for replica in replicas
    ]
    transport_early = [
        role_vectors(replica, layer, "source_write", "early_role")
        for replica in replicas
    ]
    transport_late = [
        role_vectors(replica, layer, "source_write", "late_role")
        for replica in replicas
    ]
    competition_early = [
        role_vectors(replica, layer, "query_state", "early_role")
        for replica in replicas
    ]
    competition_late = [
        role_vectors(replica, layer, "query_state", "late_role")
        for replica in replicas
    ]

    formation_exact = [
        [early - late for early, late in zip(early_rows, late_rows)]
        for early_rows, late_rows in zip(formation_early, formation_late)
    ]
    transport_exact = [
        [early - late for early, late in zip(early_rows, late_rows)]
        for early_rows, late_rows in zip(transport_early, transport_late)
    ]
    competition_exact = [
        [early - late for early, late in zip(early_rows, late_rows)]
        for early_rows, late_rows in zip(competition_early, competition_late)
    ]
    formation_replica_vectors = [mean_vector(rows) for rows in formation_exact]
    transport_replica_vectors = [mean_vector(rows) for rows in transport_exact]
    competition_replica_vectors = [mean_vector(rows) for rows in competition_exact]
    formation_context_vectors = [
        mean_vector([formation_exact[0][index], formation_exact[1][index]])
        for index in range(4)
    ]
    transport_context_vectors = [
        mean_vector([transport_exact[0][index], transport_exact[1][index]])
        for index in range(4)
    ]

    formation_early_distance = mean(
        value
        for replica in replicas
        for value in role_distances(replica, layer, "source_state", "early_role")
    )
    formation_late_distance = mean(
        value
        for replica in replicas
        for value in role_distances(replica, layer, "source_state", "late_role")
    )
    transport_early_distance = mean(
        value
        for replica in replicas
        for value in role_distances(replica, layer, "source_write", "early_role")
    )
    transport_late_distance = mean(
        value
        for replica in replicas
        for value in role_distances(replica, layer, "source_write", "late_role")
    )
    competition_early_distance = mean(
        value
        for replica in replicas
        for value in role_distances(replica, layer, "query_state", "early_role")
    )
    competition_late_distance = mean(
        value
        for replica in replicas
        for value in role_distances(replica, layer, "query_state", "late_role")
    )
    formation_early_radial, formation_early_angular = zip(
        *[
            role_radial_angular(replica, layer, "source_state", "early_role")
            for replica in replicas
        ]
    )
    formation_late_radial, formation_late_angular = zip(
        *[
            role_radial_angular(replica, layer, "source_state", "late_role")
            for replica in replicas
        ]
    )
    transport_early_radial, transport_early_angular = zip(
        *[
            role_radial_angular(replica, layer, "source_write", "early_role")
            for replica in replicas
        ]
    )
    transport_late_radial, transport_late_angular = zip(
        *[
            role_radial_angular(replica, layer, "source_write", "late_role")
            for replica in replicas
        ]
    )
    lexical_state = lexical_distance(replicas, layer, "source_state")
    lexical_write = lexical_distance(replicas, layer, "source_write")
    coherence = mean(
        scalar["source_write_coherence"]
        for replica in replicas
        for entry in replica.values()
        for scalar in entry["scalars"]
        if int(scalar["layer"]) == layer
    )
    formation_main = mean_vector(formation_replica_vectors)
    transport_main = mean_vector(transport_replica_vectors)
    return {
        "layer": layer,
        "relative_depth": clean(layer / max(1, layer_count - 1)),
        "depth_bin": depth_bin(layer, layer_count),
        "formation_early_role_distance": formation_early_distance,
        "formation_late_role_distance": formation_late_distance,
        "formation_exact_specificity": clean(
            formation_early_distance - formation_late_distance
        ),
        "formation_radial_specificity": clean(
            mean(formation_early_radial) - mean(formation_late_radial)
        ),
        "formation_angular_specificity": clean(
            mean(formation_early_angular) - mean(formation_late_angular)
        ),
        "formation_role_covariance": clean(
            cosine_or_zero(formation_replica_vectors[0], formation_replica_vectors[1])
        ),
        "formation_conditional_covariance": pairwise_cosine(
            formation_context_vectors
        ),
        "formation_replica_signal_ratio": vector_signal_ratio(
            formation_replica_vectors[0], formation_replica_vectors[1]
        ),
        "formation_role_dominance": clean(
            formation_early_distance / max(lexical_state, 1e-8)
        ),
        "transport_early_role_distance": transport_early_distance,
        "transport_late_role_distance": transport_late_distance,
        "transport_exact_specificity": clean(
            transport_early_distance - transport_late_distance
        ),
        "transport_radial_specificity": clean(
            mean(transport_early_radial) - mean(transport_late_radial)
        ),
        "transport_angular_specificity": clean(
            mean(transport_early_angular) - mean(transport_late_angular)
        ),
        "transport_role_covariance": clean(
            cosine_or_zero(transport_replica_vectors[0], transport_replica_vectors[1])
        ),
        "transport_conditional_covariance": pairwise_cosine(
            transport_context_vectors
        ),
        "transport_replica_signal_ratio": vector_signal_ratio(
            transport_replica_vectors[0], transport_replica_vectors[1]
        ),
        "transport_role_dominance": clean(
            transport_early_distance / max(lexical_write, 1e-8)
        ),
        "source_to_write_identity_alignment": clean(
            cosine_or_zero(formation_main, transport_main)
        ),
        "competition_specificity": clean(
            competition_early_distance - competition_late_distance
        ),
        "competition_role_covariance": clean(
            cosine_or_zero(
                competition_replica_vectors[0], competition_replica_vectors[1]
            )
        ),
        "source_write_coherence": coherence,
    }


def summarize_group(
    model: str,
    pair_rows: list[dict[str, Any]],
    replicas: list[dict[str, dict[str, Any]]],
    layer_rows: list[dict[str, Any]],
    behavior_by_condition: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    conditions = [entry for replica in replicas for entry in replica.values()]
    behavior = [
        behavior_by_condition[entry["row"]["condition_id"]] for entry in conditions
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase426-ReplicaGroupSummary",
        "created_at": now(),
        "model": model,
        "block_id": pair_rows[0]["block_id"],
        "family_id": pair_rows[0]["family_id"],
        "mechanism_id": pair_rows[0]["mechanism_id"],
        "candidate": pair_rows[0]["candidate"],
        "matched_control_block_id": pair_rows[0]["matched_control_block_id"],
        "replica_group_id": pair_rows[0]["replica_group_id"],
        "split": pair_rows[0]["split"],
        "instrument": pair_rows[0]["instrument"],
        "pair_count": 2,
        "condition_count": len(conditions),
        "executed_token_count_mean": mean(
            entry["summary"]["executed_token_count"] for entry in conditions
        ),
        "target_sequence_token_count_mean": mean(
            row["target_sequence_token_count"] for row in behavior
        ),
        "teacher_sequence_margin_mean": mean(
            row["teacher_sequence_logprob_margin"] for row in behavior
        ),
        "teacher_sequence_correct_fraction": mean(
            float(row["teacher_sequence_correct"]) for row in behavior
        ),
        "natural_target_fraction": mean(
            float(row["natural_exact_target_first"]) for row in behavior
        ),
        "natural_revision_fraction": mean(
            float(row["natural_revision"]) for row in behavior
        ),
        "natural_boundary_fraction": mean(
            float(row["natural_boundary"]) for row in behavior
        ),
        "natural_stop_fraction": mean(float(row["natural_stop"]) for row in behavior),
        "natural_censoring_fraction": mean(
            float(row["natural_censoring"]) for row in behavior
        ),
        "max_component_ledger_relative_error": max(
            float(entry["summary"]["max_component_ledger_relative_error"])
            for entry in conditions
        ),
        "numerical_retry_count": sum(
            int(entry["summary"]["numerical_retry_count"]) for entry in conditions
        ),
        "exact_position_contract_pass": True,
        "physical": True,
        "observer": True,
        "predictive": False,
        "causal": False,
    }
    for timing in TIMINGS:
        timing_rows = [row for row in behavior if row["timing"] == timing]
        summary[f"{timing}_teacher_sequence_margin_mean"] = mean(
            row["teacher_sequence_logprob_margin"] for row in timing_rows
        )
        summary[f"{timing}_teacher_sequence_correct_fraction"] = mean(
            float(row["teacher_sequence_correct"]) for row in timing_rows
        )
        summary[f"{timing}_natural_target_fraction"] = mean(
            float(row["natural_exact_target_first"]) for row in timing_rows
        )
    for depth in ("early", "middle", "late"):
        rows = [row for row in layer_rows if row["depth_bin"] == depth]
        for feature in LAYER_FEATURES:
            summary[f"{depth}_{feature}_median"] = median(
                row[feature] for row in rows
            )
    return summary


def stage_contract(stage: str, protocol: dict[str, Any]) -> tuple[Path, Path, set[str]]:
    if stage == "instrument":
        return (
            OUT / "phase426_instrument_conditions.jsonl",
            OUT / "phase426_instrument_pairs.jsonl",
            {row["block_id"] for row in protocol["blocks"]},
        )
    if stage == "open":
        return (
            OUT / "phase426_registered_conditions_open.jsonl",
            OUT / "phase426_registered_pairs.jsonl",
            {row["block_id"] for row in protocol["blocks"]},
        )
    gate_path = OUT / "phase426_gate_freeze.json"
    if not gate_path.exists() or not read_json(gate_path).get("sealed_unlock"):
        raise RuntimeError("Phase426 sealed split is not unlocked")
    gate = read_json(gate_path)
    authorized = set(gate["sealed_unlock_blocks"])
    contracts = {row["block_id"]: row for row in protocol["blocks"]}
    authorized.update(contracts[value]["matched_control_block_id"] for value in tuple(authorized))
    return (
        OUT / "sealed" / "phase426_registered_conditions_sealed.jsonl",
        OUT / "phase426_registered_pairs.jsonl",
        authorized,
    )


def run_model(model: str, stage: str) -> dict[str, Any]:
    protocol = read_json(OUT / "phase426_protocol.json")
    if not protocol["validation"]["valid"]:
        raise RuntimeError("Phase426 protocol is invalid")
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if implementation_hash != protocol["implementation_commitments"][Path(__file__).name]:
        raise RuntimeError("Phase426 collector changed after protocol freeze")
    condition_path, pair_path, authorized_blocks = stage_contract(stage, protocol)
    all_conditions = read_jsonl(condition_path)
    conditions = [
        row
        for row in all_conditions
        if row["model"] == model and row["block_id"] in authorized_blocks
    ]
    pairs = [
        row
        for row in read_jsonl(pair_path)
        if row["block_id"] in authorized_blocks
        and (
            (stage == "instrument" and row["instrument"])
            or (stage == "open" and row["split"] in {"discovery", "calibration", "behavior_holdout"})
            or (stage == "sealed" and row["split"] == "sealed_physical_holdout")
        )
    ]
    condition_index = {row["condition_id"]: row for row in conditions}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_group[pair["replica_group_id"]].append(pair)
    groups = sorted(by_group)
    expected_groups_per_block = 2 if stage == "instrument" else (96 if stage == "open" else 32)
    expected_groups = len(authorized_blocks) * expected_groups_per_block
    expected_conditions = expected_groups * 32
    if len(groups) != expected_groups or len(conditions) != expected_conditions:
        raise RuntimeError(
            f"Phase426 {stage}/{model}: expected {expected_groups} groups and "
            f"{expected_conditions} conditions, got {len(groups)} and {len(conditions)}"
        )
    model_root = OUT / "models" / model / stage
    complete_path = model_root / "phase426_collection_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        existing = read_json(complete_path)
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return existing
    model_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = model_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    behavior_path = model_root / "phase426_behavior_rows.jsonl"
    expected_dtype = protocol["execution_dtype_by_model"][model]
    loaded = None
    handles: list[Any] = []
    started = time.monotonic()
    try:
        print(
            f"[Phase426] loading {model}; stage={stage}; groups={len(groups)}; "
            f"conditions={len(conditions)}",
            flush=True,
        )
        loaded = load_probe_model(model)
        actual_dtype = str(next(loaded.model.parameters()).dtype).removeprefix("torch.")
        if actual_dtype != expected_dtype:
            raise RuntimeError(f"Execution dtype mismatch: {actual_dtype} != {expected_dtype}")
        behavior = collect_behavior(
            loaded,
            conditions,
            behavior_path,
            int(protocol["registered_thresholds"]["natural_generation_max_new_tokens"]),
        )
        behavior_by_condition = {row["condition_id"]: row for row in behavior}
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], torch.Tensor] = {}
        handles = install_hooks(layers, captures)
        completed_groups: set[str] = set()
        for path in sorted(checkpoint_root.glob("phase426_group_summaries_*.jsonl")):
            completed_groups.update(row["replica_group_id"] for row in read_jsonl(path))
        shard_layers: list[dict[str, Any]] = []
        shard_summaries: list[dict[str, Any]] = []
        shard_start: int | None = None

        def flush(end_index: int) -> None:
            nonlocal shard_start
            if shard_start is None or not shard_summaries:
                return
            stem = f"{shard_start:04d}_{end_index:04d}"
            write_jsonl(
                checkpoint_root / f"phase426_group_layers_{stem}.jsonl", shard_layers
            )
            write_jsonl(
                checkpoint_root / f"phase426_group_summaries_{stem}.jsonl",
                shard_summaries,
            )
            shard_layers.clear()
            shard_summaries.clear()
            shard_start = None

        with capture_actual_qkv(loaded.key, tuple(range(len(layers))), captures):
            for number, group_id in enumerate(groups, start=1):
                if group_id in completed_groups:
                    continue
                if shard_start is None:
                    shard_start = number - 1
                group_pairs = sorted(
                    by_group[group_id], key=lambda row: int(row["lexical_replica"])
                )
                replicas: list[dict[str, dict[str, Any]]] = []
                for pair in group_pairs:
                    entries: dict[str, dict[str, Any]] = {}
                    for role in ROLES:
                        for interface in INTERFACES:
                            for history in HISTORIES:
                                for timing in TIMINGS:
                                    key = condition_key(role, interface, history, timing)
                                    row = condition_index[
                                        f"{pair['pair_id']}__{key}__{model}"
                                    ]
                                    scalars, tensors, summary = collect_condition(
                                        loaded,
                                        layers,
                                        captures,
                                        {**row, "pair_identity": key},
                                    )
                                    add_transport_diagnostics(
                                        layers, captures, row, scalars
                                    )
                                    trimmed = {
                                        layer: {
                                            name: payload[name].float()
                                            for name in (
                                                "source_state",
                                                "source_write",
                                                "query_state",
                                            )
                                        }
                                        for layer, payload in tensors.items()
                                    }
                                    entries[key] = {
                                        "row": row,
                                        "scalars": [
                                            {
                                                "layer": value["layer"],
                                                "source_write_coherence": value[
                                                    "source_write_coherence"
                                                ],
                                            }
                                            for value in scalars
                                        ],
                                        "tensors": trimmed,
                                        "summary": summary,
                                    }
                                    del scalars, tensors, trimmed
                                    captures.clear()
                    replicas.append(entries)
                group_layers = []
                for layer in range(len(layers)):
                    payload = group_layer_payload(replicas, layer, len(layers))
                    group_layers.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase426-ReplicaGroupLayer",
                            "created_at": now(),
                            "model": model,
                            "block_id": group_pairs[0]["block_id"],
                            "family_id": group_pairs[0]["family_id"],
                            "mechanism_id": group_pairs[0]["mechanism_id"],
                            "candidate": group_pairs[0]["candidate"],
                            "matched_control_block_id": group_pairs[0][
                                "matched_control_block_id"
                            ],
                            "replica_group_id": group_id,
                            "split": group_pairs[0]["split"],
                            "instrument": group_pairs[0]["instrument"],
                            **payload,
                            "physical": True,
                            "observer": True,
                            "predictive": False,
                            "causal": False,
                        }
                    )
                summary = summarize_group(
                    model,
                    group_pairs,
                    replicas,
                    group_layers,
                    behavior_by_condition,
                )
                shard_layers.extend(group_layers)
                shard_summaries.append(summary)
                del replicas, group_layers, summary
                captures.clear()
                if len(shard_summaries) >= 2 or number == len(groups):
                    flush(number - 1)
                if number == 1 or number % 8 == 0:
                    allocated_gb, reserved_gb = vram_gb()
                    print(
                        f"[Phase426] {model} {stage} physical {number}/{len(groups)}; "
                        f"VRAM={allocated_gb:.2f}/{reserved_gb:.2f} GiB allocated/reserved",
                        flush=True,
                    )
        flush(len(groups) - 1)
        layer_paths = sorted(checkpoint_root.glob("phase426_group_layers_*.jsonl"))
        summary_paths = sorted(checkpoint_root.glob("phase426_group_summaries_*.jsonl"))
        layer_rows = [row for path in layer_paths for row in read_jsonl(path)]
        group_summaries = [row for path in summary_paths for row in read_jsonl(path)]
        unique_summaries = {row["replica_group_id"]: row for row in group_summaries}
        unique_layers = {
            (row["replica_group_id"], int(row["layer"])): row for row in layer_rows
        }
        group_summaries = [unique_summaries[key] for key in sorted(unique_summaries)]
        layer_rows = [unique_layers[key] for key in sorted(unique_layers)]
        write_jsonl(model_root / "phase426_group_layer_rows.jsonl", layer_rows)
        write_jsonl(model_root / "phase426_group_summary_rows.jsonl", group_summaries)
        max_ledger = max(
            float(row["max_component_ledger_relative_error"])
            for row in group_summaries
        )
        complete = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "stage": stage,
            "execution_dtype": actual_dtype,
            "layer_count": len(layers),
            "condition_count": len(conditions),
            "behavior_row_count": len(behavior),
            "replica_group_count": len(group_summaries),
            "group_layer_row_count": len(layer_rows),
            "max_component_ledger_relative_error": clean(max_ledger),
            "component_ledger_gate_pass": max_ledger <= LEDGER_THRESHOLD,
            "numerical_retry_count": sum(
                int(row["numerical_retry_count"]) for row in group_summaries
            ),
            "elapsed_seconds": clean(time.monotonic() - started),
            "all_rows_complete": bool(
                len(behavior) == len(conditions)
                and len(group_summaries) == len(groups)
                and len(layer_rows) == len(groups) * len(layers)
            ),
            "sealed_read": stage == "sealed",
            "causal_tested": False,
        }
        if not complete["all_rows_complete"] or not complete[
            "component_ledger_gate_pass"
        ]:
            raise RuntimeError(json.dumps(complete, ensure_ascii=False, indent=2))
        write_json(complete_path, complete)
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return complete
    finally:
        for handle in handles:
            handle.remove()
        if loaded is not None:
            release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("instrument", "open", "sealed"), required=True)
    args = parser.parse_args()
    run_model(args.model, args.stage)


if __name__ == "__main__":
    main()
