#!/usr/bin/env python3
"""Run the Phase1065 FP16 cross-pattern behavior and response atlas."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1040_expanded_mlp_replication_protocol as material
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1058_multitoken_translation_scan as generation
import phase1062_text_equivalence_scan as text_tools
import phase1065_multimode_response_atlas_protocol as protocol


UNIT_BATCH_SIZE = {
    "qwen3": 2,
    "glm4": 1,
    "deepseek7b": 1,
}
EPSILON = 1e-12


def event_definitions(n_layers: int) -> list[dict[str, Any]]:
    events = [{
        "event_index": 0,
        "event_id": "residual.d00",
        "component": "residual",
        "depth": 0,
        "relative_depth": 0.0,
    }]
    for depth in range(1, n_layers + 1):
        for component in ("residual", "attention_output", "mlp_output"):
            events.append({
                "event_index": len(events),
                "event_id": f"{component}.d{depth:02d}",
                "component": component,
                "depth": depth,
                "relative_depth": depth / n_layers,
            })
    return events


class RoleCapture:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.positions: torch.Tensor | None = None
        self.values: dict[tuple[str, int], torch.Tensor] = {}
        self.counts: Counter = Counter()
        self.handles = []

    def _hook(self, component: str, depth: int):
        key = (component, depth)

        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(value, torch.Tensor):
                raise RuntimeError("capture was not initialized")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            self.values[key] = value[batch, positions, :].detach()
            self.counts[key] += 1
            return output

        return hook

    def register(self) -> None:
        self.handles.append(
            self.model.get_input_embeddings().register_forward_hook(
                self._hook("residual", 0)
            )
        )
        for depth, layer in enumerate(self.layers, 1):
            self.handles.append(
                layer.register_forward_hook(
                    self._hook("residual", depth)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._hook("attention_output", depth)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._hook("mlp_output", depth)
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.counts = Counter()

    def validate(self) -> None:
        expected = {("residual", 0)}
        for depth in range(1, len(self.layers) + 1):
            expected.update({
                ("residual", depth),
                ("attention_output", depth),
                ("mlp_output", depth),
            })
        missing = expected - set(self.values)
        repeated = {
            str(key): count
            for key, count in self.counts.items()
            if count != 1
        }
        if missing or repeated:
            raise RuntimeError(
                f"capture drift missing={list(missing)[:5]} "
                f"repeated={repeated}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


def pad_rows(
    rows: list[dict[str, Any]],
    pad_id: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    positions = torch.zeros(
        (len(rows), len(protocol.CAPTURE_ROLES)),
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        values = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        )
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
        positions[index] = torch.tensor([
            int(row["role_positions"][role])
            for role in protocol.CAPTURE_ROLES
        ], dtype=torch.long, device=device)
    return input_ids, attention_mask, lengths, positions


def pairwise_direction_consistency(
    direction_sum: np.ndarray,
    count: int,
) -> float | None:
    if count < 2:
        return None
    squared = float(np.dot(
        direction_sum.astype(np.float64, copy=False),
        direction_sum.astype(np.float64, copy=False),
    ))
    return (squared - count) / (count * (count - 1))


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denominator = (
        torch.linalg.vector_norm(a, dim=-1)
        * torch.linalg.vector_norm(b, dim=-1)
    )
    result = torch.zeros_like(denominator, dtype=torch.float32)
    valid = denominator > EPSILON
    result[valid] = (
        (a[valid] * b[valid]).sum(dim=-1) / denominator[valid]
    )
    return result


def natural_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    per_family = protocol.NATURAL_AUDIT_CASES_PER_FAMILY
    for family in protocol.FAMILIES:
        eligible = [
            row for row in rows
            if row["family"] == family
            and row["state"] in {"b0_l0", "b1_l1"}
        ]
        selected.extend(
            generation.evenly_spaced(eligible, per_family)
        )
    return selected


def strict_generated_answer(
    tokenizer,
    output_ids: list[int],
    eos_ids: set[int],
) -> str:
    return text_tools.decode_content(tokenizer, output_ids, eos_ids)


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1065 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[str(row["unit_id"])].append(row)
    units = []
    for unit_id, values in sorted(by_unit.items()):
        by_state = {str(row["state"]): row for row in values}
        if set(by_state) != set(protocol.STATES):
            raise RuntimeError(f"incomplete unit {unit_id}")
        units.append({
            "unit_id": unit_id,
            "family": values[0]["family"],
            "split": values[0]["split"],
            "states": by_state,
        })

    started = time.time()
    model = tokenizer = capture = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        events = event_definitions(len(layers))
        event_keys = [
            (str(row["component"]), int(row["depth"]))
            for row in events
        ]
        d_model = int(model.get_input_embeddings().weight.shape[1])
        family_index = {
            family: index
            for index, family in enumerate(protocol.FAMILIES)
        }
        split_index = {
            split: index for index, split in enumerate(protocol.SPLITS)
        }
        role_index = {
            role: index
            for index, role in enumerate(protocol.CAPTURE_ROLES)
        }
        shape = (
            len(protocol.FAMILIES),
            len(protocol.SPLITS),
            len(events),
            len(protocol.CAPTURE_ROLES),
        )
        direction_sum = np.zeros((*shape, d_model), dtype=np.float32)
        semantic_count = np.zeros(shape, dtype=np.int32)
        semantic_relative_magnitude_sum = np.zeros(shape, dtype=np.float64)
        surface_relative_magnitude_sum = np.zeros(shape, dtype=np.float64)
        surface_count = np.zeros(shape, dtype=np.int32)
        branch_cosine_sum = np.zeros(shape, dtype=np.float64)
        branch_cosine_count = np.zeros(shape, dtype=np.int32)
        interaction_relative_sum = np.zeros(shape, dtype=np.float64)
        identity_maximum = 0.0
        behavior_records = []
        case_hit: dict[int, bool] = {}
        candidate_margin: dict[int, float] = {}
        valid_pairs = Counter()
        valid_units = Counter()
        total_cases = Counter()
        hit_cases = Counter()
        greedy_hits = Counter()
        nonfinite_candidate_count = 0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")
        capture = RoleCapture(model, layers)
        capture.register()
        batch_size = UNIT_BATCH_SIZE[model_name]
        state_order = list(protocol.STATES)
        with torch.inference_mode():
            for batch_start in range(0, len(units), batch_size):
                batch_units = units[batch_start:batch_start + batch_size]
                forward_rows = []
                unit_offsets = []
                for unit in batch_units:
                    offset = len(forward_rows)
                    state_rows = [
                        unit["states"][state] for state in state_order
                    ]
                    forward_rows.extend(state_rows)
                    forward_rows.append(unit["states"]["b0_l0"])
                    unit_offsets.append(offset)
                (
                    input_ids,
                    attention_mask,
                    lengths,
                    positions,
                ) = pad_rows(forward_rows, int(pad_id), device)
                capture.begin(positions)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                capture.validate()
                logits = output.logits
                last_positions = (lengths - 1).to(logits.device)
                batch_axis = torch.arange(
                    logits.shape[0], device=logits.device
                )
                last_logits = logits[batch_axis, last_positions, :].float()
                del output, logits

                for unit, offset in zip(batch_units, unit_offsets):
                    local_hits = {}
                    for local_index, state in enumerate(state_order):
                        row = unit["states"][state]
                        values = last_logits[offset + local_index]
                        class_scores = {}
                        for class_name in ("b0", "b1"):
                            token_ids = torch.tensor(
                                row["candidate_first_token_ids"][
                                    class_name
                                ],
                                dtype=torch.long,
                                device=values.device,
                            )
                            class_scores[class_name] = float(
                                values[token_ids].max().item()
                            )
                        expected = str(row["expected_class"])
                        other = "b1" if expected == "b0" else "b0"
                        margin = (
                            class_scores[expected] - class_scores[other]
                        )
                        finite_candidate = all(
                            math.isfinite(value)
                            for value in class_scores.values()
                        ) and math.isfinite(margin)
                        if not finite_candidate:
                            nonfinite_candidate_count += 1
                        hit = finite_candidate and margin > 0.0
                        greedy_token = int(torch.argmax(values).item())
                        greedy_hit = greedy_token in set(
                            int(value)
                            for value in row[
                                "candidate_first_token_ids"
                            ][expected]
                        )
                        index = int(row["semantic_case_index"])
                        case_hit[index] = hit
                        candidate_margin[index] = margin
                        local_hits[state] = hit
                        key = (unit["family"], unit["split"])
                        total_cases[key] += 1
                        hit_cases[key] += int(hit)
                        greedy_hits[key] += int(greedy_hit)
                        behavior_records.append({
                            "schema_version": (
                                "phase1065_candidate_behavior.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "semantic_case_index": index,
                            "unit_id": unit["unit_id"],
                            "family": unit["family"],
                            "split": unit["split"],
                            "state": state,
                            "expected_class": expected,
                            "candidate_class_scores": {
                                key: value if math.isfinite(value) else None
                                for key, value in class_scores.items()
                            },
                            "candidate_margin": (
                                margin if math.isfinite(margin) else None
                            ),
                            "nonfinite_candidate": not finite_candidate,
                            "candidate_hit": hit,
                            "greedy_first_token_id": greedy_token,
                            "greedy_first_token_text": tokenizer.decode(
                                [greedy_token]
                            ),
                            "greedy_first_token_hit": greedy_hit,
                        })
                    for lexical in (0, 1):
                        if (
                            local_hits[f"b0_l{lexical}"]
                            and local_hits[f"b1_l{lexical}"]
                        ):
                            valid_pairs[
                                (unit["family"], unit["split"])
                            ] += 1
                    if all(local_hits.values()):
                        valid_units[
                            (unit["family"], unit["split"])
                        ] += 1

                for event_index, key in enumerate(event_keys):
                    value = capture.values[key].float()
                    for unit, offset in zip(batch_units, unit_offsets):
                        family = family_index[str(unit["family"])]
                        split = split_index[str(unit["split"])]
                        states = {
                            state: value[offset + local_index]
                            for local_index, state in enumerate(state_order)
                        }
                        identity = value[offset + len(state_order)]
                        identity_maximum = max(
                            identity_maximum,
                            float(
                                torch.max(torch.abs(
                                    identity - states["b0_l0"]
                                )).item()
                            ),
                        )
                        deltas = {
                            0: states["b1_l0"] - states["b0_l0"],
                            1: states["b1_l1"] - states["b0_l1"],
                        }
                        for lexical, delta in deltas.items():
                            pair_valid = (
                                case_hit[int(unit["states"][
                                    f"b0_l{lexical}"
                                ]["semantic_case_index"])]
                                and case_hit[int(unit["states"][
                                    f"b1_l{lexical}"
                                ]["semantic_case_index"])]
                            )
                            if not pair_valid:
                                continue
                            base = 0.5 * (
                                torch.linalg.vector_norm(
                                    states[f"b0_l{lexical}"], dim=-1
                                )
                                + torch.linalg.vector_norm(
                                    states[f"b1_l{lexical}"], dim=-1
                                )
                            )
                            norms = torch.linalg.vector_norm(
                                delta, dim=-1
                            )
                            valid = norms > EPSILON
                            units_direction = torch.zeros_like(delta)
                            units_direction[valid] = (
                                delta[valid] / norms[valid, None]
                            )
                            relative = torch.zeros_like(norms)
                            base_valid = base > EPSILON
                            relative[base_valid] = (
                                norms[base_valid] / base[base_valid]
                            )
                            direction_sum[
                                family, split, event_index
                            ] += units_direction.cpu().numpy()
                            semantic_count[
                                family, split, event_index
                            ] += valid.cpu().numpy().astype(np.int32)
                            semantic_relative_magnitude_sum[
                                family, split, event_index
                            ] += relative.cpu().numpy()

                        for branch in (0, 1):
                            left_row = unit["states"][f"b{branch}_l0"]
                            right_row = unit["states"][f"b{branch}_l1"]
                            if not (
                                case_hit[int(
                                    left_row["semantic_case_index"]
                                )]
                                and case_hit[int(
                                    right_row["semantic_case_index"]
                                )]
                            ):
                                continue
                            surface = (
                                states[f"b{branch}_l1"]
                                - states[f"b{branch}_l0"]
                            )
                            base = 0.5 * (
                                torch.linalg.vector_norm(
                                    states[f"b{branch}_l0"], dim=-1
                                )
                                + torch.linalg.vector_norm(
                                    states[f"b{branch}_l1"], dim=-1
                                )
                            )
                            norms = torch.linalg.vector_norm(
                                surface, dim=-1
                            )
                            relative = torch.zeros_like(norms)
                            valid = base > EPSILON
                            relative[valid] = norms[valid] / base[valid]
                            surface_relative_magnitude_sum[
                                family, split, event_index
                            ] += relative.cpu().numpy()
                            surface_count[
                                family, split, event_index
                            ] += valid.cpu().numpy().astype(np.int32)

                        if all(
                            case_hit[int(unit["states"][state][
                                "semantic_case_index"
                            ])]
                            for state in state_order
                        ):
                            branch_cos = cosine(
                                deltas[0], deltas[1]
                            )
                            denominator = (
                                torch.linalg.vector_norm(
                                    deltas[0], dim=-1
                                )
                                + torch.linalg.vector_norm(
                                    deltas[1], dim=-1
                                )
                            )
                            interaction = torch.linalg.vector_norm(
                                deltas[1] - deltas[0], dim=-1
                            )
                            relative_interaction = torch.zeros_like(
                                interaction
                            )
                            valid = denominator > EPSILON
                            relative_interaction[valid] = (
                                interaction[valid] / denominator[valid]
                            )
                            branch_cosine_sum[
                                family, split, event_index
                            ] += branch_cos.cpu().numpy()
                            branch_cosine_count[
                                family, split, event_index
                            ] += 1
                            interaction_relative_sum[
                                family, split, event_index
                            ] += relative_interaction.cpu().numpy()
                    del value
                del last_logits, input_ids, attention_mask, lengths, positions
                capture.values = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                completed = min(
                    batch_start + len(batch_units), len(units)
                )
                if completed % 30 == 0 or completed == len(units):
                    print(
                        json.dumps({
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "units_complete": completed,
                            "units_total": len(units),
                        }),
                        flush=True,
                    )
        capture.close()
        capture = None

        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        natural_rows = natural_selection(rows)
        natural_outputs = generation.generate_case_outputs(
            model,
            device,
            natural_rows,
            eos_ids=eos_ids,
            batch_size=bridge.PAIR_BATCH_SIZE[model_name],
            steps=int(prereg["natural_generation_steps"]),
        )
        natural_records = []
        natural_counts = Counter()
        natural_exact = Counter()
        natural_terminated = Counter()
        for row in natural_rows:
            index = int(row["semantic_case_index"])
            output_ids = natural_outputs[index]
            answer = strict_generated_answer(
                tokenizer, output_ids, eos_ids
            )
            acceptable = {
                text_tools.normalize_text(str(value))
                for value in row["acceptable_labels"]
            }
            terminated = generation.terminated(output_ids, eos_ids)
            exact = terminated and answer in acceptable
            family = str(row["family"])
            natural_counts[family] += 1
            natural_exact[family] += int(exact)
            natural_terminated[family] += int(terminated)
            natural_records.append({
                "schema_version": "phase1065_natural_audit.v1",
                "phase": protocol.PHASE,
                "model": model_name,
                "semantic_case_index": index,
                "unit_id": row["unit_id"],
                "family": family,
                "split": row["split"],
                "state": row["state"],
                "generated_token_ids": [
                    int(value) for value in output_ids
                ],
                "generated_text": answer,
                "acceptable_labels": sorted(acceptable),
                "terminated": terminated,
                "exact": exact,
            })

        metric_rows = []
        mean_directions = np.zeros_like(direction_sum, dtype=np.float16)
        for family_name, family in family_index.items():
            for split_name, split in split_index.items():
                for event_index, event in enumerate(events):
                    for role_name, role in role_index.items():
                        count = int(semantic_count[
                            family, split, event_index, role
                        ])
                        vector = direction_sum[
                            family, split, event_index, role
                        ]
                        norm = float(np.linalg.norm(
                            vector.astype(np.float64, copy=False)
                        ))
                        if norm > EPSILON:
                            mean_directions[
                                family, split, event_index, role
                            ] = (vector / norm).astype(np.float16)
                        surface_n = int(surface_count[
                            family, split, event_index, role
                        ])
                        branch_n = int(branch_cosine_count[
                            family, split, event_index, role
                        ])
                        metric_rows.append({
                            "schema_version": (
                                "phase1065_response_metric.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "family": family_name,
                            "split": split_name,
                            "role": role_name,
                            **event,
                            "semantic_pair_count": count,
                            "semantic_direction_consistency": (
                                pairwise_direction_consistency(
                                    vector, count
                                )
                            ),
                            "mean_semantic_relative_magnitude": (
                                float(
                                    semantic_relative_magnitude_sum[
                                        family,
                                        split,
                                        event_index,
                                        role,
                                    ] / count
                                )
                                if count else None
                            ),
                            "surface_observation_count": surface_n,
                            "mean_surface_relative_magnitude": (
                                float(
                                    surface_relative_magnitude_sum[
                                        family,
                                        split,
                                        event_index,
                                        role,
                                    ] / surface_n
                                )
                                if surface_n else None
                            ),
                            "complete_unit_count": branch_n,
                            "mean_surface_branch_semantic_cosine": (
                                float(
                                    branch_cosine_sum[
                                        family,
                                        split,
                                        event_index,
                                        role,
                                    ] / branch_n
                                )
                                if branch_n else None
                            ),
                            "mean_interaction_relative_magnitude": (
                                float(
                                    interaction_relative_sum[
                                        family,
                                        split,
                                        event_index,
                                        role,
                                    ] / branch_n
                                )
                                if branch_n else None
                            ),
                        })

        family_summaries = {}
        for family in protocol.FAMILIES:
            by_split = {}
            total = hit = greedy = pairs = complete = 0
            for split in protocol.SPLITS:
                key = (family, split)
                by_split[split] = {
                    "case_count": total_cases[key],
                    "candidate_hit_count": hit_cases[key],
                    "candidate_first_token_accuracy": (
                        hit_cases[key] / total_cases[key]
                        if total_cases[key] else 0.0
                    ),
                    "greedy_first_token_hit_count": greedy_hits[key],
                    "greedy_first_token_accuracy": (
                        greedy_hits[key] / total_cases[key]
                        if total_cases[key] else 0.0
                    ),
                    "valid_semantic_pair_count": valid_pairs[key],
                    "complete_factorial_unit_count": valid_units[key],
                }
                total += total_cases[key]
                hit += hit_cases[key]
                greedy += greedy_hits[key]
                pairs += valid_pairs[key]
                complete += valid_units[key]
            natural_total = natural_counts[family]
            natural_rate = (
                natural_exact[family] / natural_total
                if natural_total else 0.0
            )
            candidate_accuracy = hit / total if total else 0.0
            behavior_pass = (
                candidate_accuracy
                >= prereg["gates"][
                    "candidate_first_token_accuracy_min"
                ]
                and pairs
                >= prereg["gates"]["valid_semantic_pair_min"]
                and all(
                    valid_pairs[(family, split)]
                    >= prereg["gates"][
                        "valid_semantic_pair_per_split_min"
                    ]
                    for split in protocol.SPLITS
                )
            )
            strong_pass = (
                behavior_pass
                and pairs
                >= prereg["gates"][
                    "strong_valid_semantic_pair_min"
                ]
                and natural_rate
                >= prereg["gates"]["natural_audit_exact_rate_min"]
            )
            family_summaries[family] = {
                "case_count": total,
                "candidate_hit_count": hit,
                "candidate_first_token_accuracy": candidate_accuracy,
                "greedy_first_token_hit_count": greedy,
                "greedy_first_token_accuracy": (
                    greedy / total if total else 0.0
                ),
                "valid_semantic_pair_count": pairs,
                "complete_factorial_unit_count": complete,
                "natural_audit_case_count": natural_total,
                "natural_audit_terminated_count": (
                    natural_terminated[family]
                ),
                "natural_audit_exact_count": natural_exact[family],
                "natural_audit_exact_rate": natural_rate,
                "by_split": by_split,
                "behavior_gate_passed": behavior_pass,
                "strong_behavior_gate_passed": strong_pass,
            }

        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_jsonl(
            atlas_root / "candidate_behavior.jsonl",
            behavior_records,
        )
        protocol.write_jsonl(
            atlas_root / "natural_generation_audit.jsonl",
            natural_records,
        )
        protocol.write_jsonl(
            atlas_root / "response_metrics.jsonl",
            metric_rows,
        )
        np.savez_compressed(
            atlas_root / "mean_directions.fp16.npz",
            mean_directions=mean_directions,
            semantic_counts=semantic_count,
            family_names=np.array(protocol.FAMILIES),
            split_names=np.array(protocol.SPLITS),
            role_names=np.array(protocol.CAPTURE_ROLES),
            event_ids=np.array([
                row["event_id"] for row in events
            ]),
        )
        summary = {
            "schema_version": "phase1065_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "n_layers": len(layers),
                "d_model": d_model,
            },
            "case_count": len(rows),
            "unit_count": len(units),
            "event_count": len(events),
            "identity_maximum": identity_maximum,
            "nonfinite_candidate_count": nonfinite_candidate_count,
            "families": family_summaries,
            "elapsed_seconds": time.time() - started,
            "interpretation_limits": prereg[
                "interpretation_limits"
            ],
        }
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "families": {
                family: {
                    "candidate_accuracy": row[
                        "candidate_first_token_accuracy"
                    ],
                    "valid_pairs": row[
                        "valid_semantic_pair_count"
                    ],
                    "natural_exact": row[
                        "natural_audit_exact_rate"
                    ],
                    "behavior_gate": row[
                        "behavior_gate_passed"
                    ],
                    "strong_gate": row[
                        "strong_behavior_gate_passed"
                    ],
                }
                for family, row in family_summaries.items()
            },
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False), flush=True)
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
