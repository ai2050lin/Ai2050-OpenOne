#!/usr/bin/env python3
"""Run the Phase1029 two-binding competition atlas in FP16."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info
import phase1029_multibinding_competition_protocol as protocol
from phase1023_fp16_utils import (
    MODELS,
    load_fp16,
    quantization_audit,
    release_fp16,
)


BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
ROLE_INDEX = {role: index for index, role in enumerate(protocol.ROLES)}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def make_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    positions = torch.empty(
        (len(rows), len(protocol.ROLES)),
        dtype=torch.long,
    )
    for index, row in enumerate(rows):
        value = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[index, :len(value)] = value
        mask[index, :len(value)] = 1
        for role_index, role in enumerate(protocol.ROLES):
            positions[index, role_index] = int(
                row["role_positions"][role]
            )
    return ids.to(device), mask.to(device), positions


def replace_output(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


class MultiDepthCleanCapture:
    def __init__(
        self,
        layers,
        patch_depths: list[int],
        readout_depth: int,
    ):
        self.layers = layers
        self.patch_depths = patch_depths
        self.readout_depth = readout_depth
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.readout: torch.Tensor | None = None
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def _role_hook(self, depth: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.positions is None:
                raise RuntimeError("clean positions missing")
            positions = self.positions.to(hidden.device)
            batch = torch.arange(
                hidden.shape[0], device=hidden.device
            )[:, None]
            self.values[depth] = hidden[batch, positions, :].detach()
            self.counts[f"depth/{depth}"] += 1
            return output
        return hook

    def _readout_hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.positions is None:
            raise RuntimeError("readout positions missing")
        positions = self.positions[
            :, ROLE_INDEX["pre_output"]
        ].to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        self.readout = hidden[batch, positions, :].detach()
        self.counts["readout"] += 1
        return output

    def register(self) -> None:
        for depth in self.patch_depths:
            self.handles.append(
                self.layers[depth - 1].register_forward_hook(
                    self._role_hook(depth)
                )
            )
        self.handles.append(
            self.layers[
                self.readout_depth - 1
            ].register_forward_hook(self._readout_hook)
        )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.readout = None
        self.counts = defaultdict(int)

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        expected = {
            f"depth/{depth}": 1 for depth in self.patch_depths
        } | {"readout": 1}
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"clean hook count drift: {dict(self.counts)}"
            )
        if self.readout is None:
            raise RuntimeError("clean readout missing")
        return (
            torch.stack(
                [self.values[depth] for depth in self.patch_depths],
                dim=2,
            ).to("cpu"),
            self.readout.to("cpu"),
        )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class MultiPatchCapture:
    def __init__(self, layers, depths: list[int], readout_depth: int):
        self.layers = layers
        self.depths = sorted(set(depths))
        self.readout_depth = readout_depth
        self.specs: dict[
            int,
            list[tuple[torch.Tensor, torch.Tensor]],
        ] = {}
        self.pre_positions: torch.Tensor | None = None
        self.readout: torch.Tensor | None = None
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def _patch_hook(self, depth: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            for positions, replacement in self.specs.get(depth, []):
                patched[
                    batch,
                    positions.to(hidden.device),
                    :,
                ] = replacement.to(hidden.device, dtype=hidden.dtype)
            self.counts[f"patch/{depth}"] += 1
            return replace_output(output, patched)
        return hook

    def _readout_hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.pre_positions is None:
            raise RuntimeError("readout positions missing")
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        self.readout = hidden[
            batch,
            self.pre_positions.to(hidden.device),
            :,
        ].detach()
        self.counts["readout"] += 1
        return output

    def register(self) -> None:
        for depth in self.depths:
            self.handles.append(
                self.layers[depth - 1].register_forward_hook(
                    self._patch_hook(depth)
                )
            )
        self.handles.append(
            self.layers[
                self.readout_depth - 1
            ].register_forward_hook(self._readout_hook)
        )

    def begin(
        self,
        *,
        specs: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
        pre_positions: torch.Tensor,
    ) -> None:
        if set(specs) != set(self.depths):
            raise RuntimeError(
                f"patch depth drift: {sorted(specs)} != {self.depths}"
            )
        self.specs = specs
        self.pre_positions = pre_positions
        self.readout = None
        self.counts = defaultdict(int)

    def value(self) -> torch.Tensor:
        expected = {
            f"patch/{depth}": 1 for depth in self.depths
        } | {"readout": 1}
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"patch hook count drift: {dict(self.counts)}"
            )
        if self.readout is None:
            raise RuntimeError("patched readout missing")
        return self.readout.to("cpu")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def prototype_map(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    for split in protocol.SPLITS:
        surface_count = 4 if split == "discovery" else 8
        for held_surface in range(surface_count):
            values = np.empty(
                (8, clean_readout.shape[-1]),
                dtype=np.float32,
            )
            for concept_index in range(8):
                indices = [
                    int(row["case_index"])
                    for row in cases
                    if row["split"] == split
                    and int(row["expected_index"]) == concept_index
                    and int(row["surface_index"]) != held_surface
                ]
                values[concept_index] = np.asarray(
                    clean_readout[indices], dtype=np.float32
                ).mean(axis=0)
            result[(split, held_surface)] = normalize_rows(values)
    return result


def classify_values(
    values: np.ndarray,
    rows: list[dict[str, Any]],
    prototypes: dict[tuple[str, int], np.ndarray],
    *,
    base_field: str,
    alternate_field: str,
    scrambled_field: str | None = None,
) -> dict[str, Any]:
    base_hits = []
    alternate_hits = []
    scrambled_hits = []
    other_hits = []
    alternate_margins = []
    base_margins = []
    for local_index, row in enumerate(rows):
        split = row["split"]
        proto = prototypes[(split, int(row["surface_index"]))]
        similarity = normalize_rows(
            values[local_index:local_index + 1]
        )[0] @ proto.T
        predicted = int(np.argmax(similarity))
        base_index = int(row[base_field])
        alternate_index = int(row[alternate_field])
        base_hits.append(int(predicted == base_index))
        alternate_hits.append(int(predicted == alternate_index))
        if scrambled_field is not None:
            scrambled_index = int(row[scrambled_field])
            scrambled_hits.append(int(predicted == scrambled_index))
        other_hits.append(
            int(predicted not in {base_index, alternate_index})
        )
        alternate_margins.append(
            float(similarity[alternate_index] - similarity[base_index])
        )
        base_margins.append(
            float(similarity[base_index] - similarity[alternate_index])
        )
    result = {
        "row_count": len(rows),
        "base_top1": float(np.mean(base_hits)),
        "alternate_top1": float(np.mean(alternate_hits)),
        "other_top1": float(np.mean(other_hits)),
        "alternate_vs_base_margin": float(
            np.mean(alternate_margins)
        ),
        "base_vs_alternate_margin": float(np.mean(base_margins)),
        "chance": 0.125,
    }
    if scrambled_hits:
        result["scrambled_top1"] = float(np.mean(scrambled_hits))
    return result


def clean_world_metrics(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
    prototypes: dict[tuple[str, int], np.ndarray],
    split: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    all_hits = []
    for world in protocol.WORLD_CODES:
        rows = [
            row
            for row in cases
            if row["split"] == split and row["world"] == world
        ]
        values = np.asarray([
            clean_readout[int(row["case_index"])] for row in rows
        ])
        hits = []
        margins = []
        for value, row in zip(values, rows):
            proto = prototypes[
                (split, int(row["surface_index"]))
            ]
            similarity = normalize_rows(value[None, :])[0] @ proto.T
            expected = int(row["expected_index"])
            wrong = np.delete(similarity, expected)
            hit = int(np.argmax(similarity) == expected)
            hits.append(hit)
            all_hits.append(hit)
            margins.append(
                float(similarity[expected] - np.max(wrong))
            )
        result[world] = {
            "case_count": len(rows),
            "expected_top1": float(np.mean(hits)),
            "expected_vs_wrong_margin": float(np.mean(margins)),
            "chance": 0.125,
        }
    result["all_worlds"] = {
        "case_count": len(all_hits),
        "expected_top1": float(np.mean(all_hits)),
        "chance": 0.125,
    }
    return result


def role_answer_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    rows = [row for row in cases if row["split"] == split]
    state = np.asarray(
        [values[int(row["case_index"])] for row in rows],
        dtype=np.float32,
    )
    surface_count = 4 if split == "discovery" else 8
    prototypes: dict[int, np.ndarray] = {}
    for held_surface in range(surface_count):
        current = np.empty((8, state.shape[-1]), dtype=np.float32)
        for concept_index in range(8):
            indices = [
                index
                for index, candidate in enumerate(rows)
                if int(candidate["expected_index"]) == concept_index
                and int(candidate["surface_index"]) != held_surface
            ]
            current[concept_index] = state[indices].mean(axis=0)
        prototypes[held_surface] = normalize_rows(current)
    normalized_state = normalize_rows(state)
    hits = []
    margins = []
    for local_index, row in enumerate(rows):
        similarity = normalized_state[local_index] @ prototypes[
            int(row["surface_index"])
        ].T
        expected = int(row["expected_index"])
        wrong = np.delete(similarity, expected)
        hits.append(int(np.argmax(similarity) == expected))
        margins.append(float(similarity[expected] - np.max(wrong)))
    if surface_count < 2:
        raise RuntimeError("leave-surface-out requires multiple surfaces")
    return {
        "case_count": len(rows),
        "answer_top1": float(np.mean(hits)),
        "answer_vs_wrong_margin": float(np.mean(margins)),
        "chance": 0.125,
    }


def safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(
        b, axis=-1
    )
    return numerator / np.maximum(denominator, EPS)


def difference_metrics(
    values: np.ndarray,
    units: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    rows = [row for row in units if row["split"] == split]
    world = {
        code: np.asarray([
            values[int(row["world_case_indices"][code])]
            for row in rows
        ], dtype=np.float32)
        for code in protocol.WORLD_CODES
    }
    d_b = world["10"] - world["00"]
    d_q = world["01"] - world["00"]
    d_bq = world["11"] - world["00"]
    interaction = (
        world["11"] - world["10"] - world["01"] + world["00"]
    )
    scale = 0.5 * (
        np.linalg.norm(d_b, axis=-1)
        + np.linalg.norm(d_q, axis=-1)
    )
    same_answer_ratio = (
        np.linalg.norm(d_bq, axis=-1) / np.maximum(scale, EPS)
    )
    interaction_ratio = (
        np.linalg.norm(interaction, axis=-1)
        / np.maximum(scale, EPS)
    )
    b_q_cosine = safe_cosine(d_b, d_q)
    return {
        "unit_count": len(rows),
        "binding_delta_norm_mean": float(
            np.mean(np.linalg.norm(d_b, axis=-1))
        ),
        "query_delta_norm_mean": float(
            np.mean(np.linalg.norm(d_q, axis=-1))
        ),
        "same_answer_delta_norm_mean": float(
            np.mean(np.linalg.norm(d_bq, axis=-1))
        ),
        "binding_query_cosine_mean": float(np.mean(b_q_cosine)),
        "binding_query_cosine_std": float(np.std(b_q_cosine)),
        "same_answer_ratio_mean": float(np.mean(same_answer_ratio)),
        "same_answer_ratio_median": float(
            np.median(same_answer_ratio)
        ),
        "interaction_ratio_mean": float(np.mean(interaction_ratio)),
        "interaction_ratio_median": float(
            np.median(interaction_ratio)
        ),
    }


def unit_rows(
    units: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    return [row for row in units if row["split"] == split]


def role_names(
    rows: list[dict[str, Any]],
    field: str,
) -> list[str]:
    return [str(row[field]) for row in rows]


def case_indices(
    rows: list[dict[str, Any]],
    world: str,
    *,
    units_all: list[dict[str, Any]],
    scrambled: bool = False,
) -> list[int]:
    if not scrambled:
        return [
            int(row["world_case_indices"][world]) for row in rows
        ]
    return [
        int(
            units_all[int(row["scrambled_unit_index"])]
            ["world_case_indices"][world]
        )
        for row in rows
    ]


def gather_states(
    clean_states: np.ndarray,
    indices: list[int],
    roles: list[str],
    depth_index: int,
) -> torch.Tensor:
    values = np.stack([
        np.asarray(
            clean_states[
                case_index,
                ROLE_INDEX[role],
                depth_index,
                :,
            ],
            dtype=np.float16,
        )
        for case_index, role in zip(indices, roles)
    ])
    return torch.from_numpy(values.copy())


def target_positions(
    positions: torch.Tensor,
    roles: list[str],
) -> torch.Tensor:
    return torch.tensor([
        int(positions[index, ROLE_INDEX[role]])
        for index, role in enumerate(roles)
    ], dtype=torch.long)


def condition_patch_plan(
    condition: str,
    rows: list[dict[str, Any]],
    *,
    units_all: list[dict[str, Any]],
    positions: torch.Tensor,
    clean_states: np.ndarray,
    depths: list[int],
    source_depth: int,
    query_depth: int,
    preoutput_depth: int,
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    plans: list[tuple[int, list[str], list[int], list[str]]] = []
    selected = role_names(rows, "selected_concept_role")
    unselected = role_names(rows, "unselected_concept_role")
    selected_def = role_names(rows, "selected_definition_role")
    source_a = ["concept_a_end"] * len(rows)
    source_b = ["concept_b_end"] * len(rows)
    definition_a = ["definition_nonce_a_end"] * len(rows)
    definition_b = ["definition_nonce_b_end"] * len(rows)
    query = ["query_nonce_end"] * len(rows)
    pre_output = ["pre_output"] * len(rows)
    b_indices = case_indices(
        rows, "10", units_all=units_all
    )
    q_indices = case_indices(
        rows, "01", units_all=units_all
    )
    bq_indices = case_indices(
        rows, "11", units_all=units_all
    )

    if condition == "selected_source_b":
        plans.append((source_depth, selected, b_indices, selected))
    elif condition == "unselected_source_b":
        plans.append((source_depth, unselected, b_indices, unselected))
    elif condition == "source_pair_b":
        plans.extend([
            (source_depth, source_a, b_indices, source_a),
            (source_depth, source_b, b_indices, source_b),
        ])
    elif condition == "query_q":
        plans.append((query_depth, query, q_indices, query))
    elif condition == "source_pair_plus_query_mixed":
        plans.extend([
            (source_depth, source_a, b_indices, source_a),
            (source_depth, source_b, b_indices, source_b),
            (query_depth, query, q_indices, query),
        ])
    elif condition == "full_bq":
        plans.extend([
            (source_depth, source_a, bq_indices, source_a),
            (source_depth, source_b, bq_indices, source_b),
            (query_depth, query, bq_indices, query),
        ])
    elif condition == "source_pair_scrambled":
        scrambled_indices = case_indices(
            rows,
            "10",
            units_all=units_all,
            scrambled=True,
        )
        plans.extend([
            (source_depth, source_a, scrambled_indices, source_a),
            (source_depth, source_b, scrambled_indices, source_b),
        ])
    elif condition == "source_pair_wrong_position":
        plans.extend([
            (source_depth, definition_a, b_indices, source_a),
            (source_depth, definition_b, b_indices, source_b),
        ])
    elif condition == "query_wrong_position":
        plans.append((query_depth, selected_def, q_indices, query))
    elif condition == "pre_output_b":
        plans.append((
            preoutput_depth, pre_output, b_indices, pre_output
        ))
    elif condition == "pre_output_bq":
        plans.append((
            preoutput_depth, pre_output, bq_indices, pre_output
        ))
    else:
        raise ValueError(condition)

    result: dict[
        int,
        list[tuple[torch.Tensor, torch.Tensor]],
    ] = defaultdict(list)
    for depth, target_roles, donor_indices, donor_roles in plans:
        depth_index = depths.index(depth)
        result[depth].append((
            target_positions(positions, target_roles),
            gather_states(
                clean_states,
                donor_indices,
                donor_roles,
                depth_index,
            ),
        ))
    return dict(result)


def run_condition(
    *,
    base_model,
    layers,
    condition: str,
    rows: list[dict[str, Any]],
    units_all: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    clean_states: np.ndarray,
    depths: list[int],
    source_depth: int,
    query_depth: int,
    preoutput_depth: int,
    readout_depth: int,
    tokenizer,
    device: torch.device,
    model_name: str,
    output: np.ndarray,
) -> None:
    used_depths = {
        "selected_source_b": [source_depth],
        "unselected_source_b": [source_depth],
        "source_pair_b": [source_depth],
        "query_q": [query_depth],
        "source_pair_plus_query_mixed": [source_depth, query_depth],
        "full_bq": [source_depth, query_depth],
        "source_pair_scrambled": [source_depth],
        "source_pair_wrong_position": [source_depth],
        "query_wrong_position": [query_depth],
        "pre_output_b": [preoutput_depth],
        "pre_output_bq": [preoutput_depth],
    }[condition]
    capture = MultiPatchCapture(
        layers, sorted(set(used_depths)), readout_depth
    )
    capture.register()
    try:
        offset = 0
        for batch_index, row_batch in enumerate(
            chunks(rows, BATCH_SIZE[model_name]), 1
        ):
            target_cases = [
                cases[int(row["world_case_indices"]["00"])]
                for row in row_batch
            ]
            input_ids, attention_mask, positions = make_batch(
                target_cases,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            specs = condition_patch_plan(
                condition,
                row_batch,
                units_all=units_all,
                positions=positions,
                clean_states=clean_states,
                depths=depths,
                source_depth=source_depth,
                query_depth=query_depth,
                preoutput_depth=preoutput_depth,
            )
            capture.begin(
                specs=specs,
                pre_positions=positions[:, ROLE_INDEX["pre_output"]],
            )
            with torch.inference_mode():
                base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            value = capture.value().numpy().astype(
                np.float16, copy=False
            )
            output[offset:offset + len(row_batch)] = value
            offset += len(row_batch)
            if batch_index % 16 == 0:
                print(
                    f"[phase1029] {model_name} {condition} "
                    f"units={offset}/{len(rows)}",
                    flush=True,
                )
    finally:
        capture.close()
    output.flush()


def intervention_metrics(
    values: np.ndarray,
    rows: list[dict[str, Any]],
    prototypes: dict[tuple[str, int], np.ndarray],
) -> dict[str, Any]:
    metric_rows = [
        {
            **row,
            "base_index": int(row["target_index"]),
            "alternate_index": int(row["donor_index"]),
        }
        for row in rows
    ]
    return classify_values(
        values,
        metric_rows,
        prototypes,
        base_field="base_index",
        alternate_field="alternate_index",
        scrambled_field="scrambled_donor_index",
    )


def select_depth(
    rows: list[dict[str, Any]],
    condition: str,
) -> int:
    candidates = [row for row in rows if row["condition"] == condition]
    best = max(
        candidates,
        key=lambda row: (
            row["metrics"]["alternate_top1"],
            row["metrics"]["alternate_vs_base_margin"],
            -int(row["depth"]),
        ),
    )
    return int(best["depth"])


def finite_audit(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for name, values in arrays.items():
        total = int(np.prod(values.shape))
        nonfinite = int(np.count_nonzero(~np.isfinite(values)))
        result[name] = {
            "shape": list(values.shape),
            "value_count": total,
            "nonfinite_count": nonfinite,
            "all_finite": nonfinite == 0,
        }
    return {
        "arrays": result,
        "all_finite": all(row["all_finite"] for row in result.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()

    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    units = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "units.jsonl"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{args.model}.jsonl"
    )
    discovery_units = unit_rows(units, "discovery")
    confirmation_units = unit_rows(units, "confirmation")
    depths = [
        int(value) for value in prereg["patch_depths"][args.model]
    ]
    readout_depth = int(prereg["readout_depth"][args.model])
    preoutput_depth = int(prereg["preoutput_depth"][args.model])
    started = time.time()
    model = tokenizer = None
    atlas_dir = protocol.OUT_ROOT / "atlas" / args.model
    atlas_dir.mkdir(parents=True, exist_ok=True)
    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        precision_audit = quantization_audit(model)
        if (
            precision_audit["has_quantized_modules"]
            or precision_audit["has_bf16_parameters"]
            or not precision_audit["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, args.model)
        if max(depths) >= readout_depth:
            raise RuntimeError("patch depth must precede readout depth")
        if preoutput_depth not in depths:
            raise RuntimeError("preoutput depth is outside frozen grid")
        base_model = model.model

        clean_states = np.lib.format.open_memmap(
            atlas_dir / "clean_role_states.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases),
                len(protocol.ROLES),
                len(depths),
                info.d_model,
            ),
        )
        clean_readout = np.lib.format.open_memmap(
            atlas_dir / "clean_readout.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(cases), info.d_model),
        )
        capture = MultiDepthCleanCapture(
            layers, depths, readout_depth
        )
        capture.register()
        try:
            offset = 0
            for batch_index, batch in enumerate(
                chunks(cases, BATCH_SIZE[args.model]), 1
            ):
                input_ids, attention_mask, positions = make_batch(
                    batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                capture.begin(positions)
                with torch.inference_mode():
                    base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                states, readout = capture.stacked()
                clean_states[offset:offset + len(batch)] = (
                    states.numpy().astype(np.float16, copy=False)
                )
                clean_readout[offset:offset + len(batch)] = (
                    readout.numpy().astype(np.float16, copy=False)
                )
                offset += len(batch)
                if batch_index % 32 == 0:
                    print(
                        f"[phase1029] {args.model} clean "
                        f"cases={offset}/{len(cases)}",
                        flush=True,
                    )
        finally:
            capture.close()
        clean_states.flush()
        clean_readout.flush()

        prototypes = prototype_map(clean_readout, cases)
        clean_metrics = {
            split: clean_world_metrics(
                clean_readout, cases, prototypes, split
            )
            for split in protocol.SPLITS
        }
        observational = []
        for role_index, role in enumerate(protocol.ROLES):
            for depth_index, depth in enumerate(depths):
                values = np.asarray(
                    clean_states[:, role_index, depth_index, :]
                )
                observational.append({
                    "role": role,
                    "depth": depth,
                    "answer_readout": {
                        split: role_answer_metrics(
                            values, cases, split
                        )
                        for split in protocol.SPLITS
                    },
                    "four_world_differences": {
                        split: difference_metrics(
                            values, units, split
                        )
                        for split in protocol.SPLITS
                    },
                })

        discovery_output = np.lib.format.open_memmap(
            atlas_dir / "discovery_depth_scan.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                2,
                len(depths),
                len(discovery_units),
                info.d_model,
            ),
        )
        discovery_rows = []
        for condition_index, condition in enumerate(
            ("source_pair_b", "query_q")
        ):
            for depth_index, depth in enumerate(depths):
                run_condition(
                    base_model=base_model,
                    layers=layers,
                    condition=condition,
                    rows=discovery_units,
                    units_all=units,
                    cases=cases,
                    clean_states=clean_states,
                    depths=depths,
                    source_depth=depth,
                    query_depth=depth,
                    preoutput_depth=preoutput_depth,
                    readout_depth=readout_depth,
                    tokenizer=tokenizer,
                    device=device,
                    model_name=args.model,
                    output=discovery_output[
                        condition_index, depth_index
                    ],
                )
                discovery_rows.append({
                    "condition": condition,
                    "depth": depth,
                    "metrics": intervention_metrics(
                        np.asarray(
                            discovery_output[
                                condition_index, depth_index
                            ]
                        ),
                        discovery_units,
                        prototypes,
                    ),
                })
        discovery_output.flush()
        source_depth = select_depth(
            discovery_rows, "source_pair_b"
        )
        query_depth = select_depth(discovery_rows, "query_q")
        selection = {
            "schema_version": "phase1029_selection.v1",
            "selection_source": "discovery_only",
            "policy": prereg["discovery_selection"],
            "source_depth": source_depth,
            "query_depth": query_depth,
            "preoutput_depth": preoutput_depth,
            "discovery_rows": discovery_rows,
        }
        protocol.write_json(atlas_dir / "selection.json", selection)

        confirmation_output = np.lib.format.open_memmap(
            atlas_dir / "confirmation_conditions.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(protocol.CONFIRMATION_CONDITIONS),
                len(confirmation_units),
                info.d_model,
            ),
        )
        confirmation_rows = []
        for condition_index, condition in enumerate(
            protocol.CONFIRMATION_CONDITIONS
        ):
            run_condition(
                base_model=base_model,
                layers=layers,
                condition=condition,
                rows=confirmation_units,
                units_all=units,
                cases=cases,
                clean_states=clean_states,
                depths=depths,
                source_depth=source_depth,
                query_depth=query_depth,
                preoutput_depth=preoutput_depth,
                readout_depth=readout_depth,
                tokenizer=tokenizer,
                device=device,
                model_name=args.model,
                output=confirmation_output[condition_index],
            )
            confirmation_rows.append({
                "condition": condition,
                "metrics": intervention_metrics(
                    np.asarray(
                        confirmation_output[condition_index]
                    ),
                    confirmation_units,
                    prototypes,
                ),
            })
        confirmation_output.flush()

        arrays = {
            "clean_role_states": np.asarray(clean_states),
            "clean_readout": np.asarray(clean_readout),
            "discovery_depth_scan": np.asarray(discovery_output),
            "confirmation_conditions": np.asarray(
                confirmation_output
            ),
        }
        finiteness = finite_audit(arrays)
        metrics = {
            "schema_version": "phase1029_metrics.v1",
            "model": args.model,
            "depths": depths,
            "readout_depth": readout_depth,
            "clean_four_world_readout": clean_metrics,
            "observational_role_depth": observational,
            "discovery_depth_scan": discovery_rows,
            "selection": selection,
            "confirmation_conditions": confirmation_rows,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        summary = {
            "schema_version": "phase1029_model_summary.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "model": args.model,
            "precision": "fp16",
            "quantization": "none",
            "placement": placement,
            "runtime_precision_audit": precision_audit,
            "patch_depths": depths,
            "readout_depth": readout_depth,
            "selected_source_depth": source_depth,
            "selected_query_depth": query_depth,
            "preoutput_depth": preoutput_depth,
            "selection_source": "discovery_only",
            "finiteness": finiteness,
            "elapsed_seconds": time.time() - started,
            "claim_limit": prereg["claim_limit"],
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps({
            "model": args.model,
            "clean_confirmation": clean_metrics["confirmation"],
            "selected_depths": {
                "source": source_depth,
                "query": query_depth,
                "pre_output": preoutput_depth,
            },
            "confirmation": confirmation_rows,
            "finiteness": finiteness,
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        model = tokenizer = None


if __name__ == "__main__":
    main()
