#!/usr/bin/env python3
"""Map Phase1023 semantic niches and the optional execution fork in FP16.

This is an observational atlas.  It first selects residual regions on the
discovery split, then inspects real pre-o_proj attention heads and real
pre-down_proj MLP coordinates at those frozen layers.  Confirmation data
never participates in candidate selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info
import phase1023_ecological_niche_protocol as protocol
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


ROLES = protocol.ATLAS_ROLES
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
BATCH_SIZE = {"qwen3": 32, "glm4": 16, "deepseek7b": 16}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    positions = torch.empty((len(rows), len(ROLES)), dtype=torch.long)
    for batch_index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[batch_index, :len(values)] = values
        attention_mask[batch_index, :len(values)] = 1
        for role_index, role in enumerate(ROLES):
            positions[batch_index, role_index] = int(
                row["role_positions"][role]
            )
    return (
        input_ids.to(device),
        attention_mask.to(device),
        positions,
    )


class ResidualCapture:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = defaultdict(int)
        self.handles = []

    def _hook(self, depth: int):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(value, torch.Tensor):
                raise RuntimeError("residual capture is not initialized")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            self.values[depth] = value[batch, positions, :].detach()
            self.counts[depth] += 1
            return output
        return hook

    def register(self) -> None:
        self.handles.append(
            self.model.get_input_embeddings().register_forward_hook(
                self._hook(0)
            )
        )
        for depth, layer in enumerate(self.layers, 1):
            self.handles.append(
                layer.register_forward_hook(self._hook(depth))
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.counts = defaultdict(int)

    def stacked_cpu(self) -> torch.Tensor:
        expected = set(range(len(self.layers) + 1))
        if set(self.values) != expected:
            missing = sorted(expected - set(self.values))
            raise RuntimeError(f"missing residual depths: {missing[:8]}")
        repeated = {
            depth: count for depth, count in self.counts.items() if count != 1
        }
        if repeated:
            raise RuntimeError(f"residual hook repetition: {repeated}")
        return torch.stack(
            [self.values[depth].to("cpu") for depth in sorted(expected)],
            dim=2,
        )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


class ComponentCapture:
    def __init__(self, layers, selected_layers: list[int], head_count: int):
        self.layers = layers
        self.selected_layers = selected_layers
        self.head_count = head_count
        self.positions: torch.Tensor | None = None
        self.heads: dict[int, torch.Tensor] = {}
        self.mlp: dict[int, torch.Tensor] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def _select(self, value: torch.Tensor) -> torch.Tensor:
        if self.positions is None:
            raise RuntimeError("component capture is not initialized")
        positions = self.positions.to(value.device)
        batch = torch.arange(value.shape[0], device=value.device)[:, None]
        return value[batch, positions, :].detach()

    def _head_hook(self, depth: int):
        def hook(module, args):
            selected = self._select(args[0])
            if selected.shape[-1] % self.head_count:
                raise RuntimeError("pre-o_proj width is not head aligned")
            self.heads[depth] = selected.reshape(
                selected.shape[0],
                selected.shape[1],
                self.head_count,
                selected.shape[-1] // self.head_count,
            )
            self.counts[f"head/{depth}"] += 1
        return hook

    def _mlp_hook(self, depth: int):
        def hook(module, args):
            self.mlp[depth] = self._select(args[0])
            self.counts[f"mlp/{depth}"] += 1
        return hook

    def register(self) -> None:
        for depth in self.selected_layers:
            layer = self.layers[depth - 1]
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    self._head_hook(depth)
                )
            )
            self.handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    self._mlp_hook(depth)
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.heads = {}
        self.mlp = {}
        self.counts = defaultdict(int)

    def validate(self) -> None:
        expected = {
            *{f"head/{depth}" for depth in self.selected_layers},
            *{f"mlp/{depth}" for depth in self.selected_layers},
        }
        bad = {
            key: self.counts[key]
            for key in expected
            if self.counts[key] != 1
        }
        if bad:
            raise RuntimeError(f"component hook count drift: {bad}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = None
        self.heads = {}
        self.mlp = {}


def condition_center(
    values: np.ndarray,
    cases: list[dict[str, Any]],
) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float32).copy()
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(cases):
        groups[
            (row["prompt_split"], row["task"], row["source_language"])
        ].append(index)
    for indices in groups.values():
        centered[indices] -= centered[indices].mean(axis=0, keepdims=True)
    return centered


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPS)


def case_lookup(cases: list[dict[str, Any]]) -> dict[tuple[str, ...], int]:
    return {
        (
            row["prompt_split"],
            row["task"],
            row["source_language"],
            row["concept_id"],
        ): index
        for index, row in enumerate(cases)
    }


def partition_concepts(
    cases: list[dict[str, Any]],
    partition: str,
) -> list[tuple[str, str]]:
    values = {
        (row["concept_id"], row["category"])
        for row in cases
        if row["concept_partition"] == partition
    }
    return sorted(values)


def context_pairs(mode: str) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    if mode == "cross_language":
        return [
            ((task, left), (task, right))
            for task in protocol.ATLAS_TASKS
            for left in protocol.LANGUAGES
            for right in protocol.LANGUAGES
            if left != right
        ]
    if mode == "cross_task":
        return [
            (("mention", language), ("translate", language))
            for language in protocol.LANGUAGES
        ] + [
            (("translate", language), ("mention", language))
            for language in protocol.LANGUAGES
        ]
    if mode == "joint":
        return [
            (("mention", left), ("translate", right))
            for left in protocol.LANGUAGES
            for right in protocol.LANGUAGES
            if left != right
        ] + [
            (("translate", left), ("mention", right))
            for left in protocol.LANGUAGES
            for right in protocol.LANGUAGES
            if left != right
        ]
    raise ValueError(mode)


def shifted_indices(categories: list[str]) -> np.ndarray:
    result = np.empty(len(categories), dtype=np.int64)
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, category in enumerate(categories):
        by_category[category].append(index)
    for indices in by_category.values():
        for offset, index in enumerate(indices):
            result[index] = indices[(offset + 1) % len(indices)]
    return result


def retrieval_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    *,
    partition: str,
    prompt_split: str,
    mode: str,
) -> dict[str, float]:
    lookup = case_lookup(cases)
    concepts = partition_concepts(cases, partition)
    concept_ids = [item[0] for item in concepts]
    categories = [item[1] for item in concepts]
    shifted = shifted_indices(categories)
    all_accuracy = []
    family_accuracy = []
    same_values = []
    shifted_values = []
    for (left_task, left_language), (
        right_task,
        right_language,
    ) in context_pairs(mode):
        left_indices = [
            lookup[(prompt_split, left_task, left_language, concept_id)]
            for concept_id in concept_ids
        ]
        right_indices = [
            lookup[(prompt_split, right_task, right_language, concept_id)]
            for concept_id in concept_ids
        ]
        left = normalize_rows(values[left_indices])
        right = normalize_rows(values[right_indices])
        similarity = left @ right.T
        all_accuracy.append(float(np.mean(
            np.argmax(similarity, axis=1) == np.arange(len(concept_ids))
        )))
        within_correct = []
        for index, category in enumerate(categories):
            candidates = [
                candidate
                for candidate, value in enumerate(categories)
                if value == category
            ]
            best = candidates[int(np.argmax(similarity[index, candidates]))]
            within_correct.append(best == index)
        family_accuracy.append(float(np.mean(within_correct)))
        same_values.append(float(np.mean(np.diag(similarity))))
        shifted_values.append(float(np.mean(
            similarity[np.arange(len(concept_ids)), shifted]
        )))
    return {
        "all_concept_top1": float(np.mean(all_accuracy)),
        "within_family_top1": float(np.mean(family_accuracy)),
        "same_concept_cosine": float(np.mean(same_values)),
        "shifted_within_family_cosine": float(np.mean(shifted_values)),
        "same_vs_shifted_margin": float(
            np.mean(same_values) - np.mean(shifted_values)
        ),
        "context_pair_count": len(all_accuracy),
    }


def family_transfer_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, float]:
    lookup = case_lookup(cases)
    source_concepts = partition_concepts(cases, "calibration")
    target_concepts = partition_concepts(cases, "heldout")
    family_to_index = {
        family: index for index, family in enumerate(protocol.CATEGORIES)
    }
    accuracies = []
    margins = []
    per_family_accuracy: dict[str, list[float]] = defaultdict(list)
    per_family_margin: dict[str, list[float]] = defaultdict(list)
    for source_language in protocol.LANGUAGES:
        for target_language in protocol.LANGUAGES:
            if source_language == target_language:
                continue
            centroids = []
            for family in protocol.CATEGORIES:
                indices = [
                    lookup[
                        (
                            "discovery",
                            "mention",
                            source_language,
                            concept_id,
                        )
                    ]
                    for concept_id, category in source_concepts
                    if category == family
                ]
                centroids.append(values[indices].mean(axis=0))
            centroids = normalize_rows(np.asarray(centroids))
            query_indices = [
                lookup[
                    (
                        "confirmation",
                        "translate",
                        target_language,
                        concept_id,
                    )
                ]
                for concept_id, _ in target_concepts
            ]
            queries = normalize_rows(values[query_indices])
            similarity = queries @ centroids.T
            labels = np.asarray([
                family_to_index[category]
                for _, category in target_concepts
            ])
            accuracies.append(float(np.mean(
                np.argmax(similarity, axis=1) == labels
            )))
            predictions = np.argmax(similarity, axis=1)
            true_score = similarity[np.arange(len(labels)), labels]
            wrong = similarity.copy()
            wrong[np.arange(len(labels)), labels] = -np.inf
            item_margin = true_score - np.max(wrong, axis=1)
            margins.append(float(np.mean(item_margin)))
            for family, family_index in family_to_index.items():
                mask = labels == family_index
                per_family_accuracy[family].append(float(np.mean(
                    predictions[mask] == labels[mask]
                )))
                per_family_margin[family].append(float(np.mean(
                    item_margin[mask]
                )))
    return {
        "accuracy": float(np.mean(accuracies)),
        "true_vs_best_wrong_margin": float(np.mean(margins)),
        "context_pair_count": len(accuracies),
        "per_family_accuracy": {
            family: float(np.mean(per_family_accuracy[family]))
            for family in protocol.CATEGORIES
        },
        "per_family_margin": {
            family: float(np.mean(per_family_margin[family]))
            for family in protocol.CATEGORIES
        },
    }


def all_metrics(
    raw: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    values = condition_center(raw, cases)
    result: dict[str, Any] = {}
    for label, partition, prompt_split in (
        ("discovery", "calibration", "discovery"),
        ("confirmation", "heldout", "confirmation"),
    ):
        result[label] = {
            mode: retrieval_metrics(
                values,
                cases,
                partition=partition,
                prompt_split=prompt_split,
                mode=mode,
            )
            for mode in ("cross_language", "cross_task", "joint")
        }
    result["strict_family_transfer"] = family_transfer_metrics(values, cases)
    return result


def select_layers(
    rows: list[dict[str, Any]],
    *,
    n_layers: int,
    count: int,
    minimum_fraction: float,
) -> list[int]:
    minimum = max(1, int(math.ceil(n_layers * minimum_fraction)))
    ranked = sorted(
        [row for row in rows if int(row["depth"]) >= 1],
        key=lambda row: (
            row["metrics"]["discovery"]["joint"]["within_family_top1"],
            row["metrics"]["discovery"]["joint"]["all_concept_top1"],
            row["metrics"]["strict_family_transfer"]["accuracy"],
            row["metrics"]["discovery"]["joint"][
                "same_vs_shifted_margin"
            ],
            -int(row["depth"]),
        ),
        reverse=True,
    )
    selected: list[int] = []
    for row in ranked:
        depth = int(row["depth"])
        if all(abs(depth - prior) >= minimum for prior in selected):
            selected.append(depth)
        if len(selected) == count:
            break
    if len(selected) < count:
        for row in ranked:
            depth = int(row["depth"])
            if depth not in selected:
                selected.append(depth)
            if len(selected) == count:
                break
    return sorted(selected)


def scan_residual(
    model,
    tokenizer,
    device: torch.device,
    layers,
    cases: list[dict[str, Any]],
    out_dir: Path,
    batch_size: int,
    d_model: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, list[int]]]:
    raw_path = out_dir / "residual_states.fp16.npy"
    metric_path = out_dir / "residual_metrics.jsonl"
    selection_path = out_dir / "selected_layers.json"
    if raw_path.exists() and metric_path.exists() and selection_path.exists():
        existing = np.load(raw_path, mmap_mode="r")
        expected_shape = (
            len(cases),
            len(ROLES),
            len(layers) + 1,
            d_model,
        )
        if existing.shape == expected_shape and np.any(existing[-1] != 0):
            print("[ecology-residual] reusing complete capture", flush=True)
            return (
                raw_path,
                protocol.read_jsonl(metric_path),
                protocol.read_json(selection_path),
            )
    raw = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(cases), len(ROLES), len(layers) + 1, d_model),
    )
    capture = ResidualCapture(model, layers)
    capture.register()
    try:
        offset = 0
        for batch_index, batch in enumerate(chunks(cases, batch_size), 1):
            input_ids, attention_mask, positions = make_batch(
                batch,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            capture.begin(positions)
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            values = capture.stacked_cpu().numpy().astype(
                np.float16,
                copy=False,
            )
            raw[offset:offset + len(batch)] = values
            offset += len(batch)
            if batch_index % 10 == 0:
                print(
                    f"[ecology-residual] batch={batch_index} "
                    f"cases={offset}/{len(cases)}",
                    flush=True,
                )
    finally:
        capture.close()
    raw.flush()

    metric_rows = []
    for role_index, role in enumerate(ROLES):
        for depth in range(len(layers) + 1):
            metric_rows.append({
                "schema_version": "phase1023_residual_metric.v1",
                "role": role,
                "depth": depth,
                "relative_depth": depth / max(len(layers), 1),
                "metrics": all_metrics(
                    np.asarray(raw[:, role_index, depth, :]),
                    cases,
                ),
            })
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    selection = {}
    for role in ROLES:
        selection[role] = select_layers(
            [row for row in metric_rows if row["role"] == role],
            n_layers=len(layers),
            count=int(
                prereg["component_selection"]["layers_per_role"]
            ),
            minimum_fraction=float(
                prereg["component_selection"][
                    "minimum_layer_separation_fraction"
                ]
            ),
        )
    write_jsonl(out_dir / "residual_metrics.jsonl", metric_rows)
    write_json(out_dir / "selected_layers.json", selection)
    return raw_path, metric_rows, selection


def down_projection_norms(
    layers,
    selected_layers: list[int],
    intermediate: int,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    result = {}
    provenance = {}
    for depth in selected_layers:
        module = layers[depth - 1].mlp.down_proj
        weight = module.weight
        source = "live_parameter"
        if getattr(weight, "is_meta", False):
            hook = getattr(module, "_hf_hook", None)
            weights_map = getattr(hook, "weights_map", None)
            try:
                weight = weights_map["weight"] if weights_map is not None else None
                source = "accelerate_offload_weights_map"
            except (KeyError, TypeError):
                weight = None
        if weight is None or getattr(weight, "is_meta", False):
            result[depth] = np.ones(intermediate, dtype=np.float32)
            provenance[depth] = "unavailable_neutral_rank"
            continue
        weight = weight.detach().float().cpu()
        if weight.shape[1] != intermediate:
            raise RuntimeError(
                f"down projection width drift at depth {depth}: "
                f"{tuple(weight.shape)}"
            )
        result[depth] = torch.linalg.vector_norm(weight, dim=0).numpy()
        provenance[depth] = source
    return result, provenance


def scan_components(
    model,
    tokenizer,
    device: torch.device,
    layers,
    cases: list[dict[str, Any]],
    out_dir: Path,
    batch_size: int,
    selection: dict[str, list[int]],
    head_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_layers = sorted({
        depth for values in selection.values() for depth in values
    })
    layer_index = {
        depth: index for index, depth in enumerate(selected_layers)
    }
    sample_layer = layers[selected_layers[0] - 1]
    attention_width = int(sample_layer.self_attn.o_proj.weight.shape[1])
    head_width = attention_width // head_count
    intermediate = int(sample_layer.mlp.down_proj.weight.shape[1])
    head_path = out_dir / "attention_heads.fp16.npy"
    mlp_path = out_dir / "mlp_intermediate.fp16.npy"
    head_shape = (
        len(cases),
        len(ROLES),
        len(selected_layers),
        head_count,
        head_width,
    )
    mlp_shape = (
        len(cases),
        len(ROLES),
        len(selected_layers),
        intermediate,
    )
    reuse = False
    if head_path.exists() and mlp_path.exists():
        prior_heads = np.load(head_path, mmap_mode="r")
        prior_mlp = np.load(mlp_path, mmap_mode="r")
        reuse = bool(
            prior_heads.shape == head_shape
            and prior_mlp.shape == mlp_shape
            and np.any(prior_heads[-1] != 0)
            and np.any(prior_mlp[-1] != 0)
        )
    if reuse:
        print("[ecology-components] reusing complete capture", flush=True)
        head_raw = np.load(head_path, mmap_mode="r")
        mlp_raw = np.load(mlp_path, mmap_mode="r")
    else:
        head_raw = np.lib.format.open_memmap(
            head_path,
            mode="w+",
            dtype=np.float16,
            shape=head_shape,
        )
        mlp_raw = np.lib.format.open_memmap(
            mlp_path,
            mode="w+",
            dtype=np.float16,
            shape=mlp_shape,
        )
        capture = ComponentCapture(layers, selected_layers, head_count)
        capture.register()
        try:
            offset = 0
            for batch_index, batch in enumerate(chunks(cases, batch_size), 1):
                input_ids, attention_mask, positions = make_batch(
                    batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                capture.begin(positions)
                with torch.inference_mode():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                capture.validate()
                for depth in selected_layers:
                    target = layer_index[depth]
                    head_raw[
                        offset:offset + len(batch), :, target, :, :
                    ] = capture.heads[depth].to("cpu").numpy().astype(
                        np.float16,
                        copy=False,
                    )
                    mlp_raw[
                        offset:offset + len(batch), :, target, :
                    ] = capture.mlp[depth].to("cpu").numpy().astype(
                        np.float16,
                        copy=False,
                    )
                offset += len(batch)
                if batch_index % 10 == 0:
                    print(
                        f"[ecology-components] batch={batch_index} "
                        f"cases={offset}/{len(cases)}",
                        flush=True,
                    )
        finally:
            capture.close()
        head_raw.flush()
        mlp_raw.flush()

    head_rows = []
    for role in ROLES:
        role_index = ROLE_INDEX[role]
        for depth in selection[role]:
            selected_index = layer_index[depth]
            for head in range(head_count):
                head_rows.append({
                    "schema_version": "phase1023_attention_head_metric.v1",
                    "role": role,
                    "depth": depth,
                    "relative_depth": depth / max(len(layers), 1),
                    "head": head,
                    "metrics": all_metrics(
                        np.asarray(
                            head_raw[
                                :, role_index, selected_index, head, :
                            ]
                        ),
                        cases,
                    ),
                })
    write_jsonl(out_dir / "attention_head_metrics.jsonl", head_rows)

    write_norms, write_provenance = down_projection_norms(
        layers,
        selected_layers,
        intermediate,
    )
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    top_k = int(
        prereg["component_selection"]["mlp_discovery_top_k_per_layer_role"]
    )
    neuron_rows = []
    for role in ROLES:
        role_index = ROLE_INDEX[role]
        for depth in selection[role]:
            selected_index = layer_index[depth]
            raw = np.asarray(
                mlp_raw[:, role_index, selected_index, :],
                dtype=np.float32,
            )
            centered = condition_center(raw, cases)
            discovery = profile_repeat(
                centered,
                cases,
                partition="calibration",
                prompt_split="discovery",
            )
            confirmation = profile_repeat(
                centered,
                cases,
                partition="heldout",
                prompt_split="confirmation",
            )
            norms = write_norms[depth]
            normalized_write = norms / max(float(np.mean(norms)), EPS)
            ranking = np.lexsort((
                -normalized_write,
                -discovery["family_excess"],
                -discovery["niche_excess"],
            ))
            for rank, neuron in enumerate(ranking[:top_k], 1):
                neuron = int(neuron)
                neuron_rows.append({
                    "schema_version": "phase1023_mlp_neuron_candidate.v1",
                    "role": role,
                    "depth": depth,
                    "relative_depth": depth / max(len(layers), 1),
                    "neuron": neuron,
                    "discovery_rank": rank,
                    "discovery_niche_excess": float(
                        discovery["niche_excess"][neuron]
                    ),
                    "confirmation_niche_excess": float(
                        confirmation["niche_excess"][neuron]
                    ),
                    "discovery_family_excess": float(
                        discovery["family_excess"][neuron]
                    ),
                    "confirmation_family_excess": float(
                        confirmation["family_excess"][neuron]
                    ),
                    "down_projection_norm": float(norms[neuron]),
                    "normalized_write_relevance": float(
                        normalized_write[neuron]
                    ),
                    "write_relevance_source": write_provenance[depth],
                    "confirmation_repeated": bool(
                        confirmation["niche_excess"][neuron] > 0
                        and confirmation["family_excess"][neuron] > 0
                    ),
                })
    write_jsonl(out_dir / "mlp_neuron_candidates.jsonl", neuron_rows)
    return head_rows, neuron_rows


def profile_repeat(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    *,
    partition: str,
    prompt_split: str,
) -> dict[str, np.ndarray]:
    lookup = case_lookup(cases)
    concepts = partition_concepts(cases, partition)
    concept_ids = [item[0] for item in concepts]
    categories = [item[1] for item in concepts]
    shifted = shifted_indices(categories)
    family_groups = [
        [index for index, category in enumerate(categories) if category == family]
        for family in protocol.CATEGORIES
    ]
    niche_same = []
    niche_shifted = []
    family_same = []
    family_shifted = []
    for (left_task, left_language), (
        right_task,
        right_language,
    ) in context_pairs("joint"):
        left_indices = [
            lookup[(prompt_split, left_task, left_language, concept_id)]
            for concept_id in concept_ids
        ]
        right_indices = [
            lookup[(prompt_split, right_task, right_language, concept_id)]
            for concept_id in concept_ids
        ]
        left = values[left_indices]
        right = values[right_indices]
        left_norm = left / np.maximum(
            np.linalg.norm(left, axis=0, keepdims=True),
            EPS,
        )
        right_norm = right / np.maximum(
            np.linalg.norm(right, axis=0, keepdims=True),
            EPS,
        )
        niche_same.append(np.sum(left_norm * right_norm, axis=0))
        niche_shifted.append(np.sum(
            left_norm * right_norm[shifted],
            axis=0,
        ))
        left_family = np.stack([
            left[indices].mean(axis=0) for indices in family_groups
        ])
        right_family = np.stack([
            right[indices].mean(axis=0) for indices in family_groups
        ])
        left_family /= np.maximum(
            np.linalg.norm(left_family, axis=0, keepdims=True),
            EPS,
        )
        right_family /= np.maximum(
            np.linalg.norm(right_family, axis=0, keepdims=True),
            EPS,
        )
        family_same.append(np.sum(
            left_family * right_family,
            axis=0,
        ))
        family_shifted.append(np.sum(
            left_family * np.roll(right_family, 1, axis=0),
            axis=0,
        ))
    niche_same_mean = np.mean(niche_same, axis=0)
    niche_shifted_mean = np.mean(niche_shifted, axis=0)
    family_same_mean = np.mean(family_same, axis=0)
    family_shifted_mean = np.mean(family_shifted, axis=0)
    return {
        "niche_same": niche_same_mean,
        "niche_shifted": niche_shifted_mean,
        "niche_excess": niche_same_mean - niche_shifted_mean,
        "family_same": family_same_mean,
        "family_shifted": family_shifted_mean,
        "family_excess": family_same_mean - family_shifted_mean,
    }


def pattern_template_index(row: dict[str, Any]) -> int:
    return int(row["case_key"].split(".")[2])


def pattern_label(row: dict[str, Any]) -> str:
    if row["family"] == "rare_definition":
        return row["concept_id"]
    if row["family"] == "punctuation":
        value = protocol.normalize(row["accepted_outputs"][0])
        return value[:1]
    item = row["concept_id"].split("_", 1)[1]
    return "contrast" if item.startswith("c") else "result"


def pattern_center(
    values: np.ndarray,
    cases: list[dict[str, Any]],
) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float32).copy()
    groups: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(cases):
        groups[(
            row["family"],
            row["prompt_split"],
            pattern_template_index(row),
            row["source_language"],
        )].append(index)
    for indices in groups.values():
        centered[indices] -= centered[indices].mean(axis=0, keepdims=True)
    return centered


def paired_identity_metric(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    *,
    prompt_split: str,
    concept_partition: str,
) -> dict[str, float]:
    rows = [
        (index, row)
        for index, row in enumerate(cases)
        if row["family"] == "rare_definition"
        and row["prompt_split"] == prompt_split
        and row["concept_partition"] == concept_partition
    ]
    by_template = {
        template: {
            row["concept_id"]: index
            for index, row in rows
            if pattern_template_index(row) == template
        }
        for template in (0, 1)
    }
    concept_ids = sorted(set(by_template[0]) & set(by_template[1]))
    left = normalize_rows(values[[
        by_template[0][concept_id] for concept_id in concept_ids
    ]])
    right = normalize_rows(values[[
        by_template[1][concept_id] for concept_id in concept_ids
    ]])
    similarity = left @ right.T
    shifted = np.roll(np.arange(len(concept_ids)), 1)
    return {
        "identity_top1": float(np.mean(
            np.argmax(similarity, axis=1) == np.arange(len(concept_ids))
        )),
        "same_cosine": float(np.mean(np.diag(similarity))),
        "shifted_cosine": float(np.mean(
            similarity[np.arange(len(concept_ids)), shifted]
        )),
        "same_vs_shifted_margin": float(
            np.mean(np.diag(similarity))
            - np.mean(similarity[np.arange(len(concept_ids)), shifted])
        ),
        "concept_count": len(concept_ids),
        "chance": 1 / max(len(concept_ids), 1),
    }


def pattern_class_metric(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    *,
    family: str,
    source_split: str,
    target_split: str,
) -> dict[str, float]:
    accuracies = []
    margins = []
    class_counts = []
    for language in protocol.LANGUAGES:
        source = [
            (index, row)
            for index, row in enumerate(cases)
            if row["family"] == family
            and row["prompt_split"] == source_split
            and row["source_language"] == language
            and (
                pattern_template_index(row) == 0
                if source_split == target_split
                else True
            )
        ]
        target = [
            (index, row)
            for index, row in enumerate(cases)
            if row["family"] == family
            and row["prompt_split"] == target_split
            and row["source_language"] == language
            and (
                pattern_template_index(row) == 1
                if source_split == target_split
                else True
            )
        ]
        labels = sorted({pattern_label(row) for _, row in source})
        if not source or not target or len(labels) < 2:
            continue
        centroids = []
        for label in labels:
            indices = [
                index for index, row in source
                if pattern_label(row) == label
            ]
            centroids.append(values[indices].mean(axis=0))
        centroids = normalize_rows(np.asarray(centroids))
        query = normalize_rows(values[[index for index, _ in target]])
        similarity = query @ centroids.T
        true = np.asarray([
            labels.index(pattern_label(row)) for _, row in target
        ])
        accuracies.append(float(np.mean(
            np.argmax(similarity, axis=1) == true
        )))
        true_score = similarity[np.arange(len(true)), true]
        wrong = similarity.copy()
        wrong[np.arange(len(true)), true] = -np.inf
        margins.append(float(np.mean(
            true_score - np.max(wrong, axis=1)
        )))
        class_counts.append(len(labels))
    return {
        "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "true_vs_best_wrong_margin": (
            float(np.mean(margins)) if margins else 0.0
        ),
        "language_count": len(accuracies),
        "chance": (
            float(np.mean([1 / count for count in class_counts]))
            if class_counts else 0.0
        ),
    }


def pattern_metrics(
    raw: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    values = pattern_center(raw, cases)
    return {
        "rare_definition": {
            "discovery": paired_identity_metric(
                values,
                cases,
                prompt_split="discovery",
                concept_partition="calibration",
            ),
            "confirmation": paired_identity_metric(
                values,
                cases,
                prompt_split="confirmation",
                concept_partition="heldout",
            ),
        },
        "punctuation": {
            "discovery": pattern_class_metric(
                values,
                cases,
                family="punctuation",
                source_split="discovery",
                target_split="discovery",
            ),
            "confirmation": pattern_class_metric(
                values,
                cases,
                family="punctuation",
                source_split="discovery",
                target_split="confirmation",
            ),
        },
        "connector": {
            "discovery": pattern_class_metric(
                values,
                cases,
                family="connector",
                source_split="discovery",
                target_split="discovery",
            ),
            "confirmation": pattern_class_metric(
                values,
                cases,
                family="connector",
                source_split="discovery",
                target_split="confirmation",
            ),
        },
    }


def scan_language_patterns(
    model,
    tokenizer,
    device: torch.device,
    layers,
    model_name: str,
    out_dir: Path,
    batch_size: int,
    d_model: int,
) -> dict[str, Any]:
    behavior_cases = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"behavior.{model_name}.jsonl"
    )
    cases = [
        dict(row)
        for row in behavior_cases
        if row["family"] in (
            "rare_definition",
            "punctuation",
            "connector",
        )
    ]
    for row in cases:
        pre_output = len(row["input_ids"]) - 1
        row["role_positions"] = {
            "source_end": pre_output,
            "pre_output": pre_output,
        }
    raw_path = out_dir / "pattern_residual_states.fp16.npy"
    raw = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(cases), len(layers) + 1, d_model),
    )
    capture = ResidualCapture(model, layers)
    capture.register()
    try:
        offset = 0
        for batch_index, batch in enumerate(chunks(cases, batch_size), 1):
            input_ids, attention_mask, positions = make_batch(
                batch,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            capture.begin(positions)
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            values = capture.stacked_cpu().numpy().astype(
                np.float16,
                copy=False,
            )
            raw[offset:offset + len(batch)] = values[:, 1, :, :]
            offset += len(batch)
            if batch_index % 5 == 0:
                print(
                    f"[pattern-residual] {model_name} "
                    f"cases={offset}/{len(cases)}",
                    flush=True,
                )
    finally:
        capture.close()
    raw.flush()
    rows = []
    for depth in range(len(layers) + 1):
        rows.append({
            "schema_version": "phase1023_language_pattern_metric.v1",
            "model": model_name,
            "depth": depth,
            "relative_depth": depth / max(len(layers), 1),
            "metrics": pattern_metrics(np.asarray(raw[:, depth, :]), cases),
        })
    write_jsonl(out_dir / "language_pattern_metrics.jsonl", rows)
    selected = {}
    for family in ("rare_definition", "punctuation", "connector"):
        metric_name = (
            "identity_top1"
            if family == "rare_definition" else "accuracy"
        )
        ranked = sorted(
            rows,
            key=lambda row: (
                row["metrics"][family]["discovery"][metric_name],
                row["metrics"][family]["discovery"].get(
                    "same_vs_shifted_margin",
                    row["metrics"][family]["discovery"].get(
                        "true_vs_best_wrong_margin",
                        0.0,
                    ),
                ),
                -row["depth"],
            ),
            reverse=True,
        )
        selected[family] = [
            {
                "depth": row["depth"],
                "relative_depth": row["relative_depth"],
                "discovery": row["metrics"][family]["discovery"],
                "confirmation": row["metrics"][family]["confirmation"],
            }
            for row in ranked[:3]
        ]
    result = {
        "schema_version": "phase1023_language_pattern_summary.v1",
        "model": model_name,
        "case_count": len(cases),
        "raw_path": str(raw_path.relative_to(ROOT)),
        "selected_discovery_layers": selected,
        "claim_limit": (
            "output-preceding residual repeat only; no component, causal, "
            "or complete language-pattern mechanism claim"
        ),
    }
    write_json(out_dir / "language_pattern_summary.json", result)
    return result


def ability_scan(
    model,
    tokenizer,
    device: torch.device,
    layers,
    model_name: str,
    out_dir: Path,
) -> dict[str, Any]:
    pairing_summary = protocol.read_json(
        protocol.OUT_ROOT / "pairing" / "summary.json"
    )
    authorized = bool(
        pairing_summary["ability_scan_authorized_by_model"][model_name]
    )
    if not authorized:
        result = {
            "schema_version": "phase1023_ability_scan_summary.v1",
            "model": model_name,
            "authorized": False,
            "reason": "exact semantic success/error pair gate not met",
        }
        write_json(out_dir / "ability_summary.json", result)
        return result

    behavior = protocol.read_jsonl(
        protocol.OUT_ROOT / "behavior" / model_name / "formal.jsonl"
    )
    by_key = {row["case_key"]: row for row in behavior}
    pairs = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "pairing"
        / f"ability_pairs.{model_name}.jsonl"
    )
    capture = ResidualCapture(model, layers)
    capture.register()
    accumulators: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for pair_index, pair in enumerate(pairs, 1):
            rows = [
                by_key[pair["left_case_key"]],
                by_key[pair["right_case_key"]],
            ]
            input_ids, attention_mask, positions = make_batch(
                rows,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            capture.begin(positions)
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            values = capture.stacked_cpu().float().numpy()
            delta = values[0] - values[1]
            magnitude = np.linalg.norm(delta, axis=-1)
            unit = delta / np.maximum(magnitude[..., None], EPS)
            key = (pair["pair_type"], pair["prompt_split"])
            if key not in accumulators:
                accumulators[key] = {
                    "count": 0,
                    "sum_unit": np.zeros_like(unit, dtype=np.float64),
                    "sum_magnitude": np.zeros_like(magnitude, dtype=np.float64),
                }
            accumulators[key]["count"] += 1
            accumulators[key]["sum_unit"] += unit
            accumulators[key]["sum_magnitude"] += magnitude
            if pair_index % 25 == 0:
                print(
                    f"[ability] {model_name} pairs={pair_index}/{len(pairs)}",
                    flush=True,
                )
    finally:
        capture.close()

    rows = []
    indexed = {}
    for key, accumulator in accumulators.items():
        pair_type, prompt_split = key
        count = accumulator["count"]
        mean_unit = accumulator["sum_unit"] / count
        consistency = np.linalg.norm(mean_unit, axis=-1)
        mean_magnitude = accumulator["sum_magnitude"] / count
        indexed[key] = (consistency, mean_magnitude)
        for role_index, role in enumerate(ROLES):
            for depth in range(len(layers) + 1):
                rows.append({
                    "schema_version": "phase1023_ability_residual_metric.v1",
                    "model": model_name,
                    "pair_type": pair_type,
                    "prompt_split": prompt_split,
                    "role": role,
                    "depth": depth,
                    "relative_depth": depth / max(len(layers), 1),
                    "pair_count": count,
                    "direction_consistency": float(
                        consistency[role_index, depth]
                    ),
                    "mean_delta_magnitude": float(
                        mean_magnitude[role_index, depth]
                    ),
                })
    candidates = []
    for prompt_split in protocol.PROMPT_SPLITS:
        target = indexed.get(("semantic_success_error", prompt_split))
        controls = [
            indexed.get((pair_type, prompt_split))
            for pair_type in (
                "semantic_success_success",
                "semantic_error_error",
            )
        ]
        if target is None or any(value is None for value in controls):
            continue
        target_consistency, target_magnitude = target
        control_consistency = np.maximum(
            controls[0][0],
            controls[1][0],
        )
        control_magnitude = np.maximum(
            controls[0][1],
            controls[1][1],
        )
        for role_index, role in enumerate(ROLES):
            for depth in range(len(layers) + 1):
                candidates.append({
                    "prompt_split": prompt_split,
                    "role": role,
                    "depth": depth,
                    "relative_depth": depth / max(len(layers), 1),
                    "consistency_excess": float(
                        target_consistency[role_index, depth]
                        - control_consistency[role_index, depth]
                    ),
                    "magnitude_ratio": float(
                        target_magnitude[role_index, depth]
                        / max(control_magnitude[role_index, depth], EPS)
                    ),
                })
    write_jsonl(out_dir / "ability_residual_metrics.jsonl", rows)
    write_jsonl(out_dir / "ability_candidates.jsonl", candidates)
    result = {
        "schema_version": "phase1023_ability_scan_summary.v1",
        "model": model_name,
        "authorized": True,
        "pair_count": len(pairs),
        "max_discovery_consistency_excess": max(
            (
                row["consistency_excess"]
                for row in candidates
                if row["prompt_split"] == "discovery"
            ),
            default=None,
        ),
        "max_confirmation_consistency_excess": max(
            (
                row["consistency_excess"]
                for row in candidates
                if row["prompt_split"] == "confirmation"
            ),
            default=None,
        ),
    }
    write_json(out_dir / "ability_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=protocol.MODELS,
    )
    args = parser.parse_args()
    model_name = args.model
    protocol_summary = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "summary.json"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"atlas.{model_name}.jsonl"
    )
    if len(cases) != protocol_summary["atlas_case_count_per_model"]:
        raise RuntimeError("atlas case count drift")

    out_dir = protocol.OUT_ROOT / "ecology" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, placement = load_fp16(model_name)
    layers = get_layers(model)
    info = get_model_info(model, model_name)
    head_count = int(model.config.num_attention_heads)
    runtime_audit = quantization_audit(model)
    if runtime_audit["has_quantized_modules"]:
        release_fp16(model)
        raise RuntimeError("quantized module detected")
    try:
        raw_path, residual_rows, selection = scan_residual(
            model,
            tokenizer,
            device,
            layers,
            cases,
            out_dir,
            BATCH_SIZE[model_name],
            info.d_model,
        )
        head_rows, neuron_rows = scan_components(
            model,
            tokenizer,
            device,
            layers,
            cases,
            out_dir,
            BATCH_SIZE[model_name],
            selection,
            head_count,
        )
        language_patterns = scan_language_patterns(
            model,
            tokenizer,
            device,
            layers,
            model_name,
            out_dir,
            BATCH_SIZE[model_name],
            info.d_model,
        )
        ability = ability_scan(
            model,
            tokenizer,
            device,
            layers,
            model_name,
            out_dir,
        )
        summary = {
            "schema_version": "phase1023_ecology_scan_summary.v1",
            "phase": protocol.PHASE,
            "protocol_revision": protocol.PROTOCOL_REVISION,
            "protocol_digest": protocol_summary["protocol_digest"],
            "model": model_name,
            "precision": "fp16",
            "quantization": "none",
            "placement": placement,
            "runtime_audit": runtime_audit,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "intermediate_size": info.intermediate_size,
                "head_count": head_count,
            },
            "case_count": len(cases),
            "residual_raw_path": str(raw_path.relative_to(ROOT)),
            "selected_layers": selection,
            "residual_metric_count": len(residual_rows),
            "attention_head_metric_count": len(head_rows),
            "mlp_candidate_count": len(neuron_rows),
            "confirmed_mlp_candidate_count": sum(
                row["confirmation_repeated"] for row in neuron_rows
            ),
            "language_pattern_scan": language_patterns,
            "ability_scan": ability,
        }
        write_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        release_fp16(model)


if __name__ == "__main__":
    main()
