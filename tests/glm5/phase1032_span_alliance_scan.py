#!/usr/bin/env python3
"""Run the Phase1032 span-aware source/query alliance atlas in FP16."""

from __future__ import annotations

import argparse
import importlib
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
import phase1032_span_alliance_protocol as protocol
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)


BATCH_SIZE = {"qwen3": 24, "glm4": 8, "deepseek7b": 8}
CAPTURE_ROLES = (
    "concept_a",
    "concept_b",
    "query_nonce",
    "query_clause",
)
MAX_SPAN = {
    "concept_a": 2,
    "concept_b": 2,
    "query_nonce": 2,
    "query_clause": 8,
}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def replace_output(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def make_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    span_positions = {
        role: torch.zeros(
            (len(rows), MAX_SPAN[role]), dtype=torch.long
        )
        for role in CAPTURE_ROLES
    }
    span_masks = {
        role: torch.zeros(
            (len(rows), MAX_SPAN[role]), dtype=torch.bool
        )
        for role in CAPTURE_ROLES
    }
    pre_positions = torch.empty(len(rows), dtype=torch.long)

    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[index, :len(values)] = values
        mask[index, :len(values)] = 1
        for role in CAPTURE_ROLES:
            start, end = (
                int(value) for value in row["role_spans"][role]
            )
            positions = list(range(start, end + 1))
            if len(positions) > MAX_SPAN[role]:
                raise RuntimeError(
                    f"{role} span exceeds frozen maximum: {positions}"
                )
            span_positions[role][index, :len(positions)] = torch.tensor(
                positions, dtype=torch.long
            )
            if len(positions) < MAX_SPAN[role]:
                span_positions[role][index, len(positions):] = positions[-1]
            span_masks[role][index, :len(positions)] = True
        pre_positions[index] = int(row["role_spans"]["pre_output"][1])

    return (
        ids.to(device),
        mask.to(device),
        span_positions,
        span_masks,
        pre_positions,
    )


def gather_hidden(
    hidden: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    positions = positions.to(hidden.device)
    batch = torch.arange(
        hidden.shape[0], device=hidden.device
    )[:, None]
    return hidden[batch, positions, :].detach()


class SpanCleanCapture:
    def __init__(
        self,
        layers,
        source_depth: int,
        query_depth: int,
        readout_depth: int,
    ):
        self.layers = layers
        self.source_depth = source_depth
        self.query_depth = query_depth
        self.readout_depth = readout_depth
        self.span_positions: dict[str, torch.Tensor] = {}
        self.pre_positions: torch.Tensor | None = None
        self.values: dict[str, torch.Tensor] = {}
        self.readout: torch.Tensor | None = None
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def _depth_hook(self, depth: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if depth == self.source_depth:
                for role in ("concept_a", "concept_b"):
                    self.values[role] = gather_hidden(
                        hidden, self.span_positions[role]
                    )
            if depth == self.query_depth:
                for role in ("query_nonce", "query_clause"):
                    self.values[role] = gather_hidden(
                        hidden, self.span_positions[role]
                    )
            self.counts[f"depth/{depth}"] += 1
            return output
        return hook

    def _readout_hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.pre_positions is None:
            raise RuntimeError("clean readout positions missing")
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        self.readout = hidden[
            batch,
            self.pre_positions.to(hidden.device),
            :,
        ].detach()
        self.counts["readout"] += 1
        return output

    def register(self) -> None:
        for depth in sorted({self.source_depth, self.query_depth}):
            self.handles.append(
                self.layers[depth - 1].register_forward_hook(
                    self._depth_hook(depth)
                )
            )
        self.handles.append(
            self.layers[
                self.readout_depth - 1
            ].register_forward_hook(self._readout_hook)
        )

    def begin(
        self,
        span_positions: dict[str, torch.Tensor],
        pre_positions: torch.Tensor,
    ) -> None:
        self.span_positions = span_positions
        self.pre_positions = pre_positions
        self.values = {}
        self.readout = None
        self.counts = defaultdict(int)

    def stacked(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        expected = {
            f"depth/{depth}": 1
            for depth in sorted({self.source_depth, self.query_depth})
        } | {"readout": 1}
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"clean hook count drift: {dict(self.counts)}"
            )
        if set(self.values) != set(CAPTURE_ROLES):
            raise RuntimeError(
                f"clean role drift: {sorted(self.values)}"
            )
        if self.readout is None:
            raise RuntimeError("clean readout missing")
        return (
            {
                role: value.to("cpu")
                for role, value in self.values.items()
            },
            self.readout.to("cpu"),
        )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


PatchSpec = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class SpanPatchCapture:
    def __init__(self, layers, depths: list[int], readout_depth: int):
        self.layers = layers
        self.depths = sorted(set(depths))
        self.readout_depth = readout_depth
        self.specs: dict[int, list[PatchSpec]] = {}
        self.pre_positions: torch.Tensor | None = None
        self.readout: torch.Tensor | None = None
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def _patch_hook(self, depth: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            for positions, replacement, mask in self.specs.get(depth, []):
                positions = positions.to(hidden.device)
                replacement = replacement.to(
                    hidden.device, dtype=hidden.dtype
                )
                mask = mask.to(hidden.device)
                batch = torch.arange(
                    hidden.shape[0], device=hidden.device
                )[:, None].expand_as(positions)
                patched[
                    batch[mask],
                    positions[mask],
                    :,
                ] = replacement[mask]
            self.counts[f"patch/{depth}"] += 1
            return replace_output(output, patched)
        return hook

    def _readout_hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.pre_positions is None:
            raise RuntimeError("patched readout positions missing")
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
        specs: dict[int, list[PatchSpec]],
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


def candidate_logits_from_base_output(
    model,
    base_output,
    pre_positions: torch.Tensor,
    candidate_ids: list[int],
) -> torch.Tensor:
    hidden = base_output[0]
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    selected = hidden[
        batch,
        pre_positions.to(hidden.device),
        :,
    ]
    head = model.get_output_embeddings()
    weight = getattr(head, "weight", None)
    if weight is not None:
        selected = selected.to(weight.device)
    logits = head(selected)
    ids = torch.tensor(
        candidate_ids, dtype=torch.long, device=logits.device
    )
    return logits.index_select(-1, ids).detach().float().to("cpu")


def candidate_logits_from_full_output(
    full_output,
    pre_positions: torch.Tensor,
    candidate_ids: list[int],
) -> torch.Tensor:
    logits = full_output.logits
    batch = torch.arange(logits.shape[0], device=logits.device)
    selected = logits[
        batch,
        pre_positions.to(logits.device),
        :,
    ]
    ids = torch.tensor(
        candidate_ids, dtype=torch.long, device=logits.device
    )
    return selected.index_select(
        -1, ids
    ).detach().float().to("cpu")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def scope_indices(
    rows: list[dict[str, Any]],
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {
        "all": list(range(len(rows))),
    }
    for template in range(len(protocol.TEMPLATES)):
        result[f"template_{template}"] = [
            index
            for index, row in enumerate(rows)
            if int(row["template_index"]) == template
        ]
    for bank in protocol.CONCEPT_BANKS:
        result[f"bank_{bank}"] = [
            index
            for index, row in enumerate(rows)
            if str(row["bank_name"]) == bank
        ]
    for template in range(len(protocol.TEMPLATES)):
        for bank in protocol.CONCEPT_BANKS:
            result[f"template_{template}/bank_{bank}"] = [
                index
                for index, row in enumerate(rows)
                if int(row["template_index"]) == template
                and str(row["bank_name"]) == bank
            ]
    return result


def prototype_map(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
    scheme: str,
) -> dict[tuple[int, int, str], np.ndarray]:
    result = {}
    for held_surface in range(len(protocol.NONCE_PAIRS)):
        for current_template in range(len(protocol.TEMPLATES)):
            for bank in protocol.CONCEPT_BANKS:
                values = np.empty(
                    (
                        len(protocol.CATEGORY_LABELS),
                        clean_readout.shape[-1],
                    ),
                    dtype=np.float32,
                )
                for concept_index in range(
                    len(protocol.CATEGORY_LABELS)
                ):
                    indices = []
                    for row in cases:
                        if (
                            str(row["bank_name"]) != bank
                            or int(row["expected_index"]) != concept_index
                            or int(row["surface_index"]) == held_surface
                        ):
                            continue
                        template = int(row["template_index"])
                        if (
                            scheme == "within_template"
                            and template != current_template
                        ):
                            continue
                        if (
                            scheme == "cross_template"
                            and template == current_template
                        ):
                            continue
                        indices.append(int(row["case_index"]))
                    if not indices:
                        raise RuntimeError(
                            f"empty prototype {scheme}/"
                            f"{held_surface}/{current_template}/"
                            f"{bank}/{concept_index}"
                        )
                    values[concept_index] = np.asarray(
                        clean_readout[indices], dtype=np.float32
                    ).mean(axis=0)
                result[(held_surface, current_template, bank)] = (
                    normalize_rows(values)
                )
    return result


def readout_scores(
    values: np.ndarray,
    rows: list[dict[str, Any]],
    prototypes: dict[tuple[int, int, str], np.ndarray],
) -> np.ndarray:
    result = np.empty(
        (len(rows), len(protocol.CATEGORY_LABELS)),
        dtype=np.float32,
    )
    normalized = normalize_rows(values)
    for index, row in enumerate(rows):
        key = (
            int(row["surface_index"]),
            int(row["template_index"]),
            str(row["bank_name"]),
        )
        result[index] = normalized[index] @ prototypes[key].T
    return result


def summarize_scores(
    scores: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    clean_expected: bool = False,
) -> dict[str, Any]:
    result = {}
    for scope, indices in scope_indices(rows).items():
        if not indices:
            continue
        selected_scores = scores[indices]
        selected_rows = [rows[index] for index in indices]
        finite = np.all(np.isfinite(selected_scores), axis=-1)
        total_count = len(indices)
        selected_scores = selected_scores[finite]
        selected_rows = [
            row
            for row, keep in zip(selected_rows, finite.tolist())
            if keep
        ]
        coverage = {
            "total_count": total_count,
            "count": len(selected_rows),
            "finite_row_rate": float(np.mean(finite)),
            "dropped_nonfinite_rows": int(np.sum(~finite)),
        }
        if not selected_rows:
            result[scope] = coverage | {
                "metrics_available": False,
            }
            continue
        predicted = np.argmax(selected_scores, axis=-1)
        if clean_expected:
            expected = np.asarray(
                [int(row["expected_index"]) for row in selected_rows]
            )
            wrong = np.array(selected_scores, copy=True)
            wrong[np.arange(len(expected)), expected] = -np.inf
            margins = (
                selected_scores[np.arange(len(expected)), expected]
                - np.max(wrong, axis=-1)
            )
            result[scope] = coverage | {
                "metrics_available": True,
                "expected_top1": float(np.mean(predicted == expected)),
                "expected_vs_best_wrong_margin_mean": float(
                    np.mean(margins)
                ),
                "chance": 1.0 / len(protocol.CATEGORY_LABELS),
            }
            continue

        base = np.asarray(
            [int(row["target_index"]) for row in selected_rows]
        )
        alternate = np.asarray(
            [int(row["donor_index"]) for row in selected_rows]
        )
        scrambled = np.asarray(
            [int(row["scrambled_donor_index"]) for row in selected_rows]
        )
        arange = np.arange(len(selected_rows))
        result[scope] = coverage | {
            "metrics_available": True,
            "base_top1": float(np.mean(predicted == base)),
            "alternate_top1": float(np.mean(predicted == alternate)),
            "scrambled_top1": float(np.mean(predicted == scrambled)),
            "other_top1": float(np.mean(
                (predicted != base)
                & (predicted != alternate)
                & (predicted != scrambled)
            )),
            "alternate_vs_base_margin_mean": float(np.mean(
                selected_scores[arange, alternate]
                - selected_scores[arange, base]
            )),
            "base_vs_alternate_margin_mean": float(np.mean(
                selected_scores[arange, base]
                - selected_scores[arange, alternate]
            )),
            "chance": 1.0 / len(protocol.CATEGORY_LABELS),
        }
    return result


def condition_metrics(
    readout_values: np.ndarray,
    candidate_logits: np.ndarray,
    units: list[dict[str, Any]],
    prototypes_by_scheme: dict[
        str,
        dict[tuple[int, int, str], np.ndarray],
    ],
) -> dict[str, Any]:
    readout = {}
    for scheme, prototypes in prototypes_by_scheme.items():
        scores = readout_scores(readout_values, units, prototypes)
        readout[scheme] = summarize_scores(scores, units)
    return {
        "prototype_readout": readout,
        "next_token_candidate_logits": summarize_scores(
            np.asarray(candidate_logits, dtype=np.float32),
            units,
        ),
    }


def clean_metrics(
    clean_readout: np.ndarray,
    clean_logits: np.ndarray,
    cases: list[dict[str, Any]],
    prototypes_by_scheme: dict[
        str,
        dict[tuple[int, int, str], np.ndarray],
    ],
) -> dict[str, Any]:
    readout = {}
    for scheme, prototypes in prototypes_by_scheme.items():
        scores = readout_scores(clean_readout, cases, prototypes)
        readout[scheme] = summarize_scores(
            scores, cases, clean_expected=True
        )
    return {
        "prototype_readout": readout,
        "next_token_candidate_logits": summarize_scores(
            np.asarray(clean_logits, dtype=np.float32),
            cases,
            clean_expected=True,
        ),
    }


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


def roles(rows: list[dict[str, Any]], field: str) -> list[str]:
    return [str(row[field]) for row in rows]


def target_span_spec(
    target_cases: list[dict[str, Any]],
    role_names: list[str],
    *,
    endpoint: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [
        int(row["role_spans"][role][1])
        - int(row["role_spans"][role][0])
        + 1
        for row, role in zip(target_cases, role_names)
    ]
    width = 1 if endpoint else max(lengths)
    positions = torch.zeros(
        (len(target_cases), width), dtype=torch.long
    )
    mask = torch.zeros(
        (len(target_cases), width), dtype=torch.bool
    )
    for index, (row, role, length) in enumerate(
        zip(target_cases, role_names, lengths)
    ):
        start, end = (
            int(value) for value in row["role_spans"][role]
        )
        values = [end] if endpoint else list(range(start, end + 1))
        positions[index, :len(values)] = torch.tensor(
            values, dtype=torch.long
        )
        if len(values) < width:
            positions[index, len(values):] = values[-1]
        mask[index, :len(values)] = True
        if not endpoint and len(values) != length:
            raise RuntimeError("target span length drift")
    return positions, mask


def role_length(
    case: dict[str, Any],
    role: str,
) -> int:
    start, end = (int(value) for value in case["role_spans"][role])
    return end - start + 1


def gather_donor_states(
    clean_states: dict[str, np.ndarray],
    donor_cases: list[dict[str, Any]],
    donor_roles: list[str],
    *,
    endpoint: bool,
    width: int,
) -> torch.Tensor:
    d_model = next(iter(clean_states.values())).shape[-1]
    output = np.zeros(
        (len(donor_cases), width, d_model), dtype=np.float16
    )
    for index, (case, role) in enumerate(
        zip(donor_cases, donor_roles)
    ):
        case_index = int(case["case_index"])
        length = role_length(case, role)
        values = np.asarray(
            clean_states[role][case_index, :length, :]
        )
        if endpoint:
            output[index, 0, :] = values[-1]
        else:
            output[index, :length, :] = values
    return torch.from_numpy(output)


def operation(
    *,
    depth: int,
    target_cases: list[dict[str, Any]],
    target_roles: list[str],
    donor_cases: list[dict[str, Any]],
    donor_roles: list[str],
    clean_states: dict[str, np.ndarray],
    endpoint: bool,
) -> tuple[int, PatchSpec]:
    positions, mask = target_span_spec(
        target_cases, target_roles, endpoint=endpoint
    )
    replacements = gather_donor_states(
        clean_states,
        donor_cases,
        donor_roles,
        endpoint=endpoint,
        width=positions.shape[1],
    )
    donor_lengths = [
        1 if endpoint else role_length(case, role)
        for case, role in zip(donor_cases, donor_roles)
    ]
    target_lengths = [
        int(value) for value in mask.sum(dim=1).tolist()
    ]
    if donor_lengths != target_lengths:
        raise RuntimeError(
            f"span alignment drift: {donor_lengths} != {target_lengths}"
        )
    return depth, (positions, replacements, mask)


def condition_operations(
    condition: str,
    rows: list[dict[str, Any]],
    *,
    units_all: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    clean_states: dict[str, np.ndarray],
    source_depth: int,
    query_depth: int,
) -> dict[int, list[PatchSpec]]:
    target_indices = case_indices(
        rows, "00", units_all=units_all
    )
    target_cases = [cases[index] for index in target_indices]
    b_cases = [
        cases[index]
        for index in case_indices(rows, "10", units_all=units_all)
    ]
    q_cases = [
        cases[index]
        for index in case_indices(rows, "01", units_all=units_all)
    ]
    bq_cases = [
        cases[index]
        for index in case_indices(rows, "11", units_all=units_all)
    ]
    scrambled_b_cases = [
        cases[index]
        for index in case_indices(
            rows, "10", units_all=units_all, scrambled=True
        )
    ]
    scrambled_q_cases = [
        cases[index]
        for index in case_indices(
            rows, "01", units_all=units_all, scrambled=True
        )
    ]
    source_a = ["concept_a"] * len(rows)
    source_b = ["concept_b"] * len(rows)
    selected = roles(rows, "selected_concept_role")
    unselected = roles(rows, "unselected_concept_role")
    definition_a = ["definition_nonce_a"] * len(rows)
    definition_b = ["definition_nonce_b"] * len(rows)
    query_nonce = ["query_nonce"] * len(rows)
    query_clause = ["query_clause"] * len(rows)
    operations: list[tuple[
        int,
        list[str],
        list[dict[str, Any]],
        list[str],
        bool,
    ]] = []

    if condition == "selected_source_endpoint_b":
        operations.append((
            source_depth, selected, b_cases, selected, True
        ))
    elif condition == "selected_source_span_b":
        operations.append((
            source_depth, selected, b_cases, selected, False
        ))
    elif condition == "unselected_source_span_b":
        operations.append((
            source_depth, unselected, b_cases, unselected, False
        ))
    elif condition == "source_pair_endpoint_b":
        operations.extend([
            (source_depth, source_a, b_cases, source_a, True),
            (source_depth, source_b, b_cases, source_b, True),
        ])
    elif condition == "source_pair_span_b":
        operations.extend([
            (source_depth, source_a, b_cases, source_a, False),
            (source_depth, source_b, b_cases, source_b, False),
        ])
    elif condition == "source_pair_span_scrambled":
        operations.extend([
            (
                source_depth,
                source_a,
                scrambled_b_cases,
                source_a,
                False,
            ),
            (
                source_depth,
                source_b,
                scrambled_b_cases,
                source_b,
                False,
            ),
        ])
    elif condition == "source_pair_span_wrong_position":
        operations.extend([
            (
                source_depth,
                definition_a,
                b_cases,
                source_a,
                False,
            ),
            (
                source_depth,
                definition_b,
                b_cases,
                source_b,
                False,
            ),
        ])
    elif condition == "source_pair_span_self":
        operations.extend([
            (
                source_depth,
                source_a,
                target_cases,
                source_a,
                False,
            ),
            (
                source_depth,
                source_b,
                target_cases,
                source_b,
                False,
            ),
        ])
    elif condition in {
        "query_endpoint_q",
        "query_nonce_span_q",
        "query_clause_span_q",
        "query_endpoint_bq",
        "query_nonce_span_bq",
        "query_clause_span_bq",
        "query_clause_span_scrambled",
        "query_clause_span_self",
    }:
        donor = q_cases
        donor_roles = query_nonce
        target_roles = query_nonce
        endpoint = condition.startswith("query_endpoint")
        if "clause" in condition:
            donor_roles = query_clause
            target_roles = query_clause
        if condition.endswith("_bq"):
            donor = bq_cases
        elif condition == "query_clause_span_scrambled":
            donor = scrambled_q_cases
        elif condition == "query_clause_span_self":
            donor = target_cases
        operations.append((
            query_depth,
            target_roles,
            donor,
            donor_roles,
            endpoint,
        ))
    elif condition.startswith("source_pair_span_plus_query_"):
        operations.extend([
            (source_depth, source_a, b_cases, source_a, False),
            (source_depth, source_b, b_cases, source_b, False),
        ])
        donor = bq_cases if condition.endswith("_bq") else q_cases
        if "query_nonce" in condition:
            operations.append((
                query_depth,
                query_nonce,
                donor,
                query_nonce,
                False,
            ))
        elif "query_clause" in condition:
            operations.append((
                query_depth,
                query_clause,
                donor,
                query_clause,
                False,
            ))
        else:
            raise ValueError(condition)
    else:
        raise ValueError(condition)

    result: dict[int, list[PatchSpec]] = defaultdict(list)
    for depth, target_roles, donor, donor_roles, endpoint in operations:
        op_depth, spec = operation(
            depth=depth,
            target_cases=target_cases,
            target_roles=target_roles,
            donor_cases=donor,
            donor_roles=donor_roles,
            clean_states=clean_states,
            endpoint=endpoint,
        )
        result[op_depth].append(spec)
    return dict(result)


def valid_unit_indices(
    condition: str,
    units: list[dict[str, Any]],
) -> list[int]:
    if condition == "source_pair_span_wrong_position":
        return [
            index
            for index, row in enumerate(units)
            if row["bank_name"] == "double"
        ]
    return list(range(len(units)))


def run_condition(
    *,
    model,
    base_model,
    layers,
    condition: str,
    units: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    clean_states: dict[str, np.ndarray],
    source_depth: int,
    query_depth: int,
    readout_depth: int,
    tokenizer,
    device: torch.device,
    model_name: str,
    candidate_ids: list[int],
    output_readout: np.ndarray,
    output_logits: np.ndarray,
) -> list[int]:
    valid_indices = valid_unit_indices(condition, units)
    selected_units = [units[index] for index in valid_indices]
    depths = []
    if (
        "source" in condition
        or condition.startswith("selected_")
        or condition.startswith("unselected_")
    ):
        depths.append(source_depth)
    if "query_" in condition:
        depths.append(query_depth)
    depths = sorted(set(depths))
    capture = SpanPatchCapture(layers, depths, readout_depth)
    capture.register()
    try:
        for batch_number, index_batch in enumerate(
            chunks(valid_indices, BATCH_SIZE[model_name]), 1
        ):
            row_batch = [units[index] for index in index_batch]
            target_cases = [
                cases[int(row["world_case_indices"]["00"])]
                for row in row_batch
            ]
            (
                input_ids,
                attention_mask,
                _,
                _,
                pre_positions,
            ) = make_batch(
                target_cases,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            specs = condition_operations(
                condition,
                row_batch,
                units_all=units,
                cases=cases,
                clean_states=clean_states,
                source_depth=source_depth,
                query_depth=query_depth,
            )
            capture.begin(specs, pre_positions)
            with torch.inference_mode():
                if model_name == "qwen3":
                    base_output = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    logits = candidate_logits_from_base_output(
                        model,
                        base_output,
                        pre_positions,
                        candidate_ids,
                    )
                else:
                    full_output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    logits = candidate_logits_from_full_output(
                        full_output,
                        pre_positions,
                        candidate_ids,
                    )
            output_readout[index_batch] = (
                capture.value().numpy().astype(np.float16, copy=False)
            )
            output_logits[index_batch] = logits.numpy().astype(
                np.float32, copy=False
            )
            if batch_number % 32 == 0:
                print(
                    f"[phase{protocol.PHASE}] {model_name} {condition} "
                    f"units={min(batch_number * BATCH_SIZE[model_name], len(selected_units))}/"
                    f"{len(selected_units)}",
                    flush=True,
                )
    finally:
        capture.close()
    return valid_indices


def margins(
    scores: np.ndarray,
    rows: list[dict[str, Any]],
) -> np.ndarray:
    index = np.arange(len(rows))
    alternate = np.asarray(
        [int(row["donor_index"]) for row in rows]
    )
    base = np.asarray([int(row["target_index"]) for row in rows])
    return scores[index, alternate] - scores[index, base]


def paired_summary(
    difference: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for scope, indices in scope_indices(rows).items():
        values = np.asarray(difference[indices], dtype=np.float32)
        total_count = len(values)
        values = values[np.isfinite(values)]
        if not len(values):
            result[scope] = {
                "total_count": total_count,
                "count": 0,
                "finite_rate": 0.0,
                "mean": None,
                "median": None,
                "positive_rate": None,
                "negative_rate": None,
            }
            continue
        result[scope] = {
            "total_count": total_count,
            "count": len(values),
            "finite_rate": float(len(values) / total_count),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "positive_rate": float(np.mean(values > 0)),
            "negative_rate": float(np.mean(values < 0)),
        }
    return result


def comparison_metrics(
    *,
    units: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    clean_readout: np.ndarray,
    clean_logits: np.ndarray,
    condition_readouts: np.ndarray,
    condition_logits: np.ndarray,
    prototypes: dict[tuple[int, int, str], np.ndarray],
) -> dict[str, Any]:
    condition_index = {
        name: index for index, name in enumerate(protocol.CONDITIONS)
    }
    readout_score_map = {
        name: readout_scores(
            np.asarray(condition_readouts[index]),
            units,
            prototypes,
        )
        for name, index in condition_index.items()
        if name != "source_pair_span_wrong_position"
    }
    logit_score_map = {
        name: np.asarray(condition_logits[index], dtype=np.float32)
        for name, index in condition_index.items()
        if name != "source_pair_span_wrong_position"
    }
    comparisons = {
        "selected_source_full_minus_endpoint": (
            "selected_source_span_b",
            "selected_source_endpoint_b",
        ),
        "source_pair_full_minus_endpoint": (
            "source_pair_span_b",
            "source_pair_endpoint_b",
        ),
        "query_nonce_minus_endpoint_q": (
            "query_nonce_span_q",
            "query_endpoint_q",
        ),
        "query_clause_minus_nonce_q": (
            "query_clause_span_q",
            "query_nonce_span_q",
        ),
        "query_nonce_minus_endpoint_bq": (
            "query_nonce_span_bq",
            "query_endpoint_bq",
        ),
        "query_clause_minus_nonce_bq": (
            "query_clause_span_bq",
            "query_nonce_span_bq",
        ),
    }
    result = {}
    for label, (left, right) in comparisons.items():
        result[label] = {
            "within_template_prototype_margin_gain": paired_summary(
                margins(readout_score_map[left], units)
                - margins(readout_score_map[right], units),
                units,
            ),
            "next_token_logit_margin_gain": paired_summary(
                margins(logit_score_map[left], units)
                - margins(logit_score_map[right], units),
                units,
            ),
        }

    target_case_indices = [
        int(row["world_case_indices"]["00"]) for row in units
    ]
    clean_target_readout = np.asarray(
        clean_readout[target_case_indices]
    )
    clean_target_logits = np.asarray(
        clean_logits[target_case_indices], dtype=np.float32
    )
    result["self_patch_numerical_audit"] = {
        "source_readout_max_abs": float(np.nanmax(np.abs(
            np.asarray(
                condition_readouts[
                    condition_index["source_pair_span_self"]
                ],
                dtype=np.float32,
            )
            - clean_target_readout.astype(np.float32)
        ))),
        "source_logits_max_abs": float(np.nanmax(np.abs(
            np.asarray(
                condition_logits[
                    condition_index["source_pair_span_self"]
                ],
                dtype=np.float32,
            )
            - clean_target_logits
        ))),
        "query_readout_max_abs": float(np.nanmax(np.abs(
            np.asarray(
                condition_readouts[
                    condition_index["query_clause_span_self"]
                ],
                dtype=np.float32,
            )
            - clean_target_readout.astype(np.float32)
        ))),
        "query_logits_max_abs": float(np.nanmax(np.abs(
            np.asarray(
                condition_logits[
                    condition_index["query_clause_span_self"]
                ],
                dtype=np.float32,
            )
            - clean_target_logits
        ))),
    }

    single_indices = [
        index
        for index, row in enumerate(units)
        if row["bank_name"] == "single"
    ]
    result["single_token_endpoint_full_identity_audit"] = {
        "selected_source_readout_max_abs": float(np.nanmax(np.abs(
            np.asarray(
                condition_readouts[
                    condition_index["selected_source_span_b"],
                    single_indices,
                ],
                dtype=np.float32,
            )
            - np.asarray(
                condition_readouts[
                    condition_index["selected_source_endpoint_b"],
                    single_indices,
                ],
                dtype=np.float32,
            )
        ))),
        "selected_source_logits_max_abs": float(np.nanmax(np.abs(
            np.asarray(
                condition_logits[
                    condition_index["selected_source_span_b"],
                    single_indices,
                ],
                dtype=np.float32,
            )
            - np.asarray(
                condition_logits[
                    condition_index["selected_source_endpoint_b"],
                    single_indices,
                ],
                dtype=np.float32,
            )
        ))),
        "source_pair_readout_max_abs": float(np.nanmax(np.abs(
            np.asarray(
                condition_readouts[
                    condition_index["source_pair_span_b"],
                    single_indices,
                ],
                dtype=np.float32,
            )
            - np.asarray(
                condition_readouts[
                    condition_index["source_pair_endpoint_b"],
                    single_indices,
                ],
                dtype=np.float32,
            )
        ))),
        "source_pair_logits_max_abs": float(np.nanmax(np.abs(
            np.asarray(
                condition_logits[
                    condition_index["source_pair_span_b"],
                    single_indices,
                ],
                dtype=np.float32,
            )
            - np.asarray(
                condition_logits[
                    condition_index["source_pair_endpoint_b"],
                    single_indices,
                ],
                dtype=np.float32,
            )
        ))),
    }

    clean_readout_scores = readout_scores(
        clean_target_readout, units, prototypes
    )
    clean_readout_margin = margins(clean_readout_scores, units)
    clean_logit_margin = margins(clean_target_logits, units)
    source_readout_margin = margins(
        readout_score_map["source_pair_span_b"], units
    )
    source_logit_margin = margins(
        logit_score_map["source_pair_span_b"], units
    )
    for suffix, query_name in (
        ("query_nonce_q", "query_nonce_span_q"),
        ("query_clause_q", "query_clause_span_q"),
        ("query_nonce_bq", "query_nonce_span_bq"),
        ("query_clause_bq", "query_clause_span_bq"),
    ):
        combined_name = f"source_pair_span_plus_{suffix}"
        query_readout_margin = margins(
            readout_score_map[query_name], units
        )
        query_logit_margin = margins(
            logit_score_map[query_name], units
        )
        combined_readout_margin = margins(
            readout_score_map[combined_name], units
        )
        combined_logit_margin = margins(
            logit_score_map[combined_name], units
        )
        result[f"composition/{suffix}"] = {
            "within_template_prototype_interaction": paired_summary(
                (
                    combined_readout_margin
                    - source_readout_margin
                    - query_readout_margin
                    + clean_readout_margin
                ),
                units,
            ),
            "next_token_logit_interaction": paired_summary(
                (
                    combined_logit_margin
                    - source_logit_margin
                    - query_logit_margin
                    + clean_logit_margin
                ),
                units,
            ),
        }
    return result


def finite_audit(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for name, values in arrays.items():
        array = np.asarray(values)
        finite = np.isfinite(array)
        if array.ndim >= 2:
            finite_rows = np.all(
                finite.reshape(-1, array.shape[-1]), axis=-1
            )
            finite_row_rate = float(np.mean(finite_rows))
        else:
            finite_row_rate = float(np.mean(finite))
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "all_finite": bool(finite.all()),
            "finite_rate": float(np.mean(finite)),
            "finite_row_rate": finite_row_rate,
        }
    result["all_arrays_finite"] = all(
        row["all_finite"] for row in result.values()
        if isinstance(row, dict)
    )
    result["all_state_arrays_finite"] = all(
        row["all_finite"]
        for name, row in result.items()
        if isinstance(row, dict) and "logits" not in name
    )
    result["candidate_logit_finite_row_rates"] = {
        name: row["finite_row_rate"]
        for name, row in result.items()
        if isinstance(row, dict) and "logits" in name
    }
    return result


def main() -> None:
    global protocol
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    parser.add_argument(
        "--protocol-module",
        default="phase1032_span_alliance_protocol",
    )
    args = parser.parse_args()
    if args.protocol_module != "phase1032_span_alliance_protocol":
        protocol = importlib.import_module(args.protocol_module)

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
    selected = prereg[
        "selected_depths_frozen_from_phase1029"
    ][args.model]
    source_depth = int(selected["source"])
    query_depth = int(selected["query"])
    readout_depth = int(selected["readout"])
    candidate_ids = [
        int(value)
        for value in prereg["model_tokenization_audits"][
            args.model
        ]["candidate_token_ids"]
    ]
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
        if max(source_depth, query_depth) >= readout_depth:
            raise RuntimeError("patch depth must precede readout depth")
        base_model = model.model

        clean_states = {
            role: np.lib.format.open_memmap(
                atlas_dir / f"clean_{role}.fp16.npy",
                mode="w+",
                dtype=np.float16,
                shape=(
                    len(cases),
                    MAX_SPAN[role],
                    info.d_model,
                ),
            )
            for role in CAPTURE_ROLES
        }
        clean_readout = np.lib.format.open_memmap(
            atlas_dir / "clean_readout.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(cases), info.d_model),
        )
        clean_logits = np.lib.format.open_memmap(
            atlas_dir / "clean_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases), len(protocol.CATEGORY_LABELS)),
        )

        capture = SpanCleanCapture(
            layers,
            source_depth,
            query_depth,
            readout_depth,
        )
        capture.register()
        manual_full_logit_max_abs: float | None = None
        try:
            offset = 0
            for batch_number, row_batch in enumerate(
                chunks(cases, BATCH_SIZE[args.model]), 1
            ):
                (
                    input_ids,
                    attention_mask,
                    span_positions,
                    _,
                    pre_positions,
                ) = make_batch(
                    row_batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                capture.begin(span_positions, pre_positions)
                with torch.inference_mode():
                    if args.model == "qwen3":
                        base_output = base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        logits = candidate_logits_from_base_output(
                            model,
                            base_output,
                            pre_positions,
                            candidate_ids,
                        )
                    else:
                        full_output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        logits = candidate_logits_from_full_output(
                            full_output,
                            pre_positions,
                            candidate_ids,
                        )
                values, readout = capture.stacked()
                if batch_number == 1 and args.model == "qwen3":
                    with torch.inference_mode():
                        full_output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                    full_logits = full_output.logits
                    batch_axis = torch.arange(
                        full_logits.shape[0],
                        device=full_logits.device,
                    )
                    candidate_axis = torch.tensor(
                        candidate_ids,
                        dtype=torch.long,
                        device=full_logits.device,
                    )
                    reference = full_logits[
                        batch_axis,
                        pre_positions.to(full_logits.device),
                        :,
                    ].index_select(-1, candidate_axis)
                    manual_full_logit_max_abs = float(
                        torch.max(
                            torch.abs(
                                reference.float().to("cpu") - logits
                            )
                        ).item()
                    )
                    del full_output, full_logits, reference
                end = offset + len(row_batch)
                for role in CAPTURE_ROLES:
                    clean_states[role][offset:end] = (
                        values[role].numpy().astype(
                            np.float16, copy=False
                        )
                    )
                clean_readout[offset:end] = readout.numpy().astype(
                    np.float16, copy=False
                )
                clean_logits[offset:end] = logits.numpy().astype(
                    np.float32, copy=False
                )
                offset = end
                if batch_number % 32 == 0:
                    print(
                        f"[phase{protocol.PHASE}] {args.model} clean "
                        f"cases={offset}/{len(cases)}",
                        flush=True,
                    )
        finally:
            capture.close()
        for values in clean_states.values():
            values.flush()
        clean_readout.flush()
        clean_logits.flush()

        prototypes_by_scheme = {
            scheme: prototype_map(
                clean_readout, cases, scheme
            )
            for scheme in (
                "within_template",
                "pooled_template",
                "cross_template",
            )
        }
        clean_summary = clean_metrics(
            clean_readout,
            clean_logits,
            cases,
            prototypes_by_scheme,
        )

        condition_readouts = np.lib.format.open_memmap(
            atlas_dir / "condition_readouts.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(protocol.CONDITIONS),
                len(units),
                info.d_model,
            ),
        )
        condition_logits = np.lib.format.open_memmap(
            atlas_dir / "condition_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(protocol.CONDITIONS),
                len(units),
                len(protocol.CATEGORY_LABELS),
            ),
        )
        condition_readouts[:] = 0
        condition_logits[:] = 0
        condition_rows = []

        for condition_index, condition in enumerate(protocol.CONDITIONS):
            valid = run_condition(
                model=model,
                base_model=base_model,
                layers=layers,
                condition=condition,
                units=units,
                cases=cases,
                clean_states=clean_states,
                source_depth=source_depth,
                query_depth=query_depth,
                readout_depth=readout_depth,
                tokenizer=tokenizer,
                device=device,
                model_name=args.model,
                candidate_ids=candidate_ids,
                output_readout=condition_readouts[condition_index],
                output_logits=condition_logits[condition_index],
            )
            valid_units = [units[index] for index in valid]
            row = {
                "condition": condition,
                "valid_unit_count": len(valid),
                "metrics": condition_metrics(
                    np.asarray(
                        condition_readouts[condition_index, valid]
                    ),
                    np.asarray(
                        condition_logits[condition_index, valid]
                    ),
                    valid_units,
                    prototypes_by_scheme,
                ),
            }
            condition_rows.append(row)
            print(
                json.dumps({
                    "model": args.model,
                    "condition": condition,
                    "valid_unit_count": len(valid),
                    "within_template": row["metrics"][
                        "prototype_readout"
                    ]["within_template"]["all"],
                    "next_token": row["metrics"][
                        "next_token_candidate_logits"
                    ]["all"],
                }, ensure_ascii=False),
                flush=True,
            )
        condition_readouts.flush()
        condition_logits.flush()

        comparisons = comparison_metrics(
            units=units,
            cases=cases,
            clean_readout=clean_readout,
            clean_logits=clean_logits,
            condition_readouts=condition_readouts,
            condition_logits=condition_logits,
            prototypes=prototypes_by_scheme["within_template"],
        )
        arrays = {
            **{
                f"clean_{role}": np.asarray(values)
                for role, values in clean_states.items()
            },
            "clean_readout": np.asarray(clean_readout),
            "clean_logits": np.asarray(clean_logits),
            "condition_readouts": np.asarray(condition_readouts),
            "condition_logits": np.asarray(condition_logits),
        }
        finiteness = finite_audit(arrays)
        metrics = {
            "schema_version": "phase1032_metrics.v1",
            "phase": protocol.PHASE,
            "model": args.model,
            "selected_depths": selected,
            "clean": clean_summary,
            "conditions": condition_rows,
            "paired_comparisons": comparisons,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        summary = {
            "schema_version": "phase1032_model_summary.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "model": args.model,
            "precision": "fp16",
            "quantization": "none",
            "placement": placement,
            "runtime_precision_audit": precision_audit,
            "selected_depths": selected,
            "selection_source": "phase1029_frozen_before_phase1032",
            "candidate_logit_source": (
                "base_model_final_state_plus_output_head"
                if args.model == "qwen3"
                else "full_causal_lm_forward"
            ),
            "manual_vs_full_model_candidate_logit_max_abs": (
                manual_full_logit_max_abs
            ),
            "finiteness": finiteness,
            "elapsed_seconds": time.time() - started,
            "claim_limit": prereg["claim_limit"],
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps({
            "model": args.model,
            "clean": clean_summary,
            "paired_comparisons": comparisons,
            "finiteness": finiteness,
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        model = tokenizer = None


if __name__ == "__main__":
    main()
