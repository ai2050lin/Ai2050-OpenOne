#!/usr/bin/env python3
"""Phase1253: planted Transformer-edge mechanism library and cut recovery.

This is a mechanism-truth calibration. Every system uses the same universal
Transformer-shaped QK/OV/MLP supernetwork. Runtime truth selects a planted
mechanism, while the recovery procedure sees only opaque edge identifiers and
intervention responses.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PHASE = 1253
CONTRACT_ID = "EXP-C006-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1253_c006_planted_mechanism_cut_library_audit.py"
OUT = ROOT / "tests/glm5/result/phase1253_c006_planted_mechanism_cut_library"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PUBLIC_SYSTEMS = OUT / "material/public_systems.jsonl"
SEALED_TRUTH = OUT / "material/sealed_mechanism_truth.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
DETAILS = OUT / "raw/system_results.jsonl"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/mechanism_cut_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

K = 8
RECORDS = 4
WORLD_COUNT = 64
TASKS_PER_SPLIT = 4
GAUGES_PER_TASK = 4
SPLITS = ("discovery", "confirmation")
MECHANISMS = (
    "serial_lookup",
    "serial_mlp",
    "redundant_dual",
    "conditional_gate",
    "pair_composition",
    "rotated_lookup",
)

EDGE_ROLES = {
    "a.record.qk": "qk",
    "a.record.ov": "ov",
    "a.map.qk": "qk",
    "a.map.ov": "ov",
    "b.record.qk": "qk",
    "b.record.ov": "ov",
    "b.map.mlp": "mlp",
    "gate.qk": "qk",
    "gate.ov": "ov",
    "p.first.qk": "qk",
    "p.first.ov": "ov",
    "p.second.qk": "qk",
    "p.second.ov": "ov",
    "p.compose.mlp": "mlp",
    "p.map.mlp": "mlp",
}
EDGES = tuple(EDGE_ROLES)
A_PATH = ("a.record.qk", "a.record.ov", "a.map.qk", "a.map.ov")
B_PATH = ("b.record.qk", "b.record.ov", "b.map.mlp")
GATE_PATH = ("gate.qk", "gate.ov")
PAIR_PATH = (
    "p.first.qk", "p.first.ov", "p.second.qk", "p.second.ov",
    "p.compose.mlp", "p.map.mlp",
)
PROBES = ("address_roll", "payload_roll", "rule_roll")
PROBE_FOR_ROLE = {"qk": "address_roll", "ov": "payload_roll", "mlp": "rule_roll"}

TASK_SEEDS = {
    "discovery": [1_253_110 + i for i in range(TASKS_PER_SPLIT)],
    "confirmation": [1_253_910 + i for i in range(TASKS_PER_SPLIT)],
}
GAUGE_SEEDS = {
    "discovery": [1_253_210 + i for i in range(GAUGES_PER_TASK)],
    "confirmation": [1_253_810 + i for i in range(GAUGES_PER_TASK)],
}

THRESHOLDS = {
    "behavior_accuracy_min": 0.999,
    "cut_destroy_accuracy_max": 0.25,
    "strict_subset_survival_min": 0.75,
    "role_probe_damage_min": 0.50,
    "role_accuracy_min": 0.99,
    "identity_correct_recovery_min": 0.95,
    "identity_wrong_recovery_max": 0.35,
    "identity_margin_min": 0.60,
    "identity_top1_min": 0.99,
    "inactive_edge_damage_max": 1.0e-6,
    "twin_output_gap_max": 1.0e-7,
    "cut_f1_min": 0.999,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def opaque_id(seed: int, internal: str) -> str:
    return "edge_" + hashlib.sha256(f"{seed}:{internal}".encode("ascii")).hexdigest()[:12]


def one_hot(values: torch.Tensor, classes: int = K) -> torch.Tensor:
    return F.one_hot(values.long(), num_classes=classes).float()


def cyclic_derangement(permutation: np.ndarray, shift: int) -> np.ndarray:
    return ((permutation + shift) % K).astype(np.int64)


def make_task(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mapping_a = rng.permutation(K).astype(np.int64)
    mapping_b = cyclic_derangement(mapping_a, 1 + seed % (K - 1))
    pair_table = np.asarray([(left + right) % K for left in range(K) for right in range(K)], dtype=np.int64)
    return {
        "seed": seed,
        "mapping_a": mapping_a.tolist(),
        "mapping_b": mapping_b.tolist(),
        "pair_table": pair_table.tolist(),
        "task_digest": digest({"mapping_a": mapping_a.tolist(), "mapping_b": mapping_b.tolist()}),
    }


def make_worlds(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    entities = np.empty((WORLD_COUNT, RECORDS), dtype=np.int64)
    values = np.empty((WORLD_COUNT, RECORDS), dtype=np.int64)
    query_first = np.empty(WORLD_COUNT, dtype=np.int64)
    query_second = np.empty(WORLD_COUNT, dtype=np.int64)
    gate = np.empty(WORLD_COUNT, dtype=np.int64)
    for index in range(WORLD_COUNT):
        entities[index] = rng.choice(K, RECORDS, replace=False)
        values[index] = rng.choice(K, RECORDS, replace=False)
        first_index = index % RECORDS
        second_index = (first_index + 1 + (index // RECORDS) % (RECORDS - 1)) % RECORDS
        query_first[index] = entities[index, first_index]
        query_second[index] = entities[index, second_index]
        gate[index] = index % 2
    return {
        "entities": entities,
        "values": values,
        "query_first": query_first,
        "query_second": query_second,
        "gate": gate,
    }


def signed_permutation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(K)
    signs = rng.choice(np.asarray([-1.0, 1.0]), K)
    matrix = np.zeros((K, K), dtype=np.float32)
    matrix[np.arange(K), permutation] = signs
    return matrix


def active_edges(mechanism: str) -> tuple[str, ...]:
    if mechanism in {"serial_lookup", "rotated_lookup"}:
        return A_PATH
    if mechanism == "serial_mlp":
        return B_PATH
    if mechanism == "redundant_dual":
        return A_PATH + B_PATH
    if mechanism == "conditional_gate":
        return A_PATH + B_PATH + GATE_PATH
    if mechanism == "pair_composition":
        return PAIR_PATH
    raise ValueError(mechanism)


def planted_cuts(mechanism: str) -> tuple[tuple[str, ...], ...]:
    if mechanism in {"serial_lookup", "rotated_lookup"}:
        return tuple((edge,) for edge in A_PATH)
    if mechanism == "serial_mlp":
        return tuple((edge,) for edge in B_PATH)
    if mechanism == "pair_composition":
        return tuple((edge,) for edge in PAIR_PATH)
    cross = tuple(tuple(sorted((left, right))) for left in A_PATH for right in B_PATH)
    if mechanism == "redundant_dual":
        return cross
    if mechanism == "conditional_gate":
        # The gate edges dominate every branch-specific cut in the global graph.
        # Branch roles and identities are tested separately under gate-conditioned masks.
        return tuple((edge,) for edge in GATE_PATH)
    raise ValueError(mechanism)


def make_system_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for split in SPLITS:
        for task_index, task_seed in enumerate(TASK_SEEDS[split]):
            task = make_task(task_seed)
            for gauge_index, gauge_seed in enumerate(GAUGE_SEEDS[split]):
                for mechanism_index, mechanism in enumerate(MECHANISMS):
                    permutation_seed = task_seed * 1000 + gauge_seed * 10 + mechanism_index
                    edge_map = {edge: opaque_id(permutation_seed, edge) for edge in EDGES}
                    system_id = "sys_" + hashlib.sha256(
                        f"{split}:{task_seed}:{gauge_seed}:{mechanism}".encode("ascii")
                    ).hexdigest()[:16]
                    public_row = {
                        "system_id": system_id,
                        "split": split,
                        "task_slot": task_index,
                        "gauge_slot": gauge_index,
                        "opaque_edges": sorted(edge_map.values()),
                        "world_seed": task_seed + 50_000 + gauge_index,
                        "architecture": "universal_qk_ov_mlp_supernetwork_v1",
                        "parameter_budget": 4096,
                        "parameter_l2_class": "orthogonal_permutation_constant",
                    }
                    truth_row = {
                        **public_row,
                        "mechanism": mechanism,
                        "task": task,
                        "gauge_seed": gauge_seed,
                        "edge_map": edge_map,
                        "active_edges": list(active_edges(mechanism)),
                        "edge_roles": EDGE_ROLES,
                        "planted_cuts": [list(cut) for cut in planted_cuts(mechanism)],
                    }
                    public_row["row_digest"] = digest(public_row)
                    truth_row["row_digest"] = digest(truth_row)
                    public.append(public_row)
                    truth.append(truth_row)
    return public, truth


@dataclass
class Intervention:
    blocked: frozenset[str] = frozenset()
    probes: tuple[tuple[str, str], ...] = ()
    overrides: dict[str, torch.Tensor] | None = None


class PlantedTransformerSupernetwork:
    """A fixed-shape causal QK/OV/MLP graph with explicit edge hooks."""

    def __init__(self, task: dict[str, Any], gauge_seed: int, mechanism: str, device: torch.device) -> None:
        self.device = device
        self.mechanism = mechanism
        self.mapping_a = torch.tensor(task["mapping_a"], device=device)
        self.mapping_b = torch.tensor(task["mapping_b"], device=device)
        self.pair_table = torch.tensor(task["pair_table"], device=device)
        self.gauge = torch.tensor(signed_permutation(gauge_seed), device=device)
        self.identity = torch.eye(K, device=device)
        self.trace: dict[str, torch.Tensor] = {}

    def apply_edge(self, edge: str, value: torch.Tensor, role: str, intervention: Intervention) -> torch.Tensor:
        self.trace[edge] = value.detach().clone()
        if intervention.overrides and edge in intervention.overrides:
            return intervention.overrides[edge]
        if edge in intervention.blocked:
            return torch.zeros_like(value)
        probe_map = dict(intervention.probes)
        probe = probe_map.get(edge)
        if probe == PROBE_FOR_ROLE[role]:
            return torch.roll(value, shifts=1, dims=-1)
        return value

    def attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        qk_edge: str,
        ov_edge: str,
        intervention: Intervention,
    ) -> torch.Tensor:
        scores = torch.einsum("bd,bnd->bn", query, keys) * 30.0
        weights = torch.softmax(scores, dim=-1)
        weights = self.apply_edge(qk_edge, weights, "qk", intervention)
        payload = torch.einsum("bn,bnd->bd", weights, values)
        return self.apply_edge(ov_edge, payload, "ov", intervention)

    def record_path(
        self,
        entities: torch.Tensor,
        values: torch.Tensor,
        query: torch.Tensor,
        prefix: str,
        intervention: Intervention,
    ) -> torch.Tensor:
        return self.attention(
            one_hot(query), one_hot(entities), one_hot(values),
            f"{prefix}.record.qk", f"{prefix}.record.ov", intervention,
        )

    def map_lookup(self, source: torch.Tensor, intervention: Intervention, rotated: bool) -> torch.Tensor:
        gauge = self.gauge if rotated else self.identity
        query = source @ gauge
        keys = gauge[None, :, :].expand(source.shape[0], -1, -1)
        values = one_hot(self.mapping_a)[None, :, :].expand(source.shape[0], -1, -1)
        return self.attention(query, keys, values, "a.map.qk", "a.map.ov", intervention)

    def mlp_map(self, source: torch.Tensor, mapping: torch.Tensor, edge: str, intervention: Intervention) -> torch.Tensor:
        matrix = one_hot(mapping)
        output = torch.relu(source) @ matrix
        return self.apply_edge(edge, output, "mlp", intervention)

    def forward(self, worlds: dict[str, torch.Tensor], intervention: Intervention | None = None) -> torch.Tensor:
        intervention = intervention or Intervention()
        self.trace = {}
        entities = worlds["entities"]
        values = worlds["values"]
        q1 = worlds["query_first"]
        q2 = worlds["query_second"]

        a_source = self.record_path(entities, values, q1, "a", intervention)
        a_output = self.map_lookup(a_source, intervention, self.mechanism == "rotated_lookup")

        b_source = self.record_path(entities, values, q1, "b", intervention)
        b_output = self.mlp_map(b_source, self.mapping_b if self.mechanism == "conditional_gate" else self.mapping_a, "b.map.mlp", intervention)

        first = self.attention(one_hot(q1), one_hot(entities), one_hot(values), "p.first.qk", "p.first.ov", intervention)
        second = self.attention(one_hot(q2), one_hot(entities), one_hot(values), "p.second.qk", "p.second.ov", intervention)
        pair_features = torch.einsum("bi,bj->bij", first, second).reshape(first.shape[0], K * K)
        pair_value = torch.relu(pair_features) @ one_hot(self.pair_table)
        pair_value = self.apply_edge("p.compose.mlp", pair_value, "mlp", intervention)
        pair_output = self.mlp_map(pair_value, self.mapping_a, "p.map.mlp", intervention)

        gate_weights = one_hot(worlds["gate"], classes=2)
        gate_weights = self.apply_edge("gate.qk", gate_weights, "qk", intervention)
        gate_payload = self.apply_edge("gate.ov", gate_weights, "ov", intervention)

        if self.mechanism in {"serial_lookup", "rotated_lookup"}:
            output = a_output
        elif self.mechanism == "serial_mlp":
            output = b_output
        elif self.mechanism == "redundant_dual":
            output = 0.5 * (a_output + b_output)
        elif self.mechanism == "conditional_gate":
            output = gate_payload[:, :1] * a_output + gate_payload[:, 1:2] * b_output
        elif self.mechanism == "pair_composition":
            output = pair_output
        else:
            raise ValueError(self.mechanism)
        return 12.0 * output


def tensor_worlds(row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    worlds = make_worlds(int(row["world_seed"]))
    return {name: torch.tensor(value, device=device) for name, value in worlds.items()}


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((torch.argmax(logits, dim=-1) == target).float().mean().item())


def task_target(row: dict[str, Any], worlds: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute labels from the task contract, independently of model output."""

    entities = worlds["entities"]
    values = worlds["values"]

    def looked_up(query: torch.Tensor) -> torch.Tensor:
        index = torch.argmax((entities == query[:, None]).to(torch.int64), dim=1)
        return values.gather(1, index[:, None]).squeeze(1)

    first = looked_up(worlds["query_first"])
    mapping_a = torch.tensor(row["task"]["mapping_a"], device=first.device)
    if row["mechanism"] in {"serial_lookup", "serial_mlp", "redundant_dual", "rotated_lookup"}:
        return mapping_a[first]
    if row["mechanism"] == "conditional_gate":
        mapping_b = torch.tensor(row["task"]["mapping_b"], device=first.device)
        return torch.where(worlds["gate"] == 0, mapping_a[first], mapping_b[first])
    if row["mechanism"] == "pair_composition":
        second = looked_up(worlds["query_second"])
        return mapping_a[(first + second) % K]
    raise ValueError(row["mechanism"])


def intervention_accuracy(
    model: PlantedTransformerSupernetwork,
    worlds: dict[str, torch.Tensor],
    target: torch.Tensor,
    intervention: Intervention,
    mask: torch.Tensor | None = None,
) -> float:
    prediction = torch.argmax(model.forward(worlds, intervention), dim=-1)
    if mask is not None:
        prediction = prediction[mask]
        target = target[mask]
    return float((prediction == target).float().mean().item())


def proper_subsets(cut: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    for size in range(len(cut)):
        yield from itertools.combinations(cut, size)


def infer_minimal_cuts(
    model: PlantedTransformerSupernetwork,
    worlds: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> tuple[list[tuple[str, ...]], dict[tuple[str, ...], float]]:
    candidates: dict[tuple[str, ...], float] = {(): accuracy(model.forward(worlds), target)}
    for size in (1, 2):
        for subset in itertools.combinations(EDGES, size):
            candidates[subset] = intervention_accuracy(model, worlds, target, Intervention(blocked=frozenset(subset)))
    cuts: list[tuple[str, ...]] = []
    for subset, score in candidates.items():
        if not subset or score > THRESHOLDS["cut_destroy_accuracy_max"]:
            continue
        if all(candidates[tuple(parent)] >= THRESHOLDS["strict_subset_survival_min"] for parent in proper_subsets(subset)):
            cuts.append(tuple(sorted(subset)))
    return sorted(set(cuts)), candidates


def f1_sets(predicted: set[tuple[str, ...]], expected: set[tuple[str, ...]]) -> float:
    if not predicted and not expected:
        return 1.0
    true_positive = len(predicted & expected)
    precision = true_positive / max(1, len(predicted))
    recall = true_positive / max(1, len(expected))
    return 2.0 * precision * recall / max(1.0e-12, precision + recall)


def best_witness(
    model: PlantedTransformerSupernetwork,
    worlds: dict[str, torch.Tensor],
    target: torch.Tensor,
    edge: str,
    mask: torch.Tensor | None = None,
) -> tuple[str | None, float]:
    choices: list[tuple[str | None, float]] = []
    for context in (None, *[candidate for candidate in EDGES if candidate != edge]):
        blocked = frozenset(() if context is None else (context,))
        baseline = intervention_accuracy(model, worlds, target, Intervention(blocked=blocked), mask)
        if baseline < 0.90:
            continue
        damaged = intervention_accuracy(model, worlds, target, Intervention(blocked=blocked | {edge}), mask)
        choices.append((context, baseline - damaged))
    return max(choices, key=lambda item: item[1]) if choices else (None, 0.0)


def inferred_edge_mask(
    model: PlantedTransformerSupernetwork,
    worlds: dict[str, torch.Tensor],
    edge: str,
    context: str | None,
) -> torch.Tensor:
    """Infer the edge's responsive worlds without consulting planted mechanism truth."""

    context_block = frozenset(() if context is None else (context,))
    baseline = model.forward(worlds, Intervention(blocked=context_block))
    damaged = model.forward(worlds, Intervention(blocked=context_block | {edge}))
    return torch.amax(torch.abs(baseline - damaged), dim=-1) > 1.0e-7


def recover_roles(
    model: PlantedTransformerSupernetwork,
    worlds: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, str | None]]:
    recovered: dict[str, str] = {}
    scores: dict[str, dict[str, float]] = {}
    witnesses: dict[str, str | None] = {}
    for edge in EDGES:
        context, _ = best_witness(model, worlds, target, edge)
        witnesses[edge] = context
        mask = inferred_edge_mask(model, worlds, edge, context)
        if not bool(mask.any().item()):
            recovered[edge] = "inactive"
            scores[edge] = {probe: 0.0 for probe in PROBES}
            continue
        blocked = frozenset(() if context is None else (context,))
        baseline = intervention_accuracy(model, worlds, target, Intervention(blocked=blocked), mask)
        edge_scores: dict[str, float] = {}
        for probe in PROBES:
            probed = intervention_accuracy(
                model, worlds, target,
                Intervention(blocked=blocked, probes=((edge, probe),)), mask,
            )
            edge_scores[probe] = baseline - probed
        scores[edge] = edge_scores
        winner = max(edge_scores, key=edge_scores.get)
        recovered[edge] = (
            {"address_roll": "qk", "payload_roll": "ov", "rule_roll": "mlp"}[winner]
            if edge_scores[winner] >= THRESHOLDS["role_probe_damage_min"] else "inactive"
        )
    return recovered, scores, witnesses


def wrong_worlds(worlds: dict[str, torch.Tensor], delta: int) -> dict[str, torch.Tensor]:
    """Create one global wrong-identity donor panel shared by every edge."""

    result = {name: value.clone() for name, value in worlds.items()}
    for query_name in ("query_first", "query_second"):
        entities = result["entities"]
        current = result[query_name]
        indices = (torch.argmax((entities == current[:, None]).float(), dim=1) + delta) % RECORDS
        result[query_name] = entities.gather(1, indices[:, None]).squeeze(1)
    result["values"] = (result["values"] + delta) % K
    result["gate"] = 1 - result["gate"]
    return result


def identity_recovery(
    model: PlantedTransformerSupernetwork,
    worlds: dict[str, torch.Tensor],
    target: torch.Tensor,
    recovered_roles: dict[str, str],
    witnesses: dict[str, str | None],
) -> dict[str, Any]:
    baseline_logits = model.forward(worlds)
    correct_trace = {name: value.clone() for name, value in model.trace.items()}
    rows: list[dict[str, Any]] = []
    for edge in EDGES:
        if recovered_roles[edge] == "inactive":
            continue
        context = witnesses[edge]
        mask = inferred_edge_mask(model, worlds, edge, context)
        blocked_context = frozenset(() if context is None else (context,)) | {edge}
        correct = intervention_accuracy(
            model, worlds, target,
            Intervention(blocked=blocked_context, overrides={edge: correct_trace[edge]}), mask,
        )
        wrong_scores: list[float] = []
        for delta in (1, 2, 3):
            donor_worlds = wrong_worlds(worlds, delta)
            model.forward(donor_worlds)
            donor = model.trace[edge].clone()
            score = intervention_accuracy(
                model, worlds, target,
                Intervention(blocked=blocked_context, overrides={edge: donor}), mask,
            )
            wrong_scores.append(score)
        wrong_max = max(wrong_scores)
        rows.append({
            "edge": edge,
            "correct_recovery": correct,
            "wrong_recoveries": wrong_scores,
            "wrong_max": wrong_max,
            "margin": correct - wrong_max,
            "top1": correct > wrong_max,
        })
    return {
        "edges": rows,
        "correct_min": min((row["correct_recovery"] for row in rows), default=1.0),
        "wrong_max": max((row["wrong_max"] for row in rows), default=0.0),
        "margin_min": min((row["margin"] for row in rows), default=1.0),
        "top1_fraction": float(np.mean([row["top1"] for row in rows])) if rows else 1.0,
        "baseline_digest": digest(baseline_logits.detach().cpu().tolist()),
    }


def internal_to_opaque(values: Iterable[str], edge_map: dict[str, str]) -> tuple[str, ...]:
    return tuple(sorted(edge_map[value] for value in values))


def run_system(row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    mechanism = row["mechanism"]
    model = PlantedTransformerSupernetwork(row["task"], int(row["gauge_seed"]), mechanism, device)
    worlds = tensor_worlds(row, device)
    logits = model.forward(worlds)
    target = task_target(row, worlds)
    baseline_accuracy = accuracy(logits, target)
    natural_logits = logits.detach().cpu().numpy()

    cuts, cut_scores = infer_minimal_cuts(model, worlds, target)
    expected = {tuple(sorted(cut)) for cut in row["planted_cuts"]}
    predicted = set(cuts)
    cut_f1 = f1_sets(predicted, expected)

    recovered_roles, role_scores, witnesses = recover_roles(model, worlds, target)
    active = set(row["active_edges"])
    role_accuracy = float(np.mean([
        recovered_roles[edge] == (EDGE_ROLES[edge] if edge in active else "inactive") for edge in EDGES
    ]))
    inactive_damage = max(
        (max(role_scores[edge].values()) for edge in EDGES if edge not in active), default=0.0,
    )
    identity = identity_recovery(model, worlds, target, recovered_roles, witnesses)

    opaque_expected = sorted(internal_to_opaque(cut, row["edge_map"]) for cut in expected)
    opaque_predicted = sorted(internal_to_opaque(cut, row["edge_map"]) for cut in predicted)
    abstract_cuts = sorted(tuple(sorted(recovered_roles[edge] for edge in cut)) for cut in predicted)

    generic_spectrum = {
        "natural": natural_logits.tolist(),
        "scale_half": (0.5 * natural_logits).tolist(),
        "common_bias": (natural_logits + 3.0).tolist(),
        "class_roll": np.roll(natural_logits, 1, axis=-1).tolist(),
    }
    return {
        "system_id": row["system_id"],
        "split": row["split"],
        "mechanism": mechanism,
        "task_slot": row["task_slot"],
        "gauge_slot": row["gauge_slot"],
        "behavior_accuracy": baseline_accuracy,
        "cut_f1": cut_f1,
        "cut_exact": predicted == expected,
        "expected_cuts_opaque": opaque_expected,
        "predicted_cuts_opaque": opaque_predicted,
        "abstract_predicted_cuts": abstract_cuts,
        "role_accuracy": role_accuracy,
        "inactive_edge_damage_max": inactive_damage,
        "recovered_roles_opaque": {row["edge_map"][edge]: recovered_roles[edge] for edge in EDGES},
        "role_probe_scores": {row["edge_map"][edge]: role_scores[edge] for edge in EDGES},
        "identity": identity,
        "generic_spectrum": generic_spectrum,
        "single_block_survival_min": min(
            (score for subset, score in cut_scores.items() if len(subset) == 1), default=1.0,
        ),
    }


def protocol_payload(public: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "schema_version": "phase1253.c006.planted_mechanism_cut.protocol.v1",
        "contract_id": CONTRACT_ID,
        "claim_type": "mechanism_truth_component_edge_and_minimal_cut_calibration",
        "question": "Can opaque response interventions recover planted QK, OV and MLP roles, minimal cuts, redundancy and identity paths across gauges and held-out tasks?",
        "system_count": len(public),
        "splits": list(SPLITS),
        "tasks_per_split": TASKS_PER_SPLIT,
        "gauges_per_task": GAUGES_PER_TASK,
        "mechanism_count": len(MECHANISMS),
        "mechanisms_sha256": digest(list(MECHANISMS)),
        "edge_count": len(EDGES),
        "public_digest": digest(public),
        "sealed_truth_digest": digest(truth),
        "thresholds": THRESHOLDS,
        "budgets": {"max_gpu_hours": 0.5, "max_formal_runs": 1, "max_adaptive_rounds": 0},
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "hard_stops": [
            "Task truth, endpoint truth and planted mechanism truth remain separate evidence types.",
            "Opaque edge identifiers, held-out tasks and gauge permutations are frozen before execution.",
            "Cut recovery must match the complete planted minimal-cut family, not one convenient cut.",
            "Correct donor must beat every registered wrong donor; threshold relaxation is forbidden.",
            "Output-equivalent twins must trigger output-only abstention and edge-level separation.",
            "Failure blocks free-network and pretrained-model escalation.",
            "A pass calibrates an instrument; it is not a natural-language mechanism discovery.",
        ],
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "fp32 deterministic planted operations",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    public, truth = make_system_rows()
    write_jsonl(PUBLIC_SYSTEMS, public)
    write_jsonl(SEALED_TRUTH, truth)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(public, truth))
    print(canonical_json({"status": "preregistered", "systems": len(public)}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    public = read_jsonl(PUBLIC_SYSTEMS)
    truth = read_jsonl(SEALED_TRUTH)
    expected = protocol_payload(public, truth)
    if protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("source changed after preregistration")
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["public_digest"] != digest(public) or protocol["sealed_truth_digest"] != digest(truth):
        raise RuntimeError("material digest mismatch")
    return protocol, public, truth


def run(device_name: str) -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal completion marker already exists")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit has not passed")
    protocol, public, truth = verify_protocol()
    if not torch.cuda.is_available() or device_name != "cuda":
        raise RuntimeError("formal run requires CUDA")
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1_253_000_001)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, row in enumerate(truth):
            results.append(run_system(row, device))
            if (index + 1) % 24 == 0:
                print(canonical_json({"completed": index + 1, "total": len(truth)}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    write_jsonl(DETAILS, results)
    raw = {
        "phase": PHASE,
        "schema_version": "phase1253.c006.run.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "system_count": len(results),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "pretrained_model_loaded": False,
        "details_sha256": file_sha256(DETAILS),
        "run_digest": digest(results),
    }
    atomic_json(RAW, raw)
    marker = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": raw["run_digest"],
        "raw_sha256": file_sha256(RAW),
        "details_sha256": file_sha256(DETAILS),
        "status": "formal_run_complete",
    }
    marker["marker_digest"] = digest(marker)
    atomic_json(COMPLETE, marker)
    print(canonical_json({"status": "formal_run_complete", "elapsed_seconds": elapsed}))


def summarize_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split]
    per_mechanism: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        values = [row for row in subset if row["mechanism"] == mechanism]
        per_mechanism[mechanism] = {
            "systems": len(values),
            "behavior_min": min(row["behavior_accuracy"] for row in values),
            "cut_f1_min": min(row["cut_f1"] for row in values),
            "cut_exact_all": all(row["cut_exact"] for row in values),
            "role_accuracy_min": min(row["role_accuracy"] for row in values),
            "identity_correct_min": min(row["identity"]["correct_min"] for row in values),
            "identity_wrong_max": max(row["identity"]["wrong_max"] for row in values),
            "identity_margin_min": min(row["identity"]["margin_min"] for row in values),
            "identity_top1_min": min(row["identity"]["top1_fraction"] for row in values),
            "inactive_damage_max": max(row["inactive_edge_damage_max"] for row in values),
        }

    keyed = {(row["task_slot"], row["gauge_slot"], row["mechanism"]): row for row in subset}
    twin_gaps: list[float] = []
    twin_cut_different: list[bool] = []
    gauge_matches: list[bool] = []
    for task_slot in range(TASKS_PER_SPLIT):
        for gauge_slot in range(GAUGES_PER_TASK):
            lookup = keyed[(task_slot, gauge_slot, "serial_lookup")]
            mlp = keyed[(task_slot, gauge_slot, "serial_mlp")]
            rotated = keyed[(task_slot, gauge_slot, "rotated_lookup")]
            left = np.asarray(lookup["generic_spectrum"]["natural"], dtype=np.float64)
            right = np.asarray(mlp["generic_spectrum"]["natural"], dtype=np.float64)
            twin_gaps.append(float(np.max(np.abs(left - right))))
            twin_cut_different.append(lookup["abstract_predicted_cuts"] != mlp["abstract_predicted_cuts"])
            gauge_matches.append(lookup["abstract_predicted_cuts"] == rotated["abstract_predicted_cuts"])

    checks = {
        "behavior": all(value["behavior_min"] >= THRESHOLDS["behavior_accuracy_min"] for value in per_mechanism.values()),
        "cut_recovery": all(value["cut_f1_min"] >= THRESHOLDS["cut_f1_min"] and value["cut_exact_all"] for value in per_mechanism.values()),
        "role_recovery": all(value["role_accuracy_min"] >= THRESHOLDS["role_accuracy_min"] for value in per_mechanism.values()),
        "identity_recovery": all(
            value["identity_correct_min"] >= THRESHOLDS["identity_correct_recovery_min"]
            and value["identity_wrong_max"] <= THRESHOLDS["identity_wrong_recovery_max"]
            and value["identity_margin_min"] >= THRESHOLDS["identity_margin_min"]
            and value["identity_top1_min"] >= THRESHOLDS["identity_top1_min"]
            for value in per_mechanism.values()
        ),
        "inactive_null": all(value["inactive_damage_max"] <= THRESHOLDS["inactive_edge_damage_max"] for value in per_mechanism.values()),
        "output_twin_abstention": max(twin_gaps) <= THRESHOLDS["twin_output_gap_max"],
        "edge_twin_separation": all(twin_cut_different),
        "gauge_invariance": all(gauge_matches),
    }
    return {
        "split": split,
        "systems": len(subset),
        "per_mechanism": per_mechanism,
        "twin_output_gap_max": max(twin_gaps),
        "twin_edge_separation_fraction": float(np.mean(twin_cut_different)),
        "gauge_invariance_fraction": float(np.mean(gauge_matches)),
        "checks": checks,
        "passed": all(checks.values()),
    }


def analyze() -> None:
    if not COMPLETE.exists():
        raise RuntimeError("formal run is incomplete")
    protocol, _, _ = verify_protocol()
    raw = read_json(RAW)
    marker = read_json(COMPLETE)
    if raw["details_sha256"] != file_sha256(DETAILS) or marker["raw_sha256"] != file_sha256(RAW):
        raise RuntimeError("formal artifact hash mismatch")
    rows = read_jsonl(DETAILS)
    summaries = {split: summarize_split(rows, split) for split in SPLITS}
    gates = {
        "G-MECHANISM-TRUTH-BREADTH": all(summary["passed"] for summary in summaries.values()),
        "G-CUT-RECOVERY": all(summary["checks"]["cut_recovery"] for summary in summaries.values()),
        "G-ROLE-RECOVERY": all(summary["checks"]["role_recovery"] for summary in summaries.values()),
        "G-IDENTITY-RECOVERY": all(summary["checks"]["identity_recovery"] for summary in summaries.values()),
        "G-TWIN-ABSTENTION-SEPARATION": all(
            summary["checks"]["output_twin_abstention"] and summary["checks"]["edge_twin_separation"]
            for summary in summaries.values()
        ),
        "G-GAUGE-INVARIANCE": all(summary["checks"]["gauge_invariance"] for summary in summaries.values()),
    }
    passed = all(gates.values())
    verdict = "planted_component_edge_cut_camera_confirmed" if passed else "planted_component_edge_cut_camera_not_confirmed"
    analysis = {
        "phase": PHASE,
        "schema_version": "phase1253.c006.analysis.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": raw["run_digest"],
        "verdict": verdict,
        "gates": gates,
        "splits": summaries,
        "authorization": {
            "free_network_component_edge_external_validity_contract": passed,
            "pretrained_model_contract": False,
            "natural_language_mechanism_claim": False,
            "new_foundational_mathematics": False,
        },
        "interpretation": {
            "task_truth": True,
            "endpoint_truth": True,
            "planted_mechanism_truth": True,
            "free_network_mechanism_truth": False,
            "scope": "instrument calibration in a deterministic planted Transformer-shaped supernetwork",
        },
    }
    analysis["analysis_digest"] = digest(analysis)
    atomic_json(ANALYSIS, analysis)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "gates": gates,
        "authorization": analysis["authorization"],
        "artifact_hashes": {
            "protocol": file_sha256(PROTOCOL),
            "public_material": file_sha256(PUBLIC_SYSTEMS),
            "sealed_truth": file_sha256(SEALED_TRUTH),
            "environment": file_sha256(ENVIRONMENT),
            "preaudit": file_sha256(PREAUDIT),
            "raw": file_sha256(RAW),
            "details": file_sha256(DETAILS),
            "complete": file_sha256(COMPLETE),
            "analysis": file_sha256(ANALYSIS),
        },
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"verdict": verdict, "gates": gates}))


def probe() -> None:
    _, truth = make_system_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = [row for row in truth if row["split"] == "discovery" and row["task_slot"] == 0 and row["gauge_slot"] == 0]
    output = [run_system(row, device) for row in selected]
    path = ROOT / "tests/glm5_temp/phase1253_planted_mechanism_probe.json"
    atomic_json(path, output)
    print(canonical_json({"probe": str(path), "systems": len(output)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("probe", "preregister", "run", "analyze"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "probe":
        probe()
    elif args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        run(args.device)
    else:
        analyze()


if __name__ == "__main__":
    main()
