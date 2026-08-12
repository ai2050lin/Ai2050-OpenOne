#!/usr/bin/env python3
"""Known-truth calibration of functional targets and typed probe contracts.

Each public cube has 32 sealed completions: 16 functionally distinct active
codes crossed with a functionally null state bit.  Exact targets therefore
have 32 classes while the registered functional quotient has 16.  Frozen
probe contracts test whether functional and exact targets require different
observations without treating a hidden vector as the semantic target.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
PHASE = 1212
OUT_ROOT = TEST_ROOT / "result/phase1212_functional_target_quotient_minimal_observation"
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1212_functional_target_quotient_minimal_observation_audit.py")
SOURCE1211 = TEST_ROOT / "result/phase1211_semantic_intervention_compiler_identifiability"

PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
SELECTION_PATH = OUT_ROOT / "analysis/discovery_probe_selection.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

EXPECTED_1211_FINAL = "205d3629e6a118d491b6c6659cbe6a5f2d65adb0806dbb5b324c552c7cdd1361"
EXPECTED_1211_AUDIT = "db8ddb48960198106991171d23b8796001e98bed8ef7bb470615c437981cad09"

DIMENSION = 16
FUNCTION_DIMENSION = 4
REPLICATES = 16
GAUGES = ("signed_permutation", "orthogonal_dense")
FAMILIES_BY_SPLIT = {
    "discovery": ("additive", "semantic_pair"),
    "confirmation": ("carrier_pair", "full_pairwise"),
}
VERTICES = tuple((a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))
ACTIVE_CODES = tuple(tuple((mask >> index) & 1 for index in range(FUNCTION_DIMENSION)) for mask in range(1 << FUNCTION_DIMENSION))
NULL_CODES = (0, 1)
EPSILON = 1.0e-12
TOLERANCE = 1.0e-9

PROBES = (
    {"name": "baseline_none", "cost": 0, "priority": 0, "kind": "control"},
    {"name": "functional_edge_ch0", "cost": 1, "priority": 10, "kind": "partial_functional"},
    {"name": "functional_edge_ch01", "cost": 2, "priority": 20, "kind": "partial_functional"},
    {"name": "functional_edge_ch012", "cost": 3, "priority": 30, "kind": "partial_functional"},
    {"name": "visible_functional_value_control", "cost": 4, "priority": 40, "kind": "control"},
    {"name": "wrong_axis_functional_jacobian_control", "cost": 4, "priority": 50, "kind": "control"},
    {"name": "functional_edge_all", "cost": 4, "priority": 60, "kind": "functional"},
    {"name": "functional_midpoint_all", "cost": 4, "priority": 70, "kind": "functional"},
    {"name": "full_edge_jacobian", "cost": 16, "priority": 80, "kind": "exact"},
    {"name": "full_midpoint", "cost": 16, "priority": 90, "kind": "exact"},
)
PROBE_NAMES = tuple(row["name"] for row in PROBES)
PROBE_BY_NAME = {row["name"]: row for row in PROBES}

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "visible_world_identity_max": 0.0,
    "functional_class_count": 16,
    "exact_class_count": 32,
    "identifiable_fraction_min": 1.0,
    "nonidentifiable_fraction_max": 0.0,
    "decoder_max_error_max": 1.0e-10,
    "gauge_metric_gap_max": 1.0e-10,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    os.replace(temporary, path)


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_digest(value: dict[str, Any], field: str) -> None:
    clean = dict(value)
    stored = clean.pop(field)
    if digest(clean) != stored:
        raise RuntimeError(f"digest mismatch: {field}")


def split_root(split: str) -> Path:
    return OUT_ROOT / "runs" / split


def latent_seed(split: str, family: str, target_index: int, replicate: int) -> int:
    key = f"phase1212|latent|{split}|{family}|{target_index}|{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def gauge_seed(split: str, family: str, gauge: str, target_index: int, replicate: int) -> int:
    key = f"phase1212|gauge|{split}|{family}|{gauge}|{target_index}|{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def design(vertex: tuple[float, float, float]) -> np.ndarray:
    a, b, c = vertex
    return np.asarray((1.0, a, b, c, a * b, a * c, b * c), dtype=np.float64)


def derivative_design(vertex: tuple[float, float, float], axis: int) -> np.ndarray:
    a, b, c = vertex
    rows = (
        (0.0, 1.0, 0.0, 0.0, b, c, 0.0),
        (0.0, 0.0, 1.0, 0.0, a, 0.0, c),
        (0.0, 0.0, 0.0, 1.0, 0.0, a, b),
    )
    return np.asarray(rows[axis], dtype=np.float64)


def indicator(vertex: tuple[float, float, float], target: tuple[int, int, int]) -> float:
    return float(np.prod([(1.0 + value * target_value) / 2.0 for value, target_value in zip(vertex, target)]))


def indicator_derivative(vertex: tuple[float, float, float], target: tuple[int, int, int], axis: int) -> float:
    value = target[axis] / 2.0
    for index in range(3):
        if index != axis:
            value *= (1.0 + vertex[index] * target[index]) / 2.0
    return float(value)


def gauge_matrix(rng: np.random.Generator, gauge: str) -> np.ndarray:
    if gauge == "signed_permutation":
        permutation = rng.permutation(DIMENSION)
        signs = rng.choice((-1.0, 1.0), size=DIMENSION)
        matrix = np.zeros((DIMENSION, DIMENSION), dtype=np.float64)
        matrix[np.arange(DIMENSION), permutation] = signs
        return matrix
    raw = rng.normal(size=(DIMENSION, DIMENSION))
    q, r = np.linalg.qr(raw)
    return (q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))).astype(np.float64)


def active_basis(family: str) -> set[int]:
    return {
        "additive": {0, 1, 2, 3},
        "semantic_pair": {0, 1, 2, 3, 4},
        "carrier_pair": {0, 1, 2, 3, 5, 6},
        "full_pairwise": set(range(7)),
    }[family]


def system_spec(split: str, family: str, gauge: str, target_index: int, replicate: int) -> dict[str, Any]:
    latent_rng = np.random.default_rng(latent_seed(split, family, target_index, replicate))
    gauge_rng = np.random.default_rng(gauge_seed(split, family, gauge, target_index, replicate))
    target = VERTICES[target_index]
    ta, tb, tc = target
    recipient = (-ta, tb, tc)
    coefficients = np.zeros((7, DIMENSION), dtype=np.float64)
    amplitudes = latent_rng.uniform(0.75, 1.75, size=7) * latent_rng.choice((-1.0, 1.0), size=7)
    for index in active_basis(family):
        coefficients[index, index] = amplitudes[index]
    gauge_value = gauge_matrix(gauge_rng, gauge)
    base_latent = np.stack([design(vertex) @ coefficients for vertex in VERTICES])
    base_hidden = base_latent @ gauge_value.T
    active_scales = latent_rng.uniform(0.75, 1.75, size=FUNCTION_DIMENSION) * latent_rng.choice((-1.0, 1.0), size=FUNCTION_DIMENSION)
    active_latent = np.zeros((FUNCTION_DIMENSION, DIMENSION), dtype=np.float64)
    for index in range(FUNCTION_DIMENSION):
        active_latent[index, index] = active_scales[index]
    null_latent = np.zeros(DIMENSION, dtype=np.float64)
    null_latent[7] = float(latent_rng.uniform(0.75, 1.75) * latent_rng.choice((-1.0, 1.0)))
    active_hidden = active_latent @ gauge_value.T
    null_hidden = null_latent @ gauge_value.T
    readout_latent = np.zeros((DIMENSION, FUNCTION_DIMENSION), dtype=np.float64)
    readout_latent[:FUNCTION_DIMENSION, :] = np.eye(FUNCTION_DIMENSION)
    readout_hidden = gauge_value @ readout_latent
    return {
        "pair_id": f"p1212:{split}:{family}:{gauge}:t{target_index}:r{replicate:03d}",
        "split": split,
        "family": family,
        "gauge": gauge,
        "target_index": target_index,
        "target": target,
        "recipient": recipient,
        "coefficients": coefficients,
        "gauge_matrix": gauge_value,
        "base_hidden": base_hidden,
        "active_hidden": active_hidden,
        "null_hidden": null_hidden,
        "readout_hidden": readout_hidden,
    }


def specs_for(split: str) -> Iterable[dict[str, Any]]:
    for family in FAMILIES_BY_SPLIT[split]:
        for gauge in GAUGES:
            for target_index in range(len(VERTICES)):
                for replicate in range(REPLICATES):
                    yield system_spec(split, family, gauge, target_index, replicate)


def world_id(pair_id: str, active_code: tuple[int, ...], null_code: int) -> str:
    key = f"{pair_id}|{''.join(map(str, active_code))}|{null_code}"
    return "w" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def worlds_for(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for active_code in ACTIVE_CODES:
        for null_code in NULL_CODES:
            delta = np.sum(np.asarray(active_code, dtype=np.float64)[:, None] * spec["active_hidden"], axis=0)
            delta = delta + null_code * spec["null_hidden"]
            yield {
                "world_id": world_id(spec["pair_id"], active_code, null_code),
                "active_code": active_code,
                "null_code": null_code,
                "delta_hidden": delta,
            }


def base_state(spec: dict[str, Any], vertex: tuple[float, float, float]) -> np.ndarray:
    latent = design(vertex) @ spec["coefficients"]
    return latent @ spec["gauge_matrix"].T


def completion_state(spec: dict[str, Any], world: dict[str, Any], vertex: tuple[float, float, float]) -> np.ndarray:
    return base_state(spec, vertex) + indicator(vertex, spec["target"]) * world["delta_hidden"]


def completion_derivative(spec: dict[str, Any], world: dict[str, Any], vertex: tuple[float, float, float], axis: int) -> np.ndarray:
    latent = derivative_design(vertex, axis) @ spec["coefficients"]
    base = latent @ spec["gauge_matrix"].T
    return base + indicator_derivative(vertex, spec["target"], axis) * world["delta_hidden"]


def public_row(spec: dict[str, Any]) -> dict[str, Any]:
    observed = [index for index in range(len(VERTICES)) if index != spec["target_index"]]
    return {
        "pair_id": spec["pair_id"],
        "split": spec["split"],
        "target": list(spec["target"]),
        "recipient": list(spec["recipient"]),
        "observed_vertices": [list(VERTICES[index]) for index in observed],
        "observed_states": [spec["base_hidden"][index].tolist() for index in observed],
        "functional_readout": spec["readout_hidden"].tolist(),
        "probe_contract": [dict(row) for row in PROBES],
    }


def probe_response(spec: dict[str, Any], world: dict[str, Any], probe: str) -> np.ndarray:
    recipient = spec["recipient"]
    target = spec["target"]
    midpoint = (0.0, float(target[1]), float(target[2]))
    readout = spec["readout_hidden"]
    if probe == "baseline_none":
        return np.zeros(1, dtype=np.float64)
    if probe == "visible_functional_value_control":
        return completion_state(spec, world, recipient) @ readout
    if probe == "wrong_axis_functional_jacobian_control":
        return completion_derivative(spec, world, recipient, 1) @ readout
    edge_functional = completion_derivative(spec, world, recipient, 0) @ readout
    if probe == "functional_edge_ch0":
        return edge_functional[:1]
    if probe == "functional_edge_ch01":
        return edge_functional[:2]
    if probe == "functional_edge_ch012":
        return edge_functional[:3]
    if probe == "functional_edge_all":
        return edge_functional
    if probe == "functional_midpoint_all":
        return completion_state(spec, world, midpoint) @ readout
    if probe == "full_edge_jacobian":
        return completion_derivative(spec, world, recipient, 0)
    if probe == "full_midpoint":
        return completion_state(spec, world, midpoint)
    raise KeyError(probe)


def probe_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for world in worlds_for(spec):
        for probe in PROBE_NAMES:
            rows.append({
                "pair_id": spec["pair_id"],
                "split": spec["split"],
                "world_id": world["world_id"],
                "probe": probe,
                "response": probe_response(spec, world, probe).tolist(),
            })
    return rows


def truth_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    public_digest = digest(public_row(spec)["observed_states"])
    for world in worlds_for(spec):
        exact = completion_state(spec, world, spec["target"])
        functional = exact @ spec["readout_hidden"]
        rows.append({
            "pair_id": spec["pair_id"],
            "split": spec["split"],
            "world_id": world["world_id"],
            "family": spec["family"],
            "gauge": spec["gauge"],
            "active_code": list(world["active_code"]),
            "null_code": world["null_code"],
            "exact_target": exact.tolist(),
            "functional_target": functional.tolist(),
            "visible_state_digest": public_digest,
        })
    return rows


def observed_lookup(public: dict[str, Any]) -> dict[tuple[int, int, int], np.ndarray]:
    return {
        tuple(int(value) for value in vertex): np.asarray(state, dtype=np.float64)
        for vertex, state in zip(public["observed_vertices"], public["observed_states"])
    }


def decode_probe(public: dict[str, Any], probe: str, response: np.ndarray) -> dict[str, np.ndarray | None]:
    lookup = observed_lookup(public)
    recipient = lookup[tuple(public["recipient"])]
    readout = np.asarray(public["functional_readout"], dtype=np.float64)
    target_a = int(public["target"][0])
    recipient_functional = recipient @ readout
    exact = None
    functional = None
    if probe == "functional_edge_all":
        functional = recipient_functional + 2.0 * target_a * response
    elif probe == "functional_midpoint_all":
        functional = 2.0 * response - recipient_functional
    elif probe == "full_edge_jacobian":
        exact = recipient + 2.0 * target_a * response
        functional = exact @ readout
    elif probe == "full_midpoint":
        exact = 2.0 * response - recipient
        functional = exact @ readout
    return {"exact": exact, "functional": functional}


def same(left: np.ndarray, right: np.ndarray) -> bool:
    return float(np.linalg.norm(left - right)) <= TOLERANCE


def class_count(values: list[np.ndarray]) -> int:
    representatives: list[np.ndarray] = []
    for value in values:
        if not any(same(value, representative) for representative in representatives):
            representatives.append(value)
    return len(representatives)


def identifiable(signatures: list[np.ndarray], targets: list[np.ndarray]) -> bool:
    for left in range(len(signatures)):
        for right in range(left + 1, len(signatures)):
            if same(signatures[left], signatures[right]) and not same(targets[left], targets[right]):
                return False
    return True


def aggregate_score(public_rows: list[dict[str, Any]], probes: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    public_by_pair = {row["pair_id"]: row for row in public_rows}
    truth_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    probe_by_key: dict[tuple[str, str], dict[str, np.ndarray]] = defaultdict(dict)
    for row in truth:
        truth_by_pair[row["pair_id"]].append(row)
    for row in probes:
        probe_by_key[(row["pair_id"], row["probe"])][row["world_id"]] = np.asarray(row["response"], dtype=np.float64)

    pair_metrics: list[dict[str, Any]] = []
    for pair_id, worlds in truth_by_pair.items():
        worlds = sorted(worlds, key=lambda row: row["world_id"])
        exact_targets = [np.asarray(row["exact_target"], dtype=np.float64) for row in worlds]
        functional_targets = [np.asarray(row["functional_target"], dtype=np.float64) for row in worlds]
        for probe in PROBE_NAMES:
            signatures = [probe_by_key[(pair_id, probe)][row["world_id"]] for row in worlds]
            exact_errors: list[float] = []
            functional_errors: list[float] = []
            for row, signature, exact_target, functional_target in zip(worlds, signatures, exact_targets, functional_targets):
                decoded = decode_probe(public_by_pair[pair_id], probe, signature)
                if decoded["exact"] is not None:
                    exact_errors.append(float(np.linalg.norm(decoded["exact"] - exact_target)))
                if decoded["functional"] is not None:
                    functional_errors.append(float(np.linalg.norm(decoded["functional"] - functional_target)))
            pair_metrics.append({
                "pair_id": pair_id,
                "family": worlds[0]["family"],
                "gauge": worlds[0]["gauge"],
                "probe": probe,
                "functional_identifiable": identifiable(signatures, functional_targets),
                "exact_identifiable": identifiable(signatures, exact_targets),
                "signature_class_count": class_count(signatures),
                "functional_class_count": class_count(functional_targets),
                "exact_class_count": class_count(exact_targets),
                "functional_decoder_max_error": None if not functional_errors else max(functional_errors),
                "exact_decoder_max_error": None if not exact_errors else max(exact_errors),
            })

    probe_summary: dict[str, Any] = {}
    for probe in PROBE_NAMES:
        members = [row for row in pair_metrics if row["probe"] == probe]
        functional_errors = [row["functional_decoder_max_error"] for row in members if row["functional_decoder_max_error"] is not None]
        exact_errors = [row["exact_decoder_max_error"] for row in members if row["exact_decoder_max_error"] is not None]
        probe_summary[probe] = {
            "cost": PROBE_BY_NAME[probe]["cost"],
            "priority": PROBE_BY_NAME[probe]["priority"],
            "kind": PROBE_BY_NAME[probe]["kind"],
            "functional_identifiable_fraction": float(np.mean([row["functional_identifiable"] for row in members])),
            "exact_identifiable_fraction": float(np.mean([row["exact_identifiable"] for row in members])),
            "mean_signature_class_count": float(np.mean([row["signature_class_count"] for row in members])),
            "functional_decoder_max_error": None if not functional_errors else float(max(functional_errors)),
            "exact_decoder_max_error": None if not exact_errors else float(max(exact_errors)),
        }

    functional_candidates = [
        name for name, row in probe_summary.items()
        if row["functional_identifiable_fraction"] >= THRESHOLDS["identifiable_fraction_min"]
        and row["functional_decoder_max_error"] is not None
        and row["functional_decoder_max_error"] <= THRESHOLDS["decoder_max_error_max"]
    ]
    exact_candidates = [
        name for name, row in probe_summary.items()
        if row["exact_identifiable_fraction"] >= THRESHOLDS["identifiable_fraction_min"]
        and row["exact_decoder_max_error"] is not None
        and row["exact_decoder_max_error"] <= THRESHOLDS["decoder_max_error_max"]
    ]
    key = lambda name: (PROBE_BY_NAME[name]["cost"], PROBE_BY_NAME[name]["priority"])
    selected_functional = min(functional_candidates, key=key) if functional_candidates else None
    selected_exact = min(exact_candidates, key=key) if exact_candidates else None

    gauge_summaries: dict[str, dict[str, dict[str, float]]] = {}
    for gauge in GAUGES:
        gauge_summaries[gauge] = {}
        for probe in PROBE_NAMES:
            members = [row for row in pair_metrics if row["gauge"] == gauge and row["probe"] == probe]
            gauge_summaries[gauge][probe] = {
                "functional_identifiable_fraction": float(np.mean([row["functional_identifiable"] for row in members])),
                "exact_identifiable_fraction": float(np.mean([row["exact_identifiable"] for row in members])),
                "mean_signature_class_count": float(np.mean([row["signature_class_count"] for row in members])),
            }
    gauge_gap = max(
        abs(gauge_summaries[GAUGES[0]][probe][metric] - gauge_summaries[GAUGES[1]][probe][metric])
        for probe in PROBE_NAMES
        for metric in gauge_summaries[GAUGES[0]][probe]
    )

    functional_class_counts = [row["functional_class_count"] for row in pair_metrics if row["probe"] == "baseline_none"]
    exact_class_counts = [row["exact_class_count"] for row in pair_metrics if row["probe"] == "baseline_none"]
    visible_digests: dict[str, set[str]] = defaultdict(set)
    for row in truth:
        visible_digests[row["pair_id"]].add(row["visible_state_digest"])
    finite_values = [value for row in probes for value in row["response"]]
    finite_values += [value for row in truth for key in ("exact_target", "functional_target") for value in row[key]]
    metrics = {
        "pair_count": len(public_rows),
        "system_count": len(truth),
        "probe_row_count": len(probes),
        "finite_fraction": float(np.mean([math.isfinite(value) for value in finite_values])),
        "visible_world_identity_max": 0.0 if all(len(values) == 1 for values in visible_digests.values()) else float("inf"),
        "functional_class_count_min": int(min(functional_class_counts)),
        "functional_class_count_max": int(max(functional_class_counts)),
        "exact_class_count_min": int(min(exact_class_counts)),
        "exact_class_count_max": int(max(exact_class_counts)),
        "selected_functional_probe": selected_functional,
        "selected_exact_probe": selected_exact,
        "selected_functional_cost": None if selected_functional is None else PROBE_BY_NAME[selected_functional]["cost"],
        "selected_exact_cost": None if selected_exact is None else PROBE_BY_NAME[selected_exact]["cost"],
        "gauge_metric_gap": float(gauge_gap),
    }
    sf = probe_summary.get(selected_functional, {})
    se = probe_summary.get(selected_exact, {})
    lower_functional = [row["functional_identifiable_fraction"] for row in probe_summary.values() if selected_functional is not None and row["cost"] < PROBE_BY_NAME[selected_functional]["cost"]]
    lower_exact = [row["exact_identifiable_fraction"] for row in probe_summary.values() if selected_exact is not None and row["cost"] < PROBE_BY_NAME[selected_exact]["cost"]]
    checks = {
        "finite": metrics["finite_fraction"] >= THRESHOLDS["finite_fraction_min"],
        "visible_worlds": metrics["visible_world_identity_max"] <= THRESHOLDS["visible_world_identity_max"],
        "target_class_counts": metrics["functional_class_count_min"] == metrics["functional_class_count_max"] == THRESHOLDS["functional_class_count"] and metrics["exact_class_count_min"] == metrics["exact_class_count_max"] == THRESHOLDS["exact_class_count"],
        "baseline_nonidentifiable": probe_summary["baseline_none"]["functional_identifiable_fraction"] <= THRESHOLDS["nonidentifiable_fraction_max"] and probe_summary["baseline_none"]["exact_identifiable_fraction"] <= THRESHOLDS["nonidentifiable_fraction_max"],
        "partial_functional_insufficient": all(probe_summary[name]["functional_identifiable_fraction"] <= THRESHOLDS["nonidentifiable_fraction_max"] for name in ("functional_edge_ch0", "functional_edge_ch01", "functional_edge_ch012")),
        "matched_controls_insufficient": all(probe_summary[name]["functional_identifiable_fraction"] <= THRESHOLDS["nonidentifiable_fraction_max"] and probe_summary[name]["exact_identifiable_fraction"] <= THRESHOLDS["nonidentifiable_fraction_max"] for name in ("visible_functional_value_control", "wrong_axis_functional_jacobian_control")),
        "functional_probe_selected": selected_functional == "functional_edge_all" and sf.get("functional_identifiable_fraction") == 1.0 and sf.get("functional_decoder_max_error", float("inf")) <= THRESHOLDS["decoder_max_error_max"],
        "functional_probe_not_exact": sf.get("exact_identifiable_fraction") == 0.0,
        "exact_probe_selected": selected_exact == "full_edge_jacobian" and se.get("exact_identifiable_fraction") == 1.0 and se.get("exact_decoder_max_error", float("inf")) <= THRESHOLDS["decoder_max_error_max"],
        "lower_cost_boundary": max(lower_functional, default=0.0) == 0.0 and max(lower_exact, default=0.0) == 0.0,
        "functional_cheaper_than_exact": metrics["selected_functional_cost"] < metrics["selected_exact_cost"],
        "gauge": metrics["gauge_metric_gap"] <= THRESHOLDS["gauge_metric_gap_max"],
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "probe_summary": probe_summary,
        "gauge_summary": gauge_summaries,
        "pair_metric_digest": digest(pair_metrics),
    }


def expected_pair_count(split: str) -> int:
    return len(FAMILIES_BY_SPLIT[split]) * len(GAUGES) * len(VERTICES) * REPLICATES


def source_hashes() -> dict[str, str]:
    return {"main": sha256_file(SCRIPT), "audit": sha256_file(AUDIT_SCRIPT)}


def protocol_payload() -> dict[str, Any]:
    final1211 = read_json(SOURCE1211 / "analysis/final.json")
    audit1211 = read_json(SOURCE1211 / "audit/independent_result_audit.json")
    validate_digest(final1211, "final_digest")
    validate_digest(audit1211, "audit_digest")
    checks = {
        "phase1211_final_frozen": final1211["final_digest"] == EXPECTED_1211_FINAL,
        "phase1211_audit_frozen": audit1211["audit_digest"] == EXPECTED_1211_AUDIT,
        "phase1211_audit_passed": audit1211["all_checks_passed"] is True,
        "phase1211_boundary_confirmed": final1211["known_truth_identifiability_boundary_confirmed"] is True,
        "phase1211_transfer_denied": final1211["compiler_transfer_authorized"] is False,
        "functional_target_frozen_before_probe_selection": True,
        "known_truth_only": True,
        "pretrained_models_forbidden": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    payload = {
        "phase": PHASE,
        "schema_version": "phase1212.functional_target_quotient.v1",
        "created_at_utc": utc_now(),
        "title": "Functional target quotient and typed minimum-observation calibration",
        "source_hashes": source_hashes(),
        "source_phase1211_final_digest": final1211["final_digest"],
        "source_phase1211_audit_digest": audit1211["audit_digest"],
        "dimension": DIMENSION,
        "functional_dimension": FUNCTION_DIMENSION,
        "replicates": REPLICATES,
        "families_by_split": {key: list(value) for key, value in FAMILIES_BY_SPLIT.items()},
        "gauges": list(GAUGES),
        "vertices": [list(vertex) for vertex in VERTICES],
        "active_codes": [list(code) for code in ACTIVE_CODES],
        "null_codes": list(NULL_CODES),
        "worlds_per_pair": len(ACTIVE_CODES) * len(NULL_CODES),
        "pairs_per_split": {split: expected_pair_count(split) for split in FAMILIES_BY_SPLIT},
        "systems_per_split": {split: expected_pair_count(split) * len(ACTIVE_CODES) * len(NULL_CODES) for split in FAMILIES_BY_SPLIT},
        "probes": [dict(row) for row in PROBES],
        "thresholds": THRESHOLDS,
        "target_contract": {
            "exact": "the full 16-dimensional target state",
            "functional": "the four-dimensional registered downstream readout; states in its nullspace are equivalent",
            "claim": "target type is frozen before any probe is ranked",
        },
        "completion_contract": (
            "All 32 worlds share seven visible vertices. Four active bits produce 16 functional classes; one readout-null "
            "bit doubles exact-state classes to 32 without changing functional identity."
        ),
        "selection_contract": (
            "Discovery selects the lowest-cost qualified probe within the frozen typed registry, breaking cost ties by "
            "frozen priority. Confirmation uses unseen mechanism families and cannot add probes."
        ),
        "execution_order": [
            "preregister and independent zero-output preaudit",
            "generate discovery public observations and sealed opaque-world probe responses",
            "reveal discovery target classes, score all frozen probes, and seal selections",
            "generate confirmation using unseen mechanism families and the frozen registry",
            "reveal confirmation targets and test the frozen selections",
            "independent exact regeneration audit",
        ],
        "new_math_upgrade_gate": {
            "stable_empirical_object_across_natural_models": False,
            "stable_valid_intervention_family": False,
            "stable_composition_law": False,
            "existing_mathematics_compression_failure": False,
            "status": "OPEN_NOT_CONFIRMED",
        },
        "hard_stops": [
            "No Qwen3, GLM4, DS7B, free Transformer, head, layer, or neuron scan is authorized.",
            "No probe may be added after preregistration or after discovery selection.",
            "Minimum means minimum only within the frozen typed registry and cost model.",
            "Functional target identification is not exact hidden-state identification.",
            "Probe responses contain no active-code, null-code, family, gauge, or target labels before truth reveal.",
            "A passed known-truth quotient calibration does not authorize language-model transfer.",
        ],
        "checks": checks,
    }
    payload["protocol_digest"] = digest(payload)
    return payload


def preregister() -> dict[str, Any]:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1212 artifacts")
    payload = protocol_payload()
    write_json(PROTOCOL_PATH, payload)
    print(json.dumps({"protocol_digest": payload["protocol_digest"], "pairs_per_split": payload["pairs_per_split"]}, indent=2))
    return payload


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_digest(protocol, "protocol_digest")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source hash drift")
    return protocol


def require_preaudit() -> dict[str, Any]:
    preaudit = read_json(PREAUDIT_PATH)
    validate_digest(preaudit, "audit_digest")
    if not preaudit["all_checks_passed"] or preaudit["protocol_digest"] != verify_protocol()["protocol_digest"]:
        raise RuntimeError("independent preaudit failed")
    return preaudit


def generate_split(split: str) -> dict[str, Any]:
    protocol = verify_protocol()
    require_preaudit()
    if split_root(split).exists():
        raise RuntimeError(f"{split} already generated")
    if split == "confirmation":
        selection = read_json(SELECTION_PATH)
        validate_digest(selection, "selection_digest")
        if not selection["confirmation_authorized"]:
            raise RuntimeError("discovery denied confirmation")
    public, probes = [], []
    for spec in specs_for(split):
        public.append(public_row(spec))
        probes.extend(probe_rows(spec))
    write_jsonl_gz(split_root(split) / "public_observations.jsonl.gz", public)
    write_jsonl_gz(split_root(split) / "sealed_probe_responses.jsonl.gz", probes)
    truth_path = split_root(split) / "sealed_target_truth.jsonl.gz"
    manifest = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "pair_count": len(public),
        "probe_row_count": len(probes),
        "public_digest": digest(public),
        "probe_digest": digest(probes),
        "truth_absent_at_probe_seal": not truth_path.exists(),
        "created_at_utc": utc_now(),
    }
    manifest["manifest_digest"] = digest(manifest)
    write_json(split_root(split) / "probe_manifest.json", manifest)
    print(json.dumps({"split": split, "pair_count": len(public), "probe_row_count": len(probes), "manifest_digest": manifest["manifest_digest"]}, indent=2))
    return manifest


def reveal_split(split: str) -> dict[str, Any]:
    protocol = verify_protocol()
    require_preaudit()
    manifest = read_json(split_root(split) / "probe_manifest.json")
    validate_digest(manifest, "manifest_digest")
    truth_path = split_root(split) / "sealed_target_truth.jsonl.gz"
    if truth_path.exists() or not manifest["truth_absent_at_probe_seal"]:
        raise RuntimeError("target truth was not sealed after probe responses")
    public = read_jsonl_gz(split_root(split) / "public_observations.jsonl.gz")
    probes = read_jsonl_gz(split_root(split) / "sealed_probe_responses.jsonl.gz")
    truth = []
    for spec in specs_for(split):
        truth.extend(truth_rows(spec))
    write_jsonl_gz(truth_path, truth)
    score = aggregate_score(public, probes, truth)
    value = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "manifest_digest": manifest["manifest_digest"],
        **score,
        "gate": all(score["checks"].values()),
    }
    value["score_digest"] = digest(value)
    write_json(OUT_ROOT / "analysis" / f"{split}_score.json", value)
    if split == "discovery":
        selection = {
            "phase": PHASE,
            "protocol_digest": protocol["protocol_digest"],
            "discovery_score_digest": value["score_digest"],
            "functional_probe": value["metrics"]["selected_functional_probe"],
            "exact_probe": value["metrics"]["selected_exact_probe"],
            "functional_cost": value["metrics"]["selected_functional_cost"],
            "exact_cost": value["metrics"]["selected_exact_cost"],
            "confirmation_authorized": value["gate"],
        }
        selection["selection_digest"] = digest(selection)
        write_json(SELECTION_PATH, selection)
    print(json.dumps({"split": split, "metrics": value["metrics"], "checks": value["checks"], "gate": value["gate"]}, indent=2))
    return value


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
    confirmation = read_json(OUT_ROOT / "analysis/confirmation_score.json")
    selection = read_json(SELECTION_PATH)
    validate_digest(discovery, "score_digest")
    validate_digest(confirmation, "score_digest")
    validate_digest(selection, "selection_digest")
    confirmation_selection_matches = (
        confirmation["metrics"]["selected_functional_probe"] == selection["functional_probe"]
        and confirmation["metrics"]["selected_exact_probe"] == selection["exact_probe"]
    )
    confirmed = bool(discovery["gate"] and confirmation["gate"] and confirmation_selection_matches)
    value = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "status": "functional_target_quotient_and_typed_probe_boundary_confirmed" if confirmed else "functional_target_quotient_calibration_not_confirmed",
        "known_truth_quotient_calibration_confirmed": confirmed,
        "discovery_metrics": discovery["metrics"],
        "confirmation_metrics": confirmation["metrics"],
        "selection": selection,
        "confirmation_selection_matches": confirmation_selection_matches,
        "free_transformer_transfer_authorized": False,
        "language_model_transfer_authorized": False,
        "new_math_hypothesis": {
            "status": "OPEN_NOT_CONFIRMED",
            "upgrade_gates_passed": 0,
            "upgrade_gate_count": 4,
            "reason": "Functional quotient identification and typed probe costs are expressed by finite equivalence classes, rank, and interpolation.",
        },
        "new_k_item": {
            "id": "K192",
            "level": "E3-KT",
            "statement": (
                "In 32 observation-equivalent completions, 32 exact target states collapse to 16 registered functional "
                "classes. A frozen four-channel edge-response probe identifies and reconstructs the functional quotient "
                "across unseen mechanism families and gauges while remaining non-identifying for exact state; the frozen "
                "full-state edge probe identifies all 32 exact targets. Target type therefore changes the minimum qualified "
                "observation within the preregistered typed probe registry."
            ),
        },
        "claim_boundary": (
            "This is a known-truth target-quotient and typed-registry calibration. It does not prove a globally minimal probe, "
            "a natural-language semantic quotient, or a valid intervention in a pretrained model."
        ),
        "authorized_next": (
            "Design an independent free-TinyTransformer protocol that defines functional equivalence behaviorally and tests "
            "the frozen quotient-aware probe contract before any Qwen3 transfer."
        ),
        "auto_continue": False,
    }
    value["final_digest"] = digest(value)
    write_json(FINAL_PATH, value)
    print(json.dumps(value, indent=2))
    return value


def cli() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preregister")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    reveal = subparsers.add_parser("reveal")
    reveal.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "preregister":
        preregister()
    elif args.command == "generate":
        generate_split(args.split)
    elif args.command == "reveal":
        reveal_split(args.split)
    else:
        finalize()


if __name__ == "__main__":
    cli()
