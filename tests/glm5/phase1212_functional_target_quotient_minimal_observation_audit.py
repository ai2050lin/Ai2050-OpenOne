#!/usr/bin/env python3
"""Independent protocol and exact-regeneration audit for Phase1212."""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import phase1212_functional_target_quotient_minimal_observation as main


FINAL_AUDIT_PATH = main.OUT_ROOT / "audit/independent_result_audit.json"
EXPECTED_DIMENSION = 16
EXPECTED_FUNCTION_DIMENSION = 4
EXPECTED_REPLICATES = 16
EXPECTED_GAUGES = ("signed_permutation", "orthogonal_dense")
EXPECTED_FAMILIES = {
    "discovery": ("additive", "semantic_pair"),
    "confirmation": ("carrier_pair", "full_pairwise"),
}
EXPECTED_VERTICES = tuple((a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))
EXPECTED_ACTIVE_CODES = tuple(
    tuple((mask >> index) & 1 for index in range(EXPECTED_FUNCTION_DIMENSION))
    for mask in range(1 << EXPECTED_FUNCTION_DIMENSION)
)
EXPECTED_NULL_CODES = (0, 1)
EXPECTED_PROBES = (
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
EXPECTED_PROBE_NAMES = tuple(row["name"] for row in EXPECTED_PROBES)
EXPECTED_PROBE_BY_NAME = {row["name"]: row for row in EXPECTED_PROBES}
TOLERANCE = 1.0e-9


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def finish(
    checks: list[dict[str, Any]],
    stage: str,
    path: Path,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "phase": main.PHASE,
        "stage": stage,
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "failed_count": sum(not row["passed"] for row in checks),
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
        **(extras or {}),
    }
    value["audit_digest"] = main.digest(value)
    main.write_json(path, value)
    return value


def numeric_max_error(left: Any, right: Any) -> float:
    errors: list[float] = []

    def walk(a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                errors.append(float("inf"))
                return
            for key in sorted(a):
                walk(a[key], b[key])
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                errors.append(float("inf"))
                return
            for x, y in zip(a, b):
                walk(x, y)
        elif (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and not isinstance(a, bool)
            and not isinstance(b, bool)
        ):
            errors.append(abs(float(a) - float(b)))
        elif a != b:
            errors.append(float("inf"))

    walk(left, right)
    return max(errors, default=0.0)


def independent_seed(split: str, family: str, target_index: int, replicate: int) -> int:
    key = f"phase1212|latent|{split}|{family}|{target_index}|{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def independent_gauge_seed(split: str, family: str, gauge: str, target_index: int, replicate: int) -> int:
    key = f"phase1212|gauge|{split}|{family}|{gauge}|{target_index}|{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def independent_design(vertex: tuple[float, float, float]) -> np.ndarray:
    a, b, c = vertex
    return np.asarray((1.0, a, b, c, a * b, a * c, b * c), dtype=np.float64)


def independent_derivative_design(vertex: tuple[float, float, float], axis: int) -> np.ndarray:
    a, b, c = vertex
    rows = (
        (0.0, 1.0, 0.0, 0.0, b, c, 0.0),
        (0.0, 0.0, 1.0, 0.0, a, 0.0, c),
        (0.0, 0.0, 0.0, 1.0, 0.0, a, b),
    )
    return np.asarray(rows[axis], dtype=np.float64)


def independent_indicator(vertex: tuple[float, float, float], target: tuple[int, int, int]) -> float:
    result = 1.0
    for value, target_value in zip(vertex, target):
        result *= (1.0 + value * target_value) / 2.0
    return float(result)


def independent_indicator_derivative(
    vertex: tuple[float, float, float],
    target: tuple[int, int, int],
    axis: int,
) -> float:
    result = target[axis] / 2.0
    for index in range(3):
        if index != axis:
            result *= (1.0 + vertex[index] * target[index]) / 2.0
    return float(result)


def independent_gauge(rng: np.random.Generator, gauge: str) -> np.ndarray:
    if gauge == "signed_permutation":
        permutation = rng.permutation(EXPECTED_DIMENSION)
        signs = rng.choice((-1.0, 1.0), size=EXPECTED_DIMENSION)
        matrix = np.zeros((EXPECTED_DIMENSION, EXPECTED_DIMENSION), dtype=np.float64)
        matrix[np.arange(EXPECTED_DIMENSION), permutation] = signs
        return matrix
    raw = rng.normal(size=(EXPECTED_DIMENSION, EXPECTED_DIMENSION))
    q, r = np.linalg.qr(raw)
    return (q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))).astype(np.float64)


def independent_active_basis(family: str) -> set[int]:
    return {
        "additive": {0, 1, 2, 3},
        "semantic_pair": {0, 1, 2, 3, 4},
        "carrier_pair": {0, 1, 2, 3, 5, 6},
        "full_pairwise": set(range(7)),
    }[family]


def independent_spec(
    split: str,
    family: str,
    gauge: str,
    target_index: int,
    replicate: int,
) -> dict[str, Any]:
    latent_rng = np.random.default_rng(independent_seed(split, family, target_index, replicate))
    gauge_rng = np.random.default_rng(independent_gauge_seed(split, family, gauge, target_index, replicate))
    target = EXPECTED_VERTICES[target_index]
    recipient = (-target[0], target[1], target[2])
    amplitudes = latent_rng.uniform(0.75, 1.75, size=7) * latent_rng.choice((-1.0, 1.0), size=7)
    coefficients = np.zeros((7, EXPECTED_DIMENSION), dtype=np.float64)
    for index in independent_active_basis(family):
        coefficients[index, index] = amplitudes[index]
    gauge_value = independent_gauge(gauge_rng, gauge)
    base_latent = np.stack([independent_design(vertex) @ coefficients for vertex in EXPECTED_VERTICES])
    active_scales = latent_rng.uniform(0.75, 1.75, size=EXPECTED_FUNCTION_DIMENSION) * latent_rng.choice(
        (-1.0, 1.0), size=EXPECTED_FUNCTION_DIMENSION
    )
    active_latent = np.zeros((EXPECTED_FUNCTION_DIMENSION, EXPECTED_DIMENSION), dtype=np.float64)
    for index in range(EXPECTED_FUNCTION_DIMENSION):
        active_latent[index, index] = active_scales[index]
    null_latent = np.zeros(EXPECTED_DIMENSION, dtype=np.float64)
    null_latent[7] = float(latent_rng.uniform(0.75, 1.75) * latent_rng.choice((-1.0, 1.0)))
    readout_latent = np.zeros((EXPECTED_DIMENSION, EXPECTED_FUNCTION_DIMENSION), dtype=np.float64)
    readout_latent[:EXPECTED_FUNCTION_DIMENSION, :] = np.eye(EXPECTED_FUNCTION_DIMENSION)
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
        "base_hidden": base_latent @ gauge_value.T,
        "active_hidden": active_latent @ gauge_value.T,
        "null_hidden": null_latent @ gauge_value.T,
        "readout_hidden": gauge_value @ readout_latent,
    }


def independent_specs(split: str) -> Iterable[dict[str, Any]]:
    for family in EXPECTED_FAMILIES[split]:
        for gauge in EXPECTED_GAUGES:
            for target_index in range(len(EXPECTED_VERTICES)):
                for replicate in range(EXPECTED_REPLICATES):
                    yield independent_spec(split, family, gauge, target_index, replicate)


def independent_world_id(pair_id: str, active_code: tuple[int, ...], null_code: int) -> str:
    key = f"{pair_id}|{''.join(map(str, active_code))}|{null_code}"
    return "w" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def independent_worlds(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for active_code in EXPECTED_ACTIVE_CODES:
        for null_code in EXPECTED_NULL_CODES:
            delta = np.sum(np.asarray(active_code, dtype=np.float64)[:, None] * spec["active_hidden"], axis=0)
            delta = delta + null_code * spec["null_hidden"]
            yield {
                "world_id": independent_world_id(spec["pair_id"], active_code, null_code),
                "active_code": active_code,
                "null_code": null_code,
                "delta_hidden": delta,
            }


def independent_base(spec: dict[str, Any], vertex: tuple[float, float, float]) -> np.ndarray:
    return (independent_design(vertex) @ spec["coefficients"]) @ spec["gauge_matrix"].T


def independent_state(
    spec: dict[str, Any],
    world: dict[str, Any],
    vertex: tuple[float, float, float],
) -> np.ndarray:
    return independent_base(spec, vertex) + independent_indicator(vertex, spec["target"]) * world["delta_hidden"]


def independent_derivative(
    spec: dict[str, Any],
    world: dict[str, Any],
    vertex: tuple[float, float, float],
    axis: int,
) -> np.ndarray:
    base = (independent_derivative_design(vertex, axis) @ spec["coefficients"]) @ spec["gauge_matrix"].T
    return base + independent_indicator_derivative(vertex, spec["target"], axis) * world["delta_hidden"]


def independent_public(spec: dict[str, Any]) -> dict[str, Any]:
    observed = [index for index in range(len(EXPECTED_VERTICES)) if index != spec["target_index"]]
    return {
        "pair_id": spec["pair_id"],
        "split": spec["split"],
        "target": list(spec["target"]),
        "recipient": list(spec["recipient"]),
        "observed_vertices": [list(EXPECTED_VERTICES[index]) for index in observed],
        "observed_states": [spec["base_hidden"][index].tolist() for index in observed],
        "functional_readout": spec["readout_hidden"].tolist(),
        "probe_contract": [dict(row) for row in EXPECTED_PROBES],
    }


def independent_probe(spec: dict[str, Any], world: dict[str, Any], probe: str) -> np.ndarray:
    recipient = spec["recipient"]
    target = spec["target"]
    midpoint = (0.0, float(target[1]), float(target[2]))
    readout = spec["readout_hidden"]
    if probe == "baseline_none":
        return np.zeros(1, dtype=np.float64)
    if probe == "visible_functional_value_control":
        return independent_state(spec, world, recipient) @ readout
    if probe == "wrong_axis_functional_jacobian_control":
        return independent_derivative(spec, world, recipient, 1) @ readout
    edge = independent_derivative(spec, world, recipient, 0) @ readout
    if probe == "functional_edge_ch0":
        return edge[:1]
    if probe == "functional_edge_ch01":
        return edge[:2]
    if probe == "functional_edge_ch012":
        return edge[:3]
    if probe == "functional_edge_all":
        return edge
    if probe == "functional_midpoint_all":
        return independent_state(spec, world, midpoint) @ readout
    if probe == "full_edge_jacobian":
        return independent_derivative(spec, world, recipient, 0)
    if probe == "full_midpoint":
        return independent_state(spec, world, midpoint)
    raise KeyError(probe)


def independent_probe_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for world in independent_worlds(spec):
        for probe in EXPECTED_PROBE_NAMES:
            rows.append({
                "pair_id": spec["pair_id"],
                "split": spec["split"],
                "world_id": world["world_id"],
                "probe": probe,
                "response": independent_probe(spec, world, probe).tolist(),
            })
    return rows


def independent_truth(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    visible_digest = main.digest(independent_public(spec)["observed_states"])
    for world in independent_worlds(spec):
        exact = independent_state(spec, world, spec["target"])
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
            "visible_state_digest": visible_digest,
        })
    return rows


def independent_lookup(public: dict[str, Any]) -> dict[tuple[int, int, int], np.ndarray]:
    return {
        tuple(int(value) for value in vertex): np.asarray(state, dtype=np.float64)
        for vertex, state in zip(public["observed_vertices"], public["observed_states"])
    }


def independent_decode(public: dict[str, Any], probe: str, response: np.ndarray) -> dict[str, np.ndarray | None]:
    lookup = independent_lookup(public)
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


def near(left: np.ndarray, right: np.ndarray) -> bool:
    return float(np.linalg.norm(left - right)) <= TOLERANCE


def independent_class_count(values: list[np.ndarray]) -> int:
    representatives: list[np.ndarray] = []
    for value in values:
        if not any(near(value, representative) for representative in representatives):
            representatives.append(value)
    return len(representatives)


def independent_identifiable(signatures: list[np.ndarray], targets: list[np.ndarray]) -> bool:
    for left in range(len(signatures)):
        for right in range(left + 1, len(signatures)):
            if near(signatures[left], signatures[right]) and not near(targets[left], targets[right]):
                return False
    return True


def independent_aggregate(
    public_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    public_by_pair = {row["pair_id"]: row for row in public_rows}
    truth_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    response_groups: dict[tuple[str, str], dict[str, np.ndarray]] = defaultdict(dict)
    for row in truth_rows:
        truth_groups[row["pair_id"]].append(row)
    for row in probe_rows:
        response_groups[(row["pair_id"], row["probe"])][row["world_id"]] = np.asarray(
            row["response"], dtype=np.float64
        )

    pair_metrics: list[dict[str, Any]] = []
    for pair_id, worlds in truth_groups.items():
        worlds = sorted(worlds, key=lambda row: row["world_id"])
        exact_targets = [np.asarray(row["exact_target"], dtype=np.float64) for row in worlds]
        functional_targets = [np.asarray(row["functional_target"], dtype=np.float64) for row in worlds]
        for probe in EXPECTED_PROBE_NAMES:
            signatures = [response_groups[(pair_id, probe)][row["world_id"]] for row in worlds]
            exact_errors: list[float] = []
            functional_errors: list[float] = []
            for row, signature, exact_target, functional_target in zip(
                worlds, signatures, exact_targets, functional_targets
            ):
                decoded = independent_decode(public_by_pair[pair_id], probe, signature)
                if decoded["exact"] is not None:
                    exact_errors.append(float(np.linalg.norm(decoded["exact"] - exact_target)))
                if decoded["functional"] is not None:
                    functional_errors.append(float(np.linalg.norm(decoded["functional"] - functional_target)))
            pair_metrics.append({
                "pair_id": pair_id,
                "family": worlds[0]["family"],
                "gauge": worlds[0]["gauge"],
                "probe": probe,
                "functional_identifiable": independent_identifiable(signatures, functional_targets),
                "exact_identifiable": independent_identifiable(signatures, exact_targets),
                "signature_class_count": independent_class_count(signatures),
                "functional_class_count": independent_class_count(functional_targets),
                "exact_class_count": independent_class_count(exact_targets),
                "functional_decoder_max_error": None if not functional_errors else max(functional_errors),
                "exact_decoder_max_error": None if not exact_errors else max(exact_errors),
            })

    summaries: dict[str, Any] = {}
    for probe in EXPECTED_PROBE_NAMES:
        members = [row for row in pair_metrics if row["probe"] == probe]
        functional_errors = [
            row["functional_decoder_max_error"]
            for row in members
            if row["functional_decoder_max_error"] is not None
        ]
        exact_errors = [
            row["exact_decoder_max_error"] for row in members if row["exact_decoder_max_error"] is not None
        ]
        summaries[probe] = {
            "cost": EXPECTED_PROBE_BY_NAME[probe]["cost"],
            "priority": EXPECTED_PROBE_BY_NAME[probe]["priority"],
            "kind": EXPECTED_PROBE_BY_NAME[probe]["kind"],
            "functional_identifiable_fraction": float(np.mean([row["functional_identifiable"] for row in members])),
            "exact_identifiable_fraction": float(np.mean([row["exact_identifiable"] for row in members])),
            "mean_signature_class_count": float(np.mean([row["signature_class_count"] for row in members])),
            "functional_decoder_max_error": None if not functional_errors else float(max(functional_errors)),
            "exact_decoder_max_error": None if not exact_errors else float(max(exact_errors)),
        }

    thresholds = main.THRESHOLDS
    functional_candidates = [
        probe
        for probe, row in summaries.items()
        if row["functional_identifiable_fraction"] >= thresholds["identifiable_fraction_min"]
        and row["functional_decoder_max_error"] is not None
        and row["functional_decoder_max_error"] <= thresholds["decoder_max_error_max"]
    ]
    exact_candidates = [
        probe
        for probe, row in summaries.items()
        if row["exact_identifiable_fraction"] >= thresholds["identifiable_fraction_min"]
        and row["exact_decoder_max_error"] is not None
        and row["exact_decoder_max_error"] <= thresholds["decoder_max_error_max"]
    ]
    rank_key = lambda probe: (
        EXPECTED_PROBE_BY_NAME[probe]["cost"],
        EXPECTED_PROBE_BY_NAME[probe]["priority"],
    )
    selected_functional = min(functional_candidates, key=rank_key) if functional_candidates else None
    selected_exact = min(exact_candidates, key=rank_key) if exact_candidates else None

    gauge_summary: dict[str, dict[str, dict[str, float]]] = {}
    for gauge in EXPECTED_GAUGES:
        gauge_summary[gauge] = {}
        for probe in EXPECTED_PROBE_NAMES:
            members = [row for row in pair_metrics if row["gauge"] == gauge and row["probe"] == probe]
            gauge_summary[gauge][probe] = {
                "functional_identifiable_fraction": float(
                    np.mean([row["functional_identifiable"] for row in members])
                ),
                "exact_identifiable_fraction": float(np.mean([row["exact_identifiable"] for row in members])),
                "mean_signature_class_count": float(np.mean([row["signature_class_count"] for row in members])),
            }
    gauge_gap = max(
        abs(gauge_summary[EXPECTED_GAUGES[0]][probe][metric] - gauge_summary[EXPECTED_GAUGES[1]][probe][metric])
        for probe in EXPECTED_PROBE_NAMES
        for metric in gauge_summary[EXPECTED_GAUGES[0]][probe]
    )
    functional_counts = [
        row["functional_class_count"] for row in pair_metrics if row["probe"] == "baseline_none"
    ]
    exact_counts = [row["exact_class_count"] for row in pair_metrics if row["probe"] == "baseline_none"]
    visible_digests: dict[str, set[str]] = defaultdict(set)
    for row in truth_rows:
        visible_digests[row["pair_id"]].add(row["visible_state_digest"])
    finite_values = [value for row in probe_rows for value in row["response"]]
    finite_values += [
        value
        for row in truth_rows
        for key in ("exact_target", "functional_target")
        for value in row[key]
    ]
    metrics = {
        "pair_count": len(public_rows),
        "system_count": len(truth_rows),
        "probe_row_count": len(probe_rows),
        "finite_fraction": float(np.mean([math.isfinite(value) for value in finite_values])),
        "visible_world_identity_max": 0.0
        if all(len(values) == 1 for values in visible_digests.values())
        else float("inf"),
        "functional_class_count_min": int(min(functional_counts)),
        "functional_class_count_max": int(max(functional_counts)),
        "exact_class_count_min": int(min(exact_counts)),
        "exact_class_count_max": int(max(exact_counts)),
        "selected_functional_probe": selected_functional,
        "selected_exact_probe": selected_exact,
        "selected_functional_cost": None
        if selected_functional is None
        else EXPECTED_PROBE_BY_NAME[selected_functional]["cost"],
        "selected_exact_cost": None if selected_exact is None else EXPECTED_PROBE_BY_NAME[selected_exact]["cost"],
        "gauge_metric_gap": float(gauge_gap),
    }
    sf = summaries.get(selected_functional, {})
    se = summaries.get(selected_exact, {})
    lower_functional = [
        row["functional_identifiable_fraction"]
        for row in summaries.values()
        if selected_functional is not None and row["cost"] < EXPECTED_PROBE_BY_NAME[selected_functional]["cost"]
    ]
    lower_exact = [
        row["exact_identifiable_fraction"]
        for row in summaries.values()
        if selected_exact is not None and row["cost"] < EXPECTED_PROBE_BY_NAME[selected_exact]["cost"]
    ]
    checks = {
        "finite": metrics["finite_fraction"] >= thresholds["finite_fraction_min"],
        "visible_worlds": metrics["visible_world_identity_max"] <= thresholds["visible_world_identity_max"],
        "target_class_counts": metrics["functional_class_count_min"]
        == metrics["functional_class_count_max"]
        == thresholds["functional_class_count"]
        and metrics["exact_class_count_min"]
        == metrics["exact_class_count_max"]
        == thresholds["exact_class_count"],
        "baseline_nonidentifiable": summaries["baseline_none"]["functional_identifiable_fraction"]
        <= thresholds["nonidentifiable_fraction_max"]
        and summaries["baseline_none"]["exact_identifiable_fraction"]
        <= thresholds["nonidentifiable_fraction_max"],
        "partial_functional_insufficient": all(
            summaries[name]["functional_identifiable_fraction"] <= thresholds["nonidentifiable_fraction_max"]
            for name in ("functional_edge_ch0", "functional_edge_ch01", "functional_edge_ch012")
        ),
        "matched_controls_insufficient": all(
            summaries[name]["functional_identifiable_fraction"] <= thresholds["nonidentifiable_fraction_max"]
            and summaries[name]["exact_identifiable_fraction"] <= thresholds["nonidentifiable_fraction_max"]
            for name in ("visible_functional_value_control", "wrong_axis_functional_jacobian_control")
        ),
        "functional_probe_selected": selected_functional == "functional_edge_all"
        and sf.get("functional_identifiable_fraction") == 1.0
        and sf.get("functional_decoder_max_error", float("inf")) <= thresholds["decoder_max_error_max"],
        "functional_probe_not_exact": sf.get("exact_identifiable_fraction") == 0.0,
        "exact_probe_selected": selected_exact == "full_edge_jacobian"
        and se.get("exact_identifiable_fraction") == 1.0
        and se.get("exact_decoder_max_error", float("inf")) <= thresholds["decoder_max_error_max"],
        "lower_cost_boundary": max(lower_functional, default=0.0) == 0.0
        and max(lower_exact, default=0.0) == 0.0,
        "functional_cheaper_than_exact": metrics["selected_functional_cost"] < metrics["selected_exact_cost"],
        "gauge": metrics["gauge_metric_gap"] <= thresholds["gauge_metric_gap_max"],
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "probe_summary": summaries,
        "gauge_summary": gauge_summary,
        "pair_metric_digest": main.digest(pair_metrics),
    }


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def preaudit() -> dict[str, Any]:
    if main.PREAUDIT_PATH.exists():
        raise RuntimeError("Phase1212 preaudit already exists")
    checks: list[dict[str, Any]] = []
    protocol = main.read_json(main.PROTOCOL_PATH)
    clean = dict(protocol)
    stored = clean.pop("protocol_digest")
    add(checks, "protocol_digest", main.digest(clean) == stored)
    add(checks, "source_hashes", protocol["source_hashes"] == main.source_hashes())
    add(checks, "main_hash", protocol["source_hashes"]["main"] == main.sha256_file(main.SCRIPT))
    add(checks, "audit_hash", protocol["source_hashes"]["audit"] == main.sha256_file(Path(__file__).resolve()))
    add(checks, "phase1211_final", protocol["source_phase1211_final_digest"] == main.EXPECTED_1211_FINAL)
    add(checks, "phase1211_audit", protocol["source_phase1211_audit_digest"] == main.EXPECTED_1211_AUDIT)
    add(checks, "protocol_checks", all(protocol["checks"].values()), protocol["checks"])
    add(checks, "dimension", protocol["dimension"] == EXPECTED_DIMENSION)
    add(checks, "functional_dimension", protocol["functional_dimension"] == EXPECTED_FUNCTION_DIMENSION)
    add(checks, "replicates", protocol["replicates"] == EXPECTED_REPLICATES)
    add(checks, "families", protocol["families_by_split"] == {key: list(value) for key, value in EXPECTED_FAMILIES.items()})
    add(checks, "family_holdout", set(EXPECTED_FAMILIES["discovery"]).isdisjoint(EXPECTED_FAMILIES["confirmation"]))
    add(checks, "gauges", tuple(protocol["gauges"]) == EXPECTED_GAUGES)
    add(checks, "vertices", tuple(tuple(row) for row in protocol["vertices"]) == EXPECTED_VERTICES)
    add(checks, "active_codes", tuple(tuple(row) for row in protocol["active_codes"]) == EXPECTED_ACTIVE_CODES)
    add(checks, "null_codes", tuple(protocol["null_codes"]) == EXPECTED_NULL_CODES)
    add(checks, "world_count", protocol["worlds_per_pair"] == 32)
    add(checks, "pair_counts", protocol["pairs_per_split"] == {"discovery": 512, "confirmation": 512})
    add(checks, "system_counts", protocol["systems_per_split"] == {"discovery": 16384, "confirmation": 16384})
    add(checks, "probe_registry", tuple(protocol["probes"]) == EXPECTED_PROBES)
    add(checks, "probe_names_unique", len(set(EXPECTED_PROBE_NAMES)) == len(EXPECTED_PROBE_NAMES))
    add(checks, "priorities_unique", len({row["priority"] for row in EXPECTED_PROBES}) == len(EXPECTED_PROBES))
    add(checks, "cost_contract", [row["cost"] for row in EXPECTED_PROBES] == [0, 1, 2, 3, 4, 4, 4, 4, 16, 16])
    add(checks, "thresholds", protocol["thresholds"] == main.THRESHOLDS)
    add(checks, "target_types_distinct", "full 16-dimensional" in protocol["target_contract"]["exact"] and "four-dimensional" in protocol["target_contract"]["functional"])
    add(checks, "minimum_scope_limited", any("Minimum means minimum only" in row for row in protocol["hard_stops"]))
    sample = independent_spec("discovery", "additive", "orthogonal_dense", 0, 0)
    readout_rank = int(np.linalg.matrix_rank(sample["readout_hidden"]))
    null_readout_norm = float(np.linalg.norm(sample["null_hidden"] @ sample["readout_hidden"]))
    add(checks, "readout_rank", readout_rank == EXPECTED_FUNCTION_DIMENSION, readout_rank)
    add(checks, "null_direction_functionally_silent", null_readout_norm <= 1.0e-12, null_readout_norm)
    sample_specs = [
        independent_spec("discovery", "additive", gauge, 0, 0)
        for gauge in EXPECTED_GAUGES
    ]
    sample_public = [independent_public(spec) for spec in sample_specs]
    sample_probe_rows = [row for spec in sample_specs for row in independent_probe_rows(spec)]
    sample_truth = [row for spec in sample_specs for row in independent_truth(spec)]
    sample_score = independent_aggregate(sample_public, sample_probe_rows, sample_truth)
    add(checks, "sample_class_counts", sample_score["metrics"]["functional_class_count_min"] == 16 and sample_score["metrics"]["exact_class_count_min"] == 32)
    add(checks, "sample_partial_signature_ladder", [sample_score["probe_summary"][name]["mean_signature_class_count"] for name in ("functional_edge_ch0", "functional_edge_ch01", "functional_edge_ch012", "functional_edge_all")] == [2.0, 4.0, 8.0, 16.0])
    add(checks, "sample_functional_boundary", sample_score["metrics"]["selected_functional_probe"] == "functional_edge_all" and sample_score["probe_summary"]["functional_edge_all"]["exact_identifiable_fraction"] == 0.0)
    add(checks, "sample_exact_boundary", sample_score["metrics"]["selected_exact_probe"] == "full_edge_jacobian" and sample_score["probe_summary"]["full_edge_jacobian"]["exact_identifiable_fraction"] == 1.0)
    add(checks, "decoder_algebra", sample_score["probe_summary"]["functional_edge_all"]["functional_decoder_max_error"] <= 1.0e-10 and sample_score["probe_summary"]["full_edge_jacobian"]["exact_decoder_max_error"] <= 1.0e-10)
    add(checks, "new_math_open", protocol["new_math_upgrade_gate"]["status"] == "OPEN_NOT_CONFIRMED" and not any(value for key, value in protocol["new_math_upgrade_gate"].items() if key != "status"))
    imports = imported_modules(main.SCRIPT)
    add(checks, "no_model_runtime", "torch" not in imports and "transformers" not in imports, sorted(imports))
    add(checks, "known_truth_hard_stop", any("No Qwen3" in row for row in protocol["hard_stops"]))
    forbidden = [main.OUT_ROOT / "runs", main.OUT_ROOT / "analysis", main.FINAL_PATH, FINAL_AUDIT_PATH]
    add(checks, "zero_formal_outputs", not any(path.exists() for path in forbidden))
    return finish(
        checks,
        "independent zero-output target-type and typed-probe audit",
        main.PREAUDIT_PATH,
        {"protocol_digest": protocol["protocol_digest"]},
    )


def audit_split(checks: list[dict[str, Any]], split: str) -> dict[str, Any]:
    root = main.split_root(split)
    public = main.read_jsonl_gz(root / "public_observations.jsonl.gz")
    probes = main.read_jsonl_gz(root / "sealed_probe_responses.jsonl.gz")
    truth = main.read_jsonl_gz(root / "sealed_target_truth.jsonl.gz")
    manifest = main.read_json(root / "probe_manifest.json")
    score = main.read_json(main.OUT_ROOT / "analysis" / f"{split}_score.json")
    main.validate_digest(manifest, "manifest_digest")
    main.validate_digest(score, "score_digest")

    expected_public: list[dict[str, Any]] = []
    expected_probes: list[dict[str, Any]] = []
    expected_truth: list[dict[str, Any]] = []
    for spec in independent_specs(split):
        expected_public.append(independent_public(spec))
        expected_probes.extend(independent_probe_rows(spec))
        expected_truth.extend(independent_truth(spec))
    add(checks, f"{split}_pair_count", len(public) == len(expected_public) == 512)
    add(checks, f"{split}_probe_count", len(probes) == len(expected_probes) == 163840)
    add(checks, f"{split}_system_count", len(truth) == len(expected_truth) == 16384)
    add(checks, f"{split}_public_regeneration", main.digest(public) == main.digest(expected_public))
    add(checks, f"{split}_probe_regeneration", main.digest(probes) == main.digest(expected_probes))
    add(checks, f"{split}_truth_regeneration", main.digest(truth) == main.digest(expected_truth))
    add(checks, f"{split}_manifest_public", manifest["public_digest"] == main.digest(public))
    add(checks, f"{split}_manifest_probe", manifest["probe_digest"] == main.digest(probes))
    add(checks, f"{split}_truth_absent_at_seal", manifest["truth_absent_at_probe_seal"] is True)
    add(checks, f"{split}_probe_precedes_truth", (root / "probe_manifest.json").stat().st_mtime_ns <= (root / "sealed_target_truth.jsonl.gz").stat().st_mtime_ns)
    public_fields = set().union(*(row.keys() for row in public))
    probe_fields = set().union(*(row.keys() for row in probes))
    add(checks, f"{split}_public_has_no_world_truth", not ({"active_code", "null_code", "exact_target", "functional_target", "family", "gauge"} & public_fields), sorted(public_fields))
    add(checks, f"{split}_probe_has_no_truth_labels", probe_fields == {"pair_id", "split", "world_id", "probe", "response"}, sorted(probe_fields))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in truth:
        groups[row["pair_id"]].append(row)
    class_contract = True
    for rows in groups.values():
        exact = [np.asarray(row["exact_target"], dtype=np.float64) for row in rows]
        functional = [np.asarray(row["functional_target"], dtype=np.float64) for row in rows]
        class_contract = class_contract and len(rows) == 32
        class_contract = class_contract and independent_class_count(exact) == 32
        class_contract = class_contract and independent_class_count(functional) == 16
        class_contract = class_contract and len({row["visible_state_digest"] for row in rows}) == 1
    add(checks, f"{split}_target_quotient_contract", len(groups) == 512 and class_contract)
    add(checks, f"{split}_families", {row["family"] for row in truth} == set(EXPECTED_FAMILIES[split]))
    add(checks, f"{split}_gauges", {row["gauge"] for row in truth} == set(EXPECTED_GAUGES))

    recomputed = independent_aggregate(public, probes, truth)
    add(checks, f"{split}_score_metrics", numeric_max_error(score["metrics"], recomputed["metrics"]) <= 1.0e-12)
    add(checks, f"{split}_score_checks", score["checks"] == recomputed["checks"] and score["gate"] == all(recomputed["checks"].values()))
    add(checks, f"{split}_probe_summary", numeric_max_error(score["probe_summary"], recomputed["probe_summary"]) <= 1.0e-12)
    add(checks, f"{split}_gauge_summary", numeric_max_error(score["gauge_summary"], recomputed["gauge_summary"]) <= 1.0e-12)
    add(checks, f"{split}_pair_metric_digest", score["pair_metric_digest"] == recomputed["pair_metric_digest"])
    add(checks, f"{split}_all_checks", all(recomputed["checks"].values()), recomputed["metrics"])
    add(checks, f"{split}_functional_not_exact", recomputed["probe_summary"]["functional_edge_all"]["functional_identifiable_fraction"] == 1.0 and recomputed["probe_summary"]["functional_edge_all"]["exact_identifiable_fraction"] == 0.0)
    add(checks, f"{split}_exact_requires_full", recomputed["probe_summary"]["full_edge_jacobian"]["exact_identifiable_fraction"] == 1.0 and recomputed["metrics"]["selected_exact_cost"] == 16)
    return score


def final_audit() -> dict[str, Any]:
    if FINAL_AUDIT_PATH.exists():
        raise RuntimeError("Phase1212 final audit already exists")
    checks: list[dict[str, Any]] = []
    protocol = main.verify_protocol()
    preaudit_value = main.read_json(main.PREAUDIT_PATH)
    main.validate_digest(preaudit_value, "audit_digest")
    add(checks, "preaudit_passed", preaudit_value["all_checks_passed"] and preaudit_value["protocol_digest"] == protocol["protocol_digest"])
    discovery = audit_split(checks, "discovery")
    confirmation = audit_split(checks, "confirmation")
    add(checks, "family_holdout", set(EXPECTED_FAMILIES["discovery"]).isdisjoint(EXPECTED_FAMILIES["confirmation"]))
    add(checks, "split_seed_independence", independent_seed("discovery", "additive", 0, 0) != independent_seed("confirmation", "carrier_pair", 0, 0))
    selection = main.read_json(main.SELECTION_PATH)
    main.validate_digest(selection, "selection_digest")
    add(checks, "confirmation_authorized", selection["confirmation_authorized"] is True and discovery["gate"])
    add(checks, "selection_frozen", selection["functional_probe"] == "functional_edge_all" and selection["exact_probe"] == "full_edge_jacobian" and selection["functional_cost"] == 4 and selection["exact_cost"] == 16)
    add(checks, "confirmation_selection", confirmation["metrics"]["selected_functional_probe"] == selection["functional_probe"] and confirmation["metrics"]["selected_exact_probe"] == selection["exact_probe"])
    final = main.read_json(main.FINAL_PATH)
    main.validate_digest(final, "final_digest")
    add(checks, "final_protocol", final["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "quotient_confirmed", final["known_truth_quotient_calibration_confirmed"] is True and discovery["gate"] and confirmation["gate"])
    add(checks, "status", final["status"] == "functional_target_quotient_and_typed_probe_boundary_confirmed")
    add(checks, "transfer_denied", final["free_transformer_transfer_authorized"] is False and final["language_model_transfer_authorized"] is False)
    add(checks, "new_math_open", final["new_math_hypothesis"]["status"] == "OPEN_NOT_CONFIRMED" and final["new_math_hypothesis"]["upgrade_gates_passed"] == 0)
    add(checks, "k192", final["new_k_item"]["id"] == "K192" and final["new_k_item"]["level"] == "E3-KT")
    add(checks, "claim_scope", "does not prove a globally minimal probe" in final["claim_boundary"] and "natural-language semantic quotient" in final["claim_boundary"])
    add(checks, "auto_stop", final["auto_continue"] is False)
    return finish(
        checks,
        "independent exact-regeneration quotient and typed-probe result audit",
        FINAL_AUDIT_PATH,
        {
            "protocol_digest": protocol["protocol_digest"],
            "final_digest": final["final_digest"],
            "status": final["status"],
        },
    )


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "final"))
    args = parser.parse_args()
    value = preaudit() if args.stage == "preaudit" else final_audit()
    print({
        "stage": value["stage"],
        "passed": value["passed_count"],
        "total": value["check_count"],
        "all_checks_passed": value["all_checks_passed"],
        "audit_digest": value["audit_digest"],
    })


if __name__ == "__main__":
    cli()
