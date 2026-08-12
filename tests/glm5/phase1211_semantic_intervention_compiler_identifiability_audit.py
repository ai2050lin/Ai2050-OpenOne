#!/usr/bin/env python3
"""Independent zero-output and exact-regeneration audit for Phase1211."""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import phase1211_semantic_intervention_compiler_identifiability as main


FINAL_AUDIT_PATH = main.OUT_ROOT / "audit/independent_result_audit.json"
EXPECTED_FAMILIES = ("additive", "semantic_pair", "carrier_pair", "full_pairwise")
EXPECTED_GAUGES = ("signed_permutation", "orthogonal_dense")
EXPECTED_WORLDS = ("degree2_completion", "alternative_completion")
EXPECTED_CANDIDATES = (
    "full_replacement",
    "matched_carrier_did",
    "carrier_projection",
    "degree2_completion",
)
EXPECTED_VERTICES = tuple((a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))


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
    key = f"phase1211|latent|{split}|{family}|{target_index}|{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def independent_gauge_seed(split: str, family: str, gauge: str, target_index: int, replicate: int) -> int:
    key = f"phase1211|gauge|{split}|{family}|{gauge}|{target_index}|{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def independent_design(vertex: tuple[int, int, int]) -> np.ndarray:
    a, b, c = vertex
    return np.asarray((1.0, a, b, c, a * b, a * c, b * c), dtype=np.float64)


def independent_gauge(rng: np.random.Generator, gauge: str) -> np.ndarray:
    if gauge == "signed_permutation":
        permutation = rng.permutation(16)
        signs = rng.choice((-1.0, 1.0), size=16)
        matrix = np.zeros((16, 16), dtype=np.float64)
        matrix[np.arange(16), permutation] = signs
        return matrix
    raw = rng.normal(size=(16, 16))
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
    ta, tb, tc = target
    recipient = (-ta, tb, tc)
    donor = (ta, tb, -tc)
    donor_pair = (-ta, tb, -tc)
    amplitudes = latent_rng.uniform(0.75, 1.75, size=7) * latent_rng.choice((-1.0, 1.0), size=7)
    coefficients = np.zeros((7, 16), dtype=np.float64)
    for index in independent_active_basis(family):
        coefficients[index, index] = amplitudes[index]
    base_latent = np.stack([independent_design(vertex) @ coefficients for vertex in EXPECTED_VERTICES])
    recipient_index = EXPECTED_VERTICES.index(recipient)
    target_change = float(np.linalg.norm(base_latent[target_index] - base_latent[recipient_index]))
    tau = np.zeros(16, dtype=np.float64)
    tau[7] = max(target_change, 1.0) * float(latent_rng.choice((-1.0, 1.0)))
    gauge_value = independent_gauge(gauge_rng, gauge)
    return {
        "pair_id": f"p1211:{split}:{family}:{gauge}:t{target_index}:r{replicate:03d}",
        "split": split,
        "family": family,
        "gauge": gauge,
        "target_index": target_index,
        "target": target,
        "recipient": recipient,
        "donor": donor,
        "donor_pair": donor_pair,
        "base_hidden": base_latent @ gauge_value.T,
        "tau_hidden": tau @ gauge_value.T,
    }


def independent_specs(split: str) -> Iterable[dict[str, Any]]:
    for family in EXPECTED_FAMILIES:
        for gauge in EXPECTED_GAUGES:
            for target_index in range(8):
                for replicate in range(16):
                    yield independent_spec(split, family, gauge, target_index, replicate)


def independent_public(spec: dict[str, Any]) -> dict[str, Any]:
    observed = [index for index in range(8) if index != spec["target_index"]]
    return {
        "pair_id": spec["pair_id"],
        "split": spec["split"],
        "target": list(spec["target"]),
        "recipient": list(spec["recipient"]),
        "donor": list(spec["donor"]),
        "donor_pair": list(spec["donor_pair"]),
        "observed_vertices": [list(EXPECTED_VERTICES[index]) for index in observed],
        "observed_states": [spec["base_hidden"][index].tolist() for index in observed],
        "candidate_contract": list(EXPECTED_CANDIDATES),
    }


def independent_lookup(public: dict[str, Any]) -> dict[tuple[int, int, int], np.ndarray]:
    return {
        tuple(int(value) for value in vertex): np.asarray(state, dtype=np.float64)
        for vertex, state in zip(public["observed_vertices"], public["observed_states"])
    }


def independent_carrier_projection(lookup: dict[tuple[int, int, int], np.ndarray]) -> np.ndarray:
    differences = []
    for a in (-1, 1):
        for b in (-1, 1):
            left, right = (a, b, -1), (a, b, 1)
            if left in lookup and right in lookup:
                differences.append(lookup[right] - lookup[left])
    matrix = np.stack(differences)
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * max(float(singular[0]), 1.0e-12)
    basis = vh[: int(np.sum(singular > tolerance))]
    return basis.T @ basis


def independent_compile(public: dict[str, Any]) -> dict[str, Any]:
    lookup = independent_lookup(public)
    recipient = lookup[tuple(public["recipient"])]
    donor = lookup[tuple(public["donor"])]
    donor_pair = lookup[tuple(public["donor_pair"])]
    full = donor.copy()
    relative = recipient + (donor - recipient)
    did = recipient + (donor - donor_pair)
    projection = independent_carrier_projection(lookup)
    projected = recipient + (np.eye(16) - projection) @ (donor - recipient)
    vertices = [tuple(vertex) for vertex in public["observed_vertices"]]
    x = np.stack([independent_design(vertex) for vertex in vertices])
    h = np.stack([lookup[vertex] for vertex in vertices])
    degree2 = independent_design(tuple(public["target"])) @ np.linalg.solve(x, h)
    candidates = {
        "full_replacement": full,
        "matched_carrier_did": did,
        "carrier_projection": projected,
        "degree2_completion": degree2,
    }
    return {
        "pair_id": public["pair_id"],
        "split": public["split"],
        "candidate_states": {name: value.tolist() for name, value in candidates.items()},
        "full_relative_identity_error": float(np.max(np.abs(full - relative))),
        "visible_state_digest": main.digest(public["observed_states"]),
    }


def independent_truth(spec: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    base_target = np.asarray(spec["base_hidden"][spec["target_index"]], dtype=np.float64)
    alternative = base_target + np.asarray(spec["tau_hidden"], dtype=np.float64)
    recipient = np.asarray(spec["base_hidden"][EXPECTED_VERTICES.index(spec["recipient"])], dtype=np.float64)
    state_sets = {
        "degree2_completion": np.asarray(spec["base_hidden"], dtype=np.float64),
        "alternative_completion": np.asarray(spec["base_hidden"], dtype=np.float64).copy(),
    }
    state_sets["alternative_completion"][spec["target_index"]] = alternative
    ideals = {"degree2_completion": base_target, "alternative_completion": alternative}
    rows = []
    for world in EXPECTED_WORLDS:
        ideal = ideals[world]
        denominator = max(float(np.linalg.norm(ideal - recipient)), 1.0e-12)
        metrics = {}
        for name, raw in prediction["candidate_states"].items():
            candidate = np.asarray(raw, dtype=np.float64)
            distances = np.linalg.norm(state_sets[world] - candidate[None, :], axis=1)
            nearest_index = int(np.argmin(distances))
            nearest = EXPECTED_VERTICES[nearest_index]
            error = float(np.linalg.norm(candidate - ideal))
            metrics[name] = {
                "absolute_error": error,
                "normalized_error": error / denominator,
                "exact_target_state": error <= 1.0e-10,
                "nearest_target_vertex": nearest_index == spec["target_index"],
                "target_semantic_correct": nearest[0] == spec["target"][0],
                "other_semantic_preserved": nearest[1] == spec["target"][1],
                "carrier_preserved": nearest[2] == spec["target"][2],
            }
        rows.append({
            "system_id": f"{spec['pair_id']}:{world}",
            "pair_id": spec["pair_id"],
            "split": spec["split"],
            "family": spec["family"],
            "gauge": spec["gauge"],
            "world": world,
            "ideal_target_state": ideal.tolist(),
            "twin_target_separation": float(np.linalg.norm(alternative - base_target)),
            "visible_state_digest": prediction["visible_state_digest"],
            "candidate_metrics": metrics,
            "oracle_error": 0.0,
        })
    return rows


def independent_aggregate(truth: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    twins: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in truth:
        for candidate, metrics in row["candidate_metrics"].items():
            candidate_rows[candidate].append({**metrics, "gauge": row["gauge"], "world": row["world"]})
        twins[row["pair_id"]][row["world"]] = row
    summaries = {}
    for candidate in EXPECTED_CANDIDATES:
        worlds = {}
        for world in EXPECTED_WORLDS:
            members = [row for row in candidate_rows[candidate] if row["world"] == world]
            worlds[world] = {
                "count": len(members),
                "mean_normalized_error": float(np.mean([row["normalized_error"] for row in members])),
                "max_normalized_error": float(max(row["normalized_error"] for row in members)),
                "min_normalized_error": float(min(row["normalized_error"] for row in members)),
                "exact_fraction": float(np.mean([row["exact_target_state"] for row in members])),
                "target_semantic_fraction": float(np.mean([row["target_semantic_correct"] for row in members])),
                "carrier_preserved_fraction": float(np.mean([row["carrier_preserved"] for row in members])),
            }
        universal = bool(
            min(worlds[world]["exact_fraction"] for world in EXPECTED_WORLDS) >= 0.99
            and max(worlds[world]["mean_normalized_error"] for world in EXPECTED_WORLDS) <= 1.0e-8
        )
        summaries[candidate] = {"worlds": worlds, "universally_qualified": universal}
    lower_bounds = []
    visible_errors = []
    for worlds in twins.values():
        base = worlds["degree2_completion"]
        alternative = worlds["alternative_completion"]
        visible_errors.append(0.0 if base["visible_state_digest"] == alternative["visible_state_digest"] else float("inf"))
        for candidate in EXPECTED_CANDIDATES:
            e0 = base["candidate_metrics"][candidate]["absolute_error"]
            e1 = alternative["candidate_metrics"][candidate]["absolute_error"]
            lower_bounds.append(max(e0, e1) + 1.0e-10 >= 0.5 * base["twin_target_separation"])
    gauge_errors = {
        candidate: {
            gauge: float(np.mean([
                row["candidate_metrics"][candidate]["normalized_error"]
                for row in truth if row["gauge"] == gauge
            ]))
            for gauge in EXPECTED_GAUGES
        }
        for candidate in EXPECTED_CANDIDATES
    }
    degree2_base = summaries["degree2_completion"]["worlds"]["degree2_completion"]
    degree2_alt = summaries["degree2_completion"]["worlds"]["alternative_completion"]
    metrics = {
        "pair_count": len(predictions),
        "system_count": len(truth),
        "finite_fraction": float(np.mean([
            math.isfinite(metric["normalized_error"])
            for row in truth for metric in row["candidate_metrics"].values()
        ])),
        "visible_twin_identity_max": float(max(visible_errors, default=float("inf"))),
        "full_relative_identity_max": float(max(row["full_relative_identity_error"] for row in predictions)),
        "degree2_base_exact_fraction": degree2_base["exact_fraction"],
        "degree2_base_max_error": degree2_base["max_normalized_error"],
        "alternative_degree2_min_error": degree2_alt["min_normalized_error"],
        "twin_lower_bound_fraction": float(np.mean(lower_bounds)),
        "universal_candidate_count": sum(value["universally_qualified"] for value in summaries.values()),
        "oracle_max_error": float(max(row["oracle_error"] for row in truth)),
        "gauge_mean_error_gap": max(
            abs(values[EXPECTED_GAUGES[0]] - values[EXPECTED_GAUGES[1]])
            for values in gauge_errors.values()
        ),
    }
    thresholds = main.THRESHOLDS
    checks = {
        "finite": metrics["finite_fraction"] >= thresholds["finite_fraction_min"],
        "visible_twins": metrics["visible_twin_identity_max"] <= thresholds["visible_twin_identity_max"],
        "full_relative_identity": metrics["full_relative_identity_max"] <= thresholds["full_relative_identity_max"],
        "degree2_base_exact": metrics["degree2_base_exact_fraction"] >= thresholds["degree2_base_exact_fraction_min"] and metrics["degree2_base_max_error"] <= thresholds["degree2_base_max_error_max"],
        "alternative_separates": metrics["alternative_degree2_min_error"] >= thresholds["alternative_degree2_min_error_min"],
        "twin_lower_bound": metrics["twin_lower_bound_fraction"] >= thresholds["twin_lower_bound_fraction_min"],
        "no_universal_candidate": metrics["universal_candidate_count"] <= thresholds["universal_candidate_count_max"],
        "oracle": metrics["oracle_max_error"] <= thresholds["oracle_max_error_max"],
        "gauge": metrics["gauge_mean_error_gap"] <= thresholds["gauge_mean_error_gap_max"],
    }
    return {"metrics": metrics, "checks": checks, "candidate_summary": summaries, "gauge_errors": gauge_errors}


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
        raise RuntimeError("Phase1211 preaudit already exists")
    checks: list[dict[str, Any]] = []
    protocol = main.read_json(main.PROTOCOL_PATH)
    clean = dict(protocol)
    stored = clean.pop("protocol_digest")
    add(checks, "protocol_digest", main.digest(clean) == stored)
    add(checks, "source_hashes", protocol["source_hashes"] == main.source_hashes())
    add(checks, "main_hash", protocol["source_hashes"]["main"] == main.sha256_file(main.SCRIPT))
    add(checks, "audit_hash", protocol["source_hashes"]["audit"] == main.sha256_file(Path(__file__).resolve()))
    add(checks, "phase1210_final", protocol["source_phase1210_final_digest"] == main.EXPECTED_1210_FINAL)
    add(checks, "phase1210_audit", protocol["source_phase1210_audit_digest"] == main.EXPECTED_1210_AUDIT)
    add(checks, "protocol_checks", all(protocol["checks"].values()), protocol["checks"])
    add(checks, "dimension", protocol["dimension"] == 16)
    add(checks, "replicates", protocol["replicates"] == 16)
    add(checks, "families", tuple(protocol["families"]) == EXPECTED_FAMILIES)
    add(checks, "gauges", tuple(protocol["gauges"]) == EXPECTED_GAUGES)
    add(checks, "worlds", tuple(protocol["worlds"]) == EXPECTED_WORLDS)
    add(checks, "candidates", tuple(protocol["candidates"]) == EXPECTED_CANDIDATES)
    add(checks, "candidate_distinctness", len(set(protocol["candidates"])) == 4 and "relative_displacement" not in protocol["candidates"])
    add(checks, "eight_vertices", tuple(tuple(row) for row in protocol["vertices"]) == EXPECTED_VERTICES)
    ranks = [
        int(np.linalg.matrix_rank(np.stack([independent_design(vertex) for index, vertex in enumerate(EXPECTED_VERTICES) if index != missing])))
        for missing in range(8)
    ]
    add(checks, "leave_one_out_design_rank", ranks == [7] * 8, ranks)
    add(checks, "pair_count", protocol["pairs_per_split"] == 1024)
    add(checks, "system_count", protocol["systems_per_split"] == 2048)
    add(checks, "thresholds", protocol["thresholds"] == main.THRESHOLDS)
    add(checks, "relative_identity_declared", "equals full replacement" in protocol["candidate_definitions"]["relative_displacement_identity"])
    probe_r = np.arange(16, dtype=np.float64)
    probe_d = np.arange(16, dtype=np.float64)[::-1]
    add(checks, "relative_identity_exact", float(np.max(np.abs(probe_d - (probe_r + probe_d - probe_r)))) == 0.0)
    add(checks, "new_math_open", protocol["new_math_upgrade_gate"]["status"] == "OPEN_NOT_CONFIRMED" and not any(value for key, value in protocol["new_math_upgrade_gate"].items() if key != "status"))
    imports = imported_modules(main.SCRIPT)
    add(checks, "no_model_runtime", "torch" not in imports and "transformers" not in imports, sorted(imports))
    add(checks, "known_truth_hard_stop", any("No Qwen3" in row for row in protocol["hard_stops"]))
    forbidden = [main.OUT_ROOT / "runs", main.OUT_ROOT / "analysis", main.FINAL_PATH, FINAL_AUDIT_PATH]
    add(checks, "zero_formal_outputs", not any(path.exists() for path in forbidden))
    return finish(
        checks,
        "independent zero-output protocol and algebra audit",
        main.PREAUDIT_PATH,
        {"protocol_digest": protocol["protocol_digest"]},
    )


def audit_split(checks: list[dict[str, Any]], split: str) -> dict[str, Any]:
    root = main.split_root(split)
    public = main.read_jsonl_gz(root / "public_observed_cubes.jsonl.gz")
    predictions = main.read_jsonl_gz(root / "sealed_candidate_predictions.jsonl.gz")
    truth = main.read_jsonl_gz(root / "sealed_completion_truth.jsonl.gz")
    manifest = main.read_json(root / "prediction_manifest.json")
    score = main.read_json(main.OUT_ROOT / "analysis" / f"{split}_score.json")
    main.validate_digest(manifest, "manifest_digest")
    main.validate_digest(score, "score_digest")
    expected_public, expected_predictions, expected_truth = [], [], []
    for spec in independent_specs(split):
        public_row = independent_public(spec)
        prediction = independent_compile(public_row)
        expected_public.append(public_row)
        expected_predictions.append(prediction)
        expected_truth.extend(independent_truth(spec, prediction))
    add(checks, f"{split}_pair_count", len(public) == len(predictions) == 1024)
    add(checks, f"{split}_system_count", len(truth) == 2048)
    add(checks, f"{split}_public_regeneration", main.digest(public) == main.digest(expected_public))
    add(checks, f"{split}_prediction_regeneration", main.digest(predictions) == main.digest(expected_predictions))
    add(checks, f"{split}_truth_regeneration", main.digest(truth) == main.digest(expected_truth))
    add(checks, f"{split}_manifest_public", manifest["public_digest"] == main.digest(public))
    add(checks, f"{split}_manifest_predictions", manifest["prediction_digest"] == main.digest(predictions))
    add(checks, f"{split}_truth_absent_at_prediction", manifest["truth_absent_at_prediction"] is True)
    add(checks, f"{split}_prediction_precedes_truth", (root / "prediction_manifest.json").stat().st_mtime_ns <= (root / "sealed_completion_truth.jsonl.gz").stat().st_mtime_ns)
    target_absent = all(tuple(row["target"]) not in {tuple(vertex) for vertex in row["observed_vertices"]} and len(row["observed_vertices"]) == 7 for row in public)
    add(checks, f"{split}_target_withheld", target_absent)
    prediction_fields = set().union(*(row.keys() for row in predictions))
    add(checks, f"{split}_prediction_has_no_truth", not ({"world", "ideal_target_state", "tau_hidden", "family", "gauge"} & prediction_fields), sorted(prediction_fields))
    twin_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in truth:
        twin_groups[row["pair_id"]].append(row)
    twins_valid = all(
        len(rows) == 2
        and {row["world"] for row in rows} == set(EXPECTED_WORLDS)
        and len({row["visible_state_digest"] for row in rows}) == 1
        and rows[0]["twin_target_separation"] > 0.0
        for rows in twin_groups.values()
    )
    add(checks, f"{split}_indistinguishable_twins", len(twin_groups) == 1024 and twins_valid)
    recomputed = independent_aggregate(truth, predictions)
    add(checks, f"{split}_score_metrics", numeric_max_error(score["metrics"], recomputed["metrics"]) <= 1.0e-12)
    add(checks, f"{split}_score_checks", score["checks"] == recomputed["checks"] and score["gate"] == all(recomputed["checks"].values()))
    add(checks, f"{split}_candidate_summary", numeric_max_error(score["candidate_summary"], recomputed["candidate_summary"]) <= 1.0e-12)
    add(checks, f"{split}_all_boundary_checks", all(recomputed["checks"].values()), recomputed["metrics"])
    add(checks, f"{split}_no_universal_candidate", recomputed["metrics"]["universal_candidate_count"] == 0)
    add(checks, f"{split}_gauge_invariance", recomputed["metrics"]["gauge_mean_error_gap"] <= 1.0e-10)
    return score


def final_audit() -> dict[str, Any]:
    if FINAL_AUDIT_PATH.exists():
        raise RuntimeError("Phase1211 final audit already exists")
    checks: list[dict[str, Any]] = []
    protocol = main.verify_protocol()
    preaudit_value = main.read_json(main.PREAUDIT_PATH)
    main.validate_digest(preaudit_value, "audit_digest")
    add(checks, "preaudit_passed", preaudit_value["all_checks_passed"] and preaudit_value["protocol_digest"] == protocol["protocol_digest"])
    discovery = audit_split(checks, "discovery")
    confirmation = audit_split(checks, "confirmation")
    add(checks, "confirmation_authorized", discovery["boundary_confirmation_authorized"] is True)
    add(checks, "disjoint_split_seeds", independent_seed("discovery", "additive", 0, 0) != independent_seed("confirmation", "additive", 0, 0))
    final = main.read_json(main.FINAL_PATH)
    main.validate_digest(final, "final_digest")
    add(checks, "final_protocol", final["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "boundary_confirmed", final["known_truth_identifiability_boundary_confirmed"] is True and discovery["gate"] and confirmation["gate"])
    add(checks, "status", final["status"] == "semantic_intervention_not_identifiable_from_seven_visible_vertices")
    add(checks, "compiler_transfer_denied", final["compiler_transfer_authorized"] is False)
    add(checks, "new_math_not_upgraded", final["new_math_hypothesis"]["status"] == "OPEN_NOT_CONFIRMED" and final["new_math_hypothesis"]["upgrade_gates_passed"] == 0)
    add(checks, "k191", final["new_k_item"]["id"] == "K191" and final["new_k_item"]["level"] == "E3-KT")
    add(checks, "claim_boundary", "not evidence that Qwen3 lacks semantic mechanisms" in final["claim_boundary"])
    add(checks, "auto_stop", final["auto_continue"] is False)
    return finish(
        checks,
        "independent exact-regeneration result audit",
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
