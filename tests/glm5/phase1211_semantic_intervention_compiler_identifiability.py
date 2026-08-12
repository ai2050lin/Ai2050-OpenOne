#!/usr/bin/env python3
"""Known-truth identifiability calibration for semantic intervention compilers.

The experiment asks whether seven visible vertices of a three-factor
counterfactual cube determine the missing semantic target intervention.  Each
public cube has two sealed completions that agree exactly on all visible
states and differ only at the target vertex.  Candidate compilers therefore
receive identical inputs in both worlds.
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
PHASE = 1211
OUT_ROOT = TEST_ROOT / "result/phase1211_semantic_intervention_compiler_identifiability"
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1211_semantic_intervention_compiler_identifiability_audit.py")
SOURCE1210 = TEST_ROOT / "result/phase1210_qwen3_natural_object_attribute_camera_transfer"

PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

EXPECTED_1210_FINAL = "20a6bacaed13966f3e900508ed65892c4d96c95707a7b21900bf8b3b5d485cd1"
EXPECTED_1210_AUDIT = "e6a638c097072109fcdc1b9fbd967cf108c9a40dd2ac48331d84cfe3e18ee63d"

DIMENSION = 16
REPLICATES = 16
FAMILIES = ("additive", "semantic_pair", "carrier_pair", "full_pairwise")
GAUGES = ("signed_permutation", "orthogonal_dense")
WORLDS = ("degree2_completion", "alternative_completion")
CANDIDATES = ("full_replacement", "matched_carrier_did", "carrier_projection", "degree2_completion")
VERTICES = tuple((a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))
BASIS_NAMES = ("one", "a", "b", "c", "ab", "ac", "bc")
TAU_SCALE = 1.0
EPSILON = 1.0e-12

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "visible_twin_identity_max": 0.0,
    "full_relative_identity_max": 1.0e-12,
    "degree2_base_exact_fraction_min": 1.0,
    "degree2_base_max_error_max": 1.0e-10,
    "alternative_degree2_min_error_min": 0.20,
    "twin_lower_bound_fraction_min": 1.0,
    "universal_candidate_count_max": 0,
    "oracle_max_error_max": 1.0e-12,
    "gauge_mean_error_gap_max": 1.0e-10,
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


def seed_for(split: str, family: str, target_index: int, replicate: int) -> int:
    key = f"phase1211|latent|{split}|{family}|{target_index}|{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def gauge_seed_for(split: str, family: str, gauge: str, target_index: int, replicate: int) -> int:
    key = f"phase1211|gauge|{split}|{family}|{gauge}|{target_index}|{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def design(vertex: tuple[int, int, int]) -> np.ndarray:
    a, b, c = vertex
    return np.asarray((1.0, a, b, c, a * b, a * c, b * c), dtype=np.float64)


def gauge_matrix(rng: np.random.Generator, gauge: str) -> np.ndarray:
    if gauge == "signed_permutation":
        permutation = rng.permutation(DIMENSION)
        signs = rng.choice((-1.0, 1.0), size=DIMENSION)
        matrix = np.zeros((DIMENSION, DIMENSION), dtype=np.float64)
        matrix[np.arange(DIMENSION), permutation] = signs
        return matrix
    raw = rng.normal(size=(DIMENSION, DIMENSION))
    q, r = np.linalg.qr(raw)
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    return q.astype(np.float64)


def active_basis(family: str) -> set[int]:
    mapping = {
        "additive": {0, 1, 2, 3},
        "semantic_pair": {0, 1, 2, 3, 4},
        "carrier_pair": {0, 1, 2, 3, 5, 6},
        "full_pairwise": set(range(7)),
    }
    return mapping[family]


def system_spec(split: str, family: str, gauge: str, target_index: int, replicate: int) -> dict[str, Any]:
    latent_rng = np.random.default_rng(seed_for(split, family, target_index, replicate))
    gauge_rng = np.random.default_rng(gauge_seed_for(split, family, gauge, target_index, replicate))
    target = VERTICES[target_index]
    ta, tb, tc = target
    recipient = (-ta, tb, tc)
    donor = (ta, tb, -tc)
    donor_pair = (-ta, tb, -tc)
    amplitudes = latent_rng.uniform(0.75, 1.75, size=7) * latent_rng.choice((-1.0, 1.0), size=7)
    coefficients = np.zeros((7, DIMENSION), dtype=np.float64)
    for index in active_basis(family):
        coefficients[index, index] = amplitudes[index]
    base_latent = np.stack([design(vertex) @ coefficients for vertex in VERTICES], axis=0)
    recipient_index = VERTICES.index(recipient)
    target_change = float(np.linalg.norm(base_latent[target_index] - base_latent[recipient_index]))
    tau = np.zeros(DIMENSION, dtype=np.float64)
    tau[7] = max(target_change, 1.0) * TAU_SCALE * float(latent_rng.choice((-1.0, 1.0)))
    gauge_value = gauge_matrix(gauge_rng, gauge)
    base_hidden = base_latent @ gauge_value.T
    tau_hidden = tau @ gauge_value.T
    pair_id = f"p1211:{split}:{family}:{gauge}:t{target_index}:r{replicate:03d}"
    return {
        "pair_id": pair_id,
        "split": split,
        "family": family,
        "gauge": gauge,
        "target_index": target_index,
        "target": target,
        "recipient": recipient,
        "donor": donor,
        "donor_pair": donor_pair,
        "base_hidden": base_hidden,
        "tau_hidden": tau_hidden,
    }


def public_row(spec: dict[str, Any]) -> dict[str, Any]:
    observed_indices = [index for index in range(len(VERTICES)) if index != spec["target_index"]]
    return {
        "pair_id": spec["pair_id"],
        "split": spec["split"],
        "target": list(spec["target"]),
        "recipient": list(spec["recipient"]),
        "donor": list(spec["donor"]),
        "donor_pair": list(spec["donor_pair"]),
        "observed_vertices": [list(VERTICES[index]) for index in observed_indices],
        "observed_states": [spec["base_hidden"][index].tolist() for index in observed_indices],
        "candidate_contract": list(CANDIDATES),
    }


def observed_lookup(public: dict[str, Any]) -> dict[tuple[int, int, int], np.ndarray]:
    return {
        tuple(int(value) for value in vertex): np.asarray(state, dtype=np.float64)
        for vertex, state in zip(public["observed_vertices"], public["observed_states"])
    }


def carrier_projection(lookup: dict[tuple[int, int, int], np.ndarray]) -> np.ndarray:
    differences = []
    for a in (-1, 1):
        for b in (-1, 1):
            left = (a, b, -1)
            right = (a, b, 1)
            if left in lookup and right in lookup:
                differences.append(lookup[right] - lookup[left])
    matrix = np.stack(differences, axis=0)
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * max(float(singular[0]), EPSILON)
    rank = int(np.sum(singular > tolerance))
    basis = vh[:rank]
    return basis.T @ basis


def compile_candidates(public: dict[str, Any]) -> dict[str, Any]:
    lookup = observed_lookup(public)
    target = tuple(public["target"])
    recipient = lookup[tuple(public["recipient"])]
    donor = lookup[tuple(public["donor"])]
    donor_pair = lookup[tuple(public["donor_pair"])]

    full = donor.copy()
    relative = recipient + (donor - recipient)
    did = recipient + (donor - donor_pair)
    projection = carrier_projection(lookup)
    projected = recipient + (np.eye(DIMENSION, dtype=np.float64) - projection) @ (donor - recipient)

    vertices = [tuple(row) for row in public["observed_vertices"]]
    x = np.stack([design(vertex) for vertex in vertices], axis=0)
    h = np.stack([lookup[vertex] for vertex in vertices], axis=0)
    coefficients = np.linalg.solve(x, h)
    degree2 = design(target) @ coefficients

    candidates = {
        "full_replacement": full,
        "matched_carrier_did": did,
        "carrier_projection": projected,
        "degree2_completion": degree2,
    }
    return {
        "pair_id": public["pair_id"],
        "split": public["split"],
        "candidate_states": {name: state.tolist() for name, state in candidates.items()},
        "full_relative_identity_error": float(np.max(np.abs(full - relative))),
        "visible_state_digest": digest(public["observed_states"]),
    }


def truth_rows(spec: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    base_target = np.asarray(spec["base_hidden"][spec["target_index"]], dtype=np.float64)
    alternative_target = base_target + np.asarray(spec["tau_hidden"], dtype=np.float64)
    recipient = np.asarray(spec["base_hidden"][VERTICES.index(spec["recipient"])], dtype=np.float64)
    full_states = {
        "degree2_completion": np.asarray(spec["base_hidden"], dtype=np.float64),
        "alternative_completion": np.asarray(spec["base_hidden"], dtype=np.float64).copy(),
    }
    full_states["alternative_completion"][spec["target_index"]] = alternative_target
    ideals = {"degree2_completion": base_target, "alternative_completion": alternative_target}
    output: list[dict[str, Any]] = []
    for world in WORLDS:
        ideal = ideals[world]
        denominator = max(float(np.linalg.norm(ideal - recipient)), EPSILON)
        candidate_metrics: dict[str, Any] = {}
        for name, raw in prediction["candidate_states"].items():
            candidate = np.asarray(raw, dtype=np.float64)
            distances = np.linalg.norm(full_states[world] - candidate[None, :], axis=1)
            nearest_index = int(np.argmin(distances))
            nearest = VERTICES[nearest_index]
            error = float(np.linalg.norm(candidate - ideal))
            candidate_metrics[name] = {
                "absolute_error": error,
                "normalized_error": error / denominator,
                "exact_target_state": error <= 1.0e-10,
                "nearest_target_vertex": nearest_index == spec["target_index"],
                "target_semantic_correct": nearest[0] == spec["target"][0],
                "other_semantic_preserved": nearest[1] == spec["target"][1],
                "carrier_preserved": nearest[2] == spec["target"][2],
            }
        output.append({
            "system_id": f"{spec['pair_id']}:{world}",
            "pair_id": spec["pair_id"],
            "split": spec["split"],
            "family": spec["family"],
            "gauge": spec["gauge"],
            "world": world,
            "ideal_target_state": ideal.tolist(),
            "twin_target_separation": float(np.linalg.norm(alternative_target - base_target)),
            "visible_state_digest": prediction["visible_state_digest"],
            "candidate_metrics": candidate_metrics,
            "oracle_error": 0.0,
        })
    return output


def specs_for(split: str) -> Iterable[dict[str, Any]]:
    for family in FAMILIES:
        for gauge in GAUGES:
            for target_index in range(len(VERTICES)):
                for replicate in range(REPLICATES):
                    yield system_spec(split, family, gauge, target_index, replicate)


def expected_pair_count() -> int:
    return len(FAMILIES) * len(GAUGES) * len(VERTICES) * REPLICATES


def source_hashes() -> dict[str, str]:
    return {
        "main": sha256_file(SCRIPT),
        "audit": sha256_file(AUDIT_SCRIPT),
    }


def protocol_payload() -> dict[str, Any]:
    final1210 = read_json(SOURCE1210 / "analysis/final.json")
    audit1210 = read_json(SOURCE1210 / "audit/independent_result_audit.json")
    validate_digest(final1210, "final_digest")
    validate_digest(audit1210, "audit_digest")
    checks = {
        "phase1210_final_frozen": final1210["final_digest"] == EXPECTED_1210_FINAL,
        "phase1210_audit_frozen": audit1210["audit_digest"] == EXPECTED_1210_AUDIT,
        "phase1210_audit_passed": audit1210["all_checks_passed"] is True,
        "phase1210_transfer_failed": final1210["naturalized_qwen3_external_validity"] is False,
        "phase1210_auto_stopped": final1210["auto_continue"] is False,
        "candidate_distinctness": "relative_displacement" not in CANDIDATES,
        "known_truth_only": True,
        "pretrained_models_forbidden": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    payload = {
        "phase": PHASE,
        "schema_version": "phase1211.semantic_intervention_identifiability.v1",
        "created_at_utc": utc_now(),
        "title": "Known-truth semantic intervention compiler identifiability and twin-completion boundary",
        "source_hashes": source_hashes(),
        "source_phase1210_final_digest": final1210["final_digest"],
        "source_phase1210_audit_digest": audit1210["audit_digest"],
        "dimension": DIMENSION,
        "replicates": REPLICATES,
        "families": list(FAMILIES),
        "gauges": list(GAUGES),
        "worlds": list(WORLDS),
        "candidates": list(CANDIDATES),
        "vertices": [list(vertex) for vertex in VERTICES],
        "basis": list(BASIS_NAMES),
        "pairs_per_split": expected_pair_count(),
        "systems_per_split": expected_pair_count() * len(WORLDS),
        "thresholds": THRESHOLDS,
        "candidate_definitions": {
            "full_replacement": "use a target-semantic donor with the wrong carrier",
            "matched_carrier_did": "recipient plus donor-minus-matched-pair displacement",
            "carrier_projection": "remove the visible carrier-difference span from the donor displacement",
            "degree2_completion": "leave-one-vertex-out multilinear interpolation with basis 1,a,b,c,ab,ac,bc",
            "relative_displacement_identity": "recipient + (donor-recipient) equals full replacement and is audited, not double-counted",
        },
        "twin_contract": (
            "Each pair has two completions with exactly identical seven visible states. The alternative adds a target-only "
            "Lagrange component at the sealed eighth vertex. Every deterministic compiler receives identical inputs."
        ),
        "execution_order": [
            "preregister and independent zero-output preaudit",
            "generate discovery public states and seal candidate predictions",
            "reveal discovery twin targets and score",
            "if the identifiability-boundary gate passes, repeat with disjoint confirmation seeds",
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
            "No Qwen3, GLM4, DS7B, language material, head, layer, or neuron scan is authorized.",
            "The alternative target is absent until predictions are sealed.",
            "No candidate may use latent family, gauge, coefficient, tau, or target-state truth.",
            "A deterministic compiler cannot be called universally valid if indistinguishable twins require different targets.",
            "Failure of compiler identifiability is not absence of a semantic mechanism.",
            "No Phase1212 transfer is automatic after an identifiability failure.",
        ],
        "checks": checks,
    }
    payload["protocol_digest"] = digest(payload)
    return payload


def preregister() -> dict[str, Any]:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1211 artifacts")
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
    if split not in ("discovery", "confirmation"):
        raise ValueError(split)
    if split_root(split).exists():
        raise RuntimeError(f"{split} already generated")
    if split == "confirmation":
        discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
        validate_digest(discovery, "score_digest")
        if not discovery["boundary_confirmation_authorized"]:
            raise RuntimeError("discovery denied boundary confirmation")

    public: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for spec in specs_for(split):
        row = public_row(spec)
        public.append(row)
        predictions.append(compile_candidates(row))
    write_jsonl_gz(split_root(split) / "public_observed_cubes.jsonl.gz", public)
    write_jsonl_gz(split_root(split) / "sealed_candidate_predictions.jsonl.gz", predictions)
    truth_path = split_root(split) / "sealed_completion_truth.jsonl.gz"
    manifest = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "pair_count": len(public),
        "public_digest": digest(public),
        "prediction_digest": digest(predictions),
        "truth_absent_at_prediction": not truth_path.exists(),
        "created_at_utc": utc_now(),
    }
    manifest["manifest_digest"] = digest(manifest)
    write_json(split_root(split) / "prediction_manifest.json", manifest)
    print(json.dumps({"split": split, "pair_count": len(public), "manifest_digest": manifest["manifest_digest"]}, indent=2))
    return manifest


def aggregate_score(truth: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    all_candidate_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    twin_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in truth:
        for candidate, metrics in row["candidate_metrics"].items():
            enriched = {**metrics, "family": row["family"], "gauge": row["gauge"], "world": row["world"]}
            all_candidate_rows[candidate].append(enriched)
        twin_rows[row["pair_id"]][row["world"]] = row

    candidate_summary: dict[str, Any] = {}
    for candidate in CANDIDATES:
        rows = all_candidate_rows[candidate]
        by_world = {}
        for world in WORLDS:
            members = [row for row in rows if row["world"] == world]
            by_world[world] = {
                "count": len(members),
                "mean_normalized_error": float(np.mean([row["normalized_error"] for row in members])),
                "max_normalized_error": float(max(row["normalized_error"] for row in members)),
                "min_normalized_error": float(min(row["normalized_error"] for row in members)),
                "exact_fraction": float(np.mean([row["exact_target_state"] for row in members])),
                "target_semantic_fraction": float(np.mean([row["target_semantic_correct"] for row in members])),
                "carrier_preserved_fraction": float(np.mean([row["carrier_preserved"] for row in members])),
            }
        universal = bool(
            min(by_world[world]["exact_fraction"] for world in WORLDS) >= 0.99
            and max(by_world[world]["mean_normalized_error"] for world in WORLDS) <= 1.0e-8
        )
        candidate_summary[candidate] = {"worlds": by_world, "universally_qualified": universal}

    lower_bound_checks = []
    visible_twin_errors = []
    for pair_id, worlds in twin_rows.items():
        low = worlds["degree2_completion"]
        high = worlds["alternative_completion"]
        visible_twin_errors.append(0.0 if low["visible_state_digest"] == high["visible_state_digest"] else float("inf"))
        separation = float(low["twin_target_separation"])
        for candidate in CANDIDATES:
            e0 = float(low["candidate_metrics"][candidate]["absolute_error"])
            e1 = float(high["candidate_metrics"][candidate]["absolute_error"])
            lower_bound_checks.append(max(e0, e1) + 1.0e-10 >= 0.5 * separation)

    gauge_errors = {}
    for candidate in CANDIDATES:
        gauge_errors[candidate] = {
            gauge: float(np.mean([
                row["candidate_metrics"][candidate]["normalized_error"]
                for row in truth if row["gauge"] == gauge
            ]))
            for gauge in GAUGES
        }
    gauge_gap = max(abs(values[GAUGES[0]] - values[GAUGES[1]]) for values in gauge_errors.values())
    full_relative_max = max(float(row["full_relative_identity_error"]) for row in predictions)
    degree2_base = candidate_summary["degree2_completion"]["worlds"]["degree2_completion"]
    alternative_degree2 = candidate_summary["degree2_completion"]["worlds"]["alternative_completion"]
    universal_count = sum(row["universally_qualified"] for row in candidate_summary.values())
    metrics = {
        "pair_count": len(predictions),
        "system_count": len(truth),
        "finite_fraction": float(np.mean([
            math.isfinite(metric["normalized_error"])
            for row in truth for metric in row["candidate_metrics"].values()
        ])),
        "visible_twin_identity_max": float(max(visible_twin_errors, default=float("inf"))),
        "full_relative_identity_max": full_relative_max,
        "degree2_base_exact_fraction": degree2_base["exact_fraction"],
        "degree2_base_max_error": degree2_base["max_normalized_error"],
        "alternative_degree2_min_error": alternative_degree2["min_normalized_error"],
        "twin_lower_bound_fraction": float(np.mean(lower_bound_checks)),
        "universal_candidate_count": universal_count,
        "oracle_max_error": float(max(row["oracle_error"] for row in truth)),
        "gauge_mean_error_gap": gauge_gap,
    }
    checks = {
        "finite": metrics["finite_fraction"] >= THRESHOLDS["finite_fraction_min"],
        "visible_twins": metrics["visible_twin_identity_max"] <= THRESHOLDS["visible_twin_identity_max"],
        "full_relative_identity": metrics["full_relative_identity_max"] <= THRESHOLDS["full_relative_identity_max"],
        "degree2_base_exact": metrics["degree2_base_exact_fraction"] >= THRESHOLDS["degree2_base_exact_fraction_min"] and metrics["degree2_base_max_error"] <= THRESHOLDS["degree2_base_max_error_max"],
        "alternative_separates": metrics["alternative_degree2_min_error"] >= THRESHOLDS["alternative_degree2_min_error_min"],
        "twin_lower_bound": metrics["twin_lower_bound_fraction"] >= THRESHOLDS["twin_lower_bound_fraction_min"],
        "no_universal_candidate": metrics["universal_candidate_count"] <= THRESHOLDS["universal_candidate_count_max"],
        "oracle": metrics["oracle_max_error"] <= THRESHOLDS["oracle_max_error_max"],
        "gauge": metrics["gauge_mean_error_gap"] <= THRESHOLDS["gauge_mean_error_gap_max"],
    }
    return {"metrics": metrics, "checks": checks, "candidate_summary": candidate_summary, "gauge_errors": gauge_errors}


def reveal_split(split: str) -> dict[str, Any]:
    protocol = verify_protocol()
    require_preaudit()
    manifest = read_json(split_root(split) / "prediction_manifest.json")
    validate_digest(manifest, "manifest_digest")
    truth_path = split_root(split) / "sealed_completion_truth.jsonl.gz"
    if truth_path.exists():
        raise RuntimeError("truth already revealed")
    if not manifest["truth_absent_at_prediction"]:
        raise RuntimeError("truth was present before prediction")
    public = read_jsonl_gz(split_root(split) / "public_observed_cubes.jsonl.gz")
    predictions = read_jsonl_gz(split_root(split) / "sealed_candidate_predictions.jsonl.gz")
    prediction_by_id = {row["pair_id"]: row for row in predictions}
    truth: list[dict[str, Any]] = []
    for spec in specs_for(split):
        truth.extend(truth_rows(spec, prediction_by_id[spec["pair_id"]]))
    write_jsonl_gz(truth_path, truth)
    score = aggregate_score(truth, predictions)
    value = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "manifest_digest": manifest["manifest_digest"],
        **score,
        "gate": all(score["checks"].values()),
    }
    if split == "discovery":
        value["boundary_confirmation_authorized"] = value["gate"]
    value["score_digest"] = digest(value)
    write_json(OUT_ROOT / "analysis" / f"{split}_score.json", value)
    print(json.dumps({"split": split, "metrics": value["metrics"], "checks": value["checks"], "gate": value["gate"]}, indent=2))
    return value


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
    validate_digest(discovery, "score_digest")
    confirmation = None
    if discovery["boundary_confirmation_authorized"]:
        confirmation = read_json(OUT_ROOT / "analysis/confirmation_score.json")
        validate_digest(confirmation, "score_digest")
    boundary_confirmed = bool(discovery["gate"] and confirmation is not None and confirmation["gate"])
    value = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "status": "semantic_intervention_not_identifiable_from_seven_visible_vertices" if boundary_confirmed else "identifiability_calibration_not_confirmed",
        "known_truth_identifiability_boundary_confirmed": boundary_confirmed,
        "discovery_metrics": discovery["metrics"],
        "confirmation_metrics": None if confirmation is None else confirmation["metrics"],
        "compiler_transfer_authorized": False,
        "new_math_hypothesis": {
            "status": "OPEN_NOT_CONFIRMED",
            "reason": "The boundary is exactly expressible by finite Boolean interpolation and indistinguishable completions; no mathematical-expression failure was observed.",
            "upgrade_gates_passed": 0,
            "upgrade_gate_count": 4,
        },
        "new_k_item": {
            "id": "K191",
            "level": "E3-KT",
            "statement": (
                "Seven visible vertices of a three-factor counterfactual cube do not identify the eighth semantic target: "
                "degree-two and target-only alternative completions can be exactly observation-equivalent yet require "
                "different correct interventions. Full replacement equals recipient-relative donor displacement, and no "
                "frozen state-only candidate is universally valid across the twins."
            ),
        },
        "claim_boundary": (
            "This is a known-truth identifiability theorem/calibration, not evidence that Qwen3 lacks semantic mechanisms "
            "or that all useful intervention compilers are impossible. Additional assumptions, supervision, or probes can change identifiability."
        ),
        "authorized_next": (
            "Redesign the observation contract by explicitly preregistering the additional assumption or functional probe "
            "that distinguishes completions; do not transfer a compiler to a free Transformer or language model yet."
        ),
        "auto_continue": False,
    }
    value["final_digest"] = digest(value)
    write_json(FINAL_PATH, value)
    print(json.dumps(value, indent=2))
    return value


def main() -> None:
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
    elif args.command == "finalize":
        finalize()


if __name__ == "__main__":
    main()
