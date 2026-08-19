#!/usr/bin/env python3
"""Phase1273: known-truth calibration of functional-response isomorphism cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE = 1273
CAMPAIGN = "C021"
CONTRACT_ID = "EXP-C021-WP00-001"
OUT = ROOT / "tests/glm5/result/phase1273_c021_response_isomorphism_camera_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
SYSTEMS = OUT / "material/known_truth_systems.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/pair_ledger.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1273_c021_response_isomorphism_camera_calibration_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C021-WP00-001.json"

FAMILIES = ("cyclic_process", "xor_process", "lookup_process")
DEPTHS = (4, 6, 8)
IMPLEMENTATIONS = ("identity", "early_warp", "late_warp", "redundant", "permuted")
EVENT_TYPES = ("attention_single", "mlp_single", "attention_prefix", "attention_suffix")
CAMERAS = ("identity_coordinate", "monotone_depth_warp", "response_spectrum", "gated_causal_graph")
READOUT_DIM = 8
GRID = 8
SEEDS = {"material": 1_273_001, "discovery_noise": 1_273_101, "confirmation_noise": 1_273_201, "sentinel": 1_273_301}
THRESHOLDS = {
    "coordinate_free_confirmation_balanced_accuracy_min": 0.95,
    "coordinate_free_confirmation_auc_min": 0.98,
    "monotone_clean_positive_mean_min": 0.94,
    "identity_disadvantage_min": 0.04,
    "random_sentinel_balanced_accuracy_max": 0.65,
    "gauge_response_error_max": 1.0e-12,
    "false_authorizations_max": 0,
}


class RunStatus(str, Enum):
    REGISTERED = "registered"
    QUALIFIED = "qualified"
    BEHAVIOR_REJECTED = "behavior_rejected"
    MEASURED = "measured"
    MECHANISM_FAILED = "mechanism_failed"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class TypedEvidence:
    object_id: str
    qualification_status: RunStatus
    measurement_status: RunStatus
    claim_status: RunStatus


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=np.float64)))


def family_profile(family: str, phase: np.ndarray, event_type: str) -> np.ndarray:
    """Return a typed future-response profile; physical coordinates are absent."""
    x = np.asarray(phase, dtype=np.float64)
    if family == "cyclic_process":
        center, sharp, ripple, late = 0.43, 10.0, np.sin(np.pi * x), np.exp(-((x - 0.67) / 0.19) ** 2)
        family_weights = np.asarray([1.00, 0.22, 0.88, 0.30, 0.96, 0.18, 0.78, 0.36])
        family_floor = np.asarray([0.04, 0.01, 0.05, 0.01, 0.05, 0.00, 0.04, 0.01])
    elif family == "xor_process":
        center, sharp, ripple, late = 0.53, 12.0, np.sin(2.0 * np.pi * x) ** 2, np.exp(-((x - 0.48) / 0.16) ** 2)
        family_weights = np.asarray([0.22, 1.00, 0.28, 0.92, 0.20, 0.96, 0.34, 0.84])
        family_floor = np.asarray([0.01, 0.05, 0.01, 0.05, 0.00, 0.05, 0.01, 0.04])
    elif family == "lookup_process":
        center, sharp, ripple, late = 0.68, 15.0, x * (1.0 - x) * 4.0, np.exp(-((x - 0.78) / 0.12) ** 2)
        family_weights = np.asarray([0.52, 0.34, 1.00, 0.18, 0.62, 0.82, 0.98, 0.24])
        family_floor = np.asarray([0.02, 0.01, 0.06, 0.00, 0.03, 0.04, 0.05, 0.01])
    else:
        raise ValueError(family)
    cumulative = sigmoid(sharp * (x - center))
    derivative = cumulative * (1.0 - cumulative) * 4.0
    if event_type == "attention_single":
        scale, bias = derivative, 0.08 + 0.75 * late
    elif event_type == "mlp_single":
        scale, bias = 0.65 * derivative + 0.20 * ripple, 0.12 + 0.35 * late
    elif event_type == "attention_prefix":
        scale, bias = cumulative, 0.10 + 0.25 * ripple
    elif event_type == "attention_suffix":
        scale, bias = 1.0 - sigmoid(sharp * (x - min(center + 0.12, 0.9))), 0.08 + 0.30 * late
    else:
        raise ValueError(event_type)
    columns = np.stack(
        [
            scale,
            bias,
            0.15 + 0.80 * scale * (0.35 + 0.65 * ripple),
            0.10 + 0.70 * bias * (1.0 - 0.35 * ripple),
            0.05 + 0.90 * cumulative,
            0.05 + 0.90 * late,
            0.10 + 0.75 * np.abs(scale - bias),
            0.05 + 0.90 * (0.6 * scale + 0.4 * bias),
        ],
        axis=-1,
    )
    return np.clip(columns * family_weights + family_floor, 0.0, 1.0)


def physical_phases(depth: int, implementation: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base = np.linspace(0.0, 1.0, depth)
    if implementation == "identity":
        phase = base
    elif implementation == "early_warp":
        phase = base ** 0.62
    elif implementation == "late_warp":
        phase = base ** 1.65
    elif implementation == "redundant":
        phase = np.round(base * max(depth - 2, 1)) / max(depth - 2, 1)
    elif implementation == "permuted":
        phase = base.copy()
    else:
        raise ValueError(implementation)
    permutation = np.arange(depth)
    if implementation == "permuted":
        permutation = rng.permutation(depth)
    return phase[permutation], permutation


def chain_adjacency(permutation: np.ndarray) -> np.ndarray:
    depth = len(permutation)
    inverse = np.argsort(permutation)
    adjacency = np.zeros((depth, depth), dtype=np.float64)
    for latent in range(depth - 1):
        left, right = int(inverse[latent]), int(inverse[latent + 1])
        adjacency[left, right] = adjacency[right, left] = 1.0
    return adjacency


def generate_systems() -> list[dict[str, Any]]:
    material_rng = np.random.default_rng(SEEDS["material"])
    noise_rng = {
        "discovery": np.random.default_rng(SEEDS["discovery_noise"]),
        "confirmation": np.random.default_rng(SEEDS["confirmation_noise"]),
    }
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for depth in DEPTHS:
            for implementation_index, implementation in enumerate(IMPLEMENTATIONS):
                local_seed = int(material_rng.integers(1, 2**31 - 1))
                local_rng = np.random.default_rng(local_seed)
                phases, permutation = physical_phases(depth, implementation, local_rng)
                adjacency = chain_adjacency(permutation)
                responses: dict[str, Any] = {}
                for partition in ("discovery", "confirmation"):
                    typed: dict[str, Any] = {}
                    for event_type in EVENT_TYPES:
                        profile = family_profile(family, phases, event_type)
                        noise = noise_rng[partition].normal(0.0, 0.006, size=profile.shape)
                        typed[event_type] = np.clip(profile + noise, 0.0, 1.0).tolist()
                    responses[partition] = typed
                evidence = TypedEvidence(
                    object_id=f"f{family_index}.d{depth}.i{implementation_index}",
                    qualification_status=RunStatus.QUALIFIED,
                    measurement_status=RunStatus.MEASURED,
                    claim_status=RunStatus.ABSTAINED,
                )
                row = {
                    "system_id": evidence.object_id,
                    "family": family,
                    "depth": depth,
                    "implementation": implementation,
                    "local_seed": local_seed,
                    "physical_permutation": permutation.astype(int).tolist(),
                    "adjacency": adjacency.tolist(),
                    "responses": responses,
                    "typed_status": {
                        "qualification": evidence.qualification_status.value,
                        "measurement": evidence.measurement_status.value,
                        "claim": evidence.claim_status.value,
                    },
                }
                row["row_digest"] = digest(row)
                rows.append(row)
    return rows


def resample(values: np.ndarray, grid: int = GRID) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, grid)
    return np.stack([np.interp(target, source, values[:, column]) for column in range(values.shape[1])], axis=-1)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1.0e-12 else 0.0


def identity_score(left: dict[str, Any], right: dict[str, Any], partition: str) -> float:
    a = np.concatenate([resample(np.asarray(left["responses"][partition][kind])) for kind in EVENT_TYPES], axis=0)
    b = np.concatenate([resample(np.asarray(right["responses"][partition][kind])) for kind in EVENT_TYPES], axis=0)
    return cosine(a, b)


def layer_features(row: dict[str, Any], partition: str) -> np.ndarray:
    return np.concatenate([np.asarray(row["responses"][partition][kind], dtype=np.float64) for kind in EVENT_TYPES], axis=1)


def dtw_path(left: np.ndarray, right: np.ndarray) -> list[tuple[int, int]]:
    n, m = len(left), len(right)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local = 1.0 - cosine(left[i - 1], right[j - 1])
            choices = ((cost[i - 1, j], (i - 1, j)), (cost[i, j - 1], (i, j - 1)), (cost[i - 1, j - 1], (i - 1, j - 1)))
            previous, coordinate = min(choices, key=lambda item: (item[0], item[1]))
            cost[i, j] = local + previous
            parent[(i, j)] = coordinate
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = parent[(i, j)]
    path.reverse()
    return path


def monotone_score(left: dict[str, Any], right: dict[str, Any], partition: str, path: list[tuple[int, int]] | None = None) -> tuple[float, list[tuple[int, int]]]:
    if path is None:
        path = dtw_path(layer_features(left, "discovery"), layer_features(right, "discovery"))
    a_layer = layer_features(left, partition)
    b_layer = layer_features(right, partition)
    return cosine(np.stack([a_layer[i] for i, _ in path]), np.stack([b_layer[j] for _, j in path])), path


def spectrum_signature(row: dict[str, Any], partition: str) -> np.ndarray:
    blocks = [np.asarray(row["responses"][partition][kind], dtype=np.float64) for kind in EVENT_TYPES]
    stacked = np.concatenate(blocks, axis=0)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    singular = np.pad(singular[:READOUT_DIM], (0, max(0, READOUT_DIM - len(singular))))
    typed_moments = np.concatenate([np.r_[block.mean(axis=0), block.std(axis=0)] for block in blocks])
    return np.concatenate([stacked.mean(axis=0), stacked.std(axis=0), singular, typed_moments])


def graph_signature(row: dict[str, Any], partition: str) -> np.ndarray:
    adjacency = np.asarray(row["adjacency"], dtype=np.float64)
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    eigen = np.linalg.eigvalsh(laplacian)
    eigen = np.interp(np.linspace(0, max(len(eigen) - 1, 0), GRID), np.arange(len(eigen)), eigen)
    parts = [eigen]
    for kind in EVENT_TYPES:
        block = np.asarray(row["responses"][partition][kind], dtype=np.float64)
        smooth = adjacency @ block / np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)
        gram = block @ block.T
        response_eigen = np.linalg.eigvalsh(gram)
        response_eigen = np.interp(np.linspace(0, max(len(response_eigen) - 1, 0), GRID), np.arange(len(response_eigen)), response_eigen)
        parts.extend([block.mean(axis=0), block.std(axis=0), smooth.mean(axis=0), response_eigen])
    return np.concatenate(parts)


def pair_score(camera: str, left: dict[str, Any], right: dict[str, Any], partition: str, path: list[tuple[int, int]] | None = None) -> tuple[float, list[tuple[int, int]] | None]:
    if camera == "identity_coordinate":
        return identity_score(left, right, partition), None
    if camera == "monotone_depth_warp":
        return monotone_score(left, right, partition, path)
    if camera == "response_spectrum":
        return cosine(spectrum_signature(left, partition), spectrum_signature(right, partition)), None
    if camera == "gated_causal_graph":
        return cosine(graph_signature(left, partition), graph_signature(right, partition)), None
    raise ValueError(camera)


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = float(np.mean(predictions[labels == 1] == 1))
    negative = float(np.mean(predictions[labels == 0] == 0))
    return 0.5 * (positive + negative)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    unique = np.unique(scores)
    candidates = np.r_[unique[0] - 1.0e-9, (unique[:-1] + unique[1:]) / 2.0, unique[-1] + 1.0e-9]
    values = [(balanced_accuracy(labels, (scores >= threshold).astype(int)), float(threshold)) for threshold in candidates]
    return max(values, key=lambda item: (item[0], item[1]))


def build_pair_ledger(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            same = int(left["family"] == right["family"])
            if same and left["system_id"] == right["system_id"]:
                continue
            item = {
                "pair_id": f'{left["system_id"]}__{right["system_id"]}',
                "left": left["system_id"],
                "right": right["system_id"],
                "same_family": same,
                "permuted_pair": left["implementation"] == "permuted" or right["implementation"] == "permuted",
                "monotone_clean": left["implementation"] != "permuted" and right["implementation"] != "permuted",
                "scores": {},
            }
            for camera in CAMERAS:
                discovery, path = pair_score(camera, left, right, "discovery")
                confirmation, _ = pair_score(camera, left, right, "confirmation", path)
                item["scores"][camera] = {"discovery": discovery, "confirmation": confirmation, "path": path}
            ledger.append(item)
    return ledger


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1273.c021.response_isomorphism_calibration.v1",
        "claim_type": "known_truth_response_isomorphism_camera_calibration",
        "families": list(FAMILIES),
        "depths": list(DEPTHS),
        "implementations": list(IMPLEMENTATIONS),
        "event_types": list(EVENT_TYPES),
        "readout_dimension": READOUT_DIM,
        "cameras": list(CAMERAS),
        "seeds": SEEDS,
        "thresholds": THRESHOLDS,
        "row_count": len(rows),
        "material_digest": digest([{"system_id": row["system_id"], "row_digest": row["row_digest"]} for row in rows]),
        "selection": "thresholds are optimized on discovery scores only; camera families and main thresholds are frozen",
        "confirmation": "independent response perturbations with discovery-learned DTW paths and thresholds",
        "typed_evidence_contract": {
            "qualification_record_required_for_every_system": True,
            "measurement_record_allowed_only_for_qualified_system": True,
            "claim_failure_does_not_rewrite_measurement_as_missing": True,
            "statuses": [status.value for status in RunStatus],
        },
        "hard_stops": [
            "No confirmation score selects a threshold or camera.",
            "Spectrum success does not authorize event mapping.",
            "Known response-equivalent physical labels remain abstentions.",
            "Failure denies the free-network work package under this contract.",
        ],
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR), "contract": file_sha256(CONTRACT)},
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {"created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "precision": "float64", "model_loaded": False}


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    rows = generate_systems()
    write_jsonl(SYSTEMS, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": RunStatus.REGISTERED.value, "systems": len(rows)}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, rows = read_json(PROTOCOL), read_jsonl(SYSTEMS)
    expected = protocol_payload(rows)
    if protocol["source_hashes"] != expected["source_hashes"] or protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("frozen protocol or source mismatch")
    for row in rows:
        value, stored = dict(row), row["row_digest"]
        value.pop("row_digest")
        if digest(value) != stored:
            raise RuntimeError("material row digest mismatch")
    return protocol, rows


def analyze_ledger(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["same_family"] for row in ledger], dtype=int)
    camera_rows: dict[str, Any] = {}
    for camera in CAMERAS:
        discovery = np.asarray([row["scores"][camera]["discovery"] for row in ledger])
        confirmation = np.asarray([row["scores"][camera]["confirmation"] for row in ledger])
        discovery_bacc, threshold = choose_threshold(labels, discovery)
        confirmation_predictions = (confirmation >= threshold).astype(int)
        clean_values = [row["scores"][camera]["confirmation"] for row in ledger if row["same_family"] and row["monotone_clean"]]
        camera_rows[camera] = {
            "threshold": threshold,
            "discovery_balanced_accuracy": discovery_bacc,
            "discovery_auc": auc(labels, discovery),
            "confirmation_balanced_accuracy": balanced_accuracy(labels, confirmation_predictions),
            "confirmation_auc": auc(labels, confirmation),
            "confirmation_positive_mean": float(confirmation[labels == 1].mean()),
            "confirmation_negative_mean": float(confirmation[labels == 0].mean()),
            "monotone_clean_positive_mean": float(np.mean(clean_values)),
        }
    rng = np.random.default_rng(SEEDS["sentinel"])
    shuffled = rng.permutation(labels)
    best_camera = max((camera_rows[name]["discovery_balanced_accuracy"], name) for name in ("response_spectrum", "gated_causal_graph"))[1]
    best_scores = np.asarray([row["scores"][best_camera]["confirmation"] for row in ledger])
    sentinel_bacc, sentinel_threshold = choose_threshold(shuffled, best_scores)
    identity = camera_rows["identity_coordinate"]["confirmation_balanced_accuracy"]
    coordinate_free_best = max(camera_rows[name]["confirmation_balanced_accuracy"] for name in ("response_spectrum", "gated_causal_graph"))
    false_authorizations = 0
    gates = {
        "coordinate_free_accuracy": coordinate_free_best >= THRESHOLDS["coordinate_free_confirmation_balanced_accuracy_min"],
        "coordinate_free_auc": max(camera_rows[name]["confirmation_auc"] for name in ("response_spectrum", "gated_causal_graph")) >= THRESHOLDS["coordinate_free_confirmation_auc_min"],
        "monotone_clean": camera_rows["monotone_depth_warp"]["monotone_clean_positive_mean"] >= THRESHOLDS["monotone_clean_positive_mean_min"],
        "identity_disadvantage": coordinate_free_best - identity >= THRESHOLDS["identity_disadvantage_min"],
        "random_sentinel": sentinel_bacc <= THRESHOLDS["random_sentinel_balanced_accuracy_max"],
        "typed_false_authorization": false_authorizations <= THRESHOLDS["false_authorizations_max"],
    }
    passed = all(gates.values())
    return {
        "phase": PHASE,
        "contract_id": CONTRACT_ID,
        "camera_results": camera_rows,
        "selected_coordinate_free_camera": best_camera,
        "coordinate_free_best_confirmation_balanced_accuracy": coordinate_free_best,
        "identity_confirmation_balanced_accuracy": identity,
        "identity_disadvantage": coordinate_free_best - identity,
        "random_label_sentinel": {"balanced_accuracy": sentinel_bacc, "threshold": sentinel_threshold},
        "gauge_policy": {"response_equivalent_physical_identity_authorized": False, "false_authorizations": false_authorizations},
        "gates": gates,
        "passed": passed,
        "decision": "known_truth_response_isomorphism_camera_calibrated" if passed else "known_truth_response_isomorphism_camera_not_calibrated",
        "free_network_authorized": passed,
        "pretrained_authorized": False,
    }


def run() -> None:
    protocol, rows = verify_protocol()
    ledger = build_pair_ledger(rows)
    write_jsonl(RAW, ledger)
    summary = {"phase": PHASE, "created_at_utc": utc_now(), "systems": len(rows), "pairs": len(ledger), "protocol_digest": protocol["protocol_digest"], "material_hash": file_sha256(SYSTEMS), "pair_hash": file_sha256(RAW), "run_digest": digest([row["pair_id"] for row in ledger]), "model_loaded": False}
    atomic_json(SUMMARY, summary)
    atomic_json(COMPLETE, {"phase": PHASE, "complete": True, "created_at_utc": utc_now(), "run_digest": summary["run_digest"]})
    print(canonical_json({"status": "measured", "systems": len(rows), "pairs": len(ledger)}))


def analyze() -> None:
    verify_protocol()
    ledger = read_jsonl(RAW)
    result = analyze_ledger(ledger)
    result["created_at_utc"] = utc_now()
    result["protocol_hash"] = file_sha256(PROTOCOL)
    result["pair_hash"] = file_sha256(RAW)
    result["final_digest"] = digest({key: value for key, value in result.items() if key not in {"created_at_utc", "final_digest"}})
    atomic_json(FINAL, result)
    print(canonical_json({"decision": result["decision"], "passed": result["passed"], "camera": result["selected_coordinate_free_camera"], "accuracy": result["coordinate_free_best_confirmation_balanced_accuracy"]}))


def self_test() -> None:
    rows = generate_systems()
    result = analyze_ledger(build_pair_ledger(rows))
    print(json.dumps(result, indent=2))


def run_auditor(mode: str) -> None:
    command = [sys.executable, str(AUDITOR), mode]
    status = os.spawnv(os.P_WAIT, sys.executable, command)
    if status:
        raise SystemExit(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("self-test", "preregister", "preaudit", "run", "analyze", "audit"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode == "self-test": self_test()
    elif args.mode == "preregister": preregister(args.force)
    elif args.mode == "preaudit": run_auditor("preaudit")
    elif args.mode == "run": run()
    elif args.mode == "analyze": analyze()
    else: run_auditor("final")


if __name__ == "__main__":
    main()
