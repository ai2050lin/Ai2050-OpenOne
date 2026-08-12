"""Known-truth calibration for a conditional causal trace mapper.

The benchmark is intentionally not a language-model experiment. It asks a
narrow measurement question: can a mapper recover functional equivalence
classes when physical coordinates rotate across contexts, identify a shared
shell and family-specific residuals, predict held-out additive compositions,
and refuse to distinguish mechanisms that are observationally equivalent
under the frozen intervention registry?

Only elementary operations are used: matched differences, Gram matrices,
cosine similarity, nearest centroids, and fixed finite gates.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE = 1200
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1200_conditional_causal_trace_mapper_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1200_conditional_causal_trace_mapper"
DEVELOPMENT_ROWS = OUT_ROOT / "development/rows.json"
DEVELOPMENT_SUMMARY = OUT_ROOT / "development/summary.json"
DEVELOPMENT_AUDIT = OUT_ROOT / "development/independent_audit.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
FORMAL_DISCOVERY_ROWS = OUT_ROOT / "runs/formal/discovery_rows.json"
FORMAL_CONFIRMATION_ROWS = OUT_ROOT / "runs/formal/confirmation_rows.json"
FORMAL_SEAL = OUT_ROOT / "runs/formal/seal.json"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

N_PRIMITIVES = 12
OBSERVATION_DIM = 96
CORE_PROBES = 12
EXTENDED_PROBES = 16
CONTEXTS = 4
REPLICATES = 3
TWIN_REPLICATES = 4
SUPPORT_CONTEXTS = (0, 1)
EVALUATION_CONTEXTS = (2, 3)
BASE_BLOCK = 4
SHELL_ROWS = slice(48, 52)
INTERACTION_ROWS = slice(52, 80)
HIDDEN_ROWS = slice(80, 96)
SURFACE_STRENGTH = 3.5
INSTANCE_RESIDUAL_STRENGTH = 0.025
MEASUREMENT_NOISE_STRENGTH = 0.008
INTERACTION_STRENGTH = 2.25
HIDDEN_STRENGTH = 1.50

ALL_PAIRS = tuple(itertools.combinations(range(N_PRIMITIVES), 2))
PAIR_ORDER_SEED = 1_200_007
FORMAL_SEEDS = {"discovery": 1_200_101, "confirmation": 1_200_211}
DEVELOPMENT_SEED = 1_200_019

THRESHOLDS = {
    "family_invariant_accuracy_min": 0.90,
    "family_invariant_advantage_min": 0.60,
    "raw_coordinate_accuracy_max": 0.25,
    "matched_null_accuracy_max": 0.20,
    "shuffled_registry_accuracy_max": 0.25,
    "norm_only_accuracy_max": 0.20,
    "leakage_sentinel_accuracy_min": 0.999,
    "shared_shell_cosine_mean_min": 0.98,
    "shared_shell_cosine_min_min": 0.94,
    "additive_composition_top1_min": 0.90,
    "additive_composition_cosine_mean_min": 0.95,
    "interaction_balanced_accuracy_min": 0.90,
    "interaction_residual_gap_min": 0.12,
    "core_twin_accuracy_max": 0.55,
    "core_twin_centroid_distance_max": 1e-10,
    "extended_twin_accuracy_min": 0.95,
    "extended_twin_centroid_distance_min": 0.10,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(root: int, *parts: Any) -> int:
    payload = canonical_json([root, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-15:
        return np.zeros_like(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit(left), unit(right)))


def normalized_random_matrix(seed: int, rows: slice | None = None, columns: slice | None = None) -> np.ndarray:
    result = np.zeros((OBSERVATION_DIM, EXTENDED_PROBES), dtype=np.float64)
    row_slice = rows if rows is not None else slice(0, OBSERVATION_DIM)
    column_slice = columns if columns is not None else slice(0, EXTENDED_PROBES)
    rng = np.random.default_rng(seed)
    values = rng.normal(size=result[row_slice, column_slice].shape)
    values /= max(float(np.linalg.norm(values)), 1e-15)
    result[row_slice, column_slice] = values
    return result


def balanced_probe_matrix(seed: int, rows: slice) -> np.ndarray:
    """Build a basis with equal core and extended energy across primitives."""
    result = np.zeros((OBSERVATION_DIM, EXTENDED_PROBES), dtype=np.float64)
    rng = np.random.default_rng(seed)
    core = rng.normal(size=result[rows, :CORE_PROBES].shape)
    core /= max(float(np.linalg.norm(core)), 1e-15)
    extended = rng.normal(size=result[rows, CORE_PROBES:].shape)
    extended /= max(float(np.linalg.norm(extended)), 1e-15)
    result[rows, :CORE_PROBES] = core
    result[rows, CORE_PROBES:] = 0.5 * extended
    return result


def signed_gauge(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(OBSERVATION_DIM)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=OBSERVATION_DIM)
    return permutation, signs


def apply_gauge(matrix: np.ndarray, gauge: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    permutation, signs = gauge
    return matrix[permutation, :] * signs[:, None]


def gram_vector(matrix: np.ndarray, probes: int) -> np.ndarray:
    gram = matrix[:, :probes].T @ matrix[:, :probes]
    indices = np.triu_indices(probes)
    return gram[indices].astype(np.float64)


def raw_vector(matrix: np.ndarray) -> np.ndarray:
    return unit(matrix[:, :CORE_PROBES].reshape(-1))


def interaction_truth(pair: tuple[int, int]) -> bool:
    left, right = pair
    return ((left + 2 * right) % 3) == 0


def pair_assignment() -> dict[str, tuple[tuple[int, int], ...]]:
    rng = np.random.default_rng(PAIR_ORDER_SEED)
    order = rng.permutation(len(ALL_PAIRS))
    midpoint = len(ALL_PAIRS) // 2
    return {
        "discovery": tuple(ALL_PAIRS[int(index)] for index in order[:midpoint]),
        "confirmation": tuple(ALL_PAIRS[int(index)] for index in order[midpoint:]),
        "development": ALL_PAIRS,
    }


def make_basis(seed: int) -> dict[str, Any]:
    primitives = []
    for primitive in range(N_PRIMITIVES):
        rows = slice(primitive * BASE_BLOCK, (primitive + 1) * BASE_BLOCK)
        primitives.append(balanced_probe_matrix(stable_seed(seed, "primitive", primitive), rows))
    shell = balanced_probe_matrix(stable_seed(seed, "shell"), SHELL_ROWS)
    interactions = {
        f"{left}-{right}": balanced_probe_matrix(stable_seed(seed, "interaction", left, right), INTERACTION_ROWS)
        for left, right in ALL_PAIRS
    }
    hidden = normalized_random_matrix(
        stable_seed(seed, "hidden"), rows=HIDDEN_ROWS, columns=slice(CORE_PROBES, EXTENDED_PROBES)
    )
    return {"primitives": primitives, "shell": shell, "interactions": interactions, "hidden": hidden}


def panel_estimates(
    seed: int,
    key: tuple[Any, ...],
    content: np.ndarray,
    shell: np.ndarray,
    gauge: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface = SURFACE_STRENGTH * normalized_random_matrix(stable_seed(seed, *key, "surface"))
    active_noise = MEASUREMENT_NOISE_STRENGTH * normalized_random_matrix(stable_seed(seed, *key, "active_noise"))
    null_noise = MEASUREMENT_NOISE_STRENGTH * normalized_random_matrix(stable_seed(seed, *key, "null_noise"))
    surface_noise = MEASUREMENT_NOISE_STRENGTH * normalized_random_matrix(stable_seed(seed, *key, "surface_noise"))
    active = apply_gauge(shell + content, gauge) + surface + active_noise
    matched_null = apply_gauge(shell, gauge) + surface + null_noise
    surface_control = surface + surface_noise
    return active - matched_null, matched_null - surface_control, surface_control


def make_family_rows(corpus: str, seed: int, basis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context in range(CONTEXTS):
        gauge = signed_gauge(stable_seed(seed, corpus, "gauge", context))
        role = "support" if context in SUPPORT_CONTEXTS else "evaluation"
        for family in range(N_PRIMITIVES):
            for replicate in range(REPLICATES):
                residual = INSTANCE_RESIDUAL_STRENGTH * normalized_random_matrix(
                    stable_seed(seed, corpus, "family_residual", family, context, replicate)
                )
                content = basis["primitives"][family] + residual
                key = (corpus, "family", family, context, replicate)
                content_estimate, shell_estimate, surface = panel_estimates(
                    seed, key, content, basis["shell"], gauge
                )
                rng = np.random.default_rng(stable_seed(seed, corpus, "registry_shuffle", family, context, replicate))
                shuffled = content_estimate[:, rng.permutation(CORE_PROBES)]
                gram = gram_vector(content_estimate, CORE_PROBES)
                rows.append(
                    {
                        "row_type": "family",
                        "corpus": corpus,
                        "context": context,
                        "role": role,
                        "family": family,
                        "replicate": replicate,
                        "gram_core": gram.tolist(),
                        "signature": unit(gram).tolist(),
                        "raw": raw_vector(content_estimate).tolist(),
                        "null_signature": unit(gram_vector(shell_estimate, CORE_PROBES)).tolist(),
                        "shuffled_signature": unit(gram_vector(shuffled, CORE_PROBES)).tolist(),
                        "norm_only": float(np.linalg.norm(content_estimate[:, :CORE_PROBES])),
                        "surface_norm": float(np.linalg.norm(surface[:, :CORE_PROBES])),
                        "sentinel": [1.0 if index == family else 0.0 for index in range(N_PRIMITIVES)],
                    }
                )
    return rows


def make_pair_rows(
    corpus: str,
    seed: int,
    basis: dict[str, Any],
    pairs: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context in EVALUATION_CONTEXTS:
        gauge = signed_gauge(stable_seed(seed, corpus, "gauge", context))
        for left, right in pairs:
            interacting = interaction_truth((left, right))
            for replicate in range(REPLICATES):
                residual = INSTANCE_RESIDUAL_STRENGTH * normalized_random_matrix(
                    stable_seed(seed, corpus, "pair_residual", left, right, context, replicate)
                )
                content = basis["primitives"][left] + basis["primitives"][right] + residual
                if interacting:
                    content = content + INTERACTION_STRENGTH * basis["interactions"][f"{left}-{right}"]
                key = (corpus, "pair", left, right, context, replicate)
                content_estimate, _, _ = panel_estimates(seed, key, content, basis["shell"], gauge)
                gram = gram_vector(content_estimate, CORE_PROBES)
                rows.append(
                    {
                        "row_type": "pair",
                        "corpus": corpus,
                        "context": context,
                        "pair": [left, right],
                        "interacting": interacting,
                        "replicate": replicate,
                        "gram_core": gram.tolist(),
                        "signature": unit(gram).tolist(),
                    }
                )
    return rows


def make_twin_rows(corpus: str, seed: int, basis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = basis["primitives"][0] + basis["primitives"][1]
    for context in range(CONTEXTS):
        gauge = signed_gauge(stable_seed(seed, corpus, "gauge", context))
        role = "support" if context in SUPPORT_CONTEXTS else "evaluation"
        for replicate in range(TWIN_REPLICATES):
            residual = INSTANCE_RESIDUAL_STRENGTH * normalized_random_matrix(
                stable_seed(seed, corpus, "twin_residual", context, replicate)
            )
            key = (corpus, "twin", context, replicate)
            for label in (0, 1):
                content = base + residual + (HIDDEN_STRENGTH * basis["hidden"] if label else 0.0)
                content_estimate, _, _ = panel_estimates(seed, key, content, basis["shell"], gauge)
                core = gram_vector(content_estimate, CORE_PROBES)
                extended = gram_vector(content_estimate, EXTENDED_PROBES)
                rows.append(
                    {
                        "row_type": "twin",
                        "corpus": corpus,
                        "context": context,
                        "role": role,
                        "replicate": replicate,
                        "label": label,
                        "core_signature": unit(core).tolist(),
                        "extended_signature": unit(extended).tolist(),
                    }
                )
    return rows


def build_corpus(corpus: str, seed: int) -> dict[str, Any]:
    basis = make_basis(seed)
    pairs = pair_assignment()[corpus]
    return {
        "phase": PHASE,
        "corpus": corpus,
        "seed": seed,
        "pair_assignment": [list(pair) for pair in pairs],
        "family_rows": make_family_rows(corpus, seed, basis),
        "pair_rows": make_pair_rows(corpus, seed, basis, pairs),
        "twin_rows": make_twin_rows(corpus, seed, basis),
    }


def centroid(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.mean(np.asarray([row[field] for row in rows], dtype=np.float64), axis=0)


def nearest_label(feature: np.ndarray, centroids: dict[int, np.ndarray]) -> int:
    labels = sorted(centroids)
    scores = [cosine(feature, centroids[label]) for label in labels]
    return labels[int(np.argmax(scores))]


def classification_accuracy(
    support: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    field: str,
    label_field: str,
) -> tuple[float, dict[int, np.ndarray]]:
    labels = sorted({int(row[label_field]) for row in support})
    centroids = {
        label: centroid([row for row in support if int(row[label_field]) == label], field)
        for label in labels
    }
    correct = sum(
        nearest_label(np.asarray(row[field], dtype=np.float64), centroids) == int(row[label_field])
        for row in evaluation
    )
    return correct / max(len(evaluation), 1), centroids


def scalar_accuracy(
    support: list[dict[str, Any]], evaluation: list[dict[str, Any]], field: str, label_field: str
) -> float:
    labels = sorted({int(row[label_field]) for row in support})
    centers = {
        label: float(np.mean([float(row[field]) for row in support if int(row[label_field]) == label]))
        for label in labels
    }
    correct = 0
    for row in evaluation:
        prediction = min(labels, key=lambda label: (abs(float(row[field]) - centers[label]), label))
        correct += prediction == int(row[label_field])
    return correct / max(len(evaluation), 1)


def balanced_binary_accuracy(rows: list[dict[str, Any]], predictions: list[bool], truth_field: str) -> float:
    values = []
    for truth in (False, True):
        indices = [index for index, row in enumerate(rows) if bool(row[truth_field]) == truth]
        values.append(float(np.mean([predictions[index] == truth for index in indices])))
    return float(np.mean(values))


def summarize_corpus(payload: dict[str, Any]) -> dict[str, Any]:
    family_rows = payload["family_rows"]
    support = [row for row in family_rows if row["role"] == "support"]
    evaluation = [row for row in family_rows if row["role"] == "evaluation"]
    invariant_accuracy, invariant_centroids = classification_accuracy(support, evaluation, "signature", "family")
    raw_accuracy, _ = classification_accuracy(support, evaluation, "raw", "family")
    null_accuracy, _ = classification_accuracy(support, evaluation, "null_signature", "family")
    shuffled_accuracy, _ = classification_accuracy(support, evaluation, "shuffled_signature", "family")
    sentinel_accuracy, _ = classification_accuracy(support, evaluation, "sentinel", "family")
    norm_accuracy = scalar_accuracy(support, evaluation, "norm_only", "family")
    baseline_max = max(raw_accuracy, null_accuracy, shuffled_accuracy, norm_accuracy)

    shell_center = unit(centroid(support, "null_signature"))
    shell_cosines = [cosine(np.asarray(row["null_signature"]), shell_center) for row in evaluation]

    base_grams = {
        family: centroid([row for row in support if row["family"] == family], "gram_core")
        for family in range(N_PRIMITIVES)
    }
    pair_predictions = {
        pair: base_grams[pair[0]] + base_grams[pair[1]] for pair in ALL_PAIRS
    }
    pair_rows = payload["pair_rows"]
    composition_cosines: list[float] = []
    composition_correct: list[bool] = []
    residuals: list[float] = []
    interaction_predictions: list[bool] = []
    for row in pair_rows:
        observed = np.asarray(row["gram_core"], dtype=np.float64)
        pair = tuple(int(value) for value in row["pair"])
        scores = {candidate: cosine(observed, prediction) for candidate, prediction in pair_predictions.items()}
        correct_score = scores[pair]
        residual = 1.0 - correct_score
        residuals.append(residual)
        interaction_predictions.append(residual >= 0.10)
        if not row["interacting"]:
            composition_cosines.append(correct_score)
            composition_correct.append(max(scores, key=scores.get) == pair)
    additive_residuals = [value for value, row in zip(residuals, pair_rows) if not row["interacting"]]
    interaction_residuals = [value for value, row in zip(residuals, pair_rows) if row["interacting"]]
    interaction_balanced = balanced_binary_accuracy(pair_rows, interaction_predictions, "interacting")

    twin_rows = payload["twin_rows"]
    twin_support = [row for row in twin_rows if row["role"] == "support"]
    twin_evaluation = [row for row in twin_rows if row["role"] == "evaluation"]
    core_twin_accuracy, core_centroids = classification_accuracy(
        twin_support, twin_evaluation, "core_signature", "label"
    )
    extended_twin_accuracy, extended_centroids = classification_accuracy(
        twin_support, twin_evaluation, "extended_signature", "label"
    )
    core_twin_distance = float(np.linalg.norm(core_centroids[0] - core_centroids[1]))
    extended_twin_distance = float(np.linalg.norm(extended_centroids[0] - extended_centroids[1]))

    metrics = {
        "family_invariant_accuracy": invariant_accuracy,
        "family_raw_coordinate_accuracy": raw_accuracy,
        "family_matched_null_accuracy": null_accuracy,
        "family_shuffled_registry_accuracy": shuffled_accuracy,
        "family_norm_only_accuracy": norm_accuracy,
        "family_leakage_sentinel_accuracy": sentinel_accuracy,
        "family_invariant_advantage": invariant_accuracy - baseline_max,
        "shared_shell_cosine_mean": float(np.mean(shell_cosines)),
        "shared_shell_cosine_min": float(np.min(shell_cosines)),
        "additive_composition_top1_accuracy": float(np.mean(composition_correct)),
        "additive_composition_cosine_mean": float(np.mean(composition_cosines)),
        "interaction_balanced_accuracy": interaction_balanced,
        "additive_residual_mean": float(np.mean(additive_residuals)),
        "interaction_residual_mean": float(np.mean(interaction_residuals)),
        "interaction_residual_gap": float(np.mean(interaction_residuals) - np.mean(additive_residuals)),
        "core_twin_accuracy": core_twin_accuracy,
        "core_twin_centroid_distance": core_twin_distance,
        "extended_twin_accuracy": extended_twin_accuracy,
        "extended_twin_centroid_distance": extended_twin_distance,
    }
    gates = {
        "family_mapping": bool(
            metrics["family_invariant_accuracy"] >= THRESHOLDS["family_invariant_accuracy_min"]
            and metrics["family_invariant_advantage"] >= THRESHOLDS["family_invariant_advantage_min"]
            and metrics["family_raw_coordinate_accuracy"] <= THRESHOLDS["raw_coordinate_accuracy_max"]
            and metrics["family_matched_null_accuracy"] <= THRESHOLDS["matched_null_accuracy_max"]
            and metrics["family_shuffled_registry_accuracy"] <= THRESHOLDS["shuffled_registry_accuracy_max"]
            and metrics["family_norm_only_accuracy"] <= THRESHOLDS["norm_only_accuracy_max"]
            and metrics["family_leakage_sentinel_accuracy"] >= THRESHOLDS["leakage_sentinel_accuracy_min"]
        ),
        "shared_reuse": bool(
            metrics["shared_shell_cosine_mean"] >= THRESHOLDS["shared_shell_cosine_mean_min"]
            and metrics["shared_shell_cosine_min"] >= THRESHOLDS["shared_shell_cosine_min_min"]
        ),
        "composition": bool(
            metrics["additive_composition_top1_accuracy"] >= THRESHOLDS["additive_composition_top1_min"]
            and metrics["additive_composition_cosine_mean"] >= THRESHOLDS["additive_composition_cosine_mean_min"]
            and metrics["interaction_balanced_accuracy"] >= THRESHOLDS["interaction_balanced_accuracy_min"]
            and metrics["interaction_residual_gap"] >= THRESHOLDS["interaction_residual_gap_min"]
        ),
        "identifiability_boundary": bool(
            metrics["core_twin_accuracy"] <= THRESHOLDS["core_twin_accuracy_max"]
            and metrics["core_twin_centroid_distance"] <= THRESHOLDS["core_twin_centroid_distance_max"]
            and metrics["extended_twin_accuracy"] >= THRESHOLDS["extended_twin_accuracy_min"]
            and metrics["extended_twin_centroid_distance"] >= THRESHOLDS["extended_twin_centroid_distance_min"]
        ),
    }
    return {
        "corpus": payload["corpus"],
        "seed": payload["seed"],
        "counts": {
            "family_rows": len(family_rows),
            "pair_rows": len(pair_rows),
            "twin_rows": len(twin_rows),
            "pair_families": len(payload["pair_assignment"]),
            "additive_pair_rows": sum(not row["interacting"] for row in pair_rows),
            "interaction_pair_rows": sum(row["interacting"] for row in pair_rows),
        },
        "metrics": metrics,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def source_hashes() -> dict[str, str]:
    return {"phase1200": file_sha256(SCRIPT), "phase1200_audit": file_sha256(AUDIT_SCRIPT)}


def develop() -> None:
    if DEVELOPMENT_ROWS.exists() or DEVELOPMENT_SUMMARY.exists():
        raise RuntimeError("Phase1200 development outcomes already exist")
    payload = build_corpus("development", DEVELOPMENT_SEED)
    summary = summarize_corpus(payload)
    output = {
        "phase": PHASE,
        "kind": "known_truth_development",
        "created_at": utc_now(),
        "summary": summary,
        "development_gate_pass": summary["overall_pass"],
        "authorized_next": {"formal_preregistration": summary["overall_pass"]},
    }
    write_json(DEVELOPMENT_ROWS, payload)
    write_json(DEVELOPMENT_SUMMARY, output)
    print(canonical_json({"development_gate_pass": output["development_gate_pass"], "metrics": summary["metrics"]}))


def preregister() -> None:
    if PROTOCOL_PATH.exists() or FORMAL_SEAL.exists():
        raise RuntimeError("Phase1200 formal protocol already exists")
    development = read_json(DEVELOPMENT_SUMMARY)
    audit = read_json(DEVELOPMENT_AUDIT)
    if not development["development_gate_pass"] or not audit.get("gate_pass", False):
        raise RuntimeError("development calibration did not authorize formal execution")
    protocol = {
        "phase": PHASE,
        "schema_version": "phase1200.trace_mapper.protocol.v1",
        "created_at": utc_now(),
        "object": "conditional causal response equivalence class, not a fixed neural path",
        "formulas": {
            "content": "C=A-N",
            "shared_shell": "S=N-U",
            "gauge_signature": "q(C)=vec_upper(C^T C)",
            "additive_prediction": "q_unnorm(i+j)=q_unnorm(i)+q_unnorm(j)",
            "identifiability": "same response under registry implies same observable equivalence class only",
        },
        "dimensions": {
            "primitives": N_PRIMITIVES,
            "observation_dim": OBSERVATION_DIM,
            "core_probes": CORE_PROBES,
            "extended_probes": EXTENDED_PROBES,
            "contexts": CONTEXTS,
            "replicates": REPLICATES,
        },
        "formal_seeds": FORMAL_SEEDS,
        "pair_assignment": {key: [list(pair) for pair in value] for key, value in pair_assignment().items() if key != "development"},
        "thresholds": THRESHOLDS,
        "source_hashes": source_hashes(),
        "development": {
            "rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
            "audit_sha256": file_sha256(DEVELOPMENT_AUDIT),
        },
        "scope": {
            "allowed": "known-truth calibration of a trace-mapping measurement contract",
            "forbidden": [
                "natural-language mechanism claim",
                "one-pattern-one-fixed-path claim",
                "physical identity from observational equivalence",
                "automatic future-learning or optimizer-state rescue",
            ],
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"protocol_digest": protocol["protocol_digest"]}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    if digest(candidate) != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    return protocol


def run_formal() -> None:
    if FORMAL_DISCOVERY_ROWS.exists() or FORMAL_CONFIRMATION_ROWS.exists() or FORMAL_SEAL.exists():
        raise RuntimeError("Phase1200 formal outcomes already exist")
    protocol = verify_protocol()
    discovery = build_corpus("discovery", FORMAL_SEEDS["discovery"])
    confirmation = build_corpus("confirmation", FORMAL_SEEDS["confirmation"])
    write_json(FORMAL_DISCOVERY_ROWS, discovery)
    write_json(FORMAL_CONFIRMATION_ROWS, confirmation)
    seal = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "row_hashes": {
            "discovery": file_sha256(FORMAL_DISCOVERY_ROWS),
            "confirmation": file_sha256(FORMAL_CONFIRMATION_ROWS),
        },
    }
    seal["seal_digest"] = digest(seal)
    write_json(FORMAL_SEAL, seal)
    print(canonical_json({"formal_rows": {"discovery": len(discovery["family_rows"]), "confirmation": len(confirmation["family_rows"])}, "seal_digest": seal["seal_digest"]}))


def analyze() -> None:
    verify_protocol()
    if SUMMARY_PATH.exists() or CLAIMS_PATH.exists():
        raise RuntimeError("Phase1200 analysis already exists")
    discovery = summarize_corpus(read_json(FORMAL_DISCOVERY_ROWS))
    confirmation = summarize_corpus(read_json(FORMAL_CONFIRMATION_ROWS))
    positive = discovery["overall_pass"] and confirmation["overall_pass"]
    status = "known_truth_trace_mapper_confirmed" if positive else "known_truth_trace_mapper_not_confirmed"
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": status,
        "discovery": discovery,
        "confirmation": confirmation,
        "formal_decision": "positive" if positive else "not_confirmed",
    }
    claims = {
        "conditional_causal_trace_mapper": {
            "type": "E3-KT" if positive else "E3-KT-scope-boundary",
            "status": status,
            "confirmed_if_positive": "A matched-difference Gram mapper can recover known functional families across coordinate gauges, predict additive held-out compositions, and preserve a registry-relative non-identifiability boundary.",
            "not_claimed": [
                "language is ontologically exhausted by the synthetic pattern families",
                "every linguistic pattern has one physical neural path",
                "the mapper discovers endogenous semantic primitives in a natural model",
                "observational equivalence proves physical mechanism identity",
            ],
        }
    }
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, claims)
    print(canonical_json({"status": status, "discovery": discovery["metrics"], "confirmation": confirmation["metrics"]}))


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    positive = summary["formal_decision"] == "positive"
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": summary["status"],
        "evidence": claims,
        "protocol_digest": protocol["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "formal_summary": summary,
        "authorized_next": {
            "theory_and_measurement_consolidation": True,
            "natural_language_trace_scan": False,
            "future_learning_rescue": False,
            "optimizer_state_rescue": False,
            "new_mechanism_algebra": False,
        },
        "scope": {
            "confirmed": "known-truth measurement contract only" if positive else "scope boundary only",
            "next_requirement": "freeze a natural pattern family, intervention registry, matched controls, and behavior gate before any language-model application",
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("develop", "preregister", "run-formal", "analyze", "finalize"))
    command = parser.parse_args().command
    {"develop": develop, "preregister": preregister, "run-formal": run_formal, "analyze": analyze, "finalize": finalize}[command]()


if __name__ == "__main__":
    main()
