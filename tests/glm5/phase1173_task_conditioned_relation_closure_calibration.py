"""Phase1173: known-truth calibration of a task-conditioned relation camera.

The camera fits relation-specific affine transports on training backgrounds and
tests them on held-out backgrounds. It then asks whether a direct relation and
the composition of two shorter relations agree with the held-out target. The
formal split contains matched systems with reusable relations, instance memory,
surface-pairing scrambles, sparse exceptions, redundant routes, and contextual
gates. No Phase1172 confirmation task or holdout value is consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE = 1173
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1173_task_conditioned_relation_closure_calibration"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
SOURCE_FINAL = ROOT / "tests/glm5/result/phase1172_cross_quotient_event_time_prediction/analysis/final.json"
SCRIPT_PATH = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1173_task_conditioned_relation_closure_calibration_audit.py"

MORPHOLOGIES = (
    "shared_closed",
    "instance_memory",
    "surface_scramble",
    "sparse_exception",
    "redundant_closed",
    "gated_closed",
)
HIGH_MORPHOLOGIES = ("shared_closed", "redundant_closed", "gated_closed")
LOW_MORPHOLOGIES = ("instance_memory", "surface_scramble")
RELATIONS = (1, 2, 3)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
NOISE_SD = 0.015
RIDGE = 1.0e-7
WHITEN_RELATIVE_EIGEN_FLOOR = 1.0e-10
GAUGE_COUNT = 2
REPLICATES = 32

THRESHOLDS = {
    "high_median_min": 0.90,
    "high_replicate_min": 0.75,
    "low_median_max": 0.20,
    "low_replicate_max": 0.45,
    "sparse_median_min": 0.30,
    "sparse_median_max": 0.80,
    "shared_sparse_gap_min": 0.15,
    "gated_conditioning_gain_min": 0.20,
    "matched_spectrum_gap_max": 1.0e-10,
    "matched_camera_gap_min": 0.70,
    "gauge_error_max": 1.0e-5,
    "trajectory_start_max": 0.20,
    "trajectory_end_min": 0.90,
    "trajectory_range_min": 0.65,
    "trajectory_step_drop_max": 0.03,
}


@dataclass(frozen=True)
class SplitConfig:
    name: str
    modulus: int
    contexts: int
    train_contexts: tuple[int, ...]
    test_contexts: tuple[int, ...]
    seed_base: int


SPLITS = {
    "discovery": SplitConfig(
        name="discovery",
        modulus=17,
        contexts=16,
        train_contexts=tuple(range(12)),
        test_contexts=tuple(range(12, 16)),
        seed_base=1_173_500_000,
    ),
    "confirmation": SplitConfig(
        name="confirmation",
        modulus=19,
        contexts=20,
        train_contexts=tuple(range(16)),
        test_contexts=tuple(range(16, 20)),
        seed_base=1_174_300_000,
    ),
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seed_for(config: SplitConfig, replicate: int) -> int:
    return config.seed_base + replicate * 10_007


def fourier(value: int, harmonics: int, modulus: int) -> np.ndarray:
    parts: list[float] = []
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * math.pi * harmonic * value / modulus
        parts.extend((math.cos(angle), math.sin(angle)))
    return np.asarray(parts, dtype=np.float64)


def orthogonal(rng: np.random.Generator, size: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(size, size)))
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    return q * signs


def invertible(rng: np.random.Generator, size: int) -> np.ndarray:
    left = orthogonal(rng, size)
    right = orthogonal(rng, size)
    singular = np.geomspace(0.5, 2.0, size)
    return left @ np.diag(singular) @ right.T


@dataclass
class Components:
    base_relation: np.ndarray
    context: np.ndarray
    context_gauges: list[np.ndarray]
    gate_gauges: list[np.ndarray]
    permutations: list[np.ndarray]
    redundant_left: np.ndarray
    redundant_right: np.ndarray
    redundant_noise: np.ndarray


def make_components(config: SplitConfig, seed: int) -> Components:
    rng = np.random.default_rng(seed)
    core = np.stack([fourier(value, 4, config.modulus) for value in range(config.modulus)])
    noise = rng.normal(scale=NOISE_SD, size=(config.contexts, config.modulus, 8))
    base_relation = core[None, :, :] + noise
    context = rng.normal(size=(config.contexts, 3))
    context = context @ orthogonal(rng, 3)
    context /= max(float(np.std(context)), 1.0e-12)
    context_gauges = [orthogonal(rng, 8) for _ in range(config.contexts)]
    gate_gauges = [orthogonal(rng, 8), orthogonal(rng, 8)]
    permutations = [rng.permutation(config.modulus) for _ in range(config.contexts)]
    redundant_left = orthogonal(rng, 4)
    redundant_right = orthogonal(rng, 4)
    redundant_noise = np.zeros((config.contexts, config.modulus, 8), dtype=np.float64)
    return Components(
        base_relation=base_relation,
        context=context,
        context_gauges=context_gauges,
        gate_gauges=gate_gauges,
        permutations=permutations,
        redundant_left=redundant_left,
        redundant_right=redundant_right,
        redundant_noise=redundant_noise,
    )


def make_states(config: SplitConfig, components: Components, morphology: str) -> np.ndarray:
    rows = np.empty((config.contexts, config.modulus, 11), dtype=np.float64)
    for context_index in range(config.contexts):
        for value in range(config.modulus):
            base = components.base_relation[context_index, value]
            if morphology == "shared_closed":
                relation = base
            elif morphology == "instance_memory":
                relation = base @ components.context_gauges[context_index]
            elif morphology == "surface_scramble":
                assigned = int(components.permutations[context_index][value])
                relation = components.base_relation[context_index, assigned]
            elif morphology == "sparse_exception":
                if context_index % 4 == 0:
                    assigned = int(components.permutations[context_index][value])
                    relation = components.base_relation[context_index, assigned]
                else:
                    relation = base
            elif morphology == "redundant_closed":
                phi = fourier(value, 2, config.modulus)
                relation = np.concatenate(
                    (phi @ components.redundant_left, phi @ components.redundant_right)
                ) + components.redundant_noise[context_index, value]
            elif morphology == "gated_closed":
                relation = base @ components.gate_gauges[context_index % 2]
            else:
                raise ValueError(f"Unknown morphology: {morphology}")
            rows[context_index, value] = np.concatenate((relation, components.context[context_index]))
    return rows


def make_blend_states(config: SplitConfig, components: Components, alpha: float) -> np.ndarray:
    rows = np.empty((config.contexts, config.modulus, 11), dtype=np.float64)
    for context_index in range(config.contexts):
        memory = components.base_relation[context_index] @ components.context_gauges[context_index]
        relation = alpha * components.base_relation[context_index] + (1.0 - alpha) * memory
        for value in range(config.modulus):
            rows[context_index, value] = np.concatenate((relation[value], components.context[context_index]))
    return rows


def whiten(states: np.ndarray) -> np.ndarray:
    flat = states.reshape(-1, states.shape[-1])
    centered = flat - flat.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / centered.shape[0]
    values, vectors = np.linalg.eigh(covariance)
    keep = values > max(float(values.max()), 1.0e-12) * WHITEN_RELATIVE_EIGEN_FLOOR
    if int(np.sum(keep)) < 2:
        raise RuntimeError("Whitening rank collapsed")
    transform = vectors[:, keep] @ np.diag(1.0 / np.sqrt(values[keep]))
    white = centered @ transform
    return white.reshape(states.shape[0], states.shape[1], -1)


def fit_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.concatenate((source, np.ones((len(source), 1))), axis=1)
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def apply_map(source: np.ndarray, operator: np.ndarray) -> np.ndarray:
    design = np.concatenate((source, np.ones((len(source), 1))), axis=1)
    return design @ operator


def relation_key(context_index: int, relation: int, conditioned: bool) -> tuple[int, ...]:
    return (context_index % 2, relation) if conditioned else (relation,)


def state_spectrum(states: np.ndarray) -> list[float]:
    flat = states.reshape(-1, states.shape[-1])
    covariance = np.cov(flat, rowvar=False)
    values = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    values /= max(float(values.sum()), 1.0e-12)
    return [float(value) for value in values]


def relation_camera(states: np.ndarray, config: SplitConfig, conditioned: bool) -> dict[str, float]:
    z = whiten(states)
    operators: dict[tuple[int, ...], np.ndarray] = {}
    for context_index in config.train_contexts:
        for relation in RELATIONS:
            key = relation_key(context_index, relation, conditioned)
            if key in operators:
                continue
            matching = [
                other
                for other in config.train_contexts
                if relation_key(other, relation, conditioned) == key
            ]
            source = np.concatenate([z[other] for other in matching])
            target = np.concatenate(
                [np.roll(z[other], -relation, axis=0) for other in matching]
            )
            operators[key] = fit_map(source, target)

    reuse_numerator = 0.0
    reuse_denominator = 0.0
    closure_numerator = 0.0
    closure_denominator = 0.0
    for context_index in config.test_contexts:
        all_targets = np.concatenate(
            [np.roll(z[context_index], -relation, axis=0) for relation in RELATIONS]
        )
        target_mean = all_targets.mean(axis=0, keepdims=True)
        for relation in RELATIONS:
            source = z[context_index]
            target = np.roll(z[context_index], -relation, axis=0)
            prediction = apply_map(
                source,
                operators[relation_key(context_index, relation, conditioned)],
            )
            reuse_numerator += float(np.sum((prediction - target) ** 2))
            reuse_denominator += float(np.sum((target - target_mean) ** 2))

        direct = apply_map(
            z[context_index],
            operators[relation_key(context_index, 3, conditioned)],
        )
        first = apply_map(
            z[context_index],
            operators[relation_key(context_index, 1, conditioned)],
        )
        composed = apply_map(
            first,
            operators[relation_key(context_index, 2, conditioned)],
        )
        actual = np.roll(z[context_index], -3, axis=0)
        closure_numerator += float(
            np.sum((direct - composed) ** 2) + np.sum((composed - actual) ** 2)
        )
        closure_denominator += float(
            2.0 * np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2)
        )

    reuse_error = reuse_numerator / max(reuse_denominator, 1.0e-12)
    closure_error = closure_numerator / max(closure_denominator, 1.0e-12)
    reuse_score = float(np.clip(1.0 - reuse_error, 0.0, 1.0))
    closure_score = float(np.clip(1.0 - closure_error, 0.0, 1.0))
    score = float(math.sqrt(reuse_score * closure_score))
    return {
        "reuse_error": float(reuse_error),
        "closure_error": float(closure_error),
        "reuse_score": reuse_score,
        "closure_score": closure_score,
        "score": score,
    }


def label_only_digest(config: SplitConfig) -> str:
    payload = {
        "modulus": config.modulus,
        "relations": list(RELATIONS),
        "train_relation_keys": [
            [context_index % 2, relation]
            for context_index in config.train_contexts
            for relation in RELATIONS
        ],
        "test_relation_keys": [
            [context_index % 2, relation]
            for context_index in config.test_contexts
            for relation in RELATIONS
        ],
    }
    return digest(payload)


def system_row(config: SplitConfig, replicate: int, morphology: str) -> dict[str, Any]:
    seed = seed_for(config, replicate)
    components = make_components(config, seed)
    states = make_states(config, components, morphology)
    conditioned = relation_camera(states, config, conditioned=True)
    unconditioned = relation_camera(states, config, conditioned=False)
    gauges = []
    for gauge_index in range(GAUGE_COUNT):
        rng = np.random.default_rng(seed + 71_000 + gauge_index * 997)
        transformed = states @ invertible(rng, states.shape[-1])
        measured = relation_camera(transformed, config, conditioned=True)
        gauges.append(
            {
                "gauge_index": gauge_index,
                "camera": measured,
                "max_abs_error": max(
                    abs(conditioned[key] - measured[key])
                    for key in ("reuse_score", "closure_score", "score")
                ),
            }
        )
    return {
        "system_id": f"{config.name}_r{replicate:02d}_{morphology}",
        "split": config.name,
        "replicate": replicate,
        "seed": seed,
        "morphology": morphology,
        "conditioned_camera": conditioned,
        "unconditioned_camera": unconditioned,
        "state_spectrum": state_spectrum(states),
        "label_only_digest": label_only_digest(config),
        "gauges": gauges,
        "state_digest": digest(np.round(states, 10).tolist()),
    }


def trajectory_row(config: SplitConfig, replicate: int, alpha: float) -> dict[str, Any]:
    seed = seed_for(config, replicate)
    components = make_components(config, seed)
    states = make_blend_states(config, components, alpha)
    return {
        "trajectory_id": f"{config.name}_r{replicate:02d}_a{alpha:.2f}",
        "split": config.name,
        "replicate": replicate,
        "alpha": alpha,
        "camera": relation_camera(states, config, conditioned=True),
        "state_digest": digest(np.round(states, 10).tolist()),
    }


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize_rows(
    config: SplitConfig,
    systems: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    expected_replicates: int = REPLICATES,
) -> dict[str, Any]:
    by_morphology: dict[str, Any] = {}
    for morphology in MORPHOLOGIES:
        selected = [row for row in systems if row["morphology"] == morphology]
        scores = [float(row["conditioned_camera"]["score"]) for row in selected]
        unconditioned = [float(row["unconditioned_camera"]["score"]) for row in selected]
        by_morphology[morphology] = {
            "count": len(selected),
            "score_min": min(scores),
            "score_median": median(scores),
            "score_max": max(scores),
            "unconditioned_score_median": median(unconditioned),
        }

    spectrum_gaps = []
    camera_gaps = []
    for replicate in range(expected_replicates):
        shared = next(
            row
            for row in systems
            if row["replicate"] == replicate and row["morphology"] == "shared_closed"
        )
        scrambled = next(
            row
            for row in systems
            if row["replicate"] == replicate and row["morphology"] == "surface_scramble"
        )
        spectrum_gaps.append(
            max(
                abs(float(left) - float(right))
                for left, right in zip(shared["state_spectrum"], scrambled["state_spectrum"])
            )
        )
        camera_gaps.append(
            float(shared["conditioned_camera"]["score"])
            - float(scrambled["conditioned_camera"]["score"])
        )

    alpha_medians: dict[str, float] = {}
    for alpha in ALPHAS:
        scores = [
            float(row["camera"]["score"])
            for row in trajectories
            if abs(float(row["alpha"]) - alpha) < 1.0e-12
        ]
        alpha_medians[f"{alpha:.2f}"] = median(scores)
    ordered = [alpha_medians[f"{alpha:.2f}"] for alpha in ALPHAS]
    step_drops = [max(0.0, ordered[index] - ordered[index + 1]) for index in range(len(ordered) - 1)]

    gauge_error = max(
        float(gauge["max_abs_error"])
        for row in systems
        for gauge in row["gauges"]
    )
    gated = by_morphology["gated_closed"]
    sparse = by_morphology["sparse_exception"]
    shared = by_morphology["shared_closed"]
    checks: dict[str, bool] = {
        "row_count": len(systems) == expected_replicates * len(MORPHOLOGIES),
        "trajectory_count": len(trajectories) == expected_replicates * len(ALPHAS),
        "label_only_identical": len({row["label_only_digest"] for row in systems}) == 1,
        "gauge_invariant": gauge_error <= THRESHOLDS["gauge_error_max"],
        "high_medians": all(
            by_morphology[name]["score_median"] >= THRESHOLDS["high_median_min"]
            for name in HIGH_MORPHOLOGIES
        ),
        "high_replicate_floor": all(
            by_morphology[name]["score_min"] >= THRESHOLDS["high_replicate_min"]
            for name in HIGH_MORPHOLOGIES
        ),
        "low_medians": all(
            by_morphology[name]["score_median"] <= THRESHOLDS["low_median_max"]
            for name in LOW_MORPHOLOGIES
        ),
        "low_replicate_ceiling": all(
            by_morphology[name]["score_max"] <= THRESHOLDS["low_replicate_max"]
            for name in LOW_MORPHOLOGIES
        ),
        "sparse_intermediate": (
            THRESHOLDS["sparse_median_min"]
            <= sparse["score_median"]
            <= THRESHOLDS["sparse_median_max"]
        ),
        "shared_sparse_separated": (
            shared["score_median"] - sparse["score_median"]
            >= THRESHOLDS["shared_sparse_gap_min"]
        ),
        "gated_conditioning_gain": (
            gated["score_median"] - gated["unconditioned_score_median"]
            >= THRESHOLDS["gated_conditioning_gain_min"]
        ),
        "matched_spectrum": max(spectrum_gaps) <= THRESHOLDS["matched_spectrum_gap_max"],
        "matched_camera_separation": median(camera_gaps) >= THRESHOLDS["matched_camera_gap_min"],
        "trajectory_start": ordered[0] <= THRESHOLDS["trajectory_start_max"],
        "trajectory_end": ordered[-1] >= THRESHOLDS["trajectory_end_min"],
        "trajectory_range": ordered[-1] - ordered[0] >= THRESHOLDS["trajectory_range_min"],
        "trajectory_monotone": max(step_drops, default=0.0)
        <= THRESHOLDS["trajectory_step_drop_max"],
    }
    return {
        "phase": PHASE,
        "split": config.name,
        "config": {
            "modulus": config.modulus,
            "contexts": config.contexts,
            "train_contexts": list(config.train_contexts),
            "test_contexts": list(config.test_contexts),
            "replicates": expected_replicates,
        },
        "by_morphology": by_morphology,
        "matched_controls": {
            "max_spectrum_gap": max(spectrum_gaps),
            "median_camera_gap": median(camera_gaps),
        },
        "gauge_max_abs_error": gauge_error,
        "trajectory_score_medians": alpha_medians,
        "trajectory_max_step_drop": max(step_drops, default=0.0),
        "checks": checks,
        "passed": all(checks.values()),
    }


def protocol_payload() -> dict[str, Any]:
    if not AUDIT_SCRIPT.exists():
        raise RuntimeError("Audit script must exist before protocol freeze")
    source_sha = sha256_file(SOURCE_FINAL)
    payload = {
        "phase": PHASE,
        "title": "task-conditioned relation transport and closure known-truth calibration",
        "created_date": "2026-08-07",
        "source_phase1172_final_sha256": source_sha,
        "scope": (
            "Calibrate one predefined representation camera. Passing establishes that the camera can identify "
            "explicit reusable affine relation transports and composition closure under the declared systems; "
            "it does not establish causal use, natural-network presence, language semantics, or event-time prediction."
        ),
        "splits": {
            name: {
                "modulus": config.modulus,
                "contexts": config.contexts,
                "train_contexts": list(config.train_contexts),
                "test_contexts": list(config.test_contexts),
                "seed_base": config.seed_base,
            }
            for name, config in SPLITS.items()
        },
        "replicates": REPLICATES,
        "morphologies": list(MORPHOLOGIES),
        "relations": list(RELATIONS),
        "formation_alphas": list(ALPHAS),
        "noise_sd": NOISE_SD,
        "ridge": RIDGE,
        "whiten_relative_eigen_floor": WHITEN_RELATIVE_EIGEN_FLOOR,
        "gauge_count": GAUGE_COUNT,
        "thresholds": THRESHOLDS,
        "primary_endpoint": (
            "Discovery and confirmation must independently pass every frozen calibration check: high/low known-truth "
            "separation, sparse-exception ordering, gated conditioning gain, exact state-spectrum matched control, "
            "global invertible-gauge stability, and monotone controlled formation response."
        ),
        "analytic_scope_controls": {
            "high_closure_without_use": (
                "The shared_closed state can be paired with a bypass output head without changing the camera."
            ),
            "perfect_behavior_without_closure": (
                "The instance_memory state can be paired with an exact lookup output without changing the camera."
            ),
            "interpretation": (
                "Therefore the endpoint is representation-level reusable relation closure, not behavioral accuracy or causal use."
            ),
        },
        "staged_reveal": [
            "Freeze both scripts and all thresholds before creating formal outputs.",
            "Generate and score discovery first.",
            "A discovery failure forbids confirmation generation.",
            "Only an independently passing confirmation authorizes a future all-new task-class prediction protocol.",
        ],
        "forbidden": [
            "No Phase1172 confirmation row, score, task, feature, or holdout value enters the camera.",
            "No morphology, threshold, relation key, gauge family, or trajectory alpha changes after protocol freeze.",
            "No failed morphology may be deleted or relabeled.",
            "No pretrained-model scan, hidden-state claim, or causal claim follows from known-truth calibration alone.",
            "No alternative camera is opened in this registry.",
        ],
        "script_sha256": sha256_file(SCRIPT_PATH),
        "audit_script_sha256": sha256_file(AUDIT_SCRIPT),
    }
    payload["protocol_digest"] = digest(payload)
    return payload


def protocol_command() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    write_json(PROTOCOL_PATH, protocol_payload())
    print(f"protocol={PROTOCOL_PATH}")


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    if protocol["script_sha256"] != sha256_file(SCRIPT_PATH):
        raise RuntimeError("Main script changed after protocol freeze")
    if protocol["audit_script_sha256"] != sha256_file(AUDIT_SCRIPT):
        raise RuntimeError("Audit script changed after protocol freeze")
    payload = dict(protocol)
    expected_digest = payload.pop("protocol_digest")
    if digest(payload) != expected_digest:
        raise RuntimeError("Protocol digest mismatch")
    return protocol


def run_command(split: str) -> None:
    verify_protocol()
    config = SPLITS[split]
    if split == "confirmation":
        discovery_gate = read_json(OUT_ROOT / "analysis/discovery_gate.json")
        if not discovery_gate["passed"]:
            raise RuntimeError("Discovery failed; confirmation is forbidden")
    systems = [
        system_row(config, replicate, morphology)
        for replicate in range(REPLICATES)
        for morphology in MORPHOLOGIES
    ]
    trajectories = [
        trajectory_row(config, replicate, alpha)
        for replicate in range(REPLICATES)
        for alpha in ALPHAS
    ]
    run_root = OUT_ROOT / f"runs/{split}"
    public_systems = [
        {key: value for key, value in row.items() if key != "morphology"}
        for row in systems
    ]
    truth = [
        {
            "system_id": row["system_id"],
            "morphology": row["morphology"],
            "representation_use_scope": (
                "camera_unchanged_under_output_bypass_or_exact_lookup"
            ),
        }
        for row in systems
    ]
    write_jsonl(run_root / "public_systems.jsonl", public_systems)
    write_jsonl(run_root / "sealed_truth.jsonl", truth)
    write_jsonl(run_root / "formation_trajectory.jsonl", trajectories)
    summary = summarize_rows(config, systems, trajectories)
    summary["public_systems_sha256"] = sha256_file(run_root / "public_systems.jsonl")
    summary["sealed_truth_sha256"] = sha256_file(run_root / "sealed_truth.jsonl")
    summary["formation_trajectory_sha256"] = sha256_file(run_root / "formation_trajectory.jsonl")
    write_json(run_root / "summary.json", summary)
    print(canonical({"split": split, "passed": summary["passed"], "checks": summary["checks"]}))


def score_command(split: str) -> None:
    protocol = verify_protocol()
    run_root = OUT_ROOT / f"runs/{split}"
    public = read_jsonl(run_root / "public_systems.jsonl")
    truth_rows = read_jsonl(run_root / "sealed_truth.jsonl")
    truth = {row["system_id"]: row["morphology"] for row in truth_rows}
    systems = [dict(row, morphology=truth[row["system_id"]]) for row in public]
    trajectories = read_jsonl(run_root / "formation_trajectory.jsonl")
    summary = summarize_rows(SPLITS[split], systems, trajectories)
    summary["protocol_digest"] = protocol["protocol_digest"]
    summary["run_summary_sha256"] = sha256_file(run_root / "summary.json")
    target = OUT_ROOT / f"analysis/{'discovery_gate' if split == 'discovery' else 'confirmation_score'}.json"
    write_json(target, summary)
    print(canonical({"score": str(target), "passed": summary["passed"]}))


def finalize_command() -> None:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "analysis/discovery_gate.json")
    if not discovery["passed"]:
        final = {
            "phase": PHASE,
            "protocol_digest": protocol["protocol_digest"],
            "discovery_passed": False,
            "confirmation_tested": False,
            "relation_closure_camera_calibrated": False,
            "outcome": "discovery_calibration_failed",
            "auto_continue": False,
        }
    else:
        confirmation = read_json(OUT_ROOT / "analysis/confirmation_score.json")
        passed = bool(confirmation["passed"])
        final = {
            "phase": PHASE,
            "protocol_digest": protocol["protocol_digest"],
            "discovery_passed": True,
            "confirmation_tested": True,
            "confirmation_passed": passed,
            "relation_closure_camera_calibrated": passed,
            "outcome": (
                "known_truth_relation_closure_camera_confirmed"
                if passed
                else "known_truth_relation_closure_camera_not_confirmed"
            ),
            "evidence_scope": (
                "Known-truth representation camera only. It identifies held-out-background affine relation reuse and "
                "composition closure under the frozen synthetic systems. It does not identify causal use, behavior, "
                "natural-network mechanisms, or future event times."
            ),
            "analytic_non_implications": [
                "A high score does not imply that an output uses the representation.",
                "A low score does not imply poor behavior because exact instance lookup may bypass shared relations.",
                "Calibration does not imply the same affine transport exists in a freely trained network.",
            ],
            "next_step": (
                "If passed, design one all-new task-class prospective event-time protocol with this camera fixed; "
                "do not reuse Phase1172 confirmation or search another camera."
                if passed
                else "Close this camera registry without replacement inside Phase1173."
            ),
            "auto_continue": False,
            "auto_continue_reason": (
                "A positive known-truth calibration licenses protocol design, not a mechanical transfer. All-new task "
                "classes and a leakage audit must be frozen before any prospective run."
                if passed
                else "The frozen known-truth endpoint failed."
            ),
            "discovery_gate_sha256": sha256_file(OUT_ROOT / "analysis/discovery_gate.json"),
            "confirmation_score_sha256": sha256_file(OUT_ROOT / "analysis/confirmation_score.json"),
        }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def smoke_command() -> None:
    config = SplitConfig(
        name="excluded_smoke",
        modulus=11,
        contexts=10,
        train_contexts=tuple(range(6)),
        test_contexts=tuple(range(6, 10)),
        seed_base=117_300,
    )
    systems = [
        system_row(config, replicate, morphology)
        for replicate in range(4)
        for morphology in MORPHOLOGIES
    ]
    trajectories = [
        trajectory_row(config, replicate, alpha)
        for replicate in range(4)
        for alpha in ALPHAS
    ]
    print(json.dumps(summarize_rows(config, systems, trajectories, expected_replicates=4), indent=2, sort_keys=True))


def all_command() -> None:
    protocol_command()
    run_command("discovery")
    score_command("discovery")
    discovery = read_json(OUT_ROOT / "analysis/discovery_gate.json")
    if not discovery["passed"]:
        finalize_command()
        return
    run_command("confirmation")
    score_command("confirmation")
    finalize_command()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("protocol", "run", "score", "finalize", "smoke", "all"),
    )
    parser.add_argument("--split", choices=tuple(SPLITS))
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        if args.split is None:
            parser.error("run requires --split")
        run_command(args.split)
    elif args.command == "score":
        if args.split is None:
            parser.error("score requires --split")
        score_command(args.split)
    elif args.command == "finalize":
        finalize_command()
    elif args.command == "smoke":
        smoke_command()
    else:
        all_command()


if __name__ == "__main__":
    main()
