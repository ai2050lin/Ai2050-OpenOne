"""Exclusion-only pilot for the Phase1173 relation-closure camera.

This file uses seeds and dimensions that are forbidden from the formal run.
It only checks whether the proposed measurement is numerically coherent before
the formal protocol is frozen.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5_temp/phase1173_relation_closure_camera_pilot.json"
P = 11
CONTEXTS = 10
RELATIONS = (1, 2, 3)
TRAIN_CONTEXTS = tuple(range(6))
TEST_CONTEXTS = tuple(range(6, 10))


def fourier(value: int, harmonics: int) -> np.ndarray:
    parts: list[float] = []
    for k in range(1, harmonics + 1):
        angle = 2.0 * math.pi * k * value / P
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


def make_states(morphology: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    core = np.stack([fourier(u, 4) for u in range(P)])
    context = rng.normal(size=(CONTEXTS, 2))
    context = context @ orthogonal(rng, 2)
    context /= np.std(context)
    context_gauges = [orthogonal(rng, 8) for _ in range(CONTEXTS)]
    gate_gauges = [orthogonal(rng, 8), orthogonal(rng, 8)]
    permutations = [rng.permutation(P) for _ in range(CONTEXTS)]
    redundant_left = orthogonal(rng, 4)
    redundant_right = orthogonal(rng, 4)

    rows = np.empty((CONTEXTS, P, 10), dtype=np.float64)
    for c in range(CONTEXTS):
        for u in range(P):
            if morphology == "shared_closed":
                relation = core[u]
            elif morphology == "instance_memory":
                relation = core[u] @ context_gauges[c]
            elif morphology == "surface_scramble":
                relation = core[int(permutations[c][u])]
            elif morphology == "sparse_exception":
                relation = core[int(permutations[c][u])] if c % 4 == 0 else core[u]
            elif morphology == "redundant_closed":
                phi = fourier(u, 2)
                relation = np.concatenate((phi @ redundant_left, phi @ redundant_right))
            elif morphology == "gated_closed":
                relation = core[u] @ gate_gauges[c % 2]
            else:
                raise ValueError(morphology)
            rows[c, u] = np.concatenate((relation, context[c]))
    return rows


def whiten(states: np.ndarray) -> np.ndarray:
    flat = states.reshape(-1, states.shape[-1])
    centered = flat - flat.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / centered.shape[0]
    values, vectors = np.linalg.eigh(covariance)
    keep = values > values.max() * 1.0e-10
    white = centered @ vectors[:, keep] @ np.diag(1.0 / np.sqrt(values[keep]))
    return white.reshape(states.shape[0], states.shape[1], -1)


def fit_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.concatenate((source, np.ones((len(source), 1))), axis=1)
    ridge = np.eye(design.shape[1]) * 1.0e-7
    ridge[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + ridge, design.T @ target)


def apply_map(source: np.ndarray, operator: np.ndarray) -> np.ndarray:
    design = np.concatenate((source, np.ones((len(source), 1))), axis=1)
    return design @ operator


def relation_key(context: int, relation: int, conditioned: bool) -> tuple[int, ...]:
    return (context % 2, relation) if conditioned else (relation,)


def camera(states: np.ndarray, conditioned: bool) -> dict[str, float]:
    z = whiten(states)
    operators: dict[tuple[int, ...], np.ndarray] = {}
    for c in TRAIN_CONTEXTS:
        for relation in RELATIONS:
            key = relation_key(c, relation, conditioned)
            if key in operators:
                continue
            contexts = [cc for cc in TRAIN_CONTEXTS if relation_key(cc, relation, conditioned) == key]
            source = np.concatenate([z[cc] for cc in contexts])
            target = np.concatenate([np.roll(z[cc], -relation, axis=0) for cc in contexts])
            operators[key] = fit_map(source, target)

    reuse_numer = 0.0
    reuse_denom = 0.0
    closure_numer = 0.0
    closure_denom = 0.0
    for c in TEST_CONTEXTS:
        all_targets = np.concatenate([np.roll(z[c], -r, axis=0) for r in RELATIONS])
        target_mean = all_targets.mean(axis=0, keepdims=True)
        for relation in RELATIONS:
            source = z[c]
            target = np.roll(z[c], -relation, axis=0)
            prediction = apply_map(source, operators[relation_key(c, relation, conditioned)])
            reuse_numer += float(np.sum((prediction - target) ** 2))
            reuse_denom += float(np.sum((target - target_mean) ** 2))

        direct = apply_map(z[c], operators[relation_key(c, 3, conditioned)])
        first = apply_map(z[c], operators[relation_key(c, 1, conditioned)])
        composed = apply_map(first, operators[relation_key(c, 2, conditioned)])
        actual = np.roll(z[c], -3, axis=0)
        closure_numer += float(np.sum((direct - composed) ** 2) + np.sum((composed - actual) ** 2))
        closure_denom += float(2.0 * np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2))

    reuse = float(np.clip(1.0 - reuse_numer / max(reuse_denom, 1.0e-12), 0.0, 1.0))
    closure = float(np.clip(1.0 - closure_numer / max(closure_denom, 1.0e-12), 0.0, 1.0))
    score = float(math.sqrt(reuse * closure))
    spectrum = np.linalg.eigvalsh(np.cov(states.reshape(-1, states.shape[-1]), rowvar=False))
    spectrum = np.sort(np.maximum(spectrum, 0.0))
    spectrum /= max(float(spectrum.sum()), 1.0e-12)
    return {
        "reuse": reuse,
        "closure": closure,
        "score": score,
        "spectrum_checksum": float(np.dot(spectrum, np.arange(1, len(spectrum) + 1))),
    }


def main() -> None:
    morphologies = (
        "shared_closed",
        "instance_memory",
        "surface_scramble",
        "sparse_exception",
        "redundant_closed",
        "gated_closed",
    )
    rows = []
    for replicate in range(4):
        seed = 117300 + replicate * 101
        gauge_rng = np.random.default_rng(seed + 77)
        for morphology in morphologies:
            states = make_states(morphology, seed)
            conditioned = camera(states, conditioned=True)
            unconditioned = camera(states, conditioned=False)
            gauge = invertible(gauge_rng, states.shape[-1])
            gauged = camera(states @ gauge, conditioned=True)
            rows.append(
                {
                    "replicate": replicate,
                    "morphology": morphology,
                    "conditioned": conditioned,
                    "unconditioned": unconditioned,
                    "gauge_max_abs": max(abs(conditioned[k] - gauged[k]) for k in ("reuse", "closure", "score")),
                }
            )
    summary = {}
    for morphology in morphologies:
        selected = [row for row in rows if row["morphology"] == morphology]
        summary[morphology] = {
            key: float(np.mean([row["conditioned"][key] for row in selected]))
            for key in ("reuse", "closure", "score", "spectrum_checksum")
        }
        summary[morphology]["unconditioned_score"] = float(
            np.mean([row["unconditioned"]["score"] for row in selected])
        )
    payload = {
        "scope": "exclusion_only_non_evidential_pilot",
        "summary": summary,
        "max_gauge_error": max(row["gauge_max_abs"] for row in rows),
        "shared_surface_spectrum_gap": abs(
            summary["shared_closed"]["spectrum_checksum"] - summary["surface_scramble"]["spectrum_checksum"]
        ),
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"max_gauge_error={payload['max_gauge_error']:.3e}")
    print(f"shared_surface_spectrum_gap={payload['shared_surface_spectrum_gap']:.3e}")


if __name__ == "__main__":
    main()
