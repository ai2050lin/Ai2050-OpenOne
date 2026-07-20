#!/usr/bin/env python3
"""Run the frozen 1,024-pair maximum-block null audit for Phase571."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase571_relation_block"
MODELS = ("qwen3", "glm4", "deepseek7b")
POOLS = ("block_discovery", "block_confirmation")
ROUNDS = 1024
SEED = 5711024
PROTOCOL_PATH = OUT_DIR / "phase571_frozen_protocol.json"
ANALYSIS_PATH = OUT_DIR / "phase571_continuous_block_analysis.json"
OUTPUT_PATH = OUT_DIR / "phase571_max_block_permutation_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def depth_band(layer: int, layer_count: int, band_count: int) -> int:
    return min(band_count - 1, (layer * band_count) // layer_count)


def candidate_arrays(
    model: str,
    pool: str,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    matched = read_json(OUT_DIR / f"phase571_{model}_matched_behavior_summary.json")
    correct_ids = matched["selected_case_ids_by_pool_phenotype"][pool]["stable_correct"]
    confusion_ids = matched["selected_case_ids_by_pool_phenotype"][pool][
        "stable_relation_confusion"
    ]
    if len(correct_ids) != 128 or len(confusion_ids) != 128:
        raise RuntimeError(f"Phase571 {model} {pool} matched denominator drift")
    rows_path = OUT_DIR / f"phase571_{model}_signed_write_rows.jsonl.gz"
    summary = read_json(OUT_DIR / f"phase571_{model}_signed_write_summary.json")
    if summary["rows_sha256"] != sha256_file(rows_path):
        raise RuntimeError(f"Phase571 {model} trace hash drift")
    bank: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(rows_path):
        if row["pool"] == pool:
            bank[(row["case_id"], row["semantic_role"])] = row
    layer_count = int(summary["layer_count"])
    band_count = int(protocol["depth_band_count"])
    max_width = int(
        protocol["block_discovery_rule"]["maximum_interval_width_in_bands"]
    )
    candidates: list[dict[str, Any]] = []
    for role in protocol["causal_role_priority"]:
        for start_band in range(band_count):
            for end_band in range(start_band, min(band_count, start_band + max_width)):
                layers = [
                    layer
                    for layer in range(layer_count)
                    if start_band <= depth_band(layer, layer_count, band_count) <= end_band
                ]
                correct = []
                confusion = []
                for correct_id, confusion_id in zip(correct_ids, confusion_ids):
                    left = bank[(correct_id, role)]
                    right = bank[(confusion_id, role)]
                    correct.append(
                        sum(
                            left["attention_signed_unit_projection"][layer]
                            + left["mlp_signed_unit_projection"][layer]
                            for layer in layers
                        )
                    )
                    confusion.append(
                        sum(
                            right["attention_signed_unit_projection"][layer]
                            + right["mlp_signed_unit_projection"][layer]
                            for layer in layers
                        )
                    )
                left = np.asarray(correct, dtype=np.float64)
                right = np.asarray(confusion, dtype=np.float64)
                candidates.append(
                    {
                        "semantic_role": role,
                        "start_band": start_band,
                        "end_band": end_band,
                        "band_width": end_band - start_band + 1,
                        "difference": left - right,
                        "positive_difference": (left > 0.0).astype(np.float64)
                        - (right > 0.0).astype(np.float64),
                        "scale": float(np.mean(np.abs(np.concatenate((left, right))))),
                    }
                )
    return candidates, layer_count


def score(
    mean_difference: np.ndarray | float,
    mean_positive_difference: np.ndarray | float,
    scale: float,
    relative_threshold: float,
    positive_threshold: float,
) -> np.ndarray | float:
    relative = (
        np.abs(np.asarray(mean_difference) / np.maximum(scale, 1e-12))
        / relative_threshold
    )
    positive = np.abs(mean_positive_difference) / positive_threshold
    return np.minimum(relative, positive)


def selected_key(selected: dict[str, Any]) -> tuple[str, int, int]:
    return (
        selected["semantic_role"],
        int(selected["start_band"]),
        int(selected["end_band"]),
    )


def audit_model(
    model: str,
    protocol: dict[str, Any],
    observer_report: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    rule = protocol["block_discovery_rule"]
    relative_threshold = float(rule["minimum_absolute_relative_gap_each_split"])
    positive_threshold = float(
        rule["minimum_absolute_positive_rate_gap_each_split"]
    )
    arrays = {}
    for pool in POOLS:
        arrays[pool], _layer_count = candidate_arrays(model, pool, protocol)
    keys = [
        (row["semantic_role"], row["start_band"], row["end_band"])
        for row in arrays["block_discovery"]
    ]
    if keys != [
        (row["semantic_role"], row["start_band"], row["end_band"])
        for row in arrays["block_confirmation"]
    ]:
        raise RuntimeError(f"Phase571 {model} pool candidate drift")

    discovery_difference = np.stack(
        [row["difference"] for row in arrays["block_discovery"]]
    )
    discovery_positive = np.stack(
        [row["positive_difference"] for row in arrays["block_discovery"]]
    )
    confirmation_difference = np.stack(
        [row["difference"] for row in arrays["block_confirmation"]]
    )
    confirmation_positive = np.stack(
        [row["positive_difference"] for row in arrays["block_confirmation"]]
    )
    discovery_scales = np.asarray(
        [row["scale"] for row in arrays["block_discovery"]], dtype=np.float64
    )
    confirmation_scales = np.asarray(
        [row["scale"] for row in arrays["block_confirmation"]], dtype=np.float64
    )

    observed_discovery_scores = score(
        np.mean(discovery_difference, axis=1),
        np.mean(discovery_positive, axis=1),
        discovery_scales,
        relative_threshold,
        positive_threshold,
    )
    observed_confirmation_scores = score(
        np.mean(confirmation_difference, axis=1),
        np.mean(confirmation_positive, axis=1),
        confirmation_scales,
        relative_threshold,
        positive_threshold,
    )
    discovery_signs = rng.choice((-1.0, 1.0), size=(ROUNDS, 128))
    confirmation_signs = rng.choice((-1.0, 1.0), size=(ROUNDS, 128))
    perm_discovery_means = discovery_difference @ discovery_signs.T / 128.0
    perm_discovery_positive = discovery_positive @ discovery_signs.T / 128.0
    perm_confirmation_means = confirmation_difference @ confirmation_signs.T / 128.0
    perm_confirmation_positive = confirmation_positive @ confirmation_signs.T / 128.0
    perm_discovery_scores = score(
        perm_discovery_means,
        perm_discovery_positive,
        discovery_scales[:, None],
        relative_threshold,
        positive_threshold,
    )
    perm_confirmation_scores = score(
        perm_confirmation_means,
        perm_confirmation_positive,
        confirmation_scales[:, None],
        relative_threshold,
        positive_threshold,
    )
    null_max_discovery = np.max(perm_discovery_scores, axis=0)

    selected = observer_report["selected_continuous_block"]
    if selected is None:
        return {
            "model": model,
            "selected_block": None,
            "candidate_count": len(keys),
            "permutation_rounds": ROUNDS,
            "observed_max_discovery_score": float(np.max(observed_discovery_scores)),
            "null_max_discovery_score_mean": float(np.mean(null_max_discovery)),
            "null_any_discovery_threshold_pass_rate": float(
                np.mean(null_max_discovery >= 1.0)
            ),
            "selected_block_familywise_discovery_p": None,
            "selected_block_confirmation_p": None,
            "permutation_gate_pass": False,
            "reason": "No discovery-and-confirmation block was selected before permutation.",
        }

    index = keys.index(selected_key(selected))
    observed_selected_discovery = float(observed_discovery_scores[index])
    observed_selected_confirmation = float(observed_confirmation_scores[index])
    discovery_p = float(
        (1 + np.sum(null_max_discovery >= observed_selected_discovery))
        / (ROUNDS + 1)
    )
    confirmation_p = float(
        (
            1
            + np.sum(
                perm_confirmation_scores[index] >= observed_selected_confirmation
            )
        )
        / (ROUNDS + 1)
    )
    permutation_pass = discovery_p <= 0.05 and confirmation_p <= 0.05
    return {
        "model": model,
        "selected_block": {
            "semantic_role": selected["semantic_role"],
            "start_band": selected["start_band"],
            "end_band": selected["end_band"],
        },
        "candidate_count": len(keys),
        "permutation_rounds": ROUNDS,
        "observed_selected_discovery_score": observed_selected_discovery,
        "observed_selected_confirmation_score": observed_selected_confirmation,
        "observed_max_discovery_score": float(np.max(observed_discovery_scores)),
        "null_max_discovery_score_mean": float(np.mean(null_max_discovery)),
        "null_max_discovery_score_q95": float(np.quantile(null_max_discovery, 0.95)),
        "null_any_discovery_threshold_pass_rate": float(
            np.mean(null_max_discovery >= 1.0)
        ),
        "selected_block_familywise_discovery_p": discovery_p,
        "selected_block_confirmation_p": confirmation_p,
        "permutation_gate_pass": permutation_pass,
    }


def audit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    observer = read_json(ANALYSIS_PATH)
    reports_by_model = {row["model"]: row for row in observer["model_reports"]}
    reports = []
    for model_index, model in enumerate(MODELS):
        rng = np.random.default_rng(SEED + model_index * 100003)
        reports.append(audit_model(model, protocol, reports_by_model[model], rng))
    passed = [row["model"] for row in reports if row["permutation_gate_pass"]]
    output = {
        "schema_version": "phase571_max_block_permutation_audit.v1",
        "phase_id": "Phase571",
        "created_at": now(),
        "status": "complete",
        "method": (
            "Exact-pair phenotype-label sign flips. Discovery uses the maximum score "
            "across every frozen role and contiguous depth interval; confirmation tests "
            "the discovery-selected block on the independent pool."
        ),
        "rounds": ROUNDS,
        "seed": SEED,
        "score": (
            "min(abs(relative_mean_gap)/frozen_relative_threshold, "
            "abs(positive_rate_gap)/frozen_positive_rate_threshold)"
        ),
        "model_reports": reports,
        "passed_models": passed,
        "passed_model_count": len(passed),
        "permutation_gate_required_for_confirmed_observer_block": True,
        "intervention_outcomes_used": False,
        "sealed_split_read": False,
        "closure_claimed": False,
    }
    write_json(OUTPUT_PATH, output)
    print(
        json.dumps(
            {
                "passed_models": passed,
                "models": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output


if __name__ == "__main__":
    audit()
