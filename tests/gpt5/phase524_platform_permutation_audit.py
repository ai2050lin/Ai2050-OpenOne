#!/usr/bin/env python3
"""Run the frozen Phase518 full-pipeline grouped permutation audit.

Each null replicate flips labels within complete four-row source pairs, refits the
observer, rediscovers role-local contiguous platforms, and evaluates the maximum
frozen-platform score on an independently flipped prediction split.  Index zero
is the natural-label pipeline and is never included in the null distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase523_world_query_platform_physical import (  # noqa: E402
    MODELS,
    SURFACES,
    TASKS,
    binary_rate,
    metrics,
    read_json,
    read_jsonl,
    task_labels,
    unit_vectors,
    wilson,
)


PHASE523_DIR = ROOT / "tests/gpt5/result/phase523_world_query_platform_physical"
CONTRACT_PATH = (
    ROOT
    / "tests/gpt5/result/phase518_world_query_platform_protocol"
    / "phase518_frozen_contract.json"
)
OUT_DIR = ROOT / "tests/gpt5/result/phase524_platform_permutation_audit"


def wilson_lower_array(counts: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros_like(counts, dtype=np.float64)
    values = counts.astype(np.float64)
    p = values / n
    z = 1.96
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return np.maximum(0.0, center - radius)


def grouped_indices(metadata: list[dict[str, Any]]) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    folds: dict[str, int] = {}
    for index, row in enumerate(metadata):
        groups[row["source_pair_id"]].append(index)
        folds[row["source_pair_id"]] = int(row["pair_index"]) % 4
    group_ids = sorted(groups)
    indices = [np.asarray(groups[group_id], dtype=int) for group_id in group_ids]
    if any(len(item) != 4 for item in indices):
        raise RuntimeError("permutation groups must contain exactly four rows")
    return group_ids, indices, np.asarray([folds[group_id] for group_id in group_ids], dtype=int)


def factor_matrix(count: int, group_count: int, rng: np.random.Generator) -> np.ndarray:
    factors = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(count + 1, group_count))
    factors[0] = 1.0
    return factors


def permuted_labels(
    base_labels: np.ndarray,
    group_rows: list[np.ndarray],
    factors: np.ndarray,
) -> np.ndarray:
    labels = np.broadcast_to(base_labels, (len(factors), len(base_labels))).copy()
    for group_index, rows in enumerate(group_rows):
        flip = factors[:, group_index] < 0
        labels[:, rows] ^= flip[:, None]
    return labels


def pair_contrasts(
    vectors: np.ndarray,
    labels: np.ndarray,
    group_rows: list[np.ndarray],
) -> np.ndarray:
    signs = np.where(labels, 1.0, -1.0).astype(np.float32)
    contrasts = []
    for rows in group_rows:
        if int(labels[rows].sum()) != 2:
            raise RuntimeError("each source pair must remain label balanced")
        contrasts.append(np.einsum("n,nplrd->plrd", signs[rows], vectors[rows]))
    return np.stack(contrasts, axis=0)


def batched_oof_predictions(
    vectors: np.ndarray,
    contrasts: np.ndarray,
    factors: np.ndarray,
    group_rows: list[np.ndarray],
    group_folds: np.ndarray,
) -> np.ndarray:
    predictions = np.zeros((len(factors),) + vectors.shape[:-1], dtype=bool)
    for fold in range(4):
        train_groups = group_folds != fold
        test_groups = np.flatnonzero(group_folds == fold)
        test_rows = np.concatenate([group_rows[index] for index in test_groups])
        train_rows = np.concatenate([
            group_rows[index] for index in np.flatnonzero(train_groups)
        ])
        directions = np.einsum(
            "kg,gplrd->kplrd",
            factors[:, train_groups],
            contrasts[train_groups],
            optimize=True,
        )
        center = vectors[train_rows].mean(axis=0)
        centered = vectors[test_rows] - center[None, :, :, :, :]
        scores = np.einsum(
            "nplrd,kplrd->knplr",
            centered,
            directions,
            optimize=True,
        )
        predictions[:, test_rows] = scores > 0
    return predictions


def batched_prediction_grid(
    discovery_vectors: np.ndarray,
    prediction_vectors: np.ndarray,
    contrasts: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    directions = np.einsum("kg,gplrd->kplrd", factors, contrasts, optimize=True)
    center = discovery_vectors.mean(axis=0)
    centered = prediction_vectors - center[None, :, :, :, :]
    scores = np.einsum(
        "nplrd,kplrd->knplr",
        centered,
        directions,
        optimize=True,
    )
    return scores > 0


def fold_pass_counts(
    predictions: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    group_rows: list[np.ndarray],
    group_folds: np.ndarray,
    gate: dict[str, float],
) -> np.ndarray:
    passes = np.zeros(predictions.shape[0:1] + predictions.shape[2:], dtype=np.uint8)
    surfaces = np.asarray([row["surface"] for row in metadata])
    correct_global = predictions == labels[:, :, None, None, None]
    for fold in range(4):
        groups = np.flatnonzero(group_folds == fold)
        rows = np.concatenate([group_rows[index] for index in groups])
        correct = correct_global[:, rows]
        overall_lcb = wilson_lower_array(correct.sum(axis=1), len(rows))
        surface_lcbs = []
        for surface in SURFACES:
            local = surfaces[rows] == surface
            surface_lcbs.append(wilson_lower_array(correct[:, local].sum(axis=1), int(local.sum())))
        pair_correct = np.stack(
            [correct_global[:, group_rows[index]].all(axis=1) for index in groups],
            axis=1,
        )
        pair_lcb = wilson_lower_array(pair_correct.sum(axis=1), len(groups))
        fold_pass = (
            (overall_lcb >= gate["overall_lcb95_min"])
            & (surface_lcbs[0] >= gate["surface_lcb95_min"])
            & (surface_lcbs[1] >= gate["surface_lcb95_min"])
            & (pair_lcb >= gate["four_way_lcb95_min"])
        )
        passes += fold_pass.astype(np.uint8)
    return passes


def contiguous_platforms(
    fold_passes: np.ndarray,
    roles: tuple[str, ...],
    design: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    projection_pass = fold_passes >= int(design["minimum_fold_passes"])
    consensus = projection_pass.sum(axis=1) >= int(design["projection_consensus_required"])
    minimum = int(design["role_local_minimum_contiguous_layers"])
    all_platforms = []
    for replicate in range(consensus.shape[0]):
        platforms = []
        for role_index, role in enumerate(roles):
            layers = np.flatnonzero(consensus[replicate, :, role_index]).tolist()
            runs: list[list[int]] = []
            for layer in layers:
                if not runs or layer != runs[-1][-1] + 1:
                    runs.append([layer])
                else:
                    runs[-1].append(layer)
            for run in runs:
                if len(run) >= minimum:
                    platforms.append({
                        "platform_id": f"{role}:L{run[0]}-L{run[-1]}",
                        "role_index": role_index,
                        "position_role": role,
                        "layers_with_embedding": run,
                    })
        all_platforms.append(platforms)
    return all_platforms


def gate_margin(
    predictions: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    gate: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    report = metrics(predictions, labels, metadata)
    margins = [
        report["overall"]["lcb95"] - gate["overall_lcb95_min"],
        report["four_way_pair"]["lcb95"] - gate["four_way_lcb95_min"],
    ]
    margins.extend(
        item["lcb95"] - gate["surface_lcb95_min"]
        for item in report["by_surface"].values()
    )
    return min(margins), report


def platform_score(
    grid: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    platform: dict[str, Any],
    gate: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    role = int(platform["role_index"])
    layers = list(platform["layers_with_embedding"])
    votes = grid[:, :, layers, role]
    aggregate = votes.mean(axis=(1, 2)) > 0.5
    aggregate_margin, aggregate_report = gate_margin(aggregate, labels, metadata, gate)
    projection_reports = {}
    margins = [aggregate_margin]
    for projection in range(grid.shape[1]):
        prediction = votes[:, projection, :].mean(axis=1) > 0.5
        margin, report = gate_margin(prediction, labels, metadata, gate)
        margins.append(margin)
        projection_reports[str(projection)] = {"margin": margin, "metrics": report}
    return min(margins), {
        "aggregate_margin": aggregate_margin,
        "aggregate_metrics": aggregate_report,
        "by_projection": projection_reports,
        "familywise_gate_margin": min(margins),
    }


def task_audit(
    model: str,
    task: str,
    discovery_vectors: np.ndarray,
    prediction_vectors: np.ndarray,
    discovery_meta: list[dict[str, Any]],
    prediction_meta: list[dict[str, Any]],
    roles: tuple[str, ...],
    design: dict[str, Any],
    permutation_count: int,
    rng: np.random.Generator,
    frozen_platform_ids: set[str],
) -> dict[str, Any]:
    discovery_labels = task_labels(discovery_meta, task)
    prediction_labels = task_labels(prediction_meta, task)
    _disc_ids, disc_groups, disc_folds = grouped_indices(discovery_meta)
    _pred_ids, pred_groups, _pred_folds = grouped_indices(prediction_meta)
    discovery_factors = factor_matrix(permutation_count, len(disc_groups), rng)
    prediction_factors = factor_matrix(permutation_count, len(pred_groups), rng)
    discovery_label_matrix = permuted_labels(discovery_labels, disc_groups, discovery_factors)
    prediction_label_matrix = permuted_labels(prediction_labels, pred_groups, prediction_factors)
    contrasts = pair_contrasts(discovery_vectors, discovery_labels, disc_groups)

    oof = batched_oof_predictions(
        discovery_vectors,
        contrasts,
        discovery_factors,
        disc_groups,
        disc_folds,
    )
    fold_passes = fold_pass_counts(
        oof,
        discovery_label_matrix,
        discovery_meta,
        disc_groups,
        disc_folds,
        design["discovery_local_gate"],
    )
    platforms = contiguous_platforms(fold_passes, roles, design)
    observed_ids = {item["platform_id"] for item in platforms[0]}
    if observed_ids != frozen_platform_ids:
        raise RuntimeError(
            f"{model} {task} fast pipeline disagrees with frozen ledger: "
            f"{sorted(observed_ids)} != {sorted(frozen_platform_ids)}"
        )

    prediction_grid = batched_prediction_grid(
        discovery_vectors,
        prediction_vectors,
        contrasts,
        discovery_factors,
    )
    maximum_margins = []
    platform_counts = []
    observed_platform_reports = []
    for replicate, candidates in enumerate(platforms):
        platform_counts.append(len(candidates))
        scores = []
        for platform in candidates:
            score, report = platform_score(
                prediction_grid[replicate],
                prediction_label_matrix[replicate],
                prediction_meta,
                platform,
                design["prediction_gate"],
            )
            scores.append(score)
            if replicate == 0:
                observed_platform_reports.append({
                    **platform,
                    "prediction": report,
                    "prediction_gate_pass": score >= 0.0,
                })
        maximum_margins.append(max(scores) if scores else -1.0)

    observed_margin = maximum_margins[0]
    null_margins = np.asarray(maximum_margins[1:], dtype=np.float64)
    quantile = float(np.quantile(null_margins, design["permutation_quantile"], method="higher"))
    exceedances = int((null_margins >= observed_margin).sum())
    p_value = (exceedances + 1) / (len(null_margins) + 1)
    observed_platform_reports.sort(
        key=lambda item: (-item["prediction"]["familywise_gate_margin"], item["platform_id"])
    )
    return {
        "task": task,
        "pipeline_recomputed_from_projection_arrays": True,
        "observed_platform_count": platform_counts[0],
        "observed_platforms": observed_platform_reports,
        "observed_maximum_familywise_gate_margin": observed_margin,
        "null_maximum_familywise_gate_margins": null_margins.tolist(),
        "null_platform_counts": platform_counts[1:],
        "null_quantile": quantile,
        "permutation_p_value": p_value,
        "familywise_significant": bool(
            observed_margin >= 0.0 and observed_margin > quantile
        ),
        "natural_labels_are_replicate_zero": True,
        "null_replicate_count": len(null_margins),
    }


def behavior_qualified(summary: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    gate = contract["gates"]["natural_relation"]
    discovery = bool(summary["discovery_behavior_gate_pass"])
    prediction = summary["prediction_behavior"]
    prediction_pass = (
        prediction["overall"]["lcb95"] >= gate["surface_lcb95_min"]
        and all(item["lcb95"] >= gate["surface_lcb95_min"] for item in prediction["by_surface"].values())
        and prediction["four_way_pair"]["lcb95"] >= gate["four_way_lcb95_min"]
        and prediction["unrecoverable"]["ucb95"] <= gate["unrecoverable_ucb95_max"]
    )
    return {
        "discovery_gate_pass": discovery,
        "prediction_gate_pass": prediction_pass,
        "both_splits_pass": discovery and prediction_pass,
    }


def audit_model(model: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"phase524_{model}_platform_permutation_summary.json"
    source_summary_path = PHASE523_DIR / f"phase523_{model}_world_query_platform_summary.json"
    source_summary = read_json(source_summary_path)
    if source_summary["status"] != "complete":
        payload = {
            "schema_version": "phase524_platform_permutation_summary.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "not_authorized",
            "model": model,
            "cuda_used": False,
            "sealed_split_read": False,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_path)
        return output_path

    contract = read_json(CONTRACT_PATH)
    design = contract["physical_design"]
    roles = tuple(design["position_roles"])
    discovery_npz = np.load(
        PHASE523_DIR / f"phase523_{model}_discovery_projection.npz"
    )
    prediction_npz = np.load(
        PHASE523_DIR / f"phase523_{model}_prediction_projection.npz"
    )
    discovery_vectors = unit_vectors(discovery_npz["projected"])
    prediction_vectors = unit_vectors(prediction_npz["projected"])
    discovery_meta = read_jsonl(PHASE523_DIR / f"phase523_{model}_discovery_metadata.jsonl")
    prediction_meta = read_jsonl(PHASE523_DIR / f"phase523_{model}_prediction_metadata.jsonl")
    ledger = read_json(PHASE523_DIR / f"phase523_{model}_frozen_platform_ledger.json")
    rng = np.random.default_rng(int(design["pipeline_permutation_seed"]))
    reports = {}
    for task in TASKS:
        frozen_ids = {item["platform_id"] for item in ledger["tasks"][task]["platforms"]}
        reports[task] = task_audit(
            model,
            task,
            discovery_vectors,
            prediction_vectors,
            discovery_meta,
            prediction_meta,
            roles,
            design,
            int(design["pipeline_permutation_count"]),
            rng,
            frozen_ids,
        )
    behavior = behavior_qualified(source_summary, contract)
    payload = {
        "schema_version": "phase524_platform_permutation_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "model": model,
        "behavior_qualification": behavior,
        "tasks": reports,
        "observational_platform_confirmed": any(
            report["familywise_significant"] for report in reports.values()
        ) and behavior["both_splits_pass"],
        "cuda_used": False,
        "model_weights_loaded": False,
        "prediction_arrays_read": True,
        "sealed_split_read": False,
        "causal": False,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    audit_model(args.model)


if __name__ == "__main__":
    main()
