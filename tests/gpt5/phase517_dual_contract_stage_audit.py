#!/usr/bin/env python3
"""Audit Phase509-516 without changing any frozen gate or model result."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE509_DIR = ROOT / "tests/gpt5/result/phase509_dual_contract_protocol"
PHASE511_PATH = (
    ROOT
    / "tests/gpt5/result/phase511_calibration_authorization"
    / "phase511_calibration_authorization.json"
)
PHASE513_PATH = (
    ROOT
    / "tests/gpt5/result/phase513_confirmation_authorization"
    / "phase513_confirmation_authorization.json"
)
PHASE515_PATH = (
    ROOT
    / "tests/gpt5/result/phase515_physical_authorization"
    / "phase515_physical_authorization.json"
)
PHASE516_PROTOCOL_PATH = (
    ROOT
    / "tests/gpt5/result/phase516_relation_physical_protocol"
    / "phase516_frozen_relation_physical_contract.json"
)
PHASE516_STATIC_PATH = (
    ROOT
    / "tests/gpt5/result/phase516_relation_physical_protocol"
    / "phase516_relation_physical_static_audit.json"
)
PHASE516_DIR = ROOT / "tests/gpt5/result/phase516_glm4_relation_physical"
PHASE516_SUMMARY_PATH = PHASE516_DIR / "phase516_glm4_relation_physical_summary.json"
OUT_DIR = ROOT / "tests/gpt5/result/phase517_dual_contract_stage_audit"
OUT_PATH = OUT_DIR / "phase517_dual_contract_stage_audit.json"
ATLAS_DIR = ROOT / "frontend/public/vis_data/phase517_relation_binding_decomposition_atlas"
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
REPORT_PATH = (
    ROOT
    / "research/MainAnalysis/20260717_05_Phase509-517关系求值与标签编译双合同审计.md"
)

MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱GLM4）",
    "deepseek7b": "DS7B（深度求索7B）",
}
SURFACES = ("identity", "native_plain_candidate")
BALANCED_CONTROL_SEEDS = tuple(range(517001, 517033))
Z = 1.96


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def binary_rate(predictions: np.ndarray, truths: np.ndarray) -> dict[str, Any]:
    n = int(len(truths))
    count = int((predictions == truths).sum())
    lower, upper = wilson(count, n)
    return {
        "n": n,
        "count": count,
        "rate": count / n if n else 0.0,
        "lcb95": lower,
        "ucb95": upper,
    }


def four_way_rate(
    predictions: np.ndarray,
    truths: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[row["source_pair_id"]].append(index)
    correct = []
    for pair_id, indices in groups.items():
        if len(indices) != 4:
            raise RuntimeError(f"incomplete four-way pair {pair_id}: {len(indices)}")
        correct.append(bool(np.all(predictions[indices] == truths[indices])))
    values = np.asarray(correct, dtype=bool)
    return binary_rate(values, np.ones_like(values, dtype=bool))


def window_metrics(
    predictions: np.ndarray,
    truths: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    by_surface = {}
    for surface in SURFACES:
        mask = np.asarray([row["surface"] == surface for row in metadata], dtype=bool)
        by_surface[surface] = binary_rate(predictions[mask], truths[mask])
    return {
        "overall": binary_rate(predictions, truths),
        "by_surface": by_surface,
        "four_way_pair": four_way_rate(predictions, truths, metadata),
    }


def unit_vectors(values: np.ndarray) -> np.ndarray:
    vectors = values.astype(np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def fit_observers(
    vectors: np.ndarray,
    truths: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = vectors[mask]
    labels = truths[mask]
    mean_true = train[labels].mean(axis=0)
    mean_false = train[~labels].mean(axis=0)
    directions = mean_true - mean_false
    directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
    thresholds = np.einsum("lrd,lrd->lr", (mean_true + mean_false) / 2, directions)
    return directions.astype(np.float32), thresholds.astype(np.float32)


def predict_grid(
    vectors: np.ndarray,
    directions: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    scores = np.einsum("nlrd,lrd->nlr", vectors, directions) - thresholds[None, :, :]
    return scores > 0


def passes_prediction_gate(metrics: dict[str, Any], gate: dict[str, float]) -> bool:
    return bool(
        metrics["by_surface"]["identity"]["lcb95"] >= gate["identity_lcb95_min"]
        and metrics["by_surface"]["native_plain_candidate"]["lcb95"]
        >= gate["native_plain_lcb95_min"]
        and metrics["overall"]["lcb95"] >= gate["overall_lcb95_min"]
        and metrics["four_way_pair"]["lcb95"] >= gate["four_way_pair_lcb95_min"]
    )


def physical_reanalysis(
    protocol: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    observer = np.load(PHASE516_DIR / "phase516_glm4_frozen_observer.npz")
    fit = np.load(PHASE516_DIR / "phase516_glm4_fit_projection.npz")
    prediction = np.load(PHASE516_DIR / "phase516_glm4_prediction_projection.npz")
    fit_metadata = read_jsonl(PHASE516_DIR / "phase516_glm4_fit_metadata.jsonl")
    prediction_metadata = read_jsonl(
        PHASE516_DIR / "phase516_glm4_prediction_metadata.jsonl"
    )

    fit_vectors = unit_vectors(fit["projected"])
    prediction_vectors = unit_vectors(prediction["projected"])
    fit_truths = np.asarray([row["truth_value"] for row in fit_metadata], dtype=bool)
    prediction_truths = np.asarray(
        [row["truth_value"] for row in prediction_metadata], dtype=bool
    )
    train_mask = np.asarray(
        [row["pair_index"] % 2 == 0 for row in fit_metadata], dtype=bool
    )
    directions = observer["directions"].astype(np.float32)
    thresholds = observer["thresholds"].astype(np.float32)
    grid = predict_grid(prediction_vectors, directions, thresholds)
    roles = tuple(protocol["position_roles"])
    gate = protocol["prediction_gate"]
    primary_layer = int(summary["primary_window"]["layer_with_embedding"])
    primary_role = summary["primary_window"]["position_role"]

    windows = []
    for layer in range(grid.shape[1]):
        for role_index, role in enumerate(roles):
            metrics = window_metrics(
                grid[:, layer, role_index], prediction_truths, prediction_metadata
            )
            windows.append({
                "layer_with_embedding": layer,
                "relative_depth": layer / max(1, grid.shape[1] - 1),
                "position_role": role,
                "pre_registered_primary": layer == primary_layer and role == primary_role,
                "passes_numeric_gate": passes_prediction_gate(metrics, gate),
                "minimum_surface_accuracy": min(
                    item["rate"] for item in metrics["by_surface"].values()
                ),
                "metrics": metrics,
                "interpretation": (
                    "frozen primary prediction"
                    if layer == primary_layer and role == primary_role
                    else "post-prediction descriptive window; cannot replace primary"
                ),
            })

    ranked = sorted(
        windows,
        key=lambda row: (
            row["minimum_surface_accuracy"],
            row["metrics"]["four_way_pair"]["rate"],
            row["metrics"]["overall"]["rate"],
        ),
        reverse=True,
    )
    primary = next(row for row in windows if row["pre_registered_primary"])
    passing_by_role = []
    for role in roles:
        passing = [
            row for row in windows
            if row["position_role"] == role and row["passes_numeric_gate"]
        ]
        if passing:
            passing_by_role.append({
                "position_role": role,
                "count": len(passing),
                "minimum_layer_with_embedding": min(
                    row["layer_with_embedding"] for row in passing
                ),
                "maximum_layer_with_embedding": max(
                    row["layer_with_embedding"] for row in passing
                ),
            })

    train_pair_ids = sorted({
        row["source_pair_id"]
        for row, keep in zip(fit_metadata, train_mask, strict=True)
        if keep
    })
    if len(train_pair_ids) % 2:
        raise RuntimeError("balanced sign control requires an even train-pair count")
    balanced_controls = []
    role_index = roles.index(primary_role)
    for seed in BALANCED_CONTROL_SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(train_pair_ids)
        flipped = set(order[: len(order) // 2].tolist())
        random_truths = np.asarray([
            bool(truth) ^ (row["source_pair_id"] in flipped)
            for truth, row in zip(fit_truths, fit_metadata, strict=True)
        ])
        random_directions, random_thresholds = fit_observers(
            fit_vectors, random_truths, train_mask
        )
        random_grid = predict_grid(
            prediction_vectors, random_directions, random_thresholds
        )
        metrics = window_metrics(
            random_grid[:, primary_layer, role_index],
            prediction_truths,
            prediction_metadata,
        )
        balanced_controls.append({
            "seed": seed,
            "train_pair_count": len(train_pair_ids),
            "flipped_train_pair_count": len(flipped),
            "metrics": metrics,
        })

    control_rates = [item["metrics"]["overall"]["rate"] for item in balanced_controls]
    control_pair_rates = [
        item["metrics"]["four_way_pair"]["rate"] for item in balanced_controls
    ]
    original_rates = [item["prediction"]["rate"] for item in summary["random_label_controls"]]
    return {
        "schema_version": "phase517_glm4_relation_physical_reanalysis.v1",
        "roles": list(roles),
        "layer_count_with_embedding": int(grid.shape[1]),
        "window_count": len(windows),
        "pre_registered_primary": primary,
        "primary_matches_phase516_summary": (
            primary["metrics"] == summary["primary_prediction"]
            and primary["passes_numeric_gate"]
            == summary["primary_prediction_gate_pass"]
        ),
        "numeric_gate_passing_window_count": sum(
            bool(row["passes_numeric_gate"]) for row in windows
        ),
        "numeric_gate_passing_windows_by_role": passing_by_role,
        "posthoc_window_policy": (
            "All non-primary windows are descriptive even when their numeric score passes; "
            "none can replace the frozen primary window."
        ),
        "top_posthoc_descriptive_windows": ranked[:12],
        "all_prediction_windows": windows,
        "phase516_unbalanced_random_control_audit": {
            "diagnostic": False,
            "overall_rate_min": min(original_rates),
            "overall_rate_max": max(original_rates),
            "reason": (
                "The original pair-level flips were not constrained to equal positive and "
                "negative orientation among the 24 observer-training pairs, so a residual real "
                "relation direction can dominate either orientation. These controls do not "
                "support or refute the primary observer."
            ),
        },
        "balanced_sign_controls": {
            "status": "supplemental_quality_audit_not_a_registered_gate",
            "seed_count": len(balanced_controls),
            "train_pair_count": len(train_pair_ids),
            "flipped_pair_count_each_seed": len(train_pair_ids) // 2,
            "overall_rate_mean": float(np.mean(control_rates)),
            "overall_rate_min": min(control_rates),
            "overall_rate_max": max(control_rates),
            "four_way_pair_rate_mean": float(np.mean(control_pair_rates)),
            "four_way_pair_rate_min": min(control_pair_rates),
            "four_way_pair_rate_max": max(control_pair_rates),
            "controls": balanced_controls,
        },
    }


def behavior_node(
    model: str,
    contract: str,
    summary: dict[str, Any],
    x: float,
    y: float,
) -> dict[str, Any]:
    if contract == "R":
        score = summary["surface_intersection"]
        pair = summary["paired_world"]
        label = "关系求值合同 R"
    else:
        label_systems = summary["by_label_system"]
        best = max(
            label_systems.items(),
            key=lambda item: item[1]["surface_intersection"]["rate"],
        )[1]
        score = best["surface_intersection"]
        pair = best["mapping_reversal"]
        label = "标签编译合同 B"
    gate_pass = bool(summary["gate_pass"])
    return {
        "id": f"phase517:{model}:behavior:{contract}",
        "label": f"{MODEL_LABELS[model]} / {label}",
        "type": "dual_contract_behavior_gate",
        "model": model,
        "family_id": "reasoning_relation_binding",
        "mechanism_id": "relation_evaluation" if contract == "R" else "label_binding",
        "contract": contract,
        "layer": -1,
        "relative_depth": 0.0,
        "position_role": "behavior_contract",
        "position": [x, y, 0.0],
        "score": score["rate"],
        "lcb95": score["lcb95"],
        "paired_score": pair["rate"],
        "paired_lcb95": pair["lcb95"],
        "gate_pass": gate_pass,
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "evidence_level": "frozen_behavior_contract",
        "color": "#22c55e" if gate_pass else "#ef4444",
        "size": 0.75,
        "show_label": True,
    }


def graph_payload(
    model: str,
    calibration: dict[str, Any],
    confirmation: dict[str, Any],
    physical: dict[str, Any],
) -> dict[str, Any]:
    nodes = [
        behavior_node(model, "R", calibration["contract_summaries"]["R"], -28.0, -9.0),
        behavior_node(model, "B", calibration["contract_summaries"]["B"], -28.0, 9.0),
    ]
    edges: list[dict[str, Any]] = []
    relation_pass = "R" in calibration["passed_contracts"]
    binding_pass = "B" in calibration["passed_contracts"]
    confirmation_id = f"phase517:{model}:relation_confirmation"
    if confirmation["status"] == "complete" and "R" in confirmation["contract_summaries"]:
        result = confirmation["contract_summaries"]["R"]
        nodes.append({
            "id": confirmation_id,
            "label": f"{MODEL_LABELS[model]} / R独立确认",
            "type": "independent_behavior_confirmation",
            "model": model,
            "contract": "R",
            "layer": -1,
            "relative_depth": 0.0,
            "position_role": "behavior_confirmation",
            "position": [-10.0, -9.0, 0.0],
            "score": result["surface_intersection"]["rate"],
            "lcb95": result["surface_intersection"]["lcb95"],
            "paired_score": result["paired_world"]["rate"],
            "paired_lcb95": result["paired_world"]["lcb95"],
            "gate_pass": result["gate_pass"],
            "physical": False,
            "observer": True,
            "predictive": True,
            "causal": False,
            "compute_edge": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "evidence_level": "frozen_independent_behavior_confirmation",
            "color": "#22c55e",
            "size": 0.82,
            "show_label": True,
        })
        edges.append({
            "id": f"phase517:{model}:R_behavior_to_confirmation",
            "source": f"phase517:{model}:behavior:R",
            "target": confirmation_id,
            "type": "measurement_authorization",
            "label": "R校准通过后允许独立确认",
            "score": 1.0,
            "evidence_level": "protocol_gate",
            "predictive": False,
            "compute_edge": False,
            "causal": False,
        })
    else:
        nodes.append({
            "id": confirmation_id,
            "label": f"{MODEL_LABELS[model]} / R独立确认未授权",
            "type": "independent_confirmation_blocked",
            "model": model,
            "contract": "R",
            "layer": -1,
            "relative_depth": 0.0,
            "position_role": "behavior_confirmation",
            "position": [-10.0, -9.0, 0.0],
            "score": 0.0,
            "gate_pass": False,
            "physical": False,
            "observer": True,
            "predictive": False,
            "causal": False,
            "compute_edge": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "evidence_level": "behavior_gate_failure",
            "color": "#64748b",
            "size": 0.62,
            "show_label": True,
        })

    joint_id = f"phase517:{model}:joint_blocked"
    nodes.append({
        "id": joint_id,
        "label": f"{MODEL_LABELS[model]} / 组合合同 J 未授权",
        "type": "joint_contract_blocked",
        "model": model,
        "contract": "J",
        "layer": -1,
        "relative_depth": 0.0,
        "position_role": "joint_gate",
        "position": [-10.0, 9.0, 0.0],
        "score": 0.0,
        "gate_pass": False,
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "evidence_level": "binding_contract_failure",
        "color": "#ef4444",
        "size": 0.72,
        "show_label": True,
    })
    edges.append({
        "id": f"phase517:{model}:R_to_J_requirement",
        "source": f"phase517:{model}:behavior:R",
        "target": joint_id,
        "type": "joint_contract_requirement",
        "label": "J需要R与B共同通过",
        "score": 1.0 if relation_pass else 0.0,
        "evidence_level": "protocol_gate",
        "predictive": False,
        "compute_edge": False,
        "causal": False,
    })
    edges.append({
        "id": f"phase517:{model}:B_to_J_requirement",
        "source": f"phase517:{model}:behavior:B",
        "target": joint_id,
        "type": "joint_contract_requirement",
        "label": "J需要R与B共同通过",
        "score": 1.0 if binding_pass else 0.0,
        "evidence_level": "protocol_gate",
        "predictive": False,
        "compute_edge": False,
        "causal": False,
    })

    if model != "glm4":
        nodes.append({
            "id": f"phase517:{model}:physical_blocked",
            "label": f"{MODEL_LABELS[model]} / 关系物理轨迹未授权",
            "type": "physical_measurement_blocked",
            "model": model,
            "contract": "R",
            "layer": -1,
            "relative_depth": 0.0,
            "position_role": "physical_gate",
            "position": [10.0, -9.0, 0.0],
            "score": 0.0,
            "gate_pass": False,
            "physical": False,
            "observer": True,
            "predictive": False,
            "causal": False,
            "compute_edge": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "evidence_level": "behavior_confirmation_not_passed",
            "color": "#64748b",
            "size": 0.68,
            "show_label": True,
        })
    else:
        primary = physical["pre_registered_primary"]
        role = primary["position_role"]
        role_windows = [
            row for row in physical["all_prediction_windows"]
            if row["position_role"] == role
        ]
        previous_id = confirmation_id
        for row in role_windows:
            layer = row["layer_with_embedding"]
            node_id = f"phase517:glm4:physical:{role}:L{layer}"
            is_primary = row["pre_registered_primary"]
            nodes.append({
                "id": node_id,
                "label": f"GLM4 / {role} / L{layer}",
                "type": "relation_observer_trajectory",
                "model": "glm4",
                "family_id": "reasoning_relation_binding",
                "mechanism_id": "relation_evaluation",
                "contract": "R",
                "layer": layer,
                "relative_depth": row["relative_depth"],
                "position_role": role,
                "position": [10.0 + row["relative_depth"] * 48.0, -9.0, 0.0],
                "score": row["metrics"]["overall"]["rate"],
                "lcb95": row["metrics"]["overall"]["lcb95"],
                "paired_score": row["metrics"]["four_way_pair"]["rate"],
                "paired_lcb95": row["metrics"]["four_way_pair"]["lcb95"],
                "gate_pass": row["passes_numeric_gate"] if is_primary else False,
                "pre_registered_primary": is_primary,
                "posthoc_descriptive": not is_primary,
                "physical": True,
                "observer": True,
                "predictive": True,
                "causal": False,
                "compute_edge": False,
                "single_neuron": False,
                "pipeline_sealed": False,
                "evidence_level": (
                    "frozen_primary_open_prediction_gate_failed"
                    if is_primary
                    else "open_prediction_descriptive_trajectory"
                ),
                "color": "#ef4444" if is_primary else "#38bdf8",
                "size": 0.92 if is_primary else 0.34,
                "show_label": is_primary or layer in {0, len(role_windows) - 1},
            })
            edges.append({
                "id": f"{previous_id}->{node_id}",
                "source": previous_id,
                "target": node_id,
                "type": (
                    "measurement_authorization"
                    if previous_id == confirmation_id
                    else "observational_depth_order"
                ),
                "label": "冻结观察轨迹；不是计算边",
                "score": row["metrics"]["overall"]["rate"],
                "evidence_level": "observational_prediction_order",
                "predictive": True,
                "compute_edge": False,
                "causal": False,
            })
            previous_id = node_id

        role_y = {
            "target_evidence_end": -25.0,
            "distractor_evidence_end": -19.0,
            "claim_entity_end": -9.0,
            "claim_relation_end": -3.0,
            "claim_end": 3.0,
            "prompt_end": 9.0,
        }
        for row in physical["all_prediction_windows"]:
            if (
                row["position_role"] == role
                or not row["passes_numeric_gate"]
            ):
                continue
            current_role = row["position_role"]
            layer = row["layer_with_embedding"]
            node_id = f"phase517:glm4:physical_candidate:{current_role}:L{layer}"
            nodes.append({
                "id": node_id,
                "label": f"GLM4 / {current_role} / L{layer} / 描述候选",
                "type": "posthoc_relation_observer_candidate",
                "model": "glm4",
                "family_id": "reasoning_relation_binding",
                "mechanism_id": "relation_evaluation",
                "contract": "R",
                "layer": layer,
                "relative_depth": row["relative_depth"],
                "position_role": current_role,
                "position": [10.0 + row["relative_depth"] * 48.0, role_y[current_role], 0.0],
                "score": row["metrics"]["overall"]["rate"],
                "lcb95": row["metrics"]["overall"]["lcb95"],
                "paired_score": row["metrics"]["four_way_pair"]["rate"],
                "paired_lcb95": row["metrics"]["four_way_pair"]["lcb95"],
                "gate_pass": False,
                "passes_numeric_gate": True,
                "pre_registered_primary": False,
                "posthoc_descriptive": True,
                "physical": True,
                "observer": True,
                "predictive": True,
                "causal": False,
                "compute_edge": False,
                "single_neuron": False,
                "pipeline_sealed": False,
                "evidence_level": "open_prediction_posthoc_descriptive_candidate",
                "color": "#14b8a6",
                "size": 0.28,
                "show_label": False,
            })

    return {
        "schema_version": "phase517_relation_binding_decomposition_atlas.v1",
        "model": model,
        "evidence_scope": (
            "separate R/B/J behavior gates; GLM4 alone has an observational R trajectory whose "
            "pre-registered open-prediction gate failed; no compute, causal, neuron, or sealed edge"
        ),
        "graph": {
            "meta": {
                "model": model,
                "relation_calibration_pass": relation_pass,
                "binding_calibration_pass": binding_pass,
                "relation_confirmation_pass": (
                    confirmation["status"] == "complete"
                    and "R" in confirmation["passed_contracts"]
                ),
                "physical_measurement": model == "glm4",
                "physical_prediction_gate_pass": (
                    physical["pre_registered_primary"]["passes_numeric_gate"]
                    if model == "glm4"
                    else False
                ),
                "sealed_split_read": False,
                "causal": False,
                "single_neuron": False,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def write_atlas(
    calibration: dict[str, Any],
    confirmation: dict[str, Any],
    physical: dict[str, Any],
) -> None:
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase517_{model}_relation_binding.json"
        payload = graph_payload(
            model,
            calibration["model_summaries"][model],
            confirmation["model_summaries"][model],
            physical,
        )
        (ATLAS_DIR / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        items.append({
            "id": f"phase517_{model}",
            "model": model,
            "path": filename,
            "label": f"{MODEL_LABELS[model]} 关系求值与标签编译分解图谱",
        })
    manifest = {
        "schema_version": "phase517_relation_binding_decomposition_atlas_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": "gpt5",
        "evidence_scope": (
            "R/B/J分合同资格链与GLM4模型专属R观察轨迹；主预测门失败，非计算边、非因果、非神经元、未读密封集"
        ),
        "items": items,
    }
    (ATLAS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    registry = read_json(REGISTRY_PATH)
    source_id = "gpt5_phase517_relation_binding_decomposition_atlas"
    source = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase517 关系求值与标签编译分解图谱",
        "description": "三模型R/B/J行为资格链及GLM4模型专属关系观察轨迹。",
        "manifest_path": "/vis_data/phase517_relation_binding_decomposition_atlas/manifest.json",
        "manifest_schema": "phase517_relation_binding_decomposition_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase517_relation_binding_decomposition_atlas",
        "models": list(MODELS),
        "evidence_scope": "行为与模型专属观察轨迹；预注册物理预测门失败；非计算边、非因果、非神经元闭合",
        "color": "#14b8a6",
    }
    existing = {item["id"]: index for index, item in enumerate(registry["sources"])}
    if source_id in existing:
        registry["sources"][existing[source_id]] = source
    else:
        registry["sources"].append(source)
    registry["generated_at"] = datetime.now(timezone.utc).isoformat()
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_report(audit: dict[str, Any]) -> None:
    calibration = audit["behavior"]["calibration"]
    confirmation = audit["behavior"]["confirmation"]
    primary = audit["physical"]["reanalysis"]["pre_registered_primary"]
    metrics = primary["metrics"]
    controls = audit["physical"]["reanalysis"]["balanced_sign_controls"]
    lines = []
    for model in MODELS:
        summary = calibration["model_summaries"][model]
        relation = summary["contract_summaries"]["R"]
        binding = summary["contract_summaries"]["B"]
        lines.append(
            f"- {MODEL_LABELS[model]}：R门={'通过' if relation['gate_pass'] else '失败'}，"
            f"R两表面交集 {relation['surface_intersection']['count']}/"
            f"{relation['surface_intersection']['n']}，四联世界 "
            f"{relation['paired_world']['count']}/{relation['paired_world']['n']}；"
            f"B门={'通过' if binding['gate_pass'] else '失败'}。"
        )
    report = rf"""# Phase509-517 关系求值与标签编译双合同系统审计

生成时间：{audit['created_at']}

## 一、附件判断复核

附件提出把关系语义求值、任意标签编译和完整输出事件拆成 `R/B/J` 三个合同，方向正确；三观察器联合门适合作为组合闭合门，不应作为关系求值是否值得研究的唯一门；模型专属观察图谱和跨模型共享图谱也必须分账。本轮已经按这四点完成冻结执行。

需要收紧四处。第一，行为分叉只能证明两个任务合同可分，不证明内部已经存在先后串联的两个模块。第二，标签编译提示若直接出现“语义结果为真/假”，模型可通过词面复制取巧，所以本轮状态线索不含候选符号，并同时使用释义表面与映射翻转。第三，`A_R A_B` 只是近似独立串联的描述性零假设，不是组合准确率的数学上界。第四，三个观察器全通过是组合构念门，不是任何模型专属自然真假关系观察轨迹的必要条件。

## 二、实际算法

冻结合同为：

$$
G_R=G_{{固定断言}}\land G_{{四联世界}}\land G_{{双表面}}\land G_{{独立确认}}
$$

$$
G_B=G_{{标签候选}}\land G_{{映射翻转}}\land G_{{双表面}}，\qquad
G_J=G_R\land G_B\land G_{{组合}}
$$

`R` 使用固定断言、相同事实词元多重集和真假世界连接反事实；`B` 排除关系推理，仅交叉平衡状态、A/B与0/1映射、映射方向及两种释义表面；`J` 仅在同一模型的 `R/B` 都通过时开放。物理阶段只对行为独立确认通过的模型与子合同开放，跨模型共享主张仍要求至少两个模型。

物理观察器在拟合集上训练、在独立选择集冻结层和位置，写出观察器账本以后才读取开放预测集：

$$
u_{{l,p}}=\frac{{\bar z_{{l,p}}^+-\bar z_{{l,p}}^-}}{{\|\bar z_{{l,p}}^+-\bar z_{{l,p}}^-\|}},\qquad
\hat r=\mathbf 1[(z\cdot u_{{l,p}})>b_{{l,p}}]
$$

这里的线性观察器只是测量工具，不是机制公式。

## 三、客观结果

{chr(10).join(lines)}

只有 GLM4（智谱GLM4）的 `R` 通过校准与全新关系词独立确认。独立确认中，两表面交集为 189/192，四联世界为 93/96。三个模型的 `B` 均失败，因此 `J` 的授权集为空；不能声称已经验证关系到标签的串联模块。

GLM4（智谱GLM4）获得模型专属 `R` 观察轨迹资格。拟合集冻结的主窗口是带嵌入计数的第 {primary['layer_with_embedding']} 层、`{primary['position_role']}`。开放预测集结果为：

- 原表面：{metrics['by_surface']['identity']['count']}/{metrics['by_surface']['identity']['n']}，LCB95={metrics['by_surface']['identity']['lcb95']:.4f}；
- 自然释义表面：{metrics['by_surface']['native_plain_candidate']['count']}/{metrics['by_surface']['native_plain_candidate']['n']}，LCB95={metrics['by_surface']['native_plain_candidate']['lcb95']:.4f}；
- 全部行：{metrics['overall']['count']}/{metrics['overall']['n']}，LCB95={metrics['overall']['lcb95']:.4f}；
- 四联世界：{metrics['four_way_pair']['count']}/{metrics['four_way_pair']['n']}，LCB95={metrics['four_way_pair']['lcb95']:.4f}。

冻结门要求四联世界下界至少0.75，实际只有 {metrics['four_way_pair']['lcb95']:.4f}，所以主预测门失败。不能事后换成其他高分窗口。固定投影中世界变化距离与表面变化距离比为 {audit['physical']['phase516_summary']['primary_distance_controls']['world_to_surface_distance_ratio']:.3f}，这是描述性几何现象，不是运输或因果边。

全层六角色共有 {audit['physical']['reanalysis']['window_count']} 个窗口，其中 {audit['physical']['reanalysis']['numeric_gate_passing_window_count']} 个非主窗口达到相同数值门：断言实体末端位于第9-24层，断言关系末端位于第17-31层，断言整体末端位于第13-25层，提示终端位于第19-40层；目标证据和干扰证据末端没有窗口通过。由于这些窗口没有经过冻结的唯一窗口选择，只能登记为“中晚层查询/输出侧广泛候选分布”。它提示下一次选择规则应冻结平台区或窗口族，而不是在拟合集并列满分时机械选择最早单点；本轮不能据此改判成功。

原随机标签对照的准确率范围达到 {audit['physical']['reanalysis']['phase516_unbalanced_random_control_audit']['overall_rate_min']:.3f}-{audit['physical']['reanalysis']['phase516_unbalanced_random_control_audit']['overall_rate_max']:.3f}，由于24个训练世界对的翻转方向没有强制平衡，该对照不能诊断观察器。补充的32个严格12/12平衡翻转仅作为质量审计，其逐行准确率均值为 {controls['overall_rate_mean']:.3f}，范围 {controls['overall_rate_min']:.3f}-{controls['overall_rate_max']:.3f}；它不参与预注册成功门。

## 四、问题与硬伤

第一，`R` 的候选真/假裕量高度可分，但自然自由输出事件为0；当前观察到的是候选间相对偏好，不是模型会自然输出该标签的完整事件。第二，主窗口从24个选择世界对、全层乘六角色中挑选，开放预测门本应防止多窗口乐观偏差，而它确实失败。第三，四联世界失败说明91.15%的逐行分数包含不稳定样本，不能池化掩盖。第四，固定64维投影会丢失信息，也可能保留模板相关方向。第五，只有GLM4（智谱GLM4）通过，不能称为跨模型关系编码规律。第六，本轮没有组件运输、注意力头、通道、神经元、必要性、充分性或因果中介证据。

小模型可能与更大模型或人脑存在30%-50%的编码差异，这限制外推范围，但不改变本轮 `Qwen3/DS7B R失败、三模型B失败、GLM4物理预测门失败` 的事实。

## 五、图谱与理论更新

图谱新增的是：三模型 `R/B/J` 资格链，以及 GLM4（智谱GLM4）在断言实体末端的模型专属、开放预测失败的层级观察轨迹。所有层间连线都标记为观察顺序，`compute_edge=false`、`causal=false`、`single_neuron=false`；没有用高分节点伪造计算脉络。

理论名称继续保持“语言是动态模式网络”。本轮只支持更细的候选分解：

$$
X_{{s+1}}=\mathcal F_\Theta(X_s,c_s,\kappa_s)
$$

$$
(W,C)\xrightarrow{{\mathcal R_f}}r
\xrightarrow{{\mathcal B_\mu}}l
\xrightarrow{{\mathcal S}}y
$$

其中 `R/B/S` 是需要分别验证的功能合同，不是已经恢复的内部模块。全局图谱继续区分测量、观察、预测、计算和因果边：

$$
E^{{measurement}}\ne E^{{observation}}\ne E^{{prediction}}
\ne E^{{compute}}\ne E^{{causal}}
$$

本轮没有得到可替代线性公式的真实运行机制，也没有识别稀疏门、模式竞争、关系算子或功能到物理状态的映射。

## 六、闭合、进度与阶段决定

严格机制闭合仍为0/72，总体科学成熟度保持25%，合理区间24%-26%。新增的是一个失败但可视化的模型专属观察轨迹，不是机制闭合进度。

Phase509（阶段509）的完整分解目标已经执行完毕：`R` 只有一个模型通过，`B` 无模型通过，`J` 为空，唯一获准的物理主预测门失败，密封集未读取。当前同阶段没有合法的因果或神经元后续，必须停止自动扩张。

下一阶段应重新冻结两个大任务，而不是微调阈值：先重建不依赖候选词元裕量的自然关系输出合同，并在更大的四联世界样本上复验；同时对 `B` 做错误分解，区分映射理解、候选选择和自由输出，但禁止使用显式真假词面捷径。只有新的独立行为合同通过，才重新开放物理图谱。该工作改变合同和分母，属于新阶段，不能用本轮结果后改门。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    contract = read_json(PHASE509_DIR / "phase509_frozen_contract.json")
    static = read_json(PHASE509_DIR / "phase509_static_audit.json")
    calibration = read_json(PHASE511_PATH)
    confirmation = read_json(PHASE513_PATH)
    physical_auth = read_json(PHASE515_PATH)
    physical_protocol = read_json(PHASE516_PROTOCOL_PATH)
    physical_static = read_json(PHASE516_STATIC_PATH)
    phase516_summary = read_json(PHASE516_SUMMARY_PATH)

    if calibration["models_in_required_order"] != list(MODELS):
        raise RuntimeError("calibration model order drift")
    if physical_auth["physical_models_in_required_order"] != ["glm4"]:
        raise RuntimeError("unexpected physical authorization")
    if physical_static["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase516 physical protocol static audit failed")
    if any(
        summary["sealed_split_read"]
        for stage in (calibration, confirmation, physical_auth)
        for summary in stage["model_summaries"].values()
    ):
        raise RuntimeError("sealed split was unexpectedly read")

    reanalysis = physical_reanalysis(physical_protocol, phase516_summary)
    calibration_rows = sum(
        item["row_count"] for item in calibration["model_summaries"].values()
    )
    confirmation_rows = sum(
        item["row_count"] for item in confirmation["model_summaries"].values()
    )
    joint_rows = sum(
        item["row_count"] for item in physical_auth["model_summaries"].values()
    )
    audit = {
        "schema_version": "phase517_dual_contract_stage_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stage_complete_prediction_gate_stopped",
        "attachment_audit": {
            "overall_direction_correct": True,
            "behavioral_dissociation_proves_sequential_internal_modules": False,
            "explicit_truth_word_binding_prompt_has_lexical_shortcut": True,
            "accuracy_product_is_upper_bound": False,
            "accuracy_product_role": "descriptive independence null only",
            "three_observer_gate_required_for_combined_contract": True,
            "three_observer_gate_required_for_model_specific_R_observation": False,
        },
        "protocol": {
            "phase509_contract": contract,
            "phase509_static_audit": static,
            "phase516_physical_contract": physical_protocol,
            "phase516_static_audit": physical_static,
        },
        "behavior": {
            "calibration": calibration,
            "confirmation": confirmation,
            "joint_and_physical_authorization": physical_auth,
        },
        "physical": {
            "phase516_summary": phase516_summary,
            "reanalysis": reanalysis,
        },
        "denominators": {
            "calibration_model_rows": calibration_rows,
            "confirmation_model_rows": confirmation_rows,
            "joint_model_rows": joint_rows,
            "physical_fit_rows": phase516_summary["fit_row_count"],
            "physical_prediction_rows": phase516_summary["prediction_row_count"],
            "physical_total_rows": (
                phase516_summary["fit_row_count"]
                + phase516_summary["prediction_row_count"]
            ),
            "strict_closed_mechanisms": 0,
            "mechanism_denominator": 72,
        },
        "gates": {
            "relation_calibration_models": calibration["relation_models"],
            "binding_calibration_models": calibration["binding_models"],
            "relation_confirmation_models": confirmation["relation_models"],
            "binding_confirmation_models": confirmation["binding_models"],
            "joint_confirmation_models": physical_auth["joint_models"],
            "model_specific_physical_contracts": physical_auth[
                "physical_contracts_by_model"
            ],
            "shared_relation_physical": physical_auth["authorization"][
                "shared_relation_physical"
            ],
            "glm4_relation_primary_prediction": phase516_summary[
                "primary_prediction_gate_pass"
            ],
            "sealed_authorized": False,
            "causal_authorized": False,
            "neuron_authorized": False,
        },
        "evidence_boundary": {
            "sealed_split_read": False,
            "hidden_state_collected": True,
            "model_specific_observational_trajectory": True,
            "shared_cross_model_physical_rule": False,
            "compute_transport_measured": False,
            "causal_intervention": False,
            "head_channel_neuron_scan": False,
            "strict_mechanism_closure": False,
        },
        "theory_audit": {
            "R_B_J_are_valid_separate_functional_contracts": True,
            "R_B_J_are_identified_sequential_internal_modules": False,
            "glm4_primary_window_is_stable_relation_physical_path": False,
            "candidate_margin_equals_natural_output_event": False,
            "dynamic_pattern_network": "candidate_experimental_framework_only",
            "mode_operator_identified": False,
            "global_physical_atlas_advanced": "observational_failed-gate puzzle only",
        },
        "progress": {
            "point_percent": 25,
            "range_percent": [24, 26],
            "strict_closure_rate": 0.0,
        },
        "stage_decision": {
            "phase509_objective_complete": True,
            "automatic_physical_followup_executed": True,
            "automatic_causal_or_neuron_followup_executed": False,
            "same_stage_has_authorized_next_step": False,
            "stop_reason": (
                "B and J are empty, shared R is empty, and the sole model-specific GLM4 R "
                "primary open-prediction gate failed."
            ),
            "next_distinct_phase": (
                "freeze a natural-output R contract and a decomposed B error ledger before any "
                "new physical collection"
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_atlas(calibration, confirmation, reanalysis)
    write_report(audit)
    print(OUT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
