#!/usr/bin/env python3
"""Analyze and publish the Phase424 global physical-path census."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase330_nine_family_case_bank import FAMILY_MECHANISMS, FAMILY_NAMES  # noqa: E402
from phase424_global_physical_protocol import MODELS, OUT, SCHEMA_VERSION  # noqa: E402


VIS = ROOT / "frontend/public/vis_data/phase424_global_physical_path_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
BASELINE_FEATURES = (
    "executed_token_count_mean",
    "source_token_count_mean",
    "query_token_count_mean",
    "control_token_count_mean",
    "target_word_count_mean",
    "target_leak_fraction",
)
PHYSICAL_BASE_FEATURES = (
    "formation_specificity",
    "transport_contrast_specificity",
    "source_mass_specificity",
    "source_target_specificity",
    "query_target_alignment",
    "cancellation_index",
)
PHYSICAL_FEATURES = tuple(
    f"{depth}_{feature}_median"
    for depth in ("early", "middle", "late")
    for feature in PHYSICAL_BASE_FEATURES
)
SIGNALS = {
    "formation": "formation_specificity",
    "transport": "transport_contrast_specificity",
    "competition": "source_target_specificity",
}
DEPTH_ORDER = {"early": 0, "middle": 1, "late": 2}
COLORS = {
    "formation": "#22c55e",
    "transport": "#06b6d4",
    "competition": "#f59e0b",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase424 non-finite scalar: {value}")
    return round(float(value), 10)


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def positive_fraction(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(sum(value > 0.0 for value in rows) / len(rows)) if rows else 0.0


def matrix(rows: list[dict[str, Any]], features: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[key]) for key in features] for row in rows], dtype=np.float64)


def fit_ridge(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    alpha: float = 1.0,
) -> dict[str, Any]:
    x = matrix(rows, features)
    y = np.asarray([float(row["behavior_margin_mean"]) for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    y_mean = float(y.mean())
    centered = y - y_mean
    regularizer = alpha * np.eye(z.shape[1], dtype=np.float64)
    coefficient = np.linalg.solve(z.T @ z + regularizer, z.T @ centered)
    return {
        "features": list(features),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept_center": y_mean,
        "alpha": alpha,
    }


def predict(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    features = tuple(model["features"])
    x = matrix(rows, features)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    return float(model["intercept_center"]) + ((x - mean) / scale) @ coefficient


def regression_metrics(model: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "r2": None,
            "mae": None,
            "sign_accuracy": None,
        }
    truth = np.asarray([float(row["behavior_margin_mean"]) for row in rows], dtype=np.float64)
    estimate = predict(model, rows)
    residual = float(np.square(truth - estimate).sum())
    total = float(np.square(truth - truth.mean()).sum())
    r2 = None if total <= 1e-12 else clean(1.0 - residual / total)
    return {
        "count": len(rows),
        "r2": r2,
        "mae": clean(float(np.abs(truth - estimate).mean())),
        "sign_accuracy": clean(float(((truth > 0.0) == (estimate > 0.0)).mean())),
        "prediction_mean": clean(float(estimate.mean())),
        "truth_mean": clean(float(truth.mean())),
    }


def prediction_audits(pair_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_summaries:
        grouped[(row["model"], row["family_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (model, family), values in sorted(grouped.items()):
        by_split = {
            split: [row for row in values if row["split"] == split]
            for split in (
                "discovery",
                "calibration",
                "behavior_holdout",
                "legacy_physical_holdout",
            )
        }
        baseline = fit_ridge(by_split["discovery"], BASELINE_FEATURES)
        physical = fit_ridge(
            by_split["discovery"], BASELINE_FEATURES + PHYSICAL_FEATURES
        )
        split_metrics: dict[str, Any] = {}
        for split, rows in by_split.items():
            baseline_metrics = regression_metrics(baseline, rows)
            physical_metrics = regression_metrics(physical, rows)
            delta_r2 = None
            if baseline_metrics["r2"] is not None and physical_metrics["r2"] is not None:
                delta_r2 = clean(
                    float(physical_metrics["r2"]) - float(baseline_metrics["r2"])
                )
            mae_gain = clean(
                float(baseline_metrics["mae"]) - float(physical_metrics["mae"])
            )
            split_metrics[split] = {
                "baseline": baseline_metrics,
                "physical": physical_metrics,
                "delta_r2": delta_r2,
                "mae_gain": mae_gain,
            }
        required = [split_metrics["calibration"], split_metrics["behavior_holdout"]]
        gate = all(
            item["delta_r2"] is not None
            and float(item["delta_r2"]) > 0.05
            and float(item["mae_gain"]) > 0.0
            for item in required
        )
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase424-FrozenPredictionAudit",
                "created_at": now(),
                "model": model,
                "family_id": family,
                "discovery_count": len(by_split["discovery"]),
                "calibration_count": len(by_split["calibration"]),
                "behavior_holdout_count": len(by_split["behavior_holdout"]),
                "legacy_physical_holdout_count": len(by_split["legacy_physical_holdout"]),
                "baseline_features": list(BASELINE_FEATURES),
                "physical_features": list(PHYSICAL_FEATURES),
                "ridge_alpha": 1.0,
                "split_metrics": split_metrics,
                "calibration_and_behavior_prediction_gate_pass": gate,
                "legacy_holdout_used_for_gate": False,
                "strict_double_blind": False,
                "predictive": gate,
                "causal": False,
            }
        )
    return output


def selected_depth(
    rows: list[dict[str, Any]],
    feature: str,
) -> tuple[str, dict[str, float]]:
    discovery = [row for row in rows if row["split"] == "discovery"]
    medians = {
        depth: median(
            row[feature] for row in discovery if row["depth_bin"] == depth
        )
        for depth in ("early", "middle", "late")
    }
    selected = max(medians, key=lambda depth: (medians[depth], -DEPTH_ORDER[depth]))
    return selected, medians


def split_replication(
    rows: list[dict[str, Any]],
    split: str,
    depth: str,
    feature: str,
) -> dict[str, Any]:
    values = [
        float(row[feature])
        for row in rows
        if row["split"] == split and row["depth_bin"] == depth
    ]
    return {
        "layer_row_count": len(values),
        "median": median(values),
        "positive_fraction": positive_fraction(values),
        "gate_pass": bool(median(values) > 0.0 and positive_fraction(values) >= 0.75),
    }


def mechanism_maps(
    pair_layers: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prediction = {
        (row["model"], row["family_id"]): row for row in prediction_rows
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_layers:
        grouped[(row["model"], row["family_id"], row["mechanism_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (model, family, mechanism), values in sorted(grouped.items()):
        behavior_by_split: dict[str, Any] = {}
        for split in ("calibration", "behavior_holdout", "legacy_physical_holdout"):
            pair_status: dict[str, bool] = {}
            for row in values:
                if row["split"] == split:
                    pair_status[row["pair_id"]] = bool(row["both_branches_correct"])
            correct_fraction = clean(
                sum(pair_status.values()) / len(pair_status) if pair_status else 0.0
            )
            behavior_by_split[split] = {
                "pair_count": len(pair_status),
                "both_branches_correct_fraction": correct_fraction,
                "gate_pass": correct_fraction >= 0.75,
            }
        behavior_qualification_pass = bool(
            behavior_by_split["calibration"]["gate_pass"]
            and behavior_by_split["behavior_holdout"]["gate_pass"]
        )
        signal_rows: dict[str, Any] = {}
        selected_bins: dict[str, str] = {}
        signal_gates: dict[str, bool] = {}
        for signal, feature in SIGNALS.items():
            depth, discovery_medians = selected_depth(values, feature)
            selected_bins[signal] = depth
            replications = {
                split: split_replication(values, split, depth, feature)
                for split in (
                    "calibration",
                    "behavior_holdout",
                    "legacy_physical_holdout",
                )
            }
            gate = bool(
                replications["calibration"]["gate_pass"]
                and replications["behavior_holdout"]["gate_pass"]
            )
            signal_gates[signal] = gate
            signal_rows[signal] = {
                "feature": feature,
                "selected_depth_bin": depth,
                "selection_uses_discovery_only": True,
                "discovery_medians_by_depth": discovery_medians,
                "replication": replications,
                "calibration_and_behavior_gate_pass": gate,
                "legacy_replication_used_for_gate": False,
            }
        order_pass = bool(
            DEPTH_ORDER[selected_bins["formation"]]
            <= DEPTH_ORDER[selected_bins["transport"]] + 1
            and DEPTH_ORDER[selected_bins["transport"]]
            <= DEPTH_ORDER[selected_bins["competition"]] + 1
        )
        family_prediction = prediction[(model, family)][
            "calibration_and_behavior_prediction_gate_pass"
        ]
        formation_transport_candidate = bool(
            signal_gates["formation"]
            and signal_gates["transport"]
            and DEPTH_ORDER[selected_bins["formation"]]
            <= DEPTH_ORDER[selected_bins["transport"]] + 1
        )
        behavior_qualified_formation_transport_candidate = bool(
            formation_transport_candidate and behavior_qualification_pass
        )
        path_candidate = bool(
            all(signal_gates.values())
            and order_pass
            and behavior_qualification_pass
        )
        predictive_path_candidate = bool(path_candidate and family_prediction)
        legacy_replication_pass = bool(
            all(
                signal_rows[signal]["replication"]["legacy_physical_holdout"][
                    "gate_pass"
                ]
                for signal in SIGNALS
            )
        )
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase424-MechanismPhysicalMap",
                "created_at": now(),
                "model": model,
                "family_id": family,
                "mechanism_id": mechanism,
                "signals": signal_rows,
                "selected_event_topology": [
                    f"formation:{selected_bins['formation']}",
                    f"transport:{selected_bins['transport']}",
                    f"competition:{selected_bins['competition']}",
                ],
                "partial_order_gate_pass": order_pass,
                "behavior_qualification": behavior_by_split,
                "behavior_qualification_gate_pass": behavior_qualification_pass,
                "observer_independent_formation_transport_candidate": formation_transport_candidate,
                "behavior_qualified_formation_transport_candidate": (
                    behavior_qualified_formation_transport_candidate
                ),
                "formation_transport_competition_candidate": path_candidate,
                "family_prediction_gate_pass": bool(family_prediction),
                "predictive_path_candidate": predictive_path_candidate,
                "legacy_exposed_replication_pass": legacy_replication_pass,
                "strict_double_blind_gate_pass": False,
                "causal_gate_pass": False,
                "mechanism_closed": False,
                "physical": True,
                "predictive": predictive_path_candidate,
                "causal": False,
            }
        )
    return output


def cross_model_maps(mechanisms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mechanisms:
        grouped[(row["family_id"], row["mechanism_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (family, mechanism), values in sorted(grouped.items()):
        topology_counts = Counter(
            tuple(row["selected_event_topology"])
            for row in values
            if row["formation_transport_competition_candidate"]
        )
        best_topology: tuple[str, ...] | None = None
        best_count = 0
        if topology_counts:
            best_topology, best_count = topology_counts.most_common(1)[0]
        predictive_models = [
            row["model"] for row in values if row["predictive_path_candidate"]
        ]
        topology_models = [
            row["model"]
            for row in values
            if best_topology is not None
            and tuple(row["selected_event_topology"]) == best_topology
            and row["formation_transport_competition_candidate"]
        ]
        cross_model_candidate = bool(best_count >= 2 and len(predictive_models) >= 2)
        formation_transport_counts = Counter(
            tuple(row["selected_event_topology"][:2])
            for row in values
            if row["observer_independent_formation_transport_candidate"]
        )
        best_formation_transport: tuple[str, ...] | None = None
        best_formation_transport_count = 0
        if formation_transport_counts:
            best_formation_transport, best_formation_transport_count = (
                formation_transport_counts.most_common(1)[0]
            )
        formation_transport_models = [
            row["model"]
            for row in values
            if best_formation_transport is not None
            and tuple(row["selected_event_topology"][:2]) == best_formation_transport
            and row["observer_independent_formation_transport_candidate"]
        ]
        qualified_formation_transport_counts = Counter(
            tuple(row["selected_event_topology"][:2])
            for row in values
            if row["behavior_qualified_formation_transport_candidate"]
        )
        qualified_topology: tuple[str, ...] | None = None
        qualified_topology_count = 0
        if qualified_formation_transport_counts:
            qualified_topology, qualified_topology_count = (
                qualified_formation_transport_counts.most_common(1)[0]
            )
        qualified_topology_models = [
            row["model"]
            for row in values
            if qualified_topology is not None
            and tuple(row["selected_event_topology"][:2]) == qualified_topology
            and row["behavior_qualified_formation_transport_candidate"]
        ]
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase424-CrossModelEventTopology",
                "created_at": now(),
                "family_id": family,
                "mechanism_id": mechanism,
                "model_count": len(values),
                "best_typed_event_topology": list(best_topology or []),
                "topology_replication_model_count": best_count,
                "topology_replication_models": topology_models,
                "predictive_path_models": predictive_models,
                "best_formation_transport_topology": list(
                    best_formation_transport or []
                ),
                "formation_transport_topology_replication_model_count": (
                    best_formation_transport_count
                ),
                "formation_transport_topology_replication_models": (
                    formation_transport_models
                ),
                "cross_model_formation_transport_candidate": (
                    best_formation_transport_count >= 2
                ),
                "best_behavior_qualified_formation_transport_topology": list(
                    qualified_topology or []
                ),
                "behavior_qualified_formation_transport_replication_model_count": (
                    qualified_topology_count
                ),
                "behavior_qualified_formation_transport_replication_models": (
                    qualified_topology_models
                ),
                "cross_model_behavior_qualified_formation_transport_candidate": (
                    qualified_topology_count >= 2
                ),
                "cross_model_predictive_topology_candidate": cross_model_candidate,
                "strict_double_blind_gate_pass": False,
                "causal_gate_pass": False,
                "mechanism_closed": False,
            }
        )
    return output


def family_summaries(
    pair_summaries: list[dict[str, Any]],
    mechanisms: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        for family in FAMILY_MECHANISMS:
            pairs = [
                row
                for row in pair_summaries
                if row["model"] == model and row["family_id"] == family
            ]
            maps = [
                row
                for row in mechanisms
                if row["model"] == model and row["family_id"] == family
            ]
            prediction = next(
                row
                for row in prediction_rows
                if row["model"] == model and row["family_id"] == family
            )
            split_behavior = {}
            for split in (
                "discovery",
                "calibration",
                "behavior_holdout",
                "legacy_physical_holdout",
            ):
                selected = [row for row in pairs if row["split"] == split]
                split_behavior[split] = {
                    "pair_count": len(selected),
                    "both_branches_correct_fraction": clean(
                        sum(row["both_branches_correct"] for row in selected) / len(selected)
                    ),
                    "median_margin": median(row["behavior_margin_mean"] for row in selected),
                    "target_leak_fraction": clean(
                        statistics.fmean(row["target_leak_fraction"] for row in selected)
                    ),
                }
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase424-FamilySummary",
                    "created_at": now(),
                    "model": model,
                    "family_id": family,
                    "family_name": FAMILY_NAMES[family],
                    "pair_count": len(pairs),
                    "mechanism_count": len(maps),
                    "physical_path_candidate_count": sum(
                        row["formation_transport_competition_candidate"] for row in maps
                    ),
                    "predictive_path_candidate_count": sum(
                        row["predictive_path_candidate"] for row in maps
                    ),
                    "prediction_gate_pass": prediction[
                        "calibration_and_behavior_prediction_gate_pass"
                    ],
                    "behavior": split_behavior,
                    "strict_double_blind": False,
                    "closed_mechanism_count": 0,
                }
            )
    return output


def graph_for_model(model: str, mechanisms: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in mechanisms if row["model"] == model]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    family_index = {family: index for index, family in enumerate(FAMILY_MECHANISMS)}
    for mechanism_row in selected:
        family = mechanism_row["family_id"]
        mechanism = mechanism_row["mechanism_id"]
        mechanism_index = FAMILY_MECHANISMS[family].index(mechanism)
        node_ids: dict[str, str] = {}
        for signal_index, signal in enumerate(("formation", "transport", "competition")):
            evidence = mechanism_row["signals"][signal]
            depth = evidence["selected_depth_bin"]
            replication = evidence["replication"]["behavior_holdout"]
            node_id = f"phase424:{model}:{family}:{mechanism}:{signal}"
            node_ids[signal] = node_id
            nodes.append(
                {
                    "id": node_id,
                    "label": f"{FAMILY_NAMES[family]} / {mechanism} / {signal}",
                    "type": f"{signal}_event",
                    "model": model,
                    "family_id": family,
                    "mechanism_id": mechanism,
                    "event_role": signal,
                    "selected_depth_bin": depth,
                    "selected_relative_depth": (DEPTH_ORDER[depth] + 0.5) / 3.0,
                    "calibration_gate_pass": evidence["replication"]["calibration"][
                        "gate_pass"
                    ],
                    "behavior_holdout_gate_pass": replication["gate_pass"],
                    "behavior_holdout_median": replication["median"],
                    "behavior_holdout_positive_fraction": replication["positive_fraction"],
                    "legacy_replication_gate_pass": evidence["replication"][
                        "legacy_physical_holdout"
                    ]["gate_pass"],
                    "physical": True,
                    "observer": signal == "competition",
                    "native_compute_event": signal == "transport",
                    "compute_edge": False,
                    "predictive": mechanism_row["predictive_path_candidate"],
                    "causal": False,
                    "strict_double_blind": False,
                    "evidence_level": (
                        "readout_observer"
                        if signal == "competition"
                        else "physical_observation"
                    ),
                    "score": replication["positive_fraction"],
                    "size": 0.55 + 0.45 * replication["positive_fraction"],
                    "color": COLORS[signal],
                    "position": [
                        family_index[family] * 10.0,
                        (DEPTH_ORDER[depth] + 0.5) * 10.0,
                        mechanism_index * 5.0 + signal_index * 1.5,
                    ],
                    "show_label": signal == "formation",
                }
            )
        for source, target in (("formation", "transport"), ("transport", "competition")):
            edges.append(
                {
                    "id": f"{node_ids[source]}->{node_ids[target]}",
                    "source": node_ids[source],
                    "target": node_ids[target],
                    "type": "observed_event_order",
                    "physical": True,
                    "observer": target == "competition",
                    "compute_edge": False,
                    "predictive": mechanism_row["predictive_path_candidate"],
                    "causal": False,
                    "strict_double_blind": False,
                    "evidence_level": (
                        "observer_order"
                        if target == "competition"
                        else "physical_order"
                    ),
                    "color": "#64748b",
                    "weight": 1.0,
                }
            )
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase424-GlobalPhysicalPathAtlas",
        "title": f"Phase424 {model} 九族形成—运输—竞争物理图",
        "model": model,
        "evidence_scope": (
            "previously exposed semantic census; legal attention writes plus observer overlays; "
            "no strict blind, causal, workspace, or neuron closure"
        ),
        "graph": {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "phase": 424,
                "family_count": 9,
                "mechanism_count": 72,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "strict_double_blind": False,
                "compute_edge_count": 0,
                "causal": False,
            },
        },
    }


def publish_visual(mechanisms: list[dict[str, Any]]) -> None:
    VIS.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase424_{model}_global_physical_path.json"
        write_json(VIS / filename, graph_for_model(model, mechanisms))
        items.append(
            {
                "id": f"phase424_{model}_global_physical_path",
                "label": f"Phase424 {model} 九族形成—运输—竞争物理图",
                "filename": filename,
                "model": model,
                "phase": 424,
                "evidence_scope": (
                    "legacy-exposed denominator; physical observation plus readout observer; "
                    "non-causal and not strictly blind"
                ),
            }
        )
    write_json(
        VIS / "manifest.json",
        {
            "schema_version": "phase424_global_physical_path_manifest.v1",
            "generated_at": now(),
            "default_item_id": items[0]["id"],
            "items": items,
        },
    )
    registry = read_json(REGISTRY)
    source_id = "gpt5_phase424_global_physical_path_atlas"
    entry = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase424 九族形成—运输—竞争物理图谱",
        "description": "三模型九族72机制的来源形成、合法注意力运输、查询竞争和留出预测审计。",
        "manifest_path": "/vis_data/phase424_global_physical_path_atlas/manifest.json",
        "manifest_schema": "phase424_global_physical_path_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase424_global_physical_path_atlas",
        "models": list(MODELS),
        "evidence_scope": "旧分母全量物理复核；合法计算边与观察器覆盖并存；非双盲、非因果、非神经元闭合",
        "color": "#10b981",
    }
    existing = [row for row in registry["sources"] if row["id"] != source_id]
    registry["sources"] = [*existing, entry]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def markdown_report(summary: dict[str, Any]) -> str:
    model_lines = []
    for model in MODELS:
        row = summary["models"][model]
        model_lines.append(
            f"| {model} | {row['branch_correct_pair_count']}/864 | "
            f"{row['observer_independent_formation_transport_candidate_count']}/72 | "
            f"{row['behavior_qualified_formation_transport_candidate_count']}/72 | "
            f"{row['physical_path_candidate_count']}/72 | "
            f"{row['predictive_path_candidate_count']}/72 | "
            f"{row['prediction_family_gate_count']}/9 |"
        )
    return "\n".join(
        [
            "# Phase424 九族形成—运输—竞争物理图谱审计",
            "",
            f"生成时间：{summary['created_at']}",
            "",
            "## 客观分母",
            "",
            "- 9 个语言模式族，72 个机制。",
            "- 每机制 24 个既有语义条目，配成 12 个不同答案对象对。",
            "- 每模型 1,728 个条件；三模型共 5,184 个条件。",
            "- 全层记录来源状态、真实注意力来源写入、查询状态和同层 MLP 重写。",
            "- 不保存全量神经元激活。",
            "",
            "## 结果",
            "",
            "| 模型 | 两分支均正确对象对 | 原始形成—运输 | 行为合格形成—运输 | 三段候选 | 带留出预测候选 | 预测通过族 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *model_lines,
            "",
            f"跨模型预测拓扑候选：{summary['cross_model_predictive_topology_candidate_count']}/72。",
            f"跨模型形成—运输拓扑候选：{summary['cross_model_formation_transport_candidate_count']}/72。",
            f"跨模型行为合格形成—运输拓扑候选：{summary['cross_model_behavior_qualified_formation_transport_candidate_count']}/72。",
            "",
            "## 证据边界",
            "",
            "注意力概率、值状态和输出投影构成合法计算边；目标读出方向和岭回归只属于观察器。",
            "Phase330 语义条目此前已经暴露，因此最后一段只是旧分母复核，不是新的双盲物理留出。",
            "本阶段没有干预，也没有保存或扫描单神经元，严格机制闭合仍为 0/72。",
            "",
            "## 结论",
            "",
            summary["conclusion"],
            "",
        ]
    )


def main() -> None:
    protocol = read_json(OUT / "phase424_protocol.json")
    pair_summaries: list[dict[str, Any]] = []
    pair_layers: list[dict[str, Any]] = []
    collection: dict[str, Any] = {}
    for model in MODELS:
        model_root = OUT / "models" / model
        complete = read_json(model_root / "phase424_collection_complete.json")
        if not complete["all_rows_complete"]:
            raise RuntimeError(f"Incomplete Phase424 collection for {model}")
        collection[model] = complete
        pair_summaries.extend(read_jsonl(model_root / "phase424_pair_summary_rows.jsonl"))
        pair_layers.extend(read_jsonl(model_root / "phase424_pair_layer_rows.jsonl"))

    predictions = prediction_audits(pair_summaries)
    mechanisms = mechanism_maps(pair_layers, predictions)
    cross_model = cross_model_maps(mechanisms)
    families = family_summaries(pair_summaries, mechanisms, predictions)

    model_summary: dict[str, Any] = {}
    for model in MODELS:
        model_pairs = [row for row in pair_summaries if row["model"] == model]
        model_maps = [row for row in mechanisms if row["model"] == model]
        model_predictions = [row for row in predictions if row["model"] == model]
        model_summary[model] = {
            "layer_count": collection[model]["layer_count"],
            "condition_count": collection[model]["condition_count"],
            "pair_count": collection[model]["pair_count"],
            "branch_correct_pair_count": sum(row["both_branches_correct"] for row in model_pairs),
            "physical_path_candidate_count": sum(
                row["formation_transport_competition_candidate"] for row in model_maps
            ),
            "observer_independent_formation_transport_candidate_count": sum(
                row["observer_independent_formation_transport_candidate"]
                for row in model_maps
            ),
            "behavior_qualified_formation_transport_candidate_count": sum(
                row["behavior_qualified_formation_transport_candidate"]
                for row in model_maps
            ),
            "predictive_path_candidate_count": sum(
                row["predictive_path_candidate"] for row in model_maps
            ),
            "prediction_family_gate_count": sum(
                row["calibration_and_behavior_prediction_gate_pass"]
                for row in model_predictions
            ),
            "max_component_ledger_relative_error": collection[model][
                "max_component_ledger_relative_error"
            ],
            "elapsed_seconds": collection[model]["elapsed_seconds"],
        }

    cross_count = sum(
        row["cross_model_predictive_topology_candidate"] for row in cross_model
    )
    cross_formation_transport_count = sum(
        row["cross_model_formation_transport_candidate"] for row in cross_model
    )
    cross_behavior_qualified_formation_transport_count = sum(
        row["cross_model_behavior_qualified_formation_transport_candidate"]
        for row in cross_model
    )
    any_predictive = sum(
        row["predictive_path_candidate"] for row in mechanisms
    )
    if any_predictive == 0:
        conclusion = (
            "全层物理脉络已按统一分母完成采集，但冻结物理特征没有在校准和行为留出上形成稳定增量预测；"
            "因此这些轨迹只能作为分布拼图，不能登记为功能机制。"
        )
    elif cross_count == 0:
        conclusion = (
            "部分模型内路径通过冻结预测门，但没有形成至少两个模型共享的有类型事件拓扑；"
            "它们仍是模型内候选，不是共同语言算法。"
        )
    else:
        conclusion = (
            "至少一个机制出现跨模型预测拓扑候选，但旧分母暴露和无干预两项硬门仍失败；"
            "结果只能授权真正新双盲语义组上的块级验证。"
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase424-GlobalPhysicalPathAnalysis",
        "created_at": now(),
        "protocol_valid": protocol["validation"]["valid"],
        "models": model_summary,
        "pair_summary_count": len(pair_summaries),
        "pair_layer_row_count": len(pair_layers),
        "prediction_audit_count": len(predictions),
        "mechanism_map_count": len(mechanisms),
        "cross_model_map_count": len(cross_model),
        "family_summary_count": len(families),
        "cross_model_predictive_topology_candidate_count": cross_count,
        "cross_model_formation_transport_candidate_count": (
            cross_formation_transport_count
        ),
        "cross_model_behavior_qualified_formation_transport_candidate_count": (
            cross_behavior_qualified_formation_transport_count
        ),
        "strict_double_blind_mechanism_count": 0,
        "causally_closed_mechanism_count": 0,
        "strict_mechanism_closure": "0/72",
        "overall_scientific_progress_percent": 21,
        "progress_interval_percent": [18, 24],
        "conclusion": conclusion,
        "observer_jacobian_overlay": {
            "source_phase": 423,
            "qualified_models": ["glm4"],
            "used_as_compute_edge": False,
            "used_as_cross_model_coordinate": False,
        },
        "hard_limits": [
            "Phase330 semantic items were previously exposed; legacy holdout is not double blind.",
            "Target-versus-opposite readout directions are observers, not native compute edges.",
            "Frozen ridge prediction is an audit and cannot establish mediation or causality.",
            "No head, channel, or neuron intervention was executed.",
            "Small-model topology may differ materially from larger language models.",
        ],
    }
    write_jsonl(OUT / "phase424_prediction_audit.jsonl", predictions)
    write_jsonl(OUT / "phase424_mechanism_maps.jsonl", mechanisms)
    write_jsonl(OUT / "phase424_cross_model_maps.jsonl", cross_model)
    write_jsonl(OUT / "phase424_family_summaries.jsonl", families)
    write_json(OUT / "phase424_global_summary.json", summary)
    (OUT / "phase424_report.md").write_text(markdown_report(summary), encoding="utf-8")
    publish_visual(mechanisms)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
