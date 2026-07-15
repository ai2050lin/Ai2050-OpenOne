#!/usr/bin/env python3
"""Analyze Phase426 without reading the sealed split before authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase426_exact_position_protocol import (  # noqa: E402
    BLOCKS,
    MODELS,
    OUT,
    SCHEMA_VERSION,
)


PHASE_ID = "Phase426-ExactPositionRoleAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase426_exact_position_role_validation"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
DEPTHS = ("early", "middle", "late")
DEPTH_ORDER = {value: index for index, value in enumerate(DEPTHS)}
SIGNALS = {
    "formation": "formation_exact_specificity",
    "transport": "transport_exact_specificity",
    "competition": "competition_specificity",
}
BASELINE_FEATURES = (
    "executed_token_count_mean",
    "target_sequence_token_count_mean",
)
TARGETS = (
    "early_role_teacher_sequence_margin_mean",
    "early_role_natural_target_fraction",
)
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
        raise RuntimeError(f"Phase426 non-finite scalar: {value}")
    return round(float(value), 10)


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.fmean(rows)) if rows else 0.0


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def positive_fraction(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(sum(value > 0 for value in rows) / len(rows)) if rows else 0.0


def fit_ridge(
    rows: list[dict[str, Any]], features: tuple[str, ...], target: str
) -> dict[str, Any]:
    x = np.asarray(
        [[float(row[key]) for key in features] for row in rows], dtype=np.float64
    )
    y = np.asarray([float(row[target]) for row in rows], dtype=np.float64)
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    z = (x - x_mean) / x_scale
    y_mean = float(y.mean())
    coefficient = np.linalg.solve(
        z.T @ z + np.eye(z.shape[1], dtype=np.float64),
        z.T @ (y - y_mean),
    )
    return {
        "features": list(features),
        "target": target,
        "mean": x_mean.tolist(),
        "scale": x_scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": y_mean,
        "ridge_alpha": 1.0,
    }


def prediction_metrics(model: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = tuple(model["features"])
    target = str(model["target"])
    x = np.asarray(
        [[float(row[key]) for key in features] for row in rows], dtype=np.float64
    )
    y = np.asarray([float(row[target]) for row in rows], dtype=np.float64)
    estimate = float(model["intercept"]) + (
        (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    ) @ np.asarray(model["coefficient"])
    residual = float(np.square(y - estimate).sum())
    total = float(np.square(y - y.mean()).sum())
    r2 = None if total <= 1e-12 else clean(1.0 - residual / total)
    return {
        "count": len(rows),
        "r2": r2,
        "mae": clean(float(np.abs(y - estimate).mean())),
        "truth_mean": clean(float(y.mean())),
        "prediction_mean": clean(float(estimate.mean())),
    }


def add_selected_features(
    rows: list[dict[str, Any]], selected_depths: dict[str, str]
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        for signal, feature in SIGNALS.items():
            item[f"selected_{signal}"] = float(
                row[f"{selected_depths[signal]}_{feature}_median"]
            )
        output.append(item)
    return output


def select_depths(
    candidate_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    selected: dict[str, str] = {}
    gaps: dict[str, dict[str, float]] = {}
    candidate_discovery = [row for row in candidate_rows if row["split"] == "discovery"]
    control_discovery = [row for row in control_rows if row["split"] == "discovery"]
    for signal, feature in SIGNALS.items():
        gaps[signal] = {
            depth: clean(
                median(
                    row[f"{depth}_{feature}_median"] for row in candidate_discovery
                )
                - median(row[f"{depth}_{feature}_median"] for row in control_discovery)
            )
            for depth in DEPTHS
        }
        selected[signal] = max(
            DEPTHS, key=lambda depth: (gaps[signal][depth], -DEPTH_ORDER[depth])
        )
    return selected, gaps


def split_signal_audit(
    candidate_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    split: str,
    signal: str,
    depth: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    feature = SIGNALS[signal]
    candidates = [row for row in candidate_rows if row["split"] == split]
    controls = [row for row in control_rows if row["split"] == split]
    candidate_values = [float(row[f"{depth}_{feature}_median"]) for row in candidates]
    control_values = [float(row[f"{depth}_{feature}_median"]) for row in controls]
    control_median = median(control_values)
    exact_gate = bool(
        median(candidate_values) > float(thresholds["exact_position_specificity_min"])
        and median(candidate_values) - control_median
        > float(thresholds["candidate_minus_control_min"])
        and mean(value > control_median for value in candidate_values)
        >= float(thresholds["positive_group_fraction_min"])
    )
    extra: dict[str, Any] = {}
    if signal in {"formation", "transport"}:
        prefix = signal
        covariance = median(
            row[f"{depth}_{prefix}_role_covariance_median"] for row in candidates
        )
        conditional = median(
            row[f"{depth}_{prefix}_conditional_covariance_median"]
            for row in candidates
        )
        signal_ratio = median(
            row[f"{depth}_{prefix}_replica_signal_ratio_median"] for row in candidates
        )
        extra = {
            "role_covariance_median": covariance,
            "conditional_covariance_median": conditional,
            "replica_signal_ratio_median": signal_ratio,
            "role_covariance_gate_pass": covariance
            >= float(thresholds["role_covariance_min"]),
            "conditional_covariance_gate_pass": conditional
            >= float(thresholds["conditional_covariance_min"]),
            "replica_signal_ratio_gate_pass": signal_ratio
            > float(thresholds["replica_signal_ratio_min"]),
        }
        exact_gate = bool(
            exact_gate
            and extra["role_covariance_gate_pass"]
            and extra["conditional_covariance_gate_pass"]
            and extra["replica_signal_ratio_gate_pass"]
        )
    else:
        covariance = median(
            row[f"{depth}_competition_role_covariance_median"] for row in candidates
        )
        extra = {
            "role_covariance_median": covariance,
            "role_covariance_gate_pass": covariance
            >= float(thresholds["role_covariance_min"]),
        }
        exact_gate = bool(exact_gate and extra["role_covariance_gate_pass"])
    return {
        "split": split,
        "independent_candidate_group_count": len(candidates),
        "independent_control_group_count": len(controls),
        "candidate_median": median(candidate_values),
        "control_median": control_median,
        "candidate_minus_control": clean(median(candidate_values) - control_median),
        "candidate_above_control_median_fraction": mean(
            value > control_median for value in candidate_values
        ),
        **extra,
        "gate_pass": exact_gate,
    }


def prediction_audit(
    rows: list[dict[str, Any]],
    selected_depths: dict[str, str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    enriched = add_selected_features(rows, selected_depths)
    discovery = [row for row in enriched if row["split"] == "discovery"]
    physical_features = BASELINE_FEATURES + tuple(
        f"selected_{signal}" for signal in SIGNALS
    )
    targets: dict[str, Any] = {}
    overall_gate = True
    for target in TARGETS:
        baseline = fit_ridge(discovery, BASELINE_FEATURES, target)
        physical = fit_ridge(discovery, physical_features, target)
        split_results: dict[str, Any] = {}
        for split in ("calibration", "behavior_holdout"):
            split_rows = [row for row in enriched if row["split"] == split]
            baseline_metrics = prediction_metrics(baseline, split_rows)
            physical_metrics = prediction_metrics(physical, split_rows)
            delta_r2 = None
            if baseline_metrics["r2"] is not None and physical_metrics["r2"] is not None:
                delta_r2 = clean(
                    float(physical_metrics["r2"]) - float(baseline_metrics["r2"])
                )
            mae_gain = clean(
                float(baseline_metrics["mae"]) - float(physical_metrics["mae"])
            )
            gate = bool(
                physical_metrics["r2"] is not None
                and float(physical_metrics["r2"]) > float(thresholds["prediction_r2_min"])
                and delta_r2 is not None
                and delta_r2 > float(thresholds["prediction_delta_r2_min"])
                and mae_gain > float(thresholds["prediction_mae_gain_min"])
            )
            split_results[split] = {
                "baseline": baseline_metrics,
                "physical": physical_metrics,
                "delta_r2": delta_r2,
                "mae_gain": mae_gain,
                "gate_pass": gate,
            }
            overall_gate = overall_gate and gate
        targets[target] = {
            "baseline_model": baseline,
            "physical_model": physical,
            "split_results": split_results,
            "gate_pass": all(value["gate_pass"] for value in split_results.values()),
        }
    return {
        "effective_independent_fit_count": len(discovery),
        "targets": targets,
        "gate_pass": bool(overall_gate),
    }


def candidate_audit(
    model: str,
    block: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    selected, discovery_gaps = select_depths(candidate_rows, control_rows)
    signals: dict[str, Any] = {}
    for signal in SIGNALS:
        split_results = {
            split: split_signal_audit(
                candidate_rows,
                control_rows,
                split,
                signal,
                selected[signal],
                thresholds,
            )
            for split in ("discovery", "calibration", "behavior_holdout")
        }
        signals[signal] = {
            "feature": SIGNALS[signal],
            "selected_depth": selected[signal],
            "discovery_candidate_minus_control_by_depth": discovery_gaps[signal],
            "split_results": split_results,
            "calibration_and_behavior_gate_pass": bool(
                split_results["calibration"]["gate_pass"]
                and split_results["behavior_holdout"]["gate_pass"]
            ),
        }
    identity_results = {}
    for split in ("calibration", "behavior_holdout"):
        rows = [row for row in candidate_rows if row["split"] == split]
        values = [
            float(
                row[
                    f"{selected['transport']}_source_to_write_identity_alignment_median"
                ]
            )
            for row in rows
        ]
        identity_results[split] = {
            "median": median(values),
            "gate_pass": median(values)
            >= float(thresholds["source_to_write_identity_alignment_min"]),
        }
    behavior_results = {}
    for split in ("calibration", "behavior_holdout"):
        rows = [row for row in candidate_rows if row["split"] == split]
        teacher = mean(row["early_role_teacher_sequence_correct_fraction"] for row in rows)
        natural = mean(row["early_role_natural_target_fraction"] for row in rows)
        behavior_results[split] = {
            "independent_group_count": len(rows),
            "teacher_sequence_correct_fraction": teacher,
            "natural_target_fraction": natural,
            "teacher_gate_pass": teacher
            >= float(thresholds["teacher_sequence_correct_fraction_min"]),
            "natural_gate_pass": natural
            >= float(thresholds["natural_target_fraction_min"]),
            "gate_pass": bool(
                teacher >= float(thresholds["teacher_sequence_correct_fraction_min"])
                and natural >= float(thresholds["natural_target_fraction_min"])
            ),
        }
    prediction = prediction_audit(candidate_rows, selected, thresholds)
    partial_order = bool(
        DEPTH_ORDER[selected["formation"]]
        <= DEPTH_ORDER[selected["transport"]]
        <= DEPTH_ORDER[selected["competition"]]
    )
    identity_gate = all(row["gate_pass"] for row in identity_results.values())
    behavior_gate = all(row["gate_pass"] for row in behavior_results.values())
    path_gate = bool(
        all(
            signals[signal]["calibration_and_behavior_gate_pass"]
            for signal in SIGNALS
        )
        and identity_gate
        and behavior_gate
        and prediction["gate_pass"]
        and partial_order
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase426-OpenCandidateAudit",
        "created_at": now(),
        "model": model,
        "block_id": block["block_id"],
        "family_id": block["family_id"],
        "mechanism_id": block["mechanism_id"],
        "candidate": True,
        "matched_control_block_id": block["matched_control_block_id"],
        "independent_discovery_group_count": sum(
            row["split"] == "discovery" for row in candidate_rows
        ),
        "selected_depths": selected,
        "signals": signals,
        "source_to_write_identity": identity_results,
        "identity_map_gate_pass": identity_gate,
        "behavior": behavior_results,
        "behavior_gate_pass": behavior_gate,
        "prediction": prediction,
        "partial_order_gate_pass": partial_order,
        "open_path_gate_pass": path_gate,
        "role_dominance_descriptive_only": {
            signal: {
                split: median(
                    row[
                        f"{selected[signal]}_{signal}_role_dominance_median"
                    ]
                    for row in candidate_rows
                    if row["split"] == split
                )
                for split in ("discovery", "calibration", "behavior_holdout")
            }
            for signal in ("formation", "transport")
        },
        "sealed_tested": False,
        "causal": False,
    }


def cross_model_audits(
    audits: list[dict[str, Any]], cross_model_minimum: int
) -> list[dict[str, Any]]:
    output = []
    for block in (row for row in BLOCKS if row["candidate"]):
        rows = [row for row in audits if row["block_id"] == block["block_id"]]
        by_topology: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for row in rows:
            if row["open_path_gate_pass"]:
                topology = tuple(row["selected_depths"][signal] for signal in SIGNALS)
                by_topology[topology].append(row["model"])
        best_topology = None
        best_models: list[str] = []
        for topology, models in by_topology.items():
            if len(models) > len(best_models):
                best_topology, best_models = topology, sorted(models)
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase426-CrossModelOpenAudit",
                "created_at": now(),
                "block_id": block["block_id"],
                "family_id": block["family_id"],
                "mechanism_id": block["mechanism_id"],
                "matched_control_block_id": block["matched_control_block_id"],
                "open_path_models": sorted(
                    row["model"] for row in rows if row["open_path_gate_pass"]
                ),
                "best_topology": list(best_topology) if best_topology else None,
                "best_topology_models": best_models,
                "cross_model_open_gate_pass": len(best_models) >= cross_model_minimum,
                "sealed_tested": False,
                "causal": False,
            }
        )
    return output


def instrument_audit() -> dict[str, Any]:
    protocol = read_json(OUT / "phase426_protocol.json")
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if implementation_hash != protocol["implementation_commitments"][Path(__file__).name]:
        raise RuntimeError("Phase426 analysis changed after protocol freeze")
    model_rows = {}
    for model in MODELS:
        root = OUT / "models" / model / "instrument"
        complete = read_json(root / "phase426_collection_complete.json")
        behavior = read_jsonl(root / "phase426_behavior_rows.jsonl")
        summaries = read_jsonl(root / "phase426_group_summary_rows.jsonl")
        finite_behavior = all(
            math.isfinite(float(row[key]))
            for row in behavior
            for key in (
                "target_sequence_logprob",
                "opposite_sequence_logprob",
                "teacher_sequence_logprob_margin",
            )
        )
        parser_complete = all(
            all(
                key in row
                for key in (
                    "natural_target_first",
                    "natural_revision",
                    "natural_boundary",
                    "natural_stop",
                    "natural_censoring",
                )
            )
            for row in behavior
        )
        gate = bool(
            complete["all_rows_complete"]
            and complete["component_ledger_gate_pass"]
            and len(behavior) == 256
            and len(summaries) == 8
            and finite_behavior
            and parser_complete
        )
        model_rows[model] = {
            "condition_count": len(behavior),
            "independent_group_count": len(summaries),
            "max_component_ledger_relative_error": complete[
                "max_component_ledger_relative_error"
            ],
            "finite_full_sequence_scores": finite_behavior,
            "natural_parser_complete": parser_complete,
            "gate_pass": gate,
        }
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase426-InstrumentAudit",
        "created_at": now(),
        "protocol_valid": protocol["validation"]["valid"],
        "position_counterfactual_mismatch_count": protocol["validation"][
            "position_counterfactual_mismatch_count"
        ],
        "model_results": model_rows,
        "instrument_gate_pass": bool(
            protocol["validation"]["position_counterfactual_mismatch_count"] == 0
            and all(row["gate_pass"] for row in model_rows.values())
        ),
        "thresholds_or_theory_updated": False,
    }
    write_json(OUT / "phase426_instrument_audit.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def graph_for_model(model: str, audits: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in audits if row["model"] == model]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for block_index, row in enumerate(rows):
        node_ids = {}
        for signal_index, signal in enumerate(SIGNALS):
            depth = row["selected_depths"][signal]
            signal_gate = row["signals"][signal][
                "calibration_and_behavior_gate_pass"
            ]
            node_id = f"phase426:{model}:{row['block_id']}:{signal}"
            node_ids[signal] = node_id
            nodes.append(
                {
                    "id": node_id,
                    "label": f"{row['mechanism_id']} / {signal}",
                    "type": f"{signal}_event",
                    "model": model,
                    "block_id": row["block_id"],
                    "family_id": row["family_id"],
                    "mechanism_id": row["mechanism_id"],
                    "candidate": True,
                    "event_role": signal,
                    "selected_depth_bin": depth,
                    "selected_relative_depth": (DEPTH_ORDER[depth] + 0.5) / 3,
                    "exact_position_gate_pass": signal_gate,
                    "matched_control_gate_pass": signal_gate,
                    "open_path_gate_pass": row["open_path_gate_pass"],
                    "physical": True,
                    "observer": signal == "competition",
                    "native_compute_event": signal == "transport",
                    "compute_edge": False,
                    "predictive": row["prediction"]["gate_pass"],
                    "causal": False,
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "evidence_level": "exact_position_open_observation",
                    "score": 1.0 if signal_gate else 0.2,
                    "size": 1.0 if signal_gate else 0.65,
                    "color": COLORS[signal],
                    "position": [
                        block_index * 12.0,
                        (DEPTH_ORDER[depth] + 0.5) * 10.0,
                        signal_index * 4.0,
                    ],
                    "show_label": True,
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
                    "predictive": row["prediction"]["gate_pass"],
                    "causal": False,
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "evidence_level": "open_observational_order",
                    "color": "#64748b",
                    "weight": 1.0,
                }
            )
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase426-ExactPositionRoleAtlas",
        "title": f"Phase426 {model} 精确同位置角色图谱",
        "model": model,
        "evidence_scope": (
            "independent open denominator with exact source/query token positions and "
            "matched negative controls; sealed and causal gates remain explicit"
        ),
        "graph": {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "phase": 426,
                "candidate_block_count": 2,
                "open_only": True,
                "pipeline_sealed": False,
                "strict_double_blind": False,
                "compute_edge_count": 0,
                "causal": False,
            },
        },
    }


def publish_visual(audits: list[dict[str, Any]]) -> None:
    VIS.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase426_{model}_exact_position_role.json"
        write_json(VIS / filename, graph_for_model(model, audits))
        items.append(
            {
                "id": f"phase426_{model}_exact_position_role",
                "label": f"Phase426 {model} 精确同位置角色图谱",
                "filename": filename,
                "model": model,
                "phase": 426,
                "evidence_scope": "independent open exact-position denominator; non-causal",
            }
        )
    write_json(
        VIS / "manifest.json",
        {
            "schema_version": "phase426_exact_position_role_manifest.v1",
            "generated_at": now(),
            "default_item_id": items[0]["id"],
            "items": items,
        },
    )
    registry = read_json(REGISTRY)
    entry = {
        "id": "gpt5_phase426_exact_position_role_validation",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase426 精确同位置角色验证",
        "description": "同一来源与查询词元索引下的角色前置/后置、匹配负对照、完整序列与自然短生成图谱。",
        "manifest_path": "/vis_data/phase426_exact_position_role_validation/manifest.json",
        "manifest_schema": "phase426_exact_position_role_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase426_exact_position_role_validation",
        "models": list(MODELS),
        "evidence_scope": "独立开放集精确同位置观测；密封、因果与神经元门保持显式",
        "color": "#10b981",
    }
    registry["sources"] = [
        row for row in registry["sources"] if row["id"] != entry["id"]
    ] + [entry]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def analyze_preseal() -> dict[str, Any]:
    protocol = read_json(OUT / "phase426_protocol.json")
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if implementation_hash != protocol["implementation_commitments"][Path(__file__).name]:
        raise RuntimeError("Phase426 analysis changed after protocol freeze")
    instrument = read_json(OUT / "phase426_instrument_audit.json")
    if not instrument["instrument_gate_pass"]:
        raise RuntimeError("Phase426 instrument gate did not pass")
    thresholds = protocol["registered_thresholds"]
    all_rows: dict[str, list[dict[str, Any]]] = {}
    model_completes = {}
    for model in MODELS:
        root = OUT / "models" / model / "open"
        complete = read_json(root / "phase426_collection_complete.json")
        if not complete["all_rows_complete"] or not complete["component_ledger_gate_pass"]:
            raise RuntimeError(f"Incomplete Phase426 open collection for {model}")
        rows = read_jsonl(root / "phase426_group_summary_rows.jsonl")
        if len(rows) != 384:
            raise RuntimeError(f"Expected 384 open groups for {model}, got {len(rows)}")
        all_rows[model] = rows
        model_completes[model] = complete
    audits: list[dict[str, Any]] = []
    for model in MODELS:
        by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in all_rows[model]:
            by_block[row["block_id"]].append(row)
        for block in (row for row in BLOCKS if row["candidate"]):
            candidate_rows = by_block[block["block_id"]]
            control_rows = by_block[block["matched_control_block_id"]]
            if len(candidate_rows) != 96 or len(control_rows) != 96:
                raise RuntimeError(f"Phase426 group denominator mismatch for {model}/{block['block_id']}")
            audits.append(
                candidate_audit(
                    model,
                    block,
                    candidate_rows,
                    control_rows,
                    thresholds,
                )
            )
    cross = cross_model_audits(
        audits, int(thresholds["cross_model_replication_min"])
    )
    unlock_blocks = [
        row["block_id"] for row in cross if row["cross_model_open_gate_pass"]
    ]
    gate_freeze = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase426-GateFreeze",
        "created_at": now(),
        "open_audit_count": len(audits),
        "cross_model_audit_count": len(cross),
        "sealed_unlock": bool(unlock_blocks),
        "sealed_unlock_blocks": unlock_blocks,
        "selected_depths_by_model_block": {
            f"{row['model']}::{row['block_id']}": row["selected_depths"]
            for row in audits
        },
        "prediction_models_by_model_block": {
            f"{row['model']}::{row['block_id']}": row["prediction"]
            for row in audits
        },
        "transport_map": "identity_only",
        "thresholds": thresholds,
        "prohibited_after_freeze": [
            "depth_change",
            "feature_change",
            "threshold_change",
            "map_change",
            "predictor_refit",
            "sample_append",
        ],
        "causal_unlock": False,
        "head_channel_neuron_scan_allowed": False,
    }
    write_jsonl(OUT / "phase426_open_candidate_audits.jsonl", audits)
    write_jsonl(OUT / "phase426_cross_model_open_audits.jsonl", cross)
    write_json(OUT / "phase426_gate_freeze.json", gate_freeze)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "preseal",
        "protocol_valid": protocol["validation"]["valid"],
        "instrument_gate_pass": instrument["instrument_gate_pass"],
        "registered_formal_condition_count": protocol["validation"][
            "formal_condition_count"
        ],
        "executed_instrument_condition_count": sum(
            row["condition_count"] for row in instrument["model_results"].values()
        ),
        "executed_open_condition_count": sum(
            int(row["condition_count"]) for row in model_completes.values()
        ),
        "executed_sealed_condition_count": 0,
        "exact_position_mismatch_count": protocol["validation"][
            "position_counterfactual_mismatch_count"
        ],
        "open_path_gate_count_by_model": {
            model: sum(
                row["open_path_gate_pass"] for row in audits if row["model"] == model
            )
            for model in MODELS
        },
        "cross_model_open_candidate_count": len(unlock_blocks),
        "sealed_unlock": bool(unlock_blocks),
        "sealed_unlock_blocks": unlock_blocks,
        "sealed_tested": False,
        "causal_tested": False,
        "strict_human_double_blind": False,
        "strict_mechanism_closure": "0/72",
        "overall_scientific_progress_percent": 21,
        "progress_interval_percent": [18, 24],
        "conclusion": (
            "Frozen open gates authorize a pipeline-sealed physical audit."
            if unlock_blocks
            else "No candidate passed exact-position, matched-control, full-event and cross-model gates; sealed and causal stages remain closed."
        ),
    }
    write_json(OUT / "phase426_global_summary.json", summary)
    publish_visual(audits)
    report = [
        "# Phase426 精确同位置角色分量预密封审计",
        "",
        f"- 正式注册条件：{summary['registered_formal_condition_count']}",
        f"- 仪器条件：{summary['executed_instrument_condition_count']}",
        f"- 开放条件：{summary['executed_open_condition_count']}",
        f"- 位置不匹配：{summary['exact_position_mismatch_count']}",
        f"- 跨模型完整候选：{summary['cross_model_open_candidate_count']}",
        f"- 密封解锁：{summary['sealed_unlock']}",
        "- 教师强制完整序列与自然短生成分别计账。",
        "- 角色主导只作描述，不参与角色分量关闭。",
        "- 合法来源写入仍不是因果边；神经元扫描保持禁止。",
        "",
        summary["conclusion"],
    ]
    (OUT / "phase426_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("instrument", "preseal"), required=True)
    args = parser.parse_args()
    if args.stage == "instrument":
        instrument_audit()
    else:
        analyze_preseal()


if __name__ == "__main__":
    main()
