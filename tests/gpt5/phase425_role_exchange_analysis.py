#!/usr/bin/env python3
"""Analyze Phase425 open data, freeze gates, and optionally audit sealed data."""

from __future__ import annotations

import argparse
import hashlib
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

from phase425_role_exchange_protocol import BLOCKS, MODELS, OUT, SCHEMA_VERSION  # noqa: E402


PHASE_ID = "Phase425-RoleExchangeAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase425_role_exchange_validation"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
DEPTHS = ("early", "middle", "late")
DEPTH_ORDER = {value: index for index, value in enumerate(DEPTHS)}
SIGNALS = {
    "formation": "formation_specificity",
    "transport": "transport_specificity",
    "competition": "competition_specificity",
}
BASELINE_FEATURES = (
    "executed_token_count_mean",
    "source_token_count_mean",
    "query_token_count_mean",
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
        raise RuntimeError(f"Phase425 non-finite scalar: {value}")
    return round(float(value), 10)


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def positive_fraction(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(sum(value > 0 for value in rows) / len(rows)) if rows else 0.0


def fit_ridge(rows: list[dict[str, Any]], features: tuple[str, ...]) -> dict[str, Any]:
    x = np.asarray([[float(row[key]) for key in features] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["behavior_margin_mean"]) for row in rows], dtype=np.float64)
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
        "mean": x_mean.tolist(),
        "scale": x_scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": y_mean,
        "ridge_alpha": 1.0,
    }


def prediction_metrics(model: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = tuple(model["features"])
    x = np.asarray([[float(row[key]) for key in features] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["behavior_margin_mean"]) for row in rows], dtype=np.float64)
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
        "sign_accuracy": clean(float(((y > 0) == (estimate > 0)).mean())),
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


def signal_audit(
    rows: list[dict[str, Any]],
    signal: str,
    thresholds: dict[str, Any],
    feature_override: str | None = None,
) -> dict[str, Any]:
    feature = feature_override or SIGNALS[signal]
    discovery = [row for row in rows if row["split"] == "discovery"]
    discovery_medians = {
        depth: median(row[f"{depth}_{feature}_median"] for row in discovery)
        for depth in DEPTHS
    }
    selected_depth = max(
        DEPTHS,
        key=lambda depth: (discovery_medians[depth], -DEPTH_ORDER[depth]),
    )
    split_results: dict[str, Any] = {}
    for split in ("discovery", "calibration", "behavior_holdout"):
        values = [
            float(row[f"{selected_depth}_{feature}_median"])
            for row in rows
            if row["split"] == split
        ]
        gate = bool(
            median(values) > float(thresholds["signal_median_min"])
            and positive_fraction(values)
            >= float(thresholds["signal_positive_fraction_min"])
        )
        split_results[split] = {
            "independent_replica_group_count": len(values),
            "median": median(values),
            "positive_fraction": positive_fraction(values),
            "gate_pass": gate,
        }
    return {
        "feature": feature,
        "selected_depth": selected_depth,
        "discovery_medians_by_depth": discovery_medians,
        "split_results": split_results,
        "calibration_and_behavior_gate_pass": bool(
            split_results["calibration"]["gate_pass"]
            and split_results["behavior_holdout"]["gate_pass"]
        ),
    }


def block_audit(
    model: str,
    block: dict[str, Any],
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    signals = {
        signal: signal_audit(rows, signal, thresholds) for signal in SIGNALS
    }
    functional_signals = {
        "formation": signal_audit(
            rows, "formation", thresholds, "formation_functional_specificity"
        ),
        "transport": signal_audit(
            rows, "transport", thresholds, "transport_functional_specificity"
        ),
    }
    selected = {signal: value["selected_depth"] for signal, value in signals.items()}
    enriched = add_selected_features(rows, selected)
    coherence_results: dict[str, Any] = {}
    for split in ("calibration", "behavior_holdout"):
        values = [
            float(row[f"{selected['formation']}_role_delta_coherence_median"])
            for row in rows
            if row["split"] == split
        ]
        coherence_results[split] = {
            "median": median(values),
            "gate_pass": median(values) >= float(thresholds["role_delta_coherence_min"]),
        }
    behavior_results: dict[str, Any] = {}
    for split in ("calibration", "behavior_holdout"):
        values = [
            float(row["condition_correct_fraction"])
            for row in rows
            if row["split"] == split
        ]
        behavior_results[split] = {
            "independent_replica_group_count": len(values),
            "condition_correct_fraction": mean(values),
            "gate_pass": mean(values)
            >= float(thresholds["behavior_correct_fraction_min"]),
        }
    discovery = [row for row in enriched if row["split"] == "discovery"]
    baseline_model = fit_ridge(discovery, BASELINE_FEATURES)
    physical_features = BASELINE_FEATURES + (
        "selected_formation",
        "selected_transport",
        "selected_competition",
    )
    physical_model = fit_ridge(discovery, physical_features)
    prediction_splits: dict[str, Any] = {}
    for split in ("calibration", "behavior_holdout"):
        split_rows = [row for row in enriched if row["split"] == split]
        baseline_metrics = prediction_metrics(baseline_model, split_rows)
        physical_metrics = prediction_metrics(physical_model, split_rows)
        delta_r2 = None
        if baseline_metrics["r2"] is not None and physical_metrics["r2"] is not None:
            delta_r2 = clean(float(physical_metrics["r2"]) - float(baseline_metrics["r2"]))
        mae_gain = clean(float(baseline_metrics["mae"]) - float(physical_metrics["mae"]))
        gate = bool(
            physical_metrics["r2"] is not None
            and float(physical_metrics["r2"]) > float(thresholds["prediction_r2_min"])
            and delta_r2 is not None
            and delta_r2 > float(thresholds["prediction_delta_r2_min"])
            and mae_gain > float(thresholds["prediction_mae_gain_min"])
        )
        prediction_splits[split] = {
            "baseline": baseline_metrics,
            "physical": physical_metrics,
            "delta_r2": delta_r2,
            "mae_gain": mae_gain,
            "gate_pass": gate,
        }
    partial_order = bool(
        DEPTH_ORDER[selected["formation"]]
        <= DEPTH_ORDER[selected["transport"]]
        <= DEPTH_ORDER[selected["competition"]]
    )
    behavior_gate = all(value["gate_pass"] for value in behavior_results.values())
    coherence_gate = all(value["gate_pass"] for value in coherence_results.values())
    prediction_gate = all(value["gate_pass"] for value in prediction_splits.values())
    functional_role_observation_gate = bool(
        behavior_gate
        and coherence_gate
        and all(
            value["calibration_and_behavior_gate_pass"]
            for value in functional_signals.values()
        )
    )
    path_gate = bool(
        behavior_gate
        and coherence_gate
        and prediction_gate
        and partial_order
        and all(
            signals[signal]["calibration_and_behavior_gate_pass"]
            for signal in SIGNALS
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase425-OpenBlockAudit",
        "created_at": now(),
        "model": model,
        "block_id": block["block_id"],
        "family_id": block["family_id"],
        "mechanism_id": block["mechanism_id"],
        "candidate": block["candidate"],
        "matched_control_block_id": block["matched_control_block_id"],
        "independent_discovery_group_count": len(discovery),
        "selected_depths": selected,
        "signals": signals,
        "functional_role_signals": functional_signals,
        "functional_role_observation_gate_pass": functional_role_observation_gate,
        "role_delta_coherence": coherence_results,
        "behavior": behavior_results,
        "prediction": {
            "effective_independent_fit_count": len(discovery),
            "baseline_model": baseline_model,
            "physical_model": physical_model,
            "split_results": prediction_splits,
            "gate_pass": prediction_gate,
        },
        "partial_order_gate_pass": partial_order,
        "behavior_gate_pass": behavior_gate,
        "role_coherence_gate_pass": coherence_gate,
        "open_path_gate_pass": path_gate,
        "sealed_tested": False,
        "sealed_gate_pass": False,
        "physical": True,
        "observer": True,
        "predictive": prediction_gate,
        "causal": False,
    }


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.fmean(rows)) if rows else 0.0


def cross_model_audits(
    audits: list[dict[str, Any]], cross_model_minimum: int
) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audits:
        by_block[row["block_id"]].append(row)
    output = []
    for block in BLOCKS:
        rows = by_block[block["block_id"]]
        topology_models: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for row in rows:
            if row["open_path_gate_pass"]:
                topology = tuple(row["selected_depths"][signal] for signal in SIGNALS)
                topology_models[topology].append(row["model"])
        best_topology: tuple[str, str, str] | None = None
        best_models: list[str] = []
        for topology, models in topology_models.items():
            if len(models) > len(best_models):
                best_topology, best_models = topology, sorted(models)
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase425-CrossModelOpenAudit",
                "created_at": now(),
                "block_id": block["block_id"],
                "family_id": block["family_id"],
                "mechanism_id": block["mechanism_id"],
                "candidate": block["candidate"],
                "matched_control_block_id": block["matched_control_block_id"],
                "open_path_models": sorted(
                    row["model"] for row in rows if row["open_path_gate_pass"]
                ),
                "best_topology": list(best_topology) if best_topology else None,
                "best_topology_models": best_models,
                "cross_model_open_gate_pass": len(best_models) >= cross_model_minimum,
                "sealed_tested": False,
                "sealed_gate_pass": False,
                "causal": False,
            }
        )
    by_id = {row["block_id"]: row for row in output}
    for row in output:
        control = by_id[row["matched_control_block_id"]]
        row["matched_control_cross_model_gate_pass"] = control[
            "cross_model_open_gate_pass"
        ]
        row["specificity_gate_pass"] = bool(
            row["candidate"]
            and row["cross_model_open_gate_pass"]
            and not control["cross_model_open_gate_pass"]
        )
    return output


def graph_for_model(model: str, audits: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in audits if row["model"] == model]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for block_index, row in enumerate(rows):
        node_ids = {}
        for signal_index, signal in enumerate(SIGNALS):
            depth = row["selected_depths"][signal]
            signal_gate = row["signals"][signal]["calibration_and_behavior_gate_pass"]
            node_id = f"phase425:{model}:{row['block_id']}:{signal}"
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
                    "candidate": row["candidate"],
                    "event_role": signal,
                    "selected_depth_bin": depth,
                    "selected_relative_depth": (DEPTH_ORDER[depth] + 0.5) / 3,
                    "calibration_and_behavior_gate_pass": signal_gate,
                    "open_path_gate_pass": row["open_path_gate_pass"],
                    "physical": True,
                    "observer": signal == "competition",
                    "native_compute_event": signal == "transport",
                    "compute_edge": False,
                    "predictive": row["prediction"]["gate_pass"],
                    "causal": False,
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "evidence_level": (
                        "readout_observer"
                        if signal == "competition"
                        else "matched_role_physical_observation"
                    ),
                    "score": 1.0 if signal_gate else 0.25,
                    "size": 1.0 if signal_gate else 0.65,
                    "color": COLORS[signal],
                    "position": [block_index * 11.0, (DEPTH_ORDER[depth] + 0.5) * 10.0, signal_index * 4.0],
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
        "phase_id": "Phase425-RoleExchangeAtlas",
        "title": f"Phase425 {model} 同词元角色交换图谱",
        "model": model,
        "evidence_scope": (
            "fresh open split with matched lexical replicas, interfaces and histories; "
            "sealed, causal and neuron gates remain closed"
        ),
        "graph": {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "phase": 425,
                "block_count": 4,
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
        filename = f"phase425_{model}_role_exchange.json"
        write_json(VIS / filename, graph_for_model(model, audits))
        items.append(
            {
                "id": f"phase425_{model}_role_exchange",
                "label": f"Phase425 {model} 同词元角色交换图谱",
                "filename": filename,
                "model": model,
                "phase": 425,
                "evidence_scope": "fresh open split; pipeline sealed split not yet promoted; non-causal",
            }
        )
    write_json(
        VIS / "manifest.json",
        {
            "schema_version": "phase425_role_exchange_manifest.v1",
            "generated_at": now(),
            "default_item_id": items[0]["id"],
            "items": items,
        },
    )
    registry = read_json(REGISTRY)
    entry = {
        "id": "gpt5_phase425_role_exchange_validation",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase425 同词元角色交换验证",
        "description": "翻译、关系从句及匹配负对照的同词元角色形成、合法运输和竞争复核。",
        "manifest_path": "/vis_data/phase425_role_exchange_validation/manifest.json",
        "manifest_schema": "phase425_role_exchange_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase425_role_exchange_validation",
        "models": list(MODELS),
        "evidence_scope": "fresh open split; sealed and causal gates remain explicit",
        "color": "#06b6d4",
    }
    sources = [row for row in registry["sources"] if row["id"] != entry["id"]]
    sources.append(entry)
    registry["sources"] = sources
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def analyze_preseal() -> dict[str, Any]:
    protocol = read_json(OUT / "phase425_protocol.json")
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    frozen_hash = protocol["implementation_commitments"][Path(__file__).name]
    if implementation_hash != frozen_hash:
        raise RuntimeError("Phase425 analysis changed after protocol freeze")
    thresholds = protocol["registered_thresholds"]
    all_rows: dict[str, list[dict[str, Any]]] = {}
    model_completes = {}
    for model in MODELS:
        root = OUT / "models" / model / "open"
        complete = read_json(root / "phase425_collection_complete.json")
        if not complete["all_rows_complete"]:
            raise RuntimeError(f"Incomplete Phase425 open collection for {model}")
        model_completes[model] = complete
        all_rows[model] = read_jsonl(root / "phase425_group_summary_rows.jsonl")
    audits: list[dict[str, Any]] = []
    for model in MODELS:
        by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in all_rows[model]:
            by_block[row["block_id"]].append(row)
        for block in BLOCKS:
            rows = by_block[block["block_id"]]
            if len(rows) != 36:
                raise RuntimeError(f"Expected 36 open groups for {model}/{block['block_id']}")
            audits.append(block_audit(model, block, rows, thresholds))
    cross = cross_model_audits(
        audits, int(thresholds["cross_model_replication_min"])
    )
    unlock_blocks = [row["block_id"] for row in cross if row["specificity_gate_pass"]]
    sealed_unlock = bool(unlock_blocks)
    gate_freeze = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase425-GateFreeze",
        "created_at": now(),
        "open_audit_count": len(audits),
        "cross_model_audit_count": len(cross),
        "sealed_unlock": sealed_unlock,
        "sealed_unlock_blocks": unlock_blocks,
        "selected_depths_by_model_block": {
            f"{row['model']}::{row['block_id']}": row["selected_depths"]
            for row in audits
        },
        "prediction_models_by_model_block": {
            f"{row['model']}::{row['block_id']}": {
                "baseline_model": row["prediction"]["baseline_model"],
                "physical_model": row["prediction"]["physical_model"],
            }
            for row in audits
        },
        "cross_model_open_audits": {
            row["block_id"]: row for row in cross
        },
        "thresholds": thresholds,
        "prohibited_after_freeze": [
            "depth_change",
            "feature_change",
            "threshold_change",
            "topology_change",
            "predictor_refit",
        ],
        "causal_unlock": False,
        "head_channel_neuron_scan_allowed": False,
    }
    write_jsonl(OUT / "phase425_open_block_audits.jsonl", audits)
    write_jsonl(OUT / "phase425_cross_model_open_audits.jsonl", cross)
    write_json(OUT / "phase425_gate_freeze.json", gate_freeze)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "preseal",
        "protocol_valid": protocol["validation"]["valid"],
        "registered_condition_count": protocol["validation"]["condition_count"],
        "executed_open_condition_count": sum(
            int(value["condition_count"]) for value in model_completes.values()
        ),
        "executed_sealed_condition_count": 0,
        "open_path_gate_count_by_model": {
            model: sum(
                row["open_path_gate_pass"] for row in audits if row["model"] == model
            )
            for model in MODELS
        },
        "cross_model_open_candidate_count": sum(
            row["cross_model_open_gate_pass"] and row["candidate"] for row in cross
        ),
        "functional_role_observation_count_by_model": {
            model: sum(
                row["functional_role_observation_gate_pass"]
                for row in audits
                if row["model"] == model
            )
            for model in MODELS
        },
        "cross_model_specific_candidate_count": len(unlock_blocks),
        "sealed_unlock": sealed_unlock,
        "sealed_unlock_blocks": unlock_blocks,
        "sealed_tested": False,
        "causal_tested": False,
        "causal_gate_pass": False,
        "strict_human_double_blind": False,
        "strict_mechanism_closure": "0/72",
        "overall_scientific_progress_percent": 21,
        "progress_interval_percent": [18, 24],
        "conclusion": (
            "Open gates justify a pipeline-sealed audit."
            if sealed_unlock
            else "No candidate passed the frozen cross-model, negative-control and prediction gates; sealed and causal stages remain closed."
        ),
    }
    write_json(OUT / "phase425_global_summary.json", summary)
    publish_visual(audits)
    report_lines = [
        "# Phase425 同词元角色交换预密封审计",
        "",
        f"- 注册条件：{summary['registered_condition_count']}",
        f"- 已执行开放条件：{summary['executed_open_condition_count']}",
        f"- 跨模型开放候选：{summary['cross_model_open_candidate_count']}",
        f"- 通过匹配负对照的候选：{summary['cross_model_specific_candidate_count']}",
        f"- 密封解锁：{summary['sealed_unlock']}",
        "- 严格人类双盲：否；本阶段只有流水线密封。",
        "- 因果、头、通道和神经元门保持关闭。",
        "",
        summary["conclusion"],
    ]
    (OUT / "phase425_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preseal",), default="preseal")
    parser.parse_args()
    analyze_preseal()


if __name__ == "__main__":
    main()
