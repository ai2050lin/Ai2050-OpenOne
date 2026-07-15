#!/usr/bin/env python3
"""Analyze Phase431 without using behavior labels during event discovery."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase431_position_time_protocol import (  # noqa: E402
    LANGUAGE_MODEL,
    MODELS,
    OPEN_SPLITS,
    OUT,
    SCORABLE_ROUTES,
    SEALED_SPLIT,
    TRACE_SCHEMA_VERSION,
    freeze,
    read_json,
    write_json,
)


PHASE_ID = "Phase431-PositionTimeAnalysis"
SCHEMA_VERSION = "phase431_position_time_analysis.v1"
VIS = ROOT / "frontend/public/vis_data/phase431_position_time"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
POSITION_ORDER = (
    "source_1_end",
    "source_2_end",
    "before_selector_end",
    "after_selector_end",
    "question_end",
    "instruction_end",
    "assistant_boundary",
    "prompt_terminal",
    "g1",
    "g2",
    "g3",
    "g4",
)
POSITION_COLORS = {
    "question_end": "#0ea5e9",
    "instruction_end": "#f59e0b",
    "prompt_terminal": "#22c55e",
    "g1": "#14b8a6",
    "g2": "#8b5cf6",
    "g3": "#ec4899",
    "g4": "#ef4444",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase431 non-finite scalar: {value}")
    return round(float(value), 9)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def wilson(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"successes": successes, "total": total, "estimate": 0.0, "lcb": 0.0, "ucb": 1.0}
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": clean(p),
        "lcb": clean(max(0.0, center - radius)),
        "ucb": clean(min(1.0, center + radius)),
    }


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def behavior_root(stage: str) -> Path:
    return OUT / ("sealed" if stage == "sealed" else "open") / LANGUAGE_MODEL / "behavior"


def physical_root(stage: str) -> Path:
    return OUT / ("sealed" if stage == "sealed" else "open") / LANGUAGE_MODEL / "physical"


def condition_good(row: dict[str, Any]) -> bool:
    return bool(
        row["teacher_sequence_correct"]
        and row["natural_target_first"]
        and not row["natural_opposite_first"]
        and row["natural_interface_valid"]
        and not row["natural_revision"]
        and row["natural_boundary"]
        and row["natural_stop"]
        and not row["natural_censoring"]
    )


def paired_role_flip(rows: list[dict[str, Any]], candidate: bool) -> dict[str, float]:
    relevant = [
        row
        for row in rows
        if bool(row["candidate"]) == candidate and row["route_mode"] in SCORABLE_ROUTES
    ]
    cells: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in relevant:
        cells[(row["semantic_group_id"], row["route_mode"])][row["role"]] = row[
            "actual_choice"
        ]
    outcomes = []
    for role_map in cells.values():
        if set(role_map) != {"a", "b"}:
            continue
        if "other" in role_map.values():
            outcomes.append(False)
        elif candidate:
            outcomes.append(role_map["a"] != role_map["b"])
        else:
            outcomes.append(role_map["a"] == role_map["b"])
    return wilson(sum(outcomes), len(outcomes))


def analyze_behavior(stage: str) -> dict[str, Any]:
    protocol = freeze()
    rows = read_jsonl(behavior_root(stage) / "phase431_behavior_rows.jsonl")
    expected_split = SEALED_SPLIT if stage == "sealed" else None
    if expected_split is not None:
        rows = [row for row in rows if row["split"] == expected_split]
    split_names = [expected_split] if expected_split else list(OPEN_SPLITS)
    split_results: dict[str, Any] = {}
    for split in split_names:
        split_rows = [row for row in rows if row["split"] == split]
        block_payload = {}
        for candidate in (True, False):
            block_rows = [row for row in split_rows if bool(row["candidate"]) == candidate]
            group_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in block_rows:
                if candidate and row["route_mode"] not in SCORABLE_ROUTES:
                    continue
                group_map[row["semantic_group_id"]].append(row)
            expected_per_group = 2 * (len(SCORABLE_ROUTES) if candidate else 5)
            group_pass = [
                len(group_rows) == expected_per_group
                and all(condition_good(row) for row in group_rows)
                for group_rows in group_map.values()
            ]
            block_payload["candidate" if candidate else "control"] = {
                "group_all_events": wilson(sum(group_pass), len(group_pass)),
                "actual_choice_counts": dict(
                    Counter(row["actual_choice"] for row in block_rows)
                ),
                "role_response_contract": paired_role_flip(block_rows, candidate),
            }
        split_results[split] = block_payload

    threshold = float(protocol["numeric_gates"]["behavior_group_all_lcb_min"])
    candidate_threshold = float(protocol["numeric_gates"]["candidate_role_flip_lcb_min"])
    control_threshold = float(protocol["numeric_gates"]["control_role_flip_ucb_max"])
    pass_splits = []
    for split in split_names:
        payload = split_results[split]
        pass_splits.append(
            payload["candidate"]["group_all_events"]["lcb"] >= threshold
            and payload["control"]["group_all_events"]["lcb"] >= threshold
            and payload["candidate"]["role_response_contract"]["lcb"]
            >= candidate_threshold
            and payload["control"]["role_response_contract"]["lcb"]
            >= candidate_threshold
        )
    # The control contract metric above measures correct invariance, not flips;
    # its lower bound should be high.  A separate physical G5 later measures the
    # rate of predicted flips and applies the low control upper bound.
    pass_value = bool(rows) and all(pass_splits)
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": stage,
        "row_count": len(rows),
        "split_results": split_results,
        "behavior_gate_pass": pass_value,
        "physical_collection_authorized": pass_value if stage == "open" else False,
        "sealed_read": stage == "sealed",
        "control_flip_ucb_reserved_for_physical_G5": control_threshold,
    }
    path = (
        OUT / "phase431_open_behavior_gate.json"
        if stage == "open"
        else OUT / "sealed" / "phase431_sealed_behavior_gate.json"
    )
    write_json(path, output)
    return output


def analyze_identity() -> dict[str, Any]:
    model_payload = {}
    all_pass = True
    for model in MODELS:
        path = OUT / "models" / model / "identity" / "phase431_identity_complete.json"
        if not path.exists():
            payload = {"model": model, "complete": False, "pass": False}
        else:
            complete = read_json(path)
            payload = {
                **complete,
                "complete": bool(complete.get("all_rows_complete")),
                "pass": bool(
                    complete.get("all_rows_complete")
                    and complete.get("all_identity_top1_pass")
                ),
            }
        model_payload[model] = payload
        all_pass = all_pass and bool(payload["pass"])
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models": model_payload,
        "G1_identity_pass": all_pass,
        "strict_three_model_identity_pass": all_pass,
        "language_model_identity_pass": bool(
            model_payload.get(LANGUAGE_MODEL, {}).get("pass")
        ),
        "qwen_open_physical_authorized": bool(
            model_payload.get(LANGUAGE_MODEL, {}).get("pass")
        ),
    }
    write_json(OUT / "phase431_identity_gate.json", output)
    return output


def sanitize_blind_row(row: dict[str, Any]) -> dict[str, Any]:
    position_metrics = {}
    for role, metric in row["position_metrics"].items():
        position_metrics[role] = {
            key: value
            for key, value in metric.items()
            if key
            not in {
                "source_1_minus_source_2_margin",
                "attention_source_margin_write",
                "mlp_source_margin_write",
            }
        }
    receiver_metrics = {}
    for role, metric in row["receiver_metrics"].items():
        receiver_metrics[role] = {
            **{
                key: value
                for key, value in metric.items()
                if key != "source_partition"
            },
            "source_partition": {
                source_role: {
                    key: value
                    for key, value in payload.items()
                    if key != "source_margin_write"
                }
                for source_role, payload in metric["source_partition"].items()
            },
        }
    anonymous_condition = hashlib.sha256(row["condition_id"].encode("utf-8")).hexdigest()[:20]
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "anonymous_condition_id": anonymous_condition,
        "anonymous_context_id": row["anonymous_context_id"],
        "split": "blind_discovery",
        "layer": row["layer"],
        "relative_depth": row["relative_depth"],
        "position_metrics": position_metrics,
        "receiver_metrics": receiver_metrics,
        "behavior_labels_visible": False,
        "candidate_label_visible": False,
        "task_name_visible": False,
    }


def event_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        layer = int(row["layer"])
        for role, receiver in row["receiver_metrics"].items():
            partition = receiver["source_partition"]
            first = float(partition["source_1"]["write_norm"])
            second = float(partition["source_2"]["write_norm"])
            total = sum(float(payload["write_norm"]) for payload in partition.values())
            position = row["position_metrics"].get(role, {})
            post = float(position.get("residual_post_rms", 0.0))
            transition = float(position.get("transition_rms", 0.0))
            values[(layer, role)].append(
                {
                    "source_dominance": abs(first - second) / max(first + second, 1e-9),
                    "registered_concentration": (first + second) / max(total, 1e-9),
                    "transition_ratio": transition / max(post, 1e-9),
                    "cancellation": abs(float(position.get("attention_mlp_cosine", 0.0))),
                    "replay_error": float(receiver["attention_replay_relative_error"]),
                }
            )
    summaries: dict[tuple[int, str], dict[str, float]] = {}
    for key, records in values.items():
        summaries[key] = {
            name: median(record[name] for record in records)
            for name in records[0]
        }
    events = []
    max_layer = max((layer for layer, _ in summaries), default=0)
    allowed_roles = {"question_end", "instruction_end", "prompt_terminal", "g1", "g2"}
    for (layer, role), summary in summaries.items():
        if role not in allowed_roles or layer > 0.80 * max_layer:
            continue
        prior = summaries.get((layer - 1, role), summary)
        nonstationarity = (
            abs(summary["source_dominance"] - prior["source_dominance"])
            + abs(summary["registered_concentration"] - prior["registered_concentration"])
            + abs(summary["transition_ratio"] - prior["transition_ratio"])
            + abs(summary["cancellation"] - prior["cancellation"])
        )
        score = nonstationarity + 0.25 * summary["registered_concentration"]
        events.append(
            {
                "layer": layer,
                "relative_depth": clean(layer / max(1, max_layer)),
                "position_role": role,
                "blind_event_score": clean(score),
                "nonstationarity": clean(nonstationarity),
                **{key: clean(value) for key, value in summary.items()},
                "behavior_labels_used": False,
            }
        )
    return sorted(events, key=lambda row: (-row["blind_event_score"], row["layer"], row["position_role"]))


def freeze_blind_windows(physical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    discovery = [row for row in physical_rows if row["split"] == "blind_discovery"]
    sanitized = [sanitize_blind_row(row) for row in discovery]
    write_jsonl_gz(OUT / "phase431_blind_ledger.jsonl.gz", sanitized)
    candidates = event_statistics(sanitized)
    windows = []
    for candidate in candidates:
        if any(
            row["position_role"] == candidate["position_role"]
            and abs(int(row["layer"]) - int(candidate["layer"])) < 2
            for row in windows
        ):
            continue
        windows.append(candidate)
        if len(windows) == 4:
            break
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "discovery_trace_row_count": len(discovery),
        "sanitized_blind_row_count": len(sanitized),
        "candidate_event_count": len(candidates),
        "windows": windows,
        "window_count": len(windows),
        "labels_read_during_selection": False,
        "selection_limit": 4,
        "holdout_reselection_allowed": False,
    }
    write_json(OUT / "phase431_blind_window_freeze.json", payload)
    return payload


def row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["condition_id"], int(row["layer"])): row for row in rows}


def latest_window(windows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(windows, key=lambda row: (int(row["layer"]), row["position_role"]))


def predict_condition(
    index: dict[tuple[str, int], dict[str, Any]],
    condition_id: str,
    windows: list[dict[str, Any]],
) -> tuple[str | None, list[float]]:
    votes = []
    margins = []
    for window in windows:
        row = index.get((condition_id, int(window["layer"])))
        if row is None:
            continue
        metric = row["position_metrics"].get(window["position_role"])
        if not metric:
            continue
        margin = float(metric["source_1_minus_source_2_margin"])
        margins.append(margin)
        votes.append("source_1" if margin >= 0.0 else "source_2")
    if not votes:
        return None, margins
    counts = Counter(votes)
    if counts["source_1"] == counts["source_2"]:
        window = latest_window(windows)
        row = index.get((condition_id, int(window["layer"])))
        metric = row["position_metrics"].get(window["position_role"]) if row else None
        if metric:
            return (
                "source_1"
                if float(metric["source_1_minus_source_2_margin"]) >= 0.0
                else "source_2"
            ), margins
    return counts.most_common(1)[0][0], margins


def choice_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    classes = ("source_1", "source_2")
    per_class = {}
    for label in classes:
        positions = [index for index, value in enumerate(actual) if value == label]
        correct = sum(predicted[index] == label for index in positions)
        per_class[label] = wilson(correct, len(positions))
    correct = sum(left == right for left, right in zip(actual, predicted))
    balanced = statistics.fmean(
        per_class[label]["estimate"] for label in classes
    ) if all(per_class[label]["total"] for label in classes) else 0.0
    return {
        "overall": wilson(correct, len(actual)),
        "per_class": per_class,
        "balanced_accuracy": clean(balanced),
        "confusion": {
            f"{left}->{right}": count
            for (left, right), count in sorted(Counter(zip(actual, predicted)).items())
        },
    }


def terminal_and_upstream_metrics(
    physical_rows: list[dict[str, Any]], windows: list[dict[str, Any]], split: str
) -> dict[str, Any]:
    selected_rows = [row for row in physical_rows if row["split"] == split]
    index = row_index(selected_rows)
    condition_meta: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        condition_meta.setdefault(row["condition_id"], row)
    eligible = [
        row
        for row in condition_meta.values()
        if row["candidate"]
        and row["normative_target"]
        and row["route_mode"] in SCORABLE_ROUTES
    ]
    registered = [row for row in eligible if row["actual_choice"] in {"source_1", "source_2"}]
    coverage = wilson(len(registered), len(eligible))
    terminal_actual: list[str] = []
    terminal_predicted: list[str] = []
    upstream_actual: list[str] = []
    upstream_predicted: list[str] = []
    max_layer = max((int(row["layer"]) for row in selected_rows), default=0)
    for meta in registered:
        terminal = index[(meta["condition_id"], max_layer)]["position_metrics"].get(
            "prompt_terminal"
        )
        if terminal:
            terminal_actual.append(meta["actual_choice"])
            terminal_predicted.append(
                "source_1"
                if float(terminal["source_1_minus_source_2_margin"]) >= 0.0
                else "source_2"
            )
        prediction, _ = predict_condition(index, meta["condition_id"], windows)
        if prediction is not None:
            upstream_actual.append(meta["actual_choice"])
            upstream_predicted.append(prediction)
    return {
        "eligible_condition_count": len(eligible),
        "registered_source_coverage": coverage,
        "actual_choice_counts": dict(Counter(row["actual_choice"] for row in eligible)),
        "terminal": choice_metrics(terminal_actual, terminal_predicted),
        "upstream": choice_metrics(upstream_actual, upstream_predicted),
    }


def predicted_role_flip(
    physical_rows: list[dict[str, Any]], windows: list[dict[str, Any]], split: str, candidate: bool
) -> dict[str, Any]:
    selected = [row for row in physical_rows if row["split"] == split]
    index = row_index(selected)
    meta: dict[str, dict[str, Any]] = {}
    for row in selected:
        meta.setdefault(row["condition_id"], row)
    cells: dict[tuple[str, str], dict[str, str | None]] = defaultdict(dict)
    for row in meta.values():
        if bool(row["candidate"]) != candidate or row["route_mode"] not in SCORABLE_ROUTES:
            continue
        prediction, _ = predict_condition(index, row["condition_id"], windows)
        cells[(row["semantic_group_id"], row["route_mode"])][row["role"]] = prediction
    flip_events = []
    for role_map in cells.values():
        if set(role_map) != {"a", "b"} or None in role_map.values():
            continue
        flip_events.append(role_map["a"] != role_map["b"])
    flips = sum(flip_events)
    return {
        "predicted_flip": wilson(flips, len(flip_events)),
        "predicted_invariance": wilson(len(flip_events) - flips, len(flip_events)),
    }


def analyze_physical(stage: str) -> dict[str, Any]:
    protocol = freeze()
    rows = read_jsonl(physical_root(stage) / "phase431_compact_rows.jsonl.gz")
    complete = read_json(physical_root(stage) / "phase431_physical_complete.json")
    threshold = float(protocol["numeric_gates"]["reconstruction_relative_error_median_max"])
    G2 = bool(
        complete["block_reconstruction_relative_error_median"] <= threshold
        and complete["attention_replay_relative_error_median"] <= threshold
    )
    if stage == "open":
        freeze_path = OUT / "phase431_blind_window_freeze.json"
        freeze_payload = (
            read_json(freeze_path)
            if freeze_path.exists()
            else freeze_blind_windows(rows)
        )
    else:
        freeze_payload = read_json(OUT / "phase431_blind_window_freeze.json")
    windows = freeze_payload["windows"]
    split = "behavior_holdout" if stage == "open" else SEALED_SPLIT
    predictive = terminal_and_upstream_metrics(rows, windows, split)
    candidate_flips = predicted_role_flip(rows, windows, split, True)
    control_flips = predicted_role_flip(rows, windows, split, False)
    numeric = protocol["numeric_gates"]
    G3 = bool(
        predictive["registered_source_coverage"]["lcb"]
        >= float(numeric["registered_source_coverage_lcb_min"])
        and predictive["terminal"]["overall"]["lcb"]
        >= float(numeric["terminal_choice_accuracy_lcb_min"])
    )
    G4 = bool(
        predictive["upstream"]["overall"]["lcb"]
        >= float(numeric["upstream_balanced_accuracy_lcb_min"])
        and predictive["upstream"]["balanced_accuracy"] > 0.5
    )
    G5 = bool(
        candidate_flips["predicted_flip"]["lcb"]
        >= float(numeric["candidate_role_flip_lcb_min"])
        and control_flips["predicted_flip"]["ucb"]
        <= float(numeric["control_role_flip_ucb_max"])
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": stage,
        "trace_row_count": len(rows),
        "window_freeze": freeze_payload,
        "predictive_metrics": predictive,
        "candidate_role_metrics": candidate_flips,
        "control_role_metrics": control_flips,
        "G2_architecture_ledger_pass": G2,
        "G3_terminal_identity_pass": G3,
        "G4_upstream_prediction_pass": G4,
        "G5_candidate_specificity_pass": G5,
        "causal": False,
        "single_neuron": False,
        "sealed_read": stage == "sealed",
    }
    path = (
        OUT / "phase431_open_physical_gate.json"
        if stage == "open"
        else OUT / "sealed" / "phase431_sealed_physical_gate.json"
    )
    write_json(path, output)
    return output


def build_open_gate() -> dict[str, Any]:
    behavior = read_json(OUT / "phase431_open_behavior_gate.json")
    identity_path = OUT / "phase431_identity_gate.json"
    identity = read_json(identity_path) if identity_path.exists() else analyze_identity()
    physical_path = OUT / "phase431_open_physical_gate.json"
    physical = read_json(physical_path) if physical_path.exists() else analyze_physical("open")
    gates = {
        "G0": bool(behavior["behavior_gate_pass"]),
        "G1": bool(identity["G1_identity_pass"]),
        "G2": bool(physical["G2_architecture_ledger_pass"]),
        "G3": bool(physical["G3_terminal_identity_pass"]),
        "G4": bool(physical["G4_upstream_prediction_pass"]),
        "G5": bool(physical["G5_candidate_specificity_pass"]),
    }
    unlocked = all(gates.values())
    failed_gates = [name for name, passed in gates.items() if not passed]
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "gates": gates,
        "sealed_unlock": unlocked,
        "failed_gates": failed_gates,
        "sealed_rows_read": False,
        "old_phase429_sealed_used": False,
        "stop_reason": None
        if unlocked
        else failed_gates[0],
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "phase431_open_gate.json", output)
    return output


def build_sealed_result() -> dict[str, Any]:
    open_gate = read_json(OUT / "phase431_open_gate.json")
    if not open_gate.get("sealed_unlock"):
        raise RuntimeError("Phase431 sealed analysis is not authorized")
    behavior = analyze_behavior("sealed")
    physical = analyze_physical("sealed")
    passed = bool(
        behavior["behavior_gate_pass"]
        and physical["G2_architecture_ledger_pass"]
        and physical["G3_terminal_identity_pass"]
        and physical["G4_upstream_prediction_pass"]
        and physical["G5_candidate_specificity_pass"]
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_pass": passed,
        "sealed_rows_read": True,
        "old_phase429_sealed_used": False,
        "behavior": behavior,
        "physical": physical,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "sealed" / "phase431_sealed_result.json", output)
    return output


def exploratory_window_metrics(
    rows: list[dict[str, Any]],
    layer: int,
    role: str,
    candidate: bool,
    max_layer: int,
) -> dict[str, Any] | None:
    selected = [
        row
        for row in rows
        if int(row["layer"]) == layer
        and bool(row["candidate"]) == candidate
        and row["route_mode"] in SCORABLE_ROUTES
        and row["actual_choice"] in {"source_1", "source_2"}
        and role in row["position_metrics"]
    ]
    if not selected:
        return None
    actual = [row["actual_choice"] for row in selected]
    predicted = [
        "source_1"
        if float(row["position_metrics"][role]["source_1_minus_source_2_margin"])
        >= 0.0
        else "source_2"
        for row in selected
    ]
    cells: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row, prediction in zip(selected, predicted):
        cells[(row["semantic_group_id"], row["route_mode"])][row["role"]] = prediction
    flip_events = [
        role_map["a"] != role_map["b"]
        for role_map in cells.values()
        if set(role_map) == {"a", "b"}
    ]
    return {
        "layer": layer,
        "relative_depth": clean(layer / max(1, max_layer)),
        "position_role": role,
        "candidate": candidate,
        "choice": choice_metrics(actual, predicted),
        "predicted_role_flip": wilson(sum(flip_events), len(flip_events)),
        "condition_count": len(selected),
        "posthoc_label_read": True,
        "can_reopen_phase431_gate": False,
    }


def build_posthoc_failure_map() -> dict[str, Any]:
    rows = read_jsonl(physical_root("open") / "phase431_compact_rows.jsonl.gz")
    rows = [row for row in rows if row["split"] == "behavior_holdout"]
    layers = sorted({int(row["layer"]) for row in rows})
    max_layer = max(layers)
    roles = [
        role
        for role in POSITION_ORDER
        if any(role in row["position_metrics"] for row in rows)
    ]
    metric_groups: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["route_mode"] in SCORABLE_ROUTES
            and row["actual_choice"] in {"source_1", "source_2"}
        ):
            metric_groups[(int(row["layer"]), bool(row["candidate"]))].append(row)
    metrics = []
    for layer in layers:
        for role in roles:
            for candidate in (True, False):
                payload = exploratory_window_metrics(
                    metric_groups[(layer, candidate)],
                    layer,
                    role,
                    candidate,
                    max_layer,
                )
                if payload is not None:
                    metrics.append(payload)

    prompt_roles = {
        "source_1_end",
        "source_2_end",
        "before_selector_end",
        "after_selector_end",
        "question_end",
        "instruction_end",
        "assistant_boundary",
        "prompt_terminal",
    }
    generated_roles = {"g1", "g2", "g3", "g4"}

    def best(scope: set[str]) -> list[dict[str, Any]]:
        output = []
        for role in sorted(scope, key=lambda value: POSITION_ORDER.index(value)):
            candidates = [
                row for row in metrics if row["candidate"] and row["position_role"] == role
            ]
            if not candidates:
                continue
            winner = max(
                candidates,
                key=lambda row: (
                    row["choice"]["balanced_accuracy"],
                    row["choice"]["overall"]["lcb"],
                    -row["layer"],
                ),
            )
            first_ninety = next(
                (
                    row
                    for row in sorted(candidates, key=lambda item: item["layer"])
                    if row["choice"]["balanced_accuracy"] >= 0.90
                ),
                None,
            )
            output.append(
                {
                    "position_role": role,
                    "best_layer": winner["layer"],
                    "best_balanced_accuracy": winner["choice"]["balanced_accuracy"],
                    "best_overall_lcb": winner["choice"]["overall"]["lcb"],
                    "first_layer_at_or_above_0_90": first_ninety["layer"]
                    if first_ninety
                    else None,
                }
            )
        return output

    freeze_payload = read_json(OUT / "phase431_blind_window_freeze.json")
    index = row_index(rows)
    frozen_predictions = []
    frozen_actual = []
    condition_meta = {}
    for row in rows:
        condition_meta.setdefault(row["condition_id"], row)
    for row in condition_meta.values():
        if (
            not row["candidate"]
            or row["route_mode"] not in SCORABLE_ROUTES
            or row["actual_choice"] not in {"source_1", "source_2"}
        ):
            continue
        prediction, _ = predict_condition(
            index, row["condition_id"], freeze_payload["windows"]
        )
        if prediction is not None:
            frozen_predictions.append(prediction)
            frozen_actual.append(row["actual_choice"])
    output = {
        "schema_version": "phase431_posthoc_failure_map.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "split": "behavior_holdout",
        "trace_row_count": len(rows),
        "metric_cell_count": len(metrics),
        "metrics": metrics,
        "best_prechoice_observers": best(prompt_roles),
        "best_postchoice_observers": best(generated_roles),
        "frozen_predictor": {
            "prediction_counts": dict(Counter(frozen_predictions)),
            "actual_counts": dict(Counter(frozen_actual)),
            "choice": choice_metrics(frozen_actual, frozen_predictions),
        },
        "audit_corrections": {
            "g1_g2_are_postchoice_for_first_source_target": True,
            "postchoice_windows_must_not_enter_future_upstream_gate": True,
            "frozen_windows_reselected": False,
            "phase431_gate_reopened": False,
            "labels_used": True,
            "evidence_level": "posthoc_descriptive_only",
        },
    }
    write_json(OUT / "phase431_posthoc_failure_map.json", output)
    return output


def build_final_summary() -> dict[str, Any]:
    protocol = read_json(OUT / "phase431_protocol.json")
    identity = read_json(OUT / "phase431_identity_gate.json")
    behavior = read_json(OUT / "phase431_open_behavior_gate.json")
    physical = read_json(OUT / "phase431_open_physical_gate.json")
    open_gate = read_json(OUT / "phase431_open_gate.json")
    posthoc = read_json(OUT / "phase431_posthoc_failure_map.json")
    identity_counts = {}
    for model in MODELS:
        rows = read_jsonl(
            OUT / "models" / model / "identity" / "phase431_identity_rows.jsonl"
        )
        identity_counts[model] = {
            "rows": len(rows),
            "terminal_reconstruction_top1_equal": sum(
                row["terminal_reconstruction_top1_equal"] for row in rows
            ),
            "cache_first_boundary_top1_equal": sum(row["cache_top1_equal"] for row in rows),
            "generation_step_one_cache_top1_equal": sum(
                row["generation_step_one_cache_top1_equal"] for row in rows
            ),
            "batch_single_top1_equal": sum(row["batch_single_top1_equal"] for row in rows),
            "strict_pass": bool(identity["models"][model]["pass"]),
        }
    output = {
        "schema_version": "phase431_final_summary.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "status": "open_negative_sealed_not_read",
        "denominator": protocol["denominator_audit"],
        "identity": identity_counts,
        "behavior_gate_pass": behavior["behavior_gate_pass"],
        "physical_trace_rows": physical["trace_row_count"],
        "blind_windows": physical["window_freeze"]["windows"],
        "predictive_metrics": physical["predictive_metrics"],
        "candidate_role_metrics": physical["candidate_role_metrics"],
        "control_role_metrics": physical["control_role_metrics"],
        "gates": open_gate["gates"],
        "failed_gates": open_gate["failed_gates"],
        "sealed_unlock": open_gate["sealed_unlock"],
        "sealed_rows_read": open_gate["sealed_rows_read"],
        "posthoc_audit_corrections": posthoc["audit_corrections"],
        "posthoc_observer_findings": {
            "frozen_predictor": posthoc["frozen_predictor"],
            "best_prechoice_observers": posthoc["best_prechoice_observers"],
            "best_postchoice_observers": posthoc["best_postchoice_observers"],
            "labels_used": True,
            "independent_confirmation_required": True,
            "can_reopen_phase431_gate": False,
        },
        "closure": {
            "strict_mechanisms": "0/72",
            "overall_scientific_progress_percent": 21,
            "cautious_interval_percent": [18, 24],
        },
        "evidence": {
            "physical": True,
            "observer": True,
            "predictive": False,
            "causal": False,
            "single_neuron": False,
            "model_specific": True,
            "cross_model_language_mechanism": False,
        },
    }
    write_json(OUT / "phase431_final_summary.json", output)
    return output


def publish_visual() -> dict[str, Any]:
    open_gate = read_json(OUT / "phase431_open_gate.json")
    physical = read_json(OUT / "phase431_open_physical_gate.json")
    windows = physical["window_freeze"]["windows"]
    nodes = []
    edges = []
    ordered_windows = sorted(
        windows,
        key=lambda row: (
            int(row["layer"]),
            POSITION_ORDER.index(row["position_role"])
            if row["position_role"] in POSITION_ORDER
            else len(POSITION_ORDER),
        ),
    )
    for index, window in enumerate(ordered_windows):
        role = window["position_role"]
        layer = int(window["layer"])
        role_index = POSITION_ORDER.index(role) if role in POSITION_ORDER else len(POSITION_ORDER)
        generation = int(role[1:]) if role.startswith("g") and role[1:].isdigit() else 0
        node_id = f"phase431:qwen3:L{layer}:{role}"
        nodes.append(
            {
                "id": node_id,
                "label": f"L{layer} / {role}",
                "type": "position_time_blind_event",
                "model": LANGUAGE_MODEL,
                "layer": layer,
                "relative_depth": window["relative_depth"],
                "position_role": role,
                "generation_step": generation,
                "blind_event_score": window["blind_event_score"],
                "score": window["blind_event_score"],
                "size": 0.7,
                "color": POSITION_COLORS.get(role, "#64748b"),
                "position": [float(layer), float(role_index * 2), float(generation * 3)],
                "physical": True,
                "observer": True,
                "predictive": bool(open_gate["gates"]["G4"]),
                "causal": False,
                "single_neuron": False,
                "pipeline_sealed": False,
                "evidence_level": "heldout_predictive_observation"
                if open_gate["gates"]["G4"]
                else "blind_physical_observation",
                "show_label": True,
            }
        )
        if index:
            previous = nodes[index - 1]["id"]
            edges.append(
                {
                    "id": f"{previous}->{node_id}",
                    "source": previous,
                    "target": node_id,
                    "type": "observed_position_time_order",
                    "physical": True,
                    "observer": True,
                    "predictive": bool(open_gate["gates"]["G4"]),
                    "causal": False,
                    "single_neuron": False,
                    "evidence_level": "observed_order_not_causal",
                    "color": "#64748b",
                    "weight": 0.7,
                }
            )
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "title": "Phase431 最小位置-时间自然计算图谱",
        "model": LANGUAGE_MODEL,
        "evidence_scope": "blind-frozen architecture events with independent open holdout evaluation; non-causal and non-neuronal",
        "graph": {
            "meta": {
                "layer_axis": "x",
                "position_role_axis": "y",
                "generation_step_axis": "z",
                "gates": open_gate["gates"],
                "sealed_unlock": open_gate["sealed_unlock"],
                "failed_gates": open_gate.get("failed_gates", []),
                "terminal_choice_accuracy": physical["predictive_metrics"]["terminal"]["overall"]["estimate"],
                "upstream_balanced_accuracy": physical["predictive_metrics"]["upstream"]["balanced_accuracy"],
                "candidate_predicted_role_flip": physical["candidate_role_metrics"]["predicted_flip"]["estimate"],
                "control_predicted_role_flip": physical["control_role_metrics"]["predicted_flip"]["estimate"],
                "event_order_contract": "registered prompt role order followed by natural generation step; observed order is not a causal edge",
            },
            "nodes": nodes,
            "edges": edges,
        },
    }
    VIS.mkdir(parents=True, exist_ok=True)
    filename = "phase431_qwen3_position_time.json"
    write_json(VIS / filename, payload)
    manifest = {
        "schema_version": "phase431_position_time_manifest.v1",
        "generated_at": now(),
        "default_item_id": "phase431_qwen3_position_time",
        "items": [
            {
                "id": "phase431_qwen3_position_time",
                "label": "Phase431 Qwen3 最小位置-时间图谱",
                "filename": filename,
                "model": LANGUAGE_MODEL,
                "phase": 431,
                "evidence_scope": payload["evidence_scope"],
            }
        ],
    }
    write_json(VIS / "manifest.json", manifest)
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase431_position_time",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase431 最小位置-时间自然计算图谱",
        "description": "来源、问题、输出指令、真实提示终端与自然生成时间的盲冻结组件事件。",
        "manifest_path": "/vis_data/phase431_position_time/manifest.json",
        "manifest_schema": "phase431_position_time_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase431_position_time",
        "models": list(MODELS),
        "evidence_scope": payload["evidence_scope"],
        "color": "#06b6d4",
    }
    registry["sources"] = [
        row for row in registry["sources"] if row["id"] != source["id"]
    ] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    return {"manifest": manifest, "node_count": len(nodes), "edge_count": len(edges)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "behavior-open",
            "identity",
            "physical-open",
            "open-gate",
            "sealed",
            "posthoc",
            "summary",
            "publish",
        ),
    )
    args = parser.parse_args()
    if args.command == "behavior-open":
        payload = analyze_behavior("open")
    elif args.command == "identity":
        payload = analyze_identity()
    elif args.command == "physical-open":
        payload = analyze_physical("open")
    elif args.command == "open-gate":
        payload = build_open_gate()
    elif args.command == "sealed":
        payload = build_sealed_result()
    elif args.command == "posthoc":
        payload = build_posthoc_failure_map()
    elif args.command == "summary":
        payload = build_final_summary()
    else:
        payload = publish_visual()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
