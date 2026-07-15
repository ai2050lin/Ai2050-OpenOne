#!/usr/bin/env python3
"""Analyze Phase435 interfaces, natural behavior, and physical relation paths."""

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
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase435_natural_relation_collect import POSITION_ROLES, RECEIVER_ROLES  # noqa: E402
from phase435_natural_relation_protocol import (  # noqa: E402
    BEHAVIOR_SPLITS,
    CONTRACTS,
    GENERIC_CONTROL,
    INTERFACE_SIMPLICITY,
    INTERFACES,
    MODELS,
    OUT,
    RELATION_FAMILIES,
    SCHEMA_VERSION,
    freeze,
    read_json,
    read_jsonl,
    write_json,
)


PHASE_ID = "Phase435-NaturalRelationAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase435_natural_relation"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODEL_COLORS = {"qwen3": "#22c55e", "glm4": "#0ea5e9", "deepseek7b": "#f97316"}
CONTRACT_LABELS = {
    "field_extract": "字段抽取",
    "natural_qa": "自然问答",
    "relation_rewrite": "关系改写",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase435 non-finite scalar: {value}")
    return round(float(value), 9)


def wilson(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {
            "successes": successes,
            "total": total,
            "estimate": 0.0,
            "lcb": 0.0,
            "ucb": 1.0,
        }
    estimate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        estimate * (1.0 - estimate) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": clean(estimate),
        "lcb": clean(max(0.0, center - radius)),
        "ucb": clean(min(1.0, center + radius)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_gzip_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def behavior_path(stage: str, model: str) -> Path:
    return OUT / stage / model / "behavior/phase435_behavior_rows.jsonl"


def physical_path(stage: str, model: str) -> Path:
    return OUT / stage / model / "physical/phase435_physical_rows.jsonl.gz"


def behavior_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "condition_count": total,
        "content": wilson(sum(bool(row["natural_content_good"]) for row in rows), total),
        "teacher": wilson(sum(bool(row["teacher_sequence_correct"]) for row in rows), total),
        "first_answer": wilson(sum(bool(row["natural_first_answer_good"]) for row in rows), total),
        "complete_answer": wilson(sum(bool(row["natural_complete_answer_good"]) for row in rows), total),
        "interface_parse": wilson(sum(bool(row["natural_interface_valid"]) for row in rows), total),
        "stop_separate": wilson(sum(bool(row["natural_stop_good"]) for row in rows), total),
        "other": wilson(sum(bool(row["natural_other"]) for row in rows), total),
        "choice_counts": dict(Counter(str(row["actual_choice"]) for row in rows)),
    }


def position_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions = {
        position: behavior_metrics(
            [row for row in rows if row["target_position"] == position]
        )
        for position in ("first", "second")
    }
    positions["content_position_gap"] = clean(
        abs(
            float(positions["first"]["content"]["estimate"])
            - float(positions["second"]["content"]["estimate"])
        )
    )
    return positions


def interface_input_hashes() -> dict[str, str]:
    return {
        model: sha256_file(behavior_path("interface", model))
        for model in MODELS
    }


def analyze_interface_model(model: str) -> dict[str, Any]:
    rows = read_jsonl(behavior_path("interface", model))
    protocol = read_json(OUT / "phase435_protocol.json")
    thresholds = protocol["interface_calibration_gate"]
    interfaces: dict[str, Any] = {}
    for interface in INTERFACES:
        contract_audits = {}
        for contract in CONTRACTS:
            selected = [
                row
                for row in rows
                if row["interface"] == interface and row["contract"] == contract
            ]
            metrics = behavior_metrics(selected)
            positions = position_metrics(selected)
            passed = bool(
                metrics["content"]["lcb"] >= thresholds["per_contract_wilson_lcb"]
                and all(
                    positions[position]["content"]["lcb"]
                    >= thresholds["per_position_wilson_lcb"]
                    for position in ("first", "second")
                )
                and positions["content_position_gap"]
                <= thresholds["maximum_position_gap"]
                and metrics["other"]["ucb"] <= thresholds["other_wilson_ucb"]
            )
            contract_audits[contract] = {
                "metrics": metrics,
                "positions": positions,
                "pass": passed,
            }
        minimum_contract_lcb = min(
            float(value["metrics"]["content"]["lcb"])
            for value in contract_audits.values()
        )
        minimum_position_lcb = min(
            float(value["positions"][position]["content"]["lcb"])
            for value in contract_audits.values()
            for position in ("first", "second")
        )
        maximum_gap = max(
            float(value["positions"]["content_position_gap"])
            for value in contract_audits.values()
        )
        maximum_other_ucb = max(
            float(value["metrics"]["other"]["ucb"])
            for value in contract_audits.values()
        )
        interfaces[interface] = {
            "contracts": contract_audits,
            "qualified": all(value["pass"] for value in contract_audits.values()),
            "selection_rank": [
                clean(minimum_contract_lcb),
                clean(minimum_position_lcb),
                clean(-maximum_gap),
                clean(-maximum_other_ucb),
                -INTERFACE_SIMPLICITY.index(interface),
            ],
        }
    qualified = [
        interface for interface in INTERFACE_SIMPLICITY if interfaces[interface]["qualified"]
    ]
    selected_interface = (
        qualified[0]
        if qualified
        else max(
            INTERFACES,
            key=lambda interface: tuple(interfaces[interface]["selection_rank"]),
        )
    )
    return {
        "model": model,
        "row_count": len(rows),
        "interfaces": interfaces,
        "selected_interface": selected_interface,
        "calibration_qualified": bool(qualified),
        "qualified_interfaces": qualified,
        "selection_used_only_independent_calibration": True,
        "natural_corpus_claim": False,
    }


def analyze_interface_freeze() -> dict[str, Any]:
    freeze()
    hashes = interface_input_hashes()
    path = OUT / "phase435_interface_freeze.json"
    if path.exists():
        existing = read_json(path)
        if existing.get("input_hashes") != hashes:
            raise RuntimeError(
                "Phase435 interface calibration changed after freeze; refusing to retune"
            )
        return existing
    models = {model: analyze_interface_model(model) for model in MODELS}
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "input_hashes": hashes,
        "models": models,
        "frozen_before_behavior_holdout": True,
        "retuning_on_behavior_or_physical_holdout_forbidden": True,
    }
    write_json(path, output)
    return output


def analyze_behavior_model(model: str, interface_freeze: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(behavior_path("behavior", model))
    selected_interface = interface_freeze["models"][model]["selected_interface"]
    if any(row["interface"] != selected_interface for row in rows):
        raise RuntimeError(f"Phase435 behavior interface drift for {model}")
    protocol = read_json(OUT / "phase435_protocol.json")
    thresholds = protocol["natural_behavior_gate"]
    contracts: dict[str, Any] = {}
    for contract in CONTRACTS:
        splits = {}
        split_passes = []
        for split in BEHAVIOR_SPLITS:
            selected = [
                row
                for row in rows
                if row["contract"] == contract and row["split"] == split
            ]
            metrics = behavior_metrics(selected)
            positions = position_metrics(selected)
            pass_value = bool(
                all(
                    positions[position]["content"]["lcb"]
                    >= thresholds["discovery_and_holdout_per_position_wilson_lcb"]
                    and positions[position]["teacher"]["lcb"]
                    >= thresholds["teacher_event_per_position_wilson_lcb"]
                    and positions[position]["interface_parse"]["lcb"]
                    >= thresholds["interface_parse_per_position_wilson_lcb"]
                    for position in ("first", "second")
                )
                and positions["content_position_gap"]
                <= thresholds["maximum_position_gap"]
                and (
                    split != "behavior_holdout"
                    or metrics["other"]["ucb"] <= thresholds["holdout_other_wilson_ucb"]
                )
            )
            splits[split] = {
                "metrics": metrics,
                "positions": positions,
                "pass": pass_value,
            }
            split_passes.append(pass_value)
        eligible = bool(
            interface_freeze["models"][model]["calibration_qualified"]
            and all(split_passes)
        )
        contracts[contract] = {
            "interface": selected_interface,
            "splits": splits,
            "behavior_eligible": eligible,
            "stop_excluded_from_content_gate": True,
        }
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "row_count": len(rows),
        "selected_interface": selected_interface,
        "interface_calibration_qualified": interface_freeze["models"][model]["calibration_qualified"],
        "contracts": contracts,
    }
    write_json(OUT / f"phase435_{model}_behavior_audit.json", output)
    return output


def analyze_behavior_gate() -> dict[str, Any]:
    interface_freeze = read_json(OUT / "phase435_interface_freeze.json")
    behavior = {}
    for model in MODELS:
        if interface_freeze["models"][model]["calibration_qualified"]:
            if not behavior_path("behavior", model).exists():
                raise RuntimeError(f"Qualified Phase435 behavior rows missing for {model}")
            behavior[model] = analyze_behavior_model(model, interface_freeze)
        else:
            behavior[model] = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                "model": model,
                "row_count": 0,
                "selected_interface": interface_freeze["models"][model]["selected_interface"],
                "interface_calibration_qualified": False,
                "status": "interface_gate_failed_behavior_denominator_unread",
                "contracts": {
                    contract: {
                        "interface": interface_freeze["models"][model]["selected_interface"],
                        "splits": {},
                        "behavior_eligible": False,
                        "stop_excluded_from_content_gate": True,
                    }
                    for contract in CONTRACTS
                },
            }
            write_json(OUT / f"phase435_{model}_behavior_audit.json", behavior[model])
    eligible = [
        {"model": model, "contract": contract}
        for model in MODELS
        for contract in CONTRACTS
        if behavior[model]["contracts"][contract]["behavior_eligible"]
    ]
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "interface_freeze": interface_freeze,
        "behavior": behavior,
        "eligible_model_contracts": eligible,
        "eligible_count": len(eligible),
        "physical_unlock": bool(eligible),
        "physical_not_run_for_ineligible_contracts": True,
        "sealed_rows_read": False,
    }
    write_json(OUT / "phase435_behavior_gate.json", output)
    return output


def cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0 if left_norm <= 1e-12 and right_norm <= 1e-12 else 1.0
    return max(0.0, min(2.0, 1.0 - dot / (left_norm * right_norm)))


def median(values: Iterable[float]) -> float:
    materialized = list(values)
    return clean(statistics.median(materialized)) if materialized else 0.0


def process_geometry_group(
    rows: list[dict[str, Any]], accumulators: dict[tuple[Any, ...], dict[str, Any]]
) -> None:
    indexed = {
        (row["contract"], row["record_order"], row["mapping"], row["query_role"]): row
        for row in rows
    }
    contracts = sorted({row["contract"] for row in rows})
    for contract in contracts:
        sample = next(row for row in rows if row["contract"] == contract)
        fold = sample["physical_fold"]
        family = sample["relation_family"]
        for layer_index in range(len(sample["layers"])):
            for position in POSITION_ROLES:
                same_distances = []
                different_distances = []
                for mapping in ("direct", "swapped"):
                    for query_role in ("a", "b"):
                        left = indexed[(contract, "ab", mapping, query_role)]["layers"][layer_index]["position_metrics"][position]["state_sketch"]
                        right = indexed[(contract, "ba", mapping, query_role)]["layers"][layer_index]["position_metrics"][position]["state_sketch"]
                        same_distances.append(cosine_distance(left, right))
                    for order in ("ab", "ba"):
                        left = indexed[(contract, order, mapping, "a")]["layers"][layer_index]["position_metrics"][position]["state_sketch"]
                        right = indexed[(contract, order, mapping, "b")]["layers"][layer_index]["position_metrics"][position]["state_sketch"]
                        different_distances.append(cosine_distance(left, right))
                effect = statistics.median(different_distances) - statistics.median(same_distances)
                key = (contract, fold, layer_index, position)
                accumulator = accumulators[key]
                accumulator["effects"].append(effect)
                accumulator["same"].extend(same_distances)
                accumulator["different"].extend(different_distances)
                accumulator["families"][family].append(effect)


def geometry_ledger(path: Path) -> dict[str, Any]:
    accumulators: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "effects": [],
            "same": [],
            "different": [],
            "families": defaultdict(list),
        }
    )
    current_group = None
    group_rows: list[dict[str, Any]] = []
    max_block_error = 0.0
    max_replay_error = 0.0
    row_count = 0
    layer_count = 0
    for row in iter_gzip_rows(path):
        row_count += 1
        layer_count = max(layer_count, len(row["layers"]))
        for layer in row["layers"]:
            for payload in layer["position_metrics"].values():
                max_block_error = max(
                    max_block_error,
                    float(payload["block_reconstruction_relative_error"]),
                )
            for payload in layer["receiver_metrics"].values():
                max_replay_error = max(
                    max_replay_error,
                    float(payload["attention_replay_relative_error"]),
                )
        group_id = row["semantic_group_id"]
        if current_group is not None and group_id != current_group:
            process_geometry_group(group_rows, accumulators)
            group_rows = []
        current_group = group_id
        group_rows.append(row)
    if group_rows:
        process_geometry_group(group_rows, accumulators)
    cells = []
    for (contract, fold, layer, position), values in sorted(accumulators.items()):
        cells.append(
            {
                "contract": contract,
                "fold": fold,
                "layer": layer,
                "relative_depth": clean(layer / max(1, layer_count - 1)),
                "position_role": position,
                "group_count": len(values["effects"]),
                "same_target_order_distance_median": median(values["same"]),
                "different_target_distance_median": median(values["different"]),
                "geometry_effect_median": median(values["effects"]),
                "relation_family_effects": {
                    family: median(values["families"].get(family, []))
                    for family in RELATION_FAMILIES
                },
                "output_label_blind": True,
            }
        )
    return {
        "row_count": row_count,
        "layer_count": layer_count,
        "max_block_reconstruction_relative_error": clean(max_block_error),
        "max_attention_replay_relative_error": clean(max_replay_error),
        "cells": cells,
    }


def choice_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_source = {}
    source_recalls = []
    for source in ("source_1", "source_2"):
        selected = [row for row in rows if row["actual"] == source]
        metric = wilson(sum(row["predicted"] == source for row in selected), len(selected))
        per_source[source] = metric
        source_recalls.append(float(metric["estimate"]))
    position_accuracy = {}
    for position in ("first", "second"):
        selected = [row for row in rows if row["target_position"] == position]
        position_accuracy[position] = wilson(
            sum(row["predicted"] == row["actual"] for row in selected), len(selected)
        )
    return {
        "condition_count": len(rows),
        "accuracy": clean(
            sum(row["predicted"] == row["actual"] for row in rows) / max(1, len(rows))
        ),
        "balanced_accuracy": clean(statistics.mean(source_recalls)),
        "per_source": per_source,
        "position_accuracy": position_accuracy,
        "position_gap": clean(
            abs(
                float(position_accuracy["first"]["estimate"])
                - float(position_accuracy["second"]["estimate"])
            )
        ),
        "score_median_by_source": {
            source: median(row["score"] for row in rows if row["actual"] == source)
            for source in ("source_1", "source_2")
        },
    }


def transport_ledger(path: Path) -> dict[str, Any]:
    cells: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_gzip_rows(path):
        for layer in row["layers"]:
            for receiver, receiver_payload in layer["receiver_metrics"].items():
                source_payload = receiver_payload["source_partition"]
                score = (
                    float(source_payload["source_1_record"]["source_1_minus_source_2_margin_write"])
                    - float(source_payload["source_2_record"]["source_1_minus_source_2_margin_write"])
                )
                cells[(row["contract"], row["physical_fold"], int(layer["layer"]), receiver)].append(
                    {
                        "actual": row["semantic_target_source"],
                        "predicted": "source_1" if score >= 0 else "source_2",
                        "score": score,
                        "target_position": row["target_position"],
                        "relation_family": row["relation_family"],
                    }
                )
    return {
        "cells": [
            {
                "contract": key[0],
                "fold": key[1],
                "layer": key[2],
                "receiver_role": key[3],
                "metrics": choice_metrics(values),
            }
            for key, values in sorted(cells.items())
        ]
    }


def lookup_cell(cells: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matches = [
        row for row in cells if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Phase435 cell for {criteria}, found {len(matches)}")
    return matches[0]


def select_geometry_window(cells: list[dict[str, Any]], contract: str) -> dict[str, Any]:
    discovery = [
        row for row in cells if row["contract"] == contract and row["fold"] == "discovery"
    ]
    selected = max(
        discovery,
        key=lambda row: (
            float(row["geometry_effect_median"]),
            -int(row["layer"]),
            -POSITION_ROLES.index(row["position_role"]),
        ),
    )
    return {
        "contract": contract,
        "selected_layer": selected["layer"],
        "selected_position_role": selected["position_role"],
        "discovery_cell": selected,
        "selected_without_holdout_access": True,
    }


def select_transport_window(cells: list[dict[str, Any]], contract: str) -> dict[str, Any]:
    discovery = [
        row for row in cells if row["contract"] == contract and row["fold"] == "discovery"
    ]
    selected = max(
        discovery,
        key=lambda row: (
            float(row["metrics"]["balanced_accuracy"]),
            min(
                float(row["metrics"]["per_source"][source]["estimate"])
                for source in ("source_1", "source_2")
            ),
            -float(row["metrics"]["position_gap"]),
            -int(row["layer"]),
            -RECEIVER_ROLES.index(row["receiver_role"]),
        ),
    )
    return {
        "contract": contract,
        "selected_layer": selected["layer"],
        "selected_receiver_role": selected["receiver_role"],
        "discovery_cell": selected,
        "selected_without_holdout_access": True,
    }


def geometry_gate(
    candidate: dict[str, Any], generic: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    positive_families = sum(
        float(candidate["relation_family_effects"].get(family, 0.0))
        >= thresholds["label_blind_geometry_effect_min"]
        for family in RELATION_FAMILIES
    )
    effect = float(candidate["geometry_effect_median"])
    generic_effect = float(generic["geometry_effect_median"])
    return {
        "candidate_effect": clean(effect),
        "generic_effect": clean(generic_effect),
        "candidate_over_generic": clean(effect - generic_effect),
        "positive_relation_family_count": positive_families,
        "pass": bool(
            effect >= thresholds["label_blind_geometry_effect_min"]
            and positive_families >= thresholds["geometry_positive_relation_families_min"]
            and effect - generic_effect >= thresholds["geometry_candidate_over_generic_min"]
        ),
    }


def transport_gates(
    candidate: dict[str, Any], generic: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    metrics = candidate["metrics"]
    generic_metrics = generic["metrics"]
    source_transport = all(
        float(metrics["per_source"][source]["lcb"])
        >= thresholds["transport_per_source_wilson_lcb_min"]
        for source in ("source_1", "source_2")
    )
    prediction = bool(
        float(metrics["balanced_accuracy"])
        >= thresholds["transport_balanced_accuracy_min"]
        and float(metrics["position_gap"])
        <= thresholds["transport_maximum_position_gap"]
    )
    specificity_gap = (
        float(metrics["balanced_accuracy"])
        - float(generic_metrics["balanced_accuracy"])
    )
    specificity = specificity_gap >= thresholds["transport_candidate_over_generic_accuracy_min"]
    return {
        "source_transport_pass": source_transport,
        "frozen_prediction_pass": prediction,
        "generic_specificity_pass": specificity,
        "candidate_over_generic_balanced_accuracy": clean(specificity_gap),
    }


def evaluate_physical_contract(
    contract: str,
    geometry: dict[str, Any],
    transport: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    stage: str,
    open_freeze: dict[str, Any] | None = None,
    generic_contract: str = GENERIC_CONTROL,
) -> dict[str, Any]:
    fold = "holdout" if stage == "physical" else "sealed"
    if stage == "physical":
        geometry_window = select_geometry_window(geometry["cells"], contract)
        transport_window = select_transport_window(transport["cells"], contract)
    else:
        if open_freeze is None:
            raise RuntimeError("Sealed Phase435 analysis requires open windows")
        geometry_window = open_freeze["geometry_window"]
        transport_window = open_freeze["transport_window"]
    geometry_candidate = lookup_cell(
        geometry["cells"],
        contract=contract,
        fold=fold,
        layer=geometry_window["selected_layer"],
        position_role=geometry_window["selected_position_role"],
    )
    geometry_generic = lookup_cell(
        geometry["cells"],
        contract=generic_contract,
        fold=fold,
        layer=geometry_window["selected_layer"],
        position_role=geometry_window["selected_position_role"],
    )
    transport_candidate = lookup_cell(
        transport["cells"],
        contract=contract,
        fold=fold,
        layer=transport_window["selected_layer"],
        receiver_role=transport_window["selected_receiver_role"],
    )
    transport_generic = lookup_cell(
        transport["cells"],
        contract=generic_contract,
        fold=fold,
        layer=transport_window["selected_layer"],
        receiver_role=transport_window["selected_receiver_role"],
    )
    geometry_result = geometry_gate(geometry_candidate, geometry_generic, thresholds)
    transport_result = transport_gates(transport_candidate, transport_generic, thresholds)
    return {
        "contract": contract,
        "stage": stage,
        "geometry_window": geometry_window,
        "transport_window": transport_window,
        "geometry_candidate": geometry_candidate,
        "geometry_generic_control": geometry_generic,
        "geometry_gate": geometry_result,
        "transport_candidate": transport_candidate,
        "transport_generic_control": transport_generic,
        "transport_gates": transport_result,
        "gates": {
            "G3_label_blind_order_geometry": geometry_result["pass"],
            "G4_semantic_source_transport": transport_result["source_transport_pass"],
            "G5_frozen_holdout_prediction": transport_result["frozen_prediction_pass"],
            "G6_generic_pairing_specificity": bool(
                geometry_result["pass"] and transport_result["generic_specificity_pass"]
            ),
        },
    }


def analyze_physical_model(model: str, stage: str = "physical") -> dict[str, Any]:
    protocol = read_json(OUT / "phase435_protocol.json")
    thresholds = protocol["numeric_gates"]
    path = physical_path(stage, model)
    geometry = geometry_ledger(path)
    transport = transport_ledger(path)
    identity_pass = bool(
        geometry["max_block_reconstruction_relative_error"]
        <= thresholds["component_reconstruction_relative_error_max"]
        and geometry["max_attention_replay_relative_error"]
        <= thresholds["attention_replay_relative_error_max"]
    )
    if stage == "physical":
        contracts = sorted(
            row["contract"]
            for row in read_json(OUT / "phase435_behavior_gate.json")["eligible_model_contracts"]
            if row["model"] == model
        )
        open_audit = None
    else:
        open_gate = read_json(OUT / "phase435_open_gate.json")
        contracts = sorted(
            row["contract"]
            for row in open_gate["sealed_authorized_model_contracts"]
            if row["model"] == model
        )
        open_audit = read_json(OUT / f"phase435_{model}_physical_audit.json")
    contract_results = {}
    for contract in contracts:
        contract_results[contract] = evaluate_physical_contract(
            contract,
            geometry,
            transport,
            thresholds,
            stage=stage,
            open_freeze=(open_audit["contracts"][contract] if open_audit else None),
        )
        contract_results[contract]["gates"]["G2_component_identity_and_position_registration"] = identity_pass
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "identity": {
            "max_block_reconstruction_relative_error": geometry["max_block_reconstruction_relative_error"],
            "max_attention_replay_relative_error": geometry["max_attention_replay_relative_error"],
            "pass": identity_pass,
        },
        "contracts": contract_results,
        "geometry_ledger": geometry,
        "transport_ledger": transport,
        "source_transport_not_inferred_from_geometry": True,
        "physical": True,
        "observer": True,
        "predictive": any(
            value["gates"]["G5_frozen_holdout_prediction"]
            for value in contract_results.values()
        ),
        "causal": False,
        "single_neuron": False,
    }
    suffix = "physical_audit" if stage == "physical" else "sealed_physical_audit"
    write_json(OUT / f"phase435_{model}_{suffix}.json", output)
    return output


def analyze_open() -> dict[str, Any]:
    behavior_gate = read_json(OUT / "phase435_behavior_gate.json")
    models_with_eligible = sorted(
        {row["model"] for row in behavior_gate["eligible_model_contracts"]}
    )
    physical = {
        model: analyze_physical_model(model)
        for model in models_with_eligible
        if physical_path("physical", model).exists()
    }
    authorized = []
    contract_gates = []
    for row in behavior_gate["eligible_model_contracts"]:
        model = row["model"]
        contract = row["contract"]
        physical_contract = physical.get(model, {}).get("contracts", {}).get(contract)
        gates = {
            "G0_interface_calibration": True,
            "G1_natural_content_and_order_balance": True,
            "G2_component_identity_and_position_registration": bool(
                physical_contract
                and physical_contract["gates"]["G2_component_identity_and_position_registration"]
            ),
            "G3_label_blind_order_geometry": bool(
                physical_contract and physical_contract["gates"]["G3_label_blind_order_geometry"]
            ),
            "G4_semantic_source_transport": bool(
                physical_contract and physical_contract["gates"]["G4_semantic_source_transport"]
            ),
            "G5_frozen_holdout_prediction": bool(
                physical_contract and physical_contract["gates"]["G5_frozen_holdout_prediction"]
            ),
            "G6_generic_pairing_specificity": bool(
                physical_contract and physical_contract["gates"]["G6_generic_pairing_specificity"]
            ),
        }
        audit = {"model": model, "contract": contract, "gates": gates, "pass": all(gates.values())}
        contract_gates.append(audit)
        if audit["pass"]:
            authorized.append({"model": model, "contract": contract})
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "open",
        "behavior_gate": behavior_gate,
        "physical": physical,
        "model_contract_gates": contract_gates,
        "sealed_authorized_model_contracts": authorized,
        "sealed_unlock": bool(authorized),
        "sealed_rows_read": False,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "phase435_open_gate.json", output)
    return output


def sealed_behavior_gate(model: str, contract: str) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(behavior_path("sealed", model))
        if row["contract"] == contract
    ]
    metrics = behavior_metrics(rows)
    positions = position_metrics(rows)
    passed = bool(
        all(
            positions[position]["content"]["lcb"] >= 0.80
            for position in ("first", "second")
        )
        and positions["content_position_gap"] <= 0.05
        and metrics["other"]["ucb"] <= 0.05
    )
    return {"metrics": metrics, "positions": positions, "pass": passed}


def analyze_sealed() -> dict[str, Any]:
    open_gate = read_json(OUT / "phase435_open_gate.json")
    if not open_gate["sealed_unlock"]:
        raise RuntimeError("Phase435 sealed analysis is not authorized")
    models = sorted({row["model"] for row in open_gate["sealed_authorized_model_contracts"]})
    physical = {
        model: analyze_physical_model(model, stage="sealed")
        for model in models
        if physical_path("sealed", model).exists()
    }
    results = []
    for row in open_gate["sealed_authorized_model_contracts"]:
        model = row["model"]
        contract = row["contract"]
        behavior = sealed_behavior_gate(model, contract)
        physical_contract = physical.get(model, {}).get("contracts", {}).get(contract)
        physical_pass = bool(
            physical_contract
            and physical_contract["gates"]["G2_component_identity_and_position_registration"]
            and all(
                physical_contract["gates"][key]
                for key in (
                    "G3_label_blind_order_geometry",
                    "G4_semantic_source_transport",
                    "G5_frozen_holdout_prediction",
                    "G6_generic_pairing_specificity",
                )
            )
        )
        results.append(
            {
                "model": model,
                "contract": contract,
                "sealed_behavior": behavior,
                "sealed_physical_pass": physical_pass,
                "G7_sealed_physical_replication": bool(behavior["pass"] and physical_pass),
            }
        )
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "sealed",
        "authorized_count": len(open_gate["sealed_authorized_model_contracts"]),
        "evaluated_count": len(results),
        "results": results,
        "sealed_pass": bool(results) and all(row["G7_sealed_physical_replication"] for row in results),
        "sealed_rows_read": True,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "sealed/phase435_sealed_result.json", output)
    return output


def build_summary() -> dict[str, Any]:
    protocol = read_json(OUT / "phase435_protocol.json")
    interfaces = read_json(OUT / "phase435_interface_freeze.json")
    behavior = read_json(OUT / "phase435_behavior_gate.json")
    open_path = OUT / "phase435_open_gate.json"
    sealed_path = OUT / "sealed/phase435_sealed_result.json"
    open_gate = read_json(open_path) if open_path.exists() else None
    sealed = read_json(sealed_path) if sealed_path.exists() else None
    if not behavior["eligible_model_contracts"]:
        status = "natural_behavior_failed_physical_not_run"
        progress = 21
    elif not open_gate:
        status = "behavior_qualified_physical_pending"
        progress = 21
    elif not open_gate["sealed_unlock"]:
        status = "open_physical_gates_failed_sealed_unread"
        progress = 22
    elif not sealed:
        status = "open_physical_passed_sealed_pending"
        progress = 22
    elif sealed["sealed_pass"]:
        status = "sealed_observational_path_replication_passed"
        progress = 24
    else:
        status = "sealed_replication_failed"
        progress = 22
    summary = {
        "schema_version": "phase435_natural_relation_summary.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "status": status,
        "denominator": protocol["denominator_audit"],
        "interfaces": interfaces,
        "behavior": behavior,
        "open": open_gate,
        "sealed": sealed,
        "evidence": {
            "controlled_natural_language": True,
            "natural_corpus_sample": False,
            "physical": bool(open_gate and open_gate["physical"]),
            "observer": True,
            "predictive": bool(open_gate and open_gate["sealed_authorized_model_contracts"]),
            "causal": False,
            "single_neuron": False,
            "mechanism_closure": False,
        },
        "closure": {
            "strict_mechanisms": "0/72",
            "overall_scientific_progress_percent": progress,
            "cautious_interval_percent": [max(0, progress - 3), progress + 3],
        },
        "small_model_structural_deviation_risk": "30%-50% project-level caution; not a confidence interval",
    }
    write_json(OUT / "phase435_final_summary.json", summary)
    return summary


def publish_visual() -> dict[str, Any]:
    summary = build_summary()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        interface = summary["interfaces"]["models"][model]
        interface_id = f"phase435:{model}:interface"
        nodes.append(
            {
                "id": interface_id,
                "label": f"{model} / {interface['selected_interface']}",
                "type": "frozen_output_interface",
                "model": model,
                "layer": -1,
                "relative_depth": 0.0,
                "position_role": "interface",
                "position": [0.0, float(model_index * 7), -6.0],
                "score": 1.0 if interface["calibration_qualified"] else 0.0,
                "color": MODEL_COLORS[model],
                "size": 0.8,
                "physical": False,
                "observer": True,
                "predictive": False,
                "causal": False,
                "single_neuron": False,
                "pipeline_sealed": False,
                "evidence_level": "independent_interface_calibration",
                "show_label": True,
            }
        )
        for contract_index, contract in enumerate(CONTRACTS):
            audit = summary["behavior"]["behavior"][model]["contracts"][contract]
            holdout = (
                audit["splits"].get("behavior_holdout", {}).get("positions")
                if audit["splits"]
                else None
            )
            score = (
                statistics.mean(
                    float(holdout[position]["content"]["estimate"])
                    for position in ("first", "second")
                )
                if holdout
                else 0.0
            )
            node_id = f"phase435:{model}:{contract}:behavior"
            nodes.append(
                {
                    "id": node_id,
                    "label": f"{model} / {CONTRACT_LABELS[contract]}",
                    "type": "natural_relation_behavior",
                    "model": model,
                    "contract": contract,
                    "layer": -1,
                    "relative_depth": 0.0,
                    "position_role": "balanced_first_second",
                    "position": [float(5 + contract_index * 5), float(model_index * 7), -3.0],
                    "score": clean(score),
                    "first_score": holdout["first"]["content"]["estimate"] if holdout else 0.0,
                    "second_score": holdout["second"]["content"]["estimate"] if holdout else 0.0,
                    "color": MODEL_COLORS[model],
                    "size": 1.0 if audit["behavior_eligible"] else 0.6,
                    "physical": False,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                    "pipeline_sealed": False,
                    "evidence_level": (
                        "independent_natural_behavior_holdout"
                        if holdout
                        else "interface_gate_failed_behavior_unread"
                    ),
                    "show_label": audit["behavior_eligible"],
                }
            )
            edges.append(
                {
                    "id": f"{interface_id}->{node_id}",
                    "source": interface_id,
                    "target": node_id,
                    "type": "frozen_interface_used_by_contract",
                    "physical": False,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                    "evidence_level": "protocol_dependency",
                    "color": "#64748b",
                    "weight": 0.35,
                }
            )
    if summary.get("open"):
        for model, model_audit in summary["open"]["physical"].items():
            for contract, audit in model_audit["contracts"].items():
                geometry = audit["geometry_candidate"]
                transport = audit["transport_candidate"]
                geometry_id = f"phase435:{model}:{contract}:geometry"
                transport_id = f"phase435:{model}:{contract}:transport"
                nodes.extend(
                    [
                        {
                            "id": geometry_id,
                            "label": f"{model} L{geometry['layer']} / 顺序几何",
                            "type": "label_blind_relation_geometry",
                            "model": model,
                            "contract": contract,
                            "layer": geometry["layer"],
                            "relative_depth": geometry["relative_depth"],
                            "position_role": geometry["position_role"],
                            "position": [float(geometry["layer"]), float(MODELS.index(model) * 7), 2.0],
                            "score": geometry["geometry_effect_median"],
                            "color": "#8b5cf6",
                            "size": 1.0,
                            "physical": True,
                            "observer": True,
                            "predictive": False,
                            "causal": False,
                            "single_neuron": False,
                            "pipeline_sealed": False,
                            "evidence_level": "open_physical_holdout",
                            "show_label": True,
                        },
                        {
                            "id": transport_id,
                            "label": f"{model} L{transport['layer']} / 来源运输",
                            "type": "semantic_source_transport",
                            "model": model,
                            "contract": contract,
                            "layer": transport["layer"],
                            "relative_depth": clean(transport["layer"] / max(1, model_audit["geometry_ledger"]["layer_count"] - 1)),
                            "position_role": transport["receiver_role"],
                            "position": [float(transport["layer"]), float(MODELS.index(model) * 7 + 2), 5.0],
                            "score": transport["metrics"]["balanced_accuracy"],
                            "color": "#06b6d4",
                            "size": 1.0,
                            "physical": True,
                            "observer": True,
                            "predictive": audit["gates"]["G5_frozen_holdout_prediction"],
                            "causal": False,
                            "single_neuron": False,
                            "pipeline_sealed": False,
                            "evidence_level": "open_component_write_holdout",
                            "show_label": True,
                        },
                    ]
                )
                edges.append(
                    {
                        "id": f"{geometry_id}->{transport_id}",
                        "source": geometry_id,
                        "target": transport_id,
                        "type": "co_registered_not_causal",
                        "physical": True,
                        "observer": True,
                        "predictive": False,
                        "causal": False,
                        "single_neuron": False,
                        "evidence_level": "co_registered_open_windows",
                        "color": "#64748b",
                        "weight": 0.5,
                    }
                )
    physical_stage_run = bool(summary.get("open") and summary["open"]["physical"])
    evidence_scope = (
        "controlled-natural behavior plus open physical geometry and legal component-write transport; non-causal and non-neuronal"
        if physical_stage_run
        else "controlled-natural interface and behavior qualification only; physical, causal, and neuronal claims remain locked"
    )
    payload = {
        "schema_version": "phase435_natural_relation_graph.v1",
        "phase_id": PHASE_ID,
        "title": "Phase435 受控自然关系与顺序捷径排除图谱",
        "model": "multi_model",
        "evidence_scope": evidence_scope,
        "graph": {
            "meta": {
                "eligible_model_contracts": summary["behavior"]["eligible_model_contracts"],
                "sealed_pass": bool(summary.get("sealed") and summary["sealed"]["sealed_pass"]),
                "physical_stage_run": physical_stage_run,
                "natural_corpus_sample": False,
                "causal": False,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }
    VIS.mkdir(parents=True, exist_ok=True)
    filename = "phase435_natural_relation.json"
    write_json(VIS / filename, payload)
    manifest = {
        "schema_version": "phase435_natural_relation_manifest.v1",
        "generated_at": now(),
        "default_item_id": "phase435_natural_relation",
        "items": [
            {
                "id": "phase435_natural_relation",
                "label": "Phase435 受控自然关系与顺序捷径排除",
                "filename": filename,
                "model": "multi_model",
                "phase": 435,
                "evidence_scope": evidence_scope,
            }
        ],
    }
    write_json(VIS / "manifest.json", manifest)
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase435_natural_relation",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase435 受控自然关系与顺序捷径排除",
        "description": "独立冻结三模型输出接口，交叉记录顺序、映射和查询对象；仅合格合同可进入物理临摹。",
        "manifest_path": "/vis_data/phase435_natural_relation/manifest.json",
        "manifest_schema": "phase435_natural_relation_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase435_natural_relation",
        "models": list(MODELS),
        "evidence_scope": evidence_scope,
        "color": "#10b981",
    }
    registry["sources"] = [
        item for item in registry["sources"] if item["id"] != source["id"]
    ] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    return {"manifest": manifest, "node_count": len(nodes), "edge_count": len(edges)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("interface", "behavior", "open", "sealed", "summary"), required=True
    )
    parser.add_argument("--publish-visual", action="store_true")
    args = parser.parse_args()
    if args.stage == "interface":
        output = analyze_interface_freeze()
    elif args.stage == "behavior":
        output = analyze_behavior_gate()
    elif args.stage == "open":
        output = analyze_open()
    elif args.stage == "sealed":
        output = analyze_sealed()
    else:
        output = build_summary()
    if args.publish_visual:
        output = {"analysis": output, "visual": publish_visual()}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
