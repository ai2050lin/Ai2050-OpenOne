#!/usr/bin/env python3
"""Analyze Phase437 position factors before any physical interpretation."""

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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase435_natural_relation_analysis as a435  # noqa: E402
from phase437_position_factor_protocol import (  # noqa: E402
    BEHAVIOR_SPLITS,
    CONTRACTS,
    FROZEN_INTERFACES,
    MODELS,
    OUT,
    POST_GAPS,
    PRIMARY_VARIANTS,
    RELATION_FAMILIES,
    SCHEMA_VERSION,
    freeze,
    read_json,
    read_jsonl,
    write_json,
)


PHASE_ID = "Phase437-PositionFactorAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase437_position_factor"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODEL_COLORS = {"qwen3": "#22c55e", "glm4": "#0ea5e9", "deepseek7b": "#f97316"}
CONTRACT_LABELS = {
    "field_extract": "字段抽取",
    "natural_qa": "自然问答",
    "relation_rewrite": "关系改写",
}
FACTOR_LABELS = {
    "boundary": "边界",
    "connector": "连接词",
    "record_length": "记录长度",
    "label_order": "标签顺序",
    "relation_family": "关系族",
    "variant": "位置-间隔条件",
}
OUTER_FACTORS = ("boundary", "connector", "record_length", "label_order")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase437 non-finite scalar: {value}")
    return round(float(value), 9)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_good(row: dict[str, Any]) -> bool:
    if "semantic_content_good" in row:
        return bool(row["semantic_content_good"])
    return bool(
        row["teacher_sequence_correct"]
        and row["actual_choice"] == row["semantic_target_source"]
        and row["natural_target_first"]
        and not row["natural_opposite_first"]
        and not row["natural_revision"]
    )


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "condition_count": total,
        "semantic_content": a435.wilson(sum(semantic_good(row) for row in rows), total),
        "teacher": a435.wilson(sum(bool(row["teacher_sequence_correct"]) for row in rows), total),
        "registered_value": a435.wilson(sum(row["actual_choice"] != "other" for row in rows), total),
        "exact_format": a435.wilson(sum(bool(row["natural_interface_valid"]) for row in rows), total),
        "exact_target_format": a435.wilson(sum(bool(row["natural_exact_target_contract"]) for row in rows), total),
        "stop": a435.wilson(sum(bool(row["natural_stop_good"]) for row in rows), total),
        "other": a435.wilson(sum(row["actual_choice"] == "other" for row in rows), total),
        "choice_counts": dict(Counter(str(row["actual_choice"]) for row in rows)),
    }


def position_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        position: metrics([row for row in rows if row["target_position"] == position])
        for position in ("first", "second")
    }
    output["semantic_position_gap"] = clean(
        abs(
            float(output["first"]["semantic_content"]["estimate"])
            - float(output["second"]["semantic_content"]["estimate"])
        )
    )
    return output


def behavior_path(stage: str, model: str) -> Path:
    return OUT / stage / model / "behavior/phase435_behavior_rows.jsonl"


def materialized_path(stage: str, model: str) -> Path:
    return OUT / stage / model / "behavior/phase435_materialized_conditions.jsonl"


def physical_path(stage: str, model: str) -> Path:
    return OUT / stage / model / "physical/phase435_physical_rows.jsonl.gz"


def analyze_observer_model(model: str) -> dict[str, Any]:
    rows = read_jsonl(behavior_path("observer", model))
    thresholds = read_json(OUT / "phase437_protocol.json")["observer_gate"]
    contracts: dict[str, Any] = {}
    for contract in CONTRACTS:
        selected = [row for row in rows if row["contract"] == contract]
        if any(row["interface"] != FROZEN_INTERFACES[model][contract] for row in selected):
            raise RuntimeError(f"Phase437 observer interface drift for {model}/{contract}")
        aggregate = metrics(selected)
        positions = position_metrics(selected)
        gate_components = {
            "semantic_first_lcb": bool(
                positions["first"]["semantic_content"]["lcb"]
                >= thresholds["per_position_wilson_lcb"]
            ),
            "semantic_second_lcb": bool(
                positions["second"]["semantic_content"]["lcb"]
                >= thresholds["per_position_wilson_lcb"]
            ),
            "teacher_first_lcb": bool(
                positions["first"]["teacher"]["lcb"]
                >= thresholds["teacher_per_position_wilson_lcb"]
            ),
            "teacher_second_lcb": bool(
                positions["second"]["teacher"]["lcb"]
                >= thresholds["teacher_per_position_wilson_lcb"]
            ),
            "natural_position_gap": bool(
                positions["semantic_position_gap"]
                <= thresholds["maximum_natural_position_gap"]
            ),
            "other_ucb": bool(
                aggregate["other"]["ucb"] <= thresholds["other_wilson_ucb"]
            ),
        }
        contracts[contract] = {
            "selected_interface": FROZEN_INTERFACES[model][contract],
            "metrics": aggregate,
            "positions": positions,
            "screening_factor_effects": {
                factor: factor_effect(selected, factor)
                for factor in (*OUTER_FACTORS, "relation_family", "variant")
            },
            "gate_components": gate_components,
            "observer_qualified": bool(selected) and all(gate_components.values()),
            "format_and_stop_reported_separately": True,
        }
    return {"model": model, "row_count": len(rows), "contracts": contracts}


def analyze_observer_freeze() -> dict[str, Any]:
    freeze()
    paths = {model: behavior_path("observer", model) for model in MODELS}
    missing = [model for model, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Phase437 observer rows missing: {missing}")
    input_hashes = {model: sha256_file(path) for model, path in paths.items()}
    path = OUT / "phase437_observer_freeze.json"
    if path.exists():
        existing = read_json(path)
        if existing["input_hashes"] != input_hashes:
            raise RuntimeError("Phase437 observer rows changed after observer freeze")
        return existing
    models = {model: analyze_observer_model(model) for model in MODELS}
    qualified = [
        {"model": model, "contract": contract, "interface": payload["selected_interface"]}
        for model in MODELS
        for contract, payload in models[model]["contracts"].items()
        if payload["observer_qualified"]
    ]
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "input_hashes": input_hashes,
        "models": models,
        "qualified_model_contracts": qualified,
        "qualified_count": len(qualified),
        "phase436_interfaces_not_reselected": True,
        "phase437_other_ucb_unified_at_0_05": True,
        "format_and_stop_not_used_as_semantic_gate": True,
    }
    write_json(path, output)
    return output


def paired_variant_effect(
    rows: list[dict[str, Any]], left_variant: str, right_variant: str
) -> dict[str, Any]:
    indexed = {
        (row["semantic_group_id"], row["variant"]): row
        for row in rows
        if row["variant"] in {left_variant, right_variant}
    }
    group_ids = sorted({key[0] for key in indexed})
    pairs = [
        (indexed[(group_id, left_variant)], indexed[(group_id, right_variant)])
        for group_id in group_ids
        if (group_id, left_variant) in indexed and (group_id, right_variant) in indexed
    ]
    left_rows = [left for left, _ in pairs]
    right_rows = [right for _, right in pairs]
    signed = [int(semantic_good(left)) - int(semantic_good(right)) for left, right in pairs]
    return {
        "left_variant": left_variant,
        "right_variant": right_variant,
        "paired_group_count": len(pairs),
        "left": metrics(left_rows),
        "right": metrics(right_rows),
        "absolute_semantic_gap": clean(
            abs(
                float(metrics(left_rows)["semantic_content"]["estimate"])
                - float(metrics(right_rows)["semantic_content"]["estimate"])
            )
        ),
        "paired_signed_difference": clean(statistics.mean(signed) if signed else 0.0),
        "left_only_success": sum(value == 1 for value in signed),
        "right_only_success": sum(value == -1 for value in signed),
    }


def factor_effect(rows: list[dict[str, Any]], factor: str) -> dict[str, Any]:
    values = sorted({str(row[factor]) for row in rows})
    levels = {value: metrics([row for row in rows if str(row[factor]) == value]) for value in values}
    estimates = [float(payload["semantic_content"]["estimate"]) for payload in levels.values()]
    return {
        "factor": factor,
        "levels": levels,
        "semantic_range": clean(max(estimates) - min(estimates)) if estimates else 0.0,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return clean(ordered[index])


def distance_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in rows}):
        values = [float(row["target_to_question_token_distance"]) for row in rows if row["variant"] == variant]
        variants[variant] = {
            "count": len(values),
            "minimum": clean(min(values)) if values else 0.0,
            "maximum": clean(max(values)) if values else 0.0,
            "median": clean(statistics.median(values)) if values else 0.0,
            "mean": clean(statistics.mean(values)) if values else 0.0,
        }
    matched: dict[str, Any] = {}
    for gap in POST_GAPS:
        left_name = f"first_natural_{gap}"
        right_name = f"second_matched_{gap}"
        indexed = {(row["semantic_group_id"], row["variant"]): row for row in rows}
        errors = []
        for group_id in sorted({row["semantic_group_id"] for row in rows}):
            left = indexed.get((group_id, left_name))
            right = indexed.get((group_id, right_name))
            if left and right:
                errors.append(
                    abs(
                        float(left["target_to_question_token_distance"])
                        - float(right["target_to_question_token_distance"])
                    )
                )
        left = variants.get(left_name, {})
        right = variants.get(right_name, {})
        overlap = bool(
            left and right
            and max(float(left["minimum"]), float(right["minimum"]))
            <= min(float(left["maximum"]), float(right["maximum"]))
        )
        matched[gap] = {
            "pair_count": len(errors),
            "range_overlap": overlap,
            "median_absolute_error_tokens": clean(statistics.median(errors)) if errors else 0.0,
            "p95_absolute_error_tokens": percentile(errors, 0.95),
            "maximum_absolute_error_tokens": clean(max(errors)) if errors else 0.0,
        }
    indexed = {(row["semantic_group_id"], row["variant"]): row for row in rows}
    ordering_checks = []
    for group_id in sorted({row["semantic_group_id"] for row in rows}):
        for gap in POST_GAPS:
            first = indexed.get((group_id, f"first_natural_{gap}"))
            second = indexed.get((group_id, f"second_natural_{gap}"))
            if first and second:
                ordering_checks.append(
                    int(first["target_to_question_token_distance"])
                    > int(second["target_to_question_token_distance"])
                )
    return {
        "variants": variants,
        "matched": matched,
        "natural_ordering_check_count": len(ordering_checks),
        "first_record_is_structurally_farther_than_second": bool(ordering_checks) and all(ordering_checks),
        "distance_claim_is_conditional_not_fully_orthogonal": True,
    }


def analyze_behavior_split(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    primary = [row for row in rows if row["variant"] in PRIMARY_VARIANTS]
    aggregate = metrics(rows)
    positions = position_metrics(primary)
    ordinal = {
        gap: paired_variant_effect(rows, f"first_natural_{gap}", f"second_natural_{gap}")
        for gap in POST_GAPS
    }
    matched = {
        gap: paired_variant_effect(rows, f"first_natural_{gap}", f"second_matched_{gap}")
        for gap in POST_GAPS
    }
    post_gap = {
        position: paired_variant_effect(
            rows, f"{position}_natural_near", f"{position}_natural_far"
        )
        for position in ("first", "second")
    }
    factors = {factor: factor_effect(primary, factor) for factor in OUTER_FACTORS}
    distances = distance_audit(rows)
    maximum_ordinal = max(value["absolute_semantic_gap"] for value in ordinal.values())
    maximum_matched = max(value["absolute_semantic_gap"] for value in matched.values())
    maximum_post_gap = max(value["absolute_semantic_gap"] for value in post_gap.values())
    maximum_factor = max(value["semantic_range"] for value in factors.values())
    distance_registration = bool(
        distances["first_record_is_structurally_farther_than_second"]
        and
        all(value["range_overlap"] for value in distances["matched"].values())
        and all(
            value["median_absolute_error_tokens"]
            <= thresholds["maximum_matched_token_distance_median_error"]
            and value["p95_absolute_error_tokens"]
            <= thresholds["maximum_matched_token_distance_p95_error"]
            for value in distances["matched"].values()
        )
    )
    gate_components = {
        "semantic_first_lcb": bool(
            positions["first"]["semantic_content"]["lcb"]
            >= thresholds["discovery_and_holdout_per_position_wilson_lcb"]
        ),
        "semantic_second_lcb": bool(
            positions["second"]["semantic_content"]["lcb"]
            >= thresholds["discovery_and_holdout_per_position_wilson_lcb"]
        ),
        "teacher_first_lcb": bool(
            positions["first"]["teacher"]["lcb"]
            >= thresholds["teacher_per_position_wilson_lcb"]
        ),
        "teacher_second_lcb": bool(
            positions["second"]["teacher"]["lcb"]
            >= thresholds["teacher_per_position_wilson_lcb"]
        ),
        "natural_position_gap": maximum_ordinal <= thresholds["maximum_natural_position_gap"],
        "matched_position_gap": maximum_matched <= thresholds["maximum_matched_position_gap"],
        "post_gap_effect": maximum_post_gap <= thresholds["maximum_post_gap_effect"],
        "outer_factor_range": maximum_factor <= thresholds["maximum_outer_factor_range"],
        "other_ucb": aggregate["other"]["ucb"] <= thresholds["other_wilson_ucb"],
        "actual_token_distance_registration": distance_registration,
    }
    return {
        "metrics": aggregate,
        "primary_positions": positions,
        "ordinal_effects": ordinal,
        "matched_distance_effects": matched,
        "post_gap_effects": post_gap,
        "outer_factor_effects": factors,
        "distance_audit": distances,
        "maximum_effects": {
            "ordinal": maximum_ordinal,
            "matched": maximum_matched,
            "post_gap": maximum_post_gap,
            "outer_factor": maximum_factor,
        },
        "gate_components": gate_components,
        "pass": bool(rows) and all(gate_components.values()),
        "regression_not_used_as_mechanism_model": True,
    }


def analyze_behavior_model(model: str, observer: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(behavior_path("behavior", model))
    thresholds = read_json(OUT / "phase437_protocol.json")["behavior_gate"]
    contracts: dict[str, Any] = {}
    for contract in CONTRACTS:
        observer_payload = observer["models"][model]["contracts"][contract]
        if not observer_payload["observer_qualified"]:
            contracts[contract] = {
                "observer_qualified": False,
                "splits": {},
                "behavior_eligible": False,
            }
            continue
        selected = [row for row in rows if row["contract"] == contract]
        if any(row["interface"] != FROZEN_INTERFACES[model][contract] for row in selected):
            raise RuntimeError(f"Phase437 interface drift for {model}/{contract}")
        splits = {
            split: analyze_behavior_split(
                [row for row in selected if row["split"] == split], thresholds
            )
            for split in BEHAVIOR_SPLITS
        }
        contracts[contract] = {
            "observer_qualified": True,
            "selected_interface": FROZEN_INTERFACES[model][contract],
            "splits": splits,
            "behavior_eligible": all(value["pass"] for value in splits.values()),
            "conditional_distance_design": True,
        }
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "row_count": len(rows),
        "contracts": contracts,
    }
    write_json(OUT / f"phase437_{model}_behavior_audit.json", output)
    return output


def analyze_behavior_gate() -> dict[str, Any]:
    observer = read_json(OUT / "phase437_observer_freeze.json")
    behavior: dict[str, Any] = {}
    for model in MODELS:
        qualified = any(
            value["observer_qualified"]
            for value in observer["models"][model]["contracts"].values()
        )
        if qualified:
            if not behavior_path("behavior", model).exists():
                raise RuntimeError(f"Phase437 qualified behavior rows missing for {model}")
            behavior[model] = analyze_behavior_model(model, observer)
        else:
            behavior[model] = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                "model": model,
                "row_count": 0,
                "status": "observer_gate_failed_behavior_unread",
                "contracts": {
                    contract: {
                        "observer_qualified": False,
                        "splits": {},
                        "behavior_eligible": False,
                    }
                    for contract in CONTRACTS
                },
            }
            write_json(OUT / f"phase437_{model}_behavior_audit.json", behavior[model])
    eligible = [
        {"model": model, "contract": contract}
        for model in MODELS
        for contract, payload in behavior[model]["contracts"].items()
        if payload["behavior_eligible"]
    ]
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "behavior": behavior,
        "eligible_model_contracts": eligible,
        "eligible_count": len(eligible),
        "physical_unlock": bool(eligible),
        "sealed_rows_read": False,
        "all_failures_preserve_component_ledgers": True,
    }
    write_json(OUT / "phase437_behavior_gate.json", output)
    return output


def cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 1.0
    return clean(1.0 - dot / (left_norm * right_norm))


def baseline_balanced_accuracy(values: list[dict[str, Any]], key: str) -> float:
    recalls = []
    for source in ("source_1", "source_2"):
        selected = [row for row in values if row["actual"] == source]
        recalls.append(
            sum(row[key] == source for row in selected) / len(selected) if selected else 0.0
        )
    return clean(statistics.mean(recalls))


def transport_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
    base = a435.choice_metrics(values)
    baselines = {
        key: baseline_balanced_accuracy(values, key)
        for key in ("first_record_prediction", "second_record_prediction")
    }
    baselines["majority_prediction"] = 0.5 if values else 0.0
    base["surface_baselines"] = baselines
    base["best_surface_baseline"] = max(baselines.values()) if baselines else 0.0
    base["over_best_surface_baseline"] = clean(
        float(base["balanced_accuracy"]) - float(base["best_surface_baseline"])
    )
    return base


def physical_ledgers(stage: str, model: str) -> dict[str, Any]:
    metadata = {
        row["condition_id"]: row for row in read_jsonl(materialized_path(stage, model))
    }
    geometry_acc: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {"effects": [], "families": defaultdict(list)}
    )
    transport_acc: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    pending_pairs: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    current_group: str | None = None
    group_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    max_block_error = 0.0
    max_replay_error = 0.0
    row_count = 0
    layer_count = 0

    def process_group(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
        if not items:
            return
        by_kind: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(dict)
        for trace, meta in items:
            by_kind[trace["condition_kind"]][meta["variant"]] = (trace, meta)
        for kind, variants in by_kind.items():
            required = ("first_natural_near", "second_natural_near", "second_matched_near")
            if not all(value in variants for value in required):
                continue
            first = variants[required[0]][0]
            second = variants[required[1]][0]
            matched = variants[required[2]][0]
            meta = variants[required[0]][1]
            summary: dict[tuple[int, str], dict[str, Any]] = {}
            for layer_index, layer in enumerate(first["layers"]):
                for position, first_payload in layer["position_metrics"].items():
                    second_payload = second["layers"][layer_index]["position_metrics"][position]
                    matched_payload = matched["layers"][layer_index]["position_metrics"][position]
                    within = statistics.median(
                        (
                            cosine_distance(first_payload["state_sketch"], second_payload["state_sketch"]),
                            cosine_distance(first_payload["state_sketch"], matched_payload["state_sketch"]),
                        )
                    )
                    summary[(layer_index, position)] = {
                        "within": within,
                        "reference": first_payload["state_sketch"],
                    }
            pair_key = (first["contract"], first["physical_fold"], kind, str(meta["geometry_pair_id"]))
            payload = {
                "contract": first["contract"],
                "fold": first["physical_fold"],
                "kind": kind,
                "family": first["relation_family"],
                "summary": summary,
            }
            mate = pending_pairs.pop(pair_key, None)
            if mate is None:
                pending_pairs[pair_key] = payload
                continue
            for cell_key, cell in summary.items():
                other = mate["summary"][cell_key]
                different = cosine_distance(cell["reference"], other["reference"])
                effect = different - statistics.mean((cell["within"], other["within"]))
                key = (first["contract"], first["physical_fold"], kind, cell_key[0], cell_key[1])
                geometry_acc[key]["effects"].append(effect)
                geometry_acc[key]["families"][first["relation_family"]].append(effect)
                geometry_acc[key]["families"][mate["family"]].append(effect)

    for trace in a435.iter_gzip_rows(physical_path(stage, model)):
        row_count += 1
        meta = metadata[trace["condition_id"]]
        layer_count = max(layer_count, len(trace["layers"]))
        for layer in trace["layers"]:
            for payload in layer["position_metrics"].values():
                max_block_error = max(
                    max_block_error,
                    float(payload["block_reconstruction_relative_error"]),
                )
            for receiver, receiver_payload in layer["receiver_metrics"].items():
                max_replay_error = max(
                    max_replay_error,
                    float(receiver_payload["attention_replay_relative_error"]),
                )
                source = receiver_payload["source_partition"]
                score = (
                    float(source["source_1_record"]["source_1_minus_source_2_margin_write"])
                    - float(source["source_2_record"]["source_1_minus_source_2_margin_write"])
                )
                first_source = meta["record_entries"][0]["value_source"]
                second_source = meta["record_entries"][1]["value_source"]
                transport_acc[
                    (
                        trace["contract"], trace["physical_fold"], trace["condition_kind"],
                        int(layer["layer"]), receiver,
                    )
                ].append(
                    {
                        "actual": trace["semantic_target_source"],
                        "predicted": "source_1" if score >= 0 else "source_2",
                        "score": score,
                        "target_position": trace["target_position"],
                        "first_record_prediction": first_source,
                        "second_record_prediction": second_source,
                    }
                )
        if current_group is not None and trace["semantic_group_id"] != current_group:
            process_group(group_rows)
            group_rows = []
        current_group = trace["semantic_group_id"]
        group_rows.append((trace, meta))
    process_group(group_rows)
    if pending_pairs:
        raise RuntimeError(f"Phase437 unmatched physical geometry pairs: {len(pending_pairs)}")

    geometry_cells = [
        {
            "contract": key[0],
            "fold": key[1],
            "condition_kind": key[2],
            "layer": key[3],
            "relative_depth": clean(key[3] / max(1, layer_count - 1)),
            "position_role": key[4],
            "pair_count": len(values["effects"]),
            "geometry_effect_median": clean(statistics.median(values["effects"])),
            "relation_family_effects": {
                family: clean(statistics.median(values["families"].get(family, [0.0])))
                for family in RELATION_FAMILIES
            },
            "output_label_blind": True,
        }
        for key, values in sorted(geometry_acc.items())
    ]
    transport_cells = [
        {
            "contract": key[0],
            "fold": key[1],
            "condition_kind": key[2],
            "layer": key[3],
            "receiver_role": key[4],
            "metrics": transport_metrics(values),
        }
        for key, values in sorted(transport_acc.items())
    ]
    return {
        "row_count": row_count,
        "layer_count": layer_count,
        "max_block_reconstruction_relative_error": clean(max_block_error),
        "max_attention_replay_relative_error": clean(max_replay_error),
        "geometry_cells": geometry_cells,
        "transport_cells": transport_cells,
    }


def one_cell(cells: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matches = [row for row in cells if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Phase437 expected one cell for {criteria}; found {len(matches)}")
    return matches[0]


def evaluate_physical_contract(
    contract: str,
    ledgers: dict[str, Any],
    thresholds: dict[str, Any],
    stage: str,
    open_freeze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage == "physical":
        geometry_discovery = [
            row for row in ledgers["geometry_cells"]
            if row["contract"] == contract and row["fold"] == "discovery"
            and row["condition_kind"] == "candidate"
        ]
        transport_discovery = [
            row for row in ledgers["transport_cells"]
            if row["contract"] == contract and row["fold"] == "discovery"
            and row["condition_kind"] == "candidate"
        ]
        geometry_window = max(
            geometry_discovery,
            key=lambda row: (float(row["geometry_effect_median"]), -int(row["layer"]), row["position_role"]),
        )
        transport_window = max(
            transport_discovery,
            key=lambda row: (
                float(row["metrics"]["over_best_surface_baseline"]),
                float(row["metrics"]["balanced_accuracy"]),
                -int(row["layer"]),
                row["receiver_role"],
            ),
        )
        fold = "holdout"
    else:
        if open_freeze is None:
            raise RuntimeError("Phase437 sealed physical analysis requires open windows")
        geometry_window = open_freeze["geometry_window"]
        transport_window = open_freeze["transport_window"]
        fold = "sealed"
    candidate_geometry = one_cell(
        ledgers["geometry_cells"], contract=contract, fold=fold,
        condition_kind="candidate", layer=geometry_window["layer"],
        position_role=geometry_window["position_role"],
    )
    geometry_controls = [
        row for row in ledgers["geometry_cells"]
        if row["contract"] == contract and row["fold"] == fold
        and row["condition_kind"] != "candidate"
        and row["layer"] == geometry_window["layer"]
        and row["position_role"] == geometry_window["position_role"]
    ]
    max_geometry_control = max(
        (float(row["geometry_effect_median"]) for row in geometry_controls), default=0.0
    )
    positive_families = sum(
        float(candidate_geometry["relation_family_effects"][family])
        >= thresholds["label_blind_geometry_effect_min"]
        for family in RELATION_FAMILIES
    )
    geometry_pass = bool(
        float(candidate_geometry["geometry_effect_median"])
        >= thresholds["label_blind_geometry_effect_min"]
        and positive_families >= thresholds["geometry_positive_relation_families_min"]
        and float(candidate_geometry["geometry_effect_median"]) - max_geometry_control
        >= thresholds["geometry_candidate_over_control_min"]
    )
    candidate_transport = one_cell(
        ledgers["transport_cells"], contract=contract, fold=fold,
        condition_kind="candidate", layer=transport_window["layer"],
        receiver_role=transport_window["receiver_role"],
    )
    transport_controls = [
        row for row in ledgers["transport_cells"]
        if row["contract"] == contract and row["fold"] == fold
        and row["condition_kind"] != "candidate"
        and row["layer"] == transport_window["layer"]
        and row["receiver_role"] == transport_window["receiver_role"]
    ]
    max_transport_control = max(
        (float(row["metrics"]["balanced_accuracy"]) for row in transport_controls), default=0.0
    )
    transport = candidate_transport["metrics"]
    source_transport_pass = all(
        float(transport["per_source"][source]["lcb"])
        >= thresholds["transport_per_source_wilson_lcb_min"]
        for source in ("source_1", "source_2")
    )
    prediction_pass = bool(
        float(transport["balanced_accuracy"]) >= thresholds["transport_balanced_accuracy_min"]
        and float(transport["position_gap"]) <= thresholds["transport_maximum_position_gap"]
        and float(transport["over_best_surface_baseline"])
        >= thresholds["transport_over_best_surface_baseline_min"]
    )
    specificity_pass = bool(
        float(transport["balanced_accuracy"]) - max_transport_control
        >= thresholds["transport_candidate_over_control_accuracy_min"]
    )
    return {
        "contract": contract,
        "stage": stage,
        "geometry_window": {
            "layer": geometry_window["layer"],
            "position_role": geometry_window["position_role"],
            "selected_without_holdout_access": stage == "physical",
        },
        "transport_window": {
            "layer": transport_window["layer"],
            "receiver_role": transport_window["receiver_role"],
            "selected_without_holdout_access": stage == "physical",
        },
        "geometry_candidate": candidate_geometry,
        "maximum_geometry_control_effect": clean(max_geometry_control),
        "positive_relation_family_count": positive_families,
        "transport_candidate": candidate_transport,
        "maximum_transport_control_balanced_accuracy": clean(max_transport_control),
        "gates": {
            "G3_label_blind_relation_geometry": geometry_pass,
            "G4_semantic_source_transport": source_transport_pass,
            "G5_frozen_holdout_prediction": prediction_pass,
            "G6_control_specificity": bool(geometry_pass and specificity_pass),
        },
    }


def analyze_physical_model(model: str, stage: str = "physical") -> dict[str, Any]:
    protocol = read_json(OUT / "phase437_protocol.json")
    thresholds = protocol["physical_numeric_gates"]
    ledgers = physical_ledgers(stage, model)
    identity_pass = bool(
        ledgers["max_block_reconstruction_relative_error"]
        <= thresholds["component_reconstruction_relative_error_max"]
        and ledgers["max_attention_replay_relative_error"]
        <= thresholds["attention_replay_relative_error_max"]
    )
    if stage == "physical":
        rows = [
            row for row in read_json(OUT / "phase437_behavior_gate.json")["eligible_model_contracts"]
            if row["model"] == model
        ]
        prior = None
    else:
        open_gate = read_json(OUT / "phase437_open_gate.json")
        rows = [row for row in open_gate["sealed_authorized_model_contracts"] if row["model"] == model]
        prior = read_json(OUT / f"phase437_{model}_physical_audit.json")
    contracts = {}
    for item in rows:
        contract = item["contract"]
        contracts[contract] = evaluate_physical_contract(
            contract, ledgers, thresholds, stage,
            open_freeze=(prior["contracts"][contract] if prior else None),
        )
        contracts[contract]["gates"]["G2_component_identity_and_position_registration"] = identity_pass
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "identity": {
            "max_block_reconstruction_relative_error": ledgers["max_block_reconstruction_relative_error"],
            "max_attention_replay_relative_error": ledgers["max_attention_replay_relative_error"],
            "pass": identity_pass,
        },
        "contracts": contracts,
        "ledgers": ledgers,
        "physical": True,
        "observer": True,
        "predictive": any(value["gates"]["G5_frozen_holdout_prediction"] for value in contracts.values()),
        "causal": False,
        "single_neuron": False,
    }
    suffix = "physical_audit" if stage == "physical" else "sealed_physical_audit"
    write_json(OUT / f"phase437_{model}_{suffix}.json", output)
    return output


def analyze_open() -> dict[str, Any]:
    behavior = read_json(OUT / "phase437_behavior_gate.json")
    models = sorted({row["model"] for row in behavior["eligible_model_contracts"]})
    physical = {
        model: analyze_physical_model(model)
        for model in models if physical_path("physical", model).exists()
    }
    audits = []
    authorized = []
    for row in behavior["eligible_model_contracts"]:
        model, contract = row["model"], row["contract"]
        candidate = physical.get(model, {}).get("contracts", {}).get(contract)
        gates = {
            "G0_observer_schema_and_semantic_gate": True,
            "G1_factorized_behavior_discovery_and_holdout": True,
            "G2_actual_token_distance_and_component_registration": bool(
                candidate and candidate["gates"]["G2_component_identity_and_position_registration"]
            ),
            "G3_label_blind_relation_geometry": bool(
                candidate and candidate["gates"]["G3_label_blind_relation_geometry"]
            ),
            "G4_semantic_source_transport": bool(
                candidate and candidate["gates"]["G4_semantic_source_transport"]
            ),
            "G5_frozen_holdout_prediction": bool(
                candidate and candidate["gates"]["G5_frozen_holdout_prediction"]
            ),
            "G6_control_specificity": bool(
                candidate and candidate["gates"]["G6_control_specificity"]
            ),
        }
        audit = {"model": model, "contract": contract, "gates": gates, "pass": all(gates.values())}
        audits.append(audit)
        if audit["pass"]:
            authorized.append({"model": model, "contract": contract})
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "open",
        "physical": physical,
        "model_contract_gates": audits,
        "sealed_authorized_model_contracts": authorized,
        "sealed_unlock": bool(authorized),
        "sealed_rows_read": False,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "phase437_open_gate.json", output)
    return output


def sealed_behavior(model: str, contract: str) -> dict[str, Any]:
    rows = [
        row for row in read_jsonl(behavior_path("sealed", model))
        if row["contract"] == contract and row["condition_kind"] == "candidate"
    ]
    natural = [row for row in rows if row["variant"] in {"first_natural_near", "second_natural_near"}]
    aggregate = metrics(natural)
    positions = position_metrics(natural)
    passed = bool(
        all(positions[position]["semantic_content"]["lcb"] >= 0.80 for position in ("first", "second"))
        and positions["semantic_position_gap"] <= 0.05
        and aggregate["other"]["ucb"] <= 0.05
    )
    return {"metrics": aggregate, "positions": positions, "pass": passed}


def analyze_sealed() -> dict[str, Any]:
    open_gate = read_json(OUT / "phase437_open_gate.json")
    if not open_gate["sealed_unlock"]:
        raise RuntimeError("Phase437 sealed analysis is not authorized")
    models = sorted({row["model"] for row in open_gate["sealed_authorized_model_contracts"]})
    physical = {
        model: analyze_physical_model(model, "sealed")
        for model in models if physical_path("sealed", model).exists()
    }
    results = []
    for row in open_gate["sealed_authorized_model_contracts"]:
        model, contract = row["model"], row["contract"]
        behavior = sealed_behavior(model, contract)
        candidate = physical.get(model, {}).get("contracts", {}).get(contract)
        physical_pass = bool(candidate and all(candidate["gates"].values()))
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
        "results": results,
        "sealed_pass": bool(results) and all(row["G7_sealed_physical_replication"] for row in results),
        "sealed_rows_read": True,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "sealed/phase437_sealed_result.json", output)
    return output


def build_summary() -> dict[str, Any]:
    protocol = read_json(OUT / "phase437_protocol.json")
    observer = read_json(OUT / "phase437_observer_freeze.json")
    behavior_path_value = OUT / "phase437_behavior_gate.json"
    open_path = OUT / "phase437_open_gate.json"
    sealed_path = OUT / "sealed/phase437_sealed_result.json"
    behavior = read_json(behavior_path_value) if behavior_path_value.exists() else None
    open_gate = read_json(open_path) if open_path.exists() else None
    sealed = read_json(sealed_path) if sealed_path.exists() else None
    if not observer["qualified_model_contracts"]:
        status, progress = "strict_observer_failed_behavior_unread", 21
    elif not behavior or not behavior["eligible_model_contracts"]:
        status, progress = "factorized_behavior_failed_physical_unread", 21
    elif not open_gate or not open_gate["sealed_unlock"]:
        status, progress = "open_physical_gates_failed_or_pending", 22
    elif not sealed:
        status, progress = "open_physical_passed_sealed_pending", 22
    elif sealed["sealed_pass"]:
        status, progress = "sealed_observational_path_replication_passed", 24
    else:
        status, progress = "sealed_replication_failed", 22
    full_behavior_executed = bool(
        behavior
        and any(
            int(payload.get("row_count", 0)) > 0
            for payload in behavior["behavior"].values()
        )
    )
    summary = {
        "schema_version": "phase437_position_factor_summary.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "status": status,
        "denominator": protocol["denominator_audit"],
        "phase436_threshold_audit": protocol["phase436_threshold_audit"],
        "observer": observer,
        "behavior": behavior,
        "open": open_gate,
        "sealed": sealed,
        "evidence": {
            "observer_factor_screening": True,
            "full_position_factor_behavior_executed": full_behavior_executed,
            "actual_token_distance_registered_at_observer": True,
            "physical": bool(open_gate and open_gate["physical"]),
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
    }
    write_json(OUT / "phase437_final_summary.json", summary)
    return summary


def publish_visual() -> dict[str, Any]:
    summary = build_summary()
    nodes = []
    edges = []
    for model_index, model in enumerate(MODELS):
        for contract_index, contract in enumerate(CONTRACTS):
            observer = summary["observer"]["models"][model]["contracts"][contract]
            observer_id = f"phase437:{model}:{contract}:observer"
            nodes.append(
                {
                    "id": observer_id,
                    "label": f"{model} / {CONTRACT_LABELS[contract]}观察门",
                    "type": "factorized_semantic_observer",
                    "model": model,
                    "contract": contract,
                    "interface": observer["selected_interface"],
                    "layer": -1,
                    "relative_depth": 0.0,
                    "position_role": "first_second_balanced",
                    "position": [float(contract_index * 6), float(model_index * 7), -5.0],
                    "score": clean(statistics.mean(float(observer["positions"][p]["semantic_content"]["estimate"]) for p in ("first", "second"))),
                    "color": MODEL_COLORS[model],
                    "size": 1.0 if observer["observer_qualified"] else 0.6,
                    "physical": False,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                    "pipeline_sealed": False,
                    "evidence_level": "strict_fresh_observer_calibration",
                    "show_label": observer["observer_qualified"],
                }
            )
            top_factor = max(
                observer["screening_factor_effects"],
                key=lambda key: float(
                    observer["screening_factor_effects"][key]["semantic_range"]
                ),
            )
            for factor_index, (factor, factor_payload) in enumerate(
                observer["screening_factor_effects"].items()
            ):
                factor_id = f"phase437:{model}:{contract}:factor:{factor}"
                factor_range = float(factor_payload["semantic_range"])
                nodes.append(
                    {
                        "id": factor_id,
                        "label": f"{CONTRACT_LABELS[contract]} / {FACTOR_LABELS.get(factor, factor)}",
                        "type": "observer_factor_response",
                        "model": model,
                        "contract": contract,
                        "factor": factor,
                        "layer": -1,
                        "relative_depth": 0.0,
                        "position_role": "observer_factor_screening",
                        "position": [
                            float(contract_index * 6 + 1 + (factor_index % 3) * 0.8),
                            float(model_index * 7 + (factor_index // 3) * 1.2),
                            -3.0,
                        ],
                        "score": clean(factor_range),
                        "factor_levels": factor_payload["levels"],
                        "color": MODEL_COLORS[model],
                        "size": clean(0.55 + min(0.8, factor_range * 2.0)),
                        "physical": False,
                        "observer": True,
                        "predictive": False,
                        "causal": False,
                        "single_neuron": False,
                        "pipeline_sealed": False,
                        "evidence_level": "balanced_observer_factor_screening",
                        "show_label": factor == top_factor and factor_range >= 0.05,
                    }
                )
                edges.append(
                    {
                        "id": f"{observer_id}->{factor_id}",
                        "source": observer_id,
                        "target": factor_id,
                        "type": "screening_factor_decomposition",
                        "physical": False,
                        "observer": True,
                        "predictive": False,
                        "causal": False,
                        "single_neuron": False,
                        "evidence_level": "balanced_observer_contrast",
                        "color": "#94a3b8",
                        "weight": clean(max(0.2, min(1.0, factor_range * 3.0))),
                    }
                )
            if summary.get("behavior"):
                behavior = summary["behavior"]["behavior"][model]["contracts"][contract]
                if behavior["splits"]:
                    holdout = behavior["splits"]["behavior_holdout"]
                    behavior_id = f"phase437:{model}:{contract}:behavior"
                    nodes.append(
                        {
                            "id": behavior_id,
                            "label": f"{model} / {CONTRACT_LABELS[contract]}因素留出",
                            "type": "position_factor_behavior",
                            "model": model,
                            "contract": contract,
                            "layer": -1,
                            "relative_depth": 0.0,
                            "position_role": "factorized_behavior_holdout",
                            "position": [float(contract_index * 6 + 2), float(model_index * 7), -1.0],
                            "score": clean(statistics.mean(float(holdout["primary_positions"][p]["semantic_content"]["estimate"]) for p in ("first", "second"))),
                            "ordinal_effect": holdout["maximum_effects"]["ordinal"],
                            "matched_distance_effect": holdout["maximum_effects"]["matched"],
                            "outer_factor_effect": holdout["maximum_effects"]["outer_factor"],
                            "color": MODEL_COLORS[model],
                            "size": 1.0,
                            "physical": False,
                            "observer": True,
                            "predictive": False,
                            "causal": False,
                            "single_neuron": False,
                            "pipeline_sealed": False,
                            "evidence_level": "independent_factorized_behavior_holdout",
                            "show_label": behavior["behavior_eligible"],
                        }
                    )
                    edges.append(
                        {
                            "id": f"{observer_id}->{behavior_id}",
                            "source": observer_id,
                            "target": behavior_id,
                            "type": "frozen_observer_application",
                            "physical": False,
                            "observer": True,
                            "predictive": False,
                            "causal": False,
                            "single_neuron": False,
                            "evidence_level": "protocol_dependency",
                            "color": "#64748b",
                            "weight": 0.5,
                        }
                    )
    physical_stage_run = bool(summary.get("open") and summary["open"]["physical"])
    evidence_scope = (
        "factorized behavior plus gate-authorized open physical ledgers; non-causal"
        if physical_stage_run
        else "observer and factorized behavior only; physical, neuron and causal claims remain locked"
    )
    payload = {
        "schema_version": "phase437_position_factor_graph.v1",
        "phase_id": PHASE_ID,
        "title": "Phase437 记录位置因素分解",
        "model": "multi_model",
        "evidence_scope": evidence_scope,
        "graph": {
            "meta": {
                "qualified_observers": summary["observer"]["qualified_model_contracts"],
                "eligible_behavior": (
                    summary["behavior"]["eligible_model_contracts"] if summary.get("behavior") else []
                ),
                "physical_stage_run": physical_stage_run,
                "causal": False,
                "single_neuron": False,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }
    VIS.mkdir(parents=True, exist_ok=True)
    filename = "phase437_position_factor.json"
    write_json(VIS / filename, payload)
    manifest = {
        "schema_version": "phase437_position_factor_manifest.v1",
        "generated_at": now(),
        "default_item_id": "phase437_position_factor",
        "items": [
            {
                "id": "phase437_position_factor",
                "label": "Phase437 记录位置因素分解",
                "filename": filename,
                "model": "multi_model",
                "phase": 437,
                "evidence_scope": evidence_scope,
            }
        ],
    }
    write_json(VIS / "manifest.json", manifest)
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase437_position_factor",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase437 记录位置因素分解",
        "description": "拆分顺序、真实词元距离、边界、连接词、长度和标签顺序；未过门时不绘制物理或神经元路径。",
        "manifest_path": "/vis_data/phase437_position_factor/manifest.json",
        "manifest_schema": "phase437_position_factor_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase437_position_factor",
        "models": list(MODELS),
        "evidence_scope": evidence_scope,
        "color": "#eab308",
    }
    registry["sources"] = [item for item in registry["sources"] if item["id"] != source["id"]] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    return {"manifest": manifest, "node_count": len(nodes), "edge_count": len(edges)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("observer", "behavior", "open", "sealed", "summary"), required=True)
    parser.add_argument("--publish-visual", action="store_true")
    args = parser.parse_args()
    if args.stage == "observer":
        output = analyze_observer_freeze()
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
