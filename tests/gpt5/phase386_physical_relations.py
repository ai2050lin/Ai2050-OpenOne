#!/usr/bin/env python3
"""Run the one-time physical holdout evaluation for frozen Phase386 relations."""

from __future__ import annotations

import functools
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase386_calibration_relations import (  # noqa: E402
    COORDINATES,
    CONDITIONS,
    cosine,
    extract_family,
    wrong_depth,
)


PHASE_ROOT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
COLLECTION = PHASE_ROOT / "collection"
DISCOVERY_CASES = PHASE_ROOT / "protocol/private/phase386_discovery_cases.jsonl"
PHYSICAL_CASES = PHASE_ROOT / "protocol/private/phase386_physical_holdout_cases.jsonl"
FROZEN = PHASE_ROOT / "phase386_frozen_physical_candidates.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def grouped_cases(path: Path) -> dict[
    tuple[str, str, str], dict[str, dict[str, Any]]
]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in read_jsonl(path):
        key = (
            case["private_execution_model"],
            case["mechanism_id"],
            case["phase386_public_parallel_group_id"],
        )
        grouped[key][case["contrast_condition"]] = case
    if any(set(rows) != set(CONDITIONS) for rows in grouped.values()):
        raise RuntimeError("Incomplete Phase386 physical condition group")
    return grouped


def main() -> None:
    audit = read_json(PHASE_ROOT / "phase386_physical_collection_summary.json")
    if not audit["authorization"]["physical_holdout_candidate_evaluation"]:
        raise RuntimeError("Phase386 physical evaluation is not authorized")
    protocol = read_json(PHASE_ROOT / "phase386_physical_holdout_protocol.json")
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != protocol[
        "candidate_file_sha256"
    ]:
        raise RuntimeError("Phase386 physical candidate checksum changed")
    candidates = read_jsonl(FROZEN)
    if len(candidates) != protocol["frozen_candidate_count"]:
        raise RuntimeError("Phase386 physical candidate count changed")
    contract = read_json(PHASE_ROOT / "phase386_relation_contract.json")
    grouped = {
        "discovery": grouped_cases(DISCOVERY_CASES),
        "physical_holdout": grouped_cases(PHYSICAL_CASES),
    }
    group_ids: dict[tuple[str, str, str], list[str]] = {}
    mechanisms = {row["mechanism_id"] for row in candidates}
    for split, expected in (("discovery", 8), ("physical_holdout", 4)):
        for model in MODELS:
            for mechanism in mechanisms:
                ids = sorted(
                    group_id
                    for candidate_model, candidate_mechanism, group_id in grouped[split]
                    if candidate_model == model and candidate_mechanism == mechanism
                )
                if len(ids) != expected:
                    raise RuntimeError(
                        f"Phase386 {split}/{model}/{mechanism}: {len(ids)} != {expected}"
                    )
                group_ids[(split, model, mechanism)] = ids

    @functools.lru_cache(maxsize=None)
    def effects(
        split: str,
        model: str,
        mechanism: str,
        group_id: str,
        layer_index: int,
        family: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cases = grouped[split][(model, mechanism, group_id)]
        values: dict[str, torch.Tensor] = {}
        for condition in CONDITIONS:
            case = cases[condition]
            payload = torch.load(
                COLLECTION
                / split
                / "private/models"
                / model
                / case["blind_case_id"]
                / f"layer_{layer_index:03d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            values[condition] = extract_family(payload, family)
        return (
            values[CONDITIONS[0]] - values[CONDITIONS[1]],
            values[CONDITIONS[2]] - values[CONDITIONS[3]],
        )

    wrong_time_map = contract["calibration_prediction"]["controls"]["wrong_time"]
    relation_min = contract["calibration_prediction"][
        "calibration_relation_median_min"
    ]
    positive_min = contract["calibration_prediction"][
        "calibration_positive_group_rate_min"
    ]
    margin_min = contract["calibration_prediction"][
        "prediction_control_margin_min"
    ]
    prediction_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        mechanism = candidate["mechanism_id"]
        family = candidate["vector_family"]
        source = candidate["source_coordinate"]
        target = candidate["target_coordinate"]
        source_index = COORDINATES.index(source)
        target_index = COORDINATES.index(target)
        wrong_time_coordinate = wrong_time_map[f"{source}->{target}"]
        wrong_time_index = COORDINATES.index(wrong_time_coordinate)
        per_model = []
        for model in MODELS:
            layer_index = int(candidate["model_layers"][model])
            layer_count = int(
                read_json(
                    COLLECTION
                    / "physical_holdout/models"
                    / model
                    / "manifest.json"
                )["layer_count"]
            )
            wrong_layer = wrong_depth(layer_index, layer_count)
            discovery_ids = group_ids[("discovery", model, mechanism)]
            physical_ids = group_ids[("physical_holdout", model, mechanism)]
            relations = {"lex_x": [], "lex_y": []}
            predictions = {
                variant: {
                    "actual": [],
                    "shuffled_pair": [],
                    "wrong_time": [],
                    "wrong_depth": [],
                }
                for variant in ("lex_x", "lex_y")
            }
            for physical_group in physical_ids:
                physical_x, physical_y = effects(
                    "physical_holdout",
                    model,
                    mechanism,
                    physical_group,
                    layer_index,
                    family,
                )
                for variant, physical_effect in (
                    ("lex_x", physical_x),
                    ("lex_y", physical_y),
                ):
                    relations[variant].append(
                        cosine(
                            physical_effect[source_index],
                            physical_effect[target_index],
                        )
                    )
                    library = [
                        effects(
                            "discovery",
                            model,
                            mechanism,
                            group_id,
                            layer_index,
                            family,
                        )[0 if variant == "lex_x" else 1]
                        for group_id in discovery_ids
                    ]
                    similarities = [
                        cosine(physical_effect[source_index], row[source_index])
                        for row in library
                    ]
                    nearest = max(
                        range(len(library)),
                        key=lambda index: (similarities[index], -index),
                    )
                    shuffled = (nearest + 1) % len(library)
                    wrong_depth_effect = effects(
                        "discovery",
                        model,
                        mechanism,
                        discovery_ids[nearest],
                        wrong_layer,
                        family,
                    )[0 if variant == "lex_x" else 1]
                    actual_target = physical_effect[target_index]
                    values = {
                        "actual": cosine(actual_target, library[nearest][target_index]),
                        "shuffled_pair": cosine(
                            actual_target, library[shuffled][target_index]
                        ),
                        "wrong_time": cosine(
                            actual_target, library[nearest][wrong_time_index]
                        ),
                        "wrong_depth": cosine(
                            actual_target, wrong_depth_effect[target_index]
                        ),
                    }
                    for name, value in values.items():
                        predictions[variant][name].append(value)
                    prediction_rows.append(
                        {
                            "schema_version": "60.13.0",
                            "phase_id": "Phase386-PhysicalRelations",
                            "candidate_id": candidate["candidate_id"],
                            "model": model,
                            "physical_group_id": physical_group,
                            "lexical_variant": variant,
                            "nearest_source_cosine": similarities[nearest],
                            "actual_prediction_cosine": values["actual"],
                            "shuffled_pair_cosine": values["shuffled_pair"],
                            "wrong_time_cosine": values["wrong_time"],
                            "wrong_depth_cosine": values["wrong_depth"],
                        }
                    )
            variant_metrics = {}
            for variant in ("lex_x", "lex_y"):
                actual = median(predictions[variant]["actual"])
                controls = {
                    name: median(values)
                    for name, values in predictions[variant].items()
                    if name != "actual"
                }
                variant_metrics[variant] = {
                    "median_relation_cosine": median(relations[variant]),
                    "positive_relation_group_rate": sum(
                        value > 0 for value in relations[variant]
                    )
                    / len(relations[variant]),
                    "median_actual_prediction_cosine": actual,
                    "median_shuffled_pair_cosine": controls["shuffled_pair"],
                    "median_wrong_time_cosine": controls["wrong_time"],
                    "median_wrong_depth_cosine": controls["wrong_depth"],
                    "shuffled_pair_advantage": actual - controls["shuffled_pair"],
                    "wrong_time_advantage": actual - controls["wrong_time"],
                    "wrong_depth_advantage": actual - controls["wrong_depth"],
                }
                variant_metrics[variant]["relation_replication_pass"] = bool(
                    variant_metrics[variant]["median_relation_cosine"]
                    >= relation_min
                    and variant_metrics[variant]["positive_relation_group_rate"]
                    >= positive_min
                )
                variant_metrics[variant]["all_prediction_controls_pass"] = bool(
                    variant_metrics[variant]["shuffled_pair_advantage"] >= margin_min
                    and variant_metrics[variant]["wrong_time_advantage"] >= margin_min
                    and variant_metrics[variant]["wrong_depth_advantage"] >= margin_min
                )
            model_row = {
                "schema_version": "60.13.0",
                "phase_id": "Phase386-PhysicalRelations",
                "candidate_id": candidate["candidate_id"],
                "model": model,
                "mechanism_id": mechanism,
                "vector_family": family,
                "source_coordinate": source,
                "target_coordinate": target,
                "layer_index": layer_index,
                "physical_group_count": len(physical_ids),
                "lex_x": variant_metrics["lex_x"],
                "lex_y": variant_metrics["lex_y"],
                "both_lexical_relation_replication_pass": all(
                    variant_metrics[variant]["relation_replication_pass"]
                    for variant in ("lex_x", "lex_y")
                ),
                "both_lexical_prediction_controls_pass": all(
                    variant_metrics[variant]["all_prediction_controls_pass"]
                    for variant in ("lex_x", "lex_y")
                ),
            }
            model_rows.append(model_row)
            per_model.append(model_row)
        relation_all = all(
            row["both_lexical_relation_replication_pass"] for row in per_model
        )
        prediction_all = all(
            row["both_lexical_prediction_controls_pass"] for row in per_model
        )
        candidate_rows.append(
            {
                **candidate,
                "physical_holdout_used": True,
                "physical_relation_replication_all_three_models": relation_all,
                "physical_prediction_controls_all_three_models": prediction_all,
                "physical_predictive_relation_path_gate_pass": relation_all
                and prediction_all,
                "causal_path_claim": False,
            }
        )
        print(
            f"[Phase386 physical] {len(candidate_rows)}/{len(candidates)} "
            f"survive={sum(row['physical_predictive_relation_path_gate_pass'] for row in candidate_rows)}",
            flush=True,
        )

    write_jsonl(PHASE_ROOT / "private/phase386_physical_prediction_rows.jsonl", prediction_rows)
    write_jsonl(PHASE_ROOT / "phase386_physical_model_rows.jsonl", model_rows)
    write_jsonl(PHASE_ROOT / "phase386_physical_candidate_rows.jsonl", candidate_rows)
    survivors = [
        row
        for row in candidate_rows
        if row["physical_predictive_relation_path_gate_pass"]
    ]
    summary = {
        "schema_version": "60.13.0",
        "phase_id": "Phase386-PhysicalRelations",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "frozen_candidate_count": len(candidates),
            "model_candidate_row_count": len(model_rows),
            "prediction_control_row_count": len(prediction_rows),
            "physical_groups_per_mechanism": 4,
        },
        "results": {
            "physical_relation_replication_count": sum(
                row["physical_relation_replication_all_three_models"]
                for row in candidate_rows
            ),
            "physical_predictive_relation_path_count": len(survivors),
            "survivor_counts_by_mechanism": dict(
                sorted(Counter(row["mechanism_id"] for row in survivors).items())
            ),
            "survivor_counts_by_vector_family": dict(
                sorted(Counter(row["vector_family"] for row in survivors).items())
            ),
            "holdout_reused": False,
            "candidate_replacement_used": False,
            "causal_relation_established": False,
            "language_encoding_closed": False,
        },
        "authorization": {
            "register_descriptive_predictive_relations": bool(survivors),
            "causal_intervention": False,
            "additional_holdout_reuse": False,
        },
        "claim_boundary": {
            "physical_prediction_is_causality": False,
            "physical_prediction_is_neuron_level_closure": False,
            "survivor_is_a_descriptive_predictive_relation_only": True,
        },
    }
    write_json(PHASE_ROOT / "phase386_physical_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
