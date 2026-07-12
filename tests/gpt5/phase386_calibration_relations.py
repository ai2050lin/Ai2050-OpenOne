#!/usr/bin/env python3
"""Evaluate frozen Phase386 relations with fresh groups and matched controls."""

from __future__ import annotations

import functools
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
COLLECTION = PHASE_ROOT / "collection"
DISCOVERY_CASES = PHASE_ROOT / "protocol/private/phase386_discovery_cases.jsonl"
CALIBRATION_CASES = PHASE_ROOT / "protocol/private/phase386_calibration_cases.jsonl"
CANDIDATES = PHASE_ROOT / "discovery_relations/phase386_frozen_relation_candidates.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
COORDINATES = (
    "source_encoded",
    "query_integrated",
    "pre_decision",
    "target_encoded",
    "post_decision_next_token",
)
EPSILON = 1e-8


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


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    denominator = float(
        torch.linalg.vector_norm(left).item()
        * torch.linalg.vector_norm(right).item()
    )
    if denominator <= EPSILON:
        return 0.0
    return float(torch.dot(left, right).item() / denominator)


def median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def repeat_key_value(values: torch.Tensor, head_count: int) -> torch.Tensor:
    if values.shape[1] == head_count:
        return values
    if head_count % values.shape[1]:
        raise RuntimeError("Invalid Phase386 KV-head repetition")
    return values.repeat_interleave(head_count // values.shape[1], dim=1)


def extract_family(payload: dict[str, Any], family: str) -> torch.Tensor:
    if family in {"layer_input", "attention_output", "mlp_output", "layer_output"}:
        return payload["component_vectors"][family][0].float()
    if family == "mlp_channel_product":
        return payload["mlp"][
            "down_projection_input_product_at_coordinates"
        ][0].float()
    if family != "attention_head_state":
        raise KeyError(family)
    by_coordinate: dict[str, torch.Tensor] = {}
    for frame in payload["attention"]["frames"]:
        probabilities = frame["probabilities_receivers_all_sources"].float()
        values = repeat_key_value(
            frame["value_states_all_sources"].float(),
            int(frame["head_count"]),
        )
        head_states = torch.einsum(
            "bhqs,bhsd->bqhd", probabilities, values
        )[0]
        for index, coordinate in enumerate(frame["coordinate_names"]):
            by_coordinate[coordinate] = head_states[index].flatten()
    return torch.stack([by_coordinate[name] for name in COORDINATES])


def group_cases(paths: dict[str, Path]) -> dict[
    str, dict[tuple[str, str, str], dict[str, dict[str, Any]]]
]:
    result: dict[
        str, dict[tuple[str, str, str], dict[str, dict[str, Any]]]
    ] = {}
    for split, path in paths.items():
        grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for case in read_jsonl(path):
            key = (
                case["private_execution_model"],
                case["mechanism_id"],
                case["phase386_public_parallel_group_id"],
            )
            grouped[key][case["contrast_condition"]] = case
        if any(set(rows) != set(CONDITIONS) for rows in grouped.values()):
            raise RuntimeError(f"Incomplete Phase386 {split} condition group")
        result[split] = grouped
    return result


def wrong_depth(layer_index: int, layer_count: int) -> int:
    displacement = max(2, layer_count // 5)
    candidate = layer_index + displacement
    if candidate >= layer_count:
        candidate = layer_index - displacement
    candidate = max(0, min(layer_count - 1, candidate))
    if candidate == layer_index:
        candidate = 0 if layer_index else layer_count - 1
    return candidate


def main() -> None:
    audit = read_json(PHASE_ROOT / "phase386_calibration_collection_summary.json")
    if not audit["authorization"]["frozen_candidate_evaluation"]:
        raise RuntimeError("Phase386 frozen candidate evaluation is not authorized")
    contract = read_json(PHASE_ROOT / "phase386_relation_contract.json")
    freeze = read_json(PHASE_ROOT / "phase386_discovery_relation_freeze.json")
    if hashlib.sha256(CANDIDATES.read_bytes()).hexdigest() != freeze[
        "candidate_file_sha256"
    ]:
        raise RuntimeError("Phase386 frozen candidate checksum changed")
    candidates = read_jsonl(CANDIDATES)
    if len(candidates) != freeze["candidate_count"]:
        raise RuntimeError("Phase386 candidate denominator changed")
    grouped = group_cases(
        {"discovery": DISCOVERY_CASES, "calibration": CALIBRATION_CASES}
    )
    group_ids: dict[tuple[str, str, str], list[str]] = {}
    for split in ("discovery", "calibration"):
        for model in MODELS:
            for mechanism in {
                candidate["mechanism_id"] for candidate in candidates
            }:
                ids = sorted(
                    group_id
                    for candidate_model, candidate_mechanism, group_id in grouped[split]
                    if candidate_model == model and candidate_mechanism == mechanism
                )
                expected = 8 if split == "discovery" else 4
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
        vectors: dict[str, torch.Tensor] = {}
        for condition in CONDITIONS:
            case = cases[condition]
            path = (
                COLLECTION
                / split
                / "private/models"
                / model
                / case["blind_case_id"]
                / f"layer_{layer_index:03d}.pt"
            )
            vectors[condition] = extract_family(
                torch.load(path, map_location="cpu", weights_only=False),
                family,
            )
        return (
            vectors[CONDITIONS[0]] - vectors[CONDITIONS[1]],
            vectors[CONDITIONS[2]] - vectors[CONDITIONS[3]],
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
    model_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, 1):
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
            manifest = read_json(
                COLLECTION / "calibration/models" / model / "manifest.json"
            )
            layer_count = int(manifest["layer_count"])
            wrong_layer = wrong_depth(layer_index, layer_count)
            discovery_ids = group_ids[("discovery", model, mechanism)]
            calibration_ids = group_ids[("calibration", model, mechanism)]
            relation_by_variant: dict[str, list[float]] = {"lex_x": [], "lex_y": []}
            prediction_by_variant: dict[str, dict[str, list[float]]] = {
                variant: {
                    "actual": [],
                    "shuffled_pair": [],
                    "wrong_time": [],
                    "wrong_depth": [],
                }
                for variant in ("lex_x", "lex_y")
            }
            source_lexical: list[float] = []
            target_lexical: list[float] = []
            for calibration_group in calibration_ids:
                calibration_x, calibration_y = effects(
                    "calibration",
                    model,
                    mechanism,
                    calibration_group,
                    layer_index,
                    family,
                )
                source_lexical.append(
                    cosine(
                        calibration_x[source_index],
                        calibration_y[source_index],
                    )
                )
                target_lexical.append(
                    cosine(
                        calibration_x[target_index],
                        calibration_y[target_index],
                    )
                )
                for variant, calibration_effect in (
                    ("lex_x", calibration_x),
                    ("lex_y", calibration_y),
                ):
                    relation_by_variant[variant].append(
                        cosine(
                            calibration_effect[source_index],
                            calibration_effect[target_index],
                        )
                    )
                    discovery_effects = [
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
                        cosine(
                            calibration_effect[source_index],
                            value[source_index],
                        )
                        for value in discovery_effects
                    ]
                    nearest = max(
                        range(len(discovery_ids)),
                        key=lambda index: (similarities[index], -index),
                    )
                    shuffled = (nearest + 1) % len(discovery_ids)
                    predicted = discovery_effects[nearest][target_index]
                    shuffled_predicted = discovery_effects[shuffled][target_index]
                    wrong_time_predicted = discovery_effects[nearest][wrong_time_index]
                    wrong_depth_effect = effects(
                        "discovery",
                        model,
                        mechanism,
                        discovery_ids[nearest],
                        wrong_layer,
                        family,
                    )[0 if variant == "lex_x" else 1]
                    actual_target = calibration_effect[target_index]
                    values = {
                        "actual": cosine(actual_target, predicted),
                        "shuffled_pair": cosine(actual_target, shuffled_predicted),
                        "wrong_time": cosine(actual_target, wrong_time_predicted),
                        "wrong_depth": cosine(
                            actual_target, wrong_depth_effect[target_index]
                        ),
                    }
                    for key, value in values.items():
                        prediction_by_variant[variant][key].append(value)
                    prediction_rows.append(
                        {
                            "schema_version": "60.10.0",
                            "phase_id": "Phase386-CalibrationRelations",
                            "candidate_id": candidate["candidate_id"],
                            "model": model,
                            "mechanism_id": mechanism,
                            "vector_family": family,
                            "calibration_group_id": calibration_group,
                            "lexical_variant": variant,
                            "layer_index": layer_index,
                            "wrong_depth_layer_index": wrong_layer,
                            "wrong_time_coordinate": wrong_time_coordinate,
                            "nearest_source_cosine": similarities[nearest],
                            "actual_prediction_cosine": values["actual"],
                            "shuffled_pair_cosine": values["shuffled_pair"],
                            "wrong_time_cosine": values["wrong_time"],
                            "wrong_depth_cosine": values["wrong_depth"],
                        }
                    )

            variant_metrics: dict[str, dict[str, Any]] = {}
            for variant in ("lex_x", "lex_y"):
                relation_values = relation_by_variant[variant]
                predictions = prediction_by_variant[variant]
                actual = median(predictions["actual"])
                controls = {
                    name: median(values)
                    for name, values in predictions.items()
                    if name != "actual"
                }
                variant_metrics[variant] = {
                    "median_relation_cosine": median(relation_values),
                    "positive_relation_group_rate": sum(
                        value > 0 for value in relation_values
                    )
                    / len(relation_values),
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
                "schema_version": "60.10.0",
                "phase_id": "Phase386-CalibrationRelations",
                "candidate_id": candidate["candidate_id"],
                "model": model,
                "mechanism_id": mechanism,
                "vector_family": family,
                "source_coordinate": source,
                "target_coordinate": target,
                "layer_index": layer_index,
                "wrong_depth_layer_index": wrong_layer,
                "calibration_group_count": len(calibration_ids),
                "median_source_lexical_replication": median(source_lexical),
                "median_target_lexical_replication": median(target_lexical),
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
                "calibration_relation_replication_all_three_models": relation_all,
                "calibration_prediction_controls_all_three_models": prediction_all,
                "predictive_relation_path_gate_pass": relation_all and prediction_all,
                "causal_path_claim": False,
                "physical_holdout_used": False,
            }
        )
        if candidate_index % 10 == 0 or candidate_index == len(candidates):
            print(
                f"[Phase386 calibration] {candidate_index}/{len(candidates)} "
                f"relation={sum(row['calibration_relation_replication_all_three_models'] for row in candidate_rows)} "
                f"predictive={sum(row['predictive_relation_path_gate_pass'] for row in candidate_rows)}",
                flush=True,
            )

    write_jsonl(PHASE_ROOT / "private/phase386_calibration_prediction_rows.jsonl", prediction_rows)
    write_jsonl(PHASE_ROOT / "phase386_calibration_model_rows.jsonl", model_rows)
    write_jsonl(PHASE_ROOT / "phase386_calibrated_relation_candidates.jsonl", candidate_rows)
    replicated = [
        row
        for row in candidate_rows
        if row["calibration_relation_replication_all_three_models"]
    ]
    predictive = [
        row for row in candidate_rows if row["predictive_relation_path_gate_pass"]
    ]
    counts = {
        "replicated_by_mechanism": dict(
            sorted(Counter(row["mechanism_id"] for row in replicated).items())
        ),
        "replicated_by_vector_family": dict(
            sorted(Counter(row["vector_family"] for row in replicated).items())
        ),
        "predictive_by_mechanism": dict(
            sorted(Counter(row["mechanism_id"] for row in predictive).items())
        ),
        "predictive_by_vector_family": dict(
            sorted(Counter(row["vector_family"] for row in predictive).items())
        ),
    }
    summary = {
        "schema_version": "60.10.0",
        "phase_id": "Phase386-CalibrationRelations",
        "created_at": now(),
        "denominator": {
            "frozen_candidate_count": len(candidates),
            "model_candidate_row_count": len(model_rows),
            "prediction_control_row_count": len(prediction_rows),
            "calibration_groups_per_mechanism": 4,
        },
        "results": {
            "crossmodel_relation_replication_count": len(replicated),
            "crossmodel_predictive_relation_path_count": len(predictive),
            "counts": counts,
            "nearest_neighbor_fitted_operator_used": False,
            "all_three_controls_required": True,
            "physical_holdout_opened": False,
            "causal_relation_established": False,
            "language_encoding_closed": False,
        },
        "authorization": {
            "physical_holdout_collection": bool(predictive),
            "causal_intervention": False,
        },
        "claim_boundary": {
            "relation_replication_is_causality": False,
            "prediction_without_intervention_is_causality": False,
            "small_model_distribution_may_not_transfer_to_large_models": True,
        },
    }
    write_json(PHASE_ROOT / "phase386_calibration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
