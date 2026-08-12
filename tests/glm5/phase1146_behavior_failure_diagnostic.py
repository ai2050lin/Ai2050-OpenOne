from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import phase1146_learned_composition_benchmark as primary


PHASE = 1146
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1146_learned_composition_benchmark"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def decode_records(sequence: np.ndarray, lexicon: list[int]) -> tuple[list[dict[str, Any]], int, int]:
    inverse = {physical: semantic for semantic, physical in enumerate(lexicon)}
    semantic = [inverse[int(token)] for token in sequence]
    records: list[dict[str, Any]] = []
    cursor = 1
    for _ in range(primary.RECORD_COUNT):
        entity = semantic[cursor] - primary.ENTITY_START
        fields: dict[int, int] = {}
        for offset in range(primary.FIELD_COUNT):
            field_token = semantic[cursor + 1 + 2 * offset]
            value_token = semantic[cursor + 2 + 2 * offset]
            fields[field_token - primary.FIELD_START] = value_token - primary.VALUE_START
        records.append({"entity": entity, "fields": fields})
        cursor += 1 + 2 * primary.FIELD_COUNT + 1
    if semantic[cursor] != primary.QUERY or semantic[cursor + 3] != primary.ANSWER:
        raise RuntimeError("Malformed query suffix")
    query_entity = semantic[cursor + 1] - primary.ENTITY_START
    query_field = semantic[cursor + 2] - primary.FIELD_START
    return records, query_entity, query_field


def dataset_for_split(spec: dict[str, Any], prereg: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    lexicon = spec["lexicon"]
    pairs = prereg["data"]["pairs"]
    if split == "seen":
        return primary.make_dataset(4096, pairs["train"], int(spec["data_seeds"]["seen_evaluation"]), lexicon)
    if split == "holdout":
        return primary.make_dataset(
            4096,
            pairs[spec["split"]],
            int(spec["data_seeds"]["holdout_evaluation"]),
            lexicon,
        )
    if split == "quartet":
        return primary.make_quartets(
            pairs[spec["split"]], int(spec["data_seeds"]["quartet"]), lexicon
        )[0]
    raise ValueError(split)


def analyze_split(
    rows: list[dict[str, Any]],
    dataset: dict[str, np.ndarray],
    lexicon: list[int],
) -> dict[str, Any]:
    if len(rows) != len(dataset["inputs"]):
        raise RuntimeError("Prediction/dataset length mismatch")
    counts: Counter[str] = Counter()
    output_counts: Counter[int] = Counter()
    heuristic_correct: Counter[str] = Counter()
    record_agreements = np.zeros(primary.RECORD_COUNT, dtype=np.int64)
    uniform_coverage = Counter()
    cases = len(rows)
    for index, row in enumerate(rows):
        records, query_entity, query_field = decode_records(dataset["inputs"][index], lexicon)
        prediction = int(row["predicted_value"])
        target = int(row["target_value"])
        output_counts[prediction] += 1
        field_values = [int(record["fields"][query_field]) for record in records]
        query_record = next(record for record in records if int(record["entity"]) == query_entity)
        entity_values = [int(query_record["fields"][field]) for field in range(primary.FIELD_COUNT)]
        counts["model_correct"] += int(prediction == target)
        counts["prediction_in_query_field_set"] += int(prediction in field_values)
        counts["prediction_in_query_entity_set"] += int(prediction in entity_values)
        counts["prediction_in_either_factor_set"] += int(
            prediction in field_values or prediction in entity_values
        )
        counts["prediction_in_both_factor_sets"] += int(
            prediction in field_values and prediction in entity_values
        )
        counts["prediction_in_field_not_entity"] += int(
            prediction in field_values and prediction not in entity_values
        )
        counts["prediction_in_entity_not_field"] += int(
            prediction in entity_values and prediction not in field_values
        )
        field_set = set(field_values)
        entity_set = set(entity_values)
        intersection = field_set & entity_set
        if len(intersection) == 1:
            counts["unique_factor_intersection_cases"] += 1
            counts["correct_when_factor_intersection_unique"] += int(prediction == target)
            counts["prediction_matches_unique_factor_intersection"] += int(prediction in intersection)
        else:
            counts["ambiguous_factor_intersection_cases"] += 1
            counts["correct_when_factor_intersection_ambiguous"] += int(prediction == target)
        uniform_coverage["field"] += len(field_set) / primary.VALUE_COUNT
        uniform_coverage["entity"] += len(entity_set) / primary.VALUE_COUNT
        uniform_coverage["either"] += len(field_set | entity_set) / primary.VALUE_COUNT
        uniform_coverage["both"] += len(field_set & entity_set) / primary.VALUE_COUNT
        for record_index, value in enumerate(field_values):
            record_agreements[record_index] += int(prediction == value)
            heuristic_correct[f"record_{record_index}_field_value"] += int(value == target)
        last_value = int(records[-1]["fields"][max(records[-1]["fields"].keys())])
        counts["prediction_is_last_record_query_field"] += int(prediction == field_values[-1])
        counts["prediction_is_first_record_query_field"] += int(prediction == field_values[0])
        counts["prediction_is_last_record_last_semantic_field"] += int(prediction == last_value)
    frequency = [output_counts[value] / cases for value in range(primary.VALUE_COUNT)]
    rates = {name: value / cases for name, value in sorted(counts.items())}
    uniform_rates = {name: value / cases for name, value in sorted(uniform_coverage.items())}
    unique_cases = counts["unique_factor_intersection_cases"]
    ambiguous_cases = counts["ambiguous_factor_intersection_cases"]
    return {
        "case_count": cases,
        "rates": rates,
        "uniform_set_coverage_baselines": uniform_rates,
        "set_coverage_excess_over_uniform": {
            "field": rates["prediction_in_query_field_set"] - uniform_rates["field"],
            "entity": rates["prediction_in_query_entity_set"] - uniform_rates["entity"],
            "either": rates["prediction_in_either_factor_set"] - uniform_rates["either"],
            "both": rates["prediction_in_both_factor_sets"] - uniform_rates["both"],
        },
        "factor_intersection_diagnostic": {
            "unique_case_fraction": unique_cases / cases,
            "accuracy_when_unique": counts["correct_when_factor_intersection_unique"] / unique_cases
            if unique_cases
            else None,
            "prediction_matches_unique_intersection": counts[
                "prediction_matches_unique_factor_intersection"
            ]
            / unique_cases
            if unique_cases
            else None,
            "accuracy_when_ambiguous": counts["correct_when_factor_intersection_ambiguous"]
            / ambiguous_cases
            if ambiguous_cases
            else None,
        },
        "record_position_prediction_agreement": {
            str(index): float(record_agreements[index] / cases)
            for index in range(primary.RECORD_COUNT)
        },
        "record_position_heuristic_accuracy": {
            name: value / cases for name, value in sorted(heuristic_correct.items())
        },
        "prediction_value_frequency": frequency,
        "maximum_output_frequency": max(frequency),
        "field_route_excess_over_accuracy": (
            counts["prediction_in_query_field_set"] - counts["model_correct"]
        )
        / cases,
        "entity_route_excess_over_accuracy": (
            counts["prediction_in_query_entity_set"] - counts["model_correct"]
        )
        / cases,
    }


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    final = read_json(OUT_ROOT / "analysis" / "final.json")
    if final["mechanism_phase_authorized"]:
        raise RuntimeError("Failure diagnostic is only valid after a denied mechanism gate")
    models: dict[str, Any] = {}
    for model_name in prereg["selection"]["discovery_models"]:
        spec = prereg["models"][model_name]
        run_dir = OUT_ROOT / "runs" / "discovery" / model_name
        summary = read_json(run_dir / "summary.json")
        rows = read_jsonl(run_dir / "predictions.jsonl")
        split_results: dict[str, Any] = {}
        for split in ["seen", "holdout", "quartet"]:
            selected = [row for row in rows if row["split"] == split]
            dataset = dataset_for_split(spec, prereg, split)
            split_results[split] = analyze_split(selected, dataset, spec["lexicon"])
        models[model_name] = {
            "summary_digest": summary["summary_digest"],
            "qualified": summary["qualified"],
            "splits": split_results,
        }
    report = {
        "phase": PHASE,
        "scope": "posthoc_behavior_only_no_hidden_state_claim",
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
        "models": models,
    }
    report["diagnostic_digest"] = canonical_digest(report)
    write_json(OUT_ROOT / "analysis" / "failure_diagnostic.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
