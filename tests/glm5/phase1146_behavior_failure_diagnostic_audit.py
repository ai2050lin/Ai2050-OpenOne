from __future__ import annotations

import hashlib
import json
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


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def regenerate(spec: dict[str, Any], prereg: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    if split == "seen":
        return primary.make_dataset(
            4096,
            prereg["data"]["pairs"]["train"],
            int(spec["data_seeds"]["seen_evaluation"]),
            spec["lexicon"],
        )
    if split == "holdout":
        return primary.make_dataset(
            4096,
            prereg["data"]["pairs"][spec["split"]],
            int(spec["data_seeds"]["holdout_evaluation"]),
            spec["lexicon"],
        )
    return primary.make_quartets(
        prereg["data"]["pairs"][spec["split"]],
        int(spec["data_seeds"]["quartet"]),
        spec["lexicon"],
    )[0]


def factor_sets(sequence: np.ndarray, lexicon: list[int]) -> tuple[set[int], set[int]]:
    inverse = {physical: semantic for semantic, physical in enumerate(lexicon)}
    tokens = [inverse[int(token)] for token in sequence]
    records: list[tuple[int, dict[int, int]]] = []
    cursor = 1
    for _ in range(primary.RECORD_COUNT):
        entity = tokens[cursor] - primary.ENTITY_START
        fields: dict[int, int] = {}
        for offset in range(primary.FIELD_COUNT):
            field = tokens[cursor + 1 + 2 * offset] - primary.FIELD_START
            value = tokens[cursor + 2 + 2 * offset] - primary.VALUE_START
            fields[field] = value
        records.append((entity, fields))
        cursor += 1 + 2 * primary.FIELD_COUNT + 1
    query_entity = tokens[cursor + 1] - primary.ENTITY_START
    query_field = tokens[cursor + 2] - primary.FIELD_START
    field_values = {fields[query_field] for _, fields in records}
    entity_values = set(next(fields for entity, fields in records if entity == query_entity).values())
    return field_values, entity_values


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    diagnostic = read_json(OUT_ROOT / "analysis" / "failure_diagnostic.json")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, model: str | None = None, split: str | None = None) -> None:
        row: dict[str, Any] = {"name": name, "passed": bool(passed)}
        if model is not None:
            row["model"] = model
        if split is not None:
            row["split"] = split
        checks.append(row)

    body = dict(diagnostic)
    digest = body.pop("diagnostic_digest")
    add("diagnostic_digest", canonical_digest(body) == digest)
    add("scope", diagnostic["scope"] == "posthoc_behavior_only_no_hidden_state_claim")
    add("protocol_digest", diagnostic["protocol_digest"] == prereg["protocol_digest"])

    for model_name in prereg["selection"]["discovery_models"]:
        spec = prereg["models"][model_name]
        rows = read_jsonl(
            OUT_ROOT / "runs" / "discovery" / model_name / "predictions.jsonl"
        )
        for split in ["seen", "holdout", "quartet"]:
            selected = [row for row in rows if row["split"] == split]
            dataset = regenerate(spec, prereg, split)
            field_hits = 0
            entity_hits = 0
            both_hits = 0
            either_hits = 0
            correct = 0
            uniform_both = 0.0
            unique_cases = 0
            unique_correct = 0
            ambiguous_cases = 0
            ambiguous_correct = 0
            for index, row in enumerate(selected):
                field_values, entity_values = factor_sets(dataset["inputs"][index], spec["lexicon"])
                prediction = int(row["predicted_value"])
                is_correct = prediction == int(row["target_value"])
                field_hits += int(prediction in field_values)
                entity_hits += int(prediction in entity_values)
                both_hits += int(prediction in field_values and prediction in entity_values)
                either_hits += int(prediction in field_values or prediction in entity_values)
                correct += int(is_correct)
                intersection = field_values & entity_values
                uniform_both += len(intersection) / primary.VALUE_COUNT
                if len(intersection) == 1:
                    unique_cases += 1
                    unique_correct += int(is_correct)
                else:
                    ambiguous_cases += 1
                    ambiguous_correct += int(is_correct)
            count = len(selected)
            stored = diagnostic["models"][model_name]["splits"][split]
            recomputed = {
                "accuracy": correct / count,
                "field": field_hits / count,
                "entity": entity_hits / count,
                "both": both_hits / count,
                "either": either_hits / count,
                "uniform_both": uniform_both / count,
                "unique_fraction": unique_cases / count,
                "unique_accuracy": unique_correct / unique_cases if unique_cases else None,
                "ambiguous_accuracy": ambiguous_correct / ambiguous_cases if ambiguous_cases else None,
            }
            match = (
                close(recomputed["accuracy"], stored["rates"]["model_correct"])
                and close(recomputed["field"], stored["rates"]["prediction_in_query_field_set"])
                and close(recomputed["entity"], stored["rates"]["prediction_in_query_entity_set"])
                and close(recomputed["both"], stored["rates"]["prediction_in_both_factor_sets"])
                and close(recomputed["either"], stored["rates"]["prediction_in_either_factor_set"])
                and close(recomputed["uniform_both"], stored["uniform_set_coverage_baselines"]["both"])
                and close(
                    recomputed["unique_fraction"],
                    stored["factor_intersection_diagnostic"]["unique_case_fraction"],
                )
                and close(
                    recomputed["unique_accuracy"],
                    stored["factor_intersection_diagnostic"]["accuracy_when_unique"],
                )
                and close(
                    recomputed["ambiguous_accuracy"],
                    stored["factor_intersection_diagnostic"]["accuracy_when_ambiguous"],
                )
            )
            add("factor_metrics_recomputed", match, model_name, split)
            add("case_count", count == stored["case_count"], model_name, split)

    report = {
        "phase": PHASE,
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "all_checks_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "diagnostic_digest": digest,
    }
    report["audit_digest"] = canonical_digest(report)
    write_json(OUT_ROOT / "audit" / "failure_diagnostic_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
