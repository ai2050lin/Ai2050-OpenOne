#!/usr/bin/env python3
"""Aggregate the three frozen Phase601 model runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase601_foodon_public_ontology_protocol as protocol  # noqa: E402
from phase601_foodon_public_ontology import output_paths  # noqa: E402


OUT_PATH = protocol.OUT_DIR / "phase601_cross_model_analysis.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def analyze() -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    rows_by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in protocol.MODELS:
        paths = output_paths(model)
        if not paths["summary"].exists() or not paths["rows"].exists():
            raise RuntimeError(f"Missing Phase601 complete run for {model}")
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise RuntimeError(f"Incomplete Phase601 run for {model}")
        if summary["rows_sha256"] != sha256_file(paths["rows"]):
            raise RuntimeError(f"Phase601 row drift for {model}")
        summaries[model] = summary
        for row in read_rows(paths["rows"]):
            rows_by_case[row["case_id"]][model] = row
    if any(set(values) != set(protocol.MODELS) for values in rows_by_case.values()):
        raise RuntimeError("Phase601 cross-model denominator mismatch")
    case_count = len(rows_by_case)
    all_models_correct_count = sum(
        all(values[model]["forced_choice_correct"] for model in protocol.MODELS)
        for values in rows_by_case.values()
    )
    unanimous_prediction_count = sum(
        len({values[model]["forced_choice_prediction"] for model in protocol.MODELS}) == 1
        for values in rows_by_case.values()
    )
    all_models_wrong = [
        values for values in rows_by_case.values()
        if not any(values[model]["forced_choice_correct"] for model in protocol.MODELS)
    ]
    family_cross_model = {}
    for family in protocol.FAMILIES:
        values = [
            model_rows for model_rows in rows_by_case.values()
            if next(iter(model_rows.values()))["family"] == family
        ]
        family_cross_model[family] = {
            "case_count": len(values),
            "all_models_correct_rate": sum(
                all(row[model]["forced_choice_correct"] for model in protocol.MODELS)
                for row in values
            ) / max(1, len(values)),
            "unanimous_prediction_rate": sum(
                len({row[model]["forced_choice_prediction"] for model in protocol.MODELS}) == 1
                for row in values
            ) / max(1, len(values)),
        }
    model_qualification = {
        model: bool(summary["behavior_qualified"]) for model, summary in summaries.items()
    }
    qualified_models = [model for model, passed in model_qualification.items() if passed]
    pairwise_agreement = {}
    for left_index, left in enumerate(protocol.MODELS):
        for right in protocol.MODELS[left_index + 1 :]:
            pairwise_agreement[f"{left}/{right}"] = sum(
                values[left]["forced_choice_prediction"]
                == values[right]["forced_choice_prediction"]
                for values in rows_by_case.values()
            ) / max(1, case_count)
    true_false_pair_metrics = {}
    true_false_pairs = sorted({
        (next(iter(values.values()))["family"], next(iter(values.values()))["false_family"])
        for values in rows_by_case.values()
    })
    for true_family, false_family in true_false_pairs:
        pair_rows = [
            values for values in rows_by_case.values()
            if next(iter(values.values()))["family"] == true_family
            and next(iter(values.values()))["false_family"] == false_family
        ]
        true_false_pair_metrics[f"{true_family}->{false_family}"] = {
            "case_count": len(pair_rows),
            "accuracy_by_model": {
                model: sum(values[model]["forced_choice_correct"] for values in pair_rows)
                / max(1, len(pair_rows))
                for model in protocol.MODELS
            },
            "all_models_correct_rate": sum(
                all(values[model]["forced_choice_correct"] for model in protocol.MODELS)
                for values in pair_rows
            ) / max(1, len(pair_rows)),
            "mean_model_accuracy": sum(
                values[model]["forced_choice_correct"]
                for values in pair_rows for model in protocol.MODELS
            ) / max(1, len(pair_rows) * len(protocol.MODELS)),
        }
    concept_rows: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for values in rows_by_case.values():
        concept_rows[next(iter(values.values()))["concept_id"]].append(values)
    hardest_concepts = []
    for concept_id, surfaces in concept_rows.items():
        example = next(iter(surfaces[0].values()))
        correct_count = sum(
            values[model]["forced_choice_correct"]
            for values in surfaces for model in protocol.MODELS
        )
        hardest_concepts.append({
            "concept_id": concept_id,
            "concept_label": example["concept_label"],
            "family": example["family"],
            "false_family": example["false_family"],
            "split": example["split"],
            "distance_to_family_root": example["distance_to_family_root"],
            "lexical_cue": example["lexical_cue"],
            "correct_model_surface_count": correct_count,
            "model_surface_denominator": len(surfaces) * len(protocol.MODELS),
        })
    hardest_concepts.sort(
        key=lambda row: (
            row["correct_model_surface_count"] / row["model_surface_denominator"],
            row["concept_id"],
        )
    )
    payload = {
        "schema_version": "phase601_cross_model_analysis.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete",
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "source_sha256": sha256_file(protocol.SOURCE_PATH),
        "case_count": case_count,
        "model_qualification": model_qualification,
        "qualified_models": qualified_models,
        "model_metrics": {
            model: {
                "forced_choice_accuracy": summary["overall"]["forced_choice_accuracy"],
                "direct_candidate_output_rate": summary["overall"]["direct_candidate_output_rate"],
                "direct_exact_accuracy": summary["overall"]["direct_exact_accuracy"],
                "concept_unanimous_rate": summary["overall"]["concept_unanimous_rate"],
                "heldout_forced_choice_accuracy": summary["split_metrics"]["heldout"]["forced_choice_accuracy"],
                "nonlexical_forced_choice_accuracy": summary["lexical_metrics"]["nonlexical"]["forced_choice_accuracy"],
                "gate_checks": summary["gate_checks"],
            }
            for model, summary in summaries.items()
        },
        "all_models_correct_rate": all_models_correct_count / max(1, case_count),
        "unanimous_prediction_rate": unanimous_prediction_count / max(1, case_count),
        "all_models_wrong_case_count": len(all_models_wrong),
        "all_models_wrong_rate": len(all_models_wrong) / max(1, case_count),
        "all_models_wrong_concept_count": len({
            next(iter(values.values()))["concept_id"] for values in all_models_wrong
        }),
        "all_models_wrong_counts": {
            "by_family": dict(Counter(
                next(iter(values.values()))["family"] for values in all_models_wrong
            )),
            "by_target_letter": dict(Counter(
                next(iter(values.values()))["target_letter"] for values in all_models_wrong
            )),
            "by_surface": dict(Counter(
                next(iter(values.values()))["surface_id"] for values in all_models_wrong
            )),
            "by_true_false_family": dict(Counter(
                f"{next(iter(values.values()))['family']}->"
                f"{next(iter(values.values()))['false_family']}"
                for values in all_models_wrong
            )),
        },
        "pairwise_prediction_agreement": pairwise_agreement,
        "family_cross_model_metrics": family_cross_model,
        "true_false_pair_metrics": true_false_pair_metrics,
        "hardest_concepts": hardest_concepts[:40],
        "cross_model_internal_observer_followup_authorized": len(qualified_models) == len(protocol.MODELS),
        "model_specific_internal_observer_followup_authorized": qualified_models,
        "causal_intervention_authorized": False,
        "mechanism_claim_authorized": False,
        "ontology_semantic_calibration_required": True,
        "posthoc_case_removal_authorized": False,
        "evidence_boundary": (
            "Public-ontology forced-choice behavior only; no hidden-state, neuron, parameter, "
            "or causal mechanism evidence was collected."
        ),
        "failure_counts_by_model": {
            model: case_count - sum(
                values[model]["forced_choice_correct"] for values in rows_by_case.values()
            )
            for model in protocol.MODELS
        },
        "forced_choice_prediction_counts_by_model": {
            model: dict(Counter(values[model]["forced_choice_prediction"] for values in rows_by_case.values()))
            for model in protocol.MODELS
        },
    }
    write_json(OUT_PATH, payload)
    return payload


def main() -> None:
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
