#!/usr/bin/env python3
"""Reveal labels only after candidate freeze and test mechanism selectivity."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    candidates = read_jsonl(OUT / "phase361_frozen_predictive_candidates.jsonl")
    evaluations = read_jsonl(OUT / "phase361_next_layer_evaluation_rows.jsonl")
    labels = {
        row["blind_case_id"]: row
        for row in read_jsonl(OUT / "private" / "phase361_label_key.jsonl")
    }
    candidate_keys = {(row["depth_bin"], row["feature_id"]): row for row in candidates}
    per_case: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in evaluations:
        key = (row["depth_bin"], row["feature_id"])
        if key not in candidate_keys:
            continue
        label = labels[row["blind_case_id"]]
        mechanism = f"{label['family_id']}/{label['mechanism_id']}"
        per_case[(candidate_keys[key]["candidate_id"], mechanism, row["anonymous_model_id"], row["blind_case_id"])].append(
            float(row["prediction_gain"])
        )
    per_model: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (candidate, mechanism, model, _case), values in per_case.items():
        per_model[(candidate, mechanism, model)].append(sum(values) / len(values))

    audit_rows = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        mechanisms = sorted({key[1] for key in per_model if key[0] == candidate_id})
        support = {}
        for mechanism in mechanisms:
            model_gains = {
                model: sum(values) / len(values)
                for (value_candidate, value_mechanism, model), values in per_model.items()
                if value_candidate == candidate_id and value_mechanism == mechanism
            }
            support[mechanism] = {
                "model_case_mean_gains": {key: round(value, 9) for key, value in model_gains.items()},
                "all_three_models_positive": len(model_gains) == 3 and all(value > 0 for value in model_gains.values()),
            }
        positive = sorted(key for key, value in support.items() if value["all_three_models_positive"])
        audit_rows.append({
            **candidate,
            "mechanism_support": support,
            "positive_mechanism_count": len(positive),
            "positive_mechanisms": positive,
            "universally_positive_across_four_mechanisms": len(positive) == 4,
            "selective_association_only": 0 < len(positive) < 4,
            "operation_specific_mechanism_proved": False,
        })

    universal = sum(row["universally_positive_across_four_mechanisms"] for row in audit_rows)
    selective = sum(row["selective_association_only"] for row in audit_rows)
    summary = {
        "schema_version": "38.2.0", "phase_id": "Phase361", "created_at": now(),
        "denominator": {
            "frozen_candidate_count": len(candidates),
            "calibration_case_count": len({row["blind_case_id"] for row in evaluations}),
            "mechanism_count": len({f"{row['family_id']}/{row['mechanism_id']}" for row in labels.values()}),
        },
        "results": {
            "universally_positive_candidate_count": universal,
            "selective_association_candidate_count": selective,
            "unsupported_candidate_count": len(audit_rows) - universal - selective,
            "operation_specific_mechanism_count": 0,
        },
        "claim_boundary": {
            "candidate_freeze_preceded_label_reveal": True,
            "selective_association_is_operation_mechanism": False,
            "calibration_cases_per_model_mechanism": 2,
            "multi_step_generation_tested": False,
            "physical_heldout_revealed": False,
            "causal_intervention_executed": False,
        },
        "next_decision": "increase_independent_calibration_and_add_generation_time_before_any_mechanism_claim",
    }
    write_jsonl(OUT / "phase361_posthoc_predictive_candidate_rows.jsonl", audit_rows)
    write_json(OUT / "phase361_posthoc_predictive_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
