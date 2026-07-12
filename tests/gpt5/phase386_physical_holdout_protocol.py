#!/usr/bin/env python3
"""Freeze Phase386 predictive relations before opening the physical holdout."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
SOURCE = OUT / "phase386_calibrated_relation_candidates.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def main() -> None:
    summary = read_json(OUT / "phase386_calibration_summary.json")
    if not summary["authorization"]["physical_holdout_collection"]:
        raise RuntimeError("Phase386 physical holdout is not authorized")
    candidates = [
        row for row in read_jsonl(SOURCE) if row["predictive_relation_path_gate_pass"]
    ]
    if len(candidates) != summary["results"][
        "crossmodel_predictive_relation_path_count"
    ]:
        raise RuntimeError("Phase386 predictive candidate denominator changed")
    frozen = OUT / "phase386_frozen_physical_candidates.jsonl"
    write_jsonl(frozen, candidates)
    payload = {
        "schema_version": "60.11.0",
        "phase_id": "Phase386-PhysicalHoldoutProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_candidate_count": len(candidates),
        "candidate_file": frozen.name,
        "candidate_file_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "candidate_counts": {
            "by_mechanism": dict(
                sorted(Counter(row["mechanism_id"] for row in candidates).items())
            ),
            "by_vector_family": dict(
                sorted(Counter(row["vector_family"] for row in candidates).items())
            ),
        },
        "physical_denominator": {
            "all_frozen_mechanisms_collected_even_without_candidate": True,
            "mechanism_count": 3,
            "groups_per_mechanism": 4,
            "models": 3,
            "conditions_per_group": 4,
            "case_count": 144,
        },
        "evaluation": {
            "same_relation_and_prediction_controls_as_calibration": True,
            "new_thresholds_after_opening_allowed": False,
            "candidate_replacement_allowed": False,
            "one_time_holdout": True,
        },
        "physical_holdout_data_read_before_freeze": False,
        "causal_intervention_authorized": False,
        "authorization": {
            "physical_holdout_collection": bool(candidates),
            "physical_holdout_candidate_evaluation": False,
            "causal_intervention": False,
        },
    }
    path = OUT / "phase386_physical_holdout_protocol.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
