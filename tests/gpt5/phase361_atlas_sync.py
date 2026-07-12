#!/usr/bin/env python3
"""Publish Phase361 summaries while keeping prompts, labels, and tensors sealed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tests/gpt5/result/phase361_contract_repair/seven_contract_repair"
TRACE = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace"
TARGETS = (
    ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    contract = read_json(CONTRACT / "phase361_behavior_summary.json")
    r0r1 = read_json(TRACE / "phase361_r0_r1_summary.json")
    prediction = read_json(TRACE / "phase361_blind_prediction_summary.json")
    posthoc = read_json(TRACE / "phase361_posthoc_predictive_summary.json")
    matrix = read_jsonl(CONTRACT / "phase361_frozen_mechanism_matrix.jsonl")
    updated_at = now()
    public_contract = {
        key: contract[key]
        for key in ("schema_version", "phase_id", "created_at", "denominator", "quality", "results", "admitted_mechanisms", "evidence_boundary", "next_decision")
    }
    for target in TARGETS:
        write_json(target / "phase361_contract_repair_summary.json", public_contract)
        write_jsonl(target / "phase361_frozen_mechanism_matrix.jsonl", matrix)
        write_json(target / "phase361_r0_r1_summary.json", r0r1)
        write_json(target / "phase361_blind_prediction_summary.json", prediction)
        write_json(target / "phase361_posthoc_predictive_summary.json", posthoc)
        manifest_path = target / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["updated_at"] = updated_at
        manifest["phase361"] = {
            "status": "r0_r1_complete_predictive_candidates_not_mechanisms",
            "repaired_contract_count": contract["denominator"]["repaired_contract_count"],
            "repaired_behavior_case_count": contract["denominator"]["registered_case_count"],
            "blind_discovery_admitted_count": contract["results"]["total_blind_discovery_admitted_count"],
            "r0_r1_case_count": r0r1["denominator"]["case_count"],
            "r0_r1_ledger_row_count": r0r1["denominator"]["ledger_row_count"],
            "r0_r1_sealed_byte_count": r0r1["denominator"]["sealed_byte_count"],
            "frozen_predictive_candidate_count": prediction["denominator"]["shared_positive_candidate_count"],
            "universal_backbone_candidate_count": posthoc["results"]["universally_positive_candidate_count"],
            "selective_association_candidate_count": posthoc["results"]["selective_association_candidate_count"],
            "operation_specific_mechanism_count": 0,
            "physical_heldout_revealed": False,
            "single_unit_causal_count": 0,
            "raw_tensors_frontend_exported": False,
            "files": [
                "phase361_contract_repair_summary.json",
                "phase361_frozen_mechanism_matrix.jsonl",
                "phase361_r0_r1_summary.json",
                "phase361_blind_prediction_summary.json",
                "phase361_posthoc_predictive_summary.json",
            ],
        }
        write_json(manifest_path, manifest)
    print(json.dumps({
        "updated_at": updated_at,
        "admitted": contract["results"]["total_blind_discovery_admitted_count"],
        "predictive_candidates": prediction["denominator"]["shared_positive_candidate_count"],
        "operation_specific_mechanisms": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
