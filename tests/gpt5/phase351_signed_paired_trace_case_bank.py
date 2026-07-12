#!/usr/bin/env python3
"""Freeze discovery/calibration-only cases for signed paired physical tracing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase351"
SCHEMA_VERSION = "27.0.0"
ROUND_DEFAULT = "signed_paired_physical_trace"
OUT = ROOT / "tests/gpt5/result/phase351_signed_paired_trace"
SOURCE = ROOT / "tests/gpt5/result/phase350_nine_family_minimal_contrast/nine_family_minimal_contrast_qualification"
FAMILIES = ("closure", "language_action", "state_drift")
SPLITS = ("physical_discovery", "physical_calibration")


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


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    source = read_jsonl(SOURCE / "phase350_registered_cases.jsonl")
    rows = []
    for row in source:
        if row["family_id"] not in FAMILIES or row["split"] not in SPLITS:
            continue
        rows.append({
            **row, "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
            "created_at": now(), "source_case_id": row["case_id"],
            "case_id": row["case_id"].replace("phase350_", "phase351_", 1),
            "natural_trace_only": True, "internal_intervention_allowed": False,
            "physical_heldout_trace_allowed": False, "causal_sealed_trace_allowed": False,
        })
    if len(rows) != 576 or len({row["case_id"] for row in rows}) != 576:
        raise RuntimeError(f"Invalid Phase351 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Measure signed target-versus-competitor paired trajectories without revealing physical heldout cases.",
        "registered_case_count": len(rows), "families": list(FAMILIES), "splits": list(SPLITS),
        "components": ["attention_output", "mlp_output", "residual_increment"],
        "depth_bins": ["early", "middle", "late"],
        "position_roles": ["source", "query", "answer_start"],
        "metrics": ["component_l2_norm", "signed_target_cosine", "signed_best_competitor_cosine", "signed_competition_margin"],
        "official_execution_mode": "b1_left_cache0",
        "claim_boundaries": [
            "Only physical discovery and calibration traces are visible.",
            "Signed unembedding alignment remains a descriptive readout, not a causal contribution.",
            "The explicit shortcut control is not a pure operation-off state.",
            "No generated-time trajectory, heldout trace, or intervention is included.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows),
        "model_case_count": {model: sum(row["model"] == model for row in rows) for model in ("qwen3", "glm4", "deepseek7b")},
        "family_case_count": {family: sum(row["family_id"] == family for row in rows) for family in FAMILIES},
        "physical_heldout_case_count": sum(row["split"] == "physical_heldout" for row in rows),
        "causal_sealed_case_count": sum(row["split"] == "causal_sealed" for row in rows),
        "valid": True,
    }
    root = OUT / round_name
    write_jsonl(root / "phase351_registered_cases.jsonl", rows)
    write_json(root / "phase351_registered_protocol.json", protocol)
    write_json(root / "phase351_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
