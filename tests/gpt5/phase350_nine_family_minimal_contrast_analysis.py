#!/usr/bin/env python3
"""Aggregate Phase350 paired-contrast behavior qualification."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase350_nine_family_minimal_contrast"
PHASE = "Phase350"
SCHEMA_VERSION = "26.0.0"
ROUND_DEFAULT = "nine_family_minimal_contrast_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("physical_discovery", "physical_calibration", "physical_heldout", "causal_sealed")
CONDITIONS = ("A_operation_lex_x", "B_control_lex_x", "C_operation_lex_y", "D_control_lex_y")


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


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase350_registered_protocol.json")
    cases = read_jsonl(root / "phase350_registered_cases.jsonl")
    completions = [read_json(root / "models" / model / "complete.json") for model in MODELS]
    phrase = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase350_phrase_rows.jsonl")]
    rollout = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase350_rollout_rows.jsonl")]
    thresholds = protocol["thresholds"]
    cell_rows = []
    families = sorted({row["family_id"] for row in cases})
    for model in MODELS:
        for family in families:
            result = {
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "family_id": family,
            }
            gates = []
            for split in SPLITS:
                split_gates = []
                for condition in CONDITIONS:
                    p = [row for row in phrase if row["model"] == model and row["family_id"] == family and row["split"] == split and row["contrast_condition"] == condition]
                    r = [row for row in rollout if row["model"] == model and row["family_id"] == family and row["split"] == split and row["contrast_condition"] == condition]
                    valid_rate = sum(row["score_valid"] for row in p) / len(p)
                    phrase_win_rate = sum(row["target_wins"] for row in p) / len(p)
                    semantic_accuracy = sum(row["answer_head_semantic_correct"] for row in r) / len(r)
                    gate = bool(
                        valid_rate >= thresholds["split_phrase_valid_rate_min"]
                        and semantic_accuracy >= thresholds["split_semantic_accuracy_min"]
                    )
                    split_gates.append(gate)
                    prefix = f"{split}__{condition[0]}"
                    result.update({
                        f"{prefix}_case_count": len(r), f"{prefix}_phrase_valid_rate": round(valid_rate, 7),
                        f"{prefix}_phrase_target_win_rate": round(phrase_win_rate, 7),
                        f"{prefix}_semantic_accuracy": round(semantic_accuracy, 7),
                        f"{prefix}_gate_pass": gate,
                    })
                split_pass = all(split_gates)
                result[f"{split}_gate_pass"] = split_pass
                gates.append(split_pass)
            result["full_contract_gate_pass"] = all(gates)
            result["natural_trace_allowed"] = result["physical_discovery_gate_pass"] and result["physical_calibration_gate_pass"]
            result["physical_heldout_trace_revealed"] = False
            result["causal_sealed_trace_revealed"] = False
            result["internal_intervention"] = False
            cell_rows.append(result)
    cross_model_family = []
    for family in families:
        values = [row for row in cell_rows if row["family_id"] == family]
        cross_model_family.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "family_id": family,
            "qualified_model_count": sum(row["full_contract_gate_pass"] for row in values),
            "natural_trace_model_count": sum(row["natural_trace_allowed"] for row in values),
            "cross_model_full_contract_gate_pass": all(row["full_contract_gate_pass"] for row in values),
            "cross_model_natural_trace_entry": all(row["natural_trace_allowed"] for row in values),
        })
    eligible = [row["family_id"] for row in cross_model_family if row["cross_model_natural_trace_entry"]]
    nodes = [{
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "node_id": f"phase350:{row['model']}:{row['family_id']}",
        "model": row["model"], "family_id": row["family_id"],
        "mapping_status": "contrast_baseline_qualified" if row["full_contract_gate_pass"] else "contrast_baseline_partial",
        "natural_trace_allowed": row["natural_trace_allowed"],
        "causal_status": "not_tested", "single_unit_causal": False,
    } for row in cell_rows]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": len(cases), "phrase_row_count": len(phrase),
            "rollout_row_count": len(rollout), "family_count": len(families),
            "model_family_cell_count": len(cell_rows),
            "invalid_phrase_row_count": sum(not row["score_valid"] for row in phrase),
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "actual_model_batch_size": 1,
        },
        "results": {
            "full_contract_model_family_count": sum(row["full_contract_gate_pass"] for row in cell_rows),
            "cross_model_full_contract_family_count": sum(row["cross_model_full_contract_gate_pass"] for row in cross_model_family),
            "cross_model_natural_trace_entry_family_count": len(eligible),
            "cross_model_natural_trace_entry_families": eligible,
            "physical_heldout_trace_revealed": False,
            "causal_sealed_trace_revealed": False,
            "internal_intervention_executed_count": 0,
            "behavior_mechanism_closed_count": 0, "single_unit_causal_count": 0,
        },
        "next_decision": "run_signed_trace_on_discovery_calibration_only" if eligible else "repair_family_contracts",
        "claim_boundary": {
            "explicit_shortcut_control_is_pure_operation_off": False,
            "baseline_qualification_is_physical_mechanism": False,
            "cross_model_family_entry_is_causal": False,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(root / "phase350_model_family_contract_summary.jsonl", cell_rows)
    write_jsonl(root / "phase350_cross_model_family_summary.jsonl", cross_model_family)
    write_jsonl(root / "phase350_atlas_nodes.jsonl", nodes)
    write_json(root / "phase350_global_summary.json", summary)
    report = [
        "# Phase350 Nine-Family Minimal Contrast Qualification", "",
        f"- Registered/phrase/rollout rows: {len(cases)}/{len(phrase)}/{len(rollout)}",
        f"- Full model-family contracts: {summary['results']['full_contract_model_family_count']}/27",
        f"- Cross-model full family contracts: {summary['results']['cross_model_full_contract_family_count']}/9",
        f"- Cross-model natural-trace entry families: {', '.join(eligible) if eligible else 'none'}", "",
        "No internal trace selection, intervention, or closure claim was executed.",
    ]
    (root / "phase350_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
