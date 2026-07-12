#!/usr/bin/env python3
"""Aggregate Phase353 behavior qualification while keeping physical and causal seals closed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase353_family_contracts"
PHASE = "Phase353"
SCHEMA_VERSION = "29.0.0"
ROUND_DEFAULT = "family_specific_contract_compiler"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("physical_discovery", "physical_calibration", "physical_heldout", "causal_sealed")
CONDITIONS = ("A", "B", "C", "D")


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
    contracts = [row for row in read_jsonl(root / "phase353_contract_registry.jsonl") if row["strict_contract_gate_pass"]]
    phrase = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase353_phrase_rows.jsonl")]
    rollout = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase353_rollout_rows.jsonl")]
    completions = [read_json(root / "models" / model / "complete.json") for model in MODELS]
    cells = []
    for model in MODELS:
        for contract in contracts:
            family, mechanism = contract["family_id"], contract["mechanism_id"]
            result = {
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "family_id": family, "mechanism_id": mechanism,
                "manipulated_variable": contract["manipulated_variable"],
            }
            split_gates = []
            for split in SPLITS:
                condition_gates = []
                for condition in CONDITIONS:
                    p = [row for row in phrase if row["model"] == model and row["family_id"] == family and row["mechanism_id"] == mechanism and row["split"] == split and row["contrast_condition"].startswith(condition)]
                    r = [row for row in rollout if row["model"] == model and row["family_id"] == family and row["mechanism_id"] == mechanism and row["split"] == split and row["contrast_condition"].startswith(condition)]
                    valid = sum(row["score_valid"] for row in p) / len(p)
                    semantic = sum(row["semantic_correct"] for row in r) / len(r)
                    protocol = sum(row["protocol_correct"] for row in r) / len(r)
                    gate = bool(valid == 1 and semantic >= 0.8 and protocol >= 0.8)
                    condition_gates.append(gate)
                    prefix = f"{split}__{condition}"
                    result.update({
                        f"{prefix}_case_count": len(r), f"{prefix}_phrase_valid_rate": round(valid, 7),
                        f"{prefix}_semantic_accuracy": round(semantic, 7),
                        f"{prefix}_protocol_accuracy": round(protocol, 7), f"{prefix}_gate_pass": gate,
                    })
                split_pass = all(condition_gates)
                result[f"{split}_gate_pass"] = split_pass
                split_gates.append(split_pass)
            result["full_behavior_contract_gate_pass"] = all(split_gates)
            result["trace_discovery_calibration_gate_pass"] = result["physical_discovery_gate_pass"] and result["physical_calibration_gate_pass"]
            result["physical_heldout_trace_revealed"] = False
            result["causal_sealed_trace_revealed"] = False
            cells.append(result)
    cross = []
    for contract in contracts:
        family, mechanism = contract["family_id"], contract["mechanism_id"]
        values = [row for row in cells if row["family_id"] == family and row["mechanism_id"] == mechanism]
        cross.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "family_id": family, "mechanism_id": mechanism,
            "full_behavior_model_count": sum(row["full_behavior_contract_gate_pass"] for row in values),
            "trace_entry_model_count": sum(row["trace_discovery_calibration_gate_pass"] for row in values),
            "cross_model_full_behavior_gate_pass": all(row["full_behavior_contract_gate_pass"] for row in values),
            "cross_model_trace_entry": all(row["trace_discovery_calibration_gate_pass"] for row in values),
        })
    trace_entries = [f"{row['family_id']}/{row['mechanism_id']}" for row in cross if row["cross_model_trace_entry"]]
    summary = read_json(root / "phase353_contract_summary.json")
    summary["created_at"] = now()
    summary["denominator"].update({
        "executed_case_count": len(phrase), "phrase_row_count": len(phrase),
        "rollout_row_count": len(rollout), "model_contract_cell_count": len(cells),
        "invalid_phrase_row_count": sum(not row["score_valid"] for row in phrase),
        "all_model_completions_valid": all(row["valid"] for row in completions),
        "actual_model_batch_size": 1,
    })
    summary["results"].update({
        "model_execution_started": True,
        "full_behavior_model_contract_count": sum(row["full_behavior_contract_gate_pass"] for row in cells),
        "cross_model_full_behavior_contract_count": sum(row["cross_model_full_behavior_gate_pass"] for row in cross),
        "cross_model_trace_entry_contract_count": len(trace_entries),
        "cross_model_trace_entry_contracts": trace_entries,
        "physical_heldout_trace_revealed": False, "causal_sealed_trace_revealed": False,
    })
    summary["next_decision"] = "build_semantic_time_trace_for_qualified_contracts" if trace_entries else "repair_behavior_contracts"
    write_jsonl(root / "phase353_model_contract_summary.jsonl", cells)
    write_jsonl(root / "phase353_cross_model_contract_summary.jsonl", cross)
    write_json(root / "phase353_global_summary.json", summary)
    report = [
        "# Phase353 Family-Specific Contract Qualification", "",
        f"- Mechanical contracts: {summary['results']['strict_contract_count']}/18",
        f"- Executed phrase/rollout rows: {len(phrase)}/{len(rollout)}",
        f"- Cross-model full behavior contracts: {summary['results']['cross_model_full_behavior_contract_count']}/{len(contracts)}",
        f"- Cross-model discovery/calibration trace entries: {', '.join(trace_entries) if trace_entries else 'none'}", "",
        "Physical heldout, causal sealed traces, interventions, and neuron search remain closed.",
    ]
    (root / "phase353_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
