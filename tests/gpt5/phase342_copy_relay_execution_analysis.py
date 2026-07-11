#!/usr/bin/env python3
"""Aggregate Phase342 execution-invariance gates without mechanism claims."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase342_copy_relay_execution"
PHASE = "Phase342"
SCHEMA_VERSION = "18.0.0"
ROUND_DEFAULT = "copy_relay_execution_invariance"
MODELS = ("qwen3", "glm4", "deepseek7b")


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
    protocol = read_json(root / "phase342_registered_protocol.json")
    thresholds = protocol["thresholds"]
    summaries = []
    completions = []
    all_rows = []
    for model in MODELS:
        model_root = root / "models" / model
        rows = read_jsonl(model_root / "phase342_execution_rows.jsonl")
        completions.append(read_json(model_root / "complete.json"))
        all_rows.extend(rows)
        for mode in protocol["execution_modes"]:
            values = [row for row in rows if row["mode_id"] == mode["mode_id"]]
            text_rate = sum(row["text_equal_to_reference"] for row in values) / len(values)
            correctness_rate = sum(row["correctness_equal_to_reference"] for row in values) / len(values)
            top_rate = sum(row["top_token_equal_to_reference"] for row in values) / len(values)
            finite_rate = sum(row["forward_finite"] for row in values) / len(values)
            cosines = [
                row["source_hidden_cosine_to_reference"] for row in values
                if row["source_hidden_cosine_to_reference"] is not None
            ]
            deltas = [
                row["target_first_logit_abs_delta"] for row in values
                if row["target_first_logit_abs_delta"] is not None
            ]
            min_cosine = min(cosines) if cosines else 0.0
            mean_cosine = mean(cosines) if cosines else 0.0
            max_logit_delta = max(deltas) if deltas else 0.0
            gate = bool(
                finite_rate == 1.0
                and text_rate >= thresholds["text_invariance_rate_min"]
                and correctness_rate >= thresholds["correctness_invariance_rate_min"]
                and top_rate >= thresholds["top_token_invariance_rate_min"]
                and min_cosine >= thresholds["source_hidden_cosine_min"]
                and max_logit_delta <= thresholds["target_first_logit_abs_delta_max"]
            )
            summaries.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "created_at": now(), "model": model, **mode,
                "case_count": len(values), "finite_rate": round(finite_rate, 7),
                "text_invariance_rate": round(text_rate, 7),
                "correctness_invariance_rate": round(correctness_rate, 7),
                "top_token_invariance_rate": round(top_rate, 7),
                "source_hidden_cosine_min": round(min_cosine, 7),
                "source_hidden_cosine_mean": round(mean_cosine, 7),
                "target_first_logit_abs_delta_max": round(max_logit_delta, 7),
                "execution_gate_pass": gate,
                "eligible_for_causal_behavior": gate,
                "internal_intervention": False,
            })
    accepted = {
        model: [
            row["mode_id"] for row in summaries
            if row["model"] == model and row["execution_gate_pass"]
        ]
        for model in MODELS
    }
    official = {
        model: "b1_left_cache0" if "b1_left_cache0" in modes else None
        for model, modes in accepted.items()
    }
    nodes = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
            "created_at": now(), "node_id": f"phase342_exec:{row['model']}:{row['mode_id']}",
            "model": row["model"], "family_id": "measurement_execution",
            "mechanism_id": "batch_cache_padding_invariance",
            "mode_id": row["mode_id"], "execution_gate_pass": row["execution_gate_pass"],
            "mapping_status": "qualified_execution_path" if row["execution_gate_pass"] else "rejected_execution_path",
            "internal_intervention": False, "single_unit_causal": False,
        }
        for row in summaries
    ]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": protocol["registered_case_count"],
            "execution_mode_count": len(protocol["execution_modes"]),
            "result_row_count": len(all_rows),
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "nonfinite_row_count": sum(row["nonfinite_row_count"] for row in completions),
        },
        "results": {
            "accepted_modes_by_model": accepted,
            "official_causal_execution_mode_by_model": official,
            "all_models_have_official_path": all(official.values()),
            "copy_candidate_test_entry_gate_open": all(official.values()),
            "internal_intervention_executed_count": 0,
            "behavior_mechanism_closed_count": 0,
            "single_unit_causal_count": 0,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    claims = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase342_official_execution_paths",
            "claim": "Every model has at least one execution-qualified path for later causal behavior tests.",
            "status": "supported" if summary["results"]["all_models_have_official_path"] else "not_supported",
            "evidence_level": "L2_measurement_execution_qualification",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase342_language_mechanism", "claim": "Execution invariance identifies a language mechanism.",
            "status": "not_supported", "evidence_level": "measurement_only",
        },
    ]
    write_jsonl(root / "phase342_execution_mode_summary.jsonl", summaries)
    write_jsonl(root / "phase342_execution_nodes.jsonl", nodes)
    write_jsonl(root / "phase342_claim_registry.jsonl", claims)
    write_json(root / "phase342_global_summary.json", summary)
    report = [
        "# Phase342 Copy-Relay Execution Invariance", "",
        f"- Registered cases: {protocol['registered_case_count']}",
        f"- Result rows: {len(all_rows)}", "",
    ]
    for model in MODELS:
        report.append(f"- {model}: accepted={accepted[model]}, official={official[model]}")
    report.extend(["", "No activation intervention or mechanism closure was performed."])
    (root / "phase342_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
