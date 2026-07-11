#!/usr/bin/env python3
"""Aggregate Phase338 staged block results with heldout/private separation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase338_block_causal_screen"
PHASE = "Phase338"
SCHEMA_VERSION = "14.0.0"
ROUND_DEFAULT = "material_relation_block_screen"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONTROL_CONDITIONS = ("wrong_depth_zero", "wrong_position_zero")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def condition_mean(rows: list[dict[str, Any]], condition: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["condition"] == condition]
    return mean(values) if values else 0.0


def condition_rate(rows: list[dict[str, Any]], condition: str, predicate) -> float:
    values = [row for row in rows if row["condition"] == condition]
    return sum(predicate(row) for row in values) / len(values) if values else 0.0


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase338_registered_protocol.json")
    registered = read_jsonl(root / "phase338_registered_cases.jsonl")
    model_summaries = []
    all_discovery = []
    all_calibration = []
    all_heldout = []
    completes = []
    for model in MODELS:
        model_root = root / "models" / model
        all_discovery.extend(read_jsonl(model_root / "phase338_discovery_block_summary.jsonl"))
        all_calibration.extend(read_jsonl(model_root / "phase338_calibration_block_summary.jsonl"))
        heldout_rows = read_jsonl(model_root / "phase338_heldout_rows.jsonl")
        all_heldout.extend(heldout_rows)
        completes.extend([
            read_json(model_root / "phase338_discovery_complete.json"),
            read_json(model_root / "phase338_calibration_complete.json"),
            read_json(model_root / "phase338_heldout_complete.json"),
        ])
        frozen = read_jsonl(model_root / "phase338_frozen_heldout_block.jsonl")
        result = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "frozen_block_count": len(frozen),
            "selected_block_id": frozen[0]["block_id"] if frozen else None,
            "calibration_gate_pass": bool(frozen),
        }
        thresholds = protocol["thresholds"]
        for split in ("heldout", "private_heldout"):
            rows = [row for row in heldout_rows if row["split"] == split]
            correct_loss = condition_mean(rows, "correct_zero", "phrase_margin_loss_vs_baseline")
            positive_rate = condition_rate(
                rows, "correct_zero", lambda row: row["phrase_margin_loss_vs_baseline"] > 0
            )
            behavior_loss = condition_rate(
                rows, "correct_zero", lambda row: row["behavior_lost_vs_baseline"]
            )
            control_losses = {
                condition: condition_mean(rows, condition, "phrase_margin_loss_vs_baseline")
                for condition in CONTROL_CONDITIONS
            }
            control_behavior = {
                condition: condition_rate(
                    rows, condition, lambda row: row["behavior_lost_vs_baseline"]
                )
                for condition in CONTROL_CONDITIONS
            }
            max_control = max(control_losses.values(), default=0.0)
            permutation_loss = condition_mean(
                rows, "correct_permutation", "phrase_margin_loss_vs_baseline"
            )
            permutation_behavior = condition_rate(
                rows, "correct_permutation", lambda row: row["behavior_lost_vs_baseline"]
            )
            baseline_rate = condition_rate(
                rows, "baseline", lambda row: row["answer_head_semantic_correct"]
            )
            split_pass = bool(
                frozen and baseline_rate == 1.0
                and correct_loss >= thresholds["heldout_phrase_loss_min"]
                and positive_rate >= thresholds["heldout_positive_case_rate_min"]
                and correct_loss - max_control >= thresholds["heldout_control_superiority_min"]
                and behavior_loss >= thresholds["heldout_behavior_loss_rate_min"]
                and max(control_behavior.values(), default=0.0)
                <= thresholds["heldout_control_behavior_loss_rate_max"]
            )
            result.update({
                f"{split}_case_count": len(rows) // 6 if rows else 0,
                f"{split}_baseline_success_rate": round(baseline_rate, 7),
                f"{split}_correct_phrase_loss": round(correct_loss, 7),
                f"{split}_correct_positive_rate": round(positive_rate, 7),
                f"{split}_correct_behavior_loss_rate": round(behavior_loss, 7),
                f"{split}_max_control_phrase_loss": round(max_control, 7),
                f"{split}_max_control_behavior_loss_rate": round(
                    max(control_behavior.values(), default=0.0), 7
                ),
                f"{split}_permutation_phrase_loss": round(permutation_loss, 7),
                f"{split}_permutation_behavior_loss_rate": round(permutation_behavior, 7),
                f"{split}_gate_pass": split_pass,
            })
        result["full_model_gate_pass"] = bool(
            result["heldout_gate_pass"] and result["private_heldout_gate_pass"]
        )
        result["single_unit_causal"] = False
        result["evidence_level"] = (
            "L4_block_heldout_candidate" if result["full_model_gate_pass"]
            else "L3_block_intervention_not_confirmed"
        )
        model_summaries.append(result)

    cross_model_count = sum(row["full_model_gate_pass"] for row in model_summaries)
    cross_model_gate = cross_model_count >= protocol["thresholds"]["cross_model_pass_min"]
    calibration_lookup = {
        (row["model"], row["block_id"]): row for row in all_calibration
    }
    model_lookup = {row["model"]: row for row in model_summaries}
    block_nodes = []
    for row in all_discovery:
        calibration = calibration_lookup.get((row["model"], row["block_id"]))
        model_result = model_lookup[row["model"]]
        selected = model_result["selected_block_id"] == row["block_id"]
        local_gate = bool(selected and model_result["full_model_gate_pass"])
        block_nodes.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "node_id": f"phase338:{row['model']}:{row['block_id']}",
            "model": row["model"], "family_id": "content_knowledge",
            "mechanism_id": "material_relation_binding", "block_id": row["block_id"],
            "component": row["component"], "depth_bin": row["depth_bin"],
            "position_role": row["position_role"],
            "discovery_mean_phrase_loss": row["mean_zero_phrase_margin_loss"],
            "discovery_gate_pass": row["stage_gate_pass"],
            "calibration_audited": calibration is not None,
            "calibration_gate_pass": bool(calibration and calibration["stage_gate_pass"]),
            "selected_for_heldout": selected,
            "local_heldout_private_gate_pass": local_gate,
            "cross_model_gate_pass": False,
            "single_unit_causal": False,
            "mapping_status": (
                "model_specific_coarse_block_candidate" if local_gate
                else "coarse_block_observation_not_causal"
            ),
            "evidence_level": "L4_model_specific_block_candidate" if local_gate else "L2_block_observation",
        })
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": len(registered),
            "registered_block_count": len(read_jsonl(root / "phase338_registered_blocks.jsonl")),
            "discovery_block_summary_count": len(all_discovery),
            "calibration_block_summary_count": len(all_calibration),
            "heldout_condition_row_count": len(all_heldout),
            "all_stage_completions_valid": all(row["valid"] for row in completes),
        },
        "results": {
            "discovery_gate_block_count": sum(row["stage_gate_pass"] for row in all_discovery),
            "calibration_gate_block_count": sum(row["stage_gate_pass"] for row in all_calibration),
            "heldout_model_gate_count": sum(row["heldout_gate_pass"] for row in model_summaries),
            "private_heldout_model_gate_count": sum(
                row["private_heldout_gate_pass"] for row in model_summaries
            ),
            "full_model_block_gate_count": cross_model_count,
            "cross_model_block_gate_pass": cross_model_gate,
            "minimal_causal_set_entry_gate_open": cross_model_gate,
            "behavior_mechanism_closed_count": 0,
            "single_unit_causal_count": 0,
        },
        "progress_vector": {
            "nine_family_registered_coverage": "9/9",
            "mechanism_census_coverage": "72/72",
            "protocol_qualified_pilot": "1 mechanism",
            "coarse_block_deep_audit_attempted": "1/72",
            "cross_model_coarse_block_candidates": "1/1" if cross_model_gate else "0/1",
            "behavior_mechanism_closure": "0/72",
            "single_unit_causal_closure": "0/72",
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(root / "phase338_model_gate_summary.jsonl", model_summaries)
    write_jsonl(root / "phase338_physical_block_nodes.jsonl", block_nodes)
    claims = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase338_protocol_denominator",
            "claim": "The answer-aligned material-relation denominator remained baseline-capable.",
            "status": "supported", "evidence_level": "L2_protocol_qualified",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase338_glm4_early_source_mlp_block",
            "claim": "GLM4 has a model-specific early-source MLP coarse-block candidate.",
            "status": "model_specific_candidate" if any(
                row["model"] == "glm4" and row["full_model_gate_pass"] for row in model_summaries
            ) else "not_supported",
            "evidence_level": "L4_model_specific_block_candidate",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase338_cross_model_block",
            "claim": "A cross-model coarse causal block has been identified.",
            "status": "not_supported", "evidence_level": "L4_cross_model_gate_failed",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase338_minimal_set_entry",
            "claim": "The recursive minimal-causal-set entry gate is open.",
            "status": "not_supported", "evidence_level": "L4_entry_gate_closed",
        },
    ]
    write_jsonl(root / "phase338_claim_registry.jsonl", claims)
    write_json(root / "phase338_global_summary.json", summary)
    report = [
        "# Phase338 Staged Coarse-Block Causal Screen", "",
        f"- Registered cases: {len(registered)}",
        f"- Discovery block summaries: {len(all_discovery)}",
        f"- Discovery gate blocks: {summary['results']['discovery_gate_block_count']}",
        f"- Calibration gate blocks: {summary['results']['calibration_gate_block_count']}",
        f"- Full model block gates: {cross_model_count}/3",
        f"- Cross-model block gate: {cross_model_gate}",
        f"- Minimal causal set entry gate: {cross_model_gate}",
        "- Behavior mechanism closure: 0/72", "- Single-unit causal closure: 0/72", "",
        "## Model Results", "",
    ]
    for row in model_summaries:
        report.append(
            f"- {row['model']}: block={row['selected_block_id']}, calibration={row['calibration_gate_pass']}, "
            f"heldout={row['heldout_gate_pass']}, private={row['private_heldout_gate_pass']}, "
            f"full={row['full_model_gate_pass']}"
        )
    report.extend([
        "", "Matched attribute binding was not used as a null control because it is a real relation-binding mechanism.",
        "Mean replacement and recursive neuron splitting remain closed unless the cross-model block gate passes.",
    ])
    (root / "phase338_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
