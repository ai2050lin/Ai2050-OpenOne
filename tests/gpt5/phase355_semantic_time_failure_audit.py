#!/usr/bin/env python3
"""Decompose Phase354 failures and export conservative semantic-time atlas data."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace/qualified_contract_semantic_time"
OUT = ROOT / "tests/gpt5/result/phase355_semantic_time_failure_audit"
ROUND_NAME = "strict_failure_decomposition"
DESTINATIONS = (
    ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
)
PHASE = "Phase355"
SCHEMA_VERSION = "31.0.0"
SPLITS = ("physical_discovery", "physical_calibration")


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


def split_checks(row: dict[str, Any], split: str) -> dict[str, bool]:
    signs = row[f"{split}_template_signs"]
    return {
        "signal_sufficient": abs(row[f"{split}_mean_operation_delta"]) >= 0.005,
        "lexical_sign_stable": row[f"{split}_lexical_sign_agreement_rate"] >= 0.75,
        "operation_exceeds_lexical": row[f"{split}_mean_operation_magnitude"] > row[f"{split}_mean_lexical_instability"],
        "template_sign_stable": signs[0] == signs[1] != 0,
    }


def run() -> dict[str, Any]:
    global_summary = read_json(SOURCE / "phase354_global_summary.json")
    nodes = read_jsonl(SOURCE / "phase354_dynamic_nodes.jsonl")
    edges = read_jsonl(SOURCE / "phase354_graph_edges.jsonl")
    convergence = read_jsonl(SOURCE / "phase354_cross_model_convergence.jsonl")
    annotated, near = [], []
    failure_counts: Counter[str] = Counter()
    by_contract: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for node in nodes:
        failures = []
        checks_by_split = {}
        for split in SPLITS:
            if not node[f"{split}_event_count"]:
                failures.append(f"{split}:missing_complete_pairs")
                continue
            checks = split_checks(node, split)
            checks_by_split[split] = checks
            for check, passed in checks.items():
                if not passed:
                    failures.append(f"{split}:{check}")
                    failure_counts[f"{split}:{check}"] += 1
        discovery_signal = node["physical_discovery_mean_operation_delta"]
        calibration_signal = node["physical_calibration_mean_operation_delta"]
        replicated_direction = bool(
            len(checks_by_split) == 2
            and abs(discovery_signal) >= 0.005
            and abs(calibration_signal) >= 0.005
            and discovery_signal * calibration_signal > 0
            and checks_by_split["physical_discovery"]["template_sign_stable"]
            and checks_by_split["physical_calibration"]["template_sign_stable"]
        )
        strict = node["dynamic_discovery_calibration_gate_pass"]
        tier = "strict_dynamic_candidate" if strict else "replicated_direction_near_candidate" if replicated_direction else "descriptive_only"
        annotated_row = {
            **node,
            "source_phase_id": "Phase354",
            "audit_phase_id": PHASE,
            "candidate_tier": tier,
            "failure_reasons": failures,
            "replicated_direction_near_candidate": replicated_direction,
            "heldout_eligible": strict,
            "causal_eligible": False,
        }
        annotated.append(annotated_row)
        by_contract[(node["family_id"], node["mechanism_id"])][tier] += 1
        if replicated_direction:
            near.append(annotated_row)
    near.sort(key=lambda row: (
        -min(abs(row["physical_discovery_mean_operation_delta"]), abs(row["physical_calibration_mean_operation_delta"])),
        row["node_id"],
    ))
    contract_rows = []
    for (family, mechanism), counts in sorted(by_contract.items()):
        contract_rows.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "family_id": family, "mechanism_id": mechanism,
            "strict_dynamic_candidate_count": counts["strict_dynamic_candidate"],
            "replicated_direction_near_candidate_count": counts["replicated_direction_near_candidate"],
            "descriptive_only_count": counts["descriptive_only"],
            "physical_heldout_entry_open": counts["strict_dynamic_candidate"] > 0,
            "causal_status": "not_tested",
        })
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "source_phase_id": "Phase354",
        "denominator": {
            "fixed_node_count": len(nodes), "graph_edge_count": len(edges),
            "qualified_contract_count": len(contract_rows),
            "raw_trace_row_count": global_summary["denominator"]["raw_trace_row_count"],
            "complete_pair_rate": global_summary["denominator"]["complete_pair_rate"],
        },
        "results": {
            "strict_dynamic_candidate_count": sum(row["dynamic_discovery_calibration_gate_pass"] for row in nodes),
            "replicated_direction_near_candidate_count": len(near),
            "descriptive_only_count": sum(row["candidate_tier"] == "descriptive_only" for row in annotated),
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "physical_heldout_entry_contract_count": 0,
            "internal_intervention_executed_count": 0,
            "single_unit_causal_count": 0,
        },
        "claim_boundary": {
            "near_candidate_is_strict_candidate": False,
            "near_candidate_is_heldout_eligible": False,
            "natural_trace_is_causal": False,
            "atlas_complete": False,
        },
        "next_stage": {
            "objective": "redesign_position_matched_multi_lexical_contracts_before_new_model_execution",
            "required_changes": [
                "freeze at least four lexical replicas per operation pair",
                "match answer surface and token count in every pair",
                "match source/query token positions between target and control prompts",
                "keep physical heldout and causal sealed splits closed until discovery/calibration replication",
            ],
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    out = OUT / ROUND_NAME
    write_jsonl(out / "phase355_annotated_nodes.jsonl", annotated)
    write_jsonl(out / "phase355_near_candidates.jsonl", near)
    write_jsonl(out / "phase355_contract_failure_summary.jsonl", contract_rows)
    write_json(out / "phase355_global_summary.json", summary)
    report = [
        "# Phase355 Semantic-Time Failure Audit", "",
        f"- Fixed nodes: {len(nodes)}",
        f"- Strict dynamic candidates: {summary['results']['strict_dynamic_candidate_count']}",
        f"- Replicated-direction near candidates: {len(near)}",
        f"- Descriptive-only nodes: {summary['results']['descriptive_only_count']}",
        "- Physical heldout and causal entry remain closed.",
    ]
    (out / "phase355_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    for destination in DESTINATIONS:
        write_jsonl(destination / "phase354_semantic_time_nodes.jsonl", annotated)
        write_jsonl(destination / "phase354_semantic_time_edges.jsonl", edges)
        write_jsonl(destination / "phase354_cross_model_convergence.jsonl", convergence)
        write_jsonl(destination / "phase355_near_candidates.jsonl", near)
        write_json(destination / "phase355_failure_summary.json", summary)
        manifest_path = destination / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {"schema_version": "pattern_family_atlas.v2"}
        manifest["updated_at"] = now()
        manifest["phase354_355"] = {
            "status": "strict_negative_with_near_candidates",
            "qualified_contracts": 3,
            "raw_trace_rows": global_summary["denominator"]["raw_trace_row_count"],
            "fixed_nodes": len(nodes), "graph_edges": len(edges),
            "strict_dynamic_candidates": 0,
            "replicated_direction_near_candidates": len(near),
            "physical_heldout_revealed": False,
            "single_unit_causal_count": 0,
            "files": [
                "phase354_semantic_time_nodes.jsonl", "phase354_semantic_time_edges.jsonl",
                "phase354_cross_model_convergence.jsonl", "phase355_near_candidates.jsonl",
                "phase355_failure_summary.json",
            ],
        }
        write_json(manifest_path, manifest)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
