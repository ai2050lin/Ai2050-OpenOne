#!/usr/bin/env python3
"""Publish Phase333 frozen dynamic-event anchors without changing prior geometry."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas"
OUTPUT = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
PUBLIC = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_json_artifacts(root: Path) -> int:
    changed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "checksums.json":
            continue
        if path.suffix == ".json":
            payload = read_json(path)
            safe = json_safe(payload)
            if safe != payload:
                write_json(path, safe)
                changed += 1
        elif path.suffix == ".jsonl":
            rows = read_jsonl(path)
            safe = json_safe(rows)
            if safe != rows:
                write_jsonl(path, safe)
                changed += 1
    return changed


def publish(output: Path = OUTPUT, public: Path = PUBLIC) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = read_json(SOURCE / "phase333_global_summary.json")
    plans = read_jsonl(SOURCE / "phase333_block_plans.jsonl")
    sequence = read_jsonl(SOURCE / "phase333_sequence_summary.jsonl")
    alignment = read_jsonl(SOURCE / "phase333_interface_alignment.jsonl")
    local = read_jsonl(SOURCE / "phase333_block_local_summary.jsonl")
    compensation = read_jsonl(SOURCE / "phase333_compensation_candidates.jsonl")
    cross = read_jsonl(SOURCE / "phase333_cross_model_summary.jsonl")
    claims = read_jsonl(SOURCE / "phase333_claim_registry.jsonl")
    sequence_map = {(row["model"], row["mechanism_id"], row["interface"]): row for row in sequence}
    local_map = {(row["model"], row["mechanism_id"], row["exchange_direction"]): row for row in local}
    index = read_json(output / "neuron_index.json")
    refs = index["partitions"]
    ref_map = {(row["model"], row["family_id"]): row for row in refs}
    nodes = []
    for plan in plans:
        model = plan["model"]
        mechanism = plan["mechanism_id"]
        interface = plan["interface"]
        partition = read_json(output / ref_map[(model, "reasoning_constraint")]["path"])
        seq = sequence_map[(model, mechanism, interface)]
        direction = (
            "answer_aligned_to_raw" if interface == "raw_completion"
            else "raw_to_answer_aligned" if interface == "answer_aligned_chat"
            else None
        )
        effect = local_map.get((model, mechanism, direction)) if direction else None
        cohort = "positive" if mechanism == "missing_condition_control" else "matched_negative_control"
        layer = int(plan["median_peak_layer"])
        nodes.append({
            "schema_version": "neuron_atlas_node.v1",
            "node_id": f"p333:{model}:reasoning_constraint:{mechanism}:{interface}:L{layer}:dynamic_write",
            "node_type": "dynamic_path_event",
            "family_id": "reasoning_constraint",
            "family_name": partition["family"]["family_name"],
            "mechanism_id": mechanism,
            "relation": mechanism,
            "cohort": cohort,
            "model": model,
            "model_revision": partition["model_snapshot"]["model_revision"],
            "layer": layer,
            "component": "residual",
            "unit_kind": "dynamic_residual_block",
            "unit_index": layer,
            "natural_activation": 0.0,
            "candidate_score": 1.0,
            "display_priority": 2.4 if cohort == "positive" else 1.7,
            "case_count": int(plan["discovery_case_count"]),
            "natural_observed": True,
            "phase333_tested": True,
            "phase333_event_role": "largest_positive_residual_write_increment",
            "phase333_interface": interface,
            "phase333_block_windows": plan["block_windows"],
            "phase333_dynamic_sequence_stable": seq["dynamic_sequence_stable"],
            "phase333_heldout_peak_depth": seq["heldout_median_relative_peak_depth"],
            "phase333_correct_block_specific": effect["correct_block_specific"] if effect else False,
            "phase333_phrase_delta": effect["mean_correct_block_4_phrase_delta"] if effect else 0.0,
            "phase333_rank_improvement": effect["mean_correct_block_4_rank_improvement"] if effect else 0.0,
            "phase333_behavior_gain_rate": effect["free_generation_gain_rate"] if effect else 0.0,
            "evidence_level": "L3_dynamic_path_not_causally_closed",
            "evidence_status": "heldout_dynamic_event_anchor_not_causally_closed",
            "causal_scope": "continuous_residual_block_not_single_neuron",
            "single_unit_causal": False,
            "evidence_boundary": (
                "Discovery selected the largest positive residual write increment. Calibration and heldout depth "
                "were audited separately. This is a dynamic component anchor, not a causal neuron or closed path."
            ),
            "source_artifacts": [
                "tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_block_plans.jsonl",
                "tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_sequence_summary.jsonl",
            ],
            "published_at": generated_at,
        })
    old_nodes = [row for row in read_jsonl(output / "neuron_nodes.jsonl") if not row.get("phase333_tested")]
    write_jsonl(output / "neuron_nodes.jsonl", [*old_nodes, *nodes])
    write_jsonl(output / "phase333_dynamic_event_nodes.jsonl", nodes)
    write_jsonl(output / "phase333_sequence_summary.jsonl", sequence)
    write_jsonl(output / "phase333_interface_alignment.jsonl", alignment)
    write_jsonl(output / "phase333_block_local_summary.jsonl", local)
    write_jsonl(output / "phase333_compensation_candidates.jsonl", compensation)
    write_jsonl(output / "phase333_cross_model_summary.jsonl", cross)
    write_jsonl(output / "phase333_claim_registry.jsonl", claims)
    shutil.copy2(SOURCE / "phase333_global_summary.json", output / "phase333_global_summary.json")
    shutil.copy2(SOURCE / "phase333_report.md", output / "phase333_report.md")
    by_model = {model: [row for row in nodes if row["model"] == model] for model in ("qwen3", "glm4", "deepseek7b")}
    updated = 0
    for ref in refs:
        if ref["family_id"] != "reasoning_constraint" or ref["model"] not in by_model:
            continue
        partition_path = output / ref["path"]
        partition = read_json(partition_path)
        partition["nodes"] = [row for row in partition.get("nodes", []) if not row.get("phase333_tested")]
        partition["nodes"].extend(by_model[ref["model"]])
        partition["scope"]["phase"] = 333
        partition["scope"]["source_phases"] = sorted(set([*partition["scope"].get("source_phases", []), 333]))
        model_alignment = [row for row in alignment if row["model"] == ref["model"]]
        model_local = [row for row in local if row["model"] == ref["model"]]
        model_compensation = [row for row in compensation if row["model"] == ref["model"]]
        partition["metrics"].update({
            "phase333_dynamic_event_count": len(by_model[ref["model"]]),
            "phase333_stable_sequence_count": sum(row["dynamic_sequence_stable"] for row in sequence if row["model"] == ref["model"]),
            "phase333_functional_alignment_count": sum(row["functional_interface_alignment"] for row in model_alignment),
            "phase333_specific_block_cell_count": sum(row["correct_block_specific"] for row in model_local),
            "phase333_compensation_candidate_count": sum(row["compensation_explained"] for row in model_compensation),
            "phase333_full_gate_pass_count": summary["results"]["full_gate_pass_count"],
            "phase333_single_unit_causal_count": 0,
        })
        partition["path"]["phase333_dynamic_path"] = {
            "sequence_summary": [row for row in sequence if row["model"] == ref["model"]],
            "interface_alignment": model_alignment,
            "block_summary": model_local,
            "cross_model_summary": cross[0],
            "display": "frozen_dynamic_event_anchors_only",
        }
        partition["mapping_status"] = "phase333_dynamic_path_mapped_not_causally_closed"
        partition["evidence_boundary"] = (
            "Phase333 maps frozen functional-time residual-write anchors. Continuous-block exchanges, controls, "
            "and lagged recovery are reported separately; no node is a causal single neuron."
        )
        partition["generated_at"] = generated_at
        write_json(partition_path, partition)
        ref.update({
            "mapping_status": partition["mapping_status"],
            "phase333_dynamic_event_count": len(by_model[ref["model"]]),
            "phase333_specific_block_cell_count": sum(row["correct_block_specific"] for row in model_local),
            "phase333_full_gate_pass_count": summary["results"]["full_gate_pass_count"],
        })
        updated += 1
    index["generated_at"] = generated_at
    write_json(output / "neuron_index.json", index)
    manifest = read_json(output / "manifest.json")
    manifest["phase"] = 333
    manifest["generated_at"] = generated_at
    manifest["partitions"] = refs
    manifest["metrics"].update({
        "phase333_registered_case_count": summary["denominator"]["registered_case_count"],
        "phase333_token_row_count": summary["denominator"]["token_row_count"],
        "phase333_dynamic_path_row_count": summary["denominator"]["dynamic_path_row_count"],
        "phase333_condition_row_count": summary["denominator"]["condition_row_count"],
        "phase333_dynamic_response_row_count": summary["denominator"]["dynamic_response_row_count"],
        "phase333_dynamic_event_count": len(nodes),
        "phase333_dynamic_sequence_stable_count": summary["results"]["cross_model_dynamic_sequence_stable_count"],
        "phase333_state_block_effective_count": summary["results"]["cross_model_state_block_effective_count"],
        "phase333_full_gate_pass_count": summary["results"]["full_gate_pass_count"],
        "single_unit_causal_count": 0,
    })
    manifest["evidence_boundary"] = {
        **manifest.get("evidence_boundary", {}),
        "statement": (
            "Phase333 adds dynamic event anchors and continuous residual-block audits. Full gate counts are "
            "reported explicitly; no displayed event is a causal single neuron."
        ),
        "phase333_full_gate_pass_count": summary["results"]["full_gate_pass_count"],
        "language_encoding_mechanism_closed": False,
        "single_unit_intervention_gate_open": False,
    }
    manifest["source_artifacts"] = list(dict.fromkeys([
        *manifest.get("source_artifacts", []),
        "tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_report.md",
        "tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_global_summary.json",
        "tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_claim_registry.jsonl",
    ]))
    write_json(output / "manifest.json", manifest)
    sanitized = sanitize_json_artifacts(output)
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            checksums.append({"path": path.relative_to(output).as_posix(), "sha256": sha256(path)})
    write_json(output / "checksums.json", {"schema_version": "artifact_checksums.v1", "files": checksums})
    shutil.copytree(output, public, dirs_exist_ok=True)
    return {
        "phase": 333,
        "prior_node_count": len(old_nodes),
        "dynamic_event_count": len(nodes),
        "updated_partition_count": updated,
        "full_gate_pass_count": summary["results"]["full_gate_pass_count"],
        "single_unit_causal_count": 0,
        "sanitized_artifact_count": sanitized,
        "valid": len(nodes) == 18 and updated == 3,
    }


if __name__ == "__main__":
    print(json.dumps(publish(), ensure_ascii=False, indent=2))
