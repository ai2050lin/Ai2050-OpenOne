#!/usr/bin/env python3
"""Publish Phase334 component-level natural-necessity candidates without inventing neurons."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas"
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
    summary = read_json(SOURCE / "phase334_global_summary.json")
    plans = read_jsonl(SOURCE / "phase334_frozen_necessity_plans.jsonl")
    local = read_jsonl(SOURCE / "phase334_local_necessity_summary.jsonl")
    propagation = read_jsonl(SOURCE / "phase334_propagation_candidates.jsonl")
    cross = read_jsonl(SOURCE / "phase334_cross_model_summary.jsonl")
    claims = read_jsonl(SOURCE / "phase334_claim_registry.jsonl")
    local_map = {
        (row["model"], row["family_id"], row["mechanism_id"], row["interface"]): row
        for row in local
    }
    index = read_json(output / "neuron_index.json")
    refs = index["partitions"]
    ref_map = {(row["model"], row["family_id"]): row for row in refs}
    nodes = []
    for plan in plans:
        key = (plan["model"], plan["family_id"], plan["mechanism_id"], plan["interface"])
        effect = local_map[key]
        partition = read_json(output / ref_map[(plan["model"], plan["family_id"])]["path"])
        layer = int(plan["selected_layer"])
        nodes.append({
            "schema_version": "neuron_atlas_node.v1",
            "node_id": (
                f"p334:{plan['model']}:{plan['family_id']}:{plan['mechanism_id']}:"
                f"{plan['interface']}:L{layer}:{plan['selected_component']}:{plan['selected_position_role']}"
            ),
            "node_type": "natural_necessity_component_candidate",
            "family_id": plan["family_id"],
            "family_name": partition["family"]["family_name"],
            "mechanism_id": plan["mechanism_id"],
            "relation": plan["mechanism_id"],
            "cohort": plan.get("cohort", effect["cohort"]),
            "model": plan["model"],
            "model_revision": partition["model_snapshot"]["model_revision"],
            "layer": layer,
            "component": plan["selected_component"],
            "unit_kind": "natural_necessity_component_path",
            "unit_index": layer,
            "natural_activation": float(plan["median_contrast_norm"]),
            "candidate_score": float(plan["median_contrast_norm"]),
            "display_priority": 2.8 if effect["local_gate_pass"] else 1.9,
            "case_count": int(effect["common_valid_case_count"]),
            "natural_observed": True,
            "phase334_tested": True,
            "phase334_interface": plan["interface"],
            "phase334_depth_bin": plan["depth_bin"],
            "phase334_position_role": plan["selected_position_role"],
            "phase334_component": plan["selected_component"],
            "phase334_baseline_eligible_case_count": effect["baseline_eligible_case_count"],
            "phase334_common_valid_case_count": effect["common_valid_case_count"],
            "phase334_phrase_logprob_loss": effect["mean_correct_phrase_logprob_loss"],
            "phase334_target_rank_loss": effect["mean_correct_target_rank_loss"],
            "phase334_behavior_loss_rate": effect["correct_behavior_loss_rate"],
            "phase334_control_phrase_loss": effect["max_control_phrase_logprob_loss"],
            "phase334_natural_necessity_specific": effect["natural_necessity_specific"],
            "phase334_propagation_candidate_rate": effect["propagation_candidate_rate"],
            "phase334_local_gate_pass": effect["local_gate_pass"],
            "evidence_level": effect["evidence_level"],
            "evidence_status": (
                "controlled_component_necessity_candidate" if effect["local_gate_pass"]
                else "natural_component_candidate_not_causally_confirmed"
            ),
            "causal_scope": "component_level_natural_deletion_not_single_neuron",
            "single_unit_causal": False,
            "evidence_boundary": (
                "Discovery used target-independent paired natural contrast; calibration selected the depth; "
                "heldout used common-valid controls. This node is a component path candidate, not a neuron."
            ),
            "source_artifacts": [
                "tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_frozen_necessity_plans.jsonl",
                "tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_local_necessity_summary.jsonl",
            ],
            "published_at": generated_at,
        })
    old_nodes = [row for row in read_jsonl(output / "neuron_nodes.jsonl") if not row.get("phase334_tested")]
    write_jsonl(output / "neuron_nodes.jsonl", [*old_nodes, *nodes])
    write_jsonl(output / "phase334_natural_necessity_nodes.jsonl", nodes)
    write_jsonl(output / "phase334_local_necessity_summary.jsonl", local)
    write_jsonl(output / "phase334_propagation_candidates.jsonl", propagation)
    write_jsonl(output / "phase334_cross_model_summary.jsonl", cross)
    write_jsonl(output / "phase334_claim_registry.jsonl", claims)
    shutil.copy2(SOURCE / "phase334_global_summary.json", output / "phase334_global_summary.json")
    shutil.copy2(SOURCE / "phase334_report.md", output / "phase334_report.md")
    updated = 0
    for ref in refs:
        key = (ref["model"], ref["family_id"])
        model_family_nodes = [row for row in nodes if (row["model"], row["family_id"]) == key]
        if not model_family_nodes:
            continue
        partition_path = output / ref["path"]
        partition = read_json(partition_path)
        partition["nodes"] = [row for row in partition.get("nodes", []) if not row.get("phase334_tested")]
        partition["nodes"].extend(model_family_nodes)
        partition["scope"]["phase"] = 334
        partition["scope"]["source_phases"] = sorted(set([*partition["scope"].get("source_phases", []), 334]))
        model_local = [row for row in local if (row["model"], row["family_id"]) == key]
        model_propagation = [row for row in propagation if (row["model"], row["family_id"]) == key]
        partition["metrics"].update({
            "phase334_candidate_node_count": len(model_family_nodes),
            "phase334_baseline_eligible_cell_count": sum(row["baseline_eligible_case_count"] >= 6 for row in model_local),
            "phase334_natural_necessity_candidate_count": sum(row["natural_necessity_specific"] for row in model_local),
            "phase334_propagation_candidate_count": sum(row["propagation_candidate"] for row in model_propagation),
            "phase334_local_gate_pass_count": sum(row["local_gate_pass"] for row in model_local),
            "phase334_cross_model_gate_count": summary["results"]["cross_model_natural_necessity_gate_count"],
            "phase334_single_unit_causal_count": 0,
        })
        partition["path"]["phase334_natural_necessity"] = {
            "local_summary": model_local,
            "cross_model_summary": [row for row in cross if row["family_id"] == ref["family_id"]],
            "display": "component_level_receiver_natural_necessity_candidates",
        }
        partition["mapping_status"] = "phase334_natural_necessity_audited_not_mechanism_closed"
        partition["evidence_boundary"] = (
            "Phase334 maps receiver-natural component candidates. A local pass remains component-level "
            "necessity evidence and is neither a causal neuron nor complete mechanism closure."
        )
        partition["generated_at"] = generated_at
        write_json(partition_path, partition)
        ref.update({
            "mapping_status": partition["mapping_status"],
            "phase334_candidate_node_count": len(model_family_nodes),
            "phase334_local_gate_pass_count": sum(row["local_gate_pass"] for row in model_local),
        })
        updated += 1
    index["generated_at"] = generated_at
    write_json(output / "neuron_index.json", index)
    manifest = read_json(output / "manifest.json")
    manifest["phase"] = 334
    manifest["generated_at"] = generated_at
    manifest["partitions"] = refs
    manifest["metrics"].update({
        "phase334_registered_case_count": summary["denominator"]["registered_case_count"],
        "phase334_natural_contrast_row_count": summary["denominator"]["natural_contrast_row_count"],
        "phase334_heldout_condition_row_count": summary["denominator"]["heldout_condition_row_count"],
        "phase334_candidate_node_count": len(nodes),
        "phase334_baseline_eligible_cell_count": summary["results"]["baseline_eligible_cell_count"],
        "phase334_natural_necessity_candidate_count": summary["results"]["local_natural_necessity_candidate_count"],
        "phase334_local_propagation_pass_count": summary["results"]["local_propagation_pass_count"],
        "phase334_cross_model_gate_count": summary["results"]["cross_model_natural_necessity_gate_count"],
        "phase334_training_checkpoint_track_count": 0,
        "single_unit_causal_count": 0,
    })
    manifest["evidence_boundary"] = {
        **manifest.get("evidence_boundary", {}),
        "statement": (
            "Phase334 adds component-level receiver-natural necessity candidates across three core families. "
            "No node is a causal single neuron and no mechanism is declared closed by necessity alone."
        ),
        "phase334_cross_model_gate_count": summary["results"]["cross_model_natural_necessity_gate_count"],
        "training_formation_track_available_count": 0,
        "language_encoding_mechanism_closed": False,
    }
    manifest["source_artifacts"] = list(dict.fromkeys([
        *manifest.get("source_artifacts", []),
        "tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_report.md",
        "tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_global_summary.json",
        "tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_claim_registry.jsonl",
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
        "phase": 334, "prior_node_count": len(old_nodes),
        "natural_necessity_node_count": len(nodes), "updated_partition_count": updated,
        "cross_model_gate_count": summary["results"]["cross_model_natural_necessity_gate_count"],
        "single_unit_causal_count": 0, "sanitized_artifact_count": sanitized,
        "valid": len(nodes) == 54 and updated == 9,
    }


if __name__ == "__main__":
    print(json.dumps(publish(), ensure_ascii=False, indent=2))
