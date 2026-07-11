#!/usr/bin/env python3
"""Publish Phase331 evidence overlays without changing the established 3D geometry."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit"
OUTPUT = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
PUBLIC = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(output: Path = OUTPUT, public: Path = PUBLIC) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    global_summary = read_json(SOURCE / "phase331_global_summary.json")
    local_rows = read_jsonl(SOURCE / "phase331_local_summary.jsonl")
    compensation_rows = read_jsonl(SOURCE / "phase331_compensation_summary.jsonl")
    cross_rows = read_jsonl(SOURCE / "phase331_cross_model_summary.jsonl")
    claims = read_jsonl(SOURCE / "phase331_claim_registry.jsonl")
    local = {
        (row["model"], row["family_id"], row["mechanism_id"], row["interface"]): row
        for row in local_rows if row["cohort"] == "positive"
    }
    compensation = {
        (row["model"], row["family_id"], row["mechanism_id"], row["interface"]): row
        for row in compensation_rows if row["cohort"] == "positive"
    }
    cross = {(row["family_id"], row["mechanism_id"]): row for row in cross_rows}
    selected = set(cross)
    nodes = read_jsonl(output / "neuron_nodes.jsonl")
    updated_nodes = 0
    for node in nodes:
        key = (node.get("family_id"), node.get("mechanism_id"))
        local_key_raw = (node.get("model"), *key, "raw_completion")
        local_key_chat = (node.get("model"), *key, "chat_template")
        if key not in selected or local_key_raw not in local or local_key_chat not in local:
            continue
        raw = local[local_key_raw]
        chat = local[local_key_chat]
        raw_comp = compensation[local_key_raw]
        chat_comp = compensation[local_key_chat]
        global_row = cross[key]
        node.update({
            "phase331_tested": True,
            "phase331_interfaces": ["raw_completion", "chat_template"],
            "phase331_expanded_heldout_items": [19, 20, 21, 22],
            "phase331_template_count": 3,
            "phase331_raw_readout_specific": raw["readout_specific"],
            "phase331_chat_readout_specific": chat["readout_specific"],
            "phase331_raw_joint_margin_delta": raw["joint_mean_margin_delta"],
            "phase331_chat_joint_margin_delta": chat["joint_mean_margin_delta"],
            "phase331_raw_phrase_logprob_delta": raw["joint_mean_phrase_logprob_delta"],
            "phase331_chat_phrase_logprob_delta": chat["joint_mean_phrase_logprob_delta"],
            "phase331_raw_behavior_changed_rate": raw["joint_behavior_changed_rate"],
            "phase331_chat_behavior_changed_rate": chat["joint_behavior_changed_rate"],
            "phase331_raw_compensation_ratio": raw_comp["mean_unselected_component_compensation_ratio"],
            "phase331_chat_compensation_ratio": chat_comp["mean_unselected_component_compensation_ratio"],
            "phase331_raw_late_recovery_fraction": raw_comp["mean_late_residual_recovery_fraction"],
            "phase331_chat_late_recovery_fraction": chat_comp["mean_late_residual_recovery_fraction"],
            "phase331_member_localized": global_row["gate"]["member_localized"],
            "phase331_full_generation_changed": global_row["gate"]["full_generation_changed"],
            "phase331_full_gate_pass": global_row["full_gate_pass"],
            "phase331_evidence_level": global_row["evidence_level"],
            "phase331_status": (
                "expanded_mechanism_candidate" if global_row["full_gate_pass"]
                else "frozen_set_candidate_not_behavior_closed"
            ),
            "phase331_evidence_boundary": (
                "Expanded four-item, three-template, raw/chat, three-model audit. The displayed unit remains a "
                "member of a distributed set; no Phase331 row establishes single-neuron causality."
            ),
            "single_unit_causal": False,
        })
        node["evidence_level"] = global_row["evidence_level"]
        node["evidence_status"] = node["phase331_status"]
        updated_nodes += 1
    write_jsonl(output / "neuron_nodes.jsonl", nodes)
    write_jsonl(output / "phase331_mechanism_overlays.jsonl", cross_rows)
    write_jsonl(output / "phase331_claim_registry.jsonl", claims)
    write_jsonl(output / "phase331_compensation_summary.jsonl", compensation_rows)
    shutil.copy2(SOURCE / "phase331_global_summary.json", output / "phase331_global_summary.json")
    shutil.copy2(SOURCE / "phase331_report.md", output / "phase331_report.md")

    index = read_json(output / "neuron_index.json")
    partition_refs = index["partitions"]
    for ref in partition_refs:
        family_cross = [row for row in cross_rows if row["family_id"] == ref["family_id"]]
        if not family_cross:
            continue
        partition_path = output / ref["path"]
        partition = read_json(partition_path)
        node_map = {row["node_id"]: row for row in nodes if row.get("model") == ref["model"] and row.get("family_id") == ref["family_id"]}
        for partition_node in partition.get("nodes", []):
            updated = node_map.get(partition_node["node_id"])
            if not updated or not updated.get("phase331_tested"):
                continue
            for key, value in updated.items():
                if key.startswith("phase331_") or key in {"evidence_level", "evidence_status", "single_unit_causal"}:
                    partition_node[key] = value
        partition["scope"]["phase"] = 331
        partition["scope"]["source_phases"] = sorted(set([*partition["scope"].get("source_phases", []), 331]))
        partition["metrics"].update({
            "phase331_refined_mechanism_count": len(family_cross),
            "phase331_full_gate_pass_count": sum(row["full_gate_pass"] for row in family_cross),
            "phase331_behavior_mechanism_closed_count": sum(row["behavior_mechanism_closed"] for row in family_cross),
            "phase331_single_unit_causal_count": 0,
        })
        partition["path"]["phase331_evidence_overlay"] = family_cross
        partition["mapping_status"] = (
            "phase331_refined_set_candidates_not_single_unit_causal"
            if family_cross else partition["mapping_status"]
        )
        partition["evidence_boundary"] = (
            "Phase331 refines only five frozen Phase330 component sets with four untouched heldout objects, three "
            "templates, raw/chat interfaces, member ablations, natural controls, and compensation tracing. "
            "Collection coverage and set-level effects do not establish single-neuron or language-mechanism closure."
        )
        partition["generated_at"] = generated_at
        write_json(partition_path, partition)
        ref.update({
            "mapping_status": partition["mapping_status"],
            "phase331_refined_mechanism_count": len(family_cross),
            "phase331_full_gate_pass_count": sum(row["full_gate_pass"] for row in family_cross),
        })
    index["generated_at"] = generated_at
    write_json(output / "neuron_index.json", index)

    manifest = read_json(output / "manifest.json")
    manifest["phase"] = 331
    manifest["generated_at"] = generated_at
    manifest["partitions"] = partition_refs
    manifest["metrics"].update({
        "phase331_registered_interface_case_count": global_summary["denominator"]["interface_case_count"],
        "phase331_condition_row_count": global_summary["denominator"]["condition_row_count"],
        "phase331_generation_row_count": global_summary["denominator"]["generation_row_count"],
        "phase331_compensation_path_row_count": global_summary["denominator"]["compensation_path_row_count"],
        "phase331_component_response_row_count": global_summary["denominator"]["component_response_row_count"],
        "phase331_refined_mechanism_count": 5,
        "phase331_expanded_readout_pass_count": global_summary["results"]["expanded_cross_model_cross_interface_readout_count"],
        "phase331_full_gate_pass_count": global_summary["results"]["full_gate_pass_count"],
        "phase331_behavior_mechanism_closed_count": global_summary["results"]["behavior_mechanism_closed_count"],
        "phase331_updated_component_member_count": updated_nodes,
        "single_unit_causal_count": 0,
    })
    manifest["evidence_boundary"] = {
        **manifest.get("evidence_boundary", {}),
        "statement": (
            "Phase331 completed expanded dual-interface and compensation audits for the five frozen Phase330 "
            "cross-model readout candidates and five matched controls. Full gate and behavior closure counts are "
            "reported separately; no displayed node is a causal single neuron."
        ),
        "phase331_full_gate_pass_count": global_summary["results"]["full_gate_pass_count"],
        "cross_model_behavior_mechanism_closed": global_summary["results"]["behavior_mechanism_closed_count"] > 0,
        "single_unit_intervention_gate_open": global_summary["single_unit_intervention_gate_open_count"] > 0,
        "language_encoding_mechanism_closed": False,
    }
    manifest["source_artifacts"] = list(dict.fromkeys([
        *manifest.get("source_artifacts", []),
        "tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit/phase331_report.md",
        "tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit/phase331_global_summary.json",
        "tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit/phase331_claim_registry.jsonl",
    ]))
    write_json(output / "manifest.json", manifest)
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            checksums.append({"path": path.relative_to(output).as_posix(), "sha256": sha256(path)})
    write_json(output / "checksums.json", {"schema_version": "artifact_checksums.v1", "files": checksums})
    shutil.copytree(output, public, dirs_exist_ok=True)
    return {
        "phase": 331,
        "updated_component_members": updated_nodes,
        "refined_mechanisms": 5,
        "full_gate_pass_count": global_summary["results"]["full_gate_pass_count"],
        "behavior_mechanism_closed_count": global_summary["results"]["behavior_mechanism_closed_count"],
        "single_unit_causal_count": 0,
        "valid": updated_nodes == 60,
    }


if __name__ == "__main__":
    print(json.dumps(publish(), ensure_ascii=False, indent=2))
