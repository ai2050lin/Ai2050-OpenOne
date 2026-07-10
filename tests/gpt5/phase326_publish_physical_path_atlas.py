#!/usr/bin/env python3
"""Merge Phase326 distributed carrier evidence into the Phase325 client atlas."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase325_pattern_family_neuron_atlas as p325  # noqa: E402


PHASE326 = ROOT / "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas"
OUTPUT = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
PUBLIC = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def convert_node(row: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    confirmed = bool(row.get("expanded_confirmation_pass"))
    supported = bool(row.get("heldout_set_necessity_pass"))
    if confirmed:
        status = "expanded_cross_object_component_set_member"
    elif supported:
        status = "heldout_component_set_member"
    else:
        status = "discovery_component_set_candidate"
    return {
        "schema_version": "neuron_atlas_node.v1",
        "node_id": row["node_id"],
        "node_type": "component_set_member",
        "family_id": row["family_id"],
        "family_name": family["family_name"],
        "relation": row["mechanism_id"],
        "mechanism_id": row["mechanism_id"],
        "model": row["model"],
        "model_revision": row.get("model_revision") or "local_unknown",
        "layer": int(row["layer"]),
        "component": row["component"],
        "unit_kind": row["unit_kind"],
        "unit_index": int(row["unit_index"]),
        "position_role": row.get("position_role"),
        "token_position": row.get("position_role"),
        "candidate_score": float(row.get("display_priority") or 0.0),
        "candidate_score_mean": float(row.get("display_priority") or 0.0),
        "activation_abs_mean": None,
        "readout_contribution_max_abs": None,
        "case_count": int(row.get("discovery_independent_cases") or 0),
        "case_ids": [],
        "target_labels": [],
        "coverage_objects": int(row.get("discovery_independent_cases") or 0),
        "coverage_templates": int(row.get("discovery_templates") or 0),
        "natural_observed": True,
        "natural_activation": None,
        "natural_case_id": None,
        "group_intervention_supported": supported,
        "expanded_confirmation_pass": confirmed,
        "group_margin_delta_min": None,
        "causal_scope": "distributed_component_set_not_single_unit",
        "evidence_level": row.get("evidence_level") or "L3_candidate",
        "evidence_status": status,
        "evidence_boundary": "naturally observed component-set member; set ablation does not establish this member as individually causal",
        "display_priority": float(row.get("display_priority") or 0.0) + (0.18 if confirmed else 0.0) + (0.08 if supported else 0.0),
        "source_artifacts": [
            "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas/phase326_carrier_sets.jsonl",
            "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas/phase326_registered_heldout.jsonl",
            "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas/phase326_expanded_confirmation_audits.jsonl",
        ],
        "run_id": "phase326_distributed_carrier_atlas",
        "single_unit_causal": False,
    }


def mechanism_anchors(family_id: str, model: str, nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    anchors: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        grouped[(node["mechanism_id"], int(node["layer"]))].append(node)
    source_id = f"{family_id}:{model}:source"
    readout_id = f"{family_id}:{model}:readout"
    for mechanism in sorted({key[0] for key in grouped}):
        mechanism_anchors_rows = []
        for key in sorted((key for key in grouped if key[0] == mechanism), key=lambda item: item[1]):
            layer_nodes = grouped[key]
            anchor_id = f"{family_id}:{model}:{mechanism}:L{key[1]}:anchor"
            anchor = {
                "anchor_id": anchor_id,
                "mechanism_id": mechanism,
                "layer": key[1],
                "candidate_count": len(layer_nodes),
                "natural_overlap_count": len(layer_nodes),
                "group_supported_count": sum(bool(node["group_intervention_supported"]) for node in layer_nodes),
                "expanded_confirmation_count": sum(bool(node["expanded_confirmation_pass"]) for node in layer_nodes),
                "attention_metrics": None,
                "mlp_metrics": None,
                "residual_metrics": None,
                "evidence_level": "L3+L4",
                "evidence_boundary": "frozen distributed component-set candidates; anchor order is not a neuron-to-neuron causal edge",
            }
            anchors.append(anchor)
            mechanism_anchors_rows.append(anchor)
            for node in layer_nodes:
                memberships.append({
                    "schema_version": "neuron_atlas_edge.v1",
                    "edge_id": f"membership:{anchor_id}:{node['node_id']}",
                    "family_id": family_id,
                    "mechanism_id": mechanism,
                    "model": model,
                    "source_id": anchor_id,
                    "target_id": node["node_id"],
                    "relation": "contains_distributed_component_candidate",
                    "evidence_level": node["evidence_level"],
                    "causal": False,
                    "evidence_boundary": node["evidence_boundary"],
                })
        path = [source_id, *[anchor["anchor_id"] for anchor in mechanism_anchors_rows], readout_id]
        for source, target in zip(path, path[1:]):
            sequences.append({
                "schema_version": "neuron_atlas_edge.v1",
                "edge_id": f"phase326-sequence:{mechanism}:{source}:{target}",
                "family_id": family_id,
                "mechanism_id": mechanism,
                "model": model,
                "source_id": source,
                "target_id": target,
                "relation": "observed_distributed_component_sequence",
                "evidence_level": "L3",
                "causal": False,
                "evidence_boundary": "ordered candidate layers, not a demonstrated causal propagation edge",
            })
    return anchors, memberships, sequences


def build_phase326_partition(
    family: dict[str, Any], model: str, snapshot: dict[str, Any], source_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes = [convert_node(row, family) for row in source_nodes]
    anchors, memberships, sequences = mechanism_anchors(family["family_id"], model, nodes)
    return {
        "schema_version": "neuron_atlas_partition.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "model": model,
        "model_snapshot": snapshot,
        "scope": {"relations": sorted({node["mechanism_id"] for node in nodes}), "phase": 326, "source_phases": [326]},
        "mapping_status": "distributed_component_set_candidates_not_single_unit_causal",
        "evidence_boundary": "Phase326 supports distributed attention-head and MLP-product-group candidates; it does not identify single causal neurons or natural sufficiency",
        "path": {
            "source": {"anchor_id": f"{family['family_id']}:{model}:source", "stage": "registered_prompt", "evidence_level": "L2"},
            "layer_anchors": anchors,
            "readout": {"anchor_id": f"{family['family_id']}:{model}:readout", "stage": "candidate_answer_readout", "evidence_level": "L2", "global_closed": False},
            "sequence_edges": sequences,
        },
        "nodes": nodes,
        "membership_edges": memberships,
        "metrics": {
            "source_row_count": len(nodes),
            "unique_unit_count": len(nodes),
            "single_neuron_candidate_count": 0,
            "component_set_member_count": len(nodes),
            "natural_overlap_count": len(nodes),
            "group_supported_candidate_count": sum(bool(node["group_intervention_supported"]) for node in nodes),
            "expanded_confirmed_candidate_count": sum(bool(node["expanded_confirmation_pass"]) for node in nodes),
            "candidate_layer_count": len({node["layer"] for node in nodes}),
            "trace_event_count": 0,
            "group_intervention_row_count": 0,
            "single_unit_causal_count": 0,
        },
        "source_artifacts": nodes[0]["source_artifacts"] if nodes else [],
    }


def merge_partition(base: dict[str, Any], addition: dict[str, Any]) -> dict[str, Any]:
    base["nodes"].extend(addition["nodes"])
    base["membership_edges"].extend(addition["membership_edges"])
    base["path"]["layer_anchors"].extend(addition["path"]["layer_anchors"])
    base["path"]["sequence_edges"].extend(addition["path"]["sequence_edges"])
    base["scope"] = {"relations": ["color", *addition["scope"]["relations"]], "phase": 326, "source_phases": [286, 287, 326]}
    base["mapping_status"] = "single_unit_candidates_plus_distributed_component_set_candidates"
    base["evidence_boundary"] = "color includes Phase286/287 unit candidates; Phase326 adds set members, not single-unit causality"
    base["source_artifacts"].extend(addition["source_artifacts"])
    metrics = base["metrics"]
    metrics["unique_unit_count"] = len(base["nodes"])
    metrics["single_neuron_candidate_count"] = len([node for node in base["nodes"] if node["node_type"] == "unit_candidate"])
    metrics["component_set_member_count"] = len([node for node in base["nodes"] if node["node_type"] == "component_set_member"])
    metrics["natural_overlap_count"] = sum(bool(node["natural_observed"]) for node in base["nodes"])
    metrics["group_supported_candidate_count"] = sum(bool(node["group_intervention_supported"]) for node in base["nodes"])
    metrics["expanded_confirmed_candidate_count"] = sum(bool(node.get("expanded_confirmation_pass")) for node in base["nodes"])
    metrics["candidate_layer_count"] = len({node["layer"] for node in base["nodes"]})
    return base


def validate(output: Path) -> dict[str, int]:
    manifest = read_json(output / "manifest.json")
    nodes = read_jsonl(output / "neuron_nodes.jsonl")
    edges = read_jsonl(output / "neuron_edges.jsonl")
    assert manifest["metrics"]["mapped_family_count"] == 2
    assert manifest["metrics"]["unique_unit_count"] == len(nodes)
    assert manifest["metrics"]["edge_count"] == len(edges)
    assert len({node["node_id"] for node in nodes}) == len(nodes)
    assert all(not edge["causal"] for edge in edges)
    assert manifest["metrics"]["single_unit_causal_count"] == 0
    return {"mapped_families": 2, "nodes": len(nodes), "edges": len(edges)}


def build(output: Path, public: Path) -> dict[str, Any]:
    temporary_public = ROOT / "tests/gpt5_temp/phase326_base_public"
    shutil.rmtree(temporary_public, ignore_errors=True)
    p325.build_bundle(output, temporary_public)
    shutil.rmtree(temporary_public, ignore_errors=True)
    family_payload = read_json(output / "families.json")
    family_by_id = {row["family_id"]: row for row in family_payload["families"]}
    manifest = read_json(output / "manifest.json")
    phase326_nodes = read_jsonl(PHASE326 / "phase326_atlas_nodes.jsonl")
    confirmation_audits = read_jsonl(PHASE326 / "phase326_expanded_confirmation_audits.jsonl")
    confirmation_map = {
        (row["model"], row["family_id"], row["mechanism_id"]): bool(
            row.get("strict_confirmation_pass", row["expanded_confirmation_pass"])
        )
        for row in confirmation_audits
    }
    for row in phase326_nodes:
        row["expanded_confirmation_pass"] = confirmation_map.get(
            (row["model"], row["family_id"], row["mechanism_id"]), False
        )

    partitions = []
    for model in MODELS:
        content_path = output / "partitions/content_knowledge" / f"{model}.json"
        content = read_json(content_path)
        snapshot = content["model_snapshot"]
        content_addition = build_phase326_partition(
            family_by_id["content_knowledge"], model, snapshot,
            [row for row in phase326_nodes if row["model"] == model and row["family_id"] == "content_knowledge"],
        )
        content = merge_partition(content, content_addition)
        write_json(content_path, content)
        partitions.append({
            "family_id": "content_knowledge", "model": model,
            "path": f"partitions/content_knowledge/{model}.json",
            "mapping_status": content["mapping_status"], **content["metrics"],
        })
        reasoning = build_phase326_partition(
            family_by_id["reasoning_constraint"], model, snapshot,
            [row for row in phase326_nodes if row["model"] == model and row["family_id"] == "reasoning_constraint"],
        )
        reasoning_path = output / "partitions/reasoning_constraint" / f"{model}.json"
        write_json(reasoning_path, reasoning)
        partitions.append({
            "family_id": "reasoning_constraint", "model": model,
            "path": f"partitions/reasoning_constraint/{model}.json",
            "mapping_status": reasoning["mapping_status"], **reasoning["metrics"],
        })

    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    for item in partitions:
        partition = read_json(output / item["path"])
        all_nodes.extend(partition["nodes"])
        all_edges.extend(partition["membership_edges"])
        all_edges.extend(partition["path"]["sequence_edges"])
    write_jsonl(output / "neuron_nodes.jsonl", all_nodes)
    write_jsonl(output / "neuron_edges.jsonl", all_edges)

    intervention_rows = read_jsonl(output / "neuron_interventions.jsonl")
    for row in read_jsonl(PHASE326 / "phase326_intervention_rows.jsonl"):
        intervention_rows.append({
            **row,
            "schema_version": "neuron_atlas_group_intervention.v1",
            "causal_scope": "distributed_component_set_not_single_unit",
            "single_unit_causal": False,
        })
    for row in read_jsonl(PHASE326 / "phase326_expanded_confirmation_audits.jsonl"):
        intervention_rows.append({
            **row,
            "schema_version": "neuron_atlas_expanded_confirmation.v1",
            "causal_scope": "distributed_component_set_not_single_unit",
            "single_unit_causal": False,
        })
    write_jsonl(output / "neuron_interventions.jsonl", intervention_rows)

    run_rows = read_jsonl(output / "neuron_runs.jsonl")
    for model in MODELS:
        for family_id in ("content_knowledge", "reasoning_constraint"):
            current = [node for node in all_nodes if node["model"] == model and node["family_id"] == family_id and node["node_type"] == "component_set_member"]
            run_rows.append({
                "schema_version": "neuron_atlas_run.v1",
                "run_id": f"phase326_{family_id}_{model}",
                "model": model,
                "family_id": family_id,
                "status": "complete",
                "source_phase": 326,
                "component_set_member_count": len(current),
                "expanded_confirmed_candidate_count": sum(bool(node["expanded_confirmation_pass"]) for node in current),
                "single_unit_causal_count": 0,
            })
    write_jsonl(output / "neuron_runs.jsonl", run_rows)

    for family in family_payload["families"]:
        refs = [item for item in partitions if item["family_id"] == family["family_id"]]
        family["physical_mapping"] = {
            "status": "mapped_physical_candidates" if refs else "not_mapped_to_real_units",
            "models": [item["model"] for item in refs],
            "unique_unit_count": sum(item["unique_unit_count"] for item in refs),
            "single_neuron_candidate_count": sum(item.get("single_neuron_candidate_count", 0) for item in refs),
            "component_set_member_count": sum(item.get("component_set_member_count", 0) for item in refs),
            "expanded_confirmed_candidate_count": sum(item.get("expanded_confirmed_candidate_count", 0) for item in refs),
            "single_unit_causal_count": 0,
            "partition_refs": [item["path"] for item in refs],
        }
    write_json(output / "families.json", family_payload)
    write_json(output / "neuron_index.json", {
        "schema_version": "neuron_atlas_index.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "partitions": partitions,
        "selection_rule": "load only the selected family and model partition",
        "default_family_id": "content_knowledge",
        "default_model": "qwen3",
    })

    manifest["phase"] = 326
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["title"] = "Evidence-scoped pattern-family physical path atlas"
    manifest["partitions"] = partitions
    manifest["metrics"].update({
        "mapped_family_count": 2,
        "unique_unit_count": len(all_nodes),
        "single_neuron_candidate_count": sum(node["node_type"] == "unit_candidate" for node in all_nodes),
        "component_set_member_count": sum(node["node_type"] == "component_set_member" for node in all_nodes),
        "expanded_confirmed_candidate_count": sum(bool(node.get("expanded_confirmation_pass")) for node in all_nodes),
        "edge_count": len(all_edges),
        "group_intervention_row_count": len(intervention_rows),
        "single_unit_causal_count": 0,
    })
    manifest["evidence_boundary"] = {
        "mapped_scope": "content_knowledge and reasoning_constraint physical candidates",
        "unmapped_families": [
            family["family_id"] for family in family_payload["families"]
            if not family["physical_mapping"]["models"]
        ],
        "statement": "Phase326 adds distributed attention-head and MLP-product-group set members. These are physical component candidates, not single causal neurons.",
        "single_unit_causal_closure": False,
        "natural_sufficiency_closure": False,
    }
    manifest["source_artifacts"].extend([
        "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas/phase326_cross_model_summary.json",
        "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas/phase326_expanded_confirmation_cross_model.json",
    ])
    write_json(output / "manifest.json", manifest)
    validation = validate(output)
    manifest["validation"] = {"status": "passed", **validation}
    write_json(output / "manifest.json", manifest)

    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            checksums.append({"path": path.relative_to(output).as_posix(), "sha256": p325.file_sha256(path)})
    write_json(output / "checksums.json", {"schema_version": "artifact_checksums.v1", "files": checksums})
    if public.exists():
        shutil.rmtree(public)
    shutil.copytree(output, public)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate(OUTPUT), ensure_ascii=False, indent=2))
        return
    manifest = build(OUTPUT, PUBLIC)
    print(json.dumps({"status": "ok", "metrics": manifest["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
