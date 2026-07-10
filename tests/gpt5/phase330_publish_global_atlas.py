#!/usr/bin/env python3
"""Publish the Phase330 nine-family physical atlas without changing DNN geometry."""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase325_pattern_family_neuron_atlas as phase325  # noqa: E402
import phase329_publish_full_vocabulary_atlas as phase329  # noqa: E402
from phase330_nine_family_case_bank import FAMILY_MECHANISMS, FAMILY_NAMES, MODELS  # noqa: E402


SOURCE = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas"
OUTPUT = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
PUBLIC = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


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
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def node_id(row: dict[str, Any]) -> str:
    return (
        f"p330:{row['model']}:{row['family_id']}:{row['mechanism_id']}:"
        f"{row['component_type']}:L{row['component_layer']}:{row['position_role']}:{row['component_index']}"
    )


def build_nodes(
    model: str,
    family: str,
    carriers: list[dict[str, Any]],
    matched: dict[tuple[str, str, str], dict[str, Any]],
    cross: dict[tuple[str, str], dict[str, Any]],
    model_revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = []
    edges = []
    for row in carriers:
        local = matched[(model, family, row["mechanism_id"])]
        global_result = cross[(family, row["mechanism_id"])]
        supported = bool(local["joint_readout_specific_vs_both_controls"])
        identifier = node_id(row)
        nodes.append({
            "schema_version": "neuron_atlas_node.v1",
            "node_id": identifier,
            "node_type": "component_set_member",
            "family_id": family,
            "family_name": f"{FAMILY_NAMES[family]}模式族",
            "relation": row["mechanism_id"],
            "mechanism_id": row["mechanism_id"],
            "model": model,
            "model_revision": model_revision,
            "layer": int(row["component_layer"]),
            "component": "attention" if row["component_type"] == "attention_head_input" else "mlp",
            "unit_kind": "attention_head" if row["component_type"] == "attention_head_input" else "mlp_product_group",
            "unit_index": int(row["component_index"]),
            "unit_start": int(row["component_start"]),
            "unit_end": int(row["component_end"]),
            "position_role": row["position_role"],
            "token_position": row["position_role"],
            "candidate_score": row["selection_score"],
            "candidate_score_mean": row["discovery_mean_contribution"],
            "activation_abs_mean": None,
            "readout_contribution_max_abs": abs(float(row["discovery_mean_contribution"])),
            "case_count": 24,
            "case_ids": [],
            "target_labels": [],
            "coverage_objects": 12,
            "coverage_templates": 2,
            "natural_observed": True,
            "natural_activation": None,
            "natural_case_id": None,
            "group_intervention_supported": supported,
            "expanded_confirmation_pass": False,
            "group_margin_delta_min": min(
                float(local["joint_minus_random_margin"]),
                float(local["joint_minus_wrong_layer_margin"]),
            ),
            "causal_scope": "registered_heldout_component_set_not_single_unit",
            "evidence_level": "L4_set_readout" if supported else "L3_candidate",
            "evidence_status": "registered_set_readout_specific" if supported else "frozen_component_candidate",
            "evidence_boundary": (
                "Phase330 selected this head/group on discovery data and audited the complete set on two balanced "
                "heldout cases. Set-level target-margin evidence does not make this member individually causal."
            ),
            "display_priority": max(0.01, abs(float(row["selection_score"]))),
            "source_artifacts": [
                "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/carrier_sets.jsonl",
                "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/matched_control_summary.jsonl",
                "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/cross_model_mechanism_summary.jsonl",
            ],
            "run_id": "phase330_nine_family_global_atlas",
            "single_unit_causal": False,
            "phase330_tested": True,
            "phase330_registered_set_support": supported,
            "phase330_cross_model_readout_specific": bool(global_result["cross_model_joint_readout_specific"]),
            "phase330_cross_model_natural_identity": bool(global_result["cross_model_natural_identity_positive"]),
            "phase330_cross_model_behavior_necessity": bool(global_result["cross_model_behavior_necessity_positive"]),
            "phase330_joint_minus_random_margin": local["joint_minus_random_margin"],
            "phase330_joint_minus_wrong_layer_margin": local["joint_minus_wrong_layer_margin"],
            "phase330_natural_minus_wrong_donor_margin": local["natural_minus_wrong_donor_margin"],
            "phase330_status": "set_readout_candidate_not_behavior_closed" if supported else "candidate_not_set_specific",
            "phase330_evidence_boundary": (
                "No Phase330 mechanism passed cross-model visible-behavior necessity. Single-unit CUDA gate remains closed."
            ),
        })
        anchor_id = f"phase330:{model}:{family}:{row['mechanism_id']}:L{row['component_layer']}:anchor"
        edges.append({
            "schema_version": "neuron_atlas_edge.v1",
            "edge_id": f"membership:{anchor_id}:{identifier}",
            "family_id": family,
            "mechanism_id": row["mechanism_id"],
            "model": model,
            "source_id": anchor_id,
            "target_id": identifier,
            "relation": "contains_phase330_frozen_component_candidate",
            "evidence_level": "L4_set_readout" if supported else "L3_candidate",
            "causal": False,
            "evidence_boundary": "Set-level intervention does not establish individual-member causality.",
        })
    return nodes, edges


def anchors_and_edges(
    model: str, family: str, registry: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_layer: dict[int, dict[str, Any]] = {}
    for row in registry:
        layer = int(row["component_layer"])
        item = by_layer.setdefault(layer, {
            "anchor_id": f"phase330:{model}:{family}:L{layer}:display-anchor",
            "layer": layer,
            "candidate_count": 0,
            "natural_overlap_count": 0,
            "group_supported_count": 0,
            "expanded_confirmation_count": 0,
            "mechanism_ids": set(),
            "evidence_level": "L2_frozen_path",
            "evidence_boundary": "Discovery-selected path peak; heldout prediction is conditioned on position role.",
        })
        item["mechanism_ids"].add(row["mechanism_id"])
    for node in nodes:
        item = by_layer.setdefault(int(node["layer"]), {
            "anchor_id": f"phase330:{model}:{family}:L{node['layer']}:display-anchor",
            "layer": int(node["layer"]),
            "candidate_count": 0,
            "natural_overlap_count": 0,
            "group_supported_count": 0,
            "expanded_confirmation_count": 0,
            "mechanism_ids": set(),
            "evidence_level": "L3_candidate",
            "evidence_boundary": "Frozen Phase330 component candidates.",
        })
        item["candidate_count"] += 1
        item["natural_overlap_count"] += int(node["natural_observed"])
        item["group_supported_count"] += int(node["group_intervention_supported"])
        item["mechanism_ids"].add(node["mechanism_id"])
    anchors = []
    for item in sorted(by_layer.values(), key=lambda row: row["layer"]):
        anchors.append({**item, "mechanism_ids": sorted(item["mechanism_ids"])})
    sequence = []
    ordered = [f"phase330:{model}:{family}:source", *[row["anchor_id"] for row in anchors], f"phase330:{model}:{family}:readout"]
    for index, (source, target) in enumerate(zip(ordered, ordered[1:])):
        sequence.append({
            "schema_version": "neuron_atlas_edge.v1",
            "edge_id": f"phase330:path:{model}:{family}:{index}",
            "family_id": family,
            "model": model,
            "source_id": source,
            "target_id": target,
            "relation": "observed_target_direction_sequence",
            "evidence_level": "L2_observational",
            "causal": False,
            "evidence_boundary": "Layer order is observational and does not prove information flow causality.",
        })
    return anchors, sequence


def build(output: Path = OUTPUT, public: Path = PUBLIC) -> dict[str, Any]:
    phase329.build(output, public)
    summary = read_json(SOURCE / "phase330_global_summary.json")
    carriers = read_jsonl(SOURCE / "carrier_sets.jsonl")
    path_registry = read_jsonl(SOURCE / "path_registry.jsonl")
    matched_rows = read_jsonl(SOURCE / "matched_control_summary.jsonl")
    cross_rows = read_jsonl(SOURCE / "cross_model_mechanism_summary.jsonl")
    family_summary = read_jsonl(SOURCE / "family_summary.jsonl")
    predictions = read_jsonl(SOURCE / "heldout_predictions.jsonl")
    claims = read_jsonl(SOURCE / "claim_registry.jsonl")
    matched = {(row["model"], row["family_id"], row["mechanism_id"]): row for row in matched_rows}
    cross = {(row["family_id"], row["mechanism_id"]): row for row in cross_rows}
    behavior = {(row["model"], row["family_id"]): row for row in family_summary}
    old_manifest = read_json(output / "manifest.json")
    old_refs = {(row["model"], row["family_id"]): row for row in old_manifest["partitions"]}
    snapshots = {}
    for ref in old_manifest["partitions"]:
        partition = read_json(output / ref["path"])
        snapshots.setdefault(ref["model"], partition["model_snapshot"])
    families_payload = read_json(output / "families.json")
    family_defs = {row["family_id"]: row for row in families_payload["families"]}
    all_refs = []
    all_nodes = []
    all_edges = []
    generated_at = datetime.now(timezone.utc).isoformat()
    for model in MODELS:
        for family, mechanisms in FAMILY_MECHANISMS.items():
            old_ref = old_refs.get((model, family))
            if old_ref:
                partition_path = output / old_ref["path"]
                partition = read_json(partition_path)
            else:
                partition_path = output / "partitions" / family / f"{model}.json"
                partition = {
                    "schema_version": "neuron_atlas_partition.v1",
                    "generated_at": generated_at,
                    "family": family_defs[family],
                    "model": model,
                    "model_snapshot": snapshots[model],
                    "scope": {"relations": [], "phase": 330, "source_phases": [330]},
                    "mapping_status": "distributed_component_set_candidates_not_single_unit_causal",
                    "evidence_boundary": "",
                    "path": {"source": {}, "layer_anchors": [], "readout": {}, "sequence_edges": []},
                    "nodes": [],
                    "membership_edges": [],
                    "metrics": {},
                    "source_artifacts": [],
                }
            model_carriers = [row for row in carriers if row["model"] == model and row["family_id"] == family]
            model_registry = [row for row in path_registry if row["model"] == model and row["family_id"] == family]
            new_nodes, new_membership = build_nodes(
                model, family, model_carriers, matched, cross, snapshots[model].get("model_revision", "local_unknown")
            )
            anchors, sequence = anchors_and_edges(model, family, model_registry, new_nodes)
            existing_ids = {row["node_id"] for row in partition.get("nodes", [])}
            partition["nodes"] = [*partition.get("nodes", []), *[row for row in new_nodes if row["node_id"] not in existing_ids]]
            partition["membership_edges"] = [*partition.get("membership_edges", []), *new_membership]
            partition["path"]["layer_anchors"] = [*partition["path"].get("layer_anchors", []), *anchors]
            partition["path"]["sequence_edges"] = [*partition["path"].get("sequence_edges", []), *sequence]
            partition["path"]["phase330_global_paths"] = model_registry
            metric = behavior[(model, family)]
            partition["path"]["source"] = {
                **partition["path"].get("source", {}),
                "token_position": "source/query/last",
                "phase330_registered_cases": 576,
            }
            partition["path"]["readout"] = {
                **partition["path"].get("readout", {}),
                "layer": snapshots[model]["num_hidden_layers"] - 1,
                "global_closed": False,
                "metrics": {
                    "candidate_winner_is_target": metric["candidate_winner_is_target"],
                    "target_in_top50": metric["target_in_top50"],
                    "target_match": metric["target_match"],
                    "behavior_success": metric["behavior_success"],
                },
            }
            model_matched = [row for row in matched_rows if row["model"] == model and row["family_id"] == family]
            model_cross = [row for row in cross_rows if row["family_id"] == family]
            prediction_rows = [row for row in predictions if row["model"] == model and row["family_id"] == family]
            partition["metrics"].update({
                "unique_unit_count": len({row["node_id"] for row in partition["nodes"]}),
                "component_set_member_count": sum(row.get("node_type") == "component_set_member" for row in partition["nodes"]),
                "candidate_layer_count": len({int(row["layer"]) for row in partition["nodes"]}),
                "natural_overlap_count": sum(bool(row.get("natural_observed")) for row in partition["nodes"]),
                "group_supported_candidate_count": sum(bool(row.get("group_intervention_supported")) for row in partition["nodes"]),
                "single_unit_causal_count": 0,
                "phase330_mechanism_count": len(mechanisms),
                "phase330_prompt_case_count": 576,
                "phase330_component_event_count": 576 * int(snapshots[model]["num_hidden_layers"]) * 9,
                "phase330_path_signature_count": 5184,
                "phase330_component_member_count": len(new_nodes),
                "phase330_registered_causal_case_count": 16,
                "phase330_registered_causal_row_count": 160,
                "phase330_local_set_readout_specific_mechanism_count": sum(row["joint_readout_specific_vs_both_controls"] for row in model_matched),
                "phase330_cross_model_set_readout_specific_mechanism_count": sum(row["cross_model_joint_readout_specific"] for row in model_cross),
                "phase330_cross_model_natural_identity_mechanism_count": sum(row["cross_model_natural_identity_positive"] for row in model_cross),
                "phase330_cross_model_behavior_necessity_mechanism_count": sum(row["cross_model_behavior_necessity_positive"] for row in model_cross),
                "phase330_heldout_peak_exact_rate": round(sum(row["exact_layer_match"] for row in prediction_rows) / len(prediction_rows), 7),
                "phase330_heldout_peak_10pct_rate": round(sum(row["within_10pct_depth"] for row in prediction_rows) / len(prediction_rows), 7),
                "single_unit_intervention_gate_open_count": 0,
            })
            partition["generated_at"] = generated_at
            partition["mapping_status"] = "phase330_nine_family_distributed_candidates_not_single_unit_causal"
            partition["scope"] = {
                **partition.get("scope", {}),
                "relations": sorted(set([*partition.get("scope", {}).get("relations", []), *mechanisms])),
                "phase": 330,
                "source_phases": sorted(set([*partition.get("scope", {}).get("source_phases", []), 330])),
            }
            partition["evidence_boundary"] = (
                "Phase330 maps all eight registered mechanisms in this family through full-layer attention/MLP/residual "
                "events and frozen head/group candidates. Only five of 72 mechanisms have cross-model set-level "
                "readout specificity; zero have cross-model visible-behavior necessity. Nodes are not single neurons."
            )
            partition["source_artifacts"] = list(dict.fromkeys([
                *partition.get("source_artifacts", []),
                "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/phase330_global_summary.json",
                "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/path_registry.jsonl",
                "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/carrier_sets.jsonl",
                "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/cross_model_mechanism_summary.jsonl",
            ]))
            write_json(partition_path, partition)
            ref = {
                "family_id": family,
                "model": model,
                "path": partition_path.relative_to(output).as_posix(),
                "mapping_status": partition["mapping_status"],
                **partition["metrics"],
            }
            all_refs.append(ref)
            all_nodes.extend(partition["nodes"])
            all_edges.extend(partition["membership_edges"])
            all_edges.extend(partition["path"]["sequence_edges"])

    write_jsonl(output / "neuron_nodes.jsonl", all_nodes)
    write_jsonl(output / "neuron_edges.jsonl", all_edges)
    write_jsonl(output / "phase330_paths.jsonl", path_registry)
    write_jsonl(output / "phase330_carrier_sets.jsonl", carriers)
    write_jsonl(output / "phase330_claim_registry.jsonl", claims)
    index = read_json(output / "neuron_index.json")
    index["generated_at"] = generated_at
    index["partitions"] = all_refs
    write_json(output / "neuron_index.json", index)

    for family in families_payload["families"]:
        refs = [row for row in all_refs if row["family_id"] == family["family_id"]]
        partitions = [read_json(output / row["path"]) for row in refs]
        family["physical_mapping"] = {
            "status": "phase330_mapped_distributed_candidates",
            "models": list(MODELS),
            "unique_unit_count": sum(row["metrics"]["unique_unit_count"] for row in partitions),
            "single_neuron_candidate_count": sum(row["metrics"].get("single_neuron_candidate_count", 0) for row in partitions),
            "component_set_member_count": sum(row["metrics"]["component_set_member_count"] for row in partitions),
            "phase330_component_member_count": 96,
            "phase330_registered_causal_case_count": 48,
            "single_unit_causal_count": 0,
            "partition_refs": [row["path"] for row in refs],
        }
    families_payload["generated_at"] = generated_at
    write_json(output / "families.json", families_payload)

    manifest = read_json(output / "manifest.json")
    manifest["phase"] = 330
    manifest["generated_at"] = generated_at
    manifest["partitions"] = all_refs
    manifest["metrics"].update({
        "mapped_family_count": 9,
        "model_family_partition_count": 27,
        "registered_mechanism_count": 72,
        "prompt_model_case_count": 15552,
        "top50_row_count": 777600,
        "component_event_count": 4852224,
        "path_signature_count": 139968,
        "phase330_component_candidate_count": 487296,
        "phase330_carrier_member_count": 864,
        "phase330_registered_causal_case_count": 432,
        "phase330_registered_causal_row_count": 4320,
        "phase330_cross_model_set_readout_specific_mechanism_count": summary["cross_model"]["joint_readout_specific_mechanisms"],
        "phase330_cross_model_natural_identity_mechanism_count": summary["cross_model"]["natural_identity_positive_mechanisms"],
        "phase330_cross_model_behavior_necessity_mechanism_count": summary["cross_model"]["behavior_necessity_mechanisms"],
        "phase330_heldout_peak_exact_rate": summary["heldout_prediction"]["exact_layer_match_rate"],
        "phase330_heldout_peak_10pct_rate": summary["heldout_prediction"]["within_10pct_depth_rate"],
        "unique_unit_count": len({row["node_id"] for row in all_nodes}),
        "component_set_member_count": sum(row.get("node_type") == "component_set_member" for row in all_nodes),
        "edge_count": len(all_edges),
        "single_unit_causal_count": 0,
        "single_unit_intervention_gate_open_count": 0,
    })
    manifest["evidence_boundary"] = {
        **manifest.get("evidence_boundary", {}),
        "statement": (
            "Phase330 completes the frozen nine-family observational denominator and balanced heldout set audit. "
            "Five of 72 mechanisms replicate set-level target-margin specificity across three small models; "
            "zero replicate visible-behavior necessity. No node is established as a causal single neuron."
        ),
        "all_nine_families_observationally_mapped": True,
        "cross_model_behavior_mechanism_closed": False,
        "single_unit_intervention_gate_open": False,
        "language_encoding_mechanism_closed": False,
    }
    manifest["source_artifacts"] = list(dict.fromkeys([
        *manifest.get("source_artifacts", []),
        "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/phase330_report.md",
        "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/phase330_global_summary.json",
        "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/claim_registry.jsonl",
    ]))
    write_json(output / "manifest.json", manifest)
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            checksums.append({"path": path.relative_to(output).as_posix(), "sha256": phase325.file_sha256(path)})
    write_json(output / "checksums.json", {"schema_version": "artifact_checksums.v1", "files": checksums})
    shutil.copytree(output, public, dirs_exist_ok=True)
    return manifest


def validate(output: Path = OUTPUT) -> dict[str, Any]:
    manifest = read_json(output / "manifest.json")
    families = read_json(output / "families.json")
    index = read_json(output / "neuron_index.json")
    nodes = read_jsonl(output / "neuron_nodes.jsonl")
    phase330_nodes = [row for row in nodes if row.get("phase330_tested")]
    assert manifest["phase"] == 330
    assert len(families["families"]) == 9
    assert len(index["partitions"]) == 27
    assert len(phase330_nodes) == 864
    assert all(not row["single_unit_causal"] for row in phase330_nodes)
    assert manifest["metrics"]["phase330_cross_model_set_readout_specific_mechanism_count"] == 5
    assert manifest["metrics"]["phase330_cross_model_behavior_necessity_mechanism_count"] == 0
    assert manifest["metrics"]["single_unit_causal_count"] == 0
    return {
        "family_count": 9,
        "partition_count": 27,
        "phase330_component_members": len(phase330_nodes),
        "all_nodes": len(nodes),
        "single_unit_causal": 0,
        "valid": True,
    }


if __name__ == "__main__":
    result = build()
    print(json.dumps({"metrics": result["metrics"], "validation": validate()}, ensure_ascii=False, indent=2))
