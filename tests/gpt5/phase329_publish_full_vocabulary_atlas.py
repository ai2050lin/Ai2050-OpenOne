#!/usr/bin/env python3
"""Publish Phase329 competition/mediation evidence onto the physical atlas."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase325_pattern_family_neuron_atlas as phase325  # noqa: E402
import phase328_publish_upstream_mediation_atlas as phase328  # noqa: E402


SOURCE = ROOT / "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation"
OUTPUT = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
PUBLIC = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = ("color_retrieval", "category_retrieval", "habitat_retrieval")


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


def build_paths(
    audits: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    condition_by_key = {
        (row["model"], row["mechanism_id"], row["condition"]): row
        for row in conditions
    }
    paths = []
    for audit in audits:
        key = (audit["model"], audit["mechanism_id"])
        correct = condition_by_key[(*key, "correct_baseline")]
        recipient = condition_by_key[(*key, "recipient_baseline")]
        tokenwise = condition_by_key[(*key, "recipient_tokenwise_correct")]
        pooled = condition_by_key[(*key, "recipient_pooled_correct")]
        paths.append({
            "schema_version": "pattern_family_full_vocabulary_path.v1",
            "phase_id": "Phase329",
            "path_id": f"phase329:{audit['model']}:{audit['mechanism_id']}",
            "model": audit["model"],
            "family_id": audit["family_id"],
            "mechanism_id": audit["mechanism_id"],
            "position_role": "query",
            "residual_observation_layer": audit["residual_observation_layer"],
            "intervention_input_layer": audit["intervention_input_layer"],
            "positive_residual_identity": audit["positive_identity_at_selection"],
            "tokenwise_beats_pooled": audit["tokenwise_beats_pooled"],
            "blocker_decline_pass": audit["blocker_decline_pass"],
            "carrier_member_mediation_pass": audit["carrier_member_mediation_pass"],
            "top1_unlock_pass": audit["top1_unlock_pass"],
            "generation_improvement_pass": audit["generation_improvement_pass"],
            "full_chain_candidate": audit["full_chain_candidate"],
            "single_unit_intervention_gate_open": audit["single_unit_intervention_gate_open"],
            "correct_baseline_target_top1_rate": correct["target_top1_rate"],
            "correct_baseline_target_top50_rate": correct["target_top50_rate"],
            "recipient_target_top1_rate": recipient["target_top1_rate"],
            "recipient_target_top50_rate": recipient["target_top50_rate"],
            "tokenwise_target_top1_rate": tokenwise["target_top1_rate"],
            "tokenwise_target_top50_rate": tokenwise["target_top50_rate"],
            "tokenwise_rank_gain": tokenwise["mean_global_target_rank_gain"],
            "pooled_rank_gain": pooled["mean_global_target_rank_gain"],
            "tokenwise_blocker_decline": tokenwise["mean_global_blocker_decline"],
            "tokenwise_top1_category_counts": tokenwise["top1_category_counts"],
            "surface_protocol_confounded": correct["target_top1_rate"] == 0.0,
            "causal": False,
            "single_unit_causal": False,
            "evidence_boundary": (
                "Registered distributed residual-to-carrier competition path. No mechanism passed "
                "cross-model carrier mediation, top-1 unlock, or generation improvement. Function and "
                "format tokens at answer onset are surface-protocol competitors, not established semantic blockers."
            ),
            "source_artifacts": [
                "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329_cross_model_summary.json",
                "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329A_condition_summaries.jsonl",
                "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329C_carrier_member_mediation_rows.jsonl",
                "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329D_full_generation_rows.jsonl",
            ],
        })
    return paths


def validate(output: Path) -> dict[str, int]:
    manifest = read_json(output / "manifest.json")
    nodes = read_jsonl(output / "neuron_nodes.jsonl")
    paths = read_jsonl(output / "phase329_full_vocabulary_paths.jsonl")
    assert manifest["phase"] == 329
    assert len(nodes) == 1121
    assert len(paths) == 9
    assert all(not row["causal"] and not row["single_unit_causal"] for row in paths)
    assert manifest["metrics"]["full_vocabulary_mediation_path_count"] == 9
    assert manifest["metrics"]["cross_model_blocker_decline_mechanism_count"] == 2
    assert manifest["metrics"]["cross_model_carrier_member_mediation_mechanism_count"] == 0
    assert manifest["metrics"]["cross_model_top1_unlock_mechanism_count"] == 0
    assert manifest["metrics"]["cross_model_full_chain_candidate_count"] == 0
    assert manifest["metrics"]["single_unit_causal_count"] == 0
    return {
        "nodes": len(nodes),
        "full_vocabulary_paths": len(paths),
        "cross_model_blocker_decline_mechanisms": 2,
        "cross_model_full_chain_candidates": 0,
        "single_unit_causal": 0,
    }


def build(output: Path = OUTPUT, public: Path = PUBLIC) -> dict[str, Any]:
    phase328.build(output, public)
    cross = read_json(SOURCE / "phase329_cross_model_summary.json")
    audits = read_jsonl(SOURCE / "phase329_model_mechanism_audits.jsonl")
    conditions = read_jsonl(SOURCE / "phase329A_condition_summaries.jsonl")
    paths = build_paths(audits, conditions)
    write_jsonl(SOURCE / "phase329_full_vocabulary_paths.jsonl", paths)
    write_jsonl(output / "phase329_full_vocabulary_paths.jsonl", paths)
    audit_by_key = {(row["model"], row["mechanism_id"]): row for row in audits}

    index = read_json(output / "neuron_index.json")
    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    for ref in index["partitions"]:
        path = output / ref["path"]
        partition = read_json(path)
        if ref["family_id"] == "content_knowledge":
            model_paths = [row for row in paths if row["model"] == ref["model"]]
            partition["path"]["full_vocabulary_mediation_paths"] = model_paths
            model_audits = [row for row in audits if row["model"] == ref["model"]]
            partition["metrics"].update({
                "full_vocabulary_mediation_path_count": len(model_paths),
                "positive_residual_identity_count": sum(
                    row["positive_identity_at_selection"] for row in model_audits
                ),
                "tokenwise_beats_pooled_count": sum(
                    row["tokenwise_beats_pooled"] for row in model_audits
                ),
                "blocker_decline_pass_count": sum(row["blocker_decline_pass"] for row in model_audits),
                "carrier_member_mediation_pass_count": sum(
                    row["carrier_member_mediation_pass"] for row in model_audits
                ),
                "top1_unlock_pass_count": sum(row["top1_unlock_pass"] for row in model_audits),
                "generation_improvement_pass_count": sum(
                    row["generation_improvement_pass"] for row in model_audits
                ),
                "full_vocabulary_chain_candidate_count": sum(
                    row["full_chain_candidate"] for row in model_audits
                ),
                "single_unit_intervention_gate_open_count": sum(
                    row["single_unit_intervention_gate_open"] for row in model_audits
                ),
            })
            for node in partition["nodes"]:
                if node.get("node_type") != "component_set_member":
                    continue
                audit = audit_by_key.get((ref["model"], node.get("mechanism_id")))
                if not audit:
                    node["phase329_tested"] = False
                    node["phase329_status"] = "not_in_registered_scope"
                    continue
                node.update({
                    "phase329_tested": True,
                    "phase329_residual_observation_layer": audit["residual_observation_layer"],
                    "phase329_intervention_input_layer": audit["intervention_input_layer"],
                    "phase329_positive_residual_identity": audit["positive_identity_at_selection"],
                    "phase329_tokenwise_beats_pooled": audit["tokenwise_beats_pooled"],
                    "phase329_blocker_decline_pass": audit["blocker_decline_pass"],
                    "phase329_carrier_member_mediation_pass": audit["carrier_member_mediation_pass"],
                    "phase329_top1_unlock_pass": audit["top1_unlock_pass"],
                    "phase329_generation_improvement_pass": audit["generation_improvement_pass"],
                    "phase329_full_chain_candidate": audit["full_chain_candidate"],
                    "phase329_single_unit_gate_open": audit["single_unit_intervention_gate_open"],
                    "phase329_status": (
                        "full_chain_candidate"
                        if audit["full_chain_candidate"]
                        else "registered_competition_path_not_closed"
                    ),
                    "phase329_evidence_boundary": audit["evidence_boundary"],
                })
                node["source_artifacts"] = list(dict.fromkeys([
                    *node.get("source_artifacts", []),
                    "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329_model_mechanism_audits.jsonl",
                    "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329_cross_model_summary.json",
                ]))
            partition["scope"] = {
                **partition.get("scope", {}),
                "phase": 329,
                "source_phases": sorted(set([
                    *partition.get("scope", {}).get("source_phases", []), 329
                ])),
            }
            partition["evidence_boundary"] = (
                "Phase329 maps registered full-vocabulary competition and tokenwise query paths onto "
                "the frozen component members. Category and habitat show cross-model blocker decline, "
                "but no carrier-member mediation, top-1 unlock, generation improvement, full chain, or single-unit gate."
            )
            partition["source_artifacts"] = list(dict.fromkeys([
                *partition.get("source_artifacts", []),
                "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329_cross_model_summary.json",
                "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329_full_vocabulary_paths.jsonl",
            ]))
        write_json(path, partition)
        ref.update(partition["metrics"])
        all_nodes.extend(partition["nodes"])
        all_edges.extend(partition["membership_edges"])
        all_edges.extend(partition["path"]["sequence_edges"])
    write_jsonl(output / "neuron_nodes.jsonl", all_nodes)
    write_jsonl(output / "neuron_edges.jsonl", all_edges)
    index["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "neuron_index.json", index)

    interventions = read_jsonl(output / "neuron_interventions.jsonl")
    interventions.extend({
        **row,
        "schema_version": "neuron_atlas_full_vocabulary_mediation.v1",
        "causal_scope": "distributed_residual_and_carrier_set_not_single_unit",
        "single_unit_causal": False,
    } for row in audits)
    write_jsonl(output / "neuron_interventions.jsonl", interventions)

    runs = read_jsonl(output / "neuron_runs.jsonl")
    for model in MODELS:
        model_audits = [row for row in audits if row["model"] == model]
        runs.append({
            "schema_version": "neuron_atlas_run.v1",
            "run_id": f"phase329_full_vocabulary_{model}",
            "model": model,
            "family_id": "content_knowledge",
            "status": "complete",
            "source_phase": 329,
            "full_vocabulary_path_count": len(model_audits),
            "blocker_decline_pass_count": sum(row["blocker_decline_pass"] for row in model_audits),
            "carrier_member_mediation_pass_count": sum(
                row["carrier_member_mediation_pass"] for row in model_audits
            ),
            "top1_unlock_pass_count": sum(row["top1_unlock_pass"] for row in model_audits),
            "full_chain_candidate_count": sum(row["full_chain_candidate"] for row in model_audits),
            "single_unit_causal_count": 0,
        })
    write_jsonl(output / "neuron_runs.jsonl", runs)

    manifest = read_json(output / "manifest.json")
    manifest["phase"] = 329
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["partitions"] = index["partitions"]
    manifest["metrics"].update({
        "full_vocabulary_mediation_path_count": len(paths),
        "cross_model_tokenwise_beats_pooled_mechanism_count": cross[
            "cross_model_tokenwise_beats_pooled_mechanism_count"
        ],
        "cross_model_blocker_decline_mechanism_count": cross[
            "cross_model_blocker_decline_mechanism_count"
        ],
        "cross_model_carrier_member_mediation_mechanism_count": cross[
            "cross_model_carrier_member_mediation_mechanism_count"
        ],
        "cross_model_top1_unlock_mechanism_count": cross[
            "cross_model_top1_unlock_mechanism_count"
        ],
        "cross_model_generation_improvement_mechanism_count": cross[
            "cross_model_generation_improvement_mechanism_count"
        ],
        "cross_model_full_chain_candidate_count": cross[
            "cross_model_full_chain_candidate_count"
        ],
        "single_unit_intervention_gate_open_count": int(
            cross["single_unit_intervention_gate_open"]
        ),
        "single_unit_causal_count": 0,
    })
    manifest["evidence_boundary"].update({
        "statement": (
            "Phase329 adds full-vocabulary rank/blocker, tokenwise query, carrier-member, and natural "
            "generation evidence. Category and habitat replicate blocker decline in two models, but no "
            "mechanism replicates carrier mediation, top-1 unlock, generation improvement, or a full chain."
        ),
        "full_vocabulary_chain_closure": False,
        "single_unit_intervention_gate_open": False,
        "surface_protocol_blockers_are_semantic_blockers": False,
    })
    manifest["source_artifacts"] = list(dict.fromkeys([
        *manifest["source_artifacts"],
        "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329_cross_model_summary.json",
        "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329A_condition_summaries.jsonl",
        "tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329_full_vocabulary_paths.jsonl",
    ]))
    write_json(output / "manifest.json", manifest)
    manifest["validation"] = {"status": "passed", **validate(output)}
    write_json(output / "manifest.json", manifest)

    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            checksums.append({
                "path": path.relative_to(output).as_posix(),
                "sha256": phase325.file_sha256(path),
            })
    write_json(output / "checksums.json", {"schema_version": "artifact_checksums.v1", "files": checksums})
    if public.exists():
        shutil.rmtree(public)
    shutil.copytree(output, public)
    return manifest


if __name__ == "__main__":
    result = build()
    print(json.dumps({"status": "ok", "metrics": result["metrics"]}, ensure_ascii=False, indent=2))
