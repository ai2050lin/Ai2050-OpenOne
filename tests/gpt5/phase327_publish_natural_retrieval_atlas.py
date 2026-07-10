#!/usr/bin/env python3
"""Publish Phase327 natural retrieval evidence onto the existing physical atlas."""

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
import phase326_publish_physical_path_atlas as phase326  # noqa: E402


SOURCE = ROOT / "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path"
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
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def audit_status(audit: dict[str, Any]) -> str:
    stages = []
    for field, label in (
        ("natural_gate_observational_pass", "natural_identity"),
        ("position_necessity_pass", "position_necessity"),
        ("natural_state_transplant_pass", "natural_transplant"),
        ("complete_generation_pass", "generation"),
    ):
        if audit.get(field):
            stages.append(label)
    return "+".join(stages) if stages else "phase327_no_registered_stage_pass"


def annotate_partition(
    partition: dict[str, Any],
    model: str,
    audits: dict[tuple[str, str], dict[str, Any]],
    paths: list[dict[str, Any]],
) -> dict[str, Any]:
    model_paths = [row for row in paths if row["model"] == model]
    for node in partition["nodes"]:
        if node.get("node_type") != "component_set_member":
            continue
        audit = audits.get((model, node.get("mechanism_id")))
        if not audit:
            node.update({
                "phase327_natural_gate_observational_pass": False,
                "phase327_position_necessity_pass": False,
                "phase327_natural_state_transplant_pass": False,
                "phase327_complete_generation_pass": False,
                "phase327_full_chain_pass": False,
                "phase327_status": "phase327_not_in_registered_scope",
                "phase327_evidence_boundary": (
                    "This mechanism was not in the Phase327 registered scope; false fields mean not tested, not a negative result."
                ),
            })
            continue
        node.update({
            "phase327_natural_gate_observational_pass": bool(audit["natural_gate_observational_pass"]),
            "phase327_position_necessity_pass": bool(audit["position_necessity_pass"]),
            "phase327_natural_state_transplant_pass": bool(audit["natural_state_transplant_pass"]),
            "phase327_complete_generation_pass": bool(audit["complete_generation_pass"]),
            "phase327_full_chain_pass": bool(audit["full_chain_pass"]),
            "phase327_status": audit_status(audit),
            "phase327_evidence_boundary": (
                "Mechanism-level frozen-set evidence; it does not make this member individually causal."
            ),
        })
        node["source_artifacts"] = list(dict.fromkeys([
            *node.get("source_artifacts", []),
            "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path/phase327_mechanism_audits.jsonl",
            "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path/phase327_atlas_paths.jsonl",
        ]))
    partition["scope"] = {
        **partition.get("scope", {}),
        "phase": 327,
        "source_phases": sorted(set([*partition.get("scope", {}).get("source_phases", []), 327])),
    }
    partition["path"]["natural_retrieval_paths"] = model_paths
    partition["metrics"].update({
        "natural_retrieval_path_count": len(model_paths),
        "natural_gate_pass_count": sum(row["natural_gate_observational_pass"] for row in model_paths),
        "position_necessity_pass_count": sum(row["position_necessity_pass"] for row in model_paths),
        "natural_state_transplant_pass_count": sum(row["natural_state_transplant_pass"] for row in model_paths),
        "complete_generation_pass_count": sum(row["complete_generation_pass"] for row in model_paths),
        "full_natural_chain_pass_count": sum(row["full_chain_pass"] for row in model_paths),
    })
    partition["evidence_boundary"] = (
        "Phase327 links natural identity observations, position-separated set necessity, natural-state transplantation, "
        "and generation outcomes to the frozen Phase326 members. Paths remain noncausal and single-unit causality is zero."
    )
    partition["source_artifacts"] = list(dict.fromkeys([
        *partition.get("source_artifacts", []),
        "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path/phase327_cross_model_summary.json",
        "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path/phase327_atlas_paths.jsonl",
    ]))
    return partition


def validate(output: Path) -> dict[str, int]:
    manifest = read_json(output / "manifest.json")
    nodes = read_jsonl(output / "neuron_nodes.jsonl")
    paths = read_jsonl(SOURCE / "phase327_atlas_paths.jsonl")
    assert manifest["phase"] == 327
    assert manifest["metrics"]["unique_unit_count"] == 1121
    assert manifest["metrics"]["component_set_member_count"] == 288
    assert manifest["metrics"]["natural_retrieval_path_count"] == 9
    assert manifest["metrics"]["full_natural_chain_pass_count"] == 0
    assert manifest["metrics"]["single_unit_causal_count"] == 0
    assert len(nodes) == 1121
    assert len(paths) == 9 and all(not row["causal"] for row in paths)
    return {
        "nodes": len(nodes),
        "natural_paths": len(paths),
        "full_chains": 0,
        "single_unit_causal": 0,
    }


def build(output: Path = OUTPUT, public: Path = PUBLIC) -> dict[str, Any]:
    phase326.build(output, public)
    paths = read_jsonl(SOURCE / "phase327_atlas_paths.jsonl")
    audit_rows = read_jsonl(SOURCE / "phase327_mechanism_audits.jsonl")
    cross_model = read_json(SOURCE / "phase327_cross_model_summary.json")
    audits = {(row["model"], row["mechanism_id"]): row for row in audit_rows}

    partition_index = read_json(output / "neuron_index.json")
    partition_refs = partition_index["partitions"]
    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    for ref in partition_refs:
        path = output / ref["path"]
        partition = read_json(path)
        if ref["family_id"] == "content_knowledge":
            partition = annotate_partition(partition, ref["model"], audits, paths)
        write_json(path, partition)
        all_nodes.extend(partition["nodes"])
        all_edges.extend(partition["membership_edges"])
        all_edges.extend(partition["path"]["sequence_edges"])
        ref.update(partition["metrics"])
    write_jsonl(output / "neuron_nodes.jsonl", all_nodes)
    write_jsonl(output / "neuron_edges.jsonl", all_edges)
    partition_index["generated_at"] = datetime.now(timezone.utc).isoformat()
    partition_index["partitions"] = partition_refs
    write_json(output / "neuron_index.json", partition_index)

    interventions = read_jsonl(output / "neuron_interventions.jsonl")
    interventions.extend({
        **row,
        "schema_version": "neuron_atlas_natural_retrieval_audit.v1",
        "causal_scope": "distributed_component_set_not_single_unit",
        "single_unit_causal": False,
    } for row in audit_rows)
    write_jsonl(output / "neuron_interventions.jsonl", interventions)

    runs = read_jsonl(output / "neuron_runs.jsonl")
    for model in MODELS:
        model_audits = [row for row in audit_rows if row["model"] == model]
        runs.append({
            "schema_version": "neuron_atlas_run.v1",
            "run_id": f"phase327_natural_retrieval_{model}",
            "model": model,
            "family_id": "content_knowledge",
            "status": "complete",
            "source_phase": 327,
            "natural_retrieval_path_count": len(model_audits),
            "full_natural_chain_pass_count": sum(row["full_chain_pass"] for row in model_audits),
            "single_unit_causal_count": 0,
        })
    write_jsonl(output / "neuron_runs.jsonl", runs)

    manifest = read_json(output / "manifest.json")
    manifest["phase"] = 327
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["partitions"] = partition_refs
    manifest["metrics"].update({
        "natural_retrieval_path_count": len(paths),
        "cross_model_natural_gate_mechanism_count": sum(
            len(row["natural_gate_pass_models"]) >= 2 for row in cross_model["mechanism_results"]
        ),
        "cross_model_position_necessity_mechanism_count": sum(
            len(row["position_necessity_pass_models"]) >= 2 for row in cross_model["mechanism_results"]
        ),
        "cross_model_natural_transplant_mechanism_count": sum(
            len(row["natural_state_transplant_pass_models"]) >= 2 for row in cross_model["mechanism_results"]
        ),
        "cross_model_complete_generation_mechanism_count": sum(
            len(row["complete_generation_pass_models"]) >= 2 for row in cross_model["mechanism_results"]
        ),
        "full_natural_chain_pass_count": cross_model["cross_model_full_chain_count"],
        "single_unit_causal_count": 0,
    })
    manifest["evidence_boundary"].update({
        "statement": (
            "Phase327 adds registered natural retrieval-stage evidence to existing Phase326 carrier members. "
            "No full chain replicated across models; paths and member links remain noncausal."
        ),
        "natural_sufficiency_closure": False,
        "full_natural_chain_closure": False,
    })
    manifest["source_artifacts"] = list(dict.fromkeys([
        *manifest["source_artifacts"],
        "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path/phase327_cross_model_summary.json",
        "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path/phase327_atlas_paths.jsonl",
        "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path/phase327_residual_layer_summaries.jsonl",
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
