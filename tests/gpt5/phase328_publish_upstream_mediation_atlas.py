#!/usr/bin/env python3
"""Publish Phase328 upstream residual mediation without adding synthetic neurons."""

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
import phase327_publish_natural_retrieval_atlas as phase327  # noqa: E402


SOURCE = ROOT / "tests/gpt5/result/phase328_upstream_residual_mediation/upstream_residual_mediation"
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


def validate(output: Path) -> dict[str, int]:
    manifest = read_json(output / "manifest.json")
    nodes = read_jsonl(output / "neuron_nodes.jsonl")
    edges = read_jsonl(SOURCE / "phase328_atlas_edges.jsonl")
    assert manifest["phase"] == 328
    assert len(nodes) == 1121
    assert manifest["metrics"]["upstream_residual_mediation_edge_count"] == 3
    assert manifest["metrics"]["cross_model_causal_path_edge_count"] == 0
    assert manifest["metrics"]["single_unit_causal_count"] == 0
    assert len(edges) == 3 and all(not row["causal"] for row in edges)
    return {
        "nodes": len(nodes),
        "upstream_residual_edges": len(edges),
        "causal_path_edges": 0,
        "single_unit_causal": 0,
    }


def build(output: Path = OUTPUT, public: Path = PUBLIC) -> dict[str, Any]:
    phase327.build(output, public)
    summary = read_json(SOURCE / "phase328_cross_model_summary.json")
    audits = read_jsonl(SOURCE / "phase328_model_audits.jsonl")
    edges = read_jsonl(SOURCE / "phase328_atlas_edges.jsonl")
    audit_by_model = {row["model"]: row for row in audits}
    edge_by_model = {row["model"]: row for row in edges}
    index = read_json(output / "neuron_index.json")
    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    for ref in index["partitions"]:
        path = output / ref["path"]
        partition = read_json(path)
        if ref["family_id"] == "content_knowledge":
            audit = audit_by_model[ref["model"]]
            edge = edge_by_model[ref["model"]]
            partition["path"]["upstream_residual_mediation_edges"] = [edge]
            partition["metrics"].update({
                "upstream_residual_mediation_edge_count": 1,
                "upstream_residual_mediation_pass_count": int(audit["upstream_mediation_pass"]),
                "natural_generation_unlock_pass_count": int(audit["natural_generation_unlock_pass"]),
                "causal_path_edge_count": int(edge["causal"]),
            })
            for node in partition["nodes"]:
                if node.get("node_type") != "component_set_member" or node.get("mechanism_id") != "category_retrieval":
                    continue
                node.update({
                    "phase328_selected_residual_layer": audit["selected_residual_layer"],
                    "phase328_residual_position_role": audit["position_role"],
                    "phase328_upstream_mediation_pass": audit["upstream_mediation_pass"],
                    "phase328_natural_generation_unlock_pass": audit["natural_generation_unlock_pass"],
                    "phase328_causal_edge": edge["causal"],
                    "phase328_evidence_boundary": edge["evidence_boundary"],
                })
                node["source_artifacts"] = list(dict.fromkeys([
                    *node.get("source_artifacts", []),
                    "tests/gpt5/result/phase328_upstream_residual_mediation/upstream_residual_mediation/phase328_cross_model_summary.json",
                    "tests/gpt5/result/phase328_upstream_residual_mediation/upstream_residual_mediation/phase328_atlas_edges.jsonl",
                ]))
            partition["scope"] = {
                **partition.get("scope", {}),
                "phase": 328,
                "source_phases": sorted(set([*partition.get("scope", {}).get("source_phases", []), 328])),
            }
            partition["evidence_boundary"] = (
                "Phase328 tests pooled query-residual mediation into the frozen carrier set. "
                "Only GLM4 passed mediation, no model unlocked natural top-1, and every path edge remains noncausal."
            )
            partition["source_artifacts"] = list(dict.fromkeys([
                *partition.get("source_artifacts", []),
                "tests/gpt5/result/phase328_upstream_residual_mediation/upstream_residual_mediation/phase328_cross_model_summary.json",
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
        "schema_version": "neuron_atlas_upstream_residual_mediation.v1",
        "causal_scope": "pooled_residual_state_to_distributed_component_set",
        "single_unit_causal": False,
    } for row in audits)
    write_jsonl(output / "neuron_interventions.jsonl", interventions)

    runs = read_jsonl(output / "neuron_runs.jsonl")
    for audit in audits:
        runs.append({
            "schema_version": "neuron_atlas_run.v1",
            "run_id": f"phase328_upstream_residual_{audit['model']}",
            "model": audit["model"],
            "family_id": "content_knowledge",
            "status": "complete",
            "source_phase": 328,
            "mechanism_id": "category_retrieval",
            "upstream_mediation_pass": audit["upstream_mediation_pass"],
            "natural_generation_unlock_pass": audit["natural_generation_unlock_pass"],
            "causal_path_edge_count": 0,
            "single_unit_causal_count": 0,
        })
    write_jsonl(output / "neuron_runs.jsonl", runs)

    manifest = read_json(output / "manifest.json")
    manifest["phase"] = 328
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["partitions"] = index["partitions"]
    manifest["metrics"].update({
        "upstream_residual_mediation_edge_count": len(edges),
        "upstream_residual_mediation_pass_model_count": len(summary["upstream_mediation_pass_models"]),
        "natural_generation_unlock_pass_model_count": len(summary["natural_generation_unlock_pass_models"]),
        "cross_model_causal_path_edge_count": int(summary["cross_model_causal_edge_replicated"]),
        "single_unit_causal_count": 0,
    })
    manifest["evidence_boundary"].update({
        "statement": (
            "Phase328 adds pooled query-residual mediation candidates for category retrieval. "
            "Mediation passed only in GLM4, natural top-1 unlock passed in no model, so all path edges remain noncausal."
        ),
        "upstream_residual_causal_closure": False,
    })
    manifest["source_artifacts"] = list(dict.fromkeys([
        *manifest["source_artifacts"],
        "tests/gpt5/result/phase328_upstream_residual_mediation/upstream_residual_mediation/phase328_cross_model_summary.json",
        "tests/gpt5/result/phase328_upstream_residual_mediation/upstream_residual_mediation/phase328_atlas_edges.jsonl",
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
