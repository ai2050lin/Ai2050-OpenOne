#!/usr/bin/env python3
"""Build the evidence-scoped pattern-family neuron atlas used by the 3D client."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PATTERN_ATLAS = ROOT / "tests/result/pattern_family_atlas/v2"
RESEARCH_KERNEL = ROOT / "tests/result/research_kernel"
OUTPUT_DIR = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
PUBLIC_DIR = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(row.get("layer", -1)),
        str(row.get("unit_kind") or "unknown_unit"),
        int(row.get("unit_index", -1)),
    )


def finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def aggregate_unit(
    model: str,
    family: dict[str, Any],
    rows: list[dict[str, Any]],
    natural_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    representative = max(rows, key=lambda row: float(row.get("candidate_score") or 0.0))
    natural = max(natural_rows, key=lambda row: float(row.get("activation_abs") or 0.0), default=None)
    scores = finite_values(rows, "candidate_score")
    activations = finite_values(rows, "activation_abs")
    readout = finite_values(rows, "readout_contribution")
    margins = finite_values(rows, "causal_margin_delta")
    group_supported = any(bool(row.get("causal_supported")) for row in rows)
    natural_observed = natural is not None
    case_ids = sorted({str(row.get("case_id")) for row in rows if row.get("case_id")})
    target_labels = sorted({str(row.get("target_label")) for row in rows if row.get("target_label")})
    source_artifacts = sorted({str(row.get("source_artifact")) for row in rows if row.get("source_artifact")})
    layer, kind, index = unit_key(representative)
    max_score = max(scores, default=0.0)
    display_priority = (
        max_score
        + (0.12 if natural_observed else 0.0)
        + (0.08 if group_supported else 0.0)
        + min(len(case_ids), 20) * 0.004
    )
    if group_supported:
        evidence_status = "component_candidate_with_group_support"
    elif natural_observed:
        evidence_status = "component_candidate_naturally_observed"
    else:
        evidence_status = "component_candidate"
    return {
        "schema_version": "neuron_atlas_node.v1",
        "node_id": f"{family['family_id']}:{model}:L{layer}:{kind}:{index}",
        "node_type": "unit_candidate",
        "family_id": family["family_id"],
        "family_name": family["family_name"],
        "relation": str(representative.get("relation") or "unknown"),
        "model": model,
        "model_revision": representative.get("model_revision"),
        "layer": layer,
        "component": representative.get("component") or "mlp",
        "unit_kind": kind,
        "unit_index": index,
        "token_position": representative.get("token_position"),
        "candidate_score": max_score,
        "candidate_score_mean": fmean(scores) if scores else 0.0,
        "activation_abs_mean": fmean(activations) if activations else None,
        "readout_contribution_max_abs": max((abs(value) for value in readout), default=None),
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "target_labels": target_labels,
        "coverage_objects": max((int(row.get("coverage_objects") or 0) for row in rows), default=0),
        "coverage_templates": max((int(row.get("coverage_templates") or 0) for row in rows), default=0),
        "natural_observed": natural_observed,
        "natural_activation": natural.get("activation") if natural else None,
        "natural_case_id": natural.get("case_id") if natural else None,
        "group_intervention_supported": group_supported,
        "group_margin_delta_min": min(margins, default=None),
        "causal_scope": "channel_group_not_single_unit" if margins else "not_tested",
        "evidence_level": "L4",
        "evidence_status": evidence_status,
        "evidence_boundary": "real unit candidate with readout attribution; group intervention does not establish single-unit causality",
        "display_priority": display_priority,
        "source_artifacts": source_artifacts[:8],
        "run_id": representative.get("run_id"),
    }


def build_partition(family: dict[str, Any], model: str) -> dict[str, Any]:
    stable_run = RESEARCH_KERNEL / "runs" / f"phase286_color_real_units_{model}"
    natural_run = RESEARCH_KERNEL / "runs" / f"phase287_{model}_red_component_trace"
    snapshot = read_json(stable_run / "model_snapshot.json")
    stable_rows = [row for row in read_jsonl(stable_run / "unit_evidence.jsonl") if row.get("family_id") == family["family_id"]]
    natural_rows = [row for row in read_jsonl(natural_run / "unit_evidence.jsonl") if row.get("family_id") == family["family_id"]]
    trace_rows = read_jsonl(stable_run / "trace_events.jsonl")
    intervention_rows = read_jsonl(stable_run / "intervention_rows.jsonl")

    stable_groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    natural_groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in stable_rows:
        stable_groups[unit_key(row)].append(row)
    for row in natural_rows:
        natural_groups[unit_key(row)].append(row)

    nodes = [
        aggregate_unit(model, family, rows, natural_groups.get(key, []))
        for key, rows in stable_groups.items()
    ]
    nodes.sort(key=lambda row: (-float(row["display_priority"]), int(row["layer"]), int(row["unit_index"])))

    trace_by_layer: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in trace_rows:
        trace_by_layer[int(event.get("layer", -1))][str(event.get("event_type"))] = event

    nodes_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        nodes_by_layer[int(node["layer"])].append(node)

    source_event = next((event for event in trace_rows if event.get("event_type") == "embedding"), None)
    readout_event = next((event for event in reversed(trace_rows) if event.get("event_type") == "unembedding_readout"), None)
    layer_anchors = []
    for layer in sorted(nodes_by_layer):
        events = trace_by_layer.get(layer, {})
        candidate_nodes = nodes_by_layer[layer]
        layer_anchors.append({
            "anchor_id": f"{family['family_id']}:{model}:L{layer}:anchor",
            "layer": layer,
            "candidate_count": len(candidate_nodes),
            "natural_overlap_count": sum(bool(node["natural_observed"]) for node in candidate_nodes),
            "group_supported_count": sum(bool(node["group_intervention_supported"]) for node in candidate_nodes),
            "attention_metrics": (events.get("attention_output") or {}).get("metrics"),
            "mlp_metrics": (events.get("mlp_product_write") or {}).get("metrics"),
            "residual_metrics": (events.get("residual_update") or {}).get("metrics"),
            "evidence_level": "L2+L4",
            "evidence_boundary": "observed component sequence plus localized unit candidates; not a neuron-to-neuron causal edge",
        })

    source_id = f"{family['family_id']}:{model}:source"
    readout_id = f"{family['family_id']}:{model}:readout"
    sequence_ids = [source_id, *[anchor["anchor_id"] for anchor in layer_anchors], readout_id]
    sequence_edges = [
        {
            "schema_version": "neuron_atlas_edge.v1",
            "edge_id": f"sequence:{source}:{target}",
            "family_id": family["family_id"],
            "model": model,
            "source_id": source,
            "target_id": target,
            "relation": "observed_component_sequence",
            "evidence_level": "L2",
            "causal": False,
            "evidence_boundary": "ordered events in one integrated trace; not a causal edge",
        }
        for source, target in zip(sequence_ids, sequence_ids[1:])
    ]
    membership_edges = [
        {
            "schema_version": "neuron_atlas_edge.v1",
            "edge_id": f"membership:{family['family_id']}:{model}:L{node['layer']}:{node['unit_index']}",
            "family_id": family["family_id"],
            "model": model,
            "source_id": f"{family['family_id']}:{model}:L{node['layer']}:anchor",
            "target_id": node["node_id"],
            "relation": "contains_localized_candidate",
            "evidence_level": "L4",
            "causal": False,
            "evidence_boundary": node["evidence_boundary"],
        }
        for node in nodes
    ]

    source_metrics = (source_event or {}).get("metrics") or {}
    readout_metrics = (readout_event or {}).get("metrics") or {}
    path = {
        "source": {
            "anchor_id": source_id,
            "stage": "prompt_embedding",
            "token": (source_event or {}).get("token"),
            "token_position": (source_event or {}).get("token_position"),
            "metrics": source_metrics,
            "evidence_level": "L2",
        },
        "layer_anchors": layer_anchors,
        "readout": {
            "anchor_id": readout_id,
            "stage": "unembedding_readout",
            "layer": (readout_event or {}).get("layer"),
            "metrics": readout_metrics,
            "evidence_level": "L2",
            "global_closed": bool(readout_metrics.get("global_closed")),
        },
        "sequence_edges": sequence_edges,
    }

    return {
        "schema_version": "neuron_atlas_partition.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "model": model,
        "model_snapshot": snapshot,
        "scope": {"relation": "color", "phase": 325, "source_phases": [286, 287]},
        "mapping_status": "component_path_with_unit_candidates_not_single_unit_causal",
        "evidence_boundary": "Phase286/287 support physical addresses and component/group evidence, not single-unit mechanism closure",
        "path": path,
        "nodes": nodes,
        "membership_edges": membership_edges,
        "metrics": {
            "source_row_count": len(stable_rows),
            "unique_unit_count": len(nodes),
            "natural_overlap_count": sum(bool(node["natural_observed"]) for node in nodes),
            "group_supported_candidate_count": sum(bool(node["group_intervention_supported"]) for node in nodes),
            "candidate_layer_count": len(layer_anchors),
            "trace_event_count": len(trace_rows),
            "group_intervention_row_count": len(intervention_rows),
            "single_unit_causal_count": 0,
        },
        "source_artifacts": [
            str(stable_run.relative_to(ROOT) / "unit_evidence.jsonl"),
            str(stable_run.relative_to(ROOT) / "trace_events.jsonl"),
            str(stable_run.relative_to(ROOT) / "intervention_rows.jsonl"),
            str(natural_run.relative_to(ROOT) / "unit_evidence.jsonl"),
        ],
    }


def validate_bundle(bundle_dir: Path) -> dict[str, int]:
    manifest = read_json(bundle_dir / "manifest.json")
    families = read_json(bundle_dir / "families.json")["families"]
    family_ids = {family["family_id"] for family in families}
    assert manifest["schema_version"] == "pattern_family_neuron_atlas.v1"
    assert len(family_ids) == manifest["metrics"]["family_count"]
    assert manifest["metrics"]["mapped_family_count"] == 1
    node_count = 0
    edge_count = 0
    for item in manifest["partitions"]:
        partition = read_json(bundle_dir / item["path"])
        assert partition["family"]["family_id"] in family_ids
        assert partition["model"] in MODELS
        assert partition["metrics"]["single_unit_causal_count"] == 0
        ids = [node["node_id"] for node in partition["nodes"]]
        assert len(ids) == len(set(ids))
        assert all(node["family_id"] == partition["family"]["family_id"] for node in partition["nodes"])
        assert all(edge["causal"] is False for edge in partition["membership_edges"])
        assert all(edge["causal"] is False for edge in partition["path"]["sequence_edges"])
        node_count += len(ids)
        edge_count += len(partition["membership_edges"]) + len(partition["path"]["sequence_edges"])
    assert node_count == manifest["metrics"]["unique_unit_count"]
    assert edge_count == manifest["metrics"]["edge_count"]
    return {"families": len(family_ids), "nodes": node_count, "edges": edge_count}


def build_bundle(output_dir: Path, public_dir: Path) -> dict[str, Any]:
    families = read_jsonl(PATTERN_ATLAS / "families.jsonl")
    family_by_id = {family["family_id"]: family for family in families}
    mapped_family = family_by_id["content_knowledge"]
    generated_at = datetime.now(timezone.utc).isoformat()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partitions = []
    all_nodes = []
    all_edges = []
    all_events = []
    all_interventions = []
    run_rows = []

    for model in MODELS:
        partition = build_partition(mapped_family, model)
        relative = Path("partitions") / mapped_family["family_id"] / f"{model}.json"
        write_json(output_dir / relative, partition)
        metrics = partition["metrics"]
        partitions.append({
            "family_id": mapped_family["family_id"],
            "model": model,
            "path": relative.as_posix(),
            "mapping_status": partition["mapping_status"],
            **metrics,
        })
        all_nodes.extend(partition["nodes"])
        all_edges.extend(partition["membership_edges"])
        all_edges.extend(partition["path"]["sequence_edges"])

        stable_run = RESEARCH_KERNEL / "runs" / f"phase286_color_real_units_{model}"
        for event in read_jsonl(stable_run / "trace_events.jsonl"):
            all_events.append({
                **event,
                "family_id": mapped_family["family_id"],
                "evidence_level": "L2",
                "causal": False,
            })
        for intervention in read_jsonl(stable_run / "intervention_rows.jsonl"):
            all_interventions.append({
                **intervention,
                "schema_version": "neuron_atlas_group_intervention.v1",
                "family_id": mapped_family["family_id"],
                "causal_scope": "channel_group_not_single_unit",
                "single_unit_causal": False,
            })
        run_rows.append({
            "schema_version": "neuron_atlas_run.v1",
            "run_id": f"phase286_color_real_units_{model}",
            "model": model,
            "family_id": mapped_family["family_id"],
            "status": "complete",
            "source_phase": 286,
            "unit_count": metrics["unique_unit_count"],
            "trace_event_count": metrics["trace_event_count"],
            "group_intervention_row_count": metrics["group_intervention_row_count"],
            "single_unit_causal_count": 0,
        })

    enriched_families = []
    for family in families:
        family_partitions = [item for item in partitions if item["family_id"] == family["family_id"]]
        enriched_families.append({
            **family,
            "physical_mapping": {
                "status": "mapped_candidate_path" if family_partitions else "not_mapped_to_real_units",
                "models": [item["model"] for item in family_partitions],
                "unique_unit_count": sum(item["unique_unit_count"] for item in family_partitions),
                "single_unit_causal_count": 0,
                "partition_refs": [item["path"] for item in family_partitions],
            },
        })

    write_json(output_dir / "families.json", {
        "schema_version": "neuron_atlas_family_index.v1",
        "generated_at": generated_at,
        "families": enriched_families,
    })
    write_jsonl(output_dir / "neuron_nodes.jsonl", all_nodes)
    write_jsonl(output_dir / "neuron_edges.jsonl", all_edges)
    write_jsonl(output_dir / "neuron_events.jsonl", all_events)
    write_jsonl(output_dir / "neuron_interventions.jsonl", all_interventions)
    write_jsonl(output_dir / "neuron_runs.jsonl", run_rows)

    index_payload = {
        "schema_version": "neuron_atlas_index.v1",
        "generated_at": generated_at,
        "partitions": partitions,
        "selection_rule": "load only the selected family and model partition",
        "default_family_id": "content_knowledge",
        "default_model": "qwen3",
    }
    write_json(output_dir / "neuron_index.json", index_payload)

    manifest = {
        "schema_version": "pattern_family_neuron_atlas.v1",
        "phase": 325,
        "generated_at": generated_at,
        "title": "Evidence-scoped pattern-family neuron path atlas",
        "families_path": "families.json",
        "index_path": "neuron_index.json",
        "partitions": partitions,
        "files": {
            "nodes": "neuron_nodes.jsonl",
            "edges": "neuron_edges.jsonl",
            "events": "neuron_events.jsonl",
            "interventions": "neuron_interventions.jsonl",
            "runs": "neuron_runs.jsonl",
        },
        "metrics": {
            "family_count": len(families),
            "mapped_family_count": sum(bool(family["physical_mapping"]["models"]) for family in enriched_families),
            "model_count": len(MODELS),
            "unique_unit_count": len(all_nodes),
            "edge_count": len(all_edges),
            "natural_event_count": len(all_events),
            "group_intervention_row_count": len(all_interventions),
            "single_unit_causal_count": 0,
        },
        "evidence_boundary": {
            "mapped_scope": "content_knowledge/color",
            "unmapped_families": [
                family["family_id"] for family in enriched_families
                if not family["physical_mapping"]["models"]
            ],
            "statement": "Only evidence-bearing units are published. No placeholder units are synthesized for unmapped families.",
            "single_unit_causal_closure": False,
        },
        "source_artifacts": [
            "tests/result/pattern_family_atlas/v2/families.jsonl",
            "tests/result/research_kernel/manifest.json",
            "tests/result/research_kernel/runs/phase286_color_real_units_*/",
            "tests/result/research_kernel/runs/phase287_*_red_component_trace/",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)

    validation = validate_bundle(output_dir)
    manifest["validation"] = {"status": "passed", **validation}
    write_json(output_dir / "manifest.json", manifest)

    checksums = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            checksums.append({"path": path.relative_to(output_dir).as_posix(), "sha256": file_sha256(path)})
    write_json(output_dir / "checksums.json", {"schema_version": "artifact_checksums.v1", "files": checksums})

    if public_dir.exists():
        shutil.rmtree(public_dir)
    shutil.copytree(output_dir, public_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--public-dir", type=Path, default=PUBLIC_DIR)
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_bundle(args.output_dir), ensure_ascii=False, indent=2))
        return
    manifest = build_bundle(args.output_dir, args.public_dir)
    print(json.dumps({"status": "ok", "metrics": manifest["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
