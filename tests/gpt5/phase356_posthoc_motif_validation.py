#!/usr/bin/env python3
"""Reveal labels only after blind motif candidates are frozen and validate coverage."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/gpt5/result/phase356_blind_neural_path_cartography/coarse_trace_feasibility"
DESTINATIONS = (
    ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
)
PHASE = "Phase356"
SCHEMA_VERSION = "32.0.0"
MODELS = {"qwen3", "glm4", "deepseek7b"}


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


def validate() -> dict[str, Any]:
    schema_quality = read_json(BASE / "phase356_schema_quality_summary.json")
    discovery_summary = read_json(BASE / "phase356_blind_discovery_summary.json")
    registry = read_jsonl(BASE / "phase356_blind_motif_registry.jsonl")
    assignments = read_jsonl(BASE / "phase356_blind_motif_assignments.jsonl")
    edges = read_jsonl(BASE / "phase356_blind_graph_edges.jsonl")
    labels = {
        row["blind_case_id"]: row
        for row in read_jsonl(BASE / "sealed_labels" / "phase356_private_label_key.jsonl")
    }
    candidate_ids = {row["motif_id"] for row in registry if row["blind_calibration_stable"]}
    cases_by_motif_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in assignments:
        if row["motif_id"] in candidate_ids:
            cases_by_motif_split[(row["motif_id"], row["blind_split"])].add(row["blind_case_id"])

    nodes = []
    for row in registry:
        if row["motif_id"] not in candidate_ids:
            continue
        discovery_cases = cases_by_motif_split[(row["motif_id"], "blind_discovery")]
        calibration_cases = cases_by_motif_split[(row["motif_id"], "blind_calibration")]
        discovery_models = {labels[case]["model"] for case in discovery_cases}
        calibration_models = {labels[case]["model"] for case in calibration_cases}
        mechanism_counts = Counter(labels[case]["mechanism_id"] for case in discovery_cases | calibration_cases)
        family_counts = Counter(labels[case]["family_id"] for case in discovery_cases | calibration_cases)
        total = sum(mechanism_counts.values())
        dominant_mechanism, dominant_count = mechanism_counts.most_common(1)[0]
        cross_model = discovery_models == MODELS and calibration_models == MODELS
        nodes.append({
            **row,
            "node_id": row["motif_id"], "node_type": "blind_repeated_physical_motif",
            "posthoc_labels_revealed": True,
            "discovery_model_count": len(discovery_models),
            "calibration_model_count": len(calibration_models),
            "cross_model_discovery_calibration": cross_model,
            "observed_family_count": len(family_counts),
            "observed_mechanism_count": len(mechanism_counts),
            "family_case_counts": dict(sorted(family_counts.items())),
            "mechanism_case_counts": dict(sorted(mechanism_counts.items())),
            "dominant_mechanism": dominant_mechanism,
            "dominant_mechanism_share": round(dominant_count / total, 7),
            "functional_scope": "shared_coarse_skeleton" if len(mechanism_counts) >= 2 else "single_mechanism_coarse_skeleton",
            "mapping_status": "blind_calibration_repeated_coarse_motif" if cross_model else "model_incomplete_coarse_motif",
            "physical_heldout_tested": False,
            "causal_status": "not_tested", "single_unit_causal": False,
        })
    nodes.sort(key=lambda row: row["node_id"])
    repeated = [row for row in nodes if row["cross_model_discovery_calibration"]]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            **discovery_summary["denominator"],
            "posthoc_candidate_node_count": len(nodes),
            "posthoc_graph_edge_count": len(edges),
        },
        "quality": schema_quality["quality"],
        "results": {
            **discovery_summary["results"],
            "cross_model_discovery_calibration_motif_count": len(repeated),
            "shared_mechanism_motif_count": sum(row["observed_mechanism_count"] >= 2 for row in repeated),
            "single_mechanism_motif_count": sum(row["observed_mechanism_count"] == 1 for row in repeated),
            "physical_heldout_stable_motif_count": 0,
            "causal_motif_count": 0,
            "single_unit_causal_count": 0,
        },
        "claim_boundary": {
            "coarse_motif_is_full_neural_path": False,
            "calibration_repetition_is_physical_heldout": False,
            "shared_motif_is_language_invariant": False,
            "motif_is_causal": False,
            "phase356_full_success_gate": False,
        },
        "stage_assessment": {
            "blind_pipeline_feasible": bool(repeated),
            "full_trace_schema_complete": False,
            "reason_full_trace_incomplete": "Phase354 retained component norms at four roles, not raw residual vectors, attention matrices, normalization states, heads, channels, or neurons.",
            "next_required_work": "instrument reconstruction-valid coarse skeleton before balanced neuron shards and anchor cases",
        },
        "physical_heldout_revealed": False,
        "causal_intervention_executed": False,
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(BASE / "phase356_posthoc_motif_nodes.jsonl", nodes)
    write_json(BASE / "phase356_global_summary.json", summary)
    report = [
        "# Phase356 Blind Neural Path Cartography Feasibility", "",
        f"- Source raw trace rows: {schema_quality['denominator']['source_raw_trace_row_count']}",
        f"- Blind skeleton rows: {schema_quality['denominator']['blind_skeleton_row_count']}",
        f"- Unique blind motifs: {discovery_summary['denominator']['unique_motif_count']}",
        f"- Blind calibration-stable motifs: {discovery_summary['results']['blind_calibration_stable_motif_count']}",
        f"- Cross-model discovery/calibration motifs: {len(repeated)}", "",
        "This is a coarse feasibility result. Full-vector reconstruction, physical heldout, causality, and neuron coverage remain closed.",
    ]
    (BASE / "phase356_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    for destination in DESTINATIONS:
        write_jsonl(destination / "phase356_blind_motif_nodes.jsonl", nodes)
        write_jsonl(destination / "phase356_blind_motif_edges.jsonl", edges)
        write_json(destination / "phase356_blind_cartography_summary.json", summary)
        manifest_path = destination / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {"schema_version": "pattern_family_atlas.v2"}
        manifest["updated_at"] = now()
        manifest["phase356"] = {
            "status": "coarse_blind_cartography_feasibility_only",
            "source_raw_trace_rows": schema_quality["denominator"]["source_raw_trace_row_count"],
            "blind_skeleton_rows": schema_quality["denominator"]["blind_skeleton_row_count"],
            "blind_calibration_stable_motifs": discovery_summary["results"]["blind_calibration_stable_motif_count"],
            "cross_model_repeated_motifs": len(repeated),
            "full_trace_schema_complete": False,
            "physical_heldout_revealed": False,
            "causal_motif_count": 0, "single_unit_causal_count": 0,
            "files": [
                "phase356_blind_motif_nodes.jsonl", "phase356_blind_motif_edges.jsonl",
                "phase356_blind_cartography_summary.json",
            ],
        }
        write_json(manifest_path, manifest)
    return summary


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
