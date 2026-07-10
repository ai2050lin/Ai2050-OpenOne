#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
PHASE = "Phase300"
SCHEMA_VERSION = "2.27.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(k)) for k in keys)].append(row)
    return out


def main() -> None:
    behavior = read_jsonl(V2 / "phase294_expanded_behavior_rows.jsonl")
    readout = read_jsonl(V2 / "phase294_expanded_readout_rows.jsonl")
    component = read_jsonl(V2 / "phase296_component_summary_rows.jsonl")
    causal = read_jsonl(V2 / "phase298_mlp_causal_audit_rows.jsonl")
    feature_v5 = read_jsonl(V2 / "phase299_feature_matrix_v5_rows.jsonl")
    by_cell_behavior = group(behavior, "model", "family_id")
    by_cell_readout = group(readout, "model", "family_id")
    by_cell_component = group(component, "model", "family_id")
    by_cell_causal = group(causal, "model", "family_id")
    cells: list[dict[str, Any]] = []
    for row in feature_v5:
        key = (str(row.get("model")), str(row.get("family_id")))
        b = by_cell_behavior.get(key, [])
        r = by_cell_readout.get(key, [])
        c = by_cell_component.get(key, [])
        a = by_cell_causal.get(key, [])
        behavior_score = mean_safe([1.0 if x.get("answer_correct_proxy") else 0.0 for x in b])
        pattern_score = mean_safe([1.0 if x.get("pattern_matched_proxy") else 0.0 for x in b])
        stop_rate = mean_safe([1.0 if x.get("model_stop_executed") else 0.0 for x in b])
        continue_rate = mean_safe([1.0 if x.get("competition_winner") == "continue" else 0.0 for x in r])
        mlp_rate = mean_safe([1.0 if x.get("dominant_positive_component") == "mlp" else 0.0 for x in c])
        attn_rate = mean_safe([1.0 if x.get("dominant_positive_component") == "attention" else 0.0 for x in c])
        causal_support = mean_safe([1.0 if x.get("necessity_supported") else 0.0 for x in a])
        winner_flip = mean_safe([1.0 if x.get("winner_changed") else 0.0 for x in a])
        evidence_layers = sum(1 for count in [len(b), len(r), len(c), len(a)] if count > 0)
        evidence_completeness = evidence_layers / 4.0
        physical_path_confidence = clamp(0.30 * evidence_completeness + 0.25 * mlp_rate + 0.15 * attn_rate + 0.20 * causal_support + 0.10 * (1.0 - winner_flip))
        closure_gap = clamp(1.0 - (0.25 * pattern_score + 0.20 * stop_rate + 0.25 * winner_flip + 0.30 * evidence_completeness))
        cells.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "model": row.get("model"),
                "family_id": row.get("family_id"),
                "atlas_completion_v5": row.get("atlas_completion_v5"),
                "behavior_rows": len(b),
                "readout_rows": len(r),
                "component_rows": len(c),
                "causal_rows": len(a),
                "behavior_answer_correct_proxy_rate": behavior_score,
                "behavior_pattern_matched_proxy_rate": pattern_score,
                "behavior_model_stop_executed_rate": stop_rate,
                "readout_continue_winner_rate": continue_rate,
                "component_mlp_dominance_rate": mlp_rate,
                "component_attention_dominance_rate": attn_rate,
                "causal_necessity_supported_rate": causal_support,
                "causal_winner_flip_rate": winner_flip,
                "evidence_layer_count": evidence_layers,
                "evidence_completeness": round(evidence_completeness, 6),
                "physical_path_confidence": round(physical_path_confidence, 6),
                "closure_gap": round(closure_gap, 6),
                "atlas_status": classify_cell(evidence_completeness, physical_path_confidence, closure_gap),
                "next_priority": choose_priority(pattern_score, stop_rate, mlp_rate, causal_support, winner_flip, evidence_completeness),
            }
        )
    nodes = build_nodes(cells)
    edges = build_edges(cells)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "title": "Pattern Atlas v3 candidate synthesis",
        "cell_rows": len(cells),
        "node_rows": len(nodes),
        "edge_rows": len(edges),
        "behavior_rows": len(behavior),
        "readout_rows": len(readout),
        "component_rows": len(component),
        "causal_rows": len(causal),
        "mean_atlas_completion_v5": mean_safe([safe_float(x.get("atlas_completion_v5")) for x in cells]),
        "mean_physical_path_confidence": mean_safe([safe_float(x.get("physical_path_confidence")) for x in cells]),
        "mean_closure_gap": mean_safe([safe_float(x.get("closure_gap")) for x in cells]),
        "atlas_status_counts": dict(Counter(str(x.get("atlas_status")) for x in cells)),
        "next_priority_counts": dict(Counter(str(x.get("next_priority")) for x in cells)),
        "progress": {
            "language_pattern_family_atlas": 0.80,
            "sample_type_coverage": 0.70,
            "large_data_feature_mining": 0.68,
            "physical_distribution_puzzle": 0.74,
            "mechanism_causal_audit": 0.52,
            "closure": 0.21,
        },
        "hard_limits": [
            "Current causal patches reduce continuation margin but do not flip winner.",
            "Stop-source natural path is still not mapped.",
            "Only 27 expanded cases have component path measurements.",
            "Small-model mechanisms may deviate materially from larger language models.",
        ],
    }
    write_jsonl(V2 / "phase300_pattern_atlas_v3_cell_rows.jsonl", cells)
    write_jsonl(V2 / "phase300_pattern_atlas_v3_node_rows.jsonl", nodes)
    write_jsonl(V2 / "phase300_pattern_atlas_v3_edge_rows.jsonl", edges)
    write_json(V2 / "phase300_pattern_atlas_v3_summary.json", summary)
    write_json(V2 / "progress.json", {**read_json(V2 / "progress.json"), **summary["progress"], "last_phase": PHASE, "updated_at": now()})
    update_manifest(summary)
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def classify_cell(evidence: float, confidence: float, gap: float) -> str:
    if evidence >= 1.0 and confidence >= 0.65 and gap <= 0.45:
        return "candidate_physical_path_not_closed"
    if evidence >= 0.75 and confidence >= 0.45:
        return "partial_physical_path"
    if evidence >= 0.5:
        return "behavior_readout_component_partial"
    return "insufficient_internal_evidence"


def choose_priority(pattern: float, stop: float, mlp: float, causal: float, flip: float, evidence: float) -> str:
    if evidence < 1.0:
        return "fill_component_or_causal_evidence"
    if stop < 0.25:
        return "search_stop_source_path"
    if mlp >= 0.7 and causal > 0.0 and flip == 0.0:
        return "stronger_causal_intervention_design"
    if pattern < 0.5:
        return "protocol_pattern_failure_analysis"
    return "atlas_cell_consolidation"


def build_nodes(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for family in sorted({str(x.get("family_id")) for x in cells}):
        vals = [x for x in cells if str(x.get("family_id")) == family]
        nodes.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "node_id": f"phase300:family:{family}",
                "node_type": "pattern_family",
                "family_id": family,
                "cell_count": len(vals),
                "mean_physical_path_confidence": mean_safe([safe_float(x.get("physical_path_confidence")) for x in vals]),
                "mean_closure_gap": mean_safe([safe_float(x.get("closure_gap")) for x in vals]),
                "dominant_status": Counter(str(x.get("atlas_status")) for x in vals).most_common(1)[0][0],
            }
        )
    for model in sorted({str(x.get("model")) for x in cells}):
        vals = [x for x in cells if str(x.get("model")) == model]
        nodes.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "node_id": f"phase300:model:{model}",
                "node_type": "model",
                "model": model,
                "cell_count": len(vals),
                "mean_physical_path_confidence": mean_safe([safe_float(x.get("physical_path_confidence")) for x in vals]),
                "mean_closure_gap": mean_safe([safe_float(x.get("closure_gap")) for x in vals]),
            }
        )
    return nodes


def build_edges(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for row in cells:
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "edge_id": f"phase300:{row.get('model')}:{row.get('family_id')}:family_to_model",
                "source": f"phase300:family:{row.get('family_id')}",
                "target": f"phase300:model:{row.get('model')}",
                "edge_type": "pattern_family_physical_path_cell",
                "weight": row.get("physical_path_confidence"),
                "closure_gap": row.get("closure_gap"),
                "status": row.get("atlas_status"),
            }
        )
    return edges


def update_manifest(summary: dict[str, Any]) -> None:
    manifest_path = V2 / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.setdefault("generated_files", [])
    for name in [
        "phase300_pattern_atlas_v3_cell_rows.jsonl",
        "phase300_pattern_atlas_v3_node_rows.jsonl",
        "phase300_pattern_atlas_v3_edge_rows.jsonl",
        "phase300_pattern_atlas_v3_summary.json",
    ]:
        if name not in manifest["generated_files"]:
            manifest["generated_files"].append(name)
    manifest["last_phase"] = PHASE
    manifest["updated_at"] = now()
    manifest["phase300_summary"] = {
        "cell_rows": summary["cell_rows"],
        "mean_physical_path_confidence": summary["mean_physical_path_confidence"],
        "mean_closure_gap": summary["mean_closure_gap"],
    }
    write_json(manifest_path, manifest)


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase300 Pattern Atlas v3 Candidate",
        "",
        f"- cell_rows: {summary['cell_rows']}",
        f"- node_rows: {summary['node_rows']}",
        f"- edge_rows: {summary['edge_rows']}",
        f"- behavior_rows: {summary['behavior_rows']}",
        f"- readout_rows: {summary['readout_rows']}",
        f"- component_rows: {summary['component_rows']}",
        f"- causal_rows: {summary['causal_rows']}",
        f"- mean_physical_path_confidence: {summary['mean_physical_path_confidence']}",
        f"- mean_closure_gap: {summary['mean_closure_gap']}",
        f"- atlas_status_counts: {json.dumps(summary['atlas_status_counts'], ensure_ascii=False)}",
        f"- next_priority_counts: {json.dumps(summary['next_priority_counts'], ensure_ascii=False)}",
        "",
        "This is a synthesized atlas candidate, not closure.",
    ]
    (V2 / "phase300_pattern_atlas_v3_candidate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
