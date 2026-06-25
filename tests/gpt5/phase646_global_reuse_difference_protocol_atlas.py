#!/usr/bin/env python3
"""
Phase 646: Global Reuse-Difference Protocol Atlas Schema and First Batch.

This is an atlas-infrastructure phase. It consumes existing Phase 641-645
artifacts and writes a small, inspectable protocol-mechanism atlas. It does not
run CUDA model inference.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


MODELS = ["qwen3", "glm4", "deepseek7b"]
OUT_ROOT = Path("results/glm5_phase646_global_reuse_difference_protocol_atlas")

PHASE_PATHS = {
    641: Path("results/glm5_phase641_separator_protocol_formation_interval_audit"),
    642: Path("results/glm5_phase642_endpoint_dominance_vs_distributed_formation"),
    643: Path("results/glm5_phase643_protocol_trajectory_natural_generation_closure"),
    645: Path("results/glm5_phase645_protocol_trajectory_side_effect_boundary_atlas"),
}

MECHANISMS = [
    {
        "node_id": "mechanism:value_short_answer_protocol",
        "name": "value_short_answer_protocol",
        "mechanism_family": "protocol_gate",
        "description": "Boundary-conditioned trajectory that pushes generation toward direct category value output.",
    },
    {
        "node_id": "mechanism:newline_explanation_protocol",
        "name": "newline_explanation_protocol",
        "mechanism_family": "protocol_gate",
        "description": "Protocol tendency that opens multiline reasoning/explanation rather than direct value output.",
    },
    {
        "node_id": "mechanism:non_value_answer_protocol",
        "name": "non_value_answer_protocol",
        "mechanism_family": "protocol_gate",
        "description": "Task protocol for yes/no or other non-category-value answers.",
    },
]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pct(row: Dict[str, Any], field: str) -> float | None:
    n = row.get("n")
    if not n:
        return None
    val = row.get(field)
    if val is None:
        return None
    return float(val) / float(n)


def by_mode(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["mode"]: r for r in rows}


def phase_file(phase: int, model: str) -> Path:
    stems = {
        641: f"phase641_{model}_separator_protocol_formation_interval_audit_confirm.json",
        642: f"phase642_{model}_endpoint_dominance_vs_distributed_formation_confirm.json",
        643: f"phase643_{model}_protocol_trajectory_natural_generation_closure_confirm.json",
        645: f"phase645_{model}_protocol_trajectory_side_effect_boundary_atlas_confirm.json",
    }
    return PHASE_PATHS[phase] / stems[phase]


def classify_polarity(model: str, split: str, original: Dict[str, Any], restore: Dict[str, Any]) -> str:
    n = max(1, int(restore.get("n", 1)))
    exact_gain = int(restore.get("exact", 0)) - int(original.get("exact", 0))
    newline_drop = int(original.get("newline_top0", 0)) - int(restore.get("newline_top0", 0))
    if split in {"target_failure", "original_correct"}:
        if exact_gain >= max(3, n // 8) and newline_drop >= max(3, n // 8):
            return "beneficial_value_protocol"
        if exact_gain <= -max(3, n // 8):
            return "harmful_or_opposite_protocol"
        return "weak_or_neutral"
    if split in {"relation_changed", "explanation_needed", "non_value"}:
        if int(restore.get("exact", 0)) >= max(6, n // 4):
            return "side_effect_value_absorption"
        if newline_drop >= max(3, n // 8) and int(restore.get("exact", 0)) > int(original.get("exact", 0)):
            return "possible_side_effect"
        return "boundary_respected_or_neutral"
    return "insufficient_data"


def load_phase645_boundary_rows() -> List[Dict[str, Any]]:
    rows = []
    for model in MODELS:
        path = phase_file(645, model)
        if not path.exists():
            continue
        data = read_json(path)
        split_rows = data["summary"]["by_split"]
        for split, rs in split_rows.items():
            modes = by_mode(rs)
            original = modes.get("original", {})
            inline = modes.get("inline", {})
            restore = modes.get("to_original_middle_restore", {})
            remove = modes.get("remove_from_inline_middle_restore", {})
            row = {
                "atlas_row_id": f"boundary:{model}:{split}",
                "model": model,
                "split": split,
                "prompt_kind": restore.get("prompt_kind") or original.get("prompt_kind"),
                "n": restore.get("n") or original.get("n"),
                "original_exact": original.get("exact"),
                "inline_exact": inline.get("exact"),
                "to_original_restore_exact": restore.get("exact"),
                "remove_from_inline_restore_exact": remove.get("exact"),
                "original_newline": original.get("newline_top0"),
                "inline_newline": inline.get("newline_top0"),
                "to_original_restore_newline": restore.get("newline_top0"),
                "remove_from_inline_restore_newline": remove.get("newline_top0"),
                "original_gen_short": original.get("gen_short"),
                "to_original_restore_gen_short": restore.get("gen_short"),
                "polarity": classify_polarity(model, split, original, restore),
                "source_phase": 645,
                "source_path": str(path),
            }
            rows.append(row)
    return rows


def load_trajectory_evidence() -> List[Dict[str, Any]]:
    evidence = []
    for model in MODELS:
        for phase in [641, 642, 643]:
            path = phase_file(phase, model)
            if not path.exists():
                continue
            data = read_json(path)
            for row in data["summary"].get("by_mode", []):
                mode = row["mode"]
                keep = False
                if phase == 641:
                    keep = mode in {"original", "inline", "L17_20_restore", "L17_20_random", "L17_20_reverse"}
                elif phase == 642:
                    keep = (
                        mode in {"original", "inline"}
                        or (row.get("interval") == "L17_20" and row.get("variant") in {"full", "middle"})
                    )
                elif phase == 643:
                    keep = mode in {
                        "original",
                        "inline",
                        "to_original_full_restore",
                        "to_original_middle_restore",
                        "remove_from_inline_full_restore",
                        "remove_from_inline_middle_restore",
                    }
                if not keep:
                    continue
                evidence.append({
                    "evidence_id": f"phase{phase}:{model}:{mode}",
                    "source_phase": phase,
                    "model": model,
                    "mode": mode,
                    "interval": row.get("interval") or data.get("interval"),
                    "variant": row.get("variant"),
                    "control": row.get("control"),
                    "layers": row.get("layers") or data.get("middle_layers") or data.get("layers"),
                    "n": row.get("n"),
                    "tok0_hit": row.get("tok0_hit"),
                    "exact": row.get("exact"),
                    "newline_top0": row.get("newline_top0"),
                    "mean_prefix_rank": row.get("mean_prefix_rank"),
                    "mean_prefix_minus_newline": row.get("mean_prefix_minus_newline"),
                    "causal_level": 7 if phase == 643 else 6 if phase == 642 else 5,
                    "source_path": str(path),
                })
    return evidence


def build_nodes(boundary_rows: List[Dict[str, Any]], trajectory_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    nodes["atlas:reuse_difference_protocol_v1"] = {
        "node_id": "atlas:reuse_difference_protocol_v1",
        "node_type": "atlas",
        "title": "Global Reuse-Difference Protocol Atlas v1",
        "phase": 646,
    }
    for mech in MECHANISMS:
        nodes[mech["node_id"]] = {"node_type": "mechanism", **mech}
    for model in MODELS:
        nodes[f"model:{model}"] = {"node_id": f"model:{model}", "node_type": "model", "name": model}
    for split in sorted({r["split"] for r in boundary_rows}):
        nodes[f"boundary:{split}"] = {"node_id": f"boundary:{split}", "node_type": "boundary_condition", "name": split}
    for row in trajectory_evidence:
        nodes[f"evidence:{row['evidence_id']}"] = {
            "node_id": f"evidence:{row['evidence_id']}",
            "node_type": "evidence",
            "phase": row["source_phase"],
            "model": row["model"],
            "mode": row["mode"],
            "causal_level": row["causal_level"],
        }
    return list(nodes.values())


def build_edges(boundary_rows: List[Dict[str, Any]], trajectory_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    edges = []
    for mech in MECHANISMS:
        edges.append({
            "source": "atlas:reuse_difference_protocol_v1",
            "target": mech["node_id"],
            "edge_type": "contains_mechanism",
        })
    for row in boundary_rows:
        target_mech = "mechanism:value_short_answer_protocol"
        if row["split"] == "non_value":
            target_mech = "mechanism:non_value_answer_protocol"
        if row["split"] == "explanation_needed":
            target_mech = "mechanism:newline_explanation_protocol"
        edges.append({
            "source": target_mech,
            "target": f"boundary:{row['split']}",
            "edge_type": "has_boundary_profile",
            "model": row["model"],
            "polarity": row["polarity"],
            "source_phase": row["source_phase"],
        })
        edges.append({
            "source": f"model:{row['model']}",
            "target": target_mech,
            "edge_type": "model_exhibits_or_contrasts",
            "polarity": row["polarity"],
            "split": row["split"],
        })
    for ev in trajectory_evidence:
        edges.append({
            "source": "mechanism:value_short_answer_protocol",
            "target": f"evidence:{ev['evidence_id']}",
            "edge_type": "supported_by",
            "model": ev["model"],
            "phase": ev["source_phase"],
            "causal_level": ev["causal_level"],
        })
    return edges


def write_boundary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "atlas_row_id",
        "model",
        "split",
        "n",
        "original_exact",
        "inline_exact",
        "to_original_restore_exact",
        "remove_from_inline_restore_exact",
        "original_newline",
        "inline_newline",
        "to_original_restore_newline",
        "remove_from_inline_restore_newline",
        "original_gen_short",
        "to_original_restore_gen_short",
        "polarity",
        "source_phase",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_report(out_dir: Path, boundary_rows: List[Dict[str, Any]], trajectory_evidence: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("# Phase 646 Global Reuse-Difference Protocol Atlas\n")
    lines.append("本阶段不运行模型，只把 Phase 641-645 的客观结果整理为第一版协议图谱。")
    lines.append("")
    lines.append("## Atlas Nodes\n")
    for mech in MECHANISMS:
        lines.append(f"- `{mech['node_id']}`: {mech['description']}")
    lines.append("")
    lines.append("## Boundary Matrix Highlights\n")
    for model in MODELS:
        lines.append(f"### {model}\n")
        for row in [r for r in boundary_rows if r["model"] == model]:
            lines.append(
                f"- {row['split']}: n={row['n']}, original exact/newline="
                f"{row['original_exact']}/{row['original_newline']}, "
                f"restore exact/newline={row['to_original_restore_exact']}/{row['to_original_restore_newline']}, "
                f"polarity=`{row['polarity']}`"
            )
        lines.append("")
    lines.append("## Trajectory Evidence Count\n")
    lines.append(f"- trajectory_evidence_rows: {len(trajectory_evidence)}")
    lines.append("")
    lines.append("## Strict Interpretation\n")
    lines.append("- DS7B 的 value short-answer protocol 已有生成闭环和边界副作用证据。")
    lines.append("- qwen3 和 GLM4 不应被硬套为同一层区间、同一 separator 字符机制。")
    lines.append("- atlas 当前是标准化索引，不是完整全局理论。下一步应把 writer graph 和更多输出类型补进节点。")
    lines.append("")
    lines.append("## Next Phase\n")
    lines.append("Phase 647 应执行 protocol writer graph audit，把 atlas 中的 value_short_answer_protocol 节点从 layer_out trajectory 继续拆到 attention / MLP / residual update writer。")
    (out_dir / "phase646_atlas_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_atlas(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    boundary_rows = load_phase645_boundary_rows()
    trajectory_evidence = load_trajectory_evidence()
    nodes = build_nodes(boundary_rows, trajectory_evidence)
    edges = build_edges(boundary_rows, trajectory_evidence)

    write_jsonl(out_dir / "atlas_nodes.jsonl", nodes)
    write_jsonl(out_dir / "atlas_edges.jsonl", edges)
    write_jsonl(out_dir / "atlas_evidence.jsonl", trajectory_evidence)
    write_jsonl(out_dir / "atlas_boundary_profiles.jsonl", boundary_rows)
    write_boundary_csv(out_dir / "atlas_boundary_matrix.csv", boundary_rows)

    schema = {
        "phase": 646,
        "name": "Global Reuse-Difference Protocol Atlas v1",
        "node_types": ["atlas", "mechanism", "model", "boundary_condition", "evidence"],
        "mechanism_required_fields": [
            "causal_unit",
            "trajectory_interval",
            "sufficiency",
            "necessity",
            "generation_closure",
            "semantic_boundary",
            "task_boundary",
            "output_type_boundary",
            "side_effect_profile",
            "cross_model_polarity",
        ],
        "boundary_profile_fields": sorted(boundary_rows[0].keys()) if boundary_rows else [],
        "evidence_fields": sorted(trajectory_evidence[0].keys()) if trajectory_evidence else [],
    }
    (out_dir / "atlas_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, boundary_rows, trajectory_evidence)

    index = {
        "phase": 646,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "nodes": len(nodes),
        "edges": len(edges),
        "boundary_profiles": len(boundary_rows),
        "trajectory_evidence": len(trajectory_evidence),
        "files": {
            "nodes": "atlas_nodes.jsonl",
            "edges": "atlas_edges.jsonl",
            "evidence": "atlas_evidence.jsonl",
            "boundary_profiles": "atlas_boundary_profiles.jsonl",
            "boundary_matrix": "atlas_boundary_matrix.csv",
            "schema": "atlas_schema.json",
            "report": "phase646_atlas_report.md",
        },
    }
    (out_dir / "phase646_atlas_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    args = parser.parse_args()
    index = build_atlas(Path(args.output_dir))
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
