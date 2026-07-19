#!/usr/bin/env python3
"""Publish Phase563 multi-position block evidence on the fixed-identity route."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PARENT_PUBLIC = ROOT / "frontend/public/vis_data/phase562_fixed_identity_color_route_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase563_fixed_identity_color_route_atlas"
OUT_DIR = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
RESULT_ATLAS = OUT_DIR / "phase563_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱GLM4）",
    "deepseek7b": "DS7B（深度求索7B）",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def publish() -> None:
    analysis = read_json(OUT_DIR / "phase563_multiposition_reader_analysis.json")
    items = []
    passed = 0
    rejected = 0
    for model in MODELS:
        parent_name = f"phase562_{model}_fixed_identity_color_route.json"
        payload = copy.deepcopy(read_json(PARENT_PUBLIC / parent_name))
        payload["schema_version"] = "phase563_fixed_identity_color_route_atlas.v1"
        payload["phase_id"] = "Phase563"
        payload["generated_at"] = now()
        if model == "qwen3":
            graph = payload["graph"]
            graph["meta"]["phase563_multiposition_candidate_count"] = len(
                analysis["candidate_reports"]
            )
            graph["meta"]["phase563_qualified_block_count"] = analysis[
                "qualified_block_count"
            ]
            graph["meta"]["full_residual_multiposition_operator_tested"] = False
            graph["meta"]["strict_closed_mechanisms"] = 0
            for report in analysis["candidate_reports"]:
                is_pass = bool(report["validation_gate_pass"])
                passed += int(is_pass)
                rejected += int(not is_pass)
                layer = int(report["layer"])
                block = report["position_block"]
                node_id = f"phase563:qwen3:attention_L{layer}:{block}"
                graph["nodes"].append({
                    "id": node_id,
                    "label": (
                        f"L{layer} {block} / "
                        f"{'多位置充分性通过' if is_pass else '多位置注意力块否定'}"
                    ),
                    "model": "qwen3",
                    "family_id": "semantic_knowledge",
                    "mechanism_id": "fixed_identity_color_binding",
                    "type": (
                        "multiposition_attention_block_sufficiency"
                        if is_pass else "rejected_multiposition_attention_block"
                    ),
                    "position": [
                        float(layer * 2 - 36),
                        30.0 if block == "all_semantic_roles" else 24.0,
                        0.0,
                    ],
                    "layer": layer,
                    "component": report["component"],
                    "semantic_positions": report["semantic_positions"],
                    "score": float(report["correct_donor_win_rate"]),
                    "color": "#15803d" if is_pass else "#dc2626",
                    "observer": not is_pass,
                    "physical": True,
                    "causal": is_pass,
                    "compute_edge": False,
                    "single_neuron": False,
                    "pipeline_sealed": False,
                    "show_label": True,
                    "correct_donor_win_rate": report["correct_donor_win_rate"],
                    "minimum_factorial_cell_donor_win_rate": report[
                        "minimum_factorial_cell_donor_win_rate"
                    ],
                    "correct_donor_mean_switch_effect": report[
                        "correct_donor_mean_switch_effect"
                    ],
                    "validation_gate_pass": is_pass,
                    "negative_evidence": not is_pass,
                })
                graph["edges"].append({
                    "id": f"phase563:qwen3:source_L3_to_attention_L{layer}:{block}",
                    "source": "phase562:qwen3:source_color:L3",
                    "target": node_id,
                    "type": (
                        "multiposition_block_sufficiency"
                        if is_pass else "rejected_multiposition_block"
                    ),
                    "label": (
                        "独立多位置块充分性"
                        if is_pass else "独立验证未支持该注意力块"
                    ),
                    "score": float(report["correct_donor_win_rate"]),
                    "predictive": False,
                    "causal": is_pass,
                    "compute_edge": False,
                    "negative_evidence": not is_pass,
                    "evidence_level": (
                        "block_sufficiency" if is_pass else "independent_causal_rejection"
                    ),
                })
            payload["evidence_scope"] = (
                "Phase559-562 source-color control and propagation evidence plus Phase563 independent "
                "multi-position attention-block validation; no compute closure or neuron claim"
            )
        filename = f"phase563_{model}_fixed_identity_color_route.json"
        write_json(PUBLIC / filename, payload)
        write_json(RESULT_ATLAS / filename, payload)
        items.append({
            "id": f"phase563_{model}",
            "model": model,
            "label": f"{LABELS[model]} 固定身份颜色多位置路线",
            "path": filename,
        })
    manifest = {
        "schema_version": "phase563_fixed_identity_color_route_atlas_manifest.v1",
        "generated_at": now(),
        "route_id": "gpt5",
        "evidence_scope": "来源颜色粗因果路线、传播观测、单位置与多位置读出验证",
        "items": items,
    }
    write_json(PUBLIC / "manifest.json", manifest)
    write_json(RESULT_ATLAS / "manifest.json", manifest)

    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase563_fixed_identity_color_route_atlas",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase559-563 固定身份颜色多位置路线",
        "description": "固定身份颜色来源控制、跨位置传播、单位置及多位置注意力块验证。",
        "manifest_path": "/vis_data/phase563_fixed_identity_color_route_atlas/manifest.json",
        "manifest_schema": "phase563_fixed_identity_color_route_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase563_fixed_identity_color_route_atlas",
        "models": list(MODELS),
        "evidence_scope": "证据分级图谱；不包含绑定算子、严格计算边、参数或神经元闭合",
        "color": "#15803d",
    }
    positions = {row["id"]: index for index, row in enumerate(registry["sources"])}
    if source["id"] in positions:
        registry["sources"][positions[source["id"]]] = source
    else:
        registry["sources"].append(source)
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    write_json(OUT_DIR / "phase563_atlas_publish_summary.json", {
        "schema_version": "phase563_atlas_publish_summary.v1",
        "phase_id": "Phase563",
        "created_at": now(),
        "source_id": source["id"],
        "model_count": len(MODELS),
        "qualified_multiposition_block_count": passed,
        "rejected_multiposition_block_count": rejected,
        "compute_edge_count": 0,
        "single_neuron_node_count": 0,
        "strict_closed_mechanisms": 0,
        "sealed_split_read": False,
    })
    print(OUT_DIR / "phase563_atlas_publish_summary.json")


if __name__ == "__main__":
    publish()
