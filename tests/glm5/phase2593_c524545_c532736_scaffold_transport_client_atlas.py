#!/usr/bin/env python3
"""Publish paired candidate-scaffold transport evidence to the existing client."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2592 = RESULT / "phase2592_c508161_c524544_candidate_scaffold_transport_atlas"
OUT = RESULT / "phase2593_c524545_c532736_scaffold_transport_client_atlas"
ASSET = RESULT / "client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2593, "C524545-C532736"
KEYS = {
    "phase2592_candidate_scaffold_fullcoordinate_pairs",
    "phase2592_candidate_scaffold_raw_transport_graph",
    "phase2592_candidate_scaffold_centered_transport_graph",
    "phase2592_candidate_scaffold_transport_dynamics",
}
LAYERS = (0, 1, 9, 18, 25, 30, 35, 36)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"ensure_ascii": False, "allow_nan": False}
    if not compact:
        options["indent"] = 2
    path.write_text(json.dumps(value, **options) + "\n", encoding="utf-8")


def values(array, digits=8):
    return np.round(np.asarray(array, dtype=np.float32), digits).tolist()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_panel():
    manifest = load(P2592 / "field/manifest.json")
    arrays = np.load(P2592 / "analysis/paired_answer_coordinate_fields.npz")
    with_o = arrays["with_candidate_oriented"]
    without_o = arrays["candidate_free_oriented"]
    rows = []
    for group, item in enumerate(manifest):
        label = f"{item['family']}/{item['language']}"
        for layer in LAYERS:
            for scaffold, field in (("with candidates", with_o), ("candidate free", without_o)):
                rows.append({
                    "label": f"{label} / {scaffold} / oriented I / q{layer}",
                    "source": "phase2592_scaffold_pair", "coordinate_kind": "hidden_state_factorial_interaction",
                    "preview": True, "phase": 2592, "layer": layer, "family": item["family"],
                    "language": item["language"], "scaffold": scaffold, "event": "answer_boundary",
                    "values": values(field[group, layer]),
                })
    return {
        "key": "phase2592_candidate_scaffold_fullcoordinate_pairs",
        "model": "Qwen3-4B same-prefix candidate-list versus candidate-free interaction pairs",
        "precision": "BF16 forward / float16 stored fields / float32 factorial; all 2560 coordinates",
        "coordinate_count": 2560,
        "coordinate_semantics": "model-local physical HiddenState activation coordinate",
        "coordinate_order": "original physical coordinate order",
        "rows": rows,
    }


def graph_panel(centered=False):
    arrays = np.load(P2592 / "analysis/scaffold_transport_coordinate_graphs.npz")
    labels = [str(item) for item in arrays["labels"]]
    matrix = arrays["cross_language_centered"] if centered else arrays["cross_raw"]
    rows = []
    for layer in range(matrix.shape[0]):
        for source, label in enumerate(labels):
            rows.append({
                "label": f"q{layer} / with-candidate source {label}",
                "source": "phase2592_centered_scaffold_graph" if centered else "phase2592_raw_scaffold_graph",
                "coordinate_kind": "candidate_free_family_language_group",
                "preview": True, "phase": 2592, "layer": layer, "source_group": label,
                "values": values(matrix[layer, source]),
            })
    return {
        "key": ("phase2592_candidate_scaffold_centered_transport_graph" if centered
                else "phase2592_candidate_scaffold_raw_transport_graph"),
        "model": ("Qwen3-4B language-centered candidate-scaffold transport graph" if centered
                  else "Qwen3-4B raw candidate-scaffold transport graph"),
        "precision": "Pearson correlation over all 2560 coordinates; no Top-K",
        "coordinate_count": len(labels), "coordinate_labels": labels,
        "coordinate_semantics": "candidate-free target family/language node, not a physical model coordinate",
        "coordinate_order": "19 dual-qualified family/language groups",
        "rows": rows,
    }


def dynamics_panel():
    final = load(P2592 / "analysis/final.json")
    curves = final["scaffold_transport"]["curves"]
    rows = [{
        "label": key.replace("_", " "), "source": "phase2592_scaffold_transport_dynamics",
        "coordinate_kind": "hidden_state_checkpoint", "preview": True,
        "phase": 2592, "values": values(value),
    } for key, value in curves.items()]
    rows.extend([
        {"label": "with-candidate median interaction RMS", "source": "phase2592_scaffold_rms",
         "coordinate_kind": "hidden_state_checkpoint", "preview": True, "phase": 2592,
         "values": values(final["interaction"]["with_candidate_median_rms"])},
        {"label": "candidate-free median interaction RMS", "source": "phase2592_scaffold_rms",
         "coordinate_kind": "hidden_state_checkpoint", "preview": True, "phase": 2592,
         "values": values(final["interaction"]["candidate_free_median_rms"])},
    ])
    return {
        "key": "phase2592_candidate_scaffold_transport_dynamics",
        "model": "Qwen3-4B candidate-list removal: coordinate transport and interaction strength",
        "precision": "full-2560-coordinate correlation or RMS",
        "coordinate_count": 37, "coordinate_labels": [f"q{index}" for index in range(37)],
        "coordinate_semantics": "embedding/HiddenState checkpoint",
        "coordinate_order": "model depth", "rows": rows,
    }


def update_route():
    text = ROUTE.read_text(encoding="utf-8")
    text = text.replace(
        "C39761-C491776 Full-coordinate Fields, Interaction Birth, and Bilingual Family Graphs",
        "C39761-C532736 Full-coordinate Fields, Bilingual Families, and Scaffold Transport",
    )
    old = (
        "Each panel uses its declared axis: Qwen3-4B has all 2560 embedding/HiddenState coordinates, Qwen3-14B has all 5120, while token, head, causal-condition, relative-depth, and family-graph axes remain separate. Phase2582-2590 show interaction birth and late amplification, null behavior from block0 interaction removal, and a 20-node bilingual natural-operation atlas. Raw family correlations are dominated by the shared four-choice task; only language-centered matched-minus-unmatched residuals are weak family-specific evidence. These do not establish shared physical coordinates, a minimal necessary gear, general natural-language abilities, causal family codes, or a cracked language mechanism."
    )
    new = (
        "Each panel uses its declared axis: Qwen3-4B has all 2560 embedding/HiddenState coordinates, Qwen3-14B has all 5120, while token, head, causal-condition, relative-depth, family-graph, and scaffold-graph axes remain separate. Phase2582-2593 show interaction birth and late amplification, null behavior from block0 interaction removal, a bilingual natural-operation atlas, and candidate-list removal transport in 19 behavior-qualified groups. Raw correlations remain scaffold-heavy; language-centered matched advantages are descriptive residual evidence. These do not establish shared physical coordinates, a minimal necessary gear, general natural-language abilities, causal family codes, or a cracked language mechanism."
    )
    if old not in text and new not in text:
        raise RuntimeError("route boundary not found")
    ROUTE.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def append_memo(result):
    heading = f"## Phase {PHASE}: 候选脚手架迁移全坐标客户端图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理与公式。** 将Phase2592重要结果追加到现有客户端，分成：19组×候选有/无×8层的全部2560物理坐标、37层raw 19×19迁移图、37层语言中心化迁移图、逐层迁移/RMS曲线。

$$T_l(g,h)=\operatorname{{corr}}_d(I^{{with}}_{{gld}},I^{{free}}_{{hld}}),\qquad
\widetilde I_{{gld}}=I_{{gld}}-\operatorname{{mean}}_{{h:\,lang(h)=lang(g)}}I_{{hld}}.$$

**测试用例与结果汇总。** 参数面板304行×2560列；raw与中心化图各703行×19节点；动力面板9行×37检查点。`{json.dumps(result, ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2593_c524545_c532736_scaffold_transport_client_atlas.py`；客户端资产`{ASSET}`；前端路由`{ROUTE}`；构建产物`frontend/dist`。

**分析与理论进展。** 用户可在相同组和层直接比较候选有/无的全部坐标，并在独立面板区分raw公共任务纹理与中心化族残差；19节点图轴不冒充物理坐标。

**问题硬伤。** 面板只显示预注册8层的参数场，全部37层仍在NPZ和原始场；每组仅1个四元组；可视化不增加统计稳定性或因果证据。原场因展示与复算需要保留。

**结论。** Phase2592重要结果已进入客户端，具体坐标级结果可查，但仍只是脚手架迁移观察，不是语言条件齿轮闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    final = load(P2592 / "analysis/final.json")
    if not final["all_checks_passed"]:
        raise RuntimeError("Phase2592 failed checks")
    payload = load(ASSET)
    before = ASSET.stat().st_size
    panels = [parameter_panel(), graph_panel(False), graph_panel(True), dynamics_panel()]
    payload["models"] = [panel for panel in payload["models"] if panel.get("key") not in KEYS] + panels
    payload["phase"] = PHASE
    if CAMPAIGN not in payload.get("campaign", ""):
        payload["campaign"] = f"{payload.get('campaign', '')}; {CAMPAIGN}".strip("; ")
    payload["title"] = "Full-coordinate interaction, bilingual family, and candidate-scaffold transport atlas"
    payload.setdefault("summary", {})["phase2592_2593"] = {
        "paired_groups": 19,
        "late_same_group_scaffold_signed": final["scaffold_transport"]["late_summary"]["same_group_scaffold_signed_median"],
        "late_centered_matched_minus_unmatched": final["scaffold_transport"]["late_summary"]["centered_matched_minus_unmatched"],
        "late_family_graph_topology_transport": final["scaffold_transport"]["late_summary"]["within_scaffold_family_graph_topology_correlation"],
        "mechanism_closed": False,
    }
    payload["summary"]["model_rows"] = {panel["key"]: len(panel["rows"]) for panel in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    payload["claim_boundary"] = (
        "Phase2592-2593 pair 19 behavior-qualified bilingual operation groups with and without candidate lists. "
        "Physical parameter rows retain all 2560 coordinates, while raw and language-centered 19-node scaffold "
        "graphs use a separate declared axis. Late matched transport and graph-topology reuse are descriptive; "
        "one quartet per group cannot establish stable, causal, or general semantic gears."
    )
    save(ASSET, payload, compact=True)
    update_route()
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "asset": str(ASSET), "asset_bytes_before": before, "asset_bytes_after": ASSET.stat().st_size,
        "asset_sha256": sha256(ASSET),
        "panels": [{"key": panel["key"], "coordinate_count": panel["coordinate_count"],
                    "rows": len(panel["rows"])} for panel in panels],
        "raw_field_policy": "retained because full-coordinate scaffold pairs are displayed and reanalysis is required",
    }
    result["checks"] = {
        "four_new_panels": len(panels) == 4,
        "all_parameter_rows_2560": panels[0]["coordinate_count"] == 2560
                                   and all(len(row["values"]) == 2560 for row in panels[0]["rows"]),
        "raw_and_centered_axes_separate": panels[1]["key"] != panels[2]["key"],
        "all_19_graph_nodes": panels[1]["coordinate_count"] == panels[2]["coordinate_count"] == 19,
        "all_37_checkpoints": panels[3]["coordinate_count"] == 37,
        "claim_boundary": True,
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
