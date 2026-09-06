#!/usr/bin/env python3
"""Add Phase2588-2589 bilingual natural-operation evidence to the research client."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2588 = RESULT / "phase2588_c450817_c467200_bilingual_natural_operation_behavior"
P2589 = RESULT / "phase2589_c467201_c483584_bilingual_operation_fullcoordinate_atlas"
OUT = RESULT / "phase2590_c483585_c491776_bilingual_family_client_atlas"
ASSET = RESULT / "client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2590, "C483585-C491776"
KEYS = {
    "phase2589_bilingual_operation_exact_parameter_field",
    "phase2589_bilingual_operation_raw_family_graph",
    "phase2589_bilingual_operation_centered_family_graph",
    "phase2588_2589_bilingual_operation_dynamics",
}
SELECTED_LAYERS = (0, 1, 9, 18, 25, 30, 35, 36)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"ensure_ascii": False, "allow_nan": False}
    if not compact:
        options["indent"] = 2
    path.write_text(json.dumps(value, **options) + "\n", encoding="utf-8")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values(array, digits=8):
    return np.round(np.asarray(array, dtype=np.float32), digits).tolist()


def exact_parameter_panel():
    manifest = load(P2589 / "field/manifest.json")
    exemplar = np.load(P2589 / "analysis/exemplar_parameter_fields.npz")
    graphs = np.load(P2589 / "analysis/family_coordinate_graphs.npz")
    labels = [str(item) for item in graphs["labels"]]
    group_profiles = graphs["group_oriented_profiles"]
    rows = []
    first = manifest[0]
    embeddings = exemplar["embedding_parameter"]
    answer_hidden = exemplar["answer_hidden_parameter"]
    positions = [
        ("query_relation", first["regions"]["query_relation"][0]),
        ("query_value", first["regions"]["query_value"][0]),
        ("answer_boundary", first["answer_boundary_token"]),
    ]
    for name, position in positions:
        rows.append({
            "label": f"exact embedding / cell00 / token {position} / {name}",
            "source": "phase2589_exact_embedding", "coordinate_kind": "embedding_activation",
            "preview": True, "phase": 2589, "layer": 0, "event": name,
            "token_index": position, "token_text": first["tokens"][position],
            "values": values(embeddings[0, position]),
        })
    for cell in range(4):
        for layer in SELECTED_LAYERS:
            rows.append({
                "label": f"exact H / cell {cell} / q{layer} / answer_boundary",
                "source": "phase2589_exact_hidden", "coordinate_kind": "hidden_state_activation",
                "preview": True, "phase": 2589, "layer": layer, "event": "answer_boundary",
                "cell": first["cell_order"][cell], "token_index": first["answer_boundary_token"],
                "values": values(answer_hidden[cell, layer]),
            })
    for group_index, label in enumerate(labels):
        for layer in SELECTED_LAYERS:
            family, language = label.split("/")
            rows.append({
                "label": f"{label} / mean oriented I / q{layer}",
                "source": "phase2589_family_coordinate_profile",
                "coordinate_kind": "hidden_state_factorial_interaction",
                "preview": True, "phase": 2589, "layer": layer,
                "family": family, "language": language, "event": "answer_boundary",
                "surface_replicates": 2,
                "values": values(group_profiles[group_index, layer]),
            })
    return {
        "key": "phase2589_bilingual_operation_exact_parameter_field",
        "model": "Qwen3-4B bilingual natural-operation exact embedding/HiddenState and 20 family profiles",
        "precision": "BF16 forward; exact stored fields float16; two-surface family means float32",
        "coordinate_count": 2560,
        "coordinate_semantics": "Qwen3-4B model-local physical embedding/HiddenState activation coordinate; every row retains all coordinates",
        "coordinate_order": "original physical coordinate order",
        "rows": rows,
    }


def graph_panel(*, centered: bool):
    graphs = np.load(P2589 / "analysis/family_coordinate_graphs.npz")
    labels = [str(item) for item in graphs["labels"]]
    matrix = graphs["language_centered_signed"] if centered else graphs["raw_signed"]
    rows = []
    for layer in range(matrix.shape[0]):
        for group_index, label in enumerate(labels):
            rows.append({
                "label": f"q{layer} / source {label}",
                "source": "phase2589_language_centered_graph" if centered else "phase2589_raw_graph",
                "coordinate_kind": "family_language_group",
                "preview": True, "phase": 2589, "layer": layer, "source_group": label,
                "values": values(matrix[layer, group_index]),
            })
    return {
        "key": ("phase2589_bilingual_operation_centered_family_graph" if centered
                else "phase2589_bilingual_operation_raw_family_graph"),
        "model": ("Qwen3-4B language-centered family-specific coordinate graph" if centered
                  else "Qwen3-4B raw common-task-plus-family coordinate graph"),
        "precision": "Pearson correlation over all 2560 physical coordinates; no Top-K",
        "coordinate_count": len(labels),
        "coordinate_labels": labels,
        "coordinate_semantics": "target family/language node, not a model activation coordinate",
        "coordinate_order": "ten families, English then Chinese",
        "rows": rows,
    }


def dynamics_panel():
    behavior = load(P2588 / "analysis/final.json")
    final = load(P2589 / "analysis/final.json")
    curves = final["coordinate_graph"]["curves"]
    selected = (
        "surface_replicate_signed_median",
        "matched_bilingual_raw_signed_median",
        "unmatched_bilingual_raw_signed_median",
        "matched_bilingual_centered_signed_median",
        "unmatched_bilingual_centered_signed_median",
    )
    rows = [{
        "label": name.replace("_", " "), "source": "phase2589_family_graph_dynamics",
        "coordinate_kind": "hidden_state_checkpoint", "preview": True, "phase": 2589,
        "values": values(curves[name]),
    } for name in selected]
    rows.extend([
        {"label": "Phase2588 full accuracy", "source": "phase2588_behavior",
         "coordinate_kind": "hidden_state_checkpoint", "preview": True, "phase": 2588,
         "values": [behavior["summary"]["conditions"]["full"]["accuracy"]] * 37},
        {"label": "Phase2588 relation-missing accuracy", "source": "phase2588_behavior",
         "coordinate_kind": "hidden_state_checkpoint", "preview": True, "phase": 2588,
         "values": [behavior["summary"]["conditions"]["relation_missing"]["accuracy"]] * 37},
        {"label": "Phase2588 value-missing accuracy", "source": "phase2588_behavior",
         "coordinate_kind": "hidden_state_checkpoint", "preview": True, "phase": 2588,
         "values": [behavior["summary"]["conditions"]["value_missing"]["accuracy"]] * 37},
        {"label": "Phase2588 both-missing accuracy", "source": "phase2588_behavior",
         "coordinate_kind": "hidden_state_checkpoint", "preview": True, "phase": 2588,
         "values": [behavior["summary"]["conditions"]["both_missing"]["accuracy"]] * 37},
    ])
    return {
        "key": "phase2588_2589_bilingual_operation_dynamics",
        "model": "Qwen3-4B bilingual natural-operation behavior and full-coordinate graph dynamics",
        "precision": "behavior accuracy or Pearson correlation over all 2560 coordinates",
        "coordinate_count": 37,
        "coordinate_labels": [f"q{index}" for index in range(37)],
        "coordinate_semantics": "embedding/HiddenState checkpoint; behavior constants are reference rows",
        "coordinate_order": "model depth",
        "rows": rows,
    }


def update_route():
    text = ROUTE.read_text(encoding="utf-8")
    text = text.replace(
        "C39761-C450816 Full-coordinate Fields, Interaction Birth, and Causal Controls",
        "C39761-C491776 Full-coordinate Fields, Interaction Birth, and Bilingual Family Graphs",
    )
    old = (
        "Each panel uses its declared axis: Qwen3-4B has all 2560 embedding/HiddenState coordinates, Qwen3-14B has all 5120, while token, head, causal-condition, and relative-depth axes remain separate. Phase2582-2587 show relation×value interaction emerging after transformer processing, late natural/nonce coordinate-profile convergence, a 4B/14B coarse dynamics correlation, and null behavioral effects from block0 interaction removal. These do not establish shared physical coordinates, a minimal necessary gear, natural-language closure, or a cracked language mechanism."
    )
    new = (
        "Each panel uses its declared axis: Qwen3-4B has all 2560 embedding/HiddenState coordinates, Qwen3-14B has all 5120, while token, head, causal-condition, relative-depth, and family-graph axes remain separate. Phase2582-2590 show interaction birth and late amplification, null behavior from block0 interaction removal, and a 20-node bilingual natural-operation atlas. Raw family correlations are dominated by the shared four-choice task; only language-centered matched-minus-unmatched residuals are weak family-specific evidence. These do not establish shared physical coordinates, a minimal necessary gear, general natural-language abilities, causal family codes, or a cracked language mechanism."
    )
    if old not in text and new not in text:
        raise RuntimeError("route boundary text not found")
    text = text.replace(old, new)
    ROUTE.write_text(text, encoding="utf-8", newline="\n")


def append_memo(result):
    heading = f"## Phase {PHASE}: 双语语言操作族参数场与中心化图谱客户端（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** 将Phase2588—2589重要结果加入现有研究客户端，但分成四种不可混用的坐标轴：Qwen3-4B真实2560物理坐标、20个族/语言图节点、语言中心化的20节点、37个embedding/HiddenState检查点。参数面板显示真实token embedding、四格各cell的answer HiddenState，以及20族在预注册8层上的全部坐标：

$$I_{{ltd}}=H^{{11}}_{{ltd}}-H^{{10}}_{{ltd}}-H^{{01}}_{{ltd}}+H^{{00}}_{{ltd}},\qquad
\widetilde G_{{gld}}=G_{{gld}}-\operatorname{{mean}}_{{h:\,lang(h)=lang(g)}}G_{{hld}}.$$

**测试用例。** 具体参数面板含Phase2589第一个四元组的3行exact embedding、4 cell×8层exact HiddenState、20组×8层全坐标交互均值；raw与中心化图均显示37层×20源节点对20目标节点；动力面板显示5条全坐标相关曲线和Phase2588四个行为条件。

**结果汇总。** `{json.dumps(result, ensure_ascii=False)}`。

**相关文件。** 生成脚本`tests/glm5/phase2590_c483585_c491776_bilingual_family_client_atlas.py`；客户端资产`{ASSET}`；前端路由`{ROUTE}`；构建产物`frontend/dist`；输入来自Phase2588—2589的final、NPZ与SHA-256 raw manifest。

**分析与理论进展。** 客户端现在可以先看20族晚层raw高相关，再切到语言中心化图检查对角优势，同时回到2560列核验实际坐标纹理；因此公共任务场与族特异残差不再被同一热力图混称为“普遍语义齿轮”。

**问题硬伤。** JSON资产进一步增大；参数面板为精确exemplar或两个冻结表面的均值，标签已经区分；图节点不是模型坐标；可视化没有提高观察证据的因果等级。原始3.56GB场已进入参数级图谱并需保留作复算，因此不清理。

**结论。** Phase2589的重要全坐标结果已显示到现有客户端，且物理坐标、层、族节点和因果证据等级分离；仍未得到必要齿轮或语言机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    p2588 = load(P2588 / "analysis/final.json")
    p2589 = load(P2589 / "analysis/final.json")
    if not p2588["all_checks_passed"] or not p2589["all_checks_passed"]:
        raise RuntimeError("input phase checks failed")
    payload = load(ASSET)
    old_bytes = ASSET.stat().st_size
    panels = [exact_parameter_panel(), graph_panel(centered=False), graph_panel(centered=True), dynamics_panel()]
    payload["models"] = [panel for panel in payload["models"] if panel.get("key") not in KEYS] + panels
    payload["phase"] = PHASE
    if CAMPAIGN not in payload.get("campaign", ""):
        payload["campaign"] = f"{payload.get('campaign', '')}; {CAMPAIGN}".strip("; ")
    payload["title"] = "Full-coordinate interaction birth, causal controls, and bilingual language-family atlas"
    payload.setdefault("summary", {})["phase2588_2590"] = {
        "full_behavior_accuracy": p2588["summary"]["conditions"]["full"]["accuracy"],
        "eligible_aligned_quartets": p2588["eligible_aligned_quartets"],
        "field_quartets": p2589["selection"]["selected_quartets"],
        "raw_field_bytes": p2589["field"]["raw_bytes"],
        "late_matched_minus_unmatched_raw_signed": p2589["coordinate_graph"]["late_summary"]["matched_minus_unmatched_raw_signed"],
        "late_matched_minus_unmatched_centered_signed": p2589["coordinate_graph"]["late_summary"]["matched_minus_unmatched_centered_signed"],
        "mechanism_closed": False,
    }
    payload["summary"]["model_rows"] = {panel["key"]: len(panel["rows"]) for panel in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    payload["claim_boundary"] = (
        "Phase2588-2590 add a behavior-qualified 20-node bilingual natural-operation atlas. Every physical parameter row retains "
        "all 2560 Qwen3-4B coordinates; family graph, checkpoint, head, token, and causal-condition axes are declared separately. "
        "High raw late correlations mainly reflect the common four-choice scaffold. The positive language-centered matched-family "
        "advantage is descriptive residual evidence, not a decoder, causal family code, general language ability, or mechanism closure."
    )
    save(ASSET, payload, compact=True)
    update_route()
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "asset": str(ASSET),
        "asset_bytes_before": old_bytes,
        "asset_bytes_after": ASSET.stat().st_size,
        "asset_sha256": sha256(ASSET),
        "panels": [{"key": panel["key"], "coordinate_count": panel["coordinate_count"],
                    "rows": len(panel["rows"])} for panel in panels],
        "raw_field_policy": "retained because parameter-level family results are displayed and provenance/reanalysis are required",
    }
    result["checks"] = {
        "four_new_panels": len(panels) == 4,
        "qwen4_all_2560_coordinates": panels[0]["coordinate_count"] == 2560
                                             and all(len(row["values"]) == 2560 for row in panels[0]["rows"]),
        "exact_embedding_rows_present": any(row["coordinate_kind"] == "embedding_activation" for row in panels[0]["rows"]),
        "exact_hidden_rows_present": any(row["coordinate_kind"] == "hidden_state_activation" for row in panels[0]["rows"]),
        "all_20_family_nodes": panels[1]["coordinate_count"] == panels[2]["coordinate_count"] == 20,
        "raw_and_centered_graphs_separate": panels[1]["key"] != panels[2]["key"],
        "all_37_hidden_checkpoints": len(panels[3]["coordinate_labels"]) == 37,
        "raw_fields_retained": all((ROOT / item["path"]).is_file() for item in load(P2589 / "field/manifest.json")),
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
