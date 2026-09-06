#!/usr/bin/env python3
"""Publish all-88 scaffold-transport lockbox fields to the research client."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2594 = RESULT / "phase2594_c532737_c549120_scaffold_transport_all88_lockbox"
OUT = RESULT / "phase2595_c549121_c557312_all88_transport_client_atlas"
ASSET = RESULT / "client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2595, "C549121-C557312"
KEYS = {
    "phase2594_all88_coordinate_transport_field",
    "phase2594_confirmation69_centered_family_graph",
    "phase2594_all88_transport_dynamics",
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
        block = stream.read(16 * 1024 * 1024)
        while block:
            digest.update(block)
            block = stream.read(16 * 1024 * 1024)
    return digest.hexdigest()


def coordinate_panel():
    arrays = np.load(P2594 / "analysis/all_coordinate_transport_fields.npz")
    sources = (
        ("coordinate correlation / all 88", arrays["coordinate_correlation_all88"], "coordinate_sample_correlation"),
        ("coordinate correlation / confirmation 69", arrays["coordinate_correlation_confirmation69"], "coordinate_sample_correlation"),
        ("coordinate sign agreement / all 88", arrays["coordinate_sign_agreement_all88"], "coordinate_sign_agreement"),
        ("mean interaction / with candidates", arrays["with_candidate_mean_interaction"], "hidden_state_factorial_interaction"),
        ("mean interaction / candidate free", arrays["candidate_free_mean_interaction"], "hidden_state_factorial_interaction"),
    )
    rows = []
    for label, array, kind in sources:
        for layer in LAYERS:
            rows.append({"label": f"{label} / q{layer}", "source": "phase2594_all88_coordinate_transport",
                         "coordinate_kind": kind, "preview": True, "phase": 2594, "layer": layer,
                         "values": values(array[layer])})
    return {
        "key": "phase2594_all88_coordinate_transport_field",
        "model": "Qwen3-4B all-88 per-physical-coordinate scaffold transport",
        "precision": "BF16 forward / float32 interaction / float64 correlations; no Top-K",
        "coordinate_count": 2560,
        "coordinate_semantics": "model-local physical HiddenState coordinate; correlation is computed across frozen quartets per coordinate",
        "coordinate_order": "original physical coordinate order",
        "rows": rows,
    }


def confirmation_graph_panel():
    arrays = np.load(P2594 / "analysis/all_coordinate_transport_fields.npz")
    labels = [str(item) for item in arrays["labels"]]
    matrix = arrays["confirmation_cross_centered"]
    rows = []
    for layer in range(matrix.shape[0]):
        for source, label in enumerate(labels):
            rows.append({"label": f"q{layer} / with-candidate source {label}",
                         "source": "phase2594_confirmation69_centered_graph",
                         "coordinate_kind": "candidate_free_family_language_group",
                         "preview": True, "phase": 2594, "layer": layer, "source_group": label,
                         "values": values(matrix[layer, source])})
    return {
        "key": "phase2594_confirmation69_centered_family_graph",
        "model": "Qwen3-4B independent-69 language-centered scaffold-transport graph",
        "precision": "group profiles and Pearson correlations use all 2560 coordinates",
        "coordinate_count": len(labels), "coordinate_labels": labels,
        "coordinate_semantics": "confirmation-set target family/language group, not a model coordinate",
        "coordinate_order": "18 groups with at least one independent-confirmation quartet",
        "rows": rows,
    }


def dynamics_panel():
    final = load(P2594 / "analysis/final.json")
    paired = final["paired_transport"]
    curves = {**paired["curves"], **paired["coordinate_distributions"],
              **paired["confirmation_group_curves"]}
    rows = [{"label": key.replace("_", " "), "source": "phase2594_all88_transport_dynamics",
             "coordinate_kind": "hidden_state_checkpoint", "preview": True, "phase": 2594,
             "values": values(value)} for key, value in curves.items()]
    return {
        "key": "phase2594_all88_transport_dynamics",
        "model": "Qwen3-4B discovery/confirmation scaffold-transport dynamics",
        "precision": "all-coordinate medians, quantiles, sign agreement, and group-graph correlations",
        "coordinate_count": 37, "coordinate_labels": [f"q{index}" for index in range(37)],
        "coordinate_semantics": "embedding/HiddenState checkpoint",
        "coordinate_order": "model depth", "rows": rows,
    }


def update_route():
    text = ROUTE.read_text(encoding="utf-8")
    text = text.replace(
        "C39761-C532736 Full-coordinate Fields, Bilingual Families, and Scaffold Transport",
        "C39761-C557312 Full-coordinate Fields, Bilingual Families, and Confirmed Scaffold Transport",
    )
    old = (
        "Each panel uses its declared axis: Qwen3-4B has all 2560 embedding/HiddenState coordinates, Qwen3-14B has all 5120, while token, head, causal-condition, relative-depth, family-graph, and scaffold-graph axes remain separate. Phase2582-2593 show interaction birth and late amplification, null behavior from block0 interaction removal, a bilingual natural-operation atlas, and candidate-list removal transport in 19 behavior-qualified groups. Raw correlations remain scaffold-heavy; language-centered matched advantages are descriptive residual evidence. These do not establish shared physical coordinates, a minimal necessary gear, general natural-language abilities, causal family codes, or a cracked language mechanism."
    )
    new = (
        "Each panel uses its declared axis: Qwen3-4B has all 2560 embedding/HiddenState coordinates, Qwen3-14B has all 5120, while token, head, causal-condition, relative-depth, family-graph, and scaffold-graph axes remain separate. Phase2594 expands candidate-list removal transport from 19 discovery quartets to all 88 dual-behavior-qualified quartets with an independent 69-quartet split and every-coordinate distributions. Late transport replicates but is heterogeneous across coordinates. This remains selection-conditioned descriptive evidence, not shared physical coordinates, a minimal necessary gear, general natural-language abilities, causal family codes, or a cracked mechanism."
    )
    if old not in text and new not in text:
        raise RuntimeError("route boundary not found")
    ROUTE.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def append_memo(result):
    heading = f"## Phase {PHASE}: 88锁箱逐坐标迁移与确认族图客户端（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理与公式。** 把Phase2594的重要扩大复验加入客户端：8个预注册层上，逐个显示全部2560坐标的88样本迁移相关、69确认相关、符号一致率及候选有/无平均交互；另显示69确认集的18节点中心化族图和全部逐层曲线。

$$r_{{ld}}=\operatorname{{corr}}_q(I^{{with}}_{{qld}},I^{{free}}_{{qld}}),\qquad
a_{{ld}}=\frac1{{88}}\sum_q\mathbf1[\operatorname{{sign}}I^{{with}}_{{qld}}=\operatorname{{sign}}I^{{free}}_{{qld}}].$$

**测试用例与结果汇总。** 逐坐标参数面板40行×2560列；确认族图666行×18节点；动力面板16行×37检查点。`{json.dumps(result, ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2595_c549121_c557312_all88_transport_client_atlas.py`；客户端资产`{ASSET}`；前端路由`{ROUTE}`；构建产物`frontend/dist`。

**分析与理论进展。** 客户端不再只给整体向量相关，而能检查每个低值物理坐标跨88样本的迁移分布；确认集族节点保持独立轴，避免将18节点图冒充神经元坐标。

**问题硬伤。** 只把预注册8层的2560坐标作为参数热图行，全部37层在NPZ；确认集仍由双行为成功条件筛选；显示不构成因果证明。133MB answer场及Phase2592逐token场均保留用于复算。

**结论。** 全88锁箱的重要逐坐标结果已进入客户端；它支持不均匀分布式迁移纹理，不支持固定语义坐标或机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    final = load(P2594 / "analysis/final.json")
    if not final["all_checks_passed"]:
        raise RuntimeError("Phase2594 checks failed")
    payload = load(ASSET)
    before = ASSET.stat().st_size
    panels = [coordinate_panel(), confirmation_graph_panel(), dynamics_panel()]
    payload["models"] = [panel for panel in payload["models"] if panel.get("key") not in KEYS] + panels
    payload["phase"] = PHASE
    if CAMPAIGN not in payload.get("campaign", ""):
        payload["campaign"] = f"{payload.get('campaign', '')}; {CAMPAIGN}".strip("; ")
    payload["title"] = "Full-coordinate interaction, bilingual family, and confirmed scaffold-transport atlas"
    payload.setdefault("summary", {})["phase2594_2595"] = {
        "dual_quartets": 88, "independent_confirmation": 69,
        "late_confirmation_same_prefix": final["paired_transport"]["late_summary"]["independent_confirmation69_signed_median"],
        "late_coordinate_correlation_median": final["paired_transport"]["late_summary"]["confirmation69_coordinate_correlation_median"],
        "late_coordinate_sign_agreement": final["paired_transport"]["late_summary"]["coordinate_sign_agreement_mean"],
        "mechanism_closed": False,
    }
    payload["summary"]["model_rows"] = {panel["key"]: len(panel["rows"]) for panel in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    payload["claim_boundary"] = (
        "Phase2594-2595 expand candidate-scaffold transport to all 88 dual-behavior-qualified quartets and a "
        "disjoint 69-quartet confirmation split. Full-physical-coordinate panels reveal heterogeneous but broad late "
        "transport, and the confirmation family graph preserves a centered matched advantage. The evidence is still "
        "conditioned on successful structured lookup and is neither causal nor a general language code."
    )
    save(ASSET, payload, compact=True)
    update_route()
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "asset": str(ASSET), "asset_bytes_before": before, "asset_bytes_after": ASSET.stat().st_size,
        "asset_sha256": sha256(ASSET),
        "panels": [{"key": panel["key"], "coordinate_count": panel["coordinate_count"],
                    "rows": len(panel["rows"])} for panel in panels],
        "raw_field_policy": "retained because all-coordinate confirmation fields are displayed and reanalysis is required",
    }
    result["checks"] = {
        "three_new_panels": len(panels) == 3,
        "all_coordinate_rows_2560": panels[0]["coordinate_count"] == 2560
                                    and all(len(row["values"]) == 2560 for row in panels[0]["rows"]),
        "confirmation18_nodes": panels[1]["coordinate_count"] == 18,
        "all37_checkpoints": panels[2]["coordinate_count"] == 37,
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
