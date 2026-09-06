#!/usr/bin/env python3
"""Publish strict matched-delta causal evidence to the existing research client."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2597 = RESULT / "phase2597_c573697_c590080_delta_matched_scaffold_causal_controls"
OUT = RESULT / "phase2598_c590081_c598272_matched_delta_causal_client_atlas"
ASSET = RESULT / "client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2598, "C590081-C598272"
KEYS = {"phase2597_matched_delta_exact_parameter_field", "phase2597_matched_delta_causal_conditions"}

sys.path.insert(0, str(TESTS))
import phase2596_c557313_c573696_fullcoordinate_scaffold_causal_walsh as p2596  # noqa: E402
import phase2597_c573697_c590080_delta_matched_scaffold_causal_controls as p2597  # noqa: E402


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


def exact_delta_panel():
    item = p2596.material_and_targets()[0]
    rows = []
    for qpoint in p2597.QPOINTS:
        rows.extend([
            {"label": f"exact candidate-free I / q{qpoint}", "source": "phase2597_exact_delta",
             "coordinate_kind": "hidden_state_factorial_interaction", "preview": True,
             "phase": 2597, "layer": qpoint, "condition": "candidate_free",
             "values": values(item["interactions"][1, qpoint])},
            {"label": f"exact with-candidate I / q{qpoint}", "source": "phase2597_exact_delta",
             "coordinate_kind": "hidden_state_factorial_interaction", "preview": True,
             "phase": 2597, "layer": qpoint, "condition": "with_candidate",
             "values": values(item["interactions"][0, qpoint])},
        ])
        for kind in p2597.KINDS:
            rows.append({"label": f"exact {kind} / q{qpoint}", "source": "phase2597_exact_delta",
                         "coordinate_kind": "hidden_state_walsh_delta", "preview": True,
                         "phase": 2597, "layer": qpoint, "condition": kind,
                         "values": values(p2597.intervention_delta(item, qpoint, kind))})
    return {
        "key": "phase2597_matched_delta_exact_parameter_field",
        "model": "Qwen3-4B exact full-coordinate scaffold-interaction deltas and controls",
        "precision": "BF16 source states stored float16; delta construction float32",
        "coordinate_count": 2560,
        "coordinate_semantics": "model-local physical HiddenState activation coordinate; every delta uses all coordinates",
        "coordinate_order": "original physical coordinate order",
        "rows": rows,
    }


def causal_condition_panel():
    final = load(P2597 / "analysis/final.json")
    conditions = final["design"]["conditions"]
    base_margin = final["behavior"]["baseline"]["mean_target_margin"]
    rows = [
        {"label": "complete-candidate accuracy", "source": "phase2597_causal",
         "coordinate_kind": "causal_condition", "preview": True, "phase": 2597,
         "values": [final["behavior"][condition]["accuracy"] for condition in conditions]},
        {"label": "target-margin change vs baseline", "source": "phase2597_causal",
         "coordinate_kind": "causal_condition", "preview": True, "phase": 2597,
         "values": [final["behavior"][condition]["mean_target_margin"] - base_margin for condition in conditions]},
        {"label": "changed predictions vs baseline", "source": "phase2597_causal",
         "coordinate_kind": "causal_condition", "preview": True, "phase": 2597,
         "values": [final["behavior"][condition]["changed_predictions_vs_baseline"] for condition in conditions]},
        {"label": "full-vocabulary next-token factorial RMS", "source": "phase2597_next_token",
         "coordinate_kind": "causal_condition", "preview": True, "phase": 2597,
         "values": [final["next_token_field"][condition]["median_factorial_rms"] for condition in conditions]},
        {"label": "next-token factorial correlation to baseline", "source": "phase2597_next_token",
         "coordinate_kind": "causal_condition", "preview": True, "phase": 2597,
         "values": [final["next_token_field"][condition]["median_correlation_to_baseline"] for condition in conditions]},
        {"label": "median perturbation RMS", "source": "phase2597_diagnostics",
         "coordinate_kind": "causal_condition", "preview": True, "phase": 2597,
         "values": [0.0] + [final["intervention_diagnostics"][condition]["perturbation_rms"]
                            for condition in conditions[1:]]},
    ]
    return {"key": "phase2597_matched_delta_causal_conditions",
            "model": "Qwen3-4B strict equal-delta-norm q25/q35 Walsh causal controls",
            "precision": "BF16 intervention; complete candidate likelihood and 151936-vocabulary logits",
            "coordinate_count": len(conditions), "coordinate_labels": conditions,
            "coordinate_semantics": "experimental condition, not a physical model coordinate",
            "coordinate_order": "baseline; q25 five matched controls; q35 five matched controls",
            "rows": rows}


def update_route():
    text = ROUTE.read_text(encoding="utf-8")
    text = text.replace(
        "C39761-C557312 Full-coordinate Fields, Bilingual Families, and Confirmed Scaffold Transport",
        "C39761-C598272 Full-coordinate Fields, Confirmed Transport, and Matched-Delta Causality",
    )
    old = (
        "Each panel uses its declared axis: Qwen3-4B has all 2560 embedding/HiddenState coordinates, Qwen3-14B has all 5120, while token, head, causal-condition, relative-depth, family-graph, and scaffold-graph axes remain separate. Phase2594 expands candidate-list removal transport from 19 discovery quartets to all 88 dual-behavior-qualified quartets with an independent 69-quartet split and every-coordinate distributions. Late transport replicates but is heterogeneous across coordinates. This remains selection-conditioned descriptive evidence, not shared physical coordinates, a minimal necessary gear, general natural-language abilities, causal family codes, or a cracked mechanism."
    )
    new = (
        "Each panel declares its axis: Qwen3-4B/Qwen3-14B physical coordinates, token, head, causal condition, depth, family graph, or scaffold graph. Phase2594 confirms heterogeneous late scaffold transport across all 88 qualified quartets. Phase2597 adds q25/q35 all-2560-coordinate Walsh interventions with strictly delta-norm-matched rolls and wrong-family controls: the transported delta improves margin, while zeroing and wrong-family deltas selectively damage behavior. This is coupled four-prompt causal evidence, not a single-prompt natural gear, shared cross-model coordinate code, or complete language compiler."
    )
    if old not in text and new not in text:
        raise RuntimeError("route boundary not found")
    ROUTE.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def append_memo(result):
    heading = f"## Phase {PHASE}: 严格等Δ全坐标因果参数与条件客户端（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理与公式。** 将Phase2597的重要严格等$\Delta$因果结果加入现有客户端。参数面板显示同一冻结四元组在q25/q35的候选有/无$I$、真实移植$\Delta_T$、等范数坐标roll、归零$\Delta_0$及其roll、等范数异族$\Delta$的全部2560物理坐标：

$$H'_i=H_i+\frac{{c_i}}4\Delta,\qquad c=(+1,-1,-1,+1).$$

因果条件面板把11条件的完整序列准确率/margin/翻转数、完整词表二阶RMS/相关及实际扰动RMS并列，且条件轴不冒充模型坐标。

**测试用例与结果汇总。** 参数面板14行×2560列；因果面板6行×11条件。`{json.dumps(result, ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2598_c590081_c598272_matched_delta_causal_client_atlas.py`；客户端资产`{ASSET}`；前端路由`{ROUTE}`；构建产物`frontend/dist`；原始2860候选分数与869MB完整词表场保留在Phase2597。

**分析与理论进展。** 客户端能直接核验“真实$\Delta$与等范数错误方向”在每个物理坐标上的差异，并把实现扰动、行为效应与输出词表场放在独立证据轴上。

**问题硬伤。** exact参数只显示一个冻结四元组，统计结果来自65组；显示不消除四prompt联合干预、成功样本选择和双层窗口限制；可视化不提高因果外推等级。

**结论。** Phase2597的方向性因果结果已进入参数级客户端；支持后层选择性使用完整坐标联合方向，不支持单prompt语义齿轮或语言机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    final = load(P2597 / "analysis/final.json")
    if not final["all_checks_passed"]:
        raise RuntimeError("Phase2597 checks failed")
    payload = load(ASSET)
    before = ASSET.stat().st_size
    panels = [exact_delta_panel(), causal_condition_panel()]
    payload["models"] = [panel for panel in payload["models"] if panel.get("key") not in KEYS] + panels
    payload["phase"] = PHASE
    if CAMPAIGN not in payload.get("campaign", ""):
        payload["campaign"] = f"{payload.get('campaign', '')}; {CAMPAIGN}".strip("; ")
    payload["title"] = "Full-coordinate interaction, confirmed scaffold transport, and matched-delta causal atlas"
    payload.setdefault("summary", {})["phase2596_2598"] = {
        "causal_quartets": 65,
        "q25_transplant_margin_gain": final["effects_vs_baseline"]["q25_transplant_delta"]["margin_delta"],
        "q35_transplant_margin_gain": final["effects_vs_baseline"]["q35_transplant_delta"]["margin_delta"],
        "q25_zero_accuracy_delta": final["effects_vs_baseline"]["q25_zero_delta"]["accuracy_delta"],
        "q35_zero_accuracy_delta": final["effects_vs_baseline"]["q35_zero_delta"]["accuracy_delta"],
        "mechanism_closed": False,
    }
    payload["summary"]["model_rows"] = {panel["key"]: len(panel["rows"]) for panel in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    payload["claim_boundary"] = (
        "Phase2597-2598 add strict equal-delta-norm causal controls at q25/q35. The true transported full-coordinate "
        "delta improves target margin; equal-norm rolls do not, zeroing is selectively harmful relative to its equal-delta "
        "roll, and wrong-family deltas reduce accuracy. This supports conditional use of a distributed interaction direction "
        "inside the structured task, not a single-prompt natural semantic gear or complete language compiler."
    )
    save(ASSET, payload, compact=True)
    update_route()
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "asset": str(ASSET), "asset_bytes_before": before, "asset_bytes_after": ASSET.stat().st_size,
              "asset_sha256": sha256(ASSET),
              "panels": [{"key": panel["key"], "coordinate_count": panel["coordinate_count"],
                          "rows": len(panel["rows"])} for panel in panels],
              "raw_field_policy": "retained because matched-delta parameter and causal results are displayed and reanalysis is required"}
    result["checks"] = {"two_new_panels": len(panels) == 2,
                        "all_exact_delta_rows_2560": panels[0]["coordinate_count"] == 2560
                                                     and all(len(row["values"]) == 2560 for row in panels[0]["rows"]),
                        "all_11_causal_conditions": panels[1]["coordinate_count"] == 11,
                        "physical_and_condition_axes_separate": panels[0]["coordinate_semantics"] != panels[1]["coordinate_semantics"],
                        "claim_boundary": True}
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
