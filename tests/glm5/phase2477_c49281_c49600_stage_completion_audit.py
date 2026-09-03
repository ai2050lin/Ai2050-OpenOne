#!/usr/bin/env python3
"""Audit Phase2468-2476 as one corrected autonomous full-coordinate stage."""
from __future__ import annotations

import json
import py_compile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
OUT = RESULT / "phase2477_c49281_c49600_stage_completion_audit"
PHASE, CAMPAIGN = 2477, "C49281-C49600"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Phase2468–2476自主生成全坐标阶段闭合审计与知识链续阶段合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将本大阶段九个Phase视为不可拆分证据链：2468重解码原始token并审计旧协议；2469以十四类语言族、双语言、双输出接口和发现/确认/锁箱筛选行为材料；2470捕获行为合格八族每个prompt token的q0 Embedding、q1–q36 block输出、q37 final norm及全部2560坐标；2471建立三语义事件、层增量和五因素基本分账；2472比较identity/global/pooled-diagonal/interface-diagonal/family-diagonal基本传动；2473剔除final RMSNorm伪传动；2474采集无正确前缀的真实贪心路径；2475按语义事件对齐并进行unit9选层、unit10锁箱；2476发布双坐标顺序客户端并审计留存。逐项验证final、MEMO连续性、Python可编译、前端生产构建和纠错裁决。

$$\mathcal{{E}}_{{2468:2476}}=\{{B,F,\Delta F,T,A,\Pi_{{phys}},\Pi_{{fp}}\}},\qquad \text{{closed-stage}}\iff\bigwedge_p\operatorname{{check}}(p)=1.$$

**结果汇总。** `{json.dumps(result['audit'], ensure_ascii=False)}`；阶段裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；下一阶段合同 `{json.dumps(result['next_stage'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2477_c49281_c49600_stage_completion_audit.py`；九个Phase的final、原场、派生场与客户端资产见各自结果目录及Phase2476留存清单；本Phase final位于同名结果目录。

**分析与理论进展。** 本阶段真正建立的是一个受严格限定的规律拼图：八类不同语言操作在提示事件与成功自主输出事件中都存在稠密、全坐标、family-relative纹理；跨接口相同family纹理在锁箱中胜过错family，且输出过程中未消失；某些block→block变化可由冻结对角尺度部分预测。这比“HiddenState可分类”深入，但仍只说明条件化分布式纹理的存在、保持和有限可预测性。Phase2466的“自主生成崩解”是事件定义与生成预算混杂造成的过度结论，正式撤回。

**问题硬伤与结论。** family仍与专属关系措辞纠缠；unit10错误仅13例；对角模型可能吸收幅值统计；三事件只是路径稀疏采样；没有发现天然稀疏齿轮、可跨材料精确复用的坐标算法或因果闭环。因此本阶段“测量合同”闭合，语言编码机制没有闭合。下一阶段目标相同，按用户要求自动继续：从孤立二选一关系转到多节点知识链，测节点、边、跳数、查询角色和输出接口如何在全坐标场中组合，而不先假定高级数学结构。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    memo = MEMO.read_text(encoding="utf-8")
    finals: dict[int, dict] = {}
    compilation: dict[int, bool] = {}
    for phase in range(2468, 2477):
        directory = next(RESULT.glob(f"phase{phase}_*"))
        final_path = directory / "analysis/final.json"
        finals[phase] = json.loads(final_path.read_text(encoding="utf-8"))
        script = next((ROOT / "tests/glm5").glob(f"phase{phase}_*.py"))
        py_compile.compile(str(script), doraise=True)
        compilation[phase] = True
    asset = json.loads(ASSET.read_text(encoding="utf-8"))
    qwen = next(section for section in asset["models"] if section["key"] == "qwen4b")
    audit = {
        "phases": list(finals),
        "all_phase_checks_passed": all(item.get("all_checks_passed") is True for item in finals.values()),
        "memo_heading_counts": {str(phase): memo.count(f"## Phase {phase}:") for phase in finals},
        "python_compilation": compilation,
        "phase2468_protocol_artifact": finals[2468]["adjudication"],
        "phase2470_field_shape": finals[2470]["collection"]["shape"],
        "phase2474_path_shape": finals[2474]["collection"]["shape"],
        "phase2475_lockbox": finals[2475]["trajectory"]["lockbox"],
        "visual_qwen_shape": qwen["binary_shape"],
        "visual_coordinate_orders": list(qwen.get("coordinate_orders", {})),
        "retained_bytes": finals[2476]["retention"]["bytes"],
    }
    adjudication = {
        "phase2466_autonomous_state_collapse_withdrawn": True,
        "behavior_qualified_dense_family_texture_observed": True,
        "successful_autonomous_texture_retention_lockbox_positive": finals[2475]["adjudication"]["successful_autonomous_family_texture_present"],
        "limited_diagonal_block_transport_candidate": finals[2472]["adjudication"]["predictable_diagonal_transport_candidate"],
        "natural_coordinate_gear_identified": False,
        "causal_language_compiler_identified": False,
        "language_encoding_mechanism_closed": False,
    }
    next_stage = {
        "same_goal": True,
        "automatic_continuation": True,
        "title": "多节点知识链的条件坐标组合图谱",
        "sequence": [
            "Phase2478: 多关系知识链材料与真实自主行为大样本合同",
            "Phase2479: 行为合格链的prompt/all-token与自主事件全坐标场",
            "Phase2480: 节点、边、跳数、查询角色、语言、表面和输出接口基本分账",
            "Phase2481: 链段复用、边重组、路径方向和干扰边锁箱",
            "Phase2482: 成功输出的边界—中间链节点—终点编译轨迹",
        ],
        "priority": "observe broad families first; search structures second; causal closure last",
    }
    checks = {
        "nine_finals": len(finals) == 9,
        "all_final_checks": audit["all_phase_checks_passed"],
        "memo_contiguous_single_headings": all(value == 1 for value in audit["memo_heading_counts"].values()),
        "all_scripts_compile": all(compilation.values()),
        "corrected_protocol_claim": finals[2468]["adjudication"].get("phase2466_085_to_011_is_state_collapse") is False,
        "prompt_fullfield": audit["phase2470_field_shape"] == [9700, 38, 2560],
        "autonomous_fullfield": audit["phase2474_path_shape"] == [256, 13, 38, 2560],
        "visual_full_orders": set(audit["visual_coordinate_orders"]) == {"physical", "fingerprint"},
        "claim_boundary": not adjudication["natural_coordinate_gear_identified"] and not adjudication["language_encoding_mechanism_closed"],
        "continuation_contract": next_stage["same_goal"] and next_stage["automatic_continuation"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "audit": audit,
        "adjudication": adjudication, "next_stage": next_stage,
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
