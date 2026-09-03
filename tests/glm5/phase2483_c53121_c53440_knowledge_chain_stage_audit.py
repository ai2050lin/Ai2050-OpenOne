#!/usr/bin/env python3
"""Close Phase2478-2482 and authorize the surface-confound continuation."""
from __future__ import annotations

import json
import py_compile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = RESULT / "phase2483_c53121_c53440_knowledge_chain_stage_audit"
PHASE, CAMPAIGN = 2483, "C53121-C53440"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Phase2478–2482多节点知识链续阶段闭合审计与表面混杂续研究合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将五个Phase作为完整链审计：2478以1152条八族、1–3跳、双语双接口终点任务暴露协议失败；2479要求完整四节点路径，分离链遍历与代码编译；2480只围绕确认/锁箱通过的part-whole、causal、handoff四个family-surface组合，捕获2187 prompt token及24×49自主事件的38×2560原场；2481建立主/干扰/生成节点的基本护照并用unit12选层、unit13锁箱；2482发布132条参数级切片、第三种完整坐标顺序并留存850,422,656字节唯一场。验证final、MEMO、编译、前端与结论边界。

$$\operatorname{{stage}}_{{chain}}=B_{{1152}}\rightarrow B_{{path}}\rightarrow F_{{all-token/path}}\rightarrow A_{{node/distractor}}\rightarrow V_{{2560}}.$$

**结果汇总。** 审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；综合裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；续阶段 `{json.dumps(result['next_stage'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2483_c53121_c53440_knowledge_chain_stage_audit.py`；Phase2478–2482全部材料、原场、派生场、客户端资产及final。

**分析与理论进展。** 终点直接选择失败而逐步路径在三个family的特定表面通过，说明“小模型是否执行长链”高度依赖输出协议和外显工作记忆；这不能被简化成内部有/无多跳机制。通过材料的unit13中，family同一性跨语言0.862、相邻节点0.314、prompt→generated 0.857、main-minus-distractor跨语言0.785，均胜三family错配；50%能量需163坐标、90%需1536坐标，呈头部集中加低值长尾，而不是少数坐标齿轮。

**问题硬伤与结论。** 只有三family、两个错配、causal表面数多一倍，且relation wording与family同一；跨语言还复用相同英文专名。因而“普遍链编码”尚未成立。下一阶段目标相同并自动继续，但第一任务不是更高数学，而是用现有causal s0/s2做跨表面锁箱，检查query-path与family纹理是否跨事实顺序保存；随后再扩展真正正交的表面材料。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    memo = MEMO.read_text(encoding="utf-8"); finals = {}; compilation = {}
    for phase in range(2478, 2483):
        directory = next(RESULT.glob(f"phase{phase}_*")); finals[phase] = json.loads((directory / "analysis/final.json").read_text(encoding="utf-8"))
        script = next((ROOT / "tests/glm5").glob(f"phase{phase}_*.py")); py_compile.compile(str(script), doraise=True); compilation[phase] = True
    audit = {
        "phases": list(finals), "all_phase_checks_passed": all(value["all_checks_passed"] for value in finals.values()),
        "memo_counts": {str(phase): memo.count(f"## Phase {phase}:") for phase in finals}, "compilation": compilation,
        "phase2478_qualified_family_hops": finals[2478]["adjudication"]["qualified_family_hops"],
        "phase2479_dual_interface_qualified": finals[2479]["adjudication"]["qualified_family_surfaces"],
        "phase2480_behavior": finals[2480]["quality"],
        "phase2481_lockbox": finals[2481]["analysis"]["unit13_lockbox"],
        "phase2481_density": finals[2481]["analysis"]["density"],
        "phase2482_asset_shape": finals[2482]["asset"]["qwen_shape"],
    }
    adjudication = {
        "direct_endpoint_protocol_supported": False, "externalized_path_subset_supported": True,
        "three_family_chain_texture_candidate": True, "universal_chain_code_identified": False,
        "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False,
    }
    next_stage = {
        "same_goal": True, "automatic_continuation": True,
        "first_test": "causal surface0 versus surface2 full-coordinate node and query-path contrast lockbox",
        "reason": "separate relation-family identity from fact order/interleaving before adding mathematics or causality",
    }
    checks = {
        "five_finals": len(finals) == 5, "all_final_checks": audit["all_phase_checks_passed"],
        "memo_contiguous": all(value == 1 for value in audit["memo_counts"].values()), "all_compile": all(compilation.values()),
        "failure_not_hidden": audit["phase2478_qualified_family_hops"] == 0 and audit["phase2479_dual_interface_qualified"] == 0,
        "fullfield_retained": finals[2480]["collection"]["prompt_field"]["shape"] == [2187, 38, 2560],
        "visual_built": finals[2482]["frontend"]["dist_newer"], "claim_boundary": not adjudication["language_encoding_mechanism_closed"],
        "continuation": next_stage["same_goal"] and next_stage["automatic_continuation"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "audit": audit, "adjudication": adjudication, "next_stage": next_stage, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
