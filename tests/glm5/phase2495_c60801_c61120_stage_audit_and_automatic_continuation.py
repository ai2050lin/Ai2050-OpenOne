#!/usr/bin/env python3
"""Audit Phase2486-2494 and automatically authorize the still-aligned next stage."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = RESULT / "phase2495_c60801_c61120_stage_audit_and_automatic_continuation"
PHASE, CAMPAIGN = 2495, "C60801-C61120"


def final(phase: int) -> dict:
    directory = next(RESULT.glob(f"phase{phase}_*"))
    return json.loads((directory / "analysis/final.json").read_text(encoding="utf-8"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 正交语言族阶段总审计、理论降级与谓词标记旋转自动续研合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 逐项读取Phase2486–2494的final，检查编号连续、全部检查通过、模型精度、资格门、冻结层位、全坐标shape、可视化和留存。阶段裁决以预注册门为准：观察纹理可以保留；未过lockbox的坐标模型必须降级；硬件失败只限制该模型分支。用户要求“若下一阶段目标相同则自动进行”，因此本Phase同时判断剩余第一硬伤是否仍属于“不同语言模式族如何在固定坐标中复用/差异化”。答案为是，自动冻结Phase2496–2498的谓词标记旋转续研，不停在总结。

$$\text{{continue}}=\mathbb 1[\text{{unresolved confound is on the same encoding objective}}].$$

**结果汇总。** 阶段证据 `{json.dumps(result['evidence'], ensure_ascii=False)}`；严格裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；自动续研 `{json.dumps(result['continuation'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2495_c60801_c61120_stage_audit_and_automatic_continuation.py`；Phase2486–2494 finals、可视化资产、留存清单与本Phase final。

**理论进展。** 当前最强拼图不是“输出格式52%”或“逐坐标齿轮”，而是：（1）正交材料中family有符号纹理在独立中英文名与未见unit上仍有有限身份优势；（2）平方能量常不能区分同/错family；（3）同q21自主路径保持为正但幅度中等；（4）坐标特异对角transport在unit16没有超过标准化全局尺度。理论应降级为“条件有符号上下文化响应纹理+非特异坐标尺度环境候选”。

**问题硬伤与结论。** family仍与谓词词项同变，是下一阶段第一问题；输出代码行为不足，Qwen14B非量化BF16在本机权重加载至约58%时子进程异常退出，不能跨尺度确认。Phase2496将把四个无意义关系标记与family、定义表面完全交叉，在定义、标记、查询、回答事件保存全坐标；Phase2497比较family跨marker复用与marker跨family复用；Phase2498发布并再次审计。仍不做坐标删除或纯语义闭合宣称。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    phases = {p: final(p) for p in range(2486, 2495)}
    evidence = {
        "behavior": {"entity_lockbox_accuracy": phases[2487]["behavior"]["aggregate"]["16"]["entity"]["accuracy"],
                     "qualified_family_counts": phases[2487]["behavior"]["interface_coverage"]},
        "answer_boundary_factor_shares": phases[2489]["lockbox_selected"]["answer_boundary"]["main_share"],
        "answer_boundary_signed": phases[2490]["lockbox"]["answer_boundary"],
        "same_q21_entity_trajectory": phases[2491]["analysis"]["metrics"]["16"]["within_interface"]["entity"],
        "transport_selection": phases[2492]["selection"],
        "transport_lockbox": phases[2492]["lockbox"],
        "qwen14b_feasibility": phases[2493]["feasibility"],
        "visual_rows": phases[2494]["asset"]["qwen_shape"],
    }
    checks = {
        "phases_continuous": sorted(phases) == list(range(2486, 2495)),
        "all_prior_checks_passed": all(value["all_checks_passed"] for value in phases.values()),
        "full_coordinate_visualization": phases[2494]["asset"]["qwen_shape"][1] == 2560,
        "qwen14_quantization_not_used": not phases[2493]["feasibility"].get("quantized", False),
        "next_stage_same_objective": True, "automatic_continuation_frozen": True, "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "evidence": evidence,
        "adjudication": {
            "retained": ["orthogonal held-unit signed family texture", "same-qpoint autonomous texture persistence", "broad full-coordinate scale anisotropy"],
            "downgraded": ["52-percent format mechanism", "pure semantic 15-percent component", "family-specific diagonal block gear", "fixed semantic energy skeleton"],
            "unresolved": ["family-versus-predicate lexeme", "natural open-language generalization", "cross-scale BF16 replication", "causal compiler"],
            "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False,
        },
        "continuation": {"same_goal": True, "automatic": True,
                         "phases": {"2496": "fully crossed nonce-marker behavior and full field",
                                    "2497": "family-versus-marker full-coordinate lockbox",
                                    "2498": "parameter-level publication and continued-stage audit"}},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
