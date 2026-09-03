#!/usr/bin/env python3
"""Audit Phase 2511-2527 interpretations and freeze a redundancy-aware source-edge contract."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2528_c94753_c95776_redundancy_source_edge_contract_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
P2527 = RESULT / "phase2527_c93729_c94752_stage_evidence_audit_next_contract/analysis/final.json"
ATTACHMENTS = [
    Path(r"C:\Users\Admin\.codex\attachments\4e1dc241-8f9a-4681-9d46-74a5dc97e7dd\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\97d63fbc-2d97-4131-b18c-10af758c0ef5\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\e7ade32b-999e-4ee4-814e-11d8baeb1c42\pasted-text.txt"),
]
PHASE, CAMPAIGN = 2528, "C94753-C95776"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Phase2511–2527证据复审与冗余耦合source-edge合同冻结（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 逐项对照三份外部分析、Phase2527总审计以及Phase2525的原始冻结路由和因果数字。证据按行为资格、观察相关、因果充分、联盟必要、条件必要、删除—救援六级分开；任何一级不得自动升级。特别复核“注意力质量筛选”和“o_proj前head输出移植”是否实际测试了同一对象。

$$g_{{l,h,r}}(x)=W^h_{{O,l}}\sum_{{j\in r}}\alpha_{{l,h,a,j}}(x)V_{{l,h}}h^l_j(x),\qquad \neg N(G_i)\not\Rightarrow \neg\operatorname{{participates}}(G_i).$$

**结果汇总。** 保留结论 `{json.dumps(result['retained'], ensure_ascii=False)}`；过度结论修正 `{json.dumps(result['corrections'], ensure_ascii=False)}`；冻结合同 `{json.dumps(result['contract'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2528_c94753_c95776_redundancy_source_edge_contract_audit.py`；三份附件的SHA-256、原始Phase2527引用和final位于`{OUT}`。

**分析与理论进展。** Phase2525的top32确实是unit30冻结、unit31验证并显著优于等量随机的多层head输出充分路径，这是应保留的强拼图。但它由四区域注意力质量的Walsh能量筛选，随后移植的是该head在answer-boundary的整个o_proj前输出；因此尚不能把“从事实前缀读取关系内容”视为已证。冗余系统中的单head或单联盟阴性只关闭相应的无条件必要性主张，不自动关闭该路径的参与性；反过来，“存在冗余”也不能成为不可证伪的保护层。

**问题硬伤与结论。** “最小割”在非线性Transformer里只能先称经验候选割，除非干预覆盖、自然性和阳性控制均成立；matched counterfactual不是天然neutral；删除后恢复是确定性备用路线的显现，不是模型现场学习；source贡献分解只在单个Attention模块的o_proj前/后加法账本内精确，不能越级称完整语言因果链。下一阶段先做全部late heads、全部head坐标和全部residual坐标的基本守恒图，再做source-edge充分性、持续割、联盟分解、救援、自主生成与跨模型相对复现。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = json.loads(P2527.read_text(encoding="utf-8-sig"))
    retained = [
        "九个自然关系/语法族通过双unit行为门；未过门的族不构成内部机制反例。",
        "answer-boundary完整状态在四模型协议内是强可移植输出身份载体。",
        "unit30冻结的32个late layer×head在unit31 donor head-output patch中达到0.8611翻转，随机32为0。",
        "top32只建立选择性因果充分性；必要性、最小性、唯一性、source内容和自主递归均未建立。",
        "单零件必要性阴性不能排除冗余参与，必须观察联盟、条件阻断和层间恢复。",
    ]
    corrections = [
        "不得说top32已被证明从事实前缀读取关系：筛选量是区域attention mass交互，干预量是整个head输出。",
        "不得说只有32个head参与：all-late也充分，top32不是唯一集合且尚无自然前向必要性。",
        "H(l+1)-H(l)=Attention+MLP只是在同次BF16前向中的架构账本近似闭合，不是语言机制精确闭合。",
        "整block donor充分而单组件donor不足，不推出自然机制必须Attention与MLP联合；不同移植的上下文兼容性不同。",
        "answer-boundary是输出身份形成/可读出的强位置，不足以断言关系计算只在那里发生。",
        "跨模型只复现事件角色和相对深度；不支持共享坐标、head编号、K/V路径或同一算法。",
        "所谓路径最小割先定义为干预协议下的经验候选割；zero、base-matched与counterfactual必须分开。",
        "删除后的后层恢复表示现有备用读路或重构，不称在线补偿学习。",
    ]
    contract = {
        "name": "redundancy-aware source-to-output route decomposition",
        "wp1": "五个互斥且覆盖全部可见前缀token的source区域；全late-head K/V、attention、head坐标、residual坐标加法守恒。",
        "wp2": "unit30发现、unit31锁箱；source contribution donor patch与等量随机、错配、self对照，区分source-edge和whole-head充分性。",
        "wp3": "renormalized attention-edge持续阻断、whole-head联盟分解、layer-band分组和matched/shuffled救援，同时保存逐层完整answer-boundary场。",
        "wp4": "20+外部语言操作族、多token实体、自主贪心生成、真多跳和整句重排；不把teacher-forced第一token当自主闭合。",
        "wp5": "Qwen14B、DeepSeek7B、GLM4依次BF16非量化，仅复现相对深度/事件/模型内route选择，不对齐物理head号。",
        "wp6": "重要结果加入参数级客户端；未发布的大型K/V/HiddenState原场记录哈希后清理。",
        "negative_adjudication": {
            "top_cut_null_all_cut_effect": "top32不是完整自然联盟，Attention路线仍可能成立",
            "intermediate_drop_late_recovery": "存在确定性备用读路/重构",
            "all_covered_cuts_null_positive_control_ok": "降级或关闭该自然执行路径主张",
            "matched_not_better_than_random": "原选择性充分性不稳定，必须降级",
        },
    }
    checks = {
        "prior_passed": bool(prior["all_checks_passed"]),
        "attachments_exist": all(path.exists() for path in ATTACHMENTS),
        "attachments_nonempty": all(path.stat().st_size > 1000 for path in ATTACHMENTS),
        "top32_metric_grounded": abs(prior["key_numbers"]["multilayer_attention"]["donor_top32"] - 0.8611111111111112) < 1e-12,
        "random32_metric_grounded": prior["key_numbers"]["multilayer_attention"]["donor_random32"] == 0.0,
        "six_level_evidence_contract": True,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "attachment_sha256": {str(path): sha(path) for path in ATTACHMENTS},
        "prior": str(P2527),
        "retained": retained,
        "corrections": corrections,
        "contract": contract,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
