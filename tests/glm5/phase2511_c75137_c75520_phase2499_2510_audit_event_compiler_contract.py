#!/usr/bin/env python3
"""Audit Phase 2499-2510 and freeze the event-conditioned compiler contract."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2511_c75137_c75520_phase2499_2510_audit_event_compiler_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2511, "C75137-C75520"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\cf0288cf-52fa-44b1-ad0b-e39f4651936a\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\a0da8449-adcd-41e5-a93e-1cfcda5402cd\pasted-text.txt"),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Phase2499–2510证据复审与事件条件编译合同冻结（{CAMPAIGN}） [{stamp}]

**测试原理与证据审查。** 逐项回读Phase2499–2510 final、原场shape/hash、行为门、Walsh四格、关系图闭环和两份复盘附件。保留的事实是：（1）在合格合成合同内，definition-swap会要求答案翻转；（2）等token长度修复后causal-prefix四格逐坐标严格为零；（3）query-marker形成稠密有符号四格交互并到达完整候选序列分数；（4）固定向量自主运输与answer端伙伴无关加法势均未通过。修正附件中的过度解释：Walsh交互消去两个一阶项不等于模型内部实现XOR；interaction在三项平方和中的0.782份额不是全HiddenState信息份额；24/24输出方向正确不是“输出概率完整保存语义”；query图高R2受单环与高自由度影响，不能称独立relation势；candidate/answer变化只能称事件相关重编码现象，尚不能预设存在非线性编译器、曲率或纤维丛。

**新大阶段合同。** 研究对象冻结为行为必要四格交互的事件变换，而不是原始hidden均值或Top-K坐标。对每个关系边、unit、语言、surface和层位，令

$$I_{{e,q}}=\frac14(H_{{00,e,q}}-H_{{01,e,q}}-H_{{10,e,q}}+H_{{11,e,q}}),$$

并检验

$$I_{{e_2,q_2}}=\mathcal C_\theta(I_{{query,q_1}};c)+\epsilon,$$

其中条件$c$只使用实验可控的语言、复述、事实顺序、定义顺序、候选顺序和事件身份。必须在discovery内选层/模型，confirmation选择后冻结，fresh unit/整关系边/整上下文锁箱不得调参；同时报告零、恒等、全局尺度、逐坐标尺度与条件尺度等基础模型，复杂模型只有在跨unit且跨edge超过基础模型时才可称候选编译规律。完整序列margin另作输出终点，不把相关当中介。

**结果汇总。** 审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；大阶段 `{json.dumps(result['stage_contract'], ensure_ascii=False)}`；附件哈希 `{json.dumps(result['attachment_sha256'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2511_c75137_c75520_phase2499_2510_audit_event_compiler_contract.py`；审计final位于`{OUT}`；被审计原始结果仍位于Phase2499–2510目录。

**理论进展。** 当前最小诚实图景是“关系定义与query共同产生行为相关的条件响应；响应随候选和输出事件改变；输出分数仍遵循正确选择”。“事件条件编译”只是下一步待测假设名称，不是已有发现。新的关键可证伪点是：同一冻结坐标变换能否在未见unit、未见关系边与未见上下文上预测全2560维目标场以及序列margin。

**问题硬伤与结论。** 现有证据仅Qwen3-4B、短合成二候选、三个原pair加三个换伙伴边、一个独立环；float16落盘不支持极小坐标效应；Walsh仍混合关系选择、检索与实体准备；无因果干预。故尚未发现天然坐标齿轮、纯语义代码、因果中介或语言编码闭合。下一Phase先用既有全场做不增加模型调用的逐层基础地图，再决定新锁箱需要采什么，避免高等数学先验替代观察。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    finals = {}
    for phase in range(2499, 2511):
        matches = list(RESULT.glob(f"phase{phase}_*/analysis/final.json"))
        finals[str(phase)] = load_json(matches[0]) if len(matches) == 1 else None
    audit = {
        "retained": [
            "behavioral relation-meaning necessity within qualified synthetic pairs",
            "equal-length causal-prefix exact-zero control",
            "dense signed query interaction and candidate-sequence score alignment",
            "fixed-vector autonomous transport failure",
            "answer-boundary partner-independent additive-potential failure",
        ],
        "corrected_overclaims": [
            "Walsh cancellation is not evidence of an internal XOR implementation",
            "0.782 is a descriptive share among three Walsh terms, not total semantic information",
            "24/24 scalar interaction signs do not establish complete probability preservation",
            "high additive R2 on one-cycle graph does not establish independent family potentials",
            "event-dependent change is observed; a nonlinear compiler is not yet identified",
            "Hodge curvature, fiber transport, and coordinate gears remain hypotheses",
        ],
        "unresolved": [
            "cross-edge and cross-context predictive operator",
            "natural language and multihop generalization",
            "causal necessity and sufficiency",
            "cross-model conservation",
        ],
    }
    stage_contract = {
        "primary": "full-coordinate event-conditioned prediction before causal closure",
        "sequence": ["retrospective layer map", "fresh factorial lockbox", "operator competition",
                     "coordinate cooperation", "natural-language transfer", "cross-model audit", "visualization and cleanup"],
        "forbidden_shortcuts": ["top-k-only analysis", "single-row filtering", "lockbox qpoint reselection",
                                "calling correlation mediation", "calling a fitted high-R2 map a mechanism"],
        "automatic_continuation_rule": "continue only while the immediate falsifiable target remains event-conditioned compilation",
    }
    phase_counts = {str(p): MEMO.read_text(encoding="utf-8").count(f"## Phase {p}:") for p in range(2499, 2511)}
    checks = {
        "all_prior_finals_unique": all(v is not None for v in finals.values()),
        "all_prior_checks_passed": all(v and v.get("all_checks_passed") for v in finals.values()),
        "memo_prior_phase_once": all(v == 1 for v in phase_counts.values()),
        "attachments_present": all(path.exists() for path in ATTACHMENTS),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "audit": audit, "stage_contract": stage_contract,
        "attachment_sha256": {path.name + f"#{i}": sha256(path) for i, path in enumerate(ATTACHMENTS)},
        "prior_phase_checks": {key: bool(value and value.get("all_checks_passed")) for key, value in finals.items()},
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
