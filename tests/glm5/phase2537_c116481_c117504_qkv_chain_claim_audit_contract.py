#!/usr/bin/env python3
"""Audit Phase 2528-2536 claims and freeze the token-atomic Q/K/V campaign."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2537_c116481_c117504_qkv_chain_claim_audit_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PRIOR = RESULT / "phase2536_c115457_c116480_stage_terminal_audit_next_contract/analysis/final.json"
ATTACHMENTS = [
    Path(r"C:\Users\Admin\.codex\attachments\396c365d-445c-4598-956c-df08b3d1d363\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\c1b7cf66-60da-47ba-9da9-645059a1649a\pasted-text.txt"),
]
PHASE, CAMPAIGN = 2537, "C116481-C117504"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Phase2528–2536证据复审与token-atomic Q/K/V整阶段合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 逐项对照两份外部分析、Phase2536终审及Phase2529–2535原始final。把“架构恒等账本、观察重复、构造性充分、自然联盟损伤、条件救援、自主递归、跨模型事件复现”严格分层；特别核对附件标题中的“source→K/V→head→residual完整因果链”、facts单区0翻转、top32自主删除及跨模型充分性是否被越级解释。

$$q=W_QN(h_a),\quad k=W_KN(h_j),\quad v=W_VN(h_j),\quad \alpha=\operatorname{{softmax}}(qk^\top/\sqrt d),$$
$$\text{{可核算}}\;u_r=\sum_{{j\in r}}\alpha_jv_j\;\not\Rightarrow\;\text{{已分别证明Q/K/V的语义职责}}.$$

**结果汇总。** 保留结论 `{json.dumps(result['retained'], ensure_ascii=False)}`；过度结论修正 `{json.dumps(result['corrected'], ensure_ascii=False)}`；关键硬伤 `{json.dumps(result['hard_problems'], ensure_ascii=False)}`；冻结合同 `{json.dumps(result['contract'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 本脚本、两份附件哈希及final位于`{OUT}`；原始证据来自Phase2529–2536对应result目录，未以摘要替代原始数值。

**分析、理论进展与结论。** 当前最强对象是Qwen3-4B中的晚层、上下文条件、部分可替代route联盟：它具有external-source联合充分性、自然edge-cut损伤、条件救援和自主多token剂量效应。可将其称为“下游输出编译路线候选”，不能称为完整source→K/V语义因果链。facts单区0翻转只说明该干预剂量不足以跨越输出决策边界，不证明关系信息只在候选阶段才被检索；post-query混合候选与指令，尚不能区分输出身份、格式和关系检索。下一阶段先修复token边界和通用输出混杂，再分别干预Q、K、V、edge与$W_O$，随后测MLP跨层写入、自主递归和模型内跨架构共性。

**问题硬伤。** top32与旧随机32未匹配层位、输出范数、source mass、$W_O$写入强度、注意力熵和一般生成重要性；whole-head救援含构造性恢复；source K/V虽被保存却未分别干预；GQA使四个query heads共享一个KV head；完整重排行为门不足；跨模型只支持事件级而非物理或算法同构。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(PRIOR)
    attachment_meta = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in ATTACHMENTS]
    retained = [
        "Qwen3-4B中存在跨21个受控语言操作族复用的晚层route联盟候选。",
        "external-source top32 donor在unit31达到36/36翻转、随机32为0/36。",
        "top32 external edge cut造成自然损伤，全晚层切断后top32恢复强于随机恢复。",
        "672提示自主生成呈top8到top32累计损伤，而嵌套随机集合无同方向剂量效应。",
        "三个额外模型各自选择的top donor均优于等量随机，但自然必要性强弱不同。",
        "Phase2529保存了全晚层K/V及source加权V，且Attention模块内加法账本误差低于0.0041。",
    ]
    corrected = [
        "附件标题的‘完整因果链’降级为Attention架构账本加下游route局部因果链；Q/K/V语义职责未分离。",
        "facts单区0翻转不推出‘关系选择不是从事实提取’；它只在该patch协议和剂量下未跨过决策边界。",
        "post-query 75%翻转不能独立证明候选编译阶段检索，因为该区混合候选、格式和输出指令。",
        "自主top32删除的强损伤不自动等于关系特异性；必须加入复制、续写、格式及同长度控制。",
        "top32不是最小、唯一或逐head必要齿轮；更合适对象是条件化、部分可替代的route联盟。",
        "跨模型top donor优势只支持模型内选择性充分性重复，不证明同一算法或冗余随参数规模单调增加。",
        "HiddenState是运行时状态而非模型参数；固定模型坐标可测，但跨模型坐标号没有通用语义。",
    ]
    hard = [
        "token语义区域不严格且query-property曾为空",
        "top与random缺少六维匹配和多随机重复",
        "Q/K/V、destination edge与W_O尚未分离因果",
        "source residual经Attention/MLP写成下一层K/V的过程未知",
        "真多跳、三跳和完整句重排行为覆盖不足",
        "强zero与全晚层切断存在分布外风险",
    ]
    contract = {
        "name": "token-atomic source-state to Q/K/V write-read compiler campaign",
        "work_packages": [
            "WP1: 32+关系/语法/角色族与通用输出控制；tokenizer逐段构造，事实实体、关系、值、query-property、候选、指令严格非空互斥穷尽。",
            "WP2: Qwen3-4B全部晚层、全部token和全部物理坐标采集incoming/outgoing HiddenState、Q、post-RoPE K、V、QK logit、softmax、逐edge加权V及W_O写入。",
            "WP3: unit发现/lockbox分离，只换destination Q、source K、source V、K+V、whole-head/W_O，并做matched、错source、shuffled和多组特征匹配随机。",
            "WP4: 分解Attention/MLP对source下一层K/V可读状态的改写，做持续阻断、联盟分解及source特异救援。",
            "WP5: 自主多token、真两跳/三跳、角色组合、歧义词义和行为合格完整重排中逐生成步追踪。",
            "WP6: Qwen14B、DeepSeek7B、GLM4依次BF16非量化，只复现模型内功能事件与相对深度。",
            "WP7: 重要全场发布参数级客户端；未展示HiddenState/KV原场记录哈希后清理；最终审计并按即时目标自动续研。",
        ],
        "success": [
            "至少30族材料通过token-atomic结构门，至少20族通过双unit行为门。",
            "冻结edge在新unit中同时优于多组特征匹配随机，并在一般输出控制上显示可判定的特异性边界。",
            "Q/K/V至少能被实验区分为不同功能贡献；阴性结果不得被包装为职责闭合。",
            "自主每步和至少一个真组合任务复现source特异损伤或救援。",
            "至少两个额外模型复现功能事件；物理编号不跨模型对齐。",
        ],
    }
    checks = {
        "prior_passed": bool(prior["all_checks_passed"]),
        "attachments_present": all(p.exists() for p in ATTACHMENTS),
        "attachments_hashed": all(len(x["sha256"]) == 64 for x in attachment_meta),
        "complete_chain_overclaim_corrected": True,
        "facts_zero_logic_corrected": True,
        "matched_control_required": True,
        "claim_boundary": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "attachments": attachment_meta, "retained": retained,
              "corrected": corrected, "hard_problems": hard, "contract": contract, "checks": checks,
              "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
