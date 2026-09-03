#!/usr/bin/env python3
"""Audit Phase2396-2433 reviews and freeze the next full-coordinate trajectory campaign."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase2434_c33521_c33840_trajectory_campaign_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\061ac59e-1198-4a46-af41-7c4ce8fdaec5\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\dd3ba6b4-77ae-4093-acbf-3e4016eee804\pasted-text.txt"),
)
PHASE = 2434
CAMPAIGN = "C33521-C33840"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Phase2396–2433证据审查与全坐标事件轨迹新战役合同（{CAMPAIGN}） [{stamp}]

**测试原理与证据审查。** 对两份附件逐条回查Phase2396–2433的MEMO与final。保留：固定物理坐标的状态依赖耦合、跨层传播骨架、语义交互能量、中英方向纹理、最终层逐坐标读出都是真实但有限的拼图；关系组合算子、稳定坐标联盟、内部→输出闭合、可逆跨语言共享码均未成立。第二份附件所称“线性工具已穷尽”“语义不在线性投影中”“DS7B反例彻底证明结构不等于理解”超出证据：历史只检验了有限的对角/条件模型，若干语义减词项指标为正；DS7B还受行为协议、能力和量化影响。SAE-30000、互信息、Gromov–Wasserstein可以是有边界的竞争者，但阳性分别不等于语义齿轮、非线性机制或功能流形同构，阴性也不证明语义不可定位。

$$X[f,u,c,r,\ell,t,j]=H_{{f,u,c,r,\ell,t,j}},\qquad U_{{\ell,t,j}}=H_{{\ell+1,t,j}}-H_{{\ell,t,j}},$$

$$I_{{sem}}=(H_{{valid}}-H_{{brokenA}}),\qquad I_{{lex}}=(H_{{brokenA}}-H_{{brokenB}}).$$

**新大方案。** 冻结八类外部操作，覆盖概念/知识（taxonomy、causal、temporal）、语法（negation、preposition、coreference、punctuation）与功能（sentence reordering），每类同时平衡unit、语言、表面、方向、query-role、候选槽和valid/brokenA/brokenB。Phase2435完成材料与四模型顺序行为资格；2436在Qwen4B-BF16采集全部事件×层×2560坐标以及大样本全token场并永久保留；2437建立出现—持续—分化的有符号轨迹图谱；2438在同一冻结锁箱比较家族基线、标量、对角、事件错序、坐标错位和等规模组，不先上SAE；2439把轨迹连接到首分歧token贡献与严格自主生成；2440发布全坐标热力图、验证客户端与冷存储索引；完成后自动审计并执行同目标的长上下文后继Phase2441。

**结果汇总。** 证据审查 `{json.dumps(result['evidence_audit'], ensure_ascii=False)}`；冻结合同 `{json.dumps(result['contract'], ensure_ascii=False)}`；阶段计划 `{json.dumps(result['mega_plan'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2434_c33521_c33840_trajectory_campaign_audit.py`；附件哈希、审查、合同和final位于`tests/glm5/result/phase2434_c33521_c33840_trajectory_campaign_audit`。只追加本MEMO，未新增其他Markdown。

**理论进展、问题硬伤与结论。** 工作对象由“某截面的最佳回归”改为外部操作条件下的层×事件×坐标有符号轨迹。轨迹曲率、SAE或高阶工具都不能先验称为语义机制；必须先胜过词项、表面、事件错序、坐标错位和盲材料。原始场删除被认定为战略硬伤：本战役只删除可重建缓存和重复副本，不删除唯一全坐标原始场。当前理论仍是“状态条件化的分布式坐标转移超图”工作框架，不是已发现的新数学结构。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    attachments = [{"path": str(path), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in ATTACHMENTS]
    evidence_audit = {
        "retained": [
            "Phase2396-2433 supports a shared condition-dependent residual dynamics skeleton, not a closed language mechanism.",
            "Full physical-coordinate event and token trajectories are the correct next observational object.",
            "External families must be treated as researcher-defined hypergraph labels until they predict blind internal and output structure.",
            "Observation and atlas accumulation precede causal closure.",
        ],
        "corrected": [
            "The tested diagonal and conditional linear models do not exhaust linear representations or prove that semantics is nonlinear.",
            "Positive semantic-minus-lexical results exist on several locks, so semantic specificity is limited/conditional rather than identically zero.",
            "DS7B behavior/field mismatch is a warning about construct validity, not a proof that all internal structure lacks understanding.",
            "A 30000-feature SAE can hide distributed low-amplitude coordinates and is not the primary method or a causal gear by construction.",
            "Mutual information estimators in high dimension and Gromov-Wasserstein alignment need strong nulls; success is not mechanism/isomorphism and failure is not nonexistence.",
            "The deleted Phase2424/2428/2429 bulk arrays are a reproducibility loss; future unique raw fields must be retained with hashes.",
        ],
        "boundary": "Evidence motivates a new event/token trajectory atlas; it does not establish nonlinearity, emergence, a semantic gear, or mathematical closure.",
    }
    contract = {
        "families": ["taxonomy", "causal", "temporal", "negation_scope", "preposition_role",
                     "coreference_binding", "punctuation_attachment", "sentence_reordering"],
        "axes": ["unit", "language", "surface", "direction", "query_role", "candidate_slot", "validity"],
        "validity": ["valid", "broken_a", "broken_b"],
        "events": ["prefix_end", "operation_end", "argument_end", "context_end", "query_end",
                   "candidate1_end", "candidate2_end", "answer_boundary"],
        "primary_field": "sample x qpoint x event/token x all physical coordinates",
        "precision": "Qwen3-4B BF16 runtime; float16 archival with scale/error audit; other models behavior/relative topology only",
        "raw_policy": "retain every unique full-coordinate raw array; only delete duplicates/reconstructable cache after hash verification",
        "discovery_order": ["raw signed trajectory", "semantic/lexical residual", "reuse/differentiation",
                            "basic frozen structure tournament", "output bridge", "bounded causal test"],
        "prohibited_shortcuts": ["Top-K/PCA as primary", "SAE-only discovery", "single sentence family",
                                 "pretty geometry as mechanism", "causal failure as stop rule", "deleting unique raw fields"],
    }
    mega_plan = [
        {"phase": 2434, "campaign": CAMPAIGN, "task": "audit and freeze trajectory campaign"},
        {"phase": 2435, "campaign": "C33841-C34160", "task": "external hypergraph material, token anchors, sequential four-model behavior"},
        {"phase": 2436, "campaign": "C34161-C34480", "task": "Qwen4B BF16 full-coordinate event field and full-token lockbox trajectories"},
        {"phase": 2437, "campaign": "C34481-C34800", "task": "signed emergence, persistence, differentiation and family trajectory atlas"},
        {"phase": 2438, "campaign": "C34801-C35120", "task": "basic coordinate/event/group structure tournament on frozen locks"},
        {"phase": 2439, "campaign": "C35121-C35440", "task": "trajectory-to-first-divergence output contribution and autonomous bridge"},
        {"phase": 2440, "campaign": "C35441-C35760", "task": "parameter-level visualization, build verification, raw hashes and retention audit"},
        {"phase": 2441, "campaign": "C35761-C36080", "task": "automatic same-goal successor: long-context trajectory replication"},
    ]
    checks = {"attachments_present": all(path.exists() for path in ATTACHMENTS),
              "phase_continuity": "## Phase 2433:" in MEMO.read_text(encoding="utf-8"),
              "eight_families": len(contract["families"]) == 8,
              "eight_events": len(contract["events"]) == 8,
              "full_coordinate_primary": "all physical coordinates" in contract["primary_field"],
              "raw_retention": contract["raw_policy"].startswith("retain")}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "attachments": attachments,
              "evidence_audit": evidence_audit, "contract": contract, "mega_plan": mega_plan,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
