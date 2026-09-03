#!/usr/bin/env python3
"""Audit Phase2405-2421 claims and freeze the semantic-specific operator campaign."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase2422_c29681_c30000_evidence_audit_campaign_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2422
CAMPAIGN = "C29681-C30000"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\25d79396-1d22-4071-9807-0ec9cfcfbcae\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\4e8dc546-c553-4468-b627-3c3805c0e2d7\pasted-text.txt"),
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_audit() -> dict:
    return {
        "retained": [
            "Phase2405-2421 moved the object from static class separability to event-aligned residual dynamics and exact paired contrasts.",
            "Whole-template and label-order audits were necessary: family residuals and cross-model comparisons are otherwise confounded.",
            "The robust positive observation is current-state/same-coordinate predictability of local residual updates plus persistent relation-level Gram geometry.",
            "Valid-chain second-order interaction energy exceeds a lexical broken-vs-broken control in Qwen4B and Qwen14B; this is a candidate signal worth mapping.",
            "Qwen14B behavior distinguishes valid from broken chains much better than Qwen4B, so capability qualification matters.",
            "The next campaign must keep full physical coordinates, use multiple language families and lock out unit, template, language and family simultaneously.",
        ],
        "corrected": [
            "The strongest current account is a generic state-dependent residual field, not a semantic conditional gear.",
            "Phase2410 sign groups are descriptive partitions: they did not consistently beat diagonal/contiguous/random controls and are not discovered gears.",
            "Phase2411/2414 relation Gram persistence is functional geometry, not coordinate transport, identical topology or cross-model isomorphism.",
            "Phase2420/2421 energy ratios do not establish a pure logic tensor: reusable semantic-over-lexical state specificity was negative for most components.",
            "Attention transport and MLP semantic carving are hypotheses only; component additivity does not identify their linguistic roles.",
            "No manifold curvature, attractor, geodesic, bifurcation, causal singularity, compiler or recursive composition closure has been measured.",
            "The output/behavior bridge remains weak and teacher-forced; it is not an autonomous language mechanism.",
        ],
        "phase2415_numeric_correction": {
            "reported_closure_cosine": 1.0002944469451904,
            "mathematical_bound": [-1.0, 1.0],
            "bound_violation": 0.0002944469451904297,
            "cause": "The old helper accumulated a very large flattened float32 dot product and norms without float64 stabilization.",
            "status": "invalid_out_of_range_statistic",
            "effect": "It cannot be cited as closure evidence. Phase2424 must recompute component closure using float64 sums, zero-norm guards and a final bound check.",
        },
        "claim_boundary": (
            "The evidence supports distributed, condition-sensitive residual texture and candidate validity energy; "
            "it does not yet support a reusable semantic operator or a conditional coordinate gear."
        ),
    }


def frozen_contract() -> dict:
    return {
        "families": ["preference", "ownership", "spatial", "temporal", "causal", "comparison", "role_binding", "taxonomy"],
        "factorial": {
            "units": 8, "languages": ["en", "zh"],
            "surfaces": ["canonical", "paraphrase", "discourse", "natural"],
            "directions": [0, 1], "validity": ["valid", "broken_a", "broken_b"],
            "query_roles": ["source", "target"], "configurations": 1024, "rows": 6144,
        },
        "pairing": "Within each configuration, facts/candidates/order are frozen; validity and queried role are the only planned contrasts.",
        "splits": {
            "discovery": "unit0-5, controlled surfaces", "fresh_unit": "unit6-7, controlled surfaces",
            "template": "unit0-5, naturalized surfaces", "joint": "unit6-7, naturalized surfaces",
            "language": "English discovery to Chinese controlled lockbox", "family": "leave-one-family-out",
        },
        "observation": "Qwen3-4B BF16: embedding, every block state, final norm, Attention and MLP at relation/query/answer events, all 2560 coordinates; representative all-token paths.",
        "precision": "Qwen3-14B BF16-weight device_map=auto is preferred, but prior repeated Windows access violations are retained as feasibility evidence; NF4-weight/BF16-compute results, if used, are explicitly secondary and never amplitude-equated.",
        "raw_policy": "Keep raw full-coordinate fields through Phase2431; publish reusable full-coordinate derived fields, verify, then remove non-visualized raw fields once.",
    }


def mega_plan() -> list[dict]:
    return [
        {"phase": 2422, "campaign": "C29681-C30000", "task": "evidence audit, Phase2415 numerical correction, frozen contract and claim gates"},
        {"phase": 2423, "campaign": "C30001-C30320", "task": "6144-row eight-family valid/broken dual-role material, Qwen4B behavior and Qwen14B capability-qualified behavior"},
        {"phase": 2424, "campaign": "C30321-C30640", "task": "Qwen4B multi-event H/A/M full-coordinate field, representative all-token path and stable component closure"},
        {"phase": 2425, "campaign": "C30641-C30960", "task": "semantic-validity interaction atlas against lexical and generic residual baselines on five lockboxes"},
        {"phase": 2426, "campaign": "C30961-C31280", "task": "same-coordinate identity against shifts, random, variance-bin and sample-shuffle nulls"},
        {"phase": 2427, "campaign": "C31281-C31600", "task": "intrinsic coordinate cooperation discovery and stability versus diagonal/contiguous/random groups"},
        {"phase": 2428, "campaign": "C31601-C31920", "task": "cross-layer propagation, path consistency and family holdout"},
        {"phase": 2429, "campaign": "C31921-C32240", "task": "new direct-versus-composed relation material and algebraic composition tests"},
        {"phase": 2430, "campaign": "C32241-C32560", "task": "unembedding/output compilation and autonomous behavior bridge"},
        {"phase": 2431, "campaign": "C32561-C32880", "task": "full-coordinate visualization publication, build verification, raw cleanup and successor audit"},
    ]


def gates() -> dict:
    return {
        "candidate_signal": "semantic interaction differs from broken-vs-broken lexical interaction after role pairing",
        "reusable_operator": "matched semantic prediction beats all coordinate and sample nulls on fresh unit, template, joint, language and family lockboxes",
        "cross_layer_law": "a map fitted before the lockbox predicts adjacent and multi-hop layers with path consistency",
        "composition_law": "the same fitted rule predicts direct and composed relations, not just higher interaction energy",
        "output_bridge": "pre-registered field score predicts first-token margin and autonomous success on held-out configurations",
        "gear": "requires reusable physical-coordinate group, cross-layer recurrence, semantic specificity and output/causal relevance; no current result passes this gate",
    }


def append_memo(result: dict) -> None:
    existing = MEMO.read_text(encoding="utf-8")
    if f"## Phase {PHASE}:" in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Phase2405–2421证据审计、数值更正与语义专属算子总合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本Phase逐条审查两份附件，并以Phase2405–2421原始记录作为数字依据。审查不把修辞强度当证据，而按“是否直接测量、是否有严格零假设、是否跨内容/模板/语言/家族、是否连到输出”分级。同时检查Phase2415报告的余弦范围，并冻结Phase2423–2431共6144条、八关系族×八unit×中英×四表面×双方向×三有效性×双查询角色的统一材料合同。后续所有结构拟合只在discovery上发生，fresh-unit、template、joint、language和leave-one-family-out为锁箱。

$$I_{{sem}}=(U_{{valid,target}}-U_{{valid,source}})-(U_{{brokenA,target}}-U_{{brokenA,source}}),$$

$$I_{{lex}}=(U_{{brokenA,target}}-U_{{brokenA,source}})-(U_{{brokenB,target}}-U_{{brokenB,source}}),$$

$$\text{{gear gate}}=\text{{semantic specificity}}\land\text{{coordinate reuse}}\land\text{{cross-layer recurrence}}\land\text{{output relevance}}.$$

**结果汇总。** 附件指纹 `{json.dumps(result['attachments'], ensure_ascii=False)}`；保留结论 `{json.dumps(result['audit']['retained'], ensure_ascii=False)}`；修正/拒绝 `{json.dumps(result['audit']['corrected'], ensure_ascii=False)}`。Phase2415数值更正 `{json.dumps(result['audit']['phase2415_numeric_correction'], ensure_ascii=False)}`：`1.0002944469`超出余弦数学上界，正式标为无效统计量，不再作为闭合证据。统一合同 `{json.dumps(result['contract'], ensure_ascii=False)}`；阶段大方案 `{json.dumps(result['mega_plan'], ensure_ascii=False)}`；主张门 `{json.dumps(result['gates'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 审计脚本`tests/glm5/phase2422_c29681_c30000_evidence_audit_campaign_contract.py`；`audit.json`、`frozen_contract.json`、`mega_plan.json`、`claim_gates.json`和`final.json`位于`tests/glm5/result/phase2422_c29681_c30000_evidence_audit_campaign_contract`。未修改其他Markdown。

**分析与理论进展。** 第一份附件关于“由静态可分转向残差动力学、当前主效应更像通用状态依赖场、需要语义专属残差和严格锁箱”的判断成立。第二份附件保留组件物理可加、标签顺序修正、二阶对比值得继续三点；把Attention/MLP命名为搬运工/雕刻师、把二阶能量称为纯逻辑张量、把跨模型弱关系称为严格同构，均越过证据。研究对象因此明确为：同一外部语言操作在固定物理坐标上产生的语义专属交互，能否在多种零假设之外跨层复用并编译到输出。

**问题硬伤与结论。** 审计本身不产生新模型场；冻结材料仍是人工短关系与二候选任务。小模型可能只呈粗糙编码，阴性不等于大模型不存在机制。Phase2415原始场已经删除，无法事后高精度重算，故这里只能撤销越界余弦，不能伪造替代值；Phase2424将用float64稳定累加复测。当前结论仍是“分布式条件敏感残差纹理+有效性候选能量”，不是条件齿轮、逻辑张量或闭合理论。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    attachments = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                    "lines": len(path.read_text(encoding="utf-8-sig").splitlines())} for path in ATTACHMENTS]
    audit = evidence_audit()
    contract = frozen_contract()
    plan = mega_plan()
    claim_gates = gates()
    checks = {
        "attachments_present": all(path.exists() for path in ATTACHMENTS),
        "phase2415_cosine_flagged": audit["phase2415_numeric_correction"]["status"] == "invalid_out_of_range_statistic",
        "eight_families": len(contract["families"]) == 8,
        "six_thousand_one_hundred_forty_four_rows": contract["factorial"]["rows"] == 6144,
        "six_split_axes": len(contract["splits"]) == 6,
        "full_coordinates_primary": "all 2560 coordinates" in contract["observation"],
        "claim_boundary": "does not yet support" in audit["claim_boundary"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "attachments": attachments, "audit": audit,
              "contract": contract, "mega_plan": plan, "gates": claim_gates, "checks": checks}
    save(OUT / "audit.json", {"attachments": attachments, **audit})
    save(OUT / "frozen_contract.json", contract)
    save(OUT / "mega_plan.json", plan)
    save(OUT / "claim_gates.json", claim_gates)
    save(OUT / "final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
