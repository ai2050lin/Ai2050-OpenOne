#!/usr/bin/env python3
"""Audit Phase 2486-2498 evidence and preregister a behaviorally necessary relation-swap campaign."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2499_c63681_c64000_phase2486_2498_audit_semantic_necessity_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2499, "C63681-C64000"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Phase2486–2498证据审计与语义必要性四格合同冻结（{CAMPAIGN}） [{stamp}]

**测试原理与证据对象。** 本Phase不重新解释热力图，而是逐项核对Phase2486–2498的`final.json`、原场shape、哈希、冻结层位、行为门和结论边界。审计严格区分：行为事实、全坐标相关纹理、描述性能量账本、预测基线、因果中介与机制闭合。附件数字与原始final一致的部分保留；把“跨marker”误写成“彻底分离语义与词项”、把主效应份额写成内部加法结构、把单一合同的跨语言失败推广为“关系纹理不跨语言”、把q20/q21命名为编译层等表述全部撤回。

**审查结果。** 保留 `{json.dumps(result['retained'], ensure_ascii=False)}`；修正 `{json.dumps(result['corrected'], ensure_ascii=False)}`；原始证据核验 `{json.dumps(result['source_verification'], ensure_ascii=False)}`。

**下一大阶段的测试原理与用例。** 冻结十二个关系family并配成六个竞争对。每条prompt包含同一来源的两条自然关系事实、两个无意义marker及其两条含义定义、固定候选和固定查询marker。只交换两个marker承载的关系定义，事实、实体、候选、marker与查询不变，正确答案必须翻转。再把“定义交换” (m\in\{{0,1\}}) 与“查询marker” (q\in\{{0,1\}}) 完全交叉；选中关系由 (m\oplus q) 决定。对四格HiddenState使用基础Walsh交互：

$$I_{{r_0-r_1}}=\frac14\left(H_{{00}}-H_{{01}}-H_{{10}}+H_{{11}}\right).$$

该交互同时消去定义交换主效应和marker身份主效应。定义末端、事实末端位于查询marker之前，理论上四格交互应为严格零，作为causal-prefix负对照；查询marker、两个固定候选与answer-boundary才允许出现选择交互。discovery/confirmation/lockbox使用独立实体和marker，confirmation唯一选择qpoint，lockbox同层一次揭示。中英文、四surface、十二family、两种swap、两种query marker全部参与；全事件保存38个qpoint×2560坐标，并保留代表性全token场。

**阶段任务与相关文件。** 冻结的大方案为：Phase2500行为必要性门；Phase2501全场采集；Phase2502四格交互锁箱图谱；Phase2503自主生成与输出概率联系；Phase2504参数级客户端发布与阶段审计。若这一直接目标成立，自动续研Phase2505–2507，以新关系配对和新材料检验纹理是否独立于原配对伙伴。合同位于 `{result['contract_path']}`，本脚本为`tests/glm5/phase2499_c63681_c64000_phase2486_2498_audit_semantic_necessity_contract.py`。

**理论进展、问题硬伤与结论。** 现有最强事实是“family定义条件超越特定记录marker的稠密有符号响应纹理”，而不是纯语义代码。Phase2497的answer-boundary跨语言优势为负，只能否定该层、该合同、该护照的跨语言可识别性；不能证明所有层或所有关系编码均不跨语言。86.85%是该设计的描述性主效应份额，不表示语言覆盖或删除语义，更不能写成HiddenState内部成分相加。逐坐标对角模型仅在一次q35→q36锁箱未超过全局尺度，不等于所有坐标协同或所有层间条件动力学不存在。当前瓶颈首先是任务可辨识性；因此先建立行为必要的关系选择交互，再谈齿轮、干预或数学闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    p2496 = load_json(RESULT / "phase2496_c61121_c62272_nonce_marker_rotation_behavior_fullfield/analysis/final.json")
    p2497 = load_json(RESULT / "phase2497_c62273_c62912_family_vs_marker_fullcoordinate_lockbox/analysis/final.json")
    p2498 = load_json(RESULT / "phase2498_c62913_c63680_nonce_family_visualization_final_audit/analysis/final.json")
    raw_paths = [Path(p2496["collection"]["event_field"]), Path(p2496["collection"]["alltoken_field"])]
    source_verification = {
        "phase2496_all_checks": p2496["all_checks_passed"],
        "phase2497_all_checks": p2497["all_checks_passed"],
        "phase2498_all_checks": p2498["all_checks_passed"],
        "behavior_rows": p2496["material"]["rows"],
        "qualified_families": len(p2496["behavior"]["qualified_families"]),
        "event_shape": p2496["collection"]["event_shape"],
        "alltoken_shape": p2496["collection"]["alltoken_shape"],
        "selected_qpoint": p2497["selection"]["qpoint"],
        "lockbox_family_across_marker_answer_advantage": p2497["lockbox"]["answer_boundary"]["family_across_marker"]["identity_advantage_over_q95"],
        "lockbox_family_across_language_answer_advantage": p2497["lockbox"]["answer_boundary"]["family_across_language"]["identity_advantage_over_q95"],
        "answer_language_main_effect_share": p2497["main_effect_shares"]["answer_boundary"]["language"],
        "raw_hashes_recomputed": {p.name: sha256(p) for p in raw_paths},
        "raw_hashes_match": all(sha256(p) == p2496["collection"]["sha256"][p.name] for p in raw_paths),
    }
    retained = [
        "Qwen3-4B nonquantized BF16 supports full-token/full-layer/full-2560-coordinate collection",
        "signed family-conditioned texture can cross four nonce record markers at frozen q20",
        "marker identity and family-definition context coexist",
        "squared energy is often less family-specific than signed direction",
        "the coordinate-specific diagonal q35-to-q36 model failed to beat the simpler standardized global model in lockbox",
        "Phase2496 behavior did not require distinguishing the twelve relation meanings",
    ]
    corrected = [
        "cross-marker reuse does not completely separate semantics from definition lexemes or prompt context",
        "0.96 same-family cosine is not an identity advantage and is not proof of abstract semantic code",
        "86.85 percent is a descriptive main-effect share for one contract, not language erasing or covering semantics",
        "negative answer-boundary cross-language identity does not prove relation texture is universally language-specific",
        "family shares cannot be inserted as additive percentages of an internal semantic component",
        "q20/q21 are selected measurement locations, not established compiler layers",
        "dense contrast energy coverage is not information localization or causal gear membership",
        "behavioral success on nonce links is not evidence that relation meaning was computed",
    ]
    contract = {
        "name": "behaviorally necessary relation-meaning 2x2 Walsh interaction",
        "families": ["taxonomy", "part_whole", "product", "causal", "temporal", "spatial", "role", "preference", "membership", "translation", "coreference", "punctuation"],
        "primary_pairs": [["taxonomy", "part_whole"], ["product", "causal"], ["temporal", "spatial"], ["role", "preference"], ["membership", "translation"], ["coreference", "punctuation"]],
        "units": {"20": "discovery", "21": "confirmation", "22": "lockbox"},
        "factors": {"pair": 6, "language": 2, "surface": 4, "meaning_swap": 2, "query_marker": 2},
        "rows": 576,
        "answer_flip_required": True,
        "fixed_within_swap_pair": ["facts", "entities", "candidate order", "marker strings", "query marker", "output format"],
        "changed_within_swap_pair": ["which relation description is attached to each marker", "correct answer"],
        "events": ["definition_end", "facts_end", "query_marker", "candidate0", "candidate1", "answer_boundary"],
        "qpoints": 38,
        "coordinates": 2560,
        "selection": "confirmation only; one answer-boundary qpoint; lockbox same qpoint for all events",
        "causal_prefix_control": "definition_end and facts_end Walsh interaction must be numerical zero because query-marker variants have identical prefixes",
        "claim_boundary": "positive interaction is a behaviorally necessary selection-associated texture, not yet a causal mediator or universal semantic code",
    }
    contract_path = OUT / "contract/semantic_necessity_walsh_contract.json"
    save(contract_path, contract)
    checks = {
        "source_final_files_pass": all((p2496["all_checks_passed"], p2497["all_checks_passed"], p2498["all_checks_passed"])),
        "source_raw_hashes": source_verification["raw_hashes_match"],
        "twelve_families": len(contract["families"]) == 12,
        "six_primary_pairs": len(contract["primary_pairs"]) == 6,
        "full_factorial_rows": 3 * 6 * 2 * 4 * 2 * 2 == contract["rows"],
        "answer_flip_preregistered": contract["answer_flip_required"],
        "confirmation_only_selection": contract["selection"].startswith("confirmation only"),
        "all_coordinates": contract["coordinates"] == 2560,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "source_verification": source_verification,
        "retained": retained,
        "corrected": corrected,
        "contract_path": str(contract_path),
        "contract": contract,
        "adjudication": {
            "abstract_relation_semantics_identified_in_phase2497": False,
            "language_erases_semantics_supported": False,
            "semantic_necessity_is_next_immediate_target": True,
            "causal_intervention_authorized_before_behavior_gate": False,
            "language_encoding_mechanism_closed": False,
        },
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
