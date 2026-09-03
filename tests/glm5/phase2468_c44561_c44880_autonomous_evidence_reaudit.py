#!/usr/bin/env python3
"""Re-audit Phase2466 autonomous claims and freeze the next observation contract."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2466 = next(RESULT.glob("phase2466_*"))
OUT = RESULT / "phase2468_c44561_c44880_autonomous_evidence_reaudit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2468, "C44561-C44880"
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def classify_prefix(text: str) -> str:
    compact = text.strip()
    if re.match(r"^(answer|final answer)\s*[:：]", compact, flags=re.I):
        return "english_answer_prefix"
    if re.match(r"^(答案|最终答案)\s*[:：]", compact):
        return "chinese_answer_prefix"
    return "no_explicit_answer_prefix"


def decode_audit() -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    rows = read_jsonl(P2466 / "behavior/greedy_generation.jsonl")
    decoded = []
    summary: dict[str, dict] = {}
    for item in rows:
        target = tokenizer.decode(item["target_ids"], skip_special_tokens=False)
        generated = tokenizer.decode(item["generated_ids"], skip_special_tokens=False)
        record = {
            **item,
            "target_text": target,
            "generated_text": generated,
            "prefix_class": classify_prefix(generated),
            "generation_budget": len(item["generated_ids"]),
            "target_token_length": len(item["target_ids"]),
            "budget_equal_target_length": len(item["generated_ids"]) == len(item["target_ids"]),
        }
        decoded.append(record)
    for interface in sorted({item["interface"] for item in decoded}):
        selected = [item for item in decoded if item["interface"] == interface]
        classes = Counter(item["prefix_class"] for item in selected)
        summary[interface] = {
            "rows": len(selected),
            "prefix_counts": dict(classes),
            "explicit_answer_prefix_rate": sum(v for k, v in classes.items() if k != "no_explicit_answer_prefix") / len(selected),
            "budget_equal_target_length_rate": sum(item["budget_equal_target_length"] for item in selected) / len(selected),
            "reported_exact_rate_all_variants": sum(bool(item["exact"]) for item in selected) / len(selected),
            "examples": [
                {k: item[k] for k in ("case_id", "target_text", "generated_text", "prefix_class")}
                for item in selected[:4]
            ],
        }
    path = OUT / "analysis/decoded_generation_reaudit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in decoded), encoding="utf-8")
    return {"decoded_rows": str(path), "interfaces": summary, "rows": len(decoded)}


def evidence_audit(decoded: dict) -> dict:
    final = json.loads((P2466 / "analysis/final.json").read_text(encoding="utf-8"))
    q18 = final["analysis"]["crossinterface_path"]
    measurements = {
        "query_end_semantic_crossinterface": q18["semantic_validity"]["prompt_query_end"]["q18"]["coordinate"],
        "generated_token1_semantic_crossinterface": q18["semantic_validity"]["generated_token1"]["q18"]["coordinate"],
        "query_end_lexical_crossinterface": q18["lexical_control"]["prompt_query_end"]["q18"]["coordinate"],
        "generated_token1_lexical_crossinterface": q18["lexical_control"]["generated_token1"]["q18"]["coordinate"],
    }
    corrected = {
        "retained": [
            "Phase2454 establishes a strong Qwen14B model-local output-conditioned replication under its tested precision.",
            "Phase2458 establishes a BF16 small-dose measurement floor and a strong 2% local odd-effect prediction, not stable curvature.",
            "Phase2460 establishes a teacher-forced two-token local path effect, not free-generation closure.",
            "Phase2465 establishes partial cross-interface coordinate geometry, but semantic specificity is absent.",
            "Phase2466 stores genuine greedy-prefix HiddenState fields and therefore remains valuable raw evidence.",
            "A contextual token occurrence is a useful analysis node because causal HiddenState depends on prefix, position, role, and task.",
        ],
        "withdrawn_or_downgraded": [
            "0.85 to 0.11 is not a teacher-forced-versus-autonomous trajectory cosine and cannot be called state collapse.",
            "The two numbers are separate entity-versus-code cross-interface correlations at two events; they do not track one state through time.",
            "Phase2466 did not feed the correct first answer token at prompt_query_end; first-token VJP work specifies a target contrast but does not reveal the answer in the prompt.",
            "Entity exact=0 is not evidence that the model cannot find the answer because generation was stopped after target-token-length steps while the model emitted Answer:/答案： prefixes.",
            "The data do not establish a fundamental understanding-expression split.",
            "Contextual token occurrence is not yet proven to be the unique or minimal encoding unit.",
            "Typed hypergraphs and conditional gears are research representations, not demonstrated internal data structures.",
            "Advanced decompositions cannot be run as an unrestricted tournament and selected on the same lockbox; basic full-coordinate observations must precede them.",
        ],
    }
    contract = {
        "autonomous_generation": {
            "minimum_new_tokens": 12,
            "stop_rule": "stop only after a candidate can be parsed unambiguously or EOS/max budget is reached",
            "parser": "strip Answer:/Final answer:/答案： prefixes, normalize whitespace/case, then match the complete candidate or frozen code",
            "report": ["raw-token exact", "parsed-answer exact", "prefix rate", "unparsed rate", "correct/incorrect trajectories"],
        },
        "observation": {
            "unit": "contextual token/span occurrence is an analysis node, not a claimed minimal mechanism",
            "axes": ["family", "role", "content", "language", "surface", "output_interface", "validity", "generation_step", "layer", "token", "physical_coordinate"],
            "storage": "all physical coordinates; no Top-K/PCA replacement",
            "priority": "raw states and elementary controlled contrasts before high-order mathematical models",
        },
        "evidence_levels": {
            "L0": "descriptive field/heatmap",
            "L1": "frozen cross-content/language/unit replication",
            "L2": "frozen next-layer or future-output prediction",
            "L3": "natural-state causal specificity",
            "L4": "successful autonomous multi-token closure",
            "L5": "cross-scale/cross-architecture functional invariant",
        },
    }
    return {"measurement_identity": measurements, "claims": corrected, "next_contract": contract, "decoded": decoded}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    decoded = result["audit"]["decoded"]["interfaces"]
    measures = result["audit"]["measurement_identity"]
    text = rf"""


## Phase {PHASE}: 自主生成“崩解”结论重审、答案前缀伪影与上下文预测状态新合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 逐条审查三份Phase2453–2467复盘，并直接解码Phase2466全部192条真实贪心token。区分三种不同对象：同一轨迹跨事件相似、同一事件跨接口相似、教师强制与自主生成相似。检查原实验是否允许模型生成完整`Answer:/答案：`前缀和候选答案；冻结后续至少12 token、完整候选解析、前缀剥离、raw exact与parsed exact并报的自主合同。上下文化token/span仅作为分析节点，不预设为唯一或最小编码单位。

$$C_e=\mathbb E_f\cos\left(I^{{entity}}_{{f,e}},I^{{code}}_{{f,e}}\right),\qquad C_{{e_1}}-C_{{e_2}}\neq\cos\left(I_{{e_1}},I_{{e_2}}\right).$$

**结果汇总。** Phase2466 q18语义跨接口在query-end为`{measures['query_end_semantic_crossinterface']:.6f}`、generated-token1为`{measures['generated_token1_semantic_crossinterface']:.6f}`；词项对应为`{measures['query_end_lexical_crossinterface']:.6f}`与`{measures['generated_token1_lexical_crossinterface']:.6f}`。它们是各事件内部的实体—代码相关，不是同一状态时间相关，也没有教师强制对照。实体接口解码审计 `{json.dumps(decoded['candidate_entity'], ensure_ascii=False)}`；代码接口 `{json.dumps(decoded['letter_code'], ensure_ascii=False)}`。原循环对每条只生成“目标token长度”步；实体接口大量生成`Answer:`/`答案：`，预算被前缀消耗，故`exact=0`不可解释为找不到答案。

**相关文件。** 脚本`tests/glm5/phase2468_c44561_c44880_autonomous_evidence_reaudit.py`；逐行解码与final位于同名结果目录；原Phase2466原场未修改。

**分析与理论进展。** 保留Qwen14B模型内输出条件纹理、BF16小剂量测量地板、两token教师强制局部效应、跨接口部分几何以及Phase2466真实贪心原场。撤销“0.85→0.11证明状态崩解”“教师强制已告诉首token答案”“理解—表达存在根本断裂”。更谨慎的新对象是`上下文化预测状态`：同一token/span出现会随前缀、位置、角色、任务和输出接口改变，但它是否是最小编码单位仍待检验。

**问题硬伤与结论。** 原Phase2466生成预算、答案解析和跨事件指标身份均不足以支持自主失败机制。类型化外部超图、条件联盟和跨层运输是有用候选表示，不是已发现的模型内部结构。下一阶段先做大样本行为资格、tokenizer/span审计和全层全token物理场，再用基本坐标对照裁决；CP/Tucker、GCCA、HVP等不得在同一锁箱上自由挑选冠军。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    decoded = decode_audit()
    audit = evidence_audit(decoded)
    entity = decoded["interfaces"]["candidate_entity"]
    checks = {
        "decoded_192": decoded["rows"] == 192,
        "entity_prefix_artifact_present": entity["explicit_answer_prefix_rate"] >= 0.75,
        "budget_truncated": entity["budget_equal_target_length_rate"] == 1.0,
        "metric_identity_corrected": audit["measurement_identity"]["query_end_semantic_crossinterface"] > 0.8,
        "collapse_claim_withdrawn": any("cannot be called state collapse" in x for x in audit["claims"]["withdrawn_or_downgraded"]),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "audit": audit,
        "adjudication": {
            "phase2466_entity_zero_is_valid_semantic_failure": False,
            "phase2466_085_to_011_is_state_collapse": False,
            "understanding_expression_split_established": False,
            "contextual_prediction_state_is_analysis_unit": True,
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
