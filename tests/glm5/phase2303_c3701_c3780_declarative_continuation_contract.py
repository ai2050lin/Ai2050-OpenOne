#!/usr/bin/env python3
"""Freeze a raw declarative-continuation interface over the Phase2296 facts."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase2296_c3101_c3160_ntp_predictive_contract"
OUT = RESULT / "phase2303_c3701_c3780_declarative_continuation_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
SOURCE = PARENT / "material/ntp_natural_bilingual.jsonl"
sys.path.insert(0, str(TESTS))

import model_utils  # noqa: E402


PHASE = 2303
CAMPAIGN = "C3701-C3780"
FAMILIES = (
    "agent_patient", "attitude_event", "comparison_order",
    "location_binding", "possession_query", "relative_binding",
)
PARTITIONS = ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")
QPOINTS_4B = (0, 1, 5, 10, 15, 20, 25, 30, 36, 37)
BEHAVIOR_GATE = 0.75
FORMATION_GATE = 0.75
Q14_FAMILIES = ("agent_patient", "attitude_event", "location_binding")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def answer_ids(tokenizer, answer: str, language: str) -> list[int]:
    text = answer if language == "zh" else " " + answer
    values = tokenizer.encode(text, add_special_tokens=False)
    if not values:
        raise RuntimeError(("empty_answer", answer, language))
    return [int(value) for value in values]


def source_fact(prompt: str, language: str) -> str:
    if language == "en":
        match = re.search(r"\s(?:Who|Where|What|Whom|Which)\b", prompt)
        if not match:
            raise RuntimeError(("english_question_boundary", prompt))
        return prompt[:match.start()].strip()
    boundary = prompt.rfind("。")
    if boundary < 0:
        raise RuntimeError(("chinese_question_boundary", prompt))
    stop = boundary + 1
    if stop < len(prompt) and prompt[stop] in ("”", "’"):
        stop += 1
    return prompt[:stop]


def target_first_in_source(row: dict) -> bool:
    fact = source_fact(row["prompt_core"], row["language"])
    target = row["correct_answer"]
    wrong = row["wrong_answer"]
    return fact.find(target) < fact.find(wrong)


def alternate_fact(row: dict, target_first: bool) -> str:
    language = row["language"]
    surface = row["surface"]
    family = row["family"]
    roles = row["role_values"]
    primary, secondary = roles["primary"], roles["secondary"]
    context, query = roles["context"], roles["query"]
    graph = row["semantic_graph"]

    if family == "agent_patient":
        if language == "en":
            body = (f"{primary} handed the {context} to {secondary}." if target_first else
                    f"The {context} was handed to {secondary} by {primary}.")
            return body if surface == "narrative" else f"A witness said, '{body}'"
        body = (f"{primary}把{context}交给了{secondary}。" if target_first else
                f"{context}先到了{secondary}手中，交出它的人是{primary}。")
        return body if surface == "narrative" else f"目击者说：“{body}”"

    if family == "attitude_event":
        other = graph[2][2]
        if language == "en":
            events = (f"{primary} ate the {query}, while {secondary} ate the {other}" if target_first else
                      f"{secondary} ate the {other}, while {primary} ate the {query}")
            return (f"{context} likes the fact that {events}." if surface == "narrative" else
                    f"{context} said, 'I am glad that {events}.'")
        events = (f"{primary}吃了{query}，而{secondary}吃了{other}" if target_first else
                  f"{secondary}吃了{other}，而{primary}吃了{query}")
        return (f"{context}喜欢这样一个事实：{events}。" if surface == "narrative" else
                f"{context}说：“我很高兴{events}。”")

    if family == "comparison_order":
        if language == "en":
            body = (f"{primary} is more {context} than {secondary}." if target_first else
                    f"{secondary} is less {context} than {primary}.")
            return body if surface == "narrative" else f"A reviewer said, '{body}'"
        body = (f"{primary}比{secondary}更{context}。" if target_first else
                f"{secondary}不如{primary}{context}。")
        return body if surface == "narrative" else f"评审说：“{body}”"

    if family == "location_binding":
        if language == "en":
            body = (f"The {primary} is in the {secondary}, not in the {context}." if target_first else
                    f"The {primary} is not in the {context}, but in the {secondary}.")
            return body if surface == "narrative" else f"A guide said, '{body}'"
        body = (f"{primary}在{secondary}里，不在{context}里。" if target_first else
                f"{primary}不在{context}里，而在{secondary}里。")
        return body if surface == "narrative" else f"向导说：“{body}”"

    if family == "possession_query":
        other = graph[1][2]
        if language == "en":
            facts = (f"{primary} owns the {context}; {secondary} owns the {other}." if target_first else
                     f"{secondary} owns the {other}; {primary} owns the {context}.")
            return facts if surface == "narrative" else f"The clerk said, '{facts}'"
        facts = (f"{primary}拥有{context}，{secondary}拥有{other}。" if target_first else
                 f"{secondary}拥有{other}，{primary}拥有{context}。")
        return facts if surface == "narrative" else f"管理员说：“{facts}”"

    if family == "relative_binding":
        if language == "en":
            body = (f"The editor thanked {primary} and later interviewed {secondary}." if target_first else
                    f"After interviewing {secondary}, the editor thanked {primary}.")
            return body if surface == "narrative" else f"A record states, '{body}'"
        body = (f"编辑感谢了{primary}，随后采访了{secondary}。" if target_first else
                f"采访{secondary}之后，编辑感谢了{primary}。")
        return body if surface == "narrative" else f"记录写道：“{body}”"
    raise KeyError(family)


def continuation_cue(row: dict) -> str:
    family, language, surface = row["family"], row["language"], row["surface"]
    context, query = row["role_values"]["context"], row["role_values"]["query"]
    if language == "en":
        cues = {
            "agent_patient": f"The person who handed over the {context} was",
            "attitude_event": f"The person who ate the {query} was",
            "comparison_order": f"The person who was more {context} was",
            "location_binding": f"The {query} could be found in the",
            "possession_query": f"The item owned by {query} was the",
            "relative_binding": "The person thanked by the editor was",
        }
        lead = "According to that statement, " if surface == "dialogue" else ""
        return lead + cues[family]
    cues = {
        "agent_patient": f"递出{context}的人是",
        "attitude_event": f"吃掉{query}的人是",
        "comparison_order": f"两人中更{context}的是",
        "location_binding": f"能找到{query}的地方是",
        "possession_query": f"{query}拥有的物品是",
        "relative_binding": "被编辑感谢的人是",
    }
    lead = "根据这段话，" if surface == "dialogue" else ""
    return lead + cues[family]


def compile_contract() -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    source_rows = [row for row in read_rows(SOURCE) if row["family"] in FAMILIES]
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    rows: list[dict] = []
    collisions: list[str] = []
    forbidden: list[str] = []
    unicode_replacement: list[str] = []
    balance = defaultdict(lambda: {"target_first": 0, "target_last": 0, "state0": 0, "state1": 0})
    for source in source_rows:
        surface_index = 0 if source["surface"] == "narrative" else 1
        target_first = (int(source["unit"]) + surface_index) % 2 == 0
        source_order = target_first_in_source(source)
        fact = (source_fact(source["prompt_core"], source["language"])
                if source_order == target_first else alternate_fact(source, target_first))
        prefix = fact + (" " if source["language"] == "en" else "") + continuation_cue(source)
        target_ids = answer_ids(tokenizer, source["correct_answer"], source["language"])
        wrong_ids = answer_ids(tokenizer, source["wrong_answer"], source["language"])
        if target_ids[0] == wrong_ids[0]:
            collisions.append(source["case_id"])
        if any(marker in prefix for marker in ("?", "？", "Answer", "Options", "只回答", "回答：")):
            forbidden.append(source["case_id"])
        if "\ufffd" in prefix or "\ufffd" in source["correct_answer"] or "\ufffd" in source["wrong_answer"]:
            unicode_replacement.append(source["case_id"])
        target_first_observed = prefix.find(source["correct_answer"]) < prefix.find(source["wrong_answer"])
        key = (source["family"], source["language"], source["surface"], source["partition"])
        balance[key]["target_first" if target_first_observed else "target_last"] += 1
        balance[key][f"state{int(source['state'])}"] += 1
        rows.append({
            **source,
            "source_case_id": source["case_id"],
            "case_id": source["case_id"] + "-decl",
            "declarative_prefix": prefix,
            "ntp_prompt_ids": [int(value) for value in tokenizer.encode(prefix, add_special_tokens=False)],
            "ntp_target_ids": target_ids,
            "ntp_wrong_ids": wrong_ids,
            "ntp_target_text": source["correct_answer"],
            "ntp_wrong_text": source["wrong_answer"],
            "target_mention_order": "first" if target_first_observed else "last",
            "source_fact_order_matched": bool(source_order == target_first),
            "ntp_interface": "raw_declarative_continuation_without_chat_template",
        })
    first_accuracy = sum(row["target_mention_order"] == "first" for row in rows) / len(rows)
    last_accuracy = sum(row["target_mention_order"] == "last" for row in rows) / len(rows)
    prompt_widths = [len(row["ntp_prompt_ids"]) for row in rows]
    audit = {
        "rows": len(rows), "families": list(FAMILIES),
        "languages": dict(Counter(row["language"] for row in rows)),
        "surfaces": dict(Counter(row["surface"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "prompt_width_min_max": [min(prompt_widths), max(prompt_widths)],
        "target_token_widths": dict(Counter(len(row["ntp_target_ids"]) for row in rows)),
        "wrong_token_widths": dict(Counter(len(row["ntp_wrong_ids"]) for row in rows)),
        "first_token_collision_count": len(collisions),
        "forbidden_marker_count": len(forbidden),
        "unicode_replacement_count": len(unicode_replacement),
        "first_mention_zero_accuracy": first_accuracy,
        "last_mention_zero_accuracy": last_accuracy,
        "every_cell_exact_order_and_state_balance": all(
            value["target_first"] == value["target_last"] and value["state0"] == value["state1"]
            for value in balance.values()
        ),
        "source_order_matched_rows": sum(row["source_fact_order_matched"] for row in rows),
        "machine_naturality": "templates_use_complete_facts_and_grammatical_declarative_cues",
        "human_blind_naturality": "NA_not_independently_reviewed",
    }
    return rows, audit


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    audit = result["audit"]
    text = rf"""

## Phase {PHASE}: 自然陈述续写边界合同与顺序捷径审计（{CAMPAIGN}） [{stamp}]

**测试原理与证据纠偏。** 本期先审查附件与 Phase2296–2302 原始证据。正确部分是：自回归训练要求直接研究“前缀如何约束下一词竞争”，上一阶段的完整词表、全检查点和逐坐标账值得保留。需要收紧之处是：问答提示仍可能主要测量任务接口；训练目标也不自动推出 HiddenState 是充分统计量、贝叶斯信念、流形、波动场或因果齿轮。本期不加载模型，而是把同一六族、32 单元、四分区、中英双语、两表面、两状态材料编译为原始文本续写，不使用 chat template、问题、Options 或 Answer 指令。

**测试用例与公式。** 例如 `Leo Bell handed the atlas to Amina Arden. The person who handed over the atlas was`，候选续写为 `Leo Bell` 与 `Amina Arden`。为封住“总选首个/末个提及实体”，每个族、语言、表面、分区内按 unit 与 surface 的奇偶配额交换事实顺序：

$$
o_i=(\operatorname{{unit}}_i+\operatorname{{surfaceIndex}}_i)\bmod 2,
\qquad
\operatorname{{Acc}}_{{first}}=\operatorname{{Acc}}_{{last}}=0.5.
$$

完整候选仍按 teacher forcing 计分：

$$
s(y_{{1:k}}\mid x)=\sum_{{r=1}}^k\log p_\theta(y_r\mid x,y_{{<r}}),
\qquad
\widehat y=\operatorname{{argmax}}_{{a\in(+,-)}}\frac{{s(y^a\mid x)}}{{|y^a|}}.
$$

**结果汇总与门槛。** 共编译 `{audit['rows']}` 行；首提及与末提及零模型准确率分别为 `{audit['first_mention_zero_accuracy']:.6f}`、`{audit['last_mention_zero_accuracy']:.6f}`；每个族×语言×表面×分区的顺序和状态均精确平衡 `{audit['every_cell_exact_order_and_state_balance']}`；首 token 碰撞 `{audit['first_token_collision_count']}`，禁用标记 `{audit['forbidden_marker_count']}`，Unicode 替换字符 `{audit['unicode_replacement_count']}`。此前控制台出现的中文乱码经码点审计确认只是 PowerShell 显示解码问题，文件中的中文码点完好。冻结行为门为每族 overall、语言、表面、分区的完整候选均值分和总分准确率均不低于 `{result['config']['behavior_gate']}`；错误样本不删除，只分账。

**相关文件、理论进展与硬伤。** 脚本 `tests/glm5/phase2303_c3701_c3780_declarative_continuation_contract.py`；材料和冻结配置位于 `tests/glm5/result/phase2303_c3701_c3780_declarative_continuation_contract`。本期只建立“问答接口—自然续写接口”的可识别比较，没有产生模型机制结论。机器模板自然度不能替代独立人类盲评；同一模型、同一事实和相近措辞仍不能代表开放自然文本；原词汇分区的共享情况沿用旧材料。基础计数、配额和概率公式已经足够，未引入高等数学。下一步授权仅为顺序审计通过后运行 Qwen3-4B，保存完整词表、38 个边界检查点、六族代表样本的全 token×全坐标场及自由续写观察。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((PARENT / "analysis/final.json").read_text(encoding="utf-8"))
    rows, audit = compile_contract()
    material = OUT / "material/declarative_continuation_bilingual.jsonl"
    write_rows(material, rows)
    config = {
        "phase": PHASE, "campaign": CAMPAIGN, "frozen_before_model_load": True,
        "research_object": "raw declarative next-token continuation boundary over paired Phase2296 facts",
        "families": list(FAMILIES), "partitions": list(PARTITIONS),
        "languages": ["en", "zh"], "surfaces": ["narrative", "dialogue"],
        "behavior_gate": BEHAVIOR_GATE, "formation_gate": FORMATION_GATE,
        "qpoints_4b": list(QPOINTS_4B), "q14_families": list(Q14_FAMILIES),
        "q14_selection": "fixed_semantic_diversity_before_qwen4_execution",
        "model_order": ["Qwen3-4B", "Qwen3-14B_if_q4_qualifies"],
        "null_models": ["first_candidate_mention", "last_candidate_mention", "wrong_candidate", "matched_surface"],
        "missingness": "retain_all_rows; mechanism claims stratified by behavior qualification",
        "stopping": "route-level only; failed families remain observational and do not stop other families",
        "forbidden_primary_methods": ["PCA", "Top-K", "cosine-only", "mean-delta transport"],
        "allowed_math": "basic counts, log probabilities, full-vocabulary divergences, exact-coordinate accounting",
    }
    save(OUT / "config/frozen_contract.json", config)
    checks = {
        "parent_closed": bool(parent["all_checks_passed"]),
        "row_count": len(rows) == len(FAMILIES) * 2 * 2 * 32 * 2,
        "all_prompt_ids_nonempty": all(row["ntp_prompt_ids"] for row in rows),
        "all_candidate_ids_nonempty": all(row["ntp_target_ids"] and row["ntp_wrong_ids"] for row in rows),
        "first_tokens_discriminative": audit["first_token_collision_count"] == 0,
        "no_question_or_answer_markers": audit["forbidden_marker_count"] == 0,
        "unicode_intact": audit["unicode_replacement_count"] == 0,
        "exact_order_and_state_balance": audit["every_cell_exact_order_and_state_balance"],
        "zero_models_exact_half": audit["first_mention_zero_accuracy"] == 0.5 and audit["last_mention_zero_accuracy"] == 0.5,
        "all_partitions_present": set(row["partition"] for row in rows) == set(PARTITIONS),
    }
    hashes = {"source": file_hash(SOURCE), "material": file_hash(material),
              "config": file_hash(OUT / "config/frozen_contract.json")}
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "audit": audit, "config": config, "checks": checks,
        "hashes": hashes, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "A raw declarative-continuation interface has been frozen with exact mention-order and state balance; "
            "this repairs a shortcut risk but contains no new model evidence."
        ),
        "next_authorization": (
            "If every material check passes, collect Qwen3-4B complete candidate scores, free continuations, "
            "full-vocabulary logits, all 38 boundary checkpoints, and six-family representative token fields."
        ),
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
