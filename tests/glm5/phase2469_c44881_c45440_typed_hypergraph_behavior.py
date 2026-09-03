#!/usr/bin/env python3
"""Typed language-hypergraph materials, tokenizer audit, and parsed autonomous behavior."""
from __future__ import annotations

import gc
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2469_c44881_c45440_typed_hypergraph_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2469, "C44881-C45440"
UNITS = {
    8: ("Aroven", "Belune", "Cireth", "Dovian", "Elaro", "Fesin"),
    9: ("Garen", "Helio", "Ivara", "Jorin", "Kelor", "Lumea"),
    10: ("Maren", "Neris", "Orlan", "Pavia", "Quorin", "Ressa"),
}
FAMILIES = (
    "taxonomy", "part_of", "product_of", "word_sense", "role_binding",
    "preference_binding", "negation_scope", "temporal_order", "causal_chain",
    "coreference_binding", "punctuation_attachment", "spatial_relation",
    "sentence_reordering", "translation_binding",
)
INTERFACES = ("entity", "code")
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def pair_design(names: tuple[str, ...], trial: int) -> tuple[str, str, str, str, str, str]:
    a, b, c, d, e, f = names
    designs = (
        (a, b, c, d, a, b),
        (a, b, c, d, c, d),
        (b, a, d, c, b, a),
        (e, f, c, d, e, f),
    )
    x1, y1, x2, y2, query, target = designs[trial]
    foil = y2 if target == y1 else y1
    return x1, y1, x2, y2, query, target


def content(family: str, names: tuple[str, ...], trial: int, language: str) -> tuple[str, str, str, dict]:
    x1, y1, x2, y2, query, target = pair_design(names, trial)
    foil = y2 if target == y1 else y1
    a, b, c, d, e, f = names
    en: dict[str, tuple[str, str, str, dict]] = {
        "taxonomy": (f"{x1} is a fruit-kind. {x2} is a tool-kind. Which item is a fruit-kind?", x1, x2, {"edge":"is_a","query_role":"member"}),
        "part_of": (f"The amber module is part of {y1}. The cobalt module is part of {y2}. Which whole contains the module paired with {query}? {query} is paired with the {'amber' if query == x1 else 'cobalt'} module.", y1 if query == x1 else y2, y2 if query == x1 else y1, {"edge":"part_of","query_role":"whole"}),
        "product_of": (f"{x1} produces {y1}. {x2} produces {y2}. What is produced by {query}?", target, foil, {"edge":"product_of","query_role":"product"}),
        "word_sense": (f"In this glossary, 'luma' means {y1} in a finance context and {y2} in a river context. The sentence says: 'They deposited coins at the luma.' Which meaning is intended?", y1, y2, {"edge":"sense_in_context","query_role":"sense"}),
        "role_binding": (f"{x1} thanked {y1}. {x2} thanked {y2}. Who received thanks from {query}?", target, foil, {"edge":"recipient","query_role":"object"}),
        "preference_binding": (f"{x1} likes {y1}. {x2} likes {y2}. What does {query} like?", target, foil, {"edge":"preference","query_role":"object"}),
        "negation_scope": (f"{x1} is not active, while {x2} is active. Which one is active?", x2, x1, {"edge":"negation_scope","query_role":"positive"}),
        "temporal_order": (f"{x1} arrived before {x2}. Which one arrived later?", x2, x1, {"edge":"before","query_role":"later"}),
        "causal_chain": (f"{x1} caused {y1}. {x2} caused {y2}. What did {query} cause?", target, foil, {"edge":"causes","query_role":"effect"}),
        "coreference_binding": (f"{x1} told {y1}, 'you won.' {x2} was elsewhere. Who does 'you' refer to?", y1, x2, {"edge":"coreference","query_role":"referent"}),
        "punctuation_attachment": (f"{x1}, not {x2}, opened the door. Who opened the door?", x1, x2, {"edge":"punctuation_scope","query_role":"agent"}),
        "spatial_relation": (f"{x1} is left of {x2}. Which item is on the right?", x2, x1, {"edge":"left_of","query_role":"right"}),
        "sentence_reordering": (f"Event {x1} happened after event {x2}. Which event happened first?", x2, x1, {"edge":"discourse_order","query_role":"first"}),
        "translation_binding": (f"The codebook says the Chinese label '甲类' refers to {x1}, while '乙类' refers to {x2}. What does '甲类' refer to?", x1, x2, {"edge":"translation_equivalence","query_role":"referent"}),
    }
    zh: dict[str, tuple[str, str, str, dict]] = {
        "taxonomy": (f"{x1}属于水果类，{x2}属于工具类。哪一个属于水果类？", x1, x2, {"edge":"is_a","query_role":"member"}),
        "part_of": (f"琥珀模块是{y1}的一部分，钴蓝模块是{y2}的一部分。{query}配对的是{'琥珀' if query == x1 else '钴蓝'}模块。这个模块属于哪个整体？", y1 if query == x1 else y2, y2 if query == x1 else y1, {"edge":"part_of","query_role":"whole"}),
        "product_of": (f"{x1}生产{y1}，{x2}生产{y2}。{query}生产什么？", target, foil, {"edge":"product_of","query_role":"product"}),
        "word_sense": (f"在词典中，金融语境里的“露玛”指{y1}，河流语境里的“露玛”指{y2}。句子说：“他们在露玛存入硬币。”这里的“露玛”指什么？", y1, y2, {"edge":"sense_in_context","query_role":"sense"}),
        "role_binding": (f"{x1}感谢了{y1}，{x2}感谢了{y2}。谁接受了{query}的感谢？", target, foil, {"edge":"recipient","query_role":"object"}),
        "preference_binding": (f"{x1}喜欢{y1}，{x2}喜欢{y2}。{query}喜欢什么？", target, foil, {"edge":"preference","query_role":"object"}),
        "negation_scope": (f"{x1}没有启动，而{x2}已经启动。哪一个已经启动？", x2, x1, {"edge":"negation_scope","query_role":"positive"}),
        "temporal_order": (f"{x1}在{x2}之前到达。哪一个到得更晚？", x2, x1, {"edge":"before","query_role":"later"}),
        "causal_chain": (f"{x1}导致了{y1}，{x2}导致了{y2}。{query}导致了什么？", target, foil, {"edge":"causes","query_role":"effect"}),
        "coreference_binding": (f"{x1}对{y1}说：“你获胜了。”{x2}当时在别处。“你”指谁？", y1, x2, {"edge":"coreference","query_role":"referent"}),
        "punctuation_attachment": (f"{x1}，而不是{x2}，打开了门。谁打开了门？", x1, x2, {"edge":"punctuation_scope","query_role":"agent"}),
        "spatial_relation": (f"{x1}在{x2}左边。哪一个在右边？", x2, x1, {"edge":"left_of","query_role":"right"}),
        "sentence_reordering": (f"事件{x1}发生在事件{x2}之后。哪个事件先发生？", x2, x1, {"edge":"discourse_order","query_role":"first"}),
        "translation_binding": (f"对照表说明中文标签“甲类”指{x1}，“乙类”指{x2}。“甲类”指什么？", x1, x2, {"edge":"translation_equivalence","query_role":"referent"}),
    }
    return (en if language == "en" else zh)[family]


def compile_rows(tokenizer) -> list[dict]:
    rows: list[dict] = []
    case = 44881
    for unit, names in UNITS.items():
        for family in FAMILIES:
            for language in ("en", "zh"):
                for trial in range(4):
                    statement, target, foil, edge = content(family, names, trial, language)
                    candidates = [target, foil] if trial % 2 == 0 else [foil, target]
                    for interface in INTERFACES:
                        if language == "en":
                            if interface == "entity":
                                suffix = f"\nCandidates: {candidates[0]} | {candidates[1]}\nReturn exactly one candidate name.\nAnswer:"
                                expected = target
                            else:
                                suffix = f"\n1 = {candidates[0]}; 2 = {candidates[1]}\nReturn exactly 1 or 2.\nAnswer:"
                                expected = "1" if target == candidates[0] else "2"
                        else:
                            if interface == "entity":
                                suffix = f"\n候选：{candidates[0]} | {candidates[1]}\n只返回一个候选名称。\n答案："
                                expected = target
                            else:
                                suffix = f"\n1 = {candidates[0]}；2 = {candidates[1]}\n只返回1或2。\n答案："
                                expected = "1" if target == candidates[0] else "2"
                        prompt = statement + suffix
                        ids = [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]
                        spans = {}
                        for label, value in (("candidate0", candidates[0]), ("candidate1", candidates[1]), ("target", target), ("foil", foil)):
                            value_ids = [int(x) for x in tokenizer.encode(value, add_special_tokens=False)]
                            positions = []
                            for start in range(len(ids) - len(value_ids) + 1):
                                if ids[start:start + len(value_ids)] == value_ids:
                                    positions.append([start, start + len(value_ids)])
                            spans[label] = {"text": value, "token_ids": value_ids, "occurrences": positions}
                        rows.append({
                            "case_id": f"c{case:05d}-{family}-u{unit}-{language}-t{trial}-{interface}",
                            "unit": unit,
                            "split": {8:"discovery",9:"confirmation",10:"lockbox"}[unit],
                            "family": family,
                            "language": language,
                            "surface": trial,
                            "output_interface": interface,
                            "statement": statement,
                            "prompt": prompt,
                            "prompt_ids": ids,
                            "answer_boundary_token": len(ids) - 1,
                            "candidates": candidates,
                            "target": target,
                            "foil": foil,
                            "expected_output": expected,
                            "typed_edge": {**edge, "source": target if edge["query_role"] in ("member", "agent") else None, "target": target},
                            "tokenizer_spans": spans,
                        })
                        case += 1
    return rows


def strip_prefix(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^(?:final\s+answer|answer|最终答案|答案)\s*[:：]\s*", "", value, flags=re.I)
    return value.strip()


def parse_answer(text: str, row: dict) -> tuple[str | None, bool, bool]:
    cleaned = strip_prefix(text)
    prefix = cleaned != text.strip()
    if row["output_interface"] == "code":
        match = re.search(r"(?<!\d)([12])(?!\d)", cleaned)
        parsed = match.group(1) if match else None
    else:
        hits = []
        norm = normalize(cleaned)
        for candidate in row["candidates"]:
            if normalize(candidate) in norm:
                hits.append(candidate)
        parsed = hits[0] if len(set(hits)) == 1 else None
    return parsed, parsed == row["expected_output"], prefix


def run_behavior(model, tokenizer, rows: list[dict]) -> list[dict]:
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    generated: list[dict] = []
    batch_size = 8
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer([row["prompt"] for row in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=12,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        width = encoded["input_ids"].shape[1]
        for row, sequence in zip(batch, output):
            new_ids = [int(x) for x in sequence[width:].detach().cpu().tolist()]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            parsed, correct, prefix = parse_answer(text, row)
            generated.append({
                "case_id": row["case_id"],
                "unit": row["unit"],
                "family": row["family"],
                "language": row["language"],
                "surface": row["surface"],
                "output_interface": row["output_interface"],
                "expected": row["expected_output"],
                "generated_ids": new_ids,
                "generated_text": text,
                "parsed_answer": parsed,
                "parsed_correct": correct,
                "answer_prefix": prefix,
                "raw_normalized_exact": normalize(strip_prefix(text)) == normalize(row["expected_output"]),
            })
        if (start + len(batch)) % 64 == 0:
            print(f"[phase2469 behavior] {start + len(batch)}/{len(rows)}", flush=True)
    return generated


def summarize(generated: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for item in generated:
        grouped[(item["unit"], item["output_interface"], item["family"])].append(item)
    detail: dict[str, dict] = {}
    for unit in UNITS:
        detail[str(unit)] = {}
        for interface in INTERFACES:
            detail[str(unit)][interface] = {}
            for family in FAMILIES:
                values = grouped[(unit, interface, family)]
                detail[str(unit)][interface][family] = {
                    "rows": len(values),
                    "parsed_accuracy": sum(x["parsed_correct"] for x in values) / len(values),
                    "raw_normalized_exact": sum(x["raw_normalized_exact"] for x in values) / len(values),
                    "prefix_rate": sum(x["answer_prefix"] for x in values) / len(values),
                    "unparsed_rate": sum(x["parsed_answer"] is None for x in values) / len(values),
                    "en_accuracy": sum(x["parsed_correct"] for x in values if x["language"] == "en") / 4,
                    "zh_accuracy": sum(x["parsed_correct"] for x in values if x["language"] == "zh") / 4,
                }
            all_values = [x for x in generated if x["unit"] == unit and x["output_interface"] == interface]
            detail[str(unit)][interface]["aggregate"] = {
                "rows": len(all_values),
                "parsed_accuracy": sum(x["parsed_correct"] for x in all_values) / len(all_values),
                "raw_normalized_exact": sum(x["raw_normalized_exact"] for x in all_values) / len(all_values),
                "prefix_rate": sum(x["answer_prefix"] for x in all_values) / len(all_values),
                "unparsed_rate": sum(x["parsed_answer"] is None for x in all_values) / len(all_values),
            }
    qualified = {}
    for interface in INTERFACES:
        qualified[interface] = [family for family in FAMILIES if detail["9"][interface][family]["parsed_accuracy"] >= 0.75 and detail["10"][interface][family]["parsed_accuracy"] >= 0.75]
    qualified["both_interfaces"] = sorted(set(qualified["entity"]) & set(qualified["code"]))
    return {"by_unit_interface_family": detail, "qualified_families": qualified}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    agg = {unit: {interface: result["behavior"]["by_unit_interface_family"][unit][interface]["aggregate"] for interface in INTERFACES} for unit in ("8", "9", "10")}
    text = rf"""


## Phase {PHASE}: 十四类类型化语言超图、Tokenizer/span审计与双接口自主行为锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 建立14类关系（taxonomy、part-of、product-of、词义、角色、偏好、否定、时间、因果、指代、标点、空间、篇章顺序、翻译绑定），unit8/9/10分别作discovery/confirmation/lockbox；每族中英、四个表面/内容试次、实体名与1/2代码双接口，共672条。每条保存typed edge、完整prompt token IDs、候选/目标/foil token IDs及所有span位置。真实贪心允许12个新token，先剥离`Answer:/答案：`，再解析完整候选或冻结代码；raw exact、parsed exact、前缀率和无法解析率并报。

$$N=14\times3\times2\times4\times2=672,\qquad Q_{{beh}}(f,o)=\mathbb 1[A_{{u9}}\ge0.75\land A_{{u10}}\ge0.75].$$

**结果汇总。** 聚合行为 `{json.dumps(agg, ensure_ascii=False)}`；冻结合格族 `{json.dumps(result['behavior']['qualified_families'], ensure_ascii=False)}`；tokenizer审计 `{json.dumps(result['tokenizer_audit'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2469_c44881_c45440_typed_hypergraph_behavior.py`；672条统一材料、逐行生成、tokenizer审计和final位于同名结果目录。

**分析与理论进展。** 本Phase只建立外部实验图与行为资格，不宣称模型内部以超图存储。它修复Phase2466的预算和解析问题，使后续自主轨迹只在confirmation与lockbox均达到0.75的族上升级为语言执行证据。实体与代码接口共享语义问题但输出身份不同，为回答边界分化提供受控材料。

**问题硬伤与结论。** 显式记录使任务更像上下文推理而非参数知识；四试次/语言仍包含同模板相关性。typed edge是研究者标签。代码接口即使行为合格，也可能使用位置映射。下一步保留所有物理坐标，先采全层全token原场和基本差分，不用低秩压缩替代原数据。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows = compile_rows(tokenizer)
        material = OUT / "material/typed_hypergraph_rows.jsonl"
        material.parent.mkdir(parents=True, exist_ok=True)
        material.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        generated = run_behavior(model, tokenizer, rows)
    finally:
        model_utils.release_model(model)
        gc.collect()
    generation_path = OUT / "behavior/autonomous_generation.jsonl"
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    generation_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated), encoding="utf-8")
    behavior = summarize(generated)
    audit = {
        "rows": len(rows),
        "all_prompt_ids": all(len(row["prompt_ids"]) > 0 for row in rows),
        "all_candidate_ids": all(all(len(row["tokenizer_spans"][key]["token_ids"]) > 0 for key in ("candidate0", "candidate1", "target", "foil")) for row in rows),
        "span_occurrence_rate": sum(bool(row["tokenizer_spans"]["target"]["occurrences"]) for row in rows) / len(rows),
        "prompt_token_length": {"min": min(len(row["prompt_ids"]) for row in rows), "max": max(len(row["prompt_ids"]) for row in rows), "mean": sum(len(row["prompt_ids"]) for row in rows) / len(rows)},
    }
    checks = {
        "rows_672": len(rows) == 672,
        "generated_672": len(generated) == 672,
        "three_frozen_units": sorted({row["unit"] for row in rows}) == [8, 9, 10],
        "fourteen_families": len({row["family"] for row in rows}) == 14,
        "two_interfaces": sorted({row["output_interface"] for row in rows}) == ["code", "entity"],
        "tokenizer_audited": audit["all_prompt_ids"] and audit["all_candidate_ids"],
        "finite": all(math.isfinite(x["parsed_accuracy"]) for unit in behavior["by_unit_interface_family"].values() for interface in INTERFACES for x in unit[interface].values()),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "material": {"path": str(material), "rows": len(rows), "families": list(FAMILIES), "units": UNITS, "interfaces": list(INTERFACES)},
        "generation": str(generation_path),
        "tokenizer_audit": audit,
        "behavior": behavior,
        "adjudication": {"typed_hypergraph_is_internal_mechanism": False, "behavior_gate_only": True, "language_encoding_mechanism_closed": False},
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
