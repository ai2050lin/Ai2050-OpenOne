#!/usr/bin/env python3
"""Orthogonal 12-family x 4-surface x 4-interface autonomous behavior contract."""
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
OUT = TESTS / "result/phase2487_c54721_c55872_orthogonal_family_interface_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2487, "C54721-C55872"
UNITS = (14, 15, 16)
SPLIT = {14: "discovery", 15: "confirmation", 16: "lockbox"}
INTERFACES = ("entity", "digit", "letter", "side")
FAMILIES = (
    "taxonomy", "part_whole", "product", "causal", "temporal", "spatial",
    "role", "preference", "membership", "translation", "coreference", "punctuation",
)
EN_NAMES = {
    14: ("Aroven", "Belune", "Cireth", "Dovian", "Elaro", "Fesin", "Gavrel", "Hunia"),
    15: ("Ivaren", "Jorune", "Kelith", "Lumera", "Mavon", "Neris", "Orveth", "Palia"),
    16: ("Quaren", "Rovik", "Selune", "Tavian", "Umera", "Virel", "Worin", "Xaleth"),
}
ZH_NAMES = {
    14: ("岚舟", "沐岩", "青禾", "若川", "星澜", "云岫", "竹汀", "砚秋"),
    15: ("安澈", "白榆", "初弦", "冬砚", "归岚", "禾汀", "静川", "临舟"),
    16: ("明屿", "宁河", "朴山", "清禾", "如松", "素川", "听澜", "微岫"),
}

# Every family has four independently worded predicates, but uses the same record/query skeleton.
PREDICATES: dict[str, dict[str, tuple[str, ...]]] = {
    "taxonomy": {"en": ("belongs to category", "is classified as", "is a kind of", "falls under type"), "zh": ("属于类别", "被归类为", "是一种", "归入类型")},
    "part_whole": {"en": ("is a component of", "forms part of", "is contained in", "is a module within"), "zh": ("是组成部分，隶属于", "构成了", "包含在", "是内部模块，位于")},
    "product": {"en": ("produces", "creates", "manufactures", "yields"), "zh": ("生产", "制造", "产出", "生成")},
    "causal": {"en": ("causes", "brings about", "triggers", "leads to"), "zh": ("导致", "引起", "触发", "造成")},
    "temporal": {"en": ("occurs before", "precedes", "happens earlier than", "takes place prior to"), "zh": ("发生在之前：", "先于", "比更早发生：", "在时间上早于")},
    "spatial": {"en": ("stands left of", "is west of", "is positioned before", "occupies the left side of"), "zh": ("位于左侧：", "在西边：", "排在前面：", "处在左方：")},
    "role": {"en": ("reports to", "is supervised by", "answers to", "works under"), "zh": ("向汇报：", "受监督于", "听命于", "在手下工作：")},
    "preference": {"en": ("prefers", "favors", "likes best", "chooses by preference"), "zh": ("偏爱", "更喜欢", "最喜欢", "优先选择")},
    "membership": {"en": ("joins group", "is enrolled in", "holds membership in", "belongs with"), "zh": ("加入组", "登记在", "是成员，属于", "归属到")},
    "translation": {"en": ("is translated as", "has the codebook equivalent", "corresponds across languages to", "maps in translation to"), "zh": ("翻译为", "在对照表中等价于", "跨语言对应于", "在翻译中映射到")},
    "coreference": {"en": ("uses the pronoun to refer to", "has a pronoun pointing to", "binds the reference to", "resolves the pronoun as"), "zh": ("用代词指代", "使代词指向", "把指称绑定到", "将代词解析为")},
    "punctuation": {"en": ("is selected by the comma contrast over", "wins the parenthetical contrast against", "is the attached item instead of", "is chosen by punctuation rather than"), "zh": ("由逗号对比选中而非", "在插入语对比中胜过", "是标点附着项而不是", "由标点选择而非")},
}

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _occurrences(tokenizer, prompt: str, text: str) -> list[list[int]]:
    spans: list[list[int]] = []
    start = 0
    while True:
        char_start = prompt.find(text, start)
        if char_start < 0:
            break
        char_end = char_start + len(text)
        spans.append([
            len(tokenizer.encode(prompt[:char_start], add_special_tokens=False)),
            len(tokenizer.encode(prompt[:char_end], add_special_tokens=False)),
        ])
        start = char_end
    return spans


def compile_rows(tokenizer) -> list[dict]:
    rows: list[dict] = []
    case = 54721
    for unit in UNITS:
        for family_index, family in enumerate(FAMILIES):
            for language in ("en", "zh"):
                base_names = EN_NAMES[unit] if language == "en" else ZH_NAMES[unit]
                shift = (2 * family_index) % len(base_names)
                names = base_names[shift:] + base_names[:shift]
                for surface in range(4):
                    # Surfaces balance queried record, record order, and candidate order for every family.
                    a, b, c, d = names[0], names[1], names[2], names[3]
                    query_source, target, foil = (a, b, d) if surface in (0, 3) else (c, d, b)
                    records = [(a, b), (c, d)]
                    if surface in (1, 3):
                        records.reverse()
                    predicate = PREDICATES[family][language][surface]
                    if language == "en":
                        statement = f"Record one: {records[0][0]} {predicate} {records[0][1]}. Record two: {records[1][0]} {predicate} {records[1][1]}. Query source: {query_source}. Select the item linked to that source by the stated relation."
                    else:
                        statement = f"记录一：{records[0][0]}{predicate}{records[0][1]}。记录二：{records[1][0]}{predicate}{records[1][1]}。查询源：{query_source}。请选择按上述关系与该来源相连的项目。"
                    candidates = [target, foil] if surface % 2 == 0 else [foil, target]
                    for interface in INTERFACES:
                        answer_codes = {
                            "digit": ("1", "2"), "letter": ("A", "B"), "side": ("LEFT", "RIGHT")
                        }
                        if language == "en":
                            if interface == "entity":
                                suffix = f"\nCandidates: {candidates[0]} | {candidates[1]}\nReturn exactly the linked item name.\nAnswer:"
                                expected = target
                            else:
                                left, right = answer_codes[interface]
                                suffix = f"\n{left} = {candidates[0]}; {right} = {candidates[1]}\nReturn exactly {left} or {right}.\nAnswer:"
                                expected = left if target == candidates[0] else right
                        else:
                            if interface == "entity":
                                suffix = f"\n候选：{candidates[0]} | {candidates[1]}\n只返回相连项目的名称。\n答案："
                                expected = target
                            else:
                                left, right = answer_codes[interface]
                                suffix = f"\n{left} = {candidates[0]}；{right} = {candidates[1]}\n只返回{left}或{right}。\n答案："
                                expected = left if target == candidates[0] else right
                        prompt = statement + suffix
                        prompt_ids = [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]
                        spans = {key: _occurrences(tokenizer, prompt, value) for key, value in {
                            "query_source": query_source, "target": target, "foil": foil, "predicate": predicate,
                        }.items()}
                        rows.append({
                            "case_id": f"c{case:05d}-{family}-u{unit}-{language}-s{surface}-{interface}",
                            "unit": unit, "split": SPLIT[unit], "family": family, "language": language,
                            "surface": surface, "output_interface": interface, "statement": statement,
                            "predicate": predicate, "abstract_graph": [["n0", "n1"], ["n2", "n3"]],
                            "query_abstract_node": "n0" if query_source == a else "n2",
                            "target_abstract_node": "n1" if target == b else "n3",
                            "query_source": query_source, "target": target, "foil": foil,
                            "candidates": candidates, "expected_output": expected, "prompt": prompt,
                            "prompt_ids": prompt_ids, "answer_boundary_token": len(prompt_ids) - 1,
                            "token_spans": spans,
                        })
                        case += 1
    return rows


def strip_prefix(text: str) -> str:
    return re.sub(r"^(?:final\s+answer|answer|最终答案|答案)\s*[:：]\s*", "", text.strip(), flags=re.I).strip()


def parse_answer(text: str, row: dict) -> tuple[str | None, bool, bool]:
    cleaned = strip_prefix(text)
    prefix = cleaned != text.strip()
    expected_set = {"digit": ("1", "2"), "letter": ("A", "B"), "side": ("LEFT", "RIGHT")}
    if row["output_interface"] == "entity":
        norm = normalize(cleaned)
        hits = [candidate for candidate in row["candidates"] if normalize(candidate) in norm]
        parsed = hits[0] if len(set(hits)) == 1 else None
    else:
        choices = expected_set[row["output_interface"]]
        matches = re.findall(r"(?<![A-Z0-9])(" + "|".join(choices) + r")(?![A-Z0-9])", cleaned.upper())
        parsed = matches[0] if len(set(matches)) == 1 else None
    return parsed, parsed == row["expected_output"], prefix


def run_behavior(model, tokenizer, rows: list[dict]) -> list[dict]:
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    output: list[dict] = []
    batch_size = 8
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer([row["prompt"] for row in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=10, do_sample=False, use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        width = encoded["input_ids"].shape[1]
        for row, sequence in zip(batch, generated):
            ids = [int(x) for x in sequence[width:].detach().cpu().tolist()]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            parsed, correct, prefix = parse_answer(text, row)
            output.append({
                "case_id": row["case_id"], "unit": row["unit"], "family": row["family"],
                "language": row["language"], "surface": row["surface"],
                "output_interface": row["output_interface"], "expected": row["expected_output"],
                "generated_ids": ids, "generated_text": text, "parsed_answer": parsed,
                "parsed_correct": correct, "answer_prefix": prefix,
            })
        if (start + len(batch)) % 128 == 0:
            print(f"[phase2487] {start + len(batch)}/{len(rows)}", flush=True)
    return output


def summarize(generated: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in generated:
        groups[(row["unit"], row["family"], row["output_interface"])].append(row)
    detail: dict[str, dict] = {}
    qualified: dict[str, list[str]] = {interface: [] for interface in INTERFACES}
    for unit in UNITS:
        detail[str(unit)] = {}
        for family in FAMILIES:
            detail[str(unit)][family] = {}
            for interface in INTERFACES:
                values = groups[(unit, family, interface)]
                detail[str(unit)][family][interface] = {
                    "rows": len(values),
                    "accuracy": sum(x["parsed_correct"] for x in values) / len(values),
                    "en_accuracy": sum(x["parsed_correct"] for x in values if x["language"] == "en") / 4,
                    "zh_accuracy": sum(x["parsed_correct"] for x in values if x["language"] == "zh") / 4,
                    "unparsed_rate": sum(x["parsed_answer"] is None for x in values) / len(values),
                }
    for interface in INTERFACES:
        for family in FAMILIES:
            ok = True
            for unit in (15, 16):
                item = detail[str(unit)][family][interface]
                ok &= item["accuracy"] >= 0.75 and item["en_accuracy"] >= 0.5 and item["zh_accuracy"] >= 0.5
            if ok:
                qualified[interface].append(family)
    aggregate = {}
    for unit in UNITS:
        aggregate[str(unit)] = {}
        for interface in INTERFACES:
            values = [x for x in generated if x["unit"] == unit and x["output_interface"] == interface]
            aggregate[str(unit)][interface] = {
                "rows": len(values), "accuracy": sum(x["parsed_correct"] for x in values) / len(values),
                "unparsed_rate": sum(x["parsed_answer"] is None for x in values) / len(values),
            }
    common_families = sorted(set.intersection(*(set(qualified[i]) for i in INTERFACES)))
    interface_coverage = {i: len(qualified[i]) for i in INTERFACES}
    common_interfaces = [i for i in INTERFACES if len(qualified[i]) >= 8]
    return {"detail": detail, "aggregate": aggregate, "qualified": qualified,
            "common_all_interface_families": common_families,
            "interface_coverage": interface_coverage, "common_interfaces": common_interfaces}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 十二语言族×四表面×四输出接口正交材料与1152条自主行为资格（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 使用taxonomy、part-whole、product、causal、temporal、spatial、role、preference、membership、translation、coreference、punctuation十二族；unit14/15/16分别发现/确认/锁箱，中英使用完全不同字符串，只用外部抽象节点$n0\to n1,n2\to n3$对应。每族都有四套谓词表达，但共享相同Record/Query骨架；四surface把查询记录、事实顺序、候选顺序完全交叉；entity、1/2、A/B、LEFT/RIGHT四接口只改变输出编码。Qwen3-4B非量化BF16 CUDA真实贪心，最多10 token，剥离可选答案前缀后解析。

$$N=12\times3\times2\times4\times4=1152,$$

$$Q(f,o)=\mathbb 1\left[\bigwedge_{{u\in\{{15,16\}}}} A_{{u,f,o}}\ge0.75\land A_{{u,f,o,en}},A_{{u,f,o,zh}}\ge0.50\right].$$

**结果汇总。** 聚合行为 `{json.dumps(result['behavior']['aggregate'], ensure_ascii=False)}`；各接口合格族 `{json.dumps(result['behavior']['qualified'], ensure_ascii=False)}`；覆盖 `{json.dumps(result['behavior']['interface_coverage'], ensure_ascii=False)}`；四接口共同族 `{json.dumps(result['behavior']['common_all_interface_families'], ensure_ascii=False)}`；Tokenizer `{json.dumps(result['tokenizer_audit'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2487_c54721_c55872_orthogonal_family_interface_behavior.py`；1152条材料、逐行真实生成和`analysis/final.json`位于同名结果目录。

**分析与理论进展。** 这一步显著削弱“每族独占整套句型”和“中英文复用同名token”两个混杂，并把四种输出编码预先交叉。合格门只决定哪些行为可叫成功语言执行；不合格族仍保留为输入响应，不把小模型失败解释成机制不存在。

**问题硬伤与结论。** 关系谓词仍然与family定义共变，因而family纹理仍是“关系语义+谓词词项”的组合，不能叫纯语义。四接口本质上三种位置代码加实体名，不覆盖开放长文本。合成记录任务不是参数知识。当前只完成大样本行为合同，尚未读取任何坐标机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows = compile_rows(tokenizer)
        write_jsonl(OUT / "material/orthogonal_family_interface_rows.jsonl", rows)
        generated = run_behavior(model, tokenizer, rows)
    finally:
        model_utils.release_model(model)
        gc.collect()
    write_jsonl(OUT / "behavior/autonomous_generation.jsonl", generated)
    behavior = summarize(generated)
    audit = {
        "prompt_tokens": {"min": min(len(r["prompt_ids"]) for r in rows), "max": max(len(r["prompt_ids"]) for r in rows),
                          "mean": sum(len(r["prompt_ids"]) for r in rows) / len(rows)},
        "all_event_spans": all(all(r["token_spans"][k] for k in ("query_source", "target", "foil", "predicate")) for r in rows),
        "english_chinese_name_overlap": sorted(set(sum((list(v) for v in EN_NAMES.values()), [])) & set(sum((list(v) for v in ZH_NAMES.values()), []))),
    }
    checks = {
        "rows_1152": len(rows) == 1152, "generated_1152": len(generated) == 1152,
        "twelve_families": len({r["family"] for r in rows}) == 12,
        "four_surfaces_each": all(len({r["surface"] for r in rows if r["family"] == f}) == 4 for f in FAMILIES),
        "four_interfaces": sorted({r["output_interface"] for r in rows}) == sorted(INTERFACES),
        "independent_language_names": not audit["english_chinese_name_overlap"],
        "token_spans_complete": audit["all_event_spans"],
        "finite": all(math.isfinite(item["accuracy"]) for unit in behavior["detail"].values() for family in unit.values() for item in family.values()),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "model": {"name": "Qwen3-4B", "dtype": "BF16", "quantized": False, "device": "cuda:0"},
        "material": {"path": str(OUT / "material/orthogonal_family_interface_rows.jsonl"), "rows": len(rows),
                     "families": list(FAMILIES), "interfaces": list(INTERFACES), "units": list(UNITS)},
        "generation": str(OUT / "behavior/autonomous_generation.jsonl"),
        "tokenizer_audit": audit, "behavior": behavior,
        "adjudication": {"orthogonal_behavior_contract_complete": True, "family_lexeme_deconfounded": False,
                         "internal_language_graph_identified": False, "language_encoding_mechanism_closed": False},
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
