#!/usr/bin/env python3
"""Freeze a 12-family bilingual lexical/task/surface/output-code factorial atlas."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = OUT / "material/bilingual_factorial_fixed_code.jsonl"
BLIND_PACK = OUT / "material/naturality_blind_pack.jsonl"
PHASE = 2344
CAMPAIGN = "C8241-C8360"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402

FAMILIES = (
    "attitude_event", "comparison_order", "location_binding", "relative_binding",
    "taxonomy_chain", "attribute_binding", "causal_relation", "negation_scope",
    "quantifier_scope", "coreference_anaphora", "punctuation_boundary", "translation_equivalence",
)
MACROTYPE = {
    "attitude_event": "context_scope", "comparison_order": "logical_relation",
    "location_binding": "relational_binding", "relative_binding": "grammar_binding",
    "taxonomy_chain": "concept_graph", "attribute_binding": "concept_graph",
    "causal_relation": "logical_relation", "negation_scope": "context_scope",
    "quantifier_scope": "logical_relation", "coreference_anaphora": "context_scope",
    "punctuation_boundary": "form_boundary", "translation_equivalence": "cross_language",
}
LANGUAGES = ("en", "zh")
LEXICAL_SETS = ("lex0", "lex1")
TASKS = ("select_supported", "select_contradicted")
SURFACES = ("direct", "reported", "natural")
CODEBOOKS = {"AB": ("A", "B"), "CD": ("C", "D")}
CONDITIONS = ("original", "option_swapped")
PARTITIONS = ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")
UNITS = 16
PARTITION_BY_UNIT = {unit: PARTITIONS[unit // 4] for unit in range(UNITS)}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def graph_hash(graph: list[list[str]]) -> str:
    return hashlib.sha256(canonical(graph).encode("utf-8")).hexdigest()


def entities(unit: int, lexical_set: str) -> dict[str, str]:
    offset = unit + (0 if lexical_set == "lex0" else 100)
    return {
        "x": f"Nilo-{offset:03d}", "y": f"Rava-{offset:03d}",
        "object": f"item-{offset:03d}", "object2": f"item-{offset + 300:03d}",
        "place": f"zone-{offset:03d}", "place2": f"zone-{offset + 300:03d}",
        "class": f"class-{offset:03d}", "class2": f"class-{offset + 300:03d}",
        "property": f"property-{offset:03d}", "label": f"term-{offset:03d}",
    }


def render(family: str, language: str, e: dict[str, str], state: int) -> dict:
    target, wrong = ((e["x"], e["y"]) if state == 0 else (e["y"], e["x"]))
    if family == "attitude_event":
        graph = [[target, "values", e["object"]], [wrong, "avoids", e["object"]]]
        if language == "en":
            direct = f'{target} values {e["object"]}; {wrong} avoids it.'
            natural = f'Whenever {e["object"]} appears, {target} welcomes it, whereas {wrong} keeps away.'
            true, false = f'{target} has the positive attitude toward {e["object"]}.', f'{wrong} has the positive attitude toward {e["object"]}.'
        else:
            direct = f'{target}重视{e["object"]}；{wrong}回避它。'
            natural = f'每当出现{e["object"]}，{target}都会欢迎它，而{wrong}会避开。'
            true, false = f'对{e["object"]}持积极态度的是{target}。', f'对{e["object"]}持积极态度的是{wrong}。'
    elif family == "comparison_order":
        graph = [[target, "ranked_above", wrong]]
        if language == "en":
            direct = f'{target} ranks above {wrong}.'
            natural = f'In the ordering, {wrong} comes below {target}.'
            true, false = f'{target} is higher in the order.', f'{wrong} is higher in the order.'
        else:
            direct = f'{target}的排序高于{wrong}。'
            natural = f'在这个次序中，{wrong}排在{target}下面。'
            true, false = f'排序较高的是{target}。', f'排序较高的是{wrong}。'
    elif family == "location_binding":
        graph = [[e["object"], "located_in", target], [e["object2"], "located_in", wrong]]
        if language == "en":
            direct = f'{e["object"]} is in {target}; {e["object2"]} is in {wrong}.'
            natural = f'Inside {target} one finds {e["object"]}, while {wrong} contains {e["object2"]}.'
            true, false = f'{e["object"]} is located in {target}.', f'{e["object"]} is located in {wrong}.'
        else:
            direct = f'{e["object"]}在{target}中；{e["object2"]}在{wrong}中。'
            natural = f'{target}里面放着{e["object"]}，而{wrong}里面是{e["object2"]}。'
            true, false = f'{e["object"]}位于{target}。', f'{e["object"]}位于{wrong}。'
    elif family == "relative_binding":
        graph = [[target, "sibling_of", e["object"]], [wrong, "colleague_of", e["object"]]]
        if language == "en":
            direct = f'{target} is the sibling of {e["object"]}; {wrong} is only a colleague.'
            natural = f'{e["object"]} and {target} are siblings, but {wrong} works with them.'
            true, false = f'The sibling of {e["object"]} is {target}.', f'The sibling of {e["object"]} is {wrong}.'
        else:
            direct = f'{target}是{e["object"]}的手足；{wrong}只是同事。'
            natural = f'{e["object"]}与{target}是手足关系，而{wrong}和他们一起工作。'
            true, false = f'{e["object"]}的手足是{target}。', f'{e["object"]}的手足是{wrong}。'
    elif family == "taxonomy_chain":
        item, foil = ((e["object"], e["object2"]) if state == 0 else (e["object2"], e["object"]))
        graph = [[item, "instance_of", e["class"]], [foil, "instance_of", e["class2"]]]
        if language == "en":
            direct = f'{item} belongs to {e["class"]}; {foil} belongs to {e["class2"]}.'
            natural = f'The catalog files {item} under {e["class"]}, not under {e["class2"]}.'
            true, false = f'{item} is a member of {e["class"]}.', f'{foil} is a member of {e["class"]}.'
        else:
            direct = f'{item}属于{e["class"]}；{foil}属于{e["class2"]}。'
            natural = f'目录把{item}归入{e["class"]}，而不是{e["class2"]}。'
            true, false = f'{item}是{e["class"]}的成员。', f'{foil}是{e["class"]}的成员。'
        target, wrong = item, foil
    elif family == "attribute_binding":
        item, foil = ((e["object"], e["object2"]) if state == 0 else (e["object2"], e["object"]))
        graph = [[item, "has_property", e["property"]], [foil, "lacks_property", e["property"]]]
        if language == "en":
            direct = f'{item} has {e["property"]}; {foil} lacks it.'
            natural = f'The inspection found {e["property"]} on {item}, never on {foil}.'
            true, false = f'{e["property"]} belongs to {item}.', f'{e["property"]} belongs to {foil}.'
        else:
            direct = f'{item}具有{e["property"]}；{foil}不具有。'
            natural = f'检查只在{item}上发现了{e["property"]}，在{foil}上没有发现。'
            true, false = f'具有{e["property"]}的是{item}。', f'具有{e["property"]}的是{foil}。'
        target, wrong = item, foil
    elif family == "causal_relation":
        graph = [[target, "caused", e["object"] + "_change"], [wrong, "observed", e["object"] + "_change"]]
        if language == "en":
            direct = f'{target} opened a valve, causing {e["object"]} to change; {wrong} only observed.'
            natural = f'Because of the valve opened by {target}, {e["object"]} changed as {wrong} watched.'
            true, false = f'{target} caused the change in {e["object"]}.', f'{wrong} caused the change in {e["object"]}.'
        else:
            direct = f'{target}打开阀门，导致{e["object"]}变化；{wrong}只是在观察。'
            natural = f'由于{target}打开了阀门，{e["object"]}在{wrong}观察时发生变化。'
            true, false = f'导致{e["object"]}变化的是{target}。', f'导致{e["object"]}变化的是{wrong}。'
    elif family == "negation_scope":
        graph = [[target, "carried", e["object"]], [wrong, "not_carried", e["object"]]]
        if language == "en":
            direct = f'{target} carried {e["object"]}; {wrong} did not carry it.'
            natural = f'It was {target}, not {wrong}, who arrived with {e["object"]}.'
            true, false = f'{target} carried {e["object"]}.', f'{wrong} carried {e["object"]}.'
        else:
            direct = f'{target}携带了{e["object"]}；{wrong}没有携带它。'
            natural = f'带着{e["object"]}到场的是{target}，不是{wrong}。'
            true, false = f'{target}携带了{e["object"]}。', f'{wrong}携带了{e["object"]}。'
    elif family == "quantifier_scope":
        graph = [["only", "scope", target], [target, "inspected", e["object"]], [wrong, "not_inspected", e["object"]]]
        if language == "en":
            direct = f'Only {target}, not {wrong}, inspected {e["object"]}.'
            natural = f'Of the two, the sole inspector of {e["object"]} was {target}.'
            true, false = f'{target} inspected {e["object"]}.', f'{wrong} inspected {e["object"]}.'
        else:
            direct = f'只有{target}检查了{e["object"]}，{wrong}没有检查。'
            natural = f'两者之中，{e["object"]}唯一的检查者是{target}。'
            true, false = f'{target}检查了{e["object"]}。', f'{wrong}检查了{e["object"]}。'
    elif family == "coreference_anaphora":
        graph = [["former", "refers_to", target], ["latter", "refers_to", wrong]]
        if language == "en":
            direct = f'{target} called {wrong}; “the former” means {target} and “the latter” means {wrong}.'
            natural = f'After naming {target} before {wrong}, the note uses “the former” for the first name.'
            true, false = f'“The former” refers to {target}.', f'“The former” refers to {wrong}.'
        else:
            direct = f'{target}给{wrong}打电话；“前者”指{target}，“后者”指{wrong}。'
            natural = f'记录先提到{target}，再提到{wrong}，随后用“前者”指第一个名字。'
            true, false = f'“前者”指{target}。', f'“前者”指{wrong}。'
    elif family == "punctuation_boundary":
        mark, foil = (("question_mark", "period") if state == 0 else ("period", "question_mark"))
        graph = [["utterance", "has_function", "question" if state == 0 else "statement"], ["utterance", "closed_by", mark]]
        if language == "en":
            function = "a direct question" if state == 0 else "a declarative statement"
            direct = f'The utterance is {function}.'
            natural = f'Its communicative role is {function}, which determines the closing mark.'
            true, false = f'The correct closing mark is {mark}.', f'The correct closing mark is {foil}.'
        else:
            function = "直接疑问句" if state == 0 else "陈述句"
            direct = f'这个话语是{function}。'
            natural = f'它的交际功能属于{function}，因此结尾符号是确定的。'
            true, false = f'正确的结尾符号是{mark}。', f'正确的结尾符号是{foil}。'
        target, wrong = mark, foil
    elif family == "translation_equivalence":
        gloss, foil = ((e["class"], e["class2"]) if state == 0 else (e["class2"], e["class"]))
        graph = [[e["label"], "translates_to", gloss], [e["label"], "not_translates_to", foil]]
        if language == "en":
            direct = f'In this glossary, {e["label"]} translates as {gloss}, not {foil}.'
            natural = f'The bilingual index assigns {gloss} as the meaning of {e["label"]}; {foil} is a distractor.'
            true, false = f'{e["label"]} maps to {gloss}.', f'{e["label"]} maps to {foil}.'
        else:
            direct = f'在这份词表中，{e["label"]}翻译为{gloss}，不是{foil}。'
            natural = f'双语索引把{e["label"]}的含义记为{gloss}；{foil}是干扰项。'
            true, false = f'{e["label"]}对应{gloss}。', f'{e["label"]}对应{foil}。'
        target, wrong = gloss, foil
    else:
        raise KeyError(family)
    return {"graph": graph, "direct": direct, "natural": natural, "true": true, "false": false,
            "semantic_target": target, "semantic_wrong": wrong}


def prompt(language: str, surface: str, task: str, facts: str, option0: str, option1: str,
           code0: str, code1: str) -> str:
    if language == "en":
        if surface == "direct":
            opening = f"Record: {facts}"
        elif surface == "reported":
            opening = f'A registrar reported, “{facts}”'
        else:
            opening = f"Read this short account carefully. {facts}"
        rule = "supported by" if task == "select_supported" else "contradicted by"
        return (f"{opening}\nOption {code0}: {option0}\nOption {code1}: {option1}\n"
                f"Choose the one option {rule} the record. Reply with exactly {code0} or {code1}.\nAnswer:")
    if surface == "direct":
        opening = f"记录：{facts}"
    elif surface == "reported":
        opening = f"一位登记员报告：“{facts}”"
    else:
        opening = f"请仔细阅读这段简短叙述。{facts}"
    rule = "得到记录支持" if task == "select_supported" else "与记录矛盾"
    return (f"{opening}\n选项{code0}：{option0}\n选项{code1}：{option1}\n"
            f"选择唯一{rule}的选项，只回答{code0}或{code1}。\n答案：")


def compile_material(tokenizer) -> tuple[list[dict], dict, list[dict]]:
    code_ids = {}
    for code in sorted({code for pair in CODEBOOKS.values() for code in pair}):
        ids = tokenizer.encode(" " + code, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(("code_not_single_token", code, ids))
        code_ids[code] = int(ids[0])
    rows = []
    for condition in CONDITIONS:
        for family_index, family in enumerate(FAMILIES):
            for language_index, language in enumerate(LANGUAGES):
                for lexical_index, lexical_set in enumerate(LEXICAL_SETS):
                    for task_index, task in enumerate(TASKS):
                        for surface_index, surface in enumerate(SURFACES):
                            for codebook_index, (codebook, codes) in enumerate(CODEBOOKS.items()):
                                for unit in range(UNITS):
                                    for state in (0, 1):
                                        semantic = render(family, language, entities(unit, lexical_set), state)
                                        facts = semantic["natural"] if surface == "natural" else semantic["direct"]
                                        selected_truth = task == "select_supported"
                                        selected_text = semantic["true"] if selected_truth else semantic["false"]
                                        rejected_text = semantic["false"] if selected_truth else semantic["true"]
                                        target_first = ((unit + state + language_index + lexical_index + task_index
                                                         + surface_index + codebook_index) % 2 == 0)
                                        if condition == "option_swapped":
                                            target_first = not target_first
                                        option0 = selected_text if target_first else rejected_text
                                        option1 = rejected_text if target_first else selected_text
                                        target_code = codes[0] if target_first else codes[1]
                                        wrong_code = codes[1] if target_first else codes[0]
                                        text = prompt(language, surface, task, facts, option0, option1, codes[0], codes[1])
                                        prompt_ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
                                        case_id = (f"c8241-{family}-{language}-{lexical_set}-{task}-{surface}-{codebook}-"
                                                   f"u{unit:02d}-s{state}-{'orig' if condition == 'original' else 'swap'}")
                                        rows.append({
                                            "case_id": case_id, "design_index": len(rows), "family": family,
                                            "family_index": family_index, "macrotype": MACROTYPE[family],
                                            "language": language, "lexical_set": lexical_set, "task": task,
                                            "surface": surface, "codebook": codebook, "condition": condition,
                                            "unit": unit, "state": state, "partition": PARTITION_BY_UNIT[unit],
                                            "semantic_graph": semantic["graph"], "semantic_graph_hash": graph_hash(semantic["graph"]),
                                            "semantic_target": semantic["semantic_target"], "semantic_wrong": semantic["semantic_wrong"],
                                            "selected_truth_value": selected_truth, "facts": facts,
                                            "true_claim": semantic["true"], "false_claim": semantic["false"],
                                            "option_first_text": option0, "option_second_text": option1,
                                            "target_code": target_code, "wrong_code": wrong_code,
                                            "target_code_id": code_ids[target_code], "wrong_code_id": code_ids[wrong_code],
                                            "code_first": codes[0], "code_second": codes[1],
                                            "future_prompt": text, "future_prompt_ids": prompt_ids,
                                            "future_target_text": " " + target_code, "future_wrong_text": " " + wrong_code,
                                            "future_target_ids": [code_ids[target_code]], "future_wrong_ids": [code_ids[wrong_code]],
                                            "boundary_position": len(prompt_ids) - 1,
                                            "all_token_reference": unit == 12 and lexical_set == "lex0" and surface == "natural"
                                                                   and state == 0 and condition == "original" and codebook == "AB",
                                            "natural_answer_probe": {
                                                "question_en": "State the supported conclusion." if language == "en" else "请写出得到支持的结论。",
                                                "target": semantic["true"], "wrong": semantic["false"],
                                            },
                                        })
    graph_groups: dict[tuple, set[str]] = defaultdict(set)
    for row in rows:
        graph_groups[(row["condition"], row["family"], row["lexical_set"], row["task"], row["surface"],
                      row["codebook"], row["unit"], row["state"])].add(row["semantic_graph_hash"])
    balance = Counter((row["condition"], row["family"], row["language"], row["lexical_set"], row["task"],
                       row["surface"], row["codebook"], row["partition"], row["target_code"])
                      for row in rows)
    balance_groups: dict[tuple, dict[str, int]] = defaultdict(dict)
    for key, count in balance.items():
        balance_groups[key[:-1]][key[-1]] = count
    lengths = [len(row["future_prompt_ids"]) for row in rows]
    audit = {
        "rows": len(rows), "families": len(FAMILIES), "macrotypes": len(set(MACROTYPE.values())),
        "units_per_family": UNITS, "languages": list(LANGUAGES), "lexical_sets": list(LEXICAL_SETS),
        "tasks": list(TASKS), "surfaces": list(SURFACES), "codebooks": CODEBOOKS, "conditions": list(CONDITIONS),
        "code_ids": code_ids, "unique_case_ids": len({row["case_id"] for row in rows}) == len(rows),
        "strict_parallel_graph_hash_across_language": all(len(values) == 1 for values in graph_groups.values()),
        "balanced_every_factorial_partition_cell": all(len(set(counts.values())) == 1 for counts in balance_groups.values()),
        "complete_factorial_rows": len(rows) == len(FAMILIES) * 2 * 2 * 2 * 3 * 2 * UNITS * 2 * 2,
        "prompt_token_length": {"minimum": min(lengths), "median": sorted(lengths)[len(lengths) // 2], "maximum": max(lengths)},
        "all_token_reference_rows": sum(row["all_token_reference"] for row in rows),
        "human_naturality_blind_review_completed": False,
        "human_review_limitation": "No human panel was available; a blinded pack is emitted but no human score is fabricated.",
    }
    rng = random.Random(8241)
    candidates = [row for row in rows if row["condition"] == "original" and row["codebook"] == "AB"]
    blind = []
    for row in rng.sample(candidates, 96):
        blind.append({"blind_id": f"blind-{len(blind):03d}", "language": row["language"],
                      "surface": row["surface"], "text": row["future_prompt"],
                      "rating_fluency_1_5": None, "rating_semantic_fidelity_1_5": None, "notes": None})
    return rows, audit, blind


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 十二族双语语义图与词汇—任务—表述—输出码正交合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 从Phase2340的17个行为合格族中预先选取12族、覆盖7宏类；每族16个词汇单元，严格交叉中/英、两套词汇、支持项/矛盾项两种任务、direct/reported/natural三表述、AB/CD两输出码、两个语义状态和逐题选项交换。共 `{result['material_audit']['rows']}` 条固定单token决策材料。中英文共享语言无关的有向语义图与实体占位符，逐单元图哈希必须完全相同；单位0–3/4–7/8–11/12–15分别作为discovery/confirmation/fresh_confirmation/fresh_lockbox。未用Top-K或投影。

$$
\mathcal D=F\times L\times V\times T\times S\times O\times U\times Z\times P,
\qquad |\mathcal D|=12\cdot2\cdot2\cdot2\cdot3\cdot2\cdot16\cdot2\cdot2.
$$

$$
G_{{f,u,v,z}}^{{en}}=G_{{f,u,v,z}}^{{zh}},\qquad
y^{{swap}}=1-y^{{original}}.
$$

**结果汇总与相关文件。** 材料审计 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；冻结门 `{json.dumps(result['freeze'], ensure_ascii=False)}`。主材料 `tests/glm5/result/phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract/material/bilingual_factorial_fixed_code.jsonl`；盲评包 `tests/glm5/result/phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract/material/naturality_blind_pack.jsonl`；脚本 `tests/glm5/phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract.py`。

**理论进展、问题硬伤与结论。** 这是可识别性合同，不是模型结果。它把“同词汇跨任务、同任务跨词汇、同图跨语言、同图跨表述、同任务跨输出码”放到一张析因图中，能够比Phase2341更直接地区分语义图、语言编译、任务政策、表述模板和读出码。实体占位符严格平行会降低自然语言生态真实性；reported仍是包装式改写；natural虽逐族改写但不是独立人类作者。已输出96条匿名盲评包，但当前没有人类评审，明确记为未完成，绝不伪造评分。

**下一阶段路线判断。** 目标相同，自动用Qwen3-4B FP16/CUDA采集全部材料的边界token×全38检查点×全2560坐标；另对冻结的48条关键材料采集全token×全层×全坐标。行为先于机制解释，只有在中英、两输出码和所有锁箱分区上达到冻结门的族进入内部图谱竞争。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    rows, audit, blind = compile_material(tokenizer)
    write_rows(MATERIAL, rows)
    write_rows(BLIND_PACK, blind)
    freeze = {
        "frozen_before_model_load": True, "families": list(FAMILIES), "macrotypes": MACROTYPE,
        "partitions": list(PARTITIONS), "train_partitions": ["discovery", "confirmation"],
        "selection_partition": "fresh_confirmation", "adjudication_partition": "fresh_lockbox",
        "behavior_family_language_codebook_partition_minimum_accuracy": 0.70,
        "atlas_bidirectional_transfer_minimum_accuracy": 0.30,
        "coordinate_advantage_over_row_sorted": 0.10,
        "output_weight_advantage_over_absolute_hidden": 0.05,
        "q23_role": "frozen observation window; all checkpoints remain mandatory",
        "coordinate_policy": "all 2560 coordinates; no Top-K/PCA/projection",
        "causal_policy": "only routes passing vocabulary, task, language, surface, codebook and lockbox gates may be intervened on",
    }
    save(OUT / "config/frozen_contract.json", freeze)
    checks = {
        "row_count": audit["complete_factorial_rows"], "parallel_graphs": audit["strict_parallel_graph_hash_across_language"],
        "balanced": audit["balanced_every_factorial_partition_cell"], "unique": audit["unique_case_ids"],
        "families_at_least_12": audit["families"] >= 12, "units_at_least_16": audit["units_per_family"] >= 16,
        "reference_rows": audit["all_token_reference_rows"] == 48,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "material_audit": audit,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2344_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
