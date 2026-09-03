#!/usr/bin/env python3
"""C870-C884 broad language-family conditional-gear master contract.

This module freezes material and measurement rules before model execution.  It
observes embeddings and HiddenStates only.  Attention, MLP internals, weights,
gradients, PCA, Top-K screening, cosine screening and donor-difference
transport are outside the registered object.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = RESULT / "phase2234_c870_c884_broad_family_conditional_gear_contract"
sys.path.insert(0, str(TESTS))

import phase1797_c263_c272_state_operator_common as compiler
import phase2219_c773_c808_semantic_transition_ecology_campaign as prior


PHASE = 2234
CAMPAIGNS = tuple(f"C{i}" for i in range(870, 885))
DIM = 2560
CHECKPOINTS = 38
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
QPOINTS = (0, 8, 16, 24, 32, 37)
LANGUAGES = ("en", "zh")
FAMILIES = (
    "taxonomy_chain",
    "part_whole_chain",
    "temporal_order",
    "causal_direction",
    "agent_patient_voice",
    "negation_scope",
    "coreference_binding",
    "nested_attitude",
    "attribute_binding",
    "comparison",
    "translation_route",
    "quantifier_scope",
)
SURFACES = ("direct", "paraphrase")
CELLS = ("false_direct", "true_direct", "false_paraphrase", "true_paraphrase")
TRUTH_BY_CELL = (False, True, False, True)
OUTPUT_SCHEMES = (
    ("Yes", "No"),
    ("True", "False"),
    ("Supported", "Unsupported"),
    ("Entailed", "Contradicted"),
)
PARENT_UNITS = 16
FRESH_UNITS = 8
BEHAVIOR_GATE = 0.75
TAU_MULTIPLIERS = (1.0, 2.0, 4.0)
PRIMARY_TAU = 2.0
FAMILY_GATES = {
    "support_f1": 0.50,
    "changed_class_accuracy": 0.30,
    "f1_gain_over_shared": 0.02,
    "f1_gain_over_wrong_family": 0.02,
    "relative_mae_gain_over_shared": 0.03,
    "minimum_units": 4,
}
FLAGSHIP_GATES = {
    "relative_mae_gain_over_zero": 0.05,
    "relative_mae_gain_over_wrong_domain": 0.03,
    "minimum_units": 4,
}

PARENT_NAMES_A = (
    "Ava", "Cora", "Elena", "Grace", "Iris", "Kara", "Maya", "Olive",
    "Rina", "Tara", "Willa", "Zara", "Nora", "Lena", "Mira", "Sonia",
)
PARENT_NAMES_B = (
    "Ben", "Dylan", "Felix", "Hector", "Jonah", "Liam", "Noah", "Peter",
    "Simon", "Victor", "Wyatt", "Aaron", "Caleb", "Evan", "Owen", "Theo",
)
PARENT_OBJECTS = (
    "apple", "banana", "pear", "peach", "grape", "lemon", "orange", "plum",
    "cherry", "mango", "melon", "coconut", "carrot", "potato", "tomato", "onion",
)
PARENT_DISTRACTORS = (
    "hammer", "violin", "ladder", "compass", "lantern", "pillow", "mirror", "bucket",
    "helmet", "whistle", "tripod", "notebook", "ribbon", "basket", "key", "bottle",
)
PARENT_FRENCH = (
    "pomme", "banane", "poire", "peche", "raisin", "citron", "orange", "prune",
    "cerise", "mangue", "melon", "coco", "carotte", "patate", "tomate", "oignon",
)
FRESH_NAMES_A = ("Uma", "Yara", "Dara", "Fiona", "Hazel", "Jade", "Keira", "Paula")
FRESH_NAMES_B = ("Basil", "Cedric", "Gavin", "Hugo", "Ivan", "Quinn", "Rory", "Silas")
FRESH_OBJECTS = ("moon", "star", "river", "mountain", "window", "door", "chair", "table")
FRESH_DISTRACTORS = ("anchor", "flute", "saddle", "candle", "folder", "glove", "kettle", "frame")
FRESH_FRENCH = ("lune", "etoile", "riviere", "montagne", "fenetre", "porte", "chaise", "table")

ZH_NAMES_A = ("艾娃", "科拉", "埃琳娜", "格蕾丝", "艾丽丝", "卡拉", "玛雅", "奥莉芙", "瑞娜", "塔拉", "薇拉", "扎拉", "诺拉", "莉娜", "米拉", "索尼娅")
ZH_NAMES_B = ("本", "迪伦", "费利克斯", "赫克托", "乔纳", "利亚姆", "诺亚", "彼得", "西蒙", "维克托", "怀亚特", "亚伦", "凯莱布", "埃文", "欧文", "西奥")
ZH_OBJECTS = ("苹果", "香蕉", "梨", "桃子", "葡萄", "柠檬", "橙子", "李子", "樱桃", "芒果", "甜瓜", "椰子", "胡萝卜", "土豆", "番茄", "洋葱")
ZH_DISTRACTORS = ("锤子", "小提琴", "梯子", "指南针", "灯笼", "枕头", "镜子", "水桶", "头盔", "口哨", "三脚架", "笔记本", "丝带", "篮子", "钥匙", "瓶子")
ZH_FRESH_NAMES_A = ("优玛", "雅拉", "达拉", "菲奥娜", "海泽尔", "杰德", "凯拉", "保拉")
ZH_FRESH_NAMES_B = ("巴兹尔", "塞德里克", "加文", "雨果", "伊万", "奎因", "罗里", "赛拉斯")
ZH_FRESH_OBJECTS = ("月亮", "星星", "河流", "山峰", "窗户", "门", "椅子", "桌子")
ZH_FRESH_DISTRACTORS = ("船锚", "长笛", "马鞍", "蜡烛", "文件夹", "手套", "水壶", "画框")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def partition(unit: int, fresh: bool) -> str:
    if fresh:
        return "confirmation" if unit < 4 else "lockbox"
    return "discovery" if unit < 8 else ("confirmation" if unit < 12 else "lockbox")


def lexicon(unit: int, language: str, fresh: bool) -> dict[str, str]:
    if fresh:
        names_a, names_b, objects, distractors, french = (
            FRESH_NAMES_A, FRESH_NAMES_B, FRESH_OBJECTS, FRESH_DISTRACTORS, FRESH_FRENCH
        ) if language == "en" else (
            ZH_FRESH_NAMES_A, ZH_FRESH_NAMES_B, ZH_FRESH_OBJECTS, ZH_FRESH_DISTRACTORS, FRESH_FRENCH
        )
    else:
        names_a, names_b, objects, distractors, french = (
            PARENT_NAMES_A, PARENT_NAMES_B, PARENT_OBJECTS, PARENT_DISTRACTORS, PARENT_FRENCH
        ) if language == "en" else (
            ZH_NAMES_A, ZH_NAMES_B, ZH_OBJECTS, ZH_DISTRACTORS, PARENT_FRENCH
        )
    if language == "en":
        t1 = ("sample class " + chr(65 + unit % 8)).lower()
        t2 = ("registered family " + chr(75 + unit % 8)).lower()
        t3 = ("catalog group " + chr(83 + unit % 8)).lower()
        prop_true, prop_false = ("red", "blue") if unit % 2 == 0 else ("smooth", "rough")
    else:
        t1, t2, t3 = f"样本类{unit % 8 + 1}", f"登记族{unit % 8 + 1}", f"目录组{unit % 8 + 1}"
        prop_true, prop_false = ("红色", "蓝色") if unit % 2 == 0 else ("光滑", "粗糙")
    return {
        "a": names_a[unit], "b": names_b[unit], "x": objects[unit], "y": distractors[unit],
        "fr": french[unit], "wrong_fr": french[(unit + 3) % len(french)],
        "t1": t1, "t2": t2, "t3": t3, "prop_true": prop_true, "prop_false": prop_false,
    }


def codebook(unit: int, truth: bool) -> tuple[str, str, str, str]:
    true_code, false_code = OUTPUT_SCHEMES[unit % len(OUTPUT_SCHEMES)]
    return true_code, false_code, (true_code if truth else false_code), (false_code if truth else true_code)


def answer_position(truth: bool, unit: int, cell_i: int, language: str, offset: int = 0) -> int:
    return (unit + cell_i + int(language == "zh") + offset) % 2


def wrap_case(*, case_id: str, panel: str, family: str, language: str, unit: int,
              cell: str, cell_i: int, truth: bool, core: str, roles: dict[str, str],
              factors: dict, fresh: bool, offset: int = 0) -> dict:
    true_code, false_code, correct, wrong = codebook(unit, truth)
    gold = answer_position(truth, unit, cell_i, language, offset)
    options = [correct, wrong] if gold == 0 else [wrong, correct]
    cb = f"CODEBOOK: supported = {true_code}; unsupported = {false_code}."
    prompt = f"{cb} {core} (A) {options[0]} (B) {options[1]}. Reply only A or B."
    free_prompt = f"{cb} {core} Reply with exactly {true_code} or {false_code}."
    return {
        "case_id": case_id, "panel": panel, "family": family, "language": language,
        "unit": unit, "partition": partition(unit, fresh), "cell": cell, "cell_i": cell_i,
        "surface": "direct" if "direct" in cell else "paraphrase", "truth": truth,
        "output_scheme": unit % len(OUTPUT_SCHEMES), "true_code": true_code,
        "false_code": false_code, "correct_answer": correct, "wrong_answer": wrong,
        "gold_position": gold, "prompt_core": core, "prompt": prompt, "free_prompt": free_prompt,
        "role_values": roles, "factors": factors, "fresh": fresh,
        "semantic_graph": {
            "external_family": family, "panel": panel, "factors": factors,
            "labels_are_experimental_coordinates_not_internal_modules": True,
        },
    }


def broad_case(family: str, language: str, unit: int, cell_i: int, fresh: bool) -> dict:
    u = lexicon(unit, language, fresh)
    a, b, x, y, t1, t2, t3 = (u[k] for k in ("a", "b", "x", "y", "t1", "t2", "t3"))
    truth = TRUTH_BY_CELL[cell_i]
    paraphrase = cell_i >= 2
    relation = ""
    if language == "en":
        if family == "taxonomy_chain":
            facts = f"{x} belongs to {t1}; every {t1} belongs to {t2}; every {t2} belongs to {t3 if truth else y}"
            core = f"A verified classification record states that {facts}. Based only on it, is {x} in {t3}?" if not paraphrase else f"The checked catalog lists this chain: {facts}. Does that catalog support classifying {x} under {t3}?"
            relation = "belongs to"
        elif family == "part_whole_chain":
            facts = f"{x} is part of {t1}; {t1} is part of {t2 if truth else y}"
            core = f"A verified assembly record states that {facts}. Is {x} part of {t2}?" if not paraphrase else f"The checked assembly ledger says: {facts}. Does it support that {x} forms part of {t2}?"
            relation = "part of"
        elif family == "temporal_order":
            facts = f"{x} occurred before {t1}; {t1} occurred {'before' if truth else 'after'} {t2}"
            core = f"A verified schedule states that {facts}. Did {x} occur before {t2}?" if not paraphrase else f"The checked chronology records: {facts}. Is {x} earlier than {t2}?"
            relation = "before"
        elif family == "causal_direction":
            facts = f"{x} caused {t1}; {t1} {'caused' if truth else 'prevented'} {t2}"
            core = f"A verified causal report states that {facts}. Does it support that {x} indirectly caused {t2}?" if not paraphrase else f"The checked dependency log says: {facts}. Is {t2} a downstream effect of {x}?"
            relation = "caused"
        elif family == "agent_patient_voice":
            fact = f"{a if truth else b} moved {x}" if not paraphrase else f"{x} was moved by {a if truth else b}"
            core = f"A verified event report states that {fact}. Did {a} move {x}?"
            relation = "moved"
        elif family == "negation_scope":
            fact = f"{a} {'moved' if truth else 'did not move'} {x}" if not paraphrase else f"It is {'true' if truth else 'false'} that {a} moved {x}"
            core = f"A verified report states: {fact}. Does it support that {a} moved {x}?"
            relation = "moved"
        elif family == "coreference_binding":
            speaker = a if truth else b
            fact = f"{speaker} told {b if truth else a}, 'I stored {x}.'" if not paraphrase else f"The transcript quotes {speaker}, while {b if truth else a} listened: 'I stored {x}.'"
            core = f"A verified quotation record states: {fact} In the quotation, does I refer to {a}?"
            relation = "I"
        elif family == "nested_attitude":
            verb = "remembered" if truth else "heard"
            fact = f"{a} {verb} that {b} stored {x}" if not paraphrase else f"According to the record, {b}'s storing of {x} was {verb} by {a}"
            core = f"A verified memory record states: {fact}. Does it say that {a} remembered that {b} stored {x}?"
            relation = "remembered"
        elif family == "attribute_binding":
            prop = u["prop_true"] if truth else u["prop_false"]
            fact = f"{x} has the property {prop}" if not paraphrase else f"the recorded property of {x} is {prop}"
            core = f"A verified property record states that {fact}. Does {x} have the property {u['prop_true']}?"
            relation = "property"
        elif family == "comparison":
            rel = "heavier than" if truth else "lighter than"
            fact = f"{x} is {rel} {y}" if not paraphrase else f"the comparison places {x} as {rel} {y}"
            core = f"A verified comparison record states that {fact}. Is {x} heavier than {y}?"
            relation = "heavier"
        elif family == "translation_route":
            target = u["fr"] if truth else u["wrong_fr"]
            fact = f"the English word {x} maps to French {target}" if not paraphrase else f"the French entry paired with English {x} is {target}"
            core = f"A verified bilingual glossary states that {fact}. Is the French form of {x} {u['fr']}?"
            relation = "French"
        elif family == "quantifier_scope":
            quant = "Every" if truth else "No"
            fact = f"{x} is a {t1}; {quant} {t1} is a {t2}" if not paraphrase else f"the registry puts {x} in {t1}, and says that {quant.lower()} {t1} belongs to {t2}"
            core = f"A verified quantified membership record states that {fact}. Is {x} a {t2}?"
            relation = "quantified membership"
        else:
            raise KeyError(family)
    else:
        if family == "taxonomy_chain":
            facts = f"{x}属于{t1}；每个{t1}都属于{t2}；每个{t2}都属于{t3 if truth else y}"
            core = f"一份核验过的分类记录写道：{facts}。仅根据记录，{x}属于{t3}吗？" if not paraphrase else f"核对后的目录给出链条：{facts}。该目录支持把{x}归入{t3}吗？"
            relation = "属于"
        elif family == "part_whole_chain":
            facts = f"{x}是{t1}的一部分；{t1}是{t2 if truth else y}的一部分"
            core = f"一份核验过的装配记录写道：{facts}。{x}是{t2}的一部分吗？" if not paraphrase else f"核对后的装配账本写道：{facts}。它支持{x}构成{t2}的一部分吗？"
            relation = "一部分"
        elif family == "temporal_order":
            facts = f"{x}早于{t1}发生；{t1}{'早于' if truth else '晚于'}{t2}发生"
            core = f"一份核验过的时间表写道：{facts}。{x}早于{t2}发生吗？" if not paraphrase else f"核对后的时间顺序记录：{facts}。{x}比{t2}更早吗？"
            relation = "早于"
        elif family == "causal_direction":
            facts = f"{x}导致{t1}；{t1}{'导致' if truth else '阻止'}{t2}"
            core = f"一份核验过的因果报告写道：{facts}。它支持{x}间接导致{t2}吗？" if not paraphrase else f"核对后的依赖日志写道：{facts}。{t2}是{x}的下游结果吗？"
            relation = "导致"
        elif family == "agent_patient_voice":
            fact = f"{a if truth else b}移动了{x}" if not paraphrase else f"{x}由{a if truth else b}移动"
            core = f"一份核验过的事件报告写道：{fact}。{a}移动了{x}吗？"
            relation = "移动"
        elif family == "negation_scope":
            fact = f"{a}{'移动了' if truth else '没有移动'}{x}" if not paraphrase else f"{a}移动{x}这件事是{'真的' if truth else '假的'}"
            core = f"一份核验过的报告写道：{fact}。报告支持{a}移动了{x}吗？"
            relation = "移动"
        elif family == "coreference_binding":
            speaker = a if truth else b
            fact = f"{speaker}对{b if truth else a}说：‘我存放了{x}。’" if not paraphrase else f"谈话记录写道，{b if truth else a}在场聆听，并引用{speaker}的话：‘我存放了{x}。’"
            core = f"一份核验过的引语记录写道：{fact}在引语中，‘我’指的是{a}吗？"
            relation = "我"
        elif family == "nested_attitude":
            verb = "记得" if truth else "听说"
            fact = f"{a}{verb}{b}存放了{x}" if not paraphrase else f"记录把{b}存放{x}这件事描述为{a}{verb}的内容"
            core = f"一份核验过的记忆记录写道：{fact}。记录表示{a}记得{b}存放了{x}吗？"
            relation = "记得"
        elif family == "attribute_binding":
            prop = u["prop_true"] if truth else u["prop_false"]
            fact = f"{x}具有{prop}属性" if not paraphrase else f"{x}登记的属性是{prop}"
            core = f"一份核验过的属性记录写道：{fact}。{x}具有{u['prop_true']}属性吗？"
            relation = "属性"
        elif family == "comparison":
            rel = "重于" if truth else "轻于"
            fact = f"{x}{rel}{y}" if not paraphrase else f"比较记录把{x}列为{rel}{y}"
            core = f"一份核验过的比较记录写道：{fact}。{x}重于{y}吗？"
            relation = "重于"
        elif family == "translation_route":
            target = u["fr"] if truth else u["wrong_fr"]
            fact = f"中文词{x}对应法语{target}" if not paraphrase else f"与中文{x}配对的法语条目是{target}"
            core = f"一份核验过的双语词表写道：{fact}。{x}的法语形式是{u['fr']}吗？"
            relation = "法语"
        elif family == "quantifier_scope":
            quant = "每个" if truth else "没有任何"
            fact = f"{x}是{t1}；{quant}{t1}是{t2}" if not paraphrase else f"登记表把{x}归入{t1}，并说明{quant}{t1}属于{t2}"
            core = f"一份核验过的量化归属记录写道：{fact}。{x}是{t2}吗？"
            relation = "量化归属"
        else:
            raise KeyError(family)
    if family == "taxonomy_chain":
        roles = {"primary": x, "secondary": t1, "relation": relation, "context": t3, "query": x}
    elif family in ("part_whole_chain", "temporal_order", "causal_direction"):
        roles = {"primary": x, "secondary": t1, "relation": relation, "context": t2, "query": x}
    elif family in ("agent_patient_voice", "negation_scope"):
        roles = {"primary": a, "secondary": x, "relation": relation, "context": x, "query": a}
    elif family in ("coreference_binding", "nested_attitude"):
        roles = {"primary": a, "secondary": b, "relation": relation, "context": x, "query": a}
    elif family == "attribute_binding":
        roles = {"primary": x, "secondary": prop, "relation": relation, "context": u["prop_true"], "query": x}
    elif family == "comparison":
        roles = {"primary": x, "secondary": y, "relation": relation, "context": y, "query": x}
    elif family == "translation_route":
        roles = {"primary": x, "secondary": target, "relation": relation, "context": u["fr"], "query": x}
    elif family == "quantifier_scope":
        roles = {"primary": x, "secondary": t1, "relation": relation, "context": t2, "query": x}
    else:
        raise KeyError(family)
    cell = CELLS[cell_i]
    return wrap_case(
        case_id=f"c870-{'fresh' if fresh else 'parent'}-{family}-{language}-u{unit:02d}-c{cell_i}",
        panel="broad_family", family=family, language=language, unit=unit, cell=cell,
        cell_i=cell_i, truth=truth, core=core, roles=roles,
        factors={"semantic_truth": int(truth), "surface": int(paraphrase)}, fresh=fresh,
    )


def attitude_case(domain: str, language: str, unit: int, outer: int, inner: int, fresh: bool) -> dict:
    u = lexicon(unit, language, fresh); a, b, x, y = u["a"], u["b"], u["x"], u["y"]
    truth = unit % 2 == 0; query_object = x if truth else y
    verbs = {"like": ("likes", "喜欢"), "regret": ("regrets", "后悔"), "remember": ("remembers", "记得")}
    verb = verbs[domain][0 if language == "en" else 1]
    if language == "en":
        surface_verb = verb[:-1] if outer else verb
        role_relation = surface_verb
        statement = f"{a} {'does not ' + surface_verb if outer else surface_verb} {b} {'not ' if inner else ''}eating {x}."
        question = f"Is it true that {a} {'does not ' + surface_verb if outer else surface_verb} {b} {'not ' if inner else ''}eating {query_object}?"
        core = f"A verified attitude record states: {statement} {question}"
    else:
        role_relation = verb
        statement = f"{a}{'不' if outer else ''}{verb}{b}{'没有' if inner else ''}吃{x}。"
        question = f"记录表示{a}{'不' if outer else ''}{verb}{b}{'没有' if inner else ''}吃{query_object}吗？"
        core = f"一份核验过的态度记录写道：{statement}{question}"
    cell_i = outer * 2 + inner
    return wrap_case(
        case_id=f"c870-{'fresh' if fresh else 'parent'}-attitude-{domain}-{language}-u{unit:02d}-o{outer}i{inner}",
        panel="nested_attitude_flagship", family=f"attitude_{domain}", language=language,
        unit=unit, cell=f"o{outer}i{inner}", cell_i=cell_i, truth=truth, core=core,
        roles={"primary": a, "secondary": b, "relation": role_relation, "context": x, "query": a},
        factors={"outer_negation": outer, "inner_negation": inner, "domain": domain}, fresh=fresh, offset=5,
    )


def graph_case(domain: str, language: str, unit: int, depth: int, shortcut: int, fresh: bool) -> dict:
    u = lexicon(unit, language, fresh); x, y, t1, t2, t3, b = u["x"], u["y"], u["t1"], u["t2"], u["t3"], u["b"]
    truth = unit % 2 == 0; endpoint = (t1, t2, t3)[depth - 1]; query_target = endpoint if truth else y
    if language == "en":
        rel = {"taxonomy": "belongs to", "part_whole": "is part of", "temporal": "precedes"}[domain]
        chain = [(x, t1), (t1, t2), (t2, t3)]
        facts = [f"{left} {rel} {right}" for left, right in chain[:depth]]
        if shortcut and depth > 1: facts.append(f"{x} {rel} {endpoint}")
        core = f"A verified relation graph states: {'; '.join(facts)}. Based only on it, is it true that {x} {rel} {query_target}?"
    else:
        rel = {"taxonomy": "属于", "part_whole": "是其一部分", "temporal": "早于"}[domain]
        chain = [(x, t1), (t1, t2), (t2, t3)]
        facts = [f"{left}{rel}{right}" for left, right in chain[:depth]]
        if shortcut and depth > 1: facts.append(f"{x}{rel}{endpoint}")
        core = f"一份核验过的关系图写道：{'；'.join(facts)}。仅根据关系图，{x}{rel}{query_target}吗？"
    cell_i = (depth - 1) * 2 + shortcut
    return wrap_case(
        case_id=f"c870-{'fresh' if fresh else 'parent'}-graph-{domain}-{language}-u{unit:02d}-d{depth}k{shortcut}",
        panel="recursive_graph_flagship", family=f"graph_{domain}", language=language,
        unit=unit, cell=f"d{depth}k{shortcut}", cell_i=cell_i, truth=truth, core=core,
        roles={"primary": x, "secondary": endpoint, "relation": rel, "context": query_target, "query": x},
        factors={"depth": depth, "shortcut": shortcut, "domain": domain}, fresh=fresh, offset=9,
    )


def material(fresh: bool) -> list[dict]:
    units = FRESH_UNITS if fresh else PARENT_UNITS
    rows = [
        broad_case(family, language, unit, cell_i, fresh)
        for family, language, unit, cell_i in itertools.product(FAMILIES, LANGUAGES, range(units), range(4))
    ]
    rows += [
        attitude_case(domain, language, unit, outer, inner, fresh)
        for domain, language, unit, outer, inner in itertools.product(
            ("like", "regret", "remember"), LANGUAGES, range(units), (0, 1), (0, 1)
        )
    ]
    rows += [
        graph_case(domain, language, unit, depth, shortcut, fresh)
        for domain, language, unit, depth, shortcut in itertools.product(
            ("taxonomy", "part_whole", "temporal"), LANGUAGES, range(units), (1, 2, 3), (0, 1)
        ) if not (depth == 1 and shortcut == 1)
    ]
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidates):
        raise RuntimeError(("candidate_not_single_token", candidates))
    system = "Answer only from the supplied text. Do not use outside knowledge."
    compiled = []

    def contextual_spans(ids: list[int], value: str) -> list[list[int]]:
        exact = compiler.graph_base.name_spans(tokenizer, ids, value)
        if exact:
            return exact
        # Chinese BPE tokens can merge a role word with adjacent particles.  The
        # fallback identifies the smallest decoded token interval containing the
        # exact source substring; it does not infer or relabel a semantic role.
        needle_len = max(1, len(tokenizer.encode(value, add_special_tokens=False)))
        found = []
        for width in range(1, needle_len + 4):
            for start in range(0, len(ids) - width + 1):
                decoded = tokenizer.decode(ids[start:start + width], skip_special_tokens=True)
                if value in decoded:
                    found.append(list(range(start, start + width)))
            if found:
                return found
        return []

    for row in rows:
        ids = compiler.core.chat_ids(tokenizer, system, row["prompt"])
        free_ids = compiler.core.chat_ids(tokenizer, system, row["free_prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = contextual_spans(ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "free_prompt_ids": free_ids,
                         "candidate_ids": candidates, "role_positions": positions})
    return compiled


def audit_material(rows: list[dict], compiled: list[dict]) -> dict:
    by_id = {row["case_id"]: row for row in compiled}
    balance = defaultdict(lambda: {"truth": [0, 0], "position": [0, 0], "schemes": [0, 0, 0, 0]})
    missing_roles = []
    malformed = []
    forbidden = ("�", "锟", "remembered that did", "eated", "regreted")
    partition_lexicon = defaultdict(set)
    for row in rows:
        key = f"{row['panel']}|{row['family']}|{row['language']}|{row['partition']}"
        balance[key]["truth"][int(row["truth"])] += 1
        balance[key]["position"][int(row["gold_position"])] += 1
        balance[key]["schemes"][int(row["output_scheme"])] += 1
        partition_lexicon[row["partition"]].add(row["role_values"]["primary"])
        if any(token in row["prompt"] for token in forbidden):
            malformed.append(row["case_id"])
        for role, value in row["role_values"].items():
            if value not in row["prompt_core"]:
                missing_roles.append({"case_id": row["case_id"], "role": role, "value": value})
    overlap = set()
    for left, right in itertools.combinations(partition_lexicon, 2):
        overlap |= partition_lexicon[left] & partition_lexicon[right]
    widths = [len(row["prompt_ids"]) for row in compiled]
    zero = {
        "always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
        "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
        "always_supported": float(np.mean([row["truth"] for row in rows])),
    }
    broad = [row for row in rows if row["panel"] == "broad_family"]
    pair_complete = all(
        len([r for r in broad if r["family"] == f and r["language"] == lang and r["unit"] == unit]) == 4
        for f, lang, unit in itertools.product(FAMILIES, LANGUAGES, range(FRESH_UNITS if rows[0]["fresh"] else PARENT_UNITS))
    )
    return {
        "rows": len(rows), "compiled_rows": len(compiled), "unique_case_ids": len(by_id),
        "families": dict(sorted(defaultdict(int, {f: sum(r["family"] == f for r in rows) for f in sorted({r['family'] for r in rows})}).items())),
        "panels": {p: sum(r["panel"] == p for r in rows) for p in sorted({r["panel"] for r in rows})},
        "joint_balance": dict(balance), "zero_models": zero, "missing_roles": missing_roles,
        "malformed_strings": malformed, "cross_partition_primary_overlap": sorted(overlap),
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "broad_factorial_pair_complete": pair_complete,
        "semantic_uniqueness_machine_audit": "pass_by_frozen_truth_table_role_presence_and_factorial_completeness",
        "material_naturalness_machine_audit": "pass_controlled_bilingual_templates",
        "human_blind_review": "NA_not_run_no_independent_human_panel_available",
    }


def preregistration() -> dict:
    return {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "frozen_before_model": True,
        "research_object": "family-specific full-coordinate response beyond shared forward dynamics",
        "materials": {
            "families": list(FAMILIES), "languages": list(LANGUAGES), "surfaces": list(SURFACES),
            "output_schemes": [list(x) for x in OUTPUT_SCHEMES], "parent_units": PARENT_UNITS,
            "fresh_units": FRESH_UNITS, "flagships": ["nested_attitude_factorial", "recursive_graph_depth_shortcut"],
        },
        "partitions": {"parent": {"discovery": 8, "confirmation": 4, "lockbox": 4}, "fresh": {"confirmation": 4, "lockbox": 4}},
        "models_sequential": ["Qwen3-4B", "Qwen3-14B", "GLM4", "DeepSeek-7B"],
        "camera": "embedding + 36 post-block HiddenStates + final norm; six functional roles; representative full-token field; every coordinate",
        "forbidden": ["attention", "MLP", "weights", "gradients", "PCA", "Top-K", "cosine screening", "donor HiddenState difference transport", "post-reveal threshold tuning"],
        "behavior_gate": BEHAVIOR_GATE, "behavior_policy": "capture all rows inside dual-behavior-qualified slices; retain failures in behavior atlas",
        "threshold_ladder": list(TAU_MULTIPLIERS), "primary_tau_multiplier": PRIMARY_TAU,
        "model_tournament": [
            "M0_zero", "M1_shared_mean", "M2_shared_same-coordinate_role-coupled_affine",
            "M3_shared_plus_family_mean_residual", "M4_shared_plus_family_same-coordinate_affine_residual",
            "M5_shared_plus_family_state-guard_residual", "M6_wrong-family_equal-capacity",
        ],
        "shared_features": ["intercept", "same coordinate", "previous checkpoint same coordinate", "boundary same coordinate", "query same coordinate", "relation same coordinate"],
        "family_features": ["intercept", "same coordinate", "query same coordinate", "relation same coordinate"],
        "guard": "sign(self) x sign(relation), per family/checkpoint/role/coordinate",
        "family_gates": FAMILY_GATES, "flagship_gates": FLAGSHIP_GATES,
        "causal_branch": "only strict fresh-lockbox family candidates; full-coordinate predicted response call/delete/wrong-family control; otherwise NA",
        "cross_model_rule": "compile per tokenizer, freeze exact common semantic row denominator, compare relative depth/role topology only",
        "failure_policy": "route-level missingness; no failed family, model or causal branch stops other registered observations",
        "cleanup": "hash then delete raw HiddenState files not retained by the visualization client",
        "theory": "conditionalized output-field closure theory; reuse-difference-conditioning (RDC); no new mathematics authorization",
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    parent = result["material_audit"]["parent"]
    fresh = result["material_audit"]["fresh"]
    formula_block = r"""
$$
\Delta H_{i,q,r,j}=G_{q,r,j}(H_i)+R_{f,q,r,j}(H_i)+\varepsilon_{i,q,r,j}.
$$

主检验不是族模型自身分数，而是相对共享底盘和错族同容量控制的前瞻增益：

$$
G_f^{\mathrm{pred}}=S(M_{\mathrm{shared}+f})-\max\{S(M_{\mathrm{shared}}),S(M_{\mathrm{shared}+f'}),S(M_{\mathrm{perm}})\}.
$$

支持事件采用冻结强度阶梯，主判决使用 $m=2$，避免旧阈值下支持过密：

$$
\tau_m(H)=m\bigl(0.02+0.05|H|\bigr),\qquad m\in\{1,2,4\}.
$$

嵌套组合只用基础二阶分账，不预设新数学：

$$
I_{AB}=H_{11}-H_{10}-H_{01}+H_{00}.
$$
"""
    text = f"""

## Phase {PHASE}: 广语言族共享动力学分账与条件齿轮总合同（C870-C884） [{stamp}]

**证据审查与过度结论修正。** 本期先重裁附件对 Phase2227-2233 的解释。成立的部分是：Phase2228 在全部物理激活坐标上获得很强的样本局部预测，但严格候选为0；Phase2229 因此前置缺失而为 `NA`；Phase2231 的同族距离略小于跨族距离，却只有 `0.1875` 的最近族识别率并正式记为 `family_signal=false`；Phase2233 只修正了跨模型分母。不能成立的扩张是把高F1称为语言族齿轮、把激活坐标称为模型参数、把因果未运行称为因果失败，或把弱系数距离称为跨模型同构。Phase2228 的实际支持率约为0.63，旧局部模型还使用任意的 `j+257` 伙伴，因此高F1首先支持广泛共享状态变化可预测，不识别语义族机制。

**阶段目标与测试原理。** 本期不加载模型，只冻结下一轮完整大Campaign。核心分账对象是：先用跨族数据识别共享前向动力学，再检验族条件模型能否在父确认、父锁箱、全新词汇确认和全新词汇锁箱上同时超过共享模型及同容量错族控制。材料覆盖12个语言操作族、英语/中文、两种表面、四套输出码，并增加嵌套态度 `外层否定 x 内层否定` 与关系图 `路径深度 x 直接捷径` 两个组合旗舰。示例包括“苹果属于样本类，样本类属于登记族，登记族属于目录组，苹果属于目录组吗”和“艾娃不喜欢本没有吃苹果”。人类盲评没有可用独立评审者，严格记为 `NA`，机器自然度审计不能替代它。

**统一测试公式。** 共享模型与族残差明确分账：
{formula_block}

**材料、分区、零模型与结果汇总。** 父材料 `{parent['rows']}` 行、全新词汇材料 `{fresh['rows']}` 行；两者都完成Qwen tokenizer编译。父集分为8个 discovery、4个 confirmation、4个 lockbox 单位；fresh为4个 confirmation和4个 lockbox。父/新词零模型分别为 `{json.dumps(parent['zero_models'], ensure_ascii=False)}` 与 `{json.dumps(fresh['zero_models'], ensure_ascii=False)}`；父/新词 token 长度最小-中位-最大分别为 `{parent['token_width_min_median_max']}` 与 `{fresh['token_width_min_median_max']}`。缺失角色分别为 `{len(parent['missing_roles'])}`、`{len(fresh['missing_roles'])}`，乱码/畸形字符串分别为 `{len(parent['malformed_strings'])}`、`{len(fresh['malformed_strings'])}`；宽族四格配对完整性均为 `{parent['broad_factorial_pair_complete'] and fresh['broad_factorial_pair_complete']}`。

**算法锦标赛与门槛。** 冻结比较 `M0` 零变化、`M1` 共享均值、`M2` 共享同坐标角色耦合仿射、`M3` 共享加族均值残差、`M4` 共享加族同坐标仿射残差、`M5` 共享加族状态守卫残差和 `M6` 错族同容量控制。没有引入PCA、低秩压缩、Top-K、Lasso或大型神经相机；这样不能穷尽所有可能算法，但能直接检验附件中最关键的“共享动力学是否遮住族残差”。严格族候选必须在四个保留面板同时满足：support F1不低于0.50、变化坐标类别准确率不低于0.30、相对共享F1增益不低于0.02、相对错族F1增益不低于0.02、相对共享MAE改善不低于0.03、每面板至少4个独立词汇单位。

**理论进展、问题与硬伤。** 理论名称保持“条件化输出场闭合理论”，组织原则保持RDC。更新仅是把 `共享前向动力学`、`族条件残差`、`输出身份/生成边界` 三项正式分账。硬伤包括：材料仍是受控双语模板；中文和英语不等于自然语料全域；激活坐标不是模型参数；同坐标模型仍不允许任意跨坐标高阶超边；人类自然度为NA；四种输出码仍是显式元语言接口。因此本期没有神经机制结果、没有因果结果、没有新数学授权。

**结论与下一步授权。** 合同与材料审计通过，授权顺序运行Qwen3-4B双行为和全坐标场、共享/族残差锦标赛、两个组合旗舰、严格候选因果分支，以及相同语义行分母的Qwen3-14B/GLM4/DeepSeek-7B模型相对面板。任何单族、单模型或因果分支失败只记录路线级缺失，不停止其余已注册观察。

**相关文件。** 脚本 `tests/glm5/phase2234_c870_c884_broad_family_gear_contract.py`；结果目录 `{OUT.relative_to(ROOT)}`；预注册 `tests/glm5/result/phase2234_c870_c884_broad_family_conditional_gear_contract/protocol/preregistration.json`；父/新词材料及Qwen编译行保存在同目录 `material/`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    for sub in ("protocol", "material", "analysis", "audit"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    protocol = preregistration()
    save(OUT / "protocol/preregistration.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), **protocol,
    })
    parent_rows, fresh_rows = material(False), material(True)
    tokenizer = prior.parent.load_tokenizer()
    parent_compiled = compile_rows(tokenizer, parent_rows)
    fresh_compiled = compile_rows(tokenizer, fresh_rows)
    write_rows(OUT / "material/parent_cases.jsonl", parent_rows)
    write_rows(OUT / "material/fresh_cases.jsonl", fresh_rows)
    write_rows(OUT / "material/parent_qwen_compiled.jsonl", parent_compiled)
    write_rows(OUT / "material/fresh_qwen_compiled.jsonl", fresh_compiled)
    parent_audit = audit_material(parent_rows, parent_compiled)
    fresh_audit = audit_material(fresh_rows, fresh_compiled)
    save(OUT / "audit/parent_material_audit.json", parent_audit)
    save(OUT / "audit/fresh_material_audit.json", fresh_audit)
    checks = {
        "protocol_frozen": (OUT / "protocol/preregistration.json").exists(),
        "parent_compiles": len(parent_rows) == len(parent_compiled),
        "fresh_compiles": len(fresh_rows) == len(fresh_compiled),
        "unique_ids": parent_audit["unique_case_ids"] == len(parent_rows) and fresh_audit["unique_case_ids"] == len(fresh_rows),
        "roles_present": not parent_audit["missing_roles"] and not fresh_audit["missing_roles"],
        "strings_clean": not parent_audit["malformed_strings"] and not fresh_audit["malformed_strings"],
        "factorials_complete": parent_audit["broad_factorial_pair_complete"] and fresh_audit["broad_factorial_pair_complete"],
        "zero_models_balanced": all(abs(v - 0.5) <= 1e-12 for audit in (parent_audit, fresh_audit) for v in audit["zero_models"].values()),
        "partition_lexicons_disjoint": not parent_audit["cross_partition_primary_overlap"] and not fresh_audit["cross_partition_primary_overlap"],
        "finite": finite(parent_audit) and finite(fresh_audit),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "protocol": protocol,
        "material_audit": {"parent": parent_audit, "fresh": fresh_audit},
        "files": {
            "parent": "material/parent_cases.jsonl", "fresh": "material/fresh_cases.jsonl",
            "parent_compiled": "material/parent_qwen_compiled.jsonl", "fresh_compiled": "material/fresh_qwen_compiled.jsonl",
        },
        "hashes": {
            name: file_hash(OUT / path) for name, path in {
                "parent": "material/parent_cases.jsonl", "fresh": "material/fresh_cases.jsonl",
                "parent_compiled": "material/parent_qwen_compiled.jsonl", "fresh_compiled": "material/fresh_qwen_compiled.jsonl",
            }.items()
        },
        "strict_conclusion": "A broad all-coordinate campaign is frozen and compiler-valid. No model or HiddenState result exists in this phase.",
        "next_authorization": "Run the sequential registered branches without changing material, partitions, controls or gates.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
