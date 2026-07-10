#!/usr/bin/env python3
"""Frozen 9-family, 72-mechanism case bank for the Phase330 global atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas"
PHASE = "Phase330"
SCHEMA_VERSION = "8.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
TEMPLATES = ("template_a", "template_b", "template_c")


FAMILY_MECHANISMS: dict[str, tuple[str, ...]] = {
    "content_knowledge": (
        "category", "attribute", "function", "part", "material", "habitat",
        "comparison_relation", "negated_attribute",
    ),
    "output_protocol": (
        "answer_only", "single_sentence", "single_item_list", "json", "quote_closure",
        "newline_closure", "format_template", "no_explanation",
    ),
    "reasoning_constraint": (
        "direct_entailment", "direct_contradiction", "two_hop_entailment",
        "two_hop_blocked", "transitive_order", "reversed_order_control",
        "conjunction_rule", "missing_condition_control",
    ),
    "syntax_structure": (
        "subject_role", "object_role", "singular_agreement", "plural_agreement",
        "past_tense", "pronoun_number", "adjective_attachment", "relative_clause_role",
    ),
    "language_action": (
        "answer", "classify", "extract", "transform", "translate", "rewrite",
        "summarize", "refuse_or_comply",
    ),
    "cross_lingual": (
        "semantic_equivalence", "translation", "negation", "question", "role_binding",
        "number_agreement", "protocol_preservation", "mixed_language_routing",
    ),
    "readout_competition": (
        "target_vs_wrong", "target_vs_continue", "target_vs_echo", "target_vs_protocol",
        "target_vs_punctuation", "answer_alias", "multi_token_answer", "full_vocabulary_blockers",
    ),
    "state_drift": (
        "entity_drift", "attribute_drift", "role_drift", "language_drift", "format_drift",
        "reasoning_drift", "repetition_drift", "long_context_drift",
    ),
    "closure": (
        "semantic_completion", "protocol_completion", "stop_wins", "continue_suppression",
        "multi_token_completion", "alias_completion", "generation_stability",
        "client_visible_closure",
    ),
}


FAMILY_NAMES = {
    "content_knowledge": "内容知识",
    "output_protocol": "输出协议",
    "reasoning_constraint": "推理约束",
    "syntax_structure": "语法结构",
    "language_action": "语言动作",
    "cross_lingual": "跨语言",
    "readout_competition": "读出竞争",
    "state_drift": "状态漂移",
    "closure": "闭合",
}


BASE_QA = [
    ("A ripe banana is being described.", "What is its usual color?", "yellow", ["green", "blue"]),
    ("Grass in a healthy lawn is being described.", "What is its usual color?", "green", ["red", "black"]),
    ("Fresh snow is being described.", "What is its usual color?", "white", ["yellow", "blue"]),
    ("Coal is being described.", "What is its usual color?", "black", ["white", "green"]),
    ("A lemon is being tasted.", "What is its usual taste?", "sour", ["sweet", "salty"]),
    ("Sugar is being tasted.", "What is its usual taste?", "sweet", ["sour", "bitter"]),
    ("A small hammer is on a workbench.", "What broad kind of object is it?", "tool", ["animal", "plant"]),
    ("A sparrow is perched on a branch.", "What broad kind of animal is it?", "bird", ["fish", "mammal"]),
    ("A rose is growing in a garden.", "What broad kind of living thing is it?", "plant", ["tool", "vehicle"]),
    ("A canoe is moving across a lake.", "What broad kind of object is it?", "vehicle", ["plant", "mammal"]),
    ("Two plus three is being calculated.", "What is the result?", "5", ["4", "6"]),
    ("Seven plus four is being calculated.", "What is the result?", "11", ["10", "12"]),
    ("Paris is a major European capital.", "Which country contains it?", "France", ["Italy", "Spain"]),
    ("Rome is a major European capital.", "Which country contains it?", "Italy", ["France", "Spain"]),
    ("Tokyo is a major Asian capital.", "Which country contains it?", "Japan", ["China", "Korea"]),
    ("Madrid is a major European capital.", "Which country contains it?", "Spain", ["Italy", "France"]),
    ("Water freezes under standard conditions.", "At what Celsius temperature does it freeze?", "0", ["10", "100"]),
    ("A triangle is a simple polygon.", "How many sides does it have?", "3", ["4", "5"]),
    ("A week is a calendar interval.", "How many days does it contain?", "7", ["5", "8"]),
    ("Earth has one natural satellite.", "What is that satellite commonly called?", "moon", ["sun", "mars"]),
    ("Bread is paired in a familiar phrase.", "Complete: bread and what?", "butter", ["water", "stone"]),
    ("A duck makes a familiar sound.", "What sound word is commonly used?", "quack", ["bark", "meow"]),
    ("The opposite of up is requested.", "What is the opposite?", "down", ["left", "inside"]),
    ("The opposite of yes is requested.", "What is the opposite?", "no", ["maybe", "always"]),
]


CATEGORY_GROUPS = {
    "bird": ["heron", "finch", "albatross", "sparrow"],
    "tool": ["hammer", "chisel", "mallet", "crowbar"],
    "plant": ["orchid", "fern", "sequoia", "moss"],
    "vehicle": ["tram", "canoe", "scooter", "truck"],
    "mammal": ["giraffe", "badger", "otter", "horse"],
    "instrument": ["clarinet", "cello", "flute", "harp"],
}


ATTRIBUTE_GROUPS = {
    "red": ["ripe tomato", "stop sign", "cranberry", "cardinal"],
    "yellow": ["ripe banana", "daffodil", "school bus", "lemon peel"],
    "green": ["healthy grass", "pea pod", "fern leaf", "lime peel"],
    "blue": ["clear daytime sky", "cobalt glass", "lapis stone", "blueberry skin"],
    "black": ["coal", "rubber tire", "crow feather", "ink"],
    "white": ["fresh snow", "chalk", "egret feather", "table salt"],
}


FUNCTION_PAIRS = [
    ("hammer", "hit"), ("knife", "cut"), ("broom", "sweep"), ("key", "unlock"),
    ("pen", "write"), ("saw", "cut"), ("shovel", "dig"), ("needle", "sew"),
    ("thermometer", "measure"), ("camera", "photograph"), ("kettle", "boil"),
    ("umbrella", "shelter"), ("scissors", "cut"), ("brush", "paint"),
    ("telescope", "observe"), ("compass", "navigate"), ("ladle", "serve"),
    ("eraser", "erase"), ("ruler", "measure"), ("whistle", "signal"),
    ("flashlight", "illuminate"), ("stapler", "fasten"), ("drill", "bore"),
    ("calculator", "calculate"),
]


PART_PAIRS = [
    ("wheel", "car"), ("wing", "bird"), ("leaf", "plant"), ("handle", "cup"),
    ("key", "keyboard"), ("screen", "phone"), ("blade", "knife"), ("pedal", "bicycle"),
    ("door", "house"), ("page", "book"), ("finger", "hand"), ("toe", "foot"),
    ("branch", "tree"), ("petal", "flower"), ("engine", "vehicle"), ("sail", "boat"),
    ("button", "shirt"), ("lens", "camera"), ("string", "guitar"), ("drawer", "desk"),
    ("roof", "building"), ("tail", "animal"), ("rung", "ladder"), ("strap", "bag"),
]


MATERIAL_PAIRS = [
    ("window", "glass"), ("spoon", "metal"), ("shirt", "cotton"), ("table", "wood"),
    ("tire", "rubber"), ("bottle", "plastic"), ("brick", "clay"), ("ring", "gold"),
    ("wire", "copper"), ("blanket", "wool"), ("notebook", "paper"), ("shoe", "leather"),
    ("statue", "stone"), ("can", "aluminum"), ("rope", "fiber"), ("mirror", "glass"),
    ("pencil", "wood"), ("coin", "metal"), ("sweater", "wool"), ("cup", "ceramic"),
    ("helmet", "plastic"), ("pan", "steel"), ("scarf", "silk"), ("floor tile", "ceramic"),
]


HABITAT_GROUPS = {
    "ocean": ["swordfish", "octopus", "manta ray", "tuna"],
    "forest": ["lynx", "chipmunk", "deer", "woodpecker"],
    "desert": ["jerboa", "fennec", "sidewinder", "camel"],
    "arctic": ["ptarmigan", "walrus", "narwhal", "musk ox"],
    "pond": ["tadpole", "newt", "dragonfly nymph", "freshwater snail"],
    "cave": ["blind shrimp", "swiftlet", "horseshoe bat", "salamander"],
}


NAMES = [
    "Ava", "Ben", "Cora", "Dion", "Eli", "Faye", "Gus", "Hana", "Ivan", "Jade", "Kian", "Lena",
    "Milo", "Nora", "Omar", "Pia", "Quin", "Rina", "Seth", "Tara", "Uma", "Vik", "Wren", "Yara",
]


EN_ZH = [
    ("cat", "猫"), ("dog", "狗"), ("bird", "鸟"), ("water", "水"), ("fire", "火"),
    ("snow", "雪"), ("moon", "月亮"), ("sun", "太阳"), ("red", "红色"), ("blue", "蓝色"),
    ("green", "绿色"), ("white", "白色"), ("black", "黑色"), ("book", "书"),
    ("tree", "树"), ("flower", "花"), ("car", "汽车"), ("train", "火车"),
    ("apple", "苹果"), ("banana", "香蕉"), ("one", "一"), ("two", "二"),
    ("large", "大"), ("small", "小"),
]


SYNONYMS = [
    ("quick", "fast"), ("large", "big"), ("small", "tiny"), ("begin", "start"),
    ("finish", "end"), ("silent", "quiet"), ("angry", "mad"), ("glad", "happy"),
    ("smart", "clever"), ("simple", "easy"), ("difficult", "hard"), ("purchase", "buy"),
    ("assist", "help"), ("select", "choose"), ("reply", "answer"), ("infant", "baby"),
    ("error", "mistake"), ("close", "shut"), ("near", "close"), ("rapid", "fast"),
    ("job", "work"), ("gift", "present"), ("idea", "thought"), ("route", "path"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record(
    context: str,
    question: str,
    target: str,
    distractors: list[str],
    *,
    aliases: list[str] | None = None,
    instruction: str = "Give the shortest correct answer.",
    language: str = "en",
    protocol: str = "short",
    polarity: str = "positive",
    expected_structure: str = "plain",
) -> dict[str, Any]:
    return {
        "context": context,
        "question": question,
        "target": target,
        "target_aliases": list(dict.fromkeys([target, *(aliases or [])])),
        "distractors": [value for value in distractors if value != target],
        "instruction": instruction,
        "language": language,
        "protocol": protocol,
        "polarity": polarity,
        "expected_structure": expected_structure,
    }


def content_task(mechanism: str, index: int) -> dict[str, Any]:
    if mechanism in {"category", "attribute", "habitat"}:
        groups = {
            "category": CATEGORY_GROUPS,
            "attribute": ATTRIBUTE_GROUPS,
            "habitat": HABITAT_GROUPS,
        }[mechanism]
        targets = list(groups)
        target = targets[index // 4]
        subject = groups[target][index % 4]
        label = {"category": "category", "attribute": "ordinary attribute", "habitat": "usual habitat"}[mechanism]
        return record(
            f"The subject under study is the {subject}.",
            f"What is its {label}?",
            target,
            [value for value in targets if value != target],
        )
    if mechanism == "function":
        subject, target = FUNCTION_PAIRS[index]
        return record(
            f"The object under study is a {subject}.",
            "What is one ordinary function of this object?",
            target,
            [FUNCTION_PAIRS[(index + 5) % 24][1], FUNCTION_PAIRS[(index + 11) % 24][1]],
        )
    if mechanism == "part":
        subject, target = PART_PAIRS[index]
        return record(
            f"The component under study is a {subject}.",
            "What larger object does this component commonly belong to?",
            target,
            [PART_PAIRS[(index + 7) % 24][1], PART_PAIRS[(index + 13) % 24][1]],
        )
    if mechanism == "material":
        subject, target = MATERIAL_PAIRS[index]
        return record(
            f"The object under study is a {subject}.",
            "What material is it commonly made from?",
            target,
            [MATERIAL_PAIRS[(index + 5) % 24][1], MATERIAL_PAIRS[(index + 9) % 24][1]],
        )
    if mechanism in {"comparison_relation", "negated_attribute"}:
        if mechanism == "comparison_relation":
            left = 7 + index
            right = 3 + ((index * 5) % 29)
            if left == right:
                right += 2
            ask_larger = index % 2 == 0
            target = str(max(left, right) if ask_larger else min(left, right))
            return record(
                f"The two numbers are {left} and {right}.",
                f"Which number is {'larger' if ask_larger else 'smaller'}?",
                target,
                [str(left if target != str(left) else right)],
            )
        colors = list(ATTRIBUTE_GROUPS)
        subject = list(sum(ATTRIBUTE_GROUPS.values(), []))[index]
        wrong = colors[index % len(colors)]
        target = colors[(index + 2) % len(colors)]
        return record(
            f"The {subject} is not {wrong}; in this case it is {target}.",
            f"Which stated attribute applies to the {subject}?",
            target,
            [wrong, colors[(index + 4) % len(colors)]],
            polarity="negative_contrast",
        )
    raise KeyError(mechanism)


def base_qa_task(index: int, instruction: str, protocol: str, expected: str) -> dict[str, Any]:
    context, question, target, distractors = BASE_QA[index]
    return record(
        context, question, target, distractors,
        instruction=instruction, protocol=protocol, expected_structure=expected,
    )


def output_protocol_task(mechanism: str, index: int) -> dict[str, Any]:
    specs = {
        "answer_only": ("Return only the answer and nothing else.", "answer_only", "plain"),
        "single_sentence": ("Answer in exactly one short sentence.", "single_sentence", "sentence"),
        "single_item_list": ("Return exactly one bullet item.", "single_item_list", "list"),
        "json": ('Return JSON only with key "answer".', "json", "json"),
        "quote_closure": ("Put the answer inside one matching pair of double quotes.", "quote", "quoted"),
        "newline_closure": ("Write the answer on one line and do not add another line.", "one_line", "one_line"),
        "format_template": ("Use exactly this template: ANSWER=<answer>", "format_template", "key_value"),
        "no_explanation": ("Give the answer without explanation or reasoning.", "no_explanation", "plain"),
    }
    return base_qa_task(index, *specs[mechanism])


def reasoning_task(mechanism: str, index: int) -> dict[str, Any]:
    name = NAMES[index]
    cls = f"class{chr(97 + index % 6)}"
    prop1 = ["blue", "round", "warm", "quiet", "large", "swift"][index % 6]
    prop2 = ["calm", "bright", "solid", "kind", "open", "young"][index % 6]
    positive = index % 2 == 0
    yes_no = "yes" if positive else "no"
    if mechanism == "direct_entailment":
        stated = prop1 if positive else prop2
        return record(
            f"Every {cls} is {stated}. {name} is a {cls}.",
            f"Is {name} {prop1}?",
            yes_no, ["no" if positive else "yes"], polarity="balanced_yes_no",
        )
    if mechanism == "direct_contradiction":
        context = f"No {cls} is {prop1}. {name} is a {cls}." if not positive else f"Every {cls} is {prop1}. {name} is a {cls}."
        return record(context, f"Is {name} {prop1}?", yes_no, ["no" if positive else "yes"], polarity="balanced_yes_no")
    if mechanism in {"two_hop_entailment", "two_hop_blocked"}:
        if mechanism == "two_hop_entailment":
            second = f"Every {prop1} thing is {prop2}." if positive else f"No {prop1} thing is {prop2}."
        else:
            second = f"No {prop1} thing is {prop2}." if positive else f"Every {prop1} thing is {prop2}."
            yes_no = "no" if positive else "yes"
        return record(
            f"Every {cls} is {prop1}. {second} {name} is a {cls}.",
            f"Is {name} {prop2}?", yes_no, ["no" if yes_no == "yes" else "yes"], polarity="balanced_yes_no",
        )
    if mechanism in {"transitive_order", "reversed_order_control"}:
        a, b, c = NAMES[index], NAMES[(index + 3) % 24], NAMES[(index + 7) % 24]
        if mechanism == "transitive_order":
            question = f"Who is taller, {a} or {c}?" if positive else f"Who is shorter, {a} or {c}?"
            target = a if positive else c
        else:
            question = f"Who is shorter, {a} or {c}?" if positive else f"Who is taller, {a} or {c}?"
            target = c if positive else a
        return record(f"{a} is taller than {b}. {b} is taller than {c}.", question, target, [c if target == a else a])
    if mechanism in {"conjunction_rule", "missing_condition_control"}:
        has_both = positive if mechanism == "conjunction_rule" else False
        facts = f"{name} is red and round." if has_both else f"{name} is red but its shape is not given."
        target = "yes" if has_both else "no"
        return record(
            f"If something is red and round, it is marked. {facts}",
            f"Can we conclude that {name} is marked?",
            target, ["no" if target == "yes" else "yes"], polarity="conjunction_control",
        )
    raise KeyError(mechanism)


def syntax_task(mechanism: str, index: int) -> dict[str, Any]:
    subject = NAMES[index]
    obj = ["book", "cup", "ball", "letter", "map", "key"][index % 6]
    verb = ["carried", "opened", "found", "moved", "painted", "washed"][index % 6]
    if mechanism == "subject_role":
        return record(f"{subject} {verb} the {obj}.", "Who performed the action?", subject, [obj])
    if mechanism == "object_role":
        return record(f"{subject} {verb} the {obj}.", "What received the action?", obj, [subject])
    if mechanism in {"singular_agreement", "plural_agreement"}:
        plural = mechanism == "plural_agreement"
        noun = "dogs" if plural else "dog"
        target = "run" if plural else "runs"
        return record(f"The subject is '{noun}'.", "Complete with the agreeing verb: The subject ___ daily.", target, ["runs" if plural else "run"])
    if mechanism == "past_tense":
        base, past = [("walk", "walked"), ("jump", "jumped"), ("play", "played"), ("open", "opened"), ("wash", "washed"), ("call", "called")][index % 6]
        return record(f"Yesterday, {subject} completed the action '{base}'.", "Give the regular past-tense verb.", past, [base, base + "s"])
    if mechanism == "pronoun_number":
        plural = index % 2 == 0
        entity = f"{NAMES[index]} and {NAMES[(index + 1) % 24]}" if plural else NAMES[index]
        target = "they" if plural else "she"
        return record(f"The referenced entity is {entity}.", "Which pronoun agrees in number?", target, ["she" if plural else "they"])
    if mechanism == "adjective_attachment":
        adjective = ["red", "small", "old", "wooden", "bright", "round"][index % 6]
        return record(f"The phrase is 'the {adjective} {obj}'.", f"What noun does '{adjective}' describe?", obj, [subject])
    if mechanism == "relative_clause_role":
        other = NAMES[(index + 5) % 24]
        return record(f"{subject}, who greeted {other}, smiled.", "Who greeted the other person?", subject, [other])
    raise KeyError(mechanism)


def action_task(mechanism: str, index: int) -> dict[str, Any]:
    context, question, target, distractors = BASE_QA[index]
    if mechanism == "answer":
        return record(context, question, target, distractors, instruction="Answer the question directly.")
    if mechanism == "classify":
        groups = list(CATEGORY_GROUPS)
        expected = groups[index % len(groups)]
        subject = CATEGORY_GROUPS[expected][(index // len(groups)) % 4]
        return record(f"The item is {subject}.", "Classify the item with one category word.", expected, [x for x in groups if x != expected])
    if mechanism == "extract":
        code = f"K{100 + index}"
        return record(f"Record: owner={NAMES[index]}; code={code}; status=active.", "Extract the code value.", code, [NAMES[index], "active"])
    if mechanism == "transform":
        word = SYNONYMS[index][0]
        target_value = word.upper() if index % 2 == 0 else word.lower()
        instruction = "Convert the word to uppercase." if index % 2 == 0 else "Convert the word to lowercase."
        return record(f"The source word is {word}.", instruction, target_value, [word.capitalize()])
    if mechanism == "translate":
        en, zh = EN_ZH[index]
        if index % 2 == 0:
            return record(f"The source word is {en}.", "Translate it to Chinese.", zh, [en], language="en_zh")
        return record(f"源词是“{zh}”。", "Translate it to English.", en, [zh], language="zh_en")
    if mechanism == "rewrite":
        source, target_value = SYNONYMS[index]
        return record(f"The source word is {source}.", "Rewrite it with the registered synonym.", target_value, [source])
    if mechanism == "summarize":
        topic = ["rain", "snow", "music", "travel", "health", "science"][index % 6]
        return record(f"This short passage repeatedly discusses {topic}, its causes, and its effects.", "Summarize the topic in one word.", topic, ["other", "unknown"])
    if mechanism == "refuse_or_comply":
        refuse = index % 2 == 0
        context_value = "The request asks to reveal a private password." if refuse else "The request asks to compute two plus two."
        target_value = "refuse" if refuse else "comply"
        return record(context_value, "Choose the appropriate action label.", target_value, ["comply" if refuse else "refuse"], polarity="safe_action_control")
    raise KeyError(mechanism)


def cross_lingual_task(mechanism: str, index: int) -> dict[str, Any]:
    en, zh = EN_ZH[index]
    if mechanism == "semantic_equivalence":
        equivalent = index % 2 == 0
        other = zh if equivalent else EN_ZH[(index + 5) % 24][1]
        target = "yes" if equivalent else "no"
        return record(f"English '{en}' is compared with Chinese '{other}'.", "Do they have the same basic meaning?", target, ["no" if equivalent else "yes"], language="mixed")
    if mechanism == "translation":
        return record(f"The Chinese source word is '{zh}'.", "Translate it into English.", en, [EN_ZH[(index + 3) % 24][0]], language="zh_en")
    if mechanism == "negation":
        colors = ["red", "blue", "green", "white", "black", "yellow"]
        target = colors[index % 6]
        wrong = colors[(index + 1) % 6]
        return record(f"这个物体不是{wrong}，而是{target}。", "Answer the stated color in English.", target, [wrong], language="mixed", polarity="negative_contrast")
    if mechanism == "question":
        return record(f"词语“{zh}”需要识别。", "What is this word in English?", en, [EN_ZH[(index + 4) % 24][0]], language="mixed")
    if mechanism == "role_binding":
        subject = NAMES[index]
        obj = EN_ZH[index][0]
        return record(f"{subject}拿着一个{zh}。", "Who holds the object?", subject, [obj], language="mixed")
    if mechanism == "number_agreement":
        plural = index % 2 == 0
        target = "are" if plural else "is"
        noun = "两只猫" if plural else "一只猫"
        return record(f"The bilingual subject refers to {noun}.", "Choose the English agreeing verb: is or are.", target, ["is" if plural else "are"], language="mixed")
    if mechanism == "protocol_preservation":
        return record(f"Translate '{zh}' while preserving a one-word response.", "Give only the English translation.", en, [zh], language="mixed", protocol="answer_only")
    if mechanism == "mixed_language_routing":
        target = en if index % 2 == 0 else zh
        question = "Answer in English." if index % 2 == 0 else "请用中文回答。"
        return record(f"The concept is given as {en} / {zh}.", question, target, [zh if index % 2 == 0 else en], language="mixed")
    raise KeyError(mechanism)


def wrapper_task(family: str, mechanism: str, index: int) -> dict[str, Any]:
    context, question, target, distractors = BASE_QA[index]
    instruction = "Give the shortest correct answer."
    protocol = "short"
    expected = "plain"
    aliases: list[str] = []
    if family == "readout_competition":
        mapping = {
            "target_vs_wrong": "Choose the correct content, not a distractor.",
            "target_vs_continue": "Give the answer first; do not begin with a continuation phrase.",
            "target_vs_echo": "Do not copy the question; output the answer.",
            "target_vs_protocol": "Give content before any explanation.",
            "target_vs_punctuation": "Begin with content rather than punctuation.",
            "answer_alias": "A registered answer alias is acceptable.",
            "multi_token_answer": "Return the complete answer phrase.",
            "full_vocabulary_blockers": "Return the most likely correct content answer.",
        }
        instruction = mapping[mechanism]
        if mechanism == "answer_alias":
            aliases = [target.upper() if target.isalpha() else target]
        if mechanism == "multi_token_answer":
            target = f"the {target}"
            distractors = [f"the {value}" for value in distractors]
            expected = "multi_token"
    elif family == "state_drift":
        mapping = {
            "entity_drift": "Keep the named entity fixed and answer once.",
            "attribute_drift": "Keep the stated attribute fixed and answer once.",
            "role_drift": "Keep subject and object roles fixed.",
            "language_drift": "Answer only in English and do not switch language.",
            "format_drift": "Use ANSWER=<answer> and do not change format.",
            "reasoning_drift": "Keep the stated conclusion and do not reverse it.",
            "repetition_drift": "Give the answer once; do not repeat it.",
            "long_context_drift": "Ignore repeated filler and keep the original task.",
        }
        instruction = mapping[mechanism]
        if mechanism == "format_drift":
            protocol, expected = "format_template", "key_value"
        if mechanism == "long_context_drift":
            context = context + " " + "Background detail is irrelevant. " * 12
    elif family == "closure":
        mapping = {
            "semantic_completion": "Give the complete semantic answer.",
            "protocol_completion": "Give only the answer in the required form.",
            "stop_wins": "Give the answer and stop immediately.",
            "continue_suppression": "Do not continue after the answer.",
            "multi_token_completion": "Return the full phrase without truncation.",
            "alias_completion": "A registered alias is acceptable; finish the answer.",
            "generation_stability": "Give the same final answer once and stop.",
            "client_visible_closure": "Use ANSWER=<answer> so completion is machine visible.",
        }
        instruction = mapping[mechanism]
        if mechanism == "multi_token_completion":
            target = f"the {target}"
            distractors = [f"the {value}" for value in distractors]
            expected = "multi_token"
        if mechanism == "alias_completion":
            aliases = [target.upper() if target.isalpha() else target]
        if mechanism == "client_visible_closure":
            protocol, expected = "format_template", "key_value"
    return record(
        context, question, target, distractors,
        aliases=aliases, instruction=instruction, protocol=protocol, expected_structure=expected,
    )


def task_for(family: str, mechanism: str, index: int) -> dict[str, Any]:
    if family == "content_knowledge":
        return content_task(mechanism, index)
    if family == "output_protocol":
        return output_protocol_task(mechanism, index)
    if family == "reasoning_constraint":
        return reasoning_task(mechanism, index)
    if family == "syntax_structure":
        return syntax_task(mechanism, index)
    if family == "language_action":
        return action_task(mechanism, index)
    if family == "cross_lingual":
        return cross_lingual_task(mechanism, index)
    return wrapper_task(family, mechanism, index)


def render_prompt(task: dict[str, Any], template: str) -> tuple[str, str, str]:
    context = task["context"]
    question = task["question"]
    instruction = task["instruction"]
    if template == "template_a":
        prompt = f"{context}\n{question}\n{instruction}\nAnswer:"
    elif template == "template_b":
        prompt = f"Context: {context}\nTask: {question}\nInstruction: {instruction}\nResponse:"
    elif template == "template_c":
        prompt = f"{context}\n{instruction}\nQuestion: {question}\nFinal:"
    else:
        raise KeyError(template)
    return prompt, context, question


def item_split(index: int) -> str:
    if index < 12:
        return "discovery"
    if index < 18:
        return "calibration"
    return "heldout"


def target_bucket(target: str) -> str:
    if target.lower() in {"yes", "no"}:
        return "yes_no"
    if target.isdigit():
        return "numeric"
    if any("\u4e00" <= char <= "\u9fff" for char in target):
        return "zh_lexical"
    if len(target.split()) > 1:
        return "multi_token"
    return "lexical"


def build_cases() -> list[dict[str, Any]]:
    rows = []
    created_at = now()
    for family, mechanisms in FAMILY_MECHANISMS.items():
        for mechanism in mechanisms:
            for item_index in range(24):
                task = task_for(family, mechanism, item_index)
                item_id = f"phase330_{family}_{mechanism}_{item_index:02d}"
                for template in TEMPLATES:
                    prompt, source_fragment, query_fragment = render_prompt(task, template)
                    rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": created_at,
                        "case_id": f"{item_id}_{template}",
                        "item_id": item_id,
                        "family_id": family,
                        "family_name": FAMILY_NAMES[family],
                        "mechanism_id": mechanism,
                        "item_index": item_index,
                        "split": item_split(item_index),
                        "template_id": template,
                        "template_role": "heldout_style" if template == "template_c" else "selection_style",
                        "selection_eligible": template != "template_c" and item_index < 18,
                        "language": task["language"],
                        "prompt": prompt,
                        "context": task["context"],
                        "question": task["question"],
                        "instruction": task["instruction"],
                        "source_fragments": [source_fragment],
                        "query_fragment": query_fragment,
                        "target": task["target"],
                        "target_aliases": task["target_aliases"],
                        "distractors": task["distractors"],
                        "target_bucket": target_bucket(task["target"]),
                        "target_word_count": len(task["target"].split()),
                        "candidate_set_size": 1 + len(task["distractors"]),
                        "protocol": task["protocol"],
                        "expected_structure": task["expected_structure"],
                        "polarity": task["polarity"],
                        "negative_control": task["polarity"] != "positive" or item_index % 6 == 5,
                        "open_set_control": item_index >= 20,
                        "designed_frequency_bucket": "common",
                        "target_absent_from_prompt": task["target"].lower() not in prompt.lower(),
                        "selection_updates_allowed": False,
                        "scientific_unit": {
                            "family": family,
                            "mechanism": mechanism,
                            "language": task["language"],
                            "template": template,
                            "item": item_id,
                        },
                    })
    return rows


def validate_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(row["family_id"] for row in rows)
    mechanism_counts = Counter((row["family_id"], row["mechanism_id"]) for row in rows)
    unique_split_items: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        key = (row["family_id"], row["mechanism_id"], row["split"])
        unique_split_items.setdefault(key, set()).add(row["item_id"])
    order_errors = sum(
        row["prompt"].index(row["source_fragments"][0])
        >= row["prompt"].index(row["query_fragment"])
        for row in rows
    )
    expected_mechanisms = {
        (family, mechanism)
        for family, mechanisms in FAMILY_MECHANISMS.items()
        for mechanism in mechanisms
    }
    observed_mechanisms = set(mechanism_counts)
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "family_count": len(family_counts),
        "mechanism_count": len(mechanism_counts),
        "independent_item_count": len({row["item_id"] for row in rows}),
        "prompt_case_count": len(rows),
        "prompt_model_plan_count": len(rows) * len(MODELS),
        "family_prompt_counts": dict(sorted(family_counts.items())),
        "mechanism_prompt_count_values": sorted(set(mechanism_counts.values())),
        "template_counts": dict(sorted(Counter(row["template_id"] for row in rows).items())),
        "split_item_count_values": {
            split: sorted({
                len(values) for (family, mechanism, key_split), values in unique_split_items.items()
                if key_split == split
            })
            for split in ("discovery", "calibration", "heldout")
        },
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "source_query_order_error_count": order_errors,
        "missing_mechanism_count": len(expected_mechanisms - observed_mechanisms),
        "unexpected_mechanism_count": len(observed_mechanisms - expected_mechanisms),
        "empty_target_count": sum(not row["target"] for row in rows),
        "empty_distractor_count": sum(not row["distractors"] for row in rows),
        "target_leak_count": sum(not row["target_absent_from_prompt"] for row in rows),
        "selection_eligible_count": sum(row["selection_eligible"] for row in rows),
        "heldout_style_count": sum(row["template_role"] == "heldout_style" for row in rows),
        "selection_updates_allowed": False,
        "case_bank_sha256": stable_hash([
            {key: row[key] for key in (
                "case_id", "family_id", "mechanism_id", "split", "template_id",
                "prompt", "target", "distractors",
            )}
            for row in rows
        ]),
    }
    result["valid"] = (
        result["family_count"] == 9
        and result["mechanism_count"] == 72
        and result["independent_item_count"] == 1728
        and result["prompt_case_count"] == 5184
        and result["prompt_model_plan_count"] == 15552
        and set(result["family_prompt_counts"].values()) == {576}
        and result["mechanism_prompt_count_values"] == [72]
        and set(result["template_counts"].values()) == {1728}
        and result["split_item_count_values"] == {
            "discovery": [12], "calibration": [6], "heldout": [6]
        }
        and all(result[key] == 0 for key in (
            "duplicate_case_id_count", "source_query_order_error_count", "missing_mechanism_count",
            "unexpected_mechanism_count", "empty_target_count", "empty_distractor_count",
        ))
    )
    return result


def manifest(rows: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Nine-family global physical atlas census",
        "status": "case_bank_frozen",
        "models": list(MODELS),
        "families": [
            {
                "family_id": family,
                "family_name": FAMILY_NAMES[family],
                "mechanisms": list(mechanisms),
            }
            for family, mechanisms in FAMILY_MECHANISMS.items()
        ],
        "planned_counts": {
            "families": 9,
            "mechanisms": 72,
            "items_per_mechanism": 24,
            "templates_per_item": 3,
            "prompt_cases": 5184,
            "prompt_model_cases": 15552,
            "behavior_rows": 15552,
            "readout_rows": 15552,
            "rollout_rows": 15552,
            "top50_rows": 777600,
            "component_event_rows": 4852224,
            "registered_causal_cases": 432,
            "registered_causal_condition_rows": 4320,
        },
        "split_contract": {"discovery_items": 12, "calibration_items": 6, "heldout_items": 6},
        "template_contract": {
            "selection_styles": ["template_a", "template_b"],
            "heldout_style": "template_c",
        },
        "analysis_freeze": (
            "During model execution only completeness, missing rows, hashes, schema errors, and runtime "
            "failures may be inspected. Scientific claims are generated only after all three models complete."
        ),
        "case_bank_sha256": validation["case_bank_sha256"],
        "validation": validation,
        "single_unit_intervention_gate_open": False,
        "selection_updates_allowed": False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="nine_family_global_atlas")
    args = parser.parse_args()
    rows = build_cases()
    validation = validate_cases(rows)
    output = OUT / args.round
    write_jsonl(output / "phase330_case_bank.jsonl", rows)
    write_json(output / "phase330_case_bank_validation.json", validation)
    write_json(output / "phase330_manifest.json", manifest(rows, validation))
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
