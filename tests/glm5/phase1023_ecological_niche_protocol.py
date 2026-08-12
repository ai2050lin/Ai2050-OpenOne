#!/usr/bin/env python3
"""Freeze the Phase1023 ecological-niche and execution-fork protocol.

The protocol does not assume a language mechanism formula.  It creates new,
balanced observations for three questions:

1. Do semantic families generalize to unseen concepts and prompts?
2. Does a concept retain a distinguishable niche across languages and tasks?
3. After separating protocol errors, is there an output-preceding difference
   between semantically correct and semantically wrong translation?
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import render_chat, tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans


PHASE = 1023
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
LANGUAGES = ("en", "zh", "fr")
PROMPT_SPLITS = ("discovery", "confirmation")
CONCEPT_PARTITIONS = ("calibration", "heldout")
ATLAS_TASKS = ("mention", "translate")
ATLAS_ROLES = ("source_end", "pre_output")
CATEGORIES = (
    "fruit",
    "animal",
    "vehicle",
    "profession",
    "place",
    "object",
    "color",
    "body_part",
)
LANGUAGE_DIRECTIONS = tuple(
    (source, target)
    for source in LANGUAGES
    for target in LANGUAGES
    if source != target
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1023_ecological_niche_execution_fork"
)

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
}

CATEGORY_OUTPUTS = {
    "fruit": ("fruit",),
    "animal": ("animal",),
    "vehicle": ("vehicle", "transport"),
    "profession": ("profession", "occupation", "job"),
    "place": ("place", "location"),
    "object": ("object", "item"),
    "color": ("color", "colour"),
    "body_part": ("body part", "part of the body"),
}

TARGET_BY_SPLIT = {
    "discovery": {"en": "zh", "zh": "fr", "fr": "en"},
    "confirmation": {"en": "fr", "zh": "en", "fr": "zh"},
}

MENTION_TEMPLATES = {
    "discovery": (
        "Read this {source_name} term and retain its meaning for the next "
        "step.\nTerm: {term}\nMeaning state:",
        "Read",
    ),
    "confirmation": (
        "Consider the meaning of the following {source_name} expression.\n"
        "Expression: {term}\nInternal meaning:",
        "Consider",
    ),
}

TRANSLATION_TEMPLATES = {
    "discovery": (
        "Translate the {source_name} term below into {target_name}. "
        "Your response must contain the translation.\n"
        "Term: {term}\nAnswer:",
        "Translate",
    ),
    "confirmation": (
        "What is the {target_name} equivalent of this {source_name} term? "
        "Put the equivalent first.\n"
        "Input: {term}\nEquivalent:",
        "equivalent",
    ),
}

CLASSIFICATION_TEMPLATES = {
    "discovery": (
        "Which broad semantic category best describes this {source_name} "
        "term? Return one English category label.\nTerm: {term}\nCategory:"
    ),
    "confirmation": (
        "Classify this {source_name} expression by its general meaning. "
        "Give only its broad English class.\nExpression: {term}\nClass:"
    ),
}


def concept(
    concept_id: str,
    category: str,
    partition: str,
    en: str,
    zh: str,
    fr: str,
    *,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    terms = {"en": (en,), "zh": (zh,), "fr": (fr,)}
    for language, values in (aliases or {}).items():
        terms[language] = tuple(dict.fromkeys((*terms[language], *values)))
    return {
        "concept_id": concept_id,
        "category": category,
        "concept_partition": partition,
        "terms": terms,
    }


# All 80 concepts are new relative to the Phase1022 family list.  Five
# concepts per family select candidate regions; five different concepts per
# family test those frozen regions.
CONCEPTS = (
    concept("mango", "fruit", "calibration", "mango", "芒果", "mangue"),
    concept("cherry", "fruit", "calibration", "cherry", "樱桃", "cerise"),
    concept(
        "watermelon", "fruit", "calibration",
        "watermelon", "西瓜", "pastèque",
    ),
    concept(
        "pineapple", "fruit", "calibration",
        "pineapple", "菠萝", "ananas",
    ),
    concept(
        "coconut", "fruit", "calibration",
        "coconut", "椰子", "noix de coco",
    ),
    concept("kiwi", "fruit", "heldout", "kiwi", "猕猴桃", "kiwi"),
    concept("plum", "fruit", "heldout", "plum", "李子", "prune"),
    concept("apricot", "fruit", "heldout", "apricot", "杏子", "abricot"),
    concept(
        "blueberry", "fruit", "heldout",
        "blueberry", "蓝莓", "myrtille",
    ),
    concept(
        "raspberry", "fruit", "heldout",
        "raspberry", "覆盆子", "framboise",
    ),
    concept("wolf", "animal", "calibration", "wolf", "狼", "loup"),
    concept("fox", "animal", "calibration", "fox", "狐狸", "renard"),
    concept("bear", "animal", "calibration", "bear", "熊", "ours"),
    concept("deer", "animal", "calibration", "deer", "鹿", "cerf"),
    concept("monkey", "animal", "calibration", "monkey", "猴子", "singe"),
    concept("cow", "animal", "heldout", "cow", "牛", "vache"),
    concept("sheep", "animal", "heldout", "sheep", "羊", "mouton"),
    concept("pig", "animal", "heldout", "pig", "猪", "cochon"),
    concept("eagle", "animal", "heldout", "eagle", "鹰", "aigle"),
    concept("dolphin", "animal", "heldout", "dolphin", "海豚", "dauphin"),
    concept("subway", "vehicle", "calibration", "subway", "地铁", "métro"),
    concept(
        "scooter", "vehicle", "calibration",
        "scooter", "踏板车", "scooter",
    ),
    concept(
        "helicopter", "vehicle", "calibration",
        "helicopter", "直升机", "hélicoptère",
    ),
    concept("taxi", "vehicle", "calibration", "taxi", "出租车", "taxi"),
    concept(
        "van", "vehicle", "calibration",
        "van", "厢式货车", "fourgonnette",
    ),
    concept(
        "tram", "vehicle", "heldout",
        "tram", "有轨电车", "tramway",
    ),
    concept("ferry", "vehicle", "heldout", "ferry", "渡轮", "ferry"),
    concept("rocket", "vehicle", "heldout", "rocket", "火箭", "fusée"),
    concept(
        "tractor", "vehicle", "heldout",
        "tractor", "拖拉机", "tracteur",
    ),
    concept(
        "ambulance", "vehicle", "heldout",
        "ambulance", "救护车", "ambulance",
    ),
    concept(
        "scientist", "profession", "calibration",
        "scientist", "科学家", "scientifique",
    ),
    concept(
        "artist", "profession", "calibration",
        "artist", "艺术家", "artiste",
    ),
    concept(
        "writer", "profession", "calibration",
        "writer", "作家", "écrivain",
    ),
    concept("judge", "profession", "calibration", "judge", "法官", "juge"),
    concept(
        "dentist", "profession", "calibration",
        "dentist", "牙医", "dentiste",
    ),
    concept(
        "baker", "profession", "heldout",
        "baker", "面包师", "boulanger",
    ),
    concept(
        "mechanic", "profession", "heldout",
        "mechanic", "机械师", "mécanicien",
    ),
    concept(
        "firefighter", "profession", "heldout",
        "firefighter", "消防员", "pompier",
    ),
    concept(
        "architect", "profession", "heldout",
        "architect", "建筑师", "architecte",
    ),
    concept(
        "musician", "profession", "heldout",
        "musician", "音乐家", "musicien",
    ),
    concept("museum", "place", "calibration", "museum", "博物馆", "musée"),
    concept("hotel", "place", "calibration", "hotel", "酒店", "hôtel"),
    concept("bank", "place", "calibration", "bank", "银行", "banque"),
    concept("factory", "place", "calibration", "factory", "工厂", "usine"),
    concept("church", "place", "calibration", "church", "教堂", "église"),
    concept("beach", "place", "heldout", "beach", "海滩", "plage"),
    concept("farm", "place", "heldout", "farm", "农场", "ferme"),
    concept("theater", "place", "heldout", "theater", "剧院", "théâtre"),
    concept("office", "place", "heldout", "office", "办公室", "bureau"),
    concept("village", "place", "heldout", "village", "村庄", "village"),
    concept("cup", "object", "calibration", "cup", "杯子", "tasse"),
    concept("spoon", "object", "calibration", "spoon", "勺子", "cuillère"),
    concept("knife", "object", "calibration", "knife", "刀", "couteau"),
    concept("pencil", "object", "calibration", "pencil", "铅笔", "crayon"),
    concept(
        "computer", "object", "calibration",
        "computer", "电脑", "ordinateur",
    ),
    concept("camera", "object", "heldout", "camera", "相机", "appareil photo"),
    concept("umbrella", "object", "heldout", "umbrella", "雨伞", "parapluie"),
    concept("mirror", "object", "heldout", "mirror", "镜子", "miroir"),
    concept("pillow", "object", "heldout", "pillow", "枕头", "oreiller"),
    concept("suitcase", "object", "heldout", "suitcase", "行李箱", "valise"),
    concept("gray", "color", "calibration", "gray", "灰色", "gris"),
    concept("brown", "color", "calibration", "brown", "棕色", "marron"),
    concept("cyan", "color", "calibration", "cyan", "青色", "cyan"),
    concept("magenta", "color", "calibration", "magenta", "品红色", "magenta"),
    concept("gold", "color", "calibration", "gold", "金色", "doré"),
    concept("silver", "color", "heldout", "silver", "银色", "argenté"),
    concept("beige", "color", "heldout", "beige", "米色", "beige"),
    concept(
        "turquoise", "color", "heldout",
        "turquoise", "青绿色", "turquoise",
    ),
    concept("navy", "color", "heldout", "navy", "深蓝色", "bleu marine"),
    concept("indigo", "color", "heldout", "indigo", "靛蓝色", "indigo"),
    concept("arm", "body_part", "calibration", "arm", "手臂", "bras"),
    concept("leg", "body_part", "calibration", "leg", "腿", "jambe"),
    concept("finger", "body_part", "calibration", "finger", "手指", "doigt"),
    concept("toe", "body_part", "calibration", "toe", "脚趾", "orteil"),
    concept(
        "shoulder", "body_part", "calibration",
        "shoulder", "肩膀", "épaule",
    ),
    concept("knee", "body_part", "heldout", "knee", "膝盖", "genou"),
    concept("elbow", "body_part", "heldout", "elbow", "肘", "coude"),
    concept("neck", "body_part", "heldout", "neck", "脖子", "cou"),
    concept("back", "body_part", "heldout", "back", "背", "dos"),
    concept("skin", "body_part", "heldout", "skin", "皮肤", "peau"),
)
CONCEPT_BY_ID = {row["concept_id"]: row for row in CONCEPTS}


RARE_TERMS = (
    ("饕餮", "calibration", ("凶兽", "神兽", "贪食", "贪吃", "贪婪", "暴食", "gluttony", "mythical beast")),
    ("獬豸", "calibration", ("神兽", "司法", "公正", "辨别是非", "justice", "law")),
    ("貔貅", "calibration", ("神兽", "招财", "财富", "只进不出", "wealth", "mythical creature")),
    ("囹圄", "calibration", ("监狱", "牢狱", "入狱", "prison", "jail")),
    ("龃龉", "calibration", ("不合", "冲突", "意见不合", "discord", "disagreement")),
    ("赑屃", "calibration", ("龙子", "石碑", "驮碑", "神兽", "stele", "turtle")),
    ("耄耋", "calibration", ("高龄", "老人", "八九十岁", "old age", "elderly")),
    ("蹉跎", "calibration", ("虚度", "耽误", "岁月", "waste time", "delay")),
    ("觊觎", "heldout", ("企图得到", "非分之想", "垂涎", "covet", "desire")),
    ("纨绔", "heldout", ("富家子弟", "享乐", "不务正业", "dandy", "spoiled")),
    ("桎梏", "heldout", ("束缚", "枷锁", "限制", "shackle", "constraint")),
    ("狴犴", "heldout", ("龙子", "监狱", "司法", "神兽", "prison", "justice")),
    ("饔飧", "heldout", ("饭食", "早饭晚饭", "餐食", "meals", "food")),
    ("旮旯", "heldout", ("角落", "偏僻地方", "corner", "nook")),
    ("倥偬", "heldout", ("匆忙", "忙乱", "困苦", "hurry", "busy")),
    ("醍醐", "heldout", ("精华", "灌顶", "醒悟", "enlightenment", "essence")),
)

RARE_TEMPLATES = {
    "discovery": (
        "用一句简短的话解释“{term}”的核心含义：",
        "Define the Chinese term “{term}” in plain language:",
    ),
    "confirmation": (
        "“{term}”通常表示什么？请给出简明释义：",
        "Give a concise dictionary-style meaning for “{term}”:",
    ),
}

PUNCTUATION_ITEMS = (
    ("en_q1", "discovery", "Where did the train stop", ("?",)),
    ("en_q2", "discovery", "Can you open the window", ("?",)),
    ("en_e1", "discovery", "Watch out for the falling box", ("!",)),
    ("en_s1", "discovery", "The meeting begins at noon", (".",)),
    ("en_q3", "confirmation", "Why is the sky red", ("?",)),
    ("en_q4", "confirmation", "Did the doctor arrive", ("?",)),
    ("en_e2", "confirmation", "What a beautiful view", ("!",)),
    ("en_s2", "confirmation", "The library closes at six", (".",)),
    ("zh_q1", "discovery", "火车停在了哪里", ("？", "?")),
    ("zh_q2", "discovery", "你能把窗户打开吗", ("？", "?")),
    ("zh_e1", "discovery", "小心掉下来的箱子", ("！", "!")),
    ("zh_s1", "discovery", "会议中午开始", ("。", ".")),
    ("zh_q3", "confirmation", "天空为什么是红色的", ("？", "?")),
    ("zh_q4", "confirmation", "医生到了吗", ("？", "?")),
    ("zh_e2", "confirmation", "多么美丽的景色", ("！", "!")),
    ("zh_s2", "confirmation", "图书馆六点关门", ("。", ".")),
    ("fr_q1", "discovery", "Où le train s'est-il arrêté", ("?",)),
    ("fr_q2", "discovery", "Peux-tu ouvrir la fenêtre", ("?",)),
    ("fr_e1", "discovery", "Attention à la boîte qui tombe", ("!",)),
    ("fr_s1", "discovery", "La réunion commence à midi", (".",)),
    ("fr_q3", "confirmation", "Pourquoi le ciel est-il rouge", ("?",)),
    ("fr_q4", "confirmation", "Le médecin est-il arrivé", ("?",)),
    ("fr_e2", "confirmation", "Quelle vue magnifique", ("!",)),
    ("fr_s2", "confirmation", "La bibliothèque ferme à six heures", (".",)),
)

CONNECTOR_ITEMS = (
    ("en_c1", "discovery", "The fruit looked ripe. ___, it was still sour.", ("however", "yet", "nevertheless")),
    ("en_c2", "discovery", "The road was flooded. ___, the bus continued.", ("however", "yet", "nevertheless")),
    ("en_r1", "discovery", "The alarm rang. ___, everyone left the building.", ("therefore", "thus", "consequently")),
    ("en_r2", "discovery", "The temperature fell below zero. ___, the lake froze.", ("therefore", "thus", "consequently")),
    ("en_c3", "confirmation", "The task was difficult. ___, she completed it.", ("however", "yet", "nevertheless")),
    ("en_c4", "confirmation", "The room was small. ___, it felt comfortable.", ("however", "yet", "nevertheless")),
    ("en_r3", "confirmation", "The evidence was conclusive. ___, the claim was accepted.", ("therefore", "thus", "consequently")),
    ("en_r4", "confirmation", "The engine overheated. ___, the machine stopped.", ("therefore", "thus", "consequently")),
    ("zh_c1", "discovery", "水果看起来成熟了。___，它仍然很酸。", ("然而", "但是", "不过")),
    ("zh_c2", "discovery", "道路被水淹了。___，公交车继续前进。", ("然而", "但是", "不过")),
    ("zh_r1", "discovery", "警报响了。___，所有人都离开了大楼。", ("因此", "所以", "因而")),
    ("zh_r2", "discovery", "气温降到零度以下。___，湖面结冰了。", ("因此", "所以", "因而")),
    ("zh_c3", "confirmation", "任务很困难。___，她还是完成了。", ("然而", "但是", "不过")),
    ("zh_c4", "confirmation", "房间很小。___，住起来很舒服。", ("然而", "但是", "不过")),
    ("zh_r3", "confirmation", "证据很确凿。___，这个结论被接受了。", ("因此", "所以", "因而")),
    ("zh_r4", "confirmation", "发动机过热。___，机器停止了。", ("因此", "所以", "因而")),
    ("fr_c1", "discovery", "Le fruit semblait mûr. ___, il était encore acide.", ("cependant", "pourtant", "néanmoins")),
    ("fr_c2", "discovery", "La route était inondée. ___, le bus a continué.", ("cependant", "pourtant", "néanmoins")),
    ("fr_r1", "discovery", "L'alarme a sonné. ___, tout le monde est sorti.", ("donc", "ainsi", "par conséquent")),
    ("fr_r2", "discovery", "La température est tombée sous zéro. ___, le lac a gelé.", ("donc", "ainsi", "par conséquent")),
    ("fr_c3", "confirmation", "La tâche était difficile. ___, elle l'a terminée.", ("cependant", "pourtant", "néanmoins")),
    ("fr_c4", "confirmation", "La pièce était petite. ___, elle était confortable.", ("cependant", "pourtant", "néanmoins")),
    ("fr_r3", "confirmation", "La preuve était concluante. ___, l'affirmation a été acceptée.", ("donc", "ainsi", "par conséquent")),
    ("fr_r4", "confirmation", "Le moteur a surchauffé. ___, la machine s'est arrêtée.", ("donc", "ainsi", "par conséquent")),
)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", value)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def translated_outputs(row: dict[str, Any], language: str) -> list[str]:
    values = list(row["terms"][language])
    if language == "fr":
        for value in tuple(values):
            values.extend((
                f"un {value}",
                f"une {value}",
                f"le {value}",
                f"la {value}",
            ))
    return list(dict.fromkeys(values))


def surface_identity(row: dict[str, Any], source: str, target: str) -> bool:
    left = {normalize(value) for value in row["terms"][source]}
    right = {normalize(value) for value in row["terms"][target]}
    return bool(left & right)


def role_fragment(prompt: str, term: str) -> dict[str, tuple[int, int, str]]:
    start = prompt.rfind(term)
    if start < 0:
        raise RuntimeError(f"term not found in prompt: {term!r}")
    return {"source_end": (start, start + len(term), term)}


def atlas_prompt(
    row: dict[str, Any],
    *,
    prompt_split: str,
    task: str,
    source: str,
) -> tuple[str, str | None]:
    term = row["terms"][source][0]
    if task == "mention":
        template, _ = MENTION_TEMPLATES[prompt_split]
        return template.format(
            source_name=LANGUAGE_NAMES[source],
            term=term,
        ), None
    target = TARGET_BY_SPLIT[prompt_split][source]
    template, _ = TRANSLATION_TEMPLATES[prompt_split]
    return template.format(
        source_name=LANGUAGE_NAMES[source],
        target_name=LANGUAGE_NAMES[target],
        term=term,
    ), target


def build_atlas_cases() -> list[dict[str, Any]]:
    result = []
    for row in CONCEPTS:
        for prompt_split in PROMPT_SPLITS:
            for task in ATLAS_TASKS:
                for source in LANGUAGES:
                    prompt, target = atlas_prompt(
                        row,
                        prompt_split=prompt_split,
                        task=task,
                        source=source,
                    )
                    term = row["terms"][source][0]
                    result.append({
                        "schema_version": "phase1023_common_atlas_case.v1",
                        "phase": PHASE,
                        "case_key": (
                            f"atlas.{prompt_split}.{task}.{source}."
                            f"{row['concept_id']}"
                        ),
                        "prompt_split": prompt_split,
                        "task": task,
                        "source_language": source,
                        "target_language": target,
                        "concept_id": row["concept_id"],
                        "concept_partition": row["concept_partition"],
                        "category": row["category"],
                        "source_term": term,
                        "prompt": prompt,
                        "prompt_mode": "native_chat",
                        "role_fragments": role_fragment(prompt, term),
                    })
    return result


def translation_behavior_case(
    row: dict[str, Any],
    *,
    prompt_split: str,
    source: str,
    target: str,
) -> dict[str, Any]:
    term = row["terms"][source][0]
    template, _ = TRANSLATION_TEMPLATES[prompt_split]
    prompt = template.format(
        source_name=LANGUAGE_NAMES[source],
        target_name=LANGUAGE_NAMES[target],
        term=term,
    )
    return {
        "schema_version": "phase1023_common_behavior_case.v1",
        "phase": PHASE,
        "case_key": (
            f"translation.{prompt_split}.{source}_{target}."
            f"{row['concept_id']}"
        ),
        "family": "translation",
        "prompt_split": prompt_split,
        "source_language": source,
        "target_language": target,
        "concept_id": row["concept_id"],
        "concept_partition": row["concept_partition"],
        "category": row["category"],
        "source_term": term,
        "all_terms": {
            language: list(values)
            for language, values in row["terms"].items()
        },
        "accepted_outputs": translated_outputs(row, target),
        "prompt": prompt,
        "prompt_mode": "native_chat",
        "max_new_tokens": 24,
        "evaluation_type": "translation",
        "surface_identity": surface_identity(row, source, target),
        "role_fragments": role_fragment(prompt, term),
    }


def build_behavior_cases() -> list[dict[str, Any]]:
    result = []
    for row in CONCEPTS:
        for prompt_split in PROMPT_SPLITS:
            for source, target in LANGUAGE_DIRECTIONS:
                result.append(translation_behavior_case(
                    row,
                    prompt_split=prompt_split,
                    source=source,
                    target=target,
                ))
            for source in LANGUAGES:
                term = row["terms"][source][0]
                prompt = CLASSIFICATION_TEMPLATES[prompt_split].format(
                    source_name=LANGUAGE_NAMES[source],
                    term=term,
                )
                result.append({
                    "schema_version": "phase1023_common_behavior_case.v1",
                    "phase": PHASE,
                    "case_key": (
                        f"classification.{prompt_split}.{source}."
                        f"{row['concept_id']}"
                    ),
                    "family": "classification",
                    "prompt_split": prompt_split,
                    "source_language": source,
                    "target_language": "en",
                    "concept_id": row["concept_id"],
                    "concept_partition": row["concept_partition"],
                    "category": row["category"],
                    "source_term": term,
                    "all_terms": {
                        language: list(values)
                        for language, values in row["terms"].items()
                    },
                    "accepted_outputs": list(CATEGORY_OUTPUTS[row["category"]]),
                    "prompt": prompt,
                    "prompt_mode": "native_chat",
                    "max_new_tokens": 8,
                    "evaluation_type": "short",
                    "surface_identity": False,
                    "role_fragments": role_fragment(prompt, term),
                })

    for term, partition, accepted in RARE_TERMS:
        for prompt_split in PROMPT_SPLITS:
            for template_index, template in enumerate(
                RARE_TEMPLATES[prompt_split]
            ):
                prompt = template.format(term=term)
                result.append({
                    "schema_version": "phase1023_common_behavior_case.v1",
                    "phase": PHASE,
                    "case_key": (
                        f"rare_definition.{prompt_split}.{template_index}."
                        f"{term}"
                    ),
                    "family": "rare_definition",
                    "prompt_split": prompt_split,
                    "source_language": "zh",
                    "target_language": "semantic_definition",
                    "concept_id": term,
                    "concept_partition": partition,
                    "category": "rare_term",
                    "source_term": term,
                    "all_terms": {"zh": [term]},
                    "accepted_outputs": list(accepted),
                    "prompt": prompt,
                    "prompt_mode": "native_chat",
                    "max_new_tokens": 40,
                    "evaluation_type": "definition",
                    "surface_identity": False,
                    "role_fragments": {},
                })

    for item_id, prompt_split, text, accepted in PUNCTUATION_ITEMS:
        for template_index, instruction in enumerate((
            "Return only the missing final punctuation mark for this text:\n{text}",
            "Which single punctuation symbol should end the text below?\n{text}\nSymbol:",
        )):
            result.append({
                "schema_version": "phase1023_common_behavior_case.v1",
                "phase": PHASE,
                "case_key": (
                    f"punctuation.{prompt_split}.{template_index}.{item_id}"
                ),
                "family": "punctuation",
                "prompt_split": prompt_split,
                "source_language": item_id[:2],
                "target_language": "symbol",
                "concept_id": item_id,
                "concept_partition": prompt_split,
                "category": "punctuation",
                "source_term": text,
                "all_terms": {},
                "accepted_outputs": list(accepted),
                "prompt": instruction.format(text=text),
                "prompt_mode": "native_chat",
                "max_new_tokens": 4,
                "evaluation_type": "punctuation",
                "surface_identity": False,
                "role_fragments": {},
            })

    for item_id, prompt_split, text, accepted in CONNECTOR_ITEMS:
        for template_index, instruction in enumerate((
            "Replace the blank with one natural connector. Return the connector only.\n{text}",
            "Fill ___ with a suitable discourse connector:\n{text}\nConnector:",
        )):
            result.append({
                "schema_version": "phase1023_common_behavior_case.v1",
                "phase": PHASE,
                "case_key": f"connector.{prompt_split}.{template_index}.{item_id}",
                "family": "connector",
                "prompt_split": prompt_split,
                "source_language": item_id[:2],
                "target_language": item_id[:2],
                "concept_id": item_id,
                "concept_partition": prompt_split,
                "category": "connector",
                "source_term": text,
                "all_terms": {},
                "accepted_outputs": list(accepted),
                "prompt": instruction.format(text=text),
                "prompt_mode": "native_chat",
                "max_new_tokens": 10,
                "evaluation_type": "connector",
                "surface_identity": False,
                "role_fragments": {},
            })
    return result


def model_case(
    tokenizer,
    model_name: str,
    row: dict[str, Any],
    *,
    atlas: bool,
) -> dict[str, Any]:
    rendered = (
        render_chat(tokenizer, model_name, row["prompt"])
        if row["prompt_mode"] == "native_chat"
        else row["prompt"]
    )
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_positions: dict[str, int] = {}
    if row["role_fragments"]:
        spans = offset_token_spans(
            tokenizer,
            rendered,
            row["prompt"],
            row["role_fragments"],
        )
        role_positions["source_end"] = int(spans["source_end"][1])
        role_positions["pre_output"] = len(input_ids) - 1
    target_counts = [
        len(tokenizer.encode(value, add_special_tokens=False))
        for value in row.get("accepted_outputs", [])
    ]
    result = dict(row)
    result.pop("role_fragments", None)
    result.update({
        "schema_version": (
            "phase1023_model_atlas_case.v1"
            if atlas else "phase1023_model_behavior_case.v1"
        ),
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "record_id": f"{model_name}.{row['case_key']}",
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": role_positions,
        "prompt_token_count": len(input_ids),
        "source_token_count": len(
            tokenizer.encode(row["source_term"], add_special_tokens=False)
        ),
        "accepted_token_counts": target_counts,
    })
    return result


def balanced_family_pairs() -> list[dict[str, Any]]:
    by_partition_category = defaultdict(list)
    for row in CONCEPTS:
        by_partition_category[
            (row["concept_partition"], row["category"])
        ].append(row["concept_id"])
    result = []
    for partition in CONCEPT_PARTITIONS:
        for left_index, left_category in enumerate(CATEGORIES):
            for right_category in CATEGORIES[left_index + 1:]:
                left = sorted(by_partition_category[(partition, left_category)])
                right = sorted(by_partition_category[(partition, right_category)])
                if len(left) != 5 or len(right) != 5:
                    raise RuntimeError("balanced family list drift")
                for index, (left_id, right_id) in enumerate(zip(left, right)):
                    result.append({
                        "schema_version": "phase1023_balanced_family_pair.v1",
                        "phase": PHASE,
                        "concept_partition": partition,
                        "left_category": left_category,
                        "right_category": right_category,
                        "pair_index": index,
                        "left_concept_id": left_id,
                        "right_concept_id": right_id,
                    })
    return result


def audit_common(
    atlas_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    family_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    concept_counts = Counter(
        (row["category"], row["concept_partition"]) for row in CONCEPTS
    )
    atlas_counts = Counter(
        (row["prompt_split"], row["task"], row["source_language"])
        for row in atlas_rows
    )
    translation = [
        row for row in behavior_rows if row["family"] == "translation"
    ]
    behavior_family_counts = Counter(row["family"] for row in behavior_rows)
    leakage = []
    for row in behavior_rows:
        if row["surface_identity"]:
            continue
        prompt_norm = normalize(row["prompt"])
        source_norm = normalize(row["source_term"])
        prompt_without_source = prompt_norm.replace(source_norm, " ")
        for accepted in row["accepted_outputs"]:
            accepted_norm = normalize(accepted)
            if (
                len(accepted_norm) >= 2
                and accepted_norm in prompt_without_source
                and row["family"] not in ("punctuation",)
            ):
                leakage.append((row["case_key"], accepted))
    prompt_overlap = {
        task: len(
            {
                row["prompt"]
                for row in atlas_rows
                if row["task"] == task
                and row["prompt_split"] == "discovery"
            }
            & {
                row["prompt"]
                for row in atlas_rows
                if row["task"] == task
                and row["prompt_split"] == "confirmation"
            }
        )
        for task in ATLAS_TASKS
    }
    audit = {
        "concept_count": len(CONCEPTS),
        "concept_id_unique": (
            len(CONCEPTS) == len({row["concept_id"] for row in CONCEPTS})
        ),
        "concept_partition_category_counts": {
            f"{category}|{partition}": concept_counts[(category, partition)]
            for category in CATEGORIES
            for partition in CONCEPT_PARTITIONS
        },
        "atlas_case_count": len(atlas_rows),
        "atlas_case_keys_unique": (
            len(atlas_rows) == len({row["case_key"] for row in atlas_rows})
        ),
        "atlas_condition_counts": {
            "|".join(key): value for key, value in atlas_counts.items()
        },
        "behavior_case_count": len(behavior_rows),
        "behavior_case_keys_unique": (
            len(behavior_rows)
            == len({row["case_key"] for row in behavior_rows})
        ),
        "behavior_family_counts": dict(behavior_family_counts),
        "translation_direction_counts": dict(Counter(
            f"{row['prompt_split']}|{row['source_language']}_"
            f"{row['target_language']}"
            for row in translation
        )),
        "answer_leakage_count": len(leakage),
        "answer_leakage_examples": leakage[:10],
        "prompt_split_text_overlap": prompt_overlap,
        "balanced_family_pair_count": len(family_pairs),
        "balanced_pair_counts": dict(Counter(
            f"{row['concept_partition']}|{row['left_category']}|"
            f"{row['right_category']}"
            for row in family_pairs
        )),
    }
    audit["all_checks_passed"] = bool(
        audit["concept_count"] == 80
        and audit["concept_id_unique"]
        and all(value == 5 for value in concept_counts.values())
        and audit["atlas_case_count"] == 960
        and audit["atlas_case_keys_unique"]
        and all(value == 80 for value in atlas_counts.values())
        and audit["behavior_case_keys_unique"]
        and behavior_family_counts["translation"] == 960
        and behavior_family_counts["classification"] == 480
        and behavior_family_counts["rare_definition"] == 64
        and behavior_family_counts["punctuation"] == 48
        and behavior_family_counts["connector"] == 48
        and audit["answer_leakage_count"] == 0
        and all(value == 0 for value in prompt_overlap.values())
        and len(family_pairs) == 280
        and all(
            value == 5 for value in audit["balanced_pair_counts"].values()
        )
    )
    return audit


def main() -> None:
    atlas_common = build_atlas_cases()
    behavior_common = build_behavior_cases()
    family_pairs = balanced_family_pairs()
    audit = audit_common(atlas_common, behavior_common, family_pairs)
    if not audit["all_checks_passed"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))

    protocol_root = OUT_ROOT / "protocol"
    write_jsonl(protocol_root / "atlas.common.jsonl", atlas_common)
    write_jsonl(protocol_root / "behavior.common.jsonl", behavior_common)
    write_jsonl(protocol_root / "balanced_family_pairs.jsonl", family_pairs)
    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        atlas_model = [
            model_case(
                tokenizer,
                model_name,
                row,
                atlas=True,
            )
            for row in atlas_common
        ]
        behavior_model = [
            model_case(
                tokenizer,
                model_name,
                row,
                atlas=False,
            )
            for row in behavior_common
        ]
        write_jsonl(
            protocol_root / f"atlas.{model_name}.jsonl",
            atlas_model,
        )
        write_jsonl(
            protocol_root / f"behavior.{model_name}.jsonl",
            behavior_model,
        )
        model_audits[model_name] = {
            "atlas_count": len(atlas_model),
            "behavior_count": len(behavior_model),
            "atlas_role_complete": all(
                set(row["role_positions"]) == set(ATLAS_ROLES)
                for row in atlas_model
            ),
            "translation_role_complete": all(
                set(row["role_positions"]) == set(ATLAS_ROLES)
                for row in behavior_model
                if row["family"] in ("translation", "classification")
            ),
            "nonempty_inputs": all(
                row["input_ids"] for row in (*atlas_model, *behavior_model)
            ),
            "max_prompt_tokens": max(
                row["prompt_token_count"]
                for row in (*atlas_model, *behavior_model)
            ),
        }
        model_audits[model_name]["all_checks_passed"] = bool(
            model_audits[model_name]["atlas_count"] == 960
            and model_audits[model_name]["behavior_count"] == 1600
            and model_audits[model_name]["atlas_role_complete"]
            and model_audits[model_name]["translation_role_complete"]
            and model_audits[model_name]["nonempty_inputs"]
        )
        del tokenizer

    frozen_payload = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "precision": PRECISION,
        "models": MODELS,
        "concepts": CONCEPTS,
        "atlas_common": atlas_common,
        "behavior_common": behavior_common,
        "family_pairs": family_pairs,
    }
    protocol_digest = digest(frozen_payload)
    preregistration = {
        "schema_version": "phase1023_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": protocol_digest,
        "precision": "fp16",
        "quantization": "none",
        "principle": (
            "Discover repeated internal structure before proposing a "
            "mechanism equation."
        ),
        "behavior_gates": {
            "translation_semantic_two_model_accuracy": 0.70,
            "classification_two_model_accuracy": 0.70,
            "rare_definition_two_model_accuracy": 0.70,
            "punctuation_two_model_accuracy": 0.85,
            "connector_two_model_accuracy": 0.70,
        },
        "ecological_niche_gates": {
            "within_family_identity_top1": 0.35,
            "all_concept_identity_top1": 0.075,
            "family_transfer_accuracy": 0.30,
            "same_vs_shifted_cosine_margin": 0.05,
            "must_repeat_in_calibration_and_heldout": True,
        },
        "component_selection": {
            "discovery_only": True,
            "layers_per_role": 3,
            "minimum_layer_separation_fraction": 0.10,
            "mlp_discovery_top_k_per_layer_role": 64,
            "confirmation_required": True,
        },
        "ability_pairing": {
            "semantic_success_vs_true_semantic_or_language_error": True,
            "exclude_format_echo_and_truncation_errors": True,
            "exact_match_fields": [
                "prompt_split",
                "source_language",
                "target_language",
                "category",
                "source_token_count",
                "minimum_target_token_count",
                "prompt_token_count",
            ],
            "minimum_pairs_per_split": 24,
        },
        "causal_authorization": {
            "requires_output_preceding_candidate": True,
            "requires_discovery_confirmation": True,
            "requires_two_successful_models": True,
            "requires_format_length_negative_control_separation": True,
            "automatic_if_all_gates_pass": True,
        },
        "claim_limits": [
            "Centering is a control operation, not a proposed mechanism.",
            "Retrieval is observational and does not establish causality.",
            "A repeated MLP neuron response is not a word storage cell.",
            "Reuse does not establish energetic or evolutionary optimality.",
            "Cross-model physical neuron coordinates are not aligned.",
        ],
    }
    write_json(protocol_root / "preregistration.json", preregistration)
    write_json(protocol_root / "audit.common.json", audit)
    write_json(protocol_root / "audit.models.json", model_audits)
    summary = {
        "schema_version": "phase1023_protocol_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": protocol_digest,
        "precision": PRECISION,
        "concept_count": len(CONCEPTS),
        "atlas_case_count_per_model": len(atlas_common),
        "behavior_case_count_per_model": len(behavior_common),
        "balanced_family_pair_count": len(family_pairs),
        "common_audit_passed": audit["all_checks_passed"],
        "model_audits_passed": all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
    }
    write_json(protocol_root / "summary.json", summary)
    if not summary["model_audits_passed"]:
        raise RuntimeError(json.dumps(model_audits, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
