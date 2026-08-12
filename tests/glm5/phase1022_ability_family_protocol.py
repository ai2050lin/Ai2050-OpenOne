#!/usr/bin/env python3
"""Freeze the Phase1022 ability/family/timeline protocol.

This phase is descriptive before it is explanatory.  It first measures
candidate-free behavior, then freezes matched success/failure and semantic
family comparisons.  Internal measurements are not called mechanisms and no
cross-model neuron coordinates are assumed to align.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import render_chat, tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans


PHASE = 1022
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "confirmation")
LANGUAGES = ("en", "zh", "fr")
LANGUAGE_DIRECTIONS = tuple(
    (source, target)
    for source in LANGUAGES
    for target in LANGUAGES
    if source != target
)
INTERNAL_ROLES = (
    "source_end",
    "operator_end",
    "target_language_end",
    "pre_output",
    "output_1",
    "output_2",
    "output_last",
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1022_ability_family_timeline_atlas"
)

LANGUAGE_NAMES = {
    "en": {"en": "English", "zh": "英语"},
    "zh": {"en": "Chinese", "zh": "中文"},
    "fr": {"en": "French", "zh": "法语"},
}

CATEGORY_LABELS = {
    "fruit": {
        "en": ("fruit",),
        "zh": ("水果", "果物"),
        "fr": ("fruit",),
    },
    "animal": {
        "en": ("animal",),
        "zh": ("动物",),
        "fr": ("animal",),
    },
    "vehicle": {
        "en": ("vehicle", "transport"),
        "zh": ("交通工具", "车辆"),
        "fr": ("véhicule", "moyen de transport"),
    },
    "profession": {
        "en": ("profession", "occupation", "job"),
        "zh": ("职业",),
        "fr": ("profession", "métier"),
    },
    "place": {
        "en": ("place", "location"),
        "zh": ("场所", "地点"),
        "fr": ("lieu", "endroit"),
    },
    "object": {
        "en": ("object", "item"),
        "zh": ("物品", "物体"),
        "fr": ("objet",),
    },
    "color": {
        "en": ("color", "colour"),
        "zh": ("颜色", "色彩"),
        "fr": ("couleur",),
    },
    "body_part": {
        "en": ("body part", "part of the body"),
        "zh": ("身体部位", "人体部位"),
        "fr": ("partie du corps",),
    },
}


def concept(
    concept_id: str,
    category: str,
    split: str,
    en: str,
    zh: str,
    fr: str,
    *,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    values = {"en": (en,), "zh": (zh,), "fr": (fr,)}
    for language, extra in (aliases or {}).items():
        values[language] = tuple(dict.fromkeys((*values[language], *extra)))
    return {
        "concept_id": concept_id,
        "category": category,
        "split": split,
        "terms": values,
    }


# Four disjoint concepts per family are used for discovery and four for
# confirmation.  Family replication therefore cannot be caused by reusing
# the same lexical item in both splits.
CONCEPTS = (
    concept("apple", "fruit", "discovery", "apple", "苹果", "pomme"),
    concept("banana", "fruit", "discovery", "banana", "香蕉", "banane"),
    concept("orange", "fruit", "discovery", "orange", "橙子", "orange"),
    concept("grape", "fruit", "discovery", "grape", "葡萄", "raisin"),
    concept("lemon", "fruit", "confirmation", "lemon", "柠檬", "citron"),
    concept("peach", "fruit", "confirmation", "peach", "桃子", "pêche"),
    concept("pear", "fruit", "confirmation", "pear", "梨", "poire"),
    concept(
        "strawberry",
        "fruit",
        "confirmation",
        "strawberry",
        "草莓",
        "fraise",
    ),
    concept("cat", "animal", "discovery", "cat", "猫", "chat"),
    concept("dog", "animal", "discovery", "dog", "狗", "chien"),
    concept("horse", "animal", "discovery", "horse", "马", "cheval"),
    concept("rabbit", "animal", "discovery", "rabbit", "兔子", "lapin"),
    concept("tiger", "animal", "confirmation", "tiger", "老虎", "tigre"),
    concept("lion", "animal", "confirmation", "lion", "狮子", "lion"),
    concept(
        "elephant",
        "animal",
        "confirmation",
        "elephant",
        "大象",
        "éléphant",
    ),
    concept("panda", "animal", "confirmation", "panda", "熊猫", "panda"),
    concept("car", "vehicle", "discovery", "car", "汽车", "voiture"),
    concept("train", "vehicle", "discovery", "train", "火车", "train"),
    concept(
        "bicycle",
        "vehicle",
        "discovery",
        "bicycle",
        "自行车",
        "vélo",
        aliases={"zh": ("单车",), "fr": ("bicyclette",)},
    ),
    concept(
        "airplane",
        "vehicle",
        "discovery",
        "airplane",
        "飞机",
        "avion",
        aliases={"en": ("plane",)},
    ),
    concept("boat", "vehicle", "confirmation", "boat", "船", "bateau"),
    concept("bus", "vehicle", "confirmation", "bus", "公交车", "bus"),
    concept("truck", "vehicle", "confirmation", "truck", "卡车", "camion"),
    concept(
        "motorcycle",
        "vehicle",
        "confirmation",
        "motorcycle",
        "摩托车",
        "moto",
        aliases={"fr": ("motocyclette",)},
    ),
    concept(
        "teacher",
        "profession",
        "discovery",
        "teacher",
        "教师",
        "professeur",
        aliases={"zh": ("老师",)},
    ),
    concept("doctor", "profession", "discovery", "doctor", "医生", "médecin"),
    concept(
        "farmer",
        "profession",
        "discovery",
        "farmer",
        "农民",
        "agriculteur",
    ),
    concept("chef", "profession", "discovery", "chef", "厨师", "cuisinier"),
    concept("pilot", "profession", "confirmation", "pilot", "飞行员", "pilote"),
    concept(
        "nurse",
        "profession",
        "confirmation",
        "nurse",
        "护士",
        "infirmier",
        aliases={"fr": ("infirmière",)},
    ),
    concept(
        "engineer",
        "profession",
        "confirmation",
        "engineer",
        "工程师",
        "ingénieur",
    ),
    concept(
        "lawyer",
        "profession",
        "confirmation",
        "lawyer",
        "律师",
        "avocat",
        aliases={"fr": ("avocate",)},
    ),
    concept("school", "place", "discovery", "school", "学校", "école"),
    concept("hospital", "place", "discovery", "hospital", "医院", "hôpital"),
    concept(
        "library",
        "place",
        "discovery",
        "library",
        "图书馆",
        "bibliothèque",
    ),
    concept("station", "place", "discovery", "station", "车站", "gare"),
    concept("market", "place", "confirmation", "market", "市场", "marché"),
    concept("airport", "place", "confirmation", "airport", "机场", "aéroport"),
    concept(
        "restaurant",
        "place",
        "confirmation",
        "restaurant",
        "餐厅",
        "restaurant",
    ),
    concept("park", "place", "confirmation", "park", "公园", "parc"),
    concept("book", "object", "discovery", "book", "书", "livre"),
    concept("chair", "object", "discovery", "chair", "椅子", "chaise"),
    concept("table", "object", "discovery", "table", "桌子", "table"),
    concept("key", "object", "discovery", "key", "钥匙", "clé"),
    concept("lamp", "object", "confirmation", "lamp", "灯", "lampe"),
    concept(
        "phone",
        "object",
        "confirmation",
        "phone",
        "电话",
        "téléphone",
        aliases={"en": ("telephone",), "zh": ("手机",)},
    ),
    concept("clock", "object", "confirmation", "clock", "钟", "horloge"),
    concept("bottle", "object", "confirmation", "bottle", "瓶子", "bouteille"),
    concept("red", "color", "discovery", "red", "红色", "rouge"),
    concept("blue", "color", "discovery", "blue", "蓝色", "bleu"),
    concept("green", "color", "discovery", "green", "绿色", "vert"),
    concept("yellow", "color", "discovery", "yellow", "黄色", "jaune"),
    concept("black", "color", "confirmation", "black", "黑色", "noir"),
    concept("white", "color", "confirmation", "white", "白色", "blanc"),
    concept("purple", "color", "confirmation", "purple", "紫色", "violet"),
    concept("pink", "color", "confirmation", "pink", "粉色", "rose"),
    concept("head", "body_part", "discovery", "head", "头", "tête"),
    concept("hand", "body_part", "discovery", "hand", "手", "main"),
    concept("eye", "body_part", "discovery", "eye", "眼睛", "œil"),
    concept("ear", "body_part", "discovery", "ear", "耳朵", "oreille"),
    concept("nose", "body_part", "confirmation", "nose", "鼻子", "nez"),
    concept("mouth", "body_part", "confirmation", "mouth", "嘴", "bouche"),
    concept("foot", "body_part", "confirmation", "foot", "脚", "pied"),
    concept("heart", "body_part", "confirmation", "heart", "心脏", "cœur"),
)
CONCEPT_BY_ID = {row["concept_id"]: row for row in CONCEPTS}
CATEGORIES = tuple(CATEGORY_LABELS)

TRANSLATION_TEMPLATES = {
    "discovery": (
        (
            "Translate this {source_en} word into {target_en}. "
            "Return only the translated word.\n"
            "Source word: {term}\nTranslation:",
            "Translate",
            "{target_en}",
        ),
        (
            "请把下面的{source_zh}词翻译成{target_zh}。只输出译词。\n"
            "原词：{term}\n译词：",
            "翻译",
            "{target_zh}",
        ),
        (
            "Convert the following term from {source_en} to {target_en}. "
            "Give one target-language term only.\n"
            "Term: {term}\nResult:",
            "Convert",
            "{target_en}",
        ),
        (
            "执行词语翻译：{source_zh}转为{target_zh}。不要解释。\n"
            "待翻译词：{term}\n答案：",
            "翻译",
            "{target_zh}",
        ),
    ),
    "confirmation": (
        (
            "Give the {target_en} equivalent of this {source_en} word. "
            "Output the equivalent only.\n"
            "Word: {term}\nAnswer:",
            "equivalent",
            "{target_en}",
        ),
        (
            "将这个{source_zh}词转换为对应的{target_zh}词，只写结果。\n"
            "输入：{term}\n输出：",
            "转换",
            "{target_zh}",
        ),
        (
            "Render this single {source_en} lexical item in {target_en}; "
            "do not add an explanation.\n"
            "Item: {term}\nRendering:",
            "Render",
            "{target_en}",
        ),
        (
            "请给出下列{source_zh}词在{target_zh}中的对应词。\n"
            "词语：{term}\n对应词：",
            "对应词",
            "{target_zh}",
        ),
    ),
}

CLASSIFICATION_TEMPLATES = {
    "discovery": (
        (
            "Name the broad semantic category of this {source_en} word. "
            "Answer in {target_en} with one category term.\n"
            "Word: {term}\nCategory:",
            "category",
            "{target_en}",
        ),
        (
            "判断这个{source_zh}词所属的宽泛语义类别。"
            "用{target_zh}输出一个类别词。\n"
            "词语：{term}\n类别：",
            "类别",
            "{target_zh}",
        ),
    ),
    "confirmation": (
        (
            "Classify this {source_en} word by its general meaning. "
            "Return one {target_en} class label.\n"
            "Word: {term}\nClass:",
            "Classify",
            "{target_en}",
        ),
        (
            "给下面的{source_zh}词标注一般语义类别，"
            "只用{target_zh}写一个类别名称。\n"
            "输入：{term}\n标签：",
            "语义类别",
            "{target_zh}",
        ),
    ),
}

RARE_ITEMS = (
    ("饕餮", "discovery", ("贪食", "贪吃", "暴食", "贪欲", "greed for food", "gluttony")),
    ("獬豸", "discovery", ("公正", "司法", "断案", "辨曲直", "justice", "law")),
    ("貔貅", "discovery", ("招财", "辟邪", "瑞兽", "wealth", "mythical beast")),
    ("麒麟", "discovery", ("祥瑞", "仁德", "瑞兽", "auspicious", "mythical beast")),
    ("赑屃", "discovery", ("石碑", "驮碑", "龟形", "stele", "turtle-like")),
    ("螭吻", "discovery", ("屋脊", "防火", "建筑装饰", "roof", "ornament")),
    ("圭臬", "discovery", ("准则", "标准", "规范", "criterion", "standard")),
    ("鸿鹄", "discovery", ("志向", "远志", "天鹅", "ambition", "swan")),
    ("桎梏", "discovery", ("束缚", "限制", "枷锁", "constraint", "shackle")),
    ("樊笼", "discovery", ("牢笼", "困境", "束缚", "cage", "confinement")),
    ("扶桑", "discovery", ("神树", "东方", "日本", "mythic tree", "Japan")),
    ("青鸟", "discovery", ("信使", "传信", "使者", "messenger", "blue bird")),
    ("耄耋", "discovery", ("高龄", "老人", "老年", "advanced age", "elderly")),
    ("桑梓", "discovery", ("故乡", "家乡", "故土", "hometown", "homeland")),
    ("蒹葭", "discovery", ("芦苇", "思念", "reed", "longing")),
    ("社稷", "discovery", ("国家", "江山", "政权", "state", "country")),
    ("甪端", "confirmation", ("祥瑞", "神兽", "明辨是非", "auspicious beast")),
    ("狻猊", "confirmation", ("狮子", "神兽", "香炉", "lion", "mythical beast")),
    ("夔牛", "confirmation", ("雷声", "独脚", "神兽", "thunder", "one-legged beast")),
    ("精卫", "confirmation", ("填海", "神鸟", "坚持", "fills the sea", "mythic bird")),
    ("鲲鹏", "confirmation", ("巨鸟", "远志", "巨大", "giant bird", "vast ambition")),
    ("蓬莱", "confirmation", ("仙山", "仙境", "仙岛", "immortal isle", "paradise")),
    ("不啻", "confirmation", ("如同", "不只", "无异于", "no less than", "as if")),
    ("踟蹰", "confirmation", ("犹豫", "徘徊", "迟疑", "hesitate", "linger")),
    ("龃龉", "confirmation", ("冲突", "不合", "矛盾", "conflict", "discord")),
    ("囹圄", "confirmation", ("监狱", "牢狱", "入狱", "prison", "jail")),
    ("饔飧", "confirmation", ("饭食", "饮食", "三餐", "meals", "food")),
    ("觥筹", "confirmation", ("酒宴", "饮酒", "酒杯", "banquet", "drinking")),
    ("罅隙", "confirmation", ("缝隙", "裂缝", "空隙", "gap", "crack")),
    ("倥偬", "confirmation", ("忙乱", "匆忙", "急迫", "hurried", "busy")),
    ("葳蕤", "confirmation", ("茂盛", "繁盛", "枝叶繁茂", "lush", "flourishing")),
    ("踽踽", "confirmation", ("孤独", "独行", "孤零零", "alone", "solitary")),
)

RARE_TEMPLATES = {
    "discovery": (
        "用一个简短词语解释“{term}”的常见含义。只写释义：",
        "“{term}”大致是什么意思？请给出一个短释义：",
        "Define the Chinese term “{term}” briefly. Give only its core meaning:",
        "请用最短的同义表达说明“{term}”：",
    ),
    "confirmation": (
        "请给出“{term}”最常见的一种简明释义：",
        "这个古汉语词“{term}”通常指什么？短答：",
        "State one accepted meaning of the Chinese term “{term}” in a short phrase:",
        "不用造句，只解释“{term}”的核心意思：",
    ),
}

# These contexts end immediately before a natural next punctuation mark.
# They are auxiliary behavior probes and are not authorized for an internal
# mechanism scan unless at least two models pass the preregistered gate.
PUNCTUATION_CONTEXTS = (
    ("en_statement_1", "discovery", "The library closes at six", (".",)),
    ("en_question_1", "discovery", "Does the library close at six", ("?",)),
    ("en_exclaim_1", "discovery", "What a remarkable discovery", ("!",)),
    ("en_statement_2", "discovery", "The train has already left", (".",)),
    ("en_question_2", "discovery", "Has the train already left", ("?",)),
    ("en_exclaim_2", "discovery", "How quickly the weather changed", ("!",)),
    ("zh_statement_1", "discovery", "图书馆六点关门", ("。",)),
    ("zh_question_1", "discovery", "图书馆是六点关门吗", ("？", "?")),
    ("zh_exclaim_1", "discovery", "这真是一个惊人的发现", ("！", "!")),
    ("zh_statement_2", "discovery", "火车已经离站", ("。",)),
    ("zh_question_2", "discovery", "火车已经离站了吗", ("？", "?")),
    ("zh_exclaim_2", "discovery", "天气变化得真快", ("！", "!")),
    ("fr_statement_1", "confirmation", "La bibliothèque ferme à six heures", (".",)),
    ("fr_question_1", "confirmation", "La bibliothèque ferme-t-elle à six heures", ("?",)),
    ("fr_exclaim_1", "confirmation", "Quelle découverte remarquable", ("!",)),
    ("fr_statement_2", "confirmation", "Le train est déjà parti", (".",)),
    ("fr_question_2", "confirmation", "Le train est-il déjà parti", ("?",)),
    ("fr_exclaim_2", "confirmation", "Comme le temps a changé vite", ("!",)),
    ("en_statement_3", "confirmation", "The experiment finished on time", (".",)),
    ("en_question_3", "confirmation", "Did the experiment finish on time", ("?",)),
    ("zh_statement_3", "confirmation", "实验按时完成", ("。",)),
    ("zh_question_3", "confirmation", "实验按时完成了吗", ("？", "?")),
    ("fr_statement_3", "confirmation", "L'expérience s'est terminée à temps", (".",)),
    ("fr_question_3", "confirmation", "L'expérience s'est-elle terminée à temps", ("?",)),
)

CONNECTOR_ITEMS = (
    ("en_contrast_1", "discovery", "The road was flooded. ", ("However", "Nevertheless", "Yet")),
    ("en_cause_1", "discovery", "The road was flooded. ", ("Therefore", "Consequently", "Thus")),
    ("en_contrast_2", "discovery", "The device is expensive. ", ("However", "Nevertheless", "Yet")),
    ("en_cause_2", "discovery", "The device overheated. ", ("Therefore", "Consequently", "Thus")),
    ("zh_contrast_1", "discovery", "道路被洪水淹没。", ("然而", "但是", "不过")),
    ("zh_cause_1", "discovery", "道路被洪水淹没。", ("因此", "所以", "因而")),
    ("zh_contrast_2", "discovery", "这个设备价格昂贵。", ("然而", "但是", "不过")),
    ("zh_cause_2", "discovery", "这个设备过热。", ("因此", "所以", "因而")),
    ("fr_contrast_1", "confirmation", "La route était inondée. ", ("Cependant", "Pourtant", "Néanmoins")),
    ("fr_cause_1", "confirmation", "La route était inondée. ", ("Donc", "Ainsi", "Par conséquent")),
    ("fr_contrast_2", "confirmation", "Cet appareil est cher. ", ("Cependant", "Pourtant", "Néanmoins")),
    ("fr_cause_2", "confirmation", "Cet appareil a surchauffé. ", ("Donc", "Ainsi", "Par conséquent")),
    ("en_contrast_3", "confirmation", "The evidence is incomplete. ", ("However", "Nevertheless", "Yet")),
    ("en_cause_3", "confirmation", "The evidence is incomplete. ", ("Therefore", "Consequently", "Thus")),
    ("zh_contrast_3", "confirmation", "证据还不完整。", ("然而", "但是", "不过")),
    ("zh_cause_3", "confirmation", "证据还不完整。", ("因此", "所以", "因而")),
)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )
    value = re.sub(r"\s+", " ", value)
    return value


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


def render_template(
    template: tuple[str, str, str],
    *,
    source: str,
    target: str,
    term: str,
) -> tuple[str, str, str]:
    values = {
        "source_en": LANGUAGE_NAMES[source]["en"],
        "source_zh": LANGUAGE_NAMES[source]["zh"],
        "target_en": LANGUAGE_NAMES[target]["en"],
        "target_zh": LANGUAGE_NAMES[target]["zh"],
        "term": term,
    }
    prompt = template[0].format(**values)
    operator = template[1]
    target_text = template[2].format(**values)
    return prompt, operator, target_text


def char_span(prompt: str, text: str, *, last: bool = False) -> tuple[int, int, str]:
    start = prompt.rfind(text) if last else prompt.find(text)
    if start < 0:
        raise RuntimeError(f"cannot locate {text!r} in prompt")
    return start, start + len(text), text


def translation_outputs(row: dict[str, Any], target: str) -> list[str]:
    values = list(row["terms"][target])
    if target == "fr":
        for term in tuple(values):
            values.extend((f"un {term}", f"une {term}", f"le {term}", f"la {term}"))
    return list(dict.fromkeys(values))


def build_common_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in CONCEPTS:
        split = row["split"]
        for source, target in LANGUAGE_DIRECTIONS:
            source_term = row["terms"][source][0]
            target_values = translation_outputs(row, target)
            for template_index, template in enumerate(
                TRANSLATION_TEMPLATES[split]
            ):
                prompt, operator, target_text = render_template(
                    template,
                    source=source,
                    target=target,
                    term=source_term,
                )
                case_key = (
                    f"translation.{split}.{source}_{target}."
                    f"t{template_index}.{row['concept_id']}"
                )
                rows.append({
                    "case_key": case_key,
                    "family": "translation",
                    "task": "translate_word",
                    "split": split,
                    "template": template_index,
                    "concept_id": row["concept_id"],
                    "category": row["category"],
                    "source_language": source,
                    "target_language": target,
                    "source_term": source_term,
                    "accepted_outputs": target_values,
                    "prompt": prompt,
                    "prompt_mode": "native_chat",
                    "max_new_tokens": 8,
                    "evaluation_type": "short_semantic",
                    "role_fragments": {
                        "source_end": char_span(prompt, source_term, last=True),
                        "operator_end": char_span(prompt, operator),
                        "target_language_end": char_span(prompt, target_text),
                    },
                    "surface_identity": bool(
                        normalize(source_term)
                        in {normalize(value) for value in target_values}
                    ),
                })
            for template_index, template in enumerate(
                CLASSIFICATION_TEMPLATES[split]
            ):
                prompt, operator, target_text = render_template(
                    template,
                    source=source,
                    target=target,
                    term=source_term,
                )
                case_key = (
                    f"classification.{split}.{source}_{target}."
                    f"t{template_index}.{row['concept_id']}"
                )
                rows.append({
                    "case_key": case_key,
                    "family": "classification",
                    "task": "classify_word",
                    "split": split,
                    "template": template_index,
                    "concept_id": row["concept_id"],
                    "category": row["category"],
                    "source_language": source,
                    "target_language": target,
                    "source_term": source_term,
                    "accepted_outputs": list(CATEGORY_LABELS[row["category"]][target]),
                    "prompt": prompt,
                    "prompt_mode": "native_chat",
                    "max_new_tokens": 10,
                    "evaluation_type": "short_semantic",
                    "role_fragments": {
                        "source_end": char_span(prompt, source_term, last=True),
                        "operator_end": char_span(prompt, operator),
                        "target_language_end": char_span(prompt, target_text),
                    },
                    "surface_identity": False,
                })

    for term, split, accepted in RARE_ITEMS:
        for template_index, template in enumerate(RARE_TEMPLATES[split]):
            prompt = template.format(term=term)
            rows.append({
                "case_key": f"rare_definition.{split}.t{template_index}.{term}",
                "family": "rare_definition",
                "task": "define_rare_word",
                "split": split,
                "template": template_index,
                "concept_id": term,
                "category": "rare_word",
                "source_language": "zh",
                "target_language": "mixed",
                "source_term": term,
                "accepted_outputs": list(accepted),
                "prompt": prompt,
                "prompt_mode": "native_chat",
                "max_new_tokens": 24,
                "evaluation_type": "definition",
                "role_fragments": {},
                "surface_identity": False,
            })

    for item_id, split, context, accepted in PUNCTUATION_CONTEXTS:
        rows.append({
            "case_key": f"punctuation.{split}.{item_id}",
            "family": "punctuation",
            "task": "natural_next_punctuation",
            "split": split,
            "template": 0,
            "concept_id": item_id,
            "category": "punctuation",
            "source_language": item_id[:2],
            "target_language": item_id[:2],
            "source_term": context,
            "accepted_outputs": list(accepted),
            "prompt": context,
            "prompt_mode": "raw",
            "max_new_tokens": 2,
            "evaluation_type": "punctuation",
            "role_fragments": {},
            "surface_identity": False,
        })

    for item_id, split, context, accepted in CONNECTOR_ITEMS:
        relation = "contrast" if "contrast" in item_id else "cause"
        instruction = (
            f"{context}\nContinue with one natural {relation} connector only:"
            if item_id.startswith("en")
            else (
                f"{context}\n只写一个自然的"
                f"{'转折' if relation == 'contrast' else '因果'}连接词："
            )
        )
        rows.append({
            "case_key": f"connector.{split}.{item_id}",
            "family": "connector",
            "task": f"{relation}_connector",
            "split": split,
            "template": 0,
            "concept_id": item_id,
            "category": relation,
            "source_language": item_id[:2],
            "target_language": item_id[:2],
            "source_term": context,
            "accepted_outputs": list(accepted),
            "prompt": instruction,
            "prompt_mode": "native_chat",
            "max_new_tokens": 8,
            "evaluation_type": "connector",
            "role_fragments": {},
            "surface_identity": False,
        })
    return rows


def model_case(tokenizer, model_name: str, row: dict[str, Any]) -> dict[str, Any]:
    prompt = row["prompt"]
    rendered = (
        render_chat(tokenizer, model_name, prompt)
        if row["prompt_mode"] == "native_chat"
        else prompt
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
            prompt,
            row["role_fragments"],
        )
        role_positions = {
            role: int(bounds[1]) for role, bounds in spans.items()
        }
        role_positions["pre_output"] = len(input_ids) - 1
    result = dict(row)
    result.pop("role_fragments", None)
    result.update({
        "schema_version": "phase1022_protocol_case.v1",
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
        "accepted_token_counts": [
            len(tokenizer.encode(value, add_special_tokens=False))
            for value in row["accepted_outputs"]
        ],
    })
    return result


def audit_common(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [row["case_key"] for row in rows]
    split_concepts = {
        split: {
            row["concept_id"]
            for row in rows
            if row["family"] == "translation" and row["split"] == split
        }
        for split in SPLITS
    }
    translation = [row for row in rows if row["family"] == "translation"]
    classification = [row for row in rows if row["family"] == "classification"]
    leakage = []
    for row in rows:
        if row["surface_identity"]:
            continue
        prompt_norm = normalize(row["prompt"])
        source_norm = normalize(row["source_term"])
        prompt_without_source = prompt_norm.replace(source_norm, " <source> ")
        for accepted in row["accepted_outputs"]:
            accepted_norm = normalize(accepted)
            if not accepted_norm:
                continue
            if re.fullmatch(r"[a-z0-9 ]+", accepted_norm):
                leaked = bool(re.search(
                    rf"(?<![a-z0-9]){re.escape(accepted_norm)}(?![a-z0-9])",
                    prompt_without_source,
                ))
            else:
                leaked = accepted_norm in prompt_without_source
            if leaked:
                leakage.append((row["case_key"], accepted))
    return {
        "unique_case_keys": len(keys) == len(set(keys)),
        "discovery_confirmation_concept_overlap": sorted(
            split_concepts["discovery"] & split_concepts["confirmation"]
        ),
        "translation_case_count": len(translation),
        "classification_case_count": len(classification),
        "translation_by_split": dict(Counter(
            row["split"] for row in translation
        )),
        "translation_by_direction": dict(Counter(
            f"{row['source_language']}_{row['target_language']}"
            for row in translation
        )),
        "translation_by_category": dict(Counter(
            row["category"] for row in translation
        )),
        "nonidentity_answer_leakage": leakage,
        "all_checks_passed": bool(
            len(keys) == len(set(keys))
            and not (
                split_concepts["discovery"]
                & split_concepts["confirmation"]
            )
            and not leakage
        ),
    }


def main() -> None:
    common = build_common_cases()
    audit = audit_common(common)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"protocol audit failed: {audit}")

    protocol_identity = [{
        key: row[key]
        for key in (
            "case_key",
            "family",
            "task",
            "split",
            "template",
            "concept_id",
            "category",
            "source_language",
            "target_language",
            "source_term",
            "accepted_outputs",
            "prompt",
            "prompt_mode",
            "max_new_tokens",
            "evaluation_type",
            "surface_identity",
        )
    } for row in common]
    protocol_digest = digest({
        "phase": PHASE,
        "revision": PROTOCOL_REVISION,
        "cases": protocol_identity,
        "internal_roles": INTERNAL_ROLES,
    })
    protocol_root = OUT_ROOT / "protocol"
    write_jsonl(protocol_root / "common_cases.jsonl", common)
    write_json(protocol_root / "audit.json", audit)
    write_json(
        protocol_root / "preregistration.json",
        {
            "schema_version": "phase1022_preregistration.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": protocol_digest,
            "principle": (
                "Discover stable repeated structures first; treat formulas "
                "as measurement definitions until later causal evidence."
            ),
            "behavior_gates": {
                "translation_two_model_accuracy": 0.70,
                "classification_two_model_accuracy": 0.70,
                "rare_two_model_accuracy": 0.70,
                "punctuation_two_model_accuracy": 0.85,
                "connector_two_model_accuracy": 0.70,
            },
            "internal_authorization": {
                "translation": (
                    "At least two models pass behavior and each split has "
                    "at least 24 matched success/failure pairs."
                ),
                "classification": "At least two models pass behavior.",
                "rare_definition": "At least two models pass behavior.",
                "punctuation": "At least two models pass behavior.",
                "connector": "At least two models pass behavior.",
            },
            "family_scan": {
                "directions": ("en_zh", "zh_en", "fr_en"),
                "comparison": (
                    "Within-model family-minus-matched-other-family "
                    "differences, independently repeated on disjoint concepts."
                ),
                "shuffle_control": (
                    "Category labels are cyclically permuted within direction "
                    "and source-token-length strata."
                ),
            },
            "ability_scan": {
                "pairing": (
                    "Within model, match correct and incorrect translations "
                    "by split, direction, template, category, source token "
                    "count, and generated token count before internal capture."
                ),
                "controls": ("success_success", "failure_failure"),
                "cross_model_rule": (
                    "Compare normalized relative-depth profiles only; never "
                    "subtract hidden vectors or align neuron IDs across models."
                ),
                "timeline_rule": (
                    "pre_output can precede output; output_1/output_2/output_last "
                    "contain generated-token consequences."
                ),
            },
            "theory_status": {
                "relative_encoding": "working_hypothesis",
                "reuse_efficiency_tradeoff": "working_hypothesis",
                "near_optimality": "unproven_hypothesis",
                "small_model_roughness": "plausible_limit_not_explanation",
            },
            "automatic_continuation": (
                "Run a targeted neuron/subspace confirmation only if a "
                "pre-output ability-correlated event repeats in discovery and "
                "confirmation, exceeds both matched controls, and separates "
                "from generic prompt/target-language responses."
            ),
        },
    )

    model_summaries = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = [model_case(tokenizer, model_name, row) for row in common]
        model_audit = {
            "model": model_name,
            "case_count": len(cases),
            "unique_record_ids": (
                len(cases) == len({row["record_id"] for row in cases})
            ),
            "translation_roles_complete": all(
                set(row["role_positions"]) == {
                    "source_end",
                    "operator_end",
                    "target_language_end",
                    "pre_output",
                }
                for row in cases
                if row["family"] in ("translation", "classification")
            ),
            "minimum_prompt_tokens": min(
                row["prompt_token_count"] for row in cases
            ),
            "maximum_prompt_tokens": max(
                row["prompt_token_count"] for row in cases
            ),
        }
        model_audit["all_checks_passed"] = bool(
            model_audit["unique_record_ids"]
            and model_audit["translation_roles_complete"]
        )
        if not model_audit["all_checks_passed"]:
            raise RuntimeError(f"{model_name} protocol audit failed")
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", model_audit)
        model_summaries[model_name] = model_audit
        del tokenizer

    summary = {
        "schema_version": "phase1022_protocol_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": protocol_digest,
        "common_case_count": len(common),
        "family_counts": dict(Counter(row["family"] for row in common)),
        "split_counts": dict(Counter(row["split"] for row in common)),
        "category_count": len(CATEGORIES),
        "concept_count": len(CONCEPTS),
        "models": model_summaries,
        "audit": audit,
    }
    write_json(protocol_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
