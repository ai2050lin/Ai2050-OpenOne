#!/usr/bin/env python3
"""Freeze Phase1021 candidate-free natural language operation protocol.

The protocol observes behavior before internal structure.  It uses only
semantically valid contrasts:

* translate versus classify at a fixed target language;
* target-language changes at a fixed operation;
* concept changes at fixed operation and target language;
* copy versus translate as a separate, intentionally coupled control;
* irrelevant instruction wording with an unchanged answer.

No output candidates are displayed in any prompt.
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

from phase1018_language_pattern_protocol import (
    SpanBuilder,
    continuation_token_ids,
    render_chat,
    tokenizer_for,
)


PHASE = 1021
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PROMPT_MODES = ("raw", "native_chat")
FAMILIES = (
    "multilingual_operation",
    "rare_definition",
    "punctuation_next",
    "contrast_relation",
)
SPLITS = ("discovery", "confirmation")
FACTORIAL_STATES = ("b0_l0", "b1_l0", "b0_l1", "b1_l1")
STATES = FACTORIAL_STATES + ("identity",)
CAPTURE_ROLES = (
    "prefix_anchor",
    "carrier_start",
    "carrier_end",
    "context_anchor",
    "operator",
    "query_anchor",
    "answer_boundary",
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1021_natural_language_operation_atlas"
)

LANGUAGE_NAMES = {
    "en": ("English", "英语"),
    "zh": ("Chinese", "中文"),
    "fr": ("French", "法语"),
}
LANGUAGES = tuple(LANGUAGE_NAMES)
LANGUAGE_DIRECTIONS = tuple(
    (source, target)
    for source in LANGUAGES
    for target in LANGUAGES
    if source != target
)

CATEGORY_TERMS = {
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
        "en": ("profession", "occupation"),
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
}


def concept(
    concept_id: str,
    category: str,
    en: str,
    zh: str,
    fr: str,
    *,
    en_article: str,
    fr_article: str,
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "category": category,
        "terms": {"en": en, "zh": zh, "fr": fr},
        "articles": {"en": en_article, "fr": fr_article},
    }


CONCEPTS = (
    concept("apple", "fruit", "apple", "苹果", "pomme", en_article="an", fr_article="une"),
    concept("banana", "fruit", "banana", "香蕉", "banane", en_article="a", fr_article="une"),
    concept("orange", "fruit", "orange", "橙子", "orange", en_article="an", fr_article="une"),
    concept("grape", "fruit", "grape", "葡萄", "raisin", en_article="a", fr_article="un"),
    concept("lemon", "fruit", "lemon", "柠檬", "citron", en_article="a", fr_article="un"),
    concept("cat", "animal", "cat", "猫", "chat", en_article="a", fr_article="un"),
    concept("dog", "animal", "dog", "狗", "chien", en_article="a", fr_article="un"),
    concept("horse", "animal", "horse", "马", "cheval", en_article="a", fr_article="un"),
    concept("rabbit", "animal", "rabbit", "兔子", "lapin", en_article="a", fr_article="un"),
    concept("tiger", "animal", "tiger", "老虎", "tigre", en_article="a", fr_article="un"),
    concept("car", "vehicle", "car", "汽车", "voiture", en_article="a", fr_article="une"),
    concept("train", "vehicle", "train", "火车", "train", en_article="a", fr_article="un"),
    concept("bicycle", "vehicle", "bicycle", "自行车", "vélo", en_article="a", fr_article="un"),
    concept("airplane", "vehicle", "airplane", "飞机", "avion", en_article="an", fr_article="un"),
    concept("boat", "vehicle", "boat", "船", "bateau", en_article="a", fr_article="un"),
    concept("teacher", "profession", "teacher", "教师", "professeur", en_article="a", fr_article="un"),
    concept("doctor", "profession", "doctor", "医生", "médecin", en_article="a", fr_article="un"),
    concept("farmer", "profession", "farmer", "农民", "agriculteur", en_article="a", fr_article="un"),
    concept("chef", "profession", "chef", "厨师", "cuisinier", en_article="a", fr_article="un"),
    concept("pilot", "profession", "pilot", "飞行员", "pilote", en_article="a", fr_article="un"),
    concept("school", "place", "school", "学校", "école", en_article="a", fr_article="une"),
    concept("hospital", "place", "hospital", "医院", "hôpital", en_article="a", fr_article="un"),
    concept("library", "place", "library", "图书馆", "bibliothèque", en_article="a", fr_article="une"),
    concept("station", "place", "station", "车站", "gare", en_article="a", fr_article="une"),
    concept("market", "place", "market", "市场", "marché", en_article="a", fr_article="un"),
    concept("book", "object", "book", "书", "livre", en_article="a", fr_article="un"),
    concept("chair", "object", "chair", "椅子", "chaise", en_article="a", fr_article="une"),
    concept("table", "object", "table", "桌子", "table", en_article="a", fr_article="une"),
    concept("key", "object", "key", "钥匙", "clé", en_article="a", fr_article="une"),
    concept("lamp", "object", "lamp", "灯", "lampe", en_article="a", fr_article="une"),
)
CONCEPT_BY_ID = {row["concept_id"]: row for row in CONCEPTS}
DISCOVERY_CONCEPTS = tuple(row["concept_id"] for row in CONCEPTS[:18])
CONFIRMATION_CONCEPTS = tuple(row["concept_id"] for row in CONCEPTS[18:])
CONCEPTS_BY_SPLIT = {
    "discovery": DISCOVERY_CONCEPTS,
    "confirmation": CONFIRMATION_CONCEPTS,
}
FROZEN_CONTROL_CONCEPTS = (
    "apple",
    "cat",
    "car",
    "teacher",
    "school",
    "book",
    "doctor",
    "lamp",
    "banana",
    "dog",
)


RARE_PAIRS = (
    (("饕餮", ("贪食", "贪欲")), ("獬豸", ("公正", "司法"))),
    (("貔貅", ("招财", "辟邪")), ("麒麟", ("祥瑞", "仁德"))),
    (("赑屃", ("石碑", "驮碑")), ("螭吻", ("屋脊", "防火"))),
    (("圭臬", ("准则", "标准")), ("鸿鹄", ("志向", "远志"))),
    (("桎梏", ("束缚", "限制")), ("樊笼", ("牢笼", "困境"))),
    (("扶桑", ("神树", "东方")), ("青鸟", ("信使", "传信"))),
    (("耄耋", ("高龄", "老人")), ("桑梓", ("故乡", "家乡"))),
    (("蒹葭", ("芦苇",)), ("社稷", ("国家", "江山"))),
    (("甪端", ("祥瑞", "神兽")), ("狻猊", ("狮子", "神兽"))),
    (("夔牛", ("雷声", "独脚")), ("精卫", ("填海",))),
    (("鲲鹏", ("巨鸟", "远志")), ("蓬莱", ("仙山", "仙境"))),
    (("不啻", ("如同", "不只")), ("踟蹰", ("犹豫", "徘徊"))),
    (("龃龉", ("冲突", "不合")), ("囹圄", ("监狱", "牢狱"))),
    (("饔飧", ("饭食", "饮食")), ("觥筹", ("酒宴", "饮酒"))),
    (("罅隙", ("缝隙", "裂缝")), ("倥偬", ("忙乱", "匆忙"))),
    (("葳蕤", ("茂盛", "繁盛")), ("踽踽", ("孤独", "独行"))),
)


PUNCTUATION_ITEMS = (
    {
        "item_id": "terminal_archive",
        "subgroup": "statement_question",
        "branches": (
            ("The archive is open", (".", "。")),
            ("Is the archive open", ("?", "？")),
        ),
    },
    {
        "item_id": "terminal_train",
        "subgroup": "statement_question",
        "branches": (
            ("The last train has arrived", (".", "。")),
            ("Has the last train arrived", ("?", "？")),
        ),
    },
    {
        "item_id": "terminal_meeting",
        "subgroup": "statement_question",
        "branches": (
            ("The meeting begins at noon", (".", "。")),
            ("Does the meeting begin at noon", ("?", "？")),
        ),
    },
    {
        "item_id": "terminal_key",
        "subgroup": "statement_question",
        "branches": (
            ("This key opens the cabinet", (".", "。")),
            ("Does this key open the cabinet", ("?", "？")),
        ),
    },
    {
        "item_id": "terminal_weather",
        "subgroup": "statement_exclamation",
        "branches": (
            ("The weather is warm today", (".", "。")),
            ("What wonderful weather", ("!", "！")),
        ),
    },
    {
        "item_id": "terminal_view",
        "subgroup": "statement_exclamation",
        "branches": (
            ("The mountain is visible", (".", "。")),
            ("What a magnificent view", ("!", "！")),
        ),
    },
    {
        "item_id": "terminal_news",
        "subgroup": "statement_exclamation",
        "branches": (
            ("The team won the match", (".", "。")),
            ("What amazing news", ("!", "！")),
        ),
    },
    {
        "item_id": "terminal_speed",
        "subgroup": "statement_exclamation",
        "branches": (
            ("The train moves quickly", (".", "。")),
            ("How fast the train is moving", ("!", "！")),
        ),
    },
    {
        "item_id": "zh_terminal_library",
        "subgroup": "statement_question",
        "branches": (
            ("图书馆已经开放", ("。", ".")),
            ("图书馆开放了吗", ("？", "?")),
        ),
    },
    {
        "item_id": "zh_terminal_train",
        "subgroup": "statement_question",
        "branches": (
            ("末班车已经到站", ("。", ".")),
            ("末班车到站了吗", ("？", "?")),
        ),
    },
    {
        "item_id": "zh_terminal_key",
        "subgroup": "statement_question",
        "branches": (
            ("这把钥匙能打开柜子", ("。", ".")),
            ("这把钥匙能打开柜子吗", ("？", "?")),
        ),
    },
    {
        "item_id": "zh_terminal_meeting",
        "subgroup": "statement_question",
        "branches": (
            ("会议在中午开始", ("。", ".")),
            ("会议是在中午开始吗", ("？", "?")),
        ),
    },
    {
        "item_id": "zh_terminal_view",
        "subgroup": "statement_exclamation",
        "branches": (
            ("山景十分壮丽", ("。", ".")),
            ("多么壮丽的山景啊", ("！", "!")),
        ),
    },
    {
        "item_id": "zh_terminal_news",
        "subgroup": "statement_exclamation",
        "branches": (
            ("球队赢得了比赛", ("。", ".")),
            ("球队竟然赢得了比赛", ("！", "!")),
        ),
    },
    {
        "item_id": "zh_terminal_speed",
        "subgroup": "statement_exclamation",
        "branches": (
            ("列车行驶得很快", ("。", ".")),
            ("列车跑得真快啊", ("！", "!")),
        ),
    },
    {
        "item_id": "zh_terminal_weather",
        "subgroup": "statement_exclamation",
        "branches": (
            ("今天天气很好", ("。", ".")),
            ("今天天气真好啊", ("！", "!")),
        ),
    },
    {
        "item_id": "boundary_list_bag",
        "subgroup": "comma_colon",
        "branches": (
            ("After packing the bag [MARK] she left", (",", "，")),
            ("She packed three things [MARK] a coat, a map, and a lamp", (":", "：")),
        ),
    },
    {
        "item_id": "boundary_list_plan",
        "subgroup": "comma_colon",
        "branches": (
            ("Before announcing the plan [MARK] she checked it", (",", "，")),
            ("The plan had three stages [MARK] test, revise, and deploy", (":", "：")),
        ),
    },
    {
        "item_id": "boundary_list_food",
        "subgroup": "comma_colon",
        "branches": (
            ("After dinner ended [MARK] the guests departed", (",", "，")),
            ("Dinner included three dishes [MARK] soup, rice, and fish", (":", "：")),
        ),
    },
    {
        "item_id": "boundary_list_tools",
        "subgroup": "comma_colon",
        "branches": (
            ("Before using the tools [MARK] he inspected them", (",", "，")),
            ("He needed three tools [MARK] a saw, a drill, and a level", (":", "：")),
        ),
    },
    {
        "item_id": "zh_boundary_list_bag",
        "subgroup": "comma_colon",
        "branches": (
            ("收拾完行李后 [MARK] 她离开了", ("，", ",")),
            ("她带了三件东西 [MARK] 外套、地图和灯", ("：", ":")),
        ),
    },
    {
        "item_id": "zh_boundary_list_plan",
        "subgroup": "comma_colon",
        "branches": (
            ("宣布计划之前 [MARK] 她又检查了一遍", ("，", ",")),
            ("计划分为三个阶段 [MARK] 测试、修改和部署", ("：", ":")),
        ),
    },
    {
        "item_id": "zh_boundary_list_food",
        "subgroup": "comma_colon",
        "branches": (
            ("晚餐结束后 [MARK] 客人们离开了", ("，", ",")),
            ("晚餐有三道菜 [MARK] 汤、米饭和鱼", ("：", ":")),
        ),
    },
    {
        "item_id": "zh_boundary_list_tools",
        "subgroup": "comma_colon",
        "branches": (
            ("使用工具之前 [MARK] 他逐一检查", ("，", ",")),
            ("他需要三件工具 [MARK] 锯子、电钻和水平仪", ("：", ":")),
        ),
    },
)


CONTRAST_TOPICS = (
    (
        "road",
        ("The road was icy", "traffic slowed"),
        ("The road was icy", "traffic continued normally"),
        ("The road was covered with ice", "cars moved slowly"),
        ("The road was covered with ice", "cars kept their usual speed"),
    ),
    (
        "rain",
        ("Heavy rain began", "the match was delayed"),
        ("Heavy rain began", "the match continued"),
        ("A downpour started", "the game was postponed"),
        ("A downpour started", "the game went on"),
    ),
    (
        "alarm",
        ("The alarm sounded", "everyone left the building"),
        ("The alarm sounded", "nobody moved"),
        ("The warning bell rang", "people evacuated"),
        ("The warning bell rang", "people stayed seated"),
    ),
    (
        "exam",
        ("Mira studied carefully", "she passed the exam"),
        ("The exam was difficult", "Mira remained calm"),
        ("Mira prepared thoroughly", "she succeeded"),
        ("The test was hard", "Mira stayed relaxed"),
    ),
    (
        "battery",
        ("The battery was empty", "the device stopped"),
        ("The battery was empty", "the device kept running"),
        ("The battery had no charge", "the machine shut down"),
        ("The battery had no charge", "the machine continued working"),
    ),
    (
        "engine",
        ("The engine overheated", "the driver stopped"),
        ("The engine was old", "it ran quietly"),
        ("The motor became too hot", "the driver pulled over"),
        ("The motor was aged", "it operated smoothly"),
    ),
    (
        "door",
        ("The door was locked", "we used another entrance"),
        ("The door was locked", "we entered immediately"),
        ("The entrance was secured", "we chose a different route"),
        ("The entrance was secured", "we walked straight in"),
    ),
    (
        "cloud",
        ("Dark clouds gathered", "rain followed"),
        ("The sky was cloudy", "the afternoon stayed warm"),
        ("Storm clouds formed", "rain soon began"),
        ("Clouds covered the sky", "the day remained warm"),
    ),
    (
        "price",
        ("The price fell", "demand increased"),
        ("The phone was inexpensive", "the camera was excellent"),
        ("The cost dropped", "more people bought it"),
        ("The phone was cheap", "its camera was outstanding"),
    ),
    (
        "signal",
        ("The signal failed", "the train stopped"),
        ("The signal failed", "the train continued safely"),
        ("The signal went dark", "the train halted"),
        ("The signal went dark", "the train proceeded without trouble"),
    ),
    (
        "practice",
        ("The team practiced daily", "its performance improved"),
        ("The team was inexperienced", "it won the match"),
        ("The team trained every day", "it became stronger"),
        ("The team lacked experience", "it still won"),
    ),
    (
        "bridge",
        ("The bridge closed", "traffic took a detour"),
        ("The bridge was narrow", "traffic moved quickly"),
        ("The crossing was shut", "vehicles used another road"),
        ("The crossing was tight", "vehicles moved fast"),
    ),
    (
        "medicine",
        ("She took the medicine", "the fever declined"),
        ("She took the medicine", "the fever remained high"),
        ("She used the treatment", "her temperature fell"),
        ("She used the treatment", "her temperature stayed high"),
    ),
    (
        "book",
        ("The explanation was clear", "the students understood"),
        ("The book was long", "the argument stayed clear"),
        ("The account was precise", "the class followed it"),
        ("The volume was lengthy", "its reasoning remained clear"),
    ),
    (
        "map",
        ("The map was accurate", "we found the village"),
        ("The map was old", "we found the village"),
        ("The chart was correct", "we reached the town"),
        ("The chart was outdated", "we still reached the town"),
    ),
    (
        "internet",
        ("The network failed", "the upload stopped"),
        ("The house was remote", "the internet was reliable"),
        ("The connection dropped", "the transfer ended"),
        ("The home was isolated", "the connection stayed stable"),
    ),
)


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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row))
            handle.write("\n")
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def state_factors(state: str) -> tuple[int, int]:
    if state == "identity":
        return 0, 0
    return int(state[1]), int(state[-1])


def offset_token_spans(
    tokenizer,
    rendered: str,
    raw_prompt: str,
    spans: dict[str, tuple[int, int, str]],
) -> dict[str, tuple[int, int]]:
    """Locate marked roles using tokenizer offsets.

    Current local tokenizers expose fast offset mappings even when their
    historical loader requests ``use_fast=False``.  Offsets avoid the prefix
    retokenization drift that appears around multilingual punctuation.
    """

    raw_start = rendered.index(raw_prompt)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    expected = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    if input_ids != expected:
        raise RuntimeError("offset tokenizer ids differ from encode ids")
    offsets = [
        (int(start), int(end))
        for start, end in encoded["offset_mapping"]
    ]
    result = {}
    for role, (start, end, marked_text) in spans.items():
        absolute_start = raw_start + start
        absolute_end = raw_start + end
        hits = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start
            and token_start < absolute_end
            and token_end > absolute_start
        ]
        if not hits:
            raise RuntimeError(
                f"{role}: no offset tokens for {marked_text!r}"
            )
        result[role] = (min(hits), max(hits))
    return result


def normalized_text(value: str, *, accents: bool = True) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    if not accents:
        value = "".join(
            char
            for char in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(char)
        )
    value = re.sub(r"\s+", " ", value)
    return value


def sentence_for(row: dict[str, Any], language: str) -> str:
    term = row["terms"][language]
    if language == "en":
        return f"This is {row['articles']['en']} {term}."
    if language == "zh":
        return f"这是{term}。"
    return f"C'est {row['articles']['fr']} {term}."


def term_outputs(row: dict[str, Any], language: str) -> list[str]:
    term = row["terms"][language]
    values = [term]
    if language == "fr":
        article = row["articles"]["fr"]
        values.extend((f"{article} {term}", f"le {term}", f"la {term}"))
    return list(dict.fromkeys(values))


def operation_outputs(
    row: dict[str, Any],
    operation: str,
    source_language: str,
    target_language: str,
) -> list[str]:
    if operation == "copy":
        return [row["terms"][source_language]]
    if operation == "translate":
        return term_outputs(row, target_language)
    if operation == "classify":
        return list(CATEGORY_TERMS[row["category"]][target_language])
    if operation == "translate_sentence":
        return [sentence_for(row, target_language)]
    raise KeyError(operation)


def prompt_template(split: str, lexical: int) -> int:
    if split == "discovery":
        return lexical
    return lexical + 2


OPERATION_WORDS = {
    0: {
        "translate": "translate",
        "classify": "name the broad category of",
        "copy": "copy exactly",
        "translate_sentence": "translate the complete sentence",
    },
    1: {
        "translate": "翻译",
        "classify": "判断宽泛类别",
        "copy": "原样复制",
        "translate_sentence": "翻译完整句子",
    },
    2: {
        "translate": "render in the requested language",
        "classify": "return its broad semantic class",
        "copy": "repeat without changing",
        "translate_sentence": "render the entire sentence",
    },
    3: {
        "translate": "转换为目标语言",
        "classify": "输出其宽泛语义类别",
        "copy": "不作修改地复述",
        "translate_sentence": "转换整句语言",
    },
}


def render_operation_prompt(
    *,
    split: str,
    lexical: int,
    source: str,
    source_language: str,
    target_language: str,
    operation: str,
    style: str,
) -> tuple[str, dict[str, tuple[int, int, str]]]:
    template = prompt_template(split, lexical)
    english = template in (0, 2)
    builder = SpanBuilder()
    builder.mark(
        "prefix_anchor",
        "Natural language operation" if english else "自然语言操作",
        prefix="",
    )
    if template in (0, 1):
        builder.add(". Source language:" if english else "。源语言：")
        builder.add(LANGUAGE_NAMES[source_language][0 if english else 1])
        builder.add(". Source:" if english else "。源内容：")
        builder.mark("carrier", source)
        builder.add(". Operation:" if english else "。操作：")
        builder.mark(
            "operator",
            OPERATION_WORDS[template][operation],
        )
        builder.add(". Target language:" if english else "。目标语言：")
        builder.mark(
            "context_anchor",
            LANGUAGE_NAMES[target_language][0 if english else 1],
        )
        if style:
            builder.add(f". {style}" if english else f"。{style}")
        builder.add(
            ". Return only the result, without explanation."
            if english
            else "。只输出结果，不要解释。"
        )
        builder.mark(
            "query_anchor",
            "Complete the operation" if english else "完成操作",
        )
        builder.mark("answer_boundary", "Answer:" if english else "答案：")
    else:
        builder.add(". Input language=" if english else "。输入语言=")
        builder.add(LANGUAGE_NAMES[source_language][0 if english else 1])
        builder.add("; input=" if english else "；输入=")
        builder.mark("carrier", source)
        builder.add("; task=" if english else "；任务=")
        builder.mark(
            "operator",
            OPERATION_WORDS[template][operation],
        )
        builder.add("; output language=" if english else "；输出语言=")
        builder.mark(
            "context_anchor",
            LANGUAGE_NAMES[target_language][0 if english else 1],
        )
        if style:
            builder.add(f"; {style}" if english else f"；{style}")
        builder.add(
            ". Write only the transformed result."
            if english
            else "。仅写变换后的结果。"
        )
        builder.mark(
            "query_anchor",
            "Execute now" if english else "现在执行",
        )
        builder.mark("answer_boundary", "Output:" if english else "输出：")
    return builder.finish()


def resolve_operation_case(
    unit: dict[str, Any],
    branch: int,
) -> tuple[dict[str, Any], str, str, str, str, list[str], list[str]]:
    contrast = unit["contrast_type"]
    source_language = unit["source_language"]
    style = ""
    if contrast == "operation_switch":
        row = CONCEPT_BY_ID[unit["concept_id"]]
        target_language = unit["target_language"]
        operation = ("translate", "classify")[branch]
        foil_operation = ("classify", "translate")[branch]
        foil = operation_outputs(
            row, foil_operation, source_language, target_language
        )
    elif contrast == "target_switch":
        row = CONCEPT_BY_ID[unit["concept_id"]]
        target_language = unit["target_languages"][branch]
        operation = "translate"
        foil = operation_outputs(
            row,
            operation,
            source_language,
            unit["target_languages"][1 - branch],
        )
    elif contrast == "content_switch":
        row = CONCEPT_BY_ID[unit["concept_ids"][branch]]
        target_language = unit["target_language"]
        operation = "translate"
        other = CONCEPT_BY_ID[unit["concept_ids"][1 - branch]]
        foil = operation_outputs(
            other, operation, source_language, target_language
        )
    elif contrast == "mode_switch":
        row = CONCEPT_BY_ID[unit["concept_id"]]
        operation = ("copy", "translate")[branch]
        target_language = (
            source_language if branch == 0 else unit["target_language"]
        )
        foil_operation = ("translate", "copy")[branch]
        foil_target = (
            unit["target_language"]
            if foil_operation == "translate"
            else source_language
        )
        foil = operation_outputs(
            row, foil_operation, source_language, foil_target
        )
    elif contrast == "irrelevant_switch":
        row = CONCEPT_BY_ID[unit["concept_id"]]
        operation = "translate"
        target_language = unit["target_language"]
        style = (
            ("Be concise", "Be precise")
            if branch == 0
            else ("Respond carefully", "Answer steadily")
        )[unit["world"] % 2]
        other = CONCEPT_BY_ID[unit["foil_concept_id"]]
        foil = operation_outputs(
            other, operation, source_language, target_language
        )
    elif contrast == "sentence_rollout":
        row = CONCEPT_BY_ID[unit["concept_id"]]
        operation = ("copy", "translate_sentence")[branch]
        target_language = (
            source_language if branch == 0 else unit["target_language"]
        )
        foil_operation = ("translate_sentence", "copy")[branch]
        foil_target = (
            unit["target_language"]
            if foil_operation == "translate_sentence"
            else source_language
        )
        foil = operation_outputs(
            row, foil_operation, source_language, foil_target
        )
    else:
        raise KeyError(contrast)

    source = (
        sentence_for(row, source_language)
        if contrast == "sentence_rollout"
        else row["terms"][source_language]
    )
    gold = operation_outputs(
        row, operation, source_language, target_language
    )
    return (
        row,
        source,
        source_language,
        target_language,
        operation,
        gold,
        foil,
    )


def render_rare_prompt(
    unit: dict[str, Any],
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], list[str], list[str], str]:
    term, accepted = RARE_PAIRS[unit["pair_index"]][branch]
    other_accepted = RARE_PAIRS[unit["pair_index"]][1 - branch][1]
    template = prompt_template(unit["split"], lexical)
    english = template in (0, 2)
    builder = SpanBuilder()
    if template < 2:
        rare_prefix = (
            "Rare Chinese word" if english else "生僻汉语词"
        )
    else:
        rare_prefix = (
            "Uncommon Chinese expression" if english else "罕见汉语表达"
        )
    builder.mark(
        "prefix_anchor",
        rare_prefix,
        prefix="",
    )
    builder.add(". Term:" if english else "。词语：")
    builder.mark("carrier", term)
    builder.add(
        ". Knowledge domain:"
        if english
        else "。知识范围："
    )
    builder.mark(
        "context_anchor",
        "Chinese lexical meaning" if english else "汉语词义",
    )
    builder.add(". Operation:" if english else "。操作：")
    builder.mark(
        "operator",
        "retrieve its best-known short meaning"
        if english
        else "提取最典型的简短含义",
    )
    builder.add(
        ". Return one short expression only."
        if english
        else "。只输出一个简短词语。"
    )
    builder.mark(
        "query_anchor",
        "Give the meaning" if english else "给出含义",
    )
    builder.mark("answer_boundary", "Answer:" if english else "答案：")
    prompt, spans = builder.finish()
    return prompt, spans, list(accepted), list(other_accepted), term


def render_punctuation_prompt(
    unit: dict[str, Any],
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], list[str], list[str], str]:
    spec = PUNCTUATION_ITEMS[unit["punctuation_index"]]
    text, accepted = spec["branches"][branch]
    _, foil = spec["branches"][1 - branch]
    template = prompt_template(unit["split"], lexical)
    english = template in (0, 2)
    builder = SpanBuilder()
    if template < 2:
        punctuation_prefix = (
            "Natural punctuation continuation" if english else "自然标点续写"
        )
    else:
        punctuation_prefix = (
            "Punctuation boundary completion" if english else "标点边界补全"
        )
    builder.mark(
        "prefix_anchor",
        punctuation_prefix,
        prefix="",
    )
    builder.add(". Text:" if english else "。文本：")
    builder.mark("carrier", text)
    builder.add(". Boundary type:" if english else "。边界类型：")
    builder.mark(
        "context_anchor",
        "the marked or final boundary" if english else "标记处或句末边界",
    )
    builder.add(". Operation:" if english else "。操作：")
    builder.mark(
        "operator",
        "supply one punctuation mark" if english else "补全一个标点",
    )
    builder.add(
        ". Output the mark only." if english else "。只输出标点本身。"
    )
    builder.mark(
        "query_anchor",
        "Complete the boundary" if english else "补全边界",
    )
    builder.mark("answer_boundary", "Mark:" if english else "标点：")
    prompt, spans = builder.finish()
    return prompt, spans, list(accepted), list(foil), spec["subgroup"]


def render_contrast_prompt(
    unit: dict[str, Any],
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], list[str], list[str], str]:
    topic = CONTRAST_TOPICS[unit["topic_index"]]
    lexical_offset = 0 if lexical == 0 else 2
    pair_offset = 1 if branch == 1 else 0
    left, right = topic[1 + lexical_offset + pair_offset]
    task_kind = unit["task_kind"]
    if task_kind == "relation_recognition":
        accepted = (
            ("cause", "causal", "因果")
            if branch == 0
            else ("contrast", "concession", "转折")
        )
        foil = (
            ("contrast", "concession", "转折")
            if branch == 0
            else ("cause", "causal", "因果")
        )
        operator = "name the semantic relation"
        query = "Return one relation label"
    else:
        accepted = (
            ("therefore", "so", "因此", "所以")
            if branch == 0
            else ("however", "but", "但是", "然而")
        )
        foil = (
            ("however", "but", "但是", "然而")
            if branch == 0
            else ("therefore", "so", "因此", "所以")
        )
        operator = "supply a connector between the clauses"
        query = "Return one connector"
    template = prompt_template(unit["split"], lexical)
    english = template in (0, 2)
    builder = SpanBuilder()
    if template < 2:
        contrast_prefix = "Clause relation" if english else "分句关系"
    else:
        contrast_prefix = (
            "Inter-clause relation task" if english else "分句间关系任务"
        )
    builder.mark(
        "prefix_anchor",
        contrast_prefix,
        prefix="",
    )
    builder.add(". Clauses:" if english else "。分句：")
    builder.mark("carrier", f"{left} [LINK] {right}")
    builder.add(". Relation domain:" if english else "。关系范围：")
    builder.mark(
        "context_anchor",
        "the relation between the clauses" if english else "两个分句之间的关系",
    )
    builder.add(". Operation:" if english else "。操作：")
    builder.mark(
        "operator",
        operator if english else (
            "命名语义关系"
            if task_kind == "relation_recognition"
            else "补充分句连接词"
        ),
    )
    builder.add(
        ". Do not explain." if english else "。不要解释。"
    )
    builder.mark(
        "query_anchor",
        query if english else (
            "只输出关系标签"
            if task_kind == "relation_recognition"
            else "只输出一个连接词"
        ),
    )
    builder.mark("answer_boundary", "Answer:" if english else "答案：")
    prompt, spans = builder.finish()
    return prompt, spans, list(accepted), list(foil), task_kind


def render_unit_state(
    unit: dict[str, Any],
    state: str,
) -> tuple[
    str,
    dict[str, tuple[int, int, str]],
    list[str],
    list[str],
    dict[str, Any],
]:
    branch, lexical = state_factors(state)
    if unit["family"] == "multilingual_operation":
        (
            row,
            source,
            source_language,
            target_language,
            operation,
            gold,
            foil,
        ) = resolve_operation_case(unit, branch)
        prompt, spans = render_operation_prompt(
            split=unit["split"],
            lexical=lexical,
            source=source,
            source_language=source_language,
            target_language=target_language,
            operation=operation,
            style=(
                (
                    ("Be concise", "Be precise")
                    if branch == 0
                    else ("Respond carefully", "Answer steadily")
                )[unit["world"] % 2]
                if unit["contrast_type"] == "irrelevant_switch"
                else ""
            ),
        )
        metadata = {
            "subgroup": unit["contrast_type"],
            "task_kind": (
                "sentence_translation"
                if unit["contrast_type"] == "sentence_rollout"
                and operation == "translate_sentence"
                else operation
            ),
            "operation": operation,
            "source_language": source_language,
            "target_language": target_language,
            "concept_id": row["concept_id"],
            "source_text": source,
            "evaluation_type": (
                "sentence"
                if operation == "translate_sentence"
                else "short_text"
            ),
            "max_new_tokens": 18 if operation == "translate_sentence" else 6,
        }
    elif unit["family"] == "rare_definition":
        prompt, spans, gold, foil, term = render_rare_prompt(
            unit, branch, lexical
        )
        metadata = {
            "subgroup": "rare_short_definition",
            "task_kind": "rare_definition",
            "term": term,
            "evaluation_type": "short_text",
            "max_new_tokens": 8,
        }
    elif unit["family"] == "punctuation_next":
        prompt, spans, gold, foil, subgroup = render_punctuation_prompt(
            unit, branch, lexical
        )
        metadata = {
            "subgroup": subgroup,
            "task_kind": "punctuation",
            "evaluation_type": "punctuation",
            "max_new_tokens": 2,
        }
    elif unit["family"] == "contrast_relation":
        prompt, spans, gold, foil, subgroup = render_contrast_prompt(
            unit, branch, lexical
        )
        metadata = {
            "subgroup": subgroup,
            "task_kind": subgroup,
            "evaluation_type": "short_text",
            "max_new_tokens": 5,
        }
    else:
        raise KeyError(unit["family"])
    return prompt, spans, gold, foil, metadata


def make_unit(
    *,
    family: str,
    item_id: str,
    split: str,
    subgroup: str,
    unit_key: str,
    scan_eligible: bool,
    **metadata: Any,
) -> dict[str, Any]:
    unit_id = f"p1021.{family}.{split}.{unit_key}"
    return {
        "schema_version": "phase1021_natural_unit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "family": family,
        "item_id": item_id,
        "split": split,
        "subgroup": subgroup,
        "unit_id": unit_id,
        "scan_eligible": bool(scan_eligible),
        "world": int(metadata.pop("world", 0)),
        **metadata,
    }


def concept_pairs_for_split(split: str) -> list[tuple[str, str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for concept_id in CONCEPTS_BY_SPLIT[split]:
        row = CONCEPT_BY_ID[concept_id]
        by_category[row["category"]].append(concept_id)
    pairs = []
    for values in by_category.values():
        for index in range(0, len(values) - 1, 2):
            pairs.append((values[index], values[index + 1]))
    return pairs


def build_units() -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for split in SPLITS:
        for concept_index, concept_id in enumerate(CONCEPTS_BY_SPLIT[split]):
            for source_language, target_language in LANGUAGE_DIRECTIONS:
                row = CONCEPT_BY_ID[concept_id]
                if normalized_text(
                    row["terms"][source_language], accents=False
                ) == normalized_text(
                    row["terms"][target_language], accents=False
                ):
                    continue
                direction = f"{source_language}_{target_language}"
                units.append(make_unit(
                    family="multilingual_operation",
                    item_id=f"operation_{direction}",
                    split=split,
                    subgroup="operation_switch",
                    unit_key=f"operation.{direction}.{concept_id}",
                    scan_eligible=True,
                    contrast_type="operation_switch",
                    concept_id=concept_id,
                    source_language=source_language,
                    target_language=target_language,
                    world=concept_index % 4,
                ))
                units.append(make_unit(
                    family="multilingual_operation",
                    item_id=f"mode_{direction}",
                    split=split,
                    subgroup="mode_switch",
                    unit_key=f"mode.{direction}.{concept_id}",
                    scan_eligible=True,
                    contrast_type="mode_switch",
                    concept_id=concept_id,
                    source_language=source_language,
                    target_language=target_language,
                    world=concept_index % 4,
                ))
                if concept_id in FROZEN_CONTROL_CONCEPTS:
                    foil_index = (
                        list(CONCEPT_BY_ID).index(concept_id) + 1
                    ) % len(CONCEPTS)
                    units.append(make_unit(
                        family="multilingual_operation",
                        item_id=f"irrelevant_{direction}",
                        split=split,
                        subgroup="irrelevant_switch",
                        unit_key=f"irrelevant.{direction}.{concept_id}",
                        scan_eligible=True,
                        contrast_type="irrelevant_switch",
                        concept_id=concept_id,
                        foil_concept_id=CONCEPTS[foil_index]["concept_id"],
                        source_language=source_language,
                        target_language=target_language,
                        world=concept_index % 4,
                    ))
                    units.append(make_unit(
                        family="multilingual_operation",
                        item_id=f"sentence_{direction}",
                        split=split,
                        subgroup="sentence_rollout",
                        unit_key=f"sentence.{direction}.{concept_id}",
                        scan_eligible=False,
                        contrast_type="sentence_rollout",
                        concept_id=concept_id,
                        source_language=source_language,
                        target_language=target_language,
                        world=concept_index % 4,
                    ))
            for source_language in LANGUAGES:
                targets = tuple(
                    language
                    for language in LANGUAGES
                    if language != source_language
                )
                row = CONCEPT_BY_ID[concept_id]
                surfaces = [
                    normalized_text(
                        row["terms"][language], accents=False
                    )
                    for language in (source_language, *targets)
                ]
                if len(set(surfaces)) != len(surfaces):
                    continue
                units.append(make_unit(
                    family="multilingual_operation",
                    item_id=f"target_{source_language}",
                    split=split,
                    subgroup="target_switch",
                    unit_key=f"target.{source_language}.{concept_id}",
                    scan_eligible=True,
                    contrast_type="target_switch",
                    concept_id=concept_id,
                    source_language=source_language,
                    target_languages=targets,
                    world=concept_index % 4,
                ))

        for pair_index, pair in enumerate(concept_pairs_for_split(split)):
            for source_language, target_language in LANGUAGE_DIRECTIONS:
                if any(
                    normalized_text(
                        CONCEPT_BY_ID[concept_id]["terms"][source_language],
                        accents=False,
                    )
                    == normalized_text(
                        CONCEPT_BY_ID[concept_id]["terms"][target_language],
                        accents=False,
                    )
                    for concept_id in pair
                ):
                    continue
                direction = f"{source_language}_{target_language}"
                units.append(make_unit(
                    family="multilingual_operation",
                    item_id=f"content_{direction}",
                    split=split,
                    subgroup="content_switch",
                    unit_key=(
                        f"content.{direction}.{pair[0]}_{pair[1]}"
                    ),
                    scan_eligible=True,
                    contrast_type="content_switch",
                    concept_ids=pair,
                    source_language=source_language,
                    target_language=target_language,
                    world=pair_index % 4,
                ))

        for pair_index in range(len(RARE_PAIRS)):
            for world in range(2):
                units.append(make_unit(
                    family="rare_definition",
                    item_id=f"rare_pair_{pair_index:02d}",
                    split=split,
                    subgroup="rare_short_definition",
                    unit_key=f"rare.{pair_index:02d}.w{world}",
                    scan_eligible=True,
                    pair_index=pair_index,
                    world=world,
                ))

        for punctuation_index, spec in enumerate(PUNCTUATION_ITEMS):
            units.append(make_unit(
                family="punctuation_next",
                item_id=spec["subgroup"],
                split=split,
                subgroup=spec["subgroup"],
                unit_key=f"punctuation.{punctuation_index:02d}",
                scan_eligible=True,
                punctuation_index=punctuation_index,
                world=punctuation_index % 4,
            ))

        for topic_index, topic in enumerate(CONTRAST_TOPICS):
            for task_kind in (
                "relation_recognition",
                "connector_generation",
            ):
                units.append(make_unit(
                    family="contrast_relation",
                    item_id=task_kind,
                    split=split,
                    subgroup=task_kind,
                    unit_key=f"contrast.{task_kind}.{topic[0]}",
                    scan_eligible=True,
                    topic_index=topic_index,
                    task_kind=task_kind,
                    world=topic_index % 4,
                ))
    return units


def build_case(
    *,
    tokenizer,
    model_name: str,
    prompt_mode: str,
    unit: dict[str, Any],
    state: str,
) -> dict[str, Any]:
    raw_prompt, spans, accepted, foil_outputs, metadata = (
        render_unit_state(unit, state)
    )
    rendered = (
        raw_prompt
        if prompt_mode == "raw"
        else render_chat(tokenizer, model_name, raw_prompt)
    )
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    located = offset_token_spans(tokenizer, rendered, raw_prompt, spans)
    positions = {
        "prefix_anchor": located["prefix_anchor"][1],
        "carrier_start": located["carrier"][0],
        "carrier_end": located["carrier"][1],
        "context_anchor": located["context_anchor"][1],
        "operator": located["operator"][1],
        "query_anchor": located["query_anchor"][1],
        "answer_boundary": located["answer_boundary"][1],
    }
    canonical_gold = accepted[0]
    canonical_foil = foil_outputs[0]
    gold_ids = continuation_token_ids(
        tokenizer, rendered, canonical_gold
    )
    foil_ids = continuation_token_ids(
        tokenizer, rendered, canonical_foil
    )
    return {
        "schema_version": "phase1021_natural_case.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "family": unit["family"],
        "item_id": unit["item_id"],
        "split": unit["split"],
        "subgroup": unit["subgroup"],
        "unit_id": unit["unit_id"],
        "record_id": f"{unit['unit_id']}.{state}",
        "state": state,
        "world": int(unit["world"]),
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": positions,
        "carrier_token_count": (
            located["carrier"][1] - located["carrier"][0] + 1
        ),
        "accepted_outputs": accepted,
        "gold": canonical_gold,
        "foil": canonical_foil,
        "gold_token_ids": gold_ids,
        "foil_token_ids": foil_ids,
        "candidate_token_ids": {
            canonical_gold: gold_ids,
            canonical_foil: foil_ids,
        },
        "candidate_first_token_ids": {
            canonical_gold: gold_ids[0],
            canonical_foil: foil_ids[0],
        },
        **metadata,
    }


def audit_units(
    units: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    case_by_id = {row["record_id"]: row for row in cases}
    identity_ok = True
    answer_not_displayed = True
    prefix_ok = True
    state_count_ok = True
    failures = []
    for unit in units:
        by_state = {
            state: case_by_id[f"{unit['unit_id']}.{state}"]
            for state in STATES
        }
        identity_ok &= (
            by_state["identity"]["input_ids"]
            == by_state["b0_l0"]["input_ids"]
        )
        state_count_ok &= len(by_state) == len(STATES)
        for state, case in by_state.items():
            if (
                case["family"] == "punctuation_next"
                or (
                    case["family"] == "multilingual_operation"
                    and (
                        case["operation"] == "copy"
                        or normalized_text(
                            case["gold"], accents=False
                        )
                        == normalized_text(
                            case["source_text"], accents=False
                        )
                    )
                )
            ):
                continue
            prompt_norm = normalized_text(case["raw_prompt"], accents=False)
            gold_norm = normalized_text(case["gold"], accents=False)
            if re.search(r"[a-z0-9]", gold_norm):
                displayed = bool(re.search(
                    rf"(?<!\w){re.escape(gold_norm)}(?!\w)",
                    prompt_norm,
                ))
            else:
                displayed = bool(gold_norm and gold_norm in prompt_norm)
            if displayed:
                answer_not_displayed = False
                failures.append((case["record_id"], case["gold"]))
        for left_name, right_name in (
            ("b0_l0", "b1_l0"),
            ("b0_l1", "b1_l1"),
        ):
            left = by_state[left_name]
            right = by_state[right_name]
            boundary = min(
                left["role_positions"]["prefix_anchor"],
                right["role_positions"]["prefix_anchor"],
            )
            prefix_ok &= (
                left["input_ids"][:boundary + 1]
                == right["input_ids"][:boundary + 1]
            )
    discovery_prompts = {
        row["rendered_prompt"]
        for row in cases
        if row["split"] == "discovery"
    }
    confirmation_prompts = {
        row["rendered_prompt"]
        for row in cases
        if row["split"] == "confirmation"
    }
    return {
        "unit_count": len(units),
        "case_count": len(cases),
        "identity_exact": bool(identity_ok),
        "state_count_ok": bool(state_count_ok),
        "prefix_branch_unchanged": bool(prefix_ok),
        "candidate_answers_not_displayed": bool(answer_not_displayed),
        "candidate_display_failures": failures[:20],
        "discovery_confirmation_rendered_overlap": len(
            discovery_prompts & confirmation_prompts
        ),
    }


def preregistration(units: list[dict[str, Any]]) -> dict[str, Any]:
    frozen = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "families": list(FAMILIES),
        "languages": list(LANGUAGES),
        "language_directions": [
            f"{source}_{target}"
            for source, target in LANGUAGE_DIRECTIONS
        ],
        "factorial_states": list(FACTORIAL_STATES),
        "capture_roles": list(CAPTURE_ROLES),
        "unit_manifest": [
            {
                key: value
                for key, value in row.items()
                if key not in {"schema_version"}
            }
            for row in units
        ],
        "behavior_gates": {
            "minimum_models": 2,
            "multilingual_operation": {
                "translate_short_text_semantic_accuracy": 0.70,
                "classify_semantic_accuracy": 0.70,
                "sentence_translation_semantic_accuracy": 0.65,
            },
            "rare_definition": {
                "semantic_accuracy": 0.70,
            },
            "punctuation_next": {
                "semantic_accuracy": 0.85,
            },
            "contrast_relation": {
                "relation_recognition_accuracy": 0.70,
                "connector_generation_accuracy": 0.70,
            },
        },
        "generation_evaluation": {
            "primary_metric": "candidate-free greedy semantic accuracy",
            "short_text": (
                "accent-insensitive exact match after removing answer "
                "prefixes, quotes, and terminal punctuation"
            ),
            "rare_definition": (
                "accepted keyword exact match or containment in a short "
                "generated expression"
            ),
            "punctuation": "first generated punctuation mark",
            "sentence_translation": {
                "normalized_sequence_match_ratio": 0.72,
                "note": (
                    "This is a rollout stress metric against one canonical "
                    "reference, not a claim of unique translation."
                ),
            },
            "teacher_forced_margin": (
                "gold versus one hidden foil is diagnostic only; neither "
                "candidate is shown in the prompt"
            ),
        },
        "descriptive_repeat_thresholds": {
            "discovery_confirmation_cosine": 0.40,
            "cross_item_consistency": 0.30,
            "prevalence": 0.50,
        },
        "automatic_causal_gate": {
            "minimum_models": 2,
            "requires_natural_behavior_gate": True,
            "requires_operation_over_irrelevant_separation": True,
            "requires_cross_protocol_topology_repetition": True,
            "requires_target_and_operation_partial_separation": True,
            "claim_limit": (
                "Passing this gate only authorizes a separately "
                "preregistered causal test."
            ),
        },
        "claim_limits": [
            "Generated short answers are behavior, not internal mechanism.",
            "A repeated differential is not a transported semantic variable.",
            "Copy and target language are not treated as independent factors.",
            "Translate versus classify is the valid fixed-language operation contrast.",
            "No equation in this phase is a language law.",
        ],
    }
    frozen["protocol_digest"] = digest(frozen)
    return frozen


def main() -> None:
    units = build_units()
    prereg = preregistration(units)
    protocol_root = OUT_ROOT / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    write_json(protocol_root / "preregistration.json", prereg)

    summary = {
        "schema_version": "phase1021_protocol_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "unit_count": len(units),
        "units_by_family": dict(Counter(
            row["family"] for row in units
        )),
        "scan_units_by_family": dict(Counter(
            row["family"] for row in units if row["scan_eligible"]
        )),
        "discovery_concepts": list(DISCOVERY_CONCEPTS),
        "confirmation_concepts": list(CONFIRMATION_CONCEPTS),
        "concept_overlap": len(
            set(DISCOVERY_CONCEPTS) & set(CONFIRMATION_CONCEPTS)
        ),
        "models": list(MODELS),
        "prompt_modes": list(PROMPT_MODES),
    }

    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        for prompt_mode in PROMPT_MODES:
            cases = [
                build_case(
                    tokenizer=tokenizer,
                    model_name=model_name,
                    prompt_mode=prompt_mode,
                    unit=unit,
                    state=state,
                )
                for unit in units
                for state in STATES
            ]
            model_units = []
            for unit in units:
                model_units.append({
                    **unit,
                    "model": model_name,
                    "prompt_mode": prompt_mode,
                    "protocol_digest": prereg["protocol_digest"],
                    "record_ids": {
                        state: f"{unit['unit_id']}.{state}"
                        for state in STATES
                    },
                })
            audit = audit_units(model_units, cases)
            audit.update({
                "schema_version": "phase1021_protocol_audit.v1",
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "protocol_digest": prereg["protocol_digest"],
                "model": model_name,
                "prompt_mode": prompt_mode,
            })
            if not (
                audit["identity_exact"]
                and audit["state_count_ok"]
                and audit["prefix_branch_unchanged"]
                and audit["candidate_answers_not_displayed"]
                and audit["discovery_confirmation_rendered_overlap"] == 0
            ):
                raise RuntimeError(
                    f"protocol audit failed {model_name}/{prompt_mode}: "
                    f"{audit}"
                )
            write_jsonl(
                protocol_root
                / f"cases.{model_name}.{prompt_mode}.jsonl",
                cases,
            )
            write_jsonl(
                protocol_root
                / f"units.{model_name}.{prompt_mode}.jsonl",
                model_units,
            )
            write_json(
                protocol_root
                / f"audit.{model_name}.{prompt_mode}.json",
                audit,
            )
            summary[f"{model_name}.{prompt_mode}"] = audit
        del tokenizer

    write_json(protocol_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
