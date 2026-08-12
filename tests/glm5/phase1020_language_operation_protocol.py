#!/usr/bin/env python3
"""Freeze the Phase1020 language-pattern qualification and translation atlas."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import (
    CAPTURE_ROLES,
    FACTORIAL_STATES,
    MODELS,
    PROMPT_MODES,
    STATES,
    SpanBuilder,
    add_answer_instruction,
    add_choices,
    continuation_token_ids,
    digest,
    ordered_choices,
    read_json,
    read_jsonl,
    render_chat,
    token_spans,
    tokenizer_for,
    write_json,
    write_jsonl,
)


PHASE = 1020
PROTOCOL_REVISION = 1
FAMILIES = (
    "rare_knowledge",
    "punctuation_generation",
    "translation_mode",
    "contrast_relation",
)
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
WORLDS = tuple(range(4))
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1020_language_operation_atlas"
)


RARE_PAIRS = {
    "taotie_xiezhi": (
        ("饕餮", "以贪食著称的神话形象"),
        ("獬豸", "象征司法公正的独角神兽"),
    ),
    "pixiu_qilin": (
        ("貔貅", "常被视为招财辟邪的瑞兽"),
        ("麒麟", "象征祥瑞与仁德的神兽"),
    ),
    "bixi_chiwen": (
        ("赑屃", "常被雕成驮负石碑的龟形基座"),
        ("螭吻", "常安置在传统建筑屋脊两端"),
    ),
    "zhigu_fanlong": (
        ("桎梏", "可指束缚人的刑具或限制"),
        ("樊笼", "可指关鸟的笼子或受限处境"),
    ),
    "guinie_honghu": (
        ("圭臬", "常比喻应当遵循的准则"),
        ("鸿鹄", "常借指远大的志向"),
    ),
    "fusang_qingniao": (
        ("扶桑", "可指神话中的东方神树"),
        ("青鸟", "可指神话中传递消息的使者"),
    ),
}


PUNCTUATION_ITEMS = {
    "sq_archive": {
        "subgroup": "statement_question",
        "labels": (".", "?"),
        "texts": (
            ("The archive is open", "The archive remains open"),
            ("Is the archive open", "Does the archive remain open"),
        ),
    },
    "sq_train": {
        "subgroup": "statement_question",
        "labels": (".", "?"),
        "texts": (
            ("The final train has arrived", "The last train is here"),
            ("Has the final train arrived", "Is the last train here"),
        ),
    },
    "sq_meeting": {
        "subgroup": "statement_question",
        "labels": (".", "?"),
        "texts": (
            ("The meeting starts at noon", "The session begins at noon"),
            ("Does the meeting start at noon", "Will the session begin at noon"),
        ),
    },
    "sq_key": {
        "subgroup": "statement_question",
        "labels": (".", "?"),
        "texts": (
            ("This key opens the cabinet", "This key unlocks the cabinet"),
            ("Does this key open the cabinet", "Will this key unlock the cabinet"),
        ),
    },
    "cp_committee": {
        "subgroup": "period_colon",
        "labels": (".", ":"),
        "texts": (
            (
                "The committee reached a final decision | Everyone left",
                "The panel made its final choice | Everyone departed",
            ),
            (
                "The committee considered three options | delay, revise, or cancel",
                "The panel listed three choices | wait, edit, or withdraw",
            ),
        ),
    },
    "cp_bag": {
        "subgroup": "period_colon",
        "labels": (".", ":"),
        "texts": (
            (
                "She closed the travel bag | It was finally full",
                "She zipped the suitcase | It was completely packed",
            ),
            (
                "She packed three essentials | a coat, a map, and a lamp",
                "She brought three items | a jacket, a guide, and a torch",
            ),
        ),
    },
    "cp_cause": {
        "subgroup": "period_colon",
        "labels": (".", ":"),
        "texts": (
            (
                "The investigation ended | The report was filed",
                "The inquiry concluded | The report was submitted",
            ),
            (
                "The cause was clear | the sensor had failed",
                "There was one explanation | the bridge was closed",
            ),
        ),
    },
    "cp_goal": {
        "subgroup": "period_colon",
        "labels": (".", ":"),
        "texts": (
            (
                "She achieved her goal | The team applauded",
                "She completed the objective | Everyone celebrated",
            ),
            (
                "She had one goal | finish the manuscript",
                "He had one objective | complete the prototype",
            ),
        ),
    },
    "cs_bell": {
        "subgroup": "comma_semicolon",
        "labels": (",", ";"),
        "texts": (
            (
                "After the bell rang | the audience sat down",
                "When the bell sounded | the audience took their seats",
            ),
            (
                "The bell rang | the audience sat down",
                "The bell sounded | the audience took their seats",
            ),
        ),
    },
    "cs_storm": {
        "subgroup": "comma_semicolon",
        "labels": (",", ";"),
        "texts": (
            (
                "Although the storm ended | the roads stayed flooded",
                "Even after the storm passed | the roads remained flooded",
            ),
            (
                "The storm ended | the roads stayed flooded",
                "The storm passed | the roads remained flooded",
            ),
        ),
    },
    "cs_server": {
        "subgroup": "comma_semicolon",
        "labels": (",", ";"),
        "texts": (
            (
                "When the server restarted | the queued jobs resumed",
                "After the server rebooted | the waiting jobs resumed",
            ),
            (
                "The server restarted | the queued jobs resumed",
                "The server rebooted | the waiting jobs resumed",
            ),
        ),
    },
    "cs_sun": {
        "subgroup": "comma_semicolon",
        "labels": (",", ";"),
        "texts": (
            (
                "As the sun set | the air remained warm",
                "When the sun went down | the air stayed warm",
            ),
            (
                "The sun set | the air remained warm",
                "The sun went down | the air stayed warm",
            ),
        ),
    },
}


CONTRAST_ITEMS = {
    "room": {
        "labels": ("and", "but"),
        "left": ("The room was small", "The room was tiny"),
        "right": (
            ("it felt cramped", "it seemed crowded"),
            ("it felt comfortable", "it still seemed comfortable"),
        ),
    },
    "exam": {
        "labels": ("and", "but"),
        "left": ("The exam was difficult", "The test was demanding"),
        "right": (
            ("Mira struggled throughout", "Mira found it hard"),
            ("Mira remained calm", "Mira stayed composed"),
        ),
    },
    "engine": {
        "labels": ("and", "but"),
        "left": ("The engine was old", "The motor was aging"),
        "right": (
            ("it broke down often", "it failed frequently"),
            ("it ran quietly", "it remained surprisingly quiet"),
        ),
    },
    "book": {
        "labels": ("and", "but"),
        "left": ("The book was long", "The manuscript was lengthy"),
        "right": (
            ("the reading took weeks", "finishing it required weeks"),
            ("the argument stayed clear", "its argument remained clear"),
        ),
    },
    "team": {
        "labels": ("and", "but"),
        "left": ("The team was inexperienced", "The squad lacked experience"),
        "right": (
            ("it lost the match", "it was defeated"),
            ("it won the match", "it still secured a victory"),
        ),
    },
    "phone": {
        "labels": ("and", "but"),
        "left": ("The phone was inexpensive", "The handset was cheap"),
        "right": (
            ("the camera was basic", "its camera was limited"),
            ("the camera was excellent", "its camera was excellent"),
        ),
    },
    "rain": {
        "labels": ("therefore", "however"),
        "left": ("It rained heavily all night", "Heavy rain fell through the night"),
        "right": (
            ("the streets flooded", "the roads were flooded"),
            ("the outdoor match continued", "the open-air match went ahead"),
        ),
    },
    "alarm": {
        "labels": ("therefore", "however"),
        "left": ("The alarm sounded", "The warning siren rang"),
        "right": (
            ("everyone left the building", "the occupants evacuated"),
            ("nobody moved", "the occupants stayed in place"),
        ),
    },
    "battery": {
        "labels": ("therefore", "however"),
        "left": ("The battery was empty", "The battery had no charge"),
        "right": (
            ("the device shut down", "the device switched off"),
            ("the device kept running", "the device continued operating"),
        ),
    },
    "road": {
        "labels": ("therefore", "however"),
        "left": ("The road was blocked", "The route was closed"),
        "right": (
            ("traffic was diverted", "drivers took another route"),
            ("the bus arrived on time", "the bus still arrived punctually"),
        ),
    },
    "deadline": {
        "labels": ("therefore", "however"),
        "left": ("The deadline was extended", "The due date was postponed"),
        "right": (
            ("the team gained more time", "the group had extra time"),
            ("the team rushed to finish", "the group still hurried"),
        ),
    },
    "power": {
        "labels": ("therefore", "however"),
        "left": ("The power failed", "The electricity went out"),
        "right": (
            ("the lights went dark", "the lamps switched off"),
            ("the backup lights stayed on", "the emergency lamps remained lit"),
        ),
    },
}


TRANSLATION_CONCEPTS = {
    "apple": ("apple", "苹果"),
    "river": ("river", "河流"),
    "mountain": ("mountain", "山峰"),
    "teacher": ("teacher", "教师"),
    "window": ("window", "窗户"),
    "music": ("music", "音乐"),
    "freedom": ("freedom", "自由"),
    "science": ("science", "科学"),
    "garden": ("garden", "花园"),
    "bridge": ("bridge", "桥梁"),
    "winter": ("winter", "冬季"),
    "memory": ("memory", "记忆"),
    "circle": ("circle", "圆形"),
    "yellow": ("yellow", "黄色"),
    "moon": ("moon", "月亮"),
    "salt": ("salt", "食盐"),
}
TRANSLATION_SCAN_CONCEPTS = frozenset(
    ("apple", "river", "teacher", "music", "science", "bridge", "circle", "moon")
)
TRANSLATION_PROFILES = (
    "full",
    "operation_only",
    "language_only",
    "relation_only",
    "irrelevant_control",
)
TRANSLATION_DIRECTIONS = ("en_zh", "zh_en")


def translation_items() -> dict[str, dict[str, Any]]:
    rows = {}
    for concept, (english, chinese) in TRANSLATION_CONCEPTS.items():
        for direction in TRANSLATION_DIRECTIONS:
            if direction == "en_zh":
                source, target = english, chinese
                source_language, target_language = "English", "Chinese"
            else:
                source, target = chinese, english
                source_language, target_language = "Chinese", "English"
            for profile in TRANSLATION_PROFILES:
                item_id = f"{concept}__{direction}__{profile}"
                rows[item_id] = {
                    "concept": concept,
                    "direction": direction,
                    "profile": profile,
                    "source": source,
                    "target": target,
                    "source_language": source_language,
                    "target_language": target_language,
                    "labels": (source, target),
                    "scan_eligible": concept in TRANSLATION_SCAN_CONCEPTS,
                }
    return rows


TRANSLATION_ITEMS = translation_items()


def state_factors(state: str) -> tuple[int, int]:
    if state == "identity":
        return 0, 0
    return int(state[1]), int(state[-1])


def item_ids(family: str) -> tuple[str, ...]:
    if family == "rare_knowledge":
        return tuple(RARE_PAIRS)
    if family == "punctuation_generation":
        return tuple(PUNCTUATION_ITEMS)
    if family == "translation_mode":
        return tuple(TRANSLATION_ITEMS)
    if family == "contrast_relation":
        return tuple(CONTRAST_ITEMS)
    raise KeyError(family)


def render_rare(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    pair = RARE_PAIRS[item_id]
    term, definition = pair[branch]
    labels = (pair[0][1], pair[1][1])
    choices = ordered_choices(labels, world)
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "词义测试")
    if template == 0:
        builder.add("。待解释的古汉语词语是")
        builder.mark("carrier", term)
        builder.add("。知识范围：")
        builder.mark("context_anchor", ("传统文化", "古代语词")[lexical])
        builder.add("。")
        builder.mark("operator", ("选择", "判断")[lexical])
        builder.mark("query_anchor", "最符合该词的解释")
    elif template == 1:
        builder.add("。请识别词项")
        builder.mark("carrier", term)
        builder.add("。参考领域：")
        builder.mark("context_anchor", ("历史文化", "传统语义")[lexical])
        builder.add("。")
        builder.mark("operator", ("匹配", "选出")[lexical])
        builder.mark("query_anchor", "对应含义")
    elif template == 2:
        builder.add("。需要说明的词是")
        builder.mark("carrier", term)
        builder.add("。题目范围：")
        builder.mark("context_anchor", ("古典文化", "词语知识")[lexical])
        builder.add("。")
        builder.mark("operator", ("辨认", "确定")[lexical])
        builder.mark("query_anchor", "正确释义")
    else:
        builder.add("。词义核对对象：")
        builder.mark("carrier", term)
        builder.add("。所属知识：")
        builder.mark("context_anchor", ("传统语汇", "历史语义")[lexical])
        builder.add("。")
        builder.mark("operator", ("返回", "给出")[lexical])
        builder.mark("query_anchor", "准确解释")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, definition, {
        "subgroup": "rare_term_substitution",
        "carrier_text": term,
        "branch_labels": list(labels),
        "term": term,
        "paired_terms": [pair[0][0], pair[1][0]],
        "scan_eligible": True,
        "branch_changes_carrier": True,
        "semantic_branch_active": True,
    }


def render_punctuation(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    spec = PUNCTUATION_ITEMS[item_id]
    text = spec["texts"][branch][lexical]
    labels = spec["labels"]
    choices = ordered_choices(labels, world)
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Punctuation test")
    if template == 0:
        builder.add(". Replace the vertical bar, or finish the sentence, in:")
        builder.mark("carrier", text)
        builder.add(".")
        builder.mark("context_anchor", "the marked boundary")
        builder.add(".")
        builder.mark("operator", "Choose")
        builder.mark("query_anchor", "the grammatically appropriate mark")
    elif template == 1:
        builder.add(". Text awaiting its boundary symbol:")
        builder.mark("carrier", text)
        builder.add(".")
        builder.mark("context_anchor", "standard written English")
        builder.add(".")
        builder.mark("operator", "Supply")
        builder.mark("query_anchor", "the missing punctuation")
    elif template == 2:
        builder.add(". Complete this unpunctuated record:")
        builder.mark("carrier", text)
        builder.add(".")
        builder.mark("context_anchor", "the indicated slot or sentence ending")
        builder.add(".")
        builder.mark("operator", "Return")
        builder.mark("query_anchor", "the correct boundary mark")
    else:
        builder.add(". Copyediting item:")
        builder.mark("carrier", text)
        builder.add(".")
        builder.mark("context_anchor", "ordinary edited prose")
        builder.add(".")
        builder.mark("operator", "Select")
        builder.mark("query_anchor", "the best punctuation symbol")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, labels[branch], {
        "subgroup": spec["subgroup"],
        "carrier_text": text,
        "branch_labels": list(labels),
        "scan_eligible": True,
        "branch_changes_carrier": True,
        "semantic_branch_active": True,
    }


def translation_cues(
    spec: dict[str, Any],
    profile: str,
    branch: int,
    lexical: int,
) -> tuple[str, str, str]:
    source_language = spec["source_language"]
    target_language = spec["target_language"]
    if profile == "full":
        operator = (
            (("copy", "repeat"), ("translate", "convert"))[branch][lexical]
        )
        language = source_language if branch == 0 else target_language
        relation = "same-language output" if branch == 0 else "cross-language output"
    elif profile == "operation_only":
        operator = (
            (("copy", "echo"), ("translate", "render"))[branch][lexical]
        )
        language = ("requested form", "required form")[lexical]
        relation = "operation cue only"
    elif profile == "language_only":
        operator = ("return", "produce")[lexical]
        language = source_language if branch == 0 else target_language
        relation = "language cue only"
    elif profile == "relation_only":
        operator = (
            (
                ("return the identical expression", "give the same expression"),
                ("return its bilingual equivalent", "give the cross-language equivalent"),
            )[branch][lexical]
        )
        language = ("corresponding form", "matching form")[lexical]
        relation = "identity relation" if branch == 0 else "equivalence relation"
    elif profile == "irrelevant_control":
        operator = (
            (("answer carefully", "respond carefully"), ("answer briefly", "respond briefly"))[
                branch
            ][lexical]
        )
        language = source_language
        relation = "copy the source unchanged"
    else:
        raise KeyError(profile)
    return operator, language, relation


def render_translation(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    spec = TRANSLATION_ITEMS[item_id]
    profile = spec["profile"]
    operator, language, relation = translation_cues(
        spec, profile, branch, lexical
    )
    choices = ordered_choices(spec["labels"], world)
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Language task")
    if template == 0:
        builder.add(". Source expression:")
        builder.mark("carrier", spec["source"])
        builder.add(". Instruction:")
        builder.mark("operator", operator)
        builder.add(". Requested form:")
        builder.mark("context_anchor", language)
        builder.add(".")
        builder.mark("query_anchor", relation)
    elif template == 1:
        builder.add(". Input:")
        builder.mark("carrier", spec["source"])
        builder.add(". Action:")
        builder.mark("operator", operator)
        builder.add(". Output specification:")
        builder.mark("context_anchor", language)
        builder.add(".")
        builder.mark("query_anchor", relation)
    elif template == 2:
        builder.add(". Received expression:")
        builder.mark("carrier", spec["source"])
        builder.add(". Rendering request:")
        builder.mark("operator", operator)
        builder.add(". Destination form:")
        builder.mark("context_anchor", language)
        builder.add(".")
        builder.mark("query_anchor", relation)
    else:
        builder.add(". Entry to process:")
        builder.mark("carrier", spec["source"])
        builder.add(". Required action:")
        builder.mark("operator", operator)
        builder.add(". Required output:")
        builder.mark("context_anchor", language)
        builder.add(".")
        builder.mark("query_anchor", relation)
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    semantic_branch_active = profile != "irrelevant_control"
    gold = spec["labels"][branch] if semantic_branch_active else spec["labels"][0]
    return prompt, spans, gold, {
        "subgroup": profile,
        "carrier_text": spec["source"],
        "branch_labels": list(spec["labels"]),
        "concept": spec["concept"],
        "direction": spec["direction"],
        "profile": profile,
        "source_language": spec["source_language"],
        "target_language": spec["target_language"],
        "operator_surface": operator,
        "language_surface": language,
        "relation_surface": relation,
        "scan_eligible": bool(spec["scan_eligible"]),
        "branch_changes_carrier": False,
        "semantic_branch_active": semantic_branch_active,
    }


def render_contrast(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    spec = CONTRAST_ITEMS[item_id]
    left = spec["left"][lexical]
    right = spec["right"][branch][lexical]
    labels = spec["labels"]
    choices = ordered_choices(labels, world)
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Clause relation test")
    if template == 0:
        builder.add(". Clause one:")
        builder.mark("carrier", left)
        builder.add(". Clause two:")
        builder.mark("context_anchor", right)
        builder.add(".")
        builder.mark("operator", "Choose")
        builder.mark("query_anchor", "the connector joining the clauses")
    elif template == 1:
        builder.add(". First statement:")
        builder.mark("carrier", left)
        builder.add(". Following statement:")
        builder.mark("context_anchor", right)
        builder.add(".")
        builder.mark("operator", "Select")
        builder.mark("query_anchor", "the relation word")
    elif template == 2:
        builder.add(". Earlier clause:")
        builder.mark("carrier", left)
        builder.add(". Later clause:")
        builder.mark("context_anchor", right)
        builder.add(".")
        builder.mark("operator", "Supply")
        builder.mark("query_anchor", "the appropriate transition")
    else:
        builder.add(". Statement A:")
        builder.mark("carrier", left)
        builder.add(". Statement B:")
        builder.mark("context_anchor", right)
        builder.add(".")
        builder.mark("operator", "Return")
        builder.mark("query_anchor", "the best connector")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, labels[branch], {
        "subgroup": "additive_contrastive",
        "carrier_text": left,
        "branch_labels": list(labels),
        "right_clause": right,
        "scan_eligible": True,
        "branch_changes_carrier": False,
        "semantic_branch_active": True,
    }


def render_case(
    *,
    family: str,
    item_id: str,
    template: int,
    world: int,
    state: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    branch, lexical = state_factors(state)
    if family == "rare_knowledge":
        return render_rare(item_id, template, world, branch, lexical)
    if family == "punctuation_generation":
        return render_punctuation(item_id, template, world, branch, lexical)
    if family == "translation_mode":
        return render_translation(item_id, template, world, branch, lexical)
    if family == "contrast_relation":
        return render_contrast(item_id, template, world, branch, lexical)
    raise KeyError(family)


def build_case(
    *,
    tokenizer,
    model_name: str,
    prompt_mode: str,
    family: str,
    item_id: str,
    split: str,
    template: int,
    world: int,
    unit_id: str,
    state: str,
) -> dict[str, Any]:
    raw_prompt, spans, gold, metadata = render_case(
        family=family,
        item_id=item_id,
        template=template,
        world=world,
        state=state,
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
    located = token_spans(tokenizer, rendered, raw_prompt, spans)
    positions = {
        "prefix_anchor": located["prefix_anchor"][1],
        "carrier_start": located["carrier"][0],
        "carrier_end": located["carrier"][1],
        "context_anchor": located["context_anchor"][1],
        "operator": located["operator"][1],
        "query_anchor": located["query_anchor"][1],
        "answer_boundary": len(input_ids) - 1,
    }
    labels = tuple(metadata["branch_labels"])
    candidate_ids = {
        label: continuation_token_ids(tokenizer, rendered, label)
        for label in labels
    }
    foil = labels[1] if gold == labels[0] else labels[0]
    return {
        "schema_version": "phase1020_language_operation_case.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "family": family,
        "item_id": item_id,
        "split": split,
        "template": int(template),
        "world": int(world),
        "unit_id": unit_id,
        "record_id": f"{unit_id}.{state}",
        "state": state,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            role: int(positions[role]) for role in CAPTURE_ROLES
        },
        "carrier_token_count": (
            located["carrier"][1] - located["carrier"][0] + 1
        ),
        "gold": gold,
        "foil": foil,
        "candidate_labels": list(labels),
        "candidate_token_ids": candidate_ids,
        "candidate_first_token_ids": {
            label: values[0] for label, values in candidate_ids.items()
        },
        **metadata,
    }


def audit_unit(
    unit: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = {
        state: by_id[unit["record_ids"][state]] for state in STATES
    }
    identity_exact = (
        cases["identity"]["input_ids"] == cases["b0_l0"]["input_ids"]
    )
    candidate_unique = all(
        len({
            tuple(values)
            for values in case["candidate_token_ids"].values()
        }) == 2
        for case in cases.values()
    )
    carrier_prefix_stable = []
    for left_name, right_name in (
        ("b0_l0", "b1_l0"),
        ("b0_l1", "b1_l1"),
    ):
        left = cases[left_name]
        right = cases[right_name]
        left_end = left["role_positions"]["carrier_end"]
        right_end = right["role_positions"]["carrier_end"]
        carrier_prefix_stable.append(
            left_end == right_end
            and left["input_ids"][:left_end + 1]
            == right["input_ids"][:right_end + 1]
        )
    expected_stable = not bool(cases["b0_l0"]["branch_changes_carrier"])
    semantic_active = bool(cases["b0_l0"]["semantic_branch_active"])
    gold_changes = (
        cases["b0_l0"]["gold"] != cases["b1_l0"]["gold"]
        and cases["b0_l1"]["gold"] != cases["b1_l1"]["gold"]
    )
    return {
        "schema_version": "phase1020_protocol_unit_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": unit["model"],
        "prompt_mode": unit["prompt_mode"],
        "family": unit["family"],
        "item_id": unit["item_id"],
        "split": unit["split"],
        "template": unit["template"],
        "world": unit["world"],
        "unit_id": unit["unit_id"],
        "identity_exact": identity_exact,
        "candidate_ids_unique": candidate_unique,
        "carrier_prefix_expected": expected_stable,
        "branch_preserves_carrier_prefix": all(carrier_prefix_stable),
        "semantic_branch_active": semantic_active,
        "gold_change_matches_semantics": gold_changes == semantic_active,
        "scan_eligible": bool(unit["scan_eligible"]),
    }


def build_protocol() -> dict[str, Any]:
    protocol_root = OUT_ROOT / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "schema_version": "phase1020_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "prompt_modes": list(PROMPT_MODES),
        "families": list(FAMILIES),
        "items_per_family": {
            family: len(item_ids(family)) for family in FAMILIES
        },
        "splits": list(SPLITS),
        "templates_by_split": {
            key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()
        },
        "worlds": list(WORLDS),
        "states": list(STATES),
        "capture_roles": list(CAPTURE_ROLES),
        "translation_design": {
            "concept_count": len(TRANSLATION_CONCEPTS),
            "scan_concept_count": len(TRANSLATION_SCAN_CONCEPTS),
            "directions": list(TRANSLATION_DIRECTIONS),
            "profiles": list(TRANSLATION_PROFILES),
            "warning": (
                "Copy/translate and target language are not logically independent. "
                "Profiles are cue-removal and conflict controls, not a fictitious "
                "fully orthogonal semantic factorial."
            ),
        },
        "behavior_scan_gate": {
            "minimum_candidate_accuracy": 0.70,
            "minimum_model_count": 2,
            "translation_requires_full_profile": True,
            "other_families_use_family_accuracy": True,
        },
        "descriptive_thresholds": {
            "direction_consistency": 0.45,
            "surface_alignment": 0.40,
            "minimum_normalized_magnitude": 1e-4,
            "discovery_confirmation_cosine": 0.40,
            "cross_concept_consistency": 0.30,
            "minimum_item_prevalence": 0.50,
        },
        "automatic_followup_gate": {
            "minimum_model_count": 2,
            "full_accuracy": 0.70,
            "operation_only_accuracy": 0.70,
            "relation_only_accuracy": 0.70,
            "cross_direction_cosine": 0.15,
            "profile_over_negative_alignment_gain": 0.15,
            "requires_independent_confirmation": True,
        },
        "primary_questions": [
            "Does a rare-term substitution reliably reverse definition preference?",
            "Which punctuation families are behaviorally qualified before scanning?",
            "Which contrast relations are behaviorally qualified before scanning?",
            "Does the Phase1019 translation response follow operation wording, target-language wording, a relation cue, or generic instruction change?",
            "Does any translation-mode response retain orientation across English-to-Chinese and Chinese-to-English directions?",
        ],
        "claim_limits": [
            "All equations are measurement definitions chosen before model output.",
            "Repeated directions are descriptive structures, not complete mechanisms.",
            "A translation profile is a cue-removal probe, not a pure semantic factor.",
            "A rare-term substitution effect shows lexical control of this task, not a complete word meaning.",
            "Causal testing is not allowed until the descriptive followup gate passes.",
        ],
    }
    preregistration["protocol_digest"] = digest(preregistration)
    write_json(protocol_root / "preregistration.json", preregistration)

    global_summary = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": preregistration["protocol_digest"],
        "models": {},
    }
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        model_summary = {}
        for prompt_mode in PROMPT_MODES:
            cases = []
            units = []
            for family in FAMILIES:
                for item_id in item_ids(family):
                    for split in SPLITS:
                        for template in TEMPLATES_BY_SPLIT[split]:
                            for world in WORLDS:
                                unit_id = (
                                    f"p{PHASE}.{model_name}.{prompt_mode}."
                                    f"{family}.{item_id}.{split}.t{template}."
                                    f"w{world}"
                                )
                                record_ids = {}
                                unit_scan_eligible = None
                                for state in STATES:
                                    try:
                                        case = build_case(
                                            tokenizer=tokenizer,
                                            model_name=model_name,
                                            prompt_mode=prompt_mode,
                                            family=family,
                                            item_id=item_id,
                                            split=split,
                                            template=template,
                                            world=world,
                                            unit_id=unit_id,
                                            state=state,
                                        )
                                    except Exception as error:
                                        raise RuntimeError(
                                            f"case build failed: {unit_id}.{state}"
                                        ) from error
                                    cases.append(case)
                                    record_ids[state] = case["record_id"]
                                    if unit_scan_eligible is None:
                                        unit_scan_eligible = bool(
                                            case["scan_eligible"]
                                        )
                                    elif unit_scan_eligible != bool(
                                        case["scan_eligible"]
                                    ):
                                        raise RuntimeError(
                                            f"scan eligibility drift: {unit_id}"
                                        )
                                units.append({
                                    "schema_version": "phase1020_pattern_unit.v1",
                                    "phase": PHASE,
                                    "protocol_revision": PROTOCOL_REVISION,
                                    "model": model_name,
                                    "prompt_mode": prompt_mode,
                                    "family": family,
                                    "item_id": item_id,
                                    "split": split,
                                    "template": int(template),
                                    "world": int(world),
                                    "unit_id": unit_id,
                                    "record_ids": record_ids,
                                    "scan_eligible": bool(unit_scan_eligible),
                                })
            by_id = {case["record_id"]: case for case in cases}
            audits = [audit_unit(unit, by_id) for unit in units]
            if not all(
                row["identity_exact"]
                and row["candidate_ids_unique"]
                and row["gold_change_matches_semantics"]
                and (
                    row["branch_preserves_carrier_prefix"]
                    if row["carrier_prefix_expected"]
                    else True
                )
                for row in audits
            ):
                failed = [
                    row for row in audits
                    if not (
                        row["identity_exact"]
                        and row["candidate_ids_unique"]
                        and row["gold_change_matches_semantics"]
                        and (
                            row["branch_preserves_carrier_prefix"]
                            if row["carrier_prefix_expected"]
                            else True
                        )
                    )
                ]
                raise RuntimeError(
                    f"protocol audit failed for {model_name}/{prompt_mode}: "
                    f"{failed[:1]}"
                )
            overlap_total = 0
            for family in FAMILIES:
                for item_id in item_ids(family):
                    prompts = {
                        split: {
                            row["raw_prompt"]
                            for row in cases
                            if row["family"] == family
                            and row["item_id"] == item_id
                            and row["split"] == split
                        }
                        for split in SPLITS
                    }
                    overlap_total += len(
                        prompts["discovery"] & prompts["confirmation"]
                    )
            if overlap_total:
                raise RuntimeError(
                    f"held-out text overlap for {model_name}/{prompt_mode}"
                )
            write_jsonl(
                protocol_root / f"cases.{model_name}.{prompt_mode}.jsonl",
                cases,
            )
            write_jsonl(
                protocol_root / f"units.{model_name}.{prompt_mode}.jsonl",
                units,
            )
            write_jsonl(
                protocol_root / f"audit.{model_name}.{prompt_mode}.jsonl",
                audits,
            )
            model_summary[prompt_mode] = {
                "case_count": len(cases),
                "unit_count": len(units),
                "scan_eligible_unit_count": int(sum(
                    row["scan_eligible"] for row in units
                )),
                "cases_by_family": dict(Counter(
                    row["family"] for row in cases
                )),
                "units_by_family": dict(Counter(
                    row["family"] for row in units
                )),
                "scan_units_by_family": dict(Counter(
                    row["family"]
                    for row in units
                    if row["scan_eligible"]
                )),
                "exact_split_overlap_count": overlap_total,
                "all_identity_exact": all(
                    row["identity_exact"] for row in audits
                ),
                "all_semantic_gold_checks": all(
                    row["gold_change_matches_semantics"] for row in audits
                ),
            }
        global_summary["models"][model_name] = model_summary
        del tokenizer
    write_json(protocol_root / "summary.json", global_summary)
    print(json.dumps(global_summary, ensure_ascii=False, indent=2))
    return global_summary


if __name__ == "__main__":
    build_protocol()
