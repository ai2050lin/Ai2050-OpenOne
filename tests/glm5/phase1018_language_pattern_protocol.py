#!/usr/bin/env python3
"""Freeze the Phase1018 cross-pattern differential atlas protocol.

This protocol does not assume that rare words, punctuation, translation, and
contrast share one mechanism. It gives each family a controlled branch edit,
then asks which internal structures repeat within items, across held-out
surfaces, across items, and across families.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1009_crossfamily_response_protocol import digest


PHASE = 1018
PROTOCOL_REVISION = 3
MODELS = ("qwen3", "glm4", "deepseek7b")
PROMPT_MODES = ("raw", "native_chat")
FAMILIES = ("rare_semantics", "punctuation", "translation", "contrast")
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
WORLDS = tuple(range(4))
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
    / "phase1018_language_pattern_differential_atlas"
)


RARE_ITEMS = {
    "taotie": {
        "carrier": "饕餮",
        "labels": ("motif", "greed"),
        "discovery": (
            ("ancient bronze mask", "ritual vessel design"),
            ("insatiable appetite", "greedy banquet"),
        ),
        "confirmation": (
            ("Shang dynasty ornament", "symmetrical monster pattern"),
            ("voracious eating", "endless craving for food"),
        ),
    },
    "pixiu": {
        "carrier": "貔貅",
        "labels": ("beast", "charm"),
        "discovery": (
            ("winged legendary beast", "ancient mythical animal"),
            ("fortune charm", "shop wealth ornament"),
        ),
        "confirmation": (
            ("creature from old legends", "dragon-like beast"),
            ("feng-shui money symbol", "prosperity amulet"),
        ),
    },
    "qilin": {
        "carrier": "麒麟",
        "labels": ("beast", "talent"),
        "discovery": (
            ("benevolent legendary animal", "auspicious hoofed creature"),
            ("exceptionally talented child", "promising young person"),
        ),
        "confirmation": (
            ("mythical omen of peace", "sacred legendary beast"),
            ("remarkable offspring", "brilliant youngster"),
        ),
    },
    "xiezhi": {
        "carrier": "獬豸",
        "labels": ("beast", "justice"),
        "discovery": (
            ("one-horned beast judging disputes", "legendary animal detecting lies"),
            ("court badge of justice", "symbol used by legal institutions"),
        ),
        "confirmation": (
            ("creature that identifies guilt", "mythic judicial animal"),
            ("law-enforcement insignia", "emblem of impartial law"),
        ),
    },
    "bixi": {
        "carrier": "赑屃",
        "labels": ("beast", "pedestal"),
        "discovery": (
            ("one of the dragon sons", "powerful turtle-like creature"),
            ("base carrying a monument", "stone support under an inscription"),
        ),
        "confirmation": (
            ("legendary child of a dragon", "mythical burden-bearing beast"),
            ("pedestal beneath a memorial tablet", "carved base of a stele"),
        ),
    },
    "chiwen": {
        "carrier": "螭吻",
        "labels": ("beast", "ornament"),
        "discovery": (
            ("water-loving dragon son", "legendary dragon creature"),
            ("ceramic figure on a roof ridge", "palace roof decoration"),
        ),
        "confirmation": (
            ("mythical child of the dragon", "ancient aquatic beast"),
            ("ornament at the end of a ridge", "fire-preventing roof figure"),
        ),
    },
    "zhigu": {
        "carrier": "桎梏",
        "labels": ("chains", "constraint"),
        "discovery": (
            ("wooden restraints on a prisoner", "chains binding hands and feet"),
            ("restriction on creativity", "social limitation"),
        ),
        "confirmation": (
            ("literal instruments of confinement", "old penal fetters"),
            ("constraint on thought", "metaphorical bondage"),
        ),
    },
    "fanlong": {
        "carrier": "樊笼",
        "labels": ("cage", "restriction"),
        "discovery": (
            ("enclosure holding a bird", "wooden cage"),
            ("oppressive social environment", "confining way of life"),
        ),
        "confirmation": (
            ("literal aviary enclosure", "cage with bars"),
            ("metaphor for lost freedom", "restrictive institution"),
        ),
    },
    "guinie": {
        "carrier": "圭臬",
        "labels": ("instrument", "standard"),
        "discovery": (
            ("ancient device measuring shadows", "instrument used for surveying"),
            ("authoritative rule", "model everyone follows"),
        ),
        "confirmation": (
            ("old astronomical measuring tool", "gnomon-like device"),
            ("recognized criterion", "guiding norm"),
        ),
    },
    "fusang": {
        "carrier": "扶桑",
        "labels": ("tree", "Japan"),
        "discovery": (
            ("legendary tree where the sun rises", "mythical eastern tree"),
            ("poetic reference to Japan", "old literary name for an island nation"),
        ),
        "confirmation": (
            ("cosmic tree in ancient mythology", "sunrise tree of legend"),
            ("classical Chinese name for Japan", "literary eastern country"),
        ),
    },
    "qingniao": {
        "carrier": "青鸟",
        "labels": ("bird", "messenger"),
        "discovery": (
            ("a bird with blue feathers", "literal small blue bird"),
            ("divine messenger carrying news", "symbol of delivered letters"),
        ),
        "confirmation": (
            ("an actual azure bird", "blue-feathered animal"),
            ("mythic courier", "metaphor for a messenger"),
        ),
    },
    "honghu": {
        "carrier": "鸿鹄",
        "labels": ("bird", "ambition"),
        "discovery": (
            ("swan-like migratory bird", "large bird over a lake"),
            ("great aspiration", "ambition for high achievement"),
        ),
        "confirmation": (
            ("literal wild swan", "large flying waterfowl"),
            ("metaphor for a grand goal", "lofty personal ideal"),
        ),
    },
}


PUNCTUATION_ITEMS = {
    "question_archive": {
        "subtype": "statement_question",
        "before": "The archive is open",
        "after": "",
        "carriers": (".", "?"),
        "labels": ("statement", "question"),
    },
    "question_train": {
        "subtype": "statement_question",
        "before": "The last train has arrived",
        "after": "",
        "carriers": (".", "?"),
        "labels": ("statement", "question"),
    },
    "question_meeting": {
        "subtype": "statement_question",
        "before": "The meeting starts at noon",
        "after": "",
        "carriers": (".", "?"),
        "labels": ("statement", "question"),
    },
    "question_key": {
        "subtype": "statement_question",
        "before": "This key opens the cabinet",
        "after": "",
        "carriers": (".", "?"),
        "labels": ("statement", "question"),
    },
    "colon_sensor": {
        "subtype": "separation_explanation",
        "before": "The cause was clear",
        "after": "the sensor failed",
        "carriers": (".", ":"),
        "labels": ("separation", "explanation"),
    },
    "colon_delay": {
        "subtype": "separation_explanation",
        "before": "There was one reason for the delay",
        "after": "the bridge was closed",
        "carriers": (".", ":"),
        "labels": ("separation", "explanation"),
    },
    "colon_result": {
        "subtype": "separation_explanation",
        "before": "The result was immediate",
        "after": "the alarm sounded",
        "carriers": (".", ":"),
        "labels": ("separation", "explanation"),
    },
    "colon_goal": {
        "subtype": "separation_explanation",
        "before": "She had one goal",
        "after": "finish the manuscript",
        "carriers": (".", ":"),
        "labels": ("separation", "explanation"),
    },
    "semicolon_storm": {
        "subtype": "weak_strong_boundary",
        "before": "The storm ended",
        "after": "the boats returned",
        "carriers": (",", ";"),
        "labels": ("weak", "strong"),
    },
    "semicolon_bell": {
        "subtype": "weak_strong_boundary",
        "before": "The bell rang",
        "after": "the students entered",
        "carriers": (",", ";"),
        "labels": ("weak", "strong"),
    },
    "semicolon_sun": {
        "subtype": "weak_strong_boundary",
        "before": "The sun set",
        "after": "the streets grew quiet",
        "carriers": (",", ";"),
        "labels": ("weak", "strong"),
    },
    "semicolon_server": {
        "subtype": "weak_strong_boundary",
        "before": "The server restarted",
        "after": "the service recovered",
        "carriers": (",", ";"),
        "labels": ("weak", "strong"),
    },
}


TRANSLATION_BASE = (
    ("red", "红"),
    ("blue", "蓝"),
    ("water", "水"),
    ("fire", "火"),
    ("book", "书"),
    ("person", "人"),
)
TRANSLATION_ITEMS = {}
for english, chinese in TRANSLATION_BASE:
    TRANSLATION_ITEMS[f"{english}_en_zh"] = {
        "concept": english,
        "direction": "en_to_zh",
        "source": english,
        "target": chinese,
        "source_language": "English",
        "target_language": "Chinese",
        "labels": (english, chinese),
    }
    TRANSLATION_ITEMS[f"{english}_zh_en"] = {
        "concept": english,
        "direction": "zh_to_en",
        "source": chinese,
        "target": english,
        "source_language": "Chinese",
        "target_language": "English",
        "labels": (chinese, english),
    }


CONTRAST_ITEMS = {
    "room": ("The room was small", "it felt comfortable"),
    "exam": ("The exam was difficult", "Mira remained calm"),
    "weather": ("The sky was cloudy", "the afternoon stayed warm"),
    "engine": ("The engine was old", "it ran quietly"),
    "book": ("The book was long", "the argument stayed clear"),
    "road": ("The road was narrow", "traffic moved quickly"),
    "meal": ("The meal was simple", "the guests were delighted"),
    "team": ("The team was inexperienced", "it won the match"),
    "phone": ("The phone was inexpensive", "the camera was excellent"),
    "house": ("The house was remote", "the internet was reliable"),
    "lecture": ("The lecture was technical", "the audience followed it"),
    "garden": ("The garden was tiny", "it contained many species"),
}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class SpanBuilder:
    parts: list[str] = field(default_factory=list)
    spans: dict[str, tuple[int, int, str]] = field(default_factory=dict)
    length: int = 0

    def add(self, text: str) -> None:
        self.parts.append(text)
        self.length += len(text)

    def mark(self, role: str, text: str, prefix: str = " ") -> None:
        if role in self.spans:
            raise RuntimeError(f"duplicate role {role}")
        segment = prefix + text
        start = self.length
        self.add(segment)
        self.spans[role] = (start, self.length, segment)

    def finish(self) -> tuple[str, dict[str, tuple[int, int, str]]]:
        return "".join(self.parts), dict(self.spans)


def state_factors(state: str) -> tuple[int, int]:
    if state == "identity":
        return 0, 0
    return int(state[1]), int(state[-1])


def ordered_choices(
    labels: tuple[str, str],
    world: int,
) -> list[str]:
    order = (0, 1) if world % 2 == 0 else (1, 0)
    return [labels[index] for index in order]


def add_choices(
    builder: SpanBuilder,
    choices: list[str],
    world: int,
    template: int,
) -> None:
    if world < 2:
        builder.add(f" Choices: {choices[0]} or {choices[1]}.")
    else:
        builder.add(f" Candidate words: {choices[0]} / {choices[1]}.")


def add_answer_instruction(builder: SpanBuilder) -> None:
    builder.add(
        " Return exactly one displayed choice, with no explanation. Answer:"
    )


def render_rare(
    item_id: str,
    split: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    spec = RARE_ITEMS[item_id]
    cue = spec[split][branch][lexical]
    choices = ordered_choices(spec["labels"], world)
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Pattern")
    if template % 2 == 0:
        builder.add(" context:")
        builder.mark("context_anchor", cue)
        builder.add(". Focus term:")
        builder.mark("carrier", spec["carrier"])
        builder.add(".")
        builder.mark("operator", "Interpret")
        builder.mark("query_anchor", "reading")
    else:
        builder.add(" record. The clue is")
        builder.mark("context_anchor", cue)
        builder.add("; the Chinese term is")
        builder.mark("carrier", spec["carrier"])
        builder.add(".")
        builder.mark("operator", "Classify")
        builder.mark("query_anchor", "sense")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, spec["labels"][branch], {
        "subgroup": "rare_word",
        "carrier_text": spec["carrier"],
        "branch_labels": list(spec["labels"]),
        "cue": cue,
    }


def render_punctuation(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    spec = PUNCTUATION_ITEMS[item_id]
    carrier = spec["carriers"][branch]
    choices = ordered_choices(spec["labels"], world)
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Pattern")
    builder.add(" text:")
    builder.add(" " + spec["before"])
    builder.mark("carrier", carrier, prefix="")
    if spec["after"]:
        builder.mark("context_anchor", spec["after"])
    else:
        builder.mark("context_anchor", "sentence")
    builder.add(".")
    operator = (
        ("Classify", "Identify")[lexical]
        if template % 2 == 0
        else ("Determine", "Recognize")[lexical]
    )
    builder.mark("operator", operator)
    builder.mark("query_anchor", "punctuation")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, spec["labels"][branch], {
        "subgroup": spec["subtype"],
        "carrier_text": carrier,
        "branch_labels": list(spec["labels"]),
    }


def render_translation(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    spec = TRANSLATION_ITEMS[item_id]
    choices = ordered_choices(spec["labels"], world)
    if branch == 0:
        operator = ("copy", "repeat")[lexical]
        output_language = spec["source_language"]
    else:
        operator = ("translate", "convert")[lexical]
        output_language = spec["target_language"]
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Pattern")
    if template % 2 == 0:
        builder.add(" source:")
        builder.mark("carrier", spec["source"])
        builder.add(".")
        builder.mark("operator", operator)
        builder.add(" it into")
        builder.mark("context_anchor", output_language)
        builder.add(".")
        builder.mark("query_anchor", "output")
    else:
        builder.add(" language entry:")
        builder.mark("carrier", spec["source"])
        builder.add(". The requested operation is")
        builder.mark("operator", operator)
        builder.add("; requested language:")
        builder.mark("context_anchor", output_language)
        builder.add(".")
        builder.mark("query_anchor", "result")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, spec["labels"][branch], {
        "subgroup": spec["direction"],
        "carrier_text": spec["source"],
        "branch_labels": list(spec["labels"]),
        "concept": spec["concept"],
        "source_language": spec["source_language"],
        "target_language": spec["target_language"],
    }


def render_contrast(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    left, right = CONTRAST_ITEMS[item_id]
    labels = ("addition", "contrast")
    choices = ordered_choices(labels, world)
    if lexical == 0:
        connector = "and" if branch == 0 else "but"
        left_suffix = ","
        connector_suffix = ""
    else:
        connector = "additionally" if branch == 0 else "however"
        left_suffix = ";"
        connector_suffix = ","
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Pattern")
    builder.add(" text: " + left + left_suffix)
    builder.mark("carrier", connector)
    builder.add(connector_suffix)
    builder.mark("context_anchor", right)
    builder.add(".")
    operator = "Classify" if template % 2 == 0 else "Identify"
    builder.mark("operator", operator)
    builder.mark("query_anchor", "relation")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, labels[branch], {
        "subgroup": "additive_vs_contrast",
        "carrier_text": connector,
        "branch_labels": list(labels),
    }


def render_case(
    *,
    family: str,
    item_id: str,
    split: str,
    template: int,
    world: int,
    state: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    branch, lexical = state_factors(state)
    if family == "rare_semantics":
        return render_rare(
            item_id, split, template, world, branch, lexical
        )
    if family == "punctuation":
        return render_punctuation(
            item_id, template, world, branch, lexical
        )
    if family == "translation":
        return render_translation(
            item_id, template, world, branch, lexical
        )
    if family == "contrast":
        return render_contrast(
            item_id, template, world, branch, lexical
        )
    raise KeyError(family)


def item_ids(family: str) -> tuple[str, ...]:
    if family == "rare_semantics":
        return tuple(RARE_ITEMS)
    if family == "punctuation":
        return tuple(PUNCTUATION_ITEMS)
    if family == "translation":
        return tuple(TRANSLATION_ITEMS)
    if family == "contrast":
        return tuple(CONTRAST_ITEMS)
    raise KeyError(family)


def token_spans(
    tokenizer,
    rendered: str,
    raw_prompt: str,
    spans: dict[str, tuple[int, int, str]],
) -> dict[str, tuple[int, int]]:
    raw_start = rendered.index(raw_prompt)
    full_ids = tokenizer.encode(rendered, add_special_tokens=False)
    result = {}
    for role, (start, end, marked_text) in spans.items():
        before = tokenizer.encode(
            rendered[:raw_start + start],
            add_special_tokens=False,
        )
        through = tokenizer.encode(
            rendered[:raw_start + end],
            add_special_tokens=False,
        )
        if through[:len(before)] != before:
            raise RuntimeError(f"{role}: token prefix drift")
        added = through[len(before):]
        if not added:
            raise RuntimeError(f"{role}: empty token span for {marked_text!r}")
        if [int(value) for value in full_ids[len(before):len(through)]] != [
            int(value) for value in added
        ]:
            raise RuntimeError(f"{role}: full token span drift")
        result[role] = (len(before), len(through) - 1)
    return result


def continuation_token_ids(
    tokenizer,
    rendered: str,
    label: str,
) -> list[int]:
    base = tokenizer.encode(rendered, add_special_tokens=False)
    extended = tokenizer.encode(
        rendered + " " + label,
        add_special_tokens=False,
    )
    if extended[:len(base)] != base or len(extended) <= len(base):
        raise RuntimeError(f"candidate continuation drift for {label!r}")
    return [int(value) for value in extended[len(base):]]


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
        split=split,
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
        "schema_version": "phase1018_language_pattern_case.v1",
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


def edit_positions(left: list[int], right: list[int]) -> list[int]:
    if len(left) != len(right):
        return []
    return [
        index
        for index, (a, b) in enumerate(zip(left, right))
        if int(a) != int(b)
    ]


def audit_unit(
    unit: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = {
        state: by_id[unit["record_ids"][state]]
        for state in STATES
    }
    identity_exact = (
        cases["identity"]["input_ids"] == cases["b0_l0"]["input_ids"]
    )
    prefix_positions = {
        case["role_positions"]["prefix_anchor"]
        for case in cases.values()
    }
    prefix_tokens = {
        case["input_ids"][case["role_positions"]["prefix_anchor"]]
        for case in cases.values()
    }
    candidate_unique = all(
        len({
            tuple(values)
            for values in case["candidate_token_ids"].values()
        }) == 2
        for case in cases.values()
    )
    branch_pairs = (
        ("b0_l0", "b1_l0"),
        ("b0_l1", "b1_l1"),
    )
    branch_change = []
    no_prefix_edit = True
    for left_name, right_name in branch_pairs:
        left = cases[left_name]
        right = cases[right_name]
        edits = edit_positions(left["input_ids"], right["input_ids"])
        branch_change.append({
            "left": left_name,
            "right": right_name,
            "equal_length": len(left["input_ids"]) == len(right["input_ids"]),
            "edit_count_if_aligned": len(edits),
        })
        common_prefix = min(
            left["role_positions"]["prefix_anchor"],
            right["role_positions"]["prefix_anchor"],
        )
        no_prefix_edit &= (
            left["input_ids"][:common_prefix + 1]
            == right["input_ids"][:common_prefix + 1]
        )
    return {
        "schema_version": "phase1018_protocol_unit_audit.v1",
        "phase": PHASE,
        "model": unit["model"],
        "prompt_mode": unit["prompt_mode"],
        "family": unit["family"],
        "item_id": unit["item_id"],
        "split": unit["split"],
        "template": unit["template"],
        "world": unit["world"],
        "unit_id": unit["unit_id"],
        "identity_exact": identity_exact,
        "prefix_position_stable": len(prefix_positions) == 1,
        "prefix_token_stable": len(prefix_tokens) == 1,
        "branch_preserves_causal_prefix": no_prefix_edit,
        "candidate_ids_unique": candidate_unique,
        "branch_change": branch_change,
        "carrier_token_counts": {
            state: int(case["carrier_token_count"])
            for state, case in cases.items()
        },
        "lengths": {
            state: len(case["input_ids"]) for state, case in cases.items()
        },
    }


def build_protocol() -> dict[str, Any]:
    protocol_root = OUT_ROOT / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "schema_version": "phase1018_preregistration.v1",
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
        "primary_descriptive_thresholds": {
            "direction_consistency": 0.45,
            "surface_alignment": 0.40,
            "minimum_normalized_magnitude": 1e-4,
        },
        "threshold_grid": {
            "direction_consistency": [0.30, 0.45, 0.60],
            "surface_alignment": [0.20, 0.40, 0.60],
        },
        "primary_questions": [
            "Which response structures repeat across held-out surfaces?",
            "Does matched discovery-confirmation alignment exceed mismatched-item alignment?",
            "Which physical resources repeat within and across language-pattern families?",
            "Where does each branch edit first become visible under causal order?",
        ],
        "claim_limits": [
            "Factorial differences are measurements, not mechanism equations.",
            "Physical reuse is not direction reuse or a causal edge.",
            "Rare-word prompts test contextual interpretation, not complete lexical knowledge.",
            "Punctuation, translation, and contrast are separate families unless data show shared structure.",
            "No neuron or causal claim is allowed from the atlas alone.",
        ],
        "automatic_continuation_rule": {
            "targeted_causal_test_only_if": [
                "candidate accuracy >= 0.70 in at least two models for the family",
                "matched-minus-mismatched direction gap >= 0.15 in at least two models",
                "a discovery-selected physical event repeats in confirmation",
            ],
            "otherwise": "stop after descriptive atlas and retain claim boundary",
        },
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
                                for state in STATES:
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
                                    cases.append(case)
                                    record_ids[state] = case["record_id"]
                                units.append({
                                    "schema_version": "phase1018_pattern_unit.v1",
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
                                })
            by_id = {case["record_id"]: case for case in cases}
            audits = [audit_unit(unit, by_id) for unit in units]
            if not all(
                row["identity_exact"]
                and row["prefix_token_stable"]
                and row["branch_preserves_causal_prefix"]
                and row["candidate_ids_unique"]
                for row in audits
            ):
                raise RuntimeError(
                    f"protocol audit failed for {model_name}/{prompt_mode}"
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
            rare_counts = Counter(
                case["carrier_token_count"]
                for case in cases
                if case["family"] == "rare_semantics"
            )
            taotie_counts = Counter(
                case["carrier_token_count"]
                for case in cases
                if case["family"] == "rare_semantics"
                and case["item_id"] == "taotie"
            )
            model_summary[prompt_mode] = {
                "case_count": len(cases),
                "unit_count": len(units),
                "cases_by_family": dict(Counter(
                    case["family"] for case in cases
                )),
                "units_by_family": dict(Counter(
                    unit["family"] for unit in units
                )),
                "rare_carrier_token_count_distribution": dict(rare_counts),
                "taotie_token_count_distribution": dict(taotie_counts),
                "minimum_length": min(len(case["input_ids"]) for case in cases),
                "maximum_length": max(len(case["input_ids"]) for case in cases),
                "all_identity_exact": all(
                    row["identity_exact"] for row in audits
                ),
                "all_branch_prefixes_preserved": all(
                    row["branch_preserves_causal_prefix"] for row in audits
                ),
            }
        global_summary["models"][model_name] = model_summary
        del tokenizer
    write_json(protocol_root / "summary.json", global_summary)
    print(json.dumps(global_summary, ensure_ascii=False, indent=2))
    return global_summary


if __name__ == "__main__":
    build_protocol()
