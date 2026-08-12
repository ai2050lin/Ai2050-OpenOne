#!/usr/bin/env python3
"""Freeze a cross-pattern FP16 behavior and response-atlas protocol."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior


PHASE = 1065
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
FAMILIES = (
    "contrast",
    "punctuation",
    "taxonomy",
    "transitive_reasoning",
    "rare_semantics",
    "translation_word",
)
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
STATES = ("b0_l0", "b1_l0", "b0_l1", "b1_l1")
CAPTURE_ROLES = (
    "source_primary",
    "source_secondary",
    "operator",
    "query",
    "answer_boundary",
)
ASSISTANT_PREFILL = "Answer:"
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1065_multimode_response_atlas"
)
SOURCE_PHASE1064 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1064_cross_panel_transport"
    / "aggregate.json"
)
GATES = {
    "candidate_first_token_accuracy_min": 0.80,
    "valid_semantic_pair_min": 80,
    "valid_semantic_pair_per_split_min": 35,
    "strong_valid_semantic_pair_min": 100,
    "natural_audit_exact_rate_min": 0.70,
    "minimum_repeated_models": 2,
    "internal_discovery_confirmation_cosine_min": 0.40,
    "cross_model_depth_profile_cosine_min": 0.75,
}
NATURAL_AUDIT_CASES_PER_FAMILY = 12
NATURAL_GENERATION_STEPS = 6


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


CONTRAST_ITEMS = (
    ("room", "The room was small", "it felt cramped", "it felt comfortable"),
    ("exam", "The exam was difficult", "Mira struggled", "Mira stayed calm"),
    ("sky", "The sky was cloudy", "the day looked gloomy", "the air stayed warm"),
    ("engine", "The engine was old", "it failed often", "it ran quietly"),
    ("book", "The book was long", "reading took weeks", "the argument stayed clear"),
    ("road", "The road was narrow", "traffic moved slowly", "traffic moved quickly"),
    ("meal", "The meal was simple", "the flavors were plain", "the guests loved it"),
    ("team", "The team was inexperienced", "it lost badly", "it won the match"),
    ("phone", "The phone was inexpensive", "its camera was basic", "its camera was excellent"),
    ("house", "The house was remote", "the connection was unreliable", "the connection was reliable"),
    ("lecture", "The lecture was technical", "the audience was confused", "the audience followed it"),
    ("garden", "The garden was tiny", "it held few species", "it held many species"),
    ("coat", "The coat was thin", "it offered little warmth", "it kept Ana warm"),
    ("task", "The task was complex", "the team needed help", "the team finished early"),
    ("river", "The river was shallow", "boats moved with difficulty", "the ferry crossed easily"),
)

PUNCTUATION_ITEMS = (
    ("archive", "The archive is open", "Is the archive open"),
    ("train", "The last train has arrived", "Has the last train arrived"),
    ("meeting", "The meeting begins at noon", "Does the meeting begin at noon"),
    ("key", "This key opens the cabinet", "Does this key open the cabinet"),
    ("server", "The server is online", "Is the server online"),
    ("bridge", "The bridge is safe", "Is the bridge safe"),
    ("museum", "The museum closes at six", "Does the museum close at six"),
    ("parcel", "The parcel has arrived", "Has the parcel arrived"),
    ("alarm", "The alarm is active", "Is the alarm active"),
    ("gate", "The gate remains locked", "Does the gate remain locked"),
    ("report", "The report is complete", "Is the report complete"),
    ("flight", "The flight leaves tonight", "Does the flight leave tonight"),
    ("lamp", "The lamp still works", "Does the lamp still work"),
    ("water", "The water is warm", "Is the water warm"),
    ("clinic", "The clinic opens tomorrow", "Does the clinic open tomorrow"),
)

TAXONOMY_ITEMS = (
    ("apple_dog", "apple", "dog"),
    ("pear_cat", "pear", "cat"),
    ("banana_horse", "banana", "horse"),
    ("orange_cow", "orange", "cow"),
    ("grape_sheep", "grape", "sheep"),
    ("strawberry_goat", "strawberry", "goat"),
    ("cherry_lion", "cherry", "lion"),
    ("peach_tiger", "peach", "tiger"),
    ("mango_bear", "mango", "bear"),
    ("lemon_wolf", "lemon", "wolf"),
    ("pineapple_fox", "pineapple", "fox"),
    ("watermelon_rabbit", "watermelon", "rabbit"),
    ("blueberry_elephant", "blueberry", "elephant"),
    ("plum_giraffe", "plum", "giraffe"),
    ("apricot_zebra", "apricot", "zebra"),
)

REASONING_ITEMS = (
    ("ava_ben_cora", "Ava", "Ben", "Cora"),
    ("dara_eli_faye", "Dara", "Eli", "Faye"),
    ("gail_hugo_iris", "Gail", "Hugo", "Iris"),
    ("juno_kian_lena", "Juno", "Kian", "Lena"),
    ("mira_noah_opal", "Mira", "Noah", "Opal"),
    ("pavel_quinn_rhea", "Pavel", "Quinn", "Rhea"),
    ("sara_tomas_uma", "Sara", "Tomas", "Uma"),
    ("vera_wade_xena", "Vera", "Wade", "Xena"),
    ("yara_zane_amber", "Yara", "Zane", "Amber"),
    ("bruno_celia_damon", "Bruno", "Celia", "Damon"),
    ("esme_felix_grace", "Esme", "Felix", "Grace"),
    ("henry_ines_jonas", "Henry", "Ines", "Jonas"),
    ("kara_leon_mabel", "Kara", "Leon", "Mabel"),
    ("nina_oscar_petra", "Nina", "Oscar", "Petra"),
    ("ravi_sonia_theo", "Ravi", "Sonia", "Theo"),
)

RARE_ITEMS = (
    ("taotie", "饕餮", "a mythical beast associated with gluttony", "gluttony", "impartial justice"),
    ("pixiu", "貔貅", "an auspicious creature associated with attracting wealth", "attracting wealth", "repairing the sky"),
    ("bixi", "赑屃", "a turtle-like dragon son depicted carrying stone steles", "carrying stone steles", "chasing the sun"),
    ("chiwen", "螭吻", "a dragon-like roof-ridge creature linked with protection from fire", "roof ridges", "filling the sea with stones"),
    ("kunpeng", "鲲鹏", "a giant transformed creature symbolizing vast ambition", "vast ambition", "legal imprisonment"),
    ("honghu", "鸿鹄", "a great wild bird used as a symbol of lofty aspiration", "lofty aspiration", "household cooking"),
    ("fusang", "扶桑", "a mythical eastern tree associated with the rising sun", "the rising sun", "a prison cell"),
    ("xiezhi", "獬豸", "a mythical creature associated with justice and law", "justice", "endless hesitation"),
    ("qilin", "麒麟", "an auspicious mythical creature associated with benevolence", "auspiciousness", "heavy rainfall"),
    ("baize", "白泽", "a mythical creature said to know supernatural beings", "knowledge of spirits", "carrying a stone monument"),
    ("jingwei", "精卫", "a mythical bird remembered for trying to fill the sea", "filling the sea", "guarding roof tiles"),
    ("kuafu", "夸父", "a mythic giant remembered for pursuing the sun", "pursuing the sun", "attracting wealth"),
    ("change", "嫦娥", "a mythic woman associated with the moon", "the moon", "impartial law"),
    ("nuwa", "女娲", "a creator goddess remembered for repairing the sky", "repairing the sky", "gluttony"),
    ("yinglong", "应龙", "a winged dragon associated with rain and ancient battles", "wings and rain", "a household meal"),
)

TRANSLATION_ITEMS = (
    ("red_blue", "red", "rouge", "blue", "bleu"),
    ("green_black", "green", "vert", "black", "noir"),
    ("white_yellow", "white", "blanc", "yellow", "jaune"),
    ("orange_purple", "orange", "orange", "purple", "violet"),
    ("pink_gray", "pink", "rose", "gray", "gris"),
    ("brown_beige", "brown", "marron", "beige", "beige"),
    ("gold_silver", "gold", "or", "silver", "argent"),
    ("red_green", "red", "rouge", "green", "vert"),
    ("blue_black", "blue", "bleu", "black", "noir"),
    ("white_orange", "white", "blanc", "orange", "orange"),
    ("yellow_purple", "yellow", "jaune", "purple", "violet"),
    ("pink_brown", "pink", "rose", "brown", "marron"),
    ("gray_beige", "gray", "gris", "beige", "beige"),
    ("gold_blue", "gold", "or", "blue", "bleu"),
    ("silver_green", "silver", "argent", "green", "vert"),
)


def state_factors(state: str) -> tuple[int, int]:
    return int(state[1]), int(state[-1])


def mark(
    text: str,
    value: str,
    *,
    occurrence: str = "first",
) -> tuple[int, int, str]:
    start = text.find(value) if occurrence == "first" else text.rfind(value)
    if start < 0:
        raise RuntimeError(f"missing marked value {value!r}")
    return start, start + len(value), value


def split_for_template(template_index: int) -> str:
    return "discovery" if template_index < 2 else "confirmation"


def contrast_case(
    item: tuple[str, str, str, str],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, premise, aligned, opposed = item
    conclusion = aligned if branch == 0 else opposed
    operator = ("Supply", "Write", "Provide", "Insert")[template_index]
    if lexical == 0:
        query = "___"
        text = (
            f"{operator} the single lowercase coordinating conjunction that "
            f"best fills the blank: {premise}, {query} {conclusion}\n"
            "Return only the conjunction."
        )
    else:
        query = "[blank]"
        text = (
            f"{operator} one lowercase coordinating conjunction for this "
            f"sentence: {premise}, {query} {conclusion}\n"
            "Give the conjunction alone."
        )
    spans = {
        "source_primary": mark(text, premise),
        "source_secondary": mark(text, conclusion),
        "operator": mark(text, operator),
        "query": mark(text, query),
    }
    return text, spans, {"b0": ["and"], "b1": ["but", "yet"]}, " "


def punctuation_case(
    item: tuple[str, str, str],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, statement, question = item
    carrier = statement if branch == 0 else question
    kind = "declarative sentence" if branch == 0 else "direct question"
    operator = ("Supply", "Write", "Provide", "Insert")[template_index]
    if lexical == 0:
        text = (
            f"{operator} only the missing final punctuation mark for this "
            f"{kind}: {carrier}"
        )
    else:
        text = (
            f"{operator} the one final punctuation mark required by the "
            f"following {kind}: {carrier}"
        )
    spans = {
        "source_primary": mark(text, carrier),
        "source_secondary": mark(text, kind),
        "operator": mark(text, operator),
        "query": mark(text, carrier, occurrence="last"),
    }
    return text, spans, {"b0": ["."], "b1": ["?"]}, " "


def taxonomy_case(
    item: tuple[str, str, str],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, fruit, animal = item
    entity = fruit if branch == 0 else animal
    operator = ("Classify", "Name", "Categorize", "Identify")[
        template_index
    ]
    category_cue = "broad biological category"
    if lexical == 0:
        text = (
            f"{operator} {entity} with one lowercase {category_cue}. "
            "Return only that category noun."
        )
    else:
        text = (
            f"{operator} the {category_cue} of {entity} using one lowercase "
            "noun and no explanation."
        )
    spans = {
        "source_primary": mark(text, entity),
        "source_secondary": mark(text, category_cue),
        "operator": mark(text, operator),
        "query": mark(text, entity, occurrence="last"),
    }
    return text, spans, {"b0": ["fruit"], "b1": ["animal"]}, " "


def reasoning_case(
    item: tuple[str, str, str, str],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, a, b, c = item
    if branch == 0:
        first = f"{a} is taller than {b}"
        second = f"{b} is taller than {c}"
    else:
        first = f"{a} is shorter than {b}"
        second = f"{b} is shorter than {c}"
    operator = ("Determine", "Infer", "Find", "Deduce")[template_index]
    query = "Who is tallest"
    if lexical == 0:
        text = (
            f"{first}. {second}. {operator} the answer: {query}? "
            "Return only the person's name."
        )
    else:
        text = (
            f"Given that {first.lower()} and {second.lower()}, {operator.lower()} "
            f"{query.lower()}. Give only the name."
        )
        operator = operator.lower()
        query = query.lower()
    spans = {
        "source_primary": mark(text, first if lexical == 0 else first.lower()),
        "source_secondary": mark(text, second if lexical == 0 else second.lower()),
        "operator": mark(text, operator),
        "query": mark(text, query),
    }
    return text, spans, {"b0": [a], "b1": [c]}, " "


def rare_case(
    item: tuple[str, str, str, str, str],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, term, definition, true_property, false_property = item
    source = term if lexical == 0 else definition
    prop = true_property if branch == 0 else false_property
    operator = ("Judge", "Decide", "Assess", "Evaluate")[template_index]
    query = f"Is {source} traditionally associated with {prop}"
    text = (
        f"{operator} the semantic claim. {query}? "
        "Return one lowercase English judgment word and nothing else."
    )
    spans = {
        "source_primary": mark(text, source),
        "source_secondary": mark(text, prop),
        "operator": mark(text, operator),
        "query": mark(text, query),
    }
    return text, spans, {"b0": ["yes"], "b1": ["no"]}, " "


def translation_case(
    item: tuple[str, str, str, str, str],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, en0, fr0, en1, fr1 = item
    source = en0 if branch == 0 else en1
    operator = ("Translate", "Render", "Convert", "Express")[
        template_index
    ]
    target = "French"
    if lexical == 0:
        text = (
            f"{operator} the English color word {source} into {target}. "
            "Return exactly one lowercase French word."
        )
    else:
        text = (
            f"{operator} {source} from English to {target}, giving only the "
            "single lowercase translated word."
        )
    spans = {
        "source_primary": mark(text, source),
        "source_secondary": mark(text, target),
        "operator": mark(text, operator),
        "query": mark(text, source, occurrence="last"),
    }
    return text, spans, {"b0": [fr0], "b1": [fr1]}, " "


BUILDERS = {
    "contrast": (CONTRAST_ITEMS, contrast_case),
    "punctuation": (PUNCTUATION_ITEMS, punctuation_case),
    "taxonomy": (TAXONOMY_ITEMS, taxonomy_case),
    "transitive_reasoning": (REASONING_ITEMS, reasoning_case),
    "rare_semantics": (RARE_ITEMS, rare_case),
    "translation_word": (TRANSLATION_ITEMS, translation_case),
}


def build_model_case(
    tokenizer,
    model_name: str,
    family: str,
    item: tuple[Any, ...],
    template_index: int,
    state: str,
    semantic_case_index: int,
) -> dict[str, Any]:
    branch, lexical = state_factors(state)
    _, builder = BUILDERS[family]
    raw_prompt, raw_spans, classes, prefix = builder(
        item, branch, lexical, template_index
    )
    rendered = behavior.render_native(
        tokenizer,
        model_name,
        raw_prompt,
        with_system=False,
    )
    rendered += ASSISTANT_PREFILL
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(
        tokenizer, rendered, raw_prompt, raw_spans
    )
    role_spans["answer_boundary"] = (
        len(input_ids) - 1,
        len(input_ids) - 1,
    )
    role_positions = {
        role: int(span[1]) for role, span in role_spans.items()
    }
    candidate_token_ids = {
        class_name: [
            behavior.continuation_ids(
                tokenizer, rendered, prefix, label
            )
            for label in labels
        ]
        for class_name, labels in classes.items()
    }
    candidate_first_token_ids = {
        class_name: sorted({
            int(values[0]) for values in tokenizations
        })
        for class_name, tokenizations in candidate_token_ids.items()
    }
    expected_class = f"b{branch}"
    return {
        "schema_version": "phase1065_multimode_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": semantic_case_index,
        "record_id": (
            f"{model_name}.{family}.{item[0]}.t{template_index}.{state}"
        ),
        "unit_id": f"{family}.{item[0]}.t{template_index}",
        "family": family,
        "item_id": str(item[0]),
        "split": split_for_template(template_index),
        "template_index": template_index,
        "state": state,
        "semantic_branch": branch,
        "lexical_branch": lexical,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": {
            role: [int(span[0]), int(span[1])]
            for role, span in role_spans.items()
        },
        "role_positions": role_positions,
        "candidate_labels": classes,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": candidate_first_token_ids,
        "expected_class": expected_class,
        "acceptable_labels": classes[expected_class],
        "continuation_prefix": prefix,
    }


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter((row["family"], row["split"]) for row in cases)
    unit_counts = Counter(
        (row["family"], row["split"])
        for row in cases
        if row["state"] == "b0_l0"
    )
    roles_valid = True
    candidate_disjoint = True
    state_groups: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        state_groups.setdefault(str(row["unit_id"]), []).append(row)
        width = len(row["input_ids"])
        for role in CAPTURE_ROLES:
            start, end = row["role_spans"][role]
            roles_valid = roles_valid and 0 <= start <= end < width
        left = set(row["candidate_first_token_ids"]["b0"])
        right = set(row["candidate_first_token_ids"]["b1"])
        candidate_disjoint = candidate_disjoint and bool(left) and bool(
            right
        ) and left.isdisjoint(right)
    complete_units = all(
        {row["state"] for row in values} == set(STATES)
        for values in state_groups.values()
    )
    split_prompts = {
        split: {
            row["rendered_prompt"]
            for row in cases
            if row["split"] == split
        }
        for split in SPLITS
    }
    checks = {
        "case_count": len(cases) == len(FAMILIES) * 15 * 4 * 4,
        "unit_count": len(state_groups) == len(FAMILIES) * 15 * 4,
        "family_split_case_counts": all(
            counts[(family, split)] == 120
            for family in FAMILIES
            for split in SPLITS
        ),
        "family_split_unit_counts": all(
            unit_counts[(family, split)] == 30
            for family in FAMILIES
            for split in SPLITS
        ),
        "complete_factorial_units": complete_units,
        "role_spans_valid": roles_valid,
        "candidate_first_tokens_disjoint": candidate_disjoint,
        "discovery_confirmation_prompt_disjoint": split_prompts[
            "discovery"
        ].isdisjoint(split_prompts["confirmation"]),
    }
    return {
        "schema_version": "phase1065_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(state_groups),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    if not SOURCE_PHASE1064.exists():
        raise RuntimeError("missing Phase1064 aggregate")
    source = read_json(SOURCE_PHASE1064)
    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = []
        semantic_case_index = 0
        for family in FAMILIES:
            items, _ = BUILDERS[family]
            for item in items:
                for template_index in range(4):
                    for state in STATES:
                        cases.append(build_model_case(
                            tokenizer,
                            model_name,
                            family,
                            item,
                            template_index,
                            state,
                            semantic_case_index,
                        ))
                        semantic_case_index += 1
        audit = audit_model(model_name, cases)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"protocol audit failed for {model_name}: {audit}"
            )
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl",
            cases,
        )
        write_json(
            OUT_ROOT / "protocol" / f"audit.{model_name}.json",
            audit,
        )
        model_audits[model_name] = audit

    payload = {
        "schema_version": "phase1065_multimode_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "families": list(FAMILIES),
        "splits": list(SPLITS),
        "templates_by_split": {
            key: list(values)
            for key, values in TEMPLATES_BY_SPLIT.items()
        },
        "states": list(STATES),
        "capture_roles": list(CAPTURE_ROLES),
        "assistant_prefill": ASSISTANT_PREFILL,
        "gates": dict(GATES),
        "natural_audit_cases_per_family": (
            NATURAL_AUDIT_CASES_PER_FAMILY
        ),
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "case_count_per_model": len(FAMILIES) * 15 * 4 * 4,
        "unit_count_per_model": len(FAMILIES) * 15 * 4,
        "semantic_pair_count_per_family_model": 120,
        "source_phase1064_digest": source["protocol_digest"],
        "source_phase1064_route": source["automatic_next_decision"],
        "measurement_order": [
            "freeze prompts, roles, candidate classes, and gates",
            "measure candidate-absent next-token behavior",
            "audit a stratified natural-generation subset",
            "map residual, attention-output, and MLP-output differences",
            "compare discovery versus confirmation templates",
            "compare normalized depth profiles across models",
            "only then select a family for independent causal work",
        ],
        "interpretation_limits": [
            "First-token candidate ranking is not full language behavior.",
            "A response difference is not a causal mechanism.",
            "Cross-template direction repeat is not semantic completeness.",
            "K/V transport from translation is not assumed for new families.",
            "The rare-term branch compares a word with a definition and is not a complete lexical representation.",
            "No result establishes neural or biological optimality.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "A non-translation family has strong behavior and "
                "cross-template internal repetition in at least two models."
            ),
            "next_phase": (
                "independent role-conditioned causal test for the "
                "automatically selected family"
            ),
        },
        "model_audits": model_audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json",
        payload,
    )
    write_json(
        OUT_ROOT / "protocol" / "audit.json",
        {
            "schema_version": "phase1065_protocol_audit.v1",
            "phase": PHASE,
            "protocol_digest": payload["protocol_digest"],
            "model_audits": model_audits,
            "all_checks_passed": all(
                row["all_checks_passed"]
                for row in model_audits.values()
            ),
        },
    )
    return payload


def main() -> None:
    payload = build_protocol()
    print(
        f"Phase{PHASE} protocol {payload['protocol_digest']} "
        f"cases={payload['case_count_per_model']}/model"
    )


if __name__ == "__main__":
    main()
