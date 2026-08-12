#!/usr/bin/env python3
"""Freeze the Phase1077 nonblocking multi-family response-atlas protocol."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior


PHASE = 1077
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
FAMILIES = (
    "height_polarity",
    "contrast",
    "punctuation",
    "taxonomy",
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
CONDITIONINGS = ("all_finite", "behavior_complete")
ASSISTANT_PREFILL = "Answer:"
NATURAL_AUDIT_CASES_PER_FAMILY_SPLIT = 8
NATURAL_GENERATION_STEPS = 8
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1077_nonblocking_pattern_atlas"
)
SOURCE_PHASE1065 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1065_multimode_response_atlas"
    / "aggregate.json"
)
SOURCE_PHASE1076 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1076_polarity_head_causal"
    / "analysis"
    / "final_summary.json"
)

# These thresholds assign descriptive evidence levels. They do not remove a
# family from the atlas and do not authorize a causal claim.
EVIDENCE_THRESHOLDS = {
    "candidate_accuracy_for_behavior_annotation": 0.80,
    "natural_semantic_first_for_behavior_annotation": 0.60,
    "within_model_split_profile_cosine_l1": 0.85,
    "cross_model_raw_profile_cosine_l2": 0.75,
    "cross_model_centered_profile_cosine_l3": 0.25,
    "minimum_repeated_models_or_pairs": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


HEIGHT_DISCOVERY = (
    ("hd01", "Ava", "Ben", "Cora"),
    ("hd02", "Dara", "Eli", "Faye"),
    ("hd03", "Gail", "Hugo", "Iris"),
    ("hd04", "Juno", "Kian", "Lena"),
    ("hd05", "Mira", "Noah", "Opal"),
    ("hd06", "Pavel", "Quinn", "Rhea"),
    ("hd07", "Sara", "Tomas", "Uma"),
    ("hd08", "Vera", "Wade", "Xena"),
    ("hd09", "Yara", "Zane", "Amber"),
    ("hd10", "Bruno", "Celia", "Damon"),
    ("hd11", "Esme", "Felix", "Grace"),
    ("hd12", "Henry", "Ines", "Jonas"),
    ("hd13", "Kara", "Leon", "Mabel"),
    ("hd14", "Nina", "Oscar", "Petra"),
    ("hd15", "Ravi", "Sonia", "Theo"),
)
HEIGHT_CONFIRMATION = (
    ("hc01", "Adrian", "Bianca", "Caleb"),
    ("hc02", "Delia", "Emil", "Flora"),
    ("hc03", "Gideon", "Helena", "Ivan"),
    ("hc04", "Julia", "Kai", "Lucia"),
    ("hc05", "Mateo", "Nora", "Orin"),
    ("hc06", "Priya", "Reuben", "Selma"),
    ("hc07", "Talia", "Umar", "Violet"),
    ("hc08", "Willa", "Xavier", "Yvette"),
    ("hc09", "Zelda", "Aaron", "Beatrice"),
    ("hc10", "Cedric", "Diana", "Evan"),
    ("hc11", "Freya", "Gavin", "Hazel"),
    ("hc12", "Isabel", "Jasper", "Keira"),
    ("hc13", "Liam", "Marina", "Nolan"),
    ("hc14", "Olive", "Pedro", "Rosa"),
    ("hc15", "Silas", "Tessa", "Victor"),
)

CONTRAST_DISCOVERY = (
    ("cd01", "The room was small", "it felt cramped", "it felt comfortable"),
    ("cd02", "The exam was difficult", "Mira struggled", "Mira stayed calm"),
    ("cd03", "The sky was cloudy", "the day looked gloomy", "the air stayed warm"),
    ("cd04", "The engine was old", "it failed often", "it ran quietly"),
    ("cd05", "The book was long", "reading took weeks", "the argument stayed clear"),
    ("cd06", "The road was narrow", "traffic moved slowly", "traffic moved quickly"),
    ("cd07", "The meal was simple", "the flavors were plain", "the guests loved it"),
    ("cd08", "The team was inexperienced", "it lost badly", "it won the match"),
    ("cd09", "The phone was inexpensive", "its camera was basic", "its camera was excellent"),
    ("cd10", "The house was remote", "the connection was unreliable", "the connection was reliable"),
    ("cd11", "The lecture was technical", "the audience was confused", "the audience followed it"),
    ("cd12", "The garden was tiny", "it held few species", "it held many species"),
    ("cd13", "The coat was thin", "it offered little warmth", "it kept Ana warm"),
    ("cd14", "The task was complex", "the team needed help", "the team finished early"),
    ("cd15", "The river was shallow", "boats moved with difficulty", "the ferry crossed easily"),
)
CONTRAST_CONFIRMATION = (
    ("cc01", "The bag was heavy", "Rina walked slowly", "Rina arrived early"),
    ("cc02", "The coffee was cold", "Noel disliked it", "Noel finished it"),
    ("cc03", "The map was old", "several roads were missing", "the route was accurate"),
    ("cc04", "The screen was cracked", "the image looked distorted", "the text stayed readable"),
    ("cc05", "The hill was steep", "the climb was exhausting", "the child climbed easily"),
    ("cc06", "The budget was small", "the options were limited", "the project succeeded"),
    ("cc07", "The queue was long", "the wait took an hour", "service moved quickly"),
    ("cc08", "The fabric was rough", "it irritated the skin", "the jacket felt comfortable"),
    ("cc09", "The signal was weak", "calls often dropped", "the video remained clear"),
    ("cc10", "The room was dark", "details were hard to see", "the photograph was sharp"),
    ("cc11", "The deadline was close", "the group felt rushed", "the group checked every detail"),
    ("cc12", "The soil was dry", "the seedlings wilted", "the flowers remained healthy"),
    ("cc13", "The recipe was unusual", "the flavor seemed strange", "the guests requested seconds"),
    ("cc14", "The bicycle was rusty", "the chain slipped", "the ride was smooth"),
    ("cc15", "The instructions were brief", "several steps were unclear", "everyone completed the task"),
)

PUNCTUATION_DISCOVERY = (
    ("pd01", "The archive is open", "Is the archive open"),
    ("pd02", "The last train has arrived", "Has the last train arrived"),
    ("pd03", "The meeting begins at noon", "Does the meeting begin at noon"),
    ("pd04", "This key opens the cabinet", "Does this key open the cabinet"),
    ("pd05", "The server is online", "Is the server online"),
    ("pd06", "The bridge is safe", "Is the bridge safe"),
    ("pd07", "The museum closes at six", "Does the museum close at six"),
    ("pd08", "The parcel has arrived", "Has the parcel arrived"),
    ("pd09", "The alarm is active", "Is the alarm active"),
    ("pd10", "The gate remains locked", "Does the gate remain locked"),
    ("pd11", "The report is complete", "Is the report complete"),
    ("pd12", "The flight leaves tonight", "Does the flight leave tonight"),
    ("pd13", "The lamp still works", "Does the lamp still work"),
    ("pd14", "The water is warm", "Is the water warm"),
    ("pd15", "The clinic opens tomorrow", "Does the clinic open tomorrow"),
)
PUNCTUATION_CONFIRMATION = (
    ("pc01", "The library is quiet", "Is the library quiet"),
    ("pc02", "The oven has cooled", "Has the oven cooled"),
    ("pc03", "The elevator stops here", "Does the elevator stop here"),
    ("pc04", "The concert starts at eight", "Does the concert start at eight"),
    ("pc05", "The package is sealed", "Is the package sealed"),
    ("pc06", "The battery is charged", "Is the battery charged"),
    ("pc07", "The window faces east", "Does the window face east"),
    ("pc08", "The cafe serves breakfast", "Does the cafe serve breakfast"),
    ("pc09", "The harbor is calm", "Is the harbor calm"),
    ("pc10", "The ticket remains valid", "Does the ticket remain valid"),
    ("pc11", "The classroom is empty", "Is the classroom empty"),
    ("pc12", "The printer has stopped", "Has the printer stopped"),
    ("pc13", "The hotel accepts pets", "Does the hotel accept pets"),
    ("pc14", "The pharmacy closes soon", "Does the pharmacy close soon"),
    ("pc15", "The tunnel is open", "Is the tunnel open"),
)

TAXONOMY_DISCOVERY = (
    ("td01", "apple", "dog"),
    ("td02", "pear", "cat"),
    ("td03", "banana", "horse"),
    ("td04", "orange", "cow"),
    ("td05", "grape", "sheep"),
    ("td06", "strawberry", "goat"),
    ("td07", "cherry", "lion"),
    ("td08", "peach", "tiger"),
    ("td09", "mango", "bear"),
    ("td10", "lemon", "wolf"),
    ("td11", "pineapple", "fox"),
    ("td12", "watermelon", "rabbit"),
    ("td13", "blueberry", "elephant"),
    ("td14", "plum", "giraffe"),
    ("td15", "apricot", "zebra"),
)
TAXONOMY_CONFIRMATION = (
    ("tc01", "kiwi", "deer"),
    ("tc02", "coconut", "monkey"),
    ("tc03", "fig", "eagle"),
    ("tc04", "pomegranate", "shark"),
    ("tc05", "papaya", "otter"),
    ("tc06", "guava", "panda"),
    ("tc07", "lychee", "camel"),
    ("tc08", "raspberry", "dolphin"),
    ("tc09", "blackberry", "kangaroo"),
    ("tc10", "cranberry", "penguin"),
    ("tc11", "dragonfruit", "leopard"),
    ("tc12", "passionfruit", "rhinoceros"),
    ("tc13", "persimmon", "owl"),
    ("tc14", "nectarine", "buffalo"),
    ("tc15", "grapefruit", "seal"),
)

RARE_DISCOVERY = (
    ("rd01", "\u9955\u992e", "a mythical beast associated with gluttony", "gluttony", "impartial justice"),
    ("rd02", "\u8c94\u8c85", "an auspicious creature associated with attracting wealth", "attracting wealth", "repairing the sky"),
    ("rd03", "\u8d51\u5c43", "a turtle-like dragon son depicted carrying stone steles", "carrying stone steles", "chasing the sun"),
    ("rd04", "\u87ad\u543b", "a dragon-like roof-ridge creature linked with protection from fire", "roof ridges", "filling the sea"),
    ("rd05", "\u9cb2\u9e4f", "a giant transformed creature symbolizing vast ambition", "vast ambition", "legal imprisonment"),
    ("rd06", "\u9e3f\u9e44", "a great wild bird used as a symbol of lofty aspiration", "lofty aspiration", "household cooking"),
    ("rd07", "\u6276\u6851", "a mythical eastern tree associated with the rising sun", "the rising sun", "a prison cell"),
    ("rd08", "\u736c\u8c78", "a mythical creature associated with justice and law", "justice", "endless hesitation"),
    ("rd09", "\u9e92\u9e9f", "an auspicious mythical creature associated with benevolence", "auspiciousness", "heavy rainfall"),
    ("rd10", "\u767d\u6cfd", "a mythical creature said to know supernatural beings", "knowledge of spirits", "carrying a stone monument"),
    ("rd11", "\u7cbe\u536b", "a mythical bird remembered for trying to fill the sea", "filling the sea", "guarding roof tiles"),
    ("rd12", "\u5938\u7236", "a mythic giant remembered for pursuing the sun", "pursuing the sun", "attracting wealth"),
    ("rd13", "\u5ae6\u5a25", "a mythic woman associated with the moon", "the moon", "impartial law"),
    ("rd14", "\u5973\u5a32", "a creator goddess remembered for repairing the sky", "repairing the sky", "gluttony"),
    ("rd15", "\u5e94\u9f99", "a winged dragon associated with rain and ancient battles", "wings and rain", "a household meal"),
)
RARE_CONFIRMATION = (
    ("rc01", "\u6df7\u6c8c", "a primordial being associated with undifferentiated chaos", "primordial chaos", "measuring farmland"),
    ("rc02", "\u7a77\u5947", "a fierce mythical beast associated with encouraging wrongdoing", "wrongdoing", "healing medicine"),
    ("rc03", "\u68bc\u677c", "a mythical beast associated with stubbornness and disorder", "stubborn disorder", "the harvest moon"),
    ("rc04", "\u4e5d\u5c3e\u72d0", "a mythical fox distinguished by nine tails", "nine tails", "stone steles"),
    ("rc05", "\u9752\u9f99", "a directional guardian associated with the east", "the east", "the north"),
    ("rc06", "\u6731\u96c0", "a directional guardian associated with the south and fire", "the south and fire", "the west and metal"),
    ("rc07", "\u7384\u6b66", "a tortoise-snake guardian associated with the north", "the north", "the rising sun"),
    ("rc08", "\u767d\u864e", "a directional guardian associated with the west", "the west", "the east"),
    ("rc09", "\u6bd5\u65b9", "a one-legged mythical bird associated with fire", "fire", "river navigation"),
    ("rc10", "\u51e4\u51f0", "a mythical bird associated with auspicious renewal", "auspicious renewal", "legal punishment"),
    ("rc11", "\u91d1\u4e4c", "a mythical three-legged crow associated with the sun", "the sun", "the deep ocean"),
    ("rc12", "\u7389\u5154", "a mythical rabbit associated with the moon", "the moon", "roof ridges"),
    ("rc13", "\u6cb3\u56fe", "a cosmological diagram associated with ancient numerological order", "a cosmological diagram", "a winged horse"),
    ("rc14", "\u6d1b\u4e66", "a legendary diagram associated with a magic square", "a magic square", "gluttony"),
    ("rc15", "\u606f\u58e4", "mythical self-expanding soil used to control floods", "self-expanding soil", "a solar bird"),
)

TRANSLATION_DISCOVERY = (
    ("xd01", "red", "rouge", "blue", "bleu"),
    ("xd02", "green", "vert", "black", "noir"),
    ("xd03", "white", "blanc", "yellow", "jaune"),
    ("xd04", "orange", "orange", "purple", "violet"),
    ("xd05", "pink", "rose", "gray", "gris"),
    ("xd06", "brown", "marron", "beige", "beige"),
    ("xd07", "gold", "or", "silver", "argent"),
    ("xd08", "red", "rouge", "green", "vert"),
    ("xd09", "blue", "bleu", "black", "noir"),
    ("xd10", "white", "blanc", "orange", "orange"),
    ("xd11", "yellow", "jaune", "purple", "violet"),
    ("xd12", "pink", "rose", "brown", "marron"),
    ("xd13", "gray", "gris", "beige", "beige"),
    ("xd14", "gold", "or", "blue", "bleu"),
    ("xd15", "silver", "argent", "green", "vert"),
)
TRANSLATION_CONFIRMATION = (
    ("xc01", "book", "livre", "house", "maison"),
    ("xc02", "water", "eau", "fire", "feu"),
    ("xc03", "sun", "soleil", "moon", "lune"),
    ("xc04", "dog", "chien", "cat", "chat"),
    ("xc05", "bread", "pain", "milk", "lait"),
    ("xc06", "hand", "main", "road", "route"),
    ("xc07", "tree", "arbre", "flower", "fleur"),
    ("xc08", "bird", "oiseau", "fish", "poisson"),
    ("xc09", "table", "table", "chair", "chaise"),
    ("xc10", "door", "porte", "window", "fen\u00eatre"),
    ("xc11", "school", "\u00e9cole", "city", "ville"),
    ("xc12", "day", "jour", "night", "nuit"),
    ("xc13", "mother", "m\u00e8re", "father", "p\u00e8re"),
    ("xc14", "child", "enfant", "friend", "ami"),
    ("xc15", "car", "voiture", "train", "train"),
)


ITEMS_BY_FAMILY_SPLIT: dict[str, dict[str, tuple[tuple[str, ...], ...]]] = {
    "height_polarity": {
        "discovery": HEIGHT_DISCOVERY,
        "confirmation": HEIGHT_CONFIRMATION,
    },
    "contrast": {
        "discovery": CONTRAST_DISCOVERY,
        "confirmation": CONTRAST_CONFIRMATION,
    },
    "punctuation": {
        "discovery": PUNCTUATION_DISCOVERY,
        "confirmation": PUNCTUATION_CONFIRMATION,
    },
    "taxonomy": {
        "discovery": TAXONOMY_DISCOVERY,
        "confirmation": TAXONOMY_CONFIRMATION,
    },
    "rare_semantics": {
        "discovery": RARE_DISCOVERY,
        "confirmation": RARE_CONFIRMATION,
    },
    "translation_word": {
        "discovery": TRANSLATION_DISCOVERY,
        "confirmation": TRANSLATION_CONFIRMATION,
    },
}


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


def height_case(
    item: tuple[str, ...],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, high, middle, low = item
    fact_a = f"{high} is taller than {middle}"
    fact_b = f"{middle} is taller than {low}"
    operator = ("Resolve", "Work out", "Select", "Name")[template_index]
    query = "the tallest person" if branch == 0 else "the shortest person"
    if lexical == 0:
        text = (
            f"Facts: {fact_a}. {fact_b}. {operator} {query}. "
            "Return only the person's name."
        )
    else:
        text = (
            f"Using {fact_b.lower()} and {fact_a.lower()}, "
            f"{operator.lower()} who is {'tallest' if branch == 0 else 'shortest'}. "
            "Give one name only."
        )
        fact_a = fact_a.lower()
        fact_b = fact_b.lower()
        operator = operator.lower()
        query = "who is tallest" if branch == 0 else "who is shortest"
    spans = {
        "source_primary": mark(text, fact_a),
        "source_secondary": mark(text, fact_b),
        "operator": mark(text, operator),
        "query": mark(text, query),
    }
    return text, spans, {"b0": [high], "b1": [low]}, " "


def contrast_case(
    item: tuple[str, ...],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, premise, aligned, opposed = item
    conclusion = aligned if branch == 0 else opposed
    operator = ("Choose", "Complete", "Supply", "Insert")[template_index]
    blank = "___" if lexical == 0 else "[missing conjunction]"
    if lexical == 0:
        text = (
            f"{operator} the lowercase conjunction: {premise}, {blank} "
            f"{conclusion}. Return the conjunction alone."
        )
    else:
        text = (
            f"{operator} one lowercase conjunction to complete this sentence: "
            f"{premise}, {blank} {conclusion}. No explanation."
        )
    spans = {
        "source_primary": mark(text, premise),
        "source_secondary": mark(text, conclusion),
        "operator": mark(text, operator),
        "query": mark(text, blank),
    }
    return text, spans, {"b0": ["and"], "b1": ["but", "yet"]}, " "


def punctuation_case(
    item: tuple[str, ...],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, statement, question = item
    carrier = statement if branch == 0 else question
    kind = "statement" if branch == 0 else "question"
    operator = ("Choose", "Complete", "Supply", "Insert")[template_index]
    if lexical == 0:
        text = (
            f"{operator} only the final punctuation for this {kind}: "
            f"{carrier}"
        )
    else:
        text = (
            f"{operator} the single punctuation mark that must end the "
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
    item: tuple[str, ...],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, fruit, animal = item
    entity = fruit if branch == 0 else animal
    operator = ("Classify", "Label", "Categorize", "Identify")[template_index]
    cue = "broad biological category"
    if lexical == 0:
        text = (
            f"{operator} {entity} by its {cue}. Return one lowercase noun."
        )
    else:
        text = (
            f"{operator} the {cue} containing {entity}. "
            "Give only a lowercase category word."
        )
    spans = {
        "source_primary": mark(text, entity),
        "source_secondary": mark(text, cue),
        "operator": mark(text, operator),
        "query": mark(text, entity, occurrence="last"),
    }
    return text, spans, {"b0": ["fruit"], "b1": ["animal"]}, " "


def rare_case(
    item: tuple[str, ...],
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
        f"{operator} this claim: {query}? Return only yes or no in lowercase."
    )
    spans = {
        "source_primary": mark(text, source),
        "source_secondary": mark(text, prop),
        "operator": mark(text, operator),
        "query": mark(text, query),
    }
    return text, spans, {"b0": ["yes"], "b1": ["no"]}, " "


def translation_case(
    item: tuple[str, ...],
    branch: int,
    lexical: int,
    template_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, list[str]], str]:
    _, en0, fr0, en1, fr1 = item
    source = en0 if branch == 0 else en1
    operator = ("Translate", "Render", "Convert", "Express")[template_index]
    target = "French"
    if lexical == 0:
        text = (
            f"{operator} the English word {source} into {target}. "
            "Return exactly one lowercase French word."
        )
    else:
        text = (
            f"{operator} {source} from English to {target}; give only the "
            "lowercase translation."
        )
    spans = {
        "source_primary": mark(text, source),
        "source_secondary": mark(text, target),
        "operator": mark(text, operator),
        "query": mark(text, source, occurrence="last"),
    }
    return text, spans, {"b0": [fr0], "b1": [fr1]}, " "


BUILDERS: dict[str, Callable[..., Any]] = {
    "height_polarity": height_case,
    "contrast": contrast_case,
    "punctuation": punctuation_case,
    "taxonomy": taxonomy_case,
    "rare_semantics": rare_case,
    "translation_word": translation_case,
}


def build_model_case(
    tokenizer,
    model_name: str,
    family: str,
    split: str,
    item: tuple[str, ...],
    template_index: int,
    state: str,
    case_index: int,
) -> dict[str, Any]:
    branch, lexical = state_factors(state)
    raw_prompt, raw_spans, classes, prefix = BUILDERS[family](
        item,
        branch,
        lexical,
        template_index,
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
        tokenizer,
        rendered,
        raw_prompt,
        raw_spans,
    )
    role_spans["answer_boundary"] = (
        len(input_ids) - 1,
        len(input_ids) - 1,
    )
    candidate_token_ids = {
        class_name: [
            behavior.continuation_ids(tokenizer, rendered, prefix, label)
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
        "schema_version": "phase1077_nonblocking_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "semantic_case_index": case_index,
        "record_id": (
            f"{model_name}.{family}.{split}.{item[0]}."
            f"t{template_index}.{state}"
        ),
        "unit_id": (
            f"{family}.{split}.{item[0]}.t{template_index}"
        ),
        "family": family,
        "split": split,
        "item_id": str(item[0]),
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
        "role_positions": {
            role: int(span[1]) for role, span in role_spans.items()
        },
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
    state_groups: dict[str, list[dict[str, Any]]] = {}
    roles_valid = True
    candidate_disjoint = True
    for row in cases:
        state_groups.setdefault(str(row["unit_id"]), []).append(row)
        width = len(row["input_ids"])
        for role in CAPTURE_ROLES:
            start, end = row["role_spans"][role]
            roles_valid = roles_valid and 0 <= start <= end < width
        left = set(row["candidate_first_token_ids"]["b0"])
        right = set(row["candidate_first_token_ids"]["b1"])
        candidate_disjoint = (
            candidate_disjoint
            and bool(left)
            and bool(right)
            and left.isdisjoint(right)
        )
    item_ids = {
        split: {
            row["item_id"] for row in cases if row["split"] == split
        }
        for split in SPLITS
    }
    prompts = {
        split: {
            row["rendered_prompt"] for row in cases if row["split"] == split
        }
        for split in SPLITS
    }
    checks = {
        "case_count": len(cases) == len(FAMILIES) * 2 * 15 * 2 * 4,
        "unit_count": len(state_groups) == len(FAMILIES) * 2 * 15 * 2,
        "family_split_case_counts": all(
            counts[(family, split)] == 120
            for family in FAMILIES
            for split in SPLITS
        ),
        "complete_factorial_units": all(
            {row["state"] for row in values} == set(STATES)
            for values in state_groups.values()
        ),
        "role_spans_valid": roles_valid,
        "candidate_first_tokens_disjoint": candidate_disjoint,
        "independent_item_splits": item_ids["discovery"].isdisjoint(
            item_ids["confirmation"]
        ),
        "discovery_confirmation_prompts_disjoint": prompts[
            "discovery"
        ].isdisjoint(prompts["confirmation"]),
    }
    return {
        "schema_version": "phase1077_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(state_groups),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    if not SOURCE_PHASE1065.exists():
        raise RuntimeError("missing Phase1065 aggregate")
    if not SOURCE_PHASE1076.exists():
        raise RuntimeError("missing Phase1076 final summary")
    source_1065 = read_json(SOURCE_PHASE1065)
    source_1076 = read_json(SOURCE_PHASE1076)

    model_audits = {}
    model_case_digests = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = []
        case_index = 0
        for family in FAMILIES:
            for split in SPLITS:
                templates = TEMPLATES_BY_SPLIT[split]
                for item in ITEMS_BY_FAMILY_SPLIT[family][split]:
                    for template_index in templates:
                        for state in STATES:
                            cases.append(build_model_case(
                                tokenizer,
                                model_name,
                                family,
                                split,
                                item,
                                template_index,
                                state,
                                case_index,
                            ))
                            case_index += 1
        audit = audit_model(model_name, cases)
        audit["case_digest"] = digest(cases)
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
        model_case_digests[model_name] = audit["case_digest"]

    payload = {
        "schema_version": "phase1077_nonblocking_preregistration.v1",
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
        "conditionings": list(CONDITIONINGS),
        "assistant_prefill": ASSISTANT_PREFILL,
        "case_count_per_model": len(FAMILIES) * 2 * 15 * 2 * 4,
        "unit_count_per_model": len(FAMILIES) * 2 * 15 * 2,
        "model_case_digests": model_case_digests,
        "natural_audit_cases_per_family_split": (
            NATURAL_AUDIT_CASES_PER_FAMILY_SPLIT
        ),
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "evidence_thresholds": dict(EVIDENCE_THRESHOLDS),
        "source_phase1065_protocol_digest": source_1065[
            "protocol_digest"
        ],
        "source_phase1076_protocol_digest": source_1076[
            "protocol_digest"
        ],
        "source_phase1076_claim_status": source_1076["claim_status"],
        "primary_population": (
            "All preregistered finite forward states. Behavior errors do "
            "not delete a response from the descriptive atlas."
        ),
        "secondary_population": (
            "Complete four-state units for which all candidate comparisons "
            "are correct; this is a sensitivity ledger only."
        ),
        "evidence_levels": {
            "L0": "mapped with finite instrumentation",
            "L1": "within-model discovery/confirmation profile repetition",
            "L2": "raw normalized-depth profile repeats across models",
            "L3": "family-centered profile repeats across models",
            "L4": "L3 plus behavior-qualified repetition in at least two models",
            "L5": "causal support; not tested or assignable in Phase1077",
        },
        "measurement_order": [
            "freeze independent items, templates, roles, and candidate classes",
            "capture all finite residual, Attention-output, and MLP-output states",
            "construct unfiltered semantic, lexical, and interaction differences",
            "construct the behavior-complete sensitivity ledger separately",
            "compare independent discovery and confirmation profiles",
            "compare raw and family-centered normalized-depth profiles across models",
            "assign descriptive evidence levels without deleting failed families",
            "stop before causal interpretation",
        ],
        "interpretation_limits": [
            "A difference field is a response measurement, not a mechanism.",
            "A high raw profile cosine may reflect generic depth dynamics.",
            "Family centering is a control subtraction, not a recovered law.",
            "Behavior errors can make a response task-invalid but do not make it nonexistent.",
            "Cross-model functional repetition does not imply coordinate homology.",
            "No result can establish brain-model homology or evolutionary optimality.",
            "No Phase1077 evidence level can establish causal necessity or sufficiency.",
        ],
        "automatic_next": {
            "continue": False,
            "reason": (
                "This phase already includes independent lexical and template "
                "confirmation. A later causal phase requires a separate "
                "preregistration and must not block atlas growth."
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
            "schema_version": "phase1077_protocol_audit.v1",
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
