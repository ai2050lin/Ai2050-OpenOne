#!/usr/bin/env python3
"""Phase 979 truth x punctuation crossed-pair diagnostic dataset.

This module contains 128 deterministic role-reversal pairs across eight tasks.
It is intentionally separate from every Phase 979 natural-trajectory dataset:
IDs use ``p979_tp_cross`` and prompts use ``[TP979-TRUTH-CROSS]``.

Each pair contains two prompts and two fixed response labels, ``A`` and ``B``.
Label A is correct for prompt qA and wrong for qB; label B has the reverse
role.  Both prompts list the same two semantic options in the same order.  The
only prompt difference inside a pair is one explicitly recorded controlled
slot.  A runner can therefore cross correctness with the response strings
``A``/``B`` and ``A.``/``B.`` without using an alias matcher.

The dataset is ASCII-only and imports no Phase 977/978 holdout module.
Tokenizer-dependent claims are deliberately not made here.  A model runner
must still verify that A and B are single tokens and that appending ``.`` is a
pure, common one-token suffix in every rendered official-prefix context.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import unicodedata
from typing import Any, Iterable


SCHEMA_VERSION = 1
IDENTITY_VERSION = "phase979_truth_punctuation_cross_identity_v1"
DATASET_MARKER = "phase979_truth_punctuation_cross_v1"
PROMPT_MARKER = "[TP979-TRUTH-CROSS]"
ID_PREFIX = "p979_tp_cross"

TASKS = (
    "direct_fact",
    "classification",
    "arithmetic",
    "translation_format",
    "definition",
    "causal",
    "multistep_math",
    "logic",
)
SPLITS = ("development", "replication")
CANDIDATE_LABELS = ("A", "B")
PUNCTUATION_STATES = {
    "A": {"bare": "A", "period": "A."},
    "B": {"bare": "B", "period": "B."},
}

EXPECTED_TASK_COUNTS = {task: 16 for task in TASKS}
EXPECTED_SPLIT_COUNTS = {split: 64 for split in SPLITS}
EXPECTED_SPLIT_TASK_COUNTS = {
    split: {task: 8 for task in TASKS} for split in SPLITS
}
EXPECTED_PAIR_N = 128
EXPECTED_PROMPT_N = 256
EXPECTED_TEACHER_FORCED_ROW_N = 1024

_CONTROLLED_TOKEN = "{controlled}"
_RESPONSE_INSTRUCTION = (
    "Answer with exactly one listed label, A or B. "
    "One optional final ASCII period is allowed and does not change correctness."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _prompt_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _ordered_options(
    option_texts: dict[str, str], candidate_order: tuple[str, str]
) -> str:
    return "Listed options: " + "; ".join(
        f"{label} = {option_texts[label]}" for label in candidate_order
    ) + "."


def _make_pair(
    *,
    task: str,
    index: int,
    context: str,
    query_template: str,
    controlled_field: str,
    controlled_a: str,
    controlled_b: str,
    option_a: str,
    option_b: str,
    verification: dict[str, Any],
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    if _CONTROLLED_TOKEN not in query_template:
        raise ValueError("query template lacks the controlled slot")
    if query_template.count(_CONTROLLED_TOKEN) != 1:
        raise ValueError("query template must contain one controlled slot")
    if controlled_a == controlled_b:
        raise ValueError("controlled values must differ")

    pair_id = f"{ID_PREFIX}_{task}_{index:02d}"
    split = "development" if index <= 8 else "replication"
    candidate_order = ("A", "B") if index % 2 else ("B", "A")
    option_texts = {"A": option_a, "B": option_b}
    options_text = _ordered_options(option_texts, candidate_order)
    prompt_template = (
        f"{PROMPT_MARKER} Pair {pair_id}. {context} "
        f"Query: {query_template} {options_text} {_RESPONSE_INSTRUCTION}"
    )
    prompt_a = prompt_template.replace(_CONTROLLED_TOKEN, controlled_a)
    prompt_b = prompt_template.replace(_CONTROLLED_TOKEN, controlled_b)

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_marker": DATASET_MARKER,
        "id": pair_id,
        "task": task,
        "split": split,
        "prompt_ids": {"qA": f"{pair_id}_qa", "qB": f"{pair_id}_qb"},
        "prompt_template": prompt_template,
        "controlled_change": {
            "field": controlled_field,
            "qA": controlled_a,
            "qB": controlled_b,
        },
        "prompts": {"qA": prompt_a, "qB": prompt_b},
        "candidate_labels": ["A", "B"],
        "candidate_order": list(candidate_order),
        "option_texts": option_texts,
        "truth_table": {
            "qA": {"A": True, "B": False},
            "qB": {"A": False, "B": True},
        },
        "correct_label": {"qA": "A", "qB": "B"},
        "answer_states": deepcopy(PUNCTUATION_STATES),
        "verification": verification,
        "contracts": {
            "truth_source": "mechanical_role_reversal_not_matcher",
            "response_labels": "literal_ASCII_A_and_B",
            "period_rule": "one_optional_final_ASCII_period_truth_invariant",
            "tokenizer_runtime_gate": (
                "A/B must each be one token and period must be the same pure "
                "one-token suffix in every official-prefix context"
            ),
            "natural_trajectory_dataset": False,
            "internal_mechanism_evidence": False,
        },
    }


def _arithmetic_pairs() -> list[dict[str, Any]]:
    pairs = []
    for index in range(1, 17):
        left = 40 + 2 * index
        right_a = 6 + (index % 7)
        right_b = right_a + 1
        result_a = left + right_a
        result_b = left + right_b
        pairs.append(_make_pair(
            task="arithmetic",
            index=index,
            context="Use ordinary integer addition.",
            query_template=f"Compute {left} + {_CONTROLLED_TOKEN}.",
            controlled_field="right_operand",
            controlled_a=str(right_a),
            controlled_b=str(right_b),
            option_a=str(result_a),
            option_b=str(result_b),
            verification={
                "kind": "integer_addition",
                "left": left,
                "right": {"qA": right_a, "qB": right_b},
                "result": {"qA": result_a, "qB": result_b},
            },
        ))
    return pairs


_CAUSAL_RULES = (
    ("stopping time", "peak force", "decreases", "increases"),
    ("gas volume", "gas pressure", "decreases", "increases"),
    ("cooling airflow", "device temperature", "decreases", "increases"),
    ("electrical resistance", "circuit current", "decreases", "increases"),
    ("insulation thickness", "heat loss", "decreases", "increases"),
    ("light intensity", "sensor voltage", "increases", "decreases"),
    ("string tension", "vibration pitch", "increases", "decreases"),
    ("applied force", "object acceleration", "increases", "decreases"),
    ("surface friction", "frictional heating", "increases", "decreases"),
    ("coolant flow", "engine temperature", "decreases", "increases"),
    ("rainfall amount", "river level", "increases", "decreases"),
    ("filter blockage", "airflow rate", "decreases", "increases"),
    ("spring compression", "restoring force", "increases", "decreases"),
    ("lamp distance", "illumination", "decreases", "increases"),
    ("battery voltage", "motor speed", "increases", "decreases"),
    ("pipe diameter", "flow resistance", "decreases", "increases"),
)


def _causal_pairs() -> list[dict[str, Any]]:
    pairs = []
    for index, (factor, outcome, on_increase, on_decrease) in enumerate(
        _CAUSAL_RULES, start=1
    ):
        context = (
            f"Stated rule: when {factor} is increased, {outcome} {on_increase}; "
            f"when {factor} is decreased, {outcome} {on_decrease}."
        )
        pairs.append(_make_pair(
            task="causal",
            index=index,
            context=context,
            query_template=(
                f"The {factor} is {_CONTROLLED_TOKEN}. What happens to {outcome}?"
            ),
            controlled_field="causal_action",
            controlled_a="increased",
            controlled_b="decreased",
            option_a=f"{outcome} {on_increase}",
            option_b=f"{outcome} {on_decrease}",
            verification={
                "kind": "explicit_causal_rule",
                "factor": factor,
                "outcome": outcome,
                "mapping": {
                    "increased": f"{outcome} {on_increase}",
                    "decreased": f"{outcome} {on_decrease}",
                },
            },
        ))
    return pairs


_CLASSIFICATION_RULES = (
    ("feathers", "fur", "bird", "mammal"),
    ("six legs", "eight legs", "insect", "arachnid"),
    ("interlocking crystals", "layered grains", "igneous", "sedimentary"),
    ("divisible by two", "not divisible by two", "even", "odd"),
    ("makes seeds", "makes spores", "seed plant", "spore plant"),
    ("conducts electricity", "blocks electricity", "conductor", "insulator"),
    ("three sides", "four equal sides", "triangle", "square"),
    ("has a backbone", "lacks a backbone", "vertebrate", "invertebrate"),
    ("orbits a planet", "orbits a star", "moon", "planet"),
    ("melts below room temperature", "melts above room temperature", "low melt", "high melt"),
    ("contains one element", "contains two elements", "element", "compound"),
    ("stores charge", "resists current", "capacitor", "resistor"),
    ("uses sunlight", "eats plants", "producer", "herbivore"),
    ("parallel sides", "no parallel sides", "parallel class", "nonparallel class"),
    ("source code input", "bytecode input", "compiler", "virtual machine"),
    ("lossless rule", "lossy rule", "lossless codec", "lossy codec"),
)


def _classification_pairs() -> list[dict[str, Any]]:
    pairs = []
    for index, (property_a, property_b, class_a, class_b) in enumerate(
        _CLASSIFICATION_RULES, start=1
    ):
        context = (
            f"Classification rule: property '{property_a}' maps to '{class_a}', "
            f"and property '{property_b}' maps to '{class_b}'."
        )
        pairs.append(_make_pair(
            task="classification",
            index=index,
            context=context,
            query_template=(
                f"The test item has property '{_CONTROLLED_TOKEN}'. Which class applies?"
            ),
            controlled_field="observed_property",
            controlled_a=property_a,
            controlled_b=property_b,
            option_a=class_a,
            option_b=class_b,
            verification={
                "kind": "explicit_classification_rule",
                "mapping": {property_a: class_a, property_b: class_b},
            },
        ))
    return pairs


_DEFINITION_RULES = (
    ("latitude", "angular distance north or south of the equator", "longitude", "angular distance east or west of a reference meridian"),
    ("evaporation", "liquid changing into gas", "condensation", "gas changing into liquid"),
    ("numerator", "the top number in a fraction", "denominator", "the bottom number in a fraction"),
    ("herbivore", "an animal that eats plants", "carnivore", "an animal that eats other animals"),
    ("polygon", "a closed flat shape with straight sides", "polyhedron", "a solid with flat polygonal faces"),
    ("atom", "a basic unit of an element", "molecule", "two or more atoms chemically joined"),
    ("cache", "temporary storage for reusable data", "archive", "long-term storage for preserved data"),
    ("encryption", "transforming data to hide its content", "compression", "transforming data to reduce its size"),
    ("velocity", "speed together with direction", "acceleration", "change of velocity over time"),
    ("habitat", "the place where an organism lives", "niche", "the role of an organism in its environment"),
    ("isotope", "an atom with a different neutron count", "ion", "an atom with a net electric charge"),
    ("syntax", "rules for arranging symbols", "semantics", "meaning carried by symbols"),
    ("peninsula", "land nearly surrounded by water", "island", "land completely surrounded by water"),
    ("renewable resource", "a resource replenished on a human timescale", "nonrenewable resource", "a resource not replenished on a human timescale"),
    ("mitosis", "cell division producing two similar cells", "meiosis", "cell division producing reproductive cells"),
    ("compiler", "a program translating a whole source program", "interpreter", "a program executing source instructions directly"),
)


def _definition_pairs() -> list[dict[str, Any]]:
    pairs = []
    for index, (term_a, definition_a, term_b, definition_b) in enumerate(
        _DEFINITION_RULES, start=1
    ):
        context = (
            f"Mini-glossary: '{term_a}' means '{definition_a}'; "
            f"'{term_b}' means '{definition_b}'."
        )
        pairs.append(_make_pair(
            task="definition",
            index=index,
            context=context,
            query_template=(
                f"Which glossary term has definition '{_CONTROLLED_TOKEN}'?"
            ),
            controlled_field="queried_definition",
            controlled_a=definition_a,
            controlled_b=definition_b,
            option_a=term_a,
            option_b=term_b,
            verification={
                "kind": "explicit_definition_lookup",
                "mapping": {definition_a: term_a, definition_b: term_b},
            },
        ))
    return pairs


def _direct_fact_pairs() -> list[dict[str, Any]]:
    pairs = []
    for index in range(1, 17):
        subject_a = f"archive-{index:02d}-north"
        subject_b = f"archive-{index:02d}-south"
        value_a = f"code-{100 + 3 * index}"
        value_b = f"code-{101 + 3 * index}"
        context = (
            f"Recorded facts: {subject_a} has {value_a}; "
            f"{subject_b} has {value_b}."
        )
        pairs.append(_make_pair(
            task="direct_fact",
            index=index,
            context=context,
            query_template=f"Which recorded code belongs to {_CONTROLLED_TOKEN}?",
            controlled_field="queried_subject",
            controlled_a=subject_a,
            controlled_b=subject_b,
            option_a=value_a,
            option_b=value_b,
            verification={
                "kind": "explicit_fact_lookup",
                "mapping": {subject_a: value_a, subject_b: value_b},
            },
        ))
    return pairs


def _logic_value(operator: str, left: bool, right: bool) -> bool:
    if operator == "AND":
        return left and right
    if operator == "OR":
        return left or right
    if operator == "XOR":
        return left != right
    if operator == "IMPLIES":
        return (not left) or right
    raise ValueError(f"unknown Boolean operator: {operator}")


def _logic_pairs() -> list[dict[str, Any]]:
    configurations = (
        ("AND", True),
        ("OR", False),
        ("XOR", False),
        ("IMPLIES", True),
    )
    pairs = []
    for index in range(1, 17):
        operator, left = configurations[(index - 1) % len(configurations)]
        right_a = True
        right_b = False
        result_a = _logic_value(operator, left, right_a)
        result_b = _logic_value(operator, left, right_b)
        if not result_a or result_b:
            raise RuntimeError("logic construction must map qA to true and qB to false")
        left_text = "true" if left else "false"
        context = (
            f"Use standard Boolean logic. Proposition P{index:02d} is {left_text}."
        )
        pairs.append(_make_pair(
            task="logic",
            index=index,
            context=context,
            query_template=(
                f"Let Q{index:02d} be {_CONTROLLED_TOKEN}. What is "
                f"P{index:02d} {operator} Q{index:02d}?"
            ),
            controlled_field="right_boolean_value",
            controlled_a="true",
            controlled_b="false",
            option_a="true",
            option_b="false",
            verification={
                "kind": "boolean_evaluation",
                "operator": operator,
                "left": left,
                "right": {"qA": right_a, "qB": right_b},
                "result": {"qA": result_a, "qB": result_b},
            },
        ))
    return pairs


_UNITS = (
    "crates", "boxes", "jars", "bolts", "books", "parts", "trays", "drums",
    "seats", "rolls", "bags", "tiles", "cans", "rods", "lamps", "kits",
)


def _multistep_math_pairs() -> list[dict[str, Any]]:
    pairs = []
    for index, unit in enumerate(_UNITS, start=1):
        start_a = 45 + 3 * index
        start_b = start_a + 1
        received = 8 + index
        shipped = 4 + (index % 6)
        result_a = start_a + received - shipped
        result_b = start_b + received - shipped
        context = (
            f"A depot receives {received} {unit} and then ships {shipped} {unit}."
        )
        pairs.append(_make_pair(
            task="multistep_math",
            index=index,
            context=context,
            query_template=(
                f"It starts with {_CONTROLLED_TOKEN} {unit}. How many {unit} remain?"
            ),
            controlled_field="starting_inventory",
            controlled_a=str(start_a),
            controlled_b=str(start_b),
            option_a=f"{result_a} {unit}",
            option_b=f"{result_b} {unit}",
            verification={
                "kind": "inventory_add_subtract",
                "start": {"qA": start_a, "qB": start_b},
                "received": received,
                "shipped": shipped,
                "unit": unit,
                "result": {"qA": result_a, "qB": result_b},
            },
        ))
    return pairs


_TRANSLATION_RULES = (
    ("mesa", "table", "arbol", "tree"),
    ("luna", "moon", "sol", "sun"),
    ("chat", "cat", "chien", "dog"),
    ("haus", "house", "baum", "tree"),
    ("acqua", "water", "pane", "bread"),
    ("rio", "river", "lago", "lake"),
    ("nieve", "snow", "lluvia", "rain"),
    ("libro", "book", "silla", "chair"),
)
_FORMAT_RULES = (
    ("amber", "cedar"),
    ("raven", "otter"),
    ("silver", "violet"),
    ("cobalt", "scarlet"),
    ("planet", "comet"),
    ("harbor", "meadow"),
    ("canyon", "forest"),
    ("winter", "summer"),
)


def _translation_format_pairs() -> list[dict[str, Any]]:
    # Each split receives four translation and four formatting pairs.  The
    # sequence also gives each subtype two A-first and two B-first pairs.
    ordered_specs: list[tuple[str, tuple[str, ...]]] = []
    ordered_specs.extend(("translation", spec) for spec in _TRANSLATION_RULES[:4])
    ordered_specs.extend(("format", spec) for spec in _FORMAT_RULES[:4])
    ordered_specs.extend(("translation", spec) for spec in _TRANSLATION_RULES[4:])
    ordered_specs.extend(("format", spec) for spec in _FORMAT_RULES[4:])

    pairs = []
    for index, (subtype, spec) in enumerate(ordered_specs, start=1):
        if subtype == "translation":
            source_a, target_a, source_b, target_b = spec
            context = (
                f"Mini-dictionary: '{source_a}' means '{target_a}'; "
                f"'{source_b}' means '{target_b}'."
            )
            pair = _make_pair(
                task="translation_format",
                index=index,
                context=context,
                query_template=(
                    f"Translate the dictionary word '{_CONTROLLED_TOKEN}'."
                ),
                controlled_field="source_word",
                controlled_a=source_a,
                controlled_b=source_b,
                option_a=target_a,
                option_b=target_b,
                verification={
                    "kind": "explicit_translation_lookup",
                    "subtype": subtype,
                    "mapping": {source_a: target_a, source_b: target_b},
                },
            )
        else:
            raw_a, raw_b = spec
            context = "Formatting rule: convert the selected lowercase word to uppercase."
            pair = _make_pair(
                task="translation_format",
                index=index,
                context=context,
                query_template=f"Format the word '{_CONTROLLED_TOKEN}'.",
                controlled_field="raw_word",
                controlled_a=raw_a,
                controlled_b=raw_b,
                option_a=raw_a.upper(),
                option_b=raw_b.upper(),
                verification={
                    "kind": "uppercase_format",
                    "subtype": subtype,
                    "raw": {"qA": raw_a, "qB": raw_b},
                    "result": {"qA": raw_a.upper(), "qB": raw_b.upper()},
                },
            )
        pairs.append(pair)
    return pairs


def _build_all_pairs() -> list[dict[str, Any]]:
    by_task = {
        "direct_fact": _direct_fact_pairs(),
        "classification": _classification_pairs(),
        "arithmetic": _arithmetic_pairs(),
        "translation_format": _translation_format_pairs(),
        "definition": _definition_pairs(),
        "causal": _causal_pairs(),
        "multistep_math": _multistep_math_pairs(),
        "logic": _logic_pairs(),
    }
    return [pair for task in TASKS for pair in by_task[task]]


def build_pairs(split: str | None = None) -> list[dict[str, Any]]:
    """Return fresh pair dictionaries for ``development``, ``replication``, or all.

    ``None`` and ``"all"`` both select all 128 pairs.  Unknown split names fail
    closed rather than silently returning an empty dataset.
    """
    selected = "all" if split is None else str(split)
    if selected not in {"all", *SPLITS}:
        raise ValueError(
            f"split must be one of all/{'/'.join(SPLITS)}, got {selected!r}"
        )
    pairs = _build_all_pairs()
    if selected != "all":
        pairs = [pair for pair in pairs if pair["split"] == selected]
    return deepcopy(pairs)


def _verify_task_payload(pair: dict[str, Any]) -> list[str]:
    """Recompute each pair's two semantic options without an alias matcher."""
    errors: list[str] = []
    pair_id = str(pair.get("id", "<missing-id>"))
    verification = pair.get("verification")
    if not isinstance(verification, dict):
        return [f"{pair_id}: missing verification payload"]
    kind = verification.get("kind")
    option_texts = pair.get("option_texts", {})
    controlled = pair.get("controlled_change", {})

    try:
        if kind == "integer_addition":
            expected_a = verification["left"] + verification["right"]["qA"]
            expected_b = verification["left"] + verification["right"]["qB"]
            if verification["result"] != {"qA": expected_a, "qB": expected_b}:
                errors.append(f"{pair_id}: invalid addition results")
            if option_texts != {"A": str(expected_a), "B": str(expected_b)}:
                errors.append(f"{pair_id}: addition options do not match results")
        elif kind in {
            "explicit_causal_rule",
            "explicit_classification_rule",
            "explicit_definition_lookup",
            "explicit_fact_lookup",
            "explicit_translation_lookup",
        }:
            mapping = verification["mapping"]
            expected = {
                "A": mapping[controlled["qA"]],
                "B": mapping[controlled["qB"]],
            }
            if option_texts != expected:
                errors.append(f"{pair_id}: lookup/rule options do not match mapping")
        elif kind == "boolean_evaluation":
            expected_a = _logic_value(
                verification["operator"],
                bool(verification["left"]),
                bool(verification["right"]["qA"]),
            )
            expected_b = _logic_value(
                verification["operator"],
                bool(verification["left"]),
                bool(verification["right"]["qB"]),
            )
            if verification["result"] != {"qA": expected_a, "qB": expected_b}:
                errors.append(f"{pair_id}: invalid Boolean results")
            expected_options = {
                "A": "true" if expected_a else "false",
                "B": "true" if expected_b else "false",
            }
            if option_texts != expected_options:
                errors.append(f"{pair_id}: Boolean options do not match results")
        elif kind == "inventory_add_subtract":
            expected_a = (
                verification["start"]["qA"]
                + verification["received"]
                - verification["shipped"]
            )
            expected_b = (
                verification["start"]["qB"]
                + verification["received"]
                - verification["shipped"]
            )
            if verification["result"] != {"qA": expected_a, "qB": expected_b}:
                errors.append(f"{pair_id}: invalid inventory results")
            unit = verification["unit"]
            if option_texts != {
                "A": f"{expected_a} {unit}",
                "B": f"{expected_b} {unit}",
            }:
                errors.append(f"{pair_id}: inventory options do not match results")
        elif kind == "uppercase_format":
            expected_a = verification["raw"]["qA"].upper()
            expected_b = verification["raw"]["qB"].upper()
            if verification["result"] != {"qA": expected_a, "qB": expected_b}:
                errors.append(f"{pair_id}: invalid uppercase results")
            if option_texts != {"A": expected_a, "B": expected_b}:
                errors.append(f"{pair_id}: uppercase options do not match results")
        else:
            errors.append(f"{pair_id}: unknown verification kind {kind!r}")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{pair_id}: malformed verification payload: {exc}")
    return errors


def _previous_prompt_keys(previous_prompts: Iterable[Any] | None) -> set[str]:
    keys: set[str] = set()
    if previous_prompts is None:
        return keys
    for value in previous_prompts:
        if isinstance(value, dict):
            candidates = []
            if isinstance(value.get("prompts"), dict):
                candidates.extend(value["prompts"].values())
            for field in ("prompt", "prompt_a", "prompt_b"):
                if value.get(field):
                    candidates.append(value[field])
        else:
            candidates = [value]
        for prompt in candidates:
            if prompt:
                keys.add(_prompt_key(str(prompt)))
    return keys


def audit_pairs(
    pairs: list[dict[str, Any]] | None = None,
    previous_prompts: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Audit schema, balance, role reversal, formal truth, and prompt identity.

    The input may be all pairs or exactly one complete split.  Tokenizer-level
    one-token checks are outside this module and remain a mandatory runtime
    preregistration gate.
    """
    rows = build_pairs("all") if pairs is None else deepcopy(list(pairs))
    errors: list[str] = []
    schema_issues: list[str] = []
    encoding_issues: list[str] = []
    truth_issues: list[str] = []
    controlled_change_issues: list[str] = []

    expected_by_id = {row["id"]: row for row in _build_all_pairs()}
    ids = [str(row.get("id", "")) for row in rows]
    duplicate_ids = sorted(
        key for key, count in Counter(ids).items() if key and count > 1
    )
    unknown_ids = sorted(set(ids) - set(expected_by_id))

    all_prompt_values: list[str] = []
    all_prompt_ids: list[str] = []
    task_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    order_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_order_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    required = {
        "schema_version", "dataset_marker", "id", "task", "split",
        "prompt_ids", "prompt_template", "controlled_change", "prompts",
        "candidate_labels", "candidate_order", "option_texts", "truth_table",
        "correct_label", "answer_states", "verification", "contracts",
    }

    for row in rows:
        pair_id = str(row.get("id", "<missing-id>"))
        missing = sorted(required - set(row))
        if missing:
            schema_issues.append(f"{pair_id}: missing fields {missing}")
            continue
        task = str(row["task"])
        split = str(row["split"])
        task_counts[task] += 1
        split_counts[split] += 1
        split_task_counts[split][task] += 1

        if row["schema_version"] != SCHEMA_VERSION:
            schema_issues.append(f"{pair_id}: wrong schema version")
        if row["dataset_marker"] != DATASET_MARKER:
            schema_issues.append(f"{pair_id}: wrong dataset marker")
        if not pair_id.startswith(ID_PREFIX + "_"):
            schema_issues.append(f"{pair_id}: wrong ID namespace")
        if "natural" in pair_id.casefold():
            schema_issues.append(f"{pair_id}: natural-data marker collision")
        if task not in TASKS:
            schema_issues.append(f"{pair_id}: unknown task {task!r}")
        if split not in SPLITS:
            schema_issues.append(f"{pair_id}: unknown split {split!r}")

        if row["candidate_labels"] != ["A", "B"]:
            schema_issues.append(f"{pair_id}: response labels are not literal A/B")
        order = row["candidate_order"]
        if order not in (["A", "B"], ["B", "A"]):
            schema_issues.append(f"{pair_id}: invalid candidate order {order!r}")
        else:
            order_name = "A_first" if order[0] == "A" else "B_first"
            order_counts[task][order_name] += 1
            split_order_counts[split][task][order_name] += 1

        expected_truth = {
            "qA": {"A": True, "B": False},
            "qB": {"A": False, "B": True},
        }
        if row["truth_table"] != expected_truth:
            truth_issues.append(f"{pair_id}: truth table is not role reversed")
        if row["correct_label"] != {"qA": "A", "qB": "B"}:
            truth_issues.append(f"{pair_id}: correct-label table is invalid")
        if row["answer_states"] != PUNCTUATION_STATES:
            truth_issues.append(f"{pair_id}: A/B punctuation states changed")
        if row["option_texts"].get("A") == row["option_texts"].get("B"):
            truth_issues.append(f"{pair_id}: semantic options are identical")
        if row["contracts"].get("truth_source") != (
            "mechanical_role_reversal_not_matcher"
        ):
            truth_issues.append(f"{pair_id}: matcher-free truth contract changed")

        template = row["prompt_template"]
        controlled = row["controlled_change"]
        prompts = row["prompts"]
        if not isinstance(template, str) or template.count(_CONTROLLED_TOKEN) != 1:
            controlled_change_issues.append(
                f"{pair_id}: prompt template lacks exactly one controlled slot"
            )
        elif not isinstance(controlled, dict) or not isinstance(prompts, dict):
            controlled_change_issues.append(
                f"{pair_id}: controlled change or prompts are malformed"
            )
        else:
            value_a = controlled.get("qA")
            value_b = controlled.get("qB")
            if not isinstance(value_a, str) or not isinstance(value_b, str):
                controlled_change_issues.append(
                    f"{pair_id}: controlled values must be strings"
                )
            elif value_a == value_b:
                controlled_change_issues.append(
                    f"{pair_id}: controlled values do not differ"
                )
            else:
                expected_a = template.replace(_CONTROLLED_TOKEN, value_a)
                expected_b = template.replace(_CONTROLLED_TOKEN, value_b)
                if prompts.get("qA") != expected_a:
                    controlled_change_issues.append(
                        f"{pair_id}: qA differs beyond the controlled slot"
                    )
                if prompts.get("qB") != expected_b:
                    controlled_change_issues.append(
                        f"{pair_id}: qB differs beyond the controlled slot"
                    )
                if expected_a == expected_b:
                    controlled_change_issues.append(
                        f"{pair_id}: rendered prompts are identical"
                    )

        for role in ("qA", "qB"):
            prompt = prompts.get(role, "") if isinstance(prompts, dict) else ""
            prompt_id = (
                row["prompt_ids"].get(role, "")
                if isinstance(row["prompt_ids"], dict) else ""
            )
            all_prompt_values.append(str(prompt))
            all_prompt_ids.append(str(prompt_id))
            if PROMPT_MARKER not in str(prompt):
                schema_issues.append(f"{pair_id}/{role}: prompt marker missing")
            if _RESPONSE_INSTRUCTION not in str(prompt):
                schema_issues.append(
                    f"{pair_id}/{role}: exact A/B optional-period instruction missing"
                )
            if not str(prompt_id).startswith(pair_id + "_"):
                schema_issues.append(f"{pair_id}/{role}: prompt ID namespace mismatch")

        truth_issues.extend(_verify_task_payload(row))

        expected = expected_by_id.get(pair_id)
        if expected is not None and _canonical_json(row) != _canonical_json(expected):
            schema_issues.append(f"{pair_id}: row differs from canonical generator")

        string_values = [
            pair_id, task, split, template,
            str(row["option_texts"].get("A", "")),
            str(row["option_texts"].get("B", "")),
            *[str(value) for value in prompts.values()],
        ]
        for value in string_values:
            if unicodedata.normalize("NFC", value) != value:
                encoding_issues.append(f"{pair_id}: non-NFC text")
            if not value.isascii():
                encoding_issues.append(f"{pair_id}: non-ASCII text")
            if "\ufffd" in value:
                encoding_issues.append(f"{pair_id}: U+FFFD present")
            if any(0x80 <= ord(char) <= 0x9F for char in value):
                encoding_issues.append(f"{pair_id}: C1 control present")

    duplicate_prompt_ids = sorted(
        key for key, count in Counter(all_prompt_ids).items() if key and count > 1
    )
    prompt_keys = [_prompt_key(value) for value in all_prompt_values]
    duplicate_prompts = sorted(
        key for key, count in Counter(prompt_keys).items() if key and count > 1
    )
    previous_keys = _previous_prompt_keys(previous_prompts)
    cross_set_overlap = sorted(set(prompt_keys) & previous_keys)

    present_splits = set(split_counts)
    if present_splits == set(SPLITS):
        expected_n = EXPECTED_PAIR_N
        expected_tasks = EXPECTED_TASK_COUNTS
        expected_splits = EXPECTED_SPLIT_COUNTS
        expected_per_split_task = EXPECTED_SPLIT_TASK_COUNTS
        expected_order_per_task = {"A_first": 8, "B_first": 8}
        expected_split_order = {"A_first": 4, "B_first": 4}
    elif len(present_splits) == 1 and next(iter(present_splits), None) in SPLITS:
        only_split = next(iter(present_splits))
        expected_n = 64
        expected_tasks = {task: 8 for task in TASKS}
        expected_splits = {only_split: 64}
        expected_per_split_task = {only_split: {task: 8 for task in TASKS}}
        expected_order_per_task = {"A_first": 4, "B_first": 4}
        expected_split_order = expected_order_per_task
    else:
        expected_n = -1
        expected_tasks = {}
        expected_splits = {}
        expected_per_split_task = {}
        expected_order_per_task = {}
        expected_split_order = {}
        errors.append(f"input is not all pairs or one complete split: {present_splits}")

    if len(rows) != expected_n:
        errors.append(f"expected {expected_n} pairs, found {len(rows)}")
    if dict(sorted(task_counts.items())) != expected_tasks:
        errors.append(f"task counts differ: {dict(sorted(task_counts.items()))}")
    if dict(sorted(split_counts.items())) != expected_splits:
        errors.append(f"split counts differ: {dict(sorted(split_counts.items()))}")
    actual_split_task = {
        split: dict(sorted(counts.items()))
        for split, counts in sorted(split_task_counts.items())
    }
    if actual_split_task != expected_per_split_task:
        errors.append(f"split/task counts differ: {actual_split_task}")
    for task in expected_tasks:
        actual = dict(sorted(order_counts[task].items()))
        if actual != expected_order_per_task:
            errors.append(f"{task}: candidate order imbalance {actual}")
    for split in expected_per_split_task:
        for task in TASKS:
            actual = dict(sorted(split_order_counts[split][task].items()))
            if actual != expected_split_order:
                errors.append(
                    f"{split}/{task}: candidate order imbalance {actual}"
                )

    if duplicate_ids:
        errors.append(f"duplicate pair IDs: {duplicate_ids}")
    if unknown_ids:
        errors.append(f"unknown pair IDs: {unknown_ids}")
    if duplicate_prompt_ids:
        errors.append(f"duplicate prompt IDs: {duplicate_prompt_ids}")
    if duplicate_prompts:
        errors.append(f"duplicate prompts: {duplicate_prompts}")
    if cross_set_overlap:
        errors.append(f"cross-set prompt overlap: {cross_set_overlap}")
    errors.extend(schema_issues)
    errors.extend(truth_issues)
    errors.extend(controlled_change_issues)
    errors.extend(sorted(set(encoding_issues)))

    passed = not errors
    return {
        "ok": passed,
        "passed": passed,
        "schema_version": SCHEMA_VERSION,
        "dataset_marker": DATASET_MARKER,
        "n_pairs": len(rows),
        "n_prompts": len(all_prompt_values),
        "expected_teacher_forced_rows": len(rows) * 8,
        "task_counts": dict(sorted(task_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_task_counts": actual_split_task,
        "candidate_order_counts": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(order_counts.items())
        },
        "duplicate_ids": duplicate_ids,
        "unknown_ids": unknown_ids,
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "duplicate_prompts": duplicate_prompts,
        "cross_set_overlap": cross_set_overlap,
        "schema_issues": schema_issues,
        "truth_issues": truth_issues,
        "controlled_change_issues": controlled_change_issues,
        "encoding_issues": sorted(set(encoding_issues)),
        "tokenizer_audit_required": True,
        "tokenizer_contract": (
            "For every official-prefix context, A and B must each be one token; "
            "A./B. must preserve the bare token prefix and append the same one "
            "ASCII-period token. This module does not certify tokenizer behavior."
        ),
        "errors": errors,
    }


def dataset_identity() -> dict[str, Any]:
    """Return stable content hashes without timestamps or runtime state."""
    all_pairs = build_pairs("all")
    development = [pair for pair in all_pairs if pair["split"] == "development"]
    replication = [pair for pair in all_pairs if pair["split"] == "replication"]
    prompt_rows = [
        {
            "pair_id": pair["id"],
            "qA": pair["prompts"]["qA"],
            "qB": pair["prompts"]["qB"],
        }
        for pair in all_pairs
    ]
    core = {
        "identity_version": IDENTITY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_marker": DATASET_MARKER,
        "id_prefix": ID_PREFIX,
        "prompt_marker": PROMPT_MARKER,
        "tasks": list(TASKS),
        "splits": list(SPLITS),
        "candidate_labels": list(CANDIDATE_LABELS),
        "n_pairs": len(all_pairs),
        "n_prompts": len(prompt_rows) * 2,
        "expected_teacher_forced_rows": len(all_pairs) * 8,
        "all_pairs_sha256": _sha256_json(all_pairs),
        "development_pairs_sha256": _sha256_json(development),
        "replication_pairs_sha256": _sha256_json(replication),
        "prompts_sha256": _sha256_json(prompt_rows),
    }
    return {**core, "identity_sha256": _sha256_json(core)}


STABLE_IDENTITY = dataset_identity()


if __name__ == "__main__":
    audit = audit_pairs()
    print(json.dumps({"identity": STABLE_IDENTITY, "audit": audit}, indent=2))
    raise SystemExit(0 if audit["ok"] else 1)
