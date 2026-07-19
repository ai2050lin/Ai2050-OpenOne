#!/usr/bin/env python3
"""Build and audit the holdout-free Phase 981 fresh256 corpus.

The corpus contains eight mechanically decidable task families.  Each task has
16 construction pairs, and every pair has one ``easy`` and one ``hard`` item:

    8 tasks * 16 pairs * 2 difficulty levels = 256 items.

Within every task x difficulty stratum the literal answers A/B occur 8/8.
Item and pair identifiers are deterministic opaque hashes; prompts carry a new
Phase 981 marker namespace.  Difficulty is a frozen construction attribute,
not a claim about observed model accuracy.

Freshness is checked against the public Phase 979 diagnostic128 corpus in two
ways: (1) normalized prompt-content hashes after removing dataset markers and
response boilerplate, and (2) structural payload hashes over task/spec/options.
This module never imports or reads any Phase 977 holdout artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import phase979_diagnostic_dataset as phase979_public


PHASE = 981
SCHEMA_VERSION = 1
DATASET_NAME = "phase981_fresh256"
DATASET_MARKER = "phase981_fresh256_v1"
PROMPT_MARKER_NAMESPACE = "P981-FRESH256-V1"
DIFFICULTIES = ("easy", "hard")
LABELS = ("A", "B")
PAIRS_PER_TASK = 16
ITEMS_PER_TASK = PAIRS_PER_TASK * len(DIFFICULTIES)
ITEM_COUNT = 256
PAIR_COUNT = 128
TASKS = (
    "multistep_arithmetic",
    "modular_arithmetic",
    "boolean_logic",
    "relation_path",
    "state_machine",
    "sequence_rule",
    "string_transform",
    "constraint_order",
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "glm5" / "result" / "phase981_fresh256_confirmation"
DATASET_PATH = OUT / "dataset.json"
AUDIT_PATH = OUT / "audit.json"

EXPECTED_PHASE979_SCRIPT_SHA256 = (
    "74875eb5e00253f952969904a0164a06920e995c82735b30b1c307535b853605"
)
EXPECTED_PHASE979_IDENTITY_SHA256 = (
    "2da762df071a8a096feb017bd9fbf640454e056860bec2ac1c226fc55243330a"
)
EXPECTED_PHASE979_ITEMS_SHA256 = (
    "e884f922d77482baded1da55562df81685808dc0718049e26413ebacf56ece10"
)

_RESPONSE_INSTRUCTION = (
    "Respond with exactly one label, A or B. A single trailing ASCII period "
    "is permitted. Do not add any other text."
)
_PHASE979_RESPONSE_INSTRUCTION = (
    "Return only A or B as the final answer. One optional final ASCII period "
    "is allowed; output nothing else."
)
_ITEM_ID_RE = re.compile(r"^p981_f_[0-9a-f]{20}$")
_PAIR_ID_RE = re.compile(r"^p981_pair_[0-9a-f]{20}$")
_MARKER_RE = re.compile(r"^\[P981-FRESH256-V1\|[0-9A-F]{20}\]$")
_LEADING_MARKER_RE = re.compile(r"^\[[^\]\r\n]+\]\s*")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _opaque(namespace: str, *parts: Any) -> str:
    payload = "|".join([DATASET_MARKER, namespace, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _pair_id(task: str, pair_ordinal: int) -> str:
    return f"p981_pair_{_opaque('pair', task, pair_ordinal)}"


def _item_id(task: str, pair_ordinal: int, difficulty: str) -> str:
    return f"p981_f_{_opaque('item', task, pair_ordinal, difficulty)}"


def _marker(task: str, pair_ordinal: int, difficulty: str) -> str:
    code = _opaque("marker", task, pair_ordinal, difficulty).upper()
    return f"[{PROMPT_MARKER_NAMESPACE}|{code}]"


def _forbidden_holdout_modules() -> list[str]:
    return sorted(
        name for name in sys.modules
        if "phase977" in name.casefold() and "holdout" in name.casefold()
    )


def _assert_no_holdout_import() -> None:
    loaded = _forbidden_holdout_modules()
    if loaded:
        raise RuntimeError(f"forbidden Phase977 holdout module loaded: {loaded}")


def _normalized_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_value(nested)
            for key, nested in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_normalized_value(nested) for nested in value)
    return value


def _values_equal(left: Any, right: Any) -> bool:
    return _normalized_value(left) == _normalized_value(right)


def _assign_options(truth: Any, distractor: Any, answer: str) -> dict[str, Any]:
    if answer not in LABELS:
        raise ValueError(f"unknown answer label: {answer}")
    if _values_equal(truth, distractor):
        raise ValueError("truth and distractor must differ")
    if answer == "A":
        return {"A": deepcopy(truth), "B": deepcopy(distractor)}
    return {"A": deepcopy(distractor), "B": deepcopy(truth)}


def _answer_label(task_index: int, pair_ordinal: int, difficulty: str) -> str:
    difficulty_index = DIFFICULTIES.index(difficulty)
    return LABELS[(task_index + pair_ordinal - 1 + difficulty_index) % 2]


def _solve_multistep(spec: Mapping[str, Any]) -> int:
    value = int(spec["initial"])
    program = spec["program"]
    if not isinstance(program, Sequence) or not program:
        raise ValueError("arithmetic program must be nonempty")
    for instruction in program:
        operation = str(instruction["op"])
        operand = int(instruction["value"])
        if operation == "add":
            value += operand
        elif operation == "subtract":
            value -= operand
        elif operation == "multiply":
            value *= operand
        else:
            raise ValueError(f"unknown arithmetic operation: {operation}")
    return value


def _solve_modular(spec: Mapping[str, Any]) -> int:
    modulus = int(spec["modulus"])
    if modulus <= 2:
        raise ValueError("modulus must exceed two")
    products = spec["products"]
    if not isinstance(products, Sequence) or not products:
        raise ValueError("products must be nonempty")
    total = int(spec["offset"])
    for factors in products:
        if not isinstance(factors, Sequence) or len(factors) != 2:
            raise ValueError("each product must have two factors")
        total += int(factors[0]) * int(factors[1])
    return total % modulus


def _solve_boolean_node(node: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
    if set(node) == {"var"}:
        name = str(node["var"])
        if name not in values:
            raise ValueError(f"unknown Boolean variable: {name}")
        return bool(values[name])
    operation = str(node.get("op", ""))
    arguments = node.get("args")
    if not isinstance(arguments, Sequence):
        raise ValueError("Boolean node args must be a sequence")
    solved = [_solve_boolean_node(argument, values) for argument in arguments]
    if operation == "not" and len(solved) == 1:
        return not solved[0]
    if operation == "and" and len(solved) == 2:
        return solved[0] and solved[1]
    if operation == "or" and len(solved) == 2:
        return solved[0] or solved[1]
    if operation == "xor" and len(solved) == 2:
        return solved[0] != solved[1]
    if operation == "implies" and len(solved) == 2:
        return (not solved[0]) or solved[1]
    raise ValueError(f"invalid Boolean node: {operation}/{len(solved)}")


def _boolean_operator_count(node: Mapping[str, Any]) -> int:
    if set(node) == {"var"}:
        return 0
    return 1 + sum(_boolean_operator_count(child) for child in node["args"])


def _boolean_depth(node: Mapping[str, Any]) -> int:
    if set(node) == {"var"}:
        return 0
    return 1 + max(_boolean_depth(child) for child in node["args"])


def _solve_relation(spec: Mapping[str, Any]) -> bool:
    vertices = [str(value) for value in spec["vertices"]]
    source = str(spec["source"])
    target = str(spec["target"])
    if len(vertices) != len(set(vertices)) or source not in vertices or target not in vertices:
        raise ValueError("invalid graph vertices")
    adjacency = {vertex: [] for vertex in vertices}
    for arc in spec["arcs"]:
        if not isinstance(arc, Sequence) or len(arc) != 2:
            raise ValueError("each arc must contain two endpoints")
        left, right = map(str, arc)
        if left not in adjacency or right not in adjacency:
            raise ValueError("arc endpoint absent from graph")
        adjacency[left].append(right)
    queue = deque([source])
    visited = {source}
    while queue:
        current = queue.popleft()
        if current == target:
            return True
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _solve_machine(spec: Mapping[str, Any]) -> str:
    states = [str(value) for value in spec["states"]]
    symbols = [str(value) for value in spec["symbols"]]
    table = spec["transition_table"]
    state = str(spec["initial_state"])
    if len(states) != len(set(states)) or state not in states:
        raise ValueError("invalid machine state registry")
    if len(symbols) != len(set(symbols)) or not symbols:
        raise ValueError("invalid machine symbol registry")
    for source in states:
        if source not in table or set(table[source]) != set(symbols):
            raise ValueError(f"incomplete transition row: {source}")
        if any(str(table[source][symbol]) not in states for symbol in symbols):
            raise ValueError(f"unknown transition target: {source}")
    for symbol in str(spec["word"]):
        if symbol not in symbols:
            raise ValueError(f"unknown input symbol: {symbol}")
        state = str(table[state][symbol])
    return state


def _solve_sequence(spec: Mapping[str, Any]) -> int:
    values = [int(value) for value in spec["initial_values"]]
    coefficients = [int(value) for value in spec["coefficients"]]
    target_index = int(spec["target_index"])
    bias = int(spec["bias"])
    if not values or len(values) != len(coefficients):
        raise ValueError("sequence order mismatch")
    if target_index < len(values) - 1:
        raise ValueError("target precedes initial sequence")
    while len(values) <= target_index:
        next_value = bias + sum(
            coefficient * values[-offset - 1]
            for offset, coefficient in enumerate(coefficients)
        )
        values.append(next_value)
    return values[target_index]


def _solve_string(spec: Mapping[str, Any]) -> str:
    value = str(spec["source"])
    if not value or not value.isascii() or not value.isalpha():
        raise ValueError("source must be a nonempty ASCII alphabetic string")
    pipeline = spec["pipeline"]
    if not isinstance(pipeline, Sequence) or not pipeline:
        raise ValueError("string pipeline must be nonempty")
    for instruction in pipeline:
        operation = str(instruction["op"])
        if operation in {"rotate_left", "rotate_right"}:
            shift = int(instruction["value"]) % len(value)
            if operation == "rotate_left":
                value = value[shift:] + value[:shift]
            else:
                value = value[-shift:] + value[:-shift] if shift else value
        elif operation == "reverse":
            value = value[::-1]
        elif operation == "uppercase":
            value = value.upper()
        elif operation == "append":
            suffix = str(instruction["value"])
            if len(suffix) != 1 or not suffix.isascii() or not suffix.isalpha():
                raise ValueError("append value must be one ASCII letter")
            value += suffix
        else:
            raise ValueError(f"unknown string operation: {operation}")
    return value


def _order_satisfies(order: Sequence[str], constraints: Sequence[Sequence[str]]) -> bool:
    values = [str(value) for value in order]
    if len(values) != len(set(values)):
        return False
    positions = {value: index for index, value in enumerate(values)}
    for constraint in constraints:
        if not isinstance(constraint, Sequence) or len(constraint) != 2:
            raise ValueError("each precedence constraint needs two values")
        before, after = map(str, constraint)
        if before not in positions or after not in positions:
            return False
        if positions[before] >= positions[after]:
            return False
    return True


def _solve_order(spec: Mapping[str, Any]) -> tuple[str, ...]:
    entities = tuple(str(value) for value in spec["entities"])
    candidates = spec["candidate_orders"]
    if len(entities) != len(set(entities)) or len(entities) < 3:
        raise ValueError("invalid order entity registry")
    valid = [
        tuple(str(value) for value in candidate)
        for candidate in candidates
        if set(map(str, candidate)) == set(entities)
        and _order_satisfies(candidate, spec["precedence"])
    ]
    if len(valid) != 1:
        raise ValueError("exactly one listed order must satisfy the constraints")
    return valid[0]


def _solve(task: str, spec: Mapping[str, Any]) -> Any:
    functions = {
        "multistep_arithmetic": _solve_multistep,
        "modular_arithmetic": _solve_modular,
        "boolean_logic": lambda value: _solve_boolean_node(
            value["expression"], value["values"]
        ),
        "relation_path": _solve_relation,
        "state_machine": _solve_machine,
        "sequence_rule": _solve_sequence,
        "string_transform": _solve_string,
        "constraint_order": _solve_order,
    }
    if task not in functions:
        raise ValueError(f"unknown task: {task}")
    return functions[task](spec)


def _boolean_text(node: Mapping[str, Any]) -> str:
    if set(node) == {"var"}:
        return str(node["var"])
    operation = str(node["op"]).upper()
    children = [_boolean_text(child) for child in node["args"]]
    if operation == "NOT":
        return f"(NOT {children[0]})"
    return f"({children[0]} {operation} {children[1]})"


def _program_text(program: Sequence[Mapping[str, Any]]) -> str:
    rendered = []
    verbs = {"add": "add", "subtract": "subtract", "multiply": "multiply by"}
    for instruction in program:
        operation = str(instruction["op"])
        rendered.append(f"{verbs[operation]} {instruction['value']}")
    return "; then ".join(rendered)


def _problem_text(task: str, spec: Mapping[str, Any]) -> str:
    if task == "multistep_arithmetic":
        return (
            f"Begin with integer {spec['initial']}. In order, {_program_text(spec['program'])}. "
            "Which option is the exact final integer?"
        )
    if task == "modular_arithmetic":
        products = " + ".join(f"({a}*{b})" for a, b in spec["products"])
        return (
            f"Find the least nonnegative remainder of ({products} + {spec['offset']}) "
            f"modulo {spec['modulus']}. Which option is correct?"
        )
    if task == "boolean_logic":
        values = ", ".join(
            f"{name}={'TRUE' if value else 'FALSE'}"
            for name, value in sorted(spec["values"].items())
        )
        return (
            "Use ordinary NOT, AND, OR, XOR, and material IMPLIES rules. "
            f"Given {values}, evaluate {_boolean_text(spec['expression'])}. "
            "Which option is the result?"
        )
    if task == "relation_path":
        arcs = ", ".join(f"{left}->{right}" for left, right in spec["arcs"])
        return (
            f"A directed graph contains exactly these arcs: {arcs}. Is {spec['target']} "
            f"reachable from {spec['source']} by following arrow directions?"
        )
    if task == "state_machine":
        entries = []
        for state in spec["states"]:
            for symbol in spec["symbols"]:
                entries.append(
                    f"{state} on {symbol}->{spec['transition_table'][state][symbol]}"
                )
        return (
            f"A deterministic machine has transitions: {'; '.join(entries)}. It starts "
            f"at {spec['initial_state']} and reads {spec['word']} left to right. Which "
            "option is its final state?"
        )
    if task == "sequence_rule":
        seeds = ", ".join(
            f"a{index}={value}" for index, value in enumerate(spec["initial_values"])
        )
        terms = " + ".join(
            f"{coefficient}*a(n-{offset + 1})"
            for offset, coefficient in enumerate(spec["coefficients"])
        )
        return (
            f"A sequence starts with {seeds}. For each later n, a(n)={terms} + "
            f"{spec['bias']}. Which option is a{spec['target_index']}?"
        )
    if task == "string_transform":
        steps = []
        for instruction in spec["pipeline"]:
            operation = str(instruction["op"])
            if operation == "rotate_left":
                steps.append(f"rotate left by {instruction['value']}")
            elif operation == "rotate_right":
                steps.append(f"rotate right by {instruction['value']}")
            elif operation == "append":
                steps.append(f"append the letter {json.dumps(instruction['value'])}")
            elif operation == "reverse":
                steps.append("reverse all characters")
            elif operation == "uppercase":
                steps.append("convert all letters to uppercase")
        return (
            f"Start with the case-sensitive ASCII string {json.dumps(spec['source'])}. "
            f"In order, {'; then '.join(steps)}. Which option is the exact result?"
        )
    if task == "constraint_order":
        constraints = ", ".join(
            f"{before} before {after}" for before, after in spec["precedence"]
        )
        return (
            "X before Y means X occurs strictly left of Y. The complete constraints "
            f"are: {constraints}. Which listed order satisfies every constraint?"
        )
    raise ValueError(f"unknown task: {task}")


def _render_option(task: str, value: Any) -> str:
    if task in {"boolean_logic"}:
        return "TRUE" if bool(value) else "FALSE"
    if task == "relation_path":
        return "YES" if bool(value) else "NO"
    if task == "constraint_order":
        return " < ".join(str(part) for part in value)
    return str(value)


def _render_prompt(
    task: str, marker: str, spec: Mapping[str, Any], options: Mapping[str, Any],
) -> str:
    return (
        f"{marker}\n"
        f"{_problem_text(task, spec)}\n"
        f"A: {_render_option(task, options['A'])}\n"
        f"B: {_render_option(task, options['B'])}\n"
        f"{_RESPONSE_INSTRUCTION}"
    )


def _difficulty_profile(task: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    if task == "multistep_arithmetic":
        metrics = {"operation_count": len(spec["program"])}
        basis = "ordered_integer_operation_count"
        score = metrics["operation_count"]
    elif task == "modular_arithmetic":
        metrics = {
            "product_count": len(spec["products"]),
            "factor_count": 2 * len(spec["products"]),
        }
        basis = "modular_product_and_factor_count"
        score = metrics["product_count"] + metrics["factor_count"]
    elif task == "boolean_logic":
        metrics = {
            "operator_count": _boolean_operator_count(spec["expression"]),
            "expression_depth": _boolean_depth(spec["expression"]),
        }
        basis = "boolean_operator_count_and_depth"
        score = metrics["operator_count"] + metrics["expression_depth"]
    elif task == "relation_path":
        metrics = {
            "vertex_count": len(spec["vertices"]),
            "arc_count": len(spec["arcs"]),
        }
        basis = "directed_graph_size"
        score = metrics["vertex_count"] + metrics["arc_count"]
    elif task == "state_machine":
        metrics = {
            "state_count": len(spec["states"]),
            "input_length": len(str(spec["word"])),
        }
        basis = "machine_state_count_and_input_length"
        score = metrics["state_count"] + metrics["input_length"]
    elif task == "sequence_rule":
        metrics = {
            "recurrence_order": len(spec["coefficients"]),
            "target_index": int(spec["target_index"]),
        }
        basis = "recurrence_order_and_target_index"
        score = metrics["recurrence_order"] + metrics["target_index"]
    elif task == "string_transform":
        metrics = {
            "pipeline_length": len(spec["pipeline"]),
            "source_length": len(str(spec["source"])),
        }
        basis = "ordered_transform_pipeline_length"
        score = metrics["pipeline_length"]
    elif task == "constraint_order":
        metrics = {
            "entity_count": len(spec["entities"]),
            "constraint_count": len(spec["precedence"]),
        }
        basis = "ordering_entity_and_constraint_count"
        score = metrics["entity_count"] + metrics["constraint_count"]
    else:
        raise ValueError(f"unknown task: {task}")
    return {"basis": basis, "complexity_score": score, "metrics": metrics}


def _make_item(
    *, task: str, task_index: int, pair_ordinal: int, difficulty: str,
    spec: Mapping[str, Any], truth: Any, distractor: Any,
) -> dict[str, Any]:
    answer = _answer_label(task_index, pair_ordinal, difficulty)
    options = _assign_options(truth, distractor, answer)
    marker = _marker(task, pair_ordinal, difficulty)
    profile = _difficulty_profile(task, spec)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "dataset_marker": DATASET_MARKER,
        "id": _item_id(task, pair_ordinal, difficulty),
        "marker": marker,
        "pair_id": _pair_id(task, pair_ordinal),
        "pair_ordinal": pair_ordinal,
        "task": task,
        "difficulty": difficulty,
        "difficulty_structure": {
            "level": difficulty,
            "rank": DIFFICULTIES.index(difficulty) + 1,
            "calibration": "construction_only_not_model_observed",
            "profile": profile,
        },
        "prompt": _render_prompt(task, marker, spec, options),
        "answer": answer,
        "alias_groups": [[answer]],
        "exact": True,
        "options": deepcopy(options),
        "spec": deepcopy(dict(spec)),
        "contracts": {
            "mechanical_truth": True,
            "difficulty_is_structural": True,
            "difficulty_is_model_calibrated": False,
            "holdout_source": False,
            "internal_mechanism_evidence": False,
        },
    }


def _build_multistep(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for pair in range(1, PAIRS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            if difficulty == "easy":
                spec = {
                    "initial": 31 + 4 * pair,
                    "program": [
                        {"op": "add", "value": 3 + pair % 6},
                        {"op": "multiply", "value": 2 + pair % 2},
                        {"op": "subtract", "value": 4 + pair % 5},
                    ],
                }
            else:
                spec = {
                    "initial": 19 + 5 * pair,
                    "program": [
                        {"op": "multiply", "value": 2 + pair % 2},
                        {"op": "add", "value": 7 + pair % 7},
                        {"op": "subtract", "value": 2 + pair % 4},
                        {"op": "multiply", "value": 2 + (pair + 1) % 3},
                        {"op": "add", "value": 5 + pair % 5},
                        {"op": "subtract", "value": 8 + pair % 6},
                    ],
                }
            truth = _solve_multistep(spec)
            distractor = truth + (2 + (pair + DIFFICULTIES.index(difficulty)) % 5)
            rows.append(_make_item(
                task="multistep_arithmetic", task_index=task_index,
                pair_ordinal=pair, difficulty=difficulty, spec=spec,
                truth=truth, distractor=distractor,
            ))
    return rows


def _build_modular(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for pair in range(1, PAIRS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            modulus = 13 + 2 * (pair % 6)
            if difficulty == "easy":
                products = [[5 + pair, 3 + pair % 7]]
                offset = 4 + 2 * pair
            else:
                products = [
                    [7 + pair, 4 + pair % 5],
                    [3 + 2 * pair, 6 + pair % 4],
                    [11 + pair, 2 + pair % 6],
                ]
                offset = 9 + 3 * pair
            spec = {"products": products, "offset": offset, "modulus": modulus}
            truth = _solve_modular(spec)
            distractor = (truth + 1 + pair % (modulus - 1)) % modulus
            rows.append(_make_item(
                task="modular_arithmetic", task_index=task_index,
                pair_ordinal=pair, difficulty=difficulty, spec=spec,
                truth=truth, distractor=distractor,
            ))
    return rows


def _var(name: str) -> dict[str, str]:
    return {"var": name}


def _op(name: str, *args: Mapping[str, Any]) -> dict[str, Any]:
    return {"op": name, "args": [deepcopy(dict(arg)) for arg in args]}


def _build_boolean(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for pair in range(1, PAIRS_PER_TASK + 1):
        bits = (pair * 11 + 3) % 16
        values = {
            "P": bool(bits & 8), "Q": bool(bits & 4),
            "R": bool(bits & 2), "S": bool(bits & 1),
        }
        for difficulty in DIFFICULTIES:
            if difficulty == "easy":
                expression = (
                    _op("xor", _var("P"), _var("Q"))
                    if pair % 2 else _op("implies", _var("R"), _var("S"))
                )
            else:
                if pair % 2:
                    expression = _op(
                        "and",
                        _op("xor", _var("P"), _var("Q")),
                        _op("or", _op("not", _var("R")), _var("S")),
                    )
                else:
                    expression = _op(
                        "or",
                        _op("and", _var("P"), _op("not", _var("Q"))),
                        _op("xor", _var("R"), _op("implies", _var("S"), _var("P"))),
                    )
            spec = {"values": deepcopy(values), "expression": expression}
            truth = _solve_boolean_node(expression, values)
            rows.append(_make_item(
                task="boolean_logic", task_index=task_index,
                pair_ordinal=pair, difficulty=difficulty, spec=spec,
                truth=truth, distractor=not truth,
            ))
    return rows


def _graph_spec(pair: int, difficulty: str) -> dict[str, Any]:
    count = 5 if difficulty == "easy" else 8
    vertices = [f"g{pair:02d}{chr(97 + index)}" for index in range(count)]
    source, target = vertices[0], vertices[-1]
    reachable = pair % 2 == 1
    if difficulty == "easy":
        if reachable:
            arcs = [[vertices[0], vertices[1]], [vertices[1], target],
                    [vertices[0], vertices[2]], [vertices[2], vertices[3]]]
        else:
            arcs = [[vertices[0], vertices[1]], [vertices[1], vertices[2]],
                    [vertices[2], vertices[1]], [target, vertices[3]]]
    else:
        if reachable:
            arcs = [
                [vertices[0], vertices[1]], [vertices[1], vertices[2]],
                [vertices[2], vertices[4]], [vertices[4], target],
                [vertices[0], vertices[3]], [vertices[3], vertices[5]],
                [vertices[5], vertices[3]], [vertices[6], vertices[2]],
                [vertices[4], vertices[1]], [vertices[6], target],
            ]
        else:
            arcs = [
                [vertices[0], vertices[1]], [vertices[1], vertices[2]],
                [vertices[2], vertices[4]], [vertices[4], vertices[1]],
                [vertices[0], vertices[3]], [vertices[3], vertices[5]],
                [vertices[5], vertices[3]], [vertices[6], vertices[2]],
                [target, vertices[6]], [target, vertices[4]],
            ]
    spec = {"vertices": vertices, "arcs": arcs, "source": source, "target": target}
    if _solve_relation(spec) is not reachable:
        raise AssertionError("relation construction changed reachability")
    return spec


def _build_relation(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for pair in range(1, PAIRS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            spec = _graph_spec(pair, difficulty)
            truth = _solve_relation(spec)
            rows.append(_make_item(
                task="relation_path", task_index=task_index,
                pair_ordinal=pair, difficulty=difficulty, spec=spec,
                truth=truth, distractor=not truth,
            ))
    return rows


def _machine_spec(pair: int, difficulty: str) -> dict[str, Any]:
    state_count = 3 if difficulty == "easy" else 5
    prefix = "E" if difficulty == "easy" else "H"
    states = [f"{prefix}{pair:02d}_{index}" for index in range(state_count)]
    symbols = ["x", "y"]
    table: dict[str, dict[str, str]] = {}
    for index, state in enumerate(states):
        table[state] = {
            "x": states[(index + 1 + pair % 2) % state_count],
            "y": states[(2 * index + 1 + pair % 3) % state_count],
        }
    length = 4 if difficulty == "easy" else 9
    word = "".join(symbols[(pair + step * step + step) % 2] for step in range(length))
    return {
        "states": states, "symbols": symbols, "transition_table": table,
        "initial_state": states[pair % state_count], "word": word,
    }


def _build_machine(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for pair in range(1, PAIRS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            spec = _machine_spec(pair, difficulty)
            truth = _solve_machine(spec)
            states = list(spec["states"])
            distractor = states[(states.index(truth) + 1 + pair % (len(states) - 1)) % len(states)]
            rows.append(_make_item(
                task="state_machine", task_index=task_index,
                pair_ordinal=pair, difficulty=difficulty, spec=spec,
                truth=truth, distractor=distractor,
            ))
    return rows


def _build_sequence(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for pair in range(1, PAIRS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            if difficulty == "easy":
                spec = {
                    "initial_values": [2 + pair],
                    "coefficients": [1 + pair % 2],
                    "bias": 1 + pair % 3,
                    "target_index": 4,
                }
            else:
                spec = {
                    "initial_values": [1 + pair % 4, 3 + pair % 5, 5 + pair % 6],
                    "coefficients": [1 + pair % 2, 1, 1],
                    "bias": 1 + pair % 3,
                    "target_index": 9,
                }
            truth = _solve_sequence(spec)
            distractor = truth + 2 + pair % 7
            rows.append(_make_item(
                task="sequence_rule", task_index=task_index,
                pair_ordinal=pair, difficulty=difficulty, spec=spec,
                truth=truth, distractor=distractor,
            ))
    return rows


_FRESH_WORDS = (
    "acorn", "birch", "clover", "dune", "ember", "fern", "harbor", "islet",
    "juniper", "lagoon", "meadow", "nectar", "orchid", "prairie", "saffron", "thistle",
)


def _build_string(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for pair, source in enumerate(_FRESH_WORDS, start=1):
        for difficulty in DIFFICULTIES:
            if difficulty == "easy":
                pipeline = [
                    {"op": "rotate_right", "value": 1 + pair % (len(source) - 1)},
                    {"op": "append", "value": chr(97 + (pair + 7) % 26)},
                ]
            else:
                pipeline = [
                    {"op": "rotate_left", "value": 1 + (2 * pair) % (len(source) - 1)},
                    {"op": "reverse"},
                    {"op": "append", "value": chr(97 + (pair + 11) % 26)},
                    {"op": "uppercase"},
                ]
            spec = {"source": source, "pipeline": pipeline}
            truth = _solve_string(spec)
            replacement = "Z" if truth[-1] != "Z" else "Y"
            distractor = truth[:-1] + replacement
            rows.append(_make_item(
                task="string_transform", task_index=task_index,
                pair_ordinal=pair, difficulty=difficulty, spec=spec,
                truth=truth, distractor=distractor,
            ))
    return rows


def _order_spec(pair: int, difficulty: str) -> dict[str, Any]:
    count = 4 if difficulty == "easy" else 6
    entities = [f"o{pair:02d}{chr(97 + index)}" for index in range(count)]
    rotation = pair % count
    valid = entities[rotation:] + entities[:rotation]
    constraints = [[valid[index], valid[index + 1]] for index in range(count - 1)]
    invalid = list(valid)
    invalid[1], invalid[2] = invalid[2], invalid[1]
    return {
        "entities": entities,
        "precedence": constraints,
        "candidate_orders": [valid, invalid],
    }


def _build_order(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for pair in range(1, PAIRS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            spec = _order_spec(pair, difficulty)
            truth = _solve_order(spec)
            distractor = tuple(spec["candidate_orders"][1])
            rows.append(_make_item(
                task="constraint_order", task_index=task_index,
                pair_ordinal=pair, difficulty=difficulty, spec=spec,
                truth=truth, distractor=distractor,
            ))
    return rows


def build_items() -> list[dict[str, Any]]:
    """Return all 256 items as fresh mutable dictionaries."""
    builders = (
        _build_multistep, _build_modular, _build_boolean, _build_relation,
        _build_machine, _build_sequence, _build_string, _build_order,
    )
    rows: list[dict[str, Any]] = []
    for task_index, builder in enumerate(builders):
        rows.extend(builder(task_index))
    if len(rows) != ITEM_COUNT:
        raise AssertionError(f"builder emitted {len(rows)} rows instead of {ITEM_COUNT}")
    return deepcopy(rows)


def _normalized_prompt_content(prompt: str) -> str:
    """Normalize semantic prompt content without phase marker/answer boilerplate."""
    value = unicodedata.normalize("NFC", str(prompt)).casefold().strip()
    value = _LEADING_MARKER_RE.sub("", value, count=1).strip()
    for instruction in (_RESPONSE_INSTRUCTION, _PHASE979_RESPONSE_INSTRUCTION):
        suffix = unicodedata.normalize("NFC", instruction).casefold()
        if value.endswith(suffix):
            value = value[:-len(suffix)].strip()
    return " ".join(value.split())


def _normalized_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(_normalized_prompt_content(prompt).encode("utf-8")).hexdigest()


def _structural_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": item.get("task"),
        "spec": deepcopy(item.get("spec")),
        "options": deepcopy(item.get("options")),
    }


def _structural_payload_hash(item: Mapping[str, Any]) -> str:
    return _sha256_json(_structural_payload(item))


def _stable_identity(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    stable_items = sorted(
        (deepcopy(dict(item)) for item in items), key=lambda row: str(row.get("id", ""))
    )
    task_counts = dict(sorted(Counter(str(row.get("task", "")) for row in stable_items).items()))
    difficulty_counts = dict(sorted(Counter(
        str(row.get("difficulty", "")) for row in stable_items
    ).items()))
    prompt_hashes = sorted(_normalized_prompt_hash(str(row.get("prompt", ""))) for row in stable_items)
    structural_hashes = sorted(_structural_payload_hash(row) for row in stable_items)
    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "dataset": DATASET_NAME,
        "n_items": len(stable_items),
        "task_counts": task_counts,
        "difficulty_counts": difficulty_counts,
        "items_sha256": _sha256_json(stable_items),
        "normalized_prompt_hashes_sha256": _sha256_json(prompt_hashes),
        "structural_payload_hashes_sha256": _sha256_json(structural_hashes),
    }
    return {**core, "identity_sha256": _sha256_json(core)}


def dataset_identity(items: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = build_items() if items is None else [deepcopy(dict(item)) for item in items]
    return _stable_identity(rows)


def _matching_labels(options: Mapping[str, Any], truth: Any) -> list[str]:
    return [label for label in LABELS if _values_equal(options[label], truth)]


def _phase979_freshness(fresh_rows: list[dict[str, Any]]) -> dict[str, Any]:
    _assert_no_holdout_import()
    source_path = Path(str(phase979_public.__file__)).resolve()
    source_sha = _sha256_file(source_path)
    source_audit = phase979_public.audit_items()
    source_identity = source_audit.get("identity", {})
    errors = []
    if source_sha != EXPECTED_PHASE979_SCRIPT_SHA256:
        errors.append("sealed Phase979 public dataset script hash changed")
    if source_audit.get("passed") is not True:
        errors.append("Phase979 public dataset failed its own audit")
    if source_identity.get("identity_sha256") != EXPECTED_PHASE979_IDENTITY_SHA256:
        errors.append("Phase979 public dataset identity changed")
    if source_identity.get("items_sha256") != EXPECTED_PHASE979_ITEMS_SHA256:
        errors.append("Phase979 public item hash changed")

    source_rows = phase979_public.build_items()
    fresh_prompt_hashes = {_normalized_prompt_hash(row["prompt"]) for row in fresh_rows}
    source_prompt_hashes = {_normalized_prompt_hash(row["prompt"]) for row in source_rows}
    fresh_structural_hashes = {_structural_payload_hash(row) for row in fresh_rows}
    source_structural_hashes = {_structural_payload_hash(row) for row in source_rows}
    prompt_overlap = sorted(fresh_prompt_hashes & source_prompt_hashes)
    structural_overlap = sorted(fresh_structural_hashes & source_structural_hashes)
    if prompt_overlap:
        errors.append(f"normalized Phase979 prompt overlap: {prompt_overlap}")
    if structural_overlap:
        errors.append(f"Phase979 structural payload overlap: {structural_overlap}")
    _assert_no_holdout_import()
    return {
        "passed": not errors,
        "source_dataset": "phase979_diagnostic128_public_only",
        "source_n": len(source_rows),
        "source_script_sha256": source_sha,
        "source_identity_sha256": source_identity.get("identity_sha256"),
        "source_items_sha256": source_identity.get("items_sha256"),
        "normalization_contract": (
            "NFC+casefold+whitespace collapse after removing the leading dataset "
            "marker and either frozen response instruction"
        ),
        "structural_payload_contract": "canonical JSON of task+spec+options",
        "fresh_normalized_prompt_unique_n": len(fresh_prompt_hashes),
        "source_normalized_prompt_unique_n": len(source_prompt_hashes),
        "normalized_prompt_overlap_n": len(prompt_overlap),
        "normalized_prompt_overlap_sha256": _sha256_json(prompt_overlap),
        "fresh_structural_payload_unique_n": len(fresh_structural_hashes),
        "source_structural_payload_unique_n": len(source_structural_hashes),
        "structural_payload_overlap_n": len(structural_overlap),
        "structural_payload_overlap_sha256": _sha256_json(structural_overlap),
        "errors": errors,
    }


def audit_items(items: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Audit schema, truth, balance, paired difficulty, and Phase979 freshness."""
    _assert_no_holdout_import()
    rows = build_items() if items is None else [deepcopy(dict(item)) for item in items]
    errors: list[str] = []
    schema_errors: list[str] = []
    truth_errors: list[str] = []
    prompt_errors: list[str] = []
    difficulty_errors: list[str] = []
    encoding_errors: list[str] = []

    required = {
        "schema_version", "phase", "dataset_marker", "id", "marker", "pair_id",
        "pair_ordinal", "task", "difficulty", "difficulty_structure", "prompt",
        "answer", "alias_groups", "exact", "options", "spec", "contracts",
    }
    ids = [str(row.get("id", "")) for row in rows]
    markers = [str(row.get("marker", "")) for row in rows]
    prompts = [str(row.get("prompt", "")) for row in rows]
    prompt_hashes = [_normalized_prompt_hash(prompt) for prompt in prompts]
    structural_hashes = [_structural_payload_hash(row) for row in rows]
    duplicate_ids = sorted(value for value, n in Counter(ids).items() if n > 1)
    duplicate_markers = sorted(value for value, n in Counter(markers).items() if n > 1)
    duplicate_prompts = sorted(value for value, n in Counter(prompt_hashes).items() if n > 1)
    duplicate_structures = sorted(value for value, n in Counter(structural_hashes).items() if n > 1)

    task_counts = Counter(str(row.get("task", "")) for row in rows)
    difficulty_counts = Counter(str(row.get("difficulty", "")) for row in rows)
    label_counts = Counter(str(row.get("answer", "")) for row in rows)
    stratum_counts: dict[str, dict[str, int]] = {}
    stratum_label_counts: dict[str, dict[str, dict[str, int]]] = {}
    for task in TASKS:
        stratum_counts[task] = {}
        stratum_label_counts[task] = {}
        for difficulty in DIFFICULTIES:
            selected = [row for row in rows if row.get("task") == task and row.get("difficulty") == difficulty]
            stratum_counts[task][difficulty] = len(selected)
            counts = Counter(str(row.get("answer", "")) for row in selected)
            stratum_label_counts[task][difficulty] = {label: counts[label] for label in LABELS}

    mechanically_verified = 0
    unambiguous = 0
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item_id = str(row.get("id", "<missing-id>"))
        missing = sorted(required - set(row))
        if missing:
            schema_errors.append(f"{item_id}: missing fields {missing}")
            continue
        task = str(row["task"])
        difficulty = str(row["difficulty"])
        pair_ordinal = row["pair_ordinal"]
        if row["schema_version"] != SCHEMA_VERSION or row["phase"] != PHASE:
            schema_errors.append(f"{item_id}: schema/phase mismatch")
        if row["dataset_marker"] != DATASET_MARKER:
            schema_errors.append(f"{item_id}: dataset marker mismatch")
        if task not in TASKS or difficulty not in DIFFICULTIES:
            schema_errors.append(f"{item_id}: unknown task/difficulty")
            continue
        if not isinstance(pair_ordinal, int) or isinstance(pair_ordinal, bool) or not 1 <= pair_ordinal <= PAIRS_PER_TASK:
            schema_errors.append(f"{item_id}: invalid pair ordinal")
            continue
        expected_id = _item_id(task, pair_ordinal, difficulty)
        expected_pair_id = _pair_id(task, pair_ordinal)
        expected_marker = _marker(task, pair_ordinal, difficulty)
        if item_id != expected_id or _ITEM_ID_RE.fullmatch(item_id) is None:
            schema_errors.append(f"{item_id}: opaque item id mismatch")
        if row["pair_id"] != expected_pair_id or _PAIR_ID_RE.fullmatch(str(row["pair_id"])) is None:
            schema_errors.append(f"{item_id}: opaque pair id mismatch")
        if row["marker"] != expected_marker or _MARKER_RE.fullmatch(str(row["marker"])) is None:
            prompt_errors.append(f"{item_id}: Phase981 prompt marker mismatch")
        if str(row["prompt"]).count(expected_marker) != 1:
            prompt_errors.append(f"{item_id}: prompt must contain its marker exactly once")
        if _RESPONSE_INSTRUCTION not in str(row["prompt"]):
            prompt_errors.append(f"{item_id}: response instruction missing")
        answer = str(row["answer"])
        if answer not in LABELS or row["alias_groups"] != [[answer]] or row["exact"] is not True:
            schema_errors.append(f"{item_id}: exact answer contract mismatch")
        options = row["options"]
        spec = row["spec"]
        if not isinstance(options, Mapping) or set(options) != set(LABELS):
            schema_errors.append(f"{item_id}: options must contain exactly A/B")
            continue
        if not isinstance(spec, Mapping):
            schema_errors.append(f"{item_id}: spec must be a mapping")
            continue
        if _values_equal(options["A"], options["B"]):
            truth_errors.append(f"{item_id}: duplicate option values")
        try:
            truth = _solve(task, spec)
            matches = _matching_labels(options, truth)
            if len(matches) == 1:
                unambiguous += 1
            else:
                truth_errors.append(f"{item_id}: truth matches {len(matches)} options")
            if matches == [answer]:
                mechanically_verified += 1
            else:
                truth_errors.append(f"{item_id}: answer {answer} disagrees with {matches}")
            expected_prompt = _render_prompt(task, expected_marker, spec, options)
            if row["prompt"] != expected_prompt:
                prompt_errors.append(f"{item_id}: prompt does not exactly encode spec/options")
            profile = _difficulty_profile(task, spec)
            expected_structure = {
                "level": difficulty,
                "rank": DIFFICULTIES.index(difficulty) + 1,
                "calibration": "construction_only_not_model_observed",
                "profile": profile,
            }
            if row["difficulty_structure"] != expected_structure:
                difficulty_errors.append(f"{item_id}: difficulty structure mismatch")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            truth_errors.append(f"{item_id}: solver/profile error: {exc}")

        expected_contracts = {
            "mechanical_truth": True,
            "difficulty_is_structural": True,
            "difficulty_is_model_calibrated": False,
            "holdout_source": False,
            "internal_mechanism_evidence": False,
        }
        if row["contracts"] != expected_contracts:
            schema_errors.append(f"{item_id}: contracts mismatch")
        pair_groups[str(row["pair_id"])].append(row)
        for field in ("id", "marker", "pair_id", "task", "difficulty", "prompt", "answer"):
            value = str(row[field])
            if not value.isascii() or unicodedata.normalize("NFC", value) != value:
                encoding_errors.append(f"{item_id}: {field} must be NFC ASCII")
            if "\ufffd" in value or any(0x80 <= ord(char) <= 0x9F for char in value):
                encoding_errors.append(f"{item_id}: {field} has invalid code point")

    paired_easy_hard_n = 0
    hard_strictly_more_complex_n = 0
    for pair_id, pair_rows in sorted(pair_groups.items()):
        if len(pair_rows) != 2:
            difficulty_errors.append(f"{pair_id}: expected two difficulty rows")
            continue
        by_level = {str(row["difficulty"]): row for row in pair_rows}
        if set(by_level) != set(DIFFICULTIES):
            difficulty_errors.append(f"{pair_id}: missing easy/hard level")
            continue
        easy, hard = by_level["easy"], by_level["hard"]
        if easy["task"] != hard["task"] or easy["pair_ordinal"] != hard["pair_ordinal"]:
            difficulty_errors.append(f"{pair_id}: pair lineage mismatch")
            continue
        if easy["answer"] == hard["answer"]:
            difficulty_errors.append(f"{pair_id}: paired label counterbalance changed")
        paired_easy_hard_n += 1
        easy_score = int(easy["difficulty_structure"]["profile"]["complexity_score"])
        hard_score = int(hard["difficulty_structure"]["profile"]["complexity_score"])
        if hard_score > easy_score:
            hard_strictly_more_complex_n += 1
        else:
            difficulty_errors.append(f"{pair_id}: hard complexity does not exceed easy")

    if len(rows) != ITEM_COUNT:
        errors.append(f"expected {ITEM_COUNT} items, got {len(rows)}")
    if dict(task_counts) != {task: ITEMS_PER_TASK for task in TASKS}:
        errors.append(f"task counts changed: {dict(task_counts)}")
    if dict(difficulty_counts) != {difficulty: ITEM_COUNT // 2 for difficulty in DIFFICULTIES}:
        errors.append(f"difficulty counts changed: {dict(difficulty_counts)}")
    if dict(label_counts) != {"A": ITEM_COUNT // 2, "B": ITEM_COUNT // 2}:
        errors.append(f"global label counts changed: {dict(label_counts)}")
    for task in TASKS:
        for difficulty in DIFFICULTIES:
            if stratum_counts[task][difficulty] != PAIRS_PER_TASK:
                errors.append(f"{task}/{difficulty}: expected 16 items")
            if stratum_label_counts[task][difficulty] != {"A": 8, "B": 8}:
                errors.append(
                    f"{task}/{difficulty}: label balance changed "
                    f"{stratum_label_counts[task][difficulty]}"
                )
    if len(pair_groups) != PAIR_COUNT:
        errors.append(f"expected {PAIR_COUNT} pair ids, got {len(pair_groups)}")
    if duplicate_ids:
        errors.append(f"duplicate item ids: {duplicate_ids}")
    if duplicate_markers:
        errors.append(f"duplicate markers: {duplicate_markers}")
    if duplicate_prompts:
        errors.append(f"duplicate normalized prompts: {duplicate_prompts}")
    if duplicate_structures:
        errors.append(f"duplicate structural payloads: {duplicate_structures}")

    freshness = _phase979_freshness(rows)
    if not freshness["passed"]:
        errors.extend(freshness["errors"])
    errors.extend(schema_errors)
    errors.extend(truth_errors)
    errors.extend(prompt_errors)
    errors.extend(difficulty_errors)
    errors.extend(sorted(set(encoding_errors)))
    _assert_no_holdout_import()
    passed = not errors
    return {
        "ok": passed,
        "passed": passed,
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "dataset": DATASET_NAME,
        "role": "fresh_confirmation_input_design",
        "n_items": len(rows),
        "n_pairs": len(pair_groups),
        "task_counts": {task: task_counts[task] for task in TASKS},
        "difficulty_counts": {difficulty: difficulty_counts[difficulty] for difficulty in DIFFICULTIES},
        "label_counts": {label: label_counts[label] for label in LABELS},
        "task_difficulty_counts": stratum_counts,
        "task_difficulty_label_counts": stratum_label_counts,
        "unique_id_n": len(set(ids)),
        "unique_marker_n": len(set(markers)),
        "unique_normalized_prompt_n": len(set(prompt_hashes)),
        "unique_structural_payload_n": len(set(structural_hashes)),
        "mechanically_verified_n": mechanically_verified,
        "unambiguous_n": unambiguous,
        "paired_easy_hard_n": paired_easy_hard_n,
        "hard_strictly_more_complex_n": hard_strictly_more_complex_n,
        "difficulty_contract": (
            "easy/hard are frozen construction strata; they are not calibrated "
            "from model outputs"
        ),
        "freshness_against_phase979_public": freshness,
        "holdout_accessed": False,
        "holdout_modules_loaded": _forbidden_holdout_modules(),
        "identity": _stable_identity(rows),
        "schema_errors": schema_errors,
        "truth_errors": truth_errors,
        "prompt_errors": prompt_errors,
        "difficulty_errors": difficulty_errors,
        "encoding_errors": sorted(set(encoding_errors)),
        "errors": errors,
    }


def self_test() -> dict[str, Any]:
    _assert_no_holdout_import()
    first = build_items()
    second = build_items()
    deterministic = first == second
    fresh_objects = first is not second and first[0] is not second[0]
    audit = audit_items(first)

    tampered_answer = deepcopy(first)
    tampered_answer[0]["answer"] = "B" if tampered_answer[0]["answer"] == "A" else "A"
    answer_rejected = not audit_items(tampered_answer)["passed"]

    tampered_difficulty = deepcopy(first)
    tampered_difficulty[1]["difficulty_structure"]["profile"]["complexity_score"] = 0
    difficulty_rejected = not audit_items(tampered_difficulty)["passed"]

    tampered_prompt = deepcopy(first)
    tampered_prompt[1]["prompt"] = tampered_prompt[0]["prompt"]
    duplicate_prompt_rejected = not audit_items(tampered_prompt)["passed"]

    checks = {
        "deterministic_build": deterministic,
        "fresh_mutable_objects": fresh_objects,
        "formal_audit_passed": audit["passed"],
        "answer_tamper_rejected": answer_rejected,
        "difficulty_tamper_rejected": difficulty_rejected,
        "duplicate_prompt_rejected": duplicate_prompt_rejected,
        "phase979_normalized_prompt_overlap_zero": (
            audit["freshness_against_phase979_public"]["normalized_prompt_overlap_n"] == 0
        ),
        "phase979_structural_payload_overlap_zero": (
            audit["freshness_against_phase979_public"]["structural_payload_overlap_n"] == 0
        ),
        "holdout_modules_absent": not _forbidden_holdout_modules(),
    }
    return {"passed": all(checks.values()), "checks": checks, "identity": audit["identity"]}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _install_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_artifacts() -> dict[str, Any]:
    _assert_no_holdout_import()
    tests = self_test()
    if not tests["passed"]:
        raise RuntimeError(f"fresh256 self-test failed: {tests}")
    items = build_items()
    audit = audit_items(items)
    if not audit["passed"]:
        raise RuntimeError(f"fresh256 audit failed: {audit['errors']}")
    dataset_payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "dataset": DATASET_NAME,
        "role": "fresh_confirmation_input_design",
        "dataset_marker": DATASET_MARKER,
        "contracts": {
            "item_count": ITEM_COUNT,
            "pair_count": PAIR_COUNT,
            "task_count": len(TASKS),
            "difficulties": list(DIFFICULTIES),
            "mechanically_solved": True,
            "difficulty_is_structural_not_model_calibrated": True,
            "phase979_prompt_and_structural_overlap_zero": True,
            "phase977_holdout_accessed": False,
            "model_weights_loaded": False,
            "generation_performed": False,
        },
        "identity": audit["identity"],
        "items": items,
    }
    dataset_document = {
        **dataset_payload,
        "dataset_sha256": _sha256_json(dataset_payload),
    }
    dataset_bytes = _json_bytes(dataset_document)
    dataset_file_sha = hashlib.sha256(dataset_bytes).hexdigest()
    script_sha = _sha256_file(Path(__file__).resolve())
    audit_payload = {
        **audit,
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "script_sha256": script_sha,
        "dataset_path": DATASET_PATH.resolve().relative_to(ROOT).as_posix(),
        "dataset_document_sha256": dataset_document["dataset_sha256"],
        "dataset_file_sha256": dataset_file_sha,
        "self_test": tests,
        "cpu_only": True,
        "gpu_used": False,
        "model_weights_loaded": False,
        "generation_performed": False,
    }
    audit_document = {**audit_payload, "audit_sha256": _sha256_json(audit_payload)}
    audit_bytes = _json_bytes(audit_document)
    _install_exact(DATASET_PATH, dataset_bytes)
    _install_exact(AUDIT_PATH, audit_bytes)
    _assert_no_holdout_import()
    return {
        "passed": True,
        "script_sha256": script_sha,
        "dataset_identity_sha256": audit["identity"]["identity_sha256"],
        "items_sha256": audit["identity"]["items_sha256"],
        "dataset_document_sha256": dataset_document["dataset_sha256"],
        "dataset_file_sha256": dataset_file_sha,
        "audit_sha256": audit_document["audit_sha256"],
        "audit_file_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "dataset_path": str(DATASET_PATH),
        "audit_path": str(AUDIT_PATH),
        "holdout_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run deterministic fail-closed tests")
    parser.add_argument("--write", action="store_true", help="install frozen dataset.json and audit.json")
    args = parser.parse_args()
    _assert_no_holdout_import()
    if args.write:
        result = write_artifacts()
    elif args.self_test:
        result = self_test()
    else:
        audit = audit_items()
        result = {
            "passed": audit["passed"],
            "n_items": audit["n_items"],
            "identity": audit["identity"],
            "freshness_against_phase979_public": audit["freshness_against_phase979_public"],
            "holdout_accessed": False,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
