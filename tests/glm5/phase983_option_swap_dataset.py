#!/usr/bin/env python3
"""Build and audit the CPU-only Phase 983 fresh option-swap corpus.

The dataset contains 128 mechanically decidable semantic instances:

    8 tasks * 8 ordinals * 2 structural difficulty levels = 128 instances.

Every semantic instance has exactly two rows.  The ``swapped`` row differs in
its substantive problem representation only by exchanging options A/B and the
gold label.  Therefore the final corpus has 256 rows and, within every
task x difficulty stratum, labels A/B occur 8/8.

Prompts contain only a fresh opaque marker, the problem, and the two options.
They deliberately contain no response-format or model-control instruction;
the later cross-model protocol owns that external contract.  Freshness is
checked against the public Phase 979 diagnostic128 and Phase 981 fresh256
corpora using normalized prompt-content and structural-payload hashes.  This
module never imports or reads a Phase 977 holdout artifact and never imports a
model runtime.
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
import phase981_fresh_dataset as phase981_public


PHASE = 983
SCHEMA_VERSION = 1
DATASET_NAME = "phase983_fresh_option_swap256"
DATASET_MARKER = "phase983_cross_model_option_swap_v1"
PROMPT_MARKER_NAMESPACE = "P983-XMODEL-OPTSWAP-V1"
SEED_NAMESPACE = "P983-XMODEL-OPTSWAP-SEED-V1"
DIFFICULTIES = ("easy", "hard")
SWAP_SIDES = ("original", "swapped")
LABELS = ("A", "B")
ORDINALS_PER_TASK = 8
SEMANTIC_INSTANCES_PER_TASK = ORDINALS_PER_TASK * len(DIFFICULTIES)
ITEMS_PER_TASK = SEMANTIC_INSTANCES_PER_TASK * len(SWAP_SIDES)
SEMANTIC_INSTANCE_COUNT = 128
ITEM_COUNT = 256
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
OUT = ROOT / "tests" / "glm5" / "result" / "phase983_cross_model_external_contract"
DATASET_PATH = OUT / "dataset.json"
AUDIT_PATH = OUT / "dataset_audit.json"

EXPECTED_PHASE979_SCRIPT_SHA256 = (
    "74875eb5e00253f952969904a0164a06920e995c82735b30b1c307535b853605"
)
EXPECTED_PHASE979_IDENTITY_SHA256 = (
    "2da762df071a8a096feb017bd9fbf640454e056860bec2ac1c226fc55243330a"
)
EXPECTED_PHASE979_ITEMS_SHA256 = (
    "e884f922d77482baded1da55562df81685808dc0718049e26413ebacf56ece10"
)
EXPECTED_PHASE981_SCRIPT_SHA256 = (
    "e1c1c2127616fd328fe44d3b8dea27752df69618659e302820403b8102f4307f"
)
EXPECTED_PHASE981_IDENTITY_SHA256 = (
    "e5a12cedb7e4ab0e56896975afd58285c3401b1edf138c2d081e431fab8fa6fd"
)
EXPECTED_PHASE981_ITEMS_SHA256 = (
    "16ab8afbc06eff78d766e9a169ac45554878961fc247219483d09418c47d4eba"
)

_PHASE979_RESPONSE_INSTRUCTION = (
    "Return only A or B as the final answer. One optional final ASCII period "
    "is allowed; output nothing else."
)
_PHASE981_RESPONSE_INSTRUCTION = (
    "Respond with exactly one label, A or B. A single trailing ASCII period "
    "is permitted. Do not add any other text."
)
_ITEM_ID_RE = re.compile(r"^p983_x_[0-9a-f]{20}$")
_SEMANTIC_ID_RE = re.compile(r"^p983_sem_[0-9a-f]{20}$")
_SEED_KEY_RE = re.compile(r"^p983_seed_[0-9a-f]{20}$")
_MARKER_RE = re.compile(r"^\[P983-XMODEL-OPTSWAP-V1\|[0-9A-F]{20}\]$")
_LEADING_MARKER_RE = re.compile(r"^\[[^\]\r\n]+\]\s*")
_FORBIDDEN_RUNTIME_ROOTS = ("torch", "transformers", "model_utils")


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


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _load_json_strict_bytes(payload: bytes) -> Any:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def _opaque(namespace: str, *parts: Any) -> str:
    payload = "|".join([DATASET_MARKER, namespace, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _semantic_id(task: str, ordinal: int, difficulty: str) -> str:
    return f"p983_sem_{_opaque('semantic', task, ordinal, difficulty)}"


def _item_id(task: str, ordinal: int, difficulty: str, swap_side: str) -> str:
    return f"p983_x_{_opaque('item', task, ordinal, difficulty, swap_side)}"


def _seed_key(task: str, ordinal: int, difficulty: str) -> str:
    return f"p983_seed_{_opaque('seed', task, ordinal, difficulty)}"


def _marker(task: str, ordinal: int, difficulty: str) -> str:
    """Return one model-visible marker shared by both rows of a semantic twin."""
    code = _opaque("marker", task, ordinal, difficulty).upper()
    return f"[{PROMPT_MARKER_NAMESPACE}|{code}]"


def _forbidden_holdout_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if "phase977" in name.casefold() and "holdout" in name.casefold()
    )


def _forbidden_runtime_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0].casefold() in _FORBIDDEN_RUNTIME_ROOTS
    )


def _assert_cpu_dataset_scope() -> None:
    holdout = _forbidden_holdout_modules()
    runtimes = _forbidden_runtime_modules()
    if holdout:
        raise RuntimeError(f"forbidden Phase977 holdout module loaded: {holdout}")
    if runtimes:
        raise RuntimeError(f"forbidden model runtime loaded: {runtimes}")


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


def _json_safe_value(value: Any) -> Any:
    """Return a deep JSON-native copy (notably converting tuples to lists)."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(nested) for nested in value]
    return deepcopy(value)


def _assign_options(truth: Any, distractor: Any, answer: str) -> dict[str, Any]:
    if answer not in LABELS:
        raise ValueError(f"unknown answer label: {answer}")
    if _values_equal(truth, distractor):
        raise ValueError("truth and distractor must differ")
    if answer == "A":
        return {"A": deepcopy(truth), "B": deepcopy(distractor)}
    return {"A": deepcopy(distractor), "B": deepcopy(truth)}


def _opposite_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"unknown label: {label}")
    return "B" if label == "A" else "A"


def _original_label(task_index: int, ordinal: int, difficulty: str) -> str:
    difficulty_index = DIFFICULTIES.index(difficulty)
    return LABELS[(task_index + ordinal - 1 + difficulty_index) % 2]


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
            raise ValueError("each product must have exactly two factors")
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
        raise ValueError("Boolean args must be a sequence")
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
        raise ValueError("invalid graph registry")
    adjacency = {vertex: [] for vertex in vertices}
    for arc in spec["arcs"]:
        if not isinstance(arc, Sequence) or len(arc) != 2:
            raise ValueError("each directed arc must contain two endpoints")
        left, right = map(str, arc)
        if left not in adjacency or right not in adjacency:
            raise ValueError("graph arc endpoint is absent")
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
            raise ValueError(f"unknown transition target from {source}")
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
        raise ValueError("sequence recurrence order mismatch")
    if target_index < len(values) - 1:
        raise ValueError("sequence target precedes initial values")
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
            raise ValueError("each precedence constraint requires two entities")
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
        raise ValueError("exactly one listed order must satisfy all constraints")
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


def _var(name: str) -> dict[str, str]:
    return {"var": name}


def _op(name: str, *args: Mapping[str, Any]) -> dict[str, Any]:
    return {"op": name, "args": [deepcopy(dict(arg)) for arg in args]}


def _boolean_text(node: Mapping[str, Any]) -> str:
    if set(node) == {"var"}:
        return str(node["var"])
    operation = str(node["op"]).upper()
    children = [_boolean_text(child) for child in node["args"]]
    if operation == "NOT":
        return f"(NOT {children[0]})"
    return f"({children[0]} {operation} {children[1]})"


def _program_text(program: Sequence[Mapping[str, Any]]) -> str:
    verbs = {"add": "add", "subtract": "subtract", "multiply": "multiply by"}
    return "; then ".join(
        f"{verbs[str(instruction['op'])]} {instruction['value']}"
        for instruction in program
    )


def _problem_text(task: str, spec: Mapping[str, Any]) -> str:
    if task == "multistep_arithmetic":
        return (
            f"Start at integer {spec['initial']}. In order, {_program_text(spec['program'])}. "
            "Which option gives the exact resulting integer?"
        )
    if task == "modular_arithmetic":
        products = " + ".join(f"({a}*{b})" for a, b in spec["products"])
        return (
            f"Compute the least nonnegative remainder of ({products} + "
            f"{spec['offset']}) modulo {spec['modulus']}. Which option is correct?"
        )
    if task == "boolean_logic":
        values = ", ".join(
            f"{name}={'TRUE' if value else 'FALSE'}"
            for name, value in sorted(spec["values"].items())
        )
        return (
            "Use ordinary NOT, AND, OR, XOR, and material IMPLIES. "
            f"Given {values}, evaluate {_boolean_text(spec['expression'])}."
        )
    if task == "relation_path":
        arcs = ", ".join(f"{left}->{right}" for left, right in spec["arcs"])
        return (
            f"A directed graph has exactly these arcs: {arcs}. Can {spec['target']} "
            f"be reached from {spec['source']} while following arrow directions?"
        )
    if task == "state_machine":
        entries = []
        for state in spec["states"]:
            for symbol in spec["symbols"]:
                entries.append(
                    f"{state} on {symbol}->{spec['transition_table'][state][symbol]}"
                )
        return (
            f"A deterministic machine uses these transitions: {'; '.join(entries)}. "
            f"It starts at {spec['initial_state']} and reads {spec['word']} from left "
            "to right. Which option is its final state?"
        )
    if task == "sequence_rule":
        seeds = ", ".join(
            f"b{index}={value}" for index, value in enumerate(spec["initial_values"])
        )
        terms = " + ".join(
            f"{coefficient}*b(n-{offset + 1})"
            for offset, coefficient in enumerate(spec["coefficients"])
        )
        return (
            f"A sequence begins {seeds}. For every later n, b(n)={terms} + "
            f"{spec['bias']}. Which option equals b{spec['target_index']}?"
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
                steps.append(f"append {json.dumps(instruction['value'])}")
            elif operation == "reverse":
                steps.append("reverse every character")
            elif operation == "uppercase":
                steps.append("convert every letter to uppercase")
        return (
            f"Begin with the case-sensitive ASCII string {json.dumps(spec['source'])}. "
            f"In order, {'; then '.join(steps)}. Which option is the exact output?"
        )
    if task == "constraint_order":
        constraints = ", ".join(
            f"{before} before {after}" for before, after in spec["precedence"]
        )
        return (
            "X before Y means X is strictly left of Y. The complete constraints "
            f"are: {constraints}. Which listed order satisfies every constraint?"
        )
    raise ValueError(f"unknown task: {task}")


def _render_option(task: str, value: Any) -> str:
    if task == "boolean_logic":
        return "TRUE" if bool(value) else "FALSE"
    if task == "relation_path":
        return "YES" if bool(value) else "NO"
    if task == "constraint_order":
        return " < ".join(str(part) for part in value)
    return str(value)


def _render_prompt(
    task: str,
    marker: str,
    spec: Mapping[str, Any],
    options: Mapping[str, Any],
) -> str:
    return (
        f"{marker}\n"
        f"{_problem_text(task, spec)}\n"
        f"A: {_render_option(task, options['A'])}\n"
        f"B: {_render_option(task, options['B'])}"
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
            "symbol_count": len(spec["symbols"]),
            "input_length": len(str(spec["word"])),
        }
        basis = "machine_registry_and_input_length"
        score = metrics["state_count"] + metrics["symbol_count"] + metrics["input_length"]
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


def _make_twins(
    *,
    task: str,
    task_index: int,
    ordinal: int,
    difficulty: str,
    spec: Mapping[str, Any],
    truth: Any,
    distractor: Any,
) -> list[dict[str, Any]]:
    if not _values_equal(_solve(task, spec), truth):
        raise AssertionError("builder truth disagrees with mechanical solver")
    truth = _json_safe_value(truth)
    distractor = _json_safe_value(distractor)
    original_answer = _original_label(task_index, ordinal, difficulty)
    original_options = _assign_options(truth, distractor, original_answer)
    semantic_id = _semantic_id(task, ordinal, difficulty)
    seed_key = _seed_key(task, ordinal, difficulty)
    profile = _difficulty_profile(task, spec)
    rows = []
    for swap_side in SWAP_SIDES:
        if swap_side == "original":
            answer = original_answer
            options = deepcopy(original_options)
        else:
            answer = _opposite_label(original_answer)
            options = {
                "A": deepcopy(original_options["B"]),
                "B": deepcopy(original_options["A"]),
            }
        marker = _marker(task, ordinal, difficulty)
        prompt = _render_prompt(task, marker, spec, options)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase": PHASE,
                "dataset_marker": DATASET_MARKER,
                "id": _item_id(task, ordinal, difficulty, swap_side),
                "marker": marker,
                "semantic_id": semantic_id,
                "semantic_ordinal": ordinal,
                "swap_side": swap_side,
                "seed_namespace": SEED_NAMESPACE,
                "seed_key": seed_key,
                "task": task,
                "difficulty": difficulty,
                "difficulty_structure": {
                    "level": difficulty,
                    "rank": DIFFICULTIES.index(difficulty) + 1,
                    "calibration": "construction_only_not_model_observed",
                    "profile": profile,
                },
                "prompt": prompt,
                "problem_prompt": prompt,
                "answer": answer,
                "alias_groups": [[answer]],
                "exact": True,
                "options": deepcopy(options),
                "spec": deepcopy(dict(spec)),
                "truth": deepcopy(truth),
                "distractor": deepcopy(distractor),
                "contracts": {
                    "mechanical_truth": True,
                    "option_swap_twin": True,
                    "prompt_has_response_instruction": False,
                    "difficulty_is_structural": True,
                    "difficulty_is_model_calibrated": False,
                    "holdout_source": False,
                    "internal_mechanism_evidence": False,
                },
            }
        )
    return rows


def _build_multistep(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for ordinal in range(1, ORDINALS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            if difficulty == "easy":
                spec = {
                    "initial": 113 + 7 * ordinal,
                    "program": [
                        {"op": "subtract", "value": 9 + ordinal % 5},
                        {"op": "multiply", "value": 2 + (ordinal + 1) % 2},
                        {"op": "add", "value": 17 + 3 * ordinal},
                    ],
                }
            else:
                spec = {
                    "initial": 67 + 9 * ordinal,
                    "program": [
                        {"op": "add", "value": 13 + 2 * ordinal},
                        {"op": "multiply", "value": 2 + ordinal % 3},
                        {"op": "subtract", "value": 11 + ordinal % 7},
                        {"op": "add", "value": 19 + ordinal},
                        {"op": "multiply", "value": 2 + (ordinal + 1) % 2},
                        {"op": "subtract", "value": 23 + 2 * ordinal},
                        {"op": "add", "value": 5 + ordinal % 4},
                    ],
                }
            truth = _solve_multistep(spec)
            distractor = truth + 4 + ordinal + DIFFICULTIES.index(difficulty)
            rows.extend(
                _make_twins(
                    task="multistep_arithmetic",
                    task_index=task_index,
                    ordinal=ordinal,
                    difficulty=difficulty,
                    spec=spec,
                    truth=truth,
                    distractor=distractor,
                )
            )
    return rows


def _build_modular(task_index: int) -> list[dict[str, Any]]:
    rows = []
    moduli = (29, 31, 37, 41, 43, 47, 53, 59)
    for ordinal, modulus in enumerate(moduli, start=1):
        for difficulty in DIFFICULTIES:
            if difficulty == "easy":
                products = [
                    [17 + ordinal, 9 + ordinal % 4],
                    [23 + 2 * ordinal, 5 + ordinal % 6],
                ]
                offset = 31 + 4 * ordinal
            else:
                products = [
                    [29 + ordinal, 11 + ordinal % 5],
                    [13 + 3 * ordinal, 17 + ordinal % 7],
                    [37 + ordinal, 7 + ordinal % 6],
                    [19 + 2 * ordinal, 23 + ordinal % 4],
                ]
                offset = 47 + 5 * ordinal
            spec = {"products": products, "offset": offset, "modulus": modulus}
            truth = _solve_modular(spec)
            distractor = (truth + ordinal + 1) % modulus
            rows.extend(
                _make_twins(
                    task="modular_arithmetic",
                    task_index=task_index,
                    ordinal=ordinal,
                    difficulty=difficulty,
                    spec=spec,
                    truth=truth,
                    distractor=distractor,
                )
            )
    return rows


def _build_boolean(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for ordinal in range(1, ORDINALS_PER_TASK + 1):
        bits = (ordinal * 19 + 5) % 16
        values = {
            "U": bool(bits & 8),
            "V": bool(bits & 4),
            "W": bool(bits & 2),
            "X": bool(bits & 1),
        }
        for difficulty in DIFFICULTIES:
            if difficulty == "easy":
                if ordinal % 2:
                    expression = _op("and", _var("U"), _op("not", _var("V")))
                else:
                    expression = _op("xor", _var("W"), _var("X"))
            else:
                if ordinal % 2:
                    expression = _op(
                        "implies",
                        _op(
                            "or",
                            _op("and", _var("U"), _op("not", _var("V"))),
                            _var("W"),
                        ),
                        _op("xor", _var("X"), _op("not", _var("U"))),
                    )
                else:
                    expression = _op(
                        "xor",
                        _op("and", _var("U"), _op("or", _var("V"), _var("W"))),
                        _op("implies", _op("not", _var("X")), _var("V")),
                    )
            spec = {"values": deepcopy(values), "expression": expression}
            truth = _solve_boolean_node(expression, values)
            rows.extend(
                _make_twins(
                    task="boolean_logic",
                    task_index=task_index,
                    ordinal=ordinal,
                    difficulty=difficulty,
                    spec=spec,
                    truth=truth,
                    distractor=not truth,
                )
            )
    return rows


def _graph_spec(ordinal: int, difficulty: str) -> dict[str, Any]:
    count = 6 if difficulty == "easy" else 9
    vertices = [f"z83{ordinal}{chr(97 + index)}" for index in range(count)]
    source, target = vertices[0], vertices[-1]
    reachable = ordinal % 2 == 0
    if difficulty == "easy":
        if reachable:
            arcs = [
                [vertices[0], vertices[2]],
                [vertices[2], target],
                [vertices[0], vertices[1]],
                [vertices[1], vertices[3]],
                [vertices[3], vertices[1]],
            ]
        else:
            arcs = [
                [vertices[0], vertices[2]],
                [vertices[2], vertices[1]],
                [vertices[1], vertices[3]],
                [vertices[3], vertices[2]],
                [target, vertices[4]],
            ]
    else:
        if reachable:
            arcs = [
                [vertices[0], vertices[1]],
                [vertices[1], vertices[3]],
                [vertices[3], vertices[5]],
                [vertices[5], target],
                [vertices[0], vertices[2]],
                [vertices[2], vertices[4]],
                [vertices[4], vertices[2]],
                [vertices[6], vertices[7]],
                [vertices[7], vertices[3]],
                [vertices[5], vertices[1]],
                [vertices[6], target],
            ]
        else:
            arcs = [
                [vertices[0], vertices[1]],
                [vertices[1], vertices[3]],
                [vertices[3], vertices[5]],
                [vertices[5], vertices[1]],
                [vertices[0], vertices[2]],
                [vertices[2], vertices[4]],
                [vertices[4], vertices[2]],
                [vertices[6], vertices[7]],
                [vertices[7], vertices[4]],
                [target, vertices[6]],
                [target, vertices[3]],
            ]
    spec = {"vertices": vertices, "arcs": arcs, "source": source, "target": target}
    if _solve_relation(spec) is not reachable:
        raise AssertionError("relation construction changed reachability")
    return spec


def _build_relation(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for ordinal in range(1, ORDINALS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            spec = _graph_spec(ordinal, difficulty)
            truth = _solve_relation(spec)
            rows.extend(
                _make_twins(
                    task="relation_path",
                    task_index=task_index,
                    ordinal=ordinal,
                    difficulty=difficulty,
                    spec=spec,
                    truth=truth,
                    distractor=not truth,
                )
            )
    return rows


def _machine_spec(ordinal: int, difficulty: str) -> dict[str, Any]:
    state_count = 4 if difficulty == "easy" else 6
    prefix = "J" if difficulty == "easy" else "K"
    states = [f"{prefix}83{ordinal}_{index}" for index in range(state_count)]
    symbols = ["m", "n", "p"]
    table: dict[str, dict[str, str]] = {}
    for index, state in enumerate(states):
        table[state] = {
            "m": states[(index + 1 + ordinal % 2) % state_count],
            "n": states[(2 * index + ordinal + 1) % state_count],
            "p": states[(3 * index + ordinal + 2) % state_count],
        }
    length = 6 if difficulty == "easy" else 12
    word = "".join(symbols[(ordinal + step * step + 2 * step) % 3] for step in range(length))
    return {
        "states": states,
        "symbols": symbols,
        "transition_table": table,
        "initial_state": states[(ordinal + 1) % state_count],
        "word": word,
    }


def _build_machine(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for ordinal in range(1, ORDINALS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            spec = _machine_spec(ordinal, difficulty)
            truth = _solve_machine(spec)
            states = list(spec["states"])
            distractor = states[(states.index(truth) + 1 + ordinal) % len(states)]
            if distractor == truth:
                distractor = states[(states.index(truth) + 1) % len(states)]
            rows.extend(
                _make_twins(
                    task="state_machine",
                    task_index=task_index,
                    ordinal=ordinal,
                    difficulty=difficulty,
                    spec=spec,
                    truth=truth,
                    distractor=distractor,
                )
            )
    return rows


def _build_sequence(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for ordinal in range(1, ORDINALS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            if difficulty == "easy":
                spec = {
                    "initial_values": [7 + ordinal, 11 + 2 * ordinal],
                    "coefficients": [1, 1 + ordinal % 2],
                    "bias": 2 + ordinal % 4,
                    "target_index": 6,
                }
            else:
                spec = {
                    "initial_values": [
                        3 + ordinal,
                        5 + ordinal % 3,
                        8 + ordinal % 4,
                        13 + ordinal % 5,
                    ],
                    "coefficients": [1, 2, -1, 1 + ordinal % 2],
                    "bias": 3 + ordinal % 4,
                    "target_index": 11,
                }
            truth = _solve_sequence(spec)
            distractor = truth - (3 + ordinal)
            rows.extend(
                _make_twins(
                    task="sequence_rule",
                    task_index=task_index,
                    ordinal=ordinal,
                    difficulty=difficulty,
                    spec=spec,
                    truth=truth,
                    distractor=distractor,
                )
            )
    return rows


_FRESH_WORDS = (
    "vexora",
    "qumber",
    "zalpic",
    "norfel",
    "tavrix",
    "beldun",
    "kyroth",
    "wispan",
)


def _build_string(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for ordinal, source in enumerate(_FRESH_WORDS, start=1):
        for difficulty in DIFFICULTIES:
            if difficulty == "easy":
                pipeline = [
                    {"op": "rotate_left", "value": 1 + ordinal % (len(source) - 1)},
                    {"op": "reverse"},
                    {"op": "append", "value": chr(97 + (ordinal + 17) % 26)},
                ]
            else:
                pipeline = [
                    {"op": "rotate_right", "value": 1 + 2 * ordinal % (len(source) - 1)},
                    {"op": "append", "value": chr(97 + (ordinal + 21) % 26)},
                    {"op": "reverse"},
                    {"op": "rotate_left", "value": 2 + ordinal % 3},
                    {"op": "uppercase"},
                ]
            spec = {"source": source, "pipeline": pipeline}
            truth = _solve_string(spec)
            if truth[-1].isupper():
                replacement = "X" if truth[-1] != "X" else "Y"
            else:
                replacement = "x" if truth[-1] != "x" else "y"
            distractor = truth[:-1] + replacement
            rows.extend(
                _make_twins(
                    task="string_transform",
                    task_index=task_index,
                    ordinal=ordinal,
                    difficulty=difficulty,
                    spec=spec,
                    truth=truth,
                    distractor=distractor,
                )
            )
    return rows


def _order_spec(ordinal: int, difficulty: str) -> dict[str, Any]:
    count = 5 if difficulty == "easy" else 7
    entities = [f"u83{ordinal}{chr(97 + index)}" for index in range(count)]
    rotation = (2 * ordinal + 1) % count
    valid = entities[rotation:] + entities[:rotation]
    constraints = [[valid[index], valid[index + 1]] for index in range(count - 1)]
    invalid = list(valid)
    invalid[2], invalid[3] = invalid[3], invalid[2]
    return {
        "entities": entities,
        "precedence": constraints,
        "candidate_orders": [valid, invalid],
    }


def _build_order(task_index: int) -> list[dict[str, Any]]:
    rows = []
    for ordinal in range(1, ORDINALS_PER_TASK + 1):
        for difficulty in DIFFICULTIES:
            spec = _order_spec(ordinal, difficulty)
            truth = _solve_order(spec)
            distractor = tuple(spec["candidate_orders"][1])
            rows.extend(
                _make_twins(
                    task="constraint_order",
                    task_index=task_index,
                    ordinal=ordinal,
                    difficulty=difficulty,
                    spec=spec,
                    truth=truth,
                    distractor=distractor,
                )
            )
    return rows


def build_items() -> list[dict[str, Any]]:
    """Return all 256 rows as newly allocated mutable dictionaries."""
    _assert_cpu_dataset_scope()
    builders = (
        _build_multistep,
        _build_modular,
        _build_boolean,
        _build_relation,
        _build_machine,
        _build_sequence,
        _build_string,
        _build_order,
    )
    rows: list[dict[str, Any]] = []
    for task_index, builder in enumerate(builders):
        rows.extend(builder(task_index))
    if len(rows) != ITEM_COUNT:
        raise AssertionError(f"builder emitted {len(rows)} rows instead of {ITEM_COUNT}")
    return deepcopy(rows)


def _normalized_prompt_content(prompt: str) -> str:
    """Remove marker/legacy response boilerplate, then normalize problem+options."""
    value = unicodedata.normalize("NFC", str(prompt)).casefold().strip()
    value = _LEADING_MARKER_RE.sub("", value, count=1).strip()
    for instruction in (_PHASE979_RESPONSE_INSTRUCTION, _PHASE981_RESPONSE_INSTRUCTION):
        suffix = unicodedata.normalize("NFC", instruction).casefold()
        if value.endswith(suffix):
            value = value[: -len(suffix)].strip()
    return " ".join(value.split())


def _normalized_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(_normalized_prompt_content(prompt).encode("utf-8")).hexdigest()


def _structural_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return semantic construction identity independent of option placement."""
    return {
        "task": item.get("task"),
        "spec": deepcopy(item.get("spec")),
    }


def _option_structural_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": item.get("task"),
        "spec": deepcopy(item.get("spec")),
        "options": deepcopy(item.get("options")),
    }


def _structural_payload_hash(item: Mapping[str, Any]) -> str:
    return _sha256_json(_structural_payload(item))


def _option_structural_payload_hash(item: Mapping[str, Any]) -> str:
    return _sha256_json(_option_structural_payload(item))


def _stable_identity(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    stable_items = sorted(
        (deepcopy(dict(item)) for item in items),
        key=lambda row: str(row.get("id", "")),
    )
    prompt_hashes = sorted(
        _normalized_prompt_hash(str(row.get("prompt", ""))) for row in stable_items
    )
    structural_hashes = sorted(_structural_payload_hash(row) for row in stable_items)
    option_structural_hashes = sorted(
        _option_structural_payload_hash(row) for row in stable_items
    )
    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "dataset": DATASET_NAME,
        "n_items": len(stable_items),
        "n_semantic_instances": len(
            {str(row.get("semantic_id", "")) for row in stable_items}
        ),
        "task_counts": dict(
            sorted(Counter(str(row.get("task", "")) for row in stable_items).items())
        ),
        "difficulty_counts": dict(
            sorted(
                Counter(str(row.get("difficulty", "")) for row in stable_items).items()
            )
        ),
        "swap_side_counts": dict(
            sorted(Counter(str(row.get("swap_side", "")) for row in stable_items).items())
        ),
        "items_sha256": _sha256_json(stable_items),
        "normalized_prompt_hashes_sha256": _sha256_json(prompt_hashes),
        "structural_payload_hashes_sha256": _sha256_json(structural_hashes),
        "option_structural_payload_hashes_sha256": _sha256_json(
            option_structural_hashes
        ),
    }
    return {**core, "identity_sha256": _sha256_json(core)}


def dataset_identity(items: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = build_items() if items is None else [deepcopy(dict(item)) for item in items]
    return _stable_identity(rows)


def _matching_labels(options: Mapping[str, Any], truth: Any) -> list[str]:
    return [label for label in LABELS if _values_equal(options[label], truth)]


def _source_freshness(
    *,
    source_name: str,
    module: Any,
    expected_script_sha256: str,
    expected_identity_sha256: str,
    expected_items_sha256: str,
    fresh_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    _assert_cpu_dataset_scope()
    source_path = Path(str(module.__file__)).resolve()
    source_script_sha = _sha256_file(source_path)
    source_audit = module.audit_items()
    source_identity = source_audit.get("identity", {})
    errors: list[str] = []
    if source_script_sha != expected_script_sha256:
        errors.append(f"sealed {source_name} dataset script hash changed")
    if source_audit.get("passed") is not True:
        errors.append(f"{source_name} dataset failed its own audit")
    if source_identity.get("identity_sha256") != expected_identity_sha256:
        errors.append(f"{source_name} dataset identity changed")
    if source_identity.get("items_sha256") != expected_items_sha256:
        errors.append(f"{source_name} dataset item hash changed")

    source_rows = module.build_items()
    fresh_prompt_hashes = {_normalized_prompt_hash(row["prompt"]) for row in fresh_rows}
    source_prompt_hashes = {_normalized_prompt_hash(row["prompt"]) for row in source_rows}
    fresh_structural_hashes = {_structural_payload_hash(row) for row in fresh_rows}
    source_structural_hashes = {_structural_payload_hash(row) for row in source_rows}
    prompt_overlap = sorted(fresh_prompt_hashes & source_prompt_hashes)
    structural_overlap = sorted(fresh_structural_hashes & source_structural_hashes)
    if prompt_overlap:
        errors.append(f"normalized {source_name} prompt-content overlap: {prompt_overlap}")
    if structural_overlap:
        errors.append(f"{source_name} structural payload overlap: {structural_overlap}")
    _assert_cpu_dataset_scope()
    return {
        "passed": not errors,
        "source_dataset": source_name,
        "source_n": len(source_rows),
        "source_script_sha256": source_script_sha,
        "source_identity_sha256": source_identity.get("identity_sha256"),
        "source_items_sha256": source_identity.get("items_sha256"),
        "normalized_prompt_contract": (
            "NFC+casefold+whitespace collapse after removing one leading marker "
            "and either legacy Phase979/981 response instruction"
        ),
        "structural_payload_contract": (
            "canonical JSON of task+spec, deliberately independent of option "
            "placement and distractor changes"
        ),
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


def _freshness_against_prior(fresh_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources = {
        "phase979_public128": _source_freshness(
            source_name="phase979_diagnostic128_public_only",
            module=phase979_public,
            expected_script_sha256=EXPECTED_PHASE979_SCRIPT_SHA256,
            expected_identity_sha256=EXPECTED_PHASE979_IDENTITY_SHA256,
            expected_items_sha256=EXPECTED_PHASE979_ITEMS_SHA256,
            fresh_rows=fresh_rows,
        ),
        "phase981_fresh256": _source_freshness(
            source_name="phase981_fresh256_public_only",
            module=phase981_public,
            expected_script_sha256=EXPECTED_PHASE981_SCRIPT_SHA256,
            expected_identity_sha256=EXPECTED_PHASE981_IDENTITY_SHA256,
            expected_items_sha256=EXPECTED_PHASE981_ITEMS_SHA256,
            fresh_rows=fresh_rows,
        ),
    }
    errors = [error for source in sources.values() for error in source["errors"]]
    return {
        "passed": not errors,
        "sources": sources,
        "normalized_prompt_overlap_total_n": sum(
            source["normalized_prompt_overlap_n"] for source in sources.values()
        ),
        "structural_payload_overlap_total_n": sum(
            source["structural_payload_overlap_n"] for source in sources.values()
        ),
        "errors": errors,
    }


def audit_items(items: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Audit schema, mechanical truth, strict twins, balance, and freshness."""
    _assert_cpu_dataset_scope()
    rows = build_items() if items is None else [deepcopy(dict(item)) for item in items]
    errors: list[str] = []
    schema_errors: list[str] = []
    truth_errors: list[str] = []
    prompt_errors: list[str] = []
    twin_errors: list[str] = []
    difficulty_errors: list[str] = []
    encoding_errors: list[str] = []

    required = {
        "schema_version",
        "phase",
        "dataset_marker",
        "id",
        "marker",
        "semantic_id",
        "semantic_ordinal",
        "swap_side",
        "seed_namespace",
        "seed_key",
        "task",
        "difficulty",
        "difficulty_structure",
        "prompt",
        "problem_prompt",
        "answer",
        "alias_groups",
        "exact",
        "options",
        "spec",
        "truth",
        "distractor",
        "contracts",
    }
    ids = [str(row.get("id", "")) for row in rows]
    markers = [str(row.get("marker", "")) for row in rows]
    prompts = [str(row.get("prompt", "")) for row in rows]
    prompt_hashes = [_normalized_prompt_hash(prompt) for prompt in prompts]
    structural_hashes = [_structural_payload_hash(row) for row in rows]
    option_structural_hashes = [_option_structural_payload_hash(row) for row in rows]
    duplicate_ids = sorted(value for value, n in Counter(ids).items() if n > 1)
    marker_counts = Counter(markers)
    invalid_marker_multiplicity = sorted(
        [value, n] for value, n in marker_counts.items() if n != 2
    )
    duplicate_prompts = sorted(value for value, n in Counter(prompt_hashes).items() if n > 1)
    structural_counts = Counter(structural_hashes)
    invalid_structural_multiplicity = sorted(
        [value, n] for value, n in structural_counts.items() if n != 2
    )
    duplicate_option_structures = sorted(
        value for value, n in Counter(option_structural_hashes).items() if n > 1
    )

    task_counts = Counter(str(row.get("task", "")) for row in rows)
    difficulty_counts = Counter(str(row.get("difficulty", "")) for row in rows)
    label_counts = Counter(str(row.get("answer", "")) for row in rows)
    swap_counts = Counter(str(row.get("swap_side", "")) for row in rows)
    stratum_counts: dict[str, dict[str, int]] = {}
    stratum_label_counts: dict[str, dict[str, dict[str, int]]] = {}
    stratum_swap_counts: dict[str, dict[str, dict[str, int]]] = {}
    stratum_swap_label_counts: dict[
        str, dict[str, dict[str, dict[str, int]]]
    ] = {}
    for task in TASKS:
        stratum_counts[task] = {}
        stratum_label_counts[task] = {}
        stratum_swap_counts[task] = {}
        stratum_swap_label_counts[task] = {}
        for difficulty in DIFFICULTIES:
            selected = [
                row
                for row in rows
                if row.get("task") == task and row.get("difficulty") == difficulty
            ]
            stratum_counts[task][difficulty] = len(selected)
            labels = Counter(str(row.get("answer", "")) for row in selected)
            swaps = Counter(str(row.get("swap_side", "")) for row in selected)
            stratum_label_counts[task][difficulty] = {
                label: labels[label] for label in LABELS
            }
            stratum_swap_counts[task][difficulty] = {
                side: swaps[side] for side in SWAP_SIDES
            }
            stratum_swap_label_counts[task][difficulty] = {}
            for side in SWAP_SIDES:
                side_labels = Counter(
                    str(row.get("answer", ""))
                    for row in selected
                    if row.get("swap_side") == side
                )
                stratum_swap_label_counts[task][difficulty][side] = {
                    label: side_labels[label] for label in LABELS
                }

    mechanically_verified = 0
    unambiguous = 0
    semantic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    construction_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    expected_contracts = {
        "mechanical_truth": True,
        "option_swap_twin": True,
        "prompt_has_response_instruction": False,
        "difficulty_is_structural": True,
        "difficulty_is_model_calibrated": False,
        "holdout_source": False,
        "internal_mechanism_evidence": False,
    }
    for row in rows:
        item_id = str(row.get("id", "<missing-id>"))
        missing = sorted(required - set(row))
        if missing:
            schema_errors.append(f"{item_id}: missing fields {missing}")
            continue
        task = str(row["task"])
        difficulty = str(row["difficulty"])
        side = str(row["swap_side"])
        ordinal = row["semantic_ordinal"]
        if row["schema_version"] != SCHEMA_VERSION or row["phase"] != PHASE:
            schema_errors.append(f"{item_id}: schema/phase mismatch")
        if row["dataset_marker"] != DATASET_MARKER:
            schema_errors.append(f"{item_id}: dataset marker mismatch")
        if task not in TASKS or difficulty not in DIFFICULTIES or side not in SWAP_SIDES:
            schema_errors.append(f"{item_id}: unknown task/difficulty/swap side")
            continue
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 1 <= ordinal <= ORDINALS_PER_TASK
        ):
            schema_errors.append(f"{item_id}: invalid semantic ordinal")
            continue
        expected_id = _item_id(task, ordinal, difficulty, side)
        expected_semantic_id = _semantic_id(task, ordinal, difficulty)
        expected_seed_key = _seed_key(task, ordinal, difficulty)
        expected_marker = _marker(task, ordinal, difficulty)
        if item_id != expected_id or _ITEM_ID_RE.fullmatch(item_id) is None:
            schema_errors.append(f"{item_id}: opaque item id mismatch")
        if (
            row["semantic_id"] != expected_semantic_id
            or _SEMANTIC_ID_RE.fullmatch(str(row["semantic_id"])) is None
        ):
            schema_errors.append(f"{item_id}: opaque semantic id mismatch")
        if row["seed_namespace"] != SEED_NAMESPACE:
            schema_errors.append(f"{item_id}: seed namespace mismatch")
        if (
            row["seed_key"] != expected_seed_key
            or _SEED_KEY_RE.fullmatch(str(row["seed_key"])) is None
        ):
            schema_errors.append(f"{item_id}: semantic seed key mismatch")
        if row["marker"] != expected_marker or _MARKER_RE.fullmatch(str(row["marker"])) is None:
            prompt_errors.append(f"{item_id}: Phase983 marker mismatch")
        if str(row["prompt"]).count(expected_marker) != 1:
            prompt_errors.append(f"{item_id}: prompt must contain its marker once")
        if row["problem_prompt"] != row["prompt"]:
            prompt_errors.append(f"{item_id}: prompt/problem_prompt mismatch")
        if any(
            instruction in str(row["prompt"])
            for instruction in (_PHASE979_RESPONSE_INSTRUCTION, _PHASE981_RESPONSE_INSTRUCTION)
        ):
            prompt_errors.append(f"{item_id}: prompt contains a response instruction")

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
            solved_truth = _solve(task, spec)
            if not _values_equal(row["truth"], solved_truth):
                truth_errors.append(f"{item_id}: stored truth disagrees with solver")
            if _values_equal(row["truth"], row["distractor"]):
                truth_errors.append(f"{item_id}: truth equals distractor")
            option_values = [options[label] for label in LABELS]
            if not all(
                any(_values_equal(option, value) for option in option_values)
                for value in (row["truth"], row["distractor"])
            ):
                truth_errors.append(f"{item_id}: options do not encode truth+distractor")
            matches = _matching_labels(options, solved_truth)
            if len(matches) == 1:
                unambiguous += 1
            else:
                truth_errors.append(f"{item_id}: truth matches {len(matches)} labels")
            if matches == [answer]:
                mechanically_verified += 1
            else:
                truth_errors.append(f"{item_id}: answer {answer} disagrees with {matches}")
            expected_prompt = _render_prompt(task, expected_marker, spec, options)
            if row["prompt"] != expected_prompt:
                prompt_errors.append(f"{item_id}: prompt does not encode spec/options exactly")
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
        if row["contracts"] != expected_contracts:
            schema_errors.append(f"{item_id}: contracts mismatch")

        semantic_groups[str(row["semantic_id"])].append(row)
        construction_groups[(task, int(ordinal))].append(row)
        for field in (
            "id",
            "marker",
            "semantic_id",
            "swap_side",
            "seed_namespace",
            "seed_key",
            "task",
            "difficulty",
            "prompt",
            "problem_prompt",
            "answer",
        ):
            value = str(row[field])
            if not value.isascii() or unicodedata.normalize("NFC", value) != value:
                encoding_errors.append(f"{item_id}: {field} must be NFC ASCII")
            if "\ufffd" in value or any(0x80 <= ord(char) <= 0x9F for char in value):
                encoding_errors.append(f"{item_id}: {field} has invalid code point")

    strict_twin_n = 0
    for semantic_id, twins in sorted(semantic_groups.items()):
        if len(twins) != 2:
            twin_errors.append(f"{semantic_id}: expected exactly two option-swap rows")
            continue
        by_side = {str(row["swap_side"]): row for row in twins}
        if set(by_side) != set(SWAP_SIDES):
            twin_errors.append(f"{semantic_id}: missing original/swapped side")
            continue
        original = by_side["original"]
        swapped = by_side["swapped"]
        same_fields = (
            "semantic_id",
            "semantic_ordinal",
            "seed_namespace",
            "seed_key",
            "task",
            "difficulty",
            "difficulty_structure",
            "spec",
            "truth",
            "distractor",
            "contracts",
        )
        changed = [field for field in same_fields if original[field] != swapped[field]]
        if changed:
            twin_errors.append(f"{semantic_id}: invariant twin fields changed {changed}")
            continue
        if not (
            _values_equal(swapped["options"]["A"], original["options"]["B"])
            and _values_equal(swapped["options"]["B"], original["options"]["A"])
        ):
            twin_errors.append(f"{semantic_id}: A/B options were not exactly exchanged")
            continue
        if swapped["answer"] != _opposite_label(str(original["answer"])):
            twin_errors.append(f"{semantic_id}: swapped gold label is not opposite")
            continue
        if original["id"] == swapped["id"]:
            twin_errors.append(f"{semantic_id}: twin item ids must differ")
            continue
        if original["marker"] != swapped["marker"]:
            twin_errors.append(f"{semantic_id}: model-visible marker differs across twin")
            continue
        strict_twin_n += 1

    paired_easy_hard_n = 0
    hard_strictly_more_complex_n = 0
    for key, group in sorted(construction_groups.items()):
        if len(group) != 4:
            difficulty_errors.append(f"{key}: expected four rows across difficulty/swap")
            continue
        representatives: dict[str, dict[str, Any]] = {}
        for difficulty in DIFFICULTIES:
            selected = [
                row
                for row in group
                if row["difficulty"] == difficulty and row["swap_side"] == "original"
            ]
            if len(selected) != 1:
                difficulty_errors.append(f"{key}: missing original {difficulty} representative")
                break
            representatives[difficulty] = selected[0]
        if set(representatives) != set(DIFFICULTIES):
            continue
        paired_easy_hard_n += 1
        easy_score = int(
            representatives["easy"]["difficulty_structure"]["profile"]["complexity_score"]
        )
        hard_score = int(
            representatives["hard"]["difficulty_structure"]["profile"]["complexity_score"]
        )
        if hard_score > easy_score:
            hard_strictly_more_complex_n += 1
        else:
            difficulty_errors.append(f"{key}: hard construction score does not exceed easy")

    if len(rows) != ITEM_COUNT:
        errors.append(f"expected {ITEM_COUNT} items, got {len(rows)}")
    if dict(task_counts) != {task: ITEMS_PER_TASK for task in TASKS}:
        errors.append(f"task counts changed: {dict(task_counts)}")
    if dict(difficulty_counts) != {difficulty: ITEM_COUNT // 2 for difficulty in DIFFICULTIES}:
        errors.append(f"difficulty counts changed: {dict(difficulty_counts)}")
    if dict(label_counts) != {label: ITEM_COUNT // 2 for label in LABELS}:
        errors.append(f"global label counts changed: {dict(label_counts)}")
    if dict(swap_counts) != {side: ITEM_COUNT // 2 for side in SWAP_SIDES}:
        errors.append(f"swap-side counts changed: {dict(swap_counts)}")
    for task in TASKS:
        for difficulty in DIFFICULTIES:
            if stratum_counts[task][difficulty] != 16:
                errors.append(f"{task}/{difficulty}: expected 16 rows")
            if stratum_label_counts[task][difficulty] != {"A": 8, "B": 8}:
                errors.append(
                    f"{task}/{difficulty}: label balance changed "
                    f"{stratum_label_counts[task][difficulty]}"
                )
            if stratum_swap_counts[task][difficulty] != {"original": 8, "swapped": 8}:
                errors.append(
                    f"{task}/{difficulty}: swap balance changed "
                    f"{stratum_swap_counts[task][difficulty]}"
                )
            for side in SWAP_SIDES:
                if stratum_swap_label_counts[task][difficulty][side] != {"A": 4, "B": 4}:
                    errors.append(
                        f"{task}/{difficulty}/{side}: side-specific label balance changed "
                        f"{stratum_swap_label_counts[task][difficulty][side]}"
                    )
    if len(semantic_groups) != SEMANTIC_INSTANCE_COUNT:
        errors.append(
            f"expected {SEMANTIC_INSTANCE_COUNT} semantic ids, got {len(semantic_groups)}"
        )
    if duplicate_ids:
        errors.append(f"duplicate item ids: {duplicate_ids}")
    if invalid_marker_multiplicity:
        errors.append(
            "each semantic marker must occur on exactly two twin rows: "
            f"{invalid_marker_multiplicity}"
        )
    if len(marker_counts) != SEMANTIC_INSTANCE_COUNT:
        errors.append(
            f"expected {SEMANTIC_INSTANCE_COUNT} semantic markers, got {len(marker_counts)}"
        )
    if duplicate_prompts:
        errors.append(f"duplicate normalized prompts: {duplicate_prompts}")
    if invalid_structural_multiplicity:
        errors.append(
            "each semantic task+spec payload must occur on exactly two twin rows: "
            f"{invalid_structural_multiplicity}"
        )
    if len(structural_counts) != SEMANTIC_INSTANCE_COUNT:
        errors.append(
            f"expected {SEMANTIC_INSTANCE_COUNT} semantic task+spec payloads, "
            f"got {len(structural_counts)}"
        )
    if duplicate_option_structures:
        errors.append(
            f"duplicate task+spec+options payloads: {duplicate_option_structures}"
        )

    freshness = _freshness_against_prior(rows)
    if not freshness["passed"]:
        errors.extend(freshness["errors"])
    errors.extend(schema_errors)
    errors.extend(truth_errors)
    errors.extend(prompt_errors)
    errors.extend(twin_errors)
    errors.extend(difficulty_errors)
    errors.extend(sorted(set(encoding_errors)))
    _assert_cpu_dataset_scope()
    passed = not errors
    return {
        "ok": passed,
        "passed": passed,
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "dataset": DATASET_NAME,
        "role": "fresh_option_swap_input_design_only",
        "n_items": len(rows),
        "n_semantic_instances": len(semantic_groups),
        "task_counts": {task: task_counts[task] for task in TASKS},
        "difficulty_counts": {
            difficulty: difficulty_counts[difficulty] for difficulty in DIFFICULTIES
        },
        "swap_side_counts": {side: swap_counts[side] for side in SWAP_SIDES},
        "label_counts": {label: label_counts[label] for label in LABELS},
        "task_difficulty_counts": stratum_counts,
        "task_difficulty_label_counts": stratum_label_counts,
        "task_difficulty_swap_counts": stratum_swap_counts,
        "task_difficulty_swap_label_counts": stratum_swap_label_counts,
        "unique_id_n": len(set(ids)),
        "unique_marker_n": len(set(markers)),
        "unique_seed_key_n": len({str(row.get("seed_key", "")) for row in rows}),
        "unique_normalized_prompt_n": len(set(prompt_hashes)),
        "unique_structural_payload_n": len(set(structural_hashes)),
        "unique_option_structural_payload_n": len(set(option_structural_hashes)),
        "mechanically_verified_n": mechanically_verified,
        "unambiguous_n": unambiguous,
        "strict_option_swap_twin_n": strict_twin_n,
        "paired_easy_hard_construction_n": paired_easy_hard_n,
        "hard_strictly_more_complex_n": hard_strictly_more_complex_n,
        "difficulty_contract": (
            "easy/hard are frozen construction labels and are not calibrated from "
            "any model output"
        ),
        "prompt_contract": (
            "marker+problem+two options only; no response-format or model-control text"
        ),
        "freshness_against_prior_public_data": freshness,
        "holdout_accessed": False,
        "holdout_modules_loaded": _forbidden_holdout_modules(),
        "model_runtime_modules_loaded": _forbidden_runtime_modules(),
        "identity": _stable_identity(rows),
        "schema_errors": schema_errors,
        "truth_errors": truth_errors,
        "prompt_errors": prompt_errors,
        "twin_errors": twin_errors,
        "difficulty_errors": difficulty_errors,
        "encoding_errors": sorted(set(encoding_errors)),
        "errors": errors,
    }


def _dataset_payload(items: list[dict[str, Any]], audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "dataset": DATASET_NAME,
        "role": "fresh_option_swap_input_design_only",
        "dataset_marker": DATASET_MARKER,
        "seed_namespace": SEED_NAMESPACE,
        "contracts": {
            "item_count": ITEM_COUNT,
            "semantic_instance_count": SEMANTIC_INSTANCE_COUNT,
            "task_count": len(TASKS),
            "ordinals_per_task": ORDINALS_PER_TASK,
            "difficulties": list(DIFFICULTIES),
            "swap_sides": list(SWAP_SIDES),
            "mechanically_solved": True,
            "strict_option_swap_twins": True,
            "original_and_swapped_labels_each_balanced_within_task_difficulty": True,
            "prompt_contains_response_instruction": False,
            "difficulty_is_structural_not_model_calibrated": True,
            "phase979_and_phase981_prompt_and_structural_overlap_zero": True,
            "phase977_holdout_accessed": False,
            "model_weights_loaded": False,
            "generation_performed": False,
        },
        "identity": deepcopy(dict(audit["identity"])),
        "items": deepcopy(items),
    }


def _dataset_document(items: list[dict[str, Any]], audit: Mapping[str, Any]) -> dict[str, Any]:
    payload = _dataset_payload(items, audit)
    return {**payload, "dataset_sha256": _sha256_json(payload)}


def _verify_dataset_document(document: Mapping[str, Any]) -> bool:
    try:
        supplied = deepcopy(dict(document))
        supplied_hash = supplied.pop("dataset_sha256")
        if supplied_hash != _sha256_json(supplied):
            return False
        items = build_items()
        audit = audit_items(items)
        if audit["passed"] is not True:
            return False
        expected = _dataset_document(items, audit)
        return document == expected
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def _audit_document(
    *,
    audit: Mapping[str, Any],
    dataset_document: Mapping[str, Any],
    dataset_bytes: bytes,
    tests: Mapping[str, Any],
    script_sha256: str,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    payload = {
        **deepcopy(dict(audit)),
        "script": script_path.relative_to(ROOT).as_posix(),
        "script_sha256": script_sha256,
        "dataset_path": DATASET_PATH.resolve().relative_to(ROOT).as_posix(),
        "dataset_document_sha256": dataset_document["dataset_sha256"],
        "dataset_file_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "sealed_source_scripts": {
            "phase979_diagnostic_dataset.py": EXPECTED_PHASE979_SCRIPT_SHA256,
            "phase981_fresh_dataset.py": EXPECTED_PHASE981_SCRIPT_SHA256,
        },
        "self_test": deepcopy(dict(tests)),
        "cpu_only": True,
        "gpu_used": False,
        "model_runtime_modules_imported": False,
        "model_weights_loaded": False,
        "generation_performed": False,
        "protocol_frozen": False,
        "model_contract_defined": False,
    }
    return {**payload, "audit_sha256": _sha256_json(payload)}


def _verify_audit_document_against(
    document: Mapping[str, Any],
    *,
    audit: Mapping[str, Any],
    dataset_document: Mapping[str, Any],
    dataset_bytes: bytes,
    tests: Mapping[str, Any],
    script_sha256: str,
) -> bool:
    try:
        supplied = deepcopy(dict(document))
        supplied_hash = supplied.pop("audit_sha256")
        if supplied_hash != _sha256_json(supplied):
            return False
        expected = _audit_document(
            audit=audit,
            dataset_document=dataset_document,
            dataset_bytes=dataset_bytes,
            tests=tests,
            script_sha256=script_sha256,
        )
        return document == expected
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def _verify_audit_document(document: Mapping[str, Any]) -> bool:
    """Rebuild the full deterministic audit; a recomputed self-hash is insufficient."""
    try:
        items = build_items()
        audit = audit_items(items)
        if audit["passed"] is not True:
            return False
        dataset_document = _dataset_document(items, audit)
        dataset_bytes = _json_bytes(dataset_document)
        tests = self_test()
        if tests["passed"] is not True:
            return False
        script_sha = _sha256_file(Path(__file__).resolve())
        return _verify_audit_document_against(
            document,
            audit=audit,
            dataset_document=dataset_document,
            dataset_bytes=dataset_bytes,
            tests=tests,
            script_sha256=script_sha,
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def self_test() -> dict[str, Any]:
    _assert_cpu_dataset_scope()
    first = build_items()
    second = build_items()
    deterministic = first == second
    fresh_objects = first is not second and first[0] is not second[0]
    audit = audit_items(first)

    tampered_answer = deepcopy(first)
    tampered_answer[0]["answer"] = _opposite_label(tampered_answer[0]["answer"])
    answer_rejected = not audit_items(tampered_answer)["passed"]

    tampered_truth = deepcopy(first)
    tampered_truth[1]["truth"] = tampered_truth[1]["distractor"]
    truth_rejected = not audit_items(tampered_truth)["passed"]

    tampered_twin = deepcopy(first)
    tampered_twin[1]["options"] = deepcopy(tampered_twin[0]["options"])
    twin_exchange_rejected = not audit_items(tampered_twin)["passed"]

    tampered_spec = deepcopy(first)
    tampered_spec[2]["spec"]["initial"] += 1
    spec_rejected = not audit_items(tampered_spec)["passed"]

    tampered_prompt = deepcopy(first)
    tampered_prompt[3]["prompt"] = tampered_prompt[2]["prompt"]
    tampered_prompt[3]["problem_prompt"] = tampered_prompt[2]["problem_prompt"]
    duplicate_prompt_rejected = not audit_items(tampered_prompt)["passed"]

    tampered_semantic = deepcopy(first)
    tampered_semantic[1]["semantic_id"] = tampered_semantic[2]["semantic_id"]
    semantic_lineage_rejected = not audit_items(tampered_semantic)["passed"]

    tampered_seed = deepcopy(first)
    tampered_seed[0]["seed_namespace"] = "P983-FOREIGN-SEED"
    seed_namespace_rejected = not audit_items(tampered_seed)["passed"]

    tampered_marker = deepcopy(first)
    tampered_marker[1]["marker"] = tampered_marker[2]["marker"]
    marker_twin_rejected = not audit_items(tampered_marker)["passed"]

    phase979_probe = _source_freshness(
        source_name="phase979_diagnostic128_public_only",
        module=phase979_public,
        expected_script_sha256=EXPECTED_PHASE979_SCRIPT_SHA256,
        expected_identity_sha256=EXPECTED_PHASE979_IDENTITY_SHA256,
        expected_items_sha256=EXPECTED_PHASE979_ITEMS_SHA256,
        fresh_rows=[phase979_public.build_items()[0]],
    )
    phase981_probe = _source_freshness(
        source_name="phase981_fresh256_public_only",
        module=phase981_public,
        expected_script_sha256=EXPECTED_PHASE981_SCRIPT_SHA256,
        expected_identity_sha256=EXPECTED_PHASE981_IDENTITY_SHA256,
        expected_items_sha256=EXPECTED_PHASE981_ITEMS_SHA256,
        fresh_rows=[phase981_public.build_items()[0]],
    )
    prior_overlap_detection = (
        phase979_probe["normalized_prompt_overlap_n"] == 1
        and phase979_probe["structural_payload_overlap_n"] == 1
        and phase981_probe["normalized_prompt_overlap_n"] == 1
        and phase981_probe["structural_payload_overlap_n"] == 1
        and not phase979_probe["passed"]
        and not phase981_probe["passed"]
    )
    changed_option_probe_row = deepcopy(phase981_public.build_items()[0])
    changed_option_probe_row["options"] = {
        "A": "phase983_probe_option_left",
        "B": "phase983_probe_option_right",
    }
    changed_option_probe_row["prompt"] = first[0]["prompt"]
    semantic_overlap_probe = _source_freshness(
        source_name="phase981_fresh256_public_only",
        module=phase981_public,
        expected_script_sha256=EXPECTED_PHASE981_SCRIPT_SHA256,
        expected_identity_sha256=EXPECTED_PHASE981_IDENTITY_SHA256,
        expected_items_sha256=EXPECTED_PHASE981_ITEMS_SHA256,
        fresh_rows=[changed_option_probe_row],
    )
    semantic_overlap_despite_option_change_rejected = (
        semantic_overlap_probe["normalized_prompt_overlap_n"] == 0
        and semantic_overlap_probe["structural_payload_overlap_n"] == 1
        and not semantic_overlap_probe["passed"]
    )

    dataset_document = _dataset_document(first, audit)
    deterministic_document_valid = _verify_dataset_document(dataset_document)
    rehashed_tamper = deepcopy(dataset_document)
    rehashed_tamper["items"][0]["answer"] = _opposite_label(
        rehashed_tamper["items"][0]["answer"]
    )
    tampered_payload = deepcopy(rehashed_tamper)
    tampered_payload.pop("dataset_sha256")
    rehashed_tamper["dataset_sha256"] = _sha256_json(tampered_payload)
    rehashed_document_rejected = not _verify_dataset_document(rehashed_tamper)

    synthetic_tests = {
        "passed": True,
        "checks": {"synthetic_audit_verifier_fixture": True},
    }
    dataset_bytes = _json_bytes(dataset_document)
    script_sha = _sha256_file(Path(__file__).resolve())
    synthetic_audit_document = _audit_document(
        audit=audit,
        dataset_document=dataset_document,
        dataset_bytes=dataset_bytes,
        tests=synthetic_tests,
        script_sha256=script_sha,
    )
    deterministic_audit_document_valid = _verify_audit_document_against(
        synthetic_audit_document,
        audit=audit,
        dataset_document=dataset_document,
        dataset_bytes=dataset_bytes,
        tests=synthetic_tests,
        script_sha256=script_sha,
    )
    rehashed_audit_tamper = deepcopy(synthetic_audit_document)
    rehashed_audit_tamper["passed"] = False
    tampered_audit_payload = deepcopy(rehashed_audit_tamper)
    tampered_audit_payload.pop("audit_sha256")
    rehashed_audit_tamper["audit_sha256"] = _sha256_json(tampered_audit_payload)
    rehashed_audit_document_rejected = not _verify_audit_document_against(
        rehashed_audit_tamper,
        audit=audit,
        dataset_document=dataset_document,
        dataset_bytes=dataset_bytes,
        tests=synthetic_tests,
        script_sha256=script_sha,
    )

    duplicate_json_rejected = False
    try:
        _load_json_strict_bytes(b'{"x":1,"x":2}')
    except ValueError:
        duplicate_json_rejected = True
    nonfinite_json_rejected = False
    try:
        _load_json_strict_bytes(b'{"x":NaN}')
    except ValueError:
        nonfinite_json_rejected = True

    dummy_holdout = "phase977_holdout_self_test_probe"
    sys.modules[dummy_holdout] = object()  # type: ignore[assignment]
    try:
        holdout_tripwire_rejected = False
        try:
            _assert_cpu_dataset_scope()
        except RuntimeError:
            holdout_tripwire_rejected = True
    finally:
        sys.modules.pop(dummy_holdout, None)

    sys.modules["torch"] = object()  # type: ignore[assignment]
    try:
        runtime_tripwire_rejected = False
        try:
            _assert_cpu_dataset_scope()
        except RuntimeError:
            runtime_tripwire_rejected = True
    finally:
        sys.modules.pop("torch", None)

    checks = {
        "deterministic_build": deterministic,
        "fresh_mutable_objects": fresh_objects,
        "formal_audit_passed": audit["passed"],
        "answer_tamper_rejected": answer_rejected,
        "truth_tamper_rejected": truth_rejected,
        "twin_exchange_tamper_rejected": twin_exchange_rejected,
        "spec_tamper_rejected": spec_rejected,
        "duplicate_prompt_rejected": duplicate_prompt_rejected,
        "semantic_lineage_tamper_rejected": semantic_lineage_rejected,
        "seed_namespace_tamper_rejected": seed_namespace_rejected,
        "model_visible_marker_tamper_rejected": marker_twin_rejected,
        "prior_overlap_detection_operational": prior_overlap_detection,
        "semantic_overlap_despite_option_change_rejected": (
            semantic_overlap_despite_option_change_rejected
        ),
        "phase979_normalized_prompt_overlap_zero": (
            audit["freshness_against_prior_public_data"]["sources"]["phase979_public128"][
                "normalized_prompt_overlap_n"
            ]
            == 0
        ),
        "phase979_structural_payload_overlap_zero": (
            audit["freshness_against_prior_public_data"]["sources"]["phase979_public128"][
                "structural_payload_overlap_n"
            ]
            == 0
        ),
        "phase981_normalized_prompt_overlap_zero": (
            audit["freshness_against_prior_public_data"]["sources"]["phase981_fresh256"][
                "normalized_prompt_overlap_n"
            ]
            == 0
        ),
        "phase981_structural_payload_overlap_zero": (
            audit["freshness_against_prior_public_data"]["sources"]["phase981_fresh256"][
                "structural_payload_overlap_n"
            ]
            == 0
        ),
        "deterministic_dataset_document_verified": deterministic_document_valid,
        "rehash_tamper_rejected": rehashed_document_rejected,
        "deterministic_audit_document_verified": deterministic_audit_document_valid,
        "audit_rehash_tamper_rejected": rehashed_audit_document_rejected,
        "duplicate_json_key_rejected": duplicate_json_rejected,
        "nonfinite_json_rejected": nonfinite_json_rejected,
        "holdout_tripwire_operational": holdout_tripwire_rejected,
        "model_runtime_tripwire_operational": runtime_tripwire_rejected,
        "holdout_modules_absent_after_test": not _forbidden_holdout_modules(),
        "model_runtime_modules_absent": not _forbidden_runtime_modules(),
    }
    _assert_cpu_dataset_scope()
    return {"passed": all(checks.values()), "checks": checks, "identity": audit["identity"]}


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
    _assert_cpu_dataset_scope()
    tests = self_test()
    if not tests["passed"]:
        raise RuntimeError(f"Phase983 dataset self-test failed: {tests}")
    items = build_items()
    audit = audit_items(items)
    if not audit["passed"]:
        raise RuntimeError(f"Phase983 dataset audit failed: {audit['errors']}")
    dataset_document = _dataset_document(items, audit)
    if not _verify_dataset_document(dataset_document):
        raise RuntimeError("deterministic dataset document verification failed")
    dataset_bytes = _json_bytes(dataset_document)
    parsed_dataset = _load_json_strict_bytes(dataset_bytes)
    if parsed_dataset != dataset_document:
        raise RuntimeError("dataset strict JSON round trip failed")

    script_path = Path(__file__).resolve()
    script_sha = _sha256_file(script_path)
    audit_document = _audit_document(
        audit=audit,
        dataset_document=dataset_document,
        dataset_bytes=dataset_bytes,
        tests=tests,
        script_sha256=script_sha,
    )
    if not _verify_audit_document(audit_document):
        raise RuntimeError("deterministic audit document verification failed")
    rehashed_audit_attack = deepcopy(audit_document)
    rehashed_audit_attack["passed"] = False
    attack_payload = deepcopy(rehashed_audit_attack)
    attack_payload.pop("audit_sha256")
    rehashed_audit_attack["audit_sha256"] = _sha256_json(attack_payload)
    if _verify_audit_document(rehashed_audit_attack):
        raise RuntimeError("audit rehash attack was accepted")
    audit_bytes = _json_bytes(audit_document)
    parsed_audit = _load_json_strict_bytes(audit_bytes)
    if parsed_audit != audit_document:
        raise RuntimeError("audit strict JSON round trip failed")

    _install_exact(DATASET_PATH, dataset_bytes)
    _install_exact(AUDIT_PATH, audit_bytes)
    if DATASET_PATH.read_bytes() != dataset_bytes or AUDIT_PATH.read_bytes() != audit_bytes:
        raise RuntimeError("installed artifact bytes differ from deterministic payload")
    _assert_cpu_dataset_scope()
    return {
        "passed": True,
        "script_sha256": script_sha,
        "dataset_identity_sha256": audit["identity"]["identity_sha256"],
        "items_sha256": audit["identity"]["items_sha256"],
        "dataset_document_sha256": dataset_document["dataset_sha256"],
        "dataset_file_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "audit_sha256": audit_document["audit_sha256"],
        "audit_file_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "dataset_path": str(DATASET_PATH),
        "audit_path": str(AUDIT_PATH),
        "holdout_accessed": False,
        "model_runtime_modules_imported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run fail-closed CPU tests")
    parser.add_argument("--write", action="store_true", help="atomically install data artifacts")
    args = parser.parse_args()
    _assert_cpu_dataset_scope()
    if args.write:
        result = write_artifacts()
    elif args.self_test:
        result = self_test()
    else:
        audit = audit_items()
        result = {
            "passed": audit["passed"],
            "n_items": audit["n_items"],
            "n_semantic_instances": audit["n_semantic_instances"],
            "identity": audit["identity"],
            "freshness_against_prior_public_data": audit[
                "freshness_against_prior_public_data"
            ],
            "holdout_accessed": False,
            "model_runtime_modules_imported": False,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
