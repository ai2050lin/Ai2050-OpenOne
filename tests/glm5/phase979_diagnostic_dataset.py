#!/usr/bin/env python3
"""Phase 979 diagnostic corpus: 128 new, mechanically decidable items.

The corpus is generated entirely from constants and arithmetic in this module.
It does not read or import any Phase 977 data or holdout module.  Every item has
two displayed choices and a canonical answer that is exactly the one-character
ASCII label ``A`` or ``B``.  The prompt permits a separate optional final ASCII
period so later punctuation diagnostics can keep the answer label itself fixed.

``build_items()`` returns fresh dictionaries.  ``audit_items()`` independently
re-solves every item, reconstructs every prompt, checks balance and uniqueness,
and returns a stable canonical-JSON identity.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


SCHEMA_VERSION = 1
DATASET_NAME = "phase979_diagnostic128"
ITEMS_PER_TASK = 16
LABELS = ("A", "B")
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
EXPECTED_COUNTS = {task: ITEMS_PER_TASK for task in TASKS}
EXPECTED_LABEL_COUNTS = {"A": 64, "B": 64}
EXPECTED_TASK_LABEL_COUNTS = {"A": 8, "B": 8}

_RESPONSE_INSTRUCTION = (
    "Return only A or B as the final answer. One optional final ASCII period "
    "is allowed; output nothing else."
)
_MARKER_RE = re.compile(
    r"^\[P979-DIAG\|([a-z_]+)\|(\d{2})\]$"
)


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


def _marker(task: str, index: int) -> str:
    return f"[P979-DIAG|{task}|{index:02d}]"


def _display(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return value
    return str(value)


def _assign_options(truth: Any, distractor: Any, answer: str) -> dict[str, Any]:
    if answer not in LABELS:
        raise ValueError(f"unknown answer label: {answer}")
    if truth == distractor:
        raise ValueError("truth and distractor must differ")
    if answer == "A":
        return {"A": truth, "B": distractor}
    return {"A": distractor, "B": truth}


def _solve_multistep_arithmetic(spec: Mapping[str, Any]) -> int:
    return (
        (int(spec["start"]) + int(spec["add"]))
        * int(spec["multiply"])
        - int(spec["subtract"])
    )


def _solve_modular_arithmetic(spec: Mapping[str, Any]) -> int:
    modulus = int(spec["modulus"])
    if modulus <= 1:
        raise ValueError("modulus must exceed one")
    return (
        int(spec["left"]) * int(spec["right"]) + int(spec["add"])
    ) % modulus


def _solve_boolean_logic(spec: Mapping[str, Any]) -> bool:
    p = bool(spec["P"])
    q = bool(spec["Q"])
    r = bool(spec["R"])
    rule = str(spec["rule"])
    if rule == "and_not_or":
        return (p and (not q)) or r
    if rule == "or_and_not_or":
        return (p or q) and ((not p) or (not r))
    if rule == "not_or_and":
        return (not (p or q)) or (q and r)
    if rule == "and_or_not_and":
        return (p and q) or ((not q) and r)
    raise ValueError(f"unknown boolean rule: {rule}")


def _path_exists(spec: Mapping[str, Any]) -> bool:
    nodes = [str(node) for node in spec["nodes"]]
    start = str(spec["start"])
    target = str(spec["target"])
    if len(nodes) != len(set(nodes)) or start not in nodes or target not in nodes:
        raise ValueError("invalid relation-path node set")
    adjacency = {node: [] for node in nodes}
    for raw_edge in spec["edges"]:
        if len(raw_edge) != 2:
            raise ValueError("each directed edge must have two endpoints")
        source, destination = map(str, raw_edge)
        if source not in adjacency or destination not in adjacency:
            raise ValueError("edge endpoint is absent from node set")
        adjacency[source].append(destination)
    frontier = deque([start])
    visited = {start}
    while frontier:
        node = frontier.popleft()
        if node == target:
            return True
        for destination in adjacency[node]:
            if destination not in visited:
                visited.add(destination)
                frontier.append(destination)
    return False


def _solve_state_machine(spec: Mapping[str, Any]) -> str:
    states = [str(state) for state in spec["states"]]
    alphabet = [str(symbol) for symbol in spec["alphabet"]]
    transitions = spec["transitions"]
    state = str(spec["start"])
    if len(states) != len(set(states)) or state not in states:
        raise ValueError("invalid state-machine states")
    if len(alphabet) != len(set(alphabet)) or not alphabet:
        raise ValueError("invalid state-machine alphabet")
    for source in states:
        if source not in transitions:
            raise ValueError(f"missing transition row for {source}")
        if set(transitions[source]) != set(alphabet):
            raise ValueError(f"incomplete transition row for {source}")
        if any(str(target) not in states for target in transitions[source].values()):
            raise ValueError(f"unknown transition target from {source}")
    for symbol in str(spec["input"]):
        if symbol not in alphabet:
            raise ValueError(f"unknown input symbol: {symbol}")
        state = str(transitions[state][symbol])
    return state


def _solve_sequence_rule(spec: Mapping[str, Any]) -> int:
    kind = str(spec["kind"])
    if kind == "affine":
        value = int(spec["a0"])
        for _ in range(int(spec["steps"])):
            value = int(spec["multiply"]) * value + int(spec["add"])
        return value
    if kind == "second_order":
        values = [int(spec["a0"]), int(spec["a1"])]
        target_index = int(spec["target_index"])
        if target_index < 1:
            raise ValueError("second-order target index must be at least one")
        while len(values) <= target_index:
            values.append(values[-1] + values[-2] + int(spec["bias"]))
        return values[target_index]
    raise ValueError(f"unknown sequence rule: {kind}")


def _solve_string_transform(spec: Mapping[str, Any]) -> str:
    source = str(spec["source"])
    kind = str(spec["kind"])
    if not source or not source.isascii() or not source.isalpha():
        raise ValueError("source must be a nonempty ASCII alphabetic string")
    if kind == "reverse_append":
        suffix = str(spec["suffix"])
        if len(suffix) != 1 or not suffix.isascii() or not suffix.isalpha():
            raise ValueError("suffix must be one ASCII letter")
        return source[::-1] + suffix
    if kind == "rotate_left_upper":
        shift = int(spec["shift"])
        normalized = shift % len(source)
        return (source[normalized:] + source[:normalized]).upper()
    raise ValueError(f"unknown string transform: {kind}")


def _satisfies_constraints(order: Sequence[str], constraints: Sequence[Sequence[str]]) -> bool:
    normalized = [str(value) for value in order]
    if len(normalized) != len(set(normalized)):
        return False
    positions = {value: index for index, value in enumerate(normalized)}
    for raw_constraint in constraints:
        if len(raw_constraint) != 2:
            raise ValueError("each order constraint must contain two objects")
        before, after = map(str, raw_constraint)
        if before not in positions or after not in positions:
            return False
        if positions[before] >= positions[after]:
            return False
    return True


def _solve_constraint_order(spec: Mapping[str, Any]) -> tuple[str, ...]:
    objects = tuple(str(value) for value in spec["objects"])
    constraints = spec["constraints"]
    if len(objects) != len(set(objects)) or len(objects) < 2:
        raise ValueError("invalid order object set")
    valid = [
        tuple(str(value) for value in candidate)
        for candidate in spec["candidates"]
        if _satisfies_constraints(candidate, constraints)
    ]
    if len(valid) != 1:
        raise ValueError("exactly one candidate ordering must satisfy every constraint")
    return valid[0]


def _solve(task: str, spec: Mapping[str, Any]) -> Any:
    if task == "multistep_arithmetic":
        return _solve_multistep_arithmetic(spec)
    if task == "modular_arithmetic":
        return _solve_modular_arithmetic(spec)
    if task == "boolean_logic":
        return _solve_boolean_logic(spec)
    if task == "relation_path":
        return _path_exists(spec)
    if task == "state_machine":
        return _solve_state_machine(spec)
    if task == "sequence_rule":
        return _solve_sequence_rule(spec)
    if task == "string_transform":
        return _solve_string_transform(spec)
    if task == "constraint_order":
        return _solve_constraint_order(spec)
    raise ValueError(f"unknown task: {task}")


def _problem_text(task: str, spec: Mapping[str, Any]) -> str:
    if task == "multistep_arithmetic":
        return (
            f"Start with {spec['start']}. Add {spec['add']}, multiply the result "
            f"by {spec['multiply']}, then subtract {spec['subtract']}. Which "
            "labeled option is the exact final integer?"
        )
    if task == "modular_arithmetic":
        return (
            f"Compute (({spec['left']} * {spec['right']}) + {spec['add']}) "
            f"modulo {spec['modulus']}, using the least nonnegative remainder. "
            "Which labeled option is correct?"
        )
    if task == "boolean_logic":
        values = (
            f"P={'TRUE' if spec['P'] else 'FALSE'}, "
            f"Q={'TRUE' if spec['Q'] else 'FALSE'}, "
            f"R={'TRUE' if spec['R'] else 'FALSE'}"
        )
        expressions = {
            "and_not_or": "(P AND (NOT Q)) OR R",
            "or_and_not_or": "(P OR Q) AND ((NOT P) OR (NOT R))",
            "not_or_and": "NOT (P OR Q) OR (Q AND R)",
            "and_or_not_and": "(P AND Q) OR ((NOT Q) AND R)",
        }
        expression = expressions[str(spec["rule"])]
        return (
            f"Use standard Boolean rules: NOT flips a value, AND requires both "
            f"values, and OR requires at least one value. Given {values}, "
            f"evaluate {expression}. Which labeled option is its truth value?"
        )
    if task == "relation_path":
        edges = ", ".join(
            f"{source}->{destination}" for source, destination in spec["edges"]
        )
        return (
            f"Only these directed edges exist: {edges}. Following arrow "
            f"directions, is there a directed path from {spec['start']} to "
            f"{spec['target']}?"
        )
    if task == "state_machine":
        transitions = []
        for state in spec["states"]:
            for symbol in spec["alphabet"]:
                transitions.append(
                    f"{state} on {symbol}->{spec['transitions'][state][symbol]}"
                )
        table = "; ".join(transitions)
        return (
            f"A deterministic machine has transitions: {table}. It starts in "
            f"{spec['start']} and reads {spec['input']} from left to right. "
            "Which labeled option is the final state?"
        )
    if task == "sequence_rule":
        if spec["kind"] == "affine":
            return (
                f"A sequence has a0={spec['a0']} and follows "
                f"a(k+1)={spec['multiply']}*a(k)+{spec['add']}. Apply the rule "
                f"exactly {spec['steps']} times. Which labeled option is "
                f"a{spec['steps']}?"
            )
        return (
            f"A sequence has a0={spec['a0']} and a1={spec['a1']}. For n>=2, "
            f"a(n)=a(n-1)+a(n-2)+{spec['bias']}. Which labeled option is "
            f"a{spec['target_index']}?"
        )
    if task == "string_transform":
        if spec["kind"] == "reverse_append":
            return (
                f"Start with the case-sensitive ASCII string {json.dumps(spec['source'])}. "
                f"Reverse all characters, then append the lowercase letter "
                f"{json.dumps(spec['suffix'])}. Which labeled option is the exact "
                "result?"
            )
        return (
            f"Start with the case-sensitive ASCII string {json.dumps(spec['source'])}. "
            f"Rotate it left by {spec['shift']} characters, then convert every "
            "letter to uppercase. Which labeled option is the exact result?"
        )
    if task == "constraint_order":
        constraints = ", ".join(
            f"{before} before {after}" for before, after in spec["constraints"]
        )
        return (
            f"An ordering satisfies X before Y only when X is strictly to the "
            f"left of Y. The complete constraints are: {constraints}. Which "
            "labeled candidate satisfies every constraint?"
        )
    raise ValueError(f"unknown task: {task}")


def _render_option(task: str, value: Any) -> str:
    if task == "relation_path":
        return "YES, a directed path exists" if value else "NO, no directed path exists"
    if task == "constraint_order":
        return " < ".join(str(part) for part in value)
    return _display(value)


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
        f"B: {_render_option(task, options['B'])}\n"
        f"{_RESPONSE_INSTRUCTION}"
    )


def _make_item(
    task: str,
    index: int,
    spec: Mapping[str, Any],
    options: Mapping[str, Any],
    answer: str,
) -> dict[str, Any]:
    marker = _marker(task, index)
    item = {
        "schema_version": SCHEMA_VERSION,
        "id": f"p979_diag_{task}_{index:02d}",
        "marker": marker,
        "task": task,
        "prompt": _render_prompt(task, marker, spec, options),
        "answer": answer,
        "alias_groups": [[answer]],
        "exact": True,
        "options": deepcopy(dict(options)),
        "spec": deepcopy(dict(spec)),
    }
    return item


def _build_multistep_arithmetic() -> list[dict[str, Any]]:
    rows = []
    for offset in range(ITEMS_PER_TASK):
        index = offset + 1
        answer = "A" if offset % 2 == 0 else "B"
        spec = {
            "start": 11 + 3 * offset,
            "add": 4 + (offset % 5),
            "multiply": 2 + (offset % 3),
            "subtract": 3 + ((2 * offset) % 7),
        }
        truth = _solve_multistep_arithmetic(spec)
        distractor = truth + (1 + (offset % 3))
        rows.append(_make_item(
            "multistep_arithmetic",
            index,
            spec,
            _assign_options(truth, distractor, answer),
            answer,
        ))
    return rows


def _build_modular_arithmetic() -> list[dict[str, Any]]:
    rows = []
    for offset in range(ITEMS_PER_TASK):
        index = offset + 1
        answer = "B" if offset % 2 == 0 else "A"
        spec = {
            "left": 7 + 2 * offset,
            "right": 5 + (offset % 7),
            "add": 1 + 3 * offset,
            "modulus": 7 + (offset % 5),
        }
        truth = _solve_modular_arithmetic(spec)
        distractor = (truth + 1 + (offset % (spec["modulus"] - 1))) % spec["modulus"]
        rows.append(_make_item(
            "modular_arithmetic",
            index,
            spec,
            _assign_options(truth, distractor, answer),
            answer,
        ))
    return rows


def _build_boolean_logic() -> list[dict[str, Any]]:
    rows = []
    rules = (
        "and_not_or",
        "or_and_not_or",
        "not_or_and",
        "and_or_not_and",
    )
    for offset in range(ITEMS_PER_TASK):
        index = offset + 1
        answer = "A" if offset % 2 == 0 else "B"
        bits = (offset * 5 + 1) % 8
        spec = {
            "P": bool(bits & 4),
            "Q": bool(bits & 2),
            "R": bool(bits & 1),
            "rule": rules[offset // 4],
        }
        truth = _solve_boolean_logic(spec)
        rows.append(_make_item(
            "boolean_logic",
            index,
            spec,
            _assign_options(truth, not truth, answer),
            answer,
        ))
    return rows


def _build_relation_path() -> list[dict[str, Any]]:
    rows = []
    for offset in range(ITEMS_PER_TASK):
        index = offset + 1
        answer = "A" if offset % 2 == 0 else "B"
        nodes = [f"u{index:02d}", f"v{index:02d}", f"w{index:02d}", f"x{index:02d}"]
        if answer == "A":
            edges = [
                [nodes[0], nodes[1]],
                [nodes[1], nodes[2]],
                [nodes[2], nodes[3]],
                [nodes[3], nodes[1]],
            ]
        else:
            edges = [
                [nodes[0], nodes[1]],
                [nodes[2], nodes[1]],
                [nodes[2], nodes[3]],
                [nodes[3], nodes[2]],
            ]
        spec = {
            "nodes": nodes,
            "edges": edges,
            "start": nodes[0],
            "target": nodes[3],
        }
        truth = _path_exists(spec)
        options = {"A": True, "B": False}
        if ("A" if truth else "B") != answer:
            raise AssertionError("relation-path construction lost label balance")
        rows.append(_make_item("relation_path", index, spec, options, answer))
    return rows


def _build_state_machine() -> list[dict[str, Any]]:
    rows = []
    states = ["S0", "S1", "S2"]
    alphabet = ["0", "1"]
    input_patterns = (
        "00101", "11010", "01011", "10100",
        "01101", "10011", "11100", "00011",
    )
    for offset in range(ITEMS_PER_TASK):
        index = offset + 1
        answer = "B" if offset % 2 == 0 else "A"
        step_zero = 1 + (offset % 2)
        step_one = 1 + ((offset // 2) % 2)
        transitions = {}
        for state_index, state in enumerate(states):
            transitions[state] = {
                "0": states[(state_index + step_zero) % len(states)],
                "1": states[(2 * state_index + step_one) % len(states)],
            }
        spec = {
            "states": list(states),
            "alphabet": list(alphabet),
            "transitions": transitions,
            "start": states[offset % len(states)],
            "input": input_patterns[offset % len(input_patterns)],
        }
        truth = _solve_state_machine(spec)
        distractor = states[(states.index(truth) + 1 + (offset % 2)) % len(states)]
        rows.append(_make_item(
            "state_machine",
            index,
            spec,
            _assign_options(truth, distractor, answer),
            answer,
        ))
    return rows


def _build_sequence_rule() -> list[dict[str, Any]]:
    rows = []
    for offset in range(ITEMS_PER_TASK):
        index = offset + 1
        answer = "A" if offset % 2 == 0 else "B"
        if offset < 8:
            spec = {
                "kind": "affine",
                "a0": 2 + offset,
                "multiply": 2 + (offset % 2),
                "add": 1 + (offset % 4),
                "steps": 3 + (offset % 2),
            }
        else:
            spec = {
                "kind": "second_order",
                "a0": 1 + (offset % 4),
                "a1": 3 + (offset % 5),
                "bias": 1 + (offset % 3),
                "target_index": 5 + (offset % 2),
            }
        truth = _solve_sequence_rule(spec)
        distractor = truth + 1 + (offset % 4)
        rows.append(_make_item(
            "sequence_rule",
            index,
            spec,
            _assign_options(truth, distractor, answer),
            answer,
        ))
    return rows


def _build_string_transform() -> list[dict[str, Any]]:
    rows = []
    sources = (
        "amber", "cedar", "frost", "glyph", "mango", "river", "stone", "velvet",
        "planet", "silver", "cobalt", "timber", "quartz", "rocket", "willow", "zenith",
    )
    suffixes = ("x", "q", "m", "v", "k", "p", "d", "z")
    for offset, source in enumerate(sources):
        index = offset + 1
        answer = "B" if offset % 2 == 0 else "A"
        if offset < 8:
            spec = {
                "kind": "reverse_append",
                "source": source,
                "suffix": suffixes[offset],
            }
            truth = _solve_string_transform(spec)
            alternate_suffix = "a" if spec["suffix"] != "a" else "b"
            distractor = source[::-1] + alternate_suffix
        else:
            spec = {
                "kind": "rotate_left_upper",
                "source": source,
                "shift": 1 + (offset % (len(source) - 1)),
            }
            truth = _solve_string_transform(spec)
            shift = spec["shift"] % len(source)
            distractor = (source[-shift:] + source[:-shift]).upper()
            if distractor == truth:
                distractor = source.upper()
        rows.append(_make_item(
            "string_transform",
            index,
            spec,
            _assign_options(truth, distractor, answer),
            answer,
        ))
    return rows


def _build_constraint_order() -> list[dict[str, Any]]:
    rows = []
    for offset in range(ITEMS_PER_TASK):
        index = offset + 1
        answer = "A" if offset % 2 == 0 else "B"
        objects = [f"r{index:02d}", f"s{index:02d}", f"t{index:02d}", f"u{index:02d}"]
        rotation = offset % len(objects)
        valid = objects[rotation:] + objects[:rotation]
        constraints = [
            [valid[0], valid[1]],
            [valid[1], valid[2]],
            [valid[0], valid[3]],
        ]
        invalid = list(valid)
        invalid[0], invalid[1] = invalid[1], invalid[0]
        candidates = [valid, invalid] if answer == "A" else [invalid, valid]
        spec = {
            "objects": objects,
            "constraints": constraints,
            "candidates": candidates,
        }
        truth = _solve_constraint_order(spec)
        options = {
            "A": tuple(candidates[0]),
            "B": tuple(candidates[1]),
        }
        if options[answer] != truth:
            raise AssertionError("constraint-order construction lost label balance")
        rows.append(_make_item("constraint_order", index, spec, options, answer))
    return rows


def build_items() -> list[dict[str, Any]]:
    """Return the complete 128-item corpus as fresh mutable dictionaries."""
    builders = (
        _build_multistep_arithmetic,
        _build_modular_arithmetic,
        _build_boolean_logic,
        _build_relation_path,
        _build_state_machine,
        _build_sequence_rule,
        _build_string_transform,
        _build_constraint_order,
    )
    rows: list[dict[str, Any]] = []
    for builder in builders:
        rows.extend(builder())
    return deepcopy(rows)


def _normalized_prompt(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _matching_labels(options: Mapping[str, Any], truth: Any) -> list[str]:
    normalized_truth = tuple(truth) if isinstance(truth, tuple) else truth
    matches = []
    for label in LABELS:
        value = options[label]
        normalized_value = tuple(value) if isinstance(value, list) else value
        if normalized_value == normalized_truth:
            matches.append(label)
    return matches


def _stable_identity(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    stable_items = sorted(
        (deepcopy(dict(item)) for item in items),
        key=lambda item: str(item.get("id", "")),
    )
    task_counts = dict(sorted(Counter(
        str(item.get("task", "")) for item in stable_items
    ).items()))
    label_counts = dict(sorted(Counter(
        str(item.get("answer", "")) for item in stable_items
    ).items()))
    core = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "n_items": len(stable_items),
        "task_counts": task_counts,
        "label_counts": label_counts,
        "items_sha256": _sha256_json(stable_items),
    }
    return {
        **core,
        "identity_sha256": _sha256_json(core),
    }


def audit_items(items: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Audit corpus identity, balance, prompts, and exact mechanical truth.

    The audit intentionally depends only on the supplied rows and the independent
    solvers above.  It performs no filesystem reads and has no access to any
    Phase 977 holdout.
    """
    rows = build_items() if items is None else [deepcopy(dict(item)) for item in items]
    errors: list[str] = []
    schema_issues: list[str] = []
    encoding_issues: list[str] = []
    truth_issues: list[str] = []
    prompt_issues: list[str] = []

    required = {
        "schema_version",
        "id",
        "marker",
        "task",
        "prompt",
        "answer",
        "alias_groups",
        "exact",
        "options",
        "spec",
    }
    ids = [str(row.get("id", "")) for row in rows]
    markers = [str(row.get("marker", "")) for row in rows]
    prompts = [str(row.get("prompt", "")) for row in rows]
    prompt_keys = [_normalized_prompt(prompt) for prompt in prompts]
    problem_signatures = [
        _sha256_json({
            "task": row.get("task"),
            "spec": row.get("spec"),
            "options": row.get("options"),
        })
        for row in rows
    ]
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    duplicate_markers = sorted(value for value, count in Counter(markers).items() if count > 1)
    duplicate_prompts = sorted(value for value, count in Counter(prompt_keys).items() if count > 1)
    duplicate_problem_signatures = sorted(
        value for value, count in Counter(problem_signatures).items() if count > 1
    )

    task_counts = dict(sorted(Counter(str(row.get("task", "")) for row in rows).items()))
    label_counts = dict(sorted(Counter(str(row.get("answer", "")) for row in rows).items()))
    task_label_counts: dict[str, dict[str, int]] = {}
    for task in TASKS:
        counts = Counter(
            str(row.get("answer", ""))
            for row in rows
            if row.get("task") == task
        )
        task_label_counts[task] = {label: counts[label] for label in LABELS}

    mechanically_verified = 0
    unambiguous = 0
    for row in rows:
        item_id = str(row.get("id", "<missing-id>"))
        missing = sorted(required - set(row))
        if missing:
            schema_issues.append(f"{item_id}: missing fields {missing}")
            continue
        if row["schema_version"] != SCHEMA_VERSION:
            schema_issues.append(f"{item_id}: wrong schema_version")
        task = str(row["task"])
        if task not in TASKS:
            schema_issues.append(f"{item_id}: unknown task {task}")
            continue
        expected_prefix = f"p979_diag_{task}_"
        if not item_id.startswith(expected_prefix):
            schema_issues.append(f"{item_id}: id does not match task")
        try:
            index = int(item_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            schema_issues.append(f"{item_id}: invalid numeric id suffix")
            continue
        if not 1 <= index <= ITEMS_PER_TASK:
            schema_issues.append(f"{item_id}: id index outside 01..16")
        expected_marker = _marker(task, index)
        marker = str(row["marker"])
        marker_match = _MARKER_RE.fullmatch(marker)
        if marker != expected_marker or marker_match is None:
            prompt_issues.append(f"{item_id}: invalid or mismatched P979 marker")
        prompt = str(row["prompt"])
        if prompt.count(marker) != 1 or prompt.count("[P979-DIAG|") != 1:
            prompt_issues.append(f"{item_id}: prompt must contain one unique marker")
        if _RESPONSE_INSTRUCTION not in prompt:
            prompt_issues.append(f"{item_id}: response instruction is missing")
        answer = str(row["answer"])
        if answer not in LABELS or len(answer) != 1 or not answer.isascii():
            schema_issues.append(f"{item_id}: answer must be one ASCII label A or B")
        if row["alias_groups"] != [[answer]]:
            schema_issues.append(f"{item_id}: alias_groups must contain only the answer label")
        if row["exact"] is not True:
            schema_issues.append(f"{item_id}: exact must be true")
        options = row["options"]
        if not isinstance(options, Mapping) or set(options) != set(LABELS):
            schema_issues.append(f"{item_id}: options must have exactly A and B")
            continue
        if options["A"] == options["B"]:
            truth_issues.append(f"{item_id}: choices are identical")
        if not isinstance(row["spec"], Mapping):
            schema_issues.append(f"{item_id}: spec must be a mapping")
            continue
        try:
            truth = _solve(task, row["spec"])
            matches = _matching_labels(options, truth)
            if len(matches) == 1:
                unambiguous += 1
            else:
                truth_issues.append(
                    f"{item_id}: mechanical truth matches {len(matches)} choices"
                )
            if matches == [answer]:
                mechanically_verified += 1
            else:
                truth_issues.append(
                    f"{item_id}: answer {answer} disagrees with mechanical truth {matches}"
                )
            expected_prompt = _render_prompt(task, marker, row["spec"], options)
            if prompt != expected_prompt:
                prompt_issues.append(f"{item_id}: prompt does not exactly encode spec/options")
        except (KeyError, TypeError, ValueError) as exc:
            truth_issues.append(f"{item_id}: solver error: {exc}")

        for field_name in ("id", "marker", "task", "prompt", "answer"):
            value = str(row[field_name])
            if unicodedata.normalize("NFC", value) != value:
                encoding_issues.append(f"{item_id}: {field_name} is not NFC")
            if not value.isascii():
                encoding_issues.append(f"{item_id}: {field_name} is not ASCII")
            if "\ufffd" in value:
                encoding_issues.append(f"{item_id}: {field_name} contains U+FFFD")
            if any(0x80 <= ord(char) <= 0x9F for char in value):
                encoding_issues.append(f"{item_id}: {field_name} contains C1 control")

    if len(rows) != len(TASKS) * ITEMS_PER_TASK:
        errors.append(f"expected 128 items, found {len(rows)}")
    if task_counts != EXPECTED_COUNTS:
        errors.append(f"task counts differ: {task_counts}")
    if label_counts != EXPECTED_LABEL_COUNTS:
        errors.append(f"global label counts differ: {label_counts}")
    for task, counts in task_label_counts.items():
        if counts != EXPECTED_TASK_LABEL_COUNTS:
            errors.append(f"{task} label counts differ: {counts}")
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids}")
    if duplicate_markers:
        errors.append(f"duplicate markers: {duplicate_markers}")
    if duplicate_prompts:
        errors.append(f"duplicate prompts: {duplicate_prompts}")
    if duplicate_problem_signatures:
        errors.append(
            f"duplicate problem signatures: {duplicate_problem_signatures}"
        )
    errors.extend(schema_issues)
    errors.extend(prompt_issues)
    errors.extend(truth_issues)
    errors.extend(sorted(set(encoding_issues)))

    identity = _stable_identity(rows)
    passed = not errors
    return {
        "ok": passed,
        "passed": passed,
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "n_items": len(rows),
        "task_counts": task_counts,
        "label_counts": label_counts,
        "task_label_counts": task_label_counts,
        "single_character_label_n": sum(
            str(row.get("answer", "")) in LABELS for row in rows
        ),
        "unique_id_n": len(set(ids)),
        "unique_marker_n": len(set(markers)),
        "unique_prompt_n": len(set(prompt_keys)),
        "unique_problem_n": len(set(problem_signatures)),
        "mechanically_verified_n": mechanically_verified,
        "unambiguous_n": unambiguous,
        "duplicate_ids": duplicate_ids,
        "duplicate_markers": duplicate_markers,
        "duplicate_prompts": duplicate_prompts,
        "duplicate_problem_signatures": duplicate_problem_signatures,
        "schema_issues": schema_issues,
        "prompt_issues": prompt_issues,
        "truth_issues": truth_issues,
        "encoding_issues": sorted(set(encoding_issues)),
        "holdout_accessed": False,
        "identity": identity,
        "errors": errors,
    }


if __name__ == "__main__":
    audit = audit_items()
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2))
    raise SystemExit(0 if audit["ok"] else 1)
