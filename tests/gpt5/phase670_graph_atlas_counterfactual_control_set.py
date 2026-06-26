#!/usr/bin/env python3
"""
Phase 670: Graph Atlas Counterfactual Control Set.

This phase does not run model inference. It builds a reusable control matrix
for the Phase 669 graph atlas. The controls are synthetic and in-context so
future model tests measure language-mechanism separation rather than world
knowledge memorization.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


OUT_ROOT = Path("results/glm5_phase670_graph_atlas_counterfactual_control_set")
ATLAS_PATH = Path(
    "results/glm5_phase669_cross_mechanism_language_encoding_graph_atlas/phase669_graph_atlas.json"
)


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    axis: str
    object_name: str
    relation: str
    value: str
    format_name: str
    prompt: str
    expected_output: str
    target_nodes: tuple[str, ...]
    invariant_nodes: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class Pair:
    pair_id: str
    family: str
    left_case_id: str
    right_case_id: str
    isolated_factor: str
    expected_sensitive_nodes: tuple[str, ...]
    expected_invariant_nodes: tuple[str, ...]
    failure_signal: str


OBJECTS = [
    "daxor",
    "melin",
    "vorak",
    "sarin",
    "tovel",
    "noric",
    "zelpa",
    "brix",
    "calen",
    "faron",
    "lumen",
    "prax",
    "sovel",
    "kiren",
    "marn",
    "dorin",
    "velta",
    "riven",
]

RELATIONS = {
    "color": [
        "blue",
        "green",
        "yellow",
        "silver",
        "purple",
        "orange",
        "white",
        "black",
        "red",
    ],
    "tool": [
        "hammer",
        "ladder",
        "needle",
        "brush",
        "spoon",
        "wrench",
        "bucket",
        "pencil",
        "knife",
    ],
    "place": [
        "garden",
        "market",
        "harbor",
        "library",
        "school",
        "station",
        "kitchen",
        "office",
        "temple",
    ],
    "symbol": [
        "circle",
        "square",
        "triangle",
        "spiral",
        "anchor",
        "arrow",
        "star",
        "cross",
        "wave",
    ],
}

FORMATS = {
    "short": {
        "instruction": "Answer with only the value.",
        "render": lambda obj, rel, val: val,
        "nodes": ("semantic_value_support", "task_intent_gate", "protocol_execution_field"),
    },
    "sentence": {
        "instruction": "Answer in one complete sentence.",
        "render": lambda obj, rel, val: f"The {rel} of {obj} is {val}.",
        "nodes": ("protocol_execution_field", "format_continuation_state"),
    },
    "json": {
        "instruction": "Return JSON with keys object, relation, value.",
        "render": lambda obj, rel, val: json.dumps(
            {"object": obj, "relation": rel, "value": val}, ensure_ascii=False, separators=(",", ":")
        ),
        "nodes": ("protocol_execution_field", "format_continuation_state"),
    },
    "explanation": {
        "instruction": "Answer with a short explanation.",
        "render": lambda obj, rel, val: f"The record states that {obj} has {rel} {val}.",
        "nodes": ("task_intent_gate", "protocol_execution_field", "format_continuation_state"),
    },
    "label": {
        "instruction": "Write the answer after 'Value:'.",
        "render": lambda obj, rel, val: f"Value: {val}",
        "nodes": ("protocol_execution_field", "first_token_readout_closure"),
    },
    "list": {
        "instruction": "Return a one-item markdown list.",
        "render": lambda obj, rel, val: f"- {val}",
        "nodes": ("protocol_execution_field", "multi_competitor_readout"),
    },
}

PREFIX_GROUPS = [
    ("new", ["new york", "new jersey", "new zealand", "new hampshire"]),
    ("san", ["san diego", "san jose", "san juan", "san marino"]),
    ("north", ["north pole", "north gate", "north harbor", "north valley"]),
    ("red", ["redwood", "redstone", "redline", "redhill"]),
    ("south", ["south pole", "south gate", "south ridge", "south river"]),
    ("west", ["westport", "westfield", "westlake", "westhaven"]),
]


def relation_value(object_index: int, relation: str) -> str:
    values = RELATIONS[relation]
    return values[object_index % len(values)]


def distractor_values(object_index: int, relation: str, n: int = 3) -> list[str]:
    values = RELATIONS[relation]
    return [values[(object_index + i + 1) % len(values)] for i in range(n)]


def build_record(object_name: str, object_index: int, override: dict[str, str] | None = None) -> str:
    override = override or {}
    clauses = []
    for rel in RELATIONS:
        val = override.get(rel, relation_value(object_index, rel))
        clauses.append(f"{object_name} {rel} is {val}")
    return "Record: " + "; ".join(clauses) + "."


def make_prompt(record: str, obj: str, rel: str, instruction: str) -> str:
    return f"{record}\nQuestion: What is the {rel} of {obj}?\nInstruction: {instruction}\nAnswer:"


def build_cases() -> list[Case]:
    cases: list[Case] = []

    for oi, obj in enumerate(OBJECTS):
        record = build_record(obj, oi)
        for rel in RELATIONS:
            value = relation_value(oi, rel)
            for fmt, spec in FORMATS.items():
                cid = f"cf_{len(cases):05d}"
                cases.append(
                    Case(
                        case_id=cid,
                        family="same_value_different_format",
                        axis="format",
                        object_name=obj,
                        relation=rel,
                        value=value,
                        format_name=fmt,
                        prompt=make_prompt(record, obj, rel, spec["instruction"]),
                        expected_output=spec["render"](obj, rel, value),
                        target_nodes=tuple(dict.fromkeys(("semantic_value_support",) + spec["nodes"])),
                        invariant_nodes=("semantic_value_support", "value_specific_token1_transition_state"),
                        note="Same object/relation/value with changed output protocol.",
                    )
                )

    for oi, obj in enumerate(OBJECTS[:12]):
        for rel in RELATIONS:
            original = relation_value(oi, rel)
            changed = distractor_values(oi, rel, 1)[0]
            record = build_record(obj, oi, {rel: changed})
            cid = f"cf_{len(cases):05d}"
            cases.append(
                Case(
                    case_id=cid,
                    family="different_value_same_format",
                    axis="value",
                    object_name=obj,
                    relation=rel,
                    value=changed,
                    format_name="short",
                    prompt=make_prompt(record, obj, rel, FORMATS["short"]["instruction"]),
                    expected_output=changed,
                    target_nodes=("semantic_value_support", "value_specific_token1_transition_state"),
                    invariant_nodes=("task_intent_gate", "protocol_execution_field"),
                    note=f"Same short-answer protocol, changed value from {original} to {changed}.",
                )
            )

    for gi, (prefix, values) in enumerate(PREFIX_GROUPS):
        for vi, value in enumerate(values):
            obj = f"{prefix.replace(' ', '_')}_obj_{vi}"
            record = f"Record: {obj} route is {value}; {obj} color is blue; {obj} tool is hammer."
            cid = f"cf_{len(cases):05d}"
            cases.append(
                Case(
                    case_id=cid,
                    family="same_prefix_different_continuation",
                    axis="continuation",
                    object_name=obj,
                    relation="route",
                    value=value,
                    format_name="short",
                    prompt=make_prompt(record, obj, "route", FORMATS["short"]["instruction"]),
                    expected_output=value,
                    target_nodes=(
                        "first_token_readout_closure",
                        "format_continuation_state",
                        "value_specific_token1_transition_state",
                        "continuation_controller",
                    ),
                    invariant_nodes=("task_intent_gate", "protocol_execution_field"),
                    note=f"Values share visible prefix '{prefix}' but require different continuation.",
                )
            )

    for oi, obj in enumerate(OBJECTS):
        for rel_i, rel in enumerate(RELATIONS):
            random_value = f"nonce_{oi:02d}_{rel_i:02d}"
            record = build_record(obj, oi, {rel: random_value})
            cid = f"cf_{len(cases):05d}"
            cases.append(
                Case(
                    case_id=cid,
                    family="same_format_random_value",
                    axis="random_value",
                    object_name=obj,
                    relation=rel,
                    value=random_value,
                    format_name="short",
                    prompt=make_prompt(record, obj, rel, FORMATS["short"]["instruction"]),
                    expected_output=random_value,
                    target_nodes=("semantic_value_support", "value_specific_token1_transition_state"),
                    invariant_nodes=("protocol_execution_field", "task_intent_gate"),
                    note="Same protocol with nonce value; useful for memorized-vocabulary controls.",
                )
            )

    for oi, obj in enumerate(OBJECTS):
        rel = "color"
        value = relation_value(oi, rel)
        record = build_record(obj, oi)
        for intent_name, instruction, expected, nodes in [
            (
                "value_only",
                "Answer with only the value.",
                value,
                ("semantic_value_support", "task_intent_gate"),
            ),
            (
                "intent_only_existence",
                "Do not give the value. Answer yes or no: does the record contain this relation?",
                "yes",
                ("task_intent_gate", "protocol_execution_field"),
            ),
            (
                "protocol_only_json",
                "Ignore the value and return JSON with keys object and relation only.",
                json.dumps({"object": obj, "relation": rel}, separators=(",", ":")),
                ("protocol_execution_field",),
            ),
        ]:
            cid = f"cf_{len(cases):05d}"
            cases.append(
                Case(
                    case_id=cid,
                    family="factor_isolation",
                    axis=intent_name,
                    object_name=obj,
                    relation=rel,
                    value=value,
                    format_name=intent_name,
                    prompt=make_prompt(record, obj, rel, instruction),
                    expected_output=expected,
                    target_nodes=nodes,
                    invariant_nodes=("semantic_value_support",),
                    note="Separates value content, task intent, and protocol pressure.",
                )
            )

    return cases


def group_cases(cases: Iterable[Case]) -> dict[tuple[str, str, str, str], list[Case]]:
    grouped: dict[tuple[str, str, str, str], list[Case]] = {}
    for case in cases:
        key = (case.object_name, case.relation, case.value, case.family)
        grouped.setdefault(key, []).append(case)
    return grouped


def build_pairs(cases: list[Case]) -> list[Pair]:
    pairs: list[Pair] = []
    by_id = {c.case_id: c for c in cases}

    # Same value, different format.
    for key, group in group_cases(cases).items():
        if key[3] != "same_value_different_format":
            continue
        short = next((c for c in group if c.format_name == "short"), None)
        if not short:
            continue
        for other in group:
            if other.case_id == short.case_id:
                continue
            pairs.append(
                Pair(
                    pair_id=f"pair_{len(pairs):05d}",
                    family="same_value_different_format",
                    left_case_id=short.case_id,
                    right_case_id=other.case_id,
                    isolated_factor="format_protocol",
                    expected_sensitive_nodes=("protocol_execution_field", "format_continuation_state"),
                    expected_invariant_nodes=("semantic_value_support", "value_specific_token1_transition_state"),
                    failure_signal="Value changes when only output protocol changes.",
                )
            )

    # Different value, same format: pair each changed-value case with its base short case.
    base_lookup = {
        (c.object_name, c.relation, c.format_name): c
        for c in cases
        if c.family == "same_value_different_format" and c.format_name == "short"
    }
    for c in cases:
        if c.family != "different_value_same_format":
            continue
        base = base_lookup.get((c.object_name, c.relation, "short"))
        if not base:
            continue
        pairs.append(
            Pair(
                pair_id=f"pair_{len(pairs):05d}",
                family="different_value_same_format",
                left_case_id=base.case_id,
                right_case_id=c.case_id,
                isolated_factor="semantic_value",
                expected_sensitive_nodes=("semantic_value_support", "value_specific_token1_transition_state"),
                expected_invariant_nodes=("task_intent_gate", "protocol_execution_field"),
                failure_signal="Protocol changes when only value changes, or old value sticks.",
            )
        )

    # Same prefix, different continuation.
    prefix_cases = [c for c in cases if c.family == "same_prefix_different_continuation"]
    for i in range(0, len(prefix_cases), 4):
        group = prefix_cases[i : i + 4]
        for a, b in zip(group, group[1:]):
            pairs.append(
                Pair(
                    pair_id=f"pair_{len(pairs):05d}",
                    family="same_prefix_different_continuation",
                    left_case_id=a.case_id,
                    right_case_id=b.case_id,
                    isolated_factor="continuation_after_shared_prefix",
                    expected_sensitive_nodes=(
                        "format_continuation_state",
                        "value_specific_token1_transition_state",
                        "continuation_controller",
                    ),
                    expected_invariant_nodes=("task_intent_gate", "protocol_execution_field"),
                    failure_signal="Correct first visible prefix but wrong continuation token sequence.",
                )
            )

    # Factor isolation triples as pairs against value_only.
    factor_groups: dict[str, list[Case]] = {}
    for c in cases:
        if c.family == "factor_isolation":
            factor_groups.setdefault(c.object_name, []).append(c)
    for obj, group in factor_groups.items():
        value_case = next(c for c in group if c.axis == "value_only")
        for other in group:
            if other.case_id == value_case.case_id:
                continue
            pairs.append(
                Pair(
                    pair_id=f"pair_{len(pairs):05d}",
                    family="factor_isolation",
                    left_case_id=value_case.case_id,
                    right_case_id=other.case_id,
                    isolated_factor=other.axis,
                    expected_sensitive_nodes=other.target_nodes,
                    expected_invariant_nodes=("semantic_value_support",),
                    failure_signal="Model leaks value when task/protocol says not to output it.",
                )
            )

    # Ensure pair references are valid.
    for pair in pairs:
        if pair.left_case_id not in by_id or pair.right_case_id not in by_id:
            raise ValueError(f"Bad pair reference: {pair}")
    return pairs


def summarize(cases: list[Case], pairs: list[Pair], atlas: dict) -> dict:
    by_family: dict[str, int] = {}
    by_axis: dict[str, int] = {}
    by_format: dict[str, int] = {}
    node_hits: dict[str, int] = {}
    for c in cases:
        by_family[c.family] = by_family.get(c.family, 0) + 1
        by_axis[c.axis] = by_axis.get(c.axis, 0) + 1
        by_format[c.format_name] = by_format.get(c.format_name, 0) + 1
        for node_id in c.target_nodes:
            node_hits[node_id] = node_hits.get(node_id, 0) + 1
    pair_family: dict[str, int] = {}
    for p in pairs:
        pair_family[p.family] = pair_family.get(p.family, 0) + 1
    source_nodes = [n["id"] for n in atlas.get("nodes", [])]
    uncovered = sorted(set(source_nodes) - set(node_hits))
    return {
        "n_cases": len(cases),
        "n_pairs": len(pairs),
        "case_family_counts": dict(sorted(by_family.items())),
        "case_axis_counts": dict(sorted(by_axis.items())),
        "format_counts": dict(sorted(by_format.items())),
        "target_node_counts": dict(sorted(node_hits.items())),
        "pair_family_counts": dict(sorted(pair_family.items())),
        "source_atlas_nodes": source_nodes,
        "uncovered_by_input_output_controls": uncovered,
        "uncovered_reason": {
            "writer_topology": "Requires internal activation/component tests, not only prompt-level counterfactuals.",
            "residual_boundary_integrated_state": "Requires boundary restore/remove or trajectory capture.",
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(payload: dict) -> str:
    s = payload["summary"]
    lines = [
        "# Phase 670 Graph Atlas Counterfactual Control Set",
        "",
        f"- generated: `{payload['timestamp']}`",
        f"- source_atlas: `{payload['source_atlas']}`",
        f"- n_cases: `{s['n_cases']}`",
        f"- n_pairs: `{s['n_pairs']}`",
        "",
        "## Principle",
        "",
        "This phase does not run model inference. It creates clean counterfactual controls for the Phase 669 graph atlas.",
        "",
        "Each pair changes one intended factor while holding the others as stable as possible:",
        "",
        "- same value / different format",
        "- different value / same format",
        "- same prefix / different continuation",
        "- same format / random value",
        "- value-only / intent-only / protocol-only factor isolation",
        "",
        "## Case Families",
        "",
        "| family | count |",
        "|---|---:|",
    ]
    for k, v in s["case_family_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Pair Families", "", "| family | count |", "|---|---:|"]
    for k, v in s["pair_family_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Target Nodes", "", "| node | case_count |", "|---|---:|"]
    for k, v in s["target_node_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Nodes Not Covered By Prompt-Level Controls",
        "",
    ]
    for node_id in s["uncovered_by_input_output_controls"]:
        reason = s["uncovered_reason"].get(node_id, "Requires a later internal intervention or trajectory audit.")
        lines.append(f"- `{node_id}`: {reason}")
    lines += [
        "",
        "## Future Model Test Command Shape",
        "",
        "```bash",
        "python tests/gpt5/phase671_graph_atlas_counterfactual_model_audit.py --model qwen3 --hard-exit-after-model",
        "python tests/gpt5/phase671_graph_atlas_counterfactual_model_audit.py --model glm4 --hard-exit-after-model",
        "python tests/gpt5/phase671_graph_atlas_counterfactual_model_audit.py --model deepseek7b --hard-exit-after-model",
        "```",
        "",
        "## Stop Condition",
        "",
        "The next model phase should not start until the control set passes tokenizer validation for all three models.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    atlas = json.loads(ATLAS_PATH.read_text(encoding="utf-8")) if ATLAS_PATH.exists() else {}
    cases = build_cases()
    pairs = build_pairs(cases)
    payload = {
        "phase": 670,
        "title": "Graph Atlas Counterfactual Control Set",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_atlas": str(ATLAS_PATH),
        "principle": "Build clean graph-node counterfactual controls before new model inference.",
        "summary": summarize(cases, pairs, atlas),
        "cases": [asdict(c) for c in cases],
        "pairs": [asdict(p) for p in pairs],
        "hard_limits": [
            "Synthetic records reduce world-knowledge confounds but still need tokenizer validation.",
            "Same-prefix controls use visible text prefixes; token boundaries can differ by model.",
            "This phase creates controls only; it does not prove graph-node causality.",
        ],
        "next_phase": {
            "phase": 671,
            "title": "Graph Atlas Counterfactual Tokenizer and Natural Trajectory Audit",
            "goal": "Validate tokenization and then run natural, no-patch predictions across qwen3, GLM4, and DS7B.",
        },
    }
    (OUT_ROOT / "phase670_counterfactual_control_set.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(OUT_ROOT / "phase670_cases.jsonl", payload["cases"])
    write_jsonl(OUT_ROOT / "phase670_pairs.jsonl", payload["pairs"])
    (OUT_ROOT / "phase670_counterfactual_control_set.md").write_text(
        write_markdown(payload), encoding="utf-8"
    )
    print(f"Wrote {OUT_ROOT / 'phase670_counterfactual_control_set.json'}")
    print(f"Wrote {OUT_ROOT / 'phase670_cases.jsonl'}")
    print(f"Wrote {OUT_ROOT / 'phase670_pairs.jsonl'}")
    print(f"Wrote {OUT_ROOT / 'phase670_counterfactual_control_set.md'}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
