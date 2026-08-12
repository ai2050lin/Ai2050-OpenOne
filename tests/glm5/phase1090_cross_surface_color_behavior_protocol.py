#!/usr/bin/env python3
"""Freeze Phase1090 cross-surface color-equivalence behavior protocol.

This phase is behavior-only.  Facts and queries can express the same canonical
color with English, Chinese, French, or hexadecimal color strings.  Mixed
surface routes require the model to connect two different lexical surfaces;
same-surface routes are matched behavioral baselines.  No hidden-state scan is
authorized by this protocol itself.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1089_truth_matched_color_binding_protocol as base


PHASE = 1090
PROTOCOL_REVISION = 1
MODELS = base.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
COLORS = base.COLORS
COLOR_PAIRS = base.COLOR_PAIRS
OPERATIONS = base.OPERATIONS
BASE_WORLDS = base.WORLDS
SPLITS = base.SPLITS
PANELS = base.PANELS
ITEMS_PER_CELL_SPLIT = 4
GENERATION_STEPS = 6
GENERATION_UNITS_PER_CELL_SPLIT = 1
ASSISTANT_PREFILL = base.ASSISTANT_PREFILL
OUTPUT_PAIR = ("Yes", "No")
STATES = tuple(
    f"c{panel}_m{target}_q{binding}"
    for panel in PANELS
    for target in (0, 1)
    for binding in (0, 1)
)

SURFACE_VALUES = {
    "en": {
        "red": "red", "blue": "blue", "green": "green",
        "yellow": "yellow", "black": "black", "white": "white",
        "orange": "orange", "purple": "purple",
    },
    "zh": {
        "red": "红色", "blue": "蓝色", "green": "绿色",
        "yellow": "黄色", "black": "黑色", "white": "白色",
        "orange": "橙色", "purple": "紫色",
    },
    "fr": {
        "red": "rouge", "blue": "bleu", "green": "vert",
        "yellow": "jaune", "black": "noir", "white": "blanc",
        "orange": "orange", "purple": "violet",
    },
    "hex": {
        "red": "#FF0000", "blue": "#0000FF", "green": "#00FF00",
        "yellow": "#FFFF00", "black": "#000000", "white": "#FFFFFF",
        "orange": "#FFA500", "purple": "#800080",
    },
}
SURFACE_ROUTES = (
    "en_en", "zh_zh", "fr_fr", "hex_hex",
    "en_zh", "zh_en", "en_fr", "fr_en", "en_hex", "hex_en",
)
MIXED_SURFACE_ROUTES = tuple(
    route for route in SURFACE_ROUTES
    if route.split("_", 1)[0] != route.split("_", 1)[1]
)
WORLDS = tuple(
    f"{world}@{route}"
    for world in BASE_WORLDS
    for route in SURFACE_ROUTES
)
CELLS = tuple(
    f"{operation}__{world}"
    for operation in OPERATIONS
    for world in WORLDS
)
FAMILIES = CELLS
CAPTURE_ROLES = base.CAPTURE_ROLES
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1090_cross_surface_color_behavior"
)
SOURCE_PHASE1089 = base.OUT_ROOT
ENTITY_POOLS = base.ENTITY_POOLS
FACT_ORDERS = base.FACT_ORDERS

EVIDENCE_THRESHOLDS = {
    "minimum_candidate_accuracy": 0.80,
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_generation_accuracy": 0.75,
    "minimum_worlds_per_operation": 3,
    "minimum_operations_per_route": 6,
    "minimum_models_per_route": 2,
    "minimum_viable_mixed_routes": 2,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "All static audits pass: canonical truth, one-true/one-false panels, "
        "balanced anchors, independent entity splits, and complete routes."
    ),
    "P2": "All three models are loaded sequentially in FP16 without quantization.",
    "P3": (
        "At least two same-surface baselines are viable in two models across "
        "six color pairs and three entity worlds."
    ),
    "P4": (
        "At least two genuinely mixed lexical routes are viable in two models "
        "across six color pairs and three entity worlds."
    ),
    "P5": (
        "Natural generation reaches 0.75 target-before-distractor accuracy for "
        "every route admitted to hidden-state follow-up."
    ),
}

write_json = base.write_json
write_jsonl = base.write_jsonl
read_json = base.read_json
read_jsonl = base.read_jsonl
digest = base.digest
tokenizer_for = base.tokenizer_for
offset_token_spans = base.offset_token_spans
behavior = base.behavior
mark_source = base.mark_source
encoded_width = base.encoded_width


def split_world(value: str) -> tuple[str, str]:
    return tuple(value.split("@", 1))  # type: ignore[return-value]


def split_route(route: str) -> tuple[str, str]:
    return tuple(route.split("_", 1))  # type: ignore[return-value]


def split_cell(cell: str) -> tuple[str, str]:
    return tuple(cell.split("__", 1))  # type: ignore[return-value]


def operation_colors(operation: str) -> tuple[str, str]:
    return COLOR_PAIRS[OPERATIONS.index(operation)]


def state_factors(state: str) -> tuple[str, int, int]:
    for panel in PANELS:
        prefix = f"c{panel}_m"
        if state.startswith(prefix):
            target, binding = state[len(prefix):].split("_q", 1)
            return panel, int(target), int(binding)
    raise ValueError(f"invalid state: {state}")


def split_items(cell: str, split: str) -> tuple[dict[str, Any], ...]:
    operation, world_route = split_cell(cell)
    world, route = split_world(world_route)
    color0, color1 = operation_colors(operation)
    pool = ENTITY_POOLS[world][split]
    rows = []
    for local_index in range(ITEMS_PER_CELL_SPLIT):
        anchor_variant = local_index % 2
        rows.append({
            "item_id": f"{cell}.{split}.{local_index:02d}",
            "entity0": pool[local_index],
            "entity1": pool[(local_index + 3) % len(pool)],
            "anchor": pool[(local_index + 6) % len(pool)],
            "anchor_variant": anchor_variant,
            "anchor_color": (color0, color1)[anchor_variant],
            "fact_order_index": local_index,
            "operation": operation,
            "world": world,
            "surface_route": route,
        })
    return tuple(rows)


def build_case(
    tokenizer,
    model_name: str,
    cell: str,
    split: str,
    item: dict[str, Any],
    state: str,
    case_index: int,
) -> dict[str, Any]:
    operation, world_route = split_cell(cell)
    world, route = split_world(world_route)
    fact_surface, query_surface = split_route(route)
    panel, target_variant, binding = state_factors(state)
    color0, color1 = operation_colors(operation)
    canonical_colors = (color0, color1)
    bound_colors = canonical_colors if binding == 0 else tuple(reversed(canonical_colors))
    selected_entity = (
        str(item["entity0"]) if panel == "active" else str(item["anchor"])
    )
    if panel == "active":
        semantic_answer = int(binding != target_variant)
    else:
        semantic_answer = int(int(item["anchor_variant"]) != target_variant)
    true_word, false_word = OUTPUT_PAIR
    target_answer = (true_word, false_word)[semantic_answer]

    def fact(entity: str, canonical: str) -> str:
        value = SURFACE_VALUES[fact_surface][canonical]
        return f"{entity} has visible color {value}."

    facts = {
        "entity0": fact(str(item["entity0"]), bound_colors[0]),
        "entity1": fact(str(item["entity1"]), bound_colors[1]),
        "anchor": fact(str(item["anchor"]), str(item["anchor_color"])),
    }
    order = FACT_ORDERS[int(item["fact_order_index"])]
    dossier = " ".join(facts[name] for name in order)
    query_value = SURFACE_VALUES[query_surface][canonical_colors[target_variant]]
    question = f"Does {selected_entity} have visible color {query_value}?"
    raw_prompt = (
        "Use only the dossier. Reply Yes when the question agrees with it, "
        "and No when it does not.\n"
        f"Dossier:\n{dossier}\nQuestion: {question}\n"
        "Write only the reply word."
    )
    instruction = raw_prompt.splitlines()[0]
    raw_spans = {
        "instruction_end": mark_source.mark(raw_prompt, instruction, occurrence="first"),
        "entity0_fact_end": mark_source.mark(raw_prompt, facts["entity0"], occurrence="first"),
        "entity1_fact_end": mark_source.mark(raw_prompt, facts["entity1"], occurrence="first"),
        "dossier_end": mark_source.mark(raw_prompt, facts[order[-1]], occurrence="first"),
        "query_end": mark_source.mark(raw_prompt, question, occurrence="last"),
    }
    rendered = behavior.render_native(
        tokenizer, model_name, raw_prompt, with_system=False
    ) + ASSISTANT_PREFILL
    input_ids = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(tokenizer, rendered, raw_prompt, raw_spans)
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    prefix = " "
    candidate_token_ids = {
        f"a{index}": behavior.continuation_ids(
            tokenizer, rendered, prefix, answer
        )
        for index, answer in enumerate(OUTPUT_PAIR)
    }
    return {
        "schema_version": "phase1090_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{model_name}.{cell}.{split}.{item['item_id']}.{state}",
        "unit_id": f"{cell}.{split}.{item['item_id']}",
        "family": cell,
        "cell": cell,
        "operation": operation,
        "color_pair": [color0, color1],
        "world": world,
        "world_route": world_route,
        "surface_route": route,
        "fact_surface": fact_surface,
        "query_surface": query_surface,
        "split": split,
        "item_id": item["item_id"],
        "state": state,
        "panel": panel,
        "mapping": target_variant,
        "target_variant": target_variant,
        "query": binding,
        "binding": binding,
        "selected_entity": selected_entity,
        "entity0": item["entity0"],
        "entity1": item["entity1"],
        "anchor": item["anchor"],
        "anchor_variant": int(item["anchor_variant"]),
        "anchor_color": item["anchor_color"],
        "canonical_target_color": canonical_colors[target_variant],
        "fact_surface_values": [
            SURFACE_VALUES[fact_surface][value] for value in bound_colors
        ],
        "query_surface_value": query_value,
        "facts": facts,
        "dossier": dossier,
        "question": question,
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
        "semantic_answer_index": semantic_answer,
        "answer_index": semantic_answer,
        "target_answer": target_answer,
        "answer_labels": list(OUTPUT_PAIR),
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])] for key, values in candidate_token_ids.items()
        },
        "expected_class": f"a{semantic_answer}",
        "continuation_prefix": prefix,
    }


def audit_model(model_name: str, tokenizer, cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)
    checks: dict[str, bool] = {}
    checks["complete_factorial_units"] = all(
        {row["state"] for row in rows} == set(STATES)
        for rows in by_unit.values()
    )
    checks["all_surface_routes_present"] = {
        row["surface_route"] for row in cases
    } == set(SURFACE_ROUTES)
    checks["active_truth_formula"] = all(
        row["semantic_answer_index"]
        == int(int(row["binding"]) != int(row["target_variant"]))
        for row in cases if row["panel"] == "active"
    )
    checks["null_truth_formula"] = all(
        row["semantic_answer_index"]
        == int(int(row["anchor_variant"]) != int(row["target_variant"]))
        for row in cases if row["panel"] == "field_null"
    )
    checks["one_true_one_false_every_side"] = all(
        sorted(
            row["semantic_answer_index"]
            for row in rows
            if row["panel"] == panel and row["binding"] == binding
        ) == [0, 1]
        for rows in by_unit.values()
        for panel in PANELS
        for binding in (0, 1)
    )
    checks["role_positions_valid"] = all(
        all(0 <= int(position) < len(row["input_ids"])
            for position in row["role_positions"].values())
        for row in cases
    )
    checks["query_after_dossier"] = all(
        row["role_positions"]["dossier_end"]
        < row["role_positions"]["query_end"]
        <= row["role_positions"]["answer_boundary"]
        for row in cases
    )
    checks["single_token_outputs"] = all(
        all(len(value) == 1 for value in row["candidate_token_ids"].values())
        for row in cases
    )
    checks["independent_entity_splits"] = all(
        set(ENTITY_POOLS[world]["discovery"]).isdisjoint(
            ENTITY_POOLS[world]["confirmation"]
        ) for world in BASE_WORLDS
    )
    checks["balanced_anchor_orientation"] = all(
        Counter(
            row["anchor_variant"] for row in cases
            if row["cell"] == cell and row["split"] == split
            and row["state"] == "cactive_m0_q0"
        ) == Counter({0: 2, 1: 2})
        for cell in CELLS for split in SPLITS
    )
    checks["surface_lexicons_complete"] = all(
        set(values) == set(COLORS) for values in SURFACE_VALUES.values()
    )
    checks["all_checks_boolean"] = all(isinstance(value, bool) for value in checks.values())
    return {
        "schema_version": "phase1090_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "surface_token_widths": {
            surface: {
                color: len(tokenizer.encode(value, add_special_tokens=False))
                for color, value in values.items()
            }
            for surface, values in SURFACE_VALUES.items()
        },
        "case_digest": digest(cases),
    }


def build_model_cases(model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    case_index = 0
    for cell in CELLS:
        for split in SPLITS:
            for item in split_items(cell, split):
                for state in STATES:
                    cases.append(build_case(
                        tokenizer, model_name, cell, split, item, state, case_index
                    ))
                    case_index += 1
    return cases, audit_model(model_name, tokenizer, cases)


def main() -> None:
    protocol_root = OUT_ROOT / "protocol"
    model_case_digests = {}
    model_audits = {}
    for model_name in MODELS:
        cases, audit = build_model_cases(model_name)
        if not audit["all_checks_passed"]:
            failed = [name for name, value in audit["checks"].items() if not value]
            raise RuntimeError(f"protocol audit failed for {model_name}: {failed}")
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_case_digests[model_name] = audit["case_digest"]
        model_audits[model_name] = audit
    source = read_json(SOURCE_PHASE1089 / "analysis" / "final_summary.json")
    prereg = {
        "schema_version": "phase1090_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "operations": list(OPERATIONS),
        "color_pairs": [list(pair) for pair in COLOR_PAIRS],
        "base_worlds": list(BASE_WORLDS),
        "surface_values": SURFACE_VALUES,
        "surface_routes": list(SURFACE_ROUTES),
        "mixed_surface_routes": list(MIXED_SURFACE_ROUTES),
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "states": list(STATES),
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "generation_steps": GENERATION_STEPS,
        "generation_units_per_cell_split": GENERATION_UNITS_PER_CELL_SPLIT,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "model_case_digests": model_case_digests,
        "source_phase1089_protocol_digest": source["protocol_digest"],
        "source_phase1089_summary_digest": source["summary_digest"],
        "interpretation_limits": [
            "This is a behavioral feasibility screen, not hidden-state evidence.",
            "Mixed prompts are controlled lexical tests, not fully natural multilingual discourse.",
            "French and hexadecimal forms may differ in training frequency and tokenization.",
            "A passing route only authorizes a separately preregistered hidden-state test.",
        ],
        "automatic_next": {
            "hidden_protocol_if": "P1-P5 pass and at least two mixed routes are viable in two models.",
            "otherwise": "Stop before cross-surface hidden-state collection.",
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    audit = {
        "schema_version": "phase1090_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audits": model_audits,
        "all_checks_passed": all(row["all_checks_passed"] for row in model_audits.values()),
    }
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)
    print({
        "phase": PHASE,
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "all_checks_passed": audit["all_checks_passed"],
        "protocol_digest": prereg["protocol_digest"],
    })


if __name__ == "__main__":
    main()
