#!/usr/bin/env python3
"""Freeze the Phase1087 same-word color-relation protocol.

The protocol isolates one narrow semantic relation.  Each counterfactual keeps
the entity names, color words, syntax, and output vocabulary fixed and swaps
only which of two entities receives which color.  A query-irrelevant anchor
uses the same binding swap as a matched lexical/reordering null.
"""

from __future__ import annotations

import itertools
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1086_signed_shared_field_protocol as base


PHASE = 1087
PROTOCOL_REVISION = 1
MODELS = base.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
COLORS = (
    "red", "blue", "green", "yellow",
    "black", "white", "orange", "purple",
)
COLOR_PAIRS = tuple(
    (COLORS[index], COLORS[(index + 1) % len(COLORS)])
    for index in range(len(COLORS))
)
OPERATIONS = tuple(f"{left}_{right}" for left, right in COLOR_PAIRS)
WORLDS = ("people", "animals", "objects", "rare_entities")
CELLS = tuple(
    f"{operation}__{world}"
    for operation in OPERATIONS
    for world in WORLDS
)
FAMILIES = CELLS
SPLITS = ("discovery", "confirmation")
PANELS = ("active", "field_null")
OUTPUT_PAIRS = (("Yes", "No"), ("True", "False"))
CODE_WORDS = tuple(value for pair in OUTPUT_PAIRS for value in pair)

# mapping is the queried color variant; query is the physical binding state.
STATES = tuple(
    f"t{template}_c{panel}_m{target}_q{binding}_w{output_set}"
    for template in (0, 1)
    for panel in PANELS
    for target in (0, 1)
    for binding in (0, 1)
    for output_set in (0, 1)
)

TARGET_RELATIVE_DEPTH_MIN = 0.30
TARGET_RELATIVE_DEPTH_MAX = 0.45
CAPTURE_ROLES = (
    "entity0_fact_end", "entity1_fact_end", "dossier_end",
    "query_end", "answer_boundary",
)
PRIMARY_PROFILE_ROLES = ("answer_boundary",)
PRE_QUERY_ROLES = (
    "entity0_fact_end", "entity1_fact_end", "dossier_end",
)
ITEMS_PER_CELL_SPLIT = 6
GENERATION_UNITS_PER_FAMILY_SPLIT = 4
GENERATION_STEPS = 12
ASSISTANT_PREFILL = "Answer:"
SIGNED_PROJECTION_DIM = 96
SIGNED_PROJECTION_REPLICATES = 2
SIGNED_PROJECTION_SEED = 1087001
SIGNED_FIELDS = ("active_truth", "field_null", "content")
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1087_color_relation"
)
SOURCE_PHASE1086 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1086_signed_shared_field"
    / "analysis" / "final_summary.json"
)


EVIDENCE_THRESHOLDS = {
    "candidate_accuracy_for_operation_behavior": 0.80,
    "generation_target_before_distractor_accuracy": 0.75,
    "minimum_behavior_worlds_per_operation": 3,
    "minimum_behavior_operations": 6,
    "minimum_behavior_models": 2,
    "maximum_projection_median_abs_norm_error": 0.08,
    "maximum_projection_p95_abs_norm_error": 0.20,
    "minimum_shared_split_cosine": 0.50,
    "minimum_shared_content_over_null_advantage": 0.10,
    "minimum_shared_worlds": 3,
    "minimum_cross_world_pairs": 8,
    "minimum_surface_transfer_cosine": 0.40,
    "minimum_output_transfer_cosine": 0.40,
    "minimum_attribute_top1": 6,
    "permutation_p_max": 0.01,
    "minimum_attribute_heldout_worlds": 3,
    "minimum_heldout_pair_cells": 24,
    "maximum_surface_to_content_ratio": 1.0,
    "minimum_cross_model_geometry_cosine": 0.50,
    "minimum_cross_model_geometry_advantage": 0.10,
    "minimum_cross_model_geometry_pairs": 2,
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_hidden_finite_fraction": 0.97,
    "minimum_numeric_models": 2,
    "pre_query_tolerance": 1e-8,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": "All protocol, token-multiset, truth, role, split, and prefix audits pass.",
    "P2": "At least two FP16 models pass at least six color pairs in three entity worlds.",
    "P3": "Both frozen signed sketches pass their norm-distortion audits.",
    "P4": (
        "In at least two models and both sketches, the common color-relation "
        "field repeats across independent item splits in three worlds and "
        "beats the query-irrelevant binding null by at least 0.10."
    ),
    "P5": (
        "In at least two models and both sketches, at least eight of twelve "
        "directed entity-world pairs repeat that field with a 0.10 null advantage."
    ),
    "P6": (
        "The field transfers across two natural surfaces and two output word "
        "sets, while their magnitude controls do not exceed content."
    ),
    "P7": (
        "Centered color-pair residuals retrieve at least six of eight pairs "
        "across independent items in at least two models and both sketches."
    ),
    "P8": (
        "Color-pair residual identity transfers into at least three held-out "
        "entity worlds and beats the matched binding null."
    ),
    "P9": (
        "At least two directed model pairs repeat within-model color-pair Gram "
        "geometry and beat null geometry by at least 0.10."
    ),
    "P10": (
        "At least two models pass FP16, finite-value, identity, and pre-query "
        "audits; every failing model remains explicitly exploratory."
    ),
    "P11": (
        "A relation centroid learned without one color pair predicts that held-out "
        "pair in at least 24 of 32 pair-world cells in both sketches for two models."
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


ENTITY_POOLS = {
    "people": {
        "discovery": (
            "alice", "brian", "carol", "david", "emma",
            "frank", "grace", "henry", "irene",
        ),
        "confirmation": (
            "james", "karen", "louis", "maria", "nathan",
            "olivia", "peter", "quinn", "rachel",
        ),
    },
    "animals": {
        "discovery": (
            "tiger", "rabbit", "dolphin", "falcon", "panda",
            "otter", "camel", "zebra", "badger",
        ),
        "confirmation": (
            "lion", "beaver", "heron", "koala", "donkey",
            "gecko", "walrus", "yak", "lemur",
        ),
    },
    "objects": {
        "discovery": (
            "lantern", "kettle", "mirror", "basket", "helmet",
            "pillow", "bottle", "carpet", "violin",
        ),
        "confirmation": (
            "hammer", "teapot", "compass", "jacket", "ladder",
            "camera", "wallet", "anchor", "trumpet",
        ),
    },
    "rare_entities": {
        "discovery": (
            "gryphon", "basilisk", "chimera", "sphinx", "kraken",
            "golem", "kelpie", "banshee", "manticore",
        ),
        "confirmation": (
            "taotie", "qilin", "pixiu", "tengu", "djinn",
            "selkie", "cockatrice", "wolpertinger", "hippogriff",
        ),
    },
}

FACT_PATTERNS = {
    0: "{entity} looks {color}.",
    1: "The visible hue of {entity} is {color}.",
}
QUERY_PATTERNS = {
    0: "Does {entity} look {color}?",
    1: "Is {color} the visible hue of {entity}?",
}
SHELLS = {
    0: (
        "Use only the dossier. Reply {true_word} when the question agrees "
        "with it, and {false_word} when it does not.\n"
        "Dossier:\n{dossier}\nQuestion: {question}\n"
        "Write only the reply word."
    ),
    1: (
        "Judge the final question from the descriptions below. Return "
        "{true_word} for agreement and {false_word} for disagreement.\n"
        "Descriptions:\n{dossier}\nCheck: {question}\n"
        "Return just the reply word."
    ),
}
FACT_ORDERS = tuple(itertools.permutations(("entity0", "entity1", "anchor")))


def cell_id(operation: str, world: str) -> str:
    return f"{operation}__{world}"


def split_cell(cell: str) -> tuple[str, str]:
    return tuple(cell.split("__", 1))  # type: ignore[return-value]


def operation_colors(operation: str) -> tuple[str, str]:
    index = OPERATIONS.index(operation)
    return COLOR_PAIRS[index]


def split_items(cell: str, split: str) -> tuple[dict[str, Any], ...]:
    operation, world = split_cell(cell)
    pair_index = OPERATIONS.index(operation)
    pool = ENTITY_POOLS[world][split]
    rows = []
    for local_index in range(ITEMS_PER_CELL_SPLIT):
        entity0 = pool[local_index]
        entity1 = pool[(local_index + 3) % len(pool)]
        anchor = pool[(local_index + 6) % len(pool)]
        anchor_index = (pair_index + local_index + 3) % len(COLORS)
        while COLORS[anchor_index] in COLOR_PAIRS[pair_index]:
            anchor_index = (anchor_index + 1) % len(COLORS)
        rows.append({
            "item_id": f"{cell}.{split}.{local_index:02d}",
            "base_item_id": f"{world}.{split}.{local_index:02d}",
            "entity0": entity0,
            "entity1": entity1,
            "anchor": anchor,
            "anchor_color": COLORS[anchor_index],
            "fact_order_index": local_index,
            "operation": operation,
            "world": world,
        })
    return tuple(rows)


def state_factors(state: str) -> tuple[int, str, int, int, int]:
    match = re.fullmatch(
        r"t([01])_c(active|field_null)_m([01])_q([01])_w([01])",
        state,
    )
    if not match:
        raise ValueError(f"invalid state: {state}")
    return (
        int(match.group(1)), str(match.group(2)), int(match.group(3)),
        int(match.group(4)), int(match.group(5)),
    )


def output_code_pair(output_set: int) -> tuple[str, str]:
    return OUTPUT_PAIRS[output_set]


def encoded_width(tokenizer, text: str) -> int:
    return len(tokenizer.encode(" " + text, add_special_tokens=False))


def build_case(
    tokenizer,
    model_name: str,
    cell: str,
    split: str,
    item: dict[str, Any],
    item_local_index: int,
    state: str,
    case_index: int,
) -> dict[str, Any]:
    operation, world = split_cell(cell)
    template, panel, target_variant, binding, output_set = state_factors(state)
    color0, color1 = operation_colors(operation)
    colors = (color0, color1)
    entity_colors = (
        (color0, color1) if binding == 0 else (color1, color0)
    )
    target_color = colors[target_variant]
    true_word, false_word = output_code_pair(output_set)
    semantic_codes = (true_word, false_word)
    selected_entity = (
        str(item["entity0"]) if panel == "active" else str(item["anchor"])
    )
    semantic_answer = (
        int(binding != target_variant) if panel == "active" else 1
    )
    target_answer = semantic_codes[semantic_answer]

    facts = {
        "entity0": FACT_PATTERNS[template].format(
            entity=item["entity0"], color=entity_colors[0]
        ),
        "entity1": FACT_PATTERNS[template].format(
            entity=item["entity1"], color=entity_colors[1]
        ),
        "anchor": FACT_PATTERNS[template].format(
            entity=item["anchor"], color=item["anchor_color"]
        ),
    }
    base_order = FACT_ORDERS[int(item["fact_order_index"])]
    order = base_order if template == 0 else tuple(reversed(base_order))
    dossier = " ".join(facts[name] for name in order)
    question = QUERY_PATTERNS[template].format(
        entity=selected_entity, color=target_color
    )
    raw_prompt = SHELLS[template].format(
        true_word=true_word,
        false_word=false_word,
        dossier=dossier,
        question=question,
    )
    instruction = raw_prompt.splitlines()[0]
    raw_spans = {
        "instruction_end": mark_source.mark(
            raw_prompt, instruction, occurrence="first"
        ),
        "entity0_fact_end": mark_source.mark(
            raw_prompt, facts["entity0"], occurrence="first"
        ),
        "entity1_fact_end": mark_source.mark(
            raw_prompt, facts["entity1"], occurrence="first"
        ),
        "dossier_end": mark_source.mark(
            raw_prompt, facts[order[-1]], occurrence="first"
        ),
        "query_end": mark_source.mark(
            raw_prompt, question, occurrence="last"
        ),
    }
    rendered = behavior.render_native(
        tokenizer, model_name, raw_prompt, with_system=False
    ) + ASSISTANT_PREFILL
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(tokenizer, rendered, raw_prompt, raw_spans)
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    prefix = " "
    candidate_token_ids = {
        f"a{index}": behavior.continuation_ids(
            tokenizer, rendered, prefix, answer
        )
        for index, answer in enumerate(semantic_codes)
    }
    return {
        "schema_version": "phase1087_case.v1",
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
        "split": split,
        "item_id": item["item_id"],
        "base_item_id": item["base_item_id"],
        "item_local_index": item_local_index,
        "state": state,
        "template": template,
        "panel": panel,
        "mapping": target_variant,
        "target_variant": target_variant,
        "query": binding,
        "binding": binding,
        "label_swap": output_set,
        "output_set": output_set,
        "label_pair": list(semantic_codes),
        "answer_labels": list(semantic_codes),
        "selected_entity": selected_entity,
        "entity0": item["entity0"],
        "entity1": item["entity1"],
        "anchor": item["anchor"],
        "anchor_color": item["anchor_color"],
        "entity_colors": list(entity_colors),
        "target_color": target_color,
        "fact_order": list(order),
        "facts": facts,
        "dossier": dossier,
        "question": question,
        "control_type": (
            "query_irrelevant_same_word_binding_swap"
            if panel == "field_null" else None
        ),
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
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])]
            for key, values in candidate_token_ids.items()
        },
        "expected_class": f"a{semantic_answer}",
        "continuation_prefix": prefix,
        "label_token_widths": [
            encoded_width(tokenizer, value) for value in semantic_codes
        ],
    }


def audit_model(
    model_name: str,
    tokenizer,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)

    checks: dict[str, bool] = {}
    checks["complete_factorial_units"] = all(
        {row["state"] for row in rows} == set(STATES)
        for rows in by_unit.values()
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
    checks["active_truth_is_binding_target_equality"] = all(
        row["semantic_answer_index"]
        == int(int(row["binding"]) != int(row["target_variant"]))
        for row in cases if row["panel"] == "active"
    )
    checks["null_truth_is_binding_invariant_false"] = all(
        row["semantic_answer_index"] == 1
        for row in cases if row["panel"] == "field_null"
    )
    checks["anchor_color_outside_pair"] = all(
        row["anchor_color"] not in row["color_pair"] for row in cases
    )
    checks["output_words_single_token_in_context"] = all(
        all(len(values) == 1 for values in row["candidate_token_ids"].values())
        for row in cases
    )
    checks["no_candidate_code_indirection"] = all(
        "candidate a" not in row["raw_prompt"].casefold()
        and "candidate b" not in row["raw_prompt"].casefold()
        for row in cases
    )

    multiset_ok = True
    length_ok = True
    question_ok = True
    dossier_panel_ok = True
    for rows in by_unit.values():
        lookup = {row["state"]: row for row in rows}
        for template in (0, 1):
            for panel in PANELS:
                for target in (0, 1):
                    for output_set in (0, 1):
                        row0 = lookup[
                            f"t{template}_c{panel}_m{target}_q0_w{output_set}"
                        ]
                        row1 = lookup[
                            f"t{template}_c{panel}_m{target}_q1_w{output_set}"
                        ]
                        multiset_ok &= Counter(row0["input_ids"]) == Counter(row1["input_ids"])
                        length_ok &= len(row0["input_ids"]) == len(row1["input_ids"])
                        question_ok &= row0["question"] == row1["question"]
                for binding in (0, 1):
                    for target in (0, 1):
                        for output_set in (0, 1):
                            active = lookup[
                                f"t{template}_cactive_m{target}_q{binding}_w{output_set}"
                            ]
                            null = lookup[
                                f"t{template}_cfield_null_m{target}_q{binding}_w{output_set}"
                            ]
                            dossier_panel_ok &= active["dossier"] == null["dossier"]
    checks["binding_swap_exact_token_multiset"] = bool(multiset_ok)
    checks["binding_swap_exact_token_length"] = bool(length_ok)
    checks["question_fixed_across_binding"] = bool(question_ok)
    checks["active_null_share_exact_dossier"] = bool(dossier_panel_ok)

    checks["independent_entity_splits"] = all(
        set(ENTITY_POOLS[world]["discovery"]).isdisjoint(
            ENTITY_POOLS[world]["confirmation"]
        )
        for world in WORLDS
    )
    degree = Counter(color for pair in COLOR_PAIRS for color in pair)
    checks["balanced_color_pair_graph"] = (
        set(degree) == set(COLORS) and all(value == 2 for value in degree.values())
    )
    checks["all_checks_boolean"] = all(
        isinstance(value, bool) for value in checks.values()
    )
    return {
        "schema_version": "phase1087_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "color_token_widths": {
            color: encoded_width(tokenizer, color) for color in COLORS
        },
        "case_digest": digest(cases),
    }


def build_model_cases(
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    case_index = 0
    for cell in CELLS:
        for split in SPLITS:
            for item_local_index, item in enumerate(split_items(cell, split)):
                for state in STATES:
                    cases.append(build_case(
                        tokenizer, model_name, cell, split, item,
                        item_local_index, state, case_index,
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
            failed = [
                name for name, value in audit["checks"].items() if not value
            ]
            raise RuntimeError(
                f"protocol audit failed for {model_name}: {failed}"
            )
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_case_digests[model_name] = audit["case_digest"]
        model_audits[model_name] = audit

    source = read_json(SOURCE_PHASE1086)
    prereg = {
        "schema_version": "phase1087_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "operations": list(OPERATIONS),
        "color_pairs": [list(pair) for pair in COLOR_PAIRS],
        "worlds": list(WORLDS),
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "states": list(STATES),
        "capture_roles": list(CAPTURE_ROLES),
        "primary_profile_roles": list(PRIMARY_PROFILE_ROLES),
        "relative_depth_range": [
            TARGET_RELATIVE_DEPTH_MIN, TARGET_RELATIVE_DEPTH_MAX
        ],
        "signed_fields": list(SIGNED_FIELDS),
        "projection": {
            "type": "deterministic_rademacher",
            "dimension_per_replicate": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
            "cross_model_rule": (
                "Never compare projected coordinates across models; compare "
                "only within-model color-pair Gram geometry."
            ),
        },
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": (
            len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES)
        ),
        "unit_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "generation_units_per_family_split": GENERATION_UNITS_PER_FAMILY_SPLIT,
        "generation_steps": GENERATION_STEPS,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "model_case_digests": model_case_digests,
        "source_phase1086_protocol_digest": source["protocol_digest"],
        "source_phase1086_summary_digest": source["summary_digest"],
        "interpretation_limits": [
            "This is a controlled color-binding relation, not a complete color concept or language theory.",
            "The field-null control matches words and binding movement but does not remove all query or output work.",
            "A shared signed field may still be generic verification rather than stored color semantics.",
            "Color-pair residual retrieval may reflect lexical pair identity unless it beats the same-word null.",
            "Random projected coordinates are model-specific and are never compared directly across models.",
            "No result establishes a single neuron code, brain homology, biological optimality, or new mathematics.",
        ],
        "automatic_next": {
            "stop_before_hidden_scan_if": "P1 or P2 fails.",
            "stop_component_selection_if": (
                "Any content-specific gate P4-P9, P11, or numeric gate P10 fails."
            ),
            "causal_authorization": (
                "Only if P1-P11 all pass prospectively; otherwise retain the descriptive map."
            ),
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    global_audit = {
        "schema_version": "phase1087_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audits": model_audits,
        "all_checks_passed": all(
            audit["all_checks_passed"] for audit in model_audits.values()
        ),
    }
    global_audit["audit_digest"] = digest(global_audit)
    write_json(protocol_root / "audit.json", global_audit)
    print({
        "phase": PHASE,
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "all_checks_passed": global_audit["all_checks_passed"],
        "protocol_digest": prereg["protocol_digest"],
    })


if __name__ == "__main__":
    main()
