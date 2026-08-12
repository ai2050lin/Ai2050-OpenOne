#!/usr/bin/env python3
"""Freeze Phase1089 truth-matched color-binding protocol.

Phase1089 keeps the Phase1087 same-token binding swap, but changes the
query-irrelevant anchor from a third color to one member of the tested color
pair.  Consequently, both active and null panels contain one true and one
false query on every binding side before the signed binding contrast is
formed.  This is a stricter marginal truth control; it still cannot remove
every nonlinear truth-by-binding interaction.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1087_color_relation_protocol as base


PHASE = 1089
PROTOCOL_REVISION = 1
MODELS = base.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
COLORS = base.COLORS
COLOR_PAIRS = base.COLOR_PAIRS
OPERATIONS = base.OPERATIONS
WORLDS = base.WORLDS
CELLS = base.CELLS
FAMILIES = CELLS
SPLITS = base.SPLITS
PANELS = base.PANELS
OUTPUT_PAIRS = base.OUTPUT_PAIRS
CODE_WORDS = base.CODE_WORDS
STATES = base.STATES
TARGET_RELATIVE_DEPTH_MIN = base.TARGET_RELATIVE_DEPTH_MIN
TARGET_RELATIVE_DEPTH_MAX = base.TARGET_RELATIVE_DEPTH_MAX
CAPTURE_ROLES = base.CAPTURE_ROLES
PRIMARY_PROFILE_ROLES = base.PRIMARY_PROFILE_ROLES
PRE_QUERY_ROLES = base.PRE_QUERY_ROLES
ITEMS_PER_CELL_SPLIT = 6
GENERATION_UNITS_PER_FAMILY_SPLIT = 4
GENERATION_STEPS = base.GENERATION_STEPS
ASSISTANT_PREFILL = base.ASSISTANT_PREFILL
SIGNED_PROJECTION_DIM = base.SIGNED_PROJECTION_DIM
SIGNED_PROJECTION_REPLICATES = base.SIGNED_PROJECTION_REPLICATES
# Keep the Phase1088 projection basis to permit within-model cross-phase audit.
SIGNED_PROJECTION_SEED = 1088001
SIGNED_FIELDS = ("active_binding", "field_null", "content")
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1089_truth_matched_color_binding"
)
SOURCE_PHASE1088 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1088_answer_balanced_color_binding"
)

ENTITY_POOLS = base.ENTITY_POOLS
FACT_PATTERNS = base.FACT_PATTERNS
QUERY_PATTERNS = base.QUERY_PATTERNS
SHELLS = base.SHELLS
FACT_ORDERS = base.FACT_ORDERS


EVIDENCE_THRESHOLDS = {
    **base.EVIDENCE_THRESHOLDS,
    "candidate_accuracy_for_operation_behavior": 0.80,
    "minimum_null_candidate_accuracy": 0.80,
    "minimum_behavior_worlds_per_operation": 3,
    "minimum_behavior_operations": 6,
    "minimum_behavior_models": 2,
    "minimum_cross_phase_pair_gram_cosine": 0.50,
    "minimum_cross_phase_content_over_null_advantage": 0.10,
    "minimum_cross_phase_models": 2,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "All static audits pass, including exact token multiset, one-true/one-"
        "false balance on every binding side in both panels, and balanced "
        "anchor orientation."
    ),
    "P2": (
        "At least two FP16 models pass active and truth-matched-null behavior "
        "for at least six color pairs in three entity worlds."
    ),
    "P3": "Both signed sketches pass their frozen projection audits.",
    "P4": (
        "The truth-matched content field repeats across independent splits "
        "and worlds with a 0.10 advantage over the matched null in two models."
    ),
    "P5": (
        "Centered color-pair residuals retrieve at least six of eight pairs "
        "across independent samples in two models and both sketches."
    ),
    "P6": (
        "Color-pair residual identity transfers to at least three held-out "
        "entity worlds and beats the truth-matched null."
    ),
    "P7": (
        "At least two directed healthy model pairs repeat pair Gram geometry "
        "and beat null geometry by 0.10."
    ),
    "P8": (
        "Template and output controls do not exceed the truth-matched content "
        "magnitude in two models."
    ),
    "P9": "At least two models pass all FP16 and finite-value audits.",
    "P10": (
        "The Phase1089 pair Gram repeats the Phase1088 pair Gram within model "
        "and exceeds the corresponding null similarity by 0.10 in two models."
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
cell_id = base.cell_id
split_cell = base.split_cell
operation_colors = base.operation_colors
state_factors = base.state_factors
output_code_pair = base.output_code_pair


def split_items(cell: str, split: str) -> tuple[dict[str, Any], ...]:
    """Reuse independent entity pools while balancing the anchor color."""
    operation, world = split_cell(cell)
    color0, color1 = operation_colors(operation)
    pool = ENTITY_POOLS[world][split]
    rows = []
    for local_index in range(ITEMS_PER_CELL_SPLIT):
        anchor_variant = local_index % 2
        rows.append({
            "item_id": f"{cell}.{split}.{local_index:02d}",
            "base_item_id": f"{world}.{split}.{local_index:02d}",
            "entity0": pool[local_index],
            "entity1": pool[(local_index + 3) % len(pool)],
            "anchor": pool[(local_index + 6) % len(pool)],
            "anchor_variant": anchor_variant,
            "anchor_color": (color0, color1)[anchor_variant],
            "fact_order_index": local_index,
            "operation": operation,
            "world": world,
        })
    return tuple(rows)


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
    entity_colors = colors if binding == 0 else tuple(reversed(colors))
    target_color = colors[target_variant]
    true_word, false_word = output_code_pair(output_set)
    semantic_codes = (true_word, false_word)
    selected_entity = (
        str(item["entity0"]) if panel == "active" else str(item["anchor"])
    )
    if panel == "active":
        semantic_answer = int(binding != target_variant)
    else:
        semantic_answer = int(int(item["anchor_variant"]) != target_variant)
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
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
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
        "schema_version": "phase1089_case.v1",
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
        "anchor_variant": int(item["anchor_variant"]),
        "anchor_color": item["anchor_color"],
        "entity_colors": list(entity_colors),
        "target_color": target_color,
        "fact_order": list(order),
        "facts": facts,
        "dossier": dossier,
        "question": question,
        "control_type": (
            "truth_marginal_matched_query_irrelevant_binding_swap"
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
            key: [int(values[0])] for key, values in candidate_token_ids.items()
        },
        "expected_class": f"a{semantic_answer}",
        "continuation_prefix": prefix,
        "label_token_widths": [
            encoded_width(tokenizer, value) for value in semantic_codes
        ],
    }


def signed_pair_records(state_tensor, values, template: int, output_set: int):
    """Return answer-balanced active, truth-matched null, and interaction."""
    active_left = 0.5 * (
        state_tensor(values, template, "active", 0, 0, output_set)
        + state_tensor(values, template, "active", 1, 0, output_set)
    )
    active_right = 0.5 * (
        state_tensor(values, template, "active", 0, 1, output_set)
        + state_tensor(values, template, "active", 1, 1, output_set)
    )
    null_left = 0.5 * (
        state_tensor(values, template, "field_null", 0, 0, output_set)
        + state_tensor(values, template, "field_null", 1, 0, output_set)
    )
    null_right = 0.5 * (
        state_tensor(values, template, "field_null", 0, 1, output_set)
        + state_tensor(values, template, "field_null", 1, 1, output_set)
    )
    return (
        ("active_binding", active_left, active_right, 0),
        ("field_null", null_left, null_right, 0),
        (
            "content",
            0.5 * (active_left + null_right),
            0.5 * (active_right + null_left),
            0,
        ),
    )


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
    checks["anchor_color_inside_pair"] = all(
        row["anchor_color"] in row["color_pair"] for row in cases
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
    side_truth_balance_ok = True
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
                        multiset_ok &= (
                            Counter(row0["input_ids"]) == Counter(row1["input_ids"])
                        )
                        length_ok &= len(row0["input_ids"]) == len(row1["input_ids"])
                        question_ok &= row0["question"] == row1["question"]
                for binding in (0, 1):
                    for output_set in (0, 1):
                        for panel in PANELS:
                            answers = [
                                lookup[
                                    f"t{template}_c{panel}_m{target}_q{binding}_w{output_set}"
                                ]["semantic_answer_index"]
                                for target in (0, 1)
                            ]
                            side_truth_balance_ok &= sorted(answers) == [0, 1]
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
    checks["one_true_one_false_every_binding_side"] = bool(side_truth_balance_ok)

    checks["balanced_anchor_orientation"] = all(
        Counter(
            row["anchor_variant"]
            for row in rows
            if row["template"] == 0 and row["panel"] == "field_null"
            and row["target_variant"] == 0 and row["binding"] == 0
            and row["output_set"] == 0
        ) == Counter({0: ITEMS_PER_CELL_SPLIT // 2, 1: ITEMS_PER_CELL_SPLIT // 2})
        for family in FAMILIES
        for split in SPLITS
        for rows in [[
            row for row in cases
            if row["family"] == family and row["split"] == split
        ]]
    )
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
        "schema_version": "phase1089_protocol_model_audit.v1",
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

    source = read_json(SOURCE_PHASE1088 / "analysis" / "final_summary.json")
    prereg = {
        "schema_version": "phase1089_preregistration.v1",
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
            "cross_phase_rule": (
                "Phase1088 and Phase1089 use the same within-model projection "
                "basis; projected coordinates are never compared across models."
            ),
        },
        "contrast_definition": {
            "active_left": "mean(active target0/binding0, active target1/binding0)",
            "active_right": "mean(active target0/binding1, active target1/binding1)",
            "null_left": "mean(pair-colored anchor target0/binding0, target1/binding0)",
            "null_right": "mean(pair-colored anchor target0/binding1, target1/binding1)",
            "content": "active binding direction minus truth-matched null binding direction",
        },
        "truth_balance": {
            "active_each_binding_side": ["true", "false"],
            "null_each_binding_side": ["true", "false"],
            "anchor_orientation": "balanced 3/3 within every cell and split",
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
        "source_phase1088_protocol_digest": source["protocol_digest"],
        "source_phase1088_summary_digest": source["summary_digest"],
        "interpretation_limits": [
            "The null exactly matches marginal truth counts on each binding side, not every nonlinear truth-by-binding interaction.",
            "The same color tokens recur across splits, so pair identity can still be lexical rather than semantic.",
            "A stable pair fingerprint is descriptive until it transfers across independent lexical surfaces.",
            "The scan is restricted to relative depth 0.30-0.45 and cannot exclude structure elsewhere.",
            "No result establishes causality, a neuron code, brain homology, optimality, or new mathematics.",
        ],
        "automatic_next": {
            "cross_surface_behavior_pilot_if": "P1-P6 and P9-P10 pass in at least two models.",
            "stop_before_cross_surface_hidden_scan_if": "Cross-surface behavior fails in two models.",
            "causal_authorization": "Never from Phase1089 alone.",
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    global_audit = {
        "schema_version": "phase1089_protocol_audit.v1",
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
