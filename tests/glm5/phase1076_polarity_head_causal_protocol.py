#!/usr/bin/env python3
"""Freeze the Phase1076 held-out head-output causal protocol."""

from __future__ import annotations

import copy
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1051_natural_behavior_protocol as behavior
import phase1075_relation_polarity_protocol as source


PHASE = 1076
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4")
PRECISION = "fp16"
QUANTIZATION = "none"
RELATION = "height"
CONTRASTS = ("polarity", "surface")
TASKS_BY_CONTRAST = {
    "polarity": ("max", "min"),
    "surface": ("first", "second"),
}
TEMPLATES = (2, 3)
REPLICATES = (1, 2)
PATHS = source.PATHS
LAYOUTS = source.LAYOUTS
ORIENTATIONS = source.ORIENTATIONS
LEXICAL_BRANCHES = source.LEXICAL_BRANCHES
ASSISTANT_PREFILL = source.ASSISTANT_PREFILL
NATURAL_GENERATION_STEPS = 10
INTERVENTIONS = (
    "selected_swap",
    "control_swap",
    "selected_zero",
    "control_zero",
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1076_polarity_head_causal"
)
SOURCE_ROOT = source.OUT_ROOT

# Frozen before any Phase1076 forward pass.
GATES = {
    "behavior_finite_rate_min": 0.995,
    "behavior_contrast_accuracy_min": 0.82,
    "behavior_task_accuracy_min": 0.78,
    "behavior_path_accuracy_min": 0.78,
    "behavior_natural_semantic_first_min": 0.65,
    "minimum_behavior_models": 2,
    "causal_finite_rate_min": 0.995,
    "polarity_selected_swap_drop_min": 0.10,
    "polarity_selected_swap_positive_fraction_min": 0.60,
    "polarity_selected_minus_control_min": 0.05,
    "polarity_selected_to_control_ratio_min": 1.25,
    "polarity_selected_zero_drop_min": 0.05,
    "polarity_zero_to_control_ratio_min": 1.10,
    "polarity_each_path_swap_drop_min": 0.03,
    "polarity_to_surface_swap_ratio_min": 1.15,
    "minimum_causal_models": 2,
}

write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest
tokenizer_for = source.tokenizer_for
offset_token_spans = source.offset_token_spans


def mark(
    text: str,
    value: str,
    start: int = 0,
) -> tuple[int, int, str]:
    position = text.find(value, start)
    if position < 0:
        raise RuntimeError(f"missing marked text: {value!r}")
    return position, position + len(value), value


def surface_tail(
    task: str,
    template_index: int,
) -> tuple[str, str, str]:
    if task == "first":
        cue = "the first listed endpoint"
    elif task == "second":
        cue = "the second listed endpoint"
    else:
        raise ValueError(f"unknown surface task: {task}")
    if template_index == 2:
        operator = "Write"
        tail = (
            f"Your criterion is {cue}. {operator} exactly the matching "
            "endpoint."
        )
    elif template_index == 3:
        operator = "Choose"
        tail = (
            f"{operator} according to this request: {cue}. State one name "
            "and nothing else."
        )
    else:
        raise ValueError("Phase1076 uses confirmation templates only")
    return tail, cue, operator


def source_factor_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["path"],
        row["layout"],
        int(row["template_index"]),
        int(row["replicate"]),
        int(row["orientation"]),
        int(row["lexical_branch"]),
    )


def encode_surface_case(
    tokenizer,
    model_name: str,
    source_row: dict[str, Any],
    task: str,
    semantic_case_index: int,
) -> dict[str, Any]:
    (
        raw_source,
        raw_spans,
        semantic_raw_spans,
        _classes,
        metadata,
    ) = source.render_prompt(
        RELATION,
        "max",
        str(source_row["path"]),
        str(source_row["layout"]),
        list(source_row["cell_names"]),
        int(source_row["orientation"]),
        int(source_row["lexical_branch"]),
        int(source_row["template_index"]),
    )
    branch_end = raw_spans["branch_probe"][1]
    prefix = raw_source[:branch_end]
    tail, cue, operator = surface_tail(
        task, int(source_row["template_index"])
    )
    raw_prompt = f"{prefix}. {tail}"
    raw_spans = {
        key: value
        for key, value in raw_spans.items()
        if key in source.CAPTURE_ROLES
        and key not in {
            "task_cue",
            "operator",
            "query",
            "answer_boundary",
        }
    }
    raw_spans["task_cue"] = mark(
        raw_prompt, cue, branch_end
    )
    raw_spans["operator"] = mark(
        raw_prompt, operator, branch_end
    )
    query_start = min(
        raw_spans["operator"][0], raw_spans["task_cue"][0]
    )
    query_text = raw_prompt[query_start:].rstrip()
    raw_spans["query"] = (
        query_start,
        query_start + len(query_text),
        query_text,
    )
    rendered = behavior.render_native(
        tokenizer, model_name, raw_prompt, with_system=False
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
    endpoint_a = metadata["endpoint_a"]
    endpoint_b = metadata["endpoint_b"]
    expected = endpoint_a if task == "first" else endpoint_b
    expected_class = "b0" if task == "first" else "b1"
    classes = {"b0": [endpoint_a], "b1": [endpoint_b]}
    candidate_token_ids = {
        class_name: [
            behavior.continuation_ids(
                tokenizer, rendered, " ", label
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
    factor_id = ".".join([
        str(source_row["path"]),
        str(source_row["layout"]),
        f"t{source_row['template_index']}",
        f"r{source_row['replicate']}",
        f"o{source_row['orientation']}",
        f"l{source_row['lexical_branch']}",
    ])
    return {
        "schema_version": "phase1076_causal_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": semantic_case_index,
        "record_id": (
            f"phase1076.{model_name}.surface.{factor_id}.{task}"
        ),
        "pair_id": f"surface.{factor_id}",
        "factor_id": factor_id,
        "contrast": "surface",
        "relation": RELATION,
        "task": task,
        "path": source_row["path"],
        "layout": source_row["layout"],
        "template_index": source_row["template_index"],
        "replicate": source_row["replicate"],
        "orientation": source_row["orientation"],
        "lexical_branch": source_row["lexical_branch"],
        "cell_names": list(source_row["cell_names"]),
        "semantic_names": copy.deepcopy(
            source_row["semantic_names"]
        ),
        "expected_answer": expected,
        "expected_class": expected_class,
        "acceptable_labels": classes[expected_class],
        "candidate_labels": classes,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": candidate_first_token_ids,
        "facts_text": source_row["facts_text"],
        "task_cue_text": cue,
        "query_text": query_text,
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
        "continuation_prefix": " ",
    }


def encode_polarity_case(
    source_row: dict[str, Any],
    semantic_case_index: int,
) -> dict[str, Any]:
    row = copy.deepcopy(source_row)
    factor_id = ".".join([
        str(row["path"]),
        str(row["layout"]),
        f"t{row['template_index']}",
        f"r{row['replicate']}",
        f"o{row['orientation']}",
        f"l{row['lexical_branch']}",
    ])
    row.update({
        "schema_version": "phase1076_causal_case.v1",
        "phase": PHASE,
        "semantic_case_index": semantic_case_index,
        "record_id": (
            f"phase1076.{row['model']}.polarity."
            f"{factor_id}.{row['task']}"
        ),
        "pair_id": f"polarity.{factor_id}",
        "factor_id": factor_id,
        "contrast": "polarity",
    })
    return row


def discovery_route_matrix(model: str) -> tuple[np.ndarray, list[str]]:
    with np.load(
        SOURCE_ROOT / "internal" / model / "routing_aggregates.npz",
        allow_pickle=False,
    ) as data:
        relations = [str(value) for value in data["relations"]]
        splits = [str(value) for value in data["splits"]]
        destinations = [
            str(value) for value in data["destinations"]
        ]
        source_pairs = [
            str(value) for value in data["source_pairs"]
        ]
        metrics = [str(value) for value in data["metrics"]]
        conditionings = [
            str(value) for value in data["conditionings"]
        ]
        r = relations.index(RELATION)
        s = splits.index("discovery")
        d = destinations.index("answer_boundary")
        p = source_pairs.index("fact")
        m = metrics.index("attention_mass")
        c = conditionings.index("all")
        sums = data["sums"][r, s, :, :, :, :, d, p, m, c]
        counts = data["counts"][r, s, :, :, :, :, d, p, m, c]
    total_sums = sums.sum(axis=(0, 1))
    total_counts = counts.sum(axis=(0, 1))
    means = np.divide(
        total_sums,
        total_counts,
        out=np.full_like(total_sums, np.nan, dtype=np.float64),
        where=total_counts > 0,
    )
    return means, relations


def freeze_head_sets(model: str) -> dict[str, Any]:
    routing_rows = read_jsonl(
        SOURCE_ROOT
        / "analysis"
        / "routing_candidate_confirmation.jsonl"
    )
    selected_row = next(
        row
        for row in routing_rows
        if row["model"] == model and row["relation"] == RELATION
    )
    selected = [
        {
            "depth": int(row["depth"]),
            "head": int(row["head"]),
        }
        for row in selected_row["selected_heads"]
    ]
    selected_set = {
        (row["depth"], row["head"]) for row in selected
    }
    discovery, _ = discovery_route_matrix(model)
    used_controls: set[tuple[int, int]] = set()
    controls = []
    for event in selected:
        depth = event["depth"]
        candidates = []
        for head in range(discovery.shape[1]):
            key = (depth, head)
            if key in selected_set or key in used_controls:
                continue
            value = float(discovery[depth - 1, head])
            if not np.isfinite(value):
                continue
            candidates.append((abs(value), value, head))
        if not candidates:
            raise RuntimeError(
                f"no matched control head at {model} depth {depth}"
            )
        _, discovery_value, head = min(candidates)
        used_controls.add((depth, head))
        controls.append({
            "depth": depth,
            "head": int(head),
            "discovery_route_mean": discovery_value,
        })
    return {
        "selected": selected,
        "matched_controls": controls,
        "selection_source": (
            "Phase1075 discovery fact-route ranking at answer_boundary"
        ),
        "control_source": (
            "same-depth nonselected head with minimum absolute "
            "Phase1075 discovery route mean"
        ),
    }


def audit_model(
    model: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_factor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_pair[str(row["pair_id"])].append(row)
        by_factor[str(row["factor_id"])].append(row)
    exact_pair_prefix = True
    opposite_answers = True
    for rows in by_pair.values():
        expected_tasks = set(
            TASKS_BY_CONTRAST[rows[0]["contrast"]]
        )
        exact_pair_prefix &= (
            len(rows) == 2
            and {row["task"] for row in rows} == expected_tasks
        )
        prefixes = []
        for row in rows:
            end = int(row["role_positions"]["branch_probe"]) + 1
            prefixes.append(tuple(row["input_ids"][:end]))
        exact_pair_prefix &= len(set(prefixes)) == 1
        opposite_answers &= len({
            row["expected_answer"] for row in rows
        }) == 2
    exact_four_way_prefix = all(
        len(rows) == 4
        and len({
            tuple(row["input_ids"][
                :int(row["role_positions"]["branch_probe"]) + 1
            ])
            for row in rows
        }) == 1
        for rows in by_factor.values()
    )
    expected_cases = (
        len(CONTRASTS)
        * 2
        * len(PATHS)
        * len(LAYOUTS)
        * len(TEMPLATES)
        * len(REPLICATES)
        * len(ORIENTATIONS)
        * len(LEXICAL_BRANCHES)
    )
    balanced = Counter(
        (row["contrast"], row["task"], row["path"], row["layout"])
        for row in cases
    )
    checks = {
        "case_count": len(cases) == expected_cases,
        "pair_count": len(by_pair) == expected_cases // 2,
        "factor_count": len(by_factor) == expected_cases // 4,
        "exact_pair_prefix": exact_pair_prefix,
        "exact_four_way_prefix": exact_four_way_prefix,
        "opposite_answers": opposite_answers,
        "balanced": all(
            balanced[(contrast, task, path, layout)] == 16
            for contrast in CONTRASTS
            for task in TASKS_BY_CONTRAST[contrast]
            for path in PATHS
            for layout in LAYOUTS
        ),
        "candidate_single_token": all(
            len(tokenization) == 1
            for row in cases
            for tokenizations in row["candidate_token_ids"].values()
            for tokenization in tokenizations
        ),
        "candidate_disjoint": all(
            set(row["candidate_first_token_ids"]["b0"]).isdisjoint(
                row["candidate_first_token_ids"]["b1"]
            )
            for row in cases
        ),
        "answer_boundary_valid": all(
            0 <= int(row["role_positions"]["answer_boundary"])
            < len(row["input_ids"])
            for row in cases
        ),
        "confirmation_templates_only": all(
            int(row["template_index"]) in TEMPLATES
            and int(row["replicate"]) in REPLICATES
            for row in cases
        ),
    }
    return {
        "schema_version": "phase1076_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model,
        "case_count": len(cases),
        "pair_count": len(by_pair),
        "factor_count": len(by_factor),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_audit = read_json(
        SOURCE_ROOT / "analysis" / "integrity_audit.json"
    )
    source_next = read_json(
        SOURCE_ROOT / "analysis" / "automatic_next.json"
    )
    if (
        not source_audit["all_integrity_checks_passed"]
        or not source_next["should_continue_automatically"]
        or source_next["route"] != "freeze_targeted_causal_validation"
        or source_next["repeated_internal_relations"] != [RELATION]
    ):
        raise RuntimeError("Phase1075 source authorization drift")

    head_sets = {
        model: freeze_head_sets(model) for model in MODELS
    }
    model_audits = {}
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        source_rows = read_jsonl(
            SOURCE_ROOT / "protocol" / f"cases.{model}.jsonl"
        )
        source_by_key = {
            source_factor_key(row): row
            for row in source_rows
            if row["relation"] == RELATION
            and row["task"] == "max"
            and int(row["template_index"]) in TEMPLATES
            and int(row["replicate"]) in REPLICATES
        }
        cases = []
        semantic_case_index = 0
        for key in sorted(source_by_key):
            source_max = source_by_key[key]
            matching = [
                row
                for row in source_rows
                if source_factor_key(row) == key
                and row["relation"] == RELATION
            ]
            by_task = {row["task"]: row for row in matching}
            for task in TASKS_BY_CONTRAST["polarity"]:
                cases.append(encode_polarity_case(
                    by_task[task], semantic_case_index
                ))
                semantic_case_index += 1
            for task in TASKS_BY_CONTRAST["surface"]:
                cases.append(encode_surface_case(
                    tokenizer,
                    model,
                    source_max,
                    task,
                    semantic_case_index,
                ))
                semantic_case_index += 1
        audit = audit_model(model, cases)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1076 audit failed for {model}: {audit}"
            )
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model}.jsonl",
            cases,
        )
        write_json(
            OUT_ROOT / "protocol" / f"audit.{model}.json",
            audit,
        )
        model_audits[model] = audit

    payload = {
        "schema_version": "phase1076_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relation": RELATION,
        "contrasts": list(CONTRASTS),
        "tasks_by_contrast": {
            key: list(value)
            for key, value in TASKS_BY_CONTRAST.items()
        },
        "templates": list(TEMPLATES),
        "replicates": list(REPLICATES),
        "paths": list(PATHS),
        "layouts": list(LAYOUTS),
        "orientations": list(ORIENTATIONS),
        "lexical_branches": list(LEXICAL_BRANCHES),
        "interventions": list(INTERVENTIONS),
        "case_count_per_model": 384,
        "pair_count_per_model": 192,
        "factor_count_per_model": 96,
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "primary_analysis_population": (
            "all preregistered finite cases; behavior-conditioned pairs "
            "are reported only as a sensitivity analysis"
        ),
        "head_sets": head_sets,
        "gates": dict(GATES),
        "source_phase1075_digest": source_prereg["protocol_digest"],
        "source_phase1075_internal_digest": source_next[
            "internal_preregistration_digest"
        ],
        "causal_estimands": {
            "expected_margin": (
                "M = logit(expected_endpoint) - logit(other_endpoint)"
            ),
            "swap_drop": (
                "D_swap = M_baseline - M_opposite_task_head_swap"
            ),
            "zero_drop": (
                "D_zero = M_baseline - M_selected_head_zero"
            ),
            "coalition_specificity": (
                "D_selected - D_same_depth_control"
            ),
            "task_specificity": (
                "D_selected,polarity / "
                "max(abs(D_selected,surface), epsilon)"
            ),
        },
        "intervention_definition": {
            "physical_site": (
                "input to self_attn.o_proj at the answer-boundary token"
            ),
            "swap": (
                "replace only frozen head slices with the opposite task's "
                "baseline slice from the same facts, names, and world"
            ),
            "zero": "set only frozen head slices to zero",
            "matched_control": (
                "same-depth low-response heads frozen before this phase"
            ),
        },
        "measurement_order": [
            "freeze cases, behavior gates, selected heads, and same-depth controls",
            "verify four tasks share an exact pre-query prefix",
            "run behavior and natural-generation gates",
            "authorize intervention only if both models pass",
            "capture baseline answer-boundary pre-o_proj head slices",
            "swap opposite-task slices within each contrast",
            "repeat with same-depth control heads",
            "zero selected and control coalitions",
            "evaluate all effects on held-out templates and names only",
        ],
        "interpretation_limits": [
            "A swap effect proves local causal influence, not a complete mechanism.",
            "The selected coalition can carry answer state rather than compute the relation.",
            "Surface controls test task specificity but cannot exclude every shared output function.",
            "Zeroing is out-of-distribution and is interpreted only relative to matched controls.",
            "Cross-model agreement is functional, not head homology.",
        ],
        "automatic_next": (
            "Do not automatically broaden to a new language family. "
            "Phase1076 completes the authorized Phase1075 continuation; "
            "any later phase requires a newly frozen independent family."
        ),
        "model_audits": model_audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json", payload
    )
    write_json(
        OUT_ROOT / "protocol" / "audit.json",
        {
            "schema_version": "phase1076_protocol_audit.v1",
            "phase": PHASE,
            "protocol_digest": payload["protocol_digest"],
            "model_audits": model_audits,
            "all_checks_passed": all(
                audit["all_checks_passed"]
                for audit in model_audits.values()
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
