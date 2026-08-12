#!/usr/bin/env python3
"""Freeze the Phase1071 behavior-only prompt calibration protocol."""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1051_natural_behavior_protocol as behavior
import phase1070_process_answer_protocol as previous


PHASE = 1071
PROTOCOL_REVISION = 1
MODELS = previous.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
RELATION_NAMES = previous.RELATION_NAMES
QUERY_TYPES = previous.QUERY_TYPES
PATH_NAMES = previous.PATH_NAMES
ANSWER_BRANCHES = previous.ANSWER_BRANCHES
LEXICAL_BRANCHES = previous.LEXICAL_BRANCHES
PROMPT_STYLES = (0, 1, 2, 3)
ASSISTANT_PREFILL = "Answer:"
NATURAL_GENERATION_STEPS = 10
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1071_exposure_pattern_family"
)
CALIBRATION_ROOT = OUT_ROOT / "calibration"
SOURCE_ROOT = previous.OUT_ROOT

STYLE_LABELS = {
    0: "concise_evidence_first",
    1: "explicit_transitivity",
    2: "evidence_only_instruction",
    3: "read_all_facts",
}

# Frozen before any Phase1071 model forward pass. Selection is lexicographic,
# so no scalar weighting can be tuned after seeing model behavior.
SELECTION_RULE = {
    "model_candidate_accuracy_min": 0.80,
    "model_semantic_first_rate_min": 0.70,
    "relation_candidate_accuracy_min": 0.65,
    "path_semantic_first_rate_min": 0.55,
    "candidate_finite_rate_min": 0.995,
    "minimum_eligible_models": 2,
    "lexicographic_priority": [
        "eligible_model_count_desc",
        "worst_model_semantic_first_rate_desc",
        "worst_model_candidate_accuracy_desc",
        "macro_semantic_first_rate_desc",
        "macro_candidate_accuracy_desc",
        "prompt_style_index_asc",
    ],
}

write_json = previous.write_json
write_jsonl = previous.write_jsonl
read_json = previous.read_json
read_jsonl = previous.read_jsonl
digest = previous.digest
tokenizer_for = previous.tokenizer_for


def calibration_name_set(
    names: tuple[str, ...],
    relation: str,
    query_type: str,
    style: int,
) -> list[str]:
    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(
            (
                f"phase1071-calibration|{relation}|{query_type}|"
                f"{style}|{name}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:6]


def render_raw_prompt(
    relation: str,
    query_type: str,
    names: list[str],
    anchor: int,
    switch: int,
    answer: int,
    lexical: int,
    style: int,
) -> tuple[str, dict[str, list[str]], dict[str, Any]]:
    tagged, semantic_names, expected = previous.tagged_fact_items(
        relation,
        names,
        query_type,
        anchor,
        switch,
        answer,
        lexical,
        phrase_set=0,
    )
    facts, _ = previous.compose_facts(tagged, "forward")
    endpoint_a = semantic_names["endpoint_a"]
    endpoint_b = semantic_names["endpoint_b"]
    query_tail = previous.RELATIONS[relation][
        f"{query_type}_query"
    ][0]
    query = f"Between {endpoint_a} and {endpoint_b}, {query_tail}?"
    evidence_probe = "All evidence is now complete."

    if style == 0:
        raw_prompt = (
            f"Facts: {facts} {evidence_probe} Question: {query} "
            "Give exactly one person's name."
        )
    elif style == 1:
        raw_prompt = (
            "Use the ordering facts transitively when necessary. "
            f"Facts: {facts} {evidence_probe} Question: {query} "
            "Return one name only."
        )
    elif style == 2:
        raw_prompt = (
            f"Ordering evidence: {facts} {evidence_probe} "
            f"Using only this evidence, answer: {query} "
            "Write exactly one name."
        )
    elif style == 3:
        raw_prompt = (
            f"Read every fact before answering. {facts} "
            f"{evidence_probe} {query} Respond with one person's name "
            "and nothing else."
        )
    else:
        raise ValueError(f"unknown prompt style: {style}")

    classes = {"b0": [endpoint_a], "b1": [endpoint_b]}
    return raw_prompt, classes, {
        **semantic_names,
        "answer": expected,
        "query_text": query,
        "facts_text": facts,
        "evidence_probe": evidence_probe,
    }


def build_case(
    tokenizer,
    model_name: str,
    calibration_names: tuple[str, ...],
    relation: str,
    query_type: str,
    anchor: int,
    switch: int,
    answer: int,
    lexical: int,
    style: int,
    case_index: int,
) -> dict[str, Any]:
    names = calibration_name_set(
        calibration_names, relation, query_type, style
    )
    raw_prompt, classes, metadata = render_raw_prompt(
        relation,
        query_type,
        names,
        anchor,
        switch,
        answer,
        lexical,
        style,
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
        for value in tokenizer.encode(
            rendered, add_special_tokens=False
        )
    ]
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
    path_name = PATH_NAMES[(anchor, switch)]
    return {
        "schema_version": "phase1071_calibration_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": case_index,
        "record_id": f"{model_name}.calibration.{case_index:05d}",
        "relation": relation,
        "query_type": query_type,
        "path_name": path_name,
        "anchor_branch": anchor,
        "switch_branch": switch,
        "answer_branch": answer,
        "lexical_branch": lexical,
        "prompt_style": style,
        "prompt_style_label": STYLE_LABELS[style],
        "cell_names": names,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "candidate_labels": classes,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": candidate_first_token_ids,
        "expected_class": f"b{answer}",
        "acceptable_labels": classes[f"b{answer}"],
        "continuation_prefix": " ",
        "query_text": metadata["query_text"],
        "facts_text": metadata["facts_text"],
        "evidence_probe": metadata["evidence_probe"],
    }


def audit_cases(
    cases: list[dict[str, Any]],
    calibration_names: tuple[str, ...],
    mechanism_names: tuple[str, ...],
) -> dict[str, Any]:
    expected = (
        len(RELATION_NAMES)
        * len(QUERY_TYPES)
        * len(PATH_NAMES)
        * len(ANSWER_BRANCHES)
        * len(LEXICAL_BRANCHES)
        * len(PROMPT_STYLES)
    )
    counts = Counter(
        (
            row["prompt_style"],
            row["relation"],
            row["query_type"],
            row["path_name"],
        )
        for row in cases
    )
    checks = {
        "case_count": len(cases) == expected,
        "balanced_cells": all(
            counts[(style, relation, query, path)] == 4
            for style in PROMPT_STYLES
            for relation in RELATION_NAMES
            for query in QUERY_TYPES
            for path in PATH_NAMES.values()
        ),
        "candidate_continuations_single_token": all(
            len(values) == 1
            for row in cases
            for class_values in row["candidate_token_ids"].values()
            for values in class_values
        ),
        "candidate_classes_disjoint": all(
            set(row["candidate_first_token_ids"]["b0"]).isdisjoint(
                row["candidate_first_token_ids"]["b1"]
            )
            for row in cases
        ),
        "facts_before_query": all(
            row["raw_prompt"].find(row["facts_text"])
            < row["raw_prompt"].find(row["query_text"])
            for row in cases
        ),
        "evidence_probe_before_query": all(
            row["raw_prompt"].find(row["evidence_probe"])
            < row["raw_prompt"].find(row["query_text"])
            for row in cases
        ),
        "calibration_mechanism_names_disjoint": set(
            calibration_names
        ).isdisjoint(mechanism_names),
    }
    return {
        "schema_version": "phase1071_calibration_protocol_audit.v1",
        "phase": PHASE,
        "case_count": len(cases),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_next = read_json(
        SOURCE_ROOT / "analysis" / "automatic_next.json"
    )
    source_audit = read_json(
        SOURCE_ROOT / "analysis" / "integrity_audit.json"
    )
    if (
        bool(source_next["should_continue_automatically"])
        or source_next["route"] != "stop_at_process_answer_atlas"
        or not source_audit["all_integrity_checks_passed"]
    ):
        raise RuntimeError("Phase1070 source decision or audit drift")

    all_names = previous.single_token_names(
        source_prereg["cross_model_single_token_names"]
        + [
            name
            for name in read_json(
                previous.SOURCE_ROOT
                / "protocol"
                / "preregistration.json"
            )["cross_tokenizer_names"]
            if name
            not in source_prereg["cross_model_single_token_names"]
        ]
    )
    mechanism_names = tuple(all_names[:48])
    calibration_names = tuple(all_names[48:60])
    if len(calibration_names) != 12:
        raise RuntimeError("Phase1071 needs 12 held-out calibration names")

    model_audits = {}
    case_count = None
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = []
        case_index = 0
        for style in PROMPT_STYLES:
            for relation in RELATION_NAMES:
                for query_type in QUERY_TYPES:
                    for anchor, switch in PATH_NAMES:
                        for answer in ANSWER_BRANCHES:
                            for lexical in LEXICAL_BRANCHES:
                                cases.append(build_case(
                                    tokenizer,
                                    model_name,
                                    calibration_names,
                                    relation,
                                    query_type,
                                    anchor,
                                    switch,
                                    answer,
                                    lexical,
                                    style,
                                    case_index,
                                ))
                                case_index += 1
        audit = audit_cases(
            cases, calibration_names, mechanism_names
        )
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1071 calibration audit failed: {model_name}"
            )
        write_jsonl(
            CALIBRATION_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl",
            cases,
        )
        write_json(
            CALIBRATION_ROOT
            / "protocol"
            / f"audit.{model_name}.json",
            audit,
        )
        model_audits[model_name] = audit
        case_count = len(cases)

    payload = {
        "schema_version": "phase1071_calibration_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relations": list(RELATION_NAMES),
        "query_types": list(QUERY_TYPES),
        "path_names": list(PATH_NAMES.values()),
        "prompt_styles": {
            str(key): value for key, value in STYLE_LABELS.items()
        },
        "case_count_per_model": case_count,
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "assistant_prefill": ASSISTANT_PREFILL,
        "calibration_names": list(calibration_names),
        "reserved_mechanism_names": list(mechanism_names),
        "selection_rule": SELECTION_RULE,
        "source_phase1070_digest": source_prereg["protocol_digest"],
        "source_phase1070_decision": source_next,
        "user_override_reason": (
            "The user explicitly authorized a new Phase1071 repair study; "
            "the failed Phase1070 automatic gate is not reclassified."
        ),
        "interpretation_limits": [
            "This stage selects a common prompt protocol from behavior only; it reads no hidden state.",
            "Prompt selection is calibrated on names disjoint from the mechanism set.",
            "A selected prompt is an instrument choice, not evidence for a language mechanism.",
            "The lexicographic selection rule is frozen before model execution.",
        ],
        "model_audits": model_audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        CALIBRATION_ROOT / "protocol" / "preregistration.json",
        payload,
    )
    write_json(
        CALIBRATION_ROOT / "protocol" / "audit.json",
        {
            "schema_version": (
                "phase1071_calibration_protocol_audit.v1"
            ),
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
        f"Phase{PHASE} calibration protocol "
        f"{payload['protocol_digest']} "
        f"cases={payload['case_count_per_model']}/model"
    )


if __name__ == "__main__":
    main()
