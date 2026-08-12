#!/usr/bin/env python3
"""Freeze held-out exact-prompt behavior calibration for Phase1073."""

from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1073_late_query_protocol as formal


PHASE = formal.PHASE
PROTOCOL_REVISION = 1
MODELS = formal.MODELS
PRECISION = formal.PRECISION
QUANTIZATION = formal.QUANTIZATION
RELATION_NAMES = formal.RELATION_NAMES
QUERY_TYPES = formal.QUERY_TYPES
PATH_NAMES = formal.PATH_NAMES
PROMPT_STYLES = (0,)
STYLE_LABELS = {0: "frozen_exact_late_query_prompt"}
TEMPLATES = (0, 2)
NATURAL_GENERATION_STEPS = formal.NATURAL_GENERATION_STEPS
CALIBRATION_ROOT = formal.CALIBRATION_ROOT
SOURCE_CALIBRATION_ROOT = formal.SOURCE_CALIBRATION_ROOT

write_json = formal.write_json
write_jsonl = formal.write_jsonl
read_json = formal.read_json
read_jsonl = formal.read_jsonl
digest = formal.digest
tokenizer_for = formal.tokenizer_for


def calibration_names_for_case(
    names: tuple[str, ...],
    key: str,
) -> list[str]:
    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"phase1073-calibration|{key}|{name}".encode("utf-8")
        ).hexdigest(),
    )
    result = ranked[:6]
    if len(set(result)) != 6:
        raise RuntimeError("calibration names are not unique")
    return result


def selected_answer_lexical(key: str) -> tuple[int, int]:
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return value % 2, (value // 2) % 2


def audit_cases(
    model_name: str,
    cases: list[dict[str, Any]],
    calibration_names: set[str],
) -> dict[str, Any]:
    expected = (
        len(RELATION_NAMES)
        * len(QUERY_TYPES)
        * len(TEMPLATES)
        * len(PATH_NAMES)
    )
    counts = Counter(
        (
            row["relation"],
            row["query_type"],
            int(row["template_index"]),
            row["path_name"],
        )
        for row in cases
    )
    paired: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        paired[(
            row["base_relation"],
            row["key_alignment"],
            row["evidence_order"],
            row["query_type"],
            int(row["template_index"]),
            row["path_name"],
            row["state"],
        )].append(row)

    exact_late_prefix = True
    paired_semantics = True
    for rows in paired.values():
        if len(rows) != (
            len(formal.TASK_FAMILIES) * len(formal.PROMPT_BRANCHES)
        ):
            exact_late_prefix = False
            paired_semantics = False
            continue
        prefixes = []
        for row in rows:
            end = int(row["role_positions"]["branch_probe"]) + 1
            prefixes.append(tuple(row["input_ids"][:end]))
        exact_late_prefix = exact_late_prefix and len(set(prefixes)) == 1
        paired_semantics = (
            paired_semantics
            and len({row["facts_text"] for row in rows}) == 1
            and len({row["answer_key_text"] for row in rows}) == 1
            and len({row["query_text"] for row in rows}) == 1
            and len({tuple(row["cell_names"]) for row in rows}) == 1
        )

    checks = {
        "case_count": len(cases) == expected,
        "balanced_condition_query_template_path": all(
            counts[(condition, query, template, path)] == 1
            for condition in RELATION_NAMES
            for query in QUERY_TYPES
            for template in TEMPLATES
            for path in PATH_NAMES.values()
        ),
        "only_reserved_calibration_names": all(
            set(row["cell_names"]) <= calibration_names for row in cases
        ),
        "six_unique_names_per_case": all(
            len(set(row["cell_names"])) == 6 for row in cases
        ),
        "candidate_continuations_single_token": all(
            len(values) == 1
            for row in cases
            for class_values in row["candidate_token_ids"].values()
            for values in class_values
        ),
        "candidate_first_tokens_disjoint": all(
            set(row["candidate_first_token_ids"]["b0"]).isdisjoint(
                set(row["candidate_first_token_ids"]["b1"])
            )
            for row in cases
        ),
        "exact_late_branch_prefix": exact_late_prefix,
        "paired_prefix_semantics": paired_semantics,
        "expected_answer_valid": all(
            row["expected_answer"]
            == (
                row["chain_answer"]
                if row["task_family"] == "transitive"
                else row["key_answer"]
            )
            for row in cases
        ),
        "key_alignment_valid": all(
            (
                row["chain_answer"] == row["key_answer"]
                if row["key_alignment"] == "congruent"
                else row["chain_answer"] != row["key_answer"]
            )
            for row in cases
        ),
        "prompt_skeleton_hash_present": all(
            len(str(row["prompt_skeleton_sha256"])) == 64 for row in cases
        ),
        "semantic_indices_contiguous": sorted(
            int(row["semantic_case_index"]) for row in cases
        )
        == list(range(len(cases))),
    }
    return {
        "schema_version": "phase1073_calibration_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "paired_prefix_group_count": len(paired),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    source_calibration = read_json(
        SOURCE_CALIBRATION_ROOT / "protocol" / "preregistration.json"
    )
    calibration_names = tuple(source_calibration["calibration_names"])
    formal_names = set(source_calibration["reserved_formal_names"])
    if set(calibration_names) & formal_names:
        raise RuntimeError("calibration/formal names overlap")
    if len(calibration_names) < 6:
        raise RuntimeError("not enough held-out calibration names")

    model_audits = {}
    case_counts = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases: list[dict[str, Any]] = []
        semantic_index = 0
        name_cache: dict[tuple[Any, ...], list[str]] = {}
        state_cache: dict[tuple[Any, ...], str] = {}
        for condition in RELATION_NAMES:
            parsed = formal.parse_condition(condition)
            for query_type in QUERY_TYPES:
                for template_index in TEMPLATES:
                    for path_index, (anchor, switch) in enumerate(PATH_NAMES):
                        shared_key = (
                            parsed["base_relation"],
                            parsed["key_alignment"],
                            parsed["evidence_order"],
                            query_type,
                            template_index,
                            anchor,
                            switch,
                        )
                        selection_key = "|".join(map(str, shared_key))
                        if shared_key not in name_cache:
                            name_cache[shared_key] = calibration_names_for_case(
                                calibration_names, selection_key
                            )
                            answer, lexical = selected_answer_lexical(
                                selection_key
                            )
                            state_cache[shared_key] = (
                                f"a{anchor}_b{switch}_y{answer}_l{lexical}"
                            )
                        row = formal.encode_case(
                            tokenizer,
                            model_name,
                            name_cache[shared_key],
                            condition,
                            query_type,
                            template_index,
                            path_index % len(formal.REPLICATES),
                            state_cache[shared_key],
                            semantic_index,
                            "calibration",
                        )
                        row["schema_version"] = (
                            "phase1073_calibration_case.v1"
                        )
                        row["prompt_style"] = 0
                        cases.append(row)
                        semantic_index += 1
        audit = audit_cases(model_name, cases, set(calibration_names))
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1073 calibration audit failed: {model_name}: {audit}"
            )
        write_jsonl(
            CALIBRATION_ROOT / "protocol" / f"cases.{model_name}.jsonl",
            cases,
        )
        write_json(
            CALIBRATION_ROOT / "protocol" / f"audit.{model_name}.json",
            audit,
        )
        model_audits[model_name] = audit
        case_counts[model_name] = len(cases)

    payload = {
        "schema_version": "phase1073_calibration_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "conditions": list(RELATION_NAMES),
        "query_types": list(QUERY_TYPES),
        "templates": list(TEMPLATES),
        "paths": list(PATH_NAMES.values()),
        "prompt_styles": {"0": STYLE_LABELS[0]},
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "case_counts": case_counts,
        "calibration_names": list(calibration_names),
        "reserved_formal_names": sorted(formal_names),
        "selection_policy": (
            "The exact late-query prompt family, both task branches, both "
            "cue styles, both key alignments, and both evidence orders are "
            "frozen before any Phase1073 hidden-state measurement."
        ),
        "exact_transfer_definition": (
            "Calibration uses templates 0 and 2 from the unchanged formal "
            "renderer. Names are held out, while prompt skeletons transfer "
            "exactly into the formal protocol."
        ),
        "gates": {
            key: value
            for key, value in formal.GATES.items()
            if key.startswith("calibration_")
        },
        "model_audits": model_audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        CALIBRATION_ROOT / "protocol" / "preregistration.json", payload
    )
    write_json(
        CALIBRATION_ROOT / "protocol" / "audit.json",
        {
            "schema_version": "phase1073_calibration_protocol_audit.v1",
            "phase": PHASE,
            "protocol_digest": payload["protocol_digest"],
            "model_audits": model_audits,
            "all_checks_passed": all(
                row["all_checks_passed"] for row in model_audits.values()
            ),
        },
    )
    return payload


def main() -> None:
    payload = build_protocol()
    print(
        "Phase1073 calibration frozen: "
        f"{payload['protocol_digest']} cases={payload['case_counts']}"
    )


if __name__ == "__main__":
    main()
