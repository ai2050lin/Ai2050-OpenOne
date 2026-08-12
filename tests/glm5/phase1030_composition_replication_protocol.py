#!/usr/bin/env python3
"""Freeze the Phase1030 two-template composition replication."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import render_chat, tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1024_lexical_semantic_protocol as phase1024
import phase1026_binding_replication_protocol as phase1026
import phase1027_binding_transport_protocol as phase1027
import phase1028_role_depth_causal_map_protocol as phase1028
import phase1029_multibinding_competition_protocol as phase1029


PHASE = 1030
PROTOCOL_REVISION = 1
MODELS = phase1024.MODELS
ROLES = phase1029.ROLES
WORLD_CODES = phase1029.WORLD_CODES
DONOR_OFFSETS = (1, 2, 4, 5)
SELECTED_DEPTHS = {
    "qwen3": {"source": 9, "query": 25, "pre_output": 31, "readout": 35},
    "glm4": {"source": 4, "query": 4, "pre_output": 18, "readout": 19},
    "deepseek7b": {
        "source": 4,
        "query": 10,
        "pre_output": 25,
        "readout": 27,
    },
}
CONDITIONS = (
    "selected_source_b",
    "unselected_source_b",
    "source_pair_b",
    "query_q",
    "query_bq",
    "source_pair_plus_query_q",
    "source_pair_plus_query_bq",
    "full_bq",
    "source_pair_scrambled",
    "source_pair_wrong_position",
    "query_q_wrong_position",
    "query_bq_wrong_position",
    "pre_output_b",
    "pre_output_bq",
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1030_composition_replication"
)


CONCEPTS = (
    ("guava", "fruit"),
    ("lynx", "animal"),
    ("canoe", "vehicle"),
    ("baker", "profession"),
    ("gallery", "place"),
    ("suitcase", "object"),
    ("silver", "color"),
    ("chin", "body_part"),
)

NONCE_PAIRS = (
    ("abren", "belvik"),
    ("coran", "dars"),
    ("emlor", "favin"),
    ("gesk", "huren"),
    ("ivor", "jalen"),
    ("kesp", "lurin"),
    ("moxel", "naret"),
    ("oskin", "pravel"),
)

TEMPLATES = (
    (
        'Reference ledger: tag "{nonce_a}" is linked to {concept_a}, '
        'and tag "{nonce_b}" is linked to {concept_b}. Identify the '
        'broad class for tag "{query_nonce}":'
    ),
    (
        'Remember these two assignments. "{nonce_a}" denotes '
        '{concept_a}. "{nonce_b}" denotes {concept_b}. Which general '
        'category contains the item denoted by "{query_nonce}"? Answer:'
    ),
)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fragment(
    prompt: str,
    value: str,
    *,
    occurrence: str,
) -> tuple[int, int, str]:
    if occurrence == "first":
        start = prompt.find(value)
    elif occurrence == "last":
        start = prompt.rfind(value)
    else:
        raise ValueError(occurrence)
    if start < 0:
        raise RuntimeError(f"missing fragment {value!r}")
    return start, start + len(value), value


def build_units_and_cases() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    units: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    lookup: dict[tuple[int, int, int, int], int] = {}
    for template_index, template in enumerate(TEMPLATES):
        for surface_index, (nonce_a, nonce_b) in enumerate(NONCE_PAIRS):
            for target_index in range(8):
                for donor_offset in DONOR_OFFSETS:
                    donor_index = (target_index + donor_offset) % 8
                    q0_slot = (
                        "a"
                        if (
                            template_index + surface_index + target_index
                        ) % 2 == 0
                        else "b"
                    )
                    unit_index = len(units)
                    lookup[
                        (
                            template_index,
                            surface_index,
                            target_index,
                            donor_index,
                        )
                    ] = unit_index
                    world_case_indices: dict[str, int] = {}
                    for binding_flip, query_flip in (
                        (0, 0),
                        (1, 0),
                        (0, 1),
                        (1, 1),
                    ):
                        selected_concept = (
                            target_index if binding_flip == 0 else donor_index
                        )
                        other_concept = (
                            donor_index if binding_flip == 0 else target_index
                        )
                        if q0_slot == "a":
                            concept_a_index = selected_concept
                            concept_b_index = other_concept
                            query_slot = "a" if query_flip == 0 else "b"
                        else:
                            concept_a_index = other_concept
                            concept_b_index = selected_concept
                            query_slot = "b" if query_flip == 0 else "a"
                        concept_a = CONCEPTS[concept_a_index][0]
                        concept_b = CONCEPTS[concept_b_index][0]
                        query_nonce = (
                            nonce_a if query_slot == "a" else nonce_b
                        )
                        world = f"{binding_flip}{query_flip}"
                        expected_index = (
                            target_index
                            if binding_flip == query_flip
                            else donor_index
                        )
                        prompt = template.format(
                            nonce_a=nonce_a,
                            nonce_b=nonce_b,
                            concept_a=concept_a,
                            concept_b=concept_b,
                            query_nonce=query_nonce,
                        )
                        case_index = len(cases)
                        world_case_indices[world] = case_index
                        cases.append({
                            "schema_version": "phase1030_common_case.v1",
                            "phase": PHASE,
                            "case_index": case_index,
                            "case_key": (
                                f"t{template_index}.u{unit_index}.{world}"
                            ),
                            "unit_index": unit_index,
                            "template_index": template_index,
                            "surface_index": surface_index,
                            "surface_key": (
                                template_index * len(NONCE_PAIRS)
                                + surface_index
                            ),
                            "nonce_a": nonce_a,
                            "nonce_b": nonce_b,
                            "target_index": target_index,
                            "donor_index": donor_index,
                            "binding_flip": binding_flip,
                            "query_flip": query_flip,
                            "world": world,
                            "q0_slot": q0_slot,
                            "query_slot": query_slot,
                            "concept_a_index": concept_a_index,
                            "concept_b_index": concept_b_index,
                            "concept_a": concept_a,
                            "concept_b": concept_b,
                            "expected_index": expected_index,
                            "expected_concept": CONCEPTS[
                                expected_index
                            ][0],
                            "prompt": prompt,
                            "role_fragments": {
                                "definition_nonce_a_end": fragment(
                                    prompt, nonce_a, occurrence="first"
                                ),
                                "concept_a_end": fragment(
                                    prompt, concept_a, occurrence="first"
                                ),
                                "definition_nonce_b_end": fragment(
                                    prompt, nonce_b, occurrence="first"
                                ),
                                "concept_b_end": fragment(
                                    prompt, concept_b, occurrence="first"
                                ),
                                "query_nonce_end": fragment(
                                    prompt, query_nonce, occurrence="last"
                                ),
                            },
                        })
                    units.append({
                        "schema_version": "phase1030_unit.v1",
                        "phase": PHASE,
                        "unit_index": unit_index,
                        "unit_key": (
                            f"t{template_index}.s{surface_index}."
                            f"t{target_index}.d{donor_index}"
                        ),
                        "template_index": template_index,
                        "surface_index": surface_index,
                        "surface_key": (
                            template_index * len(NONCE_PAIRS)
                            + surface_index
                        ),
                        "target_index": target_index,
                        "donor_index": donor_index,
                        "donor_offset": donor_offset,
                        "q0_slot": q0_slot,
                        "selected_concept_role": (
                            "concept_a_end"
                            if q0_slot == "a"
                            else "concept_b_end"
                        ),
                        "unselected_concept_role": (
                            "concept_b_end"
                            if q0_slot == "a"
                            else "concept_a_end"
                        ),
                        "selected_definition_role": (
                            "definition_nonce_a_end"
                            if q0_slot == "a"
                            else "definition_nonce_b_end"
                        ),
                        "world_case_indices": world_case_indices,
                    })
    for unit in units:
        current_offset = int(unit["donor_offset"])
        next_offset = DONOR_OFFSETS[
            (DONOR_OFFSETS.index(current_offset) + 1)
            % len(DONOR_OFFSETS)
        ]
        scrambled_donor = (
            int(unit["target_index"]) + next_offset
        ) % 8
        unit["scrambled_donor_index"] = scrambled_donor
        unit["scrambled_unit_index"] = lookup[
            (
                int(unit["template_index"]),
                int(unit["surface_index"]),
                int(unit["target_index"]),
                scrambled_donor,
            )
        ]
    return units, cases


def model_case(
    tokenizer,
    model_name: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    rendered = render_chat(tokenizer, model_name, row["prompt"])
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    positions = offset_token_spans(
        tokenizer,
        rendered,
        row["prompt"],
        row["role_fragments"],
    )
    result = dict(row)
    result.pop("role_fragments", None)
    result.update({
        "schema_version": "phase1030_model_case.v1",
        "model": model_name,
        "record_id": f"{model_name}.{row['case_key']}",
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            role: int(positions[role][1])
            for role in ROLES
            if role != "pre_output"
        } | {
            "pre_output": len(input_ids) - 1,
        },
        "prompt_token_count": len(input_ids),
    })
    return result


def previous_words() -> tuple[set[str], set[str]]:
    concepts: set[str] = set()
    nonces: set[str] = set()
    for module in (
        phase1024,
        phase1026,
        phase1027,
        phase1028,
        phase1029,
    ):
        if hasattr(module, "CONCEPTS"):
            source = module.CONCEPTS
            values = source.values() if isinstance(source, dict) else (source,)
            for group in values:
                concepts.update(word for word, _ in group)
        if hasattr(module, "NONCES"):
            for group in module.NONCES.values():
                nonces.update(group)
        if hasattr(module, "NONCE_PAIRS"):
            source = module.NONCE_PAIRS
            values = source.values() if isinstance(source, dict) else (source,)
            for group in values:
                nonces.update(word for pair in group for word in pair)
    return concepts, nonces


def main() -> None:
    units, cases = build_units_and_cases()
    prior_concepts, prior_nonces = previous_words()
    new_concepts = {word for word, _ in CONCEPTS}
    new_nonces = {word for pair in NONCE_PAIRS for word in pair}
    template_units = Counter(
        int(row["template_index"]) for row in units
    )
    checks = {
        "unit_count": len(units) == 512,
        "case_count": len(cases) == 2048,
        "template_unit_balance": template_units == {0: 256, 1: 256},
        "four_worlds_per_unit": all(
            set(row["world_case_indices"]) == set(WORLD_CODES)
            for row in units
        ),
        "xor_answer_rule": all(
            int(row["expected_index"])
            == (
                int(row["target_index"])
                if int(row["binding_flip"]) == int(row["query_flip"])
                else int(row["donor_index"])
            )
            for row in cases
        ),
        "scrambled_units_distinct": all(
            int(row["scrambled_unit_index"]) != int(row["unit_index"])
            and int(row["scrambled_donor_index"])
            not in {int(row["target_index"]), int(row["donor_index"])}
            for row in units
        ),
        "new_concepts": not bool(new_concepts & prior_concepts),
        "new_nonces": not bool(new_nonces & prior_nonces),
    }
    common_audit = {
        "schema_version": "phase1030_common_audit.v1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    if not common_audit["all_checks_passed"]:
        raise RuntimeError(json.dumps(common_audit, ensure_ascii=False))

    prereg = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "precision": "fp16",
        "quantization": "none",
        "models": MODELS,
        "roles_in_causal_order": ROLES,
        "worlds": phase1029.read_json(
            phase1029.OUT_ROOT / "protocol" / "preregistration.json"
        )["worlds"],
        "selected_depths_frozen_from_phase1029": SELECTED_DEPTHS,
        "donor_offsets": DONOR_OFFSETS,
        "template_count": len(TEMPLATES),
        "unit_count": len(units),
        "conditions": CONDITIONS,
        "replication_gate": {
            "clean_expected_top1_minimum": 0.50,
            "selected_source_alternate_top1_minimum": 0.40,
            "source_pair_alternate_top1_minimum": 0.50,
            "query_q_alternate_top1_minimum": 0.25,
            "query_bq_base_top1_minimum": 0.50,
            "source_plus_query_bq_base_top1_minimum": 0.50,
            "full_bq_base_top1_minimum": 0.50,
            "selected_minus_unselected_alternate_top1_minimum": 0.10,
            "source_minus_scrambled_alternate_top1_minimum": 0.10,
            "source_minus_wrong_alternate_top1_minimum": 0.10,
            "query_q_minus_wrong_alternate_top1_minimum": 0.10,
            "source_query_bq_base_minus_single_base_top1_minimum": 0.10,
            "must_pass_each_template": True,
            "all_arrays_must_be_finite": True,
        },
        "interpretation_rule": (
            "Q-only query states and BQ query states are separate "
            "interventions. Replication requires source transport, "
            "query-Q transport, query-BQ same-answer transport, and "
            "source-plus-query causal cancellation within both templates."
        ),
        "claim_limit": (
            "replication of a residual-state composition candidate only; "
            "no minimal circuit, complete binding algorithm, general "
            "language law, brain homology, or optimality claim"
        ),
        "unit_digest": digest(units),
        "case_digest": digest(cases),
    }
    prereg["protocol_digest"] = digest(prereg)
    protocol_dir = OUT_ROOT / "protocol"
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.common.json", common_audit)
    write_jsonl(protocol_dir / "units.jsonl", units)
    write_jsonl(protocol_dir / "common_cases.jsonl", cases)

    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        rows = [model_case(tokenizer, model_name, row) for row in cases]
        model_checks = {
            "case_count": len(rows) == len(cases),
            "case_indices_dense": (
                [row["case_index"] for row in rows]
                == list(range(len(rows)))
            ),
            "roles_present": all(
                set(row["role_positions"]) == set(ROLES) for row in rows
            ),
            "positions_in_range": all(
                all(
                    0 <= position < len(row["input_ids"])
                    for position in row["role_positions"].values()
                )
                for row in rows
            ),
            "strict_causal_role_order": all(
                [
                    row["role_positions"][role] for role in ROLES
                ] == sorted(
                    row["role_positions"][role] for role in ROLES
                )
                and len({
                    row["role_positions"][role] for role in ROLES
                }) == len(ROLES)
                for row in rows
            ),
        }
        audit = {
            "model": model_name,
            "prompt_tokens": {
                "minimum": min(row["prompt_token_count"] for row in rows),
                "maximum": max(row["prompt_token_count"] for row in rows),
            },
            "checks": model_checks,
            "all_checks_passed": all(model_checks.values()),
        }
        if not audit["all_checks_passed"]:
            raise RuntimeError(json.dumps(audit, ensure_ascii=False))
        write_jsonl(protocol_dir / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_dir / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        del tokenizer
    write_json(
        protocol_dir / "audit.models.json",
        {
            "models": model_audits,
            "all_checks_passed": all(
                row["all_checks_passed"]
                for row in model_audits.values()
            ),
        },
    )
    print(json.dumps({
        "protocol_digest": prereg["protocol_digest"],
        "unit_count": len(units),
        "case_count": len(cases),
        "template_units": dict(template_units),
        "selected_depths": SELECTED_DEPTHS,
        "audit": common_audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
