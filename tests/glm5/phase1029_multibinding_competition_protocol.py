#!/usr/bin/env python3
"""Freeze the Phase1029 two-binding competition protocol."""

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


PHASE = 1029
PROTOCOL_REVISION = 1
MODELS = phase1024.MODELS
SPLITS = phase1024.SPLITS
ROLES = (
    "definition_nonce_a_end",
    "concept_a_end",
    "definition_nonce_b_end",
    "concept_b_end",
    "query_nonce_end",
    "pre_output",
)
PATCH_DEPTHS = phase1028.PATCH_DEPTHS
READOUT_DEPTH = phase1028.READOUT_DEPTH
PREOUTPUT_DEPTH = {
    "qwen3": 31,
    "glm4": 18,
    "deepseek7b": 25,
}
DISCOVERY_DONOR_OFFSETS = (1, 3)
CONFIRMATION_DONOR_OFFSETS = (1, 2, 4, 5)
WORLD_CODES = ("00", "10", "01", "11")
CONFIRMATION_CONDITIONS = (
    "selected_source_b",
    "unselected_source_b",
    "source_pair_b",
    "query_q",
    "source_pair_plus_query_mixed",
    "full_bq",
    "source_pair_scrambled",
    "source_pair_wrong_position",
    "query_wrong_position",
    "pre_output_b",
    "pre_output_bq",
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1029_multibinding_competition"
)


CONCEPTS = {
    "discovery": (
        ("papaya", "fruit"),
        ("badger", "animal"),
        ("trolley", "vehicle"),
        ("carpenter", "profession"),
        ("cathedral", "place"),
        ("kettle", "object"),
        ("crimson", "color"),
        ("thumb", "body_part"),
    ),
    "confirmation": (
        ("apricot", "fruit"),
        ("penguin", "animal"),
        ("tractor", "vehicle"),
        ("sculptor", "profession"),
        ("stadium", "place"),
        ("pillow", "object"),
        ("turquoise", "color"),
        ("forehead", "body_part"),
    ),
}

NONCE_PAIRS = {
    "discovery": (
        ("brast", "clovin"),
        ("drev", "fepal"),
        ("gornet", "halvik"),
        ("jasp", "krel"),
        ("morven", "niska"),
        ("palt", "ruxen"),
        ("sovel", "trask"),
        ("vint", "yolen"),
    ),
    "confirmation": (
        ("lumet", "navor"),
        ("pexin", "qular"),
        ("rovem", "sulk"),
        ("tivor", "wexal"),
        ("brenik", "cavor"),
        ("dovet", "faskin"),
        ("gurel", "hovin"),
        ("jex", "kurn"),
    ),
}

TEMPLATES = {
    "discovery": (
        'Dual binding record: code "{nonce_a}" represents {concept_a}; '
        'code "{nonce_b}" represents {concept_b}. Retrieval request: '
        'code "{query_nonce}" requires its broad category:'
    ),
    "confirmation": (
        'In this lookup, symbol "{nonce_a}" stands for {concept_a}, '
        'while symbol "{nonce_b}" stands for {concept_b}. Asked about '
        'symbol "{query_nonce}", return its general class:'
    ),
}


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


def donor_offsets(split: str) -> tuple[int, ...]:
    if split == "discovery":
        return DISCOVERY_DONOR_OFFSETS
    return CONFIRMATION_DONOR_OFFSETS


def build_units_and_cases() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    units: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    unit_lookup: dict[tuple[str, int, int, int], int] = {}
    for split in SPLITS:
        surface_range = range(4) if split == "discovery" else range(8)
        offsets = donor_offsets(split)
        for surface_index in surface_range:
            nonce_a, nonce_b = NONCE_PAIRS[split][surface_index]
            for target_index in range(8):
                for donor_offset in offsets:
                    donor_index = (target_index + donor_offset) % 8
                    q0_slot = (
                        "a"
                        if (surface_index + target_index) % 2 == 0
                        else "b"
                    )
                    unit_index = len(units)
                    unit_lookup[
                        (split, surface_index, target_index, donor_index)
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
                        concept_a = CONCEPTS[split][concept_a_index][0]
                        concept_b = CONCEPTS[split][concept_b_index][0]
                        query_nonce = (
                            nonce_a if query_slot == "a" else nonce_b
                        )
                        world = f"{binding_flip}{query_flip}"
                        expected_index = (
                            target_index
                            if binding_flip == query_flip
                            else donor_index
                        )
                        prompt = TEMPLATES[split].format(
                            nonce_a=nonce_a,
                            nonce_b=nonce_b,
                            concept_a=concept_a,
                            concept_b=concept_b,
                            query_nonce=query_nonce,
                        )
                        case_index = len(cases)
                        world_case_indices[world] = case_index
                        cases.append({
                            "schema_version": "phase1029_common_case.v1",
                            "phase": PHASE,
                            "case_index": case_index,
                            "case_key": f"{split}.u{unit_index}.{world}",
                            "split": split,
                            "unit_index": unit_index,
                            "surface_index": surface_index,
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
                            "expected_concept": CONCEPTS[split][
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
                        "schema_version": "phase1029_unit.v1",
                        "phase": PHASE,
                        "unit_index": unit_index,
                        "unit_key": (
                            f"{split}.s{surface_index}.t{target_index}."
                            f"d{donor_index}"
                        ),
                        "split": split,
                        "surface_index": surface_index,
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
                        "unselected_definition_role": (
                            "definition_nonce_b_end"
                            if q0_slot == "a"
                            else "definition_nonce_a_end"
                        ),
                        "world_case_indices": world_case_indices,
                    })

    for unit in units:
        offsets = donor_offsets(unit["split"])
        current_offset = int(unit["donor_offset"])
        next_offset = offsets[
            (offsets.index(current_offset) + 1) % len(offsets)
        ]
        scrambled_donor = (
            int(unit["target_index"]) + next_offset
        ) % 8
        unit["scrambled_donor_index"] = scrambled_donor
        unit["scrambled_unit_index"] = unit_lookup[
            (
                unit["split"],
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
        "schema_version": "phase1029_model_case.v1",
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
    for module in (phase1024, phase1026, phase1027, phase1028):
        for split in SPLITS:
            concepts.update(
                value for value, _ in module.CONCEPTS[split]
            )
            if hasattr(module, "NONCES"):
                nonces.update(module.NONCES[split])
    return concepts, nonces


def main() -> None:
    units, cases = build_units_and_cases()
    prior_concepts, prior_nonces = previous_words()
    unit_counts = Counter(row["split"] for row in units)
    case_counts = Counter(row["split"] for row in cases)
    new_concepts = {
        word
        for split in SPLITS
        for word, _ in CONCEPTS[split]
    }
    new_nonces = {
        word
        for split in SPLITS
        for pair in NONCE_PAIRS[split]
        for word in pair
    }
    checks = {
        "unit_count": len(units) == 320,
        "unit_split_counts": unit_counts == {
            "discovery": 64,
            "confirmation": 256,
        },
        "case_count": len(cases) == 1280,
        "case_split_counts": case_counts == {
            "discovery": 256,
            "confirmation": 1024,
        },
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
        "schema_version": "phase1029_common_audit.v1",
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
        "worlds": {
            "00": "base binding, base query, target answer",
            "10": "binding flipped, base query, alternate answer",
            "01": "base binding, query flipped, alternate answer",
            "11": "binding and query flipped, target answer",
        },
        "patch_depths": PATCH_DEPTHS,
        "readout_depth": READOUT_DEPTH,
        "preoutput_depth": PREOUTPUT_DEPTH,
        "discovery_donor_offsets": DISCOVERY_DONOR_OFFSETS,
        "confirmation_donor_offsets": CONFIRMATION_DONOR_OFFSETS,
        "confirmation_conditions": CONFIRMATION_CONDITIONS,
        "discovery_selection": (
            "using discovery units only, freeze one source-pair depth "
            "and one query depth per model by alternate-answer top1, "
            "then margin; confirmation cannot change either depth"
        ),
        "scale_free_confirmation_gate": {
            "clean_four_world_expected_top1_minimum": 0.50,
            "selected_source_alternate_top1_minimum": 0.40,
            "source_pair_alternate_top1_minimum": 0.50,
            "query_alternate_top1_minimum": 0.25,
            "combined_base_top1_minimum": 0.50,
            "full_bq_base_top1_minimum": 0.50,
            "selected_minus_unselected_alternate_top1_minimum": 0.10,
            "source_pair_minus_scrambled_alternate_top1_minimum": 0.10,
            "source_pair_minus_wrong_position_alternate_top1_minimum": 0.10,
            "query_minus_wrong_position_alternate_top1_minimum": 0.10,
            "combined_base_minus_single_base_top1_minimum": 0.10,
            "all_arrays_must_be_finite": True,
        },
        "interpretation_rule": (
            "B and Q each change the answer while BQ cancels. A binding "
            "mechanism candidate requires source transport, query "
            "transport, and causal cancellation on independent units; "
            "source-only transport is not binding closure."
        ),
        "claim_limit": (
            "two-binding residual-state competition only; no minimal "
            "circuit, component-level mechanism, general language law, "
            "brain homology, efficiency, or optimality claim"
        ),
        "unit_digest": digest(units),
        "clean_case_digest": digest(cases),
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
        "unit_counts": dict(unit_counts),
        "case_counts": dict(case_counts),
        "patch_depths": PATCH_DEPTHS,
        "audit": common_audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
