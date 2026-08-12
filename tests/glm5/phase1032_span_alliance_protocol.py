#!/usr/bin/env python3
"""Freeze the Phase1032 span-aware source/query alliance protocol."""

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


PHASE = 1032
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
WORLD_CODES = ("00", "10", "01", "11")
DONOR_OFFSETS = (1, 3)
SPAN_ROLES = (
    "definition_nonce_a",
    "concept_a",
    "definition_nonce_b",
    "concept_b",
    "query_nonce",
    "query_clause",
    "pre_output",
)
SELECTED_DEPTHS = {
    "qwen3": {"source": 9, "query": 25, "readout": 35},
    "glm4": {"source": 4, "query": 4, "readout": 19},
    "deepseek7b": {"source": 4, "query": 10, "readout": 27},
}
CONDITIONS = (
    "selected_source_endpoint_b",
    "selected_source_span_b",
    "unselected_source_span_b",
    "source_pair_endpoint_b",
    "source_pair_span_b",
    "source_pair_span_scrambled",
    "source_pair_span_wrong_position",
    "source_pair_span_self",
    "query_endpoint_q",
    "query_nonce_span_q",
    "query_clause_span_q",
    "query_endpoint_bq",
    "query_nonce_span_bq",
    "query_clause_span_bq",
    "query_clause_span_scrambled",
    "query_clause_span_self",
    "source_pair_span_plus_query_nonce_q",
    "source_pair_span_plus_query_clause_q",
    "source_pair_span_plus_query_nonce_bq",
    "source_pair_span_plus_query_clause_bq",
)

OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1032_span_aware_alliance"
)


CATEGORY_LABELS = (
    "fruit",
    "animal",
    "vehicle",
    "job",
    "place",
    "object",
    "color",
    "body",
)

# The two banks are deliberately balanced. Every item in ``single`` occupies
# one token and every item in ``double`` occupies two tokens in all three
# frozen tokenizers and both templates. The protocol audits this before any
# model is loaded.
CONCEPT_BANKS = {
    "single": (
        ("apple", "fruit"),
        ("horse", "animal"),
        ("bus", "vehicle"),
        ("teacher", "job"),
        ("park", "place"),
        ("mug", "object"),
        ("blue", "color"),
        ("hand", "body"),
    ),
    "double": (
        ("green apple", "fruit"),
        ("wild horse", "animal"),
        ("city bus", "vehicle"),
        ("school teacher", "job"),
        ("public park", "place"),
        ("coffee mug", "object"),
        ("bright blue", "color"),
        ("left hand", "body"),
    ),
}

NONCE_PAIRS = (
    ("alven", "borik"),
    ("ceron", "davel"),
    ("forik", "galen"),
    ("heron", "isken"),
    ("joral", "kelvin"),
    ("loran", "merik"),
    ("novel", "orpen"),
    ("pavin", "qurel"),
)

TEMPLATES = (
    (
        'Two-tag catalog. The entry "{nonce_a}" names {concept_a}; '
        'the entry "{nonce_b}" names {concept_b}. Return the general '
        'type of "{query_nonce}":'
    ),
    (
        'Use these assignments: {concept_a} is represented by code '
        '"{nonce_a}", while {concept_b} is represented by code '
        '"{nonce_b}". What broad kind does code "{query_nonce}" '
        'denote? Answer:'
    ),
)

QUERY_STARTS = (
    "Return the general type",
    "What broad kind",
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
    lookup: dict[tuple[int, int, str, int, int], int] = {}
    bank_names = tuple(CONCEPT_BANKS)

    for template_index, template in enumerate(TEMPLATES):
        for surface_index, (nonce_a, nonce_b) in enumerate(NONCE_PAIRS):
            for bank_index, bank_name in enumerate(bank_names):
                bank = CONCEPT_BANKS[bank_name]
                for target_index in range(len(CATEGORY_LABELS)):
                    for donor_offset in DONOR_OFFSETS:
                        donor_index = (
                            target_index + donor_offset
                        ) % len(CATEGORY_LABELS)
                        q0_slot = (
                            "a"
                            if (
                                template_index
                                + surface_index
                                + bank_index
                                + target_index
                            ) % 2 == 0
                            else "b"
                        )
                        unit_index = len(units)
                        lookup[(
                            template_index,
                            surface_index,
                            bank_name,
                            target_index,
                            donor_index,
                        )] = unit_index
                        world_case_indices: dict[str, int] = {}

                        for binding_flip, query_flip in (
                            (0, 0),
                            (1, 0),
                            (0, 1),
                            (1, 1),
                        ):
                            selected_index = (
                                target_index
                                if binding_flip == 0
                                else donor_index
                            )
                            other_index = (
                                donor_index
                                if binding_flip == 0
                                else target_index
                            )
                            if q0_slot == "a":
                                concept_a_index = selected_index
                                concept_b_index = other_index
                                query_slot = "a" if query_flip == 0 else "b"
                            else:
                                concept_a_index = other_index
                                concept_b_index = selected_index
                                query_slot = "b" if query_flip == 0 else "a"

                            concept_a = bank[concept_a_index][0]
                            concept_b = bank[concept_b_index][0]
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
                            query_start = prompt.index(
                                QUERY_STARTS[template_index]
                            )
                            query_end = (
                                prompt.rfind(query_nonce) + len(query_nonce)
                            )
                            case_index = len(cases)
                            world_case_indices[world] = case_index
                            cases.append({
                                "schema_version": "phase1032_common_case.v1",
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
                                "bank_index": bank_index,
                                "bank_name": bank_name,
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
                                "expected_label": CATEGORY_LABELS[
                                    expected_index
                                ],
                                "prompt": prompt,
                                "role_fragments": {
                                    "definition_nonce_a": fragment(
                                        prompt, nonce_a, occurrence="first"
                                    ),
                                    "concept_a": fragment(
                                        prompt, concept_a, occurrence="first"
                                    ),
                                    "definition_nonce_b": fragment(
                                        prompt, nonce_b, occurrence="first"
                                    ),
                                    "concept_b": fragment(
                                        prompt, concept_b, occurrence="first"
                                    ),
                                    "query_nonce": fragment(
                                        prompt, query_nonce, occurrence="last"
                                    ),
                                    "query_clause": (
                                        query_start,
                                        query_end,
                                        prompt[query_start:query_end],
                                    ),
                                },
                            })

                        units.append({
                            "schema_version": "phase1032_unit.v1",
                            "phase": PHASE,
                            "unit_index": unit_index,
                            "unit_key": (
                                f"t{template_index}.s{surface_index}."
                                f"{bank_name}.t{target_index}.d{donor_index}"
                            ),
                            "template_index": template_index,
                            "surface_index": surface_index,
                            "surface_key": (
                                template_index * len(NONCE_PAIRS)
                                + surface_index
                            ),
                            "bank_index": bank_index,
                            "bank_name": bank_name,
                            "target_index": target_index,
                            "donor_index": donor_index,
                            "donor_offset": donor_offset,
                            "q0_slot": q0_slot,
                            "selected_concept_role": (
                                "concept_a" if q0_slot == "a" else "concept_b"
                            ),
                            "unselected_concept_role": (
                                "concept_b" if q0_slot == "a" else "concept_a"
                            ),
                            "selected_definition_role": (
                                "definition_nonce_a"
                                if q0_slot == "a"
                                else "definition_nonce_b"
                            ),
                            "unselected_definition_role": (
                                "definition_nonce_b"
                                if q0_slot == "a"
                                else "definition_nonce_a"
                            ),
                            "world_case_indices": world_case_indices,
                        })

    for unit in units:
        other_offset = next(
            value
            for value in DONOR_OFFSETS
            if value != int(unit["donor_offset"])
        )
        scrambled_index = (
            int(unit["target_index"]) + other_offset
        ) % len(CATEGORY_LABELS)
        unit["scrambled_donor_index"] = scrambled_index
        unit["scrambled_unit_index"] = lookup[(
            int(unit["template_index"]),
            int(unit["surface_index"]),
            str(unit["bank_name"]),
            int(unit["target_index"]),
            scrambled_index,
        )]

    return units, cases


def candidate_token_ids(
    tokenizer,
    rendered: str,
    input_ids: list[int],
) -> list[int]:
    result = []
    for label in CATEGORY_LABELS:
        extended = [
            int(value)
            for value in tokenizer.encode(
                rendered + label,
                add_special_tokens=False,
            )
        ]
        if extended[:len(input_ids)] != input_ids:
            raise RuntimeError(
                f"candidate prefix retokenized for {label!r}"
            )
        suffix = extended[len(input_ids):]
        if len(suffix) != 1:
            raise RuntimeError(
                f"candidate {label!r} is not one next token: {suffix}"
            )
        result.append(int(suffix[0]))
    if len(set(result)) != len(result):
        raise RuntimeError("candidate labels do not have unique token ids")
    return result


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
    located = offset_token_spans(
        tokenizer,
        rendered,
        row["prompt"],
        row["role_fragments"],
    )
    role_spans = {
        role: [int(start), int(end)]
        for role, (start, end) in located.items()
    }
    role_spans["pre_output"] = [len(input_ids) - 1, len(input_ids) - 1]
    result = dict(row)
    result.pop("role_fragments", None)
    result.update({
        "schema_version": "phase1032_model_case.v1",
        "model": model_name,
        "record_id": f"{model_name}.{row['case_key']}",
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": role_spans,
        "role_positions": {
            role: int(span[1]) for role, span in role_spans.items()
        },
        "candidate_token_ids": candidate_token_ids(
            tokenizer, rendered, input_ids
        ),
        "prompt_token_count": len(input_ids),
    })
    return result


def common_audit(
    units: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    bank_counts = Counter(str(row["bank_name"]) for row in units)
    template_counts = Counter(int(row["template_index"]) for row in units)
    template_bank_counts = Counter(
        (int(row["template_index"]), str(row["bank_name"]))
        for row in units
    )
    checks = {
        "unit_count_512": len(units) == 512,
        "case_count_2048": len(cases) == 2048,
        "four_worlds_per_unit": all(
            set(row["world_case_indices"]) == set(WORLD_CODES)
            for row in units
        ),
        "bank_balance": bank_counts == {"single": 256, "double": 256},
        "template_balance": template_counts == {0: 256, 1: 256},
        "template_bank_balance": all(
            template_bank_counts[(template, bank)] == 128
            for template in range(len(TEMPLATES))
            for bank in CONCEPT_BANKS
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
        "scrambled_is_distinct": all(
            int(row["scrambled_unit_index"]) != int(row["unit_index"])
            and int(row["scrambled_donor_index"])
            not in {
                int(row["target_index"]),
                int(row["donor_index"]),
            }
            for row in units
        ),
    }
    return {
        "schema_version": "phase1032_common_audit.v1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "unit_counts_by_bank": dict(bank_counts),
        "unit_counts_by_template": dict(template_counts),
    }


def model_audit(
    model_name: str,
    units: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    case_by_index = {
        int(row["case_index"]): row for row in cases
    }
    role_lengths = {role: Counter() for role in SPAN_ROLES}
    bounds = []
    candidate_consistency = []
    expected_bank_lengths = []
    nonce_lengths = []
    within_unit_equalities = []

    for row in cases:
        width = len(row["input_ids"])
        for role, (start, end) in row["role_spans"].items():
            role_lengths[role][int(end) - int(start) + 1] += 1
            bounds.append(0 <= int(start) <= int(end) < width)
        candidate_consistency.append(
            len(row["candidate_token_ids"]) == len(CATEGORY_LABELS)
            and len(set(row["candidate_token_ids"]))
            == len(CATEGORY_LABELS)
        )
        expected_length = 1 if row["bank_name"] == "single" else 2
        expected_bank_lengths.extend([
            (
                row["role_spans"]["concept_a"][1]
                - row["role_spans"]["concept_a"][0]
                + 1
            ) == expected_length,
            (
                row["role_spans"]["concept_b"][1]
                - row["role_spans"]["concept_b"][0]
                + 1
            ) == expected_length,
        ])
        nonce_lengths.extend([
            (
                row["role_spans"]["definition_nonce_a"][1]
                - row["role_spans"]["definition_nonce_a"][0]
                + 1
            ) == 2,
            (
                row["role_spans"]["definition_nonce_b"][1]
                - row["role_spans"]["definition_nonce_b"][0]
                + 1
            ) == 2,
            (
                row["role_spans"]["query_nonce"][1]
                - row["role_spans"]["query_nonce"][0]
                + 1
            ) == 2,
        ])

    for unit in units:
        rows = [
            case_by_index[int(unit["world_case_indices"][world])]
            for world in WORLD_CODES
        ]
        for role in (
            "concept_a",
            "concept_b",
            "query_nonce",
            "query_clause",
        ):
            lengths = {
                int(row["role_spans"][role][1])
                - int(row["role_spans"][role][0])
                + 1
                for row in rows
            }
            within_unit_equalities.append(len(lengths) == 1)

    candidate_id_rows = {
        tuple(int(value) for value in row["candidate_token_ids"])
        for row in cases
    }
    checks = {
        "all_spans_in_bounds": all(bounds),
        "candidate_ids_unique_per_case": all(candidate_consistency),
        "candidate_ids_constant_within_model": len(candidate_id_rows) == 1,
        "concept_bank_lengths_exact": all(expected_bank_lengths),
        "nonce_spans_are_two_tokens": all(nonce_lengths),
        "all_patch_spans_equal_within_unit": all(within_unit_equalities),
    }
    return {
        "schema_version": "phase1032_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "role_span_length_counts": {
            role: dict(sorted(counts.items()))
            for role, counts in role_lengths.items()
        },
        "candidate_token_ids": list(next(iter(candidate_id_rows))),
    }


def main() -> None:
    units, common_cases = build_units_and_cases()
    common = common_audit(units, common_cases)
    if not common["all_checks_passed"]:
        raise RuntimeError(f"common audit failed: {common}")

    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "units.jsonl", units)
    write_jsonl(protocol_dir / "cases.common.jsonl", common_cases)
    write_json(protocol_dir / "common_audit.json", common)

    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        rows = [
            model_case(tokenizer, model_name, row)
            for row in common_cases
        ]
        audit = model_audit(model_name, units, rows)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"{model_name} tokenization audit failed: {audit}"
            )
        write_jsonl(protocol_dir / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_dir / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        del tokenizer

    prereg_core = {
        "schema_version": "phase1032_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Span-aware conditional source/query alliance atlas with "
            "direct next-token readout"
        ),
        "models": list(MODELS),
        "precision": "fp16",
        "quantization": "none",
        "sequential_model_order": list(MODELS),
        "unit_count": len(units),
        "case_count": len(common_cases),
        "world_codes": list(WORLD_CODES),
        "conditions": list(CONDITIONS),
        "selected_depths_frozen_from_phase1029": SELECTED_DEPTHS,
        "balanced_design": {
            "templates": len(TEMPLATES),
            "nonce_pairs": len(NONCE_PAIRS),
            "span_banks": {
                "single": "exactly one concept token",
                "double": "exactly two concept tokens",
            },
            "units_per_bank": 256,
            "units_per_template": 256,
            "candidate_categories": list(CATEGORY_LABELS),
        },
        "span_rules": {
            "primary": (
                "Patch every token one-to-one in complete, equal-length "
                "spans; no padding, pooling, truncation, or posthoc alignment."
            ),
            "endpoint_comparator": (
                "Patch only the final token on the exact same units."
            ),
            "single_token_identity_control": (
                "Endpoint and full-span source patches must be numerically "
                "identical in the single-token bank."
            ),
            "pre_output_role": (
                "The output boundary is a receiver/readout, not a member "
                "of an upstream alliance patch."
            ),
        },
        "primary_readouts": {
            "within_template_leave_surface_prototypes": (
                "Measure concept identity in a template-conditioned "
                "internal coordinate without using the held nonce pair."
            ),
            "next_token_candidate_logits": (
                "Measure the rank and margin among eight preregistered "
                "one-token category labels at the true generation boundary."
            ),
        },
        "secondary_readouts": {
            "pooled_template_prototypes": (
                "Descriptive shared-coordinate readout."
            ),
            "cross_template_prototypes": (
                "Direct coordinate-transfer audit, not a prerequisite "
                "for within-template causal evidence."
            ),
        },
        "descriptive_evidence_rules": {
            "selected_source_repetition": (
                "selected_source_span alternate Top1 exceeds "
                "unselected_source_span by at least 0.30 in both templates"
            ),
            "double_span_gain": (
                "on double-token units, full-span alternate Top1 exceeds "
                "endpoint alternate Top1 by at least 0.10"
            ),
            "distributed_query_gain": (
                "query-clause alternate Top1 exceeds query-nonce alternate "
                "Top1 by at least 0.10 and exceeds scrambled donor control "
                "by at least 0.10"
            ),
            "composition_restoration": (
                "source-pair plus query alliance raises base Top1 by at "
                "least 0.10 over both constituent interventions"
            ),
            "cross_model_repetition": (
                "A structure is called repeated only if the directional "
                "rule holds in at least two models and both templates."
            ),
        },
        "measurement_formulas": {
            "response_margin": (
                "m_i = score_i(alternate) - score_i(base)"
            ),
            "span_gain": (
                "G_span = mean(m_full_span - m_endpoint)"
            ),
            "query_distribution_gain": (
                "G_query = mean(m_query_clause - m_query_nonce)"
            ),
            "logit_interaction": (
                "I = m_source+query - m_source - m_query + m_clean"
            ),
        },
        "claim_limit": (
            "This phase can map repeated span-aware causal effects in an "
            "artificial two-binding retrieval pattern. It cannot establish "
            "a global knowledge network, a universal language mechanism, "
            "brain/LLM isomorphism, energy optimality, or a closed theory."
        ),
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    prereg["common_audit"] = common
    prereg["model_tokenization_audits"] = model_audits
    write_json(protocol_dir / "preregistration.json", prereg)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "unit_count": len(units),
        "case_count": len(common_cases),
        "common_audit": common,
        "model_audits": model_audits,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
