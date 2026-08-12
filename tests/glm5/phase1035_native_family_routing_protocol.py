#!/usr/bin/env python3
"""Freeze the native lexical-family-query routing atlas protocol.

The protocol deliberately starts from controlled observations rather than a
mechanism equation.  It crosses three factors:

* B: which semantic family is bound to each fact slot,
* Q: which nonce-defined fact slot is queried, and
* L: which lexical member instantiates each semantic family.

Discovery and confirmation use disjoint templates, nonce surfaces, and concept
surfaces.  Every prompt is a raw completion sequence with the same literal
post-query suffix and no chat template.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans


PHASE = 1035
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
SPLITS = ("discovery", "confirmation")
FAMILIES = (
    "fruit",
    "animal",
    "vehicle",
    "job",
    "place",
    "object",
    "color",
    "body",
)
DONOR_OFFSETS = (1, 3)
WORLD_FACTORS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
COMPONENTS = ("attention", "mlp")
ANCHORS = (
    "prefix_anchor",
    "concept_a",
    "concept_b",
    "query_nonce",
    "suffix_first",
    "suffix_mid",
    "pre_output",
)
FACTOR_METRICS = (
    "binding_q0_rel_norm",
    "binding_q1_rel_norm",
    "query_b0_rel_norm",
    "query_b1_rel_norm",
    "lexical_member_rel_norm",
    "bq_interaction_rel_norm",
    "bql_interaction_rel_norm",
    "binding_query_cosine",
    "bq_member_invariance",
    "binding_member_invariance_q0",
    "binding_member_invariance_q1",
    "query_member_invariance_b0",
    "query_member_invariance_b1",
    "bq_to_lexical_ratio",
)
DEPTH_BIN_COUNT = 8
FIXED_SUFFIX = "\nCategory:"
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1035_native_family_routing_atlas"
)


# Each split has two independent lexical members per family.  The protocol
# verifies that every concept occupies one token in every frozen tokenizer.
CONCEPTS = {
    "discovery": {
        "fruit": ("apple", "pear"),
        "animal": ("horse", "tiger"),
        "vehicle": ("bus", "train"),
        "job": ("teacher", "doctor"),
        "place": ("park", "beach"),
        "object": ("mug", "lamp"),
        "color": ("blue", "red"),
        "body": ("hand", "foot"),
    },
    "confirmation": {
        "fruit": ("grape", "lemon"),
        "animal": ("rabbit", "wolf"),
        "vehicle": ("car", "truck"),
        "job": ("nurse", "pilot"),
        "place": ("school", "hotel"),
        "object": ("chair", "clock"),
        "color": ("green", "black"),
        "body": ("arm", "leg"),
    },
}

NONCE_PAIRS = {
    "discovery": (
        ("alven", "borik"),
        ("ceron", "davel"),
        ("forik", "galen"),
        ("heron", "isken"),
    ),
    "confirmation": (
        ("arvik", "benor"),
        ("calen", "elvan"),
        ("feron", "gavin"),
        ("helor", "invar"),
    ),
}

TEMPLATES = {
    "discovery": (
        (
            "Definitions: {nonce_a} means {concept_a}; "
            "{nonce_b} means {concept_b}. Query: {query_nonce}"
        ),
        (
            "Glossary: {nonce_a} denotes {concept_a}; "
            "{nonce_b} denotes {concept_b}. Target: {query_nonce}"
        ),
    ),
    "confirmation": (
        (
            "Codebook: {nonce_a} stands for {concept_a}; "
            "{nonce_b} stands for {concept_b}. Asked code: {query_nonce}"
        ),
        (
            "Mappings: {nonce_a} refers to {concept_a}; "
            "{nonce_b} refers to {concept_b}. Selected symbol: {query_nonce}"
        ),
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


def build_units_and_cases() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    units: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for split in SPLITS:
        for local_template, template in enumerate(TEMPLATES[split]):
            global_template = (
                local_template
                if split == "discovery"
                else len(TEMPLATES["discovery"]) + local_template
            )
            for surface_index, (nonce_a, nonce_b) in enumerate(
                NONCE_PAIRS[split]
            ):
                for target_index in range(len(FAMILIES)):
                    for donor_offset in DONOR_OFFSETS:
                        donor_index = (
                            target_index + donor_offset
                        ) % len(FAMILIES)
                        target_family = FAMILIES[target_index]
                        donor_family = FAMILIES[donor_index]
                        q0_slot = (
                            "a"
                            if (
                                global_template
                                + surface_index
                                + target_index
                                + donor_offset
                            )
                            % 2
                            == 0
                            else "b"
                        )
                        unit_index = len(units)
                        world_case_indices: dict[str, int] = {}

                        for binding, query, lexical in WORLD_FACTORS:
                            target_concept = CONCEPTS[split][target_family][
                                lexical
                            ]
                            donor_concept = CONCEPTS[split][donor_family][
                                lexical
                            ]
                            selected_concept = (
                                target_concept if binding == 0 else donor_concept
                            )
                            other_concept = (
                                donor_concept if binding == 0 else target_concept
                            )
                            if q0_slot == "a":
                                concept_a = selected_concept
                                concept_b = other_concept
                                query_slot = "a" if query == 0 else "b"
                            else:
                                concept_a = other_concept
                                concept_b = selected_concept
                                query_slot = "b" if query == 0 else "a"
                            query_nonce = (
                                nonce_a if query_slot == "a" else nonce_b
                            )
                            expected_index = (
                                target_index
                                if binding == query
                                else donor_index
                            )
                            prompt_prefix = template.format(
                                nonce_a=nonce_a,
                                nonce_b=nonce_b,
                                concept_a=concept_a,
                                concept_b=concept_b,
                                query_nonce=query_nonce,
                            )
                            prompt = prompt_prefix + FIXED_SUFFIX
                            world = f"{binding}{query}{lexical}"
                            case_index = len(cases)
                            world_case_indices[world] = case_index
                            cases.append({
                                "schema_version": "phase1035_common_case.v1",
                                "phase": PHASE,
                                "case_index": case_index,
                                "case_key": (
                                    f"{split}.t{global_template}."
                                    f"u{unit_index}.{world}"
                                ),
                                "unit_index": unit_index,
                                "split": split,
                                "template_index": global_template,
                                "template_local_index": local_template,
                                "surface_index": surface_index,
                                "binding": binding,
                                "query": query,
                                "lexical": lexical,
                                "world": world,
                                "q0_slot": q0_slot,
                                "query_slot": query_slot,
                                "nonce_a": nonce_a,
                                "nonce_b": nonce_b,
                                "concept_a": concept_a,
                                "concept_b": concept_b,
                                "target_index": target_index,
                                "donor_index": donor_index,
                                "target_family": target_family,
                                "donor_family": donor_family,
                                "expected_index": expected_index,
                                "expected_label": FAMILIES[expected_index],
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
                                    "fixed_suffix": fragment(
                                        prompt, FIXED_SUFFIX, occurrence="last"
                                    ),
                                },
                            })

                        units.append({
                            "schema_version": "phase1035_unit.v1",
                            "phase": PHASE,
                            "unit_index": unit_index,
                            "unit_key": (
                                f"{split}.t{global_template}.s{surface_index}."
                                f"a{target_index}.b{donor_index}"
                            ),
                            "split": split,
                            "template_index": global_template,
                            "template_local_index": local_template,
                            "surface_index": surface_index,
                            "target_index": target_index,
                            "donor_index": donor_index,
                            "target_family": target_family,
                            "donor_family": donor_family,
                            "donor_offset": donor_offset,
                            "q0_slot": q0_slot,
                            "world_case_indices": world_case_indices,
                        })
    return units, cases


def candidate_token_ids(
    tokenizer,
    prompt: str,
    input_ids: list[int],
) -> list[int]:
    result = []
    for label in FAMILIES:
        extended = [
            int(value)
            for value in tokenizer.encode(
                prompt + " " + label,
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
    prompt = str(row["prompt"])
    input_ids = [
        int(value)
        for value in tokenizer.encode(prompt, add_special_tokens=False)
    ]
    located = offset_token_spans(
        tokenizer,
        prompt,
        prompt,
        row["role_fragments"],
    )
    role_spans = {
        role: [int(start), int(end)]
        for role, (start, end) in located.items()
    }
    suffix_start, suffix_end = role_spans["fixed_suffix"]
    suffix_positions = list(range(suffix_start, suffix_end + 1))
    suffix_mid = suffix_positions[(len(suffix_positions) - 1) // 2]
    result = dict(row)
    result.pop("role_fragments", None)
    result.update({
        "schema_version": "phase1035_model_case.v1",
        "model": model_name,
        "record_id": f"{model_name}.{row['case_key']}",
        "rendered_prompt": prompt,
        "input_ids": input_ids,
        "role_spans": role_spans,
        "anchor_spans": {
            "prefix_anchor": [0, 0],
            "concept_a": role_spans["concept_a"],
            "concept_b": role_spans["concept_b"],
            "query_nonce": role_spans["query_nonce"],
            "suffix_first": [suffix_start, suffix_start],
            "suffix_mid": [suffix_mid, suffix_mid],
            "pre_output": [suffix_end, suffix_end],
        },
        "suffix_token_ids": input_ids[suffix_start:suffix_end + 1],
        "suffix_token_strings": [
            str(tokenizer.convert_ids_to_tokens(int(value)))
            for value in input_ids[suffix_start:suffix_end + 1]
        ],
        "candidate_token_ids": candidate_token_ids(
            tokenizer, prompt, input_ids
        ),
        "special_token_ids_present": sorted(
            set(input_ids).intersection(
                int(value) for value in tokenizer.all_special_ids
            )
        ),
        "prompt_token_count": len(input_ids),
    })
    result["candidate_token_strings"] = [
        str(tokenizer.convert_ids_to_tokens(int(value)))
        for value in result["candidate_token_ids"]
    ]
    return result


def common_audit(
    units: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    unit_case_counts = Counter(int(row["unit_index"]) for row in cases)
    split_units = Counter(row["split"] for row in units)
    split_cases = Counter(row["split"] for row in cases)
    world_counts = Counter(row["world"] for row in cases)
    expected_by_unit = {}
    for unit in units:
        indices = [
            int(value)
            for value in unit["world_case_indices"].values()
        ]
        rows = [cases[index] for index in indices]
        expected_by_unit[int(unit["unit_index"])] = all(
            len({
                int(row["expected_index"])
                for row in rows
                if (
                    int(row["binding"]),
                    int(row["query"]),
                )
                == (binding, query)
            })
            == 1
            for binding in (0, 1)
            for query in (0, 1)
        )
    discovery_concepts = {
        value
        for words in CONCEPTS["discovery"].values()
        for value in words
    }
    confirmation_concepts = {
        value
        for words in CONCEPTS["confirmation"].values()
        for value in words
    }
    discovery_nonces = {
        value for pair in NONCE_PAIRS["discovery"] for value in pair
    }
    confirmation_nonces = {
        value for pair in NONCE_PAIRS["confirmation"] for value in pair
    }
    checks = {
        "unit_count_256": len(units) == 256,
        "case_count_2048": len(cases) == 2048,
        "eight_cases_per_unit": set(unit_case_counts.values()) == {8},
        "split_unit_balance": dict(split_units)
        == {"discovery": 128, "confirmation": 128},
        "split_case_balance": dict(split_cases)
        == {"discovery": 1024, "confirmation": 1024},
        "world_balance": set(world_counts.values()) == {256},
        "lexical_factor_preserves_expected_family": all(
            expected_by_unit.values()
        ),
        "disjoint_concept_surfaces": discovery_concepts.isdisjoint(
            confirmation_concepts
        ),
        "disjoint_nonce_surfaces": discovery_nonces.isdisjoint(
            confirmation_nonces
        ),
        "disjoint_templates": not set(
            TEMPLATES["discovery"]
        ).intersection(TEMPLATES["confirmation"]),
        "fixed_literal_suffix": all(
            str(row["prompt"]).endswith(FIXED_SUFFIX) for row in cases
        ),
    }
    return {
        "schema_version": "phase1035_common_audit.v1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "unit_count": len(units),
        "case_count": len(cases),
        "split_units": dict(split_units),
        "split_cases": dict(split_cases),
        "world_counts": dict(world_counts),
    }


def model_audit(
    model_name: str,
    units: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    unit_lengths: dict[int, set[int]] = {}
    candidate_rows = set()
    suffix_rows = set()
    concept_lengths = []
    nonce_pair_equal = []
    bounds = []
    no_specials = []
    for row in cases:
        unit_lengths.setdefault(int(row["unit_index"]), set()).add(
            len(row["input_ids"])
        )
        candidate_rows.add(tuple(row["candidate_token_ids"]))
        suffix_rows.add(tuple(row["suffix_token_ids"]))
        width = len(row["input_ids"])
        for start, end in row["anchor_spans"].values():
            bounds.append(0 <= int(start) <= int(end) < width)
        concept_lengths.extend([
            int(row["role_spans"]["concept_a"][1])
            - int(row["role_spans"]["concept_a"][0])
            + 1,
            int(row["role_spans"]["concept_b"][1])
            - int(row["role_spans"]["concept_b"][0])
            + 1,
        ])
        nonce_a_length = (
            int(row["role_spans"]["definition_nonce_a"][1])
            - int(row["role_spans"]["definition_nonce_a"][0])
            + 1
        )
        nonce_b_length = (
            int(row["role_spans"]["definition_nonce_b"][1])
            - int(row["role_spans"]["definition_nonce_b"][0])
            + 1
        )
        nonce_pair_equal.append(nonce_a_length == nonce_b_length)
        no_specials.append(not row["special_token_ids_present"])
    checks = {
        "all_anchor_spans_in_bounds": all(bounds),
        "all_concepts_exactly_one_token": set(concept_lengths) == {1},
        "nonce_pairs_equal_length": all(nonce_pair_equal),
        "all_worlds_equal_length_within_unit": all(
            len(values) == 1 for values in unit_lengths.values()
        ),
        "candidate_ids_constant": len(candidate_rows) == 1,
        "candidate_ids_unique": (
            len(next(iter(candidate_rows))) == len(FAMILIES)
            and len(set(next(iter(candidate_rows)))) == len(FAMILIES)
        ),
        "suffix_token_ids_constant": len(suffix_rows) == 1,
        "suffix_has_at_least_two_tokens": (
            len(next(iter(suffix_rows))) >= 2
        ),
        "no_special_control_tokens": all(no_specials),
        "no_chat_markers": all(
            marker not in str(row["rendered_prompt"])
            for row in cases
            for marker in (
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
                "<think>",
                "[INST]",
            )
        ),
    }
    first = cases[0]
    return {
        "schema_version": "phase1035_model_audit.v1",
        "model": model_name,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "prompt_token_count_range": [
            min(len(row["input_ids"]) for row in cases),
            max(len(row["input_ids"]) for row in cases),
        ],
        "concept_token_length_counts": dict(Counter(concept_lengths)),
        "candidate_token_ids": list(next(iter(candidate_rows))),
        "candidate_token_strings": first["candidate_token_strings"],
        "suffix_token_ids": list(next(iter(suffix_rows))),
        "suffix_token_strings": first["suffix_token_strings"],
    }


def main() -> None:
    units, common_cases = build_units_and_cases()
    common = common_audit(units, common_cases)
    if not common["all_checks_passed"]:
        raise RuntimeError(f"common protocol audit failed: {common}")

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
            write_json(protocol_dir / f"audit.{model_name}.json", audit)
            raise RuntimeError(
                f"{model_name} tokenization audit failed: {audit}"
            )
        write_jsonl(protocol_dir / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_dir / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        del tokenizer

    prereg_core = {
        "schema_version": "phase1035_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Native lexical-family-query factorial routing and event atlas"
        ),
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "unit_count": len(units),
        "case_count": len(common_cases),
        "factor_order": ["binding", "query", "lexical_member"],
        "world_order": [
            f"{binding}{query}{lexical}"
            for binding, query, lexical in WORLD_FACTORS
        ],
        "raw_sequence_controls": {
            "chat_template_used": False,
            "special_tokens_used": False,
            "fixed_literal_suffix": FIXED_SUFFIX,
            "suffix_ids_constant_within_model": True,
            "equal_token_length_across_worlds_within_unit": True,
        },
        "independent_confirmation": {
            "new_templates": True,
            "new_nonce_surfaces": True,
            "new_concept_surfaces": True,
            "same_semantic_families": True,
            "same_factorial_protocol": True,
        },
        "observed_components": {
            "residual": "embedding input and every layer output",
            "attention": "every layer output at each role anchor",
            "mlp": "every layer output at each role anchor",
            "anchors": list(ANCHORS),
            "metrics": list(FACTOR_METRICS),
        },
        "event_definition": {
            "name": "lexically_repeated_binding_query_event",
            "description": (
                "A cell where the binding response reverses with the "
                "queried slot and the BxQ interaction direction repeats "
                "across two independent lexical members."
            ),
            "discovery_rule": {
                "binding_query_cosine_median_max": 0.0,
                "binding_query_negative_rate_min": 0.75,
                "bq_member_invariance_median_min": 0.0,
                "bq_member_positive_rate_min": 0.75,
                "bq_interaction_rel_norm_median_min": 1e-4,
                "both_discovery_templates_required": True,
            },
            "confirmation_rule": (
                "Freeze the normalized depth bin, component, and anchor "
                "from discovery; require the same signed rule in both "
                "held-out confirmation templates."
            ),
            "cross_model_rule": (
                "Call an event physically conserved only when the same "
                "normalized depth bin, component, and anchor confirms in "
                "at least two instrumented models."
            ),
        },
        "behavior_readouts": {
            "primary": (
                "Accuracy and margins among eight one-token family labels."
            ),
            "secondary": (
                "Global vocabulary top-1 and expected-token rank, only "
                "on fully finite output rows."
            ),
            "internal": (
                "Discovery family prototypes classify held-out confirmation "
                "boundary states; never substitute this for output logits."
            ),
        },
        "instrumentation_gate": {
            "residual_addition_relative_error_p95_max": 0.02,
            "candidate_logit_finite_row_rate_min": 0.95,
        },
        "claim_limits": [
            (
                "A repeated BxQ event is a stable response topology, not "
                "yet a causal transport path or a language equation."
            ),
            (
                "The lexical-member factor supports family abstraction "
                "only when discovery repeats on disjoint confirmation "
                "words; it cannot prove biological optimality."
            ),
            (
                "Family labels are an artificial controlled task and do "
                "not establish a complete natural knowledge graph."
            ),
        ],
        "common_audit": common,
        "model_audits": model_audits,
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(protocol_dir / "preregistration.json", prereg)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "units": len(units),
        "cases": len(common_cases),
        "model_audits_passed": {
            model: row["all_checks_passed"]
            for model, row in model_audits.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
