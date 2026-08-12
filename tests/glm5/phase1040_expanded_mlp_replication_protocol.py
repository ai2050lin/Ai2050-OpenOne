#!/usr/bin/env python3
"""Freeze an expanded independent replication of the early source-MLP effect.

The material is new relative to Phase1035-1039: ten semantic families,
thirty ordered family pairs, new templates, new nonce symbols, and both
single-token and exactly two-token concept spans in all three tokenizers.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1035_native_family_routing_protocol as common
import phase1039_source_channel_causal_protocol as evidence


PHASE = 1040
PROTOCOL_REVISION = 1
MODELS = common.MODELS
PRECISION = common.PRECISION
QUANTIZATION = common.QUANTIZATION
EVIDENCE_ROOT = evidence.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1040_expanded_mlp_replication"
)
FAMILIES = (
    "food",
    "plant",
    "tool",
    "emotion",
    "material",
    "music",
    "sport",
    "clothing",
    "weather",
    "shape",
)
DONOR_OFFSETS = (1, 3, 7)
SURFACE_STRATA = ("single_token", "two_token")
WORLD_FACTORS = common.WORLD_FACTORS
FIXED_SUFFIX = common.FIXED_SUFFIX
CHANNELS = ("mlp_write", "layer_output")
CONDITIONS = (
    "same_family_selected",
    "cross_family_selected",
    "cross_family_unselected",
    "cross_family_wrong_target",
)
TEMPLATES = (
    (
        "Vocabulary check: {nonce_a} represents {concept_a}, and "
        "{nonce_b} represents {concept_b}. Requested code: {query_nonce}"
    ),
    (
        "Lookup table: pair {concept_a} with {nonce_a}; pair {concept_b} "
        "with {nonce_b}. Inspect symbol: {query_nonce}"
    ),
)
NONCE_PAIRS = {
    0: (
        ("jalen", "kovir"),
        ("lurem", "mavin"),
        ("navor", "pelin"),
    ),
    1: (
        ("ravel", "sorin"),
        ("tavin", "velor"),
        ("walen", "xorin"),
    ),
}
CONCEPTS = {
    "single_token": {
        "food": ("bread", "cheese"),
        "plant": ("oak", "pine"),
        "tool": ("hammer", "wrench"),
        "emotion": ("joy", "anger"),
        "material": ("steel", "cotton"),
        "music": ("piano", "violin"),
        "sport": ("tennis", "soccer"),
        "clothing": ("shirt", "jacket"),
        "weather": ("rain", "snow"),
        "shape": ("circle", "square"),
    },
    "two_token": {
        "food": ("fresh bread", "soft cheese"),
        "plant": ("tall oak", "green pine"),
        "tool": ("heavy hammer", "metal wrench"),
        "emotion": ("quiet joy", "sudden anger"),
        "material": ("hard steel", "soft cotton"),
        "music": ("grand piano", "old violin"),
        "sport": ("table tennis", "field soccer"),
        "clothing": ("blue shirt", "warm jacket"),
        "weather": ("heavy rain", "white snow"),
        "shape": ("round circle", "plain square"),
    },
}


write_json = common.write_json
write_jsonl = common.write_jsonl
read_json = common.read_json
read_jsonl = common.read_jsonl
digest = common.digest


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
    for template_index, template in enumerate(TEMPLATES):
        for surface_index, (nonce_a, nonce_b) in enumerate(
            NONCE_PAIRS[template_index]
        ):
            for stratum in SURFACE_STRATA:
                for target_index, target_family in enumerate(FAMILIES):
                    for donor_offset in DONOR_OFFSETS:
                        donor_index = (
                            target_index + donor_offset
                        ) % len(FAMILIES)
                        donor_family = FAMILIES[donor_index]
                        q0_slot = (
                            "a"
                            if (
                                template_index
                                + surface_index
                                + target_index
                                + donor_offset
                                + SURFACE_STRATA.index(stratum)
                            )
                            % 2
                            == 0
                            else "b"
                        )
                        unit_index = len(units)
                        world_case_indices: dict[str, int] = {}
                        for binding, query, lexical in WORLD_FACTORS:
                            target_concept = CONCEPTS[stratum][
                                target_family
                            ][lexical]
                            donor_concept = CONCEPTS[stratum][
                                donor_family
                            ][lexical]
                            if binding == 0:
                                slot_a_family = (
                                    target_family
                                    if q0_slot == "a"
                                    else donor_family
                                )
                                slot_b_family = (
                                    donor_family
                                    if q0_slot == "a"
                                    else target_family
                                )
                                concept_a = (
                                    target_concept
                                    if q0_slot == "a"
                                    else donor_concept
                                )
                                concept_b = (
                                    donor_concept
                                    if q0_slot == "a"
                                    else target_concept
                                )
                            else:
                                slot_a_family = (
                                    donor_family
                                    if q0_slot == "a"
                                    else target_family
                                )
                                slot_b_family = (
                                    target_family
                                    if q0_slot == "a"
                                    else donor_family
                                )
                                concept_a = (
                                    donor_concept
                                    if q0_slot == "a"
                                    else target_concept
                                )
                                concept_b = (
                                    target_concept
                                    if q0_slot == "a"
                                    else donor_concept
                                )
                            selected_slot = (
                                q0_slot
                                if query == 0
                                else ("b" if q0_slot == "a" else "a")
                            )
                            query_nonce = (
                                nonce_a if selected_slot == "a" else nonce_b
                            )
                            expected_family = (
                                slot_a_family
                                if selected_slot == "a"
                                else slot_b_family
                            )
                            body = template.format(
                                nonce_a=nonce_a,
                                nonce_b=nonce_b,
                                concept_a=concept_a,
                                concept_b=concept_b,
                                query_nonce=query_nonce,
                            )
                            prompt = body + FIXED_SUFFIX
                            case_index = len(cases)
                            world = f"{binding}{query}{lexical}"
                            world_case_indices[world] = case_index
                            cases.append({
                                "schema_version": "phase1040_common_case.v1",
                                "phase": PHASE,
                                "case_index": case_index,
                                "unit_index": unit_index,
                                "case_key": (
                                    f"t{template_index}.s{surface_index}."
                                    f"{stratum}.a{target_index}."
                                    f"b{donor_index}.w{world}"
                                ),
                                "template_index": template_index,
                                "surface_index": surface_index,
                                "surface_stratum": stratum,
                                "target_index": target_index,
                                "donor_index": donor_index,
                                "target_family": target_family,
                                "donor_family": donor_family,
                                "q0_slot": q0_slot,
                                "binding": binding,
                                "query": query,
                                "lexical": lexical,
                                "world": world,
                                "concept_a": concept_a,
                                "concept_b": concept_b,
                                "query_nonce": query_nonce,
                                "expected_index": FAMILIES.index(
                                    expected_family
                                ),
                                "expected_label": expected_family,
                                "prompt": prompt,
                                "role_fragments": {
                                    "concept_a": fragment(
                                        prompt,
                                        concept_a,
                                        occurrence="first",
                                    ),
                                    "concept_b": fragment(
                                        prompt,
                                        concept_b,
                                        occurrence="first",
                                    ),
                                    "query_nonce": fragment(
                                        prompt,
                                        query_nonce,
                                        occurrence="last",
                                    ),
                                    "fixed_suffix": fragment(
                                        prompt,
                                        FIXED_SUFFIX,
                                        occurrence="last",
                                    ),
                                },
                            })
                        units.append({
                            "schema_version": "phase1040_unit.v1",
                            "phase": PHASE,
                            "unit_index": unit_index,
                            "unit_key": (
                                f"t{template_index}.s{surface_index}."
                                f"{stratum}.a{target_index}.b{donor_index}"
                            ),
                            "template_index": template_index,
                            "surface_index": surface_index,
                            "surface_stratum": stratum,
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
            raise RuntimeError(f"candidate prefix retokenized: {label}")
        suffix = extended[len(input_ids):]
        if len(suffix) != 1:
            raise RuntimeError(
                f"candidate {label!r} is not one next token: {suffix}"
            )
        result.append(int(suffix[0]))
    if len(set(result)) != len(result):
        raise RuntimeError("candidate token collision")
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
        tokenizer, prompt, prompt, row["role_fragments"]
    )
    role_spans = {
        role: [int(start), int(end)]
        for role, (start, end) in located.items()
    }
    expected_span = (
        1 if row["surface_stratum"] == "single_token" else 2
    )
    for role in ("concept_a", "concept_b"):
        start, end = role_spans[role]
        if end - start + 1 != expected_span:
            raise RuntimeError(
                f"{model_name} {row['case_key']} {role} "
                f"has {end - start + 1} tokens, expected {expected_span}"
            )
    suffix_start, suffix_end = role_spans["fixed_suffix"]
    result = dict(row)
    result.pop("role_fragments", None)
    result.update({
        "schema_version": "phase1040_model_case.v1",
        "model": model_name,
        "record_id": f"{model_name}.{row['case_key']}",
        "rendered_prompt": prompt,
        "input_ids": input_ids,
        "role_spans": role_spans,
        "anchor_spans": {
            "concept_a": role_spans["concept_a"],
            "concept_b": role_spans["concept_b"],
            "query_nonce": role_spans["query_nonce"],
            "pre_output": [suffix_end, suffix_end],
        },
        "candidate_token_ids": candidate_token_ids(
            tokenizer, prompt, input_ids
        ),
        "candidate_token_strings": [],
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


def build_targets(
    cases: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            int(row["unit_index"]),
            int(row["binding"]),
            int(row["query"]),
            int(row["lexical"]),
        ): int(row["case_index"])
        for row in cases
    }
    targets = []
    for unit in units:
        q0_slot = str(unit["q0_slot"])
        for query in (0, 1):
            selected_slot = (
                q0_slot
                if query == 0
                else ("b" if q0_slot == "a" else "a")
            )
            unselected_slot = "b" if selected_slot == "a" else "a"
            unit_index = int(unit["unit_index"])
            target_case = lookup[(unit_index, 0, query, 0)]
            same_case = lookup[(unit_index, 0, query, 1)]
            cross_case = lookup[(unit_index, 1, query, 0)]
            target = cases[target_case]
            cross = cases[cross_case]
            targets.append({
                "schema_version": "phase1040_target.v1",
                "phase": PHASE,
                "target_index": len(targets),
                "unit_index": unit_index,
                "template_index": int(unit["template_index"]),
                "surface_stratum": str(unit["surface_stratum"]),
                "ordered_pair": (
                    f"{unit['target_family']}->{unit['donor_family']}"
                ),
                "query": query,
                "selected_role": f"concept_{selected_slot}",
                "unselected_role": f"concept_{unselected_slot}",
                "target_case_index": target_case,
                "same_family_case_index": same_case,
                "cross_family_case_index": cross_case,
                "target_family_index": int(target["expected_index"]),
                "target_family": str(target["expected_label"]),
                "cross_family_index": int(cross["expected_index"]),
                "cross_family": str(cross["expected_label"]),
            })
    return targets


def main() -> None:
    evidence_prereg = read_json(
        EVIDENCE_ROOT / "protocol" / "preregistration.json"
    )
    evidence_aggregate = read_json(EVIDENCE_ROOT / "aggregate.json")
    cells = evidence_aggregate["cross_model_single_channel_cells"]
    if not any(
        int(row["normalized_depth_slot"]) == 1
        and row["channel"] == "mlp_write"
        and set(row["models"]) == {"qwen3", "deepseek7b"}
        for row in cells
    ):
        raise RuntimeError("Phase1039 early-MLP candidate missing")

    units, common_cases = build_units_and_cases()
    targets = build_targets(common_cases, units)
    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "units.jsonl", units)
    write_jsonl(protocol_dir / "cases.common.jsonl", common_cases)
    write_jsonl(protocol_dir / "targets.jsonl", targets)

    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        model_cases = [
            model_case(tokenizer, model_name, row)
            for row in common_cases
        ]
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl", model_cases
        )
        candidate_rows = {
            tuple(row["candidate_token_ids"]) for row in model_cases
        }
        span_counts = Counter(
            (
                row["surface_stratum"],
                row["anchor_spans"]["concept_a"][1]
                - row["anchor_spans"]["concept_a"][0]
                + 1,
            )
            for row in model_cases
        )
        audit = {
            "schema_version": "phase1040_model_audit.v1",
            "model": model_name,
            "case_count": len(model_cases),
            "candidate_rows_unique": len(candidate_rows),
            "candidate_count": len(next(iter(candidate_rows))),
            "concept_span_counts": {
                f"{key[0]}/{key[1]}": value
                for key, value in sorted(span_counts.items())
            },
            "special_token_case_count": sum(
                bool(row["special_token_ids_present"])
                for row in model_cases
            ),
            "checks": {
                "all_cases_present": len(model_cases) == 2880,
                "candidate_ids_constant": len(candidate_rows) == 1,
                "candidate_count_10": (
                    len(next(iter(candidate_rows))) == 10
                ),
                "no_special_tokens": all(
                    not row["special_token_ids_present"]
                    for row in model_cases
                ),
                "single_spans_are_one": all(
                    row["anchor_spans"]["concept_a"][1]
                    - row["anchor_spans"]["concept_a"][0]
                    + 1
                    == 1
                    for row in model_cases
                    if row["surface_stratum"] == "single_token"
                ),
                "two_token_spans_are_two": all(
                    row["anchor_spans"]["concept_a"][1]
                    - row["anchor_spans"]["concept_a"][0]
                    + 1
                    == 2
                    for row in model_cases
                    if row["surface_stratum"] == "two_token"
                ),
            },
        }
        audit["all_checks_passed"] = all(audit["checks"].values())
        write_json(protocol_dir / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
    if not all(row["all_checks_passed"] for row in model_audits.values()):
        raise RuntimeError("model tokenization audit failed")

    common_audit = {
        "schema_version": "phase1040_common_audit.v1",
        "unit_count": len(units),
        "case_count": len(common_cases),
        "target_count": len(targets),
        "ordered_pair_count": len({
            (row["target_index"], row["donor_index"]) for row in units
        }),
        "unit_counts_by_template_stratum": dict(Counter(
            f"t{row['template_index']}/{row['surface_stratum']}"
            for row in units
        )),
        "checks": {
            "unit_count_360": len(units) == 360,
            "case_count_2880": len(common_cases) == 2880,
            "target_count_720": len(targets) == 720,
            "eight_worlds_per_unit": all(
                len(row["world_case_indices"]) == 8 for row in units
            ),
            "thirty_ordered_pairs": len({
                (row["target_index"], row["donor_index"])
                for row in units
            }) == 30,
            "balanced_templates": Counter(
                int(row["template_index"]) for row in targets
            ) == {0: 360, 1: 360},
            "balanced_strata": Counter(
                str(row["surface_stratum"]) for row in targets
            ) == {"single_token": 360, "two_token": 360},
        },
    }
    common_audit["all_checks_passed"] = all(
        common_audit["checks"].values()
    )
    write_json(protocol_dir / "audit.common.json", common_audit)
    if not common_audit["all_checks_passed"]:
        raise RuntimeError("common protocol audit failed")

    model_depths = {
        model: int(
            evidence_prereg["model_physical_depths"][model][0]
        )
        for model in MODELS
    }
    prereg_core: dict[str, Any] = {
        "schema_version": "phase1040_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Expanded new-family and two-token replication of the early "
            "queried-source MLP write"
        ),
        "evidence_phase": evidence.PHASE,
        "evidence_protocol_digest": evidence_prereg["protocol_digest"],
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "families": list(FAMILIES),
        "donor_offsets": list(DONOR_OFFSETS),
        "surface_strata": list(SURFACE_STRATA),
        "channels": list(CHANNELS),
        "conditions": list(CONDITIONS),
        "model_physical_depth": model_depths,
        "normalized_depth_slot": 1,
        "sample_counts": {
            "units": len(units),
            "cases_per_model": len(common_cases),
            "targets": len(targets),
            "targets_per_template": 360,
            "targets_per_surface_stratum": 360,
            "ordered_family_pairs": 30,
        },
        "selection_rule": (
            "Freeze only the normalized early depth and complete MLP write "
            "identified before these families, templates, nonce symbols, and "
            "two-token phrases were constructed. GLM4 is retained as a "
            "negative-model replication, not excluded."
        ),
        "interventions": {
            "mlp_write": (
                "Add the clean donor-minus-target complete MLP write at every "
                "aligned token in the concept span."
            ),
            "layer_output": (
                "Replace every aligned target concept-span state with the "
                "clean donor layer-output state, providing a same-task full "
                "state baseline."
            ),
        },
        "replication_gate": {
            "cross_selected_shift_median_min": 0.0,
            "cross_selected_positive_rate_min": 0.65,
            "selected_minus_unselected_median_min": 0.0,
            "selected_minus_wrong_target_median_min": 0.0,
            "cross_to_same_absolute_ratio_min": 2.0,
            "whole_state_effect_retention_min": 0.10,
            "ordered_pair_positive_median_rate_min": 0.70,
            "both_templates_and_both_surface_strata_required": True,
            "minimum_models": 2,
        },
        "claim_limits": [
            (
                "Passing establishes a repeated early-MLP causal contribution "
                "for this controlled family-routing pattern, not a universal "
                "knowledge or language mechanism."
            ),
            (
                "Two-token aligned-span success does not establish arbitrary "
                "phrase-length or compositional generalization."
            ),
            (
                "The new family labels remain an artificial candidate set; "
                "natural full-vocabulary generation is reported separately."
            ),
            (
                "The result does not prove biological optimality or that the "
                "same physical MLP coordinates are shared across models."
            ),
        ],
        "common_audit": common_audit,
        "model_audits": model_audits,
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(protocol_dir / "preregistration.json", prereg)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "sample_counts": prereg["sample_counts"],
        "model_physical_depth": model_depths,
        "all_audits_passed": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
