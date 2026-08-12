#!/usr/bin/env python3
"""Freeze Phase1080 natural relevance and computation-demand atlas.

The protocol removes explicit semantic/index mode selectors and candidate
lists.  Every natural task is rendered in three output-matched branches:
infer (neutral cue), decoy (the target cue is explicitly unrelated), and
direct (the same target cue is explicitly relevant).  Direct-minus-decoy
therefore matches target cue identity while changing its relation to the
current question.  Decoy-minus-infer separately measures target-cue
presence.  These are descriptive controls, not assumed mechanism vectors.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior
import phase1079_output_orthogonal_pattern_protocol as source


PHASE = 1080
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
FAMILIES = source.FAMILIES
BASE_FAMILIES = source.BASE_FAMILIES
HELDOUT_FAMILY = source.HELDOUT_FAMILY
SPLITS = source.SPLITS
BRANCHES = ("infer", "decoy", "direct")
TEMPLATES_BY_SPLIT = source.TEMPLATES_BY_SPLIT
STATES = tuple(
    f"t{template}_b{branch}_a{answer}_l{surface}"
    for template in (0, 1)
    for branch in BRANCHES
    for answer in (0, 1)
    for surface in (0, 1)
)
CAPTURE_ROLES = (
    "content_anchor",
    "context_end",
    "branch_end",
    "request_end",
    "answer_boundary",
)
PRE_BRANCH_ROLES = ("content_anchor", "context_end")
INTERMEDIATE_ROLES = ("request_end",)
CONDITIONINGS = ("all_finite", "behavior_supported")
ITEMS_PER_FAMILY_SPLIT = 12
GENERATION_UNITS_PER_FAMILY_SPLIT_BRANCH = 4
GENERATION_STEPS = 5
ASSISTANT_PREFILL = "Completion:"
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1080_natural_relevance_atlas"
)
SOURCE_PHASE1079 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1079_output_orthogonal_pattern_atlas"
    / "analysis"
    / "final_summary.json"
)


EVIDENCE_THRESHOLDS = {
    "candidate_accuracy_for_behavior_annotation": 0.70,
    "generation_first_accuracy": 0.50,
    "unit_behavior_support_fraction": 0.75,
    "permutation_p_max": 0.01,
    "minimum_repeated_models_or_pairs": 2,
    "minimum_base_family_top1": 5,
    "minimum_relevance_over_presence_gain": 2,
    "maximum_control_to_relevance_ratio": 1.0,
    "minimum_behavior_families": 5,
    "pre_branch_tolerance": 1e-8,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "Target-cue-matched direct-minus-decoy relevance topology "
        "retrieves at least five of eight base families across independent "
        "splits in at least two models under the exact label null."
    ),
    "P2": (
        "Family-centered relevance topology retrieves at least five of "
        "eight base families in at least two directed cross-model pairs."
    ),
    "P3": (
        "Within-model relevance retrieval exceeds target-cue presence "
        "retrieval by at least two families in at least two models."
    ),
    "P4": (
        "After excluding branch_end and answer_boundary, the downstream "
        "request-end intermediate relevance topology retrieves at least "
        "five of eight families in at least two models."
    ),
    "P5": (
        "Held-out causal_connector is nearest contrast_conjunction in "
        "answer-boundary relevance topology in at least two models and "
        "its global relevance peak is middle-to-late Attention or MLP."
    ),
    "P6": (
        "All branch differences before the branch text are numerically "
        "zero within 1e-8 in every model."
    ),
    "P7": (
        "The pooled median max(surface,shell)-to-relevance ratio is at "
        "most one in at least two models."
    ),
    "P8": (
        "At least five base families receive behavior annotation in at "
        "least two models."
    ),
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


SHELLS = {
    0: (
        "Information: {evidence}\n"
        "Additional note: {branch_text}\n"
        "Question: {request}\n"
        "Return only the missing text."
    ),
    1: (
        "Reference: {evidence}\n"
        "Supplement: {branch_text}\n"
        "Prompt: {request}\n"
        "Provide only the missing text."
    ),
    2: (
        "Evidence: {evidence}\n"
        "Extra statement: {branch_text}\n"
        "Query: {request}\n"
        "Write only the missing text."
    ),
    3: (
        "Context: {evidence}\n"
        "Further note: {branch_text}\n"
        "Request: {request}\n"
        "Emit only the missing text."
    ),
}

NEUTRAL_CUES = (
    "marker", "sample", "record", "entry", "symbol", "tag",
    "example", "signal", "alpha", "beta", "gamma", "delta",
    "circle", "square", "north", "south", "plain", "item",
    "archive", "notice", "figure", "object", "reference",
)

BANNED_MODE_WORDS = (
    "semantic", "index", "reason", "copy", "mode", "option",
    "choices", "candidate", "instruction",
)


def state_factors(state: str) -> tuple[int, str, int, int]:
    match = re.fullmatch(
        r"t([01])_b(infer|decoy|direct)_a([01])_l([01])",
        state,
    )
    if not match:
        raise ValueError(f"invalid state: {state}")
    return (
        int(match.group(1)),
        match.group(2),
        int(match.group(3)),
        int(match.group(4)),
    )


def cue_text(family: str, answer: str) -> str:
    if family == "punctuation_rule":
        return "question mark" if answer == "?" else "period"
    return answer


def encoded_width(tokenizer, value: str) -> int:
    return len(tokenizer.encode(" " + value, add_special_tokens=False))


def choose_neutral_cue(
    tokenizer,
    target: str,
    distractor: str,
) -> str:
    target_width = encoded_width(tokenizer, target)
    forbidden = {target.casefold(), distractor.casefold()}
    candidates = [
        value for value in NEUTRAL_CUES
        if value.casefold() not in forbidden
    ]
    for base in ("marker", "sample", "record", "alpha"):
        for repeat in range(2, 13):
            value = " ".join([base] * repeat)
            if value.casefold() not in forbidden:
                candidates.append(value)
    return min(
        candidates,
        key=lambda value: (
            abs(encoded_width(tokenizer, value) - target_width),
            encoded_width(tokenizer, value),
            value,
        ),
    )


def branch_text(
    branch: str,
    surface: int,
    cue: str,
    neutral: str,
) -> tuple[str, str]:
    shown = neutral if branch == "infer" else cue
    if surface == 0:
        if branch == "direct":
            text = f"For this question, the completion is {shown}."
        else:
            text = (
                f"For an unrelated display, the label is {shown}."
            )
    else:
        if branch == "direct":
            text = (
                f"Regarding the current example, the requested term is "
                f"{shown}."
            )
        else:
            text = (
                f"Regarding a separate example, the recorded term is "
                f"{shown}."
            )
    return text, shown


def build_case(
    tokenizer,
    model_name: str,
    family: str,
    split: str,
    item: tuple[str, ...],
    state: str,
    case_index: int,
) -> dict[str, Any]:
    template_local, branch, answer, surface = state_factors(state)
    template_index = TEMPLATES_BY_SPLIT[split][template_local]
    worlds = source.worlds_for(family, item, surface, split)
    world = worlds[answer]
    answers = [str(value["answer"]) for value in worlds]
    if answers[0].casefold() == answers[1].casefold():
        raise RuntimeError(f"answer collision: {family}/{item[0]}")
    cue = cue_text(family, answers[answer])
    distractor_cue = cue_text(family, answers[1 - answer])
    neutral = choose_neutral_cue(tokenizer, cue, distractor_cue)
    note, shown_cue = branch_text(branch, surface, cue, neutral)
    raw_prompt = SHELLS[template_index].format(
        evidence=world["evidence"],
        branch_text=note,
        request=world["request"],
    )
    evidence_start = raw_prompt.index(world["evidence"])
    local_anchor = source.mark(
        world["evidence"], world["anchor"], occurrence="first"
    )
    raw_spans = {
        "content_anchor": (
            evidence_start + int(local_anchor[0]),
            evidence_start + int(local_anchor[1]),
            str(local_anchor[2]),
        ),
        "context_end": (
            evidence_start,
            evidence_start + len(world["evidence"]),
            str(world["evidence"]),
        ),
        "branch_end": source.mark(
            raw_prompt, shown_cue, occurrence="last"
        ),
        "request_end": source.mark(
            raw_prompt, world["request"], occurrence="last"
        ),
    }
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
    prefix = " "
    candidate_token_ids = {
        f"a{index}": behavior.continuation_ids(
            tokenizer, rendered, prefix, label
        )
        for index, label in enumerate(answers)
    }
    return {
        "schema_version": "phase1080_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": (
            f"{model_name}.{family}.{split}.{item[0]}.{state}"
        ),
        "unit_id": f"{family}.{split}.{item[0]}",
        "family": family,
        "split": split,
        "item_id": str(item[0]),
        "state": state,
        "template_local_branch": template_local,
        "template_index": template_index,
        "branch": branch,
        "answer_branch": answer,
        "surface_branch": surface,
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
        "answer_labels": answers,
        "target_answer": answers[answer],
        "distractor_answer": answers[1 - answer],
        "cue_text": cue,
        "shown_cue": shown_cue,
        "neutral_cue": neutral,
        "cue_token_width": encoded_width(tokenizer, cue),
        "shown_cue_token_width": encoded_width(tokenizer, shown_cue),
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])]
            for key, values in candidate_token_ids.items()
        },
        "expected_class": f"a{answer}",
        "continuation_prefix": prefix,
        "target_cue_matched_direct_decoy": branch in {
            "direct", "decoy"
        },
    }


def audit_model(
    model_name: str,
    tokenizer,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roles_valid = True
    causal_role_order = True
    candidates_disjoint = True
    expected_correct = True
    no_banned_mode_words = True
    direct_decoy_cue_equal = True
    infer_neutral_width_matched = True
    output_matched = True
    branch_prefix_equal = True
    pre_branch_roles_before_divergence = True
    for row in cases:
        by_unit[row["unit_id"]].append(row)
        width = len(row["input_ids"])
        for role in CAPTURE_ROLES:
            start, end = row["role_spans"][role]
            roles_valid &= 0 <= start <= end < width
        causal_role_order &= (
            int(row["role_positions"]["content_anchor"])
            <= int(row["role_positions"]["context_end"])
            < int(row["role_positions"]["branch_end"])
            < int(row["role_positions"]["request_end"])
            < int(row["role_positions"]["answer_boundary"])
        )
        left = set(row["candidate_first_token_ids"]["a0"])
        right = set(row["candidate_first_token_ids"]["a1"])
        candidates_disjoint &= bool(left) and bool(right) and left.isdisjoint(
            right
        )
        expected_correct &= row["expected_class"] == (
            f"a{row['answer_branch']}"
        )
        prompt_folded = row["raw_prompt"].casefold()
        no_banned_mode_words &= not any(
            re.search(rf"\b{re.escape(word)}\b", prompt_folded)
            for word in BANNED_MODE_WORDS
        )
        if row["branch"] == "infer":
            infer_neutral_width_matched &= abs(
                int(row["shown_cue_token_width"])
                - int(row["cue_token_width"])
            ) <= 1

    for rows in by_unit.values():
        lookup = {row["state"]: row for row in rows}
        for template in (0, 1):
            for answer in (0, 1):
                for surface in (0, 1):
                    selected = {
                        branch: lookup[
                            f"t{template}_b{branch}_a{answer}_l{surface}"
                        ]
                        for branch in BRANCHES
                    }
                    direct_decoy_cue_equal &= (
                        selected["direct"]["shown_cue"]
                        == selected["decoy"]["shown_cue"]
                        == selected["direct"]["cue_text"]
                    )
                    output_matched &= len({
                        row["target_answer"]
                        for row in selected.values()
                    }) == 1
                    prefixes = []
                    for row in selected.values():
                        raw = row["raw_prompt"]
                        marker = next(
                            value for value in (
                                "Additional note: ", "Supplement: ",
                                "Extra statement: ", "Further note: ",
                            )
                            if value in raw
                        )
                        raw_prefix = raw[:raw.index(marker) + len(marker)]
                        rendered_prefix = behavior.render_native(
                            tokenizer,
                            model_name,
                            raw_prefix,
                            with_system=False,
                        )
                        prefixes.append(tokenizer.encode(
                            rendered_prefix, add_special_tokens=False
                        ))
                    branch_prefix_equal &= all(
                        value == prefixes[0] for value in prefixes[1:]
                    )
                    token_rows = [
                        row["input_ids"] for row in selected.values()
                    ]
                    common = min(len(value) for value in token_rows)
                    divergence = common
                    for token_index in range(common):
                        if len({
                            value[token_index] for value in token_rows
                        }) > 1:
                            divergence = token_index
                            break
                    pre_branch_roles_before_divergence &= all(
                        int(row["role_positions"][role]) < divergence
                        for row in selected.values()
                        for role in PRE_BRANCH_ROLES
                    )

    item_ids = {
        (family, split): {
            row["item_id"]
            for row in cases
            if row["family"] == family and row["split"] == split
        }
        for family in FAMILIES
        for split in SPLITS
    }
    counts = Counter((row["family"], row["split"]) for row in cases)
    checks = {
        "case_count": len(cases)
        == len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT
        * len(STATES),
        "unit_count": len(by_unit)
        == len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT,
        "complete_units": all(
            len(rows) == len(STATES) for rows in by_unit.values()
        ),
        "family_split_counts": all(
            count == ITEMS_PER_FAMILY_SPLIT * len(STATES)
            for count in counts.values()
        ) and len(counts) == len(FAMILIES) * len(SPLITS),
        "role_spans_valid": roles_valid,
        "causal_role_order_valid": causal_role_order,
        "candidate_first_tokens_disjoint": candidates_disjoint,
        "expected_class_matches_answer": expected_correct,
        "explicit_mode_words_absent": no_banned_mode_words,
        "direct_decoy_target_cue_equal": direct_decoy_cue_equal,
        "infer_neutral_width_matched": infer_neutral_width_matched,
        "branch_outputs_matched": output_matched,
        "branch_prefix_equal_before_divergence": branch_prefix_equal,
        "pre_branch_roles_before_divergence": (
            pre_branch_roles_before_divergence
        ),
        "independent_item_splits": all(
            item_ids[(family, "discovery")].isdisjoint(
                item_ids[(family, "confirmation")]
            )
            for family in FAMILIES
        ),
    }
    return {
        "schema_version": "phase1080_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(cases),
    }


def build_model_cases(model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    case_index = 0
    for family in FAMILIES:
        for split in SPLITS:
            items = source.ITEMS_BY_FAMILY_SPLIT[family][split]
            if len(items) < ITEMS_PER_FAMILY_SPLIT:
                raise RuntimeError(f"insufficient items: {family}/{split}")
            for item in items[:ITEMS_PER_FAMILY_SPLIT]:
                for state in STATES:
                    cases.append(build_case(
                        tokenizer,
                        model_name,
                        family,
                        split,
                        item,
                        state,
                        case_index,
                    ))
                    case_index += 1
    audit = audit_model(model_name, tokenizer, cases)
    return cases, audit


def main() -> None:
    protocol_root = OUT_ROOT / "protocol"
    model_case_digests = {}
    model_audits = {}
    for model_name in MODELS:
        cases, audit = build_model_cases(model_name)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"protocol audit failed for {model_name}: {audit}"
            )
        write_jsonl(
            protocol_root / f"cases.{model_name}.jsonl", cases
        )
        write_json(
            protocol_root / f"audit.{model_name}.json", audit
        )
        model_case_digests[model_name] = audit["case_digest"]
        model_audits[model_name] = audit

    source_summary = read_json(SOURCE_PHASE1079)
    prereg = {
        "schema_version": "phase1080_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "families": list(FAMILIES),
        "base_families": list(BASE_FAMILIES),
        "heldout_family": HELDOUT_FAMILY,
        "splits": list(SPLITS),
        "branches": list(BRANCHES),
        "states": list(STATES),
        "factor_definition": {
            "T": "natural shared shell wording",
            "B": (
                "infer neutral cue, decoy target cue marked unrelated, "
                "or direct same target cue marked relevant"
            ),
            "A": "same family with two answer identities",
            "L": "meaning-preserving evidence and branch paraphrase",
        },
        "capture_roles": list(CAPTURE_ROLES),
        "pre_branch_roles": list(PRE_BRANCH_ROLES),
        "intermediate_roles": list(INTERMEDIATE_ROLES),
        "conditionings": list(CONDITIONINGS),
        "assistant_prefill": ASSISTANT_PREFILL,
        "case_count_per_model": (
            len(FAMILIES) * len(SPLITS)
            * ITEMS_PER_FAMILY_SPLIT * len(STATES)
        ),
        "unit_count_per_model": (
            len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT
        ),
        "model_case_digests": model_case_digests,
        "generation_units_per_family_split_branch": (
            GENERATION_UNITS_PER_FAMILY_SPLIT_BRANCH
        ),
        "generation_steps": GENERATION_STEPS,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1079_protocol_digest": source_summary[
            "protocol_digest"
        ],
        "source_phase1079_summary_digest": source_summary[
            "summary_digest"
        ],
        "primary_population": (
            "All finite preregistered observations; behavior never deletes "
            "a sample from the descriptive map."
        ),
        "secondary_population": (
            "Units with at least 75% correct candidate comparisons across "
            "all infer, decoy, and direct states."
        ),
        "evidence_levels": {
            "L0": "finite relevance, presence, and total fields mapped",
            "L1": (
                "relevance topology retrieves across independent splits "
                "in at least two models"
            ),
            "L2": (
                "family-centered relevance topology repeats across at "
                "least two directed model comparisons"
            ),
            "L3": (
                "L2 plus request-end intermediate relevance retrieval in "
                "at least two models"
            ),
            "L4": "L3 plus behavior annotation in at least two models",
            "L5": "causal support; forbidden in Phase1080",
        },
        "measurement_order": [
            "protocol audits",
            "behavior and finite coverage",
            "target-cue presence field",
            "target-cue-matched relevance field",
            "total direct-minus-infer field",
            "answer modulation and surface/shell controls",
            "independent split retrieval",
            "cross-model retrieval",
            "request-end intermediate exclusion audit",
            "held-out family prediction",
            "automatic gate",
        ],
        "interpretation_limits": [
            "Direct-minus-decoy changes natural relevance wording and is "
            "not a pure latent operation intervention.",
            "Infer, decoy, and direct all retain the same task evidence; "
            "direct availability may reduce but does not erase computation.",
            "Answer-balanced main effects are not difference-in-differences; "
            "the latter measures answer modulation only.",
            "Target cue matching controls cue identity in direct/decoy but "
            "does not control all syntax or semantics.",
            "A family profile can still contain domain, length, frequency, "
            "tokenizer, and request statistics.",
            "Normalized-depth similarity is not physical-coordinate homology.",
            "No Phase1080 observation establishes necessity, sufficiency, "
            "minimality, optimality, or brain-model homology.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "P1-P8 all pass; at least five base families reach L3; "
                "integrity passes; and no protocol leak is found."
            ),
            "next_task_if_passed": (
                "Pre-register component-level tests for the strongest "
                "intermediate relevance family without selecting raw peaks."
            ),
            "stop_if_failed": (
                "Retain the descriptive atlas and redesign the relevance "
                "control; do not select heads or neurons."
            ),
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    audit = {
        "schema_version": "phase1080_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audits": model_audits,
        "checks": {
            "all_model_audits_passed": all(
                row["all_checks_passed"]
                for row in model_audits.values()
            ),
            "model_order_frozen": tuple(prereg["models"]) == MODELS,
            "precision_fp16": prereg["precision"] == "fp16",
            "quantization_none": prereg["quantization"] == "none",
            "explicit_mode_words_forbidden": True,
            "heldout_family_frozen": (
                prereg["heldout_family"] == "causal_connector"
            ),
            "predictions_frozen": (
                set(prereg["prospective_predictions"])
                == set(PROSPECTIVE_PREDICTIONS)
            ),
        },
    }
    audit["all_checks_passed"] = all(audit["checks"].values())
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"global protocol audit failed: {audit}")
    print({
        "phase": PHASE,
        "status": "protocol_frozen",
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "protocol_digest": prereg["protocol_digest"],
    })


if __name__ == "__main__":
    main()
