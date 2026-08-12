#!/usr/bin/env python3
"""Freeze the Phase1078 shared-shell language-pattern atlas protocol.

Phase1078 does not assume a language-mechanism equation.  It first separates
three observable factors that Phase1077 mixed together:

* B: a supported versus unsupported claim about the same item,
* L: two meaning-preserving surface realizations, and
* T: two shared decision shells used by every language family.

All families use the same yes/no output protocol.  Discovery and confirmation
use disjoint items and disjoint shared shells.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior
import phase1077_nonblocking_pattern_atlas_protocol as source


PHASE = 1078
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
FAMILIES = (
    "height_relation",
    "contrast_conjunction",
    "punctuation_rule",
    "taxonomy_fruit",
    "taxonomy_animal",
    "color_property",
    "rare_semantics",
    "translation_equivalence",
)
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
STATES = tuple(
    f"t{template}_b{truth}_l{surface}"
    for template in (0, 1)
    for truth in (0, 1)
    for surface in (0, 1)
)
CAPTURE_ROLES = (
    "evidence_anchor",
    "claim_subject",
    "claim_predicate",
    "decision_cue",
    "answer_boundary",
)
CONDITIONINGS = ("all_finite", "behavior_complete")
ASSISTANT_PREFILL = "Answer:"
ITEMS_PER_FAMILY_SPLIT = 15
NATURAL_AUDIT_CASES_PER_FAMILY_SPLIT = 8
NATURAL_GENERATION_STEPS = 6
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1078_shared_shell_pattern_atlas"
)
SOURCE_PHASE1077 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1077_nonblocking_pattern_atlas"
    / "analysis"
    / "final_summary.json"
)

# These gates classify descriptive repetition.  They do not establish a
# mechanism and cannot assign causal evidence.
EVIDENCE_THRESHOLDS = {
    "candidate_accuracy_for_behavior_annotation": 0.80,
    "natural_semantic_first_for_behavior_annotation": 0.60,
    "permutation_p_max": 0.01,
    "minimum_repeated_models_or_pairs": 2,
    "family_top1_repetitions_min": 2,
    "late_truth_alignment_gap_min": 0.05,
    "translation_direction_repeat_max": 0.20,
    "phase1077_direction_drop_min": 0.10,
    "control_to_truth_ratio_nontrivial_min": 0.25,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "Family-centered discovery-to-confirmation assignment remains above "
        "an exact family-label permutation null after all families share the "
        "same decision shell."
    ),
    "P2": (
        "A generic true/false decision field becomes more aligned across "
        "families near the late answer boundary than at early depth."
    ),
    "P3": (
        "Translation preserves a repeated response topology but has low "
        "discovery-confirmation truth-direction reuse (mean below 0.20)."
    ),
    "P4": (
        "At least one of punctuation or height loses part of its Phase1077 "
        "direction repeat after explicit family labels are removed."
    ),
    "P5": (
        "Shell and surface controls explain a nontrivial part of the raw "
        "family signature; therefore raw response similarity alone will not "
        "be treated as language-mechanism evidence."
    ),
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


SHARED_SHELLS = {
    0: (
        "Evidence: {evidence}\n"
        "Claim: {subject} {predicate}.\n"
        "Decision: Is the claim supported? Reply only yes or no."
    ),
    1: (
        "Given information: {evidence}\n"
        "Statement: {subject} {predicate}.\n"
        "Decision: Does the statement follow? Answer with yes or no only."
    ),
    2: (
        "Reference: {evidence}\n"
        "Proposition: {subject} {predicate}.\n"
        "Assessment: Is this proposition correct? Respond only yes or no."
    ),
    3: (
        "Information supplied: {evidence}\n"
        "Assertion: {subject} {predicate}.\n"
        "Assessment: Is the assertion true? Return yes or no only."
    ),
}
DECISION_CUES = {
    0: "Is the claim supported",
    1: "Does the statement follow",
    2: "Is this proposition correct",
    3: "Is the assertion true",
}


COLOR_DISCOVERY = (
    ("cod01", "banana", "yellow", "blue"),
    ("cod02", "grass", "green", "purple"),
    ("cod03", "snow", "white", "orange"),
    ("cod04", "coal", "black", "pink"),
    ("cod05", "blood", "red", "green"),
    ("cod06", "an orange", "orange", "blue"),
    ("cod07", "a lemon", "yellow", "purple"),
    ("cod08", "a clear daytime sky", "blue", "brown"),
    ("cod09", "milk", "white", "green"),
    ("cod10", "dark chocolate", "brown", "pink"),
    ("cod11", "a ripe strawberry", "red", "gray"),
    ("cod12", "a healthy leaf", "green", "orange"),
    ("cod13", "a flamingo", "pink", "blue"),
    ("cod14", "gold metal", "gold", "silver"),
    ("cod15", "silver metal", "silver", "gold"),
)
COLOR_CONFIRMATION = (
    ("coc01", "a ripe tomato", "red", "purple"),
    ("coc02", "a carrot", "orange", "black"),
    ("coc03", "an eggplant", "purple", "yellow"),
    ("coc04", "a white cloud", "white", "green"),
    ("coc05", "a clear night sky", "black", "orange"),
    ("coc06", "a fresh spinach leaf", "green", "pink"),
    ("coc07", "the open ocean", "blue", "red"),
    ("coc08", "roasted coffee", "brown", "blue"),
    ("coc09", "a red rose", "red", "gray"),
    ("coc10", "a polar bear", "white", "purple"),
    ("coc11", "a sunflower", "yellow", "blue"),
    ("coc12", "wood ash", "gray", "red"),
    ("coc13", "lavender flowers", "purple", "orange"),
    ("coc14", "a ripe lime", "green", "pink"),
    ("coc15", "a ripe blueberry", "blue", "yellow"),
)


ITEMS_BY_FAMILY_SPLIT: dict[str, dict[str, tuple[tuple[str, ...], ...]]] = {
    "height_relation": {
        "discovery": source.HEIGHT_DISCOVERY,
        "confirmation": source.HEIGHT_CONFIRMATION,
    },
    "contrast_conjunction": {
        "discovery": source.CONTRAST_DISCOVERY,
        "confirmation": source.CONTRAST_CONFIRMATION,
    },
    "punctuation_rule": {
        "discovery": source.PUNCTUATION_DISCOVERY,
        "confirmation": source.PUNCTUATION_CONFIRMATION,
    },
    "taxonomy_fruit": {
        "discovery": source.TAXONOMY_DISCOVERY,
        "confirmation": source.TAXONOMY_CONFIRMATION,
    },
    "taxonomy_animal": {
        "discovery": source.TAXONOMY_DISCOVERY,
        "confirmation": source.TAXONOMY_CONFIRMATION,
    },
    "color_property": {
        "discovery": COLOR_DISCOVERY,
        "confirmation": COLOR_CONFIRMATION,
    },
    "rare_semantics": {
        "discovery": source.RARE_DISCOVERY,
        "confirmation": source.RARE_CONFIRMATION,
    },
    "translation_equivalence": {
        "discovery": source.TRANSLATION_DISCOVERY,
        "confirmation": source.TRANSLATION_CONFIRMATION,
    },
}


def state_factors(state: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"t([01])_b([01])_l([01])", state)
    if not match:
        raise ValueError(f"invalid Phase1078 state: {state}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def mark(
    text: str,
    value: str,
    *,
    occurrence: str = "first",
) -> tuple[int, int, str]:
    start = text.find(value) if occurrence == "first" else text.rfind(value)
    if start < 0:
        raise RuntimeError(f"missing marked value {value!r}")
    return start, start + len(value), value


def item_number(item_id: str) -> int:
    match = re.search(r"(\d+)$", item_id)
    if not match:
        raise ValueError(f"item id has no numeric suffix: {item_id}")
    return int(match.group(1))


def height_content(
    item: tuple[str, ...],
    truth: int,
    surface: int,
) -> tuple[str, str, str, str]:
    _, high, middle, low = item
    if surface == 0:
        evidence = (
            f"{high} is taller than {middle}; "
            f"{middle} is taller than {low}"
        )
        subject = high if truth == 0 else low
        predicate = "is the tallest person in this group"
    else:
        evidence = (
            f"{low} is shorter than {middle}; "
            f"{middle} is shorter than {high}"
        )
        subject = high if truth == 0 else low
        predicate = "has the greatest height in this group"
    return evidence, high, subject, predicate


def contrast_content(
    item: tuple[str, ...],
    truth: int,
    surface: int,
) -> tuple[str, str, str, str]:
    item_id, premise, aligned, opposed = item
    uses_contrast = item_number(item_id) % 2 == 1
    continuation = opposed if uses_contrast else aligned
    correct = "but" if uses_contrast else "and"
    incorrect = "and" if uses_contrast else "but"
    candidate = correct if truth == 0 else incorrect
    if surface == 0:
        evidence = f"{premise}, ___ {continuation}"
        subject = "the missing conjunction"
        predicate = f'is "{candidate}"'
    else:
        evidence = (
            f"First clause: {premise}. Second clause: {continuation}. "
            "A connector is missing between them"
        )
        subject = "the connector between the two clauses"
        predicate = f'should be "{candidate}"'
    return evidence, premise, subject, predicate


def punctuation_content(
    item: tuple[str, ...],
    truth: int,
    surface: int,
) -> tuple[str, str, str, str]:
    item_id, statement, question = item
    is_question = item_number(item_id) % 2 == 0
    carrier = question if is_question else statement
    correct = "a question mark" if is_question else "a period"
    incorrect = "a period" if is_question else "a question mark"
    candidate = correct if truth == 0 else incorrect
    if surface == 0:
        evidence = f"Unpunctuated text: {carrier}"
        subject = "the required final punctuation"
        predicate = f"is {candidate}"
    else:
        evidence = f"Words requiring their final mark: {carrier}"
        subject = "the mark that must close this text"
        predicate = f"should be {candidate}"
    return evidence, carrier, subject, predicate


def taxonomy_content(
    item: tuple[str, ...],
    truth: int,
    surface: int,
    *,
    use_animal: bool,
) -> tuple[str, str, str, str]:
    _, fruit, animal = item
    entity = animal if use_animal else fruit
    correct = "animal" if use_animal else "fruit"
    incorrect = "fruit" if use_animal else "animal"
    category = correct if truth == 0 else incorrect
    if surface == 0:
        evidence = f"Item under biological classification: {entity}"
        subject = f"the item named {entity}, for classification,"
        predicate = f"belongs to the {category} category"
    else:
        evidence = f"The object being categorized is {entity}"
        subject = f"the item {entity}"
        predicate = f"is classified as a {category}"
    return evidence, entity, subject, predicate


def fruit_content(
    item: tuple[str, ...],
    truth: int,
    surface: int,
) -> tuple[str, str, str, str]:
    return taxonomy_content(
        item,
        truth,
        surface,
        use_animal=False,
    )


def animal_content(
    item: tuple[str, ...],
    truth: int,
    surface: int,
) -> tuple[str, str, str, str]:
    return taxonomy_content(
        item,
        truth,
        surface,
        use_animal=True,
    )


def color_content(
    item: tuple[str, ...],
    truth: int,
    surface: int,
) -> tuple[str, str, str, str]:
    _, entity, correct, incorrect = item
    color = correct if truth == 0 else incorrect
    if surface == 0:
        evidence = f"Object under ordinary color review: {entity}"
        subject = f"a typical {entity}"
        predicate = f"is {color}"
    else:
        evidence = f"Color reference object: {entity}"
        subject = f"the usual color associated with {entity}"
        predicate = f"is {color}"
    return evidence, entity, subject, predicate


def rare_content(
    item: tuple[str, ...],
    truth: int,
    surface: int,
) -> tuple[str, str, str, str]:
    _, term, definition, true_property, false_property = item
    prop = true_property if truth == 0 else false_property
    if surface == 0:
        evidence = f"{term}: {definition}"
        subject = term
        predicate = f"is associated with {prop}"
    else:
        evidence = f"The glossary describes {term} as {definition}"
        subject = f"the term {term}"
        predicate = f"has the association {prop}"
    return evidence, term, subject, predicate


def translation_content(
    item: tuple[str, ...],
    truth: int,
    surface: int,
) -> tuple[str, str, str, str]:
    _, english, french, _, distractor_french = item
    candidate = french if truth == 0 else distractor_french
    if surface == 0:
        evidence = f"English source word: {english}. Target language: French"
        subject = f"the French equivalent of {english}"
        predicate = f'is "{candidate}"'
    else:
        evidence = (
            f"The source entry {english} is English; "
            "the requested language is French"
        )
        subject = f"{english} rendered in French"
        predicate = f'is "{candidate}"'
    return evidence, english, subject, predicate


BUILDERS: dict[
    str,
    Callable[[tuple[str, ...], int, int], tuple[str, str, str, str]],
] = {
    "height_relation": height_content,
    "contrast_conjunction": contrast_content,
    "punctuation_rule": punctuation_content,
    "taxonomy_fruit": fruit_content,
    "taxonomy_animal": animal_content,
    "color_property": color_content,
    "rare_semantics": rare_content,
    "translation_equivalence": translation_content,
}


def build_model_case(
    tokenizer,
    model_name: str,
    family: str,
    split: str,
    item: tuple[str, ...],
    state: str,
    case_index: int,
) -> dict[str, Any]:
    template_local, truth, surface = state_factors(state)
    template_index = TEMPLATES_BY_SPLIT[split][template_local]
    evidence, evidence_anchor, subject, predicate = BUILDERS[family](
        item,
        truth,
        surface,
    )
    raw_prompt = SHARED_SHELLS[template_index].format(
        evidence=evidence,
        subject=subject,
        predicate=predicate,
    )
    raw_spans = {
        "evidence_anchor": mark(raw_prompt, evidence_anchor),
        "claim_subject": mark(
            raw_prompt,
            subject,
            occurrence="last",
        ),
        "claim_predicate": mark(
            raw_prompt,
            predicate,
            occurrence="last",
        ),
        "decision_cue": mark(
            raw_prompt,
            DECISION_CUES[template_index],
        ),
    }
    rendered = behavior.render_native(
        tokenizer,
        model_name,
        raw_prompt,
        with_system=False,
    )
    rendered += ASSISTANT_PREFILL
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(
        tokenizer,
        rendered,
        raw_prompt,
        raw_spans,
    )
    role_spans["answer_boundary"] = (
        len(input_ids) - 1,
        len(input_ids) - 1,
    )
    classes = {"b0": ["yes"], "b1": ["no"]}
    prefix = " "
    candidate_token_ids = {
        class_name: [
            behavior.continuation_ids(tokenizer, rendered, prefix, label)
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
    expected_class = f"b{truth}"
    return {
        "schema_version": "phase1078_shared_shell_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "semantic_case_index": case_index,
        "record_id": (
            f"{model_name}.{family}.{split}.{item[0]}.{state}"
        ),
        "unit_id": f"{family}.{split}.{item[0]}",
        "family": family,
        "split": split,
        "item_id": str(item[0]),
        "template_local_branch": template_local,
        "template_index": template_index,
        "truth_branch": truth,
        "surface_branch": surface,
        "state": state,
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
        "candidate_labels": classes,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": candidate_first_token_ids,
        "expected_class": expected_class,
        "acceptable_labels": classes[expected_class],
        "continuation_prefix": prefix,
        "claim_truth": truth == 0,
        "shared_shell_id": f"shell_{template_index}",
    }


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter((row["family"], row["split"]) for row in cases)
    state_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roles_valid = True
    roles_physically_distinct = True
    candidate_disjoint = True
    expected_labels_correct = True
    for row in cases:
        state_groups[str(row["unit_id"])].append(row)
        width = len(row["input_ids"])
        positions = []
        for role in CAPTURE_ROLES:
            start, end = row["role_spans"][role]
            roles_valid = roles_valid and 0 <= start <= end < width
            positions.append(int(row["role_positions"][role]))
        roles_physically_distinct = (
            roles_physically_distinct
            and len(set(positions)) == len(positions)
        )
        left = set(row["candidate_first_token_ids"]["b0"])
        right = set(row["candidate_first_token_ids"]["b1"])
        candidate_disjoint = (
            candidate_disjoint
            and bool(left)
            and bool(right)
            and left.isdisjoint(right)
        )
        expected_labels_correct = (
            expected_labels_correct
            and row["expected_class"]
            == ("b0" if row["claim_truth"] else "b1")
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
    shared_shell_coverage = all(
        {
            row["shared_shell_id"]
            for row in cases
            if row["family"] == family
        }
        == {"shell_0", "shell_1", "shell_2", "shell_3"}
        for family in FAMILIES
    )
    checks = {
        "case_count": len(cases)
        == len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT
        * len(STATES),
        "unit_count": len(state_groups)
        == len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT,
        "family_split_case_counts": all(
            counts[(family, split)]
            == ITEMS_PER_FAMILY_SPLIT * len(STATES)
            for family in FAMILIES
            for split in SPLITS
        ),
        "complete_three_factor_units": all(
            {row["state"] for row in values} == set(STATES)
            for values in state_groups.values()
        ),
        "role_spans_valid": roles_valid,
        "role_end_positions_physically_distinct": roles_physically_distinct,
        "candidate_first_tokens_disjoint": candidate_disjoint,
        "claim_truth_matches_expected_class": expected_labels_correct,
        "shared_shells_cover_every_family": shared_shell_coverage,
        "independent_item_splits": all(
            item_ids[(family, "discovery")].isdisjoint(
                item_ids[(family, "confirmation")]
            )
            for family in FAMILIES
        ),
    }
    return {
        "schema_version": "phase1078_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(state_groups),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    if not SOURCE_PHASE1077.exists():
        raise RuntimeError("missing formal Phase1077 final summary")
    source_1077 = read_json(SOURCE_PHASE1077)

    model_audits = {}
    model_case_digests = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = []
        case_index = 0
        for family in FAMILIES:
            for split in SPLITS:
                items = ITEMS_BY_FAMILY_SPLIT[family][split]
                if len(items) != ITEMS_PER_FAMILY_SPLIT:
                    raise RuntimeError(
                        f"{family}/{split} has {len(items)} items"
                    )
                for item in items:
                    for state in STATES:
                        cases.append(build_model_case(
                            tokenizer,
                            model_name,
                            family,
                            split,
                            item,
                            state,
                            case_index,
                        ))
                        case_index += 1
        audit = audit_model(model_name, cases)
        audit["case_digest"] = digest(cases)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"protocol audit failed for {model_name}: {audit}"
            )
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl",
            cases,
        )
        write_json(
            OUT_ROOT / "protocol" / f"audit.{model_name}.json",
            audit,
        )
        model_audits[model_name] = audit
        model_case_digests[model_name] = audit["case_digest"]

    case_count = (
        len(FAMILIES)
        * len(SPLITS)
        * ITEMS_PER_FAMILY_SPLIT
        * len(STATES)
    )
    unit_count = (
        len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT
    )
    payload = {
        "schema_version": "phase1078_shared_shell_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "families": list(FAMILIES),
        "splits": list(SPLITS),
        "templates_by_split": {
            key: list(values)
            for key, values in TEMPLATES_BY_SPLIT.items()
        },
        "states": list(STATES),
        "factor_definition": {
            "T": "shared decision-shell wording",
            "B": "supported (b0) versus unsupported (b1) claim",
            "L": "meaning-preserving content surface realization",
        },
        "capture_roles": list(CAPTURE_ROLES),
        "conditionings": list(CONDITIONINGS),
        "assistant_prefill": ASSISTANT_PREFILL,
        "case_count_per_model": case_count,
        "unit_count_per_model": unit_count,
        "model_case_digests": model_case_digests,
        "natural_audit_cases_per_family_split": (
            NATURAL_AUDIT_CASES_PER_FAMILY_SPLIT
        ),
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "evidence_thresholds": dict(EVIDENCE_THRESHOLDS),
        "prospective_predictions": dict(PROSPECTIVE_PREDICTIONS),
        "source_phase1077_protocol_digest": source_1077[
            "protocol_digest"
        ],
        "source_phase1077_summary_digest": source_1077[
            "summary_digest"
        ],
        "primary_population": (
            "All preregistered finite forward states. Behavior errors do "
            "not delete observations from the descriptive atlas."
        ),
        "secondary_population": (
            "Complete eight-state units for which all yes/no comparisons "
            "are correct; this remains a sensitivity ledger only."
        ),
        "evidence_levels": {
            "L0": "finite three-factor field mapped",
            "L1": (
                "family profile retrieves itself across independent splits "
                "within at least two models"
            ),
            "L2": (
                "raw profile retrieves itself across at least two directed "
                "cross-model comparisons under an exact permutation null"
            ),
            "L3": (
                "family-centered profile also retrieves itself across at "
                "least two directed cross-model comparisons"
            ),
            "L4": (
                "L3 plus behavior annotation in at least two models"
            ),
            "L5": "causal support; forbidden in Phase1078",
        },
        "measurement_order": [
            "freeze shared shells, independent items, roles, and predictions",
            "audit distinct role token positions and yes/no candidates",
            "capture all finite residual, Attention-output, and MLP-output states",
            "measure truth, surface, shell, truth-surface, and truth-shell fields",
            "keep behavior-complete observations in a secondary ledger",
            "compare independent split and cross-model normalized-depth profiles",
            "test profile assignment against all family-label permutations",
            "separate generic late truth alignment from family-centered topology",
            "stop before head, neuron, transport, or causal interpretation",
        ],
        "interpretation_limits": [
            "A truth differential changes claim tokens and is not a pure latent truth variable.",
            "A shared shell removes family-specific instruction formats but not family content.",
            "A repeated profile is a response topology, not a transport path.",
            "A late common truth direction can be output-protocol geometry.",
            "Supplied rare-word definitions test contextual use, not memorized lexical knowledge.",
            "Cross-model repetition is functional, not coordinate homology.",
            "No result establishes minimal coding, optimality, brain homology, or a complete ontology.",
            "No Phase1078 result can establish causal necessity or sufficiency.",
        ],
        "automatic_next": {
            "continue": False,
            "reason": (
                "The phase contains independent confirmation and exact-null "
                "testing. Any component or causal follow-up requires a new "
                "pre-registration and must not be triggered by visual peaks."
            ),
        },
        "model_audits": model_audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json",
        payload,
    )
    write_json(
        OUT_ROOT / "protocol" / "audit.json",
        {
            "schema_version": "phase1078_protocol_audit.v1",
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
        f"Phase{PHASE} protocol {payload['protocol_digest']} "
        f"cases={payload['case_count_per_model']}/model "
        f"units={payload['unit_count_per_model']}/model"
    )


if __name__ == "__main__":
    main()
