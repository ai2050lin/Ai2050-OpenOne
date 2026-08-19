#!/usr/bin/env python3
"""Phase1285: freeze C026 binary-status conditional response mapping contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE = 1285
CAMPAIGN = "C026"
CONTRACT_ID = "EXP-C026-WP00-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1285_c026_conditional_response_mapping_contract_audit.py"
OUT = ROOT / "tests/glm5/result/phase1285_c026_conditional_response_mapping_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_binary_status_worlds.jsonl"
SEMANTIC_REVIEW = OUT / "material/pre_model_semantic_naturalness_review.json"
FINAL = OUT / "analysis/final.json"

MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "selection", "confirmation")
SURFACE_ORDER = ("official_decision", "binary_audit", "closed_review", "signed_assessment")
ROLE_ORDER = ("expected_0", "expected_1", "opposite_0", "opposite_1", "control_0", "control_1")
PANELS = (
    "consistency", "reversal", "lexical_consistency", "lexical_reversal",
    "role_consistency", "role_reversal",
)


def axis(name: str, left: str, right: str, items: tuple[str, str, str, str]) -> dict[str, Any]:
    return {"axis": name, "left": left, "right": right, "items": items}


PARTITION_AXES = {
    "discovery": (
        axis("compliance", "compliant", "noncompliant", ("inspection", "submission", "installation", "procedure")),
        axis("eligibility", "eligible", "ineligible", ("applicant", "candidate", "entrant", "recipient")),
        axis("compatibility", "compatible", "incompatible", ("adapter", "codec", "connector", "interface")),
        axis("authorization", "authorized", "unauthorized", ("transaction", "login", "operation", "transfer")),
        axis("certification", "certified", "uncertified", ("laboratory", "product", "operator", "facility")),
        axis("resolution", "resolved", "unresolved", ("ticket", "dispute", "issue", "incident")),
        axis("verification", "verified", "unverified", ("claim", "identity", "record", "signature")),
        axis("readability", "readable", "unreadable", ("label", "scan", "document", "display")),
    ),
    "selection": (
        axis("reversibility", "reversible", "irreversible", ("change", "treatment", "reaction", "modification")),
        axis("renewability", "renewable", "nonrenewable", ("resource", "supply", "source", "material")),
        axis("taxability", "taxable", "nontaxable", ("payment", "benefit", "purchase", "dividend")),
        axis("refundability", "refundable", "nonrefundable", ("fare", "fee", "deposit", "booking")),
        axis("portability", "portable", "nonportable", ("device", "license", "profile", "subscription")),
        axis("repairability", "repairable", "irreparable", ("screen", "appliance", "mechanism", "circuit")),
        axis("recoverability", "recoverable", "unrecoverable", ("file", "archive", "backup", "dataset")),
        axis("traceability", "traceable", "untraceable", ("shipment", "parcel", "sample", "message")),
    ),
    "confirmation": (
        axis("detectability", "detectable", "undetectable", ("signal", "defect", "marker", "anomaly")),
        axis("enforceability", "enforceable", "unenforceable", ("clause", "agreement", "rule", "order")),
        axis("admissibility", "admissible", "inadmissible", ("evidence", "statement", "testimony", "exhibit")),
        axis("divisibility", "divisible", "indivisible", ("asset", "estate", "quantity", "unit")),
        axis("solubility", "soluble", "insoluble", ("compound", "powder", "coating", "substance")),
        axis("accessibility", "accessible", "inaccessible", ("repository", "entrance", "endpoint", "service")),
        axis("acceptability", "acceptable", "unacceptable", ("variance", "result", "proposal", "condition")),
        axis("removability", "removable", "nonremovable", ("panel", "seal", "module", "fixture")),
    ),
}

NAMES = {
    "discovery": ("Adrian", "Bianca", "Cedric", "Daphne", "Elias", "Fiona", "Gregor", "Hannah"),
    "selection": ("Isabel", "Jonas", "Kendra", "Leandro", "Marta", "Nolan", "Opal", "Pascal"),
    "confirmation": ("Quentin", "Rhea", "Silas", "Talia", "Ulric", "Vera", "Warren", "Zelda"),
}

SURFACES = {
    "official_decision": {
        "template": (
            "The {item} had exactly one of two possible final statuses: {option_a} or {option_b}. "
            "{name} predicted {expected}. The official decision {cue}. "
            "The final record classified the {item} as"
        ),
        "consistency": "matched that prediction",
        "reversal": "ruled out that prediction",
    },
    "binary_audit": {
        "template": (
            "The audit used two mutually exclusive labels for the {item}: {option_a} and {option_b}. "
            "{name}'s forecast was {expected}. The completed audit {cue}. "
            "It labeled the {item} as"
        ),
        "consistency": "confirmed the forecast",
        "reversal": "rejected the forecast",
    },
    "closed_review": {
        "template": (
            "For the {item}, the review could return only {option_a} or {option_b}. "
            "{name} expected {expected}. The final finding {cue}. "
            "The review recorded the {item} as"
        ),
        "consistency": "was consistent with that expectation",
        "reversal": "was inconsistent with that expectation",
    },
    "signed_assessment": {
        "template": (
            "The assessment had to assign the {item} one of two exclusive outcomes, {option_a} or {option_b}. "
            "{name} anticipated {expected}. The signed result {cue}. "
            "It described the {item} as"
        ),
        "consistency": "supported that anticipation",
        "reversal": "contradicted that anticipation",
    },
}

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "partition_surface_positive_fraction_min": 0.80,
    "partition_surface_median_effect_min": 0.20,
    "partition_surface_median_active_norm_min": 0.30,
    "axis_positive_fraction_min": 0.75,
    "axis_pass_count_per_partition_min": 6,
    "lexical_null_norm_ratio_max": 0.75,
    "role_null_norm_ratio_max": 0.75,
    "control_leakage_ratio_max": 0.60,
    "total_mean_effect_sign_agreement_min": 0.85,
    "generation_coverage_min": 0.80,
    "generation_accuracy_min": 0.80,
    "selection_simplicity_tolerance": 0.015,
    "mapping_confirmation_median_cosine_min": 0.80,
    "mapping_confirmation_positive_fraction_min": 0.90,
    "mapping_confirmation_nrmse_max": 0.80,
    "mapping_nrmse_improvement_over_h0_min": 0.08,
    "mapping_nrmse_improvement_over_role_permutation_min": 0.10,
    "mapping_active_gain_min": 0.15,
    "mapping_active_minus_lexical_gain_min": 0.10,
    "mapping_active_minus_role_gain_min": 0.10,
}

HYPOTHESES = {
    "H0_constant": "The target surface response is predicted only by its discovery mean.",
    "H1_identity": "Raw source and target surface response coordinates are equal.",
    "H2_diagonal_affine": "Each registered response role has a surface-specific gain and offset.",
    "H3_full_affine": "A frozen full affine surface map predicts the six-role target response.",
}

ROLE_PERMUTATION_NULLS = (
    (2, 3, 0, 1, 4, 5),
    (1, 0, 3, 2, 5, 4),
    (4, 5, 2, 3, 0, 1),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def core_context(
    surface: str,
    mode: str,
    name: str,
    item: str,
    option_a: str,
    option_b: str,
    expected: str,
) -> tuple[str, dict[str, list[int]]]:
    spec = SURFACES[surface]
    cue = spec[mode]
    context = spec["template"].format(
        item=item, option_a=option_a, option_b=option_b, name=name, expected=expected, cue=cue,
    )
    expected_start = context.find(expected, context.find(name))
    cue_start = context.find(cue)
    return context, {
        "expected_label": [expected_start, expected_start + len(expected)],
        "relation_cue": [cue_start, cue_start + len(cue)],
        "context_end": [len(context) - 2, len(context)],
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition_index, partition in enumerate(PARTITIONS):
        names = NAMES[partition]
        for axis_index, spec in enumerate(PARTITION_AXES[partition]):
            for item_index, item in enumerate(spec["items"]):
                for expected_side in (0, 1):
                    left, right = spec["left"], spec["right"]
                    expected = left if expected_side == 0 else right
                    opposite = right if expected_side == 0 else left
                    order_flip = (partition_index + axis_index + item_index + expected_side) % 2 == 1
                    option_a, option_b = (right, left) if order_flip else (left, right)
                    name = names[(axis_index + 2 * item_index + expected_side) % len(names)]
                    row_id = f"{partition[:1]}-{axis_index:02d}-{item_index}-{expected_side}"
                    contexts: dict[str, dict[str, str]] = {}
                    events: dict[str, dict[str, dict[str, list[int]]]] = {}
                    for surface in SURFACE_ORDER:
                        consistency, consistency_events = core_context(
                            surface, "consistency", name, item, option_a, option_b, expected,
                        )
                        reversal, reversal_events = core_context(
                            surface, "reversal", name, item, option_a, option_b, expected,
                        )
                        lexical_consistency = (
                            f'A training note quoted "{SURFACES[surface]["consistency"]}" as an example, '
                            f"without reporting this case. {consistency}"
                        )
                        lexical_reversal = (
                            f'A training note quoted "{SURFACES[surface]["reversal"]}" as an example, '
                            f"without reporting this case. {consistency}"
                        )
                        role_consistency = (
                            f"In an unrelated case, the result {SURFACES[surface]['consistency']}. "
                            f"That case did not concern the target below. {consistency}"
                        )
                        role_reversal = (
                            f"In an unrelated case, the result {SURFACES[surface]['reversal']}. "
                            f"That case did not concern the target below. {consistency}"
                        )
                        contexts[surface] = {
                            "consistency": consistency,
                            "reversal": reversal,
                            "lexical_consistency": lexical_consistency,
                            "lexical_reversal": lexical_reversal,
                            "role_consistency": role_consistency,
                            "role_reversal": role_reversal,
                        }
                        events[surface] = {
                            "consistency": consistency_events,
                            "reversal": reversal_events,
                        }
                    row = {
                        "row_id": row_id,
                        "partition": partition,
                        "axis": spec["axis"],
                        "item": item,
                        "name": name,
                        "left_label": left,
                        "right_label": right,
                        "expected_side": expected_side,
                        "expected_label": expected,
                        "opposite_label": opposite,
                        "listed_order": [option_a, option_b],
                        "candidate_continuations": {
                            "expected_0": f" {expected}.",
                            "expected_1": f" formally {expected}.",
                            "opposite_0": f" {opposite}.",
                            "opposite_1": f" formally {opposite}.",
                            "control_0": " pending.",
                            "control_1": " under review.",
                        },
                        "expected_terms": [expected],
                        "opposite_terms": [opposite],
                        "contexts": contexts,
                        "typed_events": events,
                    }
                    row["row_digest"] = digest(row)
                    rows.append(row)
    return rows


def load_tokenizers() -> dict[str, Any]:
    tokenizers = {}
    for model_name in MODEL_ORDER:
        tokenizers[model_name] = AutoTokenizer.from_pretrained(
            MODEL_CONFIGS[model_name]["path"], trust_remote_code=True,
            local_files_only=True, use_fast=True,
        )
    return tokenizers


def token_audit(rows: list[dict[str, Any]], tokenizers: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    prefix_stable = True
    all_suffixes_nonempty = True
    for model_name, tokenizer in tokenizers.items():
        continuation_lengths: list[int] = []
        context_lengths: list[int] = []
        lexical_deltas: list[int] = []
        role_deltas: list[int] = []
        event_ends_valid = True
        for row in rows:
            for surface in SURFACE_ORDER:
                panels = row["contexts"][surface]
                encoded_contexts = {
                    panel: tokenizer.encode(context, add_special_tokens=False)
                    for panel, context in panels.items()
                }
                context_lengths.extend(len(value) for value in encoded_contexts.values())
                lexical_deltas.append(abs(len(encoded_contexts["lexical_reversal"]) - len(encoded_contexts["lexical_consistency"])))
                role_deltas.append(abs(len(encoded_contexts["role_reversal"]) - len(encoded_contexts["role_consistency"])))
                for panel in ("consistency", "reversal"):
                    event = row["typed_events"][surface][panel]
                    event_ends_valid &= all(0 <= span[0] < span[1] <= len(panels[panel]) for span in event.values())
                for context in panels.values():
                    context_ids = tokenizer.encode(context, add_special_tokens=False)
                    for continuation in row["candidate_continuations"].values():
                        full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
                        prefix_stable &= full_ids[:len(context_ids)] == context_ids
                        suffix_length = len(full_ids) - len(context_ids)
                        all_suffixes_nonempty &= suffix_length > 0
                        continuation_lengths.append(suffix_length)
        summary[model_name] = {
            "context_length_min": min(context_lengths),
            "context_length_max": max(context_lengths),
            "candidate_length_min": min(continuation_lengths),
            "candidate_length_max": max(continuation_lengths),
            "lexical_pair_context_token_delta_max": max(lexical_deltas),
            "role_pair_context_token_delta_max": max(role_deltas),
            "typed_character_events_valid": bool(event_ends_valid),
        }
    return {
        "models": summary,
        "context_prefix_stable_under_all_candidates_all_models": bool(prefix_stable),
        "candidate_suffix_nonempty_all_models": bool(all_suffixes_nonempty),
        "primary_score": "continuation_mean_log_probability_per_token",
        "secondary_score": "continuation_total_log_probability",
        "length_policy": "Primary inference is token-length normalized; total log probability is a frozen sensitivity account.",
    }


def semantic_review() -> dict[str, Any]:
    axis_reviews = []
    for partition in PARTITIONS:
        for value in PARTITION_AXES[partition]:
            axis_reviews.append({
                "partition": partition,
                "axis": value["axis"],
                "labels": [value["left"], value["right"]],
                "explicit_closed_binary_contract_makes_gold_unique": True,
                "labels_are_distinct": value["left"] != value["right"],
                "all_item_combinations_grammatical": True,
                "naturalness_score_1_to_5": 4,
            })
    surface_reviews = [{
        "surface": surface,
        "consistency_selects_expected_under_closed_binary_contract": True,
        "reversal_selects_only_remaining_label_under_closed_binary_contract": True,
        "answer_slot_is_predicative": True,
        "naturalness_score_1_to_5": 4,
    } for surface in SURFACE_ORDER]
    return {
        "reviewed_before_any_c026_weight_run": True,
        "reviewer_type": "researcher_construction_time_review",
        "independent_human_labels": False,
        "scope_limit": (
            "This is a declared single-researcher pre-model construction review, not an independent human blind annotation. "
            "Logical uniqueness comes from the explicit exactly-one/two-exclusive-status contract; naturalness remains locally curated."
        ),
        "c025_construct_correction": (
            "A contradicted scalar expectation need not entail its lexical antonym. C026 therefore makes the two labels explicit, "
            "mutually exclusive, and exhaustive before applying consistency or reversal cues."
        ),
        "axis_reviews": axis_reviews,
        "surface_reviews": surface_reviews,
        "ambiguity_flags": [],
    }


def build_protocol(rows: list[dict[str, Any]], token_info: dict[str, Any]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1285.c026.conditional_response_mapping.v1",
        "research_object": (
            "World-level six-role response transport across four fixed natural surfaces for explicit mutually exclusive "
            "and exhaustive binary statuses. The object is a behavior-level conditional map, not a hidden semantic vector."
        ),
        "construct_boundary": (
            "C026 corrects C025's non-entailment: contradiction of an expectation alone does not generally entail a lexical antonym."
        ),
        "partition_roles": {
            "discovery": "fit all frozen candidate maps",
            "selection": "choose one map family with the frozen simplicity rule",
            "confirmation": "evaluate the already selected/refit family exactly once",
        },
        "counts": {
            "axes": 24,
            "axes_per_partition": 8,
            "worlds": len(rows),
            "worlds_per_partition": 64,
            "surfaces": len(SURFACE_ORDER),
            "panels": len(PANELS),
            "contexts": len(rows) * len(SURFACE_ORDER) * len(PANELS),
            "candidate_roles": len(ROLE_ORDER),
            "scored_sequences": len(rows) * len(SURFACE_ORDER) * len(PANELS) * len(ROLE_ORDER),
            "confirmation_generations": 64 * len(SURFACE_ORDER) * 2,
        },
        "models": {
            "order": list(MODEL_ORDER),
            "precision": "FP16 CUDA, no quantization",
            "authorization": (
                "Qwen3 runs first. GLM4/DS7B and hidden-state work are denied unless the Qwen3 behavior, generation, "
                "mapping, and specificity ledgers all pass. Models are loaded and released sequentially."
            ),
        },
        "surfaces": list(SURFACE_ORDER),
        "source_surface": SURFACE_ORDER[0],
        "target_surfaces": list(SURFACE_ORDER[1:]),
        "roles": list(ROLE_ORDER),
        "panels": list(PANELS),
        "hypotheses": HYPOTHESES,
        "map_fit": {
            "H0_constant": "target discovery mean",
            "H1_identity": "y_hat=x",
            "H2_diagonal_affine": "six independent ridge regressions with intercept, lambda=1e-3",
            "H3_full_affine": "six-output affine ridge regression with intercept, lambda=1e-2",
            "selection": (
                "Choose minimum selection NRMSE across all target surfaces; if models are within the frozen tolerance, "
                "choose the lowest-complexity family in H0,H1,H2,H3 order."
            ),
            "refit": "Refit only the selected family on discovery+selection before reading confirmation metrics.",
        },
        "zero_models": {
            "constant_target_mean": "tests whether a coarse role-aligned center explains the result",
            "identity": "tests raw response equality",
            "fixed_role_permutations": [list(value) for value in ROLE_PERMUTATION_NULLS],
            "lexical_null": "the reversal phrase changes only inside an explicitly nonreporting quotation",
            "role_null": "an unrelated case changes relation while the target remains consistent",
        },
        "score_accounts": token_info,
        "thresholds": THRESHOLDS,
        "formal_run_budget": {"qwen3": 1, "glm4": 1, "deepseek7b": 1, "qwen3_hidden": 1},
        "branching": {
            "qwen_fail": "stop_c026_at_qwen_behavior_mapping",
            "qwen_all_ledgers_pass": "authorize_qwen3_hidden_conditional_mapping_only",
            "hidden_fail": "stop_c026_at_qwen_hidden",
            "hidden_pass": "authorize_sequential_glm4_then_deepseek_behavior_external_validity",
        },
        "hard_stops": [
            "No C026 model weights may be loaded before the contract and independent preaudit pass.",
            "All material, partitions, surfaces, labels, role coordinates, hypotheses, thresholds, and parsers are frozen before Qwen3.",
            "Discovery fits maps; selection chooses one family; confirmation is read only after the selection decision artifact is written.",
            "No failed surface, axis, world, null, generation, or score account may be removed after unblinding.",
            "Behavior-level mapping may be described after scoring, but hidden-state hooks require all four Qwen3 ledgers to pass.",
            "Any Qwen3 ledger failure ends C026; GLM4 and DeepSeek7B may not vote to rescue it.",
            "A generic surface map that predicts nulls as well as active responses is not a semantic mapping.",
            "A single-researcher naturalness review cannot be called independent human validation.",
            "C026 is a one-shot adjudication of the conditional-map candidate, not a continuation of C025 threshold repair.",
        ],
        "claims_forbidden": [
            "C025 proved a stable cross-surface response signature",
            "C025 generation failure proved Qwen3 lacks relation understanding",
            "three researcher-built campaigns establish the intrinsic granularity of Transformers",
            "a fitted behavior map is an internal mechanism or causal operator",
            "explicit binary-label findings automatically generalize to open-ended lexical antonyms",
            "passing fixed surfaces proves transfer to an unseen surface family",
        ],
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "material_sha256": file_sha256(MATERIAL),
        "semantic_review_sha256": file_sha256(SEMANTIC_REVIEW),
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def run(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("Phase1285 contract already exists")
    rows = build_rows()
    if len(rows) != 192:
        raise RuntimeError(f"unexpected row count: {len(rows)}")
    review = semantic_review()
    write_jsonl(MATERIAL, rows)
    atomic_json(SEMANTIC_REVIEW, review)
    tokenizers = load_tokenizers()
    token_info = token_audit(rows, tokenizers)
    protocol = build_protocol(rows, token_info)
    atomic_json(PROTOCOL, protocol)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "tokenizer_paths": {name: MODEL_CONFIGS[name]["path"] for name in MODEL_ORDER},
    })
    atomic_json(FINAL, {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "verdict": "pending_independent_contract_audit",
        "authorization": "pending_independent_phase1285_audit",
        "protocol_digest": protocol["protocol_digest"],
        "material_sha256": protocol["material_sha256"],
        "semantic_review_sha256": protocol["semantic_review_sha256"],
    })
    print(canonical_json({
        "phase": PHASE,
        "rows": len(rows),
        "contexts": protocol["counts"]["contexts"],
        "scored_sequences": protocol["counts"]["scored_sequences"],
        "protocol_digest": protocol["protocol_digest"],
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.force)
