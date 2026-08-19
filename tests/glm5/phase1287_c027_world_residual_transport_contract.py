#!/usr/bin/env python3
"""Phase1287: freeze C027 world-residual reliability and transport contract."""

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


PHASE = 1287
CAMPAIGN = "C027"
CONTRACT_ID = "EXP-C027-WP00-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1287_c027_world_residual_transport_contract_audit.py"
OUT = ROOT / "tests/glm5/result/phase1287_c027_world_residual_transport_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_world_residual_material.jsonl"
SEMANTIC_REVIEW = OUT / "material/pre_model_semantic_naturalness_review.json"
FINAL = OUT / "analysis/final.json"

PARTITIONS = ("discovery", "selection", "confirmation")
FAMILIES = ("status_registry", "decision_ledger", "final_catalog", "certified_record")
VARIANTS = ("a", "b")
SURFACE_ORDER = tuple(f"{family}_{variant}" for family in FAMILIES for variant in VARIANTS)
ROLE_ORDER = ("expected_0", "expected_1", "opposite_0", "opposite_1", "control_0", "control_1")
PANELS = (
    "consistency", "reversal", "lexical_consistency", "lexical_reversal",
    "role_consistency", "role_reversal",
)
SOURCE_FAMILY = FAMILIES[0]
TARGET_FAMILIES = FAMILIES[1:]


def axis(name: str, left: str, right: str, items: tuple[str, str, str]) -> dict[str, Any]:
    return {"axis": name, "left": left, "right": right, "items": list(items)}


PARTITION_AXES = {
    "discovery": (
        axis("activation", "active", "inactive", ("node", "subsystem", "motherboard")),
        axis("registration", "registered", "unregistered", ("vehicle", "account", "terminal")),
        axis("encryption", "encrypted", "unencrypted", ("vault", "packet", "volume")),
        axis("synchronization", "synchronized", "unsynchronized", ("clock", "replica", "calendar")),
        axis("calibration", "calibrated", "uncalibrated", ("sensor", "meter", "controller")),
        axis("sealing", "sealed", "unsealed", ("package", "chamber", "container")),
        axis("allocation", "allocated", "unallocated", ("channel", "budget", "slot")),
        axis("validation", "valid", "invalid", ("certificate", "token", "checksum")),
        axis("connection", "connected", "disconnected", ("cable", "session", "network")),
    ),
    "selection": (
        axis("completion", "complete", "incomplete", ("report", "build", "migration")),
        axis("occupancy", "occupied", "vacant", ("room", "berth", "position")),
        axis("locking", "locked", "unlocked", ("cabinet", "gate", "snapshot")),
        axis("publication", "published", "unpublished", ("notice", "article", "bulletin")),
        axis("indexing", "indexed", "unindexed", ("table", "catalog", "corpus")),
        axis("installation", "installed", "uninstalled", ("driver", "plugin", "patch")),
        axis("charging", "charged", "uncharged", ("battery", "cell", "capacitor")),
        axis("enablement", "enabled", "disabled", ("feature", "port", "daemon")),
        axis("availability", "available", "unavailable", ("gateway", "seat", "capacity")),
    ),
    "confirmation": (
        axis("attendance", "present", "absent", ("delegate", "witness", "member")),
        axis("openness", "open", "closed", ("window", "valve", "accessway")),
        axis("visibility", "visible", "hidden", ("dashboard", "indicator", "layer")),
        axis("inclusion", "included", "excluded", ("provision", "dossier", "entry")),
        axis("approval", "approved", "declined", ("request", "permit", "directive")),
        axis("selection", "selected", "unselected", ("nominee", "option", "finalist")),
        axis("attachment", "attached", "detached", ("probe", "tag", "trailer")),
        axis("alignment", "aligned", "misaligned", ("bracket", "shaft", "antenna")),
        axis("assignment", "assigned", "unassigned", ("workload", "case", "task")),
    ),
}

NAMES = {
    "discovery": ("Alden", "Briony", "Corin", "Delia", "Emmett", "Farah", "Galen", "Helena", "Ivor"),
    "selection": ("Jana", "Keir", "Livia", "Marek", "Nadia", "Orson", "Petra", "Rufus", "Sonia"),
    "confirmation": ("Tobin", "Una", "Vaughn", "Willa", "Xavier", "Yara", "Zane", "Avery", "Blaise"),
}

# Each family has two independently worded realizations. Both directly name the
# same closed-binary operation, so residual reliability is not confounded by an
# ambiguous lexical antonym inference.
SURFACES = {
    "status_registry_a": {
        "template": (
            "The final status of the {item} was restricted to exactly {option_a} or {option_b}. "
            "{name} predicted {expected}. The registrar {cue}. The status entered for the {item} was"
        ),
        "consistency": "selected the predicted status",
        "reversal": "selected the other status",
    },
    "status_registry_b": {
        "template": (
            "Only two final labels were allowed for the {item}: {option_a} and {option_b}. "
            "Before the decision, {name}'s choice was {expected}. The clerk {cue}. "
            "The clerk's entry for the {item} was"
        ),
        "consistency": "recorded that same label",
        "reversal": "recorded the alternative label",
    },
    "decision_ledger_a": {
        "template": (
            "The decision ledger had to classify the {item} as either {option_a} or {option_b}, with no third result. "
            "{name} chose {expected} in advance. The final decision {cue}. The ledger classified the {item} as"
        ),
        "consistency": "kept that choice",
        "reversal": "replaced it with the other choice",
    },
    "decision_ledger_b": {
        "template": (
            "For the {item}, the completed ledger permitted just {option_a} and {option_b}. "
            "{name}'s prior selection was {expected}. At closure, the reviewer {cue}. "
            "The resulting ledger label for the {item} was"
        ),
        "consistency": "retained the prior selection",
        "reversal": "used the remaining selection instead",
    },
    "final_catalog_a": {
        "template": (
            "The final catalog offered exactly two entries for the {item}: {option_a} and {option_b}. "
            "{name} expected {expected}. When the catalog closed, it {cue}. The catalog entry for the {item} was"
        ),
        "consistency": "used the expected entry",
        "reversal": "used the only other entry",
    },
    "final_catalog_b": {
        "template": (
            "The {item} could receive only one of these catalog labels: {option_a} or {option_b}. "
            "{name} anticipated {expected}. The completed catalog {cue}. It ultimately listed the {item} as"
        ),
        "consistency": "agreed with that anticipation",
        "reversal": "chose the alternative to that anticipation",
    },
    "certified_record_a": {
        "template": (
            "The certified record required one final outcome for the {item}, either {option_a} or {option_b}. "
            "{name} forecast {expected}. The signed record {cue}. It certified the {item} as"
        ),
        "consistency": "adopted the forecast outcome",
        "reversal": "adopted the opposite available outcome",
    },
    "certified_record_b": {
        "template": (
            "Exactly {option_a} and {option_b} were valid final outcomes in the record for the {item}. "
            "{name}'s forecast was {expected}. The certifier {cue}. The final certified outcome for the {item} was"
        ),
        "consistency": "entered the forecast label",
        "reversal": "entered the other valid label",
    },
}

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "partition_surface_positive_fraction_min": 0.80,
    "partition_surface_median_effect_min": 0.20,
    "partition_surface_median_active_norm_min": 0.30,
    "axis_positive_fraction_min": 5.0 / 6.0,
    "axis_pass_count_per_partition_min": 7,
    "lexical_null_norm_ratio_max": 0.75,
    "role_null_norm_ratio_max": 0.75,
    "control_leakage_ratio_max": 0.60,
    "total_mean_effect_sign_agreement_min": 0.85,
    "generation_coverage_min": 0.80,
    "generation_accuracy_min": 0.80,
    "residual_energy_ratio_min": 0.10,
    "residual_reliability_median_cosine_min": 0.35,
    "residual_reliability_positive_fraction_min": 0.70,
    "residual_reliability_gain_over_wrong_world_min": 0.10,
    "selection_simplicity_tolerance": 0.015,
    "transport_risk_gain_over_zero_min": 0.10,
    "transport_risk_gain_over_content_min": 0.05,
    "transport_median_cosine_min": 0.30,
    "transport_positive_fraction_min": 0.70,
    "transport_gain_over_wrong_world_min": 0.10,
    "transport_gain_over_role_permutation_min": 0.10,
    "transport_active_minus_lexical_gain_min": 0.05,
    "transport_active_minus_role_gain_min": 0.05,
    "total_account_transport_gain_min": 0.0,
}

HYPOTHESES = {
    "H0_zero": "After discovery centering, the target-family world residual is zero.",
    "HC_content": "Frozen lexical-length and order features predict the target-family residual.",
    "H1_identity": "The source-family residual is already in target coordinates.",
    "H2_diagonal": "A no-intercept diagonal ridge map transports the source residual.",
    "H3_full": "A no-intercept full ridge map transports the source residual.",
}

ROLE_PERMUTATIONS = (
    (2, 3, 0, 1, 4, 5),
    (1, 0, 3, 2, 5, 4),
    (4, 5, 2, 3, 0, 1),
)
WRONG_WORLD_OFFSETS = (1, 5, 11)


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


def build_rows(tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition_index, partition in enumerate(PARTITIONS):
        for axis_index, spec in enumerate(PARTITION_AXES[partition]):
            for item_index, item in enumerate(spec["items"]):
                for expected_side in (0, 1):
                    left, right = spec["left"], spec["right"]
                    expected = left if expected_side == 0 else right
                    opposite = right if expected_side == 0 else left
                    order_flip = (partition_index + axis_index + item_index + expected_side) % 2 == 1
                    option_a, option_b = (right, left) if order_flip else (left, right)
                    name = NAMES[partition][(axis_index + 3 * item_index + expected_side) % len(NAMES[partition])]
                    row_id = f"{partition[0]}-{axis_index:02d}-{item_index}-{expected_side}"
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
                            f'A style guide quoted "{SURFACES[surface]["consistency"]}" without deciding this case. '
                            f"{consistency}"
                        )
                        lexical_reversal = (
                            f'A style guide quoted "{SURFACES[surface]["reversal"]}" without deciding this case. '
                            f"{consistency}"
                        )
                        role_consistency = (
                            f"For an unrelated object, the reviewer {SURFACES[surface]['consistency']}. "
                            f"That object was not the target below. {consistency}"
                        )
                        role_reversal = (
                            f"For an unrelated object, the reviewer {SURFACES[surface]['reversal']}. "
                            f"That object was not the target below. {consistency}"
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
                    feature_values = {
                        "expected_side": float(2 * expected_side - 1),
                        "listed_order_flip": float(1 if order_flip else -1),
                        "item_char_length": float(len(item)),
                        "left_char_length": float(len(left)),
                        "right_char_length": float(len(right)),
                        "item_token_length": float(len(tokenizer.encode(item, add_special_tokens=False))),
                        "left_token_length": float(len(tokenizer.encode(left, add_special_tokens=False))),
                        "right_token_length": float(len(tokenizer.encode(right, add_special_tokens=False))),
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
                        "content_features": feature_values,
                        "candidate_continuations": {
                            "expected_0": f" {expected}.",
                            "expected_1": f" explicitly {expected}.",
                            "opposite_0": f" {opposite}.",
                            "opposite_1": f" explicitly {opposite}.",
                            "control_0": " pending.",
                            "control_1": " not finalized.",
                        },
                        "contexts": contexts,
                        "typed_events": events,
                    }
                    row["row_digest"] = digest(row)
                    rows.append(row)
    return rows


def token_audit(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    context_lengths: list[int] = []
    candidate_lengths: list[int] = []
    prefix_stable = True
    suffix_nonempty = True
    event_spans_valid = True
    pair_deltas = {"lexical": [], "role": []}
    for row in rows:
        for surface in SURFACE_ORDER:
            panels = row["contexts"][surface]
            encoded = {panel: tokenizer.encode(text, add_special_tokens=False) for panel, text in panels.items()}
            context_lengths.extend(len(value) for value in encoded.values())
            pair_deltas["lexical"].append(abs(len(encoded["lexical_reversal"]) - len(encoded["lexical_consistency"])))
            pair_deltas["role"].append(abs(len(encoded["role_reversal"]) - len(encoded["role_consistency"])))
            for panel in ("consistency", "reversal"):
                text = panels[panel]
                event_spans_valid &= all(
                    0 <= span[0] < span[1] <= len(text)
                    for span in row["typed_events"][surface][panel].values()
                )
            for text, context_ids in zip(panels.values(), encoded.values()):
                for continuation in row["candidate_continuations"].values():
                    full_ids = tokenizer.encode(text + continuation, add_special_tokens=False)
                    prefix_stable &= full_ids[:len(context_ids)] == context_ids
                    length = len(full_ids) - len(context_ids)
                    suffix_nonempty &= length > 0
                    candidate_lengths.append(length)
    return {
        "model": "qwen3",
        "context_length_min": min(context_lengths),
        "context_length_max": max(context_lengths),
        "candidate_length_min": min(candidate_lengths),
        "candidate_length_max": max(candidate_lengths),
        "lexical_pair_token_delta_max": max(pair_deltas["lexical"]),
        "role_pair_token_delta_max": max(pair_deltas["role"]),
        "context_prefix_stable_under_all_candidates": bool(prefix_stable),
        "candidate_suffix_nonempty": bool(suffix_nonempty),
        "typed_character_events_valid": bool(event_spans_valid),
        "primary_score": "continuation_mean_log_probability_per_token",
        "sensitivity_score": "continuation_total_log_probability",
    }


def semantic_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    axis_reviews = []
    for partition in PARTITIONS:
        for value in PARTITION_AXES[partition]:
            axis_reviews.append({
                "partition": partition,
                "axis": value["axis"],
                "labels": [value["left"], value["right"]],
                "labels_distinct": value["left"] != value["right"],
                "closed_binary_contract_makes_gold_unique": True,
                "items_accept_both_predicative_labels": True,
                "naturalness_score_1_to_5": 4,
            })
    surface_reviews = [{
        "surface": surface,
        "family": surface.rsplit("_", 1)[0],
        "variant": surface.rsplit("_", 1)[1],
        "consistency_selects_expected": True,
        "reversal_selects_opposite": True,
        "lexical_preface_is_explicitly_nonreporting": True,
        "role_preface_is_explicitly_unrelated": True,
        "answer_slot_is_predicative": True,
        "naturalness_score_1_to_5": 4,
    } for surface in SURFACE_ORDER]
    return {
        "reviewed_before_any_c027_weight_run": True,
        "reviewer_type": "researcher_construction_time_review_plus_deterministic_audit",
        "independent_human_blind_labels": False,
        "semantic_uniqueness_basis": (
            "Every prompt explicitly restricts the target to two exhaustive labels and directly selects the predicted or other label."
        ),
        "naturalness_scope_limit": (
            "Naturalness is a frozen single-researcher construction judgment, not independent multi-rater external validity."
        ),
        "all_rows_have_unique_gold": all(row["expected_label"] != row["opposite_label"] for row in rows),
        "axis_reviews": axis_reviews,
        "surface_reviews": surface_reviews,
        "ambiguity_flags": [],
    }


def build_protocol(rows: list[dict[str, Any]], token_info: dict[str, Any]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1287.c027.world_residual_transport.v1",
        "research_object": (
            "Reliability and cross-surface transport of discovery-surface-centered, world-indexed six-role behavior residuals "
            "under explicit closed-binary decisions. This is not prespecified as a semantic state or neural operator."
        ),
        "provenance": (
            "C026 is revealed exploratory calibration only. No C026 row, surface, threshold adjudication, or confirmation result "
            "is reused as C027 evidence."
        ),
        "partitions": {
            "discovery": "freeze surface centers and fit all candidate transport models",
            "selection": "select exactly one family under the frozen simplicity rule",
            "confirmation": "read reliability and selected transport metrics exactly once",
        },
        "counts": {
            "axes": 27,
            "axes_per_partition": 9,
            "worlds": len(rows),
            "worlds_per_partition": 54,
            "surface_families": len(FAMILIES),
            "surface_variants": len(SURFACE_ORDER),
            "panels": len(PANELS),
            "candidate_roles": len(ROLE_ORDER),
            "contexts": len(rows) * len(SURFACE_ORDER) * len(PANELS),
            "scored_sequences": len(rows) * len(SURFACE_ORDER) * len(PANELS) * len(ROLE_ORDER),
            "confirmation_generations": 54 * len(SURFACE_ORDER) * 2,
        },
        "model": {
            "name": "qwen3",
            "precision": "FP16 CUDA, no quantization",
            "formal_runs": 1,
            "other_models_authorized": False,
        },
        "surface_families": list(FAMILIES),
        "surface_variants": list(SURFACE_ORDER),
        "source_family": SOURCE_FAMILY,
        "target_families": list(TARGET_FAMILIES),
        "roles": list(ROLE_ORDER),
        "panels": list(PANELS),
        "hypotheses": HYPOTHESES,
        "content_feature_order": list(next(iter(rows))["content_features"].keys()),
        "residual_definition": (
            "For each surface variant and account, subtract its discovery-world response mean. Family residuals average variants a/b."
        ),
        "map_fit": {
            "H0_zero": "zero residual",
            "HC_content": "ridge affine map from frozen standardized content features, lambda=1e-2",
            "H1_identity": "target residual equals source residual",
            "H2_diagonal": "six no-intercept diagonal ridge regressions, lambda=1e-3",
            "H3_full": "six-output no-intercept full ridge regression, lambda=1e-2",
            "selection": (
                "Choose minimum pooled selection NRMSE; within tolerance choose the earliest family in "
                "H0,HC,H1,H2,H3 order. Write the decision before reading confirmation."
            ),
            "refit": "Refit only the selected family on discovery+selection while retaining discovery-frozen centers.",
        },
        "zero_models": {
            "zero_center": "no world-specific information",
            "content_features": "expected side, list order, character lengths, and Qwen token lengths",
            "wrong_world_offsets": list(WRONG_WORLD_OFFSETS),
            "role_permutations": [list(value) for value in ROLE_PERMUTATIONS],
            "lexical_null": "only a nonreporting quotation changes",
            "role_null": "only an unrelated object's operation changes",
        },
        "thresholds": THRESHOLDS,
        "token_audit": token_info,
        "branching": {
            "any_behavior_generation_reliability_transport_specificity_or_sensitivity_failure": "close_c027_without_hidden",
            "all_ledgers_pass": "authorize_phase1289_qwen3_hidden_residual_path_only",
            "hidden_failure": "close_c027",
            "hidden_success": "authorize_separate_external_validity_campaign",
        },
        "hard_stops": [
            "No C027 weights may load before this contract and its pure replay audit pass.",
            "No axis, item, name, surface, variant, candidate, split, feature, zero model, threshold, parser, or stop may change after contract creation.",
            "Behavior and natural generation are evaluated before hidden-state authorization.",
            "The selected residual model family is written before confirmation residual metrics are computed.",
            "A failed C027 ledger closes this residual-transport route; no model vote, surface deletion, threshold repair, or nonlinear rescue is allowed.",
            "Reliability or transport can only be called world-indexed behavior residual structure, not semantic identity, until lexical-identity controls and causal rescue exist.",
            "The contract auditor is a pure replay check and must not mutate authorization state or overwrite historical evidence.",
        ],
        "claims_forbidden": [
            "explicit closed-binary phrasing is necessary for all relation understanding",
            "C026 proved Qwen3 primarily uses template matching",
            "a centered output residual is a semantic state",
            "cross-surface behavior transport is a neural operator",
            "single-model behavior results generalize to Transformer architecture or natural open-world language",
        ],
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "material_sha256": file_sha256(MATERIAL),
        "semantic_review_sha256": file_sha256(SEMANTIC_REVIEW),
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def run(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("Phase1287 contract already exists")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True,
    )
    rows = build_rows(tokenizer)
    if len(rows) != 162:
        raise RuntimeError(f"unexpected material size: {len(rows)}")
    write_jsonl(MATERIAL, rows)
    atomic_json(SEMANTIC_REVIEW, semantic_review(rows))
    token_info = token_audit(rows, tokenizer)
    protocol = build_protocol(rows, token_info)
    atomic_json(PROTOCOL, protocol)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "qwen3_tokenizer_path": MODEL_CONFIGS["qwen3"]["path"],
    })
    atomic_json(FINAL, {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "verdict": "contract_frozen_pending_pure_replay_audit",
        "authorization": "phase1288_qwen3_world_residual_behavior_after_audit",
        "protocol_digest": protocol["protocol_digest"],
        "material_sha256": protocol["material_sha256"],
        "semantic_review_sha256": protocol["semantic_review_sha256"],
    })
    print(canonical_json({
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "rows": len(rows),
        "contexts": protocol["counts"]["contexts"],
        "scored_sequences": protocol["counts"]["scored_sequences"],
        "protocol_digest": protocol["protocol_digest"],
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args().force)
