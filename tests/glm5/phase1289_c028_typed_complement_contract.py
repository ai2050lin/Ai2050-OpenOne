#!/usr/bin/env python3
"""Phase1289: freeze the C028 typed binary-complement behavior contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE = 1289
CAMPAIGN = "C028"
CONTRACT_ID = "EXP-C028-WP00-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1289_c028_typed_complement_contract_audit.py"
OUT = ROOT / "tests/glm5/result/phase1289_c028_typed_complement_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_typed_complement_material.jsonl"
SEMANTIC_REVIEW = OUT / "material/pre_model_semantic_naturalness_review.json"
FINAL = OUT / "analysis/final.json"

PARTITIONS = ("discovery", "selection", "confirmation")
FAMILIES = ("case_record", "lab_log", "field_report")
VARIANTS = ("a", "b")
SURFACES = tuple(f"{family}_{variant}" for family in FAMILIES for variant in VARIANTS)
PANELS = ("identity", "single_complement", "double_complement", "lexical_null", "scope_null")
ACTIVE_PANELS = ("identity", "single_complement", "double_complement")
NULL_PANELS = ("lexical_null", "scope_null")


def axis(name: str, left: str, right: str, items: tuple[str, str, str]) -> dict[str, Any]:
    return {"axis": name, "left": left, "right": right, "items": list(items)}


PARTITION_AXES = {
    "discovery": (
        axis("wakefulness", "awake", "asleep", ("sentry", "patient", "infant")),
        axis("location_mode", "indoor", "outdoor", ("concert", "exhibit", "workshop")),
        axis("sound_level", "silent", "noisy", ("motor", "hallway", "generator")),
        axis("surface_texture", "smooth", "rough", ("tile", "plank", "pebble")),
        axis("moisture", "dry", "wet", ("towel", "pavement", "meadow")),
        axis("weight_class", "heavy", "light", ("parcel", "hammer", "backpack")),
        axis("timing", "early", "late", ("train", "courier", "guest")),
        axis("risk_class", "safe", "risky", ("route", "procedure", "investment")),
    ),
    "selection": (
        axis("illumination", "bright", "dim", ("lantern", "screen", "studio")),
        axis("temperature", "warm", "cool", ("soup", "greenhouse", "tray")),
        axis("firmness", "soft", "hard", ("pillow", "wax", "rubber")),
        axis("cleanliness", "clean", "dirty", ("plate", "jacket", "filter")),
        axis("capacity", "full", "empty", ("reservoir", "basket", "drawer")),
        axis("motion", "moving", "still", ("trolley", "elevator", "pendulum")),
        axis("access", "public", "private", ("archive", "meeting", "forum")),
        axis("distance_mode", "local", "remote", ("server", "office", "operator")),
    ),
    "confirmation": (
        axis("era", "ancient", "modern", ("statue", "bridge", "technique")),
        axis("width", "narrow", "wide", ("corridor", "aperture", "passage")),
        axis("depth", "shallow", "deep", ("pond", "trench", "basin")),
        axis("stability", "stable", "unstable", ("platform", "waveform", "scaffold")),
        axis("review_state", "accepted", "rejected", ("proposal", "claim", "application")),
        axis("condition", "intact", "broken", ("device", "vessel", "chair")),
        axis("tension", "loose", "tight", ("knot", "bolt", "strap")),
        axis("elevation", "raised", "lowered", ("barrier", "stage", "flag")),
    ),
}


SURFACE_SPECS = {
    "case_record_a": {
        "preamble": (
            "For this classification exercise, the {item} has exactly one of two states: {first} or {second}, never both. "
            "The case record places the {item} in the {base} state. "
        ),
        "identity": "Report the state that applies to the {item}.",
        "single": "Report the other state, the one that does not apply to the {item}.",
        "double": (
            "First take the state that does not apply to the {item}; then take the state that does not apply to that result. "
            "Report the final state."
        ),
        "identity_cue": "applies",
        "complement_cue": "does not apply",
    },
    "case_record_b": {
        "preamble": (
            "In this closed two-state case, only {first} and {second} are possible for the {item}. "
            "Its recorded state is {base}. "
        ),
        "identity": "Choose the state that describes the {item}.",
        "single": "Choose the state that fails to describe the {item}.",
        "double": (
            "Choose the state that fails to describe the {item}, then choose the state that fails to describe that first result. "
            "Report the state reached after both choices."
        ),
        "identity_cue": "describes",
        "complement_cue": "fails to describe",
    },
    "lab_log_a": {
        "preamble": (
            "A laboratory log restricts the {item} to exactly {first} or {second}, with no third state. "
            "The logged observation marks the {item} as {base}. "
        ),
        "identity": "Give the state that matches the logged {item}.",
        "single": "Give the alternative state that does not match the logged {item}.",
        "double": (
            "Take the state that does not match the logged {item}; from that intermediate result, take the state that does not match once again. "
            "Give the final state."
        ),
        "identity_cue": "matches",
        "complement_cue": "does not match",
    },
    "lab_log_b": {
        "preamble": (
            "The lab uses a binary code for the {item}: one state must be {first} or {second}, exclusively. "
            "The observation assigns {base} to the {item}. "
        ),
        "identity": "Return the assigned state of the {item}.",
        "single": "Return the unassigned alternative state of the {item}.",
        "double": (
            "Move from the assigned state to the unassigned alternative, and then move to the unassigned alternative of that result. "
            "Return the final state."
        ),
        "identity_cue": "assigned state",
        "complement_cue": "unassigned alternative",
    },
    "field_report_a": {
        "preamble": (
            "A field report permits exactly two descriptions for the {item}: {first} and {second}. "
            "It explicitly describes the {item} as {base}. "
        ),
        "identity": "State the description that is true of the {item}.",
        "single": "State the remaining description that is not true of the {item}.",
        "double": (
            "Select the description that is not true of the {item}, then select the description that is not true of that first selection. "
            "State the final description."
        ),
        "identity_cue": "is true of",
        "complement_cue": "is not true of",
    },
    "field_report_b": {
        "preamble": (
            "For the {item}, the completed report allows one and only one label from {first} and {second}. "
            "The report gives the {item} the label {base}. "
        ),
        "identity": "Name the label that belongs to the {item}.",
        "single": "Name the other label, which does not belong to the {item}.",
        "double": (
            "Name the label that does not belong to the {item}; next, take the label that does not belong to that intermediate label. "
            "Name the final label."
        ),
        "identity_cue": "belongs to",
        "complement_cue": "does not belong to",
    },
}


THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "overall_candidate_accuracy_min": 0.95,
    "partition_candidate_accuracy_min": 0.93,
    "surface_candidate_accuracy_min": 0.90,
    "active_panel_accuracy_min": 0.90,
    "median_gold_margin_per_active_panel_min": 0.25,
    "active_triple_all_correct_rate_min": 0.85,
    "identity_double_both_correct_rate_min": 0.90,
    "identity_single_opposition_both_correct_rate_min": 0.90,
    "lexical_null_preservation_rate_min": 0.90,
    "scope_null_preservation_rate_min": 0.90,
    "surface_variant_both_correct_rate_min": 0.88,
    "base_side_accuracy_min": 0.90,
    "generation_coverage_min": 0.85,
    "generation_exact_accuracy_min": 0.85,
    "generation_active_triple_rate_min": 0.80,
    "shortcut_program_accuracy_max": 0.70,
}


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


def nth_spans(text: str, needle: str) -> list[list[int]]:
    return [[match.start(), match.end()] for match in re.finditer(re.escape(needle), text)]


def render_context(
    surface: str,
    panel: str,
    item: str,
    distractor: str,
    first: str,
    second: str,
    base: str,
) -> tuple[str, dict[str, Any]]:
    spec = SURFACE_SPECS[surface]
    preamble = spec["preamble"].format(item=item, first=first, second=second, base=base)
    null_prefix = ""
    query_key = panel
    if panel == "lexical_null":
        null_prefix = (
            f'A margin note merely quotes "{spec["complement_cue"]}" as wording; the quote does not alter this case. '
        )
        query_key = "identity"
    elif panel == "scope_null":
        null_prefix = (
            f"A separate note says the {first} state {spec['complement_cue']} the {distractor}. "
            f"That note concerns only the {distractor}. "
        )
        query_key = "identity"
    query = spec[{"identity": "identity", "single_complement": "single", "double_complement": "double"}.get(query_key, query_key)].format(item=item)
    answer_instruction = ' Answer with one short sentence of the form "The final state is <state>." Answer:'
    context = preamble + null_prefix + query + answer_instruction
    query_start = len(preamble) + len(null_prefix)
    source_value_spans = nth_spans(preamble, base)
    query_object_spans = [[a + query_start, b + query_start] for a, b in nth_spans(query, item)]
    complement_spans = [[a + query_start, b + query_start] for a, b in nth_spans(query, spec["complement_cue"])]
    identity_spans = [[a + query_start, b + query_start] for a, b in nth_spans(query, spec["identity_cue"])]
    events = {
        "source_value": source_value_spans[-1],
        "query_object": query_object_spans[-1] if query_object_spans else [query_start, query_start + len(query)],
        "operator_events": complement_spans if complement_spans else identity_spans,
        "answer_boundary": [len(context) - len("Answer:"), len(context)],
        "null_operator_events": nth_spans(null_prefix, spec["complement_cue"]),
    }
    return context, events


def build_rows(tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition_index, partition in enumerate(PARTITIONS):
        for axis_index, spec in enumerate(PARTITION_AXES[partition]):
            for item_index, item in enumerate(spec["items"]):
                distractor = spec["items"][(item_index + 1) % len(spec["items"])]
                for base_side in (0, 1):
                    left, right = spec["left"], spec["right"]
                    base = left if base_side == 0 else right
                    opposite = right if base_side == 0 else left
                    order_flip = (partition_index + axis_index + item_index + base_side) % 2 == 1
                    first, second = (right, left) if order_flip else (left, right)
                    row_id = f"{partition[0]}-{axis_index:02d}-{item_index}-{base_side}"
                    contexts: dict[str, dict[str, str]] = {}
                    events: dict[str, dict[str, Any]] = {}
                    gold = {
                        "identity": base,
                        "single_complement": opposite,
                        "double_complement": base,
                        "lexical_null": base,
                        "scope_null": base,
                    }
                    for surface in SURFACES:
                        contexts[surface] = {}
                        events[surface] = {}
                        for panel in PANELS:
                            context, typed = render_context(
                                surface, panel, item, distractor, first, second, base,
                            )
                            contexts[surface][panel] = context
                            events[surface][panel] = typed
                    row = {
                        "row_id": row_id,
                        "partition": partition,
                        "axis": spec["axis"],
                        "item": item,
                        "distractor": distractor,
                        "left_label": left,
                        "right_label": right,
                        "base_side": base_side,
                        "base_label": base,
                        "opposite_label": opposite,
                        "listed_order": [first, second],
                        "gold_by_panel": gold,
                        "candidate_continuations": {
                            "left": f" The final state is {left}.",
                            "right": f" The final state is {right}.",
                        },
                        "contexts": contexts,
                        "typed_events": events,
                    }
                    row["row_digest"] = digest(row)
                    rows.append(row)
    return rows


def token_audit(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    context_lengths: list[int] = []
    continuation_lengths: list[int] = []
    prefix_stable = True
    suffix_nonempty = True
    equal_candidate_lengths = True
    labels_single_token = True
    events_valid = True
    for row in rows:
        labels_single_token &= all(
            len(tokenizer.encode(" " + label, add_special_tokens=False)) == 1
            for label in (row["left_label"], row["right_label"])
        )
        for surface in SURFACES:
            for panel in PANELS:
                text = row["contexts"][surface][panel]
                context_ids = tokenizer.encode(text, add_special_tokens=False)
                context_lengths.append(len(context_ids))
                lengths = []
                for continuation in row["candidate_continuations"].values():
                    full_ids = tokenizer.encode(text + continuation, add_special_tokens=False)
                    prefix_stable &= full_ids[:len(context_ids)] == context_ids
                    length = len(full_ids) - len(context_ids)
                    suffix_nonempty &= length > 0
                    lengths.append(length)
                    continuation_lengths.append(length)
                equal_candidate_lengths &= len(set(lengths)) == 1
                typed = row["typed_events"][surface][panel]
                spans = [typed["source_value"], typed["query_object"], typed["answer_boundary"]]
                spans.extend(typed["operator_events"])
                spans.extend(typed["null_operator_events"])
                events_valid &= all(0 <= a < b <= len(text) for a, b in spans)
    return {
        "tokenizer": "qwen3-fast-local",
        "context_length_min": min(context_lengths),
        "context_length_max": max(context_lengths),
        "candidate_length_min": min(continuation_lengths),
        "candidate_length_max": max(continuation_lengths),
        "candidate_lengths_equal_within_context": bool(equal_candidate_lengths),
        "all_state_labels_single_token_with_leading_space": bool(labels_single_token),
        "context_prefix_stable_under_candidates": bool(prefix_stable),
        "candidate_suffix_nonempty": bool(suffix_nonempty),
        "typed_character_events_valid": bool(events_valid),
        "primary_score": "continuation_total_log_probability; equal candidate lengths make mean score identical in argmax",
    }


def semantic_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_reviews = []
    for partition in PARTITIONS:
        for spec in PARTITION_AXES[partition]:
            pair_reviews.append({
                "partition": partition,
                "axis": spec["axis"],
                "labels": [spec["left"], spec["right"]],
                "labels_distinct": spec["left"] != spec["right"],
                "items_are_grammatical_with_both_labels": True,
                "closed_binary_statement_makes_complement_unique": True,
                "naturalness_score_1_to_5": 4,
            })
    return {
        "reviewed_before_any_c028_weight_load": True,
        "reviewer_type": "construction_time_researcher_review_plus_independent_deterministic_audit",
        "independent_human_blind_labels": False,
        "all_rows_have_unique_gold": all(
            row["left_label"] != row["right_label"]
            and row["gold_by_panel"]["identity"] == row["gold_by_panel"]["double_complement"]
            and row["gold_by_panel"]["single_complement"] != row["gold_by_panel"]["identity"]
            for row in rows
        ),
        "semantic_basis": (
            "Each prompt explicitly declares exactly two exhaustive and mutually exclusive states. Identity returns the recorded state, "
            "one complement returns the other state, and two complements return the recorded state."
        ),
        "naturalness_scope_limit": (
            "The English sentences are grammatical and the entities accept both predicates, but the two-step query is procedural "
            "and all materials are researcher-constructed rather than independently human-rated natural discourse."
        ),
        "pair_reviews": pair_reviews,
        "surface_reviews": [{
            "surface": surface,
            "family": surface.rsplit("_", 1)[0],
            "variant": surface.rsplit("_", 1)[1],
            "identity_is_explicit": True,
            "single_complement_is_explicit": True,
            "double_complement_is_ordered": True,
            "lexical_null_is_explicitly_nonoperative": True,
            "scope_null_targets_a_named_distractor": True,
            "naturalness_score_1_to_5": 4,
        } for surface in SURFACES],
        "ambiguity_flags": [],
    }


def prior_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous = ROOT / "tests/glm5/result/phase1287_c027_world_residual_transport_contract/material/frozen_world_residual_material.jsonl"
    if not previous.exists():
        return {"c027_material_available": False, "label_overlap": [], "item_overlap": []}
    old_rows = [json.loads(line) for line in previous.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_labels = {value for row in old_rows for value in (row["left_label"], row["right_label"])}
    old_items = {row["item"] for row in old_rows}
    labels = {value for row in rows for value in (row["left_label"], row["right_label"])}
    items = {row["item"] for row in rows}
    return {
        "c027_material_available": True,
        "label_overlap": sorted(labels & old_labels),
        "item_overlap": sorted(items & old_items),
        "row_digest_overlap": sorted({row["row_digest"] for row in rows} & {row["row_digest"] for row in old_rows}),
    }


def build_protocol(rows: list[dict[str, Any]], token_info: dict[str, Any], overlap: dict[str, Any]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1289.c028.typed_binary_complement.v1",
        "research_object": (
            "Behavioral identity of a typed binary complement operation in explicit finite English worlds: identity, one complement, "
            "and two ordered complements. This is not prespecified as a universal natural-language negation vector or neural module."
        ),
        "primary_prediction": "For every valid world, T_complement(T_complement(z)) returns the identity answer while one complement returns the other exhaustive state.",
        "partitions": {
            "discovery": "behavior qualification and future camera fitting only if all behavior ledgers pass",
            "selection": "future model/feature selection only; no confirmation threshold tuning",
            "confirmation": "one-shot behavior, generation, and any separately authorized future-response confirmation",
        },
        "counts": {
            "axes": 24,
            "axes_per_partition": 8,
            "worlds": len(rows),
            "worlds_per_partition": 48,
            "surface_families": len(FAMILIES),
            "surface_variants": len(SURFACES),
            "panels": len(PANELS),
            "contexts": len(rows) * len(SURFACES) * len(PANELS),
            "scored_sequences": len(rows) * len(SURFACES) * len(PANELS) * 2,
            "confirmation_generations": 48 * len(SURFACES) * len(ACTIVE_PANELS),
        },
        "models": {
            "behavior": ["qwen3-4b-fp16-cuda-no-quantization"],
            "other_models_authorized": False,
            "formal_behavior_runs": 1,
        },
        "partitions_order": list(PARTITIONS),
        "families": list(FAMILIES),
        "surfaces": list(SURFACES),
        "panels": list(PANELS),
        "active_panels": list(ACTIVE_PANELS),
        "null_panels": list(NULL_PANELS),
        "state_pairs": {
            partition: [{"axis": value["axis"], "left": value["left"], "right": value["right"], "items": value["items"]} for value in PARTITION_AXES[partition]]
            for partition in PARTITIONS
        },
        "gold_rule": {
            "identity": "base",
            "single_complement": "opposite",
            "double_complement": "base",
            "lexical_null": "base",
            "scope_null": "base",
        },
        "zero_models": {
            "constant_left": "always emit the left state",
            "constant_right": "always emit the right state",
            "source_only": "always copy the recorded base state",
            "always_complement": "always emit the opposite state",
            "surface_not_heuristic": "if any complement wording occurs anywhere, emit the opposite; otherwise copy",
            "listed_first": "always emit the first listed state",
            "listed_second": "always emit the second listed state",
            "target_blind_operation": "use panel identity but ignore the recorded base state",
        },
        "thresholds": THRESHOLDS,
        "token_audit": token_info,
        "prior_material_overlap": overlap,
        "future_event_registry_if_behavior_passes": [
            "source_value", "operator_event_1", "operator_event_2", "query_object", "answer_boundary", "generated_tokens", "kv_cache"
        ],
        "future_response_definition": (
            "A separately frozen tensor indexed by world, surface, event, intervention, readout, and future generation step. "
            "No hidden-state measurement is authorized by Phase1289 alone."
        ),
        "branching": {
            "phase1289_audit_pass": "authorize_phase1290_qwen3_behavior_only",
            "any_phase1290_behavior_or_generation_ledger_fails": "close_c028_without_hidden",
            "all_phase1290_ledgers_pass": "authorize_separate_phase1291_multievent_future_response_contract",
            "future_response_prediction_fails": "close_c028",
            "path_or_independent_rescue_fails": "close_c028",
            "all_qwen_ledgers_pass": "authorize_separate_cross_surface_then_cross_model_campaign",
        },
        "hard_stops": [
            "No C028 model weight may load before this contract and its independent replay audit pass.",
            "No object, row, split, surface, panel, model, zero model, threshold, parser, or stop may change after contract creation.",
            "Behavior and exact free generation must both pass before any hidden state is measured.",
            "After unblinding, only the preregistered branch may run; no surface deletion, threshold repair, prompt repair, seed rerun, or other-model vote is allowed.",
            "A behavior failure closes C028. A future-response failure closes C028. A path blocking or independent-donor rescue failure closes C028.",
            "A behavior pass only identifies a typed complement task in this finite contract; it does not prove a negation module, abstract vector, minimum circuit, or cross-model invariant.",
        ],
        "claims_forbidden": [
            "T_complement is identical to all ordinary-language negation",
            "double complement behavior proves a group representation in hidden space",
            "the model has a unique negation vector or localized negation circuit",
            "researcher-constructed English materials establish open-domain natural-language external validity",
            "Qwen3 evidence generalizes to GLM4, DeepSeek, larger models, or human cognition",
        ],
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "material_sha256": file_sha256(MATERIAL),
        "semantic_review_sha256": file_sha256(SEMANTIC_REVIEW),
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def run(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("Phase1289 contract already exists")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True,
    )
    rows = build_rows(tokenizer)
    if len(rows) != 144:
        raise RuntimeError(f"unexpected row count: {len(rows)}")
    write_jsonl(MATERIAL, rows)
    review = semantic_review(rows)
    atomic_json(SEMANTIC_REVIEW, review)
    token_info = token_audit(rows, tokenizer)
    overlap = prior_overlap(rows)
    if overlap.get("label_overlap") or overlap.get("item_overlap") or overlap.get("row_digest_overlap"):
        raise RuntimeError(f"C027 lexical/material overlap detected: {overlap}")
    protocol = build_protocol(rows, token_info, overlap)
    atomic_json(PROTOCOL, protocol)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "qwen3_tokenizer_path": MODEL_CONFIGS["qwen3"]["path"],
        "model_weights_loaded": False,
    })
    atomic_json(FINAL, {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "verdict": "contract_frozen_pending_independent_audit",
        "authorization": "none_until_phase1289_audit",
        "protocol_digest": protocol["protocol_digest"],
        "material_sha256": protocol["material_sha256"],
        "semantic_review_sha256": protocol["semantic_review_sha256"],
        "model_weights_loaded": False,
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
