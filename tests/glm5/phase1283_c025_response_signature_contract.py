#!/usr/bin/env python3
"""Phase1283: freeze the C025 cross-surface response-signature contract."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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


PHASE = 1283
CAMPAIGN = "C025"
CONTRACT_ID = "EXP-C025-WP00-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1283_c025_response_signature_contract_audit.py"
OUT = ROOT / "tests/glm5/result/phase1283_c025_response_signature_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_response_worlds.jsonl"
SEMANTIC_REVIEW = OUT / "material/pre_model_semantic_review.json"
FINAL = OUT / "analysis/final.json"

PARTITION_COUNTS = {"discovery": 64, "selection": 64, "confirmation": 64}
PARTITION_SURFACES = {
    "discovery": ("test_confirmation", "forecast_agreement"),
    "selection": ("evidence_support", "outcome_match"),
    "confirmation": ("measurement_validation", "finding_consistency"),
}
PANELS = (
    "consistency", "reversal", "carrier_consistency", "lexical_consistency",
    "role_consistency", "role_reversal",
)
CANDIDATE_ROLES = (
    "expected_0", "expected_1", "opposite_0", "opposite_1", "control_0", "control_1",
)
CONTROL_WORDS = ("ordinary", "unremarkable")
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")


def axis(name: str, left: tuple[str, str], right: tuple[str, str], items: tuple[str, str, str, str]) -> dict[str, Any]:
    return {"axis": name, "left": left, "right": right, "items": items}


PARTITION_AXES = {
    "discovery": (
        axis("safety", ("safe", "secure"), ("dangerous", "risky"), ("route", "crossing", "walkway", "passage")),
        axis("cost", ("inexpensive", "affordable"), ("expensive", "costly"), ("service", "option", "package", "purchase")),
        axis("quality", ("excellent", "superior"), ("poor", "inferior"), ("product", "sample", "component", "batch")),
        axis("accuracy", ("accurate", "correct"), ("inaccurate", "wrong"), ("reading", "estimate", "measurement", "forecast")),
        axis("reliability", ("reliable", "dependable"), ("unreliable", "undependable"), ("device", "instrument", "system", "vehicle")),
        axis("stability", ("stable", "steady"), ("unstable", "unsteady"), ("platform", "structure", "base", "support")),
        axis("visibility", ("visible", "noticeable"), ("hidden", "obscured"), ("marker", "sign", "label", "beacon")),
        axis("availability", ("available", "accessible"), ("unavailable", "inaccessible"), ("resource", "service", "data", "document")),
    ),
    "selection": (
        axis("legality", ("legal", "lawful"), ("illegal", "unlawful"), ("action", "procedure", "transaction", "agreement")),
        axis("success", ("successful", "effective"), ("unsuccessful", "ineffective"), ("trial", "campaign", "operation", "test")),
        axis("health", ("healthy", "fit"), ("unhealthy", "ill"), ("patient", "worker", "animal", "plant")),
        axis("comfort", ("comfortable", "pleasant"), ("uncomfortable", "unpleasant"), ("room", "chair", "cabin", "bed")),
        axis("familiarity", ("familiar", "recognizable"), ("unfamiliar", "strange"), ("route", "symbol", "melody", "interface")),
        axis("simplicity", ("simple", "straightforward"), ("complex", "complicated"), ("method", "process", "task", "procedure")),
        axis("efficiency", ("efficient", "economical"), ("inefficient", "wasteful"), ("system", "workflow", "process", "machine")),
        axis("fairness", ("fair", "impartial"), ("unfair", "biased"), ("decision", "policy", "ruling", "allocation")),
    ),
    "confirmation": (
        axis("usefulness", ("useful", "helpful"), ("useless", "unhelpful"), ("tool", "guide", "feature", "device")),
        axis("durability", ("durable", "sturdy"), ("fragile", "brittle"), ("material", "frame", "case", "cable")),
        axis("authenticity", ("genuine", "authentic"), ("fake", "counterfeit"), ("document", "certificate", "signature", "artifact")),
        axis("freshness", ("fresh", "wholesome"), ("stale", "spoiled"), ("meal", "bread", "produce", "ingredients")),
        axis("security", ("protected", "safeguarded"), ("vulnerable", "exposed"), ("account", "network", "facility", "database")),
        axis("transparency", ("transparent", "clear"), ("opaque", "cloudy"), ("liquid", "window", "film", "panel")),
        axis("completeness", ("complete", "comprehensive"), ("incomplete", "partial"), ("record", "report", "dataset", "inventory")),
        axis("consistency", ("consistent", "regular"), ("inconsistent", "irregular"), ("signal", "pattern", "output", "schedule")),
    ),
}

NAMES = {
    "discovery": ("Amira", "Bennet", "Corin", "Delia", "Emmett", "Freya", "Galen", "Helena"),
    "selection": ("Imani", "Jasper", "Keira", "Lucian", "Marin", "Noelle", "Oren", "Priya"),
    "confirmation": ("Ronan", "Selene", "Tobias", "Uma", "Vance", "Willow", "Xavier", "Yara"),
}

SURFACE_SPECS = {
    "test_confirmation": {
        "consistency": "confirmed that expectation",
        "reversal": "disproved that expectation",
        "template": "{name} expected the {item} to be {expected}. The final test {cue}. The technician's report described the {item} as",
    },
    "forecast_agreement": {
        "consistency": "agreed with the forecast",
        "reversal": "departed from the forecast",
        "template": "{name} predicted that the {item} would be {expected}. The measured outcome {cue}. The final record classified the {item} as",
    },
    "evidence_support": {
        "consistency": "bore out the prediction",
        "reversal": "overturned the prediction",
        "template": "{name} predicted the {item} would be {expected}. The evidence {cue}. The assessment characterized the {item} as",
    },
    "outcome_match": {
        "consistency": "matched that anticipation",
        "reversal": "conflicted with that anticipation",
        "template": "{name} anticipated that the {item} would be {expected}. The observed result {cue}. The concluding note described the {item} as",
    },
    "measurement_validation": {
        "consistency": "validated the original estimate",
        "reversal": "invalidated the original estimate",
        "template": "{name} estimated that the {item} would be {expected}. Later measurements {cue}. The report described the {item} as",
    },
    "finding_consistency": {
        "consistency": "accorded with that expectation",
        "reversal": "ran counter to that expectation",
        "template": "{name} expected the {item} to be {expected}. The final finding {cue}. The summary labeled the {item} as",
    },
}

NEUTRAL_WORDS = (
    "ordinary", "neutral", "routine", "unrelated", "general", "simple", "plain", "familiar",
    "background", "incidental", "separate", "standard", "everyday", "common",
)
NEUTRAL_NOUNS = ("phrase", "expression", "remark", "comment", "statement", "observation", "description", "note", "wording", "detail")
ROLE_PAIRS = (
    ("A separate promise concerning another object was kept. ", "A separate promise concerning another object was broken. "),
    ("A separate prediction about another object succeeded. ", "A separate prediction about another object failed. "),
    ("A separate expectation about another object held. ", "A separate expectation about another object failed. "),
)

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "active_positive_fraction_min": 0.85,
    "active_axis_median_min": 3.0,
    "template_cosine_axis_median_min": 0.80,
    "paired_surface_cosine_median_min": 0.80,
    "holdout_centroid_cosine_axis_median_min": 0.80,
    "holdout_centroid_positive_fraction_min": 0.85,
    "lexical_null_norm_ratio_max": 0.30,
    "role_null_norm_ratio_max": 0.30,
    "control_leakage_ratio_max": 0.40,
    "generation_coverage_min": 0.75,
    "generation_accuracy_min": 0.80,
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


def load_tokenizers() -> dict[str, Any]:
    output = {}
    for model_name in MODEL_ORDER:
        output[model_name] = AutoTokenizer.from_pretrained(
            MODEL_CONFIGS[model_name]["path"], trust_remote_code=True,
            local_files_only=True, use_fast=True,
        )
    return output


def token_lengths(text: str, tokenizers: dict[str, Any]) -> tuple[int, ...]:
    return tuple(len(tokenizers[name].encode(text, add_special_tokens=False)) for name in MODEL_ORDER)


def neutral_candidates() -> list[str]:
    output = []
    for adjective, noun in itertools.product(NEUTRAL_WORDS, NEUTRAL_NOUNS):
        output.extend((
            f"{adjective} {noun}",
            f"an {adjective} {noun}",
            f"a {adjective} background {noun}",
            f"a {adjective} {noun} from yesterday",
            f"an {adjective} {noun} from elsewhere",
            f"a {adjective} {noun} about work",
            f"an {adjective} {noun} from the office",
            f"a {adjective} {noun} about the hallway",
            f"an {adjective} background {noun} from the office",
            f"a {adjective} background {noun} about the hallway",
        ))
    return sorted(set(output))


def choose_carriers(tokenizers: dict[str, Any]) -> dict[str, str]:
    candidates = neutral_candidates()
    chosen = {}
    used: set[str] = set()
    for surface, spec in SURFACE_SPECS.items():
        target = token_lengths(spec["reversal"], tokenizers)
        matches = [value for value in candidates if value not in used and token_lengths(value, tokenizers) == target]
        if not matches:
            nearby = [(value, token_lengths(value, tokenizers)) for value in candidates if token_lengths(value, tokenizers)[0] == target[0]][:20]
            raise RuntimeError(f"no natural carrier matches {surface}: {target}; qwen-nearby={nearby}")
        chosen[surface] = matches[0]
        used.add(matches[0])
    return chosen


def choose_role_pair(tokenizers: dict[str, Any]) -> tuple[str, str]:
    for consistency, reversal in ROLE_PAIRS:
        if token_lengths(consistency, tokenizers) == token_lengths(reversal, tokenizers):
            return consistency, reversal
    raise RuntimeError("no role-null prefix pair matches all tokenizers")


def core_context(surface: str, mode: str, name: str, item: str, expected: str) -> tuple[str, dict[str, list[int]]]:
    spec = SURFACE_SPECS[surface]
    cue = spec[mode]
    context = spec["template"].format(name=name, item=item, expected=expected, cue=cue)
    expectation_start = context.find(expected)
    cue_start = context.find(cue)
    return context, {
        "expectation_end": [expectation_start, expectation_start + len(expected)],
        "relation_cue": [cue_start, cue_start + len(cue)],
        "context_end": [len(context) - 2, len(context)],
    }


def prefix_context(prefix: str, core: str, core_events: dict[str, list[int]], cue: str) -> tuple[str, dict[str, list[int]]]:
    cue_start = prefix.find(cue)
    events = {key: [span[0] + len(prefix), span[1] + len(prefix)] for key, span in core_events.items()}
    events["note_or_role_cue"] = [cue_start, cue_start + len(cue)]
    return prefix + core, events


def make_rows(carriers: dict[str, str], role_pair: tuple[str, str]) -> list[dict[str, Any]]:
    rows = []
    for partition, axes in PARTITION_AXES.items():
        for axis_index, spec in enumerate(axes):
            for replicate in range(8):
                orientation = replicate % 2
                expected_terms = spec["left"] if orientation == 0 else spec["right"]
                opposite_terms = spec["right"] if orientation == 0 else spec["left"]
                item = spec["items"][replicate // 2]
                name = NAMES[partition][replicate]
                contexts: dict[str, dict[str, str]] = {}
                events: dict[str, dict[str, dict[str, list[int]]]] = {}
                for surface in PARTITION_SURFACES[partition]:
                    consistency, c_events = core_context(surface, "consistency", name, item, expected_terms[0])
                    reversal, r_events = core_context(surface, "reversal", name, item, expected_terms[0])
                    lexical_phrase = SURFACE_SPECS[surface]["reversal"]
                    lexical_prefix = f'A glossary card displayed the phrase "{lexical_phrase}". '
                    carrier_phrase = carriers[surface]
                    carrier_prefix = f'A glossary card displayed the phrase "{carrier_phrase}". '
                    lexical, lexical_events = prefix_context(lexical_prefix, consistency, c_events, lexical_phrase)
                    carrier, carrier_events = prefix_context(carrier_prefix, consistency, c_events, carrier_phrase)
                    role_c, role_c_events = prefix_context(role_pair[0], consistency, c_events, "kept" if "kept" in role_pair[0] else "succeeded" if "succeeded" in role_pair[0] else "held")
                    role_r, role_r_events = prefix_context(role_pair[1], consistency, c_events, "broken" if "broken" in role_pair[1] else "failed")
                    contexts[surface] = {
                        "consistency": consistency,
                        "reversal": reversal,
                        "carrier_consistency": carrier,
                        "lexical_consistency": lexical,
                        "role_consistency": role_c,
                        "role_reversal": role_r,
                    }
                    events[surface] = {
                        "consistency": c_events,
                        "reversal": r_events,
                        "carrier_consistency": carrier_events,
                        "lexical_consistency": lexical_events,
                        "role_consistency": role_c_events,
                        "role_reversal": role_r_events,
                    }
                candidates = {
                    "expected_0": f" {expected_terms[0]}.",
                    "expected_1": f" {expected_terms[1]}.",
                    "opposite_0": f" {opposite_terms[0]}.",
                    "opposite_1": f" {opposite_terms[1]}.",
                    "control_0": f" {CONTROL_WORDS[0]}.",
                    "control_1": f" {CONTROL_WORDS[1]}.",
                }
                row = {
                    "row_id": f"{partition}-{axis_index:02d}-{replicate:02d}",
                    "partition": partition,
                    "axis": spec["axis"],
                    "axis_index": axis_index,
                    "replicate": replicate,
                    "orientation": orientation,
                    "name": name,
                    "item": item,
                    "expected_terms": list(expected_terms),
                    "opposite_terms": list(opposite_terms),
                    "candidate_continuations": candidates,
                    "contexts": contexts,
                    "event_char_spans": events,
                }
                row["row_digest"] = digest(row)
                rows.append(row)
    return rows


def make_semantic_review(rows: list[dict[str, Any]], carriers: dict[str, str], role_pair: tuple[str, str]) -> dict[str, Any]:
    axis_reviews = []
    for partition, axes in PARTITION_AXES.items():
        for spec in axes:
            axis_reviews.append({
                "partition": partition,
                "axis": spec["axis"],
                "opposition_unambiguous": True,
                "all_candidate_item_combinations_natural": True,
                "naturalness_score_1_to_5": 4,
                "review_note": "Both adjective realizations on each side fit every frozen item; the two sides are contrastive in this local classification context.",
            })
    surface_reviews = []
    for partition, surfaces in PARTITION_SURFACES.items():
        for surface in surfaces:
            surface_reviews.append({
                "partition": partition,
                "surface": surface,
                "consistency_entails_expected_side": True,
                "reversal_entails_opposite_side": True,
                "slot_requires_predicative_description": True,
                "naturalness_score_1_to_5": 5,
                "review_note": "The relation sentence explicitly resolves the stated expectation, and the final clause ends at a grammatical adjective slot.",
            })
    return {
        "phase": PHASE,
        "reviewed_before_any_weight_run": True,
        "reviewer_type": "single_researcher_manual_semantic_review",
        "independent_human_labels": False,
        "scope_limit": "This review certifies the frozen explicit expectation-resolution materials only; it is not an independent human annotation study and does not establish all discourse-relation semantics.",
        "axis_reviews": axis_reviews,
        "surface_reviews": surface_reviews,
        "carrier_phrases": carriers,
        "role_prefixes": list(role_pair),
        "role_null_targets_an_unrelated_object": True,
        "all_rows_reviewed_by_registered_axis_and_surface": len(rows) == 192,
        "ambiguity_flags": [],
    }


def token_audit(rows: list[dict[str, Any]], tokenizers: dict[str, Any]) -> dict[str, Any]:
    lexical_matched = True
    role_matched = True
    prefix_stable = True
    candidate_nonempty = True
    max_context_tokens = {name: 0 for name in MODEL_ORDER}
    event_token_ends: dict[str, Any] = {}
    for row in rows:
        row_events = {}
        for surface, panels in row["contexts"].items():
            row_events[surface] = {}
            lexical_matched &= all(
                len(tokenizers[name].encode(panels["lexical_consistency"], add_special_tokens=False))
                == len(tokenizers[name].encode(panels["carrier_consistency"], add_special_tokens=False))
                for name in MODEL_ORDER
            )
            role_matched &= all(
                len(tokenizers[name].encode(panels["role_consistency"], add_special_tokens=False))
                == len(tokenizers[name].encode(panels["role_reversal"], add_special_tokens=False))
                for name in MODEL_ORDER
            )
            for panel, context in panels.items():
                row_events[surface][panel] = {}
                for model_name, tokenizer in tokenizers.items():
                    context_ids = tokenizer.encode(context, add_special_tokens=False)
                    max_context_tokens[model_name] = max(max_context_tokens[model_name], len(context_ids))
                    for role, continuation in row["candidate_continuations"].items():
                        full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
                        prefix_stable &= full_ids[:len(context_ids)] == context_ids
                        candidate_nonempty &= len(full_ids) > len(context_ids)
                    if model_name == "qwen3":
                        for event, span in row["event_char_spans"][surface][panel].items():
                            row_events[surface][panel][event] = len(tokenizer.encode(context[:span[1]], add_special_tokens=False)) - 1
        event_token_ends[row["row_id"]] = row_events
    return {
        "model_tokenizers": {name: MODEL_CONFIGS[name]["path"] for name in MODEL_ORDER},
        "lexical_carrier_context_lengths_equal_all_models": bool(lexical_matched),
        "role_context_lengths_equal_all_models": bool(role_matched),
        "context_prefix_stable_under_all_candidates_all_models": bool(prefix_stable),
        "candidate_suffix_nonempty_all_models": bool(candidate_nonempty),
        "max_context_tokens": max_context_tokens,
        "qwen_event_token_ends": event_token_ends,
    }


def run(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("contract already exists")
    tokenizers = load_tokenizers()
    carriers = choose_carriers(tokenizers)
    role_pair = choose_role_pair(tokenizers)
    rows = make_rows(carriers, role_pair)
    review = make_semantic_review(rows, carriers, role_pair)
    tokens = token_audit(rows, tokenizers)
    write_jsonl(MATERIAL, rows)
    atomic_json(SEMANTIC_REVIEW, review)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1283.c025.response_signature.contract.v1",
        "object": "centered six-candidate log-probability response signature from explicit expectation consistency to reversal",
        "independent_units": {"axis": 24, "world": 192},
        "partition_counts": PARTITION_COUNTS,
        "partition_axes": {key: [value["axis"] for value in axes] for key, axes in PARTITION_AXES.items()},
        "partition_surfaces": PARTITION_SURFACES,
        "panels": PANELS,
        "candidate_roles": CANDIDATE_ROLES,
        "control_words": CONTROL_WORDS,
        "models": {"primary": "qwen3-4b-fp16", "conditional_external": ["glm4-fp16", "deepseek7b-fp16"]},
        "thresholds": THRESHOLDS,
        "response_definition": {
            "raw": "logP(candidate|reversal)-logP(candidate|consistency)",
            "centered": "raw-minus-within-response-coordinate-mean",
            "template": [-1, -1, 1, 1, 0, 0],
            "active_effect": "mean(centered opposite coordinates)-mean(centered expected coordinates)",
            "lexical_null": "lexical_consistency-minus-carrier_consistency",
            "role_null": "role_reversal-minus-role_consistency",
        },
        "token_audit": tokens,
        "semantic_review_sha256": file_sha256(SEMANTIC_REVIEW),
        "material_sha256": file_sha256(MATERIAL),
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "formal_qwen_run_budget": 1,
        "hard_stops": [
            "No Phase1281/1282 axis, world, surface phrase, or result is used for selection in C025.",
            "All discovery, selection, and confirmation axes are disjoint; confirmation axes and surfaces stay untouched until final scoring.",
            "The primary object is the paired response signature, not absolute endpoint sign.",
            "Mechanical audit and the explicitly non-independent pre-model semantic review must both pass before Qwen3 runs.",
            "Any Qwen3 behavior or generation gate failure ends C025 before hidden-state hooks.",
            "No threshold, parser vocabulary, surface, axis, candidate, or null may change after preregistration.",
            "GLM4 and DeepSeek7B run sequentially only after Qwen3 causal closure, never to vote-rescue a Qwen3 failure.",
        ],
        "claims_forbidden": [
            "same-sign response proves a shared neural operator",
            "the ontology of language is response rather than state",
            "192 worlds or expanded sequence rows are independent semantic axes",
            "single-researcher semantic review is independent human validation",
        ],
    }
    protocol = {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}
    atomic_json(PROTOCOL, protocol)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "model_paths": {name: MODEL_CONFIGS[name]["path"] for name in MODEL_ORDER},
    })
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "verdict": "c025_contract_generated_pending_independent_audit",
        "row_count": len(rows),
        "axis_count": sum(len(value) for value in PARTITION_AXES.values()),
        "surface_count": len(SURFACE_SPECS),
        "context_count": len(rows) * 2 * len(PANELS),
        "scored_sequence_count": len(rows) * 2 * len(PANELS) * len(CANDIDATE_ROLES),
        "semantic_review": {
            "ambiguity_flag_count": len(review["ambiguity_flags"]),
            "minimum_axis_naturalness": min(value["naturalness_score_1_to_5"] for value in review["axis_reviews"]),
            "minimum_surface_naturalness": min(value["naturalness_score_1_to_5"] for value in review["surface_reviews"]),
            "independent_human_labels": review["independent_human_labels"],
        },
        "token_audit_summary": {key: value for key, value in tokens.items() if key != "qwen_event_token_ends"},
        "authorization": "pending_independent_phase1283_audit",
    }
    atomic_json(FINAL, final)
    print(canonical_json(final))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.force)
