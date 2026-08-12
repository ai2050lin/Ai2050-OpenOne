#!/usr/bin/env python3
"""Finalize the observation-led Phase1017 semantic-niche atlas."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1009_crossfamily_response_protocol import digest
from phase1017_semantic_niche_protocol import (
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    WORDS,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1017_semantic_niche_scan import (
    ANALYSIS_CONTRASTS,
    CAPTURE_ROLES,
    DIRECTION_CONTRASTS,
    KEY_DIRECTION_ROLES,
)


PRIMARY_THRESHOLDS = {
    "direction_consistency": 0.45,
    "lexical_alignment": 0.40,
    "interaction_fraction": 0.20,
}
ANALYSIS_ROOT = OUT_ROOT / "analysis"
SCAN_ROOT = OUT_ROOT / "formal_scan"
TARGET_ROOT = OUT_ROOT / "targeted_behavior_scan"
EPSILON = 1e-12


def finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def safe_median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.median(array))


def safe_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.mean(array))


def event_arrays(
    metrics: dict[str, np.ndarray],
    name: str,
) -> np.ndarray:
    return np.concatenate([
        metrics[f"whole_{name}"],
        metrics[f"head_{name}"],
    ], axis=-1)


def pairwise_cosines(vectors: list[np.ndarray]) -> list[float]:
    result = []
    normalized = []
    for vector in vectors:
        value = vector.astype(np.float64, copy=False)
        norm = float(np.linalg.norm(value))
        if norm > EPSILON:
            normalized.append(value / norm)
    for left, right in itertools.combinations(normalized, 2):
        result.append(float(np.dot(left, right)))
    return result


def candidate_from_row(
    row: dict[str, Any],
    thresholds: dict[str, float],
) -> bool:
    values = (
        row["bt_direction_consistency"],
        row["bt_lexical_alignment"],
        row["interaction_fraction"],
        row["median_bt_magnitude"],
    )
    return bool(
        all(value is not None for value in values)
        and row["bt_direction_consistency"]
        >= thresholds["direction_consistency"]
        and row["bt_lexical_alignment"]
        >= thresholds["lexical_alignment"]
        and row["interaction_fraction"]
        >= thresholds["interaction_fraction"]
        and row["median_bt_magnitude"] > 0
    )


def panel_rows(
    *,
    model_name: str,
    word: str,
    split: str,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    panel_root = SCAN_ROOT / model_name / word / split
    scalars = np.load(
        panel_root / "response_scalars.npz",
        allow_pickle=False,
    )
    directions = np.load(
        panel_root / "direction_metrics.npz",
        allow_pickle=False,
    )
    unit_rows = read_jsonl(panel_root / "units.jsonl")
    contrast_index = {
        str(name): index
        for index, name in enumerate(scalars["contrast_names"].tolist())
    }
    direction_index = {
        str(name): index
        for index, name in enumerate(
            directions["direction_contrast_names"].tolist()
        )
    }
    role_index = {
        str(name): index
        for index, name in enumerate(scalars["role_names"].tolist())
    }
    magnitude = scalars["normalized_magnitude"]
    consistency = event_arrays(directions, "consistency")
    lexical_alignment = event_arrays(
        directions,
        "lexical_alignment",
    )
    ambiguous_neutral_alignment = event_arrays(
        directions,
        "ambiguous_neutral_alignment",
    )
    candidate_correct = np.asarray([
        bool(row["ambiguous_candidate_all_hit"]) for row in unit_rows
    ])
    generation_correct = np.asarray([
        bool(row["ambiguous_generation_all_hit"]) for row in unit_rows
    ])
    rows = []
    for role in KEY_DIRECTION_ROLES:
        r = role_index[role]
        for event_index, event in enumerate(events):
            bt_values = magnitude[
                :,
                contrast_index["BT"],
                r,
                event_index,
            ]
            ba_values = magnitude[
                :,
                contrast_index["BA"],
                r,
                event_index,
            ]
            bn_values = magnitude[
                :,
                contrast_index["BN"],
                r,
                event_index,
            ]
            median_bt = safe_median(bt_values)
            median_ba = safe_median(ba_values)
            median_bn = safe_median(bn_values)
            interaction_fraction = (
                None
                if median_bt is None or median_ba is None
                else float(median_bt / (median_ba + EPSILON))
            )
            interaction_over_neutral = (
                None
                if median_bt is None or median_bn is None
                else float(median_bt / (median_bn + EPSILON))
            )

            def masked_median(mask: np.ndarray) -> float | None:
                return safe_median(bt_values[mask])

            row = {
                "schema_version": "phase1017_event_role_metric.v1",
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "model": model_name,
                "word": word,
                "split": split,
                "event_index": int(event_index),
                "event_id": event["event_id"],
                "component": event["component"],
                "depth": int(event["depth"]),
                "relative_depth": float(event["relative_depth"]),
                "head": event["head"],
                "role": role,
                "unit_count": len(unit_rows),
                "bt_direction_consistency": finite(
                    consistency[
                        direction_index["BT"],
                        r,
                        event_index,
                    ]
                ),
                "ba_direction_consistency": finite(
                    consistency[
                        direction_index["BA"],
                        r,
                        event_index,
                    ]
                ),
                "bn_direction_consistency": finite(
                    consistency[
                        direction_index["BN"],
                        r,
                        event_index,
                    ]
                ),
                "bt_lexical_alignment": finite(
                    lexical_alignment[r, event_index]
                ),
                "ambiguous_neutral_direction_alignment": finite(
                    ambiguous_neutral_alignment[r, event_index]
                ),
                "median_bt_magnitude": median_bt,
                "median_ba_magnitude": median_ba,
                "median_bn_magnitude": median_bn,
                "median_la_magnitude": safe_median(magnitude[
                    :,
                    contrast_index["LA"],
                    r,
                    event_index,
                ]),
                "interaction_fraction": interaction_fraction,
                "interaction_over_neutral": interaction_over_neutral,
                "candidate_correct_count": int(candidate_correct.sum()),
                "candidate_failed_count": int((~candidate_correct).sum()),
                "candidate_correct_bt_median": masked_median(
                    candidate_correct
                ),
                "candidate_failed_bt_median": masked_median(
                    ~candidate_correct
                ),
                "generation_correct_count": int(
                    generation_correct.sum()
                ),
                "generation_failed_count": int(
                    (~generation_correct).sum()
                ),
                "generation_correct_bt_median": masked_median(
                    generation_correct
                ),
                "generation_failed_bt_median": masked_median(
                    ~generation_correct
                ),
            }
            row["primary_descriptive_candidate"] = candidate_from_row(
                row,
                PRIMARY_THRESHOLDS,
            )
            rows.append(row)
    return rows


def confirmed_rows(
    discovery: list[dict[str, Any]],
    confirmation_by_key: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in discovery:
        key = (
            row["model"],
            row["word"],
            row["event_id"],
            row["role"],
        )
        heldout = confirmation_by_key.get(key)
        if heldout is None or not heldout["primary_descriptive_candidate"]:
            continue
        output.append({
            "schema_version": "phase1017_confirmed_word_core.v1",
            "phase": PHASE,
            "model": row["model"],
            "word": row["word"],
            "event_id": row["event_id"],
            "component": row["component"],
            "depth": row["depth"],
            "relative_depth": row["relative_depth"],
            "head": row["head"],
            "role": row["role"],
            "discovery": row,
            "confirmation": heldout,
        })
    return output


def vector_for_core(
    *,
    model_name: str,
    word: str,
    split: str,
    event_index: int,
    role: str,
    whole_count: int,
) -> np.ndarray | None:
    path = (
        SCAN_ROOT
        / model_name
        / word
        / split
        / "key_direction_sums.npz"
    )
    data = np.load(path, allow_pickle=False)
    direction_index = {
        str(name): index
        for index, name in enumerate(
            data["direction_contrast_names"].tolist()
        )
    }
    role_index = {
        str(name): index
        for index, name in enumerate(data["role_names"].tolist())
    }
    d = direction_index["BT"]
    r = role_index[role]
    if event_index < whole_count:
        vector = data["whole_sums"][d, r, event_index]
        count = int(data["whole_count"][d, r, event_index])
    else:
        index = event_index - whole_count
        vector = data["head_sums"][d, r, index]
        count = int(data["head_count"][d, r, index])
    if count < 2 or float(np.linalg.norm(vector)) <= EPSILON:
        return None
    return vector.astype(np.float32, copy=False)


def shared_physical_rows(
    confirmed: list[dict[str, Any]],
    model_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in confirmed:
        grouped[(row["model"], row["event_id"], row["role"])].append(row)
    output = []
    for (model_name, event_id, role), rows in grouped.items():
        words = sorted({row["word"] for row in rows})
        if len(words) < 2:
            continue
        whole_count = 1 + 3 * int(
            model_summaries[model_name]["model_info"]["n_layers"]
        )
        vectors = []
        within = []
        for row in rows:
            vector = vector_for_core(
                model_name=model_name,
                word=row["word"],
                split="confirmation",
                event_index=int(row["confirmation"]["event_index"]),
                role=role,
                whole_count=whole_count,
            )
            if vector is not None:
                vectors.append(vector)
            value = row["confirmation"]["bt_direction_consistency"]
            if value is not None:
                within.append(float(value))
        cross_cosines = pairwise_cosines(vectors)
        sample = rows[0]
        output.append({
            "schema_version": "phase1017_shared_physical_core.v1",
            "phase": PHASE,
            "model": model_name,
            "event_id": event_id,
            "component": sample["component"],
            "depth": sample["depth"],
            "relative_depth": sample["relative_depth"],
            "head": sample["head"],
            "role": role,
            "word_count": len(words),
            "words": words,
            "within_word_confirmation_consistency_median": safe_median(
                within
            ),
            "cross_word_direction_cosine_median": safe_median(
                cross_cosines
            ),
            "cross_word_direction_cosine_minimum": (
                min(cross_cosines) if cross_cosines else None
            ),
            "cross_word_pair_count": len(cross_cosines),
            "claim": (
                "physical reuse with measured word-conditioned direction; "
                "not a causal mechanism"
            ),
        })
    output.sort(
        key=lambda row: (
            -row["word_count"],
            -(
                row["within_word_confirmation_consistency_median"]
                if row["within_word_confirmation_consistency_median"]
                is not None else -1
            ),
            row["model"],
            row["event_id"],
            row["role"],
        )
    )
    return output


def threshold_sensitivity(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    grid = prereg["descriptive_threshold_grid"]
    output = []
    for consistency, lexical, fraction in itertools.product(
        grid["direction_consistency"],
        grid["lexical_alignment"],
        grid["interaction_fraction"],
    ):
        thresholds = {
            "direction_consistency": float(consistency),
            "lexical_alignment": float(lexical),
            "interaction_fraction": float(fraction),
        }
        discovery = [
            row for row in rows
            if row["split"] == "discovery"
            and candidate_from_row(row, thresholds)
        ]
        confirmation_keys = {
            (
                row["model"],
                row["word"],
                row["event_id"],
                row["role"],
            )
            for row in rows
            if row["split"] == "confirmation"
            and candidate_from_row(row, thresholds)
        }
        confirmed_count = sum(
            (
                row["model"],
                row["word"],
                row["event_id"],
                row["role"],
            ) in confirmation_keys
            for row in discovery
        )
        output.append({
            "schema_version": "phase1017_threshold_sensitivity.v1",
            "phase": PHASE,
            **thresholds,
            "discovery_candidate_count": len(discovery),
            "heldout_confirmed_count": int(confirmed_count),
        })
    return output


def frozen_selection(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    discovery = [
        row for row in rows
        if row["split"] == "discovery"
        and row["primary_descriptive_candidate"]
    ]
    grouped = defaultdict(list)
    for row in discovery:
        grouped[(row["model"], row["event_id"], row["role"])].append(row)
    ranked_by_model = defaultdict(list)
    for (model_name, event_id, role), values in grouped.items():
        ranked_by_model[model_name].append({
            "schema_version": "phase1017_target_selection_row.v1",
            "phase": PHASE,
            "model": model_name,
            "event_id": event_id,
            "component": values[0]["component"],
            "depth": values[0]["depth"],
            "relative_depth": values[0]["relative_depth"],
            "head": values[0]["head"],
            "role": role,
            "discovery_word_count": len({row["word"] for row in values}),
            "discovery_words": sorted({row["word"] for row in values}),
            "median_discovery_consistency": safe_median(
                row["bt_direction_consistency"] for row in values
            ),
            "median_discovery_lexical_alignment": safe_median(
                row["bt_lexical_alignment"] for row in values
            ),
            "median_discovery_interaction_fraction": safe_median(
                row["interaction_fraction"] for row in values
            ),
            "selection_used_confirmation": False,
            "selection_used_behavior": False,
        })
    selected = []
    for model_name in MODELS:
        ranked = sorted(
            ranked_by_model[model_name],
            key=lambda row: (
                -row["discovery_word_count"],
                -(
                    row["median_discovery_consistency"]
                    if row["median_discovery_consistency"] is not None
                    else -1
                ),
                -(
                    row["median_discovery_interaction_fraction"]
                    if row["median_discovery_interaction_fraction"]
                    is not None else -1
                ),
                row["event_id"],
                row["role"],
            ),
        )
        selected.extend(ranked[:8])
    selection_digest = digest(selected)
    summary = {
        "schema_version": "phase1017_target_selection.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "selection_digest": selection_digest,
        "selection_count": len(selected),
        "selection_used_discovery_only": True,
        "selection_used_behavior": False,
        "selection_used_confirmation": False,
        "selection_by_model": dict(Counter(
            row["model"] for row in selected
        )),
    }
    return selected, summary


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    model_summaries = {
        model_name: read_json(
            SCAN_ROOT / model_name / "summary.json"
        )
        for model_name in MODELS
    }
    for model_name, summary in model_summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model_name}: protocol digest mismatch")

    rows = []
    events_by_model = {}
    for model_name in MODELS:
        events = read_jsonl(SCAN_ROOT / model_name / "events.jsonl")
        events_by_model[model_name] = events
        for word in WORDS:
            for split in ("discovery", "confirmation"):
                rows.extend(panel_rows(
                    model_name=model_name,
                    word=word,
                    split=split,
                    events=events,
                ))
    write_jsonl(ANALYSIS_ROOT / "event_role_metrics.jsonl", rows)

    discovery = [
        row for row in rows
        if row["split"] == "discovery"
        and row["primary_descriptive_candidate"]
    ]
    confirmation_by_key = {
        (
            row["model"],
            row["word"],
            row["event_id"],
            row["role"],
        ): row
        for row in rows
        if row["split"] == "confirmation"
    }
    confirmed = confirmed_rows(discovery, confirmation_by_key)
    write_jsonl(
        ANALYSIS_ROOT / "heldout_confirmed_word_cores.jsonl",
        confirmed,
    )
    shared = shared_physical_rows(confirmed, model_summaries)
    write_jsonl(
        ANALYSIS_ROOT / "shared_physical_cores.jsonl",
        shared,
    )
    sensitivity = threshold_sensitivity(rows)
    write_jsonl(
        ANALYSIS_ROOT / "threshold_sensitivity.jsonl",
        sensitivity,
    )
    selected, selection_summary = frozen_selection(rows)
    write_jsonl(TARGET_ROOT / "selection.jsonl", selected)
    write_json(TARGET_ROOT / "selection.json", selection_summary)

    behavior = {
        model_name: read_json(
            OUT_ROOT
            / "behavior"
            / model_name
            / "formal.summary.json"
        )
        for model_name in MODELS
    }
    confirmed_by_model = Counter(row["model"] for row in confirmed)
    confirmed_by_role = Counter(
        f"{row['model']}:{row['role']}" for row in confirmed
    )
    confirmed_by_component = Counter(
        f"{row['model']}:{row['component']}" for row in confirmed
    )
    shared_by_model = Counter(row["model"] for row in shared)
    shared_ge3_by_model = Counter(
        row["model"] for row in shared if row["word_count"] >= 3
    )
    cross_word_cosines = [
        row["cross_word_direction_cosine_median"]
        for row in shared
        if row["cross_word_direction_cosine_median"] is not None
    ]
    within_word_consistency = [
        row["within_word_confirmation_consistency_median"]
        for row in shared
        if row["within_word_confirmation_consistency_median"] is not None
    ]
    candidate_behavior_rows = [
        row for row in confirmed
        if row["confirmation"]["candidate_correct_count"] >= 2
        and row["confirmation"]["candidate_failed_count"] >= 2
        and row["confirmation"]["candidate_correct_bt_median"] is not None
        and row["confirmation"]["candidate_failed_bt_median"] is not None
    ]
    generation_behavior_rows = [
        row for row in confirmed
        if row["confirmation"]["generation_correct_count"] >= 2
        and row["confirmation"]["generation_failed_count"] >= 2
        and row["confirmation"]["generation_correct_bt_median"] is not None
        and row["confirmation"]["generation_failed_bt_median"] is not None
    ]
    summary = {
        "schema_version": "phase1017_semantic_niche_analysis.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "model_count": len(MODELS),
        "word_count": len(WORDS),
        "panel_count": int(sum(
            row["panel_count"] for row in model_summaries.values()
        )),
        "unit_count": int(sum(
            row["unit_count"] for row in model_summaries.values()
        )),
        "singleton_forward_count": int(sum(
            row["singleton_forward_count"]
            for row in model_summaries.values()
        )),
        "identity_maximum": float(max(
            row["identity_maximum"]
            for row in model_summaries.values()
        )),
        "interaction_cue_maximum": float(max(
            row["interaction_cue_maximum"]
            for row in model_summaries.values()
        )),
        "target_embedding_interaction_maximum": float(max(
            row["target_embedding_interaction_maximum"]
            for row in model_summaries.values()
        )),
        "behavior": {
            model_name: {
                "selected_prompt_mode": (
                    behavior[model_name]["prompt_mode"]
                ),
                "generation_first_word_accuracy": (
                    behavior[model_name][
                        "generation_first_word_accuracy"
                    ]
                ),
                "candidate_accuracy": (
                    behavior[model_name]["candidate_accuracy"]
                ),
            }
            for model_name in MODELS
        },
        "primary_descriptive_thresholds": PRIMARY_THRESHOLDS,
        "metric_row_count": len(rows),
        "discovery_candidate_count": len(discovery),
        "heldout_confirmed_word_core_count": len(confirmed),
        "heldout_confirmed_by_model": dict(confirmed_by_model),
        "heldout_confirmed_by_model_role": dict(confirmed_by_role),
        "heldout_confirmed_by_model_component": dict(
            confirmed_by_component
        ),
        "shared_physical_core_count": len(shared),
        "shared_physical_by_model": dict(shared_by_model),
        "shared_physical_ge3_words_by_model": dict(
            shared_ge3_by_model
        ),
        "within_word_confirmation_consistency_median": safe_median(
            within_word_consistency
        ),
        "cross_word_direction_cosine_median": safe_median(
            cross_word_cosines
        ),
        "confirmed_candidate_behavior_comparable_count": len(
            candidate_behavior_rows
        ),
        "confirmed_candidate_correct_magnitude_larger_count": int(sum(
            row["confirmation"]["candidate_correct_bt_median"]
            > row["confirmation"]["candidate_failed_bt_median"]
            for row in candidate_behavior_rows
        )),
        "confirmed_generation_behavior_comparable_count": len(
            generation_behavior_rows
        ),
        "confirmed_generation_correct_magnitude_larger_count": int(sum(
            row["confirmation"]["generation_correct_bt_median"]
            > row["confirmation"]["generation_failed_bt_median"]
            for row in generation_behavior_rows
        )),
        "top_confirmed_word_cores": sorted(
            confirmed,
            key=lambda row: (
                -row["confirmation"]["bt_direction_consistency"],
                -row["confirmation"]["interaction_fraction"],
            ),
        )[:20],
        "top_shared_physical_cores": shared[:20],
        "selection": selection_summary,
        "automatic_continuation_assessment": {
            "targeted_behavior_direction_needed": bool(
                len(confirmed) > 0
                and any(row["word_count"] >= 3 for row in shared)
                and any(
                    behavior[model_name]["generation_first_word_accuracy"]
                    > 0.5
                    for model_name in MODELS
                )
            ),
            "continue_to_neuron_localization": False,
            "continue_to_causal_closure": False,
            "reason": (
                "Individual correct-versus-failed interaction directions "
                "must be measured before any neuron or causal claim."
            ),
        },
        "interpretation_limits": [
            (
                "Confirmed BT is a target-conditioned contextual "
                "interaction, not persistent weight plasticity."
            ),
            (
                "Physical overlap plus direction difference is compatible "
                "with differential reuse but does not prove a semantic niche "
                "mechanism."
            ),
            (
                "Magnitude behavior separation alone cannot identify a "
                "decision direction."
            ),
        ],
    }
    write_json(ANALYSIS_ROOT / "summary.json", summary)
    print(json.dumps({
        "phase": PHASE,
        "metric_row_count": len(rows),
        "discovery_candidate_count": len(discovery),
        "heldout_confirmed_word_core_count": len(confirmed),
        "shared_physical_core_count": len(shared),
        "shared_ge3_count": sum(
            row["word_count"] >= 3 for row in shared
        ),
        "within_word_consistency_median": (
            summary["within_word_confirmation_consistency_median"]
        ),
        "cross_word_cosine_median": (
            summary["cross_word_direction_cosine_median"]
        ),
        "selection_count": selection_summary["selection_count"],
        "automatic_continuation": (
            summary["automatic_continuation_assessment"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
