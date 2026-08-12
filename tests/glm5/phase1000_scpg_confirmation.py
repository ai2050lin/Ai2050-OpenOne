#!/usr/bin/env python3
"""Phase 1000 disjoint holdout confirmation of the frozen SCPG result."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1000_factorial_binding_protocol import (
    COLORS,
    MODEL,
    OUT_ROOT,
    PHASE,
    canonical,
    write_json,
)
from phase1000_scpg_discovery import (
    EDGE_THRESHOLDS,
    SOURCE_THRESHOLDS,
    candidate_tensor,
    joint_screen,
    natural_joint_test,
    natural_source_controls,
    read_jsonl,
    receiver_screen,
    response_map,
    source_controls,
    summarize_interventions,
    summarize_joint,
    summarize_natural,
    summarize_receivers,
    write_rows,
)


CONFIRMATION_PER_STRATUM = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_hash(pair_id: str) -> str:
    return hashlib.sha256(
        f"phase1000:confirmation:{pair_id}".encode("utf-8")
    ).hexdigest()


def select_confirmation_pairs(
    factor_pairs: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    discovery_pair_ids: set[str],
) -> list[dict[str, Any]]:
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in factor_pairs:
        if (
            pair["factor"] != "entity"
            or pair["split"] != "holdout"
            or pair["pair_id"] in discovery_pair_ids
        ):
            continue
        arm0 = case_by_id[pair["arm0_record_id"]]
        key = (
            int(arm0["template"]),
            int(arm0["display_order"]),
            int(arm0["value_swap"]),
            int(arm0["query_role"]),
        )
        strata[key].append(pair)
    selected = []
    for key, rows in sorted(strata.items()):
        ordered = sorted(rows, key=lambda row: pair_hash(row["pair_id"]))
        if len(ordered) < CONFIRMATION_PER_STRATUM:
            raise RuntimeError(f"underfilled confirmation stratum {key}: {len(ordered)}")
        selected.extend(ordered[:CONFIRMATION_PER_STRATUM])
    expected = 32 * CONFIRMATION_PER_STRATUM
    if len(selected) != expected:
        raise RuntimeError(f"confirmation count drift: {len(selected)} != {expected}")
    return selected


def one_direction_per_pair(
    selected: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in selected:
        arm0 = case_by_id[pair["arm0_record_id"]]
        key = (
            int(arm0["template"]),
            int(arm0["display_order"]),
            int(arm0["value_swap"]),
            int(arm0["query_role"]),
        )
        groups[key].append(pair)
    rows = []
    for _, pairs in sorted(groups.items()):
        ordered = sorted(pairs, key=lambda row: pair_hash(row["pair_id"]))
        for index, pair in enumerate(ordered):
            arm0 = case_by_id[pair["arm0_record_id"]]
            arm1 = case_by_id[pair["arm1_record_id"]]
            if index % 2 == 0:
                source, target, direction = arm0, arm1, "e0_to_e1"
            else:
                source, target, direction = arm1, arm0, "e1_to_e0"
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "partition": "confirmation",
                    "direction": direction,
                    "source": source,
                    "target": target,
                }
            )
    return rows


def confirmation_gate(
    behavior_summary: dict[str, Any],
    source_controls_summary: dict[str, Any],
    source_natural_summary: dict[str, Any],
    response_metrics: dict[str, dict[str, Any]],
    receiver_metrics: dict[str, dict[str, Any]],
    joint_summary: dict[str, dict[str, Any]],
    frozen_joint_size: int,
    joint_natural_summary: dict[str, dict[str, float]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    controls = ("reverse_entity", "scrambled_pair", "noop_target")
    max_candidate_control = max(
        source_controls_summary[condition]["flip_rate"] for condition in controls
    )
    max_natural_control = max(
        source_natural_summary[condition]["flip_rate"] for condition in controls
    )
    source_candidate = source_controls_summary["joint_entity"]
    source_natural = source_natural_summary["joint_entity"]
    single_pass_events = []
    for event_id, metric in receiver_metrics.items():
        if (
            response_metrics[event_id]["response_score"]
            >= EDGE_THRESHOLDS["response_score"]
            and metric["mean_sufficiency_transfer"]
            >= EDGE_THRESHOLDS["single_sufficiency_mean_transfer"]
            and metric["median_mediation_fraction"]
            >= EDGE_THRESHOLDS["single_median_mediation"]
            and metric["scrambled_flip_rate"]
            <= EDGE_THRESHOLDS["single_max_scrambled_flip"]
        ):
            single_pass_events.append(event_id)
    frozen_joint = joint_summary[str(frozen_joint_size)]
    metrics = {
        "source_candidate_flip_rate": source_candidate["flip_rate"],
        "source_candidate_mean_transfer": source_candidate["mean_transfer"],
        "source_natural_flip_rate": source_natural["flip_rate"],
        "max_candidate_control_flip_rate": max_candidate_control,
        "max_natural_control_flip_rate": max_natural_control,
        "single_receiver_pass_events": single_pass_events,
        "frozen_joint_size": frozen_joint_size,
        "frozen_joint_median_mediation": frozen_joint[
            "median_mediation_fraction"
        ],
        "frozen_joint_candidate_restoration_rate": frozen_joint[
            "restored_to_target_rate"
        ],
        "frozen_joint_natural_restoration_rate": joint_natural_summary[
            "source_plus_joint_restore"
        ]["target_rate"],
        "scrambled_joint_natural_restoration_rate": joint_natural_summary[
            "source_plus_scrambled_restore"
        ]["target_rate"],
    }
    checks = {
        "G1_behavior": bool(behavior_summary["behavior_gate_pass"]),
        "G2_source_candidate": (
            source_candidate["flip_rate"]
            >= SOURCE_THRESHOLDS["candidate_flip_rate"]
            and source_candidate["mean_transfer"]
            >= SOURCE_THRESHOLDS["mean_transfer"]
        ),
        "G2_source_natural": (
            source_natural["flip_rate"]
            >= SOURCE_THRESHOLDS["natural_flip_rate"]
        ),
        "G2_source_controls": (
            max_candidate_control <= SOURCE_THRESHOLDS["max_control_flip_rate"]
            and max_natural_control <= SOURCE_THRESHOLDS["max_control_flip_rate"]
        ),
        "G3_source_receiver_response": all(
            metric["response_score"] >= EDGE_THRESHOLDS["response_score"]
            for metric in response_metrics.values()
        ),
        "G4_G5_single_receiver": bool(single_pass_events),
        "G5_frozen_joint_mediation": (
            frozen_joint["median_mediation_fraction"]
            >= EDGE_THRESHOLDS["joint_median_mediation"]
        ),
        "G6_frozen_natural_restoration": (
            joint_natural_summary["source_plus_joint_restore"]["target_rate"]
            >= EDGE_THRESHOLDS["joint_natural_restoration_rate"]
        ),
    }
    return checks, metrics


def run(batch_size: int, natural_budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1000 confirmation requires CUDA")
    protocol_root = OUT_ROOT / "protocol"
    discovery_root = OUT_ROOT / "discovery"
    output_root = OUT_ROOT / "confirmation"
    output_root.mkdir(parents=True, exist_ok=True)

    cases = read_jsonl(protocol_root / "cases.jsonl")
    factor_pairs = read_jsonl(protocol_root / "factor_pairs.jsonl")
    protocol = json.loads((protocol_root / "protocol.json").read_text(encoding="utf-8"))
    behavior_summary = json.loads(
        (OUT_ROOT / "behavior" / "summary.json").read_text(encoding="utf-8")
    )
    discovery_summary = json.loads(
        (discovery_root / "summary.json").read_text(encoding="utf-8")
    )
    frozen_path = discovery_root / "frozen_spec.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    discovery_selected = read_jsonl(discovery_root / "selected_pairs.jsonl")
    if not frozen.get("frozen_before_holdout") or frozen.get("selection_uses_holdout"):
        raise RuntimeError("frozen discovery contract is not valid")
    if not discovery_summary.get("holdout_not_opened"):
        raise RuntimeError("discovery did not preserve holdout")

    case_by_id = {row["record_id"]: row for row in cases}
    selected_pairs = select_confirmation_pairs(
        factor_pairs,
        case_by_id,
        {row["pair_id"] for row in discovery_selected},
    )
    directional = one_direction_per_pair(selected_pairs, case_by_id)
    validation_worlds = {
        case_by_id[row["arm0_record_id"]]["world_id"]
        for row in discovery_selected
    }
    confirmation_worlds = {
        case_by_id[row["arm0_record_id"]]["world_id"] for row in selected_pairs
    }
    if validation_worlds & confirmation_worlds:
        raise RuntimeError("confirmation world leakage")
    write_rows(output_root / "selected_pairs.jsonl", selected_pairs)

    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color]) for color in COLORS
    }
    source_depth = int(frozen["source_depth"])
    ranked_ids = list(frozen["ranked_receiver_event_ids"])
    ranked_lookup = {
        item["event_id"]: item for item in discovery_summary["ranked_receivers"]
    }
    ranked_receivers = [ranked_lookup[event_id] for event_id in ranked_ids]
    frozen_joint_size = int(frozen["best_joint_size"])
    if frozen["best_joint_event_ids"] != ranked_ids[:frozen_joint_size]:
        raise RuntimeError("frozen joint event drift")

    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=False
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        info = get_model_info(model, MODEL)
        effective_eos = eos_ids(model, tokenizer)

        source_control_rows = source_controls(
            model,
            layers,
            device,
            directional,
            candidate_ids,
            source_depth,
            batch_size,
        )
        source_control_summary = summarize_interventions(source_control_rows)
        write_rows(output_root / "source_control_rows.jsonl", source_control_rows)

        source_natural_rows = natural_source_controls(
            model,
            layers,
            tokenizer,
            device,
            directional,
            candidate_ids,
            source_depth,
            effective_eos,
            batch_size,
            natural_budget,
        )
        source_natural_summary = summarize_natural(source_natural_rows)
        write_rows(output_root / "source_natural_rows.jsonl", source_natural_rows)

        response_rows, response_metrics = response_map(
            model,
            layers,
            device,
            directional,
            candidate_ids,
            source_depth,
            ranked_receivers,
            batch_size,
        )
        write_rows(output_root / "response_rows.jsonl", response_rows)
        write_json(output_root / "response_metrics.json", response_metrics)

        receiver_rows = receiver_screen(
            model,
            layers,
            device,
            directional,
            candidate_ids,
            source_depth,
            ranked_receivers,
            batch_size,
        )
        receiver_metrics = summarize_receivers(receiver_rows, response_metrics)
        write_rows(output_root / "receiver_causal_rows.jsonl", receiver_rows)
        write_json(output_root / "receiver_metrics.json", receiver_metrics)

        joint_rows = joint_screen(
            model,
            layers,
            device,
            directional,
            candidate_ids,
            source_depth,
            ranked_receivers,
            batch_size,
        )
        joint_summary = summarize_joint(joint_rows)
        write_rows(output_root / "joint_causal_rows.jsonl", joint_rows)

        joint_natural_rows = natural_joint_test(
            model,
            layers,
            tokenizer,
            device,
            directional,
            candidate_ids,
            source_depth,
            ranked_receivers,
            frozen_joint_size,
            effective_eos,
            batch_size,
            natural_budget,
        )
        joint_natural_summary = summarize_natural(joint_natural_rows)
        write_rows(output_root / "joint_natural_rows.jsonl", joint_natural_rows)

        checks, gate_metrics = confirmation_gate(
            behavior_summary,
            source_control_summary,
            source_natural_summary,
            response_metrics,
            receiver_metrics,
            joint_summary,
            frozen_joint_size,
            joint_natural_summary,
        )
        summary = {
            "schema_version": "phase1000_confirmation_summary.v1",
            "phase": PHASE,
            "model": MODEL,
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "confirmation_per_stratum": CONFIRMATION_PER_STRATUM,
            "validation_world_count": len(validation_worlds),
            "confirmation_world_count": len(confirmation_worlds),
            "worlds_disjoint": not bool(validation_worlds & confirmation_worlds),
            "pairs_disjoint": not bool(
                {row["pair_id"] for row in selected_pairs}
                & {row["pair_id"] for row in discovery_selected}
            ),
            "one_preassigned_direction_per_pair": True,
            "direction_counts": {
                direction: sum(
                    row["direction"] == direction for row in directional
                )
                for direction in ("e0_to_e1", "e1_to_e0")
            },
            "frozen_spec_sha256": sha256_file(frozen_path),
            "frozen_source_depth": source_depth,
            "frozen_receiver_event_ids": ranked_ids,
            "frozen_joint_size": frozen_joint_size,
            "frozen_joint_event_ids": frozen["best_joint_event_ids"],
            "holdout_did_not_reselect": True,
            "source_control_summary": source_control_summary,
            "source_natural_summary": source_natural_summary,
            "response_metrics": response_metrics,
            "receiver_metrics": receiver_metrics,
            "joint_summary": joint_summary,
            "joint_natural_summary": joint_natural_summary,
            "source_thresholds": SOURCE_THRESHOLDS,
            "edge_thresholds": EDGE_THRESHOLDS,
            "gate_checks": checks,
            "gate_metrics": gate_metrics,
            "confirmation_gate_pass": all(checks.values()),
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "batch_size": batch_size,
            "natural_max_new_tokens": natural_budget,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(args.batch_size, args.natural_max_new_tokens)
    print(
        json.dumps(
            {
                "passed": summary["confirmation_gate_pass"],
                "selected_pair_count": summary["selected_pair_count"],
                "direction_count": summary["direction_count"],
                "gate_checks": summary["gate_checks"],
                "gate_metrics": summary["gate_metrics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
