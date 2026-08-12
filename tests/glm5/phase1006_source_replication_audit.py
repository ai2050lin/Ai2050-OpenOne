#!/usr/bin/env python3
"""Replicate frozen Phase1006 source sets on unused protocol pairs.

This audit does not rerank positions.  It takes ranks 3-4 from every
pre-existing pair stratum, verifies behavior on those unseen directions, and
then evaluates the already frozen source sets and controls.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1006_autoregressive_temporal_aggregation_protocol import (
    MODELS,
    OUT_ROOT,
    PHASE,
    SPLITS,
    TEMPLATES_BY_SPLIT,
    canonical,
    digest,
    read_json,
    read_jsonl,
    stable_order,
    write_json,
)
from phase1006_blind_source_and_behavior import (
    BATCH_SIZE,
    choose_donors,
    chunks,
    eos_token_ids,
    evaluate_positions,
    forward_teacher_forced_step1,
    forward_two_step,
    natural_generate,
    prepare_batches,
    release_prepared,
    semantic_answer_ids,
    sequence_metrics,
)


AUDIT_ROOT = OUT_ROOT / "source_replication"
PAIR_RANK_START = 2
PAIR_COUNT_PER_STRATUM = 2


def pair_stratum(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["template"]),
        int(row["display_order"]),
        int(row["value_swap"]),
        int(row["query_role"]),
    )


def frozen_holdout_pairs(model_name: str, split: str) -> list[dict[str, Any]]:
    model_root = OUT_ROOT / "protocol" / model_name
    all_pairs = [
        row
        for row in read_jsonl(model_root / "pairs.jsonl")
        if row["split"] == split
    ]
    selected_ids = {
        row["pair_id"]
        for row in read_jsonl(
            model_root / f"{split}_selected_pairs.jsonl"
        )
    }
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in all_pairs:
        strata[pair_stratum(row)].append(row)

    holdout = []
    for key, rows in sorted(strata.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_order(
                row["pair_id"],
                f"pair:{split}:{key}",
            ),
        )
        chosen = ordered[
            PAIR_RANK_START:PAIR_RANK_START + PAIR_COUNT_PER_STRATUM
        ]
        if len(chosen) != PAIR_COUNT_PER_STRATUM:
            raise RuntimeError(f"underfilled replication stratum {key}")
        if any(row["pair_id"] in selected_ids for row in chosen):
            raise RuntimeError("replication pair overlaps formal pair")
        holdout.extend(chosen)
    if len(holdout) != 32:
        raise RuntimeError(f"{model_name}/{split}: holdout n={len(holdout)}")
    return holdout


def prepare_audit_protocol() -> dict[str, Any]:
    parent = read_json(OUT_ROOT / "protocol" / "protocol.json")
    pair_ids = {}
    overlap_audit = {}
    candidate_cells = {}
    for model_name in MODELS:
        pair_ids[model_name] = {}
        overlap_audit[model_name] = {}
        for split in SPLITS:
            rows = frozen_holdout_pairs(model_name, split)
            pair_ids[model_name][split] = [
                row["pair_id"] for row in rows
            ]
            selected = {
                row["pair_id"]
                for row in read_jsonl(
                    OUT_ROOT
                    / "protocol"
                    / model_name
                    / f"{split}_selected_pairs.jsonl"
                )
            }
            overlap_audit[model_name][split] = {
                "pair_count": len(rows),
                "formal_pair_overlap": len(
                    selected & set(pair_ids[model_name][split])
                ),
            }
        source_summary_path = (
            OUT_ROOT / "blind_source" / model_name / "summary.json"
        )
        if source_summary_path.exists():
            source_summary = read_json(source_summary_path)
            candidate_cells[model_name] = [
                {
                    "split": item["split"],
                    "template": int(item["template"]),
                    "frozen_positions": item.get("frozen_positions", []),
                }
                for item in source_summary["source_cells"]
                if item["source_run"]
            ]
        else:
            candidate_cells[model_name] = []

    payload = {
        "schema_version": "phase1006_source_replication_protocol.v1",
        "phase": PHASE,
        "parent_protocol_digest": parent["preregistration_digest"],
        "purpose": (
            "test whether frozen blind source role structure and control "
            "behavior repeat on unused protocol pairs"
        ),
        "selection": (
            "pair ranks 3-4 in every original stable-hash stratum; no "
            "position reranking, threshold change, or semantic-role use"
        ),
        "pair_rank_start_zero_based": PAIR_RANK_START,
        "pair_count_per_stratum": PAIR_COUNT_PER_STRATUM,
        "pair_ids": pair_ids,
        "overlap_audit": overlap_audit,
        "candidate_cells": candidate_cells,
        "thresholds_unchanged": {
            "behavior_each_step_candidate_accuracy": 0.95,
            "behavior_natural_exact_rate": 0.90,
            "behavior_immediate_eos_rate": 0.90,
            "source_joint_donor_sequence_rate": 0.80,
            "source_joint_median_transfer": 0.50,
            "same_answer_target_sequence_rate": 0.95,
            "noop_target_sequence_rate": 0.99,
        },
        "forbidden": [
            "rerank positions on replication cases",
            "repair a failed control with post-hoc subset selection",
            "continue to receiver decomposition when source gate fails",
        ],
        "digest": None,
    }
    digest_payload = dict(payload)
    digest_payload["digest"] = None
    payload["digest"] = digest(digest_payload)
    write_json(AUDIT_ROOT / "protocol.json", payload)
    return payload


def holdout_directional_rows(
    model_name: str,
    split: str,
    template: int,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    model_root = OUT_ROOT / "protocol" / model_name
    cases = {
        row["record_id"]: row
        for row in read_jsonl(model_root / "cases.jsonl")
    }
    pair_lookup = {
        row["pair_id"]: row
        for row in read_jsonl(model_root / "pairs.jsonl")
    }
    rows = []
    for pair_id in protocol["pair_ids"][model_name][split]:
        pair = pair_lookup[pair_id]
        if int(pair["template"]) != template:
            continue
        arm0 = cases[pair["arm0_record_id"]]
        arm1 = cases[pair["arm1_record_id"]]
        for direction, donor, target in (
            ("arm0_to_arm1", arm0, arm1),
            ("arm1_to_arm0", arm1, arm0),
        ):
            rows.append({
                "pair_id": pair_id,
                "model": model_name,
                "split": split,
                "template": template,
                "direction": direction,
                "source": donor,
                "target": target,
            })
    if len(rows) != 32:
        raise RuntimeError(
            f"{model_name}/{split}/t{template}: replication n={len(rows)}"
        )
    return rows


def behavior_audit(
    model,
    layers,
    tokenizer,
    device,
    model_name: str,
    directional: list[dict[str, Any]],
) -> dict[str, Any]:
    cases = [item["target"] for item in directional]
    effective_eos = eos_token_ids(model, tokenizer, model_name)
    step_hits = [[], []]
    teacher_hits = []
    generated_all = []
    for batch in chunks(cases, BATCH_SIZE):
        output = forward_two_step(model, layers, device, batch)
        teacher = forward_teacher_forced_step1(
            model, layers, device, batch
        )
        generated_all.extend(natural_generate(
            model,
            layers,
            tokenizer,
            device,
            batch,
            effective_eos_ids=effective_eos,
        ))
        for index, case in enumerate(batch):
            expected = semantic_answer_ids(case)
            for step in (0, 1):
                step_hits[step].append(
                    int(output["steps"][step]["prediction_ids"][index])
                    == expected[step]
                )
            teacher_hits.append(
                int(teacher["prediction_ids"][index]) == expected[1]
            )
    _, rollout = sequence_metrics(generated_all, cases, effective_eos)
    summary = {
        "n": len(cases),
        "step0_autoregressive_accuracy": float(np.mean(step_hits[0])),
        "step1_autoregressive_accuracy": float(np.mean(step_hits[1])),
        "step1_teacher_forced_accuracy": float(np.mean(teacher_hits)),
        **rollout,
    }
    summary["behavior_gate_pass"] = (
        summary["step0_autoregressive_accuracy"] >= 0.95
        and summary["step1_autoregressive_accuracy"] >= 0.95
        and summary["step1_teacher_forced_accuracy"] >= 0.95
        and summary["natural_protocol_prefix_rate"] >= 0.90
        and summary["natural_exact_rate"] >= 0.90
        and summary["immediate_eos_rate"] >= 0.90
    )
    return summary


def semantic_role_audit(
    directional: list[dict[str, Any]],
    positions: list[int],
) -> dict[str, Any]:
    raw = {position: Counter() for position in positions}
    role_classes = Counter()
    for item in directional:
        roles = item["target"]["sealed_semantic_role_positions"]
        for position in positions:
            matched = [
                role
                for role, role_position in roles.items()
                if int(role_position) == position
            ]
            raw[position].update(matched or ["other"])
    for counter in raw.values():
        names = set(counter)
        if names == {"query_name"}:
            role_classes["query_name"] += 1
        elif any(name.startswith("fact_entity_") for name in names):
            role_classes["fact_entity"] += 1
        elif any(name.endswith("word0") for name in names):
            role_classes["value_word0"] += 1
        elif any(name.endswith("word1") for name in names):
            role_classes["value_word1"] += 1
        else:
            role_classes["other"] += 1
    return {
        "by_position": {
            str(position): dict(counter)
            for position, counter in raw.items()
        },
        "role_class_counts": dict(role_classes),
    }


def evaluate_cell(
    model,
    layers,
    tokenizer,
    device,
    model_name: str,
    split: str,
    template: int,
    positions: list[int],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    directional = holdout_directional_rows(
        model_name, split, template, protocol
    )
    behavior = behavior_audit(
        model,
        layers,
        tokenizer,
        device,
        model_name,
        directional,
    )
    summary: dict[str, Any] = {
        "schema_version": "phase1006_source_replication_cell.v1",
        "phase": PHASE,
        "model": model_name,
        "split": split,
        "template": template,
        "audit_protocol_digest": protocol["digest"],
        "frozen_positions": positions,
        "positions_reranked": False,
        "behavior": behavior,
        "semantic_role_audit": semantic_role_audit(
            directional, positions
        ),
    }
    if not behavior["behavior_gate_pass"]:
        summary["source_run"] = False
        summary["source_gate_pass"] = False
        summary["skip_reason"] = "replication_behavior_gate_failed"
        return summary

    donors, donor_audit = choose_donors(
        model_name,
        split,
        template,
        directional,
        same_answer=False,
    )
    prepared = prepare_batches(
        model, layers, device, directional, donors
    )
    different, _ = evaluate_positions(
        model,
        layers,
        device,
        prepared,
        positions,
        "replication_frozen_different_answer",
    )
    release_prepared(prepared)

    same_donors, same_audit = choose_donors(
        model_name,
        split,
        template,
        directional,
        same_answer=True,
    )
    prepared = prepare_batches(
        model, layers, device, directional, same_donors
    )
    same, _ = evaluate_positions(
        model,
        layers,
        device,
        prepared,
        positions,
        "replication_frozen_same_answer",
    )
    release_prepared(prepared)

    targets = [item["target"] for item in directional]
    prepared = prepare_batches(
        model, layers, device, directional, targets
    )
    noop, _ = evaluate_positions(
        model,
        layers,
        device,
        prepared,
        positions,
        "replication_frozen_target_noop",
    )
    release_prepared(prepared)

    gate = (
        different["donor_sequence_rate"] >= 0.80
        and different["median_normalized_transfer"] >= 0.50
        and same["target_sequence_rate"] >= 0.95
        and noop["target_sequence_rate"] >= 0.99
    )
    summary.update({
        "source_run": True,
        "different_answer": different,
        "same_answer_control": same,
        "target_noop": noop,
        "donor_audit": donor_audit,
        "same_answer_donor_audit": same_audit,
        "source_gate_pass": gate,
    })
    return summary


def run_model(model_name: str) -> dict[str, Any]:
    protocol = read_json(AUDIT_ROOT / "protocol.json")
    candidates = protocol["candidate_cells"][model_name]
    if not candidates:
        summary = {
            "schema_version": "phase1006_source_replication_model.v1",
            "phase": PHASE,
            "model": model_name,
            "model_loaded": False,
            "skip_reason": "no_formal_frozen_source_set",
            "cells": [],
        }
        write_json(AUDIT_ROOT / model_name / "summary.json", summary)
        return summary

    started = time.time()
    model = tokenizer = device = None
    cells = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        for candidate in candidates:
            cell = evaluate_cell(
                model,
                layers,
                tokenizer,
                device,
                model_name,
                candidate["split"],
                int(candidate["template"]),
                [int(value) for value in candidate["frozen_positions"]],
                protocol,
            )
            cells.append(cell)
            write_json(
                AUDIT_ROOT
                / model_name
                / candidate["split"]
                / f"template{int(candidate['template'])}"
                / "summary.json",
                cell,
            )
            print(
                f"[replication] {model_name}/{candidate['split']}/"
                f"t{candidate['template']} "
                f"behavior={cell['behavior']['behavior_gate_pass']} "
                f"source={cell['source_gate_pass']}",
                flush=True,
            )
        summary = {
            "schema_version": "phase1006_source_replication_model.v1",
            "phase": PHASE,
            "model": model_name,
            "model_loaded": True,
            "audit_protocol_digest": protocol["digest"],
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "model_class": info.model_class,
            },
            "cells": cells,
            "behavior_pass_count": sum(
                item["behavior"]["behavior_gate_pass"] for item in cells
            ),
            "source_pass_count": sum(
                item["source_gate_pass"] for item in cells
            ),
            "elapsed_seconds": time.time() - started,
        }
        write_json(AUDIT_ROOT / model_name / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        del model, tokenizer, device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        nargs="?",
        choices=MODELS,
    )
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        payload = prepare_audit_protocol()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.model is None:
        raise SystemExit("provide a model or --prepare")
    summary = run_model(args.model)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
