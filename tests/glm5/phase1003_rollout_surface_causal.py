#!/usr/bin/env python3
"""Five-anchor causal rollout across frozen Phase1003 output surfaces."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1003_anchor_subset_exhaustive import choose_donors
from phase1003_crossparadigm_protocol import (
    ANCHOR_ROLES,
    MODELS,
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    selected_directional_rows,
    write_json,
    write_jsonl,
)
from phase1003_rollout_surface_behavior import (
    semantic_label,
    strip_eos,
)
from phase1003_rollout_surface_protocol import (
    ROLLOUT_ROOT,
    SURFACES,
)
from phase1003_structural_stress_causal import (
    capture_depth,
    generate,
    paired_batches,
    patch_spec,
)


def donor_map(
    model_name: str, split: str
) -> tuple[dict[str, str], dict[str, Any]]:
    rows = selected_directional_rows(
        model_name, "color", split
    )
    donors, audit = choose_donors(
        rows, model_name, "color", split
    )
    mapping = {
        row["target"]["record_id"]: donor["record_id"]
        for row, donor in zip(rows, donors)
    }
    if len(mapping) != 64:
        raise RuntimeError(
            f"{model_name}/{split}: donor map collision"
        )
    return mapping, audit


def conditions_for_batch(
    model,
    tokenizer,
    layers,
    device,
    source_depth: int,
    batch: list[dict[str, Any]],
    donor_batch: list[dict[str, Any]],
    effective_eos: list[int],
) -> dict[str, list[list[int]]]:
    _, target_hidden = capture_depth(
        model, layers, device, batch, source_depth
    )
    _, donor_hidden = capture_depth(
        model, layers, device, donor_batch, source_depth
    )
    patches = {
        "target_noop": patch_spec(
            source_depth,
            list(ANCHOR_ROLES),
            [],
            batch,
            target_hidden,
            donor_batch,
            donor_hidden,
        ),
        "full_source": patch_spec(
            source_depth,
            list(ANCHOR_ROLES),
            list(ANCHOR_ROLES),
            batch,
            target_hidden,
            donor_batch,
            donor_hidden,
        ),
    }
    result = {
        "clean": generate(
            model,
            tokenizer,
            layers,
            device,
            batch,
            effective_eos,
            None,
        )
    }
    for condition, patch in patches.items():
        result[condition] = generate(
            model,
            tokenizer,
            layers,
            device,
            batch,
            effective_eos,
            patch,
        )
    del target_hidden, donor_hidden
    return result


def run_surface_split(
    model,
    tokenizer,
    layers,
    device,
    model_name: str,
    surface: str,
    split: str,
    source_depth: int,
    cases: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    effective_eos: list[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eos_set = set(effective_eos)
    result_rows = []
    all_batches = list(
        paired_batches(cases, donors, batch_size)
    )
    for batch_number, (batch, donor_batch) in enumerate(
        all_batches, 1
    ):
        suffixes = conditions_for_batch(
            model,
            tokenizer,
            layers,
            device,
            source_depth,
            batch,
            donor_batch,
            effective_eos,
        )
        clean_stripped = [
            strip_eos(values, eos_set)[0]
            for values in suffixes["clean"]
        ]
        for condition in (
            "clean",
            "target_noop",
            "full_source",
        ):
            for index, target in enumerate(batch):
                ids, eos_position = strip_eos(
                    suffixes[condition][index], eos_set
                )
                prediction = semantic_label(ids, target)
                target_expected = [
                    int(value)
                    for value in target["answer_token_ids"]
                ]
                donor_expected = [
                    int(value)
                    for value in donor_batch[index][
                        "answer_token_ids"
                    ]
                ]
                result_rows.append({
                    "schema_version": (
                        "phase1003_rollout_surface_causal_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "surface": surface,
                    "split": split,
                    "record_id": target["record_id"],
                    "donor_record_id": (
                        donor_batch[index]["record_id"]
                    ),
                    "condition": condition,
                    "target_gold": target["gold"],
                    "donor_gold": donor_batch[index]["gold"],
                    "prediction": prediction,
                    "target_semantic": (
                        prediction == target["gold"]
                    ),
                    "donor_semantic": (
                        prediction == donor_batch[index]["gold"]
                    ),
                    "target_exact": ids == target_expected,
                    "donor_exact": ids == donor_expected,
                    "eos_observed": eos_position is not None,
                    "eos_position": eos_position,
                    "target_eos_boundary": (
                        eos_position == len(target_expected)
                    ),
                    "donor_eos_boundary": (
                        eos_position == len(donor_expected)
                    ),
                    "noop_sequence_agreement": (
                        ids == clean_stripped[index]
                        if condition == "target_noop"
                        else None
                    ),
                    "generated_ids": ids,
                    "generated_text": tokenizer.decode(
                        ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                })
        print(
            f"[causal/{model_name}/{surface}/{split}] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )
    condition_summary = {}
    for condition in ("clean", "target_noop", "full_source"):
        values = [
            row for row in result_rows
            if row["condition"] == condition
        ]
        item = {
            "n": len(values),
            "target_semantic_rate": float(np.mean([
                row["target_semantic"] for row in values
            ])),
            "donor_semantic_rate": float(np.mean([
                row["donor_semantic"] for row in values
            ])),
            "target_exact_rate": float(np.mean([
                row["target_exact"] for row in values
            ])),
            "donor_exact_rate": float(np.mean([
                row["donor_exact"] for row in values
            ])),
            "eos_observed_rate": float(np.mean([
                row["eos_observed"] for row in values
            ])),
            "target_eos_boundary_rate": float(np.mean([
                row["target_eos_boundary"] for row in values
            ])),
            "donor_eos_boundary_rate": float(np.mean([
                row["donor_eos_boundary"] for row in values
            ])),
        }
        if condition == "target_noop":
            item["noop_sequence_agreement"] = float(np.mean([
                row["noop_sequence_agreement"] for row in values
            ]))
        condition_summary[condition] = item
    prereg = read_json(
        ROLLOUT_ROOT / "preregistered_protocol.json"
    )
    full = condition_summary["full_source"]
    summary = {
        "conditions": condition_summary,
        "causal_rollout_gate": (
            condition_summary["target_noop"][
                "noop_sequence_agreement"
            ] >= prereg["noop_sequence_gate"]
            and full["donor_semantic_rate"]
            >= prereg["full_anchor_semantic_gate"]
            and full["donor_exact_rate"]
            >= prereg["full_anchor_exact_gate"]
            and full["donor_eos_boundary_rate"]
            >= prereg["eos_observed_gate"]
        ),
    }
    return result_rows, summary


def run_model(
    model_name: str, batch_size: int
) -> dict[str, Any]:
    behavior = read_json(
        ROLLOUT_ROOT
        / "behavior"
        / model_name
        / "summary.json"
    )
    surfaces = [
        surface
        for surface in SURFACES
        if behavior["surface_gates"][surface]
    ]
    cases = read_jsonl(
        ROLLOUT_ROOT
        / "protocol"
        / model_name
        / "cases.jsonl"
    )
    by_record = {
        case["record_id"]: case for case in cases
    }
    prereg_main = read_json(
        OUT_ROOT / "preregistered_protocol.json"
    )
    source_depth = int(
        prereg_main["source_depths"][model_name]
    )
    model = tokenizer = None
    started = time.time()
    surface_summaries = {}
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        effective_eos = eos_ids(model, tokenizer)
        for surface in surfaces:
            split_summaries = {}
            all_rows = []
            donor_audits = {}
            for split in ("discovery", "confirmation"):
                selected = [
                    case
                    for case in cases
                    if case["surface"] == surface
                    and case["split"] == split
                ]
                mapping, donor_audit = donor_map(
                    model_name, split
                )
                donors = [
                    by_record[
                        f"{mapping[case['base_record_id']]}"
                        f".surface.{surface}"
                    ]
                    for case in selected
                ]
                donor_audits[split] = donor_audit
                rows, summary = run_surface_split(
                    model,
                    tokenizer,
                    layers,
                    device,
                    model_name,
                    surface,
                    split,
                    source_depth,
                    selected,
                    donors,
                    effective_eos,
                    batch_size,
                )
                all_rows.extend(rows)
                split_summaries[split] = summary
            summary = {
                "schema_version": (
                    "phase1003_rollout_surface_causal_summary.v1"
                ),
                "phase": PHASE,
                "model": model_name,
                "surface": surface,
                "status": "complete",
                "source_depth": source_depth,
                "roles": list(ANCHOR_ROLES),
                "donor_audits": donor_audits,
                "splits": split_summaries,
                "rollout_surface_pass": all(
                    split_summaries[split][
                        "causal_rollout_gate"
                    ]
                    for split in ("discovery", "confirmation")
                ),
                "claim_boundary": (
                    "Full five-anchor transport is tested on a fixed "
                    "surface. This does not identify a minimal rollout "
                    "state or unrestricted explanation mechanism."
                ),
            }
            root = (
                ROLLOUT_ROOT
                / "causal"
                / model_name
                / surface
            )
            write_jsonl(root / "rows.jsonl", all_rows)
            write_json(root / "summary.json", summary)
            surface_summaries[surface] = summary
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": (
            "phase1003_rollout_surface_causal_model.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "behavior_passing_surfaces": surfaces,
        "surfaces": surface_summaries,
        "pass_count": sum(
            summary["rollout_surface_pass"]
            for summary in surface_summaries.values()
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json(
        ROLLOUT_ROOT / "causal" / model_name / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def aggregate() -> dict[str, Any]:
    summaries = {}
    for model_name in MODELS:
        path = (
            ROLLOUT_ROOT
            / "causal"
            / model_name
            / "summary.json"
        )
        if path.exists():
            summaries[model_name] = read_json(path)
    prereg = read_json(
        ROLLOUT_ROOT / "preregistered_protocol.json"
    )
    cross_surface = {}
    for surface in SURFACES:
        available = [
            summary["surfaces"][surface]
            for summary in summaries.values()
            if surface in summary["surfaces"]
        ]
        cross_surface[surface] = {
            "behavior_qualified_model_count": len(available),
            "causal_pass_count": sum(
                item["rollout_surface_pass"]
                for item in available
            ),
            "models": {
                item["model"]: {
                    "pass": item["rollout_surface_pass"],
                    "confirmation_donor_semantic_rate": item[
                        "splits"
                    ]["confirmation"]["conditions"][
                        "full_source"
                    ]["donor_semantic_rate"],
                    "confirmation_donor_exact_rate": item[
                        "splits"
                    ]["confirmation"]["conditions"][
                        "full_source"
                    ]["donor_exact_rate"],
                    "confirmation_donor_eos_boundary_rate": item[
                        "splits"
                    ]["confirmation"]["conditions"][
                        "full_source"
                    ]["donor_eos_boundary_rate"],
                }
                for item in available
            },
        }
    payload = {
        "schema_version": (
            "phase1003_rollout_surface_causal_aggregate.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "cross_surface": cross_surface,
        "cross_model_causal_gates": {
            surface: (
                item["causal_pass_count"]
                >= prereg["cross_model_minimum"]
            )
            for surface, item in cross_surface.items()
        },
    }
    write_json(ROLLOUT_ROOT / "causal" / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(args.model, args.batch_size)
    else:
        raise SystemExit("provide --model or --aggregate")


if __name__ == "__main__":
    main()
