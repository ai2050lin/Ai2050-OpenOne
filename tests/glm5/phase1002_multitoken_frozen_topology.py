#!/usr/bin/env python3
"""Replicate the Phase 1001 functional topology at a multi-token semantic step.

No Phase 1002 activation is used to select layers, components, or positions.
The only receiver events evaluated here were frozen before this denominator was
created.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1000_scpg_discovery as scpg
from model_utils import get_layers, load_model, release_model
from phase1002_multitoken_protocol import (
    COLORS,
    MODELS,
    OUT_ROOT,
    write_json,
    write_jsonl,
)


PHASE = 1002


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def event_from_id(event_id: str) -> dict[str, Any]:
    layer_text, component, role = event_id.split(".", 2)
    layer_number = int(layer_text[1:])
    return {
        "event_id": event_id,
        "block_index": layer_number - 1,
        "layer_number": layer_number,
        "component": component,
        "role": role,
    }


def at_answer_step(case: dict[str, Any], step: int) -> dict[str, Any]:
    result = dict(case)
    result["input_ids"] = (
        list(case["input_ids"]) + list(case["answer_token_ids"][:step])
    )
    result["input_token_count"] = len(result["input_ids"])
    result["role_positions"] = dict(case["role_positions"])
    result["role_positions"]["answer_boundary"] = (
        result["input_token_count"] - 1
    )
    result["evaluated_answer_step"] = step
    result["evaluated_step_role"] = case["answer_step_roles"][step]
    return result


def directional_rows(
    model_name: str, split: str
) -> list[dict[str, Any]]:
    protocol_root = OUT_ROOT / "protocol" / model_name
    cases = {
        row["record_id"]: row
        for row in read_jsonl(protocol_root / "cases.jsonl")
    }
    selected = read_jsonl(
        protocol_root / f"{split}_selected_pairs.jsonl"
    )
    rows = []
    for pair in selected:
        arm0 = cases[pair["arm0_record_id"]]
        arm1 = cases[pair["arm1_record_id"]]
        for direction, source, target in (
            ("arm0_to_arm1", arm0, arm1),
            ("arm1_to_arm0", arm1, arm0),
        ):
            step = int(target["semantic_step"])
            rows.append({
                "partition": split,
                "split": split,
                "pair_id": pair["pair_id"],
                "direction": direction,
                "template": int(target["template"]),
                "source": at_answer_step(source, step),
                "target": at_answer_step(target, step),
            })
    return sorted(
        rows,
        key=lambda row: (
            row["template"], row["pair_id"], row["direction"]
        ),
    )


def batches(
    rows: list[dict[str, Any]], batch_size: int
):
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["template"]),
            int(row["target"]["input_token_count"]),
        )
        groups[key].append(row)
    for key in sorted(groups):
        values = groups[key]
        for start in range(0, len(values), batch_size):
            yield values[start:start + batch_size]


def margin(logits: torch.Tensor, rows: list[dict[str, Any]]) -> torch.Tensor:
    color_index = {color: index for index, color in enumerate(COLORS)}
    batch_index = torch.arange(len(rows), device=logits.device)
    source_index = torch.tensor(
        [color_index[row["source"]["gold"]] for row in rows],
        device=logits.device,
    )
    target_index = torch.tensor(
        [color_index[row["target"]["gold"]] for row in rows],
        device=logits.device,
    )
    return (
        logits[batch_index, source_index]
        - logits[batch_index, target_index]
    )


def predictions(logits: torch.Tensor) -> list[str]:
    indices = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    return [COLORS[int(index)] for index in indices]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "source_clean_source_rate": float(np.mean([
            row["source_clean_prediction"] == row["source_gold"]
            for row in rows
        ])),
        "target_clean_target_rate": float(np.mean([
            row["target_clean_prediction"] == row["target_gold"]
            for row in rows
        ])),
        "source_do_source_rate": float(np.mean([
            row["source_do_prediction"] == row["source_gold"]
            for row in rows
        ])),
        "restore_target_rate": float(np.mean([
            row["restore_prediction"] == row["target_gold"]
            for row in rows
        ])),
        "restore_source_rate": float(np.mean([
            row["restore_prediction"] == row["source_gold"]
            for row in rows
        ])),
        "sufficiency_source_rate": float(np.mean([
            row["sufficiency_prediction"] == row["source_gold"]
            for row in rows
        ])),
        "median_source_transfer": float(np.median([
            row["source_transfer"] for row in rows
        ])),
        "mean_source_transfer": float(np.mean([
            row["source_transfer"] for row in rows
        ])),
        "median_mediation_fraction": float(np.median([
            row["mediation_fraction"] for row in rows
        ])),
        "mean_mediation_fraction": float(np.mean([
            row["mediation_fraction"] for row in rows
        ])),
        "median_sufficiency_transfer": float(np.median([
            row["sufficiency_transfer"] for row in rows
        ])),
        "mean_sufficiency_transfer": float(np.mean([
            row["sufficiency_transfer"] for row in rows
        ])),
        "noop_prediction_agreement": float(np.mean([
            row["noop_prediction"] == row["source_do_prediction"]
            for row in rows
        ])),
        "noop_candidate_max_abs_difference": float(max(
            row["noop_candidate_max_abs_difference"] for row in rows
        )),
        "by_template": {
            str(template): {
                "n": len(values),
                "source_do_source_rate": float(np.mean([
                    row["source_do_prediction"] == row["source_gold"]
                    for row in values
                ])),
                "restore_target_rate": float(np.mean([
                    row["restore_prediction"] == row["target_gold"]
                    for row in values
                ])),
                "median_mediation_fraction": float(np.median([
                    row["mediation_fraction"] for row in values
                ])),
            }
            for template in range(4)
            if (
                values := [
                    row for row in rows
                    if int(row["template"]) == template
                ]
            )
        },
    }


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    behavior = read_json(
        OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    if not behavior["behavior_gate_pass"]:
        raise RuntimeError(
            f"{model_name}: behavior gate did not pass; causal claims disabled"
        )

    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    frozen = prereg["frozen_phase1001_topology"][model_name]
    variants = {
        name: [event_from_id(event_id) for event_id in event_ids]
        for name, event_ids in frozen["variants"].items()
    }
    events_by_id = {
        event["event_id"]: event
        for events in variants.values()
        for event in events
    }
    all_events = list(events_by_id.values())
    source_depth = int(frozen["source_depth"])

    model = tokenizer = None
    started = time.time()
    all_rows = []
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        for split in ("discovery", "confirmation"):
            split_rows = directional_rows(model_name, split)
            split_batches = list(batches(split_rows, batch_size))
            for batch_number, batch in enumerate(split_batches, 1):
                source_cases = [row["source"] for row in batch]
                target_cases = [row["target"] for row in batch]
                candidate_ids = target_cases[0]["candidate_token_ids"]
                source_logits, source_residuals = scpg.capture_residuals(
                    model,
                    device,
                    source_cases,
                    (source_depth,),
                    candidate_ids,
                )
                target_logits, target_components = scpg.capture_components(
                    model,
                    layers,
                    device,
                    target_cases,
                    all_events,
                    candidate_ids,
                )
                source_patch = scpg.source_patch_spec(
                    source_depth,
                    target_cases,
                    source_residuals[source_depth],
                    "joint",
                )
                do_logits, do_components = scpg.capture_components(
                    model,
                    layers,
                    device,
                    target_cases,
                    all_events,
                    candidate_ids,
                    source_patch=source_patch,
                )

                source_margin = margin(source_logits, batch)
                target_margin = margin(target_logits, batch)
                do_margin = margin(do_logits, batch)
                source_predictions = predictions(source_logits)
                target_predictions = predictions(target_logits)
                do_predictions = predictions(do_logits)

                for variant_name, variant_events in variants.items():
                    target_patches = [{
                        "event": event,
                        "vectors": target_components[event["event_id"]],
                    } for event in variant_events]
                    do_patches = [{
                        "event": event,
                        "vectors": do_components[event["event_id"]],
                    } for event in variant_events]
                    restore_logits = scpg.forward_candidate(
                        model,
                        layers,
                        device,
                        target_cases,
                        candidate_ids,
                        source_patch=source_patch,
                        receiver_patches=target_patches,
                    )
                    sufficiency_logits = scpg.forward_candidate(
                        model,
                        layers,
                        device,
                        target_cases,
                        candidate_ids,
                        receiver_patches=do_patches,
                    )
                    noop_logits = scpg.forward_candidate(
                        model,
                        layers,
                        device,
                        target_cases,
                        candidate_ids,
                        source_patch=source_patch,
                        receiver_patches=do_patches,
                    )
                    restore_margin = margin(restore_logits, batch)
                    sufficiency_margin = margin(sufficiency_logits, batch)
                    restore_predictions = predictions(restore_logits)
                    sufficiency_predictions = predictions(
                        sufficiency_logits
                    )
                    noop_predictions = predictions(noop_logits)

                    for index, item in enumerate(batch):
                        clean_span = float(
                            source_margin[index] - target_margin[index]
                        )
                        source_effect = float(
                            do_margin[index] - target_margin[index]
                        )
                        all_rows.append({
                            "schema_version": (
                                "phase1002_frozen_topology_row.v1"
                            ),
                            "phase": PHASE,
                            "model": model_name,
                            "split": split,
                            "pair_id": item["pair_id"],
                            "direction": item["direction"],
                            "template": item["template"],
                            "variant": variant_name,
                            "event_ids": [
                                event["event_id"]
                                for event in variant_events
                            ],
                            "source_gold": item["source"]["gold"],
                            "target_gold": item["target"]["gold"],
                            "source_clean_prediction": (
                                source_predictions[index]
                            ),
                            "target_clean_prediction": (
                                target_predictions[index]
                            ),
                            "source_do_prediction": do_predictions[index],
                            "restore_prediction": (
                                restore_predictions[index]
                            ),
                            "sufficiency_prediction": (
                                sufficiency_predictions[index]
                            ),
                            "noop_prediction": noop_predictions[index],
                            "source_margin": float(source_margin[index]),
                            "target_margin": float(target_margin[index]),
                            "source_do_margin": float(do_margin[index]),
                            "restore_margin": float(
                                restore_margin[index]
                            ),
                            "sufficiency_margin": float(
                                sufficiency_margin[index]
                            ),
                            "clean_margin_span": clean_span,
                            "source_effect": source_effect,
                            "source_transfer": (
                                source_effect
                                / max(abs(clean_span), 1e-8)
                            ),
                            "mediation_fraction": float(
                                (do_margin[index] - restore_margin[index])
                                / max(abs(source_effect), 1e-8)
                            ),
                            "sufficiency_transfer": float(
                                (
                                    sufficiency_margin[index]
                                    - target_margin[index]
                                )
                                / max(abs(clean_span), 1e-8)
                            ),
                            "noop_candidate_max_abs_difference": float(
                                torch.max(torch.abs(
                                    noop_logits[index] - do_logits[index]
                                ))
                            ),
                        })
                    del (
                        restore_logits,
                        sufficiency_logits,
                        noop_logits,
                    )
                del (
                    source_logits,
                    source_residuals,
                    target_logits,
                    target_components,
                    do_logits,
                    do_components,
                )
                print(
                    f"[{model_name}/{split}] "
                    f"{batch_number}/{len(split_batches)}",
                    flush=True,
                )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()

    model_root = OUT_ROOT / "frozen_topology" / model_name
    write_jsonl(model_root / "rows.jsonl", all_rows)
    split_summary = {}
    for split in ("discovery", "confirmation"):
        split_summary[split] = {}
        for variant_name in variants:
            values = [
                row for row in all_rows
                if row["split"] == split
                and row["variant"] == variant_name
            ]
            split_summary[split][variant_name] = summarize(values)

    thresholds = prereg["primary_thresholds"]
    primary_checks = {}
    for split in ("discovery", "confirmation"):
        values = split_summary[split]["primary"]
        primary_checks[split] = {
            "source_do": (
                values["source_do_source_rate"]
                >= thresholds["source_do_semantic_flip_rate"]
            ),
            "restore": (
                values["restore_target_rate"]
                >= thresholds["frozen_topology_semantic_restore_rate"]
            ),
            "mediation": (
                values["median_mediation_fraction"]
                >= thresholds["semantic_mediation_median"]
            ),
            "noop": values["noop_prediction_agreement"] == 1.0,
        }
    summary = {
        "schema_version": "phase1002_frozen_topology_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "source_depth": source_depth,
        "variants": {
            name: [event["event_id"] for event in events]
            for name, events in variants.items()
        },
        "selection_uses_phase1002": False,
        "split_summary": split_summary,
        "thresholds": thresholds,
        "primary_checks": primary_checks,
        "primary_pass": all(
            all(checks.values()) for checks in primary_checks.values()
        ),
        "quantized_8bit": True,
        "elapsed_seconds": time.time() - started,
    }
    write_json(model_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def aggregate() -> dict[str, Any]:
    summaries = {
        model_name: read_json(
            OUT_ROOT / "frozen_topology" / model_name / "summary.json"
        )
        for model_name in MODELS
        if (
            OUT_ROOT
            / "frozen_topology"
            / model_name
            / "summary.json"
        ).exists()
    }
    payload = {
        "schema_version": "phase1002_frozen_topology_cross_model.v1",
        "phase": PHASE,
        "models": summaries,
        "pass_count": sum(
            summary["primary_pass"] for summary in summaries.values()
        ),
        "cross_model_pass": (
            len(summaries) == len(MODELS)
            and sum(
                summary["primary_pass"]
                for summary in summaries.values()
            ) >= 2
        ),
    }
    write_json(OUT_ROOT / "frozen_topology" / "summary.json", payload)
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
