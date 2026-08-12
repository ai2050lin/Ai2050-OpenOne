#!/usr/bin/env python3
"""Post-control coverage audit for the Phase 1002 early source state.

This test was triggered by the preregistered semantic-mismatch control failure.
It does not select a best position set. It compares fixed, nested causal
contexts to determine whether two entity positions are a complete semantic
packet or only a strong control point.
"""
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

import phase1000_scpg_discovery as scpg
from model_utils import get_layers, load_model, release_model
from phase1002_multitoken_frozen_topology import directional_rows, read_json
from phase1002_multitoken_protocol import (
    COLORS,
    MODELS,
    OUT_ROOT,
    write_json,
    write_jsonl,
)
from phase1002_multitoken_source_controls import mismatch_batches


PHASE = 1002
SCOPES = (
    "entity_pair",
    "semantic_anchors",
    "fact_to_query",
    "causal_prompt",
    "causal_all",
)


def capture_full_state(
    model,
    device,
    rows: list[dict[str, Any]],
    depth: int,
    candidate_ids: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention = scpg.case_tensors(rows, device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    candidates = scpg.candidate_tensor(
        output.logits[:, -1, :], candidate_ids
    ).detach()
    hidden = output.hidden_states[depth].detach()
    del output, input_ids, attention
    return candidates, hidden


def fixed_scopes(case: dict[str, Any]) -> dict[str, list[int]]:
    roles = case["role_positions"]
    total_length = len(case["input_ids"])
    semantic_step = int(case["semantic_step"])
    prompt_length = total_length - semantic_step
    entity_positions = sorted([
        int(roles["slot0_entity"]),
        int(roles["slot1_entity"]),
    ])
    semantic_anchors = sorted(set(entity_positions + [
        int(roles["slot0_color"]),
        int(roles["slot1_color"]),
        int(roles["query_name"]),
    ]))
    first_fact = min(
        int(roles[name])
        for name in (
            "slot0_entity",
            "slot1_entity",
            "slot0_color",
            "slot1_color",
        )
    )
    query_position = int(roles["query_name"])
    scopes = {
        "entity_pair": entity_positions,
        "semantic_anchors": semantic_anchors,
        "fact_to_query": list(range(first_fact, query_position + 1)),
        "causal_prompt": list(range(first_fact, prompt_length)),
        "causal_all": list(range(first_fact, total_length)),
    }
    if any(not values for values in scopes.values()):
        raise RuntimeError("empty source coverage scope")
    if any(
        position < 0 or position >= total_length
        for values in scopes.values()
        for position in values
    ):
        raise RuntimeError("source coverage position out of range")
    return scopes


def state_patch(
    depth: int,
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
    scope: str,
) -> dict[str, Any]:
    positions = [fixed_scopes(row)[scope] for row in rows]
    widths = {len(values) for values in positions}
    if len(widths) != 1:
        raise RuntimeError(f"nonuniform scope width for {scope}: {widths}")
    batch_index = torch.arange(len(rows), device=hidden.device)
    role_positions = {}
    role_vectors = {}
    for index in range(next(iter(widths))):
        name = f"p{index}"
        column = torch.tensor(
            [values[index] for values in positions],
            dtype=torch.long,
            device=hidden.device,
        )
        role_positions[name] = column
        role_vectors[name] = hidden[batch_index, column, :]
    return {
        "depth": depth,
        "role_positions": role_positions,
        "role_vectors": role_vectors,
    }


def prediction_colors(logits: torch.Tensor) -> list[str]:
    indices = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    return [COLORS[int(index)] for index in indices]


def run_model(model_name: str) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(
        prereg["frozen_phase1001_topology"][model_name]["source_depth"]
    )
    model = tokenizer = None
    started = time.time()
    result_rows = []
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        for split in ("discovery", "confirmation"):
            split_batches = list(
                mismatch_batches(
                    directional_rows(model_name, split),
                    exclude_target=True,
                )
            )
            for batch_number, (batch, donor_rows) in enumerate(
                split_batches, 1
            ):
                source_cases = [row["source"] for row in batch]
                target_cases = [row["target"] for row in batch]
                donor_cases = [row["source"] for row in donor_rows]
                candidate_ids = target_cases[0]["candidate_token_ids"]
                source_logits, source_hidden = capture_full_state(
                    model,
                    device,
                    source_cases,
                    source_depth,
                    candidate_ids,
                )
                target_logits, target_hidden = capture_full_state(
                    model,
                    device,
                    target_cases,
                    source_depth,
                    candidate_ids,
                )
                donor_logits, donor_hidden = capture_full_state(
                    model,
                    device,
                    donor_cases,
                    source_depth,
                    candidate_ids,
                )
                clean_source_margin = scpg.semantic_margin(
                    source_logits, batch
                )
                clean_target_margin = scpg.semantic_margin(
                    target_logits, batch
                )
                clean_span = clean_source_margin - clean_target_margin
                target_predictions = prediction_colors(target_logits)

                conditions = []
                for scope in SCOPES:
                    conditions.append((
                        f"correct_{scope}",
                        "correct",
                        scope,
                        state_patch(
                            source_depth,
                            target_cases,
                            source_hidden,
                            scope,
                        ),
                    ))
                    conditions.append((
                        f"mismatch_{scope}",
                        "mismatch",
                        scope,
                        state_patch(
                            source_depth,
                            target_cases,
                            donor_hidden,
                            scope,
                        ),
                    ))
                conditions.append((
                    "target_noop_causal_all",
                    "noop",
                    "causal_all",
                    state_patch(
                        source_depth,
                        target_cases,
                        target_hidden,
                        "causal_all",
                    ),
                ))

                for condition, origin, scope, patch in conditions:
                    patched_logits = scpg.forward_candidate(
                        model,
                        layers,
                        device,
                        target_cases,
                        candidate_ids,
                        source_patch=patch,
                    )
                    patched_margin = scpg.semantic_margin(
                        patched_logits, batch
                    )
                    predictions = prediction_colors(patched_logits)
                    for index, row in enumerate(batch):
                        donor_gold = donor_rows[index]["source"]["gold"]
                        result_rows.append({
                            "schema_version": (
                                "phase1002_source_context_coverage_row.v1"
                            ),
                            "phase": PHASE,
                            "model": model_name,
                            "split": split,
                            "pair_id": row["pair_id"],
                            "direction": row["direction"],
                            "template": row["template"],
                            "condition": condition,
                            "origin": origin,
                            "scope": scope,
                            "scope_width": len(
                                fixed_scopes(row["target"])[scope]
                            ),
                            "source_gold": row["source"]["gold"],
                            "target_gold": row["target"]["gold"],
                            "donor_gold": donor_gold,
                            "donor_pair_id": donor_rows[index]["pair_id"],
                            "donor_direction": donor_rows[index]["direction"],
                            "prediction": predictions[index],
                            "source_prediction": (
                                predictions[index]
                                == row["source"]["gold"]
                            ),
                            "target_prediction": (
                                predictions[index]
                                == row["target"]["gold"]
                            ),
                            "donor_prediction": (
                                predictions[index] == donor_gold
                            ),
                            "normalized_transfer": float(
                                (
                                    patched_margin[index]
                                    - clean_target_margin[index]
                                )
                                / max(abs(float(clean_span[index])), 1e-8)
                            ),
                            "noop_prediction_agreement": (
                                predictions[index]
                                == target_predictions[index]
                                if origin == "noop"
                                else None
                            ),
                            "noop_max_abs_difference": (
                                float(torch.max(torch.abs(
                                    patched_logits[index]
                                    - target_logits[index]
                                )))
                                if origin == "noop"
                                else None
                            ),
                        })
                    del patched_logits
                del (
                    source_logits,
                    target_logits,
                    donor_logits,
                    source_hidden,
                    target_hidden,
                    donor_hidden,
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

    split_summary = {}
    conditions = [
        f"{origin}_{scope}"
        for scope in SCOPES
        for origin in ("correct", "mismatch")
    ] + ["target_noop_causal_all"]
    for split in ("discovery", "confirmation"):
        split_summary[split] = {}
        for condition in conditions:
            values = [
                row for row in result_rows
                if row["split"] == split
                and row["condition"] == condition
            ]
            item = {
                "n": len(values),
                "median_scope_width": float(np.median([
                    row["scope_width"] for row in values
                ])),
                "source_rate": float(np.mean([
                    row["source_prediction"] for row in values
                ])),
                "target_rate": float(np.mean([
                    row["target_prediction"] for row in values
                ])),
                "donor_rate": float(np.mean([
                    row["donor_prediction"] for row in values
                ])),
                "median_normalized_transfer": float(np.median([
                    row["normalized_transfer"] for row in values
                ])),
            }
            if condition == "target_noop_causal_all":
                item["prediction_agreement"] = float(np.mean([
                    row["noop_prediction_agreement"] for row in values
                ]))
                item["max_abs_difference"] = float(max(
                    row["noop_max_abs_difference"] for row in values
                ))
            split_summary[split][condition] = item

    checks = {}
    donor_audit = {}
    for split in ("discovery", "confirmation"):
        values = split_summary[split]
        donor_rows = [
            row for row in result_rows
            if row["split"] == split
            and row["condition"] == "mismatch_entity_pair"
        ]
        unique_donors = {
            (row["donor_pair_id"], row["donor_direction"])
            for row in donor_rows
        }
        donor_audit[split] = {
            "recipient_count": len(donor_rows),
            "unique_donor_count": len(unique_donors),
            "unique_donor_fraction": (
                len(unique_donors) / max(len(donor_rows), 1)
            ),
            "all_donor_answers_exclude_source_and_target": all(
                row["donor_gold"] not in (
                    row["source_gold"],
                    row["target_gold"],
                )
                for row in donor_rows
            ),
        }
        checks[split] = {
            "correct_full_source": (
                values["correct_causal_all"]["source_rate"] >= 0.80
            ),
            "mismatch_full_donor": (
                values["mismatch_causal_all"]["donor_rate"] >= 0.80
            ),
            "mismatch_full_not_original_source": (
                values["mismatch_causal_all"]["source_rate"] <= 0.25
            ),
            "noop": (
                values["target_noop_causal_all"]["prediction_agreement"]
                >= 0.99
            ),
        }
    summary = {
        "schema_version": "phase1002_source_context_coverage_summary.v1",
        "phase": PHASE,
        "implementation_revision": 3,
        "model": model_name,
        "status": "complete",
        "source_depth": source_depth,
        "direction_count_per_split": 256,
        "scope_order": list(SCOPES),
        "scope_selection_uses_results": False,
        "triggered_by_preregistered_source_control_failure": True,
        "revision_audit": {
            "revision_1": (
                "Qwen pilot allowed donors equal to target answers; "
                "preserved under source_context_coverage_overlap_pilot."
            ),
            "revision_2": (
                "The causal suffix began at the first entity and omitted "
                "an earlier color in the color-first template; preserved "
                "under source_context_coverage_first_entity_pilot."
            ),
            "revision_3": (
                "Formal run uses third-answer donors and begins causal "
                "ranges at the earliest entity or color fact position."
            ),
        },
        "donor_audit": donor_audit,
        "split_summary": split_summary,
        "checks": checks,
        "context_coverage_pass": all(
            all(values.values()) for values in checks.values()
        ),
        "thresholds": {
            "full_source_or_donor_min": 0.80,
            "wrong_original_source_max": 0.25,
            "noop_agreement_min": 0.99,
        },
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "This adaptive follow-up distinguishes a compact control point "
            "from a complete semantic packet. It does not establish that "
            "the full causal suffix is minimal or neuron-localized."
        ),
    }
    model_root = OUT_ROOT / "source_context_coverage" / model_name
    write_jsonl(model_root / "rows.jsonl", result_rows)
    write_json(model_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def aggregate() -> dict[str, Any]:
    summaries = {
        model_name: read_json(
            OUT_ROOT
            / "source_context_coverage"
            / model_name
            / "summary.json"
        )
        for model_name in MODELS
        if (
            OUT_ROOT
            / "source_context_coverage"
            / model_name
            / "summary.json"
        ).exists()
    }
    payload = {
        "schema_version": (
            "phase1002_source_context_coverage_cross_model.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "pass_count": sum(
            summary["context_coverage_pass"]
            for summary in summaries.values()
        ),
        "cross_model_pass": (
            len(summaries) == len(MODELS)
            and sum(
                summary["context_coverage_pass"]
                for summary in summaries.values()
            ) >= 2
        ),
    }
    write_json(
        OUT_ROOT / "source_context_coverage" / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(args.model)
    else:
        raise SystemExit("provide --model or --aggregate")


if __name__ == "__main__":
    main()
