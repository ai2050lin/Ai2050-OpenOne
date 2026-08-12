#!/usr/bin/env python3
"""Phase1001 cross-model replication of the local binding topology.

This test compares functional topology, not neuron or layer identities:

1. a joint early entity-state intervention changes the selected color;
2. downstream attention/MLP receivers are discovered on one split;
3. receiver ranking and joint size are frozen;
4. the frozen topology is evaluated on disjoint worlds;
5. natural generation is checked after source intervention and restoration.

All models use the same 8-bit CUDA regime in this cross-model audit. The
full-precision Qwen3 result remains available in the preceding Phase1001 runs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase1000_scpg_discovery as scpg
from model_utils import get_layers, get_model_info, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1000_factorial_binding_protocol import render_user_prompt
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for


PHASE = 1001
MODELS = ("qwen3", "glm4", "deepseek7b")
COLORS = ("red", "blue", "green", "yellow")
NAMES = (
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Jack", "Kelly", "Paul", "Ruby", "Sam", "Blake", "Leo", "Will",
    "Iris", "Liam", "Maya", "Nora", "Oscar", "Quinn", "Tina", "Uma",
)
WORLD_COUNT_PER_SPLIT = 16
PAIRS_PER_STRATUM = 4
NATURAL_PER_STRATUM = 2
SOURCE_DEPTHS = (1, 2, 4, 8, 16)
RECEIVER_LIMIT = 12
JOINT_SIZES = (1, 2, 4, 8, 12)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1001_attention_physical_decomposition"
    / "cross_model_topology_causal_screen"
)


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temp.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_order(value: str, salt: str) -> str:
    return hashlib.sha256(f"phase1001:{salt}:{value}".encode("utf-8")).hexdigest()


def one_token_id(tokenizer, text: str) -> int:
    values = tokenizer.encode(text, add_special_tokens=False)
    if len(values) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {values}")
    return int(values[0])


def positions_of(ids: list[int], token_id: int) -> list[int]:
    return [index for index, value in enumerate(ids) if value == token_id]


def build_protocol(model_name: str) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, Any],
]:
    tokenizer = tokenizer_for(model_name)
    prompt_name_ids = {
        name: one_token_id(tokenizer, " " + name) for name in NAMES
    }
    prompt_color_ids = {
        color: one_token_id(tokenizer, " " + color) for color in COLORS
    }
    candidate_ids = {
        color: one_token_id(tokenizer, color) for color in COLORS
    }
    if len(set(prompt_name_ids.values())) != len(prompt_name_ids):
        raise RuntimeError(f"{model_name}: name token collision")
    if len(set(prompt_color_ids.values())) != len(prompt_color_ids):
        raise RuntimeError(f"{model_name}: prompt color token collision")
    if len(set(candidate_ids.values())) != len(candidate_ids):
        raise RuntimeError(f"{model_name}: candidate token collision")

    rng = random.Random(1001_20260723)
    name_pairs = list(itertools.combinations(NAMES, 2))
    rng.shuffle(name_pairs)
    color_pairs = list(itertools.combinations(COLORS, 2))
    cases: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    token_lengths: dict[int, set[int]] = defaultdict(set)

    total_worlds = WORLD_COUNT_PER_SPLIT * 2
    for world in range(total_worlds):
        split = "discovery" if world < WORLD_COUNT_PER_SPLIT else "confirmation"
        split_world = (
            world if split == "discovery" else world - WORLD_COUNT_PER_SPLIT
        )
        world_id = f"{split[:1]}w{split_world:02d}"
        base_entities = list(name_pairs[world])
        base_colors = list(color_pairs[world % len(color_pairs)])
        for template, display_order, value_swap, query_role in itertools.product(
            range(4), (0, 1), (0, 1), (0, 1)
        ):
            arms = []
            for entity_swap in (0, 1):
                slot_entities = (
                    list(base_entities)
                    if entity_swap == 0
                    else [base_entities[1], base_entities[0]]
                )
                slot_colors = (
                    list(base_colors)
                    if value_swap == 0
                    else [base_colors[1], base_colors[0]]
                )
                query_entity = base_entities[query_role]
                query_slot = slot_entities.index(query_entity)
                gold = slot_colors[query_slot]
                foil = slot_colors[1 - query_slot]
                first_slot, second_slot = (
                    (0, 1) if display_order == 0 else (1, 0)
                )
                raw_prompt = render_user_prompt(
                    template,
                    slot_entities[first_slot],
                    slot_colors[first_slot],
                    slot_entities[second_slot],
                    slot_colors[second_slot],
                    query_entity,
                )
                rendered = render_chat(tokenizer, model_name, raw_prompt)
                ids = [
                    int(value)
                    for value in tokenizer.encode(
                        rendered, add_special_tokens=False
                    )
                ]
                fact_entity_positions = {}
                for entity in base_entities:
                    found = positions_of(ids, prompt_name_ids[entity])
                    expected = 2 if entity == query_entity else 1
                    if len(found) != expected:
                        raise RuntimeError(
                            f"{model_name}: entity position drift "
                            f"{world_id}/t{template}/{entity}/{found}"
                        )
                    fact_entity_positions[entity] = found[0]
                query_positions = positions_of(
                    ids, prompt_name_ids[query_entity]
                )
                color_positions = {
                    color: positions_of(ids, prompt_color_ids[color])
                    for color in base_colors
                }
                if any(len(values) != 1 for values in color_positions.values()):
                    raise RuntimeError(
                        f"{model_name}: color position drift "
                        f"{world_id}/t{template}/{color_positions}"
                    )
                for color, token_id in candidate_ids.items():
                    extended = tokenizer.encode(
                        rendered + color, add_special_tokens=False
                    )
                    if extended != ids + [token_id]:
                        raise RuntimeError(
                            f"{model_name}: candidate boundary drift "
                            f"{world_id}/t{template}/{color}"
                        )
                record_id = (
                    f"{model_name}.{world_id}.t{template}.o{display_order}."
                    f"v{value_swap}.q{query_role}.e{entity_swap}"
                )
                row = {
                    "schema_version": "phase1001_cross_model_case.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "record_id": record_id,
                    "world": world,
                    "world_id": world_id,
                    "split": split,
                    "template": template,
                    "display_order": display_order,
                    "value_swap": value_swap,
                    "query_role": query_role,
                    "entity_swap": entity_swap,
                    "base_entities": base_entities,
                    "base_colors": base_colors,
                    "slot_entities": slot_entities,
                    "slot_colors": slot_colors,
                    "query_entity": query_entity,
                    "query_slot": query_slot,
                    "gold": gold,
                    "foil": foil,
                    "raw_prompt": raw_prompt,
                    "rendered_prompt": rendered,
                    "input_ids": ids,
                    "input_token_count": len(ids),
                    "candidate_token_ids": candidate_ids,
                    "role_positions": {
                        "slot0_entity": fact_entity_positions[
                            slot_entities[0]
                        ],
                        "slot0_color": color_positions[slot_colors[0]][0],
                        "slot1_entity": fact_entity_positions[
                            slot_entities[1]
                        ],
                        "slot1_color": color_positions[slot_colors[1]][0],
                        "query_name": query_positions[-1],
                        "answer_boundary": len(ids) - 1,
                    },
                }
                cases.append(row)
                arms.append(row)
                token_lengths[template].add(len(ids))
            arm0, arm1 = arms
            changed = [
                index
                for index, (left, right) in enumerate(
                    zip(arm0["input_ids"], arm1["input_ids"])
                )
                if left != right
            ]
            expected_changed = sorted(
                (
                    arm0["role_positions"]["slot0_entity"],
                    arm0["role_positions"]["slot1_entity"],
                )
            )
            if changed != expected_changed:
                raise RuntimeError(
                    f"{model_name}: entity counterfactual drift "
                    f"{world_id}/{changed}/{expected_changed}"
                )
            if arm0["gold"] != arm1["foil"] or arm1["gold"] != arm0["foil"]:
                raise RuntimeError(f"{model_name}: answer swap drift {world_id}")
            pair_id = (
                f"{model_name}.{world_id}.t{template}.o{display_order}."
                f"v{value_swap}.q{query_role}"
            )
            pairs.append({
                "schema_version": "phase1001_cross_model_pair.v1",
                "phase": PHASE,
                "model": model_name,
                "pair_id": pair_id,
                "factor": "entity",
                "split": split,
                "world_id": world_id,
                "template": template,
                "display_order": display_order,
                "value_swap": value_swap,
                "query_role": query_role,
                "arm0_record_id": arm0["record_id"],
                "arm1_record_id": arm1["record_id"],
                "changed_positions": changed,
            })
    if any(len(lengths) != 1 for lengths in token_lengths.values()):
        raise RuntimeError(f"{model_name}: template length drift {token_lengths}")
    audit = {
        "schema_version": "phase1001_cross_model_protocol_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "pair_count": len(pairs),
        "world_count_per_split": WORLD_COUNT_PER_SPLIT,
        "candidate_token_ids": candidate_ids,
        "one_token_name_count": len(prompt_name_ids),
        "all_pairs_change_only_two_entity_positions": True,
        "all_pairs_swap_gold_and_foil": True,
        "template_token_lengths": {
            str(key): sorted(values) for key, values in token_lengths.items()
        },
        "case_digest": sha256_text(canonical(cases)),
        "pair_digest": sha256_text(canonical(pairs)),
    }
    return cases, pairs, candidate_ids, audit


def selected_pairs(
    pairs: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if pair["split"] != split:
            continue
        key = (
            int(pair["template"]),
            int(pair["display_order"]),
            int(pair["value_swap"]),
            int(pair["query_role"]),
        )
        strata[key].append(pair)
    selected = []
    for key, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda row: stable_order(
                row["pair_id"], f"{split}:{key}"
            ),
        )
        if len(ordered) < PAIRS_PER_STRATUM:
            raise RuntimeError(f"underfilled {split} stratum {key}")
        selected.extend(ordered[:PAIRS_PER_STRATUM])
    expected = 32 * PAIRS_PER_STRATUM
    if len(selected) != expected:
        raise RuntimeError(f"{split}: selected {len(selected)} != {expected}")
    return selected


def directional(
    pairs: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        arm0 = case_by_id[pair["arm0_record_id"]]
        arm1 = case_by_id[pair["arm1_record_id"]]
        rows.extend((
            {
                "pair_id": pair["pair_id"],
                "partition": split,
                "direction": "e0_to_e1",
                "source": arm0,
                "target": arm1,
            },
            {
                "pair_id": pair["pair_id"],
                "partition": split,
                "direction": "e1_to_e0",
                "source": arm1,
                "target": arm0,
            },
        ))
    return rows


def candidate_behavior(
    model,
    device,
    cases: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_rows = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[(row["split"], int(row["template"]))].append(row)
    for key, values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: row["record_id"])
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            input_ids, attention = scpg.case_tensors(batch, device)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    use_cache=False,
                    return_dict=True,
                )
            logits = scpg.candidate_tensor(
                output.logits[:, -1, :], candidate_ids
            )
            predictions = scpg.prediction_colors(logits)
            for index, row in enumerate(batch):
                result_rows.append({
                    "schema_version": "phase1001_cross_model_behavior_row.v1",
                    "phase": PHASE,
                    "model": row["model"],
                    "record_id": row["record_id"],
                    "split": row["split"],
                    "template": row["template"],
                    "gold": row["gold"],
                    "prediction": predictions[index],
                    "correct": predictions[index] == row["gold"],
                })
            del output, logits, input_ids, attention
        print(f"[behavior] {key}", flush=True)
    summaries = {}
    for split in ("discovery", "confirmation"):
        values = [row for row in result_rows if row["split"] == split]
        summaries[split] = {
            "n": len(values),
            "accuracy": float(np.mean([row["correct"] for row in values])),
            "template_accuracy": {
                str(template): float(np.mean([
                    row["correct"] for row in values
                    if int(row["template"]) == template
                ]))
                for template in range(4)
            },
        }
    return result_rows, {
        "schema_version": "phase1001_cross_model_behavior_summary.v1",
        "phase": PHASE,
        "model": cases[0]["model"],
        "split_summary": summaries,
        "gate_threshold": 0.95,
        "gate_pass": all(
            summary["accuracy"] >= 0.95 for summary in summaries.values()
        ),
    }


def causal_screen_subset(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Take both directions from one frozen pair in each factorial stratum."""
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = row["source"]
        key = (
            int(source["template"]),
            int(source["display_order"]),
            int(source["value_swap"]),
            int(source["query_role"]),
        )
        strata[key].append(row)
    selected = []
    for key, values in sorted(strata.items()):
        pair_ids = sorted(
            {row["pair_id"] for row in values},
            key=lambda value: stable_order(value, f"causal-screen:{key}"),
        )
        chosen = pair_ids[0]
        chosen_rows = sorted(
            [row for row in values if row["pair_id"] == chosen],
            key=lambda row: row["direction"],
        )
        if len(chosen_rows) != 2:
            raise RuntimeError(f"causal screen pair drift: {key}/{chosen}")
        selected.extend(chosen_rows)
    if len(selected) != 64:
        raise RuntimeError(f"causal screen size drift: {len(selected)}")
    return selected


def causal_mediation_scan(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_depth: int,
    events: list[dict[str, Any]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Causally prescreen every event by restoring it under the source do."""
    result_rows = []
    batches = list(scpg.batches_by_template(rows, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        source_logits, source_residuals = scpg.capture_residuals(
            model, device, source_cases, (source_depth,), candidate_ids
        )
        target_logits, target_components = scpg.capture_components(
            model, layers, device, target_cases, events, candidate_ids
        )
        source_patch = scpg.source_patch_spec(
            source_depth,
            target_cases,
            source_residuals[source_depth],
            "joint",
        )
        do_logits = scpg.forward_candidate(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
        )
        source_margin = scpg.semantic_margin(source_logits, batch)
        target_margin = scpg.semantic_margin(target_logits, batch)
        do_margin = scpg.semantic_margin(do_logits, batch)
        do_predictions = scpg.prediction_colors(do_logits)
        for event in events:
            event_id = event["event_id"]
            restore_logits = scpg.forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
                receiver_patches=[{
                    "event": event,
                    "vectors": target_components[event_id],
                }],
            )
            restore_margin = scpg.semantic_margin(restore_logits, batch)
            restore_predictions = scpg.prediction_colors(restore_logits)
            for index, item in enumerate(batch):
                denominator = float(
                    source_margin[index] - target_margin[index]
                )
                source_effect = float(
                    do_margin[index] - target_margin[index]
                )
                result_rows.append({
                    "schema_version": (
                        "phase1001_cross_model_causal_screen_row.v1"
                    ),
                    "phase": PHASE,
                    "model": item["source"]["model"],
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    **{
                        key: event[key]
                        for key in (
                            "event_id", "block_index", "layer_number",
                            "component", "role",
                        )
                    },
                    "source_transfer": source_effect
                    / max(abs(denominator), 1e-8),
                    "mediation_fraction": float(
                        (do_margin[index] - restore_margin[index])
                        / max(abs(source_effect), 1e-8)
                    ),
                    "source_flipped": (
                        do_predictions[index] == item["source"]["gold"]
                    ),
                    "restored_to_target": (
                        restore_predictions[index] == item["target"]["gold"]
                    ),
                })
            del restore_logits
        del (
            source_logits, source_residuals, target_logits,
            target_components, do_logits,
        )
        print(
            f"[causal-screen] {batch_number}/{len(batches)}",
            flush=True,
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        groups[row["event_id"]].append(row)
    metrics = {}
    for event_id, values in groups.items():
        first = values[0]
        metrics[event_id] = {
            "event_id": event_id,
            "block_index": first["block_index"],
            "layer_number": first["layer_number"],
            "component": first["component"],
            "role": first["role"],
            "n": len(values),
            "mean_source_transfer": float(np.mean([
                row["source_transfer"] for row in values
            ])),
            "source_flip_rate": float(np.mean([
                row["source_flipped"] for row in values
            ])),
            "median_mediation_fraction": float(np.median([
                row["mediation_fraction"] for row in values
            ])),
            "mean_mediation_fraction": float(np.mean([
                row["mediation_fraction"] for row in values
            ])),
            "positive_mediation_rate": float(np.mean([
                row["mediation_fraction"] > 0 for row in values
            ])),
            "restored_to_target_rate": float(np.mean([
                row["restored_to_target"] for row in values
            ])),
        }
    return result_rows, metrics


def select_from_causal_screen(
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        metrics.values(),
        key=lambda item: (
            -item["median_mediation_fraction"],
            -item["mean_mediation_fraction"],
            -item["positive_mediation_rate"],
            -item["restored_to_target_rate"],
            item["layer_number"],
            item["event_id"],
        ),
    )
    selected = [dict(item) for item in ranked[:RECEIVER_LIMIT]]
    for rank, item in enumerate(selected, 1):
        item["causal_screen_rank"] = rank
        item["selection_partition"] = "discovery_causal_screen"
        item["selection_uses_confirmation"] = False
    return selected


def receiver_screen(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_depth: int,
    events: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    output = []
    batches = list(scpg.batches_by_template(rows, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        source_logits, source_residuals = scpg.capture_residuals(
            model, device, source_cases, (source_depth,), candidate_ids
        )
        target_logits, target_components = scpg.capture_components(
            model, layers, device, target_cases, events, candidate_ids
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
            events,
            candidate_ids,
            source_patch=source_patch,
        )
        source_margin = scpg.semantic_margin(source_logits, batch)
        target_margin = scpg.semantic_margin(target_logits, batch)
        do_margin = scpg.semantic_margin(do_logits, batch)
        do_predictions = scpg.prediction_colors(do_logits)
        for event in events:
            event_id = event["event_id"]
            suff_logits = scpg.forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                receiver_patches=[{
                    "event": event,
                    "vectors": do_components[event_id],
                }],
            )
            restore_logits = scpg.forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
                receiver_patches=[{
                    "event": event,
                    "vectors": target_components[event_id],
                }],
            )
            suff_margin = scpg.semantic_margin(suff_logits, batch)
            restore_margin = scpg.semantic_margin(restore_logits, batch)
            suff_predictions = scpg.prediction_colors(suff_logits)
            restore_predictions = scpg.prediction_colors(restore_logits)
            for index, item in enumerate(batch):
                denominator = float(
                    source_margin[index] - target_margin[index]
                )
                source_effect = float(
                    do_margin[index] - target_margin[index]
                )
                output.append({
                    "schema_version": "phase1001_cross_model_receiver_row.v1",
                    "phase": PHASE,
                    "model": item["source"]["model"],
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    **{
                        key: event[key]
                        for key in (
                            "event_id", "block_index", "layer_number",
                            "component", "role",
                        )
                    },
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "source_transfer": source_effect
                    / max(abs(denominator), 1e-8),
                    "sufficiency_transfer": float(
                        (suff_margin[index] - target_margin[index])
                        / max(abs(denominator), 1e-8)
                    ),
                    "mediation_fraction": float(
                        (do_margin[index] - restore_margin[index])
                        / max(abs(source_effect), 1e-8)
                    ),
                    "source_flipped": (
                        do_predictions[index] == item["source"]["gold"]
                    ),
                    "sufficiency_flipped": (
                        suff_predictions[index] == item["source"]["gold"]
                    ),
                    "restored_to_target": (
                        restore_predictions[index] == item["target"]["gold"]
                    ),
                })
            del suff_logits, restore_logits
        del (
            source_logits, source_residuals, target_logits, target_components,
            do_logits, do_components,
        )
        print(
            f"[receiver-{rows[0]['partition']}] "
            f"{batch_number}/{len(batches)}",
            flush=True,
        )
    return output


def summarize_receivers(
    rows: list[dict[str, Any]],
    response_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    summary = {}
    for event_id, values in groups.items():
        first = values[0]
        mean_sufficiency = float(np.mean([
            row["sufficiency_transfer"] for row in values
        ]))
        median_mediation = float(np.median([
            row["mediation_fraction"] for row in values
        ]))
        item = {
            "event_id": event_id,
            "block_index": first["block_index"],
            "layer_number": first["layer_number"],
            "component": first["component"],
            "role": first["role"],
            "n": len(values),
            "mean_source_transfer": float(np.mean([
                row["source_transfer"] for row in values
            ])),
            "source_flip_rate": float(np.mean([
                row["source_flipped"] for row in values
            ])),
            "mean_sufficiency_transfer": mean_sufficiency,
            "sufficiency_flip_rate": float(np.mean([
                row["sufficiency_flipped"] for row in values
            ])),
            "median_mediation_fraction": median_mediation,
            "mean_mediation_fraction": float(np.mean([
                row["mediation_fraction"] for row in values
            ])),
            "restored_to_target_rate": float(np.mean([
                row["restored_to_target"] for row in values
            ])),
            "causal_score": (
                max(0.0, mean_sufficiency)
                + max(0.0, median_mediation)
            ),
        }
        if response_metrics is not None:
            response = response_metrics[event_id]
            item.update({
                "response_score": response["response_score"],
                "median_source_to_natural_cosine": response[
                    "median_source_to_natural_cosine"
                ],
                "median_magnitude_fraction": response[
                    "median_magnitude_fraction"
                ],
            })
        summary[event_id] = item
    return summary


def rank_receivers(
    selected: list[dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        [metrics[event["event_id"]] for event in selected],
        key=lambda item: (
            -item["causal_score"],
            -item["median_mediation_fraction"],
            -item["mean_sufficiency_transfer"],
            item["layer_number"],
            item["event_id"],
        ),
    )
    for rank, item in enumerate(ranked, 1):
        item["causal_rank"] = rank
        item["selection_partition"] = "discovery"
        item["selection_uses_confirmation"] = False
    return ranked


def choose_joint_size(
    summary: dict[str, dict[str, Any]],
    natural_by_size: dict[str, dict[str, Any]],
) -> tuple[int, bool]:
    ordered = [summary[str(size)] for size in JOINT_SIZES if str(size) in summary]
    eligible = [
        item for item in ordered
        if item["median_mediation_fraction"] >= 0.30
        and item["mean_sufficiency_transfer"] >= 0.30
        and natural_by_size[str(item["joint_size"])]["conditions"][
            "source_do"
        ]["source_rate"] >= 0.70
        and natural_by_size[str(item["joint_size"])]["conditions"][
            "source_plus_joint_restore"
        ]["target_rate"] >= 0.50
    ]
    if eligible:
        return min(item["joint_size"] for item in eligible), True
    best = max(
        ordered,
        key=lambda item: (
            item["median_mediation_fraction"]
            + item["mean_sufficiency_transfer"],
            natural_by_size[str(item["joint_size"])]["conditions"][
                "source_plus_joint_restore"
            ]["target_rate"],
            -item["joint_size"],
        ),
    )
    return int(best["joint_size"]), False


def stratified_natural_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = row["source"]
        key = (
            int(source["template"]),
            int(source["display_order"]),
            int(source["value_swap"]),
            int(source["query_role"]),
        )
        strata[key].append(row)
    selected = []
    for key, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda row: stable_order(
                f"{row['pair_id']}:{row['direction']}",
                f"natural:{row['partition']}:{key}",
            ),
        )
        selected.extend(ordered[:NATURAL_PER_STRATUM])
    return selected


def natural_joint(
    model,
    layers,
    tokenizer,
    device,
    rows: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_depth: int,
    ranked: list[dict[str, Any]],
    joint_size: int,
    batch_size: int,
    budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_events = ranked[:joint_size]
    effective_eos = eos_ids(model, tokenizer)
    result_rows = []
    subset = stratified_natural_subset(rows)
    batches = list(scpg.batches_by_template(subset, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        _, source_residuals = scpg.capture_residuals(
            model, device, source_cases, (source_depth,), candidate_ids
        )
        _, target_components = scpg.capture_components(
            model,
            layers,
            device,
            target_cases,
            selected_events,
            candidate_ids,
        )
        source_patch = scpg.source_patch_spec(
            source_depth,
            target_cases,
            source_residuals[source_depth],
            "joint",
        )
        conditions = {
            "source_do": [],
            "source_plus_joint_restore": [
                {
                    "event": event,
                    "vectors": target_components[event["event_id"]],
                }
                for event in selected_events
            ],
        }
        for condition, patches in conditions.items():
            generated = scpg.generate_with_interventions(
                model,
                layers,
                tokenizer,
                device,
                target_cases,
                source_patch,
                patches,
                effective_eos,
                budget,
            )
            for index, item in enumerate(batch):
                output = generated[index]
                result_rows.append({
                    "schema_version": "phase1001_cross_model_natural_row.v1",
                    "phase": PHASE,
                    "model": item["source"]["model"],
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "condition": condition,
                    "joint_size": joint_size,
                    "event_ids": [
                        event["event_id"] for event in selected_events
                    ],
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "prediction": output["prediction"],
                    "flipped_to_source": (
                        output["prediction"] == item["source"]["gold"]
                    ),
                    "restored_to_target": (
                        output["prediction"] == item["target"]["gold"]
                    ),
                    "eos_seen": output["eos_position"] is not None,
                    "exact_short": output["exact_short"],
                    "generated_text": output["text"],
                })
        del source_residuals, target_components
        print(
            f"[natural-{rows[0]['partition']}] "
            f"{batch_number}/{len(batches)}",
            flush=True,
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[row["condition"]].append(row)
    summary = {
        condition: {
            "n": len(values),
            "source_rate": float(np.mean([
                row["flipped_to_source"] for row in values
            ])),
            "target_rate": float(np.mean([
                row["restored_to_target"] for row in values
            ])),
            "eos_rate": float(np.mean([
                row["eos_seen"] for row in values
            ])),
            "exact_short_rate": float(np.mean([
                row["exact_short"] for row in values
            ])),
        }
        for condition, values in sorted(grouped.items())
    }
    return result_rows, {
        "subset_size": len(subset),
        "conditions": summary,
    }


def source_condition(
    summary: dict[str, dict[str, float]], depth: int
) -> dict[str, float]:
    return summary[f"depth_{depth}_joint_entity"]


def run_model(
    model_name: str,
    batch_size: int,
    natural_budget: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1001 cross-model topology requires CUDA")
    scpg.MODEL = model_name
    scpg.PHASE = PHASE
    scpg.RECEIVER_LIMIT = RECEIVER_LIMIT
    scpg.JOINT_SIZES = JOINT_SIZES
    scpg.SOURCE_DEPTHS = SOURCE_DEPTHS
    output_root = OUT_ROOT / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    cases, all_pairs, candidate_ids, protocol_audit = build_protocol(model_name)
    case_by_id = {row["record_id"]: row for row in cases}
    discovery_pairs = selected_pairs(all_pairs, "discovery")
    confirmation_pairs = selected_pairs(all_pairs, "confirmation")
    discovery = directional(
        discovery_pairs, case_by_id, "discovery"
    )
    confirmation = directional(
        confirmation_pairs, case_by_id, "confirmation"
    )
    write_jsonl(output_root / "cases.jsonl", cases)
    write_jsonl(output_root / "pairs.jsonl", all_pairs)
    write_json(output_root / "protocol_audit.json", protocol_audit)
    write_jsonl(output_root / "discovery_selected_pairs.jsonl", discovery_pairs)
    write_jsonl(
        output_root / "confirmation_selected_pairs.jsonl",
        confirmation_pairs,
    )

    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        if not bool(getattr(model, "is_loaded_in_8bit", False)):
            raise RuntimeError("cross-model precision contract requires 8-bit")

        behavior_rows, behavior_summary = candidate_behavior(
            model, device, cases, candidate_ids, batch_size
        )
        write_jsonl(output_root / "behavior_rows.jsonl", behavior_rows)
        write_json(output_root / "behavior_summary.json", behavior_summary)
        if not behavior_summary["gate_pass"]:
            summary = {
                "schema_version": "phase1001_cross_model_summary.v1",
                "phase": PHASE,
                "model": model_name,
                "status": "behavior_gate_failed",
                "protocol_audit": protocol_audit,
                "behavior": behavior_summary,
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "quantized_8bit": True,
                "elapsed_seconds": time.time() - started,
            }
            write_json(output_root / "summary.json", summary)
            return summary

        discovery_source_rows, discovery_source_selection = (
            scpg.scan_source_depths(
                model,
                layers,
                device,
                discovery,
                candidate_ids,
                batch_size,
            )
        )
        source_depth = int(discovery_source_selection["selected_depth"])
        write_jsonl(
            output_root / "discovery_source_rows.jsonl",
            discovery_source_rows,
        )
        write_json(
            output_root / "discovery_source_selection.json",
            discovery_source_selection,
        )

        events = scpg.event_definitions(source_depth, info.n_layers)
        screening_rows = causal_screen_subset(discovery)
        causal_screen_rows, causal_screen_metrics = causal_mediation_scan(
            model,
            layers,
            device,
            screening_rows,
            candidate_ids,
            source_depth,
            events,
            batch_size,
        )
        selected = select_from_causal_screen(causal_screen_metrics)
        write_jsonl(
            output_root / "discovery_causal_screen_rows.jsonl",
            causal_screen_rows,
        )
        write_json(
            output_root / "discovery_causal_screen_metrics.json",
            causal_screen_metrics,
        )
        write_json(
            output_root / "discovery_selected_receivers.json",
            {
                "selection_method": (
                    "all-event single restoration under source intervention"
                ),
                "screening_direction_count": len(screening_rows),
                "selection_uses_response_similarity": False,
                "receivers": selected,
            },
        )

        discovery_receiver_rows = receiver_screen(
            model,
            layers,
            device,
            discovery,
            candidate_ids,
            source_depth,
            selected,
            batch_size,
        )
        discovery_receiver_metrics = summarize_receivers(
            discovery_receiver_rows
        )
        ranked = rank_receivers(selected, discovery_receiver_metrics)
        write_jsonl(
            output_root / "discovery_receiver_rows.jsonl",
            discovery_receiver_rows,
        )
        write_json(
            output_root / "discovery_receiver_metrics.json",
            discovery_receiver_metrics,
        )

        discovery_joint_rows = scpg.joint_screen(
            model,
            layers,
            device,
            discovery,
            candidate_ids,
            source_depth,
            ranked,
            batch_size,
        )
        discovery_joint_summary = scpg.summarize_joint(discovery_joint_rows)
        discovery_natural_rows = []
        discovery_natural_by_size = {}
        for size in JOINT_SIZES:
            if str(size) not in discovery_joint_summary:
                continue
            size_rows, size_summary = natural_joint(
                model,
                layers,
                tokenizer,
                device,
                discovery,
                candidate_ids,
                source_depth,
                ranked,
                size,
                batch_size,
                natural_budget,
            )
            discovery_natural_rows.extend(size_rows)
            discovery_natural_by_size[str(size)] = size_summary
        joint_size, joint_threshold_pass = choose_joint_size(
            discovery_joint_summary,
            discovery_natural_by_size,
        )
        frozen = {
            "schema_version": "phase1001_cross_model_frozen_topology.v1",
            "phase": PHASE,
            "model": model_name,
            "source_depth": source_depth,
            "source_relative_depth": source_depth / info.n_layers,
            "ranked_receivers": ranked,
            "joint_size": joint_size,
            "joint_event_ids": [
                item["event_id"] for item in ranked[:joint_size]
            ],
            "joint_threshold_pass_in_discovery": joint_threshold_pass,
            "joint_selection_uses_discovery_natural_output": True,
            "selection_partition": "discovery",
            "selection_uses_confirmation": False,
            "frozen_before_confirmation": True,
        }
        write_jsonl(
            output_root / "discovery_joint_rows.jsonl",
            discovery_joint_rows,
        )
        write_json(
            output_root / "discovery_joint_summary.json",
            discovery_joint_summary,
        )
        write_jsonl(
            output_root / "discovery_natural_rows.jsonl",
            discovery_natural_rows,
        )
        write_json(
            output_root / "discovery_natural_by_size.json",
            discovery_natural_by_size,
        )
        write_json(output_root / "frozen_topology.json", frozen)

        confirmation_source_rows, confirmation_source_selection = (
            scpg.scan_source_depths(
                model,
                layers,
                device,
                confirmation,
                candidate_ids,
                batch_size,
            )
        )
        write_jsonl(
            output_root / "confirmation_source_rows.jsonl",
            confirmation_source_rows,
        )
        write_json(
            output_root / "confirmation_source_selection.json",
            confirmation_source_selection,
        )

        frozen_events = [
            {
                key: item[key]
                for key in (
                    "event_id", "block_index", "layer_number",
                    "component", "role",
                )
            }
            for item in ranked
        ]
        confirmation_receiver_rows = receiver_screen(
            model,
            layers,
            device,
            confirmation,
            candidate_ids,
            source_depth,
            frozen_events,
            batch_size,
        )
        confirmation_receiver_metrics = summarize_receivers(
            confirmation_receiver_rows
        )
        write_jsonl(
            output_root / "confirmation_receiver_rows.jsonl",
            confirmation_receiver_rows,
        )
        write_json(
            output_root / "confirmation_receiver_metrics.json",
            confirmation_receiver_metrics,
        )

        confirmation_joint_rows = scpg.joint_screen(
            model,
            layers,
            device,
            confirmation,
            candidate_ids,
            source_depth,
            ranked,
            batch_size,
        )
        confirmation_joint_summary = scpg.summarize_joint(
            confirmation_joint_rows
        )
        write_jsonl(
            output_root / "confirmation_joint_rows.jsonl",
            confirmation_joint_rows,
        )
        write_json(
            output_root / "confirmation_joint_summary.json",
            confirmation_joint_summary,
        )

        natural_rows, natural_summary = natural_joint(
            model,
            layers,
            tokenizer,
            device,
            confirmation,
            candidate_ids,
            source_depth,
            ranked,
            joint_size,
            batch_size,
            natural_budget,
        )
        write_jsonl(
            output_root / "confirmation_natural_rows.jsonl", natural_rows
        )
        write_json(
            output_root / "confirmation_natural_summary.json",
            natural_summary,
        )

        discovery_source = source_condition(
            discovery_source_selection["condition_summary"], source_depth
        )
        confirmation_source = source_condition(
            confirmation_source_selection["condition_summary"], source_depth
        )
        discovery_joint = discovery_joint_summary[str(joint_size)]
        confirmation_joint = confirmation_joint_summary[str(joint_size)]
        confirmation_natural = natural_summary["conditions"]
        gate_checks = {
            "behavior": behavior_summary["gate_pass"],
            "discovery_source": (
                discovery_source["flip_rate"] >= 0.70
                and discovery_source["median_transfer"] >= 0.50
            ),
            "confirmation_source": (
                confirmation_source["flip_rate"] >= 0.70
                and confirmation_source["median_transfer"] >= 0.50
            ),
            "discovery_joint": (
                discovery_joint["median_mediation_fraction"] >= 0.30
                and discovery_joint["mean_sufficiency_transfer"] >= 0.30
            ),
            "confirmation_joint": (
                confirmation_joint["median_mediation_fraction"] >= 0.30
                and confirmation_joint["mean_sufficiency_transfer"] >= 0.30
            ),
            "natural_source": (
                confirmation_natural["source_do"]["source_rate"] >= 0.70
            ),
            "natural_restore": (
                confirmation_natural["source_plus_joint_restore"]["target_rate"]
                >= 0.50
            ),
        }
        top_roles = defaultdict(int)
        top_components = defaultdict(int)
        for item in ranked[:joint_size]:
            top_roles[item["role"]] += 1
            top_components[item["component"]] += 1
        summary = {
            "schema_version": "phase1001_cross_model_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "status": "complete",
            "protocol_audit": protocol_audit,
            "behavior": behavior_summary,
            "discovery_direction_count": len(discovery),
            "confirmation_direction_count": len(confirmation),
            "causal_screen_direction_count": len(screening_rows),
            "causal_screen_event_count": len(events),
            "causal_screen_top12": selected,
            "receiver_selection_uses_response_similarity": False,
            "source_depth": source_depth,
            "source_relative_depth": source_depth / info.n_layers,
            "discovery_source": discovery_source,
            "confirmation_source": confirmation_source,
            "ranked_receivers": ranked,
            "joint_size": joint_size,
            "joint_event_ids": frozen["joint_event_ids"],
            "joint_role_counts": dict(top_roles),
            "joint_component_counts": dict(top_components),
            "discovery_joint": discovery_joint,
            "confirmation_joint": confirmation_joint,
            "discovery_natural_by_size": discovery_natural_by_size,
            "confirmation_receiver_metrics": confirmation_receiver_metrics,
            "confirmation_natural": natural_summary,
            "gate_checks": gate_checks,
            "topology_gate_pass": all(gate_checks.values()),
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "quantized_8bit": True,
            "parameter_dtype": str(next(model.parameters()).dtype),
            "batch_size": batch_size,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
            "interpretation_boundary": (
                "Functional component-level replication only; no physical "
                "head, channel, or neuron identity is transferred across models."
            ),
            "method_revision": (
                "The initial response-selected Qwen3 audit missed a known "
                "receiver region and failed natural restoration. This run "
                "selects candidates using all-event causal restoration only."
            ),
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "source_depth": source_depth,
            "joint_size": joint_size,
            "joint_event_ids": frozen["joint_event_ids"],
            "gate_checks": gate_checks,
            "topology_gate_pass": summary["topology_gate_pass"],
        }, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate() -> dict[str, Any]:
    summaries = {}
    for model_name in MODELS:
        path = OUT_ROOT / model_name / "summary.json"
        if path.exists():
            summaries[model_name] = json.loads(path.read_text(encoding="utf-8"))
    complete = {
        key: value for key, value in summaries.items()
        if value.get("status") == "complete"
    }
    alignment_path = OUT_ROOT / "glm4_decision_boundary_audit" / "summary.json"
    glm4_alignment = (
        json.loads(alignment_path.read_text(encoding="utf-8"))
        if alignment_path.exists()
        else None
    )
    expansion_root = OUT_ROOT / "natural_confirmation_expansion"
    expansion_summaries = {}
    for model_name in MODELS:
        path = expansion_root / model_name / "summary.json"
        if path.exists():
            expansion_summaries[model_name] = json.loads(
                path.read_text(encoding="utf-8")
            )
    topology_rows = []
    for model_name, value in complete.items():
        small_sample_natural = value[
            "confirmation_natural"
        ]["conditions"]["source_plus_joint_restore"]["target_rate"]
        effective_natural = small_sample_natural
        effective_natural_n = value["confirmation_natural"]["subset_size"]
        decision_boundary_aligned = False
        expansion = expansion_summaries.get(model_name)
        if expansion is not None:
            official_size = str(expansion["official_joint_size"])
            effective_natural = expansion["size_summaries"][official_size][
                "conditions"
            ]["source_plus_joint_restore"]["target_rate"]
            effective_natural_n = expansion["confirmation_direction_count"]
            decision_boundary_aligned = (
                expansion["decision_boundary"]
                != "original_answer_boundary"
            )
        if (
            expansion is None
            and
            model_name == "glm4"
            and glm4_alignment is not None
            and glm4_alignment.get("decision_boundary_alignment_pass")
        ):
            effective_natural = glm4_alignment[
                "aligned_confirmation_natural_restore_target_rate"
            ]
            decision_boundary_aligned = True
        topology_rows.append({
            "model": model_name,
            "n_layers": value["n_layers"],
            "source_depth": value["source_depth"],
            "source_relative_depth": value["source_relative_depth"],
            "joint_size": value["joint_size"],
            "joint_role_counts": value["joint_role_counts"],
            "joint_component_counts": value["joint_component_counts"],
            "confirmation_source_flip_rate": value[
                "confirmation_source"
            ]["flip_rate"],
            "confirmation_joint_mediation": value[
                "confirmation_joint"
            ]["median_mediation_fraction"],
            "confirmation_joint_sufficiency": value[
                "confirmation_joint"
            ]["mean_sufficiency_transfer"],
            "small_sample_natural_restore_target_rate": small_sample_natural,
            "effective_natural_restore_target_rate": effective_natural,
            "effective_natural_confirmation_n": effective_natural_n,
            "decision_boundary_aligned": decision_boundary_aligned,
            "original_small_sample_topology_gate_pass": value[
                "topology_gate_pass"
            ],
            "effective_topology_gate_pass": (
                value["gate_checks"]["behavior"]
                and value["gate_checks"]["confirmation_source"]
                and value["gate_checks"]["confirmation_joint"]
                and effective_natural >= 0.50
            ),
        })
    all_complete = len(complete) == len(MODELS)
    all_source_pass = all(
        value["gate_checks"]["confirmation_source"]
        for value in complete.values()
    ) if complete else False
    all_joint_pass = all(
        value["gate_checks"]["confirmation_joint"]
        for value in complete.values()
    ) if complete else False
    all_effective_natural_pass = all(
        row["effective_natural_restore_target_rate"] >= 0.50
        for row in topology_rows
    ) if topology_rows else False
    answer_boundary_majority = all(
        value["joint_role_counts"].get("answer_boundary", 0)
        >= max(1, value["joint_size"] // 2)
        for value in complete.values()
    ) if complete else False
    payload = {
        "schema_version": "phase1001_cross_model_aggregate.v1",
        "phase": PHASE,
        "models_expected": list(MODELS),
        "models_complete": sorted(complete),
        "all_models_complete": all_complete,
        "expanded_natural_models": sorted(expansion_summaries),
        "topology_rows": topology_rows,
        "cross_model_checks": {
            "all_confirmation_sources_pass": all_source_pass,
            "all_confirmation_joint_sets_pass": all_joint_pass,
            "all_generation_aligned_natural_restores_pass": (
                all_effective_natural_pass
            ),
            "answer_boundary_is_at_least_half_of_joint_set_in_all_models": (
                answer_boundary_majority
            ),
        },
        "cross_model_candidate_topology_replication": (
            all_complete
            and all_source_pass
            and all_joint_pass
            and answer_boundary_majority
        ),
        "cross_model_generation_aligned_replication": (
            all_complete
            and all_source_pass
            and all_joint_pass
            and all_effective_natural_pass
            and answer_boundary_majority
        ),
        "cross_model_functional_replication": (
            all_complete
            and all_source_pass
            and all_joint_pass
            and all_effective_natural_pass
            and answer_boundary_majority
        ),
        "glm4_decision_boundary_audit": glm4_alignment,
        "claim_boundary": (
            "This aggregate compares intervention-defined functional topology. "
            "Expanded frozen confirmation rates are preferred when available. "
            "It does not claim shared neuron identities or a universal language law."
        ),
    }
    write_json(OUT_ROOT / "cross_model_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--natural-budget", type=int, default=6)
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.model is None:
        parser.error("--model is required unless --aggregate is used")
    run_model(args.model, args.batch_size, args.natural_budget)


if __name__ == "__main__":
    main()
