#!/usr/bin/env python3
"""Phase 1001 frozen-topology structural extrapolation on Qwen3.

The six-head cut is not re-selected. It is applied unchanged to:
1. three-entity color binding,
2. two-entity shape binding,
3. two-entity color binding with new paraphrase templates.

Each family contains 256 factorial entity-swap pairs and 512 bidirectional
interventions. The test asks whether the discovered topology generalizes
beyond the original two-entity color templates.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
import re
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

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1000_factorial_binding_protocol import COLORS, MODEL, NAMES
from phase1000_scpg_discovery import (
    batches_by_template,
    capture_residuals,
    prediction_colors,
    semantic_margin,
    source_patch_spec,
    write_rows,
)
from phase1001_attention_head_discovery import (
    SOURCE_DEPTH,
    capture_attention_states,
    forward_with_patches,
    register_head_patches,
    read_json,
    write_json,
)
from phase1001_minimum_head_cut import CUT_ROOT


OUT_ROOT = RESULT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1001_attention_physical_decomposition"
    / "structural_extrapolation"
)
FAMILY_SPECS = {
    "three_entity_color": {
        "entity_count": 3,
        "attribute_kind": "color",
        "attribute_values": {
            "red": "red",
            "blue": "blue",
            "green": "green",
            "yellow": "yellow",
        },
        "template_set": "three_entity",
    },
    "two_entity_shape": {
        "entity_count": 2,
        "attribute_kind": "shape",
        "attribute_values": {
            "red": "round",
            "blue": "square",
            "green": "oval",
            "yellow": "triangle",
        },
        "template_set": "shape",
    },
    "two_entity_color_paraphrase": {
        "entity_count": 2,
        "attribute_kind": "color",
        "attribute_values": {
            "red": "red",
            "blue": "blue",
            "green": "green",
            "yellow": "yellow",
        },
        "template_set": "paraphrase",
    },
}
WORLD_COUNT = 32
TEMPLATE_COUNT = 4
QUERY_ROLES = (0, 1)
EXTRAPOLATION_THRESHOLDS = {
    "behavior_accuracy": 0.95,
    "source_candidate_flip": 0.80,
    "median_mediation": 0.30,
    "mean_sufficiency": 0.30,
    "natural_source_flip": 0.80,
    "natural_restore_target": 0.50,
}


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[MODEL]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def one_token(tokenizer, text, leading):
    ids = tokenizer.encode(
        (" " if leading else "") + text,
        add_special_tokens=False,
    )
    if len(ids) != 1:
        raise RuntimeError(
            f"expected one token for {text!r}, leading={leading}: {ids}"
        )
    return int(ids[0])


def render_prompt(
    template_set,
    template,
    names,
    attributes,
    query_name,
    kind,
):
    pairs = list(zip(names, attributes))
    if template_set == "three_entity":
        if template == 0:
            records = " ".join(
                f"{name} carries the {value} marker."
                for name, value in pairs
            )
            return (
                f"Records: {records}\n"
                f"Question: What color marker does {query_name} carry?\n"
                "Answer with exactly one color word."
            )
        if template == 1:
            records = "; ".join(
                f"{name} is assigned {value}" for name, value in pairs
            )
            return (
                f"Assignments: {records}.\n"
                f"Which color is assigned to {query_name}? "
                "Reply with one color word."
            )
        if template == 2:
            records = ", ".join(
                f"{name} has a {value} badge" for name, value in pairs
            )
            return (
                f"Registry: {records}.\n"
                f"Using the registry, name {query_name}'s badge color. "
                "Use one word."
            )
        records = " | ".join(
            f"{name}: {value}" for name, value in pairs
        )
        return (
            f"Color table | {records}.\n"
            f"Return the color linked to {query_name}. "
            "Output only that color."
        )

    if template_set == "shape":
        first, second = pairs
        if template == 0:
            return (
                f"Objects: {first[0]} owns the {first[1]} token. "
                f"{second[0]} owns the {second[1]} token.\n"
                f"What shape token does {query_name} own? "
                "Answer with exactly one shape word."
            )
        if template == 1:
            return (
                f"Shape assignments: {first[0]} maps to {first[1]}; "
                f"{second[0]} maps to {second[1]}.\n"
                f"Which shape maps to {query_name}? "
                "Reply with one word."
            )
        if template == 2:
            return (
                f"In the shape registry, {first[0]} has {first[1]} and "
                f"{second[0]} has {second[1]}.\n"
                f"Look up the shape for {query_name}. Use one word."
            )
        return (
            f"Shape table | {first[0]}: {first[1]} | "
            f"{second[0]}: {second[1]}.\n"
            f"Return {query_name}'s shape only."
        )

    first, second = pairs
    if template == 0:
        return (
            f"Field notes say {first[0]} uses {first[1]}, while "
            f"{second[0]} uses {second[1]}.\n"
            f"Give the color associated with {query_name}, one word only."
        )
    if template == 1:
        return (
            f"Lookup entries -> {first[0]} = {first[1]}; "
            f"{second[0]} = {second[1]}.\n"
            f"Read out {query_name}'s color. No extra text."
        )
    if template == 2:
        return (
            f"During this trial, {first[1]} belongs to {first[0]} and "
            f"{second[1]} belongs to {second[0]}.\n"
            f"Which color belongs to {query_name}? Answer in one word."
        )
    return (
        f"Pairing list says {first[0]} has {first[1]}, and "
        f"{second[0]} has {second[1]}.\n"
        f"Select the color paired with {query_name}; output only the color."
    )


def render_chat(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_family(tokenizer, family, spec):
    rng = random.Random(f"phase1001:{family}")
    usable_names = [
        name
        for name in NAMES
        if len(tokenizer.encode(" " + name, add_special_tokens=False))
        == 1
    ]
    combinations = list(
        itertools.combinations(usable_names, spec["entity_count"])
    )
    rng.shuffle(combinations)
    if len(combinations) < WORLD_COUNT:
        raise RuntimeError("not enough single-token name combinations")
    values = spec["attribute_values"]
    candidate_ids = {
        label: one_token(tokenizer, value, leading=False)
        for label, value in values.items()
    }
    prompt_value_ids = {
        label: one_token(tokenizer, value, leading=True)
        for label, value in values.items()
    }
    prompt_name_ids = {
        name: one_token(tokenizer, name, leading=True)
        for name in usable_names
    }
    label_tuples = list(
        itertools.permutations(COLORS, spec["entity_count"])
    )
    directional = []
    cases = []
    for world in range(WORLD_COUNT):
        base_names = list(combinations[world])
        base_labels = list(label_tuples[world % len(label_tuples)])
        rotation = world % spec["entity_count"]
        target_names = (
            base_names[rotation:] + base_names[:rotation]
        )
        target_labels = (
            base_labels[rotation:] + base_labels[:rotation]
        )
        for template in range(TEMPLATE_COUNT):
            for query_role in QUERY_ROLES:
                query_name = base_names[query_role]
                query_position = target_names.index(query_name)
                swap_position = (
                    query_position + 1
                ) % spec["entity_count"]
                source_names = list(target_names)
                source_names[query_position], source_names[swap_position] = (
                    source_names[swap_position],
                    source_names[query_position],
                )
                target_gold = target_labels[query_position]
                source_gold = target_labels[swap_position]
                attributes = [values[label] for label in target_labels]

                arms = []
                for arm_name, record_names, gold in (
                    ("target", target_names, target_gold),
                    ("source", source_names, source_gold),
                ):
                    prompt = render_prompt(
                        spec["template_set"],
                        template,
                        record_names,
                        attributes,
                        query_name,
                        spec["attribute_kind"],
                    )
                    rendered = render_chat(tokenizer, prompt)
                    input_ids = tokenizer.encode(
                        rendered, add_special_tokens=False
                    )
                    extended = tokenizer.encode(
                        rendered + values[gold],
                        add_special_tokens=False,
                    )
                    if extended != input_ids + [candidate_ids[gold]]:
                        raise RuntimeError(
                            f"candidate boundary drift {family}/{template}"
                        )
                    query_positions = [
                        index
                        for index, token_id in enumerate(input_ids)
                        if token_id == prompt_name_ids[query_name]
                    ]
                    if not query_positions:
                        raise RuntimeError(
                            "query name occurrence drift: "
                            f"{family}/t{template}/{query_name}/"
                            f"{query_positions}"
                        )
                    arms.append(
                        {
                            "schema_version": (
                                "phase1001_extrapolation_case.v1"
                            ),
                            "family": family,
                            "world_id": f"{family}.w{world:03d}",
                            "template": template,
                            "query_role": query_role,
                            "arm": arm_name,
                            "gold": gold,
                            "foil": (
                                source_gold
                                if gold == target_gold
                                else target_gold
                            ),
                            "answer_value": values[gold],
                            "prompt": prompt,
                            "input_ids": input_ids,
                            "input_token_count": len(input_ids),
                            "role_positions": {
                                "query_name": query_positions[-1],
                                "answer_boundary": len(input_ids) - 1,
                            },
                            "candidate_token_ids": candidate_ids,
                        }
                    )
                target, source = arms
                if target["arm"] != "target":
                    raise RuntimeError("arm order drift")
                changed = [
                    index
                    for index, (left, right) in enumerate(
                        zip(target["input_ids"], source["input_ids"])
                    )
                    if left != right
                ]
                if len(changed) != 2:
                    raise RuntimeError(
                        f"entity change count drift: {changed}"
                    )
                for arm in arms:
                    arm["role_positions"]["slot0_entity"] = changed[0]
                    arm["role_positions"]["slot1_entity"] = changed[1]
                    cases.append(arm)
                pair_id = (
                    f"{family}.w{world:03d}.t{template}.q{query_role}"
                )
                directional.extend(
                    [
                        {
                            "pair_id": pair_id,
                            "partition": "extrapolation",
                            "direction": "source_to_target",
                            "source": source,
                            "target": target,
                        },
                        {
                            "pair_id": pair_id,
                            "partition": "extrapolation",
                            "direction": "target_to_source",
                            "source": target,
                            "target": source,
                        },
                    ]
                )
    expected_pairs = WORLD_COUNT * TEMPLATE_COUNT * len(QUERY_ROLES)
    if len(directional) != expected_pairs * 2:
        raise RuntimeError("directional count drift")
    return {
        "family": family,
        "spec": spec,
        "candidate_token_ids": candidate_ids,
        "prompt_value_token_ids": prompt_value_ids,
        "pair_count": expected_pairs,
        "direction_count": len(directional),
        "cases": cases,
        "directional": directional,
    }


def generate_first_token(
    model,
    layers,
    tokenizer,
    device,
    rows,
    source_patch,
    head_patches,
    candidate_ids,
    effective_eos,
):
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    full_width = input_ids.shape[1]
    source_handle = None
    head_handles = []
    try:
        from phase1000_scpg_discovery import register_source_patch

        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=full_width
        )
        head_handles, head_counts = register_head_patches(
            layers,
            rows,
            head_patches,
            device,
            full_width=full_width,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=4,
                eos_token_id=effective_eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        if source_patch is not None and source_count[0] != 1:
            raise RuntimeError("source generation count drift")
        if any(counter[0] != 1 for counter in head_counts.values()):
            raise RuntimeError("head generation count drift")
        suffix = generated.sequences[:, full_width:].detach().cpu()
        value_to_label = {
            tokenizer.decode([int(token_id)])
            .strip()
            .lower(): label
            for label, token_id in candidate_ids.items()
        }
        results = []
        for tokens in suffix:
            text = tokenizer.decode(
                tokens.tolist(), skip_special_tokens=False
            )
            match = re.search(r"[A-Za-z]+", text)
            first_word = match.group(0).lower() if match else None
            results.append(
                {
                    "prediction": value_to_label.get(first_word),
                    "first_word": first_word,
                    "first_token_id": (
                        int(tokens[0]) if len(tokens) else None
                    ),
                    "text": text,
                }
            )
        return results
    finally:
        for handle in reversed(head_handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def summarize_rows(rows):
    return {
        "n": len(rows),
        "source_behavior_accuracy": float(
            np.mean([row["source_behavior_correct"] for row in rows])
        ),
        "target_behavior_accuracy": float(
            np.mean([row["target_behavior_correct"] for row in rows])
        ),
        "source_candidate_flip_rate": float(
            np.mean([row["do_flipped_to_source"] for row in rows])
        ),
        "median_mediation_fraction": float(
            np.median([row["mediation_fraction"] for row in rows])
        ),
        "mean_mediation_fraction": float(
            np.mean([row["mediation_fraction"] for row in rows])
        ),
        "mean_sufficiency_transfer": float(
            np.mean([row["sufficiency_transfer"] for row in rows])
        ),
        "restored_candidate_target_rate": float(
            np.mean([row["restored_to_target"] for row in rows])
        ),
        "sufficiency_candidate_source_rate": float(
            np.mean([row["sufficiency_to_source"] for row in rows])
        ),
        "template_median_mediation": {
            str(template): float(
                np.median(
                    [
                        row["mediation_fraction"]
                        for row in rows
                        if int(row["template"]) == template
                    ]
                )
            )
            for template in range(TEMPLATE_COUNT)
        },
    }


def summarize_natural(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "source_rate": float(
                np.mean([row["prediction"] == row["source_gold"] for row in values])
            ),
            "target_rate": float(
                np.mean([row["prediction"] == row["target_gold"] for row in values])
            ),
            "recognized_candidate_rate": float(
                np.mean([row["prediction"] is not None for row in values])
            ),
        }
        for condition, values in groups.items()
    }


def run(batch_size):
    if not torch.cuda.is_available():
        raise RuntimeError("structural extrapolation requires CUDA")
    tokenizer_protocol = load_tokenizer()
    families = {
        family: build_family(tokenizer_protocol, family, spec)
        for family, spec in FAMILY_SPECS.items()
    }
    for family, data in families.items():
        family_root = OUT_ROOT / family
        family_root.mkdir(parents=True, exist_ok=True)
        write_rows(family_root / "cases.jsonl", data["cases"])
        write_rows(
            family_root / "directional_pairs.jsonl",
            data["directional"],
        )
        write_json(
            family_root / "protocol.json",
            {
                key: value
                for key, value in data.items()
                if key not in ("cases", "directional")
            },
        )
    del tokenizer_protocol

    frozen = read_json(CUT_ROOT / "discovery/frozen_spec.json")
    events = [
        {
            "event_id": event_id,
            "layer_number": int(event_id.split(".")[0][1:]),
            "head_index": int(event_id.split(".")[1][1:]),
            "role": "answer_boundary",
        }
        for event_id in frozen["frozen_event_ids"]
    ]
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
        family_summaries = {}
        for family, data in families.items():
            candidate_ids = data["candidate_token_ids"]
            rows = []
            natural_rows = []
            batches = list(
                batches_by_template(data["directional"], batch_size)
            )
            for batch_number, batch in enumerate(batches, 1):
                source_cases = [item["source"] for item in batch]
                target_cases = [item["target"] for item in batch]
                source_logits, source_residuals = capture_residuals(
                    model,
                    device,
                    source_cases,
                    (SOURCE_DEPTH,),
                    candidate_ids,
                )
                target_logits, target_heads, _ = capture_attention_states(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                )
                source_patch = source_patch_spec(
                    SOURCE_DEPTH,
                    target_cases,
                    source_residuals[SOURCE_DEPTH],
                    "joint",
                )
                do_logits, do_heads, _ = capture_attention_states(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                )
                restore_patches = [
                    {
                        "event": event,
                        "vectors": target_heads[
                            int(event["layer_number"])
                        ][:, int(event["head_index"]), :],
                    }
                    for event in events
                ]
                sufficiency_patches = [
                    {
                        "event": event,
                        "vectors": do_heads[
                            int(event["layer_number"])
                        ][:, int(event["head_index"]), :],
                    }
                    for event in events
                ]
                restored_logits = forward_with_patches(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                    head_patches=restore_patches,
                )
                sufficiency_logits = forward_with_patches(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    head_patches=sufficiency_patches,
                )
                source_margin = semantic_margin(source_logits, batch)
                target_margin = semantic_margin(target_logits, batch)
                do_margin = semantic_margin(do_logits, batch)
                restored_margin = semantic_margin(restored_logits, batch)
                sufficiency_margin = semantic_margin(
                    sufficiency_logits, batch
                )
                source_prediction = prediction_colors(source_logits)
                target_prediction = prediction_colors(target_logits)
                do_prediction = prediction_colors(do_logits)
                restored_prediction = prediction_colors(restored_logits)
                sufficiency_prediction = prediction_colors(
                    sufficiency_logits
                )
                for index, item in enumerate(batch):
                    do_effect = float(
                        do_margin[index] - target_margin[index]
                    )
                    natural_effect = float(
                        source_margin[index] - target_margin[index]
                    )
                    rows.append(
                        {
                            "schema_version": (
                                "phase1001_structural_extrapolation.v1"
                            ),
                            "phase": 1001,
                            "model": MODEL,
                            "family": family,
                            "pair_id": item["pair_id"],
                            "direction": item["direction"],
                            "template": item["target"]["template"],
                            "source_gold": item["source"]["gold"],
                            "target_gold": item["target"]["gold"],
                            "source_behavior_correct": (
                                source_prediction[index]
                                == item["source"]["gold"]
                            ),
                            "target_behavior_correct": (
                                target_prediction[index]
                                == item["target"]["gold"]
                            ),
                            "do_flipped_to_source": (
                                do_prediction[index]
                                == item["source"]["gold"]
                            ),
                            "mediation_fraction": float(
                                (
                                    do_margin[index]
                                    - restored_margin[index]
                                )
                                / max(abs(do_effect), 1e-8)
                            ),
                            "sufficiency_transfer": float(
                                (
                                    sufficiency_margin[index]
                                    - target_margin[index]
                                )
                                / max(abs(natural_effect), 1e-8)
                            ),
                            "restored_to_target": (
                                restored_prediction[index]
                                == item["target"]["gold"]
                            ),
                            "sufficiency_to_source": (
                                sufficiency_prediction[index]
                                == item["source"]["gold"]
                            ),
                        }
                    )
                for condition, patches in (
                    ("source_do", []),
                    ("source_plus_frozen_head_restore", restore_patches),
                ):
                    generated = generate_first_token(
                        model,
                        layers,
                        tokenizer,
                        device,
                        target_cases,
                        source_patch,
                        patches,
                        candidate_ids,
                        effective_eos,
                    )
                    for index, item in enumerate(batch):
                        natural_rows.append(
                            {
                                "schema_version": (
                                    "phase1001_structural_natural.v1"
                                ),
                                "phase": 1001,
                                "model": MODEL,
                                "family": family,
                                "pair_id": item["pair_id"],
                                "direction": item["direction"],
                                "condition": condition,
                                "source_gold": item["source"]["gold"],
                                "target_gold": item["target"]["gold"],
                                **generated[index],
                            }
                        )
                del (
                    source_logits,
                    source_residuals,
                    target_logits,
                    target_heads,
                    do_logits,
                    do_heads,
                    restored_logits,
                    sufficiency_logits,
                )
                if batch_number % 2 == 0 or batch_number == len(batches):
                    print(
                        f"[extrapolation-{family}] "
                        f"{batch_number}/{len(batches)}",
                        flush=True,
                    )
            candidate_summary = summarize_rows(rows)
            natural_summary = summarize_natural(natural_rows)
            gate_checks = {
                "source_behavior": candidate_summary[
                    "source_behavior_accuracy"
                ]
                >= EXTRAPOLATION_THRESHOLDS["behavior_accuracy"],
                "target_behavior": candidate_summary[
                    "target_behavior_accuracy"
                ]
                >= EXTRAPOLATION_THRESHOLDS["behavior_accuracy"],
                "source_candidate_flip": candidate_summary[
                    "source_candidate_flip_rate"
                ]
                >= EXTRAPOLATION_THRESHOLDS[
                    "source_candidate_flip"
                ],
                "head_mediation": candidate_summary[
                    "median_mediation_fraction"
                ]
                >= EXTRAPOLATION_THRESHOLDS["median_mediation"],
                "head_sufficiency": candidate_summary[
                    "mean_sufficiency_transfer"
                ]
                >= EXTRAPOLATION_THRESHOLDS["mean_sufficiency"],
                "natural_source_flip": natural_summary["source_do"][
                    "source_rate"
                ]
                >= EXTRAPOLATION_THRESHOLDS[
                    "natural_source_flip"
                ],
                "natural_restore": natural_summary[
                    "source_plus_frozen_head_restore"
                ]["target_rate"]
                >= EXTRAPOLATION_THRESHOLDS[
                    "natural_restore_target"
                ],
            }
            family_summary = {
                "family": family,
                "pair_count": data["pair_count"],
                "direction_count": data["direction_count"],
                "candidate_summary": candidate_summary,
                "natural_summary": natural_summary,
                "gate_checks": gate_checks,
                "family_gate_pass": all(gate_checks.values()),
            }
            family_root = OUT_ROOT / family
            write_rows(family_root / "causal_rows.jsonl", rows)
            write_rows(
                family_root / "natural_rows.jsonl", natural_rows
            )
            write_json(family_root / "summary.json", family_summary)
            family_summaries[family] = family_summary

        summary = {
            "schema_version": (
                "phase1001_structural_extrapolation_summary.v1"
            ),
            "phase": 1001,
            "model": MODEL,
            "frozen_head_event_ids": frozen["frozen_event_ids"],
            "family_count": len(family_summaries),
            "total_pair_count": sum(
                item["pair_count"] for item in family_summaries.values()
            ),
            "total_direction_count": sum(
                item["direction_count"]
                for item in family_summaries.values()
            ),
            "family_summaries": family_summaries,
            "thresholds": EXTRAPOLATION_THRESHOLDS,
            "natural_parser": (
                "first alphabetic word, case-insensitive; corrected after "
                "capitalized generations invalidated token-id-only parser"
            ),
            "all_families_pass": all(
                item["family_gate_pass"]
                for item in family_summaries.values()
            ),
            "passed_family_count": sum(
                item["family_gate_pass"]
                for item in family_summaries.values()
            ),
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        write_json(OUT_ROOT / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    summary = run(args.batch_size)
    print(
        json.dumps(
            {
                "all_families_pass": summary["all_families_pass"],
                "passed_family_count": summary["passed_family_count"],
                "families": {
                    family: {
                        "passed": item["family_gate_pass"],
                        "candidate": item["candidate_summary"],
                        "natural": item["natural_summary"],
                        "checks": item["gate_checks"],
                    }
                    for family, item in summary[
                        "family_summaries"
                    ].items()
                },
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
