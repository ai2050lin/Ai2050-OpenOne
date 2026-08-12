#!/usr/bin/env python3
"""Run the Phase1030 two-template composition replication in FP16."""

from __future__ import annotations

import argparse
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

from model_utils import get_layers, get_model_info
import phase1030_composition_replication_protocol as protocol
from phase1023_fp16_utils import (
    MODELS,
    load_fp16,
    quantization_audit,
    release_fp16,
)
from phase1029_multibinding_competition_scan import (
    BATCH_SIZE,
    ROLE_INDEX,
    MultiDepthCleanCapture,
    MultiPatchCapture,
    chunks,
    finite_audit,
    gather_states,
    make_batch,
    normalize_rows,
    safe_cosine,
    target_positions,
)


EPS = 1e-8


def prototype_map(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[int, np.ndarray]:
    result = {}
    for held_surface in range(len(protocol.NONCE_PAIRS)):
        values = np.empty(
            (8, clean_readout.shape[-1]), dtype=np.float32
        )
        for concept_index in range(8):
            indices = [
                int(row["case_index"])
                for row in cases
                if int(row["expected_index"]) == concept_index
                and int(row["surface_index"]) != held_surface
            ]
            values[concept_index] = np.asarray(
                clean_readout[indices], dtype=np.float32
            ).mean(axis=0)
        result[held_surface] = normalize_rows(values)
    return result


def classify(
    values: np.ndarray,
    units: list[dict[str, Any]],
    prototypes: dict[int, np.ndarray],
) -> dict[str, Any]:
    base_hits = []
    alternate_hits = []
    scrambled_hits = []
    other_hits = []
    alt_margins = []
    for index, unit in enumerate(units):
        similarity = normalize_rows(
            values[index:index + 1]
        )[0] @ prototypes[int(unit["surface_index"])].T
        predicted = int(np.argmax(similarity))
        base = int(unit["target_index"])
        alternate = int(unit["donor_index"])
        scrambled = int(unit["scrambled_donor_index"])
        base_hits.append(int(predicted == base))
        alternate_hits.append(int(predicted == alternate))
        scrambled_hits.append(int(predicted == scrambled))
        other_hits.append(int(predicted not in {base, alternate}))
        alt_margins.append(
            float(similarity[alternate] - similarity[base])
        )
    return {
        "unit_count": len(units),
        "base_top1": float(np.mean(base_hits)),
        "alternate_top1": float(np.mean(alternate_hits)),
        "scrambled_top1": float(np.mean(scrambled_hits)),
        "other_top1": float(np.mean(other_hits)),
        "alternate_vs_base_margin": float(np.mean(alt_margins)),
        "chance": 0.125,
    }


def condition_metrics(
    values: np.ndarray,
    units: list[dict[str, Any]],
    prototypes: dict[int, np.ndarray],
) -> dict[str, Any]:
    result = {"all": classify(values, units, prototypes)}
    for template_index in range(len(protocol.TEMPLATES)):
        indices = [
            index
            for index, unit in enumerate(units)
            if int(unit["template_index"]) == template_index
        ]
        result[f"template_{template_index}"] = classify(
            np.asarray(values[indices]),
            [units[index] for index in indices],
            prototypes,
        )
    return result


def clean_world_metrics(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
    prototypes: dict[int, np.ndarray],
) -> dict[str, Any]:
    result = {}
    for scope, template_index in (
        ("all", None),
        ("template_0", 0),
        ("template_1", 1),
    ):
        scope_result = {}
        all_hits = []
        for world in protocol.WORLD_CODES:
            rows = [
                row
                for row in cases
                if row["world"] == world
                and (
                    template_index is None
                    or int(row["template_index"]) == template_index
                )
            ]
            hits = []
            margins = []
            for row in rows:
                index = int(row["case_index"])
                similarity = normalize_rows(
                    clean_readout[index:index + 1]
                )[0] @ prototypes[int(row["surface_index"])].T
                expected = int(row["expected_index"])
                wrong = np.delete(similarity, expected)
                hit = int(np.argmax(similarity) == expected)
                hits.append(hit)
                all_hits.append(hit)
                margins.append(
                    float(similarity[expected] - np.max(wrong))
                )
            scope_result[world] = {
                "case_count": len(rows),
                "expected_top1": float(np.mean(hits)),
                "expected_vs_wrong_margin": float(np.mean(margins)),
                "chance": 0.125,
            }
        scope_result["all_worlds"] = {
            "case_count": len(all_hits),
            "expected_top1": float(np.mean(all_hits)),
            "chance": 0.125,
        }
        result[scope] = scope_result
    return result


def difference_metrics(
    values: np.ndarray,
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for scope, template_index in (
        ("all", None),
        ("template_0", 0),
        ("template_1", 1),
    ):
        rows = [
            row
            for row in units
            if (
                template_index is None
                or int(row["template_index"]) == template_index
            )
        ]
        world = {
            code: np.asarray([
                values[int(row["world_case_indices"][code])]
                for row in rows
            ], dtype=np.float32)
            for code in protocol.WORLD_CODES
        }
        d_b = world["10"] - world["00"]
        d_q = world["01"] - world["00"]
        d_bq = world["11"] - world["00"]
        interaction = (
            world["11"]
            - world["10"]
            - world["01"]
            + world["00"]
        )
        scale = 0.5 * (
            np.linalg.norm(d_b, axis=-1)
            + np.linalg.norm(d_q, axis=-1)
        )
        result[scope] = {
            "unit_count": len(rows),
            "binding_delta_norm_mean": float(
                np.mean(np.linalg.norm(d_b, axis=-1))
            ),
            "query_delta_norm_mean": float(
                np.mean(np.linalg.norm(d_q, axis=-1))
            ),
            "same_answer_delta_norm_mean": float(
                np.mean(np.linalg.norm(d_bq, axis=-1))
            ),
            "binding_query_cosine_mean": float(
                np.mean(safe_cosine(d_b, d_q))
            ),
            "same_answer_ratio_mean": float(np.mean(
                np.linalg.norm(d_bq, axis=-1)
                / np.maximum(scale, EPS)
            )),
            "interaction_ratio_mean": float(np.mean(
                np.linalg.norm(interaction, axis=-1)
                / np.maximum(scale, EPS)
            )),
        }
    return result


def case_indices(
    rows: list[dict[str, Any]],
    world: str,
    *,
    units_all: list[dict[str, Any]],
    scrambled: bool = False,
) -> list[int]:
    if not scrambled:
        return [
            int(row["world_case_indices"][world]) for row in rows
        ]
    return [
        int(
            units_all[int(row["scrambled_unit_index"])]
            ["world_case_indices"][world]
        )
        for row in rows
    ]


def role_names(
    rows: list[dict[str, Any]],
    field: str,
) -> list[str]:
    return [str(row[field]) for row in rows]


def patch_plan(
    condition: str,
    rows: list[dict[str, Any]],
    *,
    units_all: list[dict[str, Any]],
    positions: torch.Tensor,
    clean_states: np.ndarray,
    depths: list[int],
    source_depth: int,
    query_depth: int,
    preoutput_depth: int,
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    plans: list[tuple[int, list[str], list[int], list[str]]] = []
    selected = role_names(rows, "selected_concept_role")
    unselected = role_names(rows, "unselected_concept_role")
    selected_def = role_names(rows, "selected_definition_role")
    source_a = ["concept_a_end"] * len(rows)
    source_b = ["concept_b_end"] * len(rows)
    definition_a = ["definition_nonce_a_end"] * len(rows)
    definition_b = ["definition_nonce_b_end"] * len(rows)
    query = ["query_nonce_end"] * len(rows)
    pre_output = ["pre_output"] * len(rows)
    b_indices = case_indices(rows, "10", units_all=units_all)
    q_indices = case_indices(rows, "01", units_all=units_all)
    bq_indices = case_indices(rows, "11", units_all=units_all)

    if condition == "selected_source_b":
        plans.append((source_depth, selected, b_indices, selected))
    elif condition == "unselected_source_b":
        plans.append((source_depth, unselected, b_indices, unselected))
    elif condition == "source_pair_b":
        plans.extend([
            (source_depth, source_a, b_indices, source_a),
            (source_depth, source_b, b_indices, source_b),
        ])
    elif condition == "query_q":
        plans.append((query_depth, query, q_indices, query))
    elif condition == "query_bq":
        plans.append((query_depth, query, bq_indices, query))
    elif condition == "source_pair_plus_query_q":
        plans.extend([
            (source_depth, source_a, b_indices, source_a),
            (source_depth, source_b, b_indices, source_b),
            (query_depth, query, q_indices, query),
        ])
    elif condition == "source_pair_plus_query_bq":
        plans.extend([
            (source_depth, source_a, b_indices, source_a),
            (source_depth, source_b, b_indices, source_b),
            (query_depth, query, bq_indices, query),
        ])
    elif condition == "full_bq":
        plans.extend([
            (source_depth, source_a, bq_indices, source_a),
            (source_depth, source_b, bq_indices, source_b),
            (query_depth, query, bq_indices, query),
        ])
    elif condition == "source_pair_scrambled":
        scrambled_indices = case_indices(
            rows, "10", units_all=units_all, scrambled=True
        )
        plans.extend([
            (source_depth, source_a, scrambled_indices, source_a),
            (source_depth, source_b, scrambled_indices, source_b),
        ])
    elif condition == "source_pair_wrong_position":
        plans.extend([
            (source_depth, definition_a, b_indices, source_a),
            (source_depth, definition_b, b_indices, source_b),
        ])
    elif condition == "query_q_wrong_position":
        plans.append((query_depth, selected_def, q_indices, query))
    elif condition == "query_bq_wrong_position":
        plans.append((query_depth, selected_def, bq_indices, query))
    elif condition == "pre_output_b":
        plans.append((
            preoutput_depth, pre_output, b_indices, pre_output
        ))
    elif condition == "pre_output_bq":
        plans.append((
            preoutput_depth, pre_output, bq_indices, pre_output
        ))
    else:
        raise ValueError(condition)

    result: dict[
        int,
        list[tuple[torch.Tensor, torch.Tensor]],
    ] = defaultdict(list)
    for depth, target_roles, donor_indices, donor_roles in plans:
        depth_index = depths.index(depth)
        result[depth].append((
            target_positions(positions, target_roles),
            gather_states(
                clean_states,
                donor_indices,
                donor_roles,
                depth_index,
            ),
        ))
    return dict(result)


def condition_depths(
    condition: str,
    source_depth: int,
    query_depth: int,
    preoutput_depth: int,
) -> list[int]:
    if condition in {
        "selected_source_b",
        "unselected_source_b",
        "source_pair_b",
        "source_pair_scrambled",
        "source_pair_wrong_position",
    }:
        return [source_depth]
    if condition in {
        "query_q",
        "query_bq",
        "query_q_wrong_position",
        "query_bq_wrong_position",
    }:
        return [query_depth]
    if condition in {
        "source_pair_plus_query_q",
        "source_pair_plus_query_bq",
        "full_bq",
    }:
        return [source_depth, query_depth]
    return [preoutput_depth]


def run_condition(
    *,
    base_model,
    layers,
    condition: str,
    units: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    clean_states: np.ndarray,
    depths: list[int],
    source_depth: int,
    query_depth: int,
    preoutput_depth: int,
    readout_depth: int,
    tokenizer,
    device: torch.device,
    model_name: str,
    output: np.ndarray,
) -> None:
    used_depths = sorted(set(condition_depths(
        condition, source_depth, query_depth, preoutput_depth
    )))
    capture = MultiPatchCapture(layers, used_depths, readout_depth)
    capture.register()
    try:
        offset = 0
        for batch_index, row_batch in enumerate(
            chunks(units, BATCH_SIZE[model_name]), 1
        ):
            target_cases = [
                cases[int(row["world_case_indices"]["00"])]
                for row in row_batch
            ]
            input_ids, attention_mask, positions = make_batch(
                target_cases,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            specs = patch_plan(
                condition,
                row_batch,
                units_all=units,
                positions=positions,
                clean_states=clean_states,
                depths=depths,
                source_depth=source_depth,
                query_depth=query_depth,
                preoutput_depth=preoutput_depth,
            )
            capture.begin(
                specs=specs,
                pre_positions=positions[:, ROLE_INDEX["pre_output"]],
            )
            with torch.inference_mode():
                base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            value = capture.value().numpy().astype(
                np.float16, copy=False
            )
            output[offset:offset + len(row_batch)] = value
            offset += len(row_batch)
            if batch_index % 32 == 0:
                print(
                    f"[phase1030] {model_name} {condition} "
                    f"units={offset}/{len(units)}",
                    flush=True,
                )
    finally:
        capture.close()
    output.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()

    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    units = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "units.jsonl"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{args.model}.jsonl"
    )
    selected = prereg[
        "selected_depths_frozen_from_phase1029"
    ][args.model]
    source_depth = int(selected["source"])
    query_depth = int(selected["query"])
    preoutput_depth = int(selected["pre_output"])
    readout_depth = int(selected["readout"])
    depths = sorted({
        source_depth, query_depth, preoutput_depth
    })
    started = time.time()
    model = tokenizer = None
    atlas_dir = protocol.OUT_ROOT / "atlas" / args.model
    atlas_dir.mkdir(parents=True, exist_ok=True)
    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        precision_audit = quantization_audit(model)
        if (
            precision_audit["has_quantized_modules"]
            or precision_audit["has_bf16_parameters"]
            or not precision_audit["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, args.model)
        if max(depths) >= readout_depth:
            raise RuntimeError("patch depth must precede readout depth")
        base_model = model.model

        clean_states = np.lib.format.open_memmap(
            atlas_dir / "clean_role_states.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases),
                len(protocol.ROLES),
                len(depths),
                info.d_model,
            ),
        )
        clean_readout = np.lib.format.open_memmap(
            atlas_dir / "clean_readout.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(cases), info.d_model),
        )
        capture = MultiDepthCleanCapture(
            layers, depths, readout_depth
        )
        capture.register()
        try:
            offset = 0
            for batch_index, batch in enumerate(
                chunks(cases, BATCH_SIZE[args.model]), 1
            ):
                input_ids, attention_mask, positions = make_batch(
                    batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                capture.begin(positions)
                with torch.inference_mode():
                    base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                states, readout = capture.stacked()
                clean_states[offset:offset + len(batch)] = (
                    states.numpy().astype(np.float16, copy=False)
                )
                clean_readout[offset:offset + len(batch)] = (
                    readout.numpy().astype(np.float16, copy=False)
                )
                offset += len(batch)
                if batch_index % 32 == 0:
                    print(
                        f"[phase1030] {args.model} clean "
                        f"cases={offset}/{len(cases)}",
                        flush=True,
                    )
        finally:
            capture.close()
        clean_states.flush()
        clean_readout.flush()

        prototypes = prototype_map(clean_readout, cases)
        clean_metrics = clean_world_metrics(
            clean_readout, cases, prototypes
        )
        observational = []
        for role_index, role in enumerate(protocol.ROLES):
            for depth_index, depth in enumerate(depths):
                observational.append({
                    "role": role,
                    "depth": depth,
                    "four_world_differences": difference_metrics(
                        np.asarray(
                            clean_states[:, role_index, depth_index, :]
                        ),
                        units,
                    ),
                })

        outputs = np.lib.format.open_memmap(
            atlas_dir / "confirmation_conditions.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(protocol.CONDITIONS),
                len(units),
                info.d_model,
            ),
        )
        condition_rows = []
        for condition_index, condition in enumerate(protocol.CONDITIONS):
            run_condition(
                base_model=base_model,
                layers=layers,
                condition=condition,
                units=units,
                cases=cases,
                clean_states=clean_states,
                depths=depths,
                source_depth=source_depth,
                query_depth=query_depth,
                preoutput_depth=preoutput_depth,
                readout_depth=readout_depth,
                tokenizer=tokenizer,
                device=device,
                model_name=args.model,
                output=outputs[condition_index],
            )
            condition_rows.append({
                "condition": condition,
                "metrics": condition_metrics(
                    np.asarray(outputs[condition_index]),
                    units,
                    prototypes,
                ),
            })
        outputs.flush()

        arrays = {
            "clean_role_states": np.asarray(clean_states),
            "clean_readout": np.asarray(clean_readout),
            "confirmation_conditions": np.asarray(outputs),
        }
        finiteness = finite_audit(arrays)
        metrics = {
            "schema_version": "phase1030_metrics.v1",
            "model": args.model,
            "selected_depths": selected,
            "captured_depths": depths,
            "clean_four_world_readout": clean_metrics,
            "observational_role_depth": observational,
            "confirmation_conditions": condition_rows,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        summary = {
            "schema_version": "phase1030_model_summary.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "model": args.model,
            "precision": "fp16",
            "quantization": "none",
            "placement": placement,
            "runtime_precision_audit": precision_audit,
            "selected_depths": selected,
            "captured_depths": depths,
            "selection_source": "phase1029_frozen",
            "finiteness": finiteness,
            "elapsed_seconds": time.time() - started,
            "claim_limit": prereg["claim_limit"],
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps({
            "model": args.model,
            "clean": clean_metrics,
            "conditions": condition_rows,
            "finiteness": finiteness,
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        model = tokenizer = None


if __name__ == "__main__":
    main()
