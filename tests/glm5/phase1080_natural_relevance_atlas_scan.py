#!/usr/bin/env python3
"""Run one Phase1080 model in FP16 without quantization."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1058_multitoken_translation_scan as generation
from phase1065_multimode_response_atlas_scan import (
    RoleCapture,
    event_definitions,
    pairwise_direction_consistency,
    strict_generated_answer,
)
from phase1079_output_orthogonal_pattern_scan import (
    add_direction_field,
    add_scalar_field,
    delta_stats,
    interaction_relative,
    pad_rows,
    vector_cosine,
)
import phase1079_output_orthogonal_pattern_scan as scan_math
import phase1080_natural_relevance_atlas_protocol as protocol


# pad_rows is reused mechanically, but its role list must come from the
# frozen Phase1080 protocol rather than the source Phase1079 module.
scan_math.protocol = protocol


UNIT_BATCH_SIZE = {"qwen3": 1, "glm4": 1, "deepseek7b": 1}
EPSILON = 1e-12
VECTOR_FIELDS = (
    "relevance",
    "presence",
    "total",
    "infer_answer",
    "decoy_answer",
    "direct_answer",
)
SCALAR_FIELDS = (
    "surface",
    "shell",
    "relevance_answer_interaction",
    "relevance_cross_answer",
    "relevance_cross_surface",
    "relevance_cross_shell",
)


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"\s+", " ", value)


def classify_generation(generated: str, label: str) -> dict[str, Any]:
    text = normalized_text(generated)
    target = normalized_text(label)
    semantic_first = text.startswith(target)
    if semantic_first and target and target[-1].isalnum():
        tail = text[len(target):]
        semantic_first = not tail or not tail[0].isalnum()
    return {
        "normalized_text": text,
        "acceptable_normalized_labels": [target],
        "matched_label": target if semantic_first else None,
        "semantic_first": semantic_first,
        "strict_label_only": text == target,
    }


def generation_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    count = protocol.GENERATION_UNITS_PER_FAMILY_SPLIT_BRANCH
    for family in protocol.FAMILIES:
        for split in protocol.SPLITS:
            for branch in protocol.BRANCHES:
                eligible = [
                    row
                    for row in rows
                    if row["family"] == family
                    and row["split"] == split
                    and row["branch"] == branch
                    and row["state"] in {
                        f"t0_b{branch}_a0_l0",
                        f"t1_b{branch}_a1_l1",
                    }
                ]
                selected.extend({
                    **row,
                    "semantic_case_index": int(row["case_index"]),
                } for row in generation.evenly_spaced(eligible, count))
    return selected


def mean_value(
    sums: np.ndarray,
    counts: np.ndarray,
    index: tuple[int, ...],
) -> float | None:
    count = int(counts[index])
    return float(sums[index] / count) if count else None


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1080 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unit_meta: dict[str, dict[str, str]] = {}
    for row in rows:
        grouped[str(row["unit_id"])][str(row["state"])] = row
        unit_meta[str(row["unit_id"])] = {
            "family": str(row["family"]),
            "split": str(row["split"]),
        }
    units = []
    for unit_id in sorted(grouped):
        if set(grouped[unit_id]) != set(protocol.STATES):
            raise RuntimeError(f"incomplete unit: {unit_id}")
        units.append({
            "unit_id": unit_id,
            **unit_meta[unit_id],
            "states": grouped[unit_id],
        })

    started = time.time()
    model = tokenizer = capture = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")

        layers = list(get_layers(model))
        events = event_definitions(len(layers))
        event_keys = [
            (str(row["component"]), int(row["depth"])) for row in events
        ]
        d_model = int(model.get_input_embeddings().weight.shape[1])
        conditioning_index = {
            value: index for index, value in enumerate(protocol.CONDITIONINGS)
        }
        family_index = {
            value: index for index, value in enumerate(protocol.FAMILIES)
        }
        split_index = {
            value: index for index, value in enumerate(protocol.SPLITS)
        }
        role_index = {
            value: index for index, value in enumerate(protocol.CAPTURE_ROLES)
        }
        shape = (
            len(protocol.CONDITIONINGS),
            len(protocol.FAMILIES),
            len(protocol.SPLITS),
            len(events),
            len(protocol.CAPTURE_ROLES),
        )

        def vector_arrays() -> dict[str, np.ndarray]:
            return {
                "direction_sum": np.zeros((*shape, d_model), np.float32),
                "direction_count": np.zeros(shape, np.int32),
                "relative_sum": np.zeros(shape, np.float64),
                "relative_count": np.zeros(shape, np.int32),
            }

        vector_data = {name: vector_arrays() for name in VECTOR_FIELDS}
        scalar_data = {
            name: {
                "sum": np.zeros(shape, np.float64),
                "count": np.zeros(shape, np.int32),
            }
            for name in SCALAR_FIELDS
        }

        behavior_records: list[dict[str, Any]] = []
        candidate_totals: Counter = Counter()
        candidate_hits: Counter = Counter()
        candidate_finite: Counter = Counter()
        supported_units: Counter = Counter()
        nonfinite_candidate_count = 0
        nonfinite_hidden_count = 0
        pre_branch_max_abs = {
            "relevance": 0.0,
            "presence": 0.0,
            "total": 0.0,
        }
        identity_maximum = 0.0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        capture = RoleCapture(model, layers)
        capture.register()
        state_order = list(protocol.STATES)
        batch_size = UNIT_BATCH_SIZE[model_name]
        with torch.inference_mode():
            for batch_start in range(0, len(units), batch_size):
                batch_units = units[batch_start:batch_start + batch_size]
                forward_rows: list[dict[str, Any]] = []
                offsets: list[dict[str, int | None]] = []
                for unit in batch_units:
                    offset = len(forward_rows)
                    state_rows = [unit["states"][state] for state in state_order]
                    forward_rows.extend(state_rows)
                    identity_offset = None
                    if batch_start == 0:
                        forward_rows.append(state_rows[0])
                        identity_offset = len(forward_rows) - 1
                    offsets.append({"states": offset, "identity": identity_offset})

                input_ids, attention_mask, lengths, positions = pad_rows(
                    forward_rows, int(pad_id), device
                )
                capture.begin(positions)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                capture.validate()
                logits = output.logits
                final_positions = (lengths - 1).to(logits.device)
                batch_axis = torch.arange(logits.shape[0], device=logits.device)
                final_logits = logits[batch_axis, final_positions, :].float()
                del output, logits

                unit_behavior_support: dict[str, bool] = {}
                for unit, offset in zip(batch_units, offsets):
                    hits: list[int] = []
                    for local, state in enumerate(state_order):
                        row = unit["states"][state]
                        row_index = int(offset["states"]) + local
                        values = final_logits[row_index]
                        scores = {}
                        for answer_class in ("a0", "a1"):
                            token_ids = torch.tensor(
                                row["candidate_first_token_ids"][answer_class],
                                dtype=torch.long,
                                device=values.device,
                            )
                            scores[answer_class] = float(
                                values[token_ids].max().item()
                            )
                        expected = str(row["expected_class"])
                        other = "a1" if expected == "a0" else "a0"
                        margin = scores[expected] - scores[other]
                        finite = all(math.isfinite(v) for v in scores.values()) \
                            and math.isfinite(margin)
                        hit = finite and margin > 0.0
                        greedy = int(torch.argmax(values).item())
                        key = (unit["family"], unit["split"], row["branch"])
                        candidate_totals[key] += 1
                        candidate_finite[key] += int(finite)
                        candidate_hits[key] += int(hit)
                        nonfinite_candidate_count += int(not finite)
                        hits.append(int(hit))
                        behavior_records.append({
                            "schema_version": "phase1080_candidate_behavior.v1",
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "case_index": int(row["case_index"]),
                            "unit_id": unit["unit_id"],
                            "family": unit["family"],
                            "split": unit["split"],
                            "state": row["state"],
                            "branch": row["branch"],
                            "expected_class": expected,
                            "target_answer": row["target_answer"],
                            "candidate_scores": {
                                k: v if math.isfinite(v) else None
                                for k, v in scores.items()
                            },
                            "candidate_margin": margin if math.isfinite(margin) else None,
                            "finite_candidate": finite,
                            "candidate_hit": hit,
                            "greedy_first_token_id": greedy,
                            "greedy_first_token_text": tokenizer.decode([greedy]),
                        })
                    fraction = sum(hits) / len(hits) if hits else 0.0
                    supported = fraction >= float(
                        prereg["evidence_thresholds"][
                            "unit_behavior_support_fraction"
                        ]
                    )
                    unit_behavior_support[unit["unit_id"]] = supported
                    supported_units[(unit["family"], unit["split"])] += int(supported)

                for event_index, event_key in enumerate(event_keys):
                    values = capture.values[event_key].float()
                    for unit, offset in zip(batch_units, offsets):
                        states = {
                            state: values[int(offset["states"]) + local]
                            for local, state in enumerate(state_order)
                        }
                        if offset["identity"] is not None:
                            identity_delta = (
                                values[int(offset["identity"])] - states[state_order[0]]
                            )
                            if torch.isfinite(identity_delta).all():
                                identity_maximum = max(
                                    identity_maximum,
                                    float(identity_delta.abs().max().item()),
                                )

                        base_index = (
                            family_index[unit["family"]],
                            split_index[unit["split"]],
                            event_index,
                        )
                        conditionings = [conditioning_index["all_finite"]]
                        if unit_behavior_support[unit["unit_id"]]:
                            conditionings.append(
                                conditioning_index["behavior_supported"]
                            )

                        branch_deltas: dict[
                            str, dict[tuple[int, int, int], torch.Tensor]
                        ] = {
                            "relevance": {}, "presence": {}, "total": {}
                        }
                        for template in (0, 1):
                            for answer in (0, 1):
                                for surface in (0, 1):
                                    infer = states[
                                        f"t{template}_binfer_a{answer}_l{surface}"
                                    ]
                                    decoy = states[
                                        f"t{template}_bdecoy_a{answer}_l{surface}"
                                    ]
                                    direct = states[
                                        f"t{template}_bdirect_a{answer}_l{surface}"
                                    ]
                                    comparisons = {
                                        "relevance": (direct - decoy, decoy, direct),
                                        "presence": (decoy - infer, infer, decoy),
                                        "total": (direct - infer, infer, direct),
                                    }
                                    for name, (delta, left, right) in comparisons.items():
                                        branch_deltas[name][
                                            (template, answer, surface)
                                        ] = delta
                                        data = vector_data[name]
                                        nonfinite_hidden_count += add_direction_field(
                                            delta=delta,
                                            left=left,
                                            right=right,
                                            conditionings=conditionings,
                                            base_index=base_index,
                                            direction_sum=data["direction_sum"],
                                            direction_count=data["direction_count"],
                                            relative_sum=data["relative_sum"],
                                            relative_count=data["relative_count"],
                                        )
                                        for role_name in protocol.PRE_BRANCH_ROLES:
                                            role = role_index[role_name]
                                            role_delta = delta[role]
                                            if torch.isfinite(role_delta).all():
                                                pre_branch_max_abs[name] = max(
                                                    pre_branch_max_abs[name],
                                                    float(role_delta.abs().max().item()),
                                                )

                        for branch in protocol.BRANCHES:
                            field = f"{branch}_answer"
                            for template in (0, 1):
                                for surface in (0, 1):
                                    left = states[
                                        f"t{template}_b{branch}_a0_l{surface}"
                                    ]
                                    right = states[
                                        f"t{template}_b{branch}_a1_l{surface}"
                                    ]
                                    data = vector_data[field]
                                    nonfinite_hidden_count += add_direction_field(
                                        delta=right - left,
                                        left=left,
                                        right=right,
                                        conditionings=conditionings,
                                        base_index=base_index,
                                        direction_sum=data["direction_sum"],
                                        direction_count=data["direction_count"],
                                        relative_sum=data["relative_sum"],
                                        relative_count=data["relative_count"],
                                    )

                        for template in (0, 1):
                            for branch in protocol.BRANCHES:
                                for answer in (0, 1):
                                    left = states[f"t{template}_b{branch}_a{answer}_l0"]
                                    right = states[f"t{template}_b{branch}_a{answer}_l1"]
                                    relative, valid, _, _ = delta_stats(
                                        right - left, left, right
                                    )
                                    add_scalar_field(
                                        values=relative,
                                        valid=valid,
                                        conditionings=conditionings,
                                        base_index=base_index,
                                        value_sum=scalar_data["surface"]["sum"],
                                        value_count=scalar_data["surface"]["count"],
                                    )
                        for branch in protocol.BRANCHES:
                            for answer in (0, 1):
                                for surface in (0, 1):
                                    left = states[f"t0_b{branch}_a{answer}_l{surface}"]
                                    right = states[f"t1_b{branch}_a{answer}_l{surface}"]
                                    relative, valid, _, _ = delta_stats(
                                        right - left, left, right
                                    )
                                    add_scalar_field(
                                        values=relative,
                                        valid=valid,
                                        conditionings=conditionings,
                                        base_index=base_index,
                                        value_sum=scalar_data["shell"]["sum"],
                                        value_count=scalar_data["shell"]["count"],
                                    )

                        relevance = branch_deltas["relevance"]
                        for template in (0, 1):
                            for surface in (0, 1):
                                left_delta = relevance[(template, 0, surface)]
                                right_delta = relevance[(template, 1, surface)]
                                interaction, valid = interaction_relative(
                                    left_delta, right_delta
                                )
                                add_scalar_field(
                                    values=interaction,
                                    valid=valid,
                                    conditionings=conditionings,
                                    base_index=base_index,
                                    value_sum=scalar_data[
                                        "relevance_answer_interaction"
                                    ]["sum"],
                                    value_count=scalar_data[
                                        "relevance_answer_interaction"
                                    ]["count"],
                                )
                                cos, valid = vector_cosine(left_delta, right_delta)
                                add_scalar_field(
                                    values=cos,
                                    valid=valid,
                                    conditionings=conditionings,
                                    base_index=base_index,
                                    value_sum=scalar_data[
                                        "relevance_cross_answer"
                                    ]["sum"],
                                    value_count=scalar_data[
                                        "relevance_cross_answer"
                                    ]["count"],
                                )
                        for template in (0, 1):
                            for answer in (0, 1):
                                cos, valid = vector_cosine(
                                    relevance[(template, answer, 0)],
                                    relevance[(template, answer, 1)],
                                )
                                add_scalar_field(
                                    values=cos,
                                    valid=valid,
                                    conditionings=conditionings,
                                    base_index=base_index,
                                    value_sum=scalar_data[
                                        "relevance_cross_surface"
                                    ]["sum"],
                                    value_count=scalar_data[
                                        "relevance_cross_surface"
                                    ]["count"],
                                )
                        for answer in (0, 1):
                            for surface in (0, 1):
                                cos, valid = vector_cosine(
                                    relevance[(0, answer, surface)],
                                    relevance[(1, answer, surface)],
                                )
                                add_scalar_field(
                                    values=cos,
                                    valid=valid,
                                    conditionings=conditionings,
                                    base_index=base_index,
                                    value_sum=scalar_data[
                                        "relevance_cross_shell"
                                    ]["sum"],
                                    value_count=scalar_data[
                                        "relevance_cross_shell"
                                    ]["count"],
                                )
                    del values

                del final_logits, input_ids, attention_mask, lengths, positions
                capture.values = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                completed = min(batch_start + len(batch_units), len(units))
                if completed % 12 == 0 or completed == len(units):
                    print(json.dumps({
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "units_complete": completed,
                        "units_total": len(units),
                    }), flush=True)

        capture.close()
        capture = None

        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        generation_rows = generation_selection(rows)
        generated = generation.generate_case_outputs(
            model,
            device,
            generation_rows,
            eos_ids=eos_ids,
            batch_size=bridge.PAIR_BATCH_SIZE[model_name],
            steps=int(prereg["generation_steps"]),
        )
        generation_records: list[dict[str, Any]] = []
        generation_totals: Counter = Counter()
        generation_hits: Counter = Counter()
        generation_strict: Counter = Counter()
        for row in generation_rows:
            case_index = int(row["case_index"])
            output_ids = generated[case_index]
            text = strict_generated_answer(tokenizer, output_ids, eos_ids)
            classification = classify_generation(text, str(row["target_answer"]))
            key = (str(row["family"]), str(row["split"]), str(row["branch"]))
            generation_totals[key] += 1
            generation_hits[key] += int(classification["semantic_first"])
            generation_strict[key] += int(classification["strict_label_only"])
            generation_records.append({
                "schema_version": "phase1080_natural_generation.v1",
                "phase": protocol.PHASE,
                "model": model_name,
                "case_index": case_index,
                "unit_id": row["unit_id"],
                "family": row["family"],
                "split": row["split"],
                "state": row["state"],
                "branch": row["branch"],
                "target_answer": row["target_answer"],
                "generated_token_ids": [int(value) for value in output_ids],
                "generated_text": text,
                "terminated": generation.terminated(output_ids, eos_ids),
                **classification,
            })

        metric_rows: list[dict[str, Any]] = []
        for conditioning_name, conditioning in conditioning_index.items():
            for family_name, family in family_index.items():
                for split_name, split in split_index.items():
                    for event_index, event in enumerate(events):
                        for role_name, role in role_index.items():
                            index = (conditioning, family, split, event_index, role)
                            record: dict[str, Any] = {
                                "schema_version": "phase1080_response_metric.v1",
                                "phase": protocol.PHASE,
                                "model": model_name,
                                "conditioning": conditioning_name,
                                "family": family_name,
                                "split": split_name,
                                "role": role_name,
                                **event,
                            }
                            for field, data in vector_data.items():
                                direction_n = int(data["direction_count"][index])
                                record[f"{field}_magnitude_count"] = int(
                                    data["relative_count"][index]
                                )
                                record[f"mean_{field}_relative_magnitude"] = mean_value(
                                    data["relative_sum"], data["relative_count"], index
                                )
                                record[f"{field}_direction_count"] = direction_n
                                record[f"{field}_direction_consistency"] = (
                                    pairwise_direction_consistency(
                                        data["direction_sum"][index], direction_n
                                    )
                                )
                            for field, data in scalar_data.items():
                                record[f"{field}_count"] = int(data["count"][index])
                                record[f"mean_{field}"] = mean_value(
                                    data["sum"], data["count"], index
                                )
                            metric_rows.append(record)

        split_direction_rows: list[dict[str, Any]] = []
        discovery = split_index["discovery"]
        confirmation = split_index["confirmation"]
        for conditioning_name, conditioning in conditioning_index.items():
            for family_name, family in family_index.items():
                for event_index, event in enumerate(events):
                    for role_name, role in role_index.items():
                        record = {
                            "schema_version": "phase1080_split_direction_repeat.v1",
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "conditioning": conditioning_name,
                            "family": family_name,
                            "role": role_name,
                            **event,
                        }
                        for field, data in vector_data.items():
                            left_index = (
                                conditioning, family, discovery, event_index, role
                            )
                            right_index = (
                                conditioning, family, confirmation, event_index, role
                            )
                            left = data["direction_sum"][left_index].astype(
                                np.float64, copy=False
                            )
                            right = data["direction_sum"][right_index].astype(
                                np.float64, copy=False
                            )
                            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                            record[f"{field}_discovery_confirmation_cosine"] = (
                                float(np.dot(left, right) / denominator)
                                if denominator > EPSILON else None
                            )
                            record[f"{field}_discovery_count"] = int(
                                data["direction_count"][left_index]
                            )
                            record[f"{field}_confirmation_count"] = int(
                                data["direction_count"][right_index]
                            )
                        split_direction_rows.append(record)

        behavior_summary: dict[str, Any] = {}
        for family in protocol.FAMILIES:
            behavior_summary[family] = {}
            for split in protocol.SPLITS:
                split_row: dict[str, Any] = {}
                for branch in protocol.BRANCHES:
                    key = (family, split, branch)
                    total = int(candidate_totals[key])
                    generation_key = (family, split, branch)
                    generated_total = int(generation_totals[generation_key])
                    split_row[branch] = {
                        "candidate_count": total,
                        "candidate_finite_count": int(candidate_finite[key]),
                        "candidate_hit_count": int(candidate_hits[key]),
                        "candidate_accuracy": (
                            candidate_hits[key] / total if total else None
                        ),
                        "generation_case_count": generated_total,
                        "generation_semantic_first_count": int(
                            generation_hits[generation_key]
                        ),
                        "generation_semantic_first_accuracy": (
                            generation_hits[generation_key] / generated_total
                            if generated_total else None
                        ),
                        "generation_strict_count": int(
                            generation_strict[generation_key]
                        ),
                    }
                split_row["behavior_supported_unit_count"] = int(
                    supported_units[(family, split)]
                )
                behavior_summary[family][split] = split_row

        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_jsonl(atlas_root / "candidate_behavior.jsonl", behavior_records)
        protocol.write_jsonl(atlas_root / "natural_generation.jsonl", generation_records)
        protocol.write_jsonl(atlas_root / "response_metrics.jsonl", metric_rows)
        protocol.write_jsonl(
            atlas_root / "split_direction_repeat.jsonl", split_direction_rows
        )

        elapsed = time.time() - started
        summary = {
            "schema_version": "phase1080_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "case_count": len(rows),
            "unit_count": len(units),
            "event_count": len(events),
            "layer_count": len(layers),
            "d_model": d_model,
            "precision": precision,
            "placement": placement,
            "behavior_summary": behavior_summary,
            "pre_branch_max_abs": pre_branch_max_abs,
            "pre_branch_global_max_abs": max(pre_branch_max_abs.values()),
            "identity_maximum": identity_maximum,
            "nonfinite_candidate_count": nonfinite_candidate_count,
            "nonfinite_hidden_magnitude_role_count": nonfinite_hidden_count,
            "elapsed_seconds": elapsed,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "status": "complete",
            "case_count": len(rows),
            "unit_count": len(units),
            "elapsed_seconds": elapsed,
            "pre_branch_global_max_abs": summary["pre_branch_global_max_abs"],
            "summary_digest": summary["summary_digest"],
        }), flush=True)
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
