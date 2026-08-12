#!/usr/bin/env python3
"""Phase 998 matched hidden-state trace and candidate-thread discovery."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
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

from model_utils import get_model_info, load_model, release_model
from phase998_minimal_causal_thread_protocol import (
    MODEL,
    OUT_ROOT,
    PHASE,
    canonical,
    write_json,
    write_jsonl,
)


ROLES = ("source_color", "query_name", "answer_boundary")
PARTITIONS = ("discovery", "validation", "holdout")
PER_CONTRAST = {"discovery": 4, "validation": 2, "holdout": 4}
OBSERVATION_THRESHOLDS = {
    "validation_positive_rate": 0.80,
    "holdout_positive_rate": 0.80,
    "validation_min_subgroup_positive_rate": 0.70,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pair_hash(pair_id: str) -> str:
    return hashlib.sha256(("phase998:" + pair_id).encode("utf-8")).hexdigest()


def trace_partition(row: dict[str, Any]) -> str | None:
    if row["split"] == "discovery" and row["template"] < 3:
        return "discovery"
    if row["split"] == "validation" and row["template"] < 3:
        return "validation"
    if row["split"] == "holdout" and row["template"] == 3:
        return "holdout"
    return None


def select_pairs(
    cases: list[dict[str, Any]], behavior: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    behavior_by_record = {row["record_id"]: row for row in behavior}
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        pairs[row["pair_id"]].append(row)

    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    for pair_id, rows in pairs.items():
        rows = sorted(rows, key=lambda row: row["arm"])
        if len(rows) != 2:
            rejected.append({"pair_id": pair_id, "reason": "pair_count"})
            continue
        behavior_rows = [behavior_by_record[row["record_id"]] for row in rows]
        if not all(
            item["candidate_correct"]
            and item["natural_correct_both"]
            and item["repeat_stable"]
            for item in behavior_rows
        ):
            rejected.append({"pair_id": pair_id, "reason": "behavior"})
            continue
        partition = trace_partition(rows[0])
        if partition is None:
            continue
        contrast = f"{rows[0]['gold']}->{rows[1]['gold']}"
        key = (
            partition,
            rows[0]["template"],
            rows[0]["order"],
            rows[0]["query_role"],
            contrast,
        )
        strata[key].append(
            {
                "pair_id": pair_id,
                "partition": partition,
                "template": rows[0]["template"],
                "order": rows[0]["order"],
                "query_role": rows[0]["query_role"],
                "contrast": contrast,
                "arm0_record_id": rows[0]["record_id"],
                "arm1_record_id": rows[1]["record_id"],
            }
        )

    selected = []
    short_strata = []
    for key, rows in sorted(strata.items(), key=lambda item: str(item[0])):
        partition = key[0]
        limit = PER_CONTRAST[partition]
        ordered = sorted(rows, key=lambda row: pair_hash(row["pair_id"]))
        if len(ordered) < limit:
            short_strata.append({"stratum": list(key), "available": len(ordered), "need": limit})
            continue
        selected.extend(ordered[:limit])
    if short_strata:
        raise RuntimeError(f"trace strata underfilled: {short_strata[:8]}")

    counts = defaultdict(int)
    contrast_counts = defaultdict(int)
    for row in selected:
        counts[row["partition"]] += 1
        contrast_counts[f"{row['partition']}|{row['contrast']}"] += 1
    summary = {
        "schema_version": "phase998_trace_selection.v1",
        "phase": PHASE,
        "selected_pair_count": len(selected),
        "selected_prompt_count": 2 * len(selected),
        "partition_counts": dict(counts),
        "partition_contrast_counts": dict(contrast_counts),
        "per_contrast_per_stratum": PER_CONTRAST,
        "behavior_rejected_pair_count": len(rejected),
        "holdout_used_for_selection": False,
    }
    return selected, summary


def gather_roles(
    hidden_states,
    rows: list[dict[str, Any]],
    roles: tuple[str, ...],
) -> dict[tuple[int, str], np.ndarray]:
    batch = len(rows)
    device = hidden_states[0].device
    batch_index = torch.arange(batch, device=device)
    result: dict[tuple[int, str], np.ndarray] = {}
    for depth, hidden in enumerate(hidden_states):
        for role in roles:
            positions = torch.tensor(
                [row["role_positions"][role] for row in rows],
                dtype=torch.long,
                device=device,
            )
            values = hidden[batch_index, positions, :].detach().to("cpu", torch.float16)
            result[(depth, role)] = values.numpy()
    return result


def forward_roles(model, device, rows: list[dict[str, Any]]):
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows], dtype=torch.long, device=device
    )
    attention = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    values = gather_roles(output.hidden_states, rows, ROLES)
    del output, input_ids, attention
    return values


def event_key(depth: int, role: str) -> str:
    return f"d{depth:02d}.{role}"


def alignment_metrics(
    matrices: list[np.ndarray],
    metas: list[dict[str, Any]],
    directions: dict[str, np.ndarray],
) -> dict[str, Any]:
    dots = []
    cosines = []
    ratios = []
    subgroup: dict[str, list[bool]] = defaultdict(list)
    missing = 0
    for matrix, meta in zip(matrices, metas, strict=True):
        direction = directions.get(meta["contrast"])
        if direction is None:
            missing += 1
            continue
        value = matrix.astype(np.float32)
        direction = direction.astype(np.float32)
        dot = float(np.dot(value, direction))
        denom = float(np.linalg.norm(value) * np.linalg.norm(direction))
        cosine = dot / max(denom, 1e-8)
        ratio = dot / max(float(np.dot(direction, direction)), 1e-8)
        positive = dot > 0
        dots.append(dot)
        cosines.append(cosine)
        ratios.append(ratio)
        subgroup[
            f"t{meta['template']}.o{meta['order']}.q{meta['query_role']}"
        ].append(positive)
    subgroup_rates = {
        key: float(np.mean(values)) for key, values in subgroup.items()
    }
    return {
        "n": len(dots),
        "missing_direction_count": missing,
        "positive_rate": float(np.mean(np.asarray(dots) > 0)) if dots else 0.0,
        "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "median_cosine": float(np.median(cosines)) if cosines else 0.0,
        "median_direction_ratio": float(np.median(ratios)) if ratios else 0.0,
        "min_subgroup_positive_rate": min(subgroup_rates.values())
        if subgroup_rates
        else 0.0,
        "subgroup_positive_rates": subgroup_rates,
    }


def observational_score(metrics: dict[str, Any]) -> float:
    validation = metrics["validation"]
    return (
        validation["positive_rate"]
        + max(0.0, validation["mean_cosine"])
        + 0.25 * validation["min_subgroup_positive_rate"]
    )


def choose_chain(event_rows: list[dict[str, Any]], n_layers: int) -> dict[str, Any]:
    lookup = {(row["depth"], row["role"]): row for row in event_rows}
    best = None
    for write_depth in range(1, n_layers - 1):
        for read_depth in range(write_depth + 1, n_layers):
            for decision_depth in range(read_depth + 1, n_layers):
                rows = [
                    lookup[(write_depth, "source_color")],
                    lookup[(read_depth, "query_name")],
                    lookup[(decision_depth, "answer_boundary")],
                ]
                score = sum(row["selection_score"] for row in rows)
                candidate = {
                    "write": event_key(write_depth, "source_color"),
                    "read": event_key(read_depth, "query_name"),
                    "decision": event_key(decision_depth, "answer_boundary"),
                    "write_depth": write_depth,
                    "read_depth": read_depth,
                    "decision_depth": decision_depth,
                    "selection_score": score,
                }
                if best is None or candidate["selection_score"] > best["selection_score"]:
                    best = candidate
    if best is None:
        raise RuntimeError("no ordered candidate chain")
    return best


def run(batch_size: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 998 trace requires CUDA")
    protocol_root = OUT_ROOT / "protocol"
    behavior_root = OUT_ROOT / "behavior"
    output_root = OUT_ROOT / "trace"
    output_root.mkdir(parents=True, exist_ok=True)

    cases = read_jsonl(protocol_root / "cases.jsonl")
    behavior = read_jsonl(behavior_root / "behavior_rows.jsonl")
    behavior_summary = json.loads(
        (behavior_root / "summary.json").read_text(encoding="utf-8")
    )
    if not behavior_summary["behavior_gate_pass"]:
        raise RuntimeError("behavior gate did not authorize tracing")
    selected, selection_summary = select_pairs(cases, behavior)
    write_jsonl(output_root / "selected_pairs.jsonl", selected)
    write_json(output_root / "selection_summary.json", selection_summary)

    case_by_record = {row["record_id"]: row for row in cases}
    selected_by_group: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_group[(row["template"], row["partition"])].append(row)

    event_values: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    event_activation_sums: dict[str, np.ndarray] = {}
    event_activation_counts: dict[str, int] = defaultdict(int)
    partition_metas: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16, use_8bit=False)
        info = get_model_info(model, MODEL)
        total_batches = sum(
            math.ceil(len(rows) / batch_size) for rows in selected_by_group.values()
        )
        done = 0
        for (template, partition), pair_rows in sorted(selected_by_group.items()):
            pair_rows = sorted(pair_rows, key=lambda row: pair_hash(row["pair_id"]))
            for start in range(0, len(pair_rows), batch_size):
                batch_pairs = pair_rows[start : start + batch_size]
                rows_0 = [case_by_record[row["arm0_record_id"]] for row in batch_pairs]
                rows_1 = [case_by_record[row["arm1_record_id"]] for row in batch_pairs]
                values_0 = forward_roles(model, device, rows_0)
                values_1 = forward_roles(model, device, rows_1)
                for pair in batch_pairs:
                    partition_metas[pair["partition"]].append(pair)
                for (depth, role), source_values in values_0.items():
                    target_values = values_1[(depth, role)]
                    key = event_key(depth, role)
                    differences = (
                        source_values.astype(np.float32)
                        - target_values.astype(np.float32)
                    ).astype(np.float16)
                    event_values[key][partition].extend(
                        [row.copy() for row in differences]
                    )
                    if partition == "discovery":
                        activation_sum = (
                            np.abs(source_values.astype(np.float32))
                            + np.abs(target_values.astype(np.float32))
                        ).sum(axis=0)
                        if key not in event_activation_sums:
                            event_activation_sums[key] = activation_sum
                        else:
                            event_activation_sums[key] += activation_sum
                        event_activation_counts[key] += 2 * len(batch_pairs)
                del values_0, values_1
                done += 1
                if done % 4 == 0 or done == total_batches:
                    print(
                        f"[trace] {done}/{total_batches} batches, "
                        f"template={template}, partition={partition}",
                        flush=True,
                    )

        event_rows = []
        channel_sets = {}
        for depth in range(info.n_layers + 1):
            for role in ROLES:
                key = event_key(depth, role)
                discovery_values = event_values[key]["discovery"]
                discovery_metas = partition_metas["discovery"]
                if len(discovery_values) != len(discovery_metas):
                    raise RuntimeError(
                        f"event/meta count drift: {key}/"
                        f"{len(discovery_values)}/{len(discovery_metas)}"
                    )
                by_contrast: dict[str, list[np.ndarray]] = defaultdict(list)
                for value, meta in zip(
                    discovery_values, discovery_metas, strict=True
                ):
                    by_contrast[meta["contrast"]].append(value.astype(np.float32))
                directions = {
                    contrast: np.mean(np.stack(values), axis=0).astype(np.float32)
                    for contrast, values in by_contrast.items()
                }
                if len(directions) != 12:
                    raise RuntimeError(f"missing contrast directions: {key}/{directions.keys()}")
                direction_stack = np.stack(list(directions.values()))
                importance = np.sqrt(np.mean(direction_stack**2, axis=0))
                activation_mean = (
                    event_activation_sums[key] / event_activation_counts[key]
                )
                rng = np.random.default_rng(998_000 + depth * 10 + ROLES.index(role))
                channels = {
                    "difference_64": np.argsort(importance)[-64:][::-1].astype(int).tolist(),
                    "difference_256": np.argsort(importance)[-256:][::-1].astype(int).tolist(),
                    "top_activation_64": np.argsort(activation_mean)[-64:][::-1]
                    .astype(int)
                    .tolist(),
                    "top_activation_256": np.argsort(activation_mean)[-256:][::-1]
                    .astype(int)
                    .tolist(),
                    "random_64": rng.choice(info.d_model, 64, replace=False)
                    .astype(int)
                    .tolist(),
                    "random_256": rng.choice(info.d_model, 256, replace=False)
                    .astype(int)
                    .tolist(),
                }
                channel_sets[key] = channels
                metrics = {
                    partition: alignment_metrics(
                        event_values[key][partition],
                        partition_metas[partition],
                        directions,
                    )
                    for partition in PARTITIONS
                }
                row = {
                    "schema_version": "phase998_trace_event.v1",
                    "phase": PHASE,
                    "event": key,
                    "depth": depth,
                    "role": role,
                    "direction_count": len(directions),
                    "direction_norm_mean": float(
                        np.mean(np.linalg.norm(direction_stack, axis=1))
                    ),
                    "difference_channel_overlap_64_with_top_activation": len(
                        set(channels["difference_64"])
                        & set(channels["top_activation_64"])
                    ),
                    "difference_channel_overlap_256_with_top_activation": len(
                        set(channels["difference_256"])
                        & set(channels["top_activation_256"])
                    ),
                    "metrics": metrics,
                }
                row["selection_score"] = observational_score(metrics)
                event_rows.append(row)

        chain = choose_chain(event_rows, info.n_layers)
        selected_event_rows = {
            role: next(row for row in event_rows if row["event"] == chain[role])
            for role in ("write", "read", "decision")
        }
        checks = {}
        for role, row in selected_event_rows.items():
            checks[f"{role}_validation_positive"] = (
                row["metrics"]["validation"]["positive_rate"]
                >= OBSERVATION_THRESHOLDS["validation_positive_rate"]
            )
            checks[f"{role}_holdout_positive"] = (
                row["metrics"]["holdout"]["positive_rate"]
                >= OBSERVATION_THRESHOLDS["holdout_positive_rate"]
            )
            checks[f"{role}_validation_subgroup"] = (
                row["metrics"]["validation"]["min_subgroup_positive_rate"]
                >= OBSERVATION_THRESHOLDS[
                    "validation_min_subgroup_positive_rate"
                ]
            )
        observation_gate = all(checks.values())
        summary = {
            "schema_version": "phase998_trace_summary.v1",
            "phase": PHASE,
            "model": MODEL,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "selected_pair_count": len(selected),
            "partition_counts": selection_summary["partition_counts"],
            "event_count": len(event_rows),
            "selection_uses_holdout": False,
            "selected_chain": chain,
            "selected_event_metrics": selected_event_rows,
            "observation_thresholds": OBSERVATION_THRESHOLDS,
            "observation_checks": checks,
            "observation_gate_pass": observation_gate,
            "elapsed_seconds": time.time() - started,
        }
        write_jsonl(output_root / "event_metrics.jsonl", event_rows)
        write_json(output_root / "channel_sets.json", channel_sets)
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
    args = parser.parse_args()
    result = run(args.batch_size)
    print(
        json.dumps(
            {
                "passed": result["observation_gate_pass"],
                "selected_chain": result["selected_chain"],
                "partition_counts": result["partition_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
