#!/usr/bin/env python3
"""Identify frozen directional HiddenState responses without claiming a full Jacobian."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2315 = RESULT / "phase2315_c5041_c5100_active_response_contract"
P2316 = RESULT / "phase2316_c5101_c5160_qwen4b_active_baseline"
OUT = RESULT / "phase2317_c5161_c5240_directional_response_identification"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = P2315 / "material/natural_active_response_bilingual.jsonl"
BOUNDARY = P2316 / "raw/qwen4b_boundary_all_checkpoints.float16.npy"
ACTIVE_INDEX = OUT / "index/active_rows.jsonl"
DERIVATIVE = OUT / "raw/directional_derivative.float16.npy"
EVEN = OUT / "raw/even_response.float16.npy"
MARGINS = OUT / "raw/identity_margins.float32.npy"
PROGRESS = OUT / "raw/progress.json"
sys.path.insert(0, str(TESTS))

import phase1332_bf16_utils as model_base  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as contract  # noqa: E402


PHASE = 2317
CAMPAIGN = "C5161-C5240"
EPS = 1e-12
SOURCE_QPOINTS = contract.SOURCE_QPOINTS_4B
BASE_PROBES = contract.BASE_PROBES
PAIR_PROBES = contract.PAIR_PROBES
PROBE_COUNT = BASE_PROBES + len(PAIR_PROBES)
DOSE = contract.PERTURBATION_DOSE
TARGET_QPOINTS = {
    q: (q + 1, min(q + 4, 36), 37) for q in SOURCE_QPOINTS
}
MODEL_NAMES = (
    "zero_response", "global_probe_response", "family_probe_response",
    "family_state_probe_response", "family_language_probe_response",
    "family_surface_probe_response",
)


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def probe_directions(dimension: int) -> tuple[np.ndarray, list[dict]]:
    base = []
    ledger = []
    for probe in range(BASE_PROBES):
        digest = hashlib.sha256(f"phase2315|rademacher|{dimension}|{probe}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        value = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dimension)
        value /= np.linalg.norm(value.astype(np.float64))
        base.append(value)
        ledger.append({"probe": probe, "kind": "base_rademacher", "members": [probe]})
    output = list(base)
    for offset, (left, right) in enumerate(PAIR_PROBES):
        output.append(base[left] + base[right])
        ledger.append({"probe": BASE_PROBES + offset, "kind": "pair_sum", "members": [left, right]})
    return np.stack(output).astype(np.float32), ledger


def active_rows(rows: list[dict]) -> list[dict]:
    selected = []
    partition_unit = {
        partition: min(unit for unit, value in contract.PARTITION_BY_UNIT.items() if value == partition)
        for partition in contract.PARTITIONS
    }
    for partition_index, partition in enumerate(contract.PARTITIONS):
        unit = partition_unit[partition]
        for family in contract.FAMILIES:
            for language_index, language in enumerate(contract.LANGUAGES):
                for surface_index, surface in enumerate(contract.SURFACES):
                    state = (partition_index + language_index + surface_index) % 2
                    selected.append(next(row for row in rows
                                         if row["family"] == family and row["partition"] == partition
                                         and int(row["unit"]) == unit and row["language"] == language
                                         and row["surface"] == surface and int(row["state"]) == state))
    return selected


def prepare_arrays(count: int, dimension: int) -> tuple[np.memmap, np.memmap, np.memmap, int]:
    response_shape = (count, len(SOURCE_QPOINTS), PROBE_COUNT, 3, dimension)
    margin_shape = (count, len(SOURCE_QPOINTS), PROBE_COUNT, 2)
    DERIVATIVE.parent.mkdir(parents=True, exist_ok=True)
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if progress["response_shape"] != list(response_shape) or progress["margin_shape"] != list(margin_shape):
            raise RuntimeError(("resume_shape", progress, response_shape, margin_shape))
        return (np.lib.format.open_memmap(DERIVATIVE, mode="r+"),
                np.lib.format.open_memmap(EVEN, mode="r+"),
                np.lib.format.open_memmap(MARGINS, mode="r+"), int(progress["completed_cells"]))
    return (
        np.lib.format.open_memmap(DERIVATIVE, mode="w+", dtype=np.float16, shape=response_shape),
        np.lib.format.open_memmap(EVEN, mode="w+", dtype=np.float16, shape=response_shape),
        np.lib.format.open_memmap(MARGINS, mode="w+", dtype=np.float32, shape=margin_shape),
        0,
    )


def fixed_identity_tokens(row: dict) -> tuple[int, int]:
    return int(row["identity_target_ids"][0]), int(row["identity_wrong_ids"][0])


def run_cell(model, device, row: dict, source_q: int, directions: np.ndarray,
             baseline_field: np.ndarray, hidden_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    module_list = modules(model)
    targets = TARGET_QPOINTS[source_q]
    base_source = baseline_field[hidden_index, source_q].astype(np.float32)
    hidden_norm = float(np.linalg.norm(base_source.astype(np.float64)))
    variants = []
    for probe in range(PROBE_COUNT):
        for sign in (1.0, -1.0):
            variants.append((probe, sign, directions[probe] * (sign * DOSE * hidden_norm)))
    deltas = torch.tensor(np.stack([value[2] for value in variants]), dtype=torch.float32, device=device)
    captures: dict[int, torch.Tensor] = {}
    handles = []

    def source_hook(_module, _inputs, value):
        tensor = value[0] if isinstance(value, tuple) else value
        changed = tensor.clone()
        changed[:, -1] = changed[:, -1] + deltas.to(dtype=changed.dtype)
        return (changed, *value[1:]) if isinstance(value, tuple) else changed

    handles.append(module_list[source_q].register_forward_hook(source_hook))
    for target in targets:
        def target_hook(_module, _inputs, value, target=target):
            captures[target] = value[0] if isinstance(value, tuple) else value
        handles.append(module_list[target].register_forward_hook(target_hook))
    ids = torch.tensor([row["future_prompt_ids"]] * len(variants), dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    target_id, wrong_id = fixed_identity_tokens(row)
    try:
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        logits = result.logits[:, -1].float()
        margins = (logits[:, target_id] - logits[:, wrong_id]).cpu().numpy()
        derivative = np.empty((PROBE_COUNT, 3, int(model.config.hidden_size)), dtype=np.float32)
        even = np.empty_like(derivative)
        for probe in range(PROBE_COUNT):
            plus_index, minus_index = probe * 2, probe * 2 + 1
            for target_index, target_q in enumerate(targets):
                plus = captures[target_q][plus_index, -1].float().cpu().numpy()
                minus = captures[target_q][minus_index, -1].float().cpu().numpy()
                baseline = baseline_field[hidden_index, target_q].astype(np.float32)
                derivative[probe, target_index] = (plus - minus) / (2.0 * DOSE * hidden_norm)
                even[probe, target_index] = (plus + minus) * 0.5 - baseline
        return derivative, even, margins.reshape(PROBE_COUNT, 2)
    finally:
        for handle in handles:
            handle.remove()
        captures.clear()


def collect(model, device, rows: list[dict], selected: list[dict], directions: np.ndarray) -> dict:
    baseline = np.load(BOUNDARY, mmap_mode="r")
    derivative, even, margins, completed_cells = prepare_arrays(len(selected), int(model.config.hidden_size))
    total_cells = len(selected) * len(SOURCE_QPOINTS)
    try:
        for cell in range(completed_cells, total_cells):
            row_index, source_index = divmod(cell, len(SOURCE_QPOINTS))
            row = selected[row_index]
            source_q = SOURCE_QPOINTS[source_index]
            hidden_index = int(row["design_index"])
            response, symmetric, output_margins = run_cell(
                model, device, row, source_q, directions, baseline, hidden_index
            )
            derivative[row_index, source_index] = response.astype(np.float16)
            even[row_index, source_index] = symmetric.astype(np.float16)
            margins[row_index, source_index] = output_margins
            derivative.flush(); even.flush(); margins.flush()
            save(PROGRESS, {
                "completed_cells": cell + 1, "total_cells": total_cells,
                "response_shape": list(derivative.shape), "margin_shape": list(margins.shape),
            })
            print(f"[phase2317 active] {cell + 1}/{total_cells}", flush=True)
    finally:
        for value in (baseline, derivative, even, margins):
            close_memmap(value)
    index_rows = [{
        "active_index": index, "hidden_index": int(row["design_index"]),
        "case_id": row["case_id"], "family": row["family"], "partition": row["partition"],
        "language": row["language"], "surface": row["surface"],
        "unit": int(row["unit"]), "state": int(row["state"]),
        "primary_positions": row["role_positions"]["primary"],
    } for index, row in enumerate(selected)]
    write_rows(ACTIVE_INDEX, index_rows)
    return {"rows": len(selected), "cells": total_cells,
            "derivative_shape": [len(selected), len(SOURCE_QPOINTS), PROBE_COUNT, 3,
                                 int(model.config.hidden_size)],
            "margin_shape": [len(selected), len(SOURCE_QPOINTS), PROBE_COUNT, 2]}


def response_key(row: dict, model_name: str, source_index: int, probe: int, target_index: int) -> tuple:
    if model_name == "global_probe_response":
        return (source_index, probe, target_index)
    if model_name == "family_probe_response":
        return (row["family"], source_index, probe, target_index)
    if model_name == "family_state_probe_response":
        return (row["family"], int(row["state"]), source_index, probe, target_index)
    if model_name == "family_language_probe_response":
        return (row["family"], row["language"], source_index, probe, target_index)
    if model_name == "family_surface_probe_response":
        return (row["family"], row["surface"], source_index, probe, target_index)
    raise KeyError(model_name)


def fit_means(selected: list[dict], derivative: np.ndarray) -> dict[str, dict[tuple, np.ndarray]]:
    sums: dict[str, dict[tuple, np.ndarray]] = {name: {} for name in MODEL_NAMES if name != "zero_response"}
    counts: dict[str, dict[tuple, int]] = {name: {} for name in sums}
    discovery = [index for index, row in enumerate(selected) if row["partition"] == "discovery"]
    for row_index in discovery:
        row = selected[row_index]
        for source_index in range(len(SOURCE_QPOINTS)):
            for probe in range(BASE_PROBES):
                for target_index in range(3):
                    actual = derivative[row_index, source_index, probe, target_index].astype(np.float64)
                    for model_name in sums:
                        key = response_key(row, model_name, source_index, probe, target_index)
                        if key not in sums[model_name]:
                            sums[model_name][key] = np.zeros_like(actual)
                            counts[model_name][key] = 0
                        sums[model_name][key] += actual
                        counts[model_name][key] += 1
    for model_name in sums:
        for key in sums[model_name]:
            sums[model_name][key] /= counts[model_name][key]
    return sums


def prediction_analysis(selected: list[dict]) -> dict:
    derivative = np.load(DERIVATIVE, mmap_mode="r")
    means = fit_means(selected, derivative)
    records = []
    for row_index, row in enumerate(selected):
        if row["partition"] == "discovery":
            continue
        for source_index, source_q in enumerate(SOURCE_QPOINTS):
            for probe in range(BASE_PROBES):
                for target_index, target_q in enumerate(TARGET_QPOINTS[source_q]):
                    actual = derivative[row_index, source_index, probe, target_index].astype(np.float64)
                    denominator = float(np.dot(actual, actual)) + EPS
                    for model_name in MODEL_NAMES:
                        if model_name == "zero_response":
                            prediction = np.zeros_like(actual)
                        else:
                            key = response_key(row, model_name, source_index, probe, target_index)
                            prediction = means[model_name][key]
                        error = actual - prediction
                        records.append({
                            "case_id": row["case_id"], "family": row["family"],
                            "partition": row["partition"], "language": row["language"],
                            "surface": row["surface"], "state": int(row["state"]),
                            "source_q": source_q, "target_q": target_q, "probe": probe,
                            "model": model_name,
                            "relative_mse": float(np.dot(error, error) / denominator),
                            "sign_agreement": float(np.mean(actual * prediction > 0))
                            if model_name != "zero_response" else 0.0,
                        })
    write_rows(OUT / "prediction/directional_model_records.jsonl", records)
    summary = {}
    for partition in ("confirmation", "fresh_confirmation", "fresh_lockbox"):
        summary[partition] = {}
        for family in contract.FAMILIES:
            summary[partition][family] = {}
            for model_name in MODEL_NAMES:
                values = [row for row in records if row["partition"] == partition
                          and row["family"] == family and row["model"] == model_name]
                summary[partition][family][model_name] = {
                    "cells": len(values),
                    "median_relative_mse": float(np.median([row["relative_mse"] for row in values])),
                    "mean_relative_mse": float(np.mean([row["relative_mse"] for row in values])),
                    "median_sign_agreement": float(np.median([row["sign_agreement"] for row in values])),
                }
    close_memmap(derivative)
    save(OUT / "prediction/directional_model_summary.json", summary)
    return {"records": len(records), "summary": summary}


def superposition_and_even(selected: list[dict]) -> dict:
    derivative = np.load(DERIVATIVE, mmap_mode="r")
    even = np.load(EVEN, mmap_mode="r")
    baseline = np.load(BOUNDARY, mmap_mode="r")
    pair_records, even_records = [], []
    for row_index, row in enumerate(selected):
        for source_index, source_q in enumerate(SOURCE_QPOINTS):
            for target_index, target_q in enumerate(TARGET_QPOINTS[source_q]):
                for pair_offset, (left, right) in enumerate(PAIR_PROBES):
                    pair_index = BASE_PROBES + pair_offset
                    actual = derivative[row_index, source_index, pair_index, target_index].astype(np.float64)
                    predicted = (derivative[row_index, source_index, left, target_index].astype(np.float64)
                                 + derivative[row_index, source_index, right, target_index].astype(np.float64))
                    pair_records.append({
                        "case_id": row["case_id"], "family": row["family"],
                        "partition": row["partition"], "source_q": source_q, "target_q": target_q,
                        "pair": [left, right],
                        "relative_mse": float(np.sum(np.square(actual - predicted))
                                              / (np.sum(np.square(actual)) + EPS)),
                    })
                for probe in range(PROBE_COUNT):
                    odd = derivative[row_index, source_index, probe, target_index].astype(np.float64)
                    symmetric = even[row_index, source_index, probe, target_index].astype(np.float64)
                    source_norm = float(np.linalg.norm(
                        baseline[int(row["design_index"]), source_q].astype(np.float64)
                    ))
                    odd_effect = odd * (DOSE * source_norm)
                    even_records.append({
                        "case_id": row["case_id"], "family": row["family"],
                        "partition": row["partition"], "source_q": source_q, "target_q": target_q,
                        "probe": probe,
                        "even_to_odd_l2": float(np.linalg.norm(symmetric)
                                                / (np.linalg.norm(odd_effect) + EPS)),
                    })
    write_rows(OUT / "analysis/pair_superposition.jsonl", pair_records)
    write_rows(OUT / "analysis/even_response_ratios.jsonl", even_records)
    summary = {}
    for family in contract.FAMILIES:
        summary[family] = {
            "pair_relative_mse_by_partition": {
                partition: float(np.median([row["relative_mse"] for row in pair_records
                                            if row["family"] == family and row["partition"] == partition]))
                for partition in contract.PARTITIONS
            },
            "even_to_odd_by_partition": {
                partition: float(np.median([row["even_to_odd_l2"] for row in even_records
                                            if row["family"] == family and row["partition"] == partition]))
                for partition in contract.PARTITIONS
            },
        }
    close_memmap(derivative); close_memmap(even); close_memmap(baseline)
    save(OUT / "analysis/local_linearity_summary.json", summary)
    return {"pair_records": len(pair_records), "even_records": len(even_records), "families": summary}


def discovery_selections(selected: list[dict]) -> dict:
    margins = np.load(MARGINS, mmap_mode="r")
    output = {}
    for family in contract.FAMILIES:
        candidates = []
        discovery = [index for index, row in enumerate(selected)
                     if row["family"] == family and row["partition"] == "discovery"]
        for source_index, source_q in enumerate(SOURCE_QPOINTS):
            for probe in range(BASE_PROBES):
                central = np.array([(margins[index, source_index, probe, 0]
                                     - margins[index, source_index, probe, 1]) * 0.5
                                    for index in discovery], dtype=np.float64)
                median = float(np.median(central))
                candidates.append({
                    "source_q": source_q, "probe": probe,
                    "raw_median_central_effect": median,
                    "sign": 1 if median >= 0 else -1,
                    "signed_median_effect": abs(median),
                    "signed_positive_rate": float(np.mean(central * (1 if median >= 0 else -1) > 0)),
                })
        candidates.sort(key=lambda row: (row["signed_median_effect"], row["signed_positive_rate"]), reverse=True)
        output[family] = {"selected": candidates[0], "all_candidates": candidates}
    close_memmap(margins)
    value = {
        "selection_partition": "discovery_only", "families": output,
        "claim_boundary": "output-targeted structured probe, not a semantic direction",
    }
    save(OUT / "control/discovery_frozen_selections.json", value)
    return value


def confirmation_and_fresh_probe_readout(selected: list[dict], selections: dict) -> dict:
    margins = np.load(MARGINS, mmap_mode="r")
    result = {}
    for family in contract.FAMILIES:
        chosen = selections["families"][family]["selected"]
        source_index = SOURCE_QPOINTS.index(int(chosen["source_q"]))
        probe = int(chosen["probe"])
        sign = int(chosen["sign"])
        family_result = {}
        for partition in ("confirmation", "fresh_confirmation", "fresh_lockbox"):
            indices = [index for index, row in enumerate(selected)
                       if row["family"] == family and row["partition"] == partition]
            effects = np.array([sign * (margins[index, source_index, probe, 0]
                                        - margins[index, source_index, probe, 1]) * 0.5
                                for index in indices], dtype=np.float64)
            family_result[partition] = {
                "rows": len(indices), "median_signed_effect": float(np.median(effects)),
                "positive_rate": float(np.mean(effects > 0)),
            }
        confirmation = family_result["confirmation"]
        family_result["confirmation_passed"] = (
            confirmation["median_signed_effect"] > 0 and confirmation["positive_rate"] >= 0.75
        )
        result[family] = family_result
    close_memmap(margins)
    save(OUT / "control/prospective_probe_readout.json", result)
    return result


def candidate_logprob_margin(logits: torch.Tensor, prompt_length: int,
                             target: list[int], wrong: list[int]) -> tuple[float, float]:
    def score(candidate: list[int]) -> tuple[float, float]:
        selected = logits[prompt_length - 1:prompt_length - 1 + len(candidate)].float()
        ids = torch.tensor(candidate, dtype=torch.long, device=selected.device)
        values = selected.gather(1, ids[:, None])[:, 0] - torch.logsumexp(selected, dim=-1)
        return float(values.sum().item()), float(values.mean().item())
    target_sum, target_mean = score(target)
    wrong_sum, wrong_mean = score(wrong)
    return target_sum - wrong_sum, target_mean - wrong_mean


def full_sequence_controls(model, device, selected: list[dict], directions: np.ndarray,
                           selections: dict) -> dict:
    module_list = modules(model)
    records = []
    family_cycle = {family: contract.FAMILIES[(index + 1) % len(contract.FAMILIES)]
                    for index, family in enumerate(contract.FAMILIES)}
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    for row_index, row in enumerate(selected):
        if row["partition"] == "discovery":
            continue
        chosen = selections["families"][row["family"]]["selected"]
        wrong_chosen = selections["families"][family_cycle[row["family"]]]["selected"]
        primary_positions = row["role_positions"].get("primary", [])
        wrong_role_position = max(primary_positions) if primary_positions else int(row["boundary_position"])
        variants = [
            {"name": "baseline", "source_q": None, "probe": None, "sign": 0, "position": None},
            {"name": "selected", "source_q": int(chosen["source_q"]), "probe": int(chosen["probe"]),
             "sign": int(chosen["sign"]), "position": int(row["boundary_position"])},
            {"name": "reverse", "source_q": int(chosen["source_q"]), "probe": int(chosen["probe"]),
             "sign": -int(chosen["sign"]), "position": int(row["boundary_position"])},
            {"name": "wrong_family", "source_q": int(wrong_chosen["source_q"]),
             "probe": int(wrong_chosen["probe"]), "sign": int(wrong_chosen["sign"]),
             "position": int(row["boundary_position"])},
            {"name": "wrong_role", "source_q": int(chosen["source_q"]), "probe": int(chosen["probe"]),
             "sign": int(chosen["sign"]), "position": wrong_role_position},
            {"name": "wrong_layer", "source_q": min(int(chosen["source_q"]) + 1, 35),
             "probe": int(chosen["probe"]), "sign": int(chosen["sign"]),
             "position": int(row["boundary_position"])},
        ]
        items = []
        for variant in variants:
            for candidate_kind, key in (("target", "future_target_ids"), ("wrong", "future_wrong_ids")):
                candidate = row[key]
                items.append({**variant, "candidate_kind": candidate_kind,
                              "candidate": candidate, "sequence": row["future_prompt_ids"] + candidate})
        width = max(len(item["sequence"]) for item in items)
        ids = torch.full((len(items), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        positions = torch.zeros_like(ids)
        for index, item in enumerate(items):
            sequence = item["sequence"]
            ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            mask[index, :len(sequence)] = 1
            positions[index, :len(sequence)] = torch.arange(len(sequence), device=device)
        handles = []
        for source_q in sorted({item["source_q"] for item in items if item["source_q"] is not None}):
            def hook(_module, _inputs, value, source_q=source_q):
                tensor = value[0] if isinstance(value, tuple) else value
                changed = tensor.clone()
                for index, item in enumerate(items):
                    if item["source_q"] != source_q:
                        continue
                    position = int(item["position"])
                    base = tensor[index, position].float()
                    scale = DOSE * float(torch.linalg.vector_norm(base).item()) * int(item["sign"])
                    direction = torch.tensor(directions[int(item["probe"])], device=tensor.device,
                                             dtype=tensor.dtype)
                    changed[index, position] = changed[index, position] + direction * scale
                return (changed, *value[1:]) if isinstance(value, tuple) else changed
            handles.append(module_list[source_q].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                               use_cache=False, return_dict=True).logits
            by_variant: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
            for index, item in enumerate(items):
                by_variant[item["name"]][item["candidate_kind"]] = logits[index]
            for variant in variants:
                name = variant["name"]
                # Score each candidate on its own run; this preserves its own teacher-forced history.
                target_logits = by_variant[name]["target"]
                wrong_logits = by_variant[name]["wrong"]
                prompt_length = len(row["future_prompt_ids"])
                target_ids = row["future_target_ids"]
                wrong_ids = row["future_wrong_ids"]

                def sequence_score(value: torch.Tensor, candidate: list[int]) -> tuple[float, float]:
                    chosen_logits = value[prompt_length - 1:prompt_length - 1 + len(candidate)].float()
                    token_ids = torch.tensor(candidate, dtype=torch.long, device=value.device)
                    logps = chosen_logits.gather(1, token_ids[:, None])[:, 0] - torch.logsumexp(chosen_logits, dim=-1)
                    return float(logps.sum().item()), float(logps.mean().item())

                target_sum, target_mean = sequence_score(target_logits, target_ids)
                wrong_sum, wrong_mean = sequence_score(wrong_logits, wrong_ids)
                records.append({
                    "case_id": row["case_id"], "family": row["family"],
                    "partition": row["partition"], "language": row["language"],
                    "surface": row["surface"], "state": int(row["state"]),
                    "variant": name, "source_q": variant["source_q"], "probe": variant["probe"],
                    "sign": variant["sign"], "position": variant["position"],
                    "sum_margin": target_sum - wrong_sum,
                    "mean_margin": target_mean - wrong_mean,
                })
        finally:
            for handle in handles:
                handle.remove()
        print(f"[phase2317 full-sequence controls] {row_index + 1}/{len(selected)}", flush=True)
    write_rows(OUT / "control/full_sequence_controls.jsonl", records)
    summary = {}
    for family in contract.FAMILIES:
        family_summary = {}
        for partition in ("confirmation", "fresh_confirmation", "fresh_lockbox"):
            rows_by_case: dict[str, dict[str, dict]] = defaultdict(dict)
            for record in records:
                if record["family"] == family and record["partition"] == partition:
                    rows_by_case[record["case_id"]][record["variant"]] = record
            effects = defaultdict(list)
            for variants in rows_by_case.values():
                baseline = variants["baseline"]["mean_margin"]
                for name in ("selected", "reverse", "wrong_family", "wrong_role", "wrong_layer"):
                    effects[name].append(variants[name]["mean_margin"] - baseline)
            family_summary[partition] = {
                name: {"median_effect": float(np.median(values)),
                       "positive_rate": float(np.mean(np.array(values) > 0))}
                for name, values in effects.items()
            }
        confirmation = family_summary["confirmation"]
        family_summary["confirmation_control_gate"] = (
            confirmation["selected"]["median_effect"] > 0
            and confirmation["selected"]["positive_rate"] >= 0.75
            and confirmation["selected"]["median_effect"] > confirmation["reverse"]["median_effect"]
            and confirmation["selected"]["median_effect"] > confirmation["wrong_family"]["median_effect"]
            and confirmation["selected"]["median_effect"] > confirmation["wrong_role"]["median_effect"]
            and confirmation["selected"]["median_effect"] > confirmation["wrong_layer"]["median_effect"]
        )
        summary[family] = family_summary
    save(OUT / "control/full_sequence_control_summary.json", summary)
    return {"records": len(records), "families": summary,
            "qualified_families": [family for family, value in summary.items()
                                   if value["confirmation_control_gate"]]}


def coordinate_passports(selected: list[dict], prediction: dict) -> dict:
    derivative = np.load(DERIVATIVE, mmap_mode="r")
    # Preserve every coordinate: family-vs-global fresh error improvement per coordinate.
    means = fit_means(selected, derivative)
    fresh_partitions = ("fresh_confirmation", "fresh_lockbox")
    output = np.zeros((len(contract.FAMILIES), len(SOURCE_QPOINTS), BASE_PROBES, 3,
                       derivative.shape[-1]), dtype=np.float32)
    counts = np.zeros(output.shape[:-1], dtype=np.int64)
    for row_index, row in enumerate(selected):
        if row["partition"] not in fresh_partitions:
            continue
        fi = contract.FAMILIES.index(row["family"])
        for source_index in range(len(SOURCE_QPOINTS)):
            for probe in range(BASE_PROBES):
                for target_index in range(3):
                    actual = derivative[row_index, source_index, probe, target_index].astype(np.float64)
                    family_key = response_key(row, "family_probe_response", source_index, probe, target_index)
                    global_key = response_key(row, "global_probe_response", source_index, probe, target_index)
                    family_error = np.square(actual - means["family_probe_response"][family_key])
                    global_error = np.square(actual - means["global_probe_response"][global_key])
                    scale = np.square(actual) + EPS
                    output[fi, source_index, probe, target_index] += (
                        (global_error - family_error) / scale
                    ).astype(np.float32)
                    counts[fi, source_index, probe, target_index] += 1
    output /= counts[..., None]
    path = OUT / "atlas/fresh_family_vs_global_coordinate_improvement.float32.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, output, allow_pickle=False)
    close_memmap(derivative)
    return {"path": str(path.relative_to(ROOT)), "shape": list(output.shape),
            "meaning": "positive means family-conditioned discovery mean reduces fresh squared error vs global mean"}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact_prediction = {
        partition: {
            family: {
                model: round(values["median_relative_mse"], 6)
                for model, values in models.items()
            } for family, models in families.items()
        } for partition, families in result["prediction"]["summary"].items()
    }
    compact_linearity = {
        family: {"pair": value["pair_relative_mse_by_partition"],
                 "even": value["even_to_odd_by_partition"]}
        for family, value in result["linearity"]["families"].items()
    }
    text = rf"""

## Phase {PHASE}: 冻结全坐标方向响应、局部叠加与前瞻控制（{CAMPAIGN}） [{stamp}]

**测试原理与证据资格。** Phase2316 没有构式同时通过严格完整未来门和自由续写门，因此本 Phase 的八族主动扰动只能回答“语言条件下的模型局部方向响应是否可重复”，不能回答语义充分性。模型加载前已冻结 8 个遍布全部 2560 坐标的 Rademacher 方向、4 个成对和方向、源检查点 `10/20/30`、目标检查点 `q+1/q+4/final_norm`、正负剂量 `0.01 * ||h||`、四个分区各族 4 行，以及零/全局/族/状态/语言/表面六个预测模型。没有按结果选择坐标，没有 Top-K、PCA 或余弦筛选。

$$
D_{{q\to t}}(r)=\frac{{H_t(h_q+\epsilon\lVert h_q\rVert r)-H_t(h_q-\epsilon\lVert h_q\rVert r)}}{{2\epsilon\lVert h_q\rVert}},
$$

$$
E_{{a,b}}=\frac{{\lVert D(r_a+r_b)-D(r_a)-D(r_b)\rVert_2^2}}{{\lVert D(r_a+r_b)\rVert_2^2+\varepsilon}},\qquad
N_{{model}}=\frac{{\lVert D-\widehat D_{{model}}\rVert_2^2}}{{\lVert D\rVert_2^2+\varepsilon}}.
$$

**结果汇总。** 主动数据 `{json.dumps(result['collection'], ensure_ascii=False)}`。confirmation/fresh 各模型中位相对 MSE `{json.dumps(compact_prediction, ensure_ascii=False)}`。成对叠加误差与偶响应比 `{json.dumps(compact_linearity, ensure_ascii=False)}`。discovery 冻结的输出靶向方向及 confirmation/fresh 读出 `{json.dumps(result['probe_readout'], ensure_ascii=False)}`。完整未来的正确方向、反向、错族、错角色、错层控制 `{json.dumps(result['full_sequence_control'], ensure_ascii=False)}`。其中通过控制门的族 `{result['full_sequence_control']['qualified_families']}` 仍不具备语义机制资格，因为 Phase2316 双行为门交集为空；自由生成救援严格记为 `NA_not_authorized`。

**逐坐标资产、文件与审计。** 方向导数形状 `{result['collection']['derivative_shape']}`，偶响应同形；fresh 逐坐标“族均值相对全局均值的误差改善”护照 `{json.dumps(result['coordinate_passport'], ensure_ascii=False)}`。检查 `{json.dumps(result['checks'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2317_c5161_c5240_directional_response_identification.py`；结果 `tests/glm5/result/phase2317_c5161_c5240_directional_response_identification`。

**理论进展、硬伤与结论。** 理论主体仍为“条件化输出场闭合理论”。本期只有当族条件预测在锁箱上持续优于全局与零模型时，才增加“冻结方向响应具有族条件成分”这一窄拼图；无论结果如何，都没有识别完整非对角 Jacobian、固定语义方向或坐标齿轮。八方向只覆盖极小子空间；有限差分是局部的；输出方向由 discovery 靶向；完整未来使用 teacher forcing；四行/分区的主动样本仍小；所有材料无独立人类盲评。下一步独立执行 Qwen3-14B、GLM4 与 DeepSeek-7B 的小型锁箱面板，只比较模型本地功能曲线；任何模型失败不阻断其他路线。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    for parent in (P2315, P2316):
        value = json.loads((parent / "analysis/final.json").read_text(encoding="utf-8"))
        if not value["all_checks_passed"]:
            raise RuntimeError(("parent_not_authorized", parent))
    rows = read_rows(ROWS_PATH)
    selected = active_rows(rows)
    directions, probe_ledger = probe_directions(2560)
    save(OUT / "config/probe_ledger.json", probe_ledger)
    model = None
    try:
        model, tokenizer, device, placement = model_base.load_bf16("qwen3")
        collection = collect(model, device, rows, selected, directions)
        prediction = prediction_analysis(selected)
        linearity = superposition_and_even(selected)
        selections = discovery_selections(selected)
        probe_readout = confirmation_and_fresh_probe_readout(selected, selections)
        full_control = full_sequence_controls(model, device, selected, directions, selections)
        coordinate = coordinate_passports(selected, prediction)
        behavior = json.loads((P2316 / "analysis/final.json").read_text(encoding="utf-8"))
        semantic_eligible = behavior["qualified_families"]
        hashes = {
            "derivative": file_hash(DERIVATIVE), "even": file_hash(EVEN),
            "margins": file_hash(MARGINS), "active_index": file_hash(ACTIVE_INDEX),
            "coordinate_passport": file_hash(ROOT / coordinate["path"]),
        }
        checks = {
            "parents_authorized": True, "active_rows": len(selected) == 128,
            "all_cells_collected": collection["cells"] == 384,
            "all_original_coordinates": collection["derivative_shape"][-1] == 2560,
            "prediction_all_preregistered_models": all(
                set(models) == set(MODEL_NAMES)
                for families in prediction["summary"].values() for models in families.values()
            ),
            "pair_and_even_all_families": set(linearity["families"]) == set(contract.FAMILIES),
            "selections_discovery_only": selections["selection_partition"] == "discovery_only",
            "all_control_variants": full_control["records"] == 8 * 3 * 4 * 6,
            "semantic_free_generation_not_run_without_dual_gate": len(semantic_eligible) == 0,
            "full_jacobian_not_claimed": True, "no_topk_pca_cosine": True,
        }
        result = {
            "phase": PHASE, "campaign": CAMPAIGN,
            "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
            "model": "Qwen3-4B", "precision": "bfloat16", "quantization": "none",
            "placement": placement, "collection": collection, "prediction": prediction,
            "linearity": linearity, "selections": selections, "probe_readout": probe_readout,
            "full_sequence_control": full_control, "coordinate_passport": coordinate,
            "semantic_behavior_eligible_families": semantic_eligible,
            "free_generation_rescue": "NA_not_authorized_no_dual_behavior_gate_family",
            "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
            "strict_conclusion": (
                "Frozen full-coordinate directional responses, superposition errors, model comparisons, and "
                "prospective output controls were completed for all eight families. Because no family passed both "
                "Phase2316 behavior gates, these results are local system-identification observations and cannot be "
                "promoted to semantic sufficiency or a full Jacobian."
            ),
            "next_authorization": "Run sequential model-local fresh functional panels; never align physical coordinates.",
        }
        save(final_path, result)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            model_base.release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
