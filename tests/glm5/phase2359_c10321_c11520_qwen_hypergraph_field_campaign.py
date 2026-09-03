#!/usr/bin/env python3
"""Qwen3-4B full-coordinate hypergraph field, factorial, dynamics, prediction and generation campaign."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2358 = RESULT / "phase2358_c10161_c10320_external_hypergraph_factorial_contract"
OUT2359 = RESULT / "phase2359_c10321_c10560_qwen4b_hypergraph_full_field"
OUT2360 = RESULT / "phase2360_c10561_c10800_factorial_coordinate_route_scan"
OUT2361 = RESULT / "phase2361_c10801_c11040_layer_token_coordinate_dynamics"
OUT2362 = RESULT / "phase2362_c11041_c11280_composition_prediction_tournament"
OUT2363 = RESULT / "phase2363_c11281_c11520_balanced_generation_realization"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = P2358 / "material/bilingual_typed_hypergraph_factorial.jsonl"
STATES = OUT2359 / "raw/qwen4b_boundary_all_checkpoints.float16.npy"
DECISIONS = OUT2359 / "raw/qwen4b_first_token_decisions.float32.npy"
COLLECT_PROGRESS = OUT2359 / "raw/progress.json"
TOKEN_FIELD = OUT2359 / "raw/qwen4b_reference_all_token_all_checkpoints.float16.npy"
TOKEN_INDEX = OUT2359 / "index/reference_all_token_rows.jsonl"
TOKEN_PROGRESS = OUT2359 / "raw/token_progress.json"
COEFF = OUT2360 / "raw/qwen4b_boolean_factorial_coefficients.float16.npy"
COEFF_PROGRESS = OUT2360 / "raw/coeff_progress.json"
CMI = OUT2360 / "derived/conditional_sign_information.float32.npy"
TENSOR_LOADINGS = OUT2360 / "derived/family_mean_tensor_coordinate_loadings.float32.npy"
COOP = OUT2361 / "derived/coordinate_cooperation_correlation.float32.npy"
PRED_IMPROVEMENT = OUT2362 / "derived/prediction_coordinate_improvement.float32.npy"
TRAJECTORY = OUT2363 / "raw/qwen4b_generation_trajectory.float16.npy"
GENERATION_ROWS = OUT2363 / "material/generation_rows.jsonl"
GENERATION_RESULT = OUT2363 / "raw/generation_results.jsonl"
GENERATION_PROGRESS = OUT2363 / "raw/generation_progress.json"

FAMILIES = (
    "taxonomy", "attribute", "attitude", "grammar", "coreference", "translation",
    "causal", "temporal", "spatial", "possession", "partwhole", "negation",
)
LANGUAGES = ("en", "zh")
FACTORS = ("lexical_realization", "relation_variant", "branch_edge", "conflict_edge", "query_role")
UNITS = 8
CELLS = 32
GEN_STEPS = 10

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return dict(value)
    raise TypeError(type(value).__name__)


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def modules(model) -> list[Any]:
    embed = model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings()
    return [embed, *list(model.model.layers), model.model.norm]


def left_pad(sequences: list[list[int]], device: torch.device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        token_ids = torch.tensor(seq, dtype=torch.long, device=device)
        ids[i, -len(seq):] = token_ids
        mask[i, -len(seq):] = 1
    positions = (mask.cumsum(dim=1) - 1).clamp_min(0)
    return ids, mask, positions


def collect_boundary(model, device, rows: list[dict], batch_size: int = 12) -> dict:
    qmodules = modules(model)
    dimension = int(model.config.hidden_size)
    shape = (len(rows), len(qmodules), dimension)
    if STATES.exists() and DECISIONS.exists() and COLLECT_PROGRESS.exists():
        completed = int(json.loads(COLLECT_PROGRESS.read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(STATES, mode="r+")
        decisions = np.lib.format.open_memmap(DECISIONS, mode="r+")
    else:
        completed = 0
        STATES.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(STATES, mode="w+", dtype=np.float16, shape=shape)
        decisions = np.lib.format.open_memmap(DECISIONS, mode="w+", dtype=np.float32, shape=(len(rows), 5))
    capture: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(qmodules):
        def hook(_module, _inputs, value, qpoint=qpoint):
            tensor = value[0] if isinstance(value, tuple) else value
            capture[qpoint] = tensor[:, -1].detach()
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = left_pad([row["prompt_ids"] for row in batch], device, pad)
                capture.clear()
                output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                               use_cache=False, return_dict=True)
                for qpoint in range(len(qmodules)):
                    states[start:start + len(batch), qpoint] = capture[qpoint].float().cpu().numpy().astype(np.float16)
                logits = output.logits[:, -1].float()
                for local, row in enumerate(batch):
                    target = int(row["target_first_id"])
                    foil = int(row["foil_first_id"])
                    target_logit = float(logits[local, target])
                    foil_logit = float(logits[local, foil])
                    margin = target_logit - foil_logit
                    decisions[start + local] = [target_logit, foil_logit, margin, float(margin > 0),
                                                float(int(logits[local].argmax()) == target)]
                states.flush(); decisions.flush()
                save(COLLECT_PROGRESS, {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 384 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2359 boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        states.flush(); decisions.flush(); close(states); close(decisions)
    return {"shape": list(shape), "batch_size": batch_size, "dtype": "float16"}


def reference_token_indices(rows: list[dict]) -> list[int]:
    selected = []
    for family_index in range(len(FAMILIES)):
        for language_index in range(len(LANGUAGES)):
            unit = 6 + family_index % 2
            cell = 0 if unit == 6 else 31
            index = (((family_index * UNITS + unit) * 2 + language_index) * CELLS + cell)
            selected.append(index)
    return selected


def collect_all_token(model, tokenizer, device, rows: list[dict]) -> dict:
    selected_indices = reference_token_indices(rows)
    selected = [rows[index] for index in selected_indices]
    qmodules = modules(model)
    dimension = int(model.config.hidden_size)
    max_tokens = max(len(row["prompt_ids"]) for row in selected)
    shape = (len(selected), len(qmodules), max_tokens, dimension)
    if TOKEN_FIELD.exists() and TOKEN_PROGRESS.exists():
        completed = int(json.loads(TOKEN_PROGRESS.read_text(encoding="utf-8"))["completed"])
        field = np.lib.format.open_memmap(TOKEN_FIELD, mode="r+")
    else:
        completed = 0
        TOKEN_FIELD.parent.mkdir(parents=True, exist_ok=True)
        field = np.lib.format.open_memmap(TOKEN_FIELD, mode="w+", dtype=np.float16, shape=shape)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(qmodules):
        def hook(_module, _inputs, value, qpoint=qpoint):
            captures[qpoint] = (value[0] if isinstance(value, tuple) else value).detach()
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for local_index in range(completed, len(selected)):
                row = selected[local_index]
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                positions = torch.arange(ids.shape[1], device=device)[None, :]
                captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                token_count = ids.shape[1]
                for qpoint in range(len(qmodules)):
                    field[local_index, qpoint, :token_count] = captures[qpoint][0].float().cpu().numpy().astype(np.float16)
                field.flush()
                save(TOKEN_PROGRESS, {"completed": local_index + 1, "shape": shape})
                print(f"[phase2359 all-token] {local_index + 1}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush(); close(field)
    index_rows = []
    for source_index, row in zip(selected_indices, selected):
        index_rows.append({
            "source_index": source_index, "case_id": row["case_id"], "family": row["family"],
            "language": row["language"], "unit": row["unit"], "cell": row["cell"],
            "surface": row["surface"], "token_count": len(row["prompt_ids"]),
            "token_ids": row["prompt_ids"], "tokens": [tokenizer.decode([token]) for token in row["prompt_ids"]],
        })
    io.write_rows(TOKEN_INDEX, index_rows)
    return {"shape": list(shape), "rows": len(selected), "valid_tokens": sum(r["token_count"] for r in index_rows)}


def behavior(rows: list[dict]) -> dict:
    decisions = np.load(DECISIONS, mmap_mode="r")
    correct = np.asarray(decisions[:, 3], dtype=np.float32)
    cells = {}
    minima = {}
    for family in FAMILIES:
        family_values = []
        for language in LANGUAGES:
            for query in ("first", "terminal"):
                indices = [i for i, row in enumerate(rows) if row["family"] == family and row["language"] == language
                           and row["query"] == query]
                value = float(correct[indices].mean())
                cells[f"{family}:{language}:{query}"] = value
                family_values.append(value)
        minima[family] = min(family_values)
    result = {
        "first_token_target_over_foil": float(correct.mean()),
        "argmax_is_target_first_token": float(np.asarray(decisions[:, 4]).mean()),
        "family_minimum_cells": minima, "qualified_at_0_75": [family for family, value in minima.items() if value >= 0.75],
        "cell_values": cells,
        "warning": "A target-over-one-foil first-token gate is material qualification, not proof of knowledge.",
    }
    close(decisions)
    return result


def factorial_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signs = np.empty((CELLS, CELLS), dtype=np.float32)
    for cell in range(CELLS):
        for subset in range(CELLS):
            signs[cell, subset] = -1.0 if ((cell & subset).bit_count() % 2) else 1.0
    orders = np.asarray([subset.bit_count() for subset in range(CELLS)], dtype=np.int64)
    return signs, signs.T / CELLS, orders


def build_coefficients(shape: list[int]) -> dict:
    source = np.load(STATES, mmap_mode="r")
    groups = len(FAMILIES) * UNITS * len(LANGUAGES)
    qpoints, dimension = source.shape[1], source.shape[2]
    coefficient_shape = (groups, CELLS, qpoints, dimension)
    completed = 0
    if COEFF.exists() and COEFF_PROGRESS.exists():
        completed = int(json.loads(COEFF_PROGRESS.read_text(encoding="utf-8"))["completed"])
        output = np.lib.format.open_memmap(COEFF, mode="r+")
    else:
        COEFF.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(COEFF, mode="w+", dtype=np.float16, shape=coefficient_shape)
    _, transform, _ = factorial_matrices()
    cube = source.reshape(groups, CELLS, qpoints, dimension)
    for group in range(completed, groups):
        values = np.asarray(cube[group], dtype=np.float32)
        output[group] = np.einsum("sc,cqd->sqd", transform, values, optimize=True).astype(np.float16)
        output.flush()
        save(COEFF_PROGRESS, {"completed": group + 1, "shape": coefficient_shape})
        if (group + 1) % 24 == 0 or group + 1 == groups:
            print(f"[phase2360 coefficients] {group + 1}/{groups}", flush=True)
    output.flush(); close(output); close(source)
    return {"shape": list(coefficient_shape), "dtype": "float16", "normalization": "1/32 Walsh-Mobius coefficients"}


def group_index(family: int, unit: int, language: int) -> int:
    return (family * UNITS + unit) * len(LANGUAGES) + language


def group_record(group: int) -> dict:
    language = group % 2
    value = group // 2
    unit = value % UNITS
    family = value // UNITS
    return {"group": group, "family_index": family, "family": FAMILIES[family],
            "unit": unit, "language_index": language, "language": LANGUAGES[language]}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float64, copy=False).reshape(-1)
    bv = b.astype(np.float64, copy=False).reshape(-1)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / max(denom, 1e-12))


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def factorial_analysis() -> dict:
    coeff = np.load(COEFF, mmap_mode="r")
    states = np.load(STATES, mmap_mode="r")
    signs, _, orders = factorial_matrices()
    groups, _, qpoints, dimension = coeff.shape
    confirmation = [group_index(family, unit, language) for family in (8, 9)
                    for unit in (4, 5) for language in range(2)]
    energy = np.zeros((qpoints, CELLS), dtype=np.float64)
    confirmation_energy = np.zeros_like(energy)
    for qpoint in range(qpoints):
        values = np.asarray(coeff[:, :, qpoint], dtype=np.float32)
        energy[qpoint] = np.mean(values * values, axis=(0, 2))
        cv = values[confirmation]
        confirmation_energy[qpoint] = np.mean(cv * cv, axis=(0, 2))
    order_energy = np.stack([energy[:, orders == order].sum(axis=1) for order in range(6)], axis=1)
    confirm_order = np.stack([confirmation_energy[:, orders == order].sum(axis=1) for order in range(6)], axis=1)
    interaction_fraction = (confirm_order[:, 2] + confirm_order[:, 3]) / np.maximum(confirm_order[:, 1:].sum(axis=1), 1e-20)
    selected_q = int(np.argmax(np.where(np.arange(qpoints) == 0, -np.inf, interaction_fraction)))

    language_cosines = []
    subset_mask = (orders >= 1) & (orders <= 3)
    for family in range(len(FAMILIES)):
        for unit in range(UNITS):
            en = np.asarray(coeff[group_index(family, unit, 0), subset_mask, selected_q], dtype=np.float32)
            zh = np.asarray(coeff[group_index(family, unit, 1), subset_mask, selected_q], dtype=np.float32)
            language_cosines.append({"family": FAMILIES[family], "unit": unit, "cosine": cosine(en, zh)})

    # Conditional sign information is computed within every complete 32-cell cube and then averaged.
    mi = np.zeros((len(FACTORS), dimension), dtype=np.float64)
    cube = states.reshape(groups, CELLS, qpoints, dimension)
    for group in range(groups):
        h = np.asarray(cube[group, :, selected_q], dtype=np.float32)
        z = h > np.median(h, axis=0, keepdims=True)
        hz = binary_entropy(z.mean(axis=0))
        for factor in range(len(FACTORS)):
            mask0 = np.asarray([(cell >> factor) & 1 == 0 for cell in range(CELLS)])
            mask1 = ~mask0
            mi[factor] += hz - 0.5 * binary_entropy(z[mask0].mean(axis=0)) - 0.5 * binary_entropy(z[mask1].mean(axis=0))
    mi = (mi / groups).astype(np.float32)
    CMI.parent.mkdir(parents=True, exist_ok=True); np.save(CMI, mi)

    # Full coordinate loadings from a family-mean interaction tensor; every loading coordinate is retained.
    family_mean = np.asarray(coeff[:, 1:, selected_q], dtype=np.float32).reshape(len(FAMILIES), UNITS, 2, 31, dimension).mean(axis=(1, 2))
    matrix = family_mean.reshape(len(FAMILIES) * 31, dimension)
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    TENSOR_LOADINGS.parent.mkdir(parents=True, exist_ok=True); np.save(TENSOR_LOADINGS, vt.astype(np.float32))

    result = {
        "selected_qpoint": selected_q, "selection_rule": "maximum confirmation (order2+order3)/nonzero-order energy",
        "selected_interaction_fraction": float(interaction_fraction[selected_q]),
        "order_energy_by_qpoint": order_energy.tolist(),
        "selected_order_energy_fraction": (order_energy[selected_q] / max(order_energy[selected_q, 1:].sum(), 1e-20)).tolist(),
        "cross_language_interaction_cosine": {
            "mean": float(np.mean([row["cosine"] for row in language_cosines])),
            "minimum": float(np.min([row["cosine"] for row in language_cosines])), "rows": language_cosines,
        },
        "conditional_sign_information": {"shape": list(mi.shape), "mean_by_factor": mi.mean(axis=1).tolist()},
        "tensor_diagnostic": {
            "matrix_shape": list(matrix.shape), "loadings_shape": list(vt.shape),
            "singular_values": singular_values.tolist(),
            "warning": "SVD is a candidate-description diagnostic; it is not the primary representation and all loadings map back to exact coordinates.",
        },
        "finite_factor_calculus": {
            "order1": "finite Boolean-factor Jacobian analogue", "order2": "finite Boolean-factor Hessian analogue",
            "order3": "three-way Mobius/Walsh interaction", "warning": "These are discrete factorial responses, not continuous derivatives.",
        },
    }
    close(coeff); close(states)
    return result


def dynamics_analysis(selected_q: int) -> dict:
    coeff = np.load(COEFF, mmap_mode="r")
    _, _, orders = factorial_matrices()
    subset_mask = (orders >= 1) & (orders <= 3)
    values = np.asarray(coeff[:, subset_mask], dtype=np.float32)
    layer_delta = np.diff(values, axis=2)
    delta_energy = np.mean(layer_delta * layer_delta, axis=(0, 1, 3))
    peak_transition = int(np.argmax(delta_energy))

    # A full 2560 x 2560 physical-coordinate cooperation matrix, not a Top-K graph.
    feature = np.asarray(coeff[:, (orders == 2) | (orders == 3), selected_q], dtype=np.float32).reshape(-1, coeff.shape[-1])
    feature -= feature.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(feature * feature, axis=0, keepdims=True) + 1e-12)
    feature /= scale
    if torch.cuda.is_available():
        x = torch.from_numpy(feature).to("cuda", dtype=torch.float16)
        cooperation = (x.T @ x).float().cpu().numpy() / max(feature.shape[0] - 1, 1)
        del x
        torch.cuda.empty_cache()
    else:
        cooperation = (feature.T @ feature) / max(feature.shape[0] - 1, 1)
    cooperation = np.clip(cooperation, -1, 1).astype(np.float32)
    COOP.parent.mkdir(parents=True, exist_ok=True); np.save(COOP, cooperation)

    token_field = np.load(TOKEN_FIELD, mmap_mode="r")
    token_rows = io.read_rows(TOKEN_INDEX)
    delta_norms = []
    token_delta_vectors = []
    token_delta_meta = []
    for index, row in enumerate(token_rows):
        count = int(row["token_count"])
        token_values = np.asarray(token_field[index, selected_q, :count], dtype=np.float32)
        differences = np.diff(token_values, axis=0)
        delta_norms.extend(np.linalg.norm(differences, axis=1).tolist())
        token_delta_vectors.append(differences.astype(np.float32))
        token_delta_meta.extend({"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                                 "unit": row["unit"], "from_token": position, "to_token": position + 1,
                                 "qpoint": selected_q, "field": "token_increment"}
                                for position in range(count - 1))
    token_delta_path = OUT2361 / "derived/token_increment_selected_q.float32.npy"
    token_delta_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(token_delta_path, np.concatenate(token_delta_vectors, axis=0))
    save(OUT2361 / "index/token_increment_rows.json", token_delta_meta)
    result = {
        "selected_qpoint": selected_q, "layer_transition_energy": delta_energy.tolist(),
        "peak_q_to_qplus1": [peak_transition, peak_transition + 1],
        "coordinate_cooperation": {
            "shape": list(cooperation.shape), "diagonal_mean": float(np.diag(cooperation).mean()),
            "off_diagonal_abs_mean": float((np.abs(cooperation).sum() - np.abs(np.diag(cooperation)).sum()) /
                                              (cooperation.size - cooperation.shape[0])),
            "warning": "Pairwise response correlation is a coordinate-cooperation graph, not proof of a causal hyperedge.",
        },
        "token_increment": {"rows": len(token_delta_meta), "mean_norm": float(np.mean(delta_norms)),
                            "median_norm": float(np.median(delta_norms)), "full_coordinates_retained": True},
        "transport_diagnostic": {
            "cross_language_mean_cosine": None,
            "warning": "No sheaf/groupoid is asserted; transport is tested only by paired restrictions and held-out prediction in Phase2362.",
        },
    }
    close(coeff); close(token_field)
    return result


def template_for(coeff: np.ndarray, evaluation: str, family: int, language: int) -> np.ndarray:
    if evaluation == "unseen_unit":
        groups = [group_index(family, unit, language) for unit in range(4)]
    elif evaluation == "unseen_language":
        groups = [group_index(family, unit, 0) for unit in range(4)]
    elif evaluation == "whole_family":
        groups = [group_index(f, unit, lang) for f in range(8) for unit in range(4) for lang in range(2)]
    else:
        raise ValueError(evaluation)
    return np.asarray(coeff[groups], dtype=np.float32).mean(axis=0)


def prediction_analysis(model) -> dict:
    coeff = np.load(COEFF, mmap_mode="r")
    states = np.load(STATES, mmap_mode="r")
    signs, _, orders = factorial_matrices()
    qpoint = int(json.loads((OUT2360 / "analysis/final.json").read_text(encoding="utf-8"))["analysis"]["selected_qpoint"])
    qpoints = states.shape[1]
    dimension = states.shape[2]
    cube = states.reshape(-1, CELLS, qpoints, dimension)
    rng = np.random.default_rng(2362)
    permutation = rng.permutation(dimension)
    evaluation_targets = {
        "unseen_unit": [(f, u, lang) for f in range(8) for u in (6, 7) for lang in range(2)],
        "unseen_language": [(f, u, 1) for f in range(8) for u in (6, 7)],
        "whole_family": [(f, u, lang) for f in (10, 11) for u in (6, 7) for lang in range(2)],
    }
    metrics = {}
    improvement_rows = []
    improvement_matrix = []
    order_limits = (0, 1, 2, 3, 5)
    for evaluation, targets in evaluation_targets.items():
        sse = {limit: 0.0 for limit in order_limits}
        sse_sorted = 0.0
        sse_permuted = 0.0
        baseline_sse = 0.0
        coordinate_error_1 = np.zeros(dimension, dtype=np.float64)
        coordinate_error_3 = np.zeros(dimension, dtype=np.float64)
        observations = 0
        for family, unit, language in targets:
            group = group_index(family, unit, language)
            actual = np.asarray(cube[group, :, qpoint], dtype=np.float32)
            base = actual[0]
            template = template_for(coeff[:, :, qpoint], evaluation, family, language)
            truth = actual[1:]
            baseline = np.repeat(base[None, :], CELLS - 1, axis=0)
            baseline_sse += float(np.square(truth - baseline).sum())
            predictions = {}
            for limit in order_limits:
                selected = np.where((orders >= 1) & (orders <= limit))[0]
                if len(selected) == 0:
                    predicted = baseline
                else:
                    delta = (signs[1:, selected] - signs[0, selected]) @ template[selected]
                    predicted = base[None, :] + delta
                predictions[limit] = predicted
                sse[limit] += float(np.square(truth - predicted).sum())
            selected3 = np.where((orders >= 1) & (orders <= 3))[0]
            sorted_template = np.sort(template[selected3], axis=1)
            permuted_template = template[selected3][:, permutation]
            predicted_sorted = base[None, :] + (signs[1:, selected3] - signs[0, selected3]) @ sorted_template
            predicted_permuted = base[None, :] + (signs[1:, selected3] - signs[0, selected3]) @ permuted_template
            sse_sorted += float(np.square(truth - predicted_sorted).sum())
            sse_permuted += float(np.square(truth - predicted_permuted).sum())
            coordinate_error_1 += np.square(truth - predictions[1]).sum(axis=0)
            coordinate_error_3 += np.square(truth - predictions[3]).sum(axis=0)
            observations += truth.shape[0]
        r2 = {f"order_{limit}": 1.0 - value / max(baseline_sse, 1e-20) for limit, value in sse.items()}
        r2["order3_sorted_coordinate_control"] = 1.0 - sse_sorted / max(baseline_sse, 1e-20)
        r2["order3_permuted_coordinate_control"] = 1.0 - sse_permuted / max(baseline_sse, 1e-20)
        n = observations * dimension
        mdl = {}
        for limit, value in sse.items():
            parameters = int(np.sum((orders >= 1) & (orders <= limit))) * dimension
            mdl[f"order_{limit}"] = float(n * math.log(max(value / n, 1e-30)) + parameters * math.log(max(n, 2)))
        improvement_matrix.append(((coordinate_error_1 - coordinate_error_3) / max(observations, 1)).astype(np.float32))
        improvement_rows.append({"evaluation": evaluation, "meaning": "positive means order<=3 reduces coordinate MSE versus additive"})
        metrics[evaluation] = {"target_groups": len(targets), "predicted_cells_per_group": 31,
                               "normalized_r2": r2, "mdl_gaussian_parameter_penalty": mdl}
    improvement = np.stack(improvement_matrix)
    PRED_IMPROVEMENT.parent.mkdir(parents=True, exist_ok=True); np.save(PRED_IMPROVEMENT, improvement)
    save(OUT2362 / "index/prediction_improvement_rows.json", improvement_rows)

    # Full future-vocabulary diagnostic on 64 frozen fresh-unit cases, using predicted final-norm states.
    future = []
    final_q = qpoints - 1
    rows = io.read_rows(MATERIAL)
    for family in range(8):
        for unit in (6, 7):
            for language in range(2):
                group = group_index(family, unit, language)
                template = template_for(coeff[:, :, final_q], "unseen_unit", family, language)
                selected3 = np.where((orders >= 1) & (orders <= 3))[0]
                for cell in (1, 16):
                    actual = np.asarray(cube[group, cell, final_q], dtype=np.float32)
                    base = np.asarray(cube[group, 0, final_q], dtype=np.float32)
                    predicted = base + (signs[cell, selected3] - signs[0, selected3]) @ template[selected3]
                    row = rows[group * CELLS + cell]
                    with torch.inference_mode():
                        ah = torch.tensor(actual, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
                        ph = torch.tensor(predicted, device=ah.device, dtype=ah.dtype)
                        al = model.lm_head(ah).float(); pl = model.lm_head(ph).float()
                        log_a = torch.log_softmax(al, dim=-1); log_p = torch.log_softmax(pl, dim=-1)
                        pa = log_a.exp()
                        kl = float((pa * (log_a - log_p)).sum())
                        target, foil = int(row["target_first_id"]), int(row["foil_first_id"])
                        future.append({
                            "case_id": row["case_id"], "kl_actual_to_predicted": kl,
                            "top1_agreement": int(al.argmax()) == int(pl.argmax()),
                            "target_over_foil_actual": float(al[target] - al[foil]) > 0,
                            "target_over_foil_predicted": float(pl[target] - pl[foil]) > 0,
                        })
    gate = {
        "higher_order_improves_additive_unseen_unit": metrics["unseen_unit"]["normalized_r2"]["order_3"] > metrics["unseen_unit"]["normalized_r2"]["order_1"],
        "physical_beats_sorted_unseen_unit": metrics["unseen_unit"]["normalized_r2"]["order_3"] > metrics["unseen_unit"]["normalized_r2"]["order3_sorted_coordinate_control"],
        "physical_beats_permuted_unseen_unit": metrics["unseen_unit"]["normalized_r2"]["order_3"] > metrics["unseen_unit"]["normalized_r2"]["order3_permuted_coordinate_control"],
        "positive_whole_family_r2": metrics["whole_family"]["normalized_r2"]["order_3"] > 0,
    }
    result = {
        "selected_qpoint": qpoint, "evaluations": metrics, "future_vocabulary": {
            "rows": len(future), "mean_kl": float(np.mean([row["kl_actual_to_predicted"] for row in future])),
            "top1_agreement": float(np.mean([row["top1_agreement"] for row in future])),
            "target_foil_sign_agreement": float(np.mean([row["target_over_foil_actual"] == row["target_over_foil_predicted"] for row in future])),
            "full_vocabulary_used": True,
        },
        "gate": gate, "mechanism_candidate_passed": all(gate.values()),
        "warning": "Prediction from one reference cell is structural evidence; it is not causal closure or an exact decoder.",
    }
    close(coeff); close(states)
    return result


def generation_selection(rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if row["unit"] in (6, 7) and row["cell"] in (0, 8, 16, 24)]
    io.write_rows(GENERATION_ROWS, selected)
    return selected


def generate_trajectory(model, tokenizer, device, rows: list[dict], batch_size: int = 4) -> dict:
    selected = generation_selection(rows)
    qmodules = modules(model)
    shape = (len(selected), GEN_STEPS, len(qmodules), int(model.config.hidden_size))
    if TRAJECTORY.exists() and GENERATION_PROGRESS.exists():
        completed = int(json.loads(GENERATION_PROGRESS.read_text(encoding="utf-8"))["completed"])
        trajectory = np.lib.format.open_memmap(TRAJECTORY, mode="r+")
        records = io.read_rows(GENERATION_RESULT) if GENERATION_RESULT.exists() else []
    else:
        completed = 0; records = []
        TRAJECTORY.parent.mkdir(parents=True, exist_ok=True)
        trajectory = np.lib.format.open_memmap(TRAJECTORY, mode="w+", dtype=np.float16, shape=shape)
    capture: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(qmodules):
        def hook(_module, _inputs, value, qpoint=qpoint):
            capture[qpoint] = (value[0] if isinstance(value, tuple) else value)[:, -1].detach()
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    eos = int(model.config.eos_token_id or tokenizer.eos_token_id)
    try:
        with torch.inference_mode():
            for start in range(completed, len(selected), batch_size):
                batch = selected[start:start + batch_size]
                ids, mask, positions = left_pad([row["prompt_ids"] for row in batch], device, pad)
                current = ids; past = None; generated = []
                for step in range(GEN_STEPS):
                    capture.clear()
                    output = model(input_ids=current, attention_mask=mask,
                                   position_ids=positions if past is None else None,
                                   past_key_values=past, use_cache=True, return_dict=True)
                    for qpoint in range(len(qmodules)):
                        trajectory[start:start + len(batch), step, qpoint] = capture[qpoint].float().cpu().numpy().astype(np.float16)
                    token = output.logits[:, -1].argmax(dim=-1)
                    generated.append(token)
                    past = output.past_key_values
                    current = token[:, None]
                    mask = torch.cat([mask, torch.ones((len(batch), 1), dtype=mask.dtype, device=device)], dim=1)
                token_matrix = torch.stack(generated, dim=1).cpu().tolist()
                for row, token_ids in zip(batch, token_matrix):
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    normalized = " ".join(text.split()).lower()
                    target = " ".join(row["target"].split()).lower()
                    target_tokens = row["target_ids"]
                    divergence = next((i for i, (a, b) in enumerate(zip(token_ids, target_tokens)) if a != b),
                                      min(len(token_ids), len(target_tokens)))
                    records.append({
                        "case_id": row["case_id"], "family": row["family"], "language": row["language"],
                        "unit": row["unit"], "cell": row["cell"], "query": row["query"], "target": row["target"],
                        "generated": text, "token_ids": token_ids, "semantic_prefix_exact": normalized.startswith(target),
                        "first_line_exact": " ".join(text.splitlines()[0].split()).lower() == target if text else False,
                        "stop_token_seen": eos in token_ids, "first_divergence_step": divergence,
                    })
                io.write_rows(GENERATION_RESULT, records)
                trajectory.flush(); save(GENERATION_PROGRESS, {"completed": start + len(batch), "shape": shape})
                print(f"[phase2363 generation] {start + len(batch)}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        trajectory.flush(); close(trajectory)
    return {"shape": list(shape), "rows": len(selected), "steps": GEN_STEPS}


def generation_analysis(rows: list[dict]) -> dict:
    records = io.read_rows(GENERATION_RESULT)
    by_cell = {}
    for language in LANGUAGES:
        for query in ("first", "terminal"):
            subset = [row for row in records if row["language"] == language and row["query"] == query]
            success = [bool(row["semantic_prefix_exact"]) for row in subset]
            positives, negatives = sum(success), len(success) - sum(success)
            by_cell[f"{language}:{query}"] = {"rows": len(subset), "success": float(np.mean(success)),
                                                        "positives": positives, "negatives": negatives,
                                                        "matched_rows": 2 * min(positives, negatives)}
    matched = []
    for key in by_cell:
        language, query = key.split(":")
        success_rows = [row for row in records if row["language"] == language and row["query"] == query
                        and row["semantic_prefix_exact"]]
        failure_rows = [row for row in records if row["language"] == language and row["query"] == query
                        and not row["semantic_prefix_exact"]]
        count = min(len(success_rows), len(failure_rows))
        matched.extend(success_rows[:count] + failure_rows[:count])
    return {
        "rows": len(records), "semantic_prefix_exact": float(np.mean([row["semantic_prefix_exact"] for row in records])),
        "first_line_exact": float(np.mean([row["first_line_exact"] for row in records])),
        "stop_token_seen": float(np.mean([row["stop_token_seen"] for row in records])),
        "query_language_cells": by_cell, "outcome_matched_rows": len(matched),
        "first_divergence": {
            "mean": float(np.mean([row["first_divergence_step"] for row in records])),
            "median": float(np.median([row["first_divergence_step"] for row in records])),
            "matched_mean": None if not matched else float(np.mean([row["first_divergence_step"] for row in matched])),
        },
        "warning": "Outcome matching is post-hoc descriptive. Step0 is prompt preparation; only step>=1 contains model-generated history.",
    }


def publish_array(dataset_id: str, title: str, source: np.ndarray | Path, metadata: list[dict], model: str,
                  schema: str, claim: str, boundary: str, semantics: str, phase: int, campaign: str,
                  dtype: np.dtype | None = None, extra: dict | None = None) -> dict:
    if isinstance(source, Path):
        values = np.load(source, mmap_mode="r")
    else:
        values = source
    array = values.reshape(-1, values.shape[-1])
    out_dtype = np.dtype(dtype or values.dtype)
    binary = VIS / f"{dataset_id}.{out_dtype.name}.npy"
    output = atlas.create_binary(binary.name, array.shape[0], array.shape[1], out_dtype)
    output[:] = array.astype(out_dtype, copy=False)
    output.flush(); close(output)
    if isinstance(source, Path):
        close(values)
    payload = {"phase": phase, "campaign": campaign, "no_topk": True, "activation_not_parameter": True}
    if extra:
        payload.update(extra)
    return atlas.write_metadata(dataset_id, title, binary, metadata, model, schema, claim, boundary, semantics, payload)


def publish_all(rows: list[dict], qanalysis: dict, dynamics: dict, prediction: dict, generation_info: dict) -> list[dict]:
    assets = []
    selected_q = int(qanalysis["selected_qpoint"])
    states = np.load(STATES, mmap_mode="r")
    qpoints = [0, selected_q, states.shape[1] - 1]
    raw = np.stack([np.asarray(states[:, q], dtype=np.float16) for q in qpoints], axis=1)
    raw_meta = [{"case_id": row["case_id"], "family": row["family"], "category": row["category"],
                 "language": row["language"], "surface": row["surface"], "unit": row["unit"], "cell": row["cell"],
                 "bits": row["bits"], "query": row["query"], "qpoint": qpoint,
                 "checkpoint": "embedding" if qpoint == 0 else ("final_norm" if qpoint == states.shape[1] - 1 else "block_post")}
                for row in rows for qpoint in qpoints]
    assets.append(publish_array("c10321_qwen4b_hypergraph_boundary_field", "Qwen3-4B typed-hypergraph boundary field",
                                raw, raw_meta, "Qwen3-4B-FP16", "typed_hypergraph_boundary_field_v1",
                                "observational full-coordinate field", "6144 five-factor bilingual prompts",
                                "raw embedding and HiddenState activation in physical coordinate order", 2359, "C10321-C10560"))
    close(states)

    coeff = np.load(COEFF, mmap_mode="r")
    _, _, orders = factorial_matrices()
    for order, dataset_id in ((1, "c10561_qwen4b_factorial_first_order"),
                              (2, "c10562_qwen4b_factorial_second_order"),
                              (3, "c10563_qwen4b_factorial_third_order")):
        subsets = np.where(orders == order)[0]
        values = np.asarray(coeff[:, subsets, selected_q], dtype=np.float16)
        metadata = []
        for group in range(coeff.shape[0]):
            record = group_record(group)
            for subset in subsets:
                metadata.append({**record, "qpoint": selected_q, "subset": int(subset), "factor_members":
                                 [FACTORS[k] for k in range(len(FACTORS)) if (subset >> k) & 1], "order": order})
        assets.append(publish_array(dataset_id, f"Qwen3-4B order-{order} factorial exact-coordinate field", values, metadata,
                                    "Qwen3-4B-FP16", f"boolean_factorial_order{order}_v1",
                                    "paired observational interaction", "complete 32-cell cubes",
                                    "signed Walsh-Mobius coefficient for every physical activation coordinate", 2360,
                                    "C10561-C10800"))
    close(coeff)
    mi = np.load(CMI)
    assets.append(publish_array("c10564_qwen4b_conditional_sign_information", "Qwen3-4B conditional sign information",
                                mi, [{"factor": factor, "qpoint": selected_q} for factor in FACTORS], "Qwen3-4B-FP16",
                                "conditional_sign_information_v1", "discrete information diagnostic",
                                "within complete cubes then averaged across groups",
                                "conditional mutual information in bits for every coordinate", 2360, "C10561-C10800", np.float32))
    loadings = np.load(TENSOR_LOADINGS)
    assets.append(publish_array("c10565_qwen4b_tensor_coordinate_loadings", "Qwen3-4B family-mean tensor coordinate loadings",
                                loadings, [{"component": i, "qpoint": selected_q} for i in range(loadings.shape[0])],
                                "Qwen3-4B-FP16", "tensor_coordinate_loadings_v1", "candidate-description diagnostic",
                                "family-mean order1-5 coefficient matrix", "full right-singular loading mapped to every coordinate",
                                2360, "C10561-C10800", np.float32))

    token_field = np.load(TOKEN_FIELD, mmap_mode="r")
    token_rows = io.read_rows(TOKEN_INDEX)
    valid_values = []
    valid_meta = []
    for index, row in enumerate(token_rows):
        count = int(row["token_count"])
        valid_values.append(np.asarray(token_field[index, :, :count], dtype=np.float16).reshape(-1, token_field.shape[-1]))
        for qpoint in range(token_field.shape[1]):
            for position in range(count):
                valid_meta.append({"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                                   "unit": row["unit"], "surface": row["surface"], "qpoint": qpoint,
                                   "token_position": position, "token_id": row["token_ids"][position],
                                   "token_text": row["tokens"][position]})
    valid = np.concatenate(valid_values, axis=0)
    assets.append(publish_array("c10801_qwen4b_reference_all_token_field", "Qwen3-4B bilingual all-token all-layer field",
                                valid, valid_meta, "Qwen3-4B-FP16", "typed_hypergraph_all_token_field_v1",
                                "observational full-coordinate token field", "24 reference prompts spanning all families and languages",
                                "raw embedding/HiddenState at every valid token, checkpoint and physical coordinate", 2361,
                                "C10801-C11040", np.float16))
    close(token_field)
    token_delta = OUT2361 / "derived/token_increment_selected_q.float32.npy"
    delta_meta = json.loads((OUT2361 / "index/token_increment_rows.json").read_text(encoding="utf-8"))
    assets.append(publish_array("c10802_qwen4b_token_increment_field", "Qwen3-4B selected-layer token increment field",
                                token_delta, delta_meta, "Qwen3-4B-FP16", "token_increment_full_coordinate_v1",
                                "observational token dynamics", "successive valid prompt tokens",
                                "signed H[t+1]-H[t] in every physical coordinate", 2361, "C10801-C11040", np.float32))
    cooperation = np.load(COOP)
    assets.append(publish_array("c10803_qwen4b_coordinate_cooperation_matrix", "Qwen3-4B full coordinate cooperation matrix",
                                cooperation, [{"source_coordinate": i, "qpoint": selected_q} for i in range(cooperation.shape[0])],
                                "Qwen3-4B-FP16", "coordinate_cooperation_matrix_v1", "observational response graph",
                                "order2+3 interaction responses across every group",
                                "correlation from each exact source coordinate to every exact target coordinate", 2361,
                                "C10801-C11040", np.float32))
    improvement = np.load(PRED_IMPROVEMENT)
    improvement_meta = json.loads((OUT2362 / "index/prediction_improvement_rows.json").read_text(encoding="utf-8"))
    assets.append(publish_array("c11041_qwen4b_composition_prediction_improvement", "Qwen3-4B higher-order prediction improvement",
                                improvement, improvement_meta, "Qwen3-4B-FP16", "composition_prediction_improvement_v1",
                                "held-out predictive structure", "31 unseen cells predicted from cell0",
                                "per-coordinate additive-error minus order<=3 error", 2362, "C11041-C11280", np.float32,
                                {"mechanism_candidate_passed": prediction["mechanism_candidate_passed"]}))

    trajectory = np.load(TRAJECTORY, mmap_mode="r")
    generation_rows = io.read_rows(GENERATION_RESULT)
    trajectory_meta = [{"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                        "unit": row["unit"], "cell": row["cell"], "query": row["query"], "step": step,
                        "generated_token_id": row["token_ids"][step], "qpoint": qpoint,
                        "history_kind": "prompt_preparation" if step == 0 else "model_generated_history"}
                       for row in generation_rows for step in range(GEN_STEPS) for qpoint in range(trajectory.shape[2])]
    assets.append(publish_array("c11281_qwen4b_balanced_generation_trajectory", "Qwen3-4B balanced-query generation trajectory",
                                TRAJECTORY, trajectory_meta, "Qwen3-4B-FP16", "balanced_generation_trajectory_v1",
                                "observational autonomous trajectory", "192 fresh-unit bilingual balanced-query prompts",
                                "embedding and every HiddenState coordinate at prompt step0 and generated-history steps1-9",
                                2363, "C11281-C11520", np.float16, {"shape4d": generation_info["shape"]}))
    close(trajectory)
    return assets


def append_memo(phase: int, title: str, campaign: str, body: str) -> None:
    if f"## Phase {phase}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n\n## Phase {phase}: {title}（{campaign}） [{stamp}]\n\n{body}\n")


def append_all_memos(results: dict) -> None:
    r59, r60, r61, r62, r63 = (results[str(p)] for p in range(2359, 2364))
    append_memo(2359, "十二族双语有类型超图Qwen3-4B全层全坐标场", "C10321-C10560", rf"""
**测试原理与测试用例。** 对Phase2358冻结的6144条五因子完整立方体一次前向采集embedding、36个block和final norm的全部2560个激活坐标；以目标名称第一token相对冻结foil的logit margin作材料资格门，并另外保留24条跨全族/双语的全token全层场。

$$m_i=z_{{y_i}}-z_{{\tilde y_i}},\qquad H_i\in\mathbb R^{{38\times2560}}.$$

**结果汇总。** 采集 `{json.dumps(r59['collection'], ensure_ascii=False)}`；行为 `{json.dumps(r59['behavior'], ensure_ascii=False)}`；材料哈希 `{r59['material_sha256']}`；可视化`c10321`。

**相关文件。** 本脚本；结果 `tests/glm5/result/phase2359_c10321_c10560_qwen4b_hypergraph_full_field`。

**理论进展、问题硬伤与结论。** 这是全析因图谱的原场，不把target>foil称为“知道”。自然名称和显式图任务仍可能让模型依赖表面模板；后续只在锁箱预测和坐标控制通过时升级规律证据。
""")
    append_memo(2360, "全坐标布尔Möbius交互与数学路线总扫描", "C10561-C10800", rf"""
**测试原理与公式。** 对每个family×unit×language完整32格做Walsh–Möbius反演：

$$\widehat H_S=2^{{-5}}\sum_{{x\in\{{0,1\}}^5}}(-1)^{{\langle S,x\rangle}}H(x),\quad
H(x)=\sum_S(-1)^{{\langle S,x\rangle}}\widehat H_S.$$

同时计算一至三阶全坐标交互、离散符号条件信息、映射回全部坐标的SVD载荷；一阶/二阶只称有限因子Jacobian/Hessian类比，不宣称连续几何。

**结果汇总。** `{json.dumps(r60['analysis'], ensure_ascii=False)}`；可视化`c10561–c10565`。

**相关文件。** 本脚本；结果 `tests/glm5/result/phase2360_c10561_c10800_factorial_coordinate_route_scan`。

**理论进展、问题硬伤与结论。** 二三阶项是否“重要”要由Phase2362未见组合预测裁决，能量大本身不是机制。SVD、信息量和所谓超图社区均只作候选描述；没有引入流形、层丛或群胚实体。
""")
    append_memo(2361, "层增量、Token增量与2560×2560坐标协同图", "C10801-C11040", rf"""
**测试原理与公式。** 对全坐标场计算

$$\Delta_q\widehat H_S=\widehat H_{{S,q+1}}-\widehat H_{{S,q}},\qquad
\Delta_tH=H_{{t+1}}-H_t,\qquad C_{{jk}}=\mathrm{{corr}}(\widehat H_{{S,j}},\widehat H_{{S,k}}).$$

**结果汇总。** `{json.dumps(r61['analysis'], ensure_ascii=False)}`；全部token原场、token差分及完整2560×2560协同矩阵已发布为`c10801–c10803`。

**相关文件。** 本脚本；结果 `tests/glm5/result/phase2361_c10801_c11040_layer_token_coordinate_dynamics`。

**理论进展、问题硬伤与结论。** 坐标协同矩阵显示的是跨条件共同响应，不是因果连边；token差分不能自动等同“写入”。它把观察对象从单边界扩展到何时变化、哪些具体坐标协同变化。
""")
    append_memo(2362, "从单一基准格预测未见词汇、组合、语言与整族", "C11041-C11280", rf"""
**测试原理与公式。** 只给目标组cell0，以发现集平均交互预测其余31格：

$$\widehat H(x)=H_{{target}}(0)+\sum_{{1\leq|S|\leq k}}[\chi_S(x)-\chi_S(0)]\,\overline{{\widehat H_S}}_{{train}}.$$

比较加法、二阶、三阶、全阶、坐标排序和冻结置乱，并用全词表KL/Top1及target–foil符号检验未来token分布。

**结果汇总。** `{json.dumps(r62['analysis'], ensure_ascii=False)}`；逐坐标改善热力图`c11041`。

**相关文件。** 本脚本；结果 `tests/glm5/result/phase2362_c11041_c11280_composition_prediction_tournament`。

**理论进展、问题硬伤与结论。** 只有锁箱高阶优于加法且物理坐标优于排序/置乱才称候选组合规律；即便通过也只是L3预测证据，不等于内部超图同构或因果闭合。MDL惩罚防止用更多交互项无条件取胜。
""")
    append_memo(2363, "查询等量自主生成与首次分叉全坐标轨迹", "C11281-C11520", rf"""
**测试原理与公式。** 对fresh-unit锁箱中中英×first/terminal等量抽样192条，自主生成10 token；step0严格标为prompt准备，step≥1才含模型自身历史。每个query×language格只在事后等量抽取成功/失败作描述，不把它变成前瞻训练证据。

$$T_{{i,s,q}}=H_q(x_i,\hat y_{{i,<s}}),\qquad d_i=\min\{{s:\hat y_{{i,s}}\ne y_{{i,s}}\}}.$$

**结果汇总。** `{json.dumps(r63['generation'], ensure_ascii=False)}`；轨迹热力图`c11281`。

**相关文件。** 本脚本；结果 `tests/glm5/result/phase2363_c11281_c11520_balanced_generation_realization`。

**理论进展、问题硬伤与结论。** 首次分叉是时间定位指标，不证明分叉前为“知识”、分叉后为“生成函数”。停止token、语义前缀和首行精确率分开记录；本Phase继续积累动态拼图，不以前置删除/救援门中断研究。
""")


def main() -> None:
    final_paths = {phase: path / "analysis/final.json" for phase, path in (
        (2359, OUT2359), (2360, OUT2360), (2361, OUT2361), (2362, OUT2362), (2363, OUT2363))}
    if all(path.exists() for path in final_paths.values()):
        results = {str(phase): json.loads(path.read_text(encoding="utf-8")) for phase, path in final_paths.items()}
        append_all_memos(results); print(json.dumps(results, ensure_ascii=False, indent=2)); return
    rows = io.read_rows(MATERIAL)
    model = tokenizer = None
    try:
        model, tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        collection = collect_boundary(model, device, rows)
        token_collection = collect_all_token(model, tokenizer, device, rows)
        behavior_result = behavior(rows)
        result2359 = {
            "phase": 2359, "campaign": "C10321-C10560", "material_sha256": file_hash(MATERIAL),
            "collection": {"boundary": collection, "all_token": token_collection}, "behavior": behavior_result,
            "checks": {"rows": len(rows) == 6144, "shape": collection["shape"] == [6144, 38, 2560],
                       "all_token_rows": token_collection["rows"] == 24, "finite_behavior": math.isfinite(behavior_result["first_token_target_over_foil"])},
        }
        result2359["all_checks_passed"] = all(result2359["checks"].values())
        save(final_paths[2359], result2359)
        coefficient_info = build_coefficients(collection["shape"])
        qanalysis = factorial_analysis()
        result2360 = {"phase": 2360, "campaign": "C10561-C10800", "coefficients": coefficient_info,
                      "analysis": qanalysis, "checks": {"coeff_shape": coefficient_info["shape"] == [192, 32, 38, 2560],
                                                        "cmi": CMI.exists(), "loadings": TENSOR_LOADINGS.exists()}}
        result2360["all_checks_passed"] = all(result2360["checks"].values()); save(final_paths[2360], result2360)
        dynamics = dynamics_analysis(int(qanalysis["selected_qpoint"]))
        result2361 = {"phase": 2361, "campaign": "C10801-C11040", "analysis": dynamics,
                      "checks": {"cooperation_shape": dynamics["coordinate_cooperation"]["shape"] == [2560, 2560],
                                 "token_deltas": dynamics["token_increment"]["rows"] > 0}}
        result2361["all_checks_passed"] = all(result2361["checks"].values()); save(final_paths[2361], result2361)
        prediction = prediction_analysis(model)
        result2362 = {"phase": 2362, "campaign": "C11041-C11280", "analysis": prediction,
                      "checks": {"three_evaluations": len(prediction["evaluations"]) == 3,
                                 "full_vocab": prediction["future_vocabulary"]["full_vocabulary_used"],
                                 "improvement": PRED_IMPROVEMENT.exists()}}
        result2362["all_checks_passed"] = all(result2362["checks"].values()); save(final_paths[2362], result2362)
        generation_info = generate_trajectory(model, tokenizer, device, rows)
        generation_result = generation_analysis(rows)
        result2363 = {"phase": 2363, "campaign": "C11281-C11520", "collection": generation_info,
                      "generation": generation_result,
                      "checks": {"rows": generation_info["rows"] == 192, "shape": generation_info["shape"] == [192, 10, 38, 2560],
                                 "balanced_design": all(cell["rows"] == 48 for cell in generation_result["query_language_cells"].values())}}
        result2363["all_checks_passed"] = all(result2363["checks"].values()); save(final_paths[2363], result2363)
        assets = publish_all(rows, qanalysis, dynamics, prediction, generation_info)
        verification = [atlas.verify(asset) for asset in assets]
        verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
        catalog = atlas.update_catalog(assets)
        frontend = atlas.frontend_build()
        publication = {"datasets": json.loads(json.dumps(assets, default=str)), "verification": verification,
                       "verified": verified, "catalog": catalog, "frontend": frontend}
        save(OUT2363 / "analysis/publication.json", publication)
        if not verified or not frontend["passed"]:
            raise RuntimeError(("publication_failed", verification, frontend))
    finally:
        if model is not None:
            model_utils.release_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results = {str(phase): json.loads(path.read_text(encoding="utf-8")) for phase, path in final_paths.items()}
    append_all_memos(results)
    if not all(result["all_checks_passed"] for result in results.values()):
        raise RuntimeError({phase: result["checks"] for phase, result in results.items()})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
