#!/usr/bin/env python3
"""C885-C904 Qwen3-4B broad-family full-coordinate model tournament."""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = RESULT / "phase2235_c885_c904_qwen_broad_family_full_coordinate_tournament"
CONTRACT_OUT = RESULT / "phase2234_c870_c884_broad_family_conditional_gear_contract"
sys.path.insert(0, str(TESTS))

import phase2234_c870_c884_broad_family_gear_contract as contract


PHASE = 2235
CAMPAIGNS = tuple(f"C{i}" for i in range(885, 905))
RIDGE = 0.10
METHODS = (
    "M0_zero", "M1_shared_mean", "M2_shared_affine", "M3_family_mean",
    "M4_family_affine", "M5_family_guard", "M6_wrong_family_guard",
)
RAW_PARENT = OUT / "raw/parent/qualified_role_field.float16.npy"
RAW_FRESH = OUT / "raw/fresh/qualified_role_field.float16.npy"
RAW_SHARED = OUT / "raw/shared_affine_coefficients.float16.npy"
RAW_FAMILY_AFFINE = OUT / "raw/family_affine_residual_coefficients.float16.npy"
RAW_FAMILY_GUARD = OUT / "raw/family_guard_residual.float16.npy"


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    contract.write_rows(path, rows)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def file_hash(path: Path) -> str:
    return contract.file_hash(path)


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def release_model(model) -> None:
    contract.prior.release_model(model)
    gc.collect()


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def run_behavior(model, tokenizer, device, compiled: list[dict], prefix: str) -> tuple[list[dict], list[dict]]:
    candidate = contract.prior.behavior_base.batch_behavior(model, device, compiled, batch_size=20)
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    generated = []
    for start in range(0, len(compiled), 12):
        batch = compiled[start:start + 12]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False,
                pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
            )
        for i, row in enumerate(batch):
            text = tokenizer.decode(output[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_code(text, row)
            generated.append({
                "case_id": row["case_id"], "text": text, "parsed": parsed,
                "correct_answer": row["correct_answer"], "correct": parsed == row["correct_answer"],
            })
        if start % 120 == 0:
            print(f"[{prefix}] generation {start}/{len(compiled)}", flush=True)
    return candidate, generated


def slice_key(row: dict) -> str:
    return f"{row['panel']}|{row['family']}|{row['language']}"


def behavior_slices(compiled: list[dict], candidate: dict, generated: dict,
                    partitions: tuple[str, ...]) -> tuple[dict, set[str]]:
    panels, qualified = {}, set()
    keys = sorted({slice_key(row) for row in compiled})
    for key in keys:
        detail = {}
        for part in partitions:
            rows = [row for row in compiled if slice_key(row) == key and row["partition"] == part]
            detail[part] = {
                "rows": len(rows),
                "candidate_accuracy": float(np.mean([candidate[row["case_id"]]["correct"] for row in rows])) if rows else None,
                "generation_accuracy": float(np.mean([generated[row["case_id"]]["correct"] for row in rows])) if rows else None,
            }
        detail["qualified"] = all(
            detail[part][metric] is not None and detail[part][metric] >= contract.BEHAVIOR_GATE
            for part in partitions for metric in ("candidate_accuracy", "generation_accuracy")
        )
        panels[key] = detail
        if detail["qualified"]:
            qualified.add(key)
    return panels, qualified


def capture_field(model, tokenizer, device, compiled: list[dict], candidate: dict, generated: dict,
                  qualified: set[str], raw_dir: Path, prefix: str) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected = [row for row in compiled if slice_key(row) in qualified]
    index = []
    role_path = raw_dir / "qualified_role_field.float16.npy"
    field = np.lib.format.open_memmap(
        role_path, mode="w+", dtype=np.float16,
        shape=(len(selected), contract.CHECKPOINTS, len(contract.ROLES), contract.DIM),
    )
    representative_ids = set()
    for key in qualified:
        for part in sorted({row["partition"] for row in selected if slice_key(row) == key}):
            rows = [row for row in selected if slice_key(row) == key and row["partition"] == part]
            if rows:
                representative_ids.add(rows[0]["case_id"])
    representative = [row for row in selected if row["case_id"] in representative_ids]
    max_width = max([len(row["prompt_ids"]) for row in representative], default=1)
    token_path = raw_dir / "representative_full_token_qpoints.float16.npy"
    token_field = np.lib.format.open_memmap(
        token_path, mode="w+", dtype=np.float16,
        shape=(len(representative), len(contract.QPOINTS), max_width, contract.DIM),
    )
    token_map = {row["case_id"]: i for i, row in enumerate(representative)}
    qmap = {q: i for i, q in enumerate(contract.QPOINTS)}
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in modules]
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    try:
        for start in range(0, len(selected), 4):
            batch = selected[start:start + 4]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, :len(seq)] = 1
            pos = mask.long().cumsum(-1) - 1
            pos.masked_fill_(mask == 0, 0)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != contract.CHECKPOINTS:
                raise RuntimeError((prefix, "checkpoint_count", len(captured)))
            for local_i, row in enumerate(batch):
                hidden_i = start + local_i
                for q, hidden in enumerate(captured):
                    values = hidden[local_i].float().cpu().numpy().astype(np.float16)
                    for role_i, role in enumerate(contract.ROLES):
                        field[hidden_i, q, role_i] = values[row["role_positions"][role][-1]]
                    if row["case_id"] in token_map and q in qmap:
                        token_field[token_map[row["case_id"]], qmap[q], :len(row["prompt_ids"])] = values[:len(row["prompt_ids"])]
                index.append({
                    "hidden_index": hidden_i, "case_id": row["case_id"], "panel": row["panel"],
                    "family": row["family"], "language": row["language"], "surface": row["surface"],
                    "unit": row["unit"], "partition": row["partition"], "cell": row["cell"],
                    "cell_i": row["cell_i"], "factors": row["factors"], "truth": row["truth"],
                    "output_scheme": row["output_scheme"],
                    "candidate_correct": bool(candidate[row["case_id"]]["correct"]),
                    "generation_correct": bool(generated[row["case_id"]]["correct"]),
                    "candidate_scores": candidate[row["case_id"]].get("scores"),
                    "role_positions": row["role_positions"], "prompt_length": len(row["prompt_ids"]),
                })
            del captured[:]
            if start % 64 == 0:
                print(f"[{prefix}] field {start}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush(); token_field.flush(); close_mmap(field); close_mmap(token_field)
    write_rows(raw_dir / "hidden_index.jsonl", index)
    write_rows(raw_dir / "representative_token_index.jsonl", [
        {"token_index": token_map[row["case_id"]], "case_id": row["case_id"],
         "panel": row["panel"], "family": row["family"], "language": row["language"],
         "partition": row["partition"], "prompt_ids": row["prompt_ids"], "prompt_length": len(row["prompt_ids"])}
        for row in representative
    ])
    return {
        "selected_rows": len(selected), "qualified_slices": sorted(qualified),
        "role_path": str(role_path.relative_to(ROOT)),
        "role_shape": [len(selected), contract.CHECKPOINTS, len(contract.ROLES), contract.DIM],
        "token_path": str(token_path.relative_to(ROOT)),
        "token_shape": [len(representative), len(contract.QPOINTS), max_width, contract.DIM],
        "includes_incorrect_rows_inside_qualified_slices": True,
    }


def pair_records(index: list[dict], partition: str | None = None) -> list[dict]:
    rows = [row for row in index if row["panel"] == "broad_family" and (partition is None or row["partition"] == partition)]
    by_key = {(row["family"], row["language"], row["surface"], row["unit"], row["truth"]): row for row in rows}
    pairs = []
    for family, language, surface, unit in sorted({key[:4] for key in by_key}):
        left = by_key.get((family, language, surface, unit, False))
        right = by_key.get((family, language, surface, unit, True))
        if left is not None and right is not None:
            pairs.append({
                "family": family, "language": language, "surface": surface, "unit": unit,
                "partition": left["partition"], "base": left["hidden_index"], "changed": right["hidden_index"],
                "base_case_id": left["case_id"], "changed_case_id": right["case_id"],
            })
    return pairs


def arrays_for_pairs(field: np.ndarray, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_ids = np.asarray([row["base"] for row in pairs], dtype=np.int64)
    changed_ids = np.asarray([row["changed"] for row in pairs], dtype=np.int64)
    base = np.asarray(field[base_ids][:, contract.QPOINTS], dtype=np.float32)
    previous = np.asarray(field[base_ids][:, [max(0, q - 1) for q in contract.QPOINTS]], dtype=np.float32)
    changed = np.asarray(field[changed_ids][:, contract.QPOINTS], dtype=np.float32)
    return base, previous, changed - base


def shared_features(base: np.ndarray, previous: np.ndarray) -> list[np.ndarray]:
    boundary = np.broadcast_to(base[:, :, contract.ROLES.index("boundary")][:, :, None, :], base.shape)
    query = np.broadcast_to(base[:, :, contract.ROLES.index("query")][:, :, None, :], base.shape)
    relation = np.broadcast_to(base[:, :, contract.ROLES.index("relation")][:, :, None, :], base.shape)
    return [np.ones_like(base), base, previous, boundary, query, relation]


def family_features(base: np.ndarray) -> list[np.ndarray]:
    query = np.broadcast_to(base[:, :, contract.ROLES.index("query")][:, :, None, :], base.shape)
    relation = np.broadcast_to(base[:, :, contract.ROLES.index("relation")][:, :, None, :], base.shape)
    return [np.ones_like(base), base, query, relation]


def fit_coordinate_ridge(features: list[np.ndarray], target: np.ndarray, chunk: int = 1024) -> np.ndarray:
    p, n = len(features), target.shape[0]
    flat_y = target.reshape(n, -1)
    flat_x = [value.reshape(n, -1) for value in features]
    beta = np.empty((p, flat_y.shape[1]), dtype=np.float32)
    for start in range(0, flat_y.shape[1], chunk):
        end = min(start + chunk, flat_y.shape[1])
        # The frozen ridge is small relative to residual-stream magnitudes.  Form
        # the same normal equations in float64 so the regularizer is not rounded
        # away for high-energy coordinates.
        x = np.stack([value[:, start:end] for value in flat_x], axis=1).astype(np.float64)
        y = flat_y[:, start:end].astype(np.float64)
        gram = np.einsum("npk,nqk->kpq", x, x, optimize=True)
        rhs = np.einsum("npk,nk->kp", x, y, optimize=True)
        diag = np.arange(p); gram[:, diag, diag] += RIDGE; gram[:, 0, 0] -= RIDGE * 0.9
        beta[:, start:end] = np.linalg.solve(gram, rhs[..., None])[..., 0].T.astype(np.float32)
    return beta.reshape((p,) + target.shape[1:])


def existing_field_info(raw_dir: Path, qualified: set[str]) -> dict:
    role_path = raw_dir / "qualified_role_field.float16.npy"
    token_path = raw_dir / "representative_full_token_qpoints.float16.npy"
    role_shape = list(np.load(role_path, mmap_mode="r").shape)
    token_shape = list(np.load(token_path, mmap_mode="r").shape)
    return {
        "selected_rows": role_shape[0], "qualified_slices": sorted(qualified),
        "role_path": str(role_path.relative_to(ROOT)), "role_shape": role_shape,
        "token_path": str(token_path.relative_to(ROOT)), "token_shape": token_shape,
        "includes_incorrect_rows_inside_qualified_slices": True,
    }


def predict(beta: np.ndarray, features: list[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(features[0], dtype=np.float32)
    for coefficient, feature in zip(beta, features):
        result += coefficient[None] * feature
    return result


def fit_guard(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    relation = np.broadcast_to(base[:, :, contract.ROLES.index("relation")][:, :, None, :], base.shape)
    bins = (base >= 0).astype(np.uint8) * 2 + (relation >= 0).astype(np.uint8)
    table = np.zeros((4,) + residual.shape[1:], dtype=np.float32)
    for state in range(4):
        mask = bins == state
        count = np.sum(mask, axis=0)
        table[state] = np.divide(np.sum(np.where(mask, residual, 0.0), axis=0), count,
                                 out=np.zeros_like(table[state]), where=count > 0)
    return table


def guard_predict(table: np.ndarray, base: np.ndarray) -> np.ndarray:
    relation = np.broadcast_to(base[:, :, contract.ROLES.index("relation")][:, :, None, :], base.shape)
    bins = (base >= 0).astype(np.uint8) * 2 + (relation >= 0).astype(np.uint8)
    flat_table = table.reshape(4, -1); flat_bins = bins.reshape(len(base), -1)
    values = flat_table[flat_bins, np.arange(flat_bins.shape[1])[None, :]]
    return values.reshape(base.shape).astype(np.float32)


def transition_code(base: np.ndarray, delta: np.ndarray, multiplier: float) -> np.ndarray:
    changed = base + delta
    tau = multiplier * (0.02 + 0.05 * np.abs(base))
    code = np.zeros(base.shape, dtype=np.uint8)
    active = np.abs(delta) > tau
    code[active & (base <= 0) & (changed > 0)] = 1
    code[active & (base >= 0) & (changed < 0)] = 2
    same = active & (code == 0)
    code[same & (np.abs(changed) > np.abs(base))] = 3
    code[same & (np.abs(changed) <= np.abs(base))] = 4
    return code


def metric(base: np.ndarray, actual: np.ndarray, predicted: np.ndarray, multiplier: float) -> dict:
    actual_class = transition_code(base, actual, multiplier)
    predicted_class = transition_code(base, predicted, multiplier)
    support = actual_class != 0; pred_support = predicted_class != 0
    tp = int(np.sum(support & pred_support)); fp = int(np.sum(~support & pred_support)); fn = int(np.sum(support & ~pred_support))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    changed_accuracy = float(np.mean(predicted_class[support] == actual_class[support])) if support.any() else 1.0
    mae = float(np.mean(np.abs(predicted - actual))); zero_mae = float(np.mean(np.abs(actual)))
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (actual_class.reshape(-1), predicted_class.reshape(-1)), 1)
    return {
        "support_precision": precision, "support_recall": recall, "support_f1": f1,
        "changed_class_accuracy": changed_accuracy, "all_coordinate_accuracy": float(np.mean(actual_class == predicted_class)),
        "delta_mae": mae, "zero_mae": zero_mae,
        "mae_gain_over_zero": (zero_mae - mae) / zero_mae if zero_mae else 0.0,
        "actual_support_rate": float(np.mean(support)), "predicted_support_rate": float(np.mean(pred_support)),
        "confusion_5x5": confusion.tolist(),
    }


def mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def fit_models(parent_field: np.ndarray, parent_index: list[dict]) -> dict:
    train_pairs = pair_records(parent_index, "discovery")
    if not train_pairs:
        raise RuntimeError("no discovery pairs")
    base, previous, delta = arrays_for_pairs(parent_field, train_pairs)
    shared_mean = np.mean(delta, axis=0)
    shared_beta = fit_coordinate_ridge(shared_features(base, previous), delta)
    shared_prediction = predict(shared_beta, shared_features(base, previous))
    residual = delta - shared_prediction
    family_mean = np.zeros((len(contract.FAMILIES),) + delta.shape[1:], dtype=np.float32)
    family_beta = np.zeros((len(contract.FAMILIES), 4) + delta.shape[1:], dtype=np.float32)
    family_guard = np.zeros((len(contract.FAMILIES), 4) + delta.shape[1:], dtype=np.float32)
    availability = {}
    for family_i, family in enumerate(contract.FAMILIES):
        indices = [i for i, row in enumerate(train_pairs) if row["family"] == family]
        availability[family] = len(indices)
        if not indices:
            continue
        family_mean[family_i] = np.mean(residual[indices], axis=0)
        family_beta[family_i] = fit_coordinate_ridge(family_features(base[indices]), residual[indices])
        family_guard[family_i] = fit_guard(base[indices], residual[indices])
        print(f"[models] family {family_i + 1}/{len(contract.FAMILIES)} {family} n={len(indices)}", flush=True)
    np.save(RAW_SHARED, shared_beta.astype(np.float16), allow_pickle=False)
    np.save(RAW_FAMILY_AFFINE, family_beta.astype(np.float16), allow_pickle=False)
    np.save(RAW_FAMILY_GUARD, family_guard.astype(np.float16), allow_pickle=False)
    save(OUT / "analysis/model_availability.json", availability)
    del base, previous, delta, shared_prediction, residual
    return {
        "shared_mean": shared_mean, "shared_beta": shared_beta,
        "family_mean": family_mean, "family_beta": family_beta, "family_guard": family_guard,
        "availability": availability, "train_pairs": len(train_pairs),
    }


def evaluate_models(models: dict, datasets: list[tuple[str, np.ndarray, list[dict], tuple[str, ...]]]) -> tuple[list[dict], dict, list[str]]:
    rows = []
    wrong_map = {family: contract.FAMILIES[(i + 1) % len(contract.FAMILIES)] for i, family in enumerate(contract.FAMILIES)}
    for dataset_name, field, index, partitions in datasets:
        for part in partitions:
            pairs = pair_records(index, part)
            if not pairs:
                continue
            for start in range(0, len(pairs), 16):
                batch_pairs = pairs[start:start + 16]
                base, previous, actual = arrays_for_pairs(field, batch_pairs)
                shared = predict(models["shared_beta"], shared_features(base, previous))
                for local_i, spec in enumerate(batch_pairs):
                    family_i = contract.FAMILIES.index(spec["family"])
                    wrong_i = contract.FAMILIES.index(wrong_map[spec["family"]])
                    family_affine = shared[local_i] + predict(models["family_beta"][family_i], family_features(base[local_i:local_i + 1]))[0]
                    family_guard = shared[local_i] + guard_predict(models["family_guard"][family_i], base[local_i:local_i + 1])[0]
                    wrong_guard = shared[local_i] + guard_predict(models["family_guard"][wrong_i], base[local_i:local_i + 1])[0]
                    predictions = {
                        "M0_zero": np.zeros_like(actual[local_i]),
                        "M1_shared_mean": models["shared_mean"],
                        "M2_shared_affine": shared[local_i],
                        "M3_family_mean": shared[local_i] + models["family_mean"][family_i],
                        "M4_family_affine": family_affine,
                        "M5_family_guard": family_guard,
                        "M6_wrong_family_guard": wrong_guard,
                    }
                    for tau in contract.TAU_MULTIPLIERS:
                        for method, predicted in predictions.items():
                            rows.append({
                                "dataset": dataset_name, "partition": part, "family": spec["family"],
                                "language": spec["language"], "surface": spec["surface"], "unit": spec["unit"],
                                "tau_multiplier": tau, "method": method,
                                **metric(base[local_i], actual[local_i], predicted, tau),
                            })
                print(f"[evaluate] {dataset_name} {part} {min(start + len(batch_pairs), len(pairs))}/{len(pairs)}", flush=True)
    summaries = {}
    strict = []
    primary = contract.PRIMARY_TAU
    for family in contract.FAMILIES:
        panels, passes = {}, []
        for dataset_name, part in itertools.product(("parent", "fresh"), ("confirmation", "lockbox")):
            main = [row for row in rows if row["family"] == family and row["dataset"] == dataset_name and row["partition"] == part and row["method"] == "M5_family_guard" and row["tau_multiplier"] == primary]
            shared = [row for row in rows if row["family"] == family and row["dataset"] == dataset_name and row["partition"] == part and row["method"] == "M2_shared_affine" and row["tau_multiplier"] == primary]
            wrong = [row for row in rows if row["family"] == family and row["dataset"] == dataset_name and row["partition"] == part and row["method"] == "M6_wrong_family_guard" and row["tau_multiplier"] == primary]
            panel = mean_metrics(main)
            panel["units"] = len({row["unit"] for row in main})
            panel["pair_rows"] = len(main)
            shared_mean = mean_metrics(shared); wrong_mean = mean_metrics(wrong)
            panel["shared_support_f1"] = shared_mean.get("support_f1")
            panel["wrong_family_support_f1"] = wrong_mean.get("support_f1")
            panel["f1_gain_over_shared"] = panel.get("support_f1", 0.0) - shared_mean.get("support_f1", 0.0)
            panel["f1_gain_over_wrong_family"] = panel.get("support_f1", 0.0) - wrong_mean.get("support_f1", 0.0)
            shared_mae = shared_mean.get("delta_mae", 0.0)
            panel["relative_mae_gain_over_shared"] = (shared_mae - panel.get("delta_mae", shared_mae)) / shared_mae if shared_mae else 0.0
            panel["passed"] = bool(main) and panel["units"] >= contract.FAMILY_GATES["minimum_units"] and all(
                panel[key] >= threshold for key, threshold in contract.FAMILY_GATES.items() if key != "minimum_units"
            )
            panels[f"{dataset_name}_{part}"] = panel
            passes.append(panel["passed"])
        panels["strict_pass"] = all(passes)
        summaries[family] = panels
        if panels["strict_pass"]:
            strict.append(family)
    tournament = {}
    for tau in contract.TAU_MULTIPLIERS:
        tournament[f"tau_{tau:g}"] = {
            method: mean_metrics([row for row in rows if row["tau_multiplier"] == tau and row["method"] == method])
            for method in METHODS
        }
    return rows, {"family_panels": summaries, "tournament": tournament}, strict


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    behavior = result["behavior"]
    tournament = result["model_tournament"][f"tau_{contract.PRIMARY_TAU:g}"]
    family = result["family_summary"]
    formula = r"""
$$
\widehat{\Delta H}^{\mathrm{shared}}=\beta_0+\beta_1H_{q,r,j}+\beta_2H_{q-1,r,j}+\beta_3H_{q,\mathrm{boundary},j}+\beta_4H_{q,\mathrm{query},j}+\beta_5H_{q,\mathrm{relation},j}.
$$

$$
b_{i,q,r,j}=2\mathbf 1[H_{i,q,r,j}\ge0]+\mathbf 1[H_{i,q,\mathrm{relation},j}\ge0],\quad
\widehat{\Delta H}=\widehat{\Delta H}^{\mathrm{shared}}+R_{f,b_{i,q,r,j},q,r,j}.
$$
"""
    compact_tournament = {
        method: {key: value for key, value in metrics.items() if key in ("support_f1", "changed_class_accuracy", "delta_mae", "actual_support_rate")}
        for method, metrics in tournament.items()
    }
    compact_family = {
        name: {
            panel: {key: value for key, value in metrics.items() if key in ("units", "support_f1", "f1_gain_over_shared", "f1_gain_over_wrong_family", "relative_mae_gain_over_shared", "passed")}
            for panel, metrics in panels.items() if isinstance(metrics, dict)
        } | {"strict_pass": panels["strict_pass"]}
        for name, panels in family.items()
    }
    text = f"""

## Phase {PHASE}: Qwen3-4B广语言族全坐标共享动力学与族残差锦标赛（C885-C904） [{stamp}]

**阶段目标与测试原理。** 本期严格复用Phase2234冻结材料、分区、Qwen3-4B BF16模型、双行为门、阈值阶梯、模型集合和停止条件。先分别运行候选A/B与自由生成；隐藏状态只读取双行为合格的 `panel x family x language` 切片，但合格切片内行为错误行也完整保留。相机保存embedding、36个block后HiddenState、final norm、6个功能角色、全部2560个激活坐标，并保存每个合格切片/分区的一条全token代表场。没有读取attention、MLP、权重或梯度，也没有PCA、Top-K、余弦筛选和donor差分搬运。

**核心算法。** 共享前向动力学使用同一物理坐标的自身、前一检查点、边界、查询和关系角色；族守卫残差只按当前样本该坐标与关系角色同坐标的正负四格选择。每个坐标都被保留，没有先筛坐标：
{formula}

**测试用例与行为结果。** 代表例包括类型链、整体部分、时间、因果、施受事语态、否定、共指、嵌套态度、属性、比较、翻译和量词；另有“我喜欢某人吃某物”的内外否定与1-3跳关系图旗舰。父集 `{behavior['parent_rows']}` 行、新词集 `{behavior['fresh_rows']}` 行全部完成候选和自由生成。父/新词双行为合格切片分别为 `{len(behavior['parent_qualified_slices'])}` 与 `{len(behavior['fresh_qualified_slices'])}`，捕获行数分别为 `{result['field']['parent']['selected_rows']}` 与 `{result['field']['fresh']['selected_rows']}`。详细逐切片准确率保存在结果文件，未合格切片没有被解释成神经机制阴性。

**结果与冻结门槛。** 主阈值为 $2(0.02+0.05|H|)$。模型锦标赛主结果：`{json.dumps(compact_tournament, ensure_ascii=False)}`。12族四面板结果：`{json.dumps(compact_family, ensure_ascii=False)}`。严格候选为 `{result['strict_family_candidates']}`。候选要求在父确认、父锁箱、新词确认、新词锁箱同时超过共享模型和错族同容量控制，因此“模型自身F1高”不再足以通过。

**理论进展与严格分析。** 本期最重要的识别目标不是固定语义方向，而是判断族条件残差是否在共享前向动力学之外提供独立、跨词汇、跨表面的前瞻信息。若M2已经解释大部分变化而M5没有稳定增益，结论只能是共享动力学强；若M5四面板通过，才获得族条件齿轮候选。无论哪种情况，预测仍不等于因果机制，激活坐标仍不是模型参数。

**问题、硬伤与瓶颈。** 材料是受控模板，人类自然度盲评为NA；中文关系跨度使用最小解码token区间补偿BPE上下文合并，属于合法的角色编译修复但会把一个语义词映射到含邻接字符的token；模型只有Qwen3-4B主场；同坐标角色耦合仍不能表达任意跨坐标超边；四格符号守卫是基础算法，不穷尽非线性条件结构；行为合格筛选改变了各族可用样本量。支持F1受阈值密度影响，所以本期同时保存 $m=1,2,4$ 全阶梯，不用单阈值替代结论。

**结论与下一步授权。** 严格候选仅授权全坐标预测响应的调用/删除/错族控制；没有候选时因果分支记NA，但嵌套态度和关系图组合观察、跨模型相同语义分母面板、全坐标可视化与清理继续执行，不做全项目停止。理论名保持“条件化输出场闭合理论”，新数学门继续关闭。

**相关文件。** 脚本 `tests/glm5/phase2235_c885_c904_qwen_broad_family_full_coordinate_tournament.py`；结果 `{OUT.relative_to(ROOT)}`；逐行行为、隐藏索引、模型锦标赛、族面板和冻结全坐标系数均在该目录。原始大场将在后续组合、因果和可视化完成后按哈希清理。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    if not load(CONTRACT_OUT / "analysis/final.json")["all_checks_passed"]:
        raise RuntimeError("Phase2234 contract did not pass")
    for sub in ("protocol", "behavior", "raw/parent", "raw/fresh", "analysis", "audit"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol/execution_identity.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parent_material_sha256": file_hash(CONTRACT_OUT / "material/parent_qwen_compiled.jsonl"),
        "fresh_material_sha256": file_hash(CONTRACT_OUT / "material/fresh_qwen_compiled.jsonl"),
        "contract_sha256": file_hash(CONTRACT_OUT / "protocol/preregistration.json"),
        "revealed_changes": "none",
    })
    parent_compiled = read_rows(CONTRACT_OUT / "material/parent_qwen_compiled.jsonl")
    fresh_compiled = read_rows(CONTRACT_OUT / "material/fresh_qwen_compiled.jsonl")
    completed_capture = all(path.exists() for path in (
        OUT / "behavior/parent_candidate.jsonl", OUT / "behavior/parent_generation.jsonl",
        OUT / "behavior/fresh_candidate.jsonl", OUT / "behavior/fresh_generation.jsonl",
        OUT / "behavior/parent_slices.json", OUT / "behavior/fresh_slices.json",
        RAW_PARENT, RAW_FRESH, OUT / "raw/parent/hidden_index.jsonl",
        OUT / "raw/fresh/hidden_index.jsonl",
    ))
    if completed_capture:
        parent_candidate = read_rows(OUT / "behavior/parent_candidate.jsonl")
        parent_generation = read_rows(OUT / "behavior/parent_generation.jsonl")
        fresh_candidate = read_rows(OUT / "behavior/fresh_candidate.jsonl")
        fresh_generation = read_rows(OUT / "behavior/fresh_generation.jsonl")
        parent_slices = load(OUT / "behavior/parent_slices.json")
        fresh_slices = load(OUT / "behavior/fresh_slices.json")
        parent_qualified = {key for key, row in parent_slices.items() if row["qualified"]}
        fresh_qualified = {key for key, row in fresh_slices.items() if row["qualified"]}
        parent_info = existing_field_info(OUT / "raw/parent", parent_qualified)
        fresh_info = existing_field_info(OUT / "raw/fresh", fresh_qualified)
        placement = {"resume": "verified capture from the same frozen execution identity"}
        quantization = {"resume": "model released after full-precision capture"}
    else:
        model = None
        try:
            model, tokenizer, device, placement = contract.prior.qwen_model()
            parent_candidate, parent_generation = run_behavior(model, tokenizer, device, parent_compiled, "C885-parent")
            fresh_candidate, fresh_generation = run_behavior(model, tokenizer, device, fresh_compiled, "C885-fresh")
            for name, rows in (
                ("parent_candidate", parent_candidate), ("parent_generation", parent_generation),
                ("fresh_candidate", fresh_candidate), ("fresh_generation", fresh_generation),
            ):
                write_rows(OUT / f"behavior/{name}.jsonl", rows)
            pc = {row["case_id"]: row for row in parent_candidate}; pg = {row["case_id"]: row for row in parent_generation}
            fc = {row["case_id"]: row for row in fresh_candidate}; fg = {row["case_id"]: row for row in fresh_generation}
            parent_slices, parent_qualified = behavior_slices(parent_compiled, pc, pg, ("discovery", "confirmation", "lockbox"))
            fresh_slices, fresh_qualified = behavior_slices(fresh_compiled, fc, fg, ("confirmation", "lockbox"))
            parent_info = capture_field(model, tokenizer, device, parent_compiled, pc, pg, parent_qualified, OUT / "raw/parent", "C885-parent")
            fresh_info = capture_field(model, tokenizer, device, fresh_compiled, fc, fg, fresh_qualified, OUT / "raw/fresh", "C885-fresh")
            quantization = contract.prior.scope.parent.previous.model_base().quantization_audit(model)
        finally:
            release_model(model)
        save(OUT / "behavior/parent_slices.json", parent_slices); save(OUT / "behavior/fresh_slices.json", fresh_slices)
    parent_index = read_rows(OUT / "raw/parent/hidden_index.jsonl")
    fresh_index = read_rows(OUT / "raw/fresh/hidden_index.jsonl")
    parent_field = np.load(RAW_PARENT, mmap_mode="r"); fresh_field = np.load(RAW_FRESH, mmap_mode="r")
    try:
        models = fit_models(parent_field, parent_index)
        metric_rows, summaries, strict = evaluate_models(models, [
            ("parent", parent_field, parent_index, ("confirmation", "lockbox")),
            ("fresh", fresh_field, fresh_index, ("confirmation", "lockbox")),
        ])
    finally:
        close_mmap(parent_field); close_mmap(fresh_field)
    write_rows(OUT / "analysis/unit_model_tournament_metrics.jsonl", metric_rows)
    save(OUT / "analysis/model_tournament.json", summaries["tournament"])
    save(OUT / "analysis/family_panels.json", summaries["family_panels"])
    behavior = {
        "parent_rows": len(parent_compiled), "fresh_rows": len(fresh_compiled),
        "parent_qualified_slices": sorted(parent_qualified), "fresh_qualified_slices": sorted(fresh_qualified),
        "parent_candidate_accuracy": float(np.mean([row["correct"] for row in parent_candidate])),
        "parent_generation_accuracy": float(np.mean([row["correct"] for row in parent_generation])),
        "fresh_candidate_accuracy": float(np.mean([row["correct"] for row in fresh_candidate])),
        "fresh_generation_accuracy": float(np.mean([row["correct"] for row in fresh_generation])),
    }
    checks = {
        "contract_passed": True,
        "behavior_complete": len(parent_candidate) == len(parent_generation) == len(parent_compiled) and len(fresh_candidate) == len(fresh_generation) == len(fresh_compiled),
        "some_parent_qualified": bool(parent_qualified), "some_fresh_qualified": bool(fresh_qualified),
        "all_coordinates": parent_info["role_shape"][-1] == contract.DIM and fresh_info["role_shape"][-1] == contract.DIM,
        "all_checkpoints": parent_info["role_shape"][1] == contract.CHECKPOINTS and fresh_info["role_shape"][1] == contract.CHECKPOINTS,
        "incorrect_rows_retained": parent_info["includes_incorrect_rows_inside_qualified_slices"] and fresh_info["includes_incorrect_rows_inside_qualified_slices"],
        "threshold_ladder_complete": set(row["tau_multiplier"] for row in metric_rows) == set(contract.TAU_MULTIPLIERS),
        "all_models_scored": set(row["method"] for row in metric_rows) == set(METHODS),
        "finite": contract.finite(summaries),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "behavior": behavior,
        "behavior_slices": {"parent": parent_slices, "fresh": fresh_slices},
        "field": {"parent": parent_info, "fresh": fresh_info},
        "model_training": {"train_pairs": models["train_pairs"], "availability": models["availability"]},
        "model_tournament": summaries["tournament"], "family_summary": summaries["family_panels"],
        "strict_family_candidates": strict, "gates": contract.FAMILY_GATES,
        "placement": placement, "quantization": quantization,
        "strict_conclusion": "Family-specific evidence is only any candidate passing all four prospective panels beyond shared and wrong-family controls; raw model scores alone are descriptive.",
        "next_authorization": "Continue every registered flagship/cross-model/visual branch; run causal intervention only for strict candidates.",
    }
    save(final_path, result); append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
