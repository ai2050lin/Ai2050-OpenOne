#!/usr/bin/env python3
"""Discover low-capacity temporal hypotheses and freeze them before confirmation."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402


P362 = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
P361 = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace"
OUT = ROOT / "tests/gpt5/result/phase363_temporal_hypotheses"
ROUND = "strict_temporal_innovation_formulas"
MODELS = ("qwen3", "glm4", "deepseek7b")
COMPETITION_METRICS = ("target_token_margin", "target_log_rank", "vocab_entropy")
RIDGE_LAMBDA = 1e-3
KAPPA = 1.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_value(row: dict[str, Any], candidate: dict[str, Any]) -> float:
    _prefix, component, role = candidate["feature_id"].split("::", 2)
    role_index = row["role_names"].index(role)
    return float(row["component_norms"][component][role_index])


def depth_match(row: dict[str, Any], depth: str) -> bool:
    value = float(row["relative_depth"])
    return (depth == "early" and value < 1 / 3) or (depth == "middle" and 1 / 3 <= value < 2 / 3) or (depth == "late" and value >= 2 / 3)


def ridge_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([z, np.ones(len(z))])
    penalty = np.eye(design.shape[1]) * RIDGE_LAMBDA
    penalty[-1, -1] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def ridge_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    return np.column_stack([z, np.ones(len(z))]) @ model["weights"]


def single_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    variance = float(((x - x.mean()) ** 2).sum())
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / variance) if variance > 1e-12 else 0.0
    return slope, float(y.mean() - slope * x.mean())


def mechanism_prediction(
    train: list[dict[str, Any]],
    train_targets: np.ndarray,
    valid: list[dict[str, Any]],
) -> np.ndarray:
    centers = {
        mechanism: median([
            float(target) for row, target in zip(train, train_targets, strict=True)
            if row["mechanism"] == mechanism
        ])
        for mechanism in {row["mechanism"] for row in train}
    }
    return np.array([centers[row["mechanism"]] for row in valid], dtype=np.float64)


def practical_effect_floor(
    train: list[dict[str, Any]],
    group_target: Any,
    case_target: Any,
) -> dict[str, Any]:
    case_deviations = []
    group_values: dict[str, list[float]] = defaultdict(list)
    for row in train:
        values = [float(case_target(case_id)) for case_id in row["case_ids"]]
        center = median(values)
        case_deviations.extend(abs(value - center) for value in values)
        group_values[row["mechanism"]].append(float(group_target(row)))
    template_deviations = []
    for values in group_values.values():
        center = median(values)
        template_deviations.extend(abs(value - center) for value in values)
    sigma_case = median(case_deviations) if case_deviations else 0.0
    sigma_template = median(template_deviations) if template_deviations else 0.0
    return {
        "sigma_rerun": None,
        "sigma_case": float(sigma_case),
        "sigma_template": float(sigma_template),
        "effect_floor": float(max(sigma_case, sigma_template)),
    }


def main() -> None:
    survivor_rows = read_jsonl(P362 / "phase362_frozen_candidate_audit_rows.jsonl")
    candidates = [row for row in survivor_rows if row["b3_beats_all_alternatives_all_models"]]
    if len(candidates) != 7:
        raise RuntimeError(f"Expected seven frozen inputs, got {len(candidates)}")
    candidate_path = P361 / "phase361_frozen_predictive_candidates.jsonl"
    execution = [
        row for row in read_jsonl(P362 / "private" / "phase362_execution_cases.jsonl")
        if row["phase362_split"] == "independent_calibration"
    ]
    case_info = {row["blind_case_id"]: row for row in execution}
    ledger = [
        row for model in MODELS
        for row in read_jsonl(P362 / "models" / model / "phase362_generation_time_rows.jsonl")
    ]
    rows_by_case_time: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        rows_by_case_time[(row["blind_case_id"], row["generation_time"])].append(row)
    feature_by_case_time = {}
    for key, rows in rows_by_case_time.items():
        feature_by_case_time[key] = np.array([
            np.mean([feature_value(row, candidate) for row in rows if depth_match(row, candidate["depth_bin"])])
            for candidate in candidates
        ], dtype=np.float64)

    tokenizers = {}
    competition_by_case_time = {}
    surface_divergence = {}
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizers[model] = AutoTokenizer.from_pretrained(
                str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
                local_files_only=True, use_fast=False,
            )
        for case_id, case in case_info.items():
            tokenizer = tokenizers[case["model"]]
            target_ids = tokenizer(case["target"], add_special_tokens=False)["input_ids"]
            distractor_ids = [tokenizer(value, add_special_tokens=False)["input_ids"] for value in case["distractors"]]
            path = P362 / "sealed_calibration" / case["model"] / f"{case_id}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for time_row in payload["times"]:
                time = int(time_row["generation_time"])
                logits = time_row["full_vocabulary_logits"].float()
                target_id = int(target_ids[time]) if time < len(target_ids) else int(tokenizer.eos_token_id)
                wrong_ids = [int(values[time]) if time < len(values) else int(tokenizer.eos_token_id) for values in distractor_ids]
                margin = float(logits[target_id] - max(logits[value] for value in wrong_ids))
                rank = int((logits > logits[target_id]).sum().item()) + 1
                probs = torch.softmax(logits, dim=-1)
                entropy = float((-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log())).sum().item())
                competition_by_case_time[(case_id, time)] = np.array([margin, math_log_rank(rank), entropy])
                surface_divergence[(case_id, time)] = int(payload["generated_token_ids"][time] != target_id)
            del payload
    finally:
        tokenizers.clear()

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case_id, case in case_info.items():
        groups[(case["model"], case["phase362_group_id"])].append(case_id)
    group_records = []
    for (model, group_id), case_ids in groups.items():
        mechanism = f"{case_info[case_ids[0]]['family_id']}/{case_info[case_ids[0]]['mechanism_id']}"
        record = {"model": model, "group_id": group_id, "mechanism": mechanism, "case_ids": case_ids}
        for time in range(3):
            record[f"z{time}"] = np.mean([feature_by_case_time[(case_id, time)] for case_id in case_ids], axis=0)
            record[f"c{time}"] = np.mean([competition_by_case_time[(case_id, time)] for case_id in case_ids], axis=0)
            record[f"d{time}"] = np.mean([surface_divergence[(case_id, time)] for case_id in case_ids])
        group_records.append(record)
    split = {}
    for model in MODELS:
        mechanisms = sorted({row["mechanism"] for row in group_records if row["model"] == model})
        for mechanism in mechanisms:
            values = [row for row in group_records if row["model"] == model and row["mechanism"] == mechanism]
            values.sort(key=lambda row: hashlib.sha256(("phase363-split:" + row["group_id"]).encode()).hexdigest())
            for row in values[:4]: split[(model, row["group_id"])] = "formula_train"
            for row in values[4:]: split[(model, row["group_id"])] = "formula_validation"

    split_rows = [{
        "model": row["model"], "mechanism": row["mechanism"], "group_id": row["group_id"],
        "case_count": len(row["case_ids"]), "split": split[(row["model"], row["group_id"])],
    } for row in sorted(group_records, key=lambda item: (item["model"], item["mechanism"], item["group_id"]))]
    formula_rows = []
    targets = [
        ("time_innovation", index, candidates[index]["candidate_id"])
        for index in range(len(candidates))
    ] + [
        ("competition_change", index, metric)
        for index, metric in enumerate(COMPETITION_METRICS)
    ]
    for target_type, target_index, target_name in targets:
        for transition in (0, 1):
            per_model = {}
            for model in MODELS:
                train = [row for row in group_records if row["model"] == model and split[(model, row["group_id"])] == "formula_train"]
                valid = [row for row in group_records if row["model"] == model and split[(model, row["group_id"])] == "formula_validation"]
                x_train = np.stack([row[f"z{transition}"] for row in train])
                x_valid = np.stack([row[f"z{transition}"] for row in valid])
                baseline_formulas: dict[str, Any] = {}
                if target_type == "time_innovation":
                    current_train = np.array([row[f"z{transition}"][target_index] for row in train])
                    current_valid = np.array([row[f"z{transition}"][target_index] for row in valid])
                    next_train = np.array([row[f"z{transition + 1}"][target_index] for row in train])
                    next_valid = np.array([row[f"z{transition + 1}"][target_index] for row in valid])
                    ar_slope, ar_intercept = single_fit(current_train, next_train)
                    y_train = next_train - (ar_slope * current_train + ar_intercept)
                    y_valid = next_valid - (ar_slope * current_valid + ar_intercept)
                    group_target = lambda row, i=target_index, s=ar_slope, b=ar_intercept: (
                        row[f"z{transition + 1}"][i] - (s * row[f"z{transition}"][i] + b)
                    )
                    case_target = lambda case_id, i=target_index, s=ar_slope, b=ar_intercept: (
                        feature_by_case_time[(case_id, transition + 1)][i]
                        - (s * feature_by_case_time[(case_id, transition)][i] + b)
                    )
                    baseline_predictions = {
                        "zero_innovation": np.zeros(len(valid)),
                        "global_median_innovation": np.full(len(valid), median(y_train)),
                        "mechanism_median_innovation": mechanism_prediction(train, y_train, valid),
                    }
                    baseline_formulas["time_autoregression"] = {"slope": ar_slope, "intercept": ar_intercept}
                else:
                    y_train = np.array([row[f"c{transition + 1}"][target_index] - row[f"c{transition}"][target_index] for row in train])
                    y_valid = np.array([row[f"c{transition + 1}"][target_index] - row[f"c{transition}"][target_index] for row in valid])
                    group_target = lambda row, i=target_index: row[f"c{transition + 1}"][i] - row[f"c{transition}"][i]
                    case_target = lambda case_id, i=target_index: (
                        competition_by_case_time[(case_id, transition + 1)][i]
                        - competition_by_case_time[(case_id, transition)][i]
                    )
                    rank_train = np.array([row[f"c{transition}"][1] for row in train])
                    rank_valid = np.array([row[f"c{transition}"][1] for row in valid])
                    entropy_train = np.array([row[f"c{transition}"][2] for row in train])
                    entropy_valid = np.array([row[f"c{transition}"][2] for row in valid])
                    rank_slope, rank_intercept = single_fit(rank_train, y_train)
                    entropy_slope, entropy_intercept = single_fit(entropy_train, y_train)
                    baseline_predictions = {
                        "zero_change": np.zeros(len(valid)),
                        "global_median_change": np.full(len(valid), median(y_train)),
                        "mechanism_median_change": mechanism_prediction(train, y_train, valid),
                        "current_rank_only": rank_slope * rank_valid + rank_intercept,
                        "current_entropy_only": entropy_slope * entropy_valid + entropy_intercept,
                    }
                    baseline_formulas.update({
                        "current_rank_only": {"slope": rank_slope, "intercept": rank_intercept},
                        "current_entropy_only": {"slope": entropy_slope, "intercept": entropy_intercept},
                    })
                joint = ridge_fit(x_train, y_train)
                joint_pred = ridge_predict(joint, x_valid)
                errors = {
                    name: float(np.mean(np.abs(y_valid - prediction)))
                    for name, prediction in baseline_predictions.items()
                }
                errors["seven_candidate_joint"] = float(np.mean(np.abs(y_valid - joint_pred)))
                strongest_baseline_name = min(baseline_predictions, key=lambda name: errors[name])
                strongest = errors[strongest_baseline_name]
                floor_parts = practical_effect_floor(train, group_target, case_target)
                gain = strongest - errors["seven_candidate_joint"]
                per_model[model] = {
                    "errors": {key: round(value, 9) for key, value in errors.items()},
                    "strongest_baseline": strongest_baseline_name,
                    "gain": round(gain, 9),
                    "effect_floor": {key: None if value is None else round(value, 9) for key, value in floor_parts.items()},
                    "passes": bool(gain > KAPPA * floor_parts["effect_floor"]),
                    "baseline_formulas": baseline_formulas,
                    "seven_candidate_formula": {
                        "feature_mean": joint["mean"].tolist(), "feature_scale": joint["scale"].tolist(),
                        "weights": joint["weights"].tolist(),
                    },
                }
            all_models = bool(all(bool(value["passes"]) for value in per_model.values()))
            formula_rows.append({
                "formula_id": f"{target_type}::t{transition}_t{transition+1}::{target_name}",
                "target_type": target_type, "target_name": target_name,
                "transition": f"t{transition}_t{transition+1}", "target_index": target_index,
                "input_candidate_ids": [row["candidate_id"] for row in candidates],
                "ridge_lambda": RIDGE_LAMBDA, "kappa": KAPPA,
                "per_model": per_model, "all_models_discovery_pass": all_models,
            })
    frozen = [row for row in formula_rows if row["all_models_discovery_pass"]]
    root = OUT / ROUND
    write_jsonl(root / "phase363_formula_split_rows.jsonl", split_rows)
    write_jsonl(root / "phase363_all_formula_rows.jsonl", formula_rows)
    write_jsonl(root / "phase363_frozen_formula_rows.jsonl", frozen)
    summary = {
        "schema_version": "40.0.0", "phase_id": "Phase363", "created_at": now(),
        "denominator": {
            "input_candidate_count": 7, "formula_train_group_count": 48,
            "formula_validation_group_count": 24, "tested_formula_count": len(formula_rows),
            "frozen_formula_count": len(frozen),
        },
        "frozen_inputs": {
            "phase361_candidate_sha256": sha256_file(candidate_path),
            "ridge_lambda": RIDGE_LAMBDA, "kappa": KAPPA,
            "targets": ["time_innovation", *COMPETITION_METRICS],
            "surface_divergence_is_strengthening_only": True,
            "full_phrase_margin_available": False,
            "semantic_divergence_available": False,
        },
        "quality": {
            "physical_confirmation_read": False,
            "independent_group_is_analysis_unit": True,
            "layers_or_conditions_used_as_independent_units": False,
            "effect_floor_uses_case_and_template_variation": True,
            "rerun_noise_available": False,
        },
        "results": {
            "formula_positive_gain_all_models_count": sum(
                all(value["gain"] > 0 for value in row["per_model"].values()) for row in formula_rows
            ),
            "formula_above_effect_floor_all_models_count": len(frozen),
            "per_model_above_effect_floor_count": {
                model: sum(row["per_model"][model]["passes"] for row in formula_rows) for model in MODELS
            },
        },
        "claim_boundary": {
            "full_vocabulary_logits_used": True,
            "target_token_margin_identifiable": True,
            "target_token_rank_identifiable": True,
            "vocabulary_entropy_identifiable": True,
            "full_phrase_margin_identifiable": False,
            "continue_stop_protocol_margin_identifiable": False,
            "semantic_divergence_identifiable": False,
            "causal_intervention_executed": False,
        },
        "next_decision": "run_physical_confirmation" if frozen else "close_temporal_candidate_route_without_opening_confirmation",
    }
    write_json(root / "phase363_hypothesis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def math_log_rank(rank: int) -> float:
    return float(np.log1p(rank))


if __name__ == "__main__":
    main()
