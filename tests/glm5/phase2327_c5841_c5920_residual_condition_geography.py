#!/usr/bin/env python3
"""Map full-coordinate residual conditions and cross-unit predictive structure."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2325 = RESULT / "phase2325_c5721_c5800_qwen4b_fp16_large_family_confirmation"
P2326 = RESULT / "phase2326_c5801_c5840_large_confirmation_atlas_audit"
OUT = RESULT / "phase2327_c5841_c5920_residual_condition_geography"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2327
CAMPAIGN = "C5841-C5920"
DIMENSION = 2560
UNITS = (17, 18)
FACTORS = ("language", "surface", "state", "family")
CONTEXT_FACTORS = ("language", "surface", "state")
MODELS = ("lockbox_global", "recent_global", "main_context", "main_all", "pair_context")
CHANNELS = (
    "unit17_effect", "unit18_effect", "replicated_mean_effect",
    "absolute_difference", "same_sign", "normalized_difference",
)
EPS = 1e-12
SELECTION_IMPROVEMENT_MIN = 0.03
EFFECT_SIGN_GATE = 0.65
EFFECT_RELATIVE_MSE_MAX = 1.0

sys.path.insert(0, str(TESTS))
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2322_c5521_c5600_full_coordinate_reuse_passports as passport  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def level(row: dict, factor: str) -> str:
    return str(row[factor])


def fit_components(values: np.ndarray, rows: list[dict]) -> dict:
    mu = values.mean(axis=0, dtype=np.float64)
    mains = {}
    for factor in FACTORS:
        mains[factor] = {}
        for value in sorted({level(row, factor) for row in rows}):
            indices = [i for i, row in enumerate(rows) if level(row, factor) == value]
            mains[factor][value] = values[indices].mean(axis=0, dtype=np.float64) - mu
    pairs = {}
    for left, right in combinations(CONTEXT_FACTORS, 2):
        pairs[f"{left}|{right}"] = {}
        levels = sorted({(level(row, left), level(row, right)) for row in rows})
        for left_value, right_value in levels:
            indices = [i for i, row in enumerate(rows)
                       if level(row, left) == left_value and level(row, right) == right_value]
            pair_mean = values[indices].mean(axis=0, dtype=np.float64)
            pairs[f"{left}|{right}"][f"{left_value}|{right_value}"] = (
                pair_mean - mu - mains[left][left_value] - mains[right][right_value]
            )
    return {"mu": mu, "mains": mains, "pairs": pairs}


def predict(components: dict, row: dict, model: str) -> np.ndarray:
    value = components["mu"].copy()
    if model in ("main_context", "main_all", "pair_context"):
        for factor in CONTEXT_FACTORS:
            value += components["mains"][factor][level(row, factor)]
    if model == "main_all":
        value += components["mains"]["family"][level(row, "family")]
    if model == "pair_context":
        for left, right in combinations(CONTEXT_FACTORS, 2):
            key = f"{left}|{right}"
            value += components["pairs"][key][f"{level(row, left)}|{level(row, right)}"]
    return value


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.square(actual - predicted, dtype=np.float64).sum()
                 / (np.square(actual, dtype=np.float64).sum() + EPS))


def load_fields() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    derivative_meta = json.loads(
        (VIS / "c5801_qwen4b_fp16_large_confirmation_derivative.json").read_text(encoding="utf-8")
    )
    residual_meta = json.loads(
        (VIS / "c5803_qwen4b_fp16_large_confirmation_global_residual.json").read_text(encoding="utf-8")
    )
    derivative = np.load(VIS / Path(derivative_meta["binary_url"]).name, mmap_mode="r")
    residual = np.load(VIS / Path(residual_meta["binary_url"]).name, mmap_mode="r")
    if list(derivative.shape) != [6912, DIMENSION] or list(residual.shape) != [6912, DIMENSION]:
        raise RuntimeError(("published_field_shape", derivative.shape, residual.shape))
    index = atlas.read_jsonl(P2325 / "index/active_rows.jsonl")
    if len(index) != 128:
        raise RuntimeError(("index_rows", len(index)))
    return derivative.reshape(128, 3, 6, 3, DIMENSION), residual.reshape(128, 3, 6, 3, DIMENSION), index


def effect_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    amplitude = np.sqrt((np.square(left, dtype=np.float64) + np.square(right, dtype=np.float64)) / 2.0)
    positive = amplitude > 0
    median = float(np.median(amplitude[positive])) if np.any(positive) else 0.0
    low = positive & (amplitude <= median)
    high = amplitude > median
    same = left * right > 0
    denominator = 0.5 * (np.square(left, dtype=np.float64).sum()
                         + np.square(right, dtype=np.float64).sum()) + EPS
    shared_weight = np.minimum(np.abs(left), np.abs(right))
    return {
        "all_coordinate_sign_agreement": float(np.mean(same[positive])) if np.any(positive) else 0.0,
        "low_amplitude_half_sign_agreement": float(np.mean(same[low])) if np.any(low) else 0.0,
        "high_amplitude_half_sign_agreement": float(np.mean(same[high])) if np.any(high) else 0.0,
        "amplitude_weighted_same_sign_overlap": float(
            shared_weight[same].sum() / (shared_weight.sum() + EPS)
        ),
        "symmetric_relative_mse": float(np.square(left - right, dtype=np.float64).sum() / denominator),
        "median_joint_amplitude": median,
    }


def condition_geography(
    residual: np.ndarray,
    index: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    rows = []
    passports = []
    records = []
    unit_indices = {unit: [i for i, row in enumerate(index) if int(row["unit"]) == unit]
                    for unit in UNITS}
    for source_index in range(3):
        for probe in range(6):
            for target_index in range(3):
                cell = residual[:, source_index, probe, target_index].astype(np.float64)
                unit_means = {
                    unit: cell[indices].mean(axis=0) for unit, indices in unit_indices.items()
                }
                for factor in FACTORS:
                    levels = sorted({level(row, factor) for row in index})
                    for factor_level in levels:
                        effects = {}
                        for unit, indices in unit_indices.items():
                            selected = [i for i in indices if level(index[i], factor) == factor_level]
                            effects[unit] = cell[selected].mean(axis=0) - unit_means[unit]
                        left, right = effects[17], effects[18]
                        metrics = effect_metrics(left, right)
                        records.append({
                            "source_index": source_index,
                            "probe": probe,
                            "target_index": target_index,
                            "factor": factor,
                            "level": factor_level,
                            **metrics,
                        })
                        amplitude = np.sqrt((np.square(left) + np.square(right)) / 2.0)
                        normalized = np.abs(left - right) / (amplitude + EPS)
                        values = (
                            left, right, (left + right) / 2.0, np.abs(left - right),
                            (left * right > 0).astype(np.float64), normalized,
                        )
                        for channel, value in zip(CHANNELS, values):
                            passports.append(value.astype(np.float32))
                            rows.append({
                                "source_index": source_index,
                                "probe": probe,
                                "target_index": target_index,
                                "factor": factor,
                                "level": factor_level,
                                "channel": channel,
                                "unit_left": 17,
                                "unit_right": 18,
                            })
    matrix = np.stack(passports)
    by_factor = {}
    for factor in FACTORS:
        values = [row for row in records if row["factor"] == factor]
        summary = {
            "comparisons": len(values),
            "median_all_coordinate_sign_agreement": float(np.median(
                [row["all_coordinate_sign_agreement"] for row in values]
            )),
            "median_low_amplitude_sign_agreement": float(np.median(
                [row["low_amplitude_half_sign_agreement"] for row in values]
            )),
            "median_high_amplitude_sign_agreement": float(np.median(
                [row["high_amplitude_half_sign_agreement"] for row in values]
            )),
            "median_weighted_same_sign_overlap": float(np.median(
                [row["amplitude_weighted_same_sign_overlap"] for row in values]
            )),
            "median_symmetric_relative_mse": float(np.median(
                [row["symmetric_relative_mse"] for row in values]
            )),
        }
        summary["candidate_repeatable"] = (
            summary["median_all_coordinate_sign_agreement"] >= EFFECT_SIGN_GATE
            and summary["median_symmetric_relative_mse"] <= EFFECT_RELATIVE_MSE_MAX
        )
        by_factor[factor] = summary
    result = {
        "rows": len(rows),
        "coordinates": DIMENSION,
        "channels": list(CHANNELS),
        "records": len(records),
        "by_factor": by_factor,
        "thresholds": {
            "sign_agreement_min": EFFECT_SIGN_GATE,
            "symmetric_relative_mse_max": EFFECT_RELATIVE_MSE_MAX,
        },
        "claim_boundary": "unit17/unit18 observational repeatability; no prospective or causal claim",
    }
    write_rows(OUT / "analysis/condition_effect_records.jsonl", records)
    save(OUT / "analysis/condition_geography_summary.json", result)
    return matrix, rows, result


def cross_unit_prediction(
    derivative: np.ndarray,
    index: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    _meta, lockbox_cells = passport.load_cells("c5481_qwen4b_fp16_directional_derivative")
    records = []
    coordinate_rows = []
    coordinate_values = []
    for train_unit, test_unit in ((17, 18), (18, 17)):
        train_indices = [i for i, row in enumerate(index) if int(row["unit"]) == train_unit]
        test_indices = [i for i, row in enumerate(index) if int(row["unit"]) == test_unit]
        train_rows = [index[i] for i in train_indices]
        for source_index in range(3):
            for probe in range(6):
                for target_index in range(3):
                    train = derivative[train_indices, source_index, probe, target_index].astype(np.float64)
                    components = fit_components(train, train_rows)
                    lockbox_global = lockbox_cells[(source_index, probe, target_index)][0].mean(
                        axis=0, dtype=np.float64
                    )
                    errors = {model: [] for model in MODELS}
                    for actual_index in test_indices:
                        row = index[actual_index]
                        actual = derivative[actual_index, source_index, probe, target_index].astype(np.float64)
                        predictions = {"lockbox_global": lockbox_global}
                        predictions.update({model: predict(components, row, model)
                                            for model in MODELS if model != "lockbox_global"})
                        for model, prediction in predictions.items():
                            errors[model].append(np.square(actual - prediction, dtype=np.float64))
                            records.append({
                                "train_unit": train_unit,
                                "test_unit": test_unit,
                                "case_id": row["case_id"],
                                "family": row["family"],
                                "language": row["language"],
                                "surface": row["surface"],
                                "state": int(row["state"]),
                                "source_index": source_index,
                                "probe": probe,
                                "target_index": target_index,
                                "model": model,
                                "relative_mse": relative_mse(actual, prediction),
                                "sign_agreement": float(np.mean(actual * prediction > 0)),
                            })
                    for model in MODELS:
                        coordinate_values.append(np.mean(np.stack(errors[model]), axis=0).astype(np.float32))
                        coordinate_rows.append({
                            "train_unit": train_unit,
                            "test_unit": test_unit,
                            "source_index": source_index,
                            "probe": probe,
                            "target_index": target_index,
                            "model": model,
                            "channel": "mean_squared_prediction_error",
                        })
    summaries = {}
    for model in MODELS:
        values = [row for row in records if row["model"] == model]
        by_direction = {}
        for train_unit, test_unit in ((17, 18), (18, 17)):
            selected = [row for row in values if row["train_unit"] == train_unit]
            by_direction[f"{train_unit}_to_{test_unit}"] = {
                "median_relative_mse": float(np.median([row["relative_mse"] for row in selected])),
                "median_sign_agreement": float(np.median([row["sign_agreement"] for row in selected])),
            }
        by_target = {
            str(target): float(np.median([row["relative_mse"] for row in values
                                          if row["target_index"] == target]))
            for target in range(3)
        }
        summaries[model] = {
            "records": len(values),
            "median_relative_mse": float(np.median([row["relative_mse"] for row in values])),
            "median_sign_agreement": float(np.median([row["sign_agreement"] for row in values])),
            "by_direction": by_direction,
            "relative_mse_by_target": by_target,
        }
    recent = summaries["recent_global"]["median_relative_mse"]
    candidates = ("main_context", "main_all", "pair_context")
    best = min(candidates, key=lambda model: summaries[model]["median_relative_mse"])
    improvement = (recent - summaries[best]["median_relative_mse"]) / (recent + EPS)
    both_directions = all(
        summaries[best]["by_direction"][direction]["median_relative_mse"]
        < summaries["recent_global"]["by_direction"][direction]["median_relative_mse"]
        for direction in ("17_to_18", "18_to_17")
    )
    selected = best if improvement >= SELECTION_IMPROVEMENT_MIN and both_directions else "recent_global"
    result = {
        "records": len(records),
        "coordinate_error_rows": len(coordinate_rows),
        "models": summaries,
        "selection": {
            "best_conditional_model": best,
            "relative_improvement_over_recent_global": improvement,
            "both_directions_improved": both_directions,
            "minimum_improvement": SELECTION_IMPROVEMENT_MIN,
            "selected_for_prospective": selected,
            "fallback_used": selected == "recent_global",
        },
        "claim_boundary": "exploratory bidirectional unit17/unit18 model selection",
    }
    write_rows(OUT / "analysis/cross_unit_prediction_records.jsonl", records)
    save(OUT / "analysis/cross_unit_prediction_summary.json", result)
    return np.stack(coordinate_values), coordinate_rows, result


def publish_asset(
    dataset_id: str,
    title: str,
    matrix: np.ndarray,
    rows: list[dict],
    schema: str,
    semantics: str,
    boundary: str,
) -> dict:
    binary = VIS / f"{dataset_id}.float32.npy"
    output = atlas.create_binary(binary.name, matrix.shape[0], matrix.shape[1], np.float32)
    output[:] = matrix
    output.flush()
    atlas.close_memmap(output)
    return atlas.write_metadata(
        dataset_id, title, binary, rows, "Qwen3-4B-FP16", schema,
        "exploratory derived", boundary, semantics,
        {"all_physical_coordinates": True, "no_coordinate_selection": True},
    )


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 残差条件地理图与双向简单预测器观察（{CAMPAIGN}） [{stamp}]

**测试原理、材料与用例。** 本期不运行模型，使用 Phase2326 已发布的 unit17/18 共 128 条、`3×6×3×2560` 完整方向响应与“实际减冻结 lockbox 全局均值”残差。第一部分分别在 unit17、unit18 内计算语言、表面、状态和八个语言族水平相对本单元均值的逐坐标效应，再检查同一坐标是否同号复现；低幅和高幅坐标按每个比较内部的幅值中位数分开，不删除任何坐标。第二部分执行 17→18 与 18→17 双向观察，比较旧 lockbox 全局均值、训练单元新全局均值、语言+表面+状态主效应、再加语言族主效应、以及三个上下文因素两两交互五个简单预测器。一个例子是用 unit17 中英文、叙事/转述和状态差异的完整 2560 坐标效应，预测 unit18 同条件新词汇响应。

$$
C_{{u,f=a}}=\mathbb E[D\mid u,f=a]-\mathbb E[D\mid u],\qquad
E_{{rep}}=\frac{{\lVert C_{{17}}-C_{{18}}\rVert_2^2}}{{(\lVert C_{{17}}\rVert_2^2+\lVert C_{{18}}\rVert_2^2)/2+\varepsilon}}.
$$

$$
\widehat D=\mu+\sum_f C_f+\sum_{{f<g}}C_{{fg}},\qquad
C_{{fg}}=\mathbb E[D\mid f,g]-\mu-C_f-C_g.
$$

**结果汇总与观察门。** 条件地理图 `{json.dumps(result['condition_geography'], ensure_ascii=False)}`。双向预测 `{json.dumps(result['cross_unit_prediction'], ensure_ascii=False)}`。条件效应仅在全坐标同号率不低于 `0.65` 且对称相对 MSE 不高于 `1.0` 时登记为可重复候选；条件预测器必须在两个方向都优于训练单元全局均值且总体至少改善 `3%`，才能进入前瞻，否则冻结回退模型。发布资产 `{json.dumps(result['datasets'], ensure_ascii=False)}`，验证 `{json.dumps(result['verification'], ensure_ascii=False)}`，前端离线构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`；未连接本地客户端。脚本 `tests/glm5/phase2327_c5841_c5920_residual_condition_geography.py`，结果 `tests/glm5/result/phase2327_c5841_c5920_residual_condition_geography`。

**分析、理论进展、硬伤与结论。** 这些结果只描述残差在哪些条件下重复，不把语言、表面或状态效应称为模型内置模块。语言对比混有 tokenizer 与长度，表面对比混有具体措辞，状态对比可能混有答案方向，语言族对比混有模板。双向 unit 观察仍使用同一材料生成器，属于模型选择而非独立确认。所有效应均是样本均值，但输出保留每个物理坐标，低值参数没有被 Top-K 丢弃。理论主体仍为“条件化输出场闭合理论”；本期只产生一个可前瞻失败的简单条件预测器和误差图谱，不新增理论名称。下一步用 unit17+18 冻结拟合，在完全未使用的 unit19+20 上重新运行 Qwen3-4B FP16 并裁决。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((P2326 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2326 is not authorized")
    config = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "frozen_before_field_read": True,
        "units": list(UNITS),
        "factors": list(FACTORS),
        "models": list(MODELS),
        "channels": list(CHANNELS),
        "effect_repeatability": {
            "sign_agreement_min": EFFECT_SIGN_GATE,
            "symmetric_relative_mse_max": EFFECT_RELATIVE_MSE_MAX,
        },
        "prospective_selection": {
            "minimum_improvement_over_recent_global": SELECTION_IMPROVEMENT_MIN,
            "both_directions_must_improve": True,
            "fallback": "recent_global",
        },
        "coordinate_policy": "all 2560 coordinates; low/high halves reported; no Top-K or projection",
    }
    save(OUT / "config/frozen_observation_contract.json", config)
    atlas.PHASE = PHASE
    atlas.CAMPAIGN = CAMPAIGN
    atlas.OUT = OUT
    derivative, residual, index = load_fields()
    condition_matrix, condition_rows, condition_summary = condition_geography(residual, index)
    error_matrix, error_rows, prediction_summary = cross_unit_prediction(derivative, index)
    datasets = [
        publish_asset(
            "c5841_qwen4b_fp16_residual_condition_geography",
            "Qwen3-4B FP16 unit17-unit18 residual condition geography",
            condition_matrix, condition_rows,
            "full_coordinate_residual_condition_geography_v1",
            "unit-specific factor effect, replication, and disagreement in every coordinate",
            "unit17/unit18 bilingual natural panels; exploratory observation",
        ),
        publish_asset(
            "c5842_qwen4b_fp16_cross_unit_prediction_error",
            "Qwen3-4B FP16 cross-unit coordinate prediction errors",
            error_matrix, error_rows,
            "full_coordinate_cross_unit_prediction_error_v1",
            "mean squared prediction error in every physical coordinate",
            "bidirectional unit17-to-18 and unit18-to-17 exploratory model comparison",
        ),
    ]
    verification = [atlas.verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id") for row in verification):
        raise RuntimeError(("asset_verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    atlas.close_memmap(derivative)
    atlas.close_memmap(residual)
    checks = {
        "parent_authorized": True,
        "config_frozen_before_field_read": True,
        "all_2560_coordinates": condition_matrix.shape[1] == DIMENSION and error_matrix.shape[1] == DIMENSION,
        "all_four_factors": set(condition_summary["by_factor"]) == set(FACTORS),
        "both_prediction_directions": all(
            set(value["by_direction"]) == {"17_to_18", "18_to_17"}
            for value in prediction_summary["models"].values()
        ),
        "all_assets_verified": all(all(value for key, value in row.items() if key != "id")
                                       for row in verification),
        "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
        "frontend_build_passed": build["passed"],
        "no_client_connection": not build["browser_or_client_connection"],
        "no_coordinate_selection": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed",
        "config": config,
        "condition_geography": condition_summary,
        "cross_unit_prediction": prediction_summary,
        "datasets": [atlas.serializable(row) for row in datasets],
        "verification": verification,
        "catalog": catalog,
        "frontend_build": build,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Observational model selection only; prospective qualification is deferred.",
        "next_authorization": "Freeze the selected model and test unseen units 19 and 20.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
