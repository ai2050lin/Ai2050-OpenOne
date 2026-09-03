#!/usr/bin/env python3
"""Prospectively test frozen simple response models on unseen units 19 and 20."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2315 = RESULT / "phase2315_c5041_c5100_active_response_contract"
P2325 = RESULT / "phase2325_c5721_c5800_qwen4b_fp16_large_family_confirmation"
P2327 = RESULT / "phase2327_c5841_c5920_residual_condition_geography"
OUT = RESULT / "phase2328_c5921_c6000_unseen_condition_prediction"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
ROWS_PATH = P2315 / "material/natural_active_response_bilingual.jsonl"
PHASE = 2328
CAMPAIGN = "C5921-C6000"
TRAIN_UNITS = (17, 18)
TEST_UNITS = (19, 20)
MODELS = ("lockbox_global", "recent_global", "main_context", "main_all", "pair_context")
EPS = 1e-12
GATES = {
    "selected_relative_mse_max": 0.45,
    "selected_over_lockbox_ratio_max": 1.0,
    "selected_sign_agreement_min": 0.75,
    "pair_superposition_relative_mse_max": 0.05,
    "even_to_odd_l2_max": 0.30,
}

sys.path.insert(0, str(TESTS))
import phase1332_bf16_utils as model_base  # noqa: E402
import phase2318_c5241_c5320_crossmodel_directional_topology as cross  # noqa: E402
import phase2322_c5521_c5600_full_coordinate_reuse_passports as passport  # noqa: E402
import phase2327_c5841_c5920_residual_condition_geography as geography  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_training() -> tuple[np.ndarray, list[dict]]:
    metadata = json.loads(
        (VIS / "c5801_qwen4b_fp16_large_confirmation_derivative.json").read_text(encoding="utf-8")
    )
    values = np.load(VIS / Path(metadata["binary_url"]).name, mmap_mode="r")
    if list(values.shape) != [6912, 2560]:
        raise RuntimeError(("training_shape", values.shape))
    rows = cross.read_rows(P2325 / "index/active_rows.jsonl")
    if len(rows) != 128 or {int(row["unit"]) for row in rows} != set(TRAIN_UNITS):
        raise RuntimeError(("training_index", len(rows), sorted({row["unit"] for row in rows})))
    return values.reshape(128, 3, 6, 3, 2560), rows


def prospective_prediction(
    test_path: Path,
    test_rows: list[dict],
    selected_model: str,
) -> dict:
    training, training_rows = load_training()
    test = np.load(test_path, mmap_mode="r")
    if list(test.shape) != [128, 3, 6, 3, 2560]:
        raise RuntimeError(("test_shape", test.shape))
    _meta, lockbox_cells = passport.load_cells("c5481_qwen4b_fp16_directional_derivative")
    residual_path = OUT / "raw/selected_model_residual.float32.npy"
    residual = np.lib.format.open_memmap(
        residual_path, mode="w+", dtype=np.float32, shape=test.shape,
    )
    records = []
    coordinate_errors = {model: [] for model in MODELS}
    for source_index in range(3):
        for probe in range(6):
            for target_index in range(3):
                train_cell = training[:, source_index, probe, target_index].astype(np.float64)
                components = geography.fit_components(train_cell, training_rows)
                lockbox_global = lockbox_cells[(source_index, probe, target_index)][0].mean(
                    axis=0, dtype=np.float64
                )
                error_sums = {model: np.zeros(2560, dtype=np.float64) for model in MODELS}
                for row_index, row in enumerate(test_rows):
                    actual = test[row_index, source_index, probe, target_index].astype(np.float64)
                    predictions = {"lockbox_global": lockbox_global}
                    predictions.update({
                        model: geography.predict(components, row, model)
                        for model in MODELS if model != "lockbox_global"
                    })
                    residual[row_index, source_index, probe, target_index] = (
                        actual - predictions[selected_model]
                    ).astype(np.float32)
                    for model, prediction in predictions.items():
                        error = np.square(actual - prediction, dtype=np.float64)
                        error_sums[model] += error
                        records.append({
                            "case_id": row["case_id"],
                            "family": row["family"],
                            "language": row["language"],
                            "surface": row["surface"],
                            "state": int(row["state"]),
                            "unit": int(row["unit"]),
                            "source_index": source_index,
                            "probe": probe,
                            "target_index": target_index,
                            "model": model,
                            "relative_mse": geography.relative_mse(actual, prediction),
                            "sign_agreement": float(np.mean(actual * prediction > 0)),
                        })
                for model in MODELS:
                    coordinate_errors[model].append((
                        source_index, probe, target_index,
                        (error_sums[model] / len(test_rows)).astype(np.float32),
                    ))
                residual.flush()
    residual.flush()
    geography.atlas.close_memmap(residual)
    geography.atlas.close_memmap(training)
    geography.atlas.close_memmap(test)
    summaries = {}
    for model in MODELS:
        values = [row for row in records if row["model"] == model]
        summaries[model] = {
            "records": len(values),
            "median_relative_mse": float(np.median([row["relative_mse"] for row in values])),
            "median_sign_agreement": float(np.median([row["sign_agreement"] for row in values])),
            "relative_mse_by_target": {
                str(target): float(np.median([row["relative_mse"] for row in values
                                              if row["target_index"] == target]))
                for target in range(3)
            },
            "relative_mse_by_unit": {
                str(unit): float(np.median([row["relative_mse"] for row in values
                                            if row["unit"] == unit]))
                for unit in TEST_UNITS
            },
        }
    selected_value = summaries[selected_model]["median_relative_mse"]
    lockbox_value = summaries["lockbox_global"]["median_relative_mse"]
    result = {
        "records": len(records),
        "rows": len(test_rows),
        "cells": 54,
        "coordinates": 2560,
        "selected_model": selected_model,
        "models": summaries,
        "selected_over_lockbox_ratio": selected_value / (lockbox_value + EPS),
        "selected_residual": str(residual_path.relative_to(ROOT)),
        "claim_boundary": "frozen training units17-18 predicting unseen units19-20",
    }
    cross.write_rows(OUT / "analysis/prospective_prediction_records.jsonl", records)
    save(OUT / "analysis/prospective_prediction_summary.json", result)
    return result


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: unit19–20 未见词汇的冻结条件预测前瞻裁决（{CAMPAIGN}） [{stamp}]

**测试原理、冻结对象与测试用例。** Phase2327 观察到四类条件效应均未在 unit17/18 间重复，三个条件模型也没有双向改善，因此依照预注册分支把 `recent_global` 冻结为回退模型；旧 `fresh_lockbox` 全局均值仅作为历史对照，不事后改为主模型。本期训练数据固定为 unit17+18 的 128 条完整响应，测试数据固定为从未主动测量的 unit19+20 共 128 条，八族各 16 条，覆盖中英、叙事/转述和两个状态。Qwen3-4B 以本地非量化 FP16 单独加载，继续使用 3 个源深度、4 个基方向、2 个成对方向、3 个目标层与 1% 剂量；候选行为和自由生成分账。每个预测与残差保留全部 2560 个物理坐标。

$$
\widehat D_{{recent}}=\frac1{{128}}\sum_{{i\in\{{17,18\}}}}D_i,\qquad
E=\frac{{\lVert D_{{19,20}}-\widehat D\rVert_2^2}}{{\lVert D_{{19,20}}\rVert_2^2+\varepsilon}}.
$$

$$
Q_{{selected/control}}=\frac{{\operatorname{{median}}E_{{recent}}}}{{\operatorname{{median}}E_{{lockbox}}}}.
$$

**结果与冻结门。** 行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`。五模型前瞻结果 `{json.dumps(result['prediction'], ensure_ascii=False)}`。局部响应 `{json.dumps(result['functional_metrics'], ensure_ascii=False)}`。门与裁决 `{json.dumps(result['experimental_gates'], ensure_ascii=False)}`：主模型 MSE 不高于 `0.45`、相对 lockbox 对照比不高于 `1.0`、符号一致率不低于 `0.75`、成对误差不高于 `0.05`、偶/奇比不高于 `0.30`。失败只淘汰简单条件/回退预测路线，不否定 Phase2325 已确认的局部符号骨架。脚本 `tests/glm5/phase2328_c5921_c6000_unseen_condition_prediction.py`，结果 `tests/glm5/result/phase2328_c5921_c6000_unseen_condition_prediction`。

**分析、理论进展、硬伤与结论。** 本期是真正前瞻裁决，不用 unit19/20 重新选择模型。若回退模型失败而 lockbox 对照更好，说明邻近词汇单元均值不是稳定动力学基态；若两者都失败，则简单均值家族整体不足。即使条件模型偶然较优，也只能登记为未预选观察。所有预测仍基于随机方向局部响应，不是自然语言操作本身；训练和测试共享材料生成器，模型只有 Qwen3-4B FP16，行为自由生成也可能较弱。理论主体仍为“条件化输出场闭合理论”，本期只裁决简单条件模型能否预测新 HiddenState 响应幅值，不引入新理论名称。下一步发布实际响应和冻结主模型残差，清理重复原始场，然后判断该均值/主效应路线是否应结束。
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
    phase2327 = json.loads((P2327 / "analysis/final.json").read_text(encoding="utf-8"))
    if not phase2327["all_checks_passed"]:
        raise RuntimeError("Phase2327 is not authorized")
    selected_model = phase2327["cross_unit_prediction"]["selection"]["selected_for_prospective"]
    if selected_model != "recent_global":
        raise RuntimeError(("unexpected_frozen_branch", selected_model))
    all_rows = cross.read_rows(ROWS_PATH)
    selected_raw = [row for row in all_rows
                    if row["partition"] == "fresh_confirmation" and int(row["unit"]) in TEST_UNITS]
    selected_raw.sort(key=lambda row: row["design_index"])
    prior_ids = {row["case_id"] for row in cross.read_rows(P2325 / "index/active_rows.jsonl")}
    overlap = prior_ids & {row["case_id"] for row in selected_raw}
    family_counts = {family: sum(row["family"] == family for row in selected_raw)
                     for family in sorted({row["family"] for row in selected_raw})}
    strata = {
        f"{language}|{surface}|{state}": sum(
            row["language"] == language and row["surface"] == surface
            and int(row["state"]) == state for row in selected_raw
        )
        for language in ("en", "zh")
        for surface in ("narrative", "reported")
        for state in (0, 1)
    }
    config = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "frozen_before_model_load": True,
        "model": "Qwen3-4B",
        "precision": "float16_nonquantized",
        "train_units": list(TRAIN_UNITS),
        "test_units": list(TEST_UNITS),
        "selected_model": selected_model,
        "all_reported_models": list(MODELS),
        "test_rows": len(selected_raw),
        "family_counts": family_counts,
        "language_surface_state_counts": strata,
        "overlap_with_training": sorted(overlap),
        "gates": GATES,
        "coordinate_policy": "all 2560 physical coordinates; no Top-K, PCA, or projection",
    }
    if len(selected_raw) != 128 or set(family_counts.values()) != {16} or overlap:
        raise RuntimeError(("material_contract", len(selected_raw), family_counts, sorted(overlap)))
    if set(strata.values()) != {16}:
        raise RuntimeError(("strata_contract", strata))
    save(OUT / "config/frozen_prospective_contract.json", config)
    model = tokenizer = None
    try:
        model, tokenizer, device = model_base.load_model(
            "qwen3", dtype=torch.float16, use_8bit=False,
        )
        dtypes = model_base.parameter_dtype_counts(model)
        compiled = cross.compile_rows(tokenizer, selected_raw)
        cross.write_rows(OUT / "material/compiled_unseen_units.jsonl", compiled)
        scores = cross.sequence_scores(model, device, compiled, 32)
        cross.write_rows(OUT / "behavior/sequence_scores.jsonl", scores)
        free = cross.free_generation(model, tokenizer, device, compiled, 32)
        cross.write_rows(OUT / "behavior/free_generation.jsonl", free)
        behavior = cross.behavior_summary(scores, free)
        behavior["claim_boundary"] = "128-row unseen-unit descriptive behavior panel"
        save(OUT / "behavior/summary.json", behavior)
        field = cross.active_capture(model, device, compiled, OUT)
        index_rows = cross.read_rows(OUT / "index/active_rows.jsonl")
        for row in index_rows:
            row["partition"] = "fresh_confirmation"
        cross.write_rows(OUT / "index/active_rows.jsonl", index_rows)
        functional = cross.functional_metrics(
            ROOT / field["derivative"], ROOT / field["even"], ROOT / field["norms"],
            index_rows, "Qwen3-4B-FP16-unseen-units",
        )
        save(OUT / "analysis/functional_metrics.json", functional)
        prediction = prospective_prediction(ROOT / field["derivative"], index_rows, selected_model)
        selected = prediction["models"][selected_model]
        gates = {
            "selected_relative_mse": {
                "value": selected["median_relative_mse"],
                "threshold_max": GATES["selected_relative_mse_max"],
                "passed": selected["median_relative_mse"] <= GATES["selected_relative_mse_max"],
            },
            "selected_over_lockbox_ratio": {
                "value": prediction["selected_over_lockbox_ratio"],
                "threshold_max": GATES["selected_over_lockbox_ratio_max"],
                "passed": prediction["selected_over_lockbox_ratio"] <= GATES["selected_over_lockbox_ratio_max"],
            },
            "selected_sign_agreement": {
                "value": selected["median_sign_agreement"],
                "threshold_min": GATES["selected_sign_agreement_min"],
                "passed": selected["median_sign_agreement"] >= GATES["selected_sign_agreement_min"],
            },
            "pair_superposition_relative_mse": {
                "value": functional["median_pair_superposition_relative_mse"],
                "threshold_max": GATES["pair_superposition_relative_mse_max"],
                "passed": functional["median_pair_superposition_relative_mse"] <= GATES["pair_superposition_relative_mse_max"],
            },
            "even_to_odd_l2": {
                "value": functional["median_even_to_odd_l2"],
                "threshold_max": GATES["even_to_odd_l2_max"],
                "passed": functional["median_even_to_odd_l2"] <= GATES["even_to_odd_l2_max"],
            },
        }
        checks = {
            "parent_authorized": True,
            "frozen_branch_used": selected_model == "recent_global",
            "config_frozen_before_model_load": True,
            "fp16_nonquantized": dtypes.get("float16", 0) > 0,
            "all_128_unseen_rows": len(compiled) == 128,
            "no_training_overlap": not overlap,
            "balanced_eight_families": set(family_counts.values()) == {16},
            "balanced_language_surface_state": set(strata.values()) == {16},
            "all_coordinates": field["shape"][-1] == 2560,
            "all_models_reported": set(prediction["models"]) == set(MODELS),
            "all_34560_predictions": prediction["records"] == 34560,
            "experimental_failure_does_not_invalidate_execution": True,
        }
        result = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": "closed",
            "config": config,
            "parameter_dtypes": dtypes,
            "behavior": behavior,
            "field": field,
            "prediction": prediction,
            "functional_metrics": {
                "median_pair_superposition_relative_mse": functional["median_pair_superposition_relative_mse"],
                "median_even_to_odd_l2": functional["median_even_to_odd_l2"],
                "relative_response_topology": functional["relative_response_topology"],
                "claim_boundary": functional["claim_boundary"],
            },
            "experimental_gates": gates,
            "route_passed": all(value["passed"] for value in gates.values()),
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "strict_conclusion": (
                "The frozen fallback predictor passed all prospective gates."
                if all(value["passed"] for value in gates.values()) else
                "The frozen fallback predictor failed one or more gates; simple conditional amplitude prediction is not qualified."
            ),
            "next_authorization": "Publish exact-coordinate actual and selected-residual fields, then clean raw duplicates.",
        }
        save(final_path, result)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
