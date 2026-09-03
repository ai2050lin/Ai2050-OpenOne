#!/usr/bin/env python3
"""Fit and adjudicate simple same-coordinate local-update operators on the Qwen4B event field."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2398 = RESULT / "phase2398_c22001_c22320_qwen4b_event_fullfield"
OUT = RESULT / "phase2399_c22321_c22640_local_update_operator_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2399
CAMPAIGN = "C22321-C22640"
Q_UPDATES = 36  # embedding->block0 and block0->...->block35; final norm is not treated as a residual update.
OPERATORS = (
    "constant", "shared_diagonal_affine", "family_offset", "factor_additive_offset",
    "condition_offset", "family_diagonal_affine", "wrong_family_offset", "coordinate_permuted_condition",
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx = x.mean(0); my = y.mean(0)
    centered = x - mx
    variance = np.mean(centered * centered, axis=0)
    slope = np.mean(centered * (y - my), axis=0) / np.maximum(variance, 1e-8)
    slope = np.clip(slope, -8.0, 8.0)
    return slope, my - slope * mx


def group_means(y: np.ndarray, rows: list[dict], indices: np.ndarray, key_names: tuple[str, ...]) -> dict[tuple, np.ndarray]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for local, source_index in enumerate(indices.tolist()):
        key = tuple(rows[source_index].get(name) for name in key_names)
        groups[key].append(local)
    return {key: y[local_indices].mean(0) for key, local_indices in groups.items()}


def predict_group(means: dict[tuple, np.ndarray], rows: list[dict], indices: np.ndarray,
                  keys: tuple[str, ...], fallback: np.ndarray) -> np.ndarray:
    return np.stack([means.get(tuple(rows[index].get(name) for name in keys), fallback) for index in indices])


def split_indices(task: str, rows: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if task == "selection":
        train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
        tests = {
            "confirmation": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "confirmation"], dtype=np.int64),
            "fresh_unit_lockbox": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_unit_lockbox"], dtype=np.int64),
        }
    else:
        train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
        tests = {
            "one_step_confirmation": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "confirmation"], dtype=np.int64),
            "one_step_fresh_lockbox": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_unit_lockbox"], dtype=np.int64),
            "two_step_exploratory": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "exploratory_composition"], dtype=np.int64),
            "two_step_confirmation": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "composition_confirmation"], dtype=np.int64),
            "two_step_fresh_lockbox": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_composition_lockbox"], dtype=np.int64),
        }
    if not len(train) or any(not len(value) for value in tests.values()):
        raise RuntimeError((task, len(train), {key: len(value) for key, value in tests.items()}))
    return train, tests


def update_accumulator(acc: dict, prediction: np.ndarray, truth: np.ndarray, constant: np.ndarray) -> None:
    error = truth - prediction
    base_error = truth - constant
    acc["sse"] += float(np.sum(error * error, dtype=np.float64))
    acc["constant_sse"] += float(np.sum(base_error * base_error, dtype=np.float64))
    acc["abs"] += float(np.sum(np.abs(error), dtype=np.float64))
    acc["sign"] += int(np.sum(np.signbit(prediction) == np.signbit(truth)))
    acc["dot"] += float(np.sum(prediction * truth, dtype=np.float64))
    acc["pred_sq"] += float(np.sum(prediction * prediction, dtype=np.float64))
    acc["true_sq"] += float(np.sum(truth * truth, dtype=np.float64))
    acc["count"] += truth.size


def finish_accumulator(acc: dict) -> dict:
    return {
        "coordinates": int(acc["count"]),
        "mse": acc["sse"] / acc["count"],
        "mae": acc["abs"] / acc["count"],
        "gain_vs_constant": 1.0 - acc["sse"] / max(acc["constant_sse"], 1e-30),
        "sign_agreement": acc["sign"] / acc["count"],
        "global_cosine": acc["dot"] / max(math.sqrt(acc["pred_sq"] * acc["true_sq"]), 1e-30),
    }


def analyze_task(task: str, field_path: Path, rows: list[dict]) -> dict:
    field = np.load(field_path, mmap_mode="r")
    if field.shape[0] != len(rows) or field.shape[1] != 38 or field.shape[-1] != 2560:
        raise RuntimeError((task, field.shape, len(rows)))
    train, tests = split_indices(task, rows)
    event_count, dimension = field.shape[2], field.shape[3]
    factor_keys = ("family", "language", "surface", "direction", "query_role", "target_candidate_slot") if task == "selection" else ("family", "language", "surface", "direction", "target_candidate_slot")
    condition_keys = factor_keys
    accumulators = {(split, operator): defaultdict(float) for split in tests for operator in OPERATORS}
    slice_metrics: list[dict] = []
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    discovery_mean = np.lib.format.open_memmap(OUT / f"derived/{task}_update_discovery_mean.float32.npy", mode="w+", dtype=np.float32,
                                                shape=(Q_UPDATES, event_count, dimension))
    discovery_rms = np.lib.format.open_memmap(OUT / f"derived/{task}_update_discovery_rms.float32.npy", mode="w+", dtype=np.float32,
                                               shape=(Q_UPDATES, event_count, dimension))
    discovery_sign = np.lib.format.open_memmap(OUT / f"derived/{task}_update_discovery_sign_stability.float32.npy", mode="w+", dtype=np.float32,
                                                shape=(Q_UPDATES, event_count, dimension))
    lock_mean = np.lib.format.open_memmap(OUT / f"derived/{task}_update_lockbox_mean.float32.npy", mode="w+", dtype=np.float32,
                                          shape=(Q_UPDATES, event_count, dimension))
    for qpoint in range(Q_UPDATES):
        for event in range(event_count):
            x_all = np.asarray(field[:, qpoint, event, :], dtype=np.float32)
            y_all = np.asarray(field[:, qpoint + 1, event, :], dtype=np.float32) - x_all
            x_train, y_train = x_all[train], y_all[train]
            constant = y_train.mean(0)
            discovery_mean[qpoint, event] = constant
            discovery_rms[qpoint, event] = np.sqrt(np.mean(y_train * y_train, axis=0))
            discovery_sign[qpoint, event] = np.abs(np.mean(np.sign(y_train), axis=0))
            principal_lock = tests["fresh_unit_lockbox"] if task == "selection" else tests["one_step_fresh_lockbox"]
            lock_mean[qpoint, event] = y_all[principal_lock].mean(0)

            shared_a, shared_b = fit_affine(x_train, y_train)
            family_mean = group_means(y_train, rows, train, ("family",))
            condition_mean = group_means(y_train, rows, train, condition_keys)
            factor_effects = {}
            for factor in factor_keys:
                means = group_means(y_train, rows, train, (factor,))
                factor_effects[factor] = {key: value - constant for key, value in means.items()}
            family_affine = {}
            families = sorted({rows[index]["family"] for index in train})
            for family in families:
                local = np.asarray([local for local, source in enumerate(train) if rows[source]["family"] == family], dtype=np.int64)
                family_affine[family] = fit_affine(x_train[local], y_train[local])
            family_shift = {families[index]: families[(index + 1) % len(families)] for index in range(len(families))}

            for split, indices in tests.items():
                truth = y_all[indices]
                base = np.broadcast_to(constant, truth.shape)
                family_prediction = predict_group(family_mean, rows, indices, ("family",), constant)
                predictions: dict[str, np.ndarray] = {
                    "constant": base,
                    "shared_diagonal_affine": x_all[indices] * shared_a + shared_b,
                    "family_offset": family_prediction,
                    "condition_offset": predict_group(condition_mean, rows, indices, condition_keys, constant),
                }
                additive = np.broadcast_to(constant, truth.shape).copy()
                for local, source in enumerate(indices):
                    for factor in factor_keys:
                        additive[local] += factor_effects[factor].get((rows[source].get(factor),), 0.0)
                predictions["factor_additive_offset"] = additive
                predictions["family_diagonal_affine"] = np.stack([
                    x_all[source] * family_affine[rows[source]["family"]][0] + family_affine[rows[source]["family"]][1]
                    for source in indices
                ])
                predictions["wrong_family_offset"] = np.stack([
                    family_mean[(family_shift[rows[source]["family"]],)] for source in indices
                ])
                predictions["coordinate_permuted_condition"] = np.roll(predictions["condition_offset"], 1, axis=1)
                for operator, prediction in predictions.items():
                    update_accumulator(accumulators[(split, operator)], prediction, truth, base)
                    local_acc = defaultdict(float); update_accumulator(local_acc, prediction, truth, base)
                    slice_metrics.append({"task": task, "split": split, "operator": operator, "qpoint": qpoint,
                                          "event_index": event, **finish_accumulator(local_acc)})
                del truth, base, predictions, additive
            del x_all, y_all, x_train, y_train
        discovery_mean.flush(); discovery_rms.flush(); discovery_sign.flush(); lock_mean.flush()
        print(f"[phase2399 {task}] update {qpoint + 1}/{Q_UPDATES}", flush=True)
    aggregate = {split: {operator: finish_accumulator(accumulators[(split, operator)]) for operator in OPERATORS} for split in tests}
    confirmation_split = "confirmation" if task == "selection" else "one_step_confirmation"
    candidates = [operator for operator in OPERATORS if operator not in ("wrong_family_offset", "coordinate_permuted_condition")]
    chosen = max(candidates, key=lambda operator: aggregate[confirmation_split][operator]["gain_vs_constant"])
    result = {
        "task": task, "field_shape": list(field.shape), "train_rows": len(train),
        "test_rows": {key: len(value) for key, value in tests.items()}, "operators": list(OPERATORS),
        "aggregate": aggregate, "confirmation_split": confirmation_split,
        "frozen_operator": chosen,
        "frozen_operator_metrics": {split: aggregate[split][chosen] for split in tests},
        "coordinate_arrays": {
            "discovery_mean": str(OUT / f"derived/{task}_update_discovery_mean.float32.npy"),
            "discovery_rms": str(OUT / f"derived/{task}_update_discovery_rms.float32.npy"),
            "discovery_sign_stability": str(OUT / f"derived/{task}_update_discovery_sign_stability.float32.npy"),
            "lockbox_mean": str(OUT / f"derived/{task}_update_lockbox_mean.float32.npy"),
        },
    }
    write_rows(OUT / f"analysis/{task}_slice_metrics.jsonl", slice_metrics)
    save(OUT / f"analysis/{task}_operator_summary.json", result)
    discovery_mean.flush(); discovery_rms.flush(); discovery_sign.flush(); lock_mean.flush()
    close(discovery_mean); close(discovery_rms); close(discovery_sign); close(lock_mean); close(field)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    summary = result["summary"]
    text = rf"""

## Phase {PHASE}: 样本条件局部层更新律与简单算子全坐标竞赛（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 直接在Phase2398每个语义事件、每个残差block转换、每个物理坐标上定义局部更新；final norm只作输出接口，不混入36个残差更新。选择任务只用unit0–3 discovery拟合，在unit4–5 confirmation冻结一个全局算子类型，再一次性报告unit6–7 fresh lockbox；组合任务只用一步discovery拟合，在一步confirmation冻结算子，再测试一步fresh和未参与拟合的两步组合。竞赛包含层/事件常量、全局同坐标对角仿射、族条件均值、反平衡因子相加、完整条件单元均值、族内同坐标对角仿射；错族与坐标循环置换为负对照。所有误差在全部坐标累计，不以Top-K坐标得分代替。

$$U_{{r,q,e,j}}=H_{{r,q+1,e,j}}-H_{{r,q,e,j}},$$

$$\widehat U^{{diag}}_j=a_jH_j+b_j,\qquad
G=1-\frac{{\sum_{{r,q,e,j}}(U-\widehat U)^2}}{{\sum_{{r,q,e,j}}(U-\bar U_{{train}})^2}}.$$

**结果汇总。** 冻结结果 `{json.dumps(summary, ensure_ascii=False)}`；选择任务全算子 `{json.dumps(result['selection']['aggregate'], ensure_ascii=False)}`；组合任务全算子 `{json.dumps(result['composition']['aggregate'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2399_c22321_c22640_local_update_operator_atlas.py`；逐qpoint×事件×算子指标、全坐标update均值/RMS/符号稳定度与final位于 `tests/glm5/result/phase2399_c22321_c22640_local_update_operator_atlas`。Phase2398原始全场保持不变。

**理论进展。** 这是第一次要求候选规律预测“下一层同坐标会怎样变化”，而非仅从状态读出标签。若条件算子只在confirmation好而fresh失败，它是unit过拟合；若一步算子不能推广两步，它不是组合律；若错族/坐标置换相当，则条件坐标解释没有特异性。无论结果正负，每个坐标的update均值、能量和跨unit符号稳定度均被保留为下一Phase机制护照的原始拼图。

**问题硬伤与结论。** 对角仿射和条件均值是外部拟合器，不等于Transformer实际显式执行该方程；常量基线已吸收强烈的通用层变换，因此微小正增益也不能升级为齿轮。完整条件单元在discovery只有4个unit重复，容易估计不稳；fresh锁箱是关键。组合prompt在query以前与一步版本高度共享，局部预测必须按事件报告，不能用事实段的平凡相似掩盖query/答案段失败。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    selection_rows = read_rows(P2398 / "index/selection_rows.jsonl")
    composition_rows = read_rows(P2398 / "index/composition_rows.jsonl")
    selection = analyze_task("selection", P2398 / "raw/selection_event_field.float16.npy", selection_rows)
    composition = analyze_task("composition", P2398 / "raw/composition_event_field.float16.npy", composition_rows)
    summary = {
        "selection_frozen_operator": selection["frozen_operator"],
        "selection_confirmation": selection["frozen_operator_metrics"]["confirmation"],
        "selection_fresh_lockbox": selection["frozen_operator_metrics"]["fresh_unit_lockbox"],
        "composition_frozen_operator": composition["frozen_operator"],
        "composition_one_step_confirmation": composition["frozen_operator_metrics"]["one_step_confirmation"],
        "composition_one_step_fresh": composition["frozen_operator_metrics"]["one_step_fresh_lockbox"],
        "composition_two_step_confirmation": composition["frozen_operator_metrics"]["two_step_confirmation"],
        "composition_two_step_fresh": composition["frozen_operator_metrics"]["two_step_fresh_lockbox"],
    }
    checks = {
        "selection_all_coordinates": selection["field_shape"] == [2048, 38, 8, 2560],
        "composition_all_coordinates": composition["field_shape"] == [1024, 38, 12, 2560],
        "frozen_before_lockbox_claim": selection["confirmation_split"] == "confirmation" and composition["confirmation_split"] == "one_step_confirmation",
        "negative_controls": all(control in OPERATORS for control in ("wrong_family_offset", "coordinate_permuted_condition")),
        "finite": all(math.isfinite(metric["gain_vs_constant"]) for task in (selection, composition) for split in task["aggregate"].values() for metric in split.values()),
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "selection": selection, "composition": composition,
              "summary": summary, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
