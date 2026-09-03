"""C1031-C1048 frozen full-coordinate prediction and composition tournament."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
FIELD_OUT = RESULT / "phase2248_c1017_c1030_qwen_natural_full_field"
OUT = RESULT / "phase2249_c1031_c1048_full_coordinate_prediction"
sys.path.insert(0, str(TESTS))

import phase2247_c1001_c1016_natural_flagship_contract as contract


PHASE = 2249
CAMPAIGNS = tuple(f"C{i}" for i in range(1031, 1049))
METHODS = ("M0_zero", "M1_shared_mean", "M2_family_mean", "M3_shared_affine",
           "M4_family_affine", "M5_shared_dual_ridge", "M6_family_dual_ridge")
FAMILY_METHODS = ("M2_family_mean", "M4_family_affine", "M6_family_dual_ridge")
SHARED_FOR = {"M2_family_mean": "M1_shared_mean", "M4_family_affine": "M3_shared_affine",
              "M6_family_dual_ridge": "M5_shared_dual_ridge"}
QPOINTS = (8, 16, 24, 26, 30, 32)
PREDICTION_CANDIDATES = OUT / "raw/fresh_prediction_candidates.float16.npy"
PREDICTIONS = OUT / "raw/fresh_selected_predictions.float16.npy"


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def pairs(index: list[dict], panel: str) -> list[dict]:
    groups = defaultdict(dict)
    for row in index:
        if row["panel"] != panel:
            continue
        key = (row["family"], row["language"], row["unit"], row["surface"],
               row.get("cell_id"))
        groups[key][bool(row["truth"])] = row
    output = []
    for key, values in sorted(groups.items()):
        if set(values) != {False, True}:
            raise RuntimeError(("unpaired_truth_cell", key, sorted(values)))
        false, true = values[False], values[True]
        output.append({
            "pair_id": false["case_id"].rsplit("_t0_", 1)[0], "family": false["family"],
            "language": false["language"], "unit": false["unit"], "surface": false["surface"],
            "partition": false["partition"], "false_index": false["hidden_index"],
            "true_index": true["hidden_index"], "cell_id": false.get("cell_id"),
            "depth": false.get("depth"), "shortcut": false.get("shortcut"),
            "verb_index": false.get("verb_index"), "outer_neg": false.get("outer_neg"),
            "inner_neg": false.get("inner_neg"),
        })
    return output


def arrays(field, rows: list[dict], q: int, role_i: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([field[row["false_index"], q, role_i] for row in rows]).astype(np.float32)
    y = np.stack([field[row["true_index"], q, role_i] - field[row["false_index"], q, role_i]
                  for row in rows]).astype(np.float32)
    return x, y


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xm = x.mean(axis=0); ym = y.mean(axis=0)
    xc = x - xm; yc = y - ym
    slope = np.sum(xc * yc, axis=0) / (np.sum(xc * xc, axis=0) + 1e-6)
    return slope.astype(np.float32), (ym - slope * xm).astype(np.float32)


def fit_dual(x: np.ndarray, y: np.ndarray) -> dict:
    xm = x.mean(axis=0); ym = y.mean(axis=0)
    xc = x - xm; yc = y - ym
    gram = xc @ xc.T
    ridge = 0.01 * float(np.trace(gram) / max(1, len(x))) + 1e-4
    alpha = np.linalg.solve(gram + ridge * np.eye(len(x), dtype=np.float32), yc)
    return {"xm": xm, "ym": ym, "xc": xc, "alpha": alpha, "ridge": ridge}


def predict_dual(model: dict, x: np.ndarray) -> np.ndarray:
    return (x - model["xm"]) @ model["xc"].T @ model["alpha"] + model["ym"]


def fit_cell_models(field, train: list[dict], q: int, role_i: int) -> dict:
    x_all, y_all = arrays(field, train, q, role_i)
    shared_affine = fit_affine(x_all, y_all)
    models = {
        "shared_mean": y_all.mean(axis=0), "shared_affine": shared_affine,
        "shared_dual": fit_dual(x_all, y_all), "family": {},
    }
    for family in contract.FAMILIES:
        subset = [row for row in train if row["family"] == family]
        x, y = arrays(field, subset, q, role_i)
        models["family"][family] = {
            "mean": y.mean(axis=0), "affine": fit_affine(x, y), "dual": fit_dual(x, y),
        }
    return models


def cell_predictions(models: dict, family: str, x: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fm = models["family"][family]
    shared = {
        "M0_zero": np.zeros_like(x),
        "M1_shared_mean": np.broadcast_to(models["shared_mean"], x.shape),
        "M3_shared_affine": x * models["shared_affine"][0] + models["shared_affine"][1],
        "M5_shared_dual_ridge": predict_dual(models["shared_dual"], x),
    }
    family_pred = {
        "M2_family_mean": np.broadcast_to(fm["mean"], x.shape),
        "M4_family_affine": x * fm["affine"][0] + fm["affine"][1],
        "M6_family_dual_ridge": predict_dual(fm["dual"], x),
    }
    wrong_family = contract.FAMILIES[(contract.FAMILIES.index(family) + 1) % len(contract.FAMILIES)]
    wrong = models["family"][wrong_family]
    wrong_pred = {
        "M2_family_mean": np.broadcast_to(wrong["mean"], x.shape),
        "M4_family_affine": x * wrong["affine"][0] + wrong["affine"][1],
        "M6_family_dual_ridge": predict_dual(wrong["dual"], x),
    }
    return {**shared, **family_pred}, wrong_pred


def metric_accumulator() -> dict:
    return {"absolute_error_sum": 0.0, "values": 0, "sign_accuracy_cells": []}


def add_metric(acc: dict, pred: np.ndarray, actual: np.ndarray, base: np.ndarray) -> None:
    acc["absolute_error_sum"] += float(np.sum(np.abs(pred - actual), dtype=np.float64))
    acc["values"] += int(actual.size)
    for i in range(len(actual)):
        changed = np.abs(actual[i]) >= (0.04 + 0.10 * np.abs(base[i]))
        if np.any(changed):
            acc["sign_accuracy_cells"].append(float(np.mean(np.sign(pred[i, changed]) == np.sign(actual[i, changed]))))


def finish_metric(acc: dict) -> dict:
    return {
        "mae": acc["absolute_error_sum"] / max(1, acc["values"]),
        "values": acc["values"],
        "median_coordinate_sign_accuracy": float(np.median(acc["sign_accuracy_cells"])) if acc["sign_accuracy_cells"] else 0.0,
        "sign_cells": len(acc["sign_accuracy_cells"]),
    }


def predictive_tournament(field, index: list[dict], behavior: dict) -> tuple[dict, list[dict], dict]:
    broad = pairs(index, "natural_broad")
    train = [row for row in broad if row["partition"] == "discovery"]
    panels = {
        "parent_confirmation": [row for row in broad if row["partition"] == "confirmation"],
        "parent_lockbox": [row for row in broad if row["partition"] == "lockbox"],
        "fresh_confirmation": [row for row in broad if row["partition"] == "fresh_confirmation"],
        "fresh_lockbox": [row for row in broad if row["partition"] == "fresh_lockbox"],
    }
    acc = defaultdict(metric_accumulator)
    wrong_acc = defaultdict(metric_accumulator)
    cell_acc = defaultdict(metric_accumulator)
    fresh = panels["fresh_confirmation"] + panels["fresh_lockbox"]
    fresh_index = {row["pair_id"]: i for i, row in enumerate(fresh)}
    PREDICTION_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    candidate_store = np.lib.format.open_memmap(
        PREDICTION_CANDIDATES, mode="w+", dtype=np.float16,
        shape=(len(FAMILY_METHODS), len(fresh), len(QPOINTS), len(contract.ROLES), field.shape[-1]),
    )
    qmap = {q: i for i, q in enumerate(QPOINTS)}
    for q in range(field.shape[1]):
        for role_i, _role in enumerate(contract.ROLES):
            models = fit_cell_models(field, train, q, role_i)
            for panel, panel_rows in panels.items():
                for family in contract.FAMILIES:
                    subset = [row for row in panel_rows if row["family"] == family]
                    if not subset:
                        continue
                    x, y = arrays(field, subset, q, role_i)
                    predicted, wrong = cell_predictions(models, family, x)
                    for method, values in predicted.items():
                        add_metric(acc[(panel, family, method)], values, y, x)
                        if panel == "parent_confirmation":
                            add_metric(cell_acc[(q, role_i, family, method)], values, y, x)
                    for method, values in wrong.items():
                        add_metric(wrong_acc[(panel, family, method)], values, y, x)
                    if q in qmap and panel.startswith("fresh"):
                        for method_i, method in enumerate(FAMILY_METHODS):
                            for row_i, row in enumerate(subset):
                                candidate_store[method_i, fresh_index[row["pair_id"]], qmap[q], role_i] = predicted[method][row_i]
            if (q * len(contract.ROLES) + role_i) % 12 == 0:
                print(f"[predictive] checkpoint={q} role={contract.ROLES[role_i]}", flush=True)
    candidate_store.flush(); close_mmap(candidate_store)
    metrics = {}
    for key, value in acc.items():
        metrics["|".join(key)] = finish_metric(value)
    wrong_metrics = {"|".join(key): finish_metric(value) for key, value in wrong_acc.items()}
    cell_metrics = {"|".join(map(str, key)): finish_metric(value) for key, value in cell_acc.items()}
    selection = {}
    strict = []
    for family in contract.FAMILIES:
        candidates = {}
        for method in FAMILY_METHODS:
            own = metrics[f"parent_confirmation|{family}|{method}"]
            shared = metrics[f"parent_confirmation|{family}|{SHARED_FOR[method]}"]
            wrong = wrong_metrics[f"parent_confirmation|{family}|{method}"]
            candidates[method] = {
                "mae": own["mae"], "shared_mae": shared["mae"], "wrong_family_mae": wrong["mae"],
                "gain_over_shared": (shared["mae"] - own["mae"]) / max(shared["mae"], 1e-9),
                "gain_over_wrong_family": (wrong["mae"] - own["mae"]) / max(wrong["mae"], 1e-9),
                "median_coordinate_sign_accuracy": own["median_coordinate_sign_accuracy"],
            }
        winner = min(FAMILY_METHODS, key=lambda method: (candidates[method]["mae"], FAMILY_METHODS.index(method)))
        checkpoint_mae = {}
        for q in QPOINTS:
            values = [cell_metrics[f"{q}|{role_i}|{family}|{winner}"]["mae"]
                      for role_i in range(len(contract.ROLES))]
            checkpoint_mae[q] = float(np.mean(values))
        causal_checkpoint = min(QPOINTS, key=lambda q: (checkpoint_mae[q], QPOINTS.index(q)))
        panel_results = {}
        for panel in panels:
            own = metrics[f"{panel}|{family}|{winner}"]
            shared = metrics[f"{panel}|{family}|{SHARED_FOR[winner]}"]
            wrong = wrong_metrics[f"{panel}|{family}|{winner}"]
            panel_results[panel] = {
                **own,
                "gain_over_shared": (shared["mae"] - own["mae"]) / max(shared["mae"], 1e-9),
                "gain_over_wrong_family": (wrong["mae"] - own["mae"]) / max(wrong["mae"], 1e-9),
            }
        required_behavior = [
            behavior["panels"][f"natural_broad|{family}|{partition}"]["dual_qualified"]
            for partition in ("discovery", "confirmation", "lockbox", "fresh_confirmation", "fresh_lockbox")
        ]
        behavior_ok = bool(all(required_behavior))
        passed = behavior_ok and all(
            values["gain_over_shared"] >= contract.PREDICTIVE_GATES["relative_mae_gain_over_shared"]
            and values["gain_over_wrong_family"] >= contract.PREDICTIVE_GATES["relative_mae_gain_over_wrong_family"]
            and values["median_coordinate_sign_accuracy"] >= contract.PREDICTIVE_GATES["median_coordinate_sign_accuracy"]
            for values in panel_results.values()
        )
        if passed:
            strict.append(family)
        selection[family] = {"winner": winner, "behavior_qualified": behavior_ok,
                             "causal_checkpoint": causal_checkpoint,
                             "causal_checkpoint_confirmation_mae": checkpoint_mae,
                             "confirmation_candidates": candidates, "panels": panel_results,
                             "strict_predictive_pass": passed}
    candidate_store = np.load(PREDICTION_CANDIDATES, mmap_mode="r")
    selected = np.lib.format.open_memmap(
        PREDICTIONS, mode="w+", dtype=np.float16,
        shape=(len(fresh), len(QPOINTS), len(contract.ROLES), field.shape[-1]),
    )
    for i, row in enumerate(fresh):
        method_i = FAMILY_METHODS.index(selection[row["family"]]["winner"])
        selected[i] = candidate_store[method_i, i]
    selected.flush(); close_mmap(selected); close_mmap(candidate_store)
    write_rows(OUT / "raw/fresh_pair_index.jsonl", [{"prediction_index": i, **row} for i, row in enumerate(fresh)])
    save(OUT / "analysis/all_predictive_metrics.json", {"metrics": metrics, "wrong_family": wrong_metrics,
                                                          "confirmation_cells": cell_metrics})
    return selection, strict, {"candidate_shape": list(np.load(PREDICTION_CANDIDATES, mmap_mode="r").shape),
                               "selected_shape": list(np.load(PREDICTIONS, mmap_mode="r").shape),
                               "fresh_pairs": len(fresh), "qpoints": list(QPOINTS)}


def mean_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual), dtype=np.float64))


def composition_tournament(field, index: list[dict], behavior: dict) -> dict:
    graph = pairs(index, "graph_composition")
    attitude = pairs(index, "attitude_composition")
    graph_parent = [row for row in graph if row["partition"] == "composition_discovery"]
    graph_fresh = [row for row in graph if row["partition"] == "fresh_composition_lockbox"]
    graph_scores = {}
    for family in contract.GRAPH_FAMILIES:
        train = [row for row in graph_parent if row["family"] == family]
        test = [row for row in graph_fresh if row["family"] == family]
        wrong_family = contract.GRAPH_FAMILIES[(contract.GRAPH_FAMILIES.index(family) + 1) % 3]
        wrong_train = [row for row in graph_parent if row["family"] == wrong_family]
        errors = defaultdict(float); count = 0
        for q in range(field.shape[1]):
            for role_i in range(len(contract.ROLES)):
                by_cell = {}
                wrong_cell = {}
                for depth in (1, 2, 3, 4):
                    for shortcut in (0, 1):
                        if depth == 1 and shortcut == 1:
                            continue
                        subset = [row for row in train if row["depth"] == depth and row["shortcut"] == shortcut]
                        wsubset = [row for row in wrong_train if row["depth"] == depth and row["shortcut"] == shortcut]
                        by_cell[(depth, shortcut)] = arrays(field, subset, q, role_i)[1].mean(axis=0)
                        wrong_cell[(depth, shortcut)] = arrays(field, wsubset, q, role_i)[1].mean(axis=0)
                for row in test:
                    x, actual = arrays(field, [row], q, role_i)
                    pred = by_cell[(row["depth"], row["shortcut"])][None]
                    wrong = wrong_cell[(row["depth"], row["shortcut"])][None]
                    errors["prototype"] += float(np.sum(np.abs(pred - actual), dtype=np.float64))
                    errors["zero"] += float(np.sum(np.abs(actual), dtype=np.float64))
                    errors["wrong"] += float(np.sum(np.abs(wrong - actual), dtype=np.float64))
                    count += actual.size
        scores = {key + "_mae": value / count for key, value in errors.items()}
        scores["gain_over_zero"] = (scores["zero_mae"] - scores["prototype_mae"]) / max(scores["zero_mae"], 1e-9)
        scores["gain_over_wrong_family"] = (scores["wrong_mae"] - scores["prototype_mae"]) / max(scores["wrong_mae"], 1e-9)
        scores["behavior_qualified"] = all(
            behavior["panels"][f"graph_composition|{family}|{partition}"]["dual_qualified"]
            for partition in ("composition_discovery", "fresh_composition_lockbox")
        )
        scores["strict_pass"] = (scores["behavior_qualified"] and scores["gain_over_zero"] >= 0.05
                                 and scores["gain_over_wrong_family"] >= 0.03)
        graph_scores[family] = scores
    attitude_behavior_ok = all(
        behavior["panels"][f"attitude_composition|nested_attitude|{partition}"]["dual_qualified"]
        for partition in ("composition_discovery", "fresh_composition_lockbox")
    )
    if not attitude_behavior_ok:
        return {"graph_path_response": graph_scores,
                "attitude_outer_inner_interaction": {
                    "status": "NA_behavior_gate_failed",
                    "parent_discovery_dual_qualified": behavior["panels"]["attitude_composition|nested_attitude|composition_discovery"]["dual_qualified"],
                    "fresh_lockbox_dual_qualified": behavior["panels"]["attitude_composition|nested_attitude|fresh_composition_lockbox"]["dual_qualified"],
                    "strict_pass": False,
                },
                "interpretation": "Full-coordinate prototype transfer only; no causal or unique-operator claim."}
    parent = [row for row in attitude if row["partition"] == "composition_discovery"]
    fresh = [row for row in attitude if row["partition"] == "fresh_composition_lockbox"]
    errors = defaultdict(float); count = 0
    for q in range(field.shape[1]):
        for role_i in range(len(contract.ROLES)):
            prototype = {}
            for verb in range(3):
                cell_means = {}
                for outer in (0, 1):
                    for inner in (0, 1):
                        subset = [row for row in parent if row["verb_index"] == verb and row["outer_neg"] == outer and row["inner_neg"] == inner]
                        cell_means[(outer, inner)] = arrays(field, subset, q, role_i)[1].mean(axis=0)
                prototype[verb] = cell_means[(1, 1)] - cell_means[(1, 0)] - cell_means[(0, 1)] + cell_means[(0, 0)]
            for language in contract.LANGUAGES:
                for unit in sorted({row["unit"] for row in fresh}):
                    for verb in range(3):
                        cells = {}
                        for outer in (0, 1):
                            for inner in (0, 1):
                                subset = [row for row in fresh if row["language"] == language and row["unit"] == unit and row["verb_index"] == verb and row["outer_neg"] == outer and row["inner_neg"] == inner]
                                cells[(outer, inner)] = arrays(field, subset, q, role_i)[1].mean(axis=0)
                        actual = cells[(1, 1)] - cells[(1, 0)] - cells[(0, 1)] + cells[(0, 0)]
                        pred = prototype[verb]
                        wrong = prototype[(verb + 1) % 3]
                        errors["prototype"] += float(np.sum(np.abs(pred - actual), dtype=np.float64))
                        errors["zero"] += float(np.sum(np.abs(actual), dtype=np.float64))
                        errors["wrong"] += float(np.sum(np.abs(wrong - actual), dtype=np.float64))
                        count += actual.size
    attitude_scores = {key + "_mae": value / count for key, value in errors.items()}
    attitude_scores["gain_over_zero"] = (attitude_scores["zero_mae"] - attitude_scores["prototype_mae"]) / max(attitude_scores["zero_mae"], 1e-9)
    attitude_scores["gain_over_wrong_verb"] = (attitude_scores["wrong_mae"] - attitude_scores["prototype_mae"]) / max(attitude_scores["wrong_mae"], 1e-9)
    attitude_scores["strict_pass"] = attitude_scores["gain_over_zero"] >= 0.05 and attitude_scores["gain_over_wrong_verb"] >= 0.03
    return {"graph_path_response": graph_scores, "attitude_outer_inner_interaction": attitude_scores,
            "interpretation": "Full-coordinate prototype transfer only; no causal or unique-operator claim."}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {family: {"winner": row["winner"], "strict": row["strict_predictive_pass"],
                        "panels": {panel: {k: values[k] for k in ("mae", "gain_over_shared", "gain_over_wrong_family", "median_coordinate_sign_accuracy")}
                                   for panel, values in row["panels"].items()}}
               for family, row in result["selection"].items()}
    text = f"""

## Phase {PHASE}: 全轨迹条件预测与组合响应前瞻赛（C1031-C1048） [{stamp}]

**测试原理与用例。** 本期只使用父词汇 discovery 拟合完整HiddenState响应，从父 confirmation 冻结每族方法，再依次裁决父 lockbox、fresh confirmation和fresh lockbox。比较零响应、共享/族均值、共享/族同坐标仿射、共享/族全坐标对偶岭，并以循环错族作等容量控制。图面板预测1至4跳及直接捷径；嵌套态度用内外层否定的二阶交互作测试。

**公式。** 全坐标对偶岭为：

$$
\\widehat{{Y}}=(X_* - \\bar X)(X-\\bar X)^\\top\\left[(X-\\bar X)(X-\\bar X)^\\top+\\lambda I\\right]^{{-1}}(Y-\\bar Y)+\\bar Y.
$$

嵌套交互为：

$$
I_{{oi}}=R_{{11}}-R_{{10}}-R_{{01}}+R_{{00}}.
$$

所有误差和符号账使用全部检查点、六角色、全部2560坐标；没有PCA、Top-K或余弦筛选。

**结果汇总。** 严格全轨迹预测族为 `{json.dumps(result['strict_predictive_families'], ensure_ascii=False)}`。各族冻结胜者及四个前瞻面板为 `{json.dumps(compact, ensure_ascii=False)}`。组合结果为 `{json.dumps(result['composition'], ensure_ascii=False)}`。供下一期因果裁决的fresh预测场为 `{json.dumps(result['prediction_artifacts'], ensure_ascii=False)}`。

**分析与理论进展。** 通过表示“给定当前基态与语言族，父词汇中学到的条件响应可迁移到全新词汇的整条状态轨迹”，强于只比较均值方向；仍不代表模型内部按岭回归运算，也不代表某个坐标有固定语义。图原型通过只支持条件化路径响应复现；态度交互通过只支持二阶全场残差可迁移。

**问题、硬伤与结论。** 对偶岭样本数小于坐标数，解由基础岭约束确定而非唯一机制；训练材料仍受控；角色对齐依赖字符串跨度；全场平均误差可能掩盖局部失败；float16输入限制低值精度。严格失败只淘汰对应预测主张，其他族和组合路线照常保留。下一步只允许对这里预先冻结的严格胜者做双向调用/删除和错族、错角色、错层控制。

**相关文件。** 脚本 `tests/glm5/phase2249_c1031_c1048_full_coordinate_prediction.py`；结果 `tests/glm5/result/phase2249_c1031_c1048_full_coordinate_prediction`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    prior = load(FIELD_OUT / "analysis/final.json")
    if not prior["all_checks_passed"] or not prior["behavior"]["aggregate_dual_qualified"]:
        raise RuntimeError("Phase2248 is not internally qualified")
    field = np.load(ROOT / prior["field"]["path"], mmap_mode="r")
    index = read_rows(FIELD_OUT / "raw/field_index.jsonl")
    try:
        selection, strict, prediction_artifacts = predictive_tournament(field, index, prior["behavior"])
        composition = composition_tournament(field, index, prior["behavior"])
    finally:
        close_mmap(field)
    save(OUT / "analysis/family_selection.json", selection)
    save(OUT / "analysis/composition.json", composition)
    checks = {
        "frozen_source": True, "all_families_scored": set(selection) == set(contract.FAMILIES),
        "all_panels_scored": all(set(row["panels"]) == {"parent_confirmation", "parent_lockbox", "fresh_confirmation", "fresh_lockbox"} for row in selection.values()),
        "full_candidate_predictions": prediction_artifacts["candidate_shape"][-1] == 2560,
        "selected_predictions_complete": prediction_artifacts["selected_shape"][0] == prediction_artifacts["fresh_pairs"],
        "composition_complete": set(composition["graph_path_response"]) == set(contract.GRAPH_FAMILIES),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "selection": selection,
        "strict_predictive_families": strict, "composition": composition,
        "prediction_artifacts": prediction_artifacts, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Only listed families satisfy the frozen whole-trajectory, shared-control and wrong-family-control gates; predictability is not causality.",
        "next_authorization": "Run causal call/delete only for strict predictive families while preserving all other observation routes.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps({"strict": strict, "composition": composition, "checks": checks}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
