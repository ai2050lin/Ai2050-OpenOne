#!/usr/bin/env python3
"""Compete full-coordinate routes across lexical/task/language/surface/code axes."""
from __future__ import annotations

import gc
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
P2344 = RESULT / "phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract"
P2345 = RESULT / "phase2345_c8361_c8480_qwen4b_bilingual_factorial_full_field"
OUT = RESULT / "phase2346_c8481_c8600_factorial_coordinate_route_competition"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = P2344 / "material/bilingual_factorial_fixed_code.jsonl"
STATES = P2345 / "raw/boundary_all_checkpoints.float16.npy"
TRAJECTORY = P2345 / "derived/layerwise_coordinate_contribution.float32.npy"
GAMMA = P2345 / "raw/final_norm_gamma.float32.npy"
CODE_WEIGHTS = P2345 / "raw/codebook_weight_vectors.float32.npz"
DECISIONS = P2345 / "raw/decisions.float32.npy"
PHASE = 2346
CAMPAIGN = "C8481-C8600"
TRAIN_PARTITIONS = ("discovery", "confirmation")
SELECTION_PARTITION = "fresh_confirmation"
LOCKBOX_PARTITION = "fresh_lockbox"
EPS = 1e-12

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls as layer_control  # noqa: E402
import phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract as contract  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def standardize_rows(field: np.ndarray) -> np.ndarray:
    mean = field.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = field.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    return (field - mean) / np.maximum(scale, 1e-6)


def residualize_additive(field: np.ndarray, rows: list[dict]) -> np.ndarray:
    train = np.asarray([row["partition"] in TRAIN_PARTITIONS for row in rows])
    grand = field[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    prediction = np.broadcast_to(grand, field.shape).copy()
    factors = ("language", "lexical_set", "task", "surface", "codebook", "condition", "state")
    for factor in factors:
        for level in sorted({row[factor] for row in rows}, key=str):
            level_train = train & np.asarray([row[factor] == level for row in rows])
            effect = field[level_train].mean(axis=0, dtype=np.float64).astype(np.float32) - grand
            target = np.asarray([row[factor] == level for row in rows])
            prediction[target] += effect
    return field - prediction


def transfer_specs() -> list[tuple[str, str, Any, Any]]:
    return [
        ("language_en_to_zh", "language", "en", "zh"),
        ("language_zh_to_en", "language", "zh", "en"),
        ("lexicon_0_to_1", "lexical_set", "lex0", "lex1"),
        ("lexicon_1_to_0", "lexical_set", "lex1", "lex0"),
        ("task_supported_to_contradicted", "task", "select_supported", "select_contradicted"),
        ("task_contradicted_to_supported", "task", "select_contradicted", "select_supported"),
        ("surface_direct_to_reported", "surface", "direct", "reported"),
        ("surface_reported_to_direct", "surface", "reported", "direct"),
        ("surface_direct_to_natural", "surface", "direct", "natural"),
        ("surface_natural_to_direct", "surface", "natural", "direct"),
        ("codebook_ab_to_cd", "codebook", "AB", "CD"),
        ("codebook_cd_to_ab", "codebook", "CD", "AB"),
        ("option_original_to_swap", "condition", "original", "option_swapped"),
        ("option_swap_to_original", "condition", "option_swapped", "original"),
    ]


def build_transfer_plans(rows: list[dict], labels: tuple[str, ...], partition: str) -> list[dict]:
    plans = []
    family_values = np.asarray([row["family"] for row in rows], dtype=object)
    partition_values = np.asarray([row["partition"] for row in rows], dtype=object)
    train_mask = np.isin(partition_values, TRAIN_PARTITIONS)
    for name, factor, source, target in transfer_specs():
        factor_values = np.asarray([row[factor] for row in rows], dtype=object)
        prototypes = [np.flatnonzero(train_mask & (family_values == label) & (factor_values == source)) for label in labels]
        grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        test_mask = (partition_values == partition) & (factor_values == target)
        for i in np.flatnonzero(test_mask):
            row = rows[int(i)]
            if row["family"] in labels:
                grouped[(row["family"], row["unit"])].append(int(i))
        keys = sorted(grouped)
        plans.append({"name": name, "prototype_indices": prototypes,
                      "test_indices": [np.asarray(grouped[key], dtype=np.int64) for key in keys],
                      "correct": np.asarray([labels.index(key[0]) for key in keys], dtype=np.int64),
                      "semantic_units_per_family": len(keys) // len(labels)})
    return plans


def grouped_transfer_plan(field: np.ndarray, labels: tuple[str, ...], plan: dict) -> dict:
    prototypes = np.stack([field[idx].mean(axis=0, dtype=np.float64) for idx in plan["prototype_indices"]])
    actual = np.stack([field[idx].mean(axis=0, dtype=np.float64) for idx in plan["test_indices"]])
    distances = (np.sum(actual * actual, axis=1, keepdims=True) + np.sum(prototypes * prototypes, axis=1)[None, :]
                 - 2 * actual @ prototypes.T)
    distances = np.maximum(distances, 0)
    correct = plan["correct"]
    predicted = np.argmin(distances, axis=1)
    correct_distance = distances[np.arange(len(correct)), correct]
    masked = distances.copy()
    masked[np.arange(len(correct)), correct] = np.inf
    best_wrong = np.min(masked, axis=1)
    return {
        "rows": len(correct), "semantic_units_per_family": plan["semantic_units_per_family"], "labels": len(labels),
        "chance": 1 / len(labels), "accuracy": float(np.mean(predicted == correct)),
        "median_correct_over_best_wrong_ratio": float(np.median(correct_distance / (best_wrong + EPS))),
    }


def evaluate(field: np.ndarray, labels: tuple[str, ...], partition: str, plans: list[dict]) -> dict:
    transfers = {}
    for plan in plans:
        transfers[plan["name"]] = grouped_transfer_plan(field, labels, plan)
    accuracies = [value["accuracy"] for value in transfers.values()]
    ratios = [value["median_correct_over_best_wrong_ratio"] for value in transfers.values()]
    return {"partition": partition, "transfers": transfers, "minimum_accuracy": min(accuracies),
            "mean_accuracy": float(np.mean(accuracies)), "maximum_median_distance_ratio": max(ratios)}


def paired_axis(field: np.ndarray, rows: list[dict], factor: str, left: Any, right: Any) -> dict:
    match_keys = ["family", "language", "lexical_set", "task", "surface", "codebook", "condition", "unit", "state"]
    if factor in match_keys:
        match_keys.remove(factor)
    groups: dict[tuple, dict[Any, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        if row["partition"] != LOCKBOX_PARTITION:
            continue
        key = tuple(row[name] for name in match_keys)
        if row[factor] in (left, right):
            groups[key][row[factor]] = index
    a, b = [], []
    for values in groups.values():
        if left in values and right in values:
            a.append(field[values[left]])
            b.append(field[values[right]])
    return layer_control.pair_metrics(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))


def route_fields(raw: np.ndarray, contribution: np.ndarray, rows: list[dict], perm_weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    abs_h = np.abs(raw)
    abs_c = np.abs(contribution)
    perm = np.empty_like(abs_h)
    for codebook in contract.CODEBOOKS:
        idx = np.asarray([i for i, row in enumerate(rows) if row["codebook"] == codebook])
        perm[idx] = abs_h[idx] * perm_weights[codebook][None, :]
    return {
        "raw_hidden": raw,
        "absolute_hidden": abs_h,
        "standardized_absolute_hidden": standardize_rows(abs_h),
        "row_sorted_absolute_hidden": np.sort(abs_h, axis=1),
        "signed_output_contribution": contribution,
        "absolute_output_contribution": abs_c,
        "standardized_absolute_contribution": standardize_rows(abs_c),
        "row_sorted_absolute_contribution": np.sort(abs_c, axis=1),
        "permuted_output_weight_absolute_hidden": perm,
        "factorial_residual_absolute_hidden": residualize_additive(abs_h, rows),
    }


def normalized_raw(states: np.ndarray, qpoint: int, gamma: np.ndarray, eps: float) -> np.ndarray:
    value = states[:, qpoint].astype(np.float32)
    if qpoint == states.shape[1] - 1:
        return value
    return layer_control.rms_norm(value, gamma, eps)


def supported_task_authorization(rows: list[dict]) -> dict:
    decisions = np.load(DECISIONS, mmap_mode="r")
    families = {}
    qualified = []
    for family in contract.FAMILIES:
        cells = {}
        passed = True
        for language in contract.LANGUAGES:
            for codebook in contract.CODEBOOKS:
                for partition in contract.PARTITIONS:
                    idx = [i for i, row in enumerate(rows) if row["family"] == family
                           and row["task"] == "select_supported" and row["language"] == language
                           and row["codebook"] == codebook and row["partition"] == partition]
                    accuracy = float(np.mean(decisions[idx, 3]))
                    cells[f"{language}:{codebook}:{partition}"] = accuracy
                    passed = passed and accuracy >= 0.70
        families[family] = {"qualified": passed, "minimum_accuracy": min(cells.values()), "cells": cells}
        if passed:
            qualified.append(family)
    close_memmap(decisions)
    return {"task": "select_supported", "families": families, "qualified": qualified,
            "qualified_count": len(qualified), "threshold": 0.70}


def analyze(rows: list[dict], qualified: tuple[str, ...], eps: float, all_task_qualified_count: int,
            authorization: dict) -> tuple[dict, dict[str, np.ndarray]]:
    states = np.load(STATES, mmap_mode="r")
    trajectory = np.load(TRAJECTORY, mmap_mode="r")
    gamma = np.load(GAMMA).astype(np.float32)
    weights = np.load(CODE_WEIGHTS)
    rng = np.random.default_rng(8481)
    perm_weights = {name: rng.permutation(np.abs(weights[name]).astype(np.float32)) for name in contract.CODEBOOKS}
    selection_plans = build_transfer_plans(rows, qualified, SELECTION_PARTITION)
    lockbox_plans = build_transfer_plans(rows, qualified, LOCKBOX_PARTITION)
    layer_records = []
    for qpoint in range(states.shape[1]):
        raw = normalized_raw(states, qpoint, gamma, eps)
        contribution = trajectory[:, qpoint].astype(np.float32)
        routes = route_fields(raw, contribution, rows, perm_weights)
        result = {"qpoint": qpoint, "routes": {}}
        for name, field in routes.items():
            score = evaluate(field, qualified, SELECTION_PARTITION, selection_plans)
            result["routes"][name] = {
                "minimum_accuracy": score["minimum_accuracy"], "mean_accuracy": score["mean_accuracy"],
                "maximum_median_distance_ratio": score["maximum_median_distance_ratio"],
            }
        layer_records.append(result)
        del routes, raw, contribution
        gc.collect()
        print(f"[phase2346 selection] {qpoint}/{states.shape[1] - 1}", flush=True)
    candidates = []
    for record in layer_records:
        for route, score in record["routes"].items():
            candidates.append({"qpoint": record["qpoint"], "route": route, **score})
    selected = max(candidates, key=lambda row: (row["minimum_accuracy"], row["mean_accuracy"], -row["qpoint"], row["route"]))
    qpoint = int(selected["qpoint"])
    raw = normalized_raw(states, qpoint, gamma, eps)
    contribution = trajectory[:, qpoint].astype(np.float32)
    routes = route_fields(raw, contribution, rows, perm_weights)
    lockbox = {name: evaluate(field, qualified, LOCKBOX_PARTITION, lockbox_plans) for name, field in routes.items()}
    ranked = sorted(({"route": name, **value} for name, value in lockbox.items()),
                    key=lambda row: (-row["minimum_accuracy"], -row["mean_accuracy"], row["route"]))
    selected_lock = lockbox[selected["route"]]
    best_sorted = max(lockbox[name]["minimum_accuracy"] for name in
                      ("row_sorted_absolute_hidden", "row_sorted_absolute_contribution"))
    best_coordinate = max(lockbox[name]["minimum_accuracy"] for name in lockbox
                          if not name.startswith("row_sorted"))
    gate = {
        "qualified_family_minimum": 8,
        "actual_qualified_families": len(qualified),
        "selected_route": selected["route"], "selected_qpoint": qpoint,
        "lockbox_all_axis_minimum_accuracy": selected_lock["minimum_accuracy"],
        "lockbox_maximum_median_distance_ratio": selected_lock["maximum_median_distance_ratio"],
        "best_coordinate_minimum_accuracy": best_coordinate,
        "best_sorted_minimum_accuracy": best_sorted,
        "coordinate_advantage_over_sorted": best_coordinate - best_sorted,
        "supported_task_behavior_pass": len(qualified) >= 8,
        "all_task_behavior_pass": all_task_qualified_count >= 8,
        "all_axes_pass": selected_lock["minimum_accuracy"] >= 0.30,
        "distance_dominance_pass": selected_lock["maximum_median_distance_ratio"] < 1.0,
        "coordinate_identity_pass": best_coordinate >= best_sorted + 0.10,
    }
    gate["descriptive_atlas_passed"] = all((gate["supported_task_behavior_pass"], gate["all_axes_pass"],
                                             gate["distance_dominance_pass"], gate["coordinate_identity_pass"]))
    gate["passed"] = all((gate["all_task_behavior_pass"], gate["all_axes_pass"], gate["distance_dominance_pass"],
                          gate["coordinate_identity_pass"]))
    pairing = {
        name: {
            "language": paired_axis(field, rows, "language", "en", "zh"),
            "lexicon": paired_axis(field, rows, "lexical_set", "lex0", "lex1"),
            "task": paired_axis(field, rows, "task", "select_supported", "select_contradicted"),
            "codebook": paired_axis(field, rows, "codebook", "AB", "CD"),
            "option": paired_axis(field, rows, "condition", "original", "option_swapped"),
        }
        for name, field in routes.items() if name in (selected["route"], "absolute_hidden", "absolute_output_contribution")
    }
    keep_names = []
    for name in (selected["route"], "absolute_hidden", "row_sorted_absolute_hidden",
                 "absolute_output_contribution", "permuted_output_weight_absolute_hidden",
                 "factorial_residual_absolute_hidden"):
        if name not in keep_names:
            keep_names.append(name)
    publish_fields = {name: routes[name] for name in keep_names}
    analysis = {"qualified_families": list(qualified), "behavior_authorization": authorization,
                "all_task_qualified_count": all_task_qualified_count, "selection_partition": SELECTION_PARTITION,
                "full_layer_route_selection": layer_records, "selected": selected,
                "lockbox_routes_ranked": ranked, "lockbox": lockbox, "paired_lockbox": pairing,
                "gate": gate, "permutation_seed": 8481}
    weights.close(); close_memmap(states); close_memmap(trajectory)
    return analysis, publish_fields


def publish(rows: list[dict], qpoint: int, fields: dict[str, np.ndarray]) -> dict:
    dataset_id = "c8481_qwen4b_bilingual_factorial_route_competition_passport"
    binary = VIS / f"{dataset_id}.float32.npy"
    out = atlas.create_binary(binary.name, len(rows) * len(fields), 2560, np.float32)
    metadata = []
    cursor = 0
    for route, field in fields.items():
        out[cursor:cursor + len(rows)] = field
        metadata.extend({
            "case_id": row["case_id"], "route": route, "qpoint": qpoint, "family": row["family"],
            "macrotype": row["macrotype"], "language": row["language"], "lexical_set": row["lexical_set"],
            "task": row["task"], "surface": row["surface"], "codebook": row["codebook"],
            "condition": row["condition"], "partition": row["partition"], "unit": row["unit"],
            "state": row["state"], "target_code": row["target_code"],
        } for row in rows)
        cursor += len(rows)
    out.flush(); close_memmap(out)
    return atlas.write_metadata(
        dataset_id, f"Qwen3-4B q{qpoint} bilingual factorial full-coordinate route competition",
        binary, metadata, "Qwen3-4B-FP16", "bilingual_factorial_route_competition_v1",
        "observational route comparison on untouched lockbox; causal status depends on the frozen gate",
        "selected route plus absolute-H, sorted, true-output-weight, permuted-weight and factorial-residual controls",
        "all 2560 physical coordinates retained for every row and every displayed route",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "no_topk": True,
         "qpoint": qpoint, "routes": list(fields)},
    )


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 双语正交全坐标路线竞赛与fresh锁箱裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2345全部36,864条边界场运行十条不压缩坐标路线：raw H、|H|、逐行标准化|H|、逐行排序|H|、signed/absolute输出贡献、标准化/排序absolute贡献、坐标置乱输出权重和基础加性析因残差。discovery+confirmation只建族原型；每个测试向量先在同一family×unit内跨其余因素平均，避免把同一单元的成百重复行伪装成独立样本。fresh_confirmation在所有38检查点×路线中选“14个跨轴方向最低准确率最高”的唯一候选，fresh_lockbox只裁决一次。

$$
\hat f(x)=\arg\min_f\|x-\mu_f^{{train,source}}\|_2^2,
\qquad
A_{{min}}=\min_{{a\in\mathcal A}}A_a,
$$

$$
R= X-\left(\mu+\sum_{{k\in\{{L,V,T,S,O,P,Z\}}}}(\mu_{{k=x_k}}-\mu)\right).
$$

**结果汇总与相关文件。** 行为合格族 `{json.dumps(result['analysis']['qualified_families'], ensure_ascii=False)}`；选择 `{json.dumps(result['analysis']['selected'], ensure_ascii=False)}`；锁箱路线排名 `{json.dumps(result['analysis']['lockbox_routes_ranked'], ensure_ascii=False)}`；冻结门 `{json.dumps(result['analysis']['gate'], ensure_ascii=False)}`；严格配对 `{json.dumps(result['analysis']['paired_lockbox'], ensure_ascii=False)}`；客户端全坐标护照 `{json.dumps(result['dataset'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2346_c8481_c8600_factorial_coordinate_route_competition.py`；结果 `tests/glm5/result/phase2346_c8481_c8600_factorial_coordinate_route_competition`。

**理论进展、问题硬伤与结论。** 这一竞赛先问“哪种具体坐标表示能跨因素识别族”，再问输出权重是否特殊；不把高准确率称为计算齿轮。Phase2345的全任务资格为零，但重新按预先通过特征审计后，`select_supported`的12族均在language×codebook×partition最低0.70门上合格；所以保留12族做描述性全坐标图谱，所有跨`select_contradicted`结果明确只是失败轴诊断。分组平均保留全部2560坐标但会平滑token个体残差，因此另报告严格逐行配对几何。坐标排序若接近完整坐标，说明能量分布而非地址；析因残差若通过，只说明加性表面主效应后仍有族结构。没有路线通过双向语言、词汇、任务、表述、输出码、选项交换、距离优势、坐标身份及全任务行为资格全部冻结门，就不得进入Shapley、曲率或因果流形叙事。

**下一阶段路线判断。** 若门通过，目标相同，自动对通过的完整坐标路线做连续联盟多剂量调用/删除/救援、错族/错层/反号/等范数随机及无关任务副作用，并顺序复验Qwen14B、GLM4、DS7B；若门失败，自动转入失败轴的全token形成轨迹诊断，不做无授权因果闭合。
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
    parent = json.loads((P2345 / "analysis/final.json").read_text(encoding="utf-8"))
    rows = io.read_rows(MATERIAL)
    authorization = supported_task_authorization(rows)
    qualified = tuple(authorization["qualified"])
    all_task_qualified_count = len(parent["behavior"]["qualified"])
    freeze = {
        "frozen_before_field_analysis": True, "routes": [
            "raw_hidden", "absolute_hidden", "standardized_absolute_hidden", "row_sorted_absolute_hidden",
            "signed_output_contribution", "absolute_output_contribution", "standardized_absolute_contribution",
            "row_sorted_absolute_contribution", "permuted_output_weight_absolute_hidden",
            "factorial_residual_absolute_hidden",
        ],
        "selection_partition": SELECTION_PARTITION, "lockbox_partition": LOCKBOX_PARTITION,
        "transfer_directions": [name for name, *_ in transfer_specs()], "minimum_accuracy": 0.30,
        "coordinate_advantage_over_sorted": 0.10, "distance_ratio_maximum": 1.0,
        "qualified_family_minimum": 8, "coordinate_policy": "all 2560; no Top-K/PCA/projection",
    }
    save(OUT / "config/frozen_contract.json", freeze)
    analysis, fields = analyze(rows, qualified, float(parent["collection"]["rms_norm_eps"]),
                               all_task_qualified_count, authorization)
    dataset = publish(rows, int(analysis["selected"]["qpoint"]), fields)
    verification = atlas.verify(dataset)
    verified = all(value for key, value in verification.items() if key != "id")
    catalog = atlas.update_catalog([dataset])
    build = atlas.frontend_build()
    checks = {
        "all_layers": len(analysis["full_layer_route_selection"]) == 38,
        "all_routes": all(len(row["routes"]) == 10 for row in analysis["full_layer_route_selection"]),
        "all_transfer_directions": all(len(value["transfers"]) == 14 for value in analysis["lockbox"].values()),
        "asset_verified": verified, "frontend_build": build["passed"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "analysis": analysis,
              "dataset": json.loads(json.dumps(dataset, ensure_ascii=False, default=str)),
              "verification": verification, "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
              "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2346_failed", checks))
    append_memo(result)
    del fields
    gc.collect()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
