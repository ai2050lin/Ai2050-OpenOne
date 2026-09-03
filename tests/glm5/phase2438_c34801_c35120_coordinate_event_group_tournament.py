#!/usr/bin/env python3
"""Basic scalar/group/diagonal tournament for signed language-operation trajectories."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2437 = RESULT / "phase2437_c34481_c34800_signed_trajectory_atlas"
OUT = RESULT / "phase2438_c34801_c35120_coordinate_event_group_tournament"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2438
CAMPAIGN = "C34801-C35120"
INTERACTIONS = ("semantic_validity", "lexical_control")
SPLITS = ("confirmation", "fresh_unit", "surface", "language", "direction", "family_holdout")
STAGES = ("global", "family", "scalar", "contiguous32", "random32", "diagonal", "coordinate_mismatch", "event_mismatch")
EVENT_NAMES = ("prefix_end", "operation_end", "argument_end", "context_end", "query_end",
               "candidate1_end", "candidate2_end", "answer_boundary")
SHIFT = 791
EVENT_SHIFT = 3
RIDGE = 1e-8


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def masks(meta: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    unit = np.asarray([int(row["unit"]) for row in meta])
    surface = np.asarray([row["surface"] for row in meta], dtype=object)
    language = np.asarray([row["language"] for row in meta], dtype=object)
    direction = np.asarray([int(row["direction"]) for row in meta])
    return {
        "confirmation": (np.flatnonzero(unit < 4), np.flatnonzero(unit == 4)),
        "fresh_unit": (np.flatnonzero(unit < 5), np.flatnonzero(unit == 5)),
        "surface": (np.flatnonzero((unit < 5) & (surface == "canonical")), np.flatnonzero((unit == 5) & (surface == "natural"))),
        "language": (np.flatnonzero((unit < 5) & (language == "en")), np.flatnonzero((unit == 5) & (language == "zh"))),
        "direction": (np.flatnonzero((unit < 5) & (direction == 0)), np.flatnonzero((unit == 5) & (direction == 1))),
        "family_holdout": (np.flatnonzero(unit < 5), np.flatnonzero(unit == 5)),
    }


def family_centers(train: np.ndarray, families: np.ndarray, h: np.ndarray, y: np.ndarray,
                   condition: bool) -> dict:
    global_h = h[train].mean(axis=0)
    global_y = y[train].mean(axis=0)
    family_h, family_y = {}, {}
    if condition:
        for family in sorted(set(families[train])):
            chosen = train[families[train] == family]
            family_h[family] = h[chosen].mean(axis=0)
            family_y[family] = y[chosen].mean(axis=0)
    base_h = np.stack([family_h.get(families[index], global_h) for index in train])
    base_y = np.stack([family_y.get(families[index], global_y) for index in train])
    x = h[train] - base_h
    target = y[train] - base_y
    diagonal = np.sum(x * target, axis=0) / (np.sum(x * x, axis=0) + RIDGE)
    scalar = float(np.sum(x.astype(np.float64) * target.astype(np.float64)) /
                   (np.sum(x.astype(np.float64) ** 2) + RIDGE))
    return {"global_h": global_h, "global_y": global_y, "family_h": family_h, "family_y": family_y,
            "x": x, "target": target, "diagonal": diagonal, "scalar": scalar}


def group_slopes(x: np.ndarray, target: np.ndarray, groups: np.ndarray) -> np.ndarray:
    slopes = np.zeros(x.shape[1], dtype=np.float32)
    for group in range(int(groups.max()) + 1):
        chosen = groups == group
        numerator = float(np.sum(x[:, chosen].astype(np.float64) * target[:, chosen].astype(np.float64)))
        denominator = float(np.sum(x[:, chosen].astype(np.float64) ** 2))
        slopes[chosen] = numerator / (denominator + RIDGE)
    return slopes


def predictions(test: np.ndarray, families: np.ndarray, h: np.ndarray, fitted: dict,
                contiguous: np.ndarray, random_groups: np.ndarray, other_h: np.ndarray,
                other_means: dict) -> list[np.ndarray]:
    global_y = np.broadcast_to(fitted["global_y"], (len(test), h.shape[1]))
    family_y = np.stack([fitted["family_y"].get(families[index], fitted["global_y"]) for index in test])
    family_h = np.stack([fitted["family_h"].get(families[index], fitted["global_h"]) for index in test])
    centered = h[test] - family_h
    contig_slope = group_slopes(fitted["x"], fitted["target"], contiguous)
    random_slope = group_slopes(fitted["x"], fitted["target"], random_groups)
    other_base = np.stack([other_means["family_h"].get(families[index], other_means["global_h"]) for index in test])
    return [global_y, family_y,
            family_y + centered * fitted["scalar"],
            family_y + centered * contig_slope,
            family_y + centered * random_slope,
            family_y + centered * fitted["diagonal"],
            family_y + centered * np.roll(fitted["diagonal"], SHIFT),
            family_y + (other_h[test] - other_base) * fitted["diagonal"]]


def gains(truth: np.ndarray, predicted: list[np.ndarray]) -> tuple[np.ndarray, bool]:
    denominator = float(np.sum((truth.astype(np.float64) - predicted[0].astype(np.float64)) ** 2))
    scale = float(np.sum(truth.astype(np.float64) ** 2))
    if denominator <= max(1e-20, scale * 1e-12):
        return np.zeros(len(predicted), dtype=np.float32), False
    return np.asarray([0.0] + [1 - float(np.sum((truth.astype(np.float64) - value.astype(np.float64)) ** 2)) / denominator
                               for value in predicted[1:]], dtype=np.float32), True


def evaluate_split(train: np.ndarray, test: np.ndarray, families: np.ndarray, h: np.ndarray, y: np.ndarray,
                   other_h: np.ndarray, contiguous: np.ndarray, random_groups: np.ndarray,
                   family_holdout: bool) -> tuple[np.ndarray, int]:
    if not family_holdout:
        fitted = family_centers(train, families, h, y, True)
        other_means = family_centers(train, families, other_h, y, True)
        predicted = predictions(test, families, h, fitted, contiguous, random_groups, other_h, other_means)
        values, active = gains(y[test], predicted)
        return values, int(active)
    truth_all = []
    predicted_all = [[] for _ in STAGES]
    active_cells = 0
    for family in sorted(set(families)):
        tr = train[families[train] != family]
        te = test[families[test] == family]
        fitted = family_centers(tr, families, h, y, False)
        other_means = family_centers(tr, families, other_h, y, False)
        predicted = predictions(te, families, h, fitted, contiguous, random_groups, other_h, other_means)
        truth_all.append(y[te])
        for stage, value in enumerate(predicted):
            predicted_all[stage].append(value)
    values, active = gains(np.concatenate(truth_all), [np.concatenate(value) for value in predicted_all])
    active_cells += int(active)
    return values, active_cells


def analyze(meta: list[dict]) -> dict:
    path = P2437 / "derived/signed_interaction_state.float16.npy"
    state = np.load(path, mmap_mode="r")
    interactions, qpoints, events, configs, dim = state.shape
    updates = qpoints - 2
    families = np.asarray([row["family"] for row in meta], dtype=object)
    specs = masks(meta)
    contiguous = np.arange(dim, dtype=np.int64) // 80
    rng = np.random.default_rng(2438)
    permutation = rng.permutation(dim)
    random_groups = np.empty(dim, dtype=np.int64)
    random_groups[permutation] = np.arange(dim, dtype=np.int64) // 80
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    metrics = np.zeros((interactions, len(SPLITS), len(STAGES), updates, events), dtype=np.float32)
    active = np.zeros((interactions, len(SPLITS), updates, events), dtype=np.uint8)
    slopes = np.lib.format.open_memmap(derived / "discovery_diagonal_slopes.float32.npy", mode="w+",
                                       dtype=np.float32, shape=(interactions, updates, events, dim))
    slope_train = specs["fresh_unit"][0]
    for ii in range(interactions):
        for qpoint in range(updates):
            for event in range(events):
                h = np.asarray(state[ii, qpoint, event], dtype=np.float32)
                y = np.asarray(state[ii, qpoint + 1, event], dtype=np.float32) - h
                other_event = (event + EVENT_SHIFT) % events
                other_h = np.asarray(state[ii, qpoint, other_event], dtype=np.float32)
                fitted_full = family_centers(slope_train, families, h, y, True)
                slopes[ii, qpoint, event] = fitted_full["diagonal"]
                for si, split in enumerate(SPLITS):
                    train, test = specs[split]
                    values, is_active = evaluate_split(train, test, families, h, y, other_h, contiguous,
                                                       random_groups, split == "family_holdout")
                    metrics[ii, si, :, qpoint, event] = values
                    active[ii, si, qpoint, event] = is_active
            slopes.flush()
            if (qpoint + 1) % 6 == 0 or qpoint + 1 == updates:
                print(f"[phase2438] interaction={INTERACTIONS[ii]} update={qpoint + 1}/{updates}", flush=True)
    np.save(derived / "coordinate_event_group_gains.float32.npy", metrics)
    np.save(derived / "active_cells.uint8.npy", active)
    np.save(derived / "contiguous32_group_ids.int16.npy", contiguous.astype(np.int16))
    np.save(derived / "random32_group_ids.int16.npy", random_groups.astype(np.int16))
    summary = {}
    for ii, interaction in enumerate(INTERACTIONS):
        summary[interaction] = {}
        for si, split in enumerate(SPLITS):
            mask_active = active[ii, si].astype(bool)
            summary[interaction][split] = {}
            for stage, name in enumerate(STAGES):
                cells = metrics[ii, si, stage][mask_active]
                summary[interaction][split][name] = float(cells.mean()) if len(cells) else 0.0
            summary[interaction][split]["active_cells"] = int(mask_active.sum())
            summary[interaction][split]["diagonal_physical_advantage"] = float(
                summary[interaction][split]["diagonal"] - summary[interaction][split]["coordinate_mismatch"])
            summary[interaction][split]["diagonal_event_advantage"] = float(
                summary[interaction][split]["diagonal"] - summary[interaction][split]["event_mismatch"])
            summary[interaction][split]["diagonal_minus_best_group"] = float(
                summary[interaction][split]["diagonal"] - max(summary[interaction][split]["contiguous32"],
                                                               summary[interaction][split]["random32"]))
    event_summary = {interaction: {EVENT_NAMES[event]: {
        "fresh_diagonal": float(metrics[ii, SPLITS.index("fresh_unit"), STAGES.index("diagonal"), :, event][
            active[ii, SPLITS.index("fresh_unit"), :, event].astype(bool)].mean())
        if active[ii, SPLITS.index("fresh_unit"), :, event].any() else 0.0,
        "fresh_physical_advantage": float((metrics[ii, SPLITS.index("fresh_unit"), STAGES.index("diagonal"), :, event] -
                                             metrics[ii, SPLITS.index("fresh_unit"), STAGES.index("coordinate_mismatch"), :, event])[
            active[ii, SPLITS.index("fresh_unit"), :, event].astype(bool)].mean())
        if active[ii, SPLITS.index("fresh_unit"), :, event].any() else 0.0}
        for event in range(events)} for ii, interaction in enumerate(INTERACTIONS)}
    files = {"metrics": str(derived / "coordinate_event_group_gains.float32.npy"),
             "active": str(derived / "active_cells.uint8.npy"),
             "slopes": str(derived / "discovery_diagonal_slopes.float32.npy"),
             "contiguous_groups": str(derived / "contiguous32_group_ids.int16.npy"),
             "random_groups": str(derived / "random32_group_ids.int16.npy")}
    slopes.flush(); close(slopes); close(state)
    return {"interactions": interactions, "updates": updates, "events": events, "configurations": configs,
            "dimension": dim, "split_sizes": {name: [len(value[0]), len(value[1])] for name, value in specs.items()},
            "summary": summary, "event_summary": event_summary, "files": files}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 标量—坐标组—逐坐标—事件错配基础结构竞赛（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2437的两种有符号interaction，用当前状态逐坐标预测下一block更新。复杂度从全局常量、family常量、单标量斜率，升到连续32组、冻结随机32组、2560个对角斜率；循环错配791坐标和错配3个事件为物理/事件零假设。使用unit确认、fresh unit、canonical→natural、英文→中文、方向反转和留一family六种外推；前四个因果不可见事件的零变化单独标为inactive，不用零分稀释活动事件。

$$\hat U_j=\bar U_{{f,j}}+\beta_{{g(j)}}(H_j-\bar H_{{f,j}}),$$
$$G=1-\frac{{\sum_j(U_j-\hat U_j)^2}}{{\sum_j(U_j-\bar U_{{train,j}})^2}},\quad
\Delta_{{phys}}=G_{{diag}}-G_{{shift791}},\quad
\Delta_{{event}}=G_{{diag}}-G_{{event+3}}.$$

**结果汇总。** 切分 `{json.dumps(result['analysis']['split_sizes'], ensure_ascii=False)}`；复杂度竞赛 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；逐事件 `{json.dumps(result['analysis']['event_summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2438_c34801_c35120_coordinate_event_group_tournament.py`；全部interaction×split×stage×36block×8event收益、active掩码、全2560坐标斜率及两种32组编号位于`tests/glm5/result/phase2438_c34801_c35120_coordinate_event_group_tournament/derived`。

**分析与理论进展。** 这个竞赛直接回答：轨迹是否只需族均值、是否存在粗组规律、还是固定物理坐标身份提供额外预测；event mismatch同时检验同一坐标规律是否随语言事件条件化。逐坐标胜出只意味着最基础的同坐标局部律优于这些零假设，不等于坐标独立，也不排除多坐标协同。

**问题硬伤与结论。** 所有规则是离线一阶预测；family均值可能吸收大量模板特征。连续/随机32组只是基础对照，不是穷尽组结构；对角模型参数更多，必须靠严格锁箱而不是训练拟合裁决。若语义interaction未稳定胜过lexical control，只能说存在一般条件动力学，不能命名语义齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    meta = read_rows(P2437 / "index/configurations.jsonl")
    analysis = analyze(meta)
    sem, lex = analysis["summary"]["semantic_validity"], analysis["summary"]["lexical_control"]
    adjudication = {
        "semantic_diagonal_positive_all_splits": all(sem[split]["diagonal"] > 0 for split in SPLITS),
        "semantic_physical_advantage_positive_all_splits": all(sem[split]["diagonal_physical_advantage"] > 0 for split in SPLITS),
        "semantic_event_advantage_positive_all_splits": all(sem[split]["diagonal_event_advantage"] > 0 for split in SPLITS),
        "semantic_diagonal_beats_groups_all_splits": all(sem[split]["diagonal_minus_best_group"] > 0 for split in SPLITS),
        "semantic_diagonal_beats_lexical_all_splits": all(sem[split]["diagonal"] > lex[split]["diagonal"] for split in SPLITS),
        "universal_coordinate_law_proven": False,
    }
    checks = {"two_interactions": analysis["interactions"] == 2, "updates_36": analysis["updates"] == 36,
              "events_8": analysis["events"] == 8, "configs_384": analysis["configurations"] == 384,
              "dimension_2560": analysis["dimension"] == 2560, "six_splits": set(analysis["split_sizes"]) == set(SPLITS),
              "all_files": all(Path(path).exists() for path in analysis["files"].values()),
              "finite": all(math.isfinite(value) for interaction in analysis["summary"].values()
                            for split in interaction.values() for value in split.values()),
              "claim_boundary": not adjudication["universal_coordinate_law_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis, "adjudication": adjudication,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
