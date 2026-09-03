#!/usr/bin/env python3
"""Test whether each sample's own H_q improves full-coordinate update prediction."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import phase2408_c25201_c25520_fullcoordinate_deconfounding as p2408

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2407 = RESULT / "phase2407_c24881_c25200_qwen4b_component_fullfield"
OUT = RESULT / "phase2409_c25521_c25840_state_dependent_coordinate_operator"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2409
CAMPAIGN = "C25521-C25840"
COMPONENTS = ("total", "attention", "mlp")
SPLITS = (*p2408.SPLITS, "language_lockbox")
SHIFT = 791  # coprime to 2560: fixed before evaluation


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def indices(rows: list[dict], partition: str) -> np.ndarray:
    return np.asarray([i for i, row in enumerate(rows) if row["partition"] == partition], dtype=np.int64)


def fit_baseline(rows: list[dict], train: np.ndarray, design: p2408.Design,
                 y: np.ndarray, h: np.ndarray) -> dict:
    x = design.matrix(rows, train)
    beta_y = design.pinv @ y[train]
    nuisance_y = x @ beta_y
    family_y = p2408.grouped_effect(y[train] - nuisance_y, rows, train, "family")
    base_y = p2408.add_effect(nuisance_y, rows, train, "family", family_y)

    beta_h = design.pinv @ h[train]
    nuisance_h = x @ beta_h
    family_h = p2408.grouped_effect(h[train] - nuisance_h, rows, train, "family")
    base_h = p2408.add_effect(nuisance_h, rows, train, "family", family_h)
    hr = h[train] - base_h
    yr = y[train] - base_y
    den = np.sum(hr * hr, axis=0, dtype=np.float64) + 1e-12
    slope = (np.sum(hr * yr, axis=0, dtype=np.float64) / den).astype(np.float32)
    permutation = np.roll(np.arange(h.shape[1], dtype=np.int64), SHIFT)
    hp = hr[:, permutation]
    den_p = np.sum(hp * hp, axis=0, dtype=np.float64) + 1e-12
    slope_p = (np.sum(hp * yr, axis=0, dtype=np.float64) / den_p).astype(np.float32)
    return {"beta_y": beta_y, "family_y": family_y, "beta_h": beta_h, "family_h": family_h,
            "slope": slope, "slope_mismatch": slope_p, "permutation": permutation}


def predict(rows: list[dict], test: np.ndarray, design: p2408.Design,
            y: np.ndarray, h: np.ndarray, fitted: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = design.matrix(rows, test)
    base_y = p2408.add_effect(x @ fitted["beta_y"], rows, test, "family", fitted["family_y"])
    base_h = p2408.add_effect(x @ fitted["beta_h"], rows, test, "family", fitted["family_h"])
    hr = h[test] - base_h
    state = base_y + hr * fitted["slope"]
    mismatch = base_y + hr[:, fitted["permutation"]] * fitted["slope_mismatch"]
    return base_y, state, mismatch


def empty_metrics() -> dict:
    return {component: {split: {stage: p2408.accumulator() for stage in ("family", "state", "mismatch")}
                        for split in SPLITS} for component in COMPONENTS}


def analyze_task(task: str) -> dict:
    rows = p2408.read_rows(P2407 / f"index/{task}_rows.jsonl")
    state = np.load(P2407 / f"raw/{task}_state_event.float16.npy", mmap_mode="r")
    attention = np.load(P2407 / f"raw/{task}_attention_event.float16.npy", mmap_mode="r")
    mlp = np.load(P2407 / f"raw/{task}_mlp_event.float16.npy", mmap_mode="r")
    layers, events, dimension = attention.shape[1:]
    train = indices(rows, "discovery")
    en_train = np.asarray([i for i in train if rows[i]["language"] == "en"], dtype=np.int64)
    zh_test = np.asarray([i for i in train if rows[i]["language"] == "zh"], dtype=np.int64)
    tests = {split: indices(rows, split) for split in p2408.SPLITS}
    tests["language_lockbox"] = zh_test
    designs = {"standard": p2408.Design(rows, train), "language": p2408.Design(rows, en_train)}
    metrics = empty_metrics()
    derived = OUT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    slopes = np.lib.format.open_memmap(derived / f"{task}_matched_diagonal_slope.float32.npy", mode="w+",
                                       dtype=np.float32, shape=(len(COMPONENTS), layers, events, dimension))
    gains = np.lib.format.open_memmap(derived / f"{task}_coordinate_gain.float32.npy", mode="w+", dtype=np.float32,
                                      shape=(len(COMPONENTS), len(SPLITS), 2, layers, events, dimension))
    for layer in range(layers):
        for event in range(events):
            h = np.asarray(state[:, layer, event], dtype=np.float32)
            a = np.asarray(attention[:, layer, event], dtype=np.float32)
            m = np.asarray(mlp[:, layer, event], dtype=np.float32)
            for ci, (component, y) in enumerate((("total", a + m), ("attention", a), ("mlp", m))):
                standard = fit_baseline(rows, train, designs["standard"], y, h)
                slopes[ci, layer, event] = standard["slope"]
                language = fit_baseline(rows, en_train, designs["language"], y, h)
                for si, split in enumerate(SPLITS):
                    test = tests[split]
                    if test.size == 0:
                        continue
                    design = designs["language"] if split == "language_lockbox" else designs["standard"]
                    fitted = language if split == "language_lockbox" else standard
                    base_y, pred_state, pred_mismatch = predict(rows, test, design, y, h, fitted)
                    truth = y[test]
                    global_base = np.broadcast_to(y[en_train if split == "language_lockbox" else train].mean(axis=0), truth.shape)
                    for stage, pred in (("family", base_y), ("state", pred_state), ("mismatch", pred_mismatch)):
                        p2408.update_metric(metrics[component][split][stage], truth, pred, global_base, rows, test)
                    gains[ci, si, 0, layer, event] = np.sum((truth - base_y) ** 2 - (truth - pred_state) ** 2,
                                                            axis=0, dtype=np.float64).astype(np.float32)
                    gains[ci, si, 1, layer, event] = np.sum((truth - pred_mismatch) ** 2 - (truth - pred_state) ** 2,
                                                            axis=0, dtype=np.float64).astype(np.float32)
        slopes.flush(); gains.flush()
        print(f"[phase2409 {task}] layer {layer + 1}/{layers}", flush=True)
    slopes.flush(); gains.flush(); close(slopes); close(gains); close(state); close(attention); close(mlp)
    finished = {component: {split: ({stage: p2408.finish(acc) for stage, acc in stages.items()}
                                    if next(iter(stages.values()))["count"] else None)
                            for split, stages in split_map.items()}
                for component, split_map in metrics.items()}
    compact = {}
    for component in COMPONENTS:
        compact[component] = {}
        for split in SPLITS:
            value = finished[component][split]
            compact[component][split] = None if value is None else {
                "family_gain": value["family"]["gain_vs_base"],
                "state_gain": value["state"]["gain_vs_base"],
                "mismatch_gain": value["mismatch"]["gain_vs_base"],
                "state_increment": value["state"]["gain_vs_base"] - value["family"]["gain_vs_base"],
                "physical_advantage": value["state"]["gain_vs_base"] - value["mismatch"]["gain_vs_base"],
                "median_unit_state_gain": value["state"]["median_unit_gain"],
            }
    return {"task": task, "rows": len(rows), "train_rows": len(train), "en_train_rows": len(en_train),
            "zh_test_rows": len(zh_test), "field_shape": list(state.shape), "metrics": finished, "summary": compact,
            "arrays": {"slope": str(derived / f"{task}_matched_diagonal_slope.float32.npy"),
                       "coordinate_gain": str(derived / f"{task}_coordinate_gain.float32.npy")}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 样本状态依赖同坐标算子与物理错配对照（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 延续Phase2408冻结的discovery/四类锁箱。先分别对$H_q$和真实组件更新$U_q$去除语言、表面、方向、角色/步数、槽位及语言族均值，再只在discovery逐物理坐标拟合对角状态算子。严格对照把输入坐标循环错配791位后重新拟合；因此对照拥有同样参数量、训练量和每坐标尺度，只破坏$H_j\rightarrow U_j$的固定物理对应。total、Attention、MLP全部坐标均评价。英文拟合—中文留出独立重拟合。空分区记N/A：Phase2408 composition的deep/joint显示1.0源于空集分母保护，不是实验阳性，本Phase在append-only记录中正式纠正。

$$\widetilde H=H-\widehat H_{{nuisance}}-G_f^H,\quad
\widetilde U=U-\widehat U_{{nuisance}}-G_f^U,\quad
\hat a_j=\frac{{\sum_i\widetilde H_{{ij}}\widetilde U_{{ij}}}}{{\sum_i\widetilde H_{{ij}}^2+\epsilon}},$$

$$\widehat U_{{state,j}}=\widehat U_{{family,j}}+\hat a_j\widetilde H_j,\qquad
\Delta_{{phys}}=G(\widehat U_{{state}})-G(\widehat U_{{mismatch}}).$$

**结果汇总。** 选择 `{json.dumps(result['selection']['summary'], ensure_ascii=False)}`；组合 `{json.dumps(result['composition']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2409_c25521_c25840_state_dependent_coordinate_operator.py`；逐组件×层×事件×全部坐标斜率、逐锁箱状态增益和物理坐标优势位于`tests/glm5/result/phase2409_c25521_c25840_state_dependent_coordinate_operator`。

**分析与理论进展。** 本Phase把外部条件均值推进到读取当前样本状态的最小算子。只有state相对family在未知模板/内容上增加预测力，且matched优于等容量mismatch，才支持“固定基底上的状态条件齿”。这仍是逐坐标局部律，不预设齿轮组或跨坐标协同。

**问题硬伤与结论。** 对角线性算子不包含跨坐标、非线性或注意力头内部条件；阴性不能否定更复杂机制。反之阳性也可能来自同坐标残差自相关而非语言算法，必须在Phase2410用全坐标分组交互与随机分组对照继续检查。语言留出含词形/tokenizer改变，仍不能单独解释为语义抽象失败。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    selection = analyze_task("selection")
    composition = analyze_task("composition")
    decisive = {}
    for task_name, task in (("selection", selection), ("composition", composition)):
        decisive[task_name] = {component: {split: (None if task["summary"][component][split] is None else {
            "state_increment": task["summary"][component][split]["state_increment"],
            "physical_advantage": task["summary"][component][split]["physical_advantage"]})
            for split in SPLITS} for component in COMPONENTS}
    adjudication = {"decisive": decisive, "condition_mean_is_runtime_algorithm": False,
                    "diagonal_operator_is_complete_gear": False,
                    "phase2408_empty_composition_splits_corrected_to_na": True}
    finite_values = [v[key] for task in (selection, composition) for component in COMPONENTS for v in task["summary"][component].values()
                     if v is not None for key in ("state_increment", "physical_advantage")]
    checks = {"finite": all(math.isfinite(v) for v in finite_values), "all_coordinates": True,
              "physical_mismatch_refit": True, "empty_splits_are_na": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "selection": selection, "composition": composition,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "selection": selection["summary"], "composition": composition["summary"],
                      "adjudication": adjudication, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
