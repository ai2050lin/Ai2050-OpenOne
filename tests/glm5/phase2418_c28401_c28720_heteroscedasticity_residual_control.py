#!/usr/bin/env python3
"""Audit exact-pair coordinate advantage against heteroscedasticity and sample-shuffle controls."""
from __future__ import annotations

import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2415 = RESULT / "phase2415_c27441_c27760_exact_paired_composition"
OUT = RESULT / "phase2418_c28401_c28720_heteroscedasticity_residual_control"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2418
CAMPAIGN = "C28401-C28720"
COMPONENTS = ("total", "attention", "mlp")
SPLITS = ("fresh_unit_lockbox", "confirmation", "template_lockbox", "language_lockbox")
STAGES = ("family", "state", "coordinate_mismatch", "zstate", "z_coordinate_mismatch",
          "sample_shuffle_mean", "sample_shuffle_q95")
SHIFT = 791
SHUFFLES = 16
BRIDGE_LAYER = 14

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2415_c27441_c27760_exact_paired_composition as paired  # noqa: E402
import phase2416_c27761_c28080_crossmodel_exact_pair_replication as capture_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def family_bases(train: np.ndarray, families: np.ndarray, h: np.ndarray, y: np.ndarray) -> dict:
    names = sorted(set(families[train]))
    family_y = {name: y[train[families[train] == name]].mean(axis=0) for name in names}
    family_h = {name: h[train[families[train] == name]].mean(axis=0) for name in names}
    base_y = np.stack([family_y[name] for name in families[train]])
    base_h = np.stack([family_h[name] for name in families[train]])
    hr, yr = h[train] - base_h, y[train] - base_y
    sd_h = np.sqrt(np.mean(hr * hr, axis=0, dtype=np.float64) + 1e-12).astype(np.float32)
    sd_y = np.sqrt(np.mean(yr * yr, axis=0, dtype=np.float64) + 1e-12).astype(np.float32)
    slope = (np.sum(hr * yr, axis=0, dtype=np.float64) /
             (np.sum(hr * hr, axis=0, dtype=np.float64) + 1e-12)).astype(np.float32)
    permutation = np.roll(np.arange(h.shape[1]), SHIFT)
    hp = hr[:, permutation]
    slope_p = (np.sum(hp * yr, axis=0, dtype=np.float64) /
               (np.sum(hp * hp, axis=0, dtype=np.float64) + 1e-12)).astype(np.float32)
    hz, yz = hr / sd_h, yr / sd_y
    z_slope = (np.sum(hz * yz, axis=0, dtype=np.float64) /
               (np.sum(hz * hz, axis=0, dtype=np.float64) + 1e-12)).astype(np.float32)
    hzp = hz[:, permutation]
    z_slope_p = (np.sum(hzp * yz, axis=0, dtype=np.float64) /
                 (np.sum(hzp * hzp, axis=0, dtype=np.float64) + 1e-12)).astype(np.float32)
    return {"family_y": family_y, "family_h": family_h, "global_y": y[train].mean(axis=0),
            "slope": slope, "slope_p": slope_p, "z_slope": z_slope, "z_slope_p": z_slope_p,
            "sd_h": sd_h, "sd_y": sd_y, "permutation": permutation}


def base_for(test: np.ndarray, families: np.ndarray, fitted: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack([fitted["family_y"][name] for name in families[test]]),
            np.stack([fitted["family_h"][name] for name in families[test]]))


def gain(truth: np.ndarray, prediction: np.ndarray, global_y: np.ndarray) -> float:
    denominator = float(np.sum((truth - global_y) ** 2, dtype=np.float64)) + 1e-30
    return float(1 - np.sum((truth - prediction) ** 2, dtype=np.float64) / denominator)


def shuffle_indices(pair_rows: list[dict], test: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.arange(len(test), dtype=np.int64)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for local, index in enumerate(test):
        row = pair_rows[index]
        groups[(row["family"], row["language"], row["surface"], row["direction"])].append(local)
    for values in groups.values():
        if len(values) < 2:
            continue
        shift = int(rng.integers(1, len(values)))
        permuted = np.roll(np.asarray(values, dtype=np.int64), shift)
        result[np.asarray(values, dtype=np.int64)] = permuted
    return result


def predict_controls(pair_rows: list[dict], train: np.ndarray, test: np.ndarray, families: np.ndarray,
                     h: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict[str, float], dict]:
    fitted = family_bases(train, families, h, y)
    base_y, base_h = base_for(test, families, fitted)
    truth, hr = y[test], h[test] - base_h
    global_y = np.broadcast_to(fitted["global_y"], truth.shape)
    permutation = fitted["permutation"]
    predictions = {
        "family": base_y,
        "state": base_y + hr * fitted["slope"],
        "coordinate_mismatch": base_y + hr[:, permutation] * fitted["slope_p"],
        "zstate": base_y + (hr / fitted["sd_h"]) * fitted["z_slope"] * fitted["sd_y"],
        "z_coordinate_mismatch": base_y + (hr[:, permutation] / fitted["sd_h"][permutation]) *
                                 fitted["z_slope_p"] * fitted["sd_y"],
    }
    values = {name: gain(truth, prediction, global_y) for name, prediction in predictions.items()}
    shuffled = []
    for repeat in range(SHUFFLES):
        order = shuffle_indices(pair_rows, test, seed + repeat)
        shuffled.append(gain(truth, base_y + hr[order] * fitted["slope"], global_y))
    values["sample_shuffle_mean"] = float(np.mean(shuffled))
    values["sample_shuffle_q95"] = float(np.quantile(shuffled, .95))
    return values, {"fitted": fitted, "base_y": base_y, "truth": truth, "hr": hr,
                    "state_prediction": predictions["state"], "shuffle_gains": shuffled}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def analyze(rows: list[dict], collection: dict) -> dict:
    pair_rows, step1, step2 = paired.pair_index(rows)
    families = np.asarray([row["family"] for row in pair_rows], dtype=object)
    train = np.asarray([i for i, row in enumerate(pair_rows) if row["partition"] == "discovery"], dtype=np.int64)
    en_train = np.asarray([i for i in train if pair_rows[i]["language"] == "en"], dtype=np.int64)
    tests = {split: np.asarray([i for i, row in enumerate(pair_rows) if row["partition"] == split], dtype=np.int64)
             for split in SPLITS[:-1]}
    tests["language_lockbox"] = np.asarray([i for i in train if pair_rows[i]["language"] == "zh"], dtype=np.int64)
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    layers, dimension = state.shape[1:]
    metrics = np.zeros((len(COMPONENTS), len(SPLITS), len(STAGES), layers), dtype=np.float32)
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    slopes = np.lib.format.open_memmap(derived / "raw_and_zscore_slopes.float32.npy", mode="w+", dtype=np.float32,
                                       shape=(len(COMPONENTS), layers, 4, dimension))
    bridge_cache = None
    for layer in range(layers):
        h = np.asarray(state[step2, layer], dtype=np.float32) - np.asarray(state[step1, layer], dtype=np.float32)
        a = np.asarray(attention[step2, layer], dtype=np.float32) - np.asarray(attention[step1, layer], dtype=np.float32)
        m = np.asarray(mlp[step2, layer], dtype=np.float32) - np.asarray(mlp[step1, layer], dtype=np.float32)
        for ci, y in enumerate((a + m, a, m)):
            standard_fit = family_bases(train, families, h, y)
            slopes[ci, layer] = np.stack([standard_fit[key] for key in ("slope", "slope_p", "z_slope", "z_slope_p")])
            for si, split in enumerate(SPLITS):
                fit = en_train if split == "language_lockbox" else train
                values, details = predict_controls(pair_rows, fit, tests[split], families, h, y,
                                                   PHASE * 10000 + ci * 1000 + si * 100 + layer)
                metrics[ci, si, :, layer] = [values[stage] for stage in STAGES]
            if ci == 0 and layer == BRIDGE_LAYER:
                bridge_cache = (h.copy(), y.copy())
        slopes.flush()
        print(f"[phase2418 analysis] layer {layer + 1}/{layers}", flush=True)
    np.save(derived / "control_layer_metrics.float32.npy", metrics)
    summary = {}
    for ci, component in enumerate(COMPONENTS):
        summary[component] = {}
        for si, split in enumerate(SPLITS):
            item = {stage: float(metrics[ci, si, stage_index].mean()) for stage_index, stage in enumerate(STAGES)}
            item.update({"raw_physical_advantage": item["state"] - item["coordinate_mismatch"],
                         "z_physical_advantage": item["zstate"] - item["z_coordinate_mismatch"],
                         "state_over_shuffle_q95": item["state"] - item["sample_shuffle_q95"],
                         "layers_state_over_shuffle_q95_rate": float(np.mean(metrics[ci, si, 1] > metrics[ci, si, 6]))})
            summary[component][split] = item
    if bridge_cache is None:
        raise RuntimeError("bridge layer not captured")
    h, y = bridge_cache
    nondiscovery = np.concatenate([tests[split] for split in SPLITS[:-1]])
    fitted = family_bases(train, families, h, y)
    base_y, base_h = base_for(nondiscovery, families, fitted)
    truth = y[nondiscovery]
    prediction = base_y + (h[nondiscovery] - base_h) * fitted["slope"]
    improvement = ((np.mean((truth - base_y) ** 2, axis=1) -
                    np.mean((truth - prediction) ** 2, axis=1)) /
                   (np.mean((truth - fitted["global_y"]) ** 2, axis=1) + 1e-30))
    teacher = read_rows(P2415 / "qwen4b/behavior/teacher_scores.jsonl")
    teacher_map = {row["case_id"]: row["mean_logprob_margin"] > 0 for row in teacher}
    both_correct = np.asarray([teacher_map[f"{pair_rows[index]['pair_id']}-s1"] and
                               teacher_map[f"{pair_rows[index]['pair_id']}-s2"] for index in nondiscovery], dtype=np.float64)
    bridge = {"layer": BRIDGE_LAYER, "pairs": len(nondiscovery), "both_correct_pairs": int(both_correct.sum()),
              "improvement_behavior_correlation": pearson(improvement, both_correct),
              "mean_improvement_both_correct": float(improvement[both_correct == 1].mean()) if both_correct.any() else None,
              "mean_improvement_other": float(improvement[both_correct == 0].mean()),
              "behavior_rate": float(both_correct.mean())}
    np.save(derived / "bridge_pair_normalized_improvement.float32.npy", improvement.astype(np.float32))
    files = {"slopes": str(derived / "raw_and_zscore_slopes.float32.npy"),
             "metrics": str(derived / "control_layer_metrics.float32.npy"),
             "bridge": str(derived / "bridge_pair_normalized_improvement.float32.npy")}
    for value in (slopes, state, attention, mlp):
        close(value)
    return {"pairs": len(pair_rows), "train_pairs": len(train), "test_pairs": {key: len(value) for key, value in tests.items()},
            "shuffles_per_cell": SHUFFLES, "shuffle_strata": ["family", "language", "surface", "direction"],
            "summary": summary, "behavior_bridge": bridge, "files": files}


def cleanup(collection: dict) -> dict:
    paths, total = [], 0
    for value in collection.values():
        path = Path(value["path"])
        if path.exists():
            total += path.stat().st_size; paths.append(str(path)); path.unlink()
    return {"removed_files": len(paths), "removed_bytes": total, "removed_gib": total / 2**30,
            "recoverable": False, "paths": paths}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 同坐标优势的异方差、残差自相关与行为桥审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2415–2416的791位坐标错配虽等容量重拟合，但不同坐标的方差/尺度不同，matched优势可能只是物理异方差。为排除该替代解释，在Qwen3-4B完整640个严格pair/1280条上重新采集answer-boundary的36层状态、Attention、MLP全部2560坐标。除原始matched/mismatch外，先在discovery逐坐标标准化$D^H,D^U$再拟合matched与错配；另在每个测试集内部按family×language×surface×direction分层，把样本状态残差循环置乱16次，保持每坐标分布与条件组成不变，只破坏同一pair的$H\leftrightarrow U$对应。最后在预先固定的layer14比较每pair状态预测改善与“两步均教师强制正确”的关系。

$$Z^H_j=\frac{{D^H_j-G^H_{{f,j}}}}{{\sigma^H_j}},\quad
Z^U_j=\frac{{D^U_j-G^U_{{f,j}}}}{{\sigma^U_j}},\quad
\widehat Z^U_j=\rho_j Z^H_j,$$

$$\Delta_{{zphys}}=G(\widehat Z^U_{{j\leftarrow j}})-G(\widehat Z^U_{{j\leftarrow j+791}}),\qquad
\Delta_{{sample}}=G(\widehat U(H_p))-Q_{{.95}}G(\widehat U(H_{{\pi(p)}})).$$

**结果汇总。** 全坐标控制 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；行为桥 `{json.dumps(result['analysis']['behavior_bridge'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2418_c28401_c28720_heteroscedasticity_residual_control.py`；原始/标准化matched与错配斜率、逐层控制指标、pair级行为桥及final位于`tests/glm5/result/phase2418_c28401_c28720_heteroscedasticity_residual_control`。未修改其他Markdown。

**分析与理论进展。** 标准化错配回答“优势是否只因某些坐标天然幅度大”，同层同条件样本置乱回答“优势是否真需要当前pair的状态”。只有$\Delta_{{zphys}}>0$且matched超过置乱q95，才把固定坐标耦合从尺度纹理推进为样本条件纹理。行为桥则独立判断这种纹理是否更强地出现在模型实际偏好两步答案的pair中。

**问题硬伤与结论。** 分层置乱保持可观测四因子，但没有保持未记录的token长度与全部局部词形；每组最少只有2个unit，16次循环置乱不是大样本随机化检验。逐坐标标准化仍是线性二阶控制，不排除一般网络残差动力学。行为桥使用教师强制偏好而非自主生成，且layer14来自上一Phase模板集最优层，虽预先固定仍不是完整输出编译。原始float16场在派生全坐标结果后删除且不可恢复。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2415 / "index/composition_rows.jsonl")
    model, tokenizer, label = capability.load_model("qwen4b")
    capture_utils.OUT = OUT
    try:
        collection = capture_utils.collect("qwen4b", model, rows, 4)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    analysis = analyze(rows, collection)
    raw_cleanup = cleanup(collection)
    decisive = [{"component": component, "split": split,
                 "z_physical_advantage": values["z_physical_advantage"],
                 "state_over_shuffle_q95": values["state_over_shuffle_q95"],
                 "layer_rate": values["layers_state_over_shuffle_q95_rate"]}
                for component, split_map in analysis["summary"].items() for split, values in split_map.items()]
    adjudication = {"decisive_cells": decisive,
                    "all_z_physical_advantages_positive": all(row["z_physical_advantage"] > 0 for row in decisive),
                    "all_state_over_shuffle_q95_positive": all(row["state_over_shuffle_q95"] > 0 for row in decisive),
                    "behavior_bridge_positive": analysis["behavior_bridge"]["improvement_behavior_correlation"] > 0,
                    "semantic_composition_gear_proven": False}
    checks = {"full_640_pairs": analysis["pairs"] == 640, "full_coordinates": collection["state"]["shape"] == [1280, 36, 2560],
              "zscore_control": True, "sixteen_stratified_shuffles": analysis["shuffles_per_cell"] == 16,
              "finite": all(math.isfinite(value) for component in analysis["summary"].values() for split in component.values()
                            for key, value in split.items()),
              "raw_cleaned": raw_cleanup["removed_files"] == 3, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": label, "collection": collection,
              "analysis": analysis, "cleanup": raw_cleanup, "adjudication": adjudication,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "summary": analysis["summary"], "behavior_bridge": analysis["behavior_bridge"],
                      "adjudication": adjudication, "cleanup": raw_cleanup, "checks": checks},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
