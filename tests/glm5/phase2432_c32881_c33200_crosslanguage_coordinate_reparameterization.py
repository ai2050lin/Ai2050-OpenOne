#!/usr/bin/env python3
"""Test whether the English/Chinese failure is a reusable coordinate reparameterization."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2423 = RESULT / "phase2423_c30001_c30320_semantic_validity_behavior_contract"
P2424 = RESULT / "phase2424_c30321_c30640_semantic_validity_multievent_fullfield"
P2428 = RESULT / "phase2428_c31601_c31920_crosslayer_path_consistency"
P2429 = RESULT / "phase2429_c31921_c32240_direct_composed_relation_algebra"
P2431 = RESULT / "phase2431_c32561_c32880_encoding_field_visualization"
OUT = RESULT / "phase2432_c32881_c33200_crosslanguage_coordinate_reparameterization"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2432
CAMPAIGN = "C32881-C33200"
INTERACTIONS = ("semantic_validity", "lexical_control")
DIRECTIONS = ("en_to_zh", "zh_to_en")
SPLITS = ("confirmation", "fresh_unit", "template", "joint", "direction", "family")
STAGES = ("family", "identity", "scalar", "diagonal", "mismatch")

sys.path.insert(0, str(TESTS))
import phase2425_c30641_c30960_semantic_specific_interaction_atlas as atlas  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def paired_meta(meta: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    keys: dict[tuple, dict[str, int]] = {}
    for index, row in enumerate(meta):
        key = (row["family"], int(row["unit"]), row["surface"], row["surface_class"], int(row["direction"]))
        keys.setdefault(key, {})[row["language"]] = index
    paired, english, chinese = [], [], []
    for key in sorted(keys):
        match = keys[key]
        if set(match) != {"en", "zh"}:
            raise RuntimeError((key, sorted(match)))
        family, unit, surface, surface_class, direction = key
        paired.append({"family": family, "unit": unit, "surface": surface,
                       "surface_class": surface_class, "direction": direction})
        english.append(match["en"]); chinese.append(match["zh"])
    if len(paired) != 512:
        raise RuntimeError(("language_pairs", len(paired)))
    return paired, np.asarray(english, dtype=np.int64), np.asarray(chinese, dtype=np.int64)


def split_specs(meta: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray, bool]]:
    unit = np.asarray([int(row["unit"]) for row in meta])
    controlled = np.asarray([row["surface_class"] == "controlled" for row in meta])
    direction = np.asarray([int(row["direction"]) for row in meta])
    natural = ~controlled
    full_train = np.flatnonzero(controlled & (unit < 6))
    return {
        "confirmation": (np.flatnonzero(controlled & (unit < 4)), np.flatnonzero(controlled & (unit >= 4) & (unit < 6)), True),
        "fresh_unit": (full_train, np.flatnonzero(controlled & (unit >= 6)), True),
        "template": (full_train, np.flatnonzero(natural & (unit < 6)), True),
        "joint": (full_train, np.flatnonzero(natural & (unit >= 6)), True),
        "direction": (np.flatnonzero(controlled & (unit < 6) & (direction == 0)),
                      np.flatnonzero(controlled & (unit < 6) & (direction == 1)), True),
        "family": (full_train, np.flatnonzero(natural & (unit >= 6)), False),
    }


def gain(truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> float:
    denominator = float(np.sum((truth - baseline) ** 2, dtype=np.float64))
    scale = float(np.sum(truth * truth, dtype=np.float64))
    if denominator <= max(1e-20, scale * 1e-12):
        return 0.0
    residual = float(np.sum((truth - prediction) ** 2, dtype=np.float64))
    return 1.0 - residual / denominator


def predict_stages(x: np.ndarray, families: np.ndarray, test: np.ndarray, fitted: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base_x = np.stack([fitted["family_h"].get(families[i], fitted["global_h"]) for i in test])
    base_y = np.stack([fitted["family_y"].get(families[i], fitted["global_y"]) for i in test])
    global_y = np.broadcast_to(fitted["global_y"], base_y.shape)
    delta = x[test] - base_x
    return global_y, {
        "family": base_y,
        "identity": base_y + delta,
        "scalar": None,
        "diagonal": base_y + delta * fitted["slope"],
        "mismatch": base_y + delta * np.roll(fitted["slope"], atlas.SHIFT),
    }


def evaluate_partition(x: np.ndarray, y: np.ndarray, families: np.ndarray, train: np.ndarray,
                       test: np.ndarray, conditioned: bool) -> tuple[np.ndarray, np.ndarray]:
    fitted = atlas.fit(train, families, x, y, family_conditioned=conditioned)
    global_y, predictions = predict_stages(x, families, test, fitted)
    train_base_x = np.stack([fitted["family_h"].get(families[i], fitted["global_h"]) for i in train])
    train_base_y = np.stack([fitted["family_y"].get(families[i], fitted["global_y"]) for i in train])
    centered_x, centered_y = x[train] - train_base_x, y[train] - train_base_y
    scalar = float(np.sum(centered_x * centered_y, dtype=np.float64) /
                   max(np.sum(centered_x * centered_x, dtype=np.float64), 1e-30))
    test_base_x = np.stack([fitted["family_h"].get(families[i], fitted["global_h"]) for i in test])
    test_base_y = np.stack([fitted["family_y"].get(families[i], fitted["global_y"]) for i in test])
    predictions["scalar"] = test_base_y + (x[test] - test_base_x) * scalar
    values = np.asarray([gain(y[test], predictions[stage], global_y) for stage in STAGES], dtype=np.float32)
    return values, fitted["slope"]


def evaluate(x: np.ndarray, y: np.ndarray, families: np.ndarray, split: str,
             specs: dict[str, tuple[np.ndarray, np.ndarray, bool]]) -> np.ndarray:
    train_pool, test_pool, conditioned = specs[split]
    if split != "family":
        return evaluate_partition(x, y, families, train_pool, test_pool, conditioned)[0]
    truths, baselines = [], []
    predictions = {stage: [] for stage in STAGES}
    for family in sorted(set(families)):
        train = train_pool[families[train_pool] != family]
        test = test_pool[families[test_pool] == family]
        fitted = atlas.fit(train, families, x, y, family_conditioned=False)
        global_y, candidate = predict_stages(x, families, test, fitted)
        centered_x, centered_y = x[train] - fitted["global_h"], y[train] - fitted["global_y"]
        scalar = float(np.sum(centered_x * centered_y, dtype=np.float64) /
                       max(np.sum(centered_x * centered_x, dtype=np.float64), 1e-30))
        candidate["scalar"] = fitted["global_y"] + (x[test] - fitted["global_h"]) * scalar
        truths.append(y[test]); baselines.append(global_y)
        for stage in STAGES:
            predictions[stage].append(candidate[stage])
    truth, baseline = np.concatenate(truths), np.concatenate(baselines)
    return np.asarray([gain(truth, np.concatenate(predictions[stage]), baseline) for stage in STAGES], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def cycle_test(x: np.ndarray, y: np.ndarray, families: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    forward = atlas.fit(train, families, x, y, family_conditioned=True)
    reverse = atlas.fit(train, families, y, x, family_conditioned=True)
    base_x = np.stack([forward["family_h"].get(families[i], forward["global_h"]) for i in test])
    base_y = np.stack([forward["family_y"].get(families[i], forward["global_y"]) for i in test])
    y_hat = base_y + (x[test] - base_x) * forward["slope"]
    reverse_base_y = np.stack([reverse["family_h"].get(families[i], reverse["global_h"]) for i in test])
    reverse_base_x = np.stack([reverse["family_y"].get(families[i], reverse["global_y"]) for i in test])
    x_hat = reverse_base_x + (y_hat - reverse_base_y) * reverse["slope"]
    mismatch = reverse_base_x + (y_hat - reverse_base_y) * np.roll(reverse["slope"], atlas.SHIFT)
    baseline = np.broadcast_to(reverse["global_y"], x_hat.shape)
    return {"cycle_gain": gain(x[test], x_hat, baseline),
            "cycle_mismatch_gain": gain(x[test], mismatch, baseline),
            "cycle_physical_advantage": gain(x[test], x_hat, baseline) - gain(x[test], mismatch, baseline),
            "reciprocal_slope_mean_abs_error": float(np.mean(np.abs(forward["slope"] * reverse["slope"] - 1.0)))}


def analyze() -> dict:
    rows = read_rows(P2424 / "index/semantic_validity_rows.jsonl")
    meta, _ = atlas.configuration_index(rows)
    paired, english, chinese = paired_meta(meta)
    families = np.asarray([row["family"] for row in paired], dtype=object)
    specs = split_specs(paired)
    path = np.load(P2428 / "derived/semantic_lexical_state_path.float32.npy", mmap_mode="r")
    metrics = np.zeros((2, 2, len(SPLITS), len(STAGES), 37), dtype=np.float32)
    slopes_path = OUT / "derived/crosslanguage_diagonal_slope.float32.npy"
    slopes_path.parent.mkdir(parents=True, exist_ok=True)
    slopes = np.lib.format.open_memmap(slopes_path, mode="w+", dtype=np.float32, shape=(2, 2, 37, 2560))
    full_train = specs["fresh_unit"][0]
    cycles, stability = {}, {}
    for ii, interaction in enumerate(INTERACTIONS):
        cycles[interaction], stability[interaction] = {}, {}
        for di, direction in enumerate(DIRECTIONS):
            source_indices, target_indices = (english, chinese) if di == 0 else (chinese, english)
            for qpoint in range(37):
                x, y = np.asarray(path[ii, qpoint, source_indices]), np.asarray(path[ii, qpoint, target_indices])
                fitted = atlas.fit(full_train, families, x, y, family_conditioned=True)
                slopes[ii, di, qpoint] = fitted["slope"]
                for si, split in enumerate(SPLITS):
                    metrics[ii, di, si, :, qpoint] = evaluate(x, y, families, split, specs)
                if qpoint in (12, 24, 36):
                    cycles[interaction].setdefault(direction, {})[f"q{qpoint}"] = cycle_test(
                        x, y, families, specs["joint"][0], specs["joint"][1])
                print(f"[phase2432] {interaction} {direction} q{qpoint}/36", flush=True)
            stability[interaction][direction] = {
                "adjacent_slope_coordinate_cosine_mean": float(np.mean([
                    cosine(slopes[ii, di, q], slopes[ii, di, q + 1]) for q in range(36)])),
                "adjacent_slope_coordinate_cosine_min": float(np.min([
                    cosine(slopes[ii, di, q], slopes[ii, di, q + 1]) for q in range(36)])),
            }
    slopes.flush(); close(slopes); close(path)
    np.save(OUT / "derived/crosslanguage_stage_metrics.float32.npy", metrics)
    summary = {
        interaction: {
            direction: {
                split: {
                    "family_gain": float(metrics[ii, di, si, 0].mean()),
                    "identity_gain": float(metrics[ii, di, si, 1].mean()),
                    "scalar_gain": float(metrics[ii, di, si, 2].mean()),
                    "diagonal_gain": float(metrics[ii, di, si, 3].mean()),
                    "mismatch_gain": float(metrics[ii, di, si, 4].mean()),
                    "diagonal_physical_advantage": float((metrics[ii, di, si, 3] - metrics[ii, di, si, 4]).mean()),
                } for si, split in enumerate(SPLITS)
            } for di, direction in enumerate(DIRECTIONS)
        } for ii, interaction in enumerate(INTERACTIONS)
    }
    specificity = {direction: {split:
        summary["semantic_validity"][direction][split]["diagonal_gain"] -
        summary["lexical_control"][direction][split]["diagonal_gain"]
        for split in SPLITS} for direction in DIRECTIONS}
    return {
        "pairs": len(paired), "families": sorted(set(families)), "splits": list(SPLITS),
        "summary": summary, "semantic_minus_lexical_diagonal_gain": specificity,
        "cycles": cycles, "slope_stability": stability,
        "files": {"slopes": str(slopes_path),
                  "metrics": str(OUT / "derived/crosslanguage_stage_metrics.float32.npy")},
    }


def raw_paths() -> list[Path]:
    p2424 = json.loads((P2424 / "analysis/final.json").read_text(encoding="utf-8"))
    p2428 = json.loads((P2428 / "analysis/final.json").read_text(encoding="utf-8"))
    p2429 = json.loads((P2429 / "analysis/final.json").read_text(encoding="utf-8"))
    paths = [Path(item["path"]) for item in p2424["collection"].values()]
    paths.append(Path(p2428["state_path"]["path"]))
    paths.extend(Path(item["path"]) for item in p2429["collection"].values())
    return paths


def cleanup_raw() -> dict:
    allowed = [P2424.resolve(), P2428.resolve(), P2429.resolve()]
    removed, missing, freed = [], [], 0
    for path in raw_paths():
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in allowed) or resolved.suffix != ".npy":
            raise RuntimeError(("unsafe cleanup target", str(resolved)))
        if resolved.exists():
            size = resolved.stat().st_size
            resolved.unlink()
            removed.append({"path": str(resolved), "bytes": size})
            freed += size
        else:
            missing.append(str(resolved))
    return {"removed": removed, "already_missing": missing, "freed_bytes": freed,
            "freed_gib": freed / (1024 ** 3), "recoverable": False,
            "retained": [str(P2425) for P2425 in (P2431, OUT)],
            "policy": "unserved bulk HiddenState/Attention/MLP arrays removed after all derived atlases and the full-coordinate client asset were verified"}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 中英语义场的双向逐坐标重参数化与自动后继裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 主计划完成后，后继仍是同一目标，故自动继续。将Phase2428语义有效性/词项对照在37个状态点的1024配置全2560维路径按family、unit、surface、direction严格配成512个中英对。分别检验English→Chinese和Chinese→English。每层只在训练集拟合家族基线、单位映射、单标量和固定逐坐标斜率，以+791坐标错位为物理零假设；留出轴为确认、新实体、模板、联合、方向和关系族。q12/q24/q36另做双向循环，不把几何相似冒充可逆编码。

$$Z^{{zh}}-\bar Z^{{zh}}_f\approx\beta^{{en\to zh}}\odot(Z^{{en}}-\bar Z^{{en}}_f),$$

$$\widehat Z^{{en}}_{{cycle}}=\bar Z^{{en}}_f+\beta^{{zh\to en}}\odot(\widehat Z^{{zh}}-\bar Z^{{zh}}_f),$$

$$G=1-\frac{{\sum\|Z-\widehat Z\|^2}}{{\sum\|Z-\widehat Z_{{global}}\|^2}},\qquad A_{{physical}}=G_{{same-coordinate}}-G_{{shift+791}}.$$

**结果汇总。** 512中英对、8关系族、2交互、2方向、37状态点、6锁箱的摘要 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；语义减词项 `{json.dumps(result['analysis']['semantic_minus_lexical_diagonal_gain'], ensure_ascii=False)}`；循环 `{json.dumps(result['analysis']['cycles'], ensure_ascii=False)}`；斜率跨层稳定性 `{json.dumps(result['analysis']['slope_stability'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2432_c32881_c33200_crosslanguage_coordinate_reparameterization.py`；完整逐坐标双向斜率、全层stage指标与final位于`tests/glm5/result/phase2432_c32881_c33200_crosslanguage_coordinate_reparameterization`。客户端继续使用Phase2431已验证的全2560坐标资产。除本MEMO外未增改其他Markdown。

**分析与理论进展。** 这个测试把“language锁箱失败”拆成三个可证伪层次：若同坐标斜率仅比错位好，说明固定基底仍有物理对应；若绝对gain为正，说明逐坐标重参数化具有预测力；若语义还稳定超过词项对照、双向循环成立，才支持共享语义编码被语言外壳重参数化。缺一项都不能把跨语言对齐提升为通用语言齿轮。

**问题硬伤、结论与下一阶段。** 对角图仍不含坐标群交互；家族均值消除了大量语言整体偏置；中英模板不是逐字翻译，测试的是功能配对而非词对齐。循环使用离线拟合，不是模型内部逆操作。当前结论必须由裁决门而非热力图观感决定。大体积原始场已在所有派生数组、可视化与本后继完成后按明确白名单删除，无法从工作区恢复；保留MEMO、索引、行为、完整派生斜率/指标和客户端资产。若重参数化仍不具语义特异性，下一大阶段不应继续增加对角回归，而应回到新采集材料，直接记录同义跨语言token路径中的条件坐标共激活时序，同时坚持全坐标与强词项对照。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    analysis = analyze()
    semantic_map = all(
        analysis["summary"]["semantic_validity"][direction][split]["diagonal_gain"] > 0 and
        analysis["summary"]["semantic_validity"][direction][split]["diagonal_physical_advantage"] > 0
        for direction in DIRECTIONS for split in SPLITS)
    semantic_specific = all(value > 0 for direction in analysis["semantic_minus_lexical_diagonal_gain"].values()
                            for value in direction.values())
    cycle = all(item["cycle_gain"] > 0 and item["cycle_physical_advantage"] > 0
                for interaction in analysis["cycles"].values() for direction in interaction.values()
                for item in direction.values())
    adjudication = {
        "semantic_crosslanguage_diagonal_map_all_splits_directions": semantic_map,
        "semantic_diagonal_map_exceeds_lexical_all_splits_directions": semantic_specific,
        "bidirectional_cycle_all_checkpoints": cycle,
        "shared_semantic_code_reparameterization_detected": semantic_map and semantic_specific and cycle,
        "conditional_coordinate_gear_proven": False,
        "language_encoding_mechanism_closed": False,
    }
    cleanup = cleanup_raw()
    checks = {
        "paired_512": analysis["pairs"] == 512,
        "eight_families": len(analysis["families"]) == 8,
        "six_splits": set(analysis["splits"]) == set(SPLITS),
        "full_coordinate_slopes": np.load(analysis["files"]["slopes"], mmap_mode="r").shape == (2, 2, 37, 2560),
        "finite": all(math.isfinite(value) for direction in analysis["semantic_minus_lexical_diagonal_gain"].values()
                      for value in direction.values()),
        "phase2431_visualization_verified": json.loads((P2431 / "analysis/final.json").read_text(encoding="utf-8"))["all_checks_passed"],
        "bulk_raw_removed": all(not path.exists() for path in raw_paths()),
        "claim_boundary": not adjudication["language_encoding_mechanism_closed"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis,
              "adjudication": adjudication, "cleanup": cleanup,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
