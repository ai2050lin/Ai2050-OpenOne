#!/usr/bin/env python3
"""Test semantic interaction coordinate identity against four families of nulls."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import phase2425_c30641_c30960_semantic_specific_interaction_atlas as atlas


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2424 = RESULT / "phase2424_c30321_c30640_semantic_validity_multievent_fullfield"
P2425 = RESULT / "phase2425_c30641_c30960_semantic_specific_interaction_atlas"
OUT = RESULT / "phase2426_c30961_c31280_coordinate_identity_multinull"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2426
CAMPAIGN = "C30961-C31280"
TEST_SPLITS = ("fresh_unit", "template", "joint", "language", "family")
SHIFTS = (1, 31, 127, 509, 791, 1021, 1531, 2047)
REPEATS = 8
SAMPLE_REPEATS = 16
ANALYSIS_VERSION = "v2_zero_variance_guard"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def grouped_permutation(meta: list[dict], test: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.asarray(test).copy()
    groups: dict[tuple, list[int]] = {}
    for local, index in enumerate(test):
        row = meta[index]
        key = (row["family"], row["language"], row["surface"], int(row["direction"]))
        groups.setdefault(key, []).append(local)
    for locals_ in groups.values():
        values = np.asarray(locals_, dtype=np.int64)
        result[values] = test[rng.permutation(values)]
    return result


def permutation_bank(h: np.ndarray, train: np.ndarray, seed: int) -> dict[str, list[np.ndarray]]:
    dim = h.shape[1]
    random = [np.random.default_rng(seed + i).permutation(dim) for i in range(REPEATS)]
    std = np.std(h[train], axis=0)
    ordered = np.argsort(std, kind="stable")
    bins = np.array_split(ordered, 32)
    variance = []
    for repeat in range(REPEATS):
        rng = np.random.default_rng(seed + 100 + repeat)
        perm = np.arange(dim)
        for values in bins:
            perm[values] = rng.permutation(values)
        variance.append(perm)
    return {"shift": [np.roll(np.arange(dim), shift) for shift in SHIFTS], "random": random, "variance_bin": variance}


def gain(truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> float:
    denominator = float(np.sum((truth - baseline) ** 2))
    scale = float(np.sum(truth * truth))
    if denominator <= max(1e-20, scale * 1e-12):
        return 0.0
    return 1 - float(np.sum((truth - prediction) ** 2)) / denominator


def partitions_for(split: str, specs: dict, families: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    train, test, conditioned = specs[split]
    if split != "family":
        return [(train, test, conditioned)]
    return [(train[families[train] != family], test[families[test] == family], False) for family in sorted(set(families))]


def evaluate(meta: list[dict], families: np.ndarray, specs: dict, split: str, h: np.ndarray, y: np.ndarray,
             seed: int) -> dict:
    matched_truth, matched_pred, baseline, family_pred = [], [], [], []
    null_predictions = {name: [[] for _ in range(REPEATS)] for name in ("shift", "random", "variance_bin")}
    sample_predictions = [[] for _ in range(SAMPLE_REPEATS)]
    for part_index, (train, test, conditioned) in enumerate(partitions_for(split, specs, families)):
        fitted = atlas.fit(train, families, h, y, family_conditioned=conditioned)
        global_p, family_p, state_p = atlas.predict(test, families, h, fitted)
        matched_truth.append(y[test]); matched_pred.append(state_p); baseline.append(global_p); family_pred.append(family_p)
        bank = permutation_bank(h, train, seed + part_index * 1000)
        family_h = np.stack([fitted["family_h"].get(families[i], fitted["global_h"]) for i in test])
        for name, permutations in bank.items():
            for repeat, permutation in enumerate(permutations):
                slope = fitted["slope"][permutation]
                null_predictions[name][repeat].append(family_p + (h[test] - family_h) * slope)
        for repeat in range(SAMPLE_REPEATS):
            shuffled = grouped_permutation(meta, test, seed + part_index * 1000 + 500 + repeat)
            shuffled_h = np.stack([fitted["family_h"].get(families[i], fitted["global_h"]) for i in shuffled])
            sample_predictions[repeat].append(family_p + (h[shuffled] - shuffled_h) * fitted["slope"])
    truth = np.concatenate(matched_truth); base = np.concatenate(baseline); family_p = np.concatenate(family_pred)
    matched = gain(truth, np.concatenate(matched_pred), base)
    family_gain = gain(truth, family_p, base)
    nulls = {name: [gain(truth, np.concatenate(values), base) for values in repetitions]
             for name, repetitions in null_predictions.items()}
    nulls["sample_shuffle"] = [gain(truth, np.concatenate(values), base) for values in sample_predictions]
    summary = {name: {"mean": float(np.mean(values)), "q95": float(np.quantile(values, .95)),
                      "matched_over_q95": matched - float(np.quantile(values, .95))}
               for name, values in nulls.items()}
    return {"rows": len(truth), "family_gain": family_gain, "matched_gain": matched,
            "state_increment": matched - family_gain, "nulls": summary,
            "beats_all_null_q95": all(value["matched_over_q95"] > 0 for value in summary.values())}


def analyze(rows: list[dict], collection: dict, prior: dict) -> dict:
    meta, index = atlas.configuration_index(rows)
    families = np.asarray([row["family"] for row in meta], dtype=object)
    specs = atlas.split_specs(meta, families)
    metrics = np.load(prior["analysis"]["files"]["metrics"], mmap_mode="r")
    confirmation = atlas.SPLITS.index("confirmation")
    selections = {}
    for ii, interaction in enumerate(atlas.INTERACTIONS):
        selections[interaction] = {}
        for ci, component in enumerate(atlas.COMPONENTS):
            values = np.asarray(metrics[ii, ci, confirmation, 2], dtype=np.float32)
            selections[interaction][component] = [int(x) for x in np.unravel_index(np.argmax(values), values.shape)]
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    results = {}
    for ii, interaction in enumerate(atlas.INTERACTIONS):
        results[interaction] = {}
        for ci, component in enumerate(atlas.COMPONENTS):
            layer, event = selections[interaction][component]
            h_pair = atlas.interactions(state, layer, event, index)
            a_pair = atlas.interactions(attention, layer, event, index)
            m_pair = atlas.interactions(mlp, layer, event, index)
            h = h_pair[ii]
            y = ((a_pair[ii] + m_pair[ii]), a_pair[ii], m_pair[ii])[ci]
            results[interaction][component] = {split: evaluate(meta, families, specs, split, h, y,
                                                                PHASE * 10000 + ii * 1000 + ci * 100 + si)
                                                  for si, split in enumerate(TEST_SPLITS)}
            print(f"[phase2426] {interaction} {component}", flush=True)
    for value in (metrics, state, attention, mlp):
        close(value)
    summary = {interaction: {component: {
        "beats_all_nulls_split_rate": float(np.mean([value["beats_all_null_q95"] for value in splits.values()])),
        "mean_matched_gain": float(np.mean([value["matched_gain"] for value in splits.values()])),
        "mean_state_increment": float(np.mean([value["state_increment"] for value in splits.values()])),
        "worst_null_margin": float(min(null["matched_over_q95"] for value in splits.values() for null in value["nulls"].values()))}
        for component, splits in components.items()} for interaction, components in results.items()}
    return {"analysis_version": ANALYSIS_VERSION, "selections": selections, "test_splits": list(TEST_SPLITS), "results": results, "summary": summary}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 语义交互的物理坐标身份多重零假设（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 只用Phase2425的unit4–5确认集为每个交互×组件选择一个layer/event，随后冻结该选择，在fresh-unit、template、joint、英文到中文和留一家族五个独立锁箱检验。matched逐坐标斜率同时对抗8个循环移位、8个全随机坐标排列、8个按训练状态标准差分32 bin后bin内排列，以及16个保持家族×语言×表面×方向的unit样本置乱。所有预测输出仍是原始2560物理坐标；没有把场压缩成Top-K。

$$\hat U_i^{{\pi}}=\bar U_{{f,i}}+\beta_{{\pi(i)}}(H_i-\bar H_{{f,i}}),\qquad
\hat U_{{n}}^{{sample}}=\bar U_f+\beta\odot(H_{{\sigma(n)}}-\bar H_f),$$

$$M_N=G_{{matched}}-Q_{{.95}}\{{G_{{null}}\}}.$$

**结果汇总。** 冻结cell `{json.dumps(result['analysis']['selections'], ensure_ascii=False)}`；五锁箱逐零假设结果 `{json.dumps(result['analysis']['results'], ensure_ascii=False)}`；汇总 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2426_c30961_c31280_coordinate_identity_multinull.py`；选择、逐interaction×组件×split的matched/family/四类null结果及final位于`tests/glm5/result/phase2426_c30961_c31280_coordinate_identity_multinull`。全坐标斜率继续引用Phase2425派生数组。未修改其他Markdown。

**分析与理论进展。** 该Phase不再用单个+791错位代表全部坐标证据。随机排列排除任意编号，等方差bin排列排除“只要幅值等级相同”，样本置乱排除仅由家族/模板均值造成的假增益。只有语义交互而非词项交互在五锁箱对四类null均有正margin，才支持固定基底上的语义专属坐标身份。

**问题硬伤与结论。** cell由确认集择优，虽不接触五个测试锁箱，仍存在36×3多重搜索；本Phase报告最差margin而不是只报最佳值。对角模型不能发现真正的多坐标非线性条件组；Phase2427继续检验协同。样本置乱组内有时只有两个unit，null分辨率有限。通过坐标身份门也只说明可复用局部对应，不等于齿轮或因果机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8"))
        if result.get("analysis", {}).get("analysis_version") == ANALYSIS_VERSION:
            append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2424 / "index/semantic_validity_rows.jsonl")
    collection = json.loads((P2424 / "analysis/final.json").read_text(encoding="utf-8"))["collection"]
    prior = json.loads((P2425 / "analysis/final.json").read_text(encoding="utf-8"))
    analysis = analyze(rows, collection, prior)
    semantic_pass = all(analysis["results"]["semantic_validity"][component][split]["beats_all_null_q95"]
                        for component in atlas.COMPONENTS for split in TEST_SPLITS)
    lexical_pass = all(analysis["results"]["lexical_control"][component][split]["beats_all_null_q95"]
                       for component in atlas.COMPONENTS for split in TEST_SPLITS)
    adjudication = {"semantic_beats_all_nulls_all_components_splits": semantic_pass,
                    "lexical_beats_all_nulls_all_components_splits": lexical_pass,
                    "semantic_specific_physical_coordinate_identity_detected": semantic_pass and not lexical_pass,
                    "conditional_coordinate_gear_proven": False}
    checks = {"six_selected_semantic_lexical_components": sum(len(value) for value in analysis["selections"].values()) == 6,
              "five_independent_splits": set(analysis["test_splits"]) == set(TEST_SPLITS),
              "four_null_families": all(set(value["nulls"]) == {"shift", "random", "variance_bin", "sample_shuffle"}
                                        for interaction in analysis["results"].values() for component in interaction.values() for value in component.values()),
              "finite": all(math.isfinite(number) for interaction in analysis["results"].values() for component in interaction.values()
                            for value in component.values() for number in [value["matched_gain"], value["state_increment"]]),
              "raw_retained": all(Path(item["path"]).exists() for item in collection.values()),
              "claim_boundary": not adjudication["conditional_coordinate_gear_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
