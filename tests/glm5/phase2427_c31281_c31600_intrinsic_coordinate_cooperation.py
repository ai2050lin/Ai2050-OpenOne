#!/usr/bin/env python3
"""Correct zero-variance cells and test intrinsic full-coordinate cooperation groups."""
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
P2426 = RESULT / "phase2426_c30961_c31280_coordinate_identity_multinull"
OUT = RESULT / "phase2427_c31281_c31600_intrinsic_coordinate_cooperation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2427
CAMPAIGN = "C31281-C31600"
GROUPS = 32
GROUP_SIZE = 80
RANDOM_REPEATS = 8
TEST_SPLITS = ("fresh_unit", "template", "joint", "language", "family")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def center(train: np.ndarray, families: np.ndarray, h: np.ndarray, y: np.ndarray, conditioned: bool = True) -> tuple[dict, np.ndarray, np.ndarray]:
    fitted = atlas.fit(train, families, h, y, family_conditioned=conditioned)
    base_h = np.stack([fitted["family_h"].get(families[i], fitted["global_h"]) for i in train])
    base_y = np.stack([fitted["family_y"].get(families[i], fitted["global_y"]) for i in train])
    return fitted, h[train] - base_h, y[train] - base_y


def discover_groups(train: np.ndarray, families: np.ndarray, h: np.ndarray, y: np.ndarray) -> np.ndarray:
    _, x, target = center(train, families, h, y)
    coupling = x * target
    coupling -= coupling.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(coupling * coupling, axis=0))
    normalized = coupling / np.maximum(norms, 1e-12)
    first = int(np.argmax(norms))
    seeds = [first]
    best = normalized.T @ normalized[:, first]
    for _ in range(1, GROUPS):
        seed = int(np.argmin(best))
        seeds.append(seed)
        best = np.maximum(best, normalized.T @ normalized[:, seed])
    labels = np.full(h.shape[1], -1, dtype=np.int16)
    centroids = normalized[:, seeds].T
    for _ in range(6):
        similarities = normalized.T @ centroids.T
        order = np.argsort(-np.max(similarities, axis=1), kind="stable")
        labels.fill(-1); counts = np.zeros(GROUPS, dtype=np.int64)
        for coordinate in order:
            for group in np.argsort(-similarities[coordinate], kind="stable"):
                if counts[group] < GROUP_SIZE:
                    labels[coordinate] = group; counts[group] += 1; break
        if np.any(labels < 0) or np.any(counts != GROUP_SIZE):
            raise RuntimeError(("balanced_assignment", counts.tolist(), int(np.sum(labels < 0))))
        centroids = np.stack([normalized[:, labels == group].mean(axis=1) for group in range(GROUPS)])
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    return labels


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    table = np.zeros((GROUPS, GROUPS), dtype=np.int64)
    for left, right in zip(a, b):
        table[int(left), int(right)] += 1
    choose2 = lambda x: x * (x - 1) / 2
    sum_comb = float(np.sum(choose2(table)))
    row_comb = float(np.sum(choose2(table.sum(axis=1))))
    col_comb = float(np.sum(choose2(table.sum(axis=0))))
    total = choose2(len(a)); expected = row_comb * col_comb / max(total, 1)
    maximum = .5 * (row_comb + col_comb)
    return (sum_comb - expected) / max(maximum - expected, 1e-30)


def group_list(labels: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(labels == group) for group in range(GROUPS)]


def fit_group(train: np.ndarray, families: np.ndarray, h: np.ndarray, y: np.ndarray,
              labels: np.ndarray, conditioned: bool) -> dict:
    fitted, x, target = center(train, families, h, y, conditioned)
    matrices = []
    for coordinates in group_list(labels):
        design = x[:, coordinates].astype(np.float64)
        response = target[:, coordinates].astype(np.float64)
        gram = design.T @ design
        regularization = 1e-2 * float(np.trace(gram)) / max(len(coordinates), 1) + 1e-6
        matrices.append(np.linalg.solve(gram + regularization * np.eye(len(coordinates)), design.T @ response).astype(np.float32))
    return {"base": fitted, "matrices": matrices}


def predict_group(test: np.ndarray, families: np.ndarray, h: np.ndarray, fitted: dict,
                  labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = fitted["base"]
    base_h = np.stack([base["family_h"].get(families[i], base["global_h"]) for i in test])
    base_y = np.stack([base["family_y"].get(families[i], base["global_y"]) for i in test])
    prediction = base_y.copy()
    for coordinates, matrix in zip(group_list(labels), fitted["matrices"]):
        prediction[:, coordinates] += (h[test][:, coordinates] - base_h[:, coordinates]) @ matrix
    return np.broadcast_to(base["global_y"], prediction.shape), prediction


def gain(truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> float:
    denominator = float(np.sum((truth - baseline) ** 2))
    scale = float(np.sum(truth * truth))
    if denominator <= max(1e-20, scale * 1e-12):
        return 0.0
    return 1 - float(np.sum((truth - prediction) ** 2)) / denominator


def labels_contiguous(dim: int) -> np.ndarray:
    return np.repeat(np.arange(GROUPS, dtype=np.int16), GROUP_SIZE)[:dim]


def labels_random(dim: int, seed: int) -> np.ndarray:
    permutation = np.random.default_rng(seed).permutation(dim)
    labels = np.empty(dim, dtype=np.int16)
    labels[permutation] = labels_contiguous(dim)
    return labels


def partitions(split: str, specs: dict, families: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    train, test, conditioned = specs[split]
    if split != "family":
        return [(train, test, conditioned)]
    return [(train[families[train] != family], test[families[test] == family], False) for family in sorted(set(families))]


def evaluate_split(split: str, specs: dict, families: np.ndarray, h: np.ndarray, y: np.ndarray,
                   intrinsic: np.ndarray, random_labels: list[np.ndarray]) -> dict:
    truth_parts, base_parts = [], []
    predictions = {"diagonal": [], "intrinsic": [], "contiguous": []}
    random_predictions = [[] for _ in random_labels]
    for train, test, conditioned in partitions(split, specs, families):
        diagonal = atlas.fit(train, families, h, y, family_conditioned=conditioned)
        global_p, _, diagonal_p = atlas.predict(test, families, h, diagonal)
        truth_parts.append(y[test]); base_parts.append(global_p); predictions["diagonal"].append(diagonal_p)
        for name, labels in (("intrinsic", intrinsic), ("contiguous", labels_contiguous(h.shape[1]))):
            fitted = fit_group(train, families, h, y, labels, conditioned)
            _, prediction = predict_group(test, families, h, fitted, labels)
            predictions[name].append(prediction)
        for repeat, labels in enumerate(random_labels):
            fitted = fit_group(train, families, h, y, labels, conditioned)
            _, prediction = predict_group(test, families, h, fitted, labels)
            random_predictions[repeat].append(prediction)
    truth, base = np.concatenate(truth_parts), np.concatenate(base_parts)
    values = {name: gain(truth, np.concatenate(parts), base) for name, parts in predictions.items()}
    random = [gain(truth, np.concatenate(parts), base) for parts in random_predictions]
    random_q95 = float(np.quantile(random, .95))
    values.update({"random_mean": float(np.mean(random)), "random_q95": random_q95,
                   "intrinsic_over_diagonal": values["intrinsic"] - values["diagonal"],
                   "intrinsic_over_contiguous": values["intrinsic"] - values["contiguous"],
                   "intrinsic_over_random_q95": values["intrinsic"] - random_q95})
    values["intrinsic_wins_all"] = all(values[key] > 0 for key in
                                        ("intrinsic_over_diagonal", "intrinsic_over_contiguous", "intrinsic_over_random_q95"))
    return values


def analyze(rows: list[dict], collection: dict, corrected: dict) -> dict:
    meta, index = atlas.configuration_index(rows)
    families = np.asarray([row["family"] for row in meta], dtype=object)
    specs = atlas.split_specs(meta, families)
    unit = np.asarray([int(row["unit"]) for row in meta])
    language = np.asarray([row["language"] for row in meta], dtype=object)
    controlled = np.asarray([row["surface_class"] == "controlled" for row in meta])
    discovery = np.flatnonzero(controlled & (language == "en") & (unit < 4))
    half_a = np.flatnonzero(controlled & (language == "en") & (unit < 2))
    half_b = np.flatnonzero(controlled & (language == "en") & (unit >= 2) & (unit < 4))
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    selections = corrected["analysis"]["selections"]["semantic_validity"]
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    results, stability, group_files = {}, {}, {}
    for ci, component in enumerate(atlas.COMPONENTS):
        layer, event = selections[component]
        h = atlas.interactions(state, layer, event, index)[0]
        a = atlas.interactions(attention, layer, event, index)[0]
        m = atlas.interactions(mlp, layer, event, index)[0]
        y = (a + m, a, m)[ci]
        intrinsic = discover_groups(discovery, families, h, y)
        group_a = discover_groups(half_a, families, h, y)
        group_b = discover_groups(half_b, families, h, y)
        path = derived / f"{component}_intrinsic_group_labels.int16.npy"; np.save(path, intrinsic)
        group_files[component] = str(path)
        stability[component] = {"half_a_half_b_adjusted_rand": adjusted_rand(group_a, group_b),
                                "discovery_rows": len(discovery), "half_rows": [len(half_a), len(half_b)],
                                "groups": GROUPS, "group_size_min": int(min(np.bincount(intrinsic))),
                                "group_size_max": int(max(np.bincount(intrinsic)))}
        random_labels = [labels_random(h.shape[1], PHASE * 1000 + ci * 100 + repeat) for repeat in range(RANDOM_REPEATS)]
        results[component] = {split: evaluate_split(split, specs, families, h, y, intrinsic, random_labels)
                              for split in TEST_SPLITS}
        print(f"[phase2427] {component}", flush=True)
    for value in (state, attention, mlp):
        close(value)
    summary = {component: {"intrinsic_win_rate": float(np.mean([value["intrinsic_wins_all"] for value in splits.values()])),
                           "mean_intrinsic_gain": float(np.mean([value["intrinsic"] for value in splits.values()])),
                           "worst_intrinsic_margin": float(min(value[key] for value in splits.values()
                                                               for key in ("intrinsic_over_diagonal", "intrinsic_over_contiguous", "intrinsic_over_random_q95"))),
                           "half_stability_ari": stability[component]["half_a_half_b_adjusted_rand"]}
               for component, splits in results.items()}
    return {"selections": selections, "discovery": "English controlled unit0-3 only", "stability": stability,
            "results": results, "summary": summary, "group_files": group_files}


def corrected_digest(p2425: dict, p2426: dict) -> dict:
    return {
        "supersedes": ["Phase2425 initial gain fields", "Phase2426 initial [layer0,fact-event] selection and all 1.0 gains"],
        "cause": "Before the query, source/target prompts have identical causal prefixes, so the dual-role interaction is exactly zero. The old 1-SSE/(SSE+1e-30) implementation mapped a zero-variance 0/0 cell to gain 1.",
        "fix": "If baseline SSE is below max(1e-20, signal_energy*1e-12), score the cell as non-informative gain 0; recompute every layer/event and every null.",
        "phase2425_version": p2425["analysis"]["analysis_version"],
        "phase2425_adjudication": p2425["adjudication"],
        "phase2426_version": p2426["analysis"]["analysis_version"],
        "phase2426_selections": p2426["analysis"]["selections"],
        "phase2426_summary": p2426["analysis"]["summary"],
        "unaffected": "Raw fields, interaction energies, component closure and Phase2423 behavior are unchanged.",
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 零方差正式更正与内生全坐标协同组竞赛（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 首先执行强制数值审计：事实谓词事件早于查询，source/target两条prompt在该事件拥有完全相同的因果前缀，双角色交互严格为零；旧增益公式把该零方差cell的$0/0$错误映射成1。本Phase以阈值守卫重算Phase2425/2426全部增益并正式追加更正。随后在不看中文与锁箱的英文controlled unit0–3上，用每坐标的$(H_i-\bar H_{{f,i}})(U_i-\bar U_{{f,i}})$完整样本纹理发现32个平衡协同组，每组80个原始坐标；以组内80×80岭映射与逐坐标对角、连续80坐标组、8组随机平衡组竞争，在五锁箱输出完整2560坐标。

$$C_{{n,i}}=(H_{{n,i}}-\bar H_{{f(n),i}})(U_{{n,i}}-\bar U_{{f(n),i}}),$$

$$B_g=(X_g^TX_g+\lambda_g I)^{{-1}}X_g^TY_g,\qquad
\hat U_g=\bar U_{{f,g}}+(H_g-\bar H_{{f,g}})B_g.$$

**结果汇总。** 正式更正 `{json.dumps(result['correction'], ensure_ascii=False)}`；内生组半样本稳定性 `{json.dumps(result['analysis']['stability'], ensure_ascii=False)}`；五锁箱模型竞赛 `{json.dumps(result['analysis']['results'], ensure_ascii=False)}`；汇总 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2427_c31281_c31600_intrinsic_coordinate_cooperation.py`；三个组件各2560维的平衡组标签位于`tests/glm5/result/phase2427_c31281_c31600_intrinsic_coordinate_cooperation/derived`，逐split竞赛和final位于`analysis`。Phase2425/2426的`analysis/final.json`及派生数组已用`v2_zero_variance_guard`重算；MEMO遵守append-only，由本Phase更正而不篡改旧记录。未修改其他Markdown。

**分析与理论进展。** 修正后语义能量相对词项约2倍的观察不变，但旧的完美增益与layer0/fact-event最佳cell完全撤销。多重null显示total与MLP在五锁箱都保留物理坐标身份，Attention只在部分锁箱通过；然而词项对照的total/MLP也通过，因此它更像通用残差坐标耦合，而非语义专属齿轮。本Phase进一步问这种耦合是否属于由数据内生发现且可复现的多坐标组，而不是人为连续块或随机块。

**问题硬伤与结论。** 内生组发现仍是一个基础相关纹理聚类，不保证组内存在不可约联合因果；80×80岭有更多参数，必须同时胜过随机同容量组和对角模型才有意义。半样本ARI检验组成员稳定性，低稳定性意味着组只是有限样本切分。cell曾用中英确认集择优，语言锁箱存在选择层面的轻微泄漏，因此任何语言阳性仍需下一阶段固定相对深度复测。当前不宣称条件齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2424 / "index/semantic_validity_rows.jsonl")
    collection = json.loads((P2424 / "analysis/final.json").read_text(encoding="utf-8"))["collection"]
    p2425 = json.loads((P2425 / "analysis/final.json").read_text(encoding="utf-8"))
    p2426 = json.loads((P2426 / "analysis/final.json").read_text(encoding="utf-8"))
    correction = corrected_digest(p2425, p2426)
    analysis = analyze(rows, collection, p2426)
    stable = all(value["half_a_half_b_adjusted_rand"] > .5 for value in analysis["stability"].values())
    wins = all(value["intrinsic_win_rate"] == 1 for value in analysis["summary"].values())
    adjudication = {"zero_variance_bug_corrected": True,
                    "intrinsic_groups_stable_all_components": stable,
                    "intrinsic_groups_win_all_controls_all_splits": wins,
                    "stable_reusable_coordinate_groups_detected": stable and wins,
                    "conditional_coordinate_gear_proven": False}
    checks = {"corrected_phase2425": p2425["analysis"].get("analysis_version") == "v2_zero_variance_guard",
              "corrected_phase2426": p2426["analysis"].get("analysis_version") == "v2_zero_variance_guard",
              "three_full_coordinate_groupings": all(np.load(path).shape == (2560,) for path in analysis["group_files"].values()),
              "balanced_32_by_80": all(value["groups"] == 32 and value["group_size_min"] == 80 and value["group_size_max"] == 80
                                         for value in analysis["stability"].values()),
              "five_lockboxes": all(set(value) == set(TEST_SPLITS) for value in analysis["results"].values()),
              "finite": all(math.isfinite(number) for value in analysis["summary"].values() for number in value.values()),
              "raw_retained": all(Path(item["path"]).exists() for item in collection.values()),
              "claim_boundary": not adjudication["conditional_coordinate_gear_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "correction": correction, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
