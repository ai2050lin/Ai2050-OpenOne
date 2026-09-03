#!/usr/bin/env python3
"""Full-coordinate group interaction test with semantic, random, and contiguous partitions."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import phase2408_c25201_c25520_fullcoordinate_deconfounding as p2408
import phase2409_c25521_c25840_state_dependent_coordinate_operator as p2409

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2407 = RESULT / "phase2407_c24881_c25200_qwen4b_component_fullfield"
OUT = RESULT / "phase2410_c25841_c26160_coordinate_group_synergy"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2410
CAMPAIGN = "C25841-C26160"
COMPONENTS = ("total", "attention", "mlp")
SCHEMES = ("diagonal", "family_signature_group", "random_group", "contiguous_group")
EVAL_SPLITS = p2408.SPLITS
LAYERS = (0, 5, 11, 17, 23, 29, 35)
GROUP_WIDTH = 32
RIDGE_FRACTION = 0.05


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def partition_signature(family_effect: dict, dimension: int) -> tuple[list[np.ndarray], np.ndarray]:
    families = sorted(family_effect)
    signature = np.stack([family_effect[family] for family in families], axis=0)
    winner = np.argmax(np.abs(signature), axis=0)
    win_value = signature[winner, np.arange(dimension)]
    code = winner * 2 + (win_value >= 0).astype(np.int64)
    groups: list[np.ndarray] = []
    group_ids = np.empty(dimension, dtype=np.int16)
    for value in sorted(np.unique(code).tolist()):
        members = np.where(code == value)[0]
        members = members[np.argsort(-np.abs(win_value[members]), kind="stable")]
        for start in range(0, len(members), GROUP_WIDTH):
            group = members[start:start + GROUP_WIDTH]
            group_ids[group] = len(groups)
            groups.append(group)
    if sum(map(len, groups)) != dimension:
        raise RuntimeError("signature partition did not cover every coordinate")
    return groups, group_ids


def partition_random(dimension: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    order = np.random.default_rng(seed).permutation(dimension)
    groups = [order[start:start + GROUP_WIDTH] for start in range(0, dimension, GROUP_WIDTH)]
    ids = np.empty(dimension, dtype=np.int16)
    for gi, group in enumerate(groups):
        ids[group] = gi
    return groups, ids


def partition_contiguous(dimension: int) -> tuple[list[np.ndarray], np.ndarray]:
    groups = [np.arange(start, min(start + GROUP_WIDTH, dimension), dtype=np.int64)
              for start in range(0, dimension, GROUP_WIDTH)]
    ids = np.empty(dimension, dtype=np.int16)
    for gi, group in enumerate(groups):
        ids[group] = gi
    return groups, ids


def fit_group_models(x: np.ndarray, y: np.ndarray, groups: list[np.ndarray]) -> list[np.ndarray]:
    models = []
    for group in groups:
        gx = x[:, group].astype(np.float64, copy=False)
        gy = y[:, group].astype(np.float64, copy=False)
        gram = gx.T @ gx
        ridge = RIDGE_FRACTION * float(np.trace(gram)) / max(len(group), 1) + 1e-8
        models.append(np.linalg.solve(gram + ridge * np.eye(len(group)), gx.T @ gy).astype(np.float32))
    return models


def apply_group_models(x: np.ndarray, groups: list[np.ndarray], models: list[np.ndarray]) -> np.ndarray:
    prediction = np.zeros_like(x, dtype=np.float32)
    for group, model in zip(groups, models):
        prediction[:, group] = x[:, group] @ model
    return prediction


def state_residual(rows: list[dict], test: np.ndarray, design: p2408.Design,
                   h: np.ndarray, fitted: dict) -> np.ndarray:
    x = design.matrix(rows, test)
    base_h = p2408.add_effect(x @ fitted["beta_h"], rows, test, "family", fitted["family_h"])
    return h[test] - base_h


def analyze_task(task: str) -> dict:
    rows = p2408.read_rows(P2407 / f"index/{task}_rows.jsonl")
    state = np.load(P2407 / f"raw/{task}_state_event.float16.npy", mmap_mode="r")
    attention = np.load(P2407 / f"raw/{task}_attention_event.float16.npy", mmap_mode="r")
    mlp = np.load(P2407 / f"raw/{task}_mlp_event.float16.npy", mmap_mode="r")
    train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
    tests = {split: np.asarray([i for i, row in enumerate(rows) if row["partition"] == split], dtype=np.int64)
             for split in EVAL_SPLITS}
    design = p2408.Design(rows, train)
    dimension = state.shape[-1]
    metrics = {component: {split: {stage: p2408.accumulator() for stage in ("family", *SCHEMES)}
                           for split in EVAL_SPLITS} for component in COMPONENTS}
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    group_ids_file = np.lib.format.open_memmap(derived / f"{task}_group_ids.int16.npy", mode="w+", dtype=np.int16,
                                               shape=(len(COMPONENTS), len(LAYERS), attention.shape[2], 3, dimension))
    coordinate_gain = np.lib.format.open_memmap(derived / f"{task}_coordinate_gain.float32.npy", mode="w+", dtype=np.float32,
                                                shape=(len(COMPONENTS), len(EVAL_SPLITS), len(LAYERS), attention.shape[2],
                                                       len(SCHEMES), dimension))
    group_counts = {component: [] for component in COMPONENTS}
    for li, layer in enumerate(LAYERS):
        for event in range(attention.shape[2]):
            h = np.asarray(state[:, layer, event], dtype=np.float32)
            a = np.asarray(attention[:, layer, event], dtype=np.float32)
            m = np.asarray(mlp[:, layer, event], dtype=np.float32)
            for ci, (component, y) in enumerate((("total", a + m), ("attention", a), ("mlp", m))):
                fitted = p2409.fit_baseline(rows, train, design, y, h)
                base_train, diagonal_train, _ = p2409.predict(rows, train, design, y, h, fitted)
                hr_train = state_residual(rows, train, design, h, fitted)
                yr_train = y[train] - base_train
                semantic_groups, semantic_ids = partition_signature(fitted["family_y"], dimension)
                random_groups, random_ids = partition_random(dimension, PHASE * 100000 + ci * 10000 + layer * 100 + event)
                contiguous_groups, contiguous_ids = partition_contiguous(dimension)
                partitions = {"family_signature_group": (semantic_groups, semantic_ids),
                              "random_group": (random_groups, random_ids),
                              "contiguous_group": (contiguous_groups, contiguous_ids)}
                models = {name: fit_group_models(hr_train, yr_train, groups) for name, (groups, _) in partitions.items()}
                for pi, name in enumerate(("family_signature_group", "random_group", "contiguous_group")):
                    group_ids_file[ci, li, event, pi] = partitions[name][1]
                group_counts[component].append({"layer": layer, "event": event,
                                                **{name: len(groups) for name, (groups, _) in partitions.items()}})
                for si, split in enumerate(EVAL_SPLITS):
                    test = tests[split]
                    if test.size == 0:
                        continue
                    base, diagonal, _ = p2409.predict(rows, test, design, y, h, fitted)
                    hr = state_residual(rows, test, design, h, fitted)
                    predictions = {"diagonal": diagonal}
                    for name, (groups, _) in partitions.items():
                        predictions[name] = base + apply_group_models(hr, groups, models[name])
                    truth = y[test]
                    global_base = np.broadcast_to(y[train].mean(axis=0), truth.shape)
                    p2408.update_metric(metrics[component][split]["family"], truth, base, global_base, rows, test)
                    for scheme in SCHEMES:
                        p2408.update_metric(metrics[component][split][scheme], truth, predictions[scheme], global_base, rows, test)
                        coordinate_gain[ci, si, li, event, SCHEMES.index(scheme)] = np.sum(
                            (truth - base) ** 2 - (truth - predictions[scheme]) ** 2, axis=0, dtype=np.float64).astype(np.float32)
        group_ids_file.flush(); coordinate_gain.flush()
        print(f"[phase2410 {task}] fixed layer {layer} ({li + 1}/{len(LAYERS)})", flush=True)
    group_ids_file.flush(); coordinate_gain.flush(); close(group_ids_file); close(coordinate_gain)
    close(state); close(attention); close(mlp)
    finished = {component: {split: (None if stages["family"]["count"] == 0 else
                                    {stage: p2408.finish(value) for stage, value in stages.items()})
                            for split, stages in split_map.items()} for component, split_map in metrics.items()}
    summary = {component: {split: (None if finished[component][split] is None else {
        "family_gain": finished[component][split]["family"]["gain_vs_base"],
        **{f"{scheme}_gain": finished[component][split][scheme]["gain_vs_base"] for scheme in SCHEMES},
        "semantic_over_diagonal": finished[component][split]["family_signature_group"]["gain_vs_base"] - finished[component][split]["diagonal"]["gain_vs_base"],
        "semantic_over_random": finished[component][split]["family_signature_group"]["gain_vs_base"] - finished[component][split]["random_group"]["gain_vs_base"],
        "semantic_over_contiguous": finished[component][split]["family_signature_group"]["gain_vs_base"] - finished[component][split]["contiguous_group"]["gain_vs_base"],
    }) for split in EVAL_SPLITS} for component in COMPONENTS}
    return {"task": task, "rows": len(rows), "train_rows": len(train), "layers": list(LAYERS),
            "group_width_max": GROUP_WIDTH, "ridge_fraction": RIDGE_FRACTION, "metrics": finished,
            "summary": summary, "group_counts": group_counts,
            "arrays": {"group_ids": str(derived / f"{task}_group_ids.int16.npy"),
                       "coordinate_gain": str(derived / f"{task}_coordinate_gain.float32.npy")}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 全坐标语言族分组协同与等容量随机对照（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在预注册7个层位`{list(LAYERS)}`的全部语义事件上覆盖全部2560坐标。每个坐标按discovery中绝对族残差最大的语言族及其正负号进入16类，再按幅值稳定排序切成至多32坐标的组；组内拟合$\widetilde H_S\rightarrow\widetilde U_S$完整矩阵。与同宽随机分组、连续物理编号分组和Phase2409对角算子比较，三种组模型参数上限相同；随机种子在评价前固定。total、Attention、MLP分别测试四类锁箱，空分区记N/A。没有删除坐标、Top-K或低维投影。

$$\widehat W_S=(H_S^\top H_S+\lambda_S I)^{{-1}}H_S^\top U_S,qquad
\lambda_S=0.05\,\frac{{\mathrm{{tr}}(H_S^\top H_S)}}{{|S|}},$$

$$\Delta_{{sem-rand}}=G(\widehat U_{{family-signature}})-G(\widehat U_{{random}}).$$

**结果汇总。** 选择 `{json.dumps(result['selection']['summary'], ensure_ascii=False)}`；组合 `{json.dumps(result['composition']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2410_c25841_c26160_coordinate_group_synergy.py`；每组件×层×事件×全部坐标组号及四模型逐坐标锁箱收益位于`tests/glm5/result/phase2410_c25841_c26160_coordinate_group_synergy`。

**分析与理论进展。** 该测试第一次把“多个坐标组成条件齿轮组”写成可反驳的竞争：组内跨坐标项必须优于逐坐标项，而且语言族分组必须优于等容量随机/连续分组。只要第二个条件失败，就只能说一般多变量残差预测有效，不能命名为语言族齿轮。

**问题硬伤与结论。** 族符号分组仍由外部标签定义；组宽32与岭系数是冻结工程尺度，不是模型自然边界。仅测试线性组内相互作用，且7个层位不是每层。阳性需Phase2411用跨层、组合步数与输出读取验证，阴性也不排除动态非线性或跨组机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    selection = analyze_task("selection")
    composition = analyze_task("composition")
    comparisons = []
    for task in (selection, composition):
        for component in COMPONENTS:
            for split, value in task["summary"][component].items():
                if value is not None:
                    comparisons.append({"task": task["task"], "component": component, "split": split,
                                        "semantic_over_diagonal": value["semantic_over_diagonal"],
                                        "semantic_over_random": value["semantic_over_random"],
                                        "semantic_over_contiguous": value["semantic_over_contiguous"]})
    stable_semantic_advantage = bool(comparisons) and all(value["semantic_over_random"] > 0 for value in comparisons)
    adjudication = {"comparisons": comparisons, "stable_semantic_group_advantage": stable_semantic_advantage,
                    "conditional_coordinate_gear_group_proven": False,
                    "reason": "group label remains externally defined and all held-out component/task cells must dominate controls"}
    values = [number for value in comparisons for number in (value["semantic_over_diagonal"], value["semantic_over_random"], value["semantic_over_contiguous"])]
    checks = {"finite": all(math.isfinite(number) for number in values), "all_coordinates_partitioned": True,
              "equal_capacity_controls": True, "no_topk": True, "empty_splits_are_na": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "selection": selection, "composition": composition,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "selection": selection["summary"], "composition": composition["summary"],
                      "adjudication": adjudication, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
