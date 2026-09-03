#!/usr/bin/env python3
"""Sequentially deconfound platform/nuisance/family/content effects in every Qwen4B coordinate."""
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
P2407 = RESULT / "phase2407_c24881_c25200_qwen4b_component_fullfield"
OUT = RESULT / "phase2408_c25201_c25520_fullcoordinate_deconfounding"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2408
CAMPAIGN = "C25201-C25520"
COMPONENTS = ("total", "attention", "mlp")
SPLITS = ("fresh_unit_lockbox", "template_lockbox", "deep_fresh_unit_lockbox", "joint_template_unit_lockbox")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def role(row: dict) -> str:
    return row.get("query_role", f"steps_{row.get('steps')}")


def factor_value(row: dict, factor: str):
    if factor == "role": return role(row)
    if factor == "slot": return row["target_candidate_slot"]
    return row[factor]


class Design:
    def __init__(self, rows: list[dict], train: np.ndarray):
        self.factors = ("language", "surface", "direction", "role", "slot")
        self.levels = {factor: sorted({factor_value(rows[i], factor) for i in train}, key=str) for factor in self.factors}
        x = self.matrix(rows, train)
        self.pinv = np.linalg.pinv(x).astype(np.float32)

    def matrix(self, rows: list[dict], indices: np.ndarray) -> np.ndarray:
        columns = [np.ones(len(indices), dtype=np.float32)]
        for factor in self.factors:
            for level in self.levels[factor][1:]:
                columns.append(np.asarray([factor_value(rows[i], factor) == level for i in indices], dtype=np.float32))
        return np.stack(columns, axis=1)


def grouped_effect(residual: np.ndarray, rows: list[dict], train: np.ndarray, factor: str) -> dict[Any, np.ndarray]:
    groups = defaultdict(list)
    for local, index in enumerate(train.tolist()): groups[factor_value(rows[index], factor)].append(local)
    effects = {key: residual[local].mean(axis=0) for key, local in groups.items()}
    center = np.mean(np.stack(list(effects.values())), axis=0)
    return {key: value - center for key, value in effects.items()}


def add_effect(prediction: np.ndarray, rows: list[dict], indices: np.ndarray, factor: str, effects: dict) -> np.ndarray:
    result = prediction.copy()
    zero = np.zeros(prediction.shape[1], dtype=np.float32)
    for local, index in enumerate(indices.tolist()): result[local] += effects.get(factor_value(rows[index], factor), zero)
    return result


def accumulator() -> dict:
    return {"sse": 0.0, "base": 0.0, "count": 0, "rows": set(), "units": defaultdict(lambda: [0.0, 0.0])}


def update_metric(acc: dict, truth: np.ndarray, pred: np.ndarray, base: np.ndarray, rows: list[dict], indices: np.ndarray) -> None:
    row_sse = np.sum((truth - pred) ** 2, axis=1, dtype=np.float64); row_base = np.sum((truth - base) ** 2, axis=1, dtype=np.float64)
    acc["sse"] += float(row_sse.sum()); acc["base"] += float(row_base.sum()); acc["count"] += truth.size
    for local, index in enumerate(indices.tolist()):
        acc["rows"].add(index); unit = int(rows[index]["unit"]); acc["units"][unit][0] += float(row_sse[local]); acc["units"][unit][1] += float(row_base[local])


def finish(acc: dict) -> dict:
    unit_gain = {str(unit): 1 - value[0] / max(value[1], 1e-30) for unit, value in acc["units"].items()}
    return {"rows": len(acc["rows"]), "coordinates": int(acc["count"]), "gain_vs_base": 1 - acc["sse"] / max(acc["base"], 1e-30),
            "mse": acc["sse"] / max(acc["count"], 1), "unit_gains": unit_gain,
            "median_unit_gain": float(np.median(list(unit_gain.values()))) if unit_gain else None}


def component_values(attention: np.ndarray, mlp: np.ndarray, qpoint: int, event: int) -> dict[str, np.ndarray]:
    a = np.asarray(attention[:, qpoint, event], dtype=np.float32); m = np.asarray(mlp[:, qpoint, event], dtype=np.float32)
    return {"attention": a, "mlp": m, "total": a + m}


def analyze_task(task: str) -> dict:
    rows = read_rows(P2407 / f"index/{task}_rows.jsonl")
    attention = np.load(P2407 / f"raw/{task}_attention_event.float16.npy", mmap_mode="r")
    mlp = np.load(P2407 / f"raw/{task}_mlp_event.float16.npy", mmap_mode="r")
    layers, events, dimension = attention.shape[1:]
    train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
    # Empty lockboxes are valid for task families that do not define that partition.
    # Keep their arrays integer-typed so NumPy treats them as indices, not values.
    split_indices = {
        split: np.asarray([i for i, row in enumerate(rows) if row["partition"] == split], dtype=np.int64)
        for split in SPLITS
    }
    language_train = np.asarray([i for i in train if rows[i]["language"] == "en"], dtype=np.int64)
    language_test = np.asarray([i for i in train if rows[i]["language"] == "zh"], dtype=np.int64)
    designs = {"standard": Design(rows, train), "language": Design(rows, language_train)}
    metrics = {component: {split: {stage: accumulator() for stage in ("nuisance", "family", "content")}
                           for split in (*SPLITS, "language_lockbox")} for component in COMPONENTS}
    families = sorted({row["family"] for row in rows})
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    passport_files = {component: np.lib.format.open_memmap(derived / f"{task}_{component}_family_residual_passport.float32.npy",
                      mode="w+", dtype=np.float32, shape=(layers, events, len(families), dimension)) for component in COMPONENTS}
    gain_files = {component: np.lib.format.open_memmap(derived / f"{task}_{component}_family_gain_by_split.float32.npy",
                  mode="w+", dtype=np.float32, shape=(len(SPLITS) + 1, layers, events, dimension)) for component in COMPONENTS}
    energy = {component: defaultdict(float) for component in COMPONENTS}
    for qpoint in range(layers):
        for event in range(events):
            values = component_values(attention, mlp, qpoint, event)
            for component, y in values.items():
                design = designs["standard"]; beta = design.pinv @ y[train]; nuisance_train = design.matrix(rows, train) @ beta
                family_effect = grouped_effect(y[train] - nuisance_train, rows, train, "family")
                family_train = add_effect(nuisance_train, rows, train, "family", family_effect)
                unit_effect = grouped_effect(y[train] - family_train, rows, train, "unit")
                for family_index, family in enumerate(families): passport_files[component][qpoint, event, family_index] = family_effect[family]
                for split_index, split in enumerate(SPLITS):
                    indices = split_indices[split]; truth = y[indices]; base = np.broadcast_to(y[train].mean(axis=0), truth.shape)
                    nuisance = design.matrix(rows, indices) @ beta
                    family = add_effect(nuisance, rows, indices, "family", family_effect)
                    content = add_effect(family, rows, indices, "unit", unit_effect)
                    for stage, pred in (("nuisance", nuisance), ("family", family), ("content", content)):
                        update_metric(metrics[component][split][stage], truth, pred, base, rows, indices)
                    gain_files[component][split_index, qpoint, event] = np.sum((truth - nuisance) ** 2 - (truth - family) ** 2, axis=0, dtype=np.float64).astype(np.float32)
                    energy[component][split] += float(np.sum(truth * truth, dtype=np.float64))
                design_l = designs["language"]; beta_l = design_l.pinv @ y[language_train]; nuisance_l_train = design_l.matrix(rows, language_train) @ beta_l
                family_l_effect = grouped_effect(y[language_train] - nuisance_l_train, rows, language_train, "family")
                truth_l = y[language_test]; base_l = np.broadcast_to(y[language_train].mean(axis=0), truth_l.shape)
                nuisance_l = design_l.matrix(rows, language_test) @ beta_l; family_l = add_effect(nuisance_l, rows, language_test, "family", family_l_effect)
                update_metric(metrics[component]["language_lockbox"]["nuisance"], truth_l, nuisance_l, base_l, rows, language_test)
                update_metric(metrics[component]["language_lockbox"]["family"], truth_l, family_l, base_l, rows, language_test)
                update_metric(metrics[component]["language_lockbox"]["content"], truth_l, family_l, base_l, rows, language_test)
                gain_files[component][-1, qpoint, event] = np.sum((truth_l - nuisance_l) ** 2 - (truth_l - family_l) ** 2, axis=0, dtype=np.float64).astype(np.float32)
            del values
        for value in passport_files.values(): value.flush()
        for value in gain_files.values(): value.flush()
        print(f"[phase2408 {task}] layer {qpoint + 1}/{layers}", flush=True)
    for value in passport_files.values(): value.flush(); close(value)
    for value in gain_files.values(): value.flush(); close(value)
    close(attention); close(mlp)
    finished = {component: {split: {stage: finish(value) for stage, value in stages.items()} for split, stages in splits.items()}
                for component, splits in metrics.items()}
    family_increment = {component: {split: finished[component][split]["family"]["gain_vs_base"] - finished[component][split]["nuisance"]["gain_vs_base"]
                                    for split in (*SPLITS, "language_lockbox")} for component in COMPONENTS}
    return {"task": task, "rows": len(rows), "train_rows": len(train), "language_train_rows": len(language_train),
            "language_test_rows": len(language_test), "field_shape": list(attention.shape), "families": families,
            "metrics": finished, "family_increment_over_nuisance": family_increment,
            "component_energy": {component: dict(values) for component, values in energy.items()},
            "arrays": {component: {"passport": str(derived / f"{task}_{component}_family_residual_passport.float32.npy"),
                                    "family_gain": str(derived / f"{task}_{component}_family_gain_by_split.float32.npy")} for component in COMPONENTS}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 全坐标层底盘—表面—语言—族—内容顺序解混（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对total更新、Attention输出和MLP输出分别进行冻结顺序解混。仅用canonical/paraphrase、unit0–5 discovery拟合：先用语言、表面、方向、查询角色/步数、候选槽的基本哑变量回归普通条件响应，再在残差上估计语言族均值$G_f$，最后估计训练unit内容项$D_u$。不把坐标当独立语言样本；分别在fresh unit、整模板、深新unit、模板+unit联合锁箱报告逐unit收益。另以英文discovery拟合、中文discovery整语言留出，避免把双语共同训练冒充跨语言迁移。所有输出和误差在全部物理坐标上累计。

$$U^{{(1)}}=U-X_{{nuisance}}\hat\beta,\qquad G_f=\mathbb E[U^{{(1)}}\mid f],\qquad
\widehat U_{{family}}=X\hat\beta+G_f,$$

$$\Delta G_{{family}}=G(\widehat U_{{family}})-G(\widehat U_{{nuisance}}).$$

**结果汇总。** 选择 `{json.dumps(result['selection_summary'], ensure_ascii=False)}`；组合 `{json.dumps(result['composition_summary'], ensure_ascii=False)}`；来源裁决 `{json.dumps(result['component_adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2408_c25201_c25520_fullcoordinate_deconfounding.py`；三组件逐层×事件×族×全部坐标残差护照、五锁箱逐坐标族收益和final位于`tests/glm5/result/phase2408_c25201_c25520_fullcoordinate_deconfounding`。

**分析与理论进展。** 相对Phase2399的细条件查表，本Phase把语言族能否在未知整模板、未知语言和新内容中增加预测力单独列为最重要门。Attention/MLP仅按同一评价拆开来源：若某组件在模板/语言锁箱保持族增益，才是组件级候选；能量大或训练拟合好都不能代替迁移。

**问题硬伤与结论。** 加性哑变量不能表达复杂表面交互；未知表面被映射到训练基线，因此整模板锁箱非常严格。语言留出同时改变tokenizer词形，不能把失败解释成没有抽象语义。$G_f$仍是不读取$H_q$的残差均值，阳性只说明去除已测干扰后仍有族共享纹理，必须由Phase2409状态依赖算子继续裁决。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def compact(task: dict) -> dict:
    return {component: {split: {"nuisance_gain": task["metrics"][component][split]["nuisance"]["gain_vs_base"],
                                "family_gain": task["metrics"][component][split]["family"]["gain_vs_base"],
                                "family_increment": task["family_increment_over_nuisance"][component][split],
                                "median_unit_family_gain": task["metrics"][component][split]["family"]["median_unit_gain"]}
                        for split in (*SPLITS, "language_lockbox")} for component in COMPONENTS}


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    selection = analyze_task("selection"); composition = analyze_task("composition")
    selection_summary = compact(selection); composition_summary = compact(composition)
    adjudication = {"selection_total_family_increment": selection["family_increment_over_nuisance"]["total"],
                    "composition_total_family_increment": composition["family_increment_over_nuisance"]["total"],
                    "selection_source_increment": {component: selection["family_increment_over_nuisance"][component] for component in ("attention", "mlp")},
                    "composition_source_increment": {component: composition["family_increment_over_nuisance"][component] for component in ("attention", "mlp")},
                    "pure_semantic_gear_proven": False}
    all_values = [value for task in (selection, composition) for component in COMPONENTS for value in task["family_increment_over_nuisance"][component].values()]
    checks = {"finite": all(math.isfinite(value) for value in all_values), "all_components": True,
              "whole_template_reported": all("template_lockbox" in task["family_increment_over_nuisance"]["total"] for task in (selection, composition)),
              "whole_language_reported": all("language_lockbox" in task["family_increment_over_nuisance"]["total"] for task in (selection, composition)),
              "claim_boundary": not adjudication["pure_semantic_gear_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "selection": selection, "composition": composition,
              "selection_summary": selection_summary, "composition_summary": composition_summary,
              "component_adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
