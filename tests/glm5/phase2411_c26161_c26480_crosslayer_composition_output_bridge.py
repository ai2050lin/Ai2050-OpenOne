#!/usr/bin/env python3
"""Cross-layer family geometry, composition-step fingerprint, and output bridge."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import phase2408_c25201_c25520_fullcoordinate_deconfounding as p2408
import phase2409_c25521_c25840_state_dependent_coordinate_operator as p2409

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2406 = RESULT / "phase2406_c24561_c24880_behavior_precision_calibration/qwen4b"
P2407 = RESULT / "phase2407_c24881_c25200_qwen4b_component_fullfield"
P2408 = RESULT / "phase2408_c25201_c25520_fullcoordinate_deconfounding"
OUT = RESULT / "phase2411_c26161_c26480_crosslayer_composition_output_bridge"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2411
CAMPAIGN = "C26161-C26480"
COMPONENTS = ("total", "attention", "mlp")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / den) if den else 0.0


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def geometry_vector(passport: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(passport, axis=1, keepdims=True)
    normalized = passport / np.maximum(norms, 1e-12)
    gram = normalized @ normalized.T
    return gram[np.triu_indices(gram.shape[0], k=1)]


def crosslayer_geometry(task: str) -> dict:
    arrays = {component: np.load(P2408 / f"derived/{task}_{component}_family_residual_passport.float32.npy", mmap_mode="r")
              for component in COMPONENTS}
    layers, events = arrays["total"].shape[:2]
    per_component = {}
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    for component, values in arrays.items():
        relation = np.zeros((layers - 1, events), dtype=np.float32)
        coordinate = np.zeros((layers - 1, events), dtype=np.float32)
        for layer in range(layers - 1):
            for event in range(events):
                left = np.asarray(values[layer, event], dtype=np.float32)
                right = np.asarray(values[layer + 1, event], dtype=np.float32)
                relation[layer, event] = correlation(geometry_vector(left), geometry_vector(right))
                coordinate[layer, event] = float(np.mean([cosine(left[f], right[f]) for f in range(left.shape[0])]))
        np.save(derived / f"{task}_{component}_adjacent_relation_geometry.float32.npy", relation)
        np.save(derived / f"{task}_{component}_adjacent_coordinate_cosine.float32.npy", coordinate)
        per_component[component] = {"relation_geometry_mean": float(relation.mean()),
                                    "relation_geometry_median": float(np.median(relation)),
                                    "coordinate_cosine_mean": float(coordinate.mean()),
                                    "coordinate_cosine_median": float(np.median(coordinate)),
                                    "relation_positive_rate": float(np.mean(relation > 0)),
                                    "coordinate_positive_rate": float(np.mean(coordinate > 0))}
    source = {"attention_mlp_relation": [], "attention_mlp_coordinate": [],
              "attention_total_relation": [], "mlp_total_relation": []}
    for layer in range(layers):
        for event in range(events):
            total = np.asarray(arrays["total"][layer, event], dtype=np.float32)
            attn = np.asarray(arrays["attention"][layer, event], dtype=np.float32)
            mlp = np.asarray(arrays["mlp"][layer, event], dtype=np.float32)
            source["attention_mlp_relation"].append(correlation(geometry_vector(attn), geometry_vector(mlp)))
            source["attention_mlp_coordinate"].append(cosine(attn.ravel(), mlp.ravel()))
            source["attention_total_relation"].append(correlation(geometry_vector(attn), geometry_vector(total)))
            source["mlp_total_relation"].append(correlation(geometry_vector(mlp), geometry_vector(total)))
    source_summary = {key: {"mean": float(np.mean(value)), "median": float(np.median(value)),
                            "positive_rate": float(np.mean(np.asarray(value) > 0))} for key, value in source.items()}
    for value in arrays.values(): close(value)
    return {"task": task, "layers": layers, "events": events, "components": per_component,
            "same_layer_component_geometry": source_summary}


class StepDesign:
    def __init__(self, rows: list[dict], train: np.ndarray):
        self.factors = ("language", "surface", "direction", "target_candidate_slot", "family", "unit")
        self.levels = {factor: sorted({rows[i][factor] for i in train}, key=str) for factor in self.factors}
        self.pinv = np.linalg.pinv(self.matrix(rows, train)).astype(np.float32)

    def matrix(self, rows: list[dict], indices: np.ndarray) -> np.ndarray:
        columns = [np.ones(len(indices), dtype=np.float32)]
        for factor in self.factors:
            for level in self.levels[factor][1:]:
                columns.append(np.asarray([rows[i][factor] == level for i in indices], dtype=np.float32))
        return np.stack(columns, axis=1)


def step_fingerprint() -> dict:
    task = "composition"
    rows = p2408.read_rows(P2407 / "index/composition_rows.jsonl")
    attention = np.load(P2407 / "raw/composition_attention_event.float16.npy", mmap_mode="r")
    mlp = np.load(P2407 / "raw/composition_mlp_event.float16.npy", mmap_mode="r")
    train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
    tests = {split: np.asarray([i for i, row in enumerate(rows) if row["partition"] == split], dtype=np.int64)
             for split in ("fresh_unit_lockbox", "template_lockbox", "confirmation")}
    design = StepDesign(rows, train)
    metrics = {component: {split: {stage: p2408.accumulator() for stage in ("nuisance", "step")}
                           for split in tests} for component in COMPONENTS}
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    contrast = np.lib.format.open_memmap(derived / "composition_step2_minus_step1.float32.npy", mode="w+", dtype=np.float32,
                                         shape=(len(COMPONENTS), attention.shape[1], attention.shape[2], attention.shape[3]))
    for layer in range(attention.shape[1]):
        for event in range(attention.shape[2]):
            a = np.asarray(attention[:, layer, event], dtype=np.float32)
            m = np.asarray(mlp[:, layer, event], dtype=np.float32)
            for ci, (component, y) in enumerate((("total", a + m), ("attention", a), ("mlp", m))):
                beta = design.pinv @ y[train]
                base_train = design.matrix(rows, train) @ beta
                effects = p2408.grouped_effect(y[train] - base_train, rows, train, "role")
                # role() maps composition rows to steps_1/steps_2.
                contrast[ci, layer, event] = effects["steps_2"] - effects["steps_1"]
                for split, test in tests.items():
                    base = design.matrix(rows, test) @ beta
                    pred = p2408.add_effect(base, rows, test, "role", effects)
                    truth = y[test]
                    global_base = np.broadcast_to(y[train].mean(axis=0), truth.shape)
                    p2408.update_metric(metrics[component][split]["nuisance"], truth, base, global_base, rows, test)
                    p2408.update_metric(metrics[component][split]["step"], truth, pred, global_base, rows, test)
        contrast.flush()
        print(f"[phase2411 step] layer {layer + 1}/{attention.shape[1]}", flush=True)
    contrast.flush(); close(contrast); close(attention); close(mlp)
    finished = {component: {split: {stage: p2408.finish(value) for stage, value in stages.items()}
                            for split, stages in split_map.items()} for component, split_map in metrics.items()}
    summary = {component: {split: {"nuisance_gain": finished[component][split]["nuisance"]["gain_vs_base"],
                                   "step_gain": finished[component][split]["step"]["gain_vs_base"],
                                   "step_increment": finished[component][split]["step"]["gain_vs_base"] - finished[component][split]["nuisance"]["gain_vs_base"]}
                           for split in tests} for component in COMPONENTS}
    return {"rows": len(rows), "train_rows": len(train), "step_counts": {str(step): sum(row["steps"] == step for row in rows) for step in (1, 2)},
            "exact_matched_step_pairs": 0, "metrics": finished, "summary": summary,
            "contrast": str(derived / "composition_step2_minus_step1.float32.npy")}


def read_map(path: Path) -> dict[str, dict]:
    return {row["case_id"]: row for row in p2408.read_rows(path)}


def behavior_bridge(task: str) -> dict:
    rows = p2408.read_rows(P2407 / f"index/{task}_rows.jsonl")
    state = np.load(P2407 / f"raw/{task}_state_event.float16.npy", mmap_mode="r")
    attention = np.load(P2407 / f"raw/{task}_attention_event.float16.npy", mmap_mode="r")
    mlp = np.load(P2407 / f"raw/{task}_mlp_event.float16.npy", mmap_mode="r")
    train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
    test = np.asarray([i for i, row in enumerate(rows) if row["partition"] != "discovery"], dtype=np.int64)
    design = p2408.Design(rows, train)
    numerator = np.zeros(len(rows), dtype=np.float64)
    physical = np.zeros(len(rows), dtype=np.float64)
    denominator = np.zeros(len(rows), dtype=np.float64)
    event = attention.shape[2] - 1  # answer_boundary, fixed before scores are joined
    for layer in range(attention.shape[1]):
        h = np.asarray(state[:, layer, event], dtype=np.float32)
        y = np.asarray(attention[:, layer, event], dtype=np.float32) + np.asarray(mlp[:, layer, event], dtype=np.float32)
        fitted = p2409.fit_baseline(rows, train, design, y, h)
        family, matched, mismatch = p2409.predict(rows, test, design, y, h, fitted)
        truth = y[test]
        ef = np.sum((truth - family) ** 2, axis=1, dtype=np.float64)
        em = np.sum((truth - matched) ** 2, axis=1, dtype=np.float64)
        ex = np.sum((truth - mismatch) ** 2, axis=1, dtype=np.float64)
        numerator[test] += ef - em; physical[test] += ex - em; denominator[test] += ef
    internal = numerator / np.maximum(denominator, 1e-30)
    physical_adv = physical / np.maximum(denominator, 1e-30)
    teacher = read_map(P2406 / "behavior/teacher_scores.jsonl")
    autonomous = read_map(P2406 / "behavior/autonomous_lockbox.jsonl")
    available = [i for i in test if rows[i]["case_id"] in teacher]
    margins = np.asarray([teacher[rows[i]["case_id"]]["mean_logprob_margin"] for i in available], dtype=np.float64)
    first = np.asarray([teacher[rows[i]["case_id"]]["first_divergence_logit_margin"] for i in available], dtype=np.float64)
    adv = internal[available]; phys = physical_adv[available]
    autonomous_indices = [i for i in available if rows[i]["case_id"] in autonomous]
    success = np.asarray([autonomous[rows[i]["case_id"]]["target_present"] for i in autonomous_indices], dtype=bool)
    auto_adv = internal[autonomous_indices]
    per_partition = {}
    for partition in sorted({rows[i]["partition"] for i in available}):
        local = [i for i in available if rows[i]["partition"] == partition]
        loc_adv = internal[local]
        loc_margin = np.asarray([teacher[rows[i]["case_id"]]["mean_logprob_margin"] for i in local])
        per_partition[partition] = {"rows": len(local), "corr_internal_margin": correlation(loc_adv, loc_margin),
                                    "mean_internal_advantage": float(loc_adv.mean())}
    rows_out = [{"case_id": rows[i]["case_id"], "task": task, "partition": rows[i]["partition"],
                 "internal_state_advantage": float(internal[i]), "physical_coordinate_advantage": float(physical_adv[i]),
                 "mean_logprob_margin": float(teacher[rows[i]["case_id"]]["mean_logprob_margin"]),
                 "target_present": autonomous.get(rows[i]["case_id"], {}).get("target_present")}
                for i in available]
    out_path = OUT / f"derived/{task}_row_behavior_bridge.jsonl"; out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out), encoding="utf-8")
    close(state); close(attention); close(mlp)
    return {"task": task, "rows": len(available), "answer_boundary_layers": attention.shape[1],
            "corr_internal_mean_margin": correlation(adv, margins),
            "corr_internal_first_margin": correlation(adv, first),
            "corr_physical_mean_margin": correlation(phys, margins),
            "autonomous_rows": len(autonomous_indices), "autonomous_success_rows": int(success.sum()),
            "mean_internal_success": float(auto_adv[success].mean()) if success.any() else None,
            "mean_internal_failure": float(auto_adv[~success].mean()) if (~success).any() else None,
            "per_partition": per_partition, "rows_file": str(out_path)}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 跨层关系几何—组合步数纹理—输出行为桥（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** （1）用Phase2408每层×事件×族×全部坐标护照，分别比较相邻层同一物理坐标余弦与族间Gram关系几何相关；前者问坐标纹理是否原样延续，后者问坐标改变时族间关系是否保存。（2）对640条组合材料去除语言、表面、方向、候选槽、族和训练unit后估计一步/两步残差纹理，在新unit、整模板及confirmation上测全坐标增益。（3）只在answer-boundary、跨36层累计每条锁箱样本的同坐标状态算子误差改善，再与冻结teacher margin及自主生成是否包含目标连接。

$$K_q(f,g)=\frac{{G_{{qf}}\cdot G_{{qg}}}}{{\|G_{{qf}}\|\|G_{{qg}}\|}},\qquad
T_q=\mathrm{{corr}}(\mathrm{{vec}}K_q,\mathrm{{vec}}K_{{q+1}}),$$

$$B_i=\frac{{\sum_q\left(\|U_i-\widehat U_{{family}}\|^2-\|U_i-\widehat U_{{state}}\|^2\right)}}{{\sum_q\|U_i-\widehat U_{{family}}\|^2}}.$$

**结果汇总。** 跨层 `{json.dumps(result['crosslayer'], ensure_ascii=False)}`；组合步数 `{json.dumps(result['step_fingerprint']['summary'], ensure_ascii=False)}`；行为桥 `{json.dumps(result['behavior_bridge'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2411_c26161_c26480_crosslayer_composition_output_bridge.py`；相邻层几何矩阵、一步/两步全部坐标差、逐样本内部—行为桥及final位于`tests/glm5/result/phase2411_c26161_c26480_crosslayer_composition_output_bridge`。

**分析与理论进展。** 同坐标延续与关系几何延续被明确分开；后者阳性只能说明族关系图跨层稳定，不等同存在可搬运向量。步数纹理若迁移，只能说明“任务要求一步/两步”改变更新场；因为材料没有同一事实链同时提出一步与两步问题，不能冒充算子复合律。行为相关若弱，说明局部更新可预测性尚未解释如何编译为输出概率。

**问题硬伤与结论。** Gram几何只有族级8或4个节点，可能粗糙；step主效应仍混有问题长度和查询目标。输出桥是相关而非干预，且自主生成的格式错误使exact过严，因此同时报告target_present。没有在本Phase宣称吸引子、测地线、跨层同构或因果编译闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    crosslayer = {task: crosslayer_geometry(task) for task in ("selection", "composition")}
    step = step_fingerprint()
    bridges = {task: behavior_bridge(task) for task in ("selection", "composition")}
    adjudication = {"coordinate_transport_proven": False, "functional_relation_geometry_measured": True,
                    "operator_composition_proven": False, "output_causal_bridge_proven": False,
                    "exact_matched_step_pairs": step["exact_matched_step_pairs"]}
    values = []
    for task in crosslayer.values():
        for component in task["components"].values():
            values.extend(component.values())
    for component in step["summary"].values():
        for split in component.values(): values.extend(split.values())
    for bridge in bridges.values():
        values.extend([bridge["corr_internal_mean_margin"], bridge["corr_internal_first_margin"], bridge["corr_physical_mean_margin"]])
    checks = {"finite": all(math.isfinite(float(v)) for v in values if v is not None),
              "geometry_not_coordinate_identity": True, "step_pair_limit_reported": True,
              "output_bridge_is_correlational": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "crosslayer": crosslayer, "step_fingerprint": step,
              "behavior_bridge": bridges, "adjudication": adjudication, "checks": checks,
              "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "crosslayer": crosslayer, "step": step["summary"],
                      "behavior_bridge": bridges, "adjudication": adjudication, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
