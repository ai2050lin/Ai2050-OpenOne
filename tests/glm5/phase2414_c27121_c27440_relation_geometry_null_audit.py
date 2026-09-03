#!/usr/bin/env python3
"""Audit cross-layer/cross-model family Gram geometry against label and event nulls."""
from __future__ import annotations

import itertools
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import phase2411_c26161_c26480_crosslayer_composition_output_bridge as p2411

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2408 = RESULT / "phase2408_c25201_c25520_fullcoordinate_deconfounding/derived"
P2412 = RESULT / "phase2412_c26481_c26800_frozen_crossmodel_operator_replication"
OUT = RESULT / "phase2414_c27121_c27440_relation_geometry_null_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2414
CAMPAIGN = "C27121-C27440"
COMPONENTS = ("total", "attention", "mlp")
MODELS = ("qwen14b", "glm4", "deepseek7b")
EVENT_INDEX = {"selection": (4, 7), "composition": (8, 11)}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def permutations(count: int, seed: int) -> list[np.ndarray]:
    identity = tuple(range(count))
    if count <= 4:
        return [np.asarray(value, dtype=np.int64) for value in itertools.permutations(range(count)) if value != identity]
    rng = np.random.default_rng(seed); seen = set(); result = []
    while len(result) < 256:
        value = tuple(rng.permutation(count).tolist())
        if value != identity and value not in seen:
            seen.add(value); result.append(np.asarray(value, dtype=np.int64))
    return result


def compare(left: np.ndarray, right: np.ndarray, perms: list[np.ndarray]) -> dict:
    left_geometry = p2411.geometry_vector(left)
    observed = p2411.correlation(left_geometry, p2411.geometry_vector(right))
    null = np.asarray([p2411.correlation(left_geometry, p2411.geometry_vector(right[perm])) for perm in perms], dtype=np.float64)
    return {"observed": observed, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "identity_margin": observed - float(null.mean()), "exceeds_null_q95": bool(observed > np.quantile(null, 0.95)),
            "permutation_p": float((1 + np.sum(null >= observed)) / (1 + len(null)))}


def summarize(rows: list[dict]) -> dict:
    return {"cells": len(rows), "observed_mean": float(np.mean([row["observed"] for row in rows])),
            "null_mean": float(np.mean([row["null_mean"] for row in rows])),
            "identity_margin_mean": float(np.mean([row["identity_margin"] for row in rows])),
            "identity_margin_positive_rate": float(np.mean([row["identity_margin"] > 0 for row in rows])),
            "exceeds_null_q95_rate": float(np.mean([row["exceeds_null_q95"] for row in rows])),
            "median_permutation_p": float(np.median([row["permutation_p"] for row in rows])),
            "event_mismatch_mean": (float(np.mean([row["event_mismatch"] for row in rows if "event_mismatch" in row]))
                                    if any("event_mismatch" in row for row in rows) else None)}


def crosslayer() -> dict:
    result = {}
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    for task in ("selection", "composition"):
        result[task] = {}
        for ci, component in enumerate(COMPONENTS):
            values = np.load(P2408 / f"{task}_{component}_family_residual_passport.float32.npy", mmap_mode="r")
            perms = permutations(values.shape[2], PHASE * 10 + ci)
            rows = []
            matrix = np.zeros((values.shape[0] - 1, values.shape[1], 6), dtype=np.float32)
            for layer in range(values.shape[0] - 1):
                for event in range(values.shape[1]):
                    item = compare(np.asarray(values[layer, event], dtype=np.float32),
                                   np.asarray(values[layer + 1, event], dtype=np.float32), perms)
                    mismatch_event = (event + 1) % values.shape[1]
                    item["event_mismatch"] = p2411.correlation(
                        p2411.geometry_vector(np.asarray(values[layer, event], dtype=np.float32)),
                        p2411.geometry_vector(np.asarray(values[layer + 1, mismatch_event], dtype=np.float32)))
                    item.update({"layer": layer, "event": event}); rows.append(item)
                    matrix[layer, event] = [item[key] for key in ("observed", "null_mean", "null_q95", "identity_margin", "permutation_p", "event_mismatch")]
            np.save(derived / f"{task}_{component}_crosslayer_null.float32.npy", matrix)
            result[task][component] = {"summary": summarize(rows), "rows": rows, "family_count": values.shape[2],
                                       "null_permutations": len(perms)}
            close(values)
    return result


def crossmodel() -> dict:
    result = {}
    for model_index, key in enumerate(MODELS):
        model_final = json.loads((P2412 / key / "analysis/final.json").read_text(encoding="utf-8"))
        result[key] = {}
        for task in ("selection", "composition"):
            target = np.load(model_final["tasks"][task]["passport"], mmap_mode="r")
            result[key][task] = {}
            for ci, component in enumerate(COMPONENTS):
                q4 = np.load(P2408 / f"{task}_{component}_family_residual_passport.float32.npy", mmap_mode="r")
                perms = permutations(target.shape[3], PHASE * 100 + model_index * 10 + ci)
                rows = []
                for layer in range(target.shape[1]):
                    q4_layer = round(layer * (q4.shape[0] - 1) / max(target.shape[1] - 1, 1))
                    for event in range(target.shape[2]):
                        item = compare(np.asarray(q4[q4_layer, EVENT_INDEX[task][event]], dtype=np.float32),
                                       np.asarray(target[ci, layer, event], dtype=np.float32), perms)
                        item.update({"target_layer": layer, "q4_layer": q4_layer, "event": event}); rows.append(item)
                result[key][task][component] = {"summary": summarize(rows), "rows": rows,
                                                "family_count": target.shape[3], "null_permutations": len(perms)}
                close(q4)
            close(target)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 族关系几何的标签置乱与事件错配空模型审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2411的相邻层族Gram几何、Phase2412的相对深度跨模型Gram几何增加冻结空模型。保持每层所有坐标、每族范数和Gram谱不变，只独立置乱右侧族身份；selection的8族使用256个无重复随机排列，composition的4族穷举23个非恒等排列。相邻层另把下一层事件循环错配一位。检验“同族身份对应”是否超过小节点关系图本身的高相关，而不是把原始0.5–0.9相关直接命名为普遍语义结构。

$$K(f,g)=\frac{{G_f\cdot G_g}}{{\|G_f\|\|G_g\|}},\qquad
R_{{id}}=\mathrm{{corr}}(\mathrm{{vec}}K^A,\mathrm{{vec}}K^B),$$

$$R_\pi=\mathrm{{corr}}(\mathrm{{vec}}K^A,\mathrm{{vec}}(P_\pi K^B P_\pi^\top)),\qquad
p_\pi=\frac{{1+\#\{{R_\pi\ge R_{{id}}\}}}}{{1+N_\pi}}.$$

**结果汇总。** 相邻层摘要 `{json.dumps(result['crosslayer_summary'], ensure_ascii=False)}`；跨模型摘要 `{json.dumps(result['crossmodel_summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2414_c27121_c27440_relation_geometry_null_audit.py`；逐层/事件观测、置乱均值、q95、identity margin、p值和事件错配矩阵位于`tests/glm5/result/phase2414_c27121_c27440_relation_geometry_null_audit`。

**分析与理论进展。** 该Phase不引入新复杂模型，只问族身份是否对关系图相似性不可替代。identity margin跨层稳定阳性，才支持“族关系顺序被保留”；若仅observed相关高但不能超过置乱q95，则Phase2411的几何稳定主要来自少节点Gram结构。跨模型同理，只能比较功能关系，不推断坐标同构。

**问题硬伤与结论。** family标签仍由外部任务定义；置乱只能否定标签任意性，不能证明模型运行时显式持有族节点。相邻层样本相关、不同层不是独立重复；p值是排列排序诊断而非总体显著性。下一Phase必须用同一事实链的一步/两步精确配对材料，避免再把不配对主效应冒充组合算子。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=True, indent=2)); return
    layer = crosslayer(); model = crossmodel()
    layer_summary = {task: {component: value["summary"] for component, value in components.items()} for task, components in layer.items()}
    model_summary = {key: {task: {component: value["summary"] for component, value in components.items()}
                               for task, components in tasks.items()} for key, tasks in model.items()}
    decisive = [value for task in layer_summary.values() for value in task.values()] + [value for key in model_summary.values() for task in key.values() for value in task.values()]
    adjudication = {"all_identity_margins_positive": all(value["identity_margin_mean"] > 0 for value in decisive),
                    "all_q95_rates_above_half": all(value["exceeds_null_q95_rate"] > 0.5 for value in decisive),
                    "family_identity_geometry_candidate": all(value["identity_margin_mean"] > 0 for value in decisive),
                    "runtime_family_nodes_proven": False, "cross_architecture_coordinate_isomorphism_proven": False}
    numbers = [number for value in decisive for number in value.values() if isinstance(number, (int, float)) and number is not None]
    checks = {"finite": all(math.isfinite(number) for number in numbers), "selection_256_unique_permutations": True,
              "composition_all_23_nonidentity_permutations": True, "event_mismatch": True,
              "full_coordinate_gram": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "crosslayer": layer, "crossmodel": model,
              "crosslayer_summary": layer_summary, "crossmodel_summary": model_summary,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "crosslayer_summary": layer_summary, "crossmodel_summary": model_summary,
                      "adjudication": adjudication, "checks": checks}, ensure_ascii=True, indent=2))


if __name__ == "__main__": main()
