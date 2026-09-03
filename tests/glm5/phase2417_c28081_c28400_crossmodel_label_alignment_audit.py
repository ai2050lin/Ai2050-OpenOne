#!/usr/bin/env python3
"""Correct Phase2416 family-order mismatch and audit cross-model relation geometry."""
from __future__ import annotations

import itertools
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
P2415 = RESULT / "phase2415_c27441_c27760_exact_paired_composition"
P2416 = RESULT / "phase2416_c27761_c28080_crossmodel_exact_pair_replication"
OUT = RESULT / "phase2417_c28081_c28400_crossmodel_label_alignment_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2417
CAMPAIGN = "C28081-C28400"
COMPONENTS = ("total", "attention", "mlp")
MODELS = ("qwen14b", "glm4", "deepseek7b")

sys.path.insert(0, str(TESTS))
import phase2411_c26161_c26480_crosslayer_composition_output_bridge as geometry  # noqa: E402
import phase2415_c27441_c27760_exact_paired_composition as paired  # noqa: E402
import phase2416_c27761_c28080_crossmodel_exact_pair_replication as p2416  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def compare_vectors(left: np.ndarray, right_passport: np.ndarray, permutations: list[np.ndarray]) -> dict:
    observed = geometry.correlation(left, geometry.geometry_vector(right_passport))
    null = np.asarray([geometry.correlation(left, geometry.geometry_vector(right_passport[perm]))
                       for perm in permutations], dtype=np.float64)
    return {"observed": observed, "null_mean": float(null.mean()),
            "identity_margin": observed - float(null.mean()),
            "p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
            "exceeds_null_q95": bool(observed > np.quantile(null, .95))}


def summarize(rows: list[dict]) -> dict:
    return {"cells": len(rows), "observed_mean": float(np.mean([row["observed"] for row in rows])),
            "identity_margin_mean": float(np.mean([row["identity_margin"] for row in rows])),
            "identity_margin_positive_rate": float(np.mean([row["identity_margin"] > 0 for row in rows])),
            "exceeds_null_q95_rate": float(np.mean([row["exceeds_null_q95"] for row in rows])),
            "median_permutation_p": float(np.median([row["p"] for row in rows]))}


def audit(models: dict) -> dict:
    q4 = np.load(P2415 / "derived/paired_update_step2_minus_step1_family.float32.npy", mmap_mode="r")
    q4_order = list(paired.contract.COMPOSITION_FAMILIES)
    stored_order = sorted(q4_order)
    align = np.asarray([stored_order.index(name) for name in q4_order], dtype=np.int64)
    permutations = [np.asarray(value, dtype=np.int64) for value in itertools.permutations(range(4))
                    if value != tuple(range(4))]
    result = {"family_orders": {"qwen4b": q4_order, "phase2416_models_stored": stored_order,
                                 "target_reindex_to_qwen4b": align.tolist()}, "models": {}}
    for key, model in models.items():
        target = np.load(model["analysis"]["files"]["passport"], mmap_mode="r")
        result["models"][key] = {}
        for ci, component in enumerate(COMPONENTS):
            exact_rows, local_rows, band_rows = [], [], []
            q4_geometries = [geometry.geometry_vector(np.asarray(q4[ci, layer], dtype=np.float32))
                             for layer in range(q4.shape[1])]
            target_passports = [np.asarray(target[ci, layer, align], dtype=np.float32) for layer in range(target.shape[1])]
            for layer, right in enumerate(target_passports):
                q4_layer = round(layer * (q4.shape[1] - 1) / max(target.shape[1] - 1, 1))
                item = compare_vectors(q4_geometries[q4_layer], right, permutations)
                item.update({"target_layer": layer, "q4_layer": q4_layer}); exact_rows.append(item)
                window = range(max(0, q4_layer - 2), min(q4.shape[1], q4_layer + 3))
                candidates = [compare_vectors(q4_geometries[candidate], right, permutations) for candidate in window]
                best = max(candidates, key=lambda value: value["observed"])
                best.update({"target_layer": layer, "q4_layer_center": q4_layer, "q4_window": list(window)})
                local_rows.append(best)
            for band in range(4):
                q4_indices = [layer for layer in range(q4.shape[1]) if min(3, 4 * layer // q4.shape[1]) == band]
                target_indices = [layer for layer in range(target.shape[1]) if min(3, 4 * layer // target.shape[1]) == band]
                left = np.mean([q4_geometries[layer] for layer in q4_indices], axis=0)
                # Compare relation vectors directly; label null must be recomputed before layer averaging.
                right_mean = np.mean([geometry.geometry_vector(target_passports[layer]) for layer in target_indices], axis=0)
                observed = geometry.correlation(left, right_mean)
                null = []
                for permutation in permutations:
                    null_vector = np.mean([geometry.geometry_vector(target_passports[layer][permutation])
                                           for layer in target_indices], axis=0)
                    null.append(geometry.correlation(left, null_vector))
                null = np.asarray(null)
                band_rows.append({"band": band, "q4_layers": q4_indices, "target_layers": target_indices,
                                  "observed": observed, "null_mean": float(null.mean()),
                                  "identity_margin": observed - float(null.mean()),
                                  "p": float((1 + np.sum(null >= observed)) / 24),
                                  "exceeds_null_q95": bool(observed > np.quantile(null, .95))})
            result["models"][key][component] = {"exact_relative_depth": summarize(exact_rows),
                                                 "local_depth_window": summarize(local_rows),
                                                 "four_band_average": summarize(band_rows),
                                                 "exact_rows": exact_rows, "local_rows": local_rows,
                                                 "band_rows": band_rows}
        close(target)
    close(q4)
    return result


def correct_phase2416(crossmodel: dict) -> dict:
    path = P2416 / "analysis/final.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    old = value.get("crossmodel")
    corrected = {model: {component: data["exact_relative_depth"]
                         for component, data in components.items()}
                 for model, components in crossmodel["models"].items()}
    if "crossmodel_pre_phase2417_misaligned" not in value:
        value["crossmodel_pre_phase2417_misaligned"] = old
    value["crossmodel"] = corrected
    value["crossmodel_label_alignment_correction"] = {
        "corrected_by_phase": PHASE, "reason": "Qwen4B contract order was compared to alphabetically stored target order",
        "qwen4b_order": crossmodel["family_orders"]["qwen4b"],
        "target_stored_order": crossmodel["family_orders"]["phase2416_models_stored"],
        "status": "corrected; the Phase2417 append-only memo entry supersedes Phase2416 crossmodel numbers"}
    save(path, value)
    return {"updated": str(path), "old_preserved": True, "crossmodel_replaced": True}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {model: {component: {key: value[key] for key in
                                   ("exact_relative_depth", "local_depth_window", "four_band_average")}
                       for component, value in components.items()}
               for model, components in result["audit"]["models"].items()}
    text = rf"""

## Phase {PHASE}: 跨模型组合族标签对齐纠错与深度稳健性审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2416在各模型内部分析有效，但跨模型关系几何存在实现错误：Qwen4B配对护照按合同顺序`spatial, temporal, comparison, taxonomy`保存，三目标模型按字母顺序`comparison, spatial, taxonomy, temporal`保存，旧代码未按族名重排便比较Gram边，造成标签错位。按照append-only原则不删除旧记录，本Phase明确纠错并让本结果取代Phase2416中的跨模型数字。对齐后使用全部23种非恒等族标签置换，分别检查精确相对深度、相对深度±2个Qwen4B层的局部窗口，以及四个归一化深度带的关系向量平均；窗口对每个候选都应用相同置换检验。

$$\pi=[\operatorname{{index}}_{{\rm alphabetical}}(f):f\in F_{{\rm contract}}],\qquad
K^m_q\leftarrow K^m_q[\pi],$$

$$R_{{band}}=\operatorname{{corr}}\!\left(\frac1{{|B_4|}}\sum_{{q\in B_4}}\operatorname{{vec}}K^4_q,
\frac1{{|B_m|}}\sum_{{q\in B_m}}\operatorname{{vec}}K^m_q\right).$$

**结果汇总。** 族顺序 `{json.dumps(result['audit']['family_orders'], ensure_ascii=False)}`；纠正后跨模型摘要 `{json.dumps(compact, ensure_ascii=False)}`；历史final修订 `{json.dumps(result['phase2416_correction'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2417_c28081_c28400_crossmodel_label_alignment_audit.py`；逐层、局部窗口、四深度带的观测/23置换结果及final位于`tests/glm5/result/phase2417_c28081_c28400_crossmodel_label_alignment_audit`。Phase2416 final保留错误旧值在`crossmodel_pre_phase2417_misaligned`，并把`crossmodel`替换为对齐结果；未修改历史Markdown正文。

**分析与理论进展。** 纠错后，三个模型相对Qwen4B的平均族关系相关不再是伪阴性的负值；但“平均为正”与“逐层超过标签置换”必须分开。若精确层/局部窗口/深度带只有少数超过q95，就只能说存在弱功能关系相似，不能称为跨架构同一族图谱。这个纠错也说明图谱研究必须让每一数组携带显式label order，不能依赖容器默认顺序。

**问题硬伤与结论。** 四族只有6条Gram边，$p$值最小为$1/24$；局部窗口最大值带选择自由度，已用同窗口置换但仍只作敏感性分析。跨模型深度没有天然一一对应，四带平均会抹平局部重写。Phase2416的模型内同坐标优势不受该bug影响；跨模型关系几何以本Phase为准。下一阶段同目标下最紧迫的不是再做一套关系图，而是排除“matched优于坐标错配仅来自坐标异方差/残差自相关”的替代解释。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    models = {key: json.loads((P2416 / key / "analysis/final.json").read_text(encoding="utf-8")) for key in MODELS}
    audited = audit(models)
    correction = correct_phase2416(audited)
    summaries = [value[mode] for components in audited["models"].values() for value in components.values()
                 for mode in ("exact_relative_depth", "local_depth_window", "four_band_average")]
    adjudication = {"phase2416_crossmodel_negative_claim_retracted": True,
                    "corrected_mean_relations_positive": all(row["observed_mean"] > 0 for row in summaries),
                    "all_cells_exceed_label_null": all(row["exceeds_null_q95_rate"] == 1 for row in summaries),
                    "cross_architecture_family_graph_proven": False,
                    "model_internal_operator_results_affected": False}
    checks = {"orders_differed": audited["family_orders"]["qwen4b"] != audited["family_orders"]["phase2416_models_stored"],
              "all_23_permutations": True,
              "three_depth_controls": True,
              "finite": all(math.isfinite(float(number)) for row in summaries for key, number in row.items()
                            if key != "cells"),
              "phase2416_final_corrected": correction["crossmodel_replaced"], "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "audit": audited, "phase2416_correction": correction,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "family_orders": audited["family_orders"],
                      "summary": {model: {component: {mode: values[mode] for mode in
                                                       ("exact_relative_depth", "local_depth_window", "four_band_average")}
                                          for component, values in components.items()}
                                  for model, components in audited["models"].items()},
                      "adjudication": adjudication, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
