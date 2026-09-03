#!/usr/bin/env python3
"""Confirm shared and family-conditioned full-coordinate responses on 128 unseen rows."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2315 = RESULT / "phase2315_c5041_c5100_active_response_contract"
P2322 = RESULT / "phase2322_c5521_c5600_full_coordinate_reuse_passports"
P2323 = RESULT / "phase2323_c5601_c5680_qwen4b_fp16_fresh_prospective"
P2324 = RESULT / "phase2324_c5681_c5720_fresh_prospective_atlas_cleanup"
OUT = RESULT / "phase2325_c5721_c5800_qwen4b_fp16_large_family_confirmation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = P2315 / "material/natural_active_response_bilingual.jsonl"
PHASE = 2325
CAMPAIGN = "C5721-C5800"
UNITS = (17, 18)
EPS = 1e-12
GATES = {
    "global_mean_relative_mse_max": 0.35,
    "family_over_global_median_ratio_max": 0.97,
    "family_win_count_min": 5,
    "frozen_stable_sign_agreement_min": 0.90,
    "pair_superposition_relative_mse_max": 0.05,
    "even_to_odd_l2_max": 0.30,
}

sys.path.insert(0, str(TESTS))
import phase1332_bf16_utils as model_base  # noqa: E402
import phase2318_c5241_c5320_crossmodel_directional_topology as cross  # noqa: E402
import phase2322_c5521_c5600_full_coordinate_reuse_passports as passport  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        np.square(actual - predicted, dtype=np.float64).sum()
        / (np.square(actual, dtype=np.float64).sum() + EPS)
    )


def evaluate(derivative_path: Path, index_rows: list[dict]) -> dict:
    _meta, old_cells = passport.load_cells("c5481_qwen4b_fp16_directional_derivative")
    derivative = np.load(derivative_path, mmap_mode="r")
    expected = [128, 3, 6, 3, 2560]
    if list(derivative.shape) != expected:
        raise RuntimeError(("confirmation_derivative_shape", list(derivative.shape), expected))
    records = []
    cell_summaries = []
    for key in sorted(old_cells):
        source_index, probe, target_index = key
        discovery, discovery_rows = old_cells[key]
        _values, old_summary, stable = passport.coordinate_metrics(discovery, discovery_rows)
        global_mean = discovery.mean(axis=0, dtype=np.float64)
        family_means = {
            family: discovery[[i for i, row in enumerate(discovery_rows) if row["family"] == family]]
            .mean(axis=0, dtype=np.float64)
            for family in sorted({row["family"] for row in discovery_rows})
        }
        current = []
        for row_index, row in enumerate(index_rows):
            actual = derivative[row_index, source_index, probe, target_index].astype(np.float64)
            record = {
                "case_id": row["case_id"],
                "family": row["family"],
                "language": row["language"],
                "surface": row["surface"],
                "state": int(row["state"]),
                "unit": int(row["unit"]),
                "source_index": source_index,
                "probe": probe,
                "target_index": target_index,
                "global_relative_mse": relative_mse(actual, global_mean),
                "family_relative_mse": relative_mse(actual, family_means[row["family"]]),
                "stable_coordinates": int(stable.sum()),
                "stable_sign_agreement": (
                    float(np.mean(actual[stable] * global_mean[stable] > 0))
                    if np.any(stable) else None
                ),
            }
            records.append(record)
            current.append(record)
        cell_summaries.append({
            "source_index": source_index,
            "probe": probe,
            "target_index": target_index,
            "discovery_stable_fraction": old_summary["stable_shared_fraction"],
            "stable_coordinates": int(stable.sum()),
            "global_relative_mse_median": float(np.median(
                [row["global_relative_mse"] for row in current]
            )),
            "family_relative_mse_median": float(np.median(
                [row["family_relative_mse"] for row in current]
            )),
            "stable_sign_agreement_median": (
                float(np.median([row["stable_sign_agreement"] for row in current
                                 if row["stable_sign_agreement"] is not None]))
                if any(row["stable_sign_agreement"] is not None for row in current) else None
            ),
        })
    passport.atlas.close_memmap(derivative)
    by_family = {}
    for family in sorted({row["family"] for row in records}):
        values = [row for row in records if row["family"] == family]
        global_median = float(np.median([row["global_relative_mse"] for row in values]))
        family_median = float(np.median([row["family_relative_mse"] for row in values]))
        by_family[family] = {
            "records": len(values),
            "global_relative_mse_median": global_median,
            "family_relative_mse_median": family_median,
            "family_over_global_ratio": family_median / (global_median + EPS),
            "family_better_than_global": family_median < global_median,
            "stable_sign_agreement_median": float(np.median([
                row["stable_sign_agreement"] for row in values
                if row["stable_sign_agreement"] is not None
            ])),
        }
    global_median = float(np.median([row["global_relative_mse"] for row in records]))
    family_median = float(np.median([row["family_relative_mse"] for row in records]))
    result = {
        "records": len(records),
        "cells": len(cell_summaries),
        "rows": len(index_rows),
        "coordinates": 2560,
        "global_relative_mse_median": global_median,
        "family_relative_mse_median": family_median,
        "family_over_global_median_ratio": family_median / (global_median + EPS),
        "family_win_count": sum(value["family_better_than_global"] for value in by_family.values()),
        "frozen_stable_sign_agreement_median": float(np.median([
            row["stable_sign_agreement"] for row in records
            if row["stable_sign_agreement"] is not None
        ])),
        "by_family": by_family,
        "by_cell": cell_summaries,
        "claim_boundary": "128 unseen fresh_confirmation rows predicted from 32 fresh_lockbox rows",
    }
    cross.write_rows(OUT / "analysis/prediction_records.jsonl", records)
    save(OUT / "analysis/prediction_cells.json", cell_summaries)
    save(OUT / "analysis/prediction_summary.json", result)
    return result


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    prediction = result["prediction"]
    compact_prediction = {key: value for key, value in prediction.items() if key != "by_cell"}
    record = rf"""

## Phase {PHASE}: 128 条新词汇上的共享骨架与语言族条件修正确认（{CAMPAIGN}） [{stamp}]

**测试原理、冻结对象与测试用例。** Phase2323 在 32 条新词汇上发现：共享全局均值可预测完整响应，但族均值在 8 族中的 5 族进一步降低误差。本期在模型加载前把该现象改写成可失败的扩大确认合同，选择 `fresh_confirmation` 中与 Phase2323 不重叠的 unit 17–18 共 128 条，八族各 16 条，完整覆盖中英、两种自然表面和两种状态。发现端仍只使用 32 条 `fresh_lockbox`；模型为本地非量化 Qwen3-4B FP16，3 个源深度、4 个固定基方向、2 个成对方向、3 个目标层及 1% 剂量不变。行为与内部响应分账；每个响应保留全部 2560 个原始坐标。

$$
\rho_{{family/global}}=\frac{{\operatorname{{median}}_x E_f(x)}}{{\operatorname{{median}}_x E_g(x)}},\qquad
W=\sum_{{f=1}}^8\mathbf 1[\operatorname{{median}}_{{x\in f}}E_f(x)<\operatorname{{median}}_{{x\in f}}E_g(x)].
$$

$$
E_g(x)=\frac{{\lVert D_x-\bar D\rVert_2^2}}{{\lVert D_x\rVert_2^2+\varepsilon}},\qquad
A_{{sign}}=\operatorname{{median}}_x\frac1{{|S|}}\sum_{{j\in S}}\mathbf1[D_{{x,j}}\bar D_j>0].
$$

**结果汇总与门槛。** 行为汇总 `{json.dumps(result['behavior'], ensure_ascii=False)}`。全坐标扩大确认 `{json.dumps(compact_prediction, ensure_ascii=False)}`。局部响应指标 `{json.dumps(result['functional_metrics'], ensure_ascii=False)}`。冻结门及裁决 `{json.dumps(result['experimental_gates'], ensure_ascii=False)}`：全局误差不高于 `0.35`；族/全局总体误差比不高于 `0.97`；至少 `5/8` 族由族均值胜出；稳定符号一致率不低于 `0.90`；成对误差不高于 `0.05`；偶/奇比不高于 `0.30`。脚本 `tests/glm5/phase2325_c5721_c5800_qwen4b_fp16_large_family_confirmation.py`，结果 `tests/glm5/result/phase2325_c5721_c5800_qwen4b_fp16_large_family_confirmation`。

**分析、理论进展、问题硬伤与结论。** 该合同区分两层规律：全局均值测试跨族共享传播骨架，族均值相对改善测试其上的条件修正。即使全部门通过，也只能说明随机局部响应在新词汇上具有可重复的“共享项加族条件项”统计结构；它不定位语义坐标、不证明族标签是模型天然变量，也不证明唯一因果电路。发现端每族仅 4 条、材料仍为受控双语句、没有独立人类盲评，且同一模板族可能使族均值吸收句长、tokenizer 和表面格式。成对与偶/奇门仍只描述 1% 邻域数值性质。理论主体继续保持“条件化输出场闭合理论”，本期仅在较大样本上确认或淘汰一个具体拼图，不引入新理论名称。下一步仅对重要结果发布全坐标图谱，并在发布校验后清理重复原始场。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    for parent in (P2322, P2323, P2324):
        value = json.loads((parent / "analysis/final.json").read_text(encoding="utf-8"))
        if not value["all_checks_passed"]:
            raise RuntimeError(("parent_not_authorized", parent))
    all_rows = cross.read_rows(ROWS_PATH)
    selected_raw = [row for row in all_rows
                    if row["partition"] == "fresh_confirmation" and int(row["unit"]) in UNITS]
    selected_raw.sort(key=lambda row: row["design_index"])
    overlap = {row["case_id"] for row in selected_raw} & {
        row["case_id"] for row in cross.read_rows(P2323 / "index/active_rows.jsonl")
    }
    family_counts = {family: sum(row["family"] == family for row in selected_raw)
                     for family in sorted({row["family"] for row in selected_raw})}
    stratum_counts = {}
    for language in ("en", "zh"):
        for surface in ("narrative", "reported"):
            for state in (0, 1):
                stratum_counts[f"{language}|{surface}|{state}"] = sum(
                    row["language"] == language and row["surface"] == surface
                    and int(row["state"]) == state for row in selected_raw
                )
    config = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "frozen_before_model_load": True,
        "model": "Qwen3-4B",
        "precision": "float16_nonquantized",
        "discovery_partition": "fresh_lockbox",
        "confirmation_partition": "fresh_confirmation",
        "confirmation_units": list(UNITS),
        "rows": len(selected_raw),
        "family_counts": family_counts,
        "language_surface_state_counts": stratum_counts,
        "overlap_with_phase2323": sorted(overlap),
        "coordinate_policy": "all 2560 physical coordinates; no Top-K, PCA, projection, or reordering",
        "gates": GATES,
        "claim_boundary": "larger model-local confirmation of shared plus family-conditioned response",
    }
    if len(selected_raw) != 128 or set(family_counts.values()) != {16} or overlap:
        raise RuntimeError(("material_contract", len(selected_raw), family_counts, sorted(overlap)))
    if set(stratum_counts.values()) != {16}:
        raise RuntimeError(("stratum_contract", stratum_counts))
    save(OUT / "config/frozen_large_confirmation.json", config)
    model = tokenizer = None
    try:
        model, tokenizer, device = model_base.load_model(
            "qwen3", dtype=torch.float16, use_8bit=False,
        )
        dtypes = model_base.parameter_dtype_counts(model)
        compiled = cross.compile_rows(tokenizer, selected_raw)
        cross.write_rows(OUT / "material/compiled_large_confirmation.jsonl", compiled)
        scores = cross.sequence_scores(model, device, compiled, 32)
        cross.write_rows(OUT / "behavior/sequence_scores.jsonl", scores)
        free = cross.free_generation(model, tokenizer, device, compiled, 32)
        cross.write_rows(OUT / "behavior/free_generation.jsonl", free)
        behavior = cross.behavior_summary(scores, free)
        behavior["claim_boundary"] = "128-row balanced descriptive behavior panel"
        save(OUT / "behavior/summary.json", behavior)
        field = cross.active_capture(model, device, compiled, OUT)
        index_rows = cross.read_rows(OUT / "index/active_rows.jsonl")
        for row in index_rows:
            row["partition"] = "fresh_confirmation"
        cross.write_rows(OUT / "index/active_rows.jsonl", index_rows)
        functional = cross.functional_metrics(
            ROOT / field["derivative"], ROOT / field["even"], ROOT / field["norms"],
            index_rows, "Qwen3-4B-FP16-large-confirmation",
        )
        save(OUT / "analysis/functional_metrics.json", functional)
        prediction = evaluate(ROOT / field["derivative"], index_rows)
        gates = {
            "global_mean_relative_mse": {
                "value": prediction["global_relative_mse_median"],
                "threshold_max": GATES["global_mean_relative_mse_max"],
                "passed": prediction["global_relative_mse_median"] <= GATES["global_mean_relative_mse_max"],
            },
            "family_over_global_median_ratio": {
                "value": prediction["family_over_global_median_ratio"],
                "threshold_max": GATES["family_over_global_median_ratio_max"],
                "passed": prediction["family_over_global_median_ratio"] <= GATES["family_over_global_median_ratio_max"],
            },
            "family_win_count": {
                "value": prediction["family_win_count"],
                "threshold_min": GATES["family_win_count_min"],
                "passed": prediction["family_win_count"] >= GATES["family_win_count_min"],
            },
            "frozen_stable_sign_agreement": {
                "value": prediction["frozen_stable_sign_agreement_median"],
                "threshold_min": GATES["frozen_stable_sign_agreement_min"],
                "passed": prediction["frozen_stable_sign_agreement_median"] >= GATES["frozen_stable_sign_agreement_min"],
            },
            "pair_superposition_relative_mse": {
                "value": functional["median_pair_superposition_relative_mse"],
                "threshold_max": GATES["pair_superposition_relative_mse_max"],
                "passed": functional["median_pair_superposition_relative_mse"] <= GATES["pair_superposition_relative_mse_max"],
            },
            "even_to_odd_l2": {
                "value": functional["median_even_to_odd_l2"],
                "threshold_max": GATES["even_to_odd_l2_max"],
                "passed": functional["median_even_to_odd_l2"] <= GATES["even_to_odd_l2_max"],
            },
        }
        checks = {
            "parents_authorized": True,
            "config_frozen_before_model_load": True,
            "fp16_nonquantized": dtypes.get("float16", 0) > 0,
            "all_128_rows": len(compiled) == 128,
            "no_phase2323_overlap": not overlap,
            "balanced_eight_families": set(family_counts.values()) == {16},
            "balanced_language_surface_state": set(stratum_counts.values()) == {16},
            "all_coordinates": field["shape"][-1] == 2560,
            "all_6912_predictions": prediction["records"] == 6912,
            "all_54_cells": prediction["cells"] == 54,
            "experimental_failure_does_not_invalidate_execution": True,
        }
        result = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": "closed",
            "config": config,
            "parameter_dtypes": dtypes,
            "behavior": behavior,
            "field": field,
            "prediction": prediction,
            "functional_metrics": {
                "prediction": functional["prediction"],
                "families": functional["families"],
                "median_pair_superposition_relative_mse": functional["median_pair_superposition_relative_mse"],
                "median_even_to_odd_l2": functional["median_even_to_odd_l2"],
                "relative_response_topology": functional["relative_response_topology"],
                "claim_boundary": functional["claim_boundary"],
            },
            "experimental_gates": gates,
            "route_passed": all(value["passed"] for value in gates.values()),
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "strict_conclusion": (
                "Shared response plus family-conditioned correction passed the larger confirmation."
                if all(value["passed"] for value in gates.values()) else
                "One or more larger-confirmation gates failed; keep only the gates that passed."
            ),
            "next_authorization": "Publish important exact-coordinate fields and clean duplicate raw arrays.",
        }
        save(final_path, result)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
