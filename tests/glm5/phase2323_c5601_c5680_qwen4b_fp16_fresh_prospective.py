#!/usr/bin/env python3
"""Prospectively test frozen full-coordinate response rules on fresh vocabulary."""
from __future__ import annotations

import gc
import json
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
P2315 = RESULT / "phase2315_c5041_c5100_active_response_contract"
P2321 = RESULT / "phase2321_c5481_c5520_fp16_atlas_cleanup"
P2322 = RESULT / "phase2322_c5521_c5600_full_coordinate_reuse_passports"
OUT = RESULT / "phase2323_c5601_c5680_qwen4b_fp16_fresh_prospective"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = P2315 / "material/natural_active_response_bilingual.jsonl"
PHASE = 2323
CAMPAIGN = "C5601-C5680"
PARTITION = "fresh_confirmation"
EPS = 1e-12
GATES = {
    "global_mean_relative_mse_max": 0.35,
    "global_better_than_family_all_families": True,
    "frozen_stable_sign_agreement_min": 0.80,
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


def frozen_config() -> dict:
    parent = json.loads((P2322 / "analysis/final.json").read_text(encoding="utf-8"))
    return {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "frozen_before_model_load": True,
        "model": "Qwen3-4B",
        "precision": "float16_nonquantized",
        "discovery_partition": "fresh_lockbox",
        "prospective_partition": PARTITION,
        "prospective_rows": 32,
        "source_dataset": "c5481_qwen4b_fp16_directional_derivative",
        "source_phase": 2320,
        "passport_phase": 2322,
        "coordinate_policy": "all 2560 model-local physical coordinates in original order",
        "prediction_models": ["zero", "frozen_global_mean", "frozen_family_mean"],
        "stable_mask_definition": parent["config"]["thresholds"],
        "gates": GATES,
        "behavior_policy": "candidate and free generation reported separately; not used to alter internal gates",
        "claim_boundary": (
            "prospective model-local random-direction response prediction; no language-family semantic "
            "gear, unique circuit, or cross-model coordinate claim"
        ),
    }


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    numerator = np.square(actual - predicted, dtype=np.float64).sum()
    denominator = np.square(actual, dtype=np.float64).sum() + EPS
    return float(numerator / denominator)


def evaluate_frozen_predictions(
    derivative_path: Path,
    index_rows: list[dict],
) -> dict:
    _meta, discovery_cells = passport.load_cells("c5481_qwen4b_fp16_directional_derivative")
    derivative = np.load(derivative_path, mmap_mode="r")
    if list(derivative.shape) != [32, 3, 6, 3, 2560]:
        raise RuntimeError(("prospective_derivative_shape", list(derivative.shape)))
    records = []
    cells = []
    for key in sorted(discovery_cells):
        source_index, probe, target_index = key
        discovery, discovery_rows = discovery_cells[key]
        _passport_values, discovery_summary, stable_mask = passport.coordinate_metrics(
            discovery, discovery_rows,
        )
        global_mean = discovery.mean(axis=0, dtype=np.float64)
        family_means = {
            family: discovery[[i for i, row in enumerate(discovery_rows) if row["family"] == family]]
            .mean(axis=0, dtype=np.float64)
            for family in sorted({row["family"] for row in discovery_rows})
        }
        cell_records = []
        for row_index, row in enumerate(index_rows):
            actual = derivative[row_index, source_index, probe, target_index].astype(np.float64)
            family_mean = family_means[row["family"]]
            stable_sign = (
                float(np.mean(actual[stable_mask] * global_mean[stable_mask] > 0))
                if np.any(stable_mask) else None
            )
            record = {
                "case_id": row["case_id"],
                "family": row["family"],
                "language": row["language"],
                "surface": row["surface"],
                "state": int(row["state"]),
                "source_index": source_index,
                "probe": probe,
                "target_index": target_index,
                "zero_relative_mse": 1.0,
                "global_relative_mse": relative_mse(actual, global_mean),
                "family_relative_mse": relative_mse(actual, family_mean),
                "frozen_stable_coordinates": int(stable_mask.sum()),
                "frozen_stable_sign_agreement": stable_sign,
            }
            records.append(record)
            cell_records.append(record)
        cells.append({
            "source_index": source_index,
            "probe": probe,
            "target_index": target_index,
            "discovery_stable_fraction": discovery_summary["stable_shared_fraction"],
            "discovery_stable_coordinates": int(stable_mask.sum()),
            "prospective_global_relative_mse_median": float(np.median(
                [row["global_relative_mse"] for row in cell_records]
            )),
            "prospective_family_relative_mse_median": float(np.median(
                [row["family_relative_mse"] for row in cell_records]
            )),
            "prospective_stable_sign_agreement_median": (
                float(np.median([row["frozen_stable_sign_agreement"] for row in cell_records
                                 if row["frozen_stable_sign_agreement"] is not None]))
                if any(row["frozen_stable_sign_agreement"] is not None for row in cell_records)
                else None
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
            "global_better_than_family": global_median < family_median,
            "stable_sign_agreement_median": float(np.median([
                row["frozen_stable_sign_agreement"] for row in values
                if row["frozen_stable_sign_agreement"] is not None
            ])),
        }
    result = {
        "records": len(records),
        "cells": len(cells),
        "coordinates": 2560,
        "global_relative_mse_median": float(np.median(
            [row["global_relative_mse"] for row in records]
        )),
        "family_relative_mse_median": float(np.median(
            [row["family_relative_mse"] for row in records]
        )),
        "frozen_stable_sign_agreement_median": float(np.median([
            row["frozen_stable_sign_agreement"] for row in records
            if row["frozen_stable_sign_agreement"] is not None
        ])),
        "by_family": by_family,
        "by_cell": cells,
        "claim_boundary": "frozen fresh_lockbox response statistics predicting fresh_confirmation",
    }
    cross.write_rows(OUT / "analysis/prospective_prediction_records.jsonl", records)
    save(OUT / "analysis/prospective_prediction_cells.json", cells)
    save(OUT / "analysis/prospective_prediction_summary.json", result)
    return result


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    behavior = result["behavior"]
    prediction = result["prospective_prediction"]
    functional = result["functional_metrics"]
    gates = result["experimental_gates"]
    record = rf"""

## Phase {PHASE}: Qwen3-4B FP16 全坐标复用规则的全新词汇前瞻检验（{CAMPAIGN}） [{stamp}]

**测试原理、冻结对象与用例。** 本期在加载模型前冻结 Phase2322 的规则：使用 `fresh_lockbox` 32 条样本得到的逐单元全坐标全局均值、八族分别均值和稳定符号掩码，预测此前未参与 Qwen3-4B FP16 主动场分析的 `fresh_confirmation` 32 条新词汇样本。材料覆盖八种语言模式族、中文/英文、两种自然表面、两个状态；行为候选与自由生成独立记录。模型为本地非量化 Qwen3-4B FP16，源检查点、四个固定 Rademacher 基方向、两个成对方向、目标检查点和 1% 剂量与 Phase2320 完全相同。保存并分析全部 2560 个物理坐标，不用 Top-K、PCA、投影或坐标重排。一个测试例是：用旧词汇中某随机输入方向在给定源层产生的平均完整响应，前瞻预测新词汇同方向、同源层、同目标层的 2560 维响应，而不是搬运某个语义差分。

$$
E_{{g}}(x)=\frac{{\lVert D_x-\bar D_{{\mathrm{{lockbox}}}}\rVert_2^2}}{{\lVert D_x\rVert_2^2+\varepsilon}},\qquad
E_{{f}}(x)=\frac{{\lVert D_x-\bar D_{{\mathrm{{lockbox}},f(x)}}\rVert_2^2}}{{\lVert D_x\rVert_2^2+\varepsilon}}.
$$

$$
A_{{sign}}(x)=\frac{{1}}{{|S|}}\sum_{{j\in S}}\mathbf 1[D_{{x,j}}\bar D_j>0],\qquad
E_{{pair}}=\frac{{\lVert D(r_a+r_b)-D(r_a)-D(r_b)\rVert_2^2}}{{\lVert D(r_a+r_b)\rVert_2^2+\varepsilon}}.
$$

**结果汇总与冻结门。** 32 条行为结果为 `{json.dumps(behavior, ensure_ascii=False)}`。前瞻全坐标汇总为 `{json.dumps({key: value for key, value in prediction.items() if key != 'by_cell'}, ensure_ascii=False)}`。成对叠加误差中位数为 `{functional['median_pair_superposition_relative_mse']}`，偶响应/奇响应 L2 比中位数为 `{functional['median_even_to_odd_l2']}`。冻结门及裁决为 `{json.dumps(gates, ensure_ascii=False)}`；门槛依次为全局预测相对 MSE 不高于 `0.35`、八族均要求全局优于族均值、稳定掩码符号一致率不低于 `0.80`、成对误差不高于 `0.05`、偶/奇比不高于 `0.30`。结果与脚本分别位于 `tests/glm5/result/phase2323_c5601_c5680_qwen4b_fp16_fresh_prospective` 和 `tests/glm5/phase2323_c5601_c5680_qwen4b_fp16_fresh_prospective.py`。

**分析、理论进展、问题硬伤与结论。** 本期是冻结后的前瞻复验，不因结果改阈值。若全局均值优于族均值，只能说明这些随机方向下存在跨族共享的局部传播响应；不能推出语言族没有内部差异。若稳定符号门失败，则 Phase2322 的“稳定坐标”只是旧样本或测量精度条件下的描述。成对叠加和低偶响应只检验 1% 邻域的数值近线性，不等于语义组合规律。材料仍只有 32 条前瞻样本、Qwen3-4B 单模型、受控双语自然句且没有独立人类盲评；随机方向也不是模型训练中天然存在的语言操作。理论主体仍为“条件化输出场闭合理论”，本期只检验“基态条件响应含共享传播骨架”这一窄拼图，不宣称语义齿轮或唯一因果电路。完整自由生成与候选行为继续分账。下一步仅在重要、可复核的全坐标结果上更新图谱，并清理已经发布或不再需要的重复原始场。
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
    for parent in (P2321, P2322):
        value = json.loads((parent / "analysis/final.json").read_text(encoding="utf-8"))
        if not value["all_checks_passed"]:
            raise RuntimeError(("parent_not_authorized", parent))
    config = frozen_config()
    save(OUT / "config/frozen_prospective_contract.json", config)
    raw_rows = cross.behavior_rows(cross.read_rows(ROWS_PATH))
    selected_raw = [row for row in raw_rows if row["partition"] == PARTITION]
    selected_raw.sort(key=lambda row: row["design_index"])
    if len(selected_raw) != 32:
        raise RuntimeError(("prospective_rows", len(selected_raw)))
    model = tokenizer = None
    try:
        model, tokenizer, device = model_base.load_model(
            "qwen3", dtype=torch.float16, use_8bit=False,
        )
        dtypes = model_base.parameter_dtype_counts(model)
        compiled = cross.compile_rows(tokenizer, selected_raw)
        cross.write_rows(OUT / "material/compiled_fresh_confirmation.jsonl", compiled)
        scores = cross.sequence_scores(model, device, compiled, 32)
        cross.write_rows(OUT / "behavior/sequence_scores.jsonl", scores)
        free = cross.free_generation(model, tokenizer, device, compiled, 32)
        cross.write_rows(OUT / "behavior/free_generation.jsonl", free)
        behavior = cross.behavior_summary(scores, free)
        behavior["claim_boundary"] = "32-row fresh_confirmation descriptive behavior panel"
        save(OUT / "behavior/summary.json", behavior)
        field = cross.active_capture(model, device, compiled, OUT)
        index_rows = cross.read_rows(OUT / "index/active_rows.jsonl")
        for row in index_rows:
            row["partition"] = PARTITION
        cross.write_rows(OUT / "index/active_rows.jsonl", index_rows)
        functional = cross.functional_metrics(
            ROOT / field["derivative"], ROOT / field["even"], ROOT / field["norms"],
            index_rows, "Qwen3-4B-FP16-fresh-confirmation",
        )
        save(OUT / "analysis/functional_metrics.json", functional)
        prediction = evaluate_frozen_predictions(ROOT / field["derivative"], index_rows)
        family_gate = all(
            value["global_better_than_family"] for value in prediction["by_family"].values()
        )
        experimental_gates = {
            "global_mean_relative_mse": {
                "value": prediction["global_relative_mse_median"],
                "threshold_max": GATES["global_mean_relative_mse_max"],
                "passed": prediction["global_relative_mse_median"] <= GATES["global_mean_relative_mse_max"],
            },
            "global_better_than_family_all_families": {
                "value": family_gate,
                "required": True,
                "passed": family_gate,
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
        route_passed = all(value["passed"] for value in experimental_gates.values())
        checks = {
            "parents_authorized": True,
            "config_frozen_before_model_load": (OUT / "config/frozen_prospective_contract.json").exists(),
            "fp16_nonquantized": dtypes.get("float16", 0) > 0,
            "prospective_partition_only": all(row["partition"] == PARTITION for row in compiled),
            "all_behavior_rows": behavior["rows"] == 32,
            "all_active_rows": field["shape"][0] == 32,
            "all_coordinates": field["shape"][-1] == 2560,
            "all_54_cells": prediction["cells"] == 54,
            "all_1728_predictions": prediction["records"] == 1728,
            "no_coordinate_selection": True,
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
            "prospective_prediction": prediction,
            "functional_metrics": {
                "prediction": functional["prediction"],
                "families": functional["families"],
                "median_pair_superposition_relative_mse": functional["median_pair_superposition_relative_mse"],
                "median_even_to_odd_l2": functional["median_even_to_odd_l2"],
                "relative_response_topology": functional["relative_response_topology"],
                "claim_boundary": functional["claim_boundary"],
            },
            "experimental_gates": experimental_gates,
            "route_passed": route_passed,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "strict_conclusion": (
                "The frozen response rule passed all prospective gates."
                if route_passed else
                "The frozen response rule failed one or more prospective gates; execution remains valid."
            ),
            "next_authorization": (
                "Publish exact-coordinate fresh_confirmation fields, then clean duplicate raw arrays."
            ),
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
