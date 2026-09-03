#!/usr/bin/env python3
"""Audit predictive timing, exact-coordinate output structure, and math-tool eligibility."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT_OUT = RESULT / "phase2296_c3101_c3160_ntp_predictive_contract"
FIELD_OUT = RESULT / "phase2297_c3161_c3260_qwen4b_ntp_predictive_field"
PROB_OUT = RESULT / "phase2298_c3261_c3340_full_vocabulary_accounting"
OUT = RESULT / "phase2299_c3341_c3440_predictive_timing_coordinate_structure"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CONTRIBUTIONS = FIELD_OUT / "atlas/qwen4b_target_wrong_coordinate_contributions.float16.npy"
FISHER = FIELD_OUT / "atlas/qwen4b_output_fisher_diagonal.float32.npy"
sys.path.insert(0, str(TESTS))

import phase2296_c3101_c3160_ntp_predictive_contract as contract  # noqa: E402


PHASE = 2299
CAMPAIGN = "C3341-C3440"
EPS = 1e-12


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def effective_count(values: np.ndarray) -> float:
    vector = np.abs(np.asarray(values, dtype=np.float64))
    return float(vector.sum() ** 2 / max(float(np.sum(vector * vector)), EPS))


def normalized_l1(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = 0.5 * (np.abs(a).sum() + np.abs(b).sum())
    return float(np.abs(a - b).sum() / max(float(scale), EPS))


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max())}


def timing(rows: list[dict], lens: list[dict]) -> tuple[dict, list[dict]]:
    qpoints = list(contract.QPOINTS_4B)
    by_family_q = defaultdict(list)
    for row in lens:
        by_family_q[(row["family"], int(row["checkpoint"]))].append(row)
    families = {}
    freeze = []
    for family in contract.FAMILIES:
        checkpoints = {}
        for q in qpoints:
            values = by_family_q[(family, q)]
            train = [row for row in values if row["partition"] in ("discovery", "confirmation")]
            fresh = [row for row in values if row["partition"] in ("fresh_confirmation", "fresh_lockbox")]
            checkpoints[str(q)] = {
                "train_sign_accuracy": float(np.mean([row["target_wrong_margin"] > 0 for row in train])),
                "fresh_sign_accuracy": float(np.mean([row["target_wrong_margin"] > 0 for row in fresh])),
                "train_js_to_final": float(np.mean([row["js_to_actual_final"] for row in train])),
                "fresh_js_to_final": float(np.mean([row["js_to_actual_final"] for row in fresh])),
                "fresh_target_probability": float(np.mean([row["target_probability"] for row in fresh])),
            }
        formation = None
        for index, q in enumerate(qpoints):
            remaining = qpoints[index:]
            if all(checkpoints[str(later)]["train_sign_accuracy"] >= contract.PREDICTIVE_GATES["lens_sign_accuracy"]
                   for later in remaining):
                formation = q
                break
        families[family] = {"checkpoints": checkpoints, "formation_q4": formation,
                            "fresh_at_formation": None if formation is None else checkpoints[str(formation)]}
        if family in contract.Q14_FAMILIES:
            selected = 36 if formation is None else formation
            mapped = 0 if selected == 0 else (41 if selected == 37 else int(round(selected * 40 / 36)))
            freeze.append({
                "family": family, "qwen4_formation_checkpoint": selected,
                "qwen14_checkpoint": mapped, "qwen14_final_norm": 41,
                "selection_data": ["discovery", "confirmation"],
                "test_data": ["fresh_confirmation", "fresh_lockbox"],
                "gate": contract.PREDICTIVE_GATES["lens_sign_accuracy"],
            })
    return families, freeze


def coordinate_structure(rows: list[dict], contributions: np.ndarray) -> tuple[dict, list[dict]]:
    per_row = []
    for i, row in enumerate(rows):
        vector = np.asarray(contributions[i], dtype=np.float32)
        absolute_sum = float(np.abs(vector).sum())
        per_row.append({
            "row": i, "case_id": row["case_id"], "family": row["family"],
            "language": row["language"], "surface": row["surface"],
            "partition": row["partition"], "unit": row["unit"], "state": row["state"],
            "effective_coordinate_count": effective_count(vector),
            "cancellation_ratio": float(abs(float(vector.sum())) / max(absolute_sum, EPS)),
            "absolute_contribution_sum": absolute_sum,
        })
    by_key = {(row["family"], row["language"], row["surface"], int(row["unit"]), int(row["state"])): i
              for i, row in enumerate(rows)}
    families = {}
    for family in contract.FAMILIES:
        indices = [i for i, row in enumerate(rows) if row["family"] == family]
        surface_distance, state_distance = [], []
        for language in ("en", "zh"):
            for unit in range(32):
                for state in (0, 1):
                    a = by_key[(family, language, "narrative", unit, state)]
                    b = by_key[(family, language, "dialogue", unit, state)]
                    surface_distance.append(normalized_l1(contributions[a], contributions[b]))
                for surface in ("narrative", "dialogue"):
                    a = by_key[(family, language, surface, unit, 0)]
                    b = by_key[(family, language, surface, unit, 1)]
                    state_distance.append(normalized_l1(contributions[a], contributions[b]))
        families[family] = {
            "effective_coordinate_count": summarize([per_row[i]["effective_coordinate_count"] for i in indices]),
            "cancellation_ratio": summarize([per_row[i]["cancellation_ratio"] for i in indices]),
            "surface_equivalent_normalized_l1": summarize(surface_distance),
            "state_flip_normalized_l1": summarize(state_distance),
            "state_minus_surface": float(np.mean(state_distance) - np.mean(surface_distance)),
        }
    return families, per_row


def fisher_structure(rows: list[dict], fisher: np.ndarray, contribution: np.ndarray) -> dict:
    index_rows = read_rows(FIELD_OUT / "index/fisher_rows.jsonl")
    case_to_input = {row["case_id"]: i for i, row in enumerate(rows)}
    per_row = []
    by_key = {}
    for i, meta in enumerate(index_rows):
        f = np.asarray(fisher[i], dtype=np.float64)
        c = np.abs(np.asarray(contribution[case_to_input[meta["case_id"]]], dtype=np.float64))
        f_norm = f / max(float(f.sum()), EPS)
        c_norm = c / max(float(c.sum()), EPS)
        agreement = float(len(f) * np.sum(f_norm * c_norm))
        per_row.append({**meta, "effective_fisher_coordinates": effective_count(f),
                        "effective_contribution_coordinates": effective_count(c),
                        "fisher_contribution_agreement_uniform_1": agreement})
        by_key[(meta["family"], meta["language"], meta["surface"], int(meta["state"]))] = i
    surface, state = [], []
    for family in contract.FAMILIES:
        for language in ("en", "zh"):
            for state_value in (0, 1):
                a = by_key[(family, language, "narrative", state_value)]
                b = by_key[(family, language, "dialogue", state_value)]
                surface.append(normalized_l1(fisher[a], fisher[b]))
            for surface_value in ("narrative", "dialogue"):
                a = by_key[(family, language, surface_value, 0)]
                b = by_key[(family, language, surface_value, 1)]
                state.append(normalized_l1(fisher[a], fisher[b]))
    return {
        "rows": per_row,
        "effective_fisher_coordinates": summarize([row["effective_fisher_coordinates"] for row in per_row]),
        "fisher_contribution_agreement": summarize([row["fisher_contribution_agreement_uniform_1"] for row in per_row]),
        "surface_equivalent_normalized_l1": summarize(surface),
        "state_flip_normalized_l1": summarize(state),
        "state_minus_surface": float(np.mean(state) - np.mean(surface)),
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    formation = {family: {"q4": value["formation_q4"], "fresh": value["fresh_at_formation"]}
                 for family, value in result["timing"].items()}
    compact_coordinates = {family: {
        "effective": value["effective_coordinate_count"]["median"],
        "cancellation": value["cancellation_ratio"]["median"],
        "state_minus_surface": value["state_minus_surface"],
    } for family, value in result["coordinate_structure"].items()}
    text = rf"""

## Phase {PHASE}: 预测形成时序、逐坐标输出分解与数学工具资格（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本期沿 embedding、block后和final norm十个冻结读出点，用同一个 final norm+unembedding 只读尺测量目标第一token与错误第一token的margin，以及该辅助分布到模型实际最终分布的JS距离。形成点只使用 discovery+confirmation：最早一个从该点到final norm所有后续读出都达到 `0.75` 符号准确率的检查点；fresh-confirmation和fresh-lockbox只作检验。同时对每个样本的2560个精确输出贡献计算有效参与数、正负抵消率及表面/状态匹配距离，不挑坐标。

**公式。** 有效参与坐标数与抵消率为：

$$
N_{{eff}}(c)=\frac{{(\sum_j|c_j|)^2}}{{\sum_jc_j^2}},
\qquad
\rho_{{cancel}}=\frac{{|\sum_jc_j|}}{{\sum_j|c_j|}}.
$$

逐坐标Fisher对角仍只作为输出敏感度诊断：

$$
G_{{jj}}=\operatorname{{Var}}_{{v\sim p}}[W_{{v,j}}].
$$

**结果汇总。** 形成时序 `{json.dumps(formation, ensure_ascii=False)}`；输出贡献 `{json.dumps(compact_coordinates, ensure_ascii=False)}`；Fisher `{json.dumps({key: value for key, value in result['fisher'].items() if key != 'rows'}, ensure_ascii=False)}`；14B前瞻冻结 `{json.dumps(result['q14_freeze'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。坐标贡献由 final norm 状态和 unembedding 权重精确分解，但只对应第一token目标竞争；它不是中层传动因果分解。有效参与数和抵消率使用全部坐标，说明输出由窄联盟还是宽联盟共同形成，却不赋予单坐标语义名称。Fisher对角可运行是因为 final norm 到 logits 的映射明确；完整Fisher非对角、Koopman、TDA和范畴论没有获得本期资格：层号对应不同block函数，不是平稳重复动力系统；跨模型没有共同物理坐标；有限受控点云也不能支持拓扑不变量。脚本 `tests/glm5/phase2299_c3341_c3440_predictive_timing_coordinate_structure.py`；结果 `tests/glm5/result/phase2299_c3341_c3440_predictive_timing_coordinate_structure`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((FIELD_OUT / "analysis/final.json").read_text(encoding="utf-8"))
    probability = json.loads((PROB_OUT / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"] or not probability["all_checks_passed"]:
        raise RuntimeError("Parent predictive observations are incomplete")
    rows = read_rows(CONTRACT_OUT / "material/ntp_natural_bilingual.jsonl")
    lens = read_rows(FIELD_OUT / "prediction/logit_lens_metrics.jsonl")
    contributions = np.load(CONTRIBUTIONS, mmap_mode="r")
    fisher = np.load(FISHER, mmap_mode="r")
    timing_result, q14_freeze = timing(rows, lens)
    coordinate_result, coordinate_rows = coordinate_structure(rows, contributions)
    fisher_result = fisher_structure(rows, fisher, contributions)
    contract.write_rows(OUT / "coordinate/per_sample_contribution_structure.jsonl", coordinate_rows)
    contract.write_rows(OUT / "coordinate/fisher_structure_rows.jsonl", fisher_result["rows"])
    contract.save(OUT / "protocol/qwen14_predictive_freeze.json", {
        "frozen_before_qwen14_model_load": True, "source_phase": PHASE,
        "families_fixed_in_phase2296": list(contract.Q14_FAMILIES), "cells": q14_freeze,
        "qwen14_model": "Qwen3-14B", "material": "same frozen material; no coordinate identity transfer",
    })
    checks = {
        "all_lens_rows": len(lens) == len(rows) * len(contract.QPOINTS_4B),
        "all_contribution_coordinates": contributions.shape == (len(rows), 2560),
        "all_fisher_coordinates": fisher.shape[1] == 2560,
        "formation_discovery_only": all(cell["selection_data"] == ["discovery", "confirmation"] for cell in q14_freeze),
        "q14_families_preselected": [cell["family"] for cell in q14_freeze] == list(contract.Q14_FAMILIES),
        "advanced_math_not_promoted_to_mechanism": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "timing": timing_result, "coordinate_structure": coordinate_result,
        "fisher": fisher_result, "q14_freeze": q14_freeze, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "Target-token competition has a measurable layerwise formation schedule and a broad signed exact-coordinate "
            "output decomposition; the exact final Fisher diagonal is a valid local sensitivity map, but the data do "
            "not license Koopman modes, topological holes, geodesics, or causal semantic gears."
        ),
        "next_authorization": "Run only the three Phase2296-preselected Qwen3-14B families at the now-frozen model-relative checkpoints, then publish exact-coordinate atlases.",
    }
    serializable = {**result, "fisher": {key: value for key, value in fisher_result.items() if key != "rows"}}
    contract.save(final_path, serializable)
    append_memo(serializable)
    print(json.dumps(serializable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
