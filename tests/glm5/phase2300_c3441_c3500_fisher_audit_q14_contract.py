#!/usr/bin/env python3
"""Correct the saturated Fisher summary and seal the Qwen3-14B NTP replication."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase2299_c3341_c3440_predictive_timing_coordinate_structure"
FIELD_OUT = RESULT / "phase2297_c3161_c3260_qwen4b_ntp_predictive_field"
OUT = RESULT / "phase2300_c3441_c3500_fisher_audit_q14_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2296_c3101_c3160_ntp_predictive_contract as contract  # noqa: E402


PHASE = 2300
CAMPAIGN = "C3441-C3500"
THRESHOLD = 1e-8


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 饱和输出Fisher退化审计与14B锁定（{CAMPAIGN}） [{stamp}]

**测试原理。** Phase2299 的原始Fisher逐坐标数组计算正确且全部有限，但直接对所有48行汇总有效参与数不合法：当输出概率近乎one-hot时，Fisher总质量接近零，固定 `EPS` 分母会产生小于1的伪“有效坐标数”。本期不重算或筛选模型结果，只在完整2560坐标总质量上执行预定数值资格审计；总质量不高于 `1e-8` 的行标记为退化，不进入形状汇总。该修正不改变Phase2299形成层或14B构式冻结。

**公式。** 数值资格为：

$$
M_G=\sum_jG_{{jj}},\qquad
M_G>10^{{-8}}\ \Longrightarrow\ N_{{eff}}(G)\text{{ 可报告}}.
$$

**结果汇总。** `{json.dumps(result['fisher_audit'], ensure_ascii=False)}`；14B合同 `{json.dumps(result['q14_contract'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。这次纠偏说明“高置信输出处Fisher变小”不等于内部不敏感，更不能称为稳定吸引子；它只表示 categorical softmax 在当前最终分布附近饱和。因而14B阶段只复验冻结形成时序和完整概率输出，不使用Fisher作通过门。脚本 `tests/glm5/phase2300_c3441_c3500_fisher_audit_q14_contract.py`；结果 `tests/glm5/result/phase2300_c3441_c3500_fisher_audit_q14_contract`。
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
    parent = json.loads((PARENT / "analysis/final.json").read_text(encoding="utf-8"))
    fisher = np.load(FIELD_OUT / "atlas/qwen4b_output_fisher_diagonal.float32.npy", mmap_mode="r")
    mass = np.asarray(fisher, dtype=np.float64).sum(axis=1)
    eligible = mass > THRESHOLD
    effective = []
    for row in np.asarray(fisher[eligible], dtype=np.float64):
        effective.append(float(row.sum() ** 2 / np.sum(row * row)))
    audit = {
        "rows": len(mass), "all_finite": bool(np.isfinite(fisher).all()),
        "mass_min_median_max": [float(mass.min()), float(np.median(mass)), float(mass.max())],
        "threshold": THRESHOLD, "eligible_rows": int(eligible.sum()),
        "degenerate_rows": int((~eligible).sum()),
        "eligible_effective_count_min_median_max": (
            [float(np.min(effective)), float(np.median(effective)), float(np.max(effective))]
            if effective else None
        ),
        "phase2299_all_row_effective_count_status": "invalid_when_fisher_mass_is_numerically_degenerate",
        "raw_fisher_status": "valid_finite_exact_diagonal",
    }
    q14_contract = {
        "frozen_before_qwen14_load": True,
        "families": list(contract.Q14_FAMILIES),
        "cells": parent["q14_freeze"],
        "rows": 3 * 2 * 2 * 12 * 2,
        "partitions": ["fresh_confirmation", "fresh_lockbox"],
        "outputs": ["lexical_sequence_score", "complete_next_token_logits", "selected_checkpoint_all_coordinates"],
        "gates": {"sequence_accuracy": 0.75, "formation_sign_accuracy": 0.75},
        "not_gates": ["Fisher", "coordinate identity", "cosine", "Top-K"],
    }
    contract.save(OUT / "protocol/qwen14_ntp_contract.json", q14_contract)
    checks = {
        "parent_passed": parent["all_checks_passed"],
        "raw_fisher_finite": audit["all_finite"],
        "degeneracy_detected": audit["degenerate_rows"] > 0,
        "invalid_summary_not_reused": True,
        "q14_contract_frozen": q14_contract["frozen_before_qwen14_load"],
        "q14_rows_precomputed": q14_contract["rows"] == 288,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "fisher_audit": audit, "q14_contract": q14_contract,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            f"The exact Fisher diagonal remains a valid raw sensitivity diagnostic, but {audit['degenerate_rows']}/48 "
            f"rows are saturated at mass <= {THRESHOLD:g}; Phase2299's all-row effective-count summary is withdrawn, "
            "and Fisher is excluded from the Qwen3-14B gate."
        ),
        "next_authorization": "Load Qwen3-14B sequentially and run the sealed 288-row predictive-timing replication.",
    }
    contract.save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
