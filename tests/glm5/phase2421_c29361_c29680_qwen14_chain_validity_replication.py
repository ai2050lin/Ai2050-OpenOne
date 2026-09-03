#!/usr/bin/env python3
"""Replicate the Phase2420 chain-validity interaction on the stronger Qwen14B model."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2420 = RESULT / "phase2420_c29041_c29360_chain_validity_interaction"
OUT = RESULT / "phase2421_c29361_c29680_qwen14_chain_validity_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2421
CAMPAIGN = "C29361-C29680"

sys.path.insert(0, str(TESTS))
import phase2397_c21681_c22000_operation_behavior_token_calibration as behavior  # noqa: E402
import phase2412_c26481_c26800_frozen_crossmodel_operator_replication as loader  # noqa: E402
import phase2416_c27761_c28080_crossmodel_exact_pair_replication as capture_utils  # noqa: E402
import phase2420_c29041_c29360_chain_validity_interaction as interaction  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen14B事实链有效性二阶交互复现（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 完全冻结Phase2420的256配置/1536条有效链、断链A、断链B材料与unit0–3拟合、unit4–7评价合同，在能力更强的Qwen3-14B上独立token校准、目标—foil序列评分，并在answer boundary采集40层输入状态、Attention、MLP全部5120坐标。分析仍为$D=X_2-X_1$后比较$I_{{sem}}=D_{{valid}}-D_{{brokenA}}$与$I_{{lex}}=D_{{brokenA}}-D_{{brokenB}}$，每层使用同族×语言×表面×方向16次样本置乱。模型以`device_map=auto`加载，精度严格标为NF4权重存储/BF16计算；不是BF16权重。

$$\Delta^m_{{validity}}=\left[G_m(I_{{sem}})-Q_{{.95}}G_m(I_{{sem,\pi}})\right]-
\left[G_m(I_{{lex}})-Q_{{.95}}G_m(I_{{lex,\pi}})\right].$$

**结果汇总。** Qwen14B行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；Qwen14B交互 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；Qwen14B特异性 `{json.dumps(result['analysis']['semantic_specificity'], ensure_ascii=False)}`；与Qwen4B比较 `{json.dumps(result['comparison'], ensure_ascii=False)}`；行为桥 `{json.dumps(result['analysis']['behavior_bridge'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2421_c29361_c29680_qwen14_chain_validity_replication.py`；Qwen14B token索引、行为输出、逐interaction×组件×40层×全部坐标派生指标、pair桥及final位于`tests/glm5/result/phase2421_c29361_c29680_qwen14_chain_validity_replication`。未修改其他Markdown。

**分析与理论进展。** 本Phase只回答小模型阴性是否因行为能力不足。若Qwen14B在有效链上的行为优势显著增加，同时$I_{{sem}}$的matched-over-shuffle系统超过$I_{{lex}}$且与行为差相关，才支持把二阶能量差提升为事实链有效性编码候选；若仍不成立，就应停止把“当前状态能预测当前更新”当成组合语义主证据。

**问题硬伤与结论。** Qwen14B使用量化权重，不能与Qwen4B作幅度等价比较，只比较各模型内部归一化增益和方向。教师强制偏好不是自主生成；四族、人工短事实链不覆盖开放语言长程组合。二阶差分与16次置乱的限制延续Phase2420。原始float16场在全坐标派生与核验后删除且不可恢复。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source = read_rows(P2420 / "material/chain_validity_interaction.jsonl")
    model, tokenizer, label = loader.load_for_capture("qwen14b")
    behavior.OUT = OUT; capture_utils.OUT = OUT
    try:
        index = OUT / "qwen14b/index/composition_rows.jsonl"
        if index.exists():
            rows = read_rows(index)
            calibration = json.loads((OUT / "qwen14b/analysis/token_calibration.json").read_text(encoding="utf-8"))
        else:
            rows, calibration = behavior.compile_rows(tokenizer, source)
            write_rows(index, rows); save(OUT / "qwen14b/analysis/token_calibration.json", calibration)
        teacher, teacher_all = behavior.score_rows("qwen14b", model, rows, 2)
        collection = capture_utils.collect("qwen14b", model, rows, 1)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    behavior_result, behavior_arrays = interaction.behavior_summary(teacher, rows)
    interaction.OUT = OUT / "qwen14b"
    analysis = interaction.analyze(rows, collection, behavior_arrays)
    raw_cleanup = interaction.cleanup(collection)
    q4 = json.loads((P2420 / "analysis/final.json").read_text(encoding="utf-8"))
    comparison = {component: {
        "qwen4b_state_over_shuffle_margin": q4["analysis"]["semantic_specificity"][component]["state_over_shuffle_margin"],
        "qwen14b_state_over_shuffle_margin": analysis["semantic_specificity"][component]["state_over_shuffle_margin"],
        "qwen4b_energy_ratio": q4["analysis"]["semantic_specificity"][component]["energy_ratio"],
        "qwen14b_energy_ratio": analysis["semantic_specificity"][component]["energy_ratio"]}
        for component in interaction.COMPONENTS}
    specificity = analysis["semantic_specificity"]
    adjudication = {"qwen14_semantic_interaction_exceeds_lexical_all_components":
                    all(value["state_over_shuffle_margin"] > 0 for value in specificity.values()),
                    "qwen14_semantic_energy_exceeds_lexical_all_components":
                    all(value["energy_ratio"] > 1 for value in specificity.values()),
                    "qwen14_behavior_valid_exceeds_broken": behavior_result["valid_minus_broken_behavior"] > 0,
                    "qwen14_behavior_bridge_positive": analysis["behavior_bridge"]["improvement_behavior_delta_correlation"] > 0,
                    "cross_model_fact_chain_validity_operator_proven": False,
                    "recursive_composition_mechanism_proven": False}
    checks = {"frozen_1536_rows": len(rows) == 1536, "token_calibration": calibration["rows"] == 1536,
              "teacher_complete": len(teacher) == 1536,
              "full_5120_coordinates": collection["state"]["shape"] == [1536, 40, 5120],
              "nf4_bf16_labeled": True,
              "finite": all(math.isfinite(value) for item in comparison.values() for value in item.values()),
              "raw_cleaned": raw_cleanup["removed_files"] == 3, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": label,
              "precision": "NF4 storage / BF16 compute / device_map=auto", "token_calibration": calibration,
              "teacher_all": teacher_all, "behavior": behavior_result, "collection": collection,
              "analysis": analysis, "comparison": comparison, "cleanup": raw_cleanup,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "precision": result["precision"], "behavior": behavior_result,
                      "summary": analysis["summary"], "specificity": specificity,
                      "comparison": comparison, "bridge": analysis["behavior_bridge"],
                      "adjudication": adjudication, "cleanup": raw_cleanup, "checks": checks},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
