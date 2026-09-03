#!/usr/bin/env python3
"""Frozen GLM4 then DS7B replication on discovery plus untouched operation lockboxes."""
from __future__ import annotations

import gc
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2397_ROOT = RESULT / "phase2397_c21681_c22000_operation_behavior_token_calibration"
OUT = RESULT / "phase2403_c23601_c23920_crossarchitecture_frozen_operation_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2403
CAMPAIGN = "C23601-C23920"
MODEL_ORDER = ("glm4", "deepseek7b")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2402_c23281_c23600_qwen14b_frozen_operation_replication as replication  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run_model(key: str) -> dict:
    model_out = OUT / key
    final = model_out / "analysis/final.json"
    if final.exists(): return json.loads(final.read_text(encoding="utf-8"))
    source_root = P2397_ROOT / key
    rows = read_rows(source_root / "index/operation_rows.jsonl")
    selection_rows = [row for row in rows if row["task"] == "selection" and row["partition"] in ("discovery", "fresh_unit_lockbox")]
    composition_rows = [row for row in rows if row["task"] == "composition" and row["partition"] in ("discovery", "fresh_composition_lockbox")]
    write_rows(model_out / "index/selection_rows.jsonl", [replication.index_row(row) for row in selection_rows])
    write_rows(model_out / "index/composition_rows.jsonl", [replication.index_row(row) for row in composition_rows])
    model, tokenizer, label = capability.load_model(key)
    try:
        original_out, original_p2397 = replication.OUT, replication.P2397
        replication.OUT, replication.P2397 = model_out, source_root
        try:
            collection = {"selection": replication.collect_field(model, selection_rows, "selection", batch_size=4),
                          "composition": replication.collect_field(model, composition_rows, "composition", batch_size=4)}
            selection = replication.analyze_task("selection", model, Path(collection["selection"]["path"]), selection_rows)
            composition = replication.analyze_task("composition", model, Path(collection["composition"]["path"]), composition_rows)
        finally:
            replication.OUT, replication.P2397 = original_out, original_p2397
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    result = {"model": key, "model_label": label, "material": {"selection_rows": len(selection_rows), "composition_rows": len(composition_rows),
              "selection_contract": "discovery + fresh_unit_lockbox", "composition_contract": "one-step discovery + two-step fresh lockbox"},
              "collection": collection, "selection": selection, "composition": composition,
              "all_checks_passed": len(selection_rows) == 1536 and len(composition_rows) == 384 and selection["coordinate_prediction"]["condition"]["coordinates"] > 0 and composition["coordinate_prediction"]["condition"]["coordinates"] > 0}
    save(final, result); return result


def append_memo(result: dict) -> None:
    memo_text = MEMO.read_text(encoding="utf-8")
    if f"## Phase {PHASE}:" in memo_text: return
    if "## Phase 2402:" not in memo_text: return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: GLM4→DS7B冻结条件更新关系的跨架构全坐标复现（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不在新模型选层、选事件或改算子；按GLM4 INT8→DeepSeek-R1-Distill-Qwen-7B INT8顺序单模型驻留CUDA。为控制磁盘但保持大样本，两个模型都采集选择任务全部1024条discovery和512条fresh锁箱、组合任务全部256条一步discovery和128条两步fresh锁箱；每条仍保留embedding、全部block、final norm×全部语义事件×全部物理坐标。复验完整条件均值相对常量的全场收益、错族/坐标置换、族护照跨unit/表达/语言、低RMS坐标收益、答案边界输出margin编译以及行为桥。

$$G_M=1-\frac{{\sum(U_M-\widehat U_{{M,condition}})^2}}{{\sum(U_M-\bar U_{{M,train}})^2}},\qquad M\in\{{GLM4,DS7B\}}.$$

**结果汇总。** GLM4 `{json.dumps(result['models']['glm4'], ensure_ascii=False)}`；DS7B `{json.dumps(result['models']['deepseek7b'], ensure_ascii=False)}`；四模型关系裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2403_c23601_c23920_crossarchitecture_frozen_operation_replication.py`；两模型全坐标锁箱场、护照、逐坐标收益与final位于 `tests/glm5/result/phase2403_c23601_c23920_crossarchitecture_frozen_operation_replication`。

**理论进展。** 跨架构复现不要求物理坐标编号相同，只问同一冻结分析关系能否在各模型内部成立。若条件相对常量的收益普遍为正而跨语言/表面护照仍弱，最稳结论是“模板和语言条件化的局部更新可预测”，不是统一语义齿轮；若只有Qwen成立，则属于架构/规模条件规律。

**问题硬伤与结论。** GLM4与DS7B均为量化模型，DS还是Qwen蒸馏谱系，并非独立训练架构的充分样本。轻量复现没有保留confirmation分区，算子类型完全沿用Qwen阶段冻结；这避免新模型选择偏差，也意味着不能在新模型调参补救失败。行为协议差异仍须与teacher和target-present并列。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    models = {}
    for key in MODEL_ORDER: models[key] = run_model(key)
    adjudication = {key: {"selection_condition_gain": value["selection"]["coordinate_prediction"]["condition"]["gain_vs_constant"],
                          "two_step_condition_gain": value["composition"]["coordinate_prediction"]["condition"]["gain_vs_constant"],
                          "selection_answer_compilation_gain": value["selection"]["answer_boundary_output_compilation"]["gain_vs_constant"],
                          "two_step_answer_compilation_gain": value["composition"]["answer_boundary_output_compilation"]["gain_vs_constant"],
                          "selection_behavior_correlation": value["selection"]["behavior_bridge"]["coordinate_gain_to_final_margin_correlation"],
                          "two_step_behavior_correlation": value["composition"]["behavior_bridge"]["coordinate_gain_to_final_margin_correlation"]} for key, value in models.items()}
    checks = {"sequential_order": list(models) == list(MODEL_ORDER), "all_models": all(value["all_checks_passed"] for value in models.values()),
              "finite": all(math.isfinite(metric) for value in adjudication.values() for metric in value.values())}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "adjudication": adjudication,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
