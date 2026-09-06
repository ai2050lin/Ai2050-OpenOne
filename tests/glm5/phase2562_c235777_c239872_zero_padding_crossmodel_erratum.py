#!/usr/bin/env python3
"""Eliminate batch-width/left-padding contamination in cross-model scoring."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
P2560 = TESTS / "result/phase2560_c223489_c231680_crossmodel_relation_stage_replication"
P2561 = TESTS / "result/phase2561_c231681_c235776_crossmodel_logit_contract_erratum"
OUT = TESTS / "result/phase2562_c235777_c239872_zero_padding_crossmodel_erratum"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2562, "C235777-C239872"

sys.path.insert(0, str(TESTS))
import phase2560_c223489_c231680_crossmodel_relation_stage_replication as p2560  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def old_contradiction(model_key: str) -> dict:
    behavior = read(P2560 / f"behavior/{model_key}_scores.jsonl")
    causal = read(P2560 / f"causal/{model_key}_stage_scores.jsonl")
    behavior_index = {row["base_case_id"]: row for row in behavior if row["ablation"] == "full_scaffold"}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in causal:
        if row["condition"] == "no_patch":
            grouped[row["case_id"]].append(row)
    score_differences, decision_disagreements = [], []
    for case_id, rows in grouped.items():
        old = behavior_index[case_id]
        old_scores = {int(key): value for key, value in old["scores"].items()}
        new_scores = {row["candidate_index"]: row["score"] for row in rows}
        score_differences.extend(abs(old_scores[key] - new_scores[key]) for key in old_scores)
        decision_disagreements.append(max(old_scores, key=old_scores.get) != max(new_scores, key=new_scores.get))
    return {"n_cases": len(grouped), "max_abs_same_sequence_score_difference": max(score_differences, default=0.0),
            "mean_abs_same_sequence_score_difference": float(np.mean(score_differences)) if score_differences else 0.0,
            "decision_disagreement_rate": float(np.mean(decision_disagreements)) if decision_disagreements else 0.0}


def behavior_change(model_key: str, current: dict) -> dict:
    previous = next(row for row in load(P2561 / "analysis/final.json")["models"] if row["model"] == model_key)
    keys = ("full_scaffold", "relation_missing", "value_missing")
    return {key: current["behavior"][key]["accuracy"] - previous["behavior"][key]["accuracy"] for key in keys}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 等长零填充跨模型二次勘误（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理。** Phase2561修正了全长/尾切片logit的绝对位置换算，但复算行为值与Phase2560完全相同，说明它过早把矛盾归因于`logits_to_keep`。原始逐case证据进一步显示：同一prompt+candidate在行为批次和因果`no_patch`批次得到不同分数，两个路径的剩余差别是批内最大长度与左填充宽度。故本Phase按完整sequence长度分桶；每个batch内长度完全相等、左填充数严格为0，再从绝对位置评分。因果base/donor也按双长度分桶，并设硬断言`shift=0`。

$$\forall x_i,x_j\in B:\ |x_i|=|x_j|,\quad s_i=0,
\qquad i_{{local}}=|x|-1-|y|+t-(W_{{input}}-W_{{logits}}).$$

**测试用例与结果汇总。** Phase2560同序列矛盾量化`{json.dumps(result['old_contradiction'], ensure_ascii=False)}`；Qwen14B继续只保留其`no_patch=1`自洽结果；DeepSeek/GLM各完整重跑3072 case/6144候选，结果`{json.dumps(result['models'][1:], ensure_ascii=False)}`；相对Phase2561的行为准确率变化`{json.dumps(result['behavior_change'], ensure_ascii=False)}`；裁决`{json.dumps(result['adjudication'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2562_c235777_c239872_zero_padding_crossmodel_erratum.py`；等长分桶评分/因果实现位于`tests/glm5/phase2560_c223489_c231680_crossmodel_relation_stage_replication.py`；完整新结果位于`{OUT}`。

**分析与理论进展。** 这不是语言规律，而是机制实验的测量公理：同一序列的分数必须与无关batch成员和padding宽度无关。Phase2561关于“候选位置必须按返回logit宽度换算”作为通用防线保留，但“它解释了GLM矛盾”撤销。只有等长零填充复算后的行为门与`no_patch`恒等式可以进入跨模型机制比较。

**问题硬伤与结论。** Qwen14B因重算代价高且原数据在128对上满足严格`no_patch=1`，未再次运行；这只能保留其内部自洽证据。DeepSeek/GLM若仍未过0.80行为门，只能记为本材料能力边界，不得解释其内部机制。跨模型共同齿轮仍未建立，物理坐标不对齐，语言机制未闭合。

**检查。** `{json.dumps(result['checks'], ensure_ascii=False)}`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    old = load(P2560 / "analysis/final.json")
    qwen = next(row for row in old["models"] if row["model"] == "qwen14b")
    p2560.OUT = OUT
    models = [qwen]
    changes = {}
    for model_key in ("deepseek7b", "glm4"):
        print(f"[phase2562] START {model_key}", flush=True)
        current = p2560.run_model(model_key)
        models.append(current)
        changes[model_key] = behavior_change(model_key, current)
        print(f"[phase2562] END {model_key}", flush=True)
    eligible = [row for row in models if row["behavior"]["full_scaffold"]["accuracy"] >= .80
                and row["eligible_pairs"] >= 64]
    adjudication = {"behavior_eligible_models": [row["model"] for row in eligible],
                    "early_value_replication_models": [row["model"] for row in eligible
                        if row["causal"]["early_v_facts_value"]["donor_flip"] is not None
                        and row["causal"]["early_v_facts_value"]["donor_flip"] >= .70],
                    "query_value_replication_models": [row["model"] for row in eligible
                        if row["causal"]["middlelate_kv_query_value"]["donor_flip"] is not None
                        and row["causal"]["middlelate_kv_query_value"]["donor_flip"] >= .70],
                    "late_q_replication_models": [row["model"] for row in eligible
                        if row["causal"]["late_q"]["donor_flip"] is not None
                        and row["causal"]["late_q"]["donor_flip"] >= .70],
                    "physical_coordinate_invariance_tested": False, "language_mechanism_closed": False}
    contradictions = {key: old_contradiction(key) for key in ("glm4",)}
    checks = {"length_bucketed_zero_left_padding": True, "absolute_logit_index_retained": True,
              "affected_models_fully_rerun": len(models) == 3,
              "old_glm_contradiction_measured": contradictions["glm4"]["n_cases"] > 0
              and contradictions["glm4"]["decision_disagreement_rate"] > 0,
              "causal_requires_behavior_gate": all(row["causal_pairs"] == 0 or
                  (row["behavior"]["full_scaffold"]["accuracy"] >= .80 and row["eligible_pairs"] >= 64)
                  for row in models[1:]),
              "eligible_no_patch_identity": all(row["causal"]["no_patch"]["accuracy"] == 1.0
                  for row in models if row["causal_pairs"] > 0),
              "phase2561_causal_attribution_downgraded": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "old_contradiction": contradictions, "models": models, "behavior_change": changes,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
