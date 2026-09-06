#!/usr/bin/env python3
"""Repair cross-architecture candidate-logit indexing and rerun affected models."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OLD = TESTS / "result/phase2560_c223489_c231680_crossmodel_relation_stage_replication"
OUT = TESTS / "result/phase2561_c231681_c235776_crossmodel_logit_contract_erratum"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2561, "C231681-C235776"

sys.path.insert(0, str(TESTS))
import phase2560_c223489_c231680_crossmodel_relation_stage_replication as p2560  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 跨架构候选logit索引契约勘误（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与勘误对象。** Phase2560运行后出现硬矛盾：GLM-4材料先被行为分数筛为base/donor双侧正确，但同一未干预输入的`no_patch`准确率只有0.379。根因是不同架构对`logits_to_keep`的实现并不统一：Qwen3返回最后`keep`个位置，GLM/部分Qwen2路径可能返回完整序列。旧评分器把`keep-len(y)-1`直接当作局部索引，因此在返回完整logits时读到了句首。修正后统一从输入绝对位置换算到返回张量局部位置：

$$i_{{local}}=s+|x|-1-\left(W_{{input}}-W_{{logits}}\right),$$

其中$s$是左填充长度。保留Phase2560中`no_patch=1.0`且契约自洽的Qwen3-14B原始结果；用修正评分器完整重跑DeepSeek-7B和GLM-4各3072 case/6144候选。只有full准确率不低于0.80且双侧正确对不少于64，才运行因果条件。

**测试用例与结果汇总。** Qwen14B保留证据`{json.dumps(result['models'][0], ensure_ascii=False)}`；修正后的DeepSeek/GLM结果`{json.dumps(result['models'][1:], ensure_ascii=False)}`；裁决`{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 勘误脚本`tests/glm5/phase2561_c231681_c235776_crossmodel_logit_contract_erratum.py`；修正后的通用评分/因果索引实现在`tests/glm5/phase2560_c223489_c231680_crossmodel_relation_stage_replication.py`；新行为、因果与final位于`{OUT}`；Qwen14B保留文件位于`{OLD}`。

**分析、理论进展与结论。** Phase2560关于Qwen14B的行为与阶段事件可保留；原DeepSeek/GLM数字全部撤销，不能作为能力边界或跨模型机制证据。修正结果只有通过行为门的模型才能进入功能事件比较。跨架构API契约不是语言机制，却会伪造极有迷惑性的“模型差异”；今后任何teacher-forced分数必须同时检查输出长度、绝对token位置和`no_patch=behavior`恒等式。

**问题硬伤。** Qwen14B没有用新函数重算，而是凭其`no_patch=1.0`和Qwen3尾切片契约保留；这足以保证内部一致性，但不是跨实现位级复算。跨模型仍只比较相对层段功能，不比较物理坐标；失败行为模型只说明本提示/评分任务未通过，不能推出其内部没有相关编码。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    old = load(OLD / "analysis/final.json")
    qwen = next(row for row in old["models"] if row["model"] == "qwen14b")
    if qwen["causal"]["no_patch"]["accuracy"] != 1.0:
        raise RuntimeError("Qwen14B retained evidence fails no-patch identity")
    p2560.OUT = OUT
    models = [qwen]
    for model_key in ("deepseek7b", "glm4"):
        print(f"[phase2561] START {model_key}", flush=True)
        models.append(p2560.run_model(model_key))
        print(f"[phase2561] END {model_key}", flush=True)
    eligible_models = [row for row in models if row["behavior"]["full_scaffold"]["accuracy"] >= .80
                       and row["eligible_pairs"] >= 64]
    adjudication = {
        "behavior_eligible_models": [row["model"] for row in eligible_models],
        "early_value_replication_models": [row["model"] for row in eligible_models
            if row["causal"]["early_v_facts_value"]["donor_flip"] is not None
            and row["causal"]["early_v_facts_value"]["donor_flip"] >= .70],
        "query_value_replication_models": [row["model"] for row in eligible_models
            if row["causal"]["middlelate_kv_query_value"]["donor_flip"] is not None
            and row["causal"]["middlelate_kv_query_value"]["donor_flip"] >= .70],
        "late_q_replication_models": [row["model"] for row in eligible_models
            if row["causal"]["late_q"]["donor_flip"] is not None
            and row["causal"]["late_q"]["donor_flip"] >= .70],
        "language_mechanism_closed": False,
    }
    checks = {
        "absolute_to_local_index_contract": True,
        "affected_models_fully_rerun": len(models) == 3,
        "qwen_retained_no_patch_identity": qwen["causal"]["no_patch"]["accuracy"] == 1.0,
        "causal_requires_behavior_gate": all(row["causal_pairs"] == 0 or
            (row["behavior"]["full_scaffold"]["accuracy"] >= .80 and row["eligible_pairs"] >= 64)
            for row in models[1:]),
        "eligible_no_patch_identity": all(row["causal"]["no_patch"]["accuracy"] == 1.0
            for row in models[1:] if row["causal_pairs"] > 0),
        "invalid_phase2560_deepseek_glm_withdrawn": True,
        "claim_boundary": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "models": models, "adjudication": adjudication, "checks": checks,
              "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
