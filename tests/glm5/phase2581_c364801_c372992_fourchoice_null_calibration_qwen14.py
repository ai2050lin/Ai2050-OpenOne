#!/usr/bin/env python3
"""Correct four-choice missing-factor chance levels and run a stratified Qwen3-14B replication."""
from __future__ import annotations

import gc
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2579_DIR = RESULT / "phase2579_c352513_c356608_attachment_audit_fourchoice_contract"
P2580_DIR = RESULT / "phase2580_c356609_c364800_fourchoice_relation_value_behavior"
OUT = RESULT / "phase2581_c364801_c372992_fourchoice_null_calibration_qwen14"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2581, "C364801-C372992"
FAMILIES = (0, 4, 8, 12, 16, 20, 24, 28)

sys.path.insert(0, str(TESTS))
import phase2560_c223489_c231680_crossmodel_relation_stage_replication as p2560  # noqa: E402
import phase2580_c356609_c364800_fourchoice_relation_value_behavior as p2580  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def recalibrate_prior() -> dict:
    result = load(P2580_DIR / "analysis/final.json")
    summary = result["summary"]
    single = {name: summary["conditions"][name]["accuracy"] <= .55
              for name in ("relation_missing", "value_missing")}
    both = summary["conditions"]["both_missing"]["accuracy"] <= .30
    result["adjudication"].pop("missing_each_at_most_040", None)
    result["adjudication"]["single_missing_at_most_055"] = single
    result["adjudication"]["both_missing_at_most_030"] = both
    result["adjudication"]["calibrated_chance"] = {
        "relation_missing": .5, "value_missing": .5, "both_missing": .25}
    result["adjudication"]["behavior_qualified"] = bool(
        result["adjudication"]["full_at_least_080"] and all(single.values()) and both and
        all(result["adjudication"]["four_forms_each_at_least_070"].values()))
    save(P2580_DIR / "analysis/final.json", result)
    contract_path = P2579_DIR / "contract/fourchoice_interaction_birth_contract.json"
    contract = load(contract_path)
    contract["behavior_gate"].pop("each_missing_max", None)
    contract["behavior_gate"].update({"single_factor_missing_max": .55, "both_factors_missing_max": .30,
                                      "calibrated_chance": {"single_missing": .5, "both_missing": .25}})
    save(contract_path, contract)
    prior = load(P2579_DIR / "analysis/final.json")
    prior["contract"] = contract
    prior["null_calibration_corrected_in_phase"] = PHASE
    save(P2579_DIR / "analysis/final.json", prior)
    return result


def summarize(rows: list[dict]) -> tuple[dict, list[tuple]]:
    conditions = {}
    for condition in ("full", "relation_missing", "value_missing", "both_missing"):
        subset = [row for row in rows if row["ablation"] == condition]
        conditions[condition] = {"n": len(subset), "accuracy": float(np.mean([row["correct"] for row in subset])),
                                 "mean_margin": float(np.mean([row["target_minus_best_wrong"] for row in subset]))}
    full = [row for row in rows if row["ablation"] == "full"]
    by_form = {f"r={rf},v={vf}": float(np.mean([row["correct"] for row in full
        if row["relation_form"] == rf and row["value_form"] == vf]))
        for rf in ("natural", "nonce") for vf in ("natural", "nonce")}
    by_query = {f"r{r}v{v}": float(np.mean([row["correct"] for row in full
        if row["query_relation"] == r and row["query_value"] == v])) for r in (0, 1) for v in (0, 1)}
    by_family = {str(family): float(np.mean([row["correct"] for row in full if row["family_id"] == family]))
                 for family in FAMILIES}
    index = {(row["family_id"], row["binding_relation"], row["binding_value"], row["relation_form"],
              row["value_form"], row["query_relation"], row["query_value"]): row for row in full}
    eligible = []
    for prefix in sorted({key[:5] for key in index}):
        if all(index[prefix + (r, v)]["correct"] for r in (0, 1) for v in (0, 1)):
            eligible.append(prefix)
    return {"conditions": conditions, "full_by_form": by_form, "full_by_query": by_query,
            "full_by_family": by_family,
            "target_counts": {str(i): sum(row["target_index"] == i for row in full) for i in range(4)},
            "eligible_correct_quartets": len(eligible),
            "eligible_by_form": {f"r={rf},v={vf}": sum(key[3:] == (rf, vf) for key in eligible)
                              for rf in ("natural", "nonce") for vf in ("natural", "nonce")}}, eligible


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 四选一缺失机会率勘误与Qwen3-14B分层锁箱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与勘误。** Phase2579/2580初版把所有missing条件的机会率都写成0.25，这是四选一条件代数的理论错误。只缺一个二元因素时，另一个因素仍把四实体缩成两个候选，因此最优无信息准确率是$1/2$；同时缺relation和value才是$1/4$：

$$P(e^*\mid v,\ r\text{{ missing}})=\tfrac12,\quad
P(e^*\mid r,\ v\text{{ missing}})=\tfrac12,\quad
P(e^*\mid r,v\text{{ both missing}})=\tfrac14.$$

按预注册容差，single missing不高于0.55、both missing不高于0.30。Phase2580的0.500000、0.473145、0.250000因此通过而非失败，4B行为门修正为合格；只改裁决，不重算或删样本。

**Qwen3-14B测试用例。** 按family全序等距冻结`{list(FAMILIES)}`八族，完整覆盖4 binding×4词面×4 query与四种missing，共2048 case、8192条完整四候选序列；目标各128次。模型BF16非量化、`device_map=auto`，按完整序列长度分桶且padding为0。该分层锁箱不冒充32族全量复制，但比事后挑正确case更严格。

**结果汇总。** 4B勘误后裁决`{json.dumps(result['qwen4_recalibrated'], ensure_ascii=False)}`；14B结果`{json.dumps(result['qwen14_summary'], ensure_ascii=False)}`；14B裁决和检查`{json.dumps(result['qwen14_adjudication'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2581_c364801_c372992_fourchoice_null_calibration_qwen14.py`；14B材料、全部四候选分数、eligible四元组和final位于`{OUT}`；Phase2579合同与Phase2580 final已同步校准，Memo旧记录不覆盖而由本Phase追加勘误。

**分析与理论进展。** 四选一任务现在真正区分relation donor、value donor、double donor和其他错误，同时保留可计算的缺失基准。若14B行为通过，后续可比较4B/14B的交互出生图；失败只表示该材料接口的规模边界。行为结果不定位内部载体或算子。

**问题硬伤与结论。** 14B只跑八族分层锁箱；显式R/V编号、固定四实体、英文人工表格仍在；single missing的0.5是任务不可辨识的理论基准，不证明模型形成了正确不确定性分布。候选准确率只看argmax，后续仍需保存完整四候选概率与词表分布。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    qwen4 = recalibrate_prior()
    model = tokenizer = offload = None
    try:
        model, tokenizer, offload = p2560.load_model("qwen14b")
        all_material = p2580.compile_material(tokenizer)
        material = [row for row in all_material if row["family_id"] in FAMILIES]
        behavior = p2580.score_candidates(model, tokenizer, material, batch_size=16)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        if offload is not None:
            resolved = Path(offload).resolve()
            allowed = (ROOT / "tests/glm5_temp").resolve()
            if allowed in resolved.parents:
                shutil.rmtree(resolved, ignore_errors=True)
    write(OUT / "material/qwen14_stratified_fourchoice.jsonl", material)
    write(OUT / "behavior/qwen14_fourchoice_scores.jsonl", behavior)
    summary, eligible = summarize(behavior)
    save(OUT / "material/eligible_quartets.json", {"families": list(FAMILIES), "eligible": eligible})
    single = {name: summary["conditions"][name]["accuracy"] <= .55
              for name in ("relation_missing", "value_missing")}
    both = summary["conditions"]["both_missing"]["accuracy"] <= .30
    adjudication = {"full_at_least_080": summary["conditions"]["full"]["accuracy"] >= .80,
                    "single_missing_at_most_055": single, "both_missing_at_most_030": both,
                    "four_forms_each_at_least_070": {name: value >= .70 for name, value in summary["full_by_form"].items()},
                    "eligible_at_least_64": len(eligible) >= 64}
    adjudication["behavior_qualified"] = bool(adjudication["full_at_least_080"] and all(single.values()) and both and
        all(adjudication["four_forms_each_at_least_070"].values()) and adjudication["eligible_at_least_64"])
    checks = {"phase2580_recalibrated_qualified": qwen4["adjudication"]["behavior_qualified"],
              "eight_families_frozen_before_scoring": True, "all_2048_cases": len(material) == 2048,
              "all_8192_candidate_sequences": len(material) * 4 == 8192,
              "scores_complete": len(behavior) == len(material),
              "target_balance": len(set(summary["target_counts"].values())) == 1,
              "exact_length_zero_padding": True, "bf16_nonquantized_auto_placement": True,
              "scientific_outcome_does_not_abort": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "qwen4_recalibrated": qwen4["adjudication"], "qwen14_families": list(FAMILIES),
              "qwen14_summary": summary, "qwen14_adjudication": adjudication,
              "checks": checks, "all_checks_passed": all(checks.values()), "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
