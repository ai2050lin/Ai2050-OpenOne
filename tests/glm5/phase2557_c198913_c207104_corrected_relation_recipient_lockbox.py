#!/usr/bin/env python3
"""Complete causal rerun on all post-erratum relation-necessary eligible pairs."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2556 = RESULT / "phase2556_c190721_c198912_form_id_collision_erratum_recompute"
OUT = RESULT / "phase2557_c198913_c207104_corrected_relation_recipient_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2557, "C198913-C207104"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2555_c182529_c190720_relation_stage_recipient_causal_atlas as p2555  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def compile_jobs(tokenizer) -> tuple[list[dict], list[tuple]]:
    material = [row for row in read(P2556 / "material/phase2554_corrected_token_atomic.jsonl")
                if row["ablation"] == "full_scaffold"]
    behavior = [row for row in read(P2556 / "behavior/phase2554_recomputed.jsonl")
                if row["ablation"] == "full_scaffold"]
    correct = {row["base_case_id"]: row["correct"] for row in behavior}
    index = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"],
              row["query_value"], row["binding"]): row for row in material}
    region_names = ("facts_entity", "facts_relation", "facts_value", "facts_all", "query_context",
                    "query_relation", "query_value", "query_all", "candidate", "instruction",
                    "post_query", "answer_boundary", "external")
    jobs, eligible = [], []
    for family_id in range(32):
        for relation_form in ("natural", "nonce"):
            for value_form in ("natural", "nonce"):
                for query_relation in (0, 1):
                    for query_value in (0, 1):
                        key = (family_id, relation_form, value_form, query_relation, query_value)
                        base, donor = index[key + (0,)], index[key + (1,)]
                        if not (correct[base["base_case_id"]] and correct[donor["base_case_id"]]):
                            continue
                        eligible.append(key)
                        for candidate_index, entity in enumerate(base["entities"]):
                            continuation = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
                            jobs.append({"case_id": base["base_case_id"], "family_id": family_id,
                                         "family": base["family"], "relation_form": relation_form,
                                         "value_form": value_form, "query_relation": query_relation,
                                         "query_value": query_value, "candidate_index": candidate_index,
                                         "candidate": entity, "target_index": base["target_index"],
                                         "donor_target_index": donor["target_index"],
                                         "base_prompt_length": len(base["prompt_ids"]),
                                         "donor_prompt_length": len(donor["prompt_ids"]),
                                         "base": base["prompt_ids"] + continuation,
                                         "donor": donor["prompt_ids"] + continuation,
                                         "continuation": continuation,
                                         "regions_base": {name: p2555.positions(base, name) for name in region_names},
                                         "regions_donor": {name: p2555.positions(donor, name) for name in region_names}})
    return jobs, eligible


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 修正后392对关系必要任务recipient锁箱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2556修正case ID后，新实体英文锁箱512个binding对中有392对base/donor双侧正确。本Phase不再使用碰撞资格，而以完整factor tuple重新编译；对全部392对、784个完整多token候选作业重跑23个Q/K/V条件，覆盖32族、自然/nonce关系与值、四种query。每次现场捕获binding1 donor的投影输出，再写入binding0 base的严格token region；所有左填充shift均显式加入物理坐标。

$$
F_{{c,R,B}}=\Pr[\hat e_{{do(P^c_{{B,R}}\leftarrow P^{{c,d}}_{{B,R}})}}=e_d],
\qquad c\in\{{Q,K,V,KV\}}.
$$

**结果汇总。** `{json.dumps(result['summary'], ensure_ascii=False)}`。裁决为`{json.dumps(result['adjudication'], ensure_ascii=False)}`；完整性检查为`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2557_c198913_c207104_corrected_relation_recipient_lockbox.py`；392对×2候选×23条件的逐行logprob和final位于`{OUT}`。

**分析与理论进展。** 该锁箱决定Phase2537–2550的阶段候选在relation与value共同必要时是否仍成立。最关键的新量是中晚层recipient：query-value若单独接近全query/all-external，说明binding结果主要被条件化写回查询值位置；若自然/nonce差异大，则不是跨词面普遍机制。facts-relation的单区低效不表示关系未参与，因为donor只翻转binding而保持relation本身不变；它仅说明“改变答案所需的信息”不由替换relation token投影单独搬运。

**问题硬伤与结论。** 全部干预仍是九层、全head、全projection坐标的强充分性操作；eligible筛选使结果只适用于可答材料；R0/R1与表格提示仍是人工结构；binary donor使翻转与错误重合。阶段事件允许按强度比较，但不能相乘成$K^{{-1}}$式闭合，也不能外推自然开放语言。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2556 / "analysis/final.json")
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        jobs, eligible = compile_jobs(tokenizer)
        rows = p2555.run(model, tokenizer, jobs)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
    scores_path = OUT / "causal/corrected_region_stage_scores.jsonl"
    p2552.write(scores_path, rows)
    summary = p2555.summarize(rows)
    single_recipient_names = ("middlelate_kv_query_context", "middlelate_kv_query_relation",
                              "middlelate_kv_query_value", "middlelate_kv_candidate",
                              "middlelate_kv_instruction", "middlelate_kv_answer_boundary")
    strongest = max(single_recipient_names, key=lambda name: summary[name]["donor_flip"])
    adjudication = {
        "early_value_stage_replicates": summary["early_v_facts_value"]["donor_flip"] >= .70,
        "middle_value_kv_stage_replicates": summary["middle_kv_facts_value"]["donor_flip"] >= .50,
        "middlelate_external_stage_replicates": summary["middlelate_kv_external"]["donor_flip"] >= .70,
        "strongest_single_recipient": strongest,
        "strongest_single_recipient_flip": summary[strongest]["donor_flip"],
        "single_recipient_sufficient_at_070": summary[strongest]["donor_flip"] >= .70,
        "late_q_stage_replicates_at_070": summary["late_q"]["donor_flip"] >= .70,
        "late_fact_kv_absent_at_010": summary["late_kv_facts_all"]["donor_flip"] <= .10,
        "all_four_form_directions_consistent_for_strongest": min(summary[strongest]["donor_flip_by_form"].values()) >= .50,
        "language_mechanism_closed": False,
    }
    checks = {"phase2556_passed": prior["all_checks_passed"], "eligible_392": len(eligible) == 392,
              "candidate_jobs_784": len(jobs) == 784, "conditions_23": len(p2555.CONDITIONS) == 23,
              "all_rows_complete": len(rows) == len(jobs) * len(p2555.CONDITIONS),
              "baseline_gate": summary["no_patch"]["accuracy"] >= .95,
              "all_forms_reported": all(len(panel["donor_flip_by_form"]) == 4 for panel in summary.values()),
              "unique_case_ids": len({job["case_id"] for job in jobs}) == len(eligible), "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "design": {"eligible_pairs": len(eligible),
              "candidate_jobs": len(jobs), "conditions": len(p2555.CONDITIONS)}, "summary": summary,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values()),
              "files": {"scores": str(scores_path)}}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
