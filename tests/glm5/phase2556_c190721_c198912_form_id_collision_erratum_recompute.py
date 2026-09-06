#!/usr/bin/env python3
"""Repair natural/nonce case-ID collisions and recompute every affected behavior score."""
from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2556_c190721_c198912_form_id_collision_erratum_recompute"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
P2555 = RESULT / "phase2555_c182529_c190720_relation_stage_recipient_causal_atlas"
PHASE, CAMPAIGN = 2556, "C190721-C198912"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2553_c174337_c178432_relation_slot_scaffold_adjudication as p2553  # noqa: E402
import phase2554_c178433_c182528_independent_relation_lockbox_behavior as p2554  # noqa: E402
import phase2555_c182529_c190720_relation_stage_recipient_causal_atlas as p2555  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def summarize(rows: list[dict], conditions: tuple[str, ...]) -> dict:
    result = {"conditions": {}}
    for condition in conditions:
        subset = [row for row in rows if row["ablation"] == condition]
        result["conditions"][condition] = {"n": len(subset),
            "accuracy": float(np.mean([row["correct"] for row in subset])),
            "mean_margin": float(np.mean([row["target_minus_wrong"] for row in subset]))}
    first = conditions[0]
    full = [row for row in rows if row["ablation"] == first]
    result["full_by_form"] = {f"r={rf},v={vf}": float(np.mean([row["correct"] for row in full
        if row["relation_form"] == rf and row["value_form"] == vf])) for rf in ("natural", "nonce") for vf in ("natural", "nonce")}
    result["full_by_language"] = {language: float(np.mean([row["correct"] for row in full if row["language"] == language]))
                                  for language in sorted({row["language"] for row in full})}
    result["full_by_query"] = {f"r{r}v{v}": float(np.mean([row["correct"] for row in full
        if row["query_relation"] == r and row["query_value"] == v])) for r in (0, 1) for v in (0, 1)}
    result["full_by_family"] = {str(fid): float(np.mean([row["correct"] for row in full if row["family_id"] == fid]))
                                for fid in range(32)}
    return result


def eligible_pairs(rows: list[dict], condition: str) -> set[tuple]:
    full = [row for row in rows if row["ablation"] == condition]
    index = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"],
              row["query_value"], row["binding"]): row for row in full}
    eligible = set()
    for family_id in range(32):
        for relation_form in ("natural", "nonce"):
            for value_form in ("natural", "nonce"):
                for query_relation in (0, 1):
                    for query_value in (0, 1):
                        key = (family_id, relation_form, value_form, query_relation, query_value)
                        if index[key + (0,)]["correct"] and index[key + (1,)]["correct"]:
                            eligible.add(key)
    return eligible


def reanalyze_causal(rows: list[dict], eligible: set[tuple]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["family_id"], row["relation_form"], row["value_form"], row["query_relation"], row["query_value"])
        if key in eligible:
            grouped[(row["condition"],) + key].append(row)
    panels = {}
    for condition in p2555.CONDITIONS:
        groups = [values for key, values in grouped.items() if key[0] == condition]
        correct, flipped, margin = [], [], []
        by_form = defaultdict(list)
        for values in groups:
            if len(values) != 2:
                raise RuntimeError((condition, values))
            prediction = max(values, key=lambda row: row["score"])["candidate_index"]
            target, donor = values[0]["target_index"], values[0]["donor_target_index"]
            scores = {row["candidate_index"]: row["score"] for row in values}
            correct.append(prediction == target)
            flipped.append(prediction == donor)
            margin.append(scores[donor] - scores[target])
            by_form[(values[0]["relation_form"], values[0]["value_form"])].append(prediction == donor)
        panels[condition] = {"n": len(groups), "accuracy": float(np.mean(correct)) if groups else None,
                             "donor_flip": float(np.mean(flipped)) if groups else None,
                             "mean_donor_margin": float(np.mean(margin)) if groups else None,
                             "donor_flip_by_form": {f"r={key[0]},v={key[1]}": float(np.mean(value))
                                                    for key, value in sorted(by_form.items())}}
    return panels


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: natural/nonce键碰撞勘误与全量行为重算（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与错误定位。** Phase2552/2553最初用`relation_form[0]`和`value_form[0]`生成case ID；`natural`与`nonce`首字母都为`n`，四种词面组合被压到同一键。模型前向实际执行，但候选分数字典只保留每个碰撞键最后写入的组合，随后又把该分数复制给四个form；Phase2555原始因果行保留了完整form字段和逐作业分数，可重新分组，但其行为资格集合来自碰撞结果。该错误使此前“自然/nonce四格完全同分”、Phase2552/2553/2554候选总体和Phase2555初始汇总全部失效。

$$
id_{{old}}=(\ldots,r[0],v[0],\ldots),\quad r[0]=v[0]=\texttt{{n}},
\qquad id_{{new}}=(\ldots,\texttt{{natural|nonce}},\texttt{{natural|nonce}},\ldots).
$$

**重算范围。** 修正脚本源后，在同一个Qwen3-4B BF16非量化CUDA进程中重新运行Phase2552全部32768个行为case、Phase2553全部20480个行为case、Phase2554全部3072个新实体锁箱case，总计56320 case、112640条完整多token候选序列；不是对旧JSON做标签修补。Phase2555的18400条逐候选因果原始行按`condition+family+完整relation_form+完整value_form+query`重新分组，并且只保留重算后真正双侧正确且确实出现在原因果集合中的对。

**结果汇总。** Phase2552修正为`{json.dumps(result['phase2552_corrected'], ensure_ascii=False)}`；Phase2553修正为`{json.dumps(result['phase2553_corrected'], ensure_ascii=False)}`；Phase2554修正为`{json.dumps(result['phase2554_corrected'], ensure_ascii=False)}`。Phase2555可恢复子集为`{json.dumps(result['phase2555_salvage'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2556_c190721_c198912_form_id_collision_erratum_recompute.py`；三批重新前向的逐case分数、修正后的Phase2554 token材料、Phase2555重新分组结果和final位于`{OUT}`。Phase2552与Phase2553源脚本的case ID已改用完整form字符串。

**分析与理论进展。** 这次勘误不保留任何由碰撞制造的“完全相等”规律。只有重算后的full、缺失门、词面四格和eligible集合可以进入后续全坐标与因果研究。Phase2555若恢复子集不覆盖修正eligible全集，则其结果仅作为探索性线索，下一Phase必须按修正资格重跑，不能把重新汇总冒充完整锁箱。

**问题硬伤与结论。** case ID碰撞属于实现错误而不是统计波动，必须明确撤销受影响结论；自主生成逐行未通过字典键合并，数值仍可保留，但其case ID也不唯一，后续使用完整factor tuple。该事件说明“多个form恰好完全同分”应触发结构审计而非理论解释。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)

        material2 = p2552.compile_material(tokenizer, ("full", "relation", "value", "both"))
        rows2 = p2553.score_candidates(model, tokenizer, material2, batch_size=32)
        path2 = OUT / "behavior/phase2552_recomputed.jsonl"
        p2552.write(path2, rows2)
        summary2 = summarize(rows2, ("full", "relation", "value", "both"))
        del material2
        gc.collect()

        material3 = p2553.compile_material(tokenizer)
        rows3 = p2553.score_candidates(model, tokenizer, material3, batch_size=32)
        path3 = OUT / "behavior/phase2553_recomputed.jsonl"
        p2552.write(path3, rows3)
        summary3 = summarize(rows3, p2553.CONDITIONS)
        del material3
        gc.collect()

        material4 = p2554.compile_material(tokenizer)
        rows4 = p2553.score_candidates(model, tokenizer, material4, batch_size=32)
        path4 = OUT / "behavior/phase2554_recomputed.jsonl"
        material4_path = OUT / "material/phase2554_corrected_token_atomic.jsonl"
        p2552.write(path4, rows4)
        p2552.write(material4_path, material4)
        summary4 = summarize(rows4, ("full_scaffold", "relation_missing", "value_missing"))
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()

    eligible4 = eligible_pairs(rows4, "full_scaffold")
    summary4["paired_base_donor_total"] = 512
    summary4["paired_base_donor_eligible"] = len(eligible4)
    causal_raw = read(P2555 / "causal/region_stage_candidate_scores.jsonl")
    salvage = reanalyze_causal(causal_raw, eligible4)
    salvage_path = OUT / "analysis/phase2555_salvage_summary.json"
    save(salvage_path, salvage)
    present_keys = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"], row["query_value"])
                    for row in causal_raw}
    salvaged_eligible = eligible4 & present_keys
    adjudication = {
        "phase2552_full_gate": summary2["conditions"]["full"]["accuracy"] >= .80,
        "phase2552_relation_missing_near_chance": summary2["conditions"]["relation"]["accuracy"] <= .65,
        "phase2552_value_missing_near_chance": summary2["conditions"]["value"]["accuracy"] <= .65,
        "phase2553_full_gate": summary3["conditions"]["full_scaffold"]["accuracy"] >= .80,
        "phase2554_lockbox_gate": summary4["conditions"]["full_scaffold"]["accuracy"] >= .80,
        "phase2554_relation_missing_near_chance": summary4["conditions"]["relation_missing"]["accuracy"] <= .65,
        "phase2554_value_missing_near_chance": summary4["conditions"]["value_missing"]["accuracy"] <= .65,
        "phase2555_requires_complete_rerun": salvaged_eligible != eligible4,
        "language_mechanism_closed": False,
    }
    checks = {"unique_phase2552_ids": len({row["case_id"] for row in rows2}) == len(rows2),
              "unique_phase2553_ids": len({row["case_id"] for row in rows3}) == len(rows3),
              "unique_phase2554_ids": len({row["case_id"] for row in rows4}) == len(rows4),
              "recomputed_case_counts": len(rows2) == 32768 and len(rows3) == 20480 and len(rows4) == 3072,
              "causal_raw_preserved": len(causal_raw) == 18400,
              "salvage_group_count_consistent": all(panel["n"] == len(salvaged_eligible) for panel in salvage.values()),
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "phase2552_corrected": summary2,
              "phase2553_corrected": summary3, "phase2554_corrected": summary4,
              "phase2555_salvage": {"eligible_total_corrected": len(eligible4),
                                    "eligible_present_in_old_causal": len(salvaged_eligible), "conditions": salvage},
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values()),
              "files": {"phase2552": str(path2), "phase2553": str(path3), "phase2554": str(path4),
                        "phase2554_material": str(material4_path), "phase2555_salvage": str(salvage_path)}}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
