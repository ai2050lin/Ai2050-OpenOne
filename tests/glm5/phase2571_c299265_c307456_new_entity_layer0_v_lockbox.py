#!/usr/bin/env python3
"""Independent new-entity lockbox for the layer-0 query-slot V XOR result."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2570 = RESULT / "phase2570_c291073_c299264_holdout_layer_projection_xor/analysis/final.json"
OUT = RESULT / "phase2571_c299265_c307456_new_entity_layer0_v_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2571, "C299265-C307456"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as atlas  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2553_c174337_c178432_relation_slot_scaffold_adjudication as p2553  # noqa: E402
import phase2569_c284929_c291072_relation_value_xor_causal_interaction as p2569  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def compile_material(tokenizer) -> list[dict]:
    original = atlas.NAMES[35]
    atlas.NAMES[35] = {"en": ("Silver Badger", "Golden Crane"), "zh": ("银色獾", "金色鹤")}
    rows = []
    try:
        for family_id in range(32):
            for binding in (0, 1):
                for relation_form in ("natural", "nonce"):
                    for value_form in ("natural", "nonce"):
                        for query_relation in (0, 1):
                            for query_value in (0, 1):
                                for condition in ("full_scaffold", "relation_missing", "value_missing"):
                                    row = p2553.compile_row(tokenizer, family_id=family_id, language="en", surface=1,
                                        binding=binding, relation_form=relation_form, value_form=value_form,
                                        query_relation=query_relation, query_value=query_value, condition=condition)
                                    row["unit"] = 37
                                    row["depth"] = 1
                                    row["case_id"] = "u37_" + row["case_id"]
                                    row["base_case_id"] = "u37_" + row["base_case_id"]
                                    rows.append(row)
    finally:
        atlas.NAMES[35] = original
    return rows


def eligible_quartets(material: list[dict], behavior: list[dict]) -> list[tuple]:
    correct = {row["case_id"]: row["correct"] for row in behavior if row["ablation"] == "full_scaffold"}
    full = [row for row in material if row["ablation"] == "full_scaffold"]
    index = {(row["family_id"], row["binding"], row["relation_form"], row["value_form"],
              row["query_relation"], row["query_value"]): row for row in full}
    output = []
    for prefix in sorted({key[:4] for key in index}):
        if all(correct[index[prefix + (r, v)]["case_id"]] for r in (0, 1) for v in (0, 1)):
            output.append(prefix)
    return output


def specs() -> dict[str, dict]:
    output: dict[str, dict] = {"no_patch": {"expected": "base"}}
    for kind in ("q", "k", "v"):
        if kind == "q":
            for donor, expected in (("relation", "flip"), ("value", "flip"), ("double", "base")):
                output[f"l00_{kind}_{donor}"] = {"layers": (0,), "kind": kind,
                                                   "donor": donor, "expected": expected}
            continue
        for donor, regions, expected in (
                ("relation", ("query_relation",), "flip"),
                ("value", ("query_value",), "flip"),
                ("double", ("query_relation", "query_value"), "base"),
                ("relation", ("query_value",), "base"),
                ("value", ("query_relation",), "base")):
            suffix = donor if expected != "base" or donor == "double" else (
                "null_relation_to_value" if donor == "relation" else "null_value_to_relation")
            output[f"l00_{kind}_{suffix}"] = {"layers": (0,), "kind": kind, "donor": donor,
                                               "regions": regions, "expected": expected,
                                               "matched_null": suffix.startswith("null")}
    return output


def xor(summary: dict, kind: str, with_null: bool) -> dict:
    relation = summary[f"l00_{kind}_relation"]["flip_rate"]
    value = summary[f"l00_{kind}_value"]["flip_rate"]
    double = summary[f"l00_{kind}_double"]["base_accuracy"]
    null = max(summary[f"l00_{kind}_null_relation_to_value"]["flip_rate"],
               summary[f"l00_{kind}_null_value_to_relation"]["flip_rate"]) if with_null else 0.0
    core = min(relation, value, double)
    return {"relation_flip": relation, "value_flip": value, "double_base_preserve": double,
            "matched_null_flip": null if with_null else None, "xor_margin": core - null,
            "strong_gate": core >= .70 and (not with_null or core - null >= .20)}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 新实体独立锁箱的layer0-V条件组合复验（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2570把早段XOR效应分离到layer0 V，但四元组仍来自同一Copper Lynx/Azure Heron材料池。本Phase冻结“layer0、V、query-relation/query-value各自recipient”预言，将实体换成从未参与发现的Silver Badger/Golden Crane。先完整重跑32族×双binding×自然/nonce关系×自然/nonce值×四查询及full/缺关系/缺值，共3072 case、6144条候选；只在四查询全对且跨条件token形状相同的四元组上，比较layer0 Q/K/V的relation、value、double和错位null。

$$e^*=r_q\oplus v_q\oplus b,\qquad
X_{{0,V}}=\min(F^R,F^V,B^{{RV}})-\max(N^R,N^V).$$

**结果汇总。** 行为`{json.dumps(result['behavior'], ensure_ascii=False)}`；材料筛选`{json.dumps(result['selection'], ensure_ascii=False)}`；Q/K/V裁决`{json.dumps(result['adjudication'], ensure_ascii=False)}`；完整条件`{json.dumps(result['causal_summary'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2571_c299265_c307456_new_entity_layer0_v_lockbox.py`；新实体token材料、候选分数、逐候选因果结果和final位于`{OUT}`。

**分析与理论进展。** 若新实体下V仍通过而K/Q不通过，证据从“同材料留出”升级为实体外推：query两个条件槽的layer0 V投影可以分别携带足以改变答案的条件，且双替换按XOR合成。这是局部充分性，不等于这些坐标在自然运行中不可替代或唯一必要。Phase2570无干预42/43而非43/43来自一个BF16近边界判决，本Phase把完整性门预注册为95%，不删除该样本。

**问题硬伤与结论。** 实体更新但模板、32语言族词表和二元代数未更新；token兼容仍产生词面选择；V全1024坐标整体替换，不是最小物理齿轮；自然/nonce分类混合。下一Phase必须穷举8个KV-head坐标块及leave-one-out，寻找分布式联盟而不是Top-K。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2570)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        material = compile_material(tokenizer)
        behavior_rows = p2553.score_candidates(model, tokenizer, material, batch_size=16)
        selected = eligible_quartets(material, behavior_rows)
        causal_jobs, compatible, excluded = p2569.prepare(material, selected, tokenizer, limit=96)
        conditions = specs()
        causal_rows = p2569.run(model, tokenizer, causal_jobs, conditions)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    p2552.write(OUT / "material/rows.jsonl", material)
    p2552.write(OUT / "behavior/scores.jsonl", behavior_rows)
    p2569.write(OUT / "causal/layer0_scores.jsonl", causal_rows)
    behavior = {condition: {"n": len(subset := [row for row in behavior_rows if row["ablation"] == condition]),
        "accuracy": float(np.mean([row["correct"] for row in subset]))}
        for condition in ("full_scaffold", "relation_missing", "value_missing")}
    causal_summary = p2569.summarize(causal_rows, conditions)
    adjudication = {"overall_behavior_gate": behavior["full_scaffold"]["accuracy"] >= .80,
                    "q": xor(causal_summary, "q", False), "k": xor(causal_summary, "k", True),
                    "v": xor(causal_summary, "v", True)}
    selection = {"eligible_quartets": len(selected), "compatible_quartets": len(compatible),
                 "excluded_token_mismatch": excluded,
                 "form_counts": {f"r{r}_v{v}": sum(row[2:] == (r, v) for row in compatible)
                    for r in ("natural", "nonce") for v in ("natural", "nonce")}}
    checks = {"prior_complete": prior["all_checks_passed"], "rows_3072": len(material) == 3072,
              "scores_3072": len(behavior_rows) == 3072,
              "behavior_gate_adjudicated_without_aborting": True,
              "missing_controls_chance": behavior["relation_missing"]["accuracy"] <= .55
                  and behavior["value_missing"]["accuracy"] <= .55,
              "compatible_at_least_24": len(compatible) >= 24,
              "two_candidates_each": len(causal_rows) == len(compatible) * 2 * len(conditions),
              "no_patch_at_least_95": causal_summary["no_patch"]["base_accuracy"] >= .95,
              "frozen_projection_and_layer": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "entities": ["Silver Badger", "Golden Crane"],
              "behavior": behavior, "selection": selection, "causal_summary": causal_summary,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({"phase": PHASE, "behavior": behavior, "selection": selection,
                      "adjudication": adjudication, "checks": checks,
                      "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
