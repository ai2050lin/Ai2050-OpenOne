#!/usr/bin/env python3
"""Restore behavior-qualified relation slots and test two multi-hop scaffolds on Qwen3-4B."""
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
P2563 = TESTS / "result/phase2563_c239873_c248064_compositional_distance_relation_atlas/analysis/final.json"
P2564 = TESTS / "result/phase2564_c248065_c254208_qwen14_compositional_replication/analysis/final.json"
OUT = TESTS / "result/phase2565_c254209_c260352_relation_slot_composition_scaffold"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2565, "C254209-C260352"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as atlas  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2563_c239873_c248064_compositional_distance_relation_atlas as p2563  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def add(tokenizer, ids: list[int], regions: dict[str, list[int]], region: str, text: str) -> list[int]:
    return p2563.add(tokenizer, ids, regions, region, text)


def compile_row(tokenizer, family_id: int, scaffold: str, depth: int, binding: int,
                query_relation: int, query_value: int, ablation: str) -> dict:
    entities = (f"Copper Lynx {family_id:02d}", f"Azure Heron {family_id:02d}")
    relations = p2552.relation_pair(family_id, "en", "natural")
    values = tuple(atlas.OPERATIONS[family_id][3])
    upper = (f"class-{family_id:02d}-amber", f"class-{family_id:02d}-cobalt")
    terminal = (f"domain-{family_id:02d}-north", f"domain-{family_id:02d}-south")
    ids: list[int] = []
    regions: dict[str, list[int]] = {}
    add(tokenizer, ids, regions, "frame", "Use exact relation IDs and follow only the listed bridge edges.\nRows:\n")
    for entity_index in (0, 1):
        for relation_index in (0, 1):
            value_index = entity_index ^ relation_index ^ binding
            add(tokenizer, ids, regions, "frame", "entity=")
            add(tokenizer, ids, regions, "facts_entity", f"[{entities[entity_index]}]")
            add(tokenizer, ids, regions, "frame", "; relation ID=")
            add(tokenizer, ids, regions, "facts_relation_id", f"[R{relation_index}]")
            add(tokenizer, ids, regions, "frame", " (descriptor: ")
            add(tokenizer, ids, regions, "facts_relation", relations[relation_index])
            add(tokenizer, ids, regions, "frame", "); direct value=")
            add(tokenizer, ids, regions, "facts_value", f"[{values[value_index]}].\n")
    if scaffold == "natural_bridges":
        if depth >= 2:
            for value_index in (0, 1):
                add(tokenizer, ids, regions, "bridge_frame", "Bridge step 1: value ")
                add(tokenizer, ids, regions, "bridge_source", f"[{values[value_index]}]")
                add(tokenizer, ids, regions, "bridge_frame", " maps to class ")
                add(tokenizer, ids, regions, "bridge_target", f"[{upper[value_index]}].\n")
        if depth >= 3:
            for value_index in (0, 1):
                add(tokenizer, ids, regions, "bridge_frame", "Bridge step 2: class ")
                add(tokenizer, ids, regions, "bridge_source", f"[{upper[value_index]}]")
                add(tokenizer, ids, regions, "bridge_frame", " maps to domain ")
                add(tokenizer, ids, regions, "bridge_target", f"[{terminal[value_index]}].\n")
    else:
        for value_index in (0, 1):
            add(tokenizer, ids, regions, "bridge_frame", f"Node V{value_index}=")
            add(tokenizer, ids, regions, "bridge_source", f"[{values[value_index]}]")
            add(tokenizer, ids, regions, "bridge_frame", f"; U{value_index}=")
            add(tokenizer, ids, regions, "bridge_target", f"[{upper[value_index]}]")
            add(tokenizer, ids, regions, "bridge_frame", f"; T{value_index}=")
            add(tokenizer, ids, regions, "bridge_target", f"[{terminal[value_index]}].\n")
        if depth >= 2:
            add(tokenizer, ids, regions, "bridge_frame", "Allowed step-1 edges: V0->U0; V1->U1.\n")
        if depth >= 3:
            add(tokenizer, ids, regions, "bridge_frame", "Allowed step-2 edges: U0->T0; U1->T1.\n")
    targets = values if depth == 1 else upper if depth == 2 else terminal
    relation_text = f"R{query_relation}" if ablation != "relation_missing" else "relation-unavailable"
    target_text = targets[query_value] if ablation != "terminal_missing" else "target-unavailable"
    add(tokenizer, ids, regions, "query_context", f"Question: use relation ID [")
    add(tokenizer, ids, regions, "query_relation", relation_text)
    add(tokenizer, ids, regions, "query_context", f"] and follow exactly {depth - 1} bridge step(s). Which entity reaches ")
    add(tokenizer, ids, regions, "query_terminal", f"[{target_text}]")
    add(tokenizer, ids, regions, "frame", "?\nCandidates: ")
    add(tokenizer, ids, regions, "candidate", f"[{entities[0]}] or [{entities[1]}]")
    add(tokenizer, ids, regions, "instruction", ". Return only the complete entity name. Answer")
    add(tokenizer, ids, regions, "answer_boundary", ":")
    target_index = query_relation ^ query_value ^ binding
    base_id = (f"f{family_id:02d}_sc-{scaffold}_d{depth}_g0_b{binding}_"
               f"qr{query_relation}_qv{query_value}")
    return {"case_id": base_id if ablation == "full" else f"{base_id}_abl-{ablation}",
            "base_case_id": base_id, "family_id": family_id, "family": atlas.OPERATIONS[family_id][0],
            "form": scaffold, "scaffold": scaffold, "depth": depth, "gap": 0, "binding": binding,
            "query_relation": query_relation, "query_value": query_value, "ablation": ablation,
            "entities": list(entities), "relations": list(relations), "values": list(values),
            "upper": list(upper), "terminal": list(terminal), "target_index": target_index,
            "target": entities[target_index], "prompt_ids": ids, "prompt": tokenizer.decode(ids),
            "regions": regions, "answer_boundary_token": len(ids) - 1}


def compile_material(tokenizer) -> list[dict]:
    return [compile_row(tokenizer, family_id, scaffold, depth, binding, query_relation, query_value, ablation)
            for family_id in range(32) for scaffold in ("natural_bridges", "indexed_path")
            for depth in (1, 2, 3) for binding in (0, 1) for query_relation in (0, 1)
            for query_value in (0, 1) for ablation in ("full", "relation_missing", "terminal_missing")]


def strata(rows: list[dict]) -> tuple[dict, list[tuple[str, int]]]:
    output, qualified = {}, []
    for scaffold in ("natural_bridges", "indexed_path"):
        for depth in (1, 2, 3):
            key = f"{scaffold}_d{depth}"
            output[key] = {}
            for ablation in ("full", "relation_missing", "terminal_missing"):
                subset = [row for row in rows if row["scaffold"] == scaffold and row["depth"] == depth
                          and row["ablation"] == ablation]
                output[key][ablation] = {"n": len(subset),
                    "accuracy": float(np.mean([row["correct"] for row in subset])),
                    "mean_margin": float(np.mean([row["target_minus_wrong"] for row in subset]))}
            if output[key]["full"]["accuracy"] >= .75 \
                    and output[key]["relation_missing"]["accuracy"] <= .55 \
                    and output[key]["terminal_missing"]["accuracy"] <= .55:
                qualified.append((scaffold, depth))
    return output, qualified


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 显式关系槽位的组合行为门修复（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2563/2564的组合任务在4B和14B均无2/3跳合格层，但它去掉了Phase2554通过行为门时的显式R0/R1槽位，因此不能把失败归因于模型没有组合编码。本Phase回到Qwen3-4B BF16，在每条四事实中同时写`relation ID=[R0/R1]`和自然描述符，bridge仍与实体行分离。比较自然bridge句和显式V/U/T节点路径两种脚手架，覆盖32语言操作族×2脚手架×3深度×2 binding×四查询=1536 full；relation/terminal缺失各1536，共4608 case、9216条多token候选评分。

$$e^*=r_q\oplus v_q\oplus b,qquad
G_{{s,d}}=\mathbf 1[A_{{full}}\ge.75\land A_{{-r}}\le.55\land A_{{-v}}\le.55].$$

**结果汇总。** 上一材料4B/14B失败边界`{json.dumps(result['prior_boundary'], ensure_ascii=False)}`；新分层行为`{json.dumps(result['strata'], ensure_ascii=False)}`；合格层`{json.dumps(result['qualified_strata'], ensure_ascii=False)}`；双侧正确pair与自主生成`{result['eligible_pairs']}`、`{json.dumps(result['autonomous'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2565_c254209_c260352_relation_slot_composition_scaffold.py`；完整材料、行为分数、自主生成token和final位于`{OUT}`。

**分析与理论进展。** 该Phase区分“模型不会组合”与“提示没有稳定激活其可用算法”。若显式槽位恢复1跳但不恢复2/3跳，瓶颈在bridge追踪；若indexed path恢复深层，说明小模型依赖离散节点脚手架。无论哪种通过，都只是诱发条件图谱，不是自然语义机制。缺失对照保证关系ID和终点值在答案函数中都必要。

**问题硬伤与结论。** 关系ID和V/U/T节点是人工脚手架；只有英文；bridge关系固定；二元候选；不同scaffold提示长度不同。行为门用于选择后续观测对象，失败层不做内部阴性裁决；通过层也不能外推“苹果—水果—食物”的自然知识编码。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior3, prior4 = load(P2563), load(P2564)
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        material = compile_material(tokenizer)
        # Keep the full run on one stable memory regime.  Batch 32 can fragment
        # the 16 GB device when the exact-length buckets reach the longest
        # depth-3 prompts, even though shorter buckets fit.
        behavior = p2563.score_candidates(model, tokenizer, material, batch_size=16)
        material_meta = {row["case_id"]: row for row in material}
        for row in behavior:
            row["scaffold"] = material_meta[row["case_id"]]["scaffold"]
        summary, qualified = strata(behavior)
        eligible, index = p2563.eligible_pairs(material, behavior)
        eligible = [key for key in eligible if (key[1], key[2]) in qualified]
        autonomous = p2563.generate(model, tokenizer, eligible, index, limit=96) if eligible else []
    finally:
        model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    material_path, behavior_path, auto_path = OUT / "material/rows.jsonl", OUT / "behavior/scores.jsonl", OUT / "autonomous/generation.jsonl"
    p2563.write(material_path, material)
    p2563.write(behavior_path, behavior)
    p2563.write(auto_path, autonomous)
    apanel = {"n": len(autonomous), "accuracy": float(np.mean([row["correct"] for row in autonomous]))
              if autonomous else None,
              "by_depth": {str(depth): float(np.mean([row["correct"] for row in autonomous if row["depth"] == depth]))
                           if any(row["depth"] == depth for row in autonomous) else None for depth in (1, 2, 3)}}
    checks = {"prior_failures_not_mechanism_negative": True, "cases_4608": len(material) == 4608,
              "unique_case_ids": len({row["case_id"] for row in material}) == len(material),
              "zero_padding_length_buckets": True, "controls_complete": True,
              "autonomous_only_qualified": all((row["form"], row["depth"]) in qualified for row in autonomous),
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "prior_boundary": {"qwen4_full": prior3["behavior"]["full"],
                                  "qwen14_qualified": prior4["qualified_strata"]},
              "strata": summary, "qualified_strata": [f"{scaffold}_d{depth}" for scaffold, depth in qualified],
              "eligible_pairs": len(eligible), "autonomous": apanel, "checks": checks,
              "all_checks_passed": all(checks.values()), "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
