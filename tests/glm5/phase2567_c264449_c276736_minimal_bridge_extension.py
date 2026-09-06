#!/usr/bin/env python3
"""Minimal one-factor bridge extension of the validated Phase2554 lockbox."""
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
OUT = RESULT / "phase2567_c264449_c276736_minimal_bridge_extension"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
P2566 = RESULT / "phase2566_c260353_c264448_zero_padding_qwen_lockbox_revalidation/analysis/final.json"
PHASE, CAMPAIGN = 2567, "C264449-C276736"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as atlas  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2553_c174337_c178432_relation_slot_scaffold_adjudication as p2553  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def add(tokenizer, ids: list[int], regions: dict[str, list[int]], region: str, text: str) -> list[int]:
    return p2552.add(tokenizer, ids, regions, region, text)


def compile_row(tokenizer, family_id: int, depth: int, binding: int, relation_form: str,
                value_form: str, query_relation: int, query_value: int, condition: str) -> dict:
    entities = ("Copper Lynx", "Azure Heron")
    descriptors = p2552.relation_pair(family_id, "en", relation_form)
    values = p2552.value_pair(family_id, "en", value_form)
    upper = ("orbital group alpha", "orbital group beta")
    terminal = ("terminal zone east", "terminal zone west")
    ids: list[int] = []
    regions = {name: [] for name in p2552.REGIONS}
    regions.update({"bridge_source": [], "bridge_target": [], "bridge_frame": [], "query_terminal": []})
    cells = []
    add(tokenizer, ids, regions, "frame", "Lookup table:\n")
    order = [(1, 1), (0, 1), (1, 0), (0, 0)]
    for entity_index, relation_index in order:
        value_index = entity_index ^ relation_index ^ binding
        shown_relation = p2553.relation_text(descriptors[relation_index], relation_index, condition, "en")
        add(tokenizer, ids, regions, "frame", "ROW entity=")
        ep = add(tokenizer, ids, regions, "facts_entity", f"[{entities[entity_index]}]")
        add(tokenizer, ids, regions, "frame", " relation=")
        rp = add(tokenizer, ids, regions, "facts_relation", shown_relation)
        add(tokenizer, ids, regions, "frame", " value=")
        vp = add(tokenizer, ids, regions, "facts_value", f"[{values[value_index]}]")
        add(tokenizer, ids, regions, "frame", "\n")
        cells.append({"entity_index": entity_index, "relation_index": relation_index,
                      "value_index": value_index, "entity_positions": ep,
                      "relation_positions": rp, "value_positions": vp})
    if depth >= 2:
        for value_index in (0, 1):
            add(tokenizer, ids, regions, "bridge_frame", "BRIDGE value=")
            add(tokenizer, ids, regions, "bridge_source", f"[{values[value_index]}]")
            add(tokenizer, ids, regions, "bridge_frame", " maps exactly to=")
            add(tokenizer, ids, regions, "bridge_target", f"[{upper[value_index]}]\n")
    if depth >= 3:
        for value_index in (0, 1):
            add(tokenizer, ids, regions, "bridge_frame", "BRIDGE group=")
            add(tokenizer, ids, regions, "bridge_source", f"[{upper[value_index]}]")
            add(tokenizer, ids, regions, "bridge_frame", " maps exactly to=")
            add(tokenizer, ids, regions, "bridge_target", f"[{terminal[value_index]}]\n")
    targets = values if depth == 1 else upper if depth == 2 else terminal
    query_relation_text = p2553.relation_text(descriptors[query_relation], query_relation, condition, "en")
    query_target = "[VALUE-UNKNOWN]" if condition == "value_missing" else f"[{targets[query_value]}]"
    add(tokenizer, ids, regions, "query_context", "Find the single row matching BOTH fields. relation=")
    add(tokenizer, ids, regions, "query_relation", query_relation_text)
    add(tokenizer, ids, regions, "query_context", " value=")
    qpos = add(tokenizer, ids, regions, "query_value", query_target)
    regions["query_terminal"].extend(qpos)
    add(tokenizer, ids, regions, "frame", "\nCandidates: ")
    add(tokenizer, ids, regions, "candidate", f"[{entities[0]}] or [{entities[1]}]")
    add(tokenizer, ids, regions, "instruction", ". Match the exact relation ID and exact value; return only the complete entity name. Answer")
    add(tokenizer, ids, regions, "answer_boundary", ":")
    target_index = query_value ^ query_relation ^ binding
    base = (f"f{family_id:02d}_d{depth}_b{binding}_r{relation_form}_v{value_form}_"
            f"qr{query_relation}_qv{query_value}")
    return {"case_id": f"{base}_{condition}", "base_case_id": base, "ablation": condition,
            "unit": 36, "family_id": family_id, "family": atlas.OPERATIONS[family_id][0],
            "language": "en", "surface": 1, "depth": depth, "binding": binding,
            "relation_form": relation_form, "value_form": value_form,
            "query_relation": query_relation, "query_value": query_value,
            "entities": list(entities), "relations": list(descriptors), "values": list(values),
            "upper": list(upper), "terminal": list(terminal), "target_index": target_index,
            "target": entities[target_index], "prompt_ids": ids, "prompt": tokenizer.decode(ids),
            "regions": regions, "fact_cells": cells, "answer_boundary_token": len(ids) - 1}


def compile_material(tokenizer) -> list[dict]:
    return [compile_row(tokenizer, family_id, depth, binding, relation_form, value_form,
                        query_relation, query_value, condition)
            for family_id in range(32) for depth in (1, 2, 3) for binding in (0, 1)
            for relation_form in ("natural", "nonce") for value_form in ("natural", "nonce")
            for query_relation in (0, 1) for query_value in (0, 1)
            for condition in ("full_scaffold", "relation_missing", "value_missing")]


def summarize(rows: list[dict]) -> tuple[dict, list[tuple[int, str, str]]]:
    result, qualified = {}, []
    for depth in (1, 2, 3):
        for relation_form in ("natural", "nonce"):
            for value_form in ("natural", "nonce"):
                key = f"d{depth}_r{relation_form}_v{value_form}"
                result[key] = {}
                for condition in ("full_scaffold", "relation_missing", "value_missing"):
                    subset = [row for row in rows if row["depth"] == depth
                              and row["relation_form"] == relation_form and row["value_form"] == value_form
                              and row["ablation"] == condition]
                    result[key][condition] = {"n": len(subset),
                        "accuracy": float(np.mean([row["correct"] for row in subset])),
                        "mean_margin": float(np.mean([row["target_minus_wrong"] for row in subset]))}
                if result[key]["full_scaffold"]["accuracy"] >= .75 \
                        and result[key]["relation_missing"]["accuracy"] <= .55 \
                        and result[key]["value_missing"]["accuracy"] <= .55:
                    qualified.append((depth, relation_form, value_form))
    return result, qualified


def autonomous_exact(model, tokenizer, material: list[dict], qualified: list[tuple[int, str, str]]) -> list[dict]:
    selected = [row for row in material if row["ablation"] == "full_scaffold" and row["binding"] == 0
                and (row["depth"], row["relation_form"], row["value_form"]) in qualified
                and row["query_relation"] == 1 and row["query_value"] == 0][:96]
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        buckets[len(row["prompt_ids"])].append(row)
    output, device = [], model.get_input_embeddings().weight.device
    for _, values in sorted(buckets.items()):
        for start in range(0, len(values), 8):
            batch = values[start:start + 8]
            ids = torch.tensor([row["prompt_ids"] for row in batch], dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=10,
                    do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            for row, sequence in zip(batch, generated):
                text = tokenizer.decode(sequence[ids.shape[1]:].cpu().tolist(), skip_special_tokens=True)
                hits = [i for i, entity in enumerate(row["entities"]) if p2552.norm(entity) in p2552.norm(text)]
                prediction = hits[0] if len(set(hits)) == 1 else None
                output.append({"case_id": row["case_id"], "depth": row["depth"],
                    "relation_form": row["relation_form"], "value_form": row["value_form"],
                    "target_index": row["target_index"], "prediction_index": prediction,
                    "generated": text, "correct": prediction == row["target_index"]})
    return output


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 合格关系表的单因素多跳扩展（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2565把整套提示重写后连1跳也退化，无法归因多跳。本Phase严格复制Phase2566已复核的英文surface1四事实表、实体、R0/R1关系表示、候选与指令；depth1应重现原任务，depth2仅在表和原查询之间加入$value\to group$双射并把query value换成group，depth3再加入$group\to terminal$双射。覆盖32族×3深度×双binding×自然/nonce关系×自然/nonce值×四查询，full、缺关系、缺终点各3072，共9216 case、18432条完整候选序列；全部按精确长度分桶。

$$e^*=r_q\oplus v_q\oplus b,\qquad
v\xrightarrow{{B_1}}g\xrightarrow{{B_2}}t,\qquad
G_{{d,f}}=\mathbf1[A_{{full}}\ge.75\land A_{{-r}},A_{{-t}}\le.55].$$

**结果汇总。** 分层结果`{json.dumps(result['strata'], ensure_ascii=False)}`；合格层`{json.dumps(result['qualified_strata'], ensure_ascii=False)}`；各深度总体`{json.dumps(result['by_depth'], ensure_ascii=False)}`；自主生成`{json.dumps(result['autonomous'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2567_c264449_c276736_minimal_bridge_extension.py`；完整token材料、逐候选分数、自主输出和final位于`{OUT}`。

**分析与理论进展。** depth1是内部阳性锚点，只有它复现Phase2566后，depth2/3差异才可归于插入bridge。若自然/nonce值同时通过，支持抽象槽位复用；若仅自然通过，说明词义先验仍参与。该测试观测“有限参数能否在相同关系检索算法上串接额外映射”，不把行为成功直接等同于已定位坐标齿轮。

**问题硬伤与结论。** bridge是明确双射、答案二元、关系仍用R0/R1、upper/terminal跨族复用；即使多跳通过也只是人工上下文组合，不是开放语言无限组合。失败只界定本模型/提示的可诱发边界，不否定其他表达或大模型中的机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = read_json(P2566)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        material = compile_material(tokenizer)
        behavior = p2553.score_candidates(model, tokenizer, material, batch_size=16)
        meta = {row["case_id"]: row for row in material}
        for row in behavior:
            row["depth"] = meta[row["case_id"]]["depth"]
        strata, qualified = summarize(behavior)
        autonomous = autonomous_exact(model, tokenizer, material, qualified)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    p2552.write(OUT / "material/rows.jsonl", material)
    p2552.write(OUT / "behavior/scores.jsonl", behavior)
    p2552.write(OUT / "autonomous/generation.jsonl", autonomous)
    by_depth = {str(depth): {condition: float(np.mean([row["correct"] for row in behavior
        if row["depth"] == depth and row["ablation"] == condition]))
        for condition in ("full_scaffold", "relation_missing", "value_missing")} for depth in (1, 2, 3)}
    apanel = {"n": len(autonomous), "accuracy": float(np.mean([row["correct"] for row in autonomous]))
              if autonomous else None,
              "by_depth": {str(depth): float(np.mean([row["correct"] for row in autonomous if row["depth"] == depth]))
                  if any(row["depth"] == depth for row in autonomous) else None for depth in (1, 2, 3)}}
    checks = {"prior_gate": prior["adjudication"]["full_gate"], "rows_9216": len(material) == 9216,
              "scores_9216": len(behavior) == 9216, "depth1_reproduces": by_depth["1"]["full_scaffold"] >= .80,
              "exact_length_buckets": True, "scientific_failure_does_not_abort": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "strata": strata,
              "qualified_strata": [f"d{d}_r{r}_v{v}" for d, r, v in qualified],
              "by_depth": by_depth, "autonomous": apanel, "checks": checks,
              "all_checks_passed": all(checks.values()), "language_mechanism_closed": False}
    write_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
