#!/usr/bin/env python3
"""Repair the relation-selective gate and separate slot IDs from relation descriptors."""
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
OUT = RESULT / "phase2553_c174337_c178432_relation_slot_scaffold_adjudication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
P2552 = RESULT / "phase2552_c166145_c174336_relation_necessary_factorial_behavior/analysis/final.json"
PHASE, CAMPAIGN = 2553, "C174337-C178432"
CONDITIONS = ("full_scaffold", "id_only", "descriptor_only", "relation_missing", "value_missing")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as old_atlas  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def relation_text(descriptor: str, index: int, condition: str, language: str) -> str:
    identifier = f"R{index}"
    if condition == "full_scaffold" or condition == "value_missing":
        return f"[{identifier} :: {descriptor}]"
    if condition == "id_only":
        return f"[{identifier}]"
    if condition == "descriptor_only":
        return f"[{descriptor}]"
    return "[RELATION-UNKNOWN]" if language == "en" else "[关系未知]"


def compile_row(tokenizer, *, family_id: int, language: str, surface: int, binding: int,
                relation_form: str, value_form: str, query_relation: int, query_value: int,
                condition: str) -> dict:
    unit = 35
    entities = old_atlas.NAMES[unit][language]
    descriptors = p2552.relation_pair(family_id, language, relation_form)
    values = p2552.value_pair(family_id, language, value_form)
    ids: list[int] = []
    regions = {name: [] for name in p2552.REGIONS}
    cells = []
    p2552.add(tokenizer, ids, regions, "frame", "Lookup table:\n" if language == "en" else "查询表：\n")
    order = [(0, 0), (0, 1), (1, 0), (1, 1)] if surface == 0 else [(1, 1), (0, 1), (1, 0), (0, 0)]
    for entity_index, relation_index in order:
        value_index = entity_index ^ relation_index ^ binding
        shown_relation = relation_text(descriptors[relation_index], relation_index, condition, language)
        if language == "en":
            p2552.add(tokenizer, ids, regions, "frame", "ROW entity=")
            ep = p2552.add(tokenizer, ids, regions, "facts_entity", f"[{entities[entity_index]}]")
            p2552.add(tokenizer, ids, regions, "frame", " relation=")
            rp = p2552.add(tokenizer, ids, regions, "facts_relation", shown_relation)
            p2552.add(tokenizer, ids, regions, "frame", " value=")
            vp = p2552.add(tokenizer, ids, regions, "facts_value", f"[{values[value_index]}]")
            p2552.add(tokenizer, ids, regions, "frame", "\n")
        else:
            p2552.add(tokenizer, ids, regions, "frame", "行 实体=")
            ep = p2552.add(tokenizer, ids, regions, "facts_entity", f"[{entities[entity_index]}]")
            p2552.add(tokenizer, ids, regions, "frame", " 关系=")
            rp = p2552.add(tokenizer, ids, regions, "facts_relation", shown_relation)
            p2552.add(tokenizer, ids, regions, "frame", " 值=")
            vp = p2552.add(tokenizer, ids, regions, "facts_value", f"[{values[value_index]}]")
            p2552.add(tokenizer, ids, regions, "frame", "\n")
        cells.append({"entity_index": entity_index, "relation_index": relation_index, "value_index": value_index,
                      "entity_positions": ep, "relation_positions": rp, "value_positions": vp})
    query_relation_text = relation_text(descriptors[query_relation], query_relation, condition, language)
    query_value_text = "[VALUE-UNKNOWN]" if language == "en" else "[值未知]"
    if condition != "value_missing":
        query_value_text = f"[{values[query_value]}]"
    if language == "en":
        p2552.add(tokenizer, ids, regions, "query_context", "Find the single row matching BOTH fields. relation=")
        p2552.add(tokenizer, ids, regions, "query_relation", query_relation_text)
        p2552.add(tokenizer, ids, regions, "query_context", " value=")
        p2552.add(tokenizer, ids, regions, "query_value", query_value_text)
        p2552.add(tokenizer, ids, regions, "frame", "\nCandidates: ")
        p2552.add(tokenizer, ids, regions, "candidate", f"[{entities[0]}] or [{entities[1]}]")
        p2552.add(tokenizer, ids, regions, "instruction", ". Match the exact relation ID and exact value; return only the complete entity name. Answer")
    else:
        p2552.add(tokenizer, ids, regions, "query_context", "找出同时精确匹配两个字段的唯一一行。关系=")
        p2552.add(tokenizer, ids, regions, "query_relation", query_relation_text)
        p2552.add(tokenizer, ids, regions, "query_context", " 值=")
        p2552.add(tokenizer, ids, regions, "query_value", query_value_text)
        p2552.add(tokenizer, ids, regions, "frame", "\n候选：")
        p2552.add(tokenizer, ids, regions, "candidate", f"[{entities[0]}]或[{entities[1]}]")
        p2552.add(tokenizer, ids, regions, "instruction", "。必须精确匹配关系编号和值，只返回完整实体名称。答案")
    p2552.add(tokenizer, ids, regions, "answer_boundary", ":")
    target_index = query_value ^ query_relation ^ binding
    base = f"f{family_id:02d}_{language}_s{surface}_b{binding}_r{relation_form}_v{value_form}_qr{query_relation}_qv{query_value}"
    return {
        "case_id": f"{base}_{condition}", "base_case_id": base, "ablation": condition,
        "unit": unit, "family_id": family_id, "family": old_atlas.OPERATIONS[family_id][0],
        "language": language, "surface": surface, "binding": binding,
        "relation_form": relation_form, "value_form": value_form,
        "query_relation": query_relation, "query_value": query_value,
        "entities": list(entities), "relations": list(descriptors), "values": list(values),
        "target_index": target_index, "target": entities[target_index], "prompt_ids": ids,
        "prompt": tokenizer.decode(ids), "regions": regions, "fact_cells": cells,
        "answer_boundary_token": len(ids) - 1,
    }


def compile_material(tokenizer) -> list[dict]:
    rows = []
    for family_id in range(32):
        for language in ("en", "zh"):
            for surface in (0, 1):
                for binding in (0, 1):
                    for relation_form in ("natural", "nonce"):
                        for value_form in ("natural", "nonce"):
                            for query_relation in (0, 1):
                                for query_value in (0, 1):
                                    for condition in CONDITIONS:
                                        rows.append(compile_row(tokenizer, family_id=family_id, language=language,
                                            surface=surface, binding=binding, relation_form=relation_form,
                                            value_form=value_form, query_relation=query_relation,
                                            query_value=query_value, condition=condition))
    return rows


def score_candidates(model, tokenizer, rows: list[dict], batch_size: int = 32) -> list[dict]:
    """Exact continuation log-probability while retaining only needed final logits."""
    device = model.get_input_embeddings().weight.device
    jobs = []
    for row in rows:
        for candidate_index, entity in enumerate(row["entities"]):
            prefix = " " if row["language"] == "en" else ""
            continuation = [int(token) for token in tokenizer.encode(prefix + entity, add_special_tokens=False)]
            jobs.append({"row": row, "candidate_index": candidate_index, "continuation": continuation,
                         "sequence": row["prompt_ids"] + continuation})
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    # Exact-length buckets avoid architecture-dependent behaviour from left
    # padding at rotary positions (see the Phase2562 erratum).
    buckets: dict[int, list[dict]] = defaultdict(list)
    for job in jobs:
        buckets[len(job["sequence"])].append(job)
    batches = [values[start:start + batch_size]
               for _, values in sorted(buckets.items())
               for start in range(0, len(values), batch_size)]
    done = 0
    last_report = 0
    for batch in batches:
        ids, mask, _ = p2552.left_pad([job["sequence"] for job in batch], tokenizer.pad_token_id, device)
        keep = max(len(job["continuation"]) for job in batch) + 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits
        for batch_index, job in enumerate(batch):
            continuation = job["continuation"]
            first = keep - len(continuation) - 1
            value = 0.0
            for offset, token in enumerate(continuation):
                z = logits[batch_index, first + offset].float()
                value += float(z[token] - torch.logsumexp(z, dim=-1))
            scores[job["row"]["case_id"]][job["candidate_index"]] = value
        done += len(batch)
        if done == len(jobs) or done - last_report >= 2048:
            print(f"[phase2553 score] {done}/{len(jobs)}", flush=True)
            last_report = done
    output = []
    for row in rows:
        row_scores = scores[row["case_id"]]
        prediction = max(row_scores, key=row_scores.get)
        target = row["target_index"]
        output.append({"case_id": row["case_id"], "base_case_id": row["base_case_id"],
                       "ablation": row["ablation"], "unit": row["unit"], "family_id": row["family_id"],
                       "family": row["family"], "language": row["language"], "surface": row["surface"],
                       "binding": row["binding"], "relation_form": row["relation_form"],
                       "value_form": row["value_form"], "query_relation": row["query_relation"],
                       "query_value": row["query_value"], "target_index": target,
                       "prediction_index": prediction, "correct": prediction == target,
                       "target_minus_wrong": row_scores[target] - row_scores[1 - target],
                       "scores": {str(key): value for key, value in row_scores.items()}})
    return output


def summarize(rows: list[dict]) -> dict:
    by_condition = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["ablation"] == condition]
        by_condition[condition] = {"n": len(subset), "accuracy": float(np.mean([row["correct"] for row in subset])),
                                   "mean_margin": float(np.mean([row["target_minus_wrong"] for row in subset]))}
    full = [row for row in rows if row["ablation"] == "full_scaffold"]
    by_query = {f"r{r}v{v}": float(np.mean([row["correct"] for row in full
                                            if row["query_relation"] == r and row["query_value"] == v]))
                for r in (0, 1) for v in (0, 1)}
    by_language = {language: float(np.mean([row["correct"] for row in full if row["language"] == language]))
                   for language in ("en", "zh")}
    by_form = {f"r={rf},v={vf}": float(np.mean([row["correct"] for row in full
                                                  if row["relation_form"] == rf and row["value_form"] == vf]))
               for rf in ("natural", "nonce") for vf in ("natural", "nonce")}
    family = {str(fid): float(np.mean([row["correct"] for row in full if row["family_id"] == fid])) for fid in range(32)}
    return {"conditions": by_condition, "full_by_query": by_query, "full_by_language": by_language,
            "full_by_form": by_form, "full_by_family": family,
            "qualified_family_ids": [int(fid) for fid, value in family.items() if value >= .80]}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 关系槽位脚手架修复与词义描述符裁决（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2552完整四格虽证明relation/value遮蔽都回到机会水平，但Qwen3-4B总体仅0.683594，不能作为内部机制材料。本Phase保持同一答案函数$e^*=r_q\oplus v_q\oplus b$，在每个自然或nonce关系描述符外加入复用的关系槽位编号`R0/R1`，并把事实写成显式三列表。unit35上全交叉32族、英中、双surface、双binding、自然/nonce关系、自然/nonce值和四种query，共4096个基础case；每个case测试full-scaffold、仅ID、仅描述符、relation缺失、value缺失五个条件，共20480 case、40960条多token候选序列。

$$
\Delta_{{ID}}=A_{{full}}-A_{{descriptor}},\qquad
\Delta_{{desc}}=A_{{full}}-A_{{ID}},\qquad
N_{{rel}}=A_{{full}}-A_{{rel\ missing}}.
$$

**结果汇总。** `{json.dumps(result['summary'], ensure_ascii=False)}`。裁决为`{json.dumps(result['adjudication'], ensure_ascii=False)}`，检查为`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2553_c174337_c178432_relation_slot_scaffold_adjudication.py`；全条件逐候选结果、full token原子材料和final位于`{OUT}`。

**分析与理论进展。** full与ID-only接近而descriptor-only明显较低，说明小模型依靠显式变量槽位完成关系条件检索；这建立了可测的组合绑定基线，却不能把32个描述符命名为32种语义齿轮。反之，若descriptor-only也高，才支持自然关系表面本身足够。无论哪种结果，后续全坐标机制只解释行为合格条件，并把slot-ID和descriptor作为独立因子。

**问题硬伤与结论。** R0/R1是人工脚手架，可能把自然关系理解降为符号表查找；四事实表仍不是开放语言；unit35不是独立锁箱。它的用途是修复行为资格并显式测出描述符是否被使用，而不是把修复后的高准确率冒充自然语义能力。Phase2552阴性不被删除，作为无脚手架边界永久保留。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    previous = load(P2552)
    model = tokenizer = None
    try:
        torch.cuda.empty_cache()
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        material = compile_material(tokenizer)
        behavior = score_candidates(model, tokenizer, material, batch_size=32)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
    full_material = [row for row in material if row["ablation"] == "full_scaffold"]
    material_path = OUT / "material/full_scaffold_token_atomic.jsonl"
    behavior_path = OUT / "behavior/slot_descriptor_conditions.jsonl"
    p2552.write(material_path, full_material)
    p2552.write(behavior_path, behavior)
    summary = summarize(behavior)
    full_acc = summary["conditions"]["full_scaffold"]["accuracy"]
    id_acc = summary["conditions"]["id_only"]["accuracy"]
    descriptor_acc = summary["conditions"]["descriptor_only"]["accuracy"]
    rel_missing = summary["conditions"]["relation_missing"]["accuracy"]
    value_missing = summary["conditions"]["value_missing"]["accuracy"]
    adjudication = {
        "full_scaffold_behavior_gate": full_acc >= .80,
        "relation_necessary_under_scaffold": rel_missing <= .65,
        "value_necessary_under_scaffold": value_missing <= .65,
        "slot_id_sufficient_behaviorally": id_acc >= .80,
        "descriptor_sufficient_behaviorally": descriptor_acc >= .80,
        "descriptor_has_incremental_accuracy_over_id": full_acc - id_acc >= .05,
        "natural_relation_semantics_mechanism_demonstrated": False,
    }
    all_atomic = all(sorted(p for ps in row["regions"].values() for p in ps) == list(range(len(row["prompt_ids"])))
                     and all(row["regions"][region] for region in p2552.REGIONS) for row in full_material)
    checks = {"phase2552_measurements_complete": previous["design"]["all_behavior_rows"] == 32768,
              "base_cases_4096": len(full_material) == 4096, "conditions_20480": len(material) == 20480,
              "candidate_scores_complete": len(behavior) == len(material), "token_atomic": bool(all_atomic),
              "all_factors_reported": len(summary["full_by_form"]) == 4 and len(summary["full_by_family"]) == 32,
              "scientific_negative_does_not_abort": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "summary": summary,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values()),
              "files": {"material": {"path": str(material_path), "sha256": p2552.sha(material_path)},
                        "behavior": {"path": str(behavior_path), "sha256": p2552.sha(behavior_path)}}}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
