#!/usr/bin/env python3
"""Four-choice relation/value algebra behavior gate on Qwen3-4B with exact-length scoring."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2580_c356609_c364800_fourchoice_relation_value_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
P2579 = RESULT / "phase2579_c352513_c356608_attachment_audit_fourchoice_contract/analysis/final.json"
PHASE, CAMPAIGN = 2580, "C356609-C364800"
ENTITIES = ("Copper Lynx", "Azure Heron", "Silver Badger", "Golden Crane")
REGIONS = ("frame", "facts_entity", "facts_relation", "facts_value", "query_context",
           "query_relation", "query_value", "candidate", "instruction", "answer_boundary")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as old_atlas  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(tokenizer, ids: list[int], regions: dict[str, list[int]], name: str, text: str) -> list[int]:
    tokens = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    if not tokens:
        raise RuntimeError((name, text))
    start = len(ids)
    ids.extend(tokens)
    positions = list(range(start, len(ids)))
    regions[name].extend(positions)
    return positions


def descriptors(family_id: int, relation_form: str, value_form: str) -> tuple[tuple[str, str], tuple[str, str]]:
    if relation_form == "natural":
        partner = (family_id + 11) % len(old_atlas.OPERATIONS)
        relations = (old_atlas.OPERATIONS[family_id][1], old_atlas.OPERATIONS[partner][1])
    else:
        relations = (f"daxel-{family_id:02d}-alpha", f"wugor-{family_id:02d}-beta")
    if value_form == "natural":
        values = tuple(old_atlas.OPERATIONS[family_id][3])
    else:
        values = (f"kivora-{family_id:02d}-alpha", f"mexalu-{family_id:02d}-beta")
    return relations, values


def target_index(query_relation: int, query_value: int, binding_relation: int, binding_value: int) -> int:
    return 2 * (query_relation ^ binding_relation) + (query_value ^ binding_value)


def compile_row(tokenizer, *, family_id: int, binding_relation: int, binding_value: int,
                relation_form: str, value_form: str, query_relation: int, query_value: int,
                ablation: str) -> dict:
    relations, values = descriptors(family_id, relation_form, value_form)
    ids: list[int] = []
    regions = {name: [] for name in REGIONS}
    cells = []
    add(tokenizer, ids, regions, "frame", "Four-key lookup table:\n")
    order = ((1, 1), (0, 1), (1, 0), (0, 0))
    for relation_index, value_index in order:
        entity_index = target_index(relation_index, value_index, binding_relation, binding_value)
        add(tokenizer, ids, regions, "frame", "ENTRY entity=")
        entity_positions = add(tokenizer, ids, regions, "facts_entity", f"[{ENTITIES[entity_index]}]")
        add(tokenizer, ids, regions, "frame", " relation=")
        relation_positions = add(tokenizer, ids, regions, "facts_relation",
                                 f"[R{relation_index} :: {relations[relation_index]}]")
        add(tokenizer, ids, regions, "frame", " value=")
        value_positions = add(tokenizer, ids, regions, "facts_value", f"[V{value_index} :: {values[value_index]}]")
        add(tokenizer, ids, regions, "frame", ".\n")
        cells.append({"entity_index": entity_index, "relation_index": relation_index,
                      "value_index": value_index, "entity_positions": entity_positions,
                      "relation_positions": relation_positions, "value_positions": value_positions})
    relation_text = (f"[R{query_relation} :: {relations[query_relation]}]"
                     if ablation not in ("relation_missing", "both_missing") else "[RELATION UNKNOWN]")
    value_text = (f"[V{query_value} :: {values[query_value]}]"
                  if ablation not in ("value_missing", "both_missing") else "[VALUE UNKNOWN]")
    add(tokenizer, ids, regions, "query_context", "Question: select the one entity matching BOTH relation=")
    add(tokenizer, ids, regions, "query_relation", relation_text)
    add(tokenizer, ids, regions, "query_context", " and value=")
    add(tokenizer, ids, regions, "query_value", value_text)
    add(tokenizer, ids, regions, "frame", ".\nChoices: ")
    add(tokenizer, ids, regions, "candidate", " | ".join(f"[{entity}]" for entity in ENTITIES))
    add(tokenizer, ids, regions, "instruction", ". Return only the exact complete entity name. Answer")
    add(tokenizer, ids, regions, "answer_boundary", ":")
    target = target_index(query_relation, query_value, binding_relation, binding_value)
    binding_id = 2 * binding_relation + binding_value
    base = (f"f{family_id:02d}_br{binding_relation}_bv{binding_value}_r{relation_form}_v{value_form}_"
            f"qr{query_relation}_qv{query_value}")
    return {"case_id": base if ablation == "full" else f"{base}_{ablation}", "base_case_id": base,
            "ablation": ablation, "family_id": family_id, "family": old_atlas.OPERATIONS[family_id][0],
            "binding_id": binding_id, "binding_relation": binding_relation, "binding_value": binding_value,
            "relation_form": relation_form, "value_form": value_form,
            "query_relation": query_relation, "query_value": query_value,
            "entities": list(ENTITIES), "relations": list(relations), "values": list(values),
            "target_index": target, "target": ENTITIES[target],
            "donor_indices": {"relation": target_index(query_relation ^ 1, query_value, binding_relation, binding_value),
                              "value": target_index(query_relation, query_value ^ 1, binding_relation, binding_value),
                              "double": target_index(query_relation ^ 1, query_value ^ 1, binding_relation, binding_value)},
            "prompt_ids": ids, "prompt": tokenizer.decode(ids), "regions": regions,
            "fact_cells": cells, "answer_boundary_token": len(ids) - 1}


def compile_material(tokenizer) -> list[dict]:
    rows = []
    for family_id in range(32):
        for binding_relation in (0, 1):
            for binding_value in (0, 1):
                for relation_form in ("natural", "nonce"):
                    for value_form in ("natural", "nonce"):
                        for query_relation in (0, 1):
                            for query_value in (0, 1):
                                for ablation in ("full", "relation_missing", "value_missing", "both_missing"):
                                    rows.append(compile_row(tokenizer, family_id=family_id,
                                        binding_relation=binding_relation, binding_value=binding_value,
                                        relation_form=relation_form, value_form=value_form,
                                        query_relation=query_relation, query_value=query_value, ablation=ablation))
    return rows


def score_candidates(model, tokenizer, rows: list[dict], batch_size: int = 24) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    jobs = []
    for row in rows:
        for candidate_index, entity in enumerate(row["entities"]):
            continuation = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
            jobs.append({"row": row, "candidate_index": candidate_index, "continuation": continuation,
                         "sequence": row["prompt_ids"] + continuation})
    buckets: dict[int, list[dict]] = defaultdict(list)
    for job in jobs:
        buckets[len(job["sequence"])].append(job)
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    batches = [values[start:start + batch_size] for _, values in sorted(buckets.items())
               for start in range(0, len(values), batch_size)]
    done = 0
    for batch in batches:
        ids, mask, shifts = p2552.left_pad([job["sequence"] for job in batch], tokenizer.pad_token_id, device)
        if any(shifts) or not bool(torch.all(mask == 1)):
            raise RuntimeError("exact-length no-padding contract violated")
        keep = max(len(job["continuation"]) for job in batch) + 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits
        logit_offset = int(ids.shape[1] - logits.shape[1])
        for batch_index, job in enumerate(batch):
            first = len(job["row"]["prompt_ids"]) - 1 - logit_offset
            value = 0.0
            for offset, token in enumerate(job["continuation"]):
                z = logits[batch_index, first + offset].float()
                value += float(z[token] - torch.logsumexp(z, dim=-1))
            scores[job["row"]["case_id"]][job["candidate_index"]] = value
        done += len(batch)
        if done % 4096 == 0 or done == len(jobs):
            print(f"[phase2580 score] {done}/{len(jobs)}", flush=True)
    output = []
    for row in rows:
        values = scores[row["case_id"]]
        prediction = max(values, key=values.get)
        wrong_values = [value for index, value in values.items() if index != row["target_index"]]
        output.append({key: row[key] for key in ("case_id", "base_case_id", "ablation", "family_id", "family",
            "binding_id", "binding_relation", "binding_value", "relation_form", "value_form",
            "query_relation", "query_value", "target_index", "donor_indices")})
        output[-1].update({"prediction_index": prediction, "correct": prediction == row["target_index"],
                           "target_score": values[row["target_index"]],
                           "target_minus_best_wrong": values[row["target_index"]] - max(wrong_values),
                           "scores": {str(index): value for index, value in values.items()}})
    return output


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
                 for family in range(32)}
    correct = {row["case_id"]: row["correct"] for row in full}
    material_keys = {(row["family_id"], row["binding_relation"], row["binding_value"],
                      row["relation_form"], row["value_form"], row["query_relation"], row["query_value"]): row
                     for row in full}
    eligible = []
    for prefix in sorted({key[:5] for key in material_keys}):
        cells = [material_keys[prefix + (r, v)] for r in (0, 1) for v in (0, 1)]
        if all(correct[row["case_id"]] for row in cells):
            eligible.append(prefix)
    target_counts = {str(index): sum(row["target_index"] == index for row in full) for index in range(4)}
    summary = {"conditions": conditions, "full_by_form": by_form, "full_by_query": by_query,
               "full_by_family": by_family, "target_counts": target_counts,
               "eligible_correct_quartets": len(eligible),
               "eligible_by_form": {f"r={rf},v={vf}": sum(key[3:] == (rf, vf) for key in eligible)
                                    for rf in ("natural", "nonce") for vf in ("natural", "nonce")}}
    return summary, eligible


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 四选一关系×值条件代数的Qwen3-4B行为门（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 将旧二选一XOR升级为四实体二维身份。每个binding由$b_r,b_v$两bit定义，查询关系$r$和值$v$唯一选择

$$e^*(r,v;b_r,b_v)=2(r\oplus b_r)+(v\oplus b_v).$$

于是$00/10/01/11$四格对应四个不同实体，relation donor、value donor、double donor和随机损伤可被区分。材料覆盖32族、4个binding、自然/nonce关系、自然/nonce值和4种查询，共2048个full case；每格另做relation missing、value missing、both missing，共8192 case、32768条完整多token候选评分。候选目标严格四等分；按`prompt+candidate`完整长度分桶，mask全1、padding严格为0。模型为Qwen3-4B BF16 CUDA非量化。

**结果汇总。** `{json.dumps(result['summary'], ensure_ascii=False)}`。行为裁决`{json.dumps(result['adjudication'], ensure_ascii=False)}`；设计、哈希和检查为`{json.dumps(result['design'], ensure_ascii=False)}`、`{json.dumps(result['hashes'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2580_c356609_c364800_fourchoice_relation_value_behavior.py`；token级材料、完整四候选分数、eligible四元组与final位于`{OUT}`。

**分析与理论进展。** 行为通过只说明模型能在四选一表格中联合使用两个条件。四个donor身份不再退化成“任何错误都是flip”，为后续交互出生和因果特异性提供了更严格仪器。natural/nonce四格各自报告，family专属nonce描述符避免旧实验所有family复用同一对nonce词的伪稳定。

**问题硬伤与结论。** R0/R1、V0/V1仍是显式脚手架；四实体名称固定；只用英文和单surface；missing占位符属于分布外；候选likelihood不是自主开放生成。行为失败不否定语言机制，只决定该模型能否进入这套显微镜；行为成功也不是内部机制证据。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2579)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        material = compile_material(tokenizer)
        behavior = score_candidates(model, tokenizer, material, batch_size=32)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    material_path = OUT / "material/fourchoice_token_atomic.jsonl"
    behavior_path = OUT / "behavior/fourchoice_scores.jsonl"
    full_material_path = OUT / "material/fourchoice_full.jsonl"
    write(material_path, material)
    write(full_material_path, [row for row in material if row["ablation"] == "full"])
    write(behavior_path, behavior)
    summary, eligible = summarize(behavior)
    save(OUT / "material/eligible_quartets.json", {"prefix_fields": ["family_id", "binding_relation",
         "binding_value", "relation_form", "value_form"], "eligible": eligible})
    design = {"families": 32, "bindings": 4, "forms": 4, "queries": 4, "conditions": 4,
              "full_cases": 2048, "all_cases": len(material), "candidate_sequences": len(material) * 4,
              "entities": list(ENTITIES), "target_balance": summary["target_counts"],
              "max_prompt_tokens": max(len(row["prompt_ids"]) for row in material)}
    # Hiding one binary factor leaves two of four entities indistinguishable,
    # so calibrated chance is 1/2; hiding both factors gives four-way chance.
    single_missing = ("relation_missing", "value_missing")
    gate = (summary["conditions"]["full"]["accuracy"] >= .80 and
            all(summary["conditions"][name]["accuracy"] <= .55 for name in single_missing) and
            summary["conditions"]["both_missing"]["accuracy"] <= .30 and
            all(value >= .70 for value in summary["full_by_form"].values()))
    adjudication = {"behavior_qualified": gate, "full_at_least_080": summary["conditions"]["full"]["accuracy"] >= .80,
                    "single_missing_at_most_055": {name: summary["conditions"][name]["accuracy"] <= .55
                        for name in single_missing},
                    "both_missing_at_most_030": summary["conditions"]["both_missing"]["accuracy"] <= .30,
                    "four_forms_each_at_least_070": {name: value >= .70 for name, value in summary["full_by_form"].items()},
                    "eligible_at_least_64": len(eligible) >= 64}
    token_atomic = all(sorted(position for positions in row["regions"].values() for position in positions)
                       == list(range(len(row["prompt_ids"]))) for row in material)
    checks = {"phase2579_complete": prior["all_checks_passed"], "all_8192_cases": len(material) == 8192,
              "all_32768_candidates": len(material) * 4 == 32768, "all_scores_present": len(behavior) == len(material),
              "target_exactly_balanced": len(set(summary["target_counts"].values())) == 1,
              "four_donor_identities_distinct": all(len({row["target_index"], *row["donor_indices"].values()}) == 4
                                                     for row in material),
              "token_atomic": token_atomic, "exact_length_no_padding": True,
              "scientific_gate_does_not_abort_pipeline": True, "claim_boundary": True}
    hashes = {"material": sha256(material_path), "full_material": sha256(full_material_path),
              "behavior": sha256(behavior_path)}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "design": design, "summary": summary,
              "adjudication": adjudication, "hashes": hashes, "checks": checks,
              "all_checks_passed": all(checks.values()), "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
