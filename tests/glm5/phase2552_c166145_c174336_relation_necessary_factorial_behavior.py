#!/usr/bin/env python3
"""Large relation-necessary 2x2x2 factorial language-family behavior gate."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2552_c166145_c174336_relation_necessary_factorial_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PRIOR = RESULT / "phase2551_c162049_c166144_relation_degeneracy_audit_contract/analysis/final.json"
PHASE, CAMPAIGN = 2552, "C166145-C174336"
REGIONS = (
    "frame", "facts_entity", "facts_relation", "facts_value", "query_context",
    "query_relation", "query_value", "candidate", "instruction", "answer_boundary",
)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as old_atlas  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())


def add(tokenizer, ids: list[int], regions: dict[str, list[int]], region: str, text: str) -> list[int]:
    tokens = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    if not tokens:
        raise RuntimeError((region, text))
    start = len(ids)
    ids.extend(tokens)
    positions = list(range(start, len(ids)))
    regions.setdefault(region, []).extend(positions)
    return positions


def relation_pair(family_id: int, language: str, form: str) -> tuple[str, str]:
    if form == "nonce":
        return ("[relation dax]", "[relation wug]") if language == "en" else ("[关系达克斯]", "[关系沃格]")
    partner = (family_id + 11) % len(old_atlas.OPERATIONS)
    op0, op1 = old_atlas.OPERATIONS[family_id], old_atlas.OPERATIONS[partner]
    return (op0[1], op1[1]) if language == "en" else (op0[2], op1[2])


def value_pair(family_id: int, language: str, form: str) -> tuple[str, str]:
    if form == "nonce":
        return ("kivora", "mexalu") if language == "en" else ("奇沃拉", "梅萨鲁")
    operation = old_atlas.OPERATIONS[family_id]
    return tuple(operation[3] if language == "en" else operation[4])


def compile_row(
    tokenizer,
    *,
    unit: int,
    family_id: int,
    language: str,
    surface: int,
    binding: int,
    relation_form: str,
    value_form: str,
    query_relation: int,
    query_value: int,
    ablation: str = "full",
) -> dict:
    entities = old_atlas.NAMES[unit][language]
    relations = relation_pair(family_id, language, relation_form)
    values = value_pair(family_id, language, value_form)
    ids: list[int] = []
    regions: dict[str, list[int]] = {name: [] for name in REGIONS}
    cells: list[dict] = []
    add(tokenizer, ids, regions, "frame", "Facts:\n" if language == "en" else "事实：\n")
    order = [(e, r) for e in range(2) for r in range(2)]
    if surface:
        order = [(1, 1), (0, 1), (1, 0), (0, 0)]
    for entity_index, relation_index in order:
        value_index = entity_index ^ relation_index ^ binding
        if language == "en":
            add(tokenizer, ids, regions, "frame", "Entity ")
            ep = add(tokenizer, ids, regions, "facts_entity", f"[{entities[entity_index]}]")
            add(tokenizer, ids, regions, "frame", " under relation ")
            rp = add(tokenizer, ids, regions, "facts_relation", relations[relation_index])
            add(tokenizer, ids, regions, "frame", " has value ")
            vp = add(tokenizer, ids, regions, "facts_value", f"[{values[value_index]}]")
            add(tokenizer, ids, regions, "frame", ".\n")
        else:
            add(tokenizer, ids, regions, "frame", "实体")
            ep = add(tokenizer, ids, regions, "facts_entity", f"[{entities[entity_index]}]")
            add(tokenizer, ids, regions, "frame", "在关系")
            rp = add(tokenizer, ids, regions, "facts_relation", relations[relation_index])
            add(tokenizer, ids, regions, "frame", "下具有值")
            vp = add(tokenizer, ids, regions, "facts_value", f"[{values[value_index]}]")
            add(tokenizer, ids, regions, "frame", "。\n")
        cells.append({"entity_index": entity_index, "relation_index": relation_index,
                      "value_index": value_index, "entity_positions": ep,
                      "relation_positions": rp, "value_positions": vp})
    shown_relation = relations[query_relation] if ablation not in ("relation", "both") else (
        "[relation unavailable]" if language == "en" else "[关系未知]"
    )
    shown_value = values[query_value] if ablation not in ("value", "both") else (
        "[value unavailable]" if language == "en" else "[值未知]"
    )
    if language == "en":
        context = "Question: which entity has, jointly, relation " if surface == 0 else "Query: identify the entity matching both relation "
        add(tokenizer, ids, regions, "query_context", context)
        add(tokenizer, ids, regions, "query_relation", shown_relation)
        add(tokenizer, ids, regions, "query_context", " and value ")
        add(tokenizer, ids, regions, "query_value", f"[{shown_value}]" if ablation in ("value", "both") else f"[{values[query_value]}]")
        add(tokenizer, ids, regions, "frame", ".\nCandidates: ")
        add(tokenizer, ids, regions, "candidate", f"[{entities[0]}] or [{entities[1]}]")
        add(tokenizer, ids, regions, "instruction", ". Return only the complete entity name. Answer")
    else:
        context = "问题：哪个实体同时匹配关系" if surface == 0 else "查询：找出同时符合下列关系和值的实体，关系"
        add(tokenizer, ids, regions, "query_context", context)
        add(tokenizer, ids, regions, "query_relation", shown_relation)
        add(tokenizer, ids, regions, "query_context", "和值")
        add(tokenizer, ids, regions, "query_value", f"[{shown_value}]" if ablation in ("value", "both") else f"[{values[query_value]}]")
        add(tokenizer, ids, regions, "frame", "。\n候选：")
        add(tokenizer, ids, regions, "candidate", f"[{entities[0]}]或[{entities[1]}]")
        add(tokenizer, ids, regions, "instruction", "。只返回完整实体名称。答案")
    add(tokenizer, ids, regions, "answer_boundary", ":")
    target_index = query_value ^ query_relation ^ binding
    key = f"u{unit}_f{family_id:02d}_{language}_s{surface}_b{binding}_r{relation_form}_v{value_form}_qr{query_relation}_qv{query_value}"
    return {
        "case_id": key if ablation == "full" else f"{key}_abl_{ablation}",
        "base_case_id": key,
        "ablation": ablation,
        "unit": unit,
        "family_id": family_id,
        "family": old_atlas.OPERATIONS[family_id][0],
        "partner_family_id": (family_id + 11) % len(old_atlas.OPERATIONS),
        "language": language,
        "surface": surface,
        "binding": binding,
        "relation_form": relation_form,
        "value_form": value_form,
        "query_relation": query_relation,
        "query_value": query_value,
        "entities": list(entities),
        "relations": list(relations),
        "values": list(values),
        "target_index": target_index,
        "target": entities[target_index],
        "prompt_ids": ids,
        "prompt": tokenizer.decode(ids),
        "regions": regions,
        "fact_cells": cells,
        "answer_boundary_token": len(ids) - 1,
    }


def compile_material(tokenizer, ablations: tuple[str, ...] = ("full",)) -> list[dict]:
    rows = []
    for unit in (34, 35):
        for family_id in range(len(old_atlas.OPERATIONS)):
            for language in ("en", "zh"):
                for surface in (0, 1):
                    for binding in (0, 1):
                        for relation_form in ("natural", "nonce"):
                            for value_form in ("natural", "nonce"):
                                for query_relation in (0, 1):
                                    for query_value in (0, 1):
                                        for ablation in ablations:
                                            rows.append(compile_row(
                                                tokenizer, unit=unit, family_id=family_id, language=language,
                                                surface=surface, binding=binding, relation_form=relation_form,
                                                value_form=value_form, query_relation=query_relation,
                                                query_value=query_value, ablation=ablation,
                                            ))
    return rows


def left_pad(sequences: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    shifts = []
    for index, sequence in enumerate(sequences):
        shift = width - len(sequence)
        shifts.append(shift)
        ids[index, shift:] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, shift:] = 1
    return ids, mask, shifts


def score_candidates(model, tokenizer, rows: list[dict], batch_size: int = 24) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    jobs = []
    for row in rows:
        for candidate_index, entity in enumerate(row["entities"]):
            prefix = " " if row["language"] == "en" else ""
            continuation = [int(token) for token in tokenizer.encode(prefix + entity, add_special_tokens=False)]
            jobs.append({"row": row, "candidate_index": candidate_index, "entity": entity,
                         "continuation": continuation, "sequence": row["prompt_ids"] + continuation})
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        ids, mask, shifts = left_pad([job["sequence"] for job in batch], tokenizer.pad_token_id, device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
        for batch_index, (job, shift) in enumerate(zip(batch, shifts)):
            prompt_length = len(job["row"]["prompt_ids"])
            value = sum(float(log_probs[batch_index, shift + prompt_length - 1 + offset, token])
                        for offset, token in enumerate(job["continuation"]))
            scores[job["row"]["case_id"]][job["candidate_index"]] = value
        done = start + len(batch)
        if done % 1536 == 0 or done == len(jobs):
            print(f"[phase2552 score] {done}/{len(jobs)}", flush=True)
    out = []
    for row in rows:
        row_scores = scores[row["case_id"]]
        prediction = max(row_scores, key=row_scores.get)
        target = row["target_index"]
        out.append({
            "case_id": row["case_id"], "base_case_id": row["base_case_id"], "ablation": row["ablation"],
            "unit": row["unit"], "family_id": row["family_id"], "family": row["family"],
            "language": row["language"], "surface": row["surface"], "binding": row["binding"],
            "relation_form": row["relation_form"], "value_form": row["value_form"],
            "query_relation": row["query_relation"], "query_value": row["query_value"],
            "target_index": target, "prediction_index": prediction, "correct": prediction == target,
            "target_minus_wrong": row_scores[target] - row_scores[1 - target],
            "scores": {str(key): value for key, value in row_scores.items()},
        })
    return out


def autonomous(model, tokenizer, rows: list[dict]) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    selected = [row for row in rows if row["unit"] == 35 and row["surface"] == 1 and row["binding"] == 0
                and row["query_relation"] == 1 and row["query_value"] == 0]
    output = []
    for start in range(0, len(selected), 8):
        batch = selected[start:start + 8]
        ids, mask, _ = left_pad([row["prompt_ids"] for row in batch], tokenizer.pad_token_id, device)
        width = ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=10, do_sample=False,
                                       use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
        for row, sequence in zip(batch, generated):
            text = tokenizer.decode(sequence[width:].cpu().tolist(), skip_special_tokens=True)
            hits = [index for index, entity in enumerate(row["entities"]) if norm(entity) in norm(text)]
            prediction = hits[0] if len(set(hits)) == 1 else None
            output.append({"case_id": row["case_id"], "family_id": row["family_id"],
                           "language": row["language"], "relation_form": row["relation_form"],
                           "value_form": row["value_form"], "target_index": row["target_index"],
                           "prediction_index": prediction, "generated": text,
                           "correct": prediction == row["target_index"]})
    return output


def summarize(rows: list[dict], generated: list[dict]) -> dict:
    by_condition = {}
    for condition in ("full", "relation", "value", "both"):
        subset = [row for row in rows if row["ablation"] == condition]
        by_condition[condition] = {
            "n": len(subset),
            "accuracy": float(np.mean([row["correct"] for row in subset])),
            "mean_margin": float(np.mean([row["target_minus_wrong"] for row in subset])),
        }
    full = [row for row in rows if row["ablation"] == "full"]
    by_forms = {}
    for relation_form in ("natural", "nonce"):
        for value_form in ("natural", "nonce"):
            subset = [row for row in full if row["relation_form"] == relation_form and row["value_form"] == value_form]
            by_forms[f"relation={relation_form},value={value_form}"] = float(np.mean([row["correct"] for row in subset]))
    by_language = {language: float(np.mean([row["correct"] for row in full if row["language"] == language]))
                   for language in ("en", "zh")}
    by_family = {str(family_id): float(np.mean([row["correct"] for row in full if row["family_id"] == family_id]))
                 for family_id in range(len(old_atlas.OPERATIONS))}
    return {
        "conditions": by_condition,
        "full_by_form": by_forms,
        "full_by_language": by_language,
        "full_by_family": by_family,
        "qualified_family_ids": [int(fid) for fid, accuracy in by_family.items() if accuracy >= .80],
        "autonomous_n": len(generated),
        "autonomous_accuracy": float(np.mean([row["correct"] for row in generated])),
    }


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 关系和值联合必要的32族全析因行为门（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 在Qwen3-4B BF16非量化CUDA上，把旧任务改成四事实交叉格。两个实体$e\in\{{0,1\}}$、两个关系$r\in\{{0,1\}}$、两个值$v\in\{{0,1\}}$满足

$$
v(e,r;b)=e\oplus r\oplus b,
\qquad e^*(r_q,v_q;b)=r_q\oplus v_q\oplus b.
$$

同一value在两个关系下对应不同实体，同一relation包含两个value，故relation与value任何一项缺失时答案都不可辨识。材料全交叉32族、英中、unit34/35、双surface、双binding、自然/无意义关系词、自然/无意义值以及四种query，共8192条full提示；再对同样8192条分别遮蔽query relation、query value或二者，共32768个行为case、65536个多token候选序列。另在独立unit35/surface1上对256条四种词面组合进行自主生成。

**token原子合同。** 每条实际input IDs由frame、四格facts实体/关系/值、query context、query relation、query value、候选、指令和answer boundary逐段编码；所有区域非空、互斥并穷尽。每个fact cell另保存实体/关系/value的具体token坐标，后续不靠字符反推。

**结果汇总。** `{json.dumps(result['summary'], ensure_ascii=False)}`。完整任务与遮蔽门的差值直接测量外部判定对relation/value的行为必要性；自然/nonce四格只裁决结构是否跨词面复用，不把nonce成功命名为语义理解。设计与检查为`{json.dumps(result['design'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2552_c166145_c174336_relation_necessary_factorial_behavior.py`；完整token材料、32768条候选结果、自主生成与final位于`{OUT}`。文件哈希为`{json.dumps(result['hashes'], ensure_ascii=False)}`。

**分析与理论进展。** 若full高而两个单因素遮蔽均回落到接近0.5，才证明新任务的答案函数确实需要relation-value合取。若自然与nonce关系词都通过，则强结果主要反映条件联结检索/变量绑定的通用结构，而不是32个关系词的既有语义；若仅自然材料通过，才有资格进一步查找词义先验的贡献。这个裁决修复了Phase2538中关系标签对答案函数退化的致命混杂。

**问题硬伤与结论。** 四格任务仍高度结构化；“未知”遮蔽可能引入分布外token；关系自然词与value的世界知识兼容性没有保证；candidate likelihood可利用格式。行为门不解释内部机制，只冻结允许进入Q/K/V、recipient和全坐标因果测试的family与词面条件。重要结论以全部8192个full case而非少量例句建立。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(PRIOR)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        full = compile_material(tokenizer, ("full",))
        all_rows = full + compile_material(tokenizer, ("relation", "value", "both"))
        behavior = score_candidates(model, tokenizer, all_rows)
        generated = autonomous(model, tokenizer, full)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
    material_path = OUT / "material/relation_necessary_token_atomic.jsonl"
    behavior_path = OUT / "behavior/factorial_candidate_scores.jsonl"
    autonomous_path = OUT / "behavior/autonomous_lockbox.jsonl"
    write(material_path, full)
    write(behavior_path, behavior)
    write(autonomous_path, generated)
    summary = summarize(behavior, generated)
    all_atomic = all(
        sorted(position for values in row["regions"].values() for position in values) == list(range(len(row["prompt_ids"])))
        and len({position for values in row["regions"].values() for position in values}) == len(row["prompt_ids"])
        and all(row["regions"][region] for region in REGIONS)
        and row["regions"]["answer_boundary"] == [len(row["prompt_ids"]) - 1]
        for row in full
    )
    design = {"families": 32, "full_rows": len(full), "all_behavior_rows": len(all_rows),
              "candidate_sequences": 2 * len(all_rows), "autonomous_rows": len(generated),
              "factors": {"unit": 2, "language": 2, "surface": 2, "binding": 2,
                          "relation_form": 2, "value_form": 2, "query_relation": 2, "query_value": 2},
              "max_prompt_tokens": max(len(row["prompt_ids"]) for row in all_rows)}
    hashes = {"material": sha(material_path), "behavior": sha(behavior_path), "autonomous": sha(autonomous_path)}
    checks = {
        "prior_passed": prior["all_checks_passed"],
        "rows_8192": len(full) == 8192,
        "all_conditions_32768": len(all_rows) == 32768,
        "token_atomic": bool(all_atomic),
        "full_accuracy_gate": summary["conditions"]["full"]["accuracy"] >= .80,
        "relation_is_behaviorally_necessary": summary["conditions"]["relation"]["accuracy"] <= .65,
        "value_is_behaviorally_necessary": summary["conditions"]["value"]["accuracy"] <= .65,
        "at_least_24_families": len(summary["qualified_family_ids"]) >= 24,
        "all_four_forms_measured": len(summary["full_by_form"]) == 4,
        "autonomous_measured": len(generated) == 256,
        "claim_boundary": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "design": design, "summary": summary,
              "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
