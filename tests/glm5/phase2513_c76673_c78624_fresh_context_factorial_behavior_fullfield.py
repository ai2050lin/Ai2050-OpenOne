#!/usr/bin/env python3
"""Fresh fully crossed context factors for event-conditioned relation selection."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2513_c76673_c78624_fresh_context_factorial_behavior_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2513, "C76673-C78624", 2560
UNITS = (28, 29)
PAIRS = (
    ("taxonomy", "part_whole"), ("product", "causal"), ("temporal", "spatial"),
    ("role", "preference"), ("membership", "translation"), ("coreference", "punctuation"),
)
MARKERS = {28: {"en": ("navic", "pelor"), "zh": ("辰符", "巽符")},
           29: {"en": ("tebir", "wamic"), "zh": ("离符", "坎符")}}
EN_NAMES = {
    28: ("Aldren", "Bovik", "Cyren", "Dalis", "Eron", "Favil", "Goran", "Helix", "Ivera", "Jomar"),
    29: ("Kalen", "Loric", "Mavin", "Neris", "Orlan", "Pavid", "Quorin", "Ravis", "Selan", "Torik"),
}
ZH_NAMES = {
    28: ("安澈", "碧岑", "沧岭", "澹川", "峨亭", "风浦", "桂汀", "鹤洲", "霁原", "柯岸"),
    29: ("岚溪", "明屿", "宁川", "瓯野", "朴舟", "清禾", "若谷", "松澜", "桐原", "微岫"),
}
EVENTS = ("definition_end", "facts_end", "query_marker", "candidate0", "candidate1", "answer_boundary")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2500_c64001_c65152_semantic_necessity_2x2_behavior as base  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def occurrence_spans(tokenizer, prompt: str, text: str) -> list[list[int]]:
    spans, start = [], 0
    while True:
        pos = prompt.find(text, start)
        if pos < 0:
            break
        spans.append([len(tokenizer.encode(prompt[:pos], add_special_tokens=False)),
                      len(tokenizer.encode(prompt[:pos + len(text)], add_special_tokens=False))])
        start = pos + len(text)
    return spans


def compile_rows(tokenizer) -> list[dict]:
    rows, case = [], 76673
    for unit in UNITS:
        for pair_id, pair in enumerate(PAIRS):
            for language in ("en", "zh"):
                names0 = EN_NAMES[unit] if language == "en" else ZH_NAMES[unit]
                shift = (unit + pair_id * 3) % len(names0)
                names = names0[shift:] + names0[:shift]
                source, target0, target1 = names[:3]
                marker0, marker1 = MARKERS[unit][language]
                assert len(tokenizer.encode(marker0, add_special_tokens=False)) == len(tokenizer.encode(marker1, add_special_tokens=False))
                for paraphrase in (0, 1):
                    relations = []
                    for family, target in zip(pair, (target0, target1)):
                        fact_template, description = base.RELATIONS[family][language][paraphrase]
                        relations.append({"family": family, "target": target,
                                          "fact": fact_template.format(s=source, t=target), "description": description})
                    for fact_order_bit in (0, 1):
                        fact_order = (0, 1) if fact_order_bit == 0 else (1, 0)
                        for definition_order_bit in (0, 1):
                            definition_order = (0, 1) if definition_order_bit == 0 else (1, 0)
                            for candidate_order_bit in (0, 1):
                                candidates = [target0, target1] if candidate_order_bit == 0 else [target1, target0]
                                context_id = paraphrase * 8 + fact_order_bit * 4 + definition_order_bit * 2 + candidate_order_bit
                                for meaning_swap in (0, 1):
                                    mapping = (0, 1) if meaning_swap == 0 else (1, 0)
                                    definitions = {0: relations[mapping[0]]["description"],
                                                   1: relations[mapping[1]]["description"]}
                                    if language == "en":
                                        definition_parts = [f"Marker '{(marker0, marker1)[mi]}' means \"{definitions[mi]}\"."
                                                            for mi in definition_order]
                                        fact_parts = [f"Fact {rank + 1}: {relations[ri]['fact']}." for rank, ri in enumerate(fact_order)]
                                        prefix = "Definitions: " + " ".join(definition_parts) + " End definitions. Facts: " + " ".join(fact_parts) + " End facts. "
                                    else:
                                        definition_parts = [f"标记“{(marker0, marker1)[mi]}”表示“{definitions[mi]}”。" for mi in definition_order]
                                        fact_parts = [f"事实{rank + 1}：{relations[ri]['fact']}。" for rank, ri in enumerate(fact_order)]
                                        prefix = "定义：" + "".join(definition_parts) + "定义结束。事实：" + "".join(fact_parts) + "事实结束。"
                                    for query_marker in (0, 1):
                                        marker = (marker0, marker1)[query_marker]
                                        selected_relation = mapping[query_marker]
                                        target = relations[selected_relation]["target"]
                                        if language == "en":
                                            prompt = (prefix + f"Query: for source {source}, follow marker '{marker}'. Which candidate satisfies that marker's defined meaning?\n"
                                                      f"Candidates: [ {candidates[0]} ] [ {candidates[1]} ]\nReturn exactly one candidate name.\nAnswer:")
                                        else:
                                            prompt = (prefix + f"查询：对于来源{source}，请遵循标记“{marker}”。哪个候选符合该标记所定义的含义？\n"
                                                      f"候选：[ {candidates[0]} ] [ {candidates[1]} ]\n只返回一个候选名称。\n答案：")
                                        ids = [int(v) for v in tokenizer.encode(prompt, add_special_tokens=False)]
                                        anchors = {"definition_end": "End definitions" if language == "en" else "定义结束",
                                                   "facts_end": "End facts" if language == "en" else "事实结束",
                                                   "query_marker": marker, "candidate0": candidates[0], "candidate1": candidates[1]}
                                        spans = {key: occurrence_spans(tokenizer, prompt, value) for key, value in anchors.items()}
                                        rows.append({"case_id": f"c{case:05d}-p{pair_id}-u{unit}-{language}-x{context_id}-m{meaning_swap}-q{query_marker}",
                                                     "unit": unit, "pair_id": pair_id, "families": list(pair), "language": language,
                                                     "paraphrase": paraphrase, "fact_order": fact_order_bit,
                                                     "definition_order": definition_order_bit, "candidate_order": candidate_order_bit,
                                                     "context_id": context_id, "surface": context_id, "meaning_swap": meaning_swap,
                                                     "query_marker": query_marker, "markers": [marker0, marker1], "marker": marker,
                                                     "source": source, "relation_targets": [target0, target1],
                                                     "selected_relation": selected_relation, "target": target,
                                                     "candidates": candidates, "prompt": prompt, "prompt_ids": ids,
                                                     "spans": spans, "answer_boundary_token": len(ids) - 1})
                                        case += 1
    return rows


def design_audit(rows: list[dict]) -> dict:
    lookup = {(r["unit"], r["pair_id"], r["language"], r["context_id"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    length_equal, prefix_equal, flip, bag, candidate_bag = [], [], [], [], []
    for unit in UNITS:
        for pair_id in range(len(PAIRS)):
            for language in ("en", "zh"):
                for context_id in range(16):
                    for meaning_swap in (0, 1):
                        a, b = (lookup[(unit, pair_id, language, context_id, meaning_swap, q)] for q in (0, 1))
                        end = a["spans"]["facts_end"][0][1]
                        length_equal.append(len(a["prompt_ids"]) == len(b["prompt_ids"]))
                        prefix_equal.append(a["prompt_ids"][:end] == b["prompt_ids"][:end])
                    for query_marker in (0, 1):
                        a, b = (lookup[(unit, pair_id, language, context_id, m, query_marker)] for m in (0, 1))
                        flip.append(a["target"] != b["target"])
                        bag.append(Counter(a["prompt_ids"]) == Counter(b["prompt_ids"]))
                    if context_id % 2 == 0:
                        a = lookup[(unit, pair_id, language, context_id, 0, 0)]
                        b = lookup[(unit, pair_id, language, context_id + 1, 0, 0)]
                        candidate_bag.append(Counter(a["prompt_ids"]) == Counter(b["prompt_ids"]))
    return {"rows": len(rows), "full_prompt_length_equal_across_query_rate": float(np.mean(length_equal)),
            "prefix_token_equal_rate": float(np.mean(prefix_equal)), "answer_flip_rate": float(np.mean(flip)),
            "definition_swap_token_multiset_equal_rate": float(np.mean(bag)),
            "candidate_order_token_multiset_equal_rate": float(np.mean(candidate_bag)),
            "factor_counts": {key: dict(Counter(str(r[key]) for r in rows)) for key in
                              ("unit", "pair_id", "language", "paraphrase", "fact_order", "definition_order", "candidate_order", "meaning_swap", "query_marker")}}


def behavior_summary(rows: list[dict], generated: list[dict]) -> dict:
    by_id = {r["case_id"]: r for r in generated}
    detail, qualified = {}, []
    for pair_id, pair in enumerate(PAIRS):
        detail[str(pair_id)] = {}
        gates = []
        for unit in UNITS:
            cases = [r for r in rows if r["pair_id"] == pair_id and r["unit"] == unit]
            values = [by_id[r["case_id"]] for r in cases]
            lang = {language: float(np.mean([v["parsed_correct"] for v in values if v["language"] == language])) for language in ("en", "zh")}
            swap = {str(m): float(np.mean([v["parsed_correct"] for v in values if v["meaning_swap"] == m])) for m in (0, 1)}
            cand = {str(c): float(np.mean([v["parsed_correct"] for v in values if v["candidate_order"] == c])) for c in (0, 1)}
            lookup = {(r["language"], r["context_id"], r["meaning_swap"], r["query_marker"]): by_id[r["case_id"]] for r in cases}
            both = [lookup[(language, context, 0, q)]["parsed_correct"] and lookup[(language, context, 1, q)]["parsed_correct"]
                    for language in ("en", "zh") for context in range(16) for q in (0, 1)]
            panel = {"rows": len(cases), "accuracy": float(np.mean([v["parsed_correct"] for v in values])),
                     "language_accuracy": lang, "meaning_swap_accuracy": swap, "candidate_order_accuracy": cand,
                     "paired_flip_both_correct_rate": float(np.mean(both))}
            panel["gate"] = (panel["accuracy"] >= .75 and min(lang.values()) >= .625 and min(swap.values()) >= .625
                             and min(cand.values()) >= .625 and panel["paired_flip_both_correct_rate"] >= .625)
            detail[str(pair_id)][str(unit)] = panel
            gates.append(panel["gate"])
        if all(gates):
            qualified.append(pair_id)
    return {"aggregate_accuracy": {str(unit): float(np.mean([r["parsed_correct"] for r in generated if r["unit"] == unit])) for unit in UNITS},
            "qualified_pair_ids": qualified, "qualified_pairs": [list(PAIRS[p]) for p in qualified], "detail": detail}


def event_positions(row: dict) -> list[int]:
    return [row["spans"]["definition_end"][0][1] - 1, row["spans"]["facts_end"][0][1] - 1,
            row["spans"]["query_marker"][-1][1] - 1, row["spans"]["candidate0"][-1][1] - 1,
            row["spans"]["candidate1"][-1][1] - 1, row["answer_boundary_token"]]


def capture(model, rows: list[dict], qualified: set[int], behavior_map: dict[str, dict]) -> dict:
    selected = [r for r in rows if r["pair_id"] in qualified]
    qmods = field_utils.modules(model)
    raw = OUT / "raw"; raw.mkdir(parents=True, exist_ok=True)
    event_path = raw / "fresh_factorial_sixevent_allqpoint.float16.npy"
    event_field = np.lib.format.open_memmap(event_path, mode="w+", dtype=np.float16,
                                            shape=(len(selected), len(EVENTS), len(qmods), DIM))
    representative = [r for r in selected if r["paraphrase"] == r["fact_order"] == r["definition_order"] == r["candidate_order"] == 0
                      and r["query_marker"] == 0]
    total_tokens = sum(len(r["prompt_ids"]) for r in representative)
    token_path = raw / "fresh_factorial_representative_alltoken_allqpoint.float16.npy"
    token_field = np.lib.format.open_memmap(token_path, mode="w+", dtype=np.float16,
                                            shape=(total_tokens, len(qmods), DIM))
    offsets, cursor = {}, 0
    for row in representative:
        offsets[row["case_id"]] = (cursor, cursor + len(row["prompt_ids"])); cursor += len(row["prompt_ids"])
    captures, handles = {}, []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device, index = model.get_input_embeddings().weight.device, []
    try:
        with torch.inference_mode():
            for model_row, row in enumerate(selected):
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                captures.clear(); model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                positions = event_positions(row)
                for qpoint in range(len(qmods)):
                    tensor = captures[qpoint][0]
                    event_field[model_row, :, qpoint] = tensor[positions].float().cpu().numpy().astype(np.float16)
                    if row["case_id"] in offsets:
                        lo, hi = offsets[row["case_id"]]
                        token_field[lo:hi, qpoint] = tensor.float().cpu().numpy().astype(np.float16)
                index.append({**{k: row[k] for k in ("case_id", "unit", "pair_id", "families", "language", "paraphrase", "fact_order",
                                                             "definition_order", "candidate_order", "context_id", "meaning_swap", "query_marker",
                                                             "selected_relation", "target", "candidates", "prompt_ids")},
                              "model_row": model_row, "events": list(EVENTS), "event_positions": positions,
                              "behavior_correct": behavior_map[row["case_id"]]["parsed_correct"],
                              "alltoken_offset": list(offsets[row["case_id"]]) if row["case_id"] in offsets else None})
                if (model_row + 1) % 64 == 0:
                    event_field.flush(); token_field.flush(); print(f"[phase2513 field] {model_row + 1}/{len(selected)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        event_field.flush(); token_field.flush(); del event_field, token_field
    index_path = OUT / "index/field_rows.jsonl"; write_jsonl(index_path, index)
    token_index_path = OUT / "index/alltoken_rows.jsonl"
    write_jsonl(token_index_path, [{"case_id": r["case_id"], "offset": list(offsets[r["case_id"]]), "prompt_ids": r["prompt_ids"]}
                                   for r in representative])
    return {"event_field": str(event_path), "event_shape": [len(selected), len(EVENTS), len(qmods), DIM],
            "event_index": str(index_path), "events": list(EVENTS), "alltoken_field": str(token_path),
            "alltoken_shape": [total_tokens, len(qmods), DIM], "alltoken_rows": len(representative),
            "alltoken_index": str(token_index_path),
            "sha256": {event_path.name: digest(event_path), token_path.name: digest(token_path)}}


def prefix_control(collection: dict) -> dict:
    field = np.load(collection["event_field"], mmap_mode="r")
    rows = [json.loads(line) for line in Path(collection["event_index"]).read_text(encoding="utf-8").splitlines()]
    lookup = {(r["unit"], r["pair_id"], r["language"], r["context_id"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    result = {}
    for event_index, event in enumerate(EVENTS[:2]):
        maximum = 0.0
        for unit in UNITS:
            for pair_id in sorted({r["pair_id"] for r in rows}):
                for language in ("en", "zh"):
                    for context in range(16):
                        cells = {(m, q): np.asarray(field[lookup[(unit, pair_id, language, context, m, q)]["model_row"], event_index], dtype=np.float32)
                                 for m in (0, 1) for q in (0, 1)}
                        interaction = (cells[(0, 0)] - cells[(0, 1)] - cells[(1, 0)] + cells[(1, 1)]) / 4
                        maximum = max(maximum, float(np.abs(interaction).max()))
        result[event] = {"max_abs_all_qpoint_coordinate": maximum, "all_exact_zero": maximum == 0.0}
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 六关系对双新unit的上下文全析因行为门与全坐标场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 针对旧surface同时改变复述、事实顺序、定义顺序和候选顺序的混杂，建立unit28/29全新实体与等token长度nonce marker。六关系对分别全交叉语言2×谓词复述2×事实顺序2×定义顺序2×候选顺序2×meaning-swap2×query-marker2，每unit 768条、总1536条；先做真实贪心行为门，仅两个unit均满足总准确率、语言、swap、候选顺序和paired-flip门的关系对进入六事件×38qpoint×2560全坐标采集。候选事件始终按物理位置candidate0/1记录，禁止按正确答案动态选位置。

$$N=2\times6\times2^7=1536,\qquad c=(\lambda,p,f,d,o).$$

**结果汇总。** 设计 `{json.dumps(result['design_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；prefix负对照 `{json.dumps(result['prefix_control'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2513_c76673_c78624_fresh_context_factorial_behavior_fullfield.py`；1536条材料和生成、合格关系六事件原场、代表全token原场、索引、哈希与final位于`{OUT}`。

**分析与理论进展。** 本Phase不寻找最佳模型，只建立可辨识材料。相同prompt token多重集下分别交换definition绑定和candidate顺序，使下一Phase可以区分语言、复述、事实/定义顺序与候选位置是否真正改变query→target的逐坐标映射。全场保留低值坐标，不以Top-K代替分析。

**问题硬伤与结论。** 即使行为门和prefix零通过，材料仍是短合成二候选；各关系对共用模板骨架；float16落盘只适合中等以上场差；行为失败的pair不能用于内部机制裁决。下一Phase只能在unit28选择模型并对unit29锁箱，不能按unit29重新挑层或筛单行。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows = compile_rows(tokenizer); write_jsonl(OUT / "material/factorial_rows.jsonl", rows)
        generated = base.behavior(model, tokenizer, rows)
        # Add crossed factors required by the new gate.
        factors = {r["case_id"]: r for r in rows}
        for row in generated:
            src = factors[row["case_id"]]
            for key in ("paraphrase", "fact_order", "definition_order", "candidate_order", "context_id"):
                row[key] = src[key]
        write_jsonl(OUT / "behavior/autonomous_generation.jsonl", generated)
        behavior = behavior_summary(rows, generated)
        qualified = set(behavior["qualified_pair_ids"])
        collection = capture(model, rows, qualified, {r["case_id"]: r for r in generated})
    finally:
        model_utils.release_model(model); gc.collect()
    audit = design_audit(rows); prefix = prefix_control(collection)
    checks = {"rows_1536": len(rows) == 1536, "balanced_full_factorial": all(len(v) == 2 or k in ("unit", "pair_id")
                                                                                 for k, v in audit["factor_counts"].items()),
              "query_length_equal": audit["full_prompt_length_equal_across_query_rate"] == 1.0,
              "prefix_equal": audit["prefix_token_equal_rate"] == 1.0, "answer_flip": audit["answer_flip_rate"] == 1.0,
              "definition_multiset": audit["definition_swap_token_multiset_equal_rate"] == 1.0,
              "candidate_multiset": audit["candidate_order_token_multiset_equal_rate"] == 1.0,
              "at_least_three_qualified_pairs": len(qualified) >= 3,
              "field_complete": collection["event_shape"] == [len(qualified) * 256, 6, 38, DIM],
              "prefix_exact_zero_all_qpoints": all(v["all_exact_zero"] for v in prefix.values()),
              "hashes": len(collection["sha256"]) == 2, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA",
              "design_audit": audit, "behavior": behavior, "collection": collection, "prefix_control": prefix,
              "adjudication": {"fresh_factorial_contract_valid": all(prefix[e]["all_exact_zero"] for e in prefix),
                               "event_compiler_identified": False, "causal_mediator_identified": False,
                               "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "behavior": behavior, "collection": collection, "prefix": prefix,
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
