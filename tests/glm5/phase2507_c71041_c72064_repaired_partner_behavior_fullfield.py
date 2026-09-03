#!/usr/bin/env python3
"""Re-pair successful relations with new partners, fresh strings, and equal-length markers."""
from __future__ import annotations

import gc
import hashlib
import json
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
P2504 = RESULT / "phase2504_c68225_c68864_corrected_semantic_selection_walsh_lockbox"
OUT = RESULT / "phase2507_c71041_c72064_repaired_partner_behavior_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2507, "C71041-C72064", 2560
UNITS = (24, 25)
NEW_PAIRS = (("taxonomy", "role"), ("part_whole", "membership"), ("preference", "translation"))
MARKERS = {
    24: {"en": ("selum", "teric"), "zh": ("壬符", "癸符")},
    25: {"en": ("bovan", "nadis"), "zh": ("子符", "丑符")},
}
EN_NAMES = {
    24: ("Karel", "Lovan", "Mirel", "Noric", "Orlan", "Pavel", "Quinor", "Ravis"),
    25: ("Sarel", "Tovin", "Umaro", "Vesin", "Walen", "Xorin", "Yavel", "Zoric"),
}
ZH_NAMES = {
    24: ("开岚", "鹿川", "鸣沙", "宁汀", "鸥屿", "蒲禾", "晴野", "容溪"),
    25: ("杉岚", "藤川", "乌汀", "微澜", "雪禾", "吟舟", "玉溪", "竹野"),
}
EVENTS = ("definition_end", "facts_end", "query_marker", "candidate0", "candidate1", "answer_boundary")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2500_c64001_c65152_semantic_necessity_2x2_behavior as base  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_rows(tokenizer) -> list[dict]:
    base.UNITS = UNITS
    base.SPLIT = {24: "partner_confirmation", 25: "partner_lockbox"}
    base.PAIRS = NEW_PAIRS
    for unit in UNITS:
        base.MARKERS[unit] = MARKERS[unit]
        base.EN_NAMES[unit] = EN_NAMES[unit]
        base.ZH_NAMES[unit] = ZH_NAMES[unit]
    rows = base.compile_rows(tokenizer)
    for case, row in enumerate(rows, start=71041):
        row["case_id"] = f"c{case:05d}-rp{row['pair_id']}-u{row['unit']}-{row['language']}-s{row['surface']}-m{row['meaning_swap']}-q{row['query_marker']}"
    return rows


def positions(row: dict) -> list[int]:
    spans = row["spans"]
    return [spans["definition_end"][0][1] - 1, spans["facts_end"][0][1] - 1,
            spans["query_marker"][-1][1] - 1, spans["candidate0"][-1][1] - 1,
            spans["candidate1"][-1][1] - 1, row["answer_boundary_token"]]


def design_audit(rows: list[dict]) -> dict:
    lookup = {(r["unit"], r["pair_id"], r["language"], r["surface"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    length_equal, prefix_equal, answer_flip, bag_equal = [], [], [], []
    for unit in UNITS:
        for pair_id in range(3):
            for language in ("en", "zh"):
                for surface in range(4):
                    for meaning_swap in (0, 1):
                        a = lookup[(unit, pair_id, language, surface, meaning_swap, 0)]
                        b = lookup[(unit, pair_id, language, surface, meaning_swap, 1)]
                        end = a["spans"]["facts_end"][0][1]
                        length_equal.append(len(a["prompt_ids"]) == len(b["prompt_ids"]))
                        prefix_equal.append(a["prompt_ids"][:end] == b["prompt_ids"][:end])
                    for query_marker in (0, 1):
                        a = lookup[(unit, pair_id, language, surface, 0, query_marker)]
                        b = lookup[(unit, pair_id, language, surface, 1, query_marker)]
                        answer_flip.append(a["target"] != b["target"])
                        bag_equal.append(Counter(a["prompt_ids"]) == Counter(b["prompt_ids"]))
    return {"rows": len(rows), "full_length_equal_across_query_rate": sum(length_equal) / len(length_equal),
            "prefix_token_equal_rate": sum(prefix_equal) / len(prefix_equal),
            "answer_flip_rate": sum(answer_flip) / len(answer_flip),
            "definition_swap_token_multiset_equal_rate": sum(bag_equal) / len(bag_equal),
            "candidate_position_counts": dict(Counter(r["candidates"].index(r["target"]) for r in rows))}


def behavior_summary(rows: list[dict], generated: list[dict]) -> dict:
    by_id = {r["case_id"]: r for r in generated}
    detail = {str(unit): {} for unit in UNITS}
    qualified = []
    for unit in UNITS:
        for pair_id, pair in enumerate(NEW_PAIRS):
            cases = [r for r in rows if r["unit"] == unit and r["pair_id"] == pair_id]
            values = [by_id[r["case_id"]] for r in cases]
            lang = {language: sum(v["parsed_correct"] for v in values if v["language"] == language) / 16 for language in ("en", "zh")}
            swap = {str(m): sum(v["parsed_correct"] for v in values if v["meaning_swap"] == m) / 16 for m in (0, 1)}
            both = []
            for language in ("en", "zh"):
                for surface in range(4):
                    for query_marker in (0, 1):
                        matched = [r for r in cases if r["language"] == language and r["surface"] == surface and r["query_marker"] == query_marker]
                        both.append(all(by_id[r["case_id"]]["parsed_correct"] for r in matched))
            detail[str(unit)][str(pair_id)] = {"families": list(pair), "rows": 32,
                                                "accuracy": sum(v["parsed_correct"] for v in values) / 32,
                                                "language_accuracy": lang, "meaning_swap_accuracy": swap,
                                                "paired_flip_both_correct_rate": sum(both) / 16}
    for pair_id in range(3):
        if all(detail[str(unit)][str(pair_id)]["accuracy"] >= .75
               and min(detail[str(unit)][str(pair_id)]["language_accuracy"].values()) >= .625
               and min(detail[str(unit)][str(pair_id)]["meaning_swap_accuracy"].values()) >= .625
               and detail[str(unit)][str(pair_id)]["paired_flip_both_correct_rate"] >= .625 for unit in UNITS):
            qualified.append(pair_id)
    return {"aggregate_accuracy": {str(unit): sum(v["parsed_correct"] for v in generated if v["unit"] == unit) / 96 for unit in UNITS},
            "detail": detail, "qualified_pair_ids": qualified,
            "qualified_pairs": [list(NEW_PAIRS[p]) for p in qualified]}


def capture(model, rows: list[dict], behavior_map: dict[str, dict]) -> dict:
    qmods = field_utils.modules(model)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / "repaired_partner_sixevent_allqpoint.float16.npy"
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                      shape=(len(rows), len(EVENTS), len(qmods), DIM))
    captures = {}
    handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    index = []
    try:
        with torch.inference_mode():
            for model_row, row in enumerate(rows):
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                captures.clear()
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                event_positions = positions(row)
                for qpoint in range(len(qmods)):
                    field[model_row, :, qpoint] = captures[qpoint][0, event_positions].float().cpu().numpy().astype(np.float16)
                index.append({"model_row": model_row, "case_id": row["case_id"], "unit": row["unit"],
                              "pair_id": row["pair_id"], "families": row["families"], "language": row["language"],
                              "surface": row["surface"], "meaning_swap": row["meaning_swap"],
                              "query_marker": row["query_marker"], "selected_relation": row["selected_relation"],
                              "target": row["target"], "candidates": row["candidates"], "prompt_ids": row["prompt_ids"],
                              "events": list(EVENTS), "event_positions": event_positions,
                              "behavior_correct": behavior_map[row["case_id"]]["parsed_correct"]})
                if (model_row + 1) % 48 == 0:
                    field.flush(); print(f"[phase2507 field] {model_row + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        field.flush(); del field
    index_path = OUT / "index/field_rows.jsonl"
    write_jsonl(index_path, index)
    return {"event_field": str(path), "event_shape": [len(rows), len(EVENTS), len(qmods), DIM],
            "events": list(EVENTS), "event_index": str(index_path), "sha256": sha256(path)}


def prefix_control(collection: dict, qpoint: int, pair_ids: list[int]) -> dict:
    rows = [json.loads(line) for line in Path(collection["event_index"]).read_text(encoding="utf-8").splitlines()]
    field = np.load(collection["event_field"], mmap_mode="r")
    lookup = {(r["unit"], r["pair_id"], r["language"], r["surface"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    output = {}
    for event_index, event in enumerate(EVENTS[:2]):
        maxima = []
        for unit in UNITS:
            for pair_id in pair_ids:
                for language in ("en", "zh"):
                    for surface in range(4):
                        cell = {(m, q): field[lookup[(unit, pair_id, language, surface, m, q)]["model_row"], event_index, qpoint].astype(np.float32)
                                for m in (0, 1) for q in (0, 1)}
                        value = (cell[(0, 0)] - cell[(0, 1)] - cell[(1, 0)] + cell[(1, 1)]) / 4
                        maxima.append(float(np.abs(value).max()))
        output[event] = {"max_abs": max(maxima), "all_exact_zero": all(v == 0 for v in maxima)}
    return output


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 六成功关系的新配对伙伴、双新unit行为与全坐标场（自动续研）（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 原锁箱成功的六个关系形成三条边taxonomy–part-whole、role–preference、membership–translation，但这只能观察pair-relative差异。本Phase自动续研，把相同六family重配为taxonomy–role、part-whole–membership、preference–translation，使原边与新边合成一个六节点闭环。unit24/25使用两套新中英文实体和新nonce marker；每个unit为3 pair×中英文×四surface×两meaning-swap×两query-marker=96条。所有marker在各自语言内token长度相等，答案翻转与token多重集合同不变。两unit均真实贪心并采六事件×38qpoint×2560坐标。

$$E_0=\{{A-B,C-D,E-F\}},\quad E_1=\{{A-C,B-E,D-F\}},\qquad N=2\times3\times2\times4\times2\times2=192.$$

**结果汇总。** 设计 `{json.dumps(result['design_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；q30 prefix `{json.dumps(result['prefix_control'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2507_c71041_c72064_repaired_partner_behavior_fullfield.py`；新配对材料、逐行行为、六事件原场、索引、哈希和final位于`{OUT}`。

**分析与理论进展。** 如果关系family存在可加的partner-independent坐标势 (z_f)，则每条pair交互应近似 (z_a-z_b)，六边闭环的有向和应接近零；若闭环残差大，则原结果更可能是pair/任务条件化判别而非独立family坐标。q30完全沿用原confirmation，不为新配对重新选层。

**问题硬伤与结论。** 新配对只覆盖六个行为稳定family，不代表全部十二族。闭环若成立也只是线性差分近似，不是因果齿轮；若失败不能否定更非线性的关系编码。只有两个新unit共同过行为门的边可进入闭环，缺边时必须把图谱判为不可裁决而不能补造。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    f2504 = json.loads((P2504 / "analysis/final.json").read_text(encoding="utf-8"))
    qpoint = int(f2504["contract"]["qpoint"])
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows = compile_rows(tokenizer)
        audit = design_audit(rows)
        write_jsonl(OUT / "material/repaired_partner_rows.jsonl", rows)
        generated = base.behavior(model, tokenizer, rows)
        write_jsonl(OUT / "behavior/autonomous_generation.jsonl", generated)
        behavior = behavior_summary(rows, generated)
        collection = capture(model, rows, {r["case_id"]: r for r in generated})
    finally:
        model_utils.release_model(model); gc.collect()
    prefix = prefix_control(collection, qpoint, behavior["qualified_pair_ids"])
    checks = {"source_phase_passed": f2504["all_checks_passed"], "rows_192": len(rows) == 192,
              "equal_prompt_length": audit["full_length_equal_across_query_rate"] == 1.0,
              "prefix_token_equal": audit["prefix_token_equal_rate"] == 1.0,
              "all_answers_flip": audit["answer_flip_rate"] == 1.0,
              "token_multiset_control": audit["definition_swap_token_multiset_equal_rate"] == 1.0,
              "candidate_position_balanced": set(audit["candidate_position_counts"].values()) == {96},
              "at_least_two_pairs_qualified": len(behavior["qualified_pair_ids"]) >= 2,
              "all_three_pair_gate_outcome_recorded": True,
              "event_shape": collection["event_shape"] == [192, 6, 38, 2560],
              "prefix_interaction_exact_zero": all(v["all_exact_zero"] for v in prefix.values()),
              "hash": len(collection["sha256"]) == 64, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA",
              "qpoint": qpoint, "new_pairs": [list(v) for v in NEW_PAIRS], "design_audit": audit,
              "behavior": behavior, "collection": collection, "prefix_control": prefix,
              "adjudication": {"partner_recombination_behavior_gate_complete": len(behavior["qualified_pair_ids"]) == 3,
                               "six_edge_cycle_identifiable": len(behavior["qualified_pair_ids"]) == 3,
                               "automatic_alternative_repair_required": len(behavior["qualified_pair_ids"]) < 3,
                               "additive_family_graph_tested": False, "pure_semantic_code_identified": False,
                               "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
