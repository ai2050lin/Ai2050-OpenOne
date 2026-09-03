#!/usr/bin/env python3
"""Fresh equal-token-length lockbox for the behaviorally necessary relation-swap contract."""
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
P2500 = RESULT / "phase2500_c64001_c65152_semantic_necessity_2x2_behavior"
P2502 = RESULT / "phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox"
OUT = RESULT / "phase2503_c67201_c68224_equal_length_fresh_lockbox_behavior_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, UNIT, DIM = 2503, "C67201-C68224", 23, 2560
EVENTS = ("definition_end", "facts_end", "query_marker", "candidate0", "candidate1", "answer_boundary")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2500_c64001_c65152_semantic_necessity_2x2_behavior as base  # noqa: E402


FRESH_MARKERS = {"en": ("velun", "joric"), "zh": ("庚符", "辛符")}
FRESH_EN_NAMES = ("Brelan", "Cavik", "Dorem", "Elvar", "Faron", "Gilem", "Horis", "Juvan")
FRESH_ZH_NAMES = ("柏舟", "辰野", "岱川", "恩澜", "枫屿", "观禾", "寒汀", "锦溪")


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


def compile_fresh(tokenizer) -> list[dict]:
    # Reuse the frozen Phase2500 grammar only; replace all lockbox strings and IDs.
    base.UNITS = (UNIT,)
    base.SPLIT = {UNIT: "fresh_lockbox"}
    base.MARKERS[UNIT] = FRESH_MARKERS
    base.EN_NAMES[UNIT] = FRESH_EN_NAMES
    base.ZH_NAMES[UNIT] = FRESH_ZH_NAMES
    rows = base.compile_rows(tokenizer)
    for index, row in enumerate(rows, start=67201):
        row["case_id"] = f"c{index:05d}-p{row['pair_id']}-u23-{row['language']}-s{row['surface']}-m{row['meaning_swap']}-q{row['query_marker']}"
    return rows


def event_positions(row: dict) -> list[int]:
    spans = row["spans"]
    return [spans["definition_end"][0][1] - 1, spans["facts_end"][0][1] - 1,
            spans["query_marker"][-1][1] - 1, spans["candidate0"][-1][1] - 1,
            spans["candidate1"][-1][1] - 1, row["answer_boundary_token"]]


def behavior_summary(rows: list[dict], generated: list[dict], original_qualified: set[int]) -> dict:
    by_id = {r["case_id"]: r for r in generated}
    detail = {}
    qualified = []
    for pair_id, pair in enumerate(base.PAIRS):
        cases = [r for r in rows if r["pair_id"] == pair_id]
        values = [by_id[r["case_id"]] for r in cases]
        language_accuracy = {language: sum(v["parsed_correct"] for v in values if v["language"] == language) / 16
                             for language in ("en", "zh")}
        swap_accuracy = {str(swap): sum(v["parsed_correct"] for v in values if v["meaning_swap"] == swap) / 16
                         for swap in (0, 1)}
        both_correct = []
        for language in ("en", "zh"):
            for surface in range(4):
                for query_marker in (0, 1):
                    paired = [r for r in cases if r["language"] == language and r["surface"] == surface
                              and r["query_marker"] == query_marker]
                    both_correct.append(all(by_id[r["case_id"]]["parsed_correct"] for r in paired))
        panel = {"families": list(pair), "rows": 32,
                 "accuracy": sum(v["parsed_correct"] for v in values) / 32,
                 "language_accuracy": language_accuracy, "meaning_swap_accuracy": swap_accuracy,
                 "paired_flip_both_correct_rate": sum(both_correct) / 16,
                 "qualified_in_phase2500": pair_id in original_qualified}
        panel["fresh_gate"] = (panel["accuracy"] >= .75 and min(language_accuracy.values()) >= .625
                               and min(swap_accuracy.values()) >= .625 and panel["paired_flip_both_correct_rate"] >= .625)
        detail[str(pair_id)] = panel
        if pair_id in original_qualified and panel["fresh_gate"]:
            qualified.append(pair_id)
    return {"accuracy": sum(r["parsed_correct"] for r in generated) / len(generated),
            "paired_flip_both_correct_rate": sum(detail[str(p)]["paired_flip_both_correct_rate"] for p in range(6)) / 6,
            "detail": detail, "qualified_pair_ids_intersection": qualified,
            "qualified_pairs_intersection": [list(base.PAIRS[p]) for p in qualified]}


def design_audit(rows: list[dict]) -> dict:
    lookup = {(r["pair_id"], r["language"], r["surface"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    length_equal, prefix_equal, position_equal, answer_flip, bag_equal = [], [], [], [], []
    for pair_id in range(6):
        for language in ("en", "zh"):
            for surface in range(4):
                for meaning_swap in (0, 1):
                    a = lookup[(pair_id, language, surface, meaning_swap, 0)]
                    b = lookup[(pair_id, language, surface, meaning_swap, 1)]
                    end = a["spans"]["facts_end"][0][1]
                    length_equal.append(len(a["prompt_ids"]) == len(b["prompt_ids"]))
                    prefix_equal.append(a["prompt_ids"][:end] == b["prompt_ids"][:end])
                    position_equal.append(event_positions(a)[:2] == event_positions(b)[:2])
                for query_marker in (0, 1):
                    a = lookup[(pair_id, language, surface, 0, query_marker)]
                    b = lookup[(pair_id, language, surface, 1, query_marker)]
                    answer_flip.append(a["target"] != b["target"])
                    bag_equal.append(Counter(a["prompt_ids"]) == Counter(b["prompt_ids"]))
    return {"rows": len(rows), "full_prompt_length_equal_across_query_rate": sum(length_equal) / len(length_equal),
            "prefix_token_equal_rate": sum(prefix_equal) / len(prefix_equal),
            "prefix_event_position_equal_rate": sum(position_equal) / len(position_equal),
            "answer_flip_rate": sum(answer_flip) / len(answer_flip),
            "definition_swap_token_multiset_equal_rate": sum(bag_equal) / len(bag_equal)}


def capture(model, rows: list[dict], behavior_map: dict[str, dict], qualified: set[int]) -> dict:
    qmods = field_utils.modules(model)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    event_path = raw / "fresh_lockbox_sixevent_allqpoint.float16.npy"
    event_field = np.lib.format.open_memmap(event_path, mode="w+", dtype=np.float16,
                                            shape=(len(rows), len(EVENTS), len(qmods), DIM))
    full_rows = [r for r in rows if r["pair_id"] in qualified and r["surface"] == 0 and r["query_marker"] == 0]
    total_tokens = sum(len(r["prompt_ids"]) for r in full_rows)
    full_path = raw / "fresh_lockbox_surface0_querymarker0_alltoken_allqpoint.float16.npy"
    full_field = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float16,
                                           shape=(total_tokens, len(qmods), DIM))
    offsets = {}
    offset = 0
    for row in full_rows:
        offsets[row["case_id"]] = (offset, offset + len(row["prompt_ids"]))
        offset += len(row["prompt_ids"])
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
                positions = event_positions(row)
                for qpoint in range(len(qmods)):
                    tensor = captures[qpoint][0]
                    event_field[model_row, :, qpoint] = tensor[positions].float().cpu().numpy().astype(np.float16)
                    if row["case_id"] in offsets:
                        lo, hi = offsets[row["case_id"]]
                        full_field[lo:hi, qpoint] = tensor.float().cpu().numpy().astype(np.float16)
                index.append({"model_row": model_row, "case_id": row["case_id"], "unit": UNIT,
                              "pair_id": row["pair_id"], "families": row["families"], "language": row["language"],
                              "surface": row["surface"], "meaning_swap": row["meaning_swap"],
                              "query_marker": row["query_marker"], "selected_relation": row["selected_relation"],
                              "target": row["target"], "candidates": row["candidates"], "prompt_ids": row["prompt_ids"],
                              "events": list(EVENTS), "event_positions": positions,
                              "behavior_correct": behavior_map[row["case_id"]]["parsed_correct"],
                              "alltoken_offset": list(offsets[row["case_id"]]) if row["case_id"] in offsets else None})
                if (model_row + 1) % 64 == 0:
                    event_field.flush(); full_field.flush()
                    print(f"[phase2503 field] {model_row + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        event_field.flush(); full_field.flush()
        del event_field, full_field
    index_path = OUT / "index/field_rows.jsonl"
    write_jsonl(index_path, index)
    alltoken_index_path = OUT / "index/alltoken_rows.jsonl"
    write_jsonl(alltoken_index_path, [{"case_id": r["case_id"], "offset": list(offsets[r["case_id"]]),
                                      "prompt_ids": r["prompt_ids"], "pair_id": r["pair_id"],
                                      "language": r["language"], "meaning_swap": r["meaning_swap"]} for r in full_rows])
    return {"event_field": str(event_path), "event_shape": [len(rows), len(EVENTS), len(qmods), DIM],
            "events": list(EVENTS), "event_index": str(index_path),
            "alltoken_field": str(full_path), "alltoken_shape": [total_tokens, len(qmods), DIM],
            "alltoken_rows": len(full_rows), "alltoken_index": str(alltoken_index_path),
            "sha256": {event_path.name: sha256(event_path), full_path.name: sha256(full_path)}}


def prefix_interaction_max(collection: dict, qpoint: int, pair_ids: list[int]) -> dict:
    rows = [json.loads(line) for line in Path(collection["event_index"]).read_text(encoding="utf-8").splitlines()]
    field = np.load(collection["event_field"], mmap_mode="r")
    lookup = {(r["pair_id"], r["language"], r["surface"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    result = {}
    for event_index, event in enumerate(EVENTS[:2]):
        maxima = []
        for pair_id in pair_ids:
            for language in ("en", "zh"):
                for surface in range(4):
                    cell = {(m, q): field[lookup[(pair_id, language, surface, m, q)]["model_row"], event_index, qpoint].astype(np.float32)
                            for m in (0, 1) for q in (0, 1)}
                    interaction = (cell[(0, 0)] - cell[(0, 1)] - cell[(1, 0)] + cell[(1, 1)]) / 4
                    maxima.append(float(np.abs(interaction).max()))
        result[event] = {"max_abs": max(maxima), "all_exact_zero": all(v == 0.0 for v in maxima)}
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 等token长度nonce与全新实体的语义必要性锁箱重采（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2502发现首次unit22英文两个query marker的token长度不等，导致完整序列矩阵shape不同，破坏了causal-prefix严格零合同。本Phase不修改旧结果，而是建立独立unit23：英文marker `velun/joric`各为两个token，中文`庚符/辛符`各为两个token；中英文实体全部更新。仍使用六pair×两语言×四surface×两meaning-swap×两query-marker共192条，答案翻转与token多重集合同不变。先真实贪心行为，再采集六事件×38qpoint×2560坐标；合格pair代表样本另存逐token全场。

$$N=6\times2\times4\times2\times2=192,\qquad |\operatorname{{tok}}(m_0)|=|\operatorname{{tok}}(m_1)|.$$

**结果汇总。** 设计审计 `{json.dumps(result['design_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；冻结qpoint的prefix交互 `{json.dumps(result['prefix_control'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2503_c67201_c68224_equal_length_fresh_lockbox_behavior_fullfield.py`；全新材料、逐行生成、六事件原场、代表全token场、索引、哈希与final位于`{OUT}`。

**理论进展。** 本Phase的职责是恢复可辨识的物理测量合同，不用新lockbox重新选层。只有Phase2500的unit21选择qpoint和unit23共同合格pair进入下一Phase。prefix严格零若通过，说明四格后续非零不再能由完整序列长度造成的数值核差异解释。

**问题硬伤与结论。** 等token长度只消除已观察到的shape泄漏，不消除合成模板、pair-relative选择和目标实体准备混杂。行为门仍是关系含义必要性的前提，不是HiddenState因果性的结论。未通过新行为门的pair继续作为负对照，不得为了增加family数而混入主锁箱。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    f2500 = json.loads((P2500 / "analysis/final.json").read_text(encoding="utf-8"))
    f2502 = json.loads((P2502 / "analysis/final.json").read_text(encoding="utf-8"))
    original_qualified = set(f2500["behavior"]["qualified_pair_ids"])
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows = compile_fresh(tokenizer)
        audit = design_audit(rows)
        write_jsonl(OUT / "material/fresh_lockbox_rows.jsonl", rows)
        generated = base.behavior(model, tokenizer, rows)
        write_jsonl(OUT / "behavior/autonomous_generation.jsonl", generated)
        behavior = behavior_summary(rows, generated, original_qualified)
        qualified = set(behavior["qualified_pair_ids_intersection"])
        collection = capture(model, rows, {r["case_id"]: r for r in generated}, qualified)
    finally:
        model_utils.release_model(model)
        gc.collect()
    frozen_qpoint = int(f2502["selection"]["qpoint"])
    prefix_control = prefix_interaction_max(collection, frozen_qpoint, sorted(qualified))
    checks = {"rows_192": len(rows) == 192,
              "full_length_equal": audit["full_prompt_length_equal_across_query_rate"] == 1.0,
              "prefix_tokens_equal": audit["prefix_token_equal_rate"] == 1.0,
              "answer_flip": audit["answer_flip_rate"] == 1.0,
              "token_multiset_control": audit["definition_swap_token_multiset_equal_rate"] == 1.0,
              "at_least_three_jointly_qualified_pairs": len(qualified) >= 3,
              "event_shape": collection["event_shape"] == [192, 6, 38, 2560],
              "alltoken_shape": collection["alltoken_shape"][1:] == [38, 2560],
              "prefix_interaction_exact_zero": all(v["all_exact_zero"] for v in prefix_control.values()),
              "hashes": len(collection["sha256"]) == 2, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA",
              "frozen_qpoint": frozen_qpoint, "design_audit": audit, "behavior": behavior,
              "collection": collection, "prefix_control": prefix_control,
              "adjudication": {"fresh_lockbox_protocol_valid": checks["prefix_interaction_exact_zero"],
                               "semantic_code_identified": False, "causal_mediator_identified": False,
                               "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
