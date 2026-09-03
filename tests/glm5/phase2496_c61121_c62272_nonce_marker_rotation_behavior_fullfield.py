#!/usr/bin/env python3
"""Cross semantic families with four nonce relation markers and capture full-coordinate fields."""
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
TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
OUT = RESULT / "phase2496_c61121_c62272_nonce_marker_rotation_behavior_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2496, "C61121-C62272", 2560
UNITS = (17, 18, 19); SPLIT = {17: "discovery", 18: "confirmation", 19: "lockbox"}
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2487_c54721_c55872_orthogonal_family_interface_behavior as base  # noqa: E402

FAMILIES = base.FAMILIES
EN_MARKERS = ("daxen", "kelvo", "muric", "tovan")
ZH_MARKERS = ("甲符", "乙符", "丙符", "丁符")
EN_NAMES = {
    17: ("Yaren", "Zelith", "Corven", "Dalune", "Evrin", "Falor", "Gerin", "Havel"),
    18: ("Iskar", "Jalen", "Korim", "Luneth", "Merov", "Navel", "Othir", "Pevan"),
    19: ("Ralen", "Sovin", "Turel", "Ulvon", "Varin", "Wexel", "Yorin", "Zaven"),
}
ZH_NAMES = {
    17: ("晓岚", "远汀", "泽川", "澄屿", "丹枫", "风禾", "古砚", "海舟"),
    18: ("简溪", "景松", "空山", "兰汀", "墨川", "南屿", "秋禾", "平澜"),
    19: ("泉舟", "松砚", "桐溪", "晚川", "溪岫", "言澜", "知禾", "舟宁"),
}
EVENTS = ("definition_semantic", "record_marker", "query_source", "answer_boundary")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def occurrence_spans(tokenizer, prompt: str, text: str) -> list[list[int]]:
    result = []; start = 0
    while True:
        pos = prompt.find(text, start)
        if pos < 0: break
        result.append([len(tokenizer.encode(prompt[:pos], add_special_tokens=False)),
                       len(tokenizer.encode(prompt[:pos + len(text)], add_special_tokens=False))])
        start = pos + len(text)
    return result


def compile_rows(tokenizer) -> list[dict]:
    rows = []; case = 61121
    for unit in UNITS:
        for family_index, family in enumerate(FAMILIES):
            for language in ("en", "zh"):
                names0 = EN_NAMES[unit] if language == "en" else ZH_NAMES[unit]
                shift = 2 * family_index % 8; names = names0[shift:] + names0[:shift]
                markers = EN_MARKERS if language == "en" else ZH_MARKERS
                for surface in (0, 1):
                    predicate = base.PREDICATES[family][language][surface]
                    for marker_id, marker in enumerate(markers):
                        a, b, c, d = names[:4]
                        query, target, foil = (a, b, d) if (marker_id + surface) % 2 == 0 else (c, d, b)
                        records = [(a, b), (c, d)] if surface == 0 else [(c, d), (a, b)]
                        candidates = [target, foil] if (family_index + marker_id + surface) % 2 == 0 else [foil, target]
                        if language == "en":
                            prompt = (f"Rule: the relation marker '{marker}' means '{predicate}'. "
                                      f"Record one: {records[0][0]} {marker} {records[0][1]}. "
                                      f"Record two: {records[1][0]} {marker} {records[1][1]}. "
                                      f"Query source: {query}. Which item is linked from that source by the defined marker?\n"
                                      f"Candidates: {candidates[0]} | {candidates[1]}\nReturn exactly one item name.\nAnswer:")
                        else:
                            prompt = (f"规则：关系标记“{marker}”表示“{predicate}”。"
                                      f"记录一：{records[0][0]}{marker}{records[0][1]}。"
                                      f"记录二：{records[1][0]}{marker}{records[1][1]}。"
                                      f"查询源：{query}。哪个项目按已定义标记与该来源相连？\n"
                                      f"候选：{candidates[0]} | {candidates[1]}\n只返回一个项目名称。\n答案：")
                        ids = [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]
                        spans = {k: occurrence_spans(tokenizer, prompt, v) for k, v in
                                 {"predicate": predicate, "marker": marker, "query": query, "target": target, "foil": foil}.items()}
                        rows.append({"case_id": f"c{case:05d}-{family}-u{unit}-{language}-s{surface}-m{marker_id}",
                                     "unit": unit, "split": SPLIT[unit], "family": family, "language": language,
                                     "definition_surface": surface, "marker_id": marker_id, "marker": marker,
                                     "predicate": predicate, "query_source": query, "target": target, "foil": foil,
                                     "candidates": candidates, "expected_output": target, "prompt": prompt,
                                     "prompt_ids": ids, "spans": spans, "answer_boundary_token": len(ids) - 1})
                        case += 1
    return rows


def parse(text: str, row: dict) -> tuple[str | None, bool]:
    cleaned = re.sub(r"^(?:answer|答案)\s*[:：]\s*", "", text.strip(), flags=re.I)
    norm = base.normalize(cleaned)
    hits = [x for x in row["candidates"] if base.normalize(x) in norm]
    value = hits[0] if len(set(hits)) == 1 else None
    return value, value == row["target"]


def behavior(model, tokenizer, rows: list[dict]) -> list[dict]:
    tokenizer.padding_side = "left"; device = model.get_input_embeddings().weight.device; result = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        encoded = tokenizer([r["prompt"] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=10, do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        width = encoded["input_ids"].shape[1]
        for row, seq in zip(batch, output):
            new = [int(x) for x in seq[width:].cpu().tolist()]; text = tokenizer.decode(new, skip_special_tokens=True)
            parsed, correct = parse(text, row)
            result.append({"case_id": row["case_id"], "unit": row["unit"], "family": row["family"],
                           "language": row["language"], "definition_surface": row["definition_surface"],
                           "marker_id": row["marker_id"], "generated_ids": new, "generated_text": text,
                           "parsed_answer": parsed, "parsed_correct": bool(correct)})
        if (start + len(batch)) % 96 == 0: print(f"[phase2496 behavior] {start + len(batch)}/{len(rows)}", flush=True)
    return result


def event_positions(row: dict) -> list[int]:
    return [row["spans"]["predicate"][0][1] - 1, row["spans"]["marker"][1][1] - 1,
            row["spans"]["query"][-1][1] - 1, row["answer_boundary_token"]]


def capture(model, rows: list[dict], behavior_map: dict[str, dict]) -> dict:
    selected = [r for r in rows if r["unit"] in (18, 19)]; mods = field_utils.modules(model)
    raw = OUT / "raw"; raw.mkdir(parents=True, exist_ok=True)
    path = raw / "nonce_marker_fourevent_allqpoint.float16.npy"
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(selected), 4, len(mods), DIM))
    full_rows = [r for r in selected if r["unit"] == 19 and r["definition_surface"] == 0 and r["marker_id"] == 0]
    total = sum(len(r["prompt_ids"]) for r in full_rows)
    full_path = raw / "lockbox_marker0_alltoken_allqpoint.float16.npy"
    full = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float16, shape=(total, len(mods), DIM))
    offsets = {}; offset = 0
    for r in full_rows: offsets[r["case_id"]] = (offset, offset + len(r["prompt_ids"])); offset += len(r["prompt_ids"])
    captures = {}; handles = []
    for q, mod in enumerate(mods):
        def hook(_m, _i, output, q=q): captures[q] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(mod.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; index = []
    try:
        with torch.inference_mode():
            for i, row in enumerate(selected):
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device); captures.clear()
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                pos = event_positions(row)
                for q in range(len(mods)):
                    tensor = captures[q][0]
                    field[i, :, q] = tensor[pos].float().cpu().numpy().astype(np.float16)
                    if row["case_id"] in offsets:
                        lo, hi = offsets[row["case_id"]]; full[lo:hi, q] = tensor.float().cpu().numpy().astype(np.float16)
                b = behavior_map[row["case_id"]]
                index.append({"model_row": i, "case_id": row["case_id"], "unit": row["unit"], "family": row["family"],
                              "language": row["language"], "definition_surface": row["definition_surface"],
                              "marker_id": row["marker_id"], "marker": row["marker"], "events": list(EVENTS),
                              "event_positions": pos, "behavior_correct": b["parsed_correct"],
                              "alltoken_offset": list(offsets[row["case_id"]]) if row["case_id"] in offsets else None})
                if (i + 1) % 96 == 0: field.flush(); full.flush(); print(f"[phase2496 field] {i + 1}/{len(selected)}", flush=True)
    finally:
        for h in handles: h.remove()
        field.flush(); full.flush(); del field, full
    index_path = OUT / "index/field_rows.jsonl"; write_jsonl(index_path, index)
    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(16 * 1024 * 1024), b""): h.update(block)
        return h.hexdigest()
    return {"event_field": str(path), "event_shape": [len(selected), 4, len(mods), DIM],
            "events": list(EVENTS), "alltoken_field": str(full_path), "alltoken_shape": [total, len(mods), DIM],
            "alltoken_rows": len(full_rows), "index": str(index_path),
            "sha256": {path.name: sha(path), full_path.name: sha(full_path)}}


def summarize(generated: list[dict]) -> dict:
    detail = {}; qualified = []
    for unit in UNITS:
        detail[str(unit)] = {}
        for family in FAMILIES:
            vals = [r for r in generated if r["unit"] == unit and r["family"] == family]
            detail[str(unit)][family] = {"rows": len(vals), "accuracy": sum(r["parsed_correct"] for r in vals) / len(vals),
                                        "en_accuracy": sum(r["parsed_correct"] for r in vals if r["language"] == "en") / 8,
                                        "zh_accuracy": sum(r["parsed_correct"] for r in vals if r["language"] == "zh") / 8}
    for family in FAMILIES:
        if all(detail[str(u)][family]["accuracy"] >= .75 and detail[str(u)][family]["en_accuracy"] >= .625
               and detail[str(u)][family]["zh_accuracy"] >= .625 for u in (18, 19)): qualified.append(family)
    aggregate = {str(u): sum(r["parsed_correct"] for r in generated if r["unit"] == u) / 192 for u in UNITS}
    return {"aggregate_accuracy": aggregate, "detail": detail, "qualified_families": qualified}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 四无意义关系标记×十二family×定义表面的576条行为与全坐标场（自动续研）（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 为削弱family与记录位置谓词token绑定，给每条任务先定义一个无意义关系标记：英文daxen/kelvo/muric/tovan，中文甲符/乙符/丙符/丁符；四marker与十二family、两套定义表面、中英、unit17/18/19完全交叉。记录本身只出现marker，不出现family谓词；中英文实体名和marker字符串独立。每条仍需真实贪心返回外部图中相连实体，共576条。对unit18/19全部384条保存定义语义末token、第一记录marker、query source、answer boundary的38qpoint×2560坐标；另保留unit19、surface0、marker0的24条全token场。

$$N=3\times12\times2\times2\times4=576,\qquad X\in\mathbb R^{{384\times4\times38\times2560}}.$$

**结果汇总。** 行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2496_c61121_c62272_nonce_marker_rotation_behavior_fullfield.py`；材料、逐行行为、四事件原场、代表全token场、索引、哈希和final位于同名目录。

**分析与理论进展。** 该设计使同一个family跨四种无意义记录marker复现，也使同一marker跨十二family出现；因此下一Phase可直接竞争family条件纹理与marker token身份纹理。它仍是基础观察，不预设线性子空间或稀疏齿轮。

**问题硬伤与结论。** family语义仍出现在前置定义句，且任务答案可主要依靠marker链接结构，未必需要真正理解定义含义。因此若family纹理存在，只能说“上文定义条件传播到后续事件”，不能说语义计算。全场float16来自BF16激活，全部坐标保留。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows = compile_rows(tokenizer); write_jsonl(OUT / "material/nonce_marker_rows.jsonl", rows)
        generated = behavior(model, tokenizer, rows); write_jsonl(OUT / "behavior/autonomous_generation.jsonl", generated)
        collection = capture(model, rows, {r["case_id"]: r for r in generated})
    finally:
        model_utils.release_model(model); gc.collect()
    summary = summarize(generated)
    checks = {"rows_576": len(rows) == 576, "four_markers_crossed": all(len({r["marker_id"] for r in rows if r["family"] == f}) == 4 for f in FAMILIES),
              "independent_language_strings": not (set(EN_MARKERS) & set(ZH_MARKERS)), "all_spans": all(all(r["spans"][k] for k in r["spans"]) for r in rows),
              "field_shape": collection["event_shape"] == [384, 4, 38, 2560], "alltoken_24": collection["alltoken_rows"] == 24,
              "hashes": len(collection["sha256"]) == 2, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA",
              "material": {"path": str(OUT / "material/nonce_marker_rows.jsonl"), "rows": len(rows)},
              "behavior": summary, "collection": collection,
              "adjudication": {"family_marker_crossing_complete": True, "semantic_computation_required_by_task": False,
                               "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
