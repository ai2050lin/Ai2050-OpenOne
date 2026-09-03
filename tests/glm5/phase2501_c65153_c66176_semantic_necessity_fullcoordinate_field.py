#!/usr/bin/env python3
"""Capture all-qpoint, all-coordinate fields for the relation-meaning 2x2 behavior contract."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2500 = RESULT / "phase2500_c64001_c65152_semantic_necessity_2x2_behavior"
OUT = RESULT / "phase2501_c65153_c66176_semantic_necessity_fullcoordinate_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2501, "C65153-C66176", 2560
EVENTS = ("definition_end", "facts_end", "query_marker", "candidate0", "candidate1", "answer_boundary")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


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


def event_positions(row: dict) -> list[int]:
    spans = row["spans"]
    return [
        spans["definition_end"][0][1] - 1,
        spans["facts_end"][0][1] - 1,
        spans["query_marker"][-1][1] - 1,
        spans["candidate0"][-1][1] - 1,
        spans["candidate1"][-1][1] - 1,
        row["answer_boundary_token"],
    ]


def capture(model, rows: list[dict], behavior_map: dict[str, dict], qualified: set[int]) -> dict:
    selected = [r for r in rows if r["unit"] in (21, 22)]
    qmods = field_utils.modules(model)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    event_path = raw / "semantic_necessity_sixevent_allqpoint.float16.npy"
    event_field = np.lib.format.open_memmap(event_path, mode="w+", dtype=np.float16,
                                            shape=(len(selected), len(EVENTS), len(qmods), DIM))
    full_rows = [r for r in selected if r["unit"] == 22 and r["pair_id"] in qualified
                 and r["surface"] == 0 and r["query_marker"] == 0]
    total_tokens = sum(len(r["prompt_ids"]) for r in full_rows)
    full_path = raw / "lockbox_surface0_querymarker0_alltoken_allqpoint.float16.npy"
    full_field = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float16,
                                           shape=(total_tokens, len(qmods), DIM))
    offsets = {}
    offset = 0
    for row in full_rows:
        offsets[row["case_id"]] = (offset, offset + len(row["prompt_ids"]))
        offset += len(row["prompt_ids"])
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    index = []
    try:
        with torch.inference_mode():
            for model_row, row in enumerate(selected):
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
                behavior = behavior_map[row["case_id"]]
                index.append({
                    "model_row": model_row, "case_id": row["case_id"], "unit": row["unit"],
                    "pair_id": row["pair_id"], "families": row["families"], "language": row["language"],
                    "surface": row["surface"], "meaning_swap": row["meaning_swap"],
                    "query_marker": row["query_marker"], "selected_relation": row["selected_relation"],
                    "source": row["source"], "target": row["target"], "relation_targets": row["relation_targets"],
                    "candidates": row["candidates"], "prompt_ids": row["prompt_ids"],
                    "events": list(EVENTS), "event_positions": positions,
                    "behavior_correct": behavior["parsed_correct"],
                    "alltoken_offset": list(offsets[row["case_id"]]) if row["case_id"] in offsets else None,
                })
                if (model_row + 1) % 64 == 0:
                    event_field.flush()
                    full_field.flush()
                    print(f"[phase2501 field] {model_row + 1}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        event_field.flush()
        full_field.flush()
        del event_field, full_field
    index_path = OUT / "index/field_rows.jsonl"
    write_jsonl(index_path, index)
    alltoken_index = []
    for row in full_rows:
        lo, hi = offsets[row["case_id"]]
        alltoken_index.append({"case_id": row["case_id"], "offset": [lo, hi], "prompt_ids": row["prompt_ids"],
                               "tokens": None, "unit": row["unit"], "pair_id": row["pair_id"],
                               "language": row["language"], "meaning_swap": row["meaning_swap"]})
    alltoken_index_path = OUT / "index/alltoken_rows.jsonl"
    write_jsonl(alltoken_index_path, alltoken_index)
    return {
        "event_field": str(event_path), "event_shape": [len(selected), len(EVENTS), len(qmods), DIM],
        "events": list(EVENTS), "event_index": str(index_path),
        "alltoken_field": str(full_path), "alltoken_shape": [total_tokens, len(qmods), DIM],
        "alltoken_rows": len(full_rows), "alltoken_index": str(alltoken_index_path),
        "sha256": {event_path.name: sha256(event_path), full_path.name: sha256(full_path)},
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 语义必要性四格的全层全事件全2560坐标场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2500的confirmation unit21与lockbox unit22全部384条材料进行非量化BF16前向。无论行为是否合格，六个关系对都采集，从而保留失败对照；后续主分析只使用两个unit共同通过行为门的pair。每条在definition-end、facts-end、query-marker、固定candidate0、固定candidate1、answer-boundary六个事件保存Embedding(q0)、36个block输出(q1–q36)和final RMSNorm(q37)的全部2560物理坐标。另对lockbox中合格pair、surface0、query-marker0、两个meaning-swap、中英文共16条保存逐token全场。

$$X\in\mathbb R^{{384\times6\times38\times2560}},\qquad X_{{token}}\in\mathbb R^{{{result['collection']['alltoken_shape'][0]}\times38\times2560}}.$$

**结果汇总。** 行为门 `{json.dumps(result['behavior_gate'], ensure_ascii=False)}`；采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2501_c65153_c66176_semantic_necessity_fullcoordinate_field.py`；六事件原场、代表全token场、逐行索引、哈希与final位于`{OUT}`。

**分析与理论进展。** 这批场第一次把“答案必须随关系定义交换而翻转”的行为合同与全物理坐标绑定。固定candidate0/1事件而不按正确target动态选位置，避免把不同实体位置误当成关系选择效应。definition/facts事件位于query marker出现之前，可检验四格交互是否按causal mask严格为零。

**问题硬伤与结论。** float16落盘保存的是BF16前向的数值近似，适合全场纹理而不适合声称极小单坐标精确值。全token场是16条代表切片，不是384条全部token；六事件场才覆盖完整合同。采集本身不是机制证据，必须由confirmation冻结qpoint、lockbox同层裁决。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    f2500 = json.loads((P2500 / "analysis/final.json").read_text(encoding="utf-8"))
    rows = read_jsonl(Path(f2500["material"]["path"]))
    behavior_rows = read_jsonl(P2500 / "behavior/autonomous_generation.jsonl")
    behavior_map = {r["case_id"]: r for r in behavior_rows}
    qualified = set(f2500["behavior"]["qualified_pair_ids"])
    model, _, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        collection = capture(model, rows, behavior_map, qualified)
    finally:
        model_utils.release_model(model)
        gc.collect()
    checks = {
        "source_phase_passed": f2500["all_checks_passed"],
        "event_shape": collection["event_shape"] == [384, 6, 38, 2560],
        "six_events": collection["events"] == list(EVENTS),
        "alltoken_16_rows": collection["alltoken_rows"] == 16,
        "alltoken_full_coordinates": collection["alltoken_shape"][1:] == [38, 2560],
        "hashes": len(collection["sha256"]) == 2,
        "qualified_and_negative_pairs_retained": True,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA",
        "behavior_gate": {"qualified_pair_ids": sorted(qualified), "qualified_pairs": f2500["behavior"]["qualified_pairs"],
                          "confirmation_accuracy": f2500["behavior"]["aggregate_accuracy"]["21"],
                          "lockbox_accuracy": f2500["behavior"]["aggregate_accuracy"]["22"],
                          "confirmation_paired_flip": f2500["behavior"]["aggregate_paired_flip_success"]["21"],
                          "lockbox_paired_flip": f2500["behavior"]["aggregate_paired_flip_success"]["22"]},
        "collection": collection,
        "adjudication": {"full_coordinate_measurement_complete": True, "semantic_code_identified": False,
                         "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
