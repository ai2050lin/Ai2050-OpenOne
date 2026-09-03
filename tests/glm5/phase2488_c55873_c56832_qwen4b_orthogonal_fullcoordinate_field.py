#!/usr/bin/env python3
"""Capture all-coordinate event fields and an entity-interface all-token panel."""
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
P2487 = RESULT / "phase2487_c54721_c55872_orthogonal_family_interface_behavior"
OUT = RESULT / "phase2488_c55873_c56832_qwen4b_orthogonal_fullcoordinate_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2488, "C55873-C56832", 2560
EVENTS = ("record_predicate", "query_source", "candidate_target", "candidate_foil", "answer_boundary")
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows() -> tuple[list[dict], dict[str, dict]]:
    rows = [r for r in read_jsonl(P2487 / "material/orthogonal_family_interface_rows.jsonl") if r["unit"] in (15, 16)]
    behavior = {r["case_id"]: r for r in read_jsonl(P2487 / "behavior/autonomous_generation.jsonl")}
    return rows, behavior


def event_positions(row: dict) -> list[int]:
    spans = row["token_spans"]
    return [
        spans["predicate"][0][1] - 1,
        spans["query_source"][-1][1] - 1,
        spans["target"][-1][1] - 1,
        spans["foil"][-1][1] - 1,
        row["answer_boundary_token"],
    ]


def capture(model, tokenizer, rows: list[dict], behavior: dict[str, dict]) -> dict:
    qmods = field_utils.modules(model)
    if len(qmods) != 38:
        raise RuntimeError(f"Expected 38 qpoints, got {len(qmods)}")
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    event_path = raw / "orthogonal_fiveevent_allqpoint.float16.npy"
    states = np.lib.format.open_memmap(event_path, mode="w+", dtype=np.float16,
                                       shape=(len(rows), len(EVENTS), len(qmods), DIM))
    alltoken_rows = [r for r in rows if r["unit"] == 16 and r["output_interface"] == "entity"]
    total_tokens = sum(len(r["prompt_ids"]) for r in alltoken_rows)
    alltoken_path = raw / "lockbox_entity_alltoken_allqpoint.float16.npy"
    alltoken_ids_path = raw / "lockbox_entity_token_ids.int32.npy"
    alltoken = np.lib.format.open_memmap(alltoken_path, mode="w+", dtype=np.float16,
                                         shape=(total_tokens, len(qmods), DIM))
    token_ids = np.lib.format.open_memmap(alltoken_ids_path, mode="w+", dtype=np.int32, shape=(total_tokens,))
    alltoken_offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for row in alltoken_rows:
        end = offset + len(row["prompt_ids"])
        alltoken_offsets[row["case_id"]] = (offset, end)
        token_ids[offset:end] = np.asarray(row["prompt_ids"], dtype=np.int32)
        offset = end
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    batch_size = 8
    index: list[dict] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                width = max(len(r["prompt_ids"]) for r in batch)
                ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                for b, row in enumerate(batch):
                    seq = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
                    ids[b, :len(seq)] = seq
                    mask[b, :len(seq)] = 1
                captures.clear()
                model(input_ids=ids, attention_mask=mask, use_cache=False)
                if len(captures) != len(qmods):
                    raise RuntimeError(f"Missing qpoints: {len(captures)}/{len(qmods)}")
                for b, row in enumerate(batch):
                    positions = event_positions(row)
                    model_row = start + b
                    for qpoint in range(len(qmods)):
                        tensor = captures[qpoint][b]
                        states[model_row, :, qpoint, :] = tensor[positions].float().cpu().numpy().astype(np.float16)
                        if row["case_id"] in alltoken_offsets:
                            lo, hi = alltoken_offsets[row["case_id"]]
                            alltoken[lo:hi, qpoint, :] = tensor[:len(row["prompt_ids"])].float().cpu().numpy().astype(np.float16)
                    beh = behavior[row["case_id"]]
                    index.append({
                        "model_row": model_row, "case_id": row["case_id"], "unit": row["unit"],
                        "family": row["family"], "language": row["language"], "surface": row["surface"],
                        "output_interface": row["output_interface"], "events": list(EVENTS),
                        "event_token_positions": positions, "event_token_ids": [row["prompt_ids"][p] for p in positions],
                        "behavior_correct": beh["parsed_correct"], "behavior_parsed": beh["parsed_answer"],
                        "alltoken_offset": list(alltoken_offsets[row["case_id"]]) if row["case_id"] in alltoken_offsets else None,
                    })
                if (start + len(batch)) % 128 == 0:
                    states.flush(); alltoken.flush(); token_ids.flush()
                    print(f"[phase2488] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        states.flush(); alltoken.flush(); token_ids.flush()
        del states, alltoken, token_ids
    index_path = OUT / "index/orthogonal_field_rows.jsonl"
    write_jsonl(index_path, index)
    return {
        "event_field": str(event_path), "event_shape": [len(rows), len(EVENTS), len(qmods), DIM],
        "events": list(EVENTS), "alltoken_field": str(alltoken_path),
        "alltoken_shape": [total_tokens, len(qmods), DIM], "alltoken_token_ids": str(alltoken_ids_path),
        "alltoken_rows": len(alltoken_rows), "index": str(index_path),
        "bytes": {p.name: p.stat().st_size for p in (event_path, alltoken_path, alltoken_ids_path)},
        "sha256": {p.name: sha256(p) for p in (event_path, alltoken_path, alltoken_ids_path)},
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Qwen3-4B正交语言族五事件与锁箱实体接口全token全坐标原场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2487全部confirmation/unit15与lockbox/unit16共768条，无论行为是否通过，都保存record谓词、query source、target候选、foil候选和answer boundary五个真实token事件的q0 Embedding、q1–q36 block输出、q37 final norm及全部2560物理坐标。另对unit16全部十二族×中英×四surface的96条实体接口，拼接保存每个prompt token×38qpoint×2560坐标；不做Top-K、PCA、阈值或低秩归档。行为正确标签与状态同索引保存。

$$X\in\mathbb R^{{768\times5\times38\times2560}},\qquad X_{{all-token}}\in\mathbb R^{{N_{{token}}\times38\times2560}}.$$

**结果汇总。** 原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；资格引用 `{json.dumps(result['behavior_gate'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2488_c55873_c56832_qwen4b_orthogonal_fullcoordinate_field.py`；三个原始数组、逐行event/token索引、SHA256和`analysis/final.json`位于同名结果目录。

**分析与理论进展。** 这是新正交材料的L0测量层：中英文实体字符串独立，surface与family全交叉，输出接口预先固定。四接口都可研究提示如何改变输入响应；只有Phase2487行为门通过的family-interface才能升级为“成功执行中的纹理”。逐坐标数组以固定物理顺序保存，后续任何重排只允许用于显示。

**问题硬伤与结论。** float16只是BF16激活的落盘格式，保留了全坐标但不是无损BF16；五事件仍是离散切片，因此额外保存实体锁箱全token路径。输入响应不等于因果计算，answer boundary含有接口指令和标点。该Phase不报告语义比例、不命名齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    rows, behavior = selected_rows()
    final2487 = json.loads((P2487 / "analysis/final.json").read_text(encoding="utf-8"))
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        collection = capture(model, tokenizer, rows, behavior)
    finally:
        model_utils.release_model(model)
        gc.collect()
    paths = [Path(collection[k]) for k in ("event_field", "alltoken_field", "alltoken_token_ids", "index")]
    checks = {
        "rows_768": collection["event_shape"][0] == 768,
        "five_events": collection["event_shape"][1] == 5,
        "qpoints_38": collection["event_shape"][2] == 38,
        "all_2560_coordinates": collection["event_shape"][3] == 2560,
        "alltoken_96_rows": collection["alltoken_rows"] == 96,
        "files_exist": all(p.exists() and p.stat().st_size > 0 for p in paths),
        "hashes": len(collection["sha256"]) == 3,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "model": {"name": "Qwen3-4B", "inference": "nonquantized BF16 CUDA",
                  "archive": "float16 full physical coordinates"},
        "collection": collection,
        "behavior_gate": {"qualified": final2487["behavior"]["qualified"],
                          "rule": "input-response rows all retained; successful-execution claims only for qualified family-interface pairs"},
        "adjudication": {"orthogonal_fullfield_available": True, "topk_primary_analysis": False,
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
