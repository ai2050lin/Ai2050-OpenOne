#!/usr/bin/env python3
"""Collect Qwen3-4B event-aligned full-coordinate fields and all-token references."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2397 = RESULT / "phase2397_c21681_c22000_operation_behavior_token_calibration/qwen4b"
OUT = RESULT / "phase2398_c22001_c22320_qwen4b_event_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2398
CAMPAIGN = "C22001-C22320"

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def pad_right(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = 1
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def index_row(row: dict) -> dict:
    keep = ("case_id", "task", "family", "unit", "language", "surface", "direction", "partition",
            "target_candidate_slot", "query_role", "steps", "answer", "foil", "prompt_token_count")
    result = {key: row[key] for key in keep if key in row}
    result["event_names"] = [event["event"] for event in row["event_tokens"]]
    result["event_token_indices"] = [event["token_index"] for event in row["event_tokens"]]
    return result


def collect_event_field(model, rows: list[dict], task: str, batch_size: int) -> dict:
    modules = field_utils.modules(model)
    dimension = int(model.get_input_embeddings().weight.shape[1])
    event_count = len(rows[0]["event_tokens"])
    if any(len(row["event_tokens"]) != event_count for row in rows):
        raise RuntimeError((task, "variable_event_count"))
    path = OUT / f"raw/{task}_event_field.float16.npy"
    progress_path = OUT / f"raw/{task}_progress.json"
    shape = (len(rows), len(modules), event_count, dimension)
    if path.exists():
        values = np.lib.format.open_memmap(path, mode="r+")
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        completed = 0
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = pad_right([row["prompt_ids"] for row in batch], device, pad)
                captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for qpoint in range(len(modules)):
                    tensor = captures[qpoint].float().cpu()
                    selected = torch.stack([
                        tensor[local, torch.tensor([event["token_index"] for event in row["event_tokens"]], dtype=torch.long)]
                        for local, row in enumerate(batch)
                    ])
                    values[start:start + len(batch), qpoint] = selected.numpy().astype(np.float16)
                values.flush()
                save(progress_path, {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 128 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2398 {task}] {start + len(batch)}/{len(rows)}", flush=True)
                del ids, mask, positions
    finally:
        for handle in handles:
            handle.remove()
        values.flush(); close(values)
    return {"path": str(path), "shape": list(shape), "bytes": path.stat().st_size,
            "event_names": [event["event"] for event in rows[0]["event_tokens"]]}


def reference_rows(rows: list[dict]) -> list[dict]:
    selected: list[dict] = []
    # One fresh row for every selection family x language x surface (32), and composition family x language x surface x steps (32).
    strata = set()
    for row in rows:
        if row["task"] == "selection" and row["partition"] == "fresh_unit_lockbox":
            key = (row["task"], row["family"], row["language"], row["surface"])
        elif row["task"] == "composition" and row["partition"] in ("fresh_unit_lockbox", "fresh_composition_lockbox"):
            key = (row["task"], row["family"], row["language"], row["surface"], row["steps"])
        else:
            continue
        if key not in strata:
            strata.add(key); selected.append(row)
    if len(selected) != 64:
        raise RuntimeError(("reference_strata", len(selected)))
    return selected


def collect_reference(model, rows: list[dict]) -> dict:
    field_path = OUT / "raw/reference_prompt_answer_all_token.float16.npy"
    mask_path = OUT / "raw/reference_prompt_answer_mask.uint8.npy"
    index_path = OUT / "index/reference_rows.jsonl"
    if field_path.exists() and mask_path.exists() and index_path.exists():
        values = np.load(field_path, mmap_mode="r")
        result = {"path": str(field_path), "shape": list(values.shape), "bytes": field_path.stat().st_size,
                  "mask_path": str(mask_path), "rows": len(rows), "resumed": True}
        close(values); return result
    sequences = [row["prompt_ids"] + row["target_ids"] for row in rows]
    width = max(map(len, sequences))
    modules = field_utils.modules(model)
    dimension = int(model.get_input_embeddings().weight.shape[1])
    shape = (len(rows), len(modules), width, dimension)
    field_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16, shape=shape)
    valid = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(len(rows), width))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for index, sequence in enumerate(sequences):
                ids = torch.tensor([sequence], dtype=torch.long, device=device)
                mask = torch.ones_like(ids); positions = torch.arange(len(sequence), device=device)[None]
                captures.clear(); model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for qpoint in range(len(modules)):
                    values[index, qpoint, :len(sequence)] = captures[qpoint][0].float().cpu().numpy().astype(np.float16)
                valid[index, :len(sequence)] = 1
                values.flush(); valid.flush()
                print(f"[phase2398 reference] {index + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        values.flush(); valid.flush(); close(values); close(valid)
    write_rows(index_path, [index_row(row) | {"prompt_ids": row["prompt_ids"], "target_ids": row["target_ids"]} for row in rows])
    return {"path": str(field_path), "shape": list(shape), "bytes": field_path.stat().st_size,
            "mask_path": str(mask_path), "rows": len(rows), "valid_tokens": sum(map(len, sequences)), "resumed": False}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen4B事件×层×全物理坐标HiddenState场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2396全部3072条操作族prompt使用Phase2397逐模型校准后的真实chat token事件。Qwen3-4B BF16单模型驻留CUDA，采集embedding、每个Transformer block输出、final norm的每一个物理坐标；选择任务保存2048×38×8×2560场，组合任务保存1024×38×12×2560场，不做Top-K、PCA或激活幅值阈值。另从每个任务族×语言×表达×步数的fresh锁箱分层抽取64条，保存prompt+正确答案全部token、全部checkpoint、全部坐标，为事件锚点之外的连续轨迹复核。

$$\mathcal H\in\mathbb R^{{N\times Q\times E\times d}},\quad
\mathcal H_{{r,q,e,j}}=H_q(x_r)[t_{{r,e}},j],\quad j=1,\ldots,d,$$

$$U_{{r,q,e,j}}=\mathcal H_{{r,q+1,e,j}}-\mathcal H_{{r,q,e,j}},\quad q=0,\ldots,Q-3,$$

其中final norm不当作残差block更新，只保留为输出接口观察点。

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2398_c22001_c22320_qwen4b_event_fullfield.py`；全坐标场、事件行索引、64条全token参考和final位于 `tests/glm5/result/phase2398_c22001_c22320_qwen4b_event_fullfield`。

**理论进展。** 研究对象首次从“某一边界能否被外部判别”转为同一自然语言操作在事实、查询、候选和输出准备事件上的逐层全坐标轨迹。embedding与HiddenState使用完全相同的物理坐标索引保存，低幅值参数未被丢弃，因此后续可以直接比较坐标在层间的出现、延续、翻转、复用和分化。

**问题硬伤与结论。** 事件锚点是语义位置的离散近似，关系短语可能跨多个token而当前锚点取短语末token；64条全token参考用于检查这一硬伤。层差分同时包含模型通用残差变换与任务条件变换，不能直接命名为齿轮。Qwen4B在Phase2397的选择teacher锁箱较强而两步组合较弱，故后续必须分别报告，不得用选择族阳性掩盖组合失败。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2397 / "index/operation_rows.jsonl")
    selection = [row for row in rows if row["task"] == "selection"]
    composition = [row for row in rows if row["task"] == "composition"]
    write_rows(OUT / "index/selection_rows.jsonl", [index_row(row) for row in selection])
    write_rows(OUT / "index/composition_rows.jsonl", [index_row(row) for row in composition])
    model, tokenizer, label = capability.load_model("qwen4b")
    try:
        collection = {
            "selection": collect_event_field(model, selection, "selection", 8),
            "composition": collect_event_field(model, composition, "composition", 8),
            "reference": collect_reference(model, reference_rows(rows)),
        }
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    arrays = []
    for key in ("selection", "composition", "reference"):
        values = np.load(collection[key]["path"], mmap_mode="r")
        arrays.append({"key": key, "shape": list(values.shape), "finite_sample": bool(np.isfinite(values.reshape(-1)[::max(1, values.size // 100000)]).all())})
        close(values)
    raw_bytes = sum(item["bytes"] for item in collection.values())
    audit = {"model": label, "rows": len(rows), "qpoints": 38, "dimension": 2560,
             "raw_bytes": raw_bytes, "raw_gib": raw_bytes / (1024 ** 3), "arrays": arrays,
             "full_coordinate_primary": True, "topk_or_compression": False}
    checks = {"all_rows": len(rows) == 3072, "selection_shape": collection["selection"]["shape"] == [2048, 38, 8, 2560],
              "composition_shape": collection["composition"]["shape"] == [1024, 38, 12, 2560],
              "reference_rows": collection["reference"]["rows"] == 64, "finite": all(item["finite_sample"] for item in arrays),
              "full_coordinate": audit["full_coordinate_primary"] and not audit["topk_or_compression"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "audit": audit,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
