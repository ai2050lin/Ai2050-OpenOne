#!/usr/bin/env python3
"""Capture Qwen4B event/full-token state and event-aligned Attention/MLP full-coordinate fields."""
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
P2406 = RESULT / "phase2406_c24561_c24880_behavior_precision_calibration/qwen4b"
OUT = RESULT / "phase2407_c24881_c25200_qwen4b_component_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2407
CAMPAIGN = "C24881-C25200"

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
    if mmap is not None: mmap.close()


def index_row(row: dict) -> dict:
    keys = ("case_id", "task", "family", "unit", "language", "surface", "surface_class", "direction", "partition",
            "target_candidate_slot", "query_role", "steps", "answer", "foil", "prompt_token_count", "target_ids", "foil_ids")
    result = {key: row[key] for key in keys if key in row}
    result["event_names"] = [event["event"] for event in row["event_tokens"]]
    result["event_token_indices"] = [event["token_index"] for event in row["event_tokens"]]
    return result


def pad_right(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device); mask[index, :len(sequence)] = 1
    positions = mask.long().cumsum(-1) - 1; positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def open_field(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float16, shape=shape)


def collect_event_fields(model, rows: list[dict], task: str, batch_size: int = 4) -> dict:
    blocks = list(model.model.layers); states_modules = field_utils.modules(model)
    n, layers, events, dimension = len(rows), len(blocks), len(rows[0]["event_tokens"]), int(model.config.hidden_size)
    state_path = OUT / f"raw/{task}_state_event.float16.npy"
    attn_path = OUT / f"raw/{task}_attention_event.float16.npy"
    mlp_path = OUT / f"raw/{task}_mlp_event.float16.npy"
    progress = OUT / f"raw/{task}_progress.json"
    states = open_field(state_path, (n, layers + 2, events, dimension))
    attention = open_field(attn_path, (n, layers, events, dimension))
    mlp = open_field(mlp_path, (n, layers, events, dimension))
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    state_capture: dict[int, torch.Tensor] = {}; attn_capture: dict[int, torch.Tensor] = {}; mlp_capture: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(states_modules):
        def state_hook(_module, _inputs, output, qpoint=qpoint):
            state_capture[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(state_hook))
    for layer, block in enumerate(blocks):
        def attn_hook(_module, _inputs, output, layer=layer):
            attn_capture[layer] = (output[0] if isinstance(output, tuple) else output).detach()
        def mlp_hook(_module, _inputs, output, layer=layer):
            mlp_capture[layer] = output.detach()
        handles.append(block.self_attn.register_forward_hook(attn_hook))
        handles.append(block.mlp.register_forward_hook(mlp_hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, n, batch_size):
                batch = rows[start:start + batch_size]; ids, mask, positions = pad_right([row["prompt_ids"] for row in batch], device, pad)
                state_capture.clear(); attn_capture.clear(); mlp_capture.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                event_indices = [torch.tensor([event["token_index"] for event in row["event_tokens"]], dtype=torch.long) for row in batch]
                for qpoint in range(layers + 2):
                    tensor = state_capture[qpoint].float().cpu()
                    selected = torch.stack([tensor[local, event_indices[local]] for local in range(len(batch))])
                    states[start:start + len(batch), qpoint] = selected.numpy().astype(np.float16)
                for layer in range(layers):
                    for captures, output in ((attn_capture, attention), (mlp_capture, mlp)):
                        tensor = captures[layer].float().cpu()
                        selected = torch.stack([tensor[local, event_indices[local]] for local in range(len(batch))])
                        output[start:start + len(batch), layer] = selected.numpy().astype(np.float16)
                states.flush(); attention.flush(); mlp.flush(); save(progress, {"completed": start + len(batch)})
                if (start + len(batch)) % 128 == 0 or start + len(batch) == n:
                    print(f"[phase2407 {task}] {start + len(batch)}/{n}", flush=True)
                del ids, mask, positions
    finally:
        for handle in handles: handle.remove()
        states.flush(); attention.flush(); mlp.flush(); close(states); close(attention); close(mlp)
    return {"state": {"path": str(state_path), "shape": [n, layers + 2, events, dimension], "bytes": state_path.stat().st_size},
            "attention": {"path": str(attn_path), "shape": [n, layers, events, dimension], "bytes": attn_path.stat().st_size},
            "mlp": {"path": str(mlp_path), "shape": [n, layers, events, dimension], "bytes": mlp_path.stat().st_size}}


def reference_indices(rows: list[dict], task: str) -> list[int]:
    chosen = {}
    for index, row in enumerate(rows):
        key = (row["family"], row["language"], row["surface"], row["direction"])
        if key not in chosen: chosen[key] = index
    expected = 128 if task == "selection" else 64
    result = [chosen[key] for key in sorted(chosen)]
    if len(result) != expected: raise RuntimeError((task, len(result), expected))
    return result


def collect_all_token(model, task_rows: dict[str, list[dict]], batch_size: int = 2) -> dict:
    selected = [(task, index, task_rows[task][index]) for task in ("selection", "composition") for index in reference_indices(task_rows[task], task)]
    modules = field_utils.modules(model); qcount = len(modules); dimension = int(model.config.hidden_size)
    total = sum(len(row["prompt_ids"]) * qcount for _, _, row in selected)
    path = OUT / "raw/reference_all_token_state.float16.npy"; progress = OUT / "raw/reference_all_token_progress.json"
    values = open_field(path, (total, dimension)); completed_cases = int(json.loads(progress.read_text(encoding="utf-8"))["completed_cases"]) if progress.exists() else 0
    offsets = []; cursor = 0
    for task, source_index, row in selected[:completed_cases]:
        count = len(row["prompt_ids"]); begin = cursor; cursor += count * qcount
        offsets.append({"case_id": row["case_id"], "task": task, "source_index": source_index,
                        "offset": begin, "qpoints": qcount, "tokens": count, "prompt_ids": row["prompt_ids"]})
    captures = {}; handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, output, qpoint=qpoint): captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed_cases, len(selected), batch_size):
                batch = selected[start:start + batch_size]; ids, mask, positions = pad_right([row[2]["prompt_ids"] for row in batch], device, pad)
                captures.clear(); model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for local, (task, source_index, row) in enumerate(batch):
                    count = len(row["prompt_ids"]); begin = cursor
                    for qpoint in range(qcount):
                        tensor = captures[qpoint][local, :count].float().cpu().numpy().astype(np.float16)
                        values[cursor:cursor + count] = tensor; cursor += count
                    offsets.append({"case_id": row["case_id"], "task": task, "source_index": source_index,
                                    "offset": begin, "qpoints": qcount, "tokens": count, "prompt_ids": row["prompt_ids"]})
                values.flush(); save(progress, {"completed_cases": start + len(batch), "cursor": cursor})
                if (start + len(batch)) % 32 == 0 or start + len(batch) == len(selected):
                    print(f"[phase2407 all-token] {start + len(batch)}/{len(selected)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        values.flush(); close(values)
    write_rows(OUT / "index/reference_offsets.jsonl", offsets)
    if cursor != total: raise RuntimeError((cursor, total))
    return {"path": str(path), "shape": [total, dimension], "bytes": path.stat().st_size, "cases": len(selected), "qpoints": qcount}


def closure(task: str, collection: dict) -> dict:
    state = np.load(collection["state"]["path"], mmap_mode="r"); attention = np.load(collection["attention"]["path"], mmap_mode="r"); mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    indices = np.linspace(0, state.shape[0] - 1, min(64, state.shape[0]), dtype=int)
    truth = np.asarray(state[indices, 1:-1], dtype=np.float32) - np.asarray(state[indices, :-2], dtype=np.float32)
    pred = np.asarray(attention[indices], dtype=np.float32) + np.asarray(mlp[indices], dtype=np.float32)
    residual = truth - pred
    result = {"task": task, "values": int(truth.size), "mse": float(np.mean(residual * residual)),
              "relative_rmse": float(np.sqrt(np.sum(residual * residual) / max(np.sum(truth * truth), 1e-30))),
              "cosine": float(np.sum(truth * pred) / max(np.sqrt(np.sum(truth * truth) * np.sum(pred * pred)), 1e-30)),
              "finite": bool(np.isfinite(truth).all() and np.isfinite(pred).all())}
    close(state); close(attention); close(mlp); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen4B全坐标状态—Attention—MLP组件来源场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对全部1536条选择与640条组合材料，在8/12个语义事件采集embedding、36个block输出、final norm的全部2560物理坐标；并在每个block同步采集self-attention输出增量$A$与MLP输出增量$M$。另从每族×语言×四表面×双方向固定抽取192条，保留prompt全部token×全部38 qpoint×全部坐标。所有数组按原坐标顺序float16落盘；float16相对BF16源增加有效尾数但指数范围较小，必须检查finite和组件闭合，不做Top-K/PCA。

$$H_{{q+1,t}}=H_{{q,t}}+A_{{q,t}}+M_{{q,t}},\qquad U_{{q,t}}=A_{{q,t}}+M_{{q,t}}.$$

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；组件恒等检查 `{json.dumps(result['component_closure'], ensure_ascii=False)}`；总量`{result['total_gib']:.3f}` GiB；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2407_c24881_c25200_qwen4b_component_fullfield.py`；全事件state/attention/mlp、192条全token状态、索引和final位于`tests/glm5/result/phase2407_c24881_c25200_qwen4b_component_fullfield`。

**理论进展。** 该Phase第一次把整体更新拆成同一物理坐标上的两个真实加和来源，但不把Attention命名为搬运、MLP命名为语义。后续可逐坐标比较条件收益来自$A$、$M$还是二者在不同层/事件间接力，并用全token参考检查事件终点是否遗漏过程。

**问题硬伤与结论。** hook输出仍是组件输出而非单head、单MLP神经元内部计算；Attention输出已混合所有head，MLP输出已完成门控与down projection。float16落盘存在量化误差，组件恒等只能在该误差范围内核验。原始场只是物理来源图，不是功能归因或因果证据。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2406 / "index/operation_rows.jsonl"); tasks = {task: [row for row in rows if row["task"] == task] for task in ("selection", "composition")}
    for task, values in tasks.items(): write_rows(OUT / f"index/{task}_rows.jsonl", [index_row(row) for row in values])
    model, tokenizer, label = capability.load_model("qwen4b")
    try:
        collection = {task: collect_event_fields(model, values, task) for task, values in tasks.items()}
        collection["all_token"] = collect_all_token(model, tasks)
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    component_closure = {task: closure(task, collection[task]) for task in tasks}
    total_bytes = sum(item["bytes"] for task in ("selection", "composition") for item in collection[task].values()) + collection["all_token"]["bytes"]
    checks = {"selection_shapes": collection["selection"]["state"]["shape"] == [1536, 38, 8, 2560],
              "composition_shapes": collection["composition"]["state"]["shape"] == [640, 38, 12, 2560],
              "references": collection["all_token"]["cases"] == 192,
              "closure": all(value["finite"] and value["cosine"] > 0.999 for value in component_closure.values()),
              "no_topk": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": label, "collection": collection,
              "component_closure": component_closure, "total_bytes": total_bytes, "total_gib": total_bytes / 2**30,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
