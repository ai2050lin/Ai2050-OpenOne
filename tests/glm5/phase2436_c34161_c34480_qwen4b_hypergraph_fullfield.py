#!/usr/bin/env python3
"""Capture Qwen3-4B BF16 full-coordinate event and all-token hypergraph fields."""
from __future__ import annotations

import gc
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
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior/qwen4b"
OUT = RESULT / "phase2436_c34161_c34480_qwen4b_hypergraph_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2436
CAMPAIGN = "C34161-C34480"

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


def compact(row: dict, include_ids: bool = False) -> dict:
    keys = ("case_id", "config_id", "family", "unit", "language", "surface", "surface_class",
            "direction", "variant", "query_role", "partition", "target_candidate_slot", "answer", "foil",
            "prompt_token_count")
    result = {key: row[key] for key in keys}
    result["event_names"] = [event["event"] for event in row["event_tokens"]]
    result["event_token_indices"] = [event["token_index"] for event in row["event_tokens"]]
    if include_ids:
        result["prompt_ids"] = row["prompt_ids"]
        result["target_ids"] = row["target_ids"]
    return result


def hooks(model):
    modules = field_utils.modules(model)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    return modules, captures, handles


def collect_events(model, rows: list[dict], batch_size: int = 4) -> dict:
    modules, captures, handles = hooks(model)
    d = int(model.get_input_embeddings().weight.shape[1])
    e = len(rows[0]["event_tokens"])
    shape = (len(rows), len(modules), e, d)
    path = OUT / "raw/hypergraph_event_field.float16.npy"
    progress = OUT / "raw/hypergraph_event_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and progress.exists():
        values = np.lib.format.open_memmap(path, mode="r+")
        completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"])
        if tuple(values.shape) != shape:
            raise RuntimeError(("stale_event_shape", values.shape, shape))
    else:
        values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        completed = 0
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
                    selected = torch.stack([
                        captures[qpoint][local, torch.tensor(
                            [event["token_index"] for event in row["event_tokens"]],
                            dtype=torch.long, device=captures[qpoint].device)]
                        for local, row in enumerate(batch)
                    ])
                    values[start:start + len(batch), qpoint] = selected.float().cpu().numpy().astype(np.float16)
                values.flush()
                save(progress, {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 128 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2436 event] {start + len(batch)}/{len(rows)}", flush=True)
                del ids, mask, positions, selected
    finally:
        for handle in handles:
            handle.remove()
        values.flush(); close(values)
    return {"path": str(path), "shape": list(shape), "bytes": path.stat().st_size,
            "dtype": "float16", "event_names": [x["event"] for x in rows[0]["event_tokens"]]}


def reference_rows(rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if row["variant"] == "valid" and int(row["unit"]) == 5
                and row["surface"] == "natural"]
    if len(selected) != 64:
        raise RuntimeError(("reference_rows", len(selected)))
    return selected


def collect_all_tokens(model, rows: list[dict]) -> dict:
    sequences = [row["prompt_ids"] + row["target_ids"] for row in rows]
    width = max(map(len, sequences))
    modules, captures, handles = hooks(model)
    d = int(model.get_input_embeddings().weight.shape[1])
    shape = (len(rows), len(modules), width, d)
    field_path = OUT / "raw/fresh_valid_prompt_answer_all_token.float16.npy"
    mask_path = OUT / "raw/fresh_valid_prompt_answer_mask.uint8.npy"
    progress = OUT / "raw/fresh_valid_all_token_progress.json"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    if field_path.exists() and mask_path.exists() and progress.exists():
        values = np.lib.format.open_memmap(field_path, mode="r+")
        valid = np.lib.format.open_memmap(mask_path, mode="r+")
        completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"])
        if tuple(values.shape) != shape:
            raise RuntimeError(("stale_token_shape", values.shape, shape))
    else:
        values = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16, shape=shape)
        valid = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(len(rows), width))
        completed = 0
    device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for index in range(completed, len(rows)):
                sequence = sequences[index]
                ids = torch.tensor([sequence], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                positions = torch.arange(len(sequence), device=device)[None]
                captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for qpoint in range(len(modules)):
                    values[index, qpoint, :len(sequence)] = captures[qpoint][0].float().cpu().numpy().astype(np.float16)
                valid[index, :len(sequence)] = 1
                values.flush(); valid.flush()
                save(progress, {"completed": index + 1, "shape": shape, "valid_tokens": int(sum(map(len, sequences[:index + 1])))})
                if (index + 1) % 8 == 0 or index + 1 == len(rows):
                    print(f"[phase2436 all-token] {index + 1}/{len(rows)}", flush=True)
                del ids, mask, positions
    finally:
        for handle in handles:
            handle.remove()
        values.flush(); valid.flush(); close(values); close(valid)
    write_rows(OUT / "index/fresh_valid_all_token_rows.jsonl", [compact(row, True) for row in rows])
    return {"path": str(field_path), "mask_path": str(mask_path), "shape": list(shape),
            "bytes": field_path.stat().st_size + mask_path.stat().st_size, "dtype": "float16",
            "rows": len(rows), "width": width, "valid_tokens": int(sum(map(len, sequences)))}


def sampled_finite(path: str) -> bool:
    values = np.load(path, mmap_mode="r")
    sample = values.reshape(-1)[::max(1, values.size // 200000)]
    result = bool(np.isfinite(sample).all())
    close(values)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 八语言模式族事件与全token全坐标HiddenState场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2435通过材料门的Qwen3-4B BF16全部2304条prompt，采集embedding、36个Transformer block输出、final norm，共38个checkpoint；每条同时取prefix、operation、argument、context、query、两个候选和answer boundary八个真实token事件，保留2560个物理坐标，不做Top-K、PCA、阈值压缩。另将fresh unit、valid、natural的64条prompt+答案逐token全量保存，用于检验离散事件是否漏掉关键脉络。

$$\mathcal H_{{r,q,e,j}}=H_q(x_r)[t_{{r,e}},j],\quad
r=1\ldots2304,\ q=0\ldots37,\ e=1\ldots8,\ j=1\ldots2560.$$

$$U_{{r,q,e,j}}=\mathcal H_{{r,q+1,e,j}}-\mathcal H_{{r,q,e,j}},\quad q=0\ldots36.$$

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2436_c34161_c34480_qwen4b_hypergraph_fullfield.py`；全坐标`.npy`、进度、索引与final位于`tests/glm5/result/phase2436_c34161_c34480_qwen4b_hypergraph_fullfield`。

**分析与理论进展。** 本Phase只建立完整测量对象，不从幅值直接命名“齿轮”。embedding和全部HiddenState使用同一物理坐标编号，下一Phase可以问同一外部操作的有符号坐标组合何时出现、跨层是否保持、在语言/表述/方向变化中复用或分化。64条全token场使事件锚点假设本身可被反驳。

**问题硬伤与结论。** float16是存储精度而非模型推理精度；模型推理为BF16，低值坐标仍全部保留，但约低于float16分辨率的极微小差异可能量化。final norm是输出接口观察点，不等同第37个残差block。合成材料和单模型仍限制外推；DS7B行为门不合格，不能用其内部场支持成功语言计算。唯一原始场不清理，后续重要全坐标派生量将进入可视化客户端。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    write_rows(OUT / "index/hypergraph_event_rows.jsonl", [compact(row) for row in rows])
    refs = reference_rows(rows)
    model, tokenizer, label = capability.load_model("qwen4b")
    try:
        collection = {"event": collect_events(model, rows, 4), "all_token": collect_all_tokens(model, refs)}
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    finite = {key: sampled_finite(value["path"]) for key, value in collection.items()}
    raw_bytes = sum(value["bytes"] for value in collection.values())
    audit = {"model": label, "inference_precision": "BF16", "storage_precision": "float16",
             "rows": len(rows), "qpoints": 38, "events": 8, "dimension": 2560,
             "raw_bytes": raw_bytes, "raw_gib": raw_bytes / 1024 ** 3,
             "finite_sample": finite, "full_coordinate_primary": True,
             "topk_pca_threshold_primary": False, "unique_raw_retained": True}
    checks = {"rows_2304": len(rows) == 2304,
              "event_shape": collection["event"]["shape"] == [2304, 38, 8, 2560],
              "all_token_rows_64": collection["all_token"]["rows"] == 64,
              "qpoints_38": collection["all_token"]["shape"][1:2] == [38],
              "dimension_2560": collection["all_token"]["shape"][-1] == 2560,
              "finite": all(finite.values()), "full_coordinates": audit["full_coordinate_primary"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "audit": audit,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
