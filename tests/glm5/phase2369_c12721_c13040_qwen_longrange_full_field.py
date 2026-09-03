#!/usr/bin/env python3
"""Collect Qwen3-4B long-range and flagship embedding/HiddenState fields."""
from __future__ import annotations

import gc
import json
import math
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
P2368 = RESULT / "phase2368_c12481_c12720_longrange_operator_contract"
OUT = RESULT / "phase2369_c12721_c13040_qwen_longrange_full_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
LONG_MATERIAL = P2368 / "material/long_sentence_permutation.jsonl"
FLAGSHIP_MATERIAL = P2368 / "material/flagship_factorial.jsonl"
LONG_STATES = OUT / "raw/qwen4b_long_boundary_all_layers.float16.npy"
LONG_DECISIONS = OUT / "raw/qwen4b_long_first_token_decisions.float32.npy"
LONG_PROGRESS = OUT / "raw/long_progress.json"
FLAG_STATES = OUT / "raw/qwen4b_flagship_boundary_all_layers.float16.npy"
FLAG_DECISIONS = OUT / "raw/qwen4b_flagship_first_token_decisions.float32.npy"
FLAG_PROGRESS = OUT / "raw/flagship_progress.json"
TOKEN_FIELD = OUT / "raw/qwen4b_long_reference_all_token_all_layers.float16.npy"
TOKEN_INDEX = OUT / "index/long_reference_all_token_rows.jsonl"
TOKEN_PROGRESS = OUT / "raw/token_progress.json"
PHASE = 2369
CAMPAIGN = "C12721-C13040"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, Path): return str(value)
    if isinstance(value, Counter): return dict(value)
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def qmodules(model) -> list[Any]:
    embed = model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings()
    return [embed, *list(model.model.layers), model.model.norm]


def left_pad(sequences: list[list[int]], device: torch.device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, -len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, -len(seq):] = 1
    return ids, mask, (mask.cumsum(1) - 1).clamp_min(0)


def collect_boundary(model, rows: list[dict], states_path: Path, decisions_path: Path,
                     progress_path: Path, label: str, batch_size: int) -> dict:
    modules = qmodules(model)
    dim = int(model.config.hidden_size)
    shape = (len(rows), len(modules), dim)
    if states_path.exists() and decisions_path.exists() and progress_path.exists():
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(states_path, mode="r+")
        decisions = np.lib.format.open_memmap(decisions_path, mode="r+")
    else:
        completed = 0
        states_path.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(states_path, mode="w+", dtype=np.float16, shape=shape)
        decisions = np.lib.format.open_memmap(decisions_path, mode="w+", dtype=np.float32, shape=(len(rows), 6))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qi, module in enumerate(modules):
        def hook(_module, _inputs, output, qi=qi):
            value = output[0] if isinstance(output, tuple) else output
            captures[qi] = value[:, -1].detach()
        handles.append(module.register_forward_hook(hook))
    device = next(model.parameters()).device
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = left_pad([r["prompt_ids"] for r in batch], device, pad)
                captures.clear()
                output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                               use_cache=False, return_dict=True)
                for qi in range(len(modules)):
                    states[start:start + len(batch), qi] = captures[qi].float().cpu().numpy().astype(np.float16)
                logits = output.logits[:, -1].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                for local, row in enumerate(batch):
                    target, foil = int(row["target_first_id"]), int(row["foil_first_id"])
                    tl, fl = float(logits[local, target]), float(logits[local, foil])
                    decisions[start + local] = [tl, fl, tl - fl, float(tl > fl),
                                                float(int(logits[local].argmax()) == target), float(log_probs[local, target])]
                states.flush(); decisions.flush()
                save(progress_path, {"completed": start + len(batch), "shape": shape, "batch_size": batch_size})
                if (start + len(batch)) % 256 == 0 or start + len(batch) == len(rows):
                    print(f"[{label}] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        states.flush(); decisions.flush(); close(states); close(decisions)
    return {"shape": list(shape), "dtype": "float16", "batch_size": batch_size, "rows": len(rows)}


def reference_indices(rows: list[dict]) -> list[int]:
    selected = []
    for fi, family in enumerate(sorted(set(r["family"] for r in rows))):
        for language in ("en", "zh"):
            desired_source = list((0, 1, 2, 3) if fi % 2 == 0 else (2, 0, 3, 1))
            desired_perm = list((3, 2, 1, 0) if fi % 2 == 0 else (1, 0, 2, 3))
            matches = [i for i, r in enumerate(rows) if r["family"] == family and r["unit"] == 5
                       and r["language"] == language and r["task"] == "exact_copy"
                       and r["source_perm"] == desired_source and r["target_perm"] == desired_perm]
            if len(matches) != 1: raise RuntimeError((family, language, len(matches)))
            selected.append(matches[0])
    return selected


def subsequence_positions(sequence: list[int], needle: list[int]) -> list[int]:
    return [i for i in range(len(sequence) - len(needle) + 1) if sequence[i:i + len(needle)] == needle]


def collect_all_token(model, tokenizer, rows: list[dict]) -> dict:
    indices = reference_indices(rows)
    selected = [rows[i] for i in indices]
    modules = qmodules(model)
    dim = int(model.config.hidden_size)
    max_tokens = max(len(r["prompt_ids"]) for r in selected)
    shape = (len(selected), len(modules), max_tokens, dim)
    if TOKEN_FIELD.exists() and TOKEN_PROGRESS.exists():
        completed = int(json.loads(TOKEN_PROGRESS.read_text(encoding="utf-8"))["completed"])
        field = np.lib.format.open_memmap(TOKEN_FIELD, mode="r+")
    else:
        completed = 0
        TOKEN_FIELD.parent.mkdir(parents=True, exist_ok=True)
        field = np.lib.format.open_memmap(TOKEN_FIELD, mode="w+", dtype=np.float16, shape=shape)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qi, module in enumerate(modules):
        def hook(_module, _inputs, output, qi=qi):
            captures[qi] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            for local in range(completed, len(selected)):
                row = selected[local]
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                captures.clear()
                model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(ids.shape[1], device=device)[None],
                      use_cache=False, return_dict=True)
                for qi in range(len(modules)):
                    field[local, qi, :ids.shape[1]] = captures[qi][0].float().cpu().numpy().astype(np.float16)
                field.flush(); save(TOKEN_PROGRESS, {"completed": local + 1, "shape": shape})
                print(f"[phase2369 all-token] {local + 1}/{len(selected)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        field.flush(); close(field)
    index_rows = []
    for source_index, row in zip(indices, selected):
        marker_positions = {}
        for marker in row["markers"]:
            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
            marker_positions[marker] = subsequence_positions(row["prompt_ids"], marker_ids)[:4]
        index_rows.append({
            "source_index": source_index, "case_id": row["case_id"], "family": row["family"],
            "unit": row["unit"], "language": row["language"], "task": row["task"],
            "source_perm": row["source_perm"], "target_perm": row["target_perm"], "markers": row["markers"],
            "marker_token_positions": marker_positions, "token_count": len(row["prompt_ids"]),
            "token_ids": row["prompt_ids"], "tokens": [tokenizer.decode([x]) for x in row["prompt_ids"]],
        })
    write_rows(TOKEN_INDEX, index_rows)
    return {"shape": list(shape), "rows": len(selected), "max_tokens": max_tokens,
            "valid_tokens": sum(r["token_count"] for r in index_rows)}


def summarize_behavior(rows: list[dict], decisions_path: Path, dimensions: list[str]) -> dict:
    d = np.load(decisions_path, mmap_mode="r")
    result = {
        "rows": len(rows), "target_over_foil": float(np.asarray(d[:, 3]).mean()),
        "full_vocab_argmax_target": float(np.asarray(d[:, 4]).mean()),
        "mean_target_first_logprob": float(np.asarray(d[:, 5]).mean()),
        "warning": "First-token gates qualify materials; they do not prove complete reorder or semantic preservation.",
    }
    for dimension in dimensions:
        values = {}
        for value in sorted(set(str(r[dimension]) for r in rows)):
            idx = [i for i, r in enumerate(rows) if str(r[dimension]) == value]
            values[value] = {"n": len(idx), "target_over_foil": float(np.asarray(d[idx, 3]).mean()),
                             "argmax": float(np.asarray(d[idx, 4]).mean())}
        result[f"by_{dimension}"] = values
    close(d)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-4B长距离全层全坐标场与双旗舰原始场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2368冻结的全部{result['long_collection']['rows']}条长句prompt和{result['flagship_collection']['rows']}条双旗舰prompt，采集词嵌入输出、36个Transformer block输出和final norm的边界token全部2560个物理激活坐标；另对16条fresh-unit长句采集每个输入token×38检查点×2560坐标。未用Top-K、PCA或均值场代替原始场。

$$
\mathcal H_{{r,q,j}}=H_q(x_r)[-1,j],\qquad
m_r=z_{{y_r}}-z_{{\tilde y_r}},\qquad q=0,\ldots,37, j=0,\ldots,2559.
$$

**结果汇总。** 长句采集 `{json.dumps(result['long_collection'], ensure_ascii=False)}`；长句行为门 `{json.dumps(result['long_behavior'], ensure_ascii=False)}`；全token参考场 `{json.dumps(result['token_collection'], ensure_ascii=False)}`；双旗舰采集 `{json.dumps(result['flagship_collection'], ensure_ascii=False)}`；双旗舰行为门 `{json.dumps(result['flagship_behavior'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2369_c12721_c13040_qwen_longrange_full_field.py`；原始场、索引和分析位于 `tests/glm5/result/phase2369_c12721_c13040_qwen_longrange_full_field`。这些原始场将在后续重要热力图发布后清理。

**理论进展、问题硬伤与结论。** 本Phase只建立可追溯的参数级激活观测底座，不把target-over-foil或全词表首token命中解释为完整重排。长任务的首token门尤其不能替代顺序精确率、内容保持率和自由生成分叉测试。下一Phase只在行为合格分层内竞争位置模板、句对象、指针状态和$S_4$生成元组合律；若语义排序不合格，仍保留精确复制/索引任务中通过的结构继续研究。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists() and json.loads(final_path.read_text(encoding="utf-8")).get("all_checks_passed"):
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(LONG_MATERIAL)
    flag_rows = read_rows(FLAGSHIP_MATERIAL)
    model, tokenizer, _device = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        long_collection = collect_boundary(model, rows, LONG_STATES, LONG_DECISIONS, LONG_PROGRESS, "phase2369 long", 8)
        token_collection = collect_all_token(model, tokenizer, rows)
        flagship_collection = collect_boundary(model, flag_rows, FLAG_STATES, FLAG_DECISIONS, FLAG_PROGRESS, "phase2369 flagship", 16)
    finally:
        model_utils.release_model(model); del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    long_behavior = summarize_behavior(rows, LONG_DECISIONS, ["task", "language", "unit_partition", "permutation_partition"])
    flag_behavior = summarize_behavior(flag_rows, FLAG_DECISIONS, ["system", "language", "surface", "partition"])
    checks = {
        "long_shape": long_collection["shape"] == [7104, 38, 2560],
        "flagship_shape": flagship_collection["shape"] == [3072, 38, 2560],
        "token_shape": token_collection["shape"][0] == 16 and token_collection["shape"][1:2] == [38]
                       and token_collection["shape"][-1] == 2560,
        "finite_long_behavior": math.isfinite(long_behavior["target_over_foil"]),
        "finite_flag_behavior": math.isfinite(flag_behavior["target_over_foil"]),
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B-bfloat16",
              "long_collection": long_collection, "long_behavior": long_behavior,
              "token_collection": token_collection, "flagship_collection": flagship_collection,
              "flagship_behavior": flag_behavior, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
