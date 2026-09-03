#!/usr/bin/env python3
"""Qwen3-4B full-coordinate label-free input/output binding field capture."""
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
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
OUT = RESULT / "phase2379_c15921_c16240_qwen_label_free_full_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2378 / "material/label_free_natural_binding.jsonl"
PHASE = 2379
CAMPAIGN = "C15921-C16240"
BOUNDARY = OUT / "raw/qwen4b_prompt_boundary.float16.npy"
SOURCE = OUT / "raw/qwen4b_source_sentence_end.float16.npy"
OUTPUT = OUT / "raw/qwen4b_output_progress_anchors.float16.npy"
DECISIONS = OUT / "raw/qwen4b_sequence_scores.float32.npy"
ALL_TOKEN = OUT / "raw/qwen4b_reference_prompt_target_all_token.float16.npy"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def modules(model) -> list[Any]:
    embed = model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings()
    return [embed, *list(model.model.layers), model.model.norm]


def right_pad(sequences: list[list[int]], device: torch.device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[i, :len(sequence)] = 1
    positions = (mask.cumsum(1) - 1).clamp_min(0)
    return ids, mask, positions


def collect_boundary(model, rows: list[dict], batch_size: int = 8) -> dict:
    qmods = modules(model); shape = (len(rows), len(qmods), int(model.config.hidden_size))
    progress_path = OUT / "raw/boundary_progress.json"
    if BOUNDARY.exists() and progress_path.exists():
        field = np.lib.format.open_memmap(BOUNDARY, mode="r+")
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
    else:
        BOUNDARY.parent.mkdir(parents=True, exist_ok=True)
        field = np.lib.format.open_memmap(BOUNDARY, mode="w+", dtype=np.float16, shape=shape); completed = 0
    captures: dict[int, torch.Tensor] = {}; context: dict[str, Any] = {"lengths": []}
    handles = []
    for qi, module in enumerate(qmods):
        def hook(_module, _inputs, output, qi=qi):
            value = output[0] if isinstance(output, tuple) else output
            captures[qi] = torch.stack([value[i, length - 1] for i, length in enumerate(context["lengths"])]).detach().float().cpu()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]; sequences = [row["prompt_ids"] for row in batch]
                ids, mask, positions = right_pad(sequences, device, pad); context["lengths"] = list(map(len, sequences)); captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for qi in range(len(qmods)):
                    field[start:start + len(batch), qi] = captures[qi].numpy().astype(np.float16)
                field.flush(); save(progress_path, {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 256 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2379 boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        field.flush(); close(field)
    return {"shape": list(shape), "dtype": "float16", "rows": len(rows)}


def output_positions(row: dict) -> tuple[list[int], list[list[int]]]:
    prompt_length = len(row["prompt_ids"])
    source_ends = [span[1] - 1 for span in row["source_spans"]]
    anchors = []
    for start, end in row["target_spans"]:
        absolute_start, absolute_end = prompt_length + start, prompt_length + end
        anchors.append([absolute_start - 1, min(absolute_start + 2, absolute_end - 1), absolute_end - 1])
    return source_ends, anchors


def token_logprob(logits: torch.Tensor, sequence: list[int], prompt_length: int) -> tuple[float, float, int]:
    target = torch.tensor(sequence[prompt_length:], dtype=torch.long, device=logits.device)
    pred = logits[prompt_length - 1:len(sequence) - 1].float()
    selected = torch.log_softmax(pred, dim=-1).gather(1, target[:, None]).squeeze(1)
    return float(selected.mean()), float(selected.sum()), int(selected.numel())


def collect_teacher_anchors(model, rows: list[dict], batch_size: int = 2) -> dict:
    qmods = modules(model); dim = int(model.config.hidden_size); qcount = len(qmods)
    source_shape, output_shape = (len(rows), 4, qcount, dim), (len(rows), 4, 3, qcount, dim)
    progress_path = OUT / "raw/teacher_progress.json"
    if SOURCE.exists() and OUTPUT.exists() and DECISIONS.exists() and progress_path.exists():
        source_field = np.lib.format.open_memmap(SOURCE, mode="r+"); output_field = np.lib.format.open_memmap(OUTPUT, mode="r+")
        decisions = np.lib.format.open_memmap(DECISIONS, mode="r+")
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
    else:
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        source_field = np.lib.format.open_memmap(SOURCE, mode="w+", dtype=np.float16, shape=source_shape)
        output_field = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.float16, shape=output_shape)
        decisions = np.lib.format.open_memmap(DECISIONS, mode="w+", dtype=np.float32, shape=(len(rows), 7)); decisions[:] = np.nan
        completed = 0
    captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}; context: dict[str, Any] = {"sources": [], "anchors": []}
    handles = []
    for qi, module in enumerate(qmods):
        def hook(_module, _inputs, output, qi=qi):
            value = output[0] if isinstance(output, tuple) else output
            src = torch.stack([value[i, torch.tensor(pos, device=value.device)] for i, pos in enumerate(context["sources"])])
            out = torch.stack([value[i, torch.tensor(pos, device=value.device)] for i, pos in enumerate(context["anchors"])])
            captures[qi] = (src.detach().float().cpu(), out.detach().float().cpu())
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                sequences = [row["prompt_ids"] + row["target_ids"] for row in batch]
                pairs = [output_positions(row) for row in batch]
                context["sources"] = [pair[0] for pair in pairs]
                context["anchors"] = [pair[1] for pair in pairs]
                ids, mask, positions = right_pad(sequences, device, pad); captures.clear()
                result = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for qi in range(qcount):
                    src, out = captures[qi]
                    source_field[start:start + len(batch), :, qi] = src.numpy().astype(np.float16)
                    output_field[start:start + len(batch), :, :, qi] = out.numpy().astype(np.float16)
                for local, (row, sequence) in enumerate(zip(batch, sequences)):
                    mean, total, count = token_logprob(result.logits[local], sequence, len(row["prompt_ids"]))
                    decisions[start + local, 0] = mean; decisions[start + local, 3] = total; decisions[start + local, 5] = count
                source_field.flush(); output_field.flush(); decisions.flush()
                save(progress_path, {"completed": start + len(batch), "source_shape": source_shape, "output_shape": output_shape})
                if (start + len(batch)) % 128 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2379 teacher] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        source_field.flush(); output_field.flush(); decisions.flush(); close(source_field); close(output_field); close(decisions)
    return {"source_shape": list(source_shape), "output_shape": list(output_shape), "offsets": ["pre_sentence", "early_token_2", "sentence_end"]}


def collect_foil_scores(model, rows: list[dict], batch_size: int = 4) -> dict:
    decisions = np.lib.format.open_memmap(DECISIONS, mode="r+")
    progress_path = OUT / "raw/foil_progress.json"
    completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"]) if progress_path.exists() else 0
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]; sequences = [row["prompt_ids"] + row["foil_ids"] for row in batch]
                ids, mask, positions = right_pad(sequences, device, pad)
                result = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for local, (row, sequence) in enumerate(zip(batch, sequences)):
                    mean, total, count = token_logprob(result.logits[local], sequence, len(row["prompt_ids"]))
                    index = start + local; decisions[index, 1] = mean; decisions[index, 2] = float(decisions[index, 0]) - mean
                    decisions[index, 4] = total; decisions[index, 6] = count
                decisions.flush(); save(progress_path, {"completed": start + len(batch)})
                if (start + len(batch)) % 256 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2379 foil] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        decisions.flush(); close(decisions)
    values = np.load(DECISIONS, mmap_mode="r")
    return {"rows": len(rows), "target_mean_logprob_over_foil": float(np.mean(np.asarray(values[:, 2]) > 0)),
            "mean_per_token_margin": float(np.mean(values[:, 2])), "finite": bool(np.isfinite(values).all())}


def reference_indices(rows: list[dict]) -> list[int]:
    selected = []
    for fi, family in enumerate(sorted({row["family"] for row in rows})):
        for language in ("en", "zh"):
            matches = [i for i, row in enumerate(rows) if row["family"] == family and row["language"] == language
                       and row["unit"] == 6 + (fi % 2) and row["surface"] == "shared_opening"
                       and row["reverse"] == bool(fi % 2) and row["source_index"] == 4 + (fi % 2)]
            if len(matches) != 1: raise RuntimeError((family, language, len(matches)))
            selected.append(matches[0])
    return selected


def collect_all_token(model, tokenizer, rows: list[dict]) -> dict:
    indices = reference_indices(rows); selected = [rows[index] for index in indices]; qmods = modules(model)
    sequences = [row["prompt_ids"] + row["target_ids"] for row in selected]
    shape = (len(selected), len(qmods), max(map(len, sequences)), int(model.config.hidden_size))
    progress_path = OUT / "raw/all_token_progress.json"
    if ALL_TOKEN.exists() and progress_path.exists():
        field = np.lib.format.open_memmap(ALL_TOKEN, mode="r+")
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
    else:
        field = np.lib.format.open_memmap(ALL_TOKEN, mode="w+", dtype=np.float16, shape=shape); completed = 0
    captures: dict[int, torch.Tensor] = {}; handles = []
    for qi, module in enumerate(qmods):
        def hook(_module, _inputs, output, qi=qi):
            captures[qi] = (output[0] if isinstance(output, tuple) else output)[0].detach().float().cpu()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    index_rows = []
    try:
        with torch.inference_mode():
            for local in range(completed, len(selected)):
                sequence = sequences[local]; ids = torch.tensor([sequence], dtype=torch.long, device=device); captures.clear()
                model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(len(sequence), device=device)[None],
                      use_cache=False, return_dict=True)
                for qi in range(len(qmods)):
                    field[local, qi, :len(sequence)] = captures[qi].numpy().astype(np.float16)
                field.flush(); save(progress_path, {"completed": local + 1, "shape": shape})
                print(f"[phase2379 all-token] {local + 1}/{len(selected)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        field.flush(); close(field)
    for source_index, row, sequence in zip(indices, selected, sequences):
        source_ends, anchors = output_positions(row)
        index_rows.append({"source_index": source_index, "case_id": row["case_id"], "family": row["family"],
                           "language": row["language"], "unit": row["unit"], "surface": row["surface"],
                           "reverse": row["reverse"], "source_perm": row["source_perm"], "token_count": len(sequence),
                           "prompt_token_count": len(row["prompt_ids"]), "source_spans": row["source_spans"],
                           "source_end_positions": source_ends, "output_anchor_positions": anchors,
                           "token_ids": sequence, "tokens": [tokenizer.decode([token]) for token in sequence]})
    write_rows(OUT / "index/reference_all_token_rows.jsonl", index_rows)
    return {"shape": list(shape), "valid_tokens": sum(len(sequence) for sequence in sequences), "rows": len(selected)}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-4B无标签自然绑定全坐标输入－输出场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Phase2378全部3584条自然prompt上采集prompt末端的词嵌入、36个block和final norm全部2560个物理坐标；在3072条exact-copy上进一步采集四个来源句末端，以及每个目标句的输出前、早期token、句末三个锚点的全层全坐标场。另保存16条fresh联合锁箱的prompt+teacher-forced输出全部token场。序列行为门比较完整目标排列与循环错序foil的逐token平均对数概率，不用相同首token代替完整序列。无Top-K、PCA或压缩表示。

$$\mathcal H^{{src}}_{{r,k,q,j}}=H_q(x_r)[e_k,j],\quad
\mathcal H^{{out}}_{{r,t,o,q,j}}=H_q(x_r\oplus y_r)[a_{{t,o}},j],\quad
\Delta\bar\ell_r={{1\over |y|}}\log p(y|x)-{{1\over|\tilde y|}}\log p(\tilde y|x).$$

**结果汇总。** prompt边界 `{json.dumps(result['boundary'], ensure_ascii=False)}`；来源/输出锚点 `{json.dumps(result['teacher'], ensure_ascii=False)}`；完整序列行为 `{json.dumps(result['sequence_behavior'], ensure_ascii=False)}`；all-token场 `{json.dumps(result['all_token'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2379_c15921_c16240_qwen_label_free_full_field.py`；原场、索引与进度位于 `tests/glm5/result/phase2379_c15921_c16240_qwen_label_free_full_field`。原场将在重要参数级热力图发布并校验后清理。

**理论进展、问题硬伤与结论。** 本Phase建立观察底座，不从高维可分性直接推出对象或指针。来源句末端含有累积上下文，输出句末端还含目标词汇，二者都可能造成浅层匹配；所以下一Phase必须分别报告pre-sentence、early、end，并用错误来源句、坐标置乱、分层标签置乱和fresh unit+source联合锁箱裁决。完整序列target-over-foil也只说明模型更偏好正确顺序，不能替代自主生成中的内容保持。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    all_rows = read_rows(MATERIAL); exact_rows = [row for row in all_rows if row["task"] == "exact_copy"]
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        boundary = collect_boundary(model, all_rows)
        teacher = collect_teacher_anchors(model, exact_rows)
        sequence_behavior = collect_foil_scores(model, exact_rows)
        all_token = collect_all_token(model, tokenizer, exact_rows)
    finally:
        model_utils.release_model(model); del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    checks = {
        "boundary_shape": boundary["shape"] == [3584, 38, 2560],
        "source_shape": teacher["source_shape"] == [3072, 4, 38, 2560],
        "output_shape": teacher["output_shape"] == [3072, 4, 3, 38, 2560],
        "all_token_shape": all_token["shape"][0] == 16 and all_token["shape"][1] == 38 and all_token["shape"][-1] == 2560,
        "scores_finite": sequence_behavior["finite"] and math.isfinite(sequence_behavior["mean_per_token_margin"]),
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B-BF16", "boundary": boundary,
              "teacher": teacher, "sequence_behavior": sequence_behavior, "all_token": all_token,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
