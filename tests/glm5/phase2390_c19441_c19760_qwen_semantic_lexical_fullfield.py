#!/usr/bin/env python3
"""Collect Qwen4B/Qwen14B full-coordinate semantic-versus-lexical fields."""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2388 = RESULT / "phase2388_c18801_c19120_semantic_lexical_contract"
P2389 = RESULT / "phase2389_c19121_c19440_crossmodel_autonomous_capability"
OUT = RESULT / "phase2390_c19441_c19760_qwen_semantic_lexical_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2390
CAMPAIGN = "C19441-C19760"
MODEL_ORDER = ("qwen4b", "qwen14b")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402


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


def modules(model) -> list[Any]:
    embed = model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings()
    return [embed, *list(model.model.layers), model.model.norm]


def paths(key: str) -> dict[str, Path]:
    base = OUT / key
    return {
        "base": base,
        "independent_end": base / "raw/independent_end.float16.npy",
        "independent_mean": base / "raw/independent_mean.float16.npy",
        "boundary": base / "raw/semantic_selection_prompt_boundary.float16.npy",
        "decisions": base / "raw/semantic_selection_sequence_scores.float32.npy",
        "reference": base / "raw/reference_prompt_target_all_token.float16.npy",
        "reference_mask": base / "raw/reference_prompt_target_mask.uint8.npy",
        "progress_independent": base / "raw/progress_independent.json",
        "progress_boundary": base / "raw/progress_boundary.json",
        "progress_scores": base / "raw/progress_scores.json",
        "rows_independent": base / "index/independent_rows.jsonl",
        "rows_selection": base / "index/selection_rows.jsonl",
        "final": base / "analysis/final.json",
    }


def pad_right(sequences: list[list[int]], device: torch.device, pad: int):
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device); mask[index, :len(sequence)] = 1
    positions = (mask.cumsum(1) - 1).clamp_min(0)
    return ids, mask, positions


def compile_independent(tokenizer, rows: list[dict]) -> list[dict]:
    result = []
    for index, row in enumerate(rows):
        ids = [int(value) for value in tokenizer.encode(row["text"], add_special_tokens=False)]
        result.append({**row, "model_row": index, "token_ids": ids, "token_count": len(ids)})
    return result


def compile_selection(tokenizer, rows: list[dict]) -> list[dict]:
    result = []
    for index, row in enumerate(rows):
        prompt_ids = capability.chat_ids(tokenizer, row["prompt"])
        target_ids = [int(value) for value in tokenizer.encode(row["target"], add_special_tokens=False)]
        foil_ids = [int(value) for value in tokenizer.encode(row["foil"], add_special_tokens=False)]
        result.append({**row, "model_row": index, "prompt_ids": prompt_ids, "target_ids": target_ids, "foil_ids": foil_ids,
                       "prompt_tokens": len(prompt_ids), "target_tokens": len(target_ids), "foil_tokens": len(foil_ids)})
    return result


def collect_independent(key: str, model, rows: list[dict], batch_size: int) -> dict:
    p = paths(key); qmods = modules(model); dim = int(model.get_input_embeddings().weight.shape[1]); shape = (len(rows), len(qmods), dim)
    if p["independent_end"].exists() and p["independent_mean"].exists():
        end = np.lib.format.open_memmap(p["independent_end"], mode="r+"); mean = np.lib.format.open_memmap(p["independent_mean"], mode="r+")
        completed = int(json.loads(p["progress_independent"].read_text(encoding="utf-8"))["completed"])
    else:
        p["independent_end"].parent.mkdir(parents=True, exist_ok=True)
        end = np.lib.format.open_memmap(p["independent_end"], mode="w+", dtype=np.float16, shape=shape)
        mean = np.lib.format.open_memmap(p["independent_mean"], mode="w+", dtype=np.float16, shape=shape); completed = 0
    captures: dict[int, torch.Tensor] = {}; handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]; ids, mask, positions = pad_right([row["token_ids"] for row in batch], device, pad); captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                lengths = mask.sum(1).tolist()
                for qpoint in range(len(qmods)):
                    value = captures[qpoint].float().cpu()
                    end[start:start + len(batch), qpoint] = torch.stack([value[i, length - 1] for i, length in enumerate(lengths)]).numpy().astype(np.float16)
                    mean[start:start + len(batch), qpoint] = torch.stack([value[i, :length].mean(0) for i, length in enumerate(lengths)]).numpy().astype(np.float16)
                end.flush(); mean.flush(); save(p["progress_independent"], {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 192 == 0 or start + len(batch) == len(rows): print(f"[phase2390 {key} independent] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        end.flush(); mean.flush(); close(end); close(mean)
    return {"shape": list(shape), "token_range": [min(row["token_count"] for row in rows), max(row["token_count"] for row in rows)]}


def collect_boundary(key: str, model, rows: list[dict], batch_size: int) -> dict:
    p = paths(key); qmods = modules(model); dim = int(model.get_input_embeddings().weight.shape[1]); shape = (len(rows), len(qmods), dim)
    if p["boundary"].exists():
        field = np.lib.format.open_memmap(p["boundary"], mode="r+"); completed = int(json.loads(p["progress_boundary"].read_text(encoding="utf-8"))["completed"])
    else:
        p["boundary"].parent.mkdir(parents=True, exist_ok=True); field = np.lib.format.open_memmap(p["boundary"], mode="w+", dtype=np.float16, shape=shape); completed = 0
    captures: dict[int, torch.Tensor] = {}; handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, output, qpoint=qpoint): captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]; ids, mask, positions = pad_right([row["prompt_ids"] for row in batch], device, pad); captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True); ends = mask.sum(1) - 1
                for qpoint in range(len(qmods)):
                    value = captures[qpoint].float().cpu(); field[start:start + len(batch), qpoint] = torch.stack([value[i, ends[i]] for i in range(len(batch))]).numpy().astype(np.float16)
                field.flush(); save(p["progress_boundary"], {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 96 == 0 or start + len(batch) == len(rows): print(f"[phase2390 {key} boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        field.flush(); close(field)
    return {"shape": list(shape), "prompt_token_range": [min(row["prompt_tokens"] for row in rows), max(row["prompt_tokens"] for row in rows)]}


def sequence_mean_logprob(model, prompt: list[int], continuation: list[int]) -> float:
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prompt + continuation], dtype=torch.long, device=device)
    with torch.inference_mode(): logits = model(input_ids=ids, use_cache=False, return_dict=True).logits.float()
    start = len(prompt) - 1; stop = start + len(continuation)
    selected = torch.log_softmax(logits[0, start:stop], dim=-1).gather(1, torch.tensor(continuation, device=device)[:, None]).squeeze(1)
    return float(selected.mean().item())


def collect_scores(key: str, model, rows: list[dict]) -> dict:
    p = paths(key); shape = (len(rows), 3)
    if p["decisions"].exists():
        scores = np.lib.format.open_memmap(p["decisions"], mode="r+"); completed = int(json.loads(p["progress_scores"].read_text(encoding="utf-8"))["completed"])
    else:
        p["decisions"].parent.mkdir(parents=True, exist_ok=True); scores = np.lib.format.open_memmap(p["decisions"], mode="w+", dtype=np.float32, shape=shape); completed = 0
    for index in range(completed, len(rows)):
        row = rows[index]; target = sequence_mean_logprob(model, row["prompt_ids"], row["target_ids"]); foil = sequence_mean_logprob(model, row["prompt_ids"], row["foil_ids"])
        scores[index] = (target, foil, target - foil)
        if (index + 1) % 48 == 0 or index + 1 == len(rows): scores.flush(); save(p["progress_scores"], {"completed": index + 1, "shape": shape}); print(f"[phase2390 {key} scores] {index + 1}/{len(rows)}", flush=True)
    array = np.asarray(scores, dtype=np.float32); summary = {"rows": len(rows), "target_over_foil": float(np.mean(array[:, 2] > 0)),
        "mean_margin": float(array[:, 2].mean()), "by_partition": {part: float(np.mean(array[[i for i,r in enumerate(rows) if r['partition']==part], 2] > 0)) for part in sorted({r['partition'] for r in rows})}}
    scores.flush(); close(scores); return summary


def collect_reference(key: str, model, rows: list[dict]) -> dict:
    p = paths(key)
    if p["reference"].exists() and p["reference_mask"].exists():
        return {"shape": list(np.load(p["reference"], mmap_mode="r").shape), "rows": 16, "resumed": True}
    refs = [row for row in rows if row["partition"] == "fresh_unit_lockbox"][:16]
    sequences = [row["prompt_ids"] + row["target_ids"] for row in refs]; width = max(map(len, sequences)); qmods = modules(model); dim = int(model.get_input_embeddings().weight.shape[1])
    field = np.lib.format.open_memmap(p["reference"], mode="w+", dtype=np.float16, shape=(len(refs), len(qmods), width, dim))
    valid = np.lib.format.open_memmap(p["reference_mask"], mode="w+", dtype=np.uint8, shape=(len(refs), width)); captures = {}; handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, output, qpoint=qpoint): captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for index, sequence in enumerate(sequences):
                ids = torch.tensor([sequence], dtype=torch.long, device=device); mask = torch.ones_like(ids); captures.clear()
                model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                for qpoint in range(len(qmods)): field[index, qpoint, :len(sequence)] = captures[qpoint][0].float().cpu().numpy().astype(np.float16)
                valid[index, :len(sequence)] = 1; field.flush(); valid.flush(); print(f"[phase2390 {key} reference] {index + 1}/16", flush=True)
    finally:
        for handle in handles: handle.remove()
        field.flush(); valid.flush(); close(field); close(valid)
    return {"shape": [len(refs), len(qmods), width, dim], "rows": len(refs), "valid_tokens": int(sum(map(len, sequences)))}


def run_model(key: str, source_independent: list[dict], source_selection: list[dict]) -> dict:
    p = paths(key)
    if p["final"].exists(): return json.loads(p["final"].read_text(encoding="utf-8"))
    model, tokenizer, label = capability.load_model(key)
    try:
        independent = compile_independent(tokenizer, source_independent); selection = compile_selection(tokenizer, source_selection)
        write_rows(p["rows_independent"], independent); write_rows(p["rows_selection"], selection)
        batch = 12 if key == "qwen4b" else 3
        collection = {"independent": collect_independent(key, model, independent, batch), "boundary": collect_boundary(key, model, selection, batch)}
        behavior = collect_scores(key, model, selection); reference = collect_reference(key, model, selection)
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    result = {"model": key, "model_label": label, "collection": collection, "behavior": behavior, "reference": reference,
              "all_checks_passed": collection["independent"]["shape"][0] == 768 and collection["boundary"]["shape"][0] == 384 and behavior["rows"] == 384}
    save(p["final"], result); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen双规模语义—词汇全token全坐标场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Qwen4B BF16和Phase2389冻结的相对最佳Qwen14B NF4顺序驻留CUDA。各自tokenizer编码Phase2388全部768个独立句，采集embedding、每个block、final norm的句末与全token均值全部物理坐标；对384个同词反关系二选一prompt采集原生chat模板最后prompt token的全部checkpoint/坐标，并计算正确整句与反关系foil的平均逐token logprob。另保存16条fresh-unit参考的prompt+target全部token、全部checkpoint、全部坐标。

$$\Delta\bar\ell=|y|^{-1}\log p(y|x)-|\tilde y|^{-1}\log p(\tilde y|x),\qquad
C^{{mean}}_q(s)=|s|^{-1}\sum_p H_{{q,p}}(s).$$

**结果汇总。** 双模型 `{json.dumps(result['summary'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2390_c19441_c19760_qwen_semantic_lexical_fullfield.py`；全坐标场、逐行索引、行为分数和final位于 `tests/glm5/result/phase2390_c19441_c19760_qwen_semantic_lexical_fullfield`。

**理论进展、问题硬伤与结论。** 本Phase只建立不压缩的物理场与行为底座，不根据热力图目测命名语义坐标。canonical反方向共享主要词汇但token顺序不同；paraphrase仍保留实体词。NF4与BF16不可直接比较激活幅度。后续必须在unit锁箱上比较canonical→paraphrase方向泛化是否超过embedding，并报告所有失败族。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    independent = read_rows(P2388 / "material/independent_relation_sentences.jsonl"); selection = read_rows(P2388 / "material/semantic_selection_rows.jsonl")
    models = {key: run_model(key, independent, selection) for key in MODEL_ORDER}
    summary = {key: {"independent_shape": value["collection"]["independent"]["shape"], "boundary_shape": value["collection"]["boundary"]["shape"],
                     "target_over_foil": value["behavior"]["target_over_foil"], "lockbox_target_over_foil": value["behavior"]["by_partition"]["fresh_unit_lockbox"],
                     "mean_margin": value["behavior"]["mean_margin"], "reference_shape": value["reference"]["shape"]} for key, value in models.items()}
    checks = {"sequential_models": list(models) == list(MODEL_ORDER), "all_model_checks": all(value["all_checks_passed"] for value in models.values()),
              "finite_behavior": all(math.isfinite(value["behavior"]["mean_margin"]) for value in models.values())}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "summary": summary, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
