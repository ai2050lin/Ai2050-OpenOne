#!/usr/bin/env python3
"""Run Qwen3-4B behavior and full-coordinate baseline fields for Phase2315."""
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
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2315 = RESULT / "phase2315_c5041_c5100_active_response_contract"
OUT = RESULT / "phase2316_c5101_c5160_qwen4b_active_baseline"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = P2315 / "material/natural_active_response_bilingual.jsonl"
BOUNDARY = OUT / "raw/qwen4b_boundary_all_checkpoints.float16.npy"
BOUNDARY_PROGRESS = OUT / "raw/boundary_progress.json"
ALL_TOKEN = OUT / "raw/qwen4b_representative_all_token_all_checkpoints.float16.npy"
ALL_TOKEN_PROGRESS = OUT / "raw/all_token_progress.json"
sys.path.insert(0, str(TESTS))

import phase1332_bf16_utils as model_base  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as contract  # noqa: E402


PHASE = 2316
CAMPAIGN = "C5101-C5160"
EPS = 1e-12


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def pad_right(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(value) for value in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = 1
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def sequence_scores(model, device, rows: list[dict], path: Path, batch_size: int = 32) -> list[dict]:
    output = read_rows(path) if path.exists() else []
    completed = len(output)
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    with torch.inference_mode():
        for start in range(completed, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            variants = []
            for row in batch_rows:
                for label, key in (("target", "future_target_ids"), ("wrong", "future_wrong_ids")):
                    candidate = row[key]
                    variants.append((row, label, row["future_prompt_ids"] + candidate,
                                     len(row["future_prompt_ids"]), candidate))
            ids, mask, positions = pad_right([value[2] for value in variants], device, pad)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           use_cache=False, return_dict=True).logits
            gathered: dict[str, dict] = defaultdict(dict)
            for local, (row, label, _sequence, prompt_length, candidate) in enumerate(variants):
                selected = logits[local, prompt_length - 1:prompt_length - 1 + len(candidate)].float()
                token_ids = torch.tensor(candidate, dtype=torch.long, device=selected.device)
                token_logps = (selected.gather(1, token_ids[:, None])[:, 0]
                                - torch.logsumexp(selected, dim=-1)).cpu().tolist()
                gathered[row["case_id"]][label] = {
                    "token_logprobs": [float(value) for value in token_logps],
                    "sum_logprob": float(sum(token_logps)),
                    "mean_logprob": float(np.mean(token_logps)), "token_count": len(token_logps),
                    "first_logprob": float(token_logps[0]),
                }
            for row in batch_rows:
                target = gathered[row["case_id"]]["target"]
                wrong = gathered[row["case_id"]]["wrong"]
                output.append({
                    "case_id": row["case_id"], "family": row["family"],
                    "language": row["language"], "surface": row["surface"],
                    "partition": row["partition"], "unit": int(row["unit"]),
                    "state": int(row["state"]), "cue_id": row["cue_id"],
                    "target_mention_order": row["target_mention_order"],
                    "target": target, "wrong": wrong,
                    "correct_by_sum": target["sum_logprob"] > wrong["sum_logprob"],
                    "correct_by_mean": target["mean_logprob"] > wrong["mean_logprob"],
                    "correct_first": target["first_logprob"] > wrong["first_logprob"],
                    "sum_margin": target["sum_logprob"] - wrong["sum_logprob"],
                    "mean_margin": target["mean_logprob"] - wrong["mean_logprob"],
                    "first_margin": target["first_logprob"] - wrong["first_logprob"],
                })
            write_rows(path, output)
            del logits
            print(f"[phase2316 sequence] {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)
    return output


def normalized(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def free_generation(model, tokenizer, device, rows: list[dict], path: Path,
                    batch_size: int = 32) -> list[dict]:
    output = read_rows(path) if path.exists() else []
    completed = len(output)
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    max_new = 18
    for start in range(completed, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(row["future_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for local, row in enumerate(batch):
            sequence = row["future_prompt_ids"]
            ids[local, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long, device=device)
            mask[local, width - len(sequence):] = 1
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new,
                use_cache=True, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
            )
        for row, token_ids in zip(batch, generated[:, width:].detach().cpu().tolist()):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            clean = normalized(text)
            target = normalized(row["identity_target"])
            wrong = normalized(row["identity_wrong"])
            target_pos, wrong_pos = clean.find(target), clean.find(wrong)
            expected = normalized(row["future_target_text"])
            output.append({
                "case_id": row["case_id"], "family": row["family"],
                "language": row["language"], "surface": row["surface"],
                "partition": row["partition"], "unit": int(row["unit"]),
                "state": int(row["state"]), "cue_id": row["cue_id"],
                "generated_ids": [int(value) for value in token_ids], "generated_text": text,
                "first_identity_correct": target_pos >= 0 and (wrong_pos < 0 or target_pos < wrong_pos),
                "target_found": target_pos >= 0, "wrong_found": wrong_pos >= 0,
                "future_prefix_exact": clean.startswith(expected),
            })
        write_rows(path, output)
        print(f"[phase2316 free] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def summarize_scores(values: list[dict]) -> dict:
    return {
        "rows": len(values),
        "sum_accuracy": float(np.mean([row["correct_by_sum"] for row in values])),
        "mean_accuracy": float(np.mean([row["correct_by_mean"] for row in values])),
        "first_accuracy": float(np.mean([row["correct_first"] for row in values])),
        "sum_margin": float(np.mean([row["sum_margin"] for row in values])),
        "mean_margin": float(np.mean([row["mean_margin"] for row in values])),
        "first_margin": float(np.mean([row["first_margin"] for row in values])),
    }


def sequence_ledger(scores: list[dict]) -> dict:
    families, qualified = {}, []
    for family in contract.FAMILIES:
        family_rows = [row for row in scores if row["family"] == family]
        slices = {"overall:all": summarize_scores(family_rows)}
        for key, choices in (("language", contract.LANGUAGES), ("surface", contract.SURFACES),
                             ("partition", contract.PARTITIONS),
                             ("target_mention_order", ("first", "last")),
                             ("cue_id", tuple(f"cue_{index}" for index in range(4)))):
            for choice in choices:
                slices[f"{key}:{choice}"] = summarize_scores(
                    [row for row in family_rows if row[key] == choice]
                )
        passed = all(min(value["sum_accuracy"], value["mean_accuracy"])
                     >= contract.SEQUENCE_GATE for value in slices.values())
        families[family] = {"qualified": passed, "slices": slices}
        if passed:
            qualified.append(family)
    return {"gate": contract.SEQUENCE_GATE, "families": families,
            "qualified_families": qualified, "overall": summarize_scores(scores)}


def summarize_free(values: list[dict]) -> dict:
    return {
        "rows": len(values),
        "first_identity_accuracy": float(np.mean([row["first_identity_correct"] for row in values])),
        "target_found": float(np.mean([row["target_found"] for row in values])),
        "wrong_found": float(np.mean([row["wrong_found"] for row in values])),
        "future_prefix_exact": float(np.mean([row["future_prefix_exact"] for row in values])),
    }


def free_ledger(rows: list[dict]) -> dict:
    families, qualified = {}, []
    for family in contract.FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        slices = {"overall:all": summarize_free(family_rows)}
        for key, choices in (("language", contract.LANGUAGES), ("surface", contract.SURFACES),
                             ("partition", contract.PARTITIONS)):
            for choice in choices:
                slices[f"{key}:{choice}"] = summarize_free(
                    [row for row in family_rows if row[key] == choice]
                )
        passed = all(value["first_identity_accuracy"] >= contract.FREE_IDENTITY_GATE
                     for value in slices.values())
        families[family] = {"qualified": passed, "slices": slices}
        if passed:
            qualified.append(family)
    return {"gate": contract.FREE_IDENTITY_GATE, "families": families,
            "qualified_families": qualified, "overall": summarize_free(rows)}


def capture_boundary(model, device, rows: list[dict]) -> dict:
    module_list = modules(model)
    dimension = int(model.config.hidden_size)
    shape = (len(rows), len(module_list), dimension)
    BOUNDARY.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if BOUNDARY.exists() and BOUNDARY_PROGRESS.exists():
        progress = json.loads(BOUNDARY_PROGRESS.read_text(encoding="utf-8"))
        if progress["shape"] != list(shape):
            raise RuntimeError(("boundary_resume_shape", progress["shape"], shape))
        completed = int(progress["completed"])
        field = np.lib.format.open_memmap(BOUNDARY, mode="r+")
    else:
        field = np.lib.format.open_memmap(BOUNDARY, mode="w+", dtype=np.float16, shape=shape)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(module_list):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), 16):
                batch = rows[start:start + 16]
                ids, mask, positions = pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear()
                model.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                if len(captures) != len(module_list):
                    raise RuntimeError(("boundary_checkpoint_count", len(captures), len(module_list)))
                for q in range(len(module_list)):
                    selected = torch.stack([captures[q][local, ends[local]] for local in range(len(batch))])
                    field[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                field.flush()
                save(BOUNDARY_PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                print(f"[phase2316 boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_memmap(field)
    index = [{
        "hidden_index": index, "case_id": row["case_id"], "family": row["family"],
        "language": row["language"], "surface": row["surface"],
        "partition": row["partition"], "unit": int(row["unit"]), "state": int(row["state"]),
    } for index, row in enumerate(rows)]
    write_rows(OUT / "index/boundary_rows.jsonl", index)
    return {"path": str(BOUNDARY.relative_to(ROOT)), "shape": list(shape),
            "index": str((OUT / "index/boundary_rows.jsonl").relative_to(ROOT))}


def representative_rows(rows: list[dict]) -> list[dict]:
    selected = []
    first_unit = {partition: min(unit for unit, value in contract.PARTITION_BY_UNIT.items()
                                 if value == partition) for partition in contract.PARTITIONS}
    for family in contract.FAMILIES:
        for language in contract.LANGUAGES:
            for surface in contract.SURFACES:
                for partition in contract.PARTITIONS:
                    for state in (0, 1):
                        selected.append(next(row for row in rows
                                             if row["family"] == family
                                             and row["language"] == language
                                             and row["surface"] == surface
                                             and row["partition"] == partition
                                             and int(row["unit"]) == first_unit[partition]
                                             and int(row["state"]) == state))
    return selected


def capture_all_token(model, device, rows: list[dict]) -> dict:
    selected = representative_rows(rows)
    module_list = modules(model)
    dimension = int(model.config.hidden_size)
    max_tokens = max(len(row["future_prompt_ids"]) for row in selected)
    shape = (len(selected), len(module_list), max_tokens, dimension)
    completed = 0
    if ALL_TOKEN.exists() and ALL_TOKEN_PROGRESS.exists():
        progress = json.loads(ALL_TOKEN_PROGRESS.read_text(encoding="utf-8"))
        if progress["shape"] != list(shape):
            raise RuntimeError(("all_token_resume_shape", progress["shape"], shape))
        completed = int(progress["completed"])
        field = np.lib.format.open_memmap(ALL_TOKEN, mode="r+")
    else:
        field = np.lib.format.open_memmap(ALL_TOKEN, mode="w+", dtype=np.float16, shape=shape)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(module_list):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(selected), 4):
                batch = selected[start:start + 4]
                ids, mask, positions = pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear()
                model.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False, return_dict=True)
                if len(captures) != len(module_list):
                    raise RuntimeError(("all_token_checkpoint_count", len(captures), len(module_list)))
                for local, row in enumerate(batch):
                    length = len(row["future_prompt_ids"])
                    for q in range(len(module_list)):
                        field[start + local, q, :length] = (
                            captures[q][local, :length].float().cpu().numpy().astype(np.float16)
                        )
                field.flush()
                save(ALL_TOKEN_PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                print(f"[phase2316 all-token] {start + len(batch)}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_memmap(field)
    index = [{
        "hidden_index": index, "case_id": row["case_id"], "family": row["family"],
        "language": row["language"], "surface": row["surface"], "partition": row["partition"],
        "unit": int(row["unit"]), "state": int(row["state"]),
        "token_count": len(row["future_prompt_ids"]), "token_ids": row["future_prompt_ids"],
        "role_positions": row["role_positions"], "boundary_position": row["boundary_position"],
    } for index, row in enumerate(selected)]
    write_rows(OUT / "index/all_token_rows.jsonl", index)
    return {"path": str(ALL_TOKEN.relative_to(ROOT)), "shape": list(shape),
            "index": str((OUT / "index/all_token_rows.jsonl").relative_to(ROOT)),
            "rows": len(selected), "max_tokens": max_tokens}


def state_response_summary(rows: list[dict], field_path: Path) -> dict:
    field = np.load(field_path, mmap_mode="r")
    grouped: dict[tuple, dict[int, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        grouped[(row["family"], row["language"], row["surface"], int(row["unit"]))][int(row["state"])] = index
    family_sums = np.zeros((len(contract.FAMILIES), field.shape[1], field.shape[2]), dtype=np.float64)
    family_counts = np.zeros(len(contract.FAMILIES), dtype=np.int64)
    metrics = {family: [] for family in contract.FAMILIES}
    for key, pair in grouped.items():
        if set(pair) != {0, 1}:
            raise RuntimeError(("state_pair_missing", key, pair))
        family = key[0]
        fi = contract.FAMILIES.index(family)
        response = field[pair[1]].astype(np.float32) - field[pair[0]].astype(np.float32)
        family_sums[fi] += response
        family_counts[fi] += 1
        rms = np.sqrt(np.mean(np.square(response.astype(np.float64)), axis=1))
        metrics[family].append(rms)
    means = family_sums / family_counts[:, None, None]
    mean_path = OUT / "atlas/state_response_family_mean.float32.npy"
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mean_path, means.astype(np.float32), allow_pickle=False)
    summary = {}
    for family in contract.FAMILIES:
        values = np.stack(metrics[family])
        summary[family] = {
            "pairs": int(values.shape[0]),
            "checkpoint_rms_mean": np.mean(values, axis=0).tolist(),
            "checkpoint_rms_median": np.median(values, axis=0).tolist(),
            "peak_checkpoint": int(np.argmax(np.mean(values, axis=0))),
        }
    close_memmap(field)
    save(OUT / "analysis/state_response_summary.json", summary)
    return {"families": summary, "family_mean_path": str(mean_path.relative_to(ROOT)),
            "family_mean_shape": list(means.shape)}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-4B 八构式自然续写与全 token 全坐标基线场（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 在 Phase2315 全部预模型审计通过后，使用本地 Qwen3-4B、BF16、非量化 CUDA 顺序运行。每行同时比较完整正确未来与完整错误未来的 token 总 log 概率和长度均值，并独立进行贪心自由续写；两者不能互相替代。全场观察保存 2048 行全部边界检查点，以及覆盖 `family x language x surface x partition x state` 的 256 行全 token 代表场，均保留 2560 个原始坐标和原始 token 顺序，没有 Top-K、PCA、余弦筛选或坐标重排。

$$
S(y_{{1:K}}\mid x)=\sum_{{k=1}}^K\log p(y_k\mid x,y_{{<k}}),\qquad
R_{{f,q,j}}=\mathbb E[H_{{q,j}}^{{(1)}}-H_{{q,j}}^{{(0)}}\mid f].
$$

第二式是成对状态响应的逐坐标均值，只是描述图层；原始逐样本场仍单独保存。

**结果与门槛。** 完整未来门 `{result['sequence']['gate']}` 的合格族为 `{result['sequence']['qualified_families']}`，自由身份门 `{result['free']['gate']}` 的合格族为 `{result['free']['qualified_families']}`，两门交集为 `{result['qualified_families']}`。完整未来整体账 `{json.dumps(result['sequence']['overall'], ensure_ascii=False)}`；自由续写整体账 `{json.dumps(result['free']['overall'], ensure_ascii=False)}`。全场 `{json.dumps(result['field'], ensure_ascii=False)}`；状态响应峰值 `{json.dumps({k: v['peak_checkpoint'] for k, v in result['state_response']['families'].items()}, ensure_ascii=False)}`。

**相关文件与审计。** 模型审计 `{json.dumps(result['model_audit'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2316_c5101_c5160_qwen4b_active_baseline.py`；结果 `tests/glm5/result/phase2316_c5101_c5160_qwen4b_active_baseline`。

**理论进展、问题硬伤与结论。** 本 Phase 建立的是新材料上的行为资格和逐坐标观测基线，不是齿轮发现。完整候选未来依赖 teacher forcing，自由续写只观察前 18 个 token；模板由研究者编写、没有独立人类盲评；全 token 场虽覆盖全部坐标，但只抽样 256 行。行为失败的族仍保留观察图，但不得进入语义充分性结论。下一步只用模型加载前冻结的八个 Rademacher 方向和四个成对方向，主动测量非对角方向响应；任意完整 Jacobian、共享语义方向或新数学结构仍无授权。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((P2315 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2315 is not authorized")
    rows = read_rows(ROWS_PATH)
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = model_base.load_bf16("qwen3")
        model_audit = {
            "model": "Qwen3-4B", "device": str(device), "placement": placement,
            "hidden_size": int(model.config.hidden_size), "layers": int(model.config.num_hidden_layers),
            "checkpoints": len(modules(model)), "quantization": model_base.quantization_audit(model),
        }
        score_path = OUT / "behavior/sequence_scores.jsonl"
        scores = sequence_scores(model, device, rows, score_path)
        sequence = sequence_ledger(scores)
        save(OUT / "behavior/sequence_ledger.json", sequence)
        free_path = OUT / "behavior/free_generation.jsonl"
        free_rows = free_generation(model, tokenizer, device, rows, free_path)
        free = free_ledger(free_rows)
        save(OUT / "behavior/free_ledger.json", free)
        qualified = sorted(set(sequence["qualified_families"]) & set(free["qualified_families"]))
        boundary = capture_boundary(model, device, rows)
        all_token = capture_all_token(model, device, rows)
        response = state_response_summary(rows, BOUNDARY)
        field = {"boundary": boundary, "representative_all_token": all_token}
        hashes = {
            "sequence_scores": file_hash(score_path), "free_generation": file_hash(free_path),
            "boundary": file_hash(BOUNDARY), "all_token": file_hash(ALL_TOKEN),
            "state_family_mean": file_hash(ROOT / response["family_mean_path"]),
        }
        checks = {
            "parent_authorized": True, "all_rows_scored": len(scores) == len(rows),
            "all_rows_freely_generated": len(free_rows) == len(rows),
            "boundary_shape": boundary["shape"] == [len(rows), 38, 2560],
            "all_token_representative_rows": all_token["rows"] == 256,
            "all_token_all_checkpoints_coordinates": all_token["shape"][1:] == [38, all_token["max_tokens"], 2560],
            "all_state_pairs": all(value["pairs"] == 128 for value in response["families"].values()),
            "bf16_nonquantized": (model_audit["quantization"]["has_bf16_parameters"]
                                   and not model_audit["quantization"]["has_quantized_modules"]),
            "no_topk_pca_cosine": True,
        }
        result = {
            "phase": PHASE, "campaign": CAMPAIGN,
            "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
            "model_audit": model_audit, "sequence": sequence, "free": free,
            "qualified_families": qualified, "field": field, "state_response": response,
            "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
            "strict_conclusion": (
                "The new natural-sentence interface now has separate complete-candidate and free-generation ledgers, "
                "plus complete boundary and representative all-token coordinate fields. These are behavior and "
                "observation assets, not evidence for a shared gear."
            ),
            "next_authorization": "Run frozen signed structured perturbations on behavior-qualified family routes.",
        }
        save(final_path, result)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
