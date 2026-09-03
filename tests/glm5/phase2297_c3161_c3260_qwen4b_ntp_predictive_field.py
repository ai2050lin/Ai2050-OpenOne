#!/usr/bin/env python3
"""Collect Qwen3-4B full-vocabulary NTP outputs and exact-coordinate fields."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase2296_c3101_c3160_ntp_predictive_contract"
OUT = RESULT / "phase2297_c3161_c3260_qwen4b_ntp_predictive_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase1332_bf16_utils as model_base  # noqa: E402
import phase2296_c3101_c3160_ntp_predictive_contract as contract  # noqa: E402


PHASE = 2297
CAMPAIGN = "C3161-C3260"
RAW = OUT / "raw"
BOUNDARY_FIELD = RAW / "qwen4b_ntp_boundary_all_checkpoints.float16.npy"
FULL_LOGITS = RAW / "qwen4b_ntp_full_vocabulary_logits.float16.npy"
TOKEN_FIELD = RAW / "qwen4b_ntp_representative_all_token.float16.npy"
CONTRIBUTIONS = OUT / "atlas/qwen4b_target_wrong_coordinate_contributions.float16.npy"
FISHER_DIAG = OUT / "atlas/qwen4b_output_fisher_diagonal.float32.npy"
EPS = 1e-12


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    contract.write_rows(path, rows)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def file_hash(path: Path) -> str:
    return contract.file_hash(path)


def checkpoint_modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def pad_batch(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(sequences):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, :len(row)] = 1
    position_ids = mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return ids, mask, position_ids


def sequence_scores(model, device, rows: list[dict], batch_size: int = 12) -> list[dict]:
    output = []
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    items = []
    for row in rows:
        for kind, key in (("target", "ntp_target_ids"), ("wrong", "ntp_wrong_ids")):
            candidate = row[key]
            items.append((row, kind, row["ntp_prompt_ids"] + candidate, len(row["ntp_prompt_ids"]), candidate))
    scores = defaultdict(dict)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            ids, mask, position_ids = pad_batch([item[2] for item in batch], device, pad)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                           use_cache=False, return_dict=True).logits.float()
            for i, (row, kind, _sequence, prompt_len, candidate) in enumerate(batch):
                token_logps = []
                for offset, token_id in enumerate(candidate):
                    pos = prompt_len - 1 + offset
                    value = logits[i, pos, int(token_id)] - torch.logsumexp(logits[i, pos], dim=-1)
                    token_logps.append(float(value.item()))
                scores[row["case_id"]][kind] = {
                    "token_logprobs": token_logps,
                    "sum_logprob": float(sum(token_logps)),
                    "mean_logprob": float(sum(token_logps) / max(len(token_logps), 1)),
                    "token_count": len(token_logps),
                }
            print(f"[phase2297 sequence] {min(start + len(batch), len(items))}/{len(items)}", flush=True)
    for row in rows:
        target = scores[row["case_id"]]["target"]
        wrong = scores[row["case_id"]]["wrong"]
        output.append({
            "case_id": row["case_id"], "family": row["family"], "language": row["language"],
            "surface": row["surface"], "partition": row["partition"], "state": row["state"],
            "target": target, "wrong": wrong,
            "correct_by_mean": target["mean_logprob"] > wrong["mean_logprob"],
            "correct_by_sum": target["sum_logprob"] > wrong["sum_logprob"],
            "mean_margin": target["mean_logprob"] - wrong["mean_logprob"],
            "sum_margin": target["sum_logprob"] - wrong["sum_logprob"],
        })
    return output


def sequence_ledger(rows: list[dict], scores: list[dict]) -> dict:
    by_id = {row["case_id"]: row for row in scores}
    cells = {}
    qualified = []
    for family in contract.FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        slices = {}
        keys = [("overall", "all")]
        keys += [("language", value) for value in ("en", "zh")]
        keys += [("surface", value) for value in ("narrative", "dialogue")]
        keys += [("partition", value) for value in contract.PARTITIONS]
        for kind, value in keys:
            subset = family_rows if kind == "overall" else [row for row in family_rows if row[kind] == value]
            values = [by_id[row["case_id"]] for row in subset]
            slices[f"{kind}:{value}"] = {
                "rows": len(values),
                "mean_accuracy": float(np.mean([row["correct_by_mean"] for row in values])),
                "sum_accuracy": float(np.mean([row["correct_by_sum"] for row in values])),
                "mean_margin": float(np.mean([row["mean_margin"] for row in values])),
            }
        passed = all(min(value["mean_accuracy"], value["sum_accuracy"]) >= contract.BEHAVIOR_GATE
                     for value in slices.values())
        cells[family] = {"qualified": passed, "slices": slices}
        if passed:
            qualified.append(family)
    return {"families": cells, "qualified_families": qualified, "gate": contract.BEHAVIOR_GATE}


def capture_boundary_and_logits(model, device, rows: list[dict], batch_size: int = 10) -> dict:
    modules = checkpoint_modules(model)
    dimension = int(model.config.hidden_size)
    vocab = int(model.config.vocab_size)
    RAW.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(BOUNDARY_FIELD, mode="w+", dtype=np.float16,
                                      shape=(len(rows), len(modules), dimension))
    logits_file = np.lib.format.open_memmap(FULL_LOGITS, mode="w+", dtype=np.float16,
                                            shape=(len(rows), vocab))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(modules):
        def hook(_module, _inputs, output, q=q):
            captures[q] = output[0] if isinstance(output, tuple) else output
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, position_ids = pad_batch([row["ntp_prompt_ids"] for row in batch], device, pad)
                output = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                               use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(modules)):
                    tensor = captures[q]
                    selected = torch.stack([tensor[i, ends[i]] for i in range(len(batch))])
                    field[start:start + len(batch), q] = selected.detach().float().cpu().numpy().astype(np.float16)
                selected_logits = torch.stack([output.logits[i, ends[i]] for i in range(len(batch))])
                logits_file[start:start + len(batch)] = selected_logits.detach().float().cpu().numpy().astype(np.float16)
                print(f"[phase2297 boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush()
    logits_file.flush()
    return {
        "field_path": str(BOUNDARY_FIELD.relative_to(ROOT)),
        "field_shape": list(field.shape),
        "logits_path": str(FULL_LOGITS.relative_to(ROOT)),
        "logits_shape": list(logits_file.shape),
        "checkpoints": len(modules), "dimension": dimension, "vocabulary": vocab,
    }


def representative_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["partition"] == "fresh_lockbox" and int(row["unit"]) == 26]


def capture_representative_tokens(model, device, rows: list[dict]) -> dict:
    reps = representative_rows(rows)
    modules = checkpoint_modules(model)
    dimension = int(model.config.hidden_size)
    total = sum(len(row["ntp_prompt_ids"]) * len(modules) for row in reps)
    TOKEN_FIELD.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(TOKEN_FIELD, mode="w+", dtype=np.float16, shape=(total, dimension))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(modules):
        def hook(_module, _inputs, output, q=q):
            captures[q] = output[0] if isinstance(output, tuple) else output
        handles.append(module.register_forward_hook(hook))
    meta = []
    cursor = 0
    try:
        model.eval()
        with torch.inference_mode():
            for n, row in enumerate(reps):
                ids = torch.tensor([row["ntp_prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                position_ids = torch.arange(ids.shape[1], device=device).unsqueeze(0)
                model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                      use_cache=False, return_dict=True)
                for q in range(len(modules)):
                    values = captures[q][0, :ids.shape[1]].detach().float().cpu().numpy().astype(np.float16)
                    field[cursor:cursor + len(values)] = values
                    for token in range(len(values)):
                        meta.append({
                            "row": cursor + token, "case_id": row["case_id"], "family": row["family"],
                            "language": row["language"], "surface": row["surface"], "state": row["state"],
                            "checkpoint": q, "token": token, "token_id": row["ntp_prompt_ids"][token],
                        })
                    cursor += len(values)
                print(f"[phase2297 all-token] {n + 1}/{len(reps)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush()
    write_rows(OUT / "index/representative_all_token_rows.jsonl", meta)
    return {"path": str(TOKEN_FIELD.relative_to(ROOT)), "shape": list(field.shape),
            "cases": len(reps), "row_index": str((OUT / "index/representative_all_token_rows.jsonl").relative_to(ROOT))}


def exact_output_contributions(model, rows: list[dict]) -> dict:
    field = np.load(BOUNDARY_FIELD, mmap_mode="r")
    logits = np.load(FULL_LOGITS, mmap_mode="r")
    dimension = int(model.config.hidden_size)
    CONTRIBUTIONS.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(CONTRIBUTIONS, mode="w+", dtype=np.float16,
                                       shape=(len(rows), dimension))
    weight = model.lm_head.weight
    max_abs_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            h = torch.tensor(np.asarray(field[start:start + len(batch), -1], dtype=np.float32),
                             device=weight.device, dtype=weight.dtype)
            target = torch.tensor([row["ntp_target_ids"][0] for row in batch], device=weight.device)
            wrong = torch.tensor([row["ntp_wrong_ids"][0] for row in batch], device=weight.device)
            delta_w = weight.index_select(0, target) - weight.index_select(0, wrong)
            contributions = (h * delta_w).float()
            output[start:start + len(batch)] = contributions.cpu().numpy().astype(np.float16)
            direct = np.asarray(logits[start:start + len(batch)], dtype=np.float32)
            expected = direct[np.arange(len(batch)), target.cpu().numpy()] - direct[np.arange(len(batch)), wrong.cpu().numpy()]
            max_abs_error = max(max_abs_error, float(np.max(np.abs(contributions.sum(-1).cpu().numpy() - expected))))
    output.flush()
    return {"path": str(CONTRIBUTIONS.relative_to(ROOT)), "shape": list(output.shape),
            "decomposition_max_abs_error_float16": max_abs_error}


def output_fisher_diagonal(model, rows: list[dict]) -> dict:
    reps = representative_rows(rows)
    all_rows = {row["case_id"]: i for i, row in enumerate(rows)}
    indices = [all_rows[row["case_id"]] for row in reps]
    logits_file = np.load(FULL_LOGITS, mmap_mode="r")
    weight = model.lm_head.weight
    dimension = int(model.config.hidden_size)
    FISHER_DIAG.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(FISHER_DIAG, mode="w+", dtype=np.float32,
                                       shape=(len(reps), dimension))
    with torch.inference_mode():
        for n, index in enumerate(indices):
            logits = torch.tensor(np.asarray(logits_file[index], dtype=np.float32), device=weight.device)
            probabilities = torch.softmax(logits, dim=-1)
            mean = torch.zeros(dimension, dtype=torch.float32, device=weight.device)
            second = torch.zeros_like(mean)
            for start in range(0, weight.shape[0], 4096):
                stop = min(start + 4096, weight.shape[0])
                w = weight[start:stop].float()
                p = probabilities[start:stop]
                mean += p @ w
                second += p @ (w * w)
            output[n] = torch.clamp(second - mean * mean, min=0).cpu().numpy()
            print(f"[phase2297 fisher] {n + 1}/{len(reps)}", flush=True)
    output.flush()
    write_rows(OUT / "index/fisher_rows.jsonl", [
        {"row": i, "case_id": row["case_id"], "family": row["family"], "language": row["language"],
         "surface": row["surface"], "state": row["state"], "checkpoint": 37}
        for i, row in enumerate(reps)
    ])
    return {"path": str(FISHER_DIAG.relative_to(ROOT)), "shape": list(output.shape),
            "definition": "exact diagonal of the categorical Fisher pullback at final normalized state"}


def lens_metrics(model, rows: list[dict], qpoints: tuple[int, ...], batch_size: int = 8) -> dict:
    field = np.load(BOUNDARY_FIELD, mmap_mode="r")
    actual_file = np.load(FULL_LOGITS, mmap_mode="r")
    output = []
    device = model.lm_head.weight.device
    with torch.inference_mode():
        for q in qpoints:
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                h = torch.tensor(np.asarray(field[start:start + len(batch), q], dtype=np.float32),
                                 device=device, dtype=model.lm_head.weight.dtype)
                normalized = h if q == field.shape[1] - 1 else model.model.norm(h)
                lens = model.lm_head(normalized).float()
                actual = torch.tensor(np.asarray(actual_file[start:start + len(batch)], dtype=np.float32), device=device)
                p = torch.softmax(lens, dim=-1)
                final_p = torch.softmax(actual, dim=-1)
                midpoint = 0.5 * (p + final_p)
                js = 0.5 * torch.sum(p * (torch.log(p + EPS) - torch.log(midpoint + EPS)), dim=-1)
                js += 0.5 * torch.sum(final_p * (torch.log(final_p + EPS) - torch.log(midpoint + EPS)), dim=-1)
                entropy = torch.logsumexp(lens, dim=-1) - torch.sum(p * lens, dim=-1)
                top_prob, top_id = torch.max(p, dim=-1)
                for i, row in enumerate(batch):
                    target = int(row["ntp_target_ids"][0])
                    wrong = int(row["ntp_wrong_ids"][0])
                    output.append({
                        "case_id": row["case_id"], "family": row["family"], "language": row["language"],
                        "surface": row["surface"], "partition": row["partition"], "state": row["state"],
                        "checkpoint": q, "entropy": float(entropy[i].item()),
                        "js_to_actual_final": float(js[i].item()),
                        "target_wrong_margin": float((lens[i, target] - lens[i, wrong]).item()),
                        "target_probability": float(p[i, target].item()),
                        "wrong_probability": float(p[i, wrong].item()),
                        "top_token_id": int(top_id[i].item()), "top_probability": float(top_prob[i].item()),
                    })
            print(f"[phase2297 lens] q={q}", flush=True)
    path = OUT / "prediction/logit_lens_metrics.jsonl"
    write_rows(path, output)
    return {"path": str(path.relative_to(ROOT)), "rows": len(output), "qpoints": list(qpoints)}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-4B完整下一词分布与全坐标预测场（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 对 Phase2296 冻结的六构式 `{result['material_rows']}` 个自然词汇答案边界，先分别计算正确与错误完整答案的逐token teacher-forced对数概率，再一次前向保存实际完整下一token logits、embedding、36个block后状态、final norm边界状态的全部2560物理坐标。另保存 fresh-lockbox unit26 的全部真实token、全部检查点、全部坐标。错误样本不删除，只在账本中分层。模型为本地 Qwen3-4B、BF16、非量化CUDA；未读取Attention或MLP内部量。

**公式。** 自然词汇行为使用长度均值和总序列分数双账：

$$
\bar s(y\mid x)=\frac1{{|y|}}\sum_r\log p(y_r\mid x,y_{{<r}}),
\qquad
\widehat y=\arg\max_{{y\in\{{y^+,y^-\}}}}\bar s(y\mid x).
$$

final norm坐标到第一token目标竞争margin的精确线性分账为：

$$
z_{{y^+}}-z_{{y^-}}=\sum_{{j=1}}^d H_{{L,j}}\left(W_{{y^+,j}}-W_{{y^-,j}}\right).
$$

输出 Fisher 对角仅在代表样本的 final norm 处精确计算，定义为：

$$
G_{{jj}}=\sum_v p_vW_{{v,j}}^2-\left(\sum_vp_vW_{{v,j}}\right)^2.
$$

**结果汇总。** 序列行为 `{json.dumps(result['sequence_ledger'], ensure_ascii=False)}`；场 `{json.dumps(result['field'], ensure_ascii=False)}`；全token场 `{json.dumps(result['representative_token_field'], ensure_ascii=False)}`；坐标输出分账 `{json.dumps(result['contributions'], ensure_ascii=False)}`；Fisher对角 `{json.dumps(result['fisher_diagonal'], ensure_ascii=False)}`；logit-lens `{json.dumps(result['lens'], ensure_ascii=False)}`；模型 `{json.dumps(result['model'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。这里的完整词表分布仍位于“回答问题”的接口，不等于无任务条件下的全部语义；teacher forcing只比较冻结的正确/错误续写，不枚举无限未来。中层 logit lens 是统一只读辅助尺，不是模型实际在该层直接输出的概率。Fisher对角只描述 final norm 附近的局部输出敏感度，不能命名为流形曲率、因果协同或语义坐标。相关脚本 `tests/glm5/phase2297_c3161_c3260_qwen4b_ntp_predictive_field.py`；结果 `tests/glm5/result/phase2297_c3161_c3260_qwen4b_ntp_predictive_field`。
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
    parent = json.loads((PARENT / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2296 contract did not pass")
    rows = read_rows(PARENT / "material/ntp_natural_bilingual.jsonl")
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = model_base.load_bf16("qwen3")
        quantization = model_base.quantization_audit(model)
        scores = sequence_scores(model, device, rows)
        write_rows(OUT / "behavior/lexical_sequence_scores.jsonl", scores)
        ledger = sequence_ledger(rows, scores)
        save(OUT / "behavior/lexical_sequence_ledger.json", ledger)
        field = capture_boundary_and_logits(model, device, rows)
        token_field = capture_representative_tokens(model, device, rows)
        contributions = exact_output_contributions(model, rows)
        fisher = output_fisher_diagonal(model, rows)
        lens = lens_metrics(model, rows, contract.QPOINTS_4B)
        model_info = {
            "name": "Qwen3-4B", "precision": "bfloat16", "device": str(device),
            "placement": placement, "quantization": quantization,
            "hidden_size": int(model.config.hidden_size), "layers": len(model.model.layers),
            "vocabulary": int(model.config.vocab_size),
        }
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {
        "all_rows_scored": len(scores) == len(rows),
        "full_vocabulary_saved": field["logits_shape"] == [len(rows), model_info["vocabulary"]],
        "all_boundary_checkpoints_saved": field["field_shape"] == [len(rows), model_info["layers"] + 2, model_info["hidden_size"]],
        "all_coordinates_saved": field["dimension"] == 2560,
        "representative_all_token_saved": token_field["cases"] == len(contract.FAMILIES) * 2 * 2 * 2,
        "contribution_decomposition_verified": contributions["decomposition_max_abs_error_float16"] <= 0.5,
        "fisher_all_coordinates": fisher["shape"] == [token_field["cases"], 2560],
        "lens_rows_complete": lens["rows"] == len(rows) * len(contract.QPOINTS_4B),
        "bf16_nonquantized": quantization["has_bf16_parameters"] and not quantization["has_quantized_modules"],
    }
    hashes = {
        "sequence_scores": file_hash(OUT / "behavior/lexical_sequence_scores.jsonl"),
        "boundary_field": file_hash(BOUNDARY_FIELD), "full_logits": file_hash(FULL_LOGITS),
        "token_field": file_hash(TOKEN_FIELD), "contributions": file_hash(CONTRIBUTIONS),
        "fisher": file_hash(FISHER_DIAG), "lens": file_hash(OUT / "prediction/logit_lens_metrics.jsonl"),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "material_rows": len(rows), "sequence_ledger": ledger,
        "field": field, "representative_token_field": token_field,
        "contributions": contributions, "fisher_diagonal": fisher, "lens": lens,
        "model": model_info, "checks": checks, "hashes": hashes,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            f"Qwen3-4B lexical sequence behavior dual-qualified {len(ledger['qualified_families'])}/6 families; "
            "the campaign now links exact boundary HiddenStates to the complete next-token vocabulary distribution "
            "and exact final-coordinate target competition, without asserting a sufficient statistic or causal gear."
        ),
        "next_authorization": "Run basic full-vocabulary probability accounting, state/output timing, and exact-coordinate stability before any advanced dynamical interpretation.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
