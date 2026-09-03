#!/usr/bin/env python3
"""Collect Qwen3-4B raw declarative-continuation behavior and full fields."""
from __future__ import annotations

import gc
import json
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
PARENT = RESULT / "phase2303_c3701_c3780_declarative_continuation_contract"
OUT = RESULT / "phase2304_c3781_c3900_qwen4b_declarative_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = PARENT / "material/declarative_continuation_bilingual.jsonl"
RAW = OUT / "raw"
BOUNDARY = RAW / "qwen4b_declarative_boundary_all_checkpoints.float16.npy"
LOGITS = RAW / "qwen4b_declarative_full_vocabulary_logits.float16.npy"
PROGRESS = RAW / "boundary_capture_progress.json"
TOKEN_FIELD = RAW / "qwen4b_declarative_six_family_all_token.float16.npy"
CONTRIBUTIONS = OUT / "atlas/qwen4b_declarative_target_wrong_contributions.float16.npy"
sys.path.insert(0, str(TESTS))

import phase1332_bf16_utils as model_base  # noqa: E402
import phase2297_c3161_c3260_qwen4b_ntp_predictive_field as old_field  # noqa: E402
import phase2303_c3701_c3780_declarative_continuation_contract as contract  # noqa: E402


PHASE = 2304
CAMPAIGN = "C3781-C3900"
EPS = 1e-12


def checkpoint_modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def slice_accuracy(rows: list[dict], by_id: dict[str, dict]) -> dict:
    values = [by_id[row["case_id"]] for row in rows]
    return {
        "rows": len(values),
        "mean_accuracy": float(np.mean([value["correct_by_mean"] for value in values])),
        "sum_accuracy": float(np.mean([value["correct_by_sum"] for value in values])),
        "mean_margin": float(np.mean([value["mean_margin"] for value in values])),
        "sum_margin": float(np.mean([value["sum_margin"] for value in values])),
    }


def sequence_ledger(rows: list[dict], scores: list[dict]) -> dict:
    by_id = {row["case_id"]: row for row in scores}
    families, qualified = {}, []
    for family in contract.FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        slices = {"overall:all": slice_accuracy(family_rows, by_id)}
        for kind, values in (
            ("language", ("en", "zh")),
            ("surface", ("narrative", "dialogue")),
            ("partition", contract.PARTITIONS),
            ("target_mention_order", ("first", "last")),
            ("source_fact_order_matched", (True, False)),
        ):
            for value in values:
                subset = [row for row in family_rows if row[kind] == value]
                slices[f"{kind}:{value}"] = slice_accuracy(subset, by_id)
        passed = all(
            min(value["mean_accuracy"], value["sum_accuracy"]) >= contract.BEHAVIOR_GATE
            for value in slices.values()
        )
        families[family] = {"qualified": passed, "slices": slices}
        if passed:
            qualified.append(family)
    overall = slice_accuracy(rows, by_id)
    return {"overall": overall, "families": families, "qualified_families": qualified,
            "gate": contract.BEHAVIOR_GATE}


def left_pad_batch(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(sequences):
        ids[i, width - len(row):] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, width - len(row):] = 1
    return ids, mask


def starts_with(values: list[int], prefix: list[int]) -> bool:
    return len(values) >= len(prefix) and values[:len(prefix)] == prefix


def free_continuations(model, tokenizer, device, rows: list[dict], batch_size: int = 32) -> list[dict]:
    output: list[dict] = []
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        ids, mask = left_pad_batch([row["ntp_prompt_ids"] for row in batch], device, pad)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=mask, do_sample=False,
                max_new_tokens=6, use_cache=True, pad_token_id=pad,
                eos_token_id=model.config.eos_token_id,
            )
        continuations = generated[:, ids.shape[1]:].detach().cpu().tolist()
        for row, token_ids in zip(batch, continuations):
            target = [int(value) for value in row["ntp_target_ids"]]
            wrong = [int(value) for value in row["ntp_wrong_ids"]]
            output.append({
                "case_id": row["case_id"], "family": row["family"], "language": row["language"],
                "surface": row["surface"], "partition": row["partition"], "state": row["state"],
                "target_mention_order": row["target_mention_order"],
                "generated_ids": [int(value) for value in token_ids],
                "generated_text": tokenizer.decode(token_ids, skip_special_tokens=True),
                "target_prefix_exact": starts_with(token_ids, target),
                "wrong_prefix_exact": starts_with(token_ids, wrong),
            })
        print(f"[phase2304 free] {start + len(batch)}/{len(rows)}", flush=True)
    return output


def free_ledger(rows: list[dict]) -> dict:
    def summary(values: list[dict]) -> dict:
        return {
            "rows": len(values),
            "target_prefix_exact": float(np.mean([row["target_prefix_exact"] for row in values])),
            "wrong_prefix_exact": float(np.mean([row["wrong_prefix_exact"] for row in values])),
        }
    return {
        "overall": summary(rows),
        "families": {family: summary([row for row in rows if row["family"] == family])
                     for family in contract.FAMILIES},
        "status": "descriptive_not_a_frozen_behavior_gate",
    }


def capture_boundary_and_logits(model, device, rows: list[dict], batch_size: int = 10) -> dict:
    modules = checkpoint_modules(model)
    dimension, vocabulary = int(model.config.hidden_size), int(model.config.vocab_size)
    field_shape = (len(rows), len(modules), dimension)
    logit_shape = (len(rows), vocabulary)
    RAW.mkdir(parents=True, exist_ok=True)
    completed = 0
    if BOUNDARY.exists() and LOGITS.exists() and PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if progress["field_shape"] != list(field_shape) or progress["logit_shape"] != list(logit_shape):
            raise RuntimeError(("resume_shape_mismatch", progress, field_shape, logit_shape))
        completed = int(progress["completed"])
        field = np.lib.format.open_memmap(BOUNDARY, mode="r+")
        logits_file = np.lib.format.open_memmap(LOGITS, mode="r+")
    else:
        field = np.lib.format.open_memmap(BOUNDARY, mode="w+", dtype=np.float16, shape=field_shape)
        logits_file = np.lib.format.open_memmap(LOGITS, mode="w+", dtype=np.float16, shape=logit_shape)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(modules):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, position_ids = old_field.pad_batch(
                    [row["ntp_prompt_ids"] for row in batch], device, pad,
                )
                result = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                               use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(modules)):
                    selected = torch.stack([captures[q][i, ends[i]] for i in range(len(batch))])
                    field[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                selected_logits = torch.stack([result.logits[i, ends[i]] for i in range(len(batch))])
                logits_file[start:start + len(batch)] = selected_logits.float().cpu().numpy().astype(np.float16)
                field.flush()
                logits_file.flush()
                contract.save(PROGRESS, {"completed": start + len(batch), "field_shape": list(field_shape),
                                         "logit_shape": list(logit_shape)})
                print(f"[phase2304 boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return {"field_path": str(BOUNDARY.relative_to(ROOT)), "field_shape": list(field_shape),
            "logits_path": str(LOGITS.relative_to(ROOT)), "logits_shape": list(logit_shape),
            "checkpoints": len(modules), "dimension": dimension, "vocabulary": vocabulary}


def representative_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["partition"] == "fresh_lockbox" and int(row["unit"]) == 26
            and row["language"] == "en" and row["surface"] == "narrative" and int(row["state"]) == 0]


def capture_six_family_tokens(model, device, rows: list[dict]) -> dict:
    reps = representative_rows(rows)
    if len(reps) != len(contract.FAMILIES):
        raise RuntimeError(("representative_count", len(reps)))
    index_path = OUT / "index/six_family_all_token_rows.jsonl"
    if TOKEN_FIELD.exists() and index_path.exists():
        values = np.load(TOKEN_FIELD, mmap_mode="r")
        return {"path": str(TOKEN_FIELD.relative_to(ROOT)), "shape": list(values.shape),
                "cases": len(reps), "row_index": str(index_path.relative_to(ROOT)), "resumed": True}
    modules = checkpoint_modules(model)
    dimension = int(model.config.hidden_size)
    total = sum(len(row["ntp_prompt_ids"]) * len(modules) for row in reps)
    TOKEN_FIELD.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(TOKEN_FIELD, mode="w+", dtype=np.float16, shape=(total, dimension))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(modules):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    meta, cursor = [], 0
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
                    values = captures[q][0, :ids.shape[1]].float().cpu().numpy().astype(np.float16)
                    field[cursor:cursor + len(values)] = values
                    for token in range(len(values)):
                        meta.append({
                            "row": cursor + token, "case_id": row["case_id"], "family": row["family"],
                            "language": row["language"], "surface": row["surface"], "state": row["state"],
                            "checkpoint": q, "token": token, "token_id": row["ntp_prompt_ids"][token],
                        })
                    cursor += len(values)
                print(f"[phase2304 all-token] {n + 1}/{len(reps)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush()
    contract.write_rows(index_path, meta)
    return {"path": str(TOKEN_FIELD.relative_to(ROOT)), "shape": list(field.shape),
            "cases": len(reps), "row_index": str(index_path.relative_to(ROOT)), "resumed": False}


def exact_contributions(model, rows: list[dict]) -> dict:
    if CONTRIBUTIONS.exists():
        values = np.load(CONTRIBUTIONS, mmap_mode="r")
        return {"path": str(CONTRIBUTIONS.relative_to(ROOT)), "shape": list(values.shape), "resumed": True}
    field = np.load(BOUNDARY, mmap_mode="r")
    logits = np.load(LOGITS, mmap_mode="r")
    dimension = int(model.config.hidden_size)
    CONTRIBUTIONS.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(CONTRIBUTIONS, mode="w+", dtype=np.float16,
                                       shape=(len(rows), dimension))
    weight = model.lm_head.weight
    max_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            h = torch.tensor(np.asarray(field[start:start + len(batch), -1], dtype=np.float32),
                             device=weight.device, dtype=weight.dtype)
            target = torch.tensor([row["ntp_target_ids"][0] for row in batch], device=weight.device)
            wrong = torch.tensor([row["ntp_wrong_ids"][0] for row in batch], device=weight.device)
            delta_w = weight.index_select(0, target) - weight.index_select(0, wrong)
            value = (h * delta_w).float()
            output[start:start + len(batch)] = value.cpu().numpy().astype(np.float16)
            direct = np.asarray(logits[start:start + len(batch)], dtype=np.float32)
            expected = direct[np.arange(len(batch)), target.cpu().numpy()] - direct[np.arange(len(batch)), wrong.cpu().numpy()]
            max_error = max(max_error, float(np.max(np.abs(value.sum(-1).cpu().numpy() - expected))))
    output.flush()
    return {"path": str(CONTRIBUTIONS.relative_to(ROOT)), "shape": list(output.shape),
            "decomposition_max_abs_error_float16": max_error, "resumed": False}


def lens_metrics(model, rows: list[dict], qpoints: tuple[int, ...], batch_size: int = 8) -> dict:
    path = OUT / "prediction/logit_lens_metrics.jsonl"
    if path.exists():
        values = contract.read_rows(path)
        return {"path": str(path.relative_to(ROOT)), "rows": len(values),
                "qpoints": list(qpoints), "resumed": True}
    field = np.load(BOUNDARY, mmap_mode="r")
    actual_file = np.load(LOGITS, mmap_mode="r")
    output: list[dict] = []
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
                p, final_p = torch.softmax(lens, dim=-1), torch.softmax(actual, dim=-1)
                midpoint = 0.5 * (p + final_p)
                js = 0.5 * torch.sum(p * (torch.log(p + EPS) - torch.log(midpoint + EPS)), dim=-1)
                js += 0.5 * torch.sum(final_p * (torch.log(final_p + EPS) - torch.log(midpoint + EPS)), dim=-1)
                for i, row in enumerate(batch):
                    target, wrong = int(row["ntp_target_ids"][0]), int(row["ntp_wrong_ids"][0])
                    output.append({
                        "case_id": row["case_id"], "family": row["family"], "language": row["language"],
                        "surface": row["surface"], "partition": row["partition"], "state": row["state"],
                        "checkpoint": q, "js_to_actual_final": float(js[i].item()),
                        "target_wrong_margin": float((lens[i, target] - lens[i, wrong]).item()),
                        "target_probability": float(p[i, target].item()),
                        "wrong_probability": float(p[i, wrong].item()),
                    })
            print(f"[phase2304 lens] q={q}", flush=True)
    contract.write_rows(path, output)
    return {"path": str(path.relative_to(ROOT)), "rows": len(output),
            "qpoints": list(qpoints), "resumed": False}


def first_token_ledger(rows: list[dict]) -> dict:
    logits = np.load(LOGITS, mmap_mode="r")
    records = []
    for i, row in enumerate(rows):
        target, wrong = int(row["ntp_target_ids"][0]), int(row["ntp_wrong_ids"][0])
        margin = float(logits[i, target] - logits[i, wrong])
        records.append({"family": row["family"], "language": row["language"],
                        "surface": row["surface"], "partition": row["partition"],
                        "correct": margin > 0, "margin": margin})
    def summary(values: list[dict]) -> dict:
        return {"rows": len(values), "accuracy": float(np.mean([row["correct"] for row in values])),
                "mean_margin": float(np.mean([row["margin"] for row in values]))}
    return {"overall": summary(records),
            "families": {family: summary([row for row in records if row["family"] == family])
                         for family in contract.FAMILIES}}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {family: {
        "qualified": value["qualified"],
        "overall_mean_accuracy": value["slices"]["overall:all"]["mean_accuracy"],
        "overall_sum_accuracy": value["slices"]["overall:all"]["sum_accuracy"],
    } for family, value in result["sequence_ledger"]["families"].items()}
    text = rf"""

## Phase {PHASE}: Qwen3-4B 自然陈述续写行为与完整 HiddenState 场（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 严格按 Phase2303 冻结的 1536 行原始文本前缀运行本地 Qwen3-4B、BF16、非量化 CUDA。每行先对正确/错误完整候选做 teacher-forced 均值分与总分双账，再做 6 token 贪心自由续写观察；随后一次前向保存完整下一 token 词表 logits，以及 embedding、36 个 block 后、final norm 共 38 个边界检查点的全部 2560 坐标。另保存 fresh-lockbox unit26 的六族英语叙事样本在每个真实 token、每个检查点、每个坐标的场。未读取 Attention 或 MLP 内部量。

$$
\bar s(y\mid x)=\frac1{{|y|}}\sum_r\log p_\theta(y_r\mid x,y_{{<r}}),
\qquad
H_i=\{{h_{{q,p,j}}(x_i)\}}_{{q,p,j}}.
$$

final norm 对目标首 token 与错误首 token 的精确线性分账为：

$$
m_i=z_{{y_i^+}}-z_{{y_i^-}}
=\sum_{{j=1}}^{{2560}}h_{{i,j}}(W_{{y_i^+,j}}-W_{{y_i^-,j}}).
$$

**结果与门槛。** 六族完整候选行为 `{json.dumps(compact, ensure_ascii=False)}`；合格族 `{result['sequence_ledger']['qualified_families']}`，门槛为每族 overall、语言、表面、分区、提及顺序和源顺序匹配/不匹配切片的均值分与总分准确率均不低于 `{result['sequence_ledger']['gate']}`。自由续写只是冻结的描述性面板，结果 `{json.dumps(result['free_ledger'], ensure_ascii=False)}`；首 token 账 `{json.dumps(result['first_token_ledger'], ensure_ascii=False)}`。全场 `{json.dumps(result['field'], ensure_ascii=False)}`；六族全 token 场 `{json.dumps(result['token_field'], ensure_ascii=False)}`；逐坐标输出分账 `{json.dumps(result['contributions'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。完整候选通过只表示模型在这个续写接口中更偏好冻结答案，不等价于开放生成理解。自由续写的 exact-prefix 指标很严格，其他自然续写可能合法；机器材料未经过独立人类盲评。logit lens 使用统一 final norm+unembedding 辅助尺，不是模型显式共享解码器；逐坐标分账只对 final norm 后的首 token 竞争精确，不是中层因果传动图。脚本 `tests/glm5/phase2304_c3781_c3900_qwen4b_declarative_field.py`；结果 `tests/glm5/result/phase2304_c3781_c3900_qwen4b_declarative_field`。下一步只做已冻结的问答—续写逐行接口分账、状态/表面/接口完整词表距离和形成时序，不事后挑坐标。
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
        raise RuntimeError("Phase2303 contract did not pass")
    rows = contract.read_rows(ROWS_PATH)
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = model_base.load_bf16("qwen3")
        score_path = OUT / "behavior/sequence_scores.jsonl"
        if score_path.exists():
            scores = contract.read_rows(score_path)
        else:
            scores = old_field.sequence_scores(model, device, rows, batch_size=12)
            contract.write_rows(score_path, scores)
        ledger = sequence_ledger(rows, scores)
        contract.save(OUT / "behavior/sequence_ledger.json", ledger)
        free_path = OUT / "behavior/free_continuations.jsonl"
        if free_path.exists():
            free_rows = contract.read_rows(free_path)
        else:
            free_rows = free_continuations(model, tokenizer, device, rows)
            contract.write_rows(free_path, free_rows)
        free_summary = free_ledger(free_rows)
        contract.save(OUT / "behavior/free_ledger.json", free_summary)
        field = capture_boundary_and_logits(model, device, rows)
        token_field = capture_six_family_tokens(model, device, rows)
        contributions = exact_contributions(model, rows)
        lens = lens_metrics(model, rows, contract.QPOINTS_4B)
        first = first_token_ledger(rows)
        contract.save(OUT / "behavior/first_token_ledger.json", first)
        model_info = {
            "name": "Qwen3-4B", "precision": "bfloat16", "quantization": "none",
            "placement": placement, "layers": len(model.model.layers),
            "hidden_size": int(model.config.hidden_size), "vocabulary": int(model.config.vocab_size),
        }
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {
        "contract_frozen_before_load": parent["config"]["frozen_before_model_load"],
        "all_rows_sequence_scored": len(scores) == len(rows),
        "all_rows_free_generated": len(free_rows) == len(rows),
        "boundary_all_rows_all_checkpoints": field["field_shape"] == [len(rows), 38, 2560],
        "full_vocabulary_all_rows": field["logits_shape"] == [len(rows), model_info["vocabulary"]],
        "six_family_all_token_cases": token_field["cases"] == len(contract.FAMILIES),
        "all_coordinates_contributed": contributions["shape"] == [len(rows), 2560],
        "all_lens_rows": lens["rows"] == len(rows) * len(contract.QPOINTS_4B),
        "no_attention_or_mlp_internal_read": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "material_rows": len(rows), "model": model_info,
        "sequence_ledger": ledger, "free_ledger": free_summary, "first_token_ledger": first,
        "field": field, "token_field": token_field, "contributions": contributions, "lens": lens,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "hashes": {
            "rows": contract.file_hash(ROWS_PATH), "scores": contract.file_hash(score_path),
            "free": contract.file_hash(free_path), "boundary": contract.file_hash(BOUNDARY),
            "logits": contract.file_hash(LOGITS), "token_field": contract.file_hash(TOKEN_FIELD),
            "contributions": contract.file_hash(CONTRIBUTIONS),
            "lens": contract.file_hash(OUT / "prediction/logit_lens_metrics.jsonl"),
        },
        "strict_conclusion": (
            f"Qwen3-4B qualified {len(ledger['qualified_families'])}/6 families under the raw declarative "
            "continuation contract; complete output and HiddenState observations were retained for every row, "
            "including failures, without converting observational readout into a causal claim."
        ),
        "next_authorization": (
            "Pair every row to the prior QA interface, separate state, surface, fact-order, and interface effects "
            "over the complete vocabulary and full coordinates, then freeze any cross-scale replication."
        ),
    }
    contract.save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
