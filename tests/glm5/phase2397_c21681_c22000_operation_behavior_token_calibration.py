#!/usr/bin/env python3
"""Calibrate behavior and event-token anchors for the conditional-operation panel on four models."""
from __future__ import annotations

import gc
import json
import logging
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes.autograd._functions").disabled = True


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2396 = RESULT / "phase2396_c21361_c21680_conditional_operation_contract"
OUT = RESULT / "phase2397_c21681_c22000_operation_behavior_token_calibration"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2397
CAMPAIGN = "C21681-C22000"
MODEL_ORDER = ("qwen4b", "qwen14b", "glm4", "deepseek7b")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def append_row(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        return -1
    first = subsequence[0]
    for start, value in enumerate(sequence):
        if value == first and sequence[start:start + len(subsequence)] == subsequence:
            return start
    return -1


def common_prefix(a: list[int], b: list[int]) -> int:
    length = 0
    for x, y in zip(a, b):
        if x != y:
            break
        length += 1
    return length


def compile_rows(tokenizer, source: list[dict]) -> tuple[list[dict], dict]:
    compiled: list[dict] = []
    exact = 0
    monotonic = 0
    for row in source:
        prompt_ids = capability.chat_ids(tokenizer, row["prompt"])
        raw_ids = [int(x) for x in tokenizer.encode(row["prompt"], add_special_tokens=False)]
        start = find_subsequence(prompt_ids, raw_ids)
        trim = 0
        if start < 0:
            for trim in range(1, min(6, len(raw_ids))):
                start = find_subsequence(prompt_ids, raw_ids[:-trim])
                if start >= 0:
                    break
        else:
            exact += 1
        event_tokens = []
        previous = -1
        for event in row["events"]:
            if event["event"] == "answer_boundary":
                token_index = len(prompt_ids) - 1
            else:
                prefix = [int(x) for x in tokenizer.encode(row["prompt"][:event["char_end"]], add_special_tokens=False)]
                length = common_prefix(raw_ids, prefix)
                token_index = start + max(0, length - 1) if start >= 0 else max(0, len(capability.chat_ids(tokenizer, row["prompt"][:event["char_end"]])) - 1)
                token_index = min(token_index, len(prompt_ids) - 1)
            event_tokens.append({**event, "token_index": int(token_index)})
            previous = max(previous, token_index)
        ordered = [e["token_index"] for e in sorted(event_tokens, key=lambda e: (e["char_end"], e["char_start"]))]
        monotonic += int(all(a <= b for a, b in zip(ordered, ordered[1:])))
        target_ids = [int(x) for x in tokenizer.encode(row["answer"], add_special_tokens=False)]
        foil_ids = [int(x) for x in tokenizer.encode(row["foil"], add_special_tokens=False)]
        if not target_ids or not foil_ids:
            raise RuntimeError((row["case_id"], target_ids, foil_ids))
        item = {key: value for key, value in row.items() if key not in ("events",)}
        item.update({"prompt_ids": prompt_ids, "target_ids": target_ids, "foil_ids": foil_ids,
                     "event_tokens": event_tokens, "raw_prompt_exact_in_chat": start >= 0 and trim == 0,
                     "raw_prompt_trim_for_match": trim, "prompt_token_count": len(prompt_ids)})
        compiled.append(item)
    return compiled, {
        "rows": len(compiled), "raw_prompt_exact_rate": exact / len(compiled),
        "event_monotonic_rate": monotonic / len(compiled),
        "prompt_token_range": [min(r["prompt_token_count"] for r in compiled), max(r["prompt_token_count"] for r in compiled)],
        "event_counts": dict(Counter(len(r["event_tokens"]) for r in compiled)),
    }


def pad_right(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = 1
    return ids, mask


def score_rows(key: str, model, rows: list[dict], batch_rows: int) -> tuple[list[dict], dict]:
    path = OUT / key / "behavior/teacher_scores.jsonl"
    existing = read_rows(path) if path.exists() else []
    if len(existing) > len(rows):
        raise RuntimeError((path, len(existing), len(rows)))
    device = model.get_input_embeddings().weight.device
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    with torch.inference_mode():
        for start in range(len(existing), len(rows), batch_rows):
            batch = rows[start:start + batch_rows]
            sequences: list[list[int]] = []
            lengths: list[tuple[int, int, int]] = []
            for row in batch:
                prompt = row["prompt_ids"]
                target, foil = row["target_ids"], row["foil_ids"]
                sequences.extend((prompt + target, prompt + foil))
                lengths.append((len(prompt), len(target), len(foil)))
            ids, mask = pad_right(sequences, device, pad)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            for local, (row, (prompt_len, target_len, foil_len)) in enumerate(zip(batch, lengths)):
                target_ids, foil_ids = row["target_ids"], row["foil_ids"]
                t_index = torch.tensor(target_ids, dtype=torch.long, device=device)[:, None]
                f_index = torch.tensor(foil_ids, dtype=torch.long, device=device)[:, None]
                t_values = log_probs[2 * local, prompt_len - 1:prompt_len - 1 + target_len].gather(1, t_index).squeeze(1)
                f_values = log_probs[2 * local + 1, prompt_len - 1:prompt_len - 1 + foil_len].gather(1, f_index).squeeze(1)
                divergence = 0
                while divergence < min(target_len, foil_len) and target_ids[divergence] == foil_ids[divergence]:
                    divergence += 1
                if divergence == min(target_len, foil_len):
                    first_margin = float("nan")
                else:
                    position = prompt_len + divergence - 1
                    base_logits = logits[2 * local, position]
                    first_margin = float((base_logits[target_ids[divergence]] - base_logits[foil_ids[divergence]]).item())
                record = {k: row[k] for k in ("case_id", "task", "family", "unit", "language", "surface", "direction", "partition", "target_candidate_slot")}
                if "query_role" in row: record["query_role"] = row["query_role"]
                if "steps" in row: record["steps"] = row["steps"]
                record.update({"target_mean_logprob": float(t_values.mean().item()), "foil_mean_logprob": float(f_values.mean().item()),
                               "mean_logprob_margin": float(t_values.mean().item() - f_values.mean().item()),
                               "first_divergence_index": divergence, "first_divergence_logit_margin": first_margin})
                append_row(path, record)
                existing.append(record)
            if len(existing) % 256 == 0 or len(existing) == len(rows):
                print(f"[phase2397 {key} teacher] {len(existing)}/{len(rows)}", flush=True)
            del logits, log_probs, ids, mask
    return existing, summarize_behavior(existing, "mean_logprob_margin")


def clean_generation(text: str) -> str:
    text = re.sub(r"(?s)^.*?</think>\s*", "", text).strip()
    return text.replace("<|endoftext|>", "").strip()


def generate_lockbox(key: str, model, tokenizer, rows: list[dict], batch_size: int) -> tuple[list[dict], dict]:
    path = OUT / key / "behavior/autonomous_lockbox.jsonl"
    existing = read_rows(path) if path.exists() else []
    device = model.get_input_embeddings().weight.device
    pad = int(tokenizer.pad_token_id)
    eos = capability.eos_set(model, tokenizer)
    with torch.inference_mode():
        for start in range(len(existing), len(rows), batch_size):
            batch = rows[start:start + batch_size]
            sequences = [row["prompt_ids"] for row in batch]
            width = max(map(len, sequences))
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for local, sequence in enumerate(sequences):
                ids[local, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long, device=device)
                mask[local, width - len(sequence):] = 1
            generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=32,
                                       pad_token_id=pad, eos_token_id=sorted(eos) if len(eos) > 1 else next(iter(eos), None), use_cache=True)
            for row, tokens in zip(batch, generated[:, width:].detach().cpu().tolist()):
                text = clean_generation(tokenizer.decode(tokens, skip_special_tokens=True))
                first = re.sub(r"^[\s\-*•\d.)、]+", "", text.splitlines()[0] if text.splitlines() else "").strip().strip('"“”')
                record = {k: row[k] for k in ("case_id", "task", "family", "unit", "language", "surface", "direction", "partition", "target_candidate_slot")}
                if "query_role" in row: record["query_role"] = row["query_role"]
                if "steps" in row: record["steps"] = row["steps"]
                record.update({"answer": row["answer"], "foil": row["foil"], "generated": text,
                               "first_line": first, "exact": first == row["answer"],
                               "target_present": row["answer"] in first, "foil_exact": first == row["foil"],
                               "generated_tokens": len(tokens), "hit_limit": len(tokens) >= 32})
                append_row(path, record); existing.append(record)
            if len(existing) % 128 == 0 or len(existing) == len(rows):
                print(f"[phase2397 {key} autonomous] {len(existing)}/{len(rows)}", flush=True)
    return existing, summarize_behavior(existing, "exact")


def summarize_behavior(rows: list[dict], metric: str) -> dict:
    def one(items: list[dict]) -> dict:
        values = np.asarray([item[metric] for item in items], dtype=np.float64)
        if metric.endswith("margin"):
            return {"rows": len(items), "target_over_foil": float(np.mean(values > 0)), "mean": float(np.mean(values))}
        return {"rows": len(items), "exact": float(np.mean(values)), "target_present": float(np.mean([item.get("target_present", False) for item in items]))}
    result = one(rows)
    for dimension in ("task", "family", "language", "surface", "direction", "partition", "target_candidate_slot"):
        result[f"by_{dimension}"] = {str(value): one([row for row in rows if row[dimension] == value])
                                        for value in sorted({row[dimension] for row in rows}, key=str)}
    return result


def run_model(key: str, source: list[dict]) -> dict:
    final = OUT / key / "analysis/final.json"
    if final.exists():
        return json.loads(final.read_text(encoding="utf-8"))
    model, tokenizer, label = capability.load_model(key)
    try:
        compiled, calibration = compile_rows(tokenizer, source)
        write_path = OUT / key / "index/operation_rows.jsonl"
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in compiled), encoding="utf-8")
        teacher_batch = {"qwen4b": 16, "qwen14b": 4, "glm4": 8, "deepseek7b": 8}[key]
        generation_batch = {"qwen4b": 8, "qwen14b": 3, "glm4": 4, "deepseek7b": 4}[key]
        teacher_rows, teacher = score_rows(key, model, compiled, teacher_batch)
        lockbox = [row for row in compiled if row["partition"] in ("fresh_unit_lockbox", "fresh_composition_lockbox")]
        generation_rows, autonomous = generate_lockbox(key, model, tokenizer, lockbox, generation_batch)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    checks = {"compiled_rows": calibration["rows"] == 3072, "exact_anchor": calibration["raw_prompt_exact_rate"] >= 0.95,
              "monotonic_events": calibration["event_monotonic_rate"] == 1.0,
              "teacher_rows": len(teacher_rows) == 3072, "autonomous_rows": len(generation_rows) == 768,
              "finite_teacher": math.isfinite(teacher["mean"])}
    result = {"model": key, "model_label": label, "calibration": calibration, "teacher": teacher,
              "autonomous": autonomous, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 四模型操作族行为边界与事件token锚点校准（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Qwen3-4B BF16→Qwen3-14B NF4→GLM4 INT8→DS7B INT8依次单模型驻留CUDA。四模型均对Phase2396全部3072条反平衡任务计算正确实体与foil的完整token平均logprob、首个分歧token logit margin；对全部768条fresh-unit及fresh-composition锁箱做关闭thinking请求、贪心、正常EOS的原生chat自主生成。每个模型独立把字符事件映射到真实chat prompt token，要求原始prompt在chat序列精确包含率≥95%、事件顺序100%单调；不把字符位置直接当token位置。

$$M_{{seq}}(x)=|y|^{{-1}}\log p(y\mid x)-|\tilde y|^{{-1}}\log p(\tilde y\mid x),$$

$$M_{{div}}(x)=z_{{y_k}}(x,y_{{<k}})-z_{{\tilde y_k}}(x,y_{{<k}}),\quad k=\min\{{i:y_i\ne\tilde y_i\}}.$$

**结果汇总。** 四模型摘要 `{json.dumps(result['summary'], ensure_ascii=False)}`；冻结角色 `{json.dumps(result['frozen_roles'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2397_c21681_c22000_operation_behavior_token_calibration.py`；四模型逐样本teacher分数、自主生成、token事件索引和final位于 `tests/glm5/result/phase2397_c21681_c22000_operation_behavior_token_calibration`。

**理论进展。** 这一步给每个操作族建立了外部行为有效性边界和真正的token事件坐标，后续HiddenState图谱可按“事实源—关系词—事实目标—查询—候选—答案边界”对齐，而不是只看最后prompt token。首分歧margin比整串平均概率更接近下一token编译端，但仍只是输出行为指标。

**问题硬伤与结论。** 候选答案的token长度不同会影响整串均值，因此同时保留首分歧margin；自主输出受chat协议和thinking控制影响，DS失败不能直接解释为没有内部编码。反事实局部记录可能与预训练知识冲突；所有族与条件必须分别报告。Phase2398不按最高行为模型挑HiddenState层，而用Qwen4B低成本全场先发现局部更新规律，Qwen14B和异构模型只复验冻结结果。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source = read_rows(P2396 / "material/all_operation_rows.jsonl")
    models = {}
    for key in MODEL_ORDER:
        models[key] = run_model(key, source)
    summary = {key: {"teacher_target_over_foil": value["teacher"]["target_over_foil"], "teacher_mean_margin": value["teacher"]["mean"],
                     "autonomous_exact": value["autonomous"]["exact"], "autonomous_target_present": value["autonomous"]["target_present"],
                     "anchor_exact_rate": value["calibration"]["raw_prompt_exact_rate"]} for key, value in models.items()}
    roles = {"discovery": "qwen4b", "large_scale_replication": "qwen14b", "cross_architecture_replication": ["glm4", "deepseek7b"],
             "criterion": "roles fixed by campaign cost/architecture before local-update results; behavior qualifies individual families but does not select layers"}
    checks = {"sequential_order": list(models) == list(MODEL_ORDER), "all_models": all(v["all_checks_passed"] for v in models.values()),
              "finite": all(math.isfinite(v["teacher"]["mean"]) for v in models.values())}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "summary": summary,
              "frozen_roles": roles, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
