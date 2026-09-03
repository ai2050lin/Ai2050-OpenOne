#!/usr/bin/env python3
"""Uniform native-chat autonomous capability audit for the four local models."""
from __future__ import annotations

import gc
import json
import math
import os
import re
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2386 = RESULT / "phase2386_c18161_c18480_chat_generation_closure"
P2388 = RESULT / "phase2388_c18801_c19120_semantic_lexical_contract"
OUT = RESULT / "phase2389_c19121_c19440_crossmodel_autonomous_capability"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2389
CAMPAIGN = "C19121-C19440"
MODEL_ORDER = ("qwen4b", "qwen14b", "glm4", "deepseek7b")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2339_c7401_c7600_crossmodel_fixed_ab_replication as loader  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_line(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|[一二三四][、.])\s*", "", line.strip())
    return re.sub(r"\s+", " ", line).strip().strip('"“”')


def clean_generation(text: str) -> str:
    text = re.sub(r"(?s)^.*?</think>\s*", "", text).strip()
    return text.replace("<|endoftext|>", "").strip()


def reorder_score(source: dict, text: str, token_count: int, maximum: int, ended: bool) -> dict:
    text = clean_generation(text)
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    target = [normalize_line(sentence) for sentence in source["target_sentences"]]
    positions = [text.find(sentence) for sentence in source["sentences"]]
    target_positions = [positions[sid] for sid in source["target_order"]]
    all_present = all(position >= 0 for position in positions)
    return {
        "generated": text, "generated_tokens": token_count, "ended_with_eos": ended,
        "hit_max_new_tokens": token_count >= maximum,
        "sentence_recall": float(np.mean([position >= 0 for position in positions])),
        "all_sentences_present": all_present,
        "identity_order_exact": bool(all_present and target_positions == sorted(target_positions)),
        "first_four_lines_exact": lines[:4] == target,
        "verbatim_full_exact": lines == target,
        "extra_nonempty_lines": max(0, len(lines) - 4),
    }


def selection_score(source: dict, text: str, token_count: int, maximum: int, ended: bool) -> dict:
    text = clean_generation(text)
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    first = lines[0] if lines else ""
    target, foil = normalize_line(source["target"]), normalize_line(source["foil"])
    return {
        "generated": text, "generated_tokens": token_count, "ended_with_eos": ended,
        "hit_max_new_tokens": token_count >= maximum,
        "target_first_line_exact": first == target,
        "target_present": target in text,
        "foil_first_line_exact": first == foil,
        "extra_nonempty_lines": max(0, len(lines) - 1),
    }


def load_model(key: str):
    torch.set_num_threads(1)
    if key == "qwen4b":
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        label = "Qwen3-4B-BF16"
    elif key == "qwen14b":
        spec = loader.MODEL_SPECS[key]
        tokenizer = AutoTokenizer.from_pretrained(spec["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        print("[phase2389] loading qwen14b NF4 with explicit CUDA device map", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec["path"], quantization_config=quant, device_map={"": 0}, trust_remote_code=True,
            local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager",
        )
        label = spec["label"]
    else:
        model, tokenizer, _ = loader.load_model(key)
        label = loader.MODEL_SPECS[key]["label"]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, label


def chat_ids(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def eos_set(model, tokenizer) -> set[int]:
    values: list[int] = []
    for value in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        if value is None:
            continue
        values.extend(value if isinstance(value, list) else [value])
    return {int(value) for value in values}


def generate_task(key: str, model, tokenizer, task: str, rows: list[dict], maximum: int, batch_size: int) -> tuple[list[dict], dict]:
    path = OUT / key / "generation" / f"{task}.jsonl"
    existing = read_rows(path) if path.exists() else []
    if len(existing) > len(rows):
        raise RuntimeError(("too_many_cached_rows", path, len(existing), len(rows)))
    completed = len(existing); device = model.get_input_embeddings().weight.device
    pad = int(tokenizer.pad_token_id); eos = eos_set(model, tokenizer)
    with torch.inference_mode():
        for start in range(completed, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            sequences = [chat_ids(tokenizer, row["prompt"]) for row in batch]
            width = max(map(len, sequences))
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for local, sequence in enumerate(sequences):
                ids[local, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long, device=device)
                mask[local, width - len(sequence):] = 1
            generated = model.generate(
                input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=maximum,
                pad_token_id=pad, eos_token_id=sorted(eos) if len(eos) > 1 else next(iter(eos), None),
                use_cache=True,
            )
            suffix = generated[:, width:].detach().cpu().tolist()
            for source, tokens in zip(batch, suffix):
                ended = bool(tokens and int(tokens[-1]) in eos)
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                scored = selection_score(source, text, len(tokens), maximum, ended) if task == "semantic_selection" else reorder_score(source, text, len(tokens), maximum, ended)
                row = {key: source[key] for key in ("case_id", "family", "language", "partition")}
                if "surface" in source: row["surface"] = source["surface"]
                if "reverse" in source: row["reverse"] = source["reverse"]
                if "relation_bit" in source: row["relation_bit"] = source["relation_bit"]
                row.update(scored); append_row(path, row); existing.append(row)
            print(f"[phase2389 {key} {task}] {len(existing)}/{len(rows)}", flush=True)
    return existing, {"rows": len(existing), "maximum": maximum, "batch_size": batch_size,
                      "chat_template": True, "enable_thinking_requested": False, "normal_eos": sorted(eos)}


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def summarize_reorder(rows: list[dict]) -> dict:
    metrics = ("sentence_recall", "all_sentences_present", "identity_order_exact", "first_four_lines_exact",
               "verbatim_full_exact", "ended_with_eos", "hit_max_new_tokens")
    result = {"rows": len(rows), **{key: mean(rows, key) for key in metrics}}
    for dimension in ("family", "language", "surface", "reverse"):
        result[f"by_{dimension}"] = {value: {key: mean([row for row in rows if str(row[dimension]) == value], key) for key in metrics[:5]}
                                      for value in sorted({str(row[dimension]) for row in rows})}
    return result


def summarize_selection(rows: list[dict]) -> dict:
    metrics = ("target_first_line_exact", "target_present", "foil_first_line_exact", "ended_with_eos", "hit_max_new_tokens")
    result = {"rows": len(rows), **{key: mean(rows, key) for key in metrics}}
    for dimension in ("family", "language", "relation_bit"):
        result[f"by_{dimension}"] = {value: {key: mean([row for row in rows if str(row[dimension]) == value], key) for key in metrics[:3]}
                                      for value in sorted({str(row[dimension]) for row in rows})}
    return result


def run_model(key: str, reorder_rows: list[dict], selection_rows: list[dict]) -> dict:
    final = OUT / key / "analysis/final.json"
    if final.exists():
        return json.loads(final.read_text(encoding="utf-8"))
    if key == "qwen4b":
        old = read_rows(P2386 / "generation/qwen4b_chat_lockbox.jsonl")
        reorder_result, reorder_contract = old, {"rows": len(old), "source": "Phase2386 identical native-chat contract", "maximum": 192}
    model, tokenizer, label = load_model(key)
    try:
        batch = 6 if key == "qwen4b" else (3 if key == "qwen14b" else 4)
        if key != "qwen4b":
            reorder_result, reorder_contract = generate_task(key, model, tokenizer, "long_reorder", reorder_rows, 192, batch)
        selection_result, selection_contract = generate_task(key, model, tokenizer, "semantic_selection", selection_rows, 80, batch)
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    result = {"model": key, "model_label": label,
              "long_reorder": summarize_reorder(reorder_result), "semantic_selection": summarize_selection(selection_result),
              "contracts": {"long_reorder": reorder_contract, "semantic_selection": selection_contract}}
    save(final, result); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 四模型原生生成能力边界与主发现模型选择（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 按Qwen4B→Qwen14B→GLM4→DS7B顺序单模型驻留CUDA，统一原生chat template、请求关闭thinking、贪心、EOS和非oracle安全上限。长句任务使用Phase2378全部256条fresh unit+来源排列锁箱；新语义任务使用Phase2388全部96条fresh-unit锁箱，在两句主要词汇相同但关系相反的候选中，根据不同句法的同义query逐字选句。Qwen4B长句结果复用完全相同的Phase2386运行，避免重复计算。

$$S_m=\tfrac12\operatorname{{Exact}}_m^{{reorder}}+\tfrac12\operatorname{{Exact}}_m^{{semantic}},\qquad
m^*=\arg\max_m S_m.$$

**结果汇总。** 四模型 `{json.dumps(result['comparison'], ensure_ascii=False)}`；冻结选择 `{json.dumps(result['selection'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2389_c19121_c19440_crossmodel_autonomous_capability.py`；全部逐样本生成和汇总位于 `tests/glm5/result/phase2389_c19121_c19440_crossmodel_autonomous_capability`。

**理论进展、问题硬伤与结论。** 该Phase只选择有行为支撑的发现主模型，不把规模或高分当作机制。语义选择仍共享实体词，逐字指标也受聊天模板影响；模型量化不同，DS属于Qwen蒸馏谱系。主模型用于高成本图谱发现，Qwen4B保留为容量边界；其余模型后续只复验冻结的功能关系。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    reorder_rows = [row for row in read_rows(P2378 / "material/label_free_natural_binding.jsonl")
                    if row["task"] == "exact_copy" and row["partition"] == "fresh_joint_lockbox"]
    selection_rows = [row for row in read_rows(P2388 / "material/semantic_selection_rows.jsonl") if row["partition"] == "fresh_unit_lockbox"]
    models = {}
    for key in MODEL_ORDER:
        models[key] = run_model(key, reorder_rows, selection_rows)
    comparison = {key: {"reorder_exact": value["long_reorder"]["first_four_lines_exact"],
                        "reorder_sentence_recall": value["long_reorder"]["sentence_recall"],
                        "semantic_exact": value["semantic_selection"]["target_first_line_exact"],
                        "semantic_target_present": value["semantic_selection"]["target_present"]}
                  for key, value in models.items()}
    for value in comparison.values():
        value["selection_score"] = 0.5 * value["reorder_exact"] + 0.5 * value["semantic_exact"]
    chosen = max(MODEL_ORDER, key=lambda key: comparison[key]["selection_score"])
    checks = {"four_models": list(models) == list(MODEL_ORDER), "reorder_rows": all(v["long_reorder"]["rows"] == 256 for v in models.values()),
              "semantic_rows": all(v["semantic_selection"]["rows"] == 96 for v in models.values()),
              "finite": all(math.isfinite(v["selection_score"]) for v in comparison.values()), "sequential_order": list(models) == list(MODEL_ORDER)}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "comparison": comparison,
              "selection": {"primary_discovery_model": chosen, "criterion": "highest frozen equal-weight exact score",
                            "qwen4b_role": "capacity-boundary full-field control", "stable_threshold": 0.80,
                            "stable_models": [key for key, value in comparison.items() if value["reorder_exact"] >= 0.80 and value["semantic_exact"] >= 0.80]},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
