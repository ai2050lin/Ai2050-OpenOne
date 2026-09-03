#!/usr/bin/env python3
"""Run Qwen3-4B behavior and natural bilingual all-coordinate capture."""
from __future__ import annotations

import gc
import hashlib
import json
import re
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
CONTRACT_OUT = RESULT / "phase2289_c2581_c2600_partition_lexicon_repair"
OUT = RESULT / "phase2290_c2601_c2700_qwen4b_natural_dynamic_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase1332_bf16_utils as model_base  # noqa: E402
import phase2200_c684_c709_unified_relation_response_campaign as behavior_base  # noqa: E402
import phase2288_c2501_c2580_natural_sample_condition_contract as base_contract  # noqa: E402


PHASE = 2290
CAMPAIGN = "C2601-C2700"
ROLES = base_contract.ROLES
ROLE_FIELD = OUT / "raw/qwen3_4b_natural_role_field.float16.npy"
ROLE_INDEX = OUT / "raw/role_field_index.jsonl"
ROLE_PROGRESS = OUT / "raw/role_field_progress.json"
TOKEN_FIELD = OUT / "raw/qwen3_4b_representative_all_token_field.float16.npy"
TOKEN_INDEX = OUT / "raw/all_token_field_index.jsonl"
TOKEN_PROGRESS = OUT / "raw/all_token_field_progress.json"
REPRESENTATIVE_UNIT = 26


def save(path: Path, value: Any) -> None:
    base_contract.save(path, value)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    base_contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def parse_answer(text: str, row: dict) -> str | None:
    clean = normalize(text)
    hits = []
    for answer in (row["correct_answer"], row["wrong_answer"]):
        needle = normalize(answer)
        index = clean.find(needle)
        if needle and index >= 0:
            hits.append((index, -len(needle), answer))
    return min(hits)[2] if hits else None


def free_generation(model, tokenizer, device, rows: list[dict], batch_size: int = 12) -> list[dict]:
    output = []
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            sequence = row["free_prompt_ids"]
            ids[i, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long, device=device)
            mask[i, width - len(sequence):] = 1
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8,
                                       do_sample=False, pad_token_id=pad,
                                       eos_token_id=tokenizer.eos_token_id)
        for i, row in enumerate(batch):
            new_ids = generated[i, width:].tolist()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            parsed = parse_answer(text, row)
            output.append({"case_id": row["case_id"], "text": text, "generated_ids": new_ids,
                           "parsed": parsed, "correct_answer": row["correct_answer"],
                           "correct": parsed == row["correct_answer"]})
        print(f"[generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_ledger(rows: list[dict], candidates: list[dict], generations: list[dict]) -> dict:
    candidate = {row["case_id"]: row for row in candidates}
    generated = {row["case_id"]: row for row in generations}
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    def metrics(subset: list[dict]) -> dict:
        return {
            "rows": len(subset),
            "candidate_accuracy": float(np.mean([candidate[row["case_id"]]["correct"] for row in subset])),
            "generation_accuracy": float(np.mean([generated[row["case_id"]]["correct"] for row in subset])),
        }

    families, qualified = {}, []
    for family, subset in sorted(by_family.items()):
        overall = metrics(subset)
        languages = {language: metrics([row for row in subset if row["language"] == language])
                     for language in base_contract.LANGUAGES}
        surfaces = {surface: metrics([row for row in subset if row["surface"] == surface])
                    for surface in base_contract.SURFACES}
        partitions = {part: metrics([row for row in subset if row["partition"] == part])
                      for part in base_contract.PARTITION_RANGES}
        gates = [overall, *languages.values(), *surfaces.values(), *partitions.values()]
        dual = all(min(cell["candidate_accuracy"], cell["generation_accuracy"]) >= base_contract.BEHAVIOR_GATE
                   for cell in gates)
        families[family] = {**overall, "languages": languages, "surfaces": surfaces,
                            "partitions": partitions, "dual_qualified": dual}
        if dual:
            qualified.append(family)
    return {
        "gate": base_contract.BEHAVIOR_GATE,
        "qualification_requires": "overall_and_each_language_surface_partition_dual_gate",
        "candidate_rows": len(candidates), "generation_rows": len(generations),
        "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generations])),
        "families": families, "qualified_families": qualified,
    }


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def role_index(rows: list[dict], candidates: list[dict], generations: list[dict]) -> list[dict]:
    candidate = {row["case_id"]: row for row in candidates}
    generated = {row["case_id"]: row for row in generations}
    return [{
        "hidden_index": i, "case_id": row["case_id"], "family": row["family"],
        "language": row["language"], "surface": row["surface"], "unit": row["unit"],
        "state": row["state"], "partition": row["partition"],
        "role_positions": row["role_positions"], "role_position_methods": row["role_position_methods"],
        "prompt_length": len(row["prompt_ids"]), "candidate_correct": candidate[row["case_id"]]["correct"],
        "generation_correct": generated[row["case_id"]]["correct"],
    } for i, row in enumerate(rows)]


def capture_role_field(model, device, rows: list[dict], index: list[dict], batch_size: int = 12) -> dict:
    checkpoint_modules = modules(model)
    dimension = int(model.config.hidden_size)
    shape = (len(rows), len(checkpoint_modules), len(ROLES), dimension)
    ROLE_FIELD.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if ROLE_FIELD.exists() and ROLE_PROGRESS.exists() and ROLE_INDEX.exists():
        progress = json.loads(ROLE_PROGRESS.read_text(encoding="utf-8"))
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("role_resume_shape", progress["shape"], shape))
        completed = int(progress["completed"])
    field = np.lib.format.open_memmap(ROLE_FIELD, mode="r+" if ROLE_FIELD.exists() else "w+",
                                      dtype=np.float16, shape=shape)
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in checkpoint_modules]
    try:
        pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
        for start in range(completed, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, :len(seq)] = 1
            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 0)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                      use_cache=False, return_dict=True)
            if len(captured) != len(checkpoint_modules):
                raise RuntimeError(("checkpoint_count", len(captured), len(checkpoint_modules)))
            for local_i, row in enumerate(batch):
                for q, hidden in enumerate(captured):
                    for role_i, role in enumerate(ROLES):
                        position = int(row["role_positions"][role][-1])
                        field[start + local_i, q, role_i] = hidden[local_i, position].float().cpu().numpy()
            field.flush()
            save(ROLE_PROGRESS, {"shape": list(shape), "completed": start + len(batch)})
            print(f"[role-field] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        del field
    write_rows(ROLE_INDEX, index)
    return {"path": str(ROLE_FIELD.relative_to(ROOT)), "index": str(ROLE_INDEX.relative_to(ROOT)),
            "shape": list(shape), "dtype": "float16", "all_coordinates": True,
            "checkpoints": ["embedding", "post_block_1..36", "final_norm"]}


def capture_token_field(model, device, rows: list[dict], batch_size: int = 2) -> dict:
    checkpoint_modules = modules(model)
    dimension = int(model.config.hidden_size)
    max_tokens = max(len(row["prompt_ids"]) for row in rows)
    shape = (len(rows), len(checkpoint_modules), max_tokens, dimension)
    TOKEN_FIELD.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if TOKEN_FIELD.exists() and TOKEN_PROGRESS.exists() and TOKEN_INDEX.exists():
        progress = json.loads(TOKEN_PROGRESS.read_text(encoding="utf-8"))
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("token_resume_shape", progress["shape"], shape))
        completed = int(progress["completed"])
    field = np.lib.format.open_memmap(TOKEN_FIELD, mode="r+" if TOKEN_FIELD.exists() else "w+",
                                      dtype=np.float16, shape=shape)
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in checkpoint_modules]
    try:
        pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
        for start in range(completed, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, :len(seq)] = 1
            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 0)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                      use_cache=False, return_dict=True)
            for local_i, row in enumerate(batch):
                n = len(row["prompt_ids"])
                for q, hidden in enumerate(captured):
                    field[start + local_i, q, :n] = hidden[local_i, :n].float().cpu().numpy()
            field.flush()
            save(TOKEN_PROGRESS, {"shape": list(shape), "completed": start + len(batch)})
            print(f"[token-field] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        del field
    index = [{"hidden_index": i, "case_id": row["case_id"], "family": row["family"],
              "language": row["language"], "surface": row["surface"], "state": row["state"],
              "unit": row["unit"], "prompt_length": len(row["prompt_ids"]),
              "prompt_ids": row["prompt_ids"], "role_positions": row["role_positions"]}
             for i, row in enumerate(rows)]
    write_rows(TOKEN_INDEX, index)
    return {"path": str(TOKEN_FIELD.relative_to(ROOT)), "index": str(TOKEN_INDEX.relative_to(ROOT)),
            "shape": list(shape), "dtype": "float16", "all_real_tokens": True,
            "padding_is_zero_and_excluded_by_prompt_length": True}


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-4B自然双语动态全坐标场（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 严格读取Phase2289哈希冻结的2048行材料，先独立记录A/B首token候选行为和最多8-token自由生成；构式整体、每种语言、每种表面和每个分区的两种行为都不低于 `{base_contract.BEHAVIOR_GATE}` 才进入内部观察。错误样本不后筛。合格族保存 embedding、36个block后状态、final norm、六个功能角色和全部2560坐标；另为 fresh-lockbox 的 unit26 保存全部实际prompt token。测试覆盖八类自然程序，包括“某人喜欢/高兴某人吃了某物”和两跳分类链。

**数学对象。** 原场与逐样本状态响应为：

$$
mathcal F=(H_{{i,q,r,j}}),qquad R_{{i,q,r,j}}=H_{{i,q,r,j}}^{{(1)}}-H_{{i,q,r,j}}^{{(0)}}.
$$

本期不平均、不压缩、不按幅值筛坐标，也不读取 Attention/MLP。

**结果汇总与门槛。** 双行为账 `{json.dumps(result['behavior'], ensure_ascii=False)}`；角色原场 `{json.dumps(result['role_field'], ensure_ascii=False)}`；代表全token原场 `{json.dumps(result['token_field'], ensure_ascii=False)}`；模型精度与放置 `{json.dumps(result['model'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}` 行为资格只说明模型能执行冻结任务，不证明内部坐标是语义原子。六角色取语义跨度末token是测量约定；108个中文跨度使用解码文本回退；float16写盘、人类盲评NA、中英两种语言和单模型仍是硬伤。脚本 `tests/glm5/phase2290_c2601_c2700_qwen4b_natural_dynamic_field.py`；结果 `tests/glm5/result/phase2290_c2601_c2700_qwen4b_natural_dynamic_field`。下一步只用 discovery 拟合，按 confirmation、fresh-confirmation、fresh-lockbox 顺序揭示。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    contract = json.loads((CONTRACT_OUT / "analysis/final.json").read_text(encoding="utf-8"))
    if not contract["all_checks_passed"]:
        raise RuntimeError("Phase2289 contract is not authorized")
    rows = read_rows(CONTRACT_OUT / "material/qwen_compiled.jsonl")
    candidate_path = OUT / "behavior/candidate.jsonl"
    generation_path = OUT / "behavior/generation.jsonl"
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = model_base.load_bf16("qwen3")
        quantization = model_base.quantization_audit(model)
        candidates = (read_rows(candidate_path) if candidate_path.exists()
                      else behavior_base.batch_behavior(model, device, rows, batch_size=16))
        if not candidate_path.exists():
            write_rows(candidate_path, candidates)
        generations = (read_rows(generation_path) if generation_path.exists()
                       else free_generation(model, tokenizer, device, rows))
        if not generation_path.exists():
            write_rows(generation_path, generations)
        behavior = behavior_ledger(rows, candidates, generations)
        save(OUT / "behavior/ledger.json", behavior)
        qualified = set(behavior["qualified_families"])
        observed = [row for row in rows if row["family"] in qualified]
        role_field = (capture_role_field(model, device, observed,
                                         role_index(observed, candidates, generations)) if observed else
                      {"ran": False, "reason": "no_behavior_qualified_family"})
        representative = [row for row in observed if int(row["unit"]) == REPRESENTATIVE_UNIT]
        token_field = (capture_token_field(model, device, representative) if representative else
                       {"ran": False, "reason": "no_behavior_qualified_representative"})
        model_info = {"name": "Qwen3-4B", "device": str(device), "placement": placement,
                      "quantization": quantization, "hidden_size": int(model.config.hidden_size),
                      "layers": len(model.model.layers)}
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checks = {
        "contract_authorized": contract["all_checks_passed"],
        "behavior_complete": len(candidates) == len(generations) == len(rows),
        "all_behavior_rows_retained": behavior["candidate_rows"] == behavior["generation_rows"] == len(rows),
        "role_field_matches_qualified": (not role_field.get("path")) or
            role_field["shape"][0] == sum(row["family"] in set(behavior["qualified_families"]) for row in rows),
        "all_role_coordinates": (not role_field.get("path")) or role_field["shape"][-1] == 2560,
        "all_checkpoints": (not role_field.get("path")) or role_field["shape"][1] == 38,
        "representative_all_tokens": (not token_field.get("path")) or token_field["all_real_tokens"],
        "bf16_nonquantized": model_info["quantization"]["has_bf16_parameters"] and
            not model_info["quantization"]["has_quantized_modules"],
    }
    hashes = {"candidate": file_hash(candidate_path), "generation": file_hash(generation_path),
              "role_field": file_hash(ROLE_FIELD) if ROLE_FIELD.exists() else None,
              "token_field": file_hash(TOKEN_FIELD) if TOKEN_FIELD.exists() else None}
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "behavior": behavior, "role_field": role_field,
        "token_field": token_field, "model": model_info, "hashes": hashes,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"Qwen3-4B dual-qualified {len(behavior['qualified_families'])}/8 natural bilingual families; "
                              "complete all-coordinate fields were captured only for those families. This is a behavior-qualified observation asset, not a mechanism result."),
        "next_authorization": "Run the frozen sample-conditioned all-coordinate predictor tournament on qualified families.",
    }
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
