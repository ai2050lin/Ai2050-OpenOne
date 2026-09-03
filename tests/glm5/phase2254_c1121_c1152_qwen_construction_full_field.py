#!/usr/bin/env python3
"""Run Qwen3-4B behavior and full-coordinate construction-ecology fields."""
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
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CONTRACT_OUT = RESULT / "phase2253_c1097_c1120_construction_ecology_contract"
OUT = RESULT / "phase2254_c1121_c1152_qwen_construction_full_field"
sys.path.insert(0, str(TESTS))

import phase2253_c1097_c1120_construction_ecology_contract as contract  # noqa: E402


PHASE = 2254
CAMPAIGNS = tuple(f"C{i}" for i in range(1121, 1153))
MATERIALS = ("parent_broad", "fresh_broad", "parent_composition", "fresh_composition")
FIELD_PATH = OUT / "raw/qwen3_4b_qualified_role_field.float16.npy"
INDEX_PATH = OUT / "raw/role_field_index.jsonl"
PROGRESS_PATH = OUT / "raw/role_capture_progress.json"
TOKEN_FIELD_PATH = OUT / "raw/qwen3_4b_key_all_token_field.float16.npy"
TOKEN_INDEX_PATH = OUT / "raw/all_token_field_index.jsonl"
TOKEN_PROGRESS_PATH = OUT / "raw/all_token_capture_progress.json"


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def generation(model, tokenizer, device, rows: list[dict]) -> list[dict]:
    output = []
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    for start in range(0, len(rows), 12):
        batch = rows[start:start + 12]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6,
                                       do_sample=False, pad_token_id=pad,
                                       eos_token_id=tokenizer.eos_token_id)
        for i, row in enumerate(batch):
            text = tokenizer.decode(generated[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_code(text, row)
            output.append({"case_id": row["case_id"], "text": text, "parsed": parsed,
                           "correct_answer": row["correct_answer"],
                           "correct": parsed == row["correct_answer"]})
        if start % 120 == 0:
            print(f"[generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_ledger(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    partitions: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["panel"], row["family"])].append(row)
        partitions[(row["panel"], row["family"], row["partition"])].append(row)
    cells = {}
    for key, subset in sorted(groups.items()):
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in subset]))
        cells["|".join(key)] = {"rows": len(subset), "candidate_accuracy": ca,
                                "generation_accuracy": ga,
                                "dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE}
    partition_cells = {}
    for key, subset in sorted(partitions.items()):
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in subset]))
        partition_cells["|".join(key)] = {"rows": len(subset), "candidate_accuracy": ca,
                                          "generation_accuracy": ga,
                                          "dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE}
    ca = float(np.mean([row["correct"] for row in candidates]))
    ga = float(np.mean([row["correct"] for row in generated]))
    return {
        "rows": len(rows), "candidate_accuracy": ca, "generation_accuracy": ga,
        "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated])),
        "cells": cells, "partitions": partition_cells,
        "qualified_cells": sorted(key for key, value in cells.items() if value["dual_qualified"]),
        "aggregate_dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE,
    }


def qualified_rows(rows: list[dict], ledger: dict) -> list[dict]:
    allowed = set(ledger["qualified_cells"])
    return [row for row in rows if f"{row['panel']}|{row['family']}" in allowed]


def row_index(rows: list[dict], candidates: list[dict], generated: list[dict]) -> list[dict]:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}
    keep = []
    for i, row in enumerate(rows):
        keep.append({
            "hidden_index": i, "case_id": row["case_id"], "panel": row["panel"],
            "family": row["family"], "language": row["language"], "unit": row["unit"],
            "state": row["state"], "truth": row["truth"], "surface": row["surface"],
            "partition": row["partition"], "fresh": row["fresh"],
            "role_positions": row["role_positions"], "prompt_length": len(row["prompt_ids"]),
            "output_scheme": row["output_scheme"],
            "candidate_correct": bool(c[row["case_id"]]["correct"]),
            "generation_correct": bool(g[row["case_id"]]["correct"]),
            "depth": row.get("depth"), "variant": row.get("variant"),
            "verb_index": row.get("verb_index"), "outer_neg": row.get("outer_neg"),
            "inner_neg": row.get("inner_neg"), "cell_id": row.get("cell_id"),
        })
    return keep


def capture_role_field(model, device, rows: list[dict], index: list[dict]) -> dict:
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    shape = (len(rows), len(modules), len(contract.ROLES), int(modules[0].weight.shape[1]))
    FIELD_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if FIELD_PATH.exists() and PROGRESS_PATH.exists():
        progress = load(PROGRESS_PATH)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("role_resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        field = np.lib.format.open_memmap(FIELD_PATH, mode="r+")
    else:
        field = np.lib.format.open_memmap(FIELD_PATH, mode="w+", dtype=np.float16, shape=shape)
        save(PROGRESS_PATH, {"shape": list(shape), "completed_rows": 0})
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in modules]
    pad = int(model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id)
    try:
        for start in range(completed, len(rows), 4):
            batch = rows[start:start + 4]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, :len(seq)] = 1
            pos = mask.long().cumsum(-1) - 1
            pos.masked_fill_(mask == 0, 0)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            if len(captured) != len(modules):
                raise RuntimeError(("role_checkpoint_count", len(captured), len(modules)))
            for local_i, row in enumerate(batch):
                row_i = start + local_i
                for q, hidden in enumerate(captured):
                    values = hidden[local_i].float().cpu().numpy().astype(np.float16)
                    for role_i, role in enumerate(contract.ROLES):
                        field[row_i, q, role_i] = values[row["role_positions"][role][-1]]
            field.flush()
            done = min(start + len(batch), len(rows))
            save(PROGRESS_PATH, {"shape": list(shape), "completed_rows": done})
            if start % 64 == 0:
                print(f"[role-field] {done}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_mmap(field)
    write_rows(INDEX_PATH, index)
    return {"ran": True, "path": str(FIELD_PATH.relative_to(ROOT)), "shape": list(shape),
            "roles": list(contract.ROLES), "qualified_rows_only": True}


def token_subfield_rows(rows: list[dict]) -> list[dict]:
    spec = contract.preregistration()["all_token_subfield"]
    return [row for row in rows if row["family"] in spec["families"]
            and row["unit"] in spec["fresh_units"] and row["panel"] in spec["panels"]]


def capture_all_token_field(model, device, rows: list[dict]) -> dict:
    if not rows:
        return {"ran": False, "reason": "no_predeclared_qualified_rows"}
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    max_tokens = max(len(row["prompt_ids"]) for row in rows)
    shape = (len(rows), len(modules), max_tokens, int(modules[0].weight.shape[1]))
    completed = 0
    if TOKEN_FIELD_PATH.exists() and TOKEN_PROGRESS_PATH.exists():
        progress = load(TOKEN_PROGRESS_PATH)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("token_resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        field = np.lib.format.open_memmap(TOKEN_FIELD_PATH, mode="r+")
    else:
        field = np.lib.format.open_memmap(TOKEN_FIELD_PATH, mode="w+", dtype=np.float16, shape=shape)
        save(TOKEN_PROGRESS_PATH, {"shape": list(shape), "completed_rows": 0})
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in modules]
    pad = int(model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id)
    try:
        for start in range(completed, len(rows), 2):
            batch = rows[start:start + 2]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, :len(seq)] = 1
            pos = mask.long().cumsum(-1) - 1
            pos.masked_fill_(mask == 0, 0)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            for local_i, row in enumerate(batch):
                n = len(row["prompt_ids"])
                for q, hidden in enumerate(captured):
                    field[start + local_i, q, :n] = hidden[local_i, :n].float().cpu().numpy().astype(np.float16)
            field.flush()
            done = min(start + len(batch), len(rows))
            save(TOKEN_PROGRESS_PATH, {"shape": list(shape), "completed_rows": done})
            print(f"[all-token] {done}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_mmap(field)
    token_index = [{"hidden_index": i, "case_id": row["case_id"], "family": row["family"],
                    "language": row["language"], "unit": row["unit"], "surface": row["surface"],
                    "state": row["state"], "prompt_length": len(row["prompt_ids"]),
                    "prompt_ids": row["prompt_ids"]} for i, row in enumerate(rows)]
    write_rows(TOKEN_INDEX_PATH, token_index)
    return {"ran": True, "path": str(TOKEN_FIELD_PATH.relative_to(ROOT)), "shape": list(shape),
            "all_prompt_tokens": True, "families": sorted(set(row["family"] for row in rows))}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    cells = result["behavior"]["cells"]
    text = rf"""

## Phase {PHASE}: Qwen3-4B十构式双行为与全坐标观察场（C1121-C1152） [{stamp}]

**测试原理与用例。** 严格执行Phase2253重冻结材料，不修改十构式、词汇分区、图路径变体、输出码或门槛。候选选择与自由生成分别运行；资格按“面板×语言族”判断，失败路线只记NA，不阻断其他构式。合格路线保存embedding、36个block后状态、final norm、六个语义角色和全部2560个物理激活坐标；预注册的分类路径与态度作用域两个全新单元还保存每个真实prompt token的全部坐标。

**公式。** 观察场与关键全token子场为：

$$
\mathcal F_i=\{{H_{{i,q,r,j}}\}},\quad r\in\mathcal R,\qquad
\mathcal T_i=\{{H_{{i,q,t,j}}:0\le t<L_i\}}.
$$

本期只采集状态，不使用Attention、MLP、权重、梯度、PCA、Top-K、余弦或供体差分解释结构。

**结果汇总。** 总材料 `{result['behavior']['rows']}` 条，候选准确率 `{result['behavior']['candidate_accuracy']:.6f}`，自由生成准确率 `{result['behavior']['generation_accuracy']:.6f}`，生成可解析率 `{result['behavior']['parsed_generation_fraction']:.6f}`。按面板×族账为 `{json.dumps(cells, ensure_ascii=False)}`；合格格为 `{json.dumps(result['behavior']['qualified_cells'], ensure_ascii=False)}`。六角色场 `{json.dumps(result['role_field'], ensure_ascii=False)}`；关键全token子场 `{json.dumps(result['all_token_field'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`。

**分析、理论进展与边界。** 该结果只说明哪些构式在Qwen3-4B上有资格进入内部观察，以及状态数据已经按完整坐标保存。行为错误行不因答案错误而从已合格族中剔除。任何合格构式仍不能据此称为“语义齿轮”，全token子场也只是两个预声明切片。理论主体仍为“条件化输出场闭合理论”，RDC不变。

**问题、硬伤与结论。** 受控模板、人类盲评NA、元语言答案码、float16写盘与小模型外推限制仍在；六角色场不是全token场，而全token场的族覆盖很窄。工程检查 `{result['all_checks_passed']}`。下一步只在冻结discovery中形成逐坐标护照，在confirmation、lockbox与fresh单元上依次裁决，并将图路径和作用域组合单独分账。

**相关文件。** 脚本 `tests/glm5/phase2254_c1121_c1152_qwen_construction_full_field.py`；结果 `tests/glm5/result/phase2254_c1121_c1152_qwen_construction_full_field`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    if not load(CONTRACT_OUT / "analysis/final.json")["all_checks_passed"]:
        raise RuntimeError("Phase2253 contract is not valid")
    rows = []
    for name in MATERIALS:
        rows.extend(read_rows(CONTRACT_OUT / f"material/{name}_qwen_compiled.jsonl"))
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate case IDs across material files")
    candidate_path = OUT / "behavior/candidate.jsonl"
    generation_path = OUT / "behavior/generation.jsonl"
    model = None
    try:
        model, tokenizer, device, placement = contract.model_base.qwen_model()
        if candidate_path.exists() and generation_path.exists():
            candidates, generated = read_rows(candidate_path), read_rows(generation_path)
        else:
            candidates = contract.model_base.behavior_base.batch_behavior(model, device, rows, batch_size=20)
            generated = generation(model, tokenizer, device, rows)
            write_rows(candidate_path, candidates)
            write_rows(generation_path, generated)
        ledger = behavior_ledger(rows, candidates, generated)
        save(OUT / "behavior/ledger.json", ledger)
        observed = qualified_rows(rows, ledger)
        index = row_index(observed, candidates, generated)
        role_field = capture_role_field(model, device, observed, index) if observed else {
            "ran": False, "reason": "no_behavior_qualified_cells"}
        token_rows = token_subfield_rows(observed)
        all_token_field = capture_all_token_field(model, device, token_rows)
        quantization = contract.model_base.scope.parent.previous.model_base().quantization_audit(model)
    finally:
        if model is not None:
            contract.model_base.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    hashes = {"candidate": file_hash(candidate_path), "generation": file_hash(generation_path),
              "role_index": file_hash(INDEX_PATH) if INDEX_PATH.exists() else None,
              "role_field": file_hash(FIELD_PATH) if FIELD_PATH.exists() else None,
              "token_index": file_hash(TOKEN_INDEX_PATH) if TOKEN_INDEX_PATH.exists() else None,
              "token_field": file_hash(TOKEN_FIELD_PATH) if TOKEN_FIELD_PATH.exists() else None}
    checks = {
        "contract_valid": True, "behavior_complete": len(candidates) == len(generated) == len(rows),
        "some_route_observed": bool(ledger["qualified_cells"]),
        "role_rows_match": (not role_field.get("ran")) or role_field["shape"][0] == len(observed),
        "role_checkpoints": (not role_field.get("ran")) or role_field["shape"][1] == 38,
        "role_count": (not role_field.get("ran")) or role_field["shape"][2] == len(contract.ROLES),
        "coordinate_count": (not role_field.get("ran")) or role_field["shape"][3] == 2560,
        "token_rows_match": (not all_token_field.get("ran")) or all_token_field["shape"][0] == len(token_rows),
        "token_checkpoints": (not all_token_field.get("ran")) or all_token_field["shape"][1] == 38,
        "finite_behavior": bool(np.isfinite(ledger["candidate_accuracy"]) and np.isfinite(ledger["generation_accuracy"])),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "placement": placement,
        "quantization": quantization, "behavior": ledger, "observed_rows": len(observed),
        "role_field": role_field, "all_token_field": all_token_field,
        "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Behavior-qualified construction cells now have complete coordinate observations; no coordinate or causal mechanism is claimed.",
        "next_authorization": "Build frozen coordinate passports and adjudicate parent/fresh and composition lockboxes.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
