#!/usr/bin/env python3
"""Run Qwen3-4B behavior and qualified full-coordinate natural fields."""
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
CONTRACT_OUT = RESULT / "phase2258_c1241_c1264_natural_construction_state_contract"
OUT = RESULT / "phase2259_c1265_c1296_qwen_natural_full_token_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2258_c1241_c1264_natural_construction_state_contract as contract  # noqa: E402


PHASE = 2259
CAMPAIGNS = tuple(f"C{i}" for i in range(1265, 1297))
ROLE_FIELD = OUT / "raw/qwen3_4b_qualified_role_field.float16.npy"
ROLE_INDEX = OUT / "raw/role_field_index.jsonl"
ROLE_PROGRESS = OUT / "raw/role_field_progress.json"
TOKEN_FIELD = OUT / "raw/qwen3_4b_anchor_all_token_field.float16.npy"
TOKEN_INDEX = OUT / "raw/all_token_field_index.jsonl"
TOKEN_PROGRESS = OUT / "raw/all_token_field_progress.json"
GEN_FIELD = OUT / "raw/qwen3_4b_generation_boundary_field.float16.npy"
GEN_INDEX = OUT / "raw/generation_boundary_index.jsonl"
GEN_PROGRESS = OUT / "raw/generation_boundary_progress.json"
MAX_GENERATION_STEPS = 3


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
            new_ids = generated[i, width:].tolist()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            parsed = parse_code(text, row)
            output.append({"case_id": row["case_id"], "text": text, "generated_ids": new_ids,
                           "parsed": parsed, "correct_answer": row["correct_answer"],
                           "correct": parsed == row["correct_answer"]})
        if start % 120 == 0:
            print(f"[generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_ledger(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}
    family_groups: dict[str, list[dict]] = defaultdict(list)
    partition_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    surface_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        family_groups[row["family"]].append(row)
        partition_groups[(row["family"], row["partition"])].append(row)
        surface_groups[(row["family"], row["surface"])].append(row)

    def summarize(subset: list[dict]) -> dict:
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in subset]))
        return {"rows": len(subset), "candidate_accuracy": ca, "generation_accuracy": ga,
                "dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE}

    families = {family: summarize(subset) for family, subset in sorted(family_groups.items())}
    partitions = {"|".join(key): summarize(subset) for key, subset in sorted(partition_groups.items())}
    surfaces = {"|".join(key): summarize(subset) for key, subset in sorted(surface_groups.items())}
    aggregate = summarize(rows)
    return {
        **aggregate,
        "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated])),
        "families": families, "partitions": partitions, "surfaces": surfaces,
        "qualified_families": sorted(family for family, value in families.items() if value["dual_qualified"]),
    }


def index_rows(rows: list[dict], candidates: list[dict], generated: list[dict]) -> list[dict]:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}
    return [{
        "hidden_index": i, "case_id": row["case_id"], "family": row["family"],
        "language": row["language"], "unit": row["unit"], "surface": row["surface"],
        "state": row["state"], "partition": row["partition"], "fresh": row["fresh"],
        "role_positions": row["role_positions"], "prompt_length": len(row["prompt_ids"]),
        "factors": row["factors"], "output_scheme": row["output_scheme"],
        "candidate_correct": bool(c[row["case_id"]]["correct"]),
        "generation_correct": bool(g[row["case_id"]]["correct"]),
    } for i, row in enumerate(rows)]


def capture_role_field(model, device, rows: list[dict], index: list[dict]) -> dict:
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    shape = (len(rows), len(modules), len(contract.ROLES), int(modules[0].weight.shape[1]))
    ROLE_FIELD.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if ROLE_FIELD.exists() and ROLE_PROGRESS.exists():
        progress = load(ROLE_PROGRESS)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("role_resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        field = np.lib.format.open_memmap(ROLE_FIELD, mode="r+")
    else:
        field = np.lib.format.open_memmap(ROLE_FIELD, mode="w+", dtype=np.float16, shape=shape)
        save(ROLE_PROGRESS, {"shape": list(shape), "completed_rows": 0})
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
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != len(modules):
                raise RuntimeError(("checkpoint_count", len(captured), len(modules)))
            for local_i, row in enumerate(batch):
                for q, hidden in enumerate(captured):
                    values = hidden[local_i].float().cpu().numpy().astype(np.float16)
                    for role_i, role in enumerate(contract.ROLES):
                        field[start + local_i, q, role_i] = values[row["role_positions"][role][-1]]
            field.flush()
            done = min(start + len(batch), len(rows))
            save(ROLE_PROGRESS, {"shape": list(shape), "completed_rows": done})
            if start % 64 == 0:
                print(f"[role-field] {done}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_mmap(field)
    write_rows(ROLE_INDEX, index)
    return {"ran": True, "path": str(ROLE_FIELD.relative_to(ROOT)), "shape": list(shape),
            "roles": list(contract.ROLES), "all_coordinates": True}


def capture_all_token_field(model, device, rows: list[dict]) -> dict:
    if not rows:
        return {"ran": False, "reason": "no_qualified_anchor_family"}
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    max_tokens = max(len(row["prompt_ids"]) for row in rows)
    shape = (len(rows), len(modules), max_tokens, int(modules[0].weight.shape[1]))
    completed = 0
    if TOKEN_FIELD.exists() and TOKEN_PROGRESS.exists():
        progress = load(TOKEN_PROGRESS)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("token_resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        field = np.lib.format.open_memmap(TOKEN_FIELD, mode="r+")
    else:
        field = np.lib.format.open_memmap(TOKEN_FIELD, mode="w+", dtype=np.float16, shape=shape)
        save(TOKEN_PROGRESS, {"shape": list(shape), "completed_rows": 0})
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in modules]
    try:
        for start in range(completed, len(rows), 2):
            batch = rows[start:start + 2]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, :len(seq)] = 1
            pos = mask.long().cumsum(-1) - 1
            pos.masked_fill_(mask == 0, 0)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            for local_i, row in enumerate(batch):
                n = len(row["prompt_ids"])
                for q, hidden in enumerate(captured):
                    field[start + local_i, q, :n] = hidden[local_i, :n].float().cpu().numpy().astype(np.float16)
            field.flush()
            done = min(start + len(batch), len(rows))
            save(TOKEN_PROGRESS, {"shape": list(shape), "completed_rows": done})
            if start % 16 == 0:
                print(f"[all-token] {done}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_mmap(field)
    token_index = [{"hidden_index": i, "case_id": row["case_id"], "family": row["family"],
                    "language": row["language"], "unit": row["unit"], "surface": row["surface"],
                    "state": row["state"], "partition": row["partition"],
                    "prompt_length": len(row["prompt_ids"]), "prompt_ids": row["prompt_ids"],
                    "role_positions": row["role_positions"]} for i, row in enumerate(rows)]
    write_rows(TOKEN_INDEX, token_index)
    return {"ran": True, "path": str(TOKEN_FIELD.relative_to(ROOT)), "shape": list(shape),
            "all_prompt_tokens": True, "all_coordinates": True,
            "families": sorted({row["family"] for row in rows})}


def capture_generation_boundaries(model, device, rows: list[dict], generated: list[dict]) -> dict:
    if not rows:
        return {"ran": False, "reason": "no_qualified_anchor_family"}
    generated_map = {row["case_id"]: row for row in generated}
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    shape = (len(rows), MAX_GENERATION_STEPS + 1, len(modules), int(modules[0].weight.shape[1]))
    completed = 0
    if GEN_FIELD.exists() and GEN_PROGRESS.exists():
        progress = load(GEN_PROGRESS)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("generation_resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        field = np.lib.format.open_memmap(GEN_FIELD, mode="r+")
    else:
        field = np.lib.format.open_memmap(GEN_FIELD, mode="w+", dtype=np.float16, shape=shape)
        save(GEN_PROGRESS, {"shape": list(shape), "completed_rows": 0})
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in modules]
    index = []
    try:
        for row_i in range(completed, len(rows)):
            row = rows[row_i]
            new_ids = generated_map[row["case_id"]]["generated_ids"][:MAX_GENERATION_STEPS]
            valid_steps = len(new_ids) + 1
            for step in range(valid_steps):
                seq = row["free_prompt_ids"] + new_ids[:step]
                ids = torch.tensor([seq], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                pos = mask.long().cumsum(-1) - 1
                captured.clear()
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask, position_ids=pos,
                          use_cache=False, return_dict=True)
                for q, hidden in enumerate(captured):
                    field[row_i, step, q] = hidden[0, -1].float().cpu().numpy().astype(np.float16)
            field.flush()
            index.append({"hidden_index": row_i, "case_id": row["case_id"], "family": row["family"],
                          "language": row["language"], "unit": row["unit"], "surface": row["surface"],
                          "state": row["state"], "valid_steps": valid_steps, "generated_ids": new_ids})
            save(GEN_PROGRESS, {"shape": list(shape), "completed_rows": row_i + 1})
            if row_i % 16 == 0:
                print(f"[generation-boundary] {row_i + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_mmap(field)
    if GEN_INDEX.exists() and completed:
        prior = read_rows(GEN_INDEX)
        by_id = {row["case_id"]: row for row in prior}
        for row in index:
            by_id[row["case_id"]] = row
        index = [by_id[row["case_id"]] for row in rows]
    write_rows(GEN_INDEX, index)
    return {"ran": True, "path": str(GEN_FIELD.relative_to(ROOT)), "shape": list(shape),
            "steps": list(range(MAX_GENERATION_STEPS + 1)), "all_checkpoints": True,
            "all_coordinates": True}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-4B十二构式双行为与全token全坐标观察（C1265-C1296） [{stamp}]

**测试原理与用例。** 严格使用Phase2258冻结的3840行材料。候选A/B和精确自由生成独立记账，每个构式整体双行为均不低于0.75才进入HiddenState观察。合格构式保存embedding、36个block后状态、final norm、六角色、全部2560坐标；预注册的施事绑定、属性状态、位置状态若合格，还保存fresh lockbox每个真实prompt token的全部检查点与坐标，并对自由生成前缀的0-3步保存全检查点答案边界状态。状态为真假事实差，不把它命名为纯语态或纯更新算子。

**公式。** 全角色场、全token场和生成边界轨迹为：

$$
\mathcal F_i=\{{H_{{i,q,r,j}}\}},\quad
\mathcal T_i=\{{H_{{i,q,t,j}}\}},\quad
\mathcal G_i=\{{H_{{i,s,q,\partial,j}}:s=0,1,2,3\}}.
$$

本期只观察embedding与HiddenState，不读取Attention、MLP、权重或梯度，不使用PCA、Top-K、余弦或供体差分发现规律。

**结果汇总。** 行为账 `{json.dumps(result['behavior'], ensure_ascii=False)}`。六角色场 `{json.dumps(result['role_field'], ensure_ascii=False)}`；全token场 `{json.dumps(result['all_token_field'], ensure_ascii=False)}`；生成边界场 `{json.dumps(result['generation_boundary_field'], ensure_ascii=False)}`；文件哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`。

**分析、理论进展与边界。** 双行为合格只说明该受控构式可进入内部观察，不说明模型在开放自然语言中掌握对应规则。合格族内部保留全部行为正确与错误行，不做答案后筛选。全token字段是逐样本物理位置场；跨表面比较仍必须使用功能角色或逐样本观察，不能直接平均错位token。理论主体“条件化输出场闭合理论”和RDC保持不变。

**问题、硬伤、结论与相关文件。** 人类盲评NA、研究者编写材料、元语言输出码、float16写盘和Qwen3-4B单模型限制仍在。工程检查 `{result['all_checks_passed']}`。下一步只使用discovery拟合三种逐坐标基础模型，经confirmation和fresh confirmation冻结后才揭示fresh lockbox。脚本 `tests/glm5/phase2259_c1265_c1296_qwen_natural_full_token_field.py`；结果 `tests/glm5/result/phase2259_c1265_c1296_qwen_natural_full_token_field`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        return load(final)
    if not load(CONTRACT_OUT / "analysis/final.json")["all_checks_passed"]:
        raise RuntimeError("Phase2258 contract is not valid")
    rows = read_rows(CONTRACT_OUT / "material/natural_construction_qwen_compiled.jsonl")
    candidate_path = OUT / "behavior/candidate.jsonl"
    generation_path = OUT / "behavior/generation.jsonl"
    model = None
    try:
        model, tokenizer, device, placement = contract.parent.model_base.qwen_model()
        if candidate_path.exists() and generation_path.exists():
            candidates, generated = read_rows(candidate_path), read_rows(generation_path)
        else:
            candidates = contract.parent.model_base.behavior_base.batch_behavior(model, device, rows, batch_size=20)
            generated = generation(model, tokenizer, device, rows)
            write_rows(candidate_path, candidates)
            write_rows(generation_path, generated)
        ledger = behavior_ledger(rows, candidates, generated)
        save(OUT / "behavior/ledger.json", ledger)
        qualified = set(ledger["qualified_families"])
        observed = [row for row in rows if row["family"] in qualified]
        role_field = capture_role_field(model, device, observed, index_rows(observed, candidates, generated)) if observed else {
            "ran": False, "reason": "no_behavior_qualified_family"}
        anchor_rows = [row for row in observed if row["family"] in contract.ANCHOR_FAMILIES
                       and row["partition"] == "fresh_lockbox"]
        token_field = capture_all_token_field(model, device, anchor_rows)
        generation_field = capture_generation_boundaries(model, device, anchor_rows, generated)
        quantization = contract.parent.model_base.scope.parent.previous.model_base().quantization_audit(model)
    finally:
        if model is not None:
            contract.parent.model_base.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    hashes = {
        "candidate": file_hash(candidate_path), "generation": file_hash(generation_path),
        "role_field": file_hash(ROLE_FIELD) if ROLE_FIELD.exists() else None,
        "role_index": file_hash(ROLE_INDEX) if ROLE_INDEX.exists() else None,
        "token_field": file_hash(TOKEN_FIELD) if TOKEN_FIELD.exists() else None,
        "token_index": file_hash(TOKEN_INDEX) if TOKEN_INDEX.exists() else None,
        "generation_field": file_hash(GEN_FIELD) if GEN_FIELD.exists() else None,
        "generation_index": file_hash(GEN_INDEX) if GEN_INDEX.exists() else None,
    }
    checks = {
        "behavior_complete": len(candidates) == len(generated) == len(rows),
        "generation_parse_recorded": all("generated_ids" in row for row in generated),
        "role_family_match": (not role_field.get("ran")) or role_field["shape"][0] == len(observed),
        "role_shape": (not role_field.get("ran")) or role_field["shape"][1:] == [38, 6, 2560],
        "token_anchor_only": (not token_field.get("ran")) or set(token_field["families"]) <= set(contract.ANCHOR_FAMILIES),
        "token_full_coordinates": (not token_field.get("ran")) or token_field["shape"][-1] == 2560,
        "generation_full_coordinates": (not generation_field.get("ran")) or generation_field["shape"][-2:] == [38, 2560],
        "finite_behavior": bool(np.isfinite(ledger["candidate_accuracy"]) and np.isfinite(ledger["generation_accuracy"])),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "placement": placement,
        "quantization": quantization, "behavior": ledger, "role_field": role_field,
        "all_token_field": token_field, "generation_boundary_field": generation_field,
        "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Only dual-behavior-qualified controlled constructions have internal fields; state contrasts remain truth-conditioned responses, not isolated operators.",
        "next_authorization": "Fit the frozen coordinate-local model tournament using discovery only and reveal lockboxes in order.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
