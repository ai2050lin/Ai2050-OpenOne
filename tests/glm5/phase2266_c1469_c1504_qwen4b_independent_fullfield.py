#!/usr/bin/env python3
"""Run Qwen3-4B behavior and full-coordinate fields on Phase 2265 material."""
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
CONTRACT_OUT = RESULT / "phase2265_c1433_c1468_independent_bilingual_contract"
OUT = RESULT / "phase2266_c1469_c1504_qwen4b_independent_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2265_c1433_c1468_independent_bilingual_contract as contract  # noqa: E402


PHASE = 2266
CAMPAIGNS = tuple(f"C{i}" for i in range(1469, 1505))
ROLE_FIELD = OUT / "raw/qwen3_4b_qualified_role_field.float16.npy"
ROLE_INDEX = OUT / "raw/role_field_index.jsonl"
ROLE_PROGRESS = OUT / "raw/role_field_progress.json"
TOKEN_FIELD = OUT / "raw/qwen3_4b_stratified_all_token_field.float16.npy"
TOKEN_INDEX = OUT / "raw/all_token_field_index.jsonl"
TOKEN_PROGRESS = OUT / "raw/all_token_field_progress.json"
TOKEN_UNITS = (16, 24)


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
    for start in range(0, len(rows), 10):
        batch = rows[start:start + 10]
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
        if start % 100 == 0:
            print(f"[generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_ledger(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}

    def summarize(subset: list[dict]) -> dict:
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in subset]))
        return {"rows": len(subset), "candidate_accuracy": ca, "generation_accuracy": ga,
                "dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE}

    family_groups: dict[str, list[dict]] = defaultdict(list)
    partition_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    language_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    surface_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        family_groups[row["family"]].append(row)
        partition_groups[(row["family"], row["partition"])].append(row)
        language_groups[(row["family"], row["language"])].append(row)
        surface_groups[(row["family"], row["surface"])].append(row)
    families = {family: summarize(subset) for family, subset in sorted(family_groups.items())}
    partitions = {"|".join(key): summarize(subset) for key, subset in sorted(partition_groups.items())}
    languages = {"|".join(key): summarize(subset) for key, subset in sorted(language_groups.items())}
    surfaces = {"|".join(key): summarize(subset) for key, subset in sorted(surface_groups.items())}
    qualified = []
    reasons = {}
    for family in contract.FAMILIES:
        required = {
            "overall": families[family],
            "discovery": partitions[f"{family}|discovery"],
            "fresh_confirmation": partitions[f"{family}|fresh_confirmation"],
        }
        passed = all(value["dual_qualified"] for value in required.values())
        reasons[family] = {"qualified": passed, "required": required}
        if passed:
            qualified.append(family)
    aggregate = summarize(rows)
    return {**aggregate,
            "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated])),
            "families": families, "partitions": partitions, "languages": languages, "surfaces": surfaces,
            "qualification_audit": reasons, "qualified_families": qualified}


def index_rows(rows: list[dict], candidates: list[dict], generated: list[dict]) -> list[dict]:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}
    return [{"hidden_index": i, "case_id": row["case_id"], "family": row["family"],
             "language": row["language"], "unit": row["unit"], "surface": row["surface"],
             "state": row["state"], "partition": row["partition"], "fresh": row["fresh"],
             "role_positions": row["role_positions"], "prompt_length": len(row["prompt_ids"]),
             "factors": row["factors"], "output_scheme": row["output_scheme"],
             "candidate_correct": bool(c[row["case_id"]]["correct"]),
             "generation_correct": bool(g[row["case_id"]]["correct"])} for i, row in enumerate(rows)]


def modules_for(model):
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def capture_role_field(model, device, rows: list[dict], index: list[dict]) -> dict:
    if not rows:
        return {"ran": False, "reason": "no_analysis_qualified_family"}
    modules = modules_for(model)
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
            "roles": list(contract.ROLES), "all_checkpoints": True, "all_coordinates": True,
            "families": sorted({row["family"] for row in rows})}


def capture_all_token_field(model, device, rows: list[dict]) -> dict:
    if not rows:
        return {"ran": False, "reason": "no_analysis_qualified_family"}
    modules = modules_for(model)
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
            "all_prompt_tokens": True, "all_checkpoints": True, "all_coordinates": True,
            "families": sorted({row["family"] for row in rows}), "units": list(TOKEN_UNITS)}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-4B独立中英双行为与全坐标场（C1469-C1504） [{stamp}]

**测试原理与用例。** 严格使用Phase2265冻结的3072行干净中英材料。候选A/B与精确自由生成独立记账；家族必须在总体、discovery和fresh confirmation三处同时满足两种准确率不低于0.75，才获得内部观察资格，fresh lockbox不参与资格筛选。合格家族保存全部样本的embedding、36个block后状态、final norm、六个功能角色和全部2560个物理激活坐标；并在每个合格家族固定抽取unit 16与24，覆盖中英、两表面、两状态，保存所有prompt token的全坐标场。

**公式。** 行为资格和两类场为：

$$
Q_f=\mathbf 1\!\left[\min_{{p\in\{{\mathrm{{all}},\mathrm{{discovery}},\mathrm{{fresh\ confirmation}}\}}}}\min(A^{{cand}}_{{f,p}},A^{{gen}}_{{f,p}})\ge 0.75\right],
$$

$$
\mathcal F_f=\{{H_{{i,q,r,j}}\}},\qquad
\mathcal T_f=\{{H_{{i,q,t,j}}\}}.
$$

这里的 `j` 是HiddenState物理激活坐标，不是模型参数；本期不读取Attention、MLP、权重或梯度，不进行PCA、Top-K、余弦筛选或差分搬运。

**结果汇总与门槛。** 行为账 `{json.dumps(result['behavior'], ensure_ascii=False)}`。六角色场 `{json.dumps(result['role_field'], ensure_ascii=False)}`；分层全token场 `{json.dumps(result['all_token_field'], ensure_ascii=False)}`；量化审计 `{json.dumps(result['quantization'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；工程检查 `{result['checks']}`，总通过 `{result['all_checks_passed']}`。

**分析、理论进展、问题与硬伤。** 内部场只对预注册行为合格家族采集，错误回答样本仍保留，不做答案后筛选。行为通过只说明受控任务可执行，不证明自然语言规则或因果机制。中英机器材料已修复乱码，但独立人类盲评仍为NA；输出码是元语言接口，float16写盘和Qwen3-4B单模型也是限制。理论主体“条件化输出场闭合理论”和RDC保持不变；本期仅建立独立观测基座。

**结论、相关文件与下一步。** `{result['strict_conclusion']}` 下一步只能按冻结顺序，用discovery拟合家族均值、自家族同坐标仿射、纯代数、共享、错家族、错配和跨检查点模型，经confirmation和fresh confirmation冻结后再揭示fresh lockbox。脚本 `tests/glm5/phase2266_c1469_c1504_qwen4b_independent_fullfield.py`；结果 `tests/glm5/result/phase2266_c1469_c1504_qwen4b_independent_fullfield`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        return load(final)
    contract_result = load(CONTRACT_OUT / "analysis/final.json")
    if not contract_result["all_checks_passed"]:
        raise RuntimeError("Phase2265 contract is invalid")
    rows = read_rows(CONTRACT_OUT / "material/independent_bilingual_qwen_compiled.jsonl")
    candidate_path = OUT / "behavior/candidate.jsonl"
    generation_path = OUT / "behavior/generation.jsonl"
    model = None
    try:
        model, tokenizer, device, placement = contract.legacy.parent.model_base.qwen_model()
        if candidate_path.exists() and generation_path.exists():
            candidates, generated = read_rows(candidate_path), read_rows(generation_path)
        else:
            candidates = contract.legacy.parent.model_base.behavior_base.batch_behavior(model, device, rows, batch_size=18)
            generated = generation(model, tokenizer, device, rows)
            write_rows(candidate_path, candidates)
            write_rows(generation_path, generated)
        ledger = behavior_ledger(rows, candidates, generated)
        save(OUT / "behavior/ledger.json", ledger)
        qualified = set(ledger["qualified_families"])
        observed = [row for row in rows if row["family"] in qualified]
        role_field = capture_role_field(model, device, observed, index_rows(observed, candidates, generated))
        token_rows = [row for row in observed if row["unit"] in TOKEN_UNITS]
        token_field = capture_all_token_field(model, device, token_rows)
        bf16 = contract.legacy.parent.model_base.scope.parent.previous.model_base()
        quantization = bf16.quantization_audit(model)
    finally:
        if model is not None:
            contract.legacy.parent.model_base.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    hashes = {"candidate": file_hash(candidate_path), "generation": file_hash(generation_path),
              "role_field": file_hash(ROLE_FIELD) if ROLE_FIELD.exists() else None,
              "role_index": file_hash(ROLE_INDEX) if ROLE_INDEX.exists() else None,
              "token_field": file_hash(TOKEN_FIELD) if TOKEN_FIELD.exists() else None,
              "token_index": file_hash(TOKEN_INDEX) if TOKEN_INDEX.exists() else None}
    checks = {
        "behavior_complete": len(candidates) == len(generated) == len(rows),
        "generation_parse_recorded": all("generated_ids" in row for row in generated),
        "qualified_logic_exact": all(ledger["qualification_audit"][family]["qualified"] == (family in qualified) for family in contract.FAMILIES),
        "role_shape": (not role_field.get("ran")) or role_field["shape"][1:] == [38, 6, 2560],
        "role_family_match": (not role_field.get("ran")) or set(role_field["families"]) == qualified,
        "token_full_coordinates": (not token_field.get("ran")) or token_field["shape"][-1] == 2560,
        "token_family_match": (not token_field.get("ran")) or set(token_field["families"]) == qualified,
        "finite_behavior": bool(np.isfinite(ledger["candidate_accuracy"]) and np.isfinite(ledger["generation_accuracy"])),
    }
    strict = (f"{len(qualified)}/12 families passed the predeclared overall, discovery, and fresh-confirmation dual behavior qualification; "
              "only those families have internal fields, with no mechanism claim.")
    result = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(), "placement": placement,
              "quantization": quantization, "behavior": ledger, "role_field": role_field,
              "all_token_field": token_field, "hashes": hashes, "checks": checks,
              "all_checks_passed": all(checks.values()), "strict_conclusion": strict,
              "next_authorization": "Run the frozen coordinate-local model tournament and reveal partitions in order."}
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
