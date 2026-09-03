"""C1017-C1030 Qwen3-4B natural six-family behavior and full-coordinate field."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CONTRACT_OUT = RESULT / "phase2247_c1001_c1016_natural_flagship_contract"
OUT = RESULT / "phase2248_c1017_c1030_qwen_natural_full_field"
sys.path.insert(0, str(TESTS))

import phase2247_c1001_c1016_natural_flagship_contract as contract


PHASE = 2248
CAMPAIGNS = tuple(f"C{i}" for i in range(1017, 1031))
FIELD_PATH = OUT / "raw/qwen3_4b_natural_role_field.float16.npy"
INDEX_PATH = OUT / "raw/field_index.jsonl"
PROGRESS_PATH = OUT / "raw/capture_progress.json"
MATERIALS = ("parent_broad", "fresh_broad", "parent_composition", "fresh_composition")


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def run_generation(model, tokenizer, device, rows: list[dict]) -> list[dict]:
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
            generated = model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False,
                pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
            )
        for i, row in enumerate(batch):
            text = tokenizer.decode(generated[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_code(text, row)
            output.append({
                "case_id": row["case_id"], "text": text, "parsed": parsed,
                "correct_answer": row["correct_answer"], "correct": parsed == row["correct_answer"],
            })
        if start % 120 == 0:
            print(f"[generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_ledger(rows: list[dict], candidate_rows: list[dict], generation_rows: list[dict]) -> dict:
    candidate = {row["case_id"]: row for row in candidate_rows}
    generated = {row["case_id"]: row for row in generation_rows}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["panel"], row["family"], row["partition"])].append(row)
    panels = {}
    family_totals: dict[str, list[dict]] = defaultdict(list)
    for key, subset in sorted(groups.items()):
        name = "|".join(key)
        row = {
            "rows": len(subset),
            "candidate_accuracy": float(np.mean([candidate[x["case_id"]]["correct"] for x in subset])),
            "generation_accuracy": float(np.mean([generated[x["case_id"]]["correct"] for x in subset])),
        }
        row["dual_qualified"] = min(row["candidate_accuracy"], row["generation_accuracy"]) >= contract.preregistration()["family_behavior_gate"]
        panels[name] = row
        family_totals[key[1]].extend(subset)
    families = {}
    for family, subset in sorted(family_totals.items()):
        c = float(np.mean([candidate[x["case_id"]]["correct"] for x in subset]))
        g = float(np.mean([generated[x["case_id"]]["correct"] for x in subset]))
        families[family] = {"rows": len(subset), "candidate_accuracy": c,
                            "generation_accuracy": g, "dual_qualified": min(c, g) >= 0.75}
    candidate_accuracy = float(np.mean([row["correct"] for row in candidate_rows]))
    generation_accuracy = float(np.mean([row["correct"] for row in generation_rows]))
    return {
        "rows": len(rows), "candidate_accuracy": candidate_accuracy,
        "generation_accuracy": generation_accuracy,
        "aggregate_dual_qualified": min(candidate_accuracy, generation_accuracy) >= 0.75,
        "families": families, "panels": panels,
        "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generation_rows])),
    }


def build_index(rows: list[dict], candidate_rows: list[dict], generation_rows: list[dict]) -> list[dict]:
    candidate = {row["case_id"]: row for row in candidate_rows}
    generated = {row["case_id"]: row for row in generation_rows}
    return [{
        "hidden_index": i, "case_id": row["case_id"], "panel": row["panel"],
        "family": row["family"], "language": row["language"], "surface": row["surface"],
        "unit": row["unit"], "partition": row["partition"], "truth": row["truth"],
        "fresh": row["fresh"], "role_positions": row["role_positions"],
        "prompt_length": len(row["prompt_ids"]), "output_scheme": row["output_scheme"],
        "candidate_correct": bool(candidate[row["case_id"]]["correct"]),
        "generation_correct": bool(generated[row["case_id"]]["correct"]),
        "composition_kind": row.get("composition_kind"), "depth": row.get("depth"),
        "shortcut": row.get("shortcut"), "verb_index": row.get("verb_index"),
        "outer_neg": row.get("outer_neg"), "inner_neg": row.get("inner_neg"),
        "cell_id": row.get("cell_id"),
    } for i, row in enumerate(rows)]


def capture_field(model, tokenizer, device, rows: list[dict], index: list[dict]) -> dict:
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    checkpoints = len(modules)
    dim = int(modules[0].weight.shape[1])
    expected_shape = (len(rows), checkpoints, len(contract.ROLES), dim)
    FIELD_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if FIELD_PATH.exists() and PROGRESS_PATH.exists():
        progress = load(PROGRESS_PATH)
        if tuple(progress.get("shape", [])) == expected_shape:
            completed = int(progress.get("completed_rows", 0))
            field = np.lib.format.open_memmap(FIELD_PATH, mode="r+")
        else:
            raise RuntimeError(("resume_shape_mismatch", progress.get("shape"), expected_shape))
    else:
        field = np.lib.format.open_memmap(FIELD_PATH, mode="w+", dtype=np.float16, shape=expected_shape)
        save(PROGRESS_PATH, {"shape": list(expected_shape), "completed_rows": 0})
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in modules]
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
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
            if len(captured) != checkpoints:
                raise RuntimeError(("checkpoint_count", len(captured), checkpoints))
            for local_i, row in enumerate(batch):
                row_i = start + local_i
                for q, hidden in enumerate(captured):
                    values = hidden[local_i].float().cpu().numpy().astype(np.float16)
                    for role_i, role in enumerate(contract.ROLES):
                        field[row_i, q, role_i] = values[row["role_positions"][role][-1]]
            field.flush()
            done = min(start + len(batch), len(rows))
            save(PROGRESS_PATH, {"shape": list(expected_shape), "completed_rows": done})
            if start % 64 == 0:
                print(f"[full-field] {done}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_mmap(field)
    write_rows(INDEX_PATH, index)
    return {"path": str(FIELD_PATH.relative_to(ROOT)), "shape": list(expected_shape),
            "checkpoints": checkpoints, "roles": list(contract.ROLES), "coordinates": dim,
            "includes_all_behavior_rows": True}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    behavior = result["behavior"]
    field = result["field"]
    families = {key: {k: value[k] for k in ("rows", "candidate_accuracy", "generation_accuracy", "dual_qualified")}
                for key, value in behavior["families"].items()}
    text = f"""

## Phase {PHASE}: Qwen3-4B自然六族双行为与全坐标场（C1017-C1030） [{stamp}]

**测试原理与材料。** 本期执行 Phase2247 冻结合同，不修改六个语言族、父/fresh词汇、双语表面、四套输出码、分区或门槛。候选选择和自由生成分别测量“能否比较两个答案”和“能否从真实助手边界自行生成答案”。只有总体双行为均不低于0.75才采集内部场；总体通过后保留全部行为正确和错误行，避免用答案正确性筛选HiddenState。

**公式与观测对象。** 每个样本保存 embedding、每个block后状态和final norm在六个语义角色的全部物理激活坐标：

$$
\\mathcal{{F}}_i=\\left\\{{H_{{i,q,r,j}}:q=0,\\ldots,Q-1;\\ r\\in\\mathcal{{R}};\\ j=1,\\ldots,d\\right\\}}.
$$

本期只建立行为资格与观测账，不用PCA、Top-K、余弦、Attention、MLP、权重或梯度寻找结构。

**结果汇总。** 共运行 `{behavior['rows']}` 条样本；候选准确率 `{behavior['candidate_accuracy']:.6f}`，自由生成准确率 `{behavior['generation_accuracy']:.6f}`，可解析生成比例 `{behavior['parsed_generation_fraction']:.6f}`，总体双行为资格为 `{behavior['aggregate_dual_qualified']}`。六族账为 `{json.dumps(families, ensure_ascii=False)}`。全坐标场状态为 `{json.dumps(field, ensure_ascii=False)}`；原始场 SHA256 为 `{result['hashes'].get('field')}`。

**分析、理论进展与边界。** 若总体行为通过，本期只说明这批自然化材料具备内部观察资格，并得到后续预测所需的完整状态账；它不说明任何坐标已经编码语义，也不说明某层执行了某种操作。族级双行为失败只把该族标为预定缺失，不会终止其余族的观察。理论主体仍是“条件化输出场闭合理论”，RDC不变，本期不授权新数学。

**问题、硬伤与结论。** 文本仍是受控生成材料，独立人类盲评为NA；答案码是元语言接口；六角色不是全部token；float16写盘会丢失低于其分辨率的数值；小模型行为资格不能外推到更大模型。工程检查 `{result['all_checks_passed']}`。下一步授权仅在冻结分区上比较共享、族特异和错族的全坐标预测，并独立评估路径与嵌套组合。

**相关文件。** 脚本 `tests/glm5/phase2248_c1017_c1030_qwen_natural_full_field.py`；结果 `tests/glm5/result/phase2248_c1017_c1030_qwen_natural_full_field`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    if not load(CONTRACT_OUT / "analysis/final.json")["all_checks_passed"]:
        raise RuntimeError("Phase2247 contract did not pass")
    rows = []
    for material in MATERIALS:
        rows.extend(read_rows(CONTRACT_OUT / f"material/{material}_qwen_compiled.jsonl"))
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(("duplicate_case_id_across_materials", len(ids), len(set(ids))))
    candidate_path = OUT / "behavior/candidate.jsonl"
    generation_path = OUT / "behavior/generation.jsonl"
    have_behavior = candidate_path.exists() and generation_path.exists()
    model = None
    try:
        model, tokenizer, device, placement = contract.prior.qwen_model()
        if have_behavior:
            candidate_rows = read_rows(candidate_path)
            generation_rows = read_rows(generation_path)
        else:
            candidate_rows = contract.prior.behavior_base.batch_behavior(
                model, device, rows, batch_size=20)
            generation_rows = run_generation(model, tokenizer, device, rows)
            write_rows(candidate_path, candidate_rows)
            write_rows(generation_path, generation_rows)
        behavior = behavior_ledger(rows, candidate_rows, generation_rows)
        save(OUT / "behavior/ledger.json", behavior)
        if behavior["aggregate_dual_qualified"]:
            index = build_index(rows, candidate_rows, generation_rows)
            field = capture_field(model, tokenizer, device, rows, index)
        else:
            field = {"ran": False, "reason": "aggregate_dual_behavior_below_0.75"}
        quantization = contract.prior.scope.parent.previous.model_base().quantization_audit(model)
    finally:
        if model is not None:
            contract.prior.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
    hashes = {
        "candidate": file_hash(candidate_path), "generation": file_hash(generation_path),
        "index": file_hash(INDEX_PATH) if INDEX_PATH.exists() else None,
        "field": file_hash(FIELD_PATH) if FIELD_PATH.exists() else None,
    }
    checks = {
        "contract_passed": True, "unique_rows": len(ids) == len(set(ids)),
        "behavior_complete": len(candidate_rows) == len(generation_rows) == len(rows),
        "field_iff_qualified": bool(field.get("ran", True)) == behavior["aggregate_dual_qualified"],
        "all_checkpoints": (not behavior["aggregate_dual_qualified"]) or field["shape"][1] == 38,
        "all_roles": (not behavior["aggregate_dual_qualified"]) or field["shape"][2] == len(contract.ROLES),
        "all_coordinates": (not behavior["aggregate_dual_qualified"]) or field["shape"][3] == 2560,
        "incorrect_rows_retained": (not behavior["aggregate_dual_qualified"]) or field["shape"][0] == len(rows),
        "finite_behavior": bool(np.isfinite(behavior["candidate_accuracy"]) and np.isfinite(behavior["generation_accuracy"])),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "rows": len(rows),
        "behavior": behavior, "field": field, "placement": placement,
        "quantization": quantization, "hashes": hashes, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "This phase qualifies observation and captures a complete six-role activation field; it makes no semantic-coordinate or causal claim.",
        "next_authorization": "Run the frozen full-coordinate predictive tournament and composition lockbox for behavior-qualified families.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps({"behavior": behavior, "field": field, "checks": checks}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
