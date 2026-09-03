#!/usr/bin/env python3
"""Replicate selected Qwen3-4B coordinate-local predictors in Qwen3-14B."""
from __future__ import annotations

import gc
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
Q4_OPERATOR_OUT = RESULT / "phase2260_c1297_c1336_coordinate_local_operator_tournament"
OUT = RESULT / "phase2262_c1369_c1388_qwen14_coordinate_operator_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as model_worker  # noqa: E402
import phase2258_c1241_c1264_natural_construction_state_contract as contract  # noqa: E402


PHASE = 2262
CAMPAIGNS = tuple(f"C{i}" for i in range(1369, 1389))
FAMILIES = ("property_state", "recipient_binding", "quantifier_sharing")
FIELD = OUT / "raw/qwen3_14b_selected_checkpoint_field.float16.npy"
INDEX = OUT / "raw/field_index.jsonl"
PROGRESS = OUT / "raw/capture_progress.json"
RIDGE = 0.05


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


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
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    output = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
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
        if start % 48 == 0:
            print(f"[qwen14-generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}
    families = {}
    for family in FAMILIES:
        subset = [row for row in rows if row["family"] == family]
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in subset]))
        families[family] = {"rows": len(subset), "candidate_accuracy": ca,
                            "generation_accuracy": ga,
                            "dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE}
    return {"rows": len(rows), "families": families,
            "qualified_families": sorted(f for f, value in families.items() if value["dual_qualified"]),
            "candidate_accuracy": float(np.mean([row["correct"] for row in candidates])),
            "generation_accuracy": float(np.mean([row["correct"] for row in generated])),
            "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated]))}


def relative_checkpoint(q4: int, q14_layers: int) -> int:
    return max(1, min(q14_layers, int(round(q4 / 36.0 * q14_layers))))


def capture(model, device, rows: list[dict], settings: dict) -> dict:
    base = model.model
    dim = int(base.embed_tokens.weight.shape[1])
    shape = (len(rows), dim)
    FIELD.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if FIELD.exists() and PROGRESS.exists():
        progress = load(PROGRESS)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        field = np.lib.format.open_memmap(FIELD, mode="r+")
    else:
        field = np.lib.format.open_memmap(FIELD, mode="w+", dtype=np.float16, shape=shape)
        save(PROGRESS, {"shape": list(shape), "completed_rows": 0})
    checkpoints = sorted({settings[row["family"]]["q14_checkpoint"] for row in rows})
    captured: dict[int, torch.Tensor] = {}

    def make_hook(q: int):
        def hook(_module, _args, output):
            captured[q] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = [base.layers[q - 1].register_forward_hook(make_hook(q)) for q in checkpoints]
    index = []
    try:
        row_i = completed
        while row_i < len(rows):
            batch_size = 1
            while (batch_size < 8 and row_i + batch_size < len(rows)
                   and rows[row_i]["family"] == rows[row_i + batch_size]["family"]
                   and len(rows[row_i]["prompt_ids"]) == len(rows[row_i + batch_size]["prompt_ids"])):
                batch_size += 1
            batch = rows[row_i:row_i + batch_size]
            ids = torch.tensor([row["prompt_ids"] for row in batch], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = mask.long().cumsum(-1) - 1
            captured.clear()
            with torch.inference_mode():
                base(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            for local_i, row in enumerate(batch):
                setting = settings[row["family"]]
                q, role = setting["q14_checkpoint"], setting["role"]
                field[row_i + local_i] = captured[q][local_i, row["role_positions"][role][-1]].float().cpu().numpy().astype(np.float16)
                index.append({"hidden_index": row_i + local_i, "case_id": row["case_id"],
                              "family": row["family"], "language": row["language"],
                              "unit": row["unit"], "surface": row["surface"],
                              "state": row["state"], "partition": row["partition"],
                              "q4_checkpoint": setting["q4_checkpoint"],
                              "q14_checkpoint": q, "role": role})
            row_i += batch_size
            field.flush()
            save(PROGRESS, {"shape": list(shape), "completed_rows": row_i})
            if row_i % 16 <= batch_size - 1:
                print(f"[qwen14-field] {row_i}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_mmap(field)
    if INDEX.exists() and completed:
        prior = read_rows(INDEX)
        by_id = {row["case_id"]: row for row in prior}
        for row in index:
            by_id[row["case_id"]] = row
        index = [by_id[row["case_id"]] for row in rows]
    write_rows(INDEX, index)
    return {"ran": True, "path": str(FIELD.relative_to(ROOT)), "shape": list(shape),
            "all_physical_coordinates": True, "family_specific_relative_checkpoint": True}


def replication(field: np.ndarray, index: list[dict], families: list[str]) -> dict:
    output = {}
    for family in families:
        groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
        for row in index:
            if row["family"] == family:
                groups[(row["language"], row["unit"], row["surface"], row["partition"])][row["state"]] = row
        pairs = [{"partition": key[-1], "i0": states[0]["hidden_index"], "i1": states[1]["hidden_index"]}
                 for key, states in groups.items() if set(states) == {0, 1}]
        discovery = [row for row in pairs if row["partition"] == "discovery"]
        lockbox = [row for row in pairs if row["partition"] == "fresh_lockbox"]
        x = np.asarray(field[[row["i0"] for row in discovery]], np.float32)
        y = np.asarray(field[[row["i1"] for row in discovery]], np.float32) - x
        xm, ym = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - xm, y - ym
        b = np.sum(xc * yc, axis=0) / (np.sum(xc * xc, axis=0) + RIDGE)
        a = ym - b * xm
        lx = np.asarray(field[[row["i0"] for row in lockbox]], np.float32)
        ly = np.asarray(field[[row["i1"] for row in lockbox]], np.float32) - lx
        mean_error = np.mean(np.abs(ly - ym[None]), axis=0)
        affine_error = np.mean(np.abs(ly - (a[None] + b[None] * lx)), axis=0)
        gain = 1.0 - float(np.sum(affine_error)) / (float(np.sum(mean_error)) + 1e-6)
        win = float(np.mean(affine_error < mean_error))
        output[family] = {"discovery_pairs": len(discovery), "fresh_lockbox_pairs": len(lockbox),
                          "global_gain_over_family_mean": gain,
                          "coordinate_win_fraction": win,
                          "replicated": bool(gain >= contract.OPERATOR_GATES["fresh_lockbox_gain_over_mean"]
                                             and win >= contract.OPERATOR_GATES["coordinate_win_fraction"])}
    return output


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-14B同坐标条件预测跨规模前瞻确认（C1369-C1388） [{stamp}]

**测试原理与用例。** Phase2260的九族同坐标仿射fresh lockbox阳性、Phase2261严格因果阴性构成一个重要而窄的预测结果，因此按Phase2258预注册启动14B确认。为控制算力，预先选择三种不同生态：属性状态（最佳误差增益且生成局部有效）、收件人绑定（角色关系）、量词共享（唯一中层relation锚点）。使用相同3840行冻结材料中的三族320行，不增加提示或调阈值；14B先过各族双行为，再按4B相对深度映射到14B层，只保存对应角色的全部5120物理坐标。

**公式。** 相对检查点和同坐标预测为：

$$
q_{{14}}=\operatorname{{round}}\!\left(\frac{{q_4}}{{36}}\,40\right),\qquad
\widehat R_{{i,j}}=a_j+b_jH^{{(0)}}_{{i,j}}.
$$

14B参数只用父discovery拟合，直接在fresh lockbox比较族均值；不对齐4B/14B物理坐标编号。

**结果汇总。** 14B行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；相对检查点 `{json.dumps(result['settings'], ensure_ascii=False)}`；全坐标场 `{json.dumps(result['field'], ensure_ascii=False)}`；fresh lockbox复现 `{json.dumps(result['replication'], ensure_ascii=False)}`；正式跨规模复现族 `{json.dumps(result['replicated_families'], ensure_ascii=False)}`。

**分析、理论进展与边界。** 复现表示“模型内部自身坐标上，当前基态同坐标值相对族均值提供额外预测信息”可跨4B/14B规模出现；它不表示相同坐标、相同权重、相同因果电路或语言通用数学结构。失败族只关闭本次相对检查点确认，不反证该族存在其他形成层。

**问题、硬伤、结论与相关文件。** 只测三族和单个相对检查点；14B仍属Qwen3同架构；材料人类盲评NA；模型采用磁盘卸载，字段只含选定角色。工程检查 `{result['all_checks_passed']}`。脚本 `tests/glm5/phase2262_c1369_c1388_qwen14_coordinate_operator_replication.py`；结果 `tests/glm5/result/phase2262_c1369_c1388_qwen14_coordinate_operator_replication`。下一步将Q4全坐标增益、逐token样本场、生成边界和Q14确认字段接入可视化客户端并完成清理审计。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        return load(final)
    q4 = load(Q4_OPERATOR_OUT / "analysis/final.json")
    q4_selected = q4["analysis"]["selected"]
    raw = [row for row in read_rows(CONTRACT_OUT / "material/natural_construction_cases.jsonl")
           if row["family"] in FAMILIES]
    model = None
    try:
        model, tokenizer, device, placement, loader = model_worker.load_model("qwen3_14b")
        compiled = contract.compile_rows(tokenizer, raw)
        write_rows(OUT / "material/qwen14_compiled.jsonl", compiled)
        candidate_path = OUT / "behavior/candidate.jsonl"
        generation_path = OUT / "behavior/generation.jsonl"
        if candidate_path.exists() and generation_path.exists():
            candidates, generated = read_rows(candidate_path), read_rows(generation_path)
        else:
            candidates = contract.parent.model_base.behavior_base.batch_behavior(model, device, compiled, batch_size=8)
            generated = generation(model, tokenizer, device, compiled)
            write_rows(candidate_path, candidates)
            write_rows(generation_path, generated)
        ledger = behavior(compiled, candidates, generated)
        settings = {}
        q14_layers = len(model.model.layers)
        for family in FAMILIES:
            q4_setting = q4_selected[family]
            settings[family] = {"q4_checkpoint": int(q4_setting["checkpoint"]),
                                "q14_checkpoint": relative_checkpoint(int(q4_setting["checkpoint"]), q14_layers),
                                "role": q4_setting["role"]}
        observed = [row for row in compiled if row["family"] in ledger["qualified_families"]]
        field_info = capture(model, device, observed, settings) if observed else {
            "ran": False, "reason": "no_selected_family_passed_dual_behavior"}
    finally:
        model_worker.release_model("qwen3_14b", model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if field_info.get("ran"):
        field = np.load(FIELD, mmap_mode="r")
        index = read_rows(INDEX)
        try:
            repl = replication(field, index, ledger["qualified_families"])
        finally:
            close_mmap(field)
    else:
        repl = {family: {"replicated": False, "reason": "behavior_unqualified"} for family in FAMILIES}
    replicated = sorted(family for family, value in repl.items() if value.get("replicated"))
    checks = {
        "three_families_only": len(raw) == 960 and set(row["family"] for row in raw) == set(FAMILIES),
        "behavior_complete": ledger["rows"] == 960,
        "own_tokenizer_compilation": True,
        "relative_checkpoint_frozen": set(settings) == set(FAMILIES),
        "field_matches_qualification": field_info.get("ran", False) == bool(ledger["qualified_families"]),
        "all_coordinates": (not field_info.get("ran")) or field_info["shape"][1] == 5120,
        "replication_complete": set(repl) == set(ledger["qualified_families"]) if field_info.get("ran") else True,
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "loader": loader,
        "placement": placement, "behavior": ledger, "settings": settings,
        "field": field_info, "replication": repl, "replicated_families": replicated,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Replication is model-local and relative-depth only; it does not align physical coordinate identities or establish causality.",
        "next_authorization": "Publish exact-coordinate atlases, verify the client, and clean undisplayed raw sample fields.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
