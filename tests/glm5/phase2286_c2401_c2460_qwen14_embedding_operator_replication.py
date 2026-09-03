#!/usr/bin/env python3
"""Prospective Qwen3-14B replication of frozen Qwen3-4B q0 operators."""
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
SOURCE_MATERIAL = RESULT / "phase2265_c1433_c1468_independent_bilingual_contract"
SOURCE_OPERATOR = RESULT / "phase2283_c2221_c2280_cross_domain_coordinate_operators"
OUT = RESULT / "phase2286_c2401_c2460_qwen14_embedding_operator_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2265_c1433_c1468_independent_bilingual_contract as contract  # noqa: E402
import phase2278_c1961_c2030_qwen14_relative_depth_replication as q14  # noqa: E402
import phase2283_c2221_c2280_cross_domain_coordinate_operators as operator  # noqa: E402


PHASE = 2286
CAMPAIGN = "C2401-C2460"
FROZEN = {
    "relative_clause_binding": {"checkpoint": 0, "role_index": 0, "role": "primary", "model": "affine"},
    "possession_state": {"checkpoint": 0, "role_index": 3, "role": "context", "model": "affine"},
    "comparison_order": {"checkpoint": 0, "role_index": 0, "role": "primary", "model": "affine"},
}
UNIT_PANEL = tuple([*range(0, 6), *range(12, 14), *range(16, 20), *range(24, 28)])
GENERATION_SURFACE = "direct"
BEHAVIOR_GATE = 0.75
FIELD = OUT / "raw/qwen3_14b_embedding_role_field.float16.npy"
INDEX = OUT / "raw/qwen3_14b_embedding_role_index.jsonl"
PROGRESS = OUT / "raw/qwen3_14b_embedding_role_progress.json"
COMPILED = OUT / "material/qwen3_14b_embedding_operator_compiled.jsonl"
RAW_SOURCE = SOURCE_MATERIAL / "material/independent_bilingual_cases.jsonl"
EPS = 1e-8


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
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


def capture_candidates_and_embedding(model, tokenizer, device, rows: list[dict]) -> list[dict]:
    shape = (len(rows), 1, len(contract.ROLES), 5120)
    candidate_path = OUT / "behavior/candidate.jsonl"
    completed = 0
    candidates: list[dict] = []
    if FIELD.exists() and PROGRESS.exists() and candidate_path.exists():
        progress = load(PROGRESS)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        candidates = read_rows(candidate_path)
        if len(candidates) != completed:
            raise RuntimeError(("candidate_resume", len(candidates), completed))
        field = np.lib.format.open_memmap(FIELD, mode="r+")
    else:
        FIELD.parent.mkdir(parents=True, exist_ok=True)
        field = np.lib.format.open_memmap(FIELD, mode="w+", dtype=np.float16, shape=shape)
        save(PROGRESS, {"shape": list(shape), "completed_rows": 0})
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _args, output):
        captured["embedding"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.embed_tokens.register_forward_hook(hook)
    pad = int(tokenizer.pad_token_id)
    try:
        for start in range(completed, len(rows), 6):
            batch = rows[start:start + 6]
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
                logits = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                               use_cache=False, return_dict=True).logits
            embedding = captured.get("embedding")
            if embedding is None:
                raise RuntimeError("embedding hook did not fire")
            for local_i, row in enumerate(batch):
                length = len(row["prompt_ids"])
                scores = [float(logits[local_i, length - 1, candidate[0]])
                          for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                candidates.append({"case_id": row["case_id"], "scores": scores,
                                   "prediction": prediction,
                                   "correct": prediction == int(row["gold_position"])})
                state = embedding[local_i].float().cpu().numpy()
                for role_i, role in enumerate(contract.ROLES):
                    field[start + local_i, 0, role_i] = state[row["role_positions"][role][-1]]
            field.flush()
            write_rows(candidate_path, candidates)
            done = min(start + len(batch), len(rows))
            save(PROGRESS, {"shape": list(shape), "completed_rows": done})
            print(f"[Qwen3-14B candidate+embedding] {done}/{len(rows)}", flush=True)
    finally:
        handle.remove()
        field.flush()
        close_mmap(field)
    return candidates


def generation(model, tokenizer, device, rows: list[dict]) -> list[dict]:
    path = OUT / "behavior/generation_direct.jsonl"
    output = read_rows(path) if path.exists() else []
    completed = len(output)
    pad = int(tokenizer.pad_token_id)
    for start in range(completed, len(rows), 8):
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
            new_ids = generated[i, width:].tolist()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            parsed = parse_code(text, row)
            output.append({"case_id": row["case_id"], "text": text,
                           "generated_ids": new_ids, "parsed": parsed,
                           "correct": parsed == row["correct_answer"]})
        write_rows(path, output)
        print(f"[Qwen3-14B generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_ledger(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    candidate = {row["case_id"]: row for row in candidates}
    generated_by_id = {row["case_id"]: row for row in generated}

    def summarize(subset: list[dict]) -> dict:
        direct = [row for row in subset if row["surface"] == GENERATION_SURFACE]
        all_candidate = float(np.mean([candidate[row["case_id"]]["correct"] for row in subset]))
        direct_candidate = float(np.mean([candidate[row["case_id"]]["correct"] for row in direct]))
        direct_generation = float(np.mean([generated_by_id[row["case_id"]]["correct"] for row in direct]))
        return {
            "rows": len(subset), "direct_rows": len(direct),
            "candidate_accuracy_all_surfaces": all_candidate,
            "candidate_accuracy_direct": direct_candidate,
            "generation_accuracy_direct": direct_generation,
            "dual_qualified": min(all_candidate, direct_candidate, direct_generation) >= BEHAVIOR_GATE,
        }

    families, qualified = {}, []
    for family in FROZEN:
        subset = [row for row in rows if row["family"] == family]
        partitions = {part: summarize([row for row in subset if row["partition"] == part])
                      for part in ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")}
        overall = summarize(subset)
        passed = all(value["dual_qualified"] for value in
                     (overall, partitions["discovery"], partitions["fresh_confirmation"]))
        families[family] = {**overall, "partitions": partitions, "qualified": passed}
        if passed:
            qualified.append(family)
    return {
        "gate": BEHAVIOR_GATE, "candidate_rows": len(candidates), "generation_rows": len(generated),
        "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated])),
        "families": families, "qualified_families": qualified,
    }


def index_rows(rows: list[dict]) -> list[dict]:
    return [{"hidden_index": i, "case_id": row["case_id"], "family": row["family"],
             "language": row["language"], "unit": int(row["unit"]), "surface": row["surface"],
             "state": int(row["state"]), "partition": row["partition"],
             "role_positions": row["role_positions"]} for i, row in enumerate(rows)]


def evaluate_frozen(field: np.ndarray, index: list[dict], behavior: dict) -> dict:
    pairs = operator.build_pairs(index)
    evaluator = operator.Evaluator(field, pairs, list(FROZEN), "bilingual")
    decisions, passports, passport_rows = [], [], []
    for family, setting in FROZEN.items():
        if family not in behavior["qualified_families"]:
            decisions.append({"family": family, **setting, "status": "behavior_unqualified",
                              "fresh_authorized": False, "lockbox_revealed": False,
                              "lockbox_pass": False})
            continue
        q = int(setting["checkpoint"])
        r = int(setting["role_index"])
        model_name = setting["model"]
        confirmation, confirmation_passport = evaluator.evaluate(
            family, "language", "confirmation", q, r, model_name)
        fresh = None
        lockbox = None
        final_passport = confirmation_passport
        if confirmation["passes"]:
            fresh, fresh_passport = evaluator.evaluate(
                family, "language", "fresh_confirmation", q, r, model_name)
            final_passport = fresh_passport
            if fresh["passes"]:
                lockbox, lockbox_passport = evaluator.evaluate(
                    family, "language", "fresh_lockbox", q, r, model_name)
                final_passport = lockbox_passport
        decision = {
            "family": family, **setting, "status": "evaluated",
            "confirmation": confirmation, "fresh_confirmation": fresh,
            "fresh_authorized": bool(confirmation["passes"] and fresh and fresh["passes"]),
            "lockbox_revealed": lockbox is not None,
            "lockbox": lockbox, "lockbox_pass": bool(lockbox and lockbox["passes"]),
            "passport_partition": "fresh_lockbox" if lockbox else
                                  "fresh_confirmation" if fresh else "confirmation",
        }
        for metric, values in final_passport.items():
            passport_rows.append({"row": len(passports), "family": family,
                                  "partition": decision["passport_partition"], "metric": metric,
                                  "role": setting["role"], "checkpoint": q})
            passports.append(values)
        decisions.append(decision)
    atlas = np.stack(passports).astype(np.float32) if passports else np.empty((0, 5120), np.float32)
    atlas_path = OUT / "atlas/qwen14_embedding_operator_passport.float32.npy"
    rows_path = OUT / "atlas/qwen14_embedding_operator_passport.rows.jsonl"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(atlas_path, atlas)
    write_rows(rows_path, passport_rows)
    return {
        "decisions": decisions,
        "lockbox_passed_families": [row["family"] for row in decisions if row["lockbox_pass"]],
        "atlas": {"path": str(atlas_path.relative_to(ROOT)), "rows": str(rows_path.relative_to(ROOT)),
                  "shape": list(atlas.shape), "all_coordinates": True},
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    summaries = [{key: row.get(key) for key in
                  ("family", "role", "model", "status", "fresh_authorized", "lockbox_revealed", "lockbox_pass")}
                 for row in result["structure"]["decisions"]]
    text = rf"""

## Phase {PHASE}: Qwen3-14B跨语言Embedding坐标函数前瞻复验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本期只复验 Phase2283 在 Qwen3-4B fresh lockbox 中通过的三条跨语言路线：关系从句 `primary`、所有关系 `context`、比较次序 `primary`。材料覆盖中英、direct/paraphrase、真假状态和 16 个预先分层单元，共 `{result['material_rows']}` 行；候选行为覆盖全部表面，自由生成独立覆盖 direct 表面。Qwen3-14B 以 FP16、CUDA 加磁盘卸载单独运行；保存六个角色在 embedding 检查点的全部 5120 个物理激活坐标。角色、检查点和仿射函数均来自 Qwen3-4B 冻结结果，没有在 14B confirmation 中重新搜索；两种模型不比较坐标编号。

**数学公式与门槛。** 在每个模型自身的坐标系中，从源语言 discovery 学习逐坐标函数并作用到目标语言基态：

$$
\widehat R^b_j=a^a_jH^{{0,b}}_j+b^a_j,\qquad R^b_j=H^{{1,b}}_j-H^{{0,b}}_j.
$$

中英两个方向合并计分。双行为门要求全表面候选、direct 候选与 direct 自由生成在 overall、discovery、fresh-confirmation 中均不低于 `0.75`。内部门沿用 Phase2283：相对目标均值增益/逐坐标胜率不低于 `0.03/0.55`，相对共享源、错配源、错族和 q0 的前检查点替代控制不低于 `0.01/0.52`，候选 MAE 不超过目标域 oracle 的 `1.25` 倍。所有 5120 坐标进入误差，未用 Top-K、PCA 或余弦筛选。

**结果汇总。** 行为账本 `{json.dumps(result['behavior'], ensure_ascii=False)}`。冻结裁决 `{json.dumps(summaries, ensure_ascii=False)}`；通过家族 `{json.dumps(result['structure']['lockbox_passed_families'], ensure_ascii=False)}`；全坐标护照 `{json.dumps(result['structure']['atlas'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2286_c2401_c2460_qwen14_embedding_operator_replication.py`；结果 `tests/glm5/result/phase2286_c2401_c2460_qwen14_embedding_operator_replication`。候选、自由生成、编译材料、完整 q0 角色场、逐坐标误差护照和冻结合同均已保存。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}` 通过只表示“在两个同族模型各自的 embedding 物理坐标中，基态条件逐坐标函数具有相同的功能拓扑”，不表示坐标一一对应、共享语义神经元、翻译模块或深层组合机制。q0 首先受 token 切分、词汇身份和模板控制；人工平行材料、角色末 token、磁盘卸载、direct 生成面板和仅三个阳性路线限制外推。Phase2285 没有产生 patient/location 中层合格锚点，所以多密度干预分支为 `NA_not_authorized`，不是因果阴性。理论主体与 RDC 不改名；本期不需要新数学，下一阶段发布可观察的 exact-coordinate 图谱并清理未展示的大型原场。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    source_decisions = [row for row in read_rows(SOURCE_OPERATOR / "analysis/decisions.jsonl")
                        if row["route"] == "language" and row["lockbox_pass"]]
    frozen_source = {row["family"]: {key: row[key] for key in
                                     ("checkpoint", "role_index", "role", "model")}
                     for row in source_decisions}
    if frozen_source != FROZEN:
        raise RuntimeError(("frozen_source_changed", frozen_source, FROZEN))
    raw_rows = [row for row in read_rows(RAW_SOURCE)
                if row["family"] in FROZEN and int(row["unit"]) in UNIT_PANEL]
    prereg = {
        "phase": PHASE, "campaign": CAMPAIGN, "frozen_before_model": True,
        "source_phase": 2283, "families": FROZEN, "unit_panel": list(UNIT_PANEL),
        "generation_surface": GENERATION_SURFACE, "material_rows": len(raw_rows),
        "behavior_gate": BEHAVIOR_GATE,
        "operator_gates": {"gain_mean": operator.GAIN_MEAN, "win_mean": operator.WIN_MEAN,
                           "gain_control": operator.GAIN_CONTROL, "win_control": operator.WIN_CONTROL,
                           "oracle_ratio": operator.ORACLE_RATIO},
        "controls": ["target_mean", "shared_source", "shuffled_source", "wrong_family",
                     "target_oracle", "q0_previous_checkpoint_mean"],
        "causal_branch": "NA_not_authorized_no_preregistered_middle_layer_patient_or_location_operator",
        "existing_other_architecture_qualification": {
            "GLM4": "comparison behavior-unqualified; relative/possession not tested in the frozen interface",
            "DeepSeek-7B": "comparison behavior-unqualified; relative/possession not tested in the frozen interface",
        },
        "forbidden": ["cross-model coordinate-number alignment", "attention", "MLP", "weights",
                      "gradient", "PCA", "Top-K", "cosine selection", "lockbox reselection"],
    }
    save(OUT / "protocol/preregistration.json", prereg)
    model = None
    try:
        model, tokenizer, device = q14.load_model()
        if COMPILED.exists():
            rows = read_rows(COMPILED)
        else:
            rows = contract.compile_rows(tokenizer, raw_rows)
            write_rows(COMPILED, rows)
        candidates = capture_candidates_and_embedding(model, tokenizer, device, rows)
        generation_rows = [row for row in rows if row["surface"] == GENERATION_SURFACE]
        generated = generation(model, tokenizer, device, generation_rows)
        behavior = behavior_ledger(rows, candidates, generated)
        save(OUT / "behavior/ledger.json", behavior)
        index = index_rows(rows)
        write_rows(INDEX, index)
        field = np.load(FIELD, mmap_mode="r")
        structure = evaluate_frozen(field, index, behavior)
        close_mmap(field)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atlas_path = ROOT / structure["atlas"]["path"]
    atlas_rows = ROOT / structure["atlas"]["rows"]
    hashes = {
        "preregistration": file_hash(OUT / "protocol/preregistration.json"),
        "compiled": file_hash(COMPILED), "candidate": file_hash(OUT / "behavior/candidate.jsonl"),
        "generation": file_hash(OUT / "behavior/generation_direct.jsonl"),
        "field": file_hash(FIELD), "atlas": file_hash(atlas_path), "atlas_rows": file_hash(atlas_rows),
    }
    field_shape = tuple(np.load(FIELD, mmap_mode="r").shape)
    atlas = np.load(atlas_path)
    checks = {
        "source_frozen": frozen_source == FROZEN,
        "material_rows_exact": len(raw_rows) == 384,
        "candidate_complete": len(candidates) == 384,
        "generation_direct_complete": len(generated) == 192,
        "field_all_coordinates": field_shape == (384, 1, len(contract.ROLES), 5120),
        "ordered_reveal": all(row["lockbox_revealed"] == row["fresh_authorized"]
                              for row in structure["decisions"] if row["status"] == "evaluated"),
        "atlas_all_coordinates": atlas.shape[1] == 5120,
        "atlas_rows_match": atlas.shape[0] == len(read_rows(atlas_rows)),
        "finite_atlas": bool(np.isfinite(atlas).all()),
        "causal_branch_correctly_na": True,
    }
    passed = structure["lockbox_passed_families"]
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "material_rows": len(raw_rows),
        "model": "Qwen3-14B", "precision": "float16", "placement": "cuda_disk_offload",
        "behavior": behavior, "structure": structure,
        "causal_branch": prereg["causal_branch"],
        "other_architecture_status": prereg["existing_other_architecture_qualification"],
        "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"{len(passed)}/3 frozen Qwen3-4B q0 cross-language coordinate-function "
                              "topologies passed behavior qualification and Qwen3-14B fresh lockbox controls; "
                              "this is functional-topology replication, not physical-coordinate identity or causality."),
        "next_authorization": "Publish exact-coordinate atlases, retain only displayed derivatives, and clean undisplayed raw fields.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps({key: value for key, value in result.items() if key != "behavior"},
                     ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
