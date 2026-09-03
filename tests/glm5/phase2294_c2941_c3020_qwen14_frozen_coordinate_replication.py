#!/usr/bin/env python3
"""Prospective Qwen3-14B replication of five frozen Qwen3-4B cells."""
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
FREEZE_OUT = RESULT / "phase2293_c2901_c2940_causal_na_crossscale_freeze"
OUT = RESULT / "phase2294_c2941_c3020_qwen14_frozen_coordinate_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2278_c1961_c2030_qwen14_relative_depth_replication as q14  # noqa: E402
import phase2288_c2501_c2580_natural_sample_condition_contract as contract  # noqa: E402
import phase2290_c2601_c2700_qwen4b_natural_dynamic_field as behavior  # noqa: E402
import phase2291_c2701_c2800_sample_conditioned_coordinate_tournament as predictor  # noqa: E402


PHASE = 2294
CAMPAIGN = "C2941-C3020"
COMPILED = OUT / "material/qwen3_14b_compiled.jsonl"
CANDIDATE = OUT / "behavior/candidate.jsonl"
GENERATION = OUT / "behavior/generation.jsonl"
FIELD = OUT / "raw/qwen3_14b_selected_role_field.float16.npy"
FIELD_INDEX = OUT / "raw/selected_role_field_index.jsonl"
FIELD_PROGRESS = OUT / "raw/selected_role_field_progress.json"
ATLAS = OUT / "atlas/qwen3_14b_frozen_coordinate_errors.float32.npy"
ATLAS_ROWS = OUT / "atlas/qwen3_14b_frozen_coordinate_error_rows.jsonl"
EPS = 1e-8


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    contract.write_rows(path, rows)


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
        position = clean.find(needle)
        if needle and position >= 0:
            hits.append((position, -len(needle), answer))
    return min(hits)[2] if hits else None


def frozen_cells() -> list[dict]:
    return load(FREEZE_OUT / "config/qwen14_replication_freeze.json")["cells"]


def compile_material(tokenizer, cells: list[dict]) -> list[dict]:
    if COMPILED.exists():
        return read_rows(COMPILED)
    families = {row["family"] for row in cells}
    material = [row for row in read_rows(CONTRACT_OUT / "material/natural_bilingual_cases.jsonl")
                if row["family"] in families]
    compiled = contract.compile_rows(tokenizer, material)
    write_rows(COMPILED, compiled)
    return compiled


def run_candidate(model, device, rows: list[dict], batch_size: int = 16) -> list[dict]:
    output = read_rows(CANDIDATE)
    completed = len(output)
    if completed > len(rows):
        raise RuntimeError(("candidate_resume_overflow", completed, len(rows)))
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
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                           use_cache=False, return_dict=True).logits
        for i, row in enumerate(batch):
            end = len(row["prompt_ids"]) - 1
            scores = [float(logits[i, end, candidate[0]]) for candidate in row["candidate_ids"]]
            prediction = int(scores[1] > scores[0])
            output.append({"case_id": row["case_id"], "scores": scores,
                           "prediction": prediction, "gold_position": int(row["gold_position"]),
                           "correct": prediction == int(row["gold_position"])})
        write_rows(CANDIDATE, output)
        print(f"[Qwen3-14B candidate] {start + len(batch)}/{len(rows)}", flush=True)
    return output


def run_generation(model, tokenizer, device, rows: list[dict], batch_size: int = 96) -> list[dict]:
    output = read_rows(GENERATION)
    completed = len(output)
    if completed > len(rows):
        raise RuntimeError(("generation_resume_overflow", completed, len(rows)))
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    for start in range(completed, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
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
        write_rows(GENERATION, output)
        print(f"[Qwen3-14B generation] {start + len(batch)}/{len(rows)}", flush=True)
    return output


def map_checkpoints(cells: list[dict], layer_count: int) -> tuple[dict[int, int], tuple[int, ...]]:
    mapping = {}
    required = set()
    for cell in cells:
        q4 = int(cell["qwen4_checkpoint"])
        q14_value = 0 if q4 == 0 else int(round(q4 * layer_count / 36))
        mapping[q4] = q14_value
        required.add(q14_value)
        if cell["model"] == "current_previous_affine":
            required.add(q14_value - 1)
    return mapping, tuple(sorted(required))


def selected_modules(model, qpoints: tuple[int, ...]) -> dict[int, Any]:
    output = {}
    for q in qpoints:
        output[q] = model.model.embed_tokens if q == 0 else model.model.layers[q - 1]
    return output


def field_index(rows: list[dict], candidates: list[dict], generations: list[dict]) -> list[dict]:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generations}
    return [{"hidden_index": i, "case_id": row["case_id"], "family": row["family"],
             "language": row["language"], "surface": row["surface"], "unit": int(row["unit"]),
             "state": int(row["state"]), "partition": row["partition"],
             "role_positions": row["role_positions"], "prompt_length": len(row["prompt_ids"]),
             "candidate_correct": bool(c[row["case_id"]]["correct"]),
             "generation_correct": bool(g[row["case_id"]]["correct"])}
            for i, row in enumerate(rows)]


def capture_field(model, device, rows: list[dict], index: list[dict],
                  qpoints: tuple[int, ...], batch_size: int = 32) -> dict:
    dimension = int(model.config.hidden_size)
    shape = (len(rows), len(qpoints), len(contract.ROLES), dimension)
    completed = 0
    if FIELD.exists() and FIELD_PROGRESS.exists():
        progress = load(FIELD_PROGRESS)
        if tuple(progress["shape"]) != shape or tuple(progress["qpoints"]) != qpoints:
            raise RuntimeError(("field_resume_mismatch", progress, shape, qpoints))
        completed = int(progress["completed"])
    FIELD.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(FIELD, mode="r+" if FIELD.exists() else "w+",
                                      dtype=np.float16, shape=shape)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in selected_modules(model, qpoints).items():
        def hook(_module, _args, output, q=q):
            captured[q] = output[0] if isinstance(output, tuple) else output
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
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
            if set(captured) != set(qpoints):
                raise RuntimeError(("missing_checkpoints", sorted(captured), qpoints))
            for local_i, row in enumerate(batch):
                for q_i, q in enumerate(qpoints):
                    hidden = captured[q][local_i]
                    for role_i, role in enumerate(contract.ROLES):
                        position = int(row["role_positions"][role][-1])
                        field[start + local_i, q_i, role_i] = hidden[position].float().cpu().numpy()
            field.flush()
            save(FIELD_PROGRESS, {"shape": list(shape), "qpoints": list(qpoints),
                                  "completed": start + len(batch)})
            print(f"[Qwen3-14B selected field] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        del field
    write_rows(FIELD_INDEX, index)
    return {"path": str(FIELD.relative_to(ROOT)), "index": str(FIELD_INDEX.relative_to(ROOT)),
            "shape": list(shape), "qpoints": list(qpoints), "all_coordinates": True}


def build_pairs(index: list[dict]) -> dict[str, list[dict]]:
    cells = {(row["family"], row["language"], row["surface"], row["unit"], row["state"]): row
             for row in index}
    output = defaultdict(list)
    for family, language, surface, unit in sorted({key[:4] for key in cells}):
        left = cells[(family, language, surface, unit, 0)]
        right = cells[(family, language, surface, unit, 1)]
        output[family].append({"family": family, "language": language, "surface": surface,
                               "unit": unit, "partition": left["partition"],
                               "left": left["hidden_index"], "right": right["hidden_index"]})
    return dict(output)


class FrozenTournament(predictor.Tournament):
    def __init__(self, field: np.ndarray, pairs: dict[str, list[dict]], qpoints: tuple[int, ...]):
        super().__init__(field, pairs)
        self.qpoints = qpoints

    def arrays_at(self, rows: list[dict], q: int, role: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        slot = self.qpoints.index(q)
        left = np.asarray(self.field[[row["left"] for row in rows], slot, role], np.float32)
        right = np.asarray(self.field[[row["right"] for row in rows], slot, role], np.float32)
        previous = (np.asarray(self.field[[row["left"] for row in rows], self.qpoints.index(q - 1), role], np.float32)
                    if q > 0 and q - 1 in self.qpoints else np.zeros_like(left))
        return left, previous, right - left

    def fit(self, family: str, route: str, q: int, role: int, name: str,
            mode: str = "normal") -> tuple[np.ndarray, ...]:
        key = (family, route, q, role, name, mode)
        if key not in self.cache:
            rows = self.subset(family, "discovery", route, True)
            x, previous, y = self.arrays_at(rows, q, role)
            if mode == "shuffled":
                y = y[np.arange(len(y))[::-1]]
            self.cache[key] = predictor.fit_model(name, x, previous, y)
        return self.cache[key]

    def fit_shared(self, family: str, route: str, q: int, role: int,
                   name: str) -> tuple[np.ndarray, ...]:
        key = (family, route, q, role, name, "shared")
        if key not in self.cache:
            xs, ps, ys = [], [], []
            for other in self.families:
                if other == family:
                    continue
                x, previous, y = self.arrays_at(self.subset(other, "discovery", route, True), q, role)
                xs.append(x); ps.append(previous); ys.append(y)
            self.cache[key] = predictor.fit_model(name, np.concatenate(xs), np.concatenate(ps), np.concatenate(ys))
        return self.cache[key]

    def evaluate(self, family: str, route: str, partition: str, q: int, role: int,
                 name: str) -> tuple[dict, dict[str, np.ndarray]]:
        rows = self.subset(family, partition, route, False)
        x, previous, truth = self.arrays_at(rows, q, role)
        candidate = predictor.predict(name, self.fit(family, route, q, role, name), x, previous)
        wrong = self.families[(self.families.index(family) + 1) % len(self.families)]
        source_rows = self.subset(family, "discovery", route, True)
        sx, sp, train_y = self.arrays_at(source_rows, q, role)
        target_rows = self.subset(family, "discovery", route, False)
        tx, tp, ty = self.arrays_at(target_rows, q, role)
        shared = predictor.predict(name, self.fit_shared(family, route, q, role, name), x, previous)
        shuffled = predictor.predict(name, self.fit(family, route, q, role, name, "shuffled"), x, previous)
        wrong_prediction = predictor.predict(name, self.fit(wrong, route, q, role, name), x, previous)
        previous_model = predictor.fit_current(sp, train_y)
        current_model = predictor.fit_current(sx, train_y)
        predictions = {
            "candidate": candidate,
            "target_mean": np.broadcast_to(ty.mean(0), truth.shape),
            "shared_model": shared,
            "shuffled_labels": shuffled,
            "wrong_family": wrong_prediction,
            "zero": np.zeros_like(truth),
            "previous_checkpoint": previous * previous_model[0] + previous_model[1],
            "current_only": x * current_model[0] + current_model[1],
            "target_oracle": predictor.predict(name, predictor.fit_model(name, tx, tp, ty), x, previous),
        }
        errors = {key: np.mean(np.abs(value - truth), axis=0).astype(np.float32)
                  for key, value in predictions.items()}
        candidate_error = errors["candidate"]
        controls = ["target_mean", "shared_model", "shuffled_labels", "wrong_family", "zero",
                    "previous_checkpoint"]
        if name == "current_previous_affine":
            controls.append("current_only")
        gains = {key: float((errors[key].mean() - candidate_error.mean()) /
                            max(float(errors[key].mean()), EPS)) for key in controls}
        wins = {key: float(np.mean(candidate_error < errors[key])) for key in controls}
        oracle_ratio = float(candidate_error.mean() / max(float(errors["target_oracle"].mean()), EPS))
        gates = contract.PREDICTION_GATES
        passes = (len(rows) >= gates["minimum_pairs"] and
                  gains["target_mean"] >= gates["gain_over_target_mean"] and
                  all(gains[key] >= gates["gain_over_shared_model"] for key in controls if key != "target_mean") and
                  wins["target_mean"] >= gates["coordinate_win_fraction"] and
                  wins["shared_model"] >= gates["coordinate_win_fraction"] and
                  oracle_ratio <= gates["maximum_oracle_ratio"])
        record = {"family": family, "route": route, "partition": partition, "checkpoint": q,
                  "role_index": role, "role": contract.ROLES[role], "model": name,
                  "pairs": len(rows), "candidate_mae": float(candidate_error.mean()),
                  "gains": gains, "wins": wins, "oracle_ratio": oracle_ratio,
                  "passes": bool(passes)}
        return record, errors


def run_frozen_tournament(field: np.ndarray, index: list[dict], cells: list[dict],
                          qmap: dict[int, int], qpoints: tuple[int, ...],
                          qualified: list[str]) -> dict:
    tournament = FrozenTournament(field, build_pairs(index), qpoints)
    decisions = []
    atlas_values = []
    atlas_rows = []
    for cell in cells:
        family = cell["family"]
        base = {**cell, "qwen14_checkpoint": qmap[int(cell["qwen4_checkpoint"])]}
        if family not in qualified:
            decisions.append({**base, "confirmation": None, "fresh_confirmation": None,
                              "fresh_authorized": False, "lockbox_revealed": False,
                              "lockbox": None, "lockbox_pass": False,
                              "reason": "qwen14_behavior_unqualified"})
            continue
        if len(tournament.families) < 2:
            decisions.append({**base, "confirmation": None, "fresh_confirmation": None,
                              "fresh_authorized": False, "lockbox_revealed": False,
                              "lockbox": None, "lockbox_pass": False,
                              "reason": "insufficient_behavior_qualified_families_for_controls"})
            continue
        q = base["qwen14_checkpoint"]
        confirmation, confirmation_errors = tournament.evaluate(
            family, cell["route"], "confirmation", q, int(cell["role_index"]), cell["model"])
        fresh = fresh_errors = lock = lock_errors = None
        if confirmation["passes"]:
            fresh, fresh_errors = tournament.evaluate(
                family, cell["route"], "fresh_confirmation", q,
                int(cell["role_index"]), cell["model"])
        authorized = bool(fresh and fresh["passes"])
        if authorized:
            lock, lock_errors = tournament.evaluate(
                family, cell["route"], "fresh_lockbox", q,
                int(cell["role_index"]), cell["model"])
        decision = {**base, "confirmation": confirmation, "fresh_confirmation": fresh,
                    "fresh_authorized": authorized, "lockbox_revealed": authorized,
                    "lockbox": lock, "lockbox_pass": bool(lock and lock["passes"])}
        decisions.append(decision)
        final_errors = lock_errors or fresh_errors or confirmation_errors
        final_partition = "fresh_lockbox" if lock_errors else (
            "fresh_confirmation" if fresh_errors else "confirmation")
        for error, values in final_errors.items():
            atlas_rows.append({"row": len(atlas_values), "family": family, "route": cell["route"],
                               "partition": final_partition, "qwen4_checkpoint": cell["qwen4_checkpoint"],
                               "qwen14_checkpoint": q, "role": cell["role"], "model": cell["model"],
                               "error": error})
            atlas_values.append(values)
        print(f"[Qwen3-14B frozen cell] {family}/{cell['route']} q={q} "
              f"confirm={confirmation['passes']} fresh={authorized} lock={decision['lockbox_pass']}", flush=True)
    ATLAS.parent.mkdir(parents=True, exist_ok=True)
    atlas = np.stack(atlas_values).astype(np.float32) if atlas_values else np.empty((0, 5120), np.float32)
    np.save(ATLAS, atlas)
    write_rows(ATLAS_ROWS, atlas_rows)
    return {"decisions": decisions,
            "lockbox_passed": [f"{row['family']}|{row['route']}" for row in decisions if row["lockbox_pass"]],
            "atlas": {"path": str(ATLAS.relative_to(ROOT)), "rows": str(ATLAS_ROWS.relative_to(ROOT)),
                      "shape": list(atlas.shape)}}


def append_memo(result: dict) -> None:
    current = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in current:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact_behavior = {family: {"candidate": value["candidate_accuracy"],
                                 "generation": value["generation_accuracy"],
                                 "qualified": value["dual_qualified"]}
                        for family, value in result["behavior"]["families"].items()}
    compact_cells = [{key: row.get(key) for key in
                      ("family", "route", "qwen4_checkpoint", "qwen14_checkpoint", "role", "model",
                       "fresh_authorized", "lockbox_revealed", "lockbox_pass", "reason")}
                     for row in result["structure"]["decisions"]]
    text = rf"""

## Phase {PHASE}: Qwen3-14B冻结坐标函数前瞻复验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本期只复验 Phase2293 在任何14B结果出现前冻结的五个4B功能单元：态度事件跨语言 `primary@q1`、位置绑定跨语言 `secondary@q0`、所有关系跨语言 `context@q0`、关系从句跨表面 `secondary@q5`、施事受事跨表面 `secondary@q0`。材料为32个完全分区单元、中英、narrative/dialogue、真假两态。先对全部 `{result['material_rows']}` 行分别执行候选选择和自由生成；只有总体及每个语言、表面、分区的两项准确率均不低于 `{contract.BEHAVIOR_GATE}` 的构式才进入内部场。14B单独以FP16 CUDA加磁盘卸载运行；只采集预冻结映射需要的检查点，但每个检查点保存六角色全部5120个物理激活坐标。未读取Attention、MLP、权重或梯度，也未使用PCA、Top-K、余弦筛选或平均差分搬运。

**公式。** 4B到14B只映射功能深度，不比较坐标编号：

$$
q_{{14}}=\begin{{cases}}0,&q_4=0,\\ \operatorname{{round}}(q_4L_{{14}}/36),&q_4>0.\end{{cases}}
$$

每个14B物理坐标独立拟合冻结类型的样本条件函数：

$$
\widehat R_{{i,q,r,j}}=a_{{q,r,j}}H^0_{{i,q,r,j}}+b_{{q,r,j}},
$$

或：

$$
\widehat R_{{i,q,r,j}}=a_{{q,r,j}}H^0_{{i,q,r,j}}+c_{{q,r,j}}H^0_{{i,q-1,r,j}}+b_{{q,r,j}}.
$$

**结果汇总。** 行为 `{json.dumps(compact_behavior, ensure_ascii=False)}`；冻结单元 `{json.dumps(compact_cells, ensure_ascii=False)}`；完整逐坐标误差护照 `{json.dumps(result['structure']['atlas'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。阳性只表示相同的功能拓扑、相对深度和函数类型能在14B自己的物理坐标系中前瞻复验；它不表示两个模型共享坐标编号、共享单神经元、共享参数电路，也不构成因果齿轮。阴性则只淘汰该冻结跨尺度函数，不否定构式信息存在。主要硬伤是同模型族训练数据并不独立、人工材料尚无人类盲评、角色末token只是功能近似、逐坐标仿射忽略坐标间联动，且磁盘卸载增加工程复杂度。相关脚本 `tests/glm5/phase2294_c2941_c3020_qwen14_frozen_coordinate_replication.py`；结果 `tests/glm5/result/phase2294_c2941_c3020_qwen14_frozen_coordinate_replication`。

**下一步授权。** 只发布可观察的行为、相对深度、功能单元和exact-coordinate误差图；客户端资产写明观察/预测/因果层级。发布后删除不直接显示的14B原场，保留编译材料、逐批行为、配置、误差护照和哈希。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = load(final_path)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = load(FREEZE_OUT / "analysis/final.json")
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2293 freeze is not authorized")
    cells = frozen_cells()
    model = tokenizer = None
    try:
        # Keep numerical execution identical while retaining two-thirds of the prior
        # resident layers so larger, more I/O-efficient batches fit on a 16 GB GPU.
        def reduced_resident_device_map() -> dict:
            value = {"model.embed_tokens": 0}
            value.update({f"model.layers.{i}": 0 if i < 12 else "disk" for i in range(40)})
            value.update({"model.norm": "disk", "model.rotary_emb": "cpu", "lm_head": "disk"})
            return value

        q14.device_map = reduced_resident_device_map
        model, tokenizer, device = q14.load_model()
        rows = compile_material(tokenizer, cells)
        layers = len(model.model.layers)
        qmap, qpoints = map_checkpoints(cells, layers)
        preregistration = {"phase": PHASE, "campaign": CAMPAIGN, "frozen_before_model": True,
                           "source_phase": 2293, "cells": cells, "qwen14_layers": layers,
                           "checkpoint_mapping": qmap, "qpoints": list(qpoints),
                           "behavior_gate": contract.BEHAVIOR_GATE,
                           "prediction_gates": contract.PREDICTION_GATES,
                           "reveal_order": ["confirmation", "fresh_confirmation", "fresh_lockbox"],
                           "forbidden": ["cross-model-coordinate-identity", "PCA", "Top-K", "cosine-selection",
                                         "mean-delta-transfer", "attention", "MLP", "weights", "gradients"]}
        save(OUT / "protocol/preregistration.json", preregistration)
        candidates = run_candidate(model, device, rows)
        generations = run_generation(model, tokenizer, device, rows)
        ledger = behavior.behavior_ledger(rows, candidates, generations)
        save(OUT / "behavior/ledger.json", ledger)
        qualified = [family for family in ledger["qualified_families"]
                     if family in {cell["family"] for cell in cells}]
        observed = [row for row in rows if row["family"] in set(qualified)]
        index = field_index(observed, candidates, generations)
        field_info = capture_field(model, device, observed, index, qpoints) if observed else {
            "ran": False, "reason": "no_behavior_qualified_family"}
        if observed:
            field = np.load(FIELD, mmap_mode="r")
            structure = run_frozen_tournament(field, index, cells, qmap, qpoints, qualified)
            mmap = getattr(field, "_mmap", None)
            if mmap is not None:
                mmap.close()
        else:
            structure = {"decisions": [{**cell, "reason": "qwen14_behavior_unqualified",
                                         "lockbox_pass": False} for cell in cells],
                         "lockbox_passed": [], "atlas": {"shape": [0, 5120]}}
        model_info = {"name": "Qwen3-14B", "precision": "float16",
                      "placement": "cuda_disk_offload", "hidden_size": int(model.config.hidden_size),
                      "layers": layers}
    finally:
        if model is not None:
            del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {
        "parent_freeze_authorized": parent["all_checks_passed"],
        "five_frozen_cells": len(cells) == 5,
        "material_complete": len(rows) == 5 * 2 * 2 * 32 * 2,
        "behavior_complete_before_field": len(candidates) == len(generations) == len(rows),
        "field_only_behavior_qualified": (not observed) or len(observed) == field_info["shape"][0],
        "all_selected_coordinates": (not observed) or field_info["shape"][-1] == 5120,
        "ordered_reveal": all(not row.get("lockbox_revealed") or row.get("fresh_authorized")
                              for row in structure["decisions"]),
        "atlas_full_coordinates": structure["atlas"]["shape"][-1] == 5120,
    }
    hashes = {"preregistration": file_hash(OUT / "protocol/preregistration.json"),
              "compiled": file_hash(COMPILED), "candidate": file_hash(CANDIDATE),
              "generation": file_hash(GENERATION),
              "field": file_hash(FIELD) if FIELD.exists() else None,
              "atlas": file_hash(ATLAS) if ATLAS.exists() else None}
    result = {"phase": PHASE, "campaign": CAMPAIGN,
              "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
              "material_rows": len(rows), "model": model_info, "behavior": ledger,
              "field": field_info, "checkpoint_mapping": qmap, "qpoints": list(qpoints),
              "structure": structure, "hashes": hashes, "checks": checks,
              "all_checks_passed": all(checks.values()),
              "strict_conclusion": (f"{len(structure['lockbox_passed'])}/5 frozen Qwen3-4B functional cells "
                                    "passed the complete Qwen3-14B model-local prospective controls; "
                                    "this is cross-scale predictive topology evidence, not shared-coordinate or causal closure."),
              "next_authorization": "Publish exact-coordinate atlases, then remove non-published raw fields."}
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
