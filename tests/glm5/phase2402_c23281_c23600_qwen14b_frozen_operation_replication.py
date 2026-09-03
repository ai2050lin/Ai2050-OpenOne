#!/usr/bin/env python3
"""Frozen Qwen3-14B replication of event-aligned condition-update and output-link findings."""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2397 = RESULT / "phase2397_c21681_c22000_operation_behavior_token_calibration/qwen14b"
OUT = RESULT / "phase2402_c23281_c23600_qwen14b_frozen_operation_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2402
CAMPAIGN = "C23281-C23600"

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2339_c7401_c7600_crossmodel_fixed_ab_replication as replication_loader  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def index_row(row: dict) -> dict:
    keys = ("case_id", "task", "family", "unit", "language", "surface", "direction", "partition",
            "target_candidate_slot", "query_role", "steps", "answer", "foil", "prompt_token_count", "target_ids", "foil_ids")
    result = {key: row[key] for key in keys if key in row}
    result["event_names"] = [event["event"] for event in row["event_tokens"]]
    result["event_token_indices"] = [event["token_index"] for event in row["event_tokens"]]
    return result


def pad_right(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device); mask[index, :len(sequence)] = 1
    positions = mask.long().cumsum(-1) - 1; positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def collect_field(model, rows: list[dict], task: str, batch_size: int = 2) -> dict:
    modules = field_utils.modules(model); dimension = int(model.get_input_embeddings().weight.shape[1]); event_count = len(rows[0]["event_tokens"])
    path = OUT / f"raw/{task}_event_field.float16.npy"; progress = OUT / f"raw/{task}_progress.json"
    shape = (len(rows), len(modules), event_count, dimension)
    if path.exists():
        values = np.lib.format.open_memmap(path, mode="r+"); completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"])
    else:
        path.parent.mkdir(parents=True, exist_ok=True); values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape); completed = 0
    captures: dict[int, torch.Tensor] = {}; handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, output, qpoint=qpoint): captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]; ids, mask, positions = pad_right([row["prompt_ids"] for row in batch], device, pad)
                captures.clear(); model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for qpoint in range(len(modules)):
                    tensor = captures[qpoint].float().cpu()
                    selected = torch.stack([tensor[local, torch.tensor([event["token_index"] for event in row["event_tokens"]], dtype=torch.long)] for local, row in enumerate(batch)])
                    values[start:start + len(batch), qpoint] = selected.numpy().astype(np.float16)
                values.flush(); save(progress, {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 128 == 0 or start + len(batch) == len(rows): print(f"[phase2402 {task} field] {start + len(batch)}/{len(rows)}", flush=True)
                del ids, mask, positions
    finally:
        for handle in handles: handle.remove()
        values.flush(); close(values)
    return {"path": str(path), "shape": list(shape), "bytes": path.stat().st_size,
            "event_names": [event["event"] for event in rows[0]["event_tokens"]]}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1); b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def groups(y: np.ndarray, rows: list[dict], train: np.ndarray, keys: tuple[str, ...]) -> tuple[np.ndarray, dict[tuple, np.ndarray]]:
    constant = y[train].mean(0); mapping: dict[tuple, list[int]] = defaultdict(list)
    for local, source in enumerate(train.tolist()): mapping[tuple(rows[source].get(key) for key in keys)].append(local)
    return constant, {key: y[train][local].mean(0) for key, local in mapping.items()}


def family_passport(y: np.ndarray, rows: list[dict], indices: np.ndarray, families: list[str]) -> np.ndarray:
    center = y[indices].mean(0)
    return np.stack([y[[i for i in indices if rows[i]["family"] == family]].mean(0) - center for family in families])


def factor_passport(y: np.ndarray, rows: list[dict], indices: np.ndarray, families: list[str], factor: str, value: Any) -> np.ndarray:
    selected = np.asarray([i for i in indices if rows[i].get(factor) == value]); center = y[selected].mean(0)
    return np.stack([y[[i for i in selected if rows[i]["family"] == family]].mean(0) - center for family in families])


def metric_update(acc: dict, pred: np.ndarray, truth: np.ndarray, base: np.ndarray) -> None:
    acc["sse"] += float(np.sum((truth - pred) ** 2, dtype=np.float64)); acc["base"] += float(np.sum((truth - base) ** 2, dtype=np.float64))
    acc["sign"] += int(np.sum(np.signbit(pred) == np.signbit(truth))); acc["count"] += truth.size


def metric_finish(acc: dict) -> dict:
    return {"coordinates": int(acc["count"]), "gain_vs_constant": 1 - acc["sse"] / max(acc["base"], 1e-30),
            "mse": acc["sse"] / acc["count"], "sign_agreement": acc["sign"] / acc["count"]}


def output_delta(model, states: np.ndarray, target: np.ndarray, foil: np.ndarray) -> np.ndarray:
    device = model.get_output_embeddings().weight.device; weights = model.get_output_embeddings().weight.detach()
    tensor = torch.tensor(states, dtype=torch.float32, device=device); shape = tensor.shape
    normed = model.model.norm(tensor.reshape(-1, shape[-1])).reshape(shape)
    t = torch.tensor(target, dtype=torch.long, device=device); f = torch.tensor(foil, dtype=torch.long, device=device)
    delta = weights[t].float() - weights[f].float()
    return torch.sum(normed.float() * delta, dim=-1).detach().cpu().numpy()


def analyze_task(task: str, model, field_path: Path, rows: list[dict]) -> dict:
    field = np.load(field_path, mmap_mode="r"); q_updates = field.shape[1] - 2; event_count, dimension = field.shape[2], field.shape[3]
    if task == "selection":
        train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"]); lock = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_unit_lockbox"])
        keys = ("family", "language", "surface", "direction", "query_role", "target_candidate_slot")
    else:
        train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"]); lock = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_composition_lockbox"])
        keys = ("family", "language", "surface", "direction", "target_candidate_slot")
    families = sorted({row["family"] for row in rows}); family_shift = {family: families[(i + 1) % len(families)] for i, family in enumerate(families)}
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    passport = np.lib.format.open_memmap(OUT / f"derived/{task}_family_passport_lockbox.float32.npy", mode="w+", dtype=np.float32,
                                         shape=(q_updates, event_count, len(families), dimension))
    gain_array = np.lib.format.open_memmap(OUT / f"derived/{task}_condition_gain_lockbox.float32.npy", mode="w+", dtype=np.float32,
                                           shape=(q_updates, event_count, dimension))
    acc = {name: defaultdict(float) for name in ("condition", "wrong_family", "coordinate_permuted")}
    partition_cos = {family: [] for family in families}; surface_cos = {family: [] for family in families}; language_cos = {family: [] for family in families}
    quartiles = [0.0] * 4
    answer_event = event_count - 1; output_acc = defaultdict(float); row_condition = np.zeros(len(lock)); row_constant = np.zeros(len(lock))
    target = np.asarray([rows[i]["target_ids"][0] for i in lock], dtype=np.int64); foil = np.asarray([rows[i]["foil_ids"][0] for i in lock], dtype=np.int64)
    for qpoint in range(q_updates):
        for event in range(event_count):
            x = np.asarray(field[:, qpoint, event], dtype=np.float32); y = np.asarray(field[:, qpoint + 1, event], dtype=np.float32) - x
            constant, condition = groups(y, rows, train, keys); _, family_mean = groups(y, rows, train, ("family",))
            pred = np.stack([condition.get(tuple(rows[i].get(key) for key in keys), constant) for i in lock]); base = np.broadcast_to(constant, pred.shape); truth = y[lock]
            wrong = np.stack([family_mean[(family_shift[rows[i]["family"]],)] for i in lock]); permuted = np.roll(pred, 1, axis=1)
            metric_update(acc["condition"], pred, truth, base); metric_update(acc["wrong_family"], wrong, truth, base); metric_update(acc["coordinate_permuted"], permuted, truth, base)
            gain = np.sum((truth - base) ** 2 - (truth - pred) ** 2, axis=0, dtype=np.float64).astype(np.float32); gain_array[qpoint, event] = gain
            rms = np.sqrt(np.mean(y[train] * y[train], axis=0)); order = np.argsort(rms, kind="stable")
            for quartile, indices in enumerate(np.array_split(order, 4)): quartiles[quartile] += float(gain[indices].sum(dtype=np.float64))
            p_train = family_passport(y, rows, train, families); p_lock = family_passport(y, rows, lock, families); passport[qpoint, event] = p_lock
            for family_index, family in enumerate(families): partition_cos[family].append(cosine(p_train[family_index], p_lock[family_index]))
            for factor, receiver in (("surface", surface_cos), ("language", language_cos)):
                values = sorted({rows[i].get(factor) for i in lock}, key=str); pa = factor_passport(y, rows, lock, families, factor, values[0]); pb = factor_passport(y, rows, lock, families, factor, values[1])
                for family_index, family in enumerate(families): receiver[family].append(cosine(pa[family_index], pb[family_index]))
            if event == answer_event:
                actual_next = output_delta(model, x[lock] + truth, target, foil); predicted = output_delta(model, x[lock] + pred, target, foil); predicted_base = output_delta(model, x[lock] + base, target, foil)
                actual_current = output_delta(model, x[lock], target, foil); actual_delta = actual_next - actual_current; cond_delta = predicted - actual_current; base_delta = predicted_base - actual_current
                output_acc["sse"] += float(np.sum((actual_delta - cond_delta) ** 2)); output_acc["base"] += float(np.sum((actual_delta - base_delta) ** 2)); output_acc["count"] += len(lock)
            row_condition += np.sum((truth - pred) ** 2, axis=1, dtype=np.float64); row_constant += np.sum((truth - base) ** 2, axis=1, dtype=np.float64)
            del x, y, truth, pred, base, wrong, permuted
        passport.flush(); gain_array.flush(); print(f"[phase2402 {task} analysis] update {qpoint + 1}/{q_updates}", flush=True)
    teacher = {row["case_id"]: row for row in read_rows(P2397 / "behavior/teacher_scores.jsonl") if row["task"] == task}; autonomous = {row["case_id"]: row for row in read_rows(P2397 / "behavior/autonomous_lockbox.jsonl") if row["task"] == task}
    row_gain = 1 - row_condition / np.maximum(row_constant, 1e-30); final_margin = np.asarray([teacher[rows[i]["case_id"]]["first_divergence_logit_margin"] for i in lock]); present = np.asarray([autonomous[rows[i]["case_id"]]["target_present"] for i in lock])
    result = {"task": task, "field_shape": list(field.shape), "train_rows": len(train), "lockbox_rows": len(lock),
              "coordinate_prediction": {name: metric_finish(value) for name, value in acc.items()},
              "passport_new_unit_cosine": {family: float(np.mean(values)) for family, values in partition_cos.items()},
              "cross_surface_cosine": {family: float(np.mean(values)) for family, values in surface_cos.items()},
              "cross_language_cosine": {family: float(np.mean(values)) for family, values in language_cos.items()},
              "low_to_high_rms_quartile_signed_gain": quartiles,
              "answer_boundary_output_compilation": {"values": int(output_acc["count"]), "gain_vs_constant": 1 - output_acc["sse"] / max(output_acc["base"], 1e-30), "mse": output_acc["sse"] / output_acc["count"]},
              "behavior_bridge": {"coordinate_gain_to_final_margin_correlation": float(np.corrcoef(row_gain, final_margin)[0, 1]),
                                  "gain_target_present": float(np.mean(row_gain[present])) if present.any() else None,
                                  "gain_target_absent": float(np.mean(row_gain[~present])) if (~present).any() else None},
              "arrays": {"lockbox_passport": str(OUT / f"derived/{task}_family_passport_lockbox.float32.npy"), "coordinate_gain": str(OUT / f"derived/{task}_condition_gain_lockbox.float32.npy")}}
    passport.flush(); gain_array.flush(); close(passport); close(gain_array); close(field); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen14B冻结条件更新—护照—输出桥全坐标复现（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不根据Qwen14B结果重选算子、层或事件，直接复验Qwen4B冻结的完整条件坐标均值：选择任务用unit0–3 discovery拟合并测试unit6–7 fresh；组合任务只用一步discovery拟合，直接测试两步fresh。Qwen3-14B以NF4/BF16单模型驻留CUDA，全部3072条prompt采集embedding、40个block、final norm的每个事件×全部5120物理坐标；对40个残差更新比较条件算子、错族和坐标循环置换。同步复测族护照跨unit/表达/语言、低RMS四分位收益、答案边界局部更新经真实RMSNorm/LM head后的margin变化，以及逐行坐标收益与Phase2397行为的连接。

$$U_{{q,e,j}}=H_{{q+1,e,j}}-H_{{q,e,j}},\qquad
G=1-\frac{{\mathrm{{SSE}}(U,\widehat U_{{condition}})}}{{\mathrm{{SSE}}(U,\bar U_{{train}})}}.$$

**结果汇总。** 选择复现 `{json.dumps(result['selection'], ensure_ascii=False)}`；两步组合复现 `{json.dumps(result['composition'], ensure_ascii=False)}`；跨规模裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2402_c23281_c23600_qwen14b_frozen_operation_replication.py`；Qwen14B约11.5GiB全事件全坐标场、护照、逐坐标收益和final位于 `tests/glm5/result/phase2402_c23281_c23600_qwen14b_frozen_operation_replication`。

**理论进展。** 该Phase裁决Qwen4B的条件更新规律是否只是小模型/单规模纹理。复现对象不是绝对坐标编号对齐，而是同一模型内部“条件均值相对常量的全场收益、错族/置换特异性、跨表达护照以及输出margin桥”的功能关系；5120坐标全部保留。

**问题硬伤与结论。** NF4与Qwen4B BF16的绝对激活幅值不可比较；相同语言模板在同架构Qwen3两规模间并非跨架构普遍性。条件单元仍混合family、语言、表面、方向、查询/答案槽，跨表面/语言护照弱时必须降格为模板条件更新。输出编译阳性若仍不连接最终行为，不能升级为语言机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2397 / "index/operation_rows.jsonl"); selection_rows = [row for row in rows if row["task"] == "selection"]; composition_rows = [row for row in rows if row["task"] == "composition"]
    write_rows(OUT / "index/selection_rows.jsonl", [index_row(row) for row in selection_rows]); write_rows(OUT / "index/composition_rows.jsonl", [index_row(row) for row in composition_rows])
    model, tokenizer, _device = replication_loader.load_model("qwen14b")
    tokenizer.padding_side = "left"
    label = replication_loader.MODEL_SPECS["qwen14b"]["label"]
    try:
        collection = {"selection": collect_field(model, selection_rows, "selection"), "composition": collect_field(model, composition_rows, "composition")}
        selection = analyze_task("selection", model, Path(collection["selection"]["path"]), selection_rows)
        composition = analyze_task("composition", model, Path(collection["composition"]["path"]), composition_rows)
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    adjudication = {"selection_condition_gain": selection["coordinate_prediction"]["condition"]["gain_vs_constant"],
                    "two_step_condition_gain": composition["coordinate_prediction"]["condition"]["gain_vs_constant"],
                    "selection_answer_compilation_gain": selection["answer_boundary_output_compilation"]["gain_vs_constant"],
                    "two_step_answer_compilation_gain": composition["answer_boundary_output_compilation"]["gain_vs_constant"],
                    "qwen_two_scale_functional_replication": selection["coordinate_prediction"]["condition"]["gain_vs_constant"] > 0 and composition["coordinate_prediction"]["condition"]["gain_vs_constant"] > 0,
                    "mechanism_closed": False}
    checks = {"selection_shape": collection["selection"]["shape"] == [2048, 42, 8, 5120], "composition_shape": collection["composition"]["shape"] == [1024, 42, 12, 5120],
              "negative_controls": selection["coordinate_prediction"]["wrong_family"]["gain_vs_constant"] < selection["coordinate_prediction"]["condition"]["gain_vs_constant"] and selection["coordinate_prediction"]["coordinate_permuted"]["gain_vs_constant"] < selection["coordinate_prediction"]["condition"]["gain_vs_constant"],
              "finite": all(math.isfinite(value) for value in adjudication.values() if isinstance(value, float)), "claim_boundary": not adjudication["mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": label, "collection": collection, "selection": selection, "composition": composition,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
