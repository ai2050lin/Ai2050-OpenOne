#!/usr/bin/env python3
"""Connect event-aligned local updates to first-divergence next-token logits and behavior."""
from __future__ import annotations

import gc
import json
import math
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
P2397 = RESULT / "phase2397_c21681_c22000_operation_behavior_token_calibration/qwen4b"
P2398 = RESULT / "phase2398_c22001_c22320_qwen4b_event_fullfield"
OUT = RESULT / "phase2401_c22961_c23280_composition_output_compilation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2401
CAMPAIGN = "C22961-C23280"
Q_UPDATES = 36

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def align_rows(task: str) -> list[dict]:
    full = {row["case_id"]: row for row in read_rows(P2397 / "index/operation_rows.jsonl") if row["task"] == task}
    index = read_rows(P2398 / f"index/{task}_rows.jsonl")
    rows = []
    for row in index:
        source = full[row["case_id"]]
        rows.append(row | {"prompt_ids": source["prompt_ids"], "target_ids": source["target_ids"], "foil_ids": source["foil_ids"]})
    return rows


def first_tokens(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    targets, foils = [], []
    for row in rows:
        divergence = 0
        while divergence < min(len(row["target_ids"]), len(row["foil_ids"])) and row["target_ids"][divergence] == row["foil_ids"][divergence]:
            divergence += 1
        if divergence != 0:
            raise RuntimeError(("answer_boundary_not_first_divergence", row["case_id"], divergence))
        targets.append(row["target_ids"][0]); foils.append(row["foil_ids"][0])
    return np.asarray(targets, dtype=np.int64), np.asarray(foils, dtype=np.int64)


def normalized_states(model, states: torch.Tensor, already_final: bool = False) -> torch.Tensor:
    if already_final: return states
    shape = states.shape
    return model.model.norm(states.reshape(-1, shape[-1])).reshape(shape)


def collect_exact_final(task: str, model, rows: list[dict], target: np.ndarray, foil: np.ndarray, batch_size: int = 16) -> dict:
    contribution_path = OUT / f"derived/{task}_answer_final_coordinate_contribution.float32.npy"
    direct_path = OUT / f"derived/{task}_answer_direct_logit_margin.float32.npy"
    progress_path = OUT / f"derived/{task}_exact_final_progress.json"
    dimension = int(model.get_output_embeddings().weight.shape[1])
    if contribution_path.exists() and direct_path.exists() and progress_path.exists():
        contribution = np.lib.format.open_memmap(contribution_path, mode="r+")
        direct = np.lib.format.open_memmap(direct_path, mode="r+")
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
    else:
        contribution = np.lib.format.open_memmap(contribution_path, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
        direct = np.lib.format.open_memmap(direct_path, mode="w+", dtype=np.float32, shape=(len(rows),))
        completed = 0
    device = model.get_output_embeddings().weight.device
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    capture: dict[str, torch.Tensor] = {}
    handle = model.model.norm.register_forward_hook(lambda _m, _i, output: capture.__setitem__("norm", output.detach()))
    weights = model.get_output_embeddings().weight.detach()
    bias = getattr(model.get_output_embeddings(), "bias", None)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                stop = min(start + batch_size, len(rows)); batch = rows[start:stop]
                width = max(len(row["prompt_ids"]) for row in batch)
                ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                for local, row in enumerate(batch):
                    sequence = row["prompt_ids"]
                    ids[local, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
                    mask[local, :len(sequence)] = 1
                positions = mask.long().cumsum(-1) - 1; positions.masked_fill_(mask == 0, 0)
                capture.clear(); output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                t = torch.tensor(target[start:stop], dtype=torch.long, device=device); f = torch.tensor(foil[start:stop], dtype=torch.long, device=device)
                delta_w = weights[t].float() - weights[f].float()
                hidden = torch.stack([capture["norm"][local, len(row["prompt_ids"]) - 1].float() for local, row in enumerate(batch)])
                parts = hidden * delta_w
                bias_delta = torch.zeros(len(batch), device=device) if bias is None else bias[t].float() - bias[f].float()
                logits = torch.stack([output.logits[local, len(row["prompt_ids"]) - 1].float() for local, row in enumerate(batch)])
                direct_margin = logits.gather(1, t[:, None]).squeeze(1) - logits.gather(1, f[:, None]).squeeze(1)
                contribution[start:stop] = parts.cpu().numpy(); direct[start:stop] = direct_margin.cpu().numpy()
                contribution.flush(); direct.flush(); save(progress_path, {"completed": stop, "shape": [len(rows), dimension]})
                if stop % 256 == 0 or stop == len(rows): print(f"[phase2401 {task} exact-final] {stop}/{len(rows)}", flush=True)
                del ids, mask, positions, output, hidden, parts, logits, direct_margin, delta_w
    finally:
        handle.remove()
    coordinate_sum = np.asarray(contribution, dtype=np.float64).sum(1)
    direct_values = np.asarray(direct, dtype=np.float64)
    teacher = {row["case_id"]: row for row in read_rows(P2397 / "behavior/teacher_scores.jsonl") if row["task"] == task}
    expected = np.asarray([teacher[row["case_id"]]["first_divergence_logit_margin"] for row in rows], dtype=np.float64)
    bias_values = np.zeros(len(rows), dtype=np.float64) if bias is None else np.asarray((bias[torch.tensor(target, device=device)] - bias[torch.tensor(foil, device=device)]).float().cpu(), dtype=np.float64)
    result = {"coordinate_contribution_path": str(contribution_path), "shape": list(contribution.shape),
              "direct_margin_path": str(direct_path),
              "coordinate_sum_to_direct_mean_abs_error": float(np.mean(np.abs(coordinate_sum + bias_values - direct_values))),
              "direct_to_teacher_mean_abs_error": float(np.mean(np.abs(direct_values - expected))),
              "direct_to_teacher_max_abs_error": float(np.max(np.abs(direct_values - expected))),
              "target_over_foil": float(np.mean(direct_values > 0)), "teacher_target_over_foil": float(np.mean(expected > 0))}
    contribution.flush(); direct.flush(); close(contribution); close(direct)
    return result


def collect_logit_fields(task: str, model, field_path: Path, rows: list[dict], batch_size: int = 8) -> dict:
    field = np.load(field_path, mmap_mode="r")
    target, foil = first_tokens(rows)
    weights = model.get_output_embeddings().weight.detach()
    bias = getattr(model.get_output_embeddings(), "bias", None)
    event_path = OUT / f"derived/{task}_event_logit_lens_margin.float32.npy"
    contribution_path = OUT / f"derived/{task}_answer_coordinate_contribution.float16.npy"
    progress_path = OUT / f"derived/{task}_logit_progress.json"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    if event_path.exists() and contribution_path.exists() and progress_path.exists():
        margins = np.lib.format.open_memmap(event_path, mode="r+")
        contribution = np.lib.format.open_memmap(contribution_path, mode="r+")
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
    else:
        margins = np.lib.format.open_memmap(event_path, mode="w+", dtype=np.float32, shape=field.shape[:-1])
        contribution = np.lib.format.open_memmap(contribution_path, mode="w+", dtype=np.float16,
                                                 shape=(field.shape[0], field.shape[1], field.shape[-1]))
        completed = 0
    device = weights.device
    answer_event = field.shape[2] - 1
    with torch.inference_mode():
        for start in range(completed, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            states = torch.tensor(np.asarray(field[start:stop], dtype=np.float32), device=device)
            normed = torch.empty_like(states)
            normed[:, :-1] = normalized_states(model, states[:, :-1])
            normed[:, -1] = states[:, -1]
            t = torch.tensor(target[start:stop], dtype=torch.long, device=device)
            f = torch.tensor(foil[start:stop], dtype=torch.long, device=device)
            delta_w = weights[t].float() - weights[f].float()
            bias_delta = torch.zeros(len(t), dtype=torch.float32, device=device) if bias is None else bias[t].float() - bias[f].float()
            values = torch.einsum("bqed,bd->bqe", normed.float(), delta_w) + bias_delta[:, None, None]
            parts = normed[:, :, answer_event].float() * delta_w[:, None, :]
            margins[start:stop] = values.cpu().numpy().astype(np.float32)
            contribution[start:stop] = parts.cpu().numpy().astype(np.float16)
            margins.flush(); contribution.flush(); save(progress_path, {"completed": stop, "shape": list(field.shape)})
            if stop % 256 == 0 or stop == len(rows): print(f"[phase2401 {task} logits] {stop}/{len(rows)}", flush=True)
            del states, normed, values, parts, delta_w
    teacher = {row["case_id"]: row for row in read_rows(P2397 / "behavior/teacher_scores.jsonl") if row["task"] == task}
    final_margin = np.asarray(margins[:, -1, answer_event], dtype=np.float64)
    expected = np.asarray([teacher[row["case_id"]]["first_divergence_logit_margin"] for row in rows], dtype=np.float64)
    check = {"mean_abs_final_margin_error": float(np.mean(np.abs(final_margin - expected))),
             "max_abs_final_margin_error": float(np.max(np.abs(final_margin - expected))),
             "target_over_foil": float(np.mean(final_margin > 0)), "teacher_target_over_foil": float(np.mean(expected > 0))}
    exact_final = collect_exact_final(task, model, rows, target, foil)
    result = {"field_shape": list(field.shape), "event_margin_path": str(event_path), "event_margin_shape": list(margins.shape),
              "answer_coordinate_contribution_path": str(contribution_path), "answer_coordinate_contribution_shape": list(contribution.shape),
              "first_divergence_index": 0, "float16_field_interface_check": check, "exact_final_interface": exact_final}
    margins.flush(); contribution.flush(); close(margins); close(contribution); close(field)
    return result


def condition_means(y: np.ndarray, rows: list[dict], train: np.ndarray, keys: tuple[str, ...]) -> tuple[np.ndarray, dict[tuple, np.ndarray]]:
    constant = y[train].mean(0)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for local, source in enumerate(train.tolist()): groups[tuple(rows[source].get(key) for key in keys)].append(local)
    return constant, {key: y[train][local].mean(0) for key, local in groups.items()}


def compilation_splits(task: str, rows: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[str, ...]]:
    train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
    if task == "selection":
        tests = {"confirmation": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "confirmation"]),
                 "fresh_unit_lockbox": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_unit_lockbox"])}
        keys = ("family", "language", "surface", "direction", "query_role", "target_candidate_slot")
    else:
        tests = {
            "one_step_confirmation": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "confirmation"]),
            "one_step_fresh_lockbox": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_unit_lockbox"]),
            "two_step_confirmation": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "composition_confirmation"]),
            "two_step_fresh_lockbox": np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_composition_lockbox"]),
        }
        keys = ("family", "language", "surface", "direction", "target_candidate_slot")
    return train, tests, keys


def add_metric(acc: dict, actual: np.ndarray, condition: np.ndarray, constant: np.ndarray) -> None:
    acc["condition_sse"] += float(np.sum((actual - condition) ** 2, dtype=np.float64))
    acc["constant_sse"] += float(np.sum((actual - constant) ** 2, dtype=np.float64))
    acc["condition_abs"] += float(np.sum(np.abs(actual - condition), dtype=np.float64))
    acc["count"] += actual.size
    acc["sum_x"] += float(np.sum(condition)); acc["sum_y"] += float(np.sum(actual))
    acc["sum_x2"] += float(np.sum(condition * condition)); acc["sum_y2"] += float(np.sum(actual * actual)); acc["sum_xy"] += float(np.sum(condition * actual))


def finish_metric(acc: dict) -> dict:
    n = acc["count"]
    cov = acc["sum_xy"] - acc["sum_x"] * acc["sum_y"] / n
    vx = acc["sum_x2"] - acc["sum_x"] ** 2 / n; vy = acc["sum_y2"] - acc["sum_y"] ** 2 / n
    return {"values": int(n), "gain_vs_constant": 1.0 - acc["condition_sse"] / max(acc["constant_sse"], 1e-30),
            "condition_mse": acc["condition_sse"] / n, "condition_mae": acc["condition_abs"] / n,
            "correlation": cov / max(math.sqrt(max(vx, 0) * max(vy, 0)), 1e-30)}


def compile_operator(task: str, model, field_path: Path, rows: list[dict], logit_result: dict) -> dict:
    cached_path = OUT / f"analysis/{task}_compilation.json"
    if cached_path.exists():
        return json.loads(cached_path.read_text(encoding="utf-8"))
    field = np.load(field_path, mmap_mode="r")
    margins = np.load(logit_result["event_margin_path"], mmap_mode="r")
    target, foil = first_tokens(rows)
    weights = model.get_output_embeddings().weight.detach()
    device = weights.device
    train, tests, keys = compilation_splits(task, rows)
    accumulators = {(split, event): defaultdict(float) for split in tests for event in range(field.shape[2])}
    row_sse = {split: {"condition": np.zeros(len(indices), dtype=np.float64), "constant": np.zeros(len(indices), dtype=np.float64)} for split, indices in tests.items()}
    with torch.inference_mode():
        for qpoint in range(Q_UPDATES):
            for event in range(field.shape[2]):
                x = np.asarray(field[:, qpoint, event], dtype=np.float32)
                y = np.asarray(field[:, qpoint + 1, event], dtype=np.float32) - x
                constant, means = condition_means(y, rows, train, keys)
                for split, indices in tests.items():
                    pred_u = np.stack([means.get(tuple(rows[index].get(key) for key in keys), constant) for index in indices])
                    base_u = np.broadcast_to(constant, pred_u.shape)
                    truth_u = y[indices]
                    row_sse[split]["condition"] += np.sum((truth_u - pred_u) ** 2, axis=1, dtype=np.float64)
                    row_sse[split]["constant"] += np.sum((truth_u - base_u) ** 2, axis=1, dtype=np.float64)
                    t = torch.tensor(target[indices], dtype=torch.long, device=device); f = torch.tensor(foil[indices], dtype=torch.long, device=device)
                    delta_w = weights[t].float() - weights[f].float()
                    x_gpu = torch.tensor(x[indices], dtype=torch.float32, device=device)
                    condition_state = normalized_states(model, x_gpu + torch.tensor(pred_u, device=device))
                    constant_state = normalized_states(model, x_gpu + torch.tensor(base_u, device=device))
                    condition_next = torch.sum(condition_state.float() * delta_w, dim=1).cpu().numpy()
                    constant_next = torch.sum(constant_state.float() * delta_w, dim=1).cpu().numpy()
                    current = np.asarray(margins[indices, qpoint, event], dtype=np.float32)
                    actual_delta = np.asarray(margins[indices, qpoint + 1, event], dtype=np.float32) - current
                    add_metric(accumulators[(split, event)], actual_delta, condition_next - current, constant_next - current)
                    del pred_u, base_u, truth_u, x_gpu, condition_state, constant_state, delta_w
                del x, y
            print(f"[phase2401 {task} compiler] update {qpoint + 1}/{Q_UPDATES}", flush=True)
    by_event = {split: {rows[0]["event_names"][event]: finish_metric(accumulators[(split, event)]) for event in range(field.shape[2])} for split in tests}
    aggregate = {}
    for split in tests:
        combined = defaultdict(float)
        for event in range(field.shape[2]):
            for key, value in accumulators[(split, event)].items(): combined[key] += value
        aggregate[split] = finish_metric(combined)
    teacher = {row["case_id"]: row for row in read_rows(P2397 / "behavior/teacher_scores.jsonl") if row["task"] == task}
    autonomous = {row["case_id"]: row for row in read_rows(P2397 / "behavior/autonomous_lockbox.jsonl") if row["task"] == task}
    row_bridge = {}
    for split, indices in tests.items():
        gain = 1.0 - row_sse[split]["condition"] / np.maximum(row_sse[split]["constant"], 1e-30)
        final_margin = np.asarray([teacher[rows[index]["case_id"]]["first_divergence_logit_margin"] for index in indices], dtype=np.float64)
        corr = float(np.corrcoef(gain, final_margin)[0, 1]) if np.std(gain) > 0 and np.std(final_margin) > 0 else float("nan")
        present = np.asarray([autonomous.get(rows[index]["case_id"], {}).get("target_present", False) for index in indices], dtype=bool)
        row_bridge[split] = {"rows": len(indices), "mean_coordinate_gain": float(np.mean(gain)),
                             "gain_to_final_margin_correlation": corr,
                             "autonomous_rows_available": int(present.size if all(rows[index]["case_id"] in autonomous for index in indices) else 0),
                             "gain_success": float(np.mean(gain[present])) if present.any() else None,
                             "gain_failure": float(np.mean(gain[~present])) if (~present).any() else None}
    result = {"aggregate": aggregate, "by_event": by_event, "row_behavior_bridge": row_bridge,
              "operator": "one-step/discovery exact condition-coordinate offset frozen in Phase2399",
              "claim_boundary": "predicts logit-lens local changes; not an internal explicit compiler or causal attribution"}
    save(cached_path, result)
    close(field); close(margins)
    return result


def append_memo(result: dict) -> None:
    memo_text = MEMO.read_text(encoding="utf-8")
    if f"## Phase {PHASE}:" in memo_text:
        marker = "**执行修正（Phase2401 float32输出接口）**"
        if result.get("all_checks_passed") and marker not in memo_text:
            with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write("\n\n" + marker + "：首次运行使用Phase2398 float16 final-norm场直接乘LM-head权重，选择/组合与直接forward首分歧margin的平均绝对误差为"
                             f"0.560030/0.346111，未通过0.1接口门。保留失败记录后补采每条答案边界float32 final state与直接logits；修正接口与裁决 `{json.dumps(result['logit_fields'], ensure_ascii=False)}`，检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。float16全层贡献仍只作可视化，精确final坐标贡献改用float32资产。\n")
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 条件局部更新到下一token概率的编译桥与组合裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 全部3072条Qwen4B任务的正确答案与foil在第0个生成token即分歧，因此可在答案边界做精确坐标分解。对embedding、36个block输出先通过模型真实final RMSNorm，再用LM head正确token与foil token权重差得到每个事件的逐层logit lens；final norm q37直接计算。保存答案边界全部2560坐标贡献，并以Phase2397直接forward的首分歧margin核验接口。随后在每个block×事件，用Phase2399只由discovery拟合并冻结的条件均值预测下一层状态，经同一RMSNorm/LM head编译为logit变化，与层/事件常量更新比较；选择任务测confirmation/fresh，组合任务把一步规则直接用于两步confirmation/fresh。最后连接每行全坐标预测收益、最终首分歧margin和自主输出target-present。

$$c_{{r,q,j}}=\operatorname{{Norm}}(H_{{r,q}})_j\bigl(W_{{y_0,j}}-W_{{\tilde y_0,j}}\bigr),\qquad
m_{{r,q}}=\sum_jc_{{r,q,j}},$$

$$\widehat{{\Delta m}}_{{r,q,e}}=m\!\left(H_{{r,q,e}}+\widehat U_{{condition}}\right)-m(H_{{r,q,e}}).$$

**结果汇总。** 输出接口 `{json.dumps(result['logit_fields'], ensure_ascii=False)}`；选择编译 `{json.dumps(result['selection_compilation'], ensure_ascii=False)}`；组合编译 `{json.dumps(result['composition_compilation'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2401_c22961_c23280_composition_output_compilation.py`；逐事件逐层logit lens、答案边界逐坐标LM-head贡献、条件算子编译指标和final位于 `tests/glm5/result/phase2401_c22961_c23280_composition_output_compilation`。

**理论进展。** 现在外部语言操作、全坐标局部更新和下一token读出第一次位于同一坐标方程中。若条件更新只改善HiddenState MSE而不能改善逐层margin变化，它只是状态拟合；若能改善两步事实段但不能改善query/answer段或最终行为，它不是组合编译器；若收益与自主成功无关，则不能把稳定passport升级为行为齿轮。

**问题硬伤与结论。** logit lens对中间层施加final norm，是诊断读出而非模型每层真实执行的操作；坐标贡献精确分解的是线性LM head接口，不是整个网络的因果归因。Qwen4B自主exact受格式协议影响而为0，行为桥以target-present并列teacher margin，不能偷换为闭环。一步与两步prompt共享事实/模板，必须重点看query_end、candidate和answer事件而不是全场平均。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        cached = json.loads(final.read_text(encoding="utf-8"))
        if cached.get("all_checks_passed"):
            append_memo(cached); print(json.dumps(cached, ensure_ascii=False, indent=2)); return
    selection_rows, composition_rows = align_rows("selection"), align_rows("composition")
    model, tokenizer, label = capability.load_model("qwen4b")
    try:
        logit_fields = {
            "selection": collect_logit_fields("selection", model, P2398 / "raw/selection_event_field.float16.npy", selection_rows),
            "composition": collect_logit_fields("composition", model, P2398 / "raw/composition_event_field.float16.npy", composition_rows),
        }
        selection_compilation = compile_operator("selection", model, P2398 / "raw/selection_event_field.float16.npy", selection_rows, logit_fields["selection"])
        composition_compilation = compile_operator("composition", model, P2398 / "raw/composition_event_field.float16.npy", composition_rows, logit_fields["composition"])
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    answer_selection = selection_compilation["by_event"]["fresh_unit_lockbox"]["answer_boundary"]
    answer_two_step = composition_compilation["by_event"]["two_step_fresh_lockbox"]["answer_boundary"]
    adjudication = {"selection_answer_boundary_compilation_gain": answer_selection["gain_vs_constant"],
                    "two_step_answer_boundary_compilation_gain": answer_two_step["gain_vs_constant"],
                    "selection_full_event_compilation_gain": selection_compilation["aggregate"]["fresh_unit_lockbox"]["gain_vs_constant"],
                    "two_step_full_event_compilation_gain": composition_compilation["aggregate"]["two_step_fresh_lockbox"]["gain_vs_constant"],
                    "exact_lm_head_coordinate_decomposition": True, "internal_compiler_proven": False,
                    "mechanism_closed": False}
    checks = {"selection_interface": logit_fields["selection"]["exact_final_interface"]["coordinate_sum_to_direct_mean_abs_error"] < 0.1 and abs(logit_fields["selection"]["exact_final_interface"]["target_over_foil"] - logit_fields["selection"]["exact_final_interface"]["teacher_target_over_foil"]) < 0.01,
              "composition_interface": logit_fields["composition"]["exact_final_interface"]["coordinate_sum_to_direct_mean_abs_error"] < 0.1 and abs(logit_fields["composition"]["exact_final_interface"]["target_over_foil"] - logit_fields["composition"]["exact_final_interface"]["teacher_target_over_foil"]) < 0.01,
              "all_first_divergence_zero": logit_fields["selection"]["first_divergence_index"] == 0 and logit_fields["composition"]["first_divergence_index"] == 0,
              "finite": all(math.isfinite(value) for value in adjudication.values() if isinstance(value, float)),
              "claim_boundary": not adjudication["mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": label, "logit_fields": logit_fields,
              "selection_compilation": selection_compilation, "composition_compilation": composition_compilation,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
