#!/usr/bin/env python3
"""Build basic full-coordinate response maps and held-out transport ledgers."""
from __future__ import annotations

import hashlib
import json
import math
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
P2309 = RESULT / "phase2309_c4321_c4440_multistep_future_contract"
P2310 = RESULT / "phase2310_c4441_c4580_qwen4b_multistep_field"
OUT = RESULT / "phase2311_c4581_c4700_basic_future_accounting"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = P2309 / "material/multistep_future_bilingual.jsonl"
BOUNDARY = P2310 / "raw/qwen4b_multistep_boundary_all_checkpoints.float16.npy"
BOUNDARY_LOGITS = P2310 / "raw/qwen4b_multistep_boundary_full_vocabulary.float16.npy"
FUTURE_FIELD = P2310 / "raw/qwen4b_teacher_future_selected_checkpoints.float16.npy"
FUTURE_LOGITS = P2310 / "raw/qwen4b_teacher_future_full_vocabulary.float16.npy"
FUTURE_INDEX = P2310 / "index/teacher_future_rows.jsonl"
STATE_RESPONSE = OUT / "atlas/state_response_all_checkpoints.float16.npy"
SURFACE_RESPONSE = OUT / "atlas/surface_response_all_checkpoints.float16.npy"
LANGUAGE_RESPONSE = OUT / "atlas/language_package_response_all_checkpoints.float16.npy"
FUTURE_TRANSITION = OUT / "atlas/teacher_future_transition_selected_checkpoints.float16.npy"
sys.path.insert(0, str(TESTS))

import phase2309_c4321_c4440_multistep_future_contract as contract  # noqa: E402


PHASE = 2311
CAMPAIGN = "C4581-C4700"
EPS = 1e-12


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    contract.write_rows(path, rows)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def pair_indices(rows: list[dict], factor: str) -> list[dict]:
    groups: dict[tuple, dict[Any, tuple[int, dict]]] = defaultdict(dict)
    if factor == "state":
        keys = ("family", "language", "surface", "unit")
        values = (0, 1)
    elif factor == "surface":
        keys = ("family", "language", "unit", "state")
        values = ("narrative", "dialogue")
    elif factor == "language":
        keys = ("family", "surface", "unit", "state")
        values = ("en", "zh")
    else:
        raise KeyError(factor)
    for i, row in enumerate(rows):
        key = tuple(row[name] for name in keys)
        groups[key][row[factor]] = (i, row)
    output = []
    for key, pair in sorted(groups.items(), key=lambda item: str(item[0])):
        if set(pair) != set(values):
            raise RuntimeError(("missing_pair", factor, key, sorted(pair, key=str)))
        left_i, left = pair[values[0]]
        right_i, right = pair[values[1]]
        if factor == "surface" and left["target_mention_order"] != right["target_mention_order"]:
            raise RuntimeError(("surface_order_mismatch", key))
        output.append({
            "pair_row": len(output),
            "factor": factor,
            "left_index": left_i,
            "right_index": right_i,
            "left_case_id": left["case_id"],
            "right_case_id": right["case_id"],
            "family": left["family"],
            "partition": left["partition"],
            "unit": int(left["unit"]),
            "language": left.get("language") if factor != "language" else "en_to_zh_package",
            "surface": left.get("surface") if factor != "surface" else "narrative_to_dialogue",
            "state": int(left["state"]) if factor != "state" else "state0_to_state1",
            "target_mention_order": left["target_mention_order"],
        })
    return output


def create_response(source_path: Path, pairs: list[dict], output_path: Path,
                    index_path: Path) -> dict:
    source = np.load(source_path, mmap_mode="r")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and index_path.exists():
        values = np.load(output_path, mmap_mode="r")
        shape = list(values.shape)
        close_memmap(values)
        close_memmap(source)
        return {"path": str(output_path.relative_to(ROOT)), "shape": shape,
                "index": str(index_path.relative_to(ROOT)), "resumed": True}
    shape = (len(pairs), *source.shape[1:])
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=shape)
    try:
        for start in range(0, len(pairs), 32):
            batch = pairs[start:start + 32]
            left = [row["left_index"] for row in batch]
            right = [row["right_index"] for row in batch]
            delta = (np.asarray(source[right], dtype=np.float32)
                     - np.asarray(source[left], dtype=np.float32))
            output[start:start + len(batch)] = delta.astype(np.float16)
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    write_rows(index_path, pairs)
    return {"path": str(output_path.relative_to(ROOT)), "shape": list(shape),
            "index": str(index_path.relative_to(ROOT)), "resumed": False}


def future_transition_index(index: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in index:
        groups[row["case_id"]].append(row)
    output: list[dict] = []
    for case_id, values in groups.items():
        values.sort(key=lambda row: int(row["future_step"]))
        for left, right in zip(values, values[1:]):
            if int(right["future_step"]) != int(left["future_step"]) + 1:
                raise RuntimeError(("future_step_gap", case_id, left, right))
            output.append({
                "transition_row": len(output),
                "left_row": int(left["row"]),
                "right_row": int(right["row"]),
                "case_id": case_id,
                "family": left["family"],
                "language": left["language"],
                "surface": left["surface"],
                "partition": left["partition"],
                "state": int(left["state"]),
                "unit": int(left["unit"]),
                "from_step": int(left["future_step"]),
                "to_step": int(right["future_step"]),
                "consumed_token_id": int(left["next_target_token_id"]),
                "next_target_token_id": int(right["next_target_token_id"]),
            })
    return output


def create_future_transitions(index: list[dict]) -> tuple[dict, list[dict]]:
    pairs = future_transition_index(index)
    index_path = OUT / "index/teacher_future_transition_rows.jsonl"
    source = np.load(FUTURE_FIELD, mmap_mode="r")
    if FUTURE_TRANSITION.exists() and index_path.exists():
        values = np.load(FUTURE_TRANSITION, mmap_mode="r")
        shape = list(values.shape)
        close_memmap(values)
        close_memmap(source)
        return ({"path": str(FUTURE_TRANSITION.relative_to(ROOT)), "shape": shape,
                 "index": str(index_path.relative_to(ROOT)), "resumed": True}, pairs)
    FUTURE_TRANSITION.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(pairs), *source.shape[1:])
    output = np.lib.format.open_memmap(FUTURE_TRANSITION, mode="w+", dtype=np.float16,
                                       shape=shape)
    try:
        for start in range(0, len(pairs), 32):
            batch = pairs[start:start + 32]
            left = [row["left_row"] for row in batch]
            right = [row["right_row"] for row in batch]
            output[start:start + len(batch)] = (
                np.asarray(source[right], dtype=np.float32)
                - np.asarray(source[left], dtype=np.float32)
            ).astype(np.float16)
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    write_rows(index_path, pairs)
    return ({"path": str(FUTURE_TRANSITION.relative_to(ROOT)), "shape": list(shape),
             "index": str(index_path.relative_to(ROOT)), "resumed": False}, pairs)


def distribution_distances(logits_path: Path, pairs_by_factor: dict[str, list[dict]]) -> dict:
    output_path = OUT / "probability/boundary_factor_distances.jsonl"
    summary_path = OUT / "probability/boundary_factor_summary.json"
    if output_path.exists() and summary_path.exists():
        values = read_rows(output_path)
        return {"rows": len(values), "path": str(output_path.relative_to(ROOT)),
                "summary": json.loads(summary_path.read_text(encoding="utf-8")), "resumed": True}
    logits = np.load(logits_path, mmap_mode="r")
    output: list[dict] = []
    try:
        for factor, pairs in pairs_by_factor.items():
            for start in range(0, len(pairs), 8):
                batch = pairs[start:start + 8]
                left = torch.tensor(np.asarray(logits[[row["left_index"] for row in batch]],
                                                      dtype=np.float32))
                right = torch.tensor(np.asarray(logits[[row["right_index"] for row in batch]],
                                                       dtype=np.float32))
                p, q = torch.softmax(left, dim=-1), torch.softmax(right, dim=-1)
                midpoint = 0.5 * (p + q)
                js = 0.5 * torch.sum(p * (torch.log(p + EPS) - torch.log(midpoint + EPS)), dim=-1)
                js += 0.5 * torch.sum(q * (torch.log(q + EPS) - torch.log(midpoint + EPS)), dim=-1)
                tv = 0.5 * torch.sum(torch.abs(p - q), dim=-1)
                for i, pair in enumerate(batch):
                    output.append({
                        **pair,
                        "js": float(js[i].item()),
                        "total_variation": float(tv[i].item()),
                    })
    finally:
        close_memmap(logits)
    write_rows(output_path, output)

    def stats(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {"n": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
                "min": float(array.min()), "max": float(array.max())}
    summary: dict[str, Any] = {"overall": {}, "families": {}}
    for factor in pairs_by_factor:
        subset = [row for row in output if row["factor"] == factor]
        summary["overall"][factor] = {
            "js": stats([row["js"] for row in subset]),
            "total_variation": stats([row["total_variation"] for row in subset]),
        }
    for family in contract.FAMILIES:
        summary["families"][family] = {}
        for factor in pairs_by_factor:
            subset = [row for row in output if row["factor"] == factor and row["family"] == family]
            summary["families"][family][factor] = {
                "js": stats([row["js"] for row in subset]),
                "total_variation": stats([row["total_variation"] for row in subset]),
            }
    save(summary_path, summary)
    return {"rows": len(output), "path": str(output_path.relative_to(ROOT)),
            "summary": summary, "resumed": False}


def future_distribution_distances(index: list[dict], transitions: list[dict]) -> dict:
    output_path = OUT / "probability/teacher_future_consecutive_distances.jsonl"
    summary_path = OUT / "probability/teacher_future_consecutive_summary.json"
    if output_path.exists() and summary_path.exists():
        values = read_rows(output_path)
        return {"rows": len(values), "path": str(output_path.relative_to(ROOT)),
                "summary": json.loads(summary_path.read_text(encoding="utf-8")), "resumed": True}
    logits = np.load(FUTURE_LOGITS, mmap_mode="r")
    output: list[dict] = []
    try:
        for start in range(0, len(transitions), 8):
            batch = transitions[start:start + 8]
            left = torch.tensor(np.asarray(logits[[row["left_row"] for row in batch]], dtype=np.float32))
            right = torch.tensor(np.asarray(logits[[row["right_row"] for row in batch]], dtype=np.float32))
            p, q = torch.softmax(left, dim=-1), torch.softmax(right, dim=-1)
            midpoint = 0.5 * (p + q)
            js = 0.5 * torch.sum(p * (torch.log(p + EPS) - torch.log(midpoint + EPS)), dim=-1)
            js += 0.5 * torch.sum(q * (torch.log(q + EPS) - torch.log(midpoint + EPS)), dim=-1)
            tv = 0.5 * torch.sum(torch.abs(p - q), dim=-1)
            for i, row in enumerate(batch):
                output.append({**row, "js": float(js[i].item()),
                               "total_variation": float(tv[i].item())})
    finally:
        close_memmap(logits)
    write_rows(output_path, output)

    def grouped(values: list[dict]) -> dict:
        return {
            "n": len(values),
            "js_mean": float(np.mean([row["js"] for row in values])),
            "js_median": float(np.median([row["js"] for row in values])),
            "tv_mean": float(np.mean([row["total_variation"] for row in values])),
        }
    summary = {
        "overall": grouped(output),
        "families": {family: grouped([row for row in output if row["family"] == family])
                     for family in contract.FAMILIES},
        "steps": {str(step): grouped([row for row in output if row["from_step"] == step])
                  for step in sorted(set(row["from_step"] for row in output))},
    }
    save(summary_path, summary)
    return {"rows": len(output), "path": str(output_path.relative_to(ROOT)),
            "summary": summary, "resumed": False}


def response_metrics(response_paths: dict[str, tuple[Path, list[dict]]]) -> dict:
    path = OUT / "coordinate/response_metrics.jsonl"
    if path.exists():
        rows = read_rows(path)
        return {"path": str(path.relative_to(ROOT)), "rows": len(rows), "resumed": True}
    output: list[dict] = []
    for factor, (response_path, index) in response_paths.items():
        values = np.load(response_path, mmap_mode="r")
        try:
            for start in range(0, len(index), 32):
                batch = np.asarray(values[start:start + 32], dtype=np.float32)
                for local, meta in enumerate(index[start:start + len(batch)]):
                    for q in range(batch.shape[1]):
                        vector = batch[local, q]
                        output.append({
                            "factor": factor,
                            "family": meta["family"],
                            "partition": meta["partition"],
                            "pair_row": int(meta.get("pair_row", meta.get("transition_row", start + local))),
                            "checkpoint_index": q,
                            "mean_abs": float(np.mean(np.abs(vector), dtype=np.float64)),
                            "rms": float(np.sqrt(np.mean(vector.astype(np.float64) ** 2))),
                            "positive_fraction": float(np.mean(vector > 0)),
                            "negative_fraction": float(np.mean(vector < 0)),
                            "exact_zero_fraction": float(np.mean(vector == 0)),
                        })
        finally:
            close_memmap(values)
    write_rows(path, output)
    return {"path": str(path.relative_to(ROOT)), "rows": len(output), "resumed": False}


def fit_diagonal(x: np.ndarray, y: np.ndarray, extra: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    # Each physical coordinate is fitted independently; no coordinate selection or projection occurs.
    n, d = x.shape
    if extra is None:
        sx, sy = x.sum(0), y.sum(0)
        sxx, sxy = (x * x).sum(0), (x * y).sum(0)
        denom = n * sxx - sx * sx
        scale = np.maximum(np.abs(n * sxx), 1.0)
        safe = np.abs(denom) > 1e-8 * scale
        a = np.zeros(d, dtype=np.float64)
        a[safe] = (n * sxy[safe] - sx[safe] * sy[safe]) / denom[safe]
        b = (sy - a * sx) / n
        return a.astype(np.float32), b.astype(np.float32)
    z = extra
    design = np.stack([x, z, np.ones_like(x)], axis=-1).astype(np.float64)
    gram = np.einsum("ndi,ndj->dij", design, design)
    rhs = np.einsum("ndi,nd->di", design, y.astype(np.float64))
    trace = np.trace(gram, axis1=1, axis2=2)
    ridge = np.maximum(trace * 1e-6, 1e-8)
    gram[:, 0, 0] += ridge
    gram[:, 1, 1] += ridge
    gram[:, 2, 2] += ridge
    coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
    return tuple(coefficients[:, i].astype(np.float32) for i in range(3))


def route_tournament(rows: list[dict], state_pairs: list[dict]) -> dict:
    path = OUT / "transport/basic_state_response_tournament.jsonl"
    summary_path = OUT / "transport/basic_state_response_summary.json"
    if path.exists() and summary_path.exists():
        values = read_rows(path)
        return {"path": str(path.relative_to(ROOT)), "rows": len(values),
                "summary": json.loads(summary_path.read_text(encoding="utf-8")), "resumed": True}
    response = np.load(STATE_RESPONSE, mmap_mode="r")
    boundary = np.load(BOUNDARY, mmap_mode="r")
    qpoints = list(contract.QPOINTS_4B)
    records: list[dict] = []
    try:
        by_family = {family: [pair for pair in state_pairs if pair["family"] == family]
                     for family in contract.FAMILIES}
        for family, pairs in by_family.items():
            train_pairs = [pair for pair in pairs if pair["partition"] == "discovery"]
            train_idx = [pair["pair_row"] for pair in train_pairs]
            for q_left, q_right in zip(qpoints, qpoints[1:]):
                x_train = np.asarray(response[train_idx, q_left], dtype=np.float32)
                y_train = np.asarray(response[train_idx, q_right], dtype=np.float32)
                h_train = np.asarray(boundary[[pair["left_index"] for pair in train_pairs], q_left],
                                             dtype=np.float32)
                mean_target = y_train.mean(axis=0)
                diagonal = fit_diagonal(x_train, y_train)
                conditioned = fit_diagonal(x_train, y_train, h_train)
                for partition in ("confirmation", "fresh_confirmation", "fresh_lockbox"):
                    test_pairs = [pair for pair in pairs if pair["partition"] == partition]
                    idx = [pair["pair_row"] for pair in test_pairs]
                    x = np.asarray(response[idx, q_left], dtype=np.float32)
                    y = np.asarray(response[idx, q_right], dtype=np.float32)
                    h = np.asarray(boundary[[pair["left_index"] for pair in test_pairs], q_left],
                                           dtype=np.float32)
                    predictions = {
                        "zero": np.zeros_like(y),
                        "previous_response": x,
                        "discovery_mean": np.broadcast_to(mean_target, y.shape),
                        "diagonal_affine": diagonal[0] * x + diagonal[1],
                        "base_conditioned_diagonal": (
                            conditioned[0] * x + conditioned[1] * h + conditioned[2]
                        ),
                    }
                    mean_coord_error = np.mean((predictions["discovery_mean"] - y) ** 2, axis=0)
                    denom = np.sum(y.astype(np.float64) ** 2, axis=1) + EPS
                    for model_name, prediction in predictions.items():
                        residual = prediction.astype(np.float64) - y.astype(np.float64)
                        relative = np.sum(residual ** 2, axis=1) / denom
                        coord_error = np.mean(residual ** 2, axis=0)
                        records.append({
                            "family": family,
                            "from_checkpoint": q_left,
                            "to_checkpoint": q_right,
                            "partition": partition,
                            "model": model_name,
                            "rows": len(test_pairs),
                            "relative_mse_mean": float(np.mean(relative)),
                            "relative_mse_median": float(np.median(relative)),
                            "coordinate_win_fraction_vs_discovery_mean": float(np.mean(
                                coord_error < mean_coord_error
                            )),
                        })
    finally:
        close_memmap(response)
        close_memmap(boundary)
    write_rows(path, records)
    summary: dict[str, Any] = {"families": {}, "interpretation": {}}
    for family in contract.FAMILIES:
        fresh = [row for row in records if row["family"] == family
                 and row["partition"] in ("fresh_confirmation", "fresh_lockbox")]
        by_model = defaultdict(list)
        for row in fresh:
            by_model[row["model"]].append(row["relative_mse_median"])
        medians = {model: float(np.median(values)) for model, values in by_model.items()}
        best = min(medians, key=medians.get)
        summary["families"][family] = {"fresh_median_relative_mse": medians,
                                        "descriptive_best_model": best}
    summary["interpretation"] = {
        "selection_status": "descriptive_route_comparison_not_a_preregistered_mechanism_gate",
        "previous_response": "identity transport baseline",
        "discovery_mean": "mean donor baseline only, not the proposed mechanism",
        "diagonal_affine": "same-coordinate fitted transport",
        "base_conditioned_diagonal": "same-coordinate transport additionally conditioned on the current base state",
        "non_diagonal_not_identifiable": (
            "48 discovery state pairs per family cannot identify a 2560x2560 map without new perturbation data"
        ),
    }
    save(summary_path, summary)
    return {"path": str(path.relative_to(ROOT)), "rows": len(records),
            "summary": summary, "resumed": False}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    probability = result["boundary_probability"]["summary"]["overall"]
    best = {family: value["descriptive_best_model"]
            for family, value in result["transport"]["summary"]["families"].items()}
    text = rf"""

## Phase {PHASE}: 多步未来基础全坐标分账与传动路线竞争（{CAMPAIGN}） [{stamp}]

**测试原理。** 本阶段不加载模型，只读取 Phase2310 冻结生成的原始场。对同一 unit 构造三种完整配对：状态 `s0 -> s1`、同事实顺序表面 `narrative -> dialogue`、对应 unit 的 `en -> zh` 输入包；分别保存所有 38 检查点和全部 2560 坐标的有符号响应。对 teacher-forced 正确未来，把相邻真实未来步的场差保存为第四类响应。完整词表以 JS 和 total variation 分账。语言响应包含词汇、tokenization 和语言整体变化，因此明确叫“语言包响应”，不冒充纯语言算子。

$$
R_s=H_{{s=1}}-H_{{s=0}},\qquad
R_u=H_{{dialogue}}-H_{{narrative}},\qquad
R_\ell=H_{{zh}}-H_{{en}},
$$
$$
R_{{future,k}}=H(x,y_{{\le k}})-H(x,y_{{<k}}).
$$

基础路线只在每个原物理坐标上独立拟合，并保持 discovery、confirmation、fresh-confirmation、fresh-lockbox 顺序：
$$
\widehat R_{{q',j}}=a_{{q,j}}R_{{q,j}}+b_{{q,j}},
$$
$$
\widehat R_{{q',j}}=a_{{q,j}}R_{{q,j}}+c_{{q,j}}H^0_{{q,j}}+b_{{q,j}}.
$$
零预测、直接沿用上一步响应和 discovery 均值仅作控制。每族 discovery 只有 48 个状态对，无法识别 2560×2560 非对角矩阵；本阶段拒绝用不可识别的大矩阵制造“齿轮”。

**结果汇总。** 边界完整词表总体分账 `{json.dumps(probability, ensure_ascii=False)}`。完整响应资产：状态 `{json.dumps(result['responses']['state'], ensure_ascii=False)}`，表面 `{json.dumps(result['responses']['surface'], ensure_ascii=False)}`，语言包 `{json.dumps(result['responses']['language'], ensure_ascii=False)}`，未来步 `{json.dumps(result['responses']['future_transition'], ensure_ascii=False)}`。相邻 teacher-forced 未来分布 `{json.dumps(result['future_probability']['summary'], ensure_ascii=False)}`。逐坐标幅值账 `{json.dumps(result['response_metrics'], ensure_ascii=False)}`。fresh 分区描述性最佳基础路线 `{json.dumps(best, ensure_ascii=False)}`；完整路线账 `{json.dumps(result['transport'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展与硬伤。** `{result['strict_conclusion']}`。最重要的方法结论是：当前观察样本足以比较对角、基态条件对角和简单控制，却不足以唯一识别任意非对角传动图；要研究 `j -> k` 必须新增受控扰动，而不是用高级模型掩盖欠定性。JS、RMS 和逐坐标 OLS 是不同读尺，不能互相替代。状态、表面和语言包响应都可能很宽；宽不等于语义、全息或因果。teacher-forced 步变化还混合了新增目标 token 的词汇身份。脚本 `tests/glm5/phase2311_c4581_c4700_basic_future_accounting.py`；结果 `tests/glm5/result/phase2311_c4581_c4700_basic_future_accounting`。下一步只对“完整未来严格行为”和“自由身份”同时合格的族运行固定方向全坐标局部梯度与预定扰动；其余族保留观察但不作机制调用。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    p2309 = json.loads((P2309 / "analysis/final.json").read_text(encoding="utf-8"))
    p2310 = json.loads((P2310 / "analysis/final.json").read_text(encoding="utf-8"))
    if not p2309["all_checks_passed"] or not p2310["all_checks_passed"]:
        raise RuntimeError("Parent phases are not authorized")
    rows = read_rows(ROWS_PATH)
    pairs = {factor: pair_indices(rows, factor) for factor in ("state", "surface", "language")}
    state = create_response(BOUNDARY, pairs["state"], STATE_RESPONSE,
                            OUT / "index/state_response_rows.jsonl")
    surface = create_response(BOUNDARY, pairs["surface"], SURFACE_RESPONSE,
                              OUT / "index/surface_response_rows.jsonl")
    language = create_response(BOUNDARY, pairs["language"], LANGUAGE_RESPONSE,
                               OUT / "index/language_response_rows.jsonl")
    future_index = read_rows(FUTURE_INDEX)
    future_transition, future_pairs = create_future_transitions(future_index)
    probability = distribution_distances(BOUNDARY_LOGITS, pairs)
    future_probability = future_distribution_distances(future_index, future_pairs)
    metrics = response_metrics({
        "state": (STATE_RESPONSE, pairs["state"]),
        "surface": (SURFACE_RESPONSE, pairs["surface"]),
        "language_package": (LANGUAGE_RESPONSE, pairs["language"]),
        "teacher_future": (FUTURE_TRANSITION, future_pairs),
    })
    transport = route_tournament(rows, pairs["state"])
    responses = {"state": state, "surface": surface, "language": language,
                 "future_transition": future_transition}
    checks = {
        "parents_authorized": p2309["all_checks_passed"] and p2310["all_checks_passed"],
        "state_pairs_complete": len(pairs["state"]) == len(rows) // 2,
        "surface_pairs_complete": len(pairs["surface"]) == len(rows) // 2,
        "language_pairs_complete": len(pairs["language"]) == len(rows) // 2,
        "state_all_checkpoints_coordinates": state["shape"] == [len(rows) // 2, 38, 2560],
        "surface_all_checkpoints_coordinates": surface["shape"] == [len(rows) // 2, 38, 2560],
        "language_all_checkpoints_coordinates": language["shape"] == [len(rows) // 2, 38, 2560],
        "future_all_selected_checkpoints_coordinates": future_transition["shape"][1:] == [10, 2560],
        "all_probability_pairs": probability["rows"] == 3 * len(rows) // 2,
        "all_future_probability_pairs": future_probability["rows"] == len(future_pairs),
        "all_response_metrics": metrics["rows"] > 0,
        "transport_all_families": set(transport["summary"]["families"]) == set(contract.FAMILIES),
        "no_nonidentifiable_full_matrix_claim": True,
        "no_topk_or_pca": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed",
        "responses": responses,
        "boundary_probability": probability,
        "future_probability": future_probability,
        "response_metrics": metrics,
        "transport": transport,
        "hashes": {
            "state": file_hash(STATE_RESPONSE),
            "surface": file_hash(SURFACE_RESPONSE),
            "language": file_hash(LANGUAGE_RESPONSE),
            "future_transition": file_hash(FUTURE_TRANSITION),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "The campaign now has complete signed state, surface, language-package, and teacher-future response "
            "fields plus a held-out basic diagonal transport ledger. These establish broad conditional response "
            "maps, but the observational sample count cannot identify an arbitrary non-diagonal gear matrix, "
            "and no route is yet a causal or advanced-mathematical mechanism."
        ),
        "next_authorization": (
            "Run exact local full-coordinate gradients and preregistered structured perturbations only for "
            "families passing both complete-future and free-identity gates."
        ),
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
