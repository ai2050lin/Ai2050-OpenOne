#!/usr/bin/env python3
"""Compile full-coordinate hypergraph trajectories into next-token and autonomous behavior."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior/qwen4b"
P2436 = RESULT / "phase2436_c34161_c34480_qwen4b_hypergraph_fullfield"
P2437 = RESULT / "phase2437_c34481_c34800_signed_trajectory_atlas"
OUT = RESULT / "phase2439_c35121_c35440_output_autonomous_bridge"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2439
CAMPAIGN = "C35121-C35440"
INTERACTIONS = ("semantic_validity", "lexical_control")
SPLITS = ("fresh_unit", "surface", "language", "direction", "family_holdout")
STAGES = ("global", "family", "diagonal", "coordinate_mismatch")
SHIFT = 791
RIDGE = 1e-8

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2437_c34481_c34800_signed_trajectory_atlas as atlas  # noqa: E402


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
    if mmap is not None:
        mmap.close()


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(a) < 2 or float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if positive == 0 or negative == 0:
        return .5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def compile_readout(rows: list[dict]) -> dict:
    state_path = P2436 / "raw/hypergraph_event_field.float16.npy"
    contribution_path = OUT / "derived/logit_lens_coordinate_contribution.float32.npy"
    weight_path = OUT / "derived/target_foil_output_weight_difference.float32.npy"
    margin_path = OUT / "derived/logit_lens_margin.float32.npy"
    progress = OUT / "derived/readout_progress.json"
    state = np.load(state_path, mmap_mode="r")
    n, qpoints, _, dim = state.shape
    contribution_path.parent.mkdir(parents=True, exist_ok=True)
    contributions = np.lib.format.open_memmap(contribution_path, mode="r+" if contribution_path.exists() else "w+",
                                               dtype=np.float32, shape=(n, qpoints, dim))
    weights = np.lib.format.open_memmap(weight_path, mode="r+" if weight_path.exists() else "w+",
                                       dtype=np.float32, shape=(n, dim))
    margins = np.lib.format.open_memmap(margin_path, mode="r+" if margin_path.exists() else "w+",
                                       dtype=np.float32, shape=(n, qpoints))
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    model = tokenizer = None
    label = "Qwen3-4B-BF16"
    if completed < n:
        model, tokenizer, label = capability.load_model("qwen4b")
        output_weight = model.get_output_embeddings().weight.detach().float().cpu().numpy()
        norm_weight = model.model.norm.weight.detach().float().cpu().numpy()
        eps = float(getattr(model.model.norm, "variance_epsilon", getattr(model.model.norm, "eps", 1e-6)))
        bias_tensor = getattr(model.get_output_embeddings(), "bias", None)
        output_bias = bias_tensor.detach().float().cpu().numpy() if bias_tensor is not None else None
    else:
        output_weight = norm_weight = output_bias = None; eps = 1e-6
    metadata = []
    try:
        for start in range(completed, n, 32):
            batch = rows[start:start + 32]
            target = np.asarray([row["target_ids"][0] for row in batch], dtype=np.int64)
            foil = np.asarray([row["foil_ids"][0] for row in batch], dtype=np.int64)
            difference = output_weight[target] - output_weight[foil]
            hidden = np.asarray(state[start:start + len(batch), :, 7], dtype=np.float32)
            projected = hidden.copy()
            rms = np.sqrt(np.mean(projected[:, :qpoints - 1].astype(np.float64) ** 2, axis=-1, keepdims=True) + eps).astype(np.float32)
            projected[:, :qpoints - 1] = projected[:, :qpoints - 1] / rms * norm_weight[None, None, :]
            cell = projected * difference[:, None, :]
            if output_bias is not None:
                bias_difference = output_bias[target] - output_bias[foil]
            else:
                bias_difference = np.zeros(len(batch), dtype=np.float32)
            contributions[start:start + len(batch)] = cell
            weights[start:start + len(batch)] = difference
            margins[start:start + len(batch)] = cell.sum(axis=-1, dtype=np.float64).astype(np.float32) + bias_difference[:, None]
            contributions.flush(); weights.flush(); margins.flush()
            save(progress, {"completed": start + len(batch), "shape": [n, qpoints, dim], "eps": eps})
            if (start + len(batch)) % 256 == 0 or start + len(batch) == n:
                print(f"[phase2439 readout] {start + len(batch)}/{n}", flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        close(state); close(contributions); close(weights); close(margins)
    for row in rows:
        metadata.append({"case_id": row["case_id"], "target_first_token": int(row["target_ids"][0]),
                         "foil_first_token": int(row["foil_ids"][0]), "answer": row["answer"], "foil": row["foil"]})
    write_rows(OUT / "index/readout_rows.jsonl", metadata)
    return {"model": label, "inference_precision": "BF16", "state_storage": "float16", "contribution_storage": "float32",
            "contribution": {"path": str(contribution_path), "shape": [n, qpoints, dim], "bytes": contribution_path.stat().st_size},
            "weight_difference": {"path": str(weight_path), "shape": [n, dim], "bytes": weight_path.stat().st_size},
            "margins": {"path": str(margin_path), "shape": [n, qpoints], "bytes": margin_path.stat().st_size}}


def output_interactions(contribution_path: str, rows: list[dict]) -> tuple[Path, list[dict]]:
    meta, index = atlas.configuration_index(rows)
    contribution = np.load(contribution_path, mmap_mode="r")
    path = OUT / "derived/signed_output_contribution_interaction.float32.npy"
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(2, contribution.shape[1], len(meta), contribution.shape[2]))
    for qpoint in range(contribution.shape[1]):
        semantic, lexical = atlas.interaction_at(contribution[:, :, None, :], qpoint, 0, index)
        output[0, qpoint] = semantic; output[1, qpoint] = lexical
    output.flush(); close(output); close(contribution)
    return path, meta


def split_masks(meta: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    unit = np.asarray([int(row["unit"]) for row in meta])
    surface = np.asarray([row["surface"] for row in meta], dtype=object)
    language = np.asarray([row["language"] for row in meta], dtype=object)
    direction = np.asarray([int(row["direction"]) for row in meta])
    return {"fresh_unit": (np.flatnonzero(unit < 5), np.flatnonzero(unit == 5)),
            "surface": (np.flatnonzero((unit < 5) & (surface == "canonical")), np.flatnonzero((unit == 5) & (surface == "natural"))),
            "language": (np.flatnonzero((unit < 5) & (language == "en")), np.flatnonzero((unit == 5) & (language == "zh"))),
            "direction": (np.flatnonzero((unit < 5) & (direction == 0)), np.flatnonzero((unit == 5) & (direction == 1))),
            "family_holdout": (np.flatnonzero(unit < 5), np.flatnonzero(unit == 5))}


def fit(train: np.ndarray, families: np.ndarray, h: np.ndarray, y: np.ndarray, conditioned: bool) -> dict:
    gh, gy = h[train].mean(0), y[train].mean(0)
    fh, fy = {}, {}
    if conditioned:
        for family in sorted(set(families[train])):
            chosen = train[families[train] == family]
            fh[family], fy[family] = h[chosen].mean(0), y[chosen].mean(0)
    bh = np.stack([fh.get(families[index], gh) for index in train])
    by = np.stack([fy.get(families[index], gy) for index in train])
    x, target = h[train] - bh, y[train] - by
    beta = np.sum(x * target, axis=0) / (np.sum(x * x, axis=0) + RIDGE)
    return {"gh": gh, "gy": gy, "fh": fh, "fy": fy, "beta": beta}


def predict(test: np.ndarray, families: np.ndarray, h: np.ndarray, fitted: dict) -> list[np.ndarray]:
    gy = np.broadcast_to(fitted["gy"], (len(test), h.shape[1]))
    fy = np.stack([fitted["fy"].get(families[index], fitted["gy"]) for index in test])
    fh = np.stack([fitted["fh"].get(families[index], fitted["gh"]) for index in test])
    centered = h[test] - fh
    return [gy, fy, fy + centered * fitted["beta"], fy + centered * np.roll(fitted["beta"], SHIFT)]


def gain(truth: np.ndarray, predicted: list[np.ndarray]) -> tuple[np.ndarray, bool]:
    denominator = float(np.sum((truth.astype(np.float64) - predicted[0].astype(np.float64)) ** 2))
    scale = float(np.sum(truth.astype(np.float64) ** 2))
    if denominator <= max(1e-20, scale * 1e-12):
        return np.zeros(4, dtype=np.float32), False
    return np.asarray([0.] + [1 - float(np.sum((truth.astype(np.float64) - value.astype(np.float64)) ** 2)) / denominator
                              for value in predicted[1:]], dtype=np.float32), True


def evaluate(train: np.ndarray, test: np.ndarray, families: np.ndarray, h: np.ndarray, y: np.ndarray,
             holdout: bool) -> tuple[np.ndarray, bool]:
    if not holdout:
        fitted = fit(train, families, h, y, True)
        return gain(y[test], predict(test, families, h, fitted))
    truths, predictions = [], [[] for _ in STAGES]
    for family in sorted(set(families)):
        tr, te = train[families[train] != family], test[families[test] == family]
        fitted = fit(tr, families, h, y, False)
        truths.append(y[te])
        for stage, value in enumerate(predict(te, families, h, fitted)):
            predictions[stage].append(value)
    return gain(np.concatenate(truths), [np.concatenate(value) for value in predictions])


def internal_to_output(path: Path, meta: list[dict]) -> dict:
    internal = np.load(P2437 / "derived/signed_interaction_state.float16.npy", mmap_mode="r")
    output = np.load(path, mmap_mode="r")
    families = np.asarray([row["family"] for row in meta], dtype=object)
    specs = split_masks(meta)
    events = internal.shape[2]; qpoints = internal.shape[1]
    metrics = np.zeros((2, len(SPLITS), len(STAGES), qpoints, events), dtype=np.float32)
    active = np.zeros((2, len(SPLITS), qpoints, events), dtype=np.uint8)
    for ii in range(2):
        target = np.asarray(output[ii, -1], dtype=np.float32)
        for qpoint in range(qpoints):
            for event in range(events):
                h = np.asarray(internal[ii, qpoint, event], dtype=np.float32)
                for si, split in enumerate(SPLITS):
                    train, test = specs[split]
                    values, is_active = evaluate(train, test, families, h, target, split == "family_holdout")
                    metrics[ii, si, :, qpoint, event] = values; active[ii, si, qpoint, event] = int(is_active)
            if (qpoint + 1) % 6 == 0 or qpoint + 1 == qpoints:
                print(f"[phase2439 bridge] interaction={INTERACTIONS[ii]} qpoint={qpoint + 1}/{qpoints}", flush=True)
    derived = OUT / "derived"
    np.save(derived / "internal_to_final_output_gains.float32.npy", metrics)
    np.save(derived / "internal_to_final_output_active.uint8.npy", active)
    summary = {}
    for ii, interaction_name in enumerate(INTERACTIONS):
        summary[interaction_name] = {}
        for si, split in enumerate(SPLITS):
            chosen = active[ii, si].astype(bool)
            summary[interaction_name][split] = {STAGES[stage]: float(metrics[ii, si, stage][chosen].mean()) if chosen.any() else 0.0
                                                for stage in range(len(STAGES))}
            summary[interaction_name][split]["physical_advantage"] = (
                summary[interaction_name][split]["diagonal"] - summary[interaction_name][split]["coordinate_mismatch"])
            best_flat = int(np.argmax(metrics[ii, si, STAGES.index("diagonal")]))
            summary[interaction_name][split]["best_qpoint_event"] = [int(x) for x in np.unravel_index(best_flat, (qpoints, events))]
    close(internal); close(output)
    return {"summary": summary, "metrics": str(derived / "internal_to_final_output_gains.float32.npy"),
            "active": str(derived / "internal_to_final_output_active.uint8.npy")}


def closure(readout: dict, teacher: list[dict]) -> dict:
    margins = np.load(readout["margins"]["path"], mmap_mode="r")
    predicted = np.asarray(margins[:, -1], dtype=np.float64)
    truth = np.asarray([row["first_divergence_logit_margin"] for row in teacher], dtype=np.float64)
    residual = predicted - truth
    result = {"rows": len(truth), "correlation": pearson(predicted, truth),
              "rmse": float(np.sqrt(np.mean(residual ** 2))), "mae": float(np.mean(np.abs(residual))),
              "max_abs": float(np.max(np.abs(residual))),
              "relative_rmse": float(np.sqrt(np.mean(residual ** 2)) / max(np.sqrt(np.mean(truth ** 2)), 1e-30)),
              "all_first_divergence_zero": all(int(row["first_divergence_index"]) == 0 for row in teacher)}
    close(margins); return result


def autonomous_bridge(rows: list[dict], teacher: list[dict]) -> dict:
    autonomous = read_rows(P2435 / "behavior/autonomous_lockbox.jsonl")
    teacher_map = {record["case_id"]: float(record["first_divergence_logit_margin"]) for record in teacher}
    scores = np.asarray([teacher_map[record["case_id"]] for record in autonomous], dtype=np.float64)
    exact = np.asarray([record["exact"] for record in autonomous], dtype=bool)
    present = np.asarray([record["target_present"] for record in autonomous], dtype=bool)
    return {"rows": len(autonomous), "exact_rate": float(exact.mean()), "target_present_rate": float(present.mean()),
            "margin_exact_auc": auc(scores, exact), "margin_exact_correlation": pearson(scores, exact.astype(float)),
            "margin_target_present_auc": auc(scores, present),
            "margin_target_present_correlation": pearson(scores, present.astype(float)),
            "format_warning": "exact is zero because generations commonly prepend 'Answer:'; target_present is the usable but looser endpoint"}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 条件坐标轨迹到第一分歧token与自主输出的参数级编译桥（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对2304条Qwen4B材料读取answer boundary的38 checkpoint全2560坐标。对q0–q36先施加真实final RMSNorm参数，对q37直接使用已归一化状态；再乘目标/foil第一分歧token的输出嵌入权重差，保存每行、每层、每坐标贡献及logit-lens margin。把贡献场按Phase2437相同双角色/三有效性公式组合，测试各事件内部interaction能否在fresh、表述、中英、方向与留一family锁箱预测最终输出贡献；最后比较教师margin与64条自主生成。

$$\widetilde H_{{q,j}}=\frac{{H_{{q,j}}}}{{\sqrt{{d^{{-1}}\sum_k H_{{q,k}}^2+\epsilon}}}}\gamma_j,\qquad
C_{{q,j}}=\widetilde H_{{q,j}}(W_{{a,j}}-W_{{b,j}}),$$
$$m_q=\sum_j C_{{q,j}},\qquad
\hat I^C_{{37,j}}=\bar I^C_{{f,j}}+\beta_j(I^H_{{q,e,j}}-\bar I^H_{{f,j}}).$$

**结果汇总。** 参数级场 `{json.dumps(result['readout'], ensure_ascii=False)}`；输出恒等检查 `{json.dumps(result['closure'], ensure_ascii=False)}`；内部到输出 `{json.dumps(result['internal_to_output'], ensure_ascii=False)}`；自主桥 `{json.dumps(result['autonomous'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2439_c35121_c35440_output_autonomous_bridge.py`；2304×38×2560 float32逐坐标输出贡献、逐行权重差、layerwise margin、输出interaction与五锁箱桥指标位于`tests/glm5/result/phase2439_c35121_c35440_output_autonomous_bridge/derived`。

**分析与理论进展。** 这一步把HiddenState坐标放到真实输出嵌入参数上：每个坐标对具体下一token竞争贡献多少可以直接核对。输出恒等式是架构事实，不是新机制；真正证据是更早事件/层的条件纹理能否在未见实体、语言、表达、方向和family上预测最终逐坐标贡献，并与自主输出同向。

**问题硬伤与结论。** logit lens对中间层额外施加final norm，只是可比较读出，不代表模型在该层真实提前输出。内部→输出仍为离线对角映射。自主exact受`Answer:`前缀破坏，target-present较宽松且只有64条；AUC不能单独闭合机制。只有跨语言/留族和自主门共同通过，才可称为普遍语言编码编译器。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    teacher = read_rows(P2435 / "behavior/teacher_scores.jsonl")
    readout = compile_readout(rows)
    output_path, meta = output_interactions(readout["contribution"]["path"], rows)
    bridge = internal_to_output(output_path, meta)
    numeric = closure(readout, teacher)
    autonomous = autonomous_bridge(rows, teacher)
    sem = bridge["summary"]["semantic_validity"]
    adjudication = {"parameter_level_output_identity_qualified": numeric["correlation"] > .995 and numeric["relative_rmse"] < .08,
                    "semantic_internal_output_gain_positive_all_splits": all(sem[split]["diagonal"] > 0 for split in SPLITS),
                    "semantic_internal_output_physical_advantage_all_splits": all(sem[split]["physical_advantage"] > 0 for split in SPLITS),
                    "teacher_margin_predicts_autonomous_target": autonomous["margin_target_present_auc"] > .7 and
                                                                  autonomous["margin_target_present_correlation"] > .2,
                    "language_encoding_compiler_closed": False}
    checks = {"rows_2304": readout["contribution"]["shape"] == [2304, 38, 2560],
              "first_divergence_zero": numeric["all_first_divergence_zero"],
              "numeric_readout": adjudication["parameter_level_output_identity_qualified"],
              "five_splits": set(bridge["summary"]["semantic_validity"]) == set(SPLITS),
              "finite": all(math.isfinite(value) for value in numeric.values() if isinstance(value, float)) and
                        all(math.isfinite(value) for value in autonomous.values() if isinstance(value, float)),
              "raw_retained": (P2436 / "raw/hypergraph_event_field.float16.npy").exists(),
              "claim_boundary": not adjudication["language_encoding_compiler_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "readout": readout,
              "output_interaction_path": str(output_path), "closure": numeric,
              "internal_to_output": bridge, "autonomous": autonomous,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
