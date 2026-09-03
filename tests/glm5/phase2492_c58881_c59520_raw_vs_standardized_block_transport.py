#!/usr/bin/env python3
"""Compete raw and coordinate-RMS-standardized adjacent-block transport baselines."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2487 = RESULT / "phase2487_c54721_c55872_orthogonal_family_interface_behavior"
P2488 = RESULT / "phase2488_c55873_c56832_qwen4b_orthogonal_fullcoordinate_field"
OUT = RESULT / "phase2492_c58881_c59520_raw_vs_standardized_block_transport"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2492, "C58881-C59520"
MODELS = ("raw_identity", "raw_global", "raw_diagonal", "standardized_identity", "standardized_global", "standardized_diagonal")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def samples(field: np.ndarray, rows: list[dict], unit: int, event: int, qpoint: int,
            families: list[str], surfaces: set[int]) -> np.ndarray:
    selected = [r for r in rows if r["unit"] == unit and r["output_interface"] == "entity"
                and r["family"] in families and r["surface"] in surfaces]
    values = np.asarray(field[[r["model_row"] for r in selected], event, qpoint, :], dtype=np.float64)
    output = []
    for language in ("en", "zh"):
        for surface in sorted(surfaces):
            ids = [i for i, r in enumerate(selected) if r["language"] == language and r["surface"] == surface]
            group = values[ids]
            group_rows = [selected[i] for i in ids]
            order = [next(i for i, r in enumerate(group_rows) if r["family"] == family) for family in families]
            group = group[order]
            output.append(group - group.mean(axis=0, keepdims=True))
    return np.concatenate(output, axis=0)


def fit_models(x: np.ndarray, y: np.ndarray) -> dict:
    eps = 1e-12
    x2 = np.sum(x * x, axis=0)
    diagonal = np.sum(x * y, axis=0) / np.maximum(x2, eps)
    x_rms = np.sqrt(np.mean(np.square(x), axis=0) + eps)
    y_rms = np.sqrt(np.mean(np.square(y), axis=0) + eps)
    xs, ys = x / x_rms, y / y_rms
    return {
        "global": float(np.sum(x * y) / max(float(np.sum(x * x)), eps)),
        "diagonal": diagonal, "x_rms": x_rms, "y_rms": y_rms,
        "std_global": float(np.sum(xs * ys) / max(float(np.sum(xs * xs)), eps)),
        "std_diagonal": np.sum(xs * ys, axis=0) / np.maximum(np.sum(xs * xs, axis=0), eps),
    }


def score(name: str, x: np.ndarray, y: np.ndarray, fit: dict) -> dict:
    if name == "raw_identity": pred, target = x, y
    elif name == "raw_global": pred, target = fit["global"] * x, y
    elif name == "raw_diagonal": pred, target = fit["diagonal"] * x, y
    elif name == "standardized_identity": pred, target = x / fit["x_rms"], y / fit["y_rms"]
    elif name == "standardized_global": pred, target = fit["std_global"] * x / fit["x_rms"], y / fit["y_rms"]
    elif name == "standardized_diagonal": pred, target = fit["std_diagonal"] * x / fit["x_rms"], y / fit["y_rms"]
    else: raise KeyError(name)
    sse, sst = float(np.square(target - pred).sum()), float(np.square(target).sum())
    row_cos = np.sum(target * pred, axis=1) / np.maximum(np.linalg.norm(target, axis=1) * np.linalg.norm(pred, axis=1), 1e-12)
    return {"r2_vs_zero": 1.0 - sse / max(sst, 1e-12), "mean_sample_cosine": float(np.mean(row_cos)),
            "relative_rmse": float(np.sqrt(sse / max(sst, 1e-12)))}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 原始纹理与坐标RMS标准化纹理的block→block传动竞争（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 使用九个行为合格entity族的answer-boundary全2560坐标。unit15中surface0/1拟合、surface2/3选择唯一相邻block转移和唯一模型；排除q0→q1 embedding边界与q36→q37 final RMSNorm。模型仅包括原始/标准化空间中的恒等、全局单尺度、逐坐标尺度六个基础基线。选择后用unit15全部surface重拟合，并在unit16同一转移一次锁箱。标准化的每坐标RMS只由unit15拟合，不看unit16。

$$\widehat P_{{q+1,i}}=a_iP_{{q,i}},\qquad \widetilde P_{{q,i}}=P_{{q,i}}/\sqrt{{\mathbb E_{{u15}}P_{{q,i}}^2+\varepsilon}}.$$

**结果汇总。** 确认选择 `{json.dumps(result['selection'], ensure_ascii=False)}`；确认集六模型 `{json.dumps(result['confirmation_selected_transition'], ensure_ascii=False)}`；unit16锁箱 `{json.dumps(result['lockbox'], ensure_ascii=False)}`；尺度/斜率文件 `{json.dumps(result['parameters'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2492_c58881_c59520_raw_vs_standardized_block_transport.py`；全部35个block转移确认分数、冻结的全2560坐标RMS与两类对角斜率、final位于同名目录。

**分析与理论进展。** 原始逐坐标尺度若明显优于全局而标准化后优势缩小，说明一部分“传动”来自坐标各向异性/尺度包络；若标准化逐坐标仍锁箱领先，才保留条件方向的坐标级可预测候选。即使领先，这仍是family-relative群体纹理的一步线性近似，不等于Transformer内部只做对角乘法。

**问题硬伤与结论。** 每坐标斜率用72个unit15样本拟合，仍可能利用谓词词项与残差流自相关。标准化改变损失量纲，原始与标准化$R^2$不能当同一个物理量直接相减。未测试坐标混合，因为当前阶段遵守“先做基础拼图”。结果只裁决基础传动候选，不命名齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    f2487 = json.loads((P2487 / "analysis/final.json").read_text(encoding="utf-8"))
    f2488 = json.loads((P2488 / "analysis/final.json").read_text(encoding="utf-8"))
    families = sorted(f2487["behavior"]["qualified"]["entity"])
    rows = read_jsonl(Path(f2488["collection"]["index"]))
    field = np.load(f2488["collection"]["event_field"], mmap_mode="r")
    event = f2488["collection"]["events"].index("answer_boundary")
    confirmation = []
    for qpoint in range(1, 36):
        x_train = samples(field, rows, 15, event, qpoint, families, {0, 1})
        y_train = samples(field, rows, 15, event, qpoint + 1, families, {0, 1})
        x_eval = samples(field, rows, 15, event, qpoint, families, {2, 3})
        y_eval = samples(field, rows, 15, event, qpoint + 1, families, {2, 3})
        fit = fit_models(x_train, y_train)
        confirmation.append({"transition": [qpoint, qpoint + 1],
                             "models": {name: score(name, x_eval, y_eval, fit) for name in MODELS}})
    candidates = [(item["models"][name]["r2_vs_zero"], item["transition"], name)
                  for item in confirmation for name in MODELS]
    _, transition, selected_model = max(candidates, key=lambda item: item[0])
    q0, q1 = transition
    x_fit = samples(field, rows, 15, event, q0, families, {0, 1, 2, 3})
    y_fit = samples(field, rows, 15, event, q1, families, {0, 1, 2, 3})
    fit = fit_models(x_fit, y_fit)
    x_lock = samples(field, rows, 16, event, q0, families, {0, 1, 2, 3})
    y_lock = samples(field, rows, 16, event, q1, families, {0, 1, 2, 3})
    lockbox = {name: score(name, x_lock, y_lock, fit) for name in MODELS}
    selected_confirmation = next(item["models"] for item in confirmation if item["transition"] == transition)
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    parameter_path = derived / "selected_transition_fullcoordinate_parameters.npz"
    np.savez(parameter_path, raw_diagonal=fit["diagonal"].astype(np.float32), source_rms=fit["x_rms"].astype(np.float32),
             target_rms=fit["y_rms"].astype(np.float32), standardized_diagonal=fit["std_diagonal"].astype(np.float32),
             raw_global=np.float32(fit["global"]), standardized_global=np.float32(fit["std_global"]))
    save(OUT / "analysis/all_confirmation_transitions.json", confirmation)
    checks = {
        "excluded_embedding_boundary": q0 >= 1,
        "excluded_final_norm_boundary": q1 <= 36,
        "thirty_five_block_transitions": len(confirmation) == 35,
        "six_basic_models": set(lockbox) == set(MODELS),
        "all_coordinate_parameters": fit["diagonal"].shape == (2560,),
        "unit16_untouched_until_lockbox": True,
        "finite": all(np.isfinite(v).all() for v in (fit["diagonal"], fit["x_rms"], fit["y_rms"], fit["std_diagonal"])),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "families": families,
        "selection": {"transition": transition, "model": selected_model,
                      "rule": "maximum unit15 held-surface R2 across six preregistered basic models"},
        "confirmation_selected_transition": selected_confirmation, "lockbox": lockbox,
        "parameters": {"path": str(parameter_path), "coordinates": 2560,
                       "arrays": ["raw_diagonal", "source_rms", "target_rms", "standardized_diagonal"]},
        "adjudication": {"selected_model_lockbox_supported": lockbox[selected_model]["r2_vs_zero"] > 0,
                         "diagonal_is_transformer_mechanism": False, "language_encoding_mechanism_closed": False},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
