#!/usr/bin/env python3
"""Basic all-coordinate competition: persistence, density, and diagonal layer transport."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2471 = next(RESULT.glob("phase2471_*"))
OUT = RESULT / "phase2472_c46481_c47040_coordinate_transport_competition"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2472, "C46481-C47040", 2560


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(-1, DIM)
    y = np.asarray(b, dtype=np.float64).reshape(-1, DIM)
    denominator = np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    return np.sum(x * y, axis=1) / denominator


def fit_scale(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.sum(np.asarray(x, dtype=np.float64) * np.asarray(y, dtype=np.float64), axis=0)
    denominator = np.sum(np.square(np.asarray(x, dtype=np.float64)), axis=0)
    return (numerator / np.maximum(denominator, 1e-20)).astype(np.float32)


def evaluate(prediction: np.ndarray, actual: np.ndarray) -> dict:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(actual, dtype=np.float64)
    sse = float(np.sum(np.square(truth - pred)))
    zero = float(np.sum(np.square(truth)))
    return {
        "r2_vs_zero": 1.0 - sse / zero if zero > 0 else 0.0,
        "mean_sample_cosine": float(np.mean(cosine_rows(pred, truth))),
        "relative_rmse": math.sqrt(sse / max(zero, 1e-30)),
    }


def flatten_condition(array: np.ndarray) -> np.ndarray:
    # language, interface, family, coordinate -> samples, coordinate
    return np.asarray(array, dtype=np.float32).reshape(-1, DIM)


def train_predictors(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    pooled_scale = fit_scale(flatten_condition(x), flatten_condition(y))
    global_scale = float(np.sum(np.asarray(x, dtype=np.float64) * np.asarray(y, dtype=np.float64)) / max(np.sum(np.square(np.asarray(x, dtype=np.float64))), 1e-30))
    interface_scale = np.stack([fit_scale(x[:, interface].reshape(-1, DIM), y[:, interface].reshape(-1, DIM)) for interface in range(2)])
    family_scale = np.stack([fit_scale(x[:, :, family].reshape(-1, DIM), y[:, :, family].reshape(-1, DIM)) for family in range(8)])
    return {"pooled": pooled_scale, "global": global_scale, "interface": interface_scale, "family": family_scale}


def predict(x: np.ndarray, fitted: dict, kind: str) -> np.ndarray:
    if kind == "identity":
        return np.asarray(x, dtype=np.float32)
    if kind == "global":
        return np.asarray(x, dtype=np.float32) * fitted["global"]
    if kind == "pooled_diagonal":
        return np.asarray(x, dtype=np.float32) * fitted["pooled"]
    if kind == "interface_diagonal":
        return np.asarray(x, dtype=np.float32) * fitted["interface"][None, :, None, :]
    if kind == "family_diagonal":
        return np.asarray(x, dtype=np.float32) * fitted["family"][None, None, :, :]
    raise KeyError(kind)


def family_identity(prediction: np.ndarray, actual: np.ndarray, permutations: np.ndarray) -> dict:
    # language, interface, family, coordinate
    values = []
    nulls = []
    for language in range(2):
        for interface in range(2):
            p = np.asarray(prediction[language, interface], dtype=np.float64)
            a = np.asarray(actual[language, interface], dtype=np.float64)
            p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-30)
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-30)
            sim = p @ a.T
            values.append(float(np.mean(np.diag(sim))))
            nulls.extend(np.mean(sim[np.arange(8)[None, :], permutations], axis=1).tolist())
    q95 = float(np.quantile(nulls, 0.95))
    matched = float(np.mean(values))
    return {"coordinate": matched, "family_null_q95": q95, "family_identity_advantage": matched - q95}


def transport_competition(contrasts: np.ndarray) -> dict:
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2472)
    permutations = []
    while len(permutations) < 256:
        p = rng.permutation(8)
        if np.all(p != np.arange(8)):
            permutations.append(p)
    permutations = np.stack(permutations)
    transitions = {}
    model_kinds = ("identity", "global", "pooled_diagonal", "interface_diagonal", "family_diagonal")
    for qpoint in range(37):
        # Unit9: English is discovery fit, Chinese is confirmation selection.
        train_x = contrasts[0, 0:1, :, :, 2, qpoint]
        train_y = contrasts[0, 0:1, :, :, 2, qpoint + 1]
        fitted_discovery = train_predictors(train_x, train_y)
        confirmation_x = contrasts[0, 1:2, :, :, 2, qpoint]
        confirmation_y = contrasts[0, 1:2, :, :, 2, qpoint + 1]
        confirmation = {kind: evaluate(predict(confirmation_x, fitted_discovery, kind), confirmation_y) for kind in model_kinds}
        transitions[f"q{qpoint}_to_q{qpoint+1}"] = {"confirmation_unit9_zh": confirmation}
    # q0->q1 crosses embedding/block semantics and q36->q37 is the known final
    # RMSNorm. Neither is eligible as evidence for a learned block-to-block rule.
    selected = max(range(1, 36), key=lambda q: transitions[f"q{q}_to_q{q+1}"]["confirmation_unit9_zh"]["pooled_diagonal"]["r2_vs_zero"] - transitions[f"q{q}_to_q{q+1}"]["confirmation_unit9_zh"]["identity"]["r2_vs_zero"])
    # Refit the frozen predictor classes on both unit9 languages, then reveal unit10.
    x9 = contrasts[0, :, :, :, 2, selected]
    y9 = contrasts[0, :, :, :, 2, selected + 1]
    fitted = train_predictors(x9, y9)
    x10 = contrasts[1, :, :, :, 2, selected]
    y10 = contrasts[1, :, :, :, 2, selected + 1]
    lockbox = {}
    for kind in model_kinds:
        pred = predict(x10, fitted, kind)
        lockbox[kind] = {**evaluate(pred, y10), "family_identity": family_identity(pred, y10, permutations)}
    np.save(OUT / "derived/frozen_pooled_diagonal_scale.float32.npy", fitted["pooled"])
    np.save(OUT / "derived/frozen_interface_diagonal_scale.float32.npy", fitted["interface"])
    np.save(OUT / "derived/frozen_family_diagonal_scale.float32.npy", fitted["family"])
    return {
        "selection_rule": "fit unit9-en; among block-to-block q1->q2 through q35->q36 only, select by pooled-diagonal R2 gain over identity on unit9-zh; exclude embedding and final RMSNorm boundaries",
        "selected_transition": [selected, selected + 1],
        "selected_confirmation": transitions[f"q{selected}_to_q{selected+1}"]["confirmation_unit9_zh"],
        "lockbox_unit10": lockbox,
        "all_confirmation_transitions": transitions,
    }


def density_audit(contrasts: np.ndarray, selected_qpoint: int) -> dict:
    discovery = np.mean(np.square(contrasts[0, :, :, :, 2, selected_qpoint], dtype=np.float64), axis=(0, 1, 2))
    lockbox = np.mean(np.square(contrasts[1, :, :, :, 2, selected_qpoint], dtype=np.float64), axis=(0, 1, 2))
    def participation(energy: np.ndarray) -> dict:
        effective = float(np.square(np.sum(energy)) / max(np.sum(np.square(energy)), 1e-30))
        ordered = np.sort(energy)[::-1]
        cumulative = np.cumsum(ordered) / max(np.sum(ordered), 1e-30)
        return {
            "effective_coordinate_count": effective,
            "effective_fraction": effective / DIM,
            "coordinates_for_50pct_energy": int(np.searchsorted(cumulative, 0.5) + 1),
            "coordinates_for_90pct_energy": int(np.searchsorted(cumulative, 0.9) + 1),
        }
    groups_discovery = np.asarray([np.sum(discovery[np.arange(DIM) % 16 == group]) for group in range(16)])
    groups_lockbox = np.asarray([np.sum(lockbox[np.arange(DIM) % 16 == group]) for group in range(16)])
    return {
        "discovery": participation(discovery),
        "lockbox": participation(lockbox),
        "mod16_group_energy_correlation": float(np.corrcoef(groups_discovery, groups_lockbox)[0, 1]),
        "interpretation_boundary": "Energy concentration is descriptive only; no coordinates are selected or deleted for the main analysis.",
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 稀疏/稠密、条件化与跨层轨迹的基本全坐标竞争（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 使用Phase2471冻结八族answer-boundary全坐标contrast，让四类最基本对象竞争：原坐标恒等保持、全局单尺度、每坐标独立尺度、接口条件每坐标尺度、family条件每坐标尺度。unit9英文只拟合，unit9中文按“pooled diagonal相对identity的$R^2$增益”选择唯一层转移；随后用unit9双语重拟合并一次揭示unit10。另用全部坐标能量参与率审计纹理是集中还是分布，不把能量排序用于主分析或干预。

$$\widehat P_{{\ell+1,i}}=a_{{\ell,i}}P_{{\ell,i}},\qquad R^2=1-\mathrm{{SSE}}/\mathrm{{SST}}.$$

**结果汇总。** 发现/确认选择 `{json.dumps(result['transport']['selected_transition'], ensure_ascii=False)}`、`{json.dumps(result['transport']['selected_confirmation'], ensure_ascii=False)}`；unit10锁箱 `{json.dumps(result['transport']['lockbox_unit10'], ensure_ascii=False)}`；全坐标密度 `{json.dumps(result['density'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2472_c46481_c47040_coordinate_transport_competition.py`；冻结pooled/interface/family逐坐标尺度、全部37个确认转移和final位于同名结果目录。

**分析与理论进展。** 这是一种极受限的“传动规则”测试：若unit10逐坐标预测优于恒等和单尺度，并保留family身份，它支持可预测跨层纹理，但不意味着层内部真的执行对角算子。接口条件模型只有在锁箱优于pooled时，才支持输出接口改变传输规则；family条件模型样本少，只作反证或上界诊断。

**问题硬伤与结论。** 相邻层高度相关会让恒等基线很强；逐坐标尺度可能拟合数值尺度而不是语义计算。family条件每族训练样本很少，不能因高分命名为齿轮。能量参与率不等于因果参与率。所有结论仍是L1/L2候选，不能替代自然状态因果与自主生成。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final2471 = json.loads((P2471 / "analysis/final.json").read_text(encoding="utf-8"))
    contrasts = np.load(final2471["collection"]["family_contrasts"]["path"], mmap_mode="r")
    transport = transport_competition(contrasts)
    density = density_audit(contrasts, final2471["replication"]["discovery_selection"]["raw_qpoint"])
    lockbox = transport["lockbox_unit10"]
    pooled = lockbox["pooled_diagonal"]
    identity = lockbox["identity"]
    interface = lockbox["interface_diagonal"]
    adjudication = {
        "distributed_fullfield_candidate": density["lockbox"]["effective_fraction"] >= 0.1,
        "predictable_diagonal_transport_candidate": pooled["r2_vs_zero"] > identity["r2_vs_zero"] and pooled["family_identity"]["family_identity_advantage"] > 0,
        "interface_conditioned_transport_supported": interface["r2_vs_zero"] > pooled["r2_vs_zero"],
        "fixed_sparse_gear_identified": False,
        "natural_coordinate_gear_identified": False,
        "language_encoding_mechanism_closed": False,
    }
    checks = {
        "selected_one_transition": len(transport["selected_transition"]) == 2,
        "all_model_classes": set(lockbox) == {"identity", "global", "pooled_diagonal", "interface_diagonal", "family_diagonal"},
        "fullcoordinate_scales": np.load(OUT / "derived/frozen_pooled_diagonal_scale.float32.npy", mmap_mode="r").shape == (DIM,),
        "finite": all(math.isfinite(value) for model in lockbox.values() for key, value in model.items() if key != "family_identity"),
        "claim_boundary": not adjudication["language_encoding_mechanism_closed"] and not adjudication["natural_coordinate_gear_identified"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "transport": transport, "density": density, "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    close(contrasts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
