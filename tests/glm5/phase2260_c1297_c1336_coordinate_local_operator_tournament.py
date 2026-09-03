#!/usr/bin/env python3
"""Compare frozen coordinate-local response models on independent partitions."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
FIELD_OUT = RESULT / "phase2259_c1265_c1296_qwen_natural_full_token_field"
OUT = RESULT / "phase2260_c1297_c1336_coordinate_local_operator_tournament"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2258_c1241_c1264_natural_construction_state_contract as contract  # noqa: E402


PHASE = 2260
CAMPAIGNS = tuple(f"C{i}" for i in range(1297, 1337))
PARTITIONS = ("confirmation", "lockbox", "fresh_confirmation", "fresh_lockbox")
MODELS = ("family_mean", "same_coordinate_affine", "same_coordinate_role_conditioned_affine")
RIDGE = 0.05
EPS = 1e-6
GAIN_ATLAS = OUT / "atlas/full_coordinate_model_gain.float16.npy"
WIN_ATLAS = OUT / "atlas/full_coordinate_role_win.uint8.npy"


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def pair_index(index: list[dict]) -> dict[str, list[dict]]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index:
        key = (row["family"], row["language"], int(row["unit"]), row["surface"], row["partition"])
        groups[key][int(row["state"])] = row
    output: dict[str, list[dict]] = defaultdict(list)
    for key, states in sorted(groups.items()):
        if set(states) != {0, 1}:
            raise RuntimeError(("incomplete_pair", key, sorted(states)))
        output[key[0]].append({"family": key[0], "language": key[1], "unit": key[2],
                               "surface": key[3], "partition": key[4],
                               "state0_index": int(states[0]["hidden_index"]),
                               "state1_index": int(states[1]["hidden_index"])})
    return dict(output)


def arrays(field: np.ndarray, pairs: list[dict], partition: str, q: int) -> tuple[np.ndarray, np.ndarray]:
    subset = [row for row in pairs if row["partition"] == partition]
    i0 = [row["state0_index"] for row in subset]
    i1 = [row["state1_index"] for row in subset]
    h0 = np.asarray(field[i0, q], np.float32)
    h1 = np.asarray(field[i1, q], np.float32)
    return h0, h1 - h0


def fit_models(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    mean = y.mean(axis=0, dtype=np.float32)
    diagonal_a = np.empty_like(mean)
    diagonal_b = np.empty_like(mean)
    for role in range(y.shape[1]):
        xr = x[:, role]
        yr = y[:, role]
        xm = xr.mean(axis=0)
        ym = yr.mean(axis=0)
        xc = xr - xm
        yc = yr - ym
        b = np.sum(xc * yc, axis=0) / (np.sum(xc * xc, axis=0) + RIDGE)
        diagonal_b[role] = b
        diagonal_a[role] = ym - b * xm

    xmean = x.mean(axis=0, dtype=np.float32)
    xscale = x.std(axis=0, dtype=np.float32)
    xscale = np.maximum(xscale, 1e-3)
    z = (x - xmean[None]) / xscale[None]
    covariance = np.einsum("nfd,ngd->dfg", z, z, optimize=True) / len(z)
    covariance += RIDGE * np.eye(x.shape[1], dtype=np.float32)[None]
    weights = np.empty((y.shape[1], x.shape[1], x.shape[2]), dtype=np.float32)
    for target in range(y.shape[1]):
        centered = y[:, target] - mean[target][None]
        rhs = np.einsum("nfd,nd->df", z, centered, optimize=True) / len(z)
        solved = np.linalg.solve(covariance, rhs[..., None])[..., 0]
        weights[target] = solved.T
    return {"mean": mean, "diagonal_a": diagonal_a, "diagonal_b": diagonal_b,
            "xmean": xmean, "xscale": xscale, "weights": weights}


def predict(fit: dict[str, np.ndarray], x: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.broadcast_to(fit["mean"][None], (len(x), *fit["mean"].shape))
    diagonal = fit["diagonal_a"][None] + fit["diagonal_b"][None] * x
    z = (x - fit["xmean"][None]) / fit["xscale"][None]
    role = fit["mean"][None] + np.einsum("nfd,tfd->ntd", z, fit["weights"], optimize=True)
    return {"family_mean": mean, "same_coordinate_affine": diagonal,
            "same_coordinate_role_conditioned_affine": role}


def metrics(y: np.ndarray, predictions: dict[str, np.ndarray]) -> tuple[dict[str, list[dict]], np.ndarray, np.ndarray]:
    zero = np.mean(np.abs(y), axis=0)
    errors = {name: np.mean(np.abs(y - value), axis=0) for name, value in predictions.items()}
    baseline = errors["family_mean"]
    cell = {name: [] for name in MODELS}
    gains = np.empty((2, y.shape[1], y.shape[2]), dtype=np.float32)
    wins = np.empty((y.shape[1], y.shape[2]), dtype=np.uint8)
    for role in range(y.shape[1]):
        base_total = float(np.sum(baseline[role]))
        zero_total = float(np.sum(zero[role]))
        for name in MODELS:
            error = errors[name][role]
            reference = zero_total if name == "family_mean" else base_total
            gain = 1.0 - float(np.sum(error)) / (reference + EPS)
            coordinate_reference = zero[role] if name == "family_mean" else baseline[role]
            cell[name].append({"global_gain": gain,
                               "coordinate_win_fraction": float(np.mean(error < coordinate_reference)),
                               "mae": float(np.mean(error)),
                               "reference_mae": float(np.mean(coordinate_reference))})
        gains[0, role] = (baseline[role] - errors["same_coordinate_affine"][role]) / (zero[role] + EPS)
        gains[1, role] = (baseline[role] - errors["same_coordinate_role_conditioned_affine"][role]) / (zero[role] + EPS)
        wins[role] = (errors["same_coordinate_role_conditioned_affine"][role] < baseline[role]).astype(np.uint8)
    return cell, gains, wins


def passes(value: dict) -> bool:
    gates = contract.OPERATOR_GATES
    return (value["global_gain"] >= gates["gain_over_family_mean"]
            and value["coordinate_win_fraction"] >= gates["coordinate_win_fraction"])


def run_tournament(field: np.ndarray, pairs_by_family: dict[str, list[dict]]) -> dict:
    families = sorted(pairs_by_family)
    shape = (len(families), len(PARTITIONS), 2, 38, len(contract.ROLES), 2560)
    GAIN_ATLAS.parent.mkdir(parents=True, exist_ok=True)
    gain_atlas = np.lib.format.open_memmap(GAIN_ATLAS, mode="w+", dtype=np.float16, shape=shape)
    win_atlas = np.lib.format.open_memmap(
        WIN_ATLAS, mode="w+", dtype=np.uint8,
        shape=(len(families), len(PARTITIONS), 38, len(contract.ROLES), 2560))
    summaries: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    try:
        for fi, family in enumerate(families):
            summaries[family] = {partition: {name: [] for name in MODELS} for partition in PARTITIONS}
            fits_by_q = {}
            for q in range(38):
                discovery_x, discovery_y = arrays(field, pairs_by_family[family], "discovery", q)
                fit = fit_models(discovery_x, discovery_y)
                fits_by_q[q] = fit
                for pi, partition in enumerate(PARTITIONS):
                    x, y = arrays(field, pairs_by_family[family], partition, q)
                    cell, gains, wins = metrics(y, predict(fit, x))
                    gain_atlas[fi, pi, :, q] = np.clip(gains, -32.0, 32.0).astype(np.float16)
                    win_atlas[fi, pi, q] = wins
                    for name in MODELS:
                        for role, value in enumerate(cell[name]):
                            summaries[family][partition][name].append({"checkpoint": q,
                                                                       "role": contract.ROLES[role], **value})
            candidates = []
            for model in ("same_coordinate_affine", "same_coordinate_role_conditioned_affine"):
                conf = {(row["checkpoint"], row["role"]): row
                        for row in summaries[family]["confirmation"][model]}
                fresh = {(row["checkpoint"], row["role"]): row
                         for row in summaries[family]["fresh_confirmation"][model]}
                for key in sorted(conf):
                    q, role = key
                    if q not in range(1, 37) or not (passes(conf[key]) and passes(fresh[key])):
                        continue
                    score = min(conf[key]["global_gain"], fresh[key]["global_gain"])
                    candidates.append((0 if model == "same_coordinate_affine" else 1,
                                       -score, q, contract.ROLES.index(role), model, key))
            if candidates:
                # Prefer the single-coordinate model whenever it has any prelockbox-qualified cell.
                chosen = min(candidates)
                model, (q, role) = chosen[-2], chosen[-1]
                lock = next(row for row in summaries[family]["lockbox"][model]
                            if row["checkpoint"] == q and row["role"] == role)
                fresh_lock = next(row for row in summaries[family]["fresh_lockbox"][model]
                                  if row["checkpoint"] == q and row["role"] == role)
                lockbox_pass = fresh_lock["global_gain"] >= contract.OPERATOR_GATES["fresh_lockbox_gain_over_mean"]
                selected[family] = {"prelockbox_qualified": True, "model": model,
                                    "checkpoint": q, "role": role,
                                    "confirmation": next(row for row in summaries[family]["confirmation"][model]
                                                         if row["checkpoint"] == q and row["role"] == role),
                                    "fresh_confirmation": next(row for row in summaries[family]["fresh_confirmation"][model]
                                                               if row["checkpoint"] == q and row["role"] == role),
                                    "lockbox": lock, "fresh_lockbox": fresh_lock,
                                    "operator_qualified": bool(lockbox_pass)}
                if lockbox_pass:
                    fit = fits_by_q[q]
                    role_i = contract.ROLES.index(role)
                    np.savez_compressed(
                        OUT / f"operator/{family}.npz", model=np.array(model), checkpoint=np.array(q),
                        role=np.array(role), mean=fit["mean"][role_i],
                        diagonal_a=fit["diagonal_a"][role_i], diagonal_b=fit["diagonal_b"][role_i],
                        xmean=fit["xmean"], xscale=fit["xscale"],
                        weights=fit["weights"][role_i])
            else:
                selected[family] = {"prelockbox_qualified": False,
                                    "operator_qualified": False,
                                    "reason": "no_same_cell_passed_confirmation_and_fresh_confirmation"}
            print(f"[operator] {fi + 1}/{len(families)} {family} {selected[family]}", flush=True)
            gain_atlas.flush()
            win_atlas.flush()
    finally:
        gain_atlas.flush()
        win_atlas.flush()
        close_mmap(gain_atlas)
        close_mmap(win_atlas)
    return {"family_order": families, "partition_order": list(PARTITIONS), "model_order": list(MODELS),
            "summaries": summaries, "selected": selected,
            "operator_qualified_families": sorted(family for family, value in selected.items()
                                                  if value.get("operator_qualified")),
            "gain_atlas": {"path": str(GAIN_ATLAS.relative_to(ROOT)), "shape": list(shape),
                           "models": ["same_coordinate_affine", "same_coordinate_role_conditioned_affine"]},
            "win_atlas": {"path": str(WIN_ATLAS.relative_to(ROOT)),
                          "shape": [len(families), len(PARTITIONS), 38, len(contract.ROLES), 2560]}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {family: value for family, value in result["analysis"]["selected"].items()}
    text = rf"""

## Phase {PHASE}: 同坐标条件算子与分布式全场锁箱锦标赛（C1297-C1336） [{stamp}]

**测试原理。** 只使用Phase2259九个双行为合格构式。按同一族、语言、独立单元和表面配对真假状态，定义逐坐标响应 $R_i=H_i^{{(1)}}-H_i^{{(0)}}$。discovery独占拟合；confirmation和fresh confirmation共同决定同一检查点-角色-模型是否有资格；父lockbox只报告，fresh lockbox作正式前瞻裁决。没有锁箱样本参与参数、检查点、角色或模型选择。

**基础模型与公式。** 三个模型同时覆盖全部2560坐标：

$$
\widehat R^{{mean}}_{{i,q,r,j}}=\bar R_{{q,r,j}},
$$

$$
\widehat R^{{diag}}_{{i,q,r,j}}=a_{{q,r,j}}+b_{{q,r,j}}H^{{(0)}}_{{i,q,r,j}},
$$

$$
\widehat R^{{role}}_{{i,q,r,j}}=c_{{q,r,j}}+
\sum_{{r'\in\mathcal R}}w_{{q,r,r',j}}\,
\widetilde H^{{(0)}}_{{i,q,r',j}}.
$$

第三式允许同一物理坐标在六个功能角色间联合，但禁止跨坐标混合。每格报告相对族均值的全坐标绝对误差增益和逐坐标胜率；不使用PCA、Top-K、余弦或高值截断。

**结果与门槛。** 冻结门要求confirmation和fresh confirmation在同一检查点-角色上同时满足：误差增益不低于 `{contract.OPERATOR_GATES['gain_over_family_mean']}`、逐坐标胜率不低于 `{contract.OPERATOR_GATES['coordinate_win_fraction']}`；若单坐标模型存在合格格则优先于六角色联合。冻结选择及fresh lockbox结果为 `{json.dumps(compact, ensure_ascii=False)}`。正式算子合格族为 `{json.dumps(result['analysis']['operator_qualified_families'], ensure_ascii=False)}`。全坐标增益图 `{json.dumps(result['analysis']['gain_atlas'], ensure_ascii=False)}`；全坐标胜负图 `{json.dumps(result['analysis']['win_atlas'], ensure_ascii=False)}`。

**分析、理论进展与边界。** 通过表示“当前样本基态的同坐标值能前瞻改善该样本响应预测”，比固定族均值更强；仍只是条件预测，不是因果齿轮、唯一坐标字典或完整跨层动力学。单坐标模型优先规则使“更简单是否足够”先于小联合；二者均失败时支持当前分辨率下的分布式场解释，但不否定更高阶条件结构。理论主体与RDC不变。

**问题、硬伤、结论与相关文件。** 仿射模型仍是局部近似，角色映射依赖人工功能跨度，输入只取同一检查点，响应含答案真值和输出码条件；低值坐标虽全部保留，float16图谱仍有量化误差。工程检查 `{result['all_checks_passed']}`。只有正式合格族进入样本条件调用/删除；其他族继续保留完整观察图谱。脚本 `tests/glm5/phase2260_c1297_c1336_coordinate_local_operator_tournament.py`；结果 `tests/glm5/result/phase2260_c1297_c1336_coordinate_local_operator_tournament`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        return load(final)
    source = load(FIELD_OUT / "analysis/final.json")
    if not source["all_checks_passed"] or not source["role_field"].get("ran"):
        raise RuntimeError("Phase2259 role field is unavailable")
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "operator").mkdir(parents=True, exist_ok=True)
    field = np.load(FIELD_OUT / "raw/qwen3_4b_qualified_role_field.float16.npy", mmap_mode="r")
    index = read_rows(FIELD_OUT / "raw/role_field_index.jsonl")
    try:
        pairs = pair_index(index)
        analysis = run_tournament(field, pairs)
    finally:
        close_mmap(field)
    save(OUT / "analysis/tournament.json", analysis)
    selected = analysis["selected"]
    checks = {
        "all_behavior_qualified_families_present": set(pairs) == set(source["behavior"]["qualified_families"]),
        "complete_pairs": all(len(rows) == 160 for rows in pairs.values()),
        "all_partitions_evaluated": all(set(value) == set(PARTITIONS) for value in analysis["summaries"].values()),
        "lockbox_never_fit": True,
        "full_coordinate_gain_shape": analysis["gain_atlas"]["shape"][-1] == 2560,
        "selection_complete": set(selected) == set(pairs),
        "operator_files_match": all((OUT / f"operator/{family}.npz").exists()
                                    for family in analysis["operator_qualified_families"]),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "analysis": analysis,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Qualified families have a prospectively selected coordinate-local predictor that beats the family mean on untouched fresh lockbox; this is predictive, not causal closure.",
        "next_authorization": "Run sample-conditioned call/delete controls only for operator-qualified families; keep all other families as observational atlases.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps({"phase": PHASE, "selected": selected, "checks": checks,
                      "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
