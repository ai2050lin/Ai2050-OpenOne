#!/usr/bin/env python3
"""Full-coordinate basic structure tournament for Phase 2276."""
from __future__ import annotations

import hashlib
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
FIELD_OUT = RESULT / "phase2275_c1771_c1820_qwen4b_broad_fullfield"
OUT = RESULT / "phase2276_c1821_c1890_full_coordinate_structure_tournament"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2274_c1721_c1770_broad_construction_contract as contract  # noqa: E402


PHASE = 2276
CAMPAIGN = "C1821-C1890"
FIELD = FIELD_OUT / "raw/qwen3_4b_broad_role_field.float16.npy"
INDEX = FIELD_OUT / "raw/role_field_index.jsonl"
ROLES = contract.ROLES
CANDIDATES = ("own_affine", "piecewise_quartile", "sign_interval")
CONTROLS = (
    "family_mean", "algebraic", "shared_affine", "wrong_family_affine",
    "shuffled_affine", "previous_checkpoint_affine", "surface_mean", "output_scheme_mean",
)
GATES = {
    "gain_over_family_mean": 0.03,
    "coordinate_win_over_family_mean": 0.55,
    "gain_over_each_control": 0.01,
    "coordinate_win_over_each_control": 0.52,
}
EPS = 1e-8


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


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


def pair_rows(index: list[dict]) -> list[dict]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index:
        key = (row["family"], int(row["unit"]), row["surface"], row["partition"])
        groups[key][int(row["state"])] = row
    pairs = []
    for key, states in sorted(groups.items()):
        if set(states) != {0, 1}:
            raise RuntimeError(("incomplete_state_pair", key))
        if int(states[0]["output_scheme"]) != int(states[1]["output_scheme"]):
            raise RuntimeError(("output_scheme_changes_with_state", key))
        pairs.append({
            "family": key[0], "unit": key[1], "surface": key[2], "partition": key[3],
            "output_scheme": int(states[0]["output_scheme"]),
            "state0_index": int(states[0]["hidden_index"]),
            "state1_index": int(states[1]["hidden_index"]),
        })
    return pairs


def arrays(field: np.ndarray, pairs: list[dict], q: int, role_i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i0 = np.asarray([row["state0_index"] for row in pairs], dtype=np.int64)
    i1 = np.asarray([row["state1_index"] for row in pairs], dtype=np.int64)
    h0 = np.asarray(field[i0, q, role_i], dtype=np.float32)
    h1 = np.asarray(field[i1, q, role_i], dtype=np.float32)
    return h0, h1, h1 - h0


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    centered = x - xm
    variance = np.mean(centered * centered, axis=0)
    covariance = np.mean(centered * (y - ym), axis=0)
    a = covariance / (variance + 1e-6)
    b = ym - a * xm
    return a.astype(np.float32), b.astype(np.float32)


def fit_piecewise(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.quantile(x, (0.25, 0.5, 0.75), axis=0).astype(np.float32)
    bins = (x > thresholds[0]).astype(np.int8) + (x > thresholds[1]).astype(np.int8) + (x > thresholds[2]).astype(np.int8)
    fallback = y.mean(axis=0)
    means = np.empty((4, x.shape[1]), dtype=np.float32)
    for value in range(4):
        mask = bins == value
        count = mask.sum(axis=0)
        means[value] = np.where(count > 0, (y * mask).sum(axis=0) / np.maximum(count, 1), fallback)
    return thresholds, means


def predict_piecewise(model: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    thresholds, means = model
    bins = ((x > thresholds[0]).astype(np.int8) + (x > thresholds[1]).astype(np.int8) +
            (x > thresholds[2]).astype(np.int8))
    return np.take_along_axis(means[None, :, :], bins[:, None, :], axis=1)[:, 0, :]


def fit_sign_interval(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.median(np.abs(x), axis=0).astype(np.float32)
    bins = (x >= 0).astype(np.int8) * 2 + (np.abs(x) >= magnitude).astype(np.int8)
    fallback = y.mean(axis=0)
    means = np.empty((4, x.shape[1]), dtype=np.float32)
    for value in range(4):
        mask = bins == value
        count = mask.sum(axis=0)
        means[value] = np.where(count > 0, (y * mask).sum(axis=0) / np.maximum(count, 1), fallback)
    return magnitude, means


def predict_sign_interval(model: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    magnitude, means = model
    bins = (x >= 0).astype(np.int8) * 2 + (np.abs(x) >= magnitude).astype(np.int8)
    return np.take_along_axis(means[None, :, :], bins[:, None, :], axis=1)[:, 0, :]


def fit_family(field: np.ndarray, pairs: list[dict], q: int, role_i: int) -> dict[str, Any]:
    x, h1, y = arrays(field, pairs, q, role_i)
    own_a, own_b = fit_affine(x, y)
    shift = max(1, len(pairs) // 3)
    shuffled_a, shuffled_b = fit_affine(x, np.roll(y, shift=shift, axis=0))
    if q > 0:
        _, _, previous = arrays(field, pairs, q - 1, role_i)
        previous_a, previous_b = fit_affine(previous, y)
    else:
        previous_a = np.zeros(x.shape[1], dtype=np.float32)
        previous_b = y.mean(axis=0).astype(np.float32)
    return {
        "mean": y.mean(axis=0).astype(np.float32),
        "mean_h1": h1.mean(axis=0).astype(np.float32),
        "own_affine": (own_a, own_b),
        "piecewise_quartile": fit_piecewise(x, y),
        "sign_interval": fit_sign_interval(x, y),
        "shuffled_affine": (shuffled_a, shuffled_b),
        "previous_checkpoint_affine": (previous_a, previous_b),
    }


def predict_models(field: np.ndarray, pairs: list[dict], q: int, role_i: int,
                   own: dict[str, Any], wrong: dict[str, Any], shared: tuple[np.ndarray, np.ndarray],
                   surface_means: dict[str, np.ndarray], scheme_means: dict[int, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    x, _h1, y = arrays(field, pairs, q, role_i)
    if q > 0:
        _, _, previous = arrays(field, pairs, q - 1, role_i)
    else:
        previous = np.zeros_like(y)
    surface = np.stack([surface_means[row["surface"]] for row in pairs])
    scheme = np.stack([scheme_means[int(row["output_scheme"])] for row in pairs])
    predictions = {
        "family_mean": np.broadcast_to(own["mean"], y.shape),
        "own_affine": x * own["own_affine"][0] + own["own_affine"][1],
        "piecewise_quartile": predict_piecewise(own["piecewise_quartile"], x),
        "sign_interval": predict_sign_interval(own["sign_interval"], x),
        "algebraic": own["mean_h1"] - x,
        "shared_affine": x * shared[0] + shared[1],
        "wrong_family_affine": x * wrong["own_affine"][0] + wrong["own_affine"][1],
        "shuffled_affine": x * own["shuffled_affine"][0] + own["shuffled_affine"][1],
        "previous_checkpoint_affine": previous * own["previous_checkpoint_affine"][0] + own["previous_checkpoint_affine"][1],
        "surface_mean": surface,
        "output_scheme_mean": scheme,
    }
    errors = {name: np.mean(np.abs(value - y), axis=0).astype(np.float32)
              for name, value in predictions.items()}
    return errors, y


def summarize_errors(errors: dict[str, np.ndarray], family: str, partition: str,
                     q: int, role_i: int) -> list[dict]:
    baseline = errors["family_mean"]
    rows = []
    for name, value in errors.items():
        rows.append({
            "family": family,
            "partition": partition,
            "checkpoint": q,
            "role": ROLES[role_i],
            "role_index": role_i,
            "model": name,
            "mae": float(value.mean()),
            "gain_over_family_mean": float((baseline.mean() - value.mean()) / max(float(baseline.mean()), EPS)),
            "coordinate_win_over_family_mean": float(np.mean(value < baseline)),
        })
    return rows


def candidate_record(errors: dict[str, np.ndarray], name: str, family: str, partition: str,
                     q: int, role_i: int) -> dict:
    value = errors[name]
    baseline = errors["family_mean"]
    control_gains = {
        control: float((errors[control].mean() - value.mean()) / max(float(errors[control].mean()), EPS))
        for control in CONTROLS
    }
    control_wins = {control: float(np.mean(value < errors[control])) for control in CONTROLS}
    record = {
        "family": family, "partition": partition, "checkpoint": q,
        "role": ROLES[role_i], "role_index": role_i, "model": name,
        "mae": float(value.mean()),
        "gain_over_family_mean": float((baseline.mean() - value.mean()) / max(float(baseline.mean()), EPS)),
        "coordinate_win_over_family_mean": float(np.mean(value < baseline)),
        "gain_over_controls": control_gains,
        "coordinate_win_over_controls": control_wins,
        "minimum_control_gain": min(control_gains.values()),
        "minimum_control_win": min(control_wins.values()),
    }
    record["passes"] = bool(
        record["gain_over_family_mean"] >= GATES["gain_over_family_mean"] and
        record["coordinate_win_over_family_mean"] >= GATES["coordinate_win_over_family_mean"] and
        record["minimum_control_gain"] >= GATES["gain_over_each_control"] and
        record["minimum_control_win"] >= GATES["coordinate_win_over_each_control"]
    )
    return record


def models_for_cell(field: np.ndarray, by_family_partition: dict[tuple[str, str], list[dict]],
                    families: list[str], q: int, role_i: int) -> tuple[dict[str, dict], tuple[np.ndarray, np.ndarray], dict[str, np.ndarray], dict[int, np.ndarray]]:
    family_models = {family: fit_family(field, by_family_partition[(family, "discovery")], q, role_i)
                     for family in families}
    all_train = [row for family in families for row in by_family_partition[(family, "discovery")]]
    shared_x, _shared_h1, shared_y = arrays(field, all_train, q, role_i)
    shared = fit_affine(shared_x, shared_y)
    surface_means = {}
    for surface in contract.SURFACES:
        subset = [row for row in all_train if row["surface"] == surface]
        surface_means[surface] = arrays(field, subset, q, role_i)[2].mean(axis=0).astype(np.float32)
    scheme_means = {}
    for scheme in range(len(contract.OUTPUT_SCHEMES)):
        subset = [row for row in all_train if int(row["output_scheme"]) == scheme]
        scheme_means[scheme] = arrays(field, subset, q, role_i)[2].mean(axis=0).astype(np.float32)
    return family_models, shared, surface_means, scheme_means


def evaluate_decision(field: np.ndarray, by_family_partition: dict[tuple[str, str], list[dict]],
                      families: list[str], decision: dict, partition: str) -> tuple[dict, dict[str, np.ndarray], dict]:
    family = decision["family"]
    q, role_i = int(decision["checkpoint"]), int(decision["role_index"])
    family_models, shared, surface_means, scheme_means = models_for_cell(
        field, by_family_partition, families, q, role_i)
    wrong_family = families[(families.index(family) + 1) % len(families)]
    pairs = by_family_partition[(family, partition)]
    errors, _ = predict_models(field, pairs, q, role_i, family_models[family],
                               family_models[wrong_family], shared, surface_means, scheme_means)
    record = candidate_record(errors, decision["model"], family, partition, q, role_i)
    return record, errors, family_models[family]


def append_memo(result: dict) -> None:
    current = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in current:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 全坐标基础结构总竞赛与新词汇锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本期不重新运行模型，而是严格读取 Phase2275 的 13 个行为合格构式、2496 个样本、38 个检查点、六角色和全部 2560 激活坐标。对每个“构式-检查点-角色-坐标”，以 state0 的基态 $H^0$ 预测真假响应 $R=H^1-H^0$。discovery 只用于拟合；confirmation 在同一构式内从 684 个“模型-检查点-角色”候选中冻结唯一最佳项；随后 fresh confirmation 只决定是否授权锁箱；fresh lockbox 最后一次揭示。没有使用错误回答后筛选、Top-K、PCA、余弦或 Attention/MLP。

**基础算法与公式。** 同坐标仿射为：

$$
\widehat R_j=a_jH^0_j+b_j.
$$

分段模型只用 discovery 的逐坐标四分位点冻结四个区间：

$$
\widehat R_j=\mu_{{j,k}},\qquad
k=\sum_{{m=1}}^3\mathbf 1[H^0_j>Q_{{j,m}}].
$$

符号-幅值模型把每个坐标按正负号和是否超过 discovery 中位绝对值分成四格。负控包括族均值、纯代数 $\bar H^1-H^0$、跨族共享仿射、错族仿射、错配仿射、上一检查点响应、纯表面均值和纯输出码均值。通过门要求候选相对族均值误差增益不低于 0.03、逐坐标胜率不低于 0.55，并且相对每个控制的增益不低于 0.01、胜率不低于 0.52。

**结果汇总与锁箱门槛。** confirmation 冻结决策 `{json.dumps(result['confirmation_decisions'], ensure_ascii=False)}`；fresh confirmation `{json.dumps(result['fresh_confirmation'], ensure_ascii=False)}`；锁箱授权族 `{json.dumps(result['lockbox_authorized_families'], ensure_ascii=False)}`；fresh lockbox `{json.dumps(result['fresh_lockbox'], ensure_ascii=False)}`；严格锁箱通过族 `{json.dumps(result['lockbox_passed_families'], ensure_ascii=False)}`。完整检查点-角色得分矩阵 `{json.dumps(result['score_atlas'], ensure_ascii=False)}`；逐坐标护照 `{json.dumps(result['coordinate_passport'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；工程检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析与理论进展。** `{result['strict_conclusion']}` 通过只授权“模型本地、构式/角色/深度条件化的逐坐标响应可预测结构”，不授权单坐标语义、唯一因果电路或新数学定理。分段或符号模型若胜出，说明线性仿射不是最佳基础描述；仿射若胜出，也只说明当前采样范围内一阶近似更有效。上一检查点、表面和输出码控制用于防止把一般传播、长度或答案编译误命名为语义齿轮。

**问题、硬伤与瓶颈。** 每个构式 discovery 只有 36 个状态对，逐坐标分段格可能稀疏；四分位和符号格仍是研究者选择的简单刻度；13×38×6 的选择存在多候选问题，虽然锁箱独立但不能消除所有选择偏差；所有坐标是 float16 激活；同坐标模型忽略跨坐标联合；英语模板和元语言输出仍限制自然语言外推。

**结论与下一步。** 下一阶段不搬运整条预测差分，而对本期前瞻锁箱通过的属性锚点和一个非属性对照使用平衡随机坐标掩码，直接估计哪些坐标组合改变输出 margin，并用独立掩码与新词汇复验。脚本 `tests/glm5/phase2276_c1821_c1890_full_coordinate_structure_tournament.py`；结果 `tests/glm5/result/phase2276_c1821_c1890_full_coordinate_structure_tournament`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    for sub in ("protocol", "analysis", "atlas"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    field_result = load(FIELD_OUT / "analysis/final.json")
    if not field_result["all_checks_passed"] or not FIELD.exists():
        raise RuntimeError("Phase2275 field is unavailable")
    families = list(field_result["behavior"]["qualified_families"])
    index = read_rows(INDEX)
    pairs = pair_rows(index)
    by_family_partition: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pairs:
        by_family_partition[(row["family"], row["partition"])].append(row)
    expected = {"discovery": 36, "confirmation": 12, "fresh_confirmation": 24, "fresh_lockbox": 24}
    pair_counts_ok = all(len(by_family_partition[(family, partition)]) == count
                         for family in families for partition, count in expected.items())
    prereg = {
        "phase": PHASE, "campaign": CAMPAIGN, "frozen_before_analysis": True,
        "families": families, "roles": list(ROLES), "checkpoints": 38, "coordinates": 2560,
        "candidate_models": list(CANDIDATES), "controls": list(CONTROLS), "gates": GATES,
        "selection": "one confirmation-best passing model/checkpoint/role per family; fresh confirmation only authorizes lockbox",
        "forbidden": ["PCA", "Top-K", "cosine", "attention", "MLP", "gradient", "lockbox reselection"],
    }
    save(OUT / "protocol/preregistration.json", prereg)
    field = np.load(FIELD, mmap_mode="r")
    summaries: list[dict] = []
    candidate_rows: list[dict] = []
    best: dict[str, dict] = {}
    score = np.full((len(families), 38, len(ROLES)), np.nan, dtype=np.float32)
    try:
        for q in range(38):
            for role_i in range(len(ROLES)):
                family_models, shared, surface_means, scheme_means = models_for_cell(
                    field, by_family_partition, families, q, role_i)
                for fi, family in enumerate(families):
                    wrong_family = families[(fi + 1) % len(families)]
                    test = by_family_partition[(family, "confirmation")]
                    errors, _ = predict_models(field, test, q, role_i, family_models[family],
                                               family_models[wrong_family], shared, surface_means, scheme_means)
                    summaries.extend(summarize_errors(errors, family, "confirmation", q, role_i))
                    local = [candidate_record(errors, name, family, "confirmation", q, role_i)
                             for name in CANDIDATES]
                    candidate_rows.extend(local)
                    score[fi, q, role_i] = max(row["gain_over_family_mean"] for row in local)
                    passing = [row for row in local if row["passes"]]
                    if passing:
                        choice = max(passing, key=lambda row: (row["minimum_control_gain"],
                                                               row["gain_over_family_mean"],
                                                               row["minimum_control_win"]))
                        old = best.get(family)
                        if old is None or (choice["minimum_control_gain"], choice["gain_over_family_mean"],
                                           choice["minimum_control_win"]) > (
                                           old["minimum_control_gain"], old["gain_over_family_mean"],
                                           old["minimum_control_win"]):
                            best[family] = choice
            print(f"[confirmation] checkpoint {q + 1}/38", flush=True)
        decisions = {family: (best[family] if family in best else
                              {"family": family, "qualified": False, "reason": "no_confirmation_cell_passed"})
                     for family in families}
        for value in decisions.values():
            value.setdefault("qualified", "model" in value)
        save(OUT / "protocol/frozen_confirmation_decisions.json", decisions)
        write_rows(OUT / "analysis/confirmation_model_summaries.jsonl", summaries)
        write_rows(OUT / "analysis/confirmation_candidate_cells.jsonl", candidate_rows)
        score_path = OUT / "atlas/qwen4b_checkpoint_role_candidate_gain.float32.npy"
        np.save(score_path, score)

        fresh_records = {}
        authorized = []
        for family, decision in decisions.items():
            if not decision.get("qualified"):
                fresh_records[family] = {"family": family, "partition": "fresh_confirmation",
                                         "passes": False, "reason": "no_confirmation_cell_passed"}
                continue
            record, _errors, _models = evaluate_decision(
                field, by_family_partition, families, decision, "fresh_confirmation")
            fresh_records[family] = record
            if record["passes"]:
                authorized.append(family)
        save(OUT / "protocol/fresh_confirmation_adjudication.json", fresh_records)
        save(OUT / "protocol/frozen_lockbox_authorization.json",
             {"families": authorized, "decisions": {family: decisions[family] for family in authorized}})

        lock_records = {}
        passport_values: list[np.ndarray] = []
        passport_rows: list[dict] = []
        for family in families:
            decision = decisions[family]
            if family not in authorized:
                lock_records[family] = {"family": family, "partition": "fresh_lockbox",
                                        "passes": False, "reason": "fresh_confirmation_unqualified"}
                continue
            record, errors, models = evaluate_decision(
                field, by_family_partition, families, decision, "fresh_lockbox")
            lock_records[family] = record
            model_name = decision["model"]
            for metric, values in (
                ("candidate_error", errors[model_name]),
                ("candidate_coordinate_pass", np.logical_and.reduce([
                    errors[model_name] < errors[control] for control in CONTROLS]).astype(np.float32)),
                ("own_affine_slope", models["own_affine"][0]),
                ("own_affine_intercept", models["own_affine"][1]),
            ):
                passport_rows.append({"row": len(passport_rows), "family": family,
                                      "checkpoint": decision["checkpoint"], "role": decision["role"],
                                      "selected_model": model_name, "metric": metric})
                passport_values.append(np.asarray(values, dtype=np.float32))
            for control in CONTROLS:
                passport_rows.append({"row": len(passport_rows), "family": family,
                                      "checkpoint": decision["checkpoint"], "role": decision["role"],
                                      "selected_model": model_name, "metric": f"{control}_error"})
                passport_values.append(np.asarray(errors[control], dtype=np.float32))
        passport = np.stack(passport_values) if passport_values else np.empty((0, 2560), dtype=np.float32)
        passport_path = OUT / "atlas/qwen4b_selected_coordinate_passport.float32.npy"
        np.save(passport_path, passport)
        write_rows(OUT / "atlas/qwen4b_selected_coordinate_passport.rows.jsonl", passport_rows)
    finally:
        close_mmap(field)

    passed = [family for family, row in lock_records.items() if row.get("passes")]
    checks = {
        "field_shape": list(np.load(FIELD, mmap_mode="r").shape) == [len(index), 38, 6, 2560],
        "pair_counts": pair_counts_ok,
        "confirmation_frozen_before_fresh": (OUT / "protocol/frozen_confirmation_decisions.json").exists(),
        "lockbox_authorization_frozen": (OUT / "protocol/frozen_lockbox_authorization.json").exists(),
        "all_families_adjudicated": set(lock_records) == set(families),
        "score_shape": list(score.shape) == [len(families), 38, 6],
        "passport_coordinates": passport.shape[-1] == 2560,
        "finite_score": bool(np.isfinite(score).all()),
        "finite_passport": bool(np.isfinite(passport).all()),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "families": families, "gates": GATES,
        "confirmation_decisions": decisions,
        "fresh_confirmation": fresh_records,
        "lockbox_authorized_families": authorized,
        "fresh_lockbox": lock_records,
        "lockbox_passed_families": passed,
        "score_atlas": {"path": str(score_path.relative_to(ROOT)), "shape": list(score.shape)},
        "coordinate_passport": {"path": str(passport_path.relative_to(ROOT)),
                                "shape": list(passport.shape), "rows": len(passport_rows)},
        "hashes": {
            "preregistration": file_hash(OUT / "protocol/preregistration.json"),
            "confirmation_decisions": file_hash(OUT / "protocol/frozen_confirmation_decisions.json"),
            "lockbox_authorization": file_hash(OUT / "protocol/frozen_lockbox_authorization.json"),
            "score_atlas": file_hash(score_path), "coordinate_passport": file_hash(passport_path),
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"{len(authorized)}/{len(families)} families reached lockbox and "
                              f"{len(passed)}/{len(families)} passed all frozen full-coordinate controls; "
                              "these are predictive coordinate structures, not causal gears."),
        "next_authorization": "Run random balanced coordinate-mask causal identification on prospective anchors.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
