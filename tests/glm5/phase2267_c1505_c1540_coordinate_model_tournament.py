#!/usr/bin/env python3
"""Coordinate-local model tournament with ordered lockbox reveal."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
SOURCE = RESULT / "phase2266_c1469_c1504_qwen4b_independent_fullfield"
OUT = RESULT / "phase2267_c1505_c1540_coordinate_model_tournament"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2265_c1433_c1468_independent_bilingual_contract as contract  # noqa: E402


PHASE = 2267
CAMPAIGNS = tuple(f"C{i}" for i in range(1505, 1541))
GAIN_GATE = 0.03
WIN_GATE = 0.55
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
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    dx = x - mx
    var = np.mean(dx * dx, axis=0)
    cov = np.mean(dx * (y - my), axis=0)
    a = cov / np.maximum(var, EPS)
    b = my - a * mx
    return a.astype(np.float32), b.astype(np.float32)


def errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - truth), axis=0)


def score(pred: np.ndarray, truth: np.ndarray, controls: dict[str, np.ndarray]) -> dict:
    own = errors(pred, truth)
    out = {"mae": float(own.mean())}
    for name, control_pred in controls.items():
        control = errors(control_pred, truth)
        base = float(control.mean())
        out[f"{name}_mae"] = base
        out[f"gain_vs_{name}"] = float((base - out["mae"]) / max(base, EPS))
        out[f"win_vs_{name}"] = float(np.mean(own < control))
    gains = [out[key] for key in out if key.startswith("gain_vs_")]
    wins = [out[key] for key in out if key.startswith("win_vs_")]
    out["min_control_gain"] = float(min(gains))
    out["min_control_win"] = float(min(wins))
    out["passes"] = bool(out["min_control_gain"] >= GAIN_GATE and out["min_control_win"] >= WIN_GATE)
    return out


def build_pairs(index: list[dict]) -> dict[str, dict[str, list[tuple[int, int, tuple]]]]:
    by_key: dict[tuple, dict[int, int]] = {}
    for row in index:
        key = (row["family"], row["language"], row["unit"], row["surface"], row["partition"])
        by_key.setdefault(key, {})[int(row["state"])] = int(row["hidden_index"])
    output: dict[str, dict[str, list[tuple[int, int, tuple]]]] = {}
    for key, states in by_key.items():
        if set(states) != {0, 1}:
            raise RuntimeError(("incomplete_pair", key, states))
        family, language, unit, surface, partition = key
        output.setdefault(family, {}).setdefault(partition, []).append(
            (states[0], states[1], (language, int(unit), surface)))
    for family in output:
        for partition in output[family]:
            output[family][partition].sort(key=lambda item: item[2])
    return output


def arrays(field: np.ndarray, pairs: list[tuple[int, int, tuple]], q: int, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i0 = [pair[0] for pair in pairs]
    i1 = [pair[1] for pair in pairs]
    h0 = np.asarray(field[i0, q, r], dtype=np.float32)
    h1 = np.asarray(field[i1, q, r], dtype=np.float32)
    return h0, h1, h1 - h0


def shuffled_response(h0: np.ndarray, h1: np.ndarray, labels: list[tuple]) -> np.ndarray:
    permutation = np.arange(len(labels))
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (language, _unit, surface) in enumerate(labels):
        groups.setdefault((language, surface), []).append(i)
    for members in groups.values():
        if len(members) < 2:
            raise RuntimeError(("shuffle_group_too_small", members))
        for position, member in enumerate(members):
            permutation[member] = members[(position + 1) % len(members)]
    if np.any(permutation == np.arange(len(labels))):
        raise RuntimeError("shuffle_fixed_point")
    return h1[permutation] - h0


def predict(coeff: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    return x * coeff[0] + coeff[1]


def fit_models(field: np.ndarray, pairs: dict, families: list[str]) -> dict[str, np.ndarray]:
    f_count, q_count, r_count, width = len(families), field.shape[1], field.shape[2], field.shape[3]
    shape = (f_count, q_count, r_count, width)
    own_a = np.empty(shape, dtype=np.float32)
    own_b = np.empty(shape, dtype=np.float32)
    shuffled_a = np.empty(shape, dtype=np.float32)
    shuffled_b = np.empty(shape, dtype=np.float32)
    mean_r = np.empty(shape, dtype=np.float32)
    mean_h1 = np.empty(shape, dtype=np.float32)
    shared_a = np.empty((q_count, r_count, width), dtype=np.float32)
    shared_b = np.empty((q_count, r_count, width), dtype=np.float32)
    for q in range(q_count):
        for r in range(r_count):
            pooled_x, pooled_y = [], []
            for fi, family in enumerate(families):
                train_pairs = pairs[family]["discovery"]
                h0, h1, response = arrays(field, train_pairs, q, r)
                own_a[fi, q, r], own_b[fi, q, r] = fit_affine(h0, response)
                shuffled = shuffled_response(h0, h1, [pair[2] for pair in train_pairs])
                shuffled_a[fi, q, r], shuffled_b[fi, q, r] = fit_affine(h0, shuffled)
                mean_r[fi, q, r] = response.mean(axis=0)
                mean_h1[fi, q, r] = h1.mean(axis=0)
                pooled_x.append(h0)
                pooled_y.append(response)
            shared_a[q, r], shared_b[q, r] = fit_affine(np.concatenate(pooled_x), np.concatenate(pooled_y))
        print(f"[fit] checkpoint {q + 1}/{q_count}", flush=True)
    return {"own_a": own_a, "own_b": own_b, "shuffled_a": shuffled_a, "shuffled_b": shuffled_b,
            "mean_r": mean_r, "mean_h1": mean_h1, "shared_a": shared_a, "shared_b": shared_b}


def evaluate_cell(field: np.ndarray, pairs: dict, families: list[str], models: dict,
                  family: str, partition: str, q: int, r: int, wrong_fi: int | None = None) -> tuple[dict, int]:
    fi = families.index(family)
    h0, _h1, truth = arrays(field, pairs[family][partition], q, r)
    own = predict((models["own_a"][fi, q, r], models["own_b"][fi, q, r]), h0)
    controls = {
        "mean": np.broadcast_to(models["mean_r"][fi, q, r], truth.shape),
        "algebraic": models["mean_h1"][fi, q, r] - h0,
        "shared": predict((models["shared_a"][q, r], models["shared_b"][q, r]), h0),
        "shuffled": predict((models["shuffled_a"][fi, q, r], models["shuffled_b"][fi, q, r]), h0),
    }
    if wrong_fi is None:
        wrong_errors = []
        for wi, _wrong in enumerate(families):
            if wi == fi:
                continue
            wrong_pred = predict((models["own_a"][wi, q, r], models["own_b"][wi, q, r]), h0)
            wrong_errors.append((float(errors(wrong_pred, truth).mean()), wi))
        wrong_fi = min(wrong_errors)[1]
    controls["wrong"] = predict((models["own_a"][wrong_fi, q, r], models["own_b"][wrong_fi, q, r]), h0)
    value = score(own, truth, controls)
    value["wrong_family"] = families[wrong_fi]
    value["rows"] = int(len(truth))
    return value, wrong_fi


def surface_control(field: np.ndarray, index: list[dict], family: str, partition: str,
                    q: int, r: int) -> dict:
    rows = [row for row in index if row["family"] == family and row["partition"] == partition]
    grouped: dict[tuple, dict[str, int]] = {}
    for row in rows:
        key = (row["language"], int(row["unit"]), int(row["state"]))
        grouped.setdefault(key, {})[row["surface"]] = int(row["hidden_index"])
    pairs = [(value["direct"], value["paraphrase"], key) for key, value in sorted(grouped.items())]
    direct = np.asarray(field[[pair[0] for pair in pairs], q, r], dtype=np.float32)
    paraphrase = np.asarray(field[[pair[1] for pair in pairs], q, r], dtype=np.float32)
    response = paraphrase - direct
    if partition == "discovery":
        raise RuntimeError("surface_control expects held-out partition")
    train_rows = [row for row in index if row["family"] == family and row["partition"] == "discovery"]
    train_grouped: dict[tuple, dict[str, int]] = {}
    for row in train_rows:
        key = (row["language"], int(row["unit"]), int(row["state"]))
        train_grouped.setdefault(key, {})[row["surface"]] = int(row["hidden_index"])
    train_pairs = [(value["direct"], value["paraphrase"]) for _key, value in sorted(train_grouped.items())]
    x = np.asarray(field[[pair[0] for pair in train_pairs], q, r], dtype=np.float32)
    y1 = np.asarray(field[[pair[1] for pair in train_pairs], q, r], dtype=np.float32)
    delta = y1 - x
    coeff = fit_affine(x, delta)
    pred = predict(coeff, direct)
    mean_pred = np.broadcast_to(delta.mean(axis=0), response.shape)
    own_error, mean_error = errors(pred, response), errors(mean_pred, response)
    return {"rows": len(pairs), "mae": float(own_error.mean()), "mean_mae": float(mean_error.mean()),
            "gain_over_mean": float((mean_error.mean() - own_error.mean()) / max(float(mean_error.mean()), EPS)),
            "coordinate_win_fraction": float(np.mean(own_error < mean_error))}


def propagation_control(field: np.ndarray, pairs: dict, family: str, q: int, r: int,
                        partition: str) -> dict:
    if q == 0:
        return {"ran": False, "reason": "embedding_has_no_upstream_checkpoint"}
    train_prev = arrays(field, pairs[family]["discovery"], q - 1, r)[2]
    train_now = arrays(field, pairs[family]["discovery"], q, r)[2]
    coeff = fit_affine(train_prev, train_now)
    test_prev = arrays(field, pairs[family][partition], q - 1, r)[2]
    truth = arrays(field, pairs[family][partition], q, r)[2]
    pred = predict(coeff, test_prev)
    mean_pred = np.broadcast_to(train_now.mean(axis=0), truth.shape)
    own_error, mean_error = errors(pred, truth), errors(mean_pred, truth)
    return {"ran": True, "rows": len(truth), "mae": float(own_error.mean()), "mean_mae": float(mean_error.mean()),
            "gain_over_mean": float((mean_error.mean() - own_error.mean()) / max(float(mean_error.mean()), EPS)),
            "coordinate_win_fraction": float(np.mean(own_error < mean_error))}


def coordinate_atlas(field: np.ndarray, pairs: dict, families: list[str], models: dict,
                     decisions: list[dict]) -> tuple[np.ndarray, list[dict]]:
    rows, labels = [], []
    for decision in decisions:
        family, q, r = decision["family"], decision["checkpoint"], decision["role_index"]
        partition = "fresh_lockbox" if decision["lockbox_revealed"] else "fresh_confirmation"
        fi, wi = families.index(family), families.index(decision["wrong_family"])
        h0, _h1, truth = arrays(field, pairs[family][partition], q, r)
        predictions = {
            "truth_mean_abs": np.mean(np.abs(truth), axis=0),
            "own_abs_error": errors(predict((models["own_a"][fi, q, r], models["own_b"][fi, q, r]), h0), truth),
            "mean_abs_error": errors(np.broadcast_to(models["mean_r"][fi, q, r], truth.shape), truth),
            "algebraic_abs_error": errors(models["mean_h1"][fi, q, r] - h0, truth),
            "shared_abs_error": errors(predict((models["shared_a"][q, r], models["shared_b"][q, r]), h0), truth),
            "shuffled_abs_error": errors(predict((models["shuffled_a"][fi, q, r], models["shuffled_b"][fi, q, r]), h0), truth),
            "wrong_abs_error": errors(predict((models["own_a"][wi, q, r], models["own_b"][wi, q, r]), h0), truth),
        }
        for metric, values in predictions.items():
            labels.append({"row": len(rows), "family": family, "partition": partition,
                           "checkpoint": q, "role": contract.ROLES[r], "metric": metric})
            rows.append(values.astype(np.float16))
    return np.stack(rows), labels


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    summary = [{"family": row["family"], "checkpoint": row["checkpoint"], "role": row["role"],
                "fresh_confirmation": row["fresh_confirmation"], "lockbox": row["lockbox"],
                "propagation": row["propagation"], "surface_control": row["surface_control"]}
               for row in result["decisions"]]
    text = rf"""

## Phase {PHASE}: 独立全坐标模型竞赛与有序锁箱裁决（C1505-C1540） [{stamp}]

**测试原理与用例。** 使用Phase2266的10个行为合格家族、2560样本六角色全坐标场。每个家族在38检查点×6角色的228个格子上，用discovery的48个真假配对拟合逐坐标模型；confirmation的16对只选择一个候选格子及其最强错家族对照；fresh confirmation的32对必须同时击败家族均值、纯代数耦合、跨家族共享仿射、错配仿射和错家族仿射，全部相对MAE增益不低于0.03且逐坐标胜率不低于0.55，才允许揭示32对fresh lockbox。另在同一冻结格子检验上一检查点响应能否预测下一检查点响应，以及直接表面能否预测释义表面变化。

**公式。** 真假响应、同坐标模型与控制门为：

$$
R_{{i,q,r,j}}=H^1_{{i,q,r,j}}-H^0_{{i,q,r,j}},\qquad
\widehat R_{{i,q,r,j}}=a_{{f,q,r,j}}H^0_{{i,q,r,j}}+b_{{f,q,r,j}}.
$$

$$
G=\min_c\frac{{\operatorname{{MAE}}(c)-\operatorname{{MAE}}(\widehat R)}}
{{\operatorname{{MAE}}(c)}},\qquad
W=\min_c\Pr_j[e_j(\widehat R)<e_j(c)].
$$

其中控制 `c` 为均值、纯代数、共享、错配和错家族；正式通过要求fresh confirmation与锁箱都满足 `G>=0.03,W>=0.55`。这是一组预测关系，不是因果电路。

**结果汇总与门槛。** 家族裁决 `{json.dumps(summary, ensure_ascii=False)}`。通过fresh confirmation的家族 `{json.dumps(result['fresh_confirmation_survivors'], ensure_ascii=False)}`；正式锁箱通过家族 `{json.dumps(result['lockbox_survivors'], ensure_ascii=False)}`。全格扫描文件、候选裁决和2560坐标误差图谱哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；工程检查 `{result['checks']}`，总通过 `{result['all_checks_passed']}`。

**分析与理论进展。** `{result['strict_conclusion']}` 同表面控制用于判断同坐标条件化是否只是语义真假特有；一层传播只说明相邻检查点的有效预测依赖，不能命名为唯一计算路径。结果保留所有2560坐标，没有PCA、Top-K或余弦筛选。理论主体“条件化输出场闭合理论”与RDC不改名；只有跨独立词汇、严格控制和锁箱同时成立的窄规律才升级拼图。

**问题、硬伤、结论与相关文件。** 候选从228格中选择，虽有双独立分区验证，仍存在格子多重搜索和同架构单模型限制；`H0`与`R=H1-H0`的代数耦合只能由控制削弱，不能从数学上消失。中英材料无人类盲评、输出码元语言化、float16存盘仍是硬伤。下一步仅对正式锁箱通过候选授权近流形上游坐标干预；其余路线作为已登记缺失继续参与图谱，不触发项目级停止。脚本 `tests/glm5/phase2267_c1505_c1540_coordinate_model_tournament.py`；结果 `tests/glm5/result/phase2267_c1505_c1540_coordinate_model_tournament`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        return load(final)
    source = load(SOURCE / "analysis/final.json")
    if not source["all_checks_passed"]:
        raise RuntimeError("Phase2266 source is invalid")
    families = list(source["behavior"]["qualified_families"])
    index = read_rows(SOURCE / "raw/role_field_index.jsonl")
    field = np.load(SOURCE / "raw/qwen3_4b_qualified_role_field.float16.npy", mmap_mode="r")
    pairs = build_pairs(index)
    if set(families) != set(pairs):
        raise RuntimeError(("family_pair_mismatch", families, sorted(pairs)))
    models = fit_models(field, pairs, families)
    scan_rows = []
    decisions = []
    for family in families:
        candidates = []
        for q in range(field.shape[1]):
            for r in range(field.shape[2]):
                metric, wrong_fi = evaluate_cell(field, pairs, families, models, family, "confirmation", q, r)
                row = {"family": family, "checkpoint": q, "role_index": r, "role": contract.ROLES[r],
                       "partition": "confirmation", **metric}
                scan_rows.append(row)
                candidates.append((metric["min_control_gain"], metric["min_control_win"], -metric["mae"], q, r, wrong_fi, metric))
        selected = max(candidates)
        q, r, wrong_fi = selected[3], selected[4], selected[5]
        confirmation = selected[6]
        fresh, _ = evaluate_cell(field, pairs, families, models, family, "fresh_confirmation", q, r, wrong_fi)
        fresh_pass = bool(fresh["passes"])
        lockbox = None
        lock_pass = False
        if fresh_pass:
            lockbox, _ = evaluate_cell(field, pairs, families, models, family, "fresh_lockbox", q, r, wrong_fi)
            lock_pass = bool(lockbox["passes"])
        propagation = {
            "fresh_confirmation": propagation_control(field, pairs, family, q, r, "fresh_confirmation"),
            "lockbox": propagation_control(field, pairs, family, q, r, "fresh_lockbox") if fresh_pass else {"ran": False, "reason": "lockbox_not_revealed"},
        }
        surface = {
            "fresh_confirmation": surface_control(field, index, family, "fresh_confirmation", q, r),
            "lockbox": surface_control(field, index, family, "fresh_lockbox", q, r) if fresh_pass else {"ran": False, "reason": "lockbox_not_revealed"},
        }
        decisions.append({"family": family, "checkpoint": q, "role_index": r, "role": contract.ROLES[r],
                          "wrong_family": families[wrong_fi], "confirmation": confirmation,
                          "fresh_confirmation": fresh, "fresh_confirmation_pass": fresh_pass,
                          "lockbox_revealed": fresh_pass, "lockbox": lockbox, "lockbox_pass": lock_pass,
                          "propagation": propagation, "surface_control": surface})
        print(f"[decision] {family}: q={q} role={contract.ROLES[r]} fresh={fresh_pass} lock={lock_pass}", flush=True)
    scan_path = OUT / "analysis/confirmation_cell_scan.jsonl"
    decision_path = OUT / "analysis/family_decisions.jsonl"
    write_rows(scan_path, scan_rows)
    write_rows(decision_path, decisions)
    atlas, labels = coordinate_atlas(field, pairs, families, models, decisions)
    atlas_path = OUT / "atlas/qwen4b_coordinate_model_errors.float16.npy"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(atlas_path, atlas)
    labels_path = OUT / "atlas/qwen4b_coordinate_model_errors.rows.jsonl"
    write_rows(labels_path, labels)
    fresh_survivors = [row["family"] for row in decisions if row["fresh_confirmation_pass"]]
    lock_survivors = [row["family"] for row in decisions if row["lockbox_pass"]]
    checks = {
        "families_complete": len(decisions) == len(families),
        "scan_complete": len(scan_rows) == len(families) * 38 * 6,
        "selection_confirmation_only": all(row["confirmation"] is not None for row in decisions),
        "ordered_reveal": all(row["lockbox_revealed"] == row["fresh_confirmation_pass"] for row in decisions),
        "gate_exact": all((not row["fresh_confirmation_pass"] or row["fresh_confirmation"]["min_control_gain"] >= GAIN_GATE and row["fresh_confirmation"]["min_control_win"] >= WIN_GATE) for row in decisions),
        "atlas_all_coordinates": atlas.shape[1] == 2560,
        "finite_atlas": bool(np.isfinite(atlas).all()),
    }
    strict = (f"{len(fresh_survivors)}/{len(families)} candidates survived the independent fresh-confirmation controls and "
              f"{len(lock_survivors)}/{len(families)} survived the untouched lockbox; these are coordinate-local predictive relations, not causal gears.")
    hashes = {"scan": file_hash(scan_path), "decisions": file_hash(decision_path),
              "atlas": file_hash(atlas_path), "atlas_rows": file_hash(labels_path)}
    result = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(), "families": families,
              "gates": {"gain": GAIN_GATE, "coordinate_win": WIN_GATE}, "decisions": decisions,
              "fresh_confirmation_survivors": fresh_survivors, "lockbox_survivors": lock_survivors,
              "atlas": {"path": str(atlas_path.relative_to(ROOT)), "shape": list(atlas.shape),
                        "rows": str(labels_path.relative_to(ROOT)), "all_coordinates": True},
              "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
              "strict_conclusion": strict,
              "next_authorization": "Run near-manifold causal controls only for lockbox survivors; publish all-coordinate atlas."}
    save(final, result)
    append_memo(result)
    print(json.dumps({key: value for key, value in result.items() if key != "decisions"}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
