#!/usr/bin/env python3
"""Coordinate-passport ecology for Phase2254 behavior-qualified families."""
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
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
SOURCE = RESULT / "phase2254_c1121_c1152_qwen_construction_full_field"
OUT = RESULT / "phase2255_c1153_c1184_coordinate_passport_ecology"
sys.path.insert(0, str(TESTS))

import phase2253_c1097_c1120_construction_ecology_contract as contract  # noqa: E402


PHASE = 2255
CAMPAIGNS = tuple(f"C{i}" for i in range(1153, 1185))
FIELD_PATH = SOURCE / "raw/qwen3_4b_qualified_role_field.float16.npy"
INDEX_PATH = SOURCE / "raw/role_field_index.jsonl"
PASSPORT_PATH = OUT / "atlas/qwen3_4b_coordinate_passport.float32.npy"
LABEL_PATH = OUT / "atlas/qwen3_4b_coordinate_passport_rows.jsonl"
MASK_PATH = OUT / "atlas/discovery_loo_candidate_masks.uint8.npy"


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


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def paired_unit_responses(field: np.ndarray, index: list[dict]) -> dict[str, dict[int, np.ndarray]]:
    cells: dict[tuple, dict[int, int]] = defaultdict(dict)
    for row in index:
        if row["panel"] != "construction_broad":
            continue
        key = (row["family"], row["language"], int(row["unit"]), row["surface"])
        cells[key][int(row["state"])] = int(row["hidden_index"])
    grouped: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for (family, _language, unit, _surface), states in cells.items():
        if set(states) != {0, 1}:
            continue
        response = np.asarray(field[states[1]], dtype=np.float32) - np.asarray(field[states[0]], dtype=np.float32)
        grouped[family][unit].append(response)
    return {family: {unit: np.mean(values, axis=0, dtype=np.float32)
                     for unit, values in units.items()}
            for family, units in grouped.items()}


def metric_arrays(actual: np.ndarray, prediction: np.ndarray, wrong: np.ndarray) -> dict[str, np.ndarray]:
    own_error = np.mean(np.abs(actual - prediction), axis=0, dtype=np.float64)
    zero_error = np.mean(np.abs(actual), axis=0, dtype=np.float64)
    wrong_error = np.mean(np.abs(actual - wrong), axis=0, dtype=np.float64)
    eps = 1e-12
    return {
        "own_mae": own_error.astype(np.float32),
        "zero_mae": zero_error.astype(np.float32),
        "wrong_mae": wrong_error.astype(np.float32),
        "gain_zero": ((zero_error - own_error) / np.maximum(zero_error, eps)).astype(np.float32),
        "gain_wrong": ((wrong_error - own_error) / np.maximum(wrong_error, eps)).astype(np.float32),
        "sign_consistency": np.abs(np.mean(np.sign(actual), axis=0, dtype=np.float64)).astype(np.float32),
        "own_win_rate": np.mean(np.abs(actual - prediction) < np.abs(actual - wrong), axis=0,
                                 dtype=np.float64).astype(np.float32),
        "median_abs_response": np.median(np.abs(actual), axis=0).astype(np.float32),
    }


def discovery_loo(family_units: dict[int, np.ndarray], wrong_units: dict[int, np.ndarray]) -> tuple[dict, np.ndarray]:
    units = sorted(unit for unit in family_units if unit < contract.DISCOVERY_UNITS)
    if len(units) < contract.COORDINATE_GATES["minimum_independent_units"]:
        raise RuntimeError(("insufficient_discovery_units", units))
    actual, prediction, wrong = [], [], []
    for unit in units:
        others = [family_units[x] for x in units if x != unit]
        wrong_others = [wrong_units[x] for x in units if x != unit and x in wrong_units]
        actual.append(family_units[unit])
        prediction.append(np.mean(others, axis=0, dtype=np.float32))
        wrong.append(np.mean(wrong_others, axis=0, dtype=np.float32))
    actual_a = np.stack(actual)
    prediction_a = np.stack(prediction)
    wrong_a = np.stack(wrong)
    return metric_arrays(actual_a, prediction_a, wrong_a), np.mean(actual_a, axis=0, dtype=np.float32)


def panel_metrics(family_units: dict[int, np.ndarray], own_proto: np.ndarray,
                  wrong_proto: np.ndarray, units: list[int]) -> tuple[dict | None, np.ndarray | None]:
    present = [unit for unit in units if unit in family_units]
    if not present:
        return None, None
    actual = np.stack([family_units[unit] for unit in present])
    own = np.broadcast_to(own_proto, actual.shape)
    wrong = np.broadcast_to(wrong_proto, actual.shape)
    return metric_arrays(actual, own, wrong), actual


def cell_summary(metrics: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {"coordinates": 0, "pass": False, "reason": "empty_discovery_mask"}
    values = {name: float(np.mean(array[mask], dtype=np.float64)) for name, array in metrics.items()}
    values["coordinates"] = count
    gates = contract.COORDINATE_GATES
    values["pass"] = bool(values["gain_zero"] >= gates["fresh_gain_over_zero"]
                          and values["gain_wrong"] >= gates["fresh_gain_over_wrong_family"]
                          and values["sign_consistency"] >= gates["coordinate_sign_consistency"]
                          and values["own_win_rate"] >= gates["coordinate_own_family_win_rate"])
    return values


def formation_intervals(cell_pass: np.ndarray) -> list[dict]:
    output = []
    need = int(contract.COORDINATE_GATES["successive_checkpoints"])
    for role_i, role in enumerate(contract.ROLES):
        series = cell_pass[:, role_i].astype(bool)
        starts = []
        for q in range(0, len(series) - need + 1):
            if bool(np.all(series[q:q + need])):
                starts.append(q)
        if not starts:
            output.append({"role": role, "formation": None, "dissolution": None})
            continue
        start = starts[0]
        end = start + need - 1
        while end + 1 < len(series) and series[end + 1]:
            end += 1
        output.append({"role": role, "formation": start, "dissolution": end})
    return output


def run_passports(field: np.ndarray, index: list[dict], behavior: dict) -> dict:
    responses = paired_unit_responses(field, index)
    families = sorted(responses)
    if len(families) < 2:
        raise RuntimeError(("need_at_least_two_qualified_families", families))
    rows_out: list[np.ndarray] = []
    labels: list[dict] = []
    masks = np.zeros((len(families), field.shape[1], field.shape[2], field.shape[3]), dtype=np.uint8)
    result = {}
    panels = {
        "parent_confirmation": list(range(8, 12)),
        "parent_lockbox": list(range(12, 16)),
        "parent_holdout": list(range(8, 16)),
        "fresh_confirmation": list(range(16, 22)),
        "fresh_lockbox": list(range(22, 28)),
    }
    for family_i, family in enumerate(families):
        wrong_family = families[(family_i + 1) % len(families)]
        loo, proto = discovery_loo(responses[family], responses[wrong_family])
        wrong_proto = np.mean([responses[wrong_family][u] for u in range(contract.DISCOVERY_UNITS)],
                              axis=0, dtype=np.float32)
        gates = contract.COORDINATE_GATES
        candidate = ((loo["gain_zero"] >= gates["fresh_gain_over_zero"])
                     & (loo["gain_wrong"] >= gates["fresh_gain_over_wrong_family"])
                     & (loo["sign_consistency"] >= gates["coordinate_sign_consistency"])
                     & (loo["own_win_rate"] >= gates["coordinate_own_family_win_rate"]))
        masks[family_i] = candidate.astype(np.uint8)
        family_result = {
            "wrong_family_control": wrong_family,
            "discovery_units": contract.DISCOVERY_UNITS,
            "candidate_coordinates": int(candidate.sum()),
            "candidate_fraction": float(candidate.mean()),
            "panels": {},
        }
        pass_by_panel: dict[str, np.ndarray] = {}
        for panel, units in panels.items():
            metrics, _actual = panel_metrics(responses[family], proto, wrong_proto, units)
            if metrics is None:
                family_result["panels"][panel] = {"status": "NA_no_rows"}
                continue
            cell_rows = []
            cell_pass = np.zeros((field.shape[1], field.shape[2]), dtype=bool)
            for q in range(field.shape[1]):
                for role_i, role in enumerate(contract.ROLES):
                    summary = cell_summary({name: value[q, role_i] for name, value in metrics.items()},
                                           candidate[q, role_i])
                    cell_rows.append({"checkpoint": q, "role": role, **summary})
                    cell_pass[q, role_i] = bool(summary.get("pass"))
            family_result["panels"][panel] = {
                "independent_units": len(units), "passing_cells": int(sum(x.get("pass", False) for x in cell_rows)),
                "cells": cell_rows,
            }
            pass_by_panel[panel] = cell_pass
            for metric, matrix in metrics.items():
                for q in range(field.shape[1]):
                    for role_i, role in enumerate(contract.ROLES):
                        rows_out.append(matrix[q, role_i].astype(np.float32))
                        labels.append({"family": family, "panel": panel, "metric": metric,
                                       "checkpoint": q, "role": role})
        prelockbox = pass_by_panel["parent_holdout"] & pass_by_panel["fresh_confirmation"]
        validated = prelockbox & pass_by_panel["fresh_lockbox"]
        family_result["prelockbox_formation"] = formation_intervals(prelockbox)
        family_result["fresh_lockbox_validated_formation"] = formation_intervals(validated)
        family_result["causal_candidate_cells"] = [
            {"checkpoint": q, "role": contract.ROLES[r]}
            for r in range(len(contract.ROLES)) for q in range(field.shape[1] - 1)
            if prelockbox[q, r] and prelockbox[q + 1, r]
        ]
        family_result["fresh_lockbox_validated_cells"] = [
            {"checkpoint": q, "role": contract.ROLES[r]}
            for r in range(len(contract.ROLES)) for q in range(field.shape[1] - 1)
            if validated[q, r] and validated[q + 1, r]
        ]
        family_result["causal_qualified"] = bool(family_result["causal_candidate_cells"])
        result[family] = family_result
    PASSPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(rows_out).astype(np.float32)
    np.save(PASSPORT_PATH, matrix, allow_pickle=False)
    write_rows(LABEL_PATH, labels)
    np.save(MASK_PATH, masks, allow_pickle=False)
    return {
        "families": result, "family_order": families,
        "passport_shape": list(matrix.shape), "mask_shape": list(masks.shape),
        "causal_qualified_families": sorted(f for f, value in result.items() if value["causal_qualified"]),
        "behavior_missing_routes": sorted(key for key, value in behavior["cells"].items()
                                          if not value["dual_qualified"]),
        "composition_internal_status": "NA_all_graph_and_attitude_composition_cells_failed_dual_behavior",
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8-sig")
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    if marker in existing:
        correction = "**Phase2255 前瞻时序重裁附录"
        if correction not in existing:
            compact = {family: {
                "causal_candidate_cells": len(data["causal_candidate_cells"]),
                "fresh_lockbox_validated_cells": len(data["fresh_lockbox_validated_cells"]),
                "prelockbox_formation": data["prelockbox_formation"],
                "fresh_lockbox_validated_formation": data["fresh_lockbox_validated_formation"],
                "causal_qualified": data["causal_qualified"],
            } for family, data in result["analysis"]["families"].items()}
            text = f"""

{correction} [{stamp}]。** 首次账把fresh lockbox通过格用于形成层选择，会污染下一期锁箱因果的前瞻性；该“因果资格”现已作废，且当时尚未运行任何干预。重裁规定：discovery留一冻结坐标掩码，父confirmation+lockbox合并为8单元holdout，只有父holdout与fresh confirmation同时连续通过的检查点-角色格才能预先冻结；fresh lockbox完全不参与候选选择，只用于后续因果和独立预测。重裁结果为 `{json.dumps(compact, ensure_ascii=False)}`；后续因果只接受 `causal_candidate_cells`，不得使用旧 `successive_fresh_cells`。
"""
            with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        return
    compact = {}
    for family, data in result["analysis"]["families"].items():
        compact[family] = {
            "candidate_coordinates": data["candidate_coordinates"],
            "candidate_fraction": data["candidate_fraction"],
            "panel_passing_cells": {p: v.get("passing_cells") for p, v in data["panels"].items()},
            "causal_candidate_cells": data["causal_candidate_cells"],
            "fresh_lockbox_validated_cells": data["fresh_lockbox_validated_cells"],
            "causal_qualified": data["causal_qualified"],
        }
    text = rf"""

## Phase {PHASE}: 留一单元逐坐标护照与构式形成区间（C1153-C1184） [{stamp}]

**测试原理与统计单位。** 本期只分析Phase2254双行为合格的主动/被动角色与属性覆盖，不读取行为失败构式的HiddenState。每个“独立单元”先把中英和两种表面的状态差平均，避免把四条模板行伪装成四个独立样本；8个discovery单元逐一留出，用其余7个形成本族逐坐标原型，并用另一合格族作等容量错族控制。由discovery留一账冻结全部满足门槛的坐标集合，再依次裁决父confirmation、父lockbox、fresh confirmation和fresh lockbox。

**公式。** 对单元级响应与留一误差：

$$
R_{{f,u,q,r,j}}=\frac1{{|L||S|}}\sum_{{l,s}}\left(H^1-H^0\right)_{{f,u,l,s,q,r,j}},
$$

$$
e^{{own}}_{{u,q,r,j}}=\left|R_{{f,u,q,r,j}}-\frac1{{U-1}}\sum_{{v\ne u}}R_{{f,v,q,r,j}}\right|.
$$

候选不是Top-K，而是对全部坐标逐一应用冻结门槛；形成区间要求同一角色连续两个检查点在fresh lockbox通过。

**结果汇总。** 逐坐标结果为 `{json.dumps(compact, ensure_ascii=False)}`。严格因果资格族为 `{json.dumps(result['analysis']['causal_qualified_families'], ensure_ascii=False)}`。因果候选只由父holdout与fresh confirmation冻结，fresh lockbox不参与选择。所有图路径与态度组合格均未通过Phase2254双行为，因此内部组合状态严格为 `{result['analysis']['composition_internal_status']}`，不是组合机制零效应。完整护照矩阵形状 `{result['analysis']['passport_shape']}`，候选掩码形状 `{result['analysis']['mask_shape']}`，哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`。

**分析与理论进展。** 若某族获得连续fresh格，只能说明由父词汇留一形成的逐坐标集合，在全新单元的相同构式响应中同时优于零响应和错族控制，并保持冻结符号/本族胜率；它仍不是唯一坐标字典，也不证明模型内部执行均值原型。若资格为空，则只淘汰本轮这套逐坐标原型，不否定分布式编码。理论主体与RDC不改名。

**问题、硬伤与结论。** 当前只有两个行为合格族，错族控制选择有限；每族discovery仅8个独立单元；逐坐标门槛会产生多重候选且未作唯一性主张；float16和六角色采样限制低值与未观测token；图路线本轮只能作行为级结论。工程检查 `{result['all_checks_passed']}`。下一步只对这里前瞻合格的族做冻结坐标掩码因果裁决，同时所有模型继续跑fresh行为面板。

**相关文件。** 脚本 `tests/glm5/phase2255_c1153_c1184_coordinate_passport_ecology.py`；结果 `tests/glm5/result/phase2255_c1153_c1184_coordinate_passport_ecology`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        existing = load(final_path)
        families = existing.get("analysis", {}).get("families", {})
        if families and all("causal_candidate_cells" in value for value in families.values()):
            return existing
    source = load(SOURCE / "analysis/final.json")
    if not source["all_checks_passed"]:
        raise RuntimeError("Phase2254 source did not pass engineering checks")
    field = np.load(FIELD_PATH, mmap_mode="r")
    index = read_rows(INDEX_PATH)
    try:
        analysis = run_passports(field, index, source["behavior"])
    finally:
        close_mmap(field)
    hashes = {"passport": file_hash(PASSPORT_PATH), "labels": file_hash(LABEL_PATH),
              "masks": file_hash(MASK_PATH), "source_field": source["hashes"]["role_field"]}
    passport = np.load(PASSPORT_PATH, mmap_mode="r")
    masks = np.load(MASK_PATH, mmap_mode="r")
    checks = {
        "source_valid": True, "two_or_more_families": len(analysis["family_order"]) >= 2,
        "passport_2d": passport.ndim == 2 and passport.shape[1] == 2560,
        "passport_finite": bool(np.isfinite(passport).all()),
        "mask_4d": masks.ndim == 4 and masks.shape[-1] == 2560,
        "all_behavior_failures_registered": bool(analysis["behavior_missing_routes"]),
        "composition_na_not_zero": analysis["composition_internal_status"].startswith("NA_"),
    }
    close_mmap(passport)
    close_mmap(masks)
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "analysis": analysis,
        "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Coordinate passports are adjudicated only for behavior-qualified families; composition remains NA in this denominator.",
        "next_authorization": "Run causal tests only for prospectively qualified families and continue every cross-model fresh behavior panel sequentially.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps({"analysis": analysis, "hashes": hashes, "checks": checks},
                     ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
