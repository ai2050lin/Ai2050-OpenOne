#!/usr/bin/env python3
"""Prospectively re-adjudicate nondegenerate factorial cells after Phase 2284 audit."""
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
SOURCE = RESULT / "phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild"
P2284 = RESULT / "phase2284_c2281_c2340_full_coordinate_factorial_interactions"
OUT = RESULT / "phase2285_c2341_c2400_factorial_nondegenerate_reanalysis"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2281_c2101_c2160_multilingual_operator_contract as campaign  # noqa: E402
import phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild as source  # noqa: E402
import phase2284_c2281_c2340_full_coordinate_factorial_interactions as factorial  # noqa: E402


PHASE = 2285
CAMPAIGN = "C2341-C2400"
MIN_INTERACTION_TO_MAIN = 0.01


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    summary = [{k: row[k] for k in ("family", "dataset", "route", "checkpoint", "role",
                                     "confirmation_pass", "fresh_authorized", "lockbox_pass")}
               for row in result["decisions"]]
    text = rf"""

## Phase {PHASE}: 析因仪器退化格审计与非退化前瞻重裁（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2284 的公式和全格 confirmation 扫描有效，但失败候选排序存在测量合同错误：它在全部门失败时继续偏好最小 MAE，因而反复选中 embedding/boundary 上主效应和交互都等于零的退化格；23 个被选格的交互/主效应比全部为零，不能支持“0/23 无交互”。该错误在完整全格扫描后、任何新格子的 fresh confirmation 或 lockbox 揭示前被识别。本期不重扫、不改材料，只冻结非退化资格 `interaction_to_main_ratio >= 0.01`，并只从 Phase2284 已登记的 confirmation 记录选择。受事、位置、时间、量词、条件、合取和分类链等全部路线继续参与。

**数学公式与门槛。** 非退化资格为：

$$
\eta_{{f,q,r}}=
\frac{{\mathbb E|I^{{s\times c}}_{{f,q,r,j}}|}}
{{\mathbb E|R^s_{{f,q,r,j}}|}}
\ge 0.01.
$$

在合格格中仍使用 Phase2284 原门：discovery 析因均值必须在 confirmation、fresh confirmation 和 fresh lockbox 同时优于零、共享、错族、错配单元和上一检查点控制，最小 MAE 增益不低于 `0.03`、逐坐标胜率不低于 `0.55`。选择规则先看是否过门，再看最小控制增益、最小胜率、相对零控制增益和 MAE；fresh 数据不参与选择。

**结果汇总与门槛。** 重裁摘要 `{json.dumps(summary, ensure_ascii=False)}`；通过路线 `{json.dumps(result['lockbox_passed'], ensure_ascii=False)}`；Phase2284 证据重裁 `{json.dumps(result['phase2284_reclassification'], ensure_ascii=False)}`；图谱 `{json.dumps(result['atlas'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2285_c2341_c2400_factorial_nondegenerate_reanalysis.py`；结果 `tests/glm5/result/phase2285_c2341_c2400_factorial_nondegenerate_reanalysis`。Phase2284 历史记录按 append-only 保留，不在文件中间改写；本期是正式纠偏索引。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}` 本期阈值是在看到 confirmation 退化问题后冻结，因此属于透明的仪器修正，不是对原合同的原样复现；只有 fresh confirmation 与 lockbox 仍保持前瞻性。稳定析因残差仍只是预测性交互，不是算子不交换、曲率、超边或因果机制。人工平行模板和上一检查点控制依然限制外推。下一阶段只对预注册 patient/location 且出现中层前瞻资格的锚点做多密度干预；没有资格则明确记为 NA，并继续 Qwen3-14B 条件式复验和图谱发布。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    p2284 = load(P2284 / "analysis/final.json")
    scan = read_rows(P2284 / "analysis/confirmation_scan.jsonl")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in scan:
        groups[(row["dataset"], row["family"], row["route"])].append(row)
    configs = {
        "bilingual": {"field": source.BI_FIELD, "index": source.BI_INDEX,
                       "families": list(campaign.BILINGUAL_FAMILIES)},
        "complex": {"field": source.COMPLEX_FIELD, "index": source.COMPLEX_INDEX,
                     "families": list(campaign.COMPLEX_FAMILIES)},
    }
    decisions, passport_values, passport_rows = [], [], []
    field = None
    try:
        for dataset, config in configs.items():
            field = np.load(config["field"], mmap_mode="r")
            evaluator = factorial.FactorialEvaluator(field, read_rows(config["index"]), config["families"], dataset)
            dataset_groups = sorted((key, value) for key, value in groups.items() if key[0] == dataset)
            for (unused_dataset, family, route), rows in dataset_groups:
                eligible = [row for row in rows if row["interaction_to_main_ratio"] >= MIN_INTERACTION_TO_MAIN]
                if not eligible:
                    raise RuntimeError(("no_nondegenerate_cell", dataset, family, route))
                confirmation = max(eligible, key=lambda row: (
                    row["passes"], row["minimum_gain"], row["minimum_win"],
                    row["gain_over_controls"]["zero"], -row["mae"],
                ))
                q, r = int(confirmation["checkpoint"]), int(confirmation["role_index"])
                fresh, passport = evaluator.evaluate(family, route, "fresh_confirmation", q, r)
                authorized = bool(confirmation["passes"] and fresh["passes"])
                lockbox = None
                if authorized:
                    lockbox, passport = evaluator.evaluate(family, route, "fresh_lockbox", q, r)
                decision = {
                    "family": family, "dataset": dataset, "route": route,
                    "checkpoint": q, "role_index": r, "role": factorial.ROLES[r],
                    "confirmation": confirmation, "confirmation_pass": bool(confirmation["passes"]),
                    "fresh_confirmation": fresh, "fresh_authorized": authorized,
                    "lockbox_revealed": authorized, "lockbox": lockbox,
                    "lockbox_pass": bool(lockbox and lockbox["passes"]),
                    "passport_partition": "fresh_lockbox" if lockbox else "fresh_confirmation",
                }
                for metric, values in passport.items():
                    passport_rows.append({"row": len(passport_values), "family": family,
                                          "dataset": dataset, "route": route,
                                          "checkpoint": q, "role": decision["role"],
                                          "partition": decision["passport_partition"], "metric": metric})
                    passport_values.append(values)
                decisions.append(decision)
                print(f"[nondegenerate] {dataset}/{family}/{route}: q={q} role={decision['role']} "
                      f"confirmation={decision['confirmation_pass']} fresh={authorized} "
                      f"lock={decision['lockbox_pass']}", flush=True)
            close_mmap(field)
            field = None
    finally:
        if field is not None:
            close_mmap(field)
    decision_path = OUT / "analysis/corrected_decisions.jsonl"
    write_rows(decision_path, decisions)
    atlas = np.stack(passport_values).astype(np.float32)
    atlas_path = OUT / "atlas/nondegenerate_factorial_passport.float32.npy"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(atlas_path, atlas)
    rows_path = OUT / "atlas/nondegenerate_factorial_passport.rows.jsonl"
    write_rows(rows_path, passport_rows)
    passed = [f"{row['dataset']}:{row['family']}:{row['route']}" for row in decisions if row["lockbox_pass"]]
    checks = {
        "phase2284_scan_complete": p2284["checks"]["scan_complete"],
        "all_routes_reselected": len(decisions) == 23,
        "all_selected_nondegenerate": all(row["confirmation"]["interaction_to_main_ratio"] >= MIN_INTERACTION_TO_MAIN for row in decisions),
        "fresh_not_used_for_selection": True,
        "ordered_reveal": all(row["fresh_authorized"] == row["lockbox_revealed"] for row in decisions),
        "atlas_all_coordinates": atlas.shape[1] == 2560,
        "finite_atlas": bool(np.isfinite(atlas).all()),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "phase2284_reclassification": {
            "engineering_scan_valid": True,
            "reported_zero_of_23_scientifically_valid": False,
            "reason": "failure ranking selected zero-main-effect zero-interaction cells",
            "historical_record_edited": False,
        },
        "nondegenerate_gate": MIN_INTERACTION_TO_MAIN,
        "decisions": decisions, "lockbox_passed": passed,
        "atlas": {"path": str(atlas_path.relative_to(ROOT)), "rows": str(rows_path.relative_to(ROOT)),
                  "shape": list(atlas.shape), "all_coordinates": True},
        "hashes": {"decisions": file_hash(decision_path), "atlas": file_hash(atlas_path),
                   "atlas_rows": file_hash(rows_path)},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"After excluding structurally zero cells using a frozen confirmation-only nondegeneracy rule, "
                              f"{len(passed)}/23 factorial maps survived fresh confirmation and lockbox controls. "
                              "Phase2284's selected 0/23 is invalid as a no-interaction claim; corrected positives remain observational."),
        "next_authorization": "Run scale-controlled intervention only for preregistered middle-layer patient/location anchors; continue conditional Qwen3-14B replication regardless of this branch.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
