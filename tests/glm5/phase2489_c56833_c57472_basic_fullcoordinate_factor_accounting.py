#!/usr/bin/env python3
"""Balanced full-coordinate factor accounting without mechanistic percentage claims."""
from __future__ import annotations

import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2488 = RESULT / "phase2488_c55873_c56832_qwen4b_orthogonal_fullcoordinate_field"
OUT = RESULT / "phase2489_c56833_c57472_basic_fullcoordinate_factor_accounting"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2489, "C56833-C57472"
FACTORS = ("family", "language", "surface", "output_interface")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def effect_account(x: np.ndarray, rows: list[dict]) -> dict:
    x = np.asarray(x, dtype=np.float64)
    grand = x.mean(axis=0)
    centered = x - grand
    total = float(np.square(centered).sum())
    main_effects: dict[str, dict[str, np.ndarray]] = {}
    main_ss: dict[str, float] = {}
    for factor in FACTORS:
        levels = sorted({str(r[factor]) for r in rows})
        effects = {}
        ss = 0.0
        for level in levels:
            mask = np.array([str(r[factor]) == level for r in rows])
            effect = x[mask].mean(axis=0) - grand
            effects[level] = effect
            ss += int(mask.sum()) * float(np.square(effect).sum())
        main_effects[factor] = effects
        main_ss[factor] = ss
    pair_ss: dict[str, float] = {}
    for first, second in combinations(FACTORS, 2):
        ss = 0.0
        for a in main_effects[first]:
            for b in main_effects[second]:
                mask = np.array([str(r[first]) == a and str(r[second]) == b for r in rows])
                cell = x[mask].mean(axis=0) - grand - main_effects[first][a] - main_effects[second][b]
                ss += int(mask.sum()) * float(np.square(cell).sum())
        pair_ss[f"{first}*{second}"] = ss
    accounted = sum(main_ss.values()) + sum(pair_ss.values())
    higher = max(0.0, total - accounted)
    safe = total if total > 0 else 1.0
    return {
        "total_centered_energy": total,
        "main_share": {k: v / safe for k, v in main_ss.items()},
        "pair_share": {k: v / safe for k, v in pair_ss.items()},
        "higher_order_or_cell_residual_share": higher / safe,
        "orthogonality_residual": (total - accounted - higher) / safe,
    }


def family_passports(x: np.ndarray, rows: list[dict], families: list[str]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grand = x.mean(axis=0)
    return np.stack([x[np.array([r["family"] == family for r in rows])].mean(axis=0) - grand for family in families]).astype(np.float32)


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 正交全场的基本因素与二阶交互逐坐标账本（拒绝“语义/格式百分比”）（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Phase2488的完整十二族×中英×四surface×四interface平衡格上，对每个事件、每个qpoint分别计算family、language、surface、interface主效应及六个二阶交互；confirmation/unit15选择“family描述能量份额”最大的qpoint，lockbox/unit16一次报告同一qpoint。主效应按全2560坐标平方和计量，没有Top-K；同时保存每事件锁箱family有符号护照。record-predicate发生在输出接口后缀之前，因causal mask应构成接口效应近零的协议对照。

$$SS_a=\sum_n\lVert\mu_{{a(n)}}-\mu\rVert_2^2,\qquad SS_{{ab}}=\sum_n\lVert\mu_{{a(n),b(n)}}-\mu_{{a(n)}}-\mu_{{b(n)}}+\mu\rVert_2^2.$$

**结果汇总。** unit15冻结层位 `{json.dumps(result['selection'], ensure_ascii=False)}`；unit16同层账本 `{json.dumps(result['lockbox_selected'], ensure_ascii=False)}`；协议对照 `{json.dumps(result['controls'], ensure_ascii=False)}`；护照 `{json.dumps(result['passports'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2489_c56833_c57472_basic_fullcoordinate_factor_accounting.py`；全部5事件×38层的unit15/unit16账本、冻结family护照与`analysis/final.json`位于同名目录。

**分析与理论进展。** 账本只回答“在这套材料、这个事件、这个层位上，哪些受控轴带来多大均方变化”。它不把language算作格式，不把family算作纯语义，也不假定主效应可以组成模型内部的加法程序。若record-predicate接口份额接近数值地板而answer-boundary上升，说明协议能定位后缀接口影响的出现位置，但仍不能说模型在该处完成了格式选择。

**问题硬伤与结论。** family与谓词词项仍共变；每个完整factor cell只有一条样本，三阶以上项与单元残差不可分。均方份额受坐标尺度、事件位置和提示长度影响。该Phase提供正交描述账本及后续有符号分析的冻结层位，不建立因果贡献或机制比例。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final2488 = json.loads((P2488 / "analysis/final.json").read_text(encoding="utf-8"))
    field = np.load(final2488["collection"]["event_field"], mmap_mode="r")
    rows = read_jsonl(Path(final2488["collection"]["index"]))
    events = final2488["collection"]["events"]
    families = sorted({r["family"] for r in rows})
    by_unit: dict[str, dict] = {"15": {}, "16": {}}
    selection: dict[str, int] = {}
    selected_passports = np.zeros((2, len(events), len(families), field.shape[-1]), dtype=np.float32)
    for unit_index, unit in enumerate((15, 16)):
        mask = np.array([r["unit"] == unit for r in rows])
        unit_rows = [r for r in rows if r["unit"] == unit]
        for event_index, event in enumerate(events):
            by_unit[str(unit)][event] = []
            for qpoint in range(field.shape[2]):
                account = effect_account(field[mask, event_index, qpoint, :], unit_rows)
                by_unit[str(unit)][event].append(account)
            if unit == 15:
                candidates = range(1, field.shape[2] - 1)
                selection[event] = max(candidates, key=lambda q: by_unit["15"][event][q]["main_share"]["family"])
            q = selection[event]
            selected_passports[unit_index, event_index] = family_passports(field[mask, event_index, q, :], unit_rows, families)
    derived = OUT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    passport_path = derived / "selected_family_passports.float32.npy"
    np.save(passport_path, selected_passports)
    save(OUT / "analysis/all_qpoint_factor_accounting.json", by_unit)
    lockbox_selected = {event: by_unit["16"][event][q] for event, q in selection.items()}
    controls = {
        "record_predicate_interface_share_max": max(x["main_share"]["output_interface"] for x in by_unit["16"]["record_predicate"]),
        "answer_boundary_interface_share_at_selected": lockbox_selected["answer_boundary"]["main_share"]["output_interface"],
        "causal_prefix_control_expected": "record_predicate occurs before interface suffix; exact invariance is expected apart from numerical execution noise",
    }
    checks = {
        "balanced_rows": len(rows) == 768,
        "all_qpoints": all(len(by_unit[u][e]) == 38 for u in by_unit for e in events),
        "physical_coordinates_2560": field.shape[-1] == 2560,
        "selection_confirmation_only": all(1 <= q <= 36 for q in selection.values()),
        "lockbox_same_qpoint": True,
        "prefix_interface_control_small": controls["record_predicate_interface_share_max"] < 1e-5,
        "finite": all(np.isfinite(x["total_centered_energy"]) for u in by_unit.values() for e in u.values() for x in e),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "selection": selection,
        "lockbox_selected": lockbox_selected, "controls": controls,
        "passports": {"path": str(passport_path), "shape": list(selected_passports.shape),
                      "axes": ["unit15_or_unit16", "event", "family", "coordinate"],
                      "events": events, "families": families},
        "adjudication": {"factor_accounting_is_descriptive": True, "semantic_percent_established": False,
                         "format_percent_established": False, "language_encoding_mechanism_closed": False},
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
