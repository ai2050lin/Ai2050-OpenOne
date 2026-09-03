#!/usr/bin/env python3
"""Full-coordinate dual flagship factorial atlas and held-out template tournament."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2368 = RESULT / "phase2368_c12481_c12720_longrange_operator_contract"
P2369 = RESULT / "phase2369_c12721_c13040_qwen_longrange_full_field"
OUT = RESULT / "phase2372_c13681_c14000_dual_flagship_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2368 / "material/flagship_factorial.jsonl"
STATES = P2369 / "raw/qwen4b_flagship_boundary_all_layers.float16.npy"
DECISIONS = P2369 / "raw/qwen4b_flagship_first_token_decisions.float32.npy"
PHASE = 2372
CAMPAIGN = "C13681-C14000"
SYSTEMS = ("attitude_role", "taxonomy_chain")
ORDERS = np.asarray([s.bit_count() for s in range(32)])
SIGNS = np.asarray([[-1.0 if ((cell & subset).bit_count() % 2) else 1.0
                     for subset in range(32)] for cell in range(32)], dtype=np.float32)
FORWARD = SIGNS.T / 32


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Counter): return dict(value)
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def response_r2(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, np.ndarray]:
    a = actual.reshape(-1, actual.shape[-1]).astype(np.float64)
    p = predicted.reshape(-1, predicted.shape[-1]).astype(np.float64)
    sse = np.square(a - p).sum(0); sst = np.square(a - a.mean(0, keepdims=True)).sum(0)
    if float(sst.sum()) < 1e-8:
        return float("nan"), np.full(actual.shape[-1], np.nan, dtype=np.float32)
    total = 1 - float(sse.sum() / float(sst.sum()))
    return total, np.where(sst > 1e-12, 1 - sse / np.maximum(sst, 1e-12), np.nan).astype(np.float32)


def build_group_index(rows: list[dict]) -> tuple[list[tuple], np.ndarray]:
    groups = {}
    for i, r in enumerate(rows):
        key = (r["system"], r["unit"], r["language"], r["surface"])
        groups.setdefault(key, {})[r["cell"]] = i
    keys = sorted(groups, key=str)
    if not all(len(groups[k]) == 32 for k in keys): raise RuntimeError("incomplete cubes")
    return keys, np.asarray([[groups[k][cell] for cell in range(32)] for k in keys])


def split_indices(keys: list[tuple], system: str) -> dict[str, np.ndarray]:
    return {
        "train": np.asarray([i for i, k in enumerate(keys) if k[0] == system and k[1] <= 5]),
        "confirmation": np.asarray([i for i, k in enumerate(keys) if k[0] == system and 6 <= k[1] <= 7]),
        "lockbox": np.asarray([i for i, k in enumerate(keys) if k[0] == system and k[1] >= 8]),
    }


def coefficients(field: np.ndarray) -> np.ndarray:
    return np.einsum("sc,gcd->gsd", FORWARD, field, optimize=True)


def template_for(key: tuple, train_keys: list[tuple], train_coeff: np.ndarray, candidate: str,
                 other_global: np.ndarray) -> np.ndarray:
    if candidate == "global": select = np.ones(len(train_keys), dtype=bool)
    elif candidate == "language": select = np.asarray([k[2] == key[2] for k in train_keys])
    elif candidate == "surface": select = np.asarray([k[3] == key[3] for k in train_keys])
    elif candidate == "language_surface": select = np.asarray([k[2] == key[2] and k[3] == key[3] for k in train_keys])
    elif candidate == "other_system_global": return other_global
    else: raise KeyError(candidate)
    return train_coeff[select].mean(0)


def reconstruct_response(coeff: np.ndarray, max_order: int) -> np.ndarray:
    keep = (ORDERS > 0) & (ORDERS <= max_order)
    response = SIGNS[:, keep] @ coeff[keep]
    return response - response[0]


def evaluate(field: np.ndarray, keys: list[tuple], splits: dict[str, np.ndarray], other_train: np.ndarray,
             candidate: str, max_order: int, split: str) -> tuple[float, np.ndarray]:
    train_keys = [keys[i] for i in splits["train"]]
    train_coeff = coefficients(field[splits["train"]])
    other_global = coefficients(field[other_train]).mean(0)
    actual, predicted = [], []
    for gi in splits[split]:
        response = field[gi] - field[gi, 0]
        template = template_for(keys[gi], train_keys, train_coeff, candidate, other_global)
        actual.append(response[1:]); predicted.append(reconstruct_response(template, max_order)[1:])
    return response_r2(np.stack(actual), np.stack(predicted))


def behavior(rows: list[dict]) -> dict:
    d = np.load(DECISIONS, mmap_mode="r")
    out = {"overall_target_over_foil": float(np.asarray(d[:, 3]).mean()),
           "overall_argmax": float(np.asarray(d[:, 4]).mean()), "systems": {}}
    for system in SYSTEMS:
        idx = [i for i, r in enumerate(rows) if r["system"] == system]
        cells = {}
        for language in ("en", "zh"):
            for query in sorted(set(r["query"] for r in rows if r["system"] == system)):
                sub = [i for i in idx if rows[i]["language"] == language and rows[i]["query"] == query]
                if sub: cells[f"{language}:{query}"] = {"n": len(sub), "target_over_foil": float(np.asarray(d[sub, 3]).mean()),
                                                          "argmax": float(np.asarray(d[sub, 4]).mean())}
        out["systems"][system] = {"target_over_foil": float(np.asarray(d[idx, 3]).mean()),
                                  "argmax": float(np.asarray(d[idx, 4]).mean()), "cells": cells,
                                  "minimum_cell_target_over_foil": min(v["target_over_foil"] for v in cells.values())}
    return out


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 双旗舰语言族全坐标图谱与跨系统锁箱竞赛（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对“我/她—肯定/否定—喜欢/偏爱—吃/买—苹果/梨”的态度角色立方，以及“苹果/梨—关系措辞—中间边—支路—查询深度”的知识链立方，使用12个独立unit、中英、直接事实/独立叙述、32个全析因cell。unit0–5训练、6–7确认、8–11锁箱。对每个系统×层×坐标做完整Walsh响应，不以Top-K/PCA为主表示；全球、语言、表述、语言×表述、另一系统五种模板和1/2/3/5阶在相同确认集上选择，再一次性裁决锁箱。

$$
\widehat H_S=\frac1{{32}}\sum_{{x\in(\mathbb Z_2)^5}}\chi_S(x)H(x),\qquad
\widehat R_{{\leq k}}(x)=\sum_{{1\leq|S|\leq k}}[\chi_S(x)-\chi_S(0)]\overline H_S.
$$

**结果汇总。** 行为分层 `{json.dumps(result['behavior'], ensure_ascii=False)}`；两个系统的冻结选择与锁箱 `{json.dumps(result['system_summaries'], ensure_ascii=False)}`；阶数能量 `{json.dumps(result['selected_layer_order_energy'], ensure_ascii=False)}`；跨系统边界 `{json.dumps(result['cross_system_boundary'], ensure_ascii=False)}`。补充纠错：Phase2371第一次写入MEMO的CMI使用了固定$\sigma$，使槽位在给定句身份后无变化；最终结果已改用fresh unit的两种来源顺序，$n=1536$，实际中位0.12846 bit、置乱中位0.01386 bit。

**相关文件。** 脚本 `tests/glm5/phase2372_c13681_c14000_dual_flagship_atlas.py`；全局Walsh坐标系数、逐坐标锁箱R²、逐层竞赛与汇总位于 `tests/glm5/result/phase2372_c13681_c14000_dual_flagship_atlas`。

**理论进展、问题硬伤与结论。** Walsh仍只是冻结立方的精确坐标变换；跨unit正R²才是复用证据，跨旗舰成功才可能支持更普遍的形式规律。两个旗舰的foil都含“unknown”类对照，行为门会受词频和答案格式影响，因此只作材料资格而非语义理解证明。若另一系统模板失败，结论是这两个外部五因子坐标之间没有发现可搬运的统一响应模板，不是否定有限参数中的更深共享机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(MATERIAL); states = np.load(STATES, mmap_mode="r")
    keys, row_index = build_group_index(rows)
    splits = {system: split_indices(keys, system) for system in SYSTEMS}
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    global_coeff = np.lib.format.open_memmap(OUT / "derived/flagship_global_factorial_coefficients.float16.npy",
                                             mode="w+", dtype=np.float16, shape=(2, 38, 32, 2560))
    order_energy = np.zeros((2, 3, 38, 6), dtype=np.float64)
    layer_tournament = {system: [] for system in SYSTEMS}
    candidates = ("global", "language", "surface", "language_surface", "other_system_global")
    order_choices = (1, 2, 3, 5)
    coordinate_results = {}
    for qpoint in range(38):
        field = np.asarray(states[row_index, qpoint], dtype=np.float32)
        coeff_all = coefficients(field)
        for si, system in enumerate(SYSTEMS):
            split = splits[system]; other = splits[SYSTEMS[1 - si]]["train"]
            train_coeff = coeff_all[split["train"]]
            global_coeff[si, qpoint] = train_coeff.mean(0).astype(np.float16)
            for spi, split_name in enumerate(("train", "confirmation", "lockbox")):
                coeff = coeff_all[split[split_name]]
                energy = np.square(coeff.astype(np.float64)).sum((0, 2))
                for order in range(6): order_energy[si, spi, qpoint, order] = energy[ORDERS == order].sum()
            entries = []
            for candidate in candidates:
                for order in order_choices:
                    rc, _ = evaluate(field, keys, split, other, candidate, order, "confirmation")
                    rl, _ = evaluate(field, keys, split, other, candidate, order, "lockbox")
                    entries.append({"candidate": candidate, "order": order, "confirmation_response_r2": rc,
                                    "lockbox_response_r2": rl})
            layer_tournament[system].append({"qpoint": qpoint, "entries": entries})
        global_coeff.flush(); print(f"[phase2372] qpoint {qpoint}/37", flush=True)
    np.save(OUT / "derived/flagship_order_energy.float64.npy", order_energy)
    summaries = {}
    rng = np.random.default_rng(2372); coord_perm = rng.permutation(2560)
    for si, system in enumerate(SYSTEMS):
        choices = [(e["confirmation_response_r2"], layer["qpoint"], e) for layer in layer_tournament[system] for e in layer["entries"]]
        _, qpoint, selected = max((x for x in choices if np.isfinite(x[0])), key=lambda x: x[0])
        field = np.asarray(states[row_index, qpoint], dtype=np.float32); split = splits[system]; other = splits[SYSTEMS[1 - si]]["train"]
        lock_r2, coord_r2 = evaluate(field, keys, split, other, selected["candidate"], selected["order"], "lockbox")
        # Address control: permute every coordinate of the selected coefficient template while keeping targets physical.
        train_coeff = coefficients(field[split["train"]]); other_global = coefficients(field[other]).mean(0)
        actual, permuted_pred = [], []
        train_keys = [keys[i] for i in split["train"]]
        for gi in split["lockbox"]:
            template = template_for(keys[gi], train_keys, train_coeff, selected["candidate"], other_global)[:, coord_perm]
            actual.append((field[gi] - field[gi, 0])[1:]); permuted_pred.append(reconstruct_response(template, selected["order"])[1:])
        permuted_r2, _ = response_r2(np.stack(actual), np.stack(permuted_pred))
        np.save(OUT / f"derived/{system}_selected_lockbox_coordinate_r2.float32.npy", coord_r2)
        summaries[system] = {"selected_qpoint": qpoint, "selected_candidate": selected["candidate"],
                             "selected_order": selected["order"], "confirmation_response_r2": selected["confirmation_response_r2"],
                             "lockbox_response_r2": lock_r2, "coordinate_permuted_lockbox_r2": permuted_r2,
                             "passed_positive_physical_lockbox": lock_r2 > 0 and lock_r2 > permuted_r2}
    selected_energy = {}
    for si, system in enumerate(SYSTEMS):
        q = summaries[system]["selected_qpoint"]
        selected_energy[system] = {split: {str(order): float(order_energy[si, spi, q, order] / max(order_energy[si, spi, q].sum(), 1e-12))
                                                   for order in range(6)} for spi, split in enumerate(("train", "confirmation", "lockbox"))}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "behavior": behavior(rows),
              "splits": {s: {k: len(v) for k, v in splits[s].items()} for s in SYSTEMS},
              "system_summaries": summaries, "selected_layer_order_energy": selected_energy,
              "cross_system_boundary": {"attitude_other_system_r2": next(e["lockbox_response_r2"] for e in layer_tournament["attitude_role"][summaries["attitude_role"]["selected_qpoint"]]["entries"] if e["candidate"] == "other_system_global" and e["order"] == summaries["attitude_role"]["selected_order"]),
                                        "taxonomy_other_system_r2": next(e["lockbox_response_r2"] for e in layer_tournament["taxonomy_chain"][summaries["taxonomy_chain"]["selected_qpoint"]]["entries"] if e["candidate"] == "other_system_global" and e["order"] == summaries["taxonomy_chain"]["selected_order"]),
                                        "claim": "same bit index across flagships is an external alignment, not established semantic equivalence"}}
    save(OUT / "analysis/layer_tournament.json", layer_tournament); save(final_path, result); append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
