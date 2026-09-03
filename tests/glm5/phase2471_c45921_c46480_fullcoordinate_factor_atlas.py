#!/usr/bin/env python3
"""Elementary full-coordinate factor accounting and frozen response ordering."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2470 = next(RESULT.glob("phase2470_*"))
OUT = RESULT / "phase2471_c45921_c46480_fullcoordinate_factor_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2471, "C45921-C46480", 2560
EVENTS = ("statement_target", "candidate_target", "answer_boundary")
QPOINTS = tuple(range(38))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-30 else 0.0


def derangements(count: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < count:
        permutation = rng.permutation(size)
        if np.all(permutation != np.arange(size)):
            result.append(permutation)
    return np.stack(result)


def extract_events() -> tuple[np.memmap, list[dict], dict]:
    final = json.loads((P2470 / "analysis/final.json").read_text(encoding="utf-8"))
    source = np.load(final["collection"]["field"], mmap_mode="r")
    index = read_jsonl(Path(final["collection"]["index"]))
    derived = OUT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    event_path = derived / "event_states.float32.npy"
    states = np.lib.format.open_memmap(event_path, mode="w+", dtype=np.float32, shape=(len(index), len(EVENTS), 38, DIM))
    event_index = []
    for row_number, row in enumerate(index):
        target_spans = row["semantic_spans"][row["target"]]
        local_tokens = [target_spans[0][1] - 1, target_spans[-1][1] - 1, row["answer_boundary_local_token"]]
        global_tokens = [row["token_offset"][0] + token for token in local_tokens]
        for event, token in enumerate(global_tokens):
            states[row_number, event] = np.asarray(source[token], dtype=np.float32)
        event_index.append({**row, "event_local_tokens": dict(zip(EVENTS, local_tokens)), "event_global_tokens": dict(zip(EVENTS, global_tokens))})
    states.flush(); close(source)
    event_index_path = OUT / "index/event_rows.jsonl"
    event_index_path.parent.mkdir(parents=True, exist_ok=True)
    event_index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in event_index), encoding="utf-8")
    delta_path = derived / "event_layer_increments.float32.npy"
    delta = np.lib.format.open_memmap(delta_path, mode="w+", dtype=np.float32, shape=(len(index), len(EVENTS), 37, DIM))
    delta[:] = states[:, :, 1:] - states[:, :, :-1]
    delta.flush(); close(delta)
    return states, event_index, {"event_states": str(event_path), "layer_increments": str(delta_path), "event_index": str(event_index_path), "state_shape": list(states.shape), "delta_shape": [len(index), len(EVENTS), 37, DIM]}


def factor_accounting(states: np.ndarray, index: list[dict]) -> tuple[dict, dict[str, np.ndarray]]:
    factors = {
        "family": sorted({row["family"] for row in index}),
        "unit": sorted({int(row["unit"]) for row in index}),
        "language": sorted({row["language"] for row in index}),
        "surface": sorted({int(row["surface"]) for row in index}),
        "output_interface": sorted({row["output_interface"] for row in index}),
    }
    grand = np.mean(states, axis=0, dtype=np.float64).astype(np.float32)
    effect_arrays: dict[str, np.ndarray] = {}
    row_codes: dict[str, np.ndarray] = {}
    for factor, levels in factors.items():
        codes = np.asarray([levels.index(row[factor]) for row in index], dtype=np.int32)
        row_codes[factor] = codes
        effect = np.zeros((len(levels), len(EVENTS), 38, DIM), dtype=np.float32)
        for level in range(len(levels)):
            effect[level] = np.mean(states[codes == level], axis=0, dtype=np.float64).astype(np.float32) - grand
        effect_arrays[factor] = effect
        np.save(OUT / f"derived/{factor}_main_effect.float32.npy", effect)
    total = np.mean(np.square(states - grand[None, ...], dtype=np.float64), axis=(0, 3))
    energies = {factor: np.mean(np.square(effect_arrays[factor][row_codes[factor]], dtype=np.float64), axis=(0, 3)) for factor in factors}
    residual = np.asarray(states, dtype=np.float32) - grand[None, ...]
    for factor in factors:
        residual -= effect_arrays[factor][row_codes[factor]]
    residual_energy = np.mean(np.square(residual, dtype=np.float64), axis=(0, 3))
    shares = {}
    for event, event_name in enumerate(EVENTS):
        shares[event_name] = {}
        for qpoint in QPOINTS:
            denominator = float(total[event, qpoint])
            shares[event_name][f"q{qpoint}"] = {
                **{factor: float(energies[factor][event, qpoint] / denominator) if denominator > 0 else 0.0 for factor in factors},
                "unmodeled_interactions": float(residual_energy[event, qpoint] / denominator) if denominator > 0 else 0.0,
                "total_mean_square": denominator,
            }
    np.save(OUT / "derived/grand_mean.float32.npy", grand)
    return {"levels": factors, "energy_shares": shares}, effect_arrays


def family_contrasts(states: np.ndarray, index: list[dict], families: list[str]) -> tuple[np.ndarray, dict]:
    units, languages, interfaces = [9, 10], ["en", "zh"], ["code", "entity"]
    result = np.zeros((2, 2, 2, len(families), len(EVENTS), 38, DIM), dtype=np.float32)
    for ui, unit in enumerate(units):
        for li, language in enumerate(languages):
            for oi, interface in enumerate(interfaces):
                group = np.asarray([i for i, row in enumerate(index) if row["unit"] == unit and row["language"] == language and row["output_interface"] == interface])
                baseline = np.mean(states[group], axis=0, dtype=np.float64).astype(np.float32)
                for fi, family in enumerate(families):
                    chosen = np.asarray([i for i in group if index[i]["family"] == family])
                    result[ui, li, oi, fi] = np.mean(states[chosen], axis=0, dtype=np.float64).astype(np.float32) - baseline
    path = OUT / "derived/family_contrasts.float32.npy"
    np.save(path, result)
    return result, {"path": str(path), "shape": list(result.shape), "axes": ["unit(9,10)", "language(en,zh)", "interface(code,entity)", "family", "event", "qpoint", "coordinate"]}


def matched_null(a: np.ndarray, b: np.ndarray, permutations: np.ndarray) -> dict:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-30)
    similarity = x @ y.T
    matched = float(np.mean(np.diag(similarity)))
    null = np.mean(similarity[np.arange(a.shape[0])[None, :], permutations], axis=1)
    return {"coordinate": matched, "family_null_mean": float(np.mean(null)), "family_null_q95": float(np.quantile(null, 0.95)), "family_identity_advantage": matched - float(np.quantile(null, 0.95))}


def replication_atlas(contrasts: np.ndarray) -> dict:
    permutations = derangements(256, contrasts.shape[3], 2471)
    atlas = {}
    for event, event_name in enumerate(EVENTS):
        atlas[event_name] = {}
        for qpoint in QPOINTS:
            raw = {}
            for ui, unit in enumerate((9, 10)):
                cross_language = [matched_null(contrasts[ui, 0, oi, :, event, qpoint], contrasts[ui, 1, oi, :, event, qpoint], permutations) for oi in range(2)]
                raw[f"unit{unit}_crosslanguage"] = {key: float(np.mean([x[key] for x in cross_language])) for key in cross_language[0]}
            cross_unit = [matched_null(contrasts[0, li, oi, :, event, qpoint], contrasts[1, li, oi, :, event, qpoint], permutations) for li in range(2) for oi in range(2)]
            cross_interface = [matched_null(contrasts[ui, li, 0, :, event, qpoint], contrasts[ui, li, 1, :, event, qpoint], permutations) for ui in range(2) for li in range(2)]
            raw["crossunit"] = {key: float(np.mean([x[key] for x in cross_unit])) for key in cross_unit[0]}
            raw["crossinterface"] = {key: float(np.mean([x[key] for x in cross_interface])) for key in cross_interface[0]}
            atlas[event_name][f"q{qpoint}"] = raw
    discovery = atlas["answer_boundary"]
    selected_raw = max(QPOINTS, key=lambda q: discovery[f"q{q}"]["unit9_crosslanguage"]["family_identity_advantage"])
    # Layer increment comparison uses contrast(q)-contrast(q-1), preserving every coordinate.
    delta_metrics = {}
    for qpoint in range(1, 38):
        delta9 = contrasts[0, :, :, :, 2, qpoint] - contrasts[0, :, :, :, 2, qpoint - 1]
        delta10 = contrasts[1, :, :, :, 2, qpoint] - contrasts[1, :, :, :, 2, qpoint - 1]
        unit9 = [matched_null(delta9[0, oi], delta9[1, oi], permutations) for oi in range(2)]
        unit10 = [matched_null(delta10[0, oi], delta10[1, oi], permutations) for oi in range(2)]
        delta_metrics[f"q{qpoint-1}_to_q{qpoint}"] = {
            "unit9_crosslanguage": {key: float(np.mean([x[key] for x in unit9])) for key in unit9[0]},
            "unit10_crosslanguage": {key: float(np.mean([x[key] for x in unit10])) for key in unit10[0]},
        }
    selected_delta = max(range(1, 38), key=lambda q: delta_metrics[f"q{q-1}_to_q{q}"]["unit9_crosslanguage"]["family_identity_advantage"])
    return {
        "all_events_qpoints": atlas,
        "discovery_selection": {"raw_qpoint": selected_raw, "delta_transition": [selected_delta - 1, selected_delta]},
        "lockbox": {
            "raw_unit10_crosslanguage": atlas["answer_boundary"][f"q{selected_raw}"]["unit10_crosslanguage"],
            "delta_unit10_crosslanguage": delta_metrics[f"q{selected_delta-1}_to_q{selected_delta}"]["unit10_crosslanguage"],
        },
        "delta_answer_boundary": delta_metrics,
    }


def fingerprint_order(contrasts: np.ndarray, selected_qpoint: int) -> dict:
    discovery = np.mean(contrasts[0, :, :, :, 2, selected_qpoint], axis=(0, 1))  # family, coordinate
    lockbox = np.mean(contrasts[1, :, :, :, 2, selected_qpoint], axis=(0, 1))
    sign_code = np.sum((discovery > 0).astype(np.int64) * (2 ** np.arange(discovery.shape[0], dtype=np.int64))[:, None], axis=0)
    energy = np.mean(np.square(discovery, dtype=np.float64), axis=0)
    order = np.lexsort((-energy, sign_code)).astype(np.int32)
    path = OUT / "derived/discovery_coordinate_fingerprint_order.int32.npy"
    np.save(path, order)
    threshold = float(np.sqrt(np.mean(np.square(discovery, dtype=np.float64))) * 1e-3)
    mask = (np.abs(discovery) > threshold) | (np.abs(lockbox) > threshold)
    return {
        "path": str(path),
        "coordinates": len(order),
        "rule": "unit9 answer-boundary family-sign code, then descending full-family energy; all coordinates retained",
        "lockbox_flattened_correlation": cosine(discovery, lockbox),
        "lockbox_sign_agreement_above_floor": float(np.mean(np.sign(discovery[mask]) == np.sign(lockbox[mask]))),
        "floor": threshold,
        "order_frozen_before_unit10": True,
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 全坐标五因素分账、三语义事件、层增量与冻结响应指纹图谱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 从Phase2470全token原场抽取statement中目标末token、候选区目标末token、answer-boundary三事件的q0–q37全2560坐标。平衡设计按family、unit/内容、语言、surface、输出接口计算主效应能量和未建模交互；另在每个unit×语言×接口内构造八族相对基线全坐标contrast。unit9只用于选择最佳原状态qpoint、最佳层增量转移及全2560坐标指纹排序，unit10保持冻结。排序由八族响应正负码及全族能量确定，但不删除任何坐标。

$$H=B+F_f+U_u+L_\lambda+S_s+O_o+R_{{interaction}},\qquad \Delta H_\ell=H_\ell-H_{{\ell-1}}.$$

**结果汇总。** 事件原场/增量 `{json.dumps(result['collection'], ensure_ascii=False)}`；发现选择与锁箱 `{json.dumps(result['replication']['discovery_selection'], ensure_ascii=False)}`、`{json.dumps(result['replication']['lockbox'], ensure_ascii=False)}`；冻结指纹 `{json.dumps(result['fingerprint'], ensure_ascii=False)}`；关键能量分账 `{json.dumps(result['selected_energy_accounting'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2471_c45921_c46480_fullcoordinate_factor_atlas.py`；事件状态、层增量、各主效应、八族contrast、冻结全坐标顺序与final位于同名结果目录。

**分析与理论进展。** 本Phase不用低秩压缩回答“family在哪里”，而是先问family全坐标纹理能否在未见内容unit10复现，层的新增变化是否比静态状态更稳定，以及输出接口/语言/表面效应占多大。响应指纹顺序只服务可视化，锁箱不能重新排序，因此不会凭unit10结果制造条带。

**问题硬伤与结论。** 主效应分账是平衡设计的描述账本，不意味着网络按线性项相加；残差包含所有交互和token身份。family contrast仍可能含family专属措辞。坐标指纹排序不是天然模块证明，也不是Top-K选择。只有锁箱family身份优势为正，才能把相应原状态或层增量升级为L1候选。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    states, index, collection = extract_events()
    accounting, _ = factor_accounting(states, index)
    families = accounting["levels"]["family"]
    contrasts, contrast_meta = family_contrasts(states, index, families)
    replication = replication_atlas(contrasts)
    fingerprint = fingerprint_order(contrasts, replication["discovery_selection"]["raw_qpoint"])
    selected_q = replication["discovery_selection"]["raw_qpoint"]
    selected_energy = {
        event: accounting["energy_shares"][event][f"q{selected_q}"] for event in EVENTS
    }
    checks = {
        "event_shape": collection["state_shape"] == [256, 3, 38, 2560],
        "delta_shape": collection["delta_shape"] == [256, 3, 37, 2560],
        "balanced_factors": all(len(accounting["levels"][key]) == expected for key, expected in (("family", 8), ("unit", 2), ("language", 2), ("surface", 4), ("output_interface", 2))),
        "all_coordinates_ordered": fingerprint["coordinates"] == 2560,
        "discovery_only_order": fingerprint["order_frozen_before_unit10"],
        "finite": all(math.isfinite(value) for event in selected_energy.values() for value in event.values()),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "collection": {**collection, "family_contrasts": contrast_meta},
        "factor_accounting": accounting,
        "selected_energy_accounting": selected_energy,
        "replication": replication,
        "fingerprint": fingerprint,
        "adjudication": {"factor_accounting_is_mechanism": False, "frozen_fullcoordinate_order_is_visualization_only": True, "language_encoding_mechanism_closed": False},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    close(states)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
