#!/usr/bin/env python3
"""Decode sentence pointers and test S4 adjacent-swap response operators on full coordinates."""
from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2368 = RESULT / "phase2368_c12481_c12720_longrange_operator_contract"
P2369 = RESULT / "phase2369_c12721_c13040_qwen_longrange_full_field"
OUT = RESULT / "phase2370_c13041_c13360_pointer_group_operator"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2368 / "material/long_sentence_permutation.jsonl"
STATES = P2369 / "raw/qwen4b_long_boundary_all_layers.float16.npy"
PHASE = 2370
CAMPAIGN = "C13041-C13360"
PERMS = tuple(itertools.permutations(range(4)))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def swap_perm(p: tuple[int, ...], i: int) -> tuple[int, ...]:
    p = list(p); p[i], p[i + 1] = p[i + 1], p[i]; return tuple(p)


def reduced_word(target: tuple[int, ...]) -> list[int]:
    current, word = [0, 1, 2, 3], []
    for position, wanted in enumerate(target):
        j = current.index(wanted)
        while j > position:
            current[j - 1], current[j] = current[j], current[j - 1]
            word.append(j - 1); j -= 1
    assert tuple(current) == target
    return word


def response_r2(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, np.ndarray]:
    a = actual.reshape(-1, actual.shape[-1]).astype(np.float64)
    p = predicted.reshape(-1, predicted.shape[-1]).astype(np.float64)
    sse = np.square(a - p).sum(axis=0)
    sst = np.square(a - a.mean(axis=0, keepdims=True)).sum(axis=0)
    if float(sst.sum()) < 1e-8:
        return float("nan"), np.full(actual.shape[-1], np.nan, dtype=np.float32)
    total = 1.0 - float(sse.sum() / sst.sum())
    coordinate = np.where(sst > 1e-12, 1.0 - sse / np.maximum(sst, 1e-12), np.nan)
    return total, coordinate.astype(np.float32)


def nearest_centroid(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, scale: bool) -> np.ndarray:
    if scale:
        mean, std = train_x.mean(0), train_x.std(0) + 1e-4
        train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    labels = np.unique(train_y)
    centroids = np.stack([train_x[train_y == label].mean(0) for label in labels])
    scores = test_x @ centroids.T - 0.5 * np.square(centroids).sum(1)[None, :]
    return labels[scores.argmax(1)]


def build_field(rows: list[dict], states: np.ndarray, qpoint: int):
    indices = [i for i, r in enumerate(rows) if r["task"] == "index_only"]
    groups: dict[tuple, dict[int, int]] = {}
    metadata = {}
    for index in indices:
        r = rows[index]
        key = (r["family"], r["unit"], r["language"], tuple(r["source_perm"]))
        groups.setdefault(key, {})[r["permutation_index"]] = index
        metadata[key] = r
    keys = sorted(groups, key=str)
    if not all(len(groups[k]) == 24 for k in keys): raise RuntimeError("incomplete S4 groups")
    row_index = np.asarray([[groups[k][pi] for pi in range(24)] for k in keys])
    field = np.asarray(states[row_index, qpoint], dtype=np.float32)
    return keys, metadata, field, row_index


def split_groups(keys: list[tuple]) -> dict[str, np.ndarray]:
    train = [i for i, k in enumerate(keys) if k[1] <= 2 and k[3] == (0, 1, 2, 3)]
    confirmation = [i for i, k in enumerate(keys) if k[1] == 3 and k[3] == (2, 0, 3, 1)]
    lockbox = [i for i, k in enumerate(keys) if k[1] >= 4 and k[3] == (2, 0, 3, 1)]
    return {"train": np.asarray(train), "confirmation": np.asarray(confirmation), "lockbox": np.asarray(lockbox)}


def labels_for(keys: list[tuple], row_index: np.ndarray, rows: list[dict], group_indices: np.ndarray):
    x_indices, first_identity, first_slot, full_perm = [], [], [], []
    for gi in group_indices:
        source = keys[gi][3]
        for pi, source_index in enumerate(row_index[gi]):
            p = PERMS[pi]
            x_indices.append(source_index); first_identity.append(p[0]); first_slot.append(source.index(p[0])); full_perm.append(pi)
    return np.asarray(x_indices), {"first_sentence_identity": np.asarray(first_identity),
                                   "first_source_slot": np.asarray(first_slot), "full_permutation": np.asarray(full_perm)}


def pointer_layer(states: np.ndarray, rows: list[dict], keys: list[tuple], row_index: np.ndarray,
                  splits: dict[str, np.ndarray], qpoint: int) -> dict:
    index = {}; labels = {}
    for split, groups in splits.items(): index[split], labels[split] = labels_for(keys, row_index, rows, groups)
    tx = np.asarray(states[index["train"], qpoint], dtype=np.float32)
    result = {"qpoint": qpoint, "targets": {}}
    for target in labels["train"]:
        methods = {}
        for scaled in (False, True):
            name = "zscore_centroid" if scaled else "raw_centroid"
            pred_c = nearest_centroid(tx, labels["train"][target], np.asarray(states[index["confirmation"], qpoint], dtype=np.float32), scaled)
            pred_l = nearest_centroid(tx, labels["train"][target], np.asarray(states[index["lockbox"], qpoint], dtype=np.float32), scaled)
            methods[name] = {"confirmation_accuracy": float((pred_c == labels["confirmation"][target]).mean()),
                             "lockbox_accuracy": float((pred_l == labels["lockbox"][target]).mean())}
        selected = max(methods, key=lambda name: methods[name]["confirmation_accuracy"])
        result["targets"][target] = {"chance": 1 / (24 if target == "full_permutation" else 4),
                                     "methods": methods, "selected_on_confirmation": selected,
                                     "selected_lockbox_accuracy": methods[selected]["lockbox_accuracy"]}
    return result


def fit_operators(field: np.ndarray, train_groups: np.ndarray):
    translations, slopes, intercepts = [], [], []
    for generator in range(3):
        target_pi = np.asarray([PERMS.index(swap_perm(p, generator)) for p in PERMS])
        x = field[train_groups].reshape(-1, field.shape[-1])
        y = field[train_groups][:, target_pi].reshape(-1, field.shape[-1])
        xm, ym = x.mean(0), y.mean(0)
        slope = ((x - xm) * (y - ym)).mean(0) / (np.square(x - xm).mean(0) + 1e-5)
        translations.append((y - x).mean(0)); slopes.append(slope); intercepts.append(ym - slope * xm)
    direct = field[train_groups] - field[train_groups, 0][:, None, :]
    return np.stack(translations), np.stack(slopes), np.stack(intercepts), direct.mean(0)


def predict_responses(field: np.ndarray, groups: np.ndarray, translations: np.ndarray,
                      slopes: np.ndarray, intercepts: np.ndarray, direct: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    actual = field[groups] - field[groups, 0][:, None, :]
    predicted = np.empty_like(actual); predicted[:, 0] = 0
    rng = np.random.default_rng(2370)
    coord_perm = rng.permutation(field.shape[-1])
    for pi, target in enumerate(PERMS):
        if pi == 0: continue
        if method == "direct_template":
            predicted[:, pi] = direct[pi]
            continue
        value = field[groups, 0].copy()
        for generator in reduced_word(target):
            if method == "translation": value = value + translations[generator]
            elif method == "diagonal_affine": value = value * slopes[generator] + intercepts[generator]
            elif method == "coordinate_permuted_affine": value = value * slopes[generator, coord_perm] + intercepts[generator, coord_perm]
            elif method == "identity": pass
            else: raise KeyError(method)
        predicted[:, pi] = value - field[groups, 0]
    return actual[:, 1:], predicted[:, 1:]


def group_layer(field: np.ndarray, splits: dict[str, np.ndarray], qpoint: int) -> tuple[dict, np.ndarray, np.ndarray]:
    translations, slopes, intercepts, direct = fit_operators(field, splits["train"])
    methods = ("identity", "translation", "diagonal_affine", "direct_template", "coordinate_permuted_affine")
    result = {"qpoint": qpoint, "methods": {}}
    coord_confirmation, coord_lockbox = [], []
    for method in methods:
        ac, pc = predict_responses(field, splits["confirmation"], translations, slopes, intercepts, direct, method)
        al, pl = predict_responses(field, splits["lockbox"], translations, slopes, intercepts, direct, method)
        r2c, cr2c = response_r2(ac, pc); r2l, cr2l = response_r2(al, pl)
        result["methods"][method] = {"confirmation_response_r2": r2c, "lockbox_response_r2": r2l}
        coord_confirmation.append(cr2c); coord_lockbox.append(cr2l)
    selected = max((m for m in methods if m != "coordinate_permuted_affine"),
                   key=lambda m: result["methods"][m]["confirmation_response_r2"])
    result["selected_on_confirmation"] = selected
    result["selected_lockbox_response_r2"] = result["methods"][selected]["lockbox_response_r2"]
    result["diagonal_beats_direct_lockbox"] = result["methods"]["diagonal_affine"]["lockbox_response_r2"] > result["methods"]["direct_template"]["lockbox_response_r2"]
    result["diagonal_beats_permuted_lockbox"] = result["methods"]["diagonal_affine"]["lockbox_response_r2"] > result["methods"]["coordinate_permuted_affine"]["lockbox_response_r2"]
    return result, np.stack(coord_confirmation), np.stack(coord_lockbox)


def group_relations(field: np.ndarray) -> dict:
    residual = field - field[:, 0][:, None, :]
    norms = np.linalg.norm(residual, axis=-1)
    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))
    t1t3 = PERMS.index(swap_perm(swap_perm(PERMS[0], 0), 2))
    t3t1 = PERMS.index(swap_perm(swap_perm(PERMS[0], 2), 0))
    braid_a = PERMS.index(swap_perm(swap_perm(swap_perm(PERMS[0], 0), 1), 0))
    braid_b = PERMS.index(swap_perm(swap_perm(swap_perm(PERMS[0], 1), 0), 1))
    return {"identity_response_norm": float(norms[:, 0].mean()),
            "commuting_final_prompt_relative_error": rel(field[:, t1t3], field[:, t3t1]),
            "braid_final_prompt_relative_error": rel(field[:, braid_a], field[:, braid_b]),
            "boundary": "These equalities compare identical final permutations, not hidden sequential paths."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 句对象指针、来源槽位与$S_4$生成元算子基础检验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 只使用`index_only`的全部4608条$S_4$场。训练仅见unit0–2和来源顺序$\sigma=e$；unit3与新来源顺序作确认，unit4–5与新来源顺序作锁箱。全2560坐标最近质心解码首句对象身份、其来源槽位及完整24类排列。另从相邻交换$\tau_1,\tau_2,\tau_3$拟合逐坐标仿射响应，并与零响应、平移、按目标排列直接均值模板和坐标置乱控制在相同目标/样本上比较。

$$
h_{{\tau_i\pi,j}}=a_{{i,j}}h_{{\pi,j}}+b_{{i,j}}+\epsilon_{{i,\pi,j}},\qquad
\widehat h_\pi=T_{{i_m}}\circ\cdots\circ T_{{i_1}}(h_e).
$$

**结果汇总。** 指针最佳层与锁箱 `{json.dumps(result['pointer_summary'], ensure_ascii=False)}`；生成元算子选择与锁箱 `{json.dumps(result['operator_summary'], ensure_ascii=False)}`；群关系统计 `{json.dumps(result['group_relations'], ensure_ascii=False)}`；分割 `{json.dumps(result['splits'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2370_c13041_c13360_pointer_group_operator.py`；逐层结果、逐坐标R²和算子参数位于 `tests/glm5/result/phase2370_c13041_c13360_pointer_group_operator`。

**理论进展、问题硬伤与结论。** 可解码对象/槽位只说明边界场保留了答案相关绑定信息，因为prompt本身显式包含目标标记；它不是自主指针机的充分证据。逐坐标仿射若不能在fresh-unit和新$\sigma$上超过按排列直接模板，就应拒绝“统一相邻交换算子”；即便超过，也只是响应表示候选。交换/辫关系的两条最终prompt完全相同，因此其数值相等是材料恒等式，不是内部逐步路径闭合。下一Phase用群傅里叶、张量、OT、条件信息和拓扑在同一锁箱竞争，不把任何高等数学名称预先升级为机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(MATERIAL); states = np.load(STATES, mmap_mode="r")
    keys, _metadata, field0, row_index = build_field(rows, states, 0)
    splits = split_groups(keys)
    pointer_layers, operator_layers = [], []
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    coord_c = np.lib.format.open_memmap(OUT / "derived/operator_coordinate_confirmation_r2.float32.npy", mode="w+", dtype=np.float32, shape=(38, 5, 2560))
    coord_l = np.lib.format.open_memmap(OUT / "derived/operator_coordinate_lockbox_r2.float32.npy", mode="w+", dtype=np.float32, shape=(38, 5, 2560))
    for qpoint in range(38):
        keys_q, _, field, row_index_q = build_field(rows, states, qpoint)
        if keys_q != keys or not np.array_equal(row_index_q, row_index): raise RuntimeError("group order changed")
        pointer_layers.append(pointer_layer(states, rows, keys, row_index, splits, qpoint))
        metric, cc, cl = group_layer(field, splits, qpoint)
        operator_layers.append(metric); coord_c[qpoint] = cc; coord_l[qpoint] = cl
        coord_c.flush(); coord_l.flush(); print(f"[phase2370] qpoint {qpoint}/37", flush=True)
    methods = ("raw_centroid", "zscore_centroid")
    pointer_summary = {}
    for target in ("first_sentence_identity", "first_source_slot", "full_permutation"):
        choices = []
        for layer in pointer_layers:
            for method in methods:
                choices.append((layer["targets"][target]["methods"][method]["confirmation_accuracy"], layer["qpoint"], method))
        _, qpoint, method = max(choices)
        pointer_summary[target] = {"selected_qpoint": qpoint, "selected_method": method,
            "confirmation_accuracy": pointer_layers[qpoint]["targets"][target]["methods"][method]["confirmation_accuracy"],
            "lockbox_accuracy": pointer_layers[qpoint]["targets"][target]["methods"][method]["lockbox_accuracy"],
            "chance": pointer_layers[qpoint]["targets"][target]["chance"]}
    method_names = ("identity", "translation", "diagonal_affine", "direct_template")
    choices = [(layer["methods"][method]["confirmation_response_r2"], layer["qpoint"], method)
               for layer in operator_layers for method in method_names
               if np.isfinite(layer["methods"][method]["confirmation_response_r2"])]
    _, oq, om = max(choices)
    operator_summary = {"selected_qpoint": oq, "selected_method": om,
                        "confirmation_response_r2": operator_layers[oq]["methods"][om]["confirmation_response_r2"],
                        "lockbox_response_r2": operator_layers[oq]["methods"][om]["lockbox_response_r2"],
                        "same_layer_methods": operator_layers[oq]["methods"],
                        "uniform_adjacent_operator_candidate_passed": om in ("translation", "diagonal_affine")
                        and operator_layers[oq]["methods"][om]["lockbox_response_r2"] > operator_layers[oq]["methods"]["direct_template"]["lockbox_response_r2"]
                        and operator_layers[oq]["methods"][om]["lockbox_response_r2"] > 0}
    result = {"phase": PHASE, "campaign": CAMPAIGN,
              "splits": {k: int(len(v)) for k, v in splits.items()}, "pointer_summary": pointer_summary,
              "operator_summary": operator_summary, "group_relations": group_relations(field0),
              "method_order_coordinate_arrays": ["identity", "translation", "diagonal_affine", "direct_template", "coordinate_permuted_affine"],
              "interpretation_boundary": "Decoding and response prediction are observational/predictive tests, not proof of an explicit internal permutation group."}
    save(OUT / "analysis/pointer_layers.json", pointer_layers); save(OUT / "analysis/operator_layers.json", operator_layers); save(final_path, result)
    append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
