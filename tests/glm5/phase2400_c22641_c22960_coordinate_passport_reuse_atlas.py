#!/usr/bin/env python3
"""Build full-coordinate passports for reuse, differentiation, persistence, and low-magnitude coverage."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2398 = RESULT / "phase2398_c22001_c22320_qwen4b_event_fullfield"
P2399 = RESULT / "phase2399_c22321_c22640_local_update_operator_atlas"
OUT = RESULT / "phase2400_c22641_c22960_coordinate_passport_reuse_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2400
CAMPAIGN = "C22641-C22960"
Q_UPDATES = 36


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1); b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def group_means(y: np.ndarray, rows: list[dict], indices: np.ndarray, keys: tuple[str, ...]) -> dict[tuple, np.ndarray]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for local, source in enumerate(indices.tolist()):
        groups[tuple(rows[source].get(key) for key in keys)].append(local)
    return {key: y[local].mean(0) for key, local in groups.items()}


def family_passport(y: np.ndarray, rows: list[dict], indices: np.ndarray, families: list[str]) -> np.ndarray:
    center = y[indices].mean(0)
    return np.stack([y[[i for i in indices if rows[i]["family"] == family]].mean(0) - center for family in families])


def centered_factor_family(y: np.ndarray, rows: list[dict], indices: np.ndarray, families: list[str], factor: str, value: Any) -> np.ndarray:
    selected = np.asarray([i for i in indices if rows[i].get(factor) == value], dtype=np.int64)
    center = y[selected].mean(0)
    return np.stack([y[[i for i in selected if rows[i]["family"] == family]].mean(0) - center for family in families])


def task_contract(task: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], tuple[str, ...]]:
    if task == "selection":
        train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
        confirmation = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "confirmation"], dtype=np.int64)
        lockbox = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_unit_lockbox"], dtype=np.int64)
        keys = ("family", "language", "surface", "direction", "query_role", "target_candidate_slot")
    else:
        train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
        confirmation = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "confirmation"], dtype=np.int64)
        lockbox = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "fresh_unit_lockbox"], dtype=np.int64)
        keys = ("family", "language", "surface", "direction", "target_candidate_slot")
    families = sorted({row["family"] for row in rows})
    return train, confirmation, lockbox, families, keys


def analyze_task(task: str, field_path: Path, rows: list[dict]) -> dict:
    field = np.load(field_path, mmap_mode="r")
    train, confirmation, lockbox, families, condition_keys = task_contract(task, rows)
    event_count, dimension = field.shape[2], field.shape[3]
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    passport_paths = {split: OUT / f"derived/{task}_family_passport_{split}.float32.npy" for split in ("discovery", "confirmation", "lockbox")}
    passports = {split: np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                                  shape=(Q_UPDATES, event_count, len(families), dimension))
                 for split, path in passport_paths.items()}
    group_ids = np.lib.format.open_memmap(OUT / f"derived/{task}_family_sign_group.uint8.npy", mode="w+", dtype=np.uint8,
                                          shape=(Q_UPDATES, event_count, dimension))
    group_stability = np.lib.format.open_memmap(OUT / f"derived/{task}_family_sign_stability.float32.npy", mode="w+", dtype=np.float32,
                                                shape=(Q_UPDATES, event_count, dimension))
    coordinate_gain = np.lib.format.open_memmap(OUT / f"derived/{task}_condition_gain_lockbox.float32.npy", mode="w+", dtype=np.float32,
                                                shape=(Q_UPDATES, event_count, dimension))
    reuse_matrix = np.lib.format.open_memmap(OUT / f"derived/{task}_family_reuse_cosine.float32.npy", mode="w+", dtype=np.float32,
                                              shape=(Q_UPDATES, event_count, len(families), len(families)))
    factor_cosines: dict[str, dict[str, list[float]]] = {factor: {family: [] for family in families}
                                                       for factor in ("surface", "language", "direction", "target_candidate_slot") + (("query_role",) if task == "selection" else ())}
    partition_cosines = {family: [] for family in families}
    gain_quartiles = [0.0, 0.0, 0.0, 0.0]
    gain_positive_quartiles = [0.0, 0.0, 0.0, 0.0]
    gain_total = 0.0
    group_counts: Counter[int] = Counter()
    slice_summary: list[dict] = []
    rms = np.load(P2399 / f"derived/{task}_update_discovery_rms.float32.npy", mmap_mode="r")
    for qpoint in range(Q_UPDATES):
        for event in range(event_count):
            x = np.asarray(field[:, qpoint, event], dtype=np.float32)
            y = np.asarray(field[:, qpoint + 1, event], dtype=np.float32) - x
            p_train = family_passport(y, rows, train, families)
            p_confirmation = family_passport(y, rows, confirmation, families)
            p_lock = family_passport(y, rows, lockbox, families)
            passports["discovery"][qpoint, event] = p_train
            passports["confirmation"][qpoint, event] = p_confirmation
            passports["lockbox"][qpoint, event] = p_lock
            for family_index, family in enumerate(families):
                partition_cosines[family].append(cosine(p_train[family_index], p_lock[family_index]))
            codes = np.zeros(dimension, dtype=np.uint8)
            for family_index in range(len(families)):
                codes |= ((p_train[family_index] >= 0).astype(np.uint8) << family_index)
            stability = np.mean(np.signbit(p_train) == np.signbit(p_lock), axis=0).astype(np.float32)
            group_ids[qpoint, event] = codes; group_stability[qpoint, event] = stability
            group_counts.update(codes.tolist())
            for first in range(len(families)):
                for second in range(len(families)):
                    reuse_matrix[qpoint, event, first, second] = cosine(p_lock[first], p_lock[second])

            for factor in factor_cosines:
                values = sorted({rows[index].get(factor) for index in lockbox}, key=str)
                if len(values) != 2:
                    raise RuntimeError((task, factor, values))
                first = centered_factor_family(y, rows, lockbox, families, factor, values[0])
                second = centered_factor_family(y, rows, lockbox, families, factor, values[1])
                for family_index, family in enumerate(families):
                    factor_cosines[factor][family].append(cosine(first[family_index], second[family_index]))

            constant = y[train].mean(0)
            condition_means = group_means(y[train], rows, train, condition_keys)
            pred = np.stack([condition_means.get(tuple(rows[index].get(key) for key in condition_keys), constant) for index in lockbox])
            truth = y[lockbox]
            gain = np.sum((truth - constant) ** 2 - (truth - pred) ** 2, axis=0, dtype=np.float64).astype(np.float32)
            coordinate_gain[qpoint, event] = gain
            gain_total += float(gain.sum(dtype=np.float64))
            order = np.argsort(np.asarray(rms[qpoint, event], dtype=np.float32), kind="stable")
            for quartile, indices in enumerate(np.array_split(order, 4)):
                gain_quartiles[quartile] += float(gain[indices].sum(dtype=np.float64))
                gain_positive_quartiles[quartile] += float(np.maximum(gain[indices], 0).sum(dtype=np.float64))
            slice_summary.append({
                "task": task, "qpoint": qpoint, "event_index": event,
                "condition_gain_sum": float(gain.sum(dtype=np.float64)),
                "condition_positive_coordinate_fraction": float(np.mean(gain > 0)),
                "passport_lockbox_mean_cosine": float(np.mean([cosine(p_train[i], p_lock[i]) for i in range(len(families))])),
                "coordinate_family_sign_stability": float(stability.mean()),
            })
            del x, y, truth, pred, gain
        for value in passports.values(): value.flush()
        group_ids.flush(); group_stability.flush(); coordinate_gain.flush(); reuse_matrix.flush()
        print(f"[phase2400 {task}] update {qpoint + 1}/{Q_UPDATES}", flush=True)

    # A coordinate's passport is its family x event vector; adjacent-update cosine measures persistence without dropping coordinates.
    persistence = np.lib.format.open_memmap(OUT / f"derived/{task}_coordinate_adjacent_update_persistence.float32.npy", mode="w+", dtype=np.float32,
                                             shape=(Q_UPDATES - 1, dimension))
    train_passport = passports["discovery"]
    for qpoint in range(Q_UPDATES - 1):
        left = np.asarray(train_passport[qpoint], dtype=np.float32).transpose(2, 0, 1).reshape(dimension, -1)
        right = np.asarray(train_passport[qpoint + 1], dtype=np.float32).transpose(2, 0, 1).reshape(dimension, -1)
        numerator = np.sum(left * right, axis=1)
        denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
        persistence[qpoint] = numerator / np.maximum(denominator, 1e-12)
    persistence.flush()
    factor_summary = {factor: {family: float(np.mean(values)) for family, values in by_family.items()}
                      for factor, by_family in factor_cosines.items()}
    partition_summary = {family: float(np.mean(values)) for family, values in partition_cosines.items()}
    dominant_groups = [{"group_id": int(group), "coordinates_across_q_event": count,
                        "fraction": count / (Q_UPDATES * event_count * dimension)} for group, count in group_counts.most_common(16)]
    result = {
        "task": task, "families": families, "field_shape": list(field.shape),
        "partition_family_passport_cosine": partition_summary,
        "factor_invariance_cosine": factor_summary,
        "family_reuse_mean_matrix": np.asarray(reuse_matrix).mean(axis=(0, 1)).astype(float).tolist(),
        "family_sign_groups": {"possible": 2 ** len(families), "observed": len(group_counts), "dominant": dominant_groups,
                               "all_coordinates_assigned": sum(group_counts.values()) == Q_UPDATES * event_count * dimension},
        "sign_stability_mean": float(np.asarray(group_stability).mean()),
        "adjacent_update_coordinate_persistence_mean": float(np.asarray(persistence).mean()),
        "condition_gain": {"total": gain_total, "by_rms_quartile_low_to_high": gain_quartiles,
                           "positive_only_by_rms_quartile_low_to_high": gain_positive_quartiles,
                           "low_quartile_fraction_of_signed_gain": gain_quartiles[0] / gain_total if gain_total else float("nan")},
        "arrays": {"passports": {key: str(path) for key, path in passport_paths.items()},
                   "group_ids": str(OUT / f"derived/{task}_family_sign_group.uint8.npy"),
                   "group_stability": str(OUT / f"derived/{task}_family_sign_stability.float32.npy"),
                   "condition_gain": str(OUT / f"derived/{task}_condition_gain_lockbox.float32.npy"),
                   "reuse_matrix": str(OUT / f"derived/{task}_family_reuse_cosine.float32.npy"),
                   "persistence": str(OUT / f"derived/{task}_coordinate_adjacent_update_persistence.float32.npy")},
    }
    save(OUT / f"analysis/{task}_passport_summary.json", result)
    save(OUT / f"analysis/{task}_slice_summary.json", slice_summary)
    for value in passports.values(): value.flush(); close(value)
    group_ids.flush(); group_stability.flush(); coordinate_gain.flush(); reuse_matrix.flush(); persistence.flush()
    close(group_ids); close(group_stability); close(coordinate_gain); close(reuse_matrix); close(persistence); close(rms); close(field)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 条件坐标机制护照、复用—分化与跨层传播图谱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不挑Top-K，给全部2560坐标建立局部更新护照。对每个block更新×语义事件，以族相对同partition全局均值定义family passport；分别保存discovery、confirmation、fresh lockbox。测量同族跨新unit稳定性，以及在fresh锁箱中canonical/paraphrase、中英、正反方向、答案槽和source/target查询之间的族中心响应余弦。每个坐标按全部族passport符号得到完整组ID，所有坐标都被分组；保存组在锁箱中的逐坐标符号稳定度、族间复用矩阵和相邻block的逐坐标passport延续。另把Phase2399条件算子相对常量基线的SSE收益精确分配给每个坐标，按局部update RMS四分位检查低幅值坐标贡献。

$$P_{{q,e,f,j}}=\mathbb E[U_{{q,e,j}}\mid f]-\mathbb E[U_{{q,e,j}}],$$

$$g_{{q,e,j}}=\sum_{{r\in lock}}\left[(U_{{r}}-\bar U_{{train}})^2-(U_{{r}}-\widehat U_{{condition}})^2\right]_j,$$

$$\mathrm{{group}}_{{q,e,j}}=\sum_f2^{{i(f)}}\mathbf1[P_{{q,e,f,j}}\ge0].$$

**结果汇总。** 选择任务 `{json.dumps(result['selection'], ensure_ascii=False)}`；组合任务 `{json.dumps(result['composition'], ensure_ascii=False)}`；综合裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2400_c22641_c22960_coordinate_passport_reuse_atlas.py`；全坐标family passport、符号组ID、锁箱稳定度、族复用矩阵、逐坐标条件收益、相邻层持续度和final位于 `tests/glm5/result/phase2400_c22641_c22960_coordinate_passport_reuse_atlas`。

**理论进展。** Phase2399的条件收益被拆成可检查的坐标拼图：哪些族在固定坐标同号/反号，哪些坐标跨unit保留passport，哪些只依赖语言、表达、方向或答案槽，低RMS坐标是否共同贡献。符号组不是压缩表示或真实模块，只是把全部物理坐标按相对响应编目；只有跨partition、跨表达和跨层持续的组才可进入“条件化齿轮候选”。

**问题硬伤与结论。** family passport仍混合关系词、实体token位置和模板；跨表面/语言余弦是关键反证。相邻层passport持续可能来自残差连接，不代表执行同一功能。逐坐标SSE收益是外部预测收益的可加分解，不是坐标间真实交互；本Phase最多证明分布范围和条件特异性，不能由“许多坐标共同贡献”跳到因果齿轮。下一Phase必须连接到第一分歧token输出贡献，并单独裁决事实段平凡共享与query/answer事件。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    selection = analyze_task("selection", P2398 / "raw/selection_event_field.float16.npy", read_rows(P2398 / "index/selection_rows.jsonl"))
    composition = analyze_task("composition", P2398 / "raw/composition_event_field.float16.npy", read_rows(P2398 / "index/composition_rows.jsonl"))
    selection_surface = float(np.mean(list(selection["factor_invariance_cosine"]["surface"].values())))
    selection_language = float(np.mean(list(selection["factor_invariance_cosine"]["language"].values())))
    selection_partition = float(np.mean(list(selection["partition_family_passport_cosine"].values())))
    composition_partition = float(np.mean(list(composition["partition_family_passport_cosine"].values())))
    adjudication = {
        "selection_new_unit_passport_cosine": selection_partition,
        "selection_cross_surface_passport_cosine": selection_surface,
        "selection_cross_language_passport_cosine": selection_language,
        "composition_new_unit_passport_cosine": composition_partition,
        "selection_low_rms_quartile_signed_gain_fraction": selection["condition_gain"]["low_quartile_fraction_of_signed_gain"],
        "all_coordinate_groups_only_descriptive": True,
        "gear_claim_closed": False,
    }
    checks = {"selection_all_coordinates": selection["family_sign_groups"]["all_coordinates_assigned"],
              "composition_all_coordinates": composition["family_sign_groups"]["all_coordinates_assigned"],
              "finite": all(math.isfinite(value) for value in (selection_surface, selection_language, selection_partition, composition_partition)),
              "low_value_reported": math.isfinite(selection["condition_gain"]["low_quartile_fraction_of_signed_gain"]),
              "claim_boundary": not adjudication["gear_claim_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "selection": selection, "composition": composition,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
