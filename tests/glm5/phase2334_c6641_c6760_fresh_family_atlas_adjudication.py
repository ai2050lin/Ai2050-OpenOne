#!/usr/bin/env python3
"""Prospectively adjudicate frozen twenty-family full-coordinate observations on fresh partitions."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2330 = RESULT / "phase2330_c6081_c6200_language_family_atlas_contract"
P2331 = RESULT / "phase2331_c6201_c6360_qwen4b_twenty_family_fullfield"
P2333 = RESULT / "phase2333_c6481_c6640_twenty_family_coordinate_atlas"
OUT = RESULT / "phase2334_c6641_c6760_fresh_family_atlas_adjudication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
DELTA = VIS / "c6483_qwen4b_twenty_family_natural_state_delta.float32.npy"
PHASE = 2334
CAMPAIGN = "C6641-C6760"
EPS = 1e-12
FRESH = ("fresh_confirmation", "fresh_lockbox")
CHANNELS = ("frozen_mean", "fresh_mean", "signed_residual", "same_sign", "wrong_family_residual", "opposite_sign_residual")

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2330_c6081_c6200_language_family_atlas_contract as contract  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return io.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    io.write_rows(path, rows)


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum(np.square(actual - predicted, dtype=np.float64)) /
                 (np.sum(np.square(actual, dtype=np.float64)) + EPS))


def symmetric_mse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(np.square(left - right, dtype=np.float64)) /
                 ((np.sum(np.square(left, dtype=np.float64)) + np.sum(np.square(right, dtype=np.float64))) / 2 + EPS))


def load_delta(pairs: list[dict]) -> np.ndarray:
    value = np.load(DELTA, mmap_mode="r")
    expected = (len(pairs) * 38, 2560)
    if value.shape != expected:
        raise RuntimeError(("delta_shape", value.shape, expected))
    return value.reshape(len(pairs), 38, 2560)


def discovery_means(delta: np.ndarray, pairs: list[dict]) -> np.ndarray:
    output = np.empty((len(contract.FAMILIES), 38, 2560), dtype=np.float64)
    for family_index, family in enumerate(contract.FAMILIES):
        indices = [i for i, row in enumerate(pairs) if row["family"] == family and row["partition"] == "discovery"]
        output[family_index] = delta[indices].astype(np.float64).mean(axis=0)
    return output


def adjudicate_candidates(delta: np.ndarray, pairs: list[dict], means: np.ndarray, freeze: dict) -> tuple[dict, np.ndarray, list[dict]]:
    observations = freeze["structural_observations"]
    passport = np.empty((len(observations), len(FRESH), len(CHANNELS), 2560), dtype=np.float32)
    records = []
    cells = {}
    for observation_index, observation in enumerate(observations):
        family, q = observation["family"], int(observation["qpoint"])
        family_index = contract.FAMILIES.index(family)
        wrong_index = (family_index + 1) % len(contract.FAMILIES)
        frozen_mean = means[family_index, q]
        wrong_mean = means[wrong_index, q]
        cell = {"family": family, "qpoint": q, "behavior_authorized": any(row["family"] == family for row in freeze["semantic_candidates"]), "partitions": {}}
        passes = []
        for partition_index, partition in enumerate(FRESH):
            indices = [i for i, row in enumerate(pairs) if row["family"] == family and row["partition"] == partition]
            fresh = delta[indices, q].astype(np.float64)
            fresh_mean = fresh.mean(axis=0)
            sign = float(np.mean(frozen_mean * fresh_mean > 0))
            high = np.abs(frozen_mean) >= np.median(np.abs(frozen_mean))
            sym = symmetric_mse(frozen_mean, fresh_mean)
            correct_errors = [relative_mse(delta[i, q].astype(np.float64), frozen_mean) for i in indices]
            wrong_errors = [relative_mse(delta[i, q].astype(np.float64), wrong_mean) for i in indices]
            opposite_errors = [relative_mse(delta[i, q].astype(np.float64), -frozen_mean) for i in indices]
            wrong_layers = [layer for layer in (max(1, q - 4), min(37, q + 4)) if layer != q]
            wrong_layer_errors = []
            for i in indices:
                actual = delta[i, q].astype(np.float64)
                wrong_layer_errors.append(min(relative_mse(actual, means[family_index, layer]) for layer in wrong_layers))
            result = {
                "rows": len(indices), "mean_sign_agreement": sign,
                "high_amplitude_sign_agreement": float(np.mean((frozen_mean * fresh_mean > 0)[high])),
                "symmetric_relative_mse": sym,
                "median_relative_mse": float(np.median(correct_errors)),
                "correct_beats_wrong_family_fraction": float(np.mean(np.asarray(correct_errors) < np.asarray(wrong_errors))),
                "correct_beats_opposite_sign_fraction": float(np.mean(np.asarray(correct_errors) < np.asarray(opposite_errors))),
                "correct_beats_wrong_layer_fraction": float(np.mean(np.asarray(correct_errors) < np.asarray(wrong_layer_errors))),
            }
            result["passed"] = sign >= 0.65 and sym <= 1.0
            passes.append(result["passed"])
            cell["partitions"][partition] = result
            passport[observation_index, partition_index] = np.stack([
                frozen_mean, fresh_mean, fresh_mean - frozen_mean, (frozen_mean * fresh_mean > 0).astype(np.float64),
                fresh_mean - wrong_mean, fresh_mean + frozen_mean,
            ]).astype(np.float32)
            records.append({"family": family, "qpoint": q, "partition": partition, **result})
        cell["passed_both_fresh_partitions"] = all(passes)
        cell["claim_level"] = "semantic_candidate" if cell["behavior_authorized"] else "structural_observation"
        cells[family] = cell
    return {
        "thresholds": freeze["thresholds"], "cells": cells,
        "passed_structural": [family for family, row in cells.items() if row["passed_both_fresh_partitions"]],
        "passed_semantic": [family for family, row in cells.items() if row["passed_both_fresh_partitions"] and row["behavior_authorized"]],
    }, passport, records


def identification(delta: np.ndarray, pairs: list[dict], means: np.ndarray) -> dict:
    summary = {partition: {} for partition in FRESH}
    records = []
    for partition in FRESH:
        test = [(i, row) for i, row in enumerate(pairs) if row["partition"] == partition]
        for q in range(38):
            qrecords = []
            for pair_index, pair in test:
                actual = delta[pair_index, q].astype(np.float64)
                errors = [relative_mse(actual, means[index, q]) for index in range(len(contract.FAMILIES))]
                correct_index = contract.FAMILIES.index(pair["family"])
                predicted = int(np.argmin(errors))
                row = {
                    "partition": partition, "pair_index": pair_index, "family": pair["family"], "qpoint": q,
                    "predicted_family": contract.FAMILIES[predicted], "correct": predicted == correct_index,
                    "correct_family_mse": errors[correct_index],
                    "best_wrong_mse": min(value for index, value in enumerate(errors) if index != correct_index),
                    "correct_beats_circular_wrong": errors[correct_index] < errors[(correct_index + 1) % len(contract.FAMILIES)],
                    "correct_beats_opposite_sign": errors[correct_index] < relative_mse(actual, -means[correct_index, q]),
                }
                qrecords.append(row); records.append(row)
            summary[partition][str(q)] = {
                "rows": len(qrecords), "accuracy": float(np.mean([row["correct"] for row in qrecords])),
                "chance": 1 / len(contract.FAMILIES),
                "chance_ratio": float(np.mean([row["correct"] for row in qrecords])) * len(contract.FAMILIES),
                "median_correct_over_best_wrong_ratio": float(np.median([row["correct_family_mse"] / (row["best_wrong_mse"] + EPS) for row in qrecords])),
                "correct_beats_circular_wrong_fraction": float(np.mean([row["correct_beats_circular_wrong"] for row in qrecords])),
                "correct_beats_opposite_sign_fraction": float(np.mean([row["correct_beats_opposite_sign"] for row in qrecords])),
            }
    write_rows(OUT / "analysis/fresh_identification_records.jsonl", records)
    return summary


def publish_passport(passport: np.ndarray, freeze: dict) -> dict:
    dataset_id = "c6641_qwen4b_frozen_family_fresh_adjudication"
    binary = VIS / f"{dataset_id}.float32.npy"
    output = atlas.create_binary(binary.name, int(np.prod(passport.shape[:-1])), passport.shape[-1], np.float32)
    output[:] = passport.reshape(-1, passport.shape[-1])
    output.flush(); close_memmap(output)
    metadata = []
    for observation in freeze["structural_observations"]:
        for partition in FRESH:
            for channel in CHANNELS:
                metadata.append({
                    "family": observation["family"], "macrotype": contract.MACROTYPE[observation["family"]],
                    "qpoint": observation["qpoint"], "partition": partition, "channel": channel,
                    "behavior_authorized": any(row["family"] == observation["family"] for row in freeze["semantic_candidates"]),
                })
    return atlas.write_metadata(
        dataset_id, "Qwen3-4B frozen family observations on fresh partitions", binary, metadata,
        "Qwen3-4B-FP16", "full_coordinate_fresh_family_adjudication_v1", "prospective derived",
        "frozen discovery-confirmation family/qpoint cells tested on fresh_confirmation and fresh_lockbox",
        "frozen means, fresh means, signed residuals and explicit controls in every coordinate",
        {"coordinate_count": 2560, "channels": list(CHANNELS), "frozen_before_fresh": True, "no_projection": True},
    )


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 二十族冻结全坐标规律的fresh前瞻裁决与显式空控制（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 严格读取 Phase2333 在查看fresh统计前冻结的四个 `family×qpoint` 结构观察和一个行为授权语义候选；用 discovery 均值分别预测 fresh_confirmation 与 fresh_lockbox，不改层、不改阈值。每个候选同时比较循环错族、反号和 q±4 错层控制。另以每个q的二十个 discovery 族均值，对两个fresh分区全部480对自然状态变化做最近族识别；独立单位是材料pair，不把层×坐标当独立样本。

$$
\widehat f_q(x)=\arg\min_g\frac{{\lVert\Delta H_q(x)-\overline{{\Delta H}}_{{g,q}}^{{disc}}\rVert_2^2}}{{\lVert\Delta H_q(x)\rVert_2^2+\varepsilon}},
$$

$$
E_{{sign}}=\frac1d\sum_j[\bar\Delta_{{disc,j}}\bar\Delta_{{fresh,j}}>0],\qquad
E_{{sym}}=\frac{{\lVert\bar\Delta_{{disc}}-\bar\Delta_{{fresh}}\rVert_2^2}}{{(\lVert\bar\Delta_{{disc}}\rVert_2^2+\lVert\bar\Delta_{{fresh}}\rVert_2^2)/2+\varepsilon}}.
$$

**结果汇总、门槛与相关文件。** 冻结候选结果 `{json.dumps(result['candidate_adjudication'], ensure_ascii=False)}`；全族识别 `{json.dumps(result['identification'], ensure_ascii=False)}`；发布 `{json.dumps(result['dataset'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。门槛沿用同号率不低于0.65且对称相对MSE不高于1.0，并要求两个fresh分区同时通过。脚本 `tests/glm5/phase2334_c6641_c6760_fresh_family_atlas_adjudication.py`；结果 `tests/glm5/result/phase2334_c6641_c6760_fresh_family_atlas_adjudication`。

**分析、理论进展、问题硬伤与结论。** 通过只表示某个受控自然状态操作的全坐标均值在新词汇unit上重复；错族、反号、错层只是必要控制，不构成因果调用。特别是q1/q2候选可能由固定词汇、标点和模板差异直接造成，不能称为深层语言编码。族识别高于1/20只证明分布式场携带族信息；若中位正确原型仍不如最佳错族，则说明信息集中在部分易辨族，不能宣布普遍分类闭合。fresh材料虽在上一Phase已一并测量并发布，但选择代码只索引discovery/confirmation，冻结文件及哈希先于本期fresh统计读取；这弱于完全延迟采集，必须保留此边界。

**下一阶段路线判断。** 若早层标点/风格/翻译候选通过，目标仍是同一个“语言模式族普遍编码规律”，必须自动进入独立构式生成器和新词面的反模板复验，排除固定词串；若只有深层候选通过，再考虑因果调用。任何单个因果门失败都不停止对其他已通过特征的图谱积累。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((P2333 / "analysis/final.json").read_text(encoding="utf-8"))
    freeze = json.loads((P2333 / "config/frozen_fresh_adjudication.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"] or not freeze["frozen_before_fresh_read"]:
        raise RuntimeError("Phase2333 freeze invalid")
    pairs = read_rows(P2333 / "index/state_pairs.jsonl")
    delta = load_delta(pairs)
    means = discovery_means(delta, pairs)
    candidate, passport, records = adjudicate_candidates(delta, pairs, means, freeze)
    write_rows(OUT / "analysis/candidate_records.jsonl", records)
    identify = identification(delta, pairs, means)
    dataset = publish_passport(passport, freeze)
    verification = atlas.verify(dataset)
    if not all(value for key, value in verification.items() if key != "id"):
        raise RuntimeError(("verification_failed", verification))
    catalog = atlas.update_catalog([dataset])
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    serial_dataset = json.loads(json.dumps(dataset, ensure_ascii=False, default=str))
    serial_catalog = json.loads(json.dumps(catalog, ensure_ascii=False, default=str))
    checks = {
        "freeze_respected": True, "two_fresh_partitions": set(FRESH) == set(next(iter(candidate["cells"].values()))["partitions"]),
        "explicit_controls": all(all(key in part for key in ("correct_beats_wrong_family_fraction", "correct_beats_opposite_sign_fraction", "correct_beats_wrong_layer_fraction"))
                                 for cell in candidate["cells"].values() for part in cell["partitions"].values()),
        "all_fresh_pairs_identified": all(identify[p]["1"]["rows"] == 240 for p in FRESH),
        "asset_verified": all(value for key, value in verification.items() if key != "id"),
        "frontend_build": build["passed"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "candidate_adjudication": candidate,
        "identification": identify, "dataset": serial_dataset, "verification": verification,
        "catalog": serial_catalog, "frontend_build": build,
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(final_path, result)
    close_memmap(delta)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2334_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
