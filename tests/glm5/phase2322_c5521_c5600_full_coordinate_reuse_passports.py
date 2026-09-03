#!/usr/bin/env python3
"""Observe full-coordinate reuse, sign stability, family modulation, and state reversal."""
from __future__ import annotations

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
P2321 = RESULT / "phase2321_c5481_c5520_fp16_atlas_cleanup"
OUT = RESULT / "phase2322_c5521_c5600_full_coordinate_reuse_passports"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2322
CAMPAIGN = "C5521-C5600"
EPS = 1e-12
NONZERO_GATE = 0.75
SIGN_GATE = 0.75
STRICT_GATE = 0.90
FAMILY_DEVIATION_GATE = 0.50
CHANNELS = (
    "mean_signed", "mean_absolute", "rms", "nonzero_fraction",
    "sample_sign_consistency", "family_sign_consensus",
    "family_deviation_ratio", "state_sign_reversal",
)

sys.path.insert(0, str(TESTS))
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SOURCES = (
    ("qwen4_bf16", "c5323_qwen4b_directional_derivative", "Qwen3-4B-BF16"),
    ("qwen4_fp16", "c5481_qwen4b_fp16_directional_derivative", "Qwen3-4B-FP16"),
    ("qwen14_fp16", "c5326_qwen3_14b_directional_derivative", "Qwen3-14B-FP16"),
    ("glm4_bf16", "c5328_glm4_directional_derivative", "GLM-4-9B-BF16"),
    ("deepseek7b_bf16", "c5330_deepseek7b_directional_derivative", "DeepSeek-7B-BF16"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_probe(row: dict) -> int | None:
    probe = int(row["probe"])
    if probe <= 3:
        return probe
    members = list(row.get("probe_members", []))
    if members == [0, 1]:
        return 4
    if members == [2, 3]:
        return 5
    return None


def load_cells(dataset_id: str) -> tuple[dict, dict[tuple[int, int, int], tuple[np.ndarray, list[dict]]]]:
    metadata = load_json(VIS / f"{dataset_id}.json")
    binary = np.load(VIS / Path(metadata["binary_url"]).name, mmap_mode="r")
    partitions = {row.get("partition") for row in metadata["rows"]}
    implicit_fresh_lockbox = partitions == {None}
    source_values = sorted({int(row["source_q"]) for row in metadata["rows"]})
    source_slots = {value: index for index, value in enumerate(source_values)}
    target_slots = {"q_plus_1": 0, "q_plus_4": 1, "final_norm": 2}
    grouped: dict[tuple[int, int, int], list[tuple[int, dict]]] = defaultdict(list)
    for binary_index, row in enumerate(metadata["rows"]):
        if not implicit_fresh_lockbox and row.get("partition") != "fresh_lockbox":
            continue
        probe = canonical_probe(row)
        if probe is None:
            continue
        key = (source_slots[int(row["source_q"])], probe, target_slots[row["target_slot"]])
        normalized = dict(row)
        if implicit_fresh_lockbox:
            normalized["partition"] = "fresh_lockbox"
        normalized["canonical_probe"] = probe
        grouped[key].append((binary_index, normalized))
    cells = {}
    for key in sorted(grouped):
        entries = sorted(grouped[key], key=lambda item: item[1]["case_id"])
        if len(entries) != 32:
            raise RuntimeError(("cell_row_count", dataset_id, key, len(entries)))
        values = np.asarray(binary[[item[0] for item in entries]], dtype=np.float32)
        cells[key] = (values, [item[1] for item in entries])
    atlas.close_memmap(binary)
    if len(cells) != 54:
        raise RuntimeError(("cell_count", dataset_id, len(cells)))
    return metadata, cells


def coordinate_metrics(values: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, dict, np.ndarray]:
    signs = np.sign(values)
    nonzero = values != 0
    nonzero_count = nonzero.sum(axis=0)
    nonzero_fraction = nonzero.mean(axis=0)
    sample_sign = np.abs(signs.sum(axis=0)) / np.maximum(nonzero_count, 1)
    mean_signed = values.mean(axis=0, dtype=np.float64)
    mean_absolute = np.abs(values).mean(axis=0, dtype=np.float64)
    rms = np.sqrt(np.square(values, dtype=np.float64).mean(axis=0))
    families = sorted({row["family"] for row in rows})
    family_means = np.stack([
        values[[index for index, row in enumerate(rows) if row["family"] == family]].mean(axis=0)
        for family in families
    ])
    family_sign = np.abs(np.sign(family_means).sum(axis=0)) / len(families)
    family_deviation = np.sqrt(np.square(family_means - mean_signed, dtype=np.float64).mean(axis=0))
    family_deviation_ratio = family_deviation / (rms + EPS)
    state_means = []
    for state in (0, 1):
        indices = [index for index, row in enumerate(rows) if int(row["state"]) == state]
        state_means.append(values[indices].mean(axis=0))
    state_reversal = ((state_means[0] * state_means[1]) < 0).astype(np.float32)
    passport = np.stack((
        mean_signed, mean_absolute, rms, nonzero_fraction, sample_sign,
        family_sign, family_deviation_ratio, state_reversal,
    )).astype(np.float32)
    stable = ((nonzero_fraction >= NONZERO_GATE) & (sample_sign >= SIGN_GATE)
              & (family_sign >= SIGN_GATE))
    strict = ((nonzero_fraction >= STRICT_GATE) & (sample_sign >= STRICT_GATE)
              & (family_sign >= STRICT_GATE))
    positive_amplitudes = mean_absolute[mean_absolute > 0]
    amplitude_median = float(np.median(positive_amplitudes)) if positive_amplitudes.size else 0.0
    low_amplitude = mean_absolute <= amplitude_median
    family_modulated = ((nonzero_fraction >= NONZERO_GATE)
                        & (family_deviation_ratio >= FAMILY_DEVIATION_GATE))
    summary = {
        "coordinates": int(values.shape[1]),
        "nonzero_fraction_median": float(np.median(nonzero_fraction)),
        "sample_sign_consistency_median": float(np.median(sample_sign)),
        "family_sign_consensus_median": float(np.median(family_sign)),
        "family_deviation_ratio_median": float(np.median(family_deviation_ratio)),
        "stable_shared_fraction": float(stable.mean()),
        "strict_stable_shared_fraction": float(strict.mean()),
        "stable_low_amplitude_fraction": float((stable & low_amplitude).mean()),
        "stable_low_share_of_stable": float((stable & low_amplitude).sum() / max(stable.sum(), 1)),
        "family_modulated_fraction": float(family_modulated.mean()),
        "state_sign_reversal_fraction": float(state_reversal.mean()),
        "mean_absolute_median": amplitude_median,
    }
    return passport, summary, stable


def build_model_passport(key: str, dataset_id: str, label: str) -> tuple[dict, dict, np.ndarray]:
    source_meta, cells = load_cells(dataset_id)
    dimension = int(source_meta["coordinate_count"])
    binary = VIS / f"c{5521 + list(key for key, _, _ in SOURCES).index(key)}_{key}_reuse_passport.float32.npy"
    output = atlas.create_binary(binary.name, 54 * len(CHANNELS), dimension, np.dtype(np.float32))
    rows = []
    summaries = {}
    stable_masks = []
    cursor = 0
    for cell, (values, case_rows) in sorted(cells.items()):
        passport, summary, stable = coordinate_metrics(values, case_rows)
        source_slot, probe, target_slot = cell
        summaries[f"s{source_slot}_p{probe}_t{target_slot}"] = summary
        stable_masks.append(stable)
        representative = case_rows[0]
        for channel_index, channel in enumerate(CHANNELS):
            output[cursor] = passport[channel_index]
            rows.append({
                "model_condition": key, "source_slot": source_slot,
                "source_q": int(representative["source_q"]), "probe": probe,
                "probe_kind": representative["probe_kind"], "target_slot": target_slot,
                "target_q": int(representative["target_q"]), "channel": channel,
            })
            cursor += 1
    output.flush(); atlas.close_memmap(output)
    dataset = atlas.write_metadata(
        binary.stem.split(".float32")[0], f"{label} all-coordinate reuse passport",
        binary, rows, label, "full_coordinate_reuse_passport_v1", "exploratory observation",
        "32 fresh_lockbox samples per source/probe/target cell",
        "one basic coordinate statistic per row; original physical coordinate order retained",
        {"channels": list(CHANNELS), "thresholds": frozen_config()["thresholds"],
         "warning": "descriptive reuse passport, not a semantic-neuron classifier"},
    )
    aggregate = {
        "cells": 54, "coordinates": dimension,
        "median_stable_shared_fraction": float(np.median([
            value["stable_shared_fraction"] for value in summaries.values()])),
        "median_strict_stable_shared_fraction": float(np.median([
            value["strict_stable_shared_fraction"] for value in summaries.values()])),
        "median_stable_low_share_of_stable": float(np.median([
            value["stable_low_share_of_stable"] for value in summaries.values()])),
        "median_family_modulated_fraction": float(np.median([
            value["family_modulated_fraction"] for value in summaries.values()])),
        "median_state_sign_reversal_fraction": float(np.median([
            value["state_sign_reversal_fraction"] for value in summaries.values()])),
        "by_cell": summaries,
    }
    return dataset, aggregate, np.stack(stable_masks)


def frozen_config() -> dict:
    return {
        "phase": PHASE, "campaign": CAMPAIGN, "frozen_before_passport_readout": True,
        "sources": [item[0] for item in SOURCES],
        "coordinate_policy": "all physical coordinates; no Top-K, PCA, projection, or reordering",
        "channels": list(CHANNELS),
        "thresholds": {
            "nonzero_fraction": NONZERO_GATE, "sample_sign_consistency": SIGN_GATE,
            "family_sign_consensus": SIGN_GATE, "strict_all_three": STRICT_GATE,
            "family_deviation_ratio": FAMILY_DEVIATION_GATE,
            "low_amplitude": "at_or_below_within_cell_median_positive_mean_absolute",
        },
        "prospective_next_partition": "fresh_confirmation",
        "prospective_gates": {
            "global_mean_relative_mse_max": 0.35,
            "global_better_than_family_all_families": True,
            "frozen_stable_sign_agreement_min": 0.80,
            "pair_superposition_relative_mse_max": 0.05,
            "even_to_odd_l2_max": 0.30,
        },
        "claim_boundary": "observation first; thresholds describe coordinate populations and do not prove semantics",
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {key: {name: value for name, value in summary.items() if name != "by_cell"}
               for key, summary in result["model_summaries"].items()}
    record = rf"""

## Phase {PHASE}: 五条件全坐标复用、符号翻转与族偏离观察护照（{CAMPAIGN}） [{stamp}]

**测试原理、公式和冻结规则。** 本期不提出语义齿轮假说，先观察 Qwen3-4B BF16/FP16、Qwen3-14B FP16、GLM-4-9B BF16、DeepSeek-7B BF16 的完整目标坐标。在每个 `源深度 × 探针 × 目标` 单元的 32 行 fresh_lockbox 上，对每个物理坐标记录均值、平均绝对幅度、RMS、非零率、样本符号一致性、八族均值符号一致性、族均值偏离比和状态翻转。冻结配置 `{json.dumps(result['config'], ensure_ascii=False)}`。没有 Top-K、PCA、投影、余弦筛选或跨模型坐标对齐。
$$
C_j=\frac{{|\sum_i\operatorname{{sgn}}R_{{ij}}|}}{{\sum_i[ R_{{ij}}\ne0 ]}},\qquad
F_j=\frac{{|\sum_f\operatorname{{sgn}}\bar R_{{fj}}|}}{{8}},\qquad
B_j=\frac{{\sqrt{{\frac18\sum_f(\bar R_{{fj}}-\bar R_j)^2}}}}{{\sqrt{{\frac1n\sum_iR_{{ij}}^2}}+\varepsilon}}.
$$

**结果汇总与相关文件。** 五条件摘要 `{json.dumps(compact, ensure_ascii=False)}`；同一 4B 精度稳定掩码比较 `{json.dumps(result['precision_mask_comparison'], ensure_ascii=False)}`；资产 `{json.dumps(result['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；目录与构建 `{json.dumps(result['catalog'], ensure_ascii=False)}`, `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2322_c5521_c5600_full_coordinate_reuse_passports.py`；结果目录 `tests/glm5/result/phase2322_c5521_c5600_full_coordinate_reuse_passports`；完整逐单元结果保存于 `analysis/model_summaries.json`。

**分析、理论进展、问题硬伤与结论。** “稳定共享”只表示同一随机输入方向下目标坐标在 32 个样本及八族均值中大体同号，不表示坐标拥有共同语义。低幅稳定坐标比例用于检查低值参数是否被忽略，不是筛选规则。族偏离比高也可能来自词汇、tokenizer、句长或状态差异。BF16 非零率和符号稳定性会受舍入严重影响；跨模型只能比较比例分布。理论主体仍为“条件化输出场闭合理论”，本期只增加逐坐标观察图谱。下一步使用这里冻结的 FP16 stable 掩码和预测门，在未用于 FP16 主动场的 fresh_confirmation 新词汇分区上前瞻复验；失败只淘汰对应规律，不停止图谱积累。"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = load_json(final_path)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = load_json(P2321 / "analysis/final.json")
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2321 is not authorized")
    config = frozen_config()
    save_json(OUT / "config/frozen_observation_contract.json", config)
    atlas.PHASE = PHASE
    atlas.CAMPAIGN = CAMPAIGN
    atlas.OUT = OUT
    datasets = []
    summaries = {}
    stable_masks = {}
    for key, dataset_id, label in SOURCES:
        dataset, summary, masks = build_model_passport(key, dataset_id, label)
        datasets.append(dataset)
        summaries[key] = summary
        stable_masks[key] = masks
    bf16 = stable_masks["qwen4_bf16"]
    fp16 = stable_masks["qwen4_fp16"]
    intersection = np.logical_and(bf16, fp16).sum(axis=1)
    union = np.logical_or(bf16, fp16).sum(axis=1)
    precision_comparison = {
        "median_stable_mask_jaccard": float(np.median(intersection / np.maximum(union, 1))),
        "mean_fp16_stable_fraction": float(fp16.mean(axis=1).mean()),
        "mean_bf16_stable_fraction": float(bf16.mean(axis=1).mean()),
        "same_model_same_coordinates": True,
        "claim_boundary": "mask sensitivity to measurement precision, not semantic stability",
    }
    save_json(OUT / "analysis/model_summaries.json", summaries)
    verification = [atlas.verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id") for row in verification):
        raise RuntimeError(("asset_verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
        "config": config, "model_summaries": summaries,
        "precision_mask_comparison": precision_comparison,
        "datasets": [atlas.serializable(row) for row in datasets],
        "verification": verification, "catalog": catalog, "frontend_build": build,
        "checks": {
            "parent_authorized": True, "config_frozen": True,
            "five_conditions": len(datasets) == 5,
            "all_54_cells": all(value["cells"] == 54 for value in summaries.values()),
            "all_assets_verified": all(all(value for key, value in row.items() if key != "id")
                                           for row in verification),
            "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
            "frontend_build_passed": build["passed"],
            "no_client_connection": not build["browser_or_client_connection"],
            "no_coordinate_selection": True,
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
