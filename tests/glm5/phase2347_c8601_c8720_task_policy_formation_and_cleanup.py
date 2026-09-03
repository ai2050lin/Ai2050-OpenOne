#!/usr/bin/env python3
"""Diagnose the contradicted-task failure, preserve full-coordinate means, and clean raw fields."""
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
P2344 = RESULT / "phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract"
P2345 = RESULT / "phase2345_c8361_c8480_qwen4b_bilingual_factorial_full_field"
P2346 = RESULT / "phase2346_c8481_c8600_factorial_coordinate_route_competition"
OUT = RESULT / "phase2347_c8601_c8720_task_policy_formation_and_cleanup"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = P2344 / "material/bilingual_factorial_fixed_code.jsonl"
STATES = P2345 / "raw/boundary_all_checkpoints.float16.npy"
TRAJECTORY = P2345 / "derived/layerwise_coordinate_contribution.float32.npy"
DECISIONS = P2345 / "raw/decisions.float32.npy"
TOKEN_STATES = P2345 / "raw/reference_all_token_all_checkpoints.float16.npy"
TOKEN_INDEX = P2345 / "index/reference_all_token_rows.jsonl"
PROTOTYPES = OUT / "derived/supported_task_family_continuous_masks.float32.npz"
PHASE = 2347
CAMPAIGN = "C8601-C8720"
TRAIN_PARTITIONS = ("discovery", "confirmation")
LOCKBOX = "fresh_lockbox"
NORMALIZED_TOKEN_BINS = 8

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract as contract  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def behavior_policy(rows: list[dict]) -> dict:
    decisions = np.load(DECISIONS, mmap_mode="r")
    result = {"overall": {}, "cells": {}}
    for task in contract.TASKS:
        idx = [i for i, row in enumerate(rows) if row["task"] == task]
        accuracy = float(np.mean(decisions[idx, 3]))
        result["overall"][task] = {"rows": len(idx), "correct_task_accuracy": accuracy,
                                            "chooses_supported_claim_rate": accuracy if task == "select_supported" else 1 - accuracy}
    for language in contract.LANGUAGES:
        for codebook in contract.CODEBOOKS:
            for surface in contract.SURFACES:
                for task in contract.TASKS:
                    idx = [i for i, row in enumerate(rows) if row["language"] == language and row["codebook"] == codebook
                           and row["surface"] == surface and row["task"] == task]
                    accuracy = float(np.mean(decisions[idx, 3]))
                    result["cells"][f"{language}:{codebook}:{surface}:{task}"] = {
                        "rows": len(idx), "correct_task_accuracy": accuracy,
                        "chooses_supported_claim_rate": accuracy if task == "select_supported" else 1 - accuracy,
                    }
    result["policy_inversion"] = {
        "supported_accuracy": result["overall"]["select_supported"]["correct_task_accuracy"],
        "contradicted_accuracy": result["overall"]["select_contradicted"]["correct_task_accuracy"],
        "contradicted_rows_choose_supported_rate": result["overall"]["select_contradicted"]["chooses_supported_claim_rate"],
    }
    close_memmap(decisions)
    return result


def margin_trajectory(rows: list[dict]) -> list[dict]:
    trajectory = np.load(TRAJECTORY, mmap_mode="r")
    records = []
    chunk = 512
    for qpoint in range(trajectory.shape[1]):
        margins = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), chunk):
            value = trajectory[start:start + chunk, qpoint].astype(np.float64).sum(axis=1)
            margins[start:start + len(value)] = value.astype(np.float32)
        task_rows = {}
        for task in contract.TASKS:
            idx = np.asarray([i for i, row in enumerate(rows) if row["task"] == task and row["partition"] == LOCKBOX])
            task_rows[task] = {"rows": len(idx), "positive_correct_margin_fraction": float(np.mean(margins[idx] > 0)),
                               "median_correct_margin": float(np.median(margins[idx])),
                               "mean_correct_margin": float(np.mean(margins[idx]))}
        records.append({"qpoint": qpoint, "task": task_rows})
    close_memmap(trajectory)
    contradicted = [row["task"]["select_contradicted"]["positive_correct_margin_fraction"] for row in records]
    supported = [row["task"]["select_supported"]["positive_correct_margin_fraction"] for row in records]
    earliest = None
    for qpoint in range(len(records) - 2):
        if all(supported[q] >= 0.70 and contradicted[q] <= 0.30 for q in range(qpoint, qpoint + 3)):
            earliest = qpoint
            break
    return records + [{"summary": {"earliest_three_checkpoint_truth_policy_bifurcation": earliest,
                                     "interpretation": "Intermediate qpoints use an RMS logit lens and are diagnostic, not native exits."}}]


def checkpoint_means(rows: list[dict]) -> tuple[np.ndarray, list[dict]]:
    states = np.load(STATES, mmap_mode="r")
    trajectory = np.load(TRAJECTORY, mmap_mode="r")
    values = []
    metadata = []
    for view, source, absolute in (("raw_hidden", states, False), ("absolute_output_contribution", trajectory, True)):
        for family in contract.FAMILIES:
            for task in contract.TASKS:
                idx = [i for i, row in enumerate(rows) if row["family"] == family and row["task"] == task
                       and row["partition"] == LOCKBOX]
                for qpoint in range(source.shape[1]):
                    value = source[idx, qpoint].astype(np.float32)
                    if absolute:
                        value = np.abs(value)
                    values.append(value.mean(axis=0, dtype=np.float64).astype(np.float32))
                    metadata.append({"view": view, "family": family, "macrotype": contract.MACROTYPE[family],
                                     "task": task, "partition": LOCKBOX, "qpoint": qpoint,
                                     "source_rows": len(idx), "aggregation": "mean over rows; no coordinate compression"})
    close_memmap(states); close_memmap(trajectory)
    return np.stack(values), metadata


def token_bin_means() -> tuple[np.ndarray, list[dict], dict]:
    index = io.read_rows(TOKEN_INDEX)
    field = np.load(TOKEN_STATES, mmap_mode="r")
    lengths = defaultdict(int)
    for row in index:
        lengths[row["case_id"]] = max(lengths[row["case_id"]], int(row["token_index"]) + 1)
    sums: dict[tuple, np.ndarray] = {}
    counts = defaultdict(int)
    for source_index, row in enumerate(index):
        length = lengths[row["case_id"]]
        token_bin = min(NORMALIZED_TOKEN_BINS - 1, int(row["token_index"] * NORMALIZED_TOKEN_BINS / max(length, 1)))
        key = (row["family"], row["language"], row["task"], int(row["qpoint"]), token_bin)
        if key not in sums:
            sums[key] = np.zeros(field.shape[1], dtype=np.float64)
        sums[key] += field[source_index].astype(np.float64)
        counts[key] += 1
    keys = sorted(sums)
    values = np.stack([(sums[key] / counts[key]).astype(np.float32) for key in keys])
    metadata = [{"family": key[0], "macrotype": contract.MACROTYPE[key[0]], "language": key[1],
                 "task": key[2], "qpoint": key[3], "normalized_token_bin": key[4],
                 "source_token_checkpoint_rows": counts[key], "aggregation": "mean within normalized token bin; all coordinates retained"}
                for key in keys]
    close_memmap(field)
    audit = {"source_rows": len(index), "output_rows": len(keys), "coordinate_count": values.shape[1],
             "normalized_token_bins": NORMALIZED_TOKEN_BINS, "reference_prompts": len(lengths)}
    return values, metadata, audit


def save_continuous_masks(rows: list[dict], qpoint: int) -> dict:
    states = np.load(STATES, mmap_mode="r")
    train = np.asarray([row["partition"] in TRAIN_PARTITIONS and row["task"] == "select_supported" for row in rows])
    grand = states[train, qpoint].astype(np.float32).mean(axis=0, dtype=np.float64).astype(np.float32)
    output = {"grand": grand}
    summaries = {}
    for family in contract.FAMILIES:
        idx = train & np.asarray([row["family"] == family for row in rows])
        prototype = states[idx, qpoint].astype(np.float32).mean(axis=0, dtype=np.float64).astype(np.float32)
        magnitude = np.abs(prototype - grand)
        scale = float(np.quantile(magnitude, 0.95))
        mask = magnitude / max(scale, 1e-8)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        output[f"prototype__{family}"] = prototype
        output[f"mask__{family}"] = mask
        summaries[family] = {"nonzero_coordinates": int(np.count_nonzero(mask)), "coordinate_count": len(mask),
                             "mean_mask": float(mask.mean()), "p95_scale": scale}
    PROTOTYPES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PROTOTYPES, **output)
    close_memmap(states)
    return {"path": str(PROTOTYPES), "qpoint": qpoint, "families": summaries,
            "policy": "continuous all-coordinate masks; no Top-K or coordinate deletion from the atlas"}


def publish(dataset_id: str, title: str, values: np.ndarray, metadata: list[dict], schema: str, boundary: str) -> dict:
    binary = VIS / f"{dataset_id}.float32.npy"
    out = atlas.create_binary(binary.name, values.shape[0], values.shape[1], np.float32)
    out[:] = values
    out.flush(); close_memmap(out)
    return atlas.write_metadata(
        dataset_id, title, binary, metadata, "Qwen3-4B-FP16", schema,
        "failure-axis diagnostic and full-coordinate aggregation; not a causal semantic mechanism", boundary,
        "all 2560 model-local coordinates retained after averaging samples/tokens within declared cells",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "no_topk": True},
    )


def clean_raw_fields() -> dict:
    targets = (STATES, TRAJECTORY)
    root = P2345.resolve()
    records = []
    for path in targets:
        resolved = path.resolve()
        if root not in resolved.parents:
            raise RuntimeError(("cleanup_outside_phase2345", str(resolved)))
        if resolved.exists():
            size = resolved.stat().st_size
            digest = sha256(resolved)
            resolved.unlink()
            records.append({"path": str(resolved), "bytes": size, "sha256_before_delete": digest,
                            "deleted": not resolved.exists(), "recoverable_from_repo": False})
        else:
            records.append({"path": str(resolved), "bytes": 0, "already_absent": True, "deleted": True})
    return {"deleted": records, "bytes_reclaimed": sum(row["bytes"] for row in records),
            "reason": "full boundary fields were not wholly published; key checkpoints, all-token references, route passports and full-layer means are published"}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: “选择矛盾项”失败轴的全token形成轨迹与原始场清理（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2345显示12族的`select_supported`均跨语言×输出码×分区通过0.70，但`select_contradicted`普遍失败。本Phase不把失败当作“没有编码”，而是检查模型是否忽略任务反转、继续选择得到事实支持的命题。用全部36,864条最终行为和38检查点正确码margin lens，另把48条全token参考场按family×language×task×qpoint×8个归一化token位置箱汇总；每个输出仍保留2560具体坐标。并在Phase2346选择q点保存12族连续全坐标原型/掩码，为只在行为通过的`select_supported`子域进行下一轮探索性干预做准备。

$$
P_{{support}}^{{contra}}=1-\operatorname{{Acc}}(y_{{contradicted}}),\qquad
m_{{i,q}}=\sum_j c_{{i,q,j}}.
$$

$$
M_{{f,j}}=\operatorname{{clip}}_{{[0,1]}}
\frac{{|\mu_{{f,j}}-\mu_j|}}{{Q_{{0.95}}(|\mu_f-\mu|)}},
\qquad M_{{f,j}}>0\ \text{{允许覆盖全部坐标}}.
$$

**结果汇总与相关文件。** 行为政策 `{json.dumps(result['behavior_policy'], ensure_ascii=False)}`；margin形成轨迹 `{json.dumps(result['margin_trajectory'], ensure_ascii=False)}`；全token审计 `{json.dumps(result['token_bin_audit'], ensure_ascii=False)}`；连续掩码 `{json.dumps(result['continuous_masks'], ensure_ascii=False)}`；客户端资产 `{json.dumps(result['datasets'], ensure_ascii=False)}`；原始场清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2347_c8601_c8720_task_policy_formation_and_cleanup.py`；结果 `tests/glm5/result/phase2347_c8601_c8720_task_policy_formation_and_cleanup`。

**理论进展、问题硬伤与结论。** 若矛盾任务的“选择支持项率”远高于0.5，最简单解释是任务政策/指令跟随失败，而不是12种语言关系同时消失。中层margin lens只是把同一final norm和输出权重应用到早层状态，不是模型原生早退；归一化token箱会混合不等长token角色；连续族掩码来自原型与全局均值，仍混入模板和关系词。因此下一步干预只能称`select_supported`行为合格子域的探索，不能救活整个双任务语义机制。

**下一阶段路线判断。** 目标相同，自动对12个已通过支持任务族执行连续全坐标多剂量匹配删除、错族、坐标置乱、等范数随机、错层和原型救援；只有匹配删除选择性大于全部控制且救援恢复，才获得局部因果候选。Phase2345未完整发布的7.17GB HiddenState边界场和14.35GB贡献轨迹已按用户要求删除；q0/q23/final、全token参考、路线护照及本Phase全层均值仍可在客户端查看。
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
    rows = io.read_rows(MATERIAL)
    route = json.loads((P2346 / "analysis/final.json").read_text(encoding="utf-8"))
    selected_qpoint = int(route["analysis"]["selected"]["qpoint"])
    freeze = {"frozen_before_diagnostic": True, "selected_qpoint_from_phase2346": selected_qpoint,
              "token_bins": NORMALIZED_TOKEN_BINS, "lockbox": LOCKBOX,
              "cleanup_after_verified_publication": [str(STATES), str(TRAJECTORY)]}
    save(OUT / "config/frozen_contract.json", freeze)
    policy = behavior_policy(rows)
    margins = margin_trajectory(rows)
    checkpoint_values, checkpoint_metadata = checkpoint_means(rows)
    token_values, token_metadata, token_audit = token_bin_means()
    masks = save_continuous_masks(rows, selected_qpoint)
    datasets = [
        publish("c8601_qwen4b_task_policy_family_checkpoint_means",
                "Qwen3-4B supported/contradicted family fields across all checkpoints",
                checkpoint_values, checkpoint_metadata, "task_policy_family_checkpoint_means_v1",
                "12 families x 2 task policies x 38 checkpoints; raw-H and absolute-contribution views"),
        publish("c8602_qwen4b_task_policy_all_token_normalized_bins",
                "Qwen3-4B task-policy all-token formation atlas",
                token_values, token_metadata, "task_policy_all_token_bins_v1",
                "48 reference prompts grouped into 8 normalized token-position bins across all checkpoints"),
    ]
    verification = [atlas.verify(dataset) for dataset in datasets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not (verified and build["passed"]):
        raise RuntimeError(("publication_failed_before_cleanup", verification, build))
    cleanup = clean_raw_fields()
    checks = {"policy_rows": sum(value["rows"] for value in policy["overall"].values()) == len(rows),
              "checkpoint_coordinates": checkpoint_values.shape[1] == 2560,
              "token_coordinates": token_values.shape[1] == 2560, "assets_verified": verified,
              "frontend_build": build["passed"], "cleanup_complete": all(row["deleted"] for row in cleanup["deleted"])}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "behavior_policy": policy,
              "margin_trajectory": margins, "token_bin_audit": token_audit, "continuous_masks": masks,
              "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)),
              "verification": verification, "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
              "frontend_build": build, "cleanup": cleanup, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2347_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
