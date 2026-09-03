#!/usr/bin/env python3
"""Orthogonal controls for the twenty-family mid-layer coordinate-use texture."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2340 = RESULT / "phase2340_c7601_c7840_twenty_family_fixed_interface_atlas"
OUT = RESULT / "phase2341_c7841_c8000_coordinate_texture_orthogonal_controls"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
ROWS = P2340 / "material/twenty_family_fixed_ab.jsonl"
STATES = P2340 / "raw/boundary_all_checkpoints.float16.npy"
TRAJECTORY = P2340 / "derived/layerwise_coordinate_contribution.float32.npy"
PARENT_FINAL = P2340 / "analysis/final.json"
PARENT_TRAJECTORY = P2340 / "analysis/checkpoint_trajectory.jsonl"
PASSPORT = OUT / "derived/qstar_control_passport.float32.npy"
PHASE = 2341
CAMPAIGN = "C7841-C8000"
TRAIN_PARTITIONS = ("discovery", "confirmation")
TEST_PARTITION = "fresh_lockbox"
EPS = 1e-12

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2330_c6081_c6200_language_family_atlas_contract as contract  # noqa: E402
import phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls as layer_control  # noqa: E402

FAMILIES = tuple(contract.FAMILIES)

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


def select_q(parent_trajectory: list[dict]) -> dict:
    candidates = []
    for row in parent_trajectory:
        fresh = row["partitions"]["fresh_confirmation"]
        left = fresh["qualified_original_to_swap_abs"]
        right = fresh["qualified_swap_to_original_abs"]
        candidates.append({"qpoint": row["qpoint"], "min_accuracy": min(left["accuracy"], right["accuracy"]),
                           "mean_accuracy": (left["accuracy"] + right["accuracy"]) / 2})
    selected = max(candidates, key=lambda row: (row["min_accuracy"], row["mean_accuracy"], -row["qpoint"]))
    return {"selection_partition": "fresh_confirmation", "selection_rule": "max min cross-option accuracy, then mean, then earliest q",
            "selected": selected, "top_five": sorted(candidates, key=lambda row: (-row["min_accuracy"], -row["mean_accuracy"], row["qpoint"]))[:5]}


def classify(train: np.ndarray, test: np.ndarray, train_rows: list[dict], test_rows: list[dict], labels: tuple[str, ...],
             train_filter: Callable[[dict], bool], test_filter: Callable[[dict], bool]) -> dict:
    prototypes = []
    train_counts = {}
    for label in labels:
        indices = [i for i, row in enumerate(train_rows) if row["family"] == label and train_filter(row)]
        if not indices:
            raise RuntimeError(("empty_train_cell", label))
        train_counts[label] = len(indices)
        prototypes.append(train[indices].astype(np.float64).mean(axis=0))
    prototypes = np.stack(prototypes)
    indices = [i for i, row in enumerate(test_rows) if row["family"] in labels and test_filter(row)]
    if not indices:
        raise RuntimeError("empty_test_cell")
    actual = test[indices].astype(np.float64)
    distances = np.sum(np.square(actual), axis=1, keepdims=True) + np.sum(np.square(prototypes), axis=1)[None, :] - 2 * actual @ prototypes.T
    distances = np.maximum(distances, 0)
    correct = np.asarray([labels.index(test_rows[i]["family"]) for i in indices], dtype=np.int64)
    predicted = np.argmin(distances, axis=1)
    correct_distance = distances[np.arange(len(indices)), correct]
    masked = distances.copy(); masked[np.arange(len(indices)), correct] = np.inf
    best_wrong = np.min(masked, axis=1)
    return {"rows": len(indices), "labels": len(labels), "chance": 1 / len(labels),
            "accuracy": float(np.mean(predicted == correct)),
            "median_correct_over_best_wrong_ratio": float(np.median(correct_distance / (best_wrong + EPS))),
            "min_train_rows_per_family": min(train_counts.values())}


def standardize(field: np.ndarray) -> np.ndarray:
    mean = np.mean(field, axis=1, keepdims=True)
    std = np.std(field, axis=1, keepdims=True)
    return (field - mean) / (std + 1e-8)


def feature_views(field: np.ndarray, prompt_rows: list[dict]) -> dict[str, np.ndarray]:
    abs_field = np.abs(field.astype(np.float32))
    return {
        "full_coordinate_abs": abs_field,
        "row_sorted_abs_distribution": np.sort(abs_field, axis=1),
        "row_standardized_coordinate_abs": standardize(abs_field),
        "two_scalar_magnitude": np.stack((np.mean(abs_field, axis=1), np.sqrt(np.mean(np.square(abs_field), axis=1))), axis=1),
        "prompt_length_scalar": np.asarray([[len(row["future_prompt_ids"])] for row in prompt_rows], dtype=np.float32),
    }


def direction_controls(original_views: dict[str, np.ndarray], swapped_views: dict[str, np.ndarray],
                       original_rows: list[dict], swapped_rows: list[dict], labels: tuple[str, ...]) -> dict:
    controls = {}
    directions = {
        "cross_option_all": (
            lambda row: row["partition"] in TRAIN_PARTITIONS,
            lambda row: row["partition"] == TEST_PARTITION,
        ),
        "en_to_zh": (
            lambda row: row["partition"] in TRAIN_PARTITIONS and row["language"] == "en",
            lambda row: row["partition"] == TEST_PARTITION and row["language"] == "zh",
        ),
        "zh_to_en": (
            lambda row: row["partition"] in TRAIN_PARTITIONS and row["language"] == "zh",
            lambda row: row["partition"] == TEST_PARTITION and row["language"] == "en",
        ),
        "narrative_to_reported": (
            lambda row: row["partition"] in TRAIN_PARTITIONS and row["surface"] == "narrative",
            lambda row: row["partition"] == TEST_PARTITION and row["surface"] == "reported",
        ),
        "reported_to_narrative": (
            lambda row: row["partition"] in TRAIN_PARTITIONS and row["surface"] == "reported",
            lambda row: row["partition"] == TEST_PARTITION and row["surface"] == "narrative",
        ),
        "en_narrative_to_zh_reported": (
            lambda row: row["partition"] in TRAIN_PARTITIONS and row["language"] == "en" and row["surface"] == "narrative",
            lambda row: row["partition"] == TEST_PARTITION and row["language"] == "zh" and row["surface"] == "reported",
        ),
        "zh_reported_to_en_narrative": (
            lambda row: row["partition"] in TRAIN_PARTITIONS and row["language"] == "zh" and row["surface"] == "reported",
            lambda row: row["partition"] == TEST_PARTITION and row["language"] == "en" and row["surface"] == "narrative",
        ),
    }
    for view_name in original_views:
        controls[view_name] = {}
        for direction, (train_filter, test_filter) in directions.items():
            left = classify(original_views[view_name], swapped_views[view_name], original_rows, swapped_rows, labels, train_filter, test_filter)
            right = classify(swapped_views[view_name], original_views[view_name], swapped_rows, original_rows, labels, train_filter, test_filter)
            controls[view_name][direction] = {"original_to_swapped": left, "swapped_to_original": right,
                                                   "minimum_accuracy": min(left["accuracy"], right["accuracy"])}
    return controls


def analyze(rows: list[dict], qpoint: int, qualified: tuple[str, ...]) -> tuple[dict, np.ndarray, list[dict]]:
    source_count = len(rows) // 2
    original_rows, swapped_rows = rows[:source_count], rows[source_count:]
    trajectory = np.load(TRAJECTORY, mmap_mode="r")
    states = np.load(STATES, mmap_mode="r")
    original_field = trajectory[:source_count, qpoint].astype(np.float32)
    swapped_field = trajectory[source_count:, qpoint].astype(np.float32)
    original_views = feature_views(original_field, original_rows)
    swapped_views = feature_views(swapped_field, swapped_rows)
    qualified_controls = direction_controls(original_views, swapped_views, original_rows, swapped_rows, qualified)
    all20_controls = direction_controls(original_views, swapped_views, original_rows, swapped_rows, FAMILIES)
    original_hidden = states[:source_count, qpoint].astype(np.float32)
    swapped_hidden = states[source_count:, qpoint].astype(np.float32)
    hidden_views = {"full_hiddenstate": original_hidden, "row_standardized_hiddenstate": standardize(original_hidden)}
    swapped_hidden_views = {"full_hiddenstate": swapped_hidden, "row_standardized_hiddenstate": standardize(swapped_hidden)}
    hidden_controls = direction_controls(hidden_views, swapped_hidden_views, original_rows, swapped_rows, qualified)
    full_min = qualified_controls["full_coordinate_abs"]["cross_option_all"]["minimum_accuracy"]
    sorted_min = qualified_controls["row_sorted_abs_distribution"]["cross_option_all"]["minimum_accuracy"]
    standardized_min = qualified_controls["row_standardized_coordinate_abs"]["cross_option_all"]["minimum_accuracy"]
    length_min = qualified_controls["prompt_length_scalar"]["cross_option_all"]["minimum_accuracy"]
    cross_language_min = min(
        qualified_controls["full_coordinate_abs"]["en_to_zh"]["minimum_accuracy"],
        qualified_controls["full_coordinate_abs"]["zh_to_en"]["minimum_accuracy"],
    )
    cross_surface_min = min(
        qualified_controls["full_coordinate_abs"]["narrative_to_reported"]["minimum_accuracy"],
        qualified_controls["full_coordinate_abs"]["reported_to_narrative"]["minimum_accuracy"],
    )
    gate = {
        "full_coordinate_cross_option_min": full_min,
        "sorted_distribution_cross_option_min": sorted_min,
        "coordinate_increment_over_sorted": full_min - sorted_min,
        "row_standardized_cross_option_min": standardized_min,
        "prompt_length_cross_option_min": length_min,
        "cross_language_min": cross_language_min,
        "cross_surface_min": cross_surface_min,
        "threshold_accuracy": 0.30, "threshold_coordinate_increment": 0.10,
        "full_graph_pass": full_min >= 0.30,
        "coordinate_identity_pass": full_min >= sorted_min + 0.10 and standardized_min >= 0.30,
        "language_surface_transfer_pass": cross_language_min >= 0.30 and cross_surface_min >= 0.30,
        "length_control_pass": full_min >= length_min + 0.10,
    }
    gate["passed"] = all((gate["full_graph_pass"], gate["coordinate_identity_pass"],
                           gate["language_surface_transfer_pass"], gate["length_control_pass"]))
    passport_views = ("full_coordinate_abs", "row_sorted_abs_distribution", "row_standardized_coordinate_abs")
    passport = np.concatenate([original_views[name] for name in passport_views] + [swapped_views[name] for name in passport_views], axis=0).astype(np.float32)
    metadata = []
    for condition, rowset in (("original", original_rows), ("option_swapped", swapped_rows)):
        for view_name in passport_views:
            metadata.extend({"case_id": row["case_id"], "condition": condition, "view": view_name, "family": row["family"],
                             "macrotype": row["macrotype"], "language": row["language"], "surface": row["surface"],
                             "partition": row["partition"], "unit": row["unit"], "state": row["state"], "qpoint": qpoint}
                            for row in rowset)
    result = {"qpoint": qpoint, "qualified_families": list(qualified), "qualified_controls": qualified_controls,
              "all20_controls": all20_controls, "qualified_hidden_controls": hidden_controls,
              "paired_full_coordinate_abs": layer_control.pair_metrics(np.abs(original_field), np.abs(swapped_field)),
              "gate": gate}
    for value in (trajectory, states): close_memmap(value)
    return result, passport, metadata


def publish(passport: np.ndarray, metadata: list[dict], qpoint: int) -> dict:
    PASSPORT.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(PASSPORT, mode="w+", dtype=np.float32, shape=passport.shape)
    out[:] = passport; out.flush(); close_memmap(out)
    dataset_id = "c7841_qwen4b_twenty_family_qstar_coordinate_texture_controls"
    binary = VIS / f"{dataset_id}.float32.npy"
    target = atlas.create_binary(binary.name, passport.shape[0], passport.shape[1], np.float32)
    target[:] = passport; target.flush(); close_memmap(target)
    return atlas.write_metadata(
        dataset_id, f"Qwen3-4B q{qpoint} twenty-family coordinate-texture orthogonal controls", binary, metadata,
        "Qwen3-4B-FP16", "coordinate_texture_orthogonal_control_v1", "orthogonal control",
        "full coordinate, sorted distribution, and row-standardized coordinate views",
        "all 2560 coordinates retained in each view",
        {"coordinate_count": 2560, "no_topk": True, "views": ["full_coordinate_abs", "row_sorted_abs_distribution", "row_standardized_coordinate_abs"]},
    )


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 二十族中层坐标纹理的语言、表述、长度与分布形状正交排雷（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** Phase2340发现最终层广度门未过，但中层存在强族识别。本阶段不再加载模型，先用fresh_confirmation按冻结规则选择检查点，再只在fresh_lockbox裁决；避免再次用lockbox挑层。对17个行为合格族及全部20族分别执行：原始↔选项交换、英文→中文、中文→英文、narrative→reported、reported→narrative、语言与表述同时交换。主表示保留全部2560个具体坐标；另设三个排雷对照：逐行排序后仅保留数值分布、逐行标准化后去除总体平移/尺度、仅两项幅度标量及仅prompt长度。

$$
a_{{i,j}}=|c_{{i,q^*,j}}|,quad
s_i=\operatorname{{sort}}(a_i),quad
u_{{i,j}}=\frac{{a_{{i,j}}-\bar a_i}}{{\operatorname{{std}}(a_i)+10^{{-8}}}}.
$$

$$
\operatorname{{CoordinateSpecific}}=[A(a)\ge0.30]\land[A(a)-A(s)\ge0.10]\land[A(u)\ge0.30].
$$

**结果汇总与相关文件。** 选层 `{json.dumps(result['selection'], ensure_ascii=False)}`；正交结果 `{json.dumps(result['analysis'], ensure_ascii=False)}`；发布 `{json.dumps(result['dataset'], ensure_ascii=False)}`；核验 `{json.dumps(result['verification'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2341_c7841_c8000_coordinate_texture_orthogonal_controls.py`；结果 `tests/glm5/result/phase2341_c7841_c8000_coordinate_texture_orthogonal_controls`。

**理论进展、问题硬伤与结论。** full坐标胜过排序分布，才支持“固定坐标身份参与”；若排序分布已解释大部分，则应把现象降级为整体幅度/直方图纹理。逐行标准化通过只能说明相对坐标形状有信息，仍可能是词汇/模板；跨语言与跨表述通过才减少这两类解释。所有对照都是欧氏原型的基础分析，没有把高级流形术语预置为答案。q点由同一旧数据的fresh_confirmation选择，仍不是全新材料；族模板、训练数据与选择题任务共享，不能称语义机制闭合。

**下一阶段路线判断。** 若全部正交门通过，下一阶段才为通过族生成完全独立自然构式并冻结该相对层窗；若固定坐标身份门失败，则改研究“全坐标值分布如何随族与任务变化”，而不再声称坐标复用；若跨语言/表述失败，则优先重做同词汇跨任务、同任务跨词汇材料。目标仍相同，但路线由排雷结果决定。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    parent = json.loads(PARENT_FINAL.read_text(encoding="utf-8"))
    rows = read_rows(ROWS)
    selection = select_q(read_rows(PARENT_TRAJECTORY))
    qpoint = int(selection["selected"]["qpoint"])
    qualified = tuple(parent["behavior"]["qualified"])
    freeze = {"frozen_before_slice_controls": True, "q_selection": selection["selection_rule"],
              "selection_partition": "fresh_confirmation", "test_partition": TEST_PARTITION,
              "qualified_family_rule_inherited": True, "accuracy_threshold": 0.30,
              "coordinate_increment_over_sorted": 0.10}
    save(OUT / "config/frozen_contract.json", freeze)
    analysis, passport, metadata = analyze(rows, qpoint, qualified)
    dataset = publish(passport, metadata, qpoint)
    verification = atlas.verify(dataset)
    verified = all(value for key, value in verification.items() if key != "id")
    if not verified: raise RuntimeError(("verification_failed", verification))
    catalog = atlas.update_catalog([dataset])
    build = atlas.frontend_build()
    if not build["passed"]: raise RuntimeError(("frontend_build_failed", build))
    checks = {"parent_valid": parent["all_checks_passed"], "selected_without_fresh_lockbox": selection["selection_partition"] == "fresh_confirmation",
              "passport_shape": list(passport.shape) == [11520, 2560], "asset_verified": verified, "frontend_build": build["passed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "selection": selection, "analysis": analysis,
              "dataset": json.loads(json.dumps(dataset, ensure_ascii=False, default=str)), "verification": verification,
              "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)), "frontend_build": build,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]: raise RuntimeError(("phase2341_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
