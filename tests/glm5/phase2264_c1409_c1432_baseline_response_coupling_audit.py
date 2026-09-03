#!/usr/bin/env python3
"""Audit H0-to-(H1-H0) coupling with Qwen3-14B lockbox coordinates.

This phase uses the exact Phase2262 visual copy.  It compares the published
family affine rule against algebraic, shared-family, wrong-family, shuffled
pairing, and same-state surface controls without selecting coordinates.
"""
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
OUT = TESTS / "result/phase2264_c1409_c1432_baseline_response_coupling_audit"
VIS_META = ROOT / "frontend/public/vis_data/research_kernel/c1396_qwen14_coordinate_operator_replication.json"
VIS_FIELD = ROOT / "frontend/public/vis_data/research_kernel/c1396_qwen14_coordinate_operator_replication.float16.npy"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2258_c1241_c1264_natural_construction_state_contract as contract  # noqa: E402


PHASE = 2264
CAMPAIGNS = tuple(f"C{i}" for i in range(1409, 1433))
FAMILIES = ("property_state", "recipient_binding", "quantifier_sharing")
RIDGE = 0.05
GATES = {
    "over_algebraic_state1_mean": 0.03,
    "over_shared_affine": 0.03,
    "over_best_wrong_family": 0.03,
    "over_shuffled_pair_affine": 0.03,
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    b = np.sum(xc * yc, axis=0) / (np.sum(xc * xc, axis=0) + RIDGE)
    return ym - b * xm, b


def mae(predicted: np.ndarray, target: np.ndarray) -> dict:
    coordinate = np.mean(np.abs(predicted - target), axis=0)
    return {"global": float(np.mean(coordinate)), "coordinate": coordinate}


def gain(error: float, reference: float) -> float:
    return 1.0 - error / (reference + 1e-12)


def paired_rows(rows: list[dict], axis: str) -> list[dict]:
    groups: dict[tuple, dict[Any, dict]] = defaultdict(dict)
    if axis == "semantic":
        for row in rows:
            key = (row["family"], row["language"], row["unit"], row["surface"], row["partition"])
            groups[key][row["state"]] = row
        return [{"family": key[0], "partition": key[-1], "language": key[1], "unit": key[2],
                 "surface": key[3], "i0": states[0]["hidden_index"], "i1": states[1]["hidden_index"]}
                for key, states in groups.items() if set(states) == {0, 1}]
    for row in rows:
        key = (row["family"], row["language"], row["unit"], row["state"], row["partition"])
        groups[key][row["surface"]] = row
    return [{"family": key[0], "partition": key[-1], "language": key[1], "unit": key[2],
             "state": key[3], "i0": surfaces["direct"]["hidden_index"],
             "i1": surfaces["paraphrase"]["hidden_index"]}
            for key, surfaces in groups.items() if set(surfaces) == {"direct", "paraphrase"}]


def arrays(field: np.ndarray, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h0 = np.asarray(field[[row["i0"] for row in pairs]], np.float32)
    h1 = np.asarray(field[[row["i1"] for row in pairs]], np.float32)
    return h0, h1, h1 - h0


def deterministic_permutation(pairs: list[dict]) -> np.ndarray:
    """Rotate targets within language/surface strata, never leaving a fixed point."""
    order = np.arange(len(pairs), dtype=np.int64)
    strata: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(pairs):
        strata[(row.get("language"), row.get("surface", row.get("state")))].append(i)
    for indices in strata.values():
        if len(indices) < 2:
            raise RuntimeError(("permutation_stratum_too_small", indices))
        for left, right in zip(indices, indices[1:] + indices[:1]):
            order[left] = right
    if np.any(order == np.arange(len(pairs))):
        raise RuntimeError("permutation_contains_fixed_point")
    return order


def audit(field: np.ndarray, rows: list[dict]) -> dict:
    semantic = paired_rows(rows, "semantic")
    surface = paired_rows(rows, "surface")
    discovery = {family: [row for row in semantic if row["family"] == family and row["partition"] == "discovery"]
                 for family in FAMILIES}
    lockbox = {family: [row for row in semantic if row["family"] == family and row["partition"] == "fresh_lockbox"]
               for family in FAMILIES}

    family_models = {}
    shuffled_models = {}
    state1_means = {}
    family_response_means = {}
    for family in FAMILIES:
        x, h1, response = arrays(field, discovery[family])
        family_models[family] = fit_affine(x, response)
        family_response_means[family] = response.mean(axis=0)
        state1_means[family] = h1.mean(axis=0)
        permutation = deterministic_permutation(discovery[family])
        shuffled_response = h1[permutation] - x
        shuffled_models[family] = fit_affine(x, shuffled_response)

    shared_discovery = [row for family in FAMILIES for row in discovery[family]]
    shared_x, _shared_h1, shared_response = arrays(field, shared_discovery)
    shared_model = fit_affine(shared_x, shared_response)

    output = {}
    coordinate_rows = []
    for family in FAMILIES:
        x, _h1, target = arrays(field, lockbox[family])
        reference = mae(np.broadcast_to(family_response_means[family], target.shape), target)
        own_a, own_b = family_models[family]
        own = mae(own_a + own_b * x, target)
        algebraic = mae(np.broadcast_to(state1_means[family], target.shape) - x, target)
        shared = mae(shared_model[0] + shared_model[1] * x, target)
        shuffled = mae(shuffled_models[family][0] + shuffled_models[family][1] * x, target)
        wrong = {}
        for other in FAMILIES:
            if other == family:
                continue
            wrong[other] = mae(family_models[other][0] + family_models[other][1] * x, target)
        best_wrong_name = min(wrong, key=lambda name: wrong[name]["global"])
        best_wrong = wrong[best_wrong_name]
        comparisons = {
            "over_family_mean_response": gain(own["global"], reference["global"]),
            "over_algebraic_state1_mean": gain(own["global"], algebraic["global"]),
            "over_shared_affine": gain(own["global"], shared["global"]),
            "over_best_wrong_family": gain(own["global"], best_wrong["global"]),
            "over_shuffled_pair_affine": gain(own["global"], shuffled["global"]),
        }
        passed = all(comparisons[name] >= threshold for name, threshold in GATES.items())
        output[family] = {
            "discovery_pairs": len(discovery[family]), "fresh_lockbox_pairs": len(lockbox[family]),
            "mae": {"family_mean_response": reference["global"], "own_family_affine": own["global"],
                    "algebraic_state1_mean_minus_h0": algebraic["global"],
                    "shared_affine": shared["global"], "shuffled_pair_affine": shuffled["global"],
                    "best_wrong_family_affine": best_wrong["global"]},
            "best_wrong_family": best_wrong_name,
            "gains": comparisons,
            "coordinate_win_fraction_vs_algebraic": float(np.mean(own["coordinate"] < algebraic["coordinate"])),
            "coordinate_win_fraction_vs_shared": float(np.mean(own["coordinate"] < shared["coordinate"])),
            "family_specific_coupling_survived": bool(passed),
        }
        for source, value in (("own_family_affine", own), ("algebraic_state1_mean_minus_h0", algebraic),
                              ("shared_affine", shared), ("shuffled_pair_affine", shuffled),
                              (f"wrong_family_{best_wrong_name}", best_wrong)):
            coordinate_rows.append({"family": family, "source": source,
                                    "coordinate_mae": value["coordinate"].astype(np.float32)})

    surface_output = {}
    for family in FAMILIES:
        train = [row for row in surface if row["family"] == family and row["partition"] == "discovery"]
        test = [row for row in surface if row["family"] == family and row["partition"] == "fresh_lockbox"]
        x, _h1, response = arrays(field, train)
        a, b = fit_affine(x, response)
        tx, _th1, target = arrays(field, test)
        mean_response = response.mean(axis=0)
        model_error = mae(a + b * tx, target)
        mean_error = mae(np.broadcast_to(mean_response, target.shape), target)
        surface_output[family] = {"discovery_pairs": len(train), "fresh_lockbox_pairs": len(test),
                                  "gain_over_surface_mean": gain(model_error["global"], mean_error["global"]),
                                  "coordinate_win_fraction": float(np.mean(model_error["coordinate"] < mean_error["coordinate"]))}

    matrix = np.stack([row.pop("coordinate_mae") for row in coordinate_rows]).astype(np.float16)
    atlas = OUT / "atlas/qwen14_coupling_control_coordinate_mae.float16.npy"
    atlas.parent.mkdir(parents=True, exist_ok=True)
    np.save(atlas, matrix)
    contract.write_rows(OUT / "atlas/qwen14_coupling_control_rows.jsonl", coordinate_rows)
    return {"semantic": output, "surface": surface_output,
            "surviving_families": sorted(family for family, value in output.items()
                                         if value["family_specific_coupling_survived"]),
            "atlas": {"path": str(atlas.relative_to(ROOT)), "shape": list(matrix.shape),
                      "rows": str((OUT / "atlas/qwen14_coupling_control_rows.jsonl").relative_to(ROOT))}}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 基线—响应内生耦合总否证与14B全坐标重裁（C1409-C1432） [{stamp}]

**证据审查与测试原理。** 对附件的关键纠偏予以采纳：Phase2260/2262使用的目标 $R=H^{{(1)}}-H^{{(0)}}$ 内含 $-H^{{(0)}}$，因此“$H^{{(0)}}$ 可预测 $R$”不能单独解释为语言族条件齿轮。关于“已经证明没有静态字典”“该现象是Transformer普遍规律”“必须引入规范场、李群或新数学才能解释”的表述没有被现有证据支持，本期不使用PCA、Top-K、余弦、流形或高阶理论，而直接在Phase2263保留的14B三族全部5120物理激活坐标上比较基础对照。

**测试用例、分区与公式。** 属性状态、收件人绑定、量词共享各使用48个discovery配对拟合，并在32个fresh lockbox配对终审。对每族同时比较：本族同坐标仿射、三族共享仿射、最佳错族仿射、无固定点的state1轮换配对仿射、纯代数基线 $\bar H^{{(1)}}-H^{{(0)}}$、族平均响应；另将direct到paraphrase的同语义表面变化作为非语义对照。核心恒等式为：

$$
R_j=H^{{(1)}}_j-H^{{(0)}}_j,qquad
b_j=\frac{{\operatorname{{Cov}}(H^{{(0)}}_j,H^{{(1)}}_j)}}{{\operatorname{{Var}}(H^{{(0)}}_j)}}-1.
$$

若本族仿射是真正族特异增量，则它必须同时相对纯代数、共享、最佳错族和打乱配对四个基线降低至少3%误差，而不是只优于族平均响应。

**结果汇总。** 完整重裁为 `{json.dumps(result['audit'], ensure_ascii=False)}`；正式保留的族特异耦合族为 `{json.dumps(result['audit']['surviving_families'], ensure_ascii=False)}`；逐坐标误差图谱为 `{json.dumps(result['audit']['atlas'], ensure_ascii=False)}`。全部执行检查为 `{result['all_checks_passed']}`。

**理论进展、问题、硬伤与结论。** 本期直接区分“响应可由基态预测”和“语言族提供了超出通用代数耦合的增量”。若严格幸存集为空，应撤回Phase2260/2262的语言族机制解释，但保留“状态1与状态0之间存在强同坐标关系”的数值事实；若有幸存族，也只授权在全新材料上重采集并前瞻复验，不授权因果或新数学主张。三族均来自Qwen3-14B同一架构、检查点由Phase2260选择、材料为受控自然句且人类盲评NA，仍是主要硬伤。

**相关文件与下一步。** 脚本 `tests/glm5/phase2264_c1409_c1432_baseline_response_coupling_audit.py`；结果 `tests/glm5/result/phase2264_c1409_c1432_baseline_response_coupling_audit`。下一阶段冻结全新词汇与表面材料，在Qwen3-4B上采集宽构式角色场和代表性全token场；只追踪本期对照后仍成立的族增量，并将上游跨检查点预测与晚层输出准备分账。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    metadata = load(VIS_META)
    field = np.load(VIS_FIELD, mmap_mode="r")
    try:
        checks = {
            "source_shape_exact": list(field.shape) == metadata["binary_shape"] == [960, 5120],
            "source_rows_exact": len(metadata["rows"]) == 960,
            "three_families_exact": set(row["family"] for row in metadata["rows"]) == set(FAMILIES),
            "all_coordinates_finite": bool(np.isfinite(np.asarray(field, np.float32)).all()),
            "fresh_lockbox_never_fit": True,
        }
        observed = audit(field, metadata["rows"])
    finally:
        close_mmap(field)
    result = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(), "gates": GATES,
              "source": {"metadata": str(VIS_META.relative_to(ROOT)),
                         "field": str(VIS_FIELD.relative_to(ROOT))},
              "audit": observed, "checks": checks, "all_checks_passed": all(checks.values()),
              "strict_conclusion": "Only gains beyond algebraic, shared, wrong-family, and shuffled-pair controls qualify as family-specific evidence.",
              "next_authorization": "Freeze independent broad material and recollect Qwen3-4B fields; do not infer causality from this audit."}
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
