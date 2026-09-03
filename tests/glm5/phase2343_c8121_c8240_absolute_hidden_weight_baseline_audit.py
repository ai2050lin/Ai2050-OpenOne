#!/usr/bin/env python3
"""Audit Phase 2330-2342 claims and test |H| / output-weight controls."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2340 = RESULT / "phase2340_c7601_c7840_twenty_family_fixed_interface_atlas"
P2341 = RESULT / "phase2341_c7841_c8000_coordinate_texture_orthogonal_controls"
OUT = RESULT / "phase2343_c8121_c8240_absolute_hidden_weight_baseline_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
ROWS = P2340 / "material/twenty_family_fixed_ab.jsonl"
STATES = P2340 / "raw/boundary_all_checkpoints.float16.npy"
TRAJECTORY = P2340 / "derived/layerwise_coordinate_contribution.float32.npy"
GAMMA = P2340 / "raw/final_norm_gamma.float32.npy"
AB_WEIGHT = P2340 / "raw/a_minus_b_weight.float32.npy"
PHASE = 2343
CAMPAIGN = "C8121-C8240"
TRAIN_PARTITIONS = ("discovery", "confirmation")
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\e35f0bd3-58ec-4a4f-9d6e-66437c398c5d\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\dfdf3b97-a3be-4a72-9959-e5444085824a\pasted-text.txt"),
)

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls as layer_control  # noqa: E402
import phase2340_c7601_c7840_twenty_family_fixed_interface_atlas as p2340  # noqa: E402

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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def cross_option(field: np.ndarray, rows: list[dict], labels: tuple[str, ...]) -> dict:
    source_count = len(rows) // 2
    original_rows, swapped_rows = rows[:source_count], rows[source_count:]
    original, swapped = field[:source_count], field[source_count:]
    left = p2340.vector_classify(
        original, swapped, original_rows, swapped_rows, labels, "family", TRAIN_PARTITIONS, "fresh_lockbox"
    )
    right = p2340.vector_classify(
        swapped, original, swapped_rows, original_rows, labels, "family", TRAIN_PARTITIONS, "fresh_lockbox"
    )
    return {
        "original_to_swap": left,
        "swap_to_original": right,
        "minimum_accuracy": min(left["accuracy"], right["accuracy"]),
        "mean_accuracy": (left["accuracy"] + right["accuracy"]) / 2,
        "maximum_median_correct_over_best_wrong_ratio": max(
            left["median_correct_over_best_wrong_ratio"], right["median_correct_over_best_wrong_ratio"]
        ),
    }


def normalized_hidden(states: np.ndarray, q: int, gamma: np.ndarray, eps: float) -> np.ndarray:
    value = states[:, q].astype(np.float32)
    if q == states.shape[1] - 1:
        return value
    return layer_control.rms_norm(value, gamma, eps)


def audit_claims() -> dict:
    return {
        "attachments": [
            {"path": str(path), "sha256": sha256(path), "lines": len(path.read_text(encoding="utf-8").splitlines())}
            for path in ATTACHMENTS
        ],
        "retained": [
            "The fixed A/B interface removes natural-answer-token identity as a between-family confound.",
            "The observed family signal is distributed over the measured coordinate field and concrete coordinate identity matters within the old task ecology.",
            "Cross-language transfer failed and no language-invariant semantic code or causal closure has been shown.",
            "Lexicon, task, language, surface and output interface must be crossed before causal intervention.",
        ],
        "corrected": [
            "A/B removes natural answer identity, but it does not isolate pure logic: prompt lexicon, family template and multiple-choice policy remain.",
            "Qwen14B and GLM4 replicated a fixed-interface functional classification phenomenon; DeepSeek7B failed the frozen final-layer gate, and raw coordinates were never aligned across models.",
            "A high family accuracy does not prove that a coordinate coalition computes the family; prediction, deletion and rescue remain absent.",
            "The absolute contribution field is a fixed diagonal reweighting of absolute normalized HiddenState and needs direct and shuffled-weight controls.",
        ],
        "rejected": [
            "The experiment proved a pure logical routing layer independent of output preparation.",
            "It disproved single-neuron or sparse mechanisms in general.",
            "It established energy routing, an attractor basin, a geodesic, Riemannian curvature or a causal manifold.",
            "Exact coordinate Shapley values are an identified or uniquely appropriate next step.",
            "Qwen, GLM and DeepSeek share a demonstrated diffeomorphic mechanism.",
        ],
    }


def analyze(rows: list[dict], qualified: tuple[str, ...], qpoint: int) -> tuple[dict, np.ndarray, list[dict]]:
    states = np.load(STATES, mmap_mode="r")
    trajectory = np.load(TRAJECTORY, mmap_mode="r")
    gamma = np.load(GAMMA).astype(np.float32)
    abs_weight = np.abs(np.load(AB_WEIGHT).astype(np.float32))
    eps = 1e-6
    layer_records = []
    for q in range(states.shape[1]):
        abs_h = np.abs(normalized_hidden(states, q, gamma, eps))
        true_weighted = np.abs(trajectory[:, q].astype(np.float32))
        unweighted = cross_option(abs_h, rows, qualified)
        weighted = cross_option(true_weighted, rows, qualified)
        layer_records.append({
            "qpoint": q,
            "checkpoint_kind": "embedding" if q == 0 else "final_norm" if q == states.shape[1] - 1 else "block",
            "absolute_normalized_hidden": unweighted,
            "true_output_weighted_absolute_hidden": weighted,
            "weighted_minus_unweighted_min_accuracy": weighted["minimum_accuracy"] - unweighted["minimum_accuracy"],
        })
        print(f"[phase2343 layers] {q}/{states.shape[1] - 1}", flush=True)

    abs_h = np.abs(normalized_hidden(states, qpoint, gamma, eps))
    true_field = abs_h * abs_weight[None, :]
    direct = {
        "absolute_normalized_hidden": cross_option(abs_h, rows, qualified),
        "true_output_weighted": cross_option(true_field, rows, qualified),
        "uniform_equal_l2_weight": cross_option(abs_h * (np.linalg.norm(abs_weight) / np.sqrt(abs_weight.size)), rows, qualified),
    }
    rng = np.random.default_rng(8121)
    permutation = []
    random_equal_l2 = []
    representative_perm = None
    representative_random = None
    for index in range(32):
        perm = rng.permutation(abs_weight)
        random_weight = np.abs(rng.standard_normal(abs_weight.size)).astype(np.float32)
        random_weight *= np.linalg.norm(abs_weight) / max(np.linalg.norm(random_weight), 1e-12)
        perm_result = cross_option(abs_h * perm[None, :], rows, qualified)
        random_result = cross_option(abs_h * random_weight[None, :], rows, qualified)
        permutation.append({"replicate": index, "minimum_accuracy": perm_result["minimum_accuracy"],
                            "mean_accuracy": perm_result["mean_accuracy"]})
        random_equal_l2.append({"replicate": index, "minimum_accuracy": random_result["minimum_accuracy"],
                                "mean_accuracy": random_result["mean_accuracy"]})
        if index == 0:
            representative_perm = abs_h * perm[None, :]
            representative_random = abs_h * random_weight[None, :]
        print(f"[phase2343 weight controls] {index + 1}/32", flush=True)
    true_min = direct["true_output_weighted"]["minimum_accuracy"]
    perm_values = np.asarray([row["minimum_accuracy"] for row in permutation])
    random_values = np.asarray([row["minimum_accuracy"] for row in random_equal_l2])
    direct["permuted_output_weight_controls"] = {
        "replicates": permutation,
        "median_minimum_accuracy": float(np.median(perm_values)),
        "maximum_minimum_accuracy": float(np.max(perm_values)),
        "true_minus_median": float(true_min - np.median(perm_values)),
        "true_rank_fraction_strictly_better": float(np.mean(true_min > perm_values)),
    }
    direct["random_equal_l2_positive_weight_controls"] = {
        "replicates": random_equal_l2,
        "median_minimum_accuracy": float(np.median(random_values)),
        "maximum_minimum_accuracy": float(np.max(random_values)),
        "true_minus_median": float(true_min - np.median(random_values)),
        "true_rank_fraction_strictly_better": float(np.mean(true_min > random_values)),
    }
    direct["frozen_gate"] = {
        "true_minimum_accuracy_at_least_0_30": true_min >= 0.30,
        "true_beats_unweighted_by_0_05": true_min >= direct["absolute_normalized_hidden"]["minimum_accuracy"] + 0.05,
        "true_beats_all_permuted": bool(true_min > np.max(perm_values)),
        "true_beats_all_random_equal_l2": bool(true_min > np.max(random_values)),
    }
    direct["frozen_gate"]["output_weight_specific_functional_alignment_passed"] = all(direct["frozen_gate"].values())

    views = [
        ("absolute_normalized_hidden", abs_h),
        ("true_output_weighted", true_field),
        ("coordinate_permuted_output_weight", representative_perm),
        ("random_equal_l2_positive_weight", representative_random),
    ]
    passport = np.concatenate([value.astype(np.float32) for _, value in views], axis=0)
    metadata = []
    for view, _ in views:
        metadata.extend({
            "case_id": row["case_id"], "view": view, "qpoint": qpoint, "family": row["family"],
            "macrotype": row["macrotype"], "language": row["language"], "surface": row["surface"],
            "partition": row["partition"], "unit": row["unit"], "state": row["state"],
            "condition": row["condition"], "target_code": row["target_code"],
        } for row in rows)
    close_memmap(states)
    close_memmap(trajectory)
    return {"qpoint": qpoint, "full_layer_trajectory": layer_records, "qpoint_controls": direct}, passport, metadata


def publish(passport: np.ndarray, metadata: list[dict], qpoint: int) -> dict:
    dataset_id = "c8121_qwen4b_absolute_hidden_output_weight_control_passport"
    binary = VIS / f"{dataset_id}.float32.npy"
    out = atlas.create_binary(binary.name, passport.shape[0], passport.shape[1], np.float32)
    out[:] = passport
    out.flush()
    close_memmap(out)
    return atlas.write_metadata(
        dataset_id,
        f"Qwen3-4B q{qpoint} |HiddenState| and output-weight control passport",
        binary,
        metadata,
        "Qwen3-4B-FP16",
        "absolute_hidden_output_weight_controls_v1",
        "direct test of whether the A/B unembedding weights add family information beyond absolute normalized HiddenState",
        "17 behavior-qualified families, original/option-swapped, untouched fresh lockbox",
        "Full 2560-coordinate fields for |H|, true |Delta W|, one coordinate permutation and one equal-L2 random positive weighting.",
        {"coordinate_count": 2560, "no_topk": True, "full_coordinate": True, "qpoint": qpoint},
    )


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    controls = result["analysis"]["qpoint_controls"]
    record = rf"""

## Phase {PHASE}: 附件证据重审与绝对HiddenState—输出权重基线裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对两份Phase2330–2342总结逐条区分事实、有限推断和无证据理论；附件SHA256与裁决为 `{json.dumps(result['claim_audit'], ensure_ascii=False)}`。复用Phase2340全部3840条固定A/B、选项交换、20族全场，只在17个行为合格族上裁决。对每层把原来的绝对贡献写成固定对角重加权，并直接比较未加权、真实A/B输出权重、32个坐标置乱权重和32个等L2随机正权重；所有分类均以discovery+confirmation建原型、只在fresh_lockbox裁决，不把2560坐标当独立样本。

$$
|c_{{i,q,j}}|=|\operatorname{{RMSNorm}}(h_{{i,q}})_j|\,|W_{{A,j}}-W_{{B,j}}|
=|\tilde h_{{i,q,j}}|d_j.
$$

$$
G_{{W}}=[A(d)>A(\mathbf 1)+0.05]\land[A(d)>\max_\pi A(d_\pi)]
\land[A(d)>\max_r A(d^{{rand}}_r)].
$$

**结果汇总与相关文件。** q点裁决 `{json.dumps(controls, ensure_ascii=False)}`；全层记录保存在final JSON的`full_layer_trajectory`；全2560坐标客户端资产 `{json.dumps(result['dataset'], ensure_ascii=False)}`；核验 `{json.dumps(result['verification'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2343_c8121_c8240_absolute_hidden_weight_baseline_audit.py`；结果 `tests/glm5/result/phase2343_c8121_c8240_absolute_hidden_weight_baseline_audit`。

**理论进展、问题硬伤与结论。** 固定A/B确实消除了“不同族使用不同自然答案token”的直接混淆，但没有消除提示词、模板、语言和选择题策略。只有真实输出权重严格胜过未加权HiddenState以及全部置乱/等范数控制，才可把强度场升级为“与A/B读出方向特殊对齐”；否则它只是HiddenState幅值纹理的一种固定坐标度量。即便该门通过，也只表示读出相关，不等于坐标执行了逻辑计算。附件二关于纯逻辑路由、吸引子盆地、测地线、曲率、因果流形和跨模型微分同胚的断言均被拒绝；附件一提出的词汇—任务—语言正交化作为下一阶段正确主线保留。

**下一阶段路线判断。** 目标相同，自动进入至少12族、每族16单元的严格平行双语语义图；同词汇跨任务、同任务跨词汇、同语义跨语言和A/B选项交换独立交叉。q23只作为冻结观察窗，同时保存和分析全层全坐标；先观察和制图，未通过正交锁箱的路线不进入因果宣称。
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
    OUT.mkdir(parents=True, exist_ok=True)
    rows = io.read_rows(ROWS)
    parent = json.loads((P2340 / "analysis/final.json").read_text(encoding="utf-8"))
    selection = json.loads((P2341 / "analysis/final.json").read_text(encoding="utf-8"))["analysis"]["qpoint"]
    qualified = tuple(parent["behavior"]["qualified"])
    freeze = {
        "frozen_before_analysis": True,
        "qpoint_from_phase2341": selection,
        "qualified_families_from_phase2340": list(qualified),
        "permuted_replicates": 32,
        "random_equal_l2_replicates": 32,
        "minimum_accuracy": 0.30,
        "required_advantage_over_unweighted": 0.05,
        "random_seed": 8121,
    }
    save(OUT / "config/frozen_contract.json", freeze)
    claim_audit = audit_claims()
    analysis, passport, metadata = analyze(rows, qualified, int(selection))
    dataset = publish(passport, metadata, int(selection))
    verification = atlas.verify(dataset)
    verified = all(value for key, value in verification.items() if key != "id")
    catalog = atlas.update_catalog([dataset])
    build = atlas.frontend_build()
    checks = {
        "attachments_hashed": len(claim_audit["attachments"]) == 2,
        "all_layers_analyzed": len(analysis["full_layer_trajectory"]) == 38,
        "all_coordinates": passport.shape[1] == 2560,
        "all_random_controls": len(analysis["qpoint_controls"]["permuted_output_weight_controls"]["replicates"]) == 32
                               and len(analysis["qpoint_controls"]["random_equal_l2_positive_weight_controls"]["replicates"]) == 32,
        "asset_verified": verified,
        "frontend_build": build["passed"],
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "freeze": freeze,
        "claim_audit": claim_audit,
        "analysis": analysis,
        "dataset": json.loads(json.dumps(dataset, ensure_ascii=False, default=str)),
        "verification": verification,
        "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
        "frontend_build": build,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2343_failed", checks))
    append_memo(result)
    del passport
    gc.collect()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
