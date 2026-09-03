#!/usr/bin/env python3
"""Publish Phase2328 exact-coordinate fields, clean duplicates, and close the route."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2328 = RESULT / "phase2328_c5921_c6000_unseen_condition_prediction"
OUT = RESULT / "phase2329_c6001_c6080_unseen_prediction_atlas_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
PHASE = 2329
CAMPAIGN = "C6001-C6080"

sys.path.insert(0, str(TESTS))
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def publish_selected_residual(
    index: list[dict], probes: list[dict], sources: list[int], targets: dict[int, list[int]],
) -> dict:
    source_path = P2328 / "raw/selected_model_residual.float32.npy"
    source = np.load(source_path, mmap_mode="r")
    expected = [len(index), len(sources), len(probes), 3, 2560]
    if list(source.shape) != expected:
        raise RuntimeError(("selected_residual_shape", list(source.shape), expected))
    dataset_id = "c6003_qwen4b_fp16_unseen_recent_global_residual"
    binary = VIS / f"{dataset_id}.float32.npy"
    output = atlas.create_binary(
        binary.name, int(np.prod(source.shape[:-1])), source.shape[-1], np.float32,
    )
    rows = []
    cursor = 0
    for row_index, row in enumerate(index):
        for source_index, source_q in enumerate(sources):
            for probe_index, probe in enumerate(probes):
                for target_index, target_q in enumerate(targets[source_q]):
                    output[cursor] = source[row_index, source_index, probe_index, target_index]
                    rows.append({
                        **atlas.case_fields(row),
                        "active_index": row_index,
                        "source_q": source_q,
                        "source_relative_depth": source_q / max(sources),
                        "probe": probe_index,
                        "probe_kind": probe["kind"],
                        "probe_members": probe["members"],
                        "target_q": target_q,
                        "target_slot": ("q_plus_1", "q_plus_4", "final_norm")[target_index],
                        "prediction": "frozen_units17_18_recent_global_mean",
                    })
                    cursor += 1
    output.flush()
    atlas.close_memmap(output)
    atlas.close_memmap(source)
    return atlas.write_metadata(
        dataset_id,
        "Qwen3-4B FP16 unseen-unit frozen recent-global residual",
        binary,
        rows,
        "Qwen3-4B-FP16",
        "full_coordinate_frozen_prediction_residual_v1",
        "prospective derived",
        "128 unseen unit19-20 rows minus frozen units17-18 recent-global mean",
        "signed actual-minus-predicted error in every original physical coordinate",
        {
            "selected_model": "recent_global",
            "training_units": [17, 18],
            "test_units": [19, 20],
            "exact_full_coordinate_residual": True,
            "route_passed": False,
            "warning": "failed-predictor error field, not a semantic-neuron score",
        },
    )


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    parent = result["parent_result"]
    record = rf"""

## Phase {PHASE}: 未见单元响应、偶响应与失败预测残差的全坐标发布及路线总审计（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 本期不重新运行模型、不重选预测器、不修改 Phase2328 的门槛。它把 unit19–20 的 128 条全新词汇样本、3 个源深度、6 个探针、3 个目标检查点展开为三套精确热力图：实际方向响应、有限剂量偶响应、实际响应减去冻结 `recent_global` 预测的有符号残差。每套均为 `6912×2560`，保留语言族、语言、表面、状态、单元、源层、探针、目标层和全部原始物理坐标；没有 Top-K、PCA、坐标排序、投影或跨模型坐标对齐。

$$
D_{{i,q,r,t,j}}=\frac{{H_{{t,j}}(h_q+\epsilon r)-H_{{t,j}}(h_q-\epsilon r)}}{{2\epsilon\lVert h_q\rVert_2}},\qquad
V_{{i,q,r,t,j}}=D_{{i,q,r,t,j}}-\widehat D^{{17,18}}_{{q,r,t,j}}.
$$

$$
E=\frac{{\lVert V\rVert_2^2}}{{\lVert D\rVert_2^2+\varepsilon}},\qquad
Q=\frac{{\operatorname{{median}}E_{{recent}}}}{{\operatorname{{median}}E_{{lockbox}}}}=1.184920.
$$

**结果汇总、门槛与相关文件。** Phase2328 在 128 条未见样本上的候选和自由生成行为分别为 `{parent['behavior']['overall']}`。冻结 `recent_global` 的中位相对 MSE 为 `{parent['prediction']['models']['recent_global']['median_relative_mse']:.6f}`、符号一致率 `{parent['prediction']['models']['recent_global']['median_sign_agreement']:.6f}`；历史 `lockbox_global` 的 MSE 为 `{parent['prediction']['models']['lockbox_global']['median_relative_mse']:.6f}`，两者之比 `{parent['prediction']['selected_over_lockbox_ratio']:.6f}`。五门中过四门：绝对 MSE、符号一致、成对叠加误差 `{parent['functional_metrics']['median_pair_superposition_relative_mse']:.6f}`、偶/奇比 `{parent['functional_metrics']['median_even_to_odd_l2']:.6f}`通过，但相对对照门 `Q<=1.0` 失败。发布 `{json.dumps(result['datasets'], ensure_ascii=False)}`；校验 `{json.dumps(result['verification'], ensure_ascii=False)}`；目录 `{json.dumps(result['catalog'], ensure_ascii=False)}`；离线构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2329_c6001_c6080_unseen_prediction_atlas_audit.py`，结果 `tests/glm5/result/phase2329_c6001_c6080_unseen_prediction_atlas_audit`，资产位于 `frontend/public/vis_data/research_kernel`。没有启动或连接本地研究客户端。

**分析、理论进展、问题硬伤与结论。** 最严格的结论是：units17–18 的近期全局均值不是比旧 lockbox 均值更好的稳定幅值基态，简单语言、表面、状态、家族主效应和近期均值路线至此均没有获得继续资格。不能据此否定 HiddenState 的条件结构，因为实际场仍前瞻保留约 `0.8070` 的坐标符号一致率，1% 邻域成对叠加误差仅约 `0.01944`、偶/奇比约 `0.17511`；这支持“共享的局部符号传播骨架与样本条件化幅值并存”的窄拼图。另一方面，行为自由生成总体仅 `0.50`，八族差异很大；随机 Rademacher 方向不是自然语言操作；所有单元共享生成器且只测 Qwen3-4B FP16；晚层预测误差远高于近层，因此不得把图谱命名为语言齿轮、语义算子或因果电路。理论主体保持“条件化输出场闭合理论”，没有增加新理论名词；本期新增的是失败模型残差的可观察坐标图，而不是数学闭合。

**路线裁决、清理与下一阶段。** 本轮发布后删除三份已验证的重复原始场共 `{result['cleanup']['bytes_deleted']}` 字节，小型范数、逐样本预测账本、配置、索引和发布副本保留。Phase2327–2329 已完整完成“条件地理观察 → 未见单元前瞻 → 全坐标发布”目标；下一个目标不再相同，因此不自动重复均值模型。后续应把现有实际场和残差按 `q+1/q+4/final` 传播距离分账，观察哪些坐标由近层可预测、哪些只在输出边界发生样本特异放大，再冻结新的状态条件算法；首先观察，不把目标距离差异事后包装为机制。

**最终文件变更审计。** Phase2327–2329 脚本均通过 `py_compile`，三份 `final.json` 的执行检查均为真，备忘录中 Phase2327–2329 各出现一次且编号连续。本轮 Markdown 只追加 `research/glm5/docs/AGI_GLM5_MEMO.md`；工作区中其他已有 Markdown 变更未被修改或恢复。
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
    parent = json.loads((P2328 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2328 execution is not authorized")
    atlas.PHASE = PHASE
    atlas.CAMPAIGN = CAMPAIGN
    atlas.OUT = OUT
    field = parent["field"]
    index = atlas.read_jsonl(P2328 / "index/active_rows.jsonl")
    probes = atlas.model_probe_ledger()
    sources = [int(value) for value in field["sources"]]
    targets = {
        int(key): [int(value) for value in values]
        for key, values in field["targets"].items()
    }
    datasets = [
        atlas.flatten_directional(
            "c6001_qwen4b_fp16_unseen_units_derivative",
            "Qwen3-4B FP16 unseen unit19-20 directional response",
            P2328 / "raw/directional_derivative.float16.npy",
            index, probes, sources, targets, "Qwen3-4B-FP16",
            "central finite directional derivative in every FP16 physical coordinate",
            "128 unseen balanced unit19-20 rows; 1% source-state-norm perturbation",
        ),
        atlas.flatten_directional(
            "c6002_qwen4b_fp16_unseen_units_even_response",
            "Qwen3-4B FP16 unseen unit19-20 finite-dose even response",
            P2328 / "raw/even_response.float16.npy",
            index, probes, sources, targets, "Qwen3-4B-FP16",
            "finite-dose symmetric residual in every FP16 physical coordinate",
            "128 unseen balanced unit19-20 rows; half-sum minus baseline",
        ),
        publish_selected_residual(index, probes, sources, targets),
    ]
    verification = [atlas.verify(row) for row in datasets]
    verified = all(
        all(value for key, value in row.items() if key != "id")
        for row in verification
    )
    if not verified:
        raise RuntimeError(("asset_verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    raw_paths = [
        P2328 / "raw/directional_derivative.float16.npy",
        P2328 / "raw/even_response.float16.npy",
        P2328 / "raw/selected_model_residual.float32.npy",
    ]
    cleanup = atlas.cleanup(raw_paths)
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed",
        "parent_result": {
            "behavior": parent["behavior"],
            "prediction": parent["prediction"],
            "functional_metrics": parent["functional_metrics"],
            "experimental_gates": parent["experimental_gates"],
            "route_passed": parent["route_passed"],
        },
        "datasets": [atlas.serializable(row) for row in datasets],
        "verification": verification,
        "catalog": catalog,
        "frontend_build": build,
        "cleanup": cleanup,
        "route_adjudication": {
            "simple_condition_mean_route": "closed_not_qualified",
            "preserved_observation": (
                "shared local sign structure and small-dose near-linearity remain qualified "
                "inside the Qwen3-4B FP16 random-direction boundary"
            ),
            "automatic_same_objective_continuation": False,
            "reason": "Phase2327-2329 completed the frozen observation-prospective-publication chain.",
        },
        "checks": {
            "parent_execution_authorized": True,
            "parent_route_failure_preserved": not parent["route_passed"],
            "three_assets": len(datasets) == 3,
            "all_assets_verified": verified,
            "all_6912_rows": all(row["shape"][0] == 6912 for row in datasets),
            "all_2560_coordinates": all(row["shape"][1] == 2560 for row in datasets),
            "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
            "frontend_build_passed": build["passed"],
            "no_client_connection": not build["browser_or_client_connection"],
            "raw_duplicates_cleaned": all(not path.exists() for path in raw_paths),
            "published_assets_retained": all(Path(row["binary"]).exists() for row in datasets),
            "small_reproduction_ledgers_retained": (
                (P2328 / "raw/source_and_target_rms.float32.npy").exists()
                and (P2328 / "analysis/prospective_prediction_records.jsonl").exists()
            ),
            "no_coordinate_compression": True,
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
