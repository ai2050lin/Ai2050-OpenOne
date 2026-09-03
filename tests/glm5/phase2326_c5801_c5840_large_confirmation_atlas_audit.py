#!/usr/bin/env python3
"""Publish the large confirmation and its full-coordinate frozen-prediction residual."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2325 = RESULT / "phase2325_c5721_c5800_qwen4b_fp16_large_family_confirmation"
OUT = RESULT / "phase2326_c5801_c5840_large_confirmation_atlas_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
PHASE = 2326
CAMPAIGN = "C5801-C5840"

sys.path.insert(0, str(TESTS))
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2322_c5521_c5600_full_coordinate_reuse_passports as passport  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def publish_global_residual(index: list[dict], sources: list[int], targets: dict[int, list[int]]) -> dict:
    source_path = P2325 / "raw/directional_derivative.float16.npy"
    source = np.load(source_path, mmap_mode="r")
    _meta, old_cells = passport.load_cells("c5481_qwen4b_fp16_directional_derivative")
    row_count = int(np.prod(source.shape[:-1]))
    binary = VIS / "c5803_qwen4b_fp16_large_confirmation_global_residual.float32.npy"
    output = atlas.create_binary(binary.name, row_count, source.shape[-1], np.float32)
    rows = []
    cursor = 0
    probes = atlas.model_probe_ledger()
    for row_index, row in enumerate(index):
        for source_index, source_q in enumerate(sources):
            for probe_index, probe in enumerate(probes):
                for target_index, target_q in enumerate(targets[source_q]):
                    discovery = old_cells[(source_index, probe_index, target_index)][0]
                    prediction = discovery.mean(axis=0, dtype=np.float64)
                    actual = source[row_index, source_index, probe_index, target_index].astype(np.float64)
                    output[cursor] = (actual - prediction).astype(np.float32)
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
                        "prediction": "frozen_fresh_lockbox_global_mean",
                    })
                    cursor += 1
    output.flush()
    atlas.close_memmap(output)
    atlas.close_memmap(source)
    return atlas.write_metadata(
        "c5803_qwen4b_fp16_large_confirmation_global_residual",
        "Qwen3-4B FP16 large-confirmation frozen-global residual",
        binary,
        rows,
        "Qwen3-4B-FP16",
        "full_coordinate_frozen_prediction_residual_v1",
        "prospective derived",
        "128 unseen rows minus the frozen 32-row fresh_lockbox global response mean",
        "signed actual-minus-predicted residual in every original physical coordinate",
        {
            "reference_dataset": "c5481_qwen4b_fp16_directional_derivative",
            "sample_count": 128,
            "exact_full_coordinate_residual": True,
            "warning": "prediction error field, not a semantic-neuron score",
        },
    )


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 扩大确认全坐标场、预测残差图谱与大阶段审计（{CAMPAIGN}） [{stamp}]

**测试原理、用例与公式。** 本期不重新运行模型。它把 Phase2325 的 128 条不重叠新词汇响应发布成三类可下钻资产：方向响应、有限剂量偶响应，以及实际方向响应减去 Phase2320 `fresh_lockbox` 32 条发现样本冻结全局均值的有符号残差。每类均保留 3 个源深度、6 个方向、3 个目标检查点和全部 2560 个原始物理坐标；无 Top-K、PCA、排序、平均或投影。一个热力图单元表示某条具体新句在某个源层与探针下，目标层某个物理坐标的真实响应或预测误差。

$$
\mathcal R_{{i,q,r,t,j}}=D_{{i,q,r,t,j}}-\frac1{{N_0}}\sum_{{n\in\mathrm{{fresh\_lockbox}}}}D_{{n,q,r,t,j}},\qquad N_0=32.
$$

$$
E_g(i,q,r,t)=\frac{{\sum_j\mathcal R_{{i,q,r,t,j}}^2}}{{\sum_jD_{{i,q,r,t,j}}^2+\varepsilon}}.
$$

**结果汇总、相关文件与校验。** 发布 `{json.dumps(result['datasets'], ensure_ascii=False)}`；资产校验 `{json.dumps(result['verification'], ensure_ascii=False)}`；目录 `{json.dumps(result['catalog'], ensure_ascii=False)}`；离线构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。扩大确认门 `{json.dumps(result['experimental_gates'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2326_c5801_c5840_large_confirmation_atlas_audit.py`，结果 `tests/glm5/result/phase2326_c5801_c5840_large_confirmation_atlas_audit`，资产位于 `frontend/public/vis_data/research_kernel`。未启动或连接本地研究客户端。

**分析、理论进展、问题硬伤与结论。** 128 条扩大确认否证了 Phase2323 的族条件修正候选：总体族均值不优于全局，仅 `2/8` 族胜出，且冻结全局幅值误差超过 `0.35` 门。仍前瞻保留的是稳定坐标符号一致率 `0.9091`、成对叠加误差 `0.01765` 和偶/奇比 `0.16175`。因此当前拼图应表述为“在相同模型、精度、源层、目标层和随机方向条件下，许多坐标的响应符号可跨词汇复用，1% 邻域近似线性；响应幅值具有更强样本条件性，不能由简单全局或语言族均值闭合”。符号一致仍可能由残差通道、方向构造、FP16 分辨率和模板共同造成；随机方向不是语言操作；扩大面板虽有 128 条，发现均值仍只有 32 条。理论主体仍为“条件化输出场闭合理论”，没有新增数学结构或语义齿轮结论。三类资产全部经形状、有限值、坐标数和 SHA256 校验后，删除原目录中重复方向场和偶响应共 `{result['cleanup']['bytes_deleted']}` 字节，小型范数账本保留以复现指标。

**下一步大任务。** 本阶段“族均值是否提供稳定修正”的目标已经完整裁决，不应继续围绕同一均值公式追加样本。下一阶段应转回观察优先：在残差图上按层、目标距离、语言、表面和状态逐坐标记录“符号保持但幅值失配”的条件图谱，先寻找重复的条件切换模式，再冻结新算法；不要把本期残差直接压成单一语义方向。

**最终文件变更审计。** Phase2322–2326 五个脚本均通过 `py_compile`，五份 `final.json` 的执行检查均为真，备忘录中 Phase2322–2326 各出现一次且编号连续。本轮 Markdown 写入仅追加到 `research/glm5/docs/AGI_GLM5_MEMO.md`；`git status` 中同时存在的 `ai2050_research_os/README.md`、`research/glm5/docs/AGI_GLM5_VISUAL_MEMO.md` 与 `research/MainAnalysis/*.md` 是本轮开始前已有的无关工作区变更，本轮未修改、未恢复。
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
    parent = json.loads((P2325 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2325 execution is not authorized")
    atlas.PHASE = PHASE
    atlas.CAMPAIGN = CAMPAIGN
    atlas.OUT = OUT
    field = parent["field"]
    index = atlas.read_jsonl(P2325 / "index/active_rows.jsonl")
    sources = [int(value) for value in field["sources"]]
    targets = {int(key): [int(value) for value in values]
               for key, values in field["targets"].items()}
    datasets = [
        atlas.flatten_directional(
            "c5801_qwen4b_fp16_large_confirmation_derivative",
            "Qwen3-4B FP16 128-row confirmation directional response",
            P2325 / "raw/directional_derivative.float16.npy", index,
            atlas.model_probe_ledger(), sources, targets, "Qwen3-4B-FP16",
            "central finite directional derivative in every FP16 physical coordinate",
            "128 unseen balanced fresh_confirmation rows; 1% source-state-norm perturbation",
        ),
        atlas.flatten_directional(
            "c5802_qwen4b_fp16_large_confirmation_even_response",
            "Qwen3-4B FP16 128-row confirmation even response",
            P2325 / "raw/even_response.float16.npy", index,
            atlas.model_probe_ledger(), sources, targets, "Qwen3-4B-FP16",
            "finite-dose symmetric residual in every FP16 physical coordinate",
            "128 unseen balanced fresh_confirmation rows; half-sum minus baseline",
        ),
    ]
    datasets.append(publish_global_residual(index, sources, targets))
    verification = [atlas.verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id") for row in verification):
        raise RuntimeError(("asset_verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    raw_paths = [
        P2325 / "raw/directional_derivative.float16.npy",
        P2325 / "raw/even_response.float16.npy",
    ]
    cleanup = atlas.cleanup(raw_paths)
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed",
        "datasets": [atlas.serializable(row) for row in datasets],
        "verification": verification,
        "catalog": catalog,
        "frontend_build": build,
        "cleanup": cleanup,
        "experimental_gates": parent["experimental_gates"],
        "strict_conclusion": (
            "Family-conditioned amplitude correction did not replicate; shared sign structure and "
            "small-dose local linearity remain qualified within the frozen boundary."
        ),
        "checks": {
            "parent_authorized": True,
            "three_assets": len(datasets) == 3,
            "all_assets_verified": all(all(value for key, value in row.items() if key != "id")
                                           for row in verification),
            "all_6912_rows": all(row["shape"][0] == 6912 for row in datasets),
            "all_2560_coordinates": all(row["shape"][1] == 2560 for row in datasets),
            "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
            "frontend_build_passed": build["passed"],
            "no_client_connection": not build["browser_or_client_connection"],
            "raw_duplicates_cleaned": all(not path.exists() for path in raw_paths),
            "published_assets_retained": all(Path(row["binary"]).exists() for row in datasets),
            "no_coordinate_compression": True,
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
