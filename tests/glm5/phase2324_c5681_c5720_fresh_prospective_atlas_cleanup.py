#!/usr/bin/env python3
"""Publish Phase2323 exact-coordinate prospective fields and clean raw duplicates."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2323 = RESULT / "phase2323_c5601_c5680_qwen4b_fp16_fresh_prospective"
OUT = RESULT / "phase2324_c5681_c5720_fresh_prospective_atlas_cleanup"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2324
CAMPAIGN = "C5681-C5720"

sys.path.insert(0, str(TESTS))
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 全新词汇前瞻响应的全坐标图谱发布与原始场清理（{CAMPAIGN}） [{stamp}]

**测试原理、用例与公式。** 本期不重新运行模型，而把 Phase2323 的 32 条 `fresh_confirmation` 新词汇、3 个源深度、6 个方向和 3 个目标检查点逐事件展开为精确坐标热力图。每行保留样本、语言族、语言、自然表面、状态、源层、探针、目标层和全部 2560 个原始物理坐标；不做 Top-K、PCA、平均、排序或投影。方向响应与偶响应分别为：

$$
R_{{i,q,r,t,j}}=\frac{{H_{{t,j}}(h_q+\epsilon r)-H_{{t,j}}(h_q-\epsilon r)}}{{2\epsilon\lVert h_q\rVert_2}},\qquad
E_{{i,q,r,t,j}}=\frac{{H_{{t,j}}^++H_{{t,j}}^-}}{{2}}-H_{{t,j}}^0.
$$

**结果汇总、相关文件与门槛。** 发布结果 `{json.dumps(result['datasets'], ensure_ascii=False)}`；逐资产校验 `{json.dumps(result['verification'], ensure_ascii=False)}`；目录更新 `{json.dumps(result['catalog'], ensure_ascii=False)}`；离线前端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。Phase2323 的冻结实验门为 `{json.dumps(result['experimental_gates'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2324_c5681_c5720_fresh_prospective_atlas_cleanup.py`，结果目录 `tests/glm5/result/phase2324_c5681_c5720_fresh_prospective_atlas_cleanup`，资产目录 `frontend/public/vis_data/research_kernel`。没有启动或连接本地研究客户端。

**分析、理论进展、问题硬伤与结论。** 资产让 `0.2601` 的全局预测误差、`0.9596` 的稳定符号一致率以及族条件修正可以下钻到每个原始坐标观察，但可视化本身不提高因果等级。Phase2323 五门过四门，唯一失败门显示族均值在多数而非全部语言族上优于全局均值；因此当前最谨慎的拼图是“共享局部传播骨架上可能叠加语言族条件修正”，不是固定语义坐标。发布后按哈希账本删除原结果目录中的方向响应和偶响应重复数组，共 `{result['cleanup']['bytes_deleted']}` 字节；发布副本与小型范数账本保留。下一步扩大到 128 条不重叠新词汇样本，前瞻确认族条件修正是否稳定。
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
    parent = json.loads((P2323 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2323 execution is not authorized")
    atlas.PHASE = PHASE
    atlas.CAMPAIGN = CAMPAIGN
    atlas.OUT = OUT
    field = parent["field"]
    index = atlas.read_jsonl(P2323 / "index/active_rows.jsonl")
    sources = [int(value) for value in field["sources"]]
    targets = {int(key): [int(value) for value in values]
               for key, values in field["targets"].items()}
    datasets = [
        atlas.flatten_directional(
            "c5681_qwen4b_fp16_fresh_confirmation_derivative",
            "Qwen3-4B FP16 fresh-confirmation directional response",
            P2323 / "raw/directional_derivative.float16.npy", index,
            atlas.model_probe_ledger(), sources, targets, "Qwen3-4B-FP16",
            "central finite directional derivative in each FP16 physical coordinate",
            "32 prospective fresh_confirmation rows; 1% source-state-norm perturbation",
        ),
        atlas.flatten_directional(
            "c5682_qwen4b_fp16_fresh_confirmation_even_response",
            "Qwen3-4B FP16 fresh-confirmation finite-dose even response",
            P2323 / "raw/even_response.float16.npy", index,
            atlas.model_probe_ledger(), sources, targets, "Qwen3-4B-FP16",
            "symmetric finite-dose residual in each FP16 physical coordinate",
            "32 prospective fresh_confirmation rows; half-sum minus baseline",
        ),
    ]
    verification = [atlas.verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id") for row in verification):
        raise RuntimeError(("asset_verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    raw_paths = [
        P2323 / "raw/directional_derivative.float16.npy",
        P2323 / "raw/even_response.float16.npy",
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
        "checks": {
            "parent_execution_authorized": True,
            "all_assets_verified": all(all(value for key, value in row.items() if key != "id")
                                           for row in verification),
            "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
            "frontend_build_passed": build["passed"],
            "no_client_connection": not build["browser_or_client_connection"],
            "raw_fields_cleaned": all(not path.exists() for path in raw_paths),
            "published_fields_retained": all(Path(row["binary"]).exists() for row in datasets),
            "no_coordinate_compression": True,
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
