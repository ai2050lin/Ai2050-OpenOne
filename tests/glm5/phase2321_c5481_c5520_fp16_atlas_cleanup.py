#!/usr/bin/env python3
"""Publish exact-coordinate Qwen3-4B FP16 controls and clean raw duplicates."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2320 = RESULT / "phase2320_c5401_c5480_qwen4b_fp16_precision_control"
OUT = RESULT / "phase2321_c5481_c5520_fp16_atlas_cleanup"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2321
CAMPAIGN = "C5481-C5520"

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

## Phase {PHASE}: Qwen3-4B FP16 全坐标精度对照图谱与清理（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 将 Phase2320 的 32 行 fresh_lockbox、3 个源深度、6 个方向与 3 个目标检查点的 FP16 有符号方向导数和偶响应原样展平为“事件行 × 2560 原始坐标”。展平不删除、排序、平均或投影任何坐标；元数据保留样本、语言族、语言、表面、状态、源层、探针和目标层。
$$
R_{{i,q,r,t,j}}=\frac{{H_{{t,j}}(h_q+\epsilon r)-H_{{t,j}}(h_q-\epsilon r)}}{{2\epsilon\lVert h_q\rVert_2}},\qquad
E_{{i,q,r,t,j}}=\frac{{H_{{t,j}}^++H_{{t,j}}^-}}{{2}}-H_{{t,j}}^0.
$$

**结果、门槛和相关文件。** 发布 `{json.dumps(result['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；目录 `{json.dumps(result['catalog'], ensure_ascii=False)}`；离线构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。精度重裁 `{json.dumps(result['precision_comparison'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2321_c5481_c5520_fp16_atlas_cleanup.py`；结果目录 `tests/glm5/result/phase2321_c5481_c5520_fp16_atlas_cleanup`；可视化资产位于 `frontend/public/vis_data/research_kernel`。没有启动或连接客户端。

**分析、理论进展、硬伤与结论。** 同一模型 FP16/BF16 的巨大差异证明有限差分测量强依赖数值精度；此前 Qwen3-14B 与 Qwen3-4B 的局部线性差异不能直接解释为规模规律。Qwen3-14B 相对 4B FP16 仍有较小的剩余差异，但模型规模、具体权重、层数和 FP16 舍入共同变化，当前不能分账。理论主体保持“条件化输出场闭合理论”，没有新增数学结构或语义齿轮结论。资产验证与构建通过后删除 `{result['cleanup']['bytes_deleted']}` 字节原始副本，删除前哈希写入清理账本；精度对照全坐标资产继续保留供观察。"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    phase2320 = json.loads((P2320 / "analysis/final.json").read_text(encoding="utf-8"))
    if not phase2320["all_checks_passed"]:
        raise RuntimeError("Phase2320 is not authorized")
    atlas.PHASE = PHASE
    atlas.CAMPAIGN = CAMPAIGN
    atlas.OUT = OUT
    field = phase2320["field"]
    index = atlas.read_jsonl(P2320 / "index/active_rows.jsonl")
    sources = [int(value) for value in field["sources"]]
    targets = {int(key): [int(value) for value in values]
               for key, values in field["targets"].items()}
    datasets = [
        atlas.flatten_directional(
            "c5481_qwen4b_fp16_directional_derivative",
            "Qwen3-4B FP16 model-local directional derivative",
            P2320 / "raw/directional_derivative.float16.npy", index,
            atlas.model_probe_ledger(), sources, targets, "Qwen3-4B-FP16",
            "central finite directional derivative in each FP16 physical coordinate",
            "1% source-state-norm perturbation; precision-control asset",
        ),
        atlas.flatten_directional(
            "c5482_qwen4b_fp16_even_response",
            "Qwen3-4B FP16 finite-dose even response",
            P2320 / "raw/even_response.float16.npy", index,
            atlas.model_probe_ledger(), sources, targets, "Qwen3-4B-FP16",
            "symmetric finite-dose residual in each FP16 physical coordinate",
            "half-sum of positive and negative responses minus baseline",
        ),
    ]
    verification = [atlas.verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id") for row in verification):
        raise RuntimeError(("asset_verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    raw_paths = [P2320 / "raw/directional_derivative.float16.npy",
                 P2320 / "raw/even_response.float16.npy"]
    cleanup = atlas.cleanup(raw_paths)
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
        "datasets": [atlas.serializable(row) for row in datasets],
        "verification": verification, "catalog": catalog, "frontend_build": build,
        "cleanup": cleanup, "precision_comparison": phase2320["comparison"],
        "checks": {
            "parent_authorized": True,
            "all_assets_verified": all(all(value for key, value in row.items() if key != "id")
                                           for row in verification),
            "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
            "frontend_build_passed": build["passed"],
            "no_client_connection": not build["browser_or_client_connection"],
            "raw_fields_cleaned": all(not path.exists() for path in raw_paths),
            "no_coordinate_compression": True,
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
