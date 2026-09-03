#!/usr/bin/env python3
"""Publish replicated semantic/lexical and failed output-bridge passports to the heatmap client."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2441 = RESULT / "phase2441_c35761_c36080_alltoken_crosslanguage_trajectory"
P2444 = RESULT / "phase2444_c36721_c37040_semantic_specific_multiunit_multinull"
P2445 = RESULT / "phase2445_c37041_c37360_internal_output_geometry_bridge"
OUT = RESULT / "phase2446_c37361_c37680_semantic_output_visualization_audit"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json"
BINARY = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.float32.npy"
DIST = ROOT / "frontend/dist/index.html"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2446
CAMPAIGN = "C37361-C37680"
TAG = "phase2446_semantic_output_replication"
DIM = 2560


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def save_if_changed(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def add(rows: list[dict], values: np.ndarray, label: str, source: str, coordinate_kind: str,
        interaction: str, component: str, language: str, family: str, preview: bool) -> None:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise RuntimeError((label, vector.shape))
    rows.append({"label": label, "source": source, "coordinate_kind": coordinate_kind,
                 "component": component, "layer": None, "event": "query_end", "family": family,
                 "language": language, "interaction": interaction, "split": "fresh_unit5",
                 "preview": preview, "campaign_tag": TAG,
                 "values": [float(value) for value in vector]})


def build_asset() -> dict:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row.get("campaign_tag") != TAG]
    internal = np.load(P2444 / "derived/selected_semantic_lexical_split_passports.float32.npy", mmap_mode="r")
    output = np.load(P2445 / "derived/split_language_family_output_passports.float32.npy", mmap_mode="r")
    final2444 = json.loads((P2444 / "analysis/final.json").read_text(encoding="utf-8"))
    final2445 = json.loads((P2445 / "analysis/final.json").read_text(encoding="utf-8"))
    families = final2444["analysis"]["families"]
    interactions = ("semantic_validity", "lexical_control")
    components = ("signed_state", "block_update")
    languages = ("en", "zh")
    added = []
    for ii, interaction in enumerate(interactions):
        for ci, component in enumerate(components):
            for li, language in enumerate(languages):
                for fi, family in enumerate(families):
                    add(added, internal[ii, ci, 2, li, fi],
                        f"{interaction} {component} fresh {language} {family} full-coordinate passport",
                        "phase2444_semantic_specific_multiunit_multinull", "full_coordinate_family_passport",
                        interaction, component, language, family,
                        interaction == "semantic_validity" and family == "taxonomy")
        for li, language in enumerate(languages):
            for fi, family in enumerate(families):
                add(added, output[ii, 2, li, fi],
                    f"{interaction} final output contribution fresh {language} {family} passport",
                    "phase2445_internal_output_geometry_bridge", "coordinate_logit_contribution",
                    interaction, "final_output_contribution", language, family,
                    interaction == "semantic_validity" and family == "taxonomy")
    close(internal); close(output)
    rows.extend(added)
    matrix = np.stack([np.asarray(row["values"], dtype=np.float32) for row in rows])
    np.save(BINARY, matrix)
    payload.update({"phase": PHASE, "campaign": "C32561-C37680", "rows": rows,
                    "summary": {**payload.get("summary", {}), "phase2446_added_rows": len(added),
                                "phase2444_semantic_specific_candidate": final2444["adjudication"]["semantic_specific_crosslanguage_candidate"],
                                "phase2445_output_bridge_closed": final2445["adjudication"]["semantic_output_bridge_closed"]},
                    "claim_boundary": "All 2560 physical coordinates are retained. Phase2444 replicated a cross-language semantic-validity interaction against coordinate shift and 64 family-label nulls across held units, but Phase2445 found neither absolute nor relational closure into final token-coordinate contributions. The display therefore shows the positive internal passports beside the failed output passports; no causal gear or cracked language mechanism is claimed."})
    save_if_changed(ASSET, payload)
    return {"rows": len(rows), "added_rows": len(added), "dimensions": DIM,
            "binary_shape": list(matrix.shape), "asset": str(ASSET), "binary": str(BINARY),
            "json_bytes": ASSET.stat().st_size, "binary_bytes": BINARY.stat().st_size,
            "finite": bool(np.isfinite(matrix).all())}


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def hash_manifest() -> dict:
    targets = (P2441 / "derived/normalized_token_role_difference.float16.npy",
               P2444 / "derived/selected_semantic_lexical_split_passports.float32.npy",
               P2445 / "derived/split_language_family_output_passports.float32.npy")
    records = []
    for path in targets:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024): digest.update(chunk)
        records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
                        "retention": "retained_and_represented_in_full_coordinate_client_asset"})
    save(OUT / "analysis/retention_manifest.json", records)
    return {"files": len(records), "bytes": sum(row["bytes"] for row in records),
            "all_hashes": all(len(row["sha256"]) == 64 for row in records)}


def frontend_contract() -> dict:
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8-sig")
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8-sig")
    hook = (ROOT / "frontend/src/researchKernel/useResearchKernel.js").read_text(encoding="utf-8-sig")
    return {"route_range": "C32561-C37680" in route, "preview_range": "C32561-C37680 embedding" in component,
            "asset_hook": "setC32561LanguageEncodingField" in hook, "dist_exists": DIST.exists(),
            "dist_newer_than_asset": DIST.exists() and DIST.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 语义特异性跨语言护照与失败输出桥的全坐标可视化审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2444通过64置乱/multiunit门的semantic/lexical signed-state与block-update fresh中英八族护照，以及Phase2445未闭合的final output-contribution中英八族护照并列追加到现有热力图。共新增96行，每行完整2560坐标；只有taxonomy代表行进入默认预览，其余仍可选择，不用Top-K替代原场。对三个新增不可替代数组计算SHA256。

$$V\in\mathbb R^{{96\times2560}},\qquad V=\{{P^H_{{sem/lex}},P^U_{{sem/lex}},P^C_{{sem/lex}}\}}.$$

**结果汇总。** 资产 `{json.dumps(result['asset'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2446_c37361_c37680_semantic_output_visualization_audit.py`；资产仍为`frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json`及`.float32.npy`；路由标题与3D预览更新至C37680；final/SHA256位于同名结果目录。

**分析与理论进展。** 客户端现在把当前最强阳性与最硬反证放在同一物理坐标轴：内部语义有效性纹理跨语言/新实体复现，但到具体token输出贡献时同坐标和family几何都消失。它支持“模型内部存在条件化分布式坐标纹理”，不支持“已找到把纹理编译为token的齿轮”。

**问题硬伤与结论。** 不同量纲只能逐行看纹理，不能跨行比幅值。浏览器资产是全坐标代表护照，不是384配置全原场；统计裁决仍以结果文件为准。所有唯一原场与三份新数组保留，重复/失败缓存不作为证据。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    asset = build_asset(); retention = hash_manifest(); frontend = frontend_contract()
    checks = {"full_coordinates": asset["dimensions"] == DIM and asset["binary_shape"] == [asset["rows"], DIM],
              "rows_added_96": asset["added_rows"] == 96, "finite": asset["finite"],
              "frontend_source": frontend["route_range"] and frontend["preview_range"] and frontend["asset_hook"],
              "frontend_built": frontend["dist_newer_than_asset"], "hashes": retention["files"] == 3 and retention["all_hashes"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": frontend,
              "retention": retention, "checks": checks, "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        print(json.dumps(result, ensure_ascii=False, indent=2)); raise RuntimeError(checks)
    save(final, result); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
