#!/usr/bin/env python3
"""Publish the replicated output-conditioned semantic VJP passports and audit retained fields."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2447 = RESULT / "phase2447_c37681_c38000_output_conditioned_vjp_pilot"
P2448 = RESULT / "phase2448_c38001_c38480_vjp_semantic_multiunit_replication"
P2449 = RESULT / "phase2449_c38481_c38800_vjp_crosssurface_lockbox"
OUT = RESULT / "phase2450_c38801_c39120_vjp_visualization_retention_audit"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json"
BINARY = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.float32.npy"
DIST = ROOT / "frontend/dist/index.html"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2450
CAMPAIGN = "C38801-C39120"
TAG = "phase2450_output_conditioned_vjp"
DIM = 2560


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def save_if_changed(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def add(rows: list[dict], vector: np.ndarray, *, label: str, source: str, interaction: str,
        component: str, surface: str, unit: int, language: str, family: str, qpoint: int,
        preview: bool) -> None:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.shape != (DIM,) or not np.isfinite(values).all():
        raise RuntimeError((label, values.shape))
    rows.append({"label": label, "source": source, "coordinate_kind": "output_conditioned_full_coordinate",
                 "component": component, "layer": qpoint, "event": "query_end", "family": family,
                 "language": language, "interaction": interaction, "split": f"{surface}_unit{unit}",
                 "surface": surface, "unit": unit, "preview": preview, "campaign_tag": TAG,
                 "values": [float(value) for value in values]})


def build_asset() -> dict:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row.get("campaign_tag") != TAG]
    final48 = json.loads((P2448 / "analysis/final.json").read_text(encoding="utf-8"))
    final49 = json.loads((P2449 / "analysis/final.json").read_text(encoding="utf-8"))
    natural = np.load(final48["analysis"]["passports"], mmap_mode="r")
    canonical = np.load(final49["analysis"]["canonical_passports"], mmap_mode="r")
    families = final48["analysis"]["families"]
    added: list[dict] = []
    for interaction_index, interaction in enumerate(("semantic_validity", "lexical_control")):
        qpoint = int(final48["analysis"]["selection"][interaction]["state_times_gradient"])
        for unit_offset, unit in enumerate((4, 5)):
            for language_index, language in enumerate(("en", "zh")):
                for family_index, family in enumerate(families):
                    for surface, values in (("natural", natural[interaction_index, 1, unit_offset + 1, language_index, qpoint, family_index]),
                                            ("canonical", canonical[interaction_index, 1, unit_offset, language_index, qpoint, family_index])):
                        add(added, values,
                            label=f"{interaction} H×VJP {surface} unit{unit} {language} {family} q{qpoint}",
                            source=f"phase2448_2449_{surface}", interaction=interaction,
                            component="state_times_output_margin_vjp", surface=surface, unit=unit,
                            language=language, family=family, qpoint=qpoint,
                            preview=interaction == "semantic_validity" and unit == 5 and language == "en" and family == "taxonomy")
    # Add a small but full-coordinate direct-gradient comparison, while the larger raw field stays in retained arrays.
    semantic_gradient_qpoint = int(final48["analysis"]["selection"]["semantic_validity"]["gradient"])
    taxonomy = families.index("taxonomy")
    for unit_offset, unit in enumerate((4, 5)):
        for language_index, language in enumerate(("en", "zh")):
            for surface, values in (("natural", natural[0, 0, unit_offset + 1, language_index, semantic_gradient_qpoint, taxonomy]),
                                    ("canonical", canonical[0, 0, unit_offset, language_index, semantic_gradient_qpoint, taxonomy])):
                add(added, values,
                    label=f"semantic direct VJP {surface} unit{unit} {language} taxonomy q{semantic_gradient_qpoint}",
                    source=f"phase2448_2449_{surface}", interaction="semantic_validity",
                    component="output_margin_vjp_gradient", surface=surface, unit=unit,
                    language=language, family="taxonomy", qpoint=semantic_gradient_qpoint, preview=False)
    close(natural); close(canonical)
    rows.extend(added)
    matrix = np.stack([np.asarray(row["values"], dtype=np.float32) for row in rows])
    np.save(BINARY, matrix)
    payload.update({"phase": PHASE, "campaign": "C32561-C39120", "rows": rows,
                    "summary": {**payload.get("summary", {}), "phase2450_added_rows": len(added),
                                "phase2448_output_conditioned_semantic_candidate": final48["adjudication"]["output_conditioned_semantic_attribution_candidate"],
                                "phase2449_crosssurface_lockbox": final49["adjudication"]["crosssurface_output_conditioned_semantic_candidate"]},
                    "claim_boundary": "All displayed rows retain all 2560 physical coordinates. Phase2448-2449 support a Qwen3-4B output-conditioned semantic-attribution texture across languages, held units, and two surfaces at frozen qpoints. VJP is a local first-order readout, so no finite-intervention gear or complete language encoding mechanism is claimed."})
    save_if_changed(ASSET, payload)
    return {"rows": len(rows), "added_rows": len(added), "dimensions": DIM, "binary_shape": list(matrix.shape),
            "asset": str(ASSET), "binary": str(BINARY), "json_bytes": ASSET.stat().st_size,
            "binary_bytes": BINARY.stat().st_size, "finite": bool(np.isfinite(matrix).all())}


def hash_manifest() -> dict:
    targets = (
        P2447 / "raw/output_margin_vjp.float32.npy",
        P2447 / "raw/output_margin_state_times_vjp.float32.npy",
        P2448 / "raw/query_margin_vjp.float32.npy",
        P2448 / "raw/query_margin_state_times_vjp.float32.npy",
        P2448 / "derived/semantic_lexical_vjp_passports.float32.npy",
        P2449 / "raw/query_margin_vjp.float32.npy",
        P2449 / "raw/query_margin_state_times_vjp.float32.npy",
        P2449 / "derived/canonical_semantic_lexical_vjp_passports.float32.npy",
    )
    records = []
    for path in targets:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
                        "retention": "retained; important full-coordinate field or passport represented in client"})
    save(OUT / "analysis/retention_manifest.json", records)
    return {"files": len(records), "bytes": sum(row["bytes"] for row in records), "all_hashes": all(len(row["sha256"]) == 64 for row in records)}


def frontend_contract() -> dict:
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8-sig")
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8-sig")
    hook = (ROOT / "frontend/src/researchKernel/useResearchKernel.js").read_text(encoding="utf-8-sig")
    return {"route_range": "C32561-C38800" in route,
            "preview_range": "C32561-C38800 embedding" in component,
            "asset_hook": "setC32561LanguageEncodingField" in hook,
            "dist_exists": DIST.exists(),
            "dist_newer_than_asset": DIST.exists() and DIST.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 输出条件语义归因的全坐标可视化、留存与阶段审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2448/2449在冻结层上通过锁箱的semantic/lexical、unit4/5、natural/canonical、中英、八族$H\odot VJP$护照全部追加到现有参数热力图；另加入taxonomy的direct-gradient代表行。每行保留2560物理坐标，不以Top-K代替。对Phase2447–2449八份不可替代的原始梯度、归因和护照数组逐文件计算SHA256，并核对生产构建晚于资产。

$$V\in\mathbb R^{{136\times2560}},\qquad h_k=\operatorname{{SHA256}}(F_k),\quad k=1,\ldots,8.$$

**结果汇总。** 资产 `{json.dumps(result['asset'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；客户端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2450_c38801_c39120_vjp_visualization_retention_audit.py`；资产`frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json`及`.float32.npy`；标题更新到C38800；final和哈希清单位于同名结果目录。

**分析与理论进展。** 可视化现在能在同一固定坐标轴比较embedding/HiddenState、语义与词项interaction、直接输出梯度、$H\odot VJP$、跨表述和最终token贡献。这把“内部纹理存在”“真实输出在局部怎样读取”“固定跨样本输出桥为何失败”同时公开，避免只展示阳性摘要。

**问题硬伤与结论。** 新增136行是全坐标family护照，不是960条逐样本原场；逐样本原场保留在结果目录并由哈希约束。量纲不同的行不能比绝对幅值。当前最强结论仍限于Qwen3-4B的一阶输出条件语义归因候选，尚未完成跨模型、有限扰动或全生成链闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    prepare = "--prepare" in sys.argv
    asset = build_asset()
    if prepare:
        print(json.dumps({"prepared": True, "asset": asset}, ensure_ascii=False, indent=2))
        return
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    retention = hash_manifest()
    frontend = frontend_contract()
    checks = {"full_coordinates": asset["dimensions"] == DIM and asset["binary_shape"] == [asset["rows"], DIM],
              "rows_added_136": asset["added_rows"] == 136, "finite": asset["finite"],
              "frontend_source": frontend["route_range"] and frontend["preview_range"] and frontend["asset_hook"],
              "frontend_built": frontend["dist_newer_than_asset"],
              "hashes": retention["files"] == 8 and retention["all_hashes"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "retention": retention,
              "frontend": frontend, "checks": checks, "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        print(json.dumps(result, ensure_ascii=False, indent=2)); raise RuntimeError(checks)
    save(final, result); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
