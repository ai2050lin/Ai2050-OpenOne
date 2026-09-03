#!/usr/bin/env python3
"""Audit whether positive cross-language cycles imply reciprocal coordinate maps."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2432 = RESULT / "phase2432_c32881_c33200_crosslanguage_coordinate_reparameterization"
OUT = RESULT / "phase2433_c33201_c33520_crosslanguage_reciprocity_audit"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json"
BINARY = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.float32.npy"
DIST = ROOT / "frontend/dist/index.html"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2433
CAMPAIGN = "C33201-C33520"
INTERACTIONS = ("semantic_validity", "lexical_control")
CHECKPOINTS = (12, 24, 36)
SHIFT = 791


def serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize(value), encoding="utf-8")


def save_if_changed(path: Path, value: Any) -> None:
    content = serialize(value)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def describe_product(forward: np.ndarray, reverse: np.ndarray) -> dict:
    product = np.asarray(forward * reverse, dtype=np.float64)
    mismatch = np.asarray(forward * np.roll(reverse, SHIFT), dtype=np.float64)
    error, mismatch_error = np.abs(product - 1.0), np.abs(mismatch - 1.0)
    return {
        "mean_abs_product_minus_one": float(error.mean()),
        "median_abs_product_minus_one": float(np.median(error)),
        "fraction_abs_error_below_0_25": float(np.mean(error < .25)),
        "fraction_abs_error_below_0_50": float(np.mean(error < .50)),
        "same_sign_fraction": float(np.mean(product > 0)),
        "mean_product": float(product.mean()),
        "mismatch_mean_abs_product_minus_one": float(mismatch_error.mean()),
        "same_coordinate_reciprocity_advantage": float(mismatch_error.mean() - error.mean()),
        "product_coordinate_cosine_to_mismatch": cosine(product, mismatch),
    }


def analyze() -> tuple[dict, np.ndarray]:
    slopes = np.load(P2432 / "derived/crosslanguage_diagonal_slope.float32.npy", mmap_mode="r")
    summary, products = {}, []
    adjacent = {}
    for ii, interaction in enumerate(INTERACTIONS):
        summary[interaction], adjacent[interaction] = {}, {}
        all_products = []
        for qpoint in range(37):
            product = np.asarray(slopes[ii, 0, qpoint] * slopes[ii, 1, qpoint], dtype=np.float32)
            all_products.append(product)
            if qpoint in CHECKPOINTS:
                summary[interaction][f"q{qpoint}"] = describe_product(slopes[ii, 0, qpoint], slopes[ii, 1, qpoint])
                products.extend((np.asarray(slopes[ii, 0, qpoint], dtype=np.float32),
                                 np.asarray(slopes[ii, 1, qpoint], dtype=np.float32), product, product - 1.0))
        adjacent[interaction] = {
            "roundtrip_product_adjacent_cosine_mean_q1_q36": float(np.mean([
                cosine(all_products[q], all_products[q + 1]) for q in range(1, 36)])),
            "roundtrip_product_adjacent_cosine_min_q1_q36": float(np.min([
                cosine(all_products[q], all_products[q + 1]) for q in range(1, 36)])),
        }
    matrix = np.stack(products).astype(np.float32)
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "derived/crosslanguage_reciprocity_checkpoints.float32.npy", matrix)
    return {"summary": summary, "adjacent_roundtrip_stability": adjacent,
            "files": {"checkpoint_vectors": str(OUT / "derived/crosslanguage_reciprocity_checkpoints.float32.npy")}}, matrix


def update_asset(matrix: np.ndarray) -> dict:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    payload["phase"] = [2431, 2433]
    payload["campaign"] = "C32561-C33520"
    prefix = "phase2433_reciprocity"
    payload["rows"] = [row for row in payload["rows"] if row.get("source") != prefix]
    cursor = 0
    new_rows = []
    for interaction in INTERACTIONS:
        for qpoint in CHECKPOINTS:
            for kind, label in (("en_to_zh_slope", "English→Chinese slope"),
                                ("zh_to_en_slope", "Chinese→English slope"),
                                ("roundtrip_product", "same-coordinate slope product"),
                                ("roundtrip_error", "slope product - 1")):
                new_rows.append({
                    "label": f"{interaction} q{qpoint} {label}",
                    "source": prefix,
                    "coordinate_kind": kind,
                    "component": "total",
                    "layer": qpoint,
                    "event": "query_end",
                    "family": "all",
                    "preview": interaction == "semantic_validity" and qpoint == 24 and kind in ("en_to_zh_slope", "zh_to_en_slope", "roundtrip_product", "roundtrip_error"),
                    "values": [float(value) for value in matrix[cursor]],
                })
                cursor += 1
    payload["rows"].extend(new_rows)
    payload["summary"]["phase2433_reciprocity_rows"] = len(new_rows)
    extension = " Phase2433 further shows that positive fitted cycles do not by themselves establish reciprocal/invertible coordinate maps."
    if extension.strip() not in payload["claim_boundary"]:
        payload["claim_boundary"] += extension
    save_if_changed(ASSET, payload)
    full_matrix = np.stack([np.asarray(row["values"], dtype=np.float32) for row in payload["rows"]])
    np.save(BINARY, full_matrix)
    return {"rows_added": len(new_rows), "total_rows": len(payload["rows"]),
            "dimensions": len(payload["dimensions"]), "binary_shape": list(full_matrix.shape),
            "json_bytes": ASSET.stat().st_size, "finite": bool(np.isfinite(full_matrix).all())}


def frontend_contract() -> dict:
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8-sig")
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8-sig")
    return {"expanded_campaign_title": "C32561-C33520" in route and "C32561-C33520" in component,
            "dist_exists": DIST.exists(),
            "dist_newer_than_asset": DIST.exists() and DIST.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 跨语言双向循环的逐坐标互逆性审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2432的双向循环gain为正，但循环可能来自家族基线与两次收缩，而非互逆坐标编码。本Phase读取两类交互、双方向、37状态点的完整2560维斜率，不再拟合新模型。对q12/q24/q36逐坐标计算$\beta^{{en\to zh}}_i\beta^{{zh\to en}}_i$，以1为互逆目标；同时将反向斜率错位+791形成零假设。另计算q1–q36相邻层乘积纹理余弦。关键向量追加到客户端全坐标热力图，而不是只汇报均值。

$$r_{{q,i}}=\beta^{{en\to zh}}_{{q,i}}\beta^{{zh\to en}}_{{q,i}},\qquad E_q=\frac1{{2560}}\sum_i|r_{{q,i}}-1|,$$

$$A^{{recip}}_q=E^{{shift+791}}_q-E^{{same}}_q.$$

**结果汇总。** 互逆性 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；跨层乘积稳定性 `{json.dumps(result['analysis']['adjacent_roundtrip_stability'], ensure_ascii=False)}`；客户端扩展 `{json.dumps(result['visualization'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2433_c33201_c33520_crosslanguage_reciprocity_audit.py`；q12/q24/q36完整逐坐标斜率、乘积与乘积误差位于`tests/glm5/result/phase2433_c33201_c33520_crosslanguage_reciprocity_audit`；同一客户端热力图资产扩展到C32561–C33520。除本MEMO外未增改其他Markdown。

**分析与理论进展。** “跨语言可预测”与“跨语言可逆编码”必须分开。对角映射若强烈收缩，正向和反向都能借助家族均值改善预测，循环gain仍可能为正；只有同坐标乘积接近1且显著优于错位，才支持可逆重参数化。这个基本检验直接约束Phase2432的解释，不引入新的高级数学名词。

**问题硬伤与结论。** 逐坐标互逆仍只检验对角结构，真实机制可能是坐标群映射；但Phase2427未发现稳定内生组，不能为阳性强行换基。斜率受岭分母$10^{{-8}}$与低方差坐标收缩影响，因此低方差位置接近0正是需要保留的反证，而不是删除。若乘积普遍远离1，则Phase2432只能保留“有方向的跨语言预测纹理”，不能称共享语义码重参数化。下一阶段需要新采集的跨语言同义token时序场；这是新材料战役，不再是对现有场的同一后处理阶段。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    analysis, matrix = analyze()
    visualization = update_asset(matrix)
    frontend = frontend_contract()
    semantic = analysis["summary"]["semantic_validity"]
    reciprocal = all(value["median_abs_product_minus_one"] < .5 and
                     value["same_coordinate_reciprocity_advantage"] > 0
                     for value in semantic.values())
    same_better = all(value["same_coordinate_reciprocity_advantage"] > 0 for interaction in analysis["summary"].values()
                      for value in interaction.values())
    semantic_better = all(analysis["summary"]["semantic_validity"][key]["mean_abs_product_minus_one"] <
                          analysis["summary"]["lexical_control"][key]["mean_abs_product_minus_one"]
                          for key in semantic)
    adjudication = {"same_coordinate_reciprocity_beats_shift_all_interactions_checkpoints": same_better,
                    "semantic_reciprocity_error_below_half_all_checkpoints": reciprocal,
                    "semantic_reciprocity_better_than_lexical_all_checkpoints": semantic_better,
                    "invertible_crosslanguage_coordinate_reparameterization_detected": reciprocal and semantic_better,
                    "phase2432_positive_cycle_implies_reciprocal_code": False,
                    "language_encoding_mechanism_closed": False}
    checks = {"two_interactions_three_checkpoints_four_vectors": matrix.shape == (24, 2560),
              "finite": bool(np.isfinite(matrix).all()),
              "visual_full_coordinates": visualization["dimensions"] == 2560 and visualization["binary_shape"] == [visualization["total_rows"], 2560],
              "frontend_campaign_title": frontend["expanded_campaign_title"],
              "frontend_build_verified": frontend["dist_newer_than_asset"],
              "claim_boundary": not adjudication["language_encoding_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis, "visualization": visualization,
              "frontend": frontend, "adjudication": adjudication, "checks": checks,
              "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError("run frontend build, then rerun Phase2433")


if __name__ == "__main__":
    main()
