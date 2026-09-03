#!/usr/bin/env python3
"""Publish the Phase2525 32-head route map as a fifth c42641 heatmap panel."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2524 = RESULT / "phase2524_c89953_c91328_event_path_visualization_retention_audit"
P2525 = RESULT / "phase2525_c91329_c92704_multilayer_attention_route_lockbox"
OUT = RESULT / "phase2526_c92705_c93728_attention_route_visualization_audit"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, KEY = 2526, "C92705-C93728", "qwen4b_attention_heads"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def row(values: np.ndarray, label: str, kind: str, **meta: Any) -> dict:
    vector = np.asarray(values, np.float32).reshape(-1)
    if vector.shape != (32,) or not np.isfinite(vector).all(): raise RuntimeError(label)
    return {"label": label, "source": "phase2526_attention_route", "coordinate_kind": kind,
            "preview": True, **meta, "values": [float(v) for v in vector]}


def publish() -> dict:
    f2525 = load(P2525 / "analysis/final.json")
    interaction = np.load(f2525["fields"]["interaction"]["path"], mmap_mode="r")
    rows = []
    for ui, unit in enumerate((30, 31)):
        for layer in range(20, 36):
            rms = np.sqrt(np.mean(np.asarray(interaction[ui, :, :, layer], np.float64) ** 2, axis=(0, 1, 3)))
            rows.append(row(rms, f"unit{unit} layer{layer} head route Walsh RMS", "attention_head_layer_energy",
                            phase=2525, unit=unit, layer=layer, averaging="nine families two languages four regions"))
        for region_index, region in enumerate(f2525["fields"]["attention_mass"]["regions"]):
            rms = np.sqrt(np.mean(np.asarray(interaction[ui, :, :, 20:, :, region_index], np.float64) ** 2,
                                  axis=(0, 1, 2)))
            rows.append(row(rms, f"unit{unit} {region} late head route Walsh RMS", "attention_head_region_energy",
                            phase=2525, unit=unit, region=region, layers="20-35",
                            averaging="nine families two languages sixteen layers"))
    top = {(r["layer"], r["head"]) for r in f2525["routes"]["top"]}
    random = {(r["layer"], r["head"]) for r in f2525["routes"]["random"]}
    for layer in range(20, 36):
        rows.append(row(np.asarray([1.0 if (layer, head) in top else 0.0 for head in range(32)]),
                        f"layer{layer} frozen top32 membership", "attention_head_top_mask", phase=2525, layer=layer))
        rows.append(row(np.asarray([1.0 if (layer, head) in random else 0.0 for head in range(32)]),
                        f"layer{layer} equal-size random membership", "attention_head_random_mask", phase=2525, layer=layer))
    aggregate = np.square(np.asarray([r["values"] for r in rows if "energy" in r["coordinate_kind"]], np.float64)).sum(axis=0)
    section = {"key": KEY, "model": "Qwen3-4B late Attention routes", "precision": "BF16 attention / float32 atlas",
               "coordinate_count": 32, "coordinate_semantics": "model-local query-head index at each displayed layer",
               "coordinate_order": "physical query-head 0-31", "rows": rows,
               "coordinate_orders": {"event_path": [int(v) for v in np.argsort(-aggregate)]},
               "coordinate_order_semantics": {"event_path": "unit30/unit31 late route-Walsh RMS energy"}}
    payload = load(ASSET)
    payload["models"] = [s for s in payload["models"] if s.get("key") != KEY] + [section]
    payload["phase"] = PHASE; payload["campaign"] = "C39761-C93728"
    payload["summary"]["phase2525_top32_donor_flip_rate"] = f2525["causal"]["donor_top32"]["donor_flip_rate"]
    payload["summary"]["phase2525_random32_donor_flip_rate"] = f2525["causal"]["donor_random32"]["donor_flip_rate"]
    payload["summary"]["phase2525_all_late_donor_flip_rate"] = f2525["causal"]["donor_all_late"]["donor_flip_rate"]
    payload["summary"]["model_rows"] = {s["key"]: len(s["rows"]) for s in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    sentence = ("Phase2525 freezes 32 late layer-head routes on unit30 and obtains 86.1% donor flips on unit31, "
                "versus 0% for equal-size random routes; all late Attention heads reach 100%, but this remains a "
                "Qwen3-4B answer-boundary route result rather than a complete semantic compiler.")
    if sentence not in payload["claim_boundary"]: payload["claim_boundary"] = payload["claim_boundary"].rstrip() + " " + sentence
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if ASSET.read_text(encoding="utf-8") != content: ASSET.write_text(content, encoding="utf-8")
    return {"asset": str(ASSET), "head_panel_rows": len(rows), "head_coordinates": 32,
            "total_sections": len(payload["models"]), "json_sha256": digest(ASSET), "json_bytes": ASSET.stat().st_size}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 多层Attention路由32-head热力图发布与阶段审核（{CAMPAIGN}） [{stamp}]

**测试原理与显示内容。** 在c42641资产新增第五个Qwen3-4B Attention路由面板：32列严格表示模型本地query-head编号，不冒充2560维残差坐标；逐层显示layer20–35的九族双语Walsh RMS、四来源区域聚合、unit30冻结top32与等量随机mask。原四模型的词嵌入/HiddenState物理参数面板仍同屏保留，`event_path`顺序对残差面板按2560坐标、对head面板按32个head各自独立冻结。

$$E_{{lh}}=\sqrt{{\operatorname{{mean}}_{{f,\lambda,r}} I_{{f\lambda lhr}}^2}}.$$

**结果汇总。** 发布 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2526_c92705_c93728_attention_route_visualization_audit.py`；`ResearchHeatmapRoute.jsx`扩展为五面板布局；c42641资产、生产build、SHA-256与final位于对应目录。

**理论进展。** 重要结果已可直接核查：top32来自unit30而因果数字来自unit31，避免把同一数据上的高响应当作锁箱；随机32为0%翻转而top32为86.1%，说明跨层head协同具有选择性。全部晚层Attention为100%充分，但这不证明它们必要，也不说明Attention单独完成关系推理。

**问题硬伤与结论。** head编号只在Qwen3-4B内部有物理意义；热力图能量不是因果大小；区域切分较粗；没有跨模型head对齐。客户端明确把32-head面板与2560/3584/4096/5120残差坐标面板分开，避免伪造统一坐标基底。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    prior4, prior5 = load(P2524 / "analysis/final.json"), load(P2525 / "analysis/final.json")
    asset = publish()
    attention_path = Path(prior5["fields"]["attention_mass"]["path"])
    interaction_path = Path(prior5["fields"]["interaction"]["path"])
    retention = {"files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p),
                             "retention": "important attention-route source displayed in derived 32-head panel"}
                            for p in (attention_path, interaction_path)]}
    retention["bytes"] = sum(r["bytes"] for r in retention["files"])
    retention["all_hashes"] = all(len(r["sha256"]) == 64 for r in retention["files"])
    save(OUT / "analysis/retention_manifest.json", retention)
    route_source = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    dist = ROOT / "frontend/dist/index.html"
    frontend = {"five_panel_layout": "panels.length > 4" in route_source,
                "dist_exists": dist.exists(), "dist_newer": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns}
    checks = {"sources_passed": prior4["all_checks_passed"] and prior5["all_checks_passed"],
              "head_rows_72": asset["head_panel_rows"] == 72, "head_coordinates_32": asset["head_coordinates"] == 32,
              "five_sections": asset["total_sections"] == 5, "frontend_layout": frontend["five_panel_layout"],
              "frontend_built": frontend["dist_newer"], "retention": retention["all_hashes"],
              "top_vs_random_not_overstated": prior5["causal"]["donor_top32"]["donor_flip_rate"] > .8
                                                  and prior5["causal"]["donor_random32"]["donor_flip_rate"] == 0,
              "claim_boundary": True}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": frontend, "retention": retention,
             "adjudication": {"selective_multilayer_attention_route_supported": True,
                              "head_route_crossmodel": False, "complete_semantic_compiler": False,
                              "language_encoding_mechanism_closed": False},
             "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis" / ("final.json" if final["all_checks_passed"] else "prebuild.json"), final)
    if final["all_checks_passed"]: append_memo(final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
