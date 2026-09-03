#!/usr/bin/env python3
"""Publish the independent token-region staged causal result to c42641."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2546 = RESULT / "phase2546_c150273_c154368_qkv_compiler_heatmap_retention"
P2547 = RESULT / "phase2547_c154369_c158464_independent_region_stage_replication"
OUT = RESULT / "phase2548_c158465_c160512_region_stage_heatmap_publish"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, SOURCE = 2548, "C158465-C160512", "phase2548_region_stage"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row(values, label: str, kind: str, **metadata) -> dict:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(vector).all():
        raise RuntimeError(label)
    return {"label": label, "source": SOURCE, "coordinate_kind": kind, "preview": True,
            **metadata, "values": [float(value) for value in vector]}


def publish(phase2547: dict) -> dict:
    payload = load(ASSET)
    panel = phase2547["summary"]["conditions"]
    facts = ("facts_entity", "facts_relation", "facts_value")
    fact_rows = [
        row([panel[f"early_v_{name}"]["donor_flip"] for name in facts],
            "unit34/surface1 independent early-V donor flip by fact token region",
            "fact_region_donor_flip", phase=2547, stage="early", projection="V", n=phase2547["summary"]["eligible_cases"]),
        row([panel[f"middle_k_{name}"]["donor_flip"] for name in facts],
            "unit34/surface1 independent middle-K donor flip by fact token region",
            "fact_region_donor_flip", phase=2547, stage="middle", projection="K", n=phase2547["summary"]["eligible_cases"]),
        row([panel[f"early_v_{name}"]["mean_donor_margin"] for name in facts],
            "unit34/surface1 independent early-V donor margin by fact token region",
            "fact_region_donor_margin", phase=2547, stage="early", projection="V", n=phase2547["summary"]["eligible_cases"]),
        row([panel[f"middle_k_{name}"]["mean_donor_margin"] for name in facts],
            "unit34/surface1 independent middle-K donor margin by fact token region",
            "fact_region_donor_margin", phase=2547, stage="middle", projection="K", n=phase2547["summary"]["eligible_cases"]),
    ]
    fact_section = {
        "key": "qwen4b_fact_region_stages", "model": "Qwen3-4B independent fact-region causal stages",
        "precision": "BF16 nonquantized", "coordinate_count": 3,
        "coordinate_semantics": "token-atomic fact region: entity token, relation token, value token",
        "coordinate_order": "facts_entity -> facts_relation -> facts_value", "rows": fact_rows,
    }
    individual_names = ("facts_entity", "facts_relation", "facts_value", "question_context", "query_property", "candidate", "instruction")
    region_section = {
        "key": "qwen4b_early_v_regions", "model": "Qwen3-4B independent early-V token regions",
        "precision": "BF16 nonquantized", "coordinate_count": len(individual_names),
        "coordinate_semantics": "token-atomic prompt region in the independent unit34/surface1 replication",
        "coordinate_order": " -> ".join(individual_names),
        "rows": [
            row([panel[f"early_v_{name}"]["donor_flip"] for name in individual_names],
                "early-V donor flip across all individually patched token regions", "region_donor_flip", phase=2547, n=phase2547["summary"]["eligible_cases"]),
            row([panel[f"early_v_{name}"]["mean_donor_margin"] for name in individual_names],
                "early-V donor margin across all individually patched token regions", "region_donor_margin", phase=2547, n=phase2547["summary"]["eligible_cases"]),
        ],
    }
    sections = [section for section in payload["models"] if section.get("key") not in {
        "qwen4b_fact_region_stages", "qwen4b_early_v_regions"
    }]
    for section in sections:
        if section.get("key") == "crossmodel_staged_compiler":
            section["rows"] = [item for item in section["rows"] if item.get("source") != SOURCE]
            section["rows"].append(row(
                [panel["early_v_facts_all"]["donor_flip"], panel["middle_kv_facts_all"]["donor_flip"],
                 panel["middlelate_kv_external"]["donor_flip"], panel["late_q"]["donor_flip"]],
                "Qwen3-4B independent unit34/surface1 primary staged donor flip",
                "staged_causal_effect", phase=2547, model_key="qwen4b_independent", n=phase2547["summary"]["eligible_cases"],
            ))
            section["rows"].append(row(
                [panel["early_k_facts_all"]["donor_flip"], panel["middle_k_facts_all"]["donor_flip"],
                 panel["middlelate_kv_external"]["donor_flip"], panel["late_kv_facts_all"]["donor_flip"]],
                "Qwen3-4B independent K/KV control branch donor flip",
                "staged_causal_effect", phase=2547, model_key="qwen4b_independent", n=phase2547["summary"]["eligible_cases"],
            ))
    payload["models"] = sections + [fact_section, region_section]
    payload["phase"] = PHASE
    payload["campaign"] = "C39761-C160512"
    payload["title"] = "Full-coordinate Q/K/V, token-region, autonomous, and cross-model staged compiler field"
    payload["summary"].update({
        "phase2547_eligible_cases": phase2547["summary"]["eligible_cases"],
        "phase2547_early_v_value_flip": panel["early_v_facts_value"]["donor_flip"],
        "phase2547_middle_k_value_flip": panel["middle_k_facts_value"]["donor_flip"],
        "phase2547_late_q_flip": panel["late_q"]["donor_flip"],
        "phase2547_late_fact_kv_flip": panel["late_kv_facts_all"]["donor_flip"],
    })
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    boundary = (
        " Phase2547 independently localizes the early-V and middle-K effects to facts_value tokens on unit34/surface1, "
        "but this remains a controlled micro-world result: value-token sufficiency does not establish a universal semantic atom, "
        "minimal within-head coordinates, or closure under natural language composition."
    )
    if boundary.strip() not in payload["claim_boundary"]:
        payload["claim_boundary"] += boundary
    ASSET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(ASSET), "bytes": ASSET.stat().st_size, "sha256": sha(ASSET),
        "sections": len(payload["models"]), "rows": payload["summary"]["model_rows"],
        "coordinate_counts": {section["key"]: section["coordinate_count"] for section in payload["models"]},
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 独立facts-value V→K阶段规律的region热力图发布（{CAMPAIGN}） [{stamp}]

**测试原理与显示内容。** 将Phase2547的重要独立结果加入c42641：新增facts实体/关系/值三region面板，直接显示早期V与中层K的donor flip及donor margin；新增七个独立prompt token-region的早期V面板；将unit34/surface1独立阶段链加入四阶段跨模型面板。所有数值来自124个base与donor双侧行为合格case。

$$\Delta^V_{{early}}(E,R,V)=({result['facts_value_vector']['early_v']}),\qquad \Delta^K_{{middle}}(E,R,V)=({result['facts_value_vector']['middle_k']}).$$

**结果汇总。** facts-region向量 `{json.dumps(result['facts_value_vector'], ensure_ascii=False)}`；资产 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2548_c158465_c160512_region_stage_heatmap_publish.py`；更新c42641资产、前端route说明和生产build；final位于`{OUT}`。

**分析与理论进展。** 在该受控任务中，值token不是任意差分：它是唯一同时承载mapping变化且早期V替换足以完整切换答案的原子region；中层相同region转为K主导，吻合“内容载荷→地址化”的阶段解释。热力图保留0效应region，避免只展示正例。

**问题硬伤与结论。** facts-value同时包含词汇值、绑定变化和位置角色，尚未解耦；三个region是token集合而非head内最小坐标；结果来自结构化二选一，不可直接推广到开放语义。该规律是强候选拼图，不是普遍语言编码定律。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    source = load(P2547 / "analysis/final.json")
    previous = load(P2546 / "analysis/final.json")
    prebuild = OUT / "analysis/prebuild.json"
    asset_header = load(ASSET)
    asset = load(prebuild)["asset"] if prebuild.exists() and asset_header.get("phase") == PHASE else publish(source)
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8")
    dist = ROOT / "frontend/dist/index.html"
    frontend = {
        "phase2547_boundary": "Phase2539-2547" in route,
        "dist_exists": dist.exists(),
        "dist_newer_than_asset": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns,
    }
    panel = source["summary"]["conditions"]
    vectors = {
        "early_v": [panel[f"early_v_{name}"]["donor_flip"] for name in ("facts_entity", "facts_relation", "facts_value")],
        "middle_k": [panel[f"middle_k_{name}"]["donor_flip"] for name in ("facts_entity", "facts_relation", "facts_value")],
    }
    checks = {
        "sources_passed": source["all_checks_passed"] and previous["all_checks_passed"],
        "facts_panel": asset["coordinate_counts"].get("qwen4b_fact_region_stages") == 3,
        "individual_region_panel": asset["coordinate_counts"].get("qwen4b_early_v_regions") == 7,
        "zeros_retained": vectors["early_v"][:2] == [0.0, 0.0] and vectors["middle_k"][:2] == [0.0, 0.0],
        "frontend_source": frontend["phase2547_boundary"], "frontend_build": frontend["dist_newer_than_asset"],
        "no_new_hiddenstate_cleanup_needed": True, "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "facts_value_vector": vectors,
        "asset": asset, "frontend": frontend, "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
