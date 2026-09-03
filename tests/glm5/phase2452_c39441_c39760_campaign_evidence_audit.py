#!/usr/bin/env python3
"""Independent continuity, artifact, claim-boundary, and successor-plan audit for Phase2434-2451."""
from __future__ import annotations

import json
import py_compile
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2452_c39441_c39760_campaign_evidence_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json"
BINARY = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.float32.npy"
DIST = ROOT / "frontend/dist/index.html"
PHASE = 2452
CAMPAIGN = "C39441-C39760"

PHASE_DIRS = {
    2434: "phase2434_c33521_c33840_trajectory_campaign_audit",
    2435: "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior",
    2436: "phase2436_c34161_c34480_qwen4b_hypergraph_fullfield",
    2437: "phase2437_c34481_c34800_signed_trajectory_atlas",
    2438: "phase2438_c34801_c35120_coordinate_event_group_tournament",
    2439: "phase2439_c35121_c35440_output_autonomous_bridge",
    2440: "phase2440_c35441_c35760_trajectory_visualization_retention",
    2441: "phase2441_c35761_c36080_alltoken_crosslanguage_trajectory",
    2442: "phase2442_c36081_c36400_embedding_context_update_decomposition",
    2443: "phase2443_c36401_c36720_multiunit_contextual_role_replication",
    2444: "phase2444_c36721_c37040_semantic_specific_multiunit_multinull",
    2445: "phase2445_c37041_c37360_internal_output_geometry_bridge",
    2446: "phase2446_c37361_c37680_semantic_output_visualization_audit",
    2447: "phase2447_c37681_c38000_output_conditioned_vjp_pilot",
    2448: "phase2448_c38001_c38480_vjp_semantic_multiunit_replication",
    2449: "phase2449_c38481_c38800_vjp_crosssurface_lockbox",
    2450: "phase2450_c38801_c39120_vjp_visualization_retention_audit",
    2451: "phase2451_c39121_c39440_vjp_finite_intervention",
}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def update_asset() -> dict:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    p2451 = json.loads((RESULT / PHASE_DIRS[2451] / "analysis/final.json").read_text(encoding="utf-8"))
    payload["phase"] = PHASE
    payload["campaign"] = "C32561-C39440"
    payload.setdefault("summary", {})["phase2451_finite_intervention"] = {
        "unit4_actual_semantic": p2451["analysis"]["summary"]["unit4"]["actual_semantic"],
        "unit5_actual_semantic": p2451["analysis"]["summary"]["unit5"]["actual_semantic"],
        "vjp_actual_correlation": p2451["analysis"]["linearity"]["correlation"],
        "vjp_actual_sign_agreement": p2451["analysis"]["linearity"]["sign_agreement"],
        "strict_local_vjp_gate": p2451["adjudication"]["local_vjp_prediction_supported"],
    }
    payload["claim_boundary"] = "The heatmap retains all 2560 coordinates for representative embedding, HiddenState, interaction, output, gradient, and H×VJP passports. Phase2451 adds scalar evidence that frozen discovery directions cause held-unit margin changes under a 1% RMS intervention, but its strict per-sample local-linear sign gate failed. Evidence supports a Qwen3-4B finite-effect semantic-direction candidate, not a unique causal gear or a closed language encoding mechanism."
    content = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    if ASSET.read_text(encoding="utf-8") != content:
        ASSET.write_text(content, encoding="utf-8")
    matrix = np.load(BINARY, mmap_mode="r")
    result = {"rows": len(payload["rows"]), "binary_shape": list(matrix.shape), "dimensions": int(matrix.shape[1]),
              "finite": bool(np.isfinite(matrix).all()), "campaign": payload["campaign"],
              "phase2451_summary_present": "phase2451_finite_intervention" in payload["summary"]}
    mmap = getattr(matrix, "_mmap", None)
    if mmap is not None:
        mmap.close()
    return result


def phase_audit() -> dict:
    memo = MEMO.read_text(encoding="utf-8")
    records = []
    previous_end = None
    campaign_contiguous = True
    scripts_compile = True
    for phase, directory in PHASE_DIRS.items():
        final_path = RESULT / directory / "analysis/final.json"
        final = json.loads(final_path.read_text(encoding="utf-8"))
        campaign = final["campaign"]
        start, end = (int(value) for value in campaign.replace("C", "").split("-"))
        if previous_end is not None and start != previous_end + 1:
            campaign_contiguous = False
        previous_end = end
        script_path = ROOT / "tests/glm5" / f"{directory}.py"
        try:
            py_compile.compile(str(script_path), doraise=True)
            compiled = True
        except py_compile.PyCompileError:
            compiled = False
            scripts_compile = False
        records.append({"phase": phase, "campaign": campaign, "final_exists": final_path.exists(),
                        "all_checks_passed": bool(final.get("all_checks_passed")),
                        "memo_heading_count": memo.count(f"## Phase {phase}:"), "script_compiles": compiled})
    return {"records": records, "phase_count": len(records), "campaign_contiguous": campaign_contiguous,
            "all_finals_pass": all(record["all_checks_passed"] for record in records),
            "memo_exactly_once": all(record["memo_heading_count"] == 1 for record in records),
            "scripts_compile": scripts_compile}


def evidence_ladder() -> dict:
    p2435 = json.loads((RESULT / PHASE_DIRS[2435] / "analysis/final.json").read_text(encoding="utf-8"))
    p2444 = json.loads((RESULT / PHASE_DIRS[2444] / "analysis/final.json").read_text(encoding="utf-8"))
    p2445 = json.loads((RESULT / PHASE_DIRS[2445] / "analysis/final.json").read_text(encoding="utf-8"))
    p2448 = json.loads((RESULT / PHASE_DIRS[2448] / "analysis/final.json").read_text(encoding="utf-8"))
    p2449 = json.loads((RESULT / PHASE_DIRS[2449] / "analysis/final.json").read_text(encoding="utf-8"))
    p2451 = json.loads((RESULT / PHASE_DIRS[2451] / "analysis/final.json").read_text(encoding="utf-8"))
    return {
        "retained": [
            "Eight broad language families have unique, behavior-qualified material on Qwen3-4B and Qwen3-14B; GLM4 qualifies seven; DS7B panel is not aggregate-qualified.",
            "Full-coordinate semantic-validity interaction textures replicate across held entities and languages above coordinate-shift and 64 family-permutation nulls.",
            "Output-conditioned gradient and H-times-gradient textures replicate across held units and canonical/natural surfaces at frozen layers.",
            "A frozen discovery semantic-gradient direction changes held-unit token margins more than one coordinate shift and one family mismatch at a 1% RMS finite dose.",
        ],
        "corrected_overclaims": [
            "A strong internal texture is not yet a conditional gear.",
            "The fixed internal-to-final contribution bridge in Phase2445 failed; VJP supplies sample-conditioned local sensitivity, not a universal compiler.",
            "Phase2451 VJP-versus-finite correlation is high, but the strict sign-agreement gate fails; therefore local linear closure is not established.",
            "Cross-language and cross-surface reuse in one 4B model cannot establish architecture-universal encoding.",
        ],
        "key_flags": {
            "four_model_behavior_complete": bool(p2435["all_checks_passed"]),
            "internal_semantic_specific_candidate": bool(p2444["adjudication"]["semantic_specific_crosslanguage_candidate"]),
            "fixed_output_bridge_closed": bool(p2445["adjudication"]["semantic_output_bridge_closed"]),
            "output_conditioned_candidate": bool(p2448["adjudication"]["output_conditioned_semantic_attribution_candidate"]),
            "crosssurface_candidate": bool(p2449["adjudication"]["crosssurface_output_conditioned_semantic_candidate"]),
            "finite_direction_candidate": bool(p2451["adjudication"]["finite_intervention_semantic_candidate"]),
            "strict_local_linear_gate": bool(p2451["adjudication"]["local_vjp_prediction_supported"]),
            "language_encoding_mechanism_closed": False,
        },
    }


def successor_plan() -> dict:
    return {
        "stage_change": "The Qwen3-4B observation-to-local-finite-effect stage is complete. The next stage changes the primary target to portability, dose law, and autoregressive compilation.",
        "phases": [
            {"priority": 1, "task": "Freeze a model-relative event/layer contract and replicate output-conditioned semantic interactions sequentially on Qwen3-14B BF16 with device_map=auto, then GLM4 and DS7B; never compare raw amplitudes across precisions or architectures."},
            {"priority": 2, "task": "Run 0.25%, 0.5%, 1%, and 2% RMS symmetric doses with many frozen family derangements; estimate monotonicity, curvature, sign stability, and the BF16 quantization floor."},
            {"priority": 3, "task": "Replace first-token margin with a teacher-forced multi-token log-probability path and then free generation; map how the direction is reused, split, or overwritten across generated tokens."},
            {"priority": 4, "task": "Expand to new entities, new task frames, and additional language families; freeze discovery coordinates before every lockbox and retain full-coordinate fields for any result that changes the evidence level."},
        ],
        "stop_rule": "Do not demand one deletion/rescue gate. Accumulate portable coordinate/event/dose/token laws; claim mechanism closure only after the same frozen rule predicts and controls unseen families, models, and generation steps.",
    }


def frontend_contract() -> dict:
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8-sig")
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8-sig")
    return {"route_range": "C32561-C39440" in route,
            "preview_range": "C32561-C39440 embedding" in component,
            "dist_exists": DIST.exists(), "dist_newer_than_asset": DIST.exists() and DIST.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Phase2434–2451完整证据链、过度结论修正与后继阶段裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 独立遍历Phase2434–2451的18份final、脚本和MEMO标题，检查C33521–C39440区间连续、每Phase恰有一个主标题、所有脚本可编译、所有结果质量门通过；核对316×2560热力图二进制、Phase2451摘要与生产构建时间。证据审查按“行为资格→全场观察→语义特异性→输出条件读出→有限扰动”逐级降调，不用后级阳性覆盖前级反证。

$$E_0\subset E_1\subset E_2\subset E_3\subset E_4,\qquad
E_4\not\Rightarrow M_{{language}}\ \text{{closed}}.$$

**结果汇总。** 连续性 `{json.dumps(result['phase_audit'], ensure_ascii=False)}`；可视化 `{json.dumps(result['asset'], ensure_ascii=False)}`；客户端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；证据阶梯 `{json.dumps(result['evidence'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2452_c39441_c39760_campaign_evidence_audit.py`；final位于同名结果目录；Phase2434–2451原始与派生结果仍在各自目录；MEMO只在尾部追加本Phase。

**分析、理论进展与结论修正。** 保留：模型在固定物理基底上存在跨语言、跨实体、跨表述复用的分布式条件纹理；样本条件VJP能把部分纹理连接到真实下一token margin；冻结方向在held unit产生family/坐标特异的有限效应。修正：这不等于发现了唯一齿轮、通用编译器或新数学闭合；Phase2445的固定输出桥失败，Phase2451严格逐样本局部线性门也失败。当前最准确名称是“Qwen3-4B输出条件语义归因与有限方向候选”。

**问题硬伤。** 仍缺跨模型全场复制、多剂量曲率、更多family置乱的有限扰动、全生成链和新任务框架；八族仍共享候选问答协议；VJP依赖具体target/foil。任何“已破解语言编码”“线性代数已穷尽”或“因果齿轮已定位”的说法均超出证据。

**下一大阶段。** `{json.dumps(result['successor_plan'], ensure_ascii=False)}`
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    asset = update_asset()
    if "--prepare" in sys.argv:
        print(json.dumps({"prepared": True, "asset": asset}, ensure_ascii=False, indent=2)); return
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    audit = phase_audit()
    evidence = evidence_ladder()
    successor = successor_plan()
    frontend = frontend_contract()
    checks = {"eighteen_phases": audit["phase_count"] == 18, "campaign_contiguous": audit["campaign_contiguous"],
              "all_finals_pass": audit["all_finals_pass"], "memo_exactly_once": audit["memo_exactly_once"],
              "scripts_compile": audit["scripts_compile"], "asset_full_coordinate": asset["binary_shape"] == [316, 2560] and asset["finite"],
              "finite_summary_published": asset["phase2451_summary_present"],
              "frontend_built": frontend["route_range"] and frontend["preview_range"] and frontend["dist_newer_than_asset"],
              "invalid_duplicate_cache_removed": not (RESULT / "phase2435_failed_uniqueness_20260901_1647").exists(),
              "claim_boundary": not evidence["key_flags"]["language_encoding_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "phase_audit": audit, "asset": asset, "frontend": frontend,
              "evidence": evidence, "successor_plan": successor, "checks": checks, "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        print(json.dumps(result, ensure_ascii=False, indent=2)); raise RuntimeError(checks)
    save(final, result); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
