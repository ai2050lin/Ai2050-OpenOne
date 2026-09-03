#!/usr/bin/env python3
"""Independent source and asset audit for the C123-C126 heatmap integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1660_c126_factor_response_decomposition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
CARD = ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def main() -> None:
    payload = core.load(PUBLIC)
    card = CARD.read_text(encoding="utf-8")
    route = ROUTE.read_text(encoding="utf-8")
    rows = payload["c126_factor_response_batch"]["effect_rows"]
    checks = {
        "phase": payload["phase"] == 1660 and "C123-C126" in payload["campaign"],
        "asset": len(rows) == 315 and all(len(row["values"]) == 2560 for row in rows),
        "embedding": any(row["kind"] == "embedding" for row in rows),
        "hidden_checkpoints": all(any(row["kind"] == kind for row in rows) for kind in ("pre_last_block", "post_last_block_pre_norm", "post_final_norm")),
        "builder": all(token in card for token in ("transitionCoordinateRows", "transitionProfiles", "c125Rows", "c126Rows")),
        "render": all(token in card for token in ("C123-C124 状态与增量物理激活坐标", "C125 最终块与最终归一化响应分解", "C126 真值与答案码交互响应", "activation coordinate ${cell.dimension}")),
        "route": "C109-C126 Relation-Role-State and Typed Transition Atlas" in route and "不是参数权重" in route,
        "boundary": "not weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"],
    }
    report = {"phase": 1660, "campaign": "C126", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "authorization": "append_memo" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/visualization_client_integration_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
