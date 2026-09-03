#!/usr/bin/env python3
"""Independent C129 heatmap client integration audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1663_c129_direct_precedence_typed_transition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
CARD = ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    payload = core.load(PUBLIC); batch = payload["c129_direct_precedence_typed_transition_batch"]
    card = CARD.read_text(encoding="utf-8"); route = ROUTE.read_text(encoding="utf-8")
    checks = {"phase": payload["phase"] == 1663 and "C129" in payload["campaign"], "asset": len(batch["effect_rows"]) == 150 and len(batch["representative_raw_rows"]) == 35 and all(len(row["values"]) == 2560 for row in [*batch["effect_rows"], *batch["representative_raw_rows"]]), "checkpoints": any(row["checkpoint"] == "embedding" for row in batch["effect_rows"] if row["kind"] == "truth_response") and any(row["checkpoint"] == "post_final_norm" for row in batch["effect_rows"] if row["kind"] == "truth_response"), "builder": all(token in card for token in ("c129Rows", "c129Profiles", "c129RawRows")), "render": all(token in card for token in ("C129 真值响应状态与增量", "C129 代表样本原始词嵌入与 HiddenState", "activation coordinate ${cell.dimension}")), "route": "C109-C129 Relation-Role-State and Typed Transition Atlas" in route and "不是参数权重" in route, "boundary": "not weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"]}
    report = {"phase": 1663, "campaign": "C129", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "frontend_build": "not_run_npm_unavailable", "authorization": "append_memo" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/visualization_client_integration_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
