"""Verify that the 3D client's top-center research rail is removed cleanly."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "tests/glm5/result/phase1243_remove_top_research_panel/phase1243_summary.json",
    )
    args = parser.parse_args()

    app_source = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
    cockpit_source = (
        ROOT / "frontend/src/components/app/ResearchEvidenceCockpit.jsx"
    ).read_text(encoding="utf-8")
    cockpit_css = (
        ROOT / "frontend/src/components/app/ResearchEvidenceCockpit.css"
    ).read_text(encoding="utf-8")

    checks = {
        "top_rail_not_imported_or_rendered": "ResearchEvidenceRail" not in app_source,
        "top_rail_component_removed": "ResearchEvidenceRail" not in cockpit_source,
        "top_rail_styles_removed": "research-evidence-rail" not in cockpit_css,
        "left_research_cockpit_retained": "<ResearchEvidenceCockpit" in app_source,
        "research_detail_drawer_retained": "<ResearchEvidenceDrawer" in app_source,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "1.0.0",
        "phase_id": "Phase1243",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": passed,
        "checks": checks,
        "scope": "Remove only the 3D client top-center current-research rail.",
        "model_runs": 0,
        "gpu_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
