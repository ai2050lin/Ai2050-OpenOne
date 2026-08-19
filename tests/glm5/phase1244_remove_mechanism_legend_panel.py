"""Verify removal of the 3D mechanism legend without removing its workspace."""

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
        / "tests/glm5/result/phase1244_remove_mechanism_legend/phase1244_summary.json",
    )
    args = parser.parse_args()

    app_source = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
    workspace_source = (
        ROOT / "frontend/src/components/app/MechanismWorkspace.jsx"
    ).read_text(encoding="utf-8")
    workspace_css = (
        ROOT / "frontend/src/components/app/MechanismWorkspace.css"
    ).read_text(encoding="utf-8")
    combined = "\n".join((app_source, workspace_source, workspace_css))

    removed_labels = (
        "Layer深度",
        "Token位置",
        "组件状态",
        "Layer 3D模型已保留",
    )
    checks = {
        "legend_not_imported_or_rendered": "MechanismSpaceLegend" not in app_source,
        "legend_component_removed": "MechanismSpaceLegend" not in workspace_source,
        "legend_styles_removed": "mechanism-space-legend" not in workspace_css,
        "legend_labels_removed": all(label not in combined for label in removed_labels),
        "mechanism_mode_switch_retained": "<MechanismModeSwitch" in app_source,
        "mechanism_workspace_dock_retained": "<MechanismWorkspaceDock" in app_source,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "1.0.0",
        "phase_id": "Phase1244",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": passed,
        "checks": checks,
        "scope": "Remove only the Layer/token/component 3D legend panel.",
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
