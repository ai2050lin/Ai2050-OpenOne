"""Verify the heatmap research route and its 2D/3D rendering fallbacks."""

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
        / "tests/glm5/result/phase1238_heatmap_research_route/phase1238_summary.json",
    )
    args = parser.parse_args()

    app_source = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
    plugin_source = (
        ROOT / "frontend/src/plugins/researchPlugins.js"
    ).read_text(encoding="utf-8")
    preview_source = (
        ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
    ).read_text(encoding="utf-8")
    component_source = (
        ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx"
    ).read_text(encoding="utf-8")
    component_css = (
        ROOT / "frontend/src/components/app/ResearchHeatmapRoute.css"
    ).read_text(encoding="utf-8")
    renderer_source = (
        ROOT / "frontend/src/neural_vis/renderers/Heatmap3DRenderer.jsx"
    ).read_text(encoding="utf-8")

    checks = {
        "heatmap_layer_registered": (
            "heatmap: {" in plugin_source and "热力图层" in plugin_source
        ),
        "heatmap_route_registered": all(
            marker in plugin_source
            for marker in (
                "id: 'heatmap-analysis'",
                "name: '热力图分析'",
                "routeType: 'heatmap_matrix'",
                "defaultLayers: ['heatmap']",
            )
        ),
        "route_has_heatmap_panel": (
            "{ id: 'heatmap', label: '热力图效果', defaultOpen: true }"
            in plugin_source
        ),
        "two_dimensional_preview_mounted": (
            "<ResearchHeatmapRouteCard" in app_source
            and "research-heatmap-card__matrix" in component_source
        ),
        "three_dimensional_preview_mounted": (
            "<ResearchHeatmapPreview3D" in app_source
            and "byType.heatmap_3d.length === 0" in app_source
        ),
        "real_heatmap_renderer_retained": (
            "<Heatmap3DRenderer" in app_source
            and "heatmap?.cells" in renderer_source
        ),
        "preview_has_axes_and_matrix": all(
            marker in preview_source
            for marker in (
                "xAxis:",
                "yAxis:",
                "values:",
                "'L31'",
                "'后继'",
            )
        ),
        "preview_is_not_claimed_as_evidence": (
            "视觉示例 · 非实验结果" in preview_source
            and "不能作为神经网络内部结构证据" in preview_source
        ),
        "color_and_height_share_value": (
            "const height = 0.18 + value * 3.5" in component_source
            and "const color = heatmapColor(value)" in component_source
        ),
        "heatmap_panel_status_visible": (
            "热力图已开启" in app_source
            and "加载真实 heatmap_3d 资产" in app_source
        ),
        "responsive_styles_present": (
            ".research-heatmap-card__matrix" in component_css
            and "grid-template-columns" in component_css
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "1.0.0",
        "phase_id": "Phase1238",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": passed,
        "checks": checks,
        "preview_shape": [6, 5],
        "preview_cells": 30,
        "preview_is_experimental_evidence": False,
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
