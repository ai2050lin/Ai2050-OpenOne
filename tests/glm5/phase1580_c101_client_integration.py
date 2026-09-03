#!/usr/bin/env python3
"""Verify the C101 activation-coordinate heatmap client integration."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1575_c101_dual_arm"
ASSET = OUT / "visualization/c101_activation_coordinate_heatmap.json"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c101_activation_coordinate_heatmap.json"
BUILT = ROOT / "frontend/dist/vis_data/research_kernel/c101_activation_coordinate_heatmap.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8")
    hook = (ROOT / "frontend/src/researchKernel/useResearchKernel.js").read_text(encoding="utf-8")
    card = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    app = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
    payload = load(ASSET)
    hashes = {"canonical": sha(ASSET), "public": sha(PUBLIC), "built": sha(BUILT)}
    checks = {
        "schema": payload["schema"] == "c101_activation_coordinate_heatmap.v1",
        "full_coordinates": payload["dimensions"] == list(range(2560)),
        "embedding_and_hidden": {row["state_kind"] for row in payload["raw_rows"]} == {"embedding", "hidden_state"},
        "asset_identity": len(set(hashes.values())) == 1,
        "route": "C101_ACTIVATION_COORDINATE_HEATMAP_ROUTE" in route and "activation_coordinate_heatmap" in route,
        "hook": "c101ActivationHeatmap" in hook,
        "card": "buildC101ActivationHeatmapData" in card and "activation coordinate" in card,
        "app": "c101ActivationHeatmap={realResearchTrace.c101ActivationHeatmap}" in app,
        "claim_scope": "not weight parameters" in payload["coordinate_semantics"],
    }
    report = {
        "phase": 1580,
        "campaign": "C101",
        "status": "client_integration_verified",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "hashes": hashes,
        "built_asset_bytes": BUILT.stat().st_size,
        "next_authorization": {
            "status": "authorized_for_separate_preregistration",
            "candidate_campaign": "C102",
            "scope": "sequential GLM4 then DeepSeek-7B validation of layer-relative late response geometry",
            "constraints": [
                "do not compare physical coordinate numbers across model families",
                "capture embedding and all hidden-state coordinates only",
                "type behavior and internal observation as separate evidence",
                "run one CUDA model at a time",
            ],
        },
    }
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    output = OUT / "analysis/client_integration.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
