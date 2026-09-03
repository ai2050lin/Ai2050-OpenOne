#!/usr/bin/env python3
"""Export the C261-C262 coverage and full-word specificity boundary with all coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
C249 = RESULT / "phase1783_c249_third_material_event_core_prediction"
C261 = RESULT / "phase1795_c261_coordinate_coverage_generation_side_effects"
C262 = RESULT / "phase1796_c262_full_word_generation_correction"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c262_generation_specificity_atlas.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    c261 = load(C261 / "analysis/summary.json")
    c262 = load(C262 / "analysis/summary.json")
    tri = np.load(C249 / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    rows = []
    for checkpoint in range(17):
        values = np.asarray(tri[0, 0, checkpoint, 4], dtype=np.int8)
        rows.append({
            "source": "c262_generation_specificity",
            "family": "attitude_event",
            "effect": "factor_a",
            "checkpoint": checkpoint,
            "checkpoint_type": "embedding" if checkpoint == 0 else "hidden_state",
            "role": "relation",
            "event_count": int(np.count_nonzero(values)),
            "label": f"C262/attitude/factor_a/q{checkpoint}/relation",
            "values": values.tolist(),
        })
    payload = {
        "schema": "c262_generation_specificity_atlas.v1",
        "phase": 1796,
        "campaign": "C261-C262",
        "model": "Qwen3-4B",
        "dimensions": 2560,
        "default_coordinates": 64,
        "coordinate_semantics": "Columns are physical Qwen3-4B activation coordinates; q0 is embedding and q1-q16 are post-block pre-norm HiddenStates. Values are tri-material signed events, not weights or neurons.",
        "claim_boundary": "The 75% coverage threshold is registered-grid sufficiency, not a minimal coalition. Full-word generation failed specificity because reversed checkpoint masks matched the correct path at 16/16. C260 is a leading-space token-logit result, not natural word closure.",
        "summary": {
            "coverage": c261["coverage_summary"],
            "earliest_fraction": c261["earliest_registered_fraction_at_flip_0_8"],
            "midpoint_erasure_passed": c261["midpoint_erasure_gate_passed"],
            "erasure_control_margin": c261["erasure_control_margin"],
            "full_word_generation": c262["summaries"],
            "correct_minus_best_control": c262["correct_minus_best_control"],
            "full_word_generation_gate_passed": c262["full_word_generation_gate_passed"],
        },
        "rows": rows,
    }
    save(ASSET, payload)
    checks = {
        "schema": load(ASSET)["schema"] == payload["schema"],
        "rows": len(rows) == 17,
        "all_coordinates": all(len(row["values"]) == 2560 for row in rows),
        "source_audits": load(C261 / "analysis/final.json")["all_checks_passed"] and load(C262 / "analysis/final.json")["all_checks_passed"],
        "control_collision_visible": payload["summary"]["correct_minus_best_control"] == 0,
    }
    digest = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    report = {"phase": 1796, "campaign": "C262", "asset": str(ASSET.relative_to(ROOT)).replace("\\", "/"), "asset_sha256": digest, "asset_bytes": ASSET.stat().st_size, "checks": checks, "all_checks_passed": all(checks.values())}
    save(C262 / "visualization/heatmap_export_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
