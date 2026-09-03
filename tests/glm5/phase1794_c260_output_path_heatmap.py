#!/usr/bin/env python3
"""Export C260 early-path causal evidence with all 2560 activation coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
C249 = RESULT / "phase1783_c249_third_material_event_core_prediction"
C260 = RESULT / "phase1794_c260_path_ladder_natural_word_readout"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c260_output_path_causal_atlas.json"
OUT = C260 / "visualization"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary = load(C260 / "analysis/summary.json")
    tri = np.load(C249 / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    # Axis order is family, effect, checkpoint, role, coordinate.
    attitude = 0
    factor_a = 0
    roles = {"relation": 4, "boundary": 5}
    rows = []
    for checkpoint in range(17):
        for role, role_i in roles.items():
            values = np.asarray(tri[attitude, factor_a, checkpoint, role_i], dtype=np.int8)
            rows.append({
                "source": "c260_early_output_path",
                "family": "attitude_event",
                "effect": "factor_a",
                "checkpoint": checkpoint,
                "checkpoint_type": "embedding" if checkpoint == 0 else "hidden_state",
                "role": role,
                "event_count": int(np.count_nonzero(values)),
                "label": f"C260/attitude/factor_a/q{checkpoint}/{role}",
                "values": values.tolist(),
            })

    payload = {
        "schema": "c260_output_path_causal_atlas.v1",
        "phase": 1794,
        "campaign": "C260",
        "model": "Qwen3-4B",
        "dimensions": 2560,
        "default_coordinates": 64,
        "coordinate_semantics": (
            "Columns are Qwen3-4B physical activation coordinates. q0 is embedding; "
            "q1-q16 are post-block pre-norm HiddenState checkpoints. Values -1/0/+1 "
            "are tri-material same-sign events, not weights or independent neurons."
        ),
        "claim_boundary": (
            "The prefix and role ladders localize a sufficient distributed intervention route only "
            "within the registered masks. They do not establish coordinate minimality, natural "
            "necessity, free-generation closure, or an Attention/MLP circuit."
        ),
        "summary": {
            "earliest_passing_prefix_end": summary["earliest_passing_prefix_end"],
            "prefix16_vs_best_control_margin": summary["prefix16_vs_best_control_margin"],
            "path_ladder_gate_passed": summary["path_ladder_gate_passed"],
            "natural_word_control_margin": summary["natural_word_control_margin"],
            "natural_word_gate_passed": summary["natural_word_gate_passed"],
            "ladder_summary": summary["ladder_summary"],
            "natural_word_summary": summary["natural_word_summary"],
        },
        "rows": rows,
    }
    save(ASSET, payload)
    checks = {
        "schema": load(ASSET)["schema"] == payload["schema"],
        "rows": len(rows) == 34,
        "all_coordinates": all(len(row["values"]) == 2560 for row in rows),
        "embedding_and_hidden_state": {row["checkpoint_type"] for row in rows} == {"embedding", "hidden_state"},
        "source_final_passed": load(C260 / "analysis/final.json")["all_checks_passed"],
    }
    report = {
        "phase": 1794,
        "campaign": "C260",
        "asset": str(ASSET.relative_to(ROOT)).replace("\\", "/"),
        "asset_sha256": sha(ASSET),
        "asset_bytes": ASSET.stat().st_size,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save(OUT / "heatmap_export_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
