#!/usr/bin/env python3
"""C323: capture all role coordinates and a full-token natural-surface lockbox."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


def main() -> None:
    parent = common.core.load(common.OUTS["C322"] / "analysis/final.json")
    eligible = parent["headline"]["behavior_eligible"]
    checks = {"parent": parent["all_checks_passed"], "behavior_gate": eligible, "all_role_coordinates": True, "full_token_confirmation_surface": True}
    protocol = {
        "status": "natural_full_coordinate_capture_frozen",
        "role_archive": "all 1,920 rows x 38 checkpoints x six registered roles x all 2,560 coordinates",
        "full_token_archive": "witness surface, confirmation units 4-7, both answer orders, all factorial cells: 192 rows x 38 x 128 x 2,560",
        "storage": "float16 archival copy; no activation coordinate is discarded",
        "analysis_scope": "hidden states and embeddings only",
        "claim_boundary": "Role means are functionally aligned summaries. The 192-row full-token panel preserves physical token detail but is one frozen surface, not all natural language.",
    }
    out = common.prepare("C323", protocol, checks)
    rows = common.core.rows(common.OUTS["C321"] / "material/cases.jsonl")
    compiled = common.core.rows(common.OUTS["C322"] / "compiled/qwen3.jsonl")
    capture = common.batch_capture_qwen(rows, compiled, out, full_selector=lambda row: row["surface"] == "witness" and row["partition"] == "confirmation", batch_size=8)
    source_behavior = {row["case_id"]: row for row in common.core.rows(common.OUTS["C322"] / "raw/behavior.jsonl")}
    recaptured = common.core.rows(out / "raw/behavior.jsonl")
    exact = all(row["prediction"] == source_behavior[row["case_id"]]["prediction"] for row in recaptured)
    headline = {"status": "natural_full_coordinate_capture_closed", **capture, "behavior_reproduction_exact": exact, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C323", headline, {"rows": capture["rows"] == 1920, "role_shape": capture["role_shape"] == [1920, 38, 6, 2560], "full_shape": capture["full_shape"] == [192, 38, 128, 2560], "behavior_exact": exact, "role_finite": bool(np.isfinite(np.load(out / "raw/role_states.float16.npy", mmap_mode="r")[:, :, :, ::128]).all())}, "C324_natural_surface_composition_lockbox")


if __name__ == "__main__":
    main()
