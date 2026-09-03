#!/usr/bin/env python3
"""C325: close the computational natural-surface branch without faking raters."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


def main() -> None:
    finals = {campaign: common.core.load(common.OUTS[campaign] / "analysis/final.json") for campaign in ("C321", "C322", "C323", "C324")}
    checks = {"all_computational_phases": all(value["all_checks_passed"] for value in finals.values()), "human_no_test_preserved": finals["C324"]["headline"]["human_naturalness"] == "no_test", "continue_cross_model": True}
    protocol = {"status": "natural_stage_audit_frozen", "scope": ["C321", "C322", "C323", "C324"], "human_review": "external dependency remains open and is not silently imputed", "claim_boundary": "The computational branch is complete. Naturalness remains machine-audited controlled English, not human-validated natural language."}
    out = common.prepare("C325", protocol, checks)
    role_path = common.OUTS["C323"] / "raw/role_states.float16.npy"
    full_path = common.OUTS["C323"] / "raw/full_fields_holdout.float16.npy"
    audit = {
        "material_rows": len(common.core.rows(common.OUTS["C321"] / "material/cases.jsonl")) == 1920,
        "behavior_rows": len(common.core.rows(common.OUTS["C322"] / "raw/behavior.jsonl")) == 1920,
        "role_shape": list(np.load(role_path, mmap_mode="r").shape) == [1920, 38, 6, 2560],
        "full_token_shape": list(np.load(full_path, mmap_mode="r").shape) == [192, 38, 128, 2560],
        "lockbox_groups": len(common.core.rows(common.OUTS["C324"] / "raw/lockbox_group_results.jsonl")) == 96,
        "real_human_labels_absent": not (common.OUTS["C321"] / "raw/human_ratings.jsonl").exists(),
    }
    headline = {"status": "natural_surface_stage_closed", "behavior_eligible": finals["C322"]["headline"]["behavior_eligible"], "composition_breadth_gate": finals["C324"]["headline"]["breadth_gate_passed"], "human_naturalness": "no_test", "audit": audit, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C325", headline, audit, "C326_C330_cross_model_natural_panel")


if __name__ == "__main__":
    main()
