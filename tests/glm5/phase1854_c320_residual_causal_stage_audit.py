#!/usr/bin/env python3
"""C320: independent internal audit of C310-C319."""
from __future__ import annotations

import json

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


def main() -> None:
    campaigns = tuple(f"C{i}" for i in range(310, 320))
    finals = {campaign: common.core.load(common.OUTS[campaign] / "analysis/final.json") for campaign in campaigns}
    checks = {"all_preceding_closed": all(value["all_checks_passed"] for value in finals.values()), "no_missing_campaign": len(finals) == 10, "continue_natural_branch": True}
    protocol = {"status": "stage_audit_frozen", "scope": list(campaigns), "audit_rule": "Recompute structural counts and preserve strong-gate failures rather than equating execution success with scientific success.", "claim_boundary": "This audit checks provenance, shapes, counts, and claim typing; it is not independent experimental replication."}
    out = common.prepare("C320", protocol, checks)
    scientific = {
        "C311_specificity": finals["C311"]["headline"]["breadth_gate_passed"],
        "C312_atomic_transport": finals["C312"]["headline"]["breadth_gate_passed"],
        "C313_dual_axis_increment": finals["C313"]["headline"]["breadth_gate_passed"],
        "C315_coarse_field_causal": finals["C315"]["headline"]["breadth_gate_passed"],
        "C316_amplitude_phase": finals["C316"]["headline"]["breadth_gate_passed"],
        "C317_response_grammar": finals["C317"]["headline"]["breadth_gate_passed"],
        "C319_distributed_dose": finals["C319"]["headline"]["distributed_gate_passed"],
    }
    audit_checks = {
        "C311_atlas": list(np.load(common.OUTS["C311"] / "analysis/full_coordinate_error_atlas.float32.npy", mmap_mode="r").shape) == [6, 9, 37, 6, 2560],
        "C312_atlas": list(np.load(common.OUTS["C312"] / "analysis/atomic_transport_full_coordinate_errors.float32.npy", mmap_mode="r").shape) == [6, 4, 4, 37, 6, 2560],
        "C313_atlas": list(np.load(common.OUTS["C313"] / "analysis/dual_axis_full_coordinate_errors.float32.npy", mmap_mode="r").shape) == [6, 3, 4, 36, 6, 2560],
        "C314_atlas": list(np.load(common.OUTS["C314"] / "analysis/operator_passports.float32.npy", mmap_mode="r").shape) == [6, 8, 37, 6, 2560],
        "C315_samples": len(common.core.rows(common.OUTS["C315"] / "raw/sample_results.jsonl")) == 96,
        "C316_bins": len(common.core.rows(common.OUTS["C316"] / "analysis/amplitude_bins.jsonl")) == 60,
        "C317_score_shape": list(np.load(common.OUTS["C317"] / "analysis/all_coordinate_response_grammar_scores.float32.npy", mmap_mode="r").shape) == [6, 3, 36, 6, 2],
        "C318_masks": list(np.load(common.OUTS["C318"] / "protocol/coordinate_width_masks.bool.npy", mmap_mode="r").shape) == [6, 4, 2560],
        "C319_evaluations": len(common.core.rows(common.OUTS["C319"] / "raw/sample_results.jsonl")) == 1152,
        "strong_gates_typed": set(scientific.values()) <= {True, False},
    }
    headline = {"status": "residual_causal_stage_audited", "scientific_gates": scientific, "audit_checks": audit_checks, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C320", headline, audit_checks, "C321_C325_natural_surface_branch")


if __name__ == "__main__":
    main()
