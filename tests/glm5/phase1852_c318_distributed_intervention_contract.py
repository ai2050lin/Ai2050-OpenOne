#!/usr/bin/env python3
"""C318: freeze source-role counts and coordinate-width causal doses."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


WIDTHS = (16, 64, 256, 2560)


def main() -> None:
    parent = common.core.load(common.OUTS["C317"] / "analysis/final.json")
    selected = common.core.load(common.OUTS["C317"] / "protocol/selected_strata.json")["selected"]
    checks = {"parent": parent["all_checks_passed"], "six_strata": len(selected) == 6, "full_width_primary_reference": WIDTHS[-1] == 2560, "confirmation_units_only": True}
    protocol = {
        "status": "distributed_intervention_frozen",
        "base_cases": "sixth H11, order +1, units 4-7, both surfaces: 48",
        "source_role_counts": {"1": ["relation"], "2": ["relation", "query"], "3": ["primary", "relation", "query"]},
        "coordinate_widths": list(WIDTHS),
        "coordinate_selection": "16/64/256 are training-residual-amplitude diagnostic prefixes; 2560 is the primary complete-field reference. No small mask is treated as the mechanism definition.",
        "polarities": ["delete (- residual)", "enhance (+ residual)"],
        "evaluations": 48 * 3 * 4 * 2,
        "primary_question": "Does effect grow with source-role count and physical coordinate coverage, and reverse with polarity?",
        "claim_boundary": "Dose dependence can support distributed implementation, but amplitude-ranked masks are diagnostics and cannot prove minimal coordinates or unique routes.",
    }
    out = common.prepare("C318", protocol, checks)
    atlas = np.load(common.OUTS["C314"] / "analysis/operator_passports.float32.npy", mmap_mode="r")
    masks = np.zeros((6, len(WIDTHS), 2560), bool)
    configurations = []
    for family_i, row in enumerate(selected):
        q = row["q"]
        destination = row["destination_index"]
        amplitude = np.abs(np.asarray(atlas[family_i, 0, q + 1, destination], np.float32))
        ranking = np.argsort(-amplitude, kind="stable")
        for width_i, width in enumerate(WIDTHS):
            masks[family_i, width_i, ranking[:width]] = True
        configurations.append({"family": row["family"], "q": q, "destination_role": row["destination_role"], "destination_index": destination, "training_selected_source_roles": row["source_roles"], "diagnostic_widths": list(WIDTHS), "all_coordinate_reference": 2560})
    np.save(out / "protocol/coordinate_width_masks.bool.npy", masks)
    common.core.save(out / "protocol/intervention_configurations.json", {"configurations": configurations})
    headline = {"status": "distributed_intervention_contract_closed", "configurations": configurations, "evaluations": protocol["evaluations"], "strict_interpretation": protocol["claim_boundary"]}
    common.close("C318", headline, {"mask_shape": list(masks.shape) == [6, 4, 2560], "mask_counts": all(int(masks[i, j].sum()) == width for i in range(6) for j, width in enumerate(WIDTHS)), "six_configs": len(configurations) == 6}, "C319_distributed_width_dose_causal_test")


if __name__ == "__main__":
    main()
