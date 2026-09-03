#!/usr/bin/env python3
"""C324: test controlled and natural-discovery residuals on natural lockboxes."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


def main() -> None:
    parent = common.core.load(common.OUTS["C323"] / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "all_coordinates": True, "confirmation_units_and_surfaces_frozen": True}
    protocol = {
        "status": "natural_surface_composition_frozen",
        "old_model": "family interaction residual learned on controlled third/fourth/fifth materials",
        "natural_model": "family interaction residual learned on natural surfaces report/briefing/notes and units0-3",
        "lockbox": "natural surfaces archive/witness and units4-7",
        "controls": ["zero/additive", "controlled global mean", "cyclic wrong-family RMS-matched mean"],
        "family_gate": "natural discovery residual lockbox gain>=0.01 and exceeds global and wrong-family controls",
        "claim_boundary": "Passing shows transfer across these frozen wrappers and lexicon units. Without real human blind ratings it cannot be called human-validated natural-language invariance.",
    }
    out = common.prepare("C324", protocol, checks)
    states = np.load(common.OUTS["C323"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = common.core.rows(common.OUTS["C323"] / "raw/hidden_index.jsonl")
    controlled_atlas = np.load(common.OUTS["C314"] / "analysis/operator_passports.float32.npy", mmap_mode="r")
    controlled_means = {family: np.asarray(controlled_atlas[i, 0], np.float32) for i, family in enumerate(common.FAMILIES)}
    global_mean = np.mean(np.stack(list(controlled_means.values()), axis=0), axis=0)
    atlas = np.zeros((6, 5, 37, 6, 2560), np.float32)
    family_rows = []
    group_rows = []
    for family_i, family in enumerate(common.FAMILIES):
        arrays, groups = common.factorial_arrays(states, index, family)
        truth = arrays["interaction"]
        discovery_mask = np.asarray([group["unit"] < 4 and group["surface"] in common.NATURAL_SURFACES[:3] for group in groups], bool)
        lockbox_mask = np.asarray([group["unit"] >= 4 and group["surface"] in common.NATURAL_SURFACES[3:] for group in groups], bool)
        natural_mean = truth[discovery_mask].mean(axis=0)
        lockbox = truth[lockbox_mask]
        wrong = controlled_means[common.FAMILIES[(family_i + 1) % 6]]
        predictions = {
            "controlled_family": controlled_means[family],
            "controlled_global": global_mean,
            "wrong_family": common.norm_match(wrong, natural_mean),
            "natural_discovery": natural_mean,
        }
        atlas[family_i, 0] = np.mean(np.abs(lockbox), axis=0)
        gains = {}
        for model_i, (name, prediction) in enumerate(predictions.items(), start=1):
            error = lockbox - prediction[None, ...]
            atlas[family_i, model_i] = np.mean(np.abs(error), axis=0)
            gains[name] = common.relative_gain(lockbox, error)
        passed = gains["natural_discovery"] >= 0.01 and gains["natural_discovery"] > max(gains["controlled_global"], gains["wrong_family"])
        family_rows.append({"family": family, "eligible_groups": len(groups), "discovery_groups": int(discovery_mask.sum()), "lockbox_groups": int(lockbox_mask.sum()), "gains": gains, "family_gate_passed": passed})
        lockbox_groups = [group for group, keep in zip(groups, lockbox_mask) if keep]
        for group_i, group in enumerate(lockbox_groups):
            base = float(np.mean(np.abs(lockbox[group_i])))
            group_rows.append({"family": family, "surface": group["surface"], "unit": group["unit"], "order": group["order"], **{f"gain_{name}": float((base - np.mean(np.abs(lockbox[group_i] - prediction))) / max(base, 1e-12)) for name, prediction in predictions.items()}})
        print(f"[C324] {family}: old={gains['controlled_family']:+.5f} natural={gains['natural_discovery']:+.5f} pass={passed}", flush=True)
    np.save(out / "analysis/natural_surface_full_coordinate_errors.float32.npy", atlas)
    common.core.write_rows(out / "analysis/family_results.jsonl", family_rows)
    common.core.write_rows(out / "raw/lockbox_group_results.jsonl", group_rows)
    passing = [row["family"] for row in family_rows if row["family_gate_passed"]]
    headline = {"status": "natural_surface_composition_adjudicated", "families": family_rows, "families_passing": passing, "breadth_gate_passed": len(passing) >= 4, "human_naturalness": "no_test", "strict_interpretation": protocol["claim_boundary"]}
    common.close("C324", headline, {"six_families": len(family_rows) == 6, "expected_lockbox_groups": len(group_rows) == 6 * 2 * 4 * 2, "atlas_shape": list(atlas.shape) == [6, 5, 37, 6, 2560], "finite": bool(np.isfinite(atlas).all())}, "C325_natural_surface_stage_audit")


if __name__ == "__main__":
    main()
