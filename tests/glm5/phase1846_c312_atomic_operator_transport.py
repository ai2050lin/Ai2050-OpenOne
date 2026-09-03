#!/usr/bin/env python3
"""C312: forecast atomic factor edits across the other factor's state."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


OPERATORS = ("a0", "a1", "b0", "b1")
TARGETS = (("a0", "a0"), ("a0_to_b1", "a1"), ("b0", "b0"), ("b0_to_a1", "b1"))


def main() -> None:
    parent = common.core.load(common.OUTS["C311"] / "analysis/final.json")
    checks = {"parent_complete": parent["all_checks_passed"], "continue_after_specificity_result": True, "all_coordinates": True}
    protocol = {
        "status": "atomic_transport_frozen",
        "atomic_edits": {
            "A": "mean H10-H00 learned on third/fourth/fifth; predict both sixth H10-H00 and H11-H01",
            "B": "mean H01-H00 learned on third/fourth/fifth; predict both sixth H01-H00 and H11-H10",
        },
        "controls": ["zero edit", "global all-family edit", "cyclic wrong-family RMS-matched edit"],
        "family_gate": "both cross-context A and B family gains>=0.01 and each exceeds global and wrong-family gains",
        "claim_boundary": "This tests transport of mean finite differences across a second factor. It is not an operator acting on arbitrary hidden states and does not establish order-sensitive composition.",
    }
    out = common.prepare("C312", protocol, checks)
    means = {family: {} for family in common.FAMILIES}
    all_effects = {name: [] for name in OPERATORS}
    for family in common.FAMILIES:
        collected = {name: [] for name in OPERATORS}
        for state_path, index_path, _material in common.TRAIN_ROLE_SOURCES:
            arrays, _groups = common.factorial_arrays(np.load(state_path, mmap_mode="r"), common.core.rows(index_path), family)
            for name in OPERATORS:
                collected[name].append(arrays[name])
        for name in OPERATORS:
            merged = np.concatenate(collected[name], axis=0)
            means[family][name] = merged.mean(axis=0)
            all_effects[name].append(merged)
    global_means = {name: np.concatenate(all_effects[name], axis=0).mean(axis=0) for name in OPERATORS}

    test_states = np.load(common.SIXTH_STATES, mmap_mode="r")
    test_index = common.core.rows(common.SIXTH_INDEX)
    atlas = np.zeros((6, 4, 4, 37, 6, 2560), np.float32)
    family_rows = []
    group_rows = []
    for family_i, family in enumerate(common.FAMILIES):
        arrays, groups = common.factorial_arrays(test_states, test_index, family)
        wrong_family = common.FAMILIES[(family_i + 1) % 6]
        target_specs = (
            ("A_same_context", "a0", "a0"),
            ("A_cross_context", "a1", "a0"),
            ("B_same_context", "b0", "b0"),
            ("B_cross_context", "b1", "b0"),
        )
        rows = []
        for target_i, (label, truth_name, learned_name) in enumerate(target_specs):
            truth = arrays[truth_name]
            family_prediction = means[family][learned_name]
            wrong_prediction = common.norm_match(means[wrong_family][learned_name], family_prediction)
            predictions = {
                "family": family_prediction,
                "global": global_means[learned_name],
                "wrong_family": wrong_prediction,
            }
            atlas[family_i, target_i, 0] = np.mean(np.abs(truth), axis=0)
            gains = {}
            for model_i, (name, prediction) in enumerate(predictions.items(), start=1):
                error = truth - prediction[None, ...]
                atlas[family_i, target_i, model_i] = np.mean(np.abs(error), axis=0)
                gains[name] = common.relative_gain(truth, error)
            result = {"target": label, "groups": len(groups), "gains": gains}
            rows.append(result)
            for group_i, group in enumerate(groups):
                base = float(np.mean(np.abs(truth[group_i])))
                group_rows.append({
                    "family": family,
                    "target": label,
                    "surface": group["surface"],
                    "unit": group["unit"],
                    "order": group["order"],
                    **{
                        f"gain_{name}": float((base - np.mean(np.abs(truth[group_i] - prediction))) / max(base, 1e-12))
                        for name, prediction in predictions.items()
                    },
                })
        cross = {row["target"]: row for row in rows}
        passed = all(
            cross[label]["gains"]["family"] >= 0.01
            and cross[label]["gains"]["family"] > max(cross[label]["gains"]["global"], cross[label]["gains"]["wrong_family"])
            for label in ("A_cross_context", "B_cross_context")
        )
        family_rows.append({"family": family, "targets": rows, "family_gate_passed": passed})
        print(f"[C312] {family}: A={cross['A_cross_context']['gains']['family']:+.5f} B={cross['B_cross_context']['gains']['family']:+.5f} pass={passed}", flush=True)
    np.save(out / "analysis/atomic_transport_full_coordinate_errors.float32.npy", atlas)
    common.core.write_rows(out / "analysis/family_results.jsonl", family_rows)
    common.core.write_rows(out / "raw/group_results.jsonl", group_rows)
    passing = [row["family"] for row in family_rows if row["family_gate_passed"]]
    headline = {"status": "atomic_transport_adjudicated", "families": family_rows, "families_passing": passing, "breadth_gate_passed": len(passing) >= 4, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C312", headline, {"six_families": len(family_rows) == 6, "four_targets_each": all(len(row["targets"]) == 4 for row in family_rows), "atlas_shape": list(atlas.shape) == [6, 4, 4, 37, 6, 2560], "finite": bool(np.isfinite(atlas).all())}, "C313_dual_axis_square_prediction")


if __name__ == "__main__":
    main()
