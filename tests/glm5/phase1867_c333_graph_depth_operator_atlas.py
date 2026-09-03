#!/usr/bin/env python3
"""C333: prospectively predict full-coordinate graph-depth increments."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


def mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def gain(truth: np.ndarray, error: np.ndarray) -> float:
    return 1.0 - mae(error) / max(mae(truth), 1e-12)


def main() -> None:
    parent = common.core.load(common.OUTS["C332"] / "analysis/final.json")
    states = np.load(common.OUTS["C332"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = common.core.rows(common.OUTS["C332"] / "raw/hidden_index.jsonl")
    checks = {"parent": parent["all_checks_passed"], "shape": list(states.shape) == [384, 38, 6, 2560], "prospective_partition": True, "no_top_k": True}
    protocol = {
        "status": "graph_depth_operator_analysis_frozen",
        "training": "graphs 0-7 only",
        "test": "graphs 8-11 only",
        "operator": "mean full-coordinate H(depth)-H(depth-1), separately for shortcut absent/present",
        "controls": ["zero increment", "coordinate roll by 97", "wrong shortcut operator"],
        "gate": "at least two of three depth transitions have positive test gain and beat both coordinate-roll and wrong-shortcut controls",
        "claim_boundary": "A passing mean increment is a task-conditional predictive field, not a proof of recursive graph reasoning or a unique mechanism.",
    }
    out = common.prepare("C333", protocol, checks)
    cells = {}
    for row in index:
        key = (row["unit"], row["surface"], row["shortcut"], row["depth"])
        cells.setdefault(key, []).append(row["hidden_index"])
    means = {k: np.asarray(states[v], np.float32).mean(axis=0) for k, v in cells.items() if len(v) == 2}
    atlas = np.zeros((3, 2, 38, 6, 2560), np.float32)
    rows = []
    for transition_i, depth in enumerate((2, 3, 4)):
        for shortcut in (0, 1):
            train = np.stack([means[(u, s, shortcut, depth)] - means[(u, s, shortcut, depth - 1)] for u in range(8) for s in ("registry", "briefing")])
            test = np.stack([means[(u, s, shortcut, depth)] - means[(u, s, shortcut, depth - 1)] for u in range(8, 12) for s in ("registry", "briefing")])
            operator = train.mean(axis=0)
            atlas[transition_i, shortcut] = operator
            wrong_train = np.stack([means[(u, s, 1 - shortcut, depth)] - means[(u, s, 1 - shortcut, depth - 1)] for u in range(8) for s in ("registry", "briefing")]).mean(axis=0)
            pred_error = test - operator
            roll_error = test - np.roll(operator, 97, axis=-1)
            wrong_error = test - wrong_train
            rows.append({"depth_transition": f"{depth - 1}->{depth}", "shortcut": shortcut, "groups_train": len(train), "groups_test": len(test), "zero_mae": mae(test), "operator_mae": mae(pred_error), "relative_mae_gain": gain(test, pred_error), "coordinate_roll_gain": gain(test, roll_error), "wrong_shortcut_gain": gain(test, wrong_error), "beats_controls": mae(pred_error) < min(mae(roll_error), mae(wrong_error))})
    shortcut_train = np.stack([means[(u, s, 1, d)] - means[(u, s, 0, d)] for u in range(8) for s in ("registry", "briefing") for d in range(1, 5)])
    shortcut_test = np.stack([means[(u, s, 1, d)] - means[(u, s, 0, d)] for u in range(8, 12) for s in ("registry", "briefing") for d in range(1, 5)])
    shortcut_operator = shortcut_train.mean(axis=0)
    np.save(out / "analysis/depth_operator_atlas.float32.npy", atlas)
    np.save(out / "analysis/shortcut_operator.float32.npy", shortcut_operator)
    common.core.write_rows(out / "analysis/depth_results.jsonl", rows)
    aggregate = []
    for depth in (2, 3, 4):
        subset = [r for r in rows if r["depth_transition"] == f"{depth - 1}->{depth}"]
        aggregate.append({"depth_transition": f"{depth - 1}->{depth}", "mean_gain": float(np.mean([r["relative_mae_gain"] for r in subset])), "all_beat_controls": all(r["beats_controls"] for r in subset)})
    gate = sum(r["mean_gain"] > 0 and r["all_beat_controls"] for r in aggregate) >= 2
    shortcut_result = {"groups_train": len(shortcut_train), "groups_test": len(shortcut_test), "relative_mae_gain": gain(shortcut_test, shortcut_test - shortcut_operator), "coordinate_roll_gain": gain(shortcut_test, shortcut_test - np.roll(shortcut_operator, 97, axis=-1))}
    headline = {"status": "graph_depth_operator_adjudicated", "behavior_eligible": parent["headline"]["behavior_eligible"], "depth_results": rows, "depth_aggregate": aggregate, "depth_prediction_gate_passed": gate, "shortcut_prediction": shortcut_result, "atlas_shape": list(atlas.shape), "strict_interpretation": protocol["claim_boundary"]}
    common.close("C333", headline, {"cells_complete": len(means) == 12 * 2 * 2 * 4, "results": len(rows) == 6, "atlas": list(atlas.shape) == [3, 2, 38, 6, 2560], "finite": common.finite_dict(headline)}, "C334_renamed_graph_lockbox")


if __name__ == "__main__":
    main()
