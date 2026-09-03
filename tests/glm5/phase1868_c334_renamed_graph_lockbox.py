#!/usr/bin/env python3
"""C334: test frozen graph operators on renamed graphs and path controls."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer

import phase1844_c310_c335_dual_axis_common as common
from model_utils import MODEL_CONFIGS


def mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def gain(truth: np.ndarray, error: np.ndarray) -> float:
    return 1.0 - mae(error) / max(mae(truth), 1e-12)


def main() -> None:
    parent = common.core.load(common.OUTS["C333"] / "analysis/final.json")
    rows = common.graph_material(lockbox=True)
    checks = {"parent": parent["all_checks_passed"], "rows": len(rows) == 288, "renamed_units": not ({u["root"] for u in common.GRAPH_UNITS} & {u["root"] for u in common.GRAPH_LOCKBOX_UNITS}), "modes": {r["mode"] for r in rows} == {"chain", "reversed", "broken", "irrelevant", "multipath", "shortcut"}, "cuda": torch.cuda.is_available()}
    protocol = {
        "status": "renamed_graph_lockbox_frozen",
        "material": "six disjoint renamed graphs x four depths x two surfaces x six path modes",
        "modes": ["chain", "shortcut", "multipath", "irrelevant", "reversed", "broken"],
        "prediction": "C333 operators are frozen before this material is tokenized or run",
        "claim_boundary": "This lockbox separates lexical renaming and path controls, but remains controlled English and observational HiddenState analysis.",
    }
    out = common.prepare("C334", protocol, checks)
    common.core.write_rows(out / "material/cases.jsonl", rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    compiled = common.compile_general(tokenizer, rows, "strict_chat")
    common.core.write_rows(out / "compiled/qwen3.jsonl", compiled)
    capture = common.batch_capture_qwen(rows, compiled, out, full_selector=None, batch_size=8, field_width=144)
    states = np.load(out / "raw/role_states.float16.npy", mmap_mode="r")
    index = common.core.rows(out / "raw/hidden_index.jsonl")
    atlas = np.load(common.OUTS["C333"] / "analysis/depth_operator_atlas.float32.npy", mmap_mode="r")
    shortcut_operator = np.load(common.OUTS["C333"] / "analysis/shortcut_operator.float32.npy", mmap_mode="r")
    lookup = {(r["unit"], r["surface"], r["mode"], r["depth"]): np.asarray(states[r["hidden_index"]], np.float32) for r in index}
    depth_results = []
    for depth in (2, 3, 4):
        truth = np.stack([lookup[(u, s, "chain", depth)] - lookup[(u, s, "chain", depth - 1)] for u in range(6) for s in ("registry", "briefing")])
        operator = atlas[depth - 2, 0]
        depth_results.append({"depth_transition": f"{depth - 1}->{depth}", "groups": len(truth), "relative_mae_gain": gain(truth, truth - operator), "coordinate_roll_gain": gain(truth, truth - np.roll(operator, 97, axis=-1))})
    shortcut_truth = np.stack([lookup[(u, s, "shortcut", d)] - lookup[(u, s, "chain", d)] for u in range(6) for s in ("registry", "briefing") for d in range(1, 5)])
    shortcut = {"groups": len(shortcut_truth), "relative_mae_gain": gain(shortcut_truth, shortcut_truth - shortcut_operator), "coordinate_roll_gain": gain(shortcut_truth, shortcut_truth - np.roll(shortcut_operator, 97, axis=-1))}
    behavior = common.core.rows(out / "raw/behavior.jsonl")
    source = {r["case_id"]: r for r in rows}
    by_mode = {mode: float(np.mean([r["correct"] for r in behavior if source[r["case_id"]]["mode"] == mode])) for mode in sorted({r["mode"] for r in rows})}
    mode_distances = {}
    for mode in ("reversed", "broken", "irrelevant", "multipath", "shortcut"):
        values = [mae(lookup[(u, s, mode, d)] - lookup[(u, s, "chain", d)]) for u in range(6) for s in ("registry", "briefing") for d in range(1, 5)]
        mode_distances[mode] = float(np.mean(values))
    lockbox_gate = sum(r["relative_mae_gain"] > 0 and r["relative_mae_gain"] > r["coordinate_roll_gain"] for r in depth_results) >= 2
    headline = {"status": "renamed_graph_lockbox_adjudicated", "capture": capture, "behavior_by_mode": by_mode, "depth_results": depth_results, "shortcut_prediction": shortcut, "mean_full_field_distance_from_chain": mode_distances, "renamed_depth_gate_passed": lockbox_gate, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C334", headline, {"capture_rows": capture["rows"] == 288, "role_shape": capture["role_shape"] == [288, 38, 6, 2560], "depth_results": len(depth_results) == 3, "finite": common.finite_dict(headline), "renamed_disjoint": checks["renamed_units"]}, "C335_major_stage_synthesis")


if __name__ == "__main__":
    main()
