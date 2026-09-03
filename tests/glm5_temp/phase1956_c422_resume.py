#!/usr/bin/env python3
"""Resume C422 after the shared binary runner required a legacy cell field."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1949_c415_c425_dynamic_composition_campaign as campaign


def main() -> None:
    out = campaign.OUTS["C422"]
    if (out / "analysis/final.json").exists():
        raise RuntimeError("C422 is already closed")
    prereg = campaign.load(out / "protocol/preregistration.json")
    if prereg["producer_sha256"] != campaign.producer_hash():
        raise RuntimeError("Frozen C422 producer hash changed")
    rows = campaign.read_rows(out / "material/cases.jsonl")
    compiled = campaign.read_rows(out / "compiled/qwen3.jsonl")
    for row in compiled:
        row["cell"] = row["mode"]
    metrics = campaign.parent.previous.qwen_behavior(
        rows, compiled, out, batch_size=12
    )
    behavior = campaign.read_rows(out / "raw/behavior.jsonl")
    material = campaign.rows_by_id(rows)
    held = [row for row in behavior if row["partition"] != "discovery"]
    by_mode = {
        mode: float(
            np.mean(
                [
                    row["correct"]
                    for row in held
                    if material[row["case_id"]]["mode"] == mode
                ]
            )
        )
        for mode in campaign.GRAPH_MODES
    }
    by_channel = {
        channel: float(
            np.mean(
                [
                    row["correct"]
                    for row in held
                    if material[row["case_id"]]["channel"] == channel
                ]
            )
        )
        for channel in campaign.GRAPH_CHANNELS
    }
    by_polarity = {
        polarity: float(
            np.mean(
                [
                    row["correct"]
                    for row in held
                    if material[row["case_id"]]["polarity"] == polarity
                ]
            )
        )
        for polarity in campaign.GRAPH_POLARITIES
    }
    positive = {
        (
            material[row["case_id"]]["graph_id"],
            material[row["case_id"]]["channel"],
        ): row
        for row in held
        if material[row["case_id"]]["polarity"] == "positive"
        and material[row["case_id"]]["order"] == 1
        and material[row["case_id"]]["mode"] == "unknown"
    }
    unknown_graphs = sorted({key[0] for key in positive})
    joint_values = [
        positive[(graph_id, "entailment")]["correct"]
        and positive[(graph_id, "contradiction")]["correct"]
        for graph_id in unknown_graphs
        if (graph_id, "entailment") in positive
        and (graph_id, "contradiction") in positive
    ]
    unknown_joint = float(np.mean(joint_values))
    heldout = float(np.mean([row["correct"] for row in held]))
    eligible = (
        heldout >= 0.80
        and min(by_mode.values()) >= 0.65
        and min(by_channel.values()) >= 0.75
        and min(by_polarity.values()) >= 0.75
        and unknown_joint >= 0.65
    )
    headline = {
        "status": "binary_decomposed_graph_behavior_closed",
        **metrics,
        "heldout_accuracy": heldout,
        "mode_accuracy": by_mode,
        "channel_accuracy": by_channel,
        "polarity_accuracy": by_polarity,
        "unknown_joint_accuracy": unknown_joint,
        "graph_field_eligible": eligible,
        "execution_recovery": "legacy cell field mapped to frozen mode",
        "strict_interpretation": (
            "This interface tests separate entailment and contradiction judgments; "
            "failure would not negate graph reasoning in general."
        ),
    }
    campaign.save(
        out / "audit/execution_schema_recovery.json",
        {
            "failure": "shared binary behavior runner required legacy cell field",
            "mapping": "cell := frozen mode",
            "semantic_contract_changed": False,
            "thresholds_changed": False,
        },
    )
    campaign.close(
        "C422",
        headline,
        {
            "rows": len(rows) == 3584,
            "semantic_balance": sum(
                row["correct_answer"] == "Yes" for row in rows
            )
            == len(rows) // 2,
            "finite": campaign.finite(headline),
            "no_hidden": not (out / "raw/role_states.float16.npy").exists(),
        },
        "C423_graph_field",
    )


if __name__ == "__main__":
    main()
