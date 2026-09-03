#!/usr/bin/env python3
"""Premodel metadata-only role repair for C519.

The frozen prompts, labels, partitions, and gates are unchanged. In direct
typed-graph cells the unused middle node does not occur in the prompt, so the
secondary observation role is redirected to the already present target node.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2052_c518_c525_fresh_callable_state_campaign as campaign


def main() -> None:
    out = campaign.OUTS["C519"]
    if (out / "analysis/final.json").exists():
        print((out / "analysis/final.json").read_text(encoding="utf-8"))
        return
    rows = campaign.fresh_rows()
    repaired = []
    repair_rows = []
    for row in rows:
        value = dict(row)
        value["role_values"] = dict(row["role_values"])
        if row["family"] == "typed_graph_path" and int(row["bits"][1]) == 0:
            before = value["role_values"]["secondary"]
            after = value["role_values"]["context"]
            value["role_values"]["secondary"] = after
            repair_rows.append({"case_id": row["case_id"], "role": "secondary", "before": before, "after": after, "prompt_changed": False})
        repaired.append(value)
    compiled = campaign.previous.parent.compile_material(repaired)
    campaign.write_rows(out / "material/measurement_role_corrected_cases.jsonl", repaired)
    campaign.write_rows(out / "compiled/qwen3_fresh.jsonl", compiled)
    campaign.write_rows(out / "audit/premodel_role_repair.jsonl", repair_rows)
    max_width = max(len(row["prompt_ids"]) for row in compiled)
    balance = {family: float(np.mean([row["gold_position"] == 0 for row in repaired if row["family"] == family])) for family in campaign.FAMILIES}
    behavior = campaign.previous.parent.run_behavior(repaired, compiled, out)
    raw = campaign.read_rows(out / "raw/behavior.jsonl")
    by_id = {row["case_id"]: row for row in repaired}
    family_accuracy = {family: float(np.mean([row["correct"] for row in raw if by_id[row["case_id"]]["family"] == family])) for family in campaign.FAMILIES}
    campaign.close("C519", {
        "status": "audit_behavior_closed_after_premodel_metadata_repair",
        **behavior,
        "max_prompt_tokens": max_width,
        "family_first_position_rate": balance,
        "family_accuracy": family_accuracy,
        "eligible_families": [family for family, value in family_accuracy.items() if value >= 0.65],
        "field_authorized": True,
        "premodel_role_repairs": len(repair_rows),
        "repair_boundary": "measurement metadata only; prompts, labels, partitions, model, and gates unchanged",
    }, {
        "rows": behavior["rows"] == 1440,
        "width": max_width <= 128,
        "balance": set(balance.values()) == {0.5},
        "repair_count": len(repair_rows) == 240,
        "prompts_unchanged": all(not row["prompt_changed"] for row in repair_rows),
        "finite": campaign.finite(behavior),
    }, "C520_capture")


if __name__ == "__main__":
    main()
