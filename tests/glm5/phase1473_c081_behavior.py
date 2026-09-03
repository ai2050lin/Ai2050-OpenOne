#!/usr/bin/env python3
"""Phase1473: behavior qualification for the C081 rescue."""
from __future__ import annotations

import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1391_c062_family_factorized_behavior as runner
import phase1470_c080_explicit_behavior as evaluator
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

CONTRACT = TESTS / "result/phase1472_c081_validated_interface_contract"
OUT = TESTS / "result/phase1473_c081_behavior"
BATCH = 32


def adapted(protocol: dict) -> dict:
    return {
        **protocol,
        "branches": {"explicit": {"surfaces": protocol["surfaces"], "behavior": protocol["behavior"]}},
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1473 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1473_c081_behavior" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1472 did not authorize Phase1473")
    source = core.rows(CONTRACT / "material/active_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    sets = core.rows(CONTRACT / "material/interaction_sets.jsonl")
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        predictions, first = [], None
        for start in range(0, len(compiled), BATCH):
            block = runner.forward(model, compiled[start:start + BATCH], pad, device, supports)
            if first is None:
                first = block
            predictions.extend(block)
        repeated = runner.forward(model, compiled[:BATCH], pad, device, supports)
        repeat_error = max(abs(left["scores"][index] - right["scores"][index]) for left, right in zip(first, repeated) for index in range(2))
        rows = [{**row, **prediction, "correct": prediction["prediction"] == row["gold_position"]} for row, prediction in zip(source, predictions)]
        summary, eligible = evaluator.evaluate(rows, sets, adapted(protocol), repeat_error, quant)
        summary.update({"phase": 1473, "campaign": "C081", "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()}})
        core.write_rows(OUT / "raw/behavior.jsonl", rows)
        core.write_rows(OUT / "material/eligible_interaction_sets.jsonl", eligible)
        core.save(OUT / "analysis/behavior_summary.json", summary)
        authorization = "run_phase1474_c081_discovery_capture" if summary["behavior_qualified"] else "close_c081_and_explicit_interaction_route_at_behavior_gate"
        core.save(OUT / "analysis/final.json", {"phase": 1473, "campaign": "C081", "behavior_qualified": summary["behavior_qualified"], "eligible_count": len(eligible), "authorization": authorization})
        print(json.dumps({key: value for key, value in summary.items() if key not in ("equal_label_surface", "unequal_pair_surface", "runtime")}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
