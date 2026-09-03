#!/usr/bin/env python3
"""Phase1505: Qwen3 behavior and 4-case composition stratification for C087."""
from __future__ import annotations

import inspect
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
CONTRACT = TESTS / "result/phase1504_c087_cross_root_semeval_contract"
OUT = TESTS / "result/phase1505_c087_behavior_stratification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1391_c062_family_factorized_behavior as runner
import phase1457_c077_behavior as metric
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

BATCH = 24


def balanced_accuracy(rows):
    return sum(
        metric.accuracy([row for row in rows if row["semantic_match"] == truth])
        for truth in (True, False)
    ) / 2.0


def main():
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1505 exists")
    contract_final = core.load(CONTRACT / "analysis/final.json")
    contract_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if (
        contract_final["authorization"] != "run_phase1505_c087_behavior_stratification"
        or not contract_audit["all_checks_passed"]
    ):
        raise RuntimeError("Phase1504 authorization missing")
    source = core.rows(CONTRACT / "material/active_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    groups = core.rows(CONTRACT / "material/composition_sets.jsonl")
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        predictions, first = [], None
        for start in range(0, len(compiled), BATCH):
            block = runner.forward(model, compiled[start : start + BATCH], pad, device, supports)
            if first is None:
                first = block
            predictions.extend(block)
        repeat_block = runner.forward(model, compiled[:BATCH], pad, device, supports)
        repeat = max(
            abs(a["scores"][i] - b["scores"][i])
            for a, b in zip(first, repeat_block)
            for i in range(2)
        )
        rows = [
            {**case, **prediction, "correct": prediction["prediction"] == case["gold_position"]}
            for case, prediction in zip(source, predictions)
        ]
        by = {row["case_id"]: row for row in rows}
        keys = tuple(f"{surface}_{label}" for surface in protocol["surfaces"] for label in ("same", "different"))
        stratified = []
        for group in groups:
            correct_count = sum(by[group[key]]["correct"] for key in keys)
            stratum = "success" if correct_count == 4 else "failed" if correct_count == 0 else "mixed"
            stratified.append({**group, "correct_count": correct_count, "case_count": 4, "stratum": stratum})
        counts = Counter(row["stratum"] for row in stratified)
        summary = {
            "phase": 1505,
            "campaign": "C087",
            "global_accuracy": metric.accuracy(rows),
            "global_balanced_accuracy": balanced_accuracy(rows),
            "partition": {
                partition: {
                    "count": sum(row["partition"] == partition for row in rows),
                    "accuracy": metric.accuracy([row for row in rows if row["partition"] == partition]),
                    "balanced_accuracy": balanced_accuracy([row for row in rows if row["partition"] == partition]),
                }
                for partition in protocol["partitions"]
            },
            "surface": {
                surface: {
                    "accuracy": metric.accuracy([row for row in rows if row["surface"] == surface]),
                    "balanced_accuracy": balanced_accuracy([row for row in rows if row["surface"] == surface]),
                }
                for surface in protocol["surfaces"]
            },
            "truth": {
                str(truth).lower(): metric.accuracy([row for row in rows if row["semantic_match"] == truth])
                for truth in (True, False)
            },
            "stratum_counts": dict(counts),
            "stratum_partition_counts": {
                stratum: {
                    partition: sum(
                        row["stratum"] == stratum and row["partition"] == partition
                        for row in stratified
                    )
                    for partition in protocol["partitions"]
                }
                for stratum in ("success", "mixed", "failed")
            },
            "error_count": sum(not row["correct"] for row in rows),
            "error_items": dict(Counter(row["item"] for row in rows if not row["correct"])),
            "numeric_repeat_max_abs_diff": repeat,
            "runtime": {
                "placement": placement,
                "quantization": quant,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "hidden_state_accessed": False,
        }
        checks = {
            "count": len(rows) == 864,
            "sets": len(stratified) == 216 and sum(counts.values()) == 216,
            "repeat": repeat <= 1e-6,
            "finite": all(math.isfinite(value) for row in rows for value in row["scores"]),
            "bf16": quant["has_bf16_parameters"],
            "not_quantized": not quant["has_quantized_modules"],
            "hidden_not_accessed": True,
        }
        summary["checks"] = checks
        summary["all_integrity_checks_passed"] = all(checks.values())
        if not summary["all_integrity_checks_passed"]:
            raise RuntimeError(checks)
        core.write_rows(OUT / "raw/behavior.jsonl", rows)
        core.write_rows(OUT / "material/stratified_composition_sets.jsonl", stratified)
        core.save(OUT / "analysis/behavior_stratification_summary.json", summary)
        core.save(
            OUT / "analysis/final.json",
            {
                "phase": 1505,
                "campaign": "C087",
                "status": "behavior_stratification_complete",
                "stratum_counts": summary["stratum_counts"],
                "authorization": "run_phase1506_c087_all_case_field_capture",
            },
        )
        print(json.dumps({key: value for key, value in summary.items() if key != "runtime"}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
