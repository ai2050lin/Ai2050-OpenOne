#!/usr/bin/env python3
"""Post-reveal split accounting for Phase1379; changes no frozen gate."""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1379_c059_coordinate_group_evaluation"


def metrics(rows: list[dict]) -> dict:
    suff_adv = [r["suff_gain"]["correct"] - max(r["suff_gain"][d]
                                                for d in ("wrong", "status", "random")) for r in rows]
    reverse_adv = [r["reverse_damage"]["correct"] - r["reverse_damage"]["status"] for r in rows]
    fractions = [r["suff_gain"]["correct"] / r["whole_effect"]
                 for r in rows if abs(r["whole_effect"]) > 1e-12]
    return {
        "count": len(rows),
        "suff_gain_median": statistics.median(r["suff_gain"]["correct"] for r in rows),
        "suff_advantage_median": statistics.median(suff_adv),
        "suff_win_fraction": sum(v > 0 for v in suff_adv) / len(suff_adv),
        "whole_effect_fraction_median": statistics.median(fractions),
        "reverse_damage_median": statistics.median(r["reverse_damage"]["correct"] for r in rows),
        "reverse_over_status_median": statistics.median(reverse_adv),
        "reverse_over_status_win_fraction": sum(v > 0 for v in reverse_adv) / len(reverse_adv),
    }


def main() -> None:
    records = core.rows(OUT / "raw/qwen3_coordinate_groups.jsonl")
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    result = {}
    for route in manifest["routes"]:
        result[route] = {}
        for size in manifest["sizes"]:
            result[route][str(size)] = {
                partition: metrics([r for r in records if r["route"] == route and r["size"] == size and
                                    r["partition"] == partition])
                for partition in manifest["gate"]["evaluation_partitions"]
            }
    artifact = {
        "phase": 1379, "campaign": "C059", "postprocessing_only": True,
        "thresholds_or_eligibility_changed": False,
        "candidate_multiplicity_note": "four routes times nine sizes were frozen; isolated qualification is not a named mechanism",
        "split_metrics": result, "record_count": len(records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/split_stability_postanalysis.json", artifact)
    print(json.dumps({"phase": 1379, "record_count": len(records),
                      "postprocessing_only": True}, indent=2))


if __name__ == "__main__":
    main()
