#!/usr/bin/env python3
"""Phase1410: preregister-free execution of the fixed C066 state-16 prediction."""
from __future__ import annotations

import inspect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from phase1406_c065_holdout_factorial_swaps import ARMS, group_metrics, run_case

PHASE, CAMPAIGN = 1410, "C066"
CONTRACT = TESTS / "result/phase1408_c066_midstate_breadth_contract"
BEHAVIOR = TESTS / "result/phase1409_c066_behavior"
OUT = TESTS / "result/phase1410_c066_state16_factorial_replication"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1410 exists")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if behavior_final["authorization"] != "run_phase1410_c066_state16_factorial_replication" or not behavior_audit["all_checks_passed"]:
        raise RuntimeError("behavior did not authorize")
    selected = core.rows(BEHAVIOR / "material/eligible_factor_sets.jsonl")
    holdouts = [r for r in selected if r["partition"] in ("confirmation", "lockbox")]
    compiled = {r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    candidates = [{
        "candidate_id": f"{candidate['surface']}:{candidate['object']}:s16",
        "surface": candidate["surface"],
        "object": candidate["object"],
        "window_index": 1,
        "state_index": protocol["mechanism"]["state_index"],
        "role": candidate["role"],
    } for candidate in protocol["mechanism"]["candidates"]]
    gate = protocol["mechanism"]
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case in holdouts:
            case_candidates = [c for c in candidates if c["surface"] == case["surface"]]
            if len(case_candidates) != 2:
                raise RuntimeError("surface candidate count")
            records.extend(run_case(model, pad, device, supports, case, compiled, case_candidates))
        core.write_rows(OUT / "raw/state16_factorial_swaps.jsonl", records)
        candidate_summary = {}
        for candidate in candidates:
            route = [r for r in records if r["candidate_id"] == candidate["candidate_id"]]
            split_metrics = {
                split: group_metrics([r for r in route if r["partition"] == split], candidate["object"], gate)
                for split in ("confirmation", "lockbox")
            }
            family_metrics = {}
            qualified_families = []
            for family in behavior_final["qualified_families"]:
                values = {
                    split: group_metrics([r for r in route if r["partition"] == split and r["family"] == family], candidate["object"], gate)
                    for split in ("confirmation", "lockbox")
                }
                family_metrics[family] = values
                if all(v["qualified"] for v in values.values()):
                    qualified_families.append(family)
            qualified = all(v["qualified"] for v in split_metrics.values()) and len(qualified_families) >= gate["minimum_family_breadth"]
            candidate_summary[candidate["candidate_id"]] = {
                "candidate": candidate,
                "split_metrics": split_metrics,
                "family_metrics": family_metrics,
                "qualified_families": qualified_families,
                "qualified": qualified,
            }
        routes = {}
        for object_name in ("family_identity", "joint_polarity"):
            qualified_candidates = [cid for cid, result in candidate_summary.items() if result["candidate"]["object"] == object_name and result["qualified"]]
            routes[object_name] = {"qualified_candidates": qualified_candidates, "confirmed": bool(qualified_candidates)}
        checks = {
            "holdout_sets": len(holdouts) == 120,
            "split_balance": sum(r["partition"] == "confirmation" for r in holdouts) == 60 and sum(r["partition"] == "lockbox" for r in holdouts) == 60,
            "record_count": len(records) == 120 * 2 * len(ARMS),
            "state16_only": {r["state_index"] for r in records} == {16},
            "holdout_only": {r["partition"] for r in records} == {"confirmation", "lockbox"},
            "candidate_count": len(candidate_summary) == 6,
            "self_identity": max(abs(r["signed_damage"]) for r in records if r["arm"] == "self") <= gate["self_max_abs_diff"],
            "finite": all(math.isfinite(r[k]) for r in records for k in ("baseline_margin", "swap_margin", "signed_damage", "loss_fraction")),
            "bf16": quant["has_bf16_parameters"],
            "not_quantized": not quant["has_quantized_modules"],
        }
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "holdout_set_count": len(holdouts),
            "record_count": len(records),
            "candidate_summary": candidate_summary,
            "route_status": routes,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "contract_sha256": protocol["contract_sha256"],
            "behavior_sha256": core.sha(BEHAVIOR / "raw/active_behavior.jsonl"),
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/state16_replication_summary.json", summary)
        core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_checks_passed": summary["all_checks_passed"], "route_status": routes, "authorization": "run_phase1411_c066_campaign_closure"})
        print(json.dumps({k: v for k, v in summary.items() if k != "candidate_summary"}, indent=2))
        print(json.dumps({cid: {
            "qualified": result["qualified"],
            "qualified_families": result["qualified_families"],
            "split": {split: metrics for split, metrics in result["split_metrics"].items()},
        } for cid, result in candidate_summary.items()}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
