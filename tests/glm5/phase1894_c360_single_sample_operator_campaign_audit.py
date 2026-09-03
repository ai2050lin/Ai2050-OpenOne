#!/usr/bin/env python3
"""Independent artifact and claim audit for C336-C360."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
PRODUCER = ROOT / "tests/glm5/phase1870_c336_c360_single_sample_operator_campaign.py"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    producer_hash = hashlib.sha256(PRODUCER.read_bytes()).hexdigest()
    visual = load(ROOT / "frontend/public/vis_data/research_kernel/c360_single_sample_operator_field.json")
    directories = {}
    finals = {}
    protocols = {}
    for campaign in range(336, 361):
        phase = 1870 + campaign - 336
        matches = [p for p in RESULT.glob(f"phase{phase}_c{campaign}_*") if p.is_dir()]
        assert len(matches) == 1, (campaign, matches)
        directories[campaign] = matches[0]
        finals[campaign] = load(matches[0] / "analysis/final.json")
        protocols[campaign] = load(matches[0] / "protocol/preregistration.json")
    checks = {
        "all_25_campaigns_present": len(finals) == 25,
        "continuous_phases": [finals[c]["phase"] for c in finals] == list(range(1870, 1895)),
        "all_internal_audits_closed": all(v["all_checks_passed"] for v in finals.values()),
        "all_producer_hashes_match_frozen_runner": all(protocols[c]["producer_sha256"] == producer_hash for c in protocols),
        "c340_all_role_coordinates": list(np.load(directories[340] / "raw/role_states.float16.npy", mmap_mode="r").shape) == [1272, 38, 6, 2560],
        "c340_all_token_holdout": list(np.load(directories[340] / "raw/full_fields_holdout.float16.npy", mmap_mode="r").shape) == [124, 38, 192, 2560],
        "c341_full_operator": list(np.load(directories[341] / "analysis/operators.float32.npy", mmap_mode="r").shape) == [3, 2, 38, 6, 2560],
        "c341_full_predictions": list(np.load(directories[341] / "raw/confirmation_predicted_responses.float16.npy", mmap_mode="r").shape) == [24, 3, 38, 6, 2560],
        "c347_finite": bool(np.isfinite(np.load(directories[347] / "analysis/control_checkpoint_role_dispersion.float32.npy")).all()),
        "graph_ineligibility_respected": finals[349]["headline"]["graph_behavior_eligible"] is False and finals[350]["headline"]["status"].endswith("not_run_ineligible"),
        "mediation_ineligibility_respected": finals[357]["headline"]["causal_claim"] is False,
        "bisimulation_not_overclaimed": finals[359]["headline"]["functional_bisimulation_established"] is False,
        "new_math_not_overclaimed": finals[360]["headline"]["new_math_gate_passed"] is False,
        "visual_full_coordinate_axis": (
            visual["schema"] == "c360_single_sample_operator_field.v1"
            and len(visual["dimensions"]) == 2560
            and all(len(row["values"]) == 2560 for row in visual["rows"])
        ),
    }
    payload = {
        "status": "independent_audit_complete",
        "producer_sha256": producer_hash,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "claim_audit": {
            "single_sample": "not established for A/B/I jointly",
            "composition": "offline numerical rollout beats frozen discovery means on this lockbox",
            "graph": "behavior-ineligible; no recursive hidden-state result",
            "mediation": "not run because the prospective coalition was ineligible",
            "cross_model": "coarse abstract response candidate only",
            "new_math": "not authorized",
        },
    }
    target = directories[360] / "audit/independent_audit.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
