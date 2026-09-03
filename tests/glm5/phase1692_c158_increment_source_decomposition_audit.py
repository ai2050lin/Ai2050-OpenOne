#!/usr/bin/env python3
"""Independent recomputation audit for C158."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1692_c158_increment_source_decomposition"


def load(path):
    return json.loads((OUT / path).read_text(encoding="utf-8"))


def main():
    modes = load("material/modes.json")
    index = [json.loads(line) for line in (OUT / "material/intervention_index.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = np.load(OUT / "raw/intervention_candidate_logits.float32.npy", mmap_mode="r")
    result = load("analysis/decomposition.json")
    donor = np.asarray([row["donor_gold_position"] for row in index], np.int64)
    margins = np.asarray([[value[i, donor[i]] - value[i, 1 - donor[i]] for i in range(128)] for value in scores])
    gains = margins - margins[0]
    names = [mode["name"] for mode in modes]
    recomputed = {name: float(np.mean(gains[names.index(name)])) for name in ("rms_observed_x_a1_q32", "rms_predicted_y_a1_q32", "rms_predicted_sum_a1_q32", "rms_exact_target_a1_q32")}
    checks = {
        "shape": list(scores.shape) == [26, 128, 2],
        "finite": bool(np.isfinite(scores).all()),
        "modes": len(modes) == 26 and len(set(names)) == 26,
        "x_gain": abs(recomputed["rms_observed_x_a1_q32"] - result["classifications"]["observed_x"]["mean_gain"]) < 1e-5,
        "y_gain": abs(recomputed["rms_predicted_y_a1_q32"] - result["classifications"]["predicted_y"]["mean_gain"]) < 1e-5,
        "sum_gain": abs(recomputed["rms_predicted_sum_a1_q32"] - result["classifications"]["predicted_sum"]["mean_gain"]) < 1e-5,
        "exact_gain": abs(recomputed["rms_exact_target_a1_q32"] - result["classifications"]["exact_target"]["mean_gain"]) < 1e-5,
        "incident_honest": load("audit/execution_incident_and_recovery.json")["model_rerun"] is False,
        "scope": "no formation-layer" in result["claim_boundary"],
    }
    report = {"phase": 1692, "campaign": "C158", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "recomputed": recomputed, "authorization": "memo_then_C159"}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
