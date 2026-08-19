from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1277_c022_formation_dynamics_instrument_calibration as phase  # noqa: E402


def main() -> None:
    protocol = phase.read_json(phase.PROTOCOL)
    final = phase.read_json(phase.FINAL)
    replay = phase.read_json(phase.REPLAY)
    rows = phase.read_jsonl(phase.CALIBRATION)
    recomputed = phase.calibration_metrics(rows)
    checks = {
        "protocol_digest": protocol["protocol_digest"] == phase.digest({key: value for key, value in protocol.items() if key != "protocol_digest"}),
        "source_main": protocol["source_hashes"]["main"] == phase.file_sha256(phase.MAIN),
        "source_auditor": protocol["source_hashes"]["auditor"] == phase.file_sha256(phase.AUDITOR),
        "calibration_hash": final["calibration_hash"] == phase.file_sha256(phase.CALIBRATION),
        "replay_hash": final["replay_hash"] == phase.file_sha256(phase.REPLAY),
        "row_count": len(rows) == len(phase.CELLS) * phase.SEEDS_PER_CELL,
        "split_count": sum(row["split"] == "discovery" for row in rows) == len(rows) // 2,
        "model_level_units": len({row["trajectory_id"] for row in rows}) == len(rows),
        "synthetic_recompute": abs(recomputed["augmented_relative_improvement"] - final["metrics"]["augmented_relative_improvement"]) < 1.0e-12,
        "negative_recompute": abs(recomputed["nuisance_relative_improvement"] - final["metrics"]["nuisance_relative_improvement"]) < 1.0e-12,
        "future_sentinel_blocked": final["metrics"]["future_sentinel_blocked"],
        "deterministic_state": replay["state_digest_equal"],
        "deterministic_losses": replay["max_loss_abs_diff"] == 0.0,
        "deterministic_accuracy": replay["accuracy_abs_diff"] == 0.0,
        "new_formal_seeds": not set(phase.MODEL_SEEDS.values()).intersection(phase.base.MODEL_SEEDS.values()),
        "all_gates": all(final["gates"].values()),
        "final_digest": final["final_digest"] == phase.digest({key: value for key, value in final.items() if key != "final_digest"}),
        "no_scientific_claim": final["scientific_mechanism_claim"] is False,
    }
    audit = {
        "phase": phase.PHASE,
        "checks": checks,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "decision_match": final["passed"] == all(final["gates"].values()),
    }
    audit["audit_digest"] = phase.digest(audit)
    phase.atomic_json(phase.AUDIT, audit)
    print(json.dumps({"passed": audit["passed"], "checks": f"{audit['passed_count']}/{audit['total_count']}"}, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
