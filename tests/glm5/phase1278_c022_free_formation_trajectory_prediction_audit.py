from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1278_c022_free_formation_trajectory_prediction as phase  # noqa: E402


def preaudit() -> None:
    protocol, rows = phase.verify_protocol()
    checks = {
        "protocol_digest": protocol["protocol_digest"] == phase.instrument.digest({key: value for key, value in protocol.items() if key != "protocol_digest"}),
        "main_hash": protocol["source_hashes"]["main"] == phase.instrument.file_sha256(phase.MAIN),
        "auditor_hash": protocol["source_hashes"]["auditor"] == phase.instrument.file_sha256(phase.AUDITOR),
        "parent_final": phase.instrument.read_json(phase.P1277_FINAL)["formal_execution_authorized"],
        "parent_audit": phase.instrument.read_json(phase.P1277_AUDIT)["passed"],
        "material_rows": len(rows) == len(phase.base.TASKS) * phase.ROLE_PANEL_COUNT,
        "material_unique": len({row["row_id"] for row in rows}) == len(rows),
        "formal_models": len(phase.MODEL_SEEDS) == len(phase.CELLS) * phase.SEEDS_PER_CELL,
        "split_balance": len(phase.split_models("discovery")) == len(phase.split_models("confirmation")) == len(phase.MODEL_SEEDS) // 2,
        "new_seeds": not set(phase.MODEL_SEEDS.values()).intersection(phase.base.MODEL_SEEDS.values()),
        "cutoff_before_fixed": phase.THRESHOLDS["prediction_cutoff"] < phase.THRESHOLDS["fixed_budget"],
        "confirmation_absent": not phase.CONFIRMATION_SUMMARY.exists() and not list((phase.OUT / "raw/confirmation/models").glob("*.json")) if (phase.OUT / "raw/confirmation/models").exists() else True,
        "primary_features_frozen": tuple(protocol["baseline_features"]) == phase.BASELINE_FEATURES and tuple(protocol["internal_features"]) == phase.INTERNAL_FEATURES,
        "model_unit": "model trajectory" in phase.instrument.read_json(phase.instrument.PROTOCOL)["independent_unit"],
    }
    audit = {"phase": phase.PHASE, "mode": "pre", "checks": checks, "passed_count": sum(checks.values()), "total_count": len(checks), "passed": all(checks.values())}
    audit["audit_digest"] = phase.instrument.digest(audit)
    phase.instrument.atomic_json(phase.PREAUDIT, audit)
    print(json.dumps({"mode": "pre", "passed": audit["passed"], "checks": f"{audit['passed_count']}/{audit['total_count']}"}, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


def final_audit() -> None:
    protocol, rows = phase.verify_protocol()
    predictor = phase.instrument.read_json(phase.PREDICTOR)
    final = phase.instrument.read_json(phase.FINAL)
    discovery = phase.load_models("discovery")
    checks = {
        "preaudit": phase.instrument.read_json(phase.PREAUDIT)["passed"],
        "protocol_digest": protocol["protocol_digest"] == phase.instrument.digest({key: value for key, value in protocol.items() if key != "protocol_digest"}),
        "material_hash": protocol["material_hash"] == phase.instrument.digest([row["row_digest"] for row in rows]),
        "discovery_count": len(discovery) == len(phase.split_models("discovery")),
        "discovery_hashes": all(phase.instrument.file_sha256(phase.model_file("discovery", model["model_key"])) == phase.instrument.read_json(phase.DISCOVERY_SUMMARY)["model_hashes"][model["model_key"]] for model in discovery),
        "predictor_digest": predictor["predictor_digest"] == phase.instrument.digest({key: value for key, value in predictor.items() if key != "predictor_digest"}),
        "confirmation_absent_at_seal": predictor["confirmation_absent_at_seal"] and predictor["confirmation_model_count_at_seal"] == 0,
        "feature_names": predictor["baseline_feature_names"] == phase.features(discovery[0], False)[0] and predictor["augmented_feature_names"] == phase.features(discovery[0], True)[0],
        "final_digest": final["final_digest"] == phase.instrument.digest({key: value for key, value in final.items() if key != "final_digest"}),
        "decision_logic": final["causal_branch_authorized"] == final["passed"],
    }
    if predictor["confirmation_authorized"]:
        confirmation = phase.load_models("confirmation")
        recomputed = phase.score_predictions(confirmation, predictor)
        confirmation_object = phase.object_gate(confirmation, "confirmation")
        checks.update({
            "confirmation_count": len(confirmation) == len(phase.split_models("confirmation")),
            "confirmation_hashes": all(phase.instrument.file_sha256(phase.model_file("confirmation", model["model_key"])) == phase.instrument.read_json(phase.CONFIRMATION_SUMMARY)["model_hashes"][model["model_key"]] for model in confirmation),
            "confirmation_after_seal": phase.instrument.read_json(phase.CONFIRMATION_SUMMARY)["started_at_utc"] > predictor["created_at_utc"],
            "object_recompute": confirmation_object == final["confirmation_object_gate"],
            "mae_recompute": abs(recomputed["augmented_mae"] - final["scores"]["augmented_mae"]) < 1.0e-12,
            "increment_recompute": abs(recomputed["relative_mae_improvement"] - final["scores"]["relative_mae_improvement"]) < 1.0e-12,
            "order_recompute": recomputed["augmented_pair_order"] == final["scores"]["augmented_pair_order"],
            "all_models_finite": all(model["all_finite"] for model in discovery + confirmation),
            "state_parent_saved": all(any(checkpoint["step"] == phase.THRESHOLDS["prediction_cutoff"] for checkpoint in model["checkpoints"]) for model in discovery + confirmation),
            "outcome_recompute": all(model["fixed_event_step"] == phase.stable_event_step(model["trajectory"], phase.THRESHOLDS["fixed_budget"]) for model in discovery + confirmation),
            "checkpoint_hashes": all(phase.instrument.file_sha256(ROOT / checkpoint["path"]) == checkpoint["sha256"] for model in discovery + confirmation for checkpoint in model["checkpoints"]),
        })
    audit = {"phase": phase.PHASE, "mode": "final", "checks": checks, "passed_count": sum(checks.values()), "total_count": len(checks), "passed": all(checks.values())}
    audit["audit_digest"] = phase.instrument.digest(audit)
    phase.instrument.atomic_json(phase.AUDIT, audit)
    print(json.dumps({"mode": "final", "passed": audit["passed"], "checks": f"{audit['passed_count']}/{audit['total_count']}"}, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "final"
    if mode == "pre":
        preaudit()
    elif mode == "final":
        final_audit()
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
