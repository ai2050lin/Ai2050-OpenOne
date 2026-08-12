from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

import phase1148_mandatory_mediation_calibration as p1148
import phase1150_role_factorized_independent_replication as p1150


PHASE = 1150
ROOT = Path(__file__).resolve().parents[2]


def without_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(key)
    return result


def main() -> None:
    prereg = p1148.read_json(p1150.PREREG_PATH)
    p1150.verify_preregistration(prereg)
    checks: dict[str, bool] = {}

    protocol_audit = p1148.read_json(p1150.OUT_ROOT / "protocol" / "audit.json")
    checks["protocol_audit_passed"] = bool(protocol_audit["all_checks_passed"])
    checks["protocol_audit_digest"] = p1148.canonical_digest(
        without_digest(protocol_audit, "audit_digest")
    ) == protocol_audit["audit_digest"]

    for split in ("discovery", "confirmation"):
        selection_path = p1150.OUT_ROOT / "analysis" / f"{split}_selection.json"
        if not selection_path.exists():
            checks[f"{split}_absence_authorized"] = split == "confirmation" and not p1148.read_json(
                p1150.OUT_ROOT / "analysis" / "discovery_selection.json"
            )["confirmation_authorized"]
            continue

        paired: dict[str, dict[str, Any]] = {}
        for replicate in p1150.split_replicates(prereg, split):
            paired[replicate] = {}
            spec = prereg["replicates"][replicate]
            expected_material = p1150.realized_material_digests(spec, prereg)
            for condition in p1150.CONDITION_ORDER:
                summary = p1150.load_summary(replicate, condition, prereg)
                prefix = f"{split}.{replicate}.{condition}"
                checks[f"{prefix}.summary_digest"] = p1148.canonical_digest(
                    without_digest(summary, "summary_digest")
                ) == summary["summary_digest"]
                checks[f"{prefix}.model_hash"] = p1148.file_sha256(
                    ROOT / summary["model_path"]
                ) == summary["model_sha256"]
                checks[f"{prefix}.predictions_hash"] = p1148.file_sha256(
                    ROOT / summary["predictions_path"]
                ) == summary["predictions_sha256"]
                checks[f"{prefix}.training_material"] = (
                    summary["training_dataset_digest"] == expected_material["training"]
                    and summary["batch_schedule_digest"] == expected_material["schedule"]
                )
                checks[f"{prefix}.evaluation_material"] = summary[
                    "evaluation_dataset_digests"
                ] == {
                    key: expected_material[key] for key in ("seen", "holdout", "quartet")
                }
                checks[f"{prefix}.finite_metrics"] = all(
                    bool(torch.isfinite(torch.tensor(float(metrics[metric]))))
                    for metrics in summary["evaluation"].values()
                    for metric in (
                        "accuracy",
                        "row_address_accuracy",
                        "column_address_accuracy",
                        "joint_address_accuracy",
                    )
                )

                model = p1150.load_model(summary)
                evaluation, _, digests = p1148.evaluate_bundle(
                    model, "soft_EF", spec, prereg, "formal", False
                )
                checks[f"{prefix}.evaluation_recomputed"] = (
                    p1148.canonical_digest(evaluation)
                    == p1148.canonical_digest(summary["evaluation"])
                    and digests == summary["evaluation_dataset_digests"]
                )
                answer_checks = p1148.answer_gate(evaluation, prereg["thresholds"])
                address_checks = p1148.address_gate(evaluation, "soft_EF", prereg["thresholds"])
                checks[f"{prefix}.gate_recomputed"] = (
                    answer_checks == summary["answer_gate_checks"]
                    and address_checks == summary["address_gate_checks"]
                    and summary["qualified"]
                    == (all(answer_checks.values()) and all(address_checks.values()))
                )
                paired[replicate][condition] = summary
                del model
                torch.cuda.empty_cache()

            left = paired[replicate]["answer_boundary"]
            right = paired[replicate]["role_factorized"]
            checks[f"{split}.{replicate}.paired_initialization"] = (
                left["initial_state_digest"] == right["initial_state_digest"]
            )
            checks[f"{split}.{replicate}.paired_training_data"] = (
                left["training_dataset_digest"] == right["training_dataset_digest"]
            )
            checks[f"{split}.{replicate}.paired_schedule"] = (
                left["batch_schedule_digest"] == right["batch_schedule_digest"]
            )
            checks[f"{split}.{replicate}.equal_parameters"] = (
                left["parameter_count"] == right["parameter_count"]
            )

        stored_selection = p1148.read_json(selection_path)
        recomputed_selection = p1150.build_split_analysis(split, prereg)
        checks[f"{split}.selection_recomputed"] = stored_selection == recomputed_selection
        checks[f"{split}.selection_digest"] = p1148.canonical_digest(
            without_digest(stored_selection, "selection_digest")
        ) == stored_selection["selection_digest"]

    discovery = p1148.read_json(p1150.OUT_ROOT / "analysis" / "discovery_selection.json")
    confirmation_path = p1150.OUT_ROOT / "analysis" / "confirmation_selection.json"
    checks["confirmation_execution_matches_authorization"] = (
        confirmation_path.exists() == bool(discovery["confirmation_authorized"])
    )

    final_path = p1150.OUT_ROOT / "analysis" / "final.json"
    if final_path.exists():
        final = p1148.read_json(final_path)
        checks["final_digest"] = p1148.canonical_digest(
            without_digest(final, "final_digest")
        ) == final["final_digest"]
        checks["historical_claim_preserved"] = bool(final["historical_phase1149_claim_unchanged"])

    audit = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "audit_script_sha256": p1148.file_sha256(Path(__file__).resolve()),
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    audit["audit_digest"] = p1148.canonical_digest(audit)
    p1148.write_json(p1150.OUT_ROOT / "audit" / "independent_recomputation.json", audit)
    if not audit["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase1150 audit failed: {failed}")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
