from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import phase1149_role_factorized_mediation as p1149


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1149_role_factorized_mediation"


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def audit_split(split: str) -> dict[str, Any]:
    prereg = p1149.p1148.read_json(p1149.PREREG_PATH)
    p1149.verify_preregistration(prereg)
    protocol_audit = p1149.p1148.read_json(OUT_ROOT / "protocol" / "audit.json")
    replicates = p1149.split_replicates(prereg, split)
    checks: dict[str, bool] = {
        "protocol_audit": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest": protocol_audit["protocol_digest"] == prereg["protocol_digest"],
    }
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for replicate in replicates:
        summaries[replicate] = {}
        for condition in p1149.CONDITION_ORDER:
            summary = p1149.load_summary(replicate, condition, prereg)
            summaries[replicate][condition] = summary
            prefix = f"{replicate}.{condition}"
            checks[f"{prefix}.model_hash"] = p1149.p1148.file_sha256(
                ROOT / summary["model_path"]
            ) == summary["model_sha256"]
            checks[f"{prefix}.prediction_hash"] = p1149.p1148.file_sha256(
                ROOT / summary["predictions_path"]
            ) == summary["predictions_sha256"]
            checks[f"{prefix}.fixed_steps"] = summary["training_steps"] == prereg[
                "replicates"
            ][replicate]["training"]["max_steps"]
            checks[f"{prefix}.finite"] = summary["nonfinite_steps"] == 0
            for point in summary["trajectory"]:
                checks[f"{prefix}.trajectory.{point['step']}.hash"] = (
                    p1149.p1148.file_sha256(ROOT / point["checkpoint"]["path"])
                    == point["checkpoint"]["sha256"]
                )
            model = p1149.load_model(summary)
            evaluation, _, digests = p1149.p1148.evaluate_bundle(
                model, "soft_EF", prereg["replicates"][replicate], prereg, "formal", False
            )
            metric_checks = []
            for evaluation_split in ("seen", "holdout", "quartet"):
                metric_checks.append(
                    digests[evaluation_split]
                    == summary["evaluation_dataset_digests"][evaluation_split]
                )
                for key in (
                    "accuracy",
                    "minimum_field_accuracy",
                    "minimum_entity_accuracy",
                    "row_address_accuracy",
                    "column_address_accuracy",
                    "joint_address_accuracy",
                    "oracle_accuracy",
                ):
                    metric_checks.append(
                        close(
                            evaluation[evaluation_split][key],
                            summary["evaluation"][evaluation_split][key],
                        )
                    )
            checks[f"{prefix}.metrics_recomputed"] = all(metric_checks)
            answer = p1149.p1148.answer_gate(evaluation, prereg["thresholds"])
            address = p1149.p1148.address_gate(evaluation, "soft_EF", prereg["thresholds"])
            checks[f"{prefix}.gates"] = (
                answer == summary["answer_gate_checks"]
                and address == summary["address_gate_checks"]
                and summary["qualified"] == (all(answer.values()) and all(address.values()))
            )
            del model
            torch.cuda.empty_cache()
        checks[f"{replicate}.paired_initial"] = len(
            {summaries[replicate][condition]["initial_state_digest"] for condition in p1149.CONDITION_ORDER}
        ) == 1
        checks[f"{replicate}.paired_data"] = len(
            {summaries[replicate][condition]["training_dataset_digest"] for condition in p1149.CONDITION_ORDER}
        ) == 1
        checks[f"{replicate}.paired_schedule"] = len(
            {summaries[replicate][condition]["batch_schedule_digest"] for condition in p1149.CONDITION_ORDER}
        ) == 1
        checks[f"{replicate}.equal_parameters"] = len(
            {summaries[replicate][condition]["parameter_count"] for condition in p1149.CONDITION_ORDER}
        ) == 1

    official = p1149.p1148.read_json(OUT_ROOT / "analysis" / f"{split}_selection.json")
    all_qualified = {
        condition: all(summaries[replicate][condition]["qualified"] for replicate in replicates)
        for condition in p1149.CONDITION_ORDER
    }
    gain_checks = []
    for replicate in replicates:
        for evaluation_split in ("seen", "holdout", "quartet"):
            baseline = summaries[replicate]["answer_boundary"]["evaluation"][evaluation_split]["accuracy"]
            factorized = summaries[replicate]["role_factorized"]["evaluation"][evaluation_split]["accuracy"]
            gain = factorized - baseline
            observed = official["effects"][replicate][evaluation_split]
            checks[f"selection.{replicate}.{evaluation_split}.gain"] = close(
                gain, observed["paired_gain"]
            )
            checks[f"selection.{replicate}.{evaluation_split}.gate"] = (
                observed["gain_gate"]
                == (gain >= prereg["thresholds"]["minimum_paired_accuracy_gain"])
            )
            if evaluation_split in ("holdout", "quartet"):
                gain_checks.append(observed["gain_gate"])
    gain_scope = all(gain_checks)
    expected_selected = "role_factorized" if all_qualified["role_factorized"] and gain_scope else None
    if split == "confirmation":
        discovery = p1149.p1148.read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
        expected_selected = discovery["selected_condition"]
    selected_qualified = bool(
        expected_selected and all_qualified[str(expected_selected)] and gain_scope
    )
    checks["selection.qualified"] = official["condition_all_qualified"] == all_qualified
    checks["selection.gain_scope"] = official["gain_scope_pass"] == gain_scope
    checks["selection.selected"] = official["selected_condition"] == expected_selected
    key = "confirmation_authorized" if split == "discovery" else "causal_validation_authorized"
    checks["selection.authorization"] = official[key] == selected_qualified
    result = {
        "phase": p1149.PHASE,
        "split": split,
        "protocol_digest": prereg["protocol_digest"],
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
    }
    result["audit_digest"] = p1149.p1148.canonical_digest(result)
    p1149.p1148.write_json(
        OUT_ROOT / "audit" / f"{split}_independent_result_audit.json", result
    )
    if not result["all_checks_passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Phase1149 {split} audit failed: {failed}")
    return result


def audit_causal() -> dict[str, Any]:
    prereg = p1149.p1148.read_json(p1149.PREREG_PATH)
    p1149.verify_preregistration(prereg)
    official = p1149.p1148.read_json(OUT_ROOT / "analysis" / "causal_position_validation.json")
    confirmation = p1149.p1148.read_json(OUT_ROOT / "analysis" / "confirmation_selection.json")
    condition = str(confirmation["selected_condition"])
    checks: dict[str, bool] = {
        "protocol_digest": official["protocol_digest"] == prereg["protocol_digest"],
        "selected_condition": official["selected_condition"] == condition,
    }
    all_passes = []
    for replicate in p1149.split_replicates(prereg, "confirmation"):
        spec = prereg["replicates"][replicate]
        model = p1149.load_model(p1149.load_summary(replicate, condition, prereg))
        datasets, _ = p1149.p1148.build_evaluation_sets(spec, prereg, "formal")
        split_passes = []
        for split_name, dataset in datasets:
            if split_name == "seen":
                continue
            metrics = {
                mode: p1149.intervention_accuracy(
                    model, dataset, mode, int(spec["training"]["evaluation_batch_size"])
                )
                for mode in (
                    "normal",
                    "both_answer",
                    "row_answer",
                    "column_answer",
                    "swapped",
                    "oracle_both",
                )
            }
            metrics["both_answer_drop"] = metrics["normal"] - metrics["both_answer"]
            metrics["row_answer_drop"] = metrics["normal"] - metrics["row_answer"]
            metrics["column_answer_drop"] = metrics["normal"] - metrics["column_answer"]
            metrics["swapped_drop"] = metrics["normal"] - metrics["swapped"]
            expected_gate = {
                "base_behavior": metrics["normal"] >= prereg["thresholds"]["holdout_accuracy"],
                "both_answer_necessity": metrics["both_answer_drop"]
                >= prereg["thresholds"]["position_ablation_drop"],
                "row_role_necessity": metrics["row_answer_drop"]
                >= prereg["thresholds"]["single_role_ablation_drop"],
                "column_role_necessity": metrics["column_answer_drop"]
                >= prereg["thresholds"]["single_role_ablation_drop"],
                "swap_specificity": metrics["swapped_drop"]
                >= prereg["thresholds"]["position_ablation_drop"],
                "oracle_rescue": metrics["oracle_both"]
                >= prereg["thresholds"]["oracle_rescue_accuracy"],
            }
            observed = official["per_replicate"][replicate]["splits"][split_name]
            for key, value in metrics.items():
                checks[f"{replicate}.{split_name}.{key}"] = close(value, observed["metrics"][key])
            checks[f"{replicate}.{split_name}.dataset"] = (
                p1149.p1148.dataset_digest(dataset) == observed["dataset_digest"]
            )
            checks[f"{replicate}.{split_name}.gate"] = expected_gate == observed["gate"]
            split_passes.append(all(expected_gate.values()))
        expected_replicate = all(split_passes)
        checks[f"{replicate}.passed"] = (
            official["per_replicate"][replicate]["passed"] == expected_replicate
        )
        all_passes.append(expected_replicate)
        del model
        torch.cuda.empty_cache()
    checks["all_replicates_passed"] = official["all_replicates_passed"] == all(all_passes)
    result = {
        "phase": p1149.PHASE,
        "scope": "causal",
        "protocol_digest": prereg["protocol_digest"],
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
    }
    result["audit_digest"] = p1149.p1148.canonical_digest(result)
    p1149.p1148.write_json(OUT_ROOT / "audit" / "causal_independent_result_audit.json", result)
    if not result["all_checks_passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Phase1149 causal audit failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("discovery", "confirmation", "causal"), required=True)
    args = parser.parse_args()
    result = audit_causal() if args.scope == "causal" else audit_split(args.scope)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
