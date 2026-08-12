from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import phase1148_mandatory_mediation_calibration as p1148


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1148_mandatory_mediation_calibration"


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def load_model(summary: dict[str, Any]) -> p1148.MediationBindingModel:
    checkpoint = torch.load(ROOT / summary["model_path"], map_location="cpu", weights_only=True)
    config = p1148.p1147.p1146.ModelConfig(**checkpoint["config"])
    model = p1148.MediationBindingModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.cuda().eval()


def audit(split: str) -> dict[str, Any]:
    prereg = p1148.read_json(p1148.PREREG_PATH)
    p1148.verify_preregistration(prereg)
    protocol_audit = p1148.read_json(OUT_ROOT / "protocol" / "audit.json")
    replicates = p1148.split_replicates(prereg, split)
    checks: dict[str, bool] = {
        "protocol_audit": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest": protocol_audit["protocol_digest"] == prereg["protocol_digest"],
    }
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    recomputed_metrics: dict[str, Any] = {}
    for replicate in replicates:
        summaries[replicate] = {}
        for condition in p1148.CONDITIONS:
            summary = p1148.load_summary(replicate, condition, prereg)
            summaries[replicate][condition] = summary
            prefix = f"{replicate}.{condition}"
            checks[f"{prefix}.model_hash"] = p1148.file_sha256(
                ROOT / summary["model_path"]
            ) == summary["model_sha256"]
            checks[f"{prefix}.prediction_hash"] = p1148.file_sha256(
                ROOT / summary["predictions_path"]
            ) == summary["predictions_sha256"]
            checks[f"{prefix}.fixed_steps"] = summary["training_steps"] == prereg[
                "replicates"
            ][replicate]["training"]["max_steps"]
            checks[f"{prefix}.finite"] = summary["nonfinite_steps"] == 0
            for point in summary["trajectory"]:
                checkpoint = point["checkpoint"]
                checks[f"{prefix}.trajectory.{point['step']}.hash"] = p1148.file_sha256(
                    ROOT / checkpoint["path"]
                ) == checkpoint["sha256"]
            model = load_model(summary)
            evaluation, _, digests = p1148.evaluate_bundle(
                model,
                condition,
                prereg["replicates"][replicate],
                prereg,
                "formal",
                False,
            )
            detail: dict[str, bool] = {}
            for evaluation_split in ("seen", "holdout", "quartet"):
                for key in (
                    "accuracy",
                    "minimum_field_accuracy",
                    "minimum_entity_accuracy",
                    "row_address_accuracy",
                    "column_address_accuracy",
                    "joint_address_accuracy",
                    "oracle_accuracy",
                ):
                    detail[f"{evaluation_split}.{key}"] = close(
                        evaluation[evaluation_split][key],
                        summary["evaluation"][evaluation_split][key],
                    )
                detail[f"{evaluation_split}.digest"] = (
                    digests[evaluation_split]
                    == summary["evaluation_dataset_digests"][evaluation_split]
                )
            checks[f"{prefix}.metrics_recomputed"] = all(detail.values())
            recomputed_metrics[prefix] = detail
            answer_checks = p1148.answer_gate(summary["evaluation"], prereg["thresholds"])
            address_checks = p1148.address_gate(
                summary["evaluation"], condition, prereg["thresholds"]
            )
            checks[f"{prefix}.gates"] = (
                answer_checks == summary["answer_gate_checks"]
                and address_checks == summary["address_gate_checks"]
                and summary["qualified"]
                == (all(answer_checks.values()) and all(address_checks.values()))
            )
            del model
            torch.cuda.empty_cache()
        checks[f"{replicate}.paired_initial"] = len(
            {
                summaries[replicate][condition]["initial_state_digest"]
                for condition in p1148.CONDITIONS
            }
        ) == 1
        checks[f"{replicate}.paired_data"] = len(
            {
                summaries[replicate][condition]["training_dataset_digest"]
                for condition in p1148.CONDITIONS
            }
        ) == 1
        checks[f"{replicate}.paired_schedule"] = len(
            {
                summaries[replicate][condition]["batch_schedule_digest"]
                for condition in p1148.CONDITIONS
            }
        ) == 1
        checks[f"{replicate}.equal_parameters"] = len(
            {
                summaries[replicate][condition]["parameter_count"]
                for condition in p1148.CONDITIONS
            }
        ) == 1
    official = p1148.read_json(OUT_ROOT / "analysis" / f"{split}_selection.json")
    all_qualified = {
        condition: all(
            summaries[replicate][condition]["qualified"] for replicate in replicates
        )
        for condition in p1148.CONDITIONS
    }
    eligible = [
        condition
        for condition in p1148.SOFT_PRIORITY
        if all_qualified[condition]
    ]
    checks["selection.qualified"] = official["condition_all_qualified"] == all_qualified
    checks["selection.eligible"] = official["eligible_soft_conditions"] == eligible
    if split == "discovery":
        expected_selected = eligible[0] if eligible else None
        checks["selection.selected"] = official["selected_condition"] == expected_selected
        checks["selection.authorization"] = official["confirmation_authorized"] == bool(eligible)
    else:
        discovery = p1148.read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
        expected_selected = discovery["selected_condition"]
        checks["selection.selected"] = official["selected_condition"] == expected_selected
        checks["selection.authorization"] = official["causal_validation_authorized"] == (
            bool(expected_selected) and all_qualified[str(expected_selected)]
        )
    for replicate in replicates:
        for evaluation_split in ("seen", "holdout", "quartet"):
            accuracy = {
                condition: summaries[replicate][condition]["evaluation"][evaluation_split][
                    "accuracy"
                ]
                for condition in p1148.CONDITIONS
            }
            observed = official["effects"][replicate][evaluation_split]
            expected_values = {
                "force_without_aux": accuracy["soft_00"] - accuracy["free_00"],
                "force_with_aux": accuracy["soft_EF"] - accuracy["free_EF"],
                "auxiliary_free": accuracy["free_EF"] - accuracy["free_00"],
                "auxiliary_soft": accuracy["soft_EF"] - accuracy["soft_00"],
            }
            expected_values["factorial_interaction"] = (
                expected_values["auxiliary_soft"] - expected_values["auxiliary_free"]
            )
            for key, expected in expected_values.items():
                checks[f"selection.{replicate}.{evaluation_split}.{key}"] = close(
                    expected, observed[key]
                )
    result = {
        "phase": p1148.PHASE,
        "split": split,
        "protocol_digest": prereg["protocol_digest"],
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "metric_recomputation": recomputed_metrics,
        "source_hashes": {
            "primary": p1148.file_sha256(Path(p1148.__file__).resolve()),
            "audit": p1148.file_sha256(Path(__file__).resolve()),
            "phase1147_dependency": p1148.file_sha256(Path(p1148.p1147.__file__).resolve()),
            "phase1146_dependency": p1148.file_sha256(
                Path(p1148.p1147.p1146.__file__).resolve()
            ),
        },
    }
    result["audit_digest"] = p1148.canonical_digest(result)
    p1148.write_json(OUT_ROOT / "audit" / f"{split}_independent_result_audit.json", result)
    if not result["all_checks_passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Phase1148 audit failed: {failed}")
    return result


def audit_causal() -> dict[str, Any]:
    prereg = p1148.read_json(p1148.PREREG_PATH)
    p1148.verify_preregistration(prereg)
    official = p1148.read_json(
        OUT_ROOT / "analysis" / "causal_mediation_validation.json"
    )
    confirmation = p1148.read_json(
        OUT_ROOT / "analysis" / "confirmation_selection.json"
    )
    condition = confirmation["selected_condition"]
    if not condition or not confirmation["causal_validation_authorized"]:
        raise RuntimeError("Causal validation was not authorized")

    checks: dict[str, bool] = {
        "protocol_digest": official["protocol_digest"] == prereg["protocol_digest"],
        "selected_condition": official["selected_condition"] == condition,
    }
    recomputed: dict[str, Any] = {}
    all_passed = True
    for replicate in p1148.split_replicates(prereg, "confirmation"):
        summary = p1148.load_summary(replicate, condition, prereg)
        model = load_model(summary)
        spec = prereg["replicates"][replicate]
        dataset = p1148.make_dataset(
            int(prereg["data"]["evaluation_count"]),
            prereg["data"]["pairs"]["confirmation"],
            int(spec["data_seeds"]["holdout_evaluation"]),
            spec["lexicon"],
        )
        metrics = {
            mode: p1148.intervention_accuracy(
                model,
                dataset,
                mode,
                int(spec["training"]["evaluation_batch_size"]),
            )
            for mode in (
                "predicted",
                "uniform_row",
                "uniform_column",
                "oracle_row",
                "oracle_column",
                "oracle_both",
            )
        }
        metrics["uniform_row_drop"] = metrics["predicted"] - metrics["uniform_row"]
        metrics["uniform_column_drop"] = (
            metrics["predicted"] - metrics["uniform_column"]
        )
        expected = {
            "base_behavior": metrics["predicted"]
            >= prereg["thresholds"]["holdout_accuracy"],
            "row_necessity": metrics["uniform_row_drop"]
            >= prereg["thresholds"]["uniform_address_drop"],
            "column_necessity": metrics["uniform_column_drop"]
            >= prereg["thresholds"]["uniform_address_drop"],
            "oracle_rescue": metrics["oracle_both"]
            >= prereg["thresholds"]["oracle_rescue_accuracy"],
        }
        observed = official["per_replicate"][replicate]
        detail: dict[str, bool] = {}
        for key, value in metrics.items():
            detail[key] = close(value, observed["metrics"][key])
        detail["dataset_digest"] = (
            p1148.dataset_digest(dataset) == observed["dataset_digest"]
        )
        detail["gate_checks"] = expected == observed["gate"]
        detail["passed"] = observed["passed"] == all(expected.values())
        checks[f"{replicate}.recomputed"] = all(detail.values())
        recomputed[replicate] = detail
        all_passed = all_passed and all(expected.values())
        del model
        torch.cuda.empty_cache()
    checks["all_replicates_passed"] = (
        official["all_replicates_passed"] == all_passed
    )
    result = {
        "phase": p1148.PHASE,
        "scope": "causal",
        "protocol_digest": prereg["protocol_digest"],
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "recomputed": recomputed,
    }
    result["audit_digest"] = p1148.canonical_digest(result)
    p1148.write_json(
        OUT_ROOT / "audit" / "causal_independent_result_audit.json",
        result,
    )
    if not result["all_checks_passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Phase1148 causal audit failed: {failed}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["discovery", "confirmation", "causal"],
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_causal() if args.split == "causal" else audit(args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
