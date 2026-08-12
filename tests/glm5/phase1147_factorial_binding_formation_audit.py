from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase1147_factorial_binding_formation as p1147


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1147_factorial_binding_formation"


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def load_model(summary: dict[str, Any]) -> p1147.FactorialBindingModel:
    checkpoint = torch.load(ROOT / summary["model_path"], map_location="cpu", weights_only=True)
    config = p1147.p1146.ModelConfig(**checkpoint["config"])
    model = p1147.FactorialBindingModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.cuda().eval()


def recompute_summary_metrics(
    summary: dict[str, Any], replicate_spec: dict[str, Any], prereg: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    model = load_model(summary)
    evaluation, _, digests = p1147.evaluate_all(model, replicate_spec, prereg)
    checks: dict[str, bool] = {}
    for split in ("seen", "holdout", "quartet"):
        for key in (
            "accuracy",
            "minimum_field_accuracy",
            "minimum_entity_accuracy",
            "entity_address_accuracy",
            "field_address_accuracy",
            "joint_address_accuracy",
        ):
            checks[f"{split}.{key}"] = close(
                evaluation[split][key], summary["evaluation"][split][key]
            )
        checks[f"{split}.dataset_digest"] = (
            digests[split] == summary["evaluation_dataset_digests"][split]
        )
    del model
    torch.cuda.empty_cache()
    return all(checks.values()), checks


def independent_selection(
    split: str, summaries: dict[str, dict[str, dict[str, Any]]], prereg: dict[str, Any]
) -> dict[str, Any]:
    arm_all = {
        arm: all(summaries[replicate][arm]["qualified"] for replicate in summaries)
        for arm in p1147.ARMS
    }
    eligible = [arm for arm in p1147.ARM_PRIORITY if arm_all[arm]]
    factorial: dict[str, Any] = {}
    dual_passes: list[bool] = []
    for replicate, arm_summaries in summaries.items():
        factorial[replicate] = {}
        for evaluation_split in ("seen", "holdout", "quartet"):
            accuracy = {
                arm: arm_summaries[arm]["evaluation"][evaluation_split]["accuracy"]
                for arm in p1147.ARMS
            }
            interaction = accuracy["EF"] - accuracy["E0"] - accuracy["0F"] + accuracy["00"]
            over_single = accuracy["EF"] - max(accuracy["E0"], accuracy["0F"])
            factorial[replicate][evaluation_split] = (interaction, over_single)
            if evaluation_split in ("holdout", "quartet"):
                dual_passes.append(
                    interaction >= prereg["thresholds"]["factorial_interaction"]
                    and over_single >= prereg["thresholds"]["dual_over_single_accuracy"]
                )
    dual_synergy = all(dual_passes) and arm_all["EF"]
    if split == "discovery":
        selected = eligible[0] if eligible else None
        confirmation_authorized = bool(eligible)
        mechanism_authorized = None
    else:
        selected = p1147.read_json(OUT_ROOT / "analysis" / "discovery_selection.json")[
            "selected_arm"
        ]
        confirmation_authorized = None
        mechanism_authorized = bool(selected) and arm_all[str(selected)]
    return {
        "arm_all_qualified": arm_all,
        "eligible_arms": eligible,
        "selected_arm": selected,
        "dual_synergy_pass": dual_synergy,
        "confirmation_authorized": confirmation_authorized,
        "mechanism_phase_authorized": mechanism_authorized,
        "factorial": factorial,
    }


def audit(split: str) -> dict[str, Any]:
    prereg = p1147.read_json(p1147.PREREG_PATH)
    p1147.verify_preregistration(prereg)
    dependency_prereg = p1147.read_json(p1147.p1146.PREREG_PATH)
    protocol_audit = p1147.read_json(OUT_ROOT / "protocol" / "audit.json")
    replicates = p1147.split_replicates(prereg, split)
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    recompute_checks: dict[str, Any] = {}
    checks: dict[str, bool] = {
        "protocol_audit_passed": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest_matches": protocol_audit["protocol_digest"]
        == prereg["protocol_digest"],
        "phase1146_dependency_protocol_valid": p1147.canonical_digest(
            {key: value for key, value in dependency_prereg.items() if key != "protocol_digest"}
        )
        == dependency_prereg["protocol_digest"],
        "phase1146_dependency_source_hash_matches": p1147.file_sha256(
            Path(p1147.p1146.__file__).resolve()
        )
        == dependency_prereg["source_hashes"]["primary_script"],
    }
    for replicate in replicates:
        summaries[replicate] = {}
        for arm in p1147.ARMS:
            summary = p1147.load_summary(replicate, arm, prereg)
            summaries[replicate][arm] = summary
            prefix = f"{replicate}.{arm}"
            checks[f"{prefix}.protocol"] = summary["protocol_digest"] == prereg["protocol_digest"]
            checks[f"{prefix}.model_hash"] = p1147.file_sha256(
                ROOT / summary["model_path"]
            ) == summary["model_sha256"]
            checks[f"{prefix}.prediction_hash"] = p1147.file_sha256(
                ROOT / summary["predictions_path"]
            ) == summary["predictions_sha256"]
            checks[f"{prefix}.fixed_steps"] = summary["training_steps"] == prereg[
                "replicates"
            ][replicate]["training"]["max_steps"]
            checks[f"{prefix}.finite_training"] = summary["nonfinite_steps"] == 0
            answer_checks = p1147.answer_gate(summary["evaluation"], prereg["thresholds"])
            aux_checks = p1147.auxiliary_gate(
                summary["evaluation"], arm, prereg["thresholds"]
            )
            checks[f"{prefix}.gate_recomputed"] = (
                answer_checks == summary["answer_gate_checks"]
                and aux_checks == summary["auxiliary_gate_checks"]
                and summary["qualified"]
                == (all(answer_checks.values()) and all(aux_checks.values()))
            )
            recomputed, detail = recompute_summary_metrics(
                summary, prereg["replicates"][replicate], prereg
            )
            checks[f"{prefix}.metrics_recomputed"] = recomputed
            recompute_checks[prefix] = detail
        initial = {summaries[replicate][arm]["initial_state_digest"] for arm in p1147.ARMS}
        materials = {
            summaries[replicate][arm]["training_dataset_digest"] for arm in p1147.ARMS
        }
        schedules = {
            summaries[replicate][arm]["batch_schedule_digest"] for arm in p1147.ARMS
        }
        parameters = {summaries[replicate][arm]["parameter_count"] for arm in p1147.ARMS}
        checks[f"{replicate}.paired_initial_state"] = len(initial) == 1
        checks[f"{replicate}.paired_training_data"] = len(materials) == 1
        checks[f"{replicate}.paired_batch_schedule"] = len(schedules) == 1
        checks[f"{replicate}.equal_parameter_count"] = len(parameters) == 1
    official = p1147.read_json(OUT_ROOT / "analysis" / f"{split}_selection.json")
    independent = independent_selection(split, summaries, prereg)
    for key in (
        "arm_all_qualified",
        "eligible_arms",
        "selected_arm",
        "dual_synergy_pass",
        "confirmation_authorized",
        "mechanism_phase_authorized",
    ):
        checks[f"selection.{key}"] = independent[key] == official[key]
    for replicate in replicates:
        for evaluation_split in ("seen", "holdout", "quartet"):
            expected = independent["factorial"][replicate][evaluation_split]
            observed = official["factorial"][replicate][evaluation_split]
            checks[f"selection.{replicate}.{evaluation_split}.interaction"] = close(
                expected[0], observed["factorial_interaction"]
            )
            checks[f"selection.{replicate}.{evaluation_split}.over_single"] = close(
                expected[1], observed["dual_over_best_single"]
            )
    result = {
        "phase": p1147.PHASE,
        "split": split,
        "protocol_digest": prereg["protocol_digest"],
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "metric_recomputation": recompute_checks,
        "selection_recomputation": independent,
        "source_hashes": {
            "primary_script": p1147.file_sha256(Path(p1147.__file__).resolve()),
            "audit_script": p1147.file_sha256(Path(__file__).resolve()),
            "phase1146_dependency_script": p1147.file_sha256(
                Path(p1147.p1146.__file__).resolve()
            ),
        },
    }
    result["audit_digest"] = p1147.canonical_digest(result)
    p1147.write_json(OUT_ROOT / "audit" / f"{split}_independent_result_audit.json", result)
    if not result["all_checks_passed"]:
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase1147 audit failed: {failed}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["discovery", "confirmation"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
