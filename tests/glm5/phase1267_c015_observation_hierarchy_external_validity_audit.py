"""Independent audit for Phase1267/C015."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1267_c015_observation_hierarchy_external_validity as main


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def shape_and_leakage_sentinel() -> bool:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1_267_999)
    states = {
        name: torch.randn((24, 3, 7), generator=generator)
        for name in pannels()
    }
    capture = {"states": states}
    changed = {"states": {name: value.clone() for name, value in states.items()}}
    changed["states"]["h11"] = torch.randn((24, 3, 7), generator=generator) * 100.0
    for layer in range(3):
        for camera in main.CAMERAS[1:]:
            left = main.raw_components(capture, layer, camera)
            right = main.raw_components(changed, layer, camera)
            if len(left) != len(right) or not all(torch.equal(a, b) for a, b in zip(left, right)):
                return False
            model = main.fit_camera(capture, layer, camera)
            predicted = main.predict_camera(model, capture, layer)
            if predicted.shape != (24, 7) or not torch.isfinite(predicted).all():
                return False
    return True


def pannels() -> tuple[str, ...]:
    return ("h00", "h10", "h01", "h11", "hwrong10", "hwrong11")


def preaudit() -> dict[str, Any]:
    protocol = read_json(main.PROTOCOL)
    rows = read_jsonl(main.MATERIAL)
    predecessor = read_json(main.PHASE1266_FINAL)
    predecessor_audit = read_json(main.PHASE1266_AUDIT)
    erratum = read_json(main.PHASE1266_ERRATUM)
    counts = {name: sum(row["partition"] == name for row in rows) for name in main.PARTITION_COUNTS}
    expected_radius = math.sqrt(
        math.log(2.0 * main.MAX_EVENTS * len(main.CAMERAS) / main.GLOBAL_ERROR_BUDGET)
        / (2.0 * main.SELECTION_DRAWS)
    )
    row_digests = True
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        row_digests = row_digests and main.digest(value) == stored
    checks = {
        "predecessor_complete_negative": main.PHASE1266_COMPLETE.exists() and predecessor.get("passed") is False,
        "predecessor_frozen_audit_ledger": predecessor_audit.get("passed_checks") == 15 and predecessor_audit.get("total_checks") == 16,
        "predecessor_erratum": erratum.get("passed") is True and erratum.get("scientific_verdict_changed") is False,
        "contract_exists": main.CONTRACT.exists(),
        "model_count_and_unique_seeds": len(main.MODEL_SEEDS) == 9 and len(set(main.MODEL_SEEDS.values())) == 9,
        "fresh_from_development_seed": 1_267_301_001 not in set(main.MODEL_SEEDS.values()),
        "three_replicates_each_depth": all(sum(key.startswith(name) for key in main.MODEL_SEEDS) == 3 for name in main.ARCHITECTURES),
        "partition_counts": counts == main.PARTITION_COUNTS,
        "row_digests": row_digests,
        "camera_order": protocol.get("camera_order") == list(main.CAMERAS),
        "certificate_radius": abs(protocol["thresholds"]["certificate_radius"] - expected_radius) <= 1.0e-15,
        "source_hash_main": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()),
        "source_hash_auditor": protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "source_hash_contract": protocol["source_hashes"]["contract"] == sha256(main.CONTRACT),
        "protocol_recomputes": protocol["protocol_digest"] == main.protocol_payload(rows)["protocol_digest"],
        "structured_scope": protocol.get("structured_scope") == {
            "task": "synthetic cyclic-code",
            "models": "small free same-executor Transformers",
            "natural_language": False,
            "unique_circuit": False,
            "pretrained": False,
        },
        "no_formal_output_before_run": not main.COMPLETE.exists() and not main.FINAL.exists(),
        "shape_and_h11_input_leakage_sentinel": shape_and_leakage_sentinel(),
        "one_run_no_adaptation": protocol["budgets"]["max_formal_runs"] == 1 and protocol["budgets"]["max_adaptive_rounds"] == 0,
        "no_automatic_pretrained": "No pretrained model is loaded automatically." in protocol["hard_stops"],
    }
    return {
        "mode": "pre",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
    }


def confirmation_logic(results: list[dict[str, Any]]) -> bool:
    okay = True
    for row in results:
        if not row["behavior_qualified"]:
            okay = okay and row.get("cameras") == {}
            continue
        for camera in main.CAMERAS:
            value = row["cameras"][camera]
            eligible = [
                event["layer"]
                for event in value["events"]
                if event["rescue_authorized"]
                and row["causal_sentinels"].get(str(event["layer"]), row["causal_sentinels"].get(event["layer"], {})).get("causally_admissible", False)
            ]
            expected = []
            if eligible:
                expected.append(eligible[0])
                if eligible[-1] != eligible[0]:
                    expected.append(eligible[-1])
            okay = okay and value["selected_events"] == expected
            expected_pass = bool(expected) and all(item["passed"] for item in value["confirmations"])
            okay = okay and value["passed"] == expected_pass
            for item in value["confirmations"]:
                recomputed = (
                    item["state_relative_error"] <= main.THRESHOLDS["causal_state_relative_error_max"]
                    and item["correct_output"]["cosine"] >= main.THRESHOLDS["causal_output_cosine_min"]
                    and item["correct_accuracy"] >= main.THRESHOLDS["causal_correct_accuracy_min"]
                    and item["wrong_identity_accuracy"] >= main.THRESHOLDS["wrong_identity_accuracy_min"]
                    and item["wrong_false_target"] <= main.THRESHOLDS["wrong_false_target_max"]
                    and item["oracle_patch_accuracy"] >= main.THRESHOLDS["oracle_patch_accuracy_min"]
                    and item["reverse_block_accuracy"] >= main.THRESHOLDS["reverse_block_accuracy_min"]
                )
                okay = okay and item["passed"] == recomputed
    return okay


def certificate_logic(results: list[dict[str, Any]]) -> bool:
    okay = True
    for row in results:
        for value in row.get("cameras", {}).values():
            for event in value["events"]:
                expected = main.confidence(event["sample_risk"])
                okay = okay and all(abs(event["confidence"][key] - expected[key]) <= 1.0e-12 for key in expected)
                okay = okay and event["exact_pass"] == (event["population_risk"] <= main.PASS_MAX)
                okay = okay and event["certificate_pass"] == (event["confidence"]["upper"] <= main.PASS_MAX)
                okay = okay and event["robust_actionable"] == (
                    event["population_risk"] <= main.PASS_MAX - main.ROBUST_MULTIPLIER * main.CERTIFICATE_RADIUS
                )
                okay = okay and event["rescue_authorized"] == (
                    event["certificate_pass"]
                    and event["confidence"]["upper"] <= main.THRESHOLDS["rescue_authorization_upper_max"]
                )
    return okay


def final_audit() -> dict[str, Any]:
    protocol = read_json(main.PROTOCOL)
    complete = read_json(main.COMPLETE)
    run_summary = read_json(main.SUMMARY)
    results = read_jsonl(main.MODELS)
    final = read_json(main.FINAL)
    recomputed = main.summarize(results)
    final_without_digest = dict(final)
    stored_digest = final_without_digest.pop("final_digest")
    checks = {
        "formal_marker": complete.get("status") == "formal_run_complete",
        "model_count": len(results) == 9 and run_summary.get("models") == 9,
        "models_hash": run_summary.get("models_hash") == sha256(main.MODELS) == final.get("models_hash"),
        "run_digest": complete.get("run_digest") == main.digest(results) == final.get("run_digest"),
        "protocol_digest": run_summary.get("protocol_digest") == protocol.get("protocol_digest") == final.get("protocol_digest"),
        "source_hashes_frozen": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()) and protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "certificate_math": certificate_logic(results),
        "confirmation_selection_and_thresholds": confirmation_logic(results),
        "summary_recomputed": all(final.get(key) == value for key, value in recomputed.items()),
        "decision_registered": final.get("decision") in set(protocol["decision_order"].values()),
        "authorization_matches": final["authorization"]["new_pretrained_contract_design"] == final["passed"] and final["authorization"]["automatic_pretrained_run"] is False,
        "no_pretrained_loaded": run_summary.get("pretrained_model_loaded") is False,
        "structured_scope": final.get("structured_scope") == protocol.get("structured_scope") and final["structured_scope"]["natural_language"] is False,
        "final_digest": stored_digest == main.digest(final_without_digest),
    }
    return {
        "mode": "final",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed": recomputed,
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    payload = preaudit() if args.mode == "pre" else final_audit()
    target = main.PREAUDIT if args.mode == "pre" else main.FINAL_AUDIT
    write(target, payload)
    print(json.dumps({"mode": args.mode, "checks": f"{payload['passed_checks']}/{payload['total_checks']}", "passed": payload["all_checks_passed"]}))
    raise SystemExit(0 if payload["all_checks_passed"] else 1)


if __name__ == "__main__":
    main_cli()
