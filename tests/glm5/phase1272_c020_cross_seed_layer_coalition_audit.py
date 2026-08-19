"""Independent audit for Phase1272/C020."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1272_c020_cross_seed_layer_coalition as main


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def executor_sentinel() -> dict[str, Any]:
    config = main.ARCHITECTURES["shallow4"]
    torch.manual_seed(12_720_019)
    model = main.p1266.task_module.TinyCausalTransformer(config).eval()
    row = main.p1266.make_factorial_world(0, 1, 0, 2, [2, 0, 3, 1], "audit", "audit0")
    h01 = torch.tensor([row["h01_ids"]])
    h11 = torch.tensor([row["h11_ids"]])
    trace01 = main.p1271.capture_micro(model, h01)
    trace11 = main.p1271.capture_micro(model, h11)
    empty = main.forward_masks(model, h01, trace11, [[]])
    own_full = main.forward_masks(model, h01, trace01, [list(range(config.layers))])
    own_empty = main.forward_masks(model, h01, trace01, [[]])
    prefix_difference = max(float(torch.max(torch.abs(trace01[layer]["attn_write"][:, 2] - trace11[layer]["attn_write"][:, 2])).item()) for layer in range(config.layers))
    mlp = main.forward_masks(model, h01, trace11, [[0]], stage="mlp_write")
    return {
        "shape": list(empty.shape),
        "own_noop_prediction_equal": bool(torch.equal(own_full, own_empty)),
        "causal_prefix_difference": prefix_difference,
        "mlp_shape": list(mlp.shape),
    }


def preaudit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = read_json(main.PROTOCOL)
    material = main.read_jsonl(main.MATERIAL)
    predecessor = read_json(main.PREDECESSOR_FINAL)
    predecessor_audit = read_json(main.PREDECESSOR_AUDIT)
    add(checks, "dependency_authorized", predecessor.get("passed") is True and predecessor.get("authorization", {}).get("layer_coalition_minimality_contract") is True)
    add(checks, "dependency_audited", predecessor_audit.get("all_checks_passed") is True)
    add(checks, "protocol_digest", protocol["protocol_digest"] == main.protocol_payload(material)["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == main.protocol_payload(material)["source_hashes"])
    counts = {name: sum(row["partition"] == name for row in material) for name in main.PARTITION_COUNTS}
    add(checks, "partition_counts", counts == main.PARTITION_COUNTS, counts)
    add(checks, "material_row_digests", all(main.digest({key: value for key, value in row.items() if key != "row_digest"}) == row["row_digest"] for row in material))
    predecessor_seeds = {(row["architecture"], int(row["selected_index"])): int(row["seed"]) for row in main.read_jsonl(main.PREDECESSOR_MODELS)}
    expected_seeds = {(architecture, index): seed for architecture, seeds in main.SEEDS.items() for index, seed in enumerate(seeds)}
    add(checks, "frozen_model_population", predecessor_seeds == expected_seeds)
    add(checks, "discovery_heldout_split", main.DISCOVERY_INDEX not in main.HELDOUT_INDICES and len(set(main.HELDOUT_INDICES)) == 2)
    mask_counts = {name: len(main.all_masks(config.layers)) for name, config in main.ARCHITECTURES.items()}
    add(checks, "complete_power_sets", mask_counts == {name: 2 ** config.layers for name, config in main.ARCHITECTURES.items()}, mask_counts)
    add(checks, "selection_hypothesis_count", sum(mask_counts.values()) == main.MAX_SELECTION_MASKS)
    radius = math.sqrt(math.log(2.0 * main.MAX_SELECTION_MASKS * 4.0 / main.GLOBAL_ERROR_BUDGET) / (2.0 * main.PARTITION_COUNTS["selection"]))
    add(checks, "simultaneous_radius", abs(radius - main.CERTIFICATE_RADIUS) <= 1.0e-15, radius)
    add(checks, "strict_thresholds", main.PASS_MIN == 0.95 and main.NULL_MAX == 0.05 and main.IDENTITY_MIN == 0.999)
    add(checks, "selection_rule_frozen", protocol["selection"]["heldout_blind"] is True and protocol["selection"]["oracle_blind_for_selection"] is True)
    add(checks, "teacher_forced_scope", protocol["structured_scope"]["natural_necessity"] is False)
    add(checks, "no_pretrained", protocol["structured_scope"]["pretrained"] is False)
    sentinel = executor_sentinel()
    add(checks, "executor_shapes", sentinel["shape"] == [1, 1] and sentinel["mlp_shape"] == [1, 1], sentinel)
    add(checks, "same_state_noop", sentinel["own_noop_prediction_equal"] is True, sentinel)
    add(checks, "causal_prefix_null", sentinel["causal_prefix_difference"] == 0.0, sentinel)
    passed = all(item["passed"] for item in checks)
    payload = {"phase": main.PHASE, "mode": "pre", "checks": checks, "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks), "all_checks_passed": passed}
    main.atomic_json(main.PREAUDIT, payload)
    return payload


def final_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    final = read_json(main.FINAL)
    complete = read_json(main.COMPLETE)
    summary = read_json(main.SUMMARY)
    qualifications = main.read_jsonl(main.QUALIFICATION)
    discoveries = main.read_jsonl(main.DISCOVERY)
    models = main.read_jsonl(main.MODELS)
    add(checks, "formal_complete", complete.get("status") == "formal_run_complete")
    add(checks, "qualification_complete", len(qualifications) == 9 and all(row["passed"] for row in qualifications))
    add(checks, "discovery_complete", len(discoveries) == 3)
    ledger_valid = True
    selections_valid = True
    false_count = 0
    robust_count = 0
    for discovery in discoveries:
        expected_count = 2 ** main.ARCHITECTURES[discovery["architecture"]].layers
        if len(discovery["mask_ledger"]) != expected_count:
            ledger_valid = False
        rebuilt = []
        for row in discovery["mask_ledger"]:
            calculated = main.with_bounds(row)
            if any(abs(calculated[key] - row[key]) > 1.0e-12 for key in ("certificate_lower", "false_target_upper")):
                ledger_valid = False
            if bool(calculated["certificate_pass"]) != bool(row["certificate_pass"]):
                ledger_valid = False
            population_pass = main.exact_pass(row["population"])
            if population_pass != bool(row["population_pass"]):
                ledger_valid = False
            false_value = bool(row["certificate_pass"] and not population_pass)
            if false_value != bool(row["false_authorization"]):
                ledger_valid = False
            false_count += int(false_value)
            robust_count += int(row["robust_actionable"])
            rebuilt.append(row)
        selected = main.select_mask(rebuilt)
        if selected is None:
            selections_valid &= discovery["selection_abstained"] is True
            selections_valid &= discovery["selected_mask"] == list(range(main.ARCHITECTURES[discovery["architecture"]].layers))
        else:
            selections_valid &= discovery["selection_abstained"] is False
            selections_valid &= selected == discovery["selected_mask"]
    add(checks, "discovery_ledgers_recomputed", ledger_valid)
    add(checks, "selection_recomputed", selections_valid)
    add(checks, "zero_false_authorization", false_count == final["false_authorizations"] == 0, false_count)
    add(checks, "robust_masks_exist", robust_count == final["robust_masks"] and robust_count > 0, robust_count)
    discovery_masks = {row["architecture"]: row["selected_mask"] for row in discoveries}
    model_valid = True
    for row in models:
        selected = row["selected_metrics"]
        full = row["full_metrics"]
        empty = row["empty_metrics"]
        selected_pass = main.exact_pass(selected)
        proper_pass = len(row["proper_subset_passes"]) == 0
        controls = (
            main.exact_pass(full)
            and max(empty["forward"], empty["reverse"], empty["wrong"]) <= main.NULL_MAX
            and row["same_state_noop_score"] >= main.IDENTITY_MIN
            and row["pre_source_null_score"] <= main.NULL_MAX
        )
        model_valid &= row["selected_mask"] == discovery_masks[row["architecture"]]
        model_valid &= bool(row["selected_transfer_passed"]) == bool(selected_pass and controls)
        model_valid &= bool(row["shared_minimality_passed"]) == bool(selected_pass and proper_pass)
        model_valid &= bool(row["controls_passed"]) == bool(controls)
    add(checks, "model_gates_recomputed", model_valid and len(models) == 9)
    heldout_roles = {(row["architecture"], row["replicate"]) for row in models if row["role"] == "heldout"}
    expected_heldout = {(architecture, replicate) for architecture in main.ARCHITECTURES for replicate in main.HELDOUT_INDICES}
    add(checks, "heldout_roles", heldout_roles == expected_heldout)
    rebuilt_summary = main.summarize(discoveries, models, qualifications)
    summary_keys = (
        "candidate_masks", "false_authorizations", "robust_masks", "robust_coverage",
        "selected_masks", "selected_cardinalities", "all_transfer_models",
        "heldout_transfer_models", "heldout_transfer_per_depth", "shared_minimality_models",
        "shared_minimality_passed", "sparse_per_depth", "sparse_passed", "gates", "passed", "decision",
    )
    add(checks, "summary_recomputed", all(rebuilt_summary[key] == final[key] for key in summary_keys))
    add(checks, "raw_hashes", final["qualification_hash"] == main.file_sha256(main.QUALIFICATION) and final["discovery_hash"] == main.file_sha256(main.DISCOVERY) and final["models_hash"] == main.file_sha256(main.MODELS))
    run_digest = main.digest({"qualification": qualifications, "discovery": discoveries, "models": models})
    add(checks, "run_digest", run_digest == summary["run_digest"] == complete["run_digest"])
    payload_without_digest = dict(final)
    stored_digest = payload_without_digest.pop("final_digest")
    add(checks, "final_digest", stored_digest == main.digest(payload_without_digest))
    head_expected = bool(final["passed"] and final["shared_minimality_passed"] and final["sparse_passed"])
    add(checks, "authorization_scope", final["authorization"]["head_or_microcomponent_contract"] == head_expected and final["authorization"]["synthetic_layer_coalition_search_closed"] is True and final["authorization"]["automatic_pretrained_run"] is False)
    add(checks, "no_pretrained_loaded", summary["pretrained_model_loaded"] is False)
    passed = all(item["passed"] for item in checks)
    payload = {"phase": main.PHASE, "mode": "final", "checks": checks, "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks), "all_checks_passed": passed}
    main.atomic_json(main.FINAL_AUDIT, payload)
    return payload


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    payload = preaudit() if args.mode == "pre" else final_audit()
    print(json.dumps({"mode": payload["mode"], "checks": f"{payload['checks_passed']}/{payload['checks_total']}", "passed": payload["all_checks_passed"]}, ensure_ascii=False))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
