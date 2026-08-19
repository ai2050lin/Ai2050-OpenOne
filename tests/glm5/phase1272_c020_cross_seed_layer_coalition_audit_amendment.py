"""Type-correct final-audit amendment for Phase1272/C020.

The frozen auditor incorrectly required the scientific behavior gate itself to
pass before treating the qualification/model ledgers as complete.  This
amendment preserves that failed audit and checks the preregistered rejection
branch: every seed has a qualification record, while only qualified seeds may
enter the mechanism ledger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1272_c020_cross_seed_layer_coalition as main


OUT = main.OUT / "audit/independent_final_audit_amendment.json"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def run() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    preaudit = read(main.PREAUDIT)
    original = read(main.FINAL_AUDIT)
    final = read(main.FINAL)
    complete = read(main.COMPLETE)
    summary = read(main.SUMMARY)
    qualifications = main.read_jsonl(main.QUALIFICATION)
    discoveries = main.read_jsonl(main.DISCOVERY)
    models = main.read_jsonl(main.MODELS)

    add(checks, "frozen_preaudit_passed", preaudit.get("all_checks_passed") is True and preaudit.get("checks_passed") == 18)
    failed_names = {row["name"] for row in original["checks"] if not row["passed"]}
    add(checks, "original_failure_is_known_type_mismatch", original.get("all_checks_passed") is False and failed_names == {"qualification_complete", "model_gates_recomputed", "heldout_roles"}, sorted(failed_names))
    add(checks, "formal_complete", complete.get("status") == "formal_run_complete")
    add(checks, "qualification_ledger_complete", len(qualifications) == 9)
    per_depth_qual = {architecture: sum(row["architecture"] == architecture for row in qualifications) for architecture in main.ARCHITECTURES}
    add(checks, "qualification_three_per_depth", all(value == 3 for value in per_depth_qual.values()), per_depth_qual)
    scientific_behavior_gate = len(qualifications) == 9 and all(row["passed"] for row in qualifications)
    add(checks, "behavior_failure_preserved", scientific_behavior_gate is False and final["gates"]["G-FROZEN-BEHAVIOR-POPULATION"] is False)
    qualified_keys = {(row["architecture"], int(row["replicate"])) for row in qualifications if row["passed"]}
    model_keys = {(row["architecture"], int(row["replicate"])) for row in models}
    add(checks, "only_qualified_models_measured", model_keys == qualified_keys and len(models) == 8, {"qualified": sorted(qualified_keys), "measured": sorted(model_keys)})
    expected_heldout = {(row["architecture"], int(row["replicate"])) for row in qualifications if row["passed"] and int(row["replicate"]) in main.HELDOUT_INDICES}
    actual_heldout = {(row["architecture"], int(row["replicate"])) for row in models if row["role"] == "heldout"}
    add(checks, "heldout_rejection_branch", actual_heldout == expected_heldout and len(actual_heldout) == 5)
    add(checks, "discovery_ledger_complete", len(discoveries) == 3 and all(len(row["mask_ledger"]) == 2 ** main.ARCHITECTURES[row["architecture"]].layers for row in discoveries))

    ledger_ok = True
    false_authorizations = 0
    for discovery in discoveries:
        selected = main.select_mask(discovery["mask_ledger"])
        ledger_ok &= selected == discovery["selected_mask"] and discovery["selection_abstained"] is False
        for row in discovery["mask_ledger"]:
            rebuilt = main.with_bounds(row)
            ledger_ok &= abs(rebuilt["certificate_lower"] - row["certificate_lower"]) <= 1.0e-12
            ledger_ok &= abs(rebuilt["false_target_upper"] - row["false_target_upper"]) <= 1.0e-12
            ledger_ok &= rebuilt["certificate_pass"] == row["certificate_pass"]
            population_pass = main.exact_pass(row["population"])
            ledger_ok &= population_pass == row["population_pass"]
            false_authorizations += int(row["certificate_pass"] and not population_pass)
    add(checks, "selection_and_certificate_recomputed", ledger_ok)
    add(checks, "zero_false_authorization", false_authorizations == final["false_authorizations"] == 0)

    discovery_masks = {row["architecture"]: row["selected_mask"] for row in discoveries}
    model_ok = True
    for row in models:
        selected_pass = main.exact_pass(row["selected_metrics"])
        controls = (
            main.exact_pass(row["full_metrics"])
            and max(row["empty_metrics"]["forward"], row["empty_metrics"]["reverse"], row["empty_metrics"]["wrong"]) <= main.NULL_MAX
            and row["same_state_noop_score"] >= main.IDENTITY_MIN
            and row["pre_source_null_score"] <= main.NULL_MAX
        )
        model_ok &= row["selected_mask"] == discovery_masks[row["architecture"]]
        model_ok &= row["controls_passed"] == controls
        model_ok &= row["selected_transfer_passed"] == (selected_pass and controls)
        model_ok &= row["shared_minimality_passed"] == (selected_pass and len(row["proper_subset_passes"]) == 0)
    add(checks, "model_gates_recomputed_on_qualified_population", model_ok)

    rebuilt_summary = main.summarize(discoveries, models, qualifications)
    keys = (
        "candidate_masks", "false_authorizations", "robust_masks", "robust_coverage",
        "selected_masks", "selected_cardinalities", "all_transfer_models", "heldout_transfer_models",
        "heldout_transfer_per_depth", "shared_minimality_models", "shared_minimality_passed",
        "sparse_per_depth", "sparse_passed", "gates", "passed", "decision",
    )
    add(checks, "summary_recomputed", all(rebuilt_summary[key] == final[key] for key in keys))
    add(checks, "raw_hashes", final["qualification_hash"] == main.file_sha256(main.QUALIFICATION) and final["discovery_hash"] == main.file_sha256(main.DISCOVERY) and final["models_hash"] == main.file_sha256(main.MODELS))
    run_digest = main.digest({"qualification": qualifications, "discovery": discoveries, "models": models})
    add(checks, "run_digest", run_digest == summary["run_digest"] == complete["run_digest"])
    without_digest = dict(final)
    stored = without_digest.pop("final_digest")
    add(checks, "final_digest", stored == main.digest(without_digest))
    add(checks, "negative_authorization", final["passed"] is False and final["authorization"]["head_or_microcomponent_contract"] is False and final["authorization"]["synthetic_layer_coalition_search_closed"] is True)
    add(checks, "no_pretrained_loaded", summary["pretrained_model_loaded"] is False)

    passed = all(row["passed"] for row in checks)
    payload = {
        "phase": main.PHASE,
        "mode": "final_audit_amendment",
        "reason": "The frozen auditor confused scientific gate failure with ledger incompleteness; the rejection branch is audited here without changing any scientific result.",
        "checks": checks,
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "all_checks_passed": passed,
        "original_audit_hash": main.file_sha256(main.FINAL_AUDIT),
        "final_hash": main.file_sha256(main.FINAL),
    }
    main.atomic_json(OUT, payload)
    print(json.dumps({"mode": payload["mode"], "checks": f"{payload['checks_passed']}/{payload['checks_total']}", "passed": passed}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)
    return payload


if __name__ == "__main__":
    run()
