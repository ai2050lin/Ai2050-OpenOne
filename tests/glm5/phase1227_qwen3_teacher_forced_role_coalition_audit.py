#!/usr/bin/env python3
"""Independent integrity and result audit for Phase 1227."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1226_known_truth_temporal_coalition_camera as p1226


PHASE = 1227
MAIN_SCRIPT = TEST_ROOT / "phase1227_qwen3_teacher_forced_role_coalition.py"
AUDIT_SCRIPT = Path(__file__).resolve()
OUT_ROOT = TEST_ROOT / "result/phase1227_qwen3_teacher_forced_role_coalition"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MANIFEST_PATH = OUT_ROOT / "protocol/anchor_manifest.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RECORD_PATH = OUT_ROOT / "runs/coalition_records.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "runs/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

SOURCE_STATES = TEST_ROOT / "result/phase1223_passed_atom_physical_trajectory/protocol/state_manifest.jsonl"
SOURCE_PAIRS = TEST_ROOT / "result/phase1223_passed_atom_physical_trajectory/protocol/pair_manifest.jsonl"
PHASE1226_MAIN = TEST_ROOT / "phase1226_known_truth_temporal_coalition_camera.py"
SPLITS = ("discovery", "confirmation", "natural_use", "sealed")
HOLDOUT_SPLITS = ("confirmation", "natural_use", "sealed")
PRIMARY_SCOPE = "query_relation|natural"
FROZEN_DEPTH = 18
ROLE_GROUPS = {
    "R": ("record_object", "record_relation", "record_value"),
    "Q": ("query_subject", "query_relation"),
    "B": ("generation_boundary",),
}
ALLIANCES = {
    "R": ("R",), "Q": ("Q",), "B": ("B",),
    "RQ": ("R", "Q"), "RB": ("R", "B"), "QB": ("Q", "B"),
    "RQB": ("R", "Q", "B"),
}
CONDITIONS = tuple(f"correct:{name}" for name in ALLIANCES) + (
    "record_order:RQB", "paraphrase:RQB", "equal_norm_reverse:RQB", "identity:RQB", "zero:RQB",
)
EPSILON = 1e-8


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def median(values: Iterable[float]) -> float:
    data = list(values)
    return float(np.median(np.asarray(data, dtype=np.float64))) if data else 0.0


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def independent_latent_audit() -> dict[str, Any]:
    device = torch.device("cpu")
    rows: list[dict[str, Any]] = []
    for mechanism in p1226.MECHANISMS:
        specs = [p1226.system_spec("discovery", 0, mechanism, label) for label in ("u", "v")]
        models = [p1226.KnownTruthRoleTransformer(spec, device).eval() for spec in specs]
        state_equal = all(torch.equal(models[0].state_dict()[key], models[1].state_dict()[key]) for key in models[0].state_dict())
        public = []
        heldout = []
        for spec in specs:
            item_public, item_heldout, _truth = p1226.response_record(spec, device)
            public.append(item_public["correct_donor_responses"])
            heldout.append(item_heldout["responses"])
        rows.append({
            "mechanism": mechanism,
            "state_dict_equal": bool(state_equal),
            "registered_response_equal": public[0] == public[1] and heldout[0] == heldout[1],
            "only_identity_fields_differ": specs[0].system_id != specs[1].system_id and specs[0].latent_variant != specs[1].latent_variant,
        })
    return {
        "rows": rows,
        "all_physical_equal": all(row["state_dict_equal"] for row in rows),
        "all_response_equal": all(row["registered_response_equal"] for row in rows),
    }


def preaudit() -> None:
    if PREAUDIT_PATH.exists():
        raise RuntimeError("preaudit already exists")
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    pairs = read_jsonl(SOURCE_PAIRS)
    states = {row["state_id"]: row for row in read_jsonl(SOURCE_STATES)}
    latent = independent_latent_audit()
    checks: list[dict[str, Any]] = []
    checks.append(check("protocol_self_digest", protocol["protocol_digest"] == digest({key: value for key, value in protocol.items() if key != "protocol_digest"}), protocol["protocol_digest"]))
    checks.append(check("main_hash_frozen", protocol["source_hashes"]["main"] == file_sha256(MAIN_SCRIPT), protocol["source_hashes"]["main"]))
    checks.append(check("audit_hash_frozen", protocol["source_hashes"]["audit"] == file_sha256(AUDIT_SCRIPT), protocol["source_hashes"]["audit"]))
    checks.append(check("pair_source_hash", protocol["source_hashes"]["phase1223_pairs"] == file_sha256(SOURCE_PAIRS), protocol["source_hashes"]["phase1223_pairs"]))
    checks.append(check("state_source_hash", protocol["source_hashes"]["phase1223_states"] == file_sha256(SOURCE_STATES), protocol["source_hashes"]["phase1223_states"]))
    checks.append(check("phase1226_source_hash", protocol["source_hashes"]["phase1226_main"] == file_sha256(PHASE1226_MAIN), protocol["source_hashes"]["phase1226_main"]))
    checks.append(check("manifest_digest", protocol["material"]["manifest_digest"] == digest(manifest), protocol["material"]["manifest_digest"]))
    checks.append(check("manifest_count", len(manifest) == 32 == protocol["material"]["count"], len(manifest)))
    checks.append(check("split_balance", Counter(row["split"] for row in manifest) == Counter({split: 8 for split in SPLITS}), Counter(row["split"] for row in manifest)))
    checks.append(check("scope_frozen", {row["scope"] for row in manifest} == {PRIMARY_SCOPE}, sorted({row["scope"] for row in manifest})))
    checks.append(check("depth_frozen", protocol["numerical_type"]["depth"] == FROZEN_DEPTH and "midpoint" in protocol["numerical_type"]["depth_selection"], protocol["numerical_type"]))
    checks.append(check("teacher_forced_type", protocol["numerical_type"]["use_cache"] is False and len(protocol["numerical_type"]["scope_exclusions"]) == 3, protocol["numerical_type"]))
    checks.append(check("alliance_basis", protocol["alliances"] == {key: list(value) for key, value in ALLIANCES.items()}, protocol["alliances"]))
    checks.append(check("condition_registry", tuple(protocol["conditions"]) == CONDITIONS, protocol["conditions"]))
    checks.append(check("latent_physical_equality", latent["all_physical_equal"], latent))
    checks.append(check("latent_response_equality", latent["all_response_equal"], latent))
    checks.append(check("latent_claim_withdrawn", "withdrawn" in " ".join(protocol["claim_scope"]).lower() or "downgraded" in " ".join(protocol["claim_scope"]).lower(), protocol["claim_scope"]))
    checks.append(check("temporal_names_forbidden", set(protocol["discovery_rule"]["forbidden_decisions"]) == {"boundary_store", "source_query_joint", "sustained_recompute"}, protocol["discovery_rule"]))
    checks.append(check("manifest_rows_self_digest", all(row["row_digest"] == digest({key: value for key, value in row.items() if key != "row_digest"}) for row in manifest), len(manifest)))
    source_pairs = {row["pair_id"]: row for row in pairs}
    checks.append(check("all_pairs_exist", all(row["pair_id"] in source_pairs for row in manifest), len(source_pairs)))
    checks.append(check("panel_states_exist", all(all(state_id in states for state_id in row["panel_state_ids"].values()) for row in manifest), len(states)))
    checks.append(check("gold_contrast", all(row["recipient_gold"] != row["donor_gold"] for row in manifest), None))
    checks.append(check("four_candidates", all(len(row["candidates"]) == 4 for row in manifest), None))
    checks.append(check("matched_candidate_lengths", all(len({len(value) for value in row["candidate_token_ids"].values()}) == 1 for row in manifest), None))
    checks.append(check("role_positions_in_range", all(
        0 <= int(position) < int(row["panel_prompt_lengths"][panel])
        for row in manifest for panel, positions in row["panel_role_positions"].items() for position in positions.values()
    ), None))
    checks.append(check("formal_outputs_absent", not RECORD_PATH.exists() and not FINAL_PATH.exists(), {"records": RECORD_PATH.exists(), "final": FINAL_PATH.exists()}))
    result: dict[str, Any] = {
        "phase": PHASE,
        "stage": "preaudit",
        "created_at_utc": utc_now(),
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
    }
    result["audit_digest"] = digest(result)
    write_json(PREAUDIT_PATH, result)
    print(canonical_json({"stage": "preaudit", "passed": result["passed_count"], "total": result["check_count"], "all": result["all_checks_passed"], "audit_digest": result["audit_digest"]}))


def centered_cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > EPSILON else 0.0


def canonical_alliance(groups: Iterable[str]) -> str:
    selected = set(groups)
    return "".join(role for role in ("R", "Q", "B") if role in selected)


def permuted_vector(profile: dict[str, float], permutation: tuple[str, str, str]) -> list[float]:
    mapping = dict(zip(("R", "Q", "B"), permutation))
    transformed = {
        canonical_alliance(mapping[role] for role in ALLIANCES[name]): float(value)
        for name, value in profile.items()
    }
    return [transformed[name] for name in ALLIANCES]


def retrieve(observed: dict[str, float], prototype: dict[str, float]) -> tuple[bool, float, float]:
    permutations = list(itertools.permutations(("R", "Q", "B")))
    observed_vector = [observed[name] for name in ALLIANCES]
    scores = [centered_cosine(observed_vector, permuted_vector(prototype, permutation)) for permutation in permutations]
    identity_index = permutations.index(("R", "Q", "B"))
    alternatives = [score for index, score in enumerate(scores) if index != identity_index]
    return scores[identity_index] > max(alternatives) + 1e-6, scores[identity_index], max(alternatives)


def profile_for(row: dict[str, Any]) -> dict[str, float]:
    return {name: float(row["conditions"][f"correct:{name}"]["completion"]) for name in ALLIANCES}


def recompute_split(rows: list[dict[str, Any]], thresholds: dict[str, Any], prototype: dict[str, float] | None = None) -> dict[str, Any]:
    profiles = [profile_for(row) for row in rows]
    profile = {name: median(item[name] for item in profiles) for name in ALLIANCES}
    positive = {name: float(np.mean([item[name] > 0.0 for item in profiles])) for name in ALLIANCES}
    sufficient = [name for name in ALLIANCES if profile[name] >= thresholds["sufficient_completion_median_min"] and positive[name] >= thresholds["sufficient_positive_fraction_min"]]
    if sufficient:
        minimum_size = min(len(ALLIANCES[name]) for name in sufficient)
        minimum_signature = sorted(name for name in sufficient if len(ALLIANCES[name]) == minimum_size)
    else:
        minimum_signature = []
    retrieval: list[tuple[bool, float, float]] = []
    if prototype is None:
        for index, observed in enumerate(profiles):
            other = [item for other_index, item in enumerate(profiles) if other_index != index]
            leave_one_out = {name: median(item[name] for item in other) for name in ALLIANCES}
            retrieval.append(retrieve(observed, leave_one_out))
    else:
        retrieval = [retrieve(observed, prototype) for observed in profiles]
    controls = ("record_order:RQB", "paraphrase:RQB", "equal_norm_reverse:RQB", "identity:RQB", "zero:RQB")
    control_medians = {name: median(row["conditions"][name]["completion"] for row in rows) for name in controls}
    identity_drifts = [
        max(abs(float(row["conditions"]["identity:RQB"]["scores"][key]) - float(row["recipient_scores"][key])) for key in row["recipient_scores"])
        for row in rows
    ]
    result: dict[str, Any] = {
        "count": len(rows),
        "recipient_accuracy": float(np.mean([row["recipient_correct"] for row in rows])),
        "donor_accuracy": float(np.mean([row["donor_correct"] for row in rows])),
        "finite_fraction": float(np.mean([
            bool(row["recipient_finite"] and row["donor_finite"]) and all(condition["finite"] for condition in row["conditions"].values())
            for row in rows
        ])),
        "positive_target_shift_fraction": float(np.mean([float(row["target_shift"]) > 0.0 for row in rows])),
        "median_abs_target_shift": median(abs(float(row["target_shift"])) for row in rows),
        "profile": profile,
        "positive_completion_fraction": positive,
        "profile_range": float(max(profile.values()) - min(profile.values())),
        "sufficient_alliances": sufficient,
        "minimum_signature": minimum_signature,
        "role_retrieval_fraction": float(np.mean([item[0] for item in retrieval])),
        "role_identity_cosine_median": median(item[1] for item in retrieval),
        "role_best_permuted_cosine_median": median(item[2] for item in retrieval),
        "control_medians": control_medians,
        "full_correct_over_max_control": float(profile["RQB"] - max(control_medians.values())),
        "max_identity_score_drift": float(max(identity_drifts)),
        "max_hook_write_abs": float(max(float(condition["hook_write_max_abs"]) for row in rows for condition in row["conditions"].values())),
        "patch_call_values": sorted(set(int(condition["patch_calls"]) for row in rows for condition in row["conditions"].values())),
        "max_causal_prefix_row_spread": float(max(float(value) for row in rows for value in row["causal_prefix_row_spread"].values())),
    }
    if prototype is not None:
        result["profile_cosine_to_discovery"] = centered_cosine([profile[name] for name in ALLIANCES], [prototype[name] for name in ALLIANCES])
    return result


def close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def result_audit() -> None:
    if RESULT_AUDIT_PATH.exists():
        raise RuntimeError("result audit already exists")
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    records = read_jsonl(RECORD_PATH)
    run_summary = read_json(RUN_SUMMARY_PATH)
    final = read_json(FINAL_PATH)
    thresholds = protocol["thresholds"]
    checks: list[dict[str, Any]] = []
    checks.append(check("preaudit_passed", read_json(PREAUDIT_PATH)["all_checks_passed"], None))
    checks.append(check("record_count", len(records) == len(manifest) == 32, len(records)))
    checks.append(check("record_digest", run_summary["record_digest"] == digest(records) == final["record_digest"], run_summary["record_digest"]))
    checks.append(check("run_summary_digest", run_summary["summary_digest"] == digest({key: value for key, value in run_summary.items() if key != "summary_digest"}), run_summary["summary_digest"]))
    checks.append(check("final_self_digest", final["final_digest"] == digest({key: value for key, value in final.items() if key != "final_digest"}), final["final_digest"]))
    checks.append(check("record_self_digests", all(row["record_digest"] == digest({key: value for key, value in row.items() if key != "record_digest"}) for row in records), len(records)))
    checks.append(check("pair_coverage", {row["pair_id"] for row in records} == {row["pair_id"] for row in manifest}, len(records)))
    checks.append(check("split_balance", Counter(row["split"] for row in records) == Counter({split: 8 for split in SPLITS}), Counter(row["split"] for row in records)))
    checks.append(check("depth_exact", {row["depth"] for row in records} == {FROZEN_DEPTH}, sorted({row["depth"] for row in records})))
    checks.append(check("condition_exact", all(set(row["conditions"]) == set(CONDITIONS) for row in records), list(CONDITIONS)))
    checks.append(check("all_finite", all(row["recipient_finite"] and row["donor_finite"] and all(condition["finite"] for condition in row["conditions"].values()) for row in records), None))
    checks.append(check("all_patch_calls_one", all(condition["patch_calls"] == 1 for row in records for condition in row["conditions"].values()), None))
    checks.append(check("all_hook_writes_exact", max(float(condition["hook_write_max_abs"]) for row in records for condition in row["conditions"].values()) == 0.0, None))
    checks.append(check("pure_fp16", set(run_summary["precision_audit"]["parameter_dtypes"]) == {"float16"} and not run_summary["precision_audit"]["has_quantized_modules"] and not run_summary["precision_audit"]["has_bf16_parameters"], run_summary["precision_audit"]))

    by_split = {split: [row for row in records if row["split"] == split] for split in SPLITS}
    discovery = recompute_split(by_split["discovery"], thresholds)
    checks.append(check("discovery_summary_recomputed", close(discovery, final["result"]["discovery"]["summary"]), discovery))
    instrument = {
        "finite": discovery["finite_fraction"] >= thresholds["finite_fraction_min"],
        "patch_calls": discovery["patch_call_values"] == [thresholds["patch_calls_exact"]],
        "hook_write": discovery["max_hook_write_abs"] <= thresholds["hook_write_max_abs_max"],
        "causal_prefix": discovery["max_causal_prefix_row_spread"] <= thresholds["causal_prefix_row_spread_max"],
        "identity": discovery["max_identity_score_drift"] <= thresholds["identity_score_drift_max"],
    }
    behavior = {
        "recipient": discovery["recipient_accuracy"] >= thresholds["recipient_accuracy_min"],
        "donor": discovery["donor_accuracy"] >= thresholds["donor_accuracy_min"],
        "target_sign": discovery["positive_target_shift_fraction"] >= thresholds["positive_target_shift_fraction_min"],
        "target_size": discovery["median_abs_target_shift"] >= thresholds["median_abs_target_shift_min"],
    }
    signature = {
        "has_sufficient_alliance": bool(discovery["minimum_signature"]),
        "profile_range": discovery["profile_range"] >= thresholds["profile_range_min"],
        "leave_one_out_cosine": discovery["role_identity_cosine_median"] >= thresholds["discovery_leave_one_out_cosine_min"],
        "role_label_retrieval": discovery["role_retrieval_fraction"] >= thresholds["role_label_retrieval_fraction_min"],
        "controls": discovery["full_correct_over_max_control"] >= thresholds["full_correct_over_controls_min"],
    }
    discovery_pass = all(instrument.values()) and all(behavior.values()) and all(signature.values())
    checks.append(check("discovery_gates_recomputed", instrument == final["result"]["discovery"]["instrumentation_gates"] and behavior == final["result"]["discovery"]["behavior_gates"] and signature == final["result"]["discovery"]["signature_gates"] and discovery_pass == final["result"]["discovery"]["passed"], {"instrument": instrument, "behavior": behavior, "signature": signature}))

    holdout_passes: list[bool] = []
    for split in HOLDOUT_SPLITS:
        summary = recompute_split(by_split[split], thresholds, prototype=discovery["profile"])
        claimed = final["result"]["holdouts"][split]
        checks.append(check(f"{split}_summary_recomputed", close(summary, claimed["summary"]), summary))
        split_instrument = {
            "finite": summary["finite_fraction"] >= thresholds["finite_fraction_min"],
            "patch_calls": summary["patch_call_values"] == [thresholds["patch_calls_exact"]],
            "hook_write": summary["max_hook_write_abs"] <= thresholds["hook_write_max_abs_max"],
            "causal_prefix": summary["max_causal_prefix_row_spread"] <= thresholds["causal_prefix_row_spread_max"],
            "identity": summary["max_identity_score_drift"] <= thresholds["identity_score_drift_max"],
        }
        split_behavior = {
            "recipient": summary["recipient_accuracy"] >= thresholds["recipient_accuracy_min"],
            "donor": summary["donor_accuracy"] >= thresholds["donor_accuracy_min"],
            "target_sign": summary["positive_target_shift_fraction"] >= thresholds["positive_target_shift_fraction_min"],
            "target_size": summary["median_abs_target_shift"] >= thresholds["median_abs_target_shift_min"],
        }
        split_prediction = {
            "profile_cosine": summary["profile_cosine_to_discovery"] >= thresholds["holdout_profile_cosine_min"],
            "role_label_retrieval": summary["role_retrieval_fraction"] >= thresholds["role_label_retrieval_fraction_min"],
            "minimum_signature": summary["minimum_signature"] == discovery["minimum_signature"],
            "controls": summary["full_correct_over_max_control"] >= thresholds["full_correct_over_controls_min"],
        }
        split_pass = discovery_pass and all(split_instrument.values()) and all(split_behavior.values()) and all(split_prediction.values())
        holdout_passes.append(split_pass)
        checks.append(check(f"{split}_gates_recomputed", split_instrument == claimed["instrumentation_gates"] and split_behavior == claimed["behavior_gates"] and split_prediction == claimed["prediction_gates"] and split_pass == claimed["passed"], {"instrument": split_instrument, "behavior": split_behavior, "prediction": split_prediction}))

    external = discovery_pass and all(holdout_passes)
    checks.append(check("external_gate_recomputed", external == final["result"]["external_validity_gate"], external))
    checks.append(check("decision_recomputed", final["result"]["decision"] == ("SPATIAL_SIGNATURE" if discovery_pass else "ABSTAIN"), final["result"]["decision"]))
    checks.append(check("latent_correction_retained", "physically identical" in final["k_items"][0]["statement"], final["k_items"][0]))
    checks.append(check("no_temporal_overclaim", all(term not in final["k_items"][1]["statement"] for term in ("sustained_recompute", "KV-cache", "autoregressive mechanism")), final["k_items"][1]))
    checks.append(check("auto_continue_zero", final["authorization"]["auto_continue"] == 0 and not final["authorization"]["automatic_execution"], final["authorization"]))

    audit: dict[str, Any] = {
        "phase": PHASE,
        "stage": "result",
        "created_at_utc": utc_now(),
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "recomputed_external_validity_gate": external,
    }
    audit["audit_digest"] = digest(audit)
    write_json(RESULT_AUDIT_PATH, audit)
    print(canonical_json({"stage": "result", "passed": audit["passed_count"], "total": audit["check_count"], "all": audit["all_checks_passed"], "audit_digest": audit["audit_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preaudit", "result"))
    args = parser.parse_args()
    if args.stage == "preaudit":
        preaudit()
    else:
        result_audit()


if __name__ == "__main__":
    main()
