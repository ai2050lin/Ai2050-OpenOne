"""Independent recomputing audit for Phase1270/C018-WP01."""

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
import phase1270_c018_answer_excluded_causal_recomputation as main


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def package(mode: str, checks: dict[str, bool], recomputed: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if recomputed is not None:
        payload["recomputed"] = recomputed
    return payload


def mask_sentinel(rows: list[dict[str, Any]]) -> bool:
    sample = rows[:64]
    masks = {family: main.support_mask(sample, family, torch.device("cpu")) for family in main.FAMILIES}
    no_answer = all(not bool(masks[family][:, 22].any().item()) for family in main.ANALYSIS_FAMILIES)
    expected = {**main.SUPPORT_SIZES, "pre_source_prefix": 4, "answer_only": 1, "causal_suffix_with_answer": 19}
    sizes = all(bool(torch.all(masks[family].sum(1) == size).item()) for family, size in expected.items())
    nested = True
    previous = None
    for family in main.ANALYSIS_FAMILIES:
        current = masks[family]
        if previous is not None:
            nested = nested and bool(torch.all(previous <= current).item())
        previous = current
    dynamic = True
    for index, row in enumerate(sample):
        pair = 12 + 2 * row["codebook_order"].index(row["target_code"])
        dynamic = dynamic and bool(masks["source_map_query_no_answer"][index, pair]) and bool(masks["source_map_query_no_answer"][index, pair + 1])
    controls = (
        bool(torch.all(masks["pre_source_prefix"][:, :4]).item())
        and bool(torch.all(masks["answer_only"][:, 22]).item())
        and bool(torch.all(masks["causal_suffix_with_answer"][:, 4:23]).item())
    )
    return no_answer and sizes and nested and dynamic and controls


def preaudit() -> dict[str, Any]:
    protocol = read_json(main.PROTOCOL)
    rows = read_jsonl(main.MATERIAL)
    predecessor = read_json(main.PHASE1269_FINAL)
    predecessor_audit = read_json(main.PHASE1269_AUDIT)
    counts = {name: sum(row["partition"] == name for row in rows) for name in main.PARTITION_COUNTS}
    all_seeds = [seed for pool in main.SEED_POOLS.values() for seed in pool]
    old_seeds = {seed for pool in main.p1269.SEED_POOLS.values() for seed in pool}
    expected_radius = math.sqrt(
        math.log(2.0 * main.MAX_EVENTS * len(main.FAMILIES) * 2.0 / main.GLOBAL_ERROR_BUDGET)
        / (2.0 * main.SELECTION_DRAWS)
    )
    row_digests = True
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        row_digests = row_digests and main.digest(value) == stored
    checks = {
        "predecessor_authorized": main.PHASE1269_COMPLETE.exists() and predecessor.get("decision") == "causal_support_funnel_confirmed" and predecessor.get("passed") is True,
        "predecessor_audit": predecessor_audit.get("all_checks_passed") is True,
        "contract_exists": main.CONTRACT.exists(),
        "fresh_seed_pools": len(all_seeds) == 15 and len(set(all_seeds)) == 15 and not set(all_seeds).intersection(old_seeds),
        "selection_capacity": main.SELECT_PER_DEPTH == 3 and all(len(pool) == 5 for pool in main.SEED_POOLS.values()),
        "partition_counts": counts == main.PARTITION_COUNTS,
        "row_digests": row_digests,
        "answer_excluded_nested_masks": mask_sentinel([row for row in rows if row["partition"] == "oracle"]),
        "certificate_radius": abs(protocol["thresholds"]["certificate_radius"] - expected_radius) <= 1.0e-15,
        "support_and_control_registry": protocol["analysis_families"] == list(main.ANALYSIS_FAMILIES) and protocol["control_families"] == list(main.CONTROL_FAMILIES),
        "behavior_mechanism_separation": protocol["behavior_selection"]["partition"] == "qualification",
        "source_hash_main": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()),
        "source_hash_auditor": protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "source_hash_contract": protocol["source_hashes"]["contract"] == sha256(main.CONTRACT),
        "protocol_recomputes": protocol["protocol_digest"] == main.protocol_payload(rows)["protocol_digest"],
        "one_run_zero_adaptation": protocol["budgets"]["max_formal_runs"] == 1 and protocol["budgets"]["max_adaptive_rounds"] == 0,
        "no_formal_output": not main.COMPLETE.exists() and not main.FINAL.exists(),
        "no_pretrained": protocol["structured_scope"]["pretrained"] is False and any("No donor compiler or pretrained" in item for item in protocol["hard_stops"]),
    }
    return package("pre", checks)


def qualification_math(records: list[dict[str, Any]], results: list[dict[str, Any]]) -> bool:
    okay = True
    selected_keys: list[tuple[str, int, int]] = []
    for architecture, pool in main.SEED_POOLS.items():
        subset = [record for record in records if record["architecture"] == architecture]
        okay = okay and all(record["pool_index"] == index for index, record in enumerate(subset))
        admitted = 0
        for record in subset:
            training = record["training"]
            expected_pass = (
                min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], record["qualification_accuracy"])
                >= main.THRESHOLDS["behavior_accuracy_min"]
                and record["executor_gap"] <= main.THRESHOLDS["executor_gap_max"]
            )
            expected_selected = expected_pass and admitted < main.SELECT_PER_DEPTH
            okay = okay and record["seed"] == pool[record["pool_index"]]
            okay = okay and record["passed"] == expected_pass and record["selected"] == expected_selected
            if expected_selected:
                selected_keys.append((architecture, record["pool_index"], record["seed"]))
                admitted += 1
        okay = okay and admitted <= main.SELECT_PER_DEPTH
    result_keys = [(row["architecture"], row["pool_index"], row["seed"]) for row in results]
    return okay and result_keys == selected_keys


def model_math(row: dict[str, Any]) -> bool:
    okay = True
    config = main.ARCHITECTURES[row["architecture"]]
    trajectory: list[tuple[int, str]] = []
    for layer in range(config.layers):
        events = [event for event in row["event_ledger"] if event["layer"] == layer]
        okay = okay and [event["family"] for event in events] == list(main.FAMILIES)
        first = None
        for event in events:
            patch = main.bounds(event["sample_patch_accuracy"])
            reverse = main.bounds(event["sample_reverse_accuracy"])
            exact = event["population_score"] >= main.PASS_MIN
            certified = min(patch["lower"], reverse["lower"]) >= main.PASS_MIN
            robust = event["population_score"] >= main.PASS_MIN + main.ROBUST_MULTIPLIER * main.CERTIFICATE_RADIUS
            selected = event["family"] in main.ANALYSIS_FAMILIES and first is None and certified
            if selected:
                first = event["family"]
            okay = okay and all(abs(event["patch_bounds"][key] - patch[key]) <= 1.0e-12 for key in patch)
            okay = okay and all(abs(event["reverse_bounds"][key] - reverse[key]) <= 1.0e-12 for key in reverse)
            okay = okay and event["exact_pass"] == exact and event["certificate_pass"] == certified
            okay = okay and event["robust_actionable"] == robust and event["selected"] == selected
        trajectory.append((layer, first or "abstain"))
    expected_trajectory = [{"layer": layer, "family": family, "size": main.SUPPORT_SIZES.get(family)} for layer, family in trajectory]
    okay = okay and row["trajectory"] == expected_trajectory
    confirmation = {(item["layer"], item["family"]): item for item in row["confirmation_ledger"]}
    okay = okay and len(confirmation) == config.layers * len(main.FAMILIES)
    for item in confirmation.values():
        expected = min(item["patch_accuracy"], item["reverse_accuracy"]) >= main.THRESHOLDS["confirmation_accuracy_min"]
        okay = okay and item["support_passed"] == expected
    expected_selected = []
    for layer, family in trajectory:
        passed = family != "abstain" and confirmation[(layer, family)]["support_passed"]
        expected_selected.append({"layer": layer, "family": family, "passed": passed})
    okay = okay and row["confirmed_selected"] == expected_selected
    final_layer = config.layers - 1
    recompute = [item["layer"] for item in expected_selected if item["layer"] < final_layer and item["passed"]]
    sparse = [item["layer"] for item in expected_selected if item["layer"] < final_layer and item["passed"] and main.SUPPORT_SIZES.get(item["family"], 10_000) <= 6]
    null_pass = all(max(confirmation[(layer, "pre_source_prefix")]["patch_accuracy"], confirmation[(layer, "pre_source_prefix")]["reverse_accuracy"]) <= main.NULL_MAX for layer in range(config.layers))
    suffix_pass = all(min(confirmation[(layer, "causal_suffix_with_answer")]["patch_accuracy"], confirmation[(layer, "causal_suffix_with_answer")]["reverse_accuracy"]) >= main.POSITIVE_MIN for layer in range(config.layers))
    terminal_no_answer = max(confirmation[(final_layer, "causal_prefix_no_answer")]["patch_accuracy"], confirmation[(final_layer, "causal_prefix_no_answer")]["reverse_accuracy"]) <= main.NULL_MAX
    terminal_answer = min(confirmation[(final_layer, "answer_only")]["patch_accuracy"], confirmation[(final_layer, "answer_only")]["reverse_accuracy"]) >= main.POSITIVE_MIN
    controls = null_pass and suffix_pass and terminal_no_answer and terminal_answer
    last = max(recompute) if recompute else None
    okay = okay and row["recompute_layers"] == recompute and row["sparse_recompute_layers"] == sparse
    okay = okay and row["last_recompute_layer"] == last
    okay = okay and row["null_control_passed"] == null_pass and row["with_answer_suffix_control_passed"] == suffix_pass
    okay = okay and row["terminal_no_answer_negative"] == terminal_no_answer and row["terminal_answer_replay_positive"] == terminal_answer
    okay = okay and row["controls_passed"] == controls
    okay = okay and row["answer_free_recompute_passed"] == (bool(recompute) and controls)
    okay = okay and row["sparse_recompute_passed"] == (bool(sparse) and controls)
    return okay


def final_audit() -> dict[str, Any]:
    protocol = read_json(main.PROTOCOL)
    complete = read_json(main.COMPLETE)
    run_summary = read_json(main.SUMMARY)
    qualification = read_jsonl(main.QUALIFICATION)
    results = read_jsonl(main.MODELS)
    final = read_json(main.FINAL)
    recomputed = main.summarize(qualification, results)
    without_digest = dict(final)
    stored_digest = without_digest.pop("final_digest")
    checks = {
        "formal_marker": complete.get("status") == "formal_run_complete",
        "qualification_selection_math": qualification_math(qualification, results),
        "selected_model_count": len(results) == run_summary.get("selected_models") == final.get("selected_models"),
        "all_model_event_and_control_math": all(model_math(row) for row in results),
        "qualification_hash": run_summary.get("qualification_hash") == sha256(main.QUALIFICATION) == final.get("qualification_hash"),
        "models_hash": run_summary.get("models_hash") == sha256(main.MODELS) == final.get("models_hash"),
        "run_digest": complete.get("run_digest") == main.digest({"qualification": qualification, "models": results}) == final.get("run_digest"),
        "protocol_digest": run_summary.get("protocol_digest") == protocol.get("protocol_digest") == final.get("protocol_digest"),
        "source_hashes": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()) and protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "summary_recomputed": all(final.get(key) == value for key, value in recomputed.items()),
        "authorization_typed": final["authorization"]["necessity_contract_design"] == final["passed"] and final["authorization"]["independent_sparse_donor_design"] == (final["passed"] and final["sparse_recompute_breadth_passed"]),
        "no_pretrained": run_summary.get("pretrained_model_loaded") is False and final["authorization"]["automatic_pretrained_run"] is False,
        "structured_scope": final.get("structured_scope") == protocol.get("structured_scope") and final["structured_scope"]["natural_language"] is False,
        "final_digest": stored_digest == main.digest(without_digest),
    }
    return package("final", checks, recomputed)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    payload = preaudit() if args.mode == "pre" else final_audit()
    target = main.PREAUDIT if args.mode == "pre" else main.FINAL_AUDIT
    write(target, payload)
    print(json.dumps({"mode": args.mode, "checks": f"{payload['passed_checks']}/{payload['total_checks']}", "passed": payload["all_checks_passed"]}))
    raise SystemExit(0 if payload["all_checks_passed"] else 1)


if __name__ == "__main__":
    cli()
