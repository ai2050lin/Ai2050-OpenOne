"""Independent, recomputing audit for Phase1269/C017."""

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
import phase1269_c017_causal_support_funnel_confirmation as main


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


def support_mask_sentinel(rows: list[dict[str, Any]]) -> bool:
    sample = rows[:64]
    families = main.ANALYSIS_FAMILIES + (main.CONTROL_FAMILY,)
    masks = {family: main.p1268.support_mask(sample, family, torch.device("cpu")) for family in families}
    previous = None
    nested = True
    for family in main.ANALYSIS_FAMILIES:
        current = masks[family]
        if previous is not None:
            nested = nested and bool(torch.all(previous <= current).item())
        previous = current
    sizes = all(
        bool(torch.all(masks[family].sum(1) == size).item())
        for family, size in {**main.SUPPORT_SIZES, main.CONTROL_FAMILY: 23}.items()
    )
    dynamic = True
    for index, row in enumerate(sample):
        pair = 12 + 2 * row["codebook_order"].index(row["target_code"])
        dynamic = dynamic and bool(masks["source_map_query"][index, pair]) and bool(masks["source_map_query"][index, pair + 1])
    return nested and sizes and dynamic


def preaudit() -> dict[str, Any]:
    protocol = read_json(main.PROTOCOL)
    rows = read_jsonl(main.MATERIAL)
    predecessor = read_json(main.PHASE1268_FINAL)
    predecessor_audit = read_json(main.PHASE1268_AUDIT)
    predecessor_erratum = read_json(main.PHASE1268_ERRATUM)
    counts = {name: sum(row["partition"] == name for row in rows) for name in main.PARTITION_COUNTS}
    all_seeds = [seed for pool in main.SEED_POOLS.values() for seed in pool]
    old_seeds = set(main.p1268.MODEL_SEEDS.values())
    expected_radius = math.sqrt(
        math.log(2.0 * main.MAX_EVENTS * (len(main.ANALYSIS_FAMILIES) + 1) * 2.0 / main.GLOBAL_ERROR_BUDGET)
        / (2.0 * main.SELECTION_DRAWS)
    )
    row_digests = True
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        row_digests = row_digests and main.digest(value) == stored
    checks = {
        "predecessor_formal_negative": main.PHASE1268_COMPLETE.exists() and predecessor.get("passed") is False,
        "predecessor_audit_ledger": predecessor_audit.get("passed_checks") == 12 and predecessor_audit.get("total_checks") == 13 and predecessor_erratum.get("passed") is True,
        "contract_exists": main.CONTRACT.exists(),
        "frozen_seed_pools": len(all_seeds) == 15 and len(set(all_seeds)) == 15 and not set(all_seeds).intersection(old_seeds),
        "five_candidates_three_selected_per_depth": all(len(pool) == 5 for pool in main.SEED_POOLS.values()) and main.SELECT_PER_DEPTH == 3,
        "partition_counts": counts == main.PARTITION_COUNTS,
        "row_digests": row_digests,
        "behavior_mechanism_partition_separation": protocol["behavior_selection"]["qualification_partition"] == "qualification" and set(main.PARTITION_COUNTS) == {"qualification", "oracle", "confirmation"},
        "support_order_and_sizes": protocol["analysis_families"] == list(main.ANALYSIS_FAMILIES) and protocol["support_sizes"] == main.SUPPORT_SIZES,
        "support_masks_nested_dynamic": support_mask_sentinel([row for row in rows if row["partition"] == "oracle"]),
        "certificate_radius": abs(protocol["thresholds"]["certificate_radius"] - expected_radius) <= 1.0e-15,
        "funnel_definition_complete": set(protocol["funnel_definition"]) == {"complete_trajectory", "independent_confirmation", "monotone", "strict", "early_distributed", "terminal"},
        "source_hash_main": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()),
        "source_hash_auditor": protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "source_hash_contract": protocol["source_hashes"]["contract"] == sha256(main.CONTRACT),
        "protocol_recomputes": protocol["protocol_digest"] == main.protocol_payload(rows)["protocol_digest"],
        "one_run_zero_adaptation": protocol["budgets"]["max_formal_runs"] == 1 and protocol["budgets"]["max_adaptive_rounds"] == 0,
        "no_formal_output": not main.COMPLETE.exists() and not main.FINAL.exists(),
        "no_donor_or_pretrained": protocol["structured_scope"]["pretrained"] is False and any("No donor compiler" in item for item in protocol["hard_stops"]),
    }
    return package("pre", checks)


def qualification_math(records: list[dict[str, Any]], results: list[dict[str, Any]]) -> bool:
    okay = True
    selected_keys: list[tuple[str, int, int]] = []
    for architecture, pool in main.SEED_POOLS.items():
        subset = [record for record in records if record["architecture"] == architecture]
        okay = okay and all(record["pool_index"] == index for index, record in enumerate(subset))
        okay = okay and all(record["seed"] == pool[record["pool_index"]] for record in subset)
        admitted = 0
        for record in subset:
            training = record["training"]
            expected_pass = (
                min(
                    training["accuracy_overall"],
                    training["accuracy_direct"],
                    training["accuracy_code"],
                    record["qualification_accuracy"],
                )
                >= main.THRESHOLDS["behavior_accuracy_min"]
                and record["executor_gap"] <= main.THRESHOLDS["executor_gap_max"]
            )
            expected_selected = expected_pass and admitted < main.SELECT_PER_DEPTH
            okay = okay and record["passed"] == expected_pass and record["selected"] == expected_selected
            if expected_selected:
                selected_keys.append((architecture, record["pool_index"], record["seed"]))
                admitted += 1
        okay = okay and admitted <= main.SELECT_PER_DEPTH
        if admitted == main.SELECT_PER_DEPTH:
            okay = okay and len(subset) == subset[-1]["pool_index"] + 1
    result_keys = [(row["architecture"], row["pool_index"], row["seed"]) for row in results]
    rejected = {(record["architecture"], record["pool_index"], record["seed"]) for record in records if not record["selected"]}
    return okay and result_keys == selected_keys and not rejected.intersection(result_keys)


def event_and_funnel_math(results: list[dict[str, Any]]) -> bool:
    okay = True
    families = main.ANALYSIS_FAMILIES + (main.CONTROL_FAMILY,)
    for row in results:
        config = main.ARCHITECTURES[row["architecture"]]
        chosen: list[tuple[int, str]] = []
        for layer in range(config.layers):
            events = [event for event in row["event_ledger"] if event["layer"] == layer]
            okay = okay and [event["family"] for event in events] == list(families)
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
            chosen.append((layer, first or "abstain"))
        expected_trajectory = [
            {"layer": layer, "family": family, "size": main.SUPPORT_SIZES.get(family)}
            for layer, family in chosen
        ]
        okay = okay and row["trajectory"] == expected_trajectory
        indices = [main.ANALYSIS_FAMILIES.index(family) if family in main.ANALYSIS_FAMILIES else None for _layer, family in chosen]
        sizes = [main.SUPPORT_SIZES.get(family) for _layer, family in chosen]
        complete = all(value is not None for value in indices)
        monotone = complete and all(indices[index + 1] <= indices[index] for index in range(len(indices) - 1))
        strict = complete and any(indices[index + 1] < indices[index] for index in range(len(indices) - 1))
        early = complete and indices[0] > 0
        terminal = complete and indices[-1] == 0
        confirmation = True
        okay = okay and [(item["layer"], item["family"]) for item in row["confirmations"]] == chosen
        for item in row["confirmations"]:
            expected_pass = min(item["patch_accuracy"], item["reverse_accuracy"]) >= main.THRESHOLDS["confirmation_accuracy_min"]
            okay = okay and item["passed"] == expected_pass
            confirmation = confirmation and expected_pass
        funnel = complete and monotone and strict and early and terminal and confirmation
        onset = next((layer for layer, family in chosen if family == "answer_only"), None)
        okay = okay and row["support_indices"] == indices and row["support_sizes"] == sizes
        okay = okay and row["complete"] == complete and row["monotone_nonincreasing"] == monotone
        okay = okay and row["strict_contraction"] == strict and row["early_distributed"] == early
        okay = okay and row["terminal_answer_only"] == terminal and row["confirmation_passed"] == confirmation
        okay = okay and row["answer_onset_layer"] == onset and row["funnel_passed"] == funnel
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
        "selected_model_count_consistent": len(results) == run_summary.get("selected_models") == final.get("selected_models"),
        "selected_only_have_hidden_results": all(row["qualification"]["passed"] for row in results),
        "event_and_funnel_math": event_and_funnel_math(results),
        "qualification_hash": run_summary.get("qualification_hash") == sha256(main.QUALIFICATION) == final.get("qualification_hash"),
        "models_hash": run_summary.get("models_hash") == sha256(main.MODELS) == final.get("models_hash"),
        "run_digest": complete.get("run_digest") == main.digest({"qualification": qualification, "models": results}) == final.get("run_digest"),
        "protocol_digest": run_summary.get("protocol_digest") == protocol.get("protocol_digest") == final.get("protocol_digest"),
        "source_hashes": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()) and protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "summary_recomputed": all(final.get(key) == value for key, value in recomputed.items()),
        "decision_and_authorization": final["authorization"]["distributed_donor_contract_design"] == final["passed"] and final["authorization"]["automatic_pretrained_run"] is False,
        "no_pretrained_loaded": run_summary.get("pretrained_model_loaded") is False and all(final["authorization"][name] is False for name in ("qwen3", "glm4", "ds7b")),
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
