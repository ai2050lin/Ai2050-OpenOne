"""Independent recomputing audit for Phase1271/C019."""

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
import phase1271_c019_cross_layer_micro_write_trajectory as main
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer


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


def executor_sentinel(rows: list[dict[str, Any]]) -> bool:
    torch.manual_seed(1_271_999_991)
    config = ModelConfig(layers=2, width=32, heads=4, mlp_width=64, max_length=23, vocab_size=22)
    model = TinyCausalTransformer(config).eval()
    ids01 = torch.tensor([row["h01_ids"] for row in rows[:16]])
    ids11 = torch.tensor([row["h11_ids"] for row in rows[:16]])
    trace01 = main.capture_micro(model, ids01)
    trace11 = main.capture_micro(model, ids11)
    dummy = {"stage": "attn_write", "layers": [], "position": 22}
    explicit = main.forward_program(model, ids01, trace11, dummy)
    native = model(ids01)
    prefix_equal = all(torch.max(torch.abs(trace01[layer]["attn_write"][:, 2] - trace11[layer]["attn_write"][:, 2])).item() <= 1.0e-6 for layer in range(config.layers))
    return bool(torch.max(torch.abs(explicit - native)).item() <= 1.0e-5 and prefix_equal)


def registry_sentinel() -> bool:
    okay = True
    for name, config in main.ARCHITECTURES.items():
        programs = main.program_registry(config.layers)
        prefixes = [program for program in programs if program["role"] == "attention_prefix"]
        okay = okay and len(programs) == config.layers + 6 and len(prefixes) == config.layers
        okay = okay and [program["prefix_end"] for program in prefixes] == list(range(config.layers))
        okay = okay and all(program["layers"] == list(range(program["prefix_end"] + 1)) for program in prefixes)
        okay = okay and prefixes[-1]["layers"] == list(range(config.layers))
        okay = okay and {program["name"] for program in programs if program["role"] != "attention_prefix"} == {
            "attn_full_wrong", "mlp_full_correct", "mlp_full_wrong", "after_block_full_correct", "attn_pre_source_null", "attn_position8_descriptive"
        }
    return okay


def preaudit() -> dict[str, Any]:
    protocol = read_json(main.PROTOCOL)
    rows = read_jsonl(main.MATERIAL)
    predecessor = read_json(main.PHASE1270_FINAL)
    predecessor_audit = read_json(main.PHASE1270_AUDIT)
    counts = {name: sum(row["partition"] == name for row in rows) for name in main.PARTITION_COUNTS}
    seeds = [seed for pool in main.SEED_POOLS.values() for seed in pool]
    prior_seeds = {seed for pool in main.p1270.SEED_POOLS.values() for seed in pool}
    expected_radius = math.sqrt(
        math.log(2.0 * main.MAX_EVENTS * 3.0 / main.GLOBAL_ERROR_BUDGET) / (2.0 * main.SELECTION_DRAWS)
    )
    row_digests = True
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        row_digests = row_digests and main.digest(value) == stored
    checks = {
        "predecessor_complete_boundary": main.PHASE1270_COMPLETE.exists() and predecessor.get("decision") == "answer_excluded_upstream_recomputation_not_confirmed" and predecessor.get("passed") is False,
        "predecessor_audit": predecessor_audit.get("all_checks_passed") is True,
        "contract_exists": main.CONTRACT.exists(),
        "fresh_formal_seeds": len(seeds) == 15 and len(set(seeds)) == 15 and not set(seeds).intersection(prior_seeds) and not set(seeds).intersection(main.DEVELOPMENT_SEEDS),
        "selection_capacity": main.SELECT_PER_DEPTH == 3 and all(len(pool) == 5 for pool in main.SEED_POOLS.values()),
        "partition_counts": counts == main.PARTITION_COUNTS,
        "row_digests": row_digests,
        "program_registry": registry_sentinel(),
        "micro_executor_and_prefix_null": executor_sentinel([row for row in rows if row["partition"] == "oracle"]),
        "certificate_radius": abs(protocol["thresholds"]["certificate_radius"] - expected_radius) <= 1.0e-15,
        "development_excluded": protocol["development_seeds_excluded"] == sorted(main.DEVELOPMENT_SEEDS),
        "source_hash_main": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()),
        "source_hash_auditor": protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "source_hash_contract": protocol["source_hashes"]["contract"] == sha256(main.CONTRACT),
        "protocol_recomputes": protocol["protocol_digest"] == main.protocol_payload(rows)["protocol_digest"],
        "one_run_zero_adaptation": protocol["budgets"]["max_formal_runs"] == 1 and protocol["budgets"]["max_adaptive_rounds"] == 0,
        "no_formal_output": not main.COMPLETE.exists() and not main.FINAL.exists(),
        "no_pretrained": protocol["structured_scope"]["pretrained"] is False and any("No head search" in item for item in protocol["hard_stops"]),
    }
    return package("pre", checks)


def qualification_math(records: list[dict[str, Any]], results: list[dict[str, Any]]) -> bool:
    okay = True
    selected_keys = []
    for architecture, pool in main.SEED_POOLS.items():
        subset = [record for record in records if record["architecture"] == architecture]
        admitted = 0
        okay = okay and all(record["pool_index"] == index for index, record in enumerate(subset))
        for record in subset:
            training = record["training"]
            expected_pass = min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], record["qualification_accuracy"]) >= main.THRESHOLDS["behavior_accuracy_min"] and record["executor_gap"] <= main.THRESHOLDS["executor_gap_max"]
            expected_selected = expected_pass and admitted < main.SELECT_PER_DEPTH
            okay = okay and record["seed"] == pool[record["pool_index"]]
            okay = okay and record["passed"] == expected_pass and record["selected"] == expected_selected
            if expected_selected:
                selected_keys.append((architecture, record["pool_index"], record["seed"]))
                admitted += 1
    return okay and [(row["architecture"], row["pool_index"], row["seed"]) for row in results] == selected_keys


def ledger_math(items: list[dict[str, Any]]) -> bool:
    okay = True
    for item in items:
        patch = main.bounds(item["sample_patch_expected"])
        reverse = main.bounds(item["sample_reverse_base"])
        false_target = main.bounds(item["sample_patch_false_target"])
        exact = item["population_expected_score"] >= main.PASS_MIN
        certified = min(patch["lower"], reverse["lower"]) >= main.PASS_MIN
        robust = item["population_expected_score"] >= main.PASS_MIN + main.ROBUST_MULTIPLIER * main.CERTIFICATE_RADIUS
        wrong = item["role"] in ("wrong_identity", "matched_component_wrong")
        specificity = certified and (not wrong or false_target["upper"] <= main.NULL_MAX)
        okay = okay and all(abs(item["patch_bounds"][key] - patch[key]) <= 1.0e-12 for key in patch)
        okay = okay and all(abs(item["reverse_bounds"][key] - reverse[key]) <= 1.0e-12 for key in reverse)
        okay = okay and all(abs(item["false_target_bounds"][key] - false_target[key]) <= 1.0e-12 for key in false_target)
        okay = okay and item["exact_pass"] == exact and item["certificate_pass"] == certified
        okay = okay and item["robust_actionable"] == robust and item["specificity_pass"] == specificity
    return okay


def model_math(row: dict[str, Any]) -> bool:
    config = main.ARCHITECTURES[row["architecture"]]
    programs = main.program_registry(config.layers)
    okay = [item["name"] for item in row["event_ledger"]] == [program["name"] for program in programs]
    okay = okay and [item["name"] for item in row["confirmation_ledger"]] == [program["name"] for program in programs]
    okay = okay and ledger_math(row["event_ledger"]) and ledger_math(row["confirmation_ledger"])
    event = {item["name"]: item for item in row["event_ledger"]}
    confirmation = {item["name"]: item for item in row["confirmation_ledger"]}
    prefixes = [event[f"attn_prefix_{end}"] for end in range(config.layers)]
    selected = next((item["name"] for item in prefixes if item["certificate_pass"]), None)
    selected_end = next((item["prefix_end"] for item in prefixes if item["name"] == selected), None)
    full_name = f"attn_prefix_{config.layers - 1}"
    controls = confirmation["after_block_full_correct"]["population_expected_score"] >= main.POSITIVE_MIN and max(confirmation["attn_pre_source_null"]["population_patch_expected"], confirmation["attn_pre_source_null"]["population_reverse_base"]) <= main.NULL_MAX
    teacher = event[full_name]["certificate_pass"] and confirmation[full_name]["exact_pass"] and event["attn_full_wrong"]["specificity_pass"] and confirmation["attn_full_wrong"]["exact_pass"] and confirmation["attn_full_wrong"]["population_patch_false_target"] <= main.NULL_MAX and controls
    proper = teacher and selected_end is not None and selected_end < config.layers - 1 and confirmation[selected]["exact_pass"]
    advantage = confirmation[full_name]["population_expected_score"] - confirmation["mlp_full_correct"]["population_expected_score"]
    okay = okay and row["selected_prefix"] == selected and row["selected_prefix_end"] == selected_end
    okay = okay and row["controls_passed"] == controls and row["teacher_forced_attention_passed"] == teacher and row["proper_prefix_passed"] == proper
    okay = okay and abs(row["attention_over_mlp_advantage"] - advantage) <= 1.0e-12 and row["component_advantage_passed"] == (advantage >= main.ADVANTAGE_MIN)
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
        "all_model_micro_event_math": all(model_math(row) for row in results),
        "qualification_hash": run_summary.get("qualification_hash") == sha256(main.QUALIFICATION) == final.get("qualification_hash"),
        "models_hash": run_summary.get("models_hash") == sha256(main.MODELS) == final.get("models_hash"),
        "run_digest": complete.get("run_digest") == main.digest({"qualification": qualification, "models": results}) == final.get("run_digest"),
        "protocol_digest": run_summary.get("protocol_digest") == protocol.get("protocol_digest") == final.get("protocol_digest"),
        "source_hashes": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()) and protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "summary_recomputed": all(final.get(key) == value for key, value in recomputed.items()),
        "authorization_typed": final["authorization"]["layer_coalition_minimality_contract"] == final["passed"] and final["authorization"]["self_sustaining_prefix_contract"] == (final["passed"] and final["proper_prefix_breadth_passed"]),
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
