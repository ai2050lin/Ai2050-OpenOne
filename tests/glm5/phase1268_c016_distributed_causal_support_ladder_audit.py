"""Independent audit for Phase1268/C016."""

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
import phase1268_c016_distributed_causal_support_ladder as main
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


def support_mask_sentinel(rows: list[dict[str, Any]]) -> bool:
    sample = rows[:32]
    masks = {family: main.support_mask(sample, family, torch.device("cpu")) for family in main.FAMILIES}
    nested = True
    previous = None
    for family in main.SPARSE_FAMILIES:
        current = masks[family]
        if previous is not None:
            nested = nested and bool(torch.all(previous <= current).item())
        previous = current
    exact_sizes = (
        torch.all(masks["answer_only"].sum(1) == 1)
        and torch.all(masks["query_triplet"].sum(1) == 3)
        and torch.all(masks["source_query"].sum(1) == 4)
        and torch.all(masks["source_map_query"].sum(1) == 6)
        and torch.all(masks["semantic_chain"].sum(1) == 7)
        and torch.all(masks["causal_suffix"].sum(1) == 19)
        and torch.all(masks["full_sequence"].sum(1) == 23)
    )
    dynamic_pairs = True
    for index, row in enumerate(sample):
        pair = 12 + 2 * row["codebook_order"].index(row["target_code"])
        dynamic_pairs = dynamic_pairs and bool(masks["source_map_query"][index, pair]) and bool(masks["source_map_query"][index, pair + 1])
    return nested and bool(exact_sizes) and dynamic_pairs


def executor_sentinel() -> bool:
    torch.manual_seed(1_268_999)
    config = ModelConfig(layers=2, width=32, heads=4, mlp_width=64, max_length=23, vocab_size=22)
    model = TinyCausalTransformer(config).eval()
    ids = torch.randint(0, 22, (8, 23))
    native = model(ids)
    explicit, trace = main.full_residual_forward(model, ids, capture=True)
    if trace is None or trace.shape != (8, 2, 23, 32):
        return False
    return bool(torch.max(torch.abs(native - explicit)).item() <= 1.0e-5)


def preaudit() -> dict[str, Any]:
    protocol = read_json(main.PROTOCOL)
    rows = read_jsonl(main.MATERIAL)
    predecessor = read_json(main.PHASE1267_FINAL)
    predecessor_audit = read_json(main.PHASE1267_AUDIT)
    counts = {name: sum(row["partition"] == name for row in rows) for name in main.PARTITION_COUNTS}
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
        "predecessor_complete_negative": main.PHASE1267_COMPLETE.exists() and predecessor.get("decision") == "registered_hierarchy_insufficient" and predecessor.get("passed") is False,
        "predecessor_audit": predecessor_audit.get("all_checks_passed") is True,
        "contract_exists": main.CONTRACT.exists(),
        "nine_fresh_seeds": len(main.MODEL_SEEDS) == 9 and len(set(main.MODEL_SEEDS.values())) == 9 and not set(main.MODEL_SEEDS.values()).intersection(p1267_seeds()),
        "three_replicates_per_depth": all(sum(key.startswith(name) for key in main.MODEL_SEEDS) == 3 for name in main.ARCHITECTURES),
        "partition_counts": counts == main.PARTITION_COUNTS,
        "row_digests": row_digests,
        "support_order": protocol.get("support_order") == list(main.FAMILIES),
        "positive_controls_separate": set(main.POSITIVE_CONTROLS).isdisjoint(main.SPARSE_FAMILIES),
        "support_masks_nested_and_dynamic": support_mask_sentinel([row for row in rows if row["partition"] == "oracle"]),
        "full_executor_matches_native": executor_sentinel(),
        "certificate_radius": abs(protocol["thresholds"]["certificate_radius"] - expected_radius) <= 1.0e-15,
        "source_hash_main": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()),
        "source_hash_auditor": protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "source_hash_contract": protocol["source_hashes"]["contract"] == sha256(main.CONTRACT),
        "protocol_recomputes": protocol["protocol_digest"] == main.protocol_payload(rows)["protocol_digest"],
        "structured_scope": protocol["structured_scope"]["natural_language"] is False and protocol["structured_scope"]["pretrained"] is False,
        "no_formal_output": not main.COMPLETE.exists() and not main.FINAL.exists(),
        "one_run_zero_adaptation": protocol["budgets"]["max_formal_runs"] == 1 and protocol["budgets"]["max_adaptive_rounds"] == 0,
        "no_donor_or_pretrained": "No donor compiler is fit unless a separate future contract is authorized." in protocol["hard_stops"] and "No pretrained model is loaded." in protocol["hard_stops"],
    }
    return package("pre", checks)


def p1267_seeds() -> set[int]:
    return set(main.p1267.MODEL_SEEDS.values())


def package(mode: str, checks: dict[str, bool], recomputed: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
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


def event_math(results: list[dict[str, Any]]) -> bool:
    okay = True
    for row in results:
        first_by_layer: dict[int, str] = {}
        for event in row.get("event_ledger", []):
            patch = main.bounds(event["sample_patch_accuracy"])
            reverse = main.bounds(event["sample_reverse_accuracy"])
            okay = okay and all(abs(event["patch_bounds"][key] - patch[key]) <= 1.0e-12 for key in patch)
            okay = okay and all(abs(event["reverse_bounds"][key] - reverse[key]) <= 1.0e-12 for key in reverse)
            exact = event["population_score"] >= main.PASS_MIN
            certified = min(patch["lower"], reverse["lower"]) >= main.PASS_MIN
            robust = event["population_score"] >= main.PASS_MIN + main.ROBUST_MULTIPLIER * main.CERTIFICATE_RADIUS
            okay = okay and event["exact_pass"] == exact and event["certificate_pass"] == certified and event["robust_actionable"] == robust
            selected = False
            if event["family"] in main.SPARSE_FAMILIES and event["layer"] not in first_by_layer and certified:
                first_by_layer[event["layer"]] = event["family"]
                selected = True
            okay = okay and event["selected_sparse"] == selected
        expected = [{"layer": layer, "family": family} for layer, family in sorted(first_by_layer.items())]
        okay = okay and row.get("selected_events") == expected
        targets = []
        tuples = [(item["layer"], item["family"]) for item in expected]
        if tuples:
            minimum = min(tuples, key=lambda item: (main.SPARSE_FAMILIES.index(item[1]), item[0]))
            earliest = min(tuples, key=lambda item: item[0])
            latest = max(tuples, key=lambda item: item[0])
            for item in (minimum, earliest, latest):
                value = {"layer": item[0], "family": item[1]}
                if value not in targets:
                    targets.append(value)
        okay = okay and row.get("confirmation_targets") == targets
        for item in row.get("confirmations", []):
            okay = okay and item["passed"] == (min(item["patch_accuracy"], item["reverse_accuracy"]) >= main.THRESHOLDS["confirmation_accuracy_min"])
        passed = [item for item in row.get("confirmations", []) if item["passed"]]
        ceiling = min((main.SPARSE_FAMILIES.index(item["family"]) for item in passed), default=None)
        okay = okay and row.get("support_ceiling_index") == ceiling
    return okay


def final_audit() -> dict[str, Any]:
    protocol = read_json(main.PROTOCOL)
    complete = read_json(main.COMPLETE)
    run_summary = read_json(main.SUMMARY)
    results = read_jsonl(main.MODELS)
    final = read_json(main.FINAL)
    recomputed = main.summarize(results)
    without_digest = dict(final)
    stored_digest = without_digest.pop("final_digest")
    checks = {
        "formal_marker": complete.get("status") == "formal_run_complete",
        "model_count": len(results) == 9 and run_summary.get("models") == 9,
        "models_hash": run_summary.get("models_hash") == sha256(main.MODELS) == final.get("models_hash"),
        "run_digest": complete.get("run_digest") == main.digest(results) == final.get("run_digest"),
        "protocol_digest": run_summary.get("protocol_digest") == protocol.get("protocol_digest") == final.get("protocol_digest"),
        "source_hashes": protocol["source_hashes"]["main"] == sha256(Path(main.__file__).resolve()) and protocol["source_hashes"]["auditor"] == sha256(Path(__file__).resolve()),
        "event_and_selection_math": event_math(results),
        "summary_recomputed": all(final.get(key) == value for key, value in recomputed.items()),
        "decision_registered": final.get("decision").startswith("distributed_sparse_support_identified:") or final.get("decision") in {"only_trivial_causal_suffix_sufficient", "state_replacement_executor_invalid"},
        "positive_controls_not_authorization": final["authorization"]["distributed_donor_contract_design"] == final["passed"],
        "no_pretrained_loaded": run_summary.get("pretrained_model_loaded") is False and final["authorization"]["automatic_pretrained_run"] is False,
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
