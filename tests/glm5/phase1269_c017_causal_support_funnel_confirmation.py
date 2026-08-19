"""Phase1269: prospective confirmation of a depth-wise causal-support funnel.

Behavior qualification and mechanism measurement are separated.  For each
depth, the first three behavior-qualified models from a frozen five-seed pool
are admitted.  On untouched exhaustive mechanism worlds, every layer receives
the cumulative support ladder from Phase1268.  The smallest certified family,
including causal_suffix as a non-sparse ceiling, defines a complete support
trajectory.  A funnel requires independent confirmation, monotone nonincrease,
a strict contraction, and final answer-only concentration.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1266_c014_free_transformer_population_certificate as p1266
import phase1268_c016_distributed_causal_support_ladder as p1268


PHASE = 1269
CAMPAIGN = "C017"
CONTRACT_ID = "EXP-C017-WP01-001"
OUT = ROOT / "tests/glm5/result/phase1269_c017_causal_support_funnel_confirmation"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_funnel_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
QUALIFICATION = OUT / "raw/behavior_qualification.jsonl"
MODELS = OUT / "raw/model_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1269_c017_causal_support_funnel_confirmation_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C017-WP01-001.json"
PHASE1268_FINAL = ROOT / "tests/glm5/result/phase1268_c016_distributed_causal_support_ladder/analysis/final.json"
PHASE1268_AUDIT = ROOT / "tests/glm5/result/phase1268_c016_distributed_causal_support_ladder/audit/independent_final_audit.json"
PHASE1268_ERRATUM = ROOT / "tests/glm5/result/phase1268_c016_distributed_causal_support_ladder/audit/independent_final_audit_erratum.json"
PHASE1268_COMPLETE = ROOT / "tests/glm5/result/phase1268_c016_distributed_causal_support_ladder/raw/FORMAL_RUN_COMPLETE.json"

ARCHITECTURES = p1266.ARCHITECTURES
SEED_POOLS = {
    "shallow4": [1_269_401_001, 1_269_401_101, 1_269_401_201, 1_269_401_301, 1_269_401_401],
    "middle6": [1_269_601_001, 1_269_601_101, 1_269_601_201, 1_269_601_301, 1_269_601_401],
    "deep8": [1_269_801_001, 1_269_801_101, 1_269_801_201, 1_269_801_301, 1_269_801_401],
}
SELECT_PER_DEPTH = 3
MATERIAL_SEEDS = {"qualification": 1_269_910_001, "confirmation": 1_269_930_001}
PARTITION_COUNTS = {"qualification": 1024, "oracle": 3456, "confirmation": 1024}
ANALYSIS_FAMILIES = p1268.SPARSE_FAMILIES + ("causal_suffix",)
CONTROL_FAMILY = "full_sequence"
SUPPORT_SIZES = {
    "answer_only": 1,
    "query_triplet": 3,
    "source_query": 4,
    "source_map_query": 6,
    "semantic_chain": 7,
    "causal_suffix": 19,
}
SELECTION_DRAWS = 32768
GLOBAL_ERROR_BUDGET = 0.01
PASS_MIN = 0.95
ROBUST_MULTIPLIER = 2.0
MAX_MODELS = len(ARCHITECTURES) * SELECT_PER_DEPTH
MAX_EVENTS = sum(config.layers for config in ARCHITECTURES.values()) * SELECT_PER_DEPTH
CERTIFICATE_RADIUS = math.sqrt(
    math.log(2.0 * MAX_EVENTS * (len(ANALYSIS_FAMILIES) + 1) * 2.0 / GLOBAL_ERROR_BUDGET)
    / (2.0 * SELECTION_DRAWS)
)
THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "executor_gap_max": 2.0e-4,
    "population_support_accuracy_min": PASS_MIN,
    "certificate_radius": CERTIFICATE_RADIUS,
    "certificate_false_authorizations_max": 0,
    "robust_coverage_min": 0.90,
    "confirmation_accuracy_min": 0.95,
    "positive_control_accuracy_min": 0.999,
    "funnel_models_min": 6,
    "funnel_per_depth_min": 2,
    "selected_models_required": 9,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_worlds(partition: str, count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        source = int(rng.integers(4))
        target = int(rng.choice([value for value in range(4) if value != source]))
        shift0 = int(rng.integers(4))
        shift1 = int(rng.choice([value for value in range(4) if value != shift0]))
        order = rng.permutation(4).astype(int).tolist()
        rows.append(
            p1266.make_factorial_world(
                source, target, shift0, shift1, order, partition, f"{partition[0]}{index:04d}"
            )
        )
    return rows


def make_material() -> list[dict[str, Any]]:
    rows = sample_worlds("qualification", PARTITION_COUNTS["qualification"], MATERIAL_SEEDS["qualification"])
    rows.extend(p1266.enumerate_oracle_worlds())
    rows.extend(sample_worlds("confirmation", PARTITION_COUNTS["confirmation"], MATERIAL_SEEDS["confirmation"]))
    return rows


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predecessor = read_json(PHASE1268_FINAL)
    audit = read_json(PHASE1268_AUDIT)
    erratum = read_json(PHASE1268_ERRATUM)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1269.c017.causal_support_funnel.v1",
        "claim_type": "free_transformer_depthwise_causal_support_funnel",
        "question": "Does the smallest certified causal support contract monotonically with depth and finish at the answer position across behavior-qualified free Transformers?",
        "phase1268_dependency": {
            "formal_complete": PHASE1268_COMPLETE.exists(),
            "passed": predecessor.get("passed"),
            "candidate": "depthwise nonincreasing support size",
            "final_hash": file_sha256(PHASE1268_FINAL),
            "frozen_audit": {"passed_checks": audit.get("passed_checks"), "total_checks": audit.get("total_checks"), "hash": file_sha256(PHASE1268_AUDIT)},
            "erratum_passed": erratum.get("passed"),
            "erratum_hash": file_sha256(PHASE1268_ERRATUM),
        },
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "seed_pools": SEED_POOLS,
        "behavior_selection": {
            "rule": "within each depth, admit the first three seeds passing frozen training and qualification thresholds; never inspect mechanism states for rejected seeds",
            "select_per_depth": SELECT_PER_DEPTH,
            "qualification_partition": "qualification",
        },
        "partitions": PARTITION_COUNTS,
        "material_seeds": MATERIAL_SEEDS,
        "world_digest": digest([{"row_id": row["row_id"], "partition": row["partition"], "row_digest": row["row_digest"]} for row in rows]),
        "analysis_families": list(ANALYSIS_FAMILIES),
        "support_sizes": SUPPORT_SIZES,
        "positive_control": CONTROL_FAMILY,
        "funnel_definition": {
            "complete_trajectory": "first certified analysis family at every layer",
            "independent_confirmation": "selected family at every layer passes both directions on confirmation",
            "monotone": "support size never increases with depth",
            "strict": "support size decreases at least once",
            "early_distributed": "first-layer support is larger than answer_only",
            "terminal": "last-layer support is answer_only",
        },
        "selection": {"draws": SELECTION_DRAWS, "radius": CERTIFICATE_RADIUS, "pass_lower_min": PASS_MIN},
        "thresholds": THRESHOLDS,
        "budgets": {"max_formal_runs": 1, "max_adaptive_rounds": 0, "max_gpu_hours": 3.5},
        "hard_stops": [
            "Behavior screening may inspect outputs only; no hidden support result may influence seed admission.",
            "All selected models and every layer remain in the mechanism denominator.",
            "Exact oracle scores cannot select support families.",
            "Final-layer answer replay alone cannot pass the funnel; early distributed support and strict contraction are required.",
            "No donor compiler or pretrained model is run automatically.",
        ],
        "structured_scope": {
            "task": "synthetic cyclic-code",
            "population": "behavior-qualified small free same-executor Transformers",
            "object": "registered cumulative exact-state support trajectory",
            "natural_language": False,
            "pretrained": False,
        },
        "source_hashes": {
            "main": file_sha256(Path(__file__).resolve()),
            "auditor": file_sha256(AUDITOR),
            "contract": file_sha256(CONTRACT),
            "phase1268": file_sha256(ROOT / "tests/glm5/phase1268_c016_distributed_causal_support_ladder.py"),
            "task": file_sha256(ROOT / "tests/glm5/phase1251_c004_causal_slice_competition.py"),
        },
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "fp32 training/intervention; fp64 audit arithmetic",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    predecessor = read_json(PHASE1268_FINAL)
    audit = read_json(PHASE1268_AUDIT)
    erratum = read_json(PHASE1268_ERRATUM)
    if predecessor.get("passed") is not False or not PHASE1268_COMPLETE.exists():
        raise RuntimeError("Phase1268 is not a completed formal failure")
    if audit.get("passed_checks") != 12 or audit.get("total_checks") != 13 or not erratum.get("passed"):
        raise RuntimeError("Phase1268 audit ledger drift")
    if not AUDITOR.exists() or not CONTRACT.exists():
        raise RuntimeError("auditor and contract must exist")
    rows = make_material()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "seed_pool": sum(map(len, SEED_POOLS.values())), "selected_target": MAX_MODELS, "radius": CERTIFICATE_RADIUS}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol digest drift")
    if protocol["source_hashes"] != expected["source_hashes"] or protocol["thresholds"] != THRESHOLDS:
        raise RuntimeError("source or threshold drift")
    counts = {name: sum(row["partition"] == name for row in rows) for name in PARTITION_COUNTS}
    if counts != PARTITION_COUNTS:
        raise RuntimeError(f"partition drift: {counts}")
    return protocol, rows


def partition_rows(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["partition"] == partition]


def selection_indices(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7_777_269)
    return rng.integers(0, PARTITION_COUNTS["oracle"], size=SELECTION_DRAWS)


def bounds(point: float) -> dict[str, float]:
    return {
        "point": point,
        "lower": max(0.0, point - CERTIFICATE_RADIUS),
        "upper": min(1.0, point + CERTIFICATE_RADIUS),
    }


def qualify_model(model, training: dict[str, Any], rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    natural_accuracy, executor_gap = p1268.evaluate_behavior(model, rows, device)
    passed = (
        min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], natural_accuracy)
        >= THRESHOLDS["behavior_accuracy_min"]
        and executor_gap <= THRESHOLDS["executor_gap_max"]
    )
    return {
        "training": training,
        "qualification_accuracy": natural_accuracy,
        "executor_gap": executor_gap,
        "passed": passed,
    }


def measure_selected_model(
    model,
    architecture: str,
    seed: int,
    pool_index: int,
    selected_index: int,
    qualification: dict[str, Any],
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    config = ARCHITECTURES[architecture]
    oracle_rows = partition_rows(rows, "oracle")
    confirmation_rows = partition_rows(rows, "confirmation")
    families = ANALYSIS_FAMILIES + (CONTROL_FAMILY,)
    pairs = [(layer, family) for layer in range(config.layers) for family in families]
    oracle = p1268.evaluate_supports(model, oracle_rows, pairs, device)
    selected_indices = selection_indices(seed)
    ledger = []
    chosen: list[tuple[int, str]] = []
    for layer in range(config.layers):
        first = None
        for family in families:
            outcome = oracle[f"{layer}:{family}"]
            patch = np.asarray(outcome["patch"], dtype=np.float64)
            reverse = np.asarray(outcome["reverse"], dtype=np.float64)
            pop_patch, pop_reverse = float(patch.mean()), float(reverse.mean())
            sample_patch, sample_reverse = float(patch[selected_indices].mean()), float(reverse[selected_indices].mean())
            patch_bounds, reverse_bounds = bounds(sample_patch), bounds(sample_reverse)
            score = min(pop_patch, pop_reverse)
            exact_pass = score >= PASS_MIN
            certificate_pass = min(patch_bounds["lower"], reverse_bounds["lower"]) >= PASS_MIN
            robust = score >= PASS_MIN + ROBUST_MULTIPLIER * CERTIFICATE_RADIUS
            selected = family in ANALYSIS_FAMILIES and first is None and certificate_pass
            if selected:
                first = family
                chosen.append((layer, family))
            ledger.append(
                {
                    "layer": layer,
                    "family": family,
                    "population_patch_accuracy": pop_patch,
                    "population_reverse_accuracy": pop_reverse,
                    "population_score": score,
                    "sample_patch_accuracy": sample_patch,
                    "sample_reverse_accuracy": sample_reverse,
                    "patch_bounds": patch_bounds,
                    "reverse_bounds": reverse_bounds,
                    "exact_pass": exact_pass,
                    "certificate_pass": certificate_pass,
                    "robust_actionable": robust,
                    "selected": selected,
                }
            )
        if first is None:
            chosen.append((layer, "abstain"))
    confirmation_pairs = [(layer, family) for layer, family in chosen if family != "abstain"]
    confirmation_raw = p1268.evaluate_supports(model, confirmation_rows, confirmation_pairs, device) if confirmation_pairs else {}
    confirmations = []
    for layer, family in chosen:
        if family == "abstain":
            confirmations.append({"layer": layer, "family": family, "cases": len(confirmation_rows), "patch_accuracy": 0.0, "reverse_accuracy": 0.0, "passed": False})
            continue
        outcome = confirmation_raw[f"{layer}:{family}"]
        patch_accuracy = float(np.mean(outcome["patch"]))
        reverse_accuracy = float(np.mean(outcome["reverse"]))
        confirmations.append(
            {
                "layer": layer,
                "family": family,
                "cases": len(confirmation_rows),
                "patch_accuracy": patch_accuracy,
                "reverse_accuracy": reverse_accuracy,
                "passed": min(patch_accuracy, reverse_accuracy) >= THRESHOLDS["confirmation_accuracy_min"],
            }
        )
    indices = [ANALYSIS_FAMILIES.index(family) if family in ANALYSIS_FAMILIES else None for _layer, family in chosen]
    sizes = [SUPPORT_SIZES[family] if family in SUPPORT_SIZES else None for _layer, family in chosen]
    complete = all(value is not None for value in indices)
    monotone = complete and all(indices[index + 1] <= indices[index] for index in range(len(indices) - 1))
    strict = complete and any(indices[index + 1] < indices[index] for index in range(len(indices) - 1))
    early_distributed = complete and indices[0] > 0
    terminal_answer = complete and indices[-1] == 0
    confirmation_passed = all(item["passed"] for item in confirmations)
    funnel = complete and monotone and strict and early_distributed and terminal_answer and confirmation_passed
    answer_onset = next((layer for layer, family in chosen if family == "answer_only"), None)
    return {
        "model_key": f"{architecture}_s{selected_index}_p{pool_index}",
        "architecture": architecture,
        "seed": seed,
        "pool_index": pool_index,
        "selected_index": selected_index,
        "qualification": qualification,
        "event_ledger": ledger,
        "trajectory": [{"layer": layer, "family": family, "size": SUPPORT_SIZES.get(family)} for layer, family in chosen],
        "confirmations": confirmations,
        "support_indices": indices,
        "support_sizes": sizes,
        "complete": complete,
        "monotone_nonincreasing": monotone,
        "strict_contraction": strict,
        "early_distributed": early_distributed,
        "terminal_answer_only": terminal_answer,
        "confirmation_passed": confirmation_passed,
        "answer_onset_layer": answer_onset,
        "answer_onset_relative": answer_onset / max(1, config.layers - 1) if answer_onset is not None else None,
        "funnel_passed": funnel,
    }


def summarize(qualification: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_events = [event for row in rows for event in row["event_ledger"]]
    false_authorizations = sum(event["certificate_pass"] and not event["exact_pass"] for event in all_events)
    robust_events = [event for event in all_events if event["robust_actionable"]]
    robust_coverage = sum(event["certificate_pass"] for event in robust_events) / max(1, len(robust_events))
    controls = [event for event in all_events if event["family"] == CONTROL_FAMILY]
    controls_passed = bool(controls) and all(event["population_score"] >= THRESHOLDS["positive_control_accuracy_min"] for event in controls)
    funnels = [row for row in rows if row["funnel_passed"]]
    per_depth = {architecture: sum(row["architecture"] == architecture for row in funnels) for architecture in ARCHITECTURES}
    breadth = len(funnels) >= THRESHOLDS["funnel_models_min"] and all(value >= THRESHOLDS["funnel_per_depth_min"] for value in per_depth.values())
    selected_per_depth = {architecture: sum(row["architecture"] == architecture for row in rows) for architecture in ARCHITECTURES}
    qualification_complete = len(rows) == THRESHOLDS["selected_models_required"] and all(value == SELECT_PER_DEPTH for value in selected_per_depth.values())
    gates = {
        "G-BEHAVIOR-SCREEN-COMPLETE": qualification_complete,
        "G-ZERO-FALSE-AUTHORIZATION": false_authorizations <= THRESHOLDS["certificate_false_authorizations_max"],
        "G-ROBUST-COVERAGE": bool(robust_events) and robust_coverage >= THRESHOLDS["robust_coverage_min"],
        "G-POSITIVE-CONTROL": controls_passed,
        "G-FUNNEL-BREADTH": breadth,
        "G-NO-PRETRAINED": True,
    }
    onset = [row["answer_onset_relative"] for row in rows if row["answer_onset_relative"] is not None]
    return {
        "attempted_seeds": len(qualification),
        "selected_models": len(rows),
        "selected_per_depth": selected_per_depth,
        "events": len(all_events),
        "false_authorizations": false_authorizations,
        "robust_events": len(robust_events),
        "robust_coverage": robust_coverage,
        "positive_control_events": len(controls),
        "positive_controls_passed": controls_passed,
        "complete_trajectories": sum(row["complete"] for row in rows),
        "monotone_trajectories": sum(row["monotone_nonincreasing"] for row in rows),
        "strict_contractions": sum(row["strict_contraction"] for row in rows),
        "early_distributed": sum(row["early_distributed"] for row in rows),
        "terminal_answer_only": sum(row["terminal_answer_only"] for row in rows),
        "confirmation_passed": sum(row["confirmation_passed"] for row in rows),
        "funnel_models": len(funnels),
        "funnel_per_depth": per_depth,
        "answer_onset_relative_median": float(np.median(onset)) if onset else None,
        "gates": gates,
        "decision": "causal_support_funnel_confirmed" if all(gates.values()) else "causal_support_funnel_not_confirmed",
        "passed": all(gates.values()),
    }


def run(device_name: str) -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal run already completed")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit must pass")
    protocol, material = verify_protocol()
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal run requires CUDA")
    device = torch.device("cuda")
    qualification_rows = partition_rows(material, "qualification")
    started = time.perf_counter()
    qualification_records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for architecture, config in ARCHITECTURES.items():
        selected = 0
        for pool_index, seed in enumerate(SEED_POOLS[architecture]):
            if selected >= SELECT_PER_DEPTH:
                break
            p1266.set_seed(seed)
            model, training = p1266.task_module.train_model(config, seed, device)
            qualification = qualify_model(model, training, qualification_rows, device)
            record = {"architecture": architecture, "pool_index": pool_index, "seed": seed, **qualification, "selected": qualification["passed"] and selected < SELECT_PER_DEPTH}
            qualification_records.append(record)
            write_jsonl(QUALIFICATION, qualification_records)
            if record["selected"]:
                result = measure_selected_model(model, architecture, seed, pool_index, selected, qualification, material, device)
                results.append(result)
                selected += 1
                write_jsonl(MODELS, results)
                print(canonical_json({"architecture": architecture, "selected": selected, "seed": seed, "funnel": result["funnel_passed"], "trajectory": [item["family"] for item in result["trajectory"]]}), flush=True)
            else:
                print(canonical_json({"architecture": architecture, "rejected_pool_index": pool_index, "seed": seed, "qualification": qualification["qualification_accuracy"]}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if selected < SELECT_PER_DEPTH:
            print(canonical_json({"architecture": architecture, "selected_shortfall": SELECT_PER_DEPTH - selected}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    write_jsonl(QUALIFICATION, qualification_records)
    write_jsonl(MODELS, results)
    run_summary = {
        "phase": PHASE,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        "attempted_seeds": len(qualification_records),
        "selected_models": len(results),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "device": torch.cuda.get_device_name(0),
        "qualification_hash": file_sha256(QUALIFICATION),
        "models_hash": file_sha256(MODELS),
        "run_digest": digest({"qualification": qualification_records, "models": results}),
        "protocol_digest": protocol["protocol_digest"],
        "pretrained_model_loaded": False,
    }
    atomic_json(SUMMARY, run_summary)
    atomic_json(COMPLETE, {"status": "formal_run_complete", "created_at_utc": utc_now(), "run_digest": run_summary["run_digest"], "models_hash": run_summary["models_hash"]})


def analyze() -> None:
    if not COMPLETE.exists():
        raise RuntimeError("formal run incomplete")
    protocol, _material = verify_protocol()
    qualification = read_jsonl(QUALIFICATION)
    results = read_jsonl(MODELS)
    summary = summarize(qualification, results)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        **summary,
        "authorization": {
            "distributed_donor_contract_design": summary["passed"],
            "automatic_pretrained_run": False,
            "qwen3": False,
            "glm4": False,
            "ds7b": False,
        },
        "structured_scope": protocol["structured_scope"],
        "protocol_digest": protocol["protocol_digest"],
        "qualification_hash": file_sha256(QUALIFICATION),
        "models_hash": file_sha256(MODELS),
        "run_digest": digest({"qualification": qualification, "models": results}),
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"decision": summary["decision"], "passed": summary["passed"], "funnel_models": summary["funnel_models"], "per_depth": summary["funnel_per_depth"], "onset_median": summary["answer_onset_relative_median"]}))


def run_auditor(mode: str) -> None:
    subprocess.run([sys.executable, str(AUDITOR), "--mode", mode], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prereg = sub.add_parser("preregister")
    prereg.add_argument("--force", action="store_true")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--device", default="cuda")
    sub.add_parser("analyze")
    audit = sub.add_parser("audit")
    audit.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        run(args.device)
    elif args.command == "analyze":
        analyze()
    elif args.command == "audit":
        run_auditor(args.mode)


if __name__ == "__main__":
    main()
