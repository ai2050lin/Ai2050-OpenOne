"""Phase1268: distributed causal-position support ladder in fresh free Transformers.

The phase does not fit another response camera.  It asks which cumulative set
of sequence positions is sufficient for a bidirectional exact-state patch at a
given layer.  Five sparse semantic supports compete.  The causal suffix and
full sequence are algebraic positive controls and cannot authorize a mechanism
claim.  Selection uses finite-sample simultaneous bounds; exhaustive finite-
universe accuracy is adjudication only; selected events are checked on a fresh
confirmation partition.
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
from typing import Any, Callable, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1266_c014_free_transformer_population_certificate as p1266
import phase1267_c015_observation_hierarchy_external_validity as p1267


PHASE = 1268
CAMPAIGN = "C016"
CONTRACT_ID = "EXP-C016-WP01-001"
OUT = ROOT / "tests/glm5/result/phase1268_c016_distributed_causal_support_ladder"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_support_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
MODELS = OUT / "raw/model_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1268_c016_distributed_causal_support_ladder_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C016-WP01-001.json"
PHASE1267_FINAL = ROOT / "tests/glm5/result/phase1267_c015_observation_hierarchy_external_validity/analysis/final.json"
PHASE1267_AUDIT = ROOT / "tests/glm5/result/phase1267_c015_observation_hierarchy_external_validity/audit/independent_final_audit.json"
PHASE1267_COMPLETE = ROOT / "tests/glm5/result/phase1267_c015_observation_hierarchy_external_validity/raw/FORMAL_RUN_COMPLETE.json"

ARCHITECTURES = p1266.ARCHITECTURES
REPLICATES = 3
MODEL_SEEDS = {
    "shallow4_r0": 1_268_401_001,
    "shallow4_r1": 1_268_401_101,
    "shallow4_r2": 1_268_401_201,
    "middle6_r0": 1_268_601_001,
    "middle6_r1": 1_268_601_101,
    "middle6_r2": 1_268_601_201,
    "deep8_r0": 1_268_801_001,
    "deep8_r1": 1_268_801_101,
    "deep8_r2": 1_268_801_201,
}
CONFIRMATION_SEED = 1_268_930_001
PARTITION_COUNTS = {"oracle": 3456, "confirmation": 1024}
SPARSE_FAMILIES = (
    "answer_only",
    "query_triplet",
    "source_query",
    "source_map_query",
    "semantic_chain",
)
POSITIVE_CONTROLS = ("causal_suffix", "full_sequence")
FAMILIES = SPARSE_FAMILIES + POSITIVE_CONTROLS
SELECTION_DRAWS = 32768
GLOBAL_ERROR_BUDGET = 0.01
PASS_MIN = 0.95
ROBUST_MULTIPLIER = 2.0
MAX_EVENTS = sum(config.layers for config in ARCHITECTURES.values()) * REPLICATES
CERTIFICATE_RADIUS = math.sqrt(
    math.log(2.0 * MAX_EVENTS * len(FAMILIES) * 2.0 / GLOBAL_ERROR_BUDGET)
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
    "breadth_models_min": 6,
    "breadth_per_depth_min": 2,
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


def model_key(architecture: str, replicate: int) -> str:
    return f"{architecture}_r{replicate}"


def sample_confirmation(count: int, seed: int) -> list[dict[str, Any]]:
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
                source, target, shift0, shift1, order, "confirmation", f"c{index:04d}"
            )
        )
    return rows


def make_material() -> list[dict[str, Any]]:
    return p1266.enumerate_oracle_worlds() + sample_confirmation(PARTITION_COUNTS["confirmation"], CONFIRMATION_SEED)


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predecessor = read_json(PHASE1267_FINAL)
    audit = read_json(PHASE1267_AUDIT)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1268.c016.distributed_support_ladder.v1",
        "claim_type": "free_transformer_distributed_causal_support",
        "question": "Does a sparse registered position family form a bidirectionally sufficient causal interface across fresh free Transformers?",
        "phase1267_dependency": {
            "formal_complete": PHASE1267_COMPLETE.exists(),
            "passed": predecessor.get("passed"),
            "decision": predecessor.get("decision"),
            "final_hash": file_sha256(PHASE1267_FINAL),
            "audit_passed": audit.get("all_checks_passed"),
            "audit_hash": file_sha256(PHASE1267_AUDIT),
        },
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_seeds": MODEL_SEEDS,
        "partitions": PARTITION_COUNTS,
        "confirmation_seed": CONFIRMATION_SEED,
        "world_digest": digest([{"row_id": row["row_id"], "partition": row["partition"], "row_digest": row["row_digest"]} for row in rows]),
        "support_order": list(FAMILIES),
        "support_definitions": {
            "answer_only": [22],
            "query_triplet": [20, 21, 22],
            "source_query": [4, 20, 21, 22],
            "source_map_query": "source_query plus the target-code codebook key/value positions",
            "semantic_chain": "source_map_query plus shift position 11",
            "causal_suffix": "all positions 4 through 22; algebraic positive control only",
            "full_sequence": "all positions 0 through 22; algebraic positive control only",
        },
        "selection": {
            "draws": SELECTION_DRAWS,
            "radius": CERTIFICATE_RADIUS,
            "rule": "both patch and reverse lower bounds must be at least 0.95",
            "event_choice": "first certified sparse family per layer; confirm minimum-family, earliest-layer, and latest-layer events",
        },
        "thresholds": THRESHOLDS,
        "decision": {
            "sparse": "first cumulative sparse family reaching 6/9 models and 2/3 per depth",
            "positive_control_only": "only_trivial_causal_suffix_sufficient",
            "invalid": "state_replacement_executor_invalid",
        },
        "budgets": {"max_formal_runs": 1, "max_adaptive_rounds": 0, "max_gpu_hours": 3.0},
        "hard_stops": [
            "Exact oracle accuracy cannot select a support family or layer.",
            "Causal suffix and full sequence are positive controls and cannot authorize sparse-support claims.",
            "All nine models remain in the denominator.",
            "No donor compiler is fit unless a separate future contract is authorized.",
            "No pretrained model is loaded.",
        ],
        "structured_scope": {
            "task": "synthetic cyclic-code",
            "models": "small free same-executor Transformers",
            "intervention": "exact residual-state replacement",
            "natural_language": False,
            "pretrained": False,
        },
        "source_hashes": {
            "main": file_sha256(Path(__file__).resolve()),
            "auditor": file_sha256(AUDITOR),
            "contract": file_sha256(CONTRACT),
            "phase1267": file_sha256(ROOT / "tests/glm5/phase1267_c015_observation_hierarchy_external_validity.py"),
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
        "precision": "fp32 training and exact interventions; fp64 audit arithmetic",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    predecessor = read_json(PHASE1267_FINAL)
    audit = read_json(PHASE1267_AUDIT)
    if predecessor.get("decision") != "registered_hierarchy_insufficient" or predecessor.get("passed") is not False:
        raise RuntimeError("Phase1267 did not authorize support-ladder triage")
    if not audit.get("all_checks_passed") or not PHASE1267_COMPLETE.exists():
        raise RuntimeError("Phase1267 audit or completion missing")
    if not AUDITOR.exists() or not CONTRACT.exists():
        raise RuntimeError("auditor and contract must exist")
    rows = make_material()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "models": len(MODEL_SEEDS), "radius": CERTIFICATE_RADIUS}))


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


FullAction = Callable[[torch.Tensor], torch.Tensor]


def full_residual_forward(
    model,
    input_ids: torch.Tensor,
    actions: dict[int, FullAction] | None = None,
    capture: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden = model.embed(input_ids)
    residuals: list[torch.Tensor] = []
    length = input_ids.shape[1]
    causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1)
    for layer_index, block in enumerate(model.blocks):
        normalized = block.attn_norm(hidden)
        batch, _, width = normalized.shape
        qkv = block.attn.qkv(normalized).view(batch, length, 3, block.attn.heads, block.attn.head_dim)
        query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        weights = torch.softmax(scores.masked_fill(causal[None, None], float("-inf")), dim=-1)
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous().view(batch, length, width)
        hidden = hidden + block.attn.out(attended)
        hidden = hidden + block.mlp(block.mlp_norm(hidden))
        if actions and layer_index in actions:
            hidden = actions[layer_index](hidden)
        if capture:
            residuals.append(hidden.detach().clone())
    logits = model.lm_head(model.final_norm(hidden))
    return logits, torch.stack(residuals, dim=1) if capture else None


def support_mask(rows: list[dict[str, Any]], family: str, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(rows), 23), dtype=torch.bool, device=device)
    fixed = {
        "answer_only": (22,),
        "query_triplet": (20, 21, 22),
        "source_query": (4, 20, 21, 22),
    }
    if family in fixed:
        mask[:, list(fixed[family])] = True
        return mask
    if family in ("source_map_query", "semantic_chain"):
        mask[:, [4, 20, 21, 22]] = True
        if family == "semantic_chain":
            mask[:, 11] = True
        for index, row in enumerate(rows):
            pair = 12 + 2 * row["codebook_order"].index(row["target_code"])
            mask[index, pair] = True
            mask[index, pair + 1] = True
        return mask
    if family == "causal_suffix":
        mask[:, 4:23] = True
        return mask
    if family == "full_sequence":
        mask[:, :] = True
        return mask
    raise ValueError(family)


def patch_action(donor: torch.Tensor, mask: torch.Tensor) -> FullAction:
    def apply(current: torch.Tensor) -> torch.Tensor:
        result = current.clone()
        result[mask] = donor[mask]
        return result
    return apply


def evaluate_behavior(model, rows: list[dict[str, Any]], device: torch.device, batch_size: int = 512) -> tuple[float, float]:
    accuracies = []
    max_gap = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            for panel in p1266.PANELS:
                ids = torch.tensor([row[f"{panel}_ids"] for row in batch], device=device)
                native = model(ids)
                explicit, _ = full_residual_forward(model, ids)
                max_gap = max(max_gap, float(torch.max(torch.abs(native.float() - explicit.float())).item()))
                expected = torch.tensor([row["answers"][panel] for row in batch], device=device)
                predicted = torch.argmax(explicit[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
                accuracies.append((predicted == expected).float().cpu())
    return float(torch.cat(accuracies).mean().item()), max_gap


def evaluate_supports(
    model,
    rows: list[dict[str, Any]],
    pairs: list[tuple[int, str]],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, dict[str, list[bool]]]:
    outcomes = {f"{layer}:{family}": {"patch": [], "reverse": []} for layer, family in pairs}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            h01_ids = torch.tensor([row["h01_ids"] for row in batch], device=device)
            h11_ids = torch.tensor([row["h11_ids"] for row in batch], device=device)
            _h01_logits, trace01 = full_residual_forward(model, h01_ids, capture=True)
            _h11_logits, trace11 = full_residual_forward(model, h11_ids, capture=True)
            assert trace01 is not None and trace11 is not None
            target = torch.tensor([row["answers"]["h11"] for row in batch], device=device)
            base = torch.tensor([row["answers"]["h01"] for row in batch], device=device)
            masks = {family: support_mask(batch, family, device) for _layer, family in pairs}
            for layer, family in pairs:
                mask = masks[family]
                patch_logits, _ = full_residual_forward(
                    model,
                    h01_ids,
                    actions={layer: patch_action(trace11[:, layer], mask)},
                    capture=False,
                )
                reverse_logits, _ = full_residual_forward(
                    model,
                    h11_ids,
                    actions={layer: patch_action(trace01[:, layer], mask)},
                    capture=False,
                )
                patch_pred = torch.argmax(patch_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
                reverse_pred = torch.argmax(reverse_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
                key = f"{layer}:{family}"
                outcomes[key]["patch"].extend((patch_pred == target).cpu().tolist())
                outcomes[key]["reverse"].extend((reverse_pred == base).cpu().tolist())
    return outcomes


def selection_indices(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7_777_168)
    return rng.integers(0, PARTITION_COUNTS["oracle"], size=SELECTION_DRAWS)


def bounds(point: float) -> dict[str, float]:
    return {
        "point": point,
        "lower": max(0.0, point - CERTIFICATE_RADIUS),
        "upper": min(1.0, point + CERTIFICATE_RADIUS),
    }


def run_model(architecture: str, replicate: int, config, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    key = model_key(architecture, replicate)
    seed = MODEL_SEEDS[key]
    p1266.set_seed(seed)
    model, training = p1266.task_module.train_model(config, seed, device)
    oracle_rows = [row for row in rows if row["partition"] == "oracle"]
    confirmation_rows = [row for row in rows if row["partition"] == "confirmation"]
    natural_accuracy, executor_gap = evaluate_behavior(model, oracle_rows + confirmation_rows, device)
    qualified = (
        min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], natural_accuracy)
        >= THRESHOLDS["behavior_accuracy_min"]
        and executor_gap <= THRESHOLDS["executor_gap_max"]
    )
    base = {
        "model_key": key,
        "architecture": architecture,
        "replicate": replicate,
        "seed": seed,
        "training": training,
        "natural_accuracy": natural_accuracy,
        "executor_gap": executor_gap,
        "behavior_qualified": qualified,
    }
    if not qualified:
        return {**base, "event_ledger": [], "confirmations": [], "passed": False}
    pairs = [(layer, family) for layer in range(config.layers) for family in FAMILIES]
    oracle = evaluate_supports(model, oracle_rows, pairs, device)
    selected_indices = selection_indices(seed)
    ledger = []
    first_sparse_by_layer: dict[int, str] = {}
    for layer in range(config.layers):
        for family in FAMILIES:
            outcome = oracle[f"{layer}:{family}"]
            patch = np.asarray(outcome["patch"], dtype=np.float64)
            reverse = np.asarray(outcome["reverse"], dtype=np.float64)
            population_patch = float(patch.mean())
            population_reverse = float(reverse.mean())
            sample_patch = float(patch[selected_indices].mean())
            sample_reverse = float(reverse[selected_indices].mean())
            patch_bounds = bounds(sample_patch)
            reverse_bounds = bounds(sample_reverse)
            exact_score = min(population_patch, population_reverse)
            sample_score = min(sample_patch, sample_reverse)
            exact_pass = exact_score >= PASS_MIN
            certificate_pass = min(patch_bounds["lower"], reverse_bounds["lower"]) >= PASS_MIN
            robust = exact_score >= PASS_MIN + ROBUST_MULTIPLIER * CERTIFICATE_RADIUS
            event = {
                "layer": layer,
                "family": family,
                "population_patch_accuracy": population_patch,
                "population_reverse_accuracy": population_reverse,
                "population_score": exact_score,
                "sample_patch_accuracy": sample_patch,
                "sample_reverse_accuracy": sample_reverse,
                "sample_score": sample_score,
                "patch_bounds": patch_bounds,
                "reverse_bounds": reverse_bounds,
                "exact_pass": exact_pass,
                "certificate_pass": certificate_pass,
                "robust_actionable": robust,
                "selected_sparse": False,
            }
            if family in SPARSE_FAMILIES and layer not in first_sparse_by_layer and certificate_pass:
                first_sparse_by_layer[layer] = family
                event["selected_sparse"] = True
            ledger.append(event)
    selected_events = [(layer, family) for layer, family in sorted(first_sparse_by_layer.items())]
    confirmation_targets: list[tuple[int, str]] = []
    if selected_events:
        minimum = min(selected_events, key=lambda item: (SPARSE_FAMILIES.index(item[1]), item[0]))
        earliest = min(selected_events, key=lambda item: item[0])
        latest = max(selected_events, key=lambda item: item[0])
        for item in (minimum, earliest, latest):
            if item not in confirmation_targets:
                confirmation_targets.append(item)
    confirmation_raw = evaluate_supports(model, confirmation_rows, confirmation_targets, device) if confirmation_targets else {}
    confirmations = []
    for layer, family in confirmation_targets:
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
    passed_confirmations = [item for item in confirmations if item["passed"]]
    support_ceiling = min(
        (SPARSE_FAMILIES.index(item["family"]) for item in passed_confirmations),
        default=None,
    )
    return {
        **base,
        "event_ledger": ledger,
        "selected_events": [{"layer": layer, "family": family} for layer, family in selected_events],
        "confirmation_targets": [{"layer": layer, "family": family} for layer, family in confirmation_targets],
        "confirmations": confirmations,
        "support_ceiling_index": support_ceiling,
        "support_ceiling_family": SPARSE_FAMILIES[support_ceiling] if support_ceiling is not None else None,
        "passed": bool(passed_confirmations),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [row for row in rows if row["behavior_qualified"]]
    all_events = [event for row in qualified for event in row["event_ledger"]]
    false_authorizations = sum(event["certificate_pass"] and not event["exact_pass"] for event in all_events)
    point_false_authorizations = sum(event["sample_score"] >= PASS_MIN and not event["exact_pass"] for event in all_events)
    robust_events = [event for event in all_events if event["robust_actionable"]]
    robust_coverage = sum(event["certificate_pass"] for event in robust_events) / max(1, len(robust_events))
    control_events = [event for event in all_events if event["family"] in POSITIVE_CONTROLS]
    positive_controls = bool(control_events) and all(
        event["population_score"] >= THRESHOLDS["positive_control_accuracy_min"] for event in control_events
    )
    family_breadth = {}
    authorized_family = None
    for index, family in enumerate(SPARSE_FAMILIES):
        models = [row for row in qualified if row["support_ceiling_index"] is not None and row["support_ceiling_index"] <= index]
        per_depth = {
            architecture: sum(row["architecture"] == architecture for row in models)
            for architecture in ARCHITECTURES
        }
        breadth = (
            len(models) >= THRESHOLDS["breadth_models_min"]
            and all(value >= THRESHOLDS["breadth_per_depth_min"] for value in per_depth.values())
        )
        family_breadth[family] = {"models": len(models), "per_depth": per_depth, "authorized": breadth}
        if authorized_family is None and breadth:
            authorized_family = family
    if authorized_family is not None:
        decision = f"distributed_sparse_support_identified:{authorized_family}"
    elif positive_controls:
        decision = "only_trivial_causal_suffix_sufficient"
    else:
        decision = "state_replacement_executor_invalid"
    gates = {
        "G-BEHAVIOR": len(qualified) == len(rows),
        "G-ZERO-FALSE-AUTHORIZATION": false_authorizations <= THRESHOLDS["certificate_false_authorizations_max"],
        "G-ROBUST-COVERAGE": bool(robust_events) and robust_coverage >= THRESHOLDS["robust_coverage_min"],
        "G-POSITIVE-CONTROLS": positive_controls,
        "G-SPARSE-SUPPORT-BREADTH": authorized_family is not None,
        "G-NO-PRETRAINED": True,
    }
    return {
        "models": len(rows),
        "qualified": len(qualified),
        "events": len(all_events),
        "false_authorizations": false_authorizations,
        "point_false_authorizations": point_false_authorizations,
        "robust_events": len(robust_events),
        "robust_coverage": robust_coverage,
        "positive_control_events": len(control_events),
        "positive_controls_passed": positive_controls,
        "passed_models": sum(row["passed"] for row in qualified),
        "family_breadth": family_breadth,
        "authorized_family": authorized_family,
        "decision": decision,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run(device_name: str) -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal run already completed")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit must pass")
    protocol, rows = verify_protocol()
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal run requires CUDA")
    device = torch.device("cuda")
    started = time.perf_counter()
    results = []
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            result = run_model(architecture, replicate, config, rows, device)
            results.append(result)
            write_jsonl(MODELS, results)
            print(canonical_json({"completed": len(results), "total": len(MODEL_SEEDS), "model": result["model_key"], "support": result.get("support_ceiling_family")}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary = {
        "phase": PHASE,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        "models": len(results),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "device": torch.cuda.get_device_name(0),
        "models_hash": file_sha256(MODELS),
        "run_digest": digest(results),
        "protocol_digest": protocol["protocol_digest"],
        "pretrained_model_loaded": False,
    }
    atomic_json(SUMMARY, summary)
    atomic_json(COMPLETE, {"status": "formal_run_complete", "created_at_utc": utc_now(), "run_digest": summary["run_digest"], "models_hash": summary["models_hash"]})


def analyze() -> None:
    if not COMPLETE.exists():
        raise RuntimeError("formal run incomplete")
    protocol, _rows = verify_protocol()
    results = read_jsonl(MODELS)
    summary = summarize(results)
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
        "models_hash": file_sha256(MODELS),
        "run_digest": digest(results),
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"decision": summary["decision"], "authorized_family": summary["authorized_family"], "passed": summary["passed"], "family_breadth": summary["family_breadth"]}))


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
