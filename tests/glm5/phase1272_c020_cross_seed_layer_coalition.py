"""Phase1272/C020: cross-seed minimum layer-coalition transfer."""

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
import phase1269_c017_causal_support_funnel_confirmation as p1269
import phase1271_c019_cross_layer_micro_write_trajectory as p1271


PHASE = 1272
CAMPAIGN = "C020"
CONTRACT_ID = "EXP-C020-WP01-001"
OUT = ROOT / "tests/glm5/result/phase1272_c020_cross_seed_layer_coalition"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_coalition_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
DISCOVERY = OUT / "raw/discovery_results.jsonl"
MODELS = OUT / "raw/model_results.jsonl"
QUALIFICATION = OUT / "raw/behavior_qualification.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1272_c020_cross_seed_layer_coalition_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C020-WP01-001.json"
PREDECESSOR_FINAL = ROOT / "tests/glm5/result/phase1271_c019_cross_layer_micro_write_trajectory/analysis/final.json"
PREDECESSOR_AUDIT = ROOT / "tests/glm5/result/phase1271_c019_cross_layer_micro_write_trajectory/audit/independent_final_audit.json"
PREDECESSOR_MODELS = ROOT / "tests/glm5/result/phase1271_c019_cross_layer_micro_write_trajectory/raw/model_results.jsonl"

ARCHITECTURES = p1266.ARCHITECTURES
SEEDS = {
    "shallow4": [1_271_401_001, 1_271_401_101, 1_271_401_201],
    "middle6": [1_271_601_001, 1_271_601_101, 1_271_601_201],
    "deep8": [1_271_801_001, 1_271_801_101, 1_271_801_201],
}
DISCOVERY_INDEX = 0
HELDOUT_INDICES = (1, 2)
PARTITION_COUNTS = {"qualification": 1024, "selection": 16384, "oracle": 3456}
MATERIAL_SEEDS = {"qualification": 1_272_910_001, "selection": 1_272_920_001}
GLOBAL_ERROR_BUDGET = 0.01
PASS_MIN = 0.95
NULL_MAX = 0.05
IDENTITY_MIN = 0.999
MAX_SELECTION_MASKS = sum(2 ** config.layers for config in ARCHITECTURES.values())
CERTIFICATE_RADIUS = math.sqrt(
    math.log(2.0 * MAX_SELECTION_MASKS * 4.0 / GLOBAL_ERROR_BUDGET)
    / (2.0 * PARTITION_COUNTS["selection"])
)
THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "executor_gap_max": 2.0e-4,
    "transfer_accuracy_min": PASS_MIN,
    "wrong_false_target_max": NULL_MAX,
    "identity_control_min": IDENTITY_MIN,
    "empty_and_null_max": NULL_MAX,
    "certificate_radius": CERTIFICATE_RADIUS,
    "false_authorizations_max": 0,
    "robust_coverage_min": 0.90,
    "discovery_depths_required": 3,
    "heldout_models_required": 6,
    "heldout_per_depth_required": 2,
    "all_models_required": 9,
    "sparse_fraction_max": 0.50,
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


def make_material() -> list[dict[str, Any]]:
    rows = p1269.sample_worlds("qualification", PARTITION_COUNTS["qualification"], MATERIAL_SEEDS["qualification"])
    rows.extend(p1269.sample_worlds("selection", PARTITION_COUNTS["selection"], MATERIAL_SEEDS["selection"]))
    rows.extend(p1266.enumerate_oracle_worlds())
    return rows


def partition_rows(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["partition"] == partition]


def all_masks(layers: int) -> list[list[int]]:
    return [[layer for layer in range(layers) if integer & (1 << layer)] for integer in range(2 ** layers)]


def mask_name(layers: list[int]) -> str:
    return "empty" if not layers else "L" + "-".join(str(value) for value in layers)


def proper_subsets(layers: list[int]) -> list[list[int]]:
    selected = set(layers)
    return [mask for mask in all_masks(max(selected) + 1 if selected else 0) if set(mask) < selected]


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predecessor = read_json(PREDECESSOR_FINAL)
    audit = read_json(PREDECESSOR_AUDIT)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1272.c020.cross_seed_layer_coalition.v1",
        "claim_type": "same_depth_cross_seed_teacher_forced_layer_coalition_transfer",
        "question": "Does one discovery-selected minimum layer mask transfer the same identity contrast on two unseen seeds at every depth?",
        "dependency": {
            "decision": predecessor.get("decision"),
            "passed": predecessor.get("passed"),
            "authorized": predecessor.get("authorization", {}).get("layer_coalition_minimality_contract"),
            "final_hash": file_sha256(PREDECESSOR_FINAL),
            "audit_passed": audit.get("all_checks_passed"),
            "audit_hash": file_sha256(PREDECESSOR_AUDIT),
            "models_hash": file_sha256(PREDECESSOR_MODELS),
        },
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "seeds": SEEDS,
        "split_roles": {"discovery_index": DISCOVERY_INDEX, "heldout_indices": HELDOUT_INDICES},
        "partitions": PARTITION_COUNTS,
        "material_seeds": MATERIAL_SEEDS,
        "material_digest": digest([{"row_id": row["row_id"], "partition": row["partition"], "row_digest": row["row_digest"]} for row in rows]),
        "candidate_masks": {name: all_masks(config.layers) for name, config in ARCHITECTURES.items()},
        "selection": {
            "rule": "minimum cardinality then lexicographic layer tuple among certified identity-specific masks",
            "confidence": "simultaneous Hoeffding bounds over every mask and four registered rates",
            "radius": CERTIFICATE_RADIUS,
            "heldout_blind": True,
            "oracle_blind_for_selection": True,
        },
        "intervention": {
            "position": 22,
            "primary_stage": "attention_write",
            "directions": ["h01_to_h11", "h11_to_h01", "h01_to_hwrong11"],
            "controls": ["empty_mask", "full_attention", "same_state_noop", "pre_source_position2", "matched_mlp"],
        },
        "thresholds": THRESHOLDS,
        "budgets": {"max_formal_runs": 1, "max_adaptive_rounds": 0, "max_gpu_hours": 5.0},
        "hard_stops": [
            "Held-out models and oracle population cannot select a mask.",
            "A transfer coalition is not natural individual necessity.",
            "Failure closes synthetic layer-coalition localization.",
            "No head search unless transfer, shared minimality, and sparsity all pass.",
            "No pretrained model is loaded automatically.",
        ],
        "structured_scope": {
            "task": "synthetic cyclic-code",
            "population": "the same nine behavior-qualified free Transformers frozen by Phase1271",
            "intervention": "same-case teacher-forced answer-position component writes",
            "natural_necessity": False,
            "natural_language": False,
            "pretrained": False,
        },
        "source_hashes": {
            "main": file_sha256(Path(__file__).resolve()),
            "auditor": file_sha256(AUDITOR),
            "contract": file_sha256(CONTRACT),
            "predecessor": file_sha256(ROOT / "tests/glm5/phase1271_c019_cross_layer_micro_write_trajectory.py"),
            "executor": file_sha256(ROOT / "tests/glm5/phase1268_c016_distributed_causal_support_ladder.py"),
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
    predecessor = read_json(PREDECESSOR_FINAL)
    audit = read_json(PREDECESSOR_AUDIT)
    if predecessor.get("passed") is not True or predecessor.get("authorization", {}).get("layer_coalition_minimality_contract") is not True:
        raise RuntimeError("Phase1271 did not authorize C020")
    if audit.get("all_checks_passed") is not True:
        raise RuntimeError("Phase1271 final audit failed")
    if not AUDITOR.exists() or not CONTRACT.exists():
        raise RuntimeError("auditor and contract must exist before registration")
    predecessor_seeds = {(row["architecture"], int(row["selected_index"])): int(row["seed"]) for row in read_jsonl(PREDECESSOR_MODELS)}
    expected = {(architecture, index): seed for architecture, seeds in SEEDS.items() for index, seed in enumerate(seeds)}
    if predecessor_seeds != expected:
        raise RuntimeError("frozen Phase1271 model population drift")
    rows = make_material()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "candidate_masks": MAX_SELECTION_MASKS, "radius": CERTIFICATE_RADIUS}))


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


def qualify_model(model, training: dict[str, Any], rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    accuracy, gap = p1268.evaluate_behavior(model, rows, device)
    passed = (
        min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], accuracy)
        >= THRESHOLDS["behavior_accuracy_min"]
        and gap <= THRESHOLDS["executor_gap_max"]
    )
    return {"training": training, "qualification_accuracy": accuracy, "executor_gap": gap, "passed": passed}


def masks_tensor(masks: list[list[int]], layers: int, device: torch.device) -> torch.Tensor:
    value = torch.zeros((len(masks), layers), dtype=torch.bool, device=device)
    for index, mask in enumerate(masks):
        if mask:
            value[index, torch.tensor(mask, dtype=torch.long, device=device)] = True
    return value


def forward_masks(
    model,
    receiver_ids: torch.Tensor,
    donor: list[dict[str, torch.Tensor]],
    masks: list[list[int]],
    stage: str = "attn_write",
    position: int = 22,
) -> torch.Tensor:
    mask_values = masks_tensor(masks, len(model.blocks), receiver_ids.device)
    mask_count, batch = len(masks), receiver_ids.shape[0]
    hidden = model.embed(receiver_ids).repeat(mask_count, 1, 1)
    causal = torch.triu(torch.ones(receiver_ids.shape[1], receiver_ids.shape[1], dtype=torch.bool, device=receiver_ids.device), diagonal=1)
    for layer, block in enumerate(model.blocks):
        attn_write, after_attn, mlp_write = p1271.block_parts(block, hidden, causal)
        selected = mask_values[:, layer].repeat_interleave(batch)
        if stage == "attn_write" and bool(selected.any()):
            replacement = donor[layer]["attn_write"][:, position].repeat(mask_count, 1)
            attn_write = attn_write.clone()
            attn_write[:, position] = torch.where(selected[:, None], replacement, attn_write[:, position])
            after_attn = hidden + attn_write
            mlp_write = block.mlp(block.mlp_norm(after_attn))
        if stage == "mlp_write" and bool(selected.any()):
            replacement = donor[layer]["mlp_write"][:, position].repeat(mask_count, 1)
            mlp_write = mlp_write.clone()
            mlp_write[:, position] = torch.where(selected[:, None], replacement, mlp_write[:, position])
        hidden = after_attn + mlp_write
    logits = model.lm_head(model.final_norm(hidden))[:, -1, p1266.CANDIDATE_SLICE]
    return torch.argmax(logits, dim=-1).view(mask_count, batch)


def evaluate_masks(
    model,
    rows: list[dict[str, Any]],
    masks: list[list[int]],
    device: torch.device,
    *,
    stage: str = "attn_write",
    position: int = 22,
    include_wrong: bool = True,
    same_state: bool = False,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    count = len(masks)
    totals = {
        "forward": np.zeros(count, dtype=np.int64),
        "reverse": np.zeros(count, dtype=np.int64),
        "wrong": np.zeros(count, dtype=np.int64),
        "wrong_false_h11": np.zeros(count, dtype=np.int64),
    }
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            ids = {panel: torch.tensor([row[f"{panel}_ids"] for row in batch_rows], device=device) for panel in ("h01", "h11", "hwrong11")}
            traces = {panel: p1271.capture_micro(model, values) for panel, values in ids.items()}
            answers = {panel: torch.tensor([row["answers"][panel] for row in batch_rows], device=device) for panel in ("h01", "h11", "hwrong11")}
            if same_state:
                forward = forward_masks(model, ids["h01"], traces["h01"], masks, stage, position)
                reverse = forward_masks(model, ids["h11"], traces["h11"], masks, stage, position)
                totals["forward"] += (forward == answers["h01"][None]).sum(dim=1).cpu().numpy()
                totals["reverse"] += (reverse == answers["h11"][None]).sum(dim=1).cpu().numpy()
            else:
                forward = forward_masks(model, ids["h01"], traces["h11"], masks, stage, position)
                reverse = forward_masks(model, ids["h11"], traces["h01"], masks, stage, position)
                totals["forward"] += (forward == answers["h11"][None]).sum(dim=1).cpu().numpy()
                totals["reverse"] += (reverse == answers["h01"][None]).sum(dim=1).cpu().numpy()
                if include_wrong:
                    wrong = forward_masks(model, ids["h01"], traces["hwrong11"], masks, stage, position)
                    totals["wrong"] += (wrong == answers["hwrong11"][None]).sum(dim=1).cpu().numpy()
                    totals["wrong_false_h11"] += (wrong == answers["h11"][None]).sum(dim=1).cpu().numpy()
            del ids, traces, answers, forward, reverse
    result = []
    denominator = float(len(rows))
    for index, mask in enumerate(masks):
        row = {"mask": mask, "mask_name": mask_name(mask), "cardinality": len(mask)}
        row.update({name: float(values[index] / denominator) for name, values in totals.items()})
        result.append(row)
    return result


def with_bounds(row: dict[str, Any]) -> dict[str, Any]:
    rates = {name: row[name] for name in ("forward", "reverse", "wrong", "wrong_false_h11")}
    lower = min(rates["forward"], rates["reverse"], rates["wrong"]) - CERTIFICATE_RADIUS
    false_upper = rates["wrong_false_h11"] + CERTIFICATE_RADIUS
    return {
        **row,
        "certificate_lower": max(0.0, lower),
        "false_target_upper": min(1.0, false_upper),
        "certificate_pass": lower >= PASS_MIN and false_upper <= NULL_MAX,
        "robust_actionable": min(rates["forward"], rates["reverse"], rates["wrong"]) >= PASS_MIN + 2.0 * CERTIFICATE_RADIUS and rates["wrong_false_h11"] <= NULL_MAX - 2.0 * CERTIFICATE_RADIUS,
    }


def exact_pass(row: dict[str, Any]) -> bool:
    return min(row["forward"], row["reverse"], row["wrong"]) >= PASS_MIN and row["wrong_false_h11"] <= NULL_MAX


def select_mask(ledger: list[dict[str, Any]]) -> list[int] | None:
    candidates = [row for row in ledger if row["mask"] and row["certificate_pass"]]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["cardinality"], tuple(row["mask"])))["mask"]


def evaluate_fixed_model(
    model,
    architecture: str,
    seed: int,
    replicate: int,
    selected_mask: list[int],
    qualification: dict[str, Any],
    oracle: list[dict[str, Any]],
    device: torch.device,
    precomputed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    layers = ARCHITECTURES[architecture].layers
    proper = [mask for mask in all_masks(layers) if set(mask) < set(selected_mask)]
    registry_map = {tuple(mask): mask for mask in proper + [[], selected_mask, list(range(layers))]}
    registry = sorted(registry_map.values(), key=lambda mask: (len(mask), tuple(mask)))
    if precomputed is None:
        attention = evaluate_masks(model, oracle, registry, device)
    else:
        lookup = {tuple(row["mask"]): row for row in precomputed}
        attention = [lookup[tuple(mask)] for mask in registry]
    attention_map = {tuple(row["mask"]): {**row, "exact_pass": exact_pass(row)} for row in attention}
    selected = attention_map[tuple(selected_mask)]
    full = attention_map[tuple(range(layers))]
    empty = attention_map[tuple()]
    proper_passes = [row for mask, row in attention_map.items() if set(mask) < set(selected_mask) and row["exact_pass"]]
    noop_rows = evaluate_masks(model, oracle, [selected_mask], device, include_wrong=False, same_state=True)
    null_rows = evaluate_masks(model, oracle, [selected_mask], device, position=2)
    mlp_rows = evaluate_masks(model, oracle, [selected_mask], device, stage="mlp_write")
    noop_score = min(noop_rows[0]["forward"], noop_rows[0]["reverse"])
    null_score = max(null_rows[0]["forward"], null_rows[0]["reverse"], null_rows[0]["wrong"])
    mlp_score = min(mlp_rows[0]["forward"], mlp_rows[0]["reverse"], mlp_rows[0]["wrong"])
    controls = (
        exact_pass(full)
        and max(empty["forward"], empty["reverse"], empty["wrong"]) <= NULL_MAX
        and noop_score >= IDENTITY_MIN
        and null_score <= NULL_MAX
    )
    return {
        "model_key": f"{architecture}_r{replicate}",
        "architecture": architecture,
        "seed": seed,
        "replicate": replicate,
        "role": "discovery" if replicate == DISCOVERY_INDEX else "heldout",
        "qualification": qualification,
        "selected_mask": selected_mask,
        "selected_mask_name": mask_name(selected_mask),
        "selected_cardinality": len(selected_mask),
        "selected_fraction": len(selected_mask) / layers,
        "selected_metrics": selected,
        "full_metrics": full,
        "empty_metrics": empty,
        "proper_subset_passes": proper_passes,
        "shared_minimality_passed": bool(selected["exact_pass"] and not proper_passes),
        "same_state_noop_score": noop_score,
        "pre_source_null_score": null_score,
        "matched_mlp_score": mlp_score,
        "attention_over_mlp": min(selected["forward"], selected["reverse"], selected["wrong"]) - mlp_score,
        "controls_passed": controls,
        "selected_transfer_passed": bool(selected["exact_pass"] and controls),
    }


def run(device_name: str) -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal run already complete")
    if not PREAUDIT.exists() or read_json(PREAUDIT).get("all_checks_passed") is not True:
        raise RuntimeError("independent preaudit must pass")
    protocol, material = verify_protocol()
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal run requires CUDA")
    device = torch.device("cuda")
    qualification_rows = partition_rows(material, "qualification")
    selection_rows = partition_rows(material, "selection")
    oracle = partition_rows(material, "oracle")
    qualification_records: list[dict[str, Any]] = []
    discoveries: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    selected_by_depth: dict[str, list[int]] = {}
    started = time.perf_counter()

    for architecture, config in ARCHITECTURES.items():
        seed = SEEDS[architecture][DISCOVERY_INDEX]
        p1266.set_seed(seed)
        model, training = p1266.task_module.train_model(config, seed, device)
        qualification = qualify_model(model, training, qualification_rows, device)
        qualification_records.append({"architecture": architecture, "seed": seed, "replicate": DISCOVERY_INDEX, **qualification})
        if not qualification["passed"]:
            raise RuntimeError(f"frozen discovery model lost behavior qualification: {architecture}")
        masks = all_masks(config.layers)
        sampled = [with_bounds(row) for row in evaluate_masks(model, selection_rows, masks, device)]
        chosen = select_mask(sampled)
        selection_abstained = chosen is None
        exact = evaluate_masks(model, oracle, masks, device)
        exact_map = {tuple(row["mask"]): row for row in exact}
        for row in sampled:
            row["population"] = exact_map[tuple(row["mask"])]
            row["population_pass"] = exact_pass(row["population"])
            row["false_authorization"] = bool(row["certificate_pass"] and not row["population_pass"])
        if selection_abstained:
            chosen = list(range(config.layers))
        selected_by_depth[architecture] = chosen
        discovery_record = {
            "architecture": architecture,
            "seed": seed,
            "selected_mask": chosen,
            "selected_mask_name": mask_name(chosen),
            "selected_cardinality": len(chosen),
            "selection_abstained": selection_abstained,
            "mask_ledger": sampled,
        }
        discoveries.append(discovery_record)
        write_jsonl(DISCOVERY, discoveries)
        result = evaluate_fixed_model(model, architecture, seed, DISCOVERY_INDEX, chosen, qualification, oracle, device, exact)
        results.append(result)
        write_jsonl(MODELS, results)
        write_jsonl(QUALIFICATION, qualification_records)
        print(canonical_json({"architecture": architecture, "role": "discovery", "mask": chosen, "transfer": result["selected_transfer_passed"], "minimal": result["shared_minimality_passed"]}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    for architecture, config in ARCHITECTURES.items():
        chosen = selected_by_depth[architecture]
        for replicate in HELDOUT_INDICES:
            seed = SEEDS[architecture][replicate]
            p1266.set_seed(seed)
            model, training = p1266.task_module.train_model(config, seed, device)
            qualification = qualify_model(model, training, qualification_rows, device)
            qualification_records.append({"architecture": architecture, "seed": seed, "replicate": replicate, **qualification})
            if qualification["passed"]:
                result = evaluate_fixed_model(model, architecture, seed, replicate, chosen, qualification, oracle, device)
                results.append(result)
                print(canonical_json({"architecture": architecture, "role": "heldout", "replicate": replicate, "mask": chosen, "transfer": result["selected_transfer_passed"], "minimal": result["shared_minimality_passed"]}), flush=True)
            else:
                print(canonical_json({"architecture": architecture, "role": "heldout", "replicate": replicate, "qualification_failed": True}), flush=True)
            write_jsonl(QUALIFICATION, qualification_records)
            write_jsonl(MODELS, results)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    write_jsonl(QUALIFICATION, qualification_records)
    write_jsonl(DISCOVERY, discoveries)
    write_jsonl(MODELS, results)
    run_summary = {
        "phase": PHASE,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "device": torch.cuda.get_device_name(0),
        "qualification_records": len(qualification_records),
        "discovery_depths": len(discoveries),
        "model_results": len(results),
        "qualification_hash": file_sha256(QUALIFICATION),
        "discovery_hash": file_sha256(DISCOVERY),
        "models_hash": file_sha256(MODELS),
        "protocol_digest": protocol["protocol_digest"],
        "pretrained_model_loaded": False,
    }
    run_summary["run_digest"] = digest({"qualification": qualification_records, "discovery": discoveries, "models": results})
    atomic_json(SUMMARY, run_summary)
    atomic_json(COMPLETE, {"status": "formal_run_complete", "created_at_utc": utc_now(), "run_digest": run_summary["run_digest"]})


def summarize(discoveries: list[dict[str, Any]], models: list[dict[str, Any]], qualifications: list[dict[str, Any]]) -> dict[str, Any]:
    all_selection = [row for discovery in discoveries for row in discovery["mask_ledger"]]
    false_authorizations = sum(row["false_authorization"] for row in all_selection)
    robust = [row for row in all_selection if row["robust_actionable"]]
    robust_coverage = sum(row["certificate_pass"] == row["population_pass"] for row in robust) / max(1, len(robust))
    heldout = [row for row in models if row["role"] == "heldout"]
    heldout_pass = [row for row in heldout if row["selected_transfer_passed"]]
    heldout_depth = {architecture: sum(row["architecture"] == architecture for row in heldout_pass) for architecture in ARCHITECTURES}
    all_pass = [row for row in models if row["selected_transfer_passed"]]
    minimal = [row for row in models if row["shared_minimality_passed"]]
    sparse_depths = {
        discovery["architecture"]: discovery["selected_cardinality"] <= math.ceil(ARCHITECTURES[discovery["architecture"]].layers * THRESHOLDS["sparse_fraction_max"])
        for discovery in discoveries
    }
    gates = {
        "G-FROZEN-BEHAVIOR-POPULATION": len(qualifications) == THRESHOLDS["all_models_required"] and all(row["passed"] for row in qualifications),
        "G-DISCOVERY-COMPLETE": (
            len(discoveries) == THRESHOLDS["discovery_depths_required"]
            and all(discovery["selected_mask"] and not discovery["selection_abstained"] for discovery in discoveries)
        ),
        "G-ZERO-FALSE-AUTHORIZATION": false_authorizations <= THRESHOLDS["false_authorizations_max"],
        "G-ROBUST-COVERAGE": bool(robust) and robust_coverage >= THRESHOLDS["robust_coverage_min"],
        "G-CONTROLS": len(models) == THRESHOLDS["all_models_required"] and all(row["controls_passed"] for row in models),
        "G-HELDOUT-SAME-MASK-TRANSFER": len(heldout_pass) == THRESHOLDS["heldout_models_required"] and all(value == THRESHOLDS["heldout_per_depth_required"] for value in heldout_depth.values()),
        "G-NO-PRETRAINED": True,
    }
    passed = all(gates.values())
    shared_minimality = len(minimal) == THRESHOLDS["all_models_required"]
    sparse = len(sparse_depths) == len(ARCHITECTURES) and all(sparse_depths.values())
    if passed and shared_minimality and sparse:
        decision = "cross_seed_sparse_minimal_layer_coalition_confirmed"
    elif passed:
        decision = "cross_seed_layer_coalition_transfer_confirmed_with_minimality_boundary"
    else:
        decision = "cross_seed_layer_coalition_transfer_not_confirmed"
    return {
        "qualification_models": len(qualifications),
        "discovery_depths": len(discoveries),
        "candidate_masks": len(all_selection),
        "false_authorizations": false_authorizations,
        "robust_masks": len(robust),
        "robust_coverage": robust_coverage,
        "selected_masks": {row["architecture"]: row["selected_mask"] for row in discoveries},
        "selected_cardinalities": {row["architecture"]: row["selected_cardinality"] for row in discoveries},
        "all_transfer_models": len(all_pass),
        "heldout_transfer_models": len(heldout_pass),
        "heldout_transfer_per_depth": heldout_depth,
        "shared_minimality_models": len(minimal),
        "shared_minimality_passed": shared_minimality,
        "sparse_per_depth": sparse_depths,
        "sparse_passed": sparse,
        "gates": gates,
        "passed": passed,
        "decision": decision,
    }


def analyze() -> None:
    if not COMPLETE.exists():
        raise RuntimeError("formal run incomplete")
    protocol, _ = verify_protocol()
    qualifications = read_jsonl(QUALIFICATION)
    discoveries = read_jsonl(DISCOVERY)
    models = read_jsonl(MODELS)
    result = summarize(discoveries, models, qualifications)
    head_authorized = bool(result["passed"] and result["shared_minimality_passed"] and result["sparse_passed"])
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        **result,
        "authorization": {
            "head_or_microcomponent_contract": head_authorized,
            "synthetic_layer_coalition_search_closed": True,
            "automatic_pretrained_run": False,
            "qwen3": False,
            "glm4": False,
            "ds7b": False,
        },
        "structured_scope": protocol["structured_scope"],
        "protocol_digest": protocol["protocol_digest"],
        "qualification_hash": file_sha256(QUALIFICATION),
        "discovery_hash": file_sha256(DISCOVERY),
        "models_hash": file_sha256(MODELS),
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"decision": result["decision"], "passed": result["passed"], "masks": result["selected_masks"], "heldout": result["heldout_transfer_models"], "minimal": result["shared_minimality_models"], "head": head_authorized}))


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
