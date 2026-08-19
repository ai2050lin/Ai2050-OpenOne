"""Phase1271/C019: cross-layer answer-position micro-write trajectories."""

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
import phase1270_c018_answer_excluded_causal_recomputation as p1270


PHASE = 1271
CAMPAIGN = "C019"
CONTRACT_ID = "EXP-C019-WP01-001"
OUT = ROOT / "tests/glm5/result/phase1271_c019_cross_layer_micro_write_trajectory"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_micro_write_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
QUALIFICATION = OUT / "raw/behavior_qualification.jsonl"
MODELS = OUT / "raw/model_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1271_c019_cross_layer_micro_write_trajectory_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C019-WP01-001.json"
PHASE1270_FINAL = ROOT / "tests/glm5/result/phase1270_c018_answer_excluded_causal_recomputation/analysis/final.json"
PHASE1270_AUDIT = ROOT / "tests/glm5/result/phase1270_c018_answer_excluded_causal_recomputation/audit/independent_final_audit.json"
PHASE1270_COMPLETE = ROOT / "tests/glm5/result/phase1270_c018_answer_excluded_causal_recomputation/raw/FORMAL_RUN_COMPLETE.json"

ARCHITECTURES = p1266.ARCHITECTURES
SEED_POOLS = {
    "shallow4": [1_271_401_001, 1_271_401_101, 1_271_401_201, 1_271_401_301, 1_271_401_401],
    "middle6": [1_271_601_001, 1_271_601_101, 1_271_601_201, 1_271_601_301, 1_271_601_401],
    "deep8": [1_271_801_001, 1_271_801_101, 1_271_801_201, 1_271_801_301, 1_271_801_401],
}
DEVELOPMENT_SEEDS = {1_271_699_991, 1_271_699_992, 1_271_699_993, 1_271_699_994, 1_271_699_995, 1_271_699_996}
SELECT_PER_DEPTH = 3
MATERIAL_SEEDS = {"qualification": 1_271_910_001, "confirmation": 1_271_930_001}
PARTITION_COUNTS = {"qualification": 1024, "oracle": 3456, "confirmation": 1024}
SELECTION_DRAWS = 32768
GLOBAL_ERROR_BUDGET = 0.01
PASS_MIN = 0.95
NULL_MAX = 0.05
POSITIVE_MIN = 0.999
ADVANTAGE_MIN = 0.05
ROBUST_MULTIPLIER = 2.0
MAX_PROGRAMS_PER_MODEL = max(config.layers for config in ARCHITECTURES.values()) + 6
MAX_EVENTS = MAX_PROGRAMS_PER_MODEL * len(ARCHITECTURES) * SELECT_PER_DEPTH
CERTIFICATE_RADIUS = math.sqrt(
    math.log(2.0 * MAX_EVENTS * 3.0 / GLOBAL_ERROR_BUDGET) / (2.0 * SELECTION_DRAWS)
)
THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "executor_gap_max": 2.0e-4,
    "expected_accuracy_min": PASS_MIN,
    "wrong_false_target_max": NULL_MAX,
    "positive_control_accuracy_min": POSITIVE_MIN,
    "null_switch_accuracy_max": NULL_MAX,
    "component_advantage_min": ADVANTAGE_MIN,
    "certificate_radius": CERTIFICATE_RADIUS,
    "false_authorizations_max": 0,
    "robust_coverage_min": 0.90,
    "teacher_forced_models_min": 6,
    "teacher_forced_per_depth_min": 2,
    "proper_prefix_models_min": 6,
    "proper_prefix_per_depth_min": 2,
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


def make_material() -> list[dict[str, Any]]:
    rows = p1269.sample_worlds("qualification", PARTITION_COUNTS["qualification"], MATERIAL_SEEDS["qualification"])
    rows.extend(p1266.enumerate_oracle_worlds())
    rows.extend(p1269.sample_worlds("confirmation", PARTITION_COUNTS["confirmation"], MATERIAL_SEEDS["confirmation"]))
    return rows


def program_registry(layers: int) -> list[dict[str, Any]]:
    programs = [
        {
            "name": f"attn_prefix_{end}",
            "stage": "attn_write",
            "layers": list(range(end + 1)),
            "position": 22,
            "donor_panel": "h11",
            "role": "attention_prefix",
            "prefix_end": end,
        }
        for end in range(layers)
    ]
    programs.extend(
        [
            {"name": "attn_full_wrong", "stage": "attn_write", "layers": list(range(layers)), "position": 22, "donor_panel": "hwrong11", "role": "wrong_identity"},
            {"name": "mlp_full_correct", "stage": "mlp_write", "layers": list(range(layers)), "position": 22, "donor_panel": "h11", "role": "matched_component"},
            {"name": "mlp_full_wrong", "stage": "mlp_write", "layers": list(range(layers)), "position": 22, "donor_panel": "hwrong11", "role": "matched_component_wrong"},
            {"name": "after_block_full_correct", "stage": "after_block", "layers": list(range(layers)), "position": 22, "donor_panel": "h11", "role": "positive_replay_control"},
            {"name": "attn_pre_source_null", "stage": "attn_write", "layers": list(range(layers)), "position": 2, "donor_panel": "h11", "role": "causal_prefix_null"},
            {"name": "attn_position8_descriptive", "stage": "attn_write", "layers": list(range(layers)), "position": 8, "donor_panel": "h11", "role": "destination_control_descriptive"},
        ]
    )
    return programs


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predecessor = read_json(PHASE1270_FINAL)
    audit = read_json(PHASE1270_AUDIT)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1271.c019.cross_layer_micro_write.v1",
        "claim_type": "free_transformer_teacher_forced_attention_write_trajectory",
        "question": "Is an answer-position attention-write trajectory bidirectionally identity-specific, and does a proper prefix hand control back to natural computation?",
        "phase1270_dependency": {
            "formal_complete": PHASE1270_COMPLETE.exists(),
            "decision": predecessor.get("decision"),
            "passed": predecessor.get("passed"),
            "final_hash": file_sha256(PHASE1270_FINAL),
            "audit_passed": audit.get("all_checks_passed"),
            "audit_hash": file_sha256(PHASE1270_AUDIT),
        },
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "program_registry": {name: program_registry(config.layers) for name, config in ARCHITECTURES.items()},
        "seed_pools": SEED_POOLS,
        "development_seeds_excluded": sorted(DEVELOPMENT_SEEDS),
        "behavior_selection": {"rule": "first three behavior-qualified seeds per depth", "select_per_depth": SELECT_PER_DEPTH, "partition": "qualification"},
        "partitions": PARTITION_COUNTS,
        "material_seeds": MATERIAL_SEEDS,
        "world_digest": digest([{"row_id": row["row_id"], "partition": row["partition"], "row_digest": row["row_digest"]} for row in rows]),
        "selection": {
            "draws": SELECTION_DRAWS,
            "radius": CERTIFICATE_RADIUS,
            "shortest_prefix_rule": "first attention prefix whose forward and reverse lower bounds are at least 0.95",
            "wrong_identity_rule": "wrong expected-answer lower bounds at least 0.95 and h11 false-target upper bound at most 0.05",
        },
        "claims": {
            "primary": "full teacher-forced attention trajectory plus wrong-identity specificity",
            "secondary_prefix": "selected and confirmed prefix ends before final layer",
            "secondary_component": "confirmation attention score exceeds matched MLP score by at least 0.05",
        },
        "thresholds": THRESHOLDS,
        "budgets": {"max_formal_runs": 1, "max_adaptive_rounds": 0, "max_gpu_hours": 3.5},
        "hard_stops": [
            "Development seeds and outputs are excluded from formal evidence.",
            "Behavior screening cannot inspect micro-write results.",
            "Oracle population scores cannot select the shortest prefix.",
            "Confirmation cannot select prefix, component, threshold, or seed.",
            "Teacher-forced sufficiency is not self-generated natural execution.",
            "No head search, learned donor, or pretrained model is run automatically.",
        ],
        "structured_scope": {
            "task": "synthetic cyclic-code",
            "population": "behavior-qualified small free same-executor Transformers",
            "intervention": "same-case teacher-forced component writes at answer position",
            "natural_language": False,
            "pretrained": False,
        },
        "source_hashes": {
            "main": file_sha256(Path(__file__).resolve()),
            "auditor": file_sha256(AUDITOR),
            "contract": file_sha256(CONTRACT),
            "phase1270": file_sha256(ROOT / "tests/glm5/phase1270_c018_answer_excluded_causal_recomputation.py"),
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
    predecessor = read_json(PHASE1270_FINAL)
    audit = read_json(PHASE1270_AUDIT)
    if predecessor.get("decision") != "answer_excluded_upstream_recomputation_not_confirmed" or predecessor.get("passed") is not False:
        raise RuntimeError("Phase1270 boundary not frozen")
    if audit.get("all_checks_passed") is not True or not PHASE1270_COMPLETE.exists():
        raise RuntimeError("Phase1270 audit dependency failed")
    if not AUDITOR.exists() or not CONTRACT.exists():
        raise RuntimeError("auditor and contract must exist")
    rows = make_material()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "seed_pool": sum(map(len, SEED_POOLS.values())), "radius": CERTIFICATE_RADIUS}))


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


def block_parts(block, hidden: torch.Tensor, causal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = block.attn_norm(hidden)
    batch, length, width = normalized.shape
    qkv = block.attn.qkv(normalized).view(batch, length, 3, block.attn.heads, block.attn.head_dim)
    query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
    weights = torch.softmax(scores.masked_fill(causal[None, None], float("-inf")), dim=-1)
    attended = torch.matmul(weights, value).transpose(1, 2).contiguous().view(batch, length, width)
    attn_write = block.attn.out(attended)
    after_attn = hidden + attn_write
    mlp_write = block.mlp(block.mlp_norm(after_attn))
    return attn_write, after_attn, mlp_write


def capture_micro(model, ids: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    hidden = model.embed(ids)
    causal = torch.triu(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device), diagonal=1)
    traces = []
    for block in model.blocks:
        attn_write, after_attn, mlp_write = block_parts(block, hidden, causal)
        hidden = after_attn + mlp_write
        traces.append({"attn_write": attn_write.detach().clone(), "mlp_write": mlp_write.detach().clone(), "after_block": hidden.detach().clone()})
    return traces


def forward_program(model, ids: torch.Tensor, donor: list[dict[str, torch.Tensor]], program: dict[str, Any]) -> torch.Tensor:
    hidden = model.embed(ids)
    causal = torch.triu(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device), diagonal=1)
    selected = set(program["layers"])
    position = int(program["position"])
    stage = program["stage"]
    for index, block in enumerate(model.blocks):
        attn_write, after_attn, mlp_write = block_parts(block, hidden, causal)
        if index in selected and stage == "attn_write":
            attn_write = attn_write.clone()
            attn_write[:, position] = donor[index]["attn_write"][:, position]
            after_attn = hidden + attn_write
            mlp_write = block.mlp(block.mlp_norm(after_attn))
        if index in selected and stage == "mlp_write":
            mlp_write = mlp_write.clone()
            mlp_write[:, position] = donor[index]["mlp_write"][:, position]
        hidden = after_attn + mlp_write
        if index in selected and stage == "after_block":
            hidden = hidden.clone()
            hidden[:, position] = donor[index]["after_block"][:, position]
    return model.lm_head(model.final_norm(hidden))


def evaluate_programs(
    model,
    rows: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, dict[str, list[bool]]]:
    outcomes = {program["name"]: {"patch_expected": [], "reverse_base": [], "patch_false_target": []} for program in programs}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            ids = {
                panel: torch.tensor([row[f"{panel}_ids"] for row in batch], device=device)
                for panel in ("h01", "h11", "hwrong11")
            }
            traces = {panel: capture_micro(model, value) for panel, value in ids.items()}
            answers = {
                panel: torch.tensor([row["answers"][panel] for row in batch], device=device)
                for panel in ("h01", "h11", "hwrong11")
            }
            for program in programs:
                donor_panel = program["donor_panel"]
                patch_logits = forward_program(model, ids["h01"], traces[donor_panel], program)
                reverse_logits = forward_program(model, ids[donor_panel], traces["h01"], program)
                patch_pred = torch.argmax(patch_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
                reverse_pred = torch.argmax(reverse_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
                record = outcomes[program["name"]]
                record["patch_expected"].extend((patch_pred == answers[donor_panel]).cpu().tolist())
                record["reverse_base"].extend((reverse_pred == answers["h01"]).cpu().tolist())
                record["patch_false_target"].extend((patch_pred == answers["h11"]).cpu().tolist())
    return outcomes


def selection_indices(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7_777_271)
    return rng.integers(0, PARTITION_COUNTS["oracle"], size=SELECTION_DRAWS)


def bounds(point: float) -> dict[str, float]:
    return {"point": point, "lower": max(0.0, point - CERTIFICATE_RADIUS), "upper": min(1.0, point + CERTIFICATE_RADIUS)}


def qualify_model(model, training: dict[str, Any], rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    natural_accuracy, executor_gap = p1268.evaluate_behavior(model, rows, device)
    passed = (
        min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], natural_accuracy)
        >= THRESHOLDS["behavior_accuracy_min"]
        and executor_gap <= THRESHOLDS["executor_gap_max"]
    )
    return {"training": training, "qualification_accuracy": natural_accuracy, "executor_gap": executor_gap, "passed": passed}


def program_ledger(
    programs: list[dict[str, Any]],
    outcomes: dict[str, dict[str, list[bool]]],
    indices: np.ndarray | None,
) -> list[dict[str, Any]]:
    ledger = []
    for program in programs:
        raw = outcomes[program["name"]]
        patch = np.asarray(raw["patch_expected"], dtype=np.float64)
        reverse = np.asarray(raw["reverse_base"], dtype=np.float64)
        false_target = np.asarray(raw["patch_false_target"], dtype=np.float64)
        if indices is None:
            sample_patch, sample_reverse, sample_false = float(patch.mean()), float(reverse.mean()), float(false_target.mean())
        else:
            sample_patch, sample_reverse, sample_false = float(patch[indices].mean()), float(reverse[indices].mean()), float(false_target[indices].mean())
        patch_bounds, reverse_bounds, false_bounds = bounds(sample_patch), bounds(sample_reverse), bounds(sample_false)
        population_patch, population_reverse, population_false = float(patch.mean()), float(reverse.mean()), float(false_target.mean())
        expected_score = min(population_patch, population_reverse)
        exact_pass = expected_score >= PASS_MIN
        certificate_pass = min(patch_bounds["lower"], reverse_bounds["lower"]) >= PASS_MIN
        robust = expected_score >= PASS_MIN + ROBUST_MULTIPLIER * CERTIFICATE_RADIUS
        wrong_specificity = program["role"] in ("wrong_identity", "matched_component_wrong")
        specificity_pass = certificate_pass and (not wrong_specificity or false_bounds["upper"] <= NULL_MAX)
        ledger.append(
            {
                **program,
                "population_patch_expected": population_patch,
                "population_reverse_base": population_reverse,
                "population_expected_score": expected_score,
                "population_patch_false_target": population_false,
                "sample_patch_expected": sample_patch,
                "sample_reverse_base": sample_reverse,
                "sample_patch_false_target": sample_false,
                "patch_bounds": patch_bounds,
                "reverse_bounds": reverse_bounds,
                "false_target_bounds": false_bounds,
                "exact_pass": exact_pass,
                "certificate_pass": certificate_pass,
                "robust_actionable": robust,
                "specificity_pass": specificity_pass,
            }
        )
    return ledger


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
    programs = program_registry(config.layers)
    oracle = evaluate_programs(model, partition_rows(rows, "oracle"), programs, device)
    ledger = program_ledger(programs, oracle, selection_indices(seed))
    prefixes = [item for item in ledger if item["role"] == "attention_prefix"]
    selected_prefix = next((item["name"] for item in prefixes if item["certificate_pass"]), None)
    confirmation_raw = evaluate_programs(model, partition_rows(rows, "confirmation"), programs, device)
    confirmation = program_ledger(programs, confirmation_raw, None)
    confirmation_map = {item["name"]: item for item in confirmation}
    full_name = f"attn_prefix_{config.layers - 1}"
    full = next(item for item in ledger if item["name"] == full_name)
    full_confirmation = confirmation_map[full_name]
    wrong = next(item for item in ledger if item["name"] == "attn_full_wrong")
    wrong_confirmation = confirmation_map["attn_full_wrong"]
    replay_confirmation = confirmation_map["after_block_full_correct"]
    null_confirmation = confirmation_map["attn_pre_source_null"]
    mlp_confirmation = confirmation_map["mlp_full_correct"]
    selected_confirmation = confirmation_map[selected_prefix] if selected_prefix else None
    controls_passed = (
        replay_confirmation["population_expected_score"] >= POSITIVE_MIN
        and max(null_confirmation["population_patch_expected"], null_confirmation["population_reverse_base"]) <= NULL_MAX
    )
    teacher_passed = (
        full["certificate_pass"]
        and full_confirmation["exact_pass"]
        and wrong["specificity_pass"]
        and wrong_confirmation["exact_pass"]
        and wrong_confirmation["population_patch_false_target"] <= NULL_MAX
        and controls_passed
    )
    selected_end = next((item["prefix_end"] for item in prefixes if item["name"] == selected_prefix), None)
    proper_prefix = (
        teacher_passed
        and selected_end is not None
        and selected_end < config.layers - 1
        and selected_confirmation is not None
        and selected_confirmation["exact_pass"]
    )
    component_advantage = full_confirmation["population_expected_score"] - mlp_confirmation["population_expected_score"]
    return {
        "model_key": f"{architecture}_s{selected_index}_p{pool_index}",
        "architecture": architecture,
        "seed": seed,
        "pool_index": pool_index,
        "selected_index": selected_index,
        "qualification": qualification,
        "event_ledger": ledger,
        "confirmation_ledger": confirmation,
        "selected_prefix": selected_prefix,
        "selected_prefix_end": selected_end,
        "selected_prefix_relative": selected_end / max(1, config.layers - 1) if selected_end is not None else None,
        "controls_passed": controls_passed,
        "teacher_forced_attention_passed": teacher_passed,
        "proper_prefix_passed": proper_prefix,
        "attention_confirmation_score": full_confirmation["population_expected_score"],
        "mlp_confirmation_score": mlp_confirmation["population_expected_score"],
        "attention_over_mlp_advantage": component_advantage,
        "component_advantage_passed": component_advantage >= ADVANTAGE_MIN,
        "position8_confirmation_score": confirmation_map["attn_position8_descriptive"]["population_expected_score"],
    }


def summarize(qualification: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_events = [event for row in rows for event in row["event_ledger"]]
    false_authorizations = sum(event["certificate_pass"] and not event["exact_pass"] for event in all_events)
    robust_events = [event for event in all_events if event["robust_actionable"]]
    robust_coverage = sum(event["certificate_pass"] for event in robust_events) / max(1, len(robust_events))
    selected_per_depth = {architecture: sum(row["architecture"] == architecture for row in rows) for architecture in ARCHITECTURES}
    teachers = [row for row in rows if row["teacher_forced_attention_passed"]]
    proper = [row for row in rows if row["proper_prefix_passed"]]
    advantage = [row for row in rows if row["component_advantage_passed"]]
    teacher_depth = {architecture: sum(row["architecture"] == architecture for row in teachers) for architecture in ARCHITECTURES}
    proper_depth = {architecture: sum(row["architecture"] == architecture for row in proper) for architecture in ARCHITECTURES}
    advantage_depth = {architecture: sum(row["architecture"] == architecture for row in advantage) for architecture in ARCHITECTURES}
    qualification_complete = len(rows) == THRESHOLDS["selected_models_required"] and all(value == SELECT_PER_DEPTH for value in selected_per_depth.values())
    teacher_breadth = len(teachers) >= THRESHOLDS["teacher_forced_models_min"] and all(value >= THRESHOLDS["teacher_forced_per_depth_min"] for value in teacher_depth.values())
    proper_breadth = len(proper) >= THRESHOLDS["proper_prefix_models_min"] and all(value >= THRESHOLDS["proper_prefix_per_depth_min"] for value in proper_depth.values())
    gates = {
        "G-BEHAVIOR-SCREEN-COMPLETE": qualification_complete,
        "G-ZERO-FALSE-AUTHORIZATION": false_authorizations <= THRESHOLDS["false_authorizations_max"],
        "G-ROBUST-COVERAGE": bool(robust_events) and robust_coverage >= THRESHOLDS["robust_coverage_min"],
        "G-CONTROL-SUITE": len(rows) == THRESHOLDS["selected_models_required"] and all(row["controls_passed"] for row in rows),
        "G-TEACHER-FORCED-ATTENTION-BREADTH": teacher_breadth,
        "G-NO-PRETRAINED": True,
    }
    prefix_relative = [row["selected_prefix_relative"] for row in rows if row["selected_prefix_relative"] is not None]
    passed = all(gates.values())
    return {
        "attempted_seeds": len(qualification),
        "selected_models": len(rows),
        "selected_per_depth": selected_per_depth,
        "events": len(all_events),
        "false_authorizations": false_authorizations,
        "robust_events": len(robust_events),
        "robust_coverage": robust_coverage,
        "teacher_forced_attention_models": len(teachers),
        "teacher_forced_attention_per_depth": teacher_depth,
        "proper_prefix_models": len(proper),
        "proper_prefix_per_depth": proper_depth,
        "proper_prefix_breadth_passed": proper_breadth,
        "component_advantage_models": len(advantage),
        "component_advantage_per_depth": advantage_depth,
        "selected_prefix_relative_median": float(np.median(prefix_relative)) if prefix_relative else None,
        "gates": gates,
        "decision": "teacher_forced_attention_write_trajectory_confirmed" if passed else "teacher_forced_attention_write_trajectory_not_confirmed",
        "passed": passed,
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
                print(canonical_json({"architecture": architecture, "selected": selected, "seed": seed, "teacher": result["teacher_forced_attention_passed"], "prefix": result["selected_prefix"], "proper": result["proper_prefix_passed"], "advantage": result["attention_over_mlp_advantage"]}), flush=True)
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
            "layer_coalition_minimality_contract": summary["passed"],
            "self_sustaining_prefix_contract": summary["passed"] and summary["proper_prefix_breadth_passed"],
            "head_search": False,
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
    print(canonical_json({"decision": summary["decision"], "passed": summary["passed"], "teacher": summary["teacher_forced_attention_models"], "proper": summary["proper_prefix_models"], "advantage": summary["component_advantage_models"]}))


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
