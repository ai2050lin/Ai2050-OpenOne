#!/usr/bin/env python3
"""Phase1254: free-Transformer external validity of the C006 edge camera.

The internal algorithm is not planted. Discovery greedily ranks typed QK, OV
and MLP edge interventions; selection freezes one sparse prefix; confirmation
tests correct rescue, wrong-identity rejection, matched-null specificity and
reverse blocking without further component selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer
from phase1251_c004_causal_slice_competition import (
    CANDIDATE_SLICE,
    build_sequence,
    train_model,
)


PHASE = 1254
CONTRACT_ID = "EXP-C007-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1254_c007_free_transformer_edge_external_validity_audit.py"
MODEL_DEPENDENCY = ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py"
TASK_DEPENDENCY = ROOT / "tests/glm5/phase1251_c004_causal_slice_competition.py"
OUT = ROOT / "tests/glm5/result/phase1254_c007_free_transformer_edge_external_validity"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_counterfactuals.jsonl"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
MODELS = OUT / "raw/model_results.jsonl"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/edge_external_validity_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

ARCHITECTURES = {
    "shallow4": ModelConfig(layers=4, width=96, heads=4, mlp_width=192, max_length=23, vocab_size=22),
    "middle6": ModelConfig(layers=6, width=96, heads=4, mlp_width=192, max_length=23, vocab_size=22),
    "deep8": ModelConfig(layers=8, width=96, heads=4, mlp_width=192, max_length=23, vocab_size=22),
}
REPLICATES = 2
MODEL_SEEDS = {
    "shallow4_r0": 1_254_401_001,
    "shallow4_r1": 1_254_401_101,
    "middle6_r0": 1_254_601_001,
    "middle6_r1": 1_254_601_101,
    "deep8_r0": 1_254_801_001,
    "deep8_r1": 1_254_801_101,
}
WORLD_SEED = 1_254_900_001
WORLD_COUNTS = {"discovery": 32, "selection": 32, "confirmation": 64}
PREFIX_SIZES = (1, 2, 4, 8, 12)
MAX_GREEDY_COMPONENTS = max(PREFIX_SIZES)
THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "native_explicit_logit_gap_max": 2.0e-4,
    "correct_cosine_min": 0.85,
    "correct_relative_error_max": 0.65,
    "correct_projection_min": 0.55,
    "correct_accuracy_min": 0.90,
    "wrong_cosine_max": 0.65,
    "identity_cosine_margin_min": 0.20,
    "null_effect_fraction_max": 0.15,
    "block_remaining_fraction_max": 0.40,
    "loo_worsening_min": 0.02,
    "breadth_models_min": 4,
    "breadth_per_depth_min": 1,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            output.update(chunk)
    return output.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_key(architecture: str, replicate: int) -> str:
    return f"{architecture}_r{replicate}"


def make_worlds(
    seed: int = WORLD_SEED,
    counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    counts = counts or WORLD_COUNTS
    rng = np.random.default_rng(seed)
    partitions = [name for name, count in counts.items() for _ in range(count)]
    rows: list[dict[str, Any]] = []
    for group, partition in enumerate(partitions):
        codes = rng.choice(4, 2, replace=False).astype(int).tolist()
        alternatives = [value for value in range(4) if value not in codes]
        rng.shuffle(alternatives)
        target_code, wrong_code = alternatives
        null_code = target_code
        shift = int(rng.integers(4))
        order = rng.permutation(4).astype(int).tolist()
        target_codes = [target_code, codes[1]]
        wrong_codes = [wrong_code, codes[1]]
        null_codes = [codes[0], null_code]
        base, _ = build_sequence(1, codes, shift, order)
        target, _ = build_sequence(1, target_codes, shift, order)
        wrong, _ = build_sequence(1, wrong_codes, shift, order)
        null, _ = build_sequence(1, null_codes, shift, order)
        row = {
            "row_id": f"g{group:03d}",
            "group": group,
            "partition": partition,
            "base_ids": base,
            "target_ids": target,
            "wrong_ids": wrong,
            "null_ids": null,
            "answers": {
                "base": (codes[0] + shift) % 4,
                "target": (target_code + shift) % 4,
                "wrong": (wrong_code + shift) % 4,
                "null": (codes[0] + shift) % 4,
            },
        }
        row["row_digest"] = digest(row)
        rows.append(row)
    return rows


def component_ids(config: ModelConfig) -> list[str]:
    result: list[str] = []
    for layer in range(config.layers):
        for head in range(config.heads):
            result.extend((f"L{layer:02d}.H{head:02d}.qk", f"L{layer:02d}.H{head:02d}.ov"))
        result.append(f"L{layer:02d}.mlp")
    return result


def component_role(component: str) -> str:
    return component.rsplit(".", 1)[-1]


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "schema_version": "phase1254.c007.free_edge_external_validity.protocol.v1",
        "contract_id": CONTRACT_ID,
        "claim_type": "free_transformer_component_edge_external_validity",
        "question": "Does the C006 typed edge camera yield a sparse, identity-specific, bidirectionally causal coalition in freely trained Transformers?",
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_seeds": MODEL_SEEDS,
        "partitions": WORLD_COUNTS,
        "row_count": len(rows),
        "world_digest": digest([{key: row[key] for key in ("row_id", "partition", "row_digest")} for row in rows]),
        "component_ontology": {
            "qk": "replace one head's full causal attention-weight matrix",
            "ov": "replace one head's post-attention payload before output projection",
            "mlp": "replace one layer's full MLP residual write",
            "components_by_architecture": {name: len(component_ids(config)) for name, config in ARCHITECTURES.items()},
        },
        "selection": {
            "discovery": "greedy correct-rescue relative-error reduction only",
            "candidate_prefix_sizes": list(PREFIX_SIZES),
            "selection_objective": "correct cosine - correct relative error - null fraction - positive wrong cosine - block remaining fraction - 0.01*size",
            "confirmation": "frozen component identities and prefix size",
        },
        "thresholds": THRESHOLDS,
        "budgets": {"max_gpu_hours": 1.5, "max_formal_runs": 1, "max_adaptive_rounds": 0},
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "auditor": file_sha256(AUDITOR),
            "model_dependency": file_sha256(MODEL_DEPENDENCY),
            "task_dependency": file_sha256(TASK_DEPENDENCY),
        },
        "hard_stops": [
            "Free-network task truth and endpoint truth do not provide internal mechanism truth.",
            "Confirmation cannot select components, prefix size, thresholds or models.",
            "Behavior-unqualified seeds remain in the breadth denominator and are not replaced.",
            "Correct rescue alone is insufficient; wrong identity, matched null and reverse blocking are conjunctive gates.",
            "A component coalition is a typed causal implementation candidate, not a semantic circuit or minimal unique algorithm.",
            "Failure blocks pretrained-model escalation; pass authorizes only one separately frozen Qwen3 contract.",
        ],
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
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "fp32_parameters_and_execution",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    rows = make_worlds()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows)}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["source_hashes"] != expected["source_hashes"] or protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol or source changed after preregistration")
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        if digest(value) != stored:
            raise RuntimeError("material digest mismatch")
    return protocol, rows


def explicit_forward(
    model: TinyCausalTransformer,
    input_ids: torch.Tensor,
    overrides: dict[str, torch.Tensor] | None = None,
    capture: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hidden = model.embed(input_ids)
    traces: dict[str, torch.Tensor] = {}
    length = input_ids.shape[1]
    causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1)
    for layer, block in enumerate(model.blocks):
        normalized = block.attn_norm(hidden)
        batch, _, width = normalized.shape
        qkv = block.attn.qkv(normalized).view(batch, length, 3, block.attn.heads, block.attn.head_dim)
        query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        scores = scores.masked_fill(causal[None, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if overrides:
            weights = weights.clone()
            for head in range(block.attn.heads):
                name = f"L{layer:02d}.H{head:02d}.qk"
                if name in overrides:
                    weights[:, head] = overrides[name]
        if capture:
            for head in range(block.attn.heads):
                traces[f"L{layer:02d}.H{head:02d}.qk"] = weights[:, head].detach().clone()
        attended = torch.matmul(weights, value)
        if overrides:
            attended = attended.clone()
            for head in range(block.attn.heads):
                name = f"L{layer:02d}.H{head:02d}.ov"
                if name in overrides:
                    attended[:, head] = overrides[name]
        if capture:
            for head in range(block.attn.heads):
                traces[f"L{layer:02d}.H{head:02d}.ov"] = attended[:, head].detach().clone()
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        hidden = hidden + block.attn.out(attended)
        mlp = block.mlp(block.mlp_norm(hidden))
        mlp_name = f"L{layer:02d}.mlp"
        if overrides and mlp_name in overrides:
            mlp = overrides[mlp_name]
        if capture:
            traces[mlp_name] = mlp.detach().clone()
        hidden = hidden + mlp
    return model.lm_head(model.final_norm(hidden)), traces


def centered(logits: torch.Tensor) -> torch.Tensor:
    values = logits[:, -1, CANDIDATE_SLICE].float()
    return values - values.mean(dim=-1, keepdim=True)


def effect_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p = predicted.reshape(-1).double()
    t = target.reshape(-1).double()
    p_norm = torch.linalg.vector_norm(p)
    t_norm = torch.linalg.vector_norm(t).clamp_min(1.0e-12)
    cosine = float((torch.dot(p, t) / (p_norm.clamp_min(1.0e-12) * t_norm)).item())
    return {
        "cosine": cosine,
        "relative_error": float((torch.linalg.vector_norm(p - t) / t_norm).item()),
        "projection": float((torch.dot(p, t) / torch.dot(t, t).clamp_min(1.0e-12)).item()),
        "norm_fraction": float((p_norm / t_norm).item()),
    }


def subset_trace(trace: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[indices] for name, value in trace.items()}


def evaluate_coalition(
    model: TinyCausalTransformer,
    ids: dict[str, torch.Tensor],
    logits: dict[str, torch.Tensor],
    traces: dict[str, dict[str, torch.Tensor]],
    indices: torch.Tensor,
    coalition: list[str],
) -> dict[str, Any]:
    base_ids = ids["base"][indices]
    base = centered(logits["base"][indices])
    target = centered(logits["target"][indices])
    target_effect = target - base

    def patch(receiver: str, donor: str) -> torch.Tensor:
        overrides = {name: traces[donor][name][indices] for name in coalition}
        patched, _ = explicit_forward(model, ids[receiver][indices], overrides)
        return centered(patched)

    correct = patch("base", "target") - base
    wrong = patch("base", "wrong") - base
    null = patch("base", "null") - base
    blocked_state = patch("target", "base")
    correct_metrics = effect_metrics(correct, target_effect)
    wrong_metrics = effect_metrics(wrong, target_effect)
    null_fraction = float((torch.linalg.vector_norm(null.double()) / torch.linalg.vector_norm(target_effect.double()).clamp_min(1.0e-12)).item())
    block_remaining = float((torch.linalg.vector_norm((blocked_state - base).double()) / torch.linalg.vector_norm(target_effect.double()).clamp_min(1.0e-12)).item())
    patched_logits, _ = explicit_forward(
        model,
        base_ids,
        {name: traces["target"][name][indices] for name in coalition},
    )
    target_answers = torch.argmax(logits["target"][indices, -1, CANDIDATE_SLICE], dim=-1)
    correct_accuracy = float((torch.argmax(patched_logits[:, -1, CANDIDATE_SLICE], dim=-1) == target_answers).float().mean().item())
    return {
        "correct": correct_metrics,
        "wrong": wrong_metrics,
        "identity_cosine_margin": correct_metrics["cosine"] - wrong_metrics["cosine"],
        "null_effect_fraction": null_fraction,
        "block_remaining_fraction": block_remaining,
        "correct_accuracy": correct_accuracy,
    }


def selection_objective(metrics: dict[str, Any], size: int) -> float:
    return (
        metrics["correct"]["cosine"]
        - metrics["correct"]["relative_error"]
        - metrics["null_effect_fraction"]
        - max(0.0, metrics["wrong"]["cosine"])
        - metrics["block_remaining_fraction"]
        - 0.01 * size
    )


def greedy_components(
    model: TinyCausalTransformer,
    ids: dict[str, torch.Tensor],
    logits: dict[str, torch.Tensor],
    traces: dict[str, dict[str, torch.Tensor]],
    discovery: torch.Tensor,
) -> list[str]:
    remaining = component_ids(model.config)
    selected: list[str] = []
    for _ in range(min(MAX_GREEDY_COMPONENTS, len(remaining))):
        candidates: list[tuple[float, str]] = []
        for component in remaining:
            metrics = evaluate_coalition(model, ids, logits, traces, discovery, selected + [component])
            candidates.append((-metrics["correct"]["relative_error"], component))
        _, winner = max(candidates, key=lambda item: (item[0], item[1]))
        selected.append(winner)
        remaining.remove(winner)
    return selected


def passes(metrics: dict[str, Any]) -> bool:
    return (
        metrics["correct"]["cosine"] >= THRESHOLDS["correct_cosine_min"]
        and metrics["correct"]["relative_error"] <= THRESHOLDS["correct_relative_error_max"]
        and metrics["correct"]["projection"] >= THRESHOLDS["correct_projection_min"]
        and metrics["correct_accuracy"] >= THRESHOLDS["correct_accuracy_min"]
        and metrics["wrong"]["cosine"] <= THRESHOLDS["wrong_cosine_max"]
        and metrics["identity_cosine_margin"] >= THRESHOLDS["identity_cosine_margin_min"]
        and metrics["null_effect_fraction"] <= THRESHOLDS["null_effect_fraction_max"]
        and metrics["block_remaining_fraction"] <= THRESHOLDS["block_remaining_fraction_max"]
    )


def run_model(
    architecture: str,
    replicate: int,
    config: ModelConfig,
    seed: int,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    key = model_key(architecture, replicate)
    model, training = train_model(config, seed, device)
    ids = {
        name: torch.tensor([row[f"{name}_ids"] for row in rows], device=device)
        for name in ("base", "target", "wrong", "null")
    }
    logits: dict[str, torch.Tensor] = {}
    traces: dict[str, dict[str, torch.Tensor]] = {}
    explicit_gaps: dict[str, float] = {}
    natural_accuracies: dict[str, float] = {}
    with torch.inference_mode():
        for name in ids:
            native = model(ids[name])
            logits[name], traces[name] = explicit_forward(model, ids[name], capture=True)
            explicit_gaps[name] = float(torch.max(torch.abs(native.float() - logits[name].float())).item())
            expected = torch.tensor([row["answers"][name] for row in rows], device=device)
            natural_accuracies[name] = float(
                (torch.argmax(logits[name][:, -1, CANDIDATE_SLICE], dim=-1) == expected).float().mean().item()
            )
    explicit_gap = max(explicit_gaps.values())
    explicit_accuracy = min(natural_accuracies.values())
    behavior_ok = (
        min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], explicit_accuracy)
        >= THRESHOLDS["behavior_accuracy_min"]
        and explicit_gap <= THRESHOLDS["native_explicit_logit_gap_max"]
    )
    if not behavior_ok:
        return {
            "model_key": key,
            "architecture": architecture,
            "replicate": replicate,
            "seed": seed,
            "training": training,
            "natural_accuracies": natural_accuracies,
            "explicit_gaps": explicit_gaps,
            "explicit_accuracy_min": explicit_accuracy,
            "native_explicit_logit_gap": explicit_gap,
            "behavior_qualified": False,
            "passed": False,
        }

    partitions = {
        name: torch.tensor([index for index, row in enumerate(rows) if row["partition"] == name], device=device)
        for name in WORLD_COUNTS
    }
    with torch.inference_mode():
        ranking = greedy_components(model, ids, logits, traces, partitions["discovery"])
        selection_rows: list[dict[str, Any]] = []
        for size in PREFIX_SIZES:
            coalition = ranking[:size]
            metrics = evaluate_coalition(model, ids, logits, traces, partitions["selection"], coalition)
            selection_rows.append({"size": size, "coalition": coalition, "metrics": metrics, "objective": selection_objective(metrics, size)})
        chosen = max(selection_rows, key=lambda row: (row["objective"], -row["size"]))
        coalition = chosen["coalition"]
        confirmation = evaluate_coalition(model, ids, logits, traces, partitions["confirmation"], coalition)
        loo_worsening: dict[str, float] = {}
        full_error = confirmation["correct"]["relative_error"]
        for component in coalition:
            reduced = [name for name in coalition if name != component]
            reduced_metrics = evaluate_coalition(model, ids, logits, traces, partitions["confirmation"], reduced)
            loo_worsening[component] = reduced_metrics["correct"]["relative_error"] - full_error
    role_counts = {role: sum(component_role(name) == role for name in coalition) for role in ("qk", "ov", "mlp")}
    layers = [int(name[1:3]) for name in coalition]
    return {
        "model_key": key,
        "architecture": architecture,
        "replicate": replicate,
        "seed": seed,
        "training": training,
        "natural_accuracies": natural_accuracies,
        "explicit_gaps": explicit_gaps,
        "explicit_accuracy_min": explicit_accuracy,
        "native_explicit_logit_gap": explicit_gap,
        "behavior_qualified": True,
        "greedy_ranking": ranking,
        "selection_candidates": selection_rows,
        "selected_size": len(coalition),
        "selected_components": coalition,
        "selected_role_counts": role_counts,
        "selected_relative_layers": [value / max(1, config.layers - 1) for value in layers],
        "confirmation": confirmation,
        "confirmation_loo_worsening": loo_worsening,
        "confirmation_essential_fraction": float(np.mean([value >= THRESHOLDS["loo_worsening_min"] for value in loo_worsening.values()])),
        "passed": passes(confirmation),
    }


def run(device_name: str) -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal completion marker already exists")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit has not passed")
    protocol, rows = verify_protocol()
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal run requires CUDA")
    device = torch.device("cuda")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            key = model_key(architecture, replicate)
            set_seed(MODEL_SEEDS[key])
            result = run_model(architecture, replicate, config, MODEL_SEEDS[key], rows, device)
            results.append(result)
            write_jsonl(MODELS, results)
            print(canonical_json({"completed": len(results), "total": len(ARCHITECTURES) * REPLICATES, "model": key, "passed": result["passed"]}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    write_jsonl(MODELS, results)
    raw = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "model_count": len(results),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "models_sha256": file_sha256(MODELS),
        "run_digest": digest(results),
        "pretrained_model_loaded": False,
    }
    atomic_json(RAW, raw)
    marker = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": "formal_run_complete",
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": raw["run_digest"],
        "raw_sha256": file_sha256(RAW),
        "models_sha256": file_sha256(MODELS),
    }
    marker["marker_digest"] = digest(marker)
    atomic_json(COMPLETE, marker)
    print(canonical_json({"status": "formal_run_complete", "elapsed_seconds": elapsed}))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [row for row in rows if row["behavior_qualified"]]
    passed = [row for row in rows if row["passed"]]
    per_depth = {
        architecture: {
            "qualified": sum(row["architecture"] == architecture and row["behavior_qualified"] for row in rows),
            "passed": sum(row["architecture"] == architecture and row["passed"] for row in rows),
        }
        for architecture in ARCHITECTURES
    }
    breadth = len(passed) >= THRESHOLDS["breadth_models_min"] and all(value["passed"] >= THRESHOLDS["breadth_per_depth_min"] for value in per_depth.values())
    gates = {
        "G-BEHAVIOR": len(qualified) == len(rows),
        "G-CORRECT-RESCUE": breadth and all(row["confirmation"]["correct"]["cosine"] >= THRESHOLDS["correct_cosine_min"] and row["confirmation"]["correct_accuracy"] >= THRESHOLDS["correct_accuracy_min"] for row in passed),
        "G-IDENTITY": breadth and all(row["confirmation"]["wrong"]["cosine"] <= THRESHOLDS["wrong_cosine_max"] and row["confirmation"]["identity_cosine_margin"] >= THRESHOLDS["identity_cosine_margin_min"] for row in passed),
        "G-NULL": breadth and all(row["confirmation"]["null_effect_fraction"] <= THRESHOLDS["null_effect_fraction_max"] for row in passed),
        "G-REVERSE-BLOCK": breadth and all(row["confirmation"]["block_remaining_fraction"] <= THRESHOLDS["block_remaining_fraction_max"] for row in passed),
        "G-BREADTH": breadth,
    }
    role_totals = {role: sum(row.get("selected_role_counts", {}).get(role, 0) for row in qualified) for role in ("qk", "ov", "mlp")}
    return {
        "models": len(rows),
        "qualified": len(qualified),
        "passed": len(passed),
        "per_depth": per_depth,
        "role_totals": role_totals,
        "gates": gates,
        "passed_all": all(gates.values()),
    }


def analyze() -> None:
    if not COMPLETE.exists():
        raise RuntimeError("formal run incomplete")
    protocol, _ = verify_protocol()
    raw = read_json(RAW)
    marker = read_json(COMPLETE)
    if raw["models_sha256"] != file_sha256(MODELS) or marker["raw_sha256"] != file_sha256(RAW):
        raise RuntimeError("artifact hash mismatch")
    rows = read_jsonl(MODELS)
    summary = summarize(rows)
    verdict = "free_transformer_typed_edge_coalition_confirmed" if summary["passed_all"] else "free_transformer_typed_edge_coalition_not_confirmed"
    analysis = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": raw["run_digest"],
        "verdict": verdict,
        "summary": summary,
        "authorization": {
            "fresh_qwen_single_model_edge_contract": summary["passed_all"],
            "glm4_or_ds7b_contract": False,
            "semantic_mechanism_claim": False,
            "unique_minimal_algorithm_claim": False,
            "new_mathematics": False,
        },
        "evidence_types": {"task_truth": True, "endpoint_truth": True, "internal_mechanism_truth": False},
    }
    analysis["analysis_digest"] = digest(analysis)
    atomic_json(ANALYSIS, analysis)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "summary": summary,
        "authorization": analysis["authorization"],
        "artifact_hashes": {
            "protocol": file_sha256(PROTOCOL),
            "material": file_sha256(MATERIAL),
            "environment": file_sha256(ENVIRONMENT),
            "preaudit": file_sha256(PREAUDIT),
            "raw": file_sha256(RAW),
            "models": file_sha256(MODELS),
            "complete": file_sha256(COMPLETE),
            "analysis": file_sha256(ANALYSIS),
        },
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"verdict": verdict, "summary": summary}))


def smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(layers=2, width=32, heads=4, mlp_width=64, max_length=23, vocab_size=22)
    set_seed(1254)
    model = TinyCausalTransformer(config).to(device).eval()
    rows = make_worlds()[:4]
    ids = torch.tensor([row["base_ids"] for row in rows], device=device)
    with torch.inference_mode():
        native = model(ids)
        explicit, traces = explicit_forward(model, ids, capture=True)
    atomic_json(ROOT / "tests/glm5_temp/phase1254_edge_smoke.json", {
        "max_gap": float(torch.max(torch.abs(native - explicit)).item()),
        "trace_count": len(traces),
        "expected_trace_count": len(component_ids(config)),
    })


def probe() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(layers=3, width=64, heads=4, mlp_width=128, max_length=23, vocab_size=22)
    rows = make_worlds(seed=1_254_300_001, counts={"discovery": 16, "selection": 16, "confirmation": 32})
    set_seed(1_254_301_001)
    result = run_model("development3", 0, config, 1_254_301_001, rows, device)
    atomic_json(ROOT / "tests/glm5_temp/phase1254_edge_probe.json", result)
    print(canonical_json({
        "behavior_qualified": result["behavior_qualified"],
        "selected_size": result.get("selected_size"),
        "confirmation": result.get("confirmation"),
        "passed": result["passed"],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "probe", "preregister", "run", "analyze"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "probe":
        probe()
    elif args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        run(args.device)
    else:
        analyze()


if __name__ == "__main__":
    main()
