#!/usr/bin/env python3
"""Phase1311: frozen upstream-type-separation versus late-convergence trajectory for C034."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402

PHASE = 1311
CAMPAIGN = "C034"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1311_c034_upstream_type_trajectory_audit.py"
PARENT = T / "result/phase1310_c034_qwen3_typed_behavior"
CONTRACT = T / "result/phase1309_c034_typed_response_camera_contract"
MATERIAL = CONTRACT / "material/frozen_typed_response_pairs.jsonl"
OUT = T / "result/phase1311_c034_upstream_type_trajectory"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_trajectory_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/trajectory_arrays.npz"
META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/trajectory_summary.json"
SELECTED = OUT / "analysis/selected_cell.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")
QUERY_DEPTHS = (8, 14, 20, 26, 32)
CELLS = tuple(("query_end", d) for d in QUERY_DEPTHS) + (("answer_boundary", 26),)
BATCH = 2
EPS = 1e-12


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(canonical(value) + "\n")


def identity_order(pair: dict[str, Any], identities: list[int]) -> list[dict[str, Any]]:
    ordered = []
    for identity in identities:
        matches = [state for state in pair["states"] if state["gold_position"] == identity]
        if len(matches) != 1:
            raise RuntimeError("identity-aligned pair is not unique")
        ordered.append(matches[0])
    return ordered


def build_manifest() -> list[dict[str, Any]]:
    source = rows(MATERIAL)
    lookup = {(x["partition"], x["profile_index"], x["attribute"], x["surface"], x["panel"]): x for x in source}
    result = []
    for partition in PARTITIONS:
        for profile in range(8):
            for attribute in ATTRS:
                wrong_attribute = ATTRS[(ATTRS.index(attribute) + 1) % len(ATTRS)]
                for anchor_surface in SURFACES:
                    opposite = SURFACES[1 - SURFACES.index(anchor_surface)]
                    target = lookup[(partition, profile, attribute, anchor_surface, "active")]
                    identities = target["identity_positions"]
                    same = lookup[(partition, profile, attribute, opposite, "active")]
                    wrong = lookup[(partition, profile, wrong_attribute, opposite, "active")]
                    null = lookup[(partition, profile, attribute, anchor_surface, "matched_null")]
                    result.append({
                        "case_key": f"{partition}|p{profile:02d}|{attribute}|{anchor_surface}",
                        "partition": partition, "profile_index": profile, "attribute": attribute,
                        "wrong_attribute": wrong_attribute, "anchor_surface": anchor_surface,
                        "opposite_surface": opposite, "identity_positions": identities,
                        "target_states": identity_order(target, identities),
                        "same_attribute_states": identity_order(same, identities),
                        "wrong_attribute_states": identity_order(wrong, identities),
                        "null_states": null["states"],
                    })
    return result


def preregister(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "phase1311_typed_trajectory_only" or not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("Phase1310 did not authorize Phase1311")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = build_manifest()
    write_rows(MANIFEST, manifest)
    contract = load(CONTRACT / "protocol/preregistration.json")
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1311.c034.typed_trajectory.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "formal_run_budget": 1,
        "runtime": {"compiler": "right_padding", "global_fixed_length": True, "case_batch": BATCH},
        "manifest": {"sha256": sha(MANIFEST), "case_count": len(manifest),
                     "partition_counts": {p: sum(x["partition"] == p for x in manifest) for p in PARTITIONS}},
        "cells": [{"role": role, "depth": depth} for role, depth in CELLS],
        "selection": contract["trajectory"]["candidate_selection"],
        "metrics": {"same_attribute": contract["trajectory"]["same_attribute"],
                    "wrong_attribute": contract["trajectory"]["wrong_attribute"],
                    "type_gap": contract["trajectory"]["type_gap"],
                    "active_to_null_norm_ratio": "median target active-delta norm divided by median matched-null delta norm"},
        "thresholds": contract["trajectory"]["thresholds"],
        "confirmation_rule": contract["trajectory"]["confirmation_rule"],
        "success_authorization": "phase1312_upstream_selective_rescue_only",
        "failure_authorization": "close_c034_without_causal",
        "hard_stops": ["No intervention", "No component selection", "No nonregistered depth or role",
                       "No post-unblinding threshold or pairing change", "No second formal model run"],
        "dependencies": {"parent_protocol": sha(PARENT / "protocol/preregistration.json"),
                         "parent_final": sha(PARENT / "analysis/final.json"),
                         "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
                         "contract": sha(CONTRACT / "protocol/preregistration.json"),
                         "material": sha(MATERIAL), "manifest": sha(MANIFEST)},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    protocol = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_digest": digest(timeless)}
    save(PROTOCOL, protocol)
    print(canonical({"cases": len(manifest), "cells": len(CELLS), "digest": protocol["protocol_digest"]}))


def make_batch(states: list[dict[str, Any]], raw_max: int, pad: int, device: torch.device):
    ids = torch.full((len(states), raw_max), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, state in enumerate(states):
        n = len(state["ids"])
        ids[i, :n] = torch.tensor(state["ids"], dtype=torch.long, device=device)
        mask[i, :n] = 1
    return ids, mask, mask.cumsum(-1) - 1


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float()[None], b.float()[None], dim=-1, eps=1e-8).item())


def cell_metrics(same: np.ndarray, wrong: np.ndarray, active_norm: np.ndarray, null_norm: np.ndarray,
                 metadata: list[dict[str, Any]], partition: str, cell_i: int):
    indices = [i for i, x in enumerate(metadata) if x["partition"] == partition]
    gaps = same[indices, cell_i] - wrong[indices, cell_i]
    return {
        "same_attribute_cross_surface_cosine_median": float(np.median(same[indices, cell_i])),
        "wrong_attribute_cosine_median": float(np.median(wrong[indices, cell_i])),
        "type_gap_median": float(np.median(gaps)),
        "type_gap_positive_fraction": float(np.mean(gaps > 0)),
        "active_to_null_norm_ratio": float(np.median(active_norm[indices, cell_i])) / max(float(np.median(null_norm[indices, cell_i])), EPS),
    }


def eligible(metric: dict[str, float], th: dict[str, float]) -> bool:
    return (
        metric["same_attribute_cross_surface_cosine_median"] >= th["same_attribute_cross_surface_cosine_median_min"]
        and metric["type_gap_median"] >= th["type_gap_median_min"]
        and metric["type_gap_positive_fraction"] >= th["type_gap_positive_fraction_min"]
        and metric["active_to_null_norm_ratio"] >= th["active_to_null_norm_ratio_min"]
    )


def analyze(same: np.ndarray, wrong: np.ndarray, active_norm: np.ndarray, null_norm: np.ndarray,
            behavior: np.ndarray, metadata: list[dict[str, Any]], th: dict[str, float]):
    metrics = {p: {f"{role}@{depth}": cell_metrics(same, wrong, active_norm, null_norm, metadata, p, ci)
                   for ci, (role, depth) in enumerate(CELLS)} for p in PARTITIONS}
    candidates = []
    for ci, depth in enumerate(QUERY_DEPTHS):
        metric = metrics["discovery"][f"query_end@{depth}"]
        if eligible(metric, th):
            candidates.append((metric["type_gap_median"], -depth, ci, depth))
    candidates.sort(reverse=True)
    selected = None
    gates = {"finite": bool(np.isfinite(same).all() and np.isfinite(wrong).all() and np.isfinite(active_norm).all() and np.isfinite(null_norm).all()),
             "behavior_replay": float(np.mean(behavior)) >= th["behavior_replay_accuracy_min"],
             "discovery_candidate_exists": bool(candidates)}
    if candidates:
        _, _, selected_i, selected_depth = candidates[0]
        selected = {"role": "query_end", "depth": selected_depth, "cell_index": selected_i,
                    "discovery_metric": metrics["discovery"][f"query_end@{selected_depth}"]}
        late_key = "answer_boundary@26"
        for partition in ("confirmation", "holdout"):
            selected_metric = metrics[partition][f"query_end@{selected_depth}"]
            late_metric = metrics[partition][late_key]
            gates[f"{partition}_selected_cell"] = eligible(selected_metric, th)
            gates[f"{partition}_upstream_over_late"] = (
                selected_metric["type_gap_median"] - late_metric["type_gap_median"]
                >= th["upstream_over_late_type_gap_min"]
            )
    else:
        for partition in ("confirmation", "holdout"):
            gates[f"{partition}_selected_cell"] = False
            gates[f"{partition}_upstream_over_late"] = False
    return {"partition_cell_metrics": metrics, "selected_cell": selected,
            "behavior_replay_accuracy": float(np.mean(behavior)), "gates": gates,
            "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol = load(PROTOCOL)
    pre = load(PRE)
    if pre.get("authorization") != "run_phase1311_once" or not pre.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not authorize the run")
    if any(path.exists() for path in (ARRAYS, META, SUMMARY, SELECTED, FINAL, COMPLETE)):
        raise RuntimeError("formal run budget already consumed")
    manifest = rows(MANIFEST)
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(qa)
        all_states = [state for x in manifest for key in ("target_states", "same_attribute_states", "wrong_attribute_states", "null_states") for state in x[key]]
        raw_max = max(len(state["ids"]) for state in all_states)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        shape = (len(manifest), len(CELLS))
        same_cos = np.empty(shape, np.float32)
        wrong_cos = np.empty(shape, np.float32)
        active_norm = np.empty(shape, np.float32)
        null_norm = np.empty(shape, np.float32)
        behavior = np.empty((len(manifest), 2), np.bool_)
        for start in range(0, len(manifest), BATCH):
            group = manifest[start:start + BATCH]
            states = [state for x in group for key in ("target_states", "same_attribute_states", "wrong_attribute_states", "null_states") for state in x[key]]
            ids, mask, positions = make_batch(states, raw_max, pad, device)
            kw = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                  "use_cache": False, "output_hidden_states": True, "return_dict": True}
            if supports:
                kw["logits_to_keep"] = 1
            out = model(**kw)
            for local, x in enumerate(group):
                offset = 8 * local
                for ci, (role, depth) in enumerate(CELLS):
                    vectors = [out.hidden_states[depth][offset + j, states[offset + j]["positions"][role]] for j in range(8)]
                    target_delta = vectors[1] - vectors[0]
                    same_delta = vectors[3] - vectors[2]
                    wrong_delta = vectors[5] - vectors[4]
                    null_delta = vectors[7] - vectors[6]
                    same_cos[start + local, ci] = cosine(target_delta, same_delta)
                    wrong_cos[start + local, ci] = cosine(target_delta, wrong_delta)
                    active_norm[start + local, ci] = float(torch.linalg.vector_norm(target_delta.float()).item())
                    null_norm[start + local, ci] = float(torch.linalg.vector_norm(null_delta.float()).item())
                final = out.hidden_states[-1]
                for b in (0, 1):
                    state = x["target_states"][b]
                    hidden = model.model.norm(final[offset + b, state["positions"]["answer_boundary"]])
                    candidate_ids = torch.tensor(state["candidate_ids"], dtype=torch.long, device=device)
                    scores = model.lm_head.weight[candidate_ids] @ hidden
                    behavior[start + local, b] = int(torch.argmax(scores).item()) == state["gold_position"]
            del out
        metadata = [{k: x[k] for k in ("case_key", "partition", "profile_index", "attribute", "wrong_attribute", "anchor_surface", "opposite_surface")} for x in manifest]
        analysis = analyze(same_cos, wrong_cos, active_norm, null_norm, behavior, metadata, protocol["thresholds"])
        authorization = "phase1312_upstream_selective_rescue_only" if analysis["all_gates_passed"] else "close_c034_without_causal"
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARRAYS, same_attribute_cosine=same_cos, wrong_attribute_cosine=wrong_cos,
                            active_delta_norm=active_norm, null_delta_norm=null_norm, behavior_correct=behavior)
        save(META, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"],
                    "array_sha256": sha(ARRAYS), "manifest_sha256": sha(MANIFEST), "case_metadata": metadata,
                    "model_audit": qa, "placement": placement, "runtime_seconds": time.time() - started,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization})
        save(SELECTED, {"phase": PHASE, "campaign": CAMPAIGN, "selected_cell": analysis["selected_cell"],
                        "authorized": analysis["all_gates_passed"], "protocol_digest": protocol["protocol_digest"]})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                     "verdict": "typed_trajectory_qualified" if analysis["all_gates_passed"] else "typed_trajectory_gate_failed",
                     "all_gates_passed": analysis["all_gates_passed"], "authorization": authorization,
                     "protocol_digest": protocol["protocol_digest"]})
        save(COMPLETE, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "formal_runs_consumed": 1,
                        "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"selected": analysis["selected_cell"], "gates": analysis["gates"], "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()
