#!/usr/bin/env python3
"""Phase1319: capture C036 token-embedding to all-layer/all-position response fields."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402

PHASE, CAMPAIGN = 1319, "C036"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1319_c036_embedding_full_state_field_audit.py"
PARENT = T / "result/phase1318_c036_qwen3_behavior"
CONTRACT = T / "result/phase1317_c036_embedding_field_contract"
MATERIAL = CONTRACT / "material/frozen_forward_lookup_pairs.jsonl"
OUT = T / "result/phase1319_c036_embedding_full_state_field"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_field_manifest.jsonl"
PROJECTION = OUT / "protocol/fixed_signed_projection.npz"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/full_state_field_arrays.npz"
META = OUT / "raw/field_metadata.json"
SUMMARY = OUT / "analysis/field_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("schedule", "depth", "flexibility", "jurisdiction", "format", "pace")
SURFACES = ("dossier_prose", "dossier_ledger")
PANELS = ("active", "matched_null", "self_repeat")
ROLES = ("query_entity", "query_attribute", "query_end", "answer_boundary", "record_entities", "record_queried_values")
ROLE_SLOTS = {"query_entity": 1, "query_attribute": 1, "query_end": 1, "answer_boundary": 1,
              "record_entities": 3, "record_queried_values": 3}
TOTAL_SLOTS = sum(ROLE_SLOTS.values())
SKETCH_DIM, D_MODEL, LAYER_COUNT, EXACT_DEPTH, PAIR_BATCH = 64, 2560, 37, 15, 8
EPS = 1e-10


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values: f.write(canonical(value) + "\n")


def cos(left: np.ndarray, right: np.ndarray) -> float:
    l, r = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
    denom = np.linalg.norm(l) * np.linalg.norm(r)
    return float(np.dot(l, r) / denom) if denom > EPS else 0.0


def gram(vectors: np.ndarray) -> np.ndarray:
    flat = vectors.reshape(vectors.shape[0], -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    unit = flat / np.where(norms > EPS, norms, 1.0)
    return unit @ unit.T


def build_manifest() -> list[dict[str, Any]]:
    return [pair for pair in rows(MATERIAL) if pair["panel"] in PANELS]


def preregister(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "phase1319_embedding_field_only":
        raise RuntimeError("Phase1318 did not authorize hidden-state field")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("Phase1318 audit failed")
    if OUT.exists() and not force: raise RuntimeError(f"{OUT} already exists")
    if OUT.exists(): shutil.rmtree(OUT)
    manifest = build_manifest(); write_rows(MANIFEST, manifest)
    rng = np.random.default_rng(1319)
    signs = rng.choice(np.array([-1, 1], np.int8), size=(D_MODEL, SKETCH_DIM))
    PROJECTION.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROJECTION, signs=signs)
    contract = load(CONTRACT / "protocol/preregistration.json")
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1319.c036.full_state_field.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "formal_run_budget": 1,
        "manifest": {"sha256": sha(MANIFEST), "pair_count": len(manifest), "panels": list(PANELS)},
        "projection": {"sha256": sha(PROJECTION), "seed": 1319, "shape": [D_MODEL, SKETCH_DIM],
                       "formula": "DeltaH @ signs / sqrt(2560); fixed before model load; no fitted coordinates"},
        "capture": {"layers_including_embedding": LAYER_COUNT, "all_positions": True,
                    "all_position_values": "64 signed sums plus exact L2 norm",
                    "registered_roles": list(ROLES), "role_slot_order": ROLE_SLOTS,
                    "exact_full_residual_depth": EXACT_DEPTH},
        "orientation": "binding state1 minus state0; identical across surfaces for the same partition/profile/attribute",
        "decomposition": contract["field"]["decomposition"], "thresholds": contract["field"]["thresholds"],
        "gate_application": "global finiteness/replay/nonzero plus separate confirmation and holdout typed and embedding-downstream gates",
        "success_authorization": "phase1320_shared_typed_causal_only",
        "failure_authorization": "close_c036_at_descriptive_field_boundary",
        "hard_stops": ["No component or head read", "No layer/role/coordinate selection", "No threshold or metric change",
                       "No second formal model run", "Failure closes C036 without causal decomposition"],
        "claim_scope": "Token-substitution response-field repeatability and relative geometry; not a word essence, full physical circuit, or causal use claim.",
        "dependencies": {"parent_protocol": sha(PARENT / "protocol/preregistration.json"),
                         "parent_final": sha(PARENT / "analysis/final.json"), "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
                         "contract_protocol": sha(CONTRACT / "protocol/preregistration.json"), "material": sha(MATERIAL),
                         "manifest": sha(MANIFEST), "projection": sha(PROJECTION)},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    save(PROTOCOL, {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_digest": digest(timeless)})
    print(canonical({"pairs": len(manifest), "projection": list(signs.shape)}))


def candidate_scores(model: Any, hidden: torch.Tensor, candidate_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(candidate_ids, dtype=torch.long, device=hidden.device)
    return model.lm_head.weight[ids] @ model.model.norm(hidden)


def slot_positions(state: dict[str, Any]) -> list[int]:
    result = []
    for role in ROLES:
        positions = state["positions"][role]
        if len(positions) != ROLE_SLOTS[role]: raise RuntimeError((role, positions))
        result.extend(positions)
    return result


def analyze(role_sketch: np.ndarray, norms: np.ndarray, answers: np.ndarray,
            metadata: list[dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    index = {(m["partition"], m["profile_index"], m["surface"], m["attribute"], m["panel"]): i
             for i, m in enumerate(metadata)}
    active_idx = [i for i, m in enumerate(metadata) if m["panel"] == "active"]
    self_idx = [i for i, m in enumerate(metadata) if m["panel"] == "self_repeat"]
    null_idx = [i for i, m in enumerate(metadata) if m["panel"] == "matched_null"]
    cells: dict[str, dict[str, float]] = {}
    gates = {
        "finite": bool(np.isfinite(role_sketch).all() and np.isfinite(norms).all()),
        "behavior_replay": float(np.mean(answers)) >= th["behavior_replay_accuracy_min"],
        "active_nonzero": float(np.mean(np.sum(norms[active_idx] ** 2, axis=(1, 2)) > EPS)) >= th["active_nonzero_fraction_min"],
    }
    for partition in PARTITIONS:
        embedding_cos, own_cos, gaps, wins, gram_cosines, permuted_cosines = [], [], [], [], [], []
        for profile in range(4):
            surface_fields = {}
            for surface in SURFACES:
                vectors = np.stack([role_sketch[index[(partition, profile, surface, a, "active")]] for a in ATTRS]).astype(np.float32)
                surface_fields[surface] = vectors - vectors.mean(axis=0, keepdims=True)
            for ai, attribute in enumerate(ATTRS):
                left = surface_fields[SURFACES[0]][ai]
                right = surface_fields[SURFACES[1]][ai]
                own = cos(left, right)
                wrong = [cos(left, surface_fields[SURFACES[1]][bi]) for bi in range(len(ATTRS)) if bi != ai]
                own_cos.append(own); gaps.append(own - max(wrong)); wins.append(own > max(wrong))
                li = index[(partition, profile, SURFACES[0], attribute, "active")]
                ri = index[(partition, profile, SURFACES[1], attribute, "active")]
                embedding_cos.append(cos(role_sketch[li, 0, 7:10], role_sketch[ri, 0, 7:10]))
            for surface in SURFACES:
                centered = surface_fields[surface]
                eg = gram(centered[:, 0, 7:10])
                dg = gram(centered[:, 1:])
                upper = np.triu_indices(len(ATTRS), 1)
                gram_cosines.append(cos(eg[upper], dg[upper]))
                permuted = dg[np.roll(np.arange(len(ATTRS)), 1)][:, np.roll(np.arange(len(ATTRS)), 1)]
                permuted_cosines.append(cos(eg[upper], permuted[upper]))
        cell = {
            "surface_embedding_cosine_median": float(np.median(embedding_cos)),
            "typed_cross_surface_cosine_median": float(np.median(own_cos)),
            "typed_cross_surface_gap_median": float(np.median(gaps)),
            "typed_cross_surface_own_win_fraction": float(np.mean(wins)),
            "embedding_downstream_gram_cosine_median": float(np.median(gram_cosines)),
            "embedding_downstream_permuted_cosine_median": float(np.median(permuted_cosines)),
            "embedding_downstream_over_permuted_gap": float(np.median(np.array(gram_cosines) - np.array(permuted_cosines))),
        }
        cells[partition] = cell
        if partition in {"confirmation", "holdout"}:
            gates[f"{partition}_embedding_surface"] = cell["surface_embedding_cosine_median"] >= th["surface_embedding_cosine_median_min"]
            gates[f"{partition}_typed_cosine"] = cell["typed_cross_surface_cosine_median"] >= th["typed_cross_surface_cosine_median_min"]
            gates[f"{partition}_typed_gap"] = cell["typed_cross_surface_gap_median"] >= th["typed_cross_surface_gap_median_min"]
            gates[f"{partition}_typed_win"] = cell["typed_cross_surface_own_win_fraction"] >= th["typed_cross_surface_own_win_fraction_min"]
            gates[f"{partition}_gram"] = cell["embedding_downstream_gram_cosine_median"] >= th["embedding_downstream_gram_cosine_median_min"]
            gates[f"{partition}_gram_control"] = cell["embedding_downstream_over_permuted_gap"] >= th["embedding_downstream_over_permuted_gap_min"]
    metrics = {
        "finite_fraction": float(np.mean(np.isfinite(role_sketch))), "behavior_replay_accuracy": float(np.mean(answers)),
        "active_nonzero_fraction": float(np.mean(np.sum(norms[active_idx] ** 2, axis=(1, 2)) > EPS)),
        "active_total_energy_median": float(np.median(np.sum(norms[active_idx] ** 2, axis=(1, 2)))),
        "matched_null_total_energy_median": float(np.median(np.sum(norms[null_idx] ** 2, axis=(1, 2)))),
        "self_repeat_total_energy_max": float(np.max(np.sum(norms[self_idx] ** 2, axis=(1, 2)))),
    }
    return {"metrics": metrics, "partitions": cells, "gates": gates, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol, pre = load(PROTOCOL), load(PRE)
    if pre.get("authorization") != "run_phase1319_once" or not pre.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not authorize run")
    if any(x.exists() for x in (ARRAYS, META, SUMMARY, FINAL, COMPLETE)): raise RuntimeError("formal run consumed")
    manifest = rows(MANIFEST); max_len = max(len(state["ids"]) for pair in manifest for state in pair["states"])
    model = None; started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]: raise RuntimeError(qa)
        d_model = int(model.get_input_embeddings().weight.shape[1])
        if d_model != D_MODEL: raise RuntimeError((d_model, D_MODEL))
        signs_np = np.load(PROJECTION)["signs"]
        projection = torch.tensor(signs_np, dtype=torch.float16, device=device) / np.sqrt(D_MODEL)
        n = len(manifest)
        all_sketch = np.zeros((n, LAYER_COUNT, max_len, SKETCH_DIM), np.float16)
        all_norm = np.zeros((n, LAYER_COUNT, max_len), np.float32)
        role_sketch = np.zeros((n, LAYER_COUNT, TOTAL_SLOTS, SKETCH_DIM), np.float16)
        exact15 = np.zeros((n, TOTAL_SLOTS, D_MODEL), np.float16)
        embedding_values = np.zeros((n, 3, D_MODEL), np.float16)
        answers = np.zeros((n, 2), np.bool_)
        lengths = np.zeros(n, np.int16)
        metadata = [{k: pair[k] for k in ("pair_key", "partition", "profile_index", "attribute", "surface", "panel")} for pair in manifest]
        buckets: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for i, pair in enumerate(manifest): buckets[len(pair["states"][0]["ids"])].append((i, pair))
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        for length in sorted(buckets):
            bucket = buckets[length]
            for start in range(0, len(bucket), PAIR_BATCH):
                batch = bucket[start:start + PAIR_BATCH]
                flat_states = [state for _, pair in batch for state in pair["states"]]
                ids = torch.tensor([state["ids"] for state in flat_states], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                kw = {"input_ids": ids, "attention_mask": mask, "position_ids": mask.cumsum(-1) - 1,
                      "use_cache": False, "output_hidden_states": True, "return_dict": True}
                if supports: kw["logits_to_keep"] = 1
                out = model(**kw)
                if len(out.hidden_states) != LAYER_COUNT: raise RuntimeError(len(out.hidden_states))
                for local, (global_i, pair) in enumerate(batch):
                    row0, row1 = 2 * local, 2 * local + 1
                    lengths[global_i] = length
                    slots = slot_positions(pair["states"][0])
                    value_slots = pair["states"][0]["positions"]["record_queried_values"]
                    for layer_i, hidden in enumerate(out.hidden_states):
                        delta = hidden[row1, :length] - hidden[row0, :length]
                        sketch = delta @ projection
                        all_sketch[global_i, layer_i, :length] = sketch.cpu().numpy().astype(np.float16)
                        all_norm[global_i, layer_i, :length] = torch.linalg.vector_norm(delta.float(), dim=-1).cpu().numpy()
                        role_sketch[global_i, layer_i] = sketch[slots].cpu().numpy().astype(np.float16)
                        if layer_i == EXACT_DEPTH:
                            exact15[global_i] = delta[slots].cpu().numpy().astype(np.float16)
                        if layer_i == 0:
                            embedding_values[global_i] = delta[value_slots].cpu().numpy().astype(np.float16)
                    final_hidden = out.hidden_states[-1]
                    for state_i, row_i in enumerate((row0, row1)):
                        state = pair["states"][state_i]
                        position = state["positions"]["answer_boundary"][0]
                        scores = candidate_scores(model, final_hidden[row_i, position], state["candidate_ids"])
                        answers[global_i, state_i] = int(torch.argmax(scores).item()) == state["gold_position"]
                del out
        analysis = analyze(role_sketch.astype(np.float32), all_norm, answers, metadata, protocol["thresholds"])
        authorization = "phase1320_shared_typed_causal_only" if analysis["all_gates_passed"] else "close_c036_at_descriptive_field_boundary"
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARRAYS, all_position_sketch=all_sketch, all_position_norm=all_norm,
                            role_sketch=role_sketch, exact_layer15_role_delta=exact15,
                            embedding_record_value_delta=embedding_values, behavior_correct=answers, lengths=lengths)
        save(META, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"],
                    "arrays_sha256": sha(ARRAYS), "manifest_sha256": sha(MANIFEST), "projection_sha256": sha(PROJECTION),
                    "metadata": metadata, "role_slot_order": ROLE_SLOTS, "max_length": int(max_len),
                    "model_audit": qa, "placement": placement, "runtime_seconds": time.time() - started,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization,
                       "protocol_digest": protocol["protocol_digest"], "arrays_sha256": sha(ARRAYS)})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                     "verdict": "embedding_field_qualified" if analysis["all_gates_passed"] else "embedding_field_gate_failed",
                     "all_gates_passed": analysis["all_gates_passed"], "authorization": authorization,
                     "protocol_digest": protocol["protocol_digest"]})
        save(COMPLETE, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "formal_runs_consumed": 1,
                        "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"metrics": analysis["metrics"], "partitions": analysis["partitions"], "authorization": authorization}))
    finally:
        if model is not None: release_fp16(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("preregister", "run")); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); preregister(args.force) if args.command == "preregister" else run()
