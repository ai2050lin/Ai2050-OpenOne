#!/usr/bin/env python3
"""Phase1322: identity-isomorphic embedding-to-full-hidden response field."""
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

PHASE, CAMPAIGN = 1322, "C037"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1322_c037_isomorphic_full_state_field_audit.py"
PARENT = T / "result/phase1321_c037_qwen3_behavior"
CONTRACT = T / "result/phase1320_c037_event_isomorphism_boundary_contract"
MATERIAL = CONTRACT / "material/frozen_isomorphic_lookup_pairs.jsonl"
OUT = T / "result/phase1322_c037_isomorphic_full_state_field"
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
ATTRS = ("department", "region", "mode", "level", "channel", "status")
SURFACES = ("registry_narrative", "registry_table")
PANELS = ("active", "matched_null", "self_repeat")
SLOT_KEYS = ("query_entity", "query_attribute", "query_end", "assistant_boundary",
             "record_entity_0", "record_entity_1", "record_entity_2",
             "record_value_0", "record_value_1", "record_value_2")
TOTAL_SLOTS = len(SLOT_KEYS)
SKETCH_DIM, D_MODEL, LAYER_COUNT, EXACT_DEPTH, PAIR_BATCH = 64, 2560, 37, 15, 8
EPS = 1e-10


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    l = left.astype(np.float64).ravel()
    r = right.astype(np.float64).ravel()
    denominator = np.linalg.norm(l) * np.linalg.norm(r)
    return float(np.dot(l, r) / denominator) if denominator > EPS else 0.0


def gram(vectors: np.ndarray) -> np.ndarray:
    flat = vectors.reshape(vectors.shape[0], -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    unit = flat / np.where(norms > EPS, norms, 1.0)
    return unit @ unit.T


def build_manifest() -> list[dict[str, Any]]:
    return [pair for pair in rows(MATERIAL) if pair["panel"] in PANELS]


def preregister(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "phase1322_isomorphic_field_only":
        raise RuntimeError("Phase1321 did not authorize Phase1322")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("Phase1321 audit failed")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = build_manifest()
    write_rows(MANIFEST, manifest)
    rng = np.random.default_rng(1322)
    signs = rng.choice(np.array([-1, 1], np.int8), size=(D_MODEL, SKETCH_DIM))
    PROJECTION.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROJECTION, signs=signs)
    contract = load(CONTRACT / "protocol/preregistration.json")
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1322.c037.isomorphic_field.v1",
        "research_object": "material-truth identity-isomorphic token-to-all-hidden-state response field",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "formal_run_budget": 1,
        "manifest": {"sha256": sha(MANIFEST), "pair_count": len(manifest), "panels": list(PANELS)},
        "projection": {"sha256": sha(PROJECTION), "seed": 1322, "shape": [D_MODEL, SKETCH_DIM],
                       "formula": "DeltaH @ signs / sqrt(2560); fixed before model load; no fitted coordinates"},
        "capture": {"layers_including_embedding": LAYER_COUNT, "all_positions": True,
                    "all_position_values": "64 signed sums plus exact L2 norm", "identity_slot_keys": list(SLOT_KEYS),
                    "slot_order_source": "state.slot_positions in canonical entity order", "exact_full_residual_depth": EXACT_DEPTH,
                    "candidate_boundary": "state.true_boundary == len(ids)-1"},
        "orientation": "binding state1 minus state0; canonical entity identities align cross-surface record slots",
        "decomposition": contract["field"]["decomposition"], "thresholds": contract["field"]["thresholds"],
        "gate_application": "global gates plus separate confirmation and holdout typed/Gram gates",
        "success_authorization": "phase1323_shared_typed_causal_only",
        "failure_authorization": "close_c037_at_isomorphic_field_boundary",
        "hard_stops": ["No component/head/attention/probe read", "No fitted alignment or coordinate selection",
                       "No layer/role/metric/threshold change", "No second formal model run",
                       "Failure closes C037 without causal decomposition"],
        "claim_scope": "Repeatable response geometry under supplied identity phi; not latent-role discovery or causal use.",
        "dependencies": {"parent_protocol": sha(PARENT / "protocol/preregistration.json"),
                         "parent_final": sha(PARENT / "analysis/final.json"),
                         "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
                         "contract_protocol": sha(CONTRACT / "protocol/preregistration.json"),
                         "material": sha(MATERIAL), "manifest": sha(MANIFEST), "projection": sha(PROJECTION)},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    save(PROTOCOL, {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol_digest": digest(timeless)})
    print(canonical({"pairs": len(manifest), "projection": list(signs.shape)}))


def candidate_scores(model: Any, hidden: torch.Tensor, candidate_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(candidate_ids, dtype=torch.long, device=hidden.device)
    return model.lm_head.weight[ids] @ model.model.norm(hidden)


def aligned_slots(state: dict[str, Any]) -> list[int]:
    positions = list(state["slot_positions"])
    if len(positions) != TOTAL_SLOTS or state["true_boundary"] != len(state["ids"]) - 1:
        raise RuntimeError("invalid frozen phi or boundary")
    if positions[3] != state["true_boundary"]:
        raise RuntimeError("assistant boundary slot mismatch")
    return positions


def analyze(role: np.ndarray, norms: np.ndarray, answers: np.ndarray,
            metadata: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    index = {(m["partition"], m["profile_index"], m["surface"], m["attribute"], m["panel"]): i
             for i, m in enumerate(metadata)}
    active = [i for i, item in enumerate(metadata) if item["panel"] == "active"]
    null = [i for i, item in enumerate(metadata) if item["panel"] == "matched_null"]
    repeated = [i for i, item in enumerate(metadata) if item["panel"] == "self_repeat"]
    gates = {
        "finite": bool(np.isfinite(role).all() and np.isfinite(norms).all()),
        "behavior_replay": float(np.mean(answers)) >= thresholds["behavior_replay_accuracy_min"],
        "active_nonzero": float(np.mean(np.sum(norms[active] ** 2, axis=(1, 2)) > EPS))
                          >= thresholds["active_nonzero_fraction_min"],
    }
    cells: dict[str, dict[str, float]] = {}
    for partition in PARTITIONS:
        embedding_cosines, own_cosines, gaps, wins, gram_cosines, permuted_cosines = [], [], [], [], [], []
        for profile in range(4):
            fields: dict[str, np.ndarray] = {}
            for surface in SURFACES:
                values = np.stack([role[index[(partition, profile, surface, attr, "active")]] for attr in ATTRS])
                fields[surface] = values - values.mean(axis=0, keepdims=True)
            for attr_index, attr in enumerate(ATTRS):
                left, right = fields[SURFACES[0]][attr_index], fields[SURFACES[1]][attr_index]
                own = cosine(left, right)
                wrong = [cosine(left, fields[SURFACES[1]][j]) for j in range(len(ATTRS)) if j != attr_index]
                own_cosines.append(own)
                gaps.append(own - max(wrong))
                wins.append(own > max(wrong))
                li = index[(partition, profile, SURFACES[0], attr, "active")]
                ri = index[(partition, profile, SURFACES[1], attr, "active")]
                embedding_cosines.append(cosine(role[li, 0, 7:10], role[ri, 0, 7:10]))
            for surface in SURFACES:
                centered = fields[surface]
                embedding_gram = gram(centered[:, 0, 7:10])
                downstream_gram = gram(centered[:, 1:])
                upper = np.triu_indices(len(ATTRS), 1)
                gram_cosines.append(cosine(embedding_gram[upper], downstream_gram[upper]))
                order = np.roll(np.arange(len(ATTRS)), 1)
                permuted = downstream_gram[order][:, order]
                permuted_cosines.append(cosine(embedding_gram[upper], permuted[upper]))
        cell = {
            "surface_embedding_cosine_median": float(np.median(embedding_cosines)),
            "typed_cross_surface_cosine_median": float(np.median(own_cosines)),
            "typed_cross_surface_gap_median": float(np.median(gaps)),
            "typed_cross_surface_own_win_fraction": float(np.mean(wins)),
            "embedding_downstream_gram_cosine_median": float(np.median(gram_cosines)),
            "embedding_downstream_permuted_cosine_median": float(np.median(permuted_cosines)),
            "embedding_downstream_over_permuted_gap": float(np.median(np.asarray(gram_cosines) - np.asarray(permuted_cosines))),
        }
        cells[partition] = cell
        if partition in {"confirmation", "holdout"}:
            gates[f"{partition}_embedding_surface"] = cell["surface_embedding_cosine_median"] >= thresholds["surface_embedding_cosine_median_min"]
            gates[f"{partition}_typed_cosine"] = cell["typed_cross_surface_cosine_median"] >= thresholds["typed_cross_surface_cosine_median_min"]
            gates[f"{partition}_typed_gap"] = cell["typed_cross_surface_gap_median"] >= thresholds["typed_cross_surface_gap_median_min"]
            gates[f"{partition}_typed_win"] = cell["typed_cross_surface_own_win_fraction"] >= thresholds["typed_cross_surface_own_win_fraction_min"]
            gates[f"{partition}_gram"] = cell["embedding_downstream_gram_cosine_median"] >= thresholds["embedding_downstream_gram_cosine_median_min"]
            gates[f"{partition}_gram_control"] = cell["embedding_downstream_over_permuted_gap"] >= thresholds["embedding_downstream_over_permuted_gap_min"]
    metrics = {
        "finite_fraction": float(np.mean(np.isfinite(role))), "behavior_replay_accuracy": float(np.mean(answers)),
        "active_nonzero_fraction": float(np.mean(np.sum(norms[active] ** 2, axis=(1, 2)) > EPS)),
        "active_total_energy_median": float(np.median(np.sum(norms[active] ** 2, axis=(1, 2)))),
        "matched_null_total_energy_median": float(np.median(np.sum(norms[null] ** 2, axis=(1, 2)))),
        "self_repeat_total_energy_max": float(np.max(np.sum(norms[repeated] ** 2, axis=(1, 2)))),
    }
    return {"metrics": metrics, "partitions": cells, "gates": gates, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol, pre = load(PROTOCOL), load(PRE)
    if not pre.get("all_checks_passed") or pre.get("authorization") != "run_phase1322_once":
        raise RuntimeError("independent preaudit did not authorize run")
    if any(path.exists() for path in (ARRAYS, META, SUMMARY, FINAL, COMPLETE)):
        raise RuntimeError("formal run consumed")
    manifest = rows(MANIFEST)
    max_len = max(len(state["ids"]) for pair in manifest for state in pair["states"])
    model = None
    started = time.time()
    try:
        model, _, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(qa)
        if int(model.get_input_embeddings().weight.shape[1]) != D_MODEL:
            raise RuntimeError("unexpected d_model")
        projection = torch.tensor(np.load(PROJECTION)["signs"], dtype=torch.float16, device=device) / np.sqrt(D_MODEL)
        count = len(manifest)
        all_sketch = np.zeros((count, LAYER_COUNT, max_len, SKETCH_DIM), np.float16)
        all_norm = np.zeros((count, LAYER_COUNT, max_len), np.float32)
        role_sketch = np.zeros((count, LAYER_COUNT, TOTAL_SLOTS, SKETCH_DIM), np.float16)
        exact15 = np.zeros((count, TOTAL_SLOTS, D_MODEL), np.float16)
        embedding_values = np.zeros((count, 3, D_MODEL), np.float16)
        answers = np.zeros((count, 2), np.bool_)
        lengths = np.zeros(count, np.int16)
        metadata = [{key: pair[key] for key in ("pair_key", "partition", "profile_index", "attribute", "surface", "panel")}
                    for pair in manifest]
        buckets: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, pair in enumerate(manifest):
            if len(pair["states"][0]["ids"]) != len(pair["states"][1]["ids"]):
                raise RuntimeError("pair length mismatch")
            buckets[len(pair["states"][0]["ids"])].append((index, pair))
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        for length in sorted(buckets):
            for start in range(0, len(buckets[length]), PAIR_BATCH):
                batch = buckets[length][start:start + PAIR_BATCH]
                flat = [state for _, pair in batch for state in pair["states"]]
                ids = torch.tensor([state["ids"] for state in flat], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": mask.cumsum(-1) - 1,
                          "use_cache": False, "output_hidden_states": True, "return_dict": True}
                if supports:
                    kwargs["logits_to_keep"] = 1
                output = model(**kwargs)
                if len(output.hidden_states) != LAYER_COUNT:
                    raise RuntimeError("unexpected hidden-state count")
                for local, (global_index, pair) in enumerate(batch):
                    row0, row1 = 2 * local, 2 * local + 1
                    lengths[global_index] = length
                    slots0, slots1 = aligned_slots(pair["states"][0]), aligned_slots(pair["states"][1])
                    if slots0 != slots1:
                        raise RuntimeError("within-pair slot drift")
                    for layer_index, hidden in enumerate(output.hidden_states):
                        delta = hidden[row1, :length] - hidden[row0, :length]
                        sketch = delta @ projection
                        all_sketch[global_index, layer_index, :length] = sketch.cpu().numpy().astype(np.float16)
                        all_norm[global_index, layer_index, :length] = torch.linalg.vector_norm(delta.float(), dim=-1).cpu().numpy()
                        role_sketch[global_index, layer_index] = sketch[slots0].cpu().numpy().astype(np.float16)
                        if layer_index == EXACT_DEPTH:
                            exact15[global_index] = delta[slots0].cpu().numpy().astype(np.float16)
                        if layer_index == 0:
                            embedding_values[global_index] = delta[slots0[7:10]].cpu().numpy().astype(np.float16)
                    final_hidden = output.hidden_states[-1]
                    for state_index, row_index in enumerate((row0, row1)):
                        state = pair["states"][state_index]
                        scores = candidate_scores(model, final_hidden[row_index, state["true_boundary"]], state["candidate_ids"])
                        answers[global_index, state_index] = int(torch.argmax(scores).item()) == state["gold_position"]
                del output
        analysis = analyze(role_sketch.astype(np.float32), all_norm, answers, metadata, protocol["thresholds"])
        authorization = "phase1323_shared_typed_causal_only" if analysis["all_gates_passed"] else "close_c037_at_isomorphic_field_boundary"
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARRAYS, all_position_sketch=all_sketch, all_position_norm=all_norm,
                            role_sketch=role_sketch, exact_layer15_role_delta=exact15,
                            embedding_record_value_delta=embedding_values, behavior_correct=answers, lengths=lengths)
        save(META, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"],
                    "arrays_sha256": sha(ARRAYS), "manifest_sha256": sha(MANIFEST), "projection_sha256": sha(PROJECTION),
                    "metadata": metadata, "slot_keys": list(SLOT_KEYS), "max_length": int(max_len),
                    "model_audit": qa, "placement": placement, "runtime_seconds": time.time() - started,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization,
                       "protocol_digest": protocol["protocol_digest"], "arrays_sha256": sha(ARRAYS)})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                     "verdict": "isomorphic_field_qualified" if analysis["all_gates_passed"] else "isomorphic_field_gate_failed",
                     "all_gates_passed": analysis["all_gates_passed"], "authorization": authorization,
                     "protocol_digest": protocol["protocol_digest"]})
        save(COMPLETE, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "formal_runs_consumed": 1,
                        "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"metrics": analysis["metrics"], "partitions": analysis["partitions"], "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()
