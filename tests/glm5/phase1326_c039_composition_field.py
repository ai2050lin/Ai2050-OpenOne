#!/usr/bin/env python3
"""Phase1326: frozen role-aligned full-state composition response field."""
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

PHASE, CAMPAIGN = 1326, "C039"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1326_c039_composition_field_audit.py"
PARENT = T / "result/phase1325_c039_qwen3_behavior"
CONTRACT = T / "result/phase1324_c039_exact_truth_scope_contract"
MATERIAL = CONTRACT / "material/frozen_truth_scope_pairs.jsonl"
OUT = T / "result/phase1326_c039_composition_field"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_field_manifest.jsonl"
PROJECTION = OUT / "protocol/fixed_signed_projection.npz"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/full_state_composition_field.npz"
META = OUT / "raw/field_metadata.json"
SUMMARY = OUT / "analysis/field_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("discovery", "confirmation", "holdout")
SURFACES = ("prefix_scope", "reported_statement")
PANELS = ("active_single", "active_outer_context_true", "active_outer_context_false",
          "active_inner_context_true", "active_inner_context_false", "wrong_scope", "lexical_null", "self_repeat")
ACTIVE = PANELS[:5]
NESTED = PANELS[1:5]
ROLES = ("proposition_entity", "proposition_property", "active_operator", "context_operator",
         "query_entity", "query_property", "query_end", "assistant_boundary")
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
    l, r = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
    denominator = np.linalg.norm(l) * np.linalg.norm(r)
    return float(np.dot(l, r) / denominator) if denominator > EPS else 0.0


def build_manifest() -> list[dict[str, Any]]:
    return rows(MATERIAL)


def preregister(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "phase1326_c039_composition_field_only":
        raise RuntimeError("Phase1325 did not authorize Phase1326")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("Phase1325 audit failed")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = build_manifest()
    write_rows(MANIFEST, manifest)
    rng = np.random.default_rng(1326)
    signs = rng.choice(np.array([-1, 1], np.int8), size=(D_MODEL, SKETCH_DIM))
    PROJECTION.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROJECTION, signs=signs)
    contract = load(CONTRACT / "protocol/preregistration.json")
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1326.c039.composition_field.v1",
        "research_object": "role-aligned embedding-to-all-hidden-state truth-composition response field",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "formal_run_budget": 1,
        "manifest": {"sha256": sha(MANIFEST), "pair_count": len(manifest), "panels": list(PANELS)},
        "projection": {"sha256": sha(PROJECTION), "seed": 1326, "shape": [D_MODEL, SKETCH_DIM],
                       "formula": "DeltaH @ signs / sqrt(2560); no fitted coordinates"},
        "capture": {"layers_including_embedding": LAYER_COUNT, "all_positions": True,
                    "all_position_values": "64 signed sums plus exact L2 norm", "roles": list(ROLES),
                    "role_pool": "mean over each compiled token span; missing role is zero plus mask",
                    "exact_full_residual_depth": EXACT_DEPTH, "candidate_boundary": "compiled true_boundary"},
        "orientation": "state1(false active token) minus state0(true active token)",
        "analysis": {
            "cross_surface_panel": "within each partition/profile/property, center five active-panel fields per surface; own-panel cosine versus four wrong panels",
            "parity_transfer": "discovery outer-role parity prototypes classify confirmation/holdout inner-role and discovery inner-role prototypes classify outer-role; layers 1..36; no fitted alignment",
            "margin": "yes-minus-no margin at the compiled boundary; state1 minus state0",
            "gate_application": "global finite/replay/nonzero/self-repeat and separate confirmation/holdout descriptive gates",
        },
        "thresholds": contract["field"]["thresholds"],
        "success_authorization": "phase1327_c039_composition_causal_only",
        "failure_authorization": "close_c039_at_descriptive_composition_boundary",
        "hard_stops": ["No attention/MLP/head/probe read", "No fitted alignment or coordinate selection",
                       "No layer, role, metric, threshold, or prototype change", "No second formal model run",
                       "Failure closes C039 without causal decomposition"],
        "claim_scope": "Descriptive response-field repeatability under supplied linguistic roles; not a discovered scope tree or causal semantic operator.",
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


def pooled(hidden: torch.Tensor, positions: list[int]) -> torch.Tensor:
    if not positions:
        return torch.zeros(hidden.shape[-1], dtype=hidden.dtype, device=hidden.device)
    return hidden[torch.tensor(positions, dtype=torch.long, device=hidden.device)].mean(dim=0)


def analyze(role: np.ndarray, norms: np.ndarray, masks: np.ndarray, answers: np.ndarray, margins: np.ndarray,
            metadata: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    index = {(m["partition"], m["profile_index"], m["property"], m["surface"], m["panel"]): i
             for i, m in enumerate(metadata)}
    active = [i for i, m in enumerate(metadata) if m["panel"] in ACTIVE]
    repeated = [i for i, m in enumerate(metadata) if m["panel"] == "self_repeat"]
    gates = {
        "finite": bool(np.isfinite(role).all() and np.isfinite(norms).all() and np.isfinite(margins).all()),
        "behavior_replay": float(np.mean(answers)) >= thresholds["behavior_replay_accuracy_min"],
        "active_nonzero": float(np.mean(np.sum(norms[active] ** 2, axis=(1, 2)) > EPS))
                          >= thresholds["active_nonzero_fraction_min"],
        "self_repeat": float(np.max(np.sum(norms[repeated] ** 2, axis=(1, 2)))) <= thresholds["self_repeat_energy_max"],
    }

    discovery_vectors = {"outer": {0: [], 1: []}, "inner": {0: [], 1: []}}
    for i, item in enumerate(metadata):
        if item["partition"] == "discovery" and item["panel"] in NESTED:
            discovery_vectors[item["active_role"]][int(item["parity"])].append(role[i, 1:].ravel())
    prototypes = {kind: {parity: np.mean(values, axis=0) for parity, values in by_parity.items()}
                  for kind, by_parity in discovery_vectors.items()}

    cells: dict[str, dict[str, float]] = {}
    for partition in PARTITIONS:
        embedding_cosines: list[float] = []
        own_cosines: list[float] = []
        own_wins: list[bool] = []
        sign_correct: list[bool] = []
        active_abs: list[float] = []
        wrong_abs: list[float] = []
        lexical_abs: list[float] = []
        parity_correct: list[bool] = []
        parity_gaps: list[float] = []
        properties = sorted({m["property"] for m in metadata if m["partition"] == partition})
        for profile in range(4):
            for prop in properties:
                fields: dict[str, np.ndarray] = {}
                for surface in SURFACES:
                    values = np.stack([role[index[(partition, profile, prop, surface, panel)]] for panel in ACTIVE])
                    fields[surface] = values - values.mean(axis=0, keepdims=True)
                for panel_index, panel in enumerate(ACTIVE):
                    left, right = fields[SURFACES[0]][panel_index], fields[SURFACES[1]][panel_index]
                    own = cosine(left, right)
                    wrong = [cosine(left, fields[SURFACES[1]][j]) for j in range(len(ACTIVE)) if j != panel_index]
                    own_cosines.append(own)
                    own_wins.append(own > max(wrong))
                    li = index[(partition, profile, prop, SURFACES[0], panel)]
                    ri = index[(partition, profile, prop, SURFACES[1], panel)]
                    embedding_cosines.append(cosine(role[li, 0, 2], role[ri, 0, 2]))
        for i, item in enumerate(metadata):
            if item["partition"] != partition:
                continue
            delta = float(margins[i, 1] - margins[i, 0])
            if item["panel"] in ACTIVE:
                expected = 1.0 if int(item["parity"] or 0) == 1 else -1.0
                sign_correct.append(delta * expected > 0)
                active_abs.append(abs(delta))
            elif item["panel"] == "wrong_scope":
                wrong_abs.append(abs(delta))
            elif item["panel"] == "lexical_null":
                lexical_abs.append(abs(delta))
            if partition in {"confirmation", "holdout"} and item["panel"] in NESTED:
                target_kind = "outer" if item["active_role"] == "inner" else "inner"
                vector = role[i, 1:].ravel()
                scores = [cosine(vector, prototypes[target_kind][parity]) for parity in (0, 1)]
                truth = int(item["parity"])
                parity_correct.append(int(np.argmax(scores)) == truth)
                parity_gaps.append(scores[truth] - scores[1 - truth])
        cell = {
            "surface_operator_embedding_cosine_median": float(np.median(embedding_cosines)),
            "cross_surface_panel_cosine_median": float(np.median(own_cosines)),
            "cross_surface_panel_own_win_fraction": float(np.mean(own_wins)),
            "active_margin_sign_accuracy": float(np.mean(sign_correct)),
            "active_abs_margin_delta_median": float(np.median(active_abs)),
            "wrong_scope_abs_margin_delta_median": float(np.median(wrong_abs)),
            "lexical_null_abs_margin_delta_median": float(np.median(lexical_abs)),
            "cross_role_parity_accuracy": float(np.mean(parity_correct)) if parity_correct else 0.0,
            "cross_role_parity_gap_median": float(np.median(parity_gaps)) if parity_gaps else 0.0,
        }
        cells[partition] = cell
        if partition in {"confirmation", "holdout"}:
            gates[f"{partition}_operator_embedding"] = cell["surface_operator_embedding_cosine_median"] >= thresholds["surface_operator_embedding_cosine_median_min"]
            gates[f"{partition}_panel_cosine"] = cell["cross_surface_panel_cosine_median"] >= thresholds["cross_surface_panel_cosine_median_min"]
            gates[f"{partition}_panel_win"] = cell["cross_surface_panel_own_win_fraction"] >= thresholds["cross_surface_panel_own_win_fraction_min"]
            gates[f"{partition}_margin_sign"] = cell["active_margin_sign_accuracy"] >= thresholds["active_margin_sign_accuracy_min"]
            gates[f"{partition}_margin_size"] = cell["active_abs_margin_delta_median"] >= thresholds["active_abs_margin_delta_median_min"]
            gates[f"{partition}_wrong_scope"] = cell["wrong_scope_abs_margin_delta_median"] <= thresholds["wrong_scope_abs_margin_delta_median_max"]
            gates[f"{partition}_lexical_null"] = cell["lexical_null_abs_margin_delta_median"] <= thresholds["lexical_null_abs_margin_delta_median_max"]
            gates[f"{partition}_parity_accuracy"] = cell["cross_role_parity_accuracy"] >= thresholds["cross_role_parity_accuracy_min"]
            gates[f"{partition}_parity_gap"] = cell["cross_role_parity_gap_median"] >= thresholds["cross_role_parity_gap_median_min"]
    metrics = {
        "finite_fraction": float(np.mean(np.isfinite(role))), "behavior_replay_accuracy": float(np.mean(answers)),
        "active_nonzero_fraction": float(np.mean(np.sum(norms[active] ** 2, axis=(1, 2)) > EPS)),
        "active_total_energy_median": float(np.median(np.sum(norms[active] ** 2, axis=(1, 2)))),
        "self_repeat_total_energy_max": float(np.max(np.sum(norms[repeated] ** 2, axis=(1, 2)))),
        "role_presence_fraction": float(np.mean(masks)),
    }
    return {"metrics": metrics, "partitions": cells, "gates": gates, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol, pre = load(PROTOCOL), load(PRE)
    if not pre.get("all_checks_passed") or pre.get("authorization") != "run_phase1326_once":
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
        role_sketch = np.zeros((count, LAYER_COUNT, len(ROLES), SKETCH_DIM), np.float16)
        role_mask = np.zeros((count, 2, len(ROLES)), np.bool_)
        exact15 = np.zeros((count, len(ROLES), D_MODEL), np.float16)
        answers = np.zeros((count, 2), np.bool_)
        margins = np.zeros((count, 2), np.float32)
        lengths = np.zeros(count, np.int16)
        metadata = [{key: pair[key] for key in ("pair_key", "partition", "profile_index", "property",
                                                 "surface", "panel", "active_role", "parity", "context_truth")}
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
                    state0, state1 = pair["states"]
                    lengths[global_index] = length
                    for role_index, role_name in enumerate(ROLES):
                        role_mask[global_index, 0, role_index] = bool(state0["positions"].get(role_name, []))
                        role_mask[global_index, 1, role_index] = bool(state1["positions"].get(role_name, []))
                    for layer_index, hidden in enumerate(output.hidden_states):
                        delta = hidden[row1, :length] - hidden[row0, :length]
                        sketch = delta @ projection
                        all_sketch[global_index, layer_index, :length] = sketch.cpu().numpy().astype(np.float16)
                        all_norm[global_index, layer_index, :length] = torch.linalg.vector_norm(delta.float(), dim=-1).cpu().numpy()
                        for role_index, role_name in enumerate(ROLES):
                            value = pooled(hidden[row1], state1["positions"].get(role_name, [])) \
                                - pooled(hidden[row0], state0["positions"].get(role_name, []))
                            role_sketch[global_index, layer_index, role_index] = (value @ projection).cpu().numpy().astype(np.float16)
                            if layer_index == EXACT_DEPTH:
                                exact15[global_index, role_index] = value.cpu().numpy().astype(np.float16)
                    final_hidden = output.hidden_states[-1]
                    yes_index, no_index = pair["candidates"].index("yes"), pair["candidates"].index("no")
                    for state_index, row_index in enumerate((row0, row1)):
                        state = pair["states"][state_index]
                        scores = candidate_scores(model, final_hidden[row_index, state["true_boundary"]], state["candidate_ids"])
                        answers[global_index, state_index] = int(torch.argmax(scores).item()) == state["gold_position"]
                        margins[global_index, state_index] = float((scores[yes_index] - scores[no_index]).item())
                del output
        analysis = analyze(role_sketch.astype(np.float32), all_norm, role_mask, answers, margins,
                           metadata, protocol["thresholds"])
        authorization = "phase1327_c039_composition_causal_only" if analysis["all_gates_passed"] \
            else "close_c039_at_descriptive_composition_boundary"
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARRAYS, all_position_sketch=all_sketch, all_position_norm=all_norm,
                            role_sketch=role_sketch, role_mask=role_mask, exact_layer15_role_delta=exact15,
                            behavior_correct=answers, yes_no_margin=margins, lengths=lengths)
        save(META, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"],
                    "arrays_sha256": sha(ARRAYS), "manifest_sha256": sha(MANIFEST), "projection_sha256": sha(PROJECTION),
                    "metadata": metadata, "roles": list(ROLES), "max_length": int(max_len),
                    "model_audit": qa, "placement": placement, "runtime_seconds": time.time() - started,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization,
                       "protocol_digest": protocol["protocol_digest"], "arrays_sha256": sha(ARRAYS)})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                     "verdict": "composition_field_qualified" if analysis["all_gates_passed"] else "composition_field_gate_failed",
                     "all_gates_passed": analysis["all_gates_passed"], "authorization": authorization,
                     "protocol_digest": protocol["protocol_digest"]})
        save(COMPLETE, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "formal_runs_consumed": 1,
                        "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"metrics": analysis["metrics"], "partitions": analysis["partitions"],
                         "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()
