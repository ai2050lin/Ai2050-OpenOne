#!/usr/bin/env python3
"""Phase1307: frozen bidirectional answer-boundary residual substitution for C033."""
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

PHASE = 1307
CAMPAIGN = "C033"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1307_c033_bidirectional_answer_boundary_swap_audit.py"
PARENT = T / "result/phase1306_c033_frozen_answer_boundary_hidden"
CONTRACT = T / "result/phase1304_c033_role_typed_causal_graph_contract"
OUT = T / "result/phase1307_c033_bidirectional_answer_boundary_swap"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_swap_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/swap_arrays.npz"
META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/swap_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")
DIRECTIONS = ("state0_to_state1", "state1_to_state0")
ARMS = ("neutral", "correct", "matched_null", "wrong_entity", "wrong_attribute")
EVENT = "assistant_answer_boundary"
DEPTH = 26
BATCH = 4
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


def build_manifest() -> list[dict[str, Any]]:
    source = rows(PARENT / "protocol/frozen_hidden_manifest.jsonl")
    lookup = {
        (x["partition"], x["profile_index"], x["attribute"], x["surface"], x["panel"]): x
        for x in source
    }
    result = []
    for partition in PARTITIONS:
        for profile in range(8):
            for attribute in ATTRS:
                for surface in SURFACES:
                    active = lookup[(partition, profile, attribute, surface, "active")]
                    null = lookup[(partition, profile, attribute, surface, "matched_null")]
                    wrong_entity = lookup[(partition, (profile + 1) % 8, attribute, surface, "active")]
                    wrong_attribute = lookup[
                        (partition, profile, ATTRS[(ATTRS.index(attribute) + 1) % len(ATTRS)], surface, "active")
                    ]
                    directions = []
                    for source_state, destination_state, name in ((0, 1, DIRECTIONS[0]), (1, 0, DIRECTIONS[1])):
                        source_identity = active["identity_positions"][source_state]
                        wa_candidates = [
                            i for i, state in enumerate(wrong_attribute["states"])
                            if state["gold_position"] == source_identity
                        ]
                        if len(wa_candidates) != 1:
                            raise RuntimeError("wrong-attribute source-identity control is not unique")
                        directions.append(
                            {
                                "name": name,
                                "source_state": source_state,
                                "destination_state": destination_state,
                                "source_identity_position": source_identity,
                                "destination_identity_position": active["identity_positions"][destination_state],
                                "target": active["states"][source_state],
                                "destination": active["states"][destination_state],
                                "correct_donor": active["states"][destination_state],
                                "matched_null_donor": null["states"][destination_state],
                                "wrong_entity_donor": wrong_entity["states"][destination_state],
                                "wrong_attribute_donor": wrong_attribute["states"][wa_candidates[0]],
                            }
                        )
                    result.append(
                        {
                            "case_key": f"{partition}|p{profile:02d}|{attribute}|{surface}",
                            "partition": partition,
                            "profile_index": profile,
                            "attribute": attribute,
                            "surface": surface,
                            "identity_positions": active["identity_positions"],
                            "identity_token_ids": active["identity_token_ids"],
                            "directions": directions,
                            "donor_keys": {
                                "correct": active["group_id"],
                                "matched_null": null["group_id"],
                                "wrong_entity": wrong_entity["group_id"],
                                "wrong_attribute": wrong_attribute["group_id"],
                            },
                        }
                    )
    return result


def preregister(force: bool) -> None:
    parent_final = load(PARENT / "analysis/final.json")
    parent_audit = load(PARENT / "audit/independent_final_audit.json")
    contract = load(CONTRACT / "protocol/preregistration.json")
    if parent_final.get("authorization") != "phase1307_bidirectional_swap_only" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1306 did not authorize Phase1307")
    frozen = contract["bidirectional_swap"]
    if frozen["event"] != EVENT or frozen["depth"] != DEPTH:
        raise RuntimeError("event/depth drift")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = build_manifest()
    write_rows(MANIFEST, manifest)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema_version": "phase1307.c033.bidirectional_swap.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "formal_run_budget": 1,
        "runtime": {"compiler": "right_padding", "global_fixed_length": True, "batch": BATCH},
        "object": "bidirectional active-state residual substitution at the frozen answer boundary",
        "partitions": list(PARTITIONS),
        "event": EVENT,
        "depth": DEPTH,
        "directions": list(DIRECTIONS),
        "arms": list(ARMS) + ["self_patch"],
        "manifest": {
            "sha256": sha(MANIFEST),
            "case_count": len(manifest),
            "direction_count": 2 * len(manifest),
            "partition_counts": {p: sum(x["partition"] == p for x in manifest) for p in PARTITIONS},
        },
        "donor_rules": {
            "correct": "same active pair, destination binding state",
            "matched_null": "same base and destination binding index from the matched-null panel",
            "wrong_entity": "next profile modulo 8 with disjoint candidate IDs, destination binding index",
            "wrong_attribute": "next cyclic attribute and unique binding state whose gold is the source identity",
            "self_patch": "destination active state copied into itself; retention control only",
        },
        "intervention": "replace only the answer-boundary residual at the input of transformer layer 26",
        "readout": "destination-identity minus source-identity candidate margin after intervention",
        "signed_gain": "patched destination-minus-source margin minus neutral target margin",
        "aggregation": {
            "direction_partition": "correct-donor destination accuracy required separately for both directions and partitions",
            "global": "pool both directions and partitions for median gain, null ratio, and pairwise wins",
            "retention": "pool neutral source correctness and exact destination self-patch correctness",
        },
        "thresholds": frozen["thresholds"],
        "success_authorization": "phase1308_cross_surface_block_rescue_only",
        "failure_authorization": "close_c033_without_rescue",
        "claim_scope": frozen["claim_scope"],
        "hard_stops": [
            "No discovery partition",
            "No new event, depth, component, or donor search",
            "No one-direction fallback",
            "No threshold modification",
            "No second formal model run",
        ],
        "dependencies": {
            "parent_protocol": sha(PARENT / "protocol/preregistration.json"),
            "parent_manifest": sha(PARENT / "protocol/frozen_hidden_manifest.jsonl"),
            "parent_final": sha(PARENT / "analysis/final.json"),
            "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
            "contract": sha(CONTRACT / "protocol/preregistration.json"),
            "manifest": sha(MANIFEST),
        },
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
        "model_weights_loaded": False,
    }
    protocol = {
        **timeless,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": digest(timeless),
    }
    save(PROTOCOL, protocol)
    print(canonical({"cases": len(manifest), "directions": 2 * len(manifest), "digest": protocol["protocol_digest"]}))


def make_batch(states: list[dict[str, Any]], raw_max: int, pad: int, device: torch.device):
    ids = torch.full((len(states), raw_max), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, state in enumerate(states):
        n = len(state["ids"])
        ids[i, :n] = torch.tensor(state["ids"], dtype=torch.long, device=device)
        mask[i, :n] = 1
    return ids, mask, mask.cumsum(-1) - 1


def candidate_scores(model: Any, hidden: torch.Tensor, candidate_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(candidate_ids, dtype=torch.long, device=hidden.device)
    return model.lm_head.weight[ids] @ model.model.norm(hidden)


def summarize(gains: np.ndarray, answers: np.ndarray, retention: np.ndarray, metadata: list[dict[str, Any]], th: dict[str, float]):
    correct = gains[:, :, 1].reshape(-1)
    null = gains[:, :, 2].reshape(-1)
    wrong_entity = gains[:, :, 3].reshape(-1)
    wrong_attribute = gains[:, :, 4].reshape(-1)
    correct_median = float(np.median(correct))
    null_scale = max(abs(float(np.median(null))), EPS)
    direction_partition_accuracy = {}
    for di, direction in enumerate(DIRECTIONS):
        direction_partition_accuracy[direction] = {}
        for partition in PARTITIONS:
            indices = [i for i, x in enumerate(metadata) if x["partition"] == partition]
            direction_partition_accuracy[direction][partition] = float(np.mean(answers[indices, di, 1]))
    metrics = {
        "correct_signed_gain_median": correct_median,
        "matched_null_signed_gain_median": float(np.median(null)),
        "wrong_entity_signed_gain_median": float(np.median(wrong_entity)),
        "wrong_attribute_signed_gain_median": float(np.median(wrong_attribute)),
        "correct_over_matched_null_ratio": correct_median / null_scale,
        "pairwise_correct_win_fraction": float(np.mean(correct > np.maximum.reduce([null, wrong_entity, wrong_attribute]))),
        "direction_partition_accuracy": direction_partition_accuracy,
        "natural_retention": float(np.mean(retention)),
    }
    gates = {
        "finite": bool(np.isfinite(gains).all()),
        "signed_gain": metrics["correct_signed_gain_median"] >= th["signed_margin_gain_median_min"],
        "null_ratio": metrics["correct_over_matched_null_ratio"] >= th["correct_over_matched_null_ratio_min"],
        "pairwise_win": metrics["pairwise_correct_win_fraction"] >= th["pairwise_correct_win_fraction_min"],
        "natural_retention": metrics["natural_retention"] >= th["natural_retention_min"],
    }
    for direction in DIRECTIONS:
        for partition in PARTITIONS:
            gates[f"{direction}_{partition}"] = (
                direction_partition_accuracy[direction][partition] >= th["direction_partition_accuracy_min"]
            )
    return {"metrics": metrics, "gates": gates, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol = load(PROTOCOL)
    pre = load(PRE)
    if pre.get("authorization") != "run_phase1307_once" or not pre.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not authorize the run")
    if any(path.exists() for path in (ARRAYS, META, SUMMARY, FINAL, COMPLETE)):
        raise RuntimeError("formal run budget already consumed")
    manifest = rows(MANIFEST)
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(qa)
        all_states = [state for x in manifest for d in x["directions"] for state in (
            d["target"], d["destination"], d["correct_donor"], d["matched_null_donor"],
            d["wrong_entity_donor"], d["wrong_attribute_donor"]
        )]
        raw_max = max(len(state["ids"]) for state in all_states)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        gains = np.empty((len(manifest), 2, len(ARMS)), dtype=np.float32)
        answers = np.empty((len(manifest), 2, len(ARMS)), dtype=np.bool_)
        retention = np.empty((len(manifest), 2, 2), dtype=np.bool_)
        margins = np.empty((len(manifest), 2, len(ARMS)), dtype=np.float32)
        layer = model.model.layers[DEPTH]

        for direction_i in range(2):
            for start in range(0, len(manifest), BATCH):
                group = manifest[start:start + BATCH]
                donor_states = []
                for x in group:
                    d = x["directions"][direction_i]
                    donor_states.extend([
                        d["correct_donor"], d["matched_null_donor"], d["wrong_entity_donor"], d["wrong_attribute_donor"]
                    ])
                dids, dmask, dpos = make_batch(donor_states, raw_max, pad, device)
                dkw = {"input_ids": dids, "attention_mask": dmask, "position_ids": dpos,
                       "use_cache": False, "output_hidden_states": True, "return_dict": True}
                if supports:
                    dkw["logits_to_keep"] = 1
                donor_out = model(**dkw)
                donor_vectors = []
                for local in range(len(group)):
                    donor_vectors.append(torch.stack([
                        donor_out.hidden_states[DEPTH][4 * local + arm_i, donor_states[4 * local + arm_i]["position"]]
                        for arm_i in range(4)
                    ]))

                target_states = []
                patch_rows = []
                patch_positions = []
                patch_vectors = []
                for local, x in enumerate(group):
                    d = x["directions"][direction_i]
                    base = len(target_states)
                    target_states.extend([d["target"]] * 5)
                    for arm_i in range(4):
                        patch_rows.append(base + 1 + arm_i)
                        patch_positions.append(d["target"]["position"])
                        patch_vectors.append(donor_vectors[local][arm_i])
                    self_row = len(target_states)
                    target_states.append(d["destination"])
                    patch_rows.append(self_row)
                    patch_positions.append(d["destination"]["position"])
                    patch_vectors.append(donor_vectors[local][0])
                tids, tmask, tpos = make_batch(target_states, raw_max, pad, device)
                rows_t = torch.tensor(patch_rows, dtype=torch.long, device=device)
                positions_t = torch.tensor(patch_positions, dtype=torch.long, device=device)
                vectors_t = torch.stack(patch_vectors)

                def patch_hook(_module: Any, args: tuple[Any, ...]):
                    hidden = args[0].clone()
                    hidden[rows_t, positions_t] = vectors_t
                    return (hidden,) + args[1:]

                handle = layer.register_forward_pre_hook(patch_hook)
                try:
                    tkw = {"input_ids": tids, "attention_mask": tmask, "position_ids": tpos,
                           "use_cache": False, "output_hidden_states": True, "return_dict": True}
                    if supports:
                        tkw["logits_to_keep"] = 1
                    target_out = model(**tkw)
                finally:
                    handle.remove()
                final_hidden = target_out.hidden_states[-1]
                for local, x in enumerate(group):
                    d = x["directions"][direction_i]
                    base = 6 * local
                    neutral_margin = None
                    for arm_i in range(5):
                        scores = candidate_scores(model, final_hidden[base + arm_i, d["target"]["position"]], d["target"]["candidate_ids"])
                        destination = d["destination_identity_position"]
                        source = d["source_identity_position"]
                        margin = float((scores[destination] - scores[source]).float().item())
                        margins[start + local, direction_i, arm_i] = margin
                        if arm_i == 0:
                            neutral_margin = margin
                        gains[start + local, direction_i, arm_i] = margin - float(neutral_margin)
                        answers[start + local, direction_i, arm_i] = int(torch.argmax(scores).item()) == destination
                    neutral_scores = candidate_scores(model, final_hidden[base, d["target"]["position"]], d["target"]["candidate_ids"])
                    self_scores = candidate_scores(model, final_hidden[base + 5, d["destination"]["position"]], d["destination"]["candidate_ids"])
                    retention[start + local, direction_i, 0] = int(torch.argmax(neutral_scores).item()) == d["source_identity_position"]
                    retention[start + local, direction_i, 1] = int(torch.argmax(self_scores).item()) == d["destination_identity_position"]
                del donor_out, target_out

        metadata = [{k: x[k] for k in ("case_key", "partition", "profile_index", "attribute", "surface")} for x in manifest]
        analysis = summarize(gains, answers, retention, metadata, protocol["thresholds"])
        authorization = "phase1308_cross_surface_block_rescue_only" if analysis["all_gates_passed"] else "close_c033_without_rescue"
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARRAYS, signed_gain=gains, target_identity_correct=answers,
                            natural_retention=retention, destination_margin=margins)
        save(META, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"],
                    "array_sha256": sha(ARRAYS), "manifest_sha256": sha(MANIFEST), "model_audit": qa,
                    "placement": placement, "runtime_seconds": time.time() - started,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    "case_metadata": metadata})
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                     "verdict": "bidirectional_swap_qualified" if analysis["all_gates_passed"] else "bidirectional_swap_gate_failed",
                     "all_gates_passed": analysis["all_gates_passed"], "authorization": authorization,
                     "protocol_digest": protocol["protocol_digest"]})
        save(COMPLETE, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "formal_runs_consumed": 1,
                        "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"gates": analysis["gates"], "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()
