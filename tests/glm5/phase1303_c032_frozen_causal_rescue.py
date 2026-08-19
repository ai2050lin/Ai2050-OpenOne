#!/usr/bin/env python3
"""Phase1303: frozen event/depth causal substitution and rescue for C032."""
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

PHASE = 1303
CAMPAIGN = "C032"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1303_c032_frozen_causal_rescue_audit.py"
PARENT = T / "result/phase1302_c032_event_identity_path"
CONTRACT = T / "result/phase1299_c032_execution_compiler_contract"
OUT = T / "result/phase1303_c032_frozen_causal_rescue"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_causal_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/causal_arrays.npz"
META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/causal_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
CELLS = (
    ("user_answer_cue_end", 29),
    ("user_answer_cue_end", 30),
    ("assistant_answer_boundary", 25),
    ("assistant_answer_boundary", 26),
)
ARMS = ("neutral", "correct", "matched_null", "wrong_entity", "wrong_attribute", "self_state1")
BATCH = 4
EPS = 1e-12
TH = {
    "correct_donor_signed_gain_median_min": 0.5,
    "correct_over_wrong_donor_ratio_min": 1.25,
    "correct_over_matched_null_ratio_min": 1.25,
    "pairwise_correct_donor_win_fraction_min": 0.75,
    "confirmation_holdout_each_min": 0.70,
    "natural_behavior_retention_min": 0.99,
}


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


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
    source = rows(PARENT / "protocol/frozen_event_manifest.jsonl")
    lookup = {
        (x["partition"], x["profile_index"], x["attribute"], x["surface"], x["panel"]): x
        for x in source
    }
    result = []
    for partition in PARTITIONS:
        for profile in range(8):
            for attribute in ATTRS:
                for surface in ("catalog_prose", "inventory_ledger"):
                    active = lookup[(partition, profile, attribute, surface, "active")]
                    null = lookup[(partition, profile, attribute, surface, "matched_null")]
                    wrong_entity = lookup[(partition, (profile + 1) % 8, attribute, surface, "active")]
                    wrong_attribute = lookup[
                        (partition, profile, ATTRS[(ATTRS.index(attribute) + 1) % len(ATTRS)], surface, "active")
                    ]
                    result.append(
                        {
                            "case_key": f"{partition}|p{profile:02d}|{attribute}|{surface}",
                            "partition": partition,
                            "profile_index": profile,
                            "attribute": attribute,
                            "surface": surface,
                            "identity_token_ids": active["identity_token_ids"],
                            "identity_positions": active["identity_positions"],
                            "target_state0": active["states"][0],
                            "correct_state1": active["states"][1],
                            "matched_null_state1": null["states"][1],
                            "wrong_entity_state1": wrong_entity["states"][1],
                            "wrong_attribute_state1": wrong_attribute["states"][1],
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
    if parent_final.get("authorization") != "phase1303_frozen_causal_rescue" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1302 did not authorize Phase1303")
    if contract["causal"]["thresholds"] != TH:
        raise RuntimeError("causal threshold drift")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = build_manifest()
    write_rows(MANIFEST, manifest)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema_version": "phase1303.c032.frozen_causal_rescue.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "formal_run_budget": 1,
        "runtime": {"compiler": "right_padding", "global_fixed_length": True, "batch": BATCH},
        "object": "active-state0 residual substitution toward active-state1 identity",
        "partitions": list(PARTITIONS),
        "cells": [{"event": e, "depth": d} for e, d in CELLS],
        "arms": list(ARMS),
        "donor_rules": {
            "correct": "same partition/profile/attribute/surface active-state1",
            "matched_null": "same partition/profile/attribute/surface matched-null-state1",
            "wrong_entity": "next profile modulo 8, same partition/attribute/surface, active-state1",
            "wrong_attribute": "next attribute cyclically, same partition/profile/surface, active-state1; zero-model audit requires its gold identity to equal the target's state0 identity",
            "self_state1": "same active-state1 copied into itself; instrument retention only",
        },
        "manifest": {
            "sha256": sha(MANIFEST),
            "case_count": len(manifest),
            "partition_counts": {p: sum(x["partition"] == p for x in manifest) for p in PARTITIONS},
        },
        "intervention": "replace only the frozen event token residual at the input of the frozen transformer layer",
        "readout": "target identity1-minus-identity0 candidate margin after intervention",
        "gain": "patched target margin minus neutral target margin",
        "aggregation": {
            "global": "pool both partitions and all four frozen cells for gain, ratios, and pairwise wins",
            "partition": "correct-donor target identity accuracy is separately required in confirmation and holdout",
            "retention": "pooled neutral-state0 correctness and exact self-state1 patch correctness",
            "wrong_ratio_denominator": "maximum absolute pooled median gain of wrong-entity and wrong-attribute",
            "null_ratio_denominator": "absolute pooled median matched-null gain",
        },
        "thresholds": TH,
        "success": "all finite, global gain/ratio/win gates, each partition identity gate, and retention gate",
        "success_authorization": "close_c032_mechanism_stage_with_causal_sufficiency_candidate",
        "failure_authorization": "close_c032_with_descriptive_path_only",
        "hard_stops": [
            "No discovery partition",
            "No new event/depth/component scan",
            "No donor reselection",
            "No threshold modification",
            "No second formal model run",
        ],
        "engineering_correction": {
            "status": "pre-model contract compilation correction",
            "invalid_attempt_archive": "phase1303_c032_frozen_causal_rescue_invalid_preregistration_20260815",
            "reason": "the first draft's next-attribute state0 donor did not uniformly preserve the target state0 identity; no model weights were loaded and formal run budget remained unconsumed",
            "repair": "use next-attribute state1, which is checked case-by-case to preserve the target state0 identity",
        },
        "dependencies": {
            "parent_protocol": sha(PARENT / "protocol/preregistration.json"),
            "parent_manifest": sha(PARENT / "protocol/frozen_event_manifest.jsonl"),
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
    print(canonical({"cases": len(manifest), "digest": protocol["protocol_digest"]}))


def make_batch(states: list[dict[str, Any]], raw_max: int, pad: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def summarize(gains: np.ndarray, answers: np.ndarray, retention: np.ndarray, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    # gains: [cases, cells, five causal arms neutral/correct/null/wrong-entity/wrong-attribute]
    correct = gains[:, :, 1].reshape(-1)
    null = gains[:, :, 2].reshape(-1)
    wrong_entity = gains[:, :, 3].reshape(-1)
    wrong_attribute = gains[:, :, 4].reshape(-1)
    correct_median = float(np.median(correct))
    wrong_scale = max(abs(float(np.median(wrong_entity))), abs(float(np.median(wrong_attribute))), EPS)
    null_scale = max(abs(float(np.median(null))), EPS)
    pairwise = float(np.mean(correct > np.maximum.reduce([null, wrong_entity, wrong_attribute])))
    part_accuracy = {}
    for partition in PARTITIONS:
        indices = [i for i, x in enumerate(metadata) if x["partition"] == partition]
        part_accuracy[partition] = float(np.mean(answers[indices, :, 1]))
    natural_retention = float(np.mean(retention))
    metrics = {
        "correct_donor_signed_gain_median": correct_median,
        "matched_null_signed_gain_median": float(np.median(null)),
        "wrong_entity_signed_gain_median": float(np.median(wrong_entity)),
        "wrong_attribute_signed_gain_median": float(np.median(wrong_attribute)),
        "correct_over_wrong_donor_ratio": correct_median / wrong_scale,
        "correct_over_matched_null_ratio": correct_median / null_scale,
        "pairwise_correct_donor_win_fraction": pairwise,
        "correct_identity_accuracy_by_partition": part_accuracy,
        "natural_behavior_retention": natural_retention,
    }
    gates = {
        "finite": bool(np.isfinite(gains).all()),
        "signed_gain": metrics["correct_donor_signed_gain_median"] >= TH["correct_donor_signed_gain_median_min"],
        "wrong_ratio": metrics["correct_over_wrong_donor_ratio"] >= TH["correct_over_wrong_donor_ratio_min"],
        "null_ratio": metrics["correct_over_matched_null_ratio"] >= TH["correct_over_matched_null_ratio_min"],
        "pairwise_win": metrics["pairwise_correct_donor_win_fraction"] >= TH["pairwise_correct_donor_win_fraction_min"],
        "confirmation_identity": part_accuracy["confirmation"] >= TH["confirmation_holdout_each_min"],
        "holdout_identity": part_accuracy["holdout"] >= TH["confirmation_holdout_each_min"],
        "natural_retention": natural_retention >= TH["natural_behavior_retention_min"],
    }
    cells = {}
    for ci, (event, depth) in enumerate(CELLS):
        cells[f"{event}@{depth}"] = {
            "correct_gain_median": float(np.median(gains[:, ci, 1])),
            "matched_null_gain_median": float(np.median(gains[:, ci, 2])),
            "wrong_entity_gain_median": float(np.median(gains[:, ci, 3])),
            "wrong_attribute_gain_median": float(np.median(gains[:, ci, 4])),
            "correct_identity_accuracy": float(np.mean(answers[:, ci, 1])),
            "correct_pairwise_win_fraction": float(
                np.mean(gains[:, ci, 1] > np.max(gains[:, ci, 2:5], axis=-1))
            ),
        }
    return {"metrics": metrics, "gates": gates, "cells": cells, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol = load(PROTOCOL)
    pre = load(PRE)
    if pre.get("authorization") != "run_phase1303_once" or not pre.get("all_checks_passed"):
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
        raw_max = max(
            len(x[key]["ids"])
            for x in manifest
            for key in (
                "target_state0",
                "correct_state1",
                "matched_null_state1",
                "wrong_entity_state1",
                "wrong_attribute_state1",
            )
        )
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        gains = np.empty((len(manifest), len(CELLS), 5), dtype=np.float32)
        answers = np.empty((len(manifest), len(CELLS), 5), dtype=np.bool_)
        retention = np.empty((len(manifest), len(CELLS), 2), dtype=np.bool_)
        neutral_margins = np.empty((len(manifest), len(CELLS)), dtype=np.float32)

        for cell_i, (event, depth) in enumerate(CELLS):
            layer = model.model.layers[depth]
            for start in range(0, len(manifest), BATCH):
                group = manifest[start : start + BATCH]
                donor_states = []
                for x in group:
                    donor_states.extend(
                        [
                            x["correct_state1"],
                            x["matched_null_state1"],
                            x["wrong_entity_state1"],
                            x["wrong_attribute_state1"],
                        ]
                    )
                dids, dmask, dpos = make_batch(donor_states, raw_max, pad, device)
                dkw = {
                    "input_ids": dids,
                    "attention_mask": dmask,
                    "position_ids": dpos,
                    "use_cache": False,
                    "output_hidden_states": True,
                    "return_dict": True,
                }
                if supports:
                    dkw["logits_to_keep"] = 1
                donor_out = model(**dkw)
                donor_vectors = []
                for local, x in enumerate(group):
                    donor_vectors.append(
                        torch.stack(
                            [
                                donor_out.hidden_states[depth][4 * local + arm_i, donor_states[4 * local + arm_i]["positions"][event]]
                                for arm_i in range(4)
                            ]
                        )
                    )

                target_states = []
                patch_vectors = []
                patch_positions = []
                patch_rows = []
                for local, x in enumerate(group):
                    # Five target-state0 rows: neutral plus four causal donor arms.
                    base_row = len(target_states)
                    target_states.extend([x["target_state0"]] * 5)
                    for arm_i in range(4):
                        patch_rows.append(base_row + 1 + arm_i)
                        patch_positions.append(x["target_state0"]["positions"][event])
                        patch_vectors.append(donor_vectors[local][arm_i])
                    # Sixth row is an exact state1 self-patch used only for retention.
                    self_row = len(target_states)
                    target_states.append(x["correct_state1"])
                    patch_rows.append(self_row)
                    patch_positions.append(x["correct_state1"]["positions"][event])
                    patch_vectors.append(donor_vectors[local][0])

                tids, tmask, tpos = make_batch(target_states, raw_max, pad, device)
                rows_t = torch.tensor(patch_rows, dtype=torch.long, device=device)
                positions_t = torch.tensor(patch_positions, dtype=torch.long, device=device)
                vectors_t = torch.stack(patch_vectors)

                def patch_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
                    hidden = args[0].clone()
                    hidden[rows_t, positions_t] = vectors_t
                    return (hidden,) + args[1:]

                handle = layer.register_forward_pre_hook(patch_hook)
                try:
                    tkw = {
                        "input_ids": tids,
                        "attention_mask": tmask,
                        "position_ids": tpos,
                        "use_cache": False,
                        "output_hidden_states": True,
                        "return_dict": True,
                    }
                    if supports:
                        tkw["logits_to_keep"] = 1
                    target_out = model(**tkw)
                finally:
                    handle.remove()

                final_hidden = target_out.hidden_states[-1]
                for local, x in enumerate(group):
                    base_row = 6 * local
                    identity_positions = x["identity_positions"]
                    target_pos = x["target_state0"]["positions"]["assistant_answer_boundary"]
                    margins = []
                    correct_flags = []
                    for arm_i in range(5):
                        scores = candidate_scores(
                            model,
                            final_hidden[base_row + arm_i, target_pos],
                            x["target_state0"]["candidate_ids"],
                        )
                        margin = scores[identity_positions[1]] - scores[identity_positions[0]]
                        margins.append(float(margin.item()))
                        correct_flags.append(int(torch.argmax(scores).item()) == identity_positions[1])
                    neutral = margins[0]
                    gains[start + local, cell_i] = np.asarray([0.0] + [m - neutral for m in margins[1:]], dtype=np.float32)
                    answers[start + local, cell_i] = np.asarray(correct_flags, dtype=np.bool_)
                    neutral_margins[start + local, cell_i] = neutral
                    neutral_scores = candidate_scores(
                        model,
                        final_hidden[base_row, target_pos],
                        x["target_state0"]["candidate_ids"],
                    )
                    self_pos = x["correct_state1"]["positions"]["assistant_answer_boundary"]
                    self_scores = candidate_scores(
                        model,
                        final_hidden[base_row + 5, self_pos],
                        x["correct_state1"]["candidate_ids"],
                    )
                    retention[start + local, cell_i, 0] = (
                        int(torch.argmax(neutral_scores).item()) == x["target_state0"]["gold_position"]
                    )
                    retention[start + local, cell_i, 1] = (
                        int(torch.argmax(self_scores).item()) == x["correct_state1"]["gold_position"]
                    )
                del donor_out, target_out
            print(canonical({"cell": f"{event}@{depth}", "completed_cases": len(manifest)}), flush=True)

        metadata = [
            {k: x[k] for k in ("case_key", "partition", "profile_index", "attribute", "surface")}
            for x in manifest
        ]
        analysis = summarize(gains, answers, retention, metadata)
        authorization = (
            "close_c032_mechanism_stage_with_causal_sufficiency_candidate"
            if analysis["all_gates_passed"]
            else "close_c032_with_descriptive_path_only"
        )
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ARRAYS,
            signed_gain=gains,
            target_identity_correct=answers,
            natural_retention=retention,
            neutral_margin=neutral_margins,
            arms=np.asarray(ARMS[:5]),
            cells=np.asarray([f"{e}@{d}" for e, d in CELLS]),
        )
        save(
            META,
            {
                "phase": PHASE,
                "campaign": CAMPAIGN,
                "protocol_digest": protocol["protocol_digest"],
                "array_sha256": sha(ARRAYS),
                "manifest_sha256": sha(MANIFEST),
                "model_audit": qa,
                "placement": placement,
                "runtime_seconds": time.time() - started,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                "case_metadata": metadata,
            },
        )
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization})
        save(
            FINAL,
            {
                "phase": PHASE,
                "campaign": CAMPAIGN,
                "verdict": (
                    "frozen_causal_rescue_qualified"
                    if analysis["all_gates_passed"]
                    else "frozen_causal_rescue_gate_failed"
                ),
                "all_gates_passed": analysis["all_gates_passed"],
                "authorization": authorization,
                "protocol_digest": protocol["protocol_digest"],
                "array_sha256": sha(ARRAYS),
                "c032_closed": True,
            },
        )
        save(
            COMPLETE,
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "formal_runs_consumed": 1,
                "protocol_digest": protocol["protocol_digest"],
            },
        )
        print(canonical({"metrics": analysis["metrics"], "gates": analysis["gates"], "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()
