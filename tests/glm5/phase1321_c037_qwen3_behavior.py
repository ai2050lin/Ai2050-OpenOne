#!/usr/bin/env python3
"""Phase1321: one-shot behavior qualification for frozen C037 material."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
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

PHASE, CAMPAIGN = 1321, "C037"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1321_c037_qwen3_behavior_audit.py"
PARENT = T / "result/phase1320_c037_event_isomorphism_boundary_contract"
MATERIAL = PARENT / "material/frozen_isomorphic_lookup_pairs.jsonl"
OUT = T / "result/phase1321_c037_qwen3_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
GEN_MANIFEST = OUT / "protocol/frozen_generation_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
RAW = OUT / "raw/candidate_scores.jsonl"
GENERATIONS = OUT / "raw/list_free_generations.jsonl"
SUMMARY = OUT / "analysis/behavior_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("department", "region", "mode", "level", "channel", "status")
SURFACES = ("registry_narrative", "registry_table")
SCORE_BATCH, GEN_BATCH, MAX_NEW = 32, 8, 8


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


def rate(values: Any) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def generation_manifest() -> list[dict[str, Any]]:
    result = []
    for pair in rows(MATERIAL):
        if pair["panel"] != "active" or pair["partition"] not in {"confirmation", "holdout"}:
            continue
        for binding_state, state in enumerate(pair["states"]):
            result.append({
                "case_id": state["case_id"], "pair_key": pair["pair_key"], "partition": pair["partition"],
                "profile_index": pair["profile_index"], "attribute": pair["attribute"],
                "surface": pair["surface"], "binding_state": binding_state, "ids": state["ids"],
                "candidates": pair["candidates"], "gold_value": state["gold_value"],
                "true_boundary": state["true_boundary"],
            })
    return result


def preregister(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "phase1321_qwen3_behavior_only":
        raise RuntimeError("Phase1320 did not authorize Phase1321")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("Phase1320 audit failed")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    generation = generation_manifest()
    write_rows(GEN_MANIFEST, generation)
    parent_protocol = load(PARENT / "protocol/preregistration.json")
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1321.c037.behavior.v1",
        "research_object": "behavior qualification at the compiled assistant boundary; no hidden-state read",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "formal_run_budget": 1,
        "runtime": {"candidate_batch": SCORE_BATCH, "generation_batch": GEN_BATCH,
                    "max_new_tokens": MAX_NEW, "exact_length_batches": True},
        "material": {"sha256": sha(MATERIAL), "pair_count": 576, "state_count": 1152,
                     "generation_manifest_sha256": sha(GEN_MANIFEST), "generation_count": len(generation)},
        "thresholds": parent_protocol["behavior"]["thresholds"],
        "candidate_scoring": "three frozen one-token continuations at each state's compiled true_boundary",
        "generation": "active confirmation/holdout only; greedy; exactly one frozen candidate-value hit",
        "hidden_states_read": False,
        "success_authorization": "phase1322_isomorphic_field_only",
        "failure_authorization": "close_c037_without_hidden",
        "hard_stops": ["No hidden states", "No material, split, model, parser, boundary, or threshold change",
                       "No second formal model run"],
        "dependencies": {
            "parent_protocol": sha(PARENT / "protocol/preregistration.json"),
            "parent_final": sha(PARENT / "analysis/final.json"),
            "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
            "material": sha(MATERIAL), "generation_manifest": sha(GEN_MANIFEST),
        },
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    save(PROTOCOL, {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol_digest": digest(timeless)})
    print(canonical({"states": 1152, "generation": len(generation)}))


def summarize(candidate: list[dict[str, Any]], generation: list[dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    partition = {key: rate(x["correct"] for x in candidate if x["partition"] == key) for key in PARTITIONS}
    attribute = {key: rate(x["correct"] for x in candidate if x["attribute"] == key) for key in ATTRS}
    surface = {key: rate(x["correct"] for x in candidate if x["surface"] == key) for key in SURFACES}
    active: dict[str, list[dict[str, Any]]] = defaultdict(list)
    generated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidate:
        if item["panel"] == "active":
            active[item["pair_key"]].append(item)
    for item in generation:
        generated[item["pair_key"]].append(item)
    metrics = {
        "finite_fraction": rate(x["finite"] for x in candidate),
        "candidate_accuracy": rate(x["correct"] for x in candidate),
        "partition_accuracy": partition, "attribute_accuracy": attribute, "surface_accuracy": surface,
        "active_pair_success": rate(len(v) == 2 and all(x["correct"] for x in v) for v in active.values()),
        "generation_coverage": rate(x["covered"] for x in generation),
        "generation_accuracy": rate(x["label_correct"] for x in generation),
        "generation_pair_success": rate(len(v) == 2 and all(x["label_correct"] for x in v) for v in generated.values()),
    }
    gates = {
        "finite": metrics["finite_fraction"] >= th["finite_fraction_min"],
        "candidate": metrics["candidate_accuracy"] >= th["candidate_accuracy_min"],
        "partition": min(partition.values()) >= th["partition_accuracy_min"],
        "attribute": min(attribute.values()) >= th["attribute_accuracy_min"],
        "surface": min(surface.values()) >= th["surface_accuracy_min"],
        "active_pair": metrics["active_pair_success"] >= th["active_pair_success_min"],
        "generation_coverage": metrics["generation_coverage"] >= th["generation_coverage_min"],
        "generation_accuracy": metrics["generation_accuracy"] >= th["generation_accuracy_min"],
        "generation_pair": metrics["generation_pair_success"] >= th["generation_pair_success_min"],
    }
    return {"metrics": metrics, "gates": gates, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol, pre = load(PROTOCOL), load(PRE)
    if not pre.get("all_checks_passed") or pre.get("authorization") != "run_phase1321_once":
        raise RuntimeError("independent preaudit did not authorize run")
    if any(path.exists() for path in (RAW, GENERATIONS, SUMMARY, FINAL, COMPLETE)):
        raise RuntimeError("formal run already consumed")
    pairs = rows(MATERIAL)
    states = [{"pair": pair, "binding_state": b, **state} for pair in pairs for b, state in enumerate(pair["states"])]
    gen_manifest = rows(GEN_MANIFEST)
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(qa)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for state in states:
            buckets[len(state["ids"])].append(state)
        candidate_out = []
        for length in sorted(buckets):
            for start in range(0, len(buckets[length]), SCORE_BATCH):
                batch = buckets[length][start:start + SCORE_BATCH]
                ids = torch.tensor([x["ids"] for x in batch], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": mask.cumsum(-1) - 1,
                          "use_cache": False, "return_dict": True}
                if supports:
                    kwargs["logits_to_keep"] = 1
                logits = model(**kwargs).logits[:, -1, :].float()
                for index, state in enumerate(batch):
                    scores = logits[index, torch.tensor(state["candidate_ids"], device=device)]
                    prediction = int(torch.argmax(scores).item())
                    pair = state["pair"]
                    candidate_out.append({
                        "case_id": state["case_id"], "pair_key": pair["pair_key"],
                        "partition": pair["partition"], "profile_index": pair["profile_index"],
                        "attribute": pair["attribute"], "surface": pair["surface"], "panel": pair["panel"],
                        "binding_state": state["binding_state"], "gold_position": state["gold_position"],
                        "prediction_position": prediction, "correct": prediction == state["gold_position"],
                        "finite": bool(torch.isfinite(scores).all().item()),
                        "candidate_logits": [float(value) for value in scores.cpu().tolist()],
                    })
        gen_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in gen_manifest:
            gen_buckets[len(item["ids"])].append(item)
        generation_out = []
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for length in sorted(gen_buckets):
            for start in range(0, len(gen_buckets[length]), GEN_BATCH):
                batch = gen_buckets[length][start:start + GEN_BATCH]
                ids = torch.tensor([x["ids"] for x in batch], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                outputs = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW, do_sample=False,
                                         use_cache=True, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
                texts = tokenizer.batch_decode(outputs[:, ids.shape[1]:], skip_special_tokens=True)
                for item, text_value in zip(batch, texts):
                    hits = [value for value in item["candidates"] if re.search(rf"\b{re.escape(value)}\b", text_value, re.I)]
                    prediction = hits[0] if len(hits) == 1 else None
                    generation_out.append({**{key: item[key] for key in (
                        "case_id", "pair_key", "partition", "profile_index", "attribute", "surface",
                        "binding_state", "gold_value")}, "generation": text_value, "candidate_hits": hits,
                        "covered": len(hits) == 1, "prediction": prediction,
                        "label_correct": prediction == item["gold_value"]})
        candidate_out.sort(key=lambda x: x["case_id"])
        generation_out.sort(key=lambda x: x["case_id"])
        write_rows(RAW, candidate_out)
        write_rows(GENERATIONS, generation_out)
        analysis = summarize(candidate_out, generation_out, protocol["thresholds"])
        authorization = "phase1322_isomorphic_field_only" if analysis["all_gates_passed"] else "close_c037_without_hidden"
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization,
                       "protocol_digest": protocol["protocol_digest"],
                       "raw_hashes": {"candidate": sha(RAW), "generation": sha(GENERATIONS)},
                       "counts": {"candidate": len(candidate_out), "generation": len(generation_out)},
                       "model_audit": qa, "placement": placement, "runtime_seconds": time.time() - started,
                       "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                       "hidden_states_read": False})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                     "verdict": "behavior_qualified" if analysis["all_gates_passed"] else "behavior_gate_failed",
                     "all_gates_passed": analysis["all_gates_passed"], "authorization": authorization,
                     "hidden_states_read": False, "protocol_digest": protocol["protocol_digest"]})
        save(COMPLETE, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "formal_runs_consumed": 1,
                        "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"candidate": analysis["metrics"]["candidate_accuracy"],
                         "generation": analysis["metrics"]["generation_accuracy"], "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()
