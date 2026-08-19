#!/usr/bin/env python3
"""Phase1310: Qwen3 behavior qualification for the C034 typed-response material."""
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
from model_utils import MODEL_CONFIGS  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PHASE = 1310
CAMPAIGN = "C034"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1310_c034_qwen3_typed_behavior_audit.py"
PARENT = T / "result/phase1309_c034_typed_response_camera_contract"
MATERIAL = PARENT / "material/frozen_typed_response_pairs.jsonl"
SOURCE = T / "result/phase1304_c033_role_typed_causal_graph_contract/material/frozen_role_typed_lookup_cases.jsonl"
OUT = T / "result/phase1310_c034_qwen3_typed_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
GEN_MANIFEST = OUT / "protocol/frozen_generation_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
RAW = OUT / "raw/candidate_scores.jsonl"
GENERATIONS = OUT / "raw/list_free_generations.jsonl"
SUMMARY = OUT / "analysis/behavior_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

SYSTEM = "Use only the supplied catalog. Reply exactly as requested and do not explain."
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")
SCORE_BATCH = 32
GEN_BATCH = 8
MAX_NEW = 8


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


def render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def build_generation_manifest() -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                              local_files_only=True, use_fast=True)
    source = {r["case_id"]: r for r in rows(SOURCE)}
    selected = []
    for pair in rows(MATERIAL):
        if pair["panel"] != "active" or pair["partition"] not in {"confirmation", "holdout"}:
            continue
        for state in pair["states"]:
            row = source[state["case_id"]]
            text = render(tokenizer, row["generation_prompt"])
            selected.append({
                "case_id": state["case_id"], "pair_key": pair["pair_key"], "partition": pair["partition"],
                "profile_index": pair["profile_index"], "attribute": pair["attribute"], "surface": pair["surface"],
                "binding_state": 0 if state is pair["states"][0] else 1,
                "ids": tokenizer.encode(text, add_special_tokens=False), "candidates": pair["candidates"],
                "gold_candidate": pair["candidates"][state["gold_position"]],
            })
    return selected


def preregister(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "phase1310_qwen3_behavior_only" or not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("Phase1309 did not authorize Phase1310")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    generation = build_generation_manifest()
    write_rows(GEN_MANIFEST, generation)
    parent_protocol = load(PARENT / "protocol/preregistration.json")
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1310.c034.behavior.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "formal_run_budget": 1,
        "runtime": {"compiler": "right_padding", "candidate_batch": SCORE_BATCH, "generation_batch": GEN_BATCH},
        "material": {"sha256": sha(MATERIAL), "pair_count": 576, "state_count": 1152,
                     "generation_manifest_sha256": sha(GEN_MANIFEST), "generation_count": len(generation)},
        "thresholds": parent_protocol["behavior"]["thresholds"],
        "candidate_scoring": "three frozen one-token continuations from the answer boundary",
        "attribute_family": "all six active attribute questions must be correct within each partition/profile/surface/binding-state family",
        "generation": "greedy, max 8 tokens, exactly one frozen candidate-name hit",
        "hidden_states_read": False,
        "success_authorization": "phase1311_typed_trajectory_only",
        "failure_authorization": "close_c034_without_hidden",
        "hard_stops": ["No hidden states", "No parser or threshold change", "No second formal model run"],
        "dependencies": {"parent_protocol": sha(PARENT / "protocol/preregistration.json"),
                         "parent_final": sha(PARENT / "analysis/final.json"),
                         "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
                         "material": sha(MATERIAL), "generation_manifest": sha(GEN_MANIFEST)},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    protocol = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_digest": digest(timeless)}
    save(PROTOCOL, protocol)
    print(canonical({"states": 1152, "generation": len(generation), "digest": protocol["protocol_digest"]}))


def rate(values: Any) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def behavior_summary(candidate: list[dict[str, Any]], generation: list[dict[str, Any]], th: dict[str, float]):
    overall = rate(x["correct"] for x in candidate)
    partition = {p: rate(x["correct"] for x in candidate if x["partition"] == p) for p in PARTITIONS}
    attribute = {a: rate(x["correct"] for x in candidate if x["attribute"] == a) for a in ATTRS}
    surface = {s: rate(x["correct"] for x in candidate if x["surface"] == s) for s in SURFACES}
    active_groups = defaultdict(list)
    families = defaultdict(list)
    for x in candidate:
        if x["panel"] == "active":
            active_groups[x["pair_key"]].append(x)
            families[(x["partition"], x["profile_index"], x["surface"], x["binding_state"])].append(x)
    pair_success = rate(len(v) == 2 and all(x["correct"] for x in v) for v in active_groups.values())
    family_success = rate(len(v) == 6 and all(x["correct"] for x in v) for v in families.values())
    gen_groups = defaultdict(list)
    for x in generation:
        gen_groups[x["pair_key"]].append(x)
    gen_coverage = rate(x["covered"] for x in generation)
    gen_accuracy = rate(x["label_correct"] for x in generation)
    gen_pair = rate(len(v) == 2 and all(x["label_correct"] for x in v) for v in gen_groups.values())
    metrics = {"finite_fraction": rate(x["finite"] for x in candidate), "candidate_accuracy": overall,
               "partition_accuracy": partition, "attribute_accuracy": attribute, "surface_accuracy": surface,
               "active_pair_success": pair_success, "attribute_family_success": family_success,
               "generation_coverage": gen_coverage, "generation_label_accuracy": gen_accuracy,
               "generation_pair_success": gen_pair}
    gates = {
        "finite": metrics["finite_fraction"] >= th["finite_fraction_min"],
        "candidate": overall >= th["candidate_accuracy_min"],
        "partition": min(partition.values()) >= th["partition_accuracy_min"],
        "attribute": min(attribute.values()) >= th["attribute_accuracy_min"],
        "surface": min(surface.values()) >= th["surface_accuracy_min"],
        "active_pair": pair_success >= th["active_pair_success_min"],
        "attribute_family": family_success >= th["attribute_family_success_min"],
        "generation_coverage": gen_coverage >= th["generation_coverage_min"],
        "generation_accuracy": gen_accuracy >= th["generation_label_accuracy_min"],
        "generation_pair": gen_pair >= th["generation_pair_success_min"],
    }
    return {"metrics": metrics, "gates": gates, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol = load(PROTOCOL)
    pre = load(PRE)
    if pre.get("authorization") != "run_phase1310_once" or not pre.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not authorize the run")
    if any(path.exists() for path in (RAW, GENERATIONS, SUMMARY, FINAL, COMPLETE)):
        raise RuntimeError("formal run budget already consumed")
    pairs = rows(MATERIAL)
    states = [{"pair": pair, "binding_state": b, **state} for pair in pairs for b, state in enumerate(pair["states"])]
    generation_manifest = rows(GEN_MANIFEST)
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(qa)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        buckets = defaultdict(list)
        for i, state in enumerate(states):
            buckets[len(state["ids"])].append((i, state))
        candidate_out = []
        for length in sorted(buckets):
            bucket = buckets[length]
            for start in range(0, len(bucket), SCORE_BATCH):
                batch = bucket[start:start + SCORE_BATCH]
                ids = torch.tensor([x[1]["ids"] for x in batch], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                kw = {"input_ids": ids, "attention_mask": mask, "position_ids": mask.cumsum(-1) - 1,
                      "use_cache": False, "return_dict": True}
                if supports:
                    kw["logits_to_keep"] = 1
                logits = model(**kw).logits[:, -1, :].float()
                for row_i, (_, state) in enumerate(batch):
                    scores = logits[row_i, torch.tensor(state["candidate_ids"], device=device)]
                    prediction = int(torch.argmax(scores).item())
                    pair = state["pair"]
                    candidate_out.append({
                        "case_id": state["case_id"], "pair_key": pair["pair_key"], "partition": pair["partition"],
                        "profile_index": pair["profile_index"], "attribute": pair["attribute"],
                        "surface": pair["surface"], "panel": pair["panel"], "binding_state": state["binding_state"],
                        "gold_position": state["gold_position"], "prediction_position": prediction,
                        "correct": prediction == state["gold_position"],
                        "finite": bool(torch.isfinite(scores).all().item()),
                        "candidate_logits": [float(x) for x in scores.cpu().tolist()],
                    })
        gen_buckets = defaultdict(list)
        for i, item in enumerate(generation_manifest):
            gen_buckets[len(item["ids"])].append((i, item))
        generation_out = []
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for length in sorted(gen_buckets):
            bucket = gen_buckets[length]
            for start in range(0, len(bucket), GEN_BATCH):
                batch = bucket[start:start + GEN_BATCH]
                ids = torch.tensor([x[1]["ids"] for x in batch], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW, do_sample=False,
                                           use_cache=True, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)[:, ids.shape[1]:]
                texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for (_, item), text in zip(batch, texts):
                    hits = [name for name in item["candidates"] if re.search(rf"\b{re.escape(name)}\b", text, re.I)]
                    prediction = hits[0] if len(hits) == 1 else None
                    generation_out.append({**{k: item[k] for k in ("case_id", "pair_key", "partition", "profile_index", "attribute", "surface", "binding_state", "gold_candidate")},
                                           "generation": text, "candidate_hits": hits, "covered": len(hits) == 1,
                                           "prediction": prediction, "label_correct": prediction == item["gold_candidate"]})
        candidate_out.sort(key=lambda x: x["case_id"])
        generation_out.sort(key=lambda x: x["case_id"])
        write_rows(RAW, candidate_out)
        write_rows(GENERATIONS, generation_out)
        analysis = behavior_summary(candidate_out, generation_out, protocol["thresholds"])
        authorization = "phase1311_typed_trajectory_only" if analysis["all_gates_passed"] else "close_c034_without_hidden"
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
                         "generation": analysis["metrics"]["generation_label_accuracy"],
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
