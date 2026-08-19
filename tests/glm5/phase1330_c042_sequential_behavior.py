#!/usr/bin/env python3
"""Phase1330: sequential FP16 behavior qualification for C042."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402

PHASE, CAMPAIGN = 1330, "C042"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1330_c042_sequential_behavior_audit.py"
PARENT = T / "result/phase1329_c042_relational_ecology_contract"
OUT = T / "result/phase1330_c042_sequential_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def parent_ok() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load(PARENT / "protocol/preregistration.json")
    audit = load(PARENT / "audit/independent_final_audit.json")
    if protocol["authorization"] != "run_phase1330_sequential_behavior" or not audit["all_checks_passed"]:
        raise RuntimeError("Phase1329 does not authorize behavior")
    return protocol, audit


def prepare(force: bool) -> None:
    protocol, _ = parent_ok()
    manifest = OUT / "protocol/execution_manifest.json"
    if manifest.exists() and not force:
        raise RuntimeError(f"{manifest} exists")
    if any((OUT / f"raw/{model}_scores.jsonl").exists() for model in MODELS):
        raise RuntimeError("cannot rewrite manifest after model output exists")
    frozen = {"phase": PHASE, "campaign": CAMPAIGN, "schema": "phase1330.c042.behavior_execution.v1",
              "parent_protocol_sha256": sha(PARENT / "protocol/preregistration.json"),
              "parent_contract_sha256": protocol["contract_sha256"], "model_order": list(MODELS),
              "score": "mean log probability of every token in each complete candidate sequence",
              "batch_size": 8, "precision": "fp16-no-quantization", "overwrite_after_run": False,
              "behavior_gate": protocol["behavior_gate"],
              "branch": protocol["branch"], "script_sha256": sha(SCRIPT), "auditor_sha256": sha(AUDITOR)}
    save(manifest, {**frozen, "manifest_sha256": hashlib.sha256(canonical(frozen).encode()).hexdigest(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(load(manifest), indent=2))


def score_sequences(model, device, jobs: list[dict[str, Any]], pad_id: int, batch_size: int) -> list[float]:
    output: list[float] = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        lengths = [len(job["sequence"]) for job in batch]
        width = max(lengths)
        input_ids = torch.full((len(batch), width), int(pad_id), dtype=torch.long, device=device)
        attention = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        for i, job in enumerate(batch):
            ids = torch.tensor(job["sequence"], dtype=torch.long, device=device)
            input_ids[i, :len(ids)] = ids
            attention[i, :len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        for i, job in enumerate(batch):
            prompt_len = job["prompt_len"]
            candidate = job["candidate"]
            values = [float(log_probs[i, prompt_len + j - 1, token].item())
                      for j, token in enumerate(candidate)]
            output.append(sum(values) / len(values))
        del input_ids, attention, logits, log_probs
    return output


def summarize(source: list[dict[str, Any]], scores: list[list[float]], thresholds: dict[str, Any]) -> dict[str, Any]:
    records = []
    for row, values in zip(source, scores):
        gold, wrong = row["gold_position"], 1 - row["gold_position"]
        margin = values[gold] - values[wrong]
        records.append({"case_id": row["case_id"], "partition": row["partition"], "surface": row["surface"],
                        "semantic_set": row["semantic_set"], "candidate_scores": values,
                        "gold_position": gold, "margin": margin, "correct": margin > 0})
    finite = sum(all(math.isfinite(v) for v in row["candidate_scores"]) for row in records) / len(records)
    accuracy = sum(row["correct"] for row in records) / len(records)
    partitions = {name: sum(row["correct"] for row in records if row["partition"] == name)
                        / sum(row["partition"] == name for row in records) for name in ("discovery", "confirmation", "holdout")}
    surfaces = {name: sum(row["correct"] for row in records if row["surface"] == name)
                      / sum(row["surface"] == name for row in records) for name in ("reference_family", "vocabulary_kind")}
    pair_groups = defaultdict(list)
    for row in records:
        pair_groups[(row["semantic_set"], row["surface"])].append(row["correct"])
    pair_success = sum(len(values) == 2 and all(values) for values in pair_groups.values()) / len(pair_groups)
    mean_margin = sum(row["margin"] for row in records) / len(records)
    gates = {"finite_fraction": finite >= thresholds["finite_fraction_min"],
             "candidate_accuracy": accuracy >= thresholds["candidate_accuracy_min"],
             "partition_accuracy": min(partitions.values()) >= thresholds["partition_accuracy_min"],
             "surface_accuracy": min(surfaces.values()) >= thresholds["surface_accuracy_min"],
             "order_pair_success": pair_success >= thresholds["order_pair_success_min"],
             "mean_correct_margin": mean_margin >= thresholds["mean_correct_margin_min"]}
    return {"records": records, "metrics": {"finite_fraction": finite, "candidate_accuracy": accuracy,
                                             "partition_accuracy": partitions, "surface_accuracy": surfaces,
                                             "order_pair_success": pair_success, "mean_correct_margin": mean_margin},
            "gates": gates, "qualified": all(gates.values())}


def run_model(model_name: str) -> None:
    if model_name not in MODELS:
        raise ValueError(model_name)
    protocol, _ = parent_ok()
    manifest = load(OUT / "protocol/execution_manifest.json")
    frozen = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "created_at_utc"}}
    if hashlib.sha256(canonical(frozen).encode()).hexdigest() != manifest["manifest_sha256"]:
        raise RuntimeError("execution manifest hash mismatch")
    result_path = OUT / f"analysis/{model_name}_summary.json"
    if result_path.exists():
        raise RuntimeError(f"formal result already exists: {result_path}")
    source = rows(PARENT / "material/frozen_behavior_cases.jsonl")
    compiled = rows(PARENT / f"compiled/{model_name}_behavior.jsonl")
    if len(source) != len(compiled) or any(a["case_id"] != b["case_id"] for a, b in zip(source, compiled)):
        raise RuntimeError("compiled/source mismatch")
    print(f"[Phase1330] loading {model_name}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        jobs = []
        for row in compiled:
            for candidate in row["candidate_ids"]:
                jobs.append({"sequence": row["prompt_ids"] + candidate, "prompt_len": len(row["prompt_ids"]),
                             "candidate": candidate})
        flat_scores = score_sequences(model, device, jobs, int(pad_id), manifest["batch_size"])
        scores = [flat_scores[i:i + 2] for i in range(0, len(flat_scores), 2)]
        summary = summarize(source, scores, protocol["behavior_gate"])
        write_rows(OUT / f"raw/{model_name}_scores.jsonl", summary.pop("records"))
        runtime = {"model": model_name, "device": str(device), "placement": placement,
                   "quantization_audit": quantization_audit(model),
                   "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                   "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        save(OUT / f"runtime/{model_name}.json", runtime)
        save(result_path, {"phase": PHASE, "campaign": CAMPAIGN, "model": model_name, **summary,
                           "source_sha256": sha(PARENT / "material/frozen_behavior_cases.jsonl"),
                           "compiled_sha256": sha(PARENT / f"compiled/{model_name}_behavior.jsonl"),
                           "runtime_sha256": sha(OUT / f"runtime/{model_name}.json"),
                           "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(load(result_path), indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        print(f"[Phase1330] released {model_name}", flush=True)


def finalize() -> None:
    protocol, _ = parent_ok()
    summaries = {model: load(OUT / f"analysis/{model}_summary.json") for model in MODELS}
    qualified = [model for model in MODELS if summaries[model]["qualified"]]
    passed = len(qualified) >= protocol["behavior_gate"]["minimum_authorized_models"]
    final = {"phase": PHASE, "campaign": CAMPAIGN, "model_order": list(MODELS),
             "qualified_models": qualified, "qualified_model_count": len(qualified),
             "all_gates_passed": passed,
             "authorization": "run_phase1331_relation_kernels" if passed else "close_c042_before_hidden_states",
             "model_summary_sha256": {model: sha(OUT / f"analysis/{model}_summary.json") for model in MODELS},
             "execution_manifest_sha256": sha(OUT / "protocol/execution_manifest.json"),
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--model", choices=MODELS)
    group.add_argument("--finalize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.force)
    elif args.model:
        run_model(args.model)
    else:
        finalize()
