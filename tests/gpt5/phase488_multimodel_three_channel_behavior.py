#!/usr/bin/env python3
"""Phase488 independent three-channel behavior qualification.

One model per invocation. Reads only the Phase487 behavior-qualification file.
Measures:

* direct true/false candidate selection;
* direct mapped A/B candidate selection;
* greedy output events parsed by the frozen target-blind observer.

No hidden-state trace, sealed read, intervention, or parameter scan is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase487_dual_observer_native_core_protocol import parse_output_event  # noqa: E402


PHASE487_DIR = ROOT / "tests" / "gpt5" / "result" / "phase487_dual_observer_native_core_protocol"
SAMPLES_PATH = PHASE487_DIR / "phase487_behavior_qualification_samples.jsonl"
AUDIT_PATH = PHASE487_DIR / "phase487_static_audit.json"
MANIFEST_PATH = PHASE487_DIR / "phase487_manifest.json"
PROTOCOL_SOURCE = ROOT / "tests" / "gpt5" / "phase487_dual_observer_native_core_protocol.py"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase488_multimodel_three_channel_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")
NATIVE_TRACKS = ("identity", "native_plain_candidate")
Z = 1.96


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def flatten_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["split"] != "behavior_qualification" or sample["sealed"]:
            raise RuntimeError("Phase488 received a non-behavior or sealed sample")
        for variant in sample["surface_variants"]:
            rows.append({
                "model": None,
                "sample_id": sample["sample_id"],
                "semantic_case_id": f"{sample['source_pair_id']}::{sample['pair_role']}",
                "source_pair_id": sample["source_pair_id"],
                "split": sample["split"],
                "family": sample["family"],
                "pair_index": sample["pair_index"],
                "pair_role": sample["pair_role"],
                "truth_value": sample["truth_value"],
                "label_mapping": sample["label_mapping"],
                "expected_label": sample["expected_label"],
                "target_slot": sample["target_slot"],
                "mapping_position": sample["mapping_position"],
                "fact_order": sample["fact_order"],
                "track": variant["track"],
                "track_class": variant["track_class"],
                "semantic_prompt": variant["semantic_prompt"],
                "event_prompt": variant["event_prompt"],
            })
    return rows


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Candidate {text!r} is not one token: {ids}")
    return int(ids[0])


def score_semantic_prompts(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["semantic_case_id"], row["track"])
        previous = unique.get(key)
        if previous is not None and previous["semantic_prompt"] != row["semantic_prompt"]:
            raise RuntimeError(f"Semantic prompt differs across label mappings for {key}")
        unique[key] = row
    items = list(unique.items())
    true_id = single_token_id(tokenizer, " true")
    false_id = single_token_id(tokenizer, " false")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [row["semantic_prompt"] for _key, row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :]
        true_scores = logits[:, true_id].float().cpu()
        false_scores = logits[:, false_id].float().cpu()
        for index, (key, row) in enumerate(batch):
            margin = float(true_scores[index] - false_scores[index])
            prediction = margin > 0
            out[key] = {
                "semantic_true_logit": float(true_scores[index]),
                "semantic_false_logit": float(false_scores[index]),
                "semantic_margin_true_minus_false": margin,
                "semantic_candidate_prediction": prediction,
                "semantic_candidate_correct": prediction == row["truth_value"],
            }
        log(f"semantic candidates {min(start + len(batch), len(items))}/{len(items)}")
    return out


def generate_events(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    a_id = single_token_id(tokenizer, " A")
    b_id = single_token_id(tokenizer, " B")
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [row["event_prompt"] for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        prompt_width = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        if not generated.scores:
            raise RuntimeError("Generation returned no first-step scores")
        first_scores = generated.scores[0].float().cpu()
        for index, row in enumerate(batch):
            text = tokenizer.decode(generated.sequences[index, prompt_width:], skip_special_tokens=True)
            margin = float(first_scores[index, a_id] - first_scores[index, b_id])
            label_prediction = "A" if margin > 0 else "B"
            parsed = parse_output_event(text, row["label_mapping"])
            semantic_generation_correct = (
                parsed["semantic_value"] is not None
                and parsed["semantic_value"] == row["truth_value"]
            )
            label_generation_correct = (
                parsed["label_value"] is not None
                and parsed["label_value"] == row["expected_label"]
            )
            strict_event_correct = (
                parsed["event_type"] == "strict_single_label"
                and parsed["label_value"] == row["expected_label"]
            )
            out.append({
                **row,
                "label_a_first_step_score": float(first_scores[index, a_id]),
                "label_b_first_step_score": float(first_scores[index, b_id]),
                "label_margin_a_minus_b": margin,
                "label_candidate_prediction": label_prediction,
                "label_candidate_correct": label_prediction == row["expected_label"],
                "generated_text": text,
                **parsed,
                "semantic_generation_correct": semantic_generation_correct,
                "label_generation_correct": label_generation_correct,
                "strict_event_correct": strict_event_correct,
            })
        log(f"output events {min(start + len(batch), len(rows))}/{len(rows)}")
    return out


def rate_report(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    n = len(rows)
    k = sum(bool(row[field]) for row in rows)
    lcb, ucb = wilson(k, n)
    return {"n": n, "count": k, "rate": k / n if n else 0.0, "lcb95": lcb, "ucb95": ucb}


def unrecoverable_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(row["semantic_value"] is None for row in rows)
    lcb, ucb = wilson(k, n)
    return {"n": n, "count": k, "rate": k / n if n else 0.0, "lcb95": lcb, "ucb95": ucb}


def unique_semantic_rows(rows: list[dict[str, Any]], track: str) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["track"] != track:
            continue
        previous = selected.get(row["semantic_case_id"])
        if previous is not None and previous["semantic_candidate_correct"] != row["semantic_candidate_correct"]:
            raise RuntimeError("Duplicated semantic case changed across label mapping")
        selected[row["semantic_case_id"]] = row
    return list(selected.values())


def paired_intersection(rows: list[dict[str, Any]], field: str, semantic_unique: bool) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["track"] not in NATIVE_TRACKS:
            continue
        key = row["semantic_case_id"] if semantic_unique else row["sample_id"]
        previous = grouped[key].get(row["track"])
        if previous is not None:
            if semantic_unique and previous[field] == row[field]:
                continue
            raise RuntimeError(f"Duplicate paired key {key}/{row['track']}")
        grouped[key][row["track"]] = row
    complete = [pair for pair in grouped.values() if set(pair) == set(NATIVE_TRACKS)]
    k = sum(all(pair[track][field] for track in NATIVE_TRACKS) for pair in complete)
    lcb, ucb = wilson(k, len(complete))
    return {"n": len(complete), "count": k, "rate": k / len(complete) if complete else 0.0, "lcb95": lcb, "ucb95": ucb}


def counterfactual_pair_report(rows: list[dict[str, Any]], track: str) -> dict[str, Any]:
    unique = unique_semantic_rows(rows, track)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique:
        grouped[row["source_pair_id"]].append(row)
    complete = [items for items in grouped.values() if {row["pair_role"] for row in items} == {"entailed", "counterfactual"}]
    k = sum(all(row["semantic_candidate_correct"] for row in items) for items in complete)
    lcb, ucb = wilson(k, len(complete))
    return {"n_pairs": len(complete), "count": k, "rate": k / len(complete) if complete else 0.0, "lcb95": lcb, "ucb95": ucb}


def build_summary(model_key: str, rows: list[dict[str, Any]], runtime_seconds: float) -> dict[str, Any]:
    tracks = {}
    for track in sorted({row["track"] for row in rows}):
        track_rows = [row for row in rows if row["track"] == track]
        semantic_rows = unique_semantic_rows(rows, track)
        tracks[track] = {
            "n_event_rows": len(track_rows),
            "n_unique_semantic_cases": len(semantic_rows),
            "semantic_candidate": rate_report(semantic_rows, "semantic_candidate_correct"),
            "semantic_counterfactual_pairs": counterfactual_pair_report(rows, track),
            "label_candidate": rate_report(track_rows, "label_candidate_correct"),
            "semantic_generation": rate_report(track_rows, "semantic_generation_correct"),
            "label_generation": rate_report(track_rows, "label_generation_correct"),
            "strict_event": rate_report(track_rows, "strict_event_correct"),
            "unrecoverable": unrecoverable_report(track_rows),
            "event_counts": dict(Counter(row["event_type"] for row in track_rows)),
            "by_truth": {
                str(value): rate_report([row for row in semantic_rows if row["truth_value"] == value], "semantic_candidate_correct")
                for value in (False, True)
            },
            "by_family": {
                family: rate_report([row for row in semantic_rows if row["family"] == family], "semantic_candidate_correct")
                for family in sorted({row["family"] for row in semantic_rows})
            },
        }

    semantic_intersection = paired_intersection(rows, "semantic_candidate_correct", semantic_unique=True)
    label_intersection = paired_intersection(rows, "label_candidate_correct", semantic_unique=False)
    semantic_gate = (
        tracks["identity"]["semantic_candidate"]["lcb95"] >= 0.95
        and tracks["native_plain_candidate"]["semantic_candidate"]["lcb95"] >= 0.95
        and semantic_intersection["lcb95"] >= 0.90
        and tracks["identity"]["semantic_counterfactual_pairs"]["lcb95"] >= 0.90
        and tracks["native_plain_candidate"]["semantic_counterfactual_pairs"]["lcb95"] >= 0.90
    )
    label_gate = (
        tracks["identity"]["label_candidate"]["lcb95"] >= 0.95
        and tracks["native_plain_candidate"]["label_candidate"]["lcb95"] >= 0.95
        and label_intersection["lcb95"] >= 0.90
    )
    native_rows = [row for row in rows if row["track"] in NATIVE_TRACKS]
    native_unrecoverable = unrecoverable_report(native_rows)
    event_gate = native_unrecoverable["ucb95"] <= 0.05
    return {
        "schema_version": "phase488_multimodel_three_channel_behavior.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "behavior_qualification_complete",
        "model": model_key,
        "cuda_used": torch.cuda.is_available(),
        "model_weights_loaded": True,
        "runtime_seconds": runtime_seconds,
        "input": str(SAMPLES_PATH.relative_to(ROOT)),
        "input_sha256": sha256_file(SAMPLES_PATH),
        "sealed_split_read": False,
        "row_count": len(rows),
        "parser_qualification": {
            "target_blind_engineering_conformance": True,
            "independent_human_precision_qualified": False,
            "event_physical_map_authorized": False,
        },
        "tracks": tracks,
        "native_intersections": {
            "semantic_candidate": semantic_intersection,
            "label_candidate": label_intersection,
        },
        "native_event_unrecoverable": native_unrecoverable,
        "gates": {
            "relation_semantic_behavior_pass": semantic_gate,
            "label_binding_behavior_pass": label_gate,
            "output_event_behavior_pass": event_gate,
        },
        "authorization": {
            "open_relation_geometry_authorized": semantic_gate,
            "open_output_event_physical_map_authorized": False,
            "physical_prediction_authorized": False,
            "sealed_read_authorized": False,
            "head_channel_neuron_scan_authorized": False,
        },
    }


def verify_freeze() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if audit["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase487 static audit did not pass")
    if not audit["authorization"]["new_behavior_qualification_authorized"]:
        raise RuntimeError("Phase487 behavior run is not authorized")
    if sha256_file(PROTOCOL_SOURCE) != manifest["source_sha256"]:
        raise RuntimeError("Phase487 parser/protocol source changed after freeze")
    expected = manifest["split_files"]["behavior_qualification"]["sha256"]
    if sha256_file(SAMPLES_PATH) != expected:
        raise RuntimeError("Phase487 behavior sample hash drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Debug only; zero means full frozen set.")
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    verify_freeze()
    samples = load_jsonl(SAMPLES_PATH)
    rows = flatten_samples(samples)
    if args.limit:
        rows = rows[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    started = time.monotonic()
    try:
        model, tokenizer, device = load_model(args.model, use_8bit=True if args.use_8bit else None)
        tokenizer.padding_side = "left"
        semantic_scores = score_semantic_prompts(model, tokenizer, device, rows, args.batch_size)
        for row in rows:
            row["model"] = args.model
            row.update(semantic_scores[(row["semantic_case_id"], row["track"])])
        records = generate_events(model, tokenizer, device, rows, args.batch_size, args.max_new_tokens)
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    runtime = time.monotonic() - started
    rows_path = OUT_DIR / f"phase488_{args.model}_rows.jsonl"
    summary_path = OUT_DIR / f"phase488_{args.model}_summary.json"
    write_jsonl(rows_path, records)
    summary = build_summary(args.model, records, runtime)
    summary["debug_limit_used"] = args.limit
    summary["full_frozen_set"] = args.limit == 0
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(rows_path)
    print(summary_path)


if __name__ == "__main__":
    main()
