#!/usr/bin/env python3
"""Score one local CUDA model on the frozen Phase602 three-track cases."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
import phase602_three_track_protocol as protocol  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def output_paths(model: str) -> dict[str, Path]:
    stem = protocol.OUT_DIR / f"phase602_{model}_three_track_behavior"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def score_batch(loaded: Any, model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in rows]
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        result = loaded.model(
            **encoded,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    logits = result.logits[:, -1, :].float()
    ledger = json.loads(protocol.PROTOCOL_PATH.read_text())["answer_token_ledger_by_model"][model]
    a_id, b_id = int(ledger["A"][0]), int(ledger["B"][0])
    candidate_logits = logits.index_select(1, torch.tensor([a_id, b_id], device=loaded.input_device))
    candidate_probs = torch.softmax(candidate_logits, dim=1)
    greedy_ids = torch.argmax(logits, dim=1)
    prompt_width = int(encoded["input_ids"].shape[1])
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        target_index = 0 if row["target_letter"] == "A" else 1
        foil_index = 1 - target_index
        target_logit = float(candidate_logits[index, target_index].item())
        foil_logit = float(candidate_logits[index, foil_index].item())
        margin = target_logit - foil_logit
        generated_ids = [int(value) for value in generated[index, prompt_width:].tolist()]
        generated_text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)
        match = re.match(r"^\s*([AB])(?:\b|[.)])", generated_text)
        direct_letter = match.group(1) if match else None
        output.append({
            **row,
            "model": model,
            "target_token_id": a_id if target_index == 0 else b_id,
            "foil_token_id": b_id if target_index == 0 else a_id,
            "target_logit": target_logit,
            "foil_logit": foil_logit,
            "target_margin": margin,
            "target_pair_probability": float(candidate_probs[index, target_index].item()),
            "forced_choice_prediction": "A" if candidate_logits[index, 0] > candidate_logits[index, 1] else "B",
            "forced_choice_correct": margin > 0.0,
            "greedy_first_token_id": int(greedy_ids[index].item()),
            "greedy_first_token_text": loaded.tokenizer.decode([int(greedy_ids[index].item())]),
            "generated_token_ids": generated_ids,
            "generated_text": generated_text,
            "direct_candidate_letter": direct_letter,
            "direct_candidate_output": direct_letter is not None,
            "direct_exact_correct": direct_letter == row["target_letter"],
            "observer_only": True,
            "internal_state_collected": False,
            "causal": False,
        })
    del result, generated, logits, candidate_logits, candidate_probs, greedy_ids, encoded
    return output


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / max(1, len(rows))


def grouped(rows: list[dict[str, Any]], field: str, values: Iterable[Any]) -> dict[str, Any]:
    return {
        str(value): {
            "case_count": len(part := [row for row in rows if row[field] == value]),
            "forced_choice_accuracy": rate(part, "forced_choice_correct"),
        }
        for value in values
    }


def summarize_track(rows: list[dict[str, Any]], track: str) -> dict[str, Any]:
    values = [row for row in rows if row["track"] == track]
    split = grouped(values, "split", protocol.SPLITS)
    membership = grouped(values, "fruit_member", (True, False))
    surface = grouped(values, "surface_id", (f"surface_{i}" for i in range(len(protocol.SURFACE_STEMS))))
    role = grouped(values, "entity_role", protocol.ENTITY_ROLES)
    answer = grouped(values, "target_letter", ("A", "B"))
    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        by_concept[row["concept_id"]].append(row)
    unanimous = sum(
        len(parts) == len(protocol.SURFACE_STEMS) and all(row["forced_choice_correct"] for row in parts)
        for parts in by_concept.values()
    ) / max(1, len(by_concept))
    answer_gap = abs(answer["A"]["forced_choice_accuracy"] - answer["B"]["forced_choice_accuracy"])
    overall = {
        "case_count": len(values),
        "concept_count": len(by_concept),
        "forced_choice_accuracy": rate(values, "forced_choice_correct"),
        "direct_candidate_output_rate": rate(values, "direct_candidate_output"),
        "direct_exact_accuracy": rate(values, "direct_exact_correct"),
        "concept_unanimous_rate": unanimous,
        "answer_order_accuracy_gap": answer_gap,
        "mean_target_margin": sum(float(row["target_margin"]) for row in values) / max(1, len(values)),
    }
    gates = protocol.TRACK_GATES[track]
    checks = {
        "overall": overall["forced_choice_accuracy"] >= gates["overall_accuracy_min"],
        "all_splits": all(item["forced_choice_accuracy"] >= gates["split_accuracy_min"] for item in split.values()),
        "both_memberships": all(item["forced_choice_accuracy"] >= gates["membership_accuracy_min"] for item in membership.values()),
        "all_surfaces": all(item["forced_choice_accuracy"] >= gates["surface_accuracy_min"] for item in surface.values()),
        "answer_order": answer_gap <= gates["answer_order_gap_max"],
        "concept_unanimous": unanimous >= gates["concept_unanimous_rate_min"],
        "direct_candidate_output": overall["direct_candidate_output_rate"] >= gates["direct_candidate_output_rate_min"],
        "direct_exact": overall["direct_exact_accuracy"] >= gates["direct_exact_accuracy_min"],
    }
    return {
        "overall": overall,
        "split_metrics": split,
        "membership_metrics": membership,
        "surface_metrics": surface,
        "entity_role_metrics": role,
        "answer_metrics": answer,
        "gate_checks": checks,
        "behavior_qualified": all(checks.values()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not all(math.isfinite(float(row["target_margin"])) for row in rows):
        raise RuntimeError("Phase602 non-finite target margin")
    track_metrics = {track: summarize_track(rows, track) for track in protocol.TRACKS}
    qualified = [track for track, metrics in track_metrics.items() if metrics["behavior_qualified"]]
    return {
        "case_count": len(rows),
        "concept_count": len({row["concept_id"] for row in rows}),
        "track_metrics": track_metrics,
        "qualified_tracks": qualified,
        "forced_choice_prediction_counts": dict(Counter(row["forced_choice_prediction"] for row in rows)),
    }


def run(model: str, restart: bool = False) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase602 requires CUDA")
    frozen = json.loads(protocol.PROTOCOL_PATH.read_text())
    if frozen["cases_sha256"] != sha256_file(protocol.CASES_PATH):
        raise RuntimeError("Phase602 case-file drift")
    if frozen["foodon_sha256"] != sha256_file(protocol.foodon.SOURCE_PATH):
        raise RuntimeError("Phase602 FoodOn source drift")
    if frozen["wordnet_archive_sha256"] != sha256_file(protocol.ARCHIVE_PATH):
        raise RuntimeError("Phase602 WordNet source drift")
    paths = output_paths(model)
    if restart:
        for path in paths.values():
            path.unlink(missing_ok=True)
    cases = list(read_jsonl(protocol.CASES_PATH))
    write_json(paths["contract"], {
        "schema_version": "phase602_three_track_behavior_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "cases_sha256": sha256_file(protocol.CASES_PATH),
        "case_count": len(cases),
        "internal_state_collection": False,
        "causal_intervention": False,
    })
    loaded = None
    scored: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase602 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        if loaded.tokenizer.pad_token_id is None:
            loaded.tokenizer.pad_token_id = loaded.tokenizer.eos_token_id
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase602 requires BF16, got {dtype}")
        for start in range(0, len(cases), protocol.FIXED_BATCH_SIZE):
            scored.extend(score_batch(loaded, model, cases[start : start + protocol.FIXED_BATCH_SIZE]))
            if start == 0 or (start // protocol.FIXED_BATCH_SIZE + 1) % 10 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {model} Phase602 {min(start + protocol.FIXED_BATCH_SIZE, len(cases))}/{len(cases)}", flush=True)
        write_jsonl(paths["rows"], scored)
        metrics = summarize(scored)
        summary = {
            "schema_version": "phase602_three_track_behavior_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "runtime_seconds": time.monotonic() - started,
            **metrics,
            "rows_sha256": sha256_file(paths["rows"]),
            "internal_state_collected": False,
            "causal_intervention_authorized": False,
        }
        write_json(paths["summary"], summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return paths["summary"]
    finally:
        release_loaded(loaded)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    parser.add_argument("--restart", action="store_true")
    arguments = parser.parse_args()
    run(arguments.model, arguments.restart)
