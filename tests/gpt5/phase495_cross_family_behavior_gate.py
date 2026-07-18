#!/usr/bin/env python3
"""Phase495 CUDA behavior qualification for the frozen six-family protocol.

Run one model per invocation. This stage reads only the open behavior split and
scores the next-token candidates `` true`` and `` false``. It does not generate
labels, inspect hidden states, read sealed data, or perform interventions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402


PROTOCOL_DIR = ROOT / "tests" / "gpt5" / "result" / "phase494_cross_family_trajectory_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase494_frozen_contract.json"
AUDIT_PATH = PROTOCOL_DIR / "phase494_static_audit.json"
SAMPLES_PATH = PROTOCOL_DIR / "phase494_behavior_qualification_samples.jsonl"
PROTOCOL_SOURCE = ROOT / "tests" / "gpt5" / "phase494_cross_family_trajectory_protocol.py"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase495_cross_family_behavior_gate"
MODELS = ("qwen3", "glm4", "deepseek7b")
TRACKS = ("identity", "native_plain_candidate")
TRAIN_FAMILIES = ("marker_inheritance", "signal_assignment")
UNSEEN_FAMILIES = (
    "symmetric_pair",
    "directed_mentor",
    "transitive_precedence",
    "direct_nontransitive",
)
Z = 1.96


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def rate(rows: list[dict[str, Any]], field: str = "correct") -> dict[str, Any]:
    n = len(rows)
    k = sum(bool(row[field]) for row in rows)
    lcb, ucb = wilson(k, n)
    return {"n": n, "count": k, "rate": k / n if n else 0.0, "lcb95": lcb, "ucb95": ucb}


def flatten(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["split"] != "behavior_qualification" or sample["sealed"]:
            raise RuntimeError("Phase495 received a sealed or non-behavior sample")
        for variant in sample["surface_variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "world_case_id": sample["world_case_id"],
                "source_pair_id": sample["source_pair_id"],
                "family": sample["family"],
                "family_role": sample["family_role"],
                "pair_index": sample["pair_index"],
                "world_role": sample["world_role"],
                "truth_value": sample["truth_value"],
                "claim_polarity": sample["claim_polarity"],
                "length_control": sample["length_control"],
                "fact_order": sample["fact_order"],
                "track": variant["track"],
                "semantic_prompt": variant["semantic_prompt"],
            })
    expected = 6 * 72 * 2 * 2
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} behavior rows, got {len(rows)}")
    return rows


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Candidate {text!r} is not one token: {ids}")
    return int(ids[0])


def score_rows(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> None:
    true_id = single_token_id(tokenizer, " true")
    false_id = single_token_id(tokenizer, " false")
    tokenizer.padding_side = "left"
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer(
            [row["semantic_prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :]
        true_scores = logits[:, true_id].float().cpu()
        false_scores = logits[:, false_id].float().cpu()
        for index, row in enumerate(batch):
            margin = float(true_scores[index] - false_scores[index])
            prediction = margin > 0
            row.update({
                "true_logit": float(true_scores[index]),
                "false_logit": float(false_scores[index]),
                "margin_true_minus_false": margin,
                "prediction": prediction,
                "correct": prediction == row["truth_value"],
            })
        if start == 0 or start + len(batch) == len(rows) or (start // batch_size) % 16 == 15:
            log(f"semantic candidates {min(start + len(batch), len(rows))}/{len(rows)}")


def intersection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["world_case_id"]].append(row)
    out = []
    for case_id, items in grouped.items():
        if {item["track"] for item in items} != set(TRACKS) or len(items) != 2:
            raise RuntimeError(f"Incomplete native intersection {case_id}")
        base = items[0]
        out.append({
            "world_case_id": case_id,
            "source_pair_id": base["source_pair_id"],
            "family": base["family"],
            "family_role": base["family_role"],
            "truth_value": base["truth_value"],
            "claim_polarity": base["claim_polarity"],
            "length_control": base["length_control"],
            "fact_order": base["fact_order"],
            "correct": all(item["correct"] for item in items),
        })
    return out


def paired_world_rows(intersections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in intersections:
        grouped[row["source_pair_id"]].append(row)
    out = []
    for pair_id, items in grouped.items():
        if len(items) != 2 or {item["truth_value"] for item in items} != {False, True}:
            raise RuntimeError(f"Incomplete paired-world behavior item {pair_id}")
        base = items[0]
        out.append({"source_pair_id": pair_id, "family": base["family"], "correct": all(item["correct"] for item in items)})
    return out


def family_report(family: str, rows: list[dict[str, Any]], intersections: list[dict[str, Any]], pairs: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    family_rows = [row for row in rows if row["family"] == family]
    family_intersections = [row for row in intersections if row["family"] == family]
    family_pairs = [row for row in pairs if row["family"] == family]
    tracks = {track: rate([row for row in family_rows if row["track"] == track]) for track in TRACKS}
    intersection = rate(family_intersections)
    paired = rate(family_pairs)
    passed = (
        tracks["identity"]["lcb95"] >= gate["per_family_identity_lcb95_min"]
        and tracks["native_plain_candidate"]["lcb95"] >= gate["per_family_native_plain_lcb95_min"]
        and intersection["lcb95"] >= gate["per_family_native_intersection_lcb95_min"]
        and paired["lcb95"] >= gate["per_family_paired_world_lcb95_min"]
    )
    return {
        "family_role": "fit" if family in TRAIN_FAMILIES else "unseen_prediction",
        "tracks": tracks,
        "native_intersection": intersection,
        "paired_world_all_correct": paired,
        "by_length": {
            value: rate([row for row in family_intersections if row["length_control"] == value])
            for value in ("short", "medium", "long")
        },
        "by_fact_order": {
            value: rate([row for row in family_intersections if row["fact_order"] == value])
            for value in ("target_first", "distractor_first", "interleaved")
        },
        "by_claim_polarity": {
            value: rate([row for row in family_intersections if row["claim_polarity"] == value])
            for value in ("positive", "negative")
        },
        "behavior_gate_pass": passed,
    }


def build_summary(model_key: str, rows: list[dict[str, Any]], runtime: float) -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    gate = contract["behavior_gate"]
    intersections = intersection_rows(rows)
    pairs = paired_world_rows(intersections)
    families = {
        family: family_report(family, rows, intersections, pairs, gate)
        for family in TRAIN_FAMILIES + UNSEEN_FAMILIES
    }
    unseen_intersections = [row for row in intersections if row["family"] in UNSEEN_FAMILIES]
    unseen_overall = rate(unseen_intersections)
    fit_pass = all(families[family]["behavior_gate_pass"] for family in TRAIN_FAMILIES)
    passed_unseen = [family for family in UNSEEN_FAMILIES if families[family]["behavior_gate_pass"]]
    all_unseen_pass = (
        len(passed_unseen) == len(UNSEEN_FAMILIES)
        and unseen_overall["lcb95"] >= gate["overall_unseen_intersection_lcb95_min"]
    )
    return {
        "schema_version": "phase495_cross_family_behavior_gate.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "behavior_qualification_complete",
        "model": model_key,
        "cuda_used": True,
        "model_weights_loaded": True,
        "runtime_seconds": runtime,
        "input": str(SAMPLES_PATH.relative_to(ROOT)),
        "input_sha256": sha256_file(SAMPLES_PATH),
        "row_count": len(rows),
        "sealed_split_read": False,
        "families": families,
        "overall": {
            "all_native_rows": rate(rows),
            "native_intersection": rate(intersections),
            "paired_world_all_correct": rate(pairs),
            "unseen_native_intersection": unseen_overall,
        },
        "gates": {
            "fit_families_behavior_pass": fit_pass,
            "passed_unseen_families": passed_unseen,
            "all_unseen_families_behavior_pass": all_unseen_pass,
        },
        "authorization": {
            "formation_fit_authorized": fit_pass,
            "physical_prediction_families": passed_unseen if fit_pass else [],
            "cross_family_physical_candidate": fit_pass and all_unseen_pass,
            "sealed_read": False,
            "causal_intervention": False,
            "head_channel_neuron_scan": False,
        },
    }


def verify_freeze() -> dict[str, Any]:
    audit = load_json(AUDIT_PATH)
    contract = load_json(CONTRACT_PATH)
    if audit["status"] != "static_pass_no_model_run" or not audit["authorization"]["three_model_behavior_qualification"]:
        raise RuntimeError("Phase494 static audit did not authorize behavior testing")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase494 protocol source changed after freeze")
    expected = contract["split_files"]["behavior_qualification"]["sha256"]
    if sha256_file(SAMPLES_PATH) != expected:
        raise RuntimeError("Phase494 behavior sample hash drift")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Debug only; zero means the full frozen split.")
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    verify_freeze()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase495 requires CUDA")
    rows = flatten(load_jsonl(SAMPLES_PATH))
    if args.limit:
        rows = rows[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    started = time.monotonic()
    try:
        model, tokenizer, device = load_model(args.model, use_8bit=True if args.use_8bit else None)
        score_rows(model, tokenizer, device, rows, args.batch_size)
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    runtime = time.monotonic() - started
    for row in rows:
        row["model"] = args.model
        row.pop("semantic_prompt", None)
    rows_path = OUT_DIR / f"phase495_{args.model}_rows.jsonl"
    summary_path = OUT_DIR / f"phase495_{args.model}_summary.json"
    write_jsonl(rows_path, rows)
    summary = build_summary(args.model, rows, runtime)
    summary["debug_limit_used"] = args.limit
    summary["full_frozen_split"] = args.limit == 0
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(rows_path)
    print(summary_path)


if __name__ == "__main__":
    main()
