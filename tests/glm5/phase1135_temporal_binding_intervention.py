#!/usr/bin/env python3
"""Phase1135 temporal binding behavior gate and residual event localization.

The source package is external-model reviewed but not human annotated. Every
output therefore remains exploratory and is ineligible for Phase1132 human
evidence. Hidden-state scans are authorized only after a frozen two-model
behavior gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import (  # noqa: E402
    load_fp16,
    quantization_audit,
    release_fp16,
)


PHASE = 1135
MODELS = ("qwen3", "glm4", "deepseek7b")
SOURCE = (
    ROOT
    / "tests/glm5/result/phase1134_external_api_temporal_annotation"
    / "analysis/external_machine_consensus_package.jsonl"
)
OUT_ROOT = ROOT / "tests/glm5/result/phase1135_temporal_binding_intervention"

BATCH_SIZE = {"qwen3": 8, "glm4": 4, "deepseek7b": 4}
CAUSAL_ITEMS_PER_SPLIT = 16
DEPTH_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
EPSILON = 1e-8

BEHAVIOR_THRESHOLDS = {
    "finite_fraction": 0.99,
    "state_accuracy": 0.80,
    "all_four_accuracy": 0.65,
    "median_correct_margin": 0.0,
    "required_splits": ("discovery", "confirmation"),
    "minimum_authorized_models": 2,
}
CAUSAL_THRESHOLDS = {
    "finite_fraction": 0.99,
    "main_median_recovery": 0.10,
    "main_positive_fraction": 0.65,
    "panel_median_recovery": 0.05,
    "specificity_advantage": 0.05,
    "self_patch_max_abs_margin_change": 0.02,
    "minimum_confirmed_models": 2,
}

BEHAVIOR_STATES = (
    "original_pre",
    "original_post",
    "swapped_pre",
    "swapped_post",
    "prior_pre",
    "prior_post",
)
GATED_STATES = (
    "original_pre",
    "original_post",
    "swapped_pre",
    "swapped_post",
)
CAUSAL_STATES = (
    "original_pre",
    "original_post",
    "swapped_pre",
    "swapped_post",
    "original_pre_early",
    "original_post_late",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def stable_key(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def median(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(finite) if finite else None


def mean(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(finite) / len(finite) if finite else None


def parse_day(value: str) -> date:
    return date.fromisoformat(value[:10])


def replace_day(query: str, old: str, new: str) -> str:
    if query.count(old) != 1:
        raise ValueError(f"query date occurrence drift: {old!r} in {query!r}")
    return query.replace(old, new)


def swap_labels(text: str, old: str, new: str) -> str:
    if old == new:
        raise ValueError("old and new labels must differ")
    pattern = re.compile("|".join(re.escape(value) for value in sorted((old, new), key=len, reverse=True)))
    counts = {old: 0, new: 0}

    def replacement(match: re.Match[str]) -> str:
        value = match.group(0)
        counts[value] += 1
        return new if value == old else old

    result = pattern.sub(replacement, text)
    if counts[old] < 1 or counts[new] < 1:
        raise ValueError(f"context does not expose both holders: {old!r}, {new!r}")
    return result


def candidate_order(item_id: str, old: str, new: str) -> list[str]:
    return [old, new] if int(stable_key(PHASE, item_id, "order")[:2], 16) % 2 == 0 else [new, old]


def build_prompt(context: str | None, query: str, candidates: list[str]) -> str:
    choices = " | ".join(candidates)
    if context is None:
        return (
            "No dated record is provided.\n"
            f"Question: {query}\n"
            f"Choose one answer: {choices}\n"
            "Answer:"
        )
    return (
        "Use only the dated record below, even if it conflicts with prior knowledge.\n"
        f"Dated record: {context}\n"
        f"Question: {query}\n"
        f"Choose one answer: {choices}\n"
        "Answer:"
    )


def item_state_specs(item: dict[str, Any], include_causal_nulls: bool = False) -> dict[str, dict[str, Any]]:
    old = str(item["matched_null_candidate"])
    new = str(item["active_candidate"])
    original = str(item["context"])
    swapped = swap_labels(original, old, new)
    pre_query = str(item["paired_pre_query"])
    post_query = str(item["query"])
    specs: dict[str, dict[str, Any]] = {
        "original_pre": {"context": original, "query": pre_query, "expected": "old"},
        "original_post": {"context": original, "query": post_query, "expected": "new"},
        "swapped_pre": {"context": swapped, "query": pre_query, "expected": "new"},
        "swapped_post": {"context": swapped, "query": post_query, "expected": "old"},
        "prior_pre": {"context": None, "query": pre_query, "expected": None},
        "prior_post": {"context": None, "query": post_query, "expected": None},
    }
    if include_causal_nulls:
        pre_day = parse_day(str(item["pre_query_date"]))
        post_day = parse_day(str(item["post_query_date"]))
        early = (pre_day - timedelta(days=7)).isoformat()
        late = (post_day + timedelta(days=7)).isoformat()
        specs["original_pre_early"] = {
            "context": original,
            "query": replace_day(pre_query, pre_day.isoformat(), early),
            "expected": "old",
        }
        specs["original_post_late"] = {
            "context": original,
            "query": replace_day(post_query, post_day.isoformat(), late),
            "expected": "new",
        }
    return specs


def make_case(item: dict[str, Any], state: str, spec: dict[str, Any]) -> dict[str, Any]:
    old = str(item["matched_null_candidate"])
    new = str(item["active_candidate"])
    order = candidate_order(str(item["item_id"]), old, new)
    case_id = f"{item['item_id']}|{state}"
    return {
        "schema_version": "phase1135_logical_case.v1",
        "phase": PHASE,
        "case_id": case_id,
        "item_id": str(item["item_id"]),
        "split": str(item["split"]),
        "property_id": str(item["property_id"]),
        "domain": str(item["domain"]),
        "state": state,
        "context": spec["context"],
        "query": spec["query"],
        "expected_key": spec["expected"],
        "old_candidate": old,
        "new_candidate": new,
        "candidate_order": order,
        "prompt": build_prompt(spec["context"], str(spec["query"]), order),
        "machine_validation_only": True,
        "human_annotation_eligible": False,
    }


def stratified_causal_ids(items: list[dict[str, Any]], split: str) -> list[str]:
    subset = [row for row in items if row["split"] == split]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        groups[str(row["property_id"])].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: stable_key(PHASE, split, key, row["item_id"]))
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < CAUSAL_ITEMS_PER_SPLIT:
        added = False
        for key in sorted(groups):
            if round_index < len(groups[key]):
                selected.append(groups[key][round_index])
                added = True
                if len(selected) == CAUSAL_ITEMS_PER_SPLIT:
                    break
        if not added:
            break
        round_index += 1
    if len(selected) != CAUSAL_ITEMS_PER_SPLIT:
        raise RuntimeError(f"cannot select {CAUSAL_ITEMS_PER_SPLIT} causal items for {split}")
    return [str(row["item_id"]) for row in selected]


def protocol_command() -> None:
    items = read_jsonl(SOURCE)
    causal_ids = {
        split: stratified_causal_ids(items, split)
        for split in ("discovery", "confirmation")
    }
    logical_cases = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for item in items:
        specs = item_state_specs(item)
        for state in BEHAVIOR_STATES:
            logical_cases.append(make_case(item, state, specs[state]))

    check("source_exists", SOURCE.exists(), str(SOURCE))
    check("source_count_491", len(items) == 491, len(items))
    check("source_ids_unique", len({row["item_id"] for row in items}) == len(items), len(items))
    check(
        "machine_only_scope",
        all(
            row.get("machine_validation_only") is True
            and row.get("annotation_blinded_to_model_outputs") is False
            and row.get("external_machine_review", {}).get("human_reviewer") is False
            for row in items
        ),
        len(items),
    )
    split_counts = {key: sum(row["split"] == key for row in items) for key in ("discovery", "confirmation", "natural_use")}
    check("three_frozen_splits", split_counts == {"discovery": 164, "confirmation": 164, "natural_use": 163}, split_counts)
    check("logical_case_count", len(logical_cases) == 491 * len(BEHAVIOR_STATES), len(logical_cases))
    check("logical_case_ids_unique", len({row["case_id"] for row in logical_cases}) == len(logical_cases), len(logical_cases))
    check(
        "postrelease_starts",
        all(parse_day(str(row["new_start"])) >= date(2025, 1, 1) for row in items),
        min(str(row["new_start"]) for row in items),
    )
    check(
        "counterfactual_swap_changes_context",
        all(item_state_specs(row)["swapped_pre"]["context"] != row["context"] for row in items),
        len(items),
    )
    check(
        "causal_split_disjoint",
        set(causal_ids["discovery"]).isdisjoint(causal_ids["confirmation"]),
        {key: len(value) for key, value in causal_ids.items()},
    )
    check(
        "causal_property_coverage",
        all(
            len({row["property_id"] for row in items if row["item_id"] in causal_ids[split]}) >= 5
            for split in causal_ids
        ),
        {
            split: sorted({row["property_id"] for row in items if row["item_id"] in ids})
            for split, ids in causal_ids.items()
        },
    )
    audit = {
        "schema_version": "phase1135_protocol_audit.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "checks": checks,
        "passed_count": sum(row["passed"] for row in checks),
        "check_count": len(checks),
        "all_checks_passed": all(row["passed"] for row in checks),
    }
    protocol = {
        "schema_version": "phase1135_preregistration.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "source_path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256_file(SOURCE),
        "source_count": len(items),
        "evidence_scope": "exploratory_external_machine_consensus_not_human_gold",
        "human_annotation_eligible": False,
        "models": list(MODELS),
        "precision": "FP16 weights only; no quantization",
        "behavior_states": list(BEHAVIOR_STATES),
        "gated_states": list(GATED_STATES),
        "behavior_thresholds": BEHAVIOR_THRESHOLDS,
        "causal_thresholds": CAUSAL_THRESHOLDS,
        "causal_items": causal_ids,
        "causal_depth_fractions": list(DEPTH_FRACTIONS),
        "causal_component": "residual_stream_after_layer_at_answer_boundary",
        "causal_patch": "exact residual-state replacement",
        "negative_controls": [
            "same-answer temporal donor",
            "cross-item shuffled donor",
            "self replacement",
            "swapped-binding counterfactual panel",
        ],
        "hard_stops": [
            "no hidden scan unless at least two models pass discovery and confirmation behavior gates",
            "no component or neuron search unless at least two models pass independent causal confirmation",
            "machine consensus cannot upgrade Phase1132 human evidence",
        ],
    }
    protocol["protocol_digest"] = digest(protocol)
    audit["protocol_digest"] = protocol["protocol_digest"]
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    write_jsonl(OUT_ROOT / "protocol/logical_cases.jsonl", logical_cases)
    write_json(OUT_ROOT / "protocol/audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"protocol audit failed: {checks}")
    print(json.dumps({
        "phase": PHASE,
        "command": "protocol",
        "source_count": len(items),
        "logical_cases": len(logical_cases),
        "causal_items": {key: len(value) for key, value in causal_ids.items()},
        "audit": f"{audit['passed_count']}/{audit['check_count']}",
        "protocol_digest": protocol["protocol_digest"],
    }, ensure_ascii=False), flush=True)


def tokenize_case(tokenizer, case: dict[str, Any], candidate_key: str) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(str(case["prompt"]), add_special_tokens=False)
    candidate = str(case[f"{candidate_key}_candidate"])
    continuation_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
    if not prompt_ids or not continuation_ids:
        raise RuntimeError(f"empty tokenization for {case['case_id']} {candidate_key}")
    return {
        "case_id": str(case["case_id"]),
        "item_id": str(case["item_id"]),
        "split": str(case["split"]),
        "property_id": str(case["property_id"]),
        "domain": str(case["domain"]),
        "state": str(case["state"]),
        "expected_key": case["expected_key"],
        "candidate_key": candidate_key,
        "candidate": candidate,
        "candidate_order": list(case["candidate_order"]),
        "prompt_ids": prompt_ids,
        "continuation_ids": continuation_ids,
        "input_ids": prompt_ids + continuation_ids,
        "prompt_length": len(prompt_ids),
    }


def pad_sequences(rows: list[dict[str, Any]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for slot, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        ids[slot, : len(values)] = values
        mask[slot, : len(values)] = 1
    return ids, mask


def scores_from_logits(logits: torch.Tensor, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_logits = []
    selected_targets = []
    ownership = []
    for slot, row in enumerate(rows):
        prompt_length = int(row["prompt_length"])
        continuation = list(row["continuation_ids"])
        for offset, token_id in enumerate(continuation):
            selected_logits.append(logits[slot, prompt_length - 1 + offset, :].float())
            selected_targets.append(int(token_id))
            ownership.append(slot)
    matrix = torch.stack(selected_logits, dim=0)
    targets = torch.tensor(selected_targets, dtype=torch.long, device=matrix.device)
    token_logp = -F.cross_entropy(matrix, targets, reduction="none")
    grouped: list[list[float]] = [[] for _ in rows]
    for owner, value in zip(ownership, token_logp.detach().cpu().tolist()):
        grouped[owner].append(float(value))
    result = []
    for values in grouped:
        total = sum(values)
        avg = total / len(values)
        result.append({
            "token_count": len(values),
            "logp_sum": total,
            "logp_mean": avg,
            "finite": math.isfinite(total) and math.isfinite(avg),
        })
    del matrix, targets, token_logp
    return result


def score_rows(model, rows: list[dict[str, Any]], pad_id: int, device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            input_ids, attention_mask = pad_sequences(batch, pad_id, device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            scored = scores_from_logits(output.logits, batch)
            for row, score in zip(batch, scored):
                output_rows.append({**{key: value for key, value in row.items() if key not in ("input_ids", "prompt_ids", "continuation_ids")}, **score})
            del output, input_ids, attention_mask, scored
    return output_rows


def behavior_command(model_name: str) -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    audit = read_json(OUT_ROOT / "protocol/audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("protocol audit did not pass")
    logical_cases = read_jsonl(OUT_ROOT / "protocol/logical_cases.jsonl")
    started = time.time()
    model = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError(f"FP16/no-quantization audit failed: {precision}")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        expanded = [
            tokenize_case(tokenizer, case, candidate_key)
            for case in logical_cases
            for candidate_key in ("old", "new")
        ]
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        scores = score_rows(model, expanded, int(pad_id), device, BATCH_SIZE[model_name])
        elapsed = time.time() - started
        summary = {
            "schema_version": "phase1135_behavior_scan_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "protocol_digest": protocol["protocol_digest"],
            "logical_case_count": len(logical_cases),
            "candidate_score_count": len(scores),
            "finite_fraction": sum(row["finite"] for row in scores) / max(len(scores), 1),
            "precision": precision,
            "placement": placement,
            "batch_size": BATCH_SIZE[model_name],
            "elapsed_seconds": elapsed,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "evidence_scope": "exploratory_machine_consensus",
        }
        summary["summary_digest"] = digest(summary)
        output_root = OUT_ROOT / "behavior" / model_name
        write_jsonl(output_root / "scores.jsonl", scores)
        write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "phase": PHASE,
            "command": "behavior",
            "model": model_name,
            "scores": len(scores),
            "finite_fraction": summary["finite_fraction"],
            "elapsed_seconds": elapsed,
            "summary_digest": summary["summary_digest"],
        }, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            release_fp16(model)


def decisions_for_model(model_name: str) -> list[dict[str, Any]]:
    scores = read_jsonl(OUT_ROOT / "behavior" / model_name / "scores.jsonl")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scores:
        grouped[str(row["case_id"])][str(row["candidate_key"])] = row
    decisions = []
    for case_id, candidates in sorted(grouped.items()):
        if set(candidates) != {"old", "new"}:
            raise RuntimeError(f"candidate coverage drift for {model_name} {case_id}")
        old = candidates["old"]
        new = candidates["new"]
        finite = bool(old["finite"] and new["finite"])
        margin_new_minus_old = float(new["logp_mean"] - old["logp_mean"]) if finite else None
        expected = old["expected_key"]
        if expected == "new":
            correct_margin = margin_new_minus_old if finite else None
            correct = finite and margin_new_minus_old is not None and margin_new_minus_old > 0.0
        elif expected == "old":
            correct_margin = -margin_new_minus_old if finite and margin_new_minus_old is not None else None
            correct = finite and margin_new_minus_old is not None and margin_new_minus_old < 0.0
        else:
            correct_margin = None
            correct = None
        decisions.append({
            "schema_version": "phase1135_behavior_decision.v1",
            "phase": PHASE,
            "model": model_name,
            "case_id": case_id,
            "item_id": old["item_id"],
            "split": old["split"],
            "property_id": old["property_id"],
            "domain": old["domain"],
            "state": old["state"],
            "expected_key": expected,
            "candidate_order": old["candidate_order"],
            "finite": finite,
            "old_logp_mean": old["logp_mean"] if old["finite"] else None,
            "new_logp_mean": new["logp_mean"] if new["finite"] else None,
            "margin_new_minus_old": margin_new_minus_old,
            "correct_margin": correct_margin,
            "correct": correct,
        })
    return decisions


def behavior_metrics(decisions: list[dict[str, Any]], split: str) -> dict[str, Any]:
    rows = [row for row in decisions if row["split"] == split]
    by_state = {state: [row for row in rows if row["state"] == state] for state in BEHAVIOR_STATES}
    state_metrics = {}
    for state, state_rows in by_state.items():
        finite_rows = [row for row in state_rows if row["finite"]]
        gated = state in GATED_STATES
        state_metrics[state] = {
            "count": len(state_rows),
            "finite_fraction": len(finite_rows) / max(len(state_rows), 1),
            "accuracy": (
                sum(bool(row["correct"]) for row in state_rows) / max(len(state_rows), 1)
                if gated else None
            ),
            "median_correct_margin": (
                median(row["correct_margin"] for row in finite_rows if row["correct_margin"] is not None)
                if gated else None
            ),
            "median_new_minus_old": median(row["margin_new_minus_old"] for row in finite_rows),
        }
    item_states: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        item_states[str(row["item_id"])][str(row["state"])] = row
    complete = [
        states
        for states in item_states.values()
        if all(state in states for state in GATED_STATES)
    ]
    all_four_accuracy = sum(
        all(bool(states[state]["correct"]) for state in GATED_STATES)
        for states in complete
    ) / max(len(complete), 1)
    context_gains = []
    binding_interactions = []
    for states in complete:
        if all(states[name]["finite"] for name in GATED_STATES):
            binding_interactions.append(0.5 * (
                (
                    float(states["original_post"]["margin_new_minus_old"])
                    - float(states["original_pre"]["margin_new_minus_old"])
                )
                - (
                    float(states["swapped_post"]["margin_new_minus_old"])
                    - float(states["swapped_pre"]["margin_new_minus_old"])
                )
            ))
        if "prior_pre" not in states or "prior_post" not in states:
            continue
        required = (*GATED_STATES, "prior_pre", "prior_post")
        if not all(states[name]["finite"] for name in required):
            continue
        prior_pre = float(states["prior_pre"]["margin_new_minus_old"])
        prior_post = float(states["prior_post"]["margin_new_minus_old"])
        context_gains.extend((
            -float(states["original_pre"]["margin_new_minus_old"]) - (-prior_pre),
            float(states["original_post"]["margin_new_minus_old"]) - prior_post,
            float(states["swapped_pre"]["margin_new_minus_old"]) - prior_pre,
            -float(states["swapped_post"]["margin_new_minus_old"]) - (-prior_post),
        ))
    finite_fraction = sum(row["finite"] for row in rows) / max(len(rows), 1)
    passed = (
        finite_fraction >= BEHAVIOR_THRESHOLDS["finite_fraction"]
        and all(state_metrics[state]["accuracy"] >= BEHAVIOR_THRESHOLDS["state_accuracy"] for state in GATED_STATES)
        and all(state_metrics[state]["median_correct_margin"] > BEHAVIOR_THRESHOLDS["median_correct_margin"] for state in GATED_STATES)
        and all_four_accuracy >= BEHAVIOR_THRESHOLDS["all_four_accuracy"]
    )
    return {
        "split": split,
        "count": len(rows),
        "finite_fraction": finite_fraction,
        "state_metrics": state_metrics,
        "complete_item_count": len(complete),
        "all_four_accuracy": all_four_accuracy,
        "median_contextual_gain": median(context_gains),
        "posthoc_binding_interaction": {
            "definition": "0.5*((original_post-original_pre)-(swapped_post-swapped_pre)) on new-minus-old margins",
            "count": len(binding_interactions),
            "median": median(binding_interactions),
            "positive_fraction": sum(value > 0.0 for value in binding_interactions) / max(len(binding_interactions), 1),
            "used_for_gate": False,
        },
        "passed": passed,
    }


def finalize_behavior_command() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    model_results = {}
    all_decisions = []
    for model_name in MODELS:
        decisions = decisions_for_model(model_name)
        write_jsonl(OUT_ROOT / "analysis" / f"behavior_decisions.{model_name}.jsonl", decisions)
        all_decisions.extend(decisions)
        split_metrics = {
            split: behavior_metrics(decisions, split)
            for split in ("discovery", "confirmation", "natural_use")
        }
        authorized = all(split_metrics[split]["passed"] for split in BEHAVIOR_THRESHOLDS["required_splits"])
        model_results[model_name] = {
            "authorized_for_hidden_scan": authorized,
            "splits": split_metrics,
        }
    authorized_models = [name for name in MODELS if model_results[name]["authorized_for_hidden_scan"]]
    hidden_authorized = len(authorized_models) >= BEHAVIOR_THRESHOLDS["minimum_authorized_models"]
    result = {
        "schema_version": "phase1135_behavior_results.v1",
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "models": model_results,
        "authorized_models": authorized_models,
        "hidden_scan_authorized": hidden_authorized,
        "authorization_rule": "at least two models pass every frozen discovery and confirmation behavior cell",
        "evidence_scope": "exploratory_external_machine_consensus_not_human_gold",
        "human_annotation_eligible": False,
    }
    result["authorization_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/behavior_results.json", result)
    write_json(OUT_ROOT / "analysis/behavior_authorization.json", result)
    if not hidden_authorized:
        hard_stop = {
            "schema_version": "phase1135_causal_confirmation.v1",
            "phase": PHASE,
            "models": {
                name: {"confirmed": False, "reason": "behavior hard stop"}
                for name in MODELS
            },
            "confirmed_models": [],
            "cross_model_causal_event_confirmed": False,
            "component_search_authorized": False,
            "next_action": "stop hidden intervention; behavior object lacks two-model stability",
            "claim_boundary": "hidden state untested, not negative",
            "evidence_scope": "exploratory_external_machine_consensus_not_human_gold",
            "human_annotation_eligible": False,
        }
        hard_stop["confirmation_digest"] = digest(hard_stop)
        write_json(OUT_ROOT / "analysis/causal_confirmation.json", hard_stop)
    print(json.dumps({
        "phase": PHASE,
        "command": "finalize-behavior",
        "authorized_models": authorized_models,
        "hidden_scan_authorized": hidden_authorized,
        "authorization_digest": result["authorization_digest"],
    }, ensure_ascii=False), flush=True)


def sampled_depths(layer_count: int) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for fraction in DEPTH_FRACTIONS:
        depth = min(
            range(1, layer_count + 1),
            key=lambda value: (abs(value / layer_count - fraction), value),
        )
        if depth in seen:
            continue
        seen.add(depth)
        result.append({
            "depth": depth,
            "relative_depth": depth / layer_count,
            "requested_fraction": fraction,
        })
    return result


class ResidualCapture:
    def __init__(self, layers, depths: list[int]):
        self.layers = layers
        self.depths = depths
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []

    def _hook(self, depth: int):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(value, torch.Tensor):
                raise RuntimeError("capture not initialized")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)
            self.values[depth] = value[batch, positions, :].detach().float().cpu()
            return output
        return hook

    def register(self) -> None:
        for depth in self.depths:
            self.handles.append(self.layers[depth - 1].register_forward_hook(self._hook(depth)))

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}

    def validate(self) -> None:
        if set(self.values) != set(self.depths):
            raise RuntimeError(f"capture drift: {sorted(self.values)} != {self.depths}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = None
        self.values = {}


class ResidualReplacement:
    def __init__(self, layer, positions: torch.Tensor, replacements: torch.Tensor):
        self.layer = layer
        self.positions = positions
        self.replacements = replacements
        self.handle = None
        self.calls = 0

    def _hook(self, module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("replacement layer did not return a tensor")
        positions = self.positions.to(value.device)
        replacements = self.replacements.to(value.device, dtype=value.dtype)
        batch = torch.arange(value.shape[0], device=value.device)
        patched = value.clone()
        patched[batch, positions, :] = replacements
        self.calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def causal_cases(items: list[dict[str, Any]], split: str, selected_ids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    item_index = {str(row["item_id"]): row for row in items}
    cases: dict[str, dict[str, Any]] = {}
    selected_items = {}
    for item_id in selected_ids:
        item = item_index[item_id]
        selected_items[item_id] = item
        specs = item_state_specs(item, include_causal_nulls=True)
        for state in CAUSAL_STATES:
            case = make_case(item, state, specs[state])
            cases[str(case["case_id"])] = case
    return selected_items, cases


def prompt_token_row(tokenizer, case: dict[str, Any]) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(str(case["prompt"]), add_special_tokens=False)
    return {
        "case_id": str(case["case_id"]),
        "input_ids": prompt_ids,
        "prompt_length": len(prompt_ids),
    }


def capture_vectors(model, capture: ResidualCapture, rows: list[dict[str, Any]], pad_id: int, device: torch.device, batch_size: int) -> dict[str, dict[int, torch.Tensor]]:
    vectors: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            input_ids, attention_mask = pad_sequences(batch, pad_id, device)
            positions = torch.tensor([int(row["prompt_length"]) - 1 for row in batch], dtype=torch.long, device=device)
            capture.begin(positions)
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            capture.validate()
            for slot, row in enumerate(batch):
                for depth, values in capture.values.items():
                    vectors[str(row["case_id"])][depth] = values[slot].clone()
            del output, input_ids, attention_mask, positions
    return dict(vectors)


def clean_scores_for_cases(model, tokenizer, cases: dict[str, dict[str, Any]], pad_id: int, device: torch.device, batch_size: int) -> dict[str, dict[str, float]]:
    expanded = [
        tokenize_case(tokenizer, case, candidate_key)
        for case in cases.values()
        for candidate_key in ("old", "new")
    ]
    rows = score_rows(model, expanded, pad_id, device, batch_size)
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if not row["finite"]:
            grouped[str(row["case_id"])][str(row["candidate_key"])] = math.nan
        else:
            grouped[str(row["case_id"])][str(row["candidate_key"])] = float(row["logp_mean"])
    return dict(grouped)


def state_id(item_id: str, state: str) -> str:
    return f"{item_id}|{state}"


def build_patch_entries(selected_items: dict[str, dict[str, Any]], depth: int, vectors: dict[str, dict[int, torch.Tensor]]) -> list[dict[str, Any]]:
    item_ids = sorted(selected_items, key=lambda value: stable_key(PHASE, "causal-order", value))
    next_item = {item_id: item_ids[(index + 1) % len(item_ids)] for index, item_id in enumerate(item_ids)}
    entries = []

    def add(item_id: str, kind: str, target_state: str, source_item: str, source_state: str, reference_state: str, base_key: str, desired_key: str, panel: str) -> None:
        source_id = state_id(source_item, source_state)
        entries.append({
            "entry_id": f"{item_id}|d{depth}|{kind}|{target_state}|{source_item}|{source_state}",
            "item_id": item_id,
            "depth": depth,
            "patch_kind": kind,
            "panel": panel,
            "target_state": target_state,
            "target_case_id": state_id(item_id, target_state),
            "source_item_id": source_item,
            "source_state": source_state,
            "source_case_id": source_id,
            "reference_case_id": state_id(item_id, reference_state),
            "base_key": base_key,
            "desired_key": desired_key,
            "replacement": vectors[source_id][depth],
        })

    for item_id in item_ids:
        add(item_id, "main", "original_pre", item_id, "original_post", "original_post", "old", "new", "original")
        add(item_id, "main", "original_post", item_id, "original_pre", "original_pre", "new", "old", "original")
        add(item_id, "main", "swapped_pre", item_id, "swapped_post", "swapped_post", "new", "old", "swapped")
        add(item_id, "main", "swapped_post", item_id, "swapped_pre", "swapped_pre", "old", "new", "swapped")
        add(item_id, "same_answer_temporal_control", "original_pre", item_id, "original_pre_early", "original_post", "old", "new", "original")
        add(item_id, "same_answer_temporal_control", "original_post", item_id, "original_post_late", "original_pre", "new", "old", "original")
        donor = next_item[item_id]
        add(item_id, "shuffled_donor_control", "original_pre", donor, "original_post", "original_post", "old", "new", "original")
        add(item_id, "shuffled_donor_control", "original_post", donor, "original_pre", "original_pre", "new", "old", "original")
        add(item_id, "self_patch_audit", "original_pre", item_id, "original_pre", "original_post", "old", "new", "original")
    return entries


def score_patch_batch(model, layer, entries: list[dict[str, Any]], cases: dict[str, dict[str, Any]], tokenizer, pad_id: int, device: torch.device) -> list[dict[str, Any]]:
    expanded = []
    owners = []
    for entry in entries:
        case = cases[str(entry["target_case_id"])]
        for candidate_key in ("old", "new"):
            row = tokenize_case(tokenizer, case, candidate_key)
            expanded.append(row)
            owners.append(entry)
    input_ids, attention_mask = pad_sequences(expanded, pad_id, device)
    positions = torch.tensor([int(row["prompt_length"]) - 1 for row in expanded], dtype=torch.long, device=device)
    replacements = torch.stack([entry["replacement"] for entry in owners], dim=0)
    with torch.inference_mode():
        with ResidualReplacement(layer, positions, replacements) as patch:
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        if patch.calls != 1:
            raise RuntimeError(f"replacement hook called {patch.calls} times")
        scored = scores_from_logits(output.logits, expanded)
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row, entry, score in zip(expanded, owners, scored):
        grouped[str(entry["entry_id"])][str(row["candidate_key"])] = float(score["logp_mean"]) if score["finite"] else math.nan
    del output, input_ids, attention_mask, positions, replacements, scored
    return [
        {"entry": entry, "patched_scores": grouped[str(entry["entry_id"])]}
        for entry in entries
    ]


def causal_record(model_name: str, split: str, entry: dict[str, Any], patched: dict[str, float], clean: dict[str, dict[str, float]]) -> dict[str, Any]:
    target = clean[str(entry["target_case_id"])]
    reference = clean[str(entry["reference_case_id"])]
    base_key = str(entry["base_key"])
    desired_key = str(entry["desired_key"])
    values = [target[base_key], target[desired_key], reference[base_key], reference[desired_key], patched[base_key], patched[desired_key]]
    finite = all(math.isfinite(value) for value in values)
    base_margin = target[desired_key] - target[base_key]
    reference_margin = reference[desired_key] - reference[base_key]
    patched_margin = patched[desired_key] - patched[base_key]
    denominator = reference_margin - base_margin
    behavior_valid = finite and base_margin < 0.0 and reference_margin > 0.0 and denominator > EPSILON
    recovery = (patched_margin - base_margin) / denominator if behavior_valid else None
    return {
        "schema_version": "phase1135_residual_replacement_record.v1",
        "phase": PHASE,
        "model": model_name,
        "split": split,
        "entry_id": entry["entry_id"],
        "item_id": entry["item_id"],
        "depth": entry["depth"],
        "patch_kind": entry["patch_kind"],
        "panel": entry["panel"],
        "target_state": entry["target_state"],
        "source_item_id": entry["source_item_id"],
        "source_state": entry["source_state"],
        "base_key": base_key,
        "desired_key": desired_key,
        "base_margin": base_margin if math.isfinite(base_margin) else None,
        "reference_margin": reference_margin if math.isfinite(reference_margin) else None,
        "patched_margin": patched_margin if math.isfinite(patched_margin) else None,
        "denominator": denominator if math.isfinite(denominator) else None,
        "recovery": recovery if recovery is not None and math.isfinite(recovery) else None,
        "margin_change": patched_margin - base_margin if finite else None,
        "finite": finite,
        "behavior_valid": behavior_valid,
        "flip": finite and patched_margin > 0.0,
        "machine_validation_only": True,
    }


def causal_command(model_name: str, split: str) -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    authorization = read_json(OUT_ROOT / "analysis/behavior_authorization.json")
    output_root = OUT_ROOT / "causal" / split / model_name
    if not authorization["hidden_scan_authorized"] or model_name not in authorization["authorized_models"]:
        summary = {
            "schema_version": "phase1135_causal_scan_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "split": split,
            "skipped": True,
            "reason": "frozen behavior authorization denied",
            "protocol_digest": protocol["protocol_digest"],
        }
        summary["summary_digest"] = digest(summary)
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return
    selected_depths = None
    if split == "confirmation":
        selected_depths = read_json(OUT_ROOT / "analysis/causal_discovery_selection.json")
        model_selection = selected_depths["models"].get(model_name, {})
        if not model_selection.get("authorized_for_confirmation", False):
            summary = {
                "schema_version": "phase1135_causal_scan_summary.v1",
                "phase": PHASE,
                "model": model_name,
                "split": split,
                "skipped": True,
                "reason": "discovery causal gate denied",
                "protocol_digest": protocol["protocol_digest"],
            }
            summary["summary_digest"] = digest(summary)
            write_json(output_root / "summary.json", summary)
            print(json.dumps(summary), flush=True)
            return
    items = read_jsonl(SOURCE)
    selected_ids = list(protocol["causal_items"][split])
    selected_items, cases = causal_cases(items, split, selected_ids)
    started = time.time()
    model = None
    capture = None
    records = []
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError(f"FP16/no-quantization audit failed: {precision}")
        layers = get_layers(model)
        depth_rows = sampled_depths(len(layers))
        if split == "confirmation":
            chosen = int(selected_depths["models"][model_name]["selected_depth"])
            depth_rows = [row for row in depth_rows if int(row["depth"]) == chosen]
            if not depth_rows:
                raise RuntimeError(f"selected depth {chosen} not in frozen depth grid")
        depths = [int(row["depth"]) for row in depth_rows]
        capture = ResidualCapture(layers, depths)
        capture.register()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        prompt_rows = [prompt_token_row(tokenizer, case) for case in cases.values()]
        vectors = capture_vectors(model, capture, prompt_rows, int(pad_id), device, BATCH_SIZE[model_name])
        clean = clean_scores_for_cases(model, tokenizer, cases, int(pad_id), device, BATCH_SIZE[model_name])
        for depth in depths:
            entries = build_patch_entries(selected_items, depth, vectors)
            entries_per_batch = max(1, BATCH_SIZE[model_name] // 2)
            for start in range(0, len(entries), entries_per_batch):
                batch_entries = entries[start : start + entries_per_batch]
                patched_rows = score_patch_batch(
                    model,
                    layers[depth - 1],
                    batch_entries,
                    cases,
                    tokenizer,
                    int(pad_id),
                    device,
                )
                for patched_row in patched_rows:
                    records.append(causal_record(model_name, split, patched_row["entry"], patched_row["patched_scores"], clean))
            print(json.dumps({
                "phase": PHASE,
                "model": model_name,
                "split": split,
                "depth": depth,
                "records": len(records),
            }), flush=True)
        capture.close()
        capture = None
        elapsed = time.time() - started
        summary = {
            "schema_version": "phase1135_causal_scan_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "split": split,
            "skipped": False,
            "protocol_digest": protocol["protocol_digest"],
            "behavior_authorization_digest": authorization["authorization_digest"],
            "precision": precision,
            "placement": placement,
            "layer_count": len(layers),
            "sampled_depths": depth_rows,
            "item_count": len(selected_items),
            "record_count": len(records),
            "finite_fraction": sum(row["finite"] for row in records) / max(len(records), 1),
            "behavior_valid_fraction": sum(row["behavior_valid"] for row in records) / max(len(records), 1),
            "elapsed_seconds": elapsed,
            "component": "residual_stream_after_layer_at_answer_boundary",
            "intervention": "exact state replacement",
            "evidence_scope": "exploratory_machine_consensus",
        }
        summary["summary_digest"] = digest(summary)
        write_jsonl(output_root / "patch_records.jsonl", records)
        write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "phase": PHASE,
            "command": "causal",
            "model": model_name,
            "split": split,
            "records": len(records),
            "finite_fraction": summary["finite_fraction"],
            "elapsed_seconds": elapsed,
            "summary_digest": summary["summary_digest"],
        }, ensure_ascii=False), flush=True)
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


def depth_metrics(rows: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    selected = [row for row in rows if int(row["depth"]) == depth]
    valid = [row for row in selected if row["finite"] and row["behavior_valid"] and row["recovery"] is not None]
    kinds = {kind: [row for row in valid if row["patch_kind"] == kind] for kind in ("main", "same_answer_temporal_control", "shuffled_donor_control", "self_patch_audit")}
    main = kinds["main"]
    original = [row for row in main if row["panel"] == "original"]
    swapped = [row for row in main if row["panel"] == "swapped"]
    null_abs = median(abs(row["recovery"]) for row in kinds["same_answer_temporal_control"])
    shuffled_abs = median(abs(row["recovery"]) for row in kinds["shuffled_donor_control"])
    main_median = median(row["recovery"] for row in main)
    controls = [value for value in (null_abs, shuffled_abs) if value is not None]
    specificity = main_median - max(controls) if main_median is not None and controls else None
    self_changes = [abs(float(row["margin_change"])) for row in kinds["self_patch_audit"] if row["margin_change"] is not None]
    result = {
        "depth": depth,
        "record_count": len(selected),
        "valid_count": len(valid),
        "finite_fraction": sum(row["finite"] for row in selected) / max(len(selected), 1),
        "main_count": len(main),
        "main_median_recovery": main_median,
        "main_positive_fraction": sum(row["recovery"] > 0.0 for row in main) / max(len(main), 1),
        "main_flip_fraction": sum(row["flip"] for row in main) / max(len(main), 1),
        "original_median_recovery": median(row["recovery"] for row in original),
        "swapped_median_recovery": median(row["recovery"] for row in swapped),
        "same_answer_control_median_abs_recovery": null_abs,
        "shuffled_control_median_abs_recovery": shuffled_abs,
        "specificity_advantage": specificity,
        "self_patch_max_abs_margin_change": max(self_changes) if self_changes else None,
    }
    result["passed"] = (
        result["finite_fraction"] >= CAUSAL_THRESHOLDS["finite_fraction"]
        and result["main_median_recovery"] is not None
        and result["main_median_recovery"] >= CAUSAL_THRESHOLDS["main_median_recovery"]
        and result["main_positive_fraction"] >= CAUSAL_THRESHOLDS["main_positive_fraction"]
        and result["original_median_recovery"] is not None
        and result["original_median_recovery"] >= CAUSAL_THRESHOLDS["panel_median_recovery"]
        and result["swapped_median_recovery"] is not None
        and result["swapped_median_recovery"] >= CAUSAL_THRESHOLDS["panel_median_recovery"]
        and result["specificity_advantage"] is not None
        and result["specificity_advantage"] >= CAUSAL_THRESHOLDS["specificity_advantage"]
        and result["self_patch_max_abs_margin_change"] is not None
        and result["self_patch_max_abs_margin_change"] <= CAUSAL_THRESHOLDS["self_patch_max_abs_margin_change"]
    )
    return result


def finalize_discovery_command() -> None:
    behavior = read_json(OUT_ROOT / "analysis/behavior_authorization.json")
    model_results = {}
    for model_name in MODELS:
        summary_path = OUT_ROOT / "causal/discovery" / model_name / "summary.json"
        if model_name not in behavior["authorized_models"] or not summary_path.exists():
            model_results[model_name] = {"authorized_for_confirmation": False, "reason": "behavior denied or discovery absent"}
            continue
        summary = read_json(summary_path)
        if summary.get("skipped"):
            model_results[model_name] = {"authorized_for_confirmation": False, "reason": summary.get("reason")}
            continue
        rows = read_jsonl(OUT_ROOT / "causal/discovery" / model_name / "patch_records.jsonl")
        depths = sorted({int(row["depth"]) for row in rows})
        metrics = [depth_metrics(rows, depth) for depth in depths]
        passing = [row for row in metrics if row["passed"]]
        selected = max(passing, key=lambda row: (row["specificity_advantage"], row["main_median_recovery"], -row["depth"])) if passing else None
        model_results[model_name] = {
            "authorized_for_confirmation": selected is not None,
            "selected_depth": selected["depth"] if selected else None,
            "selected_metrics": selected,
            "depth_metrics": metrics,
        }
    result = {
        "schema_version": "phase1135_causal_discovery_selection.v1",
        "phase": PHASE,
        "models": model_results,
        "models_authorized_for_confirmation": [name for name in MODELS if model_results[name].get("authorized_for_confirmation")],
        "selection_rule": "best preregistered passing specificity advantage; confirmation depth frozen per model",
        "evidence_scope": "exploratory_machine_consensus",
    }
    result["selection_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/causal_discovery_selection.json", result)
    print(json.dumps({
        "phase": PHASE,
        "command": "finalize-discovery",
        "models_authorized_for_confirmation": result["models_authorized_for_confirmation"],
        "selected_depths": {name: row.get("selected_depth") for name, row in model_results.items()},
        "selection_digest": result["selection_digest"],
    }, ensure_ascii=False), flush=True)


def finalize_confirmation_command() -> None:
    selection = read_json(OUT_ROOT / "analysis/causal_discovery_selection.json")
    model_results = {}
    for model_name in MODELS:
        selected = selection["models"].get(model_name, {})
        summary_path = OUT_ROOT / "causal/confirmation" / model_name / "summary.json"
        if not selected.get("authorized_for_confirmation") or not summary_path.exists():
            model_results[model_name] = {"confirmed": False, "reason": "discovery denied or confirmation absent"}
            continue
        summary = read_json(summary_path)
        if summary.get("skipped"):
            model_results[model_name] = {"confirmed": False, "reason": summary.get("reason")}
            continue
        rows = read_jsonl(OUT_ROOT / "causal/confirmation" / model_name / "patch_records.jsonl")
        depth = int(selected["selected_depth"])
        metrics = depth_metrics(rows, depth)
        model_results[model_name] = {
            "confirmed": bool(metrics["passed"]),
            "selected_depth": depth,
            "metrics": metrics,
        }
    confirmed_models = [name for name in MODELS if model_results[name].get("confirmed")]
    component_search_authorized = len(confirmed_models) >= CAUSAL_THRESHOLDS["minimum_confirmed_models"]
    result = {
        "schema_version": "phase1135_causal_confirmation.v1",
        "phase": PHASE,
        "models": model_results,
        "confirmed_models": confirmed_models,
        "cross_model_causal_event_confirmed": component_search_authorized,
        "component_search_authorized": component_search_authorized,
        "next_action": (
            "run preregistered component decomposition on frozen confirmed event bands"
            if component_search_authorized
            else "stop intervention expansion; residual event did not clear the two-model independent gate"
        ),
        "claim_boundary": "residual-state event candidate only; no module, circuit, neuron, or complete mechanism claim",
        "evidence_scope": "exploratory_external_machine_consensus_not_human_gold",
        "human_annotation_eligible": False,
    }
    result["confirmation_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/causal_confirmation.json", result)
    print(json.dumps({
        "phase": PHASE,
        "command": "finalize-confirmation",
        "confirmed_models": confirmed_models,
        "component_search_authorized": component_search_authorized,
        "next_action": result["next_action"],
        "confirmation_digest": result["confirmation_digest"],
    }, ensure_ascii=False), flush=True)


def run_all_command() -> None:
    script = Path(__file__).resolve()

    def call(*args: str) -> None:
        subprocess.run([sys.executable, str(script), *args], cwd=str(ROOT), check=True)

    call("protocol")
    for model_name in MODELS:
        call("behavior", model_name)
    call("finalize-behavior")
    authorization = read_json(OUT_ROOT / "analysis/behavior_authorization.json")
    if authorization["hidden_scan_authorized"]:
        for model_name in authorization["authorized_models"]:
            call("causal", model_name, "discovery")
        call("finalize-discovery")
        selection = read_json(OUT_ROOT / "analysis/causal_discovery_selection.json")
        for model_name in selection["models_authorized_for_confirmation"]:
            call("causal", model_name, "confirmation")
        call("finalize-confirmation")
    else:
        result = {
            "schema_version": "phase1135_causal_confirmation.v1",
            "phase": PHASE,
            "models": {name: {"confirmed": False, "reason": "behavior hard stop"} for name in MODELS},
            "confirmed_models": [],
            "cross_model_causal_event_confirmed": False,
            "component_search_authorized": False,
            "next_action": "stop hidden intervention; behavior object lacks two-model stability",
            "claim_boundary": "hidden state untested, not negative",
            "evidence_scope": "exploratory_external_machine_consensus_not_human_gold",
            "human_annotation_eligible": False,
        }
        result["confirmation_digest"] = digest(result)
        write_json(OUT_ROOT / "analysis/causal_confirmation.json", result)
    audit_script = TEST_ROOT / "phase1135_temporal_binding_intervention_audit.py"
    subprocess.run([sys.executable, str(audit_script)], cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("protocol")
    behavior_parser = subparsers.add_parser("behavior")
    behavior_parser.add_argument("model", choices=MODELS)
    subparsers.add_parser("finalize-behavior")
    causal_parser = subparsers.add_parser("causal")
    causal_parser.add_argument("model", choices=MODELS)
    causal_parser.add_argument("split", choices=("discovery", "confirmation"))
    subparsers.add_parser("finalize-discovery")
    subparsers.add_parser("finalize-confirmation")
    subparsers.add_parser("run-all")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "behavior":
        behavior_command(args.model)
    elif args.command == "finalize-behavior":
        finalize_behavior_command()
    elif args.command == "causal":
        causal_command(args.model, args.split)
    elif args.command == "finalize-discovery":
        finalize_discovery_command()
    elif args.command == "finalize-confirmation":
        finalize_confirmation_command()
    elif args.command == "run-all":
        run_all_command()
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
