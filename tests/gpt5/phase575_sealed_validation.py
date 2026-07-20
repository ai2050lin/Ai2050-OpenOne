#!/usr/bin/env python3
"""Open the Phase575 seal once and run the frozen end-to-end validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase573_natural_transition_behavior import (  # noqa: E402
    balanced_worlds,
    generate_batch,
    stable_expected,
)
import phase575_full_generation as full_generation  # noqa: E402
import phase575_routing_causal_confirmation as confirmation  # noqa: E402
import phase575_routing_causal_discovery as discovery  # noqa: E402
import phase575_sealed_validation_protocol as sealed_protocol  # noqa: E402
import phase575_source_competition_protocol as protocol  # noqa: E402


MODEL = "qwen3"
SPLIT = "sealed"
OUT_DIR = protocol.OUT_DIR
RECEIPT_PATH = OUT_DIR / "phase575_sealed_execution_receipt.json"
BEHAVIOR_ROWS_PATH = OUT_DIR / "phase575_qwen3_sealed_behavior_rows.jsonl.gz"
CAUSAL_ROWS_PATH = OUT_DIR / "phase575_qwen3_sealed_causal_rows.jsonl.gz"
GENERATION_ROWS_PATH = OUT_DIR / "phase575_qwen3_sealed_generation_rows.jsonl.gz"
SUMMARY_PATH = OUT_DIR / "phase575_qwen3_sealed_validation_summary.json"
DECISION_PATH = OUT_DIR / "phase575_sealed_validation_decision.json"
CONTRACT_PATH = OUT_DIR / "phase575_qwen3_sealed_validation_contract.json"
CONTROL_VARIANTS = ("object_swap", "relation_object_swap", "order_swap")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_variants(
    by_repeat: dict[tuple[str, str], dict[str, Any]],
    base_id: str,
    variants: tuple[str, ...],
) -> bool:
    return all(
        stable_expected(by_repeat, f"{base_id}_{variant}")
        for variant in variants
    )


def run_behavior_rows(
    loaded: Any,
    rows: list[dict[str, Any]],
    output: list[dict[str, Any]],
    stage: str,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    for repeat in ("noop1", "noop2"):
        for start in range(0, len(rows), batch_size):
            output.extend(
                generate_batch(
                    loaded,
                    MODEL,
                    rows[start : start + batch_size],
                    repeat,
                    max_new_tokens,
                )
            )
        print(
            f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase575 sealed "
            f"{stage}/{repeat} {len(rows)}/{len(rows)}",
            flush=True,
        )


def qualify_behavior(
    loaded: Any,
    sealed_rows: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    loaded.tokenizer.padding_side = "left"
    if loaded.tokenizer.padding_side != "left":
        raise RuntimeError("Phase575 sealed behavior requires left padding")
    behavior = frozen["behavior_qualification"]
    by_world_variant = {
        (row["base_case_id"], row["variant"]): row for row in sealed_rows
    }
    base_rows = {
        row["base_case_id"]: row for row in sealed_rows if row["variant"] == "base"
    }
    output: list[dict[str, Any]] = []
    relation_rows = sorted(
        [
            row
            for row in sealed_rows
            if row["variant"] in ("base", "relation_swap")
        ],
        key=lambda row: row["case_id"],
    )
    run_behavior_rows(
        loaded,
        relation_rows,
        output,
        "relation",
        int(behavior["batch_size"]),
        int(behavior["max_new_tokens"]),
    )
    by_repeat = {
        (row["case_id"], row["execution_repeat"]): row for row in output
    }
    world_ids = sorted({row["base_case_id"] for row in relation_rows})
    relation_eligible = [
        base_id
        for base_id in world_ids
        if stable_variants(by_repeat, base_id, ("base", "relation_swap"))
    ]
    if len(relation_eligible) < int(behavior["minimum_relation_qualified"]):
        raise RuntimeError(
            f"Phase575 sealed relation behavior gate failed: {len(relation_eligible)}"
        )
    control_ids = balanced_worlds(
        base_rows, relation_eligible, int(behavior["control_screen_cap"])
    )
    controls = sorted(
        [
            by_world_variant[(base_id, variant)]
            for base_id in control_ids
            for variant in CONTROL_VARIANTS
        ],
        key=lambda row: row["case_id"],
    )
    run_behavior_rows(
        loaded,
        controls,
        output,
        "controls",
        int(behavior["batch_size"]),
        int(behavior["max_new_tokens"]),
    )
    by_repeat = {
        (row["case_id"], row["execution_repeat"]): row for row in output
    }
    five_variant = [
        base_id
        for base_id in control_ids
        if stable_variants(by_repeat, base_id, CONTROL_VARIANTS)
    ]
    selected = balanced_worlds(
        base_rows,
        five_variant,
        int(behavior["selected_five_variant_world_count"]),
    )
    if len(selected) != int(behavior["selected_five_variant_world_count"]):
        raise RuntimeError(
            f"Phase575 sealed five-variant behavior gate failed: {len(selected)}"
        )
    repeats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        repeats[row["case_id"]].append(row)
    exact_mismatch = sum(
        len(rows) != 2
        or rows[0]["normalized_generated"] != rows[1]["normalized_generated"]
        for rows in repeats.values()
    )
    semantic_mismatch = sum(
        len(rows) != 2
        or rows[0]["semantic_event"] != rows[1]["semantic_event"]
        for rows in repeats.values()
    )
    diagnostics = {
        "relation_qualified_world_count": len(relation_eligible),
        "control_screen_world_count": len(control_ids),
        "five_variant_qualified_world_count": len(five_variant),
        "selected_world_count": len(selected),
        "executed_behavior_row_count": len(output),
        "executed_unique_case_count": len(repeats),
        "noop_exact_text_mismatch_count": exact_mismatch,
        "noop_semantic_event_mismatch_count": semantic_mismatch,
        "behavior_gate_pass": exact_mismatch == 0 and semantic_mismatch == 0,
    }
    return output, selected, diagnostics


def selected_worlds(
    sealed_rows: list[dict[str, Any]], selected: list[str]
) -> list[list[dict[str, Any]]]:
    selected_set = set(selected)
    bank: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sealed_rows:
        if row["base_case_id"] in selected_set:
            bank[row["base_case_id"]][row["variant"]] = row
    worlds = []
    for base_id in selected:
        variants = bank.get(base_id, {})
        if set(variants) != set(protocol.VARIANTS):
            raise RuntimeError(f"Phase575 incomplete sealed world: {base_id}")
        worlds.append([variants[variant] for variant in protocol.VARIANTS])
    return worlds


def sealed_causal_analysis(
    rows: list[dict[str, Any]], frozen: dict[str, Any]
) -> dict[str, Any]:
    metrics = {
        condition: discovery.condition_metrics(rows, condition)
        for condition in frozen["conditions"]
    }
    relation = metrics["score_relation_replace"]
    obj = metrics["score_object_replace"]
    order = metrics["score_order_replace"]
    restore = metrics["score_relation_weight_restore"]
    gates = frozen["causal_gates"]
    relation_mean = relation["relation_route_effect_mean"]
    resample = confirmation.paired_resample(
        metrics, int(gates["pipeline_resample_count"])
    )
    causal_gate = (
        relation["relation_route_effect_positive_rate"]
        >= gates["relation_route_effect_positive_rate_minimum"]
        and relation_mean >= gates["relation_route_effect_mean_minimum"]
        and relation_mean - obj["relation_route_effect_mean"]
        >= gates["relation_vs_object_effect_gap_minimum"]
        and relation_mean - order["relation_route_effect_mean"]
        >= gates["relation_vs_order_effect_gap_minimum"]
        and abs(restore["relation_route_effect_mean"])
        <= gates["restore_route_maximum_absolute_delta"]
        and restore["maximum_candidate_logit_delta"]
        <= gates["restore_candidate_logit_maximum_absolute_delta"]
        and relation["relation_logit_effect_positive_rate"]
        >= gates["relation_logit_effect_positive_rate_minimum"]
        and relation["relation_logit_effect_mean"]
        >= gates["relation_logit_effect_mean_minimum"]
        and resample["smoothed_tail_fraction"]
        <= gates["smoothed_tail_fraction_maximum"]
    )
    return {
        "condition_metrics": {
            condition: {
                key: value
                for key, value in values.items()
                if key != "route_effect_by_world"
            }
            for condition, values in metrics.items()
        },
        "relation_vs_object_gap": relation_mean
        - obj["relation_route_effect_mean"],
        "relation_vs_order_gap": relation_mean
        - order["relation_route_effect_mean"],
        "paired_pipeline_resample": resample,
        "causal_gate_pass": causal_gate,
    }


def run_causal(
    loaded: Any,
    layers: list[Any],
    worlds: list[list[dict[str, Any]]],
    frozen: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    discovery.SPLIT = SPLIT
    output: list[dict[str, Any]] = []
    reconstruction_max = 0.0
    batch_size = int(frozen["causal_execution"]["world_batch_size"])
    for start in range(0, len(worlds), batch_size):
        batch_worlds = worlds[start : start + batch_size]
        encoded_cpu, meta = discovery.prepare_batch(
            loaded.tokenizer, batch_worlds, padding_side="right"
        )
        captures, baseline, error = discovery.natural_forward(
            loaded, layers, encoded_cpu, meta
        )
        reconstruction_max = max(reconstruction_max, error)
        for world_index, item in enumerate(meta):
            row = discovery.causal_row(
                item,
                "natural_baseline",
                baseline[world_index],
                baseline[world_index],
            )
            row["schema_version"] = "phase575_sealed_causal_row.v1"
            row["sealed"] = True
            output.append(row)
        for condition in frozen["conditions"]:
            if condition == "natural_baseline":
                continue
            outcomes = discovery.patched_forward(
                loaded, layers, encoded_cpu, meta, captures, condition
            )
            for world_index, item in enumerate(meta):
                row = discovery.causal_row(
                    item,
                    condition,
                    baseline[world_index],
                    outcomes[world_index],
                )
                row["schema_version"] = "phase575_sealed_causal_row.v1"
                row["sealed"] = True
                output.append(row)
        del encoded_cpu, captures, baseline
        print(
            f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase575 sealed-causal "
            f"{min(start + batch_size, len(worlds))}/{len(worlds)}",
            flush=True,
        )
    analysis = sealed_causal_analysis(output, frozen)
    analysis["attention_weight_reconstruction_max_abs_error"] = reconstruction_max
    analysis["attention_weight_reconstruction_pass"] = reconstruction_max <= 0.01
    analysis["causal_gate_pass"] = bool(
        analysis["causal_gate_pass"]
        and analysis["attention_weight_reconstruction_pass"]
    )
    return output, analysis


def run_full_generation(
    loaded: Any,
    layers: list[Any],
    worlds: list[list[dict[str, Any]]],
    frozen: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation = frozen["full_generation"]
    output: list[dict[str, Any]] = []
    batch_size = int(frozen["causal_execution"]["world_batch_size"])
    for start in range(0, len(worlds), batch_size):
        batch_worlds = worlds[start : start + batch_size]
        encoded_cpu, meta = discovery.prepare_batch(
            loaded.tokenizer, batch_worlds, padding_side="left"
        )
        captures, _, _ = discovery.natural_forward(
            loaded, layers, encoded_cpu, meta
        )
        for condition in frozen["conditions"]:
            for repeat in generation["execution_repeats"]:
                output.extend(
                    full_generation.generate_condition(
                        loaded,
                        layers,
                        encoded_cpu,
                        meta,
                        captures,
                        batch_worlds,
                        condition,
                        repeat,
                        int(generation["max_new_tokens"]),
                        evidence_split=SPLIT,
                        sealed=True,
                    )
                )
        del encoded_cpu, captures
        print(
            f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase575 sealed-generation "
            f"{min(start + batch_size, len(worlds))}/{len(worlds)}",
            flush=True,
        )
    analysis, _ = full_generation.analyze(output)
    analysis["schema_version"] = "phase575_sealed_generation_summary.v1"
    analysis["split"] = SPLIT
    analysis["sealed_split_read"] = True
    return output, analysis


def run() -> Path:
    if SUMMARY_PATH.exists():
        print(SUMMARY_PATH.read_text(encoding="utf-8"))
        return SUMMARY_PATH
    frozen = read_json(sealed_protocol.SEALED_PROTOCOL)
    commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
    if frozen["committed_sealed_cases_sha256"] != commitment[
        "sealed_cases_sha256"
    ]:
        raise RuntimeError("Phase575 sealed commitment drift before opening")
    receipt = {
        "schema_version": "phase575_sealed_execution_receipt.v1",
        "phase_id": protocol.PHASE,
        "opened_at": now(),
        "model": MODEL,
        "one_shot": True,
        "sealed_protocol_sha256": sha256_file(sealed_protocol.SEALED_PROTOCOL),
        "sealed_commitment_sha256": sha256_file(protocol.SEALED_COMMITMENT_PATH),
        "committed_sealed_cases_sha256": commitment["sealed_cases_sha256"],
        "sealed_split_read": True,
    }
    if RECEIPT_PATH.exists():
        existing = read_json(RECEIPT_PATH)
        for key, value in receipt.items():
            if key != "opened_at" and existing[key] != value:
                raise RuntimeError(f"Phase575 sealed receipt drift: {key}")
    else:
        write_json(RECEIPT_PATH, receipt)

    if sha256_file(protocol.SEALED_CASES_PATH) != commitment[
        "sealed_cases_sha256"
    ]:
        raise RuntimeError("Phase575 sealed cases no longer match commitment")
    sealed_rows = list(discovery.iter_jsonl(protocol.SEALED_CASES_PATH))
    expected = int(frozen["behavior_qualification"]["candidate_world_count"]) * len(
        protocol.VARIANTS
    )
    if len(sealed_rows) != expected or not all(row["sealed"] for row in sealed_rows):
        raise RuntimeError(f"Phase575 sealed denominator drift: {len(sealed_rows)}")
    contract = {
        "schema_version": "phase575_sealed_validation_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "sealed_case_count": len(sealed_rows),
        "sealed_protocol_sha256": sha256_file(sealed_protocol.SEALED_PROTOCOL),
        "receipt_sha256": sha256_file(RECEIPT_PATH),
        "sealed_cases_sha256": sha256_file(protocol.SEALED_CASES_PATH),
        "torch_dtype_requested": "torch.bfloat16",
        "cuda_required": True,
        "sealed_split_read": True,
    }
    write_json(CONTRACT_PATH, contract)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase575 sealed validation requires CUDA")

    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        if loaded.input_device.type != "cuda":
            raise RuntimeError("Phase575 sealed model is not on CUDA")
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase575 sealed validation requires BF16, got {dtype}")
        loaded.model.config._attn_implementation = "eager"
        layers = get_layers(loaded.model)
        if len(layers) != 36:
            raise RuntimeError(f"Phase575 Qwen3 layer drift: {len(layers)}")

        behavior_rows, selected, behavior_analysis = qualify_behavior(
            loaded, sealed_rows, frozen
        )
        discovery.write_jsonl(BEHAVIOR_ROWS_PATH, behavior_rows)
        worlds = selected_worlds(sealed_rows, selected)
        causal_rows, causal_analysis = run_causal(
            loaded, layers, worlds, frozen
        )
        discovery.write_jsonl(CAUSAL_ROWS_PATH, causal_rows)
        generation_rows: list[dict[str, Any]] = []
        generation_analysis: dict[str, Any] = {
            "full_generation_gate_pass": False,
            "status": "not_run_because_sealed_causal_gate_failed",
        }
        if causal_analysis["causal_gate_pass"]:
            generation_rows, generation_analysis = run_full_generation(
                loaded, layers, worlds, frozen
            )
            discovery.write_jsonl(GENERATION_ROWS_PATH, generation_rows)
        sealed_pass = bool(
            behavior_analysis["behavior_gate_pass"]
            and causal_analysis["causal_gate_pass"]
            and generation_analysis["full_generation_gate_pass"]
        )
        summary = {
            "schema_version": "phase575_sealed_validation_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "split": SPLIT,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "sealed_candidate_world_count": 1024,
            "selected_world_count": len(selected),
            "behavior_analysis": behavior_analysis,
            "causal_analysis": causal_analysis,
            "generation_analysis": generation_analysis,
            "sealed_validation_pass": sealed_pass,
            "runtime_seconds": time.monotonic() - started,
            "behavior_rows_sha256": sha256_file(BEHAVIOR_ROWS_PATH),
            "causal_rows_sha256": sha256_file(CAUSAL_ROWS_PATH),
            "generation_rows_sha256": sha256_file(GENERATION_ROWS_PATH)
            if GENERATION_ROWS_PATH.exists()
            else None,
            "receipt_sha256": sha256_file(RECEIPT_PATH),
            "output_embedding_direction_used": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": True,
        }
        write_json(SUMMARY_PATH, summary)
        decision = {
            "schema_version": "phase575_sealed_validation_decision.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "selected_coordinate": frozen["selected_coordinate"],
            "sealed_validation_pass": sealed_pass,
            "candidate_status": "sealed_local_causal_replication"
            if sealed_pass
            else "closed_on_one_shot_seal",
            "strict_mechanism_closure_claimed": False,
            "cross_model_mechanism_claimed": False,
            "broad_language_encoding_claimed": False,
            "phase575_seal_may_be_reopened": False,
            "sealed_split_read": True,
            "summary_sha256": sha256_file(SUMMARY_PATH),
        }
        write_json(DECISION_PATH, decision)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
        return SUMMARY_PATH
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
