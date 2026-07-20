#!/usr/bin/env python3
"""Confirm only the score branch selected by Phase575 causal discovery."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
import phase575_routing_causal_discovery as discovery  # noqa: E402
import phase575_routing_causal_protocol as causal_protocol  # noqa: E402
import phase575_source_competition_protocol as protocol  # noqa: E402


MODEL = "qwen3"
SPLIT = "causal_confirmation"
OUT_DIR = protocol.OUT_DIR
DISCOVERY_DECISION = OUT_DIR / "phase575_routing_causal_discovery_decision.json"
DISCOVERY_SUMMARY = OUT_DIR / "phase575_qwen3_routing_causal_discovery_summary.json"
ROWS_PATH = OUT_DIR / "phase575_qwen3_routing_causal_confirmation_rows.jsonl.gz"
SUMMARY_PATH = OUT_DIR / "phase575_qwen3_routing_causal_confirmation_summary.json"
DECISION_PATH = OUT_DIR / "phase575_routing_causal_confirmation_decision.json"
CONTRACT_PATH = OUT_DIR / "phase575_qwen3_routing_causal_confirmation_contract.json"
CONDITIONS = (
    "natural_baseline",
    "score_relation_replace",
    "score_object_replace",
    "score_order_replace",
    "score_relation_weight_restore",
    "score_equalize",
    "score_equalize_restore",
    "value_group_swap_positive_control",
)


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


def paired_resample(
    metrics: dict[str, dict[str, Any]], count: int
) -> dict[str, Any]:
    relation = metrics["score_relation_replace"]["route_effect_by_world"]
    obj = metrics["score_object_replace"]["route_effect_by_world"]
    order = metrics["score_order_replace"]["route_effect_by_world"]
    contrasts = {
        world: relation[world] - max(obj[world], order[world]) for world in relation
    }
    observed = discovery.mean(list(contrasts.values()))
    at_least = 0
    for permutation in range(count):
        values = []
        for world, value in sorted(contrasts.items()):
            digest = hashlib.sha256(
                f"Phase575|confirmation|{permutation}|{world}".encode()
            ).digest()
            values.append(value if digest[0] & 1 else -value)
        at_least += int(discovery.mean(values) >= observed)
    return {
        "observed_relation_vs_max_control_contrast_mean": observed,
        "resample_count": count,
        "count_at_least_observed": at_least,
        "smoothed_tail_fraction": (at_least + 1) / (count + 1),
    }


def analyze(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
    gates = frozen["discovery_gate"]
    metrics = {
        condition: discovery.condition_metrics(rows, condition)
        for condition in CONDITIONS
    }
    relation = metrics["score_relation_replace"]
    obj = metrics["score_object_replace"]
    order = metrics["score_order_replace"]
    restore = metrics["score_relation_weight_restore"]
    relation_mean = relation["relation_route_effect_mean"]
    remaining = abs(restore["relation_route_effect_mean"]) / max(
        abs(relation_mean), 1e-12
    )
    resample = paired_resample(
        metrics, int(gates["pipeline_resample_count"])
    )
    physical_gate = (
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
        and remaining <= gates["mediation_remaining_fraction_maximum"]
        and resample["smoothed_tail_fraction"]
        <= gates["maximum_branch_smoothed_tail_fraction"]
    )
    behavior_gate = (
        relation["relation_logit_effect_positive_rate"]
        >= gates["behavior_relation_logit_effect_positive_rate_minimum"]
        and relation["relation_logit_effect_mean"]
        >= gates["behavior_relation_logit_effect_mean_minimum"]
    )
    summary = {
        "schema_version": "phase575_routing_causal_confirmation_summary.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete",
        "model": MODEL,
        "split": SPLIT,
        "selected_branch": "score",
        "world_count": len({row["base_case_id"] for row in rows}),
        "row_count": len(rows),
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
        "mediation_remaining_fraction": remaining,
        "paired_pipeline_resample": resample,
        "physical_routing_gate_pass": physical_gate,
        "behavior_candidate_margin_gate_pass": behavior_gate,
        "open_confirmation_pass": physical_gate and behavior_gate,
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "causal_splits_read": True,
        "sealed_split_read": False,
    }
    decision = {
        "schema_version": "phase575_routing_causal_confirmation_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete",
        "model": MODEL,
        "selected_branch": "score",
        "open_discovery_physical_and_behavior_pass": True,
        "open_confirmation_physical_routing_gate_pass": physical_gate,
        "open_confirmation_behavior_candidate_margin_gate_pass": behavior_gate,
        "full_short_generation_authorized": physical_gate and behavior_gate,
        "sealed_split_authorized": False,
        "candidate_status": "open_confirmation_candidate"
        if physical_gate and behavior_gate
        else "closed_on_open_confirmation",
        "full_short_generation_executed": False,
        "sealed_split_read": False,
        "summary_sha256": None,
    }
    return summary, decision


def run(restart: bool) -> Path:
    if restart:
        for path in (ROWS_PATH, SUMMARY_PATH, DECISION_PATH, CONTRACT_PATH):
            path.unlink(missing_ok=True)
    discovery_decision = read_json(DISCOVERY_DECISION)
    discovery_summary = read_json(DISCOVERY_SUMMARY)
    if discovery_decision["selected_causal_branch"] != "score":
        raise RuntimeError("Phase575 discovery did not select the score branch")
    if discovery_decision["summary_sha256"] != sha256_file(DISCOVERY_SUMMARY):
        raise RuntimeError("Phase575 discovery decision/summary hash drift")
    if not discovery_decision["confirmation_internal_state_authorized"]:
        raise RuntimeError("Phase575 causal confirmation is not authorized")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase575 routing causal confirmation requires CUDA")

    discovery.SPLIT = SPLIT
    worlds = discovery.load_worlds()
    contract = {
        "schema_version": "phase575_routing_causal_confirmation_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "split": SPLIT,
        "selected_branch": "score",
        "world_count": len(worlds),
        "conditions": list(CONDITIONS),
        "discovery_summary_sha256": sha256_file(DISCOVERY_SUMMARY),
        "discovery_decision_sha256": sha256_file(DISCOVERY_DECISION),
        "causal_protocol_sha256": sha256_file(causal_protocol.CAUSAL_PROTOCOL),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "torch_dtype_requested": "torch.bfloat16",
        "cuda_required": True,
        "causal_splits_read": True,
        "sealed_split_read": False,
    }
    write_json(CONTRACT_PATH, contract)
    loaded = None
    rows_out: list[dict[str, Any]] = []
    reconstruction_max = 0.0
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        if loaded.input_device.type != "cuda":
            raise RuntimeError("Phase575 causal confirmation model is not on CUDA")
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(
                f"Phase575 causal confirmation requires BF16, got {dtype}"
            )
        loaded.tokenizer.padding_side = "right"
        loaded.model.config._attn_implementation = "eager"
        layers = get_layers(loaded.model)
        if len(layers) != 36:
            raise RuntimeError(f"Phase575 Qwen3 layer drift: {len(layers)}")
        frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
        batch_size = int(frozen["execution"]["world_batch_size"])
        for start in range(0, len(worlds), batch_size):
            batch_worlds = worlds[start : start + batch_size]
            encoded_cpu, meta = discovery.prepare_batch(
                loaded.tokenizer, batch_worlds
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
                row["schema_version"] = (
                    "phase575_routing_causal_confirmation_row.v1"
                )
                rows_out.append(row)
            for condition in CONDITIONS:
                if condition == "natural_baseline":
                    continue
                outcomes = discovery.patched_forward(
                    loaded,
                    layers,
                    encoded_cpu,
                    meta,
                    captures,
                    condition,
                )
                for world_index, item in enumerate(meta):
                    row = discovery.causal_row(
                        item,
                        condition,
                        baseline[world_index],
                        outcomes[world_index],
                    )
                    row["schema_version"] = (
                        "phase575_routing_causal_confirmation_row.v1"
                    )
                    rows_out.append(row)
            del encoded_cpu, captures, baseline
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase575 "
                f"causal-confirmation {min(start + batch_size, len(worlds))}/"
                f"{len(worlds)}",
                flush=True,
            )

        discovery.write_jsonl(ROWS_PATH, rows_out)
        summary, decision = analyze(rows_out)
        summary.update(
            {
                "device_type": loaded.input_device.type,
                "torch_dtype": dtype,
                "runtime_seconds": time.monotonic() - started,
                "attention_weight_reconstruction_max_abs_error": (
                    reconstruction_max
                ),
                "attention_weight_reconstruction_pass": reconstruction_max <= 0.01,
                "rows_sha256": sha256_file(ROWS_PATH),
                "contract_sha256": sha256_file(CONTRACT_PATH),
            }
        )
        write_json(SUMMARY_PATH, summary)
        decision["summary_sha256"] = sha256_file(SUMMARY_PATH)
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
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.restart)


if __name__ == "__main__":
    main()
