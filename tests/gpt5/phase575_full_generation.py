#!/usr/bin/env python3
"""Run frozen full short generation for the confirmed Phase575 score branch."""

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
from phase569_relation_competition_behavior import classify  # noqa: E402
import phase575_full_generation_protocol as generation_protocol  # noqa: E402
import phase575_routing_causal_discovery as discovery  # noqa: E402
import phase575_source_competition_protocol as protocol  # noqa: E402


MODEL = "qwen3"
SPLIT = "causal_confirmation"
OUT_DIR = protocol.OUT_DIR
PROTOCOL_PATH = generation_protocol.GENERATION_PROTOCOL
CONFIRMATION_DECISION = OUT_DIR / "phase575_routing_causal_confirmation_decision.json"
ROWS_PATH = OUT_DIR / "phase575_qwen3_full_generation_rows.jsonl.gz"
SUMMARY_PATH = OUT_DIR / "phase575_qwen3_full_generation_summary.json"
DECISION_PATH = OUT_DIR / "phase575_full_generation_decision.json"
CONTRACT_PATH = OUT_DIR / "phase575_qwen3_full_generation_contract.json"


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


def generation_hook(
    layers: list[Any],
    meta: list[dict[str, Any]],
    captures: dict[int, dict[str, torch.Tensor]],
    condition: str,
) -> Any | None:
    if condition in ("natural_baseline", "score_relation_weight_restore"):
        return None
    donor_variant = {
        "score_relation_replace": "relation_swap",
        "score_object_replace": "object_swap",
        "score_order_replace": "order_swap",
    }[condition]

    def hook(
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        position_embeddings = kwargs.get("position_embeddings")
        mask = kwargs.get("attention_mask")
        if hidden is None or position_embeddings is None or not isinstance(output, tuple):
            raise RuntimeError("Phase575 generation hook requires eager attention")
        query, key, value = discovery.projected_states(
            module, hidden, position_embeddings
        )
        cos, sin = position_embeddings
        raw_scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
        primary = output[0].clone()
        for world_index, item in enumerate(meta):
            batch_index = int(item["indices"]["base"])
            position = item["positions"]["base"]
            receiver = int(position["answer_boundary"])
            if receiver >= hidden.shape[1]:
                raise RuntimeError("Phase575 generation receiver left current sequence")
            donor_capture = discovery.capture_index(world_index, donor_variant)
            donor_pre = captures[24]["q_pre_answer"][donor_capture]
            donor_query = (
                donor_pre * cos[batch_index, receiver]
                + discovery.rotate_half(donor_pre) * sin[batch_index, receiver]
            )
            donor_scores = torch.einsum(
                "hd,hsd->hs", donor_query, key[batch_index]
            ) * module.scaling
            score_row = raw_scores[batch_index, :, receiver, :].clone()
            discovery.copy_group(
                score_row,
                donor_scores,
                position["anchor_selected"],
                position["anchor_selected"],
            )
            discovery.copy_group(
                score_row,
                donor_scores,
                position["anchor_other"],
                position["anchor_other"],
            )
            mask_row = discovery.attention_mask_row(mask, batch_index, receiver)
            weight_row = discovery.normalized_weights(score_row, mask_row)
            head_output = torch.einsum(
                "hs,hsd->hd", weight_row, value[batch_index]
            )
            primary[batch_index, receiver, :] = module.o_proj(
                head_output.reshape(1, -1)
            ).squeeze(0)
        return (primary, *output[1:])

    return layers[24].self_attn.register_forward_hook(hook, with_kwargs=True)


def generate_condition(
    loaded: Any,
    layers: list[Any],
    encoded_cpu: dict[str, torch.Tensor],
    meta: list[dict[str, Any]],
    captures: dict[int, dict[str, torch.Tensor]],
    worlds: list[list[dict[str, Any]]],
    condition: str,
    repeat: str,
    max_new_tokens: int,
    evidence_split: str = SPLIT,
    sealed: bool = False,
) -> list[dict[str, Any]]:
    input_ids = encoded_cpu["input_ids"].to(loaded.input_device)
    attention_mask = encoded_cpu["attention_mask"].to(loaded.input_device)
    input_width = int(input_ids.shape[1])
    handle = generation_hook(layers, meta, captures, condition)
    try:
        with torch.inference_mode():
            sequences = loaded.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()
    output = []
    for world_index, item in enumerate(meta):
        batch_index = int(item["indices"]["base"])
        generated_ids = sequences[batch_index, input_width:].detach().cpu().tolist()
        generated = loaded.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        base_row = worlds[world_index][0]
        classified = classify(base_row, generated)
        output.append(
            {
                "schema_version": "phase575_full_generation_row.v1",
                "phase_id": protocol.PHASE,
                "created_at": now(),
                "model": MODEL,
                "split": evidence_split,
                "base_case_id": item["base_case_id"],
                "condition": condition,
                "execution_repeat": repeat,
                "base_target": item["targets"]["base"],
                "relation_target": item["targets"]["relation_swap"],
                "object_target": item["targets"]["object_swap"],
                **classified,
                "relation_target_selected": classified["selected_candidate"]
                == item["targets"]["relation_swap"],
                "base_target_selected": classified["selected_candidate"]
                == item["targets"]["base"],
                "generated_token_ids": [int(token) for token in generated_ids],
                "full_short_generation": True,
                "use_cache": False,
                "output_embedding_direction_used": False,
                "sealed": sealed,
            }
        )
    del sequences, input_ids, attention_mask
    return output


def condition_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_world[row["base_case_id"]].append(row)
    exact_mismatch = sum(
        len(values) != 2
        or values[0]["normalized_generated"] != values[1]["normalized_generated"]
        for values in by_world.values()
    )
    semantic_mismatch = sum(
        len(values) != 2
        or values[0]["semantic_event"] != values[1]["semantic_event"]
        for values in by_world.values()
    )
    relation_by_world = {
        world: sum(row["relation_target_selected"] for row in values) / len(values)
        for world, values in by_world.items()
    }
    base_by_world = {
        world: sum(row["base_target_selected"] for row in values) / len(values)
        for world, values in by_world.items()
    }
    return {
        "world_count": len(by_world),
        "row_count": len(selected),
        "relation_target_rate": discovery.mean(list(relation_by_world.values())),
        "base_target_rate": discovery.mean(list(base_by_world.values())),
        "repeat_exact_text_mismatch_count": exact_mismatch,
        "repeat_semantic_event_mismatch_count": semantic_mismatch,
        "relation_target_rate_by_world": relation_by_world,
    }


def paired_resample(
    metrics: dict[str, dict[str, Any]], count: int
) -> dict[str, Any]:
    relation = metrics["score_relation_replace"]["relation_target_rate_by_world"]
    obj = metrics["score_object_replace"]["relation_target_rate_by_world"]
    order = metrics["score_order_replace"]["relation_target_rate_by_world"]
    contrasts = {
        world: relation[world] - max(obj[world], order[world]) for world in relation
    }
    observed = discovery.mean(list(contrasts.values()))
    at_least = 0
    for permutation in range(count):
        values = []
        for world, value in sorted(contrasts.items()):
            digest = hashlib.sha256(
                f"Phase575|full-generation|{permutation}|{world}".encode()
            ).digest()
            values.append(value if digest[0] & 1 else -value)
        at_least += int(discovery.mean(values) >= observed)
    return {
        "observed_relation_vs_max_control_rate_contrast": observed,
        "resample_count": count,
        "count_at_least_observed": at_least,
        "smoothed_tail_fraction": (at_least + 1) / (count + 1),
    }


def analyze(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = read_json(PROTOCOL_PATH)
    gates = frozen["gates"]
    metrics = {
        condition: condition_metrics(rows, condition)
        for condition in frozen["conditions"]
    }
    baseline = metrics["natural_baseline"]
    relation = metrics["score_relation_replace"]
    obj = metrics["score_object_replace"]
    order = metrics["score_order_replace"]
    restore_rows = [
        row for row in rows if row["condition"] == "score_relation_weight_restore"
    ]
    baseline_rows = {
        (row["base_case_id"], row["execution_repeat"]): row
        for row in rows
        if row["condition"] == "natural_baseline"
    }
    restore_exact_mismatch = sum(
        row["normalized_generated"]
        != baseline_rows[(row["base_case_id"], row["execution_repeat"])][
            "normalized_generated"
        ]
        for row in restore_rows
    )
    restore_semantic_mismatch = sum(
        row["semantic_event"]
        != baseline_rows[(row["base_case_id"], row["execution_repeat"])][
            "semantic_event"
        ]
        for row in restore_rows
    )
    resample = paired_resample(metrics, int(gates["pipeline_resample_count"]))
    repeat_pass = all(
        values["repeat_exact_text_mismatch_count"]
        <= gates["repeat_exact_text_mismatch_maximum_each_condition"]
        and values["repeat_semantic_event_mismatch_count"]
        <= gates["repeat_semantic_event_mismatch_maximum_each_condition"]
        for values in metrics.values()
    )
    full_gate = (
        baseline["base_target_rate"] >= gates["natural_base_target_rate_minimum"]
        and relation["relation_target_rate"]
        >= gates["relation_donor_target_rate_minimum"]
        and relation["relation_target_rate"] - baseline["relation_target_rate"]
        >= gates["relation_donor_target_rate_gain_minimum"]
        and relation["relation_target_rate"] - obj["relation_target_rate"]
        >= gates["relation_vs_object_target_rate_gap_minimum"]
        and relation["relation_target_rate"] - order["relation_target_rate"]
        >= gates["relation_vs_order_target_rate_gap_minimum"]
        and restore_exact_mismatch <= gates["restore_exact_text_mismatch_maximum"]
        and restore_semantic_mismatch
        <= gates["restore_semantic_event_mismatch_maximum"]
        and repeat_pass
        and resample["smoothed_tail_fraction"]
        <= gates["smoothed_tail_fraction_maximum"]
    )
    summary = {
        "schema_version": "phase575_full_generation_summary.v1",
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
                if key != "relation_target_rate_by_world"
            }
            for condition, values in metrics.items()
        },
        "relation_target_rate_gain_over_natural": (
            relation["relation_target_rate"] - baseline["relation_target_rate"]
        ),
        "relation_vs_object_target_rate_gap": (
            relation["relation_target_rate"] - obj["relation_target_rate"]
        ),
        "relation_vs_order_target_rate_gap": (
            relation["relation_target_rate"] - order["relation_target_rate"]
        ),
        "restore_exact_text_mismatch_count": restore_exact_mismatch,
        "restore_semantic_event_mismatch_count": restore_semantic_mismatch,
        "repeat_stability_gate_pass": repeat_pass,
        "paired_pipeline_resample": resample,
        "full_generation_gate_pass": full_gate,
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "causal_splits_read": True,
        "sealed_split_read": False,
    }
    decision = {
        "schema_version": "phase575_full_generation_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete",
        "model": MODEL,
        "selected_branch": "score",
        "open_discovery_pass": True,
        "open_confirmation_pass": True,
        "full_generation_gate_pass": full_gate,
        "sealed_validation_authorized": full_gate,
        "candidate_status": "open_full_generation_candidate"
        if full_gate
        else "closed_on_full_generation",
        "sealed_split_read": False,
        "summary_sha256": None,
    }
    return summary, decision


def run(restart: bool) -> Path:
    if restart:
        for path in (ROWS_PATH, SUMMARY_PATH, DECISION_PATH, CONTRACT_PATH):
            path.unlink(missing_ok=True)
    frozen = read_json(PROTOCOL_PATH)
    confirmation = read_json(CONFIRMATION_DECISION)
    if not confirmation["full_short_generation_authorized"]:
        raise RuntimeError("Phase575 full generation is not authorized")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase575 full generation requires CUDA")
    discovery.SPLIT = SPLIT
    worlds = discovery.load_worlds()
    contract = {
        "schema_version": "phase575_full_generation_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "split": SPLIT,
        "world_count": len(worlds),
        "conditions": frozen["conditions"],
        "execution_repeats": frozen["execution_repeats"],
        "generation_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "confirmation_decision_sha256": sha256_file(CONFIRMATION_DECISION),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "torch_dtype_requested": "torch.bfloat16",
        "cuda_required": True,
        "causal_splits_read": True,
        "sealed_split_read": False,
    }
    write_json(CONTRACT_PATH, contract)
    loaded = None
    rows_out: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        if loaded.input_device.type != "cuda":
            raise RuntimeError("Phase575 full generation model is not on CUDA")
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase575 full generation requires BF16, got {dtype}")
        loaded.model.config._attn_implementation = "eager"
        layers = get_layers(loaded.model)
        if len(layers) != 36:
            raise RuntimeError(f"Phase575 Qwen3 layer drift: {len(layers)}")
        batch_size = int(frozen["world_batch_size"])
        for start in range(0, len(worlds), batch_size):
            batch_worlds = worlds[start : start + batch_size]
            encoded_cpu, meta = discovery.prepare_batch(
                loaded.tokenizer, batch_worlds, padding_side="left"
            )
            captures, _, _ = discovery.natural_forward(
                loaded, layers, encoded_cpu, meta
            )
            for condition in frozen["conditions"]:
                for repeat in frozen["execution_repeats"]:
                    rows_out.extend(
                        generate_condition(
                            loaded,
                            layers,
                            encoded_cpu,
                            meta,
                            captures,
                            batch_worlds,
                            condition,
                            repeat,
                            int(frozen["max_new_tokens"]),
                        )
                    )
            del encoded_cpu, captures
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase575 full-generation "
                f"{min(start + batch_size, len(worlds))}/{len(worlds)}",
                flush=True,
            )
        discovery.write_jsonl(ROWS_PATH, rows_out)
        summary, decision = analyze(rows_out)
        summary.update(
            {
                "device_type": loaded.input_device.type,
                "torch_dtype": dtype,
                "runtime_seconds": time.monotonic() - started,
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
