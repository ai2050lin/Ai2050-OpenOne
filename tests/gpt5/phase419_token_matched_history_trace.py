#!/usr/bin/env python3
"""Collect exact-token matched compatible/conflicting history differences."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase416_dual_track_case_bank import read_jsonl, write_json, write_jsonl  # noqa: E402
from phase416_real_collector_qualification import (  # noqa: E402
    eos_ids,
    exact_answer,
    neutral_generation_config,
    target_match,
)
from phase418_interface_history_case_bank import MODELS, SCHEMA_VERSION  # noqa: E402
from phase418_interface_history_trace import (  # noqa: E402
    CORE_COMPONENTS,
    PromptVectorCollector,
    aggregate_contrast,
    depth_bin,
    direction_rows,
    encode_prompt,
    serialize_prompt,
)
from phase419_token_matched_history_case_bank import HISTORIES, INTERFACES, OUT  # noqa: E402


PHASE_ID = "Phase419-TokenMatchedHistoryPhysicalTrace"
REGISTERED = OUT / "phase419_registered_conditions.jsonl"
LEDGER_THRESHOLD = 1e-5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def physical_rows(
    row: dict[str, Any],
    vectors: dict[tuple[int, str], torch.Tensor],
    layer_count: int,
) -> list[dict[str, Any]]:
    output = []
    for layer in range(layer_count):
        input_norm = float(torch.linalg.vector_norm(vectors[(layer, "layer_input")]).item())
        for component in CORE_COMPONENTS:
            vector = vectors[(layer, component)]
            norm = float(torch.linalg.vector_norm(vector).item())
            finite = bool(torch.isfinite(vector).all().item())
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "created_at": now(),
                    "model": row["model"],
                    "condition_id": row["phase419_condition_id"],
                    "semantic_case_id": row["semantic_case_id"],
                    "family_id": row["family_id"],
                    "mechanism_id": row["mechanism_id"],
                    "split": row["split"],
                    "template_id": row["template_id"],
                    "interface": row["interface"],
                    "history_condition": row["history_condition"],
                    "layer": layer,
                    "relative_depth": layer / max(1, layer_count - 1),
                    "depth_bin": depth_bin(layer, layer_count),
                    "component": component,
                    "vector_width": int(vector.numel()),
                    "l2_norm": norm if finite else None,
                    "rms": float(torch.sqrt(torch.mean(vector.square())).item()) if finite else None,
                    "signed_mean": float(torch.mean(vector).item()) if finite else None,
                    "max_abs": float(torch.max(vector.abs()).item()) if finite else None,
                    "relative_write_rate": norm / max(input_norm, 1e-8) if finite and math.isfinite(input_norm) else None,
                    "numerically_finite": finite,
                    "exact_token_matched_pair": True,
                    "physical": True,
                    "reduced_measurement": True,
                    "causal": False,
                }
            )
    return output


def build_contrasts(
    semantic_rows: list[dict[str, Any]],
    vectors: dict[tuple[str, str], dict[tuple[int, str], torch.Tensor]],
    layer_count: int,
    direction_sums: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    contrasts = []
    for interface in INTERFACES:
        contrasts.append(
            (
                "history_identity",
                "compatible_to_conflict",
                interface,
                vectors[(interface, "compatible")],
                vectors[(interface, "conflict")],
                [(-1.0, vectors[(interface, "compatible")]), (1.0, vectors[(interface, "conflict")])],
            )
        )
    contrasts.append(
        (
            "interface_interaction",
            "history_identity_by_interface",
            None,
            None,
            None,
            [
                (1.0, vectors[("completion", "conflict")]),
                (-1.0, vectors[("completion", "compatible")]),
                (-1.0, vectors[("chat", "conflict")]),
                (1.0, vectors[("chat", "compatible")]),
            ],
        )
    )
    first = semantic_rows[0]
    output = []
    partition = "discovery" if first["split"] == "discovery" else "non_discovery"
    for contrast_type, contrast_name, interface, source, target, terms in contrasts:
        for component in CORE_COMPONENTS:
            for depth in ("early", "middle", "late"):
                metrics, mean_delta = aggregate_contrast(
                    source,
                    target,
                    terms,
                    component,
                    depth,
                    layer_count,
                )
                norm = torch.linalg.vector_norm(mean_delta)
                key = (
                    first["model"], first["family_id"], partition, contrast_type,
                    contrast_name, interface, component, depth,
                )
                accumulator = direction_sums.setdefault(
                    key,
                    {"sum": torch.zeros_like(mean_delta), "count": 0},
                )
                if float(norm.item()) > 1e-12:
                    accumulator["sum"] += mean_delta / norm
                    accumulator["count"] += 1
                output.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase419-RegisteredVectorContrasts",
                        "created_at": now(),
                        "model": first["model"],
                        "semantic_case_id": first["semantic_case_id"],
                        "family_id": first["family_id"],
                        "mechanism_id": first["mechanism_id"],
                        "split": first["split"],
                        "contrast_type": contrast_type,
                        "contrast_name": contrast_name,
                        "history_interface": interface,
                        "source_label": "compatible" if source is not None else None,
                        "target_label": "conflict" if target is not None else None,
                        "component": component,
                        "depth_bin": depth,
                        "prompt_token_count_delta": 0 if source is not None else None,
                        "token_length_stratum": "exact" if source is not None else "interaction",
                        "composite_override_contrast": False,
                        **metrics,
                        "physical": True,
                        "predictive": False,
                        "causal": False,
                    }
                )
    return output


@torch.inference_mode()
def run_model(model_key: str, max_new_tokens: int) -> dict[str, Any]:
    registered = [row for row in read_jsonl(REGISTERED) if row["model"] == model_key]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in registered:
        groups[row["semantic_case_id"]].append(row)
    semantic_ids = sorted(groups)
    loaded = None
    collector = None
    started = time.monotonic()
    condition_rows = []
    scalar_rows = []
    contrast_rows = []
    direction_sums: dict[tuple[Any, ...], dict[str, Any]] = {}
    anchors: dict[str, np.ndarray] = {}
    anchor_manifest = []
    try:
        print(f"[Phase419] loading {model_key}; semantic={len(semantic_ids)} conditions={len(registered)}", flush=True)
        loaded = load_probe_model(model_key)
        collector = PromptVectorCollector(loaded)
        collector.install()
        eos = eos_ids(loaded.tokenizer, loaded.model)
        layer_count = len(collector.layers)
        for semantic_index, semantic_id in enumerate(semantic_ids, start=1):
            rows = sorted(
                groups[semantic_id],
                key=lambda row: (INTERFACES.index(row["interface"]), HISTORIES.index(row["history_condition"])),
            )
            vector_bank = {}
            for row in rows:
                prompt, messages = serialize_prompt(loaded.tokenizer, row)
                encoded = encode_prompt(loaded, prompt)
                prompt_count = int(encoded["input_ids"].shape[1])
                registered_count_pass = prompt_count == int(row["registered_prompt_token_count"])
                collector.begin()
                result = loaded.model.generate(
                    **encoded,
                    generation_config=neutral_generation_config(loaded),
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                collector.active = False
                ids = [int(value) for value in result.sequences[0, prompt_count:].tolist()]
                text = loaded.tokenizer.decode(ids, skip_special_tokens=True)
                finite = all(torch.isfinite(vector).all().item() for vector in collector.vectors.values())
                ledger_max = max(collector.ledger_errors, default=math.inf)
                condition_pass = bool(
                    registered_count_pass
                    and len(collector.vectors) == layer_count * len(CORE_COMPONENTS)
                    and finite
                    and ledger_max <= LEDGER_THRESHOLD
                )
                executable = {
                    **row,
                    "created_at": now(),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_token_count": prompt_count,
                    "registered_prompt_token_count_pass": registered_count_pass,
                    "message_count": len(messages),
                    "generated_token_ids": ids,
                    "generated_text": text,
                    "target_event_match": target_match(text, row["target_aliases"]),
                    "exact_answer_match": exact_answer(text, row["target_aliases"]),
                    "emitted_stop": any(token in eos for token in ids),
                    "right_censored": not any(token in eos for token in ids) and len(ids) >= max_new_tokens,
                    "native_generation_call_count": collector.call_index + 1,
                    "prompt_vector_count": len(collector.vectors),
                    "component_ledger_max_relative_error": ledger_max,
                    "physical_finite": finite,
                    "condition_pass": condition_pass,
                    "causal": False,
                }
                condition_rows.append(executable)
                scalar_rows.extend(physical_rows(executable, collector.vectors, layer_count))
                vector_bank[(row["interface"], row["history_condition"])] = {
                    key: value.clone() for key, value in collector.vectors.items()
                }
                family_first = min(
                    item["semantic_case_id"] for item in registered if item["family_id"] == row["family_id"]
                )
                if semantic_id == family_first and row["interface"] == "chat":
                    prefix = f"{row['family_id']}__chat__{row['history_condition']}"
                    for (layer, component), vector in collector.vectors.items():
                        anchors[f"{prefix}__layer_{layer:02d}__{component}"] = vector.numpy().astype(np.float16)
                    anchor_manifest.append({
                        "condition_id": row["phase419_condition_id"],
                        "family_id": row["family_id"],
                        "history_condition": row["history_condition"],
                        "vector_count": len(collector.vectors),
                    })
                collector.end()
                del encoded, result
            if all(row["condition_pass"] for row in condition_rows[-4:]):
                contrast_rows.extend(build_contrasts(rows, vector_bank, layer_count, direction_sums))
            del vector_bank
            gc.collect()
            if semantic_index % 3 == 0 or semantic_index == len(semantic_ids):
                print(
                    f"[Phase419:{model_key}] semantic={semantic_index}/{len(semantic_ids)} "
                    f"conditions={len(condition_rows)} pass={sum(row['condition_pass'] for row in condition_rows)}",
                    flush=True,
                )

        model_root = OUT / "models" / model_key
        directions = direction_rows(direction_sums)
        for row in directions:
            row["phase_id"] = "Phase419-DirectionConsistency"
        write_jsonl(model_root / "phase419_condition_rows.jsonl", condition_rows)
        write_jsonl(model_root / "phase419_prefill_physical_rows.jsonl", scalar_rows)
        write_jsonl(model_root / "phase419_vector_contrast_rows.jsonl", contrast_rows)
        write_jsonl(model_root / "phase419_direction_consistency_rows.jsonl", directions)
        model_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model_root / "phase419_anchor_vectors.npz", **anchors)
        write_json(model_root / "phase419_anchor_vector_manifest.json", anchor_manifest)
        pair_counts = defaultdict(dict)
        for row in condition_rows:
            pair_counts[(row["semantic_case_id"], row["interface"])][row["history_condition"]] = row["prompt_token_count"]
        exact_pairs = sum(
            values.get("compatible") == values.get("conflict") for values in pair_counts.values()
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model_key,
            "semantic_case_count": len(semantic_ids),
            "condition_count": len(condition_rows),
            "required_condition_count": len(registered),
            "condition_pass_count": sum(row["condition_pass"] for row in condition_rows),
            "all_conditions_pass": bool(len(condition_rows) == len(registered) and all(row["condition_pass"] for row in condition_rows)),
            "exact_prompt_token_count_pair_count": exact_pairs,
            "required_exact_prompt_token_count_pair_count": len(semantic_ids) * len(INTERFACES),
            "target_event_match_count": sum(row["target_event_match"] for row in condition_rows),
            "exact_answer_match_count": sum(row["exact_answer_match"] for row in condition_rows),
            "right_censored_count": sum(row["right_censored"] for row in condition_rows),
            "physical_row_count": len(scalar_rows),
            "vector_contrast_row_count": len(contrast_rows),
            "direction_consistency_row_count": len(directions),
            "lossless_anchor_condition_count": len(anchor_manifest),
            "lossless_anchor_vector_count": len(anchors),
            "max_component_ledger_relative_error": max(row["component_ledger_max_relative_error"] for row in condition_rows),
            "condition_rows_sha256": hash_rows(condition_rows),
            "physical_rows_sha256": hash_rows(scalar_rows),
            "contrast_rows_sha256": hash_rows(contrast_rows),
            "elapsed_seconds": time.monotonic() - started,
            "vram_gb": vram_gb(),
            "causal_intervention_authorized": False,
            "single_neuron_scan_authorized": False,
            "claim_boundary": "exact_token_matched_prior_answer_identity_physical_difference_only",
        }
        write_json(model_root / "phase419_trace_complete.json", summary)
        return summary
    finally:
        if collector is not None:
            collector.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    summary = run_model(args.model, args.max_new_tokens)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not summary["all_conditions_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
