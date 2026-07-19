#!/usr/bin/env python3
"""Collect scalar matched-control geometry in the frozen Phase548 windows."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))

from hf_probe_env import get_layers  # noqa: E402
from model_utils import load_model, release_model  # noqa: E402
from phase548_shared_attention_compute_protocol import MODELS, OUT_DIR, VARIANTS  # noqa: E402


CASES_PATH = OUT_DIR / "phase548_registered_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase548_frozen_protocol.json"
BEHAVIOR_PATH = OUT_DIR / "phase548_behavior_qualification.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(model: str) -> Path:
    return OUT_DIR / f"phase548_{model}_matched_observer_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase548_{model}_matched_observer_execution.json"


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    raise TypeError(f"Unsupported attention output: {type(output).__name__}")


def normalized_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(left.float() - right.float()).item())
    denominator = 0.5 * (
        float(torch.linalg.vector_norm(left.float()).item())
        + float(torch.linalg.vector_norm(right.float()).item())
    )
    value = numerator / max(denominator, 1e-8)
    if not math.isfinite(value):
        raise RuntimeError("Non-finite Phase548 matched delta")
    return round(value, 9)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    denominator = float(torch.linalg.vector_norm(left).item() * torch.linalg.vector_norm(right).item())
    return round(float(torch.dot(left, right).item()) / max(denominator, 1e-8), 9)


def platform_delta(
    vectors: dict[int, dict[str, torch.Tensor]], left_name: str, right_name: str,
) -> tuple[float, float]:
    left_norm_sq = right_norm_sq = delta_norm_sq = 0.0
    left_flat = []
    right_flat = []
    for layer in sorted(vectors):
        left = vectors[layer][left_name].float()
        right = vectors[layer][right_name].float()
        left_norm_sq += float(torch.dot(left, left).item())
        right_norm_sq += float(torch.dot(right, right).item())
        difference = left - right
        delta_norm_sq += float(torch.dot(difference, difference).item())
        left_flat.append(left)
        right_flat.append(right)
    normalized = math.sqrt(delta_norm_sq) / max(
        0.5 * (math.sqrt(left_norm_sq) + math.sqrt(right_norm_sq)), 1e-8,
    )
    return round(normalized, 9), cosine(torch.cat(left_flat), torch.cat(right_flat))


def capture_anchor(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    target_layers: list[int],
    rows: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, torch.Tensor]], list[torch.Tensor]]:
    ordered = sorted(rows, key=lambda row: VARIANTS.index(row["variant"]))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in target_layers:
        def hook(_module: Any, _inputs: Any, output: Any, idx: int = layer_index) -> None:
            captures[idx] = component_tensor(output).detach()
        handles.append(layers[layer_index].self_attn.register_forward_hook(hook))
    encoded = tokenizer(
        [row["prompt"] for row in ordered], return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            result = model(**encoded, use_cache=False, return_dict=True)
        if set(captures) != set(target_layers):
            raise RuntimeError(f"Incomplete Phase548 captures: {sorted(captures)}")
        vectors: dict[int, dict[str, torch.Tensor]] = {}
        for layer_index in target_layers:
            tensor = captures[layer_index]
            vectors[layer_index] = {
                row["variant"]: tensor[index, -1].float().detach().cpu()
                for index, row in enumerate(ordered)
            }
        logits = [result.logits[index, -1].float().detach().cpu() for index in range(len(ordered))]
        return vectors, logits
    finally:
        for handle in handles:
            handle.remove()
        del encoded


def collect(model_name: str, use_8bit: bool, restart: bool) -> Path:
    protocol = read_json(PROTOCOL_PATH)
    if protocol["registered_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase548 observer case-bank drift")
    qualification = {
        row["mechanism_id"]: row
        for row in read_jsonl(BEHAVIOR_PATH)
        if row["model"] == model_name
    }
    authorized = {
        mechanism for mechanism, row in qualification.items()
        if row["observer_collection_authorized"]
    }
    target_layers = list(protocol["frozen_windows"][model_name]["target_layers"])
    if not authorized or not target_layers:
        payload = {
            "schema_version": "phase548_matched_observer_execution.v1",
            "phase_id": "Phase548", "created_at": now(), "model": model_name,
            "status": "skipped_by_behavior_or_prior_window_gate", "cuda_loaded": False,
            "authorized_mechanisms": sorted(authorized), "target_layers": target_layers,
            "row_count": 0, "new_sealed_split_read": False,
        }
        write_json(summary_path(model_name), payload)
        return summary_path(model_name)
    source = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == model_name and row["mechanism_id"] in authorized
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        grouped.setdefault(row["anchor_id"], []).append(row)
    output = output_path(model_name)
    if restart:
        output.unlink(missing_ok=True)
        summary_path(model_name).unlink(missing_ok=True)
    completed = {row["anchor_id"] for row in read_jsonl(output)} if output.exists() else set()
    pending = [(anchor, grouped[anchor]) for anchor in sorted(grouped) if anchor not in completed]
    loaded = None
    started = time.monotonic()
    new_rows = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase548 observer collection requires CUDA")
        loaded, tokenizer, device = load_model(model_name, use_8bit=True if use_8bit else None)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        layers = list(get_layers(loaded))
        if max(target_layers) >= len(layers):
            raise RuntimeError(f"Frozen Phase548 layer outside model: {model_name}")
        for index, (anchor_id, anchor_rows) in enumerate(pending, 1):
            vectors, _logits = capture_anchor(
                loaded, tokenizer, device, layers, target_layers, anchor_rows,
            )
            by_variant = {row["variant"]: row for row in anchor_rows}
            rows_out = []
            pairs = {
                "functional_delta": ("base_plus", "functional_minus"),
                "identity_delta": ("base_plus", "identity_control"),
                "answer_token_delta": ("base_plus", "answer_token_control"),
                "template_delta": ("base_plus", "template_control"),
            }
            common = {
                "schema_version": "phase548_matched_observer_row.v1",
                "phase_id": "Phase548", "created_at": now(), "model": model_name,
                "family_id": "content_knowledge",
                "mechanism_id": by_variant["base_plus"]["mechanism_id"],
                "split": by_variant["base_plus"]["split"],
                "pair_index": by_variant["base_plus"]["pair_index"],
                "anchor_id": anchor_id, "stage": "prompt_end", "role": "current",
                "component": "attention_output", "sealed": False, "observer_only": True,
                "compute_edge": False, "causal": False, "single_neuron": False,
            }
            for layer_index in target_layers:
                payload = dict(common, layer=layer_index, aggregation="single_layer")
                for name, (left_name, right_name) in pairs.items():
                    payload[name] = normalized_delta(
                        vectors[layer_index][left_name], vectors[layer_index][right_name],
                    )
                    payload[name.replace("delta", "cosine")] = cosine(
                        vectors[layer_index][left_name], vectors[layer_index][right_name],
                    )
                rows_out.append(payload)
            payload = dict(common, layer=None, aggregation="frozen_three_layer_platform")
            for name, (left_name, right_name) in pairs.items():
                delta, pair_cosine = platform_delta(vectors, left_name, right_name)
                payload[name] = delta
                payload[name.replace("delta", "cosine")] = pair_cosine
            rows_out.append(payload)
            append_jsonl(output, rows_out)
            new_rows += len(rows_out)
            del vectors, _logits
            if index == 1 or index % 32 == 0 or index == len(pending):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_name} observer "
                    f"{len(completed) + index}/{len(grouped)}",
                    flush=True,
                )
        final_rows = read_jsonl(output)
        expected_rows = len(grouped) * (len(target_layers) + 1)
        if len(final_rows) != expected_rows:
            raise RuntimeError(f"Incomplete Phase548 observer rows: {len(final_rows)}/{expected_rows}")
        payload = {
            "schema_version": "phase548_matched_observer_execution.v1",
            "phase_id": "Phase548", "created_at": now(), "model": model_name,
            "status": "complete", "cuda_loaded": True,
            "authorized_mechanisms": sorted(authorized), "target_layers": target_layers,
            "anchor_count": len(grouped), "row_count": len(final_rows),
            "rows_path": str(output.relative_to(ROOT)), "rows_sha256": sha256_file(output),
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "new_rows_this_invocation": new_rows,
            "full_hidden_vectors_persisted": False, "head_channel_neuron_scan_executed": False,
            "new_sealed_split_read": False,
        }
        write_json(summary_path(model_name), payload)
        print(summary_path(model_name))
        return summary_path(model_name)
    finally:
        if loaded is not None:
            release_model(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    collect(args.model, args.use_8bit, args.restart)


if __name__ == "__main__":
    main()
