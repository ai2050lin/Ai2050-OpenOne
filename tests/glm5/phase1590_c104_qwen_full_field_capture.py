#!/usr/bin/env python3
"""Phase1590 / C104: capture the fresh breadth field at full activation resolution."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1589_c104_upstream_candidate_validation"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

PHASE = 1590
CAMPAIGN = "C104"
STATES = 37
DIM = 2560
RAW_NAME = "qwen3_all_token_state_coordinate_field.uint16.npy"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def model_forward(model: Any, rows: list[dict[str, Any]], pad: int, device: torch.device, width: int, hidden: bool):
    ids, mask, positions, lengths = fixed_base.fixed_batch(rows, pad, device, width)
    output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False,
                         output_hidden_states=hidden, return_dict=True)
    boundary = torch.stack([output.last_hidden_state[i, length - 1] for i, length in enumerate(lengths)], dim=0)
    return output, model.lm_head(boundary).float(), ids, mask, positions, lengths


def prepare() -> None:
    contract = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/independent_pre_model_audit.json")
    if contract["authorization"] != "run_phase1590_c104_qwen_capture" or not audit["all_checks_passed"]:
        raise RuntimeError("C104 capture authorization missing")
    adapter = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_all_token_all_state_all_activation_coordinate_capture_adapter_frozen",
        "producer_sha256": core.sha(Path(__file__)),
        "contract_sha256": core.sha(OUT / "protocol/preregistration.json"),
        "barcode_sha256": contract["barcode_sha256"],
        "material_digest": contract["material_digest"],
        "archive": contract["storage"],
        "behavior_boundary": "real final valid input token; never padded tail",
        "activation_scope": "embedding plus 36 Hidden States; activation coordinates are not weight parameters",
        "authorization": "execute_qwen_full_field_capture",
    }
    core.save(OUT / "protocol/capture_adapter.json", adapter)
    print(json.dumps(adapter, indent=2))


def behavior_summary(index: list[dict[str, Any]]) -> dict[str, Any]:
    def accuracy(rows: list[dict[str, Any]]) -> float:
        return float(np.mean([row["correct"] for row in rows])) if rows else 0.0
    return {
        "global_accuracy": accuracy(index),
        "by_partition": {p: accuracy([r for r in index if r["partition"] == p]) for p in graph_base.PARTITIONS},
        "by_code": {graph_base.CODEBOOKS[c]["name"]: accuracy([r for r in index if r["code"] == c]) for c in (1, -1)},
        "by_family": {f: accuracy([r for r in index if r["family"] == f]) for f in sorted({r["family"] for r in index})},
    }


def maximum_role_difference(field: np.ndarray, first: dict[str, Any], second: dict[str, Any], roles: tuple[str, ...]) -> float:
    maximum = 0.0
    for role in roles:
        left_positions = first["role_positions"][role]
        right_positions = second["role_positions"][role]
        if len(left_positions) != len(right_positions):
            raise RuntimeError((role, len(left_positions), len(right_positions)))
        left = np.asarray(field[:, first["token_start"] + np.asarray(left_positions)], dtype=np.uint16)
        right = np.asarray(field[:, second["token_start"] + np.asarray(right_positions)], dtype=np.uint16)
        if not np.array_equal(left, right):
            maximum = max(maximum, float(np.max(np.abs(decode_bf16(left) - decode_bf16(right)))))
    return maximum


@torch.inference_mode()
def capture() -> None:
    contract = core.load(OUT / "protocol/preregistration.json")
    adapter = core.load(OUT / "protocol/capture_adapter.json")
    source_audit = core.load(OUT / "audit/independent_pre_model_audit.json")
    if adapter["authorization"] != "execute_qwen_full_field_capture" or not source_audit["all_checks_passed"]:
        raise RuntimeError("C104 capture not authorized")
    if adapter["producer_sha256"] != core.sha(Path(__file__)) or adapter["material_digest"] != contract["material_digest"]:
        raise RuntimeError("frozen capture identity changed")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    lengths_expected = [len(row["prompt_ids"]) for row in rows]
    total_tokens = sum(lengths_expected)
    if total_tokens != contract["storage"]["total_valid_tokens"]:
        raise RuntimeError((total_tokens, contract["storage"]["total_valid_tokens"]))
    token_starts = np.cumsum([0, *lengths_expected[:-1]], dtype=np.int64).tolist()
    raw_path = OUT / f"raw/{RAW_NAME}"
    if raw_path.exists():
        raise RuntimeError(f"raw archive already exists: {raw_path}")
    field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=(STATES, total_tokens, DIM))
    index: list[dict[str, Any]] = []
    model = None
    finite = True
    repeat_hidden = math.inf
    repeat_logits = math.inf
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        width = int(contract["storage"]["fixed_width"])
        batch_size = int(contract["storage"]["batch_size"])
        first_rows = None
        first_scores = None
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            output, logits, ids, mask, positions, lengths = model_forward(model, batch, pad, device, width, True)
            if len(output.hidden_states) != STATES or output.hidden_states[-1].shape[-1] != DIM:
                raise RuntimeError((len(output.hidden_states), output.hidden_states[-1].shape))
            for state_index, state in enumerate(output.hidden_states):
                if state.dtype != torch.bfloat16:
                    raise RuntimeError(("expected BF16 Hidden State", state.dtype))
                finite = finite and bool(torch.isfinite(state).all())
                for local, length in enumerate(lengths):
                    row_index = start + local
                    token_start = token_starts[row_index]
                    field[state_index, token_start:token_start + length] = state[local, :length].contiguous().view(torch.uint16).cpu().numpy()
            batch_scores = []
            for local, row in enumerate(batch):
                row_index = start + local
                length = lengths[local]
                token_start = token_starts[row_index]
                token_ids = [int(value) for value in row["prompt_ids"]]
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                batch_scores.append(scores)
                metadata_keys = ("case_id", "unit_id", "family", "world", "partition", "surface", "code", "codebook",
                                 "truth", "output_yes", "gold_position", "truth_factor", "surface_factor", "distractor_factor")
                index.append({
                    "row_index": row_index,
                    **{key: row[key] for key in metadata_keys},
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                    "scores": scores,
                    "prompt_length": length,
                    "token_start": token_start,
                    "token_end": token_start + length,
                    "token_ids": token_ids,
                    "token_texts": tok.convert_ids_to_tokens(token_ids),
                    "role_positions": row["role_positions"],
                })
            if start == 0:
                first_rows, first_scores = batch, batch_scores
            if (start // batch_size + 1) % 12 == 0:
                field.flush()
                print(f"[phase1590] captured {start + len(batch)}/{len(rows)} cases", flush=True)
            del output, logits, ids, mask, positions
        field.flush()
        if first_rows is None or first_scores is None:
            raise RuntimeError("repeat batch missing")
        output, logits, ids, mask, positions, lengths = model_forward(model, first_rows, pad, device, width, True)
        repeat_hidden = 0.0
        for state_index, state in enumerate(output.hidden_states):
            for local, length in enumerate(lengths):
                token_start = token_starts[local]
                old_bits = np.asarray(field[state_index, token_start:token_start + length], dtype=np.uint16)
                new_bits = state[local, :length].contiguous().view(torch.uint16).cpu().numpy()
                if not np.array_equal(old_bits, new_bits):
                    repeat_hidden = max(repeat_hidden, float(np.max(np.abs(decode_bf16(old_bits) - decode_bf16(new_bits)))))
        repeat_logits = 0.0
        for local, row in enumerate(first_rows):
            scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
            repeat_logits = max(repeat_logits, max(abs(scores[i] - first_scores[local][i]) for i in range(2)))
        del output, logits, ids, mask, positions
    finally:
        field.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()

    index_path = OUT / "raw/qwen3_all_token_state_coordinate_index.jsonl"
    core.write_rows(index_path, index)
    field = np.load(raw_path, mmap_mode="r")
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        by_unit[row["unit_id"]].append(row)
    causal_prefix = 0.0
    code_previsible = 0.0
    for unit_rows in by_unit.values():
        for row in unit_rows[1:]:
            causal_prefix = max(causal_prefix, maximum_role_difference(field, unit_rows[0], row, ("focus_pre",)))
        for truth, surface, distractor in itertools.product((1, -1), repeat=3):
            standard = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, 1))
            reversed_row = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, -1))
            code_previsible = max(code_previsible, maximum_role_difference(field, standard, reversed_row, ("focus_post", "query_focus", "query_anchor")))
    checks = {
        "shape": list(field.shape) == [STATES, total_tokens, DIM],
        "index": len(index) == 576 and sum(row["prompt_length"] for row in index) == total_tokens,
        "finite": finite and all(math.isfinite(value) for row in index for value in row["scores"]),
        "repeat_hidden": repeat_hidden == 0.0,
        "repeat_logits": repeat_logits == 0.0,
        "causal_prefix": causal_prefix == 0.0,
        "code_previsible": code_previsible == 0.0,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "exact_bf16_archive": field.dtype == np.uint16,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "numeric": {"repeat_hidden": repeat_hidden, "repeat_logits": repeat_logits,
                                                            "causal_prefix": causal_prefix, "code_previsible": code_previsible}})
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "fresh_qwen_full_activation_field_capture_complete",
        "shape": list(field.shape),
        "valid_tokens": total_tokens,
        "bytes": raw_path.stat().st_size,
        "raw_sha256": core.sha(raw_path),
        "index_sha256": core.sha(index_path),
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logit_max_abs": repeat_logits,
                    "causal_prefix_max_abs": causal_prefix, "code_previsible_max_abs": code_previsible},
        "behavior": behavior_summary(index),
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "finished_at_utc": now(),
        "authorization": "run_phase1591_c104_frozen_candidate_validation",
    }
    core.save(OUT / "analysis/qwen_full_field_capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "capture"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else capture()


if __name__ == "__main__":
    main()
