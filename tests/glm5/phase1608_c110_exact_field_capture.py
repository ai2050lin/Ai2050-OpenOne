#!/usr/bin/env python3
"""Phase1608 / C110: exact BF16 capture for the third lexical field."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def forward(model: Any, rows: list[dict], pad: int, device: torch.device, width: int):
    ids, mask, positions, lengths = fixed_base.fixed_batch(rows, pad, device, width)
    output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=True, return_dict=True)
    boundary = torch.stack([output.last_hidden_state[i, length - 1] for i, length in enumerate(lengths)], dim=0)
    return output, model.lm_head(boundary).float(), ids, mask, positions, lengths


def role_bits(field: np.ndarray, lookup: dict[tuple[int, str], list[int]], row_index: int, role: str) -> np.ndarray:
    return np.asarray(field[:, lookup[(row_index, role)], :], dtype=np.uint16)


@torch.inference_mode()
def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/independent_pre_model_audit.json")
    if protocol["authorization"] != "execute_phase1608_c110_exact_field_capture" or not audit["all_checks_passed"]:
        raise RuntimeError("C110 capture authorization missing")
    if protocol["producer_sha256"] != core.sha(TESTS / "phase1607_c110_fresh_readout_control_contract.py"):
        raise RuntimeError("C110 frozen contract changed")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    by_row: dict[int, list[dict]] = defaultdict(list)
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        row_index = int(occurrence["row_index"])
        by_row[row_index].append(occurrence)
        lookup[(row_index, occurrence["role"])].append(int(occurrence["occurrence_index"]))
    raw_path = OUT / protocol["archive"]["path"]
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() or logits_path.exists() or index_path.exists():
        raise RuntimeError("C110 raw archive already exists")
    field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=tuple(protocol["archive"]["shape"]))
    candidate_logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(384, 2))
    behavior = []
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        batch_size = int(protocol["archive"]["batch_size"])
        width = int(protocol["archive"]["fixed_width"])
        first_rows = None
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            output, logits, ids, mask, positions, lengths = forward(model, batch, pad, device, width)
            if len(output.hidden_states) != 37 or output.hidden_states[-1].shape[-1] != 2560:
                raise RuntimeError((len(output.hidden_states), output.hidden_states[-1].shape))
            for state_index, state in enumerate(output.hidden_states):
                if state.dtype != torch.bfloat16 or not bool(torch.isfinite(state).all()):
                    raise RuntimeError((state_index, state.dtype, bool(torch.isfinite(state).all())))
                for local, _length in enumerate(lengths):
                    row_index = start + local
                    occurrences = by_row[row_index]
                    occurrence_indices = np.asarray([int(item["occurrence_index"]) for item in occurrences], dtype=np.int64)
                    token_positions = [int(item["token_position"]) for item in occurrences]
                    field[state_index, occurrence_indices] = state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
            for local, row in enumerate(batch):
                row_index = start + local
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                candidate_logits[row_index] = scores
                prediction = int(scores[1] > scores[0])
                behavior.append({
                    "row_index": row_index, "case_id": row["case_id"], "unit_id": row["unit_id"], "family": row["family"], "partition": row["partition"],
                    "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "code": row["code"],
                    "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "yes_minus_no": scores[0] - scores[1],
                })
            if start == 0:
                first_rows = batch
            if (start // batch_size + 1) % 8 == 0:
                field.flush(); candidate_logits.flush()
                print(f"[phase1608] captured {start + len(batch)}/{len(rows)} cases", flush=True)
            del output, logits, ids, mask, positions
        field.flush(); candidate_logits.flush()
        output, logits, ids, mask, positions, lengths = forward(model, first_rows, pad, device, width)
        repeat_hidden = 0.0
        for state_index, state in enumerate(output.hidden_states):
            for local, _length in enumerate(lengths):
                occurrences = by_row[local]
                indices = np.asarray([int(item["occurrence_index"]) for item in occurrences], dtype=np.int64)
                token_positions = [int(item["token_position"]) for item in occurrences]
                old_bits = np.asarray(field[state_index, indices], dtype=np.uint16)
                new_bits = state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
                if not np.array_equal(old_bits, new_bits):
                    repeat_hidden = max(repeat_hidden, float(np.max(np.abs(decode_bf16(old_bits) - decode_bf16(new_bits)))))
        repeat_logits = 0.0
        for local, row in enumerate(first_rows):
            scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
            repeat_logits = max(repeat_logits, float(np.max(np.abs(scores - candidate_logits[local]))))
        del output, logits, ids, mask, positions
    finally:
        field.flush(); candidate_logits.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
    core.write_rows(index_path, behavior)
    field = np.load(raw_path, mmap_mode="r")
    candidate_logits = np.load(logits_path, mmap_mode="r")
    by_unit: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_unit[row["unit_id"]].append(row_index)
    causal_prefix = 0.0
    code_previsible = 0.0
    for row_indices in by_unit.values():
        reference = row_indices[0]
        for row_index in row_indices[1:]:
            left, right = role_bits(field, lookup, reference, "focus_pre"), role_bits(field, lookup, row_index, "focus_pre")
            if not np.array_equal(left, right):
                causal_prefix = max(causal_prefix, float(np.max(np.abs(decode_bf16(left) - decode_bf16(right)))))
        unit_rows = [rows[index] for index in row_indices]
        for truth, surface, distractor in itertools.product((1, -1), repeat=3):
            standard = next(index for index, row in zip(row_indices, unit_rows, strict=True) if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, 1))
            reversed_code = next(index for index, row in zip(row_indices, unit_rows, strict=True) if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, -1))
            for role in ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor"):
                left, right = role_bits(field, lookup, standard, role), role_bits(field, lookup, reversed_code, role)
                if not np.array_equal(left, right):
                    code_previsible = max(code_previsible, float(np.max(np.abs(decode_bf16(left) - decode_bf16(right)))))
    def acc(selected: list[dict]) -> float:
        return float(np.mean([row["correct"] for row in selected]))
    behavior_summary = {
        "global_accuracy": acc(behavior),
        "by_family": {family: acc([row for row in behavior if row["family"] == family]) for family in protocol["families"]},
        "by_partition": {partition: acc([row for row in behavior if row["partition"] == partition]) for partition in protocol["partitions"]},
        "by_code": {str(code): acc([row for row in behavior if row["code"] == code]) for code in (1, -1)},
    }
    checks = {
        "shape": list(field.shape) == protocol["archive"]["shape"], "dtype": field.dtype == np.uint16,
        "logits": bool(candidate_logits.shape == (384, 2) and candidate_logits.dtype == np.float32 and np.isfinite(candidate_logits).all()),
        "index": len(behavior) == 384 and all(row["row_index"] == index for index, row in enumerate(behavior)),
        "repeat_hidden": repeat_hidden == 0.0, "repeat_logits": repeat_logits == 0.0,
        "causal_prefix": causal_prefix == 0.0, "code_previsible": code_previsible == 0.0,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1608, "campaign": "C110", "created_at_utc": now(), "status": "fresh_exact_bf16_field_capture_complete",
        "producer_sha256": core.sha(Path(__file__)), "shape": list(field.shape), "raw_file_bytes": raw_path.stat().st_size, "raw_data_bytes": int(field.nbytes),
        "raw_sha256": core.sha(raw_path), "logits_sha256": core.sha(logits_path), "index_sha256": core.sha(index_path),
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logits_max_abs": repeat_logits, "causal_prefix_max_abs": causal_prefix, "code_previsible_max_abs": code_previsible},
        "behavior": behavior_summary, "runtime": {"placement": placement, "quantization": quant}, "checks": checks,
        "authorization": "run_phase1609_c110_field_prediction_and_transport_tests",
    }
    core.save(OUT / "analysis/capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
