#!/usr/bin/env python3
"""Read-only finalization after the C109 capture report hit a JSON bool type error."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
SOURCE = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def role_bits(field: np.ndarray, lookup: dict[tuple[int, str], list[int]], row_index: int, role: str) -> np.ndarray:
    return np.asarray(field[:, lookup[(row_index, role)], :], dtype=np.uint16)


@torch.inference_mode()
def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    source_audit = core.load(OUT / "audit/independent_pre_model_audit.json")
    if not source_audit["all_checks_passed"]:
        raise RuntimeError("C109 pre-model audit missing")
    rows = core.rows(SOURCE / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    behavior = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    raw_path = OUT / protocol["archive"]["path"]
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    field = np.load(raw_path, mmap_mode="r")
    candidate_logits = np.load(logits_path, mmap_mode="r")
    if list(field.shape) != protocol["archive"]["shape"] or candidate_logits.shape != (384, 2) or len(behavior) != 384:
        raise RuntimeError("partial raw archive")

    finite_archive = True
    for state in range(field.shape[0]):
        for start in range(0, field.shape[1], 256):
            finite_archive = finite_archive and bool(np.isfinite(decode_bf16(field[state, start:start + 256])).all())

    by_row: dict[int, list[dict]] = defaultdict(list)
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        row_index = int(occurrence["row_index"])
        by_row[row_index].append(occurrence)
        lookup[(row_index, occurrence["role"])].append(int(occurrence["occurrence_index"]))

    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        batch = rows[: int(protocol["archive"]["batch_size"])]
        ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, int(protocol["archive"]["fixed_width"]))
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=True, return_dict=True)
        boundary = torch.stack([output.last_hidden_state[i, length - 1] for i, length in enumerate(lengths)], dim=0)
        logits = model.lm_head(boundary).float()
        repeat_hidden = 0.0
        for state_index, state in enumerate(output.hidden_states):
            for local, _length in enumerate(lengths):
                occurrences = by_row[local]
                indices = np.asarray([int(item["occurrence_index"]) for item in occurrences], dtype=np.int64)
                positions_local = [int(item["token_position"]) for item in occurrences]
                old_bits = np.asarray(field[state_index, indices, :], dtype=np.uint16)
                new_bits = state[local, positions_local].contiguous().view(torch.uint16).cpu().numpy()
                if not np.array_equal(old_bits, new_bits):
                    repeat_hidden = max(repeat_hidden, float(np.max(np.abs(decode_bf16(old_bits) - decode_bf16(new_bits)))))
        repeat_logits = 0.0
        for local, row in enumerate(batch):
            scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
            repeat_logits = max(repeat_logits, float(np.max(np.abs(scores - candidate_logits[local]))))
        del output, logits, boundary, ids, mask, positions
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()

    by_unit: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_unit[row["unit_id"]].append(row_index)
    causal_prefix = 0.0
    code_previsible = 0.0
    pre_code_roles = ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor")
    for row_indices in by_unit.values():
        reference = row_indices[0]
        for row_index in row_indices[1:]:
            left = role_bits(field, lookup, reference, "focus_pre")
            right = role_bits(field, lookup, row_index, "focus_pre")
            if not np.array_equal(left, right):
                causal_prefix = max(causal_prefix, float(np.max(np.abs(decode_bf16(left) - decode_bf16(right)))))
        unit_rows = [rows[index] for index in row_indices]
        for truth, surface, distractor in itertools.product((1, -1), repeat=3):
            standard = next(index for index, row in zip(row_indices, unit_rows, strict=True) if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, 1))
            reversed_code = next(index for index, row in zip(row_indices, unit_rows, strict=True) if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, -1))
            for role in pre_code_roles:
                left = role_bits(field, lookup, standard, role)
                right = role_bits(field, lookup, reversed_code, role)
                if not np.array_equal(left, right):
                    code_previsible = max(code_previsible, float(np.max(np.abs(decode_bf16(left) - decode_bf16(right)))))

    def accuracy(selected: list[dict]) -> float:
        return float(np.mean([row["correct"] for row in selected]))

    behavior_summary = {
        "global_accuracy": accuracy(behavior),
        "by_family": {family: accuracy([row for row in behavior if row["family"] == family]) for family in ("attribute_binding", "agent_patient")},
        "by_partition": {partition: accuracy([row for row in behavior if row["partition"] == partition]) for partition in ("prospective_confirmation", "independent_lockbox")},
        "by_code": {str(code): accuracy([row for row in behavior if row["code"] == code]) for code in (1, -1)},
    }
    checks = {
        "shape": list(field.shape) == protocol["archive"]["shape"],
        "dtype": field.dtype == np.uint16,
        "logit_shape": list(candidate_logits.shape) == [384, 2] and candidate_logits.dtype == np.float32,
        "index": len(behavior) == 384 and all(row["row_index"] == index for index, row in enumerate(behavior)),
        "finite": finite_archive and bool(np.isfinite(candidate_logits).all()),
        "repeat_hidden": repeat_hidden == 0.0,
        "repeat_logits": repeat_logits == 0.0,
        "causal_prefix": causal_prefix == 0.0,
        "code_previsible": code_previsible == 0.0,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "repeat_hidden": repeat_hidden, "repeat_logits": repeat_logits, "causal_prefix": causal_prefix, "code_previsible": code_previsible})
    report = {
        "phase": 1604,
        "campaign": "C109",
        "created_at_utc": now(),
        "status": "exact_bf16_capture_finalized_after_json_scalar_type_repair",
        "incident": "the original capture completed and released the model, then JSON serialization rejected a numpy.bool_; this finalizer is read-only over the archive and reruns the first-batch identity check",
        "capture_producer_sha256": core.sha(TESTS / "phase1604_c109_qwen_role_state_capture.py"),
        "finalizer_sha256": core.sha(Path(__file__)),
        "shape": list(field.shape),
        "raw_file_bytes": raw_path.stat().st_size,
        "raw_data_bytes": int(field.nbytes),
        "raw_sha256": core.sha(raw_path),
        "logits_sha256": core.sha(logits_path),
        "index_sha256": core.sha(index_path),
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logits_max_abs": repeat_logits, "causal_prefix_max_abs": causal_prefix, "code_previsible_max_abs": code_previsible},
        "behavior": behavior_summary,
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "authorization": "run_phase1605_c109_basic_coordinate_observation",
    }
    core.save(OUT / "analysis/capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
