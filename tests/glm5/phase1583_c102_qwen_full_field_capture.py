#!/usr/bin/env python3
"""Phase1583 / C102: capture every valid token, state and activation coordinate."""
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
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

PHASE = 1583
CAMPAIGN = "C102"
STATES = 37
DIM = 2560
RAW_NAME = "qwen3_all_token_state_coordinate_field.uint16.npy"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def model_forward(model: Any, rows: list[dict[str, Any]], pad: int, device: torch.device, width: int, hidden: bool):
    ids, mask, positions, lengths = fixed_base.fixed_batch(rows, pad, device, width)
    output = model.model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        output_hidden_states=hidden,
        return_dict=True,
    )
    boundary = torch.stack([output.last_hidden_state[i, length - 1] for i, length in enumerate(lengths)], dim=0)
    logits = model.lm_head(boundary).float()
    return output, logits, ids, mask, positions, lengths


def prepare() -> None:
    contract = core.load(OUT / "protocol/preregistration.json")
    discovery = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    audit = core.load(OUT / "audit/independent_c101_discovery_audit.json")
    if discovery["authorization"] != "run_phase1583_c102_qwen_full_field_capture" or not audit["all_checks_passed"]:
        raise RuntimeError("C102 full-field capture authorization missing")
    adapter = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "all_token_all_state_all_activation_coordinate_capture_adapter_frozen",
        "producer_sha256": core.sha(Path(__file__)),
        "contract_sha256": core.sha(OUT / "protocol/preregistration.json"),
        "prediction_sha256": core.sha(OUT / "protocol/frozen_coordinate_barcode_predictions.json"),
        "material_digest": contract["material_digest"],
        "archive": contract["storage"],
        "numeric_gates": contract["numeric_gates"],
        "behavior_boundary": "real final valid input token; never the padded physical tail",
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
        "by_arm": {arm: accuracy([row for row in index if row["arm"] == arm]) for arm in ("graph", "breadth")},
        "by_partition": {partition: accuracy([row for row in index if row["partition"] == partition]) for partition in graph_base.PARTITIONS},
        "by_code": {graph_base.CODEBOOKS[code]["name"]: accuracy([row for row in index if row["code"] == code]) for code in (1, -1)},
        "by_family": {family: accuracy([row for row in index if row["family"] == family]) for family in sorted({row["family"] for row in index})},
    }


def maximum_bit_difference(field: np.ndarray, first: dict[str, Any], second: dict[str, Any], roles: tuple[str, ...]) -> float:
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
    discovery_audit = core.load(OUT / "audit/independent_c101_discovery_audit.json")
    if adapter["authorization"] != "execute_qwen_full_field_capture" or not discovery_audit["all_checks_passed"]:
        raise RuntimeError("C102 capture not authorized")
    if adapter["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("capture producer changed after freeze")
    if adapter["material_digest"] != contract["material_digest"]:
        raise RuntimeError("material changed after freeze")
    graph_rows = [{**row, "arm": "graph"} for row in core.rows(OUT / "compiled/qwen3_graph.jsonl")]
    breadth_rows = [{**row, "arm": "breadth"} for row in core.rows(OUT / "compiled/qwen3_breadth.jsonl")]
    rows = [*graph_rows, *breadth_rows]
    lengths_expected = [len(row["prompt_ids"]) for row in rows]
    total_tokens = sum(lengths_expected)
    if total_tokens != contract["storage"]["total_valid_tokens"]:
        raise RuntimeError((total_tokens, contract["storage"]["total_valid_tokens"]))
    token_starts = np.cumsum([0, *lengths_expected[:-1]], dtype=np.int64).tolist()
    raw_path = OUT / f"raw/{RAW_NAME}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
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
        width = int(contract["storage"]["fixed_physical_width"])
        batch_size = int(contract["storage"]["batch_size"])
        first_rows: list[dict[str, Any]] | None = None
        first_scores: list[list[float]] | None = None
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
                    bits = state[local, :length, :].contiguous().view(torch.uint16).cpu().numpy()
                    field[state_index, token_start:token_start + length] = bits
            batch_scores: list[list[float]] = []
            for local, row in enumerate(batch):
                row_index = start + local
                length = lengths[local]
                token_start = token_starts[row_index]
                token_ids = [int(value) for value in row["prompt_ids"]]
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                batch_scores.append(scores)
                factor_keys = ("x", "y", "branch") if row["arm"] == "graph" else ("truth_factor", "surface_factor", "distractor_factor")
                metadata_keys = ("case_id", "unit_id", "arm", "family", "world", "partition", "surface", "code", "codebook", "truth", "output_yes", "gold_position")
                metadata = {key: row[key] for key in metadata_keys}
                metadata.update({key: row[key] for key in factor_keys})
                index.append(
                    {
                        "row_index": row_index,
                        **metadata,
                        "prediction": prediction,
                        "correct": prediction == row["gold_position"],
                        "scores": scores,
                        "prompt_length": length,
                        "token_start": token_start,
                        "token_end": token_start + length,
                        "token_ids": token_ids,
                        "token_texts": tok.convert_ids_to_tokens(token_ids),
                        "role_positions": row["role_positions"],
                    }
                )
            if start == 0:
                first_rows = batch
                first_scores = batch_scores
            if (start // batch_size + 1) % 12 == 0:
                field.flush()
                print(f"[phase1583] captured {start + len(batch)}/{len(rows)} cases", flush=True)
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
                new_bits = state[local, :length, :].contiguous().view(torch.uint16).cpu().numpy()
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

    core.write_rows(OUT / "raw/qwen3_all_token_state_coordinate_index.jsonl", index)
    field = np.load(raw_path, mmap_mode="r")
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        by_unit[row["unit_id"]].append(row)
    causal_prefix = 0.0
    code_previsible = 0.0
    for unit_rows in by_unit.values():
        pre_role = "target_pre" if unit_rows[0]["arm"] == "graph" else "focus_pre"
        for row in unit_rows[1:]:
            causal_prefix = max(causal_prefix, maximum_bit_difference(field, unit_rows[0], row, (pre_role,)))
        if unit_rows[0]["arm"] == "graph":
            for x, y, branch in itertools.product((1, -1), repeat=3):
                standard = next(row for row in unit_rows if (row["x"], row["y"], row["branch"], row["code"]) == (x, y, branch, 1))
                reversed_row = next(row for row in unit_rows if (row["x"], row["y"], row["branch"], row["code"]) == (x, y, branch, -1))
                code_previsible = max(code_previsible, maximum_bit_difference(field, standard, reversed_row, ("target_post", "query_target", "query_endpoint")))
        else:
            for truth, surface, distractor in itertools.product((1, -1), repeat=3):
                standard = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, 1))
                reversed_row = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, -1))
                code_previsible = max(code_previsible, maximum_bit_difference(field, standard, reversed_row, ("focus_post", "query_focus", "query_anchor")))
    gates = contract["numeric_gates"]
    checks = {
        "shape": list(field.shape) == [STATES, total_tokens, DIM],
        "index": len(index) == 1152 and sum(row["prompt_length"] for row in index) == total_tokens,
        "finite": finite and all(math.isfinite(value) for row in index for value in row["scores"]),
        "repeat_hidden": repeat_hidden == 0.0 if gates["repeat_hidden_bitwise"] else True,
        "repeat_logits": repeat_logits <= gates["repeat_logit_max_abs"],
        "causal_prefix": causal_prefix == 0.0 if gates["causal_prefix_bitwise"] else True,
        "code_previsible": code_previsible == 0.0 if gates["code_previsible_bitwise"] else True,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "exact_bf16_archive": field.dtype == np.uint16,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "numeric": {"repeat_hidden": repeat_hidden, "repeat_logits": repeat_logits, "causal_prefix": causal_prefix, "code_previsible": code_previsible}})
    index_path = OUT / "raw/qwen3_all_token_state_coordinate_index.jsonl"
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "qwen_all_token_all_state_all_activation_coordinate_capture_complete",
        "shape": list(field.shape),
        "valid_tokens": total_tokens,
        "bytes": raw_path.stat().st_size,
        "raw_sha256": core.sha(raw_path),
        "index_sha256": core.sha(index_path),
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logit_max_abs": repeat_logits, "causal_prefix_max_abs": causal_prefix, "code_previsible_max_abs": code_previsible},
        "behavior": behavior_summary(index),
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "finished_at_utc": now(),
        "authorization": "run_phase1584_c102_staged_barcode_analysis",
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
