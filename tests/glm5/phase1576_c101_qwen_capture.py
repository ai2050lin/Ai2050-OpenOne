#!/usr/bin/env python3
"""Phase1576 / C101: correct C099 boundary behavior and capture both C101 arms."""
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
RESULT = TESTS / "result"
OUT = RESULT / "phase1575_c101_dual_arm"
C099 = RESULT / "phase1572_c099_fixed_width_graph_field_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

PHASE = 1576
CAMPAIGN = "C101"
STATES = 37
DIM = 2560


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/independent_pre_model_audit.json")
    if protocol["authorization"] != "run_phase1576_c101_qwen_capture" or not pre["all_checks_passed"]:
        raise RuntimeError("C101 capture authorization missing")
    adapter = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "capture_adapter_frozen",
        "producer_sha256": core.sha(Path(__file__)),
        "parent_material_digest": protocol["material_digest"],
        "fixed_global_sequence_length": protocol["storage"]["fixed_global_sequence_length"],
        "behavior_boundary": "real token boundary via base-model final Hidden State and lm_head",
        "raw_scope": protocol["storage"]["raw"],
        "authorization": "execute_qwen_capture",
    }
    core.save(OUT / "protocol/capture_adapter.json", adapter)
    print(json.dumps(adapter, indent=2))


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


def behavior_summary(index: list[dict[str, Any]]) -> dict[str, Any]:
    def acc(rows: list[dict[str, Any]]) -> float:
        return float(np.mean([row["correct"] for row in rows])) if rows else 0.0

    values: dict[str, Any] = {
        "global_accuracy": acc(index),
        "global_balanced_accuracy": graph_base.ba([row["output_yes"] for row in index], [row["prediction"] == 0 for row in index]),
        "by_arm": {arm: acc([row for row in index if row["arm"] == arm]) for arm in ("confirmation", "breadth")},
        "by_partition": {p: acc([row for row in index if row["partition"] == p]) for p in graph_base.PARTITIONS},
        "by_code": {graph_base.CODEBOOKS[c]["name"]: acc([row for row in index if row["code"] == c]) for c in (1, -1)},
    }
    values["by_family"] = {family: acc([row for row in index if row["family"] == family]) for family in sorted({row["family"] for row in index})}
    return values


@torch.inference_mode()
def recalibrate_c099(model: Any, pad: int, device: torch.device) -> dict[str, Any]:
    compiled = core.rows(C099 / "compiled/qwen3_active.jsonl")
    old_index = core.rows(C099 / "raw/all_token_field_index.jsonl")
    stored = np.load(C099 / "raw/all_token_all_state_field.float16.npy", mmap_mode="r")
    corrected: list[dict[str, Any]] = []
    hidden_identity = 0.0
    old_score_difference = 0.0
    repeat_hidden = 0.0
    repeat_logits = 0.0
    first = None
    for start in range(0, len(compiled), 8):
        batch = compiled[start:start + 8]
        output, logits, ids, mask, positions, lengths = model_forward(model, batch, pad, device, 210, False)
        fresh_boundaries = output.last_hidden_state[torch.arange(len(batch), device=device), torch.tensor([n - 1 for n in lengths], device=device)]
        if start == 0:
            first = (batch, fresh_boundaries.to(torch.float16).cpu().numpy().copy(), logits.cpu().numpy().copy())
        for local, row in enumerate(batch):
            old = old_index[start + local]
            point = old["token_start"] + old["role_positions"]["boundary"][0]
            archived = np.asarray(stored[-1, point], dtype=np.float32)
            fresh = fresh_boundaries[local].to(torch.float16).float().cpu().numpy()
            hidden_identity = max(hidden_identity, float(np.max(np.abs(archived - fresh))))
            scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
            old_score_difference = max(old_score_difference, max(abs(scores[i] - old["scores"][i]) for i in range(2)))
            prediction = int(scores[1] > scores[0])
            corrected.append({
                "case_id": row["case_id"],
                "partition": row["partition"],
                "family": row["family"],
                "world": row["world"],
                "code": row["code"],
                "output_yes": row["output_yes"],
                "gold_position": row["gold_position"],
                "prediction": prediction,
                "correct": prediction == row["gold_position"],
                "corrected_scores": scores,
                "old_physical_tail_scores": old["scores"],
            })
        del output, logits, ids, mask, positions, fresh_boundaries
    if first is None:
        raise RuntimeError("C099 calibration repeat missing")
    batch, old_hidden, old_logits = first
    output, logits, ids, mask, positions, lengths = model_forward(model, batch, pad, device, 210, False)
    again = output.last_hidden_state[torch.arange(len(batch), device=device), torch.tensor([n - 1 for n in lengths], device=device)].to(torch.float16).cpu().numpy()
    repeat_hidden = float(np.max(np.abs(again.astype(np.float32) - old_hidden.astype(np.float32))))
    repeat_logits = float(np.max(np.abs(logits.cpu().numpy() - old_logits)))
    old_accuracy = float(np.mean([row["correct"] for row in old_index]))
    new_accuracy = float(np.mean([row["correct"] for row in corrected]))
    report = {
        "status": "C099_real_boundary_behavior_recalibrated",
        "old_accuracy": old_accuracy,
        "corrected_accuracy": new_accuracy,
        "corrected_balanced_accuracy": graph_base.ba([row["output_yes"] for row in corrected], [row["prediction"] == 0 for row in corrected]),
        "by_world": {w: float(np.mean([r["correct"] for r in corrected if r["world"] == w])) for w in graph_base.WORLDS},
        "by_family": {f: float(np.mean([r["correct"] for r in corrected if r["family"] == f])) for f in graph_base.FAMILIES},
        "by_code": {graph_base.CODEBOOKS[c]["name"]: float(np.mean([r["correct"] for r in corrected if r["code"] == c])) for c in (1, -1)},
        "archived_final_hidden_max_abs": hidden_identity,
        "old_vs_corrected_score_max_abs": old_score_difference,
        "repeat_hidden_max_abs": repeat_hidden,
        "repeat_logits_max_abs": repeat_logits,
        "interpretation": "old logits addressed the padded physical tail for shorter rows; C099 Hidden State role positions remain valid",
    }
    core.write_rows(OUT / "analysis/c099_corrected_behavior_rows.jsonl", corrected)
    core.save(OUT / "analysis/c099_behavior_boundary_recalibration.json", report)
    del stored
    return report


def storage_layout(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    layout = []
    cursor = 0
    for row in rows:
        offsets = {}
        for role, positions in row["role_positions"].items():
            offsets[role] = [cursor, cursor + len(positions)]
            cursor += len(positions)
        layout.append({"role_offsets": offsets})
    return layout, cursor


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    adapter = core.load(OUT / "protocol/capture_adapter.json")
    pre = core.load(OUT / "audit/independent_pre_model_audit.json")
    if adapter["authorization"] != "execute_qwen_capture" or not pre["all_checks_passed"]:
        raise RuntimeError("capture not authorized")
    if adapter["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("capture producer changed after freeze")
    if adapter["parent_material_digest"] != protocol["material_digest"]:
        raise RuntimeError("material changed")
    confirmation = [{**row, "arm": "confirmation"} for row in core.rows(OUT / "compiled/qwen3_confirmation.jsonl")]
    breadth = [{**row, "arm": "breadth"} for row in core.rows(OUT / "compiled/qwen3_breadth.jsonl")]
    rows = [*confirmation, *breadth]
    layout, total_role_tokens = storage_layout(rows)
    raw_path = OUT / "raw/qwen3_registered_role_field.float16.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(STATES, total_role_tokens, DIM))
    index: list[dict[str, Any]] = []
    first_repeat = None
    finite = True
    model = None
    calibration = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        calibration = recalibrate_c099(model, pad, device)
        width = int(protocol["storage"]["fixed_global_sequence_length"])
        batch_size = int(protocol["storage"]["batch_size"])
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            output, logits, ids, mask, positions, lengths = model_forward(model, batch, pad, device, width, True)
            if len(output.hidden_states) != STATES or output.hidden_states[-1].shape[-1] != DIM:
                raise RuntimeError((len(output.hidden_states), output.hidden_states[-1].shape))
            repeat_blocks = []
            repeat_scores = []
            for local, row in enumerate(batch):
                offsets = layout[start + local]["role_offsets"]
                blocks = {}
                for role, token_positions in row["role_positions"].items():
                    block = torch.stack([state[local, token_positions, :] for state in output.hidden_states], dim=0)
                    finite = finite and bool(torch.isfinite(block).all())
                    cpu = block.to(dtype=torch.float16, device="cpu").numpy()
                    left, right = offsets[role]
                    field[:, left:right, :] = cpu
                    blocks[role] = cpu.copy() if start == 0 else None
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                meta_keys = ["case_id", "unit_id", "arm", "family", "world", "partition", "surface", "code", "codebook", "truth", "output_yes", "gold_position"]
                meta = {key: row[key] for key in meta_keys}
                if row["arm"] == "confirmation":
                    meta.update({key: row[key] for key in ("x", "y", "branch")})
                else:
                    meta.update({key: row[key] for key in ("truth_factor", "surface_factor", "distractor_factor")})
                index.append({
                    "row_index": start + local,
                    **meta,
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                    "scores": scores,
                    "prompt_length": lengths[local],
                    "role_positions": row["role_positions"],
                    "role_offsets": offsets,
                })
                if start == 0:
                    repeat_blocks.append(blocks)
                    repeat_scores.append(scores)
            if start == 0:
                first_repeat = (batch, repeat_blocks, repeat_scores)
            if (start // batch_size + 1) % 24 == 0:
                print(f"[phase1576] captured {start + len(batch)}/{len(rows)} cases", flush=True)
            del output, logits, ids, mask, positions
        field.flush()
        if first_repeat is None:
            raise RuntimeError("C101 repeat missing")
        batch, old_blocks, old_scores = first_repeat
        output, logits, ids, mask, positions, lengths = model_forward(model, batch, pad, device, width, True)
        repeat_hidden = 0.0
        repeat_logits = 0.0
        for local, row in enumerate(batch):
            for role, token_positions in row["role_positions"].items():
                again = torch.stack([state[local, token_positions, :] for state in output.hidden_states], dim=0).to(torch.float16).cpu().numpy()
                repeat_hidden = max(repeat_hidden, float(np.max(np.abs(again.astype(np.float32) - old_blocks[local][role].astype(np.float32)))))
            scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
            repeat_logits = max(repeat_logits, max(abs(scores[i] - old_scores[local][i]) for i in range(2)))
    finally:
        field.flush()
        del field
        if model is not None:
            release_bf16(model)
        gc.collect()
    if calibration is None:
        raise RuntimeError("calibration missing")
    core.write_rows(OUT / "raw/qwen3_registered_role_index.jsonl", index)
    field = np.load(raw_path, mmap_mode="r")
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        by_unit[row["unit_id"]].append(row)
    causal_prefix = 0.0
    code_previsible = 0.0
    for unit_rows in by_unit.values():
        pre_role = "target_pre" if unit_rows[0]["arm"] == "confirmation" else "focus_pre"
        ref_left, ref_right = unit_rows[0]["role_offsets"][pre_role]
        ref = np.asarray(field[:, ref_left:ref_right], dtype=np.float32)
        for row in unit_rows[1:]:
            left, right = row["role_offsets"][pre_role]
            value = np.asarray(field[:, left:right], dtype=np.float32)
            if value.shape != ref.shape:
                raise RuntimeError((row["case_id"], value.shape, ref.shape))
            causal_prefix = max(causal_prefix, float(np.max(np.abs(value - ref))))
        if unit_rows[0]["arm"] == "confirmation":
            for x, y, branch in itertools.product((1, -1), repeat=3):
                left_row = next(r for r in unit_rows if (r["x"], r["y"], r["branch"], r["code"]) == (x, y, branch, 1))
                right_row = next(r for r in unit_rows if (r["x"], r["y"], r["branch"], r["code"]) == (x, y, branch, -1))
                for role in ("target_post", "query_target", "query_endpoint"):
                    l0, l1 = left_row["role_offsets"][role]
                    r0, r1 = right_row["role_offsets"][role]
                    code_previsible = max(code_previsible, float(np.max(np.abs(np.asarray(field[:, l0:l1], dtype=np.float32) - np.asarray(field[:, r0:r1], dtype=np.float32)))))
        else:
            for truth, surface, distractor in itertools.product((1, -1), repeat=3):
                left_row = next(r for r in unit_rows if (r["truth_factor"], r["surface_factor"], r["distractor_factor"], r["code"]) == (truth, surface, distractor, 1))
                right_row = next(r for r in unit_rows if (r["truth_factor"], r["surface_factor"], r["distractor_factor"], r["code"]) == (truth, surface, distractor, -1))
                for role in ("focus_post", "query_focus", "query_anchor"):
                    l0, l1 = left_row["role_offsets"][role]
                    r0, r1 = right_row["role_offsets"][role]
                    code_previsible = max(code_previsible, float(np.max(np.abs(np.asarray(field[:, l0:l1], dtype=np.float32) - np.asarray(field[:, r0:r1], dtype=np.float32)))))
    gates = protocol["numeric_gates"]
    checks = {
        "shape": list(field.shape) == [STATES, total_role_tokens, DIM],
        "index": len(index) == 1920,
        "finite": finite and all(math.isfinite(v) for row in index for v in row["scores"]),
        "repeat_hidden": repeat_hidden <= gates["repeat_hidden_max_abs"],
        "repeat_logits": repeat_logits <= gates["repeat_logit_max_abs"],
        "causal_prefix": causal_prefix <= gates["causal_prefix_max_abs"],
        "code_previsible": code_previsible <= gates["code_previsible_max_abs"],
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "behavior_recalibration_identity": calibration["archived_final_hidden_max_abs"] == 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "qwen_real_boundary_and_registered_role_capture_complete",
        "shape": list(field.shape),
        "total_role_tokens": total_role_tokens,
        "bytes": raw_path.stat().st_size,
        "raw_sha256": core.sha(raw_path),
        "index_sha256": core.sha(OUT / "raw/qwen3_registered_role_index.jsonl"),
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logit_max_abs": repeat_logits, "causal_prefix_max_abs": causal_prefix, "code_previsible_max_abs": code_previsible},
        "behavior": behavior_summary(index),
        "c099_recalibration": calibration,
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "finished_at_utc": now(),
        "authorization": "run_phase1577_c101_analysis",
    }
    core.save(OUT / "analysis/qwen_capture_summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "runtime"}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "capture"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else capture()


if __name__ == "__main__":
    main()
