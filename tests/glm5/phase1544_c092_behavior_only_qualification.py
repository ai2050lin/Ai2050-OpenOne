#!/usr/bin/env python3
"""Phase1544: behavior-only qualification for C092; hidden states remain disabled."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1543_c092_truth_output_code_factorial_contract"
OUT = RESULT / "phase1544_c092_behavior_only_qualification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16


def make_batch(rows: list[dict], pad: int, device, fixed_length: int):
    ids = torch.full((len(rows), fixed_length), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for i, row in enumerate(rows):
        values = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        length = int(values.numel())
        ids[i, :length] = values
        mask[i, :length] = 1
        lengths.append(length)
    positions = mask.cumsum(dim=-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, lengths


@torch.inference_mode()
def run_batch(model, rows: list[dict], pad: int, device, fixed_length: int) -> np.ndarray:
    ids, mask, positions, lengths = make_batch(rows, pad, device, fixed_length)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
    values = np.empty((len(rows), 2), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, candidate in enumerate(row["candidate_ids"]):
            values[i, j] = float(output.logits[i, lengths[i] - 1, candidate[0]].float().cpu())
    return values


def recall(rows: list[dict], truth: bool) -> float:
    subset = [row for row in rows if row["semantic_truth"] is truth]
    return sum(row["semantic_correct"] for row in subset) / len(subset)


def balanced_accuracy(rows: list[dict]) -> float:
    return 0.5 * (recall(rows, True) + recall(rows, False))


def metrics(rows: list[dict], surfaces: list[str]) -> dict:
    return {
        "n": len(rows),
        "emitted_label_accuracy": sum(row["emitted_correct"] for row in rows) / len(rows),
        "semantic_balanced_accuracy": balanced_accuracy(rows),
        "semantic_true_recall": recall(rows, True),
        "semantic_false_recall": recall(rows, False),
        "surface_semantic_balanced_accuracy": {surface: balanced_accuracy([row for row in rows if row["surface"] == surface]) for surface in surfaces},
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1544 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1544_c092_behavior_only_qualification" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1543 authorization missing")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    groups = []
    for pair_id in dict.fromkeys(row["pair_id"] for row in compiled):
        rows = [row for row in compiled if row["pair_id"] == pair_id]
        rows.sort(key=lambda row: (protocol["surfaces"].index(row["surface"]), protocol["codebooks"].index(row["codebook"])))
        if len(rows) != 4:
            raise RuntimeError((pair_id, len(rows)))
        groups.append(rows)
    scores = {}
    repeated = []
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        fixed_length = int(protocol["execution"]["fixed_global_sequence_length"])
        for group_index, rows in enumerate(groups):
            values = run_batch(model, rows, pad, device, fixed_length)
            for i, row in enumerate(rows):
                scores[row["case_id"]] = values[i].tolist()
            if group_index < 3:
                repeated.append((rows, values))
            if (group_index + 1) % 15 == 0:
                print(f"[phase1544] completed {group_index + 1}/{len(groups)} pair batches", flush=True)
        repeat_max = 0.0
        for rows, values in repeated:
            again = run_batch(model, rows, pad, device, fixed_length)
            repeat_max = max(repeat_max, float(np.max(np.abs(values - again))))
    finally:
        if model is not None:
            release_bf16(model)

    output_rows = []
    for row in compiled:
        values = scores[row["case_id"]]
        label_logits = {row["candidates"][i]: float(values[i]) for i in range(2)}
        prediction = "是" if label_logits["是"] >= label_logits["否"] else "否"
        semantic_prediction = prediction == ("是" if row["codebook"] == "native" else "否")
        output_rows.append({
            **{key: row[key] for key in ("case_id", "pair_id", "pair_family", "partition", "concreteness", "surface", "codebook", "semantic_truth", "truth_sign", "codebook_sign", "answer_sign", "gold_label")},
            "candidate_logits": values,
            "label_logits": label_logits,
            "yes_no_margin": label_logits["是"] - label_logits["否"],
            "semantic_margin": row["codebook_sign"] * (label_logits["是"] - label_logits["否"]),
            "predicted_label": prediction,
            "predicted_semantic_truth": semantic_prediction,
            "emitted_correct": prediction == row["gold_label"],
            "semantic_correct": semantic_prediction == row["semantic_truth"],
        })
    gate = protocol["behavior_gate"]
    summaries = {}
    all_qualified = True
    for codebook in protocol["codebooks"]:
        all_rows = [row for row in output_rows if row["codebook"] == codebook]
        discovery = [row for row in all_rows if row["partition"] == "response_discovery"]
        summary = {"all": metrics(all_rows, protocol["surfaces"]), "discovery": metrics(discovery, protocol["surfaces"])}
        current = summary["discovery"]
        qualified = (
            current["semantic_balanced_accuracy"] >= gate["discovery_each_codebook_semantic_balanced_accuracy"]
            and all(value >= gate["discovery_each_codebook_each_surface_semantic_balanced_accuracy"] for value in current["surface_semantic_balanced_accuracy"].values())
            and current["semantic_true_recall"] >= gate["discovery_each_codebook_true_recall"]
            and current["semantic_false_recall"] >= gate["discovery_each_codebook_false_recall"]
        )
        summary["preview_qualified"] = qualified
        summaries[codebook] = summary
        all_qualified &= qualified
    raw_path = OUT / "raw/behavior_logits.jsonl"
    core.write_rows(raw_path, output_rows)
    checks = {
        "coverage": len(output_rows) == 240 and len(scores) == 240,
        "finite": all(np.isfinite(row["candidate_logits"]).all() for row in output_rows),
        "repeat_logits": repeat_max <= protocol["numeric_gate"]["repeat_logits_max_abs"],
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "hidden_disabled": True,
        "semantic_emission_identity": all(row["semantic_correct"] == row["emitted_correct"] for row in output_rows),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1544,
        "campaign": "C092",
        "status": "behavior_only_complete",
        "codebooks": summaries,
        "preview_both_codebooks_qualified": all_qualified,
        "repeat_logit_max_abs": repeat_max,
        "checks": checks,
        "runtime": {"placement": placement, "quantization": quant},
        "files": {"behavior_logits": {"sha256": core.sha(raw_path), "rows": len(output_rows)}},
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/behavior_summary.json", report)
    core.save(OUT / "analysis/final.json", {"phase": 1544, "campaign": "C092", "status": report["status"], "authorization": "run_phase1545_c092_behavior_gate_adjudication"})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
