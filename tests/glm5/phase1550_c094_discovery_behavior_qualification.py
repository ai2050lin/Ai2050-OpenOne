#!/usr/bin/env python3
"""Phase1550: discovery-only behavior qualification for demonstrated C094 codebooks."""
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
CONTRACT = RESULT / "phase1549_c094_demonstrated_codebook_contract"
OUT = RESULT / "phase1550_c094_discovery_behavior_qualification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16


def make_batch(rows: list[dict], pad: int, device, length: int):
    ids = torch.full((len(rows), length), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for i, row in enumerate(rows):
        values = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[i, :values.numel()] = values
        mask[i, :values.numel()] = 1
        lengths.append(int(values.numel()))
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, lengths


@torch.inference_mode()
def run_batch(model, rows: list[dict], pad: int, device, length: int) -> np.ndarray:
    ids, mask, positions, lengths = make_batch(rows, pad, device, length)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
    scores = np.empty((len(rows), 2), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, candidate in enumerate(row["candidate_ids"]):
            scores[i, j] = float(output.logits[i, lengths[i] - 1, candidate[0]].float().cpu())
    return scores


def recall(rows: list[dict], truth: bool) -> float:
    subset = [row for row in rows if row["semantic_truth"] is truth]
    return sum(row["semantic_correct"] for row in subset) / len(subset)


def ba(rows: list[dict]) -> float:
    return 0.5 * (recall(rows, True) + recall(rows, False))


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1550 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1550_c094_discovery_behavior_qualification" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1549 authorization missing")
    rows = [row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl") if row["partition"] == "response_discovery"]
    groups = []
    for pair_id in dict.fromkeys(row["pair_id"] for row in rows):
        group = [row for row in rows if row["pair_id"] == pair_id]
        group.sort(key=lambda row: (protocol["surfaces"].index(row["surface"]), protocol["codebooks"].index(row["codebook"])))
        groups.append(group)
    scores = {}
    first = None
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        length = int(protocol["execution"]["fixed_global_sequence_length"])
        for index, group in enumerate(groups):
            values = run_batch(model, group, pad, device, length)
            for i, row in enumerate(group):
                scores[row["case_id"]] = values[i].tolist()
            if index == 0:
                first = (group, values)
        again = run_batch(model, first[0], pad, device, length)
        repeat_max = float(np.max(np.abs(first[1] - again)))
    finally:
        if model is not None:
            release_bf16(model)
    output = []
    for row in rows:
        values = scores[row["case_id"]]
        label_logits = {row["candidates"][i]: float(values[i]) for i in range(2)}
        prediction = max(label_logits, key=label_logits.get)
        predicted_true = prediction == ("A" if row["codebook"] == "native" else "B")
        output.append({
            **{key: row[key] for key in ("case_id", "pair_id", "pair_family", "partition", "concreteness", "surface", "codebook", "semantic_truth", "truth_sign", "codebook_sign", "answer_sign", "gold_label")},
            "candidate_logits": values,
            "label_logits": label_logits,
            "ab_margin": label_logits["A"] - label_logits["B"],
            "semantic_margin": row["codebook_sign"] * (label_logits["A"] - label_logits["B"]),
            "predicted_label": prediction,
            "predicted_semantic_truth": predicted_true,
            "emitted_correct": prediction == row["gold_label"],
            "semantic_correct": predicted_true == row["semantic_truth"],
        })
    gate = protocol["behavior_gate"]
    results, all_pass = {}, True
    for codebook in protocol["codebooks"]:
        subset = [row for row in output if row["codebook"] == codebook]
        metrics = {
            "semantic_balanced_accuracy": ba(subset),
            "semantic_true_recall": recall(subset, True),
            "semantic_false_recall": recall(subset, False),
            "surface": {surface: ba([row for row in subset if row["surface"] == surface]) for surface in protocol["surfaces"]},
        }
        metrics["qualified"] = metrics["semantic_balanced_accuracy"] >= gate["each_codebook_BA"] and all(value >= gate["each_codebook_each_surface_BA"] for value in metrics["surface"].values()) and metrics["semantic_true_recall"] >= gate["each_codebook_true_recall"] and metrics["semantic_false_recall"] >= gate["each_codebook_false_recall"]
        results[codebook] = metrics
        all_pass &= metrics["qualified"]
    raw_path = OUT / "raw/discovery_behavior_logits.jsonl"
    core.write_rows(raw_path, output)
    checks = {"coverage": len(output) == 80, "finite": all(np.isfinite(row["candidate_logits"]).all() for row in output), "repeat": repeat_max <= 1e-6, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"], "hidden_disabled": True, "semantic_emission_identity": all(row["semantic_correct"] == row["emitted_correct"] for row in output)}
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1550, "campaign": "C094", "status": "discovery_behavior_complete", "codebooks": results, "preview_both_pass": all_pass, "repeat_logit_max_abs": repeat_max, "checks": checks, "runtime": {"placement": placement, "quantization": quant}, "files": {"logits": {"sha256": core.sha(raw_path), "rows": len(output)}}, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "analysis/discovery_behavior_summary.json", report)
    core.save(OUT / "analysis/final.json", {"phase": 1550, "campaign": "C094", "status": report["status"], "authorization": "run_phase1551_c094_discovery_behavior_adjudication"})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
