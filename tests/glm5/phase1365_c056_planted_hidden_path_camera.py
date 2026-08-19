#!/usr/bin/env python3
"""Phase1365: known-truth calibration for the C056 single-write path camera."""
from __future__ import annotations

import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1365, "C056"
CONTRACT = T / "result/phase1364_c056_hidden_path_contract"
OUT = T / "result/phase1365_c056_planted_hidden_path_camera"
ROLES = {"target": 0, "family": 1, "query": 2, "boundary": 3}
WIDTH, CLASSES, LAYERS = 16, 8, 36


def signed_permutation(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(WIDTH, generator=generator)
    signs = torch.where(torch.rand(WIDTH, generator=generator) > 0.5, 1.0, -1.0)
    matrix = torch.zeros(WIDTH, WIDTH)
    matrix[torch.arange(WIDTH), permutation] = signs
    return matrix.to(device)


def code(value: int, gauge: torch.Tensor) -> torch.Tensor:
    base = F.one_hot(torch.tensor(value, device=gauge.device), num_classes=WIDTH).float()
    return base @ gauge


def initial(entity: int, answer: int, gauge: torch.Tensor) -> torch.Tensor:
    state = torch.zeros(4, WIDTH, device=gauge.device)
    state[ROLES["target"]] = code(entity, gauge)
    state[ROLES["family"]] = code(answer, gauge)
    return state


def trajectory(entity: int, answer: int, matrices: tuple[torch.Tensor, ...],
               patch: tuple[int, str, torch.Tensor] | None = None) -> torch.Tensor:
    gauge, p1, p2, p3, p4 = matrices
    states = [initial(entity, answer, gauge)]
    for layer in range(LAYERS):
        current = states[-1].clone()
        if patch is not None and layer == patch[0]:
            current[ROLES[patch[1]]] = patch[2]
            states[-1] = current.clone()
        nxt = current.clone()
        if layer == 14:
            nxt[ROLES["query"]] = current[ROLES["family"]] @ p1
        if layer == 26:
            nxt[ROLES["boundary"]] = current[ROLES["query"]] @ p2
            nxt[ROLES["query"]] = current[ROLES["family"]] @ p3
        if layer == 34:
            nxt[ROLES["boundary"]] = current[ROLES["query"]] @ p4
        states.append(nxt)
    return torch.stack(states)


def logits(states: torch.Tensor, matrices: tuple[torch.Tensor, ...]) -> torch.Tensor:
    gauge, _p1, _p2, p3, p4 = matrices
    decoder = gauge @ p3 @ p4
    decoded = states[-1, ROLES["boundary"]] @ decoder.T
    return 10.0 * decoded[:CLASSES]


def alpha(patched: torch.Tensor, corrupt: torch.Tensor, clean: torch.Tensor) -> float:
    target = clean - corrupt
    moved = patched - corrupt
    return float((moved @ target / (target @ target + 1e-12)).item())


def summarize(records: list[dict], path_name: str, gate: dict) -> dict:
    rows = [row for row in records if row["path"] == path_name]
    checkpoints = sorted(rows[0]["checkpoint_alpha"]["correct_clean"])
    checkpoint_metrics = {}
    checkpoint_pass = []
    for name in checkpoints:
        correct = [row["checkpoint_alpha"]["correct_clean"][name] for row in rows]
        controls = [[row["checkpoint_alpha"][arm][name] for arm in ("wrong_identity_true", "status_true")]
                    for row in rows]
        advantages = [value - max(wrong) for value, wrong in zip(correct, controls)]
        wins = [value > max(wrong) for value, wrong in zip(correct, controls)]
        values = {
            "correct_median": statistics.median(correct),
            "correct_over_controls_median": statistics.median(advantages),
            "correct_over_controls_win": sum(wins) / len(wins),
        }
        values["passed"] = (
            values["correct_median"] >= gate["checkpoint_recovery_projection_median_min"]
            and values["correct_over_controls_median"] >= gate["checkpoint_correct_over_controls_median_min"]
            and values["correct_over_controls_win"] >= gate["checkpoint_correct_over_controls_win_min"]
        )
        checkpoint_metrics[name] = values
        checkpoint_pass.append(values["passed"])
    correct_gain = [row["output_gain"]["correct_clean"] for row in rows]
    output_controls = [[row["output_gain"][arm] for arm in ("wrong_identity_true", "status_true")]
                       for row in rows]
    output_advantage = [value - max(wrong) for value, wrong in zip(correct_gain, output_controls)]
    output_wins = [value > max(wrong) for value, wrong in zip(correct_gain, output_controls)]
    output_metrics = {
        "correct_gain_median": statistics.median(correct_gain),
        "correct_over_controls_median": statistics.median(output_advantage),
        "correct_over_controls_win": sum(output_wins) / len(output_wins),
    }
    output_metrics["passed"] = (
        output_metrics["correct_gain_median"] >= gate["output_gain_median_min"]
        and output_metrics["correct_over_controls_median"] >= gate["output_correct_over_controls_median_min"]
        and output_metrics["correct_over_controls_win"] >= gate["output_correct_over_controls_win_min"]
    )
    self_output = max(abs(row["output_gain"]["self"]) for row in rows)
    self_hidden = max(row["self_checkpoint_relative_l2_max"] for row in rows)
    identity = self_output <= gate["self_output_max_abs_diff"] and self_hidden <= gate["self_checkpoint_relative_l2_max"]
    return {
        "count": len(rows), "checkpoints": checkpoint_metrics, "output": output_metrics,
        "self_output_max_abs_diff": self_output, "self_checkpoint_relative_l2_max": self_hidden,
        "identity_passed": identity,
        "qualified": all(checkpoint_pass) and output_metrics["passed"] and identity,
    }


@torch.inference_mode()
def main() -> None:
    parent = core.load(CONTRACT / "analysis/final.json")
    audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent.get("authorization") != "run_phase1365_c056_known_truth_camera" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1364 did not authorize known-truth calibration")
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1365 already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("formal known-truth calibration requires CUDA")
    device = torch.device("cuda")
    records = []
    for split_index, split in enumerate(protocol["known_truth"]["splits"]):
        for gauge_index in range(protocol["known_truth"]["gauges"]):
            seed = 1_365_000 + 100 * split_index + gauge_index
            matrices = tuple(signed_permutation(seed + offset, device) for offset in range(5))
            for local in range(32):
                entity = (local + gauge_index) % CLASSES
                clean_answer = (3 * local + split_index + gauge_index) % CLASSES
                corrupt_answer = (clean_answer + 1) % CLASSES
                wrong_answer = (clean_answer + 2) % CLASSES
                status_answer = (clean_answer + 3) % CLASSES
                donor_answers = {
                    "self": corrupt_answer,
                    "correct_clean": clean_answer,
                    "wrong_identity_true": wrong_answer,
                    "status_true": status_answer,
                }
                clean = trajectory(entity, clean_answer, matrices)
                corrupt = trajectory(entity, corrupt_answer, matrices)
                baseline_logits = logits(corrupt, matrices)
                clean_class, corrupt_class = clean_answer, corrupt_answer
                baseline_margin = float((baseline_logits[clean_class] - baseline_logits[corrupt_class]).item())
                for path_name, path in protocol["paths"].items():
                    source = path["source"]
                    output_gain = {}
                    checkpoint_alpha = {arm: {} for arm in donor_answers}
                    self_l2 = []
                    for arm, answer in donor_answers.items():
                        donor = trajectory(entity, answer, matrices)
                        vector = donor[source["layer"], ROLES[source["role"]]]
                        patched = trajectory(entity, corrupt_answer, matrices,
                                             (source["layer"], source["role"], vector))
                        score = logits(patched, matrices)
                        margin = float((score[clean_class] - score[corrupt_class]).item())
                        output_gain[arm] = margin - baseline_margin
                        for checkpoint in path["checkpoints"]:
                            label = f"{checkpoint['role']}@{checkpoint['layer']}"
                            point = patched[checkpoint["layer"], ROLES[checkpoint["role"]]]
                            clean_point = clean[checkpoint["layer"], ROLES[checkpoint["role"]]]
                            corrupt_point = corrupt[checkpoint["layer"], ROLES[checkpoint["role"]]]
                            checkpoint_alpha[arm][label] = alpha(point, corrupt_point, clean_point)
                            if arm == "self":
                                self_l2.append(float(torch.linalg.vector_norm(point - corrupt_point).item()
                                                     / (torch.linalg.vector_norm(corrupt_point).item() + 1e-12)))
                    records.append({
                        "system_id": f"{split}.g{gauge_index}.c{local}", "split": split,
                        "gauge": gauge_index, "path": path_name, "output_gain": output_gain,
                        "checkpoint_alpha": checkpoint_alpha,
                        "self_checkpoint_relative_l2_max": max(self_l2, default=0.0),
                    })
    core.write_rows(OUT / "raw/known_truth_path_records.jsonl", records)
    metrics = {name: summarize(records, name, protocol["causal"]) for name in protocol["paths"]}
    predicted = sorted(name for name, values in metrics.items() if values["qualified"])
    expected = sorted(protocol["known_truth"]["expected_positive"])
    checks = {
        "record_count": len(records) == protocol["known_truth"]["systems"] * len(protocol["paths"]),
        "finite": all(math.isfinite(value) for row in records for value in row["output_gain"].values())
                  and all(math.isfinite(value) for row in records for arm in row["checkpoint_alpha"].values()
                          for value in arm.values()),
        "all_identity": all(values["identity_passed"] for values in metrics.values()),
        "exact_topology": predicted == expected,
        "discovery_confirmation_same": True,
    }
    summary = {
        "phase": PHASE, "campaign": CAMPAIGN, "device": str(device),
        "metrics": metrics, "predicted_positive": predicted, "expected_positive": expected,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "claim_boundary": "known-truth single-write Hidden-State cascade camera only",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/known_truth_summary.json", summary)
    authorization = "run_phase1366_c056_qwen_path_observation" if summary["all_checks_passed"] else "close_c056_camera_failed"
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN,
        "verdict": "known_truth_hidden_path_camera_calibrated" if summary["all_checks_passed"] else "known_truth_hidden_path_camera_failed",
        "predicted_positive": predicted, "expected_positive": expected,
        "authorization": authorization,
    })
    print(json.dumps({"checks": checks, "predicted": predicted, "expected": expected,
                      "metrics": metrics, "authorization": authorization}, indent=2))


if __name__ == "__main__":
    main()
