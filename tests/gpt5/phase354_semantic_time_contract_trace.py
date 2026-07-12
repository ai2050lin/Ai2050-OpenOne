#!/usr/bin/env python3
"""Trace qualified contracts under teacher-forced and free-rollout semantic time."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase334_natural_contrast_survey import role_positions  # noqa: E402
from phase338_block_causal_screen import continuation_ids, get_layers, prompt_ids  # noqa: E402
from phase347_three_core_natural_trace import depth_lookup, install_capture_hooks  # noqa: E402
from phase351_signed_paired_trace import unit_direction  # noqa: E402
from phase354_semantic_time_case_bank import (  # noqa: E402
    MODELS, OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


SOURCE353 = ROOT / "tests/gpt5/result/phase353_family_contracts/family_specific_contract_compiler"
ROLES = ("source", "query", "answer_start", "current_generation")
MAX_STEPS = 4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def trim_generated(ids: list[int], eos: int) -> list[int]:
    result = []
    for token_id in ids:
        result.append(int(token_id))
        if int(token_id) == eos:
            break
    return result[:MAX_STEPS]


def semantic_phase(step: int, count: int) -> str:
    if count <= 1:
        return "single"
    if step == 0:
        return "first"
    if step == count - 1:
        return "final"
    return "middle"


def natural_target_surface(case: dict[str, Any], rollout: dict[str, Any]) -> str:
    target = case["target"]
    head = rollout["answer_head_text"].lstrip()
    if rollout["semantic_correct"] and head and head[0].isupper() and target:
        target = target[0].upper() + target[1:]
    return f" {target}"


@torch.inference_mode()
def run_model(model: str, round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    cases = [row for row in read_jsonl(root / "phase354_registered_cases.jsonl") if row["model"] == model]
    rollouts = {
        row["case_id"]: row
        for row in read_jsonl(SOURCE353 / "models" / model / "phase353_rollout_rows.jsonl")
    }
    loaded = None
    handles: list[Any] = []
    rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        depths = depth_lookup(len(layers))
        output_weight = loaded.model.get_output_embeddings().weight.detach()
        eos = int(loaded.tokenizer.eos_token_id)
        state: dict[str, Any] = {"captures": [], "positions": None, "pre_inputs": {}}
        handles = install_capture_hooks(loaded, state)
        for case_index, case in enumerate(cases, 1):
            prompt = prompt_ids(loaded, case)
            role_map = role_positions(loaded, case, prompt)
            rollout = rollouts[case["case_id"]]
            target_surface = natural_target_surface(case, rollout)
            target_ids = continuation_ids(loaded, case, target_surface)[:MAX_STEPS]
            title_distractors = target_surface[1:2].isupper()
            distractor_surfaces = [
                f" {value[0].upper() + value[1:] if title_distractors and value else value}"
                for value in case["distractors"]
            ]
            distractor_ids = [continuation_ids(loaded, case, value)[:MAX_STEPS] for value in distractor_surfaces]
            free_ids = trim_generated(rollout["generated_token_ids"], eos)
            modes = {"teacher_forced": target_ids, "free_rollout": free_ids}
            first_divergence = next((step for step in range(min(len(free_ids), len(target_ids))) if free_ids[step] != target_ids[step]), None)
            if first_divergence is None and len(free_ids) < len(target_ids):
                first_divergence = len(free_ids)
            for mode, trajectory in modes.items():
                for step in range(len(trajectory)):
                    prefix = target_ids[:step] if mode == "teacher_forced" else free_ids[:step]
                    sequence = prompt + prefix
                    positions = [role_map[role][0] for role in ROLES[:3]] + [len(sequence) - 1]
                    state["positions"] = torch.tensor(positions, dtype=torch.long, device=loaded.input_device)
                    state["captures"] = []
                    state["pre_inputs"] = {}
                    expected_id = target_ids[step] if step < len(target_ids) else eos
                    actual_id = trajectory[step]
                    target_direction = unit_direction(output_weight, expected_id)
                    competitor_tokens = [values[step] if step < len(values) else eos for values in distractor_ids]
                    competitor_directions = torch.stack([unit_direction(output_weight, value) for value in competitor_tokens])
                    input_ids = torch.tensor([sequence], dtype=torch.long, device=loaded.input_device)
                    output = loaded.model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                        return_dict=True,
                    )
                    rho = step / max(len(trajectory) - 1, 1)
                    for layer_index, component, captured in state["captures"]:
                        vectors = captured.float()
                        norms = vectors.norm(dim=-1).clamp_min(1e-8)
                        target_cosine = torch.mv(vectors, target_direction.to(vectors.device)) / norms
                        competitors = torch.mm(vectors, competitor_directions.to(vectors.device).T) / norms[:, None]
                        best_competitor = competitors.max(dim=-1).values
                        margins = target_cosine - best_competitor
                        for role_index, role in enumerate(ROLES):
                            values = (norms[role_index], target_cosine[role_index], best_competitor[role_index], margins[role_index])
                            finite = all(torch.isfinite(value).item() for value in values)
                            rows.append({
                                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                                "model": model, "case_id": case["case_id"],
                                "contract_group_id": case["contract_group_id"],
                                "family_id": case["family_id"], "mechanism_id": case["mechanism_id"],
                                "split": case["split"], "template_id": case["template_id"],
                                "contrast_condition": case["contrast_condition"],
                                "operation_demanded": case["operation_demanded"],
                                "trajectory_mode": mode, "semantic_step": step,
                                "semantic_step_count": len(trajectory), "semantic_time_rho": round(rho, 7),
                                "semantic_phase": semantic_phase(step, len(trajectory)),
                                "expected_token_id": int(expected_id), "actual_token_id": int(actual_id),
                                "token_matches_expected": actual_id == expected_id,
                                "free_first_divergence_step": first_divergence,
                                "component": component, "layer_index": layer_index,
                                "depth_bin": depths[layer_index], "position_role": role,
                                "component_l2_norm": round(float(values[0].item()), 7) if finite else None,
                                "signed_target_cosine": round(float(values[1].item()), 7) if finite else None,
                                "signed_best_competitor_cosine": round(float(values[2].item()), 7) if finite else None,
                                "signed_competition_margin": round(float(values[3].item()), 7) if finite else None,
                                "finite": finite, "natural_trace_only": True,
                                "physical_heldout": False, "causal_sealed": False,
                                "internal_intervention": False, "single_unit_causal": False,
                            })
                    del output, input_ids
            case_rows.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "case_id": case["case_id"], "family_id": case["family_id"],
                "mechanism_id": case["mechanism_id"], "split": case["split"],
                "target_step_count": len(target_ids), "free_step_count": len(free_ids),
                "free_first_divergence_step": first_divergence,
                "free_exact_prefix_match": first_divergence is None,
                "target_surface": target_surface,
                "rollout_semantic_correct": rollout["semantic_correct"],
            })
            if case_index % 32 == 0 or case_index == len(cases):
                print(f"[{model}] {case_index}/{len(cases)}", flush=True)
        model_root = root / "models" / model
        write_jsonl(model_root / "phase354_case_rows.jsonl", case_rows)
        write_jsonl(model_root / "phase354_semantic_time_rows.jsonl", rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "registered_case_count": len(cases), "case_row_count": len(case_rows),
            "trace_row_count": len(rows),
            "teacher_trace_row_count": sum(row["trajectory_mode"] == "teacher_forced" for row in rows),
            "free_trace_row_count": sum(row["trajectory_mode"] == "free_rollout" for row in rows),
            "nonfinite_trace_row_count": sum(not row["finite"] for row in rows),
            "physical_heldout_trace_count": sum(row["physical_heldout"] for row in rows),
            "causal_sealed_trace_count": sum(row["causal_sealed"] for row in rows),
            "actual_model_batch_size": 1,
            "valid": len(cases) == 192 and len(case_rows) == 192 and bool(rows) and all(row["finite"] for row in rows),
        }
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run_model(args.model, args.round), ensure_ascii=False, indent=2))
