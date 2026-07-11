#!/usr/bin/env python3
"""Run Phase348 phrase scoring through true batch-one model calls."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase334_natural_contrast_survey import role_positions  # noqa: E402
from phase338_block_causal_screen import (  # noqa: E402
    continuation_ids, install_block_hooks, prompt_ids, wrong_block,
)
from phase348_adjusted_block_screen_case_bank import (  # noqa: E402
    OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


MODELS = ("qwen3", "glm4", "deepseek7b")
CONDITIONS = ("baseline", "correct_zero", "correct_half", "wrong_depth_zero", "wrong_position_zero")


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


@torch.inference_mode()
def option_score_b1(
    loaded: Any, case: dict[str, Any], value: str,
    block: dict[str, Any] | None, mode: str | None,
) -> float:
    prompt = prompt_ids(loaded, case)
    continuation = continuation_ids(loaded, case, value)
    sequence = prompt + continuation
    input_ids = torch.tensor([sequence], dtype=torch.long, device=loaded.input_device)
    attention_mask = torch.ones_like(input_ids)
    role_map = role_positions(loaded, case, prompt)
    positions = [role_map[block["position_role"]][0]] if block else [len(prompt) - 1]
    handles = install_block_hooks(loaded, block, positions, mode) if block and mode else []
    try:
        output = loaded.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    log_probs = torch.log_softmax(output.logits[0].detach().float(), dim=-1)
    values = [
        float(log_probs[len(prompt) + offset - 1, token_id].item())
        for offset, token_id in enumerate(continuation)
    ]
    del output, log_probs, input_ids, attention_mask
    return mean(values)


def score_case_b1(
    loaded: Any, case: dict[str, Any], block: dict[str, Any] | None, mode: str | None,
) -> dict[str, float]:
    values = [case["target"], *case["distractors"]]
    scores = [option_score_b1(loaded, case, value, block, mode) for value in values]
    distractor = max(scores[1:])
    return {
        "target_phrase_mean_logprob": scores[0],
        "best_distractor_phrase_mean_logprob": distractor,
        "phrase_margin": scores[0] - distractor,
    }


def run_model(model: str, round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    cases = [
        row for row in read_jsonl(root / "phase348_registered_cases.jsonl")
        if row["model"] == model and row["split"] in {"discovery", "calibration"}
    ]
    frozen = [row for row in read_jsonl(root / "phase348_frozen_blocks.jsonl") if row["model"] == model]
    block = {key: frozen[0][key] for key in ("block_id", "component", "depth_bin", "position_role")}
    wrong_depth = wrong_block(block, "depth")
    wrong_position = wrong_block(block, "position")
    specs = {
        "baseline": (None, None), "correct_zero": (block, "zero"),
        "correct_half": (block, "half"),
        "wrong_depth_zero": (wrong_depth, "zero"),
        "wrong_position_zero": (wrong_position, "zero"),
    }
    rows = []
    loaded = None
    try:
        loaded = load_probe_model(model)
        for index, case in enumerate(cases, 1):
            baseline = score_case_b1(loaded, case, None, None)
            for condition in CONDITIONS:
                selected, mode = specs[condition]
                score = baseline if condition == "baseline" else score_case_b1(loaded, case, selected, mode)
                valid = all(math.isfinite(value) for value in score.values())
                rows.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                    "model": model, "case_id": case["case_id"],
                    "mechanism_id": case["mechanism_id"], "task_class": case["task_class"],
                    "candidate_role": case["candidate_role"], "split": case["split"],
                    "template_id": case["template_id"], "condition": condition,
                    "frozen_block_id": block["block_id"],
                    "intervened_block_id": selected["block_id"] if selected else None,
                    **{key: round(value, 7) if math.isfinite(value) else None for key, value in score.items()},
                    "score_valid": valid,
                    "target_wins": bool(valid and score["phrase_margin"] > 0),
                    "phrase_margin_loss_vs_baseline": round(baseline["phrase_margin"] - score["phrase_margin"], 7) if valid else None,
                    "actual_model_batch_size": 1, "use_cache": False,
                    "single_unit_causal": False,
                })
            if index % 30 == 0 or index == len(cases):
                print(f"[{model}] {index}/{len(cases)}", flush=True)
        model_root = root / "models" / model
        write_jsonl(model_root / "phase348_screen_rows.jsonl", rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "screen_case_count": len(cases), "condition_row_count": len(rows),
            "invalid_row_count": sum(not row["score_valid"] for row in rows),
            "actual_model_batch_size": 1,
            "valid": len(cases) == 306 and len(rows) == 1530,
        }
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run_model(args.model, args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
