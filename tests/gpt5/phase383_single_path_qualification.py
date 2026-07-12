#!/usr/bin/env python3
"""Requalify Phase380 cases on the exact Phase383 single-sample runtime path."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase379_decision_aligned_trace import decision_input, token_rank  # noqa: E402


SOURCE = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
OUT = ROOT / "tests/gpt5/result/phase383_exact_component_event_map/qualification"
CASES = SOURCE / "private/phase380_qualified_trace_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
FROZEN_DTYPE_NAMES = {
    "qwen3": "float16",
    "glm4": "float16",
    "deepseek7b": "bfloat16",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


@torch.inference_mode()
def process(model: str) -> dict[str, Any]:
    cases = [
        row
        for row in read_jsonl(CASES)
        if row["private_execution_model"] == model
    ]
    if len(cases) != 260:
        raise RuntimeError(f"Expected 260 Phase380 candidates for {model}, got {len(cases)}")
    loaded = None
    rows = []
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPE_NAMES[model]:
            raise RuntimeError(
                f"Single-path dtype mismatch for {model}: {runtime_dtype} != "
                f"{FROZEN_DTYPE_NAMES[model]}"
            )
        for index, case in enumerate(cases, 1):
            sequence, _positions = decision_input(loaded, case)
            input_ids = torch.tensor(
                [sequence], dtype=torch.long, device=loaded.input_device
            )
            output = loaded.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = output.logits[0, -1].detach().float().cpu()
            target_token = int(
                case["generated_token_ids"][int(case["target_decision_step"])]
            )
            finite = bool(torch.isfinite(logits).all().item())
            if finite:
                argmax = int(torch.argmax(logits).item())
                rank = token_rank(logits, target_token)
                target_logit = float(logits[target_token].item())
                masked = logits.clone()
                masked[target_token] = -torch.inf
                best_other_logit = float(torch.max(masked).item())
                margin = target_logit - best_other_logit
            else:
                argmax = -1
                rank = -1
                target_logit = 0.0
                best_other_logit = 0.0
                margin = 0.0
            qualified = finite and argmax == target_token and rank == 1
            rows.append(
                {
                    "schema_version": "57.0.1",
                    "phase_id": "Phase383-SinglePathQualification",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "runtime_dtype": runtime_dtype,
                    "execution_batch_size": 1,
                    "output_attentions": True,
                    "blind_case_id": case["blind_case_id"],
                    "anonymous_parallel_group_id": case[
                        "anonymous_parallel_group_id"
                    ],
                    "mechanism_id_private": case["mechanism_id"],
                    "contrast_condition_private": case["contrast_condition"],
                    "sequence_length": len(sequence),
                    "target_token_id_private": target_token,
                    "argmax_token_id_private": argmax,
                    "target_rank_private": rank,
                    "target_logit_private": target_logit,
                    "best_other_logit_private": best_other_logit,
                    "target_margin_private": margin,
                    "all_logits_finite": finite,
                    "single_path_qualified": qualified,
                }
            )
            if index % 20 == 0 or index == len(cases):
                print(
                    f"[{model}] single-path qualification {index}/{len(cases)} "
                    f"pass={sum(row['single_path_qualified'] for row in rows)}",
                    flush=True,
                )
            del output, input_ids, logits
        path = OUT / "private/models" / model / "phase383_single_path_rows.jsonl"
        write_jsonl(path, rows)
        summary = {
            "schema_version": "57.0.1",
            "phase_id": "Phase383-SinglePathQualification",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "runtime_dtype": runtime_dtype,
            "execution_batch_size": 1,
            "output_attentions": True,
            "case_count": len(rows),
            "finite_case_count": sum(row["all_logits_finite"] for row in rows),
            "single_path_qualified_case_count": sum(
                row["single_path_qualified"] for row in rows
            ),
            "minimum_qualified_target_margin": min(
                (
                    row["target_margin_private"]
                    for row in rows
                    if row["single_path_qualified"]
                ),
                default=0.0,
            ),
            "valid": len(rows) == len(cases),
        }
        write_json(OUT / "models" / model / "complete.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    process(args.model)
