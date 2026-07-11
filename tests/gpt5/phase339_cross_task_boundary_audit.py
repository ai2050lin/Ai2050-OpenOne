#!/usr/bin/env python3
"""Apply Phase338 frozen blocks across the Phase339 task boundary matrix."""

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
from phase331_refined_mechanism_audit import target_match  # noqa: E402
from phase334_natural_contrast_survey import role_positions  # noqa: E402
from phase338_block_causal_screen import (  # noqa: E402
    install_block_hooks, prompt_ids, score_cases, wrong_block,
)
from phase339_cross_task_boundary_case_bank import (  # noqa: E402
    OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


MODELS = ("qwen3", "glm4", "deepseek7b")
PHRASE_CONDITIONS = (
    "baseline", "correct_zero", "correct_half", "correct_permutation",
    "wrong_depth_zero", "wrong_position_zero",
)
ROLLOUT_CONDITIONS = (
    "baseline", "correct_zero", "wrong_depth_zero", "wrong_position_zero",
)
PHASE338 = ROOT / "tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def first_nonempty_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def row_base(case: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "model": model, "case_id": case["case_id"],
        "semantic_case_id": case["semantic_case_id"],
        "family_id": case["family_id"], "mechanism_id": case["mechanism_id"],
        "task_class": case["task_class"], "item_index": case["item_index"],
        "split": case["split"], "template_id": case["template_id"],
        "interface": case["interface"], "target": case["target"],
    }


def finite_score(score: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in score.values())


def nullable_score(score: dict[str, float]) -> dict[str, float | None]:
    return {
        key: round(float(value), 7) if math.isfinite(float(value)) else None
        for key, value in score.items()
    }


@torch.inference_mode()
def generate_batch(
    loaded: Any, cases: list[dict[str, Any]], block: dict[str, Any] | None,
    mode: str | None, max_new_tokens: int,
) -> list[dict[str, Any]]:
    prompts = [prompt_ids(loaded, case) for case in cases]
    width = max(map(len, prompts))
    pad = int(loaded.tokenizer.pad_token_id)
    input_ids = torch.full(
        (len(cases), width), pad, dtype=torch.long, device=loaded.input_device
    )
    attention_mask = torch.zeros_like(input_ids)
    positions = []
    for index, (case, prompt) in enumerate(zip(cases, prompts, strict=True)):
        offset = width - len(prompt)
        input_ids[index, offset:] = torch.tensor(prompt, device=loaded.input_device)
        attention_mask[index, offset:] = 1
        role_map = role_positions(loaded, case, prompt)
        positions.append(offset + (role_map[block["position_role"]][0] if block else len(prompt) - 1))
    handles = install_block_hooks(loaded, block, positions, mode) if block and mode else []
    try:
        generated = loaded.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    finally:
        for handle in handles:
            handle.remove()
    suffix = generated[:, width:]
    rows = []
    for index, case in enumerate(cases):
        ids = [int(value) for value in suffix[index].tolist()]
        text = loaded.tokenizer.decode(ids, skip_special_tokens=True)
        head = first_nonempty_line(text)
        rows.append({
            "generated_text": text, "generated_token_ids": ids,
            "generated_token_count": len(ids), "answer_head_text": head,
            "answer_head_semantic_correct": target_match(head, case["target_aliases"]),
        })
    return rows


def run_model(
    model: str, round_name: str = ROUND_DEFAULT, batch_size: int = 6,
    max_new_tokens: int = 24,
) -> dict[str, Any]:
    root = OUT / round_name
    cases = [
        row for row in read_jsonl(root / "phase339_registered_cases.jsonl")
        if row["model"] == model
    ]
    frozen = read_jsonl(
        PHASE338 / "models" / model / "phase338_frozen_heldout_block.jsonl"
    )
    if len(frozen) != 1:
        raise RuntimeError(f"Expected one Phase338 frozen block for {model}, got {len(frozen)}")
    block = frozen[0]
    wrong_depth = wrong_block(block, "depth")
    wrong_position = wrong_block(block, "position")
    specs = {
        "baseline": (None, None), "correct_zero": (block, "zero"),
        "correct_half": (block, "half"),
        "correct_permutation": (block, "permutation"),
        "wrong_depth_zero": (wrong_depth, "zero"),
        "wrong_position_zero": (wrong_position, "zero"),
    }
    phrase_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    loaded = None
    try:
        loaded = load_probe_model(model)
        for start in range(0, len(cases), batch_size):
            batch = cases[start:start + batch_size]
            baseline = score_cases(loaded, batch, None, None)
            for condition in PHRASE_CONDITIONS:
                selected, mode = specs[condition]
                scores = baseline if condition == "baseline" else score_cases(
                    loaded, batch, selected, mode
                )
                for case, score, base in zip(batch, scores, baseline, strict=True):
                    score_valid = finite_score(score)
                    baseline_valid = finite_score(base)
                    phrase_rows.append({
                        **row_base(case, model), "condition": condition,
                        "frozen_block_id": block["block_id"],
                        "intervened_block_id": selected["block_id"] if selected else None,
                        "component": selected["component"] if selected else None,
                        "depth_bin": selected["depth_bin"] if selected else None,
                        "position_role": selected["position_role"] if selected else None,
                        **nullable_score(score),
                        "score_valid": score_valid,
                        "baseline_score_valid": baseline_valid,
                        "phrase_margin_loss_vs_baseline": (
                            round(base["phrase_margin"] - score["phrase_margin"], 7)
                            if score_valid and baseline_valid else None
                        ),
                        "single_unit_causal": False,
                    })
            if any(case["split"] in {"heldout", "private_heldout"} for case in batch):
                audit_batch = [
                    case for case in batch if case["split"] in {"heldout", "private_heldout"}
                ]
                for condition in ROLLOUT_CONDITIONS:
                    selected, mode = specs[condition]
                    generated = generate_batch(
                        loaded, audit_batch, selected, mode, max_new_tokens
                    )
                    for case, rollout in zip(audit_batch, generated, strict=True):
                        rollout_rows.append({
                            **row_base(case, model), "condition": condition,
                            "frozen_block_id": block["block_id"],
                            "intervened_block_id": selected["block_id"] if selected else None,
                            **rollout, "single_unit_causal": False,
                        })
            if (start // batch_size + 1) % 9 == 0 or start + batch_size >= len(cases):
                print(f"[{model}] {min(start + batch_size, len(cases))}/{len(cases)}", flush=True)
        baseline_rollout = {
            row["case_id"]: row for row in rollout_rows if row["condition"] == "baseline"
        }
        for row in rollout_rows:
            base = baseline_rollout[row["case_id"]]
            row["behavior_lost_vs_baseline"] = bool(
                base["answer_head_semantic_correct"]
                and not row["answer_head_semantic_correct"]
            )
        model_root = root / "models" / model
        write_jsonl(model_root / "phase339_phrase_rows.jsonl", phrase_rows)
        write_jsonl(model_root / "phase339_rollout_rows.jsonl", rollout_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "frozen_block_id": block["block_id"],
            "registered_case_count": len(cases), "phrase_row_count": len(phrase_rows),
            "rollout_case_count": len({row["case_id"] for row in rollout_rows}),
            "rollout_row_count": len(rollout_rows),
            "invalid_phrase_row_count": sum(
                not row["score_valid"] for row in phrase_rows
            ),
            "phrase_measurements_all_finite": all(
                row["score_valid"] for row in phrase_rows
            ),
            "valid": len(cases) == 486 and len(phrase_rows) == 2916
            and len(rollout_rows) == 540,
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
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(
        run_model(args.model, args.round, args.batch_size, args.max_new_tokens),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
