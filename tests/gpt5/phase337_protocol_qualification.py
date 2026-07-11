#!/usr/bin/env python3
"""Run Phase337 natural-generation protocol qualification for one local model."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase331_refined_mechanism_audit import target_match  # noqa: E402
from phase334_natural_contrast_survey import continuation_ids, encoded_prompt  # noqa: E402
from phase337_protocol_qualification_case_bank import (  # noqa: E402
    OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def phrase_valid(loaded: Any, case: dict[str, Any]) -> tuple[bool, list[int], list[list[int]]]:
    target_ids = continuation_ids(loaded, case, case["target"])
    distractor_ids = [continuation_ids(loaded, case, value) for value in case["distractors"]]
    decoded = normalize(loaded.tokenizer.decode(target_ids, skip_special_tokens=True))
    valid = bool(target_ids and normalize(case["target"]) in decoded)
    valid = valid and all(ids and ids != target_ids for ids in distractor_ids)
    return valid, target_ids, distractor_ids


@torch.inference_mode()
def phrase_score(loaded: Any, case: dict[str, Any], ids: list[int]) -> float:
    encoded = encoded_prompt(loaded, case)
    prefix = encoded["input_ids"]
    if len(ids) > 1:
        append = torch.tensor([ids[:-1]], dtype=prefix.dtype, device=prefix.device)
        input_ids = torch.cat([prefix, append], dim=1)
        attention_mask = torch.cat([
            encoded["attention_mask"],
            torch.ones_like(append, dtype=encoded["attention_mask"].dtype),
        ], dim=1)
    else:
        input_ids = prefix
        attention_mask = encoded["attention_mask"]
    output = loaded.model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True
    )
    start = int(prefix.shape[1]) - 1
    positions = torch.arange(start, start + len(ids), device=loaded.input_device)
    logits = output.logits[0, positions].float()
    token_ids = torch.tensor(ids, device=loaded.input_device)
    return float(torch.log_softmax(logits, dim=-1).gather(1, token_ids[:, None]).sum().item())


def split_answer(case: dict[str, Any], text: str) -> tuple[bool, str, str]:
    if case["answer_phase"] != "think_start":
        reached = bool(text.strip())
        return reached, text.strip(), "prealigned_answer" if reached else "empty_generation"
    if "</think>" in text:
        return True, text.rsplit("</think>", 1)[-1].strip(), "think_closed"
    return False, "", "think_not_closed"


def first_nonempty_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


@torch.inference_mode()
def run_case(loaded: Any, case: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    encoded = encoded_prompt(loaded, case)
    target_phrase_valid, target_ids, distractor_ids = phrase_valid(loaded, case)
    generated = loaded.model.generate(
        **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        return_dict_in_generate=True, output_scores=True,
        pad_token_id=loaded.tokenizer.pad_token_id,
        eos_token_id=loaded.tokenizer.eos_token_id,
    )
    suffix = generated.sequences[0, encoded["input_ids"].shape[1]:]
    ids = [int(value) for value in suffix.tolist()]
    text = loaded.tokenizer.decode(ids, skip_special_tokens=False)
    clean_text = loaded.tokenizer.decode(ids, skip_special_tokens=True)
    answer_reached, answer_text, reach_reason = split_answer(case, text)
    answer_head_text = first_nonempty_line(answer_text)
    semantic_correct = target_match(text, case["target_aliases"])
    answer_semantic_correct = answer_reached and target_match(answer_text, case["target_aliases"])
    answer_head_semantic_correct = answer_reached and target_match(
        answer_head_text, case["target_aliases"]
    )
    semantic_correct_outside_answer = semantic_correct and not answer_semantic_correct
    baseline_capability = bool(
        answer_reached and answer_head_semantic_correct and target_phrase_valid
    )
    first_logits = generated.scores[0][0].detach().float()
    target_logit = float(first_logits[target_ids[0]].item())
    target_rank = 1 + int((first_logits > target_logit).sum().item())
    distractor_logits = [float(first_logits[ids_[0]].item()) for ids_ in distractor_ids]
    target_phrase_logprob = phrase_score(loaded, case, target_ids)
    distractor_phrase_logprobs = [phrase_score(loaded, case, ids_) for ids_ in distractor_ids]
    protocol_followed = bool(
        answer_semantic_correct
        and len(normalize(answer_text).split()) <= 4
    )
    emitted_eos = bool(eos_ids(loaded.tokenizer).intersection(ids))
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "case_id": case["case_id"],
        "semantic_case_id": case["semantic_case_id"],
        "model": case["model"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "item_index": case["item_index"],
        "split": case["split"],
        "template_id": case["template_id"],
        "interface": case["interface"],
        "answer_phase": case["answer_phase"],
        "target": case["target"],
        "target_phrase_valid": target_phrase_valid,
        "target_token_ids": target_ids,
        "target_phrase_logprob": round(target_phrase_logprob, 7),
        "best_distractor_phrase_logprob": round(max(distractor_phrase_logprobs), 7),
        "target_phrase_margin": round(target_phrase_logprob - max(distractor_phrase_logprobs), 7),
        "initial_target_rank": target_rank,
        "initial_target_margin": round(target_logit - max(distractor_logits), 7),
        "generated_text": text,
        "generated_clean_text": clean_text,
        "generated_token_ids": ids,
        "generated_token_count": len(ids),
        "answer_reached": answer_reached,
        "answer_reached_reason": reach_reason,
        "answer_text": answer_text,
        "answer_head_text": answer_head_text,
        "semantic_correct": semantic_correct,
        "answer_semantic_correct": answer_semantic_correct,
        "answer_head_semantic_correct": answer_head_semantic_correct,
        "semantic_correct_outside_answer": semantic_correct_outside_answer,
        "protocol_followed": protocol_followed,
        "baseline_capability": baseline_capability,
        "eos_emitted": emitted_eos,
        "token_budget_exhausted": len(ids) >= max_new_tokens and not emitted_eos,
        "evidence_level": "L2_protocol_qualified_observation",
        "internal_activation_captured": False,
        "internal_intervention_applied": False,
        "single_unit_causal": False,
    }


def run_model(model: str, round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase337_registered_protocol.json")
    cases = [
        row for row in read_jsonl(root / "phase337_registered_cases.jsonl")
        if row["model"] == model
    ]
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for index, case in enumerate(cases, 1):
            rows.append(run_case(loaded, case, int(protocol["max_new_tokens"])))
            if index % 6 == 0:
                print(f"[{model}] {index}/{len(cases)}", flush=True)
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model_root = root / "models" / model
    write_jsonl(model_root / "phase337_qualification_rows.jsonl", rows)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "registered_case_count": len(cases),
        "executed_case_count": len(rows),
        "valid": len(rows) == len(cases),
    }
    write_json(model_root / "complete.json", complete)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("qwen3", "glm4", "deepseek7b"))
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run_model(args.model, args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
