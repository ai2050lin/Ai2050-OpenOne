#!/usr/bin/env python3
"""Measure batch/cache/padding invariance for natural model execution."""

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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase331_refined_mechanism_audit import target_match  # noqa: E402
from phase334_natural_contrast_survey import role_positions  # noqa: E402
from phase338_block_causal_screen import (  # noqa: E402
    continuation_ids, get_layers, layers_for_bin, prompt_ids,
)
from phase342_copy_relay_execution_case_bank import (  # noqa: E402
    MODES, OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


MODELS = ("qwen3", "glm4", "deepseek7b")


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


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


@torch.inference_mode()
def execute_batch(
    loaded: Any, cases: list[dict[str, Any]], padding_side: str, use_cache: bool,
) -> list[tuple[dict[str, Any], torch.Tensor]]:
    prompts = [prompt_ids(loaded, case) for case in cases]
    width = max(map(len, prompts))
    pad = int(loaded.tokenizer.pad_token_id)
    input_ids = torch.full((len(cases), width), pad, dtype=torch.long, device=loaded.input_device)
    attention_mask = torch.zeros_like(input_ids)
    last_positions: list[int] = []
    source_positions: list[int] = []
    target_ids: list[int] = []
    for index, (case, prompt) in enumerate(zip(cases, prompts, strict=True)):
        offset = width - len(prompt) if padding_side == "left" else 0
        input_ids[index, offset:offset + len(prompt)] = torch.tensor(prompt, device=loaded.input_device)
        attention_mask[index, offset:offset + len(prompt)] = 1
        last_positions.append(offset + len(prompt) - 1)
        source = role_positions(loaded, case, prompt)["source"][0]
        source_positions.append(offset + source)
        target_ids.append(continuation_ids(loaded, case, case["target"])[0])
    output = loaded.model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache,
        output_hidden_states=True, return_dict=True,
    )
    early_last = layers_for_bin(len(get_layers(loaded.model)), "early")[-1] + 1
    hidden = output.hidden_states[early_last].detach().float()
    forward_rows: list[tuple[dict[str, Any], torch.Tensor]] = []
    for index, case in enumerate(cases):
        logits = output.logits[index, last_positions[index]].detach().float()
        vector = hidden[index, source_positions[index]].detach().cpu()
        forward_finite = bool(
            torch.isfinite(logits).all().item() and torch.isfinite(vector).all().item()
        )
        target_logit = float(logits[target_ids[index]].item())
        hidden_norm = float(vector.norm().item())
        forward_rows.append(({
            "target_first_token_id": int(target_ids[index]),
            "target_first_logit": target_logit if math.isfinite(target_logit) else None,
            "top_token_id": int(logits.argmax().item()),
            "source_hidden_norm": hidden_norm if math.isfinite(hidden_norm) else None,
            "forward_finite": forward_finite,
        }, vector))
    generated = loaded.model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=16,
        do_sample=False, use_cache=use_cache,
        pad_token_id=loaded.tokenizer.pad_token_id,
        eos_token_id=loaded.tokenizer.eos_token_id,
    )
    suffix = generated[:, width:]
    results = []
    for index, case in enumerate(cases):
        ids = [int(value) for value in suffix[index].tolist()]
        text = loaded.tokenizer.decode(ids, skip_special_tokens=True)
        head = first_line(text)
        forward, vector = forward_rows[index]
        results.append(({
            **forward, "generated_text": text, "answer_head_text": head,
            "answer_head_semantic_correct": target_match(head, case["target_aliases"]),
            "generated_token_ids": ids,
        }, vector))
    del output, hidden, input_ids, attention_mask, generated, suffix
    return results


def run_model(model: str, round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    cases = [row for row in read_jsonl(root / "phase342_registered_cases.jsonl") if row["model"] == model]
    rows: list[dict[str, Any]] = []
    references: dict[str, tuple[dict[str, Any], torch.Tensor]] = {}
    loaded = None
    try:
        loaded = load_probe_model(model)
        for mode_index, (mode_id, batch_size, padding_side, use_cache) in enumerate(MODES):
            mode_rows = []
            for start in range(0, len(cases), batch_size):
                batch = cases[start:start + batch_size]
                executed = execute_batch(loaded, batch, padding_side, use_cache)
                for case, (result, vector) in zip(batch, executed, strict=True):
                    if mode_index == 0:
                        references[case["case_id"]] = (result, vector)
                    reference, ref_vector = references[case["case_id"]]
                    vectors_finite = bool(
                        torch.isfinite(vector).all().item()
                        and torch.isfinite(ref_vector).all().item()
                    )
                    cosine = (
                        float(F.cosine_similarity(
                            vector.unsqueeze(0), ref_vector.unsqueeze(0), dim=-1
                        ).item())
                        if vectors_finite else None
                    )
                    logits_finite = bool(
                        result["target_first_logit"] is not None
                        and reference["target_first_logit"] is not None
                    )
                    delta = (
                        abs(result["target_first_logit"] - reference["target_first_logit"])
                        if logits_finite else None
                    )
                    mode_rows.append({
                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                        "created_at": now(), "model": model, "case_id": case["case_id"],
                        "mechanism_id": case["mechanism_id"], "task_class": case["task_class"],
                        "split": case["split"], "template_id": case["template_id"],
                        "mode_id": mode_id, "batch_size": batch_size,
                        "padding_side": padding_side, "use_cache": use_cache,
                        **result,
                        "text_equal_to_reference": result["answer_head_text"] == reference["answer_head_text"],
                        "correctness_equal_to_reference": result["answer_head_semantic_correct"] == reference["answer_head_semantic_correct"],
                        "top_token_equal_to_reference": result["top_token_id"] == reference["top_token_id"],
                        "target_first_logit_abs_delta": round(delta, 7) if delta is not None else None,
                        "source_hidden_cosine_to_reference": (
                            round(cosine, 7) if cosine is not None and math.isfinite(cosine) else None
                        ),
                        "internal_intervention": False,
                    })
            rows.extend(mode_rows)
            print(f"[{model}] {mode_id}: {len(mode_rows)}/{len(cases)}", flush=True)
        model_root = root / "models" / model
        write_jsonl(model_root / "phase342_execution_rows.jsonl", rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "registered_case_count": len(cases),
            "mode_count": len(MODES), "result_row_count": len(rows),
            "nonfinite_row_count": sum(not row["forward_finite"] for row in rows),
            "valid": len(cases) == 72 and len(rows) == 792,
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
