#!/usr/bin/env python3
"""Run Phase1022 candidate-free BF16 behavior qualification.

Generation is greedy and candidate answers never appear in the prompt.  The
actual continuation token IDs are retained so the later timeline scan follows
the model's own successful or failed rollout.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import release_model
from phase1014_bf16_precision_confirmation import load_bf16
import phase1022_ability_family_protocol as protocol


GENERATION_BATCH = {
    "qwen3": 32,
    "glm4": 16,
    "deepseek7b": 16,
}
PUNCTUATION = ".?!,;:。？！；："
ARTICLES = {
    "a",
    "an",
    "the",
    "un",
    "une",
    "le",
    "la",
    "les",
    "l",
}


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def strip_accents(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"<think>.*?</think>", " ", value, flags=re.DOTALL)
    value = strip_accents(value)
    value = value.replace("’", "'")
    value = re.sub(
        r"^(?:answer|translation|result|output|category|class|"
        r"答案|译词|翻译|结果|输出|类别|标签|释义)\s*[:：]\s*",
        "",
        value,
    )
    value = value.strip("`*_\"'“”‘’ \t\r\n")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def first_nonempty_line(value: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL)
    for line in value.replace("\r", "\n").split("\n"):
        if line.strip():
            return line.strip()
    return ""


def lexical_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", normalize(value))


def short_hit(observed: str, accepted: list[str]) -> tuple[bool, float]:
    observed_norm = normalize(observed).rstrip(PUNCTUATION)
    observed_tokens = lexical_tokens(observed_norm)
    while observed_tokens and observed_tokens[0] in ARTICLES:
        observed_tokens.pop(0)
    observed_core = " ".join(observed_tokens)
    best = 0.0
    for value in accepted:
        expected_tokens = lexical_tokens(value)
        while expected_tokens and expected_tokens[0] in ARTICLES:
            expected_tokens.pop(0)
        expected = " ".join(expected_tokens)
        if not expected:
            continue
        if observed_core == expected:
            return True, 1.0
        if (
            observed_core.startswith(expected + " ")
            or observed_core.startswith(expected + "，")
            or observed_core.startswith(expected + ",")
        ):
            return True, 1.0
        if re.search(r"[\u3400-\u9fff]", expected):
            if expected in observed_core and len(observed_core) <= len(expected) + 12:
                return True, 1.0
        best = max(best, SequenceMatcher(None, observed_core, expected).ratio())
    return False, float(best)


def definition_hit(observed: str, accepted: list[str]) -> tuple[bool, float]:
    observed_norm = normalize(observed)
    best = 0.0
    for value in accepted:
        expected = normalize(value)
        if expected and expected in observed_norm:
            return True, 1.0
        best = max(best, SequenceMatcher(None, observed_norm, expected).ratio())
    return False, float(best)


def punctuation_hit(observed: str, accepted: list[str]) -> tuple[bool, float, str]:
    found = ""
    for char in observed:
        if char in PUNCTUATION:
            found = char
            break
    canonical = {
        "。": ".",
        "？": "?",
        "！": "!",
        "，": ",",
        "；": ";",
        "：": ":",
    }
    expected = {canonical.get(value, value) for value in accepted}
    hit = canonical.get(found, found) in expected
    return bool(hit), 1.0 if hit else 0.0, found


def connector_hit(observed: str, accepted: list[str]) -> tuple[bool, float]:
    line = normalize(first_nonempty_line(observed)).lstrip(PUNCTUATION + " ")
    best = 0.0
    for value in accepted:
        expected = normalize(value)
        if (
            line == expected
            or line.startswith(expected + " ")
            or line.startswith(expected + ",")
            or line.startswith(expected + "，")
        ):
            return True, 1.0
        best = max(best, SequenceMatcher(None, line[:32], expected).ratio())
    return False, float(best)


def evaluate(case: dict[str, Any], text: str) -> dict[str, Any]:
    evaluation_type = case["evaluation_type"]
    line = first_nonempty_line(text)
    if evaluation_type == "punctuation":
        hit, score, cleaned = punctuation_hit(text, case["accepted_outputs"])
    elif evaluation_type == "definition":
        hit, score = definition_hit(text, case["accepted_outputs"])
        cleaned = line
    elif evaluation_type == "connector":
        hit, score = connector_hit(text, case["accepted_outputs"])
        cleaned = line
    else:
        hit, score = short_hit(line, case["accepted_outputs"])
        cleaned = line
    return {
        "generated_text": text,
        "cleaned_output": cleaned,
        "semantic_hit": bool(hit),
        "semantic_score": float(score),
    }


def generation_groups(
    rows: list[dict[str, Any]],
) -> Iterable[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        width_bucket = (len(row["input_ids"]) // 16) * 16
        grouped[(int(row["max_new_tokens"]), width_bucket)].append(row)
    for key in sorted(grouped):
        yield grouped[key]


def trim_continuation(
    token_ids: list[int],
    *,
    eos_ids: set[int],
    pad_id: int,
) -> list[int]:
    result = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in eos_ids:
            break
        if token_id == pad_id and pad_id not in eos_ids:
            break
        result.append(token_id)
    return result


def run_generation(
    *,
    model,
    tokenizer,
    device,
    model_name: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    eos_value = tokenizer.eos_token_id
    if isinstance(eos_value, (list, tuple, set)):
        eos_ids = {int(value) for value in eos_value}
    elif eos_value is None:
        eos_ids = set()
    else:
        eos_ids = {int(eos_value)}

    results: dict[str, dict[str, Any]] = {}
    batch_size = GENERATION_BATCH[model_name]
    started = time.time()
    batch_index = 0
    for group in generation_groups(cases):
        for batch in chunks(group, batch_size):
            batch_index += 1
            width = max(len(row["input_ids"]) for row in batch)
            input_ids = torch.full(
                (len(batch), width),
                int(pad_id),
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for index, row in enumerate(batch):
                values = torch.tensor(
                    row["input_ids"],
                    dtype=torch.long,
                    device=device,
                )
                input_ids[index, width - len(values):] = values
                attention_mask[index, width - len(values):] = 1
            max_new_tokens = int(batch[0]["max_new_tokens"])
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=int(pad_id),
                    eos_token_id=tokenizer.eos_token_id,
                )
            continuation = generated[:, width:].detach().cpu().tolist()
            for index, row in enumerate(batch):
                actual_ids = trim_continuation(
                    continuation[index],
                    eos_ids=eos_ids,
                    pad_id=int(pad_id),
                )
                text = tokenizer.decode(
                    actual_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                observation = evaluate(row, text)
                observation.update({
                    "generated_token_ids": actual_ids,
                    "generated_token_count": len(actual_ids),
                })
                results[row["record_id"]] = observation
            del generated, continuation, input_ids, attention_mask
            if batch_index % 20 == 0:
                print(
                    f"[behavior] {model_name} batch={batch_index} "
                    f"rows={len(results)}/{len(cases)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    return [results[row["record_id"]] for row in cases]


def attach(
    cases: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    model_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for case, observation in zip(cases, observations):
        rows.append({
            "schema_version": "phase1022_behavior_row.v1",
            "phase": protocol.PHASE,
            "protocol_revision": protocol.PROTOCOL_REVISION,
            "model": model_name,
            "record_id": case["record_id"],
            "case_key": case["case_key"],
            "family": case["family"],
            "task": case["task"],
            "split": case["split"],
            "template": int(case["template"]),
            "concept_id": case["concept_id"],
            "category": case["category"],
            "source_language": case["source_language"],
            "target_language": case["target_language"],
            "source_term": case["source_term"],
            "surface_identity": bool(case["surface_identity"]),
            "prompt_token_count": int(case["prompt_token_count"]),
            "source_token_count": int(case["source_token_count"]),
            "accepted_outputs": case["accepted_outputs"],
            **observation,
        })
    return rows


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "accuracy": (
            float(np.mean([row["semantic_hit"] for row in rows]))
            if rows else None
        ),
        "mean_semantic_score": (
            float(np.mean([row["semantic_score"] for row in rows]))
            if rows else None
        ),
        "empty_generation_rate": (
            float(np.mean([
                row["generated_token_count"] == 0 for row in rows
            ]))
            if rows else None
        ),
        "median_generated_tokens": (
            float(np.median([
                row["generated_token_count"] for row in rows
            ]))
            if rows else None
        ),
    }


def grouped_metrics(
    rows: list[dict[str, Any]],
    key_names: tuple[str, ...],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(str(row[name]) for name in key_names)
        grouped[key].append(row)
    return {key: metrics(values) for key, values in sorted(grouped.items())}


def run_model(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    output_root = protocol.OUT_ROOT / "behavior" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    model = tokenizer = device = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        observations = run_generation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=model_name,
            cases=cases,
        )
        rows = attach(cases, observations, model_name)
        protocol.write_jsonl(output_root / "formal.jsonl", rows)
        nonidentity_translation = [
            row
            for row in rows
            if row["family"] == "translation"
            and not row["surface_identity"]
        ]
        summary = {
            "schema_version": "phase1022_behavior_summary.v1",
            "phase": protocol.PHASE,
            "protocol_revision": protocol.PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "precision": "bf16",
            "quantization": "none",
            "placement": placement,
            "case_count": len(rows),
            "overall": metrics(rows),
            "family": grouped_metrics(rows, ("family",)),
            "split": grouped_metrics(rows, ("family", "split")),
            "translation_nonidentity": metrics(nonidentity_translation),
            "translation_direction": grouped_metrics(
                nonidentity_translation,
                ("source_language", "target_language"),
            ),
            "translation_category": grouped_metrics(
                nonidentity_translation,
                ("category",),
            ),
            "translation_template": grouped_metrics(
                nonidentity_translation,
                ("split", "template"),
            ),
            "classification_direction": grouped_metrics(
                [row for row in rows if row["family"] == "classification"],
                ("source_language", "target_language"),
            ),
            "rare_term": grouped_metrics(
                [row for row in rows if row["family"] == "rare_definition"],
                ("concept_id",),
            ),
            "generated_token_histogram": dict(Counter(
                row["generated_token_count"] for row in rows
            )),
            "elapsed_seconds": time.time() - started,
        }
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_model(model)
        del model, tokenizer, device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
