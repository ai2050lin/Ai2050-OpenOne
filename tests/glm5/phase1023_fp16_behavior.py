#!/usr/bin/env python3
"""Run Phase1023 candidate-free behavior in FP16.

Translation is scored twice:

* semantic correctness: a valid target-language equivalent appears anywhere;
* protocol correctness: the equivalent is the first concise answer.

This prevents explanation prefixes and prompt echoes from being mislabeled as
missing lexical knowledge.
"""

from __future__ import annotations

import argparse
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

import phase1023_ecological_niche_protocol as protocol
from phase1023_fp16_utils import (
    MODELS,
    load_fp16,
    quantization_audit,
    release_fp16,
)


GENERATION_BATCH = {
    "qwen3": 32,
    "glm4": 12,
    "deepseek7b": 24,
}
PUNCTUATION = ".?!,;:。？！；，："
ARTICLES = {"a", "an", "the", "un", "une", "le", "la", "les", "l"}


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


def strip_answer_prefix(value: str) -> str:
    value = normalize(value)
    value = re.sub(
        r"^(?:answer|translation|result|output|equivalent|category|class|"
        r"答案|译词|翻译|结果|输出|对应词|类别|标签|释义)\s*[:：]\s*",
        "",
        value,
    )
    return value.strip("`*_\"'“”‘’ \t\r\n")


def phrase_in_text(phrase: str, text: str) -> bool:
    phrase_norm = normalize(phrase).strip(PUNCTUATION + " ")
    text_norm = normalize(text)
    if not phrase_norm:
        return False
    if re.search(r"[\u3400-\u9fff]", phrase_norm):
        return phrase_norm in text_norm
    pattern = r"(?<![a-z0-9])" + re.escape(phrase_norm) + r"(?![a-z0-9])"
    return re.search(pattern, text_norm) is not None


def best_similarity(observed: str, accepted: list[str]) -> float:
    observed_norm = normalize(observed)
    return max(
        (
            SequenceMatcher(
                None,
                observed_norm,
                normalize(value),
            ).ratio()
            for value in accepted
            if normalize(value)
        ),
        default=0.0,
    )


def protocol_translation_hit(
    observed: str,
    accepted: list[str],
) -> bool:
    first = strip_answer_prefix(first_nonempty_line(observed))
    tokens = lexical_tokens(first)
    while tokens and tokens[0] in ARTICLES:
        tokens.pop(0)
    core = " ".join(tokens)
    for value in accepted:
        expected_tokens = lexical_tokens(value)
        while expected_tokens and expected_tokens[0] in ARTICLES:
            expected_tokens.pop(0)
        expected = " ".join(expected_tokens)
        if not expected:
            continue
        if core == expected or core.startswith(expected + " "):
            return True
        if (
            re.search(r"[\u3400-\u9fff]", expected)
            and core.startswith(expected)
            and len(core) <= len(expected) + 8
        ):
            return True
    return False


def translation_evaluation(
    case: dict[str, Any],
    observed: str,
    *,
    generated_token_count: int,
    ended_by_eos: bool,
) -> dict[str, Any]:
    accepted = case["accepted_outputs"]
    semantic_hit = any(
        phrase_in_text(value, observed) for value in accepted
    )
    protocol_hit = protocol_translation_hit(observed, accepted)
    source_terms = case["all_terms"].get(case["source_language"], [])
    other_terms = [
        value
        for language, values in case["all_terms"].items()
        if language != case["target_language"]
        for value in values
    ]
    echo = any(phrase_in_text(value, observed) for value in source_terms)
    wrong_language = any(
        phrase_in_text(value, observed) for value in other_terms
    )
    reached_limit = (
        generated_token_count >= int(case["max_new_tokens"])
        and not ended_by_eos
    )
    if semantic_hit:
        error_class = (
            "semantic_success" if protocol_hit else "format_success"
        )
    elif wrong_language:
        error_class = "language_error"
    elif echo:
        error_class = "echo_error"
    elif reached_limit:
        error_class = "truncated_error"
    else:
        error_class = "semantic_error"
    return {
        "semantic_hit": bool(semantic_hit),
        "protocol_hit": bool(protocol_hit),
        "semantic_score": (
            1.0 if semantic_hit else best_similarity(observed, accepted)
        ),
        "error_class": error_class,
        "source_echo": bool(echo),
        "wrong_language_term": bool(wrong_language),
        "reached_generation_limit": bool(reached_limit),
        "cleaned_output": first_nonempty_line(observed),
    }


def short_evaluation(
    observed: str,
    accepted: list[str],
) -> dict[str, Any]:
    line = strip_answer_prefix(first_nonempty_line(observed))
    hit = any(phrase_in_text(value, line) for value in accepted)
    return {
        "semantic_hit": bool(hit),
        "protocol_hit": bool(hit),
        "semantic_score": (
            1.0 if hit else best_similarity(line, accepted)
        ),
        "error_class": "semantic_success" if hit else "semantic_error",
        "source_echo": False,
        "wrong_language_term": False,
        "reached_generation_limit": False,
        "cleaned_output": line,
    }


def definition_evaluation(
    observed: str,
    accepted: list[str],
) -> dict[str, Any]:
    hits = [value for value in accepted if phrase_in_text(value, observed)]
    hit = bool(hits)
    return {
        "semantic_hit": hit,
        "protocol_hit": hit,
        "semantic_score": (
            1.0 if hit else best_similarity(observed, accepted)
        ),
        "error_class": "semantic_success" if hit else "semantic_error",
        "matched_references": hits,
        "source_echo": False,
        "wrong_language_term": False,
        "reached_generation_limit": False,
        "cleaned_output": first_nonempty_line(observed),
    }


def punctuation_evaluation(
    observed: str,
    accepted: list[str],
) -> dict[str, Any]:
    canonical = {
        "。": ".",
        "？": "?",
        "！": "!",
        "，": ",",
        "；": ";",
        "：": ":",
    }
    found = next(
        (char for char in observed if char in PUNCTUATION),
        "",
    )
    expected = {canonical.get(value, value) for value in accepted}
    hit = canonical.get(found, found) in expected
    return {
        "semantic_hit": bool(hit),
        "protocol_hit": bool(hit),
        "semantic_score": 1.0 if hit else 0.0,
        "error_class": "semantic_success" if hit else "semantic_error",
        "source_echo": False,
        "wrong_language_term": False,
        "reached_generation_limit": False,
        "cleaned_output": found,
    }


def connector_evaluation(
    observed: str,
    accepted: list[str],
) -> dict[str, Any]:
    line = strip_answer_prefix(first_nonempty_line(observed))
    hit = any(phrase_in_text(value, line) for value in accepted)
    return {
        "semantic_hit": bool(hit),
        "protocol_hit": bool(hit),
        "semantic_score": (
            1.0 if hit else best_similarity(line, accepted)
        ),
        "error_class": "semantic_success" if hit else "semantic_error",
        "source_echo": False,
        "wrong_language_term": False,
        "reached_generation_limit": False,
        "cleaned_output": line,
    }


def evaluate(
    case: dict[str, Any],
    observed: str,
    *,
    generated_token_count: int,
    ended_by_eos: bool,
) -> dict[str, Any]:
    kind = case["evaluation_type"]
    if kind == "translation":
        result = translation_evaluation(
            case,
            observed,
            generated_token_count=generated_token_count,
            ended_by_eos=ended_by_eos,
        )
    elif kind == "definition":
        result = definition_evaluation(
            observed, case["accepted_outputs"]
        )
    elif kind == "punctuation":
        result = punctuation_evaluation(
            observed, case["accepted_outputs"]
        )
    elif kind == "connector":
        result = connector_evaluation(
            observed, case["accepted_outputs"]
        )
    else:
        result = short_evaluation(
            observed, case["accepted_outputs"]
        )
    result["generated_text"] = observed
    return result


def generation_groups(
    rows: list[dict[str, Any]],
) -> Iterable[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        width_bucket = (len(row["input_ids"]) // 16) * 16
        grouped[(int(row["max_new_tokens"]), width_bucket)].append(row)
    for key in sorted(grouped):
        yield grouped[key]


def split_continuation(
    token_ids: list[int],
    *,
    eos_ids: set[int],
    pad_id: int,
) -> tuple[list[int], bool]:
    result = []
    ended_by_eos = False
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in eos_ids:
            ended_by_eos = True
            break
        if token_id == pad_id and pad_id not in eos_ids:
            break
        result.append(token_id)
    return result, ended_by_eos


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

    output: dict[str, dict[str, Any]] = {}
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
                actual_ids, ended_by_eos = split_continuation(
                    continuation[index],
                    eos_ids=eos_ids,
                    pad_id=int(pad_id),
                )
                text = tokenizer.decode(
                    actual_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                evaluation = evaluate(
                    row,
                    text,
                    generated_token_count=len(actual_ids),
                    ended_by_eos=ended_by_eos,
                )
                output[row["case_key"]] = {
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in ("input_ids", "rendered_prompt")
                    },
                    **evaluation,
                    "schema_version": "phase1023_behavior_result.v1",
                    "phase": protocol.PHASE,
                    "protocol_revision": protocol.PROTOCOL_REVISION,
                    "model": model_name,
                    "precision": "fp16",
                    "quantization": "none",
                    "generated_token_ids": actual_ids,
                    "generated_token_count": len(actual_ids),
                    "ended_by_eos": bool(ended_by_eos),
                    "minimum_target_token_count": (
                        min(row["accepted_token_counts"])
                        if row["accepted_token_counts"] else 0
                    ),
                }
            if batch_index % 10 == 0:
                print(
                    f"[behavior] {model_name} batch={batch_index} "
                    f"cases={len(output)}/{len(cases)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
            del input_ids, attention_mask, generated, continuation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return [output[row["case_key"]] for row in cases]


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "semantic_accuracy": None,
            "protocol_accuracy": None,
            "mean_semantic_score": None,
            "median_generated_tokens": None,
            "error_classes": {},
        }
    return {
        "count": len(rows),
        "semantic_accuracy": float(np.mean([
            row["semantic_hit"] for row in rows
        ])),
        "protocol_accuracy": float(np.mean([
            row["protocol_hit"] for row in rows
        ])),
        "mean_semantic_score": float(np.mean([
            row["semantic_score"] for row in rows
        ])),
        "median_generated_tokens": float(np.median([
            row["generated_token_count"] for row in rows
        ])),
        "error_classes": dict(Counter(
            row["error_class"] for row in rows
        )),
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    protocol_digest: str,
    placement: dict[str, Any],
    runtime_audit: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    by_family = {
        family: group_summary([
            row for row in rows if row["family"] == family
        ])
        for family in sorted({row["family"] for row in rows})
    }
    nonidentity_translation = [
        row for row in rows
        if row["family"] == "translation"
        and not row["surface_identity"]
    ]
    direction = {}
    for source, target in protocol.LANGUAGE_DIRECTIONS:
        values = [
            row for row in nonidentity_translation
            if row["source_language"] == source
            and row["target_language"] == target
        ]
        direction[f"{source}_{target}"] = group_summary(values)
    split = {}
    for family in sorted(by_family):
        for prompt_split in protocol.PROMPT_SPLITS:
            split[f"{family}|{prompt_split}"] = group_summary([
                row for row in rows
                if row["family"] == family
                and row["prompt_split"] == prompt_split
            ])
    template = {}
    for prompt_split in protocol.PROMPT_SPLITS:
        values = [
            row for row in nonidentity_translation
            if row["prompt_split"] == prompt_split
        ]
        template[prompt_split] = group_summary(values)
    rare_terms = {
        term: group_summary([
            row for row in rows
            if row["family"] == "rare_definition"
            and row["concept_id"] == term
        ])
        for term, _, _ in protocol.RARE_TERMS
    }
    return {
        "schema_version": "phase1023_behavior_summary.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": protocol_digest,
        "model": model_name,
        "precision": "fp16",
        "quantization": "none",
        "placement": placement,
        "runtime_precision_audit": runtime_audit,
        "case_count": len(rows),
        "overall": group_summary(rows),
        "family": by_family,
        "translation_nonidentity": group_summary(nonidentity_translation),
        "translation_direction": direction,
        "translation_split": template,
        "family_split": split,
        "rare_term": rare_terms,
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()

    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"behavior.{args.model}.jsonl"
    )
    started = time.time()
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        runtime_audit = quantization_audit(model)
        if (
            runtime_audit["has_quantized_modules"]
            or runtime_audit["has_bf16_parameters"]
            or not runtime_audit["has_fp16_parameters"]
        ):
            raise RuntimeError(
                "FP16/no-quantization audit failed: "
                + json.dumps(runtime_audit)
            )
        rows = run_generation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=args.model,
            cases=cases,
        )
        summary = summarize(
            rows,
            model_name=args.model,
            protocol_digest=prereg["protocol_digest"],
            placement=placement,
            runtime_audit=runtime_audit,
            elapsed_seconds=time.time() - started,
        )
        output_root = protocol.OUT_ROOT / "behavior" / args.model
        protocol.write_jsonl(output_root / "formal.jsonl", rows)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary["family"], ensure_ascii=False, indent=2))
        print(
            json.dumps(
                summary["translation_nonidentity"],
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


if __name__ == "__main__":
    main()
