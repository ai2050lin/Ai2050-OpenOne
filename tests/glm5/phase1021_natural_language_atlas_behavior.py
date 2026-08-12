#!/usr/bin/env python3
"""Run candidate-free Phase1021 behavior qualification.

Greedy generation is the primary behavior measure.  A teacher-forced gold
versus hidden-foil margin is retained only as a diagnostic and is never shown
inside the prompt.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import release_model
from phase1014_bf16_precision_confirmation import load_bf16
import phase1021_natural_language_atlas_protocol as protocol


CALIBRATION_LIMITS = {
    "multilingual_operation": 160,
    "rare_definition": 64,
    "punctuation_next": 64,
    "contrast_relation": 64,
}
GENERATION_BATCH = {
    "qwen3": 32,
    "glm4": 16,
    "deepseek7b": 16,
}
SCORE_BATCH = {
    "qwen3": 32,
    "glm4": 16,
    "deepseek7b": 16,
}
PUNCTUATION = ".?!,;:。？！,，；："


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def round_robin_subset(
    rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["subgroup"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["record_id"])
    result = []
    cursor = 0
    names = sorted(grouped)
    while len(result) < limit:
        added = False
        for name in names:
            values = grouped[name]
            if cursor < len(values) and len(result) < limit:
                result.append(values[cursor])
                added = True
        if not added:
            break
        cursor += 1
    return result


def calibration_cases(
    model_name: str,
    prompt_mode: str,
    family: str,
) -> list[dict[str, Any]]:
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.{prompt_mode}.jsonl"
    )
    rows = [
        row
        for row in rows
        if row["family"] == family
        and row["split"] == "discovery"
        and row["state"] in ("b0_l0", "b1_l0")
    ]
    return round_robin_subset(rows, CALIBRATION_LIMITS[family])


def formal_cases(
    model_name: str,
    selected_modes: dict[str, str],
) -> list[dict[str, Any]]:
    by_mode = {
        mode: protocol.read_jsonl(
            protocol.OUT_ROOT
            / "protocol"
            / f"cases.{model_name}.{mode}.jsonl"
        )
        for mode in protocol.PROMPT_MODES
    }
    rows = []
    for family in protocol.FAMILIES:
        mode = selected_modes[family]
        rows.extend([
            row
            for row in by_mode[mode]
            if row["family"] == family
            and row["state"] in protocol.FACTORIAL_STATES
        ])
    return sorted(rows, key=lambda row: row["record_id"])


def strip_accents(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def normalized(value: str, *, accents: bool) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    value = re.sub(
        r"^(?:answer|output|result|mark|答案|输出|结果|标点)\s*[:：]\s*",
        "",
        value,
    )
    value = value.strip("`*_\"'“”‘’ ")
    value = re.sub(r"\s+", " ", value)
    if not accents:
        value = strip_accents(value)
    return value.strip()


def first_nonempty_line(value: str) -> str:
    for line in value.replace("\r", "\n").split("\n"):
        if line.strip():
            return line.strip()
    return ""


def cleaned_output(value: str, evaluation_type: str) -> str:
    line = first_nonempty_line(value)
    line = re.sub(r"^<think>.*?</think>", "", line).strip()
    if evaluation_type == "punctuation":
        for char in line:
            if char in PUNCTUATION:
                return char
        return line[:1]
    line = normalized(line, accents=True)
    if evaluation_type != "sentence":
        line = line.rstrip(".?!;:。？！；：")
    return line.strip()


def sentence_score(generated: str, accepted: list[str]) -> float:
    generated_norm = normalized(generated, accents=False)
    return max(
        SequenceMatcher(
            None,
            generated_norm,
            normalized(value, accents=False),
        ).ratio()
        for value in accepted
    )


def evaluate_generation(
    case: dict[str, Any],
    generated_text: str,
) -> dict[str, Any]:
    evaluation_type = case["evaluation_type"]
    cleaned = cleaned_output(generated_text, evaluation_type)
    accepted_exact = [
        normalized(value, accents=True).rstrip(".?!;:。？！；：")
        for value in case["accepted_outputs"]
    ]
    accepted_loose = [
        normalized(value, accents=False).rstrip(".?!;:。？！；：")
        for value in case["accepted_outputs"]
    ]

    if evaluation_type == "punctuation":
        canonical = {
            "。": ".",
            "？": "?",
            "！": "!",
            "，": ",",
            "；": ";",
            "：": ":",
        }
        observed = canonical.get(cleaned[:1], cleaned[:1])
        expected = {
            canonical.get(value[:1], value[:1])
            for value in case["accepted_outputs"]
        }
        exact_hit = normalized(cleaned[:1], accents=True) in {
            normalized(value[:1], accents=True)
            for value in case["accepted_outputs"]
        }
        semantic_hit = observed in expected
        semantic_score = 1.0 if semantic_hit else 0.0
    else:
        cleaned_exact = normalized(cleaned, accents=True).rstrip(
            ".?!;:。？！；："
        )
        cleaned_loose = normalized(cleaned, accents=False).rstrip(
            ".?!;:。？！；："
        )
        exact_hit = cleaned_exact in accepted_exact
        semantic_score = 1.0 if exact_hit else 0.0

    if evaluation_type == "punctuation":
        pass
    elif evaluation_type == "sentence":
        semantic_score = sentence_score(cleaned, case["accepted_outputs"])
        semantic_hit = semantic_score >= 0.72
    elif case["family"] == "rare_definition":
        semantic_hit = cleaned_loose in accepted_loose
        if not semantic_hit and len(cleaned_loose) <= 24:
            semantic_hit = any(
                value and value in cleaned_loose
                for value in accepted_loose
            )
        semantic_score = 1.0 if semantic_hit else max(
            SequenceMatcher(None, cleaned_loose, value).ratio()
            for value in accepted_loose
        )
    else:
        semantic_hit = cleaned_loose in accepted_loose
        semantic_score = 1.0 if semantic_hit else max(
            SequenceMatcher(None, cleaned_loose, value).ratio()
            for value in accepted_loose
        )
    return {
        "generated_text": generated_text,
        "cleaned_output": cleaned,
        "exact_hit": bool(exact_hit),
        "semantic_hit": bool(semantic_hit),
        "semantic_score": float(semantic_score),
    }


def lexical_first_hit(
    cleaned: str,
    accepted_outputs: list[str],
) -> bool:
    observed = normalized(cleaned, accents=False)
    if not observed:
        return False
    for value in accepted_outputs:
        expected = normalized(value, accents=False)
        if not expected:
            continue
        if (
            observed[0] == expected[0]
            or observed.split(" ", 1)[0] == expected.split(" ", 1)[0]
        ):
            return True
    return False


def generation_groups(
    rows: list[dict[str, Any]],
) -> Iterable[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        width_bucket = (len(row["input_ids"]) // 32) * 32
        grouped[(int(row["max_new_tokens"]), width_bucket)].append(row)
    for key in sorted(grouped):
        yield grouped[key]


def generate_rows(
    *,
    model,
    tokenizer,
    device,
    model_name: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    pad_id = int(tokenizer.pad_token_id)
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
                pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for index, row in enumerate(batch):
                values = torch.tensor(
                    row["input_ids"], dtype=torch.long, device=device
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
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            continuation = generated[:, width:].detach().cpu()
            for index, row in enumerate(batch):
                text = tokenizer.decode(
                    continuation[index],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                evaluation = evaluate_generation(row, text)
                evaluation["first_token_hit"] = lexical_first_hit(
                    evaluation["cleaned_output"],
                    row["accepted_outputs"],
                )
                results[row["record_id"]] = evaluation
            del generated, continuation, input_ids, attention_mask
            if batch_index % 20 == 0:
                print(
                    f"[generate] {model_name} batch={batch_index} "
                    f"rows={len(results)}/{len(cases)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    return [results[row["record_id"]] for row in cases]


def homogeneous_batches(
    rows: list[dict[str, Any]],
    size: int,
) -> Iterable[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[len(row["input_ids"])].append(row)
    for width in sorted(grouped):
        yield from chunks(grouped[width], size)


def teacher_forced_scores(
    *,
    model,
    device,
    model_name: str,
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    entries = []
    for case in cases:
        for label, continuation in (
            ("gold", case["gold_token_ids"]),
            ("foil", case["foil_token_ids"]),
        ):
            entries.append({
                "record_id": case["record_id"],
                "label": label,
                "prompt_length": len(case["input_ids"]),
                "continuation": [int(value) for value in continuation],
                "input_ids": (
                    list(case["input_ids"])
                    + [int(value) for value in continuation]
                ),
            })
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for batch_index, batch in enumerate(
        homogeneous_batches(entries, SCORE_BATCH[model_name]),
        1,
    ):
        input_ids = torch.tensor(
            [row["input_ids"] for row in batch],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits.float()
        for index, entry in enumerate(batch):
            values = []
            prompt_length = int(entry["prompt_length"])
            for offset, token_id in enumerate(entry["continuation"]):
                position_logits = logits[
                    index, prompt_length + offset - 1
                ]
                values.append(float(
                    position_logits[int(token_id)].item()
                    - torch.logsumexp(
                        position_logits, dim=-1
                    ).item()
                ))
            result[entry["record_id"]][entry["label"]] = float(
                np.mean(values)
            )
        del output, logits, input_ids, attention_mask
        if batch_index % 40 == 0:
            print(
                f"[score] {model_name} batch={batch_index}",
                flush=True,
            )
    return dict(result)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "exact_accuracy": float(np.mean([
            row["exact_hit"] for row in rows
        ])) if rows else None,
        "semantic_accuracy": float(np.mean([
            row["semantic_hit"] for row in rows
        ])) if rows else None,
        "mean_semantic_score": float(np.mean([
            row["semantic_score"] for row in rows
        ])) if rows else None,
        "first_token_accuracy": float(np.mean([
            row["first_token_hit"] for row in rows
        ])) if rows else None,
        "median_candidate_margin": float(np.median([
            row["candidate_margin"] for row in rows
        ])) if rows and "candidate_margin" in rows[0] else None,
    }


def attach_rows(
    cases: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    scores: dict[str, dict[str, float]] | None,
    *,
    model_name: str,
    scope: str,
) -> list[dict[str, Any]]:
    rows = []
    for case, observation in zip(cases, generated):
        score = (scores or {}).get(case["record_id"], {})
        margin = (
            float(score["gold"] - score["foil"])
            if score else float("nan")
        )
        rows.append({
            "schema_version": "phase1021_natural_behavior_row.v1",
            "phase": protocol.PHASE,
            "protocol_revision": protocol.PROTOCOL_REVISION,
            "model": model_name,
            "prompt_mode": case["prompt_mode"],
            "scope": scope,
            "record_id": case["record_id"],
            "unit_id": case["unit_id"],
            "family": case["family"],
            "item_id": case["item_id"],
            "subgroup": case["subgroup"],
            "task_kind": case["task_kind"],
            "split": case["split"],
            "state": case["state"],
            "world": int(case["world"]),
            "gold": case["gold"],
            "foil": case["foil"],
            "accepted_outputs": case["accepted_outputs"],
            "generated_text": observation["generated_text"],
            "cleaned_output": observation["cleaned_output"],
            "exact_hit": observation["exact_hit"],
            "semantic_hit": observation["semantic_hit"],
            "semantic_score": observation["semantic_score"],
            "candidate_hit": observation["semantic_hit"],
            "candidate_margin": margin,
            "gold_mean_log_probability": score.get("gold"),
            "foil_mean_log_probability": score.get("foil"),
            "first_token_hit": observation["first_token_hit"],
            "operation": case.get("operation"),
            "source_language": case.get("source_language"),
            "target_language": case.get("target_language"),
            "concept_id": case.get("concept_id"),
            "term": case.get("term"),
            "evaluation_type": case["evaluation_type"],
        })
    return rows


def selection_metrics(rows: list[dict[str, Any]]) -> tuple[float, float]:
    summary = summarize(rows)
    return (
        float(summary["semantic_accuracy"] or 0.0),
        float(summary["exact_accuracy"] or 0.0),
    )


def run_model(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    model = tokenizer = device = None
    output_root = protocol.OUT_ROOT / "behavior" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        calibration_by_mode: dict[str, list[dict[str, Any]]] = {}
        calibration_summaries = {}
        for prompt_mode in protocol.PROMPT_MODES:
            mode_rows = []
            for family in protocol.FAMILIES:
                cases = calibration_cases(
                    model_name, prompt_mode, family
                )
                generated = generate_rows(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    model_name=model_name,
                    cases=cases,
                )
                rows = attach_rows(
                    cases,
                    generated,
                    None,
                    model_name=model_name,
                    scope="calibration",
                )
                mode_rows.extend(rows)
                calibration_summaries[(prompt_mode, family)] = summarize(
                    rows
                )
            calibration_by_mode[prompt_mode] = mode_rows
            protocol.write_jsonl(
                output_root / f"calibration.{prompt_mode}.jsonl",
                mode_rows,
            )
            protocol.write_json(
                output_root / f"calibration.{prompt_mode}.summary.json",
                {
                    family: calibration_summaries[(prompt_mode, family)]
                    for family in protocol.FAMILIES
                },
            )

        selected_modes = {}
        selection_rows = []
        for family in protocol.FAMILIES:
            candidates = []
            for prompt_mode in protocol.PROMPT_MODES:
                family_rows = [
                    row
                    for row in calibration_by_mode[prompt_mode]
                    if row["family"] == family
                ]
                candidates.append((
                    selection_metrics(family_rows),
                    prompt_mode,
                ))
            candidates.sort(reverse=True)
            selected_modes[family] = candidates[0][1]
            selection_rows.append({
                "family": family,
                "selected_prompt_mode": candidates[0][1],
                "ranked_modes": [
                    {
                        "prompt_mode": mode,
                        "semantic_accuracy": metric[0],
                        "exact_accuracy": metric[1],
                    }
                    for metric, mode in candidates
                ],
            })

        cases = formal_cases(model_name, selected_modes)
        generated = generate_rows(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=model_name,
            cases=cases,
        )
        scores = teacher_forced_scores(
            model=model,
            device=device,
            model_name=model_name,
            cases=cases,
        )
        rows = attach_rows(
            cases,
            generated,
            scores,
            model_name=model_name,
            scope="formal",
        )
        protocol.write_jsonl(output_root / "formal.jsonl", rows)

        family_summary = {
            family: summarize([
                row for row in rows if row["family"] == family
            ])
            for family in protocol.FAMILIES
        }
        subgroup_summary = {
            subgroup: summarize([
                row for row in rows if row["subgroup"] == subgroup
            ])
            for subgroup in sorted({
                row["subgroup"] for row in rows
            })
        }
        task_summary = {
            task: summarize([
                row for row in rows if row["task_kind"] == task
            ])
            for task in sorted({
                row["task_kind"] for row in rows
            })
        }
        summary = {
            "schema_version": "phase1021_natural_behavior_summary.v1",
            "phase": protocol.PHASE,
            "protocol_revision": protocol.PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "precision": "bf16",
            "placement": placement,
            "case_count": len(rows),
            "selected_by_family": selected_modes,
            "family": family_summary,
            "subgroup": subgroup_summary,
            "task_kind": task_summary,
            "elapsed_seconds": time.time() - started,
        }
        protocol.write_json(output_root / "formal.summary.json", summary)
        protocol.write_json(
            output_root / "selection.json",
            {
                "schema_version": "phase1021_prompt_selection.v1",
                "phase": protocol.PHASE,
                "protocol_revision": protocol.PROTOCOL_REVISION,
                "protocol_digest": prereg["protocol_digest"],
                "model": model_name,
                "selected_by_family": selected_modes,
                "selection": selection_rows,
            },
        )
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
