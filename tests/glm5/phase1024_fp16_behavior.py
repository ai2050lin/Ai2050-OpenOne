#!/usr/bin/env python3
"""Run Phase1024 FP16 behavior and numerical qualification.

Candidate scoring is teacher-forced and length-normalized.  It is reported
separately from free generation because Phase1023 exposed severe FP16
generation degeneration in GLM4 and DeepSeek7B.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

import phase1024_lexical_semantic_protocol as protocol
from phase1023_fp16_utils import (
    MODELS,
    load_fp16,
    quantization_audit,
    release_fp16,
)


BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def append_candidate(tokenizer, prefix: list[int], candidate: str) -> list[int]:
    target = [
        int(value)
        for value in tokenizer.encode(
            " " + candidate.strip(),
            add_special_tokens=False,
        )
    ]
    if not target:
        raise RuntimeError(f"empty candidate tokens for {candidate!r}")
    return prefix + target


def score_candidates(
    model,
    tokenizer,
    device: torch.device,
    cases: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded = []
    for case_index, row in enumerate(cases):
        candidates = list(row["candidate_outputs"])
        for candidate_index, candidate in enumerate(candidates):
            prefix = list(row["input_ids"])
            sequence = append_candidate(tokenizer, prefix, candidate)
            expanded.append({
                "case_index": case_index,
                "candidate_index": candidate_index,
                "candidate": candidate,
                "prefix_length": len(prefix),
                "sequence": sequence,
                "target_length": len(sequence) - len(prefix),
            })

    scores: dict[int, list[dict[str, Any]]] = defaultdict(list)
    nonfinite_logits = 0
    inspected_logits = 0
    for batch_index, batch in enumerate(chunks(expanded, batch_size), 1):
        width = max(len(row["sequence"]) for row in batch)
        input_ids = torch.full(
            (len(batch), width),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(batch), width),
            dtype=torch.long,
        )
        for index, row in enumerate(batch):
            values = torch.tensor(row["sequence"], dtype=torch.long)
            input_ids[index, :len(values)] = values
            attention_mask[index, :len(values)] = 1
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            ).logits
        inspected_logits += int(logits.numel())
        nonfinite_logits += int((~torch.isfinite(logits)).sum().item())
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for index, row in enumerate(batch):
            token_scores = []
            start = int(row["prefix_length"])
            sequence = row["sequence"]
            for position in range(start, len(sequence)):
                token_scores.append(float(
                    log_probs[
                        index,
                        position - 1,
                        int(sequence[position]),
                    ].item()
                ))
            scores[row["case_index"]].append({
                "candidate": row["candidate"],
                "candidate_index": row["candidate_index"],
                "target_token_count": len(token_scores),
                "mean_log_probability": (
                    sum(token_scores) / max(len(token_scores), 1)
                ),
                "total_log_probability": sum(token_scores),
            })
        if batch_index % 20 == 0:
            print(
                f"[phase1024-behavior] candidate batch={batch_index} "
                f"sequences={min(batch_index * batch_size, len(expanded))}/"
                f"{len(expanded)}",
                flush=True,
            )

    result = []
    for case_index, row in enumerate(cases):
        candidate_scores = sorted(
            scores[case_index],
            key=lambda value: value["candidate_index"],
        )
        predicted = max(
            candidate_scores,
            key=lambda value: value["mean_log_probability"],
        )["candidate"]
        accepted = {normalize(value) for value in row["accepted_outputs"]}
        result.append({
            "schema_version": "phase1024_candidate_score.v1",
            "record_id": row["record_id"],
            "panel": row["panel"],
            "split": row["split"],
            "partition": row["partition"],
            "surface": row["surface"],
            "concept": row.get("concept"),
            "family": row.get("family"),
            "word": row.get("word"),
            "sense_id": row.get("sense_id"),
            "predicted": predicted,
            "correct": normalize(predicted) in accepted,
            "candidate_scores": candidate_scores,
        })
    return result, {
        "expanded_sequence_count": len(expanded),
        "inspected_logit_count": inspected_logits,
        "nonfinite_logit_count": nonfinite_logits,
        "all_logits_finite": nonfinite_logits == 0,
    }


def generation_correct(row: dict[str, Any], generated: str) -> bool:
    value = normalize(generated)
    accepted = [normalize(item) for item in row["accepted"]]
    if row["family"] == "punctuation":
        compact = re.sub(r"\s+", "", generated)
        return any(item in compact[:3] for item in row["accepted"])
    return any(item in value for item in accepted)


def run_qualification(
    model,
    tokenizer,
    device: torch.device,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(cases, 1):
        input_ids = torch.tensor(
            [row["input_ids"]],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(row["max_new_tokens"]),
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated_ids[0, input_ids.shape[1]:].detach().cpu()
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)
        compact = re.sub(r"\s+", "", generated)
        repeated_single = (
            len(compact) >= 4 and len(set(compact)) == 1
        )
        hit_limit = int(new_ids.numel()) >= int(row["max_new_tokens"])
        rows.append({
            "schema_version": "phase1024_qualification_result.v1",
            "case_key": row["case_key"],
            "family": row["family"],
            "generated": generated,
            "generated_token_count": int(new_ids.numel()),
            "correct": generation_correct(row, generated),
            "hit_token_limit": hit_limit,
            "repeated_single_symbol": repeated_single,
        })
        print(
            f"[phase1024-qualification] {index}/{len(cases)} "
            f"{row['case_key']} correct={rows[-1]['correct']} "
            f"text={generated[:80]!r}",
            flush=True,
        )
    return rows


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return (
        sum(bool(row[key]) for row in rows) / len(rows)
        if rows else float("nan")
    )


def group_rates(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    result = {}
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        label = "|".join(str(value) for value in group)
        result[label] = {
            "count": len(values),
            "accuracy": rate(values, "correct"),
        }
    return result


def summarize(
    model_name: str,
    candidate_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
    numerical_audit: dict[str, Any],
    placement: dict[str, Any],
    precision_audit: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    nonce = [
        row for row in candidate_rows if row["panel"] == "nonce_binding"
    ]
    poly = [row for row in candidate_rows if row["panel"] == "polysemy"]
    return {
        "schema_version": "phase1024_behavior_summary.v1",
        "phase": protocol.PHASE,
        "model": model_name,
        "precision": "fp16",
        "quantization": "none",
        "placement": placement,
        "runtime_precision_audit": precision_audit,
        "numerical_audit": numerical_audit,
        "candidate_case_count": len(candidate_rows),
        "nonce": {
            "overall_accuracy": rate(nonce, "correct"),
            "by_split": group_rates(nonce, ("split",)),
            "by_family": group_rates(nonce, ("split", "family")),
            "chance": 1.0 / len(protocol.FAMILIES),
        },
        "polysemy": {
            "overall_accuracy": rate(poly, "correct"),
            "by_partition_split": group_rates(
                poly, ("partition", "split")
            ),
            "chance": 0.5,
        },
        "qualification": {
            "case_count": len(qualification_rows),
            "accuracy": rate(qualification_rows, "correct"),
            "hit_token_limit_rate": rate(
                qualification_rows, "hit_token_limit"
            ),
            "repeated_single_symbol_rate": rate(
                qualification_rows, "repeated_single_symbol"
            ),
            "by_family": group_rates(qualification_rows, ("family",)),
        },
        "behavior_claim_qualified": (
            numerical_audit["all_logits_finite"]
            and rate(qualification_rows, "hit_token_limit") <= 0.10
            and rate(
                qualification_rows, "repeated_single_symbol"
            ) <= 0.10
        ),
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()

    protocol_dir = protocol.OUT_ROOT / "protocol"
    cases = protocol.read_jsonl(
        protocol_dir / f"cases.{args.model}.jsonl"
    )
    qualification = protocol.read_jsonl(
        protocol_dir / f"qualification.{args.model}.jsonl"
    )
    scored_cases = [
        row for row in cases
        if row["panel"] in ("nonce_binding", "polysemy")
    ]
    started = time.time()
    model = tokenizer = None
    out_dir = protocol.OUT_ROOT / "behavior" / args.model
    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        precision_audit = quantization_audit(model)
        if (
            precision_audit["has_quantized_modules"]
            or precision_audit["has_bf16_parameters"]
            or not precision_audit["has_fp16_parameters"]
        ):
            raise RuntimeError(
                "FP16/no-quantization audit failed: "
                + json.dumps(precision_audit)
            )
        candidate_path = out_dir / "candidate_scores.jsonl"
        numerical_path = out_dir / "candidate_numerical_audit.json"
        if candidate_path.exists() and numerical_path.exists():
            candidate_rows = protocol.read_jsonl(candidate_path)
            numerical_audit = protocol.read_json(numerical_path)
            if len(candidate_rows) != len(scored_cases):
                raise RuntimeError("candidate checkpoint length drift")
            print(
                "[phase1024-behavior] reusing candidate checkpoint",
                flush=True,
            )
        else:
            candidate_rows, numerical_audit = score_candidates(
                model,
                tokenizer,
                device,
                scored_cases,
                batch_size=BATCH_SIZE[args.model],
            )
            # Checkpoint the expensive branch before free generation.
            protocol.write_jsonl(candidate_path, candidate_rows)
            protocol.write_json(numerical_path, numerical_audit)
        qualification_rows = run_qualification(
            model,
            tokenizer,
            device,
            qualification,
        )
        summary = summarize(
            args.model,
            candidate_rows,
            qualification_rows,
            numerical_audit,
            placement,
            precision_audit,
            time.time() - started,
        )
        protocol.write_jsonl(
            out_dir / "qualification.jsonl",
            qualification_rows,
        )
        protocol.write_json(out_dir / "summary.json", summary)
        print(json.dumps({
            "model": args.model,
            "nonce": summary["nonce"],
            "polysemy": summary["polysemy"],
            "qualification": summary["qualification"],
            "behavior_claim_qualified": summary[
                "behavior_claim_qualified"
            ],
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


if __name__ == "__main__":
    main()
