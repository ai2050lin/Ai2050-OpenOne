#!/usr/bin/env python3
"""Diagnose whether Phase406 H=12 failures are response truncation.

Only formal H=12 failures are extended to 48 tokens.  The result is a
post-discovery diagnostic and is explicitly forbidden from promoting a state
candidate.  Exact prefix horizons 12/24/36/48 are evaluated with the frozen
semantic parser and aliases.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase406_conditioned_sequence_analysis import extract_semantic_label  # noqa: E402
from phase406_conditioned_sequence_protocol import (  # noqa: E402
    BATCH_SIZE_BY_MODEL,
    FROZEN_DTYPES,
    MODELS,
    OUT,
)


SOURCE = OUT / "protocol/private/phase406_all_cases.jsonl"
HORIZONS = (12, 24, 36, 48)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def unresolved_ids(model: str) -> set[str]:
    path = OUT / "analysis/discovery/private" / model / "semantic_rows.jsonl"
    return {
        row["blind_case_id"]
        for row in read_jsonl(path)
        if not row["short_sequence_semantic_correct"]
    }


@torch.inference_mode()
def collect(model: str) -> dict[str, Any]:
    selected_ids = unresolved_ids(model)
    cases = [
        row
        for row in read_jsonl(SOURCE)
        if row["private_execution_model"] == model
        and row["candidate_split_private"] == "discovery"
        and row["blind_case_id"] in selected_ids
    ]
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_length[case["prompt_token_count"]].append(case)

    loaded = None
    rows = []
    batch_count = 0
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPES[model]:
            raise RuntimeError(f"Phase406 horizon dtype mismatch: {runtime_dtype}")
        tokenizer = loaded.tokenizer
        eos_values = [
            tokenizer.eos_token_id,
            loaded.model.generation_config.eos_token_id,
        ]
        eos_ids: set[int] = set()
        for value in eos_values:
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                eos_ids.update(int(item) for item in value)
            else:
                eos_ids.add(int(value))
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("Phase406 horizon diagnostic has no pad token")

        completed = 0
        for prompt_length in sorted(by_length):
            for batch_cases in chunks(
                by_length[prompt_length], BATCH_SIZE_BY_MODEL[model]
            ):
                batch_count += 1
                input_ids = torch.tensor(
                    [case["prompt_token_ids_private"] for case in batch_cases],
                    dtype=torch.long,
                    device=loaded.input_device,
                )
                generated = loaded.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    do_sample=False,
                    max_new_tokens=max(HORIZONS),
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_logits=True,
                    pad_token_id=int(pad_id),
                )
                raw_logits = list(generated.logits or ())
                generated_ids = generated.sequences[:, prompt_length:]
                for index, case in enumerate(batch_cases):
                    ids = [int(item) for item in generated_ids[index].cpu().tolist()]
                    if eos_ids:
                        for position, token_id in enumerate(ids):
                            if token_id in eos_ids:
                                ids = ids[: position + 1]
                                break
                    horizon_rows = []
                    for horizon in HORIZONS:
                        prefix_ids = ids[:horizon]
                        text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
                        parsed = extract_semantic_label(
                            text,
                            case["semantic_candidate_labels_private"],
                            case["semantic_aliases_private"],
                        )
                        relevant_steps = min(horizon, len(raw_logits))
                        logits_valid = all(
                            bool(torch.isfinite(raw_logits[step][index]).all())
                            for step in range(relevant_steps)
                        )
                        correct = (
                            parsed["semantic_label_private"]
                            == case["target_semantic_label_private"]
                            and logits_valid
                        )
                        horizon_rows.append(
                            {
                                "horizon": horizon,
                                "generated_token_count": len(prefix_ids),
                                "semantic_label_private": parsed[
                                    "semantic_label_private"
                                ],
                                "semantic_parse_method": parsed[
                                    "semantic_parse_method"
                                ],
                                "raw_logits_valid": logits_valid,
                                "semantic_correct": correct,
                            }
                        )
                    rows.append(
                        {
                            "schema_version": "80.6.0",
                            "phase_id": "Phase406-HorizonExtensionDiagnostic",
                            "model": model,
                            "blind_case_id": case["blind_case_id"],
                            "family_id": case["family_id"],
                            "condition_id_private": case["condition_id_private"],
                            "target_semantic_label_private": case[
                                "target_semantic_label_private"
                            ],
                            "generated_token_ids_private": ids,
                            "generated_text_clean_private": tokenizer.decode(
                                ids, skip_special_tokens=True
                            ),
                            "horizons": horizon_rows,
                        }
                    )
                completed += len(batch_cases)
                if completed % 128 < len(batch_cases) or completed == len(cases):
                    print(
                        f"[{model}/phase406/horizon] {completed}/{len(cases)}",
                        flush=True,
                    )
                del generated, raw_logits, generated_ids, input_ids

        payload = {
            "schema_version": "80.6.0",
            "phase_id": "Phase406-HorizonExtensionDiagnostic",
            "created_at": now(),
            "model": model,
            "selection": "formal_H12_short_sequence_failures_only",
            "case_count": len(rows),
            "batch_count": batch_count,
            "horizons": list(HORIZONS),
            "correct_count_by_horizon": {
                str(horizon): sum(
                    next(
                        item["semantic_correct"]
                        for item in row["horizons"]
                        if item["horizon"] == horizon
                    )
                    for row in rows
                )
                for horizon in HORIZONS
            },
            "valid": len(rows) == len(cases),
            "is_formal_candidate_gate": False,
            "can_promote_state_candidate": False,
        }
        write_jsonl(
            OUT / "diagnostics/private" / f"phase406_horizon_{model}_rows.jsonl",
            rows,
        )
        write_json(
            OUT / "diagnostics" / f"phase406_horizon_{model}_complete.json",
            payload,
        )
        print(json.dumps(payload, indent=2))
        return payload
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze() -> None:
    completes = [
        read_json(OUT / "diagnostics" / f"phase406_horizon_{model}_complete.json")
        for model in MODELS
    ]
    payload = {
        "schema_version": "80.6.0",
        "phase_id": "Phase406-HorizonExtensionDiagnostic",
        "created_at": now(),
        "selection": "formal_H12_short_sequence_failures_only",
        "case_count": sum(row["case_count"] for row in completes),
        "models": completes,
        "correct_count_by_horizon": {
            str(horizon): sum(
                row["correct_count_by_horizon"][str(horizon)] for row in completes
            )
            for horizon in HORIZONS
        },
        "newly_recovered_after_H12_by_horizon": {
            str(horizon): sum(
                row["correct_count_by_horizon"][str(horizon)] for row in completes
            )
            - sum(row["correct_count_by_horizon"]["12"] for row in completes)
            for horizon in HORIZONS[1:]
        },
        "is_formal_candidate_gate": False,
        "can_promote_state_candidate": False,
        "claim_boundary": {
            "selected_failure_set_is_independent_holdout": False,
            "horizon_with_best_recovery_is_preregistered_primary_endpoint": False,
            "diagnostic_can_only_choose_next_protocol_horizon": True,
        },
    }
    write_json(OUT / "phase406_horizon_extension_diagnostic.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
    elif args.model:
        collect(args.model)
    else:
        raise SystemExit("Use --model MODEL or --analyze")
