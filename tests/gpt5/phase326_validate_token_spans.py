#!/usr/bin/env python3
"""Decode every registered Phase326 source/query span for all three tokenizers."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase326_distributed_carrier_case_bank as case_bank  # noqa: E402
from model_registry import MODEL_SPECS  # noqa: E402
from phase326_distributed_carrier_atlas import role_spans  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas"


def word_overlap(source: str, decoded: str) -> float:
    words = [word for word in re.findall(r"[a-z]+", source.lower()) if len(word) > 2]
    return sum(word in decoded.lower() for word in words) / max(1, len(words))


def main() -> None:
    cases = [*case_bank.build_cases(), *case_bank.build_confirmation_cases()]
    results = []
    for model, spec in MODEL_SPECS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=True, local_files_only=True, use_fast=False
        )
        failures = []
        for case in cases:
            ids = tokenizer(case["prompt"], add_special_tokens=True)["input_ids"]
            spans = role_spans(tokenizer, case["prompt"], case, len(ids))
            expected = {
                "source": " ".join(case["source_fragments"]),
                "query": case["query_fragment"],
            }
            for role, source in expected.items():
                start, end = spans[role]
                decoded = tokenizer.decode(ids[start : end + 1], skip_special_tokens=True)
                overlap = word_overlap(source, decoded)
                if overlap < 0.5:
                    failures.append({
                        "case_id": case["case_id"], "role": role, "span": [start, end],
                        "expected": source, "decoded": decoded, "word_overlap": overlap,
                    })
        results.append({
            "model": model,
            "case_count": len(cases),
            "checked_span_count": len(cases) * 2,
            "failure_count": len(failures),
            "failures": failures[:20],
        })
    payload = {
        "schema_version": "phase326_token_span_validation.v1",
        "phase_id": "Phase326",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": all(row["failure_count"] == 0 for row in results),
        "minimum_word_overlap": 0.5,
        "models": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase326_token_span_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
