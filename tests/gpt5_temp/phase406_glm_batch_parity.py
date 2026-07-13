#!/usr/bin/env python3
"""Diagnose Phase406 GLM4 batch-size numerical parity on frozen cases."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402


SOURCE = (
    ROOT
    / "tests/gpt5/result/phase406_conditioned_sequence_state/protocol/private/phase406_all_cases.jsonl"
)
OUT = (
    ROOT
    / "tests/gpt5/result/phase406_conditioned_sequence_state/diagnostics/phase406_glm_batch_parity.json"
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@torch.inference_mode()
def main() -> None:
    cases = [
        row
        for row in rows(SOURCE)
        if row["private_execution_model"] == "glm4"
        and row["candidate_split_private"] == "discovery"
    ]
    by_length: dict[int, list[dict]] = defaultdict(list)
    for case in cases:
        by_length[case["prompt_token_count"]].append(case)

    selected_batches = []
    for length in sorted(by_length):
        bucket = by_length[length]
        for start in range(0, len(bucket), 4):
            batch = bucket[start : start + 4]
            if length in {58, 59, 60, 77} and len(batch) == 4:
                selected_batches.append(batch)
            if len(selected_batches) >= 12:
                break
        if len(selected_batches) >= 12:
            break

    loaded = None
    records = []
    try:
        loaded = load_probe_model("glm4")
        for batch in selected_batches:
            input_ids = torch.tensor(
                [case["prompt_token_ids_private"] for case in batch],
                dtype=torch.long,
                device=loaded.input_device,
            )
            batched = loaded.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            ).logits[:, -1].float()
            for index, case in enumerate(batch):
                single_ids = input_ids[index : index + 1]
                single = loaded.model(
                    input_ids=single_ids,
                    attention_mask=torch.ones_like(single_ids),
                    use_cache=False,
                    return_dict=True,
                ).logits[0, -1].float()
                batch_finite = bool(torch.isfinite(batched[index]).all())
                single_finite = bool(torch.isfinite(single).all())
                records.append(
                    {
                        "blind_case_id": case["blind_case_id"],
                        "prompt_token_count": case["prompt_token_count"],
                        "family_id": case["family_id"],
                        "condition_id": case["condition_id_private"],
                        "batch4_finite": batch_finite,
                        "batch1_finite": single_finite,
                        "argmax_equal_when_finite": batch_finite
                        and single_finite
                        and int(torch.argmax(batched[index]).item())
                        == int(torch.argmax(single).item()),
                    }
                )
    finally:
        release_loaded(loaded)

    payload = {
        "schema_version": "80.1.0",
        "phase_id": "Phase406-GLMBatchParityDiagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(records),
        "batch4_nonfinite_count": sum(not row["batch4_finite"] for row in records),
        "batch1_nonfinite_count": sum(not row["batch1_finite"] for row in records),
        "finite_argmax_mismatch_count": sum(
            row["batch4_finite"]
            and row["batch1_finite"]
            and not row["argmax_equal_when_finite"]
            for row in records
        ),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
