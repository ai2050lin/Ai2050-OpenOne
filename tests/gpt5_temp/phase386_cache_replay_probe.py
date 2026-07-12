#!/usr/bin/env python3
"""Temporary probe for full-replay versus incremental-cache Phase386 logits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402


FAILED = {
    "p386c_64ea46122a245206b10284aa56",
    "p386c_1b977d8db4b38e55679a104177",
}
CASES = (
    ROOT
    / "tests/gpt5/result/phase386_multitime_relation_atlas/protocol/private/phase386_discovery_cases.jsonl"
)
OUT = (
    ROOT
    / "tests/gpt5/result/phase386_multitime_relation_atlas/phase386_cache_path_probe.json"
)


def main() -> None:
    rows = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        row
        for row in rows
        if row["private_execution_model"] == "deepseek7b"
        and row["blind_case_id"] in FAILED
    ]
    loaded = None
    output_rows = []
    try:
        loaded = load_probe_model("deepseek7b")
        for case in rows:
            base = loaded.tokenizer(
                case["prompt"],
                add_special_tokens=bool(case["tokenization_add_special_tokens"]),
                truncation=True,
                max_length=256,
            )["input_ids"]
            generated = [int(value) for value in case["generated_token_ids"]]
            step = int(case["target_decision_step"])
            pre = [*base, *generated[:step]]
            target_token = generated[step]
            expected_next = generated[step + 1]
            ids = torch.tensor([pre], dtype=torch.long, device=loaded.input_device)
            first = loaded.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=True,
                return_dict=True,
            )
            target_ids = torch.tensor(
                [[target_token]], dtype=torch.long, device=loaded.input_device
            )
            target_attention = torch.ones(
                (1, len(pre) + 1), dtype=torch.long, device=loaded.input_device
            )
            incremental = loaded.model(
                input_ids=target_ids,
                attention_mask=target_attention,
                past_key_values=first.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            prompt_ids = torch.tensor(
                [base], dtype=torch.long, device=loaded.input_device
            )
            actual = loaded.model(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                use_cache=True,
                return_dict=True,
            )
            past = actual.past_key_values
            running_length = len(base)
            for prefix_token in generated[:step]:
                running_length += 1
                prefix_ids = torch.tensor(
                    [[prefix_token]], dtype=torch.long, device=loaded.input_device
                )
                prefix_attention = torch.ones(
                    (1, running_length),
                    dtype=torch.long,
                    device=loaded.input_device,
                )
                actual = loaded.model(
                    input_ids=prefix_ids,
                    attention_mask=prefix_attention,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = actual.past_key_values
            running_length += 1
            actual_target_attention = torch.ones(
                (1, running_length),
                dtype=torch.long,
                device=loaded.input_device,
            )
            actual_incremental = loaded.model(
                input_ids=target_ids,
                attention_mask=actual_target_attention,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            full_ids = torch.tensor(
                [[*pre, target_token]],
                dtype=torch.long,
                device=loaded.input_device,
            )
            full = loaded.model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
                return_dict=True,
            )
            incremental_logits = incremental.logits[0, -1].float()
            actual_incremental_logits = actual_incremental.logits[0, -1].float()
            full_logits = full.logits[0, -1].float()
            output_rows.append(
                {
                    "blind_case_id": case["blind_case_id"],
                    "target_decision_step": step,
                    "expected_next": expected_next,
                    "incremental_argmax": int(incremental_logits.argmax().item()),
                    "actual_incremental_argmax": int(
                        actual_incremental_logits.argmax().item()
                    ),
                    "full_argmax": int(full_logits.argmax().item()),
                    "incremental_expected_rank": 1
                    + int((incremental_logits > incremental_logits[expected_next]).sum().item()),
                    "actual_incremental_expected_rank": 1
                    + int(
                        (
                            actual_incremental_logits
                            > actual_incremental_logits[expected_next]
                        ).sum().item()
                    ),
                    "full_expected_rank": 1
                    + int((full_logits > full_logits[expected_next]).sum().item()),
                    "max_logit_abs_difference": float(
                        (incremental_logits - full_logits).abs().max().item()
                    ),
                    "actual_vs_full_max_logit_abs_difference": float(
                        (actual_incremental_logits - full_logits).abs().max().item()
                    ),
                }
            )
        payload = {
            "schema_version": "60.4.1",
            "phase_id": "Phase386-CachePathProbe",
            "model": "deepseek7b",
            "case_count": len(output_rows),
            "rows": output_rows,
            "all_actual_incremental_transitions_match": all(
                row["actual_incremental_argmax"] == row["expected_next"]
                for row in output_rows
            ),
            "teacher_forced_full_replay_is_equivalent": False,
        }
        OUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        release_loaded(loaded)


if __name__ == "__main__":
    main()
