#!/usr/bin/env python3
"""Diagnose the frozen Phase400 DS7B first-format-token replay failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase371c_behavior_qualification import generate_batch  # noqa: E402
from phase358_multiresolution_component_conservation import install_hooks  # noqa: E402
from phase400_partial_order_common import OUT, now, read_jsonl, write_json  # noqa: E402


FAILED_CASE_ID = "p400c_513594036762abe308a66fe877"


def decode(loaded: Any, token_id: int) -> str:
    return loaded.tokenizer.decode([token_id], skip_special_tokens=False)


@torch.inference_mode()
def main() -> None:
    cases = read_jsonl(
        OUT
        / "dynamic_trace/protocol/private/phase400_calibration_dynamic_trace_cases.jsonl"
    )
    case = next(row for row in cases if row["blind_case_id"] == FAILED_CASE_ID)
    candidate_cases = [
        row
        for row in read_jsonl(OUT / "protocol/private/phase400_candidate_cases.jsonl")
        if row["private_execution_model"] == "deepseek7b"
    ]
    same_length = [
        row
        for row in candidate_cases
        if row["prompt_token_count"] == case["prompt_token_count"]
    ]
    failed_index = next(
        index for index, row in enumerate(same_length) if row["blind_case_id"] == FAILED_CASE_ID
    )
    batch_start = (failed_index // 8) * 8
    original_batch = same_length[batch_start : batch_start + 8]
    loaded = None
    handles: list[Any] = []
    try:
        loaded = load_probe_model("deepseek7b")
        ids = torch.tensor(
            [case["prompt_token_ids_private"]],
            dtype=torch.long,
            device=loaded.input_device,
        )
        modes = []
        for name, selected in (
            ("generate_batch_size_1", [case]),
            ("generate_original_batch_size_8", original_batch),
        ):
            generated = generate_batch(loaded, selected, max_new_tokens=8)
            failed_result = next(
                result
                for selected_case, result in zip(selected, generated, strict=True)
                if selected_case["blind_case_id"] == FAILED_CASE_ID
            )
            token_ids = failed_result["generated_token_ids"]
            modes.append(
                {
                    "mode": name,
                    "predicted_token_ids": token_ids[:1],
                    "predicted_token_text": [decode(loaded, token_ids[0])],
                    "full_generated_token_ids": token_ids,
                    "full_generated_text": failed_result["generated_text"],
                    "deterministic_within_loaded_model": True,
                    "batch_case_ids_private": [
                        selected_case["blind_case_id"] for selected_case in selected
                    ],
                }
            )
        for name, output_attentions, use_cache in (
            ("plain_forward", False, False),
            ("attention_output_forward", True, True),
        ):
            predictions = []
            for _ in range(3):
                output = loaded.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    output_attentions=output_attentions,
                    output_hidden_states=False,
                    use_cache=use_cache,
                    return_dict=True,
                )
                predictions.append(int(torch.argmax(output.logits[0, -1]).item()))
                del output
            modes.append(
                {
                    "mode": name,
                    "predicted_token_ids": predictions,
                    "predicted_token_text": [decode(loaded, value) for value in predictions],
                    "deterministic_within_loaded_model": len(set(predictions)) == 1,
                }
            )
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(get_layers(loaded.model), captures)
        hooked_predictions = []
        for _ in range(3):
            captures.clear()
            output = loaded.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_attentions=True,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            hooked_predictions.append(int(torch.argmax(output.logits[0, -1]).item()))
            del output
        modes.append(
            {
                "mode": "full_parent_capture_hooks",
                "predicted_token_ids": hooked_predictions,
                "predicted_token_text": [
                    decode(loaded, value) for value in hooked_predictions
                ],
                "deterministic_within_loaded_model": len(set(hooked_predictions)) == 1,
            }
        )
        expected = int(case["first_answer_token_id_private"])
        payload = {
            "schema_version": "phase400.failed_replay_diagnostic.v1",
            "phase_id": "Phase400-FailedReplayDiagnostic",
            "created_at": now(),
            "model": "deepseek7b",
            "blind_case_id_private": FAILED_CASE_ID,
            "cached_behavior_first_token_id": expected,
            "cached_behavior_first_token_text": decode(loaded, expected),
            "semantic_target_token_id": int(case["target_first_token_id_private"]),
            "semantic_target_token_text": decode(
                loaded, int(case["target_first_token_id_private"])
            ),
            "modes": modes,
            "plain_vs_hooked_equal": modes[2]["predicted_token_ids"]
            == modes[4]["predicted_token_ids"],
            "batch_size_1_vs_8_equal": modes[0]["predicted_token_ids"]
            == modes[1]["predicted_token_ids"],
            "cached_behavior_reproduced_now": any(
                expected in item["predicted_token_ids"] for item in modes
            ),
            "private_only": True,
        }
        write_json(
            OUT
            / "dynamic_trace/calibration/private/phase400_failed_replay_diagnostic.json",
            payload,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
