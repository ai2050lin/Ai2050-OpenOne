#!/usr/bin/env python3
"""Two-case CUDA compatibility smoke test for the Phase330 survey runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
import phase330_global_atlas_survey as survey  # noqa: E402


def main(model: str) -> None:
    cases = survey.read_cases("nine_family_global_atlas", "content_knowledge")[:2]
    loaded = None
    try:
        loaded = load_probe_model(model)
        events = []
        readout, top50 = survey.trace_batch(loaded, cases, model, events)
        phrase = survey.phrase_logprobs(loaded, cases)
        rollout = survey.generate_batch(loaded, cases, 4)
        expected = len(cases) * len(get_layers(loaded.model)) * 3 * 3
        result = {
            "model": model,
            "cases": len(cases),
            "event_rows": len(events),
            "expected_event_rows": expected,
            "readout_rows": len(readout),
            "top50_rows": len(top50),
            "phrase_rows": len(phrase),
            "rollout_rows": len(rollout),
            "valid": len(events) == expected and len(top50) == 100,
        }
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
    finally:
        release_loaded(loaded)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3")
