#!/usr/bin/env python3
"""Phase459 GLM4 independent generator v2 behavior replicate."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase456_glm4_independent_core_behavior import build_summary, eval_rows  # noqa: E402
from phase451_glm4_v2_pilot_behavior import load_jsonl, run_generation, write_jsonl  # noqa: E402


SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase458_independent_v2_protocol" / "phase458_independent_v2_samples.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase459_glm4_independent_v2_behavior"
GENERATIONS_PATH = OUT_DIR / "phase459_glm4_independent_v2_generations.jsonl"
SUMMARY_PATH = OUT_DIR / "phase459_glm4_independent_v2_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = eval_rows(load_jsonl(SAMPLES_PATH))
    model = None
    try:
        model, tokenizer, device = load_model("glm4", use_8bit=True if args.use_8bit else None)
        records = run_generation(model, tokenizer, device, rows, args.batch_size, args.max_new_tokens)
        write_jsonl(GENERATIONS_PATH, records)
        summary = build_summary(records)
        summary["schema_version"] = "phase459_glm4_independent_v2_behavior.v1"
        summary["target"] = "knowledge_network/independent_v2_marker_truth"
        summary["status"] = "independent_v2_behavior_complete_no_physical_trace"
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(SUMMARY_PATH)
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
