#!/usr/bin/env python3
"""Run Phase560 unseen semantic source-color delete/restore/exclusion validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from phase559_causal_screen import run


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
PARENT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(
        args.batch_size,
        args.restart,
        contract_path=OUT_DIR / "phase560_semantic_color_unseen_frozen_contract.json",
        candidates_path=OUT_DIR / "phase560_semantic_color_qualified_candidates.json",
        path_rows=PARENT_DIR / "phase559_qwen3_path_behavior_rows.jsonl",
        rows_path=OUT_DIR / "phase560_semantic_color_unseen_rows.jsonl",
        summary_path=OUT_DIR / "phase560_semantic_color_unseen_execution_summary.json",
    )
