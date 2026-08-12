#!/usr/bin/env python3
"""Phase 1132 revision 2: expand time supply without relaxing any item filter."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import phase1132_postrelease_temporal_material as base


UTC = timezone.utc
base.REVISION = "revision2_post_metadata_expanded_window"
base.FACT_START = datetime(2025, 8, 1, tzinfo=UTC)
base.FACT_END = datetime(2026, 7, 31, 23, 59, 59, tzinfo=UTC)
base.FACT_WINDOW_RATIONALE = (
    "Expands source supply to the first full day after the latest recorded Qwen3 "
    "repository metadata commit (2025-07-26). All structural and tokenizer "
    "filters remain identical to revision 1; effective dates still do not prove "
    "that an appointment was unannounced before the boundary."
)
base.RESULT_ROOT = (
    base.REPO_ROOT
    / "tests/glm5/result/phase1132_postrelease_temporal_material"
    / "revision2_expanded_window"
)
base.RAW_ROOT = base.RESULT_ROOT / "raw"
base.MATERIAL_ROOT = base.RESULT_ROOT / "material"
base.PACKAGE_PATH = base.MATERIAL_ROOT / "candidate_package_unreviewed.jsonl"


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    base.main()
