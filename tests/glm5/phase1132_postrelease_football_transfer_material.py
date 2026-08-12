#!/usr/bin/env python3
"""Phase 1132 revision 3: homogeneous post-release football-transfer family."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import phase1132_postrelease_temporal_material as base


UTC = timezone.utc
base.REVISION = "revision3_football_transfer_family"
base.FACT_START = datetime(2025, 8, 1, tzinfo=UTC)
base.FACT_END = datetime(2026, 7, 31, 23, 59, 59, tzinfo=UTC)
base.FACT_WINDOW_RATIONALE = (
    "Uses the first full day after the latest recorded Qwen3 repository metadata "
    "commit. The homogeneous P54 family is expanded instead of relaxing any "
    "surface or uniqueness filter. Effective dates still do not prove that a "
    "transfer was unannounced before the boundary."
)
base.PROPERTIES = {
    "P54": {
        "label": "football club",
        "domain": "sports",
        "object_type_qid": "Q476028",
        "template_kind": "membership",
        "member_noun": "football club",
    }
}
base.RESULT_ROOT = (
    base.REPO_ROOT
    / "tests/glm5/result/phase1132_postrelease_temporal_material"
    / "revision3_football_transfer_family"
)
base.RAW_ROOT = base.RESULT_ROOT / "raw"
base.MATERIAL_ROOT = base.RESULT_ROOT / "material"
base.PACKAGE_PATH = base.MATERIAL_ROOT / "candidate_package_unreviewed.jsonl"


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    base.main()
