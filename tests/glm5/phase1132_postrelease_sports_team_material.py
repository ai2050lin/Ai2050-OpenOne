#!/usr/bin/env python3
"""Phase 1132 revision 4: homogeneous P54 sports-team membership family."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import phase1132_postrelease_temporal_material as base


UTC = timezone.utc
base.REVISION = "revision4_sports_team_membership_family"
base.FACT_START = datetime(2025, 8, 1, tzinfo=UTC)
base.FACT_END = datetime(2026, 7, 31, 23, 59, 59, tzinfo=UTC)
base.FACT_WINDOW_RATIONALE = (
    "Uses the first full day after the latest recorded Qwen3 repository metadata "
    "commit. P54 itself denotes sports-team membership, so this revision retains "
    "the full homogeneous relation instead of selecting one sport or relaxing "
    "surface filters. Effective dates still do not prove pre-boundary secrecy."
)
base.PROPERTIES = {
    "P54": {
        "label": "sports team",
        "domain": "sports",
        "template_kind": "membership",
        "member_noun": "sports team",
    }
}
base.RESULT_ROOT = (
    base.REPO_ROOT
    / "tests/glm5/result/phase1132_postrelease_temporal_material"
    / "revision4_sports_team_membership_family"
)
base.RAW_ROOT = base.RESULT_ROOT / "raw"
base.MATERIAL_ROOT = base.RESULT_ROOT / "material"
base.PACKAGE_PATH = base.MATERIAL_ROOT / "candidate_package_unreviewed.jsonl"


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    base.main()
