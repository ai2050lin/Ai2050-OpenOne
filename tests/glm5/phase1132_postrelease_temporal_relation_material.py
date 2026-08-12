#!/usr/bin/env python3
"""Phase 1132 revision 5: pooled temporal entity-relation binding operator."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import phase1132_postrelease_temporal_material as base


UTC = timezone.utc
base.REVISION = "revision5_temporal_relation_binding_pool"
base.PRIMITIVE_FAMILY_OVERRIDE = "temporal_entity_relation_binding"
base.FACT_START = datetime(2025, 8, 1, tzinfo=UTC)
base.FACT_END = datetime(2026, 7, 31, 23, 59, 59, tzinfo=UTC)
base.FACT_WINDOW_RATIONALE = (
    "Uses the first full day after the latest recorded Qwen3 repository metadata "
    "commit. It pools two pre-audited subfamilies of the same operator: selecting "
    "the valid value of an entity-relation pair at a specified date. No structural "
    "or tokenizer threshold is relaxed. Effective dates do not prove secrecy."
)
base.PROPERTIES = {
    "P6": {"label": "head of government", "domain": "government"},
    "P35": {"label": "head of state", "domain": "government"},
    "P169": {"label": "chief executive officer", "domain": "corporate"},
    "P286": {"label": "head coach", "domain": "sports"},
    "P488": {"label": "chairperson", "domain": "organization"},
    "P54": {
        "label": "sports team",
        "domain": "sports",
        "template_kind": "membership",
        "member_noun": "sports team",
    },
}
base.RESULT_ROOT = (
    base.REPO_ROOT
    / "tests/glm5/result/phase1132_postrelease_temporal_material"
    / "revision5_temporal_relation_binding_pool"
)
base.RAW_ROOT = base.RESULT_ROOT / "raw"
base.MATERIAL_ROOT = base.RESULT_ROOT / "material"
base.PACKAGE_PATH = base.MATERIAL_ROOT / "candidate_package_unreviewed.jsonl"


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    base.main()
