#!/usr/bin/env python3
"""Phase 1132 revision 6: overprovisioned temporal relation review pool."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import phase1132_postrelease_temporal_material as base


UTC = timezone.utc
base.REVISION = "revision6_temporal_relation_binding_overprovisioned"
base.PRIMITIVE_FAMILY_OVERRIDE = "temporal_entity_relation_binding"
base.ALLOCATE_ALL_QUALIFIED = True
base.RAW_SNAPSHOT_PROVENANCE = (
    "immutable_revision5_temporal_relation_binding_pool_raw_files"
)
base.FACT_START = datetime(2025, 8, 1, tzinfo=UTC)
base.FACT_END = datetime(2026, 7, 31, 23, 59, 59, tzinfo=UTC)
base.FACT_WINDOW_RATIONALE = (
    "Uses the first full day after the latest recorded Qwen3 repository metadata "
    "commit and freezes every machine-qualified item, rather than exactly the "
    "minimum 128 per split, so blind human review can reject items without "
    "automatically destroying split volume. Effective dates do not prove secrecy."
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
    / "revision6_temporal_relation_binding_overprovisioned"
)
# Revision 6 changes allocation only, so it reuses the immutable Revision 5 raw
# snapshots rather than silently introducing a later Wikidata snapshot.
base.RAW_ROOT = (
    base.REPO_ROOT
    / "tests/glm5/result/phase1132_postrelease_temporal_material"
    / "revision5_temporal_relation_binding_pool/raw"
)
base.MATERIAL_ROOT = base.RESULT_ROOT / "material"
base.PACKAGE_PATH = base.MATERIAL_ROOT / "candidate_package_unreviewed.jsonl"


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    base.main()
