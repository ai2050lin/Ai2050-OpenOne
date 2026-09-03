#!/usr/bin/env python3
"""Attach the independent C484 adjudication and sparse-stratum labels to the atlas."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c484_program_guard_hypergraph.json"
AUDIT = ROOT / "tests/glm5/result/phase2018_c484_campaign_synthesis_visual_cleanup_and_audit/audit/independent_audit.json"

payload = json.loads(VISUAL.read_text(encoding="utf-8"))
audit = json.loads(AUDIT.read_text(encoding="utf-8"))
payload["strict_adjudication"] = audit["adjudication"]
payload["claim_boundary"] = (
    "The registered ridge beats its listed fallback, shuffle, and roll controls, but identity/shared controls were omitted. "
    "It is not a broad coupling law, a semantic-coordinate dictionary, or a causal circuit."
)
for row in payload["rows"]:
    if row.get("family") == "temporal_order":
        row["source_scope"] = "fallback_all_complete_descriptive_only"
        row["source_programs"] = 1
    elif row.get("source") == "program_effect_centroid":
        row["source_scope"] = "discovery_ledger_brief"

VISUAL.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(json.dumps({
    "rows": len(payload["rows"]),
    "strict_gates": payload["strict_adjudication"]["strict_gates"],
    "temporal_rows": sum(row.get("family") == "temporal_order" for row in payload["rows"]),
}, ensure_ascii=False))
