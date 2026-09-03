#!/usr/bin/env python3
"""C247: freeze a third material system with crossed surfaces and lexical units."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.OUTS["C247"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C246"] / "audit/independent_final_audit.json")
    rows = common.material()
    compiled = common.compile_rows(common.graph_base.tokenizer(), rows)
    old_words = {u["primary"] for u in common.old.UNITS}
    old_words |= {u["primary"] for u in __import__("phase1778_c244_independent_event_replication").UNITS}
    core_rows = [row for row in rows if row["panel"] == "core"]
    nested = [row for row in rows if row["panel"] == "nested_composition"]
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "C247_material_contract",
        "rows": len(rows) == 768,
        "core_rows": len(core_rows) == 640,
        "nested_rows": len(nested) == 128,
        "candidate_balance": sum(row["gold_position"] == 0 for row in rows) == 384,
        "surface_crossing": all({row["surface"] for row in core_rows if row["unit"] == unit and row["family"] == family} == set(common.SURFACES) for unit in range(8) for family in common.FAMILIES),
        "new_primary_lexicon": not ({u["primary"] for u in common.UNITS} & old_words),
        "unique_prompts": len({row["prompt"] for row in rows}) == len(rows),
        "semantic_roles": all(set(row["role_positions"]) == set(common.ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) <= common.WIDTH,
        "human_blind_missing_registered": True,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled)})
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": 1781, "campaign": "C247", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "third_material_frozen", "rows": len(rows), "core_rows": len(core_rows), "nested_rows": len(nested),
        "surfaces": list(common.SURFACES), "families": list(common.FAMILIES) + ["nested_attitude"],
        "crossing": "every core lexical unit appears under both surfaces, both candidate orders, and all four factorial cells",
        "behavior_gate": {"global_min": 0.85, "each_core_family_min": 0.70, "nested_min": 0.70},
        "semantic_uniqueness": "answers are compiled from explicit facts; correct and distractor differ; candidate position is exactly balanced",
        "naturalness": "controlled English with internal grammar checks; independent human blind review remains missing",
        "field_shape": [len(rows), 37, common.WIDTH, common.DIM], "producer_sha256": core.sha(Path(__file__)),
        "authorization": "C248_Qwen_full_field_capture_once",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    core.save(OUT / "analysis/final.json", {"phase": 1781, "campaign": "C247", "status": "closed", "all_checks_passed": True, "headline": protocol, "next_authorization": protocol["authorization"]})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
