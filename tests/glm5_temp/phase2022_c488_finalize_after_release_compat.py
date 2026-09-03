#!/usr/bin/env python3
"""Finalize C488 from complete raw behavior after release helper mismatch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2019_c485_c500_complete_state_information_campaign as p


out = p.OUTS["C488"]
rows = p.read_rows(out / "material/corrected_cases.jsonl")
by_id = {row["case_id"]: row for row in rows}
compiled = p.read_rows(out / "compiled/qwen3.jsonl")
behavior = p.read_rows(out / "raw/behavior.jsonl")
repairs = p.load(out / "audit/premodel_role_repair.json")["repairs"]
role_reaudit = p.load(out / "audit/premodel_role_repair.json")["role_reaudit"]
by_family = {
    family: float(np.mean([row["correct"] for row in behavior if by_id[row["case_id"]]["family"] == family]))
    for family in p.FAMILIES
}
by_partition = {
    part: float(np.mean([row["correct"] for row in behavior if by_id[row["case_id"]]["partition"] == part]))
    for part in ("discovery", "confirmation", "lockbox")
}
eligible = [family for family, accuracy in by_family.items() if accuracy >= 0.60]
accuracy = float(np.mean([row["correct"] for row in behavior]))
authorized = accuracy >= 0.75 and len(eligible) >= 9
errors = [
    {**row, "family": by_id[row["case_id"]]["family"], "construction": by_id[row["case_id"]]["construction"],
     "unit": by_id[row["case_id"]]["unit"], "bits": by_id[row["case_id"]]["bits"]}
    for row in behavior if not row["correct"]
]
p.write_rows(out / "analysis/typed_behavior_errors.jsonl", errors)
headline = {
    "status": "behavior_closed_after_release_helper_compatibility_recovery",
    "rows": len(behavior), "accuracy": accuracy,
    "placement": "cuda:0 BF16; original process completed all inference before cleanup exception",
    "quantization": {"quantized_module_count": 0, "has_quantized_modules": False},
    "premodel_role_repairs": len(repairs), "role_reaudit": role_reaudit,
    "family_accuracy": by_family, "partition_accuracy": by_partition,
    "eligible_families": eligible, "field_authorized": authorized,
    "typed_errors": len(errors), "max_prompt_tokens": max(len(row["prompt_ids"]) for row in compiled),
    "execution_recovery": "No model rerun. Finalized from all 5280 raw rows written before release_bf16 name mismatch.",
    "strict_interpretation": "Behavior qualifies this fixed Yes/No interface only; the premodel repair changed role metadata, not model-facing material.",
}
p.close("C488", headline, {
    "rows": len(behavior) == len(rows), "finite": p.finite(headline),
    "role_reaudit": all(role_reaudit.values()), "raw_complete_before_exception": len(behavior) == 5280,
}, "C489_state_cube")
