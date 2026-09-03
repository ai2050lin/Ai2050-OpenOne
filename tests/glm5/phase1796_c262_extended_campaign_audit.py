#!/usr/bin/env python3
"""Independent audit of C261-C262 and their reclassification boundaries."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    c261 = load(RESULT / "phase1795_c261_coordinate_coverage_generation_side_effects/analysis/final.json")
    c262 = load(RESULT / "phase1796_c262_full_word_generation_correction/analysis/final.json")
    asset = load(ROOT / "frontend/public/vis_data/research_kernel/c262_generation_specificity_atlas.json")
    h1, h2 = c261["headline"], c262["headline"]
    checks = {
        "source_finals": c261["all_checks_passed"] and c262["all_checks_passed"],
        "coverage_is_distributed": h1["earliest_registered_fraction_at_flip_0_8"] == 0.75,
        "midpoint_erasure_candidate": h1["midpoint_erasure_gate_passed"],
        "one_token_metric_reclassified": "invalid full-word metric" in h2["c261_one_token_reclassification"],
        "full_word_correct_path_works": next(row for row in h2["summaries"] if row["condition"] == "correct")["success_rate"] == 1,
        "reversed_control_collision": h2["correct_minus_best_control"] == 0 and not h2["full_word_generation_gate_passed"],
        "side_effect_claim_retracted": "not an unrelated-capability test" in h2["c261_side_effect_reclassification"],
        "asset_full_coordinates": asset["dimensions"] == 2560 and all(len(row["values"]) == 2560 for row in asset["rows"]),
    }
    report = {
        "phase": 1796,
        "campaign": "C262-independent-audit",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "The registered field is strongly distributed and can manipulate a generated attitude word, but the reversed-mask collision defeats path specificity. No selective natural-language causal mechanism is closed.",
        "next_authorization": "replace_absolute_checkpoint_masks_with_state_conditioned_operator_before_more_output_causality",
    }
    out = RESULT / "phase1796_c262_full_word_generation_correction/audit/extended_campaign_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
