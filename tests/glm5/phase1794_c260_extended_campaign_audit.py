#!/usr/bin/env python3
"""Independent disk audit for the C246-C260 event-to-output campaign."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    first = load(RESULT / "phase1789_c255_campaign_theory_adjudication/analysis/final.json")
    finals = {}
    for phase, name in (
        (1790, "c256_output_sensitive_attitude_causal_test"),
        (1791, "c257_fixed_codebook_output_causal_test"),
        (1792, "c258_output_readout_path_localization"),
        (1793, "c259_independent_dense_path_replication"),
        (1794, "c260_path_ladder_natural_word_readout"),
    ):
        finals[phase] = load(RESULT / f"phase{phase}_{name}/analysis/final.json")
    h = {phase: value["headline"] for phase, value in finals.items()}
    asset = load(ROOT / "frontend/public/vis_data/research_kernel/c260_output_path_causal_atlas.json")
    checks = {
        "c246_c255_audit_passed": first["all_checks_passed"],
        "c256_c260_finals_passed": all(value["all_checks_passed"] for value in finals.values()),
        "phases_contiguous": sorted(finals) == list(range(1790, 1795)),
        "c256_reclassified_in_c257": h[1791]["c256_reclassified"] == "output causality not tested due outcome-relative candidate labels",
        "fixed_codebook_sparse_gate_failed": not h[1791]["output_causal_gate_passed"],
        "dense_path_gate_passed": h[1792]["output_path_gate_passed"],
        "independent_dense_path_passed": h[1793]["independent_dense_path_gate_passed"],
        "ladder_and_word_gates_passed": h[1794]["path_ladder_gate_passed"] and h[1794]["natural_word_gate_passed"],
        "prefix_ordering_visible": h[1794]["earliest_passing_prefix_end"] == 4,
        "all_coordinate_asset": asset["dimensions"] == 2560 and len(asset["rows"]) == 34 and all(len(row["values"]) == 2560 for row in asset["rows"]),
        "no_attention_or_mlp_claim": "Attention/MLP" in asset["claim_boundary"],
    }
    report = {
        "phase": 1794,
        "campaign": "C260-independent-audit",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "The registered distributed attitude-event masks are sufficient to steer a fixed-codebook "
            "and direct-word controlled output when written over an early checkpoint path, and this "
            "replicates on new words and surfaces. The audit does not infer coordinate minimality, "
            "natural necessity, free-generation closure, or a unique circuit."
        ),
    }
    out = RESULT / "phase1794_c260_path_ladder_natural_word_readout/audit/extended_campaign_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
