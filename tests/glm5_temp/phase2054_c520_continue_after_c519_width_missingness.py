#!/usr/bin/env python3
"""Continue C520 descriptively after C519's one-token width-gate miss.

C519 remains formally failed. This script does not change its threshold or
verdict; it applies the preregistered route policy that typed interface
missingness does not stop full-coordinate observation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2052_c518_c525_fresh_callable_state_campaign as campaign


def main() -> None:
    if (campaign.OUTS["C520"] / "analysis/final.json").exists():
        print((campaign.OUTS["C520"] / "analysis/final.json").read_text(encoding="utf-8"))
        return
    c519 = campaign.final("C519")
    raw = campaign.read_rows(campaign.OUTS["C519"] / "raw/behavior.jsonl")
    out = campaign.begin("C520", {
        "status": "old_fresh_three_family_capture_with_typed_C519_width_missingness",
        "model": "local Qwen3 BF16 CUDA",
        "state": "q0 embedding, all 36 block outputs, final norm; six roles; all 2560 coordinates",
        "full_token_subset": "balanced lockbox complete cells for old and fresh panels",
        "upstream_missingness": {
            "C519_formal_pass": False,
            "reason": "max prompt width 129 exceeded frozen 128 by one token",
            "execution_order_hardness": "behavior ran before width failure was closed",
            "threshold_changed": False,
            "behavior_rows_retained_descriptively": len(raw),
        },
        "claim_restriction": "downstream prediction is descriptive/prospective on a width-missingness panel, not inheritance of a fully passed C519 behavior contract",
    }, {
        "C519_formally_failed": not c519["all_checks_passed"],
        "C519_width_exact": c519["headline"]["max_prompt_tokens"] == 129,
        "behavior_complete": len(raw) == 1440,
        "field_policy": c519["headline"]["field_authorized"],
        "cuda": campaign.torch.cuda.is_available(),
    })
    rows, compiled = campaign.combined_rows_compiled()
    full_ids = {row["case_id"] for row in rows if row["unit"] == 8 and row["construction"] == "ledger"}
    width = max(len(row["prompt_ids"]) for row in compiled)
    headline = campaign.previous.parent.capture_state_cube(rows, compiled, out, full_ids, width, batch_size=8)
    campaign.close("C520", {
        "status": "capture_closed_with_upstream_width_missingness",
        **headline,
        "old_rows": 1440,
        "fresh_rows": 1440,
        "upstream_C519_formal_pass": False,
        "upstream_width": 129,
        "strict_interpretation": "Observation continues under route-level missingness; no downstream result repairs C519.",
    }, {
        "rows": headline["rows"] == 2880,
        "role_shape": headline["role_shape"] == [2880, 38, 6, 2560],
        "full_rows": headline["full_token_rows"] == 96,
        "finite": campaign.finite(headline),
        "missingness_preserved": True,
    }, "C521_panel_replication")


if __name__ == "__main__":
    main()
