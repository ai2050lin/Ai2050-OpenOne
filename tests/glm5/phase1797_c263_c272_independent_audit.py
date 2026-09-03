#!/usr/bin/env python3
"""Independent artifact audit for any completed C263-C272 stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import phase1797_c263_c272_state_operator_common as common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", choices=tuple(common.OUTS))
    args = parser.parse_args()
    out = common.OUTS[args.campaign]
    protocol = common.core.load(out / "protocol/preregistration.json")
    final = common.core.load(out / "analysis/final.json")
    checks = {
        "phase_matches": final["phase"] == protocol["phase"],
        "campaign_matches": final["campaign"] == args.campaign,
        "closed": final["status"] == "closed",
        "internal_passed": final["all_checks_passed"],
        "protocol_precedes_result": (out / "protocol/preregistration.json").stat().st_mtime <= (out / "analysis/final.json").stat().st_mtime,
        "no_markdown_in_result": not any(out.rglob("*.md")),
    }
    report = {"campaign": args.campaign, "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final.get("next_authorization")}
    common.core.save(out / "audit/independent_campaign_audit.json", report)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
