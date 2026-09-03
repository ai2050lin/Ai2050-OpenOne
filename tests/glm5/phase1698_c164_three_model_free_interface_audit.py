#!/usr/bin/env python3
"""Independent audit for C164."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1698_c164_three_model_free_interface"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("plain_target", "natural_sentence")


def load(path): return json.loads((OUT / path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(x) for x in (OUT / path).read_text(encoding="utf-8").splitlines()]


def main():
    protocol, summary, final = load("protocol/preregistration.json"), load("analysis/summary.json"), load("analysis/final.json")
    raw = {m: rows(f"raw/{m}.jsonl") for m in MODELS}
    recomputed = {
        interface: [m for m in MODELS if summary["models"][m]["interfaces"][interface]["qualified"]]
        for interface in INTERFACES
    }
    checks = {
        "contract": load("audit/internal_contract_audit.json")["all_checks_passed"],
        "three_models": all(load(f"audit/internal_{m}_audit.json")["all_checks_passed"] for m in MODELS),
        "analysis": load("audit/internal_analysis_audit.json")["all_checks_passed"],
        "final": final["all_checks_passed"],
        "rows": all(len(raw[m]) == 192 for m in MODELS),
        "interfaces": all(sum(r["interface"] == i for r in raw[m]) == 96 for m in MODELS for i in INTERFACES),
        "formal": protocol["formal_partition"] == ["confirmation", "fresh"],
        "recomputed_common": recomputed == summary["common_interface_models"],
        "eligibility": summary["cross_model_eligibility"] == any(len(v) >= 2 for v in recomputed.values()),
        "sequential": protocol["sequential_loading"] is True,
        "scope": "no coordinate" in summary["claim_boundary"],
    }
    audit = {"phase": 1698, "campaign": "C164", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_eligibility": summary["cross_model_eligibility"], "authorization": summary["next_authorization"]}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
