#!/usr/bin/env python3
"""Independent audit for C527-C540 sample-specific dynamics campaign."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2075_c541_sample_specific_dynamics_campaign_independent_audit"
sys.path.insert(0, str(TESTS))

import phase2061_c527_c540_sample_specific_dynamics_campaign as campaign


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def check(items: dict[str, bool], name: str, value: bool) -> None:
    items[name] = bool(value)


def main() -> None:
    if OUT.exists() and not (OUT / "analysis/final.json").exists():
        raise RuntimeError(f"partial audit output exists: {OUT}")
    if (OUT / "analysis/final.json").exists():
        print((OUT / "analysis/final.json").read_text(encoding="utf-8"))
        return
    for sub in ("analysis", "audit", "protocol"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    finals = {name: campaign.final(name) for name in campaign.FUNCTIONS}
    for name, value in finals.items():
        check(checks, f"{name}_closed", value["status"] == "closed")
        check(checks, f"{name}_internal_checks", value["all_checks_passed"])
        protocol = load(campaign.OUTS[name] / "protocol/preregistration.json")
        check(checks, f"{name}_producer_hash", protocol["producer_sha256"] == sha(Path(campaign.__file__)))

    rows = campaign.read_rows(campaign.OUTS["C528"] / "material/combined_cases.jsonl")
    compiled = campaign.read_rows(campaign.OUTS["C528"] / "compiled/qwen3_combined.jsonl")
    check(checks, "material_rows", len(rows) == 8160)
    check(checks, "material_unique", len({row["case_id"] for row in rows}) == 8160)
    check(checks, "material_domains", len({row["domain_id"] for row in rows}) == 20)
    check(checks, "compiled_rows", len(compiled) == 8160)
    check(checks, "compiled_roles", all(set(row["role_positions"]) == set(campaign.ROLES) for row in compiled))
    check(checks, "width_129_preserved", finals["C529"]["headline"]["max_prompt_tokens"] == 129)
    check(checks, "c519_missingness_preserved", finals["C529"]["headline"]["formal_missingness"]["fresh_program_width_contract"])
    check(checks, "human_naturalness_na", finals["C529"]["headline"]["formal_missingness"]["human_naturalness"] == "NA_not_run")

    capture = finals["C530"]["headline"]
    check(checks, "capture_rows", capture["rows"] == 8160)
    check(checks, "capture_mean_shape", capture["mean_shape"] == [8160, 38, 6, 2560])
    check(checks, "capture_exact_shape", capture["exact_shape"] == [8160, 38, 6, 2560])
    check(checks, "capture_full_shape", capture["full_shape"] == [120, 38, 129, 2560])
    check(checks, "capture_finite", finite(capture))

    c531 = finals["C531"]["headline"]
    recomputed_c531 = {}
    controls = ("persistence", "global_mean", "family_mean", "family_surface_mean", "sample_shuffle", "role_reverse", "coordinate_roll")
    for key, metrics in c531["surface_metrics"].items():
        best = min(metrics[name]["nrmse"] for name in controls)
        recomputed_c531[key] = metrics["shared"]["nrmse"] <= best - campaign.CONTROL_MARGIN
    check(checks, "c531_strata", len(recomputed_c531) == 9)
    check(checks, "c531_gates_recomputed", recomputed_c531 == c531["surface_gates"])
    check(checks, "c531_candidate_recomputed", all(recomputed_c531.values()) == c531["strong_candidate"])

    c532 = finals["C532"]["headline"]
    recomputed_c532 = {}
    for key in c532["stratum_gates"]:
        centered_gain = c532["centered_metrics"][key]["zero"]["nrmse"] - c532["centered_metrics"][key]["shared"]["nrmse"]
        pair_gain = c532["pair_metrics"][key]["zero"]["nrmse"] - c532["pair_metrics"][key]["shared"]["nrmse"]
        recomputed_c532[key] = centered_gain >= campaign.CONTROL_MARGIN and pair_gain >= campaign.CONTROL_MARGIN
    check(checks, "c532_gates_recomputed", recomputed_c532 == c532["stratum_gates"])
    check(checks, "c532_candidate_recomputed", all(recomputed_c532.values()) == c532["sample_specific_candidate"])

    c533 = finals["C533"]["headline"]
    pass_rate = sum(value >= campaign.CONTROL_MARGIN for value in c533["domain_margins"].values()) / len(c533["domain_margins"])
    candidate533 = pass_rate >= 0.80 and all(value > 0 for value in c533["cohort_median_margin"].values())
    check(checks, "c533_domains", len(c533["domain_metrics"]) == 20)
    check(checks, "c533_pass_rate", abs(pass_rate - c533["pass_rate_at_0_02"]) < 1e-12)
    check(checks, "c533_candidate_recomputed", candidate533 == c533["response_neighborhood_candidate"])

    c534 = finals["C534"]["headline"]
    check(checks, "c534_domains", len(c534["domain_metrics"]) == 6)
    check(checks, "c534_strata", len(c534["stratum_gates"]) == 18)
    check(checks, "c534_candidate_recomputed", all(c534["stratum_gates"].values()) == c534["cross_family_candidate"])

    c535 = finals["C535"]["headline"]
    check(checks, "c535_strata", len(c535["surface_gates"]) == 9)
    check(checks, "c535_exact_sample_strata", len(c535["exact_sample_gates"]) == 9)
    check(checks, "c535_exact_sample_recomputed", all(c535["exact_sample_gates"].values()) == c535["exact_sample_specific_candidate"])
    check(checks, "c535_writable_recomputed", (c535["strong_candidate"] and c535["exact_sample_specific_candidate"]) == c535["exact_writable_state_candidate"])

    c536 = finals["C536"]["headline"]
    check(checks, "c536_domains", len(c536["domain_pass_rate"]) == 20)
    check(checks, "c536_many_comparisons", len(c536["metrics"]) >= 80)
    check(checks, "c536_finite", finite(c536["metrics"]))

    requirements = finals["C537"]["headline"]["requirements"]
    check(checks, "causal_authorization_recomputed", all(requirements.values()) == finals["C537"]["headline"]["causal_authorized"])
    check(checks, "causal_result_is_na", str(finals["C538"]["headline"]["result"]).startswith("NA_"))
    check(checks, "causal_not_misreported_failed", not finals["C538"]["headline"]["ran"])

    visual = load(campaign.VISUAL)
    check(checks, "visual_schema", visual["schema"] == "ai2050.sample_specific_dynamics_atlas.v1")
    check(checks, "visual_rows", len(visual["rows"]) == 120)
    check(checks, "visual_coordinates", all(len(row["state_exact_q37"]) == 2560 for row in visual["rows"]))
    check(checks, "visual_full_token", all(len(token) == 2560 for field in visual["full_token_panel"]["checkpoint_fields"].values() for token in field))

    cleanup = load(campaign.OUTS["C540"] / "audit/raw_field_cleanup_ledger.json")
    check(checks, "cleanup_three_files", len(cleanup["files"]) == 3)
    check(checks, "cleanup_positive_bytes", cleanup["total_bytes"] > 0)
    check(checks, "cleanup_absent", all(not (ROOT / row["path"]).exists() for row in cleanup["files"]))
    check(checks, "visual_retained", campaign.VISUAL.exists())

    passed = all(checks.values())
    protocol = {
        "phase": 2075, "campaign": "C541", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audited_campaign": "C527-C540", "auditor_sha256": sha(Path(__file__)),
        "main_producer_sha256": sha(Path(campaign.__file__)),
    }
    save(OUT / "protocol/preregistration.json", protocol)
    save(OUT / "audit/independent_audit.json", {
        "status": "passed" if passed else "failed", "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": passed,
    })
    gates = finals["C539"]["headline"]["gates"]
    final = {
        "phase": 2075, "campaign": "C541", "status": "closed", "all_checks_passed": passed,
        "headline": {
            "status": "independent_audit_closed", "audit_status": "passed" if passed else "failed",
            "checks_passed": sum(checks.values()), "checks_total": len(checks), "gates": gates,
            "next_stage_same_goal": finals["C540"]["headline"]["next_stage_same_goal"],
            "next_route": finals["C540"]["headline"]["next_route"],
            "strict_conclusion": "The audit verifies preregistered arithmetic, provenance, evidence labels, visual preservation, and cleanup; it is not an external scientific replication.",
        },
    }
    save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
