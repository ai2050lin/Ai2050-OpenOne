#!/usr/bin/env python3
"""C205: audit C194-C204 and freeze the full-sequence response-ecology campaign."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C205
PHASE, CAMPAIGN = 1739, "C205"


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C204 / "audit/independent_final_audit.json")
    anchors = common.selected_anchors()
    behavior = common.behavior_by_case()
    epsilons = common.epsilon_by_case()
    coordinates = common.source_coordinates()
    checks = {
        "authorization": parent["all_checks_passed"],
        "anchors": len(anchors) == 36,
        "programs": len({row["program"] for row in anchors}) == 9,
        "units": {row["unit"] for row in anchors} == set(common.UNITS),
        "one_surface": {row["surface"] for row in anchors} == {0},
        "one_candidate_order": {row["order"] for row in anchors} == {1},
        "all_behavior_correct": all(behavior[row["case_id"]]["correct"] for row in anchors),
        "all_epsilons": all(row["case_id"] in epsilons for row in anchors),
        "source_coordinates": len(coordinates) == 32 and len(set(coordinates)) == 32,
        "fixed_width": max(len(row["prompt_ids"]) for row in anchors) <= common.WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "compiled/qwen3_anchors.jsonl", anchors)
    core.save(OUT / "protocol/source_coordinates.json", {"coordinates": coordinates})
    core.save(OUT / "protocol/anchor_epsilons.json", {row["case_id"]: epsilons[row["case_id"]] for row in anchors})
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_sequence_response_campaign_frozen",
        "evidence_audit": {
            "retained": [
                "C195 observed a reproducible signed q23-to-q24/q25 response on 64 registered source coordinates",
                "C196 rejected its frozen 16-direction one-coordinate-derivative superposition gates",
                "C197 rejected only the tested identity and diagonal/gain families",
                "C198 established strong behavior on nine controlled-English programs and observed local trajectories",
                "C199 did not test true atomic composition; C200 remained typed-not-tested",
                "C202 correctly downgraded the three-model topology claim after a nontrivial role-permutation diagnostic",
                "C203 showed that 0.5 and 1.0 dose writes were materially represented in BF16",
                "C204 rejected a two-point per-coordinate odd-cubic extrapolator",
            ],
            "corrected_overclaims": [
                "signed persistence can be generic residual propagation and is not by itself semantic",
                "six registered roles are not known to be a closed state because every block receives the full token sequence",
                "C196 confounded per-coordinate dose with active-coordinate count and sampled only 16 of 64 orthogonal directions",
                "odd and even response components must be analyzed separately",
                "C197 did not test cross-coordinate, cross-role, full-token or state-gated maps",
                "C198 surfaces are controlled English and have no independent human naturalness audit",
                "C201 used only 18 common rows and its reanalysis was post hoc",
                "C203 does not turn the 0.25 dose into a precision derivative",
                "C204's two-point cubic extrapolation is ill-conditioned and eliminates only that narrow formula",
            ],
        },
        "object": "full-token signed finite response A(q,X,kappa,delta), with role projections treated as views rather than a presumed closed state",
        "model_order": ["qwen3", "glm4", "deepseek7b"],
        "qwen_anchor_panel": "nine programs x units 1,2,5,6 x surface 0 = 36 behavior-correct anchors",
        "partitions": {"discovery": [1, 2], "confirmation": [5], "fresh": [6]},
        "checkpoints": ["embedding", "q23", "q24", "q25"],
        "doses": list(common.DOSES),
        "source": "q23 relation role x 32 frozen physical activation coordinates",
        "response_split": {
            "odd": "(F(+delta)-F(-delta))/2",
            "even": "(F(+delta)+F(-delta))/2-F(0)",
        },
        "routes": {
            "C206": "full-sequence dose response and numerical repeat floor",
            "C207": "same-per-coordinate-dose single-versus-joint coupling separation",
            "C208": "complete 32-direction orthogonal calibration with unseen random-direction prediction",
            "C209": "role-only versus progressively added non-role token closure",
            "C210": "natural paraphrase edit trajectory on nine language programs",
            "C211": "five flagship route eligibility ledger",
            "C212": "voice x clause-order factorial and direct-versus-two-hop composition",
            "C213": "deletion/rescue only for a prospectively qualified predictive object",
            "C214": "model-specific interfaces and functional isomorphism; never same coordinate ids",
            "C215": "evidence synthesis, theory gate and parameter-level heatmap",
        },
        "route_policy": "failure eliminates only that route; other observations continue",
        "semantic_policy": "behavior failure forbids semantic naming but does not erase numerical HiddenState observation",
        "forbidden": ["attention", "MLP", "weights", "PCA", "post-reveal gate changes", "claiming a unique circuit"],
        "claim_boundary": "Qwen3 controlled language micro-programs with conditional GLM4/DeepSeek functional comparison; no complete language mechanism is presupposed",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C206_through_C215_sequentially_with_route_level_elimination",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_tokens": max(len(row["prompt_ids"]) for row in anchors), "authorization": protocol["authorization"]}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
        "campaign_complete": set(protocol["routes"]) == {f"C{value}" for value in range(206, 216)},
    }
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": protocol["evidence_audit"], "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "close"))
    args = parser.parse_args()
    {"contract": contract, "close": close}[args.command]()


if __name__ == "__main__":
    main()
