#!/usr/bin/env python3
"""Phase1364: freeze C056 single-write Hidden-State cascade paths."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from model_utils import MODEL_CONFIGS

PHASE, CAMPAIGN = 1364, "C056"
C053 = TESTS / "result/phase1353_c053_route_portfolio_contract"
BEHAVIOR = TESTS / "result/phase1354_c053_behavior_route_competition"
PARENT = TESTS / "result/phase1363_c055_hidden_state_coalition_causal"
OUT = TESTS / "result/phase1364_c056_hidden_path_contract"

PATHS = {
    "family_early": {
        "source": {"layer": 3, "role": "family"},
        "checkpoints": [{"layer": 15, "role": "query"}, {"layer": 27, "role": "boundary"}],
    },
    "family_mid": {
        "source": {"layer": 15, "role": "family"},
        "checkpoints": [{"layer": 27, "role": "query"}, {"layer": 35, "role": "boundary"}],
    },
    "family_late": {
        "source": {"layer": 27, "role": "family"},
        "checkpoints": [{"layer": 35, "role": "boundary"}],
    },
    "query_mid": {
        "source": {"layer": 15, "role": "query"},
        "checkpoints": [{"layer": 27, "role": "query"}, {"layer": 35, "role": "boundary"}],
    },
    "query_late": {
        "source": {"layer": 27, "role": "query"},
        "checkpoints": [{"layer": 35, "role": "boundary"}],
    },
}


def tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def query_position(tok, prompt_ids: list[int]) -> int:
    needles = [[int(x) for x in tok.encode(value, add_special_tokens=False)] for value in ("?", " ?")]
    return core.locate_last_subsequence(prompt_ids, needles)[-1]


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1364 already exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    behavior_summary = core.load(BEHAVIOR / "analysis/qwen3_summary.json")
    if parent.get("authorization") != "close_c055_at_hidden_state_coalition_boundary":
        raise RuntimeError("C055 is not closed")
    if not parent_audit.get("all_checks_passed") or not behavior_audit.get("all_checks_passed"):
        raise RuntimeError("required parent audit failed")
    if not {"B2_relative", "B3_choice"}.issubset(behavior_final.get("qualified_routes", [])):
        raise RuntimeError("relative behavior object is not qualified")

    source = core.rows(C053 / "material/b1_binary_cases.jsonl")
    active_compiled = core.rows(C053 / "compiled/qwen3_B1_binary.jsonl")
    status_source = core.rows(C053 / "material/status_null_cases.jsonl")
    status_compiled = core.rows(C053 / "compiled/qwen3_N_status.jsonl")
    behavior_rows = core.rows(BEHAVIOR / "raw/B1_binary_behavior.jsonl")
    compiled = {row["case_id"]: row for row in active_compiled}
    compiled.update({row["case_id"]: row for row in status_compiled})
    behavior = {row["case_id"]: row for row in behavior_rows}
    status_by_key = {(row["quartet_key"], row["cell"]): row for row in status_source}
    quartets: dict[str, list[dict]] = defaultdict(list)
    for row in source:
        quartets[row["quartet_key"]].append(row)

    eligible = []
    for quartet_key, rows in sorted(quartets.items()):
        cells = {row["cell"]: row for row in rows}
        if set(cells) != {"aa", "ab", "ba", "bb"}:
            continue
        status_key = quartet_key.rsplit(":", 1)[0] + ":status"
        for direction, clean_cell, corrupt_cell, wrong_cell in (
            ("a_to_b", "aa", "ab", "bb"),
            ("b_to_a", "bb", "ba", "aa"),
        ):
            clean, corrupt, wrong = (cells[clean_cell], cells[corrupt_cell], cells[wrong_cell])
            status = status_by_key[(status_key, wrong_cell)]
            ids = [clean["case_id"], corrupt["case_id"], wrong["case_id"]]
            if not all(behavior[case_id]["correct"] for case_id in ids):
                continue
            if not all(len(compiled[case_id]["tested_family_span"]) == 1 for case_id in ids + [status["case_id"]]):
                continue
            if clean["target"] != corrupt["target"] or clean["truth"] is not True or corrupt["truth"] is not False:
                raise RuntimeError("clean/corrupt semantics drift")
            eligible.append({
                "pair_id": f"{quartet_key}:{direction}",
                "quartet_key": quartet_key,
                "partition": clean["partition"],
                "surface": clean["surface"],
                "family_pair": clean["family_pair"],
                "direction": direction,
                "target": clean["target"],
                "clean_true": clean["case_id"],
                "corrupt_false": corrupt["case_id"],
                "wrong_identity_true": wrong["case_id"],
                "status_true": status["case_id"],
            })

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in eligible:
        cells[(row["partition"], row["surface"])].append(row)
    per_cell = min(len(rows) for rows in cells.values())
    if per_cell < 8:
        raise RuntimeError(f"insufficient balanced cell size: {per_cell}")
    per_cell = 8
    cases = []
    for key in sorted(cells):
        cases.extend(sorted(cells[key], key=lambda row: row["pair_id"])[:per_cell])

    tok = tokenizer()
    used_ids = sorted({row[key] for row in cases for key in
                       ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")})
    extended = []
    for case_id in used_ids:
        row = dict(compiled[case_id])
        row["query_position"] = query_position(tok, row["prompt_ids"])
        row["role_positions"] = {
            "target": list(row["target_span"]),
            "family": list(row["tested_family_span"]),
            "query": [row["query_position"]],
            "boundary": [row["boundary_position"]],
        }
        extended.append(row)

    case_counts = Counter((row["partition"], row["surface"]) for row in cases)
    checks = {
        "parent_closed": parent.get("authorization") == "close_c055_at_hidden_state_coalition_boundary",
        "parent_audited": parent_audit.get("all_checks_passed") is True,
        "relative_behavior": {"B2_relative", "B3_choice"}.issubset(behavior_final.get("qualified_routes", [])),
        "behavior_audited": behavior_audit.get("all_checks_passed") is True,
        "behavior_pairs_individually_correct": all(
            behavior[row[key]]["correct"] for row in cases
            for key in ("clean_true", "corrupt_false", "wrong_identity_true")
        ),
        "balanced_12_cells": len(case_counts) == 12 and set(case_counts.values()) == {8},
        "case_count": len(cases) == 96,
        "span_isomorphism": all(
            len(compiled[row[key]]["tested_family_span"]) == 1 for row in cases
            for key in ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")
        ),
        "clean_corrupt_same_target": all(
            next(x for x in source if x["case_id"] == row["clean_true"])["target"]
            == next(x for x in source if x["case_id"] == row["corrupt_false"])["target"]
            for row in cases
        ),
        "role_positions_valid": all(
            all(positions and max(positions) < len(row["prompt_ids"])
                for positions in row["role_positions"].values()) for row in extended
        ),
        "finite_paths": set(PATHS) == {"family_early", "family_mid", "family_late", "query_mid", "query_late"},
        "causal_order": all(
            path["source"]["layer"] < checkpoint["layer"]
            for path in PATHS.values() for checkpoint in path["checkpoints"]
        ),
        "semantic_uniqueness_inherited": True,
        "controlled_naturalness_inherited": True,
        "hidden_state_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    core.write_rows(OUT / "material/path_cases.jsonl", cases)
    core.write_rows(OUT / "compiled/extended_rows.jsonl", extended)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "semantic_scope": "inherits frozen C053 ordinary-noun membership judgments",
        "naturalness_scope": "inherits ordinary, dictionary, and claim surfaces",
        "independent_human_blind_review": False,
        "zero_models": {
            "self": "same-input source write must preserve every downstream state and output",
            "wrong_identity_true": "truth polarity without the recipient identity is sufficient",
            "status_true": "generic affirmative protocol state is sufficient",
            "no_propagation": "a source write changes output without following the registered downstream response",
            "late_overwrite": "simultaneous multi-layer overwrite is forbidden because the last write can dominate",
        },
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c056.single_write_hidden_cascade.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "single-source Hidden-State write followed by unforced downstream state and output propagation",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states", "candidate logits"],
        "forbidden": ["attention weights or heads", "MLP states or weights", "parameter scan", "gradients",
                      "PCA", "t-SNE", "UMAP", "SAE", "learned probe", "post-reveal hotspot search",
                      "simultaneous writes at multiple path checkpoints"],
        "material": {
            "case_count": len(cases), "balanced_cells": 12, "cases_per_cell": per_cell,
            "partitions": ["prototype_discovery", "clock_selection", "confirmation", "lockbox"],
            "surfaces": ["ordinary", "dictionary", "claim"],
            "eligibility": "clean true, corrupt false, and wrong-identity true all behavior-correct before hidden-state access",
            "case_sha256": core.sha(OUT / "material/path_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/extended_rows.jsonl"),
        },
        "paths": PATHS,
        "known_truth": {
            "systems": 256, "gauges": 4, "splits": ["discovery", "confirmation"],
            "expected_positive": ["family_early", "family_mid", "query_late"],
            "expected_negative": ["family_late", "query_mid"],
            "exact_topology_required": True,
        },
        "observation": {
            "all_paths_run": True,
            "representation": "ordered full-width clean-minus-corrupt states at frozen path events",
            "family_pair_top1_min": 0.70, "surface_top1_min": 0.60,
            "gain_over_best_event_min": 0.05,
            "observation_failure_does_not_cancel_camera_or_causal": True,
        },
        "camera": {
            "calibration_cases": 48, "all_paths_must_pass": True,
            "output_margin_max_abs_diff": 1e-5,
            "checkpoint_relative_l2_max": 1e-6,
        },
        "causal": {
            "arms": ["self", "correct_clean", "wrong_identity_true", "status_true"],
            "single_write_only": True,
            "checkpoint_recovery_projection_median_min": 0.15,
            "checkpoint_correct_over_controls_median_min": 0.10,
            "checkpoint_correct_over_controls_win_min": 0.65,
            "output_gain_median_min": 0.50,
            "output_correct_over_controls_median_min": 0.25,
            "output_correct_over_controls_win_min": 0.65,
            "self_output_max_abs_diff": 1e-4,
            "self_checkpoint_relative_l2_max": 1e-6,
            "all_paths_run_even_after_failures": True,
        },
        "branching": {
            "phase1365": "known-truth exact-topology camera",
            "phase1366": "Qwen natural-response observation for every frozen path",
            "phase1367": "same-input source-write identity camera for every path",
            "phase1368": "all-path single-write downstream cascade competition",
            "finish": "close C056 after phase1368 regardless of sign; no adaptive path addition",
        },
        "claim_boundary": {
            "allowed": "descriptive path readability and calibrated single-write downstream mediation",
            "forbidden": ["relative coding is proved", "a path is minimal", "all layers carry one invariant relation",
                          "cross-model invariance", "parameter mechanism"],
        },
        "stop_rule": "No post-reveal change to object, material, split, model, path, role, layer, null, threshold, or branch.",
        "parent_metrics": {
            "B1_accuracy": behavior_summary["summaries"]["B1_absolute"]["accuracy"],
            "B2_pairwise_win": behavior_summary["summaries"]["B2_relative"]["pairwise_win_fraction"],
            "B3_accuracy": behavior_summary["summaries"]["B3_choice"]["accuracy"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1365_c056_known_truth_camera"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "all_gates_passed": True, "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
