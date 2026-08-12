#!/usr/bin/env python3
"""Aggregate KV-head localization and natural validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

import phase1050_head_group_natural_validation_protocol as protocol


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def full_vocabulary_audit(
    model: str,
    summary: dict[str, Any],
    confirmation_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model}.jsonl"
    )
    cases = {int(row["case_index"]): row for row in cases_list}
    case_to_local = {
        int(row["case_index"]): index
        for index, row in enumerate(cases_list)
    }
    atlas = protocol.OUT_ROOT / "atlas" / model
    clean_top1 = np.load(
        atlas / "clean_full_top1.int32.npy", mmap_mode="r"
    )
    natural_top1 = np.load(
        atlas / "natural_swap_full_top1.int32.npy",
        mmap_mode="r",
    )
    selected_top2_slot = list(
        protocol.read_json(
            protocol.OUT_ROOT
            / "protocol"
            / "preregistration.json"
        )["natural_conditions"]
    ).index("selected_top2")

    clean_expected = np.zeros(
        (len(confirmation_targets), 2), dtype=np.int32
    )
    clean_observed = np.zeros_like(clean_expected)
    counterfactual_expected = np.zeros_like(clean_expected)
    for target_slot, target in enumerate(confirmation_targets):
        target_case = cases[int(target["target_case_index"])]
        donor_case = cases[int(target["cross_family_case_index"])]
        target_token = int(
            target_case["candidate_token_ids"][
                int(target["target_family_index"])
            ]
        )
        donor_token = int(
            donor_case["candidate_token_ids"][
                int(target["cross_family_index"])
            ]
        )
        clean_expected[target_slot] = (target_token, donor_token)
        counterfactual_expected[target_slot] = (
            donor_token,
            target_token,
        )
        clean_observed[target_slot] = (
            int(clean_top1[
                case_to_local[int(target["target_case_index"])]
            ]),
            int(clean_top1[
                case_to_local[int(target["cross_family_case_index"])]
            ]),
        )
    patched_observed = np.asarray(
        natural_top1[:, selected_top2_slot, :], dtype=np.int32
    )
    clean_valid = np.all(clean_observed >= 0, axis=1)
    patched_valid = np.all(patched_observed >= 0, axis=1)
    clean_match = clean_observed == clean_expected
    patched_match = patched_observed == counterfactual_expected

    rollouts = summary["rollouts"]
    clean_first_matches_expected = 0
    patched_first_matches_counterfactual = 0
    patched_first_matches_other_clean = 0
    first_token_changed = 0
    arm_count = 0
    for row in rollouts:
        clean_ids = row["clean"]["token_ids"]
        patched_ids = row["patched"]["token_ids"]
        target = next(
            item for item in confirmation_targets
            if int(item["target_index"]) == int(row["target_index"])
        )
        target_case = cases[int(target["target_case_index"])]
        donor_case = cases[int(target["cross_family_case_index"])]
        expected = (
            int(target_case["candidate_token_ids"][
                int(target["target_family_index"])
            ]),
            int(donor_case["candidate_token_ids"][
                int(target["cross_family_index"])
            ]),
        )
        counterfactual = (expected[1], expected[0])
        for arm in (0, 1):
            arm_count += 1
            clean_first_matches_expected += int(
                int(clean_ids[arm][0]) == expected[arm]
            )
            patched_first_matches_counterfactual += int(
                int(patched_ids[arm][0]) == counterfactual[arm]
            )
            patched_first_matches_other_clean += int(
                int(patched_ids[arm][0])
                == int(clean_ids[1 - arm][0])
            )
            first_token_changed += int(
                int(patched_ids[arm][0]) != int(clean_ids[arm][0])
            )
    return {
        "confirmation_pair_count": len(confirmation_targets),
        "finite_clean_pair_count": int(np.sum(clean_valid)),
        "clean_target_exact_next_token_rate": float(np.mean(
            clean_match[clean_valid, 0]
        )),
        "clean_donor_exact_next_token_rate": float(np.mean(
            clean_match[clean_valid, 1]
        )),
        "clean_both_exact_next_token_count": int(np.sum(
            np.all(clean_match, axis=1) & clean_valid
        )),
        "clean_both_exact_next_token_rate": float(np.mean(
            np.all(clean_match[clean_valid], axis=1)
        )),
        "patched_target_to_cross_exact_rate": float(np.mean(
            patched_match[patched_valid, 0]
        )),
        "patched_donor_to_target_exact_rate": float(np.mean(
            patched_match[patched_valid, 1]
        )),
        "patched_both_counterfactual_exact_count": int(np.sum(
            np.all(patched_match, axis=1) & patched_valid
        )),
        "patched_both_counterfactual_exact_rate": float(np.mean(
            np.all(patched_match[patched_valid], axis=1)
        )),
        "rollout": {
            "pair_count": len(rollouts),
            "arm_count": arm_count,
            "clean_first_exact_rate": (
                clean_first_matches_expected / arm_count
                if arm_count else None
            ),
            "patched_first_counterfactual_exact_rate": (
                patched_first_matches_counterfactual / arm_count
                if arm_count else None
            ),
            "patched_first_matches_other_clean_rate": (
                patched_first_matches_other_clean / arm_count
                if arm_count else None
            ),
            "first_token_changed_rate": (
                first_token_changed / arm_count
                if arm_count else None
            ),
        },
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    confirmation_targets = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / "confirmation_targets.jsonl"
    )
    full_vocab = {
        model: full_vocabulary_audit(
            model, summary, confirmation_targets
        )
        for model, summary in summaries.items()
    }
    for model, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model} protocol digest mismatch")
        clean = summary["clean_finite"]
        if (
            clean["finite_value_rate"] < 0.99
            or clean["nonfinite_value_count"]
            != (
                len(summary.get("clean_nonfinite_case_indices", []))
                * len(protocol.material.FAMILIES)
            )
        ):
            raise RuntimeError(f"{model} clean finite audit failed")
        for key in (
            "source_baseline_finite",
            "discovery_finite",
            "confirmation_finite",
            "natural_finite",
        ):
            if not summary[key]["all_finite"]:
                raise RuntimeError(f"{model} {key} failed")

    causal_models = [
        model for model, summary in summaries.items()
        if summary["causal_head_group_gate_passed"]
    ]
    natural_models = [
        model for model, summary in summaries.items()
        if summary["natural_head_group_gate_passed"]
    ]
    minimum = int(prereg["gates"]["minimum_models"])
    if (
        len(causal_models) >= minimum
        and len(natural_models) >= minimum
    ):
        route = "new_task_family_replication"
        rationale = (
            "Model-specific KV groups repeated under held-out causal and "
            "natural counterfactual validation in at least two models."
        )
    elif len(causal_models) >= minimum:
        route = "redesign_natural_counterfactual"
        rationale = (
            "Physical KV groups repeated for the artificial early source "
            "effect, but natural lexical transport did not repeat."
        )
    else:
        route = "stop_head_refinement"
        rationale = (
            "Head-group refinement did not causally repeat in two models; "
            "retain the band-level route without a head claim."
        )

    artifacts = []
    for path in sorted(
        item for item in protocol.OUT_ROOT.rglob("*") if item.is_file()
    ):
        artifacts.append({
            "path": str(path.relative_to(protocol.OUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    aggregate: dict[str, Any] = {
        "schema_version": "phase1050_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "model_results": {
            model: {
                "frozen_kv_groups": summary["frozen_kv_groups"],
                "discovery_ranking": summary["discovery_ranking"],
                "confirmation_analysis": summary[
                    "confirmation_analysis"
                ],
                "specificity": summary[
                    "selected_top2_minus_unselected_top2_mediation"
                ],
                "causal_head_group_gate_passed": summary[
                    "causal_head_group_gate_passed"
                ],
                "natural_analysis": summary["natural_analysis"],
                "natural_head_group_gate_passed": summary[
                    "natural_head_group_gate_passed"
                ],
                "rollout_pair_count": summary["rollout_pair_count"],
                "rollouts": summary["rollouts"],
                "full_vocabulary_audit": full_vocab[model],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
            for model, summary in summaries.items()
        },
        "cross_model_gate": {
            "minimum_models": minimum,
            "causal_passing_models": causal_models,
            "natural_passing_models": natural_models,
            "causal_repeated": len(causal_models) >= minimum,
            "natural_repeated": len(natural_models) >= minimum,
        },
        "automatic_next_decision": {
            "route": route,
            "rationale": rationale,
            "should_continue_automatically": route == (
                "new_task_family_replication"
            ),
        },
        "interpretation_limits": prereg["claim_limits"],
        "artifact_manifest": {
            "file_count": len(artifacts),
            "total_bytes": sum(row["bytes"] for row in artifacts),
            "files": artifacts,
        },
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print({
        "phase": protocol.PHASE,
        "causal_models": causal_models,
        "natural_models": natural_models,
        "next_route": route,
    })


if __name__ == "__main__":
    main()
