#!/usr/bin/env python3
"""Freeze the Phase1112 early-stop result and close the exact-key registry."""

from __future__ import annotations

import json

import phase1112_one_shot_body_reader_protocol as protocol


def split_diagnostic(model_row: dict) -> dict:
    result = {}
    for pair, pair_row in model_row["pair_results"].items():
        result[pair] = {}
        for split, split_row in pair_row["splits"].items():
            result[pair][split] = {
                "passed": split_row["passed"],
                "exact_accuracy": split_row["exact_accuracy"],
                "regime_accuracy": split_row["regime_accuracy"],
                "congruent_accuracy": split_row["congruent_accuracy"],
                "failed_gates": [
                    key for key, value in split_row["gates"].items() if not value
                ],
            }
    return result


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1112 protocol audit failed")
    if behavior["hidden_scan_authorized"]:
        raise RuntimeError("This finalizer is only valid for the behavior hard-stop branch")

    denial_digests = {}
    for model in protocol.MODELS:
        denial = {
            "schema_version": "phase1112_hidden_access_denial.v1",
            "phase": protocol.PHASE,
            "model": model,
            "hidden_access": False,
            "reason": (
                "Phase1112 cross-model behavior gate failed before hidden access; "
                "the one-shot exact-key registry is closed without head reselection."
            ),
            "behavior_authorization_digest": behavior["authorization_digest"],
        }
        denial["denial_digest"] = protocol.digest(denial)
        protocol.write_json(protocol.OUT_ROOT / "atlas" / model / "denial.json", denial)
        denial_digests[model] = denial["denial_digest"]

    diagnostic = {
        "schema_version": "phase1112_behavior_failure_diagnostic.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": {
            model: {
                "candidate_accuracy": row["candidate_accuracy"],
                "passing_pairs": row["passing_pairs"],
                "split_diagnostic": split_diagnostic(row),
            }
            for model, row in behavior["models"].items()
        },
        "frozen_reading": (
            "GLM4 passes 4/4 pairs on both splits. Qwen3 passes all four confirmation "
            "pairs but zero discovery pairs, mainly because discovery neutral-key cells "
            "and worst conflict cells fail. DS7B passes zero pairs. The interface is "
            "model- and template-world-sensitive despite high aggregate accuracy."
        ),
    }
    diagnostic["diagnostic_digest"] = protocol.digest(diagnostic)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "behavior_failure_diagnostic.json",
        diagnostic,
    )

    predictions = {
        "P1": bool(audit["all_checks_passed"]),
        "P2": False,
        "P3": False,
        "P4": False,
        "P5": False,
        "P6": False,
        "P7": False,
        "P8": True,
    }
    final = {
        "schema_version": "phase1112_one_shot_body_reader_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": audit["audit_digest"],
        "behavior_authorization_digest": behavior["authorization_digest"],
        "behavior_summary_digests": {
            model: row["summary_digest"] for model, row in behavior["models"].items()
        },
        "behavior": {
            "candidate_accuracy": {
                model: row["candidate_accuracy"] for model, row in behavior["models"].items()
            },
            "passing_pairs_by_model": {
                model: row["passing_pairs"] for model, row in behavior["models"].items()
            },
            "cross_model_pairs": behavior["cross_model_pairs"],
            "authorized_models": behavior["authorized_models"],
            "hidden_scan_authorized": behavior["hidden_scan_authorized"],
        },
        "prospective_predictions": predictions,
        "denial_digests": denial_digests,
        "diagnostic_digest": diagnostic["diagnostic_digest"],
        "evidence": {
            "new_exact_key_behavior_robustness": "E3_negative_constraint",
            "body_attention_reader": "not_tested_behavior_denied",
            "body_av_transport": "not_tested_behavior_denied",
            "causal_edge": "not_added",
            "registry_status": "closed_to_further_hotspot_search",
        },
        "causal_staircase_authorized": False,
        "component_head_qkv_neuron_localization_authorized": False,
        "automatic_next_required": False,
        "automatic_next_decision": (
            "Do not revise or rescan the exact-key registry. The next empirical axis "
            "must first freeze and behavior-audit semantic-equivalence materials or a scale arm."
        ),
        "frozen_conclusion": (
            "The one-shot second-hop study terminates at its independent behavior gate. "
            "GLM4 alone is robust; Qwen3 is split-sensitive and DS7B fails all pairs. "
            "No claim about the presence or absence of a body-reader follows because "
            "hidden access was never authorized."
        ),
        "canonical_theory_name_unchanged": "条件化输出场闭合理论",
    }
    final["final_summary_digest"] = protocol.digest(final)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
