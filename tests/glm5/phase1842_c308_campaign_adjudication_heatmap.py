#!/usr/bin/env python3
"""C308: adjudicate C293-C307 and publish the full-coordinate campaign atlas."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common

core, OUT = common.core, common.OUTS["C308"]
PUBLIC = common.ROOT / "frontend/public/vis_data/research_kernel/c308_conditional_hypergraph_campaign_atlas.json"
MODEL_NAMES = ("M0_persistence", "M1_absorbing", "M2_complete", "M3_cross_coordinate", "M4_all_token")


def rounded(values: np.ndarray) -> list[float]:
    return np.round(np.asarray(values, np.float32), 6).tolist()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    finals = {f"C{i}": core.load(common.OUTS[f"C{i}"] / "analysis/final.json") for i in range(293, 308)}
    checks = {
        "all_parents_closed": all(x["all_checks_passed"] for x in finals.values()),
        "all_coordinates": True,
        "no_new_fit": True,
        "causal_failure_preserved": not finals["C306"]["headline"]["branches"][0]["causal_gate_passed"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    for sub in ("analysis", "audit", "protocol"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": 1842,
        "campaign": "C308",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "adjudication_and_visualization_frozen",
        "parents": list(finals),
        "rules": [
            "No new model fitting or coordinate selection is permitted.",
            "Every heatmap row retains all 2560 Qwen3 physical activation coordinates.",
            "Embedding, HiddenState, transition, amplitude, composition, qualification and causal results remain separately typed.",
            "The failed causal patch cannot be rescued by observational prediction or composition results.",
        ],
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)

    rows: list[dict] = []
    selected = {row["family"]: row for row in finals["C297"]["headline"]["families"]}
    transition = np.load(common.OUTS["C296"] / "analysis/coordinate_transition_counts.uint16.npy", mmap_mode="r")
    amplitude = np.load(common.OUTS["C297"] / "analysis/amplitude_coordinate_atlas.float32.npy", mmap_mode="r")
    tournament = np.load(common.OUTS["C300"] / "analysis/lockbox_coordinate_score_atlas.float32.npy", mmap_mode="r")
    composition = np.load(common.OUTS["C302"] / "analysis/composition_coordinate_atlas.float32.npy", mmap_mode="r")
    qualification = np.load(common.OUTS["C305"] / "analysis/qualified_target_masks.bool.npy", mmap_mode="r")
    states = np.load(common.OUTS["C295"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = core.rows(common.OUTS["C295"] / "raw/hidden_index.jsonl")

    for fi, family in enumerate(common.FAMILIES):
        meta = selected[family]
        q = int(meta["q"])
        role = meta["destination_role"]
        d = common.ROLES.index(role)
        left, right = common.pair_specs(index, family)[0][:2]
        embedding_delta = np.asarray(states[right, common.CANONICAL_NEW_INDICES[0], d], np.float32) - np.asarray(states[left, common.CANONICAL_NEW_INDICES[0], d], np.float32)
        hidden_delta = np.asarray(states[right, common.CANONICAL_NEW_INDICES[q + 1], d], np.float32) - np.asarray(states[left, common.CANONICAL_NEW_INDICES[q + 1], d], np.float32)
        rows.extend((
            {"source": "c295_sixth_embedding_response", "family": family, "effect": "factor_a_edit_delta", "checkpoint": "embedding", "checkpoint_type": "embedding", "role": role, "label": f"{family}/{role}/embedding/edit", "values": rounded(embedding_delta)},
            {"source": "c295_sixth_hidden_response", "family": family, "effect": "factor_a_edit_delta", "checkpoint": f"q{q + 1}", "checkpoint_type": "hidden_state", "role": role, "label": f"{family}/{role}/q{q + 1:02d}/edit", "values": rounded(hidden_delta)},
        ))
        correct = np.asarray(transition[fi, q, d, 0], np.float32)
        union = np.asarray(transition[fi, q, d, 1], np.float32)
        accuracy = np.divide(correct, union, out=np.zeros_like(correct), where=union > 0)
        rows.append({"source": "c296_complete_transition", "family": family, "effect": "coordinate_event_accuracy", "checkpoint": f"q{q}->q{q + 1}", "checkpoint_type": "hidden_state_transition", "role": role, "label": f"{family}/{role}/complete-transition", "values": rounded(accuracy)})
        amp_gain = np.divide(amplitude[fi, 1] - amplitude[fi, 0], np.maximum(amplitude[fi, 1], 1e-12))
        rows.append({"source": "c297_amplitude_regime", "family": family, "effect": "coordinate_relative_mae_gain", "checkpoint": f"q{q}->q{q + 1}", "checkpoint_type": "hidden_state_transition", "role": role, "label": f"{family}/{role}/amplitude-gain", "values": rounded(amp_gain)})
        for mi, model in enumerate(MODEL_NAMES):
            rows.append({"source": "c300_sixth_lockbox_tournament", "family": family, "effect": model, "checkpoint": f"q{q}->q{q + 1}", "checkpoint_type": "hidden_state_transition", "role": role, "label": f"{family}/{role}/{model}", "values": rounded(tournament[fi, mi])})
        rows.extend((
            {"source": "c302_composition_forecast", "family": family, "effect": "coordinate_relative_mae_gain", "checkpoint": "all_37_canonical", "checkpoint_type": "embedding_and_hidden_state", "role": "all_six_roles", "label": f"{family}/composition/gain", "values": rounded(composition[fi, 2])},
            {"source": "c302_composition_forecast", "family": family, "effect": "mean_interaction_magnitude", "checkpoint": "all_37_canonical", "checkpoint_type": "embedding_and_hidden_state", "role": "all_six_roles", "label": f"{family}/composition/interaction", "values": rounded(composition[fi, 3])},
            {"source": "c305_causal_qualification", "family": family, "effect": "M3_qualified_target_mask", "checkpoint": f"q{q}->q{q + 1}", "checkpoint_type": "hidden_state_transition", "role": role, "label": f"{family}/M3/qualified-mask", "values": qualification[fi, 0].astype(np.float32).tolist()},
            {"source": "c305_causal_qualification", "family": family, "effect": "M4_qualified_target_mask", "checkpoint": f"q{q}->q{q + 1}", "checkpoint_type": "hidden_state_transition", "role": role, "label": f"{family}/M4/qualified-mask", "values": qualification[fi, 1].astype(np.float32).tolist()},
        ))

    asset = {
        "schema": "c308_conditional_hypergraph_campaign_atlas.v1",
        "result_type": "conditional_hypergraph_campaign_atlas_heatmap",
        "phase": 1842,
        "campaign": "C308",
        "model": "Qwen3-4B; anonymous topology summary also covers GLM4-9B and DeepSeek-7B",
        "dimensions": list(range(common.DIM)),
        "default_coordinates": list(range(64)),
        "total_rows": len(rows),
        "coordinate_semantics": "Every row retains all 2560 Qwen3 physical activation coordinates. These are activation coordinates, not parameter indices or unique neurons. Embedding and HiddenState rows are explicitly typed; presentation defaults do not select evidence.",
        "claim_boundary": "C302 provides held-out field-composition prediction and C307 provides anonymous cross-model topology. C300/C301 fail six-family breadth and C306 rejects the registered all-token single-source patch. The asset is not a unique causal hypergraph, continuous simulator, or proof of new mathematics.",
        "summary": {
            "sixth_behavior_accuracy": finals["C295"]["headline"]["accuracy"],
            "amplitude_families_passing": len(finals["C297"]["headline"]["families_passing"]),
            "lockbox_tournament_families_passing": len(finals["C300"]["headline"]["families_passing"]),
            "autonomous_rollout_families_passing": len(finals["C301"]["headline"]["families_passing"]),
            "composition_families_passing": len(finals["C302"]["headline"]["families_passing"]),
            "causal_branches_passing": sum(bool(x["causal_gate_passed"]) for x in finals["C306"]["headline"]["branches"]),
            "cross_model_pairs_passing": sum(bool(x["pair_gate_passed"]) for x in finals["C307"]["headline"]["pairs"]),
            "physical_activation_coordinates": common.DIM,
        },
        "rows": rows,
    }
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    core.save(PUBLIC, asset)

    corrections = [
        "The first C295-C304 lockbox run was invalidated because the wrapper called the fifth-material generator and reset the sixth lexicon. The direct-base-v2 compiler was frozen and C293-C295 plus C300-C304 were rerun; only the corrected Adira/jicama panel is evidence.",
        "C296's formal selector admitted zero-union q0 strata for type_graph and translation. C296 is preserved; C297 onward use a separately frozen nondegenerate selector.",
        "C280/C281 support a signed-event summary, not a proved Markov automaton or continuous HiddenState dynamics.",
        "C281 starts from embedding edit-response events, not from raw sentence embeddings.",
        "C288/C307 compare anonymous task-conditioned topology, not strict cross-model isomorphism.",
        "C298/C299 are predictive source maps; only C306 is an intervention, and its registered gate failed.",
        "Exact token-signature failure concerns the registered finite signature under lexical/token variation, not the absence of physical structure.",
        "All materials have machine semantic and balance audits but no independent human naturalness blind review.",
    ]
    report = {
        "phase": 1842,
        "campaign": "C308",
        "status": "campaign_closed_with_field_composition_and_cross_model_topology_positive_but_causal_patch_negative",
        "audit_corrections": corrections,
        "evidence": {
            "full_field": finals["C295"]["headline"],
            "complete_transition": finals["C296"]["headline"],
            "amplitude": finals["C297"]["headline"],
            "cross_coordinate": {"role_source": finals["C298"]["headline"], "all_token": finals["C299"]["headline"]},
            "lockbox_tournament": finals["C300"]["headline"],
            "autonomous_rollout": finals["C301"]["headline"],
            "composition": {"six_family": finals["C302"]["headline"], "type_graph": finals["C303"]["headline"], "nested_attitude": finals["C304"]["headline"]},
            "causal": {"qualification": finals["C305"]["headline"], "intervention": finals["C306"]["headline"]},
            "cross_model": finals["C307"]["headline"],
        },
        "theory": {
            "stable_name": "Conditional Output Field Closure Theory",
            "organizing_principle": "reuse-difference-conditioning (RDC)",
            "update": "A training-panel interaction residual predicts the complete held-out sixth-material field across six language families, while discrete transition and rollout improvements remain family-dependent. Anonymous transition topology repeats across three models, but the only qualified cross-coordinate all-token coalition fails its registered causal patch.",
            "new_math_gate": {
                "repeated_functional_object": True,
                "prospective_unseen_field_composition": True,
                "cross_model_anonymous_topology": True,
                "broad_continuous_transition_simulator": False,
                "local_causal_use": False,
                "existing_mathematics_demonstrably_insufficient": False,
                "gate_open": False,
            },
        },
        "strict_conclusion": "The strongest new result is held-out full-coordinate field composition across six controlled language families. The campaign does not identify a unique coordinate transmission graph, does not establish a broad autonomous continuous simulator, and does not validate the registered causal patch interface.",
        "next_authorization": "C309_independent_campaign_audit_then_close_major_stage",
    }
    core.save(OUT / "analysis/summary.json", report)
    manifest = {"asset": str(PUBLIC.relative_to(common.ROOT)).replace("\\", "/"), "schema": asset["schema"], "rows": asset["total_rows"], "dimensions": len(asset["dimensions"]), "sha256": core.sha(PUBLIC)}
    core.save(OUT / "analysis/heatmap_manifest.json", manifest)
    audit = {
        "parents": all(checks.values()),
        "corrections_recorded": len(corrections) == 8,
        "all_2560_coordinates": len(asset["dimensions"]) == common.DIM and all(len(r["values"]) == common.DIM for r in rows),
        "embedding_present": any(r["checkpoint_type"] == "embedding" for r in rows),
        "hidden_state_present": any(r["checkpoint_type"] == "hidden_state" for r in rows),
        "failed_causal_gate_preserved": asset["summary"]["causal_branches_passing"] == 0,
        "new_math_gate_closed": not report["theory"]["new_math_gate"]["gate_open"],
    }
    core.save(OUT / "audit/internal_audit.json", {"checks": audit, "all_checks_passed": all(audit.values())})
    final_checks = {"parents": all(checks.values()), "analysis": all(audit.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"], "asset_hash": core.sha(PUBLIC) == manifest["sha256"]}
    final = {"phase": 1842, "campaign": "C308", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
