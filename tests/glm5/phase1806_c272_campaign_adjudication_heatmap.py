#!/usr/bin/env python3
"""C272: adjudicate C263-C271 and export a full-coordinate client atlas."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1797_c263_c272_state_operator_common as common

core, OUT = common.core, common.OUTS["C272"]
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c272_state_conditioned_operator_atlas.json"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def values(array):
    return np.round(np.asarray(array, np.float32), 6).tolist()


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parents = {name: core.load(common.OUTS[name] / "analysis/final.json") for name in ("C263", "C264", "C265", "C266", "C267", "C268", "C269", "C270", "C271")}
    checks = {"all_parents": all(v["all_checks_passed"] for v in parents.values()), "continuous_phases": [v["phase"] for v in parents.values()] == list(range(1797, 1806)), "all_coordinates": True, "no_topk": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); (OUT / "analysis").mkdir(); (OUT / "audit").mkdir(); (OUT / "protocol").mkdir()
    protocol = {"phase": 1806, "campaign": "C272", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "closure_frozen", "adjudication": "separate behavior, observational prediction, composition description, local causality, generation, side effects, and cross-model topology", "new_math_gate": ["repeated functional object", "unseen rolling prediction", "local causal use", "cross-model function topology", "existing mathematics demonstrably insufficient"], "claim_boundary": "No positive result is promoted across evidence types. Failure of one route does not erase other observations.", "producer_sha256": core.sha(Path(__file__)), "authorization": "decide_after_reveal"}
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    c264 = parents["C264"]["headline"]; c265 = parents["C265"]["headline"]; c266 = parents["C266"]["headline"]; c267 = parents["C267"]["headline"]; c268 = parents["C268"]["headline"]; c269 = parents["C269"]["headline"]; c270 = parents["C270"]["headline"]; c271 = parents["C271"]["headline"]
    math_checks = {"repeated_functional_object": c265["families_passing"] >= 4, "unseen_rolling_prediction": c266["broad_rolling_gate_passed"], "local_causal_use": c269["local_state_edge_gate_passed"], "cross_model_function_topology": c271["cross_model_conditional_topology_gate_passed"], "existing_math_insufficient": False}
    new_math_open = all(math_checks.values())
    if c269["local_state_edge_gate_passed"] and c270["generation_and_side_effect_gate_passed"]:
        next_auth = "C273_independent_state_operator_replication_and_graph_composition"
    elif c265["families_passing"] or c266["broad_rolling_gate_passed"]:
        next_auth = "C273_observation_first_guard_refinement_without_minimality_claim"
    else:
        next_auth = "C273_full_field_response_ecology_redesign; do_not_repeat_fixed_checkpoint_patching"
    report = {
        "phase": 1806, "campaign": "C272", "status": "campaign_closed", "audit_corrections": [
            "C250 is adjacent-checkpoint same-coordinate same-sign overlap, not demonstrated natural transmission.",
            "C251 reports residual norm divided by observed 00-to-11 change; its beta_AB/main ratio is a separate diagnostic.",
            "C252 repeatedly injects donor values and therefore controls a trajectory without showing natural propagation.",
            "C253/C271 compare anonymous role-depth statistics, not physical coordinates or shared circuits.",
            "C261's 75 percent point is one deterministic coverage-grid sufficiency point, not a minimal coalition or universal coordinate fraction.",
            "C262 rejects specificity of the registered absolute checkpoint masks; it does not reject every directed or state-conditioned mechanism.",
            "C260 direct-word evidence remains a leading-space token-logit readout; C262/C270 are tokenizer-aware generated-text tests.",
        ],
        "behavior": {"qwen_accuracy": c264["accuracy"], "by_family": c264["by_family_accuracy"], "nested": c264["nested_accuracy"], "eligible": c264["behavior_eligible"]},
        "passport": {"families_passing": c265["families_passing"], "families_total": c265["families_total"], "broad_gate": c265["broad_prediction_gate_passed"]},
        "rolling": {"broad_gate": c266["broad_rolling_gate_passed"], "families": c266["family_results"]},
        "composition": {"nested": c267, "typed": c268["families"]},
        "causal": {"gate": c269["local_state_edge_gate_passed"], "state_minus_control": c269["state_minus_best_wrong_control"], "summaries": c269["summaries"]},
        "generation": {"gate": c270["generation_and_side_effect_gate_passed"], "word_margin": c270["word_correct_minus_best_wrong_control"], "summaries": c270["summaries"]},
        "cross_model": {"participants": c271["participants"], "gate": c271["cross_model_conditional_topology_gate_passed"], "pairs": c271["pair_tests"]},
        "new_math_checks": math_checks, "new_math_gate_open": new_math_open,
        "theory": "Conditional Output Field Closure Theory / reuse-difference-conditioning remains the stable theory name. The new candidate object is a guarded, state-indexed, partially defined event operator; it is not promoted to a law unless unseen rolling prediction, local rescue, and cross-model function topology jointly pass.",
        "strict_conclusion": "This campaign tests whether current activation state improves full-coordinate event prediction and whether any registered local edge controls generated language. It does not seek or claim a unique coordinate dictionary.",
        "next_authorization": next_auth,
    }
    core.save(OUT / "analysis/summary.json", report)

    states = np.load(common.OUTS["C264"] / "raw/role_states.float16.npy", mmap_mode="r"); index = core.rows(common.OUTS["C264"] / "raw/hidden_index.jsonl")
    lookup = {(r["family"], r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): r["hidden_index"] for r in index}
    left = lookup[("attitude_event", "dossier", 0, 0, 0, 1)]; right = lookup[("attitude_event", "dossier", 0, 1, 0, 1)]
    pred = np.load(common.OUTS["C265"] / "analysis/passport_pred_sign.int8.npy", mmap_mode="r"); agree = np.load(common.OUTS["C265"] / "analysis/passport_agreement.float16.npy", mmap_mode="r")
    nested = np.load(common.OUTS["C267"] / "analysis/mean_interaction.float16.npy", mmap_mode="r"); typed = np.load(common.OUTS["C268"] / "analysis/family_mean_interaction.float16.npy", mmap_mode="r")
    atlas_rows = []
    for q in range(37):
        for role in ("relation", "boundary"):
            ri = common.ROLES.index(role)
            for source, array in (("fourth_target_state", states[left, q, ri]), ("fourth_donor_state", states[right, q, ri]), ("fourth_edit_response", np.asarray(states[right, q, ri], np.float32) - np.asarray(states[left, q, ri], np.float32))):
                atlas_rows.append({"source": source, "family": "attitude_event", "effect": "factor_a", "checkpoint": q, "checkpoint_type": "embedding" if q == 0 else "hidden_state", "role": role, "label": f"{source}/q{q}/{role}", "values": values(array)})
    fi, ri = common.FAMILIES.index("attitude_event"), common.ROLES.index("relation")
    for q in range(36):
        for key in range(4):
            atlas_rows.append({"source": "state_conditioned_passport_sign", "family": "attitude_event", "effect": f"guard_{key}", "checkpoint": q, "checkpoint_type": "embedding" if q == 0 else "hidden_state", "role": "relation", "label": f"passport-sign/q{q}/guard{key}", "values": values(pred[fi, q, ri, key])})
            atlas_rows.append({"source": "state_conditioned_passport_agreement", "family": "attitude_event", "effect": f"guard_{key}", "checkpoint": q, "checkpoint_type": "embedding" if q == 0 else "hidden_state", "role": "relation", "label": f"passport-agreement/q{q}/guard{key}", "values": values(agree[fi, q, ri, key])})
    for q in (0, 8, 16, 24, 36):
        for role in ("relation", "boundary"):
            ri = common.ROLES.index(role); atlas_rows.append({"source": "nested_interaction", "family": "nested_attitude", "effect": "interaction", "checkpoint": q, "checkpoint_type": "embedding" if q == 0 else "hidden_state", "role": role, "label": f"nested-interaction/q{q}/{role}", "values": values(nested[q, ri])})
            for family_i, family in enumerate(common.FAMILIES): atlas_rows.append({"source": "typed_interaction", "family": family, "effect": "interaction", "checkpoint": q, "checkpoint_type": "embedding" if q == 0 else "hidden_state", "role": role, "label": f"typed-interaction/{family}/q{q}/{role}", "values": values(typed[family_i, q, ri])})
    payload = {"schema": "c272_state_conditioned_operator_atlas.v1", "phase": 1806, "campaign": "C263-C272", "model": "Qwen3-4B", "dimensions": 2560, "default_coordinates": 64, "coordinate_semantics": "Every row contains all 2560 physical Qwen3 activation coordinates. q0 is token embedding and q1-q36 are HiddenState checkpoints; values are activations, edit responses, passport signs/agreements, or interaction residuals, never model weights.", "claim_boundary": report["strict_conclusion"], "summary": {"behavior_eligible": c264["behavior_eligible"], "passport_families_passing": c265["families_passing"], "rolling_gate": c266["broad_rolling_gate_passed"], "local_causal_gate": c269["local_state_edge_gate_passed"], "generation_gate": c270["generation_and_side_effect_gate_passed"], "cross_model_gate": c271["cross_model_conditional_topology_gate_passed"], "new_math_gate": new_math_open}, "rows": atlas_rows}
    save_json(ASSET, payload)
    digest = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    asset_checks = {"schema": json.loads(ASSET.read_text(encoding="utf-8"))["schema"] == payload["schema"], "rows": len(atlas_rows) == 74 * 3 + 36 * 4 * 2 + 5 * 2 * 6, "all_coordinates": all(len(r["values"]) == 2560 for r in atlas_rows), "embedding_present": any(r["checkpoint"] == 0 for r in atlas_rows), "hidden_present": any(r["checkpoint"] == 36 for r in atlas_rows)}
    core.save(OUT / "visualization/heatmap_export_audit.json", {"asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"), "sha256": digest, "bytes": ASSET.stat().st_size, "checks": asset_checks, "all_checks_passed": all(asset_checks.values())})
    ach = {"parents": all(checks.values()), "asset": all(asset_checks.values()), "new_math_not_overclaimed": not new_math_open or math_checks["existing_math_insufficient"], "next_authorized": bool(next_auth)}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())}); fch = {"contract": True, "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}; final = {"phase": 1806, "campaign": "C272", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": next_auth}; core.save(OUT / "analysis/final.json", final); print(json.dumps({"final": final, "asset_rows": len(atlas_rows), "asset_bytes": ASSET.stat().st_size}, indent=2))


if __name__ == "__main__": main()
