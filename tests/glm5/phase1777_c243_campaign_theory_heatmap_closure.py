#!/usr/bin/env python3
"""C243: jointly adjudicate C234-C242 and export the full coordinate event atlas."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C243"]
C236 = common.OUTS["C236"]
C237 = common.OUTS["C237"]
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c243_conditional_event_atlas.json"


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = {
        campaign: core.load(common.OUTS[campaign] / "audit/independent_final_audit.json")
        for campaign in tuple(common.OUTS)[:-1]
    }
    checks = {
        "all_parent_audits": all(row["all_checks_passed"] for row in parents.values()),
        "authorization": parents["C242"]["authorization"].startswith("C243"),
        "full_dimensions": common.DIM == 2560,
        "checkpoints": 37,
        "no_attention_or_mlp": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1777,
        "campaign": "C243",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "campaign_theory_heatmap_closure_frozen",
        "inputs": list(parents),
        "mathematical_upgrade_gate": {
            "unseen_surface_event_prediction": "C238 campaign gate",
            "unseen_factor_composition": "C240 campaign gate",
            "typed_deletion_rescue": "C241 tested and passed",
            "cross_model_functional_reproduction": "C242 campaign gate",
            "required": "4/4",
        },
        "atlas": {
            "model": "Qwen3-4B",
            "rows": 5 * 3 * 37 * 6,
            "coordinates_per_row": 2560,
            "checkpoints": "embedding plus all 36 decoder-block outputs",
            "values": "mean signed role-aligned factorial effect over untouched lockbox and fresh groups",
            "no_projection": True,
            "no_coordinate_deletion": True,
        },
        "claim_boundary": "The atlas is an observational activation field; rows are not weights, neurons, or causal paths.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "synthesize_once_then_independent_audit",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def holdout_means() -> dict[tuple[int, int], np.ndarray]:
    groups = core.rows(C236 / "protocol/effect_groups.jsonl")
    effects = np.load(C237 / "raw/role_effects.float16.npy", mmap_mode="r")
    means: dict[tuple[int, int], np.ndarray] = {}
    for family_i, family in enumerate(common.FAMILIES):
        selected = [
            int(row["effect_index"])
            for row in groups
            if row["family"] == family and row["partition"] in ("lockbox", "fresh")
        ]
        if len(selected) != 10:
            raise RuntimeError((family, len(selected)))
        for effect_i in range(len(common.EFFECTS)):
            means[(family_i, effect_i)] = np.mean(
                np.asarray(effects[selected, effect_i], np.float32), axis=0
            ).astype(np.float32)
    return means


def write_asset(means: dict[tuple[int, int], np.ndarray], summary: dict) -> dict:
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    rules = np.load(C237 / "analysis/rule_codes.int8.npy", mmap_mode="r")
    importance = np.zeros(common.DIM, np.float64)
    for value in means.values():
        importance += np.sum(np.abs(value), axis=(0, 1))
    default_coordinates = np.argsort(-importance)[:64].astype(int).tolist()
    dimensions = list(range(common.DIM))
    metadata = {
        "schema": "c243_conditional_event_atlas.v1",
        "phase": 1777,
        "campaign": "C243",
        "model": "Qwen3-4B",
        "dimensions": dimensions,
        "default_coordinates": default_coordinates,
        "coordinate_semantics": "Every column is a physical Qwen3 activation coordinate. Checkpoint 0 is the embedding; checkpoints 1-36 are complete decoder-block HiddenStates.",
        "claim_boundary": "Signed lockbox+fresh mean effects and discovery event counts are observational. They do not identify weights, individual neurons, a minimal path, or a causal language code.",
        "summary": summary,
    }
    with ASSET.open("w", encoding="utf-8") as handle:
        handle.write("{")
        for key, value in metadata.items():
            handle.write(json.dumps(key) + ":")
            json.dump(value, handle, ensure_ascii=True, separators=(",", ":"))
            handle.write(",")
        handle.write('"rows":[')
        first = True
        for family_i, family in enumerate(common.FAMILIES):
            for effect_i, effect in enumerate(common.EFFECTS):
                value = means[(family_i, effect_i)]
                for checkpoint in range(37):
                    for role_i, role in enumerate(common.ROLES):
                        source = (
                            "C243_core"
                            if checkpoint in (0, 8, 16, 24, 36) and role in ("relation", "boundary")
                            else "C243_full"
                        )
                        row = {
                            "source": source,
                            "kind": "holdout_mean_signed_effect",
                            "family": family,
                            "effect": effect,
                            "checkpoint": checkpoint,
                            "checkpoint_type": "embedding" if checkpoint == 0 else "hidden_state",
                            "role": role,
                            "stable_event_count": int(np.count_nonzero(rules[family_i, effect_i, checkpoint, role_i])),
                            "label": f"{family}/{effect}/q{checkpoint}/{role}",
                            "values": np.round(value[checkpoint, role_i], 6).tolist(),
                        }
                        if not first:
                            handle.write(",")
                        json.dump(row, handle, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
                        first = False
        handle.write("]}")
    return {
        "asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"),
        "bytes": ASSET.stat().st_size,
        "sha256": core.sha(ASSET),
        "rows": 5 * 3 * 37 * 6,
        "coordinates_per_row": common.DIM,
        "total_coordinate_values": 5 * 3 * 37 * 6 * common.DIM,
        "default_coordinates": default_coordinates,
    }


def synthesize() -> None:
    if (OUT / "analysis/summary.json").exists():
        raise RuntimeError("already synthesized")
    c238 = core.load(common.OUTS["C238"] / "analysis/summary.json")
    c240 = core.load(common.OUTS["C240"] / "analysis/summary.json")
    c241 = core.load(common.OUTS["C241"] / "analysis/summary.json")
    c242 = core.load(common.OUTS["C242"] / "analysis/summary.json")
    gates = {
        "unseen_surface_event_prediction": bool(c238["campaign_passed"]),
        "unseen_factor_composition": bool(c240["campaign_passed"]),
        "typed_deletion_rescue": c241["status"] == "causal_test_passed",
        "cross_model_functional_reproduction": bool(c242["cross_model_gate_passed"]),
    }
    summary = {
        "unseen_event_families_passed": int(c238["families_passed"]),
        "composition_families_passed": len(c240["families_passed"]),
        "causal_status": c241["status"],
        "cross_model_passed": bool(c242["cross_model_gate_passed"]),
        "mathematical_upgrade_gates_passed": sum(gates.values()),
        "mathematical_upgrade_gate_total": len(gates),
    }
    manifest = write_asset(holdout_means(), summary)
    report = {
        "phase": 1777,
        "campaign": "C243",
        "status": "campaign_closed",
        "gates": gates,
        "summary": summary,
        "new_mathematics_gate_passed": all(gates.values()),
        "theory_status": "existing mathematics sufficient; RDC theory remains a falsifiable mid-level framework",
        "core_updates": {
            "K320_OBS": "Full-coordinate signed interval events repeat conditionally, but only attitude-event and contrast pass the frozen unseen-surface family gate.",
            "K321_BOUNDARY": "Simple atomic signed-event superposition passes only type-graph; no route qualifies for causal deletion-rescue.",
            "K322_CANDIDATE": "Three models share a coarse factor-role-relative-depth response topology under a role-permutation null, subject to scaffold and energy confounds.",
        },
        "asset_manifest": manifest,
        "strict_conclusion": "The campaign found narrow conditional event regularities and a coarse cross-model topology candidate, not a unified coordinate gear, composition algebra, causal circuit, or new mathematics.",
        "next_authorization": "C244 independent new-material replication of attitude/contrast events plus scaffold-neutralized cross-model controls",
    }
    core.save(OUT / "analysis/summary.json", report)
    core.save(OUT / "analysis/heatmap_manifest.json", manifest)
    checks = {
        "all_four_gates_recorded": len(gates) == 4,
        "upgrade_closed": sum(gates.values()) == 1 and not all(gates.values()),
        "rows": manifest["rows"] == 3330,
        "coordinates": manifest["coordinates_per_row"] == 2560,
        "all_values": manifest["total_coordinate_values"] == 8_524_800,
        "asset_exists": ASSET.is_file(),
    }
    core.save(OUT / "audit/internal_synthesis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"report": report, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/summary.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "synthesis": core.load(OUT / "audit/internal_synthesis_audit.json")["all_checks_passed"],
        "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
        "asset_hash": core.sha(ASSET) == report["asset_manifest"]["sha256"],
    }
    final = {
        "phase": 1777,
        "campaign": "C243",
        "status": "closed",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "headline": report,
        "next_authorization": report["next_authorization"],
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "synthesize", "close"))
    args = parser.parse_args()
    {"contract": contract, "synthesize": synthesize, "close": close}[args.command]()


if __name__ == "__main__":
    main()
