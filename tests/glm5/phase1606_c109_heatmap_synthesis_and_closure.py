#!/usr/bin/env python3
"""Phase1606 / C109: synthesize the atlas, export a parameter-level heatmap, and close."""
from __future__ import annotations

import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
SOURCE = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE = 1606
CAMPAIGN = "C109"
FAMILIES = ("attribute_binding", "agent_patient")
PARTITIONS = ("prospective_confirmation", "independent_lockbox")
KEY_STATES = (0, 4, 8, 12, 16, 19, 24, 28, 32, 36)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    observation = core.load(OUT / "analysis/basic_observation_summary.json")
    audit = core.load(OUT / "audit/independent_basic_observation_audit.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    if observation["authorization"] != "run_phase1606_c109_heatmap_synthesis_and_closure" or not audit["all_checks_passed"]:
        raise RuntimeError("C109 closure authorization missing")
    trajectory = core.rows(OUT / "analysis/role_state_truth_trajectory.jsonl")
    pair_energy = {row["pair_id"]: row for row in core.rows(OUT / "analysis/c108_pair_support_energy.jsonl")}
    interventions = {row["pair_id"]: row for row in core.rows(SOURCE / "analysis/fresh_coordinate_intervention_results.jsonl")}
    leverage = []
    for pair_id in sorted(pair_energy):
        energy = pair_energy[pair_id]
        intervention = interventions[pair_id]
        target_gain = float(intervention["write"]["frozen_support"]["truth_direction_gain"])
        wrong_gain = float(intervention["write"]["wrong_family_support"]["truth_direction_gain"])
        target_norm = math.sqrt(max(float(energy["target_support_energy"]), 1e-20))
        wrong_norm = math.sqrt(max(float(energy["same_k_wrong_support_energy"]), 1e-20))
        leverage.append({
            "pair_id": pair_id,
            "family": energy["family"],
            "partition": energy["partition"],
            "code": energy["code"],
            "target_gain": target_gain,
            "wrong_gain": wrong_gain,
            "target_l2_movement": target_norm,
            "wrong_l2_movement": wrong_norm,
            "target_gain_per_l2": target_gain / target_norm,
            "wrong_gain_per_l2": wrong_gain / wrong_norm,
        })
    core.write_rows(OUT / "analysis/c108_pair_coordinate_leverage.jsonl", leverage)
    leverage_summary = []
    for family in FAMILIES:
        for partition in PARTITIONS:
            for code in (1, -1):
                selected = [row for row in leverage if row["family"] == family and row["partition"] == partition and row["code"] == code]
                leverage_summary.append({
                    "family": family,
                    "partition": partition,
                    "code": code,
                    "pairs": len(selected),
                    "median_target_gain_per_l2": float(np.median([row["target_gain_per_l2"] for row in selected])),
                    "median_wrong_gain_per_l2": float(np.median([row["wrong_gain_per_l2"] for row in selected])),
                    "target_efficiency_exceeds_wrong_pairs": int(sum(row["target_gain_per_l2"] > row["wrong_gain_per_l2"] for row in selected)),
                })
    core.write_rows(OUT / "analysis/c108_pair_coordinate_leverage_summary.jsonl", leverage_summary)

    roles = protocol["roles"]
    role_index = {role: index for index, role in enumerate(roles)}
    mean_truth = np.load(OUT / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    effect_rows = []
    for family_index, family in enumerate(FAMILIES):
        for partition_index, partition in enumerate(PARTITIONS):
            for role in roles:
                for state in KEY_STATES:
                    effect_rows.append({
                        "family": family,
                        "partition": partition,
                        "role": role,
                        "state": state,
                        "state_kind": "embedding" if state == 0 else "hidden_state",
                        "effect": "balanced_truth_walsh",
                        "values": np.asarray(mean_truth[family_index, partition_index, role_index[role], state], dtype=np.float32).tolist(),
                    })

    raw_field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    source_rows = core.rows(SOURCE / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    lookup: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(occurrence)
    representatives = []
    for family in FAMILIES:
        for partition in PARTITIONS:
            row_index = next(index for index, row in enumerate(source_rows) if row["family"] == family and row["partition"] == partition)
            representatives.append((row_index, source_rows[row_index]))
    raw_rows = []
    for row_index, row in representatives:
        occurrence = lookup[(row_index, "query_anchor")][0]
        occurrence_index = int(occurrence["occurrence_index"])
        for state in range(37):
            raw_rows.append({
                "case_id": row["case_id"],
                "family": row["family"],
                "partition": row["partition"],
                "truth_factor": row["truth_factor"],
                "surface_factor": row["surface_factor"],
                "distractor_factor": row["distractor_factor"],
                "code": row["code"],
                "role": "query_anchor",
                "subtoken": int(occurrence["subtoken"]),
                "token_position": int(occurrence["token_position"]),
                "token_id": int(occurrence["token_id"]),
                "token_text": occurrence["token_text"],
                "state": state,
                "state_kind": "embedding" if state == 0 else "hidden_state",
                "values": decode_bf16(raw_field[state, occurrence_index]).tolist(),
            })

    candidate_vectors = [
        np.asarray(row["values"], dtype=np.float32)
        for row in effect_rows
        if row["role"] == "query_anchor" and row["state"] == 19
    ]
    mean_abs = np.mean(np.stack([np.abs(vector) for vector in candidate_vectors]), axis=0)
    default_coordinates = np.argsort(-mean_abs, kind="stable")[:64].astype(int).tolist()
    effect_abs = np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in effect_rows])
    raw_abs = np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in raw_rows])
    support_rows = [
        {"name": name, "k": len(values), "values": [1 if coordinate in set(values) else 0 for coordinate in range(2560)]}
        for name, values in protocol["supports"].items()
    ]
    payload = {
        "schema": "c109_role_state_field_atlas.v1",
        "result_type": "role_state_field_atlas_heatmap",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B",
        "title": "C109 Fresh Role-State Truth Field Atlas",
        "dimensions": list(range(2560)),
        "default_coordinates": default_coordinates,
        "coordinate_semantics": "Each column is one residual-stream activation coordinate. state0 is the embedding and state1-state36 are Hidden States. These are not weights, MLP neurons, or attention heads.",
        "scale": {"effect_symmetric_abs_q99": float(np.quantile(effect_abs, 0.99)), "raw_symmetric_abs_q99": float(np.quantile(raw_abs, 0.99))},
        "effect_rows": effect_rows,
        "raw_rows": raw_rows,
        "support_rows": support_rows,
        "trajectory_rows": trajectory,
        "candidate_query_anchor_state19": observation["candidate_query_anchor_state19"],
        "pair_energy_summary": observation["pair_energy_summary"],
        "leverage_summary": leverage_summary,
        "behavior": capture["behavior"],
        "source": {"raw_sha256": capture["raw_sha256"], "observation_sha256": core.sha(OUT / "analysis/basic_observation_summary.json"), "independent_audits": ["11/11 contract", "11/11 capture", "10/10 observation"]},
        "claim_boundary": "This is an observation atlas over C108-exposed controlled-English Qwen data. Stable readout geometry is not causal sufficiency; activations are not weights or semantic neurons; all locators require fresh lexical confirmation.",
        "created_at_utc": now(),
    }
    canonical = OUT / "visualization/c109_role_state_field_atlas.json"
    core.save(canonical, payload)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical, PUBLIC)

    candidate = {row["family"]: row for row in observation["candidate_query_anchor_state19"]}
    def earliest(family: str, role: str, threshold: float, norm_floor: float) -> int | None:
        found = [row["state"] for row in trajectory if row["family"] == family and row["role"] == role and row["state"] > 0 and row["cross_partition_cosine"] >= threshold and min(row["prospective_norm"], row["lockbox_norm"]) >= norm_floor]
        return min(found) if found else None
    closure = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_role_state_observation_atlas_complete",
        "headline": {
            "attribute_query_anchor_state19_cosine": candidate["attribute_binding"]["cross_partition_cosine"],
            "agent_query_anchor_state19_cosine": candidate["agent_patient"]["cross_partition_cosine"],
            "attribute_frozen_support_overlap": [candidate["attribute_binding"]["prospective_frozen_support_overlap"], candidate["attribute_binding"]["lockbox_frozen_support_overlap"]],
            "agent_frozen_support_overlap": [candidate["agent_patient"]["prospective_frozen_support_overlap"], candidate["agent_patient"]["lockbox_frozen_support_overlap"]],
            "attribute_boundary_stable_locator": earliest("attribute_binding", "boundary", 0.9, 1.0),
            "agent_boundary_stable_locator": earliest("agent_patient", "boundary", 0.9, 1.0),
        },
        "new_puzzles": {
            "K283-R1": "readout-control separation: agent-patient query_anchor@state19 retains a stable fresh full truth field and substantial frozen-support overlap, while C108 frozen-support write control failed; representational stability does not imply equal causal output leverage",
            "K284-OBS": "role-state propagation candidate: truth structure is visible at query_anchor before a high-amplitude cross-partition-stable boundary field appears in late states; this is a descriptive locator on already exposed data",
            "K285-CONTROL": "equal support cardinality is not equal intervention energy: target-to-wrong movement ratios differ materially, but the agent wrong support can produce more output gain despite less movement, so energy is a confound and not a sufficient explanation",
        },
        "claim_corrections": {
            "K281-R1": "attribute K256 is a fresh bidirectional truth-factor-associated raw exchange-response candidate; task closure, abstract truth purity, natural necessity, and family-general selection remain open",
            "K282-R1": "only the specific agent K128 support at query_anchor@state19 failed the frozen C108 control criterion; agent coordinate identity, internal relation structure, and alternative dynamic or multi-role mechanisms were not refuted",
        },
        "theory_update": "The working RDC theory should distinguish a readable role-state field R from intervention transport L and output compilation O: stable R may coexist with weak or redirected L-to-O leverage.",
        "claim_boundary": payload["claim_boundary"],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC)},
        "next_authorization": "C110 fresh lexical prospective test of the K283 readout-control separation: freeze field-stability predictions before capture, then compare fixed-support, energy-matched, and multi-role transports without entering attention or MLP internals",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "source": audit["all_checks_passed"],
        "leverage": len(leverage) == 192 and len(leverage_summary) == 8,
        "dimensions": payload["dimensions"] == list(range(2560)),
        "effects": len(effect_rows) == 2 * 2 * 7 * len(KEY_STATES) and all(len(row["values"]) == 2560 for row in effect_rows),
        "raw": len(raw_rows) == 4 * 37 and all(len(row["values"]) == 2560 for row in raw_rows),
        "embedding_hidden": {row["state_kind"] for row in raw_rows} == {"embedding", "hidden_state"},
        "supports": len(support_rows) == 4 and all(len(row["values"]) == 2560 for row in support_rows),
        "identity": core.sha(canonical) == core.sha(PUBLIC),
        "boundary": "not causal sufficiency" in payload["claim_boundary"] and "not weights" in payload["coordinate_semantics"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "asset_sha256": core.sha(canonical), "authorization": "audit_frontend_build_append_memo_and_close_c109"}
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"checks": checks, "headline": closure["headline"], "new_puzzles": closure["new_puzzles"], "heatmap": closure["heatmap"]}, indent=2))


if __name__ == "__main__":
    main()
