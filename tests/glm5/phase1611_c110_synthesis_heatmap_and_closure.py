#!/usr/bin/env python3
"""Phase1611 / C110: synthesize the fresh replication, update the atlas, and close."""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C109 = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
OUT = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE = 1611
CAMPAIGN = "C110"
FAMILIES = ("attribute_binding", "agent_patient")
PARTITIONS = ("fresh_confirmation", "fresh_lockbox")
RAW_STATES = (0, 19, 36)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def med(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize_pair_contrasts(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["partition"], int(row["code"]))].append(row)
    summaries = []
    for family, partition, code in sorted(groups):
        selected = groups[(family, partition, code)]
        target = [float(row["modes"]["frozen_support"]["truth_direction_gain"]) for row in selected]
        permuted = [float(row["modes"]["coordinate_permuted"]["truth_direction_gain"]) for row in selected]
        whole = [float(row["modes"]["whole_query_anchor"]["truth_direction_gain"]) for row in selected]
        multi = [float(row["modes"]["whole_query_anchor_plus_focus_record"]["truth_direction_gain"]) for row in selected]
        whole_flips = [bool(row["modes"]["whole_query_anchor"]["truth_flip"]) for row in selected]
        multi_flips = [bool(row["modes"]["whole_query_anchor_plus_focus_record"]["truth_flip"]) for row in selected]
        summaries.append({
            "family": family,
            "partition": partition,
            "code": code,
            "pairs": len(selected),
            "median_frozen_support_gain": med(target),
            "median_coordinate_permuted_gain": med(permuted),
            "frozen_support_gain_gt_permuted_pairs": int(sum(left > right for left, right in zip(target, permuted, strict=True))),
            "frozen_support_median_gt_permuted": med(target) > med(permuted),
            "median_whole_query_gain": med(whole),
            "median_query_plus_record_gain": med(multi),
            "query_plus_record_gain_gt_whole_pairs": int(sum(left > right for left, right in zip(multi, whole, strict=True))),
            "query_plus_record_median_gt_whole": med(multi) > med(whole),
            "query_plus_record_additional_truth_flips": int(sum(multi_value and not whole_value for multi_value, whole_value in zip(multi_flips, whole_flips, strict=True))),
        })
    return summaries


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    field = core.load(OUT / "analysis/field_prediction_adjudication.json")
    transport = core.load(OUT / "analysis/transport_adjudication.json")
    capture_audit = core.load(OUT / "audit/independent_capture_audit.json")
    field_audit = core.load(OUT / "audit/independent_field_adjudication_audit.json")
    transport_audit = core.load(OUT / "audit/independent_transport_audit.json")
    if not all(item["all_checks_passed"] for item in (capture_audit, field_audit, transport_audit)):
        raise RuntimeError("C110 independent source audit missing")
    if transport["authorization"] != "run_phase1611_c110_synthesis_heatmap_and_major_stage_closure":
        raise RuntimeError("C110 closure authorization missing")

    base_path = C109 / "visualization/c109_role_state_field_atlas.json"
    payload = core.load(base_path)
    mean_truth = np.load(OUT / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    roles = protocol["roles"]
    role_index = {role: index for index, role in enumerate(roles)}
    for row in payload["effect_rows"]:
        row.setdefault("dataset", "C109_exposed")
    for family_index, family in enumerate(FAMILIES):
        for partition_index, partition in enumerate(PARTITIONS):
            payload["effect_rows"].append({
                "dataset": "C110_fresh",
                "family": family,
                "partition": partition,
                "role": "query_anchor",
                "state": 19,
                "state_kind": "hidden_state",
                "effect": "balanced_truth_walsh",
                "values": np.asarray(mean_truth[family_index, partition_index, role_index["query_anchor"], 19], dtype=np.float32).tolist(),
            })

    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    occurrence_lookup: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for occurrence in manifest:
        occurrence_lookup[(int(occurrence["row_index"]), occurrence["role"])].append(occurrence)
    archive = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    for row in payload["raw_rows"]:
        row.setdefault("dataset", "C109_exposed")
    for family in FAMILIES:
        for partition in PARTITIONS:
            row_index = next(index for index, row in enumerate(compiled) if row["family"] == family and row["partition"] == partition)
            row = compiled[row_index]
            occurrence = occurrence_lookup[(row_index, "query_anchor")][0]
            occurrence_index = int(occurrence["occurrence_index"])
            for state in RAW_STATES:
                payload["raw_rows"].append({
                    "dataset": "C110_fresh",
                    "case_id": row["case_id"],
                    "family": family,
                    "partition": partition,
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
                    "values": decode_bf16(archive[state, occurrence_index]).tolist(),
                })

    transport_rows = core.rows(OUT / "analysis/fresh_transport_results.jsonl")
    pair_contrasts = summarize_pair_contrasts(transport_rows)
    core.write_rows(OUT / "analysis/transport_pair_contrast_summary.jsonl", pair_contrasts)
    trajectory = core.rows(OUT / "analysis/fresh_role_state_trajectory.jsonl")
    payload.update({
        "phase": PHASE,
        "campaign": "C109-C110",
        "title": "C109-C110 Prospective Role-State Readout / Control Atlas",
        "fresh_c110": {
            "field_prediction": field,
            "transport": transport,
            "transport_pair_contrasts": pair_contrasts,
            "trajectory_rows": trajectory,
            "behavior": capture["behavior"],
        },
        "source": {
            "c109_asset_sha256": core.sha(base_path),
            "c110_raw_sha256": capture["raw_sha256"],
            "c110_field_sha256": core.sha(OUT / "analysis/field_prediction_adjudication.json"),
            "c110_transport_sha256": core.sha(OUT / "analysis/transport_adjudication.json"),
            "independent_audits": ["9/9 capture", "7/7 field", "10/10 transport"],
        },
        "claim_boundary": "C110 prospectively confirms stable full-coordinate truth readout and relation-specific output leverage in a third fresh controlled lexicon. It does not identify one-to-one semantic coordinate values, a minimal role coalition, natural-language necessity, weights, or universal semantic neurons.",
        "created_at_utc": now(),
    })
    effect_abs = np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["effect_rows"]])
    raw_abs = np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["raw_rows"]])
    payload["scale"] = {
        "effect_symmetric_abs_q99": float(np.quantile(effect_abs, 0.99)),
        "raw_symmetric_abs_q99": float(np.quantile(raw_abs, 0.99)),
    }
    candidates = [
        np.asarray(row["values"], dtype=np.float32)
        for row in payload["effect_rows"]
        if row["role"] == "query_anchor" and int(row["state"]) == 19
    ]
    mean_abs = np.mean(np.stack([np.abs(value) for value in candidates]), axis=0)
    payload["default_coordinates"] = np.argsort(-mean_abs, kind="stable")[:64].astype(int).tolist()

    canonical = OUT / "visualization/c109_c110_role_state_readout_control_atlas.json"
    core.save(canonical, payload)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical, PUBLIC)

    field_by_family = {row["family"]: row for row in field["results"]}
    contrast_by_family = {
        family: [row for row in pair_contrasts if row["family"] == family]
        for family in FAMILIES
    }
    closure = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_readout_control_separation_major_stage_complete",
        "headline": {
            "field_cross_partition_cosine": {family: field_by_family[family]["cross_fresh_partition_cosine"] for family in FAMILIES},
            "field_prediction_passed": {family: field_by_family[family]["prediction_passed"] for family in FAMILIES},
            "attribute_target_efficiency_gt_wrong_cells": transport["prediction"]["attribute_target_efficiency_gt_wrong_cells"],
            "agent_target_efficiency_lt_wrong_cells": transport["prediction"]["agent_target_efficiency_lt_wrong_cells"],
            "attribute_frozen_median_gt_permuted_cells": int(sum(row["frozen_support_median_gt_permuted"] for row in contrast_by_family["attribute_binding"])),
            "agent_frozen_median_gt_permuted_cells": int(sum(row["frozen_support_median_gt_permuted"] for row in contrast_by_family["agent_patient"])),
            "attribute_multi_role_median_gt_whole_cells": int(sum(row["query_plus_record_median_gt_whole"] for row in contrast_by_family["attribute_binding"])),
            "agent_multi_role_median_gt_whole_cells": int(sum(row["query_plus_record_median_gt_whole"] for row in contrast_by_family["agent_patient"])),
            "additional_multi_role_truth_flips": int(sum(row["query_plus_record_additional_truth_flips"] for row in pair_contrasts)),
        },
        "new_puzzles": {
            "K286-R1": "prospective third-lexicon field stability: both attribute and agent query_anchor@state19 full truth fields pass frozen cross-partition, C109-reference, and support-overlap predictions",
            "K287-R1": "prospective readout-control separation: in all four partition-code cells the attribute frozen support is more output-efficient than same-K wrong support, while the agent frozen support is less efficient in all four; stable readout does not determine output leverage",
            "K288-BOUND": "coordinate and role identity remain unresolved: attribute coordinate-permuted donor values retain comparable median gain in three of four cells, and adding focus_record changes median gain without producing any additional truth flips over whole-query transport",
        },
        "theory_update": "The RDC working theory now has a prospectively repeated separation between readable field R and output leverage O after transport L. Support location, transported value identity, movement energy, and role coalition must be treated as separate objects.",
        "claim_boundary": payload["claim_boundary"],
        "heatmap": {
            "path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"),
            "bytes": PUBLIC.stat().st_size,
            "sha256": core.sha(PUBLIC),
        },
        "next_authorization": "C111 observation-first value-identity and role-coalition atlas on the frozen C109-C110 archives; analyze value-preserving shuffles and role-conditioned response before any new model run",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "sources": all(item["all_checks_passed"] for item in (capture_audit, field_audit, transport_audit)),
        "dimensions": payload["dimensions"] == list(range(2560)),
        "effects": len(payload["effect_rows"]) == 284 and all(len(row["values"]) == 2560 for row in payload["effect_rows"]),
        "raw": len(payload["raw_rows"]) == 160 and all(len(row["values"]) == 2560 for row in payload["raw_rows"]),
        "fresh_embedding_hidden": {row["state_kind"] for row in payload["raw_rows"] if row["dataset"] == "C110_fresh"} == {"embedding", "hidden_state"},
        "fresh_fields": all(row["prediction_passed"] for row in field["results"]),
        "transport_prediction": transport["prediction"]["attribute_prediction_passed"] and transport["prediction"]["agent_prediction_passed"],
        "pair_contrasts": len(pair_contrasts) == 8 and all(row["pairs"] == 24 for row in pair_contrasts),
        "no_extra_multi_flips": closure["headline"]["additional_multi_role_truth_flips"] == 0,
        "identity": core.sha(canonical) == core.sha(PUBLIC),
        "boundary": "does not identify one-to-one" in payload["claim_boundary"] and "minimal role coalition" in payload["claim_boundary"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "producer_sha256": core.sha(Path(__file__)),
        "asset_sha256": core.sha(canonical),
        "authorization": "audit_frontend_build_append_c110_memo_and_close_major_stage",
    }
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"checks": checks, "headline": closure["headline"], "new_puzzles": closure["new_puzzles"], "heatmap": closure["heatmap"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
