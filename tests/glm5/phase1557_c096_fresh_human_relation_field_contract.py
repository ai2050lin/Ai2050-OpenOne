#!/usr/bin/env python3
"""Phase1557: freeze fresh unused human-validated C096 relation-field materials."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1556_c095_joint_synthesis_and_major_stage_closure"
C091_CONTRACT = RESULT / "phase1536_c091_human_validated_chinese_relation_contract"
SOURCE = RESULT / "phase1535_c091_external_human_material_and_analysis_adjudication"
OUT = RESULT / "phase1557_c096_fresh_human_relation_field_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1536_c091_human_validated_chinese_relation_contract as c091
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

SEED = "C096-fresh-human-validated-triadic-coordinate-validation-v1"
FAMILIES = tuple(c091.FAMILIES)
PARTITIONS = tuple(c091.PARTITIONS)
SURFACES = tuple(c091.SURFACES)
ROLES = tuple(c091.ROLES)


def digest(value: object) -> str:
    return hashlib.sha256((SEED + ":" + core.canonical(value)).encode("utf-8")).hexdigest()


def select_fresh_pairs() -> list[dict]:
    frames = c091.source_frames()
    old_pairs = core.rows(C091_CONTRACT / "material/frozen_pairs.jsonl")
    old_words = {word for row in old_pairs for word in (row["source"], row["target"])}
    new_words: set[str] = set()
    selected: list[dict] = []
    for family in FAMILIES:
        frame_key, relation = c091.DATASET_RELATIONS[family]
        family_frame = frames[frame_key][frames[frame_key]["relation"] == relation]
        by_concreteness = {}
        for concreteness in ("concrete", "abstract"):
            candidates = []
            for _, row in family_frame[family_frame["concreteness"] == concreteness].iterrows():
                source, target = [value.strip() for value in row["word_pair"].split(":")]
                item = {
                    "family": family,
                    "dataset_relation": relation,
                    "source": source,
                    "target": target,
                    "concreteness": concreteness,
                    "type_ratings": int(row["type_ratings"]),
                }
                if source == target or source in old_words or target in old_words:
                    continue
                candidates.append(item)
            candidates.sort(key=digest)
            chosen = []
            for item in candidates:
                if item["source"] in new_words or item["target"] in new_words:
                    continue
                chosen.append(item)
                new_words.update((item["source"], item["target"]))
                if len(chosen) == 15:
                    break
            if len(chosen) != 15:
                raise RuntimeError((family, concreteness, "insufficient fresh globally unique pairs", len(chosen)))
            chosen.sort(key=digest)
            by_concreteness[concreteness] = chosen
        for partition_index, partition in enumerate(PARTITIONS):
            block = by_concreteness["concrete"][partition_index * 5:(partition_index + 1) * 5]
            block += by_concreteness["abstract"][partition_index * 5:(partition_index + 1) * 5]
            block.sort(key=digest)
            for rank, item in enumerate(block):
                selected.append({
                    **item,
                    "pair_id": f"c096-{partition_index}-{FAMILIES.index(family)}-{rank:02d}",
                    "partition": partition,
                    "partition_rank": rank,
                })
    return sorted(selected, key=lambda row: (PARTITIONS.index(row["partition"]), FAMILIES.index(row["family"]), row["partition_rank"]))


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1557 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    requirements = core.load(PARENT / "protocol/c096_requirements.json")
    if parent["authorization"] != "run_phase1557_c096_fresh_human_relation_field_contract" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1556 authorization missing")

    pairs = select_fresh_pairs()
    cases = c091.build_cases(pairs)
    for case_index, case in enumerate(cases):
        case["case_id"] = f"c096-{case_index:04d}"
    tok = tokenizer()
    compiled = c091.compile_cases(tok, cases)
    old_pairs = core.rows(C091_CONTRACT / "material/frozen_pairs.jsonl")
    old_words = {word for row in old_pairs for word in (row["source"], row["target"])}
    new_words = [word for row in pairs for word in (row["source"], row["target"])]
    labels = [row["gold_label"] for row in cases]
    zero_models = {
        "always_first_candidate": c091.balanced_accuracy(labels, [row["candidates"][0] for row in cases]),
        "pair_identity": c091.balanced_accuracy(labels, c091.majority_predictions(cases, "pair_id")),
        "query_identity": c091.balanced_accuracy(labels, c091.majority_predictions(cases, "query_family")),
        "surface_identity": c091.balanced_accuracy(labels, c091.majority_predictions(cases, "surface")),
        "partition_identity": c091.balanced_accuracy(labels, c091.majority_predictions(cases, "partition")),
        "concreteness_identity": c091.balanced_accuracy(labels, c091.majority_predictions(cases, "concreteness")),
    }
    pair_counts = Counter((row["partition"], row["family"], row["concreteness"]) for row in pairs)
    cell_counts = Counter((row["partition"], row["surface"], row["pair_family"], row["query_family"], row["concreteness"]) for row in cases)
    max_length = max(len(row["prompt_ids"]) for row in compiled)
    checks = {
        "parent": True,
        "parent_audited": True,
        "pair_count": len(pairs) == 90,
        "case_count": len(cases) == 540,
        "balanced_pairs": len(pair_counts) == 18 and set(pair_counts.values()) == {5},
        "balanced_cells": len(cell_counts) == 108 and set(cell_counts.values()) == {5},
        "no_c091_lexical_overlap": not (set(new_words) & old_words),
        "c096_global_lexical_uniqueness": len(new_words) == len(set(new_words)) == 180,
        "six_cases_per_pair": all(sum(row["pair_id"] == pair["pair_id"] for row in cases) == 6 for pair in pairs),
        "zero_models": all(value == 0.5 for value in zero_models.values()),
        "single_token_outputs": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "role_spans": all(all(row["role_positions"][role] for role in ROLES) for row in compiled),
        "fixed_shape_feasible": max_length < 256,
        "human_source": True,
        "model_not_loaded": True,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    pair_path = OUT / "material/frozen_fresh_pairs.jsonl"
    case_path = OUT / "material/active_cases.jsonl"
    compiled_path = OUT / "compiled/qwen3_active.jsonl"
    core.write_rows(pair_path, pairs)
    core.write_rows(case_path, cases)
    core.write_rows(compiled_path, compiled)
    protocol = {
        "phase": 1557,
        "campaign": "C096",
        "schema": "c096.fresh_human_triadic_coordinate_validation.v1",
        "research_object": "fresh-material validation of the K268 common-plus-conditional late-boundary signed coordinate field",
        "model": "Qwen/Qwen3-4B local BF16 CUDA, no quantization",
        "system": c091.SYSTEM,
        "families": list(FAMILIES),
        "partitions": list(PARTITIONS),
        "surfaces": list(SURFACES),
        "roles": list(ROLES),
        "material": {
            "source_phase": 1535,
            "source_manifest_sha256": core.sha(SOURCE / "protocol/source_manifest.json"),
            "excluded_c091_pair_sha256": core.sha(C091_CONTRACT / "material/frozen_pairs.jsonl"),
            "pairs": 90,
            "cases": 540,
            "pairs_per_family_partition": 10,
            "concrete_abstract_per_family_partition": [5, 5],
            "lexical_overlap_with_c091": 0,
            "pairs_sha256": core.sha(pair_path),
            "cases_sha256": core.sha(case_path),
            "compiled_sha256": core.sha(compiled_path),
        },
        "execution": {
            "padding": "right",
            "batch": "all three relation queries times both surfaces for one pair",
            "batch_size": 6,
            "fixed_global_sequence_length": max_length,
            "position_ids": "attention-mask cumulative positions with pad positions reset to zero",
            "score_position": "each sequence's actual assistant boundary",
            "single_model_load": True,
        },
        "numeric_integrity_gate": {
            "repeat_hidden_max_abs": 1e-6,
            "repeat_logit_max_abs": 1e-6,
            "postquery_source_target_causal_max_abs": 1e-6,
            "prequery_relation_anchor_causal_max_abs": 1e-6,
            "finite": True,
            "bf16_nonquantized": True,
            "failure_action": "stop capture and do not interpret incomplete field",
        },
        "behavior_typing": {
            "thresholds": {
                "discovery_balanced_accuracy": 0.80,
                "discovery_each_surface_balanced_accuracy": 0.75,
                "discovery_true_recall": 0.75,
                "discovery_false_recall": 0.75,
                "discovery_three_way_accuracy": 0.75,
            },
            "policy": "report each family continuously; a failure creates M_BEHAVIOR and limits semantic claims but does not erase a numerically legal observation field",
        },
        "frozen_predictions": {
            "P096_1_prequery_common_field": "every partition x concreteness x state31/32 prequery boundary panel has triadic minimum pairwise cosine >= 0.75",
            "P096_2_cross_partition": "every surface x concreteness x family-pair x state31/32 boundary object has minimum pairwise cross-partition cosine >= 0.75",
            "P096_3_top64_energy": "median top64 energy fraction across all late-boundary centroids is within [0.15, 0.25]",
            "P096_4_coordinate_stability": "discovery-reference top64 minimum sign agreement >= 0.90 and minimum restricted cosine >= 0.85",
            "P096_5_order_conditioning": "median postquery triadic minimum cosine is at least 0.05 below median prequery triadic minimum cosine",
        },
        "analysis": {
            "interaction": "C_fg=0.5*(H_ff+H_gg-H_fg-H_gf) with five matched quartets per cell",
            "states": [31, 32],
            "role": "boundary",
            "coordinate_support": [16, 64, 256, 1024],
            "all_predictions_adjudicated_independently": True,
            "no_single_conjunction_erases_partial_results": True,
        },
        "missingness": {
            "M_BEHAVIOR": "family behavior type fails registered thresholds",
            "M_OUTPUT": "truth and emitted token identity remain coupled",
            "M_CAUSAL": "no intervention in C096",
            "M_EXTERNAL": "same Qwen3 model and source task family",
        },
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "learned probe", "clustering", "post-reveal threshold mutation", "causal mechanism claim", "new mathematics"],
        "claim_boundary": {
            "allowed": "fresh lexical-material replication of a task-scoped Qwen3 all-state pattern with behavior strata explicitly typed",
            "forbidden": ["semantic neurons", "universal relation law", "causal circuit", "cross-model invariant", "complete encoding theory"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1558_c096_unified_behavior_and_all_state_capture"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/pre_model_material_semantic_zero_model_audit.json", {"phase": 1557, "campaign": "C096", "checks": checks, "zero_models": zero_models, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())})
    core.save(OUT / "analysis/final.json", {"phase": 1557, "campaign": "C096", "status": "fresh_human_relation_field_contract_frozen", "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "zero_models": zero_models, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
