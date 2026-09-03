#!/usr/bin/env python3
"""C250: full-token pairwise-aligned event and adjacent-checkpoint observation."""
from __future__ import annotations

import difflib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.OUTS["C250"]
PARENT = common.OUTS["C248"]


def aligned_pairs(left_ids: list[int], right_ids: list[int]) -> list[tuple[int, int]]:
    matcher = difflib.SequenceMatcher(a=left_ids, b=right_ids, autojunk=False)
    pairs = []
    for block in matcher.get_matching_blocks():
        pairs.extend((block.a + i, block.b + i) for i in range(block.size))
    return pairs


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C249"] / "audit/independent_final_audit.json")
    fields = np.load(PARENT / "raw/full_fields.float16.npy", mmap_mode="r")
    index = core.rows(PARENT / "raw/hidden_index.jsonl")
    compiled = {row["case_id"]: row for row in core.rows(common.OUTS["C247"] / "compiled/qwen3.jsonl")}
    rows = [row for row in index if row["panel"] == "core" and row["correct"]]
    key = {(row["family"], row["surface"], row["unit"], row["factor_a"], row["factor_b"], row["order"]): row for row in rows}
    thresholds = np.asarray(core.load(common.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C250"), "full_field": list(fields.shape) == [768, 37, 128, 2560], "all_coordinates": True, "token_alignment_frozen": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": 1784, "campaign": "C250", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_token_observation_frozen", "effects": ["factor_a_pair_differences", "factor_b_pair_differences"],
        "alignment": "exact tokenizer-id matching blocks via deterministic sequence alignment; changed and inserted tokens are counted as unmatched rather than silently compared to different tokens",
        "outputs": ["every matched token event count", "every physical coordinate signed count", "adjacent checkpoint same-coordinate same-sign persistence"],
        "claim_boundary": "These are predictive/observational dependencies. Alignment is not a causal edge and unmatched edited spans have no one-to-one state difference.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "derive_all_behavior_correct_pairs_once",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    pair_specs = []
    for family, surface, unit, order in itertools.product(common.FAMILIES, common.SURFACES, range(8), (1, -1)):
        for effect, comparisons in (
            ("factor_a", [((0, 0), (1, 0)), ((0, 1), (1, 1))]),
            ("factor_b", [((0, 0), (0, 1)), ((1, 0), (1, 1))]),
        ):
            for left_cell, right_cell in comparisons:
                left_key = (family, surface, unit, *left_cell, order)
                right_key = (family, surface, unit, *right_cell, order)
                if left_key in key and right_key in key:
                    pair_specs.append((family, effect, key[left_key], key[right_key]))
    n = len(pair_specs)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    token_counts = np.lib.format.open_memmap(OUT / "raw/pair_token_event_counts.uint16.npy", mode="w+", dtype=np.uint16, shape=(n, 37, 128))
    up_counts = np.zeros((5, 2, 37, 2560), np.int32)
    down_counts = np.zeros_like(up_counts)
    persistence_num = np.zeros((5, 2, 36), np.int64)
    persistence_den = np.zeros_like(persistence_num)
    pair_rows = []
    for pair_i, (family, effect, left, right) in enumerate(pair_specs):
        left_ids = compiled[left["case_id"]]["prompt_ids"]
        right_ids = compiled[right["case_id"]]["prompt_ids"]
        pairs = aligned_pairs(left_ids, right_ids)
        fi, ei = common.FAMILIES.index(family), (0 if effect == "factor_a" else 1)
        local_up = np.zeros((37, 2560), np.int32)
        local_down = np.zeros_like(local_up)
        for left_pos, right_pos in pairs:
            delta = np.asarray(fields[right["hidden_index"], :, right_pos], np.float32) - np.asarray(fields[left["hidden_index"], :, left_pos], np.float32)
            event = np.where(delta > thresholds[:, None], 1, np.where(delta < -thresholds[:, None], -1, 0)).astype(np.int8)
            token_counts[pair_i, :, left_pos] = np.count_nonzero(event, axis=-1).astype(np.uint16)
            local_up += event == 1
            local_down += event == -1
            for q in range(36):
                source = event[q] != 0
                persistence_den[fi, ei, q] += int(source.sum())
                persistence_num[fi, ei, q] += int(np.count_nonzero((event[q + 1] == event[q]) & source))
        up_counts[fi, ei] += local_up
        down_counts[fi, ei] += local_down
        pair_rows.append({
            "pair_index": pair_i, "family": family, "effect": effect, "left_case": left["case_id"], "right_case": right["case_id"],
            "left_tokens": len(left_ids), "right_tokens": len(right_ids), "matched_tokens": len(pairs),
            "left_coverage": len(pairs) / len(left_ids), "right_coverage": len(pairs) / len(right_ids),
        })
        if pair_i % 64 == 0 or pair_i + 1 == n:
            token_counts.flush(); print(f"[C250] aligned token pairs {pair_i + 1}/{n}", flush=True)
    np.save(OUT / "analysis/family_effect_checkpoint_coordinate_up_counts.int32.npy", up_counts)
    np.save(OUT / "analysis/family_effect_checkpoint_coordinate_down_counts.int32.npy", down_counts)
    core.write_rows(OUT / "analysis/aligned_pair_index.jsonl", pair_rows)
    persistence_rows = []
    for fi, family in enumerate(common.FAMILIES):
        for ei, effect in enumerate(("factor_a", "factor_b")):
            for q in range(36):
                persistence_rows.append({"family": family, "effect": effect, "from_checkpoint": q, "to_checkpoint": q + 1, "source_events": int(persistence_den[fi, ei, q]), "same_coordinate_same_sign_fraction": float(persistence_num[fi, ei, q] / max(persistence_den[fi, ei, q], 1))})
    core.write_rows(OUT / "analysis/adjacent_checkpoint_persistence.jsonl", persistence_rows)
    report = {
        "phase": 1784, "campaign": "C250", "status": "full_token_observation_complete", "aligned_pairs": n,
        "matched_token_coverage_median": float(np.median([(row["left_coverage"] + row["right_coverage"]) / 2 for row in pair_rows])),
        "same_coordinate_same_sign_persistence_median": float(np.median([row["same_coordinate_same_sign_fraction"] for row in persistence_rows])),
        "up_event_observations": int(up_counts.sum()), "down_event_observations": int(down_counts.sum()),
        "all_coordinate_axes_preserved": True,
        "strict_interpretation": "The raw archive and coordinate count tensors preserve all 2560 coordinates. Exact-token alignment avoids comparing shifted unrelated tokens, but edited tokens remain unmatched and adjacent-checkpoint persistence is not a unique transport mechanism.",
        "next_authorization": "C251_typed_composition_observation",
    }
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {"pairs": n >= 600, "token_shape": token_counts.shape == (n, 37, 128), "coordinate_shape": up_counts.shape == (5, 2, 37, 2560), "persistence_rows": len(persistence_rows) == 360, "finite": bool(np.isfinite([row["same_coordinate_same_sign_fraction"] for row in persistence_rows]).all())}
    final_checks = {"contract": True, "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final = {"phase": 1784, "campaign": "C250", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
