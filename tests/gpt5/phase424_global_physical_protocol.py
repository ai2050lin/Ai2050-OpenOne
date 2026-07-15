#!/usr/bin/env python3
"""Freeze the Phase424 nine-family physical-path denominator.

The protocol reuses the frozen Phase330 semantic census without changing its
72-mechanism denominator.  It pairs distinct targets inside each mechanism so
the physical collector can compare source formation, legal attention transport
and query competition without fitting a semantic probe during collection.

Phase330 items have already been inspected in an earlier phase.  The final
split is therefore named ``legacy_physical_holdout`` and is never presented as
a new double-blind holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase330_nine_family_case_bank import FAMILY_MECHANISMS  # noqa: E402


PHASE_ID = "Phase424-GlobalPhysicalPathProtocol"
SCHEMA_VERSION = "phase424_global_physical_path.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SOURCE = (
    ROOT
    / "tests/gpt5/result/phase330_nine_family_global_atlas"
    / "nine_family_global_atlas/phase330_case_bank.jsonl"
)
OUT = ROOT / "tests/gpt5/result/phase424_global_physical_path_atlas"
PAIR_COUNTS = {
    "discovery": 6,
    "calibration": 2,
    "behavior_holdout": 2,
    "legacy_physical_holdout": 2,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def digest_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"Phase424 requires a fast tokenizer for {model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def continuation_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(value) for value in tokenizer(" " + text, add_special_tokens=False)["input_ids"]]


def branch_contract(tokenizer: Any, left: str, right: str) -> tuple[list[int], int, int]:
    left_ids = continuation_ids(tokenizer, left)
    right_ids = continuation_ids(tokenizer, right)
    if not left_ids or not right_ids or left_ids == right_ids:
        raise ValueError("identical_or_empty_branch")
    common = 0
    while (
        common < len(left_ids)
        and common < len(right_ids)
        and left_ids[common] == right_ids[common]
    ):
        common += 1
    if common >= len(left_ids) or common >= len(right_ids):
        raise ValueError("prefix_branch")
    return left_ids[:common], int(left_ids[common]), int(right_ids[common])


def registered_opposite(
    source: dict[str, Any],
    preferred: str,
    tokenizers: dict[str, Any],
) -> str | None:
    candidates = [preferred, *source["distractors"]]
    for candidate in dict.fromkeys(str(value) for value in candidates):
        if candidate.strip().lower() == source["target"].strip().lower():
            continue
        try:
            for tokenizer in tokenizers.values():
                branch_contract(tokenizer, source["target"], candidate)
        except ValueError:
            continue
        return candidate
    return None


def pair_eligible(left: dict[str, Any], right: dict[str, Any], tokenizers: dict[str, Any]) -> bool:
    return bool(
        registered_opposite(left, right["target"], tokenizers)
        and registered_opposite(right, left["target"], tokenizers)
    )


def perfect_pairs(
    rows: list[dict[str, Any]],
    tokenizers: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Find a deterministic perfect matching with distinct answer branches."""

    ordered = sorted(rows, key=lambda row: (int(row["item_index"]), row["case_id"]))

    def search(remaining: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
        if not remaining:
            return []
        left = remaining[0]
        candidates = [
            (index, row)
            for index, row in enumerate(remaining[1:], start=1)
            if pair_eligible(left, row, tokenizers)
        ]
        candidates.sort(
            key=lambda item: (
                left["target"].strip().lower() == item[1]["target"].strip().lower(),
                abs(int(item[1]["item_index"]) - int(left["item_index"])),
                int(item[1]["item_index"]),
            )
        )
        for index, right in candidates:
            tail = remaining[1:index] + remaining[index + 1 :]
            result = search(tail)
            if result is not None:
                return [(left, right), *result]
        return None

    output = search(ordered)
    if output is None:
        family = ordered[0]["family_id"] if ordered else "empty"
        mechanism = ordered[0]["mechanism_id"] if ordered else "empty"
        raise RuntimeError(f"No complete target matching for {family}/{mechanism}")
    return output


def find_char_span(prompt: str, fragment: str, label: str) -> tuple[int, int]:
    start = prompt.find(fragment)
    if start < 0:
        raise RuntimeError(f"Missing {label} fragment: {fragment!r}")
    return start, start + len(fragment)


def token_positions(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    start, end = span
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    if not positions:
        raise RuntimeError(f"Empty token interval for {span}")
    return positions


def register_condition(
    model: str,
    tokenizer: Any,
    pair: dict[str, Any],
    identity: str,
    source: dict[str, Any],
    opposite_target: str,
) -> dict[str, Any]:
    prompt = source["prompt"]
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    prefix, target_branch, opposite_branch = branch_contract(
        tokenizer, source["target"], opposite_target
    )
    source_positions = token_positions(
        offsets, find_char_span(prompt, source["context"], "source")
    )
    query_positions = token_positions(
        offsets, find_char_span(prompt, source["question"], "query")
    )
    control_positions = token_positions(
        offsets, find_char_span(prompt, source["instruction"], "instruction_control")
    )
    condition_id = f"{pair['pair_id']}__{identity}__{model}"
    executed_ids = [*prompt_ids, *prefix]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "condition_id": condition_id,
        "pair_id": pair["pair_id"],
        "pair_index": pair["pair_index"],
        "pair_identity": identity,
        "family_id": pair["family_id"],
        "mechanism_id": pair["mechanism_id"],
        "split": pair["split"],
        "legacy_exposed_split": pair["split"] == "legacy_physical_holdout",
        "item_id": source["item_id"],
        "case_id": source["case_id"],
        "item_index": int(source["item_index"]),
        "template_id": source["template_id"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "base_prompt_token_count": len(prompt_ids),
        "common_branch_prefix_token_ids": prefix,
        "executed_token_count": len(executed_ids),
        "executed_token_ids_sha256": digest_json(executed_ids),
        "prediction_position": len(executed_ids) - 1,
        "source_positions": source_positions,
        "query_positions": query_positions,
        "instruction_control_positions": control_positions,
        "source_token_count": len(source_positions),
        "query_token_count": len(query_positions),
        "instruction_control_token_count": len(control_positions),
        "target": source["target"],
        "opposite_target": opposite_target,
        "target_branch_token_id": target_branch,
        "opposite_branch_token_id": opposite_branch,
        "target_word_count": int(source["target_word_count"]),
        "target_absent_from_prompt": bool(source["target_absent_from_prompt"]),
        "negative_control": bool(source["negative_control"]),
        "open_set_control": bool(source["open_set_control"]),
        "physical": True,
        "observer_overlay": True,
        "predictive": False,
        "causal": False,
    }


def build_protocol() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = [
        row
        for row in read_jsonl(SOURCE)
        if row["template_id"] == "template_a"
    ]
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[(row["family_id"], row["mechanism_id"])].append(row)

    pair_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    pair_counter = 0
    for family, mechanisms in FAMILY_MECHANISMS.items():
        for mechanism in mechanisms:
            mechanism_rows = grouped[(family, mechanism)]
            by_index = {int(row["item_index"]): row for row in mechanism_rows}
            if len(by_index) != 24:
                raise RuntimeError(f"Expected 24 Phase330 items for {family}/{mechanism}")
            # Phase330's contiguous item ranges can contain only one answer
            # label (for example four consecutive category items).  Match all
            # 24 items first, then freeze disjoint pair-level splits.
            matched_all = perfect_pairs(list(by_index.values()), tokenizers)
            if len(matched_all) != 12:
                raise RuntimeError(f"Pair count mismatch for {family}/{mechanism}")
            pair_offset = 0
            for split, split_count in PAIR_COUNTS.items():
                matched = matched_all[pair_offset : pair_offset + split_count]
                pair_offset += split_count
                for local_index, (left, right) in enumerate(matched):
                    opposite_a = registered_opposite(left, right["target"], tokenizers)
                    opposite_b = registered_opposite(right, left["target"], tokenizers)
                    if opposite_a is None or opposite_b is None:
                        raise RuntimeError(f"Missing branch control for {family}/{mechanism}")
                    pair_id = (
                        f"phase424_{family}_{mechanism}_{split}_{local_index:02d}"
                    )
                    pair = {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "created_at": now(),
                        "pair_id": pair_id,
                        "pair_index": pair_counter,
                        "family_id": family,
                        "mechanism_id": mechanism,
                        "split": split,
                        "item_id_a": left["item_id"],
                        "item_id_b": right["item_id"],
                        "target_a": left["target"],
                        "target_b": right["target"],
                        "opposite_a": opposite_a,
                        "opposite_b": opposite_b,
                        "targets_distinct": left["target"] != right["target"],
                        "source_phase": 330,
                        "source_cases_previously_exposed": True,
                        "strict_double_blind_eligible": False,
                    }
                    pair_rows.append(pair)
                    for model, tokenizer in tokenizers.items():
                        condition_rows.append(
                            register_condition(model, tokenizer, pair, "a", left, opposite_a)
                        )
                        condition_rows.append(
                            register_condition(model, tokenizer, pair, "b", right, opposite_b)
                        )
                    pair_counter += 1

    counts = Counter((row["family_id"], row["mechanism_id"], row["split"]) for row in pair_rows)
    expected_mechanisms = {
        (family, mechanism)
        for family, mechanisms in FAMILY_MECHANISMS.items()
        for mechanism in mechanisms
    }
    observed_mechanisms = {(row["family_id"], row["mechanism_id"]) for row in pair_rows}
    validation = {
        "family_count": len({row["family_id"] for row in pair_rows}),
        "mechanism_count": len(observed_mechanisms),
        "pair_count": len(pair_rows),
        "condition_count": len(condition_rows),
        "conditions_per_model": dict(Counter(row["model"] for row in condition_rows)),
        "pairs_per_split": dict(Counter(row["split"] for row in pair_rows)),
        "pair_count_values_by_mechanism_split": sorted(set(counts.values())),
        "missing_mechanism_count": len(expected_mechanisms - observed_mechanisms),
        "duplicate_pair_id_count": len(pair_rows) - len({row["pair_id"] for row in pair_rows}),
        "duplicate_condition_id_count": len(condition_rows)
        - len({row["condition_id"] for row in condition_rows}),
        "empty_position_condition_count": sum(
            not row["source_positions"]
            or not row["query_positions"]
            or not row["instruction_control_positions"]
            for row in condition_rows
        ),
        "legacy_physical_holdout_pair_count": sum(
            row["split"] == "legacy_physical_holdout" for row in pair_rows
        ),
        "new_double_blind_pair_count": 0,
    }
    validation["valid"] = bool(
        validation["family_count"] == 9
        and validation["mechanism_count"] == 72
        and validation["pair_count"] == 864
        and validation["condition_count"] == 5184
        and set(validation["conditions_per_model"].values()) == {1728}
        and validation["pairs_per_split"]
        == {
            "discovery": 432,
            "calibration": 144,
            "behavior_holdout": 144,
            "legacy_physical_holdout": 144,
        }
        and validation["missing_mechanism_count"] == 0
        and validation["duplicate_pair_id_count"] == 0
        and validation["duplicate_condition_id_count"] == 0
        and validation["empty_position_condition_count"] == 0
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase": 424,
        "phase_id": PHASE_ID,
        "frozen_at": now(),
        "title": "Nine-family formation-transport-competition physical path census",
        "models_in_execution_order": list(MODELS),
        "execution_dtype_by_model": {
            "qwen3": "float16",
            "glm4": "bfloat16",
            "deepseek7b": "bfloat16",
        },
        "source_case_bank": str(SOURCE.relative_to(ROOT)),
        "source_case_bank_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "denominator": {
            "families": 9,
            "mechanisms": 72,
            "items_per_mechanism": 24,
            "pairs_per_mechanism": 12,
            "total_pairs": 864,
            "conditions_per_model": 1728,
            "total_model_conditions": 5184,
            "template": "template_a",
            "split_pair_counts_per_mechanism": PAIR_COUNTS,
        },
        "evidence_contract": {
            "compute_edges": [
                "source residual -> value state",
                "value state x actual attention probability -> output projection write",
                "attention write + MLP write -> layer output",
            ],
            "observer_overlays": [
                "target-versus-opposite unembedding direction",
                "physical-to-behavior frozen ridge audit",
            ],
            "causal_claim_allowed": False,
            "single_neuron_claim_allowed": False,
            "workspace_claim_allowed": False,
        },
        "frozen_features": {
            "formation": "paired source-state contrast minus paired instruction-control contrast",
            "transport": "paired legal source-write contrast minus paired instruction-control-write contrast",
            "competition": "signed source write and query residual alignment to the frozen target-versus-opposite readout direction",
            "depth_bins": ["early", "middle", "late"],
            "prediction_baseline": [
                "executed_token_count_mean",
                "source_token_count_mean",
                "query_token_count_mean",
                "control_token_count_mean",
                "target_word_count_mean",
                "target_leak_fraction",
            ],
            "prediction_physical": [
                "formation_specificity_median",
                "transport_contrast_specificity_median",
                "source_mass_specificity_median",
                "source_target_specificity_median",
                "query_target_alignment_median",
                "cancellation_index_median",
            ],
        },
        "frozen_gates": {
            "component_ledger_relative_error_max": 0.01,
            "replication_positive_fraction_min": 0.75,
            "replication_median_min": 0.0,
            "behavior_pair_correct_fraction_min_each_split": 0.75,
            "prediction_delta_r2_min_each_split": 0.05,
            "prediction_mae_gain_min_each_split": 0.0,
            "cross_model_topology_models_min": 2,
            "strict_double_blind_required_for_closure": True,
            "causal_intervention_required_for_closure": True,
        },
        "split_status": {
            "discovery": "development",
            "calibration": "development_holdout",
            "behavior_holdout": "behavior_holdout",
            "legacy_physical_holdout": "previously_exposed_replication_only",
            "strict_double_blind_holdout": "missing",
        },
        "phase_success_boundary": (
            "This phase may complete an observer-independent physical census and "
            "identify replicated path candidates. It cannot close any mechanism "
            "because the source bank is previously exposed and no intervention is run."
        ),
        "selection_updates_allowed": False,
        "validation": validation,
        "pair_rows_sha256": digest_json(pair_rows),
        "condition_rows_sha256": digest_json(condition_rows),
    }
    return protocol, pair_rows, condition_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-frozen", action="store_true")
    args = parser.parse_args()
    protocol_path = OUT / "phase424_protocol.json"
    pair_path = OUT / "phase424_registered_pairs.jsonl"
    condition_path = OUT / "phase424_registered_conditions.jsonl"
    if args.reuse_frozen and all(path.exists() for path in (protocol_path, pair_path, condition_path)):
        print(protocol_path)
        return
    protocol, pairs, conditions = build_protocol()
    write_json(protocol_path, protocol)
    write_jsonl(pair_path, pairs)
    write_jsonl(condition_path, conditions)
    if not protocol["validation"]["valid"]:
        raise SystemExit(json.dumps(protocol["validation"], ensure_ascii=False, indent=2))
    print(json.dumps(protocol["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
