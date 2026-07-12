#!/usr/bin/env python3
"""Freeze 384 unseen prompts and nine multiresolution anchors for Phase362."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase362_generation_time_trace"
ROUND = "independent_generation_time"
P354 = ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace/qualified_contract_semantic_time"
P361_CONTRACT = ROOT / "tests/gpt5/result/phase361_contract_repair/seven_contract_repair"
P361_TRACE = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    ("content_knowledge", "relation_binding"),
    ("readout_competition", "target_vs_wrong"),
    ("state_drift", "entity_recency"),
    ("syntax_structure", "number_agreement"),
)
SCHEMA_VERSION = "39.0.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    original = read_jsonl(P354 / "phase354_registered_cases.jsonl")
    repaired = read_jsonl(P361_CONTRACT / "phase361_registered_cases.jsonl")
    phase361_used = {
        row["case_id"]
        for row in read_jsonl(P361_TRACE / "private" / "phase361_execution_cases.jsonl")
    }
    candidates_path = P361_TRACE / "phase361_frozen_predictive_candidates.jsonl"
    frozen_candidate_hash = file_sha256(candidates_path)
    execution_rows, blind_rows, private_labels = [], [], []
    tokenizers = {}
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizer = AutoTokenizer.from_pretrained(
                str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
                local_files_only=True, use_fast=False,
            )
            tokenizers[model] = tokenizer
            for family, mechanism in MECHANISMS:
                source = repaired if mechanism == "number_agreement" else original
                rows = [
                    row for row in source
                    if row["model"] == model and row["family_id"] == family
                    and row["mechanism_id"] == mechanism
                    and row["split"] in {"physical_discovery", "physical_calibration"}
                ]
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    grouped[row["contract_group_id"]].append(row)
                available = [
                    (group_id, values) for group_id, values in grouped.items()
                    if len(values) == 4 and not any(row["case_id"] in phase361_used for row in values)
                ]
                available.sort(key=lambda item: digest(f"phase362-group:{model}:{item[0]}"))
                if len(available) < 8:
                    raise RuntimeError(f"Only {len(available)} unseen groups for {model}/{mechanism}")
                for group_index, (group_id, values) in enumerate(available[:8]):
                    phase_split = "independent_calibration" if group_index < 6 else "physical_confirmation_sealed"
                    for row in sorted(values, key=lambda value: value["contrast_condition"]):
                        blind_case_id = f"p362_{digest('phase362-case:' + row['case_id'])[:24]}"
                        prompt_tokens = tokenizer(
                            row["prompt"], add_special_tokens=bool(row["tokenization_add_special_tokens"]),
                        )["input_ids"]
                        enriched = {
                            **row,
                            "blind_case_id": blind_case_id,
                            "anonymous_model_id": f"am_{digest('phase362-model:' + model)[:12]}",
                            "phase362_group_id": f"g_{digest('phase362-group-private:' + group_id)[:20]}",
                            "phase362_split": phase_split,
                            "prompt_token_count": len(prompt_tokens),
                        }
                        execution_rows.append(enriched)
                        blind_rows.append({
                            "schema_version": SCHEMA_VERSION, "phase_id": "Phase362", "created_at": now(),
                            "blind_case_id": blind_case_id,
                            "anonymous_model_id": enriched["anonymous_model_id"],
                            "anonymous_group_id": enriched["phase362_group_id"],
                            "split": phase_split,
                            "prompt_token_count": len(prompt_tokens),
                            "semantic_label_used_for_selection": False,
                        })
                        private_labels.append({
                            "blind_case_id": blind_case_id, "model": model,
                            "family_id": family, "mechanism_id": mechanism,
                            "source_case_id": row["case_id"], "source_group_id": group_id,
                            "contrast_condition": row["contrast_condition"],
                            "phase362_split": phase_split,
                        })
        calibration = [row for row in execution_rows if row["phase362_split"] == "independent_calibration"]
        anchors = []
        for model in MODELS:
            model_rows = [row for row in calibration if row["model"] == model]
            short = min(model_rows, key=lambda row: (row["prompt_token_count"], row["blind_case_id"]))
            long = max(model_rows, key=lambda row: (row["prompt_token_count"], row["blind_case_id"]))
            remaining = [row for row in model_rows if row["blind_case_id"] not in {short["blind_case_id"], long["blind_case_id"]}]
            multi = min(remaining, key=lambda row: digest("phase362-multistep:" + row["blind_case_id"]))
            for anchor_type, row, shard_count in (("short", short, 2), ("long", long, 4), ("multistep", multi, 16)):
                anchors.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": "Phase362", "created_at": now(),
                    "anchor_id": f"anchor_{model}_{anchor_type}",
                    "anchor_type": anchor_type, "model": model,
                    "blind_case_id": row["blind_case_id"],
                    "prompt_token_count": row["prompt_token_count"],
                    "generation_time_count": 3 if anchor_type == "multistep" else 1,
                    "mlp_co_shard_count": shard_count,
                    "selection_rule": "prompt_length_extreme_or_fixed_hash_without_semantic_label",
                })
        root = OUT / ROUND
        write_jsonl(root / "private" / "phase362_execution_cases.jsonl", execution_rows)
        write_jsonl(root / "private" / "phase362_label_key.jsonl", private_labels)
        write_jsonl(root / "private" / "phase362_anchor_registry.jsonl", anchors)
        write_jsonl(root / "phase362_blind_case_registry.jsonl", blind_rows)
        summary = {
            "schema_version": SCHEMA_VERSION, "phase_id": "Phase362", "created_at": now(),
            "denominator": {
                "model_count": 3, "mechanism_count": 4,
                "unseen_group_count_per_model_mechanism": 8,
                "condition_count_per_group": 4,
                "case_count": len(execution_rows),
                "independent_calibration_group_count": 3 * 4 * 6,
                "independent_calibration_case_count": len(calibration),
                "physical_confirmation_group_count": 3 * 4 * 2,
                "physical_confirmation_case_count": len(execution_rows) - len(calibration),
                "anchor_count": len(anchors),
            },
            "frozen_phase361_candidates": {
                "count": len(read_jsonl(candidates_path)),
                "sha256": frozen_candidate_hash,
            },
            "quality": {
                "phase361_case_overlap_count": sum(row["case_id"] in phase361_used for row in execution_rows),
                "semantic_label_used_for_selection": False,
                "physical_confirmation_sealed": True,
            },
            "entry_decision": "run_anchor_format_then_independent_generation_time_trace",
        }
        write_json(root / "phase362_case_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        tokenizers.clear()


if __name__ == "__main__":
    main()
