#!/usr/bin/env python3
"""Freeze decision-aligned physical cases after the Phase378 behavior gate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase376_decision_time_alignment_audit import first_target_step  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase378_physical_confirmation"
PROTOCOL = OUT / "phase378_physical_protocol.json"
BEHAVIOR = OUT / "phase378_behavior/models"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    selected = []
    distributions: dict[str, Counter[str]] = defaultdict(Counter)
    common: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    behavior_paths = {}
    for model in MODELS:
        path = BEHAVIOR / model / "private/phase378_behavior_rows.jsonl"
        behavior_paths[model] = path
        rows = read_jsonl(path)
        if len(rows) != 32:
            raise RuntimeError(f"Expected 32 physical behavior rows for {model}")
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
        for row in rows:
            if row["strict_behavior_correct"]:
                common[(row["mechanism_id"], row["semantic_group_id"])][model].add(
                    row["contrast_condition"].split("_", 1)[0]
                )
            step = first_target_step(
                tokenizer, row["generated_token_ids"], row["target_aliases"]
            )
            if step is None:
                raise RuntimeError(f"No physical decision step: {row['blind_case_id']}")
            distributions[model][str(step)] += 1
            selected.append({**row, "target_decision_step": step})
    qualified = {
        key
        for key, values in common.items()
        if set(values) == set(MODELS)
        and all(conditions == {"A", "B", "C", "D"} for conditions in values.values())
    }
    counts = Counter(mechanism for mechanism, _group in qualified)
    valid = (
        len(selected) == 96
        and all(row["strict_behavior_correct"] for row in selected)
        and counts == {"relation_binding": 4, "entity_recency": 4}
    )
    if not valid:
        raise RuntimeError(f"Physical behavior gate failed: {counts}")
    private = OUT / "private/phase378_physical_intervention_cases.jsonl"
    private.parent.mkdir(parents=True, exist_ok=True)
    with private.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "51.2.0",
        "phase_id": "Phase378-PhysicalBehaviorAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "denominator": {
            "case_count": len(selected),
            "case_count_per_model": 32,
            "common_group_counts": dict(sorted(counts.items())),
            "strict_correct_case_count": sum(
                row["strict_behavior_correct"] for row in selected
            ),
        },
        "decision_step_distributions": {
            model: dict(values) for model, values in distributions.items()
        },
        "quality": {
            "failed_groups_replaced": False,
            "other_mechanisms_opened": False,
            "private_case_hash": sha256(private),
            "behavior_hashes": {
                model: sha256(path) for model, path in behavior_paths.items()
            },
        },
        "authorization": {
            "run_physical_interventions": valid,
            "change_templates_or_gates": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    path = OUT / "phase378_physical_behavior_analysis_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
