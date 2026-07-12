#!/usr/bin/env python3
"""Attach the frozen label-blind discovery/calibration registry to 288 dynamic bundles."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase365_dynamic_flow_instrumentation import validate_blind_bundle  # noqa: E402


EXECUTION = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time/private/phase362_execution_cases.jsonl"
PROTOCOL = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/blind_motif_protocol"
BUNDLES = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/dynamic_bundle_extraction"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    execution_rows = read_jsonl(EXECUTION)
    discovery_rows = [row for row in execution_rows if row["phase362_split"] == "independent_calibration"]
    sealed_case_ids = {
        row["blind_case_id"] for row in execution_rows if row["phase362_split"] == "physical_confirmation_sealed"
    }
    execution_by_case = {row["blind_case_id"]: row for row in discovery_rows}
    private_groups = read_jsonl(PROTOCOL / "private" / "phase366_group_label_key.jsonl")
    group_by_source = {(row["model"], row["source_group_id"]): row for row in private_groups}

    group_cases: dict[str, list[str]] = defaultdict(list)
    group_slots: dict[str, set[str]] = defaultdict(set)
    split_groups: dict[str, set[str]] = defaultdict(set)
    model_split_cases: Counter[tuple[str, str]] = Counter()
    bundle_hashes = []
    valid_count = 0
    bundle_case_ids = set()

    for model in MODELS:
        paths = sorted((BUNDLES / "blind_bundles" / model).glob("*.json"))
        if len(paths) != 96:
            raise RuntimeError(f"Expected 96 bundles for {model}, got {len(paths)}")
        for path in paths:
            bundle = read_json(path)
            case_id = bundle["anonymous_case_id"]
            source = execution_by_case.get(case_id)
            if source is None or source["model"] != model:
                raise RuntimeError(f"Bundle is absent from frozen independent execution set: {case_id}")
            group = group_by_source[(model, source["phase362_group_id"])]
            split = {
                "blind_motif_discovery": "blind_discovery",
                "blind_motif_calibration": "blind_calibration",
            }[group["scientific_split"]]
            bundle["schema_version"] = "43.1.0"
            bundle["anonymous_group_id"] = group["anonymous_group_id"]
            bundle["split"] = split
            bundle["split_registry_version"] = "Phase366-frozen-v1"
            errors = validate_blind_bundle(bundle)
            if errors:
                raise RuntimeError(f"Invalid bundle {case_id}: {errors[:5]}")
            write_json(path, bundle)
            digest = sha256_file(path)
            bundle_hashes.append({
                "anonymous_case_id": case_id,
                "anonymous_model_id": bundle["anonymous_model_id"],
                "anonymous_group_id": bundle["anonymous_group_id"],
                "split": split,
                "relative_path": str(path.relative_to(BUNDLES)),
                "sha256": digest,
            })
            bundle_case_ids.add(case_id)
            group_cases[group["anonymous_group_id"]].append(case_id)
            group_slots[group["anonymous_group_id"]].add(bundle["anonymous_condition_slot"])
            split_groups[split].add(group["anonymous_group_id"])
            model_split_cases[(model, split)] += 1
            valid_count += 1

    group_counts = Counter(len(cases) for cases in group_cases.values())
    slot_counts = Counter(len(slots) for slots in group_slots.values())
    overlap = len(bundle_case_ids & sealed_case_ids)
    summary = {
        "schema_version": "43.1.0",
        "phase_id": "Phase366-A",
        "created_at": now(),
        "objective": "freeze_group_independent_label_blind_discovery_and_calibration_splits",
        "denominator": {
            "model_count": len(MODELS),
            "bundle_count": len(bundle_hashes),
            "independent_group_count": len(group_cases),
            "condition_count_per_group": 4,
            "blind_discovery_group_count": len(split_groups["blind_discovery"]),
            "blind_calibration_group_count": len(split_groups["blind_calibration"]),
            "blind_discovery_case_count": sum(value for (model, split), value in model_split_cases.items() if split == "blind_discovery"),
            "blind_calibration_case_count": sum(value for (model, split), value in model_split_cases.items() if split == "blind_calibration"),
            "physical_confirmation_case_count": len(sealed_case_ids),
        },
        "results": {
            "valid_bundle_count": valid_count,
            "group_case_count_histogram": {str(key): value for key, value in sorted(group_counts.items())},
            "group_condition_slot_count_histogram": {str(key): value for key, value in sorted(slot_counts.items())},
            "all_groups_have_four_cases": group_counts == Counter({4: 72}),
            "all_groups_have_four_condition_slots": slot_counts == Counter({4: 72}),
            "physical_confirmation_overlap_count": overlap,
            "semantic_or_target_fields_added_to_bundle": False,
            "condition_semantics_revealed": False,
        },
        "model_split_case_counts": [
            {"model": model, "split": split, "case_count": model_split_cases[(model, split)]}
            for model in MODELS for split in ("blind_discovery", "blind_calibration")
        ],
        "authorization": {
            "label_blind_descriptor_extraction_authorized": (
                len(bundle_hashes) == 288 and valid_count == 288 and overlap == 0
                and group_counts == Counter({4: 72}) and slot_counts == Counter({4: 72})
            ),
            "semantic_label_reveal_authorized": False,
            "physical_confirmation_authorized": False,
            "causal_intervention_authorized": False,
        },
        "claim_boundary": {
            "bundle_split_frozen": True,
            "motif_scoring_executed": False,
            "language_path_discovered": False,
        },
        "private_bundle_hash_manifest": "private/phase366_bundle_hash_manifest.json",
        "next_decision": "extract_label_blind_directed_event_descriptors_then_freeze_noise_thresholds",
    }
    write_json(BUNDLES / "private" / "phase366_bundle_hash_manifest.json", bundle_hashes)
    write_json(BUNDLES / "phase366_bundle_split_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
