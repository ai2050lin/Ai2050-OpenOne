#!/usr/bin/env python3
"""Descriptive, non-authorizing failure atlas for the frozen Phase1246 run."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result/phase1246_c001_wp01_typed_behavior_qualification"
MATERIAL = RESULT / "material/frozen_typed_worlds.jsonl"
RAW = RESULT / "behavior/qwen3/raw_behavior.jsonl"
ADJUDICATION = RESULT / "analysis/typed_adjudication.json"
OUT = RESULT / "analysis/descriptive_failure_atlas.json"
PROTOCOLS = ("bare_short", "prompted_short", "fixed_sentence", "natural_sentence")
PARTITIONS = ("calibration", "discovery", "selection", "confirmation")
COLLISIONS = (
    "target_change", "nontarget_noop", "query_switch", "same_bag_binding_swap",
    "order_invariance", "template_invariance",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def output_category(result: dict[str, Any], material: dict[str, Any], protocol: str) -> str:
    generation = result["generations"][protocol]
    parsed = generation["parse"]
    text = normalize(generation["text"])
    if not generation["model_stopped"]:
        return "budget_exhausted"
    if parsed["content_correct"] and parsed["format_valid"]:
        return "correct_content_correct_format"
    if parsed["content_correct"]:
        return "correct_content_wrong_format"
    if parsed["prediction"] is not None:
        return "wrong_canonical_candidate"
    numeric = re.findall(r"(?<!\d)\d+(?!\d)", text)
    if material["gold_code"] in numeric:
        return "gold_numeric_code_not_name"
    if set(numeric) & set(material["candidate_codes"]):
        return "wrong_numeric_code"
    if any(lane in text.split() for lane in material["lanes"]):
        return "lane_field_intrusion"
    if any(seal in text.split() for seal in material["seals"]):
        return "seal_field_intrusion"
    if any(obj.lower() in text for obj in material["objects"]):
        return "object_name_intrusion"
    if not text:
        return "empty"
    return "other"


def main() -> None:
    material_rows = read_jsonl(MATERIAL)
    raw = read_jsonl(RAW)
    adjudication = read_json(ADJUDICATION)
    material = {row["row_id"]: row for row in material_rows}
    candidate_partition = {
        partition: mean(row["candidate_correct"] for row in raw if row["partition"] == partition)
        for partition in PARTITIONS
    }
    candidate_by_code = {
        code: mean(row["candidate_correct"] for row in raw if material[row["row_id"]]["gold_code"] == code)
        for code in sorted({row["gold_code"] for row in material_rows}, key=int)
    }
    candidate_by_template = {
        str(template): mean(row["candidate_correct"] for row in raw if row["template_index"] == template)
        for template in range(4)
    }
    candidate_by_collision = {
        collision: mean(row["candidate_correct"] for row in raw if row["collision_group"] == collision)
        for collision in COLLISIONS
    }
    candidate_pair_complete = {}
    for collision in COLLISIONS:
        worlds = sorted({row["world_id"] for row in raw if row["collision_group"] == collision})
        candidate_pair_complete[collision] = mean(
            all(row["candidate_correct"] for row in raw if row["world_id"] == world)
            for world in worlds
        )
    categories: dict[str, dict[str, int]] = {}
    top_outputs: dict[str, list[dict[str, Any]]] = {}
    protocol_partition_content: dict[str, Any] = {}
    protocol_partition_stop: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        counts = Counter(output_category(row, material[row["row_id"]], protocol) for row in raw)
        categories[protocol] = dict(sorted(counts.items()))
        outputs = Counter(normalize(row["generations"][protocol]["text"]) for row in raw)
        top_outputs[protocol] = [{"text": text, "count": count} for text, count in outputs.most_common(20)]
        protocol_partition_content[protocol] = {
            partition: mean(row["generations"][protocol]["parse"]["content_correct"] for row in raw if row["partition"] == partition)
            for partition in PARTITIONS
        }
        protocol_partition_stop[protocol] = {
            partition: mean(row["generations"][protocol]["model_stopped"] for row in raw if row["partition"] == partition)
            for partition in PARTITIONS
        }
    candidate_errors = [row for row in raw if not row["candidate_correct"]]
    candidate_error_confusions = Counter(
        f"{row['gold']}->{row['candidate']['prediction']}" for row in candidate_errors
    )
    cache_rows = [row for row in raw if row["cache_full_recompute"] is not None]
    cache_mismatch_rows = [row for row in cache_rows if row["cache_full_recompute"]["agreement"] != 1.0]
    cache_mismatch_steps = sum(
        row["cache_full_recompute"]["step_count"] - row["cache_full_recompute"]["match_count"]
        for row in cache_rows
    )
    value: dict[str, Any] = {
        "phase": 1246,
        "schema_version": "phase1246.descriptive_failure_atlas.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_raw_digest": digest(raw),
        "source_adjudication_digest": adjudication["adjudication_digest"],
        "candidate_selection": {
            "partition_accuracy": candidate_partition,
            "by_numeric_code": candidate_by_code,
            "by_template": candidate_by_template,
            "by_collision": candidate_by_collision,
            "pair_complete_by_collision": candidate_pair_complete,
            "error_count": len(candidate_errors),
            "top_error_confusions": [
                {"confusion": label, "count": count} for label, count in candidate_error_confusions.most_common(20)
            ],
        },
        "generation": {
            "category_counts": categories,
            "partition_content_accuracy": protocol_partition_content,
            "partition_model_stop_rate": protocol_partition_stop,
            "top_outputs": top_outputs,
        },
        "cache_recompute": {
            "trajectory_count": len(cache_rows),
            "mismatch_trajectory_count": len(cache_mismatch_rows),
            "total_step_count": sum(row["cache_full_recompute"]["step_count"] for row in cache_rows),
            "mismatch_step_count": cache_mismatch_steps,
            "mismatch_world_ids": [row["world_id"] for row in cache_mismatch_rows],
        },
        "interpretation_boundary": [
            "This is a post-adjudication descriptive decomposition and cannot change any typed gate.",
            "Output categories are surface error labels, not neural mechanism classes.",
            "Candidate-only strength does not authorize hidden observation because the frozen G-CONTENT conjunct failed.",
        ],
    }
    value["atlas_digest"] = digest(value)
    OUT.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical_json({"status": "phase1246_failure_atlas_complete", "digest": value["atlas_digest"], "candidate_errors": len(candidate_errors), "cache_mismatch_steps": cache_mismatch_steps}))


if __name__ == "__main__":
    main()
