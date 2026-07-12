#!/usr/bin/env python3
"""Extract label-blind raw-vector relation features from Phase369 flow bundles."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
BUNDLES = BASE / "dynamic_bundle_extraction"
COLLECTION_FREEZE = BASE / "raw_collection_freeze/private/phase369_collection_execution_cases.jsonl"
OUT = BASE / "raw_relation_features"
MODELS = ("qwen3", "glm4", "deepseek7b")
LOW_FEATURES = (
    "route_norm_over_attention",
    "route_alignment_to_attention",
    "route_signed_projection_to_attention",
    "route_alignment_to_output_change",
    "attention_norm_over_input",
    "mlp_norm_over_post_attention",
    "attention_alignment_to_mlp",
    "attention_mlp_balance",
    "output_change_norm_over_input",
    "input_alignment_to_output",
)
VECTOR_NAMES = ("route", "layer_input", "attention", "post_attention", "mlp", "output_change")
GRAM_PAIRS = tuple(
    (left, right)
    for left in range(len(VECTOR_NAMES))
    for right in range(left + 1, len(VECTOR_NAMES))
)
RAW_FEATURES = tuple(f"cos_{VECTOR_NAMES[left]}__{VECTOR_NAMES[right]}" for left, right in GRAM_PAIRS) + tuple(
    f"norm_share_{name}" for name in VECTOR_NAMES
)
VOCAB_FEATURES = ("top1_probability", "top2_probability", "top1_top2_gap", "top5_mass", "normalized_entropy")


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
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float((torch.dot(left, right) / denominator).item())


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(abs(denominator), 1e-12)


def resolve(reference: dict[str, Any], cache: dict[str, Any]) -> torch.Tensor:
    relative = reference["relative_path"]
    if relative not in cache:
        cache[relative] = torch.load(BASE / relative, map_location="cpu", weights_only=True)
    value: Any = cache[relative]
    for part in reference["slice"]:
        value = value[part]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Reference did not resolve to tensor: {relative}")
    return value.float().reshape(-1)


def low_and_raw_features(
    route_event: dict[str, Any],
    event_by_id: dict[str, dict[str, Any]],
    cache: dict[str, Any],
) -> tuple[list[float], list[float]]:
    generation_time = int(route_event["generation_time"])
    layer_index = int(route_event["layer_index"])
    receiver = route_event["receiver_role"]
    prefix = f"t{generation_time}:l{layer_index}:r{receiver}:"
    route = resolve(route_event["vector_ref"], cache)
    layer_input = resolve(event_by_id[prefix + "input"]["vector_ref"], cache)
    attention = resolve(event_by_id[prefix + "attention_merge"]["vector_ref"], cache)
    post_attention = resolve(event_by_id[prefix + "post_attention"]["vector_ref"], cache)
    mlp = resolve(event_by_id[prefix + "mlp"]["vector_ref"], cache)
    output = resolve(event_by_id[prefix + "output"]["vector_ref"], cache)
    output_change = output - layer_input
    vectors = (route, layer_input, attention, post_attention, mlp, output_change)
    norms = [norm(value) for value in vectors]
    route_norm, input_norm, attention_norm, post_norm, mlp_norm, output_change_norm = norms
    low = [
        safe_ratio(route_norm, attention_norm),
        cosine(route, attention),
        safe_ratio(float(torch.dot(route, attention).item()), attention_norm * attention_norm),
        cosine(route, output_change),
        safe_ratio(attention_norm, input_norm),
        safe_ratio(mlp_norm, post_norm),
        cosine(attention, mlp),
        safe_ratio(attention_norm - mlp_norm, attention_norm + mlp_norm),
        safe_ratio(output_change_norm, input_norm),
        cosine(layer_input, output),
    ]
    total_norm = max(sum(norms), 1e-12)
    raw = [cosine(vectors[left], vectors[right]) for left, right in GRAM_PAIRS]
    raw.extend(value / total_norm for value in norms)
    if not all(math.isfinite(value) for value in [*low, *raw]):
        raise RuntimeError(f"Non-finite raw relation at {route_event['event_id']}")
    return low, raw


def vocab_features(logits: torch.Tensor) -> list[float]:
    probabilities = torch.softmax(logits.float().reshape(-1), dim=0)
    values = torch.topk(probabilities, k=min(5, probabilities.numel())).values
    entropy = float((-(probabilities * probabilities.clamp_min(1e-30).log()).sum()).item())
    normalized_entropy = entropy / max(math.log(probabilities.numel()), 1e-12)
    top1 = float(values[0].item())
    top2 = float(values[1].item()) if values.numel() > 1 else 0.0
    return [top1, top2, top1 - top2, float(values.sum().item()), normalized_entropy]


def main() -> None:
    case_contract = {row["blind_case_id"]: row for row in read_jsonl(COLLECTION_FREEZE)}
    model_rows = []
    total_route_rows = 0
    for model in MODELS:
        bundle_paths = sorted((BUNDLES / "blind_bundles" / model).glob("*.json"))
        output_rows = []
        for case_index, bundle_path in enumerate(bundle_paths, 1):
            bundle = read_json(bundle_path)
            case_id = bundle["anonymous_case_id"]
            contract = case_contract[case_id]
            event_by_id = {event["event_id"]: event for event in bundle["events"]}
            route_events = [event for event in bundle["events"] if event["event_type"] == "attention_source_write"]
            cache: dict[str, Any] = {}
            records = []
            low_rows = []
            raw_rows = []
            for route_event in route_events:
                low, raw = low_and_raw_features(route_event, event_by_id, cache)
                records.append({
                    "generation_time": int(route_event["generation_time"]),
                    "layer_index": int(route_event["layer_index"]),
                    "relative_depth": int(route_event["layer_index"]) / max(
                        max(event["layer_index"] for event in bundle["events"] if event["layer_index"] >= 0) - 1,
                        1,
                    ),
                    "source_role": route_event["source_role"],
                    "receiver_role": route_event["receiver_role"],
                    "route_event_id": route_event["event_id"],
                })
                low_rows.append(low)
                raw_rows.append(raw)
            vocab_rows = []
            for generation_time in range(3):
                event = event_by_id[f"t{generation_time}:vocab"]
                vocab_rows.append(vocab_features(resolve(event["vector_ref"], cache)))
            payload = {
                "schema_version": "46.1.0",
                "phase_id": "Phase369",
                "anonymous_case_id": case_id,
                "anonymous_model_id": bundle["anonymous_model_id"],
                "anonymous_group_id": bundle["anonymous_group_id"],
                "anonymous_parallel_group_id": contract["anonymous_parallel_group_id"],
                "parallel_prompt_hash": hashlib.sha256(contract["raw_prompt"].encode()).hexdigest(),
                "records": records,
                "low_descriptor_features": torch.tensor(low_rows, dtype=torch.float32),
                "raw_relation_features": torch.tensor(raw_rows, dtype=torch.float32),
                "vocab_state_features": torch.tensor(vocab_rows, dtype=torch.float32),
                "raw_vector_bundle_reference": str(bundle_path.relative_to(BASE)),
                "semantic_labels_used": False,
                "target_rank_or_margin_used": False,
            }
            output_path = OUT / "private/cases" / model / f"{case_id}.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, output_path)
            output_rows.append({
                "anonymous_case_id": case_id,
                "anonymous_group_id": bundle["anonymous_group_id"],
                "anonymous_parallel_group_id": contract["anonymous_parallel_group_id"],
                "route_row_count": len(records),
                "relative_path": str(output_path.relative_to(OUT)),
                "byte_count": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            })
            total_route_rows += len(records)
            if case_index % 8 == 0 or case_index == len(bundle_paths):
                print(f"[{model}] raw relation cases {case_index}/{len(bundle_paths)}", flush=True)
        manifest = {
            "schema_version": "46.1.0",
            "phase_id": "Phase369",
            "created_at": now(),
            "model": model,
            "case_count": len(output_rows),
            "route_row_count": sum(row["route_row_count"] for row in output_rows),
            "case_rows": output_rows,
            "valid": len(output_rows) == 112,
        }
        write_json(OUT / "models" / model / "manifest.json", manifest)
        model_rows.append({key: value for key, value in manifest.items() if key != "case_rows"})
    schema = {
        "schema_version": "46.1.0",
        "phase_id": "Phase369",
        "low_descriptor_feature_names": list(LOW_FEATURES),
        "raw_relation_feature_names": list(RAW_FEATURES),
        "vocab_state_feature_names": list(VOCAB_FEATURES),
        "raw_relation_formula": "K_ij=dot(v_i,v_j)/(norm(v_i)*norm(v_j)); r_i=norm(v_i)/sum_j(norm(v_j))",
        "raw_vector_order": list(VECTOR_NAMES),
        "coordinate_invariant": True,
        "unrestricted_cross_model_rotation_fitted": False,
        "target_rank_or_margin_used": False,
    }
    summary = {
        "schema_version": "46.1.0",
        "phase_id": "Phase369",
        "created_at": now(),
        "objective": "derive_coordinate_invariant_relations_from_replayable_raw_vectors_on_the_same_routes_as_the_old_ten_descriptor_baseline",
        "denominator": {
            "model_count": 3,
            "case_count": sum(row["case_count"] for row in model_rows),
            "route_row_count": total_route_rows,
            "low_feature_count": len(LOW_FEATURES),
            "raw_relation_feature_count": len(RAW_FEATURES),
            "vocab_feature_count": len(VOCAB_FEATURES),
        },
        "results": {
            "all_case_files_valid": all(row["valid"] for row in model_rows),
            "raw_and_low_features_share_identical_route_rows": True,
            "raw_vectors_retained_by_bundle_reference": True,
            "semantic_or_target_label_used": False,
            "target_rank_or_margin_used": False,
            "calibration_or_physical_case_used": False,
        },
        "models": model_rows,
        "authorization": {
            "blind_future_prediction_comparison": True,
            "semantic_label_reveal": False,
            "calibration_collection": False,
            "physical_holdout": False,
        },
    }
    write_json(OUT / "phase369_raw_relation_schema.json", schema)
    write_json(OUT / "phase369_raw_relation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
