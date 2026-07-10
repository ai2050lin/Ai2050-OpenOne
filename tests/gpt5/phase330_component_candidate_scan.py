#!/usr/bin/env python3
"""Freeze Phase330 layer/role paths and scan registered head/MLP candidates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase330_nine_family_case_bank import FAMILY_MECHANISMS, MODELS  # noqa: E402
import phase330_global_atlas_survey as survey  # noqa: E402
from phase326_distributed_carrier_atlas import get_down_proj, group_ranges  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402


PHASE = "Phase330"
SCHEMA_VERSION = "8.0.0"
ROUND_DEFAULT = "nine_family_global_atlas"
OUT = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", row_group_size=32768)


def path_registry(round_name: str) -> list[dict[str, Any]]:
    root = OUT / round_name
    result = []
    for model in MODELS:
        for family, mechanisms in FAMILY_MECHANISMS.items():
            signatures = read_jsonl(root / "survey" / model / family / "path_signatures.jsonl")
            discovery = [row for row in signatures if row["split"] == "discovery" and row["template_id"] in {"template_a", "template_b"}]
            calibration = [row for row in signatures if row["split"] == "calibration" and row["template_id"] in {"template_a", "template_b"}]
            for mechanism in mechanisms:
                for component in ("attention", "mlp", "residual"):
                    candidates: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
                    for row in discovery:
                        if row["mechanism_id"] == mechanism and row["component_type"] == component:
                            candidates[(int(row["peak_layer"]), row["position_role"])].append(row)
                    ranked = []
                    total = 24
                    for (layer, role), rows in candidates.items():
                        support = len(rows) / total
                        amplitude = mean(float(row["peak_absolute_projection"]) for row in rows)
                        persistence = mean(float(row["post_onset_persistence"]) for row in rows)
                        ranked.append((support * amplitude * persistence, layer, role, support, amplitude, persistence))
                    ranked.sort(reverse=True)
                    score, layer, role, support, amplitude, persistence = ranked[0]
                    held_calibration = [
                        row for row in calibration
                        if row["mechanism_id"] == mechanism and row["component_type"] == component
                        and int(row["peak_layer"]) == layer and row["position_role"] == role
                    ]
                    result.append({
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": model,
                        "family_id": family,
                        "mechanism_id": mechanism,
                        "component_type": component,
                        "component_layer": layer,
                        "position_role": role,
                        "discovery_support_rate": round(support, 7),
                        "discovery_mean_peak_amplitude": round(amplitude, 7),
                        "discovery_mean_persistence": round(persistence, 7),
                        "selection_score": round(score, 7),
                        "calibration_support_rate": round(len(held_calibration) / 12, 7),
                        "selection_split": "discovery_only",
                        "calibration_role": "audit_only_no_update",
                        "heldout_used": False,
                        "evidence_level": "L2_frozen_path_candidate",
                        "single_unit_causal": False,
                    })
    expected = len(MODELS) * 72 * 3
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} path selections, got {len(result)}")
    write_jsonl(root / "path_registry.jsonl", result)
    write_json(root / "path_registry_quality.json", {
        "phase_id": PHASE,
        "created_at": now(),
        "row_count": len(result),
        "expected_row_count": expected,
        "selection_split": "discovery_only",
        "heldout_used": False,
        "valid": len(result) == expected,
    })
    return result


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


@torch.inference_mode()
def scan_batch(
    loaded: Any,
    cases: list[dict[str, Any]],
    attention_spec: dict[str, Any],
    mlp_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    tokenizer = loaded.tokenizer
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        [case["prompt"] for case in cases], return_tensors="pt", padding=True,
        truncation=True, max_length=128,
    )
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    masks, _spans = survey.build_role_masks(tokenizer, cases, lengths, int(encoded["input_ids"].shape[1]), loaded.input_device)
    directions, _targets, _distractors = survey.make_direction_matrix(loaded, cases)
    layers = get_layers(loaded.model)
    attn_layer = int(attention_spec["component_layer"])
    mlp_layer = int(mlp_spec["component_layer"])
    o_proj, n_heads, head_dim = head_meta(loaded.model, attn_layer)
    down_proj = get_down_proj(layers[mlp_layer])
    if down_proj is None:
        raise TypeError(f"No down projection at layer {mlp_layer}")
    captures: dict[str, torch.Tensor] = {}

    def pool(tensor: torch.Tensor, role: str) -> torch.Tensor:
        mask = masks[role][:, : tensor.shape[1]]
        value = torch.einsum("bth,bt->bh", tensor.detach().float(), mask)
        return value / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def attn_pre(_module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            captures["attention"] = pool(inputs[0], attention_spec["position_role"])

    def mlp_pre(_module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            captures["mlp"] = pool(inputs[0], mlp_spec["position_role"])

    handles = [o_proj.register_forward_pre_hook(attn_pre), down_proj.register_forward_pre_hook(mlp_pre)]
    try:
        loaded.model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    attn_input_direction = directions.float() @ o_proj.weight.detach().float()
    attn_scores = (
        captures["attention"].view(len(cases), n_heads, head_dim)
        * attn_input_direction.view(len(cases), n_heads, head_dim)
    ).sum(dim=2)
    mlp_input_direction = directions.float() @ down_proj.weight.detach().float()
    ranges = group_ranges(int(captures["mlp"].shape[1]))
    mlp_scores = torch.stack([
        (captures["mlp"][:, start:end] * mlp_input_direction[:, start:end]).sum(dim=1)
        for start, end in ranges
    ], dim=1)
    rows = []
    for case_index, case in enumerate(cases):
        base = survey.output_row_base(case, loaded.key)
        for head in range(n_heads):
            start, end = head * head_dim, (head + 1) * head_dim
            vector = captures["attention"][case_index, start:end]
            rows.append({
                **base,
                "created_at": now(),
                "component_type": "attention_head_input",
                "component_layer": attn_layer,
                "position_role": attention_spec["position_role"],
                "component_index": head,
                "component_start": start,
                "component_end": end,
                "activation_rms": round(float(torch.sqrt(torch.mean(vector ** 2)).item()), 7),
                "approx_target_readout_contribution": round(float(attn_scores[case_index, head].item()), 7),
                "evidence_level": "L3_observational_component_candidate",
                "single_unit_causal": False,
            })
        for group, (start, end) in enumerate(ranges):
            vector = captures["mlp"][case_index, start:end]
            rows.append({
                **base,
                "created_at": now(),
                "component_type": "mlp_product_group",
                "component_layer": mlp_layer,
                "position_role": mlp_spec["position_role"],
                "component_index": group,
                "component_start": start,
                "component_end": end,
                "activation_rms": round(float(torch.sqrt(torch.mean(vector ** 2)).item()), 7),
                "approx_target_readout_contribution": round(float(mlp_scores[case_index, group].item()), 7),
                "evidence_level": "L3_observational_component_candidate",
                "single_unit_causal": False,
            })
    del encoded, directions, captures, attn_scores, mlp_scores
    return rows


def aggregate_carriers(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: {"discovery": [], "calibration": []})
    metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["family_id"], row["mechanism_id"], row["component_type"], row["component_layer"],
            row["position_role"], row["component_index"], row["component_start"], row["component_end"],
        )
        grouped[key][row["split"]].append(float(row["approx_target_readout_contribution"]))
        metadata[key] = row
    result = []
    for family, mechanisms in FAMILY_MECHANISMS.items():
        for mechanism in mechanisms:
            for component in ("attention_head_input", "mlp_product_group"):
                candidates = []
                for key, split_values in grouped.items():
                    if key[0] != family or key[1] != mechanism or key[2] != component:
                        continue
                    discovery = split_values["discovery"]
                    calibration = split_values["calibration"]
                    mean_score = mean(discovery)
                    positive_rate = sum(value > 0 for value in discovery) / len(discovery)
                    rank_score = max(0.0, mean_score) * positive_rate
                    candidates.append((rank_score, mean_score, positive_rate, key, calibration))
                candidates.sort(reverse=True, key=lambda value: (value[0], value[1]))
                for set_rank, (rank_score, mean_score, positive_rate, key, calibration) in enumerate(candidates[:2]):
                    source = metadata[key]
                    result.append({
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": model,
                        "family_id": family,
                        "mechanism_id": mechanism,
                        "component_type": component,
                        "component_layer": int(key[3]),
                        "position_role": key[4],
                        "component_index": int(key[5]),
                        "component_start": int(key[6]),
                        "component_end": int(key[7]),
                        "set_rank": set_rank,
                        "is_single_baseline": set_rank == 0,
                        "discovery_mean_contribution": round(mean_score, 7),
                        "discovery_positive_rate": round(positive_rate, 7),
                        "selection_score": round(rank_score, 7),
                        "calibration_mean_contribution": round(mean(calibration), 7),
                        "calibration_positive_rate": round(sum(value > 0 for value in calibration) / len(calibration), 7),
                        "selection_split": "discovery_only",
                        "calibration_role": "audit_only_no_update",
                        "heldout_used": False,
                        "evidence_level": "L3_frozen_carrier_candidate",
                        "single_unit_causal": False,
                        "component_width": int(source["component_end"]) - int(source["component_start"]),
                    })
    expected = 72 * 2 * 2
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} carriers for {model}, got {len(result)}")
    return result


def run_model(model: str, round_name: str, batch_size: int) -> dict[str, Any]:
    root = OUT / round_name
    path_rows = read_jsonl(root / "path_registry.jsonl")
    cases = survey.read_cases(round_name, next(iter(FAMILY_MECHANISMS)))
    all_cases_path = root / "phase330_case_bank.jsonl"
    all_cases = read_jsonl(all_cases_path)
    selected_cases = [
        case for case in all_cases
        if case["selection_eligible"] and case["template_id"] in {"template_a", "template_b"}
    ]
    if len(selected_cases) != 2592:
        raise RuntimeError(f"Expected 2592 scan cases, got {len(selected_cases)}")
    output = root / "component_scan" / model
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text(encoding="utf-8"))
    loaded = None
    component_rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for family, mechanisms in FAMILY_MECHANISMS.items():
            for mechanism in mechanisms:
                mechanism_cases = [
                    case for case in selected_cases
                    if case["family_id"] == family and case["mechanism_id"] == mechanism
                ]
                attention = next(row for row in path_rows if row["model"] == model and row["family_id"] == family and row["mechanism_id"] == mechanism and row["component_type"] == "attention")
                mlp = next(row for row in path_rows if row["model"] == model and row["family_id"] == family and row["mechanism_id"] == mechanism and row["component_type"] == "mlp")
                for batch in chunks(mechanism_cases, batch_size):
                    component_rows.extend(scan_batch(loaded, batch, attention, mlp))
                print(json.dumps({"quality_only": True, "model": model, "completed_mechanism": f"{family}/{mechanism}"}), flush=True)
        carriers = aggregate_carriers(component_rows, model)
        write_parquet(output / "component_candidates.parquet", component_rows)
        write_jsonl(output / "carrier_sets.jsonl", carriers)
        quality = {
            "phase_id": PHASE,
            "created_at": now(),
            "quality_only": True,
            "scientific_analysis_permitted": False,
            "model": model,
            "selection_case_count": len(selected_cases),
            "component_candidate_rows": len(component_rows),
            "carrier_set_rows": len(carriers),
            "expected_carrier_set_rows": 288,
            "heldout_used": False,
            "valid": len(carriers) == 288,
        }
        write_json(output / "complete.json", quality)
        return quality
    finally:
        release_loaded(loaded)
        gc.collect()


def collect(round_name: str) -> dict[str, Any]:
    root = OUT / round_name
    rows = []
    qualities = []
    for model in MODELS:
        rows.extend(read_jsonl(root / "component_scan" / model / "carrier_sets.jsonl"))
        qualities.append(json.loads((root / "component_scan" / model / "complete.json").read_text(encoding="utf-8")))
    write_jsonl(root / "carrier_sets.jsonl", rows)
    quality = {
        "phase_id": PHASE,
        "created_at": now(),
        "model_count": len(qualities),
        "carrier_set_rows": len(rows),
        "expected_carrier_set_rows": 864,
        "all_valid": all(row["valid"] for row in qualities),
        "heldout_used": False,
    }
    quality["valid"] = quality["model_count"] == 3 and quality["carrier_set_rows"] == 864 and quality["all_valid"]
    write_json(root / "component_scan_quality.json", quality)
    return quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps({"path_registry_rows": len(path_registry(args.round))}, indent=2))
    elif args.model:
        print(json.dumps(run_model(args.model, args.round, args.batch_size), indent=2))
    elif args.collect:
        print(json.dumps(collect(args.round), indent=2))
    else:
        raise SystemExit("Use --prepare, --model MODEL, or --collect")


if __name__ == "__main__":
    main()
