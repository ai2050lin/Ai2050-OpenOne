#!/usr/bin/env python3
"""C135 route B: all-token, all-coordinate effective transmission map."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1669_c135_all_token_coordinate_transmission"
C133 = RESULT / "phase1667_c133_multiroute_campaign_contract"
C134 = RESULT / "phase1668_c134_route_a_directed_composition"
C128 = RESULT / "phase1662_c128_direct_precedence_behavior_qualification"
C129 = RESULT / "phase1663_c129_direct_precedence_typed_transition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127

PHASE = 1669
CAMPAIGN = "C135"
CHECKPOINTS = c127.CHECKPOINTS
DIM = 2560
WIDTH = 112
BATCH = 2
DISCOVERY_UNITS = ("c128-00", "c128-01", "c128-02")
CONFIRMATION_UNITS = ("c128-16", "c128-17", "c128-18")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left.ravel(), right.ravel()) / denominator)


def select_anchors() -> list[dict]:
    rows = core.rows(C128 / "compiled/qwen3.jsonl")
    wanted = set(DISCOVERY_UNITS + CONFIRMATION_UNITS)
    selected = [row for row in rows if row["unit_id"] in wanted and row["surface_factor"] == 1 and row["distractor_factor"] == 1]
    selected.sort(key=lambda row: (0 if row["partition"] == "discovery" else 1, row["unit_id"], -row["truth_factor"]))
    return selected


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C135 exists: {OUT}")
    parent = core.load(C133 / "protocol/preregistration.json")
    route_a = core.load(C134 / "audit/independent_closure_audit.json")
    behavior = core.load(C128 / "analysis/behavior_gate.json")
    anchors = select_anchors()
    by_unit: dict[str, list[dict]] = {}
    for row in anchors:
        by_unit.setdefault(row["unit_id"], []).append(row)
    checks = {
        "campaign_parent": parent["routes"]["B"]["anchor_limit"] == 12,
        "route_a_authorization": route_a["all_checks_passed"] and route_a["authorization"] == "start_route_B_C135",
        "source_behavior": behavior["gate_passed"] and behavior["summary"]["global_accuracy"] == 1.0,
        "anchors": len(anchors) == 12 and len(by_unit) == 6,
        "partitions": sum(row["partition"] == "discovery" for row in anchors) == 6 and sum(row["partition"] == "confirmation" for row in anchors) == 6,
        "truth_pairs": all(len(rows) == 2 and {row["truth_factor"] for row in rows} == {1, -1} for rows in by_unit.values()),
        "equal_pair_lengths": all(len({len(row["prompt_ids"]) for row in rows}) == 1 for rows in by_unit.values()),
        "actual_width": max(len(row["prompt_ids"]) for row in anchors) < WIDTH,
        "checkpoints": len(CHECKPOINTS) == 38 and CHECKPOINTS[0] == "embedding" and CHECKPOINTS[-1] == "post_final_norm",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.write_rows(OUT / "material/anchors.jsonl", anchors)
    paths = {
        "c133": C133 / "protocol/preregistration.json",
        "c134_audit": C134 / "audit/independent_closure_audit.json",
        "c128_compiled": C128 / "compiled/qwen3.jsonl",
        "c128_behavior": C128 / "analysis/behavior_gate.json",
        "c129_closure": C129 / "analysis/closure.json",
    }
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "route_B_all_token_coordinate_contract_frozen",
        "object": "C129 behavior-qualified direct-precedence truth response at every actual token, strict checkpoint, and physical activation coordinate",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "anchor_units": {"discovery": list(DISCOVERY_UNITS), "confirmation": list(CONFIRMATION_UNITS)},
        "anchor_cases": len(anchors),
        "checkpoints": list(CHECKPOINTS),
        "coordinates": DIM,
        "prediction": "discovery-only diagonal coordinate gain predicts next-checkpoint truth response on untouched confirmation units",
        "gates": {"median_transition_cosine_min": 0.90, "relative_error_ratio_vs_identity_max": 0.95, "wrong_token_cosine_margin_min": 0.05, "wrong_coordinate_cosine_margin_min": 0.05},
        "controls": ["identity carry", "frozen one-token gain shift", "frozen one-coordinate gain shift"],
        "compression_report": [256, 1024, 4096, 16384],
        "claim_boundary": "effective one-step diagonal predictive dependency on 12 anchors; not a complete cross-coordinate Jacobian, unique cause, attention/MLP circuit, or universal language law",
        "forbidden": ["PCA", "SVD", "attention inspection", "MLP inspection", "weight inspection"],
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_hashes": {name: core.sha(path) for name, path in paths.items()},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "capture_c135_all_token_field",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", audit)
    print(json.dumps(audit, indent=2))


def tensor_output(value):
    return value[0] if isinstance(value, tuple) else value


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "capture_c135_all_token_field" or not core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C135 capture unauthorized")
    for name, path in protocol["source_paths"].items():
        if core.sha(Path(path)) != protocol["source_hashes"][name]:
            raise RuntimeError(f"source drift: {name}")
    rows = core.rows(OUT / "material/anchors.jsonl")
    offsets, cursor = [], 0
    for row in rows:
        offsets.append((cursor, cursor + len(row["prompt_ids"])))
        cursor += len(row["prompt_ids"])
    raw_path = OUT / "raw/qwen3_all_token_all_checkpoint.bf16.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=(len(CHECKPOINTS), cursor, DIM))
    model, repeat = None, 0
    index = []
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        if len(model.model.layers) != 36:
            raise RuntimeError(len(model.model.layers))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def run(batch):
            captured = {}
            handles = [model.model.embed_tokens.register_forward_hook(lambda _m, _a, output: captured.__setitem__("embedding", tensor_output(output).detach()))]
            for layer_index, layer in enumerate(model.model.layers):
                handles.append(layer.register_forward_hook(lambda _m, _a, output, index=layer_index: captured.__setitem__(f"block_{index}", tensor_output(output).detach())))
            handles.append(model.model.norm.register_forward_hook(lambda _m, _a, output: captured.__setitem__("norm", tensor_output(output).detach())))
            try:
                ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
                output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            tensors = [captured["embedding"], *[captured[f"block_{index}"] for index in range(36)], captured["norm"]]
            return tensors, output, ids, mask, positions, lengths

        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            tensors, output, ids, mask, positions, lengths = run(batch)
            for local, row in enumerate(batch):
                left, right = offsets[start + local]
                length = lengths[local]
                if right - left != length:
                    raise RuntimeError((row["case_id"], right - left, length))
                for checkpoint, tensor in enumerate(tensors):
                    field[checkpoint, left:right] = tensor[local, :length].contiguous().view(torch.uint16).cpu().numpy()
                index.append({"row_index": start + local, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"], "truth_factor": row["truth_factor"], "token_offset_start": left, "token_offset_end": right, "token_count": length, "token_ids": row["prompt_ids"]})
            field.flush()
            print(f"[C135] captured {start + len(batch)}/{len(rows)}", flush=True)
            del tensors, output, ids, mask, positions
        tensors, output, ids, mask, positions, lengths = run(rows[:BATCH])
        for local in range(BATCH):
            left, right = offsets[local]
            for checkpoint, tensor in enumerate(tensors):
                bits = tensor[local, :right-left].contiguous().view(torch.uint16).cpu().numpy()
                repeat = max(repeat, int(np.max(np.abs(bits.astype(np.int64) - field[checkpoint, left:right].astype(np.int64)))))
    finally:
        field.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    core.write_rows(OUT / "raw/all_token_field_index.jsonl", index)
    checks = {"shape": list(field.shape) == [38, cursor, DIM], "all_actual_tokens": sum(row["token_count"] for row in index) == cursor, "index": len(index) == 12, "finite": bool(np.isfinite(c127.decode(field[:, :2])).all()), "repeat_bits": repeat == 0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "all_token_coordinate_capture_complete", "checks": checks, "shape": list(field.shape), "total_actual_tokens": cursor, "sha256": core.sha(raw_path), "index_sha256": core.sha(OUT / "raw/all_token_field_index.jsonl"), "runtime": {"placement": placement, "quantization": quant}, "authorization": "discover_c135_transmission"}
    core.save(OUT / "analysis/capture.json", report)
    core.save(OUT / "audit/internal_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def truth_fields(partition: str) -> tuple[dict[int, np.ndarray], dict[int, list[str]]]:
    field = np.load(OUT / "raw/qwen3_all_token_all_checkpoint.bf16.npy", mmap_mode="r")
    index = core.rows(OUT / "raw/all_token_field_index.jsonl")
    units = list(DISCOVERY_UNITS if partition == "discovery" else CONFIRMATION_UNITS)
    grouped: dict[int, list[np.ndarray]] = {}
    grouped_units: dict[int, list[str]] = {}
    for unit in units:
        pair = sorted([row for row in index if row["unit_id"] == unit], key=lambda row: -row["truth_factor"])
        if len(pair) != 2 or pair[0]["token_count"] != pair[1]["token_count"]:
            raise RuntimeError((unit, pair))
        pos = c127.decode(field[:, pair[0]["token_offset_start"]:pair[0]["token_offset_end"]])
        neg = c127.decode(field[:, pair[1]["token_offset_start"]:pair[1]["token_offset_end"]])
        length = int(pair[0]["token_count"])
        grouped.setdefault(length, []).append(pos - neg)
        grouped_units.setdefault(length, []).append(unit)
    return {length: np.stack(values).astype(np.float32) for length, values in grouped.items()}, grouped_units


def transition_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    rows = []
    for transition in range(predicted.shape[1]):
        rows.append({"transition_index": transition, "from_checkpoint": CHECKPOINTS[transition], "to_checkpoint": CHECKPOINTS[transition + 1], "cosine": cosine(predicted[:, transition], target[:, transition]), "relative_error": float(np.linalg.norm(predicted[:, transition] - target[:, transition]) / max(np.linalg.norm(target[:, transition]), 1e-12))})
    return {"rows": rows, "median_cosine": float(np.median([row["cosine"] for row in rows])), "median_relative_error": float(np.median([row["relative_error"] for row in rows]))}


def pooled_transition_metrics(predicted: dict[int, np.ndarray], target: dict[int, np.ndarray]) -> dict:
    rows = []
    for transition in range(37):
        pred = np.concatenate([predicted[length][:, transition].ravel() for length in sorted(predicted)])
        truth = np.concatenate([target[length][:, transition].ravel() for length in sorted(target)])
        rows.append({"transition_index": transition, "from_checkpoint": CHECKPOINTS[transition], "to_checkpoint": CHECKPOINTS[transition + 1], "cosine": cosine(pred, truth), "relative_error": float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-12))})
    return {"rows": rows, "median_cosine": float(np.median([row["cosine"] for row in rows])), "median_relative_error": float(np.median([row["relative_error"] for row in rows]))}


def discover() -> None:
    if core.load(OUT / "analysis/capture.json")["authorization"] != "discover_c135_transmission":
        raise RuntimeError("unauthorized")
    fields, units = truth_fields("discovery")
    lengths = sorted(fields)
    max_length = max(lengths)
    gain = np.zeros((len(lengths), 37, max_length, DIM), dtype=np.float32)
    denominator = np.zeros_like(gain)
    for stratum, length in enumerate(lengths):
        x, y = fields[length][:, :-1], fields[length][:, 1:]
        numerator_local = np.sum(x * y, axis=0, dtype=np.float32)
        denominator_local = np.sum(x * x, axis=0, dtype=np.float32)
        gain[stratum, :, :length] = np.divide(numerator_local, denominator_local, out=np.zeros_like(numerator_local), where=denominator_local > 1e-8)
        denominator[stratum, :, :length] = denominator_local
    gain_path = OUT / "protocol/frozen_diagonal_gain.float32.npy"
    gain_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(gain_path, gain)
    score = np.sqrt(denominator) * np.abs(gain)
    flat = score.ravel()
    order = np.argpartition(flat, -min(16384, flat.size))[-min(16384, flat.size):]
    order = order[np.argsort(-flat[order])]
    energy = np.square(flat, dtype=np.float64)
    total_energy = float(np.sum(energy))
    cumulative = np.cumsum(np.sort(energy)[::-1]) / max(total_energy, 1e-30)
    compression = {str(k): float(cumulative[min(k, len(cumulative)) - 1]) for k in (256, 1024, 4096, 16384)}
    k90 = int(np.searchsorted(cumulative, 0.90) + 1)
    candidate_rows = []
    for flat_index in order[:4096]:
        stratum, transition, token, coordinate = np.unravel_index(int(flat_index), score.shape)
        length = lengths[stratum]
        candidate_rows.append({"length_stratum": int(length), "discovery_units_in_stratum": len(units[length]), "transition_index": int(transition), "from_checkpoint": CHECKPOINTS[transition], "to_checkpoint": CHECKPOINTS[transition + 1], "token_position": int(token), "coordinate": int(coordinate), "discovery_score": float(score[stratum, transition, token, coordinate]), "gain": float(gain[stratum, transition, token, coordinate]), "source_energy": float(denominator[stratum, transition, token, coordinate])})
    core.write_rows(OUT / "analysis/discovery_top_coordinate_edges.jsonl", candidate_rows)
    freeze = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "all_token_diagonal_transmission_frozen", "discovery_units_by_length": {str(length): units[length] for length in lengths}, "length_strata": lengths, "length_stratification_reason": "frozen anchor pairs are internally equal length but span two physical token lengths; no padding, truncation, or anchor reselection", "analysis_producer_sha256": core.sha(Path(__file__)), "gain_shape": list(gain.shape), "gain_sha256": core.sha(gain_path), "top_edge_count": len(candidate_rows), "compression_energy_fraction": compression, "coordinates_for_90_percent_discovery_score_energy": k90, "confirmation_unread": True, "authorization": "validate_c135_confirmation"}
    core.save(OUT / "protocol/frozen_transmission.json", freeze)
    checks = {"units": sum(len(value) for value in units.values()) == 3, "strata": lengths == [89, 101], "gain_shape": list(gain.shape) == [2, 37, 101, DIM], "finite": bool(np.isfinite(gain).all()), "edges": len(candidate_rows) == 4096}
    core.save(OUT / "audit/internal_discovery_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": freeze["authorization"]})
    print(json.dumps(freeze, indent=2))


def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    freeze = core.load(OUT / "protocol/frozen_transmission.json")
    if freeze["authorization"] != "validate_c135_confirmation":
        raise RuntimeError("unauthorized")
    gain = np.load(OUT / "protocol/frozen_diagonal_gain.float32.npy", mmap_mode="r")
    fields, units = truth_fields("confirmation")
    lengths = [int(value) for value in freeze["length_strata"]]
    predicted, identity, wrong_token, wrong_coordinate, target = {}, {}, {}, {}, {}
    for stratum, length in enumerate(lengths):
        x = fields[length][:, :-1]
        target[length] = fields[length][:, 1:]
        local_gain = np.asarray(gain[stratum, :, :length])
        predicted[length] = x * local_gain[None]
        identity[length] = x
        wrong_token[length] = x * np.roll(local_gain, 1, axis=1)[None]
        wrong_coordinate[length] = x * np.roll(local_gain, 1, axis=2)[None]
    metrics = pooled_transition_metrics(predicted, target)
    identity_metrics = pooled_transition_metrics(identity, target)
    token_metrics = pooled_transition_metrics(wrong_token, target)
    coordinate_metrics = pooled_transition_metrics(wrong_coordinate, target)
    rows = core.rows(OUT / "analysis/discovery_top_coordinate_edges.jsonl")
    for row in rows:
        length, q, t, a = row["length_stratum"], row["transition_index"], row["token_position"], row["coordinate"]
        row["confirmation_units_in_stratum"] = len(units[length])
        row["confirmation_cosine_across_units"] = cosine(predicted[length][:, q, t, a], target[length][:, q, t, a])
        row["confirmation_absolute_error"] = float(np.mean(np.abs(predicted[length][:, q, t, a] - target[length][:, q, t, a])))
    core.write_rows(OUT / "analysis/confirmation_top_coordinate_edges.jsonl", rows)
    gates = protocol["gates"]
    error_ratio = metrics["median_relative_error"] / max(identity_metrics["median_relative_error"], 1e-12)
    token_margin = metrics["median_cosine"] - token_metrics["median_cosine"]
    coordinate_margin = metrics["median_cosine"] - coordinate_metrics["median_cosine"]
    checks = {"cosine": metrics["median_cosine"] >= gates["median_transition_cosine_min"], "identity_error": error_ratio <= gates["relative_error_ratio_vs_identity_max"], "wrong_token": token_margin >= gates["wrong_token_cosine_margin_min"], "wrong_coordinate": coordinate_margin >= gates["wrong_coordinate_cosine_margin_min"]}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "route_B_confirmation_adjudicated", "confirmation_units_by_length": {str(length): units[length] for length in lengths}, "metrics": metrics, "identity_control": identity_metrics, "wrong_token_control": token_metrics, "wrong_coordinate_control": coordinate_metrics, "derived": {"relative_error_ratio_vs_identity": error_ratio, "wrong_token_cosine_margin": token_margin, "wrong_coordinate_cosine_margin": coordinate_margin}, "gates": checks, "prediction_gate_passed": all(checks.values()), "authorization": "close_c135_continue_C"}
    core.save(OUT / "analysis/confirmation.json", report)
    audit_checks = {"units": sum(len(value) for value in units.values()) == 3, "strata": sorted(units) == lengths, "transitions": len(metrics["rows"]) == 37, "finite": all(np.isfinite(value) for value in (metrics["median_cosine"], metrics["median_relative_error"], error_ratio, token_margin, coordinate_margin)), "edge_rows": len(rows) == 4096}
    core.save(OUT / "audit/internal_confirmation_audit.json", {"checks": audit_checks, "all_checks_passed": all(audit_checks.values()), "scientific_gates": checks, "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key not in {"metrics", "identity_control", "wrong_token_control", "wrong_coordinate_control"}}, indent=2))


def close() -> None:
    confirmation = core.load(OUT / "analysis/confirmation.json")
    closure = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "route_B_closed", "headline": {"prediction_gate_passed": confirmation["prediction_gate_passed"], "metrics": {"target_median_cosine": confirmation["metrics"]["median_cosine"], "target_median_relative_error": confirmation["metrics"]["median_relative_error"], **confirmation["derived"]}}, "theory_update": "Every-token physical-coordinate truth-response field is now measured; the frozen model tests only same-coordinate one-step effective transmission.", "problems": ["12 anchors only", "three units per split", "direct precedence only", "diagonal model omits cross-coordinate and higher-order edges", "prediction dependence is not unique causation"], "claim_boundary": "all actual token/checkpoint/coordinate observation and held-out prediction, not a unique circuit", "causal_candidate": confirmation["prediction_gate_passed"], "next_authorization": "continue route C regardless; retain causal candidate only if all C133 requirements survive later audits"}
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"], "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_checks_passed"]}
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_gate_passed": confirmation["prediction_gate_passed"], "authorization": "independent_audit_then_route_C"})
    print(json.dumps(closure, indent=2))


def main() -> None:
    modes = {"contract": contract, "capture": capture, "discover": discover, "validate": validate, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: {Path(__file__).name} {'|'.join(modes)}")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
