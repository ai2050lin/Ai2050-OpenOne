#!/usr/bin/env python3
"""C127 behavior-qualified, uniformly typed HiddenState transition observation."""
from __future__ import annotations

import gc
import inspect
import itertools
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1661_c127_typed_transition_language_family"
C126 = RESULT / "phase1660_c126_factor_response_decomposition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

FAMILY = "two_hop_precedence"
PARTITIONS = ("discovery", "confirmation")
ROLES = ("source_fact", "bridge_fact", "target_fact", "query_left", "query_right", "boundary")
CHECKPOINTS = ("embedding",) + tuple(f"post_block_{index}_pre_final_norm" for index in range(36)) + ("post_final_norm",)
DIM = 2560
WIDTH = 176
BATCH = 8
SUPPORT_K = 256
SYSTEM = "Evaluate the route notes exactly as written. Answer only yes or no."
SYLLABLES = ("bex", "civ", "dor", "fal", "gim", "hux", "jap", "kel", "lom", "mur", "nex", "piv", "qor", "rav", "sul", "tev")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def topk(values: np.ndarray, k: int = SUPPORT_K) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def decode(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def unit_values(index: int) -> tuple[str, str, str, str, str]:
    left = SYLLABLES[index % len(SYLLABLES)]
    right = SYLLABLES[(index * 5 + 3) % len(SYLLABLES)]
    return tuple(f"Sept{left}{right}{index:02d}{suffix}" for suffix in ("a", "b", "c", "d", "e"))


def prompt_for(values: tuple[str, ...], truth: int, surface: int, distractor: int) -> tuple[str, str, str]:
    source, bridge, target, extra_left, extra_right = values
    if surface == 1:
        facts = f"Route notes: {source} is upstream of {bridge}. {bridge} is upstream of {target}."
        distractor_text = f"Separately, {extra_left if distractor == 1 else extra_right} is upstream of {extra_right if distractor == 1 else extra_left}."
    else:
        facts = f"Route notes: immediately upstream of {target} is {bridge}; immediately upstream of {bridge} is {source}."
        distractor_text = f"Separately, immediately upstream of {extra_right if distractor == 1 else extra_left} is {extra_left if distractor == 1 else extra_right}."
    query_left, query_right = (source, target) if truth == 1 else (target, source)
    prompt = f"{facts} {distractor_text} Question: Is {query_left} upstream of {query_right}? Reply exactly yes or no."
    return prompt, query_left, query_right


def build_material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit_index in range(32):
        values = unit_values(unit_index)
        partition = PARTITIONS[unit_index // 16]
        unit = {"unit_id": f"c127-{unit_index:02d}", "family": FAMILY, "partition": partition, "world": "controlled_synthetic_two_hop_precedence", "values": list(values)}
        units.append(unit)
        for truth, surface, distractor in itertools.product((1, -1), repeat=3):
            prompt, query_left, query_right = prompt_for(values, truth, surface, distractor)
            cases.append({
                **unit,
                "case_id": f"c127-{len(cases):04d}",
                "truth_factor": truth,
                "surface_factor": surface,
                "distractor_factor": distractor,
                "truth": truth == 1,
                "output_yes": truth == 1,
                "gold_position": 0 if truth == 1 else 1,
                "query_left": query_left,
                "query_right": query_right,
                "prompt": prompt,
            })
    return units, cases


def historical_values() -> set[str]:
    values = set()
    for path in RESULT.glob("phase*/material/units.jsonl"):
        for row in core.rows(path):
            values.update(str(value).casefold() for value in row.get("values", []))
    return values


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidate_ids = [[int(value) for value in tokenizer.encode(" " + candidate, add_special_tokens=False)] for candidate in ("yes", "no")]
    if any(len(value) != 1 for value in candidate_ids):
        raise RuntimeError(("candidate_singleton", candidate_ids))
    compiled = []
    for row in rows:
        ids = core.chat_ids(tokenizer, SYSTEM, row["prompt"])
        source, bridge, target, _extra_left, _extra_right = row["values"]
        source_spans = graph_base.name_spans(tokenizer, ids, source)
        bridge_spans = graph_base.name_spans(tokenizer, ids, bridge)
        target_spans = graph_base.name_spans(tokenizer, ids, target)
        query_left_spans = graph_base.name_spans(tokenizer, ids, row["query_left"])
        query_right_spans = graph_base.name_spans(tokenizer, ids, row["query_right"])
        if min(len(source_spans), len(target_spans), len(query_left_spans), len(query_right_spans)) < 2 or len(bridge_spans) < 2:
            raise RuntimeError((row["case_id"], source_spans, bridge_spans, target_spans))
        roles = {
            "source_fact": source_spans[0],
            "bridge_fact": sorted({position for span in bridge_spans[:2] for position in span}),
            "target_fact": target_spans[0],
            "query_left": query_left_spans[-1],
            "query_right": query_right_spans[-1],
            "boundary": [len(ids) - 1],
        }
        if not (max(roles["source_fact"]) < min(roles["query_left"]) and max(roles["target_fact"]) < min(roles["query_right"]) and max(roles["query_left"]) < roles["boundary"][0] and max(roles["query_right"]) < roles["boundary"][0]):
            raise RuntimeError(("role_order", row["case_id"], roles))
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids, "role_positions": roles})
    return compiled


def local_semantics() -> dict:
    import transformers.models.qwen3.modeling_qwen3 as source
    path = Path(inspect.getsourcefile(source.Qwen3Model))
    return {"path": str(path), "sha256": core.sha(path), "decoder_layers": 36, "checkpoint_count": len(CHECKPOINTS)}


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C127 already exists: {OUT}")
    parent = core.load(C126 / "analysis/closure.json")
    parent_audit = core.load(C126 / "audit/independent_closure_audit.json")
    units, cases = build_material()
    tokenizer = graph_base.tokenizer()
    compiled = compile_rows(tokenizer, cases)
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    old = historical_values()
    cells = Counter((row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"]) for row in cases)
    zero = {
        "always_yes": float(np.mean([row["truth"] for row in cases])),
        "always_no": float(np.mean([not row["truth"] for row in cases])),
        "surface_only": float(np.mean([(row["surface_factor"] == 1) == row["truth"] for row in cases])),
        "distractor_only": float(np.mean([(row["distractor_factor"] == 1) == row["truth"] for row in cases])),
    }
    checks = {
        "authorization": parent_audit["all_checks_passed"] and parent["next_authorization"].startswith("C127 may remain"),
        "units": len(units) == 32,
        "cases": len(cases) == 256,
        "factorial": cells == {(partition, *cell): 16 for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=3)},
        "freshness": not (set(fresh) & old) and len(fresh) == len(set(fresh)),
        "unique_prompts": len({row["prompt"] for row in cases}) == 256,
        "zero_models": all(abs(value - 0.5) < 1e-12 for value in zero.values()),
        "candidate_ids": all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "machine_naturalness": all(row["prompt"].startswith("Route notes:") and row["prompt"].endswith("Reply exactly yes or no.") for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": 1661,
        "campaign": "C127",
        "created_at_utc": now(),
        "status": "two_hop_precedence_typed_transition_contract_frozen",
        "object": "behavior-qualified truth response trajectories for a fresh two-hop precedence language family",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "material": {"units": 32, "cases": 256, "partitions": list(PARTITIONS), "factors": ["truth", "surface", "distractor"], "human_naturalness": "not independently rated"},
        "zero_models": zero,
        "roles": list(ROLES),
        "checkpoints": list(CHECKPOINTS),
        "checkpoint_semantics": "embedding; exact output of each decoder block before final norm; exact output after final norm",
        "behavior_gate": {"global_accuracy_min": 0.90, "partition_accuracy_min": 0.85, "truth_accuracy_min": 0.85, "surface_accuracy_min": 0.85},
        "discovery_rule": {"partition": "discovery", "unit_split": "first eight versus last eight", "score": "max(0,split_half_cosine)*min(split_half_L2_norms)", "support_k": SUPPORT_K},
        "confirmation_gates": {"cosine_min": 0.90, "top256_overlap_min": 0.50, "support_sign_agreement_min": 0.75, "coordinate_clock_within_one_min": 0.70, "wrong_state_margin_gt": 0.0, "wrong_role_margin_gt": 0.0},
        "stop_conditions": ["behavior gate failure forbids HiddenState capture", "numeric failure stops analysis", "confirmation failure closes the route without threshold changes"],
        "observation_policy": "full 2560 activation coordinates; no PCA/SVD, attention, MLP, or weight analysis",
        "claim_boundary": "controlled Qwen3 activation responses, not weights, semantic neurons, a complete language operator, a complete-token causal graph, topology, or new mathematics",
        "local_capture_semantics": local_semantics(),
        "parent_paths": {"closure": str(C126 / "analysis/closure.json"), "audit": str(C126 / "audit/independent_closure_audit.json")},
        "parent_hashes": {"closure": core.sha(C126 / "analysis/closure.json"), "audit": core.sha(C126 / "audit/independent_closure_audit.json")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_c127_behavior_only",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1661, "campaign": "C127", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "max_width": max(len(row["prompt_ids"]) for row in compiled), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def accuracy(rows: list[dict]) -> float:
    return float(np.mean([row["correct"] for row in rows]))


@torch.inference_mode()
def behavior() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "run_c127_behavior_only" or not core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C127 behavior authorization missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    result_rows = []
    logits_path = OUT / "raw/qwen3_behavior_candidate_logits.float32.npy"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(rows), 2))
    model = None
    repeat = 0.0
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        first_batch = rows[:BATCH]
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
            boundary = torch.stack([output.last_hidden_state[index, length - 1] for index, length in enumerate(lengths)])
            logits = model.lm_head(boundary).float()
            for local, row in enumerate(batch):
                scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
                values[start + local] = scores
                prediction = int(scores[1] > scores[0])
                result_rows.append({"row_index": start + local, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"], "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "yes_minus_no": float(scores[0] - scores[1])})
            del output, boundary, logits, ids, mask, positions
        values.flush()
        ids, mask, positions, lengths = fixed_base.fixed_batch(first_batch, pad, device, WIDTH)
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
        boundary = torch.stack([output.last_hidden_state[index, length - 1] for index, length in enumerate(lengths)])
        logits = model.lm_head(boundary).float()
        for local, row in enumerate(first_batch):
            scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
            repeat = max(repeat, float(np.max(np.abs(scores - values[local]))))
    finally:
        values.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    core.write_rows(OUT / "raw/qwen3_behavior_index.jsonl", result_rows)
    summary = {
        "global_accuracy": accuracy(result_rows),
        "by_partition": {key: accuracy([row for row in result_rows if row["partition"] == key]) for key in PARTITIONS},
        "by_truth": {str(key): accuracy([row for row in result_rows if row["truth_factor"] == key]) for key in (1, -1)},
        "by_surface": {str(key): accuracy([row for row in result_rows if row["surface_factor"] == key]) for key in (1, -1)},
    }
    gates = protocol["behavior_gate"]
    gate = summary["global_accuracy"] >= gates["global_accuracy_min"] and min(summary["by_partition"].values()) >= gates["partition_accuracy_min"] and min(summary["by_truth"].values()) >= gates["truth_accuracy_min"] and min(summary["by_surface"].values()) >= gates["surface_accuracy_min"]
    checks = {"rows": len(result_rows) == 256, "finite": bool(np.isfinite(values).all()), "repeat": repeat == 0.0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1661, "campaign": "C127", "created_at_utc": now(), "status": "behavior_qualified" if gate else "behavior_gate_failed", "checks": checks, "summary": summary, "gate_passed": gate, "repeat_logits_max_abs": repeat, "runtime": {"placement": placement, "quantization": quant}, "authorization": "capture_uniform_typed_hiddenstates" if gate else "close_without_hiddenstate_capture"}
    core.save(OUT / "analysis/behavior_gate.json", report)
    core.save(OUT / "audit/internal_behavior_audit.json", {"phase": 1661, "campaign": "C127", "checks": checks, "all_integrity_checks_passed": all(checks.values()), "gate_passed": gate, "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def tensor_output(value):
    return value[0] if isinstance(value, tuple) else value


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior_report = core.load(OUT / "analysis/behavior_gate.json")
    if behavior_report["authorization"] != "capture_uniform_typed_hiddenstates" or not behavior_report["gate_passed"]:
        raise RuntimeError("C127 HiddenState capture forbidden by behavior gate")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    raw_path = OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=(len(rows), len(ROLES), len(CHECKPOINTS), DIM))
    model = None
    repeat = 0
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        if len(model.model.layers) != 36:
            raise RuntimeError(("layers", len(model.model.layers)))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def run_batch(batch):
            captured = {}
            handles = [model.model.embed_tokens.register_forward_hook(lambda _m, _a, output: captured.__setitem__("embedding", tensor_output(output).detach()))]
            for layer_index, layer in enumerate(model.model.layers):
                handles.append(layer.register_forward_hook(lambda _m, _a, output, index=layer_index: captured.__setitem__(f"block_{index}", tensor_output(output).detach())))
            handles.append(model.model.norm.register_forward_hook(lambda _m, _a, output: captured.__setitem__("final_norm", tensor_output(output).detach())))
            try:
                ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
                output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            tensors = [captured["embedding"], *[captured[f"block_{index}"] for index in range(36)], captured["final_norm"]]
            if len(tensors) != len(CHECKPOINTS) or any(tensor.dtype != torch.bfloat16 for tensor in tensors):
                raise RuntimeError((len(tensors), [str(tensor.dtype) for tensor in tensors]))
            return tensors, output, ids, mask, positions, lengths

        first_batch = rows[:BATCH]
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            tensors, output, ids, mask, positions, lengths = run_batch(batch)
            for local, row in enumerate(batch):
                for role_index, role in enumerate(ROLES):
                    token_positions = row["role_positions"][role]
                    for checkpoint_index, tensor in enumerate(tensors):
                        vector = tensor[local, token_positions].mean(dim=0)
                        raw[start + local, role_index, checkpoint_index] = vector.contiguous().view(torch.uint16).cpu().numpy()
            if (start // BATCH + 1) % 8 == 0:
                raw.flush()
                print(f"[C127] captured {start + len(batch)}/{len(rows)}", flush=True)
            del tensors, output, ids, mask, positions
        raw.flush()
        tensors, output, ids, mask, positions, lengths = run_batch(first_batch)
        for local, row in enumerate(first_batch):
            for role_index, role in enumerate(ROLES):
                token_positions = row["role_positions"][role]
                for checkpoint_index, tensor in enumerate(tensors):
                    bits = tensor[local, token_positions].mean(dim=0).contiguous().view(torch.uint16).cpu().numpy()
                    repeat = max(repeat, int(np.max(np.abs(bits.astype(np.int64) - raw[local, role_index, checkpoint_index].astype(np.int64)))))
    finally:
        raw.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {"shape": list(raw.shape) == [256, 6, 38, DIM], "finite": bool(np.isfinite(decode(raw[:2])).all()), "repeat_bits": repeat == 0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1661, "campaign": "C127", "created_at_utc": now(), "status": "uniform_typed_hiddenstate_capture_complete", "checks": checks, "shape": list(raw.shape), "sha256": core.sha(raw_path), "runtime": {"placement": placement, "quantization": quant}, "authorization": "discover_c127_transition"}
    core.save(OUT / "analysis/capture_summary.json", report)
    core.save(OUT / "audit/internal_capture_audit.json", {"phase": 1661, "campaign": "C127", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def unit_truth_fields() -> tuple[np.ndarray, list[dict], list[dict]]:
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = core.rows(OUT / "material/units.jsonl")
    raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r")
    lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    fields = np.zeros((len(units), len(ROLES), len(CHECKPOINTS), DIM), dtype=np.float32)
    for row_index, row in enumerate(rows):
        fields[lookup[row["unit_id"]]] += float(row["truth_factor"]) / 8.0 * decode(raw[row_index])
    return fields, rows, units


def discover() -> None:
    if core.load(OUT / "analysis/capture_summary.json")["authorization"] != "discover_c127_transition":
        raise RuntimeError("C127 discovery authorization missing")
    fields, _rows, units = unit_truth_fields()
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", fields)
    discovery_indices = [index for index, row in enumerate(units) if row["partition"] == "discovery"]
    discovery = fields[discovery_indices]
    increments = discovery[:, :, 1:] - discovery[:, :, :-1]
    left = np.mean(increments[:8], axis=0, dtype=np.float32)
    right = np.mean(increments[8:], axis=0, dtype=np.float32)
    candidates = []
    for role_index, role in enumerate(ROLES):
        for transition_index in range(len(CHECKPOINTS) - 1):
            similarity = cosine(left[role_index, transition_index], right[role_index, transition_index])
            left_norm = float(np.linalg.norm(left[role_index, transition_index]))
            right_norm = float(np.linalg.norm(right[role_index, transition_index]))
            candidates.append({"role": role, "role_index": role_index, "from_checkpoint": CHECKPOINTS[transition_index], "to_checkpoint": CHECKPOINTS[transition_index + 1], "transition_index": transition_index, "split_half_cosine": similarity, "left_norm": left_norm, "right_norm": right_norm, "score": max(0.0, similarity) * min(left_norm, right_norm)})
    candidates.sort(key=lambda row: (-row["score"], -row["split_half_cosine"], row["transition_index"], ROLES.index(row["role"])))
    nominee = dict(candidates[0])
    mean_increment = np.mean(increments[:, nominee["role_index"], nominee["transition_index"]], axis=0, dtype=np.float32)
    support = sorted(topk(mean_increment), key=lambda index: -abs(float(mean_increment[index])))
    nominee.update({"support_k": SUPPORT_K, "support": support})
    core.write_rows(OUT / "analysis/discovery_candidate_table.jsonl", candidates)
    np.save(OUT / "analysis/discovery_nominee_increment.float32.npy", mean_increment)
    freeze = {"phase": 1661, "campaign": "C127", "created_at_utc": now(), "status": "discovery_nomination_frozen", "nominee": nominee, "candidate_count": len(candidates), "confirmation_partition_unread": True, "authorization": "validate_c127_confirmation"}
    core.save(OUT / "protocol/frozen_discovery_nomination.json", freeze)
    checks = {"field_shape": list(fields.shape) == [32, 6, 38, DIM], "finite": bool(np.isfinite(fields).all()), "candidates": len(candidates) == 222, "support": len(support) == SUPPORT_K, "partition": len(discovery_indices) == 16}
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "audit/internal_discovery_audit.json", {"phase": 1661, "campaign": "C127", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": freeze["authorization"]})
    print(json.dumps(freeze, indent=2))


def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    freeze = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    if freeze["authorization"] != "validate_c127_confirmation" or not core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"]:
        raise RuntimeError("C127 confirmation authorization missing")
    fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r")
    units = core.rows(OUT / "material/units.jsonl")
    confirmation_indices = [index for index, row in enumerate(units) if row["partition"] == "confirmation"]
    confirmation = np.asarray(fields[confirmation_indices], dtype=np.float32)
    increments = confirmation[:, :, 1:] - confirmation[:, :, :-1]
    nominee = freeze["nominee"]
    role_index = int(nominee["role_index"])
    transition_index = int(nominee["transition_index"])
    discovery_vector = np.load(OUT / "analysis/discovery_nominee_increment.float32.npy")
    confirmation_vector = np.mean(increments[:, role_index, transition_index], axis=0, dtype=np.float32)
    target_cosine = cosine(discovery_vector, confirmation_vector)
    support = set(nominee["support"])
    overlap = len(topk(confirmation_vector) & support) / SUPPORT_K
    signs = [np.sign(discovery_vector[index]) == np.sign(confirmation_vector[index]) for index in support]
    sign_agreement = float(np.mean(signs))
    wrong_state = max(cosine(discovery_vector, np.mean(increments[:, role_index, index], axis=0, dtype=np.float32)) for index in range(37) if index != transition_index)
    wrong_role = max(cosine(discovery_vector, np.mean(increments[:, index, transition_index], axis=0, dtype=np.float32)) for index in range(len(ROLES)) if index != role_index)
    discovery_part = np.asarray(fields[:16], dtype=np.float32)
    discovery_increments = discovery_part[:, :, 1:] - discovery_part[:, :, :-1]
    discovery_peaks = np.argmax(np.abs(np.mean(discovery_increments[:, role_index], axis=0, dtype=np.float32)), axis=0)
    confirmation_peaks = np.argmax(np.abs(np.mean(increments[:, role_index], axis=0, dtype=np.float32)), axis=0)
    within_one = float(np.mean(np.abs(discovery_peaks[list(support)] - confirmation_peaks[list(support)]) <= 1))
    gates = protocol["confirmation_gates"]
    gate_results = {
        "cosine": target_cosine >= gates["cosine_min"],
        "top256_overlap": overlap >= gates["top256_overlap_min"],
        "sign_agreement": sign_agreement >= gates["support_sign_agreement_min"],
        "coordinate_clock": within_one >= gates["coordinate_clock_within_one_min"],
        "wrong_state": target_cosine - wrong_state > gates["wrong_state_margin_gt"],
        "wrong_role": target_cosine - wrong_role > gates["wrong_role_margin_gt"],
    }
    report = {"phase": 1661, "campaign": "C127", "created_at_utc": now(), "status": "confirmation_passed" if all(gate_results.values()) else "confirmation_failed", "nominee": nominee, "confirmation_units": 16, "metrics": {"target_cosine": target_cosine, "top256_overlap": overlap, "support_sign_agreement": sign_agreement, "coordinate_clock_within_one": within_one, "best_wrong_state_cosine": wrong_state, "best_wrong_role_cosine": wrong_role, "state_margin": target_cosine - wrong_state, "role_margin": target_cosine - wrong_role}, "gates": gate_results, "all_gates_passed": all(gate_results.values()), "authorization": "synthesize_c127_and_close"}
    core.save(OUT / "analysis/confirmation.json", report)
    core.save(OUT / "audit/internal_confirmation_audit.json", {"phase": 1661, "campaign": "C127", "integrity": {"confirmation_units": len(confirmation_indices) == 16, "finite": bool(np.isfinite(confirmation).all()), "gate_count": len(gate_results) == 6}, "all_integrity_checks_passed": len(confirmation_indices) == 16 and bool(np.isfinite(confirmation).all()) and len(gate_results) == 6, "scientific_gate_passed": all(gate_results.values()), "authorization": report["authorization"]})
    print(json.dumps(report, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    confirmation = core.load(OUT / "analysis/confirmation.json")
    behavior_report = core.load(OUT / "analysis/behavior_gate.json")
    fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r")
    raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r")
    payload = core.load(PUBLIC)
    nominee = confirmation["nominee"]
    role_index = int(nominee["role_index"])
    effect_rows = []
    profiles = []
    for partition_index, partition in enumerate(PARTITIONS):
        subset = np.asarray(fields[partition_index * 16:(partition_index + 1) * 16, role_index], dtype=np.float32)
        mean = np.mean(subset, axis=0, dtype=np.float32)
        increments = mean[1:] - mean[:-1]
        profiles.append({"partition": partition, "role": nominee["role"], "values": [float(np.linalg.norm(value)) for value in increments]})
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            effect_rows.append({"partition": partition, "role": nominee["role"], "kind": "truth_response", "checkpoint": checkpoint, "checkpoint_index": checkpoint_index, "values": mean[checkpoint_index].tolist()})
        for transition_index, value in enumerate(increments):
            effect_rows.append({"partition": partition, "role": nominee["role"], "kind": "truth_response_increment", "from_checkpoint": CHECKPOINTS[transition_index], "to_checkpoint": CHECKPOINTS[transition_index + 1], "transition_index": transition_index, "values": value.tolist()})
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    representative = []
    for role_index_local, role in enumerate(ROLES):
        for checkpoint_index in (0, 8, 16, 24, 32, 36, 37):
            representative.append({"case_id": compiled[0]["case_id"], "role": role, "checkpoint": CHECKPOINTS[checkpoint_index], "checkpoint_index": checkpoint_index, "token_positions": compiled[0]["role_positions"][role], "values": decode(raw[0, role_index_local, checkpoint_index]).tolist()})
    payload["c127_typed_transition_batch"] = {"protocol": protocol, "behavior": behavior_report, "confirmation": confirmation, "profiles": profiles, "effect_rows": effect_rows, "representative_raw_rows": representative}
    payload.update({"phase": 1661, "campaign": "C109-C117 + C123-C127", "title": "Role-State Atlas + Typed Transition Language-Family Observation", "claim_boundary": "C127 adds a behavior-qualified Qwen3 two-hop precedence family with embedding, every exact post-decoder-block pre-final-norm HiddenState, and the post-final-norm state. Full 2560 activation coordinates are shown; they are not weights, independent semantic neurons, attention/MLP mechanisms, or a complete language operator.", "created_at_utc": now()})
    canonical = OUT / "visualization/c109_c127_typed_transition_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    closure = {"phase": 1661, "campaign": "C127", "created_at_utc": now(), "status": "typed_transition_language_family_closed", "headline": {"behavior": behavior_report["summary"], "nominee": nominee, "confirmation": confirmation}, "new_puzzles": {"K319": "A fresh two-hop precedence family provides a behavior-qualified, uniformly typed embedding-to-block-output-to-final-norm activation trajectory; its frozen discovery transition is reported with independent confirmation rather than treated as a universal operator."}, "theory_update": "Transformation-graph observations now require both an explicit input contrast and an explicit checkpoint type. A graph node is not merely a layer number.", "unified_formula": "Z_0=Embed(x); Z_(l+1)=Block_l(Z_l); Z_37=FinalNorm(Z_36); E_truth(Z)=(1/8)sum_z truth(z)Z(z).", "problems": ["controlled synthetic English", "Qwen3 only", "one new semantic family", "six registered roles rather than all tokens", "no intervention or operator-composition test", "no independent human naturalness audit"], "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "effect_rows": len(effect_rows), "representative_raw_rows": len(representative), "activation_coordinates": DIM}, "claim_boundary": payload["claim_boundary"], "next_authorization": "C128 may continue the same observation-first campaign by adding a different behavior-qualified composition family and comparing typed transition profiles; it must freeze material before model execution and must not call C127's selected transition a universal language operator."}
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"behavior": behavior_report["gate_passed"], "confirmation_integrity": core.load(OUT / "audit/internal_confirmation_audit.json")["all_integrity_checks_passed"], "effect_rows": len(effect_rows) == 150, "raw_rows": len(representative) == 42, "asset": core.sha(canonical) == core.sha(PUBLIC), "boundary": "not weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"]}
    audit = {"phase": 1661, "campaign": "C127", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": confirmation["all_gates_passed"], "asset_sha256": core.sha(PUBLIC), "authorization": "independent_audit_then_append_memo"}
    core.save(OUT / "audit/internal_closure_audit.json", audit)
    print(json.dumps({"audit": audit, "headline": closure["headline"], "next_authorization": closure["next_authorization"]}, indent=2))


def close_behavior_failure() -> None:
    behavior_report = core.load(OUT / "analysis/behavior_gate.json")
    if behavior_report["authorization"] != "close_without_hiddenstate_capture" or behavior_report["gate_passed"]:
        raise RuntimeError("C127 behavior-failure closure not authorized")
    hidden_path = OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy"
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "behavior_integrity": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"],
        "behavior_failed": behavior_report["gate_passed"] is False,
        "hiddenstate_absent": not hidden_path.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    closure = {
        "phase": 1661,
        "campaign": "C127",
        "created_at_utc": now(),
        "status": "closed_at_behavior_gate_without_hiddenstate_capture",
        "headline": behavior_report["summary"],
        "result": "The two-hop precedence material failed the frozen behavior gate, specifically the false cases under the inverted surface. No HiddenState trajectory was captured or tested.",
        "theory_update": "No mechanism update. This is a language-interface qualification failure, not evidence against typed transitions or relative coding.",
        "problems": ["controlled synthetic English", "one inverted surface is behaviorally unreliable", "no human naturalness audit", "Qwen3 only", "no internal data by design"],
        "claim_boundary": "behavior-only Qwen3 result; no embedding, HiddenState, activation-coordinate, operator, causal-path, or language-mechanism conclusion",
        "next_authorization": "C128 may freeze an independent direct-precedence base family with two active-voice natural surfaces. It must qualify behavior before any new HiddenState capture and may not weaken C127 thresholds.",
    }
    core.save(OUT / "analysis/closure.json", closure)
    audit = {"phase": 1661, "campaign": "C127", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": False, "authorization": "independent_behavior_failure_audit_then_append_memo"}
    core.save(OUT / "audit/internal_closure_audit.json", audit)
    print(json.dumps({"audit": audit, "closure": closure}, indent=2))


def main() -> None:
    modes = {"contract": contract, "behavior": behavior, "capture": capture, "discover": discover, "validate": validate, "synthesize": synthesize, "close_behavior_failure": close_behavior_failure}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: {Path(__file__).name} {{{'|'.join(modes)}}}")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
