#!/usr/bin/env python3
"""C130 composed-precedence behavior, typed HiddenState observation, and cross-family adjudication."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1664_c130_composed_precedence_typed_transition"
C129 = RESULT / "phase1663_c129_direct_precedence_typed_transition"
C128 = RESULT / "phase1662_c128_direct_precedence_behavior_qualification"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127

PHASE = 1664
CAMPAIGN = "C130"
FAMILY = "two_hop_precedence"
PARTITIONS = ("discovery", "confirmation")
ROLES = ("source_fact", "bridge_left", "bridge_right", "target_fact", "query_left", "query_right", "boundary")
CHECKPOINTS = c127.CHECKPOINTS
DIM = 2560
WIDTH = 176
BATCH = 8
SUPPORT_K = 256
SYSTEM = "Use only the route record. Answer only yes or no."
SYLLABLES = ("bam", "civ", "duq", "fer", "gop", "hul", "jex", "kim", "lom", "nav", "pyr", "qes", "rov", "sud", "tiv", "wex")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def topk(values: np.ndarray, k: int = SUPPORT_K) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def values_for(index: int) -> tuple[str, str, str, str, str]:
    a = SYLLABLES[index % len(SYLLABLES)]
    b = SYLLABLES[(index * 5 + 7) % len(SYLLABLES)]
    return tuple(f"Centa{a}{b}{index:02d}{suffix}" for suffix in ("a", "b", "c", "d", "e"))


def prompt_for(values: tuple[str, ...], truth: int, surface: int, distractor: int) -> tuple[str, str, str]:
    source, bridge, target, extra_left, extra_right = values
    first_forward = truth == 1 or distractor == -1
    second_forward = truth == 1 or distractor == 1
    first_left, first_right = (source, bridge) if first_forward else (bridge, source)
    second_left, second_right = (bridge, target) if second_forward else (target, bridge)
    if surface == 1:
        first = f"{first_left} comes before {first_right}"
        second = f"{second_left} comes before {second_right}"
        extra_a, extra_b = (extra_left, extra_right) if distractor == 1 else (extra_right, extra_left)
        extra = f"{extra_a} comes before {extra_b}"
    else:
        first = f"{first_left} appears earlier than {first_right}"
        second = f"{second_left} appears earlier than {second_right}"
        extra_a, extra_b = (extra_left, extra_right) if distractor == 1 else (extra_right, extra_left)
        extra = f"{extra_a} appears earlier than {extra_b}"
    query_left, query_right = source, target
    prompt = f"Route rule: a claim is established only when the listed before-links form a directed path. Route record: {first}. Continuation: {second}. Separate record: {extra}. Question: Does the route record establish that {query_left} comes before {query_right}? Reply exactly yes or no."
    return prompt, query_left, query_right


def material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit_index in range(32):
        values = values_for(unit_index)
        partition = PARTITIONS[unit_index // 16]
        unit = {"unit_id": f"c130-{unit_index:02d}", "family": FAMILY, "partition": partition, "world": "controlled_synthetic_two_hop_precedence", "values": list(values)}
        units.append(unit)
        for truth, surface, distractor in itertools.product((1, -1), repeat=3):
            prompt, query_left, query_right = prompt_for(values, truth, surface, distractor)
            cases.append({**unit, "case_id": f"c130-{len(cases):04d}", "truth_factor": truth, "surface_factor": surface, "distractor_factor": distractor, "truth": truth == 1, "output_yes": truth == 1, "gold_position": 0 if truth == 1 else 1, "query_left": query_left, "query_right": query_right, "prompt": prompt})
    return units, cases


def historical_values() -> set[str]:
    result: set[str] = set()
    for path in RESULT.glob("phase*/material/units.jsonl"):
        for row in core.rows(path):
            result.update(str(value).casefold() for value in row.get("values", []))
    return result


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidate_ids = [[int(value) for value in tokenizer.encode(" " + candidate, add_special_tokens=False)] for candidate in ("yes", "no")]
    if any(len(value) != 1 for value in candidate_ids):
        raise RuntimeError(candidate_ids)
    compiled = []
    for row in rows:
        ids = core.chat_ids(tokenizer, SYSTEM, row["prompt"])
        source, bridge, target, _extra_left, _extra_right = row["values"]
        source_spans = graph_base.name_spans(tokenizer, ids, source)
        bridge_spans = graph_base.name_spans(tokenizer, ids, bridge)
        target_spans = graph_base.name_spans(tokenizer, ids, target)
        query_left_spans = graph_base.name_spans(tokenizer, ids, row["query_left"])
        query_right_spans = graph_base.name_spans(tokenizer, ids, row["query_right"])
        if len(source_spans) < 2 or len(bridge_spans) != 2 or len(target_spans) < 2 or len(query_left_spans) < 2 or len(query_right_spans) < 2:
            raise RuntimeError((row["case_id"], source_spans, bridge_spans, target_spans, query_left_spans, query_right_spans))
        roles = {
            "source_fact": source_spans[0],
            "bridge_left": bridge_spans[0],
            "bridge_right": bridge_spans[1],
            "target_fact": target_spans[0],
            "query_left": query_left_spans[-1],
            "query_right": query_right_spans[-1],
            "boundary": [len(ids) - 1],
        }
        fact_end = max(max(roles[name]) for name in ("source_fact", "bridge_left", "bridge_right", "target_fact"))
        query_start = min(min(roles[name]) for name in ("query_left", "query_right"))
        if not fact_end < query_start < min(roles["boundary"]):
            raise RuntimeError(("role_order", row["case_id"], roles))
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids, "role_positions": roles})
    return compiled


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C130 already exists: {OUT}")
    parent_closure = core.load(C129 / "analysis/closure.json")
    parent_audit = core.load(C129 / "audit/independent_closure_audit.json")
    c129_freeze = core.load(C129 / "protocol/frozen_discovery_nomination.json")
    units, cases = material()
    compiled = compile_rows(graph_base.tokenizer(), cases)
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    old = historical_values()
    cells = Counter((row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"]) for row in cases)
    zero = {
        "always_yes": float(np.mean([row["truth"] for row in cases])),
        "always_no": float(np.mean([not row["truth"] for row in cases])),
        "surface_only": float(np.mean([(row["surface_factor"] == 1) == row["truth"] for row in cases])),
        "distractor_only": float(np.mean([(row["distractor_factor"] == 1) == row["truth"] for row in cases])),
        "first_link_only": float(np.mean([((row["truth_factor"] == 1 or row["distractor_factor"] == -1) == row["truth"]) for row in cases])),
        "second_link_only": float(np.mean([((row["truth_factor"] == 1 or row["distractor_factor"] == 1) == row["truth"]) for row in cases])),
    }
    checks = {
        "authorization": parent_audit["all_checks_passed"] and parent_audit["authorization"] == "append_memo_and_consider_c130" and parent_closure["next_authorization"].startswith("C130 may compare"),
        "units": len(units) == 32,
        "cases": len(cases) == 256,
        "factorial": cells == {(partition, *cell): 16 for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=3)},
        "freshness": not (set(fresh) & old) and len(fresh) == len(set(fresh)),
        "unique_prompts": len({row["prompt"] for row in cases}) == 256,
        "zero_models": all(value <= 0.75 for value in zero.values()) and zero["first_link_only"] == 0.75 and zero["second_link_only"] == 0.75,
        "candidate_ids": all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "semantic_uniqueness": all(row["values"][0] != row["values"][2] and row["query_left"] != row["query_right"] for row in cases),
        "machine_naturalness": all(row["prompt"].startswith("Route rule:") and "Route record:" in row["prompt"] and "Continuation:" in row["prompt"] and row["prompt"].endswith("Reply exactly yes or no.") for row in cases),
        "typed_reference": c129_freeze["nominee"]["role"] == "boundary" and c129_freeze["nominee"]["transition_index"] == 35,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    source_paths = {
        "c129_closure": C129 / "analysis/closure.json",
        "c129_audit": C129 / "audit/independent_closure_audit.json",
        "c129_nomination": C129 / "protocol/frozen_discovery_nomination.json",
        "c129_vector": C129 / "analysis/discovery_nominee_increment.float32.npy",
    }
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "two_hop_precedence_cross_family_contract_frozen",
        "object": "behavior-qualified two-hop precedence truth-response trajectory and pre-frozen C129 direct-to-composed comparison",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "units": 32,
        "cases": 256,
        "partitions": list(PARTITIONS),
        "factors": ["truth", "surface", "distractor"],
        "roles": list(ROLES),
        "checkpoints": list(CHECKPOINTS),
        "activation_coordinates": DIM,
        "zero_models": zero,
        "behavior_gate": {"global_accuracy_min": 0.95, "partition_accuracy_min": 0.90, "truth_accuracy_min": 0.90, "surface_accuracy_min": 0.90, "global_margin_over_best_single_link_min": 0.20},
        "discovery_rule": {"partition": "discovery", "unit_split": "first eight versus last eight", "score": "max(0,split_half_cosine)*min(split_half_L2_norms)", "support_k": SUPPORT_K},
        "within_family_confirmation_gates": {"cosine_min": 0.90, "top256_overlap_min": 0.50, "support_sign_agreement_min": 0.75, "coordinate_clock_within_one_min": 0.70, "wrong_state_margin_gt": 0.0, "wrong_role_margin_gt": 0.0},
        "cross_family_frozen_candidate": {"source_family": "C129 direct_precedence", "role": "boundary", "transition_index": 35, "from_checkpoint": CHECKPOINTS[35], "to_checkpoint": CHECKPOINTS[36], "reference_vector_sha256": core.sha(C129 / "analysis/discovery_nominee_increment.float32.npy")},
        "cross_family_confirmation_gates": {"cosine_min": 0.90, "top256_overlap_min": 0.50, "support_sign_agreement_min": 0.75},
        "composition_residual_confirmation_gates": {"cosine_min": 0.90, "top256_overlap_min": 0.50, "support_sign_agreement_min": 0.75, "residual_l2_min": 2.5, "residual_fraction_min": 0.05},
        "residual_rule": "alpha_D=<C_D,D>/<D,D>; U_D=C_D-alpha_D*D; U_C=C_C-alpha_D*D",
        "stop_conditions": ["behavior failure forbids HiddenState capture", "numeric capture failure", "confirmation failures close only their named claim route without reselection or threshold changes"],
        "observation_policy": "full 2560 activation coordinates; no PCA/SVD, attention, MLP, or weight analysis",
        "naturalness_scope": "deterministic machine grammar audit only; no independent human naturalness lock",
        "claim_boundary": "controlled two-hop precedence activation observations; cross-family similarity can indicate shared truth/output preparation and is not by itself a composition operator, semantic neuron, causal path, or new mathematics",
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_hashes": {name: core.sha(path) for name, path in source_paths.items()},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_c130_behavior",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "max_width": max(len(row["prompt_ids"]) for row in compiled), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def group_accuracy(rows: list[dict]) -> float:
    return float(np.mean([row["correct"] for row in rows]))


@torch.inference_mode()
def behavior() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "run_c130_behavior" or not core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C130 behavior authorization missing")
    for name, path in protocol["source_paths"].items():
        if core.sha(Path(path)) != protocol["source_hashes"][name]:
            raise RuntimeError(f"source drift: {name}")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    logits_array = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(rows), 2))
    results, model, repeat = [], None, 0.0
    try:
        model, _tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(_tokenizer.pad_token_id if _tokenizer.pad_token_id is not None else _tokenizer.eos_token_id)

        def score_batch(batch):
            ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
            boundary = torch.stack([output.last_hidden_state[index, length - 1] for index, length in enumerate(lengths)])
            logits = model.lm_head(boundary).float()
            scores = np.asarray([[float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]] for local, row in enumerate(batch)], dtype=np.float32)
            return scores, output, boundary, logits, ids, mask, positions

        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            scores, output, boundary, logits, ids, mask, positions = score_batch(batch)
            logits_array[start:start + len(batch)] = scores
            for local, row in enumerate(batch):
                prediction = int(scores[local, 1] > scores[local, 0])
                results.append({"row_index": start + local, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"], "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "yes_minus_no": float(scores[local, 0] - scores[local, 1])})
            del output, boundary, logits, ids, mask, positions
        logits_array.flush()
        repeat_scores, output, boundary, logits, ids, mask, positions = score_batch(rows[:BATCH])
        repeat = float(np.max(np.abs(repeat_scores - np.asarray(logits_array[:BATCH]))))
    finally:
        logits_array.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    core.write_rows(OUT / "raw/qwen3_behavior_index.jsonl", results)
    summary = {
        "global_accuracy": group_accuracy(results),
        "by_partition": {key: group_accuracy([row for row in results if row["partition"] == key]) for key in PARTITIONS},
        "by_truth": {str(key): group_accuracy([row for row in results if row["truth_factor"] == key]) for key in (1, -1)},
        "by_surface": {str(key): group_accuracy([row for row in results if row["surface_factor"] == key]) for key in (1, -1)},
    }
    gates = protocol["behavior_gate"]
    best_single_link = max(protocol["zero_models"]["first_link_only"], protocol["zero_models"]["second_link_only"])
    summary["margin_over_best_single_link"] = summary["global_accuracy"] - best_single_link
    gate = summary["global_accuracy"] >= gates["global_accuracy_min"] and min(summary["by_partition"].values()) >= gates["partition_accuracy_min"] and min(summary["by_truth"].values()) >= gates["truth_accuracy_min"] and min(summary["by_surface"].values()) >= gates["surface_accuracy_min"] and summary["margin_over_best_single_link"] >= gates["global_margin_over_best_single_link_min"]
    checks = {"rows": len(results) == 256, "finite": bool(np.isfinite(logits_array).all()), "repeat": repeat == 0.0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "behavior_qualified" if gate else "behavior_gate_failed", "checks": checks, "summary": summary, "gate_passed": gate, "repeat_logits_max_abs": repeat, "runtime": {"placement": placement, "quantization": quant}, "authorization": "capture_c130_typed_hiddenstates" if gate else "close_c130_behavior_failed"}
    core.save(OUT / "analysis/behavior_gate.json", report)
    core.save(OUT / "audit/internal_behavior_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "all_integrity_checks_passed": all(checks.values()), "scientific_gate_passed": gate, "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def tensor_output(value):
    return value[0] if isinstance(value, tuple) else value


@torch.inference_mode()
def capture() -> None:
    behavior_report = core.load(OUT / "analysis/behavior_gate.json")
    if behavior_report["authorization"] != "capture_c130_typed_hiddenstates" or not behavior_report["gate_passed"]:
        raise RuntimeError("C130 HiddenState capture is not behavior-authorized")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    raw_path = OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=(len(rows), len(ROLES), len(CHECKPOINTS), DIM))
    repeat, model = 0, None
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
            if len(tensors) != 38 or any(tensor.dtype != torch.bfloat16 for tensor in tensors):
                raise RuntimeError((len(tensors), [str(tensor.dtype) for tensor in tensors]))
            return tensors, output, ids, mask, positions, lengths

        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            tensors, output, ids, mask, positions, lengths = run_batch(batch)
            for local, row in enumerate(batch):
                for role_index, role in enumerate(ROLES):
                    token_positions = row["role_positions"][role]
                    for checkpoint_index, tensor in enumerate(tensors):
                        raw[start + local, role_index, checkpoint_index] = tensor[local, token_positions].mean(dim=0).contiguous().view(torch.uint16).cpu().numpy()
            if (start // BATCH + 1) % 8 == 0:
                raw.flush()
                print(f"[C130] captured {start + len(batch)}/{len(rows)}", flush=True)
            del tensors, output, ids, mask, positions
        raw.flush()
        tensors, output, ids, mask, positions, lengths = run_batch(rows[:BATCH])
        for local, row in enumerate(rows[:BATCH]):
            for role_index, role in enumerate(ROLES):
                for checkpoint_index, tensor in enumerate(tensors):
                    bits = tensor[local, row["role_positions"][role]].mean(dim=0).contiguous().view(torch.uint16).cpu().numpy()
                    repeat = max(repeat, int(np.max(np.abs(bits.astype(np.int64) - raw[local, role_index, checkpoint_index].astype(np.int64)))))
    finally:
        raw.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {"shape": list(raw.shape) == [256, 7, 38, DIM], "finite": bool(np.isfinite(c127.decode(raw[:2])).all()), "repeat_bits": repeat == 0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "typed_hiddenstate_capture_complete", "checks": checks, "shape": list(raw.shape), "sha256": core.sha(raw_path), "runtime": {"placement": placement, "quantization": quant}, "authorization": "discover_and_freeze_c130"}
    core.save(OUT / "analysis/capture_summary.json", report)
    core.save(OUT / "audit/internal_capture_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def compute_fields() -> tuple[np.ndarray, list[dict], list[dict]]:
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = core.rows(OUT / "material/units.jsonl")
    raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r")
    lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    fields = np.zeros((len(units), len(ROLES), len(CHECKPOINTS), DIM), dtype=np.float32)
    for row_index, row in enumerate(rows):
        fields[lookup[row["unit_id"]]] += float(row["truth_factor"]) / 8.0 * c127.decode(raw[row_index])
    return fields, rows, units


def discover() -> None:
    if core.load(OUT / "analysis/capture_summary.json")["authorization"] != "discover_and_freeze_c130":
        raise RuntimeError("C130 discovery authorization missing")
    fields, _rows, units = compute_fields()
    np.save(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", fields)
    discovery = fields[:16]
    increments = discovery[:, :, 1:] - discovery[:, :, :-1]
    left = np.mean(increments[:8], axis=0, dtype=np.float32)
    right = np.mean(increments[8:], axis=0, dtype=np.float32)
    candidates = []
    for role_index, role in enumerate(ROLES):
        for transition_index in range(37):
            similarity = cosine(left[role_index, transition_index], right[role_index, transition_index])
            left_norm = float(np.linalg.norm(left[role_index, transition_index]))
            right_norm = float(np.linalg.norm(right[role_index, transition_index]))
            candidates.append({"role": role, "role_index": role_index, "transition_index": transition_index, "from_checkpoint": CHECKPOINTS[transition_index], "to_checkpoint": CHECKPOINTS[transition_index + 1], "split_half_cosine": similarity, "left_norm": left_norm, "right_norm": right_norm, "score": max(0.0, similarity) * min(left_norm, right_norm)})
    candidates.sort(key=lambda row: (-row["score"], -row["split_half_cosine"], row["transition_index"], ROLES.index(row["role"])))
    nominee = dict(candidates[0])
    nominee_vector = np.mean(increments[:, nominee["role_index"], nominee["transition_index"]], axis=0, dtype=np.float32)
    nominee["support_k"] = SUPPORT_K
    nominee["support"] = sorted(topk(nominee_vector), key=lambda index: -abs(float(nominee_vector[index])))

    direct = np.load(C129 / "analysis/discovery_nominee_increment.float32.npy").astype(np.float32)
    fixed_role = ROLES.index("boundary")
    fixed_transition = 35
    composed = np.mean(increments[:, fixed_role, fixed_transition], axis=0, dtype=np.float32)
    alpha = float(np.dot(composed, direct) / max(float(np.dot(direct, direct)), 1e-12))
    residual = composed - alpha * direct
    cross = {
        "role": "boundary",
        "role_index": fixed_role,
        "transition_index": fixed_transition,
        "from_checkpoint": CHECKPOINTS[fixed_transition],
        "to_checkpoint": CHECKPOINTS[fixed_transition + 1],
        "direct_to_composed_discovery_cosine": cosine(direct, composed),
        "direct_to_composed_top256_overlap": len(topk(direct) & topk(composed)) / SUPPORT_K,
        "direct_to_composed_support_sign_agreement": float(np.mean([np.sign(direct[index]) == np.sign(composed[index]) for index in topk(direct)])),
        "alpha_discovery": alpha,
        "direct_l2": float(np.linalg.norm(direct)),
        "composed_l2": float(np.linalg.norm(composed)),
        "residual_l2": float(np.linalg.norm(residual)),
        "residual_fraction": float(np.linalg.norm(residual) / max(np.linalg.norm(composed), 1e-12)),
        "residual_support": sorted(topk(residual), key=lambda index: -abs(float(residual[index]))),
    }
    core.write_rows(OUT / "analysis/discovery_candidate_table.jsonl", candidates)
    np.save(OUT / "analysis/discovery_nominee_increment.float32.npy", nominee_vector)
    np.save(OUT / "analysis/c129_direct_reference_increment.float32.npy", direct)
    np.save(OUT / "analysis/discovery_composed_fixed_increment.float32.npy", composed)
    np.save(OUT / "analysis/discovery_composition_residual.float32.npy", residual)
    freeze = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "discovery_and_cross_family_residual_frozen", "within_family_nominee": nominee, "cross_family": cross, "candidate_count": len(candidates), "confirmation_partition_unread": True, "authorization": "validate_c130_confirmation"}
    core.save(OUT / "protocol/frozen_discovery_nomination.json", freeze)
    checks = {"shape": list(fields.shape) == [32, 7, 38, DIM], "finite": bool(np.isfinite(fields).all()), "candidates": len(candidates) == 259, "nominee_support": len(nominee["support"]) == SUPPORT_K, "residual_support": len(cross["residual_support"]) == SUPPORT_K, "reference_hash": core.sha(C129 / "analysis/discovery_nominee_increment.float32.npy") == core.load(OUT / "protocol/preregistration.json")["cross_family_frozen_candidate"]["reference_vector_sha256"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "audit/internal_discovery_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": freeze["authorization"]})
    print(json.dumps(freeze, indent=2))


def comparison_metrics(reference: np.ndarray, target: np.ndarray, support: set[int]) -> dict:
    return {
        "cosine": cosine(reference, target),
        "top256_overlap": len(topk(target) & support) / SUPPORT_K,
        "support_sign_agreement": float(np.mean([np.sign(reference[index]) == np.sign(target[index]) for index in support])),
        "reference_l2": float(np.linalg.norm(reference)),
        "target_l2": float(np.linalg.norm(target)),
    }


def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    freeze = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    if freeze["authorization"] != "validate_c130_confirmation" or not core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"]:
        raise RuntimeError("C130 confirmation authorization missing")
    fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r")
    discovery = np.asarray(fields[:16], dtype=np.float32)
    confirmation = np.asarray(fields[16:], dtype=np.float32)
    discovery_increments = discovery[:, :, 1:] - discovery[:, :, :-1]
    confirmation_increments = confirmation[:, :, 1:] - confirmation[:, :, :-1]

    nominee = freeze["within_family_nominee"]
    role_index = int(nominee["role_index"])
    transition_index = int(nominee["transition_index"])
    discovery_vector = np.load(OUT / "analysis/discovery_nominee_increment.float32.npy")
    confirmation_vector = np.mean(confirmation_increments[:, role_index, transition_index], axis=0, dtype=np.float32)
    within = comparison_metrics(discovery_vector, confirmation_vector, set(nominee["support"]))
    wrong_state = max(cosine(discovery_vector, np.mean(confirmation_increments[:, role_index, index], axis=0, dtype=np.float32)) for index in range(37) if index != transition_index)
    wrong_role = max(cosine(discovery_vector, np.mean(confirmation_increments[:, index, transition_index], axis=0, dtype=np.float32)) for index in range(len(ROLES)) if index != role_index)
    discovery_peaks = np.argmax(np.abs(np.mean(discovery_increments[:, role_index], axis=0, dtype=np.float32)), axis=0)
    confirmation_peaks = np.argmax(np.abs(np.mean(confirmation_increments[:, role_index], axis=0, dtype=np.float32)), axis=0)
    support_list = list(set(nominee["support"]))
    within["coordinate_clock_within_one"] = float(np.mean(np.abs(discovery_peaks[support_list] - confirmation_peaks[support_list]) <= 1))
    within["best_wrong_state_cosine"] = wrong_state
    within["best_wrong_role_cosine"] = wrong_role
    within["state_margin"] = within["cosine"] - wrong_state
    within["role_margin"] = within["cosine"] - wrong_role
    gates = protocol["within_family_confirmation_gates"]
    within_gate = {
        "cosine": within["cosine"] >= gates["cosine_min"],
        "top256_overlap": within["top256_overlap"] >= gates["top256_overlap_min"],
        "sign_agreement": within["support_sign_agreement"] >= gates["support_sign_agreement_min"],
        "coordinate_clock": within["coordinate_clock_within_one"] >= gates["coordinate_clock_within_one_min"],
        "wrong_state": within["state_margin"] > gates["wrong_state_margin_gt"],
        "wrong_role": within["role_margin"] > gates["wrong_role_margin_gt"],
    }

    direct = np.load(OUT / "analysis/c129_direct_reference_increment.float32.npy")
    fixed_role, fixed_transition = ROLES.index("boundary"), 35
    composed_confirmation = np.mean(confirmation_increments[:, fixed_role, fixed_transition], axis=0, dtype=np.float32)
    cross = comparison_metrics(direct, composed_confirmation, topk(direct))
    cross_gates = protocol["cross_family_confirmation_gates"]
    cross_gate = {"cosine": cross["cosine"] >= cross_gates["cosine_min"], "top256_overlap": cross["top256_overlap"] >= cross_gates["top256_overlap_min"], "sign_agreement": cross["support_sign_agreement"] >= cross_gates["support_sign_agreement_min"]}

    residual_discovery = np.load(OUT / "analysis/discovery_composition_residual.float32.npy")
    alpha = float(freeze["cross_family"]["alpha_discovery"])
    residual_confirmation = composed_confirmation - alpha * direct
    residual = comparison_metrics(residual_discovery, residual_confirmation, set(freeze["cross_family"]["residual_support"]))
    residual["discovery_fraction"] = float(np.linalg.norm(residual_discovery) / max(np.linalg.norm(np.load(OUT / "analysis/discovery_composed_fixed_increment.float32.npy")), 1e-12))
    residual["confirmation_fraction"] = float(np.linalg.norm(residual_confirmation) / max(np.linalg.norm(composed_confirmation), 1e-12))
    residual_gates = protocol["composition_residual_confirmation_gates"]
    residual_gate = {
        "cosine": residual["cosine"] >= residual_gates["cosine_min"],
        "top256_overlap": residual["top256_overlap"] >= residual_gates["top256_overlap_min"],
        "sign_agreement": residual["support_sign_agreement"] >= residual_gates["support_sign_agreement_min"],
        "discovery_l2": residual["reference_l2"] >= residual_gates["residual_l2_min"],
        "confirmation_l2": residual["target_l2"] >= residual_gates["residual_l2_min"],
        "discovery_fraction": residual["discovery_fraction"] >= residual_gates["residual_fraction_min"],
        "confirmation_fraction": residual["confirmation_fraction"] >= residual_gates["residual_fraction_min"],
    }
    np.save(OUT / "analysis/confirmation_composed_fixed_increment.float32.npy", composed_confirmation)
    np.save(OUT / "analysis/confirmation_composition_residual.float32.npy", residual_confirmation)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "confirmation_adjudicated",
        "within_family_nominee": nominee,
        "within_family": {"metrics": within, "gates": within_gate, "all_gates_passed": all(within_gate.values())},
        "cross_family_common_response": {"metrics": cross, "gates": cross_gate, "all_gates_passed": all(cross_gate.values())},
        "composition_residual": {"alpha_frozen_on_discovery": alpha, "metrics": residual, "gates": residual_gate, "all_gates_passed": all(residual_gate.values())},
        "interpretation_rule": "cross-family success alone supports only a shared late truth/output response; a repeatable residual is required even to nominate composition-specific structure",
        "authorization": "synthesize_audit_and_close_c130",
    }
    core.save(OUT / "analysis/confirmation.json", report)
    integrity = {"confirmation_units": confirmation.shape[0] == 16, "finite": bool(np.isfinite(confirmation).all()), "within_gate_count": len(within_gate) == 6, "cross_gate_count": len(cross_gate) == 3, "residual_gate_count": len(residual_gate) == 7}
    core.save(OUT / "audit/internal_confirmation_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "integrity": integrity, "all_integrity_checks_passed": all(integrity.values()), "scientific_gates": {"within_family": all(within_gate.values()), "cross_family_common_response": all(cross_gate.values()), "composition_residual": all(residual_gate.values())}, "authorization": report["authorization"]})
    print(json.dumps(report, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior_report = core.load(OUT / "analysis/behavior_gate.json")
    confirmation = core.load(OUT / "analysis/confirmation.json")
    fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r")
    raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r")
    payload = core.load(PUBLIC)
    nominee = confirmation["within_family_nominee"]
    role_index = int(nominee["role_index"])
    effect_rows, profiles = [], []
    for partition_index, partition in enumerate(PARTITIONS):
        mean = np.mean(np.asarray(fields[partition_index * 16:(partition_index + 1) * 16, role_index], dtype=np.float32), axis=0, dtype=np.float32)
        increments = mean[1:] - mean[:-1]
        profiles.append({"partition": partition, "role": nominee["role"], "values": [float(np.linalg.norm(value)) for value in increments]})
        effect_rows.extend({"partition": partition, "role": nominee["role"], "kind": "truth_response", "checkpoint": CHECKPOINTS[index], "checkpoint_index": index, "values": mean[index].tolist()} for index in range(38))
        effect_rows.extend({"partition": partition, "role": nominee["role"], "kind": "truth_response_increment", "from_checkpoint": CHECKPOINTS[index], "to_checkpoint": CHECKPOINTS[index + 1], "transition_index": index, "values": increments[index].tolist()} for index in range(37))
    residual_rows = []
    for label, path in (
        ("c129_direct_reference", OUT / "analysis/c129_direct_reference_increment.float32.npy"),
        ("c130_composed_discovery", OUT / "analysis/discovery_composed_fixed_increment.float32.npy"),
        ("c130_composed_confirmation", OUT / "analysis/confirmation_composed_fixed_increment.float32.npy"),
        ("composition_residual_discovery", OUT / "analysis/discovery_composition_residual.float32.npy"),
        ("composition_residual_confirmation", OUT / "analysis/confirmation_composition_residual.float32.npy"),
    ):
        residual_rows.append({"label": label, "role": "boundary", "transition_index": 35, "values": np.load(path).astype(np.float32).tolist()})
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    representative = []
    for local_role_index, role in enumerate(ROLES):
        for checkpoint_index in (0, 8, 16, 24, 32, 36, 37):
            representative.append({"case_id": compiled[0]["case_id"], "role": role, "checkpoint": CHECKPOINTS[checkpoint_index], "checkpoint_index": checkpoint_index, "token_positions": compiled[0]["role_positions"][role], "values": c127.decode(raw[0, local_role_index, checkpoint_index]).tolist()})
    payload["c130_composed_precedence_typed_transition_batch"] = {"protocol": protocol, "behavior": behavior_report["summary"], "confirmation": confirmation, "profiles": profiles, "effect_rows": effect_rows, "cross_family_and_residual_rows": residual_rows, "representative_raw_rows": representative}
    payload.update({"phase": PHASE, "campaign": "C109-C130", "title": "Role-State Atlas + Direct/Composed Precedence Typed Response", "claim_boundary": "C130 adds full-coordinate Qwen3 two-hop precedence observations and a frozen C129 direct-to-composed comparison. Similar late truth responses are not a language composition operator; residuals remain activation observations rather than weights or causal paths."})
    core.save(PUBLIC, payload)
    heatmap = {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "effect_rows": len(effect_rows), "residual_rows": len(residual_rows), "representative_raw_rows": len(representative), "activation_coordinates": DIM}
    within_pass = confirmation["within_family"]["all_gates_passed"]
    cross_pass = confirmation["cross_family_common_response"]["all_gates_passed"]
    residual_pass = confirmation["composition_residual"]["all_gates_passed"]
    new_puzzles = {
        "K320": "A separately behavior-qualified two-hop precedence family has a frozen full-coordinate typed response trajectory; its own discovery nominee is adjudicated on untouched fresh vocabulary.",
        "K321": "The pre-frozen C129 direct-precedence late boundary increment is compared prospectively with C130 composition. Any passing similarity is typed as shared truth/output preparation, not a universal relation operator.",
    }
    if residual_pass:
        new_puzzles["K322_candidate"] = "After subtracting the frozen scalar C129 direct component, a nontrivial C130 composition residual repeats across lexical partitions. It is a composition-specific activation candidate, not yet a causal operator."
    closure = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "two_hop_precedence_typed_cross_family_stage_closed",
        "headline": {"behavior": behavior_report["summary"], "within_family_passed": within_pass, "cross_family_common_response_passed": cross_pass, "composition_residual_passed": residual_pass},
        "results": confirmation,
        "new_puzzles": new_puzzles,
        "theory_update": "Typed HiddenState transition comparison now separates a shared direct/composed late truth-output component from a separately tested composition residual. Neither object is promoted to an operator without intervention and broader language-family evidence.",
        "unified_formula": "D=DeltaE_truth^direct(boundary,block34->35); C=DeltaE_truth^composed(boundary,block34->35); alpha=<C_D,D>/<D,D>; U=C-alpha D.",
        "problems": ["controlled synthetic English", "Qwen3 only", "truth remains aligned with yes/no output polarity", "registered semantic roles rather than every token", "machine naturalness audit only", "no intervention, attention, MLP, or weight evidence"],
        "claim_boundary": protocol["claim_boundary"],
        "heatmap": heatmap,
        "next_authorization": "C131 may test the same frozen direct/common component and any confirmed composition residual on a new behavior-qualified relation family; otherwise it must treat the failed branch as closed and select a different pre-registered observation family.",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "behavior": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"], "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_integrity_checks_passed"], "heatmap": heatmap["effect_rows"] == 150 and heatmap["residual_rows"] == 5 and heatmap["representative_raw_rows"] == 49}
    core.save(OUT / "audit/internal_closure_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gates": {"behavior": behavior_report["gate_passed"], "within_family": within_pass, "cross_family_common_response": cross_pass, "composition_residual": residual_pass}, "asset_sha256": heatmap["sha256"], "authorization": "run_independent_c130_audit_and_integrate_client"})
    print(json.dumps(closure, indent=2))


def close_behavior_failure() -> None:
    behavior_report = core.load(OUT / "analysis/behavior_gate.json")
    if behavior_report["gate_passed"] or behavior_report["authorization"] != "close_c130_behavior_failed":
        raise RuntimeError("C130 is not on the frozen behavior-failure branch")
    results = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    cells = {}
    for truth in (1, -1):
        for surface in (1, -1):
            for distractor in (1, -1):
                subset = [row for row in results if row["truth_factor"] == truth and row["surface_factor"] == surface and row["distractor_factor"] == distractor]
                cells[f"truth={truth},surface={surface},distractor={distractor}"] = group_accuracy(subset)
    closure = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "two_hop_precedence_behavior_route_closed",
        "headline": behavior_report["summary"],
        "cell_accuracy": cells,
        "result": "The frozen two-surface C130 interface failed behavior qualification before HiddenState capture.",
        "theory_update": "No HiddenState or mechanism update. The failure is localized to the registered behavior interface, especially false cases under the appears-earlier surface.",
        "problems": ["controlled synthetic English", "appears-earlier surface underperforms", "Qwen3 only", "machine naturalness audit only"],
        "claim_boundary": "behavior qualification only; C130 did not capture embeddings or HiddenStates and did not test C129 transfer, composition residuals, an operator, or a causal mechanism",
        "heatmap": None,
        "next_authorization": "C131 may freeze a fresh two-hop precedence interface that keeps the already qualified comes-before relation wording while varying only discourse framing; it must retain the 0.75 single-link null and the original behavior thresholds.",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "independent_contract": core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"],
        "behavior_integrity": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"],
        "behavior_failed": not behavior_report["gate_passed"],
        "no_hiddenstate": not (OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy").exists(),
        "no_confirmation": not (OUT / "analysis/confirmation.json").exists(),
    }
    core.save(OUT / "audit/internal_closure_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": False, "authorization": "independent_failure_audit_then_start_c131"})
    print(json.dumps(closure, indent=2))


def main() -> None:
    modes = {"contract": contract, "behavior": behavior, "capture": capture, "discover": discover, "validate": validate, "synthesize": synthesize, "close-failure": close_behavior_failure}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: {Path(__file__).name} {{{'|'.join(modes)}}}")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
