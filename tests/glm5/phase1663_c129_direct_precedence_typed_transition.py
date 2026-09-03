#!/usr/bin/env python3
"""C129 uniformly typed HiddenState transition discovery and confirmation."""
from __future__ import annotations

import gc
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1663_c129_direct_precedence_typed_transition"
C128 = RESULT / "phase1662_c128_direct_precedence_behavior_qualification"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127

ROLES = ("left_fact", "right_fact", "query_left", "query_right", "boundary")
CHECKPOINTS = c127.CHECKPOINTS
DIM = 2560
WIDTH = 144
BATCH = 8
SUPPORT_K = 256


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def topk(values: np.ndarray, k: int = SUPPORT_K) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C129 already exists: {OUT}")
    parent_audit = core.load(C128 / "audit/independent_closure_audit.json")
    parent_closure = core.load(C128 / "analysis/closure.json")
    behavior = core.load(C128 / "analysis/behavior_gate.json")
    compiled = core.rows(C128 / "compiled/qwen3.jsonl")
    checks = {
        "authorization": parent_audit["all_checks_passed"] and parent_audit["authorization"] == "start_c129" and parent_closure["next_authorization"].startswith("C129 may capture"),
        "behavior": behavior["gate_passed"] and behavior["summary"]["global_accuracy"] == 1.0,
        "rows": len(compiled) == 256,
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "checkpoints": len(CHECKPOINTS) == 38 and CHECKPOINTS[0] == "embedding" and CHECKPOINTS[-1] == "post_final_norm",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    paths = {"compiled": C128 / "compiled/qwen3.jsonl", "units": C128 / "material/units.jsonl", "behavior": C128 / "analysis/behavior_gate.json", "parent_closure": C128 / "analysis/closure.json", "parent_audit": C128 / "audit/independent_closure_audit.json"}
    protocol = {
        "phase": 1663,
        "campaign": "C129",
        "created_at_utc": now(),
        "status": "behavior_qualified_direct_precedence_typed_transition_contract_frozen",
        "object": "balanced-truth response trajectory on exact embedding, post-decoder-block pre-final-norm, and post-final-norm checkpoints",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "source_behavior_accuracy": behavior["summary"],
        "roles": list(ROLES),
        "checkpoints": list(CHECKPOINTS),
        "activation_coordinates": DIM,
        "discovery_rule": {"partition": "discovery", "unit_split": "first eight versus last eight", "score": "max(0,split_half_cosine)*min(split_half_L2_norms)", "support_k": SUPPORT_K},
        "confirmation_gates": {"cosine_min": 0.90, "top256_overlap_min": 0.50, "support_sign_agreement_min": 0.75, "coordinate_clock_within_one_min": 0.70, "wrong_state_margin_gt": 0.0, "wrong_role_margin_gt": 0.0},
        "stop_conditions": ["numeric capture failure", "confirmation failure closes route without reselection or threshold changes"],
        "observation_policy": "full 2560 activation coordinates; no PCA/SVD, attention, MLP, or weight analysis",
        "claim_boundary": "controlled direct-precedence activation responses, not weights, semantic neurons, a universal relation operator, a complete-token causal graph, topology, or new mathematics",
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_hashes": {name: core.sha(path) for name, path in paths.items()},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "capture_c129_uniform_typed_hiddenstates",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1663, "campaign": "C129", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def tensor_output(value):
    return value[0] if isinstance(value, tuple) else value


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "capture_c129_uniform_typed_hiddenstates" or not core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C129 capture authorization missing")
    for name, path in protocol["source_paths"].items():
        if core.sha(Path(path)) != protocol["source_hashes"][name]:
            raise RuntimeError(f"source drift: {name}")
    rows = core.rows(C128 / "compiled/qwen3.jsonl")
    raw_path = OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=(len(rows), len(ROLES), len(CHECKPOINTS), DIM))
    repeat = 0
    model = None
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
                print(f"[C129] captured {start + len(batch)}/{len(rows)}", flush=True)
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
    checks = {"shape": list(raw.shape) == [256, 5, 38, DIM], "finite": bool(np.isfinite(c127.decode(raw[:2])).all()), "repeat_bits": repeat == 0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1663, "campaign": "C129", "created_at_utc": now(), "status": "uniform_typed_hiddenstate_capture_complete", "checks": checks, "shape": list(raw.shape), "sha256": core.sha(raw_path), "runtime": {"placement": placement, "quantization": quant}, "authorization": "discover_c129_transition"}
    core.save(OUT / "analysis/capture_summary.json", report)
    core.save(OUT / "audit/internal_capture_audit.json", {"phase": 1663, "campaign": "C129", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def compute_fields() -> tuple[np.ndarray, list[dict], list[dict]]:
    rows = core.rows(C128 / "compiled/qwen3.jsonl")
    units = core.rows(C128 / "material/units.jsonl")
    raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r")
    lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    fields = np.zeros((len(units), len(ROLES), len(CHECKPOINTS), DIM), dtype=np.float32)
    for row_index, row in enumerate(rows):
        fields[lookup[row["unit_id"]]] += float(row["truth_factor"]) / 8.0 * c127.decode(raw[row_index])
    return fields, rows, units


def discover() -> None:
    if core.load(OUT / "analysis/capture_summary.json")["authorization"] != "discover_c129_transition":
        raise RuntimeError("C129 discovery authorization missing")
    fields, _rows, units = compute_fields()
    np.save(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", fields)
    discovery = fields[[index for index, row in enumerate(units) if row["partition"] == "discovery"]]
    increments = discovery[:, :, 1:] - discovery[:, :, :-1]
    left = np.mean(increments[:8], axis=0, dtype=np.float32)
    right = np.mean(increments[8:], axis=0, dtype=np.float32)
    candidates = []
    for role_index, role in enumerate(ROLES):
        for transition_index in range(37):
            similarity = cosine(left[role_index, transition_index], right[role_index, transition_index])
            left_norm = float(np.linalg.norm(left[role_index, transition_index])); right_norm = float(np.linalg.norm(right[role_index, transition_index]))
            candidates.append({"role": role, "role_index": role_index, "transition_index": transition_index, "from_checkpoint": CHECKPOINTS[transition_index], "to_checkpoint": CHECKPOINTS[transition_index + 1], "split_half_cosine": similarity, "left_norm": left_norm, "right_norm": right_norm, "score": max(0.0, similarity) * min(left_norm, right_norm)})
    candidates.sort(key=lambda row: (-row["score"], -row["split_half_cosine"], row["transition_index"], ROLES.index(row["role"])))
    nominee = dict(candidates[0])
    vector = np.mean(increments[:, nominee["role_index"], nominee["transition_index"]], axis=0, dtype=np.float32)
    nominee["support_k"] = SUPPORT_K
    nominee["support"] = sorted(topk(vector), key=lambda index: -abs(float(vector[index])))
    core.write_rows(OUT / "analysis/discovery_candidate_table.jsonl", candidates)
    np.save(OUT / "analysis/discovery_nominee_increment.float32.npy", vector)
    freeze = {"phase": 1663, "campaign": "C129", "created_at_utc": now(), "status": "discovery_nomination_frozen", "nominee": nominee, "candidate_count": len(candidates), "confirmation_partition_unread": True, "authorization": "validate_c129_confirmation"}
    core.save(OUT / "protocol/frozen_discovery_nomination.json", freeze)
    checks = {"shape": list(fields.shape) == [32, 5, 38, DIM], "finite": bool(np.isfinite(fields).all()), "candidates": len(candidates) == 185, "support": len(nominee["support"]) == SUPPORT_K}
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "audit/internal_discovery_audit.json", {"phase": 1663, "campaign": "C129", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": freeze["authorization"]})
    print(json.dumps(freeze, indent=2))


def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    freeze = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    if freeze["authorization"] != "validate_c129_confirmation" or not core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"]:
        raise RuntimeError("C129 confirmation authorization missing")
    fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r")
    discovery = np.asarray(fields[:16], dtype=np.float32); confirmation = np.asarray(fields[16:], dtype=np.float32)
    discovery_increments = discovery[:, :, 1:] - discovery[:, :, :-1]; confirmation_increments = confirmation[:, :, 1:] - confirmation[:, :, :-1]
    nominee = freeze["nominee"]; role_index = int(nominee["role_index"]); transition_index = int(nominee["transition_index"])
    discovery_vector = np.load(OUT / "analysis/discovery_nominee_increment.float32.npy")
    confirmation_vector = np.mean(confirmation_increments[:, role_index, transition_index], axis=0, dtype=np.float32)
    target = cosine(discovery_vector, confirmation_vector); support = set(nominee["support"])
    overlap = len(topk(confirmation_vector) & support) / SUPPORT_K
    sign = float(np.mean([np.sign(discovery_vector[index]) == np.sign(confirmation_vector[index]) for index in support]))
    wrong_state = max(cosine(discovery_vector, np.mean(confirmation_increments[:, role_index, index], axis=0, dtype=np.float32)) for index in range(37) if index != transition_index)
    wrong_role = max(cosine(discovery_vector, np.mean(confirmation_increments[:, index, transition_index], axis=0, dtype=np.float32)) for index in range(len(ROLES)) if index != role_index)
    discovery_peaks = np.argmax(np.abs(np.mean(discovery_increments[:, role_index], axis=0, dtype=np.float32)), axis=0)
    confirmation_peaks = np.argmax(np.abs(np.mean(confirmation_increments[:, role_index], axis=0, dtype=np.float32)), axis=0)
    support_list = list(support); clock = float(np.mean(np.abs(discovery_peaks[support_list] - confirmation_peaks[support_list]) <= 1))
    gates = protocol["confirmation_gates"]
    gate_results = {"cosine": target >= gates["cosine_min"], "top256_overlap": overlap >= gates["top256_overlap_min"], "sign_agreement": sign >= gates["support_sign_agreement_min"], "coordinate_clock": clock >= gates["coordinate_clock_within_one_min"], "wrong_state": target - wrong_state > gates["wrong_state_margin_gt"], "wrong_role": target - wrong_role > gates["wrong_role_margin_gt"]}
    report = {"phase": 1663, "campaign": "C129", "created_at_utc": now(), "status": "confirmation_passed" if all(gate_results.values()) else "confirmation_failed", "nominee": nominee, "metrics": {"target_cosine": target, "top256_overlap": overlap, "support_sign_agreement": sign, "coordinate_clock_within_one": clock, "best_wrong_state_cosine": wrong_state, "best_wrong_role_cosine": wrong_role, "state_margin": target - wrong_state, "role_margin": target - wrong_role}, "gates": gate_results, "all_gates_passed": all(gate_results.values()), "authorization": "synthesize_c129_and_close"}
    core.save(OUT / "analysis/confirmation.json", report)
    integrity = {"confirmation_units": confirmation.shape[0] == 16, "finite": bool(np.isfinite(confirmation).all()), "gate_count": len(gate_results) == 6}
    core.save(OUT / "audit/internal_confirmation_audit.json", {"phase": 1663, "campaign": "C129", "integrity": integrity, "all_integrity_checks_passed": all(integrity.values()), "scientific_gate_passed": all(gate_results.values()), "authorization": report["authorization"]})
    print(json.dumps(report, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); confirmation = core.load(OUT / "analysis/confirmation.json")
    fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r"); raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r")
    payload = core.load(PUBLIC); nominee = confirmation["nominee"]; role_index = int(nominee["role_index"])
    effect_rows = []; profiles = []
    for partition_index, partition in enumerate(("discovery", "confirmation")):
        mean = np.mean(np.asarray(fields[partition_index * 16:(partition_index + 1) * 16, role_index], dtype=np.float32), axis=0, dtype=np.float32)
        increments = mean[1:] - mean[:-1]
        profiles.append({"partition": partition, "role": nominee["role"], "values": [float(np.linalg.norm(value)) for value in increments]})
        effect_rows.extend({"partition": partition, "role": nominee["role"], "kind": "truth_response", "checkpoint": CHECKPOINTS[index], "checkpoint_index": index, "values": mean[index].tolist()} for index in range(38))
        effect_rows.extend({"partition": partition, "role": nominee["role"], "kind": "truth_response_increment", "from_checkpoint": CHECKPOINTS[index], "to_checkpoint": CHECKPOINTS[index + 1], "transition_index": index, "values": increments[index].tolist()} for index in range(37))
    compiled = core.rows(C128 / "compiled/qwen3.jsonl"); representative = []
    for local_role_index, role in enumerate(ROLES):
        for checkpoint_index in (0, 8, 16, 24, 32, 36, 37):
            representative.append({"case_id": compiled[0]["case_id"], "role": role, "checkpoint": CHECKPOINTS[checkpoint_index], "checkpoint_index": checkpoint_index, "token_positions": compiled[0]["role_positions"][role], "values": c127.decode(raw[0, local_role_index, checkpoint_index]).tolist()})
    payload["c129_direct_precedence_typed_transition_batch"] = {"protocol": protocol, "behavior": protocol["source_behavior_accuracy"], "confirmation": confirmation, "profiles": profiles, "effect_rows": effect_rows, "representative_raw_rows": representative}
    payload.update({"phase": 1663, "campaign": "C109-C117 + C123-C126 + C129", "title": "Role-State Atlas + Behavior-Qualified Typed Transition Observation", "claim_boundary": "C129 adds a behavior-qualified direct-precedence Qwen3 family with exact embedding, all post-decoder-block pre-final-norm HiddenStates, and post-final-norm states. Full 2560 activation coordinates are shown; they are not weights, independent semantic neurons, attention/MLP mechanisms, or a universal language operator.", "created_at_utc": now()})
    canonical = OUT / "visualization/c109_c129_direct_precedence_typed_transition_atlas.json"; core.save(canonical, payload); shutil.copyfile(canonical, PUBLIC)
    closure = {"phase": 1663, "campaign": "C129", "created_at_utc": now(), "status": "direct_precedence_typed_transition_closed", "headline": {"behavior": protocol["source_behavior_accuracy"], "nominee": nominee, "confirmation": confirmation}, "new_puzzles": {"K319": "A behavior-qualified direct-precedence family now has a fully typed embedding-to-post-block-to-final-norm response trajectory. Its discovery-selected transition is evaluated on an untouched lexical partition and remains a family-specific response candidate rather than a universal operator."}, "theory_update": "The transformation atlas now uses nodes typed by exact computational checkpoint and edges typed by controlled truth contrast. This is a stricter empirical object than an untyped list of hidden_states.", "unified_formula": "Z_0=Embed(x); Z_(l+1)=Block_l(Z_l); Z_37=FinalNorm(Z_36); E_truth(Z_j)=(1/8)sum_z truth(z)Z_j(z); DeltaE_j=E_truth(Z_(j+1))-E_truth(Z_j).", "problems": ["controlled direct one-hop English", "Qwen3 only", "five registered roles rather than all tokens", "no intervention or composition test", "no independent human naturalness audit"], "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "effect_rows": len(effect_rows), "representative_raw_rows": len(representative), "activation_coordinates": DIM}, "claim_boundary": payload["claim_boundary"], "next_authorization": "C130 may compare the C129 typed trajectory with a separately behavior-qualified composition family. It must preserve checkpoint types and cannot infer a universal operator from coordinate similarity alone."}
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"confirmation_integrity": core.load(OUT / "audit/internal_confirmation_audit.json")["all_integrity_checks_passed"], "effect_rows": len(effect_rows) == 150, "raw_rows": len(representative) == 35, "asset": core.sha(canonical) == core.sha(PUBLIC), "boundary": "not weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"]}
    audit = {"phase": 1663, "campaign": "C129", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": confirmation["all_gates_passed"], "asset_sha256": core.sha(PUBLIC), "authorization": "independent_audit_then_append_memo"}
    core.save(OUT / "audit/internal_closure_audit.json", audit)
    print(json.dumps({"audit": audit, "headline": closure["headline"], "next_authorization": closure["next_authorization"]}, indent=2))


def main() -> None:
    modes = {"contract": contract, "capture": capture, "discover": discover, "validate": validate, "synthesize": synthesize}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: {Path(__file__).name} {{{'|'.join(modes)}}}")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
