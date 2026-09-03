#!/usr/bin/env python3
"""C125 fresh sixth-lexicon decomposition of the final recorded HiddenState transition."""
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
OUT = RESULT / "phase1659_c125_final_transition_decomposition"
C123 = RESULT / "phase1657_c123_role_transition_atlas_discovery"
C124 = RESULT / "phase1658_c124_role_transition_validation"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base
import phase1608_c110_exact_field_capture as capture_base
import phase1657_c123_role_transition_atlas as c123

FAMILIES = ("attribute_binding", "agent_patient")
PARTITION = "sixth_prospective"
ROLES = ("focus_record", "boundary")
CHECKPOINTS = ("embedding", "pre_last_block", "post_last_block_pre_norm", "post_final_norm")
DIM = 2560
WIDTH = 224
BATCH = 8
SUPPORT_K = 256
STEMS = ("brux", "cavn", "drel", "fesk", "guth", "havn", "jex", "kurn", "losp", "mert", "nuv", "prax")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def topk(values: np.ndarray, k: int = SUPPORT_K) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def inventories() -> dict[str, list[tuple[str, str, str, str, str]]]:
    attribute, agent = [], []
    for index, stem in enumerate(STEMS):
        rotations = [STEMS[(index + offset) % len(STEMS)] for offset in (0, 2, 5, 7, 9)]
        attribute.append((f"sixa{rotations[0]}meter", f"hexa{rotations[1]}hue", f"nexa{rotations[2]}node", f"orba{rotations[3]}seal", f"tova{rotations[4]}gate"))
        agent.append((f"Fi{rotations[0]}or", f"Gu{rotations[1]}ra", f"Ha{rotations[2]}is", f"Jo{rotations[3]}la", f"re{rotations[4]}scribed"))
    return {"attribute_binding": attribute, "agent_patient": agent}


def build_material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for family in FAMILIES:
        for unit_index, values in enumerate(inventories()[family]):
            unit = {"unit_id": f"c125-{family}-{unit_index:02d}", "family": family, "partition": PARTITION, "world": "controlled_synthetic_sixth_lexicon", "values": list(values)}
            units.append(unit)
            for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
                prompt, focus, anchor = breadth_base.breadth_prompt(family, values, truth, surface, distractor, code)
                output_yes = (truth == 1) == (code == 1)
                cases.append({
                    **unit,
                    "case_id": f"c125-{len(cases):04d}",
                    "truth_factor": truth,
                    "surface_factor": surface,
                    "distractor_factor": distractor,
                    "code": code,
                    "truth": truth == 1,
                    "output_yes": output_yes,
                    "gold_position": 0 if output_yes else 1,
                    "focus": focus,
                    "anchor": anchor,
                    "prompt": prompt,
                })
    return units, cases


def historical_values() -> set[str]:
    values = set()
    for path in RESULT.glob("phase*/material/units.jsonl"):
        for row in core.rows(path):
            values.update(str(value).casefold() for value in row.get("values", []))
    return values


def local_capture_semantics() -> dict:
    import transformers.models.qwen3.modeling_qwen3 as qwen_source
    import transformers.utils.output_capturing as capture_source

    qwen_path = Path(inspect.getsourcefile(qwen_source.Qwen3Model))
    capture_path = Path(inspect.getsourcefile(capture_source.capture_outputs))
    qwen_text = qwen_path.read_text(encoding="utf-8")
    capture_text = capture_path.read_text(encoding="utf-8")
    return {
        "qwen_source_path": str(qwen_path),
        "qwen_source_sha256": core.sha(qwen_path),
        "capture_source_path": str(capture_path),
        "capture_source_sha256": core.sha(capture_path),
        "final_norm_after_layers": "hidden_states = self.norm(hidden_states)" in qwen_text,
        "last_hidden_tied": "collected_outputs[key].append(outputs.last_hidden_state)" in capture_text,
        "interpretation": "output.hidden_states[-1] is post-final-norm; the pre-norm final decoder-layer output is replaced and must be recaptured explicitly",
    }


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C125 already exists: {OUT}")
    c124_audit = core.load(C124 / "audit/independent_closure_audit.json")
    c124_closure = core.load(C124 / "analysis/closure.json")
    semantics = local_capture_semantics()
    units, cases = build_material()
    tokenizer = graph_base.tokenizer()
    compiled = breadth_base.compile_breadth(tokenizer, cases)
    old = historical_values()
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    cells = Counter((row["family"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases)
    zero = breadth_base.zero_models(cases, True)
    checks = {
        "authorization": c124_audit["all_checks_passed"] and c124_closure["next_authorization"].startswith("C125 fresh semantic-program family"),
        "capture_semantics": semantics["final_norm_after_layers"] and semantics["last_hidden_tied"],
        "units": len(units) == 24,
        "cases": len(cases) == 384,
        "factorial": cells == {(family, *cell): 12 for family in FAMILIES for cell in itertools.product((1, -1), repeat=4)},
        "zero_models": all(abs(value - 0.5) < 1e-12 for key, value in zero.items() if key != "truth_x_code_oracle") and zero["truth_x_code_oracle"] == 1.0,
        "freshness": not (set(fresh) & old) and len(fresh) == len(set(fresh)),
        "unique_prompts": len({row["prompt"] for row in cases}) == 384,
        "compiled": len(compiled) == 384 and all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(ROLES).issubset(row["role_positions"]) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "machine_naturalness": all(row["prompt"].endswith("Reply exactly yes or no.") for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": 1659,
        "campaign": "C125",
        "created_at_utc": now(),
        "status": "fresh_final_transition_decomposition_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "fresh sixth-lexicon decomposition of the frozen C123 final recorded transition",
        "families": list(FAMILIES),
        "partition": PARTITION,
        "units": 24,
        "cases": 384,
        "roles": list(ROLES),
        "checkpoints": list(CHECKPOINTS),
        "activation_coordinates": DIM,
        "capture_semantics": semantics,
        "frozen_predictions": {
            "attribute_binding": {"role": "boundary", "combined_cosine_min": 0.90, "top256_overlap_min": 0.50},
            "agent_patient": {"role": "focus_record", "combined_cosine_min": 0.90, "top256_overlap_min": 0.50},
            "common_agent_boundary": {"role": "boundary", "combined_cosine_min": 0.90, "top256_overlap_min": 0.50},
        },
        "decomposition": {
            "last_block": "R(post_last_block_pre_norm)-R(pre_last_block)",
            "final_norm": "R(post_final_norm)-R(post_last_block_pre_norm)",
            "combined": "R(post_final_norm)-R(pre_last_block)",
            "identity": "combined = last_block + final_norm",
            "dominance": "descriptive only; compare L2 norms and cosine with combined without assigning semantics",
        },
        "observation_policy": "capture Embedding and HiddenState checkpoints only; do not inspect attention, MLP, or weight coordinates",
        "typed_missingness": {
            "human_naturalness": "machine-only controlled English",
            "cross_model": "Qwen3 only",
            "complete_token_graph": "two registered roles only",
            "operator_identity": "no Jacobian or independent local excitation",
        },
        "claim_boundary": "fresh Qwen3 sixth-lexicon activation-coordinate instrument decomposition; not weights, semantic neurons, attention/MLP, a language operator, complete-token causal flow, manifold, topology, or new mathematics",
        "source_paths": {
            "c123_nomination": str(C123 / "protocol/frozen_discovery_nomination.json"),
            "c123_increments": str(C123 / "analysis/discovery_selected_role_increments.float32.npy"),
            "c124_closure": str(C124 / "analysis/closure.json"),
        },
        "source_hashes": {
            "c123_nomination": core.sha(C123 / "protocol/frozen_discovery_nomination.json"),
            "c123_increments": core.sha(C123 / "analysis/discovery_selected_role_increments.float32.npy"),
            "c124_closure": core.sha(C124 / "analysis/closure.json"),
        },
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_c125_cuda_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1659, "campaign": "C125", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "max_width": max(len(row["prompt_ids"]) for row in compiled), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "execute_c125_cuda_capture" or not core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C125 capture authorization missing")
    for name, path in protocol["source_paths"].items():
        if core.sha(Path(path)) != protocol["source_hashes"][name]:
            raise RuntimeError(f"source drift: {name}")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    states_path = OUT / "raw/qwen3_role_checkpoint_states.float32.npy"
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    states_path.parent.mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(states_path, mode="w+", dtype=np.float32, shape=(len(rows), len(ROLES), len(CHECKPOINTS), DIM))
    candidate_logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(rows), 2))
    behavior = []
    model = None
    repeat_hidden = repeat_logits = 0.0
    first_rows = None

    def run_batch(batch: list[dict], model, pad: int, device: str):
        captured_pre_norm = []
        handle = model.model.norm.register_forward_pre_hook(lambda _module, args: captured_pre_norm.append(args[0].detach()))
        try:
            output, logits, ids, mask, positions, lengths = capture_base.forward(model, batch, pad, device, WIDTH)
        finally:
            handle.remove()
        if len(captured_pre_norm) != 1:
            raise RuntimeError(("pre_norm_capture_count", len(captured_pre_norm)))
        return output, logits, captured_pre_norm[0]

    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            output, logits, pre_norm = run_batch(batch, model, pad, device)
            if len(output.hidden_states) != 37 or output.hidden_states[-1].shape[-1] != DIM:
                raise RuntimeError((len(output.hidden_states), output.hidden_states[-1].shape))
            checkpoint_tensors = (output.hidden_states[0], output.hidden_states[-2], pre_norm, output.hidden_states[-1])
            for local, row in enumerate(batch):
                row_index = start + local
                for role_index, role in enumerate(ROLES):
                    positions = row["role_positions"][role]
                    for checkpoint_index, tensor in enumerate(checkpoint_tensors):
                        states[row_index, role_index, checkpoint_index] = tensor[local, positions].float().mean(dim=0).cpu().numpy()
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                candidate_logits[row_index] = scores
                prediction = int(scores[1] > scores[0])
                behavior.append({
                    "row_index": row_index,
                    "case_id": row["case_id"],
                    "unit_id": row["unit_id"],
                    "family": row["family"],
                    "truth_factor": row["truth_factor"],
                    "surface_factor": row["surface_factor"],
                    "distractor_factor": row["distractor_factor"],
                    "code": row["code"],
                    "gold_position": row["gold_position"],
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                    "yes_minus_no": scores[0] - scores[1],
                })
            if first_rows is None:
                first_rows = batch
            if (start // BATCH + 1) % 12 == 0:
                states.flush(); candidate_logits.flush()
                print(f"[C125] captured {start + len(batch)}/{len(rows)}", flush=True)
            del output, logits, pre_norm, checkpoint_tensors
        states.flush(); candidate_logits.flush()
        output, logits, pre_norm = run_batch(first_rows, model, pad, device)
        checkpoint_tensors = (output.hidden_states[0], output.hidden_states[-2], pre_norm, output.hidden_states[-1])
        for local, row in enumerate(first_rows):
            for role_index, role in enumerate(ROLES):
                positions = row["role_positions"][role]
                for checkpoint_index, tensor in enumerate(checkpoint_tensors):
                    value = tensor[local, positions].float().mean(dim=0).cpu().numpy()
                    repeat_hidden = max(repeat_hidden, float(np.max(np.abs(value - states[local, role_index, checkpoint_index]))))
            scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
            repeat_logits = max(repeat_logits, float(np.max(np.abs(scores - candidate_logits[local]))))
    finally:
        states.flush(); candidate_logits.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    core.write_rows(OUT / "raw/qwen3_behavior_index.jsonl", behavior)
    accuracy = lambda selected: float(np.mean([row["correct"] for row in selected]))
    behavior_summary = {
        "global_accuracy": accuracy(behavior),
        "by_family": {family: accuracy([row for row in behavior if row["family"] == family]) for family in FAMILIES},
        "by_code": {str(code): accuracy([row for row in behavior if row["code"] == code]) for code in (1, -1)},
    }
    checks = {
        "shape": list(states.shape) == [384, 2, 4, 2560],
        "finite": bool(np.isfinite(states).all() and np.isfinite(candidate_logits).all()),
        "repeat": repeat_hidden == 0.0 and repeat_logits == 0.0,
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "behavior_rows": len(behavior) == 384,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1659,
        "campaign": "C125",
        "created_at_utc": now(),
        "status": "fresh_final_transition_checkpoint_capture_complete",
        "checks": checks,
        "shape": list(states.shape),
        "states_sha256": core.sha(states_path),
        "logits_sha256": core.sha(logits_path),
        "behavior_sha256": core.sha(OUT / "raw/qwen3_behavior_index.jsonl"),
        "behavior": behavior_summary,
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logits_max_abs": repeat_logits},
        "runtime": {"placement": placement, "quantization": quant},
        "authorization": "adjudicate_c125_final_transition",
    }
    core.save(OUT / "analysis/capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def adjudicate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture_report = core.load(OUT / "analysis/capture_summary.json")
    if capture_report["authorization"] != "adjudicate_c125_final_transition":
        raise RuntimeError("C125 adjudication authorization missing")
    core.save(OUT / "protocol/post_unblinding_execution_amendment.json", {
        "created_at_utc": now(),
        "reason": "The frozen scientific gate failed. The original executable raised before serializing the terminal adjudication, so a deterministic failure-report branch was added.",
        "unchanged": ["research object", "material", "partition", "model", "zero models", "frozen predictions", "thresholds", "captured arrays", "computed metrics"],
        "allowed_change": "serialize, visualize, and independently audit the failed frozen gate without another model run",
        "deterministic_index_correction": "The common agent-patient boundary comparator now uses c123.ROLES.index('boundary')=6 instead of the two-role C125 local index 1. C126's complete factor decomposition exposed this mismatch. The frozen role, vectors, metrics, and thresholds are unchanged.",
        "frozen_producer_sha256": protocol["producer_sha256"],
        "terminal_producer_sha256": core.sha(Path(__file__)),
    })
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = core.rows(OUT / "material/units.jsonl")
    raw = np.load(OUT / "raw/qwen3_role_checkpoint_states.float32.npy", mmap_mode="r")
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}
    unit_fields = np.zeros((len(units), len(ROLES), len(CHECKPOINTS), DIM), dtype=np.float32)
    for row_index, row in enumerate(rows):
        unit_fields[unit_index[row["unit_id"]]] += float(row["truth_factor"]) / 16.0 * np.asarray(raw[row_index], dtype=np.float32)
    np.save(OUT / "analysis/unit_truth_role_checkpoint.float32.npy", unit_fields)
    nominations = core.load(C123 / "protocol/frozen_discovery_nomination.json")
    discovery_increments = np.load(C123 / "analysis/discovery_selected_role_increments.float32.npy", mmap_mode="r")
    family_nominations = {row["family"]: row for row in nominations["family_nominations"]}
    results = []
    effect_rows = []
    for family_index, family in enumerate(FAMILIES):
        indices = [index for index, row in enumerate(units) if row["family"] == family]
        family_fields = unit_fields[indices]
        role_names = [family_nominations[family]["role"]]
        if family == "agent_patient":
            role_names.append("boundary")
        for role in role_names:
            role_index = ROLES.index(role)
            mean = np.mean(family_fields[:, role_index], axis=0, dtype=np.float32)
            left = np.mean(family_fields[:6, role_index], axis=0, dtype=np.float32)
            right = np.mean(family_fields[6:, role_index], axis=0, dtype=np.float32)
            final_block = mean[2] - mean[1]
            final_norm = mean[3] - mean[2]
            combined = mean[3] - mean[1]
            split_components = {
                "final_block": (left[2] - left[1], right[2] - right[1]),
                "final_norm": (left[3] - left[2], right[3] - right[2]),
                "combined": (left[3] - left[1], right[3] - right[1]),
            }
            if role == family_nominations[family]["role"]:
                frozen = np.asarray(discovery_increments[family_index, 35], dtype=np.float32)
            else:
                discovery_fields = c123.discovery_fields()[family]
                frozen_role_index = c123.ROLES.index(role)
                frozen = np.mean(discovery_fields[:, frozen_role_index, 36] - discovery_fields[:, frozen_role_index, 35], axis=0, dtype=np.float32)
            prediction = protocol["frozen_predictions"][family if role == family_nominations[family]["role"] else "common_agent_boundary"]
            combined_cosine = cosine(frozen, combined)
            support_overlap = len(topk(frozen) & topk(combined)) / SUPPORT_K
            row = {
                "family": family,
                "role": role,
                "independent_units": 12,
                "frozen_combined_cosine": combined_cosine,
                "frozen_combined_top256_overlap": support_overlap,
                "frozen_prediction_passed": combined_cosine >= prediction["combined_cosine_min"] and support_overlap >= prediction["top256_overlap_min"],
                "component_norms": {"final_block": float(np.linalg.norm(final_block)), "final_norm": float(np.linalg.norm(final_norm)), "combined": float(np.linalg.norm(combined))},
                "component_to_combined_cosines": {"final_block": cosine(final_block, combined), "final_norm": cosine(final_norm, combined)},
                "split_half_cosines": {name: cosine(values[0], values[1]) for name, values in split_components.items()},
                "reconstruction_max_abs": float(np.max(np.abs(combined - final_block - final_norm))),
                "dominant_component_by_l2": "final_block" if np.linalg.norm(final_block) >= np.linalg.norm(final_norm) else "final_norm",
            }
            results.append(row)
            for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
                effect_rows.append({"family": family, "role": role, "kind": "balanced_truth_response", "checkpoint": checkpoint, "values": mean[checkpoint_index].tolist()})
            for name, values in (("final_block_increment", final_block), ("final_norm_increment", final_norm), ("combined_increment", combined)):
                effect_rows.append({"family": family, "role": role, "kind": name, "checkpoint": name, "values": values.tolist()})
    core.write_rows(OUT / "analysis/decomposition_results.jsonl", results)
    integrity_checks = {
        "unit_shape": list(unit_fields.shape) == [24, 2, 4, 2560],
        "results": len(results) == 3,
        "reconstruction": all(row["reconstruction_max_abs"] <= 1e-5 for row in results),
        "finite": bool(np.isfinite(unit_fields).all()),
    }
    if not all(integrity_checks.values()):
        raise RuntimeError(integrity_checks)
    scientific_gate = all(row["frozen_prediction_passed"] for row in results)
    checks = {**integrity_checks, "frozen_predictions": scientific_gate}
    report = {
        "phase": 1659,
        "campaign": "C125",
        "created_at_utc": now(),
        "status": "fresh_final_transition_decomposition_confirmed" if scientific_gate else "fresh_final_transition_decomposition_prediction_failed",
        "checks": checks,
        "scientific_gate_passed": scientific_gate,
        "results": results,
        "behavior": capture_report["behavior"],
        "claim_boundary": protocol["claim_boundary"],
        "authorization": "synthesize_c125_heatmap_and_close" if scientific_gate else "synthesize_c125_failed_gate_and_close",
    }
    core.save(OUT / "analysis/adjudication.json", report)
    core.save(OUT / "analysis/visualization_effect_rows.json", effect_rows)
    core.save(OUT / "audit/internal_adjudication_audit.json", {"phase": 1659, "campaign": "C125", "integrity_checks": integrity_checks, "all_integrity_checks_passed": all(integrity_checks.values()), "scientific_gate_passed": scientific_gate, "authorization": report["authorization"]})
    print(json.dumps(report, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/adjudication.json")
    if report["authorization"] not in {"synthesize_c125_heatmap_and_close", "synthesize_c125_failed_gate_and_close"} or not core.load(OUT / "audit/internal_adjudication_audit.json")["all_integrity_checks_passed"]:
        raise RuntimeError("C125 synthesis authorization missing")
    payload = core.load(PUBLIC)
    effect_rows = core.load(OUT / "analysis/visualization_effect_rows.json")
    payload["c125_final_transition_batch"] = {
        "capture_semantics": protocol["capture_semantics"],
        "adjudication": report,
        "effect_rows": effect_rows,
    }
    payload.update({
        "phase": 1659,
        "campaign": "C109-C117 + C123-C125",
        "title": "Role-State Atlas + Layer Transition and Final-Norm Decomposition",
        "claim_boundary": "C125 prospectively separates the last decoder-block HiddenState increment from the final-normalization increment on fresh Qwen3 synthetic lexicons. Full 2560 activation coordinates are shown; they are not weights, independent semantic neurons, attention/MLP mechanisms, a complete-token causal graph, or a language operator.",
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c125_final_transition_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    dominant = {f"{row['family']}::{row['role']}": row["dominant_component_by_l2"] for row in report["results"]}
    closure = {
        "phase": 1659,
        "campaign": "C125",
        "created_at_utc": now(),
        "status": "fresh_final_transition_instrument_decomposition_closed" if report["scientific_gate_passed"] else "fresh_final_transition_instrument_decomposition_closed_failed_prediction",
        "headline": {"frozen_predictions_passed": sum(row["frozen_prediction_passed"] for row in report["results"]), "cells": len(report["results"]), "dominant_component_by_l2": dominant, "results": report["results"]},
        "new_puzzles": {"K317": "The C123 final recorded transition is instrument-heterogeneous: it combines the final decoder-block response with the final-normalization response. On a sixth lexicon, attribute binding replicated, while the registered agent-patient role and common-boundary predictions failed."},
        "theory_update": "The registered transition atlas must type checkpoint semantics. A stable S35-to-S36 response is not a homogeneous decoder-layer step because the standard output tuple replaces the last block output with the post-final-norm state. The failed agent-patient predictions prohibit a universal final-transition claim.",
        "unified_formula": "DeltaR_recorded(35->36)=DeltaR_last_block+DeltaR_final_norm; neither term is a semantic operator without fresh composition prediction and causal identification.",
        "problems": ["two controlled relation families and one Qwen3 model", "fresh lexical replication reuses the same prompt grammar", "only boundary/focus_record registered roles are captured", "component dominance is based on response L2 and does not assign semantics", "no independent human naturalness review", "no complete-token or cross-model graph"],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "activation_coordinates": 2560, "includes_embedding": True, "includes_hidden_state_checkpoints": True},
        "claim_boundary": payload["claim_boundary"],
        "next_authorization": "C126 may use only the already captured C125 data to separate truth, code, output, surface, and distractor response components and diagnose the failed frozen predictions. C125 does not authorize a new model confirmation run.",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "adjudication_integrity": core.load(OUT / "audit/internal_adjudication_audit.json")["all_integrity_checks_passed"],
        "effect_rows": len(effect_rows) == 21 and all(len(row["values"]) == DIM for row in effect_rows),
        "asset": core.sha(canonical) == core.sha(PUBLIC),
        "semantic_boundary": "not weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"],
        "failed_gate_preserved": closure["headline"]["frozen_predictions_passed"] == 2 and report["scientific_gate_passed"] is False,
        "next": closure["next_authorization"].startswith("C126 may use only"),
    }
    final = {"phase": 1659, "campaign": "C125", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "authorization": "independent_audit_then_append_memo"}
    core.save(OUT / "audit/internal_closure_audit.json", final)
    print(json.dumps({"audit": final, "headline": closure["headline"], "next_authorization": closure["next_authorization"]}, indent=2))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"contract", "capture", "adjudicate", "synthesize"}:
        raise SystemExit("usage: phase1659_c125_final_transition_decomposition.py {contract|capture|adjudicate|synthesize}")
    {"contract": contract, "capture": capture, "adjudicate": adjudicate, "synthesize": synthesize}[sys.argv[1]]()


if __name__ == "__main__":
    main()
