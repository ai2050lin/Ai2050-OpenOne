#!/usr/bin/env python3
"""Phase1257: factorial-response geometry and Qwen-shaped head instrument calibration.

This phase does not run a language model. It performs two prerequisite audits:

1. It determines which Phase1256 claims are identifiable from the frozen
   aggregate artifacts, including the categorical wrong-donor cosine baseline
   and the non-identifiability of the matched-null direction.
2. It calibrates the exact tensor slicing used by the next Qwen3 head-level
   intervention contract against independently constructed references with the
   real Qwen3-4B Q/KV head geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PHASE = 1257
CONTRACT_ID = "EXP-C010-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1257_c010_factorial_head_instrument_calibration_audit.py"
OUT = ROOT / "tests/glm5/result/phase1257_c010_factorial_head_instrument_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/calibration_result.json"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/instrument_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

PHASE1256 = ROOT / "tests/glm5/result/phase1256_c009_qwen3_typed_edge_coalition"
PHASE1256_DETAILS = PHASE1256 / "raw/coalition_result.json"
PHASE1256_FINAL = PHASE1256 / "analysis/final.json"

QWEN_GEOMETRY = {
    "layers": 36,
    "hidden_size": 2560,
    "query_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "q_projection_size": 4096,
    "kv_projection_size": 1024,
    "gqa_group_size": 4,
}

THRESHOLDS = {
    "categorical_wrong_cosine_abs_error_max": 1.0e-12,
    "head_patch_reference_max_error": 0.0,
    "untouched_slice_max_error": 0.0,
    "no_op_max_error": 0.0,
    "disjoint_commutation_max_error": 0.0,
    "full_union_max_error": 0.0,
    "rope_norm_relative_error_max": 1.0e-5,
    "rms_positive_scale_error_max": 2.0e-3,
    "tensor_trials_min": 384,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_payload() -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "schema_version": "phase1257.c010.factorial_head_instrument.protocol.v1",
        "contract_id": CONTRACT_ID,
        "claim_type": "factorial_response_geometry_and_head_slice_instrument_calibration",
        "question": "Can the revised identity/null metrics and Qwen3 head-slice intervention operator be calibrated before any new pretrained-model mechanism test?",
        "qwen_geometry": QWEN_GEOMETRY,
        "tensor_trials": 384,
        "thresholds": THRESHOLDS,
        "dependencies": {
            "phase1256_details": file_sha256(PHASE1256_DETAILS),
            "phase1256_final": file_sha256(PHASE1256_FINAL),
        },
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "auditor": file_sha256(AUDITOR),
        },
        "hard_stops": [
            "No Qwen3, GLM4 or DS7B forward pass is executed in this phase.",
            "Phase1256 matched-null direction is not reconstructed unless per-world response vectors exist in frozen artifacts.",
            "A wrong-donor cosine near 0.5 is treated as the ideal shared-base categorical baseline, not independent identity evidence.",
            "Head-slice calibration establishes tensor intervention integrity only, not a semantic circuit.",
            "Failure of any exact slice/reference gate blocks the new pretrained-model contract.",
            "No Phase1256 component, layer or prefix is rescanned.",
        ],
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    if not PHASE1256_DETAILS.exists() or not PHASE1256_FINAL.exists():
        raise RuntimeError("Phase1256 dependencies missing")
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    atomic_json(PROTOCOL, protocol_payload())
    print(canonical_json({"status": "preregistered", "contract_id": CONTRACT_ID}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    expected = protocol_payload()
    if protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("source hash drift")
    if protocol["dependencies"] != expected["dependencies"]:
        raise RuntimeError("dependency hash drift")
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol digest drift")
    return protocol


def ideal_categorical_geometry(classes: int = 8) -> dict[str, Any]:
    cosines: list[float] = []
    direct_separations: list[float] = []
    for base in range(classes):
        for target in range(classes):
            if target == base:
                continue
            for wrong in range(classes):
                if wrong in {base, target}:
                    continue
                target_response = F.one_hot(torch.tensor(target), classes).double() - F.one_hot(torch.tensor(base), classes).double()
                wrong_response = F.one_hot(torch.tensor(wrong), classes).double() - F.one_hot(torch.tensor(base), classes).double()
                cosine = torch.dot(target_response, wrong_response) / (
                    torch.linalg.vector_norm(target_response) * torch.linalg.vector_norm(wrong_response)
                )
                contrast = F.one_hot(torch.tensor(target), classes).double() - F.one_hot(torch.tensor(wrong), classes).double()
                direct_separations.append(float(torch.dot(target_response - wrong_response, contrast).item()))
                cosines.append(float(cosine.item()))
    return {
        "case_count": len(cosines),
        "wrong_cosine_min": min(cosines),
        "wrong_cosine_max": max(cosines),
        "wrong_cosine_mean": sum(cosines) / len(cosines),
        "wrong_cosine_theory": 0.5,
        "wrong_cosine_abs_error": max(abs(value - 0.5) for value in cosines),
        "direct_target_wrong_separation_min": min(direct_separations),
    }


def null_identifiability(details: dict[str, Any]) -> dict[str, Any]:
    ratio = float(details["confirmation"]["null_effect_fraction"])
    target_norm = float(details["target_effect_norm"])
    examples = []
    for fraction in (-1.0, -0.5, 0.0, 0.5, 1.0):
        alpha = ratio * fraction
        orthogonal = math.sqrt(max(0.0, ratio * ratio - alpha * alpha))
        examples.append({
            "alpha_parallel": alpha,
            "epsilon_orthogonal": orthogonal,
            "reconstructed_total": math.sqrt(alpha * alpha + orthogonal * orthogonal),
        })
    required_fields = {
        "confirmation_world_responses",
        "target_response_vectors",
        "null_response_vectors",
        "patched_score_vectors",
    }
    present = sorted(required_fields.intersection(details))
    return {
        "aggregate_null_fraction": ratio,
        "aggregate_target_norm": target_norm,
        "aggregate_null_norm": ratio * target_norm,
        "required_world_vector_fields": sorted(required_fields),
        "present_world_vector_fields": present,
        "world_direction_identifiable": bool(present),
        "same_total_distinct_decompositions": examples,
        "adjudication": "not_identifiable_from_frozen_aggregates" if not present else "identifiable",
    }


def patch_heads(value: torch.Tensor, donor: torch.Tensor, heads: list[int], head_dim: int) -> torch.Tensor:
    result = value.clone()
    shaped = result.view(*result.shape[:-1], -1, head_dim)
    donor_shaped = donor.view(*donor.shape[:-1], -1, head_dim)
    shaped[..., heads, :] = donor_shaped[..., heads, :]
    return result


def reference_patch(value: torch.Tensor, donor: torch.Tensor, heads: list[int], head_dim: int) -> torch.Tensor:
    head_count = value.shape[-1] // head_dim
    mask = torch.zeros(head_count, device=value.device, dtype=torch.bool)
    mask[heads] = True
    mask = mask.repeat_interleave(head_dim).view(*([1] * (value.ndim - 1)), -1)
    return torch.where(mask, donor, value)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def tensor_instrument_calibration(device: torch.device, trials: int) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(1_257_010_001)
    max_reference = 0.0
    max_untouched = 0.0
    max_noop = 0.0
    max_commutation = 0.0
    max_union = 0.0
    shapes = (
        ("q", QWEN_GEOMETRY["query_heads"], QWEN_GEOMETRY["head_dim"]),
        ("ov", QWEN_GEOMETRY["query_heads"], QWEN_GEOMETRY["head_dim"]),
        ("v", QWEN_GEOMETRY["kv_heads"], QWEN_GEOMETRY["head_dim"]),
    )
    for trial in range(trials):
        _role, heads, head_dim = shapes[trial % len(shapes)]
        batch = 1 + trial % 3
        sequence = 3 + trial % 7
        width = heads * head_dim
        base = torch.randn(batch, sequence, width, generator=generator, device=device, dtype=torch.float16)
        donor = torch.randn(batch, sequence, width, generator=generator, device=device, dtype=torch.float16)
        first = trial % heads
        second = (first + 1 + (trial // heads) % max(1, heads - 1)) % heads
        chosen = sorted({first, second})
        patched = patch_heads(base, donor, chosen, head_dim)
        reference = reference_patch(base, donor, chosen, head_dim)
        max_reference = max(max_reference, float((patched - reference).abs().max().item()))
        shaped_patch = patched.view(batch, sequence, heads, head_dim)
        shaped_base = base.view(batch, sequence, heads, head_dim)
        untouched = [index for index in range(heads) if index not in chosen]
        if untouched:
            max_untouched = max(max_untouched, float((shaped_patch[..., untouched, :] - shaped_base[..., untouched, :]).abs().max().item()))
        no_op = patch_heads(base, base, chosen, head_dim)
        max_noop = max(max_noop, float((no_op - base).abs().max().item()))
        left = patch_heads(patch_heads(base, donor, [first], head_dim), donor, [second], head_dim)
        right = patch_heads(patch_heads(base, donor, [second], head_dim), donor, [first], head_dim)
        max_commutation = max(max_commutation, float((left - right).abs().max().item()))
        union = patch_heads(base, donor, list(range(heads)), head_dim)
        max_union = max(max_union, float((union - donor).abs().max().item()))

    angle = torch.linspace(0.01, 1.2, QWEN_GEOMETRY["head_dim"] // 2, device=device, dtype=torch.float32)
    cos = torch.cat((torch.cos(angle), torch.cos(angle)))
    sin = torch.cat((torch.sin(angle), torch.sin(angle)))
    rope_input = torch.randn(256, QWEN_GEOMETRY["head_dim"], generator=generator, device=device, dtype=torch.float32)
    rope_output = rope_input * cos + rotate_half(rope_input) * sin
    rope_error = float(((torch.linalg.vector_norm(rope_output, dim=-1) - torch.linalg.vector_norm(rope_input, dim=-1)).abs() /
                        torch.linalg.vector_norm(rope_input, dim=-1).clamp_min(1.0e-12)).max().item())

    rms_input = torch.randn(256, QWEN_GEOMETRY["hidden_size"], generator=generator, device=device, dtype=torch.float16)
    scales = torch.linspace(0.25, 4.0, 256, device=device, dtype=torch.float32).unsqueeze(-1)

    def rms(value: torch.Tensor) -> torch.Tensor:
        work = value.float()
        return (work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + 1.0e-6)).half()

    rms_error = float((rms(rms_input) - rms(rms_input.float() * scales)).abs().max().item())
    gqa_map = [index // QWEN_GEOMETRY["gqa_group_size"] for index in range(QWEN_GEOMETRY["query_heads"])]
    gqa_counts = [gqa_map.count(index) for index in range(QWEN_GEOMETRY["kv_heads"])]
    return {
        "device": str(device),
        "dtype": "float16",
        "tensor_trials": trials,
        "head_patch_reference_max_error": max_reference,
        "untouched_slice_max_error": max_untouched,
        "no_op_max_error": max_noop,
        "disjoint_commutation_max_error": max_commutation,
        "full_union_max_error": max_union,
        "rope_norm_relative_error": rope_error,
        "rms_positive_scale_error": rms_error,
        "gqa_query_to_kv_map": gqa_map,
        "gqa_queries_per_kv_head": gqa_counts,
        "gqa_mapping_exact": all(value == QWEN_GEOMETRY["gqa_group_size"] for value in gqa_counts),
    }


def calibration_passes(result: dict[str, Any]) -> bool:
    geometry = result["categorical_geometry"]
    tensor = result["tensor_instrument"]
    return (
        geometry["wrong_cosine_abs_error"] <= THRESHOLDS["categorical_wrong_cosine_abs_error_max"]
        and tensor["tensor_trials"] >= THRESHOLDS["tensor_trials_min"]
        and tensor["head_patch_reference_max_error"] <= THRESHOLDS["head_patch_reference_max_error"]
        and tensor["untouched_slice_max_error"] <= THRESHOLDS["untouched_slice_max_error"]
        and tensor["no_op_max_error"] <= THRESHOLDS["no_op_max_error"]
        and tensor["disjoint_commutation_max_error"] <= THRESHOLDS["disjoint_commutation_max_error"]
        and tensor["full_union_max_error"] <= THRESHOLDS["full_union_max_error"]
        and tensor["rope_norm_relative_error"] <= THRESHOLDS["rope_norm_relative_error_max"]
        and tensor["rms_positive_scale_error"] <= THRESHOLDS["rms_positive_scale_error_max"]
        and tensor["gqa_mapping_exact"]
        and not result["phase1256_null_identifiability"]["world_direction_identifiable"]
    )


def run() -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal completion marker exists")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("preaudit not passed")
    protocol = verify_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal FP16 tensor calibration")
    started = time.perf_counter()
    details = read_json(PHASE1256_DETAILS)
    result = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "categorical_geometry": ideal_categorical_geometry(),
        "phase1256_null_identifiability": null_identifiability(details),
        "phase1256_wrong_cosine_observed": details["confirmation"]["wrong"]["cosine"],
        "phase1256_k218_scope_correction": {
            "old_grade": "E3-KT",
            "recommended_grade": "E3-CONTROLLED-FREE",
            "reason": "Free-network task truth and endpoint truth do not expose internal mechanism truth.",
        },
        "tensor_instrument": tensor_instrument_calibration(torch.device("cuda"), int(protocol["tensor_trials"])),
    }
    result["passed"] = calibration_passes(result)
    atomic_json(RAW, result)
    elapsed = time.perf_counter() - started
    summary = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "raw_sha256": file_sha256(RAW),
        "run_digest": digest(result),
    }
    atomic_json(SUMMARY, summary)
    marker = {
        "phase": PHASE,
        "status": "formal_run_complete",
        "raw_sha256": file_sha256(RAW),
        "summary_sha256": file_sha256(SUMMARY),
        "run_digest": summary["run_digest"],
    }
    marker["marker_digest"] = digest(marker)
    atomic_json(COMPLETE, marker)
    print(canonical_json({"status": "formal_run_complete", "passed": result["passed"]}))


def analyze() -> None:
    if not COMPLETE.exists():
        raise RuntimeError("formal run incomplete")
    protocol = verify_protocol()
    result = read_json(RAW)
    summary = read_json(SUMMARY)
    marker = read_json(COMPLETE)
    if marker["raw_sha256"] != file_sha256(RAW) or marker["summary_sha256"] != file_sha256(SUMMARY):
        raise RuntimeError("artifact hash mismatch")
    verdict = "factorial_head_instrument_calibrated" if result["passed"] else "factorial_head_instrument_not_calibrated"
    authorization = {
        "new_natural_qwen_contract": bool(result["passed"]),
        "phase1256_null_reinterpretation": False,
        "phase1256_head_rescan": False,
        "semantic_mechanism_claim": False,
        "new_mathematics": False,
    }
    analysis = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "verdict": verdict,
        "categorical_geometry": result["categorical_geometry"],
        "phase1256_null_identifiability": result["phase1256_null_identifiability"],
        "tensor_instrument": result["tensor_instrument"],
        "authorization": authorization,
        "scope": "No-model Phase1256 artifact audit plus Qwen3-shaped CUDA FP16 tensor operator calibration.",
    }
    analysis["analysis_digest"] = digest(analysis)
    atomic_json(ANALYSIS, analysis)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "authorization": authorization,
        "key_results": {
            "ideal_wrong_cosine": result["categorical_geometry"]["wrong_cosine_mean"],
            "phase1256_null_norm": result["phase1256_null_identifiability"]["aggregate_null_norm"],
            "phase1256_null_direction_identifiable": result["phase1256_null_identifiability"]["world_direction_identifiable"],
            "head_patch_reference_max_error": result["tensor_instrument"]["head_patch_reference_max_error"],
            "gqa_mapping_exact": result["tensor_instrument"]["gqa_mapping_exact"],
        },
        "artifact_hashes": {
            "protocol": file_sha256(PROTOCOL),
            "environment": file_sha256(ENVIRONMENT),
            "preaudit": file_sha256(PREAUDIT),
            "raw": file_sha256(RAW),
            "summary": file_sha256(SUMMARY),
            "complete": file_sha256(COMPLETE),
            "analysis": file_sha256(ANALYSIS),
        },
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"verdict": verdict, "authorization": authorization}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run", "analyze"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        run()
    else:
        analyze()


if __name__ == "__main__":
    main()
