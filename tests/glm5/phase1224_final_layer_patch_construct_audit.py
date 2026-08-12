#!/usr/bin/env python3
"""Phase 1224: final-layer patch construct and autoregressive timing audit.

This phase does not search for a mechanism.  It freezes the final decoder
residual and audits whether Phase1223's one-position patch controls the token
that its full-continuation margin actually distinguishes.  A sustained
all-scoring-position patch is included only as an instrument positive control.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1223_passed_atom_physical_trajectory as p1223
from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1224
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1224_final_layer_patch_construct_audit_audit.py"
SOURCE_ROOT = TEST_ROOT / "result/phase1223_passed_atom_physical_trajectory"
SOURCE_FINAL = SOURCE_ROOT / "analysis/final.json"
SOURCE_RESULT_AUDIT = SOURCE_ROOT / "audit/independent_result_audit.json"
SOURCE_PROTOCOL = SOURCE_ROOT / "protocol/preregistration.json"
SOURCE_PAIRS = SOURCE_ROOT / "protocol/pair_manifest.jsonl"
SOURCE_STATES = SOURCE_ROOT / "protocol/state_manifest.jsonl"
SOURCE_ARRAYS = SOURCE_ROOT / "runs/physical_trajectory_arrays.npz"
EXPECTED_SOURCE_FINAL_DIGEST = "b1973184747d83a665b6dc3fd61bff4164e21aab0e2e14cfb2f5a69a57ab9304"
EXPECTED_SOURCE_RESULT_AUDIT_DIGEST = "d9c8a50299997ecdb1955419758121223118e7e568ecbbe49113bb71a6383542"

OUT_ROOT = TEST_ROOT / "result/phase1224_final_layer_patch_construct_audit"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MANIFEST_PATH = OUT_ROOT / "protocol/construct_manifest.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RECORD_PATH = OUT_ROOT / "runs/construct_records.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "runs/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

SPLITS = ("discovery", "confirmation", "natural_use", "sealed")
HOLDOUT_SPLITS = ("confirmation", "natural_use", "sealed")
LAYER_COUNT = 36
EPSILON = 1e-8
CONDITIONS = (
    "boundary_live",
    "boundary_stored",
    "divergence_live",
    "all_scoring_live",
    "all_scoring_zero",
)
THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "hook_write_max_abs_max": 0.0,
    "boundary_live_logit_max_abs_max": 1e-4,
    "boundary_live_top1_agreement_min": 1.0,
    "all_scoring_score_max_abs_max": 1e-4,
    "all_scoring_completion_median_min": 0.999,
    "zero_score_max_abs_max": 1e-4,
    "prompt_full_logit_max_abs_max": 0.25,
    "stored_live_hidden_relative_max": 1e-3,
    "target_shift_abs_min": 1.0,
    "divergence_token_score_max_abs_max": 1e-4,
    "holdout_lcp0_boundary_completion_median_min": 0.50,
    "holdout_lcp0_positive_fraction_min": 0.75,
    "holdout_lcp_positive_abs_completion_median_max": 0.05,
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def first_divergence(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    return min(len(left), len(right))


def verify_source() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    final = read_json(SOURCE_FINAL)
    result_audit = read_json(SOURCE_RESULT_AUDIT)
    if final.get("final_digest") != EXPECTED_SOURCE_FINAL_DIGEST:
        raise RuntimeError("Phase1223 final digest drift")
    if result_audit.get("audit_digest") != EXPECTED_SOURCE_RESULT_AUDIT_DIGEST:
        raise RuntimeError("Phase1223 result audit digest drift")
    if not result_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1223 result audit is not fully passed")
    pairs = read_jsonl(SOURCE_PAIRS)
    states = read_jsonl(SOURCE_STATES)
    if len(pairs) != 160 or len(states) != 640:
        raise RuntimeError("Phase1223 source cardinality drift")
    return final, pairs, states


def build_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _final, pairs, states = verify_source()
    state_by_id = {row["state_id"]: row for row in states}
    manifest: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda row: (SPLITS.index(row["split"]), row["scope"], row["pair_id"])):
        recipient_id = pair["panel_states"]["canonical"]
        donor_id = pair["panel_states"]["binding_permutation"]
        recipient = state_by_id[recipient_id]
        donor = state_by_id[donor_id]
        candidates = list(recipient["candidates"])
        if set(candidates) != set(donor["candidates"]):
            raise RuntimeError("candidate set differs across a pair")
        recipient_ids = {
            key: [int(value) for value in recipient["candidate_token_ids"][key]]
            for key in candidates
        }
        donor_ids = {
            key: [int(value) for value in donor["candidate_token_ids"][key]]
            for key in candidates
        }
        if recipient_ids != donor_ids:
            raise RuntimeError("candidate tokenization differs across a pair")
        lengths = {len(value) for value in recipient_ids.values()}
        if len(lengths) != 1:
            raise RuntimeError("candidate length is not matched")
        recipient_gold_ids = recipient_ids[pair["recipient_gold"]]
        donor_gold_ids = recipient_ids[pair["donor_gold"]]
        lcp = first_divergence(recipient_gold_ids, donor_gold_ids)
        if lcp >= len(recipient_gold_ids):
            raise RuntimeError("recipient and donor gold candidates do not diverge")
        row: dict[str, Any] = {
            "schema_version": "phase1224.construct-manifest.v1",
            "phase": PHASE,
            "pair_id": pair["pair_id"],
            "scope": pair["scope"],
            "split": pair["split"],
            "recipient_state_id": recipient_id,
            "donor_state_id": donor_id,
            "recipient_state_index": int(recipient["state_index"]),
            "donor_state_index": int(donor["state_index"]),
            "recipient_gold": pair["recipient_gold"],
            "donor_gold": pair["donor_gold"],
            "candidates": candidates,
            "candidate_token_ids": recipient_ids,
            "continuation_length": int(next(iter(lengths))),
            "gold_first_divergence": int(lcp),
            "gold_first_token_discriminative": bool(lcp == 0),
            "recipient_prompt_length": len(recipient["input_ids"]),
            "donor_prompt_length": len(donor["input_ids"]),
            "generation_boundary": int(recipient["role_positions"]["generation_boundary"]),
        }
        if row["recipient_prompt_length"] != row["donor_prompt_length"]:
            raise RuntimeError("pair prompt lengths differ")
        if row["generation_boundary"] != row["recipient_prompt_length"] - 1:
            raise RuntimeError("generation boundary drift")
        row["row_digest"] = digest(row)
        manifest.append(row)

    distribution = Counter(
        (row["split"], row["scope"], row["gold_first_divergence"], row["continuation_length"])
        for row in manifest
    )
    static = {
        "pair_count": len(manifest),
        "holdout_pair_count": sum(row["split"] in HOLDOUT_SPLITS for row in manifest),
        "first_token_discriminative_count": sum(row["gold_first_token_discriminative"] for row in manifest),
        "shared_first_token_count": sum(not row["gold_first_token_discriminative"] for row in manifest),
        "discovery_first_token_discriminative_count": sum(
            row["split"] == "discovery" and row["gold_first_token_discriminative"] for row in manifest
        ),
        "holdout_first_token_discriminative_count": sum(
            row["split"] in HOLDOUT_SPLITS and row["gold_first_token_discriminative"] for row in manifest
        ),
        "distribution": {
            "|".join(map(str, key)): value for key, value in sorted(distribution.items())
        },
    }
    return manifest, static


def build_protocol(manifest: list[dict[str, Any]], static: dict[str, Any]) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1224.protocol.v1",
        "created_at": utc_now(),
        "purpose": "Audit final-layer patch identity and separate one-token control from full-continuation control without searching a new mechanism.",
        "source_hashes": {
            "script": file_sha256(SCRIPT),
            "audit_script": file_sha256(AUDIT_SCRIPT),
            "phase1223_final": file_sha256(SOURCE_FINAL),
            "phase1223_result_audit": file_sha256(SOURCE_RESULT_AUDIT),
            "phase1223_protocol": file_sha256(SOURCE_PROTOCOL),
            "phase1223_pairs": file_sha256(SOURCE_PAIRS),
            "phase1223_states": file_sha256(SOURCE_STATES),
            "phase1223_arrays": file_sha256(SOURCE_ARRAYS),
        },
        "upstream": {
            "phase1223_final_digest": EXPECTED_SOURCE_FINAL_DIGEST,
            "phase1223_result_audit_digest": EXPECTED_SOURCE_RESULT_AUDIT_DIGEST,
            "k200_is_not_rewritten": True,
            "audit_question": "Does K200 primarily bind a one-step patch to a multi-token score whose discriminating token occurs later?",
        },
        "material": {
            "pair_count": len(manifest),
            "splits": list(SPLITS),
            "holdout_splits_are_prospective_for_construct_patch": list(HOLDOUT_SPLITS),
            "manifest_digest": digest(manifest),
            "static_token_audit": static,
            "no_pair_selection": True,
        },
        "interventions": {
            "fixed_component": "final decoder layer whole residual output",
            "fixed_depth": LAYER_COUNT,
            "conditions": {
                "boundary_live": "replace only prompt generation-boundary state with the live donor state",
                "boundary_stored": "replay the exact float16 donor boundary state stored by Phase1223",
                "divergence_live": "replace only the state that scores the first token where donor and recipient gold candidates diverge",
                "all_scoring_live": "replace every final-layer state used to score the candidate continuation with its row-matched donor state",
                "all_scoring_zero": "replace those states with the live recipient states",
            },
            "all_scoring_live_is_instrument_positive_control_not_mechanism_claim": True,
            "no_layer_search": True,
            "no_role_search": True,
            "no_head_or_neuron_search": True,
        },
        "readouts": {
            "fixed_margin": "S(donor_gold)-S(recipient_gold) for recipient, donor, and every patched condition",
            "first_token_full_vocab_logits": True,
            "per_token_log_probability": True,
            "full_continuation_sum": True,
            "prompt_only_vs_full_sequence_causal_parity": True,
            "target_shift_denominator_audit": True,
        },
        "thresholds": THRESHOLDS,
        "construct_gate": {
            "instrument": "finite AND hook write identity AND boundary next-token identity AND sustained full-score identity AND zero identity AND scoring-path parity AND denominator qualification",
            "prospective_holdout_prediction": "LCP=0 boundary completion passes; LCP>0 remains near zero; first-divergence patch and sustained patch reproduce donor token scores",
            "all_terms_required": True,
        },
        "authorization": {
            "pass": "authorize a separately frozen known-truth distributed-process calibration only",
            "fail": "stop and repair the intervention instrument",
            "qwen_new_mechanism_scan": False,
            "cross_model": False,
        },
        "forbidden_after_freeze": [
            "change the final layer, positions, intervention conditions, thresholds, pairs, or readout",
            "reinterpret the sustained positive control as a discovered natural mechanism",
            "use discovery-only results to override the three holdout construct prediction",
            "start a Qwen multi-position mechanism scan from this phase",
        ],
        "claim_boundary": {
            "qwen3_only": True,
            "generated_material": True,
            "construct_validity_not_language_mechanism": True,
            "phase1223_discovery_is_already_seen": True,
            "three_holdout_splits_are_new_for_patch_construct": True,
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    return protocol


def materialize() -> None:
    if RECORD_PATH.exists() or FINAL_PATH.exists():
        raise RuntimeError("refusing to rewrite protocol after model output exists")
    manifest, static = build_manifest()
    protocol = build_protocol(manifest, static)
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"status": "materialized", "pairs": len(manifest), "protocol_digest": protocol["protocol_digest"]}))


def verify_formal_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    states = read_jsonl(SOURCE_STATES)
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not pass")
    if protocol["protocol_digest"] != digest({key: value for key, value in protocol.items() if key != "protocol_digest"}):
        raise RuntimeError("protocol digest drift")
    if protocol["material"]["manifest_digest"] != digest(manifest):
        raise RuntimeError("manifest digest drift")
    return protocol, manifest, {row["state_id"]: row for row in states}


class CaptureLastLayer(AbstractContextManager["CaptureLastLayer"]):
    def __init__(self, module: Any):
        self.module = module
        self.handle: Any | None = None
        self.value: torch.Tensor | None = None
        self.calls = 0

    def _hook(self, _module: Any, _args: Any, output: Any):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("last-layer output is not tensor")
        self.value = value.detach().clone()
        self.calls += 1
        return output

    def __enter__(self) -> "CaptureLastLayer":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class MultiPositionPatch(AbstractContextManager["MultiPositionPatch"]):
    def __init__(self, module: Any, positions: list[int], replacement: torch.Tensor):
        self.module = module
        self.positions = [int(value) for value in positions]
        self.replacement = replacement
        self.handle: Any | None = None
        self.calls = 0
        self.write_max_abs = float("nan")

    def _hook(self, _module: Any, _args: Any, output: Any):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("patch output is not tensor")
        replacement = self.replacement.to(value.device, dtype=value.dtype)
        if replacement.shape != value.shape:
            raise RuntimeError("replacement shape mismatch")
        patched = value.clone()
        patched[:, self.positions, :] = replacement[:, self.positions, :]
        self.write_max_abs = float(
            (patched[:, self.positions, :] - replacement[:, self.positions, :]).abs().max().item()
        )
        self.calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def __enter__(self) -> "MultiPositionPatch":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def make_batch(state: dict[str, Any], candidates: list[str], device: torch.device) -> tuple[torch.Tensor, int, int]:
    prompt = [int(value) for value in state["input_ids"]]
    suffixes = [[int(value) for value in state["candidate_token_ids"][candidate]] for candidate in candidates]
    lengths = {len(value) for value in suffixes}
    if len(lengths) != 1:
        raise RuntimeError("candidate continuation lengths differ")
    continuation_length = int(next(iter(lengths)))
    batch = torch.tensor([prompt + suffix for suffix in suffixes], dtype=torch.long, device=device)
    return batch, continuation_length, len(prompt) - 1


def forward_capture(
    model: Any,
    module: Any,
    input_ids: torch.Tensor,
    continuation_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), CaptureLastLayer(module) as capture:
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=continuation_length + 1,
            return_dict=True,
        )
    if capture.calls != 1 or capture.value is None:
        raise RuntimeError("capture call drift")
    logits = output.logits.detach()
    hidden = capture.value
    del output
    return logits, hidden


def forward_patch(
    model: Any,
    module: Any,
    input_ids: torch.Tensor,
    continuation_length: int,
    positions: list[int],
    replacement: torch.Tensor,
) -> tuple[torch.Tensor, int, float]:
    with torch.inference_mode(), MultiPositionPatch(module, positions, replacement) as patch:
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=continuation_length + 1,
            return_dict=True,
        )
    logits = output.logits.detach()
    del output
    return logits, int(patch.calls), float(patch.write_max_abs)


def prompt_logits(model: Any, input_ids: list[int], device: torch.device) -> torch.Tensor:
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            input_ids=tensor,
            attention_mask=torch.ones_like(tensor),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    logits = output.logits[0, -1].detach()
    del output, tensor
    return logits


def score_from_logits(
    logits: torch.Tensor,
    candidates: list[str],
    candidate_token_ids: dict[str, list[int]],
) -> tuple[dict[str, float], dict[str, list[float]], bool]:
    scores: dict[str, float] = {}
    token_scores: dict[str, list[float]] = {}
    finite = bool(torch.isfinite(logits).all().item())
    for row_index, candidate in enumerate(candidates):
        values: list[float] = []
        for offset, token_id in enumerate(candidate_token_ids[candidate]):
            row_logits = logits[row_index, offset].float()
            score = row_logits[int(token_id)] - torch.logsumexp(row_logits, dim=-1)
            values.append(float(score.item()))
        token_scores[candidate] = values
        scores[candidate] = float(sum(values))
        finite = finite and all(math.isfinite(value) for value in values)
    return scores, token_scores, finite


def fixed_margin(scores: dict[str, float], donor_gold: str, recipient_gold: str) -> float:
    return float(scores[donor_gold] - scores[recipient_gold])


def token_margin(
    token_scores: dict[str, list[float]], donor_gold: str, recipient_gold: str, offset: int
) -> float:
    return float(token_scores[donor_gold][offset] - token_scores[recipient_gold][offset])


def relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.sqrt(torch.mean((left.float() - right.float()) ** 2))
    denominator = torch.sqrt(torch.mean(right.float() ** 2))
    return float((numerator / (denominator + EPSILON)).item())


def condition_result(
    name: str,
    logits: torch.Tensor,
    candidates: list[str],
    candidate_token_ids: dict[str, list[int]],
    donor_gold: str,
    recipient_gold: str,
    recipient_margin: float,
    donor_margin: float,
    donor_scores: dict[str, float],
    donor_tokens: dict[str, list[float]],
    divergence: int,
    donor_boundary_logits: torch.Tensor,
    calls: int,
    write_max_abs: float,
) -> dict[str, Any]:
    scores, tokens, finite = score_from_logits(logits, candidates, candidate_token_ids)
    margin = fixed_margin(scores, donor_gold, recipient_gold)
    target_shift = donor_margin - recipient_margin
    completion = (margin - recipient_margin) / target_shift if abs(target_shift) > EPSILON else 0.0
    score_errors = [abs(scores[key] - donor_scores[key]) for key in candidates]
    divergence_errors = [
        abs(tokens[key][divergence] - donor_tokens[key][divergence]) for key in candidates
    ]
    boundary = logits[:, 0].float()
    donor_boundary = donor_boundary_logits.float()
    return {
        "condition": name,
        "finite": bool(finite),
        "patch_calls": int(calls),
        "hook_write_max_abs": float(write_max_abs),
        "scores": scores,
        "token_scores": tokens,
        "fixed_margin": margin,
        "completion": float(completion),
        "score_max_abs_vs_donor": float(max(score_errors)),
        "divergence_token_score_max_abs_vs_donor": float(max(divergence_errors)),
        "boundary_logit_max_abs_vs_donor": float((boundary - donor_boundary).abs().max().item()),
        "boundary_logit_rms_vs_donor": float(torch.sqrt(torch.mean((boundary - donor_boundary) ** 2)).item()),
        "boundary_top1_agreement": float(
            (boundary.argmax(dim=-1) == donor_boundary.argmax(dim=-1)).float().mean().item()
        ),
    }


def run() -> None:
    protocol, manifest, state_by_id = verify_formal_inputs()
    if RECORD_PATH.exists() or FINAL_PATH.exists():
        raise RuntimeError("formal output already exists")
    source_arrays = np.load(SOURCE_ARRAYS, mmap_mode="r")
    stored_boundary = source_arrays["residual_boundary"]
    started = time.time()
    model = None
    records: list[dict[str, Any]] = []
    try:
        model, _tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError("Qwen3 is not pure FP16")
        layers = get_layers(model)
        if len(layers) != LAYER_COUNT:
            raise RuntimeError("layer count drift")
        last_layer = layers[-1]

        for index, item in enumerate(manifest):
            recipient = state_by_id[item["recipient_state_id"]]
            donor = state_by_id[item["donor_state_id"]]
            candidates = list(item["candidates"])
            candidate_ids = {
                key: [int(value) for value in item["candidate_token_ids"][key]] for key in candidates
            }
            recipient_input, continuation_length, boundary = make_batch(recipient, candidates, device)
            donor_input, donor_continuation, donor_boundary = make_batch(donor, candidates, device)
            if continuation_length != donor_continuation or boundary != donor_boundary:
                raise RuntimeError("pair sequence geometry drift")

            donor_logits, donor_hidden = forward_capture(
                model, last_layer, donor_input, continuation_length
            )
            recipient_logits, recipient_hidden = forward_capture(
                model, last_layer, recipient_input, continuation_length
            )
            donor_scores, donor_tokens, donor_finite = score_from_logits(
                donor_logits, candidates, candidate_ids
            )
            recipient_scores, recipient_tokens, recipient_finite = score_from_logits(
                recipient_logits, candidates, candidate_ids
            )
            donor_margin = fixed_margin(
                donor_scores, item["donor_gold"], item["recipient_gold"]
            )
            recipient_margin = fixed_margin(
                recipient_scores, item["donor_gold"], item["recipient_gold"]
            )
            target_shift = donor_margin - recipient_margin
            divergence = int(item["gold_first_divergence"])

            donor_prompt = prompt_logits(model, donor["input_ids"], device)
            recipient_prompt = prompt_logits(model, recipient["input_ids"], device)
            donor_prompt_full_max = float(
                (donor_prompt.float() - donor_logits[0, 0].float()).abs().max().item()
            )
            recipient_prompt_full_max = float(
                (recipient_prompt.float() - recipient_logits[0, 0].float()).abs().max().item()
            )
            donor_boundary_row_spread = float(
                (donor_logits[:, 0].float() - donor_logits[0:1, 0].float()).abs().max().item()
            )

            stored_vector = torch.tensor(
                stored_boundary[int(item["donor_state_index"]), LAYER_COUNT].astype(np.float32),
                device=device,
                dtype=donor_hidden.dtype,
            )
            stored_replacement = donor_hidden.clone()
            stored_replacement[:, boundary, :] = stored_vector[None, :]
            stored_live_relative = relative_rms(
                stored_vector, donor_hidden[0, boundary]
            )

            condition_specs = {
                "boundary_live": ([boundary], donor_hidden),
                "boundary_stored": ([boundary], stored_replacement),
                "divergence_live": ([boundary + divergence], donor_hidden),
                "all_scoring_live": (
                    list(range(boundary, boundary + continuation_length)), donor_hidden
                ),
                "all_scoring_zero": (
                    list(range(boundary, boundary + continuation_length)), recipient_hidden
                ),
            }
            conditions: dict[str, Any] = {}
            for name in CONDITIONS:
                positions, replacement = condition_specs[name]
                patched_logits, calls, write_max_abs = forward_patch(
                    model,
                    last_layer,
                    recipient_input,
                    continuation_length,
                    positions,
                    replacement,
                )
                conditions[name] = condition_result(
                    name,
                    patched_logits,
                    candidates,
                    candidate_ids,
                    item["donor_gold"],
                    item["recipient_gold"],
                    recipient_margin,
                    donor_margin,
                    donor_scores,
                    donor_tokens,
                    divergence,
                    donor_logits[:, 0],
                    calls,
                    write_max_abs,
                )
                del patched_logits

            record: dict[str, Any] = {
                "schema_version": "phase1224.construct-record.v1",
                "phase": PHASE,
                "protocol_digest": protocol["protocol_digest"],
                "pair_id": item["pair_id"],
                "scope": item["scope"],
                "split": item["split"],
                "recipient_gold": item["recipient_gold"],
                "donor_gold": item["donor_gold"],
                "continuation_length": continuation_length,
                "gold_first_divergence": divergence,
                "gold_first_token_discriminative": bool(divergence == 0),
                "finite": bool(donor_finite and recipient_finite),
                "recipient_scores": recipient_scores,
                "donor_scores": donor_scores,
                "recipient_token_scores": recipient_tokens,
                "donor_token_scores": donor_tokens,
                "recipient_margin": recipient_margin,
                "donor_margin": donor_margin,
                "target_shift": target_shift,
                "target_shift_abs": abs(target_shift),
                "donor_prompt_full_logit_max_abs": donor_prompt_full_max,
                "recipient_prompt_full_logit_max_abs": recipient_prompt_full_max,
                "donor_boundary_row_spread_max_abs": donor_boundary_row_spread,
                "stored_live_hidden_relative": stored_live_relative,
                "conditions": conditions,
            }
            record["record_digest"] = digest(record)
            records.append(record)

            del (
                donor_logits,
                recipient_logits,
                donor_hidden,
                recipient_hidden,
                donor_prompt,
                recipient_prompt,
                recipient_input,
                donor_input,
                stored_replacement,
                stored_vector,
            )
            if (index + 1) % 16 == 0:
                print(f"[phase1224/run] {index + 1}/{len(manifest)}", flush=True)

        write_jsonl(RECORD_PATH, records)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "manifest_digest": protocol["material"]["manifest_digest"],
            "record_count": len(records),
            "record_digest": digest(records),
            "precision_audit": precision,
            "placement": placement,
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        }
        summary["summary_digest"] = digest(summary)
        write_json(RUN_SUMMARY_PATH, summary)
        print(canonical_json({"status": "run_complete", "records": len(records), "summary_digest": summary["summary_digest"]}))
    finally:
        del source_arrays
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def fraction(values: list[bool]) -> float:
    return float(sum(bool(value) for value in values) / len(values)) if values else float("nan")


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    holdout = [row for row in records if row["split"] in HOLDOUT_SPLITS]
    holdout_lcp0 = [row for row in holdout if row["gold_first_token_discriminative"]]
    holdout_lcp_pos = [row for row in holdout if not row["gold_first_token_discriminative"]]
    all_conditions = [
        condition for row in records for condition in row["conditions"].values()
    ]
    boundary_live = [row["conditions"]["boundary_live"] for row in records]
    all_live = [row["conditions"]["all_scoring_live"] for row in records]
    all_zero = [row["conditions"]["all_scoring_zero"] for row in records]
    divergence_live = [row["conditions"]["divergence_live"] for row in records]
    holdout_lcp0_completion = [
        row["conditions"]["boundary_live"]["completion"] for row in holdout_lcp0
    ]
    holdout_lcp_pos_completion = [
        abs(row["conditions"]["boundary_live"]["completion"]) for row in holdout_lcp_pos
    ]
    metrics = {
        "record_count": len(records),
        "finite_fraction": fraction(
            [row["finite"] and all(c["finite"] for c in row["conditions"].values()) for row in records]
        ),
        "hook_write_max_abs": max(condition["hook_write_max_abs"] for condition in all_conditions),
        "boundary_live_logit_max_abs": max(condition["boundary_logit_max_abs_vs_donor"] for condition in boundary_live),
        "boundary_live_top1_agreement": min(condition["boundary_top1_agreement"] for condition in boundary_live),
        "all_scoring_score_max_abs": max(condition["score_max_abs_vs_donor"] for condition in all_live),
        "all_scoring_completion_median": median([condition["completion"] for condition in all_live]),
        "zero_score_max_abs": max(
            abs(condition["scores"][candidate] - row["recipient_scores"][candidate])
            for row, condition in zip(records, all_zero)
            for candidate in row["recipient_scores"]
        ),
        "prompt_full_logit_max_abs": max(
            max(row["donor_prompt_full_logit_max_abs"], row["recipient_prompt_full_logit_max_abs"])
            for row in records
        ),
        "stored_live_hidden_relative_max": max(row["stored_live_hidden_relative"] for row in records),
        "target_shift_abs_min": min(row["target_shift_abs"] for row in records),
        "divergence_token_score_max_abs": max(
            condition["divergence_token_score_max_abs_vs_donor"] for condition in divergence_live
        ),
        "holdout_lcp0_count": len(holdout_lcp0),
        "holdout_lcp0_boundary_completion_median": median(holdout_lcp0_completion),
        "holdout_lcp0_positive_fraction": fraction([value > 0 for value in holdout_lcp0_completion]),
        "holdout_lcp_positive_count": len(holdout_lcp_pos),
        "holdout_lcp_positive_abs_completion_median": median(holdout_lcp_pos_completion),
        "discovery_lcp0_boundary_completion_median": median([
            row["conditions"]["boundary_live"]["completion"]
            for row in records
            if row["split"] == "discovery" and row["gold_first_token_discriminative"]
        ]),
    }
    gates = {
        "finite": metrics["finite_fraction"] >= THRESHOLDS["finite_fraction_min"],
        "hook_write": metrics["hook_write_max_abs"] <= THRESHOLDS["hook_write_max_abs_max"],
        "boundary_next_logit": metrics["boundary_live_logit_max_abs"] <= THRESHOLDS["boundary_live_logit_max_abs_max"],
        "boundary_top1": metrics["boundary_live_top1_agreement"] >= THRESHOLDS["boundary_live_top1_agreement_min"],
        "sustained_score": metrics["all_scoring_score_max_abs"] <= THRESHOLDS["all_scoring_score_max_abs_max"],
        "sustained_completion": metrics["all_scoring_completion_median"] >= THRESHOLDS["all_scoring_completion_median_min"],
        "zero_identity": metrics["zero_score_max_abs"] <= THRESHOLDS["zero_score_max_abs_max"],
        "prompt_full_parity": metrics["prompt_full_logit_max_abs"] <= THRESHOLDS["prompt_full_logit_max_abs_max"],
        "stored_replay": metrics["stored_live_hidden_relative_max"] <= THRESHOLDS["stored_live_hidden_relative_max"],
        "denominator": metrics["target_shift_abs_min"] >= THRESHOLDS["target_shift_abs_min"],
        "divergence_token": metrics["divergence_token_score_max_abs"] <= THRESHOLDS["divergence_token_score_max_abs_max"],
        "holdout_lcp0_completion": metrics["holdout_lcp0_boundary_completion_median"] >= THRESHOLDS["holdout_lcp0_boundary_completion_median_min"],
        "holdout_lcp0_positive": metrics["holdout_lcp0_positive_fraction"] >= THRESHOLDS["holdout_lcp0_positive_fraction_min"],
        "holdout_lcp_positive_near_zero": metrics["holdout_lcp_positive_abs_completion_median"] <= THRESHOLDS["holdout_lcp_positive_abs_completion_median_max"],
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def analyze() -> None:
    protocol, manifest, _states = verify_formal_inputs()
    records = read_jsonl(RECORD_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    if len(records) != len(manifest) or summary["record_digest"] != digest(records):
        raise RuntimeError("run output digest drift")
    result = aggregate(records)
    passed = bool(result["passed"])
    k_item = {
        "identifier": "K201",
        "evidence_grade": "E3-METHOD" if passed else "E3-INSTRUMENT-BOUNDARY",
        "statement": (
            "Final-layer patch identity and sustained scoring-position replay passed; Phase1223's near-zero full-score completion is explained by one-step intervention against later discriminating candidate tokens."
            if passed
            else "The final-layer patch construct failed at least one frozen identity or prospective timing gate."
        ),
        "scope": "Qwen3 FP16; Phase1223 generated pairs; final decoder residual; construct validity only",
    }
    final: dict[str, Any] = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "construct_confirmed" if passed else "construct_not_confirmed",
        "protocol_digest": protocol["protocol_digest"],
        "manifest_digest": protocol["material"]["manifest_digest"],
        "run_summary_digest": summary["summary_digest"],
        "result": result,
        "k_item": k_item,
        "k200_scope_refinement": {
            "frozen_k200_not_deleted": True,
            "allowed": "one prompt-boundary patch is insufficient for the full multi-token continuation score",
            "not_allowed": "the answer boundary has no causal information or no transportable next-token state",
        },
        "authorized_next": {
            "automatic_execution": passed,
            "experiment": "Phase1225 known-truth finite distributed-process intervention basis" if passed else None,
            "qwen_new_mechanism_scan": False,
            "reason": (
                "construct audit passed; calibrate single-site, distributed, and recurrent mechanisms before any new Qwen intervention"
                if passed
                else "repair the intervention instrument before further causal work"
            ),
        },
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "passed": passed, "final_digest": final["final_digest"]}))


def selftest() -> None:
    assert first_divergence([1, 2], [1, 3]) == 1
    assert first_divergence([1, 2], [3, 2]) == 0
    scores = {"a": 1.0, "b": 3.0}
    assert fixed_margin(scores, "b", "a") == 2.0
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})
    print("phase1224 selftest passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("materialize", "run", "analyze", "all", "selftest"), required=True)
    args = parser.parse_args()
    if args.stage == "materialize":
        materialize()
    elif args.stage == "run":
        run()
    elif args.stage == "analyze":
        analyze()
    elif args.stage == "all":
        materialize()
        run()
        analyze()
    else:
        selftest()


if __name__ == "__main__":
    main()
