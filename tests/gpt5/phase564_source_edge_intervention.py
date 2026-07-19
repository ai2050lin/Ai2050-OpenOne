#!/usr/bin/env python3
"""Intervene on all-head source-color attention contributions for Phase564."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase557_natural_color_source_intervention import word_scores  # noqa: E402
from phase559_binding_event_collect import child_span, semantic_positions  # noqa: E402
from phase559_causal_screen import deterministic_roll, replace_primary  # noqa: E402


MODEL = "qwen3"
OUT_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
PATH_ROWS = OUT_DIR / "phase564_qwen3_edge_behavior_rows.jsonl"
MODE_PATHS = {
    "discovery": (
        OUT_DIR / "phase564_source_edge_discovery_frozen_contract.json",
        OUT_DIR / "phase564_source_edge_discovery_candidates.json",
        OUT_DIR / "phase564_source_edge_discovery_rows.jsonl",
        OUT_DIR / "phase564_source_edge_discovery_execution_summary.json",
    ),
    "confirmation": (
        OUT_DIR / "phase564_source_edge_confirmation_frozen_contract.json",
        OUT_DIR / "phase564_source_edge_confirmation_candidates.json",
        OUT_DIR / "phase564_source_edge_confirmation_rows.jsonl",
        OUT_DIR / "phase564_source_edge_confirmation_execution_summary.json",
    ),
    "unseen": (
        OUT_DIR / "phase564_source_edge_unseen_frozen_contract.json",
        OUT_DIR / "phase564_source_edge_unseen_candidates.json",
        OUT_DIR / "phase564_source_edge_unseen_rows.jsonl",
        OUT_DIR / "phase564_source_edge_unseen_execution_summary.json",
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_case_order(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[str, dict[int, dict[str, Any]]] = {}
    for row in cases:
        pairs.setdefault(row["pair_id"], {})[int(row["binding"])] = row
    if any(set(members) != {0, 1} for members in pairs.values()):
        raise RuntimeError("Phase564 lost a counterfactual pair")
    ordered = []
    for pair_id in sorted(pairs):
        ordered.extend((pairs[pair_id][0], pairs[pair_id][1]))
    return ordered


def token_span(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    indices = [
        index for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < end and token_end > start
    ]
    if not indices:
        raise RuntimeError(f"No token overlaps semantic span {start}:{end}")
    return indices


def semantic_edge_positions(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    ids, targets = semantic_positions(tokenizer, row)
    encoded = tokenizer(row["prompt"], add_special_tokens=True, return_offsets_mapping=True)
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if [int(value) for value in encoded["input_ids"]] != ids:
        raise RuntimeError("Phase564 offset tokenization drift")
    source_span = child_span(row["prompt"], row["source_fact"], row["target"])
    nontarget_span = child_span(
        row["prompt"], row["nontarget_fact"], row["nontarget_color"]
    )
    return ids, {
        "source_color_tokens": token_span(offsets, *source_span),
        "nontarget_color_tokens": token_span(offsets, *nontarget_span),
        "query_object_end": targets["query_object_end"],
        "answer_boundary": targets["answer_boundary"],
        "query_object_end_wrong": max(0, targets["query_object_end"] - 1),
        "answer_boundary_wrong": max(0, targets["answer_boundary"] - 1),
    }


def edge_contribution(
    module: Any,
    attention_weights: torch.Tensor,
    value_states: torch.Tensor,
    batch_index: int,
    target_position: int,
    source_positions: list[int],
) -> torch.Tensor:
    weights = attention_weights[batch_index, :, target_position, source_positions]
    values = value_states[batch_index, :, source_positions, :]
    head_output = (weights.unsqueeze(-1) * values).sum(dim=1)
    return F.linear(head_output.reshape(-1), module.o_proj.weight, bias=None)


def reconstructed_target(
    module: Any,
    attention_weights: torch.Tensor,
    value_states: torch.Tensor,
    batch_index: int,
    target_position: int,
) -> torch.Tensor:
    # Match eager_attention_forward's matmul accumulation order exactly. In
    # BF16, einsum can differ enough to trip the conservation audit.
    head_output = torch.matmul(
        attention_weights[batch_index, :, target_position:target_position + 1, :],
        value_states[batch_index],
    ).squeeze(1)
    return F.linear(head_output.reshape(-1), module.o_proj.weight, module.o_proj.bias)


def capture_batch(
    loaded: Any,
    layers: list[Any],
    encoded: dict[str, torch.Tensor],
    padded_positions: list[dict[str, Any]],
    capture_layers: list[int],
    target_roles: set[str],
) -> tuple[dict[int, dict[str, dict[str, torch.Tensor]]], torch.Tensor, float]:
    captures: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    reconstruction_errors: list[float] = []

    def capture(layer_index: int):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None or not isinstance(output, tuple) or output[1] is None:
                raise RuntimeError("Phase564 eager attention capture did not expose weights")
            primary, attention_weights = output[0], output[1]
            batch, sequence, _width = hidden_states.shape
            values = module.v_proj(hidden_states).view(
                batch, sequence, -1, module.head_dim
            ).transpose(1, 2)
            values = values.repeat_interleave(module.num_key_value_groups, dim=1)
            layer_capture: dict[str, dict[str, torch.Tensor]] = {}
            for role in target_roles:
                source_rows = []
                nontarget_rows = []
                for index in range(batch):
                    target_position = int(padded_positions[index][role])
                    source_rows.append(edge_contribution(
                        module,
                        attention_weights,
                        values,
                        index,
                        target_position,
                        padded_positions[index]["source_color_tokens"],
                    ))
                    nontarget_rows.append(edge_contribution(
                        module,
                        attention_weights,
                        values,
                        index,
                        target_position,
                        padded_positions[index]["nontarget_color_tokens"],
                    ))
                    reconstruction = reconstructed_target(
                        module, attention_weights, values, index, target_position
                    )
                    actual = primary[index, target_position]
                    relative_error = float(
                        (reconstruction.float() - actual.float()).norm().item()
                        / max(actual.float().norm().item(), 1e-8)
                    )
                    reconstruction_errors.append(relative_error)
                layer_capture[role] = {
                    "source": torch.stack(source_rows).detach(),
                    "nontarget": torch.stack(nontarget_rows).detach(),
                }
            captures[layer_index] = layer_capture
        return hook

    handles = [
        layers[layer_index].self_attn.register_forward_hook(
            capture(layer_index), with_kwargs=True
        )
        for layer_index in capture_layers
    ]
    with torch.inference_mode():
        result = loaded.model(**encoded, use_cache=False)
    for handle in handles:
        handle.remove()
    if set(captures) != set(capture_layers):
        raise RuntimeError("Phase564 attention capture hooks did not all fire")
    return captures, result.logits[:, -1, :].detach().float().cpu(), max(reconstruction_errors)


def condition_patch_layer(candidate: dict[str, Any], condition: str) -> int:
    if condition == "wrong_depth_donor_replace":
        return int(candidate["wrong_depth_control_layer"])
    return int(candidate["layer"])


def run(mode: str, batch_size: int, restart: bool) -> Path:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Phase564 intervention batch size must be a positive even number")
    contract_path, candidates_path, rows_path, summary_path = MODE_PATHS[mode]
    contract = read_json(contract_path)
    registry = read_json(candidates_path)
    if contract["model"] != MODEL or contract["evidence_policy"]["sealed_split_read"]:
        raise RuntimeError("Phase564 frozen intervention contract drift")
    if not registry.get("candidate_family_frozen_before_intervention", False):
        raise RuntimeError("Phase564 candidates were not frozen before intervention")
    selected = set(contract["selected_anchor_ids"])
    cases = paired_case_order([
        row for row in read_jsonl(PATH_ROWS)
        if row["split"] == contract["split"]
        and row["anchor_id"] in selected
        and row["semantic_correct"]
    ])
    if len(cases) != contract["recipient_case_count"]:
        raise RuntimeError("Phase564 intervention denominator drift")
    candidates = registry["candidates"]
    if len(candidates) != contract["candidate_count"]:
        raise RuntimeError("Phase564 candidate denominator drift")
    if restart:
        rows_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if rows_path.exists():
        raise RuntimeError("Phase564 resume is disabled; use --restart")

    donor_by_case: dict[str, dict[str, Any]] = {}
    for index in range(0, len(cases), 2):
        left, right = cases[index:index + 2]
        donor_by_case[left["case_id"]] = right
        donor_by_case[right["case_id"]] = left

    loaded = None
    started = time.monotonic()
    max_reconstruction_error = 0.0
    completed = 0
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        cuda_used = torch.cuda.is_available() and str(loaded.input_device).startswith("cuda")
        if run_dtype != "torch.bfloat16" or len(layers) != 36 or not cuda_used:
            raise RuntimeError(
                f"Phase564 model drift: dtype={run_dtype}, layers={len(layers)}, "
                f"device={loaded.input_device}"
            )
        if getattr(loaded.model.config, "_attn_implementation", None) != "eager":
            raise RuntimeError("Phase564 requires eager attention for exact contribution accounting")

        all_conditions = list(contract["conditions"])
        capture_layers = sorted({
            condition_patch_layer(candidate, condition)
            for candidate in candidates
            for condition in all_conditions
        } | {int(candidate["layer"]) for candidate in candidates})
        target_roles = {candidate["target_role"] for candidate in candidates}
        target_roles |= {f"{role}_wrong" for role in target_roles}

        for batch_start in range(0, len(cases), batch_size):
            batch_rows = cases[batch_start:batch_start + batch_size]
            batch_case_ids = {row["case_id"] for row in batch_rows}
            if any(donor_by_case[row["case_id"]]["case_id"] not in batch_case_ids for row in batch_rows):
                raise RuntimeError("Phase564 batch split a counterfactual pair")
            local_by_case = {row["case_id"]: index for index, row in enumerate(batch_rows)}
            individual = [semantic_edge_positions(loaded.tokenizer, row) for row in batch_rows]
            encoded = loaded.tokenizer(
                [row["prompt"] for row in batch_rows],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            sequence_length = int(encoded["input_ids"].shape[1])
            padded_positions = []
            for index, (ids, positions) in enumerate(individual):
                mask_ids = encoded["input_ids"][index][encoded["attention_mask"][index].bool()].tolist()
                if [int(value) for value in mask_ids] != ids:
                    raise RuntimeError("Phase564 padded tokenization drift")
                left_padding = sequence_length - len(ids)
                padded_positions.append({
                    key: (
                        [left_padding + int(value) for value in value]
                        if isinstance(value, list)
                        else left_padding + int(value)
                    )
                    for key, value in positions.items()
                })
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            captures, _natural_logits, reconstruction_error = capture_batch(
                loaded, layers, encoded, padded_positions, capture_layers, target_roles
            )
            max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
            if reconstruction_error > float(contract["reconstruction_relative_error_max"]):
                raise RuntimeError(
                    f"Phase564 source contribution reconstruction failed: {reconstruction_error}"
                )

            baseline_scores: dict[str, list[dict[str, float]]] = {}
            primary_specs = [
                (candidate, condition)
                for candidate in candidates
                for condition in all_conditions
                if condition != "wrong_depth_donor_replace"
            ]
            deferred_specs = [
                (candidate, condition)
                for candidate in candidates
                for condition in all_conditions
                if condition == "wrong_depth_donor_replace"
            ]
            for pass_specs in (primary_specs, deferred_specs):
                specs_by_layer: dict[int, list[tuple[dict[str, Any], str]]] = defaultdict(list)
                for candidate, condition in pass_specs:
                    specs_by_layer[condition_patch_layer(candidate, condition)].append((candidate, condition))
                for patch_layer in sorted(specs_by_layer):
                    specs = specs_by_layer[patch_layer]
                    expanded_encoded = {
                        key: value.repeat((len(specs),) + (1,) * (value.ndim - 1))
                        for key, value in encoded.items()
                    }
                    deltas: list[torch.Tensor] = []
                    patch_positions: list[torch.Tensor] = []
                    replacement_sources: list[list[str]] = []
                    roll_shifts: list[list[int | None]] = []
                    for candidate, condition in specs:
                        candidate_id = candidate["candidate_id"]
                        target_role = candidate["target_role"]
                        role = f"{target_role}_wrong" if condition == "wrong_target_donor_replace" else target_role
                        layer_capture = captures[patch_layer][role]
                        condition_deltas = []
                        condition_positions = []
                        condition_sources = []
                        condition_rolls = []
                        for local, row in enumerate(batch_rows):
                            donor = donor_by_case[row["case_id"]]
                            donor_local = local_by_case[donor["case_id"]]
                            recipient_source = layer_capture["source"][local]
                            donor_source = layer_capture["source"][donor_local]
                            nontarget_source = layer_capture["nontarget"][local]
                            roll = deterministic_roll(candidate_id, row["case_id"], donor_source.shape[-1])
                            if condition == "same_case_restore":
                                delta = torch.zeros_like(recipient_source)
                                source = "same_case_subtract_add_identity"
                            elif condition == "source_edge_remove":
                                delta = -recipient_source
                                source = "recipient_source_color_edge_removed"
                            elif condition == "paired_donor_edge_replace":
                                delta = donor_source - recipient_source
                                source = "paired_counterfactual_source_color_edge"
                            elif condition == "nontarget_source_edge_replace":
                                delta = nontarget_source - recipient_source
                                source = "same_case_nontarget_fact_color_edge"
                            elif condition == "wrong_target_donor_replace":
                                delta = donor_source - recipient_source
                                source = "paired_donor_edge_at_one_token_left_target"
                            elif condition == "wrong_depth_donor_replace":
                                delta = donor_source - recipient_source
                                source = "paired_donor_edge_at_wrong_depth"
                            elif condition == "channel_roll_donor_replace":
                                delta = torch.roll(donor_source, roll, dims=-1) - recipient_source
                                source = "channel_rolled_paired_donor_edge"
                            else:
                                raise RuntimeError(f"Unsupported Phase564 condition: {condition}")
                            condition_deltas.append(delta)
                            condition_positions.append(int(padded_positions[local][role]))
                            condition_sources.append(source)
                            condition_rolls.append(roll if condition == "channel_roll_donor_replace" else None)
                        deltas.append(torch.stack(condition_deltas).to(
                            device=loaded.input_device,
                            dtype=next(loaded.model.parameters()).dtype,
                        ))
                        patch_positions.append(torch.tensor(
                            condition_positions, dtype=torch.long, device=loaded.input_device
                        ))
                        replacement_sources.append(condition_sources)
                        roll_shifts.append(condition_rolls)

                    def intervention_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                        primary = output[0].clone() if isinstance(output, tuple) else output.clone()
                        local_batch = len(batch_rows)
                        for spec_index in range(len(specs)):
                            batch_indices = spec_index * local_batch + torch.arange(
                                local_batch, device=primary.device
                            )
                            primary[batch_indices, patch_positions[spec_index], :] += deltas[spec_index]
                        return replace_primary(output, primary)

                    handle = layers[patch_layer].self_attn.register_forward_hook(intervention_hook)
                    with torch.inference_mode():
                        result = loaded.model(**expanded_encoded, use_cache=False)
                    handle.remove()
                    logits = result.logits[:, -1, :].detach().float().cpu()
                    for spec_index, (candidate, condition) in enumerate(specs):
                        candidate_id = candidate["candidate_id"]
                        scores_for_spec = [
                            word_scores(
                                logits[spec_index * len(batch_rows) + index],
                                loaded.tokenizer,
                                row["all_candidates"],
                            )
                            for index, row in enumerate(batch_rows)
                        ]
                        if condition == "same_case_restore":
                            baseline_scores[candidate_id] = scores_for_spec
                        if candidate_id not in baseline_scores:
                            raise RuntimeError("Phase564 same-shape baseline was not executed first")
                        output_rows = []
                        for index, recipient in enumerate(batch_rows):
                            donor = donor_by_case[recipient["case_id"]]
                            baseline = baseline_scores[candidate_id][index]
                            scores = scores_for_spec[index]
                            baseline_switch_margin = baseline[donor["target"]] - baseline[recipient["target"]]
                            switch_margin = scores[donor["target"]] - scores[recipient["target"]]
                            output_rows.append({
                                "schema_version": "phase564_source_edge_intervention.v1",
                                "phase_id": "Phase564",
                                "created_at": now(),
                                "mode": mode,
                                "model": MODEL,
                                "torch_dtype": run_dtype,
                                "split": contract["split"],
                                "candidate_id": candidate_id,
                                "candidate_layer": int(candidate["layer"]),
                                "patch_layer": patch_layer,
                                "target_role": candidate["target_role"],
                                "source_role": candidate["source_role"],
                                "head_scope": candidate["head_scope"],
                                "condition": condition,
                                "recipient_case_id": recipient["case_id"],
                                "donor_case_id": donor["case_id"],
                                "pair_id": recipient["pair_id"],
                                "anchor_id": recipient["anchor_id"],
                                "binding": recipient["binding"],
                                "query_object_index": recipient["query_object_index"],
                                "surface_id": recipient["surface_id"],
                                "fact_order": recipient["fact_order"],
                                "factorial_cell_without_binding": (
                                    f"query{recipient['query_object_index']}_"
                                    f"surface{recipient['surface_id']}_order{recipient['fact_order']}"
                                ),
                                "color_regime": recipient["color_regime"],
                                "recipient_target": recipient["target"],
                                "donor_target": donor["target"],
                                "baseline_scores": baseline,
                                "intervention_scores": scores,
                                "baseline_switch_margin": baseline_switch_margin,
                                "intervention_switch_margin": switch_margin,
                                "donor_switch_effect": switch_margin - baseline_switch_margin,
                                "removal_damage": (
                                    switch_margin - baseline_switch_margin
                                    if condition == "source_edge_remove" else None
                                ),
                                "baseline_prediction": max(baseline, key=baseline.get),
                                "intervention_prediction": max(scores, key=scores.get),
                                "intervention_donor_wins": max(scores, key=scores.get) == donor["target"],
                                "intervention_recipient_retained": (
                                    max(scores, key=scores.get) == recipient["target"]
                                ),
                                "source_token_count": len(individual[index][1]["source_color_tokens"]),
                                "nontarget_source_token_count": len(
                                    individual[index][1]["nontarget_color_tokens"]
                                ),
                                "roll_shift": roll_shifts[spec_index][index],
                                "replacement_source": replacement_sources[spec_index][index],
                                "reconstruction_relative_error_batch_max": reconstruction_error,
                                "post_softmax_value_contribution_intervention": True,
                                "key_effect_identified": False,
                                "compute_edge": False,
                                "sealed": False,
                            })
                        append_jsonl(rows_path, output_rows)
                        completed += len(output_rows)
                    del result, logits, expanded_encoded, deltas, patch_positions, output_rows
            del captures, encoded, _natural_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_start == 0 or completed == contract["expected_intervention_rows"]:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Phase564 {mode} "
                    f"{completed}/{contract['expected_intervention_rows']}",
                    flush=True,
                )

        rows = read_jsonl(rows_path)
        if len(rows) != contract["expected_intervention_rows"]:
            raise RuntimeError("Phase564 intervention output denominator drift")
        summary = {
            "schema_version": "phase564_source_edge_execution_summary.v1",
            "phase_id": "Phase564",
            "created_at": now(),
            "status": "complete",
            "mode": mode,
            "model": MODEL,
            "torch_dtype": run_dtype,
            "cuda_used": cuda_used,
            "case_count": len(cases),
            "candidate_count": len(candidates),
            "condition_count": len(all_conditions),
            "intervention_row_count": len(rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(rows_path),
            "maximum_reconstruction_relative_error": max_reconstruction_error,
            "effect_baseline": contract["effect_baseline"],
            "post_softmax_value_contribution_intervention": True,
            "key_effect_identified": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path, summary)
        print(summary_path)
        return summary_path
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=tuple(MODE_PATHS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.batch_size, args.restart)
