#!/usr/bin/env python3
"""Run Phase573 behavior qualification and coarse semantic message interventions."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase569_relation_competition_behavior import classify  # noqa: E402
from phase569_role_position_utils import role_positions  # noqa: E402
from phase573_natural_transition_behavior import balanced_worlds, stable_expected  # noqa: E402
import phase573_natural_transition_protocol as protocol  # noqa: E402
import phase573_coarse_message_causal_protocol as causal_protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
MODEL = causal_protocol.MODEL
SOURCE_TARGET_ROLES = (
    "target_fact_object", "target_fact_relation", "target_fact_value",
)
SOURCE_OTHER_ROLES = (
    "other_fact_object", "other_fact_relation", "other_fact_value",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def behavior_rows_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_causal_split_behavior_rows.jsonl.gz"


def causal_rows_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_coarse_message_causal_rows.jsonl.gz"


def registry_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_causal_split_registry.json"


def summary_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_coarse_message_causal_summary.json"


def decision_path() -> Path:
    return OUT_DIR / "phase573_coarse_message_causal_decision.json"


def contract_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_coarse_message_causal_contract.json"


def replace_primary(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    return value


def edge_contribution(
    module: Any,
    attention_weights: torch.Tensor,
    value_states: torch.Tensor,
    batch_index: int,
    receiver_position: int,
    source_positions: list[int],
) -> torch.Tensor:
    weights = attention_weights[batch_index, :, receiver_position, source_positions]
    values = value_states[batch_index, :, source_positions, :]
    head_output = (weights.unsqueeze(-1) * values).sum(dim=1)
    return F.linear(head_output.reshape(-1), module.o_proj.weight, bias=None)


def reconstructed_receiver(
    module: Any,
    attention_weights: torch.Tensor,
    value_states: torch.Tensor,
    batch_index: int,
    receiver_position: int,
) -> torch.Tensor:
    head_output = torch.matmul(
        attention_weights[
            batch_index, :, receiver_position:receiver_position + 1, :
        ],
        value_states[batch_index],
    ).squeeze(1)
    return F.linear(
        head_output.reshape(-1), module.o_proj.weight, module.o_proj.bias
    )


def generate_batch(
    loaded: Any, rows: list[dict[str, Any]], repeat: str, max_new_tokens: int
) -> list[dict[str, Any]]:
    loaded.tokenizer.padding_side = "left"
    prompts = [render_chat(loaded.tokenizer, MODEL, row["raw_prompt"]) for row in rows]
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    prompt_width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    output = []
    for index, row in enumerate(rows):
        text = loaded.tokenizer.decode(
            generated[index, prompt_width:], skip_special_tokens=True
        )
        output.append({
            **row,
            **classify({
                **row,
                "candidate_token_ids": row["candidate_token_ids_by_model"][MODEL],
            }, text),
            "model": MODEL,
            "execution_repeat": repeat,
            "observer_only": True,
            "causal": False,
        })
    del encoded, generated
    return output


def run_behavior_stage(
    loaded: Any, case_bank: dict[str, dict[str, Any]], max_new_tokens: int
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    by_world_variant = {
        (row["base_case_id"], row["variant"]): row for row in case_bank.values()
    }
    base_rows = {
        row["base_case_id"]: row
        for row in case_bank.values() if row["variant"] == "base"
    }
    output_rows: list[dict[str, Any]] = []
    final_selected: dict[str, list[str]] = {}
    diagnostics: dict[str, Any] = {}
    for split in causal_protocol.CAUSAL_SPLITS:
        relation_cases = sorted(
            [
                row for row in case_bank.values()
                if row["split"] == split
                and row["variant"] in ("base", "relation_swap")
            ],
            key=lambda row: row["case_id"],
        )
        for repeat in ("noop1", "noop2"):
            for start in range(0, len(relation_cases), causal_protocol.BEHAVIOR_BATCH_SIZE):
                batch = relation_cases[start:start + causal_protocol.BEHAVIOR_BATCH_SIZE]
                output_rows.extend(generate_batch(loaded, batch, repeat, max_new_tokens))
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase573 {split} "
                f"relation/{repeat} {len(relation_cases)}/{len(relation_cases)}",
                flush=True,
            )
        by_case_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output_rows
        }
        world_ids = sorted({row["base_case_id"] for row in relation_cases})
        relation_eligible = [
            base_id for base_id in world_ids
            if stable_expected(by_case_repeat, f"{base_id}_base")
            and stable_expected(by_case_repeat, f"{base_id}_relation_swap")
        ]
        if len(relation_eligible) < causal_protocol.RELATION_SCREEN_MINIMUM:
            raise RuntimeError(
                f"Phase573 causal behavior relation gate failed: "
                f"{split}/{len(relation_eligible)}"
            )
        controls_selected = balanced_worlds(
            base_rows, relation_eligible, causal_protocol.CONTROL_SCREEN_CAP
        )
        controls = sorted(
            [
                by_world_variant[(base_id, variant)]
                for base_id in controls_selected
                for variant in ("object_swap", "order_swap")
            ],
            key=lambda row: row["case_id"],
        )
        for repeat in ("noop1", "noop2"):
            for start in range(0, len(controls), causal_protocol.BEHAVIOR_BATCH_SIZE):
                batch = controls[start:start + causal_protocol.BEHAVIOR_BATCH_SIZE]
                output_rows.extend(generate_batch(loaded, batch, repeat, max_new_tokens))
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase573 {split} "
                f"controls/{repeat} {len(controls)}/{len(controls)}",
                flush=True,
            )
        by_case_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output_rows
        }
        all_axis = [
            base_id for base_id in controls_selected
            if stable_expected(by_case_repeat, f"{base_id}_object_swap")
            and stable_expected(by_case_repeat, f"{base_id}_order_swap")
        ]
        final_selected[split] = balanced_worlds(
            base_rows, all_axis, causal_protocol.FINAL_WORLDS_PER_SPLIT
        )
        if len(final_selected[split]) != causal_protocol.FINAL_WORLDS_PER_SPLIT:
            raise RuntimeError(
                f"Phase573 causal behavior all-axis gate failed: "
                f"{split}/{len(final_selected[split])}"
            )
        diagnostics[split] = {
            "relation_qualified_world_count": len(relation_eligible),
            "control_screen_world_count": len(controls_selected),
            "all_axis_qualified_world_count": len(all_axis),
            "selected_world_count": len(final_selected[split]),
        }
    return output_rows, final_selected, diagnostics


def positions_for_batch(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]], dict[str, torch.Tensor]]:
    tokenizer.padding_side = "right"
    prompts = [render_chat(tokenizer, MODEL, row["raw_prompt"]) for row in rows]
    individual = [
        role_positions(tokenizer, prompt, row)
        for prompt, row in zip(prompts, rows)
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    padded_positions = []
    for batch_index, (ids, groups) in enumerate(individual):
        active_ids = encoded["input_ids"][batch_index][
            encoded["attention_mask"][batch_index].bool()
        ].tolist()
        if [int(value) for value in active_ids] != ids:
            raise RuntimeError("Phase573 causal tokenization drift")
        padded_positions.append({
            "selected_source": sorted(
                pos for role in SOURCE_TARGET_ROLES for pos in groups[role]
            ),
            "nonselected_source": sorted(
                pos for role in SOURCE_OTHER_ROLES for pos in groups[role]
            ),
            "answer_boundary": groups["answer_boundary"][-1],
            "query_terminal": groups["query_terminal"][-1],
        })
    position_ids = encoded["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
    encoded["position_ids"] = position_ids
    return prompts, padded_positions, encoded


def capture_contributions(
    loaded: Any,
    layers: list[Any],
    encoded: dict[str, torch.Tensor],
    padded_positions: list[dict[str, Any]],
    capture_layers: tuple[int, ...],
) -> tuple[dict[int, dict[str, dict[str, torch.Tensor]]], torch.Tensor, float]:
    captures: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    errors = []

    def hook_for(layer_index: int):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or not isinstance(output, tuple) or output[1] is None:
                raise RuntimeError("Phase573 causal capture requires eager attention weights")
            primary, weights = output[0], output[1]
            batch, sequence, _ = hidden.shape
            values = module.v_proj(hidden).view(
                batch, sequence, -1, module.head_dim
            ).transpose(1, 2)
            values = values.repeat_interleave(module.num_key_value_groups, dim=1)
            layer_capture: dict[str, dict[str, torch.Tensor]] = {}
            for receiver in ("answer_boundary", "query_terminal"):
                selected_rows = []
                nonselected_rows = []
                for batch_index in range(batch):
                    receiver_position = int(padded_positions[batch_index][receiver])
                    selected_rows.append(edge_contribution(
                        module, weights, values, batch_index, receiver_position,
                        padded_positions[batch_index]["selected_source"],
                    ))
                    nonselected_rows.append(edge_contribution(
                        module, weights, values, batch_index, receiver_position,
                        padded_positions[batch_index]["nonselected_source"],
                    ))
                    reconstructed = reconstructed_receiver(
                        module, weights, values, batch_index, receiver_position
                    )
                    actual = primary[batch_index, receiver_position]
                    errors.append(float(
                        (reconstructed.float() - actual.float()).norm().item()
                        / max(actual.float().norm().item(), 1e-8)
                    ))
                layer_capture[receiver] = {
                    "selected": torch.stack(selected_rows).detach(),
                    "nonselected": torch.stack(nonselected_rows).detach(),
                }
            captures[layer_index] = layer_capture
        return hook

    handles = [
        layers[layer_index].self_attn.register_forward_hook(
            hook_for(layer_index), with_kwargs=True
        )
        for layer_index in capture_layers
    ]
    with torch.inference_mode():
        result = loaded.model(
            **encoded, use_cache=False, output_attentions=True, return_dict=True
        )
    for handle in handles:
        handle.remove()
    if set(captures) != set(capture_layers):
        raise RuntimeError("Phase573 causal contribution capture drift")
    boundary_logits = torch.stack([
        result.logits[index, int(padded_positions[index]["answer_boundary"])].float().cpu()
        for index in range(len(padded_positions))
    ])
    del result
    return captures, boundary_logits, max(errors)


def deterministic_roll(case_id: str, hidden: int) -> int:
    value = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16], 16)
    return 1 + value % max(1, hidden - 1)


def causal_batch(
    loaded: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    conditions: tuple[str, ...],
    candidate_layer: int,
    wrong_layer: int,
) -> tuple[list[dict[str, Any]], float, int]:
    prompts, positions, encoded_cpu = positions_for_batch(loaded.tokenizer, rows)
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}
    captures, natural_logits, reconstruction_error = capture_contributions(
        loaded, layers, encoded, positions, (wrong_layer, candidate_layer)
    )
    local_by_case = {row["case_id"]: index for index, row in enumerate(rows)}
    donor_index = []
    for row in rows:
        donor_id = f"{row['base_case_id']}_{'relation_swap' if row['variant'] == 'base' else 'base'}"
        if donor_id not in local_by_case:
            raise RuntimeError("Phase573 causal batch split a relation pair")
        donor_index.append(local_by_case[donor_id])
    batch_indices = torch.arange(len(rows), device=loaded.input_device)
    output_rows: list[dict[str, Any]] = []
    baseline_scores: list[dict[str, float]] | None = None
    baseline_identity_mismatch = 0

    for condition in conditions:
        patch_layer = wrong_layer if condition == "wrong_depth_donor_replace" else candidate_layer
        receiver = "query_terminal" if condition == "wrong_position_donor_replace" else "answer_boundary"
        layer_capture = captures[patch_layer][receiver]
        deltas = []
        patch_positions = []
        rolls = []
        for local, row in enumerate(rows):
            recipient_selected = layer_capture["selected"][local]
            recipient_other = layer_capture["nonselected"][local]
            donor_selected = layer_capture["selected"][donor_index[local]]
            roll = deterministic_roll(row["case_id"], donor_selected.shape[-1])
            if condition == "same_case_restore":
                delta = torch.zeros_like(recipient_selected)
            elif condition == "selected_edge_remove":
                delta = -recipient_selected
            elif condition == "nonselected_edge_remove":
                delta = -recipient_other
            elif condition in (
                "paired_relation_selected_replace",
                "wrong_depth_donor_replace",
                "wrong_position_donor_replace",
            ):
                delta = donor_selected - recipient_selected
            elif condition == "channel_roll_donor_replace":
                delta = torch.roll(donor_selected, roll, dims=-1) - recipient_selected
            else:
                raise RuntimeError(f"Unknown Phase573 causal condition: {condition}")
            deltas.append(delta)
            patch_positions.append(int(positions[local][receiver]))
            rolls.append(roll if condition == "channel_roll_donor_replace" else None)
        delta_tensor = torch.stack(deltas).to(
            device=loaded.input_device,
            dtype=next(loaded.model.parameters()).dtype,
        )
        position_tensor = torch.tensor(
            patch_positions, dtype=torch.long, device=loaded.input_device
        )

        def intervention(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            primary = output[0].clone() if isinstance(output, tuple) else output.clone()
            primary[batch_indices, position_tensor, :] += delta_tensor
            return replace_primary(output, primary)

        handle = layers[patch_layer].self_attn.register_forward_hook(intervention)
        with torch.inference_mode():
            result = loaded.model(
                **encoded, use_cache=False, output_attentions=False, return_dict=True
            )
        handle.remove()
        logits = result.logits.float()
        scores_for_condition = []
        for local, row in enumerate(rows):
            boundary = int(positions[local]["answer_boundary"])
            vector = logits[local, boundary]
            scores_for_condition.append({
                value: float(vector[ids[0]].item())
                for value, ids in row["candidate_token_ids_by_model"][MODEL].items()
            })
        if condition == "same_case_restore":
            baseline_scores = scores_for_condition
            for local, row in enumerate(rows):
                natural = {
                    value: float(natural_logits[local, ids[0]].item())
                    for value, ids in row["candidate_token_ids_by_model"][MODEL].items()
                }
                baseline_identity_mismatch += int(any(
                    natural[value] != scores_for_condition[local][value]
                    for value in natural
                ))
        if baseline_scores is None:
            raise RuntimeError("Phase573 same-case baseline must execute first")
        for local, row in enumerate(rows):
            donor = rows[donor_index[local]]
            scores = scores_for_condition[local]
            baseline = baseline_scores[local]
            recipient_target = row["target"]
            donor_target = donor["target"]
            baseline_switch = baseline[donor_target] - baseline[recipient_target]
            switch = scores[donor_target] - scores[recipient_target]
            output_rows.append({
                "schema_version": "phase573_coarse_message_causal_row.v1",
                "phase_id": protocol.PHASE,
                "created_at": now(),
                "model": MODEL,
                "split": row["split"],
                "base_case_id": row["base_case_id"],
                "case_id": row["case_id"],
                "variant": row["variant"],
                "condition": condition,
                "recipient_target": recipient_target,
                "donor_target": donor_target,
                "baseline_scores": baseline,
                "intervention_scores": scores,
                "baseline_switch_margin": baseline_switch,
                "intervention_switch_margin": switch,
                "donor_switch_effect": switch - baseline_switch,
                "baseline_candidate_winner": max(baseline, key=baseline.get),
                "intervention_candidate_winner": max(scores, key=scores.get),
                "intervention_donor_wins": max(scores, key=scores.get) == donor_target,
                "candidate_layer": candidate_layer,
                "patch_layer": patch_layer,
                "candidate_receiver": "answer_boundary",
                "patch_receiver": receiver,
                "selected_source_token_count": len(positions[local]["selected_source"]),
                "nonselected_source_token_count": len(positions[local]["nonselected_source"]),
                "roll_shift": rolls[local],
                "reconstruction_relative_error_batch_max": reconstruction_error,
                "post_softmax_value_contribution_intervention": True,
                "key_effect_identified": False,
                "coarse_compute_edge": condition != "same_case_restore",
                "sealed": False,
            })
        del result, logits, delta_tensor
    del encoded, encoded_cpu, captures, natural_logits, prompts
    return output_rows, reconstruction_error, baseline_identity_mismatch


def deterministic_sign_flip_count(values: list[float], permutations: int) -> dict[str, Any]:
    observed = sum(values) / max(1, len(values))
    at_least = 0
    for permutation in range(permutations):
        total = 0.0
        for index, value in enumerate(values):
            digest = hashlib.sha256(
                f"Phase573|{permutation}|{index}".encode("utf-8")
            ).digest()
            total += value if digest[0] & 1 else -value
        permuted = total / max(1, len(values))
        at_least += int(permuted >= observed)
    return {
        "observed_mean": finite(observed),
        "permutation_count": permutations,
        "count_at_least_observed": at_least,
        "smoothed_tail_fraction": (at_least + 1) / (permutations + 1),
    }


def run(max_new_tokens: int, restart: bool) -> Path:
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
    if not read_json(causal_protocol.TRACE_DECISION)["coarse_message_causal_authorized"]:
        raise RuntimeError("Phase573 causal execution is no longer authorized")
    paths = (
        behavior_rows_path(), causal_rows_path(), registry_path(), summary_path(),
        decision_path(), contract_path(),
    )
    if restart:
        for path in paths:
            path.unlink(missing_ok=True)
    contract = {
        "schema_version": "phase573_coarse_message_causal_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "causal_protocol_sha256": sha256_file(causal_protocol.CAUSAL_PROTOCOL),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "conditions": list(causal_protocol.CONDITIONS),
        "candidate_layer": frozen["candidate_layer"],
        "candidate_receiver": frozen["candidate_receiver"],
        "causal_splits": list(causal_protocol.CAUSAL_SPLITS),
        "sealed_split_read": False,
    }
    if contract_path().exists():
        existing = read_json(contract_path())
        for key in (
            "model", "causal_protocol_sha256", "open_cases_sha256", "conditions",
            "candidate_layer", "candidate_receiver", "causal_splits",
            "sealed_split_read",
        ):
            if existing[key] != contract[key]:
                raise RuntimeError(f"Phase573 causal contract drift: {key}")
    else:
        write_json(contract_path(), contract)

    case_bank = {
        row["case_id"]: row
        for row in iter_jsonl(protocol.OPEN_CASES_PATH)
        if row["split"] in causal_protocol.CAUSAL_SPLITS
    }
    if len(case_bank) != 8192 or any(row["sealed"] for row in case_bank.values()):
        raise RuntimeError(f"Phase573 causal case denominator drift: {len(case_bank)}")
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16" or len(layers) != 36:
            raise RuntimeError(
                f"Phase573 causal model drift: dtype={run_dtype}/layers={len(layers)}"
            )
        if getattr(loaded.model.config, "_attn_implementation", None) != "eager":
            raise RuntimeError("Phase573 causal message accounting requires eager attention")

        behavior_rows, selected, behavior_diagnostics = run_behavior_stage(
            loaded, case_bank, max_new_tokens
        )
        write_jsonl(behavior_rows_path(), behavior_rows)
        registry = {
            "schema_version": "phase573_causal_split_registry.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "diagnostics_by_split": behavior_diagnostics,
            "selected_base_case_ids_by_split": selected,
            "selection_uses_internal_state": False,
            "causal_splits_read": True,
            "sealed_split_read": False,
        }
        write_json(registry_path(), registry)

        causal_rows: list[dict[str, Any]] = []
        max_reconstruction = 0.0
        baseline_mismatch = 0
        for split in causal_protocol.CAUSAL_SPLITS:
            world_ids = selected[split]
            for start in range(0, len(world_ids), causal_protocol.CAUSAL_BATCH_WORLDS):
                batch_worlds = world_ids[start:start + causal_protocol.CAUSAL_BATCH_WORLDS]
                rows = sorted(
                    [
                        case_bank[f"{base_id}_{variant}"]
                        for base_id in batch_worlds
                        for variant in ("base", "relation_swap")
                    ],
                    key=lambda row: (row["base_case_id"], row["variant"]),
                )
                batch_rows, reconstruction, mismatch = causal_batch(
                    loaded, layers, rows, causal_protocol.CONDITIONS,
                    int(frozen["candidate_layer"]),
                    int(frozen["wrong_depth_control_layer"]),
                )
                causal_rows.extend(batch_rows)
                max_reconstruction = max(max_reconstruction, reconstruction)
                baseline_mismatch += mismatch
                if reconstruction > float(frozen["reconstruction_relative_error_max"]):
                    raise RuntimeError(
                        f"Phase573 contribution reconstruction failed: {reconstruction}"
                    )
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase573 coarse causal "
                f"{split} {len(world_ids)}/{len(world_ids)}",
                flush=True,
            )
        write_jsonl(causal_rows_path(), causal_rows)

        gate = frozen["causal_gate"]
        metrics_by_split: dict[str, Any] = {}
        split_passes = {}
        for split in causal_protocol.CAUSAL_SPLITS:
            rows_for_split = [row for row in causal_rows if row["split"] == split]
            by_condition = {
                condition: [
                    row for row in rows_for_split if row["condition"] == condition
                ]
                for condition in causal_protocol.CONDITIONS
            }
            expected = causal_protocol.FINAL_WORLDS_PER_SPLIT * 2
            if any(len(rows) != expected for rows in by_condition.values()):
                raise RuntimeError("Phase573 causal condition denominator drift")
            metrics = {}
            for condition, rows in by_condition.items():
                effects = [float(row["donor_switch_effect"]) for row in rows]
                metrics[condition] = {
                    "case_count": len(rows),
                    "mean_donor_switch_effect": finite(sum(effects) / len(effects)),
                    "positive_effect_rate": sum(value > 0.0 for value in effects) / len(effects),
                    "donor_candidate_win_rate": sum(
                        row["intervention_donor_wins"] for row in rows
                    ) / len(rows),
                    "sign_flip_audit": deterministic_sign_flip_count(
                        effects, int(frozen["permutations"])
                    ),
                }
            remove = metrics["selected_edge_remove"]
            replace = metrics["paired_relation_selected_replace"]
            nonselected = metrics["nonselected_edge_remove"]
            controls = [
                metrics[condition]["mean_donor_switch_effect"]
                for condition in (
                    "channel_roll_donor_replace", "wrong_depth_donor_replace",
                    "wrong_position_donor_replace",
                )
            ]
            remove_gap = (
                remove["mean_donor_switch_effect"]
                - nonselected["mean_donor_switch_effect"]
            )
            replace_gap = replace["mean_donor_switch_effect"] - max(controls)
            passed = (
                remove["positive_effect_rate"] >= gate["minimum_positive_effect_rate"]
                and replace["positive_effect_rate"] >= gate["minimum_positive_effect_rate"]
                and remove["mean_donor_switch_effect"]
                > gate["minimum_mean_donor_switch_effect"]
                and replace["mean_donor_switch_effect"]
                > gate["minimum_mean_donor_switch_effect"]
                and remove_gap >= gate["minimum_mean_gap_vs_control"]
                and replace_gap >= gate["minimum_mean_gap_vs_control"]
                and replace["donor_candidate_win_rate"]
                >= gate["minimum_donor_candidate_win_rate"]
            )
            metrics_by_split[split] = {
                "conditions": metrics,
                "selected_vs_nonselected_removal_mean_gap": finite(remove_gap),
                "paired_replace_vs_strongest_control_mean_gap": finite(replace_gap),
                "causal_gate_pass": passed,
            }
            split_passes[split] = passed
        causal_pass = all(split_passes.values())
        summary = {
            "schema_version": "phase573_coarse_message_causal_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "behavior_diagnostics_by_split": behavior_diagnostics,
            "selected_world_count_by_split": {
                split: len(ids) for split, ids in selected.items()
            },
            "behavior_row_count": len(behavior_rows),
            "causal_row_count": len(causal_rows),
            "condition_count": len(causal_protocol.CONDITIONS),
            "candidate_layer": frozen["candidate_layer"],
            "candidate_receiver": frozen["candidate_receiver"],
            "metrics_by_split": metrics_by_split,
            "split_gate_pass": split_passes,
            "coarse_message_causal_gate_pass": causal_pass,
            "maximum_reconstruction_relative_error": finite(max_reconstruction),
            "same_shape_baseline_mismatch_count": baseline_mismatch,
            "runtime_seconds": time.monotonic() - started,
            "behavior_rows_sha256": sha256_file(behavior_rows_path()),
            "causal_rows_sha256": sha256_file(causal_rows_path()),
            "post_softmax_value_contribution_intervention": True,
            "key_effect_identified": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "causal_splits_read": True,
            "sealed_split_read": False,
        }
        write_json(summary_path(), summary)
        decision = {
            "schema_version": "phase573_coarse_message_causal_decision.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "coarse_message_causal_gate_pass": causal_pass,
            "split_gate_pass": split_passes,
            "claim_allowed": (
                "local all-head semantic-fact value message at layer24 answer boundary"
                if causal_pass else None
            ),
            "claim_not_allowed": [
                "relation selection rule closure",
                "attention key or query mechanism",
                "head, channel, neuron, or parameter mechanism",
                "cross-model mechanism",
                "sealed generalization",
                "72-mechanism closure",
            ],
            "sealed_execution_authorized": causal_pass,
            "sealed_split_read": False,
        }
        write_json(decision_path(), decision)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary_path()
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
