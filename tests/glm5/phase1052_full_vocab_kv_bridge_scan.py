#!/usr/bin/env python3
"""Test natural K/V swaps against true full-vocabulary output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1049_qkv_read_path_scan as route_tools
import phase1052_full_vocab_kv_bridge_protocol as protocol


PAIR_BATCH_SIZE = {"qwen3": 10, "glm4": 4, "deepseek7b": 4}


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


class OnlineKVSwap:
    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        groups: list[int],
        head_dim: int,
    ) -> None:
        self.layers = layers
        self.depths = depths
        self.groups = groups
        self.head_dim = head_dim
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.counts: dict[tuple[str, int], int] = {}
        self.handles = []

    def register(self) -> None:
        for depth in self.depths:
            attention = self.layers[depth - 1].self_attn
            self.handles.append(
                attention.k_proj.register_forward_hook(
                    self._hook("k", depth)
                )
            )
            self.handles.append(
                attention.v_proj.register_forward_hook(
                    self._hook("v", depth)
                )
            )

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.counts = {}

    def _hook(self, channel: str, depth: int):
        def hook(_module, _inputs, output):
            if self.positions is None or self.masks is None:
                raise RuntimeError("online K/V swap context missing")
            hidden = route_tools.output_tensor(output)
            if hidden.shape[0] % 2:
                raise RuntimeError("paired batch is not even")
            patched = hidden.clone()
            positions = self.positions.to(hidden.device)
            masks = self.masks.to(hidden.device)
            even = torch.arange(
                0, hidden.shape[0], 2, device=hidden.device
            )
            odd = even + 1
            for span_slot in range(positions.shape[1]):
                valid = (
                    masks[even, span_slot]
                    & masks[odd, span_slot]
                )
                pair_slots = torch.where(valid)[0]
                if len(pair_slots) == 0:
                    continue
                even_rows = even[pair_slots]
                odd_rows = odd[pair_slots]
                even_pos = positions[even_rows, span_slot]
                odd_pos = positions[odd_rows, span_slot]
                for group in self.groups:
                    start = group * self.head_dim
                    end = start + self.head_dim
                    even_value = hidden[
                        even_rows, even_pos, start:end
                    ].clone()
                    odd_value = hidden[
                        odd_rows, odd_pos, start:end
                    ].clone()
                    patched[
                        even_rows, even_pos, start:end
                    ] = odd_value
                    patched[
                        odd_rows, odd_pos, start:end
                    ] = even_value
            key = (channel, depth)
            self.counts[key] = self.counts.get(key, 0) + 1
            return route_tools.replace_output(output, patched)

        return hook

    def end(self) -> None:
        expected = {
            (channel, depth)
            for channel in ("k", "v")
            for depth in self.depths
        }
        if set(self.counts) != expected or any(
            value != 1 for value in self.counts.values()
        ):
            raise RuntimeError(f"K/V hook count drift: {self.counts}")
        self.positions = None
        self.masks = None

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def projection_width(module: Any) -> int:
    return route_tools.projection_width(module)


def pair_batch(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    site: str,
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    rows = []
    for target in target_rows:
        rows.extend((
            cases[int(target["target_case_index"])],
            cases[int(target["cross_case_index"])],
        ))
    lengths = torch.tensor(
        [len(row["input_ids"]) for row in rows],
        dtype=torch.long,
        device=device,
    )
    width = int(lengths.max().item())
    input_ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    positions = torch.zeros(
        (len(rows), protocol.MAX_ROLE_SPAN),
        dtype=torch.long,
        device=device,
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    self_ids = torch.empty(
        len(rows), dtype=torch.long, device=device
    )
    counter_ids = torch.empty_like(self_ids)
    for slot, row in enumerate(rows):
        values = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        )
        input_ids[slot, :len(values)] = values
        attention_mask[slot, :len(values)] = 1
        start, end = row["role_spans"][site]
        span = list(range(int(start), int(end) + 1))
        positions[slot, :len(span)] = torch.tensor(
            span, dtype=torch.long, device=device
        )
        masks[slot, :len(span)] = True
    for pair_slot, target in enumerate(target_rows):
        left = rows[2 * pair_slot]
        right = rows[2 * pair_slot + 1]
        left_id = int(left["expected_first_token_id"])
        right_id = int(right["expected_first_token_id"])
        self_ids[2 * pair_slot] = left_id
        counter_ids[2 * pair_slot] = right_id
        self_ids[2 * pair_slot + 1] = right_id
        counter_ids[2 * pair_slot + 1] = left_id
    return (
        input_ids,
        attention_mask,
        lengths,
        positions,
        masks,
        torch.stack((self_ids, counter_ids), dim=-1),
    )


def run_condition(
    model,
    device: torch.device,
    layers: list[Any],
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    condition: dict[str, Any] | None,
    *,
    head_dim: int,
    pad_token_id: int,
    pair_batch_size: int,
) -> dict[str, np.ndarray]:
    top1 = np.empty((len(target_rows), 2), dtype=np.int32)
    finite = np.empty((len(target_rows), 2), dtype=bool)
    margin = np.empty((len(target_rows), 2), dtype=np.float32)
    swap = None
    site = "selected_concept"
    if condition is not None:
        site = str(condition["site"])
        swap = OnlineKVSwap(
            layers,
            [int(value) for value in condition["depths"]],
            [int(value) for value in condition["groups"]],
            head_dim,
        )
        swap.register()
    try:
        for start in range(0, len(target_rows), pair_batch_size):
            target_batch = target_rows[
                start:start + pair_batch_size
            ]
            (
                input_ids,
                attention_mask,
                lengths,
                positions,
                masks,
                token_ids,
            ) = pair_batch(
                target_batch,
                cases,
                site,
                pad_token_id=pad_token_id,
                device=device,
            )
            if swap is not None:
                swap.begin(positions, masks)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            if swap is not None:
                swap.end()
            logits = output.logits.float()
            rows = torch.arange(
                logits.shape[0], device=logits.device
            )
            boundary = logits[
                rows, lengths.to(logits.device) - 1, :
            ]
            is_finite = torch.isfinite(boundary).all(dim=-1)
            safe = torch.where(
                torch.isfinite(boundary),
                boundary,
                torch.full_like(boundary, -torch.inf),
            )
            predicted = torch.argmax(safe, dim=-1)
            token_ids = token_ids.to(safe.device)
            own = safe.gather(1, token_ids[:, :1]).squeeze(1)
            counter = safe.gather(
                1, token_ids[:, 1:2]
            ).squeeze(1)
            count = len(target_batch)
            top1[start:start + count] = predicted.reshape(
                count, 2
            ).detach().cpu().numpy()
            finite[start:start + count] = is_finite.reshape(
                count, 2
            ).detach().cpu().numpy()
            margin[start:start + count] = (
                counter - own
            ).reshape(count, 2).detach().cpu().numpy()
            del output, logits, boundary, safe, predicted, own, counter
    finally:
        if swap is not None:
            swap.close()
    return {"top1": top1, "finite": finite, "margin": margin}


def clean_mask_and_coverage(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    clean: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    mask = np.ones(len(target_rows), dtype=bool)
    families = set()
    for index, target in enumerate(target_rows):
        left = cases[int(target["target_case_index"])]
        right = cases[int(target["cross_case_index"])]
        expected = (
            int(left["expected_first_token_id"]),
            int(right["expected_first_token_id"]),
        )
        mask[index] = (
            bool(clean["finite"][index].all())
            and int(clean["top1"][index, 0]) == expected[0]
            and int(clean["top1"][index, 1]) == expected[1]
        )
        if mask[index]:
            families.update((
                str(left["expected_label"]),
                str(right["expected_label"]),
            ))
    return mask, sorted(families)


def condition_metrics(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    clean: dict[str, np.ndarray],
    patched: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    valid_indices = np.where(mask)[0]
    target_exact = []
    cross_exact = []
    both = []
    either = []
    finite = []
    shifts = []
    for index in valid_indices:
        left = cases[int(target_rows[index]["target_case_index"])]
        right = cases[int(target_rows[index]["cross_case_index"])]
        left_counter = int(right["expected_first_token_id"])
        right_counter = int(left["expected_first_token_id"])
        left_hit = int(patched["top1"][index, 0]) == left_counter
        right_hit = int(patched["top1"][index, 1]) == right_counter
        target_exact.append(left_hit)
        cross_exact.append(right_hit)
        both.append(left_hit and right_hit)
        either.append(left_hit or right_hit)
        finite.append(bool(patched["finite"][index].all()))
        shifts.extend(
            (
                float(
                    patched["margin"][index, 0]
                    - clean["margin"][index, 0]
                ),
                float(
                    patched["margin"][index, 1]
                    - clean["margin"][index, 1]
                ),
            )
        )
    count = len(valid_indices)
    return {
        "clean_valid_pair_count": count,
        "patched_finite_pair_rate": (
            sum(finite) / count if count else 0.0
        ),
        "target_to_cross_top1_rate": (
            sum(target_exact) / count if count else 0.0
        ),
        "cross_to_target_top1_rate": (
            sum(cross_exact) / count if count else 0.0
        ),
        "both_counterfactual_top1_count": sum(both),
        "both_counterfactual_top1_rate": (
            sum(both) / count if count else 0.0
        ),
        "either_counterfactual_top1_rate": (
            sum(either) / count if count else 0.0
        ),
        "directional_margin_shift_median": (
            float(np.median(shifts)) if shifts else None
        ),
        "directional_margin_shift_mean": (
            float(np.mean(shifts)) if shifts else None
        ),
        "both_counterfactual_mask": [
            bool(value) for value in both
        ],
        "valid_target_indices": [
            int(target_rows[index]["target_index"])
            for index in valid_indices
        ],
    }


def rollout_pairs(
    model,
    tokenizer,
    device: torch.device,
    layers: list[Any],
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    condition: dict[str, Any],
    *,
    head_dim: int,
    steps: int,
    pair_limit: int,
    pair_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = target_rows[:pair_limit]
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for target in selected:
        case = cases[int(target["target_case_index"])]
        by_length[len(case["input_ids"])].append(target)
    records = []
    for length in sorted(by_length):
        for target_batch in chunks(
            by_length[length], pair_batch_size
        ):
            (
                base_ids,
                base_mask,
                _,
                positions,
                masks,
                _,
            ) = pair_batch(
                target_batch,
                cases,
                str(condition["site"]),
                pad_token_id=0,
                device=device,
            )
            if not bool(base_mask.all()):
                raise RuntimeError("rollout length grouping failed")

            def generate(use_swap: bool) -> list[list[int]]:
                input_ids = base_ids.clone()
                attention_mask = base_mask.clone()
                generated = [[] for _ in range(input_ids.shape[0])]
                swap = None
                if use_swap:
                    swap = OnlineKVSwap(
                        layers,
                        [int(v) for v in condition["depths"]],
                        [int(v) for v in condition["groups"]],
                        head_dim,
                    )
                    swap.register()
                try:
                    for _ in range(steps):
                        if swap is not None:
                            swap.begin(positions, masks)
                        with torch.inference_mode():
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                return_dict=True,
                            )
                        if swap is not None:
                            swap.end()
                        next_token = torch.argmax(
                            output.logits[:, -1, :].float(),
                            dim=-1,
                        )
                        for slot, token in enumerate(
                            next_token.detach().cpu().tolist()
                        ):
                            generated[slot].append(int(token))
                        input_ids = torch.cat(
                            (input_ids, next_token[:, None]), dim=1
                        )
                        attention_mask = torch.cat(
                            (
                                attention_mask,
                                torch.ones(
                                    (input_ids.shape[0], 1),
                                    dtype=attention_mask.dtype,
                                    device=attention_mask.device,
                                ),
                            ),
                            dim=1,
                        )
                        del output
                finally:
                    if swap is not None:
                        swap.close()
                return generated

            clean = generate(False)
            patched = generate(True)
            for pair_slot, target in enumerate(target_batch):
                left = 2 * pair_slot
                right = left + 1
                records.append({
                    "target_index": int(target["target_index"]),
                    "clean": {
                        "target": clean[left],
                        "cross": clean[right],
                        "target_text": tokenizer.decode(
                            clean[left], skip_special_tokens=False
                        ),
                        "cross_text": tokenizer.decode(
                            clean[right], skip_special_tokens=False
                        ),
                    },
                    "patched": {
                        "target": patched[left],
                        "cross": patched[right],
                        "target_text": tokenizer.decode(
                            patched[left], skip_special_tokens=False
                        ),
                        "cross_text": tokenizer.decode(
                            patched[right], skip_special_tokens=False
                        ),
                    },
                    "target_matches_other_clean": (
                        patched[left] == clean[right]
                    ),
                    "cross_matches_other_clean": (
                        patched[right] == clean[left]
                    ),
                    "both_match_other_clean": (
                        patched[left] == clean[right]
                        and patched[right] == clean[left]
                    ),
                })
    count = len(records)
    summary = {
        "pair_count": count,
        "target_matches_other_clean_rate": (
            sum(row["target_matches_other_clean"] for row in records)
            / count if count else 0.0
        ),
        "cross_matches_other_clean_rate": (
            sum(row["cross_matches_other_clean"] for row in records)
            / count if count else 0.0
        ),
        "both_match_other_clean_rate": (
            sum(row["both_match_other_clean"] for row in records)
            / count if count else 0.0
        ),
    }
    return records, summary


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1052 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    case_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
    plan = prereg["model_plans"][model_name]
    started = time.time()
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        first_attention = layers[0].self_attn
        width = projection_width(first_attention.k_proj)
        n_kv_heads = int(plan["n_kv_heads"])
        if width % n_kv_heads:
            raise RuntimeError("KV projection width drift")
        head_dim = width // n_kv_heads
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        clean = run_condition(
            model,
            device,
            layers,
            targets,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=PAIR_BATCH_SIZE[model_name],
        )
        clean_mask, family_coverage = clean_mask_and_coverage(
            targets, cases, clean
        )
        valid_targets = [
            row for row, keep in zip(targets, clean_mask) if keep
        ]
        condition_results = {}
        condition_arrays = {}
        for name in prereg["condition_order"]:
            condition = plan["conditions"][name]
            patched = run_condition(
                model,
                device,
                layers,
                targets,
                cases,
                condition,
                head_dim=head_dim,
                pad_token_id=int(pad_token_id),
                pair_batch_size=PAIR_BATCH_SIZE[model_name],
            )
            condition_arrays[name] = patched
            condition_results[name] = condition_metrics(
                targets, cases, clean, patched, clean_mask
            )

        local = condition_results[
            "selected_frozen_groups_frozen_depths"
        ]
        unselected = condition_results[
            "unselected_frozen_groups_frozen_depths"
        ]
        query = condition_results[
            "query_frozen_groups_frozen_depths"
        ]
        control_rate = max(
            float(unselected["both_counterfactual_top1_rate"]),
            float(query["both_counterfactual_top1_rate"]),
        )
        gates = prereg["gates"]
        baseline_gate = (
            bool(plan["behavior_eligible"])
            and len(valid_targets)
            >= gates["clean_correct_pair_count_min"]
            and len(family_coverage)
            >= gates["clean_family_coverage_min"]
        )
        local_gate = (
            baseline_gate
            and local["both_counterfactual_top1_count"]
            >= gates["local_both_counterfactual_pair_count_min"]
            and local["both_counterfactual_top1_rate"]
            >= gates["local_both_counterfactual_rate_min"]
            and (
                local["both_counterfactual_top1_rate"]
                - control_rate
            )
            >= gates["local_selected_minus_control_rate_min"]
        )
        broad = condition_results[
            "selected_all_groups_all_postsource"
        ]
        broad_gate = (
            baseline_gate
            and broad["both_counterfactual_top1_rate"]
            >= gates["broad_both_counterfactual_rate_min"]
        )

        rollout_condition = None
        rollout_targets = []
        for name in prereg["rollout_condition_priority"]:
            result = condition_results[name]
            if (
                result["both_counterfactual_top1_count"]
                >= gates["rollout_pair_count_min"]
            ):
                rollout_condition = name
                valid_indices = result["valid_target_indices"]
                successes = result["both_counterfactual_mask"]
                target_by_index = {
                    int(row["target_index"]): row for row in targets
                }
                rollout_targets = [
                    target_by_index[index]
                    for index, success in zip(
                        valid_indices, successes
                    )
                    if success
                ]
                break
        rollouts = []
        rollout_summary = {
            "pair_count": 0,
            "target_matches_other_clean_rate": 0.0,
            "cross_matches_other_clean_rate": 0.0,
            "both_match_other_clean_rate": 0.0,
        }
        if rollout_condition is not None:
            rollouts, rollout_summary = rollout_pairs(
                model,
                tokenizer,
                device,
                layers,
                rollout_targets,
                cases,
                plan["conditions"][rollout_condition],
                head_dim=head_dim,
                steps=int(prereg["rollout_steps"]),
                pair_limit=int(prereg["rollout_pair_limit"]),
                pair_batch_size=PAIR_BATCH_SIZE[model_name],
            )
        rollout_gate = (
            baseline_gate
            and rollout_summary["pair_count"]
            >= gates["rollout_pair_count_min"]
            and rollout_summary["both_match_other_clean_rate"]
            >= gates["rollout_both_match_other_clean_rate_min"]
        )
        summary = {
            "schema_version": "phase1052_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "behavior_eligible": bool(plan["behavior_eligible"]),
            "clean_correct_pair_count": len(valid_targets),
            "clean_correct_pair_rate": len(valid_targets) / len(targets),
            "clean_family_coverage": family_coverage,
            "condition_results": {
                name: {
                    key: value for key, value in result.items()
                    if key not in (
                        "both_counterfactual_mask",
                        "valid_target_indices",
                    )
                }
                for name, result in condition_results.items()
            },
            "selected_minus_max_control_rate": (
                local["both_counterfactual_top1_rate"] - control_rate
            ),
            "baseline_gate_passed": baseline_gate,
            "local_bridge_gate_passed": local_gate,
            "broad_graph_cut_gate_passed": broad_gate,
            "rollout_condition": rollout_condition,
            "rollout_summary": rollout_summary,
            "rollout_gate_passed": rollout_gate,
            "rollouts": rollouts,
            "elapsed_seconds": float(time.time() - started),
        }
        out = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_json(out / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "behavior_eligible": plan["behavior_eligible"],
            "clean_correct_pairs": len(valid_targets),
            "family_coverage": len(family_coverage),
            "local": local["both_counterfactual_top1_rate"],
            "unselected": unselected[
                "both_counterfactual_top1_rate"
            ],
            "query": query["both_counterfactual_top1_rate"],
            "broad": broad["both_counterfactual_top1_rate"],
            "local_gate": local_gate,
            "broad_gate": broad_gate,
            "rollout_condition": rollout_condition,
            "rollout": rollout_summary,
            "rollout_gate": rollout_gate,
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=protocol.MODELS
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
