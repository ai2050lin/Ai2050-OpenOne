#!/usr/bin/env python3
"""Run Phase1067 paired-mean and K/V-group coalition interventions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1065_multimode_response_atlas_protocol as atlas
import phase1066_reasoning_role_causal_scan as causal
import phase1067_reasoning_necessity_coalition_protocol as protocol


PAIR_BATCH_SIZE = bridge.PAIR_BATCH_SIZE


class RoleEndpointKVIntervention:
    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        groups: list[int],
        channels: list[str],
        head_dim: int,
        mode: str,
    ) -> None:
        if mode not in {"swap", "mean"}:
            raise ValueError(mode)
        self.layers = layers
        self.depths = depths
        self.groups = groups
        self.channels = channels
        self.head_dim = head_dim
        self.mode = mode
        self.positions: torch.Tensor | None = None
        self.counts: Counter = Counter()
        self.handles = []

    def register(self) -> None:
        for depth in self.depths:
            attention = self.layers[depth - 1].self_attn
            if "k" in self.channels:
                self.handles.append(
                    attention.k_proj.register_forward_hook(
                        self._hook("k", depth)
                    )
                )
            if "v" in self.channels:
                self.handles.append(
                    attention.v_proj.register_forward_hook(
                        self._hook("v", depth)
                    )
                )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.counts = Counter()

    def _hook(self, channel: str, depth: int):
        def hook(_module, _inputs, output):
            if self.positions is None:
                raise RuntimeError("intervention positions missing")
            hidden = route_tools.output_tensor(output)
            if hidden.shape[0] % 2:
                raise RuntimeError("paired batch is not even")
            positions = self.positions.to(hidden.device)
            patched = hidden.clone()
            even = torch.arange(
                0, hidden.shape[0], 2, device=hidden.device
            )
            odd = even + 1
            for site in range(positions.shape[1]):
                even_pos = positions[even, site]
                odd_pos = positions[odd, site]
                for group in self.groups:
                    start = group * self.head_dim
                    end = start + self.head_dim
                    even_value = hidden[
                        even, even_pos, start:end
                    ].clone()
                    odd_value = hidden[
                        odd, odd_pos, start:end
                    ].clone()
                    if self.mode == "swap":
                        even_new = odd_value
                        odd_new = even_value
                    else:
                        midpoint = 0.5 * (even_value + odd_value)
                        even_new = midpoint
                        odd_new = midpoint
                    patched[
                        even, even_pos, start:end
                    ] = even_new
                    patched[
                        odd, odd_pos, start:end
                    ] = odd_new
            self.counts[(channel, depth)] += 1
            return route_tools.replace_output(output, patched)

        return hook

    def validate(self) -> None:
        expected = {
            (channel, depth)
            for channel in self.channels
            for depth in self.depths
        }
        if set(self.counts) != expected or any(
            count != 1 for count in self.counts.values()
        ):
            raise RuntimeError(f"intervention hook drift: {self.counts}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = None
        self.counts = Counter()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    pair_count = len({
        int(row["pair_index"]) for row in records
    })
    accuracy = (
        sum(bool(row["own_class_correct"]) for row in records)
        / len(records)
        if records else 0.0
    )
    flips = (
        sum(bool(row["opposite_class_flip"]) for row in records)
        / len(records)
        if records else 0.0
    )
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_pair[int(row["pair_index"])].append(row)
    bidirectional = (
        sum(
            len(values) == 2
            and all(bool(row["opposite_class_flip"]) for row in values)
            for values in by_pair.values()
        ) / pair_count
        if pair_count else 0.0
    )
    mean_margin_change = (
        sum(float(row["own_margin_change"]) for row in records)
        / len(records)
        if records else 0.0
    )
    by_split = {}
    for split in sorted({str(row["split"]) for row in records}):
        values = [row for row in records if row["split"] == split]
        by_split[split] = {
            "direction_count": len(values),
            "own_class_accuracy": (
                sum(bool(row["own_class_correct"]) for row in values)
                / len(values)
                if values else 0.0
            ),
            "mean_own_margin_change": (
                sum(float(row["own_margin_change"]) for row in values)
                / len(values)
                if values else 0.0
            ),
        }
    return {
        "pair_count": pair_count,
        "direction_count": len(records),
        "own_class_accuracy": accuracy,
        "own_class_accuracy_drop": 1.0 - accuracy,
        "individual_opposite_class_flip_rate": flips,
        "bidirectional_opposite_class_flip_rate": bidirectional,
        "mean_own_margin_change": mean_margin_change,
        "by_split": by_split,
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1067 protocol audit failed")
    cases = {
        int(row["semantic_case_index"]): row
        for row in atlas.read_jsonl(
            atlas.OUT_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
    }
    pair_sets = {
        "semantic": protocol.read_jsonl(
            protocol.OUT_ROOT
            / "protocol"
            / f"semantic_pairs.{model_name}.jsonl"
        ),
        "surface": protocol.read_jsonl(
            protocol.OUT_ROOT
            / "protocol"
            / f"surface_pairs.{model_name}.jsonl"
        ),
    }
    plan = prereg["model_plans"][model_name]
    started = time.time()
    model = intervention = None
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
        width = bridge.projection_width(
            layers[0].self_attn.k_proj
        )
        n_kv_heads = int(plan["n_kv_heads"])
        if width % n_kv_heads:
            raise RuntimeError("K/V width is not head aligned")
        head_dim = width // n_kv_heads
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        clean_scores = {}
        clean_rates = {}
        for pair_set, pairs in pair_sets.items():
            scores, replay = causal.clean_replay(
                model,
                device,
                pairs,
                cases,
                int(pad_id),
                PAIR_BATCH_SIZE[model_name],
            )
            clean_scores.update(scores)
            clean_rates[pair_set] = replay
            if replay < prereg["gates"]["clean_candidate_replay_min"]:
                raise RuntimeError(
                    f"clean replay failed for {pair_set}: {replay}"
                )

        condition_results = {}
        all_records = []
        with torch.inference_mode():
            for condition in plan["conditions"]:
                pairs = pair_sets[str(condition["pair_set"])]
                intervention = RoleEndpointKVIntervention(
                    layers,
                    [int(value) for value in condition["depths"]],
                    [int(value) for value in condition["groups"]],
                    [str(value) for value in condition["channels"]],
                    head_dim,
                    str(condition["mode"]),
                )
                intervention.register()
                records = []
                for batch_pairs in causal.chunks(
                    pairs, PAIR_BATCH_SIZE[model_name]
                ):
                    (
                        rows,
                        input_ids,
                        attention_mask,
                        lengths,
                        positions,
                    ) = causal.pair_batch_tensors(
                        batch_pairs,
                        cases,
                        [str(value) for value in condition["sites"]],
                        int(pad_id),
                        device,
                    )
                    intervention.begin(positions)
                    logits = causal.forward_last_logits(
                        model,
                        input_ids,
                        attention_mask,
                        lengths,
                    )
                    intervention.validate()
                    for pair_offset, pair in enumerate(batch_pairs):
                        for direction in (0, 1):
                            row_index = pair_offset * 2 + direction
                            row = rows[row_index]
                            scores = causal.candidate_scores(
                                logits[row_index], row
                            )
                            own = str(row["expected_class"])
                            other = "b1" if own == "b0" else "b0"
                            own_margin = scores[own] - scores[other]
                            clean = clean_scores[
                                int(row["semantic_case_index"])
                            ]
                            clean_margin = clean[own] - clean[other]
                            record = {
                                "schema_version": (
                                    "phase1067_intervention_direction.v1"
                                ),
                                "phase": protocol.PHASE,
                                "model": model_name,
                                "condition": condition["condition"],
                                "condition_kind": condition["kind"],
                                "pair_set": condition["pair_set"],
                                "mode": condition["mode"],
                                "pair_index": int(pair["pair_index"]),
                                "unit_id": pair["unit_id"],
                                "split": pair["split"],
                                "direction": direction,
                                "target_case_index": int(
                                    row["semantic_case_index"]
                                ),
                                "own_class": own,
                                "other_class": other,
                                "clean_own_minus_other_margin": (
                                    clean_margin
                                ),
                                "intervened_own_minus_other_margin": (
                                    own_margin
                                ),
                                "own_margin_change": (
                                    own_margin - clean_margin
                                ),
                                "own_class_correct": own_margin > 0.0,
                                "opposite_class_flip": own_margin < 0.0,
                            }
                            records.append(record)
                            all_records.append(record)
                    del logits, input_ids, attention_mask, lengths, positions
                intervention.close()
                intervention = None
                condition_results[str(condition["condition"])] = {
                    "kind": condition["kind"],
                    "pair_set": condition["pair_set"],
                    "mode": condition["mode"],
                    "sites": condition["sites"],
                    "channels": condition["channels"],
                    "groups": condition["groups"],
                    "depth_count": len(condition["depths"]),
                    **summarize(records),
                }
                result = condition_results[str(condition["condition"])]
                print(json.dumps({
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "condition": condition["condition"],
                    "accuracy_drop": result["own_class_accuracy_drop"],
                    "bidirectional_flip": result[
                        "bidirectional_opposite_class_flip_rate"
                    ],
                    "margin_change": result[
                        "mean_own_margin_change"
                    ],
                }), flush=True)

        neutralization = condition_results["semantic_mean_kv"]
        control = condition_results["surface_preserving_swap_kv"]
        necessity_stress_gate = bool(
            neutralization["own_class_accuracy_drop"]
            >= prereg["gates"]["neutralization_accuracy_drop_min"]
            and control["own_class_accuracy_drop"]
            <= prereg["gates"][
                "semantic_preserving_control_drop_max"
            ]
        )
        summary = {
            "schema_version": "phase1067_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "clean_candidate_replay_rates": clean_rates,
            "condition_results": condition_results,
            "neutralization_accuracy_drop": neutralization[
                "own_class_accuracy_drop"
            ],
            "semantic_preserving_control_accuracy_drop": control[
                "own_class_accuracy_drop"
            ],
            "necessity_stress_gate_passed": necessity_stress_gate,
            "elapsed_seconds": time.time() - started,
            "interpretation_limits": prereg[
                "interpretation_limits"
            ],
        }
        root = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_jsonl(
            root / "condition_records.jsonl", all_records
        )
        protocol.write_json(root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "neutralization_drop": summary[
                "neutralization_accuracy_drop"
            ],
            "surface_control_drop": summary[
                "semantic_preserving_control_accuracy_drop"
            ],
            "necessity_stress_gate": necessity_stress_gate,
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
    finally:
        if intervention is not None:
            intervention.close()
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
