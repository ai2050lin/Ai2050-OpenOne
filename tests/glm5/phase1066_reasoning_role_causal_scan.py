#!/usr/bin/env python3
"""Run Phase1066 role-endpoint K/V swaps on transitive reasoning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

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
import phase1065_multimode_response_atlas_protocol as source
import phase1066_reasoning_role_causal_protocol as protocol


PAIR_BATCH_SIZE = bridge.PAIR_BATCH_SIZE


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


class RoleEndpointKVSwap:
    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        groups: list[int],
        channels: list[str],
        head_dim: int,
    ) -> None:
        self.layers = layers
        self.depths = depths
        self.groups = groups
        self.channels = channels
        self.head_dim = head_dim
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
                raise RuntimeError("K/V swap positions missing")
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
                    patched[
                        even, even_pos, start:end
                    ] = odd_value
                    patched[
                        odd, odd_pos, start:end
                    ] = even_value
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
            raise RuntimeError(f"K/V hook drift: {self.counts}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = None
        self.counts = Counter()


def pair_batch_tensors(
    pair_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    sites: list[str],
    pad_id: int,
    device,
) -> tuple[
    list[dict[str, Any]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    rows = []
    for pair in pair_rows:
        rows.extend([
            cases[int(pair["left_case_index"])],
            cases[int(pair["right_case_index"])],
        ])
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    positions = torch.zeros(
        (len(rows), len(sites)),
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        values = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        )
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
        positions[index] = torch.tensor([
            int(row["role_positions"][site]) for site in sites
        ], dtype=torch.long, device=device)
    return rows, input_ids, attention_mask, lengths, positions


def candidate_scores(
    logits: torch.Tensor,
    row: dict[str, Any],
) -> dict[str, float]:
    result = {}
    for class_name in ("b0", "b1"):
        token_ids = torch.tensor(
            row["candidate_first_token_ids"][class_name],
            dtype=torch.long,
            device=logits.device,
        )
        result[class_name] = float(logits[token_ids].max().item())
    return result


def forward_last_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = output.logits
    last = (lengths - 1).to(logits.device)
    batch = torch.arange(logits.shape[0], device=logits.device)
    selected = logits[batch, last, :].float()
    del output, logits
    return selected


def clean_replay(
    model,
    device,
    pairs: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    pad_id: int,
    batch_size: int,
) -> tuple[dict[int, dict[str, float]], float]:
    scores = {}
    correct = total = 0
    with torch.inference_mode():
        for batch_pairs in chunks(pairs, batch_size):
            (
                rows,
                input_ids,
                attention_mask,
                lengths,
                _,
            ) = pair_batch_tensors(
                batch_pairs,
                cases,
                ["answer_boundary"],
                pad_id,
                device,
            )
            logits = forward_last_logits(
                model, input_ids, attention_mask, lengths
            )
            for row, values in zip(rows, logits):
                row_scores = candidate_scores(values, row)
                index = int(row["semantic_case_index"])
                scores[index] = row_scores
                expected = str(row["expected_class"])
                other = "b1" if expected == "b0" else "b0"
                correct += int(
                    row_scores[expected] > row_scores[other]
                )
                total += 1
            del logits, input_ids, attention_mask, lengths
    return scores, correct / total if total else 0.0


def summarize_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_ids = sorted({
        int(row["pair_index"]) for row in records
    })
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_pair[int(row["pair_index"])].append(row)
    both = sum(
        len(values) == 2
        and all(bool(row["donor_class_flip"]) for row in values)
        for values in by_pair.values()
    )
    individual = sum(
        bool(row["donor_class_flip"]) for row in records
    )
    own_after = sum(
        bool(row["own_class_still_wins"]) for row in records
    )
    mean_delta = (
        sum(float(row["delta_toward_donor_margin"]) for row in records)
        / len(records)
        if records else 0.0
    )
    grouped = {}
    for field in ("split", "lexical_branch"):
        values = {}
        keys = sorted({str(row[field]) for row in records})
        for key in keys:
            subset = [
                row for row in records if str(row[field]) == key
            ]
            subset_pairs = {
                int(row["pair_index"]) for row in subset
            }
            subset_by_pair = defaultdict(list)
            for row in subset:
                subset_by_pair[int(row["pair_index"])].append(row)
            values[key] = {
                "direction_count": len(subset),
                "individual_flip_rate": (
                    sum(bool(row["donor_class_flip"]) for row in subset)
                    / len(subset)
                    if subset else 0.0
                ),
                "pair_count": len(subset_pairs),
                "bidirectional_flip_rate": (
                    sum(
                        len(rows) == 2
                        and all(
                            bool(row["donor_class_flip"])
                            for row in rows
                        )
                        for rows in subset_by_pair.values()
                    ) / len(subset_pairs)
                    if subset_pairs else 0.0
                ),
            }
        grouped[f"by_{field}"] = values
    return {
        "pair_count": len(pair_ids),
        "direction_count": len(records),
        "individual_donor_class_flip_count": individual,
        "individual_donor_class_flip_rate": (
            individual / len(records) if records else 0.0
        ),
        "bidirectional_donor_class_flip_count": both,
        "bidirectional_donor_class_flip_rate": (
            both / len(pair_ids) if pair_ids else 0.0
        ),
        "own_class_still_wins_count": own_after,
        "own_class_still_wins_rate": (
            own_after / len(records) if records else 0.0
        ),
        "mean_delta_toward_donor_margin": mean_delta,
        **grouped,
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    source_prereg = source.read_json(
        source.OUT_ROOT / "protocol" / "preregistration.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1066 protocol audit failed")
    if (
        prereg["source_phase1065_digest"]
        != source_prereg["protocol_digest"]
    ):
        raise RuntimeError("Phase1065 source digest drift")
    case_rows = source.read_jsonl(
        source.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row
        for row in case_rows
    }
    pairs = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"pairs.{model_name}.jsonl"
    )
    plan = prereg["model_plans"][model_name]
    started = time.time()
    model = patcher = None
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
        if len(layers) != int(plan["n_layers"]):
            raise RuntimeError("layer count drift")
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
        clean_scores, clean_parity = clean_replay(
            model,
            device,
            pairs,
            cases,
            int(pad_id),
            PAIR_BATCH_SIZE[model_name],
        )
        if clean_parity < prereg["gates"]["clean_candidate_replay_min"]:
            raise RuntimeError(
                f"clean candidate replay failed: {clean_parity}"
            )

        condition_results = {}
        all_records = []
        with torch.inference_mode():
            for condition in plan["conditions"]:
                condition_name = str(condition["condition"])
                patcher = RoleEndpointKVSwap(
                    layers,
                    [int(value) for value in condition["depths"]],
                    [int(value) for value in condition["groups"]],
                    [str(value) for value in condition["channels"]],
                    head_dim,
                )
                patcher.register()
                condition_records = []
                for batch_pairs in chunks(
                    pairs, PAIR_BATCH_SIZE[model_name]
                ):
                    (
                        rows,
                        input_ids,
                        attention_mask,
                        lengths,
                        positions,
                    ) = pair_batch_tensors(
                        batch_pairs,
                        cases,
                        [str(value) for value in condition["sites"]],
                        int(pad_id),
                        device,
                    )
                    patcher.begin(positions)
                    logits = forward_last_logits(
                        model,
                        input_ids,
                        attention_mask,
                        lengths,
                    )
                    patcher.validate()
                    for pair_offset, pair in enumerate(batch_pairs):
                        for direction in (0, 1):
                            row_index = pair_offset * 2 + direction
                            row = rows[row_index]
                            values = logits[row_index]
                            scores = candidate_scores(values, row)
                            own = str(row["expected_class"])
                            donor = "b1" if own == "b0" else "b0"
                            patched_margin = (
                                scores[donor] - scores[own]
                            )
                            clean = clean_scores[
                                int(row["semantic_case_index"])
                            ]
                            clean_margin = (
                                clean[donor] - clean[own]
                            )
                            record = {
                                "schema_version": (
                                    "phase1066_causal_direction.v1"
                                ),
                                "phase": protocol.PHASE,
                                "model": model_name,
                                "condition": condition_name,
                                "condition_kind": condition["kind"],
                                "pair_index": int(pair["pair_index"]),
                                "unit_id": pair["unit_id"],
                                "split": pair["split"],
                                "lexical_branch": int(
                                    pair["lexical_branch"]
                                ),
                                "direction": (
                                    "b0_to_b1"
                                    if direction == 0
                                    else "b1_to_b0"
                                ),
                                "target_case_index": int(
                                    row["semantic_case_index"]
                                ),
                                "own_class": own,
                                "donor_class": donor,
                                "clean_donor_minus_own_margin": (
                                    clean_margin
                                ),
                                "patched_donor_minus_own_margin": (
                                    patched_margin
                                ),
                                "delta_toward_donor_margin": (
                                    patched_margin - clean_margin
                                ),
                                "donor_class_flip": patched_margin > 0.0,
                                "own_class_still_wins": (
                                    patched_margin < 0.0
                                ),
                            }
                            condition_records.append(record)
                            all_records.append(record)
                    del logits, input_ids, attention_mask, lengths, positions
                patcher.close()
                patcher = None
                condition_results[condition_name] = {
                    "kind": condition["kind"],
                    "sites": condition["sites"],
                    "channels": condition["channels"],
                    "depth_count": len(condition["depths"]),
                    "group_count": len(condition["groups"]),
                    **summarize_records(condition_records),
                }
                print(json.dumps({
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "condition": condition_name,
                    "bidirectional_flip_rate": condition_results[
                        condition_name
                    ]["bidirectional_donor_class_flip_rate"],
                    "individual_flip_rate": condition_results[
                        condition_name
                    ]["individual_donor_class_flip_rate"],
                    "margin_delta": condition_results[
                        condition_name
                    ]["mean_delta_toward_donor_margin"],
                }), flush=True)

        source_conditions = {
            name: row
            for name, row in condition_results.items()
            if row["kind"] in {"source", "depth", "channel", "group"}
        }
        controls = {
            name: row
            for name, row in condition_results.items()
            if row["kind"] == "role_control"
        }
        best_source_name, best_source = max(
            source_conditions.items(),
            key=lambda item: item[1][
                "bidirectional_donor_class_flip_rate"
            ],
        )
        maximum_control = max(
            (
                row["bidirectional_donor_class_flip_rate"]
                for row in controls.values()
            ),
            default=0.0,
        )
        source_rate = best_source[
            "bidirectional_donor_class_flip_rate"
        ]
        causal_gate = bool(
            source_rate
            >= prereg["gates"]["bidirectional_flip_rate_min"]
            and maximum_control
            <= prereg["gates"]["maximum_role_control_flip_rate"]
            and source_rate - maximum_control
            >= prereg["gates"]["source_minus_control_min"]
        )
        summary = {
            "schema_version": "phase1066_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "pair_count": len(pairs),
            "clean_candidate_replay_rate": clean_parity,
            "model_info": {
                "n_layers": len(layers),
                "n_kv_heads": n_kv_heads,
                "head_dim": head_dim,
            },
            "condition_results": condition_results,
            "best_source_condition": best_source_name,
            "best_source_bidirectional_flip_rate": source_rate,
            "maximum_role_control_bidirectional_flip_rate": (
                maximum_control
            ),
            "source_minus_control": source_rate - maximum_control,
            "causal_gate_passed": causal_gate,
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
            "best_source": best_source_name,
            "best_source_rate": source_rate,
            "max_control": maximum_control,
            "causal_gate": causal_gate,
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
    finally:
        if patcher is not None:
            patcher.close()
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
