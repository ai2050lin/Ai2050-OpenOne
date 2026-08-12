#!/usr/bin/env python3
"""Run preregistered signed query-end residual transport for Phase1103."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1103_natural_relation_route_protocol as protocol


CAPTURE_BATCH_SIZE = {"qwen3": 8, "glm4": 4, "deepseek7b": 4}
PATCH_BATCH_SIZE = {"qwen3": 4, "glm4": 4, "deepseek7b": 4}
EPSILON = 1e-8


def sampled_depths(layer_count: int) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for fraction in protocol.CAUSAL_DEPTH_FRACTIONS:
        depth = min(
            range(1, layer_count + 1),
            key=lambda value: (
                abs(value / layer_count - fraction), value,
            ),
        )
        if depth in seen:
            continue
        seen.add(depth)
        result.append({
            "depth": depth,
            "relative_depth": depth / layer_count,
            "requested_fraction": fraction,
        })
    return result


def pad_rows(
    rows: list[dict[str, Any]], pad_id: int, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        values = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        )
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
    return input_ids, attention_mask, lengths


class QueryResidualCapture:
    def __init__(self, layers, depths: list[int]):
        self.layers = layers
        self.depths = depths
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []

    def _hook(self, depth: int):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(value, torch.Tensor):
                raise RuntimeError("query capture was not initialized")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)
            self.values[depth] = (
                value[batch, positions, :].detach().float().cpu()
            )
            return output

        return hook

    def register(self) -> None:
        for depth in self.depths:
            self.handles.append(
                self.layers[depth - 1].register_forward_hook(
                    self._hook(depth)
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}

    def validate(self) -> None:
        if set(self.values) != set(self.depths):
            raise RuntimeError(
                f"capture drift: got {sorted(self.values)}, "
                f"expected {self.depths}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


class ResidualPatch:
    def __init__(
        self, layer, positions: torch.Tensor, deltas: torch.Tensor,
        alpha: float,
    ):
        self.layer = layer
        self.positions = positions
        self.deltas = deltas
        self.alpha = alpha
        self.calls = 0
        self.handle = None

    def _hook(self, module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("patch layer did not return a tensor")
        positions = self.positions.to(value.device)
        deltas = self.deltas.to(value.device, dtype=value.dtype)
        batch = torch.arange(value.shape[0], device=value.device)
        patched = value.clone()
        patched[batch, positions, :] = (
            patched[batch, positions, :] + self.alpha * deltas
        )
        self.calls += 1
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def stable_seed(*parts: object) -> int:
    value = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def equal_norm_random(vector: torch.Tensor, *seed_parts: object) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(*seed_parts))
    random = torch.empty_like(vector, device="cpu")
    random.bernoulli_(0.5, generator=generator)
    random.mul_(2.0).sub_(1.0)
    norm = torch.linalg.vector_norm(vector.float())
    random_norm = torch.linalg.vector_norm(random.float())
    if norm <= EPSILON or random_norm <= EPSILON:
        return torch.zeros_like(vector, device="cpu")
    return random * (norm / random_norm)


def norm_match(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    vector_norm = torch.linalg.vector_norm(vector.float())
    reference_norm = torch.linalg.vector_norm(reference.float())
    if vector_norm <= EPSILON or reference_norm <= EPSILON:
        return torch.zeros_like(vector, device="cpu")
    return vector * (reference_norm / vector_norm)


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["relation_pair"]), str(row["surface"]),
        str(row["split"]), int(row["template"]),
        int(row["item_index"]), str(row["congruence"]),
        str(row["route_type"]), int(row["target_relation"]),
        int(row["relation_order"]), int(row["orientation"]),
    )


def causal_superunit_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["surface"]), str(row["split"]),
        int(row["template"]), int(row["item_index"]),
        int(row["relation_order"]), int(row["orientation"]),
    )


def is_causal_item(row: dict[str, Any]) -> bool:
    allowed_items = (
        protocol.CAUSAL_DISCOVERY_ITEMS
        if row["split"] == "qualification"
        else protocol.CAUSAL_CONFIRMATION_ITEMS
    )
    return (
        int(row["item_index"]) in allowed_items
        and int(row["relation_order"])
        in protocol.CAUSAL_RELATION_ORDERS
    )


def candidate_scores(
    logits: torch.Tensor, slot: int, length: int, row: dict[str, Any]
) -> dict[str, float]:
    return {
        answer_class: float(
            logits[slot, length - 1, int(ids[0])].float().item()
        )
        for answer_class, ids in row["candidate_first_token_ids"].items()
    }


def capture_baselines(
    model,
    capture: QueryResidualCapture,
    rows: list[dict[str, Any]],
    batch_size: int,
    pad_id: int,
    device,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, dict[str, float]]]:
    vectors: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    scores: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            input_ids, attention_mask, lengths = pad_rows(
                batch_rows, pad_id, device
            )
            positions = torch.tensor([
                int(row["role_positions"]["query_end"])
                for row in batch_rows
            ], dtype=torch.long, device=device)
            capture.begin(positions)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            capture.validate()
            logits = output.logits
            for slot, row in enumerate(batch_rows):
                record_id = str(row["record_id"])
                for depth, values in capture.values.items():
                    vectors[record_id][depth] = values[slot].clone()
                scores[record_id] = candidate_scores(
                    logits, slot, int(lengths[slot].item()), row
                )
            del output, logits, input_ids, attention_mask, lengths, positions
    return dict(vectors), scores


def make_entry(
    pair: str,
    surface: str,
    split: str,
    template: int,
    item_index: int,
    orientation: int,
    depth: int,
    target_row: dict[str, Any],
    donor_row: dict[str, Any],
    source_pair: str,
    source_route: str,
    target_route: str,
    patch_kind: str,
    delta: torch.Tensor,
    baseline_scores: dict[str, dict[str, float]],
    congruent: bool = False,
) -> dict[str, Any]:
    return {
        "pair": pair,
        "surface": surface,
        "split": split,
        "template": template,
        "item_index": item_index,
        "orientation": orientation,
        "depth": depth,
        "target_row": target_row,
        "donor_row": donor_row,
        "source_pair": source_pair,
        "source_route": source_route,
        "target_route": target_route,
        "patch_kind": patch_kind,
        "delta": delta,
        "baseline_scores": baseline_scores,
        "congruent": congruent,
    }


def build_patch_entries(
    eligible_pairs: list[str],
    wrong_pairs: dict[str, str],
    index: dict[tuple[Any, ...], dict[str, Any]],
    vectors: dict[str, dict[int, torch.Tensor]],
    baseline_scores: dict[str, dict[str, float]],
    superunit: tuple[Any, ...],
    depth: int,
    model_name: str,
) -> list[dict[str, Any]]:
    surface, split, template, item_index, order, orientation = superunit

    def get(
        pair: str, congruence: str, route: str, target: int
    ) -> dict[str, Any]:
        return index[(
            pair, surface, split, template, item_index, congruence,
            route, target, order, orientation,
        )]

    entries = []
    for pair in eligible_pairs:
        wrong_pair = wrong_pairs[pair]
        deltas = {}
        wrong_deltas = {}
        for route in protocol.ROUTE_TYPES:
            q0 = get(pair, "conflict", route, 0)
            q1 = get(pair, "conflict", route, 1)
            deltas[route] = (
                vectors[q1["record_id"]][depth]
                - vectors[q0["record_id"]][depth]
            )
            w0 = get(wrong_pair, "conflict", route, 0)
            w1 = get(wrong_pair, "conflict", route, 1)
            wrong_deltas[route] = (
                vectors[w1["record_id"]][depth]
                - vectors[w0["record_id"]][depth]
            )
        directions = (
            ("exact", "paraphrase"),
            ("paraphrase", "exact"),
        )
        for source_route, target_route in directions:
            target = get(pair, "conflict", target_route, 0)
            donor = get(pair, "conflict", target_route, 1)
            semantic_delta = deltas[source_route]
            ordinal_delta = norm_match(
                deltas["ordinal"], semantic_delta
            )
            wrong_delta = norm_match(
                wrong_deltas[source_route], semantic_delta
            )
            random_delta = equal_norm_random(
                semantic_delta, protocol.PHASE, model_name, pair,
                surface, split, template, item_index, orientation,
                depth, source_route, target_route,
            )
            common = dict(
                pair=pair,
                surface=surface,
                split=split,
                template=template,
                item_index=item_index,
                orientation=orientation,
                depth=depth,
                target_row=target,
                donor_row=donor,
                source_route=source_route,
                target_route=target_route,
                baseline_scores=baseline_scores,
            )
            entries.extend((
                make_entry(
                    source_pair=pair, patch_kind="same_pair",
                    delta=semantic_delta, **common,
                ),
                make_entry(
                    source_pair=pair, patch_kind="ordinal_control",
                    delta=ordinal_delta, **common,
                ),
                make_entry(
                    source_pair=wrong_pair, patch_kind="wrong_pair_control",
                    delta=wrong_delta, **common,
                ),
                make_entry(
                    source_pair="deterministic_rademacher",
                    patch_kind="equal_norm_random_control",
                    delta=random_delta, **common,
                ),
            ))
            congruent_target = get(
                pair, "congruent", target_route, 0
            )
            congruent_donor = get(
                pair, "congruent", target_route, 1
            )
            entries.append(make_entry(
                pair=pair,
                surface=surface,
                split=split,
                template=template,
                item_index=item_index,
                orientation=orientation,
                depth=depth,
                target_row=congruent_target,
                donor_row=congruent_donor,
                source_pair=pair,
                source_route=source_route,
                target_route=target_route,
                patch_kind="congruent_collateral",
                delta=semantic_delta,
                baseline_scores=baseline_scores,
                congruent=True,
            ))
    return entries


def run_patch_batch(
    model,
    layer,
    entries: list[dict[str, Any]],
    pad_id: int,
    device,
) -> list[dict[str, Any]]:
    rows = [entry["target_row"] for entry in entries]
    input_ids, attention_mask, lengths = pad_rows(rows, pad_id, device)
    positions = torch.tensor([
        int(row["role_positions"]["query_end"]) for row in rows
    ], dtype=torch.long, device=device)
    deltas = torch.stack([entry["delta"] for entry in entries])
    with torch.inference_mode():
        with ResidualPatch(
            layer, positions, deltas, protocol.PATCH_ALPHA
        ) as patch:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        if patch.calls != 1:
            raise RuntimeError(f"patch hook called {patch.calls} times")
        logits = output.logits
        records = []
        for slot, entry in enumerate(entries):
            target = entry["target_row"]
            donor = entry["donor_row"]
            patched_scores = candidate_scores(
                logits, slot, int(lengths[slot].item()), target
            )
            target_base = entry["baseline_scores"][target["record_id"]]
            donor_base = entry["baseline_scores"][donor["record_id"]]
            finite = all(math.isfinite(value) for value in (
                *patched_scores.values(), *target_base.values(),
                *donor_base.values(),
            )) and bool(torch.isfinite(entry["delta"]).all().item())
            if entry["congruent"]:
                expected_class = str(target["expected_class"])
                other_class = "e1" if expected_class == "e0" else "e0"
                base_margin = (
                    target_base[other_class] - target_base[expected_class]
                )
                patch_margin = (
                    patched_scores[other_class]
                    - patched_scores[expected_class]
                )
                behavior_valid = (
                    finite and base_margin < 0.0
                    and str(donor["expected_class"]) == expected_class
                )
                recovery = None
                flip = finite and patch_margin > 0.0
                denominator = None
            else:
                base_class = str(target["expected_class"])
                desired_class = str(donor["expected_class"])
                base_margin = (
                    target_base[desired_class] - target_base[base_class]
                )
                donor_margin = (
                    donor_base[desired_class] - donor_base[base_class]
                )
                patch_margin = (
                    patched_scores[desired_class]
                    - patched_scores[base_class]
                )
                denominator = donor_margin - base_margin
                behavior_valid = (
                    finite and base_class != desired_class
                    and base_margin < 0.0 and donor_margin > 0.0
                    and denominator > EPSILON
                )
                recovery = (
                    (patch_margin - base_margin) / denominator
                    if behavior_valid else None
                )
                flip = finite and patch_margin > 0.0
            delta_norm = float(
                torch.linalg.vector_norm(entry["delta"].float()).item()
            )
            records.append({
                "schema_version": "phase1103_causal_patch_record.v1",
                "phase": protocol.PHASE,
                "model": target["model"],
                "relation_pair": entry["pair"],
                "surface": entry["surface"],
                "split": entry["split"],
                "template": entry["template"],
                "item_index": entry["item_index"],
                "orientation": entry["orientation"],
                "depth": entry["depth"],
                "source_pair": entry["source_pair"],
                "source_route": entry["source_route"],
                "target_route": entry["target_route"],
                "patch_kind": entry["patch_kind"],
                "target_record_id": target["record_id"],
                "donor_record_id": donor["record_id"],
                "delta_norm": delta_norm if math.isfinite(delta_norm) else None,
                "base_margin": base_margin if math.isfinite(base_margin) else None,
                "patched_margin": patch_margin if math.isfinite(patch_margin) else None,
                "denominator": (
                    denominator
                    if denominator is not None and math.isfinite(denominator)
                    else None
                ),
                "recovery": (
                    recovery
                    if recovery is not None and math.isfinite(recovery)
                    else None
                ),
                "finite": finite,
                "behavior_valid": behavior_valid,
                "flip": flip,
                "congruent": entry["congruent"],
            })
        del output, logits, input_ids, attention_mask, lengths, positions, deltas
    return records


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1103 protocol audit failed")
    eligible_pairs = [
        pair for pair in authorization["causally_eligible_pairs"]
        if model_name in authorization["causal_models_by_pair"][pair]
    ]
    output_root = protocol.OUT_ROOT / "causal" / model_name
    if not authorization["causal_scan_authorized"] or not eligible_pairs:
        summary = {
            "schema_version": "phase1103_causal_scan_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "authorization_digest": authorization["authorization_digest"],
            "skipped": True,
            "reason": "no behavior-authorized pair with a matched wrong-pair control",
            "eligible_pairs": eligible_pairs,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return

    wrong_pairs = {
        pair: authorization["wrong_pair_controls"][model_name][pair]
        for pair in eligible_pairs
    }
    source_pairs = sorted(set(eligible_pairs) | set(wrong_pairs.values()))
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    index = {row_key(row): row for row in rows if is_causal_item(row)}
    superunits = sorted({
        causal_superunit_key(row)
        for row in rows
        if is_causal_item(row)
    })
    started = time.time()
    model = None
    capture = None
    all_records = []
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        depth_rows = sampled_depths(len(layers))
        depths = [int(row["depth"]) for row in depth_rows]
        capture = QueryResidualCapture(layers, depths)
        capture.register()
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        def get(
            pair: str,
            superunit: tuple[Any, ...],
            congruence: str,
            route: str,
            target: int,
        ) -> dict[str, Any]:
            surface, split, template, item_index, order, orientation = superunit
            return index[(
                pair, surface, split, template, item_index, congruence,
                route, target, order, orientation,
            )]

        for superunit_number, superunit in enumerate(superunits):
            capture_rows = []
            seen = set()
            for pair in source_pairs:
                for route in protocol.ROUTE_TYPES:
                    for target in protocol.TARGET_RELATIONS:
                        row = get(
                            pair, superunit, "conflict", route, target
                        )
                        if row["record_id"] not in seen:
                            capture_rows.append(row)
                            seen.add(row["record_id"])
            for pair in eligible_pairs:
                for route in ("exact", "paraphrase"):
                    for target in protocol.TARGET_RELATIONS:
                        row = get(
                            pair, superunit, "congruent", route, target
                        )
                        if row["record_id"] not in seen:
                            capture_rows.append(row)
                            seen.add(row["record_id"])
            vectors, baseline_scores = capture_baselines(
                model, capture, capture_rows,
                CAPTURE_BATCH_SIZE[model_name], int(pad_id), device,
            )
            for depth in depths:
                entries = build_patch_entries(
                    eligible_pairs, wrong_pairs, index, vectors,
                    baseline_scores, superunit, depth, model_name,
                )
                batch_size = PATCH_BATCH_SIZE[model_name]
                for start in range(0, len(entries), batch_size):
                    all_records.extend(run_patch_batch(
                        model, layers[depth - 1],
                        entries[start:start + batch_size], int(pad_id), device,
                    ))
            completed = superunit_number + 1
            print(json.dumps({
                "phase": protocol.PHASE,
                "model": model_name,
                "causal_superunits_complete": completed,
                "causal_superunits_total": len(superunits),
                "patch_records": len(all_records),
            }), flush=True)
            del vectors, baseline_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        capture.close()
        capture = None
        elapsed = time.time() - started
        finite_count = sum(record["finite"] for record in all_records)
        behavior_valid_count = sum(
            record["behavior_valid"] for record in all_records
            if not record["congruent"]
        )
        noncongruent_count = sum(
            not record["congruent"] for record in all_records
        )
        summary = {
            "schema_version": "phase1103_causal_scan_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "authorization_digest": authorization["authorization_digest"],
            "skipped": False,
            "eligible_pairs": eligible_pairs,
            "wrong_pair_controls": wrong_pairs,
            "precision": precision,
            "placement": placement,
            "layer_count": len(layers),
            "sampled_depths": depth_rows,
            "superunit_count": len(superunits),
            "patch_record_count": len(all_records),
            "finite_fraction": finite_count / max(len(all_records), 1),
            "behavior_valid_fraction": (
                behavior_valid_count / max(noncongruent_count, 1)
            ),
            "elapsed_seconds": elapsed,
            "patch_alpha": protocol.PATCH_ALPHA,
            "component": "residual_stream_after_sampled_layer",
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_jsonl(output_root / "patch_detail.jsonl", all_records)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "eligible_pairs": eligible_pairs,
            "patch_record_count": len(all_records),
            "finite_fraction": summary["finite_fraction"],
            "behavior_valid_fraction": summary["behavior_valid_fraction"],
            "elapsed_seconds": elapsed,
            "summary_digest": summary["summary_digest"],
        }, ensure_ascii=False), flush=True)
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
