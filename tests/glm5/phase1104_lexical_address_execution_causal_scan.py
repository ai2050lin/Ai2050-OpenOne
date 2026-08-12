#!/usr/bin/env python3
"""Run Phase1104 query-end lexical routing interventions for one model."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1103_natural_relation_route_causal_scan as low
import phase1104_lexical_address_execution_protocol as protocol


CAPTURE_BATCH_SIZE = {"qwen3": 8, "glm4": 4, "deepseek7b": 4}
PATCH_BATCH_SIZE = {"qwen3": 8, "glm4": 4, "deepseek7b": 4}
EPSILON = 1e-8


def sampled_depths(layer_count: int) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for fraction in protocol.CAUSAL_DEPTH_FRACTIONS:
        depth = min(
            range(1, layer_count + 1),
            key=lambda value: (abs(value / layer_count - fraction), value),
        )
        if depth in seen:
            continue
        seen.add(depth)
        rows.append({
            "depth": depth,
            "relative_depth": depth / layer_count,
            "requested_fraction": fraction,
        })
    return rows


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["relation_pair"]), str(row["surface"]), str(row["split"]),
        int(row["template"]), int(row["item_index"]),
        str(row["label_regime"]), str(row["congruence"]),
        str(row["route_type"]), int(row["target_relation"]),
        int(row["relation_order"]), int(row["orientation"]),
    )


def superunit_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["surface"]), str(row["split"]), int(row["template"]),
        int(row["item_index"]), int(row["relation_order"]),
        int(row["orientation"]),
    )


def is_causal_item(row: dict[str, Any]) -> bool:
    allowed = (
        protocol.CAUSAL_DISCOVERY_ITEMS
        if row["split"] == "qualification"
        else protocol.CAUSAL_CONFIRMATION_ITEMS
    )
    return (
        int(row["item_index"]) in allowed
        and int(row["relation_order"]) in protocol.CAUSAL_RELATION_ORDERS
    )


def entry(
    *, pair: str, surface: str, split: str, template: int,
    item_index: int, orientation: int, depth: int,
    target_row: dict[str, Any], donor_row: dict[str, Any],
    source_pair: str, source_regime: str, target_regime: str,
    target_direction: str, patch_kind: str, delta: torch.Tensor,
    baseline_scores: dict[str, dict[str, float]], congruent: bool = False,
    delta_origin_regime: str | None = None,
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
        "source_regime": source_regime,
        "target_regime": target_regime,
        "delta_origin_regime": delta_origin_regime or source_regime,
        "target_direction": target_direction,
        "patch_kind": patch_kind,
        "delta": delta,
        "baseline_scores": baseline_scores,
        "congruent": congruent,
    }


def build_entries(
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
        pair: str, regime: str, congruence: str, route: str, target: int
    ) -> dict[str, Any]:
        return index[(
            pair, surface, split, template, item_index, regime,
            congruence, route, target, order, orientation,
        )]

    def delta(
        pair: str, regime: str, congruence: str, route: str
    ) -> torch.Tensor:
        q0 = get(pair, regime, congruence, route, 0)
        q1 = get(pair, regime, congruence, route, 1)
        return vectors[q1["record_id"]][depth] - vectors[q0["record_id"]][depth]

    rows = []
    regime_directions = (
        ("relation_label", "neutral_label"),
        ("neutral_label", "relation_label"),
    )
    target_directions = ((0, 1), (1, 0))
    for pair in eligible_pairs:
        wrong_pair = wrong_pairs[pair]
        for source_regime, target_regime in regime_directions:
            active = delta(pair, source_regime, "conflict", "exact")
            null = delta(pair, source_regime, "congruent", "exact")
            interaction = active - null
            ordinal = delta(pair, source_regime, "conflict", "ordinal")
            wrong = delta(wrong_pair, source_regime, "conflict", "exact")
            target_within = delta(pair, target_regime, "conflict", "exact")
            controls = {
                "selector_null_control": low.norm_match(null, interaction),
                "ordinal_control": low.norm_match(ordinal, interaction),
                "wrong_pair_control": low.norm_match(wrong, interaction),
                "equal_norm_random_control": low.equal_norm_random(
                    interaction, protocol.PHASE, model_name, pair, surface,
                    split, template, item_index, orientation, depth,
                    source_regime, target_regime,
                ),
            }
            for base_target, desired_target in target_directions:
                sign = 1.0 if (base_target, desired_target) == (0, 1) else -1.0
                target = get(
                    pair, target_regime, "conflict", "exact", base_target
                )
                donor = get(
                    pair, target_regime, "conflict", "exact", desired_target
                )
                direction_name = f"q{base_target}_to_q{desired_target}"
                common = dict(
                    pair=pair, surface=surface, split=split,
                    template=template, item_index=item_index,
                    orientation=orientation, depth=depth,
                    target_row=target, donor_row=donor,
                    source_regime=source_regime,
                    target_regime=target_regime,
                    target_direction=direction_name,
                    baseline_scores=baseline_scores,
                )
                rows.extend((
                    entry(
                        source_pair=pair, patch_kind="within_regime_raw",
                        delta=sign * target_within,
                        delta_origin_regime=target_regime,
                        **common,
                    ),
                    entry(
                        source_pair=pair, patch_kind="cross_regime_raw",
                        delta=sign * active, **common,
                    ),
                    entry(
                        source_pair=pair,
                        patch_kind="cross_regime_interaction",
                        delta=sign * interaction, **common,
                    ),
                ))
                for kind, vector in controls.items():
                    rows.append(entry(
                        source_pair=(wrong_pair if kind == "wrong_pair_control" else pair),
                        patch_kind=kind,
                        delta=sign * vector,
                        **common,
                    ))
                collateral_target = get(
                    pair, target_regime, "congruent", "exact", base_target
                )
                collateral_donor = get(
                    pair, target_regime, "congruent", "exact", desired_target
                )
                rows.append(entry(
                    pair=pair, surface=surface, split=split,
                    template=template, item_index=item_index,
                    orientation=orientation, depth=depth,
                    target_row=collateral_target, donor_row=collateral_donor,
                    source_pair=pair, source_regime=source_regime,
                    target_regime=target_regime,
                    target_direction=direction_name,
                    patch_kind="congruent_collateral_interaction",
                    delta=sign * interaction,
                    baseline_scores=baseline_scores,
                    congruent=True,
                ))
    return rows


def run_patch_batch(
    model, layer, entries: list[dict[str, Any]], pad_id: int, device
) -> list[dict[str, Any]]:
    rows = [row["target_row"] for row in entries]
    input_ids, attention_mask, lengths = low.pad_rows(rows, pad_id, device)
    positions = torch.tensor([
        int(row["role_positions"]["query_end"]) for row in rows
    ], dtype=torch.long, device=device)
    deltas = torch.stack([row["delta"] for row in entries])
    with torch.inference_mode():
        with low.ResidualPatch(
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
        for slot, item in enumerate(entries):
            target = item["target_row"]
            donor = item["donor_row"]
            patched = low.candidate_scores(
                logits, slot, int(lengths[slot].item()), target
            )
            target_base = item["baseline_scores"][target["record_id"]]
            donor_base = item["baseline_scores"][donor["record_id"]]
            finite = all(math.isfinite(value) for value in (
                *patched.values(), *target_base.values(), *donor_base.values(),
            )) and bool(torch.isfinite(item["delta"]).all().item())
            if item["congruent"]:
                expected = str(target["expected_class"])
                other = "e1" if expected == "e0" else "e0"
                base_margin = target_base[other] - target_base[expected]
                patched_margin = patched[other] - patched[expected]
                behavior_valid = (
                    finite and base_margin < 0.0
                    and donor["expected_class"] == expected
                )
                donor_margin = None
                denominator = None
                recovery = None
                flip = finite and patched_margin > 0.0
            else:
                base_class = str(target["expected_class"])
                desired_class = str(donor["expected_class"])
                base_margin = target_base[desired_class] - target_base[base_class]
                donor_margin = donor_base[desired_class] - donor_base[base_class]
                patched_margin = patched[desired_class] - patched[base_class]
                denominator = donor_margin - base_margin
                behavior_valid = (
                    finite and base_class != desired_class
                    and base_margin < 0.0 and donor_margin > 0.0
                    and denominator > EPSILON
                )
                recovery = (
                    (patched_margin - base_margin) / denominator
                    if behavior_valid else None
                )
                flip = finite and patched_margin > 0.0
            delta_norm = float(torch.linalg.vector_norm(item["delta"].float()).item())
            records.append({
                "schema_version": "phase1104_causal_patch_record.v1",
                "phase": protocol.PHASE,
                "model": target["model"],
                "relation_pair": item["pair"],
                "surface": item["surface"],
                "split": item["split"],
                "template": item["template"],
                "item_index": item["item_index"],
                "orientation": item["orientation"],
                "depth": item["depth"],
                "source_pair": item["source_pair"],
                "source_regime": item["source_regime"],
                "target_regime": item["target_regime"],
                "delta_origin_regime": item["delta_origin_regime"],
                "target_direction": item["target_direction"],
                "patch_kind": item["patch_kind"],
                "target_record_id": target["record_id"],
                "donor_record_id": donor["record_id"],
                "delta_norm": delta_norm if math.isfinite(delta_norm) else None,
                "base_margin": base_margin if math.isfinite(base_margin) else None,
                "donor_margin": (
                    donor_margin
                    if donor_margin is not None and math.isfinite(donor_margin)
                    else None
                ),
                "patched_margin": (
                    patched_margin if math.isfinite(patched_margin) else None
                ),
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
                "congruent": item["congruent"],
            })
        del output, logits, input_ids, attention_mask, lengths, positions, deltas
    return records


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1104 protocol audit failed")
    eligible_pairs = list(
        authorization["models"][model_name]["causal_selected_pairs"]
    )
    output_root = protocol.OUT_ROOT / "causal" / model_name
    if not eligible_pairs:
        summary = {
            "schema_version": "phase1104_causal_scan_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "authorization_digest": authorization["authorization_digest"],
            "skipped": True,
            "reason": "no model-specific behavior-authorized pair",
            "eligible_pairs": [],
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return
    wrong_pairs = {
        pair: authorization["wrong_pair_controls"][pair]
        for pair in eligible_pairs
    }
    source_pairs = sorted(set(eligible_pairs) | set(wrong_pairs.values()))
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    index = {row_key(row): row for row in rows if is_causal_item(row)}
    superunits = sorted({
        superunit_key(row) for row in rows if is_causal_item(row)
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
        capture = low.QueryResidualCapture(layers, depths)
        capture.register()
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        def get(
            pair: str, superunit: tuple[Any, ...], regime: str,
            congruence: str, route: str, target: int,
        ) -> dict[str, Any]:
            surface, split, template, item_index, order, orientation = superunit
            return index[(
                pair, surface, split, template, item_index, regime,
                congruence, route, target, order, orientation,
            )]

        for number, superunit in enumerate(superunits):
            capture_rows = []
            seen = set()
            for pair in source_pairs:
                for regime in protocol.LABEL_REGIMES:
                    for congruence in protocol.CONGRUENCES:
                        for route in protocol.ROUTE_TYPES:
                            for target in protocol.TARGET_RELATIONS:
                                row = get(
                                    pair, superunit, regime,
                                    congruence, route, target,
                                )
                                if row["record_id"] not in seen:
                                    capture_rows.append(row)
                                    seen.add(row["record_id"])
            vectors, baseline_scores = low.capture_baselines(
                model, capture, capture_rows,
                CAPTURE_BATCH_SIZE[model_name], int(pad_id), device,
            )
            for depth in depths:
                entries = build_entries(
                    eligible_pairs, wrong_pairs, index, vectors,
                    baseline_scores, superunit, depth, model_name,
                )
                for start in range(0, len(entries), PATCH_BATCH_SIZE[model_name]):
                    all_records.extend(run_patch_batch(
                        model, layers[depth - 1],
                        entries[start:start + PATCH_BATCH_SIZE[model_name]],
                        int(pad_id), device,
                    ))
            print(json.dumps({
                "phase": protocol.PHASE,
                "model": model_name,
                "causal_superunits_complete": number + 1,
                "causal_superunits_total": len(superunits),
                "patch_records": len(all_records),
            }), flush=True)
            del vectors, baseline_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        capture.close()
        capture = None
        elapsed = time.time() - started
        noncongruent = [row for row in all_records if not row["congruent"]]
        summary = {
            "schema_version": "phase1104_causal_scan_summary.v1",
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
            "finite_fraction": sum(row["finite"] for row in all_records) / max(len(all_records), 1),
            "behavior_valid_fraction": sum(row["behavior_valid"] for row in noncongruent) / max(len(noncongruent), 1),
            "elapsed_seconds": elapsed,
            "patch_alpha": protocol.PATCH_ALPHA,
            "component": "residual_stream_after_sampled_layer_at_query_end",
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
