#!/usr/bin/env python3
"""Scan only behavior-qualified Phase1020 pattern families and frozen items."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1018_language_pattern_scan as engine
import phase1020_language_operation_protocol as protocol


ACCURACY_GATE = 0.70
MINIMUM_MODELS = 2


def configure_engine(families: tuple[str, ...]) -> None:
    engine.CAPTURE_ROLES = protocol.CAPTURE_ROLES
    engine.FAMILIES = families
    engine.MODELS = protocol.MODELS
    engine.OUT_ROOT = protocol.OUT_ROOT
    engine.PHASE = protocol.PHASE
    engine.PROTOCOL_REVISION = protocol.PROTOCOL_REVISION
    engine.STATES = protocol.STATES
    engine.STATE_INDEX = {
        state: index for index, state in enumerate(protocol.STATES)
    }
    engine.ROLE_INDEX = {
        role: index for index, role in enumerate(protocol.CAPTURE_ROLES)
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "candidate_accuracy": (
            float(np.mean([row["candidate_hit"] for row in rows]))
            if rows else None
        ),
        "median_candidate_margin": (
            float(np.median([row["candidate_margin"] for row in rows]))
            if rows else None
        ),
    }


def build_scan_gate() -> dict[str, Any]:
    by_model = {}
    for model in protocol.MODELS:
        path = protocol.OUT_ROOT / "behavior" / model / "formal.jsonl"
        if not path.exists():
            raise RuntimeError(
                "All three behavior runs must finish before scanning: "
                f"missing {path}"
            )
        rows = protocol.read_jsonl(path)
        by_model[model] = {
            "family": {
                family: metrics([
                    row for row in rows if row["family"] == family
                ])
                for family in protocol.FAMILIES
            },
            "translation_profile": {
                profile: metrics([
                    row
                    for row in rows
                    if row["family"] == "translation_mode"
                    and row["subgroup"] == profile
                ])
                for profile in protocol.TRANSLATION_PROFILES
            },
        }

    eligible = []
    decisions = []
    for family in protocol.FAMILIES:
        if family == "translation_mode":
            passing = [
                model
                for model in protocol.MODELS
                if by_model[model]["translation_profile"]["full"][
                    "candidate_accuracy"
                ] >= ACCURACY_GATE
            ]
            reason = "full_profile_behavior"
        else:
            passing = [
                model
                for model in protocol.MODELS
                if by_model[model]["family"][family][
                    "candidate_accuracy"
                ] >= ACCURACY_GATE
            ]
            reason = "family_behavior"
        allowed = len(passing) >= MINIMUM_MODELS
        if allowed:
            eligible.append(family)
        decisions.append({
            "family": family,
            "gate_basis": reason,
            "passing_models": passing,
            "eligible_for_descriptive_scan": allowed,
        })
    result = {
        "schema_version": "phase1020_scan_gate.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "thresholds": {
            "candidate_accuracy": ACCURACY_GATE,
            "minimum_model_count": MINIMUM_MODELS,
        },
        "eligible_families": eligible,
        "decisions": decisions,
        "behavior": by_model,
        "claim_limit": (
            "This gate authorizes descriptive component scanning only; "
            "it does not authorize a causal or mechanistic claim."
        ),
    }
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "scan_gate.json",
        result,
    )
    return result


def run_model(model_name: str, *, resume: bool) -> dict[str, Any]:
    gate = build_scan_gate()
    families = tuple(gate["eligible_families"])
    configure_engine(families)
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    selection = protocol.read_json(
        protocol.OUT_ROOT / "behavior" / model_name / "selection.json"
    )
    if selection["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("behavior/protocol digest mismatch")
    selected_modes = selection["selected_by_family"]
    cases = []
    units = []
    for family in families:
        mode = selected_modes[family]
        family_cases = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "protocol"
            / f"cases.{model_name}.{mode}.jsonl"
        )
        family_units = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "protocol"
            / f"units.{model_name}.{mode}.jsonl"
        )
        eligible_unit_ids = {
            row["unit_id"]
            for row in family_units
            if row["family"] == family and row["scan_eligible"]
        }
        units.extend([
            row
            for row in family_units
            if row["unit_id"] in eligible_unit_ids
        ])
        cases.extend([
            row
            for row in family_cases
            if row["unit_id"] in eligible_unit_ids
        ])
    case_by_id = {row["record_id"]: row for row in cases}
    behavior = engine.behavior_by_unit(model_name)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for unit in units:
        grouped[(unit["family"], unit["item_id"], unit["split"])].append(
            unit
        )
    panel_items = sorted(grouped.items())

    output_root = protocol.OUT_ROOT / "formal_scan"
    model_root = output_root / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    if not panel_items:
        summary = {
            "schema_version": "phase1020_pattern_scan_model.v1",
            "phase": protocol.PHASE,
            "protocol_revision": protocol.PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "eligible_families": [],
            "panel_count": 0,
            "unit_count": 0,
            "state_case_count": 0,
            "decision": "no family passed the behavior scan gate",
        }
        protocol.write_json(model_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    model = tokenizer = device = None
    state_capture = head_capture = None
    summaries = []
    started = time.time()
    try:
        model, tokenizer, device, placement = engine.load_bf16(model_name)
        info = engine.get_model_info(model, model_name)
        layers = engine.get_layers(model)
        physical_heads = int(model.config.num_attention_heads)
        events, whole_keys, head_keys = engine.event_definitions(
            int(info.n_layers), physical_heads
        )
        protocol.write_jsonl(model_root / "events.jsonl", events)
        state_capture = engine.BatchRoleStateCapture(model, layers)
        head_capture = engine.BatchRoleHeadCapture(layers, physical_heads)
        state_capture.register()
        head_capture.register()
        for panel_index, (key, panel_units) in enumerate(panel_items, 1):
            family, item_id, split = key
            panel_root = model_root / family / item_id / split
            required = (
                panel_root / "summary.json",
                panel_root / "response_scalars.npz",
                panel_root / "direction_metrics.npz",
                panel_root / "directions.npz",
                panel_root / "units.jsonl",
            )
            if resume and all(path.exists() for path in required):
                existing = protocol.read_json(panel_root / "summary.json")
                if (
                    int(existing["protocol_revision"])
                    == protocol.PROTOCOL_REVISION
                    and int(existing["unit_count"]) == len(panel_units)
                ):
                    summaries.append(existing)
                    print(
                        f"[resume] {model_name}/{family}/{item_id}/{split}",
                        flush=True,
                    )
                    continue
            summary = engine.run_panel(
                model=model,
                device=device,
                tokenizer=tokenizer,
                model_name=model_name,
                prompt_mode=selected_modes[family],
                family=family,
                item_id=item_id,
                split=split,
                units=panel_units,
                case_by_id=case_by_id,
                behavior=behavior,
                events=events,
                whole_keys=whole_keys,
                head_keys=head_keys,
                state_capture=state_capture,
                head_capture=head_capture,
                output_root=output_root,
            )
            summaries.append(summary)
            print(
                f"[scan] {model_name} panel={panel_index}/"
                f"{len(panel_items)} {family}/{item_id}/{split}",
                flush=True,
            )
        model_summary = {
            "schema_version": "phase1020_pattern_scan_model.v1",
            "phase": protocol.PHASE,
            "protocol_revision": protocol.PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "selected_modes": selected_modes,
            "eligible_families": list(families),
            "precision": "bf16",
            "placement": placement,
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "head_count": physical_heads,
                "head_width": int(
                    layers[0].self_attn.o_proj.in_features
                    // physical_heads
                ),
            },
            "panel_count": len(summaries),
            "unit_count": int(sum(row["unit_count"] for row in summaries)),
            "batched_forward_count": int(sum(
                row["batched_forward_count"] for row in summaries
            )),
            "state_case_count": int(sum(
                row["state_case_count"] for row in summaries
            )),
            "identity_maximum": float(max(
                row["identity_maximum"] for row in summaries
            )),
            "prefix_branch_maximum": float(max(
                row["prefix_branch_maximum"] for row in summaries
            )),
            "by_family": {
                family: {
                    "panel_count": sum(
                        row["family"] == family for row in summaries
                    ),
                    "unit_count": sum(
                        row["unit_count"]
                        for row in summaries
                        if row["family"] == family
                    ),
                }
                for family in families
            },
            "elapsed_seconds": time.time() - started,
        }
        protocol.write_json(model_root / "summary.json", model_summary)
        print(json.dumps(model_summary, ensure_ascii=False, indent=2))
        return model_summary
    finally:
        if head_capture is not None:
            head_capture.close()
        if state_capture is not None:
            state_capture.close()
        if model is not None:
            engine.release_model(model)
        del model, tokenizer, device, state_capture, head_capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_model(args.model, resume=args.resume)


if __name__ == "__main__":
    main()
