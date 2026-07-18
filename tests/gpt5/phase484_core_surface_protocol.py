#!/usr/bin/env python3
"""Phase484 core-surface and order-stress static protocol.

Freezes identity, light core-surface, and order-stress tracks before any model
run. Static only: no CUDA, no behavior gate, no physical geometry collection.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase484_core_surface_protocol"
PROTOCOL_PATH = OUT_DIR / "phase484_core_surface_protocol.json"
SAMPLES_PATH = OUT_DIR / "phase484_core_surface_samples.jsonl"
AUDIT_PATH = OUT_DIR / "phase484_core_surface_static_audit.json"
MANIFEST_PATH = OUT_DIR / "phase484_core_surface_manifest.json"

OPEN_SPLITS = ("geometry_window_freeze", "physical_prediction_holdout")
SEALED_SPLIT = "sealed_physical_holdout"
ALL_SPLITS = (*OPEN_SPLITS, SEALED_SPLIT)
PAIRS_PER_SPLIT = 96
SUBPROTOCOLS = ("label_post_relation_geometry", "label_pre_mapping_visible_control")
LABEL_MAPPINGS = {
    "mu_ab": {True: "A", False: "B", "instruction": "Map: true=A; false=B."},
    "mu_ba": {True: "B", False: "A", "instruction": "Map: true=B; false=A."},
}
VARIANT_TRACKS = (
    "identity",
    "core_surface_plain",
    "core_surface_light",
    "order_stress_claim_first",
)


def stable_hash(*parts: object, n: int = 16) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def code(prefix: str, split_index: int, pair_index: int, slot: int) -> str:
    return f"{prefix}{split_index:01d}{pair_index:03d}{slot:01d}"


def fact_bundle(split_index: int, pair_index: int) -> dict[str, Any]:
    entities = [code("e", split_index, pair_index, 0), code("e", split_index, pair_index, 1)]
    kinds = [code("k", split_index, pair_index, 0), code("k", split_index, pair_index, 1)]
    props = [code("p", split_index, pair_index, 0), code("p", split_index, pair_index, 1)]
    facts = [
        f"All {kinds[0]} have marker {props[0]}.",
        f"{entities[0]} is one {kinds[0]}.",
        f"All {kinds[1]} have marker {props[1]}.",
        f"{entities[1]} is one {kinds[1]}.",
    ]
    return {"entities": entities, "kinds": kinds, "props": props, "facts": facts}


def render(subprotocol: str, track: str, facts: list[str], claim: str, instruction: str) -> str:
    fact_text = " ".join(facts)
    if track in {"identity", "core_surface_plain"}:
        records = f"Records: {fact_text}"
        claim_text = f"Claim: {claim}"
    elif track == "core_surface_light":
        records = f"Records - {fact_text}"
        claim_text = f"Claim - {claim}"
    elif track == "order_stress_claim_first":
        records = f"Records: {fact_text}"
        claim_text = f"Claim: {claim}"
    else:
        raise ValueError(track)

    if subprotocol == "label_post_relation_geometry":
        if track == "order_stress_claim_first":
            return f"{claim_text} {records} {instruction}"
        return f"{records} {claim_text} {instruction}"
    if subprotocol == "label_pre_mapping_visible_control":
        if track == "order_stress_claim_first":
            return f"{instruction} {claim_text} {records}"
        return f"{instruction} {records} {claim_text}"
    raise ValueError(subprotocol)


def topology_signature(subprotocol: str, track: str) -> str:
    if subprotocol == "label_post_relation_geometry":
        return "records_before_claim_before_label" if track != "order_stress_claim_first" else "claim_before_records_before_label"
    if subprotocol == "label_pre_mapping_visible_control":
        return "label_before_records_before_claim" if track != "order_stress_claim_first" else "label_before_claim_before_records"
    raise ValueError(subprotocol)


def make_rows(split: str, split_index: int, pair_index: int) -> list[dict[str, Any]]:
    bundle = fact_bundle(split_index, pair_index)
    target_position = pair_index % 2
    target_entity = bundle["entities"][target_position]
    true_prop = bundle["props"][target_position]
    false_prop = bundle["props"][1 - target_position]
    source_pair_id = stable_hash("phase484", split, pair_index, "pair")
    rows = []
    for pair_role, truth_value in (("base_true", True), ("counterfactual_false", False)):
        query_prop = true_prop if truth_value else false_prop
        claim = f"{target_entity} has marker {query_prop}."
        for label_mapping, mapping in LABEL_MAPPINGS.items():
            expected = str(mapping[truth_value])
            for subprotocol in SUBPROTOCOLS:
                variants = []
                for track in VARIANT_TRACKS:
                    variant_class = "identity" if track == "identity" else ("order_stress" if track == "order_stress_claim_first" else "core_surface")
                    variants.append({
                        "track": track,
                        "variant_class": variant_class,
                        "topology_signature": topology_signature(subprotocol, track),
                        "subprotocol": subprotocol,
                        "text": render(subprotocol, track, bundle["facts"], claim, str(mapping["instruction"])),
                        "expected_label": expected,
                        "label_mapping": label_mapping,
                    })
                logic = {
                    "phase": "phase484",
                    "split": split,
                    "pair_index": pair_index,
                    "pair_role": pair_role,
                    "truth_value": truth_value,
                    "label_mapping": label_mapping,
                    "subprotocol": subprotocol,
                    "target_position": target_position,
                    "query_property": query_prop,
                }
                rows.append({
                    "sample_id": stable_hash("phase484", split, pair_index, pair_role, label_mapping, subprotocol),
                    "source_sample_id": stable_hash("phase484", split, pair_index, pair_role),
                    "source_pair_id": source_pair_id,
                    "pair_index": pair_index,
                    "pair_role": pair_role,
                    "split": split,
                    "sealed": split == SEALED_SPLIT,
                    "subprotocol": subprotocol,
                    "label_mapping": label_mapping,
                    "canonical_answer": expected,
                    "truth_value": truth_value,
                    "facts": bundle["facts"],
                    "claim": claim,
                    "role_nodes": {
                        "target_position": target_position,
                        "target_entity": target_entity,
                        "query_property": query_prop,
                    },
                    "logic_form": logic,
                    "semantic_hash": stable_hash(json.dumps(logic, sort_keys=True), n=20),
                    "surface_variants": variants,
                })
    return rows


def build_samples() -> list[dict[str, Any]]:
    rows = []
    for split_index, split in enumerate(ALL_SPLITS):
        for pair_index in range(PAIRS_PER_SPLIT):
            rows.extend(make_rows(split, split_index, pair_index))
    return rows


def variant_rows(samples: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [(sample, variant) for sample in samples for variant in sample["surface_variants"]]


def count_contract(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(samples)
    variant_count = sum(len(row["surface_variants"]) for row in samples)
    expected_samples = len(ALL_SPLITS) * PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS)
    expected_variants = expected_samples * len(VARIANT_TRACKS)
    return {
        "pass": sample_count == expected_samples and variant_count == expected_variants,
        "actual_sample_records": sample_count,
        "expected_sample_records": expected_samples,
        "actual_variant_records": variant_count,
        "expected_variant_records": expected_variants,
        "open_variant_records": len(OPEN_SPLITS) * PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS) * len(VARIANT_TRACKS),
        "sealed_variant_records": PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS) * len(VARIANT_TRACKS),
    }


def topology_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    for sample, variant in variant_rows(samples):
        sig = variant["topology_signature"]
        if variant["variant_class"] in {"identity", "core_surface"}:
            if sample["subprotocol"] == "label_post_relation_geometry" and sig != "records_before_claim_before_label":
                failures.append({"kind": "core_topology_changed", "sample_id": sample["sample_id"], "track": variant["track"]})
            if sample["subprotocol"] == "label_pre_mapping_visible_control" and sig != "label_before_records_before_claim":
                failures.append({"kind": "core_topology_changed", "sample_id": sample["sample_id"], "track": variant["track"]})
        if variant["variant_class"] == "order_stress" and "claim_before_records" not in sig:
            failures.append({"kind": "stress_not_order_changed", "sample_id": sample["sample_id"], "track": variant["track"]})
    return {"pass": not failures, "failure_count": len(failures), "failures": failures[:30]}


def balance_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "split": Counter(row["split"] for row in samples),
        "truth_by_split": Counter((row["split"], row["truth_value"]) for row in samples),
        "mapping_by_split": Counter((row["split"], row["label_mapping"]) for row in samples),
        "subprotocol_by_split": Counter((row["split"], row["subprotocol"]) for row in samples),
        "variant_class": Counter(variant["variant_class"] for _sample, variant in variant_rows(samples)),
        "track": Counter(variant["track"] for _sample, variant in variant_rows(samples)),
        "answer_by_mapping_truth": Counter((row["label_mapping"], row["truth_value"], row["canonical_answer"]) for row in samples),
    }
    pass_flag = (
        set(counters["split"].values()) == {PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS)}
        and set(counters["truth_by_split"].values()) == {PAIRS_PER_SPLIT * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS)}
        and set(counters["mapping_by_split"].values()) == {PAIRS_PER_SPLIT * 2 * len(SUBPROTOCOLS)}
        and set(counters["subprotocol_by_split"].values()) == {PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS)}
        and set(counters["track"].values()) == {len(samples)}
        and set(counters["answer_by_mapping_truth"].values()) == {PAIRS_PER_SPLIT * len(ALL_SPLITS) * len(SUBPROTOCOLS)}
    )
    return {"pass": pass_flag, **{key: {str(k): v for k, v in counter.items()} for key, counter in counters.items()}}


def baseline_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = variant_rows(samples)
    baselines = {
        "constant_A": lambda s, v: "A",
        "constant_B": lambda s, v: "B",
        "mu_ab": lambda s, v: "A" if s["label_mapping"] == "mu_ab" else "B",
        "truth_only_without_mapping": lambda s, v: "A" if s["truth_value"] else "B",
        "truth_plus_mapping_oracle": lambda s, v: s["canonical_answer"],
        "variant_core": lambda s, v: "A" if v["variant_class"] == "core_surface" else "B",
        "variant_stress": lambda s, v: "A" if v["variant_class"] == "order_stress" else "B",
    }
    reports = {}
    failing = []
    for name, fn in baselines.items():
        correct = sum(1 for sample, variant in rows if fn(sample, variant) == sample["canonical_answer"])
        acc = correct / len(rows)
        reports[name] = {"n": len(rows), "correct": correct, "accuracy": acc}
        if name != "truth_plus_mapping_oracle" and acc > 0.55:
            failing.append({"baseline": name, "accuracy": acc})
    if reports["truth_plus_mapping_oracle"]["accuracy"] != 1.0:
        failing.append({"baseline": "truth_plus_mapping_oracle", "accuracy": reports["truth_plus_mapping_oracle"]["accuracy"]})
    return {"pass": not failing, "reports": reports, "failing_baselines": failing}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    count = count_contract(samples)
    topology = topology_audit(samples)
    balance = balance_audit(samples)
    baseline = baseline_audit(samples)
    all_pass = count["pass"] and topology["pass"] and balance["pass"] and baseline["pass"]
    protocol = {
        "schema_version": "phase484_core_surface_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_protocol_frozen_no_model_run_no_cuda",
        "splits": {
            "open": list(OPEN_SPLITS),
            "sealed": SEALED_SPLIT,
            "phase484_may_read": list(OPEN_SPLITS),
            "phase484_must_not_read": [SEALED_SPLIT],
        },
        "pairs_per_split": PAIRS_PER_SPLIT,
        "subprotocols": list(SUBPROTOCOLS),
        "variant_tracks": list(VARIANT_TRACKS),
        "distance_definitions": {
            "d_identity": "identity vs identity repeated forward/noise track",
            "d_core": "core_surface_plain vs core_surface_light with topology preserved",
            "d_stress": "core_surface_plain vs order_stress_claim_first",
            "d_cf": "true vs false counterfactual with same track",
        },
        "quality_definitions": {
            "q_core": "(D_cf - D_core)/(D_cf + D_core + eps)",
            "q_stress": "(D_cf - D_stress)/(D_cf + D_stress + eps)",
            "t_topology": "(D_stress - D_core)/(D_stress + D_core + eps)",
        },
        "forbidden": ["sealed_split_read", "head_scan", "channel_scan", "neuron_scan", "post_result_surface_change"],
    }
    audit = {
        "schema_version": "phase484_core_surface_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if all_pass else "static_fail_no_model_run",
        "count_contract": count,
        "topology": topology,
        "balance": balance,
        "static_baselines": baseline,
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(SAMPLES_PATH, samples)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "phase484_core_surface_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(PROTOCOL_PATH.relative_to(ROOT)): sha256_file(PROTOCOL_PATH),
            str(SAMPLES_PATH.relative_to(ROOT)): sha256_file(SAMPLES_PATH),
            str(AUDIT_PATH.relative_to(ROOT)): sha256_file(AUDIT_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(PROTOCOL_PATH)
    print(SAMPLES_PATH)
    print(AUDIT_PATH)
    print("status", audit["status"])


if __name__ == "__main__":
    main()
